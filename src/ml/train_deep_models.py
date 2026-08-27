"""Train the two deep sequence anomaly detectors (LSTM autoencoder,
attention/Transformer detector) on windows drawn exclusively from normal
telemetry -- the same unsupervised setup used for the Isolation Forest
baseline, so the three models are trained under comparable conditions and
can be benchmarked fairly in evaluate_models.py.
"""

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.common.constants import FEATURE_COLUMNS, SEQUENCE_WINDOW
from src.ml.features import compute_row_features
from src.ml.models.lstm_autoencoder import LSTMAutoencoder
from src.ml.models.transformer_detector import TransformerAnomalyDetector
from src.ml.sequence_dataset import build_windows

# Fraction of normal windows held back to calibrate each model's score
# distribution (threshold, z-score normalization) instead of the same
# windows used to fit its weights.
CALIBRATION_QUANTILE = 0.97


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_or_fit_scaler(df: pd.DataFrame, scaler_path: str) -> StandardScaler:
    path = Path(scaler_path)
    if path.exists():
        return joblib.load(path)
    scaler = StandardScaler()
    scaler.fit(df.loc[df["label"] == 0, FEATURE_COLUMNS])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    return scaler


def train_reconstruction_model(model, train_loader, epochs, lr, device, returns_tuple=False):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            if returns_tuple:
                recon, _ = model(xb)
            else:
                recon = model(xb)
            loss = nn.functional.mse_loss(recon, xb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)
            n_seen += len(xb)
        print(f"  epoch {epoch}/{epochs} train_recon_mse={running_loss / n_seen:.5f}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deep sequence anomaly detectors.")
    parser.add_argument("--input", default="data/raw/synthetic_telemetry.parquet")
    parser.add_argument("--scaler", default="models/feature_scaler.joblib")
    parser.add_argument("--lstm-out", default="models/lstm_autoencoder.pt")
    parser.add_argument("--transformer-out", default="models/transformer_detector.pt")
    parser.add_argument("--calibration-out", default="models/deep_model_calibration.json")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lstm-hidden-size", type=int, default=64)
    parser.add_argument("--transformer-d-model", type=int, default=32)
    parser.add_argument("--transformer-nhead", type=int, default=4)
    parser.add_argument("--transformer-num-layers", type=int, default=2)
    parser.add_argument("--attention-weight", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    df = pd.read_parquet(args.input)
    df = compute_row_features(df)
    scaler = load_or_fit_scaler(df, args.scaler)

    df_scaled = df.copy()
    df_scaled[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])

    windows = build_windows(df_scaled, FEATURE_COLUMNS, window=SEQUENCE_WINDOW)
    X_clean = windows.X[windows.clean_mask()]
    print(f"Clean (fully normal) windows available for training: {len(X_clean)} / {len(windows.X)} total windows")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X_clean))
    X_clean = X_clean[perm]
    n_val = max(1, int(len(X_clean) * args.val_fraction))
    X_val, X_train = X_clean[:n_val], X_clean[n_val:]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=args.batch_size, shuffle=True)
    val_tensor = torch.from_numpy(X_val).to(device)

    print(f"Training windows: {len(X_train)}, calibration/validation windows: {len(X_val)}")

    # --- LSTM autoencoder -------------------------------------------------
    print("Training LSTM autoencoder...")
    lstm = LSTMAutoencoder(num_features=len(FEATURE_COLUMNS), hidden_size=args.lstm_hidden_size)
    lstm = train_reconstruction_model(lstm, train_loader, args.epochs, args.lr, device)
    lstm.eval()
    with torch.no_grad():
        lstm_val_scores = lstm.reconstruction_error(val_tensor).cpu().numpy()

    Path(args.lstm_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(lstm.state_dict(), args.lstm_out)

    # --- Attention / Transformer detector ----------------------------------
    print("Training attention/Transformer detector...")
    transformer = TransformerAnomalyDetector(
        num_features=len(FEATURE_COLUMNS),
        window=SEQUENCE_WINDOW,
        d_model=args.transformer_d_model,
        nhead=args.transformer_nhead,
        num_layers=args.transformer_num_layers,
    )
    transformer = train_reconstruction_model(transformer, train_loader, args.epochs, args.lr, device, returns_tuple=True)
    transformer.eval()
    with torch.no_grad():
        tf_recon_err, tf_concentration = transformer.anomaly_components(val_tensor)
    tf_recon_err = tf_recon_err.cpu().numpy()
    tf_concentration = tf_concentration.cpu().numpy()

    recon_mean, recon_std = float(tf_recon_err.mean()), float(tf_recon_err.std() + 1e-8)
    conc_mean, conc_std = float(tf_concentration.mean()), float(tf_concentration.std() + 1e-8)
    tf_val_scores = (tf_recon_err - recon_mean) / recon_std + args.attention_weight * (
        (tf_concentration - conc_mean) / conc_std
    )

    torch.save(transformer.state_dict(), args.transformer_out)

    calibration = {
        "sequence_window": SEQUENCE_WINDOW,
        "feature_columns": FEATURE_COLUMNS,
        "calibration_quantile": CALIBRATION_QUANTILE,
        "lstm_autoencoder": {
            "hidden_size": args.lstm_hidden_size,
            "num_layers": 1,
            "threshold": float(np.quantile(lstm_val_scores, CALIBRATION_QUANTILE)),
        },
        "transformer_detector": {
            "d_model": args.transformer_d_model,
            "nhead": args.transformer_nhead,
            "num_layers": args.transformer_num_layers,
            "attention_weight": args.attention_weight,
            "recon_mean": recon_mean,
            "recon_std": recon_std,
            "concentration_mean": conc_mean,
            "concentration_std": conc_std,
            "threshold": float(np.quantile(tf_val_scores, CALIBRATION_QUANTILE)),
        },
    }
    Path(args.calibration_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.calibration_out, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    print(f"Saved LSTM autoencoder: {args.lstm_out}")
    print(f"Saved Transformer detector: {args.transformer_out}")
    print(f"Saved calibration: {args.calibration_out}")


if __name__ == "__main__":
    main()
