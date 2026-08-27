import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.common.constants import FEATURE_COLUMNS
from src.ml.features import compute_row_features

# Same target quantile used to calibrate the deep models' thresholds
# (train_deep_models.py), so all methods are held to a comparable ~3% false
# positive rate on held-out normal data rather than an arbitrary cutoff.
CALIBRATION_QUANTILE = 0.97

__all__ = ["compute_row_features"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Isolation Forest for telemetry anomalies.")
    parser.add_argument("--input", default="data/raw/synthetic_telemetry.parquet")
    parser.add_argument("--model-out", default="models/isolation_forest.joblib")
    parser.add_argument("--scaler-out", default="models/feature_scaler.joblib")
    parser.add_argument("--calibration-out", default="models/isolation_forest_calibration.json")
    parser.add_argument("--contamination", type=float, default=0.03)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df = compute_row_features(df)

    normal_df = df[df.get("label", 0) == 0].copy()
    rng = np.random.default_rng(args.seed)
    shuffled = normal_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_val = max(1, int(len(shuffled) * args.val_fraction))
    val_df, train_df = shuffled.iloc[:n_val], shuffled.iloc[n_val:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS])

    model = IsolationForest(
        n_estimators=250,
        contamination=args.contamination,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_train)

    X_val = scaler.transform(val_df[FEATURE_COLUMNS])
    val_scores = -model.decision_function(X_val)
    threshold = float(np.quantile(val_scores, CALIBRATION_QUANTILE))

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    joblib.dump(scaler, args.scaler_out)
    with open(args.calibration_out, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "contamination": args.contamination, "calibration_quantile": CALIBRATION_QUANTILE}, f, indent=2)

    print(f"Trained model on {len(train_df)} normal rows (held out {len(val_df)} for threshold calibration)")
    print(f"Calibrated threshold (q{CALIBRATION_QUANTILE}): {threshold:.4f}")
    print(f"Saved model: {args.model_out}")
    print(f"Saved scaler: {args.scaler_out}")
    print(f"Saved calibration: {args.calibration_out}")


if __name__ == "__main__":
    main()
