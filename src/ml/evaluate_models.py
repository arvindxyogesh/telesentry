"""Unified benchmark: z-score baseline, Isolation Forest, LSTM autoencoder,
and the attention/Transformer detector, evaluated on the same held-out
dataset with the same feature pipeline.

Reports, per model: precision/recall/F1, AUROC/AUPRC, false positive rate,
mean detection delay, per-anomaly-type recall, and batched inference
latency. This is the artifact the README's "Results" section and the
generated run report are built from.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score, roc_curve

from src.common.constants import ANOMALY_TYPES, FEATURE_COLUMNS, SEQUENCE_WINDOW
from src.ml.evaluate_batch import compute_detection_delay_seconds
from src.ml.features import compute_row_features
from src.ml.models.lstm_autoencoder import LSTMAutoencoder
from src.ml.models.transformer_detector import TransformerAnomalyDetector
from src.ml.sequence_dataset import build_windows

DEEP_SCORE_BATCH = 512


def _time_block():
    start = time.perf_counter()
    return lambda: time.perf_counter() - start


def _score_zscore_baseline(df: pd.DataFrame, threshold: float) -> dict:
    timer = _time_block()
    z = np.abs((df["accel_spike_max"] - df["accel_spike_max"].mean()) / (df["accel_spike_max"].std() + 1e-8))
    elapsed = timer()
    return {"score": z.to_numpy(), "pred": (z > threshold).astype(int).to_numpy(), "elapsed_sec": elapsed, "num_params": 0}


def _score_isolation_forest(df: pd.DataFrame, scaler, model, threshold: float) -> dict:
    timer = _time_block()
    X = scaler.transform(df[FEATURE_COLUMNS])
    raw_score = -model.decision_function(X)
    elapsed = timer()
    return {
        "score": raw_score,
        "pred": (raw_score > threshold).astype(int),
        "elapsed_sec": elapsed,
        "num_params": int(sum(t.tree_.node_count for t in model.estimators_)) if hasattr(model, "estimators_") else 0,
    }


def _full_length_scores(df: pd.DataFrame, windows, raw_scores: np.ndarray) -> np.ndarray:
    """Scatter window-level scores (indexed by their originating row) back
    onto the full row-aligned array; rows without enough trailing history yet
    (a vehicle's first SEQUENCE_WINDOW-1 samples) default to the minimum
    observed score, i.e. "insufficient history, assume normal".
    """
    fill_value = float(raw_scores.min()) - 1.0 if len(raw_scores) else 0.0
    full = np.full(len(df), fill_value, dtype=np.float64)
    positions = df.index.get_indexer(windows.row_index)
    full[positions] = raw_scores
    return full


def _score_lstm(df: pd.DataFrame, scaler, model: LSTMAutoencoder, calibration: dict) -> dict:
    df_scaled = df.copy()
    df_scaled[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])
    windows = build_windows(df_scaled, FEATURE_COLUMNS, window=SEQUENCE_WINDOW)

    model.eval()
    scores = []
    timer = _time_block()
    with torch.no_grad():
        for start in range(0, len(windows.X), DEEP_SCORE_BATCH):
            batch = torch.from_numpy(windows.X[start : start + DEEP_SCORE_BATCH])
            scores.append(model.reconstruction_error(batch).numpy())
    elapsed = timer()
    raw_scores = np.concatenate(scores) if scores else np.empty((0,))

    full_scores = _full_length_scores(df, windows, raw_scores)
    threshold = calibration["lstm_autoencoder"]["threshold"]
    return {
        "score": full_scores,
        "pred": (full_scores > threshold).astype(int),
        "elapsed_sec": elapsed,
        "num_params": int(sum(p.numel() for p in model.parameters())),
    }


def _score_transformer(df: pd.DataFrame, scaler, model: TransformerAnomalyDetector, calibration: dict) -> dict:
    df_scaled = df.copy()
    df_scaled[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])
    windows = build_windows(df_scaled, FEATURE_COLUMNS, window=SEQUENCE_WINDOW)

    cfg = calibration["transformer_detector"]
    model.eval()
    recon_scores, conc_scores = [], []
    timer = _time_block()
    with torch.no_grad():
        for start in range(0, len(windows.X), DEEP_SCORE_BATCH):
            batch = torch.from_numpy(windows.X[start : start + DEEP_SCORE_BATCH])
            recon_err, concentration = model.anomaly_components(batch)
            recon_scores.append(recon_err.numpy())
            conc_scores.append(concentration.numpy())
    elapsed = timer()

    recon_arr = np.concatenate(recon_scores) if recon_scores else np.empty((0,))
    conc_arr = np.concatenate(conc_scores) if conc_scores else np.empty((0,))
    combined = (recon_arr - cfg["recon_mean"]) / cfg["recon_std"] + cfg["attention_weight"] * (
        (conc_arr - cfg["concentration_mean"]) / cfg["concentration_std"]
    )

    full_scores = _full_length_scores(df, windows, combined)
    threshold = cfg["threshold"]
    return {
        "score": full_scores,
        "pred": (full_scores > threshold).astype(int),
        "elapsed_sec": elapsed,
        "num_params": int(sum(p.numel() for p in model.parameters())),
    }


def _metrics_for_model(df: pd.DataFrame, name: str, score: np.ndarray, pred: np.ndarray, elapsed_sec: float, num_params: int) -> dict:
    y = df["label"].to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    fpr = float(((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum()))

    try:
        auroc = float(roc_auc_score(y, score))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y, score))
    except ValueError:
        auprc = float("nan")

    delay_df = df[["vehicle_id", "event_time", "label"]].copy()
    delay_df["pred"] = pred
    mean_delay = compute_detection_delay_seconds(delay_df)

    per_type_recall = {}
    for anomaly_type in ANOMALY_TYPES[1:]:
        mask = df["anomaly_type"] == anomaly_type
        if mask.sum() == 0:
            per_type_recall[anomaly_type] = None
            continue
        per_type_recall[anomaly_type] = float(pred[mask.to_numpy()].mean())

    return {
        "model": name,
        "samples": int(len(df)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": auroc,
        "auprc": auprc,
        "false_positive_rate": fpr,
        "mean_detection_delay_sec": mean_delay,
        "target_delay_sec": 5.0,
        "delay_target_met": bool(mean_delay < 5.0),
        "per_anomaly_type_recall": per_type_recall,
        "num_parameters": num_params,
        "scoring_wall_time_sec": elapsed_sec,
        "latency_ms_per_1000_rows": float(elapsed_sec / max(1, len(df)) * 1000 * 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all anomaly detectors on a held-out dataset.")
    parser.add_argument("--input", default="data/raw/synthetic_telemetry_eval.parquet")
    parser.add_argument("--iforest-model", default="models/isolation_forest.joblib")
    parser.add_argument("--scaler", default="models/feature_scaler.joblib")
    parser.add_argument("--lstm-model", default="models/lstm_autoencoder.pt")
    parser.add_argument("--transformer-model", default="models/transformer_detector.pt")
    parser.add_argument("--calibration", default="models/deep_model_calibration.json")
    parser.add_argument("--iforest-calibration", default="models/isolation_forest_calibration.json")
    parser.add_argument("--iforest-threshold", type=float, default=None, help="Overrides the calibrated threshold if set.")
    parser.add_argument("--zscore-threshold", type=float, default=3.0)
    parser.add_argument("--output-json", default="report/model_comparison.json")
    parser.add_argument("--output-csv", default="report/model_comparison.csv")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df = compute_row_features(df)
    df = df.reset_index(drop=True)

    scaler = joblib.load(args.scaler)
    iforest_model = joblib.load(args.iforest_model)
    with open(args.calibration, "r", encoding="utf-8") as f:
        calibration = json.load(f)

    if args.iforest_threshold is not None:
        iforest_threshold = args.iforest_threshold
    else:
        with open(args.iforest_calibration, "r", encoding="utf-8") as f:
            iforest_threshold = json.load(f)["threshold"]

    lstm_cfg = calibration["lstm_autoencoder"]
    lstm_model = LSTMAutoencoder(num_features=len(FEATURE_COLUMNS), hidden_size=lstm_cfg["hidden_size"])
    lstm_model.load_state_dict(torch.load(args.lstm_model, map_location="cpu"))

    tf_cfg = calibration["transformer_detector"]
    transformer_model = TransformerAnomalyDetector(
        num_features=len(FEATURE_COLUMNS),
        window=calibration["sequence_window"],
        d_model=tf_cfg["d_model"],
        nhead=tf_cfg["nhead"],
        num_layers=tf_cfg["num_layers"],
    )
    transformer_model.load_state_dict(torch.load(args.transformer_model, map_location="cpu"))

    results = {}
    roc_curves = {}

    for name, scored in [
        ("zscore_baseline", _score_zscore_baseline(df, args.zscore_threshold)),
        ("isolation_forest", _score_isolation_forest(df, scaler, iforest_model, iforest_threshold)),
        ("lstm_autoencoder", _score_lstm(df, scaler, lstm_model, calibration)),
        ("transformer_detector", _score_transformer(df, scaler, transformer_model, calibration)),
    ]:
        metrics = _metrics_for_model(df, name, scored["score"], scored["pred"], scored["elapsed_sec"], scored["num_params"])
        results[name] = metrics
        fpr_curve, tpr_curve, _ = roc_curve(df["label"].to_numpy(), scored["score"])
        # Interpolate onto a fixed FPR grid: keeps the saved JSON small
        # (raw curves can have tens of thousands of threshold points) and
        # makes the multi-model overlay directly comparable point-for-point.
        fpr_grid = np.linspace(0.0, 1.0, 101)
        tpr_grid = np.interp(fpr_grid, fpr_curve, tpr_curve)
        roc_curves[name] = {"fpr": fpr_grid.round(4).tolist(), "tpr": tpr_grid.round(4).tolist()}
        print(f"[{name}] precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
              f"auroc={metrics['auroc']:.3f} fpr={metrics['false_positive_rate']:.4f} "
              f"delay={metrics['mean_detection_delay_sec']:.2f}s")

    best_f1_model = max(results, key=lambda k: results[k]["f1"])

    summary = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "samples": int(len(df)),
        "anomaly_type_distribution": df["anomaly_type"].value_counts(normalize=True).round(4).to_dict(),
        "models": results,
        "roc_curves": roc_curves,
        "best_f1_model": best_f1_model,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    table_rows = []
    for name, metrics in results.items():
        row = {k: v for k, v in metrics.items() if k != "per_anomaly_type_recall"}
        for anomaly_type, recall in metrics["per_anomaly_type_recall"].items():
            row[f"recall_{anomaly_type}"] = recall
        table_rows.append(row)
    pd.DataFrame(table_rows).to_csv(args.output_csv, index=False)

    print(f"\nBest F1: {best_f1_model}")
    print(f"Saved comparison JSON: {args.output_json}")
    print(f"Saved comparison CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
