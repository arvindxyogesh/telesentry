"""Fast, Isolation-Forest-only batch evaluation, kept around for quick
iteration while tuning the classical model/thresholds. For the full
cross-model research benchmark (Isolation Forest vs. LSTM autoencoder vs.
attention/Transformer detector, with per-anomaly-type breakdowns), see
evaluate_models.py.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.common.constants import FEATURE_COLUMNS
from src.ml.features import compute_row_features

__all__ = ["compute_row_features", "compute_detection_delay_seconds"]


def compute_detection_delay_seconds(df: pd.DataFrame) -> float:
    delays = []
    for _, group in df.groupby("vehicle_id"):
        group = group.sort_values("event_time").copy()
        group["segment"] = (group["label"] != group["label"].shift(1)).cumsum()
        for _, seg in group[group["label"] == 1].groupby("segment"):
            start_time = seg["event_time"].iloc[0]
            detected = seg[seg["pred"] == 1]
            if not detected.empty:
                delay = (detected["event_time"].iloc[0] - start_time).total_seconds()
                delays.append(max(0.0, delay))
    if not delays:
        return float("inf")
    return float(np.mean(delays))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast batch evaluation for the Isolation Forest telemetry detector.")
    parser.add_argument("--input", default="data/raw/synthetic_telemetry_eval.parquet")
    parser.add_argument("--model", default="models/isolation_forest.joblib")
    parser.add_argument("--scaler", default="models/feature_scaler.joblib")
    parser.add_argument("--metrics-out", default="data/processed/batch_metrics.json")
    parser.add_argument("--calibration", default="models/isolation_forest_calibration.json")
    parser.add_argument("--threshold", type=float, default=None, help="Overrides the calibrated threshold if set.")
    parser.add_argument("--z-baseline-threshold", type=float, default=3.0)
    parser.add_argument("--fusion-strategy", choices=["or", "and", "iforest"], default="iforest")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df = compute_row_features(df)

    scaler = joblib.load(args.scaler)
    model = joblib.load(args.model)

    if args.threshold is not None:
        threshold = args.threshold
    elif Path(args.calibration).exists():
        with open(args.calibration, "r", encoding="utf-8") as f:
            threshold = json.load(f)["threshold"]
    else:
        threshold = 0.6
    args.threshold = threshold

    X = scaler.transform(df[FEATURE_COLUMNS])
    raw_score = -model.decision_function(X)

    z_scores = np.abs((df["accel_spike_max"] - df["accel_spike_max"].mean()) / (df["accel_spike_max"].std() + 1e-8))
    z_baseline = (z_scores > args.z_baseline_threshold).astype(int)
    iforest_only = (raw_score > args.threshold).astype(int)

    df["pred_iforest"] = (raw_score > args.threshold).astype(int)
    if args.fusion_strategy == "or":
        df["pred"] = np.where((df["pred_iforest"] == 1) | (z_baseline == 1), 1, 0)
    elif args.fusion_strategy == "and":
        df["pred"] = np.where((df["pred_iforest"] == 1) & (z_baseline == 1), 1, 0)
    else:
        df["pred"] = df["pred_iforest"]

    precision, recall, f1, _ = precision_recall_fscore_support(df["label"], df["pred"], average="binary", zero_division=0)
    false_positive_rate = float(((df["pred"] == 1) & (df["label"] == 0)).sum() / max(1, (df["label"] == 0).sum()))
    delay = compute_detection_delay_seconds(df[["vehicle_id", "event_time", "label", "pred"]])
    baseline_fpr = float(((z_baseline == 1) & (df["label"] == 0)).sum() / max(1, (df["label"] == 0).sum()))
    iforest_fpr = float(((iforest_only == 1) & (df["label"] == 0)).sum() / max(1, (df["label"] == 0).sum()))
    fpr_reduction_vs_baseline = float((baseline_fpr - false_positive_rate) / max(baseline_fpr, 1e-8))

    metrics = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "samples": int(len(df)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": false_positive_rate,
        "zscore_baseline_false_positive_rate": baseline_fpr,
        "iforest_only_false_positive_rate": iforest_fpr,
        "fpr_reduction_vs_zscore_baseline": fpr_reduction_vs_baseline,
        "fusion_strategy": args.fusion_strategy,
        "mean_detection_delay_sec": delay,
        "target_delay_sec": 5.0,
        "delay_target_met": bool(delay < 5.0),
    }

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
