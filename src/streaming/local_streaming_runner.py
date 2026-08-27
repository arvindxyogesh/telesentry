"""Dependency-light replay-based stream processor (a Spark structured
streaming fallback for local development / demos without a JVM). Feeds a
parquet telemetry file through the same row-level feature pipeline used
offline (src.ml.features.compute_row_features), so there is no
training/serving feature skew, and scores each event with a selectable
model backend: the original rule-based thresholds, the calibrated Isolation
Forest, or either trained deep sequence model.

Sequence backends (LSTM autoencoder, Transformer detector) accumulate one
scaled feature vector per raw sample -- exactly the per-row cadence used to
build training windows in sequence_dataset.py -- so a SEQUENCE_WINDOW-sample
warm-up is required per vehicle before they can score at all.
"""

import argparse
import json
import time
from collections import defaultdict, deque
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.common.constants import FEATURE_COLUMNS, ROW_WINDOW_SAMPLES, SEQUENCE_WINDOW
from src.ml.features import compute_row_features
from src.ml.models.lstm_autoencoder import LSTMAutoencoder
from src.ml.models.transformer_detector import TransformerAnomalyDetector


class ThresholdBackend:
    """The original rule-based detector: no trained model required."""

    is_sequence_backend = False

    def __init__(self, speed_var_threshold: float, accel_spike_threshold: float, heading_range_threshold: float):
        self.speed_var_threshold = speed_var_threshold
        self.accel_spike_threshold = accel_spike_threshold
        self.heading_range_threshold = heading_range_threshold

    def score(self, feature_row: dict):
        ratios = [
            feature_row["speed_var"] / self.speed_var_threshold,
            feature_row["accel_spike_max"] / self.accel_spike_threshold,
            feature_row["heading_range"] / self.heading_range_threshold,
        ]
        score = max(ratios)
        return float(score), bool(score > 1.0)


class IsolationForestBackend:
    is_sequence_backend = False

    def __init__(self, model_path: str, scaler_path: str, calibration_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        with open(calibration_path, "r", encoding="utf-8") as f:
            self.threshold = json.load(f)["threshold"]

    def score(self, feature_row: dict):
        X = self.scaler.transform(pd.DataFrame([feature_row])[FEATURE_COLUMNS])
        raw = float(-self.model.decision_function(X)[0])
        return raw, bool(raw > self.threshold)


class _SequenceBackend:
    is_sequence_backend = True

    def __init__(self, scaler_path: str):
        self.scaler = joblib.load(scaler_path)

    def scale(self, feature_row: dict) -> np.ndarray:
        return self.scaler.transform(pd.DataFrame([feature_row])[FEATURE_COLUMNS])[0].astype(np.float32)


class LSTMBackend(_SequenceBackend):
    def __init__(self, model_path: str, scaler_path: str, calibration_path: str):
        super().__init__(scaler_path)
        with open(calibration_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)["lstm_autoencoder"]
        self.threshold = cfg["threshold"]
        self.model = LSTMAutoencoder(num_features=len(FEATURE_COLUMNS), hidden_size=cfg["hidden_size"])
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def score_window(self, window: np.ndarray):
        x = torch.from_numpy(window[None, :, :])
        with torch.no_grad():
            err = float(self.model.reconstruction_error(x).item())
        return err, bool(err > self.threshold)


class TransformerBackend(_SequenceBackend):
    def __init__(self, model_path: str, scaler_path: str, calibration_path: str):
        super().__init__(scaler_path)
        with open(calibration_path, "r", encoding="utf-8") as f:
            calibration = json.load(f)
        self.cfg = calibration["transformer_detector"]
        self.model = TransformerAnomalyDetector(
            num_features=len(FEATURE_COLUMNS),
            window=calibration["sequence_window"],
            d_model=self.cfg["d_model"],
            nhead=self.cfg["nhead"],
            num_layers=self.cfg["num_layers"],
        )
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def score_window(self, window: np.ndarray):
        x = torch.from_numpy(window[None, :, :])
        with torch.no_grad():
            recon_err, concentration = self.model.anomaly_components(x)
        combined = (recon_err.item() - self.cfg["recon_mean"]) / self.cfg["recon_std"] + self.cfg["attention_weight"] * (
            (concentration.item() - self.cfg["concentration_mean"]) / self.cfg["concentration_std"]
        )
        return float(combined), bool(combined > self.cfg["threshold"])


def build_backend(args):
    if args.model_backend == "threshold":
        return ThresholdBackend(args.speed_var_threshold, args.accel_spike_threshold, args.heading_range_threshold)
    if args.model_backend == "iforest":
        return IsolationForestBackend(args.iforest_model, args.scaler, args.iforest_calibration)
    if args.model_backend == "lstm":
        return LSTMBackend(args.lstm_model, args.scaler, args.calibration)
    if args.model_backend == "transformer":
        return TransformerBackend(args.transformer_model, args.scaler, args.calibration)
    raise ValueError(f"Unknown model backend: {args.model_backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local real-time anomaly stream runner (Spark fallback).")
    parser.add_argument("--input", default="data/raw/synthetic_telemetry.parquet")
    parser.add_argument("--output", default="data/processed/anomaly_alerts/alerts.parquet")
    parser.add_argument(
        "--model-backend", choices=["threshold", "iforest", "lstm", "transformer"], default="iforest"
    )
    parser.add_argument("--slide-sec", type=int, default=5, help="How often (in event time) to emit alert checks.")
    parser.add_argument("--speed-var-threshold", type=float, default=50.0)
    parser.add_argument("--accel-spike-threshold", type=float, default=3.5)
    parser.add_argument("--heading-range-threshold", type=float, default=25.0)
    parser.add_argument("--iforest-model", default="models/isolation_forest.joblib")
    parser.add_argument("--iforest-calibration", default="models/isolation_forest_calibration.json")
    parser.add_argument("--scaler", default="models/feature_scaler.joblib")
    parser.add_argument("--lstm-model", default="models/lstm_autoencoder.pt")
    parser.add_argument("--transformer-model", default="models/transformer_detector.pt")
    parser.add_argument("--calibration", default="models/deep_model_calibration.json")
    args = parser.parse_args()

    backend = build_backend(args)

    df = pd.read_parquet(args.input).sort_values("event_time")
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)

    raw_windows = defaultdict(lambda: deque(maxlen=ROW_WINDOW_SAMPLES))
    seq_windows = defaultdict(lambda: deque(maxlen=SEQUENCE_WINDOW))
    next_emit = {}
    alerts = []

    for _, row in df.iterrows():
        vid = row["vehicle_id"]
        ts = row["event_time"]

        raw_windows[vid].append(row)
        wdf = pd.DataFrame(list(raw_windows[vid]))
        feat_row = compute_row_features(wdf).iloc[-1][FEATURE_COLUMNS].to_dict()

        if backend.is_sequence_backend:
            seq_windows[vid].append(backend.scale(feat_row))

        if vid not in next_emit:
            next_emit[vid] = ts

        if ts >= next_emit[vid]:
            start = time.perf_counter()
            if backend.is_sequence_backend:
                if len(seq_windows[vid]) == SEQUENCE_WINDOW:
                    score, is_anomaly = backend.score_window(np.stack(seq_windows[vid]))
                else:
                    score, is_anomaly = 0.0, False  # warming up
            else:
                score, is_anomaly = backend.score(feat_row)
            inference_latency_ms = (time.perf_counter() - start) * 1000.0

            if is_anomaly:
                alerts.append(
                    {
                        "vehicle_id": vid,
                        "window_end": ts,
                        **{k: float(feat_row[k]) for k in FEATURE_COLUMNS},
                        "anomaly_score": score,
                        "model_backend": args.model_backend,
                        "is_anomaly": 1,
                        "processing_time": pd.Timestamp.now(tz="UTC"),
                        "inference_latency_ms": inference_latency_ms,
                    }
                )

            next_emit[vid] = ts + timedelta(seconds=args.slide_sec)

    out = pd.DataFrame(alerts)
    out_path = Path(args.output)
    if out_path.suffix == "":
        out_path = out_path / "alerts.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"Wrote {len(out)} alerts to {out_path} (backend={args.model_backend})")
    if not out.empty:
        print(out[["vehicle_id", "window_end", "anomaly_score", "inference_latency_ms"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
