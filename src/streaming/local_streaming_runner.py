import argparse
from collections import deque, defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def compute_features(window_df: pd.DataFrame) -> dict:
    return {
        "speed_var": float(window_df["speed_kph"].var(ddof=0) if len(window_df) > 1 else 0.0),
        "accel_spike_max": float(window_df["accel_mps2"].abs().max() if not window_df.empty else 0.0),
        "heading_range": float(window_df["heading_deg"].max() - window_df["heading_deg"].min() if len(window_df) > 1 else 0.0),
        "yaw_rate_mean": float(window_df["yaw_rate_dps"].mean() if not window_df.empty else 0.0),
        "sample_count": float(len(window_df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local real-time anomaly stream runner (Spark fallback).")
    parser.add_argument("--input", default="data/raw/synthetic_telemetry.parquet")
    parser.add_argument("--output", default="data/processed/anomaly_alerts/alerts.parquet")
    parser.add_argument("--window-sec", type=int, default=10)
    parser.add_argument("--slide-sec", type=int, default=5)
    parser.add_argument("--speed-var-threshold", type=float, default=50.0)
    parser.add_argument("--accel-spike-threshold", type=float, default=3.5)
    parser.add_argument("--heading-range-threshold", type=float, default=25.0)
    args = parser.parse_args()

    df = pd.read_parquet(args.input).sort_values("event_time")
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)

    per_vehicle = defaultdict(deque)
    next_emit = {}
    alerts = []

    for _, row in df.iterrows():
        vid = row["vehicle_id"]
        ts = row["event_time"]

        if vid not in next_emit:
            next_emit[vid] = ts

        dq = per_vehicle[vid]
        dq.append(row)

        cutoff = ts - timedelta(seconds=args.window_sec)
        while dq and pd.to_datetime(dq[0]["event_time"], utc=True) < cutoff:
            dq.popleft()

        if ts >= next_emit[vid]:
            wdf = pd.DataFrame(list(dq))
            feats = compute_features(wdf)

            is_anomaly = int(
                (feats["speed_var"] > args.speed_var_threshold)
                or (feats["accel_spike_max"] > args.accel_spike_threshold)
                or (feats["heading_range"] > args.heading_range_threshold)
            )

            if is_anomaly:
                processing_time = ts + timedelta(seconds=1)
                detection_delay = max(0.0, (processing_time - ts).total_seconds())
                alerts.append(
                    {
                        "vehicle_id": vid,
                        "window_start": ts - timedelta(seconds=args.window_sec),
                        "window_end": ts,
                        **feats,
                        "iforest_score": 0.0,
                        "is_anomaly": 1,
                        "processing_time": processing_time,
                        "detection_delay_sec": detection_delay,
                    }
                )

            next_emit[vid] = ts + timedelta(seconds=args.slide_sec)

    out = pd.DataFrame(alerts)
    out_path = Path(args.output)
    if out_path.suffix == "":
        out_path = out_path / "alerts.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"Wrote {len(out)} alerts to {out_path}")
    if not out.empty:
        print(out[["vehicle_id", "window_end", "detection_delay_sec", "accel_spike_max"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
