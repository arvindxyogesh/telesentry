import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


@dataclass
class GeneratorConfig:
    vehicles: int = 20
    seconds: int = 3600
    hz: int = 5
    anomaly_probability: float = 0.02
    seed: int = 42


def _step_vehicle_state(prev_speed: float, prev_heading: float, is_anomaly: bool, rng: np.random.Generator):
    base_accel = rng.normal(0.0, 0.35)
    base_yaw = rng.normal(0.0, 1.5)

    if is_anomaly:
        base_accel += rng.choice([-4.5, 4.5]) + rng.normal(0.0, 1.2)
        base_yaw += rng.choice([-30.0, 30.0]) + rng.normal(0.0, 5.0)

    speed = np.clip(prev_speed + base_accel * 0.2, 0.0, 150.0)
    heading = (prev_heading + base_yaw * 0.2) % 360.0
    return speed, base_accel, base_yaw, heading


def generate_dataframe(config: GeneratorConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    start = datetime.now(timezone.utc) - timedelta(seconds=config.seconds)
    dt = 1.0 / config.hz
    rows = []

    for vehicle_idx in range(config.vehicles):
        vehicle_id = f"veh_{vehicle_idx:03d}"
        speed = float(rng.uniform(20.0, 80.0))
        heading = float(rng.uniform(0.0, 360.0))
        lat, lon = 37.3 + rng.normal(0.0, 0.01), -121.9 + rng.normal(0.0, 0.01)

        anomaly_remaining = 0
        for t in range(config.seconds * config.hz):
            ts = start + timedelta(seconds=t * dt)
            if anomaly_remaining <= 0 and rng.random() < config.anomaly_probability:
                anomaly_remaining = int(rng.integers(config.hz, config.hz * 4))
            is_anomaly = anomaly_remaining > 0

            speed, accel, yaw_rate, heading = _step_vehicle_state(speed, heading, is_anomaly, rng)

            meters = speed * 1000.0 / 3600.0 * dt
            lat += (meters / 111111.0) * np.cos(np.deg2rad(heading))
            lon += (meters / 111111.0) * np.sin(np.deg2rad(heading)) / np.cos(np.deg2rad(max(lat, 1e-3)))

            rows.append(
                {
                    "event_time": ts,
                    "vehicle_id": vehicle_id,
                    "speed_kph": speed,
                    "accel_mps2": accel,
                    "yaw_rate_dps": yaw_rate,
                    "heading_deg": heading,
                    "lat": lat,
                    "lon": lon,
                    "source": "synthetic",
                    "label": int(is_anomaly),
                }
            )
            anomaly_remaining -= 1

    df = pd.DataFrame(rows).sort_values(["vehicle_id", "event_time"]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic vehicle telemetry with anomalies.")
    parser.add_argument("--output", default="data/raw/synthetic_telemetry.parquet")
    parser.add_argument("--vehicles", type=int, default=20)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--hz", type=int, default=5)
    parser.add_argument("--anomaly-probability", type=float, default=0.02)
    args = parser.parse_args()

    cfg = GeneratorConfig(
        vehicles=args.vehicles,
        seconds=args.seconds,
        hz=args.hz,
        anomaly_probability=args.anomaly_probability,
    )
    df = generate_dataframe(cfg)
    df.to_parquet(args.output, index=False)

    print(f"Wrote {len(df)} telemetry rows to {args.output}")
    print("Label distribution:")
    print(df["label"].value_counts(normalize=True).rename("fraction"))


if __name__ == "__main__":
    main()
