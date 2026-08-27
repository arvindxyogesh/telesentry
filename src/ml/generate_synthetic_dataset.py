"""Synthetic multivariate vehicle telemetry generator with a labeled anomaly
taxonomy.

Rather than a single generic "spike", each anomaly type models a distinct,
realistic failure mode with its own multivariate signature, loosely informed
by the automotive fault/attack literature (multi-sensor consistency checks
for GPS spoofing, stuck/frozen sensor channels, gradual calibration drift,
and communication dropout / packet loss):

- harsh_maneuver: a real physical event (evasive steering + hard braking or
  acceleration). Shows up as a simultaneous spike across speed variance,
  acceleration, yaw rate and heading range.
- sensor_stuck: the heading/yaw channel freezes while the vehicle keeps
  turning in reality. Shows up as *implausibly low* heading variance -- a
  one-sided variance threshold cannot catch this, only a learned model of the
  full normal distribution can.
- sensor_drift: the reported speed channel accumulates a slow additive bias
  (degraded sensor calibration). Shows up as a growing trend in the reported
  speed and a growing GPS-vs-reported-speed inconsistency, without any
  sudden spike.
- gps_spoof: the reported GPS position is pinned to a fixed false location
  while the vehicle keeps moving. Shows up as a large GPS-vs-reported-speed
  inconsistency with no accelerometer/gyro signature at all.
- dropout_burst: the telemetry link drops most packets for a few seconds.
  Shows up as reduced sample density in the trailing window, with every
  individual reported value otherwise normal.
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.common.constants import ANOMALY_TYPES

ANOMALY_DURATION_SEC = {
    "harsh_maneuver": (1.0, 3.0),
    "sensor_stuck": (2.0, 6.0),
    "sensor_drift": (3.0, 8.0),
    "gps_spoof": (2.0, 6.0),
    "dropout_burst": (2.0, 5.0),
}
# Anomaly types whose reported channel "snaps back" to the true value the
# instant the episode ends (a frozen sensor unfreezing, a spoofed GPS fix
# reverting, a drifted bias resetting) produce one genuinely anomalous
# transient sample right at the boundary. That sample stays labeled as the
# same anomaly type instead of leaking a one-off artifact into the "normal"
# class statistics.
RECOVERY_LABEL_TICKS = {
    "harsh_maneuver": 0,
    "sensor_stuck": 1,
    "sensor_drift": 1,
    "gps_spoof": 1,
    "dropout_burst": 0,
}
DROPOUT_KEEP_EVERY = 4
METERS_PER_DEG_LAT = 111_111.0


@dataclass
class GeneratorConfig:
    vehicles: int = 20
    seconds: int = 3600
    hz: int = 5
    anomaly_probability: float = 0.02
    seed: int = 42


@dataclass
class _Episode:
    anomaly_type: str
    remaining: int
    total: int
    distort_ticks: int
    tick: int = 0
    freeze_heading: Optional[float] = None
    drift_target_kph: float = 0.0
    spoof_lat: Optional[float] = None
    spoof_lon: Optional[float] = None


def _new_episode(anomaly_type: str, hz: int, rng: np.random.Generator) -> _Episode:
    lo, hi = ANOMALY_DURATION_SEC[anomaly_type]
    distort_samples = max(1, int(round(rng.uniform(lo, hi) * hz)))
    total_samples = distort_samples + RECOVERY_LABEL_TICKS[anomaly_type]
    return _Episode(anomaly_type=anomaly_type, remaining=total_samples, total=total_samples, distort_ticks=distort_samples)


def _simulate_vehicle(config: GeneratorConfig, vehicle_id: str, rng: np.random.Generator) -> list:
    dt = 1.0 / config.hz
    speed = float(rng.uniform(20.0, 80.0))
    heading = float(rng.uniform(0.0, 360.0))
    lat, lon = 37.3 + rng.normal(0.0, 0.01), -121.9 + rng.normal(0.0, 0.01)
    start = datetime.now(timezone.utc) - timedelta(seconds=config.seconds)

    episode: Optional[_Episode] = None
    rows = []

    for t in range(config.seconds * config.hz):
        ts = start + timedelta(seconds=t * dt)

        if episode is None and rng.random() < config.anomaly_probability:
            anomaly_type = rng.choice(ANOMALY_TYPES[1:])
            episode = _new_episode(anomaly_type, config.hz, rng)
            if episode.anomaly_type == "sensor_stuck":
                episode.freeze_heading = heading
            elif episode.anomaly_type == "sensor_drift":
                episode.drift_target_kph = float(rng.uniform(4.0, 9.0) * rng.choice([-1.0, 1.0]))
            elif episode.anomaly_type == "gps_spoof":
                bearing = rng.uniform(0.0, 2 * np.pi)
                offset_m = rng.uniform(300.0, 3000.0)
                episode.spoof_lat = lat + (offset_m * np.cos(bearing)) / METERS_PER_DEG_LAT
                episode.spoof_lon = lon + (offset_m * np.sin(bearing)) / (
                    METERS_PER_DEG_LAT * np.cos(np.deg2rad(max(lat, 1e-3)))
                )

        anomaly_type = episode.anomaly_type if episode else "normal"

        # True physical kinematics: only a real physical event (harsh
        # maneuver) perturbs the underlying motion. Sensor-level anomalies
        # (stuck/drift/spoof/dropout) distort only what gets *reported*.
        base_accel = rng.normal(0.0, 0.35)
        base_yaw = rng.normal(0.0, 1.5)
        if anomaly_type == "harsh_maneuver":
            base_accel += rng.choice([-4.5, 4.5]) + rng.normal(0.0, 1.2)
            base_yaw += rng.choice([-30.0, 30.0]) + rng.normal(0.0, 5.0)

        speed = float(np.clip(speed + base_accel * dt, 0.0, 150.0))
        heading = float((heading + base_yaw * dt) % 360.0)

        meters = speed * 1000.0 / 3600.0 * dt
        lat += (meters / METERS_PER_DEG_LAT) * np.cos(np.deg2rad(heading))
        lon += (meters / METERS_PER_DEG_LAT) * np.sin(np.deg2rad(heading)) / np.cos(np.deg2rad(max(lat, 1e-3)))

        reported_speed, reported_heading = speed, heading
        reported_accel, reported_yaw_rate = base_accel, base_yaw
        reported_lat, reported_lon = lat, lon
        emit_row = True

        if episode is not None:
            distorting = episode.tick < episode.distort_ticks

            if anomaly_type == "sensor_stuck" and distorting:
                reported_heading = episode.freeze_heading
                reported_yaw_rate = float(rng.normal(0.0, 0.1))
            elif anomaly_type == "sensor_drift" and distorting:
                progress = episode.tick / max(1, episode.distort_ticks)
                reported_speed = float(max(0.0, speed + episode.drift_target_kph * progress))
            elif anomaly_type == "gps_spoof" and distorting:
                reported_lat, reported_lon = episode.spoof_lat, episode.spoof_lon
            elif anomaly_type == "dropout_burst":
                emit_row = episode.tick % DROPOUT_KEEP_EVERY == 0

            episode.tick += 1
            episode.remaining -= 1
            if episode.remaining <= 0:
                episode = None

        if emit_row:
            rows.append(
                {
                    "event_time": ts,
                    "vehicle_id": vehicle_id,
                    "speed_kph": reported_speed,
                    "accel_mps2": reported_accel,
                    "yaw_rate_dps": reported_yaw_rate,
                    "heading_deg": reported_heading,
                    "lat": reported_lat,
                    "lon": reported_lon,
                    "source": "synthetic",
                    "label": int(anomaly_type != "normal"),
                    "anomaly_type": anomaly_type,
                }
            )

    return rows


def generate_dataframe(config: GeneratorConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    rows = []
    for vehicle_idx in range(config.vehicles):
        vehicle_id = f"veh_{vehicle_idx:03d}"
        rows.extend(_simulate_vehicle(config, vehicle_id, rng))

    df = pd.DataFrame(rows).sort_values(["vehicle_id", "event_time"]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic vehicle telemetry with a labeled anomaly taxonomy.")
    parser.add_argument("--output", default="data/raw/synthetic_telemetry.parquet")
    parser.add_argument("--vehicles", type=int, default=20)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--hz", type=int, default=5)
    parser.add_argument("--anomaly-probability", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = GeneratorConfig(
        vehicles=args.vehicles,
        seconds=args.seconds,
        hz=args.hz,
        anomaly_probability=args.anomaly_probability,
        seed=args.seed,
    )
    df = generate_dataframe(cfg)
    df.to_parquet(args.output, index=False)

    print(f"Wrote {len(df)} telemetry rows to {args.output}")
    print("Anomaly type distribution:")
    print(df["anomaly_type"].value_counts(normalize=True).rename("fraction"))


if __name__ == "__main__":
    main()
