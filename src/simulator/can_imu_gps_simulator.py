import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
from kafka import KafkaProducer


def make_producer(bootstrap_servers: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda x: json.dumps(x).encode("utf-8"),
        linger_ms=10,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate CAN/IMU/GPS telemetry to Kafka.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="telemetry.raw")
    parser.add_argument("--vehicles", type=int, default=12)
    parser.add_argument("--hz", type=int, default=5)
    parser.add_argument("--anomaly-probability", type=float, default=0.03)
    args = parser.parse_args()

    producer = make_producer(args.bootstrap_servers)
    rng = np.random.default_rng(7)

    state = {}
    for i in range(args.vehicles):
        state[f"veh_{i:03d}"] = {
            "speed": float(rng.uniform(25.0, 85.0)),
            "heading": float(rng.uniform(0.0, 360.0)),
            "lat": 37.35 + rng.normal(0, 0.02),
            "lon": -121.95 + rng.normal(0, 0.02),
            "anomaly_remaining": 0,
        }

    dt = 1.0 / args.hz
    print(f"Producing telemetry to topic={args.topic} at {args.hz} Hz across {args.vehicles} vehicles")

    while True:
        tick_start = time.time()
        now = datetime.now(timezone.utc).isoformat()

        for vehicle_id, st in state.items():
            if st["anomaly_remaining"] <= 0 and rng.random() < args.anomaly_probability:
                st["anomaly_remaining"] = int(rng.integers(args.hz, args.hz * 4))

            is_anomaly = st["anomaly_remaining"] > 0
            accel = rng.normal(0.0, 0.35)
            yaw_rate = rng.normal(0.0, 1.4)
            if is_anomaly:
                accel += rng.choice([-5.0, 5.0]) + rng.normal(0.0, 1.0)
                yaw_rate += rng.choice([-25.0, 25.0]) + rng.normal(0.0, 4.0)

            st["speed"] = float(np.clip(st["speed"] + accel * 0.2, 0.0, 150.0))
            st["heading"] = float((st["heading"] + yaw_rate * 0.2) % 360.0)

            meters = st["speed"] * 1000.0 / 3600.0 * dt
            st["lat"] += (meters / 111111.0) * np.cos(np.deg2rad(st["heading"]))
            st["lon"] += (meters / 111111.0) * np.sin(np.deg2rad(st["heading"])) / np.cos(np.deg2rad(max(st["lat"], 1e-3)))

            payload = {
                "event_time": now,
                "vehicle_id": vehicle_id,
                "speed_kph": st["speed"],
                "accel_mps2": accel,
                "yaw_rate_dps": yaw_rate,
                "heading_deg": st["heading"],
                "lat": st["lat"],
                "lon": st["lon"],
                "source": "simulator",
            }
            producer.send(args.topic, value=payload)

            st["anomaly_remaining"] -= 1

        producer.flush()
        elapsed = time.time() - tick_start
        time.sleep(max(0.0, dt - elapsed))


if __name__ == "__main__":
    main()
