# Real-Time Vehicle Telemetry Anomaly Detection

Production-style data + ML pipeline for vehicle telemetry anomaly detection with Kafka, Spark Structured Streaming, and online alerting.

## What This Includes

- Real-time telemetry simulation (CAN/IMU/GPS-like streams)
- Kafka ingestion topic: `telemetry.raw`
- Spark Structured Streaming with sliding windows (10s window, 5s slide)
- Feature engineering in stream:
  - speed variance
  - acceleration spike magnitude
  - heading change range
  - mean yaw rate
- Anomaly detection:
  - Isolation Forest (primary)
  - z-score baseline (secondary guardrail)
- Alert outputs:
  - console alerts in streaming job
  - parquet table at `data/processed/anomaly_alerts`
  - REST API for latest alerts
  - Streamlit dashboard
- Batch evaluation with detection-delay and false-positive metrics

## Architecture

1. `src/simulator/can_imu_gps_simulator.py`
- Simulates vehicle telemetry and pushes JSON events to Kafka.

2. `src/streaming/feature_job.py`
- Spark job reads Kafka stream, computes sliding-window features, scores anomalies, writes alerts.

3. `src/ml/train_isolation_forest.py`
- Trains model on normal telemetry (from synthetic dataset).

4. `src/ml/evaluate_batch.py`
- Batch evaluation with:
  - precision / recall / f1
  - false positive rate
  - mean detection delay (target < 5 sec)

5. `src/alert_api/main.py`
- FastAPI endpoints for health, latest alerts, and metrics.

6. `src/dashboard/app.py`
- Streamlit dashboard for anomaly timeline and delay/score monitoring.

## Quick Start (Local)

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Start Kafka + Spark

```bash
docker compose up -d
```

### 3) Generate synthetic training/evaluation data

```bash
python src/ml/generate_synthetic_dataset.py --output data/raw/synthetic_telemetry.parquet
```

### 4) Train model

```bash
python src/ml/train_isolation_forest.py \
  --input data/raw/synthetic_telemetry.parquet \
  --model-out models/isolation_forest.joblib \
  --scaler-out models/feature_scaler.joblib
```

### 5) Start streaming detector

```bash
bash scripts_run_streaming.sh
```

### 6) Start telemetry simulator

```bash
python src/simulator/can_imu_gps_simulator.py \
  --bootstrap-servers localhost:9092 \
  --topic telemetry.raw
```

### 7) Run batch evaluation

```bash
python src/ml/evaluate_batch.py \
  --input data/raw/synthetic_telemetry.parquet \
  --model models/isolation_forest.joblib \
  --scaler models/feature_scaler.joblib \
  --metrics-out data/processed/batch_metrics.json
```

### 8) Start alert API + dashboard

```bash
uvicorn src.alert_api.main:app --host 0.0.0.0 --port 8000 --reload
streamlit run src/dashboard/app.py --server.port 8501
```

## API

- `GET /health`
- `GET /alerts/latest?limit=50`
- `GET /metrics`

## Output Tables

- Alerts parquet: `data/processed/anomaly_alerts`
- Batch metrics JSON: `data/processed/batch_metrics.json`

## Evaluation Metrics

- Mean detection delay seconds (`target < 5 sec`)
- False positive rate
- Precision / recall / F1

## Dataset Options

Recommended external datasets for stronger realism:

1. UAH-DriveSet
- Driver behavior + vehicle dynamics useful for anomaly signatures.

2. Ford / autonomous driving telemetry collections
- GPS, IMU, and motion features for robust behavior modeling.

3. Synthetic generator (included)
- Immediate end-to-end pipeline validation and repeatable experiments.

## Notes for Productionization

- Replace synthetic source with CAN bus connector (MQTT/Kafka Connect/custom edge collector).
- Store alerts/features in Delta Lake instead of parquet for ACID and schema evolution.
- Add model registry + scheduled retraining.
- Add alert routing to PagerDuty/Slack/webhooks.
- Add drift monitoring and adaptive thresholding.

## Copyright

Copyright (c) 2025-2026 Arvind Yogesh. All rights reserved.

See the COPYRIGHT file for additional terms.
