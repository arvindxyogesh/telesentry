#!/usr/bin/env bash
set -euo pipefail

spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
  src/streaming/feature_job.py \
  --bootstrap-servers localhost:9092 \
  --topic telemetry.raw \
  --checkpoint checkpoints/telemetry_features \
  --output data/processed/anomaly_alerts \
  --model models/isolation_forest.joblib \
  --scaler models/feature_scaler.joblib
