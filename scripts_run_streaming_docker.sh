#!/usr/bin/env bash
set -euo pipefail

docker exec telemetry-spark-master spark-submit \
  --master spark://spark-master:7077 \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
  /workspace/src/streaming/feature_job.py \
  --bootstrap-servers kafka:9094 \
  --topic telemetry.raw \
  --checkpoint /workspace/checkpoints/telemetry_features \
  --output /workspace/data/processed/anomaly_alerts
