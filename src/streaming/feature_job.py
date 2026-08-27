"""Spark structured streaming job: Kafka -> sliding-window feature
aggregation -> trained Isolation Forest scoring -> parquet alert sink.

The streaming aggregation currently computes 5 of the 7 offline features
(speed_var, accel_spike_max, heading_range, yaw_rate_mean, sample_count).
The other 2 (gps_speed_delta_mean, speed_trend; see src/ml/features.py) need
a per-vehicle ordered lag over lat/lon/event_time, which in Structured
Streaming requires stateful processing (flatMapGroupsWithState) rather than
a plain windowed aggregate -- out of scope here. Missing feature values are
imputed with the scaler's training-time mean (a neutral value that
contributes ~0 after standardization) rather than left as an unscored
placeholder, so the trained model is still genuinely applied online instead
of being replaced by static thresholds. See the README "Limitations"
section.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.common.constants import FEATURE_COLUMNS

SPARK_AVAILABLE_FEATURES = ["speed_var", "accel_spike_max", "heading_range", "yaw_rate_mean", "sample_count"]


def build_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def _score_batch_with_model(pdf: pd.DataFrame, model, scaler, threshold: float) -> pd.DataFrame:
    """Apply the trained, calibrated Isolation Forest to one micro-batch of
    window-aggregated features (called from foreachBatch, so this is plain
    single-node pandas/sklearn -- no Spark UDF serialization concerns).
    """
    X = pd.DataFrame(index=pdf.index)
    for i, col in enumerate(FEATURE_COLUMNS):
        if col in SPARK_AVAILABLE_FEATURES:
            X[col] = pdf[col]
        else:
            X[col] = scaler.mean_[i]  # neutral imputation, see module docstring

    scaled = scaler.transform(X[FEATURE_COLUMNS])
    raw_score = -model.decision_function(scaled)
    pdf = pdf.copy()
    pdf["anomaly_score"] = raw_score
    pdf["iforest_anomaly"] = (raw_score > threshold).astype(int)
    return pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Spark structured streaming feature + anomaly scoring job.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="telemetry.raw")
    parser.add_argument("--checkpoint", default="checkpoints/telemetry_features")
    parser.add_argument("--output", default="data/processed/anomaly_alerts")
    parser.add_argument("--iforest-model", default="models/isolation_forest.joblib")
    parser.add_argument("--scaler", default="models/feature_scaler.joblib")
    parser.add_argument("--iforest-calibration", default="models/isolation_forest_calibration.json")
    parser.add_argument("--speed-var-threshold", type=float, default=50.0)
    parser.add_argument("--accel-spike-threshold", type=float, default=3.5)
    parser.add_argument("--heading-range-threshold", type=float, default=25.0)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint).mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.iforest_model)
    scaler = joblib.load(args.scaler)
    import json

    with open(args.iforest_calibration, "r", encoding="utf-8") as f:
        iforest_threshold = json.load(f)["threshold"]

    spark = build_spark("vehicle-telemetry-anomaly-stream")
    spark.sparkContext.setLogLevel("WARN")

    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap_servers)
        .option("subscribe", args.topic)
        .option("startingOffsets", "latest")
        .load()
    )

    telemetry = (
        raw.selectExpr("CAST(value AS STRING) AS payload")
        .select(F.from_json(F.col("payload"), "event_time STRING, vehicle_id STRING, speed_kph DOUBLE, accel_mps2 DOUBLE, yaw_rate_dps DOUBLE, heading_deg DOUBLE, lat DOUBLE, lon DOUBLE, source STRING").alias("j"))
        .select("j.*")
        .withColumn("event_time", F.to_timestamp("event_time"))
        .filter(F.col("event_time").isNotNull())
    )

    features = (
        telemetry.withWatermark("event_time", "30 seconds")
        .groupBy(F.window("event_time", "10 seconds", "5 seconds"), F.col("vehicle_id"))
        .agg(
            F.variance("speed_kph").alias("speed_var"),
            F.max(F.abs(F.col("accel_mps2"))).alias("accel_spike_max"),
            (F.max("heading_deg") - F.min("heading_deg")).alias("heading_range"),
            F.avg("yaw_rate_dps").alias("yaw_rate_mean"),
            F.count("*").alias("sample_count"),
        )
        .select(
            F.col("vehicle_id"),
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            F.coalesce(F.col("speed_var"), F.lit(0.0)).alias("speed_var"),
            F.coalesce(F.col("accel_spike_max"), F.lit(0.0)).alias("accel_spike_max"),
            F.coalesce(F.col("heading_range"), F.lit(0.0)).alias("heading_range"),
            F.coalesce(F.col("yaw_rate_mean"), F.lit(0.0)).alias("yaw_rate_mean"),
            F.coalesce(F.col("sample_count"), F.lit(0)).cast("double").alias("sample_count"),
        )
    )

    def foreach_batch(batch_df, batch_id):
        if batch_df.rdd.isEmpty():
            return

        pdf = batch_df.toPandas()
        scored = _score_batch_with_model(pdf, model, scaler, iforest_threshold)

        rule_based = (
            (scored["speed_var"] > args.speed_var_threshold)
            | (scored["accel_spike_max"] > args.accel_spike_threshold)
            | (scored["heading_range"] > args.heading_range_threshold)
        )
        scored["is_anomaly"] = (rule_based | (scored["iforest_anomaly"] == 1)).astype(int)
        alerts = scored[scored["is_anomaly"] == 1].copy()

        if alerts.empty:
            return

        alerts["processing_time"] = pd.Timestamp.now(tz="UTC")
        alerts["detection_delay_sec"] = (
            alerts["processing_time"] - alerts["window_end"].dt.tz_localize("UTC")
        ).dt.total_seconds()
        alerts["model_backend"] = "iforest"

        print(f"[batch={batch_id}] anomalies={len(alerts)}")
        print(
            alerts[["vehicle_id", "window_end", "speed_var", "accel_spike_max", "anomaly_score", "detection_delay_sec"]]
            .head(20)
            .to_string(index=False)
        )

        spark.createDataFrame(alerts).write.mode("append").parquet(args.output)

    query = (
        features.writeStream.outputMode("update")
        .foreachBatch(foreach_batch)
        .option("checkpointLocation", args.checkpoint)
        .trigger(processingTime="5 seconds")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
