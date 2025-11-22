import argparse
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def build_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Spark structured streaming feature + anomaly scoring job.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="telemetry.raw")
    parser.add_argument("--checkpoint", default="checkpoints/telemetry_features")
    parser.add_argument("--output", default="data/processed/anomaly_alerts")
    parser.add_argument("--speed-var-threshold", type=float, default=50.0)
    parser.add_argument("--accel-spike-threshold", type=float, default=3.5)
    parser.add_argument("--heading-range-threshold", type=float, default=25.0)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint).mkdir(parents=True, exist_ok=True)

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

        alerts = (
            batch_df.withColumn(
                "is_anomaly",
                F.when(
                    (F.col("speed_var") > F.lit(args.speed_var_threshold))
                    | (F.col("accel_spike_max") > F.lit(args.accel_spike_threshold))
                    | (F.col("heading_range") > F.lit(args.heading_range_threshold)),
                    F.lit(1),
                ).otherwise(F.lit(0)),
            )
            .withColumn("iforest_score", F.lit(0.0))
            .withColumn("processing_time", F.current_timestamp())
            .withColumn("detection_delay_sec", F.unix_timestamp("processing_time") - F.unix_timestamp("window_end"))
            .filter(F.col("is_anomaly") == 1)
        )

        if alerts.rdd.isEmpty():
            return

        count = alerts.count()
        print(f"[batch={batch_id}] anomalies={count}")
        alerts.select("vehicle_id", "window_end", "speed_var", "accel_spike_max", "heading_range", "detection_delay_sec").show(20, truncate=False)
        alerts.write.mode("append").parquet(args.output)

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
