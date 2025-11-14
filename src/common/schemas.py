from pyspark.sql.types import DoubleType, StringType, StructField, StructType, TimestampType


TELEMETRY_SCHEMA = StructType(
    [
        StructField("event_time", TimestampType(), False),
        StructField("vehicle_id", StringType(), False),
        StructField("speed_kph", DoubleType(), False),
        StructField("accel_mps2", DoubleType(), False),
        StructField("yaw_rate_dps", DoubleType(), False),
        StructField("heading_deg", DoubleType(), False),
        StructField("lat", DoubleType(), False),
        StructField("lon", DoubleType(), False),
        StructField("source", StringType(), False),
    ]
)


FEATURE_COLUMNS = [
    "speed_var",
    "accel_spike_max",
    "heading_range",
    "yaw_rate_mean",
    "sample_count",
]
