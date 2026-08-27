from pyspark.sql.types import DoubleType, StringType, StructField, StructType, TimestampType

from src.common.constants import ANOMALY_TYPES, FEATURE_COLUMNS, ROW_WINDOW_SAMPLES, SEQUENCE_WINDOW


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

__all__ = [
    "TELEMETRY_SCHEMA",
    "ROW_WINDOW_SAMPLES",
    "SEQUENCE_WINDOW",
    "FEATURE_COLUMNS",
    "ANOMALY_TYPES",
]
