# Row-level rolling feature window (in samples) used for the tabular/classical
# feature set consumed by Isolation Forest and the z-score baseline.
ROW_WINDOW_SAMPLES = 10

# Number of consecutive feature-rows fed to the sequence models (LSTM
# autoencoder, attention/Transformer detector). At the default 5 Hz simulator
# rate this is a 3 second lookback, matching the <5s detection-delay target.
SEQUENCE_WINDOW = 15

FEATURE_COLUMNS = [
    "speed_var",
    "accel_spike_max",
    "heading_range",
    "yaw_rate_mean",
    "sample_count",
    "gps_speed_delta_mean",
    "speed_trend",
]

# Anomaly taxonomy used by the synthetic generator and by evaluation to break
# metrics down per failure mode. "normal" carries label=0; every other class
# carries label=1 in the binary target used for training/evaluation.
ANOMALY_TYPES = [
    "normal",
    "harsh_maneuver",
    "sensor_stuck",
    "sensor_drift",
    "gps_spoof",
    "dropout_burst",
]
