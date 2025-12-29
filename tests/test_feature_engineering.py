import pandas as pd

from src.ml.train_isolation_forest import compute_row_features


def test_feature_columns_created():
    df = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=20, freq="s"),
            "vehicle_id": ["veh_001"] * 20,
            "speed_kph": [30 + i * 0.5 for i in range(20)],
            "accel_mps2": [0.1] * 20,
            "yaw_rate_dps": [0.2] * 20,
            "heading_deg": [10 + i for i in range(20)],
        }
    )
    out = compute_row_features(df)

    assert "speed_var" in out.columns
    assert "accel_spike_max" in out.columns
    assert "heading_range" in out.columns
    assert "yaw_rate_mean" in out.columns
    assert "sample_count" in out.columns
    assert out["sample_count"].iloc[-1] == 20
