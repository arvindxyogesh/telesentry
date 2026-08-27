import numpy as np
import pandas as pd

from src.common.constants import FEATURE_COLUMNS
from src.ml.features import compute_row_features, haversine_km


def _base_df(n=20):
    return pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=n, freq="200ms", tz="UTC"),
            "vehicle_id": ["veh_001"] * n,
            "speed_kph": [30 + i * 0.5 for i in range(n)],
            "accel_mps2": [0.1] * n,
            "yaw_rate_dps": [0.2] * n,
            "heading_deg": [10 + i for i in range(n)],
            "lat": [37.3 + i * 1e-5 for i in range(n)],
            "lon": [-121.9 + i * 1e-5 for i in range(n)],
        }
    )


def test_feature_columns_created():
    out = compute_row_features(_base_df(20))
    for col in FEATURE_COLUMNS:
        assert col in out.columns
    assert out["sample_count"].iloc[-1] == 10  # windowed, not cumulative


def test_sample_count_is_windowed_not_cumulative():
    out = compute_row_features(_base_df(30))
    # After warm-up the rolling window count should plateau at the window
    # size (10), not keep growing with the row index.
    assert out["sample_count"].iloc[-1] == 10
    assert out["sample_count"].iloc[-1] == out["sample_count"].iloc[15]


def test_heading_range_handles_wraparound():
    # Headings crossing the 0/360 boundary should show a *small* circular
    # range, not the ~358 degree naive max-min artifact.
    n = 12
    df = _base_df(n)
    df["heading_deg"] = [358.0, 359.0, 0.5, 1.0, 2.0, 1.5, 359.5, 0.0, 1.0, 2.0, 1.0, 0.5]
    out = compute_row_features(df)
    assert out["heading_range"].iloc[-1] < 10.0


def test_gps_speed_delta_flags_position_speed_mismatch():
    n = 15
    df = _base_df(n)
    # Reported speed says the vehicle is moving fast, but GPS position is
    # frozen -- the classic spoofing/inconsistency signature.
    df["speed_kph"] = 80.0
    df["lat"] = 37.3
    df["lon"] = -121.9
    out = compute_row_features(df)
    assert out["gps_speed_delta_mean"].iloc[-1] > 50.0


def test_gps_speed_delta_near_zero_for_consistent_motion():
    # Build lat/lon that are physically consistent with the reported speed
    # (constant 40 kph, heading due north), rather than an arbitrary walk.
    n = 15
    df = _base_df(n)
    df["speed_kph"] = 40.0
    dt_sec = 0.2
    meters_per_step = 40.0 * 1000.0 / 3600.0 * dt_sec
    deg_per_step = meters_per_step / 111_111.0
    df["lat"] = [37.3 + i * deg_per_step for i in range(n)]
    df["lon"] = -121.9

    out = compute_row_features(df)
    assert out["gps_speed_delta_mean"].iloc[-1] < 5.0


def test_single_row_window_does_not_crash():
    out = compute_row_features(_base_df(1))
    assert len(out) == 1
    assert not out[FEATURE_COLUMNS].isna().any().any()


def test_haversine_zero_distance_for_same_point():
    d = haversine_km(np.array([37.3]), np.array([-121.9]), np.array([37.3]), np.array([-121.9]))
    assert d[0] == 0.0
