"""Shared row-level feature engineering for the vehicle telemetry anomaly
detectors.

The same rolling-window feature set is used online (local streaming runner,
Spark structured streaming job) and offline (training / batch evaluation), so
it lives in one place rather than being duplicated per entry point.
"""

import numpy as np
import pandas as pd

from src.common.constants import FEATURE_COLUMNS, ROW_WINDOW_SAMPLES

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in km between paired (lat1, lon1) -> (lat2, lon2)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _rolling_circular_range_deg(values: np.ndarray) -> float:
    """max-min heading range, correct across the 0/360 wraparound boundary."""
    if len(values) < 2:
        return 0.0
    unwrapped = np.unwrap(np.deg2rad(values))
    return float(np.degrees(np.max(unwrapped) - np.min(unwrapped)))


def _rolling_slope(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values), dtype=float)
    slope = np.polyfit(x, values, 1)[0]
    return float(slope)


def _gps_derived_speed_kph(df: pd.DataFrame) -> pd.Series:
    """Speed implied by consecutive GPS fixes, for cross-checking the reported
    speed channel. Large divergence is the signature of GPS spoofing / jumps
    or of a drifting/faulty speed sensor -- a standard multi-sensor
    consistency check used in automotive fault and spoofing detection.

    Fully vectorized (groupby + shift, no .apply) so it behaves identically
    regardless of how many rows/groups are present -- important since this
    also runs on tiny single-vehicle windows in the online streaming path.
    """
    by_vehicle = df.groupby("vehicle_id")
    lat_prev = by_vehicle["lat"].shift(1)
    lon_prev = by_vehicle["lon"].shift(1)
    dt_seconds = by_vehicle["event_time"].diff().dt.total_seconds()

    dist_km = haversine_km(lat_prev.to_numpy(), lon_prev.to_numpy(), df["lat"].to_numpy(), df["lon"].to_numpy())
    with np.errstate(divide="ignore", invalid="ignore"):
        derived_kph = np.where(dt_seconds.to_numpy() > 1e-6, dist_km / (dt_seconds.to_numpy() / 3600.0), np.nan)

    derived = pd.Series(derived_kph, index=df.index)
    # First sample per vehicle (and any degenerate dt) has no derived speed;
    # assume it agrees with the reported channel rather than penalizing it.
    return derived.fillna(df["speed_kph"])


def compute_row_features(df: pd.DataFrame, window: int = ROW_WINDOW_SAMPLES) -> pd.DataFrame:
    df = df.sort_values(["vehicle_id", "event_time"]).copy()
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)

    grouped = df.groupby("vehicle_id", group_keys=False)

    df["speed_var"] = (
        grouped["speed_kph"].rolling(window=window, min_periods=2).var().reset_index(level=0, drop=True).fillna(0.0)
    )
    df["accel_spike_max"] = (
        grouped["accel_mps2"]
        .rolling(window=window, min_periods=1)
        .apply(lambda x: float(np.max(np.abs(x))), raw=True)
        .reset_index(level=0, drop=True)
    )
    df["heading_range"] = (
        grouped["heading_deg"]
        .rolling(window=window, min_periods=2)
        .apply(_rolling_circular_range_deg, raw=True)
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    df["yaw_rate_mean"] = grouped["yaw_rate_dps"].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)

    # Windowed (not cumulative) sample density: drops during comms dropout /
    # packet-loss anomalies and is a genuine online-computable signal, unlike
    # a running row index.
    df["sample_count"] = (
        grouped["speed_kph"].rolling(window=window, min_periods=1).count().reset_index(level=0, drop=True)
    )

    gps_derived_speed = _gps_derived_speed_kph(df)
    df["gps_speed_delta_raw"] = (df["speed_kph"] - gps_derived_speed).abs()
    df["gps_speed_delta_mean"] = (
        df.groupby("vehicle_id")["gps_speed_delta_raw"]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df = df.drop(columns=["gps_speed_delta_raw"])

    df["speed_trend"] = (
        grouped["speed_kph"]
        .rolling(window=window, min_periods=1)
        .apply(_rolling_slope, raw=True)
        .reset_index(level=0, drop=True)
    )

    return df


__all__ = ["compute_row_features", "haversine_km", "FEATURE_COLUMNS"]
