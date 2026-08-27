import numpy as np
import pandas as pd

from src.common.constants import FEATURE_COLUMNS
from src.ml.features import compute_row_features
from src.ml.generate_synthetic_dataset import GeneratorConfig, generate_dataframe
from src.ml.sequence_dataset import build_windows


def test_build_windows_shape_and_alignment():
    cfg = GeneratorConfig(vehicles=3, seconds=100, hz=5, anomaly_probability=0.05, seed=1)
    df = compute_row_features(generate_dataframe(cfg))
    window = 15

    windows = build_windows(df, FEATURE_COLUMNS, window=window)

    assert windows.X.shape[1] == window
    assert windows.X.shape[2] == len(FEATURE_COLUMNS)
    assert len(windows.X) == len(windows.row_index) == len(windows.labels) == len(windows.anomaly_types)

    # Each window's last feature row must exactly match the source row it is
    # aligned to (no off-by-one).
    sample_pos = df.index.get_indexer(windows.row_index[:5])
    for i, pos in enumerate(sample_pos):
        expected = df.iloc[pos][FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        assert np.allclose(windows.X[i, -1, :], expected, atol=1e-4)


def test_clean_mask_excludes_windows_touching_anomalies():
    cfg = GeneratorConfig(vehicles=4, seconds=200, hz=5, anomaly_probability=0.05, seed=2)
    df = compute_row_features(generate_dataframe(cfg))
    windows = build_windows(df, FEATURE_COLUMNS, window=15)

    clean = windows.clean_mask()
    assert clean.sum() > 0
    assert clean.sum() < len(clean)
    assert (windows.labels[clean] == 0).all()


def test_short_series_dropped():
    df = pd.DataFrame(
        {
            "event_time": pd.date_range("2026-01-01", periods=3, freq="200ms", tz="UTC"),
            "vehicle_id": ["veh_000"] * 3,
            "label": [0, 0, 0],
            "anomaly_type": ["normal"] * 3,
            **{col: [0.0, 0.0, 0.0] for col in FEATURE_COLUMNS},
        }
    )
    windows = build_windows(df, FEATURE_COLUMNS, window=15)
    assert len(windows.X) == 0
