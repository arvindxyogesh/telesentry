"""Windowing utilities that turn the per-row rolling feature table into
trailing sequences for the sequence models (LSTM autoencoder, attention
detector).

Windows are strictly trailing/causal: the window ending at row i only uses
rows <= i from the same vehicle, so scores are computable online exactly as
they would be from a live trailing buffer (matching the local streaming
runner's semantics), and there is no leakage from the future.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.constants import SEQUENCE_WINDOW


@dataclass
class WindowedFeatures:
    """Sequence tensor plus the row-level metadata each window ends on."""

    X: np.ndarray  # (num_windows, window, num_features), float32
    row_index: np.ndarray  # original dataframe index each window's last step corresponds to
    labels: np.ndarray  # binary label at the window's last step
    anomaly_types: np.ndarray  # anomaly_type string at the window's last step
    window_has_anomaly: np.ndarray  # True if ANY timestep inside the window is anomalous

    def clean_mask(self) -> np.ndarray:
        """Windows with zero anomalous timesteps anywhere inside them -- the
        safe subset for unsupervised training on normal behavior only.
        """
        return ~self.window_has_anomaly


def build_windows(
    df: pd.DataFrame,
    feature_cols: list,
    window: int = SEQUENCE_WINDOW,
    label_col: str = "label",
    anomaly_type_col: str = "anomaly_type",
) -> WindowedFeatures:
    """Build trailing per-vehicle windows of length `window` over feature_cols.

    A vehicle needs at least `window` rows of history before it contributes
    its first window, so the first (window - 1) rows per vehicle are dropped.
    """
    xs, idxs, labels, types, any_anomaly = [], [], [], [], []
    has_labels = label_col in df.columns
    has_types = anomaly_type_col in df.columns

    for _, group in df.groupby("vehicle_id", sort=False):
        group = group.sort_values("event_time")
        values = group[feature_cols].to_numpy(dtype=np.float32)
        n = len(values)
        if n < window:
            continue

        windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window, axis=0)
        # sliding_window_view puts the new window axis last; move it to axis 1
        # so the result is (num_windows, window, num_features).
        windows = np.moveaxis(windows, -1, 1)
        xs.append(windows)

        end_positions = group.index.to_numpy()[window - 1 :]
        idxs.append(end_positions)
        if has_labels:
            label_values = group[label_col].to_numpy()
            labels.append(label_values[window - 1 :])
            label_windows = np.lib.stride_tricks.sliding_window_view(label_values, window_shape=window)
            any_anomaly.append(label_windows.max(axis=-1) > 0)
        if has_types:
            types.append(group[anomaly_type_col].to_numpy()[window - 1 :])

    if not xs:
        return WindowedFeatures(
            X=np.empty((0, window, len(feature_cols)), dtype=np.float32),
            row_index=np.empty((0,), dtype=np.int64),
            labels=np.empty((0,), dtype=np.int64),
            anomaly_types=np.empty((0,), dtype=object),
            window_has_anomaly=np.empty((0,), dtype=bool),
        )

    X = np.concatenate(xs, axis=0)
    row_index = np.concatenate(idxs, axis=0)
    labels_arr = np.concatenate(labels, axis=0).astype(np.int64) if has_labels else np.zeros(len(row_index), dtype=np.int64)
    types_arr = np.concatenate(types, axis=0) if has_types else np.full(len(row_index), "unknown", dtype=object)
    any_anomaly_arr = (
        np.concatenate(any_anomaly, axis=0) if has_labels else np.zeros(len(row_index), dtype=bool)
    )

    return WindowedFeatures(
        X=X, row_index=row_index, labels=labels_arr, anomaly_types=types_arr, window_has_anomaly=any_anomaly_arr
    )


__all__ = ["WindowedFeatures", "build_windows"]
