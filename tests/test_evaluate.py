import pandas as pd

from src.ml.evaluate_batch import compute_detection_delay_seconds


def _row(vehicle_id, t, label, pred):
    return {"vehicle_id": vehicle_id, "event_time": pd.Timestamp(t, tz="UTC"), "label": label, "pred": pred}


def test_detection_delay_zero_when_detected_immediately():
    df = pd.DataFrame(
        [
            _row("veh_1", "2026-01-01T00:00:00", 0, 0),
            _row("veh_1", "2026-01-01T00:00:01", 1, 1),
            _row("veh_1", "2026-01-01T00:00:02", 1, 1),
            _row("veh_1", "2026-01-01T00:00:03", 0, 0),
        ]
    )
    assert compute_detection_delay_seconds(df) == 0.0


def test_detection_delay_measures_lag():
    df = pd.DataFrame(
        [
            _row("veh_1", "2026-01-01T00:00:00", 0, 0),
            _row("veh_1", "2026-01-01T00:00:01", 1, 0),
            _row("veh_1", "2026-01-01T00:00:02", 1, 0),
            _row("veh_1", "2026-01-01T00:00:03", 1, 1),
            _row("veh_1", "2026-01-01T00:00:04", 0, 0),
        ]
    )
    assert compute_detection_delay_seconds(df) == 2.0


def test_detection_delay_infinite_when_never_detected():
    df = pd.DataFrame(
        [
            _row("veh_1", "2026-01-01T00:00:00", 0, 0),
            _row("veh_1", "2026-01-01T00:00:01", 1, 0),
            _row("veh_1", "2026-01-01T00:00:02", 1, 0),
        ]
    )
    assert compute_detection_delay_seconds(df) == float("inf")


def test_detection_delay_averages_across_episodes():
    df = pd.DataFrame(
        [
            _row("veh_1", "2026-01-01T00:00:00", 1, 1),  # delay 0
            _row("veh_1", "2026-01-01T00:00:01", 0, 0),
            _row("veh_1", "2026-01-01T00:00:02", 1, 0),
            _row("veh_1", "2026-01-01T00:00:03", 1, 1),  # delay 1
        ]
    )
    assert compute_detection_delay_seconds(df) == 0.5
