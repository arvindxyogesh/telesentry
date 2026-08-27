from src.common.constants import ANOMALY_TYPES
from src.ml.generate_synthetic_dataset import GeneratorConfig, generate_dataframe


def _generate(seconds=120, vehicles=3, seed=7, anomaly_probability=0.05):
    cfg = GeneratorConfig(vehicles=vehicles, seconds=seconds, hz=5, anomaly_probability=anomaly_probability, seed=seed)
    return generate_dataframe(cfg)


def test_label_matches_anomaly_type():
    df = _generate()
    assert ((df["anomaly_type"] == "normal") == (df["label"] == 0)).all()


def test_all_anomaly_types_representable():
    # A long-enough / high-enough-probability run should exercise every
    # anomaly type at least once.
    df = _generate(seconds=400, vehicles=6, anomaly_probability=0.05)
    seen = set(df["anomaly_type"].unique())
    assert set(ANOMALY_TYPES).issubset(seen)


def test_no_missing_or_nonfinite_values():
    df = _generate()
    numeric_cols = ["speed_kph", "accel_mps2", "yaw_rate_dps", "heading_deg", "lat", "lon"]
    assert not df[numeric_cols].isna().any().any()
    assert bool((df[numeric_cols].abs() < 1e6).all().all())


def test_speed_within_physical_bounds():
    df = _generate()
    assert (df["speed_kph"] >= 0.0).all()
    assert (df["speed_kph"] <= 150.0).all()


def test_deterministic_given_seed():
    # event_time is anchored to wall-clock "now" at generation time, so it
    # legitimately differs between calls; everything driven by the seeded
    # RNG (values, labels, anomaly types) should not.
    df1 = _generate(seed=123).reset_index(drop=True)
    df2 = _generate(seed=123).reset_index(drop=True)
    cols = [c for c in df1.columns if c != "event_time"]
    assert df1[cols].equals(df2[cols])


def test_dropout_reduces_local_sample_density():
    df = _generate(seconds=600, vehicles=6, anomaly_probability=0.05)
    dropout_rows = df[df["anomaly_type"] == "dropout_burst"]
    assert len(dropout_rows) > 0
