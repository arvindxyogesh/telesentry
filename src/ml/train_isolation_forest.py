import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def compute_row_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["vehicle_id", "event_time"]).copy()
    grouped = df.groupby("vehicle_id", group_keys=False)

    df["speed_var"] = grouped["speed_kph"].rolling(window=10, min_periods=2).var().reset_index(level=0, drop=True).fillna(0.0)
    df["accel_spike_max"] = grouped["accel_mps2"].rolling(window=10, min_periods=1).apply(lambda x: float(np.max(np.abs(x))), raw=True).reset_index(level=0, drop=True)
    df["heading_range"] = grouped["heading_deg"].rolling(window=10, min_periods=2).apply(lambda x: float(np.max(x) - np.min(x)), raw=True).reset_index(level=0, drop=True).fillna(0.0)
    df["yaw_rate_mean"] = grouped["yaw_rate_dps"].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    df["sample_count"] = grouped.cumcount() + 1
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Isolation Forest for telemetry anomalies.")
    parser.add_argument("--input", default="data/raw/synthetic_telemetry.parquet")
    parser.add_argument("--model-out", default="models/isolation_forest.joblib")
    parser.add_argument("--scaler-out", default="models/feature_scaler.joblib")
    parser.add_argument("--contamination", type=float, default=0.03)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    df = compute_row_features(df)

    feature_cols = ["speed_var", "accel_spike_max", "heading_range", "yaw_rate_mean", "sample_count"]
    train_df = df[df.get("label", 0) == 0].copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(train_df[feature_cols])

    model = IsolationForest(
        n_estimators=250,
        contamination=args.contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    joblib.dump(model, args.model_out)
    joblib.dump(scaler, args.scaler_out)

    print(f"Trained model on {len(train_df)} normal rows")
    print(f"Saved model: {args.model_out}")
    print(f"Saved scaler: {args.scaler_out}")


if __name__ == "__main__":
    main()
