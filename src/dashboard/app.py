from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st

ALERTS_PATH = Path("data/processed/anomaly_alerts")
METRICS_PATH = Path("data/processed/batch_metrics.json")

st.set_page_config(page_title="Vehicle Telemetry Anomaly Dashboard", layout="wide")
st.title("Vehicle Telemetry Real-Time Anomaly Monitoring")

if not ALERTS_PATH.exists():
    st.warning("No alert data yet. Start simulator + streaming job first.")
    st.stop()

alerts = pd.read_parquet(ALERTS_PATH)
if alerts.empty:
    st.info("Alerts table exists but currently empty.")
    st.stop()

alerts["processing_time"] = pd.to_datetime(alerts["processing_time"], utc=True)
alerts["window_end"] = pd.to_datetime(alerts["window_end"], utc=True)
alerts = alerts.sort_values("processing_time")

c1, c2, c3 = st.columns(3)
c1.metric("Total Alerts", int(len(alerts)))
c2.metric("Mean Detection Delay (s)", f"{alerts['detection_delay_sec'].mean():.2f}")
c3.metric("P95 Detection Delay (s)", f"{alerts['detection_delay_sec'].quantile(0.95):.2f}")

st.subheader("Anomalies Over Time")
count_by_time = alerts.set_index("processing_time").resample("10s").size().reset_index(name="alerts")
fig1 = px.line(count_by_time, x="processing_time", y="alerts", markers=True)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Anomaly Score Distribution")
fig2 = px.histogram(alerts, x="iforest_score", nbins=40)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Top Vehicles by Alert Volume")
by_vehicle = alerts.groupby("vehicle_id").size().reset_index(name="alerts").sort_values("alerts", ascending=False).head(20)
fig3 = px.bar(by_vehicle, x="vehicle_id", y="alerts")
st.plotly_chart(fig3, use_container_width=True)

if METRICS_PATH.exists():
    st.subheader("Batch Evaluation Metrics")
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        st.json(json.load(f))

st.subheader("Recent Alerts")
st.dataframe(
    alerts[["vehicle_id", "window_start", "window_end", "iforest_score", "detection_delay_sec"]]
    .sort_values("window_end", ascending=False)
    .head(200),
    use_container_width=True,
)
