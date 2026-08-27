from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st

ALERTS_PATH = Path("data/processed/anomaly_alerts")
BATCH_METRICS_PATH = Path("data/processed/batch_metrics.json")
MODEL_COMPARISON_PATH = Path("report/model_comparison.json")

st.set_page_config(page_title="Vehicle Telemetry Anomaly Dashboard", layout="wide")
st.title("Vehicle Telemetry Real-Time Anomaly Monitoring")

tab_live, tab_compare = st.tabs(["Live Alerts", "Model Comparison"])

with tab_live:
    if not ALERTS_PATH.exists():
        st.warning("No alert data yet. Run the local streaming runner or the Spark job + simulator first.")
    else:
        alerts = pd.read_parquet(ALERTS_PATH)
        if alerts.empty:
            st.info("Alerts table exists but currently empty.")
        else:
            alerts["processing_time"] = pd.to_datetime(alerts["processing_time"], utc=True)
            alerts["window_end"] = pd.to_datetime(alerts["window_end"], utc=True)
            alerts = alerts.sort_values("processing_time")

            latency_col = next((c for c in ["detection_delay_sec", "inference_latency_ms"] if c in alerts.columns), None)
            score_col = "anomaly_score" if "anomaly_score" in alerts.columns else "iforest_score"
            backend = alerts["model_backend"].iloc[-1] if "model_backend" in alerts.columns else "unknown"

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Alerts", int(len(alerts)))
            c1.caption(f"Model backend: {backend}")
            if latency_col:
                c2.metric(f"Mean {latency_col}", f"{alerts[latency_col].mean():.2f}")
                c3.metric(f"P95 {latency_col}", f"{alerts[latency_col].quantile(0.95):.2f}")

            st.subheader("Anomalies Over Time")
            count_by_time = alerts.set_index("processing_time").resample("10s").size().reset_index(name="alerts")
            st.plotly_chart(px.line(count_by_time, x="processing_time", y="alerts", markers=True), use_container_width=True)

            st.subheader("Anomaly Score Distribution")
            st.plotly_chart(px.histogram(alerts, x=score_col, nbins=40), use_container_width=True)

            st.subheader("Top Vehicles by Alert Volume")
            by_vehicle = alerts.groupby("vehicle_id").size().reset_index(name="alerts").sort_values("alerts", ascending=False).head(20)
            st.plotly_chart(px.bar(by_vehicle, x="vehicle_id", y="alerts"), use_container_width=True)

            if BATCH_METRICS_PATH.exists():
                st.subheader("Isolation Forest Batch Metrics")
                with open(BATCH_METRICS_PATH, "r", encoding="utf-8") as f:
                    st.json(json.load(f))

            st.subheader("Recent Alerts")
            display_cols = [c for c in ["vehicle_id", "window_start", "window_end", score_col, latency_col] if c and c in alerts.columns]
            st.dataframe(alerts[display_cols].sort_values("window_end", ascending=False).head(200), use_container_width=True)

with tab_compare:
    if not MODEL_COMPARISON_PATH.exists():
        st.warning("No model comparison yet. Run `python -m src.ml.evaluate_models` first.")
    else:
        with open(MODEL_COMPARISON_PATH, "r", encoding="utf-8") as f:
            comparison = json.load(f)

        models = comparison["models"]
        st.caption(f"Evaluated at {comparison['evaluated_at']} on {comparison['samples']} rows from {comparison['input']}")
        st.metric("Best F1", comparison["best_f1_model"])

        summary_rows = []
        for name, m in models.items():
            summary_rows.append(
                {
                    "model": name,
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "auroc": m["auroc"],
                    "auprc": m["auprc"],
                    "false_positive_rate": m["false_positive_rate"],
                    "mean_detection_delay_sec": m["mean_detection_delay_sec"],
                    "latency_ms_per_1000_rows": m["latency_ms_per_1000_rows"],
                }
            )
        summary_df = pd.DataFrame(summary_rows)

        st.subheader("Precision / Recall / F1 / AUROC by Model")
        metric_long = summary_df.melt(id_vars="model", value_vars=["precision", "recall", "f1", "auroc"], var_name="metric", value_name="value")
        st.plotly_chart(px.bar(metric_long, x="model", y="value", color="metric", barmode="group"), use_container_width=True)

        st.subheader("False Positive Rate vs. Detection Delay")
        st.plotly_chart(
            px.scatter(
                summary_df,
                x="false_positive_rate",
                y="mean_detection_delay_sec",
                text="model",
                size="latency_ms_per_1000_rows",
                hover_data=["precision", "recall", "f1"],
            ),
            use_container_width=True,
        )

        st.subheader("Recall by Anomaly Type")
        per_type_rows = []
        for name, m in models.items():
            for anomaly_type, recall in m["per_anomaly_type_recall"].items():
                if recall is not None:
                    per_type_rows.append({"model": name, "anomaly_type": anomaly_type, "recall": recall})
        per_type_df = pd.DataFrame(per_type_rows)
        st.plotly_chart(px.bar(per_type_df, x="anomaly_type", y="recall", color="model", barmode="group"), use_container_width=True)

        st.subheader("ROC Curves")
        roc_df_rows = []
        for name, curve in comparison.get("roc_curves", {}).items():
            for fpr, tpr in zip(curve["fpr"], curve["tpr"]):
                roc_df_rows.append({"model": name, "fpr": fpr, "tpr": tpr})
        if roc_df_rows:
            st.plotly_chart(px.line(pd.DataFrame(roc_df_rows), x="fpr", y="tpr", color="model"), use_container_width=True)

        st.subheader("Full Comparison Table")
        st.dataframe(summary_df, use_container_width=True)
