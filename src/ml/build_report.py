import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build run report assets from anomaly outputs.")
    parser.add_argument("--alerts", default="data/processed/anomaly_alerts")
    parser.add_argument("--metrics", default="data/processed/batch_metrics.json")
    parser.add_argument("--report-dir", default="report")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(args.metrics, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    alerts_path = Path(args.alerts)
    if alerts_path.exists():
        alerts = pd.read_parquet(alerts_path)
    else:
        alerts = pd.DataFrame()

    summary = {
        "batch_metrics": metrics,
        "streaming": {
            "total_alerts": int(len(alerts)),
            "mean_detection_delay_sec": float(alerts["detection_delay_sec"].mean()) if not alerts.empty else None,
            "p95_detection_delay_sec": float(alerts["detection_delay_sec"].quantile(0.95)) if not alerts.empty else None,
            "vehicles_with_alerts": int(alerts["vehicle_id"].nunique()) if not alerts.empty else 0,
        },
    }

    if not alerts.empty:
        alerts["processing_time"] = pd.to_datetime(alerts["processing_time"], utc=True)

        by_time = alerts.set_index("processing_time").resample("10s").size()
        plt.figure(figsize=(10, 4))
        by_time.plot()
        plt.title("Streaming Alerts Over Time")
        plt.xlabel("Processing Time")
        plt.ylabel("Alert Count")
        plt.tight_layout()
        plt.savefig(fig_dir / "alerts_over_time.png", dpi=140)
        plt.close()

        plt.figure(figsize=(7, 4))
        alerts["detection_delay_sec"].hist(bins=30)
        plt.title("Detection Delay Distribution")
        plt.xlabel("Seconds")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(fig_dir / "detection_delay_hist.png", dpi=140)
        plt.close()

    with open(report_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_md = report_dir / "RUN_REPORT.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Vehicle Telemetry Anomaly Detection Report\n\n")
        f.write("## Objective\n")
        f.write("Build and validate a production-style real-time anomaly detection pipeline for vehicle telemetry.\n\n")

        f.write("## Dataset\n")
        f.write("- External dataset downloaded: FordA (UCR archive) into `data/raw/forda/`\n")
        f.write("- Streaming simulation dataset: synthetic CAN/IMU/GPS telemetry generator\n\n")

        f.write("## Batch Metrics\n")
        for k, v in metrics.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

        f.write("## Streaming Metrics\n")
        f.write(f"- total_alerts: {summary['streaming']['total_alerts']}\n")
        f.write(f"- mean_detection_delay_sec: {summary['streaming']['mean_detection_delay_sec']}\n")
        f.write(f"- p95_detection_delay_sec: {summary['streaming']['p95_detection_delay_sec']}\n")
        f.write(f"- vehicles_with_alerts: {summary['streaming']['vehicles_with_alerts']}\n\n")

        f.write("## Targets\n")
        f.write(f"- Detection delay target (<5 sec): {'PASS' if metrics.get('delay_target_met') else 'FAIL'}\n")
        red = metrics.get("fpr_reduction_vs_zscore_baseline", 0.0)
        f.write(f"- False positive reduction vs z-score baseline: {red:.2%}\n\n")

        f.write("## Figures\n")
        if (fig_dir / "alerts_over_time.png").exists():
            f.write("- `report/figures/alerts_over_time.png`\n")
        if (fig_dir / "detection_delay_hist.png").exists():
            f.write("- `report/figures/detection_delay_hist.png`\n")

    print(f"Wrote report: {report_md}")


if __name__ == "__main__":
    main()
