import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _build_streaming_figures(alerts: pd.DataFrame, fig_dir: Path) -> dict:
    latency_col = next((c for c in ["detection_delay_sec", "inference_latency_ms"] if c in alerts.columns), None)
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

    if latency_col:
        plt.figure(figsize=(7, 4))
        alerts[latency_col].hist(bins=30)
        plt.title(f"{latency_col} Distribution")
        plt.xlabel(latency_col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(fig_dir / "detection_delay_hist.png", dpi=140)
        plt.close()

    return {
        "total_alerts": int(len(alerts)),
        "latency_metric": latency_col,
        "mean_latency": float(alerts[latency_col].mean()) if latency_col else None,
        "p95_latency": float(alerts[latency_col].quantile(0.95)) if latency_col else None,
        "vehicles_with_alerts": int(alerts["vehicle_id"].nunique()),
    }


def _build_model_comparison_figures(comparison: dict, fig_dir: Path) -> None:
    models = comparison["models"]
    names = list(models.keys())

    metrics_to_plot = ["precision", "recall", "f1", "auroc"]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.2
    x = range(len(names))
    for i, metric in enumerate(metrics_to_plot):
        values = [models[n][metric] for n in names]
        ax.bar([xi + i * width for xi in x], values, width=width, label=metric)
    ax.set_xticks([xi + width * 1.5 for xi in x])
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Detector Comparison: Precision / Recall / F1 / AUROC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "model_comparison_metrics.png", dpi=140)
    plt.close()

    anomaly_types = sorted({t for m in models.values() for t in m["per_anomaly_type_recall"]})
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / max(1, len(names))
    x = range(len(anomaly_types))
    for i, name in enumerate(names):
        values = [models[name]["per_anomaly_type_recall"].get(t) or 0.0 for t in anomaly_types]
        ax.bar([xi + i * width for xi in x], values, width=width, label=name)
    ax.set_xticks([xi + width * (len(names) - 1) / 2 for xi in x])
    ax.set_xticklabels(anomaly_types, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Recall by Anomaly Type")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "model_comparison_per_type_recall.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, curve in comparison.get("roc_curves", {}).items():
        ax.plot(curve["fpr"], curve["tpr"], label=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "model_comparison_roc.png", dpi=140)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build run report assets from anomaly outputs.")
    parser.add_argument("--alerts", default="data/processed/anomaly_alerts")
    parser.add_argument("--batch-metrics", default="data/processed/batch_metrics.json")
    parser.add_argument("--model-comparison", default="report/model_comparison.json")
    parser.add_argument("--report-dir", default="report")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    batch_metrics = {}
    if Path(args.batch_metrics).exists():
        with open(args.batch_metrics, "r", encoding="utf-8") as f:
            batch_metrics = json.load(f)

    comparison = {}
    if Path(args.model_comparison).exists():
        with open(args.model_comparison, "r", encoding="utf-8") as f:
            comparison = json.load(f)
        _build_model_comparison_figures(comparison, fig_dir)

    alerts_path = Path(args.alerts)
    streaming_summary = {"total_alerts": 0, "latency_metric": None, "mean_latency": None, "p95_latency": None, "vehicles_with_alerts": 0}
    if alerts_path.exists():
        alerts = pd.read_parquet(alerts_path)
        if not alerts.empty:
            streaming_summary = _build_streaming_figures(alerts, fig_dir)

    summary = {
        "batch_metrics": batch_metrics,
        "model_comparison": comparison.get("models", {}),
        "streaming": streaming_summary,
    }
    with open(report_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_md = report_dir / "RUN_REPORT.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Vehicle Telemetry Anomaly Detection: Run Report\n\n")
        f.write("## Objective\n")
        f.write(
            "Benchmark classical (Isolation Forest) and deep sequence anomaly detectors (LSTM autoencoder, "
            "attention/Transformer detector) on a labeled, multi-type vehicle telemetry benchmark, and serve "
            "the best model through a real-time streaming pipeline.\n\n"
        )

        f.write("## Dataset\n")
        if comparison:
            f.write(f"- Held-out evaluation set: `{comparison.get('input', 'n/a')}` ({comparison.get('samples', 'n/a')} rows)\n")
            f.write("- Anomaly type distribution (evaluation set):\n")
            for anomaly_type, frac in comparison.get("anomaly_type_distribution", {}).items():
                f.write(f"  - {anomaly_type}: {frac:.2%}\n")
        f.write("\n")

        if comparison:
            f.write("## Model Comparison (held-out evaluation set)\n\n")
            f.write("| Model | Precision | Recall | F1 | AUROC | AUPRC | FPR | Mean Delay (s) | Latency (ms/1000 rows) |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for name, m in comparison["models"].items():
                f.write(
                    f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['auroc']:.3f} | "
                    f"{m['auprc']:.3f} | {m['false_positive_rate']:.4f} | {m['mean_detection_delay_sec']:.2f} | "
                    f"{m['latency_ms_per_1000_rows']:.2f} |\n"
                )
            f.write(f"\n**Best F1:** `{comparison['best_f1_model']}`\n\n")

            f.write("### Recall by Anomaly Type\n\n")
            anomaly_types = sorted({t for m in comparison["models"].values() for t in m["per_anomaly_type_recall"]})
            f.write("| Model | " + " | ".join(anomaly_types) + " |\n")
            f.write("|---|" + "---|" * len(anomaly_types) + "\n")
            for name, m in comparison["models"].items():
                cells = [f"{(m['per_anomaly_type_recall'].get(t) or 0.0):.2f}" for t in anomaly_types]
                f.write(f"| {name} | " + " | ".join(cells) + " |\n")
            f.write("\n")

        if batch_metrics:
            f.write("## Isolation Forest Batch Metrics (quick-iteration path)\n")
            for k, v in batch_metrics.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")

        f.write("## Streaming Run\n")
        f.write(f"- total_alerts: {streaming_summary['total_alerts']}\n")
        f.write(f"- latency_metric: {streaming_summary['latency_metric']}\n")
        f.write(f"- mean_latency: {streaming_summary['mean_latency']}\n")
        f.write(f"- p95_latency: {streaming_summary['p95_latency']}\n")
        f.write(f"- vehicles_with_alerts: {streaming_summary['vehicles_with_alerts']}\n\n")

        f.write("## Figures\n")
        for name in [
            "model_comparison_metrics.png",
            "model_comparison_per_type_recall.png",
            "model_comparison_roc.png",
            "alerts_over_time.png",
            "detection_delay_hist.png",
        ]:
            if (fig_dir / name).exists():
                f.write(f"- `report/figures/{name}`\n")

    print(f"Wrote report: {report_md}")


if __name__ == "__main__":
    main()
