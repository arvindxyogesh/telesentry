from pathlib import Path
import json

import pandas as pd
from fastapi import FastAPI, HTTPException, Query


ALERTS_PATH = Path("data/processed/anomaly_alerts")
BATCH_METRICS_PATH = Path("data/processed/batch_metrics.json")
MODEL_COMPARISON_PATH = Path("report/model_comparison.json")

app = FastAPI(title="Vehicle Telemetry Anomaly Alerts API", version="2.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/alerts/latest")
def latest_alerts(limit: int = Query(default=50, ge=1, le=500)) -> dict:
    if not ALERTS_PATH.exists():
        raise HTTPException(status_code=404, detail="Alerts data not available")

    df = pd.read_parquet(ALERTS_PATH)
    if df.empty:
        return {"alerts": [], "count": 0}

    df = df.sort_values("processing_time", ascending=False).head(limit)
    return {
        "count": int(len(df)),
        "alerts": df.to_dict(orient="records"),
    }


@app.get("/metrics")
def metrics() -> dict:
    """Isolation-Forest-only batch metrics (evaluate_batch.py). Kept for
    backward compatibility -- see /models/comparison for the full benchmark.
    """
    if not BATCH_METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Batch metrics not available")

    with open(BATCH_METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/models/comparison")
def models_comparison() -> dict:
    """Full cross-model benchmark (evaluate_models.py): z-score baseline,
    Isolation Forest, LSTM autoencoder, and the attention/Transformer
    detector, with per-anomaly-type recall and latency.
    """
    if not MODEL_COMPARISON_PATH.exists():
        raise HTTPException(status_code=404, detail="Model comparison not available")

    with open(MODEL_COMPARISON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
