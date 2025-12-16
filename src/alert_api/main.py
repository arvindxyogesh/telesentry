from pathlib import Path
import json

import pandas as pd
from fastapi import FastAPI, HTTPException, Query


ALERTS_PATH = Path("data/processed/anomaly_alerts")
METRICS_PATH = Path("data/processed/batch_metrics.json")

app = FastAPI(title="Vehicle Telemetry Anomaly Alerts API", version="1.0.0")


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
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Batch metrics not available")

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
