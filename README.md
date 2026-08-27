# Telesentry

**Benchmarking classical vs. deep sequence anomaly detectors for vehicle telemetry.**

A research-style benchmark and a working real-time pipeline, in one repo. It compares a
classical one-class detector (Isolation Forest) against two deep sequence anomaly
detectors — an LSTM autoencoder and a self-attention/Transformer detector — on a
labeled, multi-type synthetic vehicle telemetry benchmark, then serves the winning
model through a Kafka + Spark Structured Streaming pipeline (with a dependency-light
local fallback) behind a REST API and a live dashboard.

## Why

Most "vehicle telemetry anomaly detection" demos reduce to one thing: threshold a
z-score on acceleration and call it a day. That catches harsh braking. It does not
catch a GPS feed that's been spoofed to a fixed false location, a heading sensor that's
silently frozen while the car keeps turning, or a speed sensor drifting out of
calibration over several seconds — failure modes that don't look like a spike, they
look like an *inconsistency* between what different signals imply. Those require a
model of the joint, multivariate normal — which is exactly what reconstruction-based
sequence models (LSTM autoencoders, attention-based detectors) are for, and why they've
been the standard comparison point in time-series anomaly detection research since
Malhotra et al. (2016), and especially since attention-based approaches like Anomaly
Transformer (Xu et al., ICLR 2022) and TranAD (Tuli et al., VLDB 2022). This project
applies that line of work to vehicle telemetry and measures, honestly, where it helps
and where it doesn't.

## Anomaly taxonomy

The synthetic generator (`src/ml/generate_synthetic_dataset.py`) produces five distinct,
independently-labeled anomaly types, each with a different multivariate signature —
not one generic "spike":

| Type | What happens | Signature | Catchable by a fixed threshold? |
|---|---|---|---|
| `harsh_maneuver` | Real evasive steering + hard accel/brake | Simultaneous spike in accel, yaw rate, heading range, speed variance | Yes — this is what thresholds are built for |
| `sensor_stuck` | Heading/yaw channel freezes while the vehicle keeps turning | *Implausibly low* heading variance | No — one-sided thresholds can't see "too smooth" |
| `sensor_drift` | Speed channel accumulates a slow additive bias | Growing trend + growing GPS-vs-reported-speed mismatch, no spike | No — nothing crosses a spike threshold |
| `gps_spoof` | Reported GPS pins to a fixed false location | Large GPS-vs-reported-speed mismatch, everything else normal | No — accelerometer/gyro see nothing wrong |
| `dropout_burst` | Telemetry link drops most packets for a few seconds | Reduced sample density in the trailing window | No — every individual value is fine |

Feature engineering (`src/ml/features.py`) computes 7 rolling-window features per event:
`speed_var`, `accel_spike_max`, `heading_range` (circular-wraparound-correct),
`yaw_rate_mean`, `sample_count` (windowed, not cumulative — this is what makes
`dropout_burst` detectable), `gps_speed_delta_mean` (GPS-position-implied speed vs.
reported speed — a standard multi-sensor consistency check, and what makes
`gps_spoof`/`sensor_drift` detectable), and `speed_trend` (rolling slope).

## Models compared

| Model | Class | Trained on | Idea |
|---|---|---|---|
| `zscore_baseline` | Fixed rule | n/a | 3σ rule on `accel_spike_max`. The "naive practitioner" baseline. |
| `isolation_forest` | Classical ML | Normal rows only | Tree-based density estimate over the 7 tabular features. |
| `lstm_autoencoder` | Deep, recurrent | Normal windows only | Seq2seq LSTM reconstructs a 15-step (~3s) trailing window; reconstruction error = anomaly score. |
| `transformer_detector` | Deep, attention | Normal windows only | Self-attention encoder reconstructs the same window; score = reconstruction error **+** attention-concentration (peakiness) of the last timestep's attention row. |

All four are calibrated to the same target: a decision threshold set at the 97th
percentile of scores on held-out *normal* data (≈3% target false-positive rate), so the
comparison isn't confounded by one model getting a friendlier cutoff. The
Transformer's attention term is a simplified, single-branch nod to the
reconstruction + association-discrepancy idea from Anomaly Transformer — not a
reproduction of its two-branch adversarial training, which is out of scope here (see
Limitations).

## Results (held-out evaluation set, 46,186 rows, 8 vehicles never seen in training)

| Model | Precision | Recall | F1 | AUROC | AUPRC | FPR | Mean delay (s) |
|---|---|---|---|---|---|---|---|
| zscore_baseline | 0.577 | 0.115 | 0.192 | 0.558 | 0.345 | 0.029 | 0.23 |
| isolation_forest | 0.531 | 0.097 | 0.164 | 0.820 | 0.545 | 0.030 | 0.51 |
| **lstm_autoencoder** | 0.537 | **0.522** | **0.530** | 0.734 | 0.462 | 0.157 | 0.81 |
| transformer_detector | 0.515 | 0.517 | 0.516 | 0.732 | 0.454 | 0.170 | 0.62 |

### Recall by anomaly type

| Model | harsh_maneuver | sensor_stuck | sensor_drift | gps_spoof | dropout_burst |
|---|---|---|---|---|---|
| zscore_baseline | 0.88 | 0.01 | 0.01 | 0.01 | 0.01 |
| isolation_forest | 0.73 | 0.01 | 0.01 | 0.01 | 0.01 |
| lstm_autoencoder | **1.00** | 0.09 | **0.60** | 0.75 | 0.13 |
| transformer_detector | **1.00** | 0.07 | 0.51 | **0.87** | 0.15 |

**Takeaways, honestly stated:**
- Every method nails `harsh_maneuver` — it's a spike, thresholds are made for this.
- Classical methods are **nearly blind** to `sensor_drift` and `gps_spoof` (≤1% recall):
  these anomalies never cross a one-sided threshold on any single feature, so a
  detector that only looks at "is this feature too big" cannot see them almost by
  construction. Isolation Forest's much higher AUROC (0.82 vs. 0.56–0.73) than its
  fixed-threshold recall suggests shows it *ranks* these events better than a naive
  threshold — it just needs a threshold tuned per-anomaly-type to exploit that, which
  isn't how a single deployed cutoff works.
- Deep sequence models recover 60–87% recall on drift/spoof — the multivariate,
  no-single-spike anomalies they were expected to help with — at a real cost: ~15–17%
  false-positive rate against a 3% calibration target, i.e. a meaningful
  train→held-out-vehicle generalization gap. That gap, not raw architecture, is the
  main axis for future work (see below).
- **`sensor_stuck` and `dropout_burst` are hard for everyone** (≤15% recall across the
  board). Both are "absence of expected variation" signatures that the current feature
  set only captures weakly. This is the most interesting open problem the benchmark
  surfaces, not a result to paper over.

Regenerate this table anytime with `make evaluate-all` (or `python -m
src.ml.evaluate_models`) — see `report/RUN_REPORT.md` for the full run this table was
generated from, with figures in `report/figures/`.

## Architecture

```
generate_synthetic_dataset.py --> (train.parquet, eval.parquet)
        |
        v
  features.py (shared feature pipeline: offline batch AND online streaming)
        |
        +--> train_isolation_forest.py --> isolation_forest.joblib + calibration
        +--> train_deep_models.py      --> lstm_autoencoder.pt, transformer_detector.pt + calibration
        |
        v
  evaluate_models.py --> report/model_comparison.{json,csv} (the benchmark)
        |
        v
  build_report.py --> report/RUN_REPORT.md + figures

Real-time serving (either path writes the same alerts table):
  simulator --> Kafka --> feature_job.py (Spark Structured Streaming,
                            trained Isolation Forest applied per micro-batch)
                                    |
  OR (no Kafka/Spark needed):      v
  local_streaming_runner.py --> data/processed/anomaly_alerts (parquet)
   (selectable backend: threshold / iforest / lstm / transformer)          |
                                                                             v
                                                    alert_api (FastAPI) + dashboard (Streamlit)
```

The row-level feature computation (`src/ml/features.py`) is the single source of truth
used by both training and the local streaming runner, so there's no
training/serving feature skew. The Spark path currently computes 5 of the 7 features
online (see Limitations) and imputes the other 2 with their training-mean.

## Quick Start (Local)

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # makes `src` importable from anywhere, for every entry point below
```

`torch` pulls a large CUDA-enabled build by default even on CPU-only machines; for a
much smaller install use `pip install torch --index-url https://download.pytorch.org/whl/cpu`
before the `-r requirements.txt` line.

### 2) Generate the train/eval benchmark

```bash
make generate-train   # 16 vehicles x 2400s @ 5Hz, seed 42
make generate-eval     # 8 vehicles x 1200s @ 5Hz, seed 4242 (disjoint vehicles/seed)
```

### 3) Train all three models

```bash
make train        # Isolation Forest + calibration
make train-deep    # LSTM autoencoder + Transformer detector + calibration
```

### 4) Run the benchmark and build the report

```bash
make evaluate-all   # report/model_comparison.{json,csv}
make report          # report/RUN_REPORT.md + report/figures/*.png
```

Or all at once: `make all`.

### 5) Try the real-time path (no Kafka/Spark required)

```bash
python src/streaming/local_streaming_runner.py \
  --input data/raw/synthetic_telemetry_eval.parquet \
  --model-backend lstm \
  --output data/processed/anomaly_alerts
```

`--model-backend` is one of `threshold` (no trained model needed), `iforest`, `lstm`,
`transformer`.

### 6) Or the full Kafka + Spark path

```bash
docker compose up -d
python src/simulator/can_imu_gps_simulator.py --bootstrap-servers localhost:9092 --topic telemetry.raw
python src/streaming/feature_job.py --bootstrap-servers localhost:9092 --topic telemetry.raw
```

### 7) Serve alerts + dashboard

```bash
uvicorn src.alert_api.main:app --host 0.0.0.0 --port 8000 --reload
streamlit run src/dashboard/app.py --server.port 8501
```

The dashboard has two tabs: **Live Alerts** (whatever streaming path you ran) and
**Model Comparison** (the full benchmark from step 4).

## API

- `GET /health`
- `GET /alerts/latest?limit=50`
- `GET /metrics` — Isolation-Forest-only batch metrics (quick-iteration path)
- `GET /models/comparison` — full cross-model benchmark with per-anomaly-type recall

## Repository structure

```
src/common/          shared constants + the Spark telemetry schema
src/ml/features.py   the one feature pipeline (offline + online)
src/ml/models/        LSTMAutoencoder, TransformerAnomalyDetector (PyTorch)
src/ml/               generator, sequence windowing, train/evaluate scripts, report builder
src/streaming/         Spark structured streaming job + the local (Spark-free) fallback runner
src/simulator/         Kafka telemetry producer
src/alert_api/         FastAPI alert/metrics/comparison API
src/dashboard/         Streamlit live + comparison dashboard
tests/                 unit tests: features, generator, windowing, models, evaluation
```

## Limitations & Future Work

- **Synthetic-only benchmark.** All labeled anomalies come from the generator's
  taxonomy, not real vehicle fault/attack logs. `src/ml/download_datasets.py` fetches
  FordA (a real sensor time series) as a starting point for adapting the same
  reconstruction-based models to a real dataset in future work, but it is not part of
  the current pipeline.
- **Deep models generalize imperfectly to held-out vehicles**: 15–17% FPR against a 3%
  calibration target on the eval set. Worth investigating: more training vehicles/
  epochs, regularization, or calibrating per-vehicle rather than globally.
- **`sensor_stuck` and `dropout_burst` remain hard for every method.** The feature set
  captures "value is too high" much better than "variation is absent" — a dedicated
  variance-collapse or missing-data-density feature/model is a natural next step.
- **The Transformer detector is a simplified, single-branch reconstruction +
  attention-concentration score**, not a reproduction of Anomaly Transformer's
  adversarial two-branch "association discrepancy" training objective.
- **The Spark structured streaming job computes 5 of 7 features online.**
  `gps_speed_delta_mean` and `speed_trend` need an ordered per-vehicle lag over
  lat/lon/event_time, which in Structured Streaming requires stateful processing
  (`flatMapGroupsWithState`) rather than a plain windowed aggregate; the job currently
  imputes those 2 features with their training-time mean rather than leaving the
  trained model unused. The dependency-light local streaming runner does not have this
  limitation (it computes the full feature set every row).
- **`local_streaming_runner.py` is a low-throughput reference implementation** (recomputes
  rolling features and calls into scikit-learn/PyTorch per event, not batched) — it
  exists for demos and correctness-testing of the feature/model pipeline without
  standing up Kafka/Spark, not as a throughput benchmark. The Spark job is the scalable
  real-time path.

## Notes for Productionization

- Replace the synthetic source with a real CAN bus / OBD-II connector.
- Store alerts/features in Delta Lake instead of parquet for ACID and schema evolution.
- Add a model registry + scheduled retraining as the training-vehicle-generalization gap
  above suggests drift monitoring matters here specifically.
- Add alert routing to PagerDuty/Slack/webhooks.
- Extend the Spark job with stateful per-vehicle feature tracking to close the
  5-vs-7-feature gap noted above.

## References

- Malhotra et al., "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection," 2016.
- Xu et al., "Anomaly Transformer: Time Series Anomaly Detection with Association
  Discrepancy," ICLR 2022.
- Tuli, Casale, Jennings, "TranAD: Deep Transformer Networks for Anomaly Detection in
  Multivariate Time Series Data," VLDB 2022.
- Audibert et al., "USAD: UnSupervised Anomaly Detection on Multivariate Time Series,"
  KDD 2020.

## Copyright

Copyright (c) 2025-2026 Arvind Yogesh. All rights reserved.

See the COPYRIGHT file for additional terms.
