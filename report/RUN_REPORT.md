# Vehicle Telemetry Anomaly Detection: Run Report

## Objective
Benchmark classical (Isolation Forest) and deep sequence anomaly detectors (LSTM autoencoder, attention/Transformer detector) on a labeled, multi-type vehicle telemetry benchmark, and serve the best model through a real-time streaming pipeline.

## Dataset
- Held-out evaluation set: `data/raw/synthetic_telemetry_eval.parquet` (46186 rows)
- Anomaly type distribution (evaluation set):
  - normal: 74.11%
  - sensor_drift: 8.72%
  - sensor_stuck: 6.68%
  - gps_spoof: 5.85%
  - harsh_maneuver: 3.19%
  - dropout_burst: 1.44%

## Model Comparison (held-out evaluation set)

| Model | Precision | Recall | F1 | AUROC | AUPRC | FPR | Mean Delay (s) | Latency (ms/1000 rows) |
|---|---|---|---|---|---|---|---|---|
| zscore_baseline | 0.577 | 0.115 | 0.192 | 0.558 | 0.345 | 0.0294 | 0.23 | 0.04 |
| isolation_forest | 0.531 | 0.097 | 0.164 | 0.820 | 0.545 | 0.0300 | 0.51 | 6.74 |
| lstm_autoencoder | 0.537 | 0.522 | 0.530 | 0.734 | 0.462 | 0.1571 | 0.81 | 8.88 |
| transformer_detector | 0.515 | 0.517 | 0.516 | 0.732 | 0.454 | 0.1704 | 0.62 | 18.68 |

**Best F1:** `lstm_autoencoder`

### Recall by Anomaly Type

| Model | dropout_burst | gps_spoof | harsh_maneuver | sensor_drift | sensor_stuck |
|---|---|---|---|---|---|
| zscore_baseline | 0.01 | 0.01 | 0.88 | 0.00 | 0.01 |
| isolation_forest | 0.01 | 0.01 | 0.73 | 0.01 | 0.01 |
| lstm_autoencoder | 0.13 | 0.75 | 1.00 | 0.60 | 0.09 |
| transformer_detector | 0.15 | 0.87 | 1.00 | 0.51 | 0.07 |

## Isolation Forest Batch Metrics (quick-iteration path)
- evaluated_at: 2026-08-27T12:27:16.052385+00:00
- samples: 46186
- precision: 0.5312928277752398
- recall: 0.09727333556373369
- f1: 0.16443973135383527
- false_positive_rate: 0.02997370727432077
- zscore_baseline_false_positive_rate: 0.029447852760736196
- iforest_only_false_positive_rate: 0.02997370727432077
- fpr_reduction_vs_zscore_baseline: -0.017857142857142804
- fusion_strategy: iforest
- mean_detection_delay_sec: 0.5135802469135803
- target_delay_sec: 5.0
- delay_target_met: True

## Streaming Run
- total_alerts: 467
- latency_metric: inference_latency_ms
- mean_latency: 8.254123948611094
- p95_latency: 11.830762199997466
- vehicles_with_alerts: 8

## Figures
- `report/figures/model_comparison_metrics.png`
- `report/figures/model_comparison_per_type_recall.png`
- `report/figures/model_comparison_roc.png`
- `report/figures/alerts_over_time.png`
- `report/figures/detection_delay_hist.png`
