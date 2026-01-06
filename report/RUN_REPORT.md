# Vehicle Telemetry Anomaly Detection Report

## Objective
Build and validate a production-style real-time anomaly detection pipeline for vehicle telemetry.

## Dataset
- External dataset downloaded: FordA (UCR archive) into `data/raw/forda/`
- Streaming simulation dataset: synthetic CAN/IMU/GPS telemetry generator

## Batch Metrics
- evaluated_at: 2026-04-25T16:39:25.180925Z
- samples: 288000
- precision: 0.7271564752383167
- recall: 0.11094359702021994
- f1: 0.19251485026622758
- false_positive_rate: 0.010132976426906139
- zscore_baseline_false_positive_rate: 0.05825058285122183
- iforest_only_false_positive_rate: 0.010132976426906139
- fpr_reduction_vs_zscore_baseline: 0.8260450637414765
- fusion_strategy: iforest
- mean_detection_delay_sec: 1.4283085633404105
- target_delay_sec: 5.0
- delay_target_met: True

## Streaming Metrics
- total_alerts: 8152
- mean_detection_delay_sec: 1.0
- p95_detection_delay_sec: 1.0
- vehicles_with_alerts: 24

## Targets
- Detection delay target (<5 sec): PASS
- False positive reduction vs z-score baseline: 82.60%

## Figures
- `report/figures/alerts_over_time.png`
- `report/figures/detection_delay_hist.png`
