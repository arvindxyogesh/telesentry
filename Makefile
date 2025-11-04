PYTHON ?= python

.PHONY: install generate train evaluate

install:
	$(PYTHON) -m pip install -r requirements.txt

generate:
	$(PYTHON) src/ml/generate_synthetic_dataset.py --output data/raw/synthetic_telemetry.parquet

train:
	$(PYTHON) src/ml/train_isolation_forest.py --input data/raw/synthetic_telemetry.parquet --model-out models/isolation_forest.joblib --scaler-out models/feature_scaler.joblib

evaluate:
	$(PYTHON) src/ml/evaluate_batch.py --input data/raw/synthetic_telemetry.parquet --model models/isolation_forest.joblib --scaler models/feature_scaler.joblib --metrics-out data/processed/batch_metrics.json
