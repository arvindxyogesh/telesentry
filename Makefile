PYTHON ?= python

.PHONY: install generate generate-train generate-eval train train-deep evaluate evaluate-all report test stream-local all

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

# Single-file convenience target (ad-hoc exploration).
generate:
	$(PYTHON) src/ml/generate_synthetic_dataset.py --output data/raw/synthetic_telemetry.parquet

# Proper held-out benchmark: independent train/eval corpora (different
# seeds), so evaluate_models.py measures generalization rather than
# training-set fit.
generate-train:
	$(PYTHON) src/ml/generate_synthetic_dataset.py --output data/raw/synthetic_telemetry_train.parquet --vehicles 24 --seconds 3600 --hz 5 --seed 42

generate-eval:
	$(PYTHON) src/ml/generate_synthetic_dataset.py --output data/raw/synthetic_telemetry_eval.parquet --vehicles 10 --seconds 1800 --hz 5 --seed 4242

train:
	$(PYTHON) src/ml/train_isolation_forest.py --input data/raw/synthetic_telemetry_train.parquet --model-out models/isolation_forest.joblib --scaler-out models/feature_scaler.joblib --calibration-out models/isolation_forest_calibration.json

train-deep:
	$(PYTHON) src/ml/train_deep_models.py --input data/raw/synthetic_telemetry_train.parquet --scaler models/feature_scaler.joblib --lstm-out models/lstm_autoencoder.pt --transformer-out models/transformer_detector.pt --calibration-out models/deep_model_calibration.json

evaluate:
	$(PYTHON) src/ml/evaluate_batch.py --input data/raw/synthetic_telemetry_eval.parquet --model models/isolation_forest.joblib --scaler models/feature_scaler.joblib --metrics-out data/processed/batch_metrics.json

evaluate-all:
	$(PYTHON) src/ml/evaluate_models.py --input data/raw/synthetic_telemetry_eval.parquet --output-json report/model_comparison.json --output-csv report/model_comparison.csv

report:
	$(PYTHON) src/ml/build_report.py

test:
	$(PYTHON) -m pytest tests/ -v

stream-local:
	$(PYTHON) src/streaming/local_streaming_runner.py --input data/raw/synthetic_telemetry_eval.parquet --model-backend iforest --output data/processed/anomaly_alerts

# Full local pipeline: data -> train (classical + deep) -> evaluate -> report.
all: generate-train generate-eval train train-deep evaluate evaluate-all report
