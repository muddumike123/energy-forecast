PY := .venv/bin/python
DBT := ../.venv/bin/dbt

.PHONY: setup ingest ingest-smoke dbt train train-observed backtest all ui mlflow clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	cd dbt_project && $(DBT) deps

ingest:
	$(PY) -m ingestion.run

ingest-smoke:
	$(PY) -m ingestion.run --days 14

dbt:
	cd dbt_project && $(DBT) build --profiles-dir .

train:
	$(PY) -m training.train

train-observed:
	$(PY) -m training.train --weather observed

backtest:
	$(PY) -m backtest.run

all: ingest dbt train backtest

ui:
	.venv/bin/streamlit run ui/app.py

mlflow:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	rm -rf data/raw data/leag.duckdb data/predictions data/backtest mlflow.db mlruns dbt_project/target dbt_project/dbt_packages
