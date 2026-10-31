"""MLflow setup shared by training and backtest."""

import mlflow

from config import MLFLOW_ARTIFACT_DIR, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACT_DIR.as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
