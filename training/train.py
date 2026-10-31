"""Step 3 CLI: train day-ahead models and compare them with baselines.

Usage (from the project root):
    python -m training.train                    # strict: no weather features
    python -m training.train --weather observed # upper bound: uses observed weather (leaks!)

Every forecast is what could be issued at 12:00 on D-1: calendar features,
lags available at that time, and (for price) a residual-load forecast.

Order of models:
  1. load_50hz_mw          XGBoost vs linear regression vs same hour last week
  2. residual_load_de_mw   same; its forecast feeds the price model
  3. price_eur_mwh         same, with the residual-load forecast as a feature
"""

import argparse
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd

from config import PREDICTIONS_DIR, TEST_WEEKS
from training.data import load_features, time_split
from training.models import CALENDAR, WEATHER, evaluate, fit_xgb, make_linear, out_of_fold
from training.tracking import setup_mlflow

log = logging.getLogger("training")
RESIDUAL_FORECAST = "residual_load_de_forecast"


@dataclass(frozen=True)
class TargetSpec:
    name: str
    target: str
    naive: str  # column holding the value from the same hour last week
    features: list[str]


SPECS = [
    TargetSpec(
        name="load",
        target="load_50hz_mw",
        naive="load_lag_168h",
        features=CALENDAR + ["load_lag_48h", "load_lag_168h", "load_lag_336h"],
    ),
    TargetSpec(
        name="residual_load_de",
        target="residual_load_de_mw",
        naive="residual_load_de_lag_168h",
        features=CALENDAR + ["residual_load_de_lag_48h", "residual_load_de_lag_168h", "residual_load_de_lag_336h"],
    ),
    TargetSpec(
        name="price",
        target="price_eur_mwh",
        naive="price_lag_168h",
        features=CALENDAR
        + [
            "price_lag_24h",
            "price_lag_48h",
            "price_lag_168h",
            "price_lag_336h",
            "residual_load_de_lag_48h",
            "residual_load_de_lag_168h",
            RESIDUAL_FORECAST,
        ],
    ),
]


def output_dir(weather: str) -> Path:
    return PREDICTIONS_DIR / f"weather={weather}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train day-ahead load and price models.")
    parser.add_argument(
        "--weather", choices=["none", "observed"], default="none",
        help="'observed' adds actual weather at the target hour: an upper bound, not a valid forecast",
    )
    parser.add_argument("--test-weeks", type=int, default=TEST_WEEKS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    split = time_split(load_features(), args.test_weeks)
    train, test = split.train.copy(), split.test.copy()
    log.info("Train: %d rows until %s · Test: %s → %s (%d rows)",
             len(train), split.test_start, split.test_start, split.test_end, len(test))

    setup_mlflow()
    predictions = test[["ts_utc", "ts_local", "local_date"]].copy()
    metrics: list[dict] = []
    importance: dict[str, dict[str, float]] = {}

    with mlflow.start_run(run_name=f"day-ahead weather={args.weather}"):
        mlflow.set_tags({"weather": args.weather, "leakage_free": str(args.weather == "none")})
        mlflow.log_params({
            "test_start": split.test_start, "test_end": split.test_end,
            "train_rows": len(train), "test_rows": len(test), "issue_time": "12:00 D-1",
        })

        for spec in SPECS:
            features = spec.features + (WEATHER if args.weather == "observed" else [])
            required = [f for f in features if f not in WEATHER] + [spec.target]
            rows = train.dropna(subset=required)

            xgb = fit_xgb(rows, features, spec.target)
            linear = make_linear(features).fit(rows[features], rows[spec.target])
            forecasts = {
                "xgb": xgb.predict(test[features]),
                "linear": linear.predict(test[features]),
                "naive": test[spec.naive].to_numpy(),
            }

            predictions[f"{spec.name}_actual"] = test[spec.target]
            for model_name, values in forecasts.items():
                predictions[f"{spec.name}_{model_name}"] = values

            # Score all models on the same rows.
            columns = [f"{spec.name}_{m}" for m in ["actual", *forecasts]]
            scored = predictions.dropna(subset=columns)
            naive_mae = evaluate(scored[f"{spec.name}_actual"], scored[f"{spec.name}_naive"])["mae"]
            for model_name in forecasts:
                result = evaluate(scored[f"{spec.name}_actual"], scored[f"{spec.name}_{model_name}"])
                result["skill_vs_naive"] = 1 - result["mae"] / naive_mae
                metrics.append({"target": spec.name, "model": model_name, **result, "rows": len(scored)})
                with mlflow.start_run(run_name=f"{spec.name}/{model_name}", nested=True):
                    mlflow.set_tags({"target": spec.target, "model": model_name, "weather": args.weather})
                    mlflow.log_params({"features": ",".join(features), "train_rows": len(rows)})
                    mlflow.log_metrics(result)
                    if model_name == "xgb":
                        mlflow.log_params({"n_estimators": xgb.n_estimators, **{
                            k: v for k, v in xgb.get_params().items()
                            if k in ("learning_rate", "max_depth", "min_child_weight", "subsample", "colsample_bytree")
                        }})
                        importance[spec.name] = {f: float(v) for f, v in zip(features, xgb.feature_importances_)}
                        mlflow.log_dict(importance[spec.name], "feature_importance.json")
                        with tempfile.TemporaryDirectory() as tmp:
                            path = Path(tmp) / "model.json"
                            xgb.save_model(path)
                            mlflow.log_artifact(str(path))
            log.info("%-17s MAE xgb=%.1f linear=%.1f naive=%.1f", spec.name,
                     *[m["mae"] for m in metrics[-3:]])

            if spec.name == "residual_load_de":
                # Price model input: out-of-fold forecasts on train, real forecasts on test.
                train[RESIDUAL_FORECAST] = out_of_fold(xgb, rows, features, spec.target).reindex(train.index)
                test[RESIDUAL_FORECAST] = forecasts["xgb"]

        out = output_dir(args.weather)
        out.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(out / "test_predictions.parquet", index=False)
        summary = {
            "weather": args.weather,
            "test_start": str(split.test_start),
            "test_end": str(split.test_end),
            "metrics": metrics,
            "importance": importance,
        }
        (out / "metrics.json").write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(out / "test_predictions.parquet"))
        mlflow.log_artifact(str(out / "metrics.json"))

    table = pd.DataFrame(metrics).pivot(index="target", columns="model", values="mae").round(1)
    print(f"\nTest MAE ({split.test_start} → {split.test_end}, weather={args.weather}):\n{table}\n")
    print(f"Predictions: {out / 'test_predictions.parquet'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
