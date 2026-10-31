"""Model factories and metrics."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

CATEGORICAL = ["hour_local", "day_of_week_local", "month_local"]
FLAGS = ["is_weekend", "is_holiday_de", "is_holiday_bb", "is_holiday_sn", "is_bridge_day"]
CALENDAR = CATEGORICAL + FLAGS
WEATHER = [
    "weather_temp_avg_c",
    "weather_wind_avg_kmh",
    "weather_solar_avg_kwh_m2",
    "weather_cloud_avg_pct",
    "weather_temp_east_c",
    "weather_wind_east_kmh",
    "weather_solar_east_kwh_m2",
]

XGB_PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="mae",
    early_stopping_rounds=100,
    random_state=42,
)
VALIDATION_WEEKS = 4


def make_xgb(**overrides) -> XGBRegressor:
    return XGBRegressor(**{**XGB_PARAMS, **overrides})


def fit_xgb(train: pd.DataFrame, features: list[str], target: str) -> XGBRegressor:
    """Pick the tree count by early stopping on the last weeks of train, then refit on all of train."""
    cutoff = train["ts_utc"].max() - pd.Timedelta(weeks=VALIDATION_WEEKS)
    fit, val = train[train["ts_utc"] <= cutoff], train[train["ts_utc"] > cutoff]
    probe = make_xgb().fit(fit[features], fit[target], eval_set=[(val[features], val[target])], verbose=False)
    final = make_xgb(n_estimators=probe.best_iteration + 1, early_stopping_rounds=None)
    return final.fit(train[features], train[target])


def out_of_fold(model: XGBRegressor, train: pd.DataFrame, features: list[str], target: str, n_splits: int = 5) -> pd.Series:
    """Expanding-window predictions for train rows, each from a model that never saw them.

    The first fold has no earlier data and stays NaN.
    """
    oof = pd.Series(np.nan, index=train.index)
    params = {**model.get_params(), "early_stopping_rounds": None}
    for fit_idx, pred_idx in TimeSeriesSplit(n_splits=n_splits).split(train):
        fold = XGBRegressor(**params).fit(train.iloc[fit_idx][features], train.iloc[fit_idx][target])
        oof.iloc[pred_idx] = fold.predict(train.iloc[pred_idx][features])
    return oof


def make_linear(features: list[str]) -> Pipeline:
    """Linear regression with one-hot calendar columns and median-imputed numeric columns."""
    categorical = [f for f in features if f in CATEGORICAL]
    numeric = [f for f in features if f not in CATEGORICAL]
    preprocess = ColumnTransformer(
        [
            ("calendar", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ]
    )
    return make_pipeline(preprocess, LinearRegression())


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2)))}
