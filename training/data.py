"""Load the feature table and build the time-based train/test split."""

from dataclasses import dataclass
from datetime import date, timedelta

import duckdb
import pandas as pd

from config import DUCKDB_PATH, TEST_WEEKS

TARGETS = ["load_50hz_mw", "residual_load_de_mw", "price_eur_mwh"]


def load_features() -> pd.DataFrame:
    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as con:
        df = con.execute("select * from features order by ts_utc").df()
    df["local_date"] = df["ts_local"].dt.date
    bool_columns = df.select_dtypes("bool").columns
    df[bool_columns] = df[bool_columns].astype("int8")
    return df


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    test_start: date
    test_end: date


def time_split(df: pd.DataFrame, test_weeks: int = TEST_WEEKS) -> Split:
    """Test = the last `test_weeks` local days on which every target is complete.

    Train = every earlier row with all targets present. Nothing after the test
    window is used, and nothing from the test window leaks into training.
    """
    hours_per_day = df.groupby("local_date").size()
    observed_per_day = df.groupby("local_date")[TARGETS].count().min(axis=1)
    complete_days = hours_per_day.index[observed_per_day == hours_per_day]

    test_end = max(complete_days)
    test_start = test_end - timedelta(days=7 * test_weeks - 1)
    in_test = (df["local_date"] >= test_start) & (df["local_date"] <= test_end)
    train = df[df["local_date"] < test_start].dropna(subset=TARGETS)
    return Split(train.reset_index(drop=True), df[in_test].reset_index(drop=True), test_start, test_end)
