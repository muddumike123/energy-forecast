"""Public holidays for Germany (national), Brandenburg and Saxony."""

import holidays
import pandas as pd

from config import HOLIDAY_REGIONS, RAW_DIR
from ingestion.io import write_parquet_atomic


def ingest_holidays(start_year: int, end_year: int) -> int:
    """Write one Parquet file with every holiday in the range. Returns the row count."""
    rows = []
    for region, subdiv in HOLIDAY_REGIONS.items():
        calendar = holidays.Germany(subdiv=subdiv, years=range(start_year, end_year + 1))
        rows += [
            {"holiday_date": pd.Timestamp(day), "holiday_name": name, "region": region}
            for day, name in calendar.items()
        ]
    df = pd.DataFrame(rows).sort_values(["region", "holiday_date"]).reset_index(drop=True)
    write_parquet_atomic(df, RAW_DIR / "holidays" / "holidays.parquet")
    return len(df)
