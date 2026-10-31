"""Bright Sky hourly weather -> one Parquet file per city and calendar month."""

from __future__ import annotations

import calendar
import logging
from datetime import date, datetime, timedelta, timezone
from collections.abc import Iterator

import pandas as pd

from config import BRIGHTSKY_URL, CITIES, RAW_DIR, WEATHER_REFRESH_DAYS
from ingestion.http import get_json
from ingestion.io import write_parquet_atomic

log = logging.getLogger(__name__)

# Bright Sky field -> warehouse column
FIELDS = {
    "temperature": "temperature",
    "wind_speed": "wind_speed",
    "cloud_cover": "cloud_cover",
    "sunshine": "sunshine_min",
    "solar": "solar_kwh_m2",
    "precipitation": "precipitation_mm",
    "relative_humidity": "relative_humidity",
    "pressure_msl": "pressure_msl_hpa",
}


def _month_windows(start: date, end: date) -> Iterator[tuple[date, date]]:
    cursor = start.replace(day=1)
    while cursor <= end:
        last = cursor.replace(day=calendar.monthrange(cursor.year, cursor.month)[1])
        yield max(cursor, start), min(last, end)
        cursor = last + timedelta(days=1)


def _to_frame(records: list[dict], city: str) -> pd.DataFrame:
    df = pd.DataFrame(records)
    out = pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
            if not df.empty
            else pd.Series(dtype="datetime64[ns]"),
            "city": city,
        }
    )
    for source, target in FIELDS.items():
        out[target] = pd.to_numeric(df[source], errors="coerce") if source in df.columns else float("nan")
    out["ingested_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return out


def ingest_weather(start: date, end: date, force: bool = False) -> dict[str, str]:
    """Download hourly weather (UTC) for every city. Completed months are skipped."""
    today = datetime.now(timezone.utc).date()
    last_available = min(end, today - timedelta(days=1))
    summary: dict[str, str] = {}

    for city, (lat, lon) in CITIES.items():
        fetched = skipped = 0
        for w_start, w_end in _month_windows(start, last_available):
            path = RAW_DIR / "weather" / f"city={city}" / f"{w_start:%Y-%m}.parquet"
            complete = w_end < today - timedelta(days=WEATHER_REFRESH_DAYS)
            if path.exists() and not force and complete:
                skipped += 1
                continue

            payload = get_json(
                BRIGHTSKY_URL,
                params={
                    "lat": lat,
                    "lon": lon,
                    "date": w_start.isoformat(),
                    "last_date": w_end.isoformat(),
                    "tz": "UTC",
                },
            )
            if payload is None:
                log.warning("Bright Sky returned 404 for %s %s", city, w_start)
                continue
            write_parquet_atomic(_to_frame(payload.get("weather", []), city), path)
            fetched += 1

        summary[city] = f"fetched={fetched} skipped={skipped}"
    return summary
