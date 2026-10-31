"""SMARD hourly series -> raw Parquet, one file per SMARD chunk."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from config import (
    FILTER_REGIONS,
    RAW_DIR,
    RECENT_DAYS,
    SMARD_BASE_URL,
    SMARD_FILTERS,
    SMARD_REGIONS,
)
from ingestion.http import get_json
from ingestion.io import write_parquet_atomic

log = logging.getLogger(__name__)
MS_PER_DAY = 86_400_000


def _select_chunks(timestamps: list[int], start_ms: int) -> list[int]:
    """The chunk covering start_ms plus every chunk after it."""
    ordered = sorted(timestamps)
    covering = [t for t in ordered if t <= start_ms][-1:]
    later = [t for t in ordered if t > start_ms]
    return covering + later


def _to_frame(rows: list, series: str, filter_id: int, region: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts_ms", "value"])
    return pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(df["ts_ms"], unit="ms"),  # naive UTC
            "series": series,
            "filter_id": filter_id,
            "region": region,
            "value": df["value"].astype("float64"),
            "ingested_at": pd.Timestamp.now(tz="UTC").tz_localize(None),
        }
    )


def ingest_smard(start: datetime, force: bool = False) -> dict[str, str]:
    """Download every configured filter/region from `start` onwards.

    Idempotent: a chunk already on disk is skipped unless it is still being
    revised by SMARD (recent or the newest chunk) or `force` is set.
    """
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    recent_ms = now_ms - RECENT_DAYS * MS_PER_DAY
    start_ms = int(start.timestamp() * 1000)
    summary: dict[str, str] = {}

    for filter_id, series in SMARD_FILTERS.items():
        for region in FILTER_REGIONS.get(filter_id, SMARD_REGIONS):
            key = f"{series}/{region}"
            index = get_json(f"{SMARD_BASE_URL}/{filter_id}/{region}/index_hour.json")
            if index is None:
                log.warning("SMARD has no index for %s; skipping", key)
                summary[key] = "unavailable"
                continue

            timestamps = index["timestamps"]
            newest = max(timestamps)
            fetched = skipped = 0
            for ts in _select_chunks(timestamps, start_ms):
                path = RAW_DIR / "smard" / f"filter={filter_id}" / f"region={region}" / f"{ts}.parquet"
                still_revised = ts >= recent_ms or ts == newest
                if path.exists() and not force and not still_revised:
                    skipped += 1
                    continue

                url = f"{SMARD_BASE_URL}/{filter_id}/{region}/{filter_id}_{region}_hour_{ts}.json"
                payload = get_json(url)
                if payload is None:
                    log.warning("SMARD chunk missing: %s", url)
                    continue
                write_parquet_atomic(_to_frame(payload["series"], series, filter_id, region), path)
                fetched += 1

            summary[key] = f"fetched={fetched} skipped={skipped}"
    return summary
