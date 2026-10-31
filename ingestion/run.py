"""Step 1 CLI: download raw data into data/raw as Parquet.

Usage (from the project root):
    python -m ingestion.run                 # last 2 years, all sources
    python -m ingestion.run --days 14       # quick smoke test
    python -m ingestion.run --only smard    # one source
    python -m ingestion.run --force         # ignore existing files
"""

import argparse
import logging
import sys
from datetime import date, datetime, time, timedelta, timezone

from config import LOOKBACK_DAYS
from ingestion.brightsky import ingest_weather
from ingestion.holidays_ingest import ingest_holidays
from ingestion.smard import ingest_smard

SOURCES = ("smard", "weather", "holidays")
log = logging.getLogger("ingestion")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download raw SMARD, weather and holiday data.")
    parser.add_argument("--days", type=int, default=LOOKBACK_DAYS, help="history to load (default: 730)")
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    parser.add_argument("--only", nargs="+", choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    today = datetime.now(timezone.utc).date()
    start_date: date = today - timedelta(days=args.days)
    start_dt = datetime.combine(start_date, time.min, tzinfo=timezone.utc)

    if "smard" in args.only:
        for key, status in ingest_smard(start_dt, force=args.force).items():
            log.info("SMARD %-24s %s", key, status)
    if "weather" in args.only:
        for city, status in ingest_weather(start_date, today, force=args.force).items():
            log.info("Weather %-12s %s", city, status)
    if "holidays" in args.only:
        rows = ingest_holidays(start_date.year, today.year + 1)
        log.info("Holidays: %d rows", rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
