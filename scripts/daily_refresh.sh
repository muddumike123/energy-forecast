#!/usr/bin/env bash
# Daily refresh: ingest -> dbt build (incl. data tests) -> train -> backtest.
# Stops at the first failing step, so a failed data test blocks the models.
# Scheduled by cron at 08:00 (see README). Logs go to logs/daily_YYYY-MM-DD.log.
set -euo pipefail

# cron starts with an almost empty PATH; MLflow needs git and Homebrew tools may be needed.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
export GIT_PYTHON_REFRESH=quiet

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
DBT="$ROOT/.venv/bin/dbt"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
exec >>"$LOG_DIR/daily_$(date +%Y-%m-%d).log" 2>&1

echo "=== $(date '+%F %T') daily refresh start"
cd "$ROOT"

echo "--- 1/4 ingestion"
"$PY" -m ingestion.run

echo "--- 2/4 dbt build (models + data tests)"
(cd "$ROOT/dbt_project" && "$DBT" build --profiles-dir . --no-use-colors)

echo "--- 3/4 training"
"$PY" -m training.train --weather none

echo "--- 4/4 backtest"
"$PY" -m backtest.run --weather none

echo "=== $(date '+%F %T') daily refresh done"
