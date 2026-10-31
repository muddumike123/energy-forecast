# leag-energy-forecast

Day-ahead forecasting of load and price for the German **50Hertz** control area, and a
battery-arbitrage backtest built on those forecasts.

| Step | Folder | What it does |
|---|---|---|
| 1. Ingestion | `ingestion/` | SMARD, Bright Sky and holidays → raw Parquet (idempotent) |
| 2. Warehouse | `dbt_project/` | dbt + DuckDB: staging → hourly → features, with tests |
| 3. Training | `training/` | XGBoost for load, then price (using a residual-load forecast), compared with baselines and logged to MLflow |
| 4. Backtest | `backtest/` | 100 MW / 200 MWh battery: revenue on forecasts vs perfect foresight |
| UI | `ui/` | Streamlit console: run every step, inspect data, forecasts and backtest |

## Architecture

```
SMARD API ────┐
Bright Sky ───┼─► ingestion/ ─► data/raw/*.parquet ─► dbt + DuckDB ─► features ─► training/ ─► data/predictions/ ─► backtest/ ─► data/backtest/
holidays ─────┘                                                                    │                                      │
                                                                                   └──────── MLflow (mlflow.db, mlruns/) ◄┘
                                     ui/app.py (Streamlit) runs and visualises every stage
```

### 1. Raw layer (`data/raw/`)
- `smard/filter=<id>/region=<DE|50Hertz|DE-LU>/<chunk_ts>.parquet`: one file per SMARD weekly chunk.
  Filters: 410 load, 4359 residual load, 4068 solar, 4067 wind onshore, 1225 wind offshore,
  4169 day-ahead price. The price is only published for the **DE-LU** bidding zone.
- `weather/city=<city>/<YYYY-MM>.parquet`: hourly Bright Sky data (UTC) for Cottbus, Dresden,
  Görlitz, Berlin, Hamburg, Cologne and Munich.
- `holidays/holidays.parquet`: national (DE), Brandenburg (BB) and Saxony (SN) holidays.

**Idempotency.** Files that already exist are skipped. SMARD chunks younger than 14 days (plus the
newest chunk) and weather months that ended less than 3 days ago are re-downloaded, because the
sources still revise them. Writes are atomic (temp file + rename). Staging deduplicates on the
natural key.

### 2. Warehouse (`dbt_project/` → `data/leag.duckdb`)
- `stg_smard`, `stg_weather`, `stg_holidays`: typed and deduplicated.
- `hourly`: a continuous **UTC** hourly spine with the series as columns. DST never creates gaps
  or duplicates. `ts_local` (Europe/Berlin) is a derived attribute, never a key.
- `features`: local calendar, holiday flags, a bridge-day proxy, lags and city-weather averages.

The tests check:
- keys are unique and not null;
- values are in their accepted ranges;
- the spine is complete;
- DST days have 23/25 hours, and only on the last Sunday of March/October;
- lag columns equal the source value exactly 24/48/168 h earlier.

### 3. Training (`training/`)

**Forecast issue time: 12:00 on D−1**, for hours 00–23 of D (a horizon of up to 36 h). Only
information available then is used:

| Feature | Why it's known at 12:00 D−1 |
|---|---|
| Calendar and holidays | deterministic |
| Load / residual-load lags of ≥ 48 h | D−2 values are published by then |
| Price lag of 24 h | D−1 prices cleared in the auction at ~12:45 on D−2 |
| Residual-load **forecast** (price model only) | produced by our own model |

The pipeline runs in this order:
1. **Load 50Hertz**: XGBoost on calendar features and load lags.
2. **Residual load DE**: same approach. The price is set for the whole DE-LU market, so the
   Germany-wide residual load is the driver.
3. **Price DE-LU**: XGBoost on calendar features, price lags, residual-load lags and the
   residual-load forecast. During training, the residual-load forecast comes from out-of-fold
   (expanding-window) predictions, so the price model never learns from in-sample-perfect inputs.

**Split.** The test set is the last 8 weeks (56 complete local days). Training uses everything
before that. XGBoost picks its tree count by early stopping on the last 4 weeks of
train, then refits on all of train.

**Baselines.** "Same hour last week" (the 168 h lag) and linear regression (one-hot calendar
features plus the same numeric features). Metrics are MAE and RMSE on identical rows, plus
*skill* = 1 − MAE / MAE(naive).

**Weather.** `--weather none` (the default) is leakage-free. `--weather observed` adds the
**measured** weather at the target hour. That isn't known at 12:00 D−1, so the run is tagged in
MLflow as an upper bound, not a real forecast. A leakage-free version with weather needs
archived weather *forecasts*, e.g. the Open-Meteo historical-forecast API.

### 4. Backtest (`backtest/`)
A 100 MW / 200 MWh battery with 90 % round-trip efficiency (losses split evenly between charge and
discharge) and at most 1 cycle per day. It starts and ends each day empty and never charges and
discharges in the same hour. Each test day is solved as a small MILP (`scipy.optimize.milp`):

1. Optimise the schedule on a price curve known at 12:00 D−1: the XGBoost forecast, the linear
   forecast, or "same hour last week".
2. Settle that schedule at the **actual** day-ahead prices.
3. Compare with **perfect foresight** (optimised on actual prices), the revenue ceiling.

### Results on the current data (test window 2026-09-04 → 2026-10-29)

Leakage-free run (`--weather none`), test MAE:

| Target | XGBoost | Linear | Same hour last week |
|---|---|---|---|
| Load 50Hertz (MW) | **556.5** | 565.2 | 633.7 |
| Residual load DE (MW) | **9 728** | 10 061 | 14 713 |
| Price DE-LU (EUR/MWh) | 47.5 | **45.2** | 68.3 |

Battery revenue over 56 days:

| Strategy | Revenue | % of perfect |
|---|---|---|
| Perfect foresight | €1.79 M | 100 % |
| Linear regression forecast | €1.68 M | 93.9 % |
| Same hour last week | €1.63 M | 91.4 % |
| XGBoost forecast | €1.55 M | 86.7 % |

How to read these:
- **A lower MAE doesn't guarantee more battery revenue.** Arbitrage only needs the cheapest and
  most expensive hours *ranked* correctly. XGBoost's lower price error does not translate into
  better timing here.
- **Without weather, residual load is hard to forecast,** because wind and solar drive it. With
  observed weather (the upper bound) the price MAE drops to ~38 EUR/MWh. Adding weather forecasts
  is the most valuable next improvement.
- **One 8-week test window is a noisy estimate.** A rolling-origin evaluation over several
  windows would give more reliable rankings.

## Setup

Requires Python 3.11+. On macOS, XGBoost also needs the OpenMP runtime: `brew install libomp`.

```bash
cd leag-energy-forecast
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd dbt_project && dbt deps && cd ..
```

Always run the commands below with the venv active (`source .venv/bin/activate`). The
system-wide `python`/`streamlit` won't find the project's packages.

## Running each step

All commands run from the project root.

```bash
# 1. Ingestion
python -m ingestion.run --days 14         # quick smoke test (~10 s)
python -m ingestion.run                   # full 2 years (~5 min first time; re-runs are cheap)

# 2. dbt + DuckDB
cd dbt_project && dbt build --profiles-dir . && cd ..

# 3. Training (~10 s)
python -m training.train                  # leakage-free
python -m training.train --weather observed   # optional upper bound

# 4. Backtest (~5 s)
python -m backtest.run                    # uses the leakage-free predictions
python -m backtest.run --weather observed

# UI and experiment tracking
streamlit run ui/app.py
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://127.0.0.1:5000
```

Make shortcuts: `make all` (steps 1–4), `make ingest`, `make dbt`, `make train`, `make backtest`,
`make ui`, `make mlflow`.

**Outputs**
- `data/predictions/weather=<mode>/`: `test_predictions.parquet` and `metrics.json`
- `data/backtest/weather=<mode>/`: `daily.parquet`, `schedules.parquet` and `summary.json`
- MLflow experiment `leag-day-ahead`. Each training run is a parent run with one nested run per
  target × model (params, MAE/RMSE/skill, XGBoost model and feature importance). Backtests log
  revenue and capture rate.

**UI tabs**

| Tab | Shows |
|---|---|
| Pipeline | run steps 1–4 from the browser with live logs, plus the last dbt results |
| Data coverage | rows, ranges and missing hours per series and city |
| Explorer | load, renewables and price over any date range, plus the daily load profile |
| Features & quality | feature correlations, null counts, latest rows |
| Forecasts | metrics table, actual-vs-forecast chart, XGBoost feature importance |
| Battery backtest | revenue per strategy, cumulative revenue, and per-day prices and schedules |
