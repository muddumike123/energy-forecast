"""Central configuration for the leag-energy-forecast pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DUCKDB_PATH = DATA_DIR / "leag.duckdb"

LOOKBACK_DAYS = 730
# SMARD revises recent chunks, so anything this young is re-downloaded on every run.
RECENT_DAYS = 14
# Bright Sky months that ended less than this many days ago are re-downloaded.
WEATHER_REFRESH_DAYS = 3

SMARD_BASE_URL = "https://www.smard.de/app/chart_data"
SMARD_FILTERS = {
    410: "load",
    4359: "residual_load",
    4068: "solar",
    4067: "wind_onshore",
    1225: "wind_offshore",
    4169: "price",
}
SMARD_REGIONS = ["DE", "50Hertz"]
# Day-ahead prices are only published for the DE-LU bidding zone.
FILTER_REGIONS = {4169: ["DE-LU"]}

BRIGHTSKY_URL = "https://api.brightsky.dev/weather"
CITIES = {
    "cottbus": (51.7563, 14.3329),
    "dresden": (51.0504, 13.7373),
    "goerlitz": (51.1527, 14.9852),
    "berlin": (52.5200, 13.4050),
    "hamburg": (53.5511, 9.9937),
    "cologne": (50.9375, 6.9603),
    "munich": (48.1351, 11.5820),
}
# Cities inside the 50Hertz control area, used for regional weather averages.
EASTERN_CITIES = ["cottbus", "dresden", "goerlitz", "berlin"]

# Holiday calendars: label -> holidays.Germany subdivision (None = national).
HOLIDAY_REGIONS = {"DE": None, "BB": "BB", "SN": "SN"}

# --- Training (step 3) ---------------------------------------------------------
PREDICTIONS_DIR = DATA_DIR / "predictions"
TEST_WEEKS = 8
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACT_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_EXPERIMENT = "leag-day-ahead"

# --- Backtest (step 4) ---------------------------------------------------------
BACKTEST_DIR = DATA_DIR / "backtest"
BATTERY_POWER_MW = 100.0
BATTERY_ENERGY_MWH = 200.0
BATTERY_ROUND_TRIP_EFFICIENCY = 0.9
BATTERY_CYCLES_PER_DAY = 1.0
