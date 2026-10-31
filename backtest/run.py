"""Step 4 CLI: battery arbitrage backtest on the test period.

Usage (from the project root, after training):
    python -m backtest.run
    python -m backtest.run --weather observed

For each test day the battery schedule is optimised on a price curve available
at 12:00 D-1 (forecast or naive) and then settled at the actual day-ahead
prices. Perfect foresight optimises on the actual prices and is the ceiling.
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict

import mlflow
import pandas as pd

from backtest.battery import Battery, optimise_day
from config import BACKTEST_DIR
from training.tracking import setup_mlflow
from training.train import output_dir

log = logging.getLogger("backtest")

# key -> (label, price column used for scheduling)
STRATEGIES = {
    "perfect": ("Perfect foresight", "price_actual"),
    "xgb": ("XGBoost forecast", "price_xgb"),
    "linear": ("Linear regression forecast", "price_linear"),
    "naive": ("Same hour last week", "price_naive"),
}


def run_backtest(predictions: pd.DataFrame, battery: Battery) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_columns = [column for _, column in STRATEGIES.values()]
    daily, schedules = [], []
    for day, hours in predictions.sort_values("ts_utc").groupby("local_date"):
        if hours[price_columns].isna().any().any():
            log.warning("Skipping %s: incomplete prices", day)
            continue
        actual = hours["price_actual"].to_numpy()
        for key, (label, column) in STRATEGIES.items():
            planned = hours[column].to_numpy()
            charge, discharge = optimise_day(planned, battery)
            net = discharge - charge
            daily.append({
                "local_date": day,
                "strategy": key,
                "label": label,
                "revenue_eur": float(actual @ net),
                "planned_revenue_eur": float(planned @ net),
                "charged_mwh": float(charge.sum()),
                "discharged_mwh": float(discharge.sum()),
            })
            schedules.append(pd.DataFrame({
                "ts_utc": hours["ts_utc"].to_numpy(),
                "ts_local": hours["ts_local"].to_numpy(),
                "local_date": day,
                "strategy": key,
                "price_planned": planned,
                "price_actual": actual,
                "charge_mw": charge,
                "discharge_mw": discharge,
            }))
    return pd.DataFrame(daily), pd.concat(schedules, ignore_index=True)


def summarise(daily: pd.DataFrame) -> pd.DataFrame:
    summary = daily.groupby(["strategy", "label"], sort=False).agg(
        days=("local_date", "nunique"),
        revenue_eur=("revenue_eur", "sum"),
        avg_daily_eur=("revenue_eur", "mean"),
        discharged_mwh=("discharged_mwh", "sum"),
    ).reset_index()
    perfect = summary.loc[summary["strategy"] == "perfect", "revenue_eur"].iloc[0]
    summary["capture_pct"] = 100 * summary["revenue_eur"] / perfect
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Battery arbitrage backtest.")
    parser.add_argument("--weather", choices=["none", "observed"], default="none",
                        help="which training run's predictions to use")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    source = output_dir(args.weather) / "test_predictions.parquet"
    if not source.exists():
        log.error("No predictions at %s. Run `python -m training.train --weather %s` first.", source, args.weather)
        return 1

    battery = Battery()
    daily, schedules = run_backtest(pd.read_parquet(source), battery)
    summary = summarise(daily)

    out = BACKTEST_DIR / f"weather={args.weather}"
    out.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out / "daily.parquet", index=False)
    schedules.to_parquet(out / "schedules.parquet", index=False)
    (out / "summary.json").write_text(json.dumps({
        "weather": args.weather,
        "battery": asdict(battery),
        "first_day": str(daily["local_date"].min()),
        "last_day": str(daily["local_date"].max()),
        "strategies": summary.to_dict(orient="records"),
    }, indent=2))

    setup_mlflow()
    with mlflow.start_run(run_name=f"backtest weather={args.weather}"):
        mlflow.set_tags({"stage": "backtest", "weather": args.weather})
        mlflow.log_params(asdict(battery) | {"days": int(summary["days"].max())})
        for row in summary.itertuples():
            mlflow.log_metrics({f"revenue_eur_{row.strategy}": row.revenue_eur, f"capture_pct_{row.strategy}": row.capture_pct})
        mlflow.log_artifact(str(out / "summary.json"))

    printable = summary[["label", "days", "revenue_eur", "avg_daily_eur", "capture_pct"]].round(1)
    print(f"\nBattery {battery.power_mw:.0f} MW / {battery.energy_mwh:.0f} MWh, "
          f"{battery.round_trip_efficiency:.0%} round trip, {battery.cycles_per_day:g} cycle/day\n")
    print(printable.to_string(index=False))
    print(f"\nResults: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
