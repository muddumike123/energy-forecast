"""Pipeline console for leag-energy-forecast.

Run from the project root:  streamlit run ui/app.py
"""

import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BACKTEST_DIR, DUCKDB_PATH, LOOKBACK_DAYS, PREDICTIONS_DIR, PROJECT_ROOT  # noqa: E402

DBT_DIR = PROJECT_ROOT / "dbt_project"
RUN_RESULTS = DBT_DIR / "target" / "run_results.json"

st.set_page_config(page_title="LEAG Energy Forecast", layout="wide")

# --- Chart theme (validated categorical palette, light + dark steps) --------
DARK = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500"] if DARK else ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GRID = "#2c2c2a" if DARK else "#e1e0d9"
AXIS = "#383835" if DARK else "#c3c2b7"
MUTED = "#898781"
INK = "#f0efec" if DARK else "#0b0b0b"
SURFACE = "#1a1a19" if DARK else "#fcfcfb"
SEQUENTIAL = [SURFACE, "#86b6ef", "#2a78d6", "#104281"] if not DARK else [SURFACE, "#184f95", "#3987e5", "#9ec5f4"]


def styled(fig: go.Figure, title: str, unit: str, height: int = 320) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0, xanchor="left"),
        height=height,
        margin=dict(l=8, r=8, t=56, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=MUTED, size=12),
        hovermode="x unified",
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", title=None, font=dict(color=INK)),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=dict(text=unit, font=dict(color=MUTED)))
    return fig


def line_chart(df: pd.DataFrame, x: str, series: dict[str, str], title: str, unit: str) -> go.Figure:
    """series: column -> display name. Colors follow the dict order (fixed slots)."""
    fig = go.Figure()
    for slot, (column, name) in enumerate(series.items()):
        fig.add_trace(
            go.Scatter(
                x=df[x], y=df[column], name=name, mode="lines",
                line=dict(color=SERIES[slot], width=2),
                hovertemplate=f"%{{y:,.0f}} {unit}",
            )
        )
    fig.update_layout(showlegend=len(series) > 1)
    return styled(fig, title, unit)


# --- Data access -------------------------------------------------------------
def db_version() -> float:
    return DUCKDB_PATH.stat().st_mtime if DUCKDB_PATH.exists() else 0.0


@st.cache_data(show_spinner=False)
def query(sql: str, version: float, params: tuple = ()) -> pd.DataFrame:
    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as con:
        return con.execute(sql, list(params)).df()


def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    return query(sql, db_version(), params)


def warehouse_ready() -> bool:
    if not DUCKDB_PATH.exists():
        return False
    tables = q("select table_name from information_schema.tables")
    return {"hourly", "features", "stg_smard", "stg_weather"} <= set(tables["table_name"])


def run_streaming(cmd: list[str], cwd: Path) -> int:
    """Run a command and stream its output into the page."""
    box = st.empty()
    lines: list[str] = []
    env = {**os.environ, "NO_COLOR": "1", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in proc.stdout:
        lines.append(line.rstrip())
        box.code("\n".join(lines[-300:]), language="text")
    return proc.wait()


WEATHER_MODES = {
    "none": "No weather (leakage-free)",
    "observed": "Observed weather (upper bound, leaks)",
}
TARGET_LABELS = {"load": "Load 50Hertz", "residual_load_de": "Residual load DE", "price": "Price DE-LU"}
TARGET_UNITS = {"load": "MW", "residual_load_de": "MW", "price": "EUR/MWh"}
MODEL_LABELS = {"xgb": "XGBoost", "linear": "Linear regression", "naive": "Same hour last week"}


def pick_weather_mode(base: Path, filename: str, key: str) -> str | None:
    """Radio over the weather modes whose output file exists; None if there are none."""
    modes = [m for m in WEATHER_MODES if (base / f"weather={m}" / filename).exists()]
    if not modes:
        return None
    return st.radio("Training run", modes, format_func=WEATHER_MODES.get, horizontal=True, key=key)


@st.cache_data(show_spinner=False)
def read_parquet(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "local_date" in df.columns:
        df["local_date"] = pd.to_datetime(df["local_date"]).dt.date
    return df


def load_parquet(path: Path) -> pd.DataFrame:
    return read_parquet(str(path), path.stat().st_mtime)


def dbt_results() -> pd.DataFrame | None:
    if not RUN_RESULTS.exists():
        return None
    results = json.loads(RUN_RESULTS.read_text())["results"]
    icons = {"pass": "✅ pass", "success": "✅ success", "fail": "❌ fail", "error": "❌ error", "warn": "⚠️ warn", "skipped": "⏭ skipped"}
    return pd.DataFrame(
        {
            "node": r["unique_id"].split(".", 2)[-1],
            "type": r["unique_id"].split(".")[0],
            "status": icons.get(r["status"], r["status"]),
            "seconds": round(r["execution_time"], 2),
            "message": r.get("message") or "",
        }
        for r in results
    )


# --- Header -------------------------------------------------------------------
st.title("LEAG Energy Forecast")
st.caption("Day-ahead forecasting for the 50Hertz control area · forecasts issued at 12:00 D−1")

ready = warehouse_ready()
if ready:
    stats = q(
        """
        select
            count(*) as hours,
            max(ts_utc) filter (where load_50hz_mw is not null) as last_load,
            max(ts_utc) filter (where price_eur_mwh is not null) as last_price,
            min(ts_utc) as first_ts
        from hourly
        """
    ).iloc[0]
    results = dbt_results()
    tests = results[results["type"] == "test"] if results is not None else pd.DataFrame()
    passed = int(tests["status"].str.contains("pass").sum()) if len(tests) else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hours in warehouse", f"{int(stats['hours']):,}", help=f"Since {stats['first_ts']:%Y-%m-%d %H:%M} UTC")
    c2.metric("Latest load (UTC)", f"{stats['last_load']:%d %b %H:%M}")
    c3.metric("Latest price (UTC)", f"{stats['last_price']:%d %b %H:%M}")
    c4.metric("dbt tests passing", f"{passed} / {len(tests)}" if len(tests) else "–")
else:
    st.info("No warehouse yet. Use the **Pipeline** tab to run ingestion and dbt, then come back.")

tab_pipeline, tab_coverage, tab_explorer, tab_features, tab_forecasts, tab_backtest = st.tabs(
    ["Pipeline", "Data coverage", "Explorer", "Features & quality", "Forecasts", "Battery backtest"]
)

# --- Pipeline -------------------------------------------------------------------
with tab_pipeline:
    left, right = st.columns(2)
    with left:
        st.subheader("1 · Ingestion")
        st.caption("SMARD, Bright Sky and holidays → data/raw/*.parquet. Idempotent: existing files are skipped.")
        days = st.number_input("History (days)", min_value=7, max_value=1500, value=LOOKBACK_DAYS, step=7)
        sources = st.multiselect("Sources", ["smard", "weather", "holidays"], default=["smard", "weather", "holidays"])
        force = st.checkbox("Force re-download")
        if st.button("Run ingestion", type="primary", disabled=not sources):
            cmd = [sys.executable, "-m", "ingestion.run", "--days", str(days), "--only", *sources]
            code = run_streaming(cmd + (["--force"] if force else []), PROJECT_ROOT)
            (st.success if code == 0 else st.error)(f"Ingestion finished with exit code {code}")

    with right:
        st.subheader("2 · dbt build")
        st.caption("Staging → hourly → features in DuckDB, then all dbt tests.")
        if st.button("Run dbt build", type="primary"):
            dbt = str(Path(sys.executable).parent / "dbt")
            code = 0
            if not (DBT_DIR / "dbt_packages").exists():
                code = run_streaming([dbt, "deps", "--profiles-dir", "."], DBT_DIR)
            if code == 0:
                code = run_streaming([dbt, "build", "--profiles-dir", ".", "--no-use-colors"], DBT_DIR)
            query.clear()
            (st.success if code == 0 else st.error)(f"dbt finished with exit code {code}")

    left, right = st.columns(2)
    with left:
        st.subheader("3 · Training")
        st.caption("XGBoost vs linear regression vs same hour last week, last 8 weeks as test. Logged to MLflow.")
        train_mode = st.radio("Weather features", list(WEATHER_MODES), format_func=WEATHER_MODES.get, key="train_mode")
        if st.button("Run training", type="primary"):
            code = run_streaming([sys.executable, "-m", "training.train", "--weather", train_mode], PROJECT_ROOT)
            (st.success if code == 0 else st.error)(f"Training finished with exit code {code}")

    with right:
        st.subheader("4 · Battery backtest")
        st.caption("100 MW / 200 MWh, 90 % round trip, 1 cycle/day, scheduled on each price forecast.")
        backtest_mode = st.radio("Predictions from", list(WEATHER_MODES), format_func=WEATHER_MODES.get, key="backtest_mode")
        if st.button("Run backtest", type="primary"):
            code = run_streaming([sys.executable, "-m", "backtest.run", "--weather", backtest_mode], PROJECT_ROOT)
            (st.success if code == 0 else st.error)(f"Backtest finished with exit code {code}")

    st.subheader("Last dbt run")
    results = dbt_results()
    if results is None:
        st.caption("No dbt run yet.")
    else:
        st.dataframe(results, hide_index=True, width="stretch")

# --- Coverage ------------------------------------------------------------------
with tab_coverage:
    if not ready:
        st.caption("Build the warehouse first.")
    else:
        st.subheader("SMARD series")
        smard = q(
            """
            select
                series, region,
                count(*) as rows,
                min(ts_utc) as first_utc,
                max(ts_utc) filter (where value is not null) as last_value_utc,
                round(100.0 * count(*) filter (
                    where value is null
                      and ts_utc <= (select max(ts_utc) from stg_smard s2
                                     where s2.series = stg_smard.series and s2.region = stg_smard.region
                                       and s2.value is not null)
                ) / count(*), 2) as missing_pct
            from stg_smard
            group by all
            order by series, region
            """
        )
        st.dataframe(smard, hide_index=True, width="stretch")

        st.subheader("Weather (Bright Sky)")
        weather = q(
            """
            select
                city,
                count(*) as rows,
                min(ts_utc) as first_utc,
                max(ts_utc) as last_utc,
                round(100.0 * count(*) filter (where temperature is null) / count(*), 2) as temp_missing_pct,
                round(100.0 * count(*) filter (where solar_kwh_m2 is null) / count(*), 2) as solar_missing_pct
            from stg_weather
            group by city
            order by city
            """
        )
        st.dataframe(weather, hide_index=True, width="stretch")

        gaps = q(
            """
            with observed as (
                select * from hourly
                where ts_utc <= (select max(ts_utc) from hourly where load_50hz_mw is not null)
            )
            select
                cast(ts_local as date) as day,
                count(*) filter (where load_50hz_mw is null) as "Load 50Hertz",
                count(*) filter (where residual_load_50hz_mw is null) as "Residual load 50Hertz",
                count(*) filter (where solar_50hz_mw is null) as "Solar 50Hertz",
                count(*) filter (where wind_onshore_50hz_mw is null) as "Wind onshore 50Hertz",
                count(*) filter (where wind_offshore_50hz_mw is null) as "Wind offshore 50Hertz",
                count(*) filter (where price_eur_mwh is null) as "Price DE-LU"
            from observed
            group by 1
            order by 1
            """
        ).set_index("day")
        heat = go.Figure(
            go.Heatmap(
                x=gaps.index, y=gaps.columns, z=gaps.T.values,
                colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
                zmin=0, zmax=24, xgap=1, ygap=2,
                colorbar=dict(title=dict(text="h missing", font=dict(color=MUTED)), thickness=10),
                hovertemplate="%{y}<br>%{x|%Y-%m-%d}: %{z} missing hours<extra></extra>",
            )
        )
        styled(heat, "Missing hours per local day", "", height=300).update_layout(hovermode="closest")
        st.plotly_chart(heat, width="stretch", theme=None)
        total_missing = gaps.sum()
        st.caption("Total missing hours: " + " · ".join(f"{k}: {int(v)}" for k, v in total_missing.items()))

# --- Explorer ------------------------------------------------------------------
with tab_explorer:
    if not ready:
        st.caption("Build the warehouse first.")
    else:
        first = stats["first_ts"].date()
        last = stats["last_price"].date()
        c1, c2 = st.columns([2, 1])
        picked = c1.date_input(
            "Date range (local)", value=(max(first, last - timedelta(days=28)), last),
            min_value=first, max_value=last,
        )
        region = c2.radio("Region", ["50Hertz", "DE"], horizontal=True)
        start, end = (picked if isinstance(picked, tuple) and len(picked) == 2 else (picked[0], picked[0]))
        sfx = "50hz" if region == "50Hertz" else "de"

        df = q(
            "select * from hourly where cast(ts_local as date) between ? and ? order by ts_utc",
            (start, end),
        )
        st.plotly_chart(
            line_chart(df, "ts_local", {f"load_{sfx}_mw": "Load", f"residual_load_{sfx}_mw": "Residual load"},
                       f"Load and residual load · {region}", "MW"),
            width="stretch", theme=None,
        )
        st.plotly_chart(
            line_chart(df, "ts_local", {f"solar_{sfx}_mw": "Solar", f"wind_onshore_{sfx}_mw": "Wind onshore",
                                        f"wind_offshore_{sfx}_mw": "Wind offshore"},
                       f"Renewable generation · {region}", "MW"),
            width="stretch", theme=None,
        )
        st.plotly_chart(
            line_chart(df, "ts_local", {"price_eur_mwh": "Price"}, "Day-ahead price · DE-LU", "EUR/MWh"),
            width="stretch", theme=None,
        )

        profile = q(
            f"""
            select
                hour_local,
                case when is_weekend or is_holiday_de then 'Weekend / holiday' else 'Working day' end as day_type,
                avg(load_{sfx}_mw) as avg_load
            from features
            where cast(ts_local as date) between ? and ?
            group by all
            order by hour_local
            """,
            (start, end),
        ).pivot(index="hour_local", columns="day_type", values="avg_load").reset_index()
        profile_series = {c: c for c in ["Working day", "Weekend / holiday"] if c in profile.columns}
        fig = line_chart(profile, "hour_local", profile_series, f"Average daily load profile · {region}", "MW")
        fig.update_xaxes(title=dict(text="Local hour", font=dict(color=MUTED)), dtick=3)
        st.plotly_chart(fig, width="stretch", theme=None)

# --- Features & quality ----------------------------------------------------------
with tab_features:
    if not ready:
        st.caption("Build the warehouse first.")
    else:
        st.warning(
            "Weather columns are **observed** values at the target hour. For a leakage-free model "
            "(forecast issued at 12:00 D−1), training must replace them with weather forecasts. "
            "All load/price lags are ≥ 48 h and therefore known at forecast time.",
            icon="⚠️",
        )
        features = q("select * from features order by ts_utc")
        target = st.selectbox(
            "Target", ["load_50hz_mw", "price_eur_mwh", "residual_load_50hz_mw"],
            format_func=lambda c: {"load_50hz_mw": "Load 50Hertz", "price_eur_mwh": "Price DE-LU",
                                   "residual_load_50hz_mw": "Residual load 50Hertz"}[c],
        )
        candidates = [
            c for c in features.columns
            if c.endswith(("_lag_48h", "_lag_168h", "_lag_336h")) or c.startswith("weather_")
            or c in ("hour_local", "day_of_week_local", "month_local")
        ]
        corr = (
            features[candidates + [target]].astype("float64").corr()[target]
            .drop(target).sort_values()
        )
        bars = go.Figure(
            go.Bar(
                x=corr.values, y=corr.index, orientation="h",
                marker=dict(color=SERIES[0], cornerradius=4),
                hovertemplate="%{y}: r = %{x:.2f}<extra></extra>",
            )
        )
        styled(bars, f"Pearson correlation with {target}", "", height=max(320, 26 * len(corr)))
        bars.update_layout(hovermode="closest")
        bars.update_xaxes(range=[-1, 1], showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=AXIS)
        bars.update_yaxes(gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(bars, width="stretch", theme=None)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Null counts")
            nulls = features.isna().sum().rename("nulls").to_frame()
            nulls["pct"] = (100 * nulls["nulls"] / len(features)).round(2)
            st.dataframe(nulls, width="stretch")
        with c2:
            st.subheader("Latest rows")
            st.dataframe(features.tail(48).iloc[::-1], hide_index=True, width="stretch")

# --- Forecasts -------------------------------------------------------------------
with tab_forecasts:
    mode = pick_weather_mode(PREDICTIONS_DIR, "metrics.json", key="forecast_mode")
    if mode is None:
        st.caption("No forecasts yet. Run **3 · Training** in the Pipeline tab.")
    else:
        run_dir = PREDICTIONS_DIR / f"weather={mode}"
        summary = json.loads((run_dir / "metrics.json").read_text())
        preds = load_parquet(run_dir / "test_predictions.parquet")
        if mode == "observed":
            st.warning("This run uses weather measured at the target hour, which is not known at 12:00 D−1. "
                       "Treat it as an upper bound, not a real forecast.", icon="⚠️")
        st.caption(f"Test period {summary['test_start']} → {summary['test_end']} (local days), never seen in training. "
                   "Skill = MAE improvement over the same-hour-last-week baseline.")

        metrics = pd.DataFrame(summary["metrics"])
        st.dataframe(
            pd.DataFrame({
                "Target": metrics["target"].map(TARGET_LABELS),
                "Model": metrics["model"].map(MODEL_LABELS),
                "MAE": metrics["mae"].round(1),
                "RMSE": metrics["rmse"].round(1),
                "Unit": metrics["target"].map(TARGET_UNITS),
                "Skill vs naive (%)": (100 * metrics["skill_vs_naive"]).round(1),
            }),
            hide_index=True, width="stretch",
        )

        c1, c2 = st.columns([1, 2])
        target = c1.selectbox("Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get, key="forecast_target")
        days = sorted(preds["local_date"].unique())
        window = c2.date_input("Days shown", value=(days[-14], days[-1]), min_value=days[0], max_value=days[-1],
                               key="forecast_window")
        lo, hi = window if isinstance(window, tuple) and len(window) == 2 else (window[0], window[0])
        shown = preds[(preds["local_date"] >= lo) & (preds["local_date"] <= hi)]
        series = {f"{target}_actual": "Actual", **{f"{target}_{m}": label for m, label in MODEL_LABELS.items()}}
        st.plotly_chart(
            line_chart(shown, "ts_local", series, f"{TARGET_LABELS[target]}: actual vs forecasts", TARGET_UNITS[target]),
            width="stretch", theme=None,
        )

        importance = pd.Series(summary["importance"].get(target, {}), dtype="float64").sort_values()
        if len(importance):
            bars = go.Figure(go.Bar(
                x=importance.values, y=importance.index, orientation="h",
                marker=dict(color=SERIES[0], cornerradius=4),
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            ))
            styled(bars, f"XGBoost feature importance · {TARGET_LABELS[target]}", "", height=max(300, 24 * len(importance)))
            bars.update_layout(hovermode="closest")
            bars.update_xaxes(showgrid=True, gridcolor=GRID)
            bars.update_yaxes(gridcolor="rgba(0,0,0,0)")
            st.plotly_chart(bars, width="stretch", theme=None)

# --- Battery backtest -----------------------------------------------------------
with tab_backtest:
    mode = pick_weather_mode(BACKTEST_DIR, "summary.json", key="backtest_view_mode")
    if mode is None:
        st.caption("No backtest yet. Run **4 · Battery backtest** in the Pipeline tab.")
    else:
        run_dir = BACKTEST_DIR / f"weather={mode}"
        summary = json.loads((run_dir / "summary.json").read_text())
        daily = load_parquet(run_dir / "daily.parquet")
        schedules = load_parquet(run_dir / "schedules.parquet")
        battery = summary["battery"]
        st.caption(
            f"{battery['power_mw']:.0f} MW / {battery['energy_mwh']:.0f} MWh, "
            f"{battery['round_trip_efficiency']:.0%} round trip, {battery['cycles_per_day']:g} cycle/day · "
            f"{summary['first_day']} → {summary['last_day']}. Each day is scheduled on the price curve known at "
            "12:00 D−1 and settled at the actual day-ahead prices."
        )

        strategies = pd.DataFrame(summary["strategies"]).set_index("strategy")
        labels = strategies["label"].to_dict()
        for col, (key, row) in zip(st.columns(len(strategies)), strategies.iterrows()):
            col.metric(
                row["label"], f"€{row['revenue_eur'] / 1e6:,.2f} M",
                None if key == "perfect" else f"{row['capture_pct']:.1f} % of perfect", delta_color="off",
            )

        cumulative = daily.pivot(index="local_date", columns="strategy", values="revenue_eur").cumsum().reset_index()
        st.plotly_chart(
            line_chart(cumulative, "local_date", {k: labels[k] for k in strategies.index}, "Cumulative revenue", "EUR"),
            width="stretch", theme=None,
        )

        c1, c2 = st.columns(2)
        day = c1.selectbox("Day", sorted(daily["local_date"].unique(), reverse=True),
                           format_func=lambda d: f"{d:%a %d %b %Y}")
        strategy = c2.selectbox("Compare with perfect foresight", [k for k in strategies.index if k != "perfect"],
                                format_func=labels.get)
        that_day = schedules[schedules["local_date"] == day]
        perfect_rows = that_day[that_day["strategy"] == "perfect"]
        chosen_rows = that_day[that_day["strategy"] == strategy]

        revenue = daily[daily["local_date"] == day].set_index("strategy")["revenue_eur"]
        m1, m2 = st.columns(2)
        m1.metric("Perfect foresight revenue", f"€{revenue['perfect']:,.0f}")
        m2.metric(f"{labels[strategy]} revenue", f"€{revenue[strategy]:,.0f}",
                  f"{revenue[strategy] - revenue['perfect']:,.0f} € vs perfect", delta_color="off")

        prices = pd.DataFrame({
            "ts_local": chosen_rows["ts_local"].to_numpy(),
            "actual": chosen_rows["price_actual"].to_numpy(),
            "planned": chosen_rows["price_planned"].to_numpy(),
        })
        st.plotly_chart(
            line_chart(prices, "ts_local", {"actual": "Actual price", "planned": f"{labels[strategy]} (scheduled on)"},
                       f"Prices on {day:%d %b %Y}", "EUR/MWh"),
            width="stretch", theme=None,
        )

        schedule = go.Figure()
        for slot, (name, rows) in enumerate([("Perfect foresight", perfect_rows), (labels[strategy], chosen_rows)]):
            schedule.add_trace(go.Bar(
                x=rows["ts_local"], y=rows["discharge_mw"] - rows["charge_mw"], name=name,
                marker=dict(color=SERIES[slot], cornerradius=4, line=dict(color=SURFACE, width=2)),
                hovertemplate="%{y:,.0f} MW",
            ))
        styled(schedule, "Battery schedule (positive = discharge / sell, negative = charge / buy)", "MW")
        schedule.update_layout(barmode="group", bargap=0.25)
        st.plotly_chart(schedule, width="stretch", theme=None)

        with st.expander("Daily revenue table"):
            table = daily.pivot(index="local_date", columns="strategy", values="revenue_eur")
            st.dataframe(table.rename(columns=labels).round(0).iloc[::-1], width="stretch")
