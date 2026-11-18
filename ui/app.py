"""Pipeline console for leag-energy-forecast.

Run from the project root:  streamlit run ui/app.py
"""

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BACKTEST_DIR, DUCKDB_PATH, LOOKBACK_DAYS, PREDICTIONS_DIR, PROJECT_ROOT  # noqa: E402

DBT_DIR = PROJECT_ROOT / "dbt_project"
RUN_RESULTS = DBT_DIR / "target" / "run_results.json"

st.set_page_config(page_title="LEAG Energy Forecast", page_icon=":material/bolt:", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 600; }
      div[data-testid="stMetricLabel"] p { font-size: 0.85rem; opacity: 0.7; }
      h1 { font-weight: 650; letter-spacing: -0.02em; }
      section[data-testid="stSidebar"] { border-right: 1px solid rgba(128, 128, 128, 0.15); }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Theme ------------------------------------------------------------------------
DARK = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"

if DARK:
    INK, MUTED, GRID, AXIS, SURFACE = "#f3f4f6", "#9ca3af", "#262624", "#3a3a37", "#1c1c1b"
    ACTUAL = "#f3f4f6"
    MODEL_COLOR = {"xgb": "#4c9bea", "linear": "#f08a5d", "naive": "#8b929c"}
    SOURCE_COLOR = {"solar": "#f2b134", "wind_onshore": "#4c9bea", "wind_offshore": "#2fbf8a"}
    SELL, BUY = "#2fbf8a", "#4c9bea"
    SEQUENTIAL = [SURFACE, "#184f95", "#3987e5", "#9ec5f4"]
else:
    INK, MUTED, GRID, AXIS, SURFACE = "#111827", "#6b7280", "#ebe9e2", "#cfcdc2", "#ffffff"
    ACTUAL = "#111827"
    MODEL_COLOR = {"xgb": "#2a78d6", "linear": "#eb6834", "naive": "#9aa0a6"}
    SOURCE_COLOR = {"solar": "#eda100", "wind_onshore": "#2a78d6", "wind_offshore": "#1baf7a"}
    SELL, BUY = "#1baf7a", "#2a78d6"
    SEQUENTIAL = [SURFACE, "#86b6ef", "#2a78d6", "#104281"]

STRATEGY_COLOR = {"perfect": ACTUAL, **MODEL_COLOR}

WEATHER_MODES = {
    "none": "No weather (leakage-free)",
    "observed": "Observed weather (upper bound, leaks)",
}
# Targets shown in the UI. Other trained models (the German residual load that feeds the price) are internal.
TARGET_LABELS = {"load_de": "Load DE", "residual_load_50hz": "Residual load 50Hertz", "price": "Price DE-LU"}
TARGET_UNITS = {"load_de": "MW", "residual_load_50hz": "MW", "price": "EUR/MWh"}
MODEL_LABELS = {"xgb": "XGBoost", "linear": "Linear regression", "naive": "Same hour last week"}


# --- Chart helpers -----------------------------------------------------------------
def base(fig: go.Figure, height: int, legend: bool = True, unit: str = "") -> go.Figure:
    """Shared look for every chart: transparent, quiet axes, legend above the plot."""
    fig.update_layout(
        height=height,
        # Room for tick labels: zero margins clip the axis numbers in the browser.
        margin=dict(l=56, r=16, t=40 if legend else 12, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", color=MUTED, size=12),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID, font=dict(color=INK)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None,
                    font=dict(color=INK)),
        bargap=0.3,
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, title=dict(text=unit, font=dict(color=MUTED)))
    return fig


def lines(df: pd.DataFrame, x: str, traces: list[tuple[str, str, str, str]], unit: str,
          height: int = 320, fmt: str = ",.0f") -> go.Figure:
    """traces: (column, label, color, dash). Colours and labels are fixed per series."""
    fig = go.Figure()
    for column, label, color, dash in traces:
        fig.add_trace(go.Scatter(
            x=df[x], y=df[column], name=label, mode="lines",
            line=dict(color=color, width=2, dash=dash),
            hovertemplate=f"%{{y:{fmt}}} {unit}",
        ))
    return base(fig, height, legend=len(traces) > 1, unit=unit)


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


@contextmanager
def card(title: str, caption: str | None = None):
    """A bordered panel with a title and an optional one-line explanation."""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if caption:
            st.caption(caption)
        yield


# --- Data access -------------------------------------------------------------------
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
    lines_out: list[str] = []
    env = {**os.environ, "NO_COLOR": "1", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in proc.stdout:
        lines_out.append(line.rstrip())
        box.code("\n".join(lines_out[-300:]), language="text")
    return proc.wait()


def pick_weather_mode(base_dir: Path, filename: str, key: str) -> str | None:
    """Radio over the weather modes whose output file exists; None if there are none."""
    modes = [m for m in WEATHER_MODES if (base_dir / f"weather={m}" / filename).exists()]
    if not modes:
        return None
    return st.radio("Training run", modes, format_func=WEATHER_MODES.get, horizontal=True, key=key)


def first_mode(base_dir: Path, filename: str) -> str | None:
    return next((m for m in WEATHER_MODES if (base_dir / f"weather={m}" / filename).exists()), None)


@st.cache_data(show_spinner=False)
def read_parquet(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "local_date" in df.columns:
        df["local_date"] = pd.to_datetime(df["local_date"]).dt.date
    return df


def load_parquet(path: Path) -> pd.DataFrame:
    return read_parquet(str(path), path.stat().st_mtime)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def dbt_results() -> pd.DataFrame | None:
    if not RUN_RESULTS.exists():
        return None
    results = load_json(RUN_RESULTS)["results"]
    icons = {"pass": "✅ pass", "success": "✅ success", "fail": "❌ fail", "error": "❌ error",
             "warn": "⚠️ warn", "skipped": "⏭ skipped"}
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


# --- Warehouse state (shared by the sidebar and every page) ------------------------
ready = warehouse_ready()
stats = None
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


# --- Pages -------------------------------------------------------------------------
def page_overview() -> None:
    st.title("Overview")
    st.caption("Day-ahead forecasts for the 50Hertz control area, issued at 12:00 on D−1.")
    if stats is None:
        st.info("No warehouse yet. Open **Pipeline** to run ingestion and dbt.")
        return

    fc_mode = first_mode(PREDICTIONS_DIR, "metrics.json")
    bt_mode = first_mode(BACKTEST_DIR, "summary.json")

    best = None
    if fc_mode:
        metrics = pd.DataFrame(load_json(PREDICTIONS_DIR / f"weather={fc_mode}" / "metrics.json")["metrics"])
        metrics = metrics[metrics["target"].isin(TARGET_LABELS)]
        models = metrics[metrics["model"] != "naive"]
        best = models.loc[models.groupby("target")["mae"].idxmin()].set_index("target")

    capture = None
    if bt_mode:
        strategies = pd.DataFrame(load_json(BACKTEST_DIR / f"weather={bt_mode}" / "summary.json")["strategies"])
        strategies = strategies[strategies["strategy"] != "perfect"]
        capture = strategies.loc[strategies["capture_pct"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hours of history", f"{int(stats['hours']):,}", help=f"Since {stats['first_ts']:%Y-%m-%d} UTC")
    c2.metric("Latest day-ahead price", f"{stats['last_price']:%d %b, %H:%M} UTC")
    if best is not None and "price" in best.index:
        row = best.loc["price"]
        c3.metric("Best price model", MODEL_LABELS[row["model"]],
                  f"{100 * row['skill_vs_naive']:+.0f}% vs baseline", help=f"MAE {row['mae']:.1f} EUR/MWh")
    else:
        c3.metric("Best price model", "–")
    if capture is not None:
        c4.metric("Battery capture", f"{capture['capture_pct']:.1f}%", "of perfect foresight", delta_color="off",
                  help=f"Best strategy: {capture['label']}")
    else:
        c4.metric("Battery capture", "–")

    if fc_mode is None:
        st.info("No forecasts yet. Run **3 · Training** on the Pipeline page.")
        return

    preds = load_parquet(PREDICTIONS_DIR / f"weather={fc_mode}" / "test_predictions.parquet")
    days = sorted(preds["local_date"].unique())
    last_week = preds[preds["local_date"] >= days[-7]]

    left, right = st.columns(2)
    with left:
        with card("Price, last 7 days", "Day-ahead price (DE-LU) against each forecast, local time."):
            show(lines(last_week, "ts_local", [
                ("price_actual", "Actual", ACTUAL, "solid"),
                ("price_xgb", MODEL_LABELS["xgb"], MODEL_COLOR["xgb"], "solid"),
                ("price_linear", MODEL_LABELS["linear"], MODEL_COLOR["linear"], "solid"),
                ("price_naive", MODEL_LABELS["naive"], MODEL_COLOR["naive"], "dot"),
            ], "EUR/MWh", height=360, fmt=",.1f"))
    with right:
        with card("Scorecard", "Best model per target on the test period."):
            if best is not None:
                st.dataframe(
                    pd.DataFrame({
                        "Target": [f"{TARGET_LABELS[t]} ({TARGET_UNITS[t]})" for t in best.index],
                        "Best model": best["model"].map(MODEL_LABELS).to_numpy(),
                        "MAE": best["mae"].to_numpy(),
                        "Skill": (100 * best["skill_vs_naive"]).to_numpy(),
                    }),
                    hide_index=True, width="stretch",
                    column_config={
                        "MAE": st.column_config.NumberColumn(format="%.1f"),
                        "Skill": st.column_config.NumberColumn("Skill vs baseline", format="%.0f%%"),
                    },
                )
            st.caption("Skill = MAE reduction against the same-hour-last-week baseline.")


def page_forecasts() -> None:
    st.title("Forecasts")
    mode = pick_weather_mode(PREDICTIONS_DIR, "metrics.json", key="forecast_mode")
    if mode is None:
        st.info("No forecasts yet. Run **3 · Training** on the Pipeline page.")
        return

    run_dir = PREDICTIONS_DIR / f"weather={mode}"
    summary = load_json(run_dir / "metrics.json")
    preds = load_parquet(run_dir / "test_predictions.parquet")
    if mode == "observed":
        st.warning("This run uses weather measured at the target hour, which is not known at 12:00 D−1. "
                   "Treat it as an upper bound, not a real forecast.", icon="⚠️")
    st.caption(f"Test period {summary['test_start']} → {summary['test_end']} (local days), never seen in training.")

    metrics = pd.DataFrame(summary["metrics"])
    metrics = metrics[metrics["target"].isin(TARGET_LABELS)]

    with card("Skill vs same hour last week",
              "Share of the baseline's error that each model removes. 0 % means no better than the baseline."):
        skill = metrics[metrics["model"] != "naive"]
        fig = go.Figure()
        for m in ["xgb", "linear"]:
            d = skill[skill["model"] == m]
            fig.add_trace(go.Bar(
                x=d["target"].map(TARGET_LABELS), y=100 * d["skill_vs_naive"], name=MODEL_LABELS[m],
                marker=dict(color=MODEL_COLOR[m], cornerradius=6),
                texttemplate="%{y:.0f}%", textposition="outside", cliponaxis=False,
                hovertemplate="%{y:.1f}%<extra>%{fullData.name}</extra>",
            ))
        fig.add_hline(y=0, line=dict(color=AXIS, width=1))
        base(fig, 300).update_yaxes(ticksuffix="%")
        show(fig)

    target = st.radio("Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get,
                      horizontal=True, key="forecast_target")
    unit = TARGET_UNITS[target]
    row = metrics[metrics["target"] == target].set_index("model")
    cols = st.columns(3)
    for col, m in zip(cols, ["xgb", "linear", "naive"]):
        r = row.loc[m]
        col.metric(
            MODEL_LABELS[m], f"{r['mae']:,.1f} {unit}",
            "baseline" if m == "naive" else f"{100 * r['skill_vs_naive']:+.1f}% vs baseline",
            delta_color="off" if m == "naive" else "normal",
            help=f"MAE on the test period. RMSE {r['rmse']:,.1f} {unit}.",
        )

    days = sorted(preds["local_date"].unique())
    window = st.date_input("Days shown", value=(days[-14], days[-1]), min_value=days[0], max_value=days[-1],
                           key="forecast_window")
    lo, hi = window if isinstance(window, tuple) and len(window) == 2 else (window[0], window[0])
    shown = preds[(preds["local_date"] >= lo) & (preds["local_date"] <= hi)]
    fmt = ",.1f" if target == "price" else ",.0f"
    resolution = st.radio("Resolution", ["Daily average", "Hourly"], horizontal=True, key="forecast_res")
    with card(f"{TARGET_LABELS[target]}: actual vs forecasts",
              "Daily averages remove hour-to-hour noise, so the trend is easier to read. Switch to hourly for peaks."):
        keys = [f"{target}_actual"] + [f"{target}_{m}" for m in MODEL_LABELS]
        if resolution == "Daily average":
            data, x = shown.groupby("local_date")[keys].mean().reset_index(), "local_date"
        else:
            data, x = shown, "ts_local"
        traces = [(f"{target}_actual", "Actual", ACTUAL, "solid")]
        traces += [(f"{target}_{m}", MODEL_LABELS[m], MODEL_COLOR[m], "dot" if m == "naive" else "solid")
                   for m in MODEL_LABELS]
        show(lines(data, x, traces, unit, height=360, fmt=fmt))

    left, right = st.columns(2)
    with left:
        with card("Predicted vs actual",
                  "One dot per hour. Dots on the dashed line are perfect. A cloud that is flatter than the line "
                  "means the forecast is too timid."):
            cols = [f"{target}_actual", f"{target}_xgb", f"{target}_linear"]
            lo_v, hi_v = float(preds[cols].min().min()), float(preds[cols].max().max())
            fig = go.Figure()
            for m in ["xgb", "linear"]:
                fig.add_trace(go.Scatter(
                    x=preds[f"{target}_actual"], y=preds[f"{target}_{m}"], name=MODEL_LABELS[m], mode="markers",
                    marker=dict(color=MODEL_COLOR[m], size=4, opacity=0.35),
                    hovertemplate=f"actual %{{x:{fmt}}}<br>forecast %{{y:{fmt}}}<extra>{MODEL_LABELS[m]}</extra>",
                ))
            fig.add_trace(go.Scatter(
                x=[lo_v, hi_v], y=[lo_v, hi_v], name="Perfect", mode="lines", hoverinfo="skip",
                line=dict(color=MUTED, dash="dash", width=1),
            ))
            base(fig, 340, unit=f"Forecast ({unit})")
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(title=dict(text=f"Actual ({unit})", font=dict(color=MUTED)))
            show(fig)
    with right:
        with card("Forecast error over time", "Mean absolute error per day. Spikes show the hard days."):
            daily_err = pd.DataFrame({m: (preds[f"{target}_actual"] - preds[f"{target}_{m}"]).abs()
                                      for m in MODEL_LABELS})
            daily_err["local_date"] = preds["local_date"].to_numpy()
            daily_err = daily_err.groupby("local_date").mean().reset_index()
            show(lines(daily_err, "local_date",
                       [(m, MODEL_LABELS[m], MODEL_COLOR[m], "dot" if m == "naive" else "solid") for m in MODEL_LABELS],
                       unit, height=340, fmt=fmt))

    left, right = st.columns(2)
    with left:
        with card("Error by hour of day", "Mean absolute error for each local hour, whole test period."):
            hour = pd.to_datetime(preds["ts_local"]).dt.hour.to_numpy()
            err = pd.DataFrame({m: (preds[f"{target}_actual"] - preds[f"{target}_{m}"]).abs()
                                for m in MODEL_LABELS})
            err["hour"] = hour
            by_hour = err.groupby("hour").mean().reset_index()
            fig = lines(by_hour, "hour", [(m, MODEL_LABELS[m], MODEL_COLOR[m], "solid") for m in MODEL_LABELS],
                        unit, height=320, fmt=fmt)
            fig.update_xaxes(dtick=3, title=dict(text="Local hour", font=dict(color=MUTED)))
            show(fig)
    with right:
        importance = pd.Series(summary["importance"].get(target, {}), dtype="float64").sort_values()
        with card("What drives the XGBoost model", "Feature importance for this target."):
            if len(importance):
                fig = go.Figure(go.Bar(
                    x=importance.values, y=importance.index, orientation="h",
                    marker=dict(color=MODEL_COLOR["xgb"], cornerradius=4),
                    hovertemplate="%{y}: %{x:.3f}<extra></extra>",
                ))
                base(fig, max(300, 26 * len(importance)), legend=False)
                fig.update_layout(hovermode="closest", margin=dict(l=130, r=16, t=12, b=36))
                fig.update_xaxes(showgrid=True, gridcolor=GRID)
                fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
                show(fig)
            else:
                st.caption("No importance recorded for this target.")


def page_battery() -> None:
    st.title("Battery backtest")
    mode = pick_weather_mode(BACKTEST_DIR, "summary.json", key="backtest_view_mode")
    if mode is None:
        st.info("No backtest yet. Run **4 · Battery backtest** on the Pipeline page.")
        return

    run_dir = BACKTEST_DIR / f"weather={mode}"
    summary = load_json(run_dir / "summary.json")
    daily = load_parquet(run_dir / "daily.parquet")
    schedules = load_parquet(run_dir / "schedules.parquet")
    battery = summary["battery"]
    st.caption(
        f"{battery['power_mw']:.0f} MW / {battery['energy_mwh']:.0f} MWh battery · "
        f"{battery['round_trip_efficiency']:.0%} round trip · {battery['cycles_per_day']:g} cycle/day · "
        f"{summary['first_day']} → {summary['last_day']}. Each day is scheduled on the price curve known at "
        "12:00 D−1 and settled at the actual day-ahead prices."
    )

    strategies = pd.DataFrame(summary["strategies"]).set_index("strategy")
    labels = strategies["label"].to_dict()
    for col, (key, row) in zip(st.columns(len(strategies)), strategies.iterrows()):
        col.metric(
            row["label"], f"€{row['revenue_eur'] / 1e6:,.2f} M",
            None if key == "perfect" else f"{row['capture_pct']:.1f}% of perfect", delta_color="off",
        )

    cumulative = daily.pivot(index="local_date", columns="strategy", values="revenue_eur").cumsum().reset_index()
    capture = pd.DataFrame({
        "local_date": cumulative["local_date"],
        **{k: 100 * cumulative[k] / cumulative["perfect"] for k in strategies.index if k != "perfect"},
    })
    traces = [(k, labels[k], STRATEGY_COLOR[k], "dash" if k == "perfect" else "solid") for k in strategies.index]

    left, right = st.columns(2)
    with left:
        with card("Cumulative revenue", "Total revenue over the test period."):
            show(lines(cumulative, "local_date", traces, "EUR", height=300))
    with right:
        with card("Capture rate over time", "Cumulative revenue as a share of perfect foresight."):
            show(lines(capture, "local_date",
                       [t for t in traces if t[0] != "perfect"], "%", height=300, fmt=".1f"))

    with card("Day detail", "Pick a day and a strategy to see what the battery did, hour by hour."):
        c1, c2 = st.columns(2)
        day = c1.selectbox("Day", sorted(daily["local_date"].unique(), reverse=True),
                           format_func=lambda d: f"{d:%a %d %b %Y}")
        strategy = c2.selectbox("Compare with perfect foresight",
                                [k for k in strategies.index if k != "perfect"], format_func=labels.get)

        that_day = schedules[schedules["local_date"] == day]
        perfect_rows = that_day[that_day["strategy"] == "perfect"]
        chosen_rows = that_day[that_day["strategy"] == strategy]
        revenue = daily[daily["local_date"] == day].set_index("strategy")["revenue_eur"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Perfect foresight", f"€{revenue['perfect']:,.0f}")
        m2.metric(labels[strategy], f"€{revenue[strategy]:,.0f}")
        captured = 100 * revenue[strategy] / revenue["perfect"] if revenue["perfect"] else float("nan")
        m3.metric("Gap to perfect", f"€{revenue[strategy] - revenue['perfect']:,.0f}",
                  f"{captured:.0f}% captured", delta_color="off")

        chosen_net = chosen_rows["discharge_mw"] - chosen_rows["charge_mw"]
        perfect_net = perfect_rows["discharge_mw"] - perfect_rows["charge_mw"]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=chosen_rows["ts_local"], y=chosen_rows["price_actual"], name="Actual price", mode="lines",
            line=dict(color=ACTUAL, width=2), hovertemplate="%{y:,.1f} EUR/MWh",
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=chosen_rows["ts_local"], y=chosen_rows["price_planned"], name=f"{labels[strategy]} price",
            mode="lines", line=dict(color=STRATEGY_COLOR[strategy], width=2, dash="dot"),
            hovertemplate="%{y:,.1f} EUR/MWh",
        ), secondary_y=False)
        fig.add_trace(go.Bar(
            x=chosen_rows["ts_local"], y=chosen_net, name=f"{labels[strategy]} battery",
            marker=dict(color=[SELL if v > 0 else BUY for v in chosen_net], cornerradius=3),
            opacity=0.5, hovertemplate="%{y:,.0f} MW",
        ), secondary_y=True)
        fig.add_trace(go.Scatter(
            x=perfect_rows["ts_local"], y=perfect_net, name="Perfect foresight battery", mode="lines",
            line=dict(color=MUTED, width=1.5, dash="dash"), hovertemplate="%{y:,.0f} MW",
        ), secondary_y=True)
        base(fig, 380, unit="EUR/MWh")
        fig.update_yaxes(title_text="MW · + sell / − buy", secondary_y=True, showgrid=False)
        show(fig)

    with st.expander("Daily revenue table"):
        table = daily.pivot(index="local_date", columns="strategy", values="revenue_eur")
        st.dataframe(
            table.rename(columns=labels).round(0).iloc[::-1], width="stretch",
            column_config={c: st.column_config.NumberColumn(format="€%d") for c in table.columns},
        )


def page_explorer() -> None:
    st.title("Explorer")
    st.caption("Load, renewables and prices for any date range. Dates are local (Europe/Berlin).")
    if not ready:
        st.info("Build the warehouse first, on the Pipeline page.")
        return

    first = stats["first_ts"].date()
    last = stats["last_price"].date()
    c1, c2 = st.columns([2, 1])
    picked = c1.date_input("Date range", value=(max(first, last - timedelta(days=28)), last),
                           min_value=first, max_value=last)
    region = c2.radio("Region", ["50Hertz", "DE"], horizontal=True)
    start, end = picked if isinstance(picked, tuple) and len(picked) == 2 else (picked[0], picked[0])
    sfx = "50hz" if region == "50Hertz" else "de"

    df = q("select * from hourly where cast(ts_local as date) between ? and ? order by ts_utc", (start, end))
    if df.empty:
        st.info("No data in this range.")
        return

    load_col = f"load_{sfx}_mw"
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Average load", f"{df[load_col].mean():,.0f} MW")
    k2.metric("Peak load", f"{df[load_col].max():,.0f} MW")
    k3.metric("Average price", f"{df['price_eur_mwh'].mean():,.1f} EUR/MWh")
    negative = int((df["price_eur_mwh"] < 0).sum())
    k4.metric("Negative-price hours", f"{negative:,}", help="Hours where the day-ahead price was below zero.")

    with card(f"Load and residual load · {region}"):
        show(lines(df, "ts_local", [
            (load_col, "Load", BUY, "solid"),
            (f"residual_load_{sfx}_mw", "Residual load", ACTUAL, "solid"),
        ], "MW", height=320))

    with card(f"Renewable generation · {region}", "Stacked: solar, onshore and offshore wind."):
        fig = go.Figure()
        for column, label, color in [
            (f"solar_{sfx}_mw", "Solar", SOURCE_COLOR["solar"]),
            (f"wind_onshore_{sfx}_mw", "Wind onshore", SOURCE_COLOR["wind_onshore"]),
            (f"wind_offshore_{sfx}_mw", "Wind offshore", SOURCE_COLOR["wind_offshore"]),
        ]:
            fig.add_trace(go.Scatter(
                x=df["ts_local"], y=df[column], name=label, stackgroup="renewables", mode="lines",
                line=dict(color=color, width=0.5), fillcolor=color,
                hovertemplate="%{y:,.0f} MW",
            ))
        show(base(fig, 320, unit="MW"))

    with card("Day-ahead price · DE-LU"):
        show(lines(df, "ts_local", [("price_eur_mwh", "Price", ACTUAL, "solid")], "EUR/MWh",
                   height=280, fmt=",.1f"))

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
    with card(f"Average daily load profile · {region}", "Typical load by local hour, split by day type."):
        traces = [(c, c, color, "solid") for c, color in
                  [("Working day", BUY), ("Weekend / holiday", "#eb6834" if not DARK else "#f08a5d")]
                  if c in profile.columns]
        fig = lines(profile, "hour_local", traces, "MW", height=300)
        fig.update_xaxes(dtick=3, title=dict(text="Local hour", font=dict(color=MUTED)))
        show(fig)


def page_quality() -> None:
    st.title("Data quality")
    st.caption("Source coverage, missing hours, and how each feature relates to the targets.")
    if not ready:
        st.info("Build the warehouse first, on the Pipeline page.")
        return

    results = dbt_results()
    if results is not None:
        tests = results[results["type"] == "test"]
        passed = int(tests["status"].str.contains("pass").sum()) if len(tests) else 0
        c1, c2 = st.columns(2)
        c1.metric("dbt tests passing", f"{passed} / {len(tests)}" if len(tests) else "–")
        c2.metric("Models built", f"{int((results['type'] == 'model').sum())}")

    # Stacked, not side by side: the wide source tables need the full page width.
    left, right = st.container(), st.container()
    with left:
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
        with card("SMARD series", "Row counts and gaps per series and bidding region."):
            st.dataframe(smard, hide_index=True, width="stretch")
    with right:
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
        with card("Weather (Bright Sky)", "Coverage per city."):
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
    heat = go.Figure(go.Heatmap(
        x=gaps.index, y=gaps.columns, z=gaps.T.values,
        colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
        zmin=0, zmax=24, xgap=1, ygap=2,
        colorbar=dict(title=dict(text="missing h", font=dict(color=MUTED)), thickness=10),
        hovertemplate="%{y}<br>%{x|%Y-%m-%d}: %{z} missing hours<extra></extra>",
    ))
    with card("Missing hours per local day", "Each cell is one series on one day (24 hours max)."):
        base(heat, 280, legend=False).update_layout(hovermode="closest", margin=dict(l=150, r=16, t=12, b=36))
        show(heat)
        total = gaps.sum()
        st.caption("Total missing hours · " + " · ".join(f"{k}: {int(v)}" for k, v in total.items()))

    st.warning(
        "Weather columns are **observed** values at the target hour. For a leakage-free model "
        "(forecast issued at 12:00 D−1), training must replace them with weather forecasts. "
        "All load/price lags are ≥ 48 h and therefore known at forecast time.",
        icon="⚠️",
    )

    features = q("select * from features order by ts_utc")
    target = st.radio(
        "Correlate features with", ["load_50hz_mw", "price_eur_mwh", "residual_load_50hz_mw"], horizontal=True,
        format_func=lambda c: {"load_50hz_mw": "Load 50Hertz", "price_eur_mwh": "Price DE-LU",
                               "residual_load_50hz_mw": "Residual load 50Hertz"}[c],
    )
    candidates = [
        c for c in features.columns
        if c.endswith(("_lag_48h", "_lag_168h", "_lag_336h")) or c.startswith("weather_")
        or c in ("hour_local", "day_of_week_local", "month_local")
    ]
    corr = features[candidates + [target]].astype("float64").corr()[target].drop(target).sort_values()
    bars = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h",
        marker=dict(color=[SELL if v >= 0 else BUY for v in corr.values], cornerradius=4),
        hovertemplate="%{y}: r = %{x:.2f}<extra></extra>",
    ))
    with card(f"Pearson correlation with {target}", "Green = positive, blue = negative."):
        base(bars, max(320, 24 * len(corr)), legend=False).update_layout(hovermode="closest",
                                                                       margin=dict(l=170, r=16, t=12, b=36))
        bars.update_xaxes(range=[-1, 1], showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=AXIS)
        bars.update_yaxes(gridcolor="rgba(0,0,0,0)")
        show(bars)

    left, right = st.columns([1, 2])
    with left:
        nulls = features.isna().sum().rename("nulls").to_frame()
        nulls["pct"] = (100 * nulls["nulls"] / len(features)).round(2)
        with card("Null counts"):
            st.dataframe(nulls, width="stretch")
    with right:
        with card("Latest rows", "The 48 most recent hours in the feature table."):
            st.dataframe(features.tail(48).iloc[::-1], hide_index=True, width="stretch")


def page_pipeline() -> None:
    st.title("Pipeline")
    st.caption("Run each stage from here. Output streams below the buttons.")

    left, right = st.columns(2)
    with left:
        with card("1 · Ingestion", "SMARD, Bright Sky and holidays → data/raw/*.parquet. Idempotent."):
            days = st.number_input("History (days)", min_value=7, max_value=1500, value=LOOKBACK_DAYS, step=7)
            sources = st.multiselect("Sources", ["smard", "weather", "holidays"],
                                     default=["smard", "weather", "holidays"])
            force = st.checkbox("Force re-download")
            if st.button("Run ingestion", type="primary", disabled=not sources, width="stretch"):
                cmd = [sys.executable, "-m", "ingestion.run", "--days", str(days), "--only", *sources]
                code = run_streaming(cmd + (["--force"] if force else []), PROJECT_ROOT)
                (st.success if code == 0 else st.error)(f"Ingestion finished with exit code {code}")
    with right:
        with card("2 · dbt build", "Staging → hourly → features in DuckDB, then all dbt tests."):
            st.write("")
            if st.button("Run dbt build", type="primary", width="stretch"):
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
        with card("3 · Training", "XGBoost vs linear regression vs same hour last week. Logged to MLflow."):
            train_mode = st.radio("Weather features", list(WEATHER_MODES), format_func=WEATHER_MODES.get,
                                  key="train_mode")
            if st.button("Run training", type="primary", width="stretch"):
                code = run_streaming([sys.executable, "-m", "training.train", "--weather", train_mode], PROJECT_ROOT)
                (st.success if code == 0 else st.error)(f"Training finished with exit code {code}")
    with right:
        with card("4 · Battery backtest", "100 MW / 200 MWh, 90 % round trip, 1 cycle/day."):
            backtest_mode = st.radio("Predictions from", list(WEATHER_MODES), format_func=WEATHER_MODES.get,
                                     key="backtest_mode")
            if st.button("Run backtest", type="primary", width="stretch"):
                code = run_streaming([sys.executable, "-m", "backtest.run", "--weather", backtest_mode],
                                     PROJECT_ROOT)
                (st.success if code == 0 else st.error)(f"Backtest finished with exit code {code}")

    st.subheader("Last dbt run")
    results = dbt_results()
    if results is None:
        st.caption("No dbt run yet.")
    else:
        st.dataframe(results, hide_index=True, width="stretch")


PAGES = {
    "Overview": page_overview,
    "Forecasts": page_forecasts,
    "Battery backtest": page_battery,
    "Explorer": page_explorer,
    "Data quality": page_quality,
    "Pipeline": page_pipeline,
}

with st.sidebar:
    st.markdown("**LEAG Energy Forecast**")
    st.caption("Day-ahead load & price forecasting for 50Hertz, with a battery backtest.")
    page = st.radio("Page", list(PAGES), label_visibility="collapsed", key="nav")
    st.divider()
    if stats is not None:
        st.caption(f"Warehouse · {int(stats['hours']):,} hours")
    else:
        st.caption("Warehouse · not built yet")

PAGES[page]()
