"""
🇱🇦 Laos FX & Import Price Forecast — Policy Dashboard
=====================================================
Developed for the FX Policy Division, Bank of Laos (BOL) / NSC

Purpose : Monitor and scenario-test global commodity prices and exchange rates
          that directly affect Laos’ import costs, inflation, and LAK stability.
Data    : Live via Yahoo Finance (~15-min delay)  |  5-Scenario monthly projections
Assets  : Commodity imports (Energy, Metals, Agricultural) + LAK/USD, LAK/THB cross rate
Authors : FX Policy Division — Forecasting & Scenarios Unit
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🇱🇦 Laos FX & Import Price Forecast",
    page_icon="🇱🇦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
COMMODITY_TICKERS: dict[str, str] = {
    # Energy
    "Brent Crude ($/bbl)":   "BZ=F",
    "WTI Crude ($/bbl)":     "CL=F",
    "Natural Gas ($/MMBtu)": "NG=F",
    # Metals
    "Gold ($/oz)":           "GC=F",
    "Silver ($/oz)":         "SI=F",
    "Copper ($/lb)":         "HG=F",
    "Aluminium ($/ton)":     "ALI=F",
    # Agricultural
    "Wheat (c/bu)":          "ZW=F",
    "Corn (c/bu)":           "ZC=F",
    "Soybeans (c/bu)":       "ZS=F",
    "Sugar #11 (c/lb)":      "SB=F",
}

FX_TICKERS: dict[str, str] = {
    "EUR/USD": "EURUSD=X",   # USD per 1 EUR
    "GBP/USD": "GBPUSD=X",   # USD per 1 GBP
    "USD/JPY": "JPY=X",      # JPY per 1 USD
    "USD/THB": "THB=X",      # THB per 1 USD
    "USD/LAK": "LAK=X",      # Lao Kip per 1 USD (exotic – may need manual override)
}

COMMODITY_GROUPS: dict[str, list[str]] = {
    "Energy":       ["Brent Crude ($/bbl)", "WTI Crude ($/bbl)", "Natural Gas ($/MMBtu)"],
    "Metals":       ["Gold ($/oz)", "Silver ($/oz)", "Copper ($/lb)", "Aluminium ($/ton)"],
    "Agricultural": ["Wheat (c/bu)", "Corn (c/bu)", "Soybeans (c/bu)", "Sugar #11 (c/lb)"],
}

SCENARIOS: list[str] = [
    "Strong Bull",
    "Moderate Bull",
    "Base Case",
    "Moderate Bear",
    "Strong Bear",
]

SCENARIO_COLORS: dict[str, str] = {
    "Strong Bull":   "#00FF87",   # neon green
    "Moderate Bull": "#36D399",   # teal green
    "Base Case":     "#60A5FA",   # bright blue
    "Moderate Bear": "#FBBF24",   # amber gold
    "Strong Bear":   "#F87171",   # neon red
}

SCENARIO_DASH: dict[str, str] = {
    "Strong Bull":   "dot",
    "Moderate Bull": "dash",
    "Base Case":     "solid",
    "Moderate Bear": "dash",
    "Strong Bear":   "dot",
}

SCENARIO_BG: dict[str, str] = {
    "Strong Bull":   "#0A3320",   # dark green
    "Moderate Bull": "#0D2A1C",   # deep teal
    "Base Case":     "#0D2040",   # dark navy
    "Moderate Bear": "#2D1E06",   # dark amber
    "Strong Bear":   "#2D0D0D",   # dark red
}

DEFAULT_MONTHLY_PCT: dict[str, float] = {
    "Strong Bull":    5.0,
    "Moderate Bull":  2.5,
    "Base Case":      0.0,
    "Moderate Bear": -2.5,
    "Strong Bear":   -5.0,
}

TODAY = datetime.today()

# ─────────────────────────────────────────────────────────────────────────────
#  LAOS FOCUS — DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
LAK_DEFAULT_INFLATION_PCT: float = 26.0   # current annual CPI inflation %, editable in sidebar

# Scenario adjustments to the annual inflation rate (percentage points)
LAK_INFL_SCENARIO_ADJ: dict[str, float] = {
    "Strong Bull":    8.0,   # strong growth → demand-pull inflation rises
    "Moderate Bull":  3.0,
    "Base Case":      0.0,
    "Moderate Bear": -3.0,   # slowdown eases price pressure
    "Strong Bear":   -8.0,   # sharp recession / deflation pressure
}

# Default monthly LAK depreciation per scenario (+ = LAK weakens vs USD)
LAK_DEFAULT_MONTHLY_PCT: dict[str, float] = {
    "Strong Bull":    4.0,   # strong depreciation (capital outflow / inflation)
    "Moderate Bull":  2.0,
    "Base Case":      0.5,   # gradual structural drift
    "Moderate Bear": -1.0,   # LAK firms slightly
    "Strong Bear":   -3.0,   # LAK firms on deflation / capital repatriation
}

# FX Intervention & Oil pass-through defaults
BOL_RESERVES_DEFAULT_USD_MN: float        = 1_300.0   # BOL reserves estimate (USD mn)
DAILY_FX_VOLUME_DEFAULT_USD_MN: float     = 5.0       # Lao daily FX market volume (USD mn)
TARGET_MAX_DEP_DEFAULT: float             = 1.0       # target max monthly LAK depreciation %
FX_INTV_EFFECTIVENESS_DEFAULT: float      = 50.0      # intervention effectiveness (%)
OIL_CPI_BETA_DEFAULT: float               = 0.30      # fraction: 10% oil rise → 3 pp CPI
EUR_CPI_BETA_DEFAULT: float               = 0.15      # 10% EUR/USD rise → +1.5 pp CPI via Thai/VN imports

# EUR/USD → USD/LAK inverse beta
# When EUR/USD falls 1% (USD strengthens), USD/LAK typically rises by ~BETA %
# (LAK weakens as capital flows to safe-haven USD).
# Empirical range for frontier EM currencies: 0.5 – 1.2.
EUR_LAK_BETA_DEFAULT: float               = 0.80      # 1% EUR/USD fall → +0.80% USD/LAK rise

# Default monthly USD/THB % per scenario (+ = THB weakens vs USD; THB is more stable than LAK)
THB_DEFAULT_MONTHLY_PCT: dict[str, float] = {
    "Strong Bull":    1.5,   # risk-off / EM sell-off → THB weakens vs USD
    "Moderate Bull":  0.8,
    "Base Case":      0.0,
    "Moderate Bear": -0.5,   # EM stabilise → THB firms
    "Strong Bear":   -1.5,   # safe-haven demand → THB firms
}

# THB-channel BOL intervention defaults
BOL_THB_RESERVES_DEFAULT_USD_MN: float   = 200.0      # BOL THB buffer (USD equiv mn)
DAILY_THB_VOL_DEFAULT_USD_MN: float      = 2.0        # daily LAK/THB market vol (USD equiv mn)
TARGET_LAK_THB_DEP_DEFAULT: float        = 1.5        # target max LAK/THB dep %/mo


# ─────────────────────────────────────────────────────────────────────────────
#  DATA FETCHING  (cached 5 minutes)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(tickers: dict) -> dict:
    """Return {label: latest_close_price or None} for a ticker dict."""
    result: dict = {}
    for label, ticker in tickers.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            prices_col = hist["Close"].dropna()
            result[label] = float(prices_col.iloc[-1]) if not prices_col.empty else None
        except Exception:
            result[label] = None
    return result


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker: str, period: str = "6mo") -> pd.Series:
    """Return a Close price Series for a single ticker (for sparklines)."""
    try:
        hist = yf.Ticker(ticker).history(period=period)
        return hist["Close"].dropna()
    except Exception:
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
#  SCENARIO ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def project(base: float, months: int, monthly_pct: float) -> list[float]:
    """Compound monthly projection. Month-0 = base, months 1..n projected."""
    r = monthly_pct / 100.0
    return [round(base * ((1 + r) ** m), 6) for m in range(months + 1)]


def project_gold_lak(
    gold_usd: float,
    lak_usd: float,
    months: int,
    gold_monthly_pct: float,
    lak_monthly_pct: float,
) -> list[float]:
    """Project Gold price in LAK: compounds Gold($/oz) × USD/LAK rate monthly."""
    base = gold_usd * lak_usd
    rg = gold_monthly_pct / 100.0
    rl = lak_monthly_pct / 100.0
    return [round(base * ((1 + rg) ** m) * ((1 + rl) ** m), 0) for m in range(months + 1)]


def project_inflation_index(
    base_annual_pct: float,
    months: int,
    annual_adj_pp: float,
) -> list[float]:
    """
    Project a CPI price index (base = 100 at current date).
    annual_adj_pp = scenario adjustment in percentage points added to the
    annual inflation rate.
    """
    scenario_annual = base_annual_pct + annual_adj_pp
    monthly_rate = (1 + scenario_annual / 100.0) ** (1 / 12) - 1
    return [round(100 * (1 + monthly_rate) ** m, 4) for m in range(months + 1)]


def scenario_table(prices: dict, months: int, pcts: dict) -> pd.DataFrame:
    """
    Build a wide scenario table.
    Rows = assets.  Columns = Current Price + (Scenario | Month) pairs.
    """
    month_tags = [
        (TODAY + timedelta(days=30 * m)).strftime("%b %Y")
        for m in range(1, months + 1)
    ]
    records = []
    for asset, price in prices.items():
        if price is None:
            continue
        rec: dict = {"Asset": asset, "Current Price": round(price, 4)}
        for sc in SCENARIOS:
            monthly_pct = pcts.get(sc, DEFAULT_MONTHLY_PCT[sc])
            vals = project(price, months, monthly_pct)
            for i, tag in enumerate(month_tags, start=1):
                rec[f"{sc} | {tag}"] = round(vals[i], 4)
        records.append(rec)
    return pd.DataFrame(records).set_index("Asset")


def scenario_summary(prices: dict, months: int, pcts: dict) -> pd.DataFrame:
    """
    Build a condensed summary table:
    Rows = scenarios.  Columns = assets (end-of-horizon price).
    """
    records = []
    for sc in SCENARIOS:
        monthly_pct = pcts.get(sc, DEFAULT_MONTHLY_PCT[sc])
        row: dict = {"Scenario": sc, "Monthly % chg": monthly_pct}
        for asset, price in prices.items():
            if price is None:
                continue
            vals = project(price, months, monthly_pct)
            chg_pct = ((vals[-1] / price) - 1) * 100
            row[asset] = f"{vals[-1]:,.4f}  ({chg_pct:+.1f}%)"
        records.append(row)
    return pd.DataFrame(records).set_index("Scenario")


# ─────────────────────────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def price_path_chart(label: str, base: float, months: int, pcts: dict) -> go.Figure:
    """5-scenario price/rate path chart using Plotly."""
    x_labels = ["Now"] + [
        (TODAY + timedelta(days=30 * m)).strftime("%b %Y")
        for m in range(1, months + 1)
    ]
    fig = go.Figure()
    for sc in SCENARIOS:
        monthly_pct = pcts.get(sc, DEFAULT_MONTHLY_PCT[sc])
        y = project(base, months, monthly_pct)
        # Annotate final value
        text = [""] * len(y)
        text[-1] = f"  {y[-1]:,.2f}"
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=y,
                mode="lines+markers+text",
                name=f"{sc} ({monthly_pct:+.1f}%/mo)",
                text=text,
                textposition="middle right",
                textfont=dict(
                    color=SCENARIO_COLORS[sc],
                    size=11,
                    family="Courier New, monospace",
                ),
                line=dict(
                    color=SCENARIO_COLORS[sc],
                    dash=SCENARIO_DASH[sc],
                    width=2.5,
                ),
                marker=dict(size=7, color=SCENARIO_COLORS[sc]),
                hovertemplate=f"<b>{sc}</b><br>%{{x}}: %{{y:,.4f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=dict(
            text=f"<b>{label}</b> — 5-Scenario Forecast",
            font_size=15,
            font_color="#F5A623",
        ),
        xaxis_title="Period",
        yaxis_title="Price / Rate",
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font_size=11,
            bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1",
        ),
        height=430,
        margin=dict(t=90, b=50, r=120, l=65),
        hovermode="x unified",
        paper_bgcolor="#0F1923",
        plot_bgcolor="#0B1420",
        font=dict(color="#CBD5E1"),
        xaxis=dict(
            gridcolor="#1E3A5F",
            linecolor="#1E3A5F",
            tickfont=dict(color="#94A3B8"),
        ),
        yaxis=dict(
            gridcolor="#1E3A5F",
            linecolor="#1E3A5F",
            tickfont=dict(color="#94A3B8"),
        ),
    )
    return fig


def sparkline(series: pd.Series, color: str = "#1D4ED8") -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba")
            if "rgb" in color
            else color + "15",
        )
    )
    fig.update_layout(
        height=60,
        margin=dict(t=0, b=0, l=0, r=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  EXCEL EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def generate_excel(
    comm_prices: dict,
    fx_prices: dict,
    months: int,
    pcts_comm: dict,
    pcts_fx: dict,
    lak_prices: dict | None = None,
    pcts_lak: dict | None = None,
    inflation_base: float = 26.0,
    infl_adj: dict | None = None,
) -> BytesIO:
    month_tags = [
        (TODAY + timedelta(days=30 * m)).strftime("%b %Y")
        for m in range(1, months + 1)
    ]
    buf = BytesIO()

    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book

        # ── Shared formats ────────────────────────────────────────────────────
        title_fmt = wb.add_format(
            {
                "bold": True,
                "font_size": 13,
                "font_color": "white",
                "bg_color": "#1D4ED8",
                "align": "center",
                "valign": "vcenter",
                "border": 1,
            }
        )
        hdr_fmt = wb.add_format(
            {
                "bold": True,
                "font_color": "white",
                "bg_color": "#1E3A8A",
                "align": "center",
                "valign": "vcenter",
                "border": 1,
                "text_wrap": True,
            }
        )
        label_fmt = wb.add_format(
            {"bold": True, "bg_color": "#EFF6FF", "border": 1, "valign": "vcenter"}
        )
        curr_fmt = wb.add_format(
            {
                "bold": True,
                "num_format": "#,##0.0000",
                "bg_color": "#BFDBFE",
                "border": 1,
            }
        )
        pct_fmt = wb.add_format({"num_format": "0.00", "border": 1, "align": "center"})

        # Per-scenario header + cell formats
        sc_hdr_fmts: dict = {}
        sc_cell_fmts: dict = {}
        for sc in SCENARIOS:
            sc_hdr_fmts[sc] = wb.add_format(
                {
                    "bold": True,
                    "font_color": "white",
                    "bg_color": SCENARIO_COLORS[sc],
                    "align": "center",
                    "border": 1,
                    "text_wrap": True,
                }
            )
            sc_cell_fmts[sc] = wb.add_format(
                {"num_format": "#,##0.0000", "border": 1, "bg_color": SCENARIO_BG[sc]}
            )

        # ── Helper: write one forecast sheet ──────────────────────────────────
        def write_forecast_sheet(sheet_name: str, prices: dict, pcts: dict) -> None:
            # Create sheet via a dummy write
            pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            total_data_cols = 1 + len(SCENARIOS) * months  # asset(0), current(1), data…

            # Row 0 – title banner
            ws.merge_range(
                0, 0, 0, 1 + total_data_cols,
                f"{sheet_name}  ·  5-Scenario Forecast  ·  "
                f"{months}-Month Horizon  ·  As of {TODAY.strftime('%d %b %Y')}",
                title_fmt,
            )
            ws.set_row(0, 28)

            # Row 1 – scenario group headers
            ws.write(1, 0, "Asset", hdr_fmt)
            ws.write(1, 1, f"Current\n({TODAY.strftime('%d %b %Y')})", hdr_fmt)
            col = 2
            for sc in SCENARIOS:
                ws.merge_range(1, col, 1, col + months - 1, sc, sc_hdr_fmts[sc])
                col += months
            ws.set_row(1, 32)

            # Row 2 – month labels under each scenario
            ws.write(2, 0, "", hdr_fmt)
            ws.write(2, 1, "", hdr_fmt)
            col = 2
            for _ in SCENARIOS:
                for tag in month_tags:
                    ws.write(2, col, tag, hdr_fmt)
                    col += 1
            ws.set_row(2, 18)

            # Rows 3+ – data
            row_idx = 3
            for asset, price in prices.items():
                if price is None:
                    continue
                ws.write(row_idx, 0, asset, label_fmt)
                ws.write(row_idx, 1, round(price, 4), curr_fmt)
                col = 2
                for sc in SCENARIOS:
                    monthly_pct = pcts.get(sc, DEFAULT_MONTHLY_PCT[sc])
                    vals = project(price, months, monthly_pct)
                    for m_i in range(1, months + 1):
                        ws.write(row_idx, col, round(vals[m_i], 4), sc_cell_fmts[sc])
                        col += 1
                row_idx += 1

            # Column widths
            ws.set_column(0, 0, 28)
            ws.set_column(1, 1, 18)
            ws.set_column(2, 2 + len(SCENARIOS) * months, 13)
            ws.freeze_panes(3, 2)

        # ── Write the two forecast sheets ─────────────────────────────────────
        write_forecast_sheet(
            "Commodity Scenarios",
            {k: v for k, v in comm_prices.items() if v is not None},
            pcts_comm,
        )
        write_forecast_sheet(
            "FX Scenarios",
            {k: v for k, v in fx_prices.items() if v is not None},
            pcts_fx,
        )

        # ── Laos Focus sheet ──────────────────────────────────────────────────
        if lak_prices and pcts_lak:
            lak_valid = {k: v for k, v in lak_prices.items() if v is not None}
            if lak_valid:
                write_forecast_sheet("Laos — FX & Gold (LAK)", lak_valid, pcts_lak)

            if infl_adj:
                pd.DataFrame().to_excel(writer, sheet_name="Laos — Inflation", index=False)
                ws_li = writer.sheets["Laos — Inflation"]
                n_infl_cols = len(month_tags) + 2
                ws_li.merge_range(
                    0, 0, 0, n_infl_cols,
                    f"Laos CPI Inflation Scenarios  ·  {months}-Month Horizon  ·  "
                    f"Base Rate: {inflation_base:.1f}% p.a.  ·  {TODAY.strftime('%d %b %Y')}",
                    title_fmt,
                )
                ws_li.set_row(0, 28)
                # Section A – scenario rate summary
                ws_li.merge_range(1, 0, 1, 4, "Scenario Rate Summary", hdr_fmt)
                ws_li.set_row(1, 22)
                for ci, hdr_text in enumerate(
                    ["Scenario", "Base (%)", "Adj (pp)", "Scenario Rate (%)", "Monthly Rate (%)"]
                ):
                    ws_li.write(2, ci, hdr_text, hdr_fmt)
                ws_li.set_row(2, 18)
                for i_s, sc in enumerate(SCENARIOS):
                    adj_v = infl_adj.get(sc, LAK_INFL_SCENARIO_ADJ.get(sc, 0.0))
                    sc_annual = round(inflation_base + adj_v, 2)
                    monthly_r = round(((1 + sc_annual / 100.0) ** (1 / 12) - 1) * 100, 4)
                    ws_li.write(3 + i_s, 0, sc, label_fmt)
                    ws_li.write(3 + i_s, 1, inflation_base, pct_fmt)
                    ws_li.write(3 + i_s, 2, adj_v, pct_fmt)
                    ws_li.write(3 + i_s, 3, sc_annual, pct_fmt)
                    ws_li.write(3 + i_s, 4, monthly_r, pct_fmt)
                # Section B – CPI index projection
                ws_li.merge_range(
                    9, 0, 9, len(month_tags) + 1,
                    "CPI Index Projection (Base at current date = 100)", hdr_fmt,
                )
                ws_li.set_row(9, 22)
                ws_li.write(10, 0, "Scenario", hdr_fmt)
                ws_li.write(10, 1, "Annual Rate (%)", hdr_fmt)
                for m_i, tag in enumerate(month_tags):
                    ws_li.write(10, 2 + m_i, tag, hdr_fmt)
                ws_li.set_row(10, 18)
                for i_s, sc in enumerate(SCENARIOS):
                    adj_v = infl_adj.get(sc, LAK_INFL_SCENARIO_ADJ.get(sc, 0.0))
                    sc_annual = round(inflation_base + adj_v, 2)
                    idx_vals = project_inflation_index(inflation_base, months, adj_v)
                    ws_li.write(11 + i_s, 0, sc, sc_cell_fmts[sc])
                    ws_li.write(11 + i_s, 1, sc_annual, pct_fmt)
                    for m_i in range(1, months + 1):
                        ws_li.write(11 + i_s, 1 + m_i, idx_vals[m_i], sc_cell_fmts[sc])
                ws_li.set_column(0, 0, 22)
                ws_li.set_column(1, len(month_tags) + 2, 14)

        # ── Assumptions sheet ─────────────────────────────────────────────────
        pd.DataFrame().to_excel(writer, sheet_name="Assumptions", index=False)
        ws_a = writer.sheets["Assumptions"]
        ws_a.merge_range(0, 0, 0, 3, "Scenario Assumptions", title_fmt)
        ws_a.write(1, 0, "Scenario", hdr_fmt)
        ws_a.write(1, 1, "Commodity: Monthly % Chg", hdr_fmt)
        ws_a.write(1, 2, "FX: Monthly % Chg", hdr_fmt)
        ws_a.write(1, 3, "Description", hdr_fmt)
        descriptions = {
            "Strong Bull":   "Worst-case for Laos: commodity surge, strong USD, rapid LAK depreciation, CPI spike",
            "Moderate Bull": "Moderate commodity rise, mild LAK depreciation, manageable CPI pressure",
            "Base Case":     "Consensus / status-quo trajectory — BOL reference scenario",
            "Moderate Bear": "Commodity softening, partial LAK stabilisation, easing inflation outlook",
            "Strong Bear":   "Best-case for Laos: commodity decline, USD weakens, LAK firms, CPI relief",
        }
        for i, sc in enumerate(SCENARIOS):
            ws_a.write(2 + i, 0, sc, label_fmt)
            ws_a.write(2 + i, 1, pcts_comm.get(sc, DEFAULT_MONTHLY_PCT[sc]), pct_fmt)
            ws_a.write(2 + i, 2, pcts_fx.get(sc, DEFAULT_MONTHLY_PCT[sc]), pct_fmt)
            ws_a.write(2 + i, 3, descriptions[sc], label_fmt)

        ws_a.write(9, 0, "Forecast Horizon (months)", label_fmt)
        ws_a.write(9, 1, months, pct_fmt)
        ws_a.write(10, 0, "Generated on", label_fmt)
        ws_a.write(10, 1, TODAY.strftime("%d %B %Y %H:%M"), label_fmt)
        ws_a.set_column(0, 0, 30)
        ws_a.set_column(1, 2, 28)
        ws_a.set_column(3, 3, 45)

    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* ── Base layout ── */
        [data-testid="stAppViewContainer"] {
            background: #0F1923;
        }
        [data-testid="stHeader"] {
            background: #0F1923;
            border-bottom: 1px solid #1E3A5F;
        }
        .block-container { padding-top: 1.5rem; }

        /* ── Typography ── */
        h1, h2, h3 { color: #F5A623 !important; letter-spacing: 0.015em; }
        h4, h5     { color: #94A3B8 !important; text-transform: uppercase;
                     font-size: 0.78rem !important; letter-spacing: 0.08em; }
        p, li      { color: #CBD5E1; }

        /* ── Metric cards ── */
        div[data-testid="metric-container"],
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #162032 0%, #0D1927 100%);
            border: 1px solid #1E3A5F;
            border-top: 3px solid #F5A623;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.35);
        }
        div[data-testid="metric-container"] label,
        div[data-testid="stMetric"] label {
            color: #94A3B8 !important;
            font-size: 0.72rem !important;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.09em;
        }
        div[data-testid="metric-container"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #F5E642 !important;
            font-size: 1.55rem !important;
            font-weight: 700;
            font-family: 'Courier New', monospace;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            background: #0B1420;
            border-radius: 8px;
            padding: 4px;
            border: 1px solid #1E3A5F;
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.88rem;
            font-weight: 600;
            color: #94A3B8;
            border-radius: 6px;
            padding: 6px 14px;
        }
        .stTabs [aria-selected="true"] {
            background: #1E3A5F !important;
            color: #F5A623 !important;
        }

        /* ── Dataframe ── */
        [data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #1E3A5F !important;
        }
        [data-testid="stDataFrame"] th {
            background: #1E3A5F !important;
            color: #F5A623 !important;
            font-weight: 700 !important;
            font-size: 0.78rem !important;
        }
        [data-testid="stDataFrame"] td {
            color: #E2E8F0 !important;
            font-size: 0.84rem;
        }

        /* ── Expanders ── */
        [data-testid="stExpander"] {
            background: #162032;
            border: 1px solid #1E3A5F !important;
            border-radius: 8px;
        }
        [data-testid="stExpander"] summary {
            color: #94A3B8 !important;
            font-weight: 600;
        }

        /* ── Selectbox / number inputs ── */
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stNumberInput"] input {
            background: #162032 !important;
            color: #E2E8F0 !important;
            border: 1px solid #1E3A5F !important;
            border-radius: 6px;
        }

        /* ── Divider ── */
        hr { border: none; border-top: 1px solid #1E3A5F; }

        /* ── Alert / info boxes ── */
        [data-testid="stAlert"] {
            background: #0D2040 !important;
            border-color: #1E3A5F !important;
            color: #CBD5E1 !important;
        }

        /* ── Price cards (custom HTML) ── */
        .price-card {
            background: linear-gradient(135deg, #162032 0%, #0D1927 100%);
            border: 1px solid #1E3A5F;
            border-top: 3px solid #F5A623;
            border-radius: 10px;
            padding: 0.9rem 0.7rem;
            text-align: center;
            margin: 0.2rem 0 0.5rem 0;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .price-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(245,166,35,0.18);
        }
        .price-label {
            color: #94A3B8;
            font-size: 0.70rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            font-weight: 600;
            margin-bottom: 0.45rem;
        }
        .price-value-available {
            color: #F5E642;
            font-size: 1.4rem;
            font-weight: 700;
            font-family: 'Courier New', monospace;
            letter-spacing: 0.03em;
        }
        .price-value-unavailable {
            color: #F87171;
            font-size: 1rem;
            font-weight: 600;
        }
        .group-header {
            font-size: 0.82rem;
            font-weight: 700;
            color: #CBD5E1;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            border-left: 3px solid #F5A623;
            padding: 0.2rem 0 0.2rem 0.65rem;
            margin: 1rem 0 0.5rem 0;
        }
        .info-box {
            background: #0D2040;
            border: 1px solid #1E3A5F;
            border-left: 4px solid #60A5FA;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            color: #CBD5E1;
            font-size: 0.87rem;
            margin-top: 1rem;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar       { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0F1923; }
        ::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #F5A623; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR  – Settings & Scenario Assumptions
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ BOL/NSC Analysis Settings")
    st.caption(
        "Set the forecast horizon and scenario assumptions below. "
        "Changes apply immediately to all charts and tables."
    )

    forecast_months: int = st.slider("📅 Forecast horizon (months)", 1, 6, 3)

    st.divider()
    st.markdown("### 📦 Global Commodity Assumptions")
    st.caption(
        "Monthly % price change per scenario — applied to all 11 commodity imports. "
        "Energy (oil, gas) and agricultural goods are key Laos import cost drivers."
    )
    pcts_comm: dict[str, float] = {}
    for sc in SCENARIOS:
        pcts_comm[sc] = st.number_input(
            sc,
            value=DEFAULT_MONTHLY_PCT[sc],
            step=0.5,
            format="%.1f",
            key=f"comm_{sc}",
            help=f"Monthly compounded % change for {sc} applied to all commodity prices.",
        )

    st.divider()
    st.markdown("### 💱 Global FX Assumptions (EUR/USD, GBP/USD, USD/JPY)")
    st.caption(
        "EUR/USD strength affects Laos indirectly via import pricing in Thailand and Vietnam. "
        "USD/JPY movements influence global capital flows that reach ASEAN."
    )
    same_as_comm: bool = st.checkbox("Mirror commodity assumptions", value=True)
    pcts_fx: dict[str, float] = {}
    if same_as_comm:
        pcts_fx = pcts_comm.copy()
        for sc in SCENARIOS:
            color = SCENARIO_COLORS[sc]
            st.markdown(
                f"<span style='color:{color};font-weight:700;'>{sc}</span>: "
                f"`{pcts_comm[sc]:+.1f}%/mo`",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Monthly % change per scenario")
        for sc in SCENARIOS:
            pcts_fx[sc] = st.number_input(
                sc,
                value=DEFAULT_MONTHLY_PCT[sc],
                step=0.5,
                format="%.1f",
                key=f"fx_{sc}",
            )

    st.divider()
    if st.button("🔄 Refresh Live Data", width="stretch", type="secondary"):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "📡 Live data: Yahoo Finance (∼15-min delay). Front-month futures. "
        "Exotic tickers (USD/LAK) may need manual override — use BOL daily rate."
    )

    st.divider()
    st.markdown("### 🇱🇦 Laos-Specific Assumptions")
    st.caption(
        "Inputs below drive the 🇱🇦 Laos Focus tab: USD/LAK, LAK/THB cross rate, "
        "gold price in LAK, CPI inflation, and FX intervention cost modelling."
    )
    current_inflation: float = st.number_input(
        "Current annual Laos CPI inflation (%)",
        min_value=0.0,
        max_value=200.0,
        value=LAK_DEFAULT_INFLATION_PCT,
        step=0.5,
        format="%.1f",
        key="lak_inflation",
        help="Laos latest official annual CPI inflation rate. Set manually from BOL/NSC data.",
    )
    with st.expander("🔧 LAK Depreciation per Scenario", expanded=False):
        st.caption("Monthly % change in LAK · (+) = LAK weakens (more LAK per USD)")
        pcts_lak: dict[str, float] = {}
        for sc in SCENARIOS:
            pcts_lak[sc] = st.number_input(
                sc,
                value=LAK_DEFAULT_MONTHLY_PCT[sc],
                step=0.25,
                format="%.2f",
                key=f"lak_{sc}",
                help=f"+ means LAK weakens (more LAK per USD/THB) for {sc}.",
            )
    with st.expander("🔧 Inflation Adjustment per Scenario", expanded=False):
        st.caption("Annual percentage-point (pp) added to base inflation rate")
        infl_adj: dict[str, float] = {}
        for sc in SCENARIOS:
            infl_adj[sc] = st.number_input(
                sc,
                value=LAK_INFL_SCENARIO_ADJ[sc],
                step=0.5,
                format="%.1f",
                key=f"infl_{sc}",
                help=f"pp added to base CPI inflation for {sc} scenario.",
            )

    st.divider()
    st.markdown("### 🛡️ FX Intervention & Oil")
    oil_cpi_beta_sidebar: float = st.number_input(
        "Oil → CPI pass-through fraction",
        min_value=0.0, max_value=1.0,
        value=OIL_CPI_BETA_DEFAULT,
        step=0.05, format="%.2f",
        key="oil_cpi_beta",
        help="Fraction of an oil price % change that passes through to Laos CPI. "
             "e.g. 0.30 = 10% oil rise → +3 pp CPI.",
    )
    eur_cpi_beta_sidebar: float = st.number_input(
        "EUR/USD → CPI (Thailand import channel)",
        min_value=0.0, max_value=1.0,
        value=EUR_CPI_BETA_DEFAULT,
        step=0.05, format="%.2f",
        key="eur_cpi_beta",
        help="When EUR/USD rises (USD weakens), Thai & Vietnamese goods imported via Europe "
             "get costlier → Laos CPI rises. e.g. 0.15 = 10% EUR/USD rise → +1.5 pp CPI.",
    )
    eur_lak_beta_sidebar: float = st.number_input(
        "EUR/USD → USD/LAK beta (inverse)",
        min_value=0.0, max_value=3.0,
        value=EUR_LAK_BETA_DEFAULT,
        step=0.05, format="%.2f",
        key="eur_lak_beta",
        help="How much USD/LAK rises (LAK weakens) when EUR/USD falls 1% (USD strengthens). "
             "e.g. 0.80 = EUR/USD −1% → USD/LAK +0.8%. Typical frontier EM range: 0.5–1.2. "
             "Set to 0 to disable the linkage.",
    )
    with st.expander("📊 THB Monthly % per Scenario (USD/THB)", expanded=False):
        st.caption(
            "+ = THB weakens vs USD (more THB per dollar).\n"
            "THB is more stable than LAK — correlates with USD but smaller moves."
        )
        pcts_thb: dict[str, float] = {}
        for sc in SCENARIOS:
            pcts_thb[sc] = st.number_input(
                sc,
                value=THB_DEFAULT_MONTHLY_PCT[sc],
                step=0.1, format="%.2f",
                key=f"thb_pct_{sc}",
                help=f"Monthly % change for USD/THB under {sc}. "
                     f"(+) = THB depreciates vs USD, (-) = THB appreciates.",
            )
    with st.expander("🔧 FX Intervention Parameters", expanded=False):
        st.caption("— USD/LAK Channel —")
        bol_reserves: float = st.number_input(
            "BOL USD Reserves (mn)",
            min_value=0.0, max_value=50_000.0,
            value=BOL_RESERVES_DEFAULT_USD_MN,
            step=50.0, format="%.0f",
            key="bol_reserves",
            help="Bank of Laos USD foreign currency reserves in USD million.",
        )
        daily_fx_vol_mn: float = st.number_input(
            "Daily USD/LAK Market Volume (USD mn)",
            min_value=0.1, max_value=1_000.0,
            value=DAILY_FX_VOLUME_DEFAULT_USD_MN,
            step=0.5, format="%.1f",
            key="daily_fx_vol",
            help="Estimated daily USD-LAK market turnover in Laos.",
        )
        target_max_dep: float = st.number_input(
            "Target max LAK/USD dep (%/mo)",
            min_value=0.0, max_value=20.0,
            value=TARGET_MAX_DEP_DEFAULT,
            step=0.25, format="%.2f",
            key="target_max_dep",
            help="Maximum monthly LAK depreciation vs USD BOL wants to allow.",
        )
        intv_effect: float = st.number_input(
            "Intervention effectiveness (%)",
            min_value=1.0, max_value=100.0,
            value=FX_INTV_EFFECTIVENESS_DEFAULT,
            step=5.0, format="%.0f",
            key="intv_effect",
            help="Fraction of each $1 intervention that actually absorbs market pressure. "
                 "50% = $2 needed to absorb $1 of excess demand.",
        )
        st.divider()
        st.caption("— LAK/THB Channel (THB Reserves) —")
        bol_thb_reserves: float = st.number_input(
            "BOL THB Reserves (USD equiv mn)",
            min_value=0.0, max_value=10_000.0,
            value=BOL_THB_RESERVES_DEFAULT_USD_MN,
            step=10.0, format="%.0f",
            key="bol_thb_reserves",
            help="BOL's THB buffer used to defend LAK/THB rate (stated in USD equivalent).",
        )
        daily_thb_vol_mn: float = st.number_input(
            "Daily LAK/THB Market Vol (USD equiv mn)",
            min_value=0.1, max_value=500.0,
            value=DAILY_THB_VOL_DEFAULT_USD_MN,
            step=0.5, format="%.1f",
            key="daily_thb_vol",
            help="Estimated daily LAK-THB cross market turnover (USD equivalent).",
        )
        target_lak_thb_dep: float = st.number_input(
            "Target max LAK/THB dep (%/mo)",
            min_value=0.0, max_value=20.0,
            value=TARGET_LAK_THB_DEP_DEFAULT,
            step=0.25, format="%.2f",
            key="target_lak_thb_dep",
            help="Maximum monthly LAK depreciation vs THB BOL wants to allow.",
        )


# ─────────────────────────────────────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"<h2 style='color:#F5A623;margin-bottom:0.1rem;'>"
    f"🇱🇦 Laos FX &amp; Import Price Forecast — Policy Dashboard</h2>"
    f"<p style='color:#94A3B8;font-size:0.88rem;margin-top:0.1rem;'>"
    f"Bank of Laos (BOL) · FX Policy Division &nbsp;&nbsp;"
    f"<span style='color:#1E3A5F;'>|</span>&nbsp;"
    f"<span style='color:#F5E642;font-weight:600;'>{TODAY.strftime('%d %B %Y')}</span>"
    f"&nbsp;·&nbsp; <span style='color:#36D399;'>● LIVE</span></p>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
#  FETCH LIVE PRICES
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("⏳ Fetching live market data from Yahoo Finance…"):
    comm_prices: dict = fetch_prices(COMMODITY_TICKERS)
    fx_prices: dict   = fetch_prices(FX_TICKERS)

# ── Derived LAK metrics ───────────────────────────────────────────────────────
_lak_usd  = fx_prices.get("USD/LAK")    # LAK per 1 USD
_thb_usd  = fx_prices.get("USD/THB")    # THB per 1 USD
_gold_usd = comm_prices.get("Gold ($/oz)")
lak_thb_rate   = round(_lak_usd / _thb_usd, 4) if (_lak_usd and _thb_usd)  else None
gold_lak_price = round(_gold_usd * _lak_usd, 0) if (_gold_usd and _lak_usd) else None

n_comm = sum(1 for v in comm_prices.values() if v is not None)
n_fx   = sum(1 for v in fx_prices.values()   if v is not None)

# Summary strip
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Commodity Prices", f"{n_comm} / {len(COMMODITY_TICKERS)}", help="Loaded successfully")
m2.metric("FX Pairs",         f"{n_fx} / {len(FX_TICKERS)}",          help="Loaded successfully")
m3.metric("USD / LAK Rate",   f"{_lak_usd:,.0f}" if _lak_usd else "Manual", help="Lao Kip per 1 USD")
m4.metric("Forecast Horizon", f"{forecast_months} month{'s' if forecast_months > 1 else ''}")
m5.metric("Scenarios",        "5")
m6.metric("Last Refresh",     TODAY.strftime("%H:%M"))

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_live, tab_comm, tab_fx, tab_lak, tab_summary, tab_export, tab_help = st.tabs(
    [
        "📈 Live Prices",
        "📦 Commodity Scenarios",
        "💱 FX Scenarios",
        "🇱🇦 Laos Focus",
        "📋 Summary View",
        "📥 Export to Excel",
        "📖 User Guide",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – LIVE PRICES
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown(
        "<p style='color:#94A3B8;font-size:0.84rem;margin-bottom:0.5rem;'>"
        "Live market rates from Yahoo Finance (~15-min delay) — front-month futures. "
        "These are the <b style='color:#F5E642;'>global benchmark prices</b> whose movements "
        "feed directly into Laos import costs, LAK purchasing power, and domestic inflation. "
        "See the 🇱🇦 <b>Laos Focus</b> tab for LAK-translated forecasts and policy impact analysis."
        "</p>",
        unsafe_allow_html=True,
    )

    GROUP_ACCENT = {"Energy": "#F5A623", "Metals": "#FFD700", "Agricultural": "#36D399"}
    GROUP_ICON   = {"Energy": "⚡", "Metals": "🏅", "Agricultural": "🌾"}

    for group, assets in COMMODITY_GROUPS.items():
        accent = GROUP_ACCENT.get(group, "#60A5FA")
        icon   = GROUP_ICON.get(group, "📦")
        st.markdown(
            f"<div class='group-header' style='border-left-color:{accent};'>"
            f"{icon} {group}</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(len(assets))
        for i, asset in enumerate(assets):
            price = comm_prices.get(asset)
            with cols[i]:
                if price is not None:
                    val_html = f"<div class='price-value-available'>{price:,.4f}</div>"
                else:
                    val_html = "<div class='price-value-unavailable'>N/A</div>"
                st.markdown(
                    f"<div class='price-card' style='border-top-color:{accent};'>"
                    f"<div class='price-label'>{asset}</div>{val_html}</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E3A5F;margin:0.5rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='group-header' style='border-left-color:#60A5FA;'>💱 FX Rates</div>",
        unsafe_allow_html=True,
    )
    fx_cols = st.columns(len(FX_TICKERS))
    for i, (pair, price) in enumerate(fx_prices.items()):
        with fx_cols[i]:
            if price is not None:
                val_html = f"<div class='price-value-available'>{price:,.4f}</div>"
            else:
                val_html = "<div class='price-value-unavailable'>N/A</div>"
            st.markdown(
                f"<div class='price-card' style='border-top-color:#60A5FA;'>"
                f"<div class='price-label'>{pair}</div>{val_html}</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div class='info-box'>"
        "<b style='color:#60A5FA;'>ℹ️ FX Rate Convention &amp; Laos Policy Relevance</b><br>"
        "<b>EUR/USD &amp; GBP/USD</b> = quoted as USD per 1 foreign unit. "
        "A rise means the USD is <i>weaker</i>; this makes oil (priced in USD) dearer in USD terms, "
        "but also makes EU-origin imports cheaper in LAK (if LAK tracks USD). "
        "Key channel: EUR/USD rise → Thailand &amp; Vietnam import costs rise → Laos CPI rises.<br>"
        "<b>USD/THB &amp; USD/LAK</b> = foreign units per USD (+ scenario → THB/LAK weakens vs USD). "
        "LAK weakening = all USD-denominated imports get costlier in LAK. "
        "BOL intervenes in both USD and THB markets to stabilise LAK purchasing power.<br>"
        "<b>LAK/THB cross rate</b> and <b>Gold (LAK/oz)</b> are derived — see the 🇱🇦 <b>Laos Focus</b> tab."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – COMMODITY SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_comm:
    valid_comm = {k: v for k, v in comm_prices.items() if v is not None}

    if not valid_comm:
        st.error("No commodity prices loaded. Check your connection and refresh.")
    else:
        st.subheader(f"📦 Global Commodity Scenarios — {forecast_months}-Month Horizon")
        st.caption(
            "Global commodity prices drive Laos import costs directly. "
            "Oil (Brent/WTI) is the most critical: Laos imports fuel via Thailand and Vietnam, "
            "so a 10% oil rise typically adds 2–4 pp to Laos CPI within 1–2 months. "
            "Gold and agricultural prices affect household purchasing power and food security. "
            "Adjust assumptions in the sidebar or use Oil Comparison below for CPI impact."
        )

        # ── Scenario table ──────────────────────────────────────────────────────
        st.markdown("##### Projected Prices — All Commodities × All Scenarios")
        df_comm = scenario_table(valid_comm, forecast_months, pcts_comm)

        # Colour-code scenario columns
        def style_comm(df: pd.DataFrame):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for col in df.columns:
                if col == "Current Price":
                    styles[col] = (
                        "background-color: #0D2040; color: #F5E642; "
                        "font-weight: 700; font-family: 'Courier New', monospace;"
                    )
                else:
                    for sc, bg in SCENARIO_BG.items():
                        if col.startswith(sc):
                            styles[col] = (
                                f"background-color: {bg}; color: #E2E8F0; "
                                "font-family: 'Courier New', monospace;"
                            )
                            break
            return styles

        st.dataframe(
            df_comm.style.apply(style_comm, axis=None).format("{:,.4f}"),
            width="stretch",
            height=min(60 + len(valid_comm) * 36, 500),
        )

        st.divider()

        # ── Price path chart ──────────────────────────────────────────────────
        st.markdown("##### Scenario Price Path")
        sel_comm = st.selectbox(
            "Select commodity to chart",
            list(valid_comm.keys()),
            key="sel_comm_chart",
        )
        if sel_comm:
            fig = price_path_chart(
                sel_comm, valid_comm[sel_comm], forecast_months, pcts_comm
            )
            st.plotly_chart(fig, width="stretch")

            # Show the projection table for selected asset
            with st.expander(f"📋 Detailed projection for {sel_comm}"):
                df_single = project(valid_comm[sel_comm], forecast_months, 0)
                x_labels = ["Now"] + [
                    (TODAY + timedelta(days=30 * m)).strftime("%b %Y")
                    for m in range(1, forecast_months + 1)
                ]
                detail_rows = []
                for sc in SCENARIOS:
                    pct = pcts_comm.get(sc, DEFAULT_MONTHLY_PCT[sc])
                    vals = project(valid_comm[sel_comm], forecast_months, pct)
                    row = {"Scenario": sc, "Monthly % Chg": f"{pct:+.1f}%"}
                    for i, label in enumerate(x_labels):
                        row[label] = f"{vals[i]:,.4f}"
                    detail_rows.append(row)
                st.dataframe(pd.DataFrame(detail_rows).set_index("Scenario"), width="stretch")

        st.divider()

        # ── Oil Price Scenario Comparison ────────────────────────────────────────
        st.markdown(
            "<div class='group-header' style='border-left-color:#F5A623;'>"
            "🛢️ Oil Price — Scenario Comparison & Laos Inflation Impact</div>",
            unsafe_allow_html=True,
        )
        oil_assets = [a for a in ["Brent Crude ($/bbl)", "WTI Crude ($/bbl)"] if a in valid_comm]

        if oil_assets:
            oc_left, oc_right = st.columns([3, 2])

            with oc_left:
                # Grouped bar: end-of-horizon price per scenario, one bar per oil type
                fig_oil = go.Figure()
                for oi, oa in enumerate(oil_assets):
                    end_prices = [
                        project(valid_comm[oa], forecast_months,
                                pcts_comm.get(sc, DEFAULT_MONTHLY_PCT[sc]))[-1]
                        for sc in SCENARIOS
                    ]
                    short = "Brent" if "Brent" in oa else "WTI"
                    fig_oil.add_trace(go.Bar(
                        x=SCENARIOS, y=end_prices,
                        name=short,
                        text=[f"${v:.1f}" for v in end_prices],
                        textposition="outside",
                        textfont=dict(color="#E2E8F0", size=10),
                        marker_color=[SCENARIO_COLORS[sc] for sc in SCENARIOS],
                        marker_line_color="#0F1923", marker_line_width=1.5,
                        opacity=1.0 if oi == 0 else 0.55,
                        marker_pattern_shape="" if oi == 0 else "/",
                    ))
                fig_oil.update_layout(
                    title=dict(
                        text=f"<b>Oil End-Horizon Price — Month {forecast_months}</b>",
                        font_size=13, font_color="#F5A623",
                    ),
                    barmode="group",
                    yaxis_title="USD / bbl",
                    template="plotly_dark",
                    paper_bgcolor="#0F1923", plot_bgcolor="#0B1420",
                    font=dict(color="#CBD5E1"), height=360,
                    margin=dict(t=60, b=60, l=65, r=20),
                    legend=dict(font_color="#CBD5E1", bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(gridcolor="#1E3A5F", tickfont=dict(color="#94A3B8", size=10)),
                    yaxis=dict(gridcolor="#1E3A5F", tickfont=dict(color="#94A3B8")),
                )
                st.plotly_chart(fig_oil, width="stretch")

                # Divergence: % change from current price per scenario
                fig_div = go.Figure()
                for sc in SCENARIOS:
                    pct = pcts_comm.get(sc, DEFAULT_MONTHLY_PCT[sc])
                    div_vals = [
                        ((project(valid_comm[oa], forecast_months, pct)[-1]
                          - valid_comm[oa]) / valid_comm[oa]) * 100
                        for oa in oil_assets
                    ]
                    avg_div = float(np.mean(div_vals))
                    fig_div.add_trace(go.Bar(
                        x=[sc], y=[avg_div],
                        name=sc,
                        marker_color=SCENARIO_COLORS[sc],
                        marker_line_color="#0F1923", marker_line_width=1.5,
                        text=[f"{avg_div:+.1f}%"],
                        textposition="outside",
                        textfont=dict(color=SCENARIO_COLORS[sc], size=11),
                        showlegend=False,
                    ))
                fig_div.add_hline(y=0, line_color="#64748B", line_width=1)
                fig_div.update_layout(
                    title=dict(
                        text="<b>Avg Oil % Change vs Current</b> (Brent + WTI)",
                        font_size=13, font_color="#F5A623",
                    ),
                    yaxis_title="% change from today",
                    template="plotly_dark",
                    paper_bgcolor="#0F1923", plot_bgcolor="#0B1420",
                    font=dict(color="#CBD5E1"), height=280,
                    margin=dict(t=55, b=55, l=65, r=20),
                    xaxis=dict(gridcolor="#1E3A5F", tickfont=dict(color="#94A3B8", size=10)),
                    yaxis=dict(gridcolor="#1E3A5F", tickfont=dict(color="#94A3B8")),
                    showlegend=False,
                )
                st.plotly_chart(fig_div, width="stretch")

            with oc_right:
                st.markdown(
                    "<p style='color:#94A3B8;font-size:0.78rem;font-weight:700;"
                    "text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.4rem;'>"
                    "Scenario vs Today + Laos CPI Impact</p>",
                    unsafe_allow_html=True,
                )
                oil_tbl_rows = []
                for sc in SCENARIOS:
                    pct   = pcts_comm.get(sc, DEFAULT_MONTHLY_PCT[sc])
                    row_o = {"Scenario": sc}
                    chg_list = []
                    for oa in oil_assets:
                        end_p = project(valid_comm[oa], forecast_months, pct)[-1]
                        chg   = ((end_p - valid_comm[oa]) / valid_comm[oa]) * 100
                        short = "Brent" if "Brent" in oa else "WTI"
                        row_o[f"{short} ($/bbl)"]  = f"{end_p:.2f}"
                        row_o[f"{short} % Chg"]    = f"{chg:+.1f}%"
                        chg_list.append(chg)
                    avg_chg_pct  = float(np.mean(chg_list))
                    cpi_impact   = avg_chg_pct * oil_cpi_beta_sidebar
                    row_o["LAO CPI Impact (pp)"] = f"{cpi_impact:+.2f} pp"
                    oil_tbl_rows.append(row_o)

                df_oil = pd.DataFrame(oil_tbl_rows).set_index("Scenario")

                def style_oil(df: pd.DataFrame):
                    s = pd.DataFrame("", index=df.index, columns=df.columns)
                    for sc_n in df.index:
                        bg = SCENARIO_BG.get(sc_n, "#162032")
                        for col in df.columns:
                            s.loc[sc_n, col] = (
                                f"background-color:{bg};color:#E2E8F0;"
                                "font-family:'Courier New',monospace;font-size:0.8rem;"
                            )
                    return s

                st.dataframe(
                    df_oil.style.apply(style_oil, axis=None),
                    width="stretch", height=250,
                )
                st.markdown(
                    f"<div class='info-box' style='margin-top:0.6rem;'>"
                    f"<b style='color:#F5A623;'>🛢️ → 🇱🇦 CPI Pass-through</b><br>"
                    f"Oil basket weight: <b style='color:#F5E642;'>"
                    f"{oil_cpi_beta_sidebar*100:.0f}%</b>.<br>"
                    f"10% oil rise → "
                    f"<b style='color:#F5E642;'>{oil_cpi_beta_sidebar*10:.1f} pp</b> "
                    f"additional Laos CPI pressure.<br>"
                    f"<small style='color:#64748B;'>Adjust <i>Oil→CPI pass-through</i> "
                    f"fraction in sidebar.</small></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("🛢️ No Brent / WTI price data loaded. Refresh to enable this section.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 – FX SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_fx:
    valid_fx = {k: v for k, v in fx_prices.items() if v is not None}

    if not valid_fx:
        st.error("No FX data loaded. Check your connection and refresh.")
    else:
        st.subheader(f"💱 External FX Scenarios — {forecast_months}-Month Horizon")
        st.caption(
            "⚠️ "
            "EUR/USD &amp; GBP/USD: percentage change in the quoted rate — "
            "a + scenario means USD weakens (EUR/GBP buys more USD). "
            "USD/THB: + means THB weakens vs USD — impacts Laos because ~60% of Laos imports "
            "come via Thailand, priced in THB. "
            "USD/LAK: see the 🇱🇦 Laos Focus tab for full LAK analysis and BOL intervention modelling."
        )

        # ── Scenario table ────────────────────────────────────────────────────
        df_fx = scenario_table(valid_fx, forecast_months, pcts_fx)

        def style_fx(df: pd.DataFrame):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for col in df.columns:
                if col == "Current Price":
                    styles[col] = (
                        "background-color: #0D2040; color: #F5E642; "
                        "font-weight: 700; font-family: 'Courier New', monospace;"
                    )
                else:
                    for sc, bg in SCENARIO_BG.items():
                        if col.startswith(sc):
                            styles[col] = (
                                f"background-color: {bg}; color: #E2E8F0; "
                                "font-family: 'Courier New', monospace;"
                            )
                            break
            return styles

        st.dataframe(
            df_fx.style.apply(style_fx, axis=None).format("{:,.4f}"),
            width="stretch",
            height=min(60 + len(valid_fx) * 36, 300),
        )

        st.divider()

        # ── Rate path chart ───────────────────────────────────────────────────
        st.markdown("##### Scenario Rate Path")
        sel_fx = st.selectbox(
            "Select currency pair to chart",
            list(valid_fx.keys()),
            key="sel_fx_chart",
        )
        if sel_fx:
            fig = price_path_chart(
                sel_fx, valid_fx[sel_fx], forecast_months, pcts_fx
            )
            st.plotly_chart(fig, width="stretch")

            with st.expander(f"📋 Detailed projection for {sel_fx}"):
                x_labels = ["Now"] + [
                    (TODAY + timedelta(days=30 * m)).strftime("%b %Y")
                    for m in range(1, forecast_months + 1)
                ]
                detail_rows = []
                for sc in SCENARIOS:
                    pct = pcts_fx.get(sc, DEFAULT_MONTHLY_PCT[sc])
                    vals = project(valid_fx[sel_fx], forecast_months, pct)
                    row = {"Scenario": sc, "Monthly % Chg": f"{pct:+.1f}%"}
                    for i, label in enumerate(x_labels):
                        row[label] = f"{vals[i]:,.4f}"
                    detail_rows.append(row)
                st.dataframe(pd.DataFrame(detail_rows).set_index("Scenario"), width="stretch")

        st.divider()

        # ── EUR/USD → USD/LAK Linkage Analysis ─────────────────────────────────────
        st.markdown(
            "<div class='group-header' style='border-left-color:#A78BFA;'>"
            "🔗 EUR/USD ↔ USD/LAK Linkage — When USD Strengthens, LAK Weakens</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#94A3B8;font-size:0.83rem;margin:0.2rem 0 0.8rem;'>"
            "EUR/USD and USD/LAK move <b style='color:#F87171;'>inversely</b>: "
            "when EUR/USD falls (USD strengthens), capital flows to USD safe-haven assets, "
            "putting depreciation pressure on LAK → USD/LAK rises. "
            "Adjust the <b>EUR/USD → USD/LAK beta</b> in the sidebar.</p>",
            unsafe_allow_html=True,
        )

        _eur_base = valid_fx.get("EUR/USD")
        _lak_base = _lak_usd or 21_500.0   # live rate; _lak_usd_eff defined later in tab_lak

        if _eur_base and _lak_base:
            lk_left, lk_right = st.columns([3, 2])

            with lk_left:
                x_lk = ["Now"] + [
                    (TODAY + timedelta(days=30 * m)).strftime("%b %Y")
                    for m in range(1, forecast_months + 1)
                ]
                fig_link = go.Figure()

                # ─ EUR/USD path (left y-axis) ─────────────────────────────────────
                for sc in SCENARIOS:
                    eur_mo = pcts_fx.get(sc, DEFAULT_MONTHLY_PCT[sc])
                    y_eur  = project(_eur_base, forecast_months, eur_mo)
                    fig_link.add_trace(go.Scatter(
                        x=x_lk, y=y_eur,
                        mode="lines+markers",
                        name=f"EUR/USD {sc}",
                        yaxis="y1",
                        opacity=0.85,
                        line=dict(color=SCENARIO_COLORS[sc], dash=SCENARIO_DASH[sc], width=2),
                        marker=dict(size=5, color=SCENARIO_COLORS[sc]),
                        hovertemplate=(
                            f"<b>EUR/USD — {sc}</b><br>"
                            "%{x}: %{y:.4f}<extra></extra>"
                        ),
                        showlegend=True,
                        legendgroup=sc,
                    ))

                # ─ Derived USD/LAK path (right y-axis) ───────────────────────────
                for sc in SCENARIOS:
                    eur_mo = pcts_fx.get(sc, DEFAULT_MONTHLY_PCT[sc])
                    # Inverse: EUR/USD falls X%/mo → USD/LAK rises beta*X %/mo
                    lak_linked_mo = -eur_mo * eur_lak_beta_sidebar
                    y_lak = project(_lak_base, forecast_months, lak_linked_mo)
                    txt_lk = [""] * len(y_lak)
                    txt_lk[-1] = f"  {y_lak[-1]:,.0f}"
                    fig_link.add_trace(go.Scatter(
                        x=x_lk, y=y_lak,
                        mode="lines+markers+text",
                        name=f"USD/LAK (linked) {sc}",
                        yaxis="y2",
                        text=txt_lk,
                        textposition="middle right",
                        textfont=dict(color=SCENARIO_COLORS[sc], size=10,
                                      family="Courier New, monospace"),
                        line=dict(color=SCENARIO_COLORS[sc], dash="dot",
                                  width=2.5),
                        marker=dict(size=6, color=SCENARIO_COLORS[sc],
                                    symbol="square"),
                        hovertemplate=(
                            f"<b>USD/LAK linked — {sc}</b><br>"
                            f"%{{x}}: %{{y:,.0f}} LAK "
                            f"(EUR/USD à +{lak_linked_mo:+.2f}%/mo)<extra></extra>"
                        ),
                        showlegend=True,
                        legendgroup=sc,
                    ))

                fig_link.update_layout(
                    title=dict(
                        text=(
                            "<b>EUR/USD ↓ &nbsp;⟹&nbsp; USD/LAK ↑</b> — "
                            f"Inverse linkage (beta = {eur_lak_beta_sidebar:.2f})<br>"
                            "<sup><span style='color:#60A5FA;'>solid = EUR/USD</span> &nbsp;·&nbsp; "
                            "<span style='color:#A78BFA;'>dotted = derived USD/LAK (right axis)</span></sup>"
                        ),
                        font_size=13, font_color="#F5A623",
                    ),
                    xaxis=dict(gridcolor="#1E3A5F", tickfont=dict(color="#94A3B8")),
                    yaxis=dict(
                        title=dict(text="EUR/USD rate", font=dict(color="#60A5FA")),
                        tickfont=dict(color="#60A5FA"),
                        gridcolor="#1E3A5F",
                    ),
                    yaxis2=dict(
                        title=dict(text="USD/LAK (linked)", font=dict(color="#A78BFA")),
                        tickfont=dict(color="#A78BFA"),
                        tickformat=",.0f",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                    ),
                    template="plotly_dark",
                    paper_bgcolor="#0F1923", plot_bgcolor="#0B1420",
                    font=dict(color="#CBD5E1"), height=440,
                    margin=dict(t=90, b=55, l=65, r=100),
                    hovermode="x unified",
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1,
                        font_size=9, bgcolor="rgba(0,0,0,0)",
                        font_color="#CBD5E1",
                    ),
                )
                st.plotly_chart(fig_link, width="stretch")

            with lk_right:
                st.markdown(
                    "<p style='color:#94A3B8;font-size:0.78rem;font-weight:700;"
                    "text-transform:uppercase;letter-spacing:0.07em;"
                    "margin-bottom:0.4rem;'>Derived USD/LAK per Scenario</p>",
                    unsafe_allow_html=True,
                )
                link_rows = []
                for sc in SCENARIOS:
                    eur_mo        = pcts_fx.get(sc, DEFAULT_MONTHLY_PCT[sc])
                    lak_linked_mo = -eur_mo * eur_lak_beta_sidebar
                    eur_end       = project(_eur_base, forecast_months, eur_mo)[-1]
                    lak_end       = project(_lak_base, forecast_months, lak_linked_mo)[-1]
                    eur_chg_pct   = (eur_end / _eur_base - 1) * 100
                    lak_chg_pct   = (lak_end / _lak_base  - 1) * 100
                    direction_lak = "↑ LAK weakens" if lak_chg_pct > 0 else "↓ LAK firms"
                    link_rows.append({
                        "Scenario":              sc,
                        "EUR/USD %/mo":         f"{eur_mo:+.1f}%",
                        f"EUR/USD M{forecast_months}": f"{eur_end:.4f} ({eur_chg_pct:+.1f}%)",
                        "USD/LAK %/mo (linked)": f"{lak_linked_mo:+.2f}%",
                        f"USD/LAK M{forecast_months}": f"{lak_end:,.0f} ({lak_chg_pct:+.1f}%)",
                        "LAK Direction":         direction_lak,
                    })

                df_link = pd.DataFrame(link_rows).set_index("Scenario")

                def _style_link(df: pd.DataFrame):
                    s = pd.DataFrame("", index=df.index, columns=df.columns)
                    for sc_n in df.index:
                        bg = SCENARIO_BG.get(sc_n, "#162032")
                        for col in df.columns:
                            s.loc[sc_n, col] = (
                                f"background-color:{bg};color:#E2E8F0;"
                                "font-family:'Courier New',monospace;font-size:0.78rem;"
                            )
                    return s

                st.dataframe(
                    df_link.style.apply(_style_link, axis=None),
                    width="stretch", height=270,
                )
                st.markdown(
                    f"<div class='info-box' style='margin-top:0.5rem;border-left-color:#A78BFA;'>"
                    f"<b style='color:#A78BFA;'>🔗 Inverse Linkage</b><br>"
                    f"Formula: <code>USD/LAK %/mo = −EUR/USD %/mo × {eur_lak_beta_sidebar:.2f}</code><br><br>"
                    f"<b style='color:#F5A623;'>EUR/USD ↓ (USD strong)</b> → "
                    f"<b style='color:#F87171;'>USD/LAK ↑ (LAK weakens)</b><br>"
                    f"<b style='color:#36D399;'>EUR/USD ↑ (USD weak)</b> → "
                    f"<b style='color:#36D399;'>USD/LAK ↓ (LAK firms)</b><br><br>"
                    f"<small style='color:#64748B;'>This is a <i>derived</i> estimate based on "
                    f"the beta linkage. Actual USD/LAK also depends on Laos-specific factors "
                    f"(trade balance, BOL intervention, domestic inflation). "
                    f"Compare with the standalone USD/LAK assumptions in the "
                    f"🇱🇦 Laos Focus tab.</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                "🔗 EUR/USD or USD/LAK data not loaded. "
                "Refresh data or enter USD/LAK manually in the 🇱🇦 Laos Focus tab."
            )
# ══════════════════════════════════════════════════════════════════════════════
with tab_lak:
    st.markdown(
        "<h3 style='color:#F5A623;margin-bottom:0.1rem;'>🇱🇦 Laos / LAK Focus</h3>"
        "<p style='color:#94A3B8;font-size:0.84rem;'>"
        "USD/LAK live rate · LAK/THB cross rate · Gold price in LAK · "
        "Laos CPI inflation scenarios — adjust assumptions in the sidebar</p>",
        unsafe_allow_html=True,
    )

    # ── Headline KPI cards ────────────────────────────────────────────────────
    def _lak_card(col, title: str, value, fmt: str, accent: str, note: str = "") -> None:
        vhtml = (
            f"<div class='price-value-available'>{value:{fmt}}</div>"
            if value is not None
            else "<div class='price-value-unavailable'>N/A — enter below</div>"
        )
        col.markdown(
            f"<div class='price-card' style='border-top-color:{accent};'>"
            f"<div class='price-label'>{title}</div>"
            f"{vhtml}"
            f"<div style='color:#64748B;font-size:0.65rem;margin-top:0.3rem;'>{note}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    hc1, hc2, hc3, hc4 = st.columns(4)
    _lak_card(hc1, "USD / LAK",          _lak_usd,        ",.0f",  "#F5A623", "LAK per 1 USD · Yahoo Finance")
    _lak_card(hc2, "LAK / THB (cross)",  lak_thb_rate,    ",.2f",  "#36D399", "Derived: USD/LAK ÷ USD/THB")
    _lak_card(hc3, "Gold (LAK / oz)",    gold_lak_price,  ",.0f",  "#FFD700", "Derived: Gold($/oz) × USD/LAK")
    _lak_card(hc4, "Laos Inflation",     current_inflation, ".1f", "#F87171", "Annual CPI % · manual input")

    st.divider()

    # ── Manual rate override if Yahoo Finance returns None for LAK ────────────
    if _lak_usd is None:
        st.warning(
            "⚠️ **USD/LAK live rate unavailable** from Yahoo Finance (exotic currency).  "
            "Enter the current rate manually below to enable all LAK projections.",
        )
        _manual_lak = st.number_input(
            "Enter current USD/LAK rate (LAK per 1 USD)",
            min_value=1_000.0, max_value=999_999.0,
            value=21_500.0, step=100.0, format="%.0f",
            key="manual_lak_rate",
        )
        _lak_usd_eff    = _manual_lak
        _thb_eff        = _thb_usd or 34.5
        lak_thb_eff     = round(_lak_usd_eff / _thb_eff, 4)
        gold_lak_eff    = round(_gold_usd * _lak_usd_eff, 0) if _gold_usd else None
    else:
        _lak_usd_eff = _lak_usd
        lak_thb_eff  = lak_thb_rate
        gold_lak_eff = gold_lak_price

    x_lak = ["Now"] + [
        (TODAY + timedelta(days=30 * m)).strftime("%b %Y")
        for m in range(1, forecast_months + 1)
    ]

    # ══ 1. USD/LAK Scenario Chart ═════════════════════════════════════════════
    st.markdown(
        "<div class='group-header' style='border-left-color:#F5A623;'>"
        "💱 USD/LAK — 5-Scenario Forecast</div>",
        unsafe_allow_html=True,
    )
    if _lak_usd_eff:
        fig_lak = price_path_chart(
            "USD/LAK (LAK per USD)", _lak_usd_eff, forecast_months, pcts_lak
        )
        st.plotly_chart(fig_lak, width="stretch")

        with st.expander("📋 USD/LAK Detailed Projection Table"):
            lak_rows = []
            for sc in SCENARIOS:
                pct = pcts_lak.get(sc, LAK_DEFAULT_MONTHLY_PCT[sc])
                vals = project(_lak_usd_eff, forecast_months, pct)
                row = {
                    "Scenario": sc,
                    "Monthly % Chg": f"{pct:+.2f}%",
                    "Direction": "LAK weakens ↑" if pct > 0 else ("LAK firms ↓" if pct < 0 else "Stable"),
                }
                for i_l, lbl in enumerate(x_lak):
                    row[lbl] = f"{vals[i_l]:,.0f}"
                lak_rows.append(row)
            st.dataframe(pd.DataFrame(lak_rows).set_index("Scenario"), width="stretch")

    st.divider()

    # ══ 2. LAK/THB Cross Rate Chart ═══════════════════════════════════════════
    st.markdown(
        "<div class='group-header' style='border-left-color:#36D399;'>"
        "🔄 LAK/THB — Cross Rate Scenario (Derived from USD/LAK ÷ USD/THB)</div>",
        unsafe_allow_html=True,
    )
    _thb_eff_val = _thb_usd or 34.5
    if lak_thb_eff:
        # ── Proper derived cross-rate forecast ───────────────────────────────
        # LAK/THB(m) = lak_thb_eff × (1+lak_pct/100)^m / (1+thb_pct/100)^m
        # When LAK depreciates faster than THB → LAK/THB rises (LAK weakens vs THB)
        # When THB depreciates faster than LAK → LAK/THB falls (LAK strengthens vs THB)
        fig_lak_thb = go.Figure()
        cross_rows = []
        for sc in SCENARIOS:
            lak_pct = pcts_lak.get(sc, LAK_DEFAULT_MONTHLY_PCT[sc])
            thb_pct = pcts_thb.get(sc, THB_DEFAULT_MONTHLY_PCT[sc])
            cross_mo_pct = ((1 + lak_pct / 100) / (1 + thb_pct / 100) - 1) * 100
            y_cross = [
                round(lak_thb_eff * ((1 + lak_pct / 100) ** m) / ((1 + thb_pct / 100) ** m), 4)
                for m in range(forecast_months + 1)
            ]
            direction = (
                "↑ LAK weakens vs THB" if cross_mo_pct > 0.05
                else ("↓ LAK strengthens vs THB" if cross_mo_pct < -0.05 else "→ Stable")
            )
            txt_c = [""] * len(y_cross)
            txt_c[-1] = f"  {y_cross[-1]:,.2f}"
            fig_lak_thb.add_trace(go.Scatter(
                x=x_lak, y=y_cross,
                mode="lines+markers+text",
                name=f"{sc} ({cross_mo_pct:+.2f}%/mo)",
                text=txt_c,
                textposition="middle right",
                textfont=dict(color=SCENARIO_COLORS[sc], size=11, family="Courier New, monospace"),
                line=dict(color=SCENARIO_COLORS[sc], dash=SCENARIO_DASH[sc], width=2.5),
                marker=dict(size=7, color=SCENARIO_COLORS[sc]),
                hovertemplate=f"<b>{sc}</b><br>%{{x}}: %{{y:,.4f}} LAK/THB<extra></extra>",
            ))
            # build detail rows for expander
            row_c = {
                "Scenario": sc,
                "USD/LAK %/mo": f"{lak_pct:+.2f}%",
                "USD/THB %/mo": f"{thb_pct:+.2f}%",
                "LAK/THB net %/mo": f"{cross_mo_pct:+.3f}%",
                "Direction": direction,
            }
            for i_c, lbl in enumerate(x_lak):
                row_c[lbl] = f"{y_cross[i_c]:,.2f}"
            cross_rows.append(row_c)

        fig_lak_thb.update_layout(
            title=dict(
                text="<b>LAK/THB Cross Rate</b> — Derived Scenario Forecast",
                font_size=15, font_color="#F5A623",
            ),
            xaxis_title="Period", yaxis_title="LAK per 1 Thai Baht",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font_size=11,
                        bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1"),
            height=430, margin=dict(t=90, b=50, r=120, l=80),
            hovermode="x unified", paper_bgcolor="#0F1923", plot_bgcolor="#0B1420",
            font=dict(color="#CBD5E1"),
            xaxis=dict(gridcolor="#1E3A5F", linecolor="#1E3A5F",
                       tickfont=dict(color="#94A3B8")),
            yaxis=dict(gridcolor="#1E3A5F", linecolor="#1E3A5F",
                       tickfont=dict(color="#94A3B8")),
        )
        st.plotly_chart(fig_lak_thb, width="stretch")

        with st.expander("📋 LAK/THB Derived Cross Rate — Detailed Projection"):
            def style_cross(df: pd.DataFrame):
                s = pd.DataFrame("", index=df.index, columns=df.columns)
                for sc_n in df.index:
                    bg = SCENARIO_BG.get(sc_n, "#162032")
                    for col in df.columns:
                        s.loc[sc_n, col] = (
                            f"background-color:{bg};color:#E2E8F0;"
                            "font-family:'Courier New',monospace;"
                        )
                return s
            st.dataframe(
                pd.DataFrame(cross_rows).set_index("Scenario").style.apply(style_cross, axis=None),
                width="stretch",
            )

        st.markdown(
            "<div class='info-box'>"
            "<b style='color:#36D399;'>ℹ️ LAK/THB Cross Rate Logic</b><br>"
            "Derived each month as <b>USD/LAK(m) ÷ USD/THB(m)</b> using separate compounded "
            "projections for each currency, so the <i>net</i> cross-rate change = "
            "LAK depreciation vs USD <b>minus</b> THB depreciation vs USD.<br>"
            "<b style='color:#F5A623;'>When EUR/USD strengthens (USD weakens):</b> "
            "Both LAK and THB tend to weaken vs USD, but THB typically weakens <i>less</i> "
            "(stronger economy, higher policy credibility) → "
            "<b style='color:#F87171;'>LAK/THB rises — LAK weakens vs THB.</b><br>"
            "<b style='color:#60A5FA;'>When EUR/USD weakens (USD strengthens):</b> "
            "Both currencies firm vs USD; THB may firm faster → "
            "<b style='color:#36D399;'>LAK/THB may fall — LAK strengthens vs THB.</b><br>"
            "<b>BOL can stabilise LAK/THB</b> by selling THB from reserves — "
            "see the 🛡️ <b>FX Intervention Simulator</b> below."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("LAK/THB cross rate requires both USD/LAK and USD/THB data. Check connection or enter USD/LAK manually above.")

    st.divider()

    # ══ 3. Gold Price in LAK ══════════════════════════════════════════════════
    st.markdown(
        "<div class='group-header' style='border-left-color:#FFD700;'>"
        "🥇 Gold Price in LAK — Dual-Factor Scenario</div>",
        unsafe_allow_html=True,
    )
    if gold_lak_eff and _gold_usd and _lak_usd_eff:
        gl_info, gl_sel = st.columns([3, 1])
        with gl_info:
            st.markdown(
                "<div class='info-box'>"
                "<b style='color:#FFD700;'>ℹ️ Gold in LAK Formula</b><br>"
                "Gold (LAK/oz) = Gold (USD/oz) × USD/LAK exchange rate.<br>"
                "Each scenario compounds the <b>Gold USD price % change</b> (from Commodity tab) "
                "plus the <b>LAK depreciation %</b> (from Laos sidebar) every month."
                "</div>",
                unsafe_allow_html=True,
            )
        with gl_sel:
            gold_src = st.selectbox(
                "Gold USD scenario source",
                ["Commodity Assumptions", "LAK Assumptions"],
                key="gold_lak_src",
            )
        use_comm = (gold_src == "Commodity Assumptions")

        fig_gl = go.Figure()
        for sc in SCENARIOS:
            lak_pct  = pcts_lak.get(sc, LAK_DEFAULT_MONTHLY_PCT[sc])
            gold_pct = pcts_comm.get(sc, DEFAULT_MONTHLY_PCT[sc]) if use_comm else lak_pct
            y_gl = project_gold_lak(_gold_usd, _lak_usd_eff, forecast_months, gold_pct, lak_pct)
            txt_gl = [""] * len(y_gl)
            txt_gl[-1] = f"  {y_gl[-1]:,.0f}"
            fig_gl.add_trace(go.Scatter(
                x=x_lak, y=y_gl,
                mode="lines+markers+text",
                name=f"{sc}",
                text=txt_gl,
                textposition="middle right",
                textfont=dict(color=SCENARIO_COLORS[sc], size=11, family="Courier New, monospace"),
                line=dict(color=SCENARIO_COLORS[sc], dash=SCENARIO_DASH[sc], width=2.5),
                marker=dict(size=7, color=SCENARIO_COLORS[sc]),
                hovertemplate=f"<b>{sc}</b><br>%{{x}}: %{{y:,.0f}} LAK<extra></extra>",
            ))
        fig_gl.update_layout(
            title=dict(text="<b>Gold Price in LAK</b> — 5-Scenario", font_size=15, font_color="#F5A623"),
            xaxis_title="Period", yaxis_title="LAK per Troy Ounce",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font_size=11, bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1"),
            height=420, margin=dict(t=90, b=50, r=120, l=80),
            hovermode="x unified", paper_bgcolor="#0F1923", plot_bgcolor="#0B1420",
            font=dict(color="#CBD5E1"),
            xaxis=dict(gridcolor="#1E3A5F", linecolor="#1E3A5F", tickfont=dict(color="#94A3B8")),
            yaxis=dict(gridcolor="#1E3A5F", linecolor="#1E3A5F", tickfont=dict(color="#94A3B8"),
                       tickformat=",.0f"),
        )
        st.plotly_chart(fig_gl, width="stretch")
    else:
        st.info("Gold in LAK requires both Gold ($/oz) and USD/LAK data. Enter USD/LAK manually above if needed.")

    st.divider()

    # ══ 4. Laos CPI Inflation Scenarios ═══════════════════════════════════════
    st.markdown(
        "<div class='group-header' style='border-left-color:#F87171;'>"
        "📈 Laos CPI Inflation — Scenario Projections</div>",
        unsafe_allow_html=True,
    )
    infl_chart_col, infl_tbl_col = st.columns([3, 2])

    with infl_chart_col:
        fig_infl = go.Figure()
        for sc in SCENARIOS:
            adj     = infl_adj.get(sc, LAK_INFL_SCENARIO_ADJ[sc])
            y_idx   = project_inflation_index(current_inflation, forecast_months, adj)
            txt_i   = [""] * len(y_idx)
            txt_i[-1] = f"  {y_idx[-1]:.2f}"
            sc_rate = round(current_inflation + adj, 1)
            fig_infl.add_trace(go.Scatter(
                x=x_lak, y=y_idx,
                mode="lines+markers+text",
                name=f"{sc} ({sc_rate:.1f}%/yr)",
                text=txt_i,
                textposition="middle right",
                textfont=dict(color=SCENARIO_COLORS[sc], size=11, family="Courier New, monospace"),
                line=dict(color=SCENARIO_COLORS[sc], dash=SCENARIO_DASH[sc], width=2.5),
                marker=dict(size=7, color=SCENARIO_COLORS[sc]),
                hovertemplate=(
                    f"<b>{sc}</b><br>%{{x}}: CPI = %{{y:.3f}}"
                    f"<br>Annual ≈ {sc_rate:.1f}%<extra></extra>"
                ),
            ))
        fig_infl.add_hline(
            y=100, line_dash="dot", line_color="#64748B", line_width=1.5,
            annotation_text="Base = 100", annotation_font_color="#64748B",
        )
        fig_infl.update_layout(
            title=dict(text="<b>Laos CPI Index</b> — Scenario Projection (Base = 100)",
                       font_size=15, font_color="#F5A623"),
            xaxis_title="Period", yaxis_title="CPI Index (Current = 100)",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font_size=11, bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1"),
            height=400, margin=dict(t=90, b=50, r=120, l=65),
            hovermode="x unified", paper_bgcolor="#0F1923", plot_bgcolor="#0B1420",
            font=dict(color="#CBD5E1"),
            xaxis=dict(gridcolor="#1E3A5F", linecolor="#1E3A5F", tickfont=dict(color="#94A3B8")),
            yaxis=dict(gridcolor="#1E3A5F", linecolor="#1E3A5F", tickfont=dict(color="#94A3B8")),
        )
        st.plotly_chart(fig_infl, width="stretch")

    with infl_tbl_col:
        st.markdown(
            "<p style='color:#94A3B8;font-size:0.8rem;font-weight:700;text-transform:uppercase;"
            "letter-spacing:0.07em;margin-bottom:0.4rem;'>Inflation Summary</p>",
            unsafe_allow_html=True,
        )
        infl_rows = []
        for sc in SCENARIOS:
            adj       = infl_adj.get(sc, LAK_INFL_SCENARIO_ADJ[sc])
            sc_annual = round(current_inflation + adj, 1)
            mo_rate   = round(((1 + sc_annual / 100.0) ** (1 / 12) - 1) * 100, 3)
            idx_end   = project_inflation_index(current_inflation, forecast_months, adj)[-1]
            cum_infl  = round(idx_end - 100, 2)
            infl_rows.append({
                "Scenario":          sc,
                "Annual Rate (%)": sc_annual,
                "Monthly Rate (%)": mo_rate,
                f"CPI at M{forecast_months}": idx_end,
                "Cum. Inflation (%)": cum_infl,
            })
        df_infl = pd.DataFrame(infl_rows).set_index("Scenario")

        def style_infl(df: pd.DataFrame):
            s = pd.DataFrame("", index=df.index, columns=df.columns)
            for sc in df.index:
                bg  = SCENARIO_BG.get(sc, "#162032")
                for col in df.columns:
                    s.loc[sc, col] = (
                        f"background-color:{bg}; color:#E2E8F0; "
                        "font-family:'Courier New',monospace;"
                    )
            return s

        st.dataframe(
            df_infl.style.apply(style_infl, axis=None),
            width="stretch",
            height=250,
        )
        st.markdown(
            "<div class='info-box' style='margin-top:0.6rem;'>"
            "<b style='color:#F87171;'>ℹ️ Inflation Model</b><br>"
            "CPI Index = 100 at today's date, compounded monthly at the scenario annual rate.<br>"
            f"Base rate: <b style='color:#F5E642;'>{current_inflation:.1f}% p.a.</b> — "
            "set in sidebar. Adjust each scenario's pp offset via "
            "<i>Inflation Adjustment per Scenario</i>."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Inflation Transmission Pressures (full width) ─────────────────────────
    st.markdown(
        "<div class='group-header' style='border-left-color:#F87171;font-size:0.82rem;'>"
        "🔗 CPI Transmission Pressures — Oil + EUR/USD Channel (Cumulative over horizon)</div>",
        unsafe_allow_html=True,
    )
    _eur_usd_base = fx_prices.get("EUR/USD") or 1.10
    trans_rows = []
    _oil_keys = [a for a in ["Brent Crude ($/bbl)", "WTI Crude ($/bbl)"]
                 if comm_prices.get(a) is not None]
    for sc in SCENARIOS:
        # Oil channel — cum % change at end of horizon
        if _oil_keys:
            oil_cum_pct = float(np.mean([
                ((project(comm_prices[a], forecast_months,
                          pcts_comm.get(sc, DEFAULT_MONTHLY_PCT[sc]))[-1]
                  - comm_prices[a]) / comm_prices[a]) * 100
                for a in _oil_keys
            ]))
        else:
            oil_cum_pct = 0.0
        oil_cpi = oil_cum_pct * oil_cpi_beta_sidebar

        # EUR/USD channel: EUR/USD up = USD weak = Thai/VN imports via EUR dearer → CPI rises
        eur_mo_pct = pcts_fx.get(sc, DEFAULT_MONTHLY_PCT[sc])
        eur_cum_pct = ((1 + eur_mo_pct / 100) ** forecast_months - 1) * 100
        eur_cpi = eur_cum_pct * eur_cpi_beta_sidebar

        total_pressure = oil_cpi + eur_cpi
        trans_rows.append({
            "Scenario":                       sc,
            "Oil price (cum %)": f"{oil_cum_pct:+.1f}%",
            "Oil → CPI (pp)":              f"{oil_cpi:+.2f} pp",
            "EUR/USD (cum %)": f"{eur_cum_pct:+.1f}%",
            "EUR/USD → CPI (pp)":          f"{eur_cpi:+.2f} pp",
            "Total ext. CPI pressure":     f"{total_pressure:+.2f} pp",
        })
    df_trans = pd.DataFrame(trans_rows).set_index("Scenario")

    def _style_trans(df: pd.DataFrame):
        s = pd.DataFrame("", index=df.index, columns=df.columns)
        for sc_n in df.index:
            bg = SCENARIO_BG.get(sc_n, "#162032")
            for col in df.columns:
                s.loc[sc_n, col] = (
                    f"background-color:{bg};color:#E2E8F0;"
                    "font-family:'Courier New',monospace;font-size:0.8rem;"
                )
        return s

    st.dataframe(
        df_trans.style.apply(_style_trans, axis=None),
        width="stretch", height=230,
    )
    st.markdown(
        "<div class='info-box'>"
        "<b style='color:#F87171;'>🔗 Transmission Chain: EUR/USD → Thailand/Vietnam → Laos CPI</b><br>"
        "<b style='color:#F5A623;'>① EUR/USD rises (USD weakens):</b> "
        "Thai &amp; Vietnamese exporters who source goods from Europe face higher costs in THB/VND "
        "→ passed on to Laos importers → <b>Laos CPI rises</b>.<br>"
        "<b style='color:#36D399;'>② Oil price rises:</b> "
        "Energy &amp; transport cost in Laos rises directly — most oil is imported via Thailand.<br>"
        "<b style='color:#60A5FA;'>③ LAK depreciation</b> (tracked in the CPI model above via "
        "the <i>Inflation Adjustment per Scenario</i>) amplifies all import-cost effects because "
        "Laos pays in USD/THB but earns in LAK.<br>"
        f"<small style='color:#64748B;'>EUR/USD beta: {eur_cpi_beta_sidebar:.2f} · "
        f"Oil beta: {oil_cpi_beta_sidebar:.2f} · Adjust in sidebar.</small>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ══ 5. FX Intervention Simulator ══════════════════════════════════════════
    st.markdown(
        "<div class='group-header' style='border-left-color:#A78BFA;'>"
        "🛡️ FX Intervention Simulator — Reserve Cost & Runway</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#94A3B8;font-size:0.84rem;margin:0.2rem 0 0.8rem 0;'>"
        "How much BOL must spend each month from reserves to keep LAK depreciation "
        "below the policy target, across all 5 scenarios.</p>",
        unsafe_allow_html=True,
    )

    monthly_fx_vol_mn  = daily_fx_vol_mn  * 30   # USD/LAK channel monthly volume
    monthly_thb_vol_mn = daily_thb_vol_mn * 30   # LAK/THB channel monthly volume

    intv_rows = []
    for sc in SCENARIOS:
        # ── USD / LAK channel ──────────────────────────────────────────────
        nat_dep_usd   = pcts_lak.get(sc, LAK_DEFAULT_MONTHLY_PCT[sc])
        excess_usd    = max(0.0, nat_dep_usd - target_max_dep)
        cost_usd_mo   = (
            (excess_usd / 100.0) * monthly_fx_vol_mn / (intv_effect / 100.0)
            if excess_usd > 0 and intv_effect > 0 else 0.0
        )
        runway_usd    = bol_reserves / cost_usd_mo if cost_usd_mo > 0 else 9999.0

        # ── LAK / THB channel ──────────────────────────────────────────────
        # Cross-rate natural monthly change = USD/LAK rate − USD/THB rate (compounded)
        thb_dep       = pcts_thb.get(sc, THB_DEFAULT_MONTHLY_PCT[sc])
        cross_mo_dep  = ((1 + nat_dep_usd / 100) / (1 + thb_dep / 100) - 1) * 100
        excess_thb    = max(0.0, cross_mo_dep - target_lak_thb_dep)
        cost_thb_mo   = (
            (excess_thb / 100.0) * monthly_thb_vol_mn / (intv_effect / 100.0)
            if excess_thb > 0 and intv_effect > 0 else 0.0
        )
        runway_thb    = bol_thb_reserves / cost_thb_mo if cost_thb_mo > 0 else 9999.0

        # ── Combined ───────────────────────────────────────────────────────
        total_mo_cost = cost_usd_mo + cost_thb_mo
        total_reserves = bol_reserves + bol_thb_reserves
        combined_runway = total_reserves / total_mo_cost if total_mo_cost > 0 else 9999.0
        horizon_total   = total_mo_cost * forecast_months
        reserves_end    = max(total_reserves - horizon_total, 0.0)

        intv_rows.append({
            "Scenario":                    sc,
            "USD/LAK nat dep (%/mo)":      nat_dep_usd,
            "USD excess (pp)":             round(excess_usd, 2),
            "USD cost (mn/mo)":            round(cost_usd_mo, 1),
            "USD runway (mo)":             round(min(runway_usd, 999.0), 1),
            "LAK/THB nat dep (%/mo)":      round(cross_mo_dep, 3),
            "THB excess (pp)":             round(excess_thb, 2),
            "THB cost (mn/mo)": round(cost_thb_mo, 1),
            "THB runway (mo)":             round(min(runway_thb, 999.0), 1),
            "Total cost (mn/mo)":          round(total_mo_cost, 1),
            f"Total cost M{forecast_months} (mn)": round(horizon_total, 1),
            "Combined runway (mo)":        round(min(combined_runway, 999.0), 1),
            f"Reserves left M{forecast_months} (mn)": round(reserves_end, 1),
        })
    df_intv = pd.DataFrame(intv_rows).set_index("Scenario")

    # ── 4-column chart layout: USD cost | USD runway | THB cost | Combined runway
    iv1, iv2 = st.columns([1, 1])

    def _intv_bar_chart(
        title: str, x_vals, y_vals, text_vals, hline_y: float, hline_label: str,
        yaxis_title: str, bar_colors,
    ) -> go.Figure:
        fig = go.Figure(go.Bar(
            x=x_vals, y=y_vals, text=text_vals,
            textposition="outside",
            textfont=dict(color="#E2E8F0", size=10),
            marker_color=bar_colors,
            marker_line_color="#0F1923", marker_line_width=1.5,
        ))
        if hline_y > 0:
            fig.add_hline(
                y=hline_y, line_dash="dot", line_color="#F87171",
                annotation_text=hline_label,
                annotation_font_color="#F87171",
            )
        fig.update_layout(
            title=dict(text=title, font_size=12, font_color="#F5A623"),
            yaxis_title=yaxis_title, template="plotly_dark",
            paper_bgcolor="#0F1923", plot_bgcolor="#0B1420",
            font=dict(color="#CBD5E1"), height=320,
            margin=dict(t=70, b=55, l=65, r=10),
            xaxis=dict(gridcolor="#1E3A5F", tickfont=dict(color="#94A3B8", size=9)),
            yaxis=dict(gridcolor="#1E3A5F", tickfont=dict(color="#94A3B8")),
            showlegend=False,
        )
        return fig

    def _runway_colors(runway_list):
        cols = []
        for r in runway_list:
            if r >= 999:   cols.append("#36D399")
            elif r >= 24:  cols.append("#60A5FA")
            elif r >= 12:  cols.append("#FBBF24")
            else:          cols.append("#F87171")
        return cols

    with iv1:
        # USD/LAK monthly cost
        usd_costs = [r["USD cost (mn/mo)"] for r in intv_rows]
        st.plotly_chart(
            _intv_bar_chart(
                f"<b>USD/LAK — Monthly Cost (USD mn)</b><br>"
                f"<sub>Target ≤{target_max_dep:.1f}%/mo · vol ${daily_fx_vol_mn:.1f}mn/day</sub>",
                SCENARIOS, usd_costs,
                [f"${v:.1f}mn" for v in usd_costs],
                bol_reserves / 12, "1-yr USD depletion pace",
                "USD mn / month",
                [SCENARIO_COLORS[sc] for sc in SCENARIOS],
            ),
            width="stretch",
        )
        # USD runway
        usd_run = [r["USD runway (mo)"] for r in intv_rows]
        fig_usd_rw = _intv_bar_chart(
            f"<b>USD Reserve Runway</b><br>"
            f"<sub>BOL USD: ${bol_reserves:,.0f}mn</sub>",
            SCENARIOS, [min(r, 120) for r in usd_run],
            [("∞" if r >= 999 else f"{r:.0f} mo") for r in usd_run],
            0, "", "Months remaining",
            _runway_colors(usd_run),
        )
        fig_usd_rw.add_hline(y=12, line_dash="dot", line_color="#FBBF24",
                             annotation_text="12-mo", annotation_font_color="#FBBF24")
        fig_usd_rw.add_hline(y=6,  line_dash="dot", line_color="#F87171",
                             annotation_text="6-mo",  annotation_font_color="#F87171")
        st.plotly_chart(fig_usd_rw, width="stretch")

    with iv2:
        # LAK/THB monthly cost
        thb_costs = [r["THB cost (mn/mo)"] for r in intv_rows]
        st.plotly_chart(
            _intv_bar_chart(
                f"<b>LAK/THB — Monthly THB Cost (USD equiv mn)</b><br>"
                f"<sub>Target ≤{target_lak_thb_dep:.1f}%/mo · vol ${daily_thb_vol_mn:.1f}mn/day</sub>",
                SCENARIOS, thb_costs,
                [f"${v:.1f}mn" for v in thb_costs],
                bol_thb_reserves / 12, "1-yr THB depletion pace",
                "USD equiv mn / month",
                [SCENARIO_COLORS[sc] for sc in SCENARIOS],
            ),
            width="stretch",
        )
        # Combined runway
        comb_run = [r["Combined runway (mo)"] for r in intv_rows]
        fig_comb_rw = _intv_bar_chart(
            f"<b>Combined Reserve Runway (Both Channels)</b><br>"
            f"<sub>USD ${bol_reserves:,.0f}mn + THB ${bol_thb_reserves:,.0f}mn equiv</sub>",
            SCENARIOS, [min(r, 120) for r in comb_run],
            [("∞" if r >= 999 else f"{r:.0f} mo") for r in comb_run],
            0, "", "Months remaining",
            _runway_colors(comb_run),
        )
        fig_comb_rw.add_hline(y=12, line_dash="dot", line_color="#FBBF24",
                              annotation_text="12-mo", annotation_font_color="#FBBF24")
        fig_comb_rw.add_hline(y=6,  line_dash="dot", line_color="#F87171",
                              annotation_text="6-mo",  annotation_font_color="#F87171")
        st.plotly_chart(fig_comb_rw, width="stretch")

    def _style_intv(df: pd.DataFrame):
        s = pd.DataFrame("", index=df.index, columns=df.columns)
        for sc_n in df.index:
            bg = SCENARIO_BG.get(sc_n, "#162032")
            for col in df.columns:
                s.loc[sc_n, col] = (
                    f"background-color:{bg};color:#E2E8F0;"
                    "font-family:'Courier New',monospace;font-size:0.76rem;"
                )
        return s

    st.dataframe(
        df_intv.style.apply(_style_intv, axis=None),
        width="stretch", height=250,
    )
    st.markdown(
        f"<div class='info-box'>"
        f"<b style='color:#A78BFA;'>🛡️ Two-Channel Intervention Model</b><br>"
        f"<b style='color:#60A5FA;'>USD/LAK channel:</b> BOL sells USD to buy LAK. "
        f"Cost = (excess dep pp ÷ 100) × ${monthly_fx_vol_mn:.0f}mn/mo ÷ effectiveness.<br>"
        f"<b style='color:#36D399;'>LAK/THB channel:</b> BOL sells THB to buy LAK. "
        f"LAK/THB natural dep = USD/LAK rate <i>minus</i> USD/THB rate (compounded). "
        f"Cost = (excess cross dep pp ÷ 100) × ${monthly_thb_vol_mn:.0f}mn/mo ÷ effectiveness.<br>"
        f"<b style='color:#F5A623;'>Why BOL needs BOTH channels:</b> Even when USD/LAK is "
        f"stable, LAK can weaken vs THB if Thailand's economy outperforms Laos — "
        f"hurting Lao importers who pay Thai suppliers in THB.<br>"
        f"<b style='color:#36D399;'>∞ runway</b> = no intervention needed for that scenario.<br>"
        f"<small style='color:#64748B;'>Both channels share effectiveness parameter "
        f"({intv_effect:.0f}%). Model assumes linear market response.</small>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 – SUMMARY VIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    st.subheader(f"📋 End-of-Horizon Snapshot — Month {forecast_months}")
    st.caption(
        f"Projected price or rate at the end of the {forecast_months}-month horizon for each scenario, "
        "with percentage change from today’s live price. "
        "Use this table for briefing notes, MPC submissions, or BOL/NSC scenario summaries."
    )

    valid_c = {k: v for k, v in comm_prices.items() if v is not None}
    valid_f = {k: v for k, v in fx_prices.items()   if v is not None}

    if valid_c:
        st.markdown("#### Commodities")
        df_sum_c = scenario_summary(valid_c, forecast_months, pcts_comm)
        st.dataframe(df_sum_c, width="stretch")

    if valid_f:
        st.markdown("#### FX Pairs")
        df_sum_f = scenario_summary(valid_f, forecast_months, pcts_fx)
        st.dataframe(df_sum_f, width="stretch")

    st.divider()
    st.markdown("#### Scenario Legend")
    leg_cols = st.columns(len(SCENARIOS))
    descs = {
        "Strong Bull":   "🚨 Worst-case for Laos — commodity surge, USD strong, LAK depreciates fast, CPI spikes",
        "Moderate Bull": "Moderate commodity rise, mild LAK depreciation, manageable CPI pressure",
        "Base Case":     "Consensus / status-quo trajectory — BOL reference scenario",
        "Moderate Bear": "Commodity softening, partial LAK stabilisation, easing inflation outlook",
        "Strong Bear":   "🟢 Best-case for Laos — commodity decline, USD weakens, LAK firms, CPI relief",
    }
    for i, sc in enumerate(SCENARIOS):
        with leg_cols[i]:
            st.markdown(
                f"<div style='"
                f"background:linear-gradient(135deg,#162032,#0D1927);"
                f"border:1px solid #1E3A5F;"
                f"border-left:4px solid {SCENARIO_COLORS[sc]};"
                f"border-radius:8px;"
                f"padding:0.75rem 1rem;"
                f"margin-top:0.3rem;"
                f"transition:box-shadow 0.15s ease;"
                f"'>"
                f"<b style='color:{SCENARIO_COLORS[sc]};font-size:0.9rem;"
                f"text-shadow:0 0 8px {SCENARIO_COLORS[sc]}55;'>{sc}</b><br>"
                f"<small style='color:#94A3B8;font-size:0.74rem;'>{descs[sc]}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 – EXPORT TO EXCEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.subheader("📥 Export Forecast Report to Excel")
    st.caption(
        "Generate a fully formatted Excel workbook for use in BOL/NSC briefings, "
        "MPC notes, cabinet presentations, or archiving. "
        "All scenario assumptions and live prices at time of export are saved in the Assumptions sheet."
    )

    st.markdown(
        """
        The generated workbook contains **5 worksheets**:

        | Sheet | Contents |
        |-------|----------|
        | **Commodity Scenarios** | Monthly projected prices for all 11 commodity imports × 5 scenarios |
        | **FX Scenarios** | Monthly projected rates for EUR/USD, GBP/USD, USD/JPY, USD/THB, USD/LAK × 5 scenarios |
        | **Laos — FX &amp; Gold (LAK)** | USD/LAK, LAK/THB cross rate, and Gold price in LAK × 5 scenarios |
        | **Laos — Inflation** | Laos CPI index projections and annual scenario rate summary |
        | **Assumptions** | All sidebar parameters, forecast horizon, and export timestamp |

        Cells are colour-coded by scenario (🟢 bull / 🔵 base / 🟠 bear / 🔴 strong bear).
        Columns are frozen at the asset name and current price for easy horizontal scrolling.
         
        ⚠️ **Before exporting:** enter the USD/LAK rate manually in the 🇱🇦 Laos Focus tab
        if Yahoo Finance returns N/A (this is common for LAK as an exotic currency).
        """
    )

    st.divider()

    if st.button("📊 Generate Excel Report", type="primary", width="stretch"):
        valid_c_exp = {k: v for k, v in comm_prices.items() if v is not None}
        valid_f_exp = {k: v for k, v in fx_prices.items()   if v is not None}

        if not valid_c_exp and not valid_f_exp:
            st.error("No data available to export. Please refresh live data first.")
        else:
            with st.spinner("🔨 Generating Excel workbook…"):
                lak_exp = {
                    "USD/LAK":       _lak_usd_eff if "_lak_usd_eff" in dir() else _lak_usd,
                    "LAK/THB":       lak_thb_eff  if "lak_thb_eff"  in dir() else lak_thb_rate,
                    "Gold (LAK/oz)": gold_lak_eff if "gold_lak_eff" in dir() else gold_lak_price,
                }
                excel_buf = generate_excel(
                    valid_c_exp, valid_f_exp,
                    forecast_months,
                    pcts_comm, pcts_fx,
                    lak_prices=lak_exp,
                    pcts_lak=pcts_lak,
                    inflation_base=current_inflation,
                    infl_adj=infl_adj,
                )
            fname = f"FX_Commodities_Forecast_{TODAY.strftime('%Y%m%d_%H%M')}.xlsx"
            st.download_button(
                label="⬇️ Download Excel Report",
                data=excel_buf,
                file_name=fname,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
                type="primary",
            )
            st.success(f"✅ **{fname}** is ready for download.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 7 – USER GUIDE
# ══════════════════════════════════════════════════════════════════════════════
with tab_help:
    st.markdown(
        "<h3 style='color:#F5A623;margin-bottom:0.2rem;'>📖 User Guide</h3>"
        "<p style='color:#94A3B8;font-size:0.85rem;'>"
        "How to use the Laos FX &amp; Import Price Forecast Dashboard — "
        "Bank of Laos / FX Policy Division</p>",
        unsafe_allow_html=True,
    )

    # ── 1. Purpose ───────────────────────────────────────────────────────────
    st.markdown(
        "<div class='group-header' style='border-left-color:#F5A623;'>"
        "🎯 Purpose &amp; Intended Users</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='info-box' style='border-left-color:#F5A623;'>"
        "This dashboard is built for the <b style='color:#F5E642;'>FX Policy Division</b> "
        "(BOL) and the <b style='color:#F5E642;'>National Statistics Centre (NSC)</b> to:"
        "<ul style='color:#CBD5E1;margin:0.4rem 0 0 1rem;line-height:1.8;'>"
        "<li>Monitor real-time global commodity and FX prices relevant to Laos’ import basket.</li>"
        "<li>Run 5-scenario forward projections (1–6 months) to stress-test LAK stability.</li>"
        "<li>Estimate how much BOL foreign reserves are needed to defend the LAK vs USD "
        "and vs THB under each scenario.</li>"
        "<li>Quantify how oil price and EUR/USD moves feed through to Laos CPI via Thailand "
        "and Vietnam.</li>"
        "<li>Generate formatted Excel reports for MPC briefings, cabinet presentations, "
        "and archiving.</li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 2. Sidebar walkthrough ───────────────────────────────────────────────
    st.markdown(
        "<div class='group-header' style='border-left-color:#60A5FA;'>"
        "⚙️ Sidebar — How to Set Assumptions</div>",
        unsafe_allow_html=True,
    )
    sidebar_rows = [
        ("Forecast horizon (months)", "Slider 1–6",
         "How many months ahead to project. All charts and tables update instantly."),
        ("Commodity Assumptions", "Per-scenario %/mo",
         "Monthly compounded price change applied to ALL 11 commodities simultaneously. "
         "Positive = prices rise (bad for Laos imports). Negative = prices fall."),
        ("FX Assumptions (EUR/USD etc.)", "Per-scenario %/mo or mirror",
         "Monthly % change for EUR/USD, GBP/USD, USD/JPY. Tick ‘Mirror commodity’ "
         "to use the same figures. These drive the FX Scenarios tab."),
        ("Current Laos CPI inflation (%)", "Manual entry",
         "Enter the latest official annual CPI inflation rate from BOL/NSC. "
         "This is the base rate for inflation projections — default is 26%."),
        ("LAK Depreciation per Scenario", "Per-scenario %/mo",
         "Monthly % change in USD/LAK. Positive = LAK weakens (more LAK per USD). "
         "Drives USD/LAK chart and feeds into LAK/THB cross-rate and Gold-in-LAK calculations."),
        ("THB Monthly % per Scenario", "Per-scenario %/mo",
         "Monthly % change in USD/THB. Positive = THB weakens. "
         "Used to derive the LAK/THB cross rate: net dep = LAK dep minus THB dep."),
        ("Oil → CPI pass-through fraction", "0.0 – 1.0",
         "Fraction of an oil price % change that flows into Laos CPI. "
         "Default 0.30 means a 10% oil rise adds ~3 pp to Laos CPI within the forecast horizon."),
        ("EUR/USD → CPI (Thailand import channel)", "0.0 – 1.0",
         "Fraction of a EUR/USD % change that flows into Laos CPI via Thai/Vietnamese "
         "goods importers. Default 0.15 = 10% EUR/USD rise → +1.5 pp Laos CPI."),
        ("FX Intervention Parameters", "USD & THB reserves + targets",
         "Define BOL’s reserve buffers, daily market volumes, policy targets, and "
         "intervention effectiveness for the FX Intervention Simulator below."),
    ]
    for name, ctrl, desc in sidebar_rows:
        st.markdown(
            f"<div style='margin-bottom:0.5rem;padding:0.6rem 1rem;"
            f"background:#162032;border:1px solid #1E3A5F;border-radius:6px;border-left:3px solid #60A5FA;'>"
            f"<b style='color:#60A5FA;'>{name}</b> "
            f"<span style='color:#64748B;font-size:0.78rem;'>({ctrl})</span><br>"
            f"<span style='color:#CBD5E1;font-size:0.84rem;'>{desc}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 3. Tab guide ───────────────────────────────────────────────────────────
    st.markdown(
        "<div class='group-header' style='border-left-color:#36D399;'>"
        "🗂️ Tab-by-Tab Guide</div>",
        unsafe_allow_html=True,
    )
    tab_guide = [
        ("📈 Live Prices",
         "Displays the latest Yahoo Finance price for each commodity and FX pair. "
         "Prices update every 5 minutes (cached). If a price shows N/A, the data feed "
         "is unavailable — this is common for USD/LAK (exotic). Enter it manually in "
         "the 🇱🇦 Laos Focus tab."),
        ("📦 Commodity Scenarios",
         "Shows a full summary table (all 11 commodities × 5 scenarios × all months) and "
         "a selectable price path chart. The 🛢️ Oil Comparison section at the bottom "
         "shows end-of-horizon oil prices and estimates the CPI impact on Laos for each scenario."),
        ("💱 FX Scenarios",
         "Same layout for EUR/USD, GBP/USD, USD/JPY, USD/THB, and USD/LAK. "
         "Note: USD/THB assumptions are set separately under the THB expander in the sidebar."),
        ("🇱🇦 Laos Focus",
         "The core policy module. Contains: "
         "(1) USD/LAK 5-scenario chart with detailed table, "
         "(2) LAK/THB derived cross-rate chart (USD/LAK ÷ USD/THB), "
         "(3) Gold price in LAK, "
         "(4) Laos CPI inflation index projection, "
         "(5) CPI Transmission Pressures table (oil + EUR/USD channels), "
         "(6) FX Intervention Simulator — two-channel (USD and THB) reserve runway analysis."),
        ("📋 Summary View",
         "Quick one-page snapshot: end-of-horizon price/rate for every asset in every scenario. "
         "Ideal for printing or pasting into briefing documents."),
        ("📥 Export to Excel",
         "Generates a formatted Excel workbook with 5 sheets. "
         "Click \u2018Generate Excel Report\u2019 then \u2018Download\u2019. "
         "Always refresh live data first and enter the USD/LAK rate manually if needed."),
    ]
    for tab_n, tab_desc in tab_guide:
        st.markdown(
            f"<div style='margin-bottom:0.45rem;padding:0.6rem 1rem;"
            f"background:#162032;border:1px solid #1E3A5F;border-radius:6px;"
            f"border-left:3px solid #36D399;'>"
            f"<b style='color:#36D399;'>{tab_n}</b><br>"
            f"<span style='color:#CBD5E1;font-size:0.84rem;'>{tab_desc}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 4. Scenario logic ───────────────────────────────────────────────────────
    st.markdown(
        "<div class='group-header' style='border-left-color:#FBBF24;'>"
        "📊 How the 5-Scenario Model Works</div>",
        unsafe_allow_html=True,
    )
    sc_g1, sc_g2 = st.columns(2)
    with sc_g1:
        st.markdown(
            "<div class='info-box' style='border-left-color:#FBBF24;'>"
            "<b style='color:#FBBF24;'>Compounding formula</b><br>"
            "Each projection uses monthly compounding:<br>"
            "<code style='color:#F5E642;'>Price(month m) = Price₀ × (1 + r)^m</code><br>"
            "where <b>r = monthly % / 100</b>.<br><br>"
            "The <b>LAK/THB cross rate</b> is derived per month as:<br>"
            "<code style='color:#36D399;'>LAK/THB(m) = USD/LAK(m) ÷ USD/THB(m)</code><br>"
            "using separate compounded paths for each currency, so the net "
            "cross-rate movement equals LAK depreciation <b>minus</b> THB depreciation.<br><br>"
            "<b>Gold in LAK</b> is derived as:<br>"
            "<code style='color:#FFD700;'>Gold(LAK) = Gold(USD/oz) × USD/LAK(m)</code>"
            "</div>",
            unsafe_allow_html=True,
        )
    with sc_g2:
        st.markdown(
            "<div class='info-box' style='border-left-color:#FBBF24;'>"
            "<b style='color:#FBBF24;'>Scenario interpretation for Laos</b><br>"
            "<table style='width:100%;font-size:0.8rem;border-collapse:collapse;'>"
            "<tr><th style='color:#94A3B8;text-align:left;padding:3px 6px;'>Scenario</th>"
            "<th style='color:#94A3B8;padding:3px 6px;'>Impact on Laos</th></tr>"
            f"<tr><td style='color:#00FF87;padding:3px 6px;'>Strong Bull</td>"
            "<td style='color:#CBD5E1;padding:3px 6px;'>Commodities rise fast → 🔴 worst for Laos: CPI spikes, LAK weakens, BOL costly</td></tr>"
            f"<tr><td style='color:#36D399;padding:3px 6px;'>Moderate Bull</td>"
            "<td style='color:#CBD5E1;padding:3px 6px;'>Moderate pressure, manageable if BOL has reserves</td></tr>"
            f"<tr><td style='color:#60A5FA;padding:3px 6px;'>Base Case</td>"
            "<td style='color:#CBD5E1;padding:3px 6px;'>Status quo — BOL reference scenario for planning</td></tr>"
            f"<tr><td style='color:#FBBF24;padding:3px 6px;'>Moderate Bear</td>"
            "<td style='color:#CBD5E1;padding:3px 6px;'>Commodity softening, CPI eases, LAK can firm</td></tr>"
            f"<tr><td style='color:#F87171;padding:3px 6px;'>Strong Bear</td>"
            "<td style='color:#CBD5E1;padding:3px 6px;'>Commodities fall sharply → 🟢 best for Laos: CPI relief, LAK can appreciate</td></tr>"
            "</table>"
            "<br><small style='color:#64748B;'>Note: \"Bull\" = commodity/FX rate rises. For Laos, "
            "which is a net commodity importer, rising commodity prices are generally "
            "<b>negative</b>.</small>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 5. FX Intervention model ───────────────────────────────────────────────
    st.markdown(
        "<div class='group-header' style='border-left-color:#A78BFA;'>"
        "🛡️ FX Intervention Simulator — How It Works</div>",
        unsafe_allow_html=True,
    )
    intv_g1, intv_g2 = st.columns(2)
    with intv_g1:
        st.markdown(
            "<div class='info-box' style='border-left-color:#A78BFA;'>"
            "<b style='color:#A78BFA;'>USD/LAK Channel</b><br>"
            "BOL sells USD (from foreign reserves) to buy LAK on the open market, "
            "reducing supply of LAK and supporting its value vs USD.<br><br>"
            "<b>Monthly cost formula:</b><br>"
            "<code>Excess dep = LAK dep/mo − target max dep/mo</code><br>"
            "<code>Monthly cost = (excess ÷ 100) × monthly vol ÷ effectiveness</code><br><br>"
            "If LAK dep/mo ≤ target → <b>no intervention needed</b>.<br>"
            "<b>Effectiveness</b>: 50% means BOL needs $2 to absorb $1 of excess demand."
            "</div>",
            unsafe_allow_html=True,
        )
    with intv_g2:
        st.markdown(
            "<div class='info-box' style='border-left-color:#36D399;'>"
            "<b style='color:#36D399;'>LAK/THB Channel</b><br>"
            "BOL sells THB reserves to buy LAK, defending the LAK/THB cross rate. "
            "This is important because ~60% of Laos imports are sourced from Thailand "
            "and priced in THB — if LAK weakens only vs THB (not vs USD), "
            "import costs still rise and CPI is affected.<br><br>"
            "<b>Cross-rate natural dep = USD/LAK rate − USD/THB rate</b> (compounded).<br>"
            "The THB channel cost is computed identically but uses the THB reserve buffer "
            "and the LAK/THB market volume."
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div class='info-box' style='border-left-color:#F87171;margin-top:0.6rem;'>"
        "<b style='color:#F87171;'>⚠️ Model Assumptions & Limitations</b><br>"
        "<ul style='color:#CBD5E1;margin:0.3rem 0 0 1rem;line-height:1.8;'>"
        "<li>Linear market response — real FX markets are non-linear and can have "
        "\"cliff edges\" when credibility is lost.</li>"
        "<li>Effectiveness % is constant across all scenarios — in practice, it may "
        "be lower under severe stress (large outflows).</li>"
        "<li>Both channels share the same effectiveness parameter in the current model.</li>"
        "<li>Does not account for sterilisation costs, capital account restrictions, "
        "or IMF facility availability.</li>"
        "<li>Reserve figures should be updated monthly from BOL official reserve data. "
        "Default: USD 1,300 mn (estimated).</li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 6. CPI Transmission chain ───────────────────────────────────────────────
    st.markdown(
        "<div class='group-header' style='border-left-color:#F87171;'>"
        "🔗 CPI Transmission Chain — How Global Prices Affect Laos Inflation</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='info-box' style='border-left-color:#F87171;'>"
        "<b style='color:#F87171;'>Three-stage transmission mechanism:</b><br><br>"
        "📦 <b style='color:#F5A623;'>Stage 1 — Global commodity/FX shock</b><br>"
        "&nbsp;&nbsp;Oil prices rise globally (e.g. OPEC cut or geopolitical shock), or "
        "EUR/USD rises (USD weakens globally).<br><br>"
        "🚢 <b style='color:#FBBF24;'>Stage 2 — Thailand / Vietnam import channel</b><br>"
        "&nbsp;&nbsp;Laos imports ~60% of consumer and capital goods from Thailand and ~15% "
        "from Vietnam. Thai importers who source European goods now pay more in THB "
        "(because EUR/THB rises when EUR/USD rises). This cost is passed on to Laos buyers. "
        "Fuel, which powers transport and manufacturing, is imported via Thailand at global "
        "oil prices plus a regional margin.<br><br>"
        "🇱🇦 <b style='color:#F87171;'>Stage 3 — LAK depreciation amplifier</b><br>"
        "&nbsp;&nbsp;Laos earns revenue mainly in LAK but pays for imports in USD or THB. "
        "If LAK depreciates simultaneously, all import costs rise <i>further</i> in LAK terms — "
        "amplifying the inflationary effect. This is why the CPI model adjusts per scenario "
        "for the LAK depreciation assumption.<br><br>"
        "<small style='color:#64748B;'>The pass-through fractions (Oil → CPI and EUR/USD → CPI) "
        "are simplified linear approximations. Actual pass-through depends on subsidy policy, "
        "retail market competition, and timing of importer price adjustments.</small>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 7. Data sources & update cadence ─────────────────────────────────────────
    st.markdown(
        "<div class='group-header' style='border-left-color:#94A3B8;'>"
        "📡 Data Sources &amp; Recommended Update Cadence</div>",
        unsafe_allow_html=True,
    )
    data_rows = [
        ("Live commodity & FX prices",
         "Yahoo Finance (free API)",
         "Auto-refreshed every 5 min. Click ‘Refresh Live Data’ in sidebar to force update."),
        ("USD/LAK rate",
         "BOL daily rate (manual entry)",
         "Yahoo Finance rarely has LAK. Enter BOL’s official daily USD/LAK fixing in the "
         "🇱🇦 Laos Focus tab."),
        ("Laos CPI inflation %",
         "NSC / BOL official data (manual)",
         "Update monthly when NSC publishes the CPI release. Default is 26% — "
         "adjust in sidebar under 🇱🇦 Laos-Specific Assumptions."),
        ("BOL FX Reserves",
         "BOL internal treasury data (manual)",
         "Update monthly. Default assumes ~USD 1,300 mn. "
         "Set under 🔧 FX Intervention Parameters in sidebar."),
        ("Oil beta / EUR beta",
         "Calibrated from historical Laos CPI vs oil pass-through studies",
         "Review and recalibrate quarterly. Defaults: Oil 0.30, EUR/USD 0.15."),
        ("THB monthly % scenarios",
         "Analyst judgement / IMF / BOT forecasts",
         "Regularly review against Bank of Thailand (BOT) MPC decisions "
         "and IMF ASEAN regional outlook releases."),
    ]
    for src_name, src_provider, src_note in data_rows:
        st.markdown(
            f"<div style='margin-bottom:0.45rem;padding:0.6rem 1rem;"
            f"background:#162032;border:1px solid #1E3A5F;border-radius:6px;"
            f"border-left:3px solid #94A3B8;'>"
            f"<b style='color:#E2E8F0;'>{src_name}</b> "
            f"<span style='color:#F5A623;font-size:0.8rem;'>| {src_provider}</span><br>"
            f"<span style='color:#94A3B8;font-size:0.82rem;'>{src_note}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 8. Quick-start checklist ───────────────────────────────────────────────
    st.markdown(
        "<div class='group-header' style='border-left-color:#36D399;'>"
        "✅ Monthly Analyst Checklist</div>",
        unsafe_allow_html=True,
    )
    checklist = [
        "Update <b>Current Laos CPI inflation %</b> in sidebar from latest NSC release.",
        "Update <b>BOL USD Reserves</b> from treasury data.",
        "Check if <b>USD/LAK rate</b> loaded automatically — if not, enter BOL official fixing in "
        "🇱🇦 Laos Focus tab.",
        "Click <b>🔄 Refresh Live Data</b> in sidebar to pull latest Yahoo Finance prices.",
        "Review <b>scenario assumptions</b> against latest IMF, BOT, OPEC, and Fed outlooks. "
        "Adjust if needed.",
        "Review <b>Oil → CPI beta</b> and <b>EUR/USD → CPI beta</b>. Recalibrate if historical "
        "pass-through has changed.",
        "Run <b>FX Intervention Simulator</b>: check runway under all scenarios — flag any scenario "
        "where runway falls below 12 months.",
        "Export the report under <b>📥 Export to Excel</b> and attach to MPC briefing pack.",
    ]
    for i, item in enumerate(checklist, 1):
        st.markdown(
            f"<div style='margin-bottom:0.35rem;padding:0.5rem 1rem;"
            f"background:#0A1F2E;border:1px solid #1E3A5F;border-radius:6px;"
            f"border-left:3px solid #36D399;'>"
            f"<span style='color:#36D399;font-weight:700;font-size:0.85rem;'>{i}. </span>"
            f"<span style='color:#CBD5E1;font-size:0.84rem;'>{item}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='info-box' style='margin-top:1rem;border-left-color:#64748B;'>"
        "<small style='color:#64748B;'>"
        "🇱🇦 Laos FX &amp; Import Price Forecast Dashboard — "
        "FX Policy Division, Bank of Laos (BOL) &nbsp;·&nbsp; "
        f"Built: {TODAY.strftime('%B %Y')} &nbsp;·&nbsp; "
        "Data: Yahoo Finance (Yahoo! Inc.) — for internal policy analysis only, not for redistribution."
        "</small></div>",
        unsafe_allow_html=True,
    )
