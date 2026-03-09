"""
FX & Commodities Forecast Dashboard
=====================================
Live prices via Yahoo Finance · 5-Scenario monthly projections
Commodities: Energy, Metals, Agricultural
FX Pairs   : EUR/USD, GBP/USD, USD/JPY, USD/THB
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
    page_title="FX & Commodities Forecast",
    page_icon="📊",
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
            "Strong Bull":   "Significant positive price/rate movement",
            "Moderate Bull": "Mild upward price/rate pressure",
            "Base Case":     "Consensus / status-quo trajectory",
            "Moderate Bear": "Mild downward correction",
            "Strong Bear":   "Sharp sell-off / stress scenario",
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
    st.markdown("## ⚙️ Dashboard Settings")
    st.caption("Adjust assumptions and horizon below.")

    forecast_months: int = st.slider("📅 Forecast horizon (months)", 1, 6, 3)

    st.divider()
    st.markdown("### 📦 Commodity Assumptions")
    st.caption("Monthly % price change per scenario")
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
    st.markdown("### 💱 FX Assumptions")
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
    if st.button("🔄 Refresh Live Data", use_container_width=True, type="secondary"):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "📡 Data: Yahoo Finance (free, ~15-min delay).  \n"
        "Futures contracts may occasionally be unavailable."
    )

    st.divider()
    st.markdown("### 🇱🇦 Laos Focus Assumptions")
    current_inflation: float = st.number_input(
        "Current annual CPI inflation (%)",
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


# ─────────────────────────────────────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"<h2 style='color:#F5A623;margin-bottom:0.1rem;'>📊 FX & Commodities Forecast Dashboard</h2>"
    f"<p style='color:#94A3B8;font-size:0.88rem;margin-top:0.1rem;'>"
    f"📡 Live prices &nbsp;·&nbsp; 5-Scenario projections &nbsp;·&nbsp; "
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
tab_live, tab_comm, tab_fx, tab_lak, tab_summary, tab_export = st.tabs(
    [
        "📈 Live Prices",
        "📦 Commodity Scenarios",
        "💱 FX Scenarios",
        "🇱🇦 Laos Focus",
        "📋 Summary View",
        "📥 Export to Excel",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – LIVE PRICES
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown(
        "<p style='color:#94A3B8;font-size:0.84rem;margin-bottom:0.5rem;'>"
        "Prices from Yahoo Finance (~15-min delay). Futures: front-month contracts. "
        "Values in USD unless stated.</p>",
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
        "<b style='color:#60A5FA;'>ℹ️ FX Convention</b><br>"
        "EUR/USD &amp; GBP/USD = USD per foreign unit (+ scenario → USD weakens). "
        "USD/JPY, USD/THB &amp; USD/LAK = foreign units per USD (+ scenario → currency weakens). "
        "LAK/THB and Gold (LAK) are derived — see the 🇱🇦 <b>Laos Focus</b> tab."
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – COMMODITY SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_comm:
    valid_comm = {k: v for k, v in comm_prices.items() if v is not None}

    if not valid_comm:
        st.error("No commodity prices loaded. Check your connection and refresh.")
    else:
        st.subheader(f"Commodity Price Scenarios — {forecast_months}-Month Horizon")

        # ── Scenario table ────────────────────────────────────────────────────
        st.markdown("##### Projected Prices (all assets, all scenarios)")
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
            use_container_width=True,
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
            st.plotly_chart(fig, use_container_width=True)

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
                st.dataframe(pd.DataFrame(detail_rows).set_index("Scenario"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 – FX SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_fx:
    valid_fx = {k: v for k, v in fx_prices.items() if v is not None}

    if not valid_fx:
        st.error("No FX data loaded. Check your connection and refresh.")
    else:
        st.subheader(f"FX Rate Scenarios — {forecast_months}-Month Horizon")
        st.caption(
            "⚠️ Percentage changes apply to the **quoted rate**.  "
            "For EUR/USD and GBP/USD, a + scenario = USD weakens.  "
            "For USD/JPY and USD/THB, a + scenario = JPY/THB weakens."
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
            use_container_width=True,
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
            st.plotly_chart(fig, use_container_width=True)

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
                st.dataframe(pd.DataFrame(detail_rows).set_index("Scenario"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 – LAOS FOCUS
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
        st.plotly_chart(fig_lak, use_container_width=True)

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
            st.dataframe(pd.DataFrame(lak_rows).set_index("Scenario"), use_container_width=True)

    st.divider()

    # ══ 2. LAK/THB Cross Rate Chart ═══════════════════════════════════════════
    st.markdown(
        "<div class='group-header' style='border-left-color:#36D399;'>"
        "🔄 LAK/THB — Cross Rate Scenario</div>",
        unsafe_allow_html=True,
    )
    if lak_thb_eff:
        fig_lak_thb = price_path_chart(
            "LAK/THB (LAK per 1 Thai Baht)", lak_thb_eff, forecast_months, pcts_lak
        )
        st.plotly_chart(fig_lak_thb, use_container_width=True)
        st.markdown(
            "<div class='info-box'>"
            "<b style='color:#36D399;'>ℹ️ LAK/THB Methodology</b><br>"
            "Derived as (USD/LAK) ÷ (USD/THB). Scenario assumes THB/USD stays relatively "
            "stable; movement is driven by the LAK depreciation/appreciation assumptions above.<br>"
            "<b>+ monthly % → more LAK per Baht (LAK weakens against THB)</b>."
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
        st.plotly_chart(fig_gl, use_container_width=True)
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
        st.plotly_chart(fig_infl, use_container_width=True)

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
            use_container_width=True,
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


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 – SUMMARY VIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    st.subheader(f"End-of-Horizon Summary  (Month {forecast_months})")
    st.caption(
        f"Shows the projected price/rate at the end of the {forecast_months}-month "
        "horizon for each scenario, with % change from current."
    )

    valid_c = {k: v for k, v in comm_prices.items() if v is not None}
    valid_f = {k: v for k, v in fx_prices.items()   if v is not None}

    if valid_c:
        st.markdown("#### Commodities")
        df_sum_c = scenario_summary(valid_c, forecast_months, pcts_comm)
        st.dataframe(df_sum_c, use_container_width=True)

    if valid_f:
        st.markdown("#### FX Pairs")
        df_sum_f = scenario_summary(valid_f, forecast_months, pcts_fx)
        st.dataframe(df_sum_f, use_container_width=True)

    st.divider()
    st.markdown("#### Scenario Legend")
    leg_cols = st.columns(len(SCENARIOS))
    descs = {
        "Strong Bull":   "Major upside shock / supply disruption",
        "Moderate Bull": "Gradual recovery / positive sentiment",
        "Base Case":     "Consensus / status-quo trajectory",
        "Moderate Bear": "Mild correction / softening demand",
        "Strong Bear":   "Sharp sell-off / recession / stress event",
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
    st.subheader("Export Forecast Report to Excel")

    st.markdown(
        """
        The generated workbook contains **3 worksheets**:

        | Sheet | Contents |
        |-------|----------|
        | **Commodity Scenarios** | Monthly projected prices for all 11 commodities × 5 scenarios |
        | **FX Scenarios** | Monthly projected rates for all 5 FX pairs × 5 scenarios |
        | **Laos — FX & Gold (LAK)** | USD/LAK, LAK/THB, Gold (LAK/oz) projections × 5 scenarios |
        | **Laos — Inflation** | CPI index projections and scenario rate summary |
        | **Assumptions** | Scenario parameters, horizon, and generation timestamp |

        Cells are colour-coded by scenario (🟢 bull / 🔵 base / 🟠 bear / 🔴 strong bear).  
        Columns are frozen at the asset name and current price for easy scrolling.
        """
    )

    st.divider()

    if st.button("📊 Generate Excel Report", type="primary", use_container_width=True):
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
                use_container_width=True,
                type="primary",
            )
            st.success(f"✅ **{fname}** is ready for download.")
