"""
Portfolio Analytics Dashboard
Streamlit app — trading analyst / trading operations
Run: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from datetime import datetime, timedelta

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── global style — clean, institutional ──────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }
    .stMetric { background: #F7FAFC; border-radius: 6px; padding: 0.6rem 1rem; }
    .stMetric label { font-size: 11px !important; color: #718096 !important;
                      text-transform: uppercase; letter-spacing: 0.05em; }
    .stMetric [data-testid="stMetricValue"] { font-size: 22px !important;
                      color: #1B2A4A !important; font-weight: 700; }
    .stAlert { border-radius: 4px; font-size: 13px; }
    h1 { color: #1B2A4A; font-size: 20px !important; font-weight: 700;
         border-bottom: 2px solid #1B2A4A; padding-bottom: 6px; }
    h2 { color: #2D3748; font-size: 14px !important; font-weight: 600;
         text-transform: uppercase; letter-spacing: 0.06em; margin-top: 1rem; }
    .section-label { font-size: 11px; font-weight: 600; color: #718096;
                     text-transform: uppercase; letter-spacing: 0.07em;
                     border-bottom: 1px solid #E2E8F0; padding-bottom: 4px;
                     margin-bottom: 8px; }
    .insight-box { background: #EBF4FF; border-left: 3px solid #1B2A4A;
                   padding: 10px 14px; border-radius: 0 4px 4px 0;
                   font-size: 13px; color: #2D3748; line-height: 1.7; }
    div[data-testid="stDataFrame"] { border: 0.5px solid #CBD5E0;
                                      border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PORTFOLIO CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = ["SPY", "TSLA", "AAPL", "NVDA", "IEF"]

SHARES = {
    "SPY":  50,
    "TSLA": 30,
    "AAPL": 40,
    "NVDA": 15,
    "IEF":  80,
}

# Average cost basis per share (simulated entry prices)
COST_BASIS = {
    "SPY":  460.00,
    "TSLA": 210.00,
    "AAPL": 162.00,
    "NVDA": 380.00,
    "IEF":   96.50,
}

# Realized PnL from previously closed portions (simulated, would come from OMS)
REALIZED_PNL = {
    "SPY":   820.00,
    "TSLA": -340.00,
    "AAPL":  510.00,
    "NVDA": 1250.00,
    "IEF":   -95.00,
}

TARGET_WEIGHTS = {
    "SPY":  0.40,
    "TSLA": 0.15,
    "AAPL": 0.18,
    "NVDA": 0.15,
    "IEF":  0.12,
}

TKR_COLORS = {
    "SPY":  "#1B2A4A",
    "TSLA": "#9B2335",
    "AAPL": "#276749",
    "NVDA": "#6B46C1",
    "IEF":  "#B7791F",
}

RF_ANNUAL = 0.053   # risk-free rate (~Fed funds)
VAR_THRESHOLD_PCT  = 0.10   # drawdown alert threshold
CONC_THRESHOLD_PCT = 0.25   # position concentration alert


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA — live yfinance or synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_prices():
    try:
        import yfinance as yf
        raw    = yf.download(TICKERS + ["^VIX"], period="1y",
                             auto_adjust=True, progress=False)
        prices = raw["Close"][TICKERS].dropna()
        vix    = raw["Close"]["^VIX"].dropna()
        source = "live"
        return prices, vix, source
    except Exception:
        pass

    # synthetic GBM fallback
    np.random.seed(42)
    n = 252
    dates = pd.bdate_range(end=datetime.today(), periods=n)
    cfg = {
        "SPY":  {"s": 490.0, "v": 0.15, "r": 0.12},
        "TSLA": {"s": 195.0, "v": 0.65, "r": 0.10},
        "AAPL": {"s": 175.0, "v": 0.28, "r": 0.14},
        "NVDA": {"s": 520.0, "v": 0.55, "r": 0.40},
        "IEF":  {"s":  94.0, "v": 0.06, "r": 0.04},
    }
    dt = 1 / 252
    px = {}
    for t, c in cfg.items():
        dr  = (c["r"] - 0.5 * c["v"] ** 2) * dt
        px[t] = c["s"] * np.exp(
            np.cumsum(np.random.normal(dr, c["v"] * dt ** 0.5, n))
        )
    vc  = {"s": 18.0, "v": 0.80, "r": -0.10}
    dr  = (vc["r"] - 0.5 * vc["v"] ** 2) * dt
    vix = pd.Series(
        vc["s"] * np.exp(np.cumsum(np.random.normal(dr, vc["v"] * dt ** 0.5, n))),
        index=dates,
    )
    return pd.DataFrame(px, index=dates), vix, "synthetic"


# ─────────────────────────────────────────────────────────────────────────────
# 3. PnL ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def compute_pnl(prices):
    latest = prices.iloc[-1]
    prev   = prices.iloc[-2]

    # mark-to-market
    mkt_val  = {t: SHARES[t] * latest[t] for t in TICKERS}
    prev_val = {t: SHARES[t] * prev[t]   for t in TICKERS}
    nav      = sum(mkt_val.values())
    nav_prev = sum(prev_val.values())

    # unrealized = current value vs cost basis
    cost_total    = sum(COST_BASIS[t] * SHARES[t] for t in TICKERS)
    unrealized    = {t: (latest[t] - COST_BASIS[t]) * SHARES[t] for t in TICKERS}
    unrealized_ttl = sum(unrealized.values())

    # realized (from OMS / trade history)
    realized_ttl = sum(REALIZED_PNL.values())

    # total PnL
    total_pnl = unrealized_ttl + realized_ttl

    # day PnL
    day_pnl = nav - nav_prev

    return {
        "nav":           nav,
        "nav_prev":      nav_prev,
        "mkt_val":       mkt_val,
        "prev_val":      prev_val,
        "day_pnl":       day_pnl,
        "unrealized":    unrealized,
        "unrealized_ttl": unrealized_ttl,
        "realized_ttl":  realized_ttl,
        "total_pnl":     total_pnl,
        "cost_total":    cost_total,
        "latest":        latest,
        "prev":          prev,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. RISK METRICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def compute_risk(prices, nav):
    daily_rets = prices.pct_change().dropna()
    shares_s   = pd.Series(SHARES)

    port_dollar = (
        daily_rets * prices.shift(1).dropna()[TICKERS]
    ).mul(shares_s, axis=1).sum(axis=1)
    port_pct = port_dollar / nav

    # VaR 95% and 99% — historical simulation
    var_95 = float(np.percentile(port_dollar, 5))
    var_99 = float(np.percentile(port_dollar, 1))

    # Sharpe / Sortino
    rf_daily    = RF_ANNUAL / 252
    excess      = port_pct - rf_daily
    sharpe      = float(excess.mean() / port_pct.std() * np.sqrt(252))
    down_std    = port_pct[port_pct < 0].std() * np.sqrt(252)
    sortino     = float(excess.mean() * 252 / down_std)

    # annualised vol
    ann_vol = float(port_pct.std() * np.sqrt(252))

    # max drawdown
    cum     = (1 + port_pct).cumprod()
    peak    = cum.cummax()
    dd      = (cum - peak) / peak
    max_dd  = float(dd.min())
    max_dd_date = dd.idxmin()

    # annualised return
    ann_ret = float((cum.iloc[-1] ** (252 / len(cum))) - 1)
    calmar  = ann_ret / abs(max_dd)

    return {
        "port_pct":    port_pct,
        "port_dollar": port_dollar,
        "daily_rets":  daily_rets,
        "var_95":      var_95,
        "var_99":      var_99,
        "sharpe":      sharpe,
        "sortino":     sortino,
        "ann_vol":     ann_vol,
        "max_dd":      max_dd,
        "max_dd_date": max_dd_date,
        "ann_ret":     ann_ret,
        "calmar":      calmar,
        "cum":         cum,
        "drawdown":    dd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. ALERT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def check_alerts(risk, pnl, nav):
    alerts = []

    # drawdown alert
    if abs(risk["max_dd"]) > VAR_THRESHOLD_PCT:
        alerts.append({
            "level": "error",
            "msg":   f"MAX DRAWDOWN {risk['max_dd']:.2%} exceeds -{VAR_THRESHOLD_PCT:.0%} threshold. "
                     f"Review position sizing.",
        })

    # concentration alerts
    for t in TICKERS:
        wt = pnl["mkt_val"][t] / nav
        if wt > CONC_THRESHOLD_PCT:
            alerts.append({
                "level": "warning",
                "msg":   f"{t} is {wt:.1%} of portfolio — above {CONC_THRESHOLD_PCT:.0%} "
                         f"concentration limit.",
            })

    # VaR alert — if 1-day VaR > 2% of NAV
    if abs(risk["var_95"]) / nav > 0.02:
        alerts.append({
            "level": "warning",
            "msg":   f"VaR 95% (${abs(risk['var_95']):,.0f}) exceeds 2% of NAV. "
                     f"Elevated tail risk.",
        })

    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# 6. HELPER FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_usd(v, sign=True):
    s = "+" if (sign and v > 0) else ""
    return f"{s}${abs(v):,.0f}" if v >= 0 else f"-${abs(v):,.0f}"

def fmt_pct(v, sign=True):
    s = "+" if (sign and v > 0) else ""
    return f"{s}{v*100:.2f}%"

def color_val(v):
    """Return green/red delta string for st.metric."""
    if v > 0:   return f"+{v*100:.2f}%" if abs(v) < 10 else fmt_usd(v)
    elif v < 0: return f"{v*100:.2f}%" if abs(v) < 10 else fmt_usd(v)
    return "—"


# ─────────────────────────────────────────────────────────────────────────────
# 7. CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
PLOT_STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#CBD5E0", "axes.labelcolor": "#718096",
    "xtick.color": "#718096",   "ytick.color": "#718096",
    "text.color": "#2D3748",    "grid.color": "#E8EAED",
    "grid.linewidth": 0.5,      "font.family": "DejaVu Sans",
    "font.size": 8.5,           "axes.spines.top": False,
    "axes.spines.right": False,
}

def apply_style(): plt.rcParams.update(PLOT_STYLE)

def drawdown_chart(drawdown, max_dd, max_dd_date):
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 2.8))
    fig.patch.set_facecolor("white")
    dd_pct = drawdown.values * 100
    ax.fill_between(range(len(dd_pct)), dd_pct, 0, color="#9B2335", alpha=0.18)
    ax.plot(range(len(dd_pct)), dd_pct, color="#9B2335", linewidth=1.0)
    ax.axhline(0, color="#CBD5E0", linewidth=0.5)
    ax.axhline(-10, color="#9B2335", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.text(len(dd_pct) * 0.01, -10.3, "-10% threshold",
            fontsize=7, color="#9B2335", alpha=0.7, va="top")
    # annotate max
    mi = int(np.argmin(dd_pct))
    ax.annotate(f"{max_dd:.2%}", xy=(mi, dd_pct[mi]),
                xytext=(mi + len(dd_pct) * 0.06, dd_pct[mi] + 1.5),
                fontsize=8, color="#9B2335", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#9B2335", lw=0.7))
    n_ticks = 6
    ti = np.linspace(0, len(drawdown) - 1, n_ticks).astype(int)
    ax.set_xticks(ti)
    ax.set_xticklabels([drawdown.index[i].strftime("%b %y") for i in ti], fontsize=7.5)
    ax.set_ylabel("Drawdown %", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def allocation_chart(mkt_val, nav):
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_facecolor("white")
    ys    = np.arange(len(TICKERS))
    bar_h = 0.28
    for i, t in enumerate(TICKERS):
        aw = mkt_val[t] / nav * 100
        tw = TARGET_WEIGHTS[t] * 100
        ax.barh(ys[i] + bar_h / 2, tw, height=bar_h * 0.75,
                color="none", edgecolor=TKR_COLORS[t], linewidth=1.2,
                linestyle="--", alpha=0.6)
        ax.barh(ys[i] - bar_h / 2, aw, height=bar_h * 0.75,
                color=TKR_COLORS[t], alpha=0.82, edgecolor="none")
        ax.text(max(aw, tw) + 0.5, ys[i] - bar_h / 2,
                f"{aw:.1f}%", va="center", fontsize=8,
                color=TKR_COLORS[t], fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels(TICKERS, fontsize=9, fontweight="bold")
    ax.set_xlabel("Weight (%)", fontsize=8)
    ax.set_xlim(0, 56)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    # legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend = [
        Patch(facecolor="#718096", alpha=0.82, label="Actual"),
        Line2D([0], [0], color="#718096", linestyle="--", linewidth=1.2,
               alpha=0.6, label="Target"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8,
              framealpha=0, labelcolor="#718096")
    ax.set_ylim(-0.7, len(TICKERS) - 0.2)
    fig.tight_layout()
    return fig


def vix_chart(vix_s):
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor("white")
    vals   = vix_s.values
    x_rng  = range(len(vals))
    ax.axhspan(0,  15, alpha=0.06, color="#276749")
    ax.axhspan(15, 25, alpha=0.04, color="#B7791F")
    ax.axhspan(25, max(vals) * 1.15, alpha=0.05, color="#9B2335")
    ax.axhline(15, color="#276749", linewidth=0.6, linestyle=":", alpha=0.7)
    ax.axhline(25, color="#9B2335", linewidth=0.6, linestyle=":", alpha=0.7)
    for i in range(len(vals) - 1):
        col = "#9B2335" if vals[i] > 25 else ("#B7791F" if vals[i] > 15 else "#276749")
        ax.plot([i, i + 1], [vals[i], vals[i + 1]],
                color=col, linewidth=1.0, solid_capstyle="round")
    ax.fill_between(x_rng, vals, 0, alpha=0.07, color="#718096")
    n_ticks = 6
    ti = np.linspace(0, len(vix_s) - 1, n_ticks).astype(int)
    ax.set_xticks(ti)
    ax.set_xticklabels([vix_s.index[i].strftime("%b %y") for i in ti], fontsize=7.5)
    ax.set_ylabel("VIX", fontsize=8)
    ax.set_xlim(0, len(vals) - 1)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def pnl_bar_chart(unrealized, realized):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    fig.patch.set_facecolor("white")

    # unrealized by ticker
    ax1 = axes[0]
    vals = [unrealized[t] for t in TICKERS]
    cols = ["#276749" if v >= 0 else "#9B2335" for v in vals]
    ax1.bar(TICKERS, vals, color=cols, alpha=0.82, edgecolor="none", width=0.55)
    ax1.axhline(0, color="#CBD5E0", linewidth=0.6)
    for i, v in enumerate(vals):
        ax1.text(i, v + (abs(v) * 0.05 if v >= 0 else -abs(v) * 0.05),
                 fmt_usd(v), ha="center", fontsize=8, color=cols[i], fontweight="bold",
                 va="bottom" if v >= 0 else "top")
    ax1.set_title("Unrealized P&L by position", fontsize=8.5, color="#2D3748", pad=6)
    ax1.set_ylabel("$", fontsize=8)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.set_axisbelow(True)

    # realized by ticker
    ax2 = axes[1]
    rvals = [REALIZED_PNL[t] for t in TICKERS]
    rcols = ["#276749" if v >= 0 else "#9B2335" for v in rvals]
    ax2.bar(TICKERS, rvals, color=rcols, alpha=0.82, edgecolor="none", width=0.55)
    ax2.axhline(0, color="#CBD5E0", linewidth=0.6)
    for i, v in enumerate(rvals):
        ax2.text(i, v + (abs(v) * 0.05 if v >= 0 else -abs(v) * 0.05),
                 fmt_usd(v), ha="center", fontsize=8, color=rcols[i], fontweight="bold",
                 va="bottom" if v >= 0 else "top")
    ax2.set_title("Realized P&L by position", fontsize=8.5, color="#2D3748", pad=6)
    ax2.set_ylabel("$", fontsize=8)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.set_axisbelow(True)

    fig.tight_layout(pad=1.2)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # header
    col_title, col_ts = st.columns([5, 1])
    with col_title:
        st.markdown("## PORTFOLIO ANALYTICS DASHBOARD")
    with col_ts:
        st.markdown(
            f"<p style='text-align:right; color:#718096; font-size:12px;"
            f"margin-top:14px'>{datetime.now().strftime('%d %b %Y  %H:%M')}</p>",
            unsafe_allow_html=True,
        )

    # load data
    with st.spinner("Loading market data..."):
        prices, vix_s, data_src = load_prices()

    if data_src == "synthetic":
        st.caption("Market data unavailable — running on synthetic GBM data. "
                   "Results are illustrative only.")

    # compute everything
    pnl  = compute_pnl(prices)
    risk = compute_risk(prices, pnl["nav"])

    nav      = pnl["nav"]
    day_pnl  = pnl["day_pnl"]
    unreal   = pnl["unrealized_ttl"]
    real     = pnl["realized_ttl"]
    total    = pnl["total_pnl"]

    # ── ALERTS ───────────────────────────────────────────────────────────────
    alerts = check_alerts(risk, pnl, nav)
    if alerts:
        st.markdown("### Alerts")
        for a in alerts:
            if a["level"] == "error":
                st.error(a["msg"])
            else:
                st.warning(a["msg"])

    # ── PnL ENGINE — top strip ────────────────────────────────────────────────
    st.markdown('<p class="section-label">P&L Engine</p>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total NAV",      fmt_usd(nav, sign=False),
              fmt_pct(day_pnl / pnl["nav_prev"]))
    c2.metric("Day P&L",        fmt_usd(day_pnl),
              f"{day_pnl / pnl['nav_prev'] * 100:+.2f}% on NAV")
    c3.metric("Unrealized P&L", fmt_usd(unreal),
              f"vs cost basis ${pnl['cost_total']:,.0f}")
    c4.metric("Realized P&L",   fmt_usd(real),
              "closed positions")
    c5.metric("Total P&L",      fmt_usd(total),
              "Realized + Unrealized")

    st.markdown("---")

    # ── POSITION TABLE + ALLOCATION CHART ────────────────────────────────────
    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    with left_col:
        st.markdown("### Position Breakdown")

        rows = []
        for t in TICKERS:
            mv   = pnl["mkt_val"][t]
            wt   = mv / nav
            dpnl = mv - pnl["prev_val"][t]
            dret = dpnl / pnl["prev_val"][t]
            upnl = pnl["unrealized"][t]
            rpnl = REALIZED_PNL[t]
            rows.append({
                "Ticker":       t,
                "Shares":       SHARES[t],
                "Avg Cost":     f"${COST_BASIS[t]:.2f}",
                "Price":        f"${pnl['latest'][t]:.2f}",
                "Mkt Value":    f"${mv:,.0f}",
                "Weight":       f"{wt:.1%}",
                "Day P&L":      fmt_usd(dpnl),
                "Day Ret":      fmt_pct(dret),
                "Unrealized":   fmt_usd(upnl),
                "Realized":     fmt_usd(rpnl),
            })

        df_pos = pd.DataFrame(rows).set_index("Ticker")
        st.dataframe(df_pos, use_container_width=True, height=220)

        # PnL bar charts
        st.markdown("### P&L by Position")
        st.pyplot(pnl_bar_chart(pnl["unrealized"], REALIZED_PNL), use_container_width=True)

    with right_col:
        st.markdown("### Allocation  Target vs Actual")
        st.pyplot(allocation_chart(pnl["mkt_val"], nav), use_container_width=True)

        st.markdown("### Allocation Drift")
        drift_rows = []
        for t in TICKERS:
            aw  = pnl["mkt_val"][t] / nav
            tw  = TARGET_WEIGHTS[t]
            bps = (aw - tw) * 10000
            breach = abs(bps) > 500
            drift_rows.append({
                "Ticker":     t,
                "Target":     f"{tw:.1%}",
                "Actual":     f"{aw:.1%}",
                "Drift (bps)": f"{bps:+.0f}",
                "Status":     "REBALANCE" if breach else "On target",
            })
        df_drift = pd.DataFrame(drift_rows).set_index("Ticker")
        st.dataframe(df_drift, use_container_width=True, height=218)

    st.markdown("---")

    # ── RISK METRICS ─────────────────────────────────────────────────────────
    st.markdown("### Risk Metrics")
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric("Sharpe",        f"{risk['sharpe']:.2f}")
    r2.metric("Sortino",       f"{risk['sortino']:.2f}")
    r3.metric("Calmar",        f"{risk['calmar']:.2f}")
    r4.metric("Ann. Vol",      f"{risk['ann_vol']:.2%}")
    r5.metric("VaR 95% (1d)", fmt_usd(risk["var_95"]),
              f"{risk['var_95']/nav:.2%} of NAV")
    r6.metric("Max Drawdown",  f"{risk['max_dd']:.2%}",
              risk["max_dd_date"].strftime("%d %b %y"))

    st.markdown("---")

    # ── CHARTS ROW ────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns([1.2, 0.8], gap="large")

    with ch1:
        st.markdown("### Drawdown from Peak")
        st.pyplot(
            drawdown_chart(risk["drawdown"], risk["max_dd"], risk["max_dd_date"]),
            use_container_width=True,
        )

    with ch2:
        vix_now = float(vix_s.iloc[-1])
        reg_col = ("#9B2335" if vix_now > 25 else
                   ("#276749" if vix_now <= 15 else "#B7791F"))
        reg_lbl = ("ELEVATED" if vix_now > 25 else
                   ("SUPPRESSED" if vix_now <= 15 else "NORMAL"))
        st.markdown(
            f"### VIX  "
            f"<span style='color:{reg_col};font-size:13px;font-weight:700'>"
            f"{vix_now:.1f}  {reg_lbl}</span>",
            unsafe_allow_html=True,
        )
        st.pyplot(vix_chart(vix_s), use_container_width=True)

    st.markdown("---")

    # ── INSIGHTS SECTION ─────────────────────────────────────────────────────
    st.markdown("### Insights")

    # largest position
    largest_t = max(TICKERS, key=lambda t: pnl["mkt_val"][t])
    largest_wt = pnl["mkt_val"][largest_t] / nav

    # biggest unrealized move
    best_t  = max(TICKERS, key=lambda t: pnl["unrealized"][t])
    worst_t = min(TICKERS, key=lambda t: pnl["unrealized"][t])

    # risk regime
    if vix_now > 25:
        vol_context = (f"VIX at {vix_now:.1f} signals an elevated volatility regime. "
                       f"Options are expensive — selling premium is unfavorable here.")
    elif vix_now < 15:
        vol_context = (f"VIX at {vix_now:.1f} is suppressed. Vol sellers are crowded — "
                       f"watch for gap risk if the regime shifts.")
    else:
        vol_context = (f"VIX at {vix_now:.1f} is in a normal regime. "
                       f"Reasonable options premium environment.")

    drawdown_context = (
        f"Current max drawdown is {risk['max_dd']:.2%}. "
        + ("Above the 10% alert threshold — consider reducing gross exposure."
           if abs(risk["max_dd"]) > 0.10 else
           "Within acceptable range.")
    )

    insight_text = (
        f"<b>Volatility regime:</b> {vol_context}<br>"
        f"<b>Largest exposure:</b> {largest_t} at {largest_wt:.1%} of NAV "
        f"(${pnl['mkt_val'][largest_t]:,.0f})"
        + (" — above 25% concentration limit." if largest_wt > CONC_THRESHOLD_PCT else ".")
        + f"<br>"
        f"<b>PnL drivers:</b> {best_t} is the largest unrealized winner "
        f"({fmt_usd(pnl['unrealized'][best_t])}); "
        f"{worst_t} is the largest drag "
        f"({fmt_usd(pnl['unrealized'][worst_t])}). "
        f"{drawdown_context}"
    )

    st.markdown(
        f'<div class="insight-box">{insight_text}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── FULL RISK TABLE ───────────────────────────────────────────────────────
    with st.expander("Full risk detail"):
        risk_rows = {
            "Annualised Return":   f"{risk['ann_ret']:.2%}",
            "Annualised Vol":      f"{risk['ann_vol']:.2%}",
            "Sharpe Ratio":        f"{risk['sharpe']:.4f}",
            "Sortino Ratio":       f"{risk['sortino']:.4f}",
            "Calmar Ratio":        f"{risk['calmar']:.4f}",
            "Max Drawdown":        f"{risk['max_dd']:.2%}",
            "Max DD Date":         risk["max_dd_date"].strftime("%d %b %Y"),
            "VaR 95% (1-day $)":   fmt_usd(risk["var_95"]),
            "VaR 99% (1-day $)":   fmt_usd(risk["var_99"]),
            "VaR 95% (% NAV)":     f"{abs(risk['var_95']/nav):.2%}",
            "Risk-Free Rate":      f"{RF_ANNUAL:.1%}",
        }
        df_risk = pd.DataFrame.from_dict(
            risk_rows, orient="index", columns=["Value"]
        )
        st.dataframe(df_risk, use_container_width=True)

    st.caption(
        f"Data source: {data_src}  |  "
        f"As of {prices.index[-1].strftime('%d %b %Y')}  |  "
        f"Risk-free rate {RF_ANNUAL:.1%} (Fed funds approximation)  |  "
        "VaR via historical simulation — no normality assumption."
    )


if __name__ == "__main__":
    main()
