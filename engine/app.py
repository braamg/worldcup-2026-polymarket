"""
app.py — WC2026 V2 High-Yield Dashboard
========================================
Entry point. Run with:
    streamlit run app.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import strategy_config as cfg
from utils.data_engine import (
    get_all_qualifying, compute_all, DEFAULT_MIN_WIN_PROB, DEFAULT_MAX_DAYS,
)
from components.views import (
    render_picks_tab, render_system_overview_tab, render_dictionary_tab, render_verdict_tab,
)
from model_loader import load_model_probs, load_match_probs
from polymarket_client import fetch_futures_prices, fetch_match_prices
from utils.forecast_log import snapshot_forecasts

# futures market_type -> approximate resolution date (informational in the forecast log)
_FUT_RESOLVE = {
    "advance":     "2026-06-27", "reach_r16": "2026-07-03", "reach_qf": "2026-07-07",
    "reach_sf":    "2026-07-11", "reach_final": "2026-07-15", "champion": "2026-07-19",
}

# market_type (model/Polymarket) -> human "match" label the engine + views expect
_MTYPE_TO_MATCH = {
    "champion":     "Champion market",
    "reach_final":  "Reach Final",
    "reach_sf":     "Reach Semi-Finals",
    "reach_qf":     "Reach Quarter-Finals",
    "reach_r16":    "Reach Round of 16",
    "advance":      "Group Stage",
}

# Polymarket futures event slug per market label (for the "open on Polymarket" link)
_FUT_SLUG = {
    "Champion market":      "world-cup-winner",
    "Reach Final":          "world-cup-nation-to-reach-final",
    "Reach Semi-Finals":    "world-cup-nation-to-reach-semifinals",
    "Reach Quarter-Finals": "world-cup-nation-to-reach-quarterfinals",
    "Reach Round of 16":    "world-cup-nation-to-reach-round-of-16",
    "Group Stage":          "world-cup-team-to-advance-to-knockout-stages",
}


def _event_url(bet_type: str, match_label: str, event_slug: str):
    if bet_type == "Match":
        return f"https://polymarket.com/event/{event_slug}" if str(event_slug).strip() else None
    slug = _FUT_SLUG.get(match_label)
    return f"https://polymarket.com/event/{slug}" if slug else None

st.set_page_config(
    page_title="WC2026 · High-Yield V2",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light global styling — tighter spacing, constrained width, calmer metrics.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1080px; }
      [data-testid="stMetricValue"] { font-size: 1.35rem; }
      [data-testid="stMetricLabel"] { opacity: 0.75; }
      h2 { padding-top: 0.2rem; }
      hr { margin: 1.1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — bankroll + risk-adjustment configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Controls")
    bankroll_zar = st.number_input(
        "Bankroll (R)", min_value=50.0, max_value=10_000.0,
        value=400.0, step=50.0, format="%.0f",
        help="Split equally across the bets in your blend.",
    )
    min_win_prob = st.slider(
        "Minimum Win Probability", 0.05, 0.50, DEFAULT_MIN_WIN_PROB, 0.01,
        help="Drop any bet whose used (bet-on) probability is below this — keeps you off "
             "longshots that almost certainly miss on a one-shot event.",
    )
    betting_horizon = st.slider(
        "Betting Horizon (days)", 3, 40, DEFAULT_MAX_DAYS, 1,
        help="Only show bets that settle within this many days. Lower = near-term matches; "
             "higher = include long-dated tournament futures (champion ~29d).",
    )

    st.divider()
    if st.button("🔄 Refresh live prices"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Other strategy knobs (shrinkage, Kelly, match discipline) are hardcoded in "
               "`strategy_config.py` — edit there if needed.")

_ADJ = dict(
    model_trust=cfg.MODEL_TRUST,
    kelly_multiplier=cfg.KELLY_MULTIPLIER,
    selection_haircut=cfg.SELECTION_HAIRCUT,
    spread_buffer_cents=cfg.SPREAD_BUFFER_CENTS,
)

# ---------------------------------------------------------------------------
# Live data loaders (cached) + merge into the engine's expected schema.
# Falls back to the mock set below if live prices or model probs are unavailable.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _load_market() -> pd.DataFrame:
    """Live Polymarket futures prices (public API). Empty DataFrame on failure."""
    try:
        return fetch_futures_prices()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_model() -> pd.DataFrame:
    """Monte Carlo model probabilities from sim_probabilities.json."""
    return load_model_probs()


def _build_live_df(model_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge model probabilities with live prices into the engine schema:
        match, selection, model_prob, polymarket_cents
    Returns an empty DataFrame if there is no overlap (→ caller falls back to mock).
    """
    if model_df.empty or market_df.empty:
        return pd.DataFrame()
    merged = model_df.merge(market_df[["market_id", "price"]], on="market_id", how="inner")
    if merged.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "match":            merged["market_type"].map(_MTYPE_TO_MATCH),
        "selection":        merged["description"],
        "model_prob":       merged["model_prob"].astype(float),
        "polymarket_cents": (merged["price"].astype(float) * 100).round(1),
    })
    return out.dropna(subset=["match"]).reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def _load_match_market() -> pd.DataFrame:
    """Live Polymarket single-match moneyline prices. Empty DataFrame on failure."""
    try:
        return fetch_match_prices()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_match_model() -> pd.DataFrame:
    """Model 3-way match probabilities from match_probabilities.json."""
    return load_match_probs()


def _build_match_df(model_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge match model probs with live match prices into the engine schema, carrying the
    match date as `resolve_date` (so the betting horizon works on real match dates).
        match='Match', selection, model_prob, polymarket_cents, resolve_date, group_id
    """
    if model_df.empty or market_df.empty:
        return pd.DataFrame()
    price_cols = ["market_id", "price"] + (["event_slug"] if "event_slug" in market_df.columns else [])
    merged = model_df.merge(market_df[price_cols], on="market_id", how="inner")
    if merged.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "match":            "Match",
        "selection":        merged["selection"],
        "model_prob":       merged["model_prob"].astype(float),
        "polymarket_cents": (merged["price"].astype(float) * 100).round(1),
        "resolve_date":     merged["match_date"],
        "group_id":         merged["group_id"],
        "event_slug":       merged["event_slug"] if "event_slug" in merged.columns else "",
    })
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Mock data — fallback only, used when live prices/model are unavailable.
# Covers all 6 market types; model_prob = simulator output, cents = Yes price.
# ---------------------------------------------------------------------------

_MOCK: list[dict] = [
    # ── To win the tournament (champion) ─────────────────────────────────────
    {"match": "Champion market",  "selection": "Morocco to win the tournament",     "model_prob": 0.09, "polymarket_cents": 5},
    {"match": "Champion market",  "selection": "Japan to win the tournament",       "model_prob": 0.07, "polymarket_cents": 4},
    {"match": "Champion market",  "selection": "USA to win the tournament",         "model_prob": 0.11, "polymarket_cents": 7},
    {"match": "Champion market",  "selection": "Senegal to win the tournament",     "model_prob": 0.06, "polymarket_cents": 3},
    {"match": "Champion market",  "selection": "Colombia to win the tournament",    "model_prob": 0.08, "polymarket_cents": 6},
    {"match": "Champion market",  "selection": "Mexico to win the tournament",      "model_prob": 0.05, "polymarket_cents": 5},
    {"match": "Champion market",  "selection": "Nigeria to win the tournament",     "model_prob": 0.04, "polymarket_cents": 3},
    {"match": "Champion market",  "selection": "South Korea to win tournament",     "model_prob": 0.05, "polymarket_cents": 4},
    {"match": "Champion market",  "selection": "Ecuador to win the tournament",     "model_prob": 0.04, "polymarket_cents": 6},
    # Favourites — above 40¢, filtered out by engine
    {"match": "Champion market",  "selection": "Brazil to win the tournament",      "model_prob": 0.19, "polymarket_cents": 18},
    {"match": "Champion market",  "selection": "France to win the tournament",      "model_prob": 0.17, "polymarket_cents": 16},
    {"match": "Champion market",  "selection": "Argentina to win the tournament",   "model_prob": 0.15, "polymarket_cents": 22},
    {"match": "Champion market",  "selection": "England to win the tournament",     "model_prob": 0.12, "polymarket_cents": 14},
    {"match": "Champion market",  "selection": "Spain to win the tournament",       "model_prob": 0.14, "polymarket_cents": 20},
    {"match": "Champion market",  "selection": "Germany to win the tournament",     "model_prob": 0.10, "polymarket_cents": 12},

    # ── Reach the final ───────────────────────────────────────────────────────
    {"match": "Reach Final",      "selection": "Morocco to reach the Final",        "model_prob": 0.21, "polymarket_cents": 13},
    {"match": "Reach Final",      "selection": "USA to reach the Final",            "model_prob": 0.24, "polymarket_cents": 15},
    {"match": "Reach Final",      "selection": "Japan to reach the Final",          "model_prob": 0.18, "polymarket_cents": 11},
    {"match": "Reach Final",      "selection": "Colombia to reach the Final",       "model_prob": 0.20, "polymarket_cents": 16},
    {"match": "Reach Final",      "selection": "Senegal to reach the Final",        "model_prob": 0.16, "polymarket_cents": 9},
    {"match": "Reach Final",      "selection": "South Korea to reach the Final",    "model_prob": 0.14, "polymarket_cents": 10},
    # Overpriced — negative EV
    {"match": "Reach Final",      "selection": "Mexico to reach the Final",         "model_prob": 0.13, "polymarket_cents": 19},
    {"match": "Reach Final",      "selection": "Nigeria to reach the Final",        "model_prob": 0.10, "polymarket_cents": 14},

    # ── Reach Semi-Finals ─────────────────────────────────────────────────────
    {"match": "Reach Semi-Finals","selection": "Morocco to reach the Semis",        "model_prob": 0.38, "polymarket_cents": 24},
    {"match": "Reach Semi-Finals","selection": "USA to reach the Semis",            "model_prob": 0.41, "polymarket_cents": 27},
    {"match": "Reach Semi-Finals","selection": "Japan to reach the Semis",          "model_prob": 0.33, "polymarket_cents": 20},
    {"match": "Reach Semi-Finals","selection": "Colombia to reach the Semis",       "model_prob": 0.36, "polymarket_cents": 28},
    {"match": "Reach Semi-Finals","selection": "Senegal to reach the Semis",        "model_prob": 0.30, "polymarket_cents": 18},
    {"match": "Reach Semi-Finals","selection": "Ecuador to reach the Semis",        "model_prob": 0.22, "polymarket_cents": 14},
    {"match": "Reach Semi-Finals","selection": "South Korea to reach Semis",        "model_prob": 0.27, "polymarket_cents": 17},
    {"match": "Reach Semi-Finals","selection": "Nigeria to reach the Semis",        "model_prob": 0.24, "polymarket_cents": 15},
    {"match": "Reach Semi-Finals","selection": "Canada to reach the Semis",         "model_prob": 0.28, "polymarket_cents": 21},
    # Overpriced — negative EV
    {"match": "Reach Semi-Finals","selection": "Mexico to reach the Semis",         "model_prob": 0.29, "polymarket_cents": 38},
    {"match": "Reach Semi-Finals","selection": "Poland to reach the Semis",         "model_prob": 0.18, "polymarket_cents": 31},

    # ── Reach Quarter-Finals ──────────────────────────────────────────────────
    {"match": "Reach Quarter-Finals","selection": "Morocco to reach the QF",        "model_prob": 0.55, "polymarket_cents": 35},
    {"match": "Reach Quarter-Finals","selection": "Japan to reach the QF",          "model_prob": 0.48, "polymarket_cents": 30},
    {"match": "Reach Quarter-Finals","selection": "USA to reach the QF",            "model_prob": 0.58, "polymarket_cents": 39},
    {"match": "Reach Quarter-Finals","selection": "Senegal to reach the QF",        "model_prob": 0.44, "polymarket_cents": 28},
    {"match": "Reach Quarter-Finals","selection": "Colombia to reach the QF",       "model_prob": 0.51, "polymarket_cents": 37},
    {"match": "Reach Quarter-Finals","selection": "Ecuador to reach the QF",        "model_prob": 0.38, "polymarket_cents": 22},
    {"match": "Reach Quarter-Finals","selection": "South Korea to reach the QF",    "model_prob": 0.42, "polymarket_cents": 26},
    {"match": "Reach Quarter-Finals","selection": "Nigeria to reach the QF",        "model_prob": 0.37, "polymarket_cents": 23},
    {"match": "Reach Quarter-Finals","selection": "Canada to reach the QF",         "model_prob": 0.45, "polymarket_cents": 31},
    {"match": "Reach Quarter-Finals","selection": "Australia to reach the QF",      "model_prob": 0.33, "polymarket_cents": 20},
    {"match": "Reach Quarter-Finals","selection": "Turkey to reach the QF",         "model_prob": 0.36, "polymarket_cents": 24},
    # Overpriced — negative EV
    {"match": "Reach Quarter-Finals","selection": "Mexico to reach the QF",         "model_prob": 0.40, "polymarket_cents": 55},
    {"match": "Reach Quarter-Finals","selection": "Poland to reach the QF",         "model_prob": 0.28, "polymarket_cents": 48},

    # ── Reach Round of 16 ─────────────────────────────────────────────────────
    {"match": "Reach Round of 16","selection": "Morocco to reach R16",              "model_prob": 0.74, "polymarket_cents": 60},
    {"match": "Reach Round of 16","selection": "Nigeria to reach R16",              "model_prob": 0.62, "polymarket_cents": 38},
    {"match": "Reach Round of 16","selection": "Ecuador to reach R16",              "model_prob": 0.58, "polymarket_cents": 34},
    {"match": "Reach Round of 16","selection": "South Korea to reach R16",          "model_prob": 0.65, "polymarket_cents": 39},
    {"match": "Reach Round of 16","selection": "Turkey to reach R16",               "model_prob": 0.60, "polymarket_cents": 35},
    {"match": "Reach Round of 16","selection": "Australia to reach R16",            "model_prob": 0.55, "polymarket_cents": 32},
    # Overpriced — negative EV, or above 40¢
    {"match": "Reach Round of 16","selection": "Brazil to reach R16",               "model_prob": 0.92, "polymarket_cents": 80},
    {"match": "Reach Round of 16","selection": "France to reach R16",               "model_prob": 0.88, "polymarket_cents": 76},

    # ── Advance from group ────────────────────────────────────────────────────
    {"match": "Group Stage",      "selection": "Senegal to advance from group",     "model_prob": 0.72, "polymarket_cents": 55},
    {"match": "Group Stage",      "selection": "Nigeria to advance from group",     "model_prob": 0.58, "polymarket_cents": 38},
    {"match": "Group Stage",      "selection": "Ecuador to advance from group",     "model_prob": 0.63, "polymarket_cents": 39},
    {"match": "Group Stage",      "selection": "Turkey to advance from group",      "model_prob": 0.55, "polymarket_cents": 35},
    {"match": "Group Stage",      "selection": "Australia to advance from group",   "model_prob": 0.52, "polymarket_cents": 32},
    {"match": "Group Stage",      "selection": "South Korea to advance",            "model_prob": 0.60, "polymarket_cents": 37},
    # Overpriced
    {"match": "Group Stage",      "selection": "Brazil to advance from group",      "model_prob": 0.96, "polymarket_cents": 88},
]

# Prefer live data; fall back to mock when unavailable.
_model_df = _load_model()
_market_df = _load_market()
_live_df = _build_live_df(_model_df, _market_df)

if not _live_df.empty:
    raw_df = _live_df
    data_source = "live"
else:
    raw_df = pd.DataFrame(_MOCK)
    data_source = "mock"

# ── Tournament futures (high-yield <40¢) ──────────────────────────────────
fut_q = get_all_qualifying(raw_df, min_win_prob=min_win_prob, max_days=betting_horizon,
                           max_cents=cfg.FUTURES_MAX_CENTS, **_ADJ)
fut_q = fut_q.copy()
fut_q["bet_type"] = "Future"

# ── Single matches (3-way moneyline) + discipline (hardcoded in strategy_config) ──
_match_model_df = _load_match_model()
_match_market_df = _load_match_market()
_match_df = _build_match_df(_match_model_df, _match_market_df)

# Forecast snapshot for the scientific verdict — capture model+market for EVERY market today,
# deduped to one row per (market_id, date). Only on live data, so mock never pollutes the log.
if data_source == "live":
    try:
        _added = snapshot_forecasts(
            _match_model_df, _match_market_df, datetime.now().strftime("%Y-%m-%d"),
            futures_model_df=_model_df, futures_market_df=_market_df,
            stage_resolve_dates=_FUT_RESOLVE,
        )
    except Exception:
        pass  # never let logging break the dashboard
if not _match_df.empty:
    mq = get_all_qualifying(_match_df, min_win_prob=min_win_prob, max_days=betting_horizon,
                            max_cents=cfg.MATCH_MAX_CENTS, **_ADJ)
    _is_draw = mq["selection"].str.contains("draw", case=False, na=False)
    _wins = mq[~_is_draw].copy()
    _draws = mq[_is_draw].copy()
    # Win discipline: contender floor
    _wins = _wins[_wins["model_prob"] >= cfg.MATCH_CONTENDER_FLOOR]
    # Draw discipline: must be at/above observed base rate AND model > market by 5pp
    if not cfg.EXCLUDE_DRAWS:
        _draws = _draws[_draws["model_prob"] >= cfg.DRAW_MIN_MODEL_PROB]
        _draws = _draws[(_draws["model_prob"] - _draws["implied_prob"]) * 100 >= cfg.DRAW_MIN_EDGE_PP]
    else:
        _draws = pd.DataFrame()
    mq = pd.concat([_wins, _draws], ignore_index=True)
    if cfg.EXCLUDE_INPLAY:
        mq = mq[mq["days_to_resolve"] >= 1]
    mq = mq.copy()
    mq["bet_type"] = "Match"
else:
    mq = pd.DataFrame()

# ── Combined picks (ranked by hit rate) + a Polymarket link per row ───────────
combined_df = pd.concat([fut_q, mq], ignore_index=True)
if not combined_df.empty:
    if "event_slug" not in combined_df.columns:
        combined_df["event_slug"] = ""
    combined_df["event_slug"] = combined_df["event_slug"].fillna("")
    combined_df["event_url"] = [
        _event_url(bt, m, es) for bt, m, es in
        zip(combined_df["bet_type"], combined_df["match"], combined_df["event_slug"])
    ]
    combined_df = combined_df.sort_values("model_shrunk", ascending=False).reset_index(drop=True)

# ── Full scan (everything computed, for the transparency expander) ────────────
_fut_scan = compute_all(raw_df, **_ADJ)
_fut_scan["bet_type"] = "Future"
if not _match_df.empty:
    _mt_scan = compute_all(_match_df, **_ADJ)
    _mt_scan["bet_type"] = "Match"
    combined_scan_df = pd.concat([_fut_scan, _mt_scan], ignore_index=True)
else:
    combined_scan_df = _fut_scan

# Model staleness — age of the simulator output on disk.
_SIM_PATH = ROOT / "sim_probabilities.json"
if _SIM_PATH.exists():
    mtime = datetime.fromtimestamp(_SIM_PATH.stat().st_mtime)
    age_days = (datetime.now() - mtime).days
    _model_age = f"{mtime:%Y-%m-%d %H:%M} ({age_days}d ago)"
else:
    _model_age = None

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_picks, tab_sys, tab_verdict, tab_dict = st.tabs([
    "⚡ Picks",
    "📊 System Overview",
    "📋 Verdict",
    "📖 Dictionary",
])

with tab_picks:
    if data_source == "mock":
        st.error(
            "**Showing MOCK data** — live prices and/or model probabilities unavailable. "
            "**Do not place real bets.** Click *Refresh live prices* in the sidebar.",
            icon="🚨",
        )
    else:
        n_picks = 0 if combined_df.empty else len(combined_df)
        st.caption(
            f"🟢 Live data — {len(_market_df)} futures + {len(_match_market_df)} match prices "
            f"→ **{n_picks}** +EV picks within your filters."
        )
    render_picks_tab(combined_df, combined_scan_df, bankroll_zar=bankroll_zar)

with tab_sys:
    render_system_overview_tab(
        model_age_str=_model_age,
        match_probs_df=_match_model_df,
        market_df=_match_market_df,
    )

with tab_verdict:
    render_verdict_tab()

with tab_dict:
    render_dictionary_tab()
