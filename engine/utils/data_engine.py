"""
utils/data_engine.py
====================
Metric engine for the V2 High-Yield strategy.

This module computes BOTH a naive and a risk-adjusted view of every contract, because the
naive view (raw EV vs. mid-price) systematically overstates edge. See STRATEGY_REVIEW.md.
Adjustments applied:

1. Probability shrinkage toward the market (`model_trust` w): p* = w·model + (1−w)·market.
2. Selection-bias haircut (winner's curse): shrink the edge by `selection_haircut`.
3. Executable-price buffer: EV/Kelly use exec_price = (cents + spread_buffer)/100 (the ask).
4. Fractional Kelly (`kelly_multiplier`): size on a fraction of full Kelly.
5. Monte Carlo sampling error (`mc_se`): simulation noise, surfaced separately from model error.

ONCE-OFF FINITE-EVENT LOGIC (the World Cup is a single tournament, NOT a repeatable edge):
  There is no Law of Large Numbers to rescue a miss — capital lost on a longshot is gone with
  little chance to recover. So two extra controls protect a small finite bankroll:

  6. Hit-rate floor (`min_win_prob`): a 10% chance means this ONE bet hits 1-in-10, not "wins
     10% of the time." We DROP any contract whose **used (shrunk) probability** — the chance we
     actually act on, after hedging the model toward the market — is below the floor (default
     20%), regardless of EV. Flooring on the USED probability (not the raw model) is deliberate:
     a card that says "model 23%" but is really a 13%-used bet should NOT pass a 20% floor.

  7. Recommended bet size (`recommended_bet_size`): the conservative (shrunk, executable-price)
     Kelly fraction × kelly_multiplier, CAPPED at MAX_BET_FRAC of bankroll. Kelly shrinks fast as
     win-probability drops, and the cap keeps any single once-off bet small — survive to learn.

Filter: price < MAX_CENTS, adjusted EV > MIN_EV, AND used probability >= min_win_prob.
Ranked by **hit rate (used probability) descending** — surface the bets most likely to actually
resolve, which is what a finite bankroll needs (you can't average out a miss).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.schedule import days_until

# Strategy / filter parameters
TOP_N      = 5
MAX_CENTS  = 40
MIN_EV     = 0.0      # adjusted-EV floor (already net of shrinkage + costs)

# Adjustment defaults (overridable per call from the sidebar)
DEFAULT_MODEL_TRUST       = 0.35  # w in the shrinkage blend
DEFAULT_KELLY_MULTIPLIER  = 0.25  # fraction of full Kelly used for sizing (quarter-Kelly)
DEFAULT_SELECTION_HAIRCUT = 0.25  # shrink apparent edge 25% for winner's curse
DEFAULT_SPREAD_BUFFER_C   = 2.0   # cents added to price to approximate the ask
DEFAULT_MIN_WIN_PROB      = 0.20  # hit-rate floor on the USED (shrunk) probability
DEFAULT_MAX_DAYS          = 40    # betting horizon: only bets resolving within this many days
MAX_BET_FRAC              = 0.10  # hard cap on recommended single-bet size (% of bankroll)
MC_RUNS                   = 100_000  # simulation count behind the model probabilities (matches phase9)

# Backward-compat alias (older imports referenced DEFAULT_KELLY_FRACTION)
DEFAULT_KELLY_FRACTION = DEFAULT_KELLY_MULTIPLIER

# Map the human market label to a tournament-depth rank (deeper = higher).
_STAGE_RANK = {
    "Group Stage":           1,
    "Reach Round of 16":     2,
    "Reach Quarter-Finals":  3,
    "Reach Semi-Finals":     4,
    "Reach Final":           5,
    "Champion market":       6,
}


def _team_of(selection: str) -> str:
    """Extract the team name from a selection string like 'Morocco to reach the QF'."""
    return selection.split(" to ")[0].strip()


def _compute(
    df: pd.DataFrame,
    model_trust: float = DEFAULT_MODEL_TRUST,
    kelly_multiplier: float = DEFAULT_KELLY_MULTIPLIER,
    selection_haircut: float = DEFAULT_SELECTION_HAIRCUT,
    spread_buffer_cents: float = DEFAULT_SPREAD_BUFFER_C,
) -> pd.DataFrame:
    """Attach naive and risk-adjusted metrics (incl. recommended bet size) to every row."""
    required = {"match", "selection", "model_prob", "polymarket_cents"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {missing}")

    df = df.copy()

    df["team"]        = df["selection"].map(_team_of)
    df["stage_rank"]  = df["match"].map(_STAGE_RANK).fillna(0).astype(int)
    # Days until this bet resolves. Match bets carry an explicit `resolve_date` (the match
    # date); futures derive it from the stage label. None/unknown → large sentinel.
    if "resolve_date" in df.columns:
        today = pd.Timestamp.today().normalize()
        dd = (pd.to_datetime(df["resolve_date"], errors="coerce") - today).dt.days
        df["days_to_resolve"] = dd.fillna(9999).clip(lower=0).astype(int)
    else:
        df["days_to_resolve"] = df["match"].map(lambda m: days_until(m)).map(
            lambda d: 9999 if pd.isna(d) else int(d)
        )

    df["implied_prob"]      = df["polymarket_cents"] / 100
    df["payout_multiplier"] = 100 / df["polymarket_cents"]

    # Executable price (buy at the ask) and the payout you actually get.
    exec_cents          = np.minimum(df["polymarket_cents"] + spread_buffer_cents, 99.0)
    df["exec_price"]    = exec_cents / 100
    df["payout_exec"]   = 1.0 / df["exec_price"]

    # Naive view — model vs mid-price.
    df["EV"]            = (df["model_prob"] / df["implied_prob"]) - 1            # ev_raw (legacy name)
    net_odds_raw        = df["payout_multiplier"] - 1
    df["kelly"]         = (net_odds_raw * df["model_prob"] - (1 - df["model_prob"])) / net_odds_raw

    # Risk-adjusted view — shrink the model toward the market, then haircut the edge,
    # then evaluate against the executable price.
    blend               = model_trust * (1 - selection_haircut)   # combined shrink factor
    df["model_shrunk"]  = df["implied_prob"] + blend * (df["model_prob"] - df["implied_prob"])
    df["model_shrunk"]  = df["model_shrunk"].clip(0.0, 1.0)

    df["ev_adj"]        = (df["model_shrunk"] / df["exec_price"]) - 1
    net_odds_exec       = df["payout_exec"] - 1
    # Full Kelly on the conservative (shrunk prob, executable payout) — can be negative.
    kelly_full          = (net_odds_exec * df["model_shrunk"] - (1 - df["model_shrunk"])) / net_odds_exec
    df["kelly_frac"]    = kelly_full * kelly_multiplier                          # ranking signal (unclipped)
    # Recommended single-bet size as % of bankroll: fractional Kelly, floored at 0, capped.
    df["recommended_bet_size"] = df["kelly_frac"].clip(lower=0.0, upper=MAX_BET_FRAC)

    # Monte Carlo sampling error on the marginal probability (simulation noise only).
    df["mc_se"]         = np.sqrt(df["model_prob"] * (1 - df["model_prob"]) / MC_RUNS)

    # Rounding for display stability
    for col in ["implied_prob", "exec_price", "model_shrunk"]:
        df[col] = df[col].round(4)
    for col in ["EV", "ev_adj", "kelly", "kelly_frac", "recommended_bet_size", "mc_se"]:
        df[col] = df[col].round(4)
    df["payout_multiplier"] = df["payout_multiplier"].round(2)
    df["payout_exec"]       = df["payout_exec"].round(2)

    return df


def _qualify(
    df: pd.DataFrame,
    min_ev: float = MIN_EV,
    min_win_prob: float = DEFAULT_MIN_WIN_PROB,
    max_days: int = DEFAULT_MAX_DAYS,
    max_cents: float = MAX_CENTS,
) -> pd.DataFrame:
    """
    Keep contracts that survive the filter, ranked by hit rate.

    A contract must clear:
      - the price cap (< MAX_CENTS, payout > 2.5x),
      - positive adjusted EV (net of shrinkage / haircut / ask buffer),
      - the HIT-RATE FLOOR model_shrunk >= min_win_prob (flooring on the USED probability),
      - the BETTING HORIZON days_to_resolve <= max_days (only bets that settle within the
        window you care about — a finite bankroll shouldn't be locked for weeks in a far-off
        future when nearer, faster-resolving bets are available).
    Ranked by hit rate (used probability) descending, fractional Kelly as the tiebreak.
    """
    return (
        df[
            (df["polymarket_cents"] < max_cents) &
            (df["ev_adj"] > min_ev) &
            (df["kelly_frac"] > 0) &
            (df["model_shrunk"] >= min_win_prob) &
            (df["days_to_resolve"] <= max_days)
        ]
        .copy()
        .sort_values(["model_shrunk", "kelly_frac"], ascending=[False, False])
        .reset_index(drop=True)
    )


def process_opportunities(
    df: pd.DataFrame,
    top_n: int = TOP_N,
    min_win_prob: float = DEFAULT_MIN_WIN_PROB,
    max_days: int = DEFAULT_MAX_DAYS,
    max_cents: float = MAX_CENTS,
    **adj,
) -> pd.DataFrame:
    """
    Compute metrics and return the top N risk-adjusted picks.

    `min_win_prob` is the hit-rate floor; `max_days` is the betting horizon; `max_cents` is the
    price cap (40 for high-yield futures; raise it for match bets, where favourites are the
    likely-to-resolve survive-to-learn pick). `**adj` forwards the sizing knobs.
    """
    return _qualify(_compute(df, **adj), min_win_prob=min_win_prob,
                    max_days=max_days, max_cents=max_cents).head(top_n)


def get_all_qualifying(
    df: pd.DataFrame,
    min_win_prob: float = DEFAULT_MIN_WIN_PROB,
    max_days: int = DEFAULT_MAX_DAYS,
    max_cents: float = MAX_CENTS,
    **adj,
) -> pd.DataFrame:
    """Return every contract that passes the filter (floor + horizon + price cap), uncapped in N."""
    return _qualify(_compute(df, **adj), min_win_prob=min_win_prob,
                    max_days=max_days, max_cents=max_cents)


def compute_all(df: pd.DataFrame, **adj) -> pd.DataFrame:
    """All rows with metrics attached, NO filtering — for a full scan view (e.g. every match
    outcome, including overpriced favourites the strategy would skip)."""
    if df.empty:
        return df
    return _compute(df, **adj).sort_values("ev_adj", ascending=False).reset_index(drop=True)
