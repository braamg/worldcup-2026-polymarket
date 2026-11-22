"""
strategy_config.py
==================
Hardcoded strategy parameters that are NOT exposed on the dashboard.

Edit the values here to change them; the dashboard only surfaces the three you tune often
(Bankroll, Minimum Win Probability, Betting Horizon). Everything below has a sensible default
and rarely needs touching — but it's here in one place when it does.
"""

# ── Edge adjustments — pull the model's raw EV back toward reality ──────────────────────────
MODEL_TRUST          = 0.35   # weight on model vs market in the shrinkage blend (0=market, 1=model)
SELECTION_HAIRCUT    = 0.25   # shrink apparent edge to counter the winner's curse
SPREAD_BUFFER_CENTS  = 2.0    # cents added to price to approximate buying at the ask
KELLY_MULTIPLIER     = 0.25   # fraction of full Kelly behind recommended_bet_size

# ── Single-match discipline — the model over-rates draws/longshots ──────────────────────────
MATCH_CONTENDER_FLOOR = 0.45  # only back win outcomes the model rates at least this likely
EXCLUDE_DRAWS         = False # draws re-enabled: market systematically opens 7pp below actual rate
DRAW_MIN_MODEL_PROB   = 0.28  # draw must be at or above the observed base rate (28.6% across 28 WC26 matches)
DRAW_MIN_EDGE_PP      = 5.0   # model must exceed market open by at least this many percentage points
EXCLUDE_INPLAY        = True  # drop matches resolving today (model is pre-match, price may be live)
MATCH_MAX_CENTS       = 95.0  # price cap for match bets (futures stay at the 40¢ high-yield cap)

# ── Futures price cap (the original high-yield rule) ────────────────────────────────────────
FUTURES_MAX_CENTS     = 40.0
