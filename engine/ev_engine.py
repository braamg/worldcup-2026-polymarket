"""
ev_engine.py
============
Expected Value engine for the V2 High-Yield strategy.

Core formula
------------
    Implied Probability = Market Price  (Polymarket binary: price IS the probability)
    EV = (Model Probability / Implied Probability) - 1

A positive EV means our model thinks the contract is underpriced.
E.g. EV = 0.40 means the model implies a 40% higher true probability than the market price.

Strategy filter
---------------
Only consider contracts priced UNDER 40 cents (< 0.40).
These longshot contracts have the highest potential payout multiplier when mispriced,
which is where asymmetric edge lives on a small bankroll.

    Payout Multiplier = 1 / Market Price
    (if you buy at 20c and it resolves YES, you receive $1 — a 5x payout)
"""

from __future__ import annotations

import pandas as pd

# Strategy parameters
MAX_PRICE       = 0.40   # only consider contracts cheaper than this
MIN_EV          = 0.10   # minimum EV (10%) to appear in the picks list
TOP_N           = 5      # number of top picks to surface


def compute_ev(model_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join model probabilities with live Polymarket prices and compute EV.

    Parameters
    ----------
    model_df  : from model_loader.load_model_probs()
    market_df : from polymarket_client.fetch_futures_prices()

    Returns
    -------
    DataFrame with EV and payout columns, all markets included.
    """
    if model_df.empty or market_df.empty:
        return pd.DataFrame()

    merged = model_df.merge(
        market_df[["market_id", "price", "implied_prob"]],
        on="market_id",
        how="inner",
    )

    if merged.empty:
        return merged

    merged["ev"]              = (merged["model_prob"] / merged["implied_prob"]) - 1
    merged["payout_mult"]     = (1.0 / merged["price"]).round(1)
    merged["ev"]              = merged["ev"].round(4)
    merged["price_cents"]     = (merged["price"] * 100).round(1)
    merged["model_prob_pct"]  = (merged["model_prob"] * 100).round(1)
    merged["implied_prob_pct"]= (merged["implied_prob"] * 100).round(1)

    return merged.sort_values("ev", ascending=False).reset_index(drop=True)


def get_top_picks(ev_df: pd.DataFrame, max_price: float = MAX_PRICE,
                  min_ev: float = MIN_EV, top_n: int = TOP_N) -> pd.DataFrame:
    """
    Filter to high-yield candidates and return the top N by EV.

    Applies:
      1. Price cap: only contracts < max_price (default 40c)
      2. Minimum EV floor: only contracts with EV >= min_ev
      3. Positive model probability: no zero-prob picks

    Returns a display-ready DataFrame.
    """
    if ev_df.empty:
        return pd.DataFrame()

    filtered = ev_df[
        (ev_df["price"]      <  max_price) &
        (ev_df["ev"]         >= min_ev)    &
        (ev_df["model_prob"] >  0)
    ].copy()

    top = filtered.head(top_n)

    if top.empty:
        return top

    return top[[
        "description", "market_type", "team",
        "model_prob_pct", "price_cents", "implied_prob_pct",
        "payout_mult", "ev", "market_id",
    ]].rename(columns={
        "description":      "Bet",
        "market_type":      "Type",
        "team":             "Team",
        "model_prob_pct":   "Model prob %",
        "price_cents":      "Market price (¢)",
        "implied_prob_pct": "Implied prob %",
        "payout_mult":      "Payout (x)",
        "ev":               "EV",
        "market_id":        "market_id",
    }).reset_index(drop=True)


def portfolio_summary(selected_rows: pd.DataFrame,
                      bankroll_zar: float = 400.0,
                      zar_usd: float = 18.5) -> dict:
    """
    Given a selection of picks, compute portfolio allocation.

    Splits bankroll equally across selected bets (simple flat sizing —
    appropriate for a small bankroll where Kelly-optimal stakes are
    below Polymarket's practical minimum anyway).

    Returns a dict of summary statistics.
    """
    if selected_rows.empty:
        return {}

    bankroll_usd = bankroll_zar / zar_usd
    n            = len(selected_rows)
    stake_usd    = bankroll_usd / n
    stake_zar    = bankroll_zar / n

    rows = []
    for _, r in selected_rows.iterrows():
        price      = float(r.get("Market price (¢)", 0)) / 100
        payout_x   = float(r.get("Payout (x)", 0))
        profit_usd = stake_usd * (payout_x - 1) if price > 0 else 0
        rows.append({
            "Bet":          r.get("Bet", ""),
            "Stake (R)":    round(stake_zar, 2),
            "Stake ($)":    round(stake_usd, 2),
            "Price (¢)":    r.get("Market price (¢)", 0),
            "Payout (x)":   payout_x,
            "If wins ($)":  round(profit_usd, 2),
            "If wins (R)":  round(profit_usd * zar_usd, 2),
            "EV":           r.get("EV", 0),
        })

    total_if_all_win = sum(r["If wins (R)"] for r in rows)

    return {
        "rows":              rows,
        "n_bets":            n,
        "stake_per_bet_zar": round(stake_zar, 2),
        "stake_per_bet_usd": round(stake_usd, 2),
        "total_deployed_zar":round(bankroll_zar, 2),
        "total_if_all_win":  round(total_if_all_win, 2),
    }
