"""
utils/forecast_log.py
=====================
Forward-only forecast logging for the scientific verdict.

The verdict needs many (model_prob, market_price, outcome) triples to test whether the model
beats the market. Logged *bets* are too few and selection-biased. Instead we snapshot the
model's probability AND the market price for EVERY upcoming market, once per day. When a match
resolves, each snapshot becomes a scored data point — no capital at risk, no cherry-picking.

This is forward-only by design: we log from today onward. We deliberately do NOT reconstruct
forecasts for already-played matches, because the current model is conditioned on those results
(look-ahead bias). A clean forecast must predate the outcome.

The log is append-only, deduplicated to one row per (market_id, snapshot_date). The FIRST
snapshot of a market is its forecast/entry; the LAST before kickoff is the market's close (CLV).
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

FORECAST_LOG_PATH = Path(__file__).resolve().parent.parent / "forecast_log.csv"

_COLS = ["snapshot_date", "market_id", "selection", "bet_type",
         "model_prob", "market_price", "match_date"]


def load_forecast_log() -> pd.DataFrame:
    if not FORECAST_LOG_PATH.exists():
        return pd.DataFrame(columns=_COLS)
    try:
        return pd.read_csv(FORECAST_LOG_PATH)
    except Exception:
        return pd.DataFrame(columns=_COLS)


def snapshot_forecasts(
    match_model_df: pd.DataFrame | None,
    match_market_df: pd.DataFrame | None,
    today_str: str,
    futures_model_df: pd.DataFrame | None = None,
    futures_market_df: pd.DataFrame | None = None,
    stage_resolve_dates: dict | None = None,
) -> int:
    """
    Append today's model+market snapshot for every joinable market. Deduplicated to one row per
    (market_id, snapshot_date), so calling it repeatedly in a day is a no-op. Returns rows added.
    """
    rows = []

    # ── Match markets (3-way moneyline) ──────────────────────────────────────
    if (match_model_df is not None and not match_model_df.empty
            and match_market_df is not None and not match_market_df.empty):
        m = match_model_df.merge(
            match_market_df[["market_id", "price"]], on="market_id", how="inner")
        for _, r in m.iterrows():
            rows.append({
                "snapshot_date": today_str,
                "market_id":     r["market_id"],
                "selection":     r.get("selection", ""),
                "bet_type":      "match",
                "model_prob":    round(float(r["model_prob"]), 4),
                "market_price":  round(float(r["price"]), 4),
                "match_date":    r.get("match_date", ""),
            })

    # ── Futures markets (advance / reach stage / champion) ───────────────────
    if (futures_model_df is not None and not futures_model_df.empty
            and futures_market_df is not None and not futures_market_df.empty):
        f = futures_model_df.merge(
            futures_market_df[["market_id", "price"]], on="market_id", how="inner")
        sr = stage_resolve_dates or {}
        for _, r in f.iterrows():
            rows.append({
                "snapshot_date": today_str,
                "market_id":     r["market_id"],
                "selection":     r.get("description", r["market_id"]),
                "bet_type":      "future",
                "model_prob":    round(float(r["model_prob"]), 4),
                "market_price":  round(float(r["price"]), 4),
                "match_date":    sr.get(r.get("market_type", ""), ""),
            })

    if not rows:
        return 0

    new = pd.DataFrame(rows)
    existing = load_forecast_log()
    combined = pd.concat([existing, new], ignore_index=True)
    # one row per (market_id, snapshot_date) — keep the first seen that day
    combined = combined.drop_duplicates(subset=["market_id", "snapshot_date"], keep="first")
    combined.to_csv(FORECAST_LOG_PATH, index=False)
    return len(combined) - len(existing)
