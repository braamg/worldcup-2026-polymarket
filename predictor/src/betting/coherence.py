"""
coherence.py — structural (near-riskless) arbitrage scanner for Polymarket WC markets.

The thesis (see project notes): you cannot out-predict Polymarket's liquid headline
markets, but Polymarket prices its many CORRELATED World Cup markets in separate,
independently-traded order books with nothing forcing them to cohere. Where they
violate the laws of probability, there is near-riskless money — and our job is only
to detect it, not to forecast anything.

Two structural laws we exploit:

1. MONOTONICITY across the progression markets (the big one). For one team,
   winning ⟹ reaching the final ⟹ reaching the SF ⟹ … ⟹ advancing. So
       P(champion) ≤ P(reach_final) ≤ P(reach_sf) ≤ P(reach_qf) ≤ P(reach_r16) ≤ P(advance).
   These six are SEPARATE markets (the reach-round ones are negRisk=False independent
   binaries), so the inequality is NOT auto-enforced. If a more-specific event trades
   at a higher price than a more-general one, that is an arbitrage.

   Executable, fee-aware condition for nested S ⊆ G priced (bid,ask):
     short S (buy S-NO at 1 - S_bid) + long G (buy G-YES at G_ask)
     cost = G_ask + (1 - S_bid);  min payout at resolution = 1 (proof in project notes)
     gross profit / unit = S_bid - G_ask;  net = gross - taker_fee(S) - taker_fee(G).
   Riskless when net > 0. Capital locks until the later-resolving leg settles.

2. SUM coherence within a negRisk multi-outcome market (champion: exactly one winner;
   each match: exactly one of {A, draw, B}). Σ(YES) should ≈ 1. negRisk convert keeps
   the champion market tight, but thin match markets can drift. Σ(YES) < 1 ⇒ buy every
   YES for < $1, collect $1 (riskless); Σ(YES) > 1 ⇒ buy every NO.

Fees (Polymarket sports_fees_v2, verified live 2026-06-14): taker only,
fee = 0.03 * p * (1-p) of trade value (max 0.75% at p=0.5); makers pay 0.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import requests

from src.betting.polymarket import (
    GAMMA, _HEADERS, _normalize_team, _fetch_event, EVENT_MARKET_TYPE, _ALL_WC,
)
from src.utils import get_logger

log = get_logger(__name__)

# Progression chain, MOST specific → MOST general. price[i] must be ≤ price[j] for i<j.
CHAIN = ["champion", "reach_final", "reach_sf", "reach_qf", "reach_r16", "advance"]
FEE_RATE = 0.03   # sports_fees_v2 taker rate; fee fraction of stake = FEE_RATE * p * (1-p)


def taker_fee(p: float) -> float:
    """Taker fee as a fraction of trade value at price p (0 for maker)."""
    return FEE_RATE * p * (1.0 - p)


# Minimum $ size required on EVERY leg of a 3-way sum arb before it's flagged "executable".
# Checked against on-data: in live snapshots, draw-leg depth is NOT reliably the thin one —
# any of the three legs (often a deep-longshot favourite/underdog) can be the binding
# constraint. Without this, the scanner can flag a "buy all 3 YES" arb where one leg only
# has a few dollars of size at the touch — not a real, fillable arb.
MIN_LEG_LIQUIDITY = 50.0


def leg_ask_sizes(depth: pd.DataFrame, token_ids) -> list[float]:
    """For each token_id (NaN-safe), the best_ask_size from a depth table (src/betting/
    polymarket.fetch_depth_table output); 0.0 if missing/no book/NaN. Order matches input."""
    if depth is None or depth.empty or 'best_ask_size' not in depth.columns:
        return [0.0 for _ in token_ids]
    idx = depth.drop_duplicates('token_id').set_index('token_id')['best_ask_size']
    out = []
    for t in token_ids:
        if t is None or (isinstance(t, float) and pd.isna(t)):
            out.append(0.0); continue
        v = idx.get(str(t))
        out.append(float(v) if v is not None and pd.notna(v) else 0.0)
    return out


def _bid_ask(m: dict) -> Optional[tuple[float, float]]:
    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    bid, ask = _f(m.get("bestBid")), _f(m.get("bestAsk"))
    if bid is not None and ask is not None and 0.0 < bid <= ask < 1.0:
        return bid, ask
    return None


def fetch_futures_board(timeout: float = 60.0) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Return {team: {market_type: (yes_bid, yes_ask)}} across the 6 progression markets.
    Only two-sided quotes are kept.
    """
    session = requests.Session(); session.headers.update(_HEADERS)
    board: dict[str, dict[str, tuple[float, float]]] = {}
    for slug, mtype in EVENT_MARKET_TYPE.items():
        if mtype not in CHAIN:
            continue
        try:
            ev = _fetch_event(slug, session, timeout)
        except requests.RequestException as exc:
            log.warning("fetch %s failed: %s", slug, exc); continue
        if ev is None:
            continue
        for m in ev.get("markets", []):
            ba = _bid_ask(m)
            if ba is None:
                continue
            team = _normalize_team(str(m.get("groupItemTitle", "")).strip())
            if team not in _ALL_WC:
                continue
            board.setdefault(team, {})[mtype] = ba
    return board


def scan_monotonicity_arbs(board: dict, min_net: float = 0.0) -> pd.DataFrame:
    """
    Flag riskless monotonicity violations: a more-specific event whose BID exceeds a
    more-general event's ASK (net of taker fees). Returns rows sorted by net edge.
    """
    rows = []
    for team, mk in board.items():
        for i in range(len(CHAIN)):
            for j in range(i + 1, len(CHAIN)):
                s, g = CHAIN[i], CHAIN[j]          # s more specific than g
                if s not in mk or g not in mk:
                    continue
                s_bid, s_ask = mk[s]
                g_bid, g_ask = mk[g]
                gross = s_bid - g_ask              # S_bid > G_ask ⇒ inversion
                if gross <= 0:
                    continue
                fee = taker_fee(s_bid) + taker_fee(g_ask)
                net = gross - fee
                if net > min_net:
                    rows.append({
                        "team": team, "specific": s, "general": g,
                        "specific_price": round(s_bid, 4), "general_price": round(g_ask, 4),
                        "gross_edge": round(gross, 4), "fee": round(fee, 4),
                        "net_edge": round(net, 4),
                        "action": f"SHORT {s} @ {s_bid:.3f}  +  LONG {g} @ {g_ask:.3f}",
                    })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("net_edge", ascending=False).reset_index(drop=True)
    return df


def scan_chain_consistency(board: dict) -> pd.DataFrame:
    """Per-team summary: are the six progression MIDs weakly decreasing as required?"""
    rows = []
    for team, mk in board.items():
        mids = {t: round((mk[t][0] + mk[t][1]) / 2, 4) for t in CHAIN if t in mk}
        present = [t for t in CHAIN if t in mids]
        violations = sum(1 for a, b in zip(present, present[1:]) if mids[a] > mids[b] + 1e-9)
        rows.append({"team": team, "n_markets": len(present), "violations": violations,
                     **{t: mids.get(t) for t in CHAIN}})
    return pd.DataFrame(rows).sort_values("violations", ascending=False).reset_index(drop=True)
