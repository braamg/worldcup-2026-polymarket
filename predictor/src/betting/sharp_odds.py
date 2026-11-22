"""
sharp_odds.py — pull WC match odds from a SHARP book and turn them into the same
canonical match table our model + Polymarket fetcher use, so all three join cleanly.

Why: you cannot out-predict Polymarket's liquid markets with our model, but the SHARPEST
soccer prices on earth (Pinnacle, Betfair Exchange) are a better "truth" than Polymarket's
slower, crypto-retail flow. Treat the sharp price as the fair value and bet Polymarket
whenever it diverges. This module produces sharp-implied, de-vigged P(home/draw/away)
keyed on canonical match keys; scripts/betting_cross.py joins it to Polymarket and sizes.

Source: The Odds API (the-odds-api.com) — free tier (~500 req/mo). Set ODDS_API_KEY in the
environment, or pass api_key=. It aggregates many books; we prefer Pinnacle, else the
median de-vigged consensus across books (itself a sharp estimate).

Verify the live sport key with list_soccer_sports() — it was "soccer_fifa_world_cup" in 2022.
This module makes NO bets and needs no auth beyond the read-only odds key.
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import requests

from src.betting.match_markets import canonical_match_key, match_market_ids
from src.betting.polymarket import _normalize_team, _ALL_WC
from src.utils import get_logger

log = get_logger(__name__)

ODDS_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_SPORT = "soccer_fifa_world_cup"
SHARP_BOOK = "pinnacle"


def _api_key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise RuntimeError(
            "No odds API key. Get a free key at https://the-odds-api.com and set "
            "ODDS_API_KEY in your environment (or pass api_key=)."
        )
    return key


def list_soccer_sports(api_key: Optional[str] = None) -> pd.DataFrame:
    """List active soccer competitions + their sport keys (to find the live WC key)."""
    r = requests.get(f"{ODDS_BASE}/sports", params={"apiKey": _api_key(api_key)}, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    return df[df["group"].str.contains("Soccer", case=False, na=False)][["key", "title", "active"]]


def devig_3way(odd_h: float, odd_d: float, odd_a: float) -> tuple[float, float, float]:
    """Decimal odds → de-vigged (normalised) P(home), P(draw), P(away)."""
    raw = [1.0 / odd_h, 1.0 / odd_d, 1.0 / odd_a]
    s = sum(raw)
    return tuple(x / s for x in raw)   # type: ignore[return-value]


def fetch_sharp_match_odds(
    sport_key: str = DEFAULT_SPORT,
    regions: str = "eu",
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> list[dict]:
    """Raw events with per-book h2h decimal odds. Returns [] on a clean miss."""
    r = requests.get(
        f"{ODDS_BASE}/sports/{sport_key}/odds",
        params={"apiKey": _api_key(api_key), "regions": regions,
                "markets": "h2h", "oddsFormat": "decimal"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def _consensus_probs(
    event: dict, prefer: str = SHARP_BOOK,
) -> Optional[tuple[str, str, tuple[float, float, float], int, str, float]]:
    """
    From one event, return (home, away, (p_h,p_d,p_a), n_books, source, dispersion).
    Uses Pinnacle if present, else the median de-vigged probs across all books.

    dispersion: mean per-outcome stdev of de-vigged probs ACROSS all books that quoted this
    match (0.0 if only one book — can't measure disagreement from a single source). This is a
    free, real-time proxy for how much the sharp side currently disagrees with itself, i.e. how
    much to trust "the sharp price" as fair value right now — computed regardless of whether
    the point estimate above ends up being Pinnacle or the cross-book median.
    """
    home, away = event.get("home_team"), event.get("away_team")
    if not home or not away:
        return None
    per_book = []
    pinnacle = None
    for bk in event.get("bookmakers", []):
        h2h = next((m for m in bk.get("markets", []) if m.get("key") == "h2h"), None)
        if not h2h:
            continue
        outc = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}
        if home in outc and away in outc and "Draw" in outc:
            probs = devig_3way(outc[home], outc["Draw"], outc[away])
            per_book.append(probs)
            if bk.get("key") == prefer:
                pinnacle = probs
    if not per_book:
        return None
    book_df = pd.DataFrame(per_book, columns=["h", "d", "a"])
    dispersion = float(book_df.std(ddof=0).mean()) if len(book_df) > 1 else 0.0
    if pinnacle is not None:
        return home, away, pinnacle, len(per_book), prefer, dispersion
    arr = book_df.median()
    s = arr.sum()
    return home, away, (arr["h"] / s, arr["d"] / s, arr["a"] / s), len(per_book), "consensus", dispersion


def extract_schedule(events: list[dict]) -> list[dict]:
    """Kickoff schedule from already-fetched odds events (no extra API call)."""
    from src.betting.match_markets import canonical_match_key
    out = []
    for ev in events:
        h, a = ev.get("home_team"), ev.get("away_team")
        ct = ev.get("commence_time")
        if not (h and a and ct):
            continue
        home, away = _normalize_team(h), _normalize_team(a)
        if home not in _ALL_WC or away not in _ALL_WC:
            continue
        _, _, grp = canonical_match_key(home, away)
        out.append({"match": grp, "home": home, "away": away, "commence_time": ct})
    return out


def sharp_match_table(
    sport_key: str = DEFAULT_SPORT, regions: str = "eu", api_key: Optional[str] = None,
    events: Optional[list[dict]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Sharp-implied 3-way table keyed on canonical match keys (joins to Polymarket + model).
    Pass pre-fetched `events` to avoid a second API call. Returns (df, unmapped_team_names).
    """
    if events is None:
        events = fetch_sharp_match_odds(sport_key, regions, api_key)
    rows, unmapped = [], set()
    for ev in events:
        cp = _consensus_probs(ev)
        if cp is None:
            continue
        home_raw, away_raw, (ph, pdr, pa), nbk, src, dispersion = cp
        home, away = _normalize_team(home_raw), _normalize_team(away_raw)
        if home not in _ALL_WC:
            unmapped.add(home_raw); continue
        if away not in _ALL_WC:
            unmapped.add(away_raw); continue
        t1, t2, grp = canonical_match_key(home, away)
        ids = match_market_ids(home, away)
        prob = {home: ph, "draw": pdr, away: pa}
        for outcome in (t1, "draw", t2):
            rows.append({
                "market_id":   ids[outcome],
                "sharp_prob":  round(float(prob[outcome]), 4),
                "n_books":     nbk, "source": src,
                "dispersion":  round(float(dispersion), 4),
                "description": (f"{t1} vs {t2} — draw" if outcome == "draw"
                                else f"{outcome} to win"),
            })
    df = pd.DataFrame(rows).drop_duplicates("market_id").reset_index(drop=True) if rows else pd.DataFrame()
    return df, sorted(unmapped)
