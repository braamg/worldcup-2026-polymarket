"""
polymarket_client.py
====================
Minimal Polymarket price fetcher for the engine.

Fetches WC2026 futures prices from the public Gamma API (no auth required).
Returns a clean DataFrame of (market_id, team, market_type, price, implied_prob).
"""

from __future__ import annotations

import json
import re
from typing import Optional

import pandas as pd
import requests

GAMMA = "https://gamma-api.polymarket.com"
WC_TAG_ID = 102232

_HEADERS = {"User-Agent": "wc2026-v2/1.0 (research; read-only)"}
_DATE_END = re.compile(r"\d{4}-\d{2}-\d{2}$")

# Polymarket event slug -> market type label
EVENT_SLUGS: dict[str, str] = {
    "world-cup-winner":                             "champion",
    "world-cup-nation-to-reach-final":              "reach_final",
    "world-cup-nation-to-reach-semifinals":         "reach_sf",
    "world-cup-nation-to-reach-quarterfinals":      "reach_qf",
    "world-cup-nation-to-reach-round-of-16":        "reach_r16",
    "world-cup-team-to-advance-to-knockout-stages": "advance",
}

# Normalise common Polymarket name variants to our canonical names
_ALIASES: dict[str, str] = {
    "Turkiye": "Turkey", "Türkiye": "Turkey",
    "Curacao": "Curaçao",
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea", "Republic of Korea": "South Korea",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "DR Congo": "DR Congo",
}


def _norm(name: str) -> str:
    return _ALIASES.get(name.strip(), name.strip())


def _mid(m: dict) -> Optional[float]:
    """Best-effort Yes mid-price from a Polymarket market dict."""
    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    bid, ask = _f(m.get("bestBid")), _f(m.get("bestAsk"))
    if bid and ask and 0 < bid <= ask < 1:
        return (bid + ask) / 2.0

    op = m.get("outcomePrices")
    if op:
        prices = json.loads(op) if isinstance(op, str) else op
        yes = _f(prices[0]) if prices else None
        if yes and 0 < yes < 1:
            return yes
    return None


def _fetch_event(slug: str, session: requests.Session) -> Optional[dict]:
    r = session.get(f"{GAMMA}/events", params={"slug": slug}, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data[0] if data else None


def fetch_futures_prices() -> pd.DataFrame:
    """
    Fetch live Polymarket WC2026 futures prices.

    Returns
    -------
    DataFrame with columns:
        market_id     : str   e.g. "champion:Brazil"
        team          : str   canonical team name
        market_type   : str   e.g. "champion", "reach_final"
        price         : float Yes mid-price in (0, 1)
        implied_prob  : float same as price (Polymarket = binary, price IS the probability)
    """
    session = requests.Session()
    session.headers.update(_HEADERS)
    rows = []

    for slug, mtype in EVENT_SLUGS.items():
        try:
            ev = _fetch_event(slug, session)
        except Exception:
            continue
        if not ev:
            continue
        for m in ev.get("markets", []):
            if not m.get("acceptingOrders", True):
                continue
            price = _mid(m)
            if price is None:
                continue
            team = _norm(str(m.get("groupItemTitle", "")))
            if not team:
                continue
            rows.append({
                "market_id":    f"{mtype}:{team}",
                "team":         team,
                "market_type":  mtype,
                "price":        round(price, 4),
                "implied_prob": round(price, 4),   # price IS implied prob on Polymarket
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates("market_id").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Single-MATCH (3-way moneyline) prices
# ---------------------------------------------------------------------------

def _canon_match(a: str, b: str) -> tuple[str, str, str]:
    """Canonical match key: teams sorted alphabetically. Matches the model side."""
    t1, t2 = sorted([a, b])
    return t1, t2, f"match:{t1}|{t2}"


def _fetch_tag_events(session: requests.Session, max_pages: int = 6, timeout: float = 60.0) -> list:
    """All open events under the WC Gamma tag (paginated, 100/page)."""
    events = []
    for off in range(0, max_pages * 100, 100):
        try:
            page = session.get(
                f"{GAMMA}/events",
                params={"closed": "false", "tag_id": WC_TAG_ID, "limit": 100, "offset": off},
                timeout=timeout,
            ).json()
        except Exception:
            break
        if not page:
            break
        events.extend(page)
        if len(page) < 100:
            break
    return events


def fetch_match_prices() -> pd.DataFrame:
    """
    Fetch live Polymarket 3-way moneyline prices for upcoming WC matches.

    Each base match event (slug 'fifwc-...-YYYY-MM-DD', exactly 3 markets: two teams +
    Draw) becomes three rows keyed on the canonical match key, matching the model side
    (match_probabilities.json) so they join on market_id.

    Returns DataFrame[market_id, team, price, implied_prob, group_id] (empty on failure).
    """
    session = requests.Session()
    session.headers.update(_HEADERS)
    try:
        events = _fetch_tag_events(session)
    except Exception:
        return pd.DataFrame()

    rows = []
    for ev in events:
        slug = str(ev.get("slug", ""))
        mkts = ev.get("markets", []) or []
        if not slug.startswith("fifwc-") or not _DATE_END.search(slug) or len(mkts) != 3:
            continue

        team_price: dict[str, float] = {}
        draw_price: Optional[float] = None
        for m in mkts:
            git = str(m.get("groupItemTitle", "")).strip()
            price = _mid(m)
            if price is None:
                continue
            if git.lower().startswith("draw"):
                draw_price = price
            else:
                team_price[_norm(git)] = price

        if len(team_price) != 2 or draw_price is None:
            continue
        (ta, pa), (tb, pb) = list(team_price.items())
        t1, t2, grp = _canon_match(ta, tb)
        per = {ta: pa, tb: pb}
        for outcome, price in ((t1, per[t1]), ("draw", draw_price), (t2, per[t2])):
            rows.append({
                "market_id":    f"{grp}:{outcome}",
                "team":         outcome,
                "price":        round(price, 4),
                "implied_prob": round(price, 4),
                "group_id":     grp,
                "event_slug":   slug,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates("market_id").reset_index(drop=True)
    return df
