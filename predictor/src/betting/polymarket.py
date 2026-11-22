"""
polymarket.py — live Polymarket price ingestion for WC2026 futures.

Read-only. Uses the PUBLIC Gamma API (gamma-api.polymarket.com, no auth) to fetch
current prices for the World Cup futures markets and writes the price CSV that
scripts/betting_slate.py consumes. It places NO orders — order placement requires
authenticated CLOB access and stays manual by design.

Mapping (Polymarket "Tournament Futures" event slug → our model market type):
    world-cup-winner                              -> champion
    world-cup-nation-to-reach-final               -> reach_final
    world-cup-nation-to-reach-semifinals          -> reach_sf
    world-cup-nation-to-reach-quarterfinals       -> reach_qf
    world-cup-nation-to-reach-round-of-16         -> reach_r16
    world-cup-team-to-advance-to-knockout-stages  -> advance

Each futures event holds one binary (Yes/No) market per team; groupItemTitle is the
team and outcomePrices[0] is the Yes price. market_id is "<type>:<canonical team>",
matching src/betting/model_markets.build_model_market_table so the two join cleanly.

API shapes were verified live 2026-06-14; if Polymarket changes them, this module
reports the failure rather than guessing (per project policy).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.data.constants import TEAM_NAME_ALIASES
from src.simulation.tournament import ALL_WC_TEAMS
from src.utils import get_logger

log = get_logger(__name__)

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"   # public order-book (depth); read-only, no auth

# Polymarket futures event slug -> our model market type
EVENT_MARKET_TYPE: dict[str, str] = {
    "world-cup-winner":                             "champion",
    "world-cup-nation-to-reach-final":              "reach_final",
    "world-cup-nation-to-reach-semifinals":         "reach_sf",
    "world-cup-nation-to-reach-quarterfinals":      "reach_qf",
    "world-cup-nation-to-reach-round-of-16":        "reach_r16",
    "world-cup-team-to-advance-to-knockout-stages": "advance",
}

# Polymarket spellings that differ from our canonical names (beyond TEAM_NAME_ALIASES).
# Futures markets use common names; MATCH markets use FIFA-long names — cover both.
PM_TEAM_ALIASES: dict[str, str] = {
    "Turkiye":         "Turkey",
    "Türkiye":         "Turkey",
    "Curacao":         "Curaçao",
    "USA":             "United States",
    "Côte d'Ivoire":   "Ivory Coast",
    "Cote d'Ivoire":   "Ivory Coast",
    "Cabo Verde":      "Cape Verde",
    "IR Iran":         "Iran",
    "Korea Republic":  "South Korea",
    "Republic of Korea": "South Korea",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "DR Congo":        "DR Congo",
}

WC_TAG_ID = 102232   # "FIFA World Cup" Gamma tag

_HEADERS = {"User-Agent": "wc2026-predictor/1.0 (research; read-only)"}
_ALL_WC = set(ALL_WC_TEAMS)


def _normalize_team(pm_name: str) -> str:
    n = PM_TEAM_ALIASES.get(pm_name, pm_name)
    return TEAM_NAME_ALIASES.get(n, n)


def _mid_and_spread(m: dict) -> Optional[tuple[float, float]]:
    """Return (mid, spread) for a market's Yes outcome, or None if unpriced."""
    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    bid, ask = _f(m.get("bestBid")), _f(m.get("bestAsk"))
    if bid is not None and ask is not None and 0.0 < bid <= ask < 1.0:
        return (bid + ask) / 2.0, ask - bid

    op = m.get("outcomePrices")
    if op:
        prices = json.loads(op) if isinstance(op, str) else op
        yes = _f(prices[0]) if prices else None
        if yes is not None and 0.0 < yes < 1.0:
            sp = _f(m.get("spread"))
            return yes, (sp if sp is not None else 0.0)
    return None


def _token_ids(m: dict) -> tuple[Optional[str], Optional[str]]:
    """(yes_token_id, no_token_id) for a market — clobTokenIds[0]/[1] — or (None, None)."""
    tok = m.get("clobTokenIds")
    if isinstance(tok, str):
        try:
            tok = json.loads(tok)
        except (TypeError, ValueError):
            tok = None
    if tok and len(tok) >= 2:
        return str(tok[0]), str(tok[1])
    return None, None


def _yes_token(m: dict) -> Optional[str]:
    """The CLOB token id for a market's YES outcome (clobTokenIds[0]), or None."""
    return _token_ids(m)[0]


def fetch_clob_books(token_ids, session: Optional[requests.Session] = None,
                     timeout: float = 30.0, chunk: int = 100) -> dict:
    """Batch-fetch live order books for token ids via the public CLOB /books endpoint.

    Returns {token_id: book_dict}, where book_dict has 'bids'/'asks' = [{price, size}, ...]
    (NOT sorted best-first), plus tick_size/neg_risk/last_trade_price. Polite: one POST per
    `chunk` tokens. Read-only; no auth.
    """
    ids = [str(t) for t in token_ids if t]
    if not ids:
        return {}
    own = session is None
    if own:
        session = requests.Session()
        session.headers.update(_HEADERS)
    out: dict[str, dict] = {}
    for i in range(0, len(ids), chunk):
        batch = ids[i:i + chunk]
        try:
            r = session.post(f"{CLOB}/books", json=[{"token_id": t} for t in batch], timeout=timeout)
            r.raise_for_status()
            for bk in r.json():
                aid = str(bk.get("asset_id") or "")
                if aid:
                    out[aid] = bk
        except requests.RequestException as exc:
            log.warning("CLOB /books batch failed (%s)", exc)
    return out


def _depth_summary(book: dict, within: float = 0.02) -> dict:
    """Top-of-book + near-touch liquidity from one CLOB book. Best computed by min/max price."""
    def _levels(side):
        out = []
        for o in book.get(side, []) or []:
            try:
                out.append((float(o["price"]), float(o["size"])))
            except (TypeError, ValueError, KeyError):
                pass
        return out

    bids, asks = _levels("bids"), _levels("asks")
    best_bid = max((p for p, _ in bids), default=None)
    best_ask = min((p for p, _ in asks), default=None)
    bb = sum(s for p, s in bids if best_bid is not None and p == best_bid)
    ba = sum(s for p, s in asks if best_ask is not None and p == best_ask)
    bid_2c = sum(s for p, s in bids if best_bid is not None and p >= best_bid - within)
    ask_2c = sum(s for p, s in asks if best_ask is not None and p <= best_ask + within)
    return {
        "best_bid": best_bid, "best_bid_size": round(bb, 1),
        "best_ask": best_ask, "best_ask_size": round(ba, 1),
        "bid_depth_2c": round(bid_2c, 1), "ask_depth_2c": round(ask_2c, 1),
        "tick_size": book.get("tick_size"),
    }


def fetch_depth_table(price_df: pd.DataFrame, session: Optional[requests.Session] = None,
                      timeout: float = 30.0) -> pd.DataFrame:
    """Given a price df carrying a 'token_id' column, return a per-market order-book depth table.

    Columns: market_id, token_id, best_bid, best_bid_size, best_ask, best_ask_size,
    bid_depth_2c, ask_depth_2c, tick_size. Markets with no live book are dropped.
    """
    if price_df is None or price_df.empty or "token_id" not in price_df.columns:
        return pd.DataFrame()
    pairs = [(str(r["market_id"]), str(r["token_id"]))
             for _, r in price_df.iterrows() if r.get("token_id")]
    books = fetch_clob_books([t for _, t in pairs], session=session, timeout=timeout)
    rows = []
    for mid_, tok in pairs:
        bk = books.get(tok)
        if bk is None:
            continue
        rows.append({"market_id": mid_, "token_id": tok, **_depth_summary(bk)})
    return pd.DataFrame(rows)


def _fetch_event(slug: str, session: requests.Session, timeout: float = 30.0) -> Optional[dict]:
    r = session.get(f"{GAMMA}/events", params={"slug": slug}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data[0] if data else None


def fetch_wc_futures_prices(
    event_market_type: Optional[dict[str, str]] = None,
    only_accepting_orders: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Fetch current Polymarket WC futures prices.

    Returns
    -------
    (df, unmapped)
      df       : columns market_id, market_mid, spread, market_type, team, pm_question
      unmapped : Polymarket team names that did not resolve to a WC team (reported, skipped)
    """
    mapping = event_market_type or EVENT_MARKET_TYPE
    session = requests.Session()
    session.headers.update(_HEADERS)

    rows, unmapped = [], set()
    for slug, mtype in mapping.items():
        try:
            ev = _fetch_event(slug, session)
        except requests.RequestException as exc:
            log.warning("Could not fetch %s (%s) — skipping", slug, exc)
            continue
        if ev is None:
            log.warning("Event slug not found: %s — skipping (Polymarket may have renamed it)", slug)
            continue

        n_priced = 0
        for m in ev.get("markets", []):
            if only_accepting_orders and not m.get("acceptingOrders", True):
                continue
            ms = _mid_and_spread(m)
            if ms is None:
                continue
            mid, spread = ms
            team = _normalize_team(str(m.get("groupItemTitle", "")).strip())
            if team not in _ALL_WC:
                if m.get("groupItemTitle"):
                    unmapped.add(str(m.get("groupItemTitle")))
                continue
            yes_tok, no_tok = _token_ids(m)
            rows.append({
                "market_id":   f"{mtype}:{team}",
                "market_mid":  round(mid, 4),
                "spread":      round(max(spread, 0.0), 4),
                "market_type": mtype,
                "team":        team,
                "token_id":    yes_tok,
                "no_token_id": no_tok,
                "pm_question": m.get("question"),
            })
            n_priced += 1
        log.info("  %-46s -> %-12s  %d priced market(s)", slug, mtype, n_priced)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates("market_id").sort_values(["market_type", "market_mid"],
                                                         ascending=[True, False]).reset_index(drop=True)
    return df, sorted(unmapped)


def _fetch_tag_events(tag_id: int, session: requests.Session, max_pages: int = 6,
                      timeout: float = 60.0) -> list[dict]:
    """Fetch all open events under a Gamma tag (paginated, 100/page)."""
    events = []
    for off in range(0, max_pages * 100, 100):
        try:
            page = session.get(f"{GAMMA}/events",
                               params={"closed": "false", "tag_id": tag_id, "limit": 100, "offset": off},
                               timeout=timeout).json()
        except requests.RequestException as exc:
            log.warning("tag page offset=%d failed (%s)", off, exc)
            break
        if not page:
            break
        events.extend(page)
        if len(page) < 100:
            break
    return events


_DATE_END = __import__("re").compile(r"\d{4}-\d{2}-\d{2}$")


def fetch_wc_match_prices(timeout: float = 60.0) -> tuple[pd.DataFrame, list[str]]:
    """
    Fetch current Polymarket 3-way moneyline prices for upcoming WC matches.

    Each base match event (slug ends in a date, exactly 3 markets: two teams + a Draw)
    becomes three rows keyed on the canonical match key, so it joins to
    match_markets.build_match_market_table.

    Returns (df, unmapped) where df has columns:
      market_id, market_mid, spread, market_type='match', group, description
    """
    from src.betting.match_markets import canonical_match_key, match_market_ids

    session = requests.Session()
    session.headers.update(_HEADERS)
    events = _fetch_tag_events(WC_TAG_ID, session, timeout=timeout)

    rows, unmapped = [], set()
    for ev in events:
        slug = str(ev.get("slug", ""))
        mkts = ev.get("markets", []) or []
        if not slug.startswith("fifwc-") or not _DATE_END.search(slug) or len(mkts) != 3:
            continue

        team_price: dict[str, tuple[float, float]] = {}
        team_token: dict[str, tuple[Optional[str], Optional[str]]] = {}
        draw_ms = None
        draw_tokens = (None, None)
        for m in mkts:
            git = str(m.get("groupItemTitle", "")).strip()
            ms = _mid_and_spread(m)
            if ms is None:
                continue
            if git.lower().startswith("draw"):
                draw_ms = ms
                draw_tokens = _token_ids(m)
            else:
                team = _normalize_team(git)
                if team not in _ALL_WC:
                    unmapped.add(git)
                    continue
                team_price[team] = ms
                team_token[team] = _token_ids(m)

        if len(team_price) != 2 or draw_ms is None:
            continue
        (ta, ma), (tb, mb) = list(team_price.items())
        t1, t2, grp = canonical_match_key(ta, tb)
        ids = match_market_ids(ta, tb)
        per_team = {ta: ma, tb: mb}
        for outcome, ms in ((t1, per_team[t1]), ("draw", draw_ms), (t2, per_team[t2])):
            mid, spread = ms
            yes_tok, no_tok = draw_tokens if outcome == "draw" else team_token.get(outcome, (None, None))
            rows.append({
                "market_id":   ids[outcome],
                "market_mid":  round(mid, 4),
                "spread":      round(max(spread, 0.0), 4),
                "market_type": "match",
                "group":       grp,
                "token_id":    yes_tok,
                "no_token_id": no_tok,
                "description": (f"{t1} vs {t2} — draw" if outcome == "draw"
                                else f"{outcome} to win vs {t2 if outcome == t1 else t1}"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates("market_id").reset_index(drop=True)
    return df, sorted(unmapped)


def _max_fill_column(df: pd.DataFrame) -> pd.Series:
    """Per-market liquidity ceiling for betting_engine's Opportunity.max_fill: the smaller of
    the YES-ask and YES-bid 2c-depth. 'Buying NO' here is implemented as selling into the YES
    bid (see betting_engine.evaluate_opportunity's no_cost = 1 - market_mid), so the relevant
    size constraint differs by side; taking the min of both is a single, conservative number
    that's safe regardless of which side the engine ends up choosing. Matters mainly for thin
    longshot futures — liquid match markets have depth far above any plausible stake.
    Returns '' (no cap) per row if depth couldn't be fetched."""
    if df.empty or 'token_id' not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    try:
        depth = fetch_depth_table(df)
    except Exception as exc:   # noqa: BLE001
        log.warning("depth fetch for max_fill failed (leaving max_fill blank): %s", exc)
        return pd.Series([""] * len(df), index=df.index)
    if depth.empty or 'bid_depth_2c' not in depth.columns or 'ask_depth_2c' not in depth.columns:
        return pd.Series([""] * len(df), index=df.index)
    cap = (depth.drop_duplicates('market_id').set_index('market_id')
           [['bid_depth_2c', 'ask_depth_2c']].min(axis=1))
    return df['market_id'].map(cap).fillna("")


def write_match_price_csv(path: str | Path, timeout: float = 60.0) -> pd.DataFrame:
    """Fetch live match moneyline prices and write a price CSV for betting_matches.py."""
    df, unmapped = fetch_wc_match_prices(timeout=timeout)
    if unmapped:
        log.warning("Unmapped Polymarket team names in match markets (skipped): %s", unmapped)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["market_id", "market_mid", "spread"]
    out = df[cols].copy() if not df.empty else pd.DataFrame(columns=cols)
    out["max_fill"] = _max_fill_column(df)
    out.to_csv(path, index=False)
    log.info("Wrote %d match market(s) -> %s", len(out), path)
    return df


def write_price_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Fetch live futures prices and write the betting_slate price CSV. Returns the DataFrame."""
    df, unmapped = fetch_wc_futures_prices(**kwargs)
    if unmapped:
        log.warning("Unmapped Polymarket team names (skipped): %s", unmapped)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["market_id", "market_mid", "spread"]
    out = df[cols].copy() if not df.empty else pd.DataFrame(columns=cols)
    out["max_fill"] = _max_fill_column(df)   # auto-derived from order-book depth (see helper)
    out.to_csv(path, index=False)
    log.info("Wrote %d market(s) -> %s", len(out), path)
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    root = Path(__file__).resolve().parents[2]
    df = write_price_csv(root / "data" / "raw" / "polymarket_prices.csv")
    if not df.empty:
        print(f"\nFetched {len(df)} priced markets across {df['market_type'].nunique()} type(s):")
        print(df.groupby("market_type").size().to_string())
        print("\nTop champion prices:")
        champ = df[df["market_type"] == "champion"].head(8)
        print(champ[["team", "market_mid", "spread"]].to_string(index=False))
