"""
results_fetch.py — auto-populate data/raw/wc2026_results.csv from The Odds API scores
endpoint, so settling paper bets needs no manual data entry.

The scores endpoint (1 credit/call) returns recently-completed games with final scores.
We map each to its fixture (group, team_a/team_b order, date) and append new ones, deduped.
Throttle it (the harness calls it a few times a day) to stay inside the free tier.
NO orders; read-only.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.betting.polymarket import _normalize_team
from src.betting.sharp_odds import ODDS_BASE, DEFAULT_SPORT
from src.utils import get_logger

log = get_logger(__name__)


def fetch_wc_scores(api_key: Optional[str] = None, sport: str = DEFAULT_SPORT,
                    days_from: int = 3, timeout: float = 30.0) -> list[dict]:
    """Completed games (home, away, home_score, away_score) from the scores endpoint."""
    key = api_key or os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise RuntimeError("No ODDS_API_KEY for results fetch.")
    r = requests.get(f"{ODDS_BASE}/sports/{sport}/scores",
                     params={"apiKey": key, "daysFrom": days_from}, timeout=timeout)
    r.raise_for_status()
    out = []
    for ev in r.json():
        if not ev.get("completed") or not ev.get("scores"):
            continue
        sc = {s["name"]: s["score"] for s in ev["scores"]}
        h, a = ev.get("home_team"), ev.get("away_team")
        if h in sc and a in sc:
            try:
                out.append({"home": h, "away": a, "hs": int(sc[h]), "as": int(sc[a]),
                            "commence": ev.get("commence_time", "")})
            except (TypeError, ValueError):
                continue
    return out


def update_results_csv(fixtures_path: str | Path, results_path: str | Path,
                       api_key: Optional[str] = None) -> tuple[int, list[str]]:
    """
    Append newly-completed WC results to wc2026_results.csv (deduped, mapped to fixtures).
    Returns (n_added, unmatched_descriptions).
    """
    fixtures = pd.read_csv(fixtures_path)
    results = pd.read_csv(results_path) if Path(results_path).exists() else \
        pd.DataFrame(columns=["group", "team_a", "team_b", "date", "goals_a", "goals_b"])
    have = set(zip(results["team_a"], results["team_b"]))
    have_pairs = {frozenset(p) for p in have}

    # fixture lookup by canonical team pair
    fx = {}
    for _, r in fixtures.iterrows():
        fx[frozenset({str(r["team_a"]), str(r["team_b"])})] = (
            str(r["group"]), str(r["team_a"]), str(r["team_b"]), str(r["date"]))

    added, unmatched = [], []
    for g in fetch_wc_scores(api_key):
        home, away = _normalize_team(g["home"]), _normalize_team(g["away"])
        pair = frozenset({home, away})
        if pair in have_pairs:
            continue   # already recorded
        fixt = fx.get(pair)
        if fixt is None:
            unmatched.append(f"{home} vs {away}")
            continue
        grp, ta, tb, date = fixt
        # orient goals to the fixture's team_a/team_b
        if home == ta:
            ga, gb = g["hs"], g["as"]
        else:
            ga, gb = g["as"], g["hs"]
        added.append({"group": grp, "team_a": ta, "team_b": tb, "date": date,
                      "goals_a": ga, "goals_b": gb})

    if added:
        results = pd.concat([results, pd.DataFrame(added)], ignore_index=True)
        results.to_csv(results_path, index=False)
        for a in added:
            log.info("auto-result: %s %d-%d %s", a["team_a"], a["goals_a"], a["goals_b"], a["team_b"])
    return len(added), unmatched
