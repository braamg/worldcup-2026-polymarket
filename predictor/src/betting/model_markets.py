"""
model_markets.py
================

The bridge between the tournament simulator's output and the betting engine.

The simulator (src/simulation/tournament.py, run()) produces a dict
    {team: {champion, reach_final, reach_sf, reach_qf, reach_r16, reach_r32,
            group_rank_1..4, ...}}
of calibrated probabilities. This module reshapes that into the tidy
(market_id, description, model_prob, group) table that betting_engine consumes,
covering the standard Polymarket World Cup market types.

market_id is a stable key (e.g. "champion:Brazil") so a market-price table can be
joined on it. group is the team name, so the engine's per-group cap stops all of
one team's correlated markets (winner + reach-final + win-group) from stacking.

IMPORTANT calibration caveat: single-MATCH W/D/L probabilities were validated
walk-forward (Phase 7) and on the WC slice (Phase 10). The TOURNAMENT-level
probabilities here (champion, reach-final, ...) are aggregations of many such
matches through the bracket plus Elo jitter; they sum to 1 and track the
bookmaker, but they are NOT independently calibrated (only one tournament ever
happens). Trust the shorter-horizon markets (advance / win-group) more than the
long-horizon ones (champion), and let the engine's humility haircut do its job.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

# (market_type, probability extractor, human description template)
MARKET_TYPES: list[tuple[str, Callable[[dict], float], str]] = [
    ("champion",     lambda p: p["champion"],                          "{team} to win the 2026 World Cup"),
    ("reach_final",  lambda p: p["reach_final"],                       "{team} to reach the final"),
    ("reach_sf",     lambda p: p["reach_sf"],                          "{team} to reach the semi-finals"),
    ("reach_qf",     lambda p: p["reach_qf"],                          "{team} to reach the quarter-finals"),
    ("reach_r16",    lambda p: p["reach_r16"],                         "{team} to reach the round of 16"),
    ("advance",      lambda p: p["reach_r32"],                         "{team} to advance from the group stage"),
    ("group_winner", lambda p: p["group_rank_1"],                      "{team} to win their group"),
    ("group_top2",   lambda p: p["group_rank_1"] + p["group_rank_2"],  "{team} to finish top 2 in their group"),
]

# Market types whose horizon is short enough that the per-match calibration carries
# over reasonably well. Long-horizon markets (champion, reach_final/sf) are flagged
# less-trusted so callers/analysis can weight them down if desired.
SHORT_HORIZON: frozenset = frozenset({"advance", "group_winner", "group_top2", "reach_r16"})


def build_model_market_table(
    probs: dict[str, dict],
    market_types: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Reshape the simulator probs dict into a model-market table.

    Parameters
    ----------
    probs        : {team: {stage: probability}} from WC2026Simulator.run()
    market_types : optional subset of market-type keys to include (default: all)

    Returns
    -------
    DataFrame[market_id, market_type, team, description, model_prob, group, short_horizon]
    """
    wanted = set(market_types) if market_types else {m[0] for m in MARKET_TYPES}
    rows = []
    for team, p in probs.items():
        for mtype, extract, desc in MARKET_TYPES:
            if mtype not in wanted:
                continue
            try:
                prob = float(extract(p))
            except (KeyError, TypeError):
                continue
            rows.append({
                "market_id":     f"{mtype}:{team}",
                "market_type":   mtype,
                "team":          team,
                "description":   desc.format(team=team),
                "model_prob":    round(prob, 6),
                "group":         team,                 # correlation tag: all of a team's markets
                "short_horizon": mtype in SHORT_HORIZON,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["market_type", "model_prob"], ascending=[True, False]).reset_index(drop=True)
    return df


def load_sim_probs(path: str | Path) -> dict[str, dict]:
    """Load a simulator probs dict previously saved by phase9_simulator.py."""
    with open(path) as f:
        return json.load(f)


def save_sim_probs(probs: dict[str, dict], path: str | Path) -> None:
    """Persist a simulator probs dict to JSON so the betting layer can consume it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(probs, f, indent=2)
