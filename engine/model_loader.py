"""
model_loader.py
===============
Loads simulator probabilities from sim_probabilities.json and returns a clean
DataFrame of model probabilities per market, ready to join against Polymarket prices.

Path resolution order:
  1. SIM_PROBS_PATH env var (absolute override)
  2. local copy, then data/processed/ next to the engine, then ../predictor/data/processed/
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

# Market types and how to extract their probability from the simulator output
_MARKET_TYPES = [
    ("champion",    lambda p: p.get("champion", 0),       "{team} to win the World Cup"),
    ("reach_final", lambda p: p.get("reach_final", 0),    "{team} to reach the final"),
    ("reach_sf",    lambda p: p.get("reach_sf", 0),       "{team} to reach the semis"),
    ("reach_qf",    lambda p: p.get("reach_qf", 0),       "{team} to reach the quarters"),
    ("reach_r16",   lambda p: p.get("reach_r16", 0),      "{team} to reach the Round of 16"),
    ("advance",     lambda p: p.get("reach_r32", 0),      "{team} to advance from the group"),
]

def _resolve_path() -> Path:
    """
    Find sim_probabilities.json. Order:
      1. SIM_PROBS_PATH env override
      2. local copy next to this module (v2 is self-contained)
      3. data/processed/ under v2, then the sibling parent project
    Returns the first that exists, else the local default (so callers can detect absence).
    """
    env = os.environ.get("SIM_PROBS_PATH")
    if env:
        return Path(env)

    root = Path(__file__).resolve().parent
    candidates = [
        root / "sim_probabilities.json",
        root / "data" / "processed" / "sim_probabilities.json",
        root.parent / "data" / "processed" / "sim_probabilities.json",
        root.parent / "predictor" / "data" / "processed" / "sim_probabilities.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def load_model_probs(path: Path = None) -> pd.DataFrame:
    """
    Load simulator probabilities and return a tidy DataFrame.

    Returns
    -------
    DataFrame with columns:
        market_id    : str   e.g. "champion:Brazil"
        team         : str
        market_type  : str
        description  : str   human-readable label
        model_prob   : float probability from Monte Carlo simulation
    """
    if path is None:
        path = _resolve_path()
    if not path.exists():
        return pd.DataFrame(columns=["market_id", "team", "market_type", "description", "model_prob"])

    with open(path) as f:
        raw: dict = json.load(f)

    rows = []
    for team, stages in raw.items():
        for mtype, extract, desc_tmpl in _MARKET_TYPES:
            try:
                prob = float(extract(stages))
            except (KeyError, TypeError):
                continue
            rows.append({
                "market_id":   f"{mtype}:{team}",
                "team":        team,
                "market_type": mtype,
                "description": desc_tmpl.format(team=team),
                "model_prob":  round(prob, 6),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["market_type", "model_prob"], ascending=[True, False]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Single-MATCH model probabilities (3-way moneyline) — match_probabilities.json
# ---------------------------------------------------------------------------

def _resolve_match_path() -> Path:
    env = os.environ.get("MATCH_PROBS_PATH")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parent
    candidates = [
        root / "match_probabilities.json",
        root / "data" / "processed" / "match_probabilities.json",
        root.parent / "data" / "processed" / "match_probabilities.json",
        root.parent / "predictor" / "data" / "processed" / "match_probabilities.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def load_match_probs(path: Path = None) -> pd.DataFrame:
    """
    Load the model's single-match 3-way moneyline probabilities.

    Returns DataFrame[market_id, selection, model_prob, match_date, group_id]
    (empty DataFrame if the file is absent). market_id matches the canonical key used by
    polymarket_client.fetch_match_prices() so the two join cleanly.
    """
    cols = ["market_id", "selection", "model_prob", "match_date", "group_id"]
    if path is None:
        path = _resolve_match_path()
    if not path.exists():
        return pd.DataFrame(columns=cols)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df if not df.empty else pd.DataFrame(columns=cols)
