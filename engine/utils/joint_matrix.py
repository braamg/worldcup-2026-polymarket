"""
utils/joint_matrix.py
=====================
Consume the simulator's raw joint outcome matrix for EXACT bundle probabilities.

phase9_simulator.py already persists two artifacts in the parent project's
data/processed/ (no changes needed there):

  - sim_outcome_matrix.npz       binary matrix, shape (n_sims, n_markets)
                                 row = one simulated tournament, column = one market
  - sim_outcome_matrix_ids.json  the market id for each column, format "Team:stage"
                                 e.g. "Brazil:champion", "Morocco:reach_sf"

Because each column is the actual YES/NO outcome of that market *within the same
simulated tournament*, AND-ing a bundle's columns and counting rows gives the TRUE joint
probability — cross-team mutual exclusivity and intra-team nesting included. This replaces
both the independence assumption and the single-uniform-per-team approximation in
portfolio.py.

If the files are not found (e.g. running on mock data, or the parent project isn't on this
machine), every function degrades gracefully to ``None`` and the caller falls back to the
approximation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# v2 market label  ->  phase9 stage token
_STAGE_FROM_MATCH = {
    "Champion market":       "champion",
    "Reach Final":           "reach_final",
    "Reach Semi-Finals":     "reach_sf",
    "Reach Quarter-Finals":  "reach_qf",
    "Reach Round of 16":     "reach_r16",
    "Group Stage":           "reach_r32",   # v2 calls group-advance "advance"; sim calls it reach_r32
}

# v2 / Polymarket team spellings  ->  simulator canonical names
_TEAM_ALIASES = {
    "USA":             "United States",
    "Turkiye":         "Turkey",
    "Korea Republic":  "South Korea",
    "Cote d'Ivoire":   "Ivory Coast",
    "Cabo Verde":      "Cape Verde",
}

# Module-level cache so we don't reload the matrix on every Streamlit rerun.
_CACHE: dict = {}


def _resolve_paths() -> tuple[Path | None, Path | None]:
    """Find the matrix + ids files via env override or a few default locations."""
    env = os.environ.get("SIM_MATRIX_PATH")
    if env:
        npz = Path(env)
        ids = Path(os.environ.get("SIM_MATRIX_IDS_PATH", npz.with_name("sim_outcome_matrix_ids.json")))
        return (npz, ids) if npz.exists() and ids.exists() else (None, None)

    root = Path(__file__).resolve().parents[1]          # the v2 project folder
    candidates = [
        root,                                            # local copy next to sim_probabilities.json
        root / "data" / "processed",
        root.parent / "data" / "processed",              # sibling layout
        root.parent / "predictor" / "data" / "processed",
    ]
    for base in candidates:
        npz = base / "sim_outcome_matrix.npz"
        ids = base / "sim_outcome_matrix_ids.json"
        if npz.exists() and ids.exists():
            return npz, ids
    return None, None


def load_joint_matrix() -> tuple[np.ndarray | None, dict[str, int] | None]:
    """
    Load the outcome matrix and an id->column index map. Cached. Returns (None, None)
    if the artifacts are unavailable.
    """
    if "loaded" in _CACHE:
        return _CACHE["matrix"], _CACHE["id_to_col"]

    npz_path, ids_path = _resolve_paths()
    if npz_path is None:
        _CACHE.update(loaded=True, matrix=None, id_to_col=None, n_sims=0)
        return None, None

    with np.load(npz_path) as data:
        key = "outcome_matrix" if "outcome_matrix" in data else data.files[0]
        matrix = data[key].astype(bool)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    id_to_col = {mid: i for i, mid in enumerate(ids)}

    _CACHE.update(loaded=True, matrix=matrix, id_to_col=id_to_col, n_sims=matrix.shape[0])
    return matrix, id_to_col


def is_available() -> bool:
    load_joint_matrix()
    return _CACHE.get("matrix") is not None


def n_sims() -> int:
    load_joint_matrix()
    return _CACHE.get("n_sims", 0)


def _leg_to_matrix_id(team: str, match_label: str) -> str | None:
    stage = _STAGE_FROM_MATCH.get(match_label)
    if stage is None:
        return None
    canon = _TEAM_ALIASES.get(team.strip(), team.strip())
    return f"{canon}:{stage}"


def bundle_win_matrix(legs_df: pd.DataFrame) -> np.ndarray | None:
    """
    Return a boolean (n_sims, n_legs) matrix of the actual per-simulation outcomes for the
    bundle's legs, drawn from the joint simulation. Returns None if the matrix is
    unavailable or ANY leg cannot be mapped to a column (caller then falls back).
    """
    matrix, id_to_col = load_joint_matrix()
    if matrix is None:
        return None

    cols = []
    for _, r in legs_df.iterrows():
        mid = _leg_to_matrix_id(str(r["team"]), str(r["match"]))
        if mid is None or mid not in id_to_col:
            return None
        cols.append(id_to_col[mid])
    return matrix[:, cols]


def joint_probability(legs_df: pd.DataFrame) -> float | None:
    """Exact P(all legs resolve YES) from the joint simulation, or None if unavailable."""
    wm = bundle_win_matrix(legs_df)
    if wm is None:
        return None
    return float(wm.all(axis=1).mean())
