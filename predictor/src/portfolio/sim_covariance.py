"""
sim_covariance.py — load the MC outcome matrix and build the covariance matrix
used by MultiDimensionalKellyOptimizer and CVaRPortfolioManager.

This is the bridge between phase9_simulator.py (which writes
sim_outcome_matrix.npz) and the betting layer (which needs a covariance matrix
indexed by opportunity market_ids to size positions correctly).

Usage in betting scripts (paper_harness, betting_slate, betting_matches):

    from src.portfolio.sim_covariance import load_covariance_for_opportunities

    cov_matrix, cov_market_ids = load_covariance_for_opportunities(opportunities)
    cfg = BettingConfig(
        use_multidim_kelly=True,
        covariance_matrix=cov_matrix,
    )
    slate = build_bet_slate_multidim_kelly(opportunities, bankroll, cov_matrix, cfg)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from src.portfolio.covariance_builder import MCCovarianceExtractor

_PROC = Path(__file__).resolve().parents[2] / 'data' / 'processed'
_MATRIX_PATH = _PROC / 'sim_outcome_matrix.npz'
_IDS_PATH    = _PROC / 'sim_outcome_matrix_ids.json'

# Cache the loaded matrix across calls within a process (it's ~5–10 MB).
_CACHE: dict = {}


def load_outcome_matrix() -> tuple[np.ndarray, list[str]]:
    """Load the simulation outcome matrix from disk.

    Returns:
        outcome_matrix — shape (n_sims, n_markets), dtype float32
        market_ids     — list of market_id strings (column labels)

    Raises:
        FileNotFoundError if phase9 has not been run since the fix was applied.
    """
    global _CACHE
    if 'matrix' in _CACHE:
        return _CACHE['matrix'], _CACHE['ids']

    if not _MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Outcome matrix not found at {_MATRIX_PATH}. "
            "Run `python scripts/phase9_simulator.py` to generate it. "
            "This file is written by the updated phase9 and is required for "
            "multi-dimensional Kelly sizing."
        )
    if not _IDS_PATH.exists():
        raise FileNotFoundError(f"Market ID list not found at {_IDS_PATH}.")

    data = np.load(_MATRIX_PATH)
    matrix = data['outcome_matrix']
    ids = json.load(open(_IDS_PATH))

    _CACHE['matrix'] = matrix
    _CACHE['ids'] = ids
    return matrix, ids


def clear_cache() -> None:
    """Invalidate the in-process matrix cache (call after a new phase9 run)."""
    _CACHE.clear()


def load_covariance_for_opportunities(
    opportunities,
    subsample: Optional[int] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a covariance matrix aligned to the given opportunity list.

    For each Opportunity, we look up whether its market_id matches a column in
    the simulation outcome matrix. Matched markets use the MC-derived covariance.
    Unmatched markets (e.g. cross-strategy match bets not in the tournament sim)
    are assigned diagonal variance = p*(1-p) where p = market_mid, and zero
    covariance with all other markets — a conservative independence assumption.

    Args:
        opportunities: list of Opportunity objects (from betting_engine)
        subsample: if set, use only this many MC paths (faster for debugging).
                   Default uses all paths.

    Returns:
        cov_matrix   — (n_opps × n_opps) covariance matrix aligned to opportunities
        matched_ids  — list of market_ids that were found in the simulation
    """
    try:
        outcome_matrix, sim_ids = load_outcome_matrix()
    except FileNotFoundError as exc:
        # Phase9 hasn't been re-run yet. Return a diagonal matrix (independence
        # assumption) so the optimizer still runs — it's a degraded but valid
        # fallback while the matrix is being generated.
        n = len(opportunities)
        diagonal_cov = np.diag([
            float(o.model_prob) * (1.0 - float(o.model_prob))
            for o in opportunities
        ])
        return diagonal_cov, []

    if subsample is not None and subsample < outcome_matrix.shape[0]:
        rng = np.random.default_rng(42)
        idx = rng.choice(outcome_matrix.shape[0], size=subsample, replace=False)
        outcome_matrix = outcome_matrix[idx]

    sim_id_to_col = {mid: i for i, mid in enumerate(sim_ids)}
    n = len(opportunities)
    cov = np.zeros((n, n), dtype=float)
    matched = []

    # Build a per-opportunity column index (-1 = not in sim)
    opp_cols = []
    for opp in opportunities:
        col = sim_id_to_col.get(opp.market_id, -1)
        if col == -1:
            # Try to match by team+stage pattern from the Polymarket market_id
            # e.g. market_id 'champion:Brazil' -> sim key 'Brazil:champion'
            parts = str(opp.market_id).split(':')
            if len(parts) >= 2:
                candidate = f'{parts[-1]}:{parts[0]}'
                col = sim_id_to_col.get(candidate, -1)
        opp_cols.append(col)
        if col != -1:
            matched.append(opp.market_id)

    # Extract the submatrix for matched columns, fill in diagonal for unmatched
    matched_cols_data = {}
    for col in set(c for c in opp_cols if c != -1):
        matched_cols_data[col] = outcome_matrix[:, col].astype(float)

    for i in range(n):
        for j in range(i, n):
            ci, cj = opp_cols[i], opp_cols[j]
            if ci != -1 and cj != -1:
                # Both in sim: use MC covariance
                xi = matched_cols_data[ci] - matched_cols_data[ci].mean()
                xj = matched_cols_data[cj] - matched_cols_data[cj].mean()
                c = float(np.dot(xi, xj) / (len(xi) - 1))
            elif ci == cj == -1 and i == j:
                # Diagonal for unmatched: p*(1-p)
                p = float(opportunities[i].model_prob)
                c = p * (1.0 - p)
            else:
                # Off-diagonal involving an unmatched market: assume independence
                c = 0.0
            cov[i, j] = c
            cov[j, i] = c

    # Tiny ridge for numerical stability
    cov += np.eye(n) * 1e-8
    return cov, matched
