"""
covariance_builder.py — extract realized covariance from Monte Carlo tournament paths.

The tournament simulator produces an (n_sims × n_markets) binary outcome matrix.
Covariance between two markets captures how often they resolve together — e.g.
"Brazil wins group" and "Brazil wins championship" are highly positively correlated
because the same simulator paths produce both outcomes.

This is the structural information the multi-dimensional Kelly optimizer needs to
size correlated positions correctly. A diagonal-covariance assumption (independence)
would be strictly wrong here: the bracket creates dependencies by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MarketDefinition:
    """Describes one Polymarket market as known to the portfolio layer."""
    market_id: str
    description: str
    group: str                       # correlation group (e.g. team name)
    market_type: str = "future"      # 'future' | 'moneyline'
    model_prob: float = 0.5          # model-implied probability
    market_mid: float = 0.5          # Polymarket mid price


class MCCovarianceExtractor:
    """Extract a realized covariance matrix from Monte Carlo tournament outcome paths.

    Usage:
        extractor = MCCovarianceExtractor(outcome_matrix, market_ids)
        cov = extractor.compute_covariance()
        corr = extractor.compute_correlations()
        probs = extractor.compute_marginal_probabilities()
        groups = extractor.identify_mutually_exclusive_groups(market_defs)

    Args:
        outcome_matrix: shape (n_sims, n_markets), dtype float/int, 0/1 per cell
        market_ids: list of market_id strings matching the column order
    """

    def __init__(self, outcome_matrix: np.ndarray, market_ids: list[str]):
        if outcome_matrix.ndim != 2:
            raise ValueError("outcome_matrix must be 2-D (n_sims × n_markets)")
        self.outcomes = outcome_matrix.astype(float)
        self.market_ids = list(market_ids)
        self.n_sims, self.n_markets = self.outcomes.shape
        if len(self.market_ids) != self.n_markets:
            raise ValueError("len(market_ids) must equal outcome_matrix.shape[1]")

    # ------------------------------------------------------------------
    def compute_covariance(self) -> np.ndarray:
        """Realized covariance matrix Σ, shape (n_markets, n_markets).

        Σ_ij = E[(O_i - p_i)(O_j - p_j)] estimated from MC draws.
        Symmetrized and regularized with a tiny diagonal ridge (1e-8) so the
        matrix is always positive-definite for the SLSQP solver downstream.
        """
        centered = self.outcomes - self.outcomes.mean(axis=0)
        cov = (centered.T @ centered) / (self.n_sims - 1)
        # Symmetrize (should be symmetric already; float arithmetic can drift slightly)
        cov = (cov + cov.T) / 2.0
        # Tiny ridge for numerical stability
        cov += np.eye(self.n_markets) * 1e-8
        return cov

    def compute_correlations(self) -> np.ndarray:
        """Correlation matrix (diagonal = 1), shape (n_markets, n_markets)."""
        cov = self.compute_covariance()
        std = np.sqrt(np.diag(cov))
        std[std < 1e-10] = 1e-10          # avoid division by zero for degenerate markets
        return cov / np.outer(std, std)

    def compute_marginal_probabilities(self) -> dict[str, float]:
        """Realized win probability for each market from MC paths."""
        means = self.outcomes.mean(axis=0)
        return dict(zip(self.market_ids, means.tolist()))

    def identify_mutually_exclusive_groups(
        self, market_defs: list[MarketDefinition]
    ) -> dict[str, list[int]]:
        """Find sets of markets where at most one can resolve 1 in any simulation path.

        Returns: { group_key: [col_idx, col_idx, ...] } for each detected group.
        A group is 'mutually exclusive' if the sum of its outcomes across all simulation
        paths is ≤ 1 in every row (e.g. only one team can win the championship).

        Uses a fast heuristic: group markets by their MarketDefinition.group tag, then
        verify with verify_mutual_exclusion().
        """
        id_to_idx = {mid: i for i, mid in enumerate(self.market_ids)}
        groups: dict[str, list[int]] = {}
        for md in market_defs:
            if md.market_id in id_to_idx:
                groups.setdefault(md.group, []).append(id_to_idx[md.market_id])
        # Keep only groups where mutual exclusion is confirmed
        return {g: idxs for g, idxs in groups.items()
                if len(idxs) > 1 and self.verify_mutual_exclusion(idxs)}

    def verify_mutual_exclusion(self, col_indices: list[int]) -> bool:
        """Return True iff row sums for these columns are ≤ 1 in every simulation."""
        subset = self.outcomes[:, col_indices]
        return bool((subset.sum(axis=1) <= 1.0 + 1e-6).all())

    def correlation_summary(self, top_n: int = 10) -> pd.DataFrame:
        """Return the top_n most correlated market pairs (excluding self-pairs)."""
        corr = self.compute_correlations()
        rows = []
        for i in range(self.n_markets):
            for j in range(i + 1, self.n_markets):
                rows.append({
                    "market_a": self.market_ids[i],
                    "market_b": self.market_ids[j],
                    "correlation": round(float(corr[i, j]), 4),
                })
        df = pd.DataFrame(rows).sort_values("correlation", ascending=False, key=abs)
        return df.head(top_n).reset_index(drop=True)
