"""
kelly_optimizer.py — multi-dimensional Kelly Criterion (Thorp, 2008).

Solves the constrained portfolio optimization:
    maximize:   f^T * e - (λ_risk / 2) * f^T * Σ * f
    subject to:
        sum(f) ≤ total_exposure_cap
        f[group_i] ≤ per_group_cap    for each mutually exclusive group
        0 ≤ f_i ≤ per_market_cap      per-market bounds

Where:
    e  = edge vector (model_prob - market_mid, net of costs)
    Σ  = covariance matrix from MCCovarianceExtractor
    λ  = 2 * kelly_fraction  (risk-aversion parameter)
    f* = optimal allocation vector (fraction of bankroll per market)

This is a convex QP (concave quadratic objective, linear constraints); SLSQP finds
the global optimum in milliseconds for the sizes we work with (<100 markets).

Reference: Thorp, E.O. (2008). "The Kelly Criterion and the Stock Market."
           Journal of Portfolio Management, 34(4), 11–18.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize


@dataclass
class KellyAllocation:
    """Output of MultiDimensionalKellyOptimizer.solve()."""
    allocations: np.ndarray        # fraction of bankroll for each market (n_markets,)
    expected_growth: float         # f^T * e - (λ/2) * f^T * Σ * f at the optimum
    sharpe_ratio: float            # expected_growth / portfolio_std
    success: bool = True
    message: str = "OK"


class MultiDimensionalKellyOptimizer:
    """Covariance-aware Kelly Criterion for a portfolio of correlated binary bets.

    Args:
        edges: edge vector, shape (n,). edge_i = model_prob_i - fair_price_i (net of costs).
        covariance: covariance matrix, shape (n, n). From MCCovarianceExtractor.
        kelly_fraction: fractional Kelly multiplier (default 0.25 = quarter Kelly).
        mutual_exclusion_groups: { group_key: [col_idx, ...] } — indices that form
            mutually exclusive outcome sets (e.g. "championship" markets for each team).
            The optimizer caps aggregate allocation within each group at per_group_cap.
        min_edge_threshold: markets with edge < this are zeroed before optimization.
    """

    def __init__(
        self,
        edges: np.ndarray,
        covariance: np.ndarray,
        kelly_fraction: float = 0.25,
        mutual_exclusion_groups: Optional[dict[str, list[int]]] = None,
        min_edge_threshold: float = 0.0,
    ):
        self.edges = np.asarray(edges, dtype=float)
        self.covariance = np.asarray(covariance, dtype=float)
        self.kelly_fraction = kelly_fraction
        self.mutual_exclusion_groups = mutual_exclusion_groups or {}
        self.min_edge_threshold = min_edge_threshold
        self.n = len(self.edges)

        if self.covariance.shape != (self.n, self.n):
            raise ValueError(f"covariance must be ({self.n}, {self.n}), got {self.covariance.shape}")

        # λ_risk = 2 * kelly_fraction: risk-aversion parameter in the objective
        self._lambda = 2.0 * kelly_fraction

    def _objective(self, f: np.ndarray) -> float:
        """Negative of the Kelly objective (we minimize negative = maximize)."""
        return -(f @ self.edges - (self._lambda / 2.0) * f @ self.covariance @ f)

    def _gradient(self, f: np.ndarray) -> np.ndarray:
        """Analytical gradient of the negative objective."""
        return -(self.edges - self._lambda * self.covariance @ f)

    def solve(
        self,
        bankroll: float = 1.0,
        max_total_exposure_fraction: float = 0.40,
        per_group_cap: float = 0.15,
        per_market_cap: float = 0.05,
    ) -> KellyAllocation:
        """Solve for the optimal allocation vector.

        Args:
            bankroll: current bankroll (only used to compute dollar stakes; not in optimization).
            max_total_exposure_fraction: hard cap on sum(f).
            per_group_cap: cap for each mutually exclusive group sum.
            per_market_cap: cap for any single market.

        Returns:
            KellyAllocation with .allocations as fractions of bankroll.
        """
        n = self.n

        # Zero out markets that don't clear the edge threshold
        active = self.edges >= self.min_edge_threshold
        if not active.any():
            return KellyAllocation(
                allocations=np.zeros(n),
                expected_growth=0.0,
                sharpe_ratio=0.0,
                success=True,
                message="No markets clear the minimum edge threshold.",
            )

        # Bounds: 0 ≤ f_i ≤ per_market_cap (0 for sub-threshold markets)
        bounds = [(0.0, per_market_cap if active[i] else 0.0) for i in range(n)]

        # Constraints
        constraints = [
            # Total exposure cap
            {"type": "ineq", "fun": lambda f: max_total_exposure_fraction - f.sum()},
        ]

        # Per-group caps for mutually exclusive groups
        for group_name, col_idxs in self.mutual_exclusion_groups.items():
            valid_idxs = [i for i in col_idxs if 0 <= i < n]
            if valid_idxs:
                def _group_constraint(f, idxs=valid_idxs):
                    return per_group_cap - sum(f[i] for i in idxs)
                constraints.append({"type": "ineq", "fun": _group_constraint})

        # Initial guess: uniform over active markets, scaled to a small total
        f0 = np.zeros(n)
        n_active = int(active.sum())
        if n_active:
            f0[active] = min(per_market_cap, max_total_exposure_fraction / n_active) * 0.5

        result = minimize(
            self._objective,
            f0,
            jac=self._gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "maxiter": 1000, "disp": False},
        )

        f_opt = np.clip(result.x, 0.0, per_market_cap)
        # Enforce total cap (can drift slightly from numerical rounding)
        if f_opt.sum() > max_total_exposure_fraction + 1e-9:
            f_opt *= max_total_exposure_fraction / f_opt.sum()

        expected_growth = float(
            f_opt @ self.edges - (self._lambda / 2.0) * f_opt @ self.covariance @ f_opt
        )
        portfolio_variance = float(f_opt @ self.covariance @ f_opt)
        portfolio_std = float(np.sqrt(max(portfolio_variance, 1e-12)))
        sharpe = expected_growth / portfolio_std if portfolio_std > 0 else 0.0

        return KellyAllocation(
            allocations=f_opt,
            expected_growth=round(expected_growth, 6),
            sharpe_ratio=round(sharpe, 4),
            success=result.success,
            message=result.message if not result.success else "OK",
        )

    def compare_with_uniform_kelly(
        self,
        bankroll: float = 1.0,
        per_market_cap: float = 0.05,
        total_exposure_cap: float = 0.40,
    ) -> dict:
        """Diagnostic: compare multi-dimensional Kelly vs. uniform (diagonal covariance).

        Returns a dict with 'multidim' and 'uniform' KellyAllocation objects plus
        capital-efficiency metrics.
        """
        multidim = self.solve(
            bankroll=bankroll,
            max_total_exposure_fraction=total_exposure_cap,
            per_market_cap=per_market_cap,
        )

        # Uniform Kelly: treat all markets as independent (diagonal covariance)
        diag_cov = np.diag(np.diag(self.covariance))
        uniform_opt = MultiDimensionalKellyOptimizer(
            edges=self.edges,
            covariance=diag_cov,
            kelly_fraction=self.kelly_fraction,
            mutual_exclusion_groups=self.mutual_exclusion_groups,
            min_edge_threshold=self.min_edge_threshold,
        )
        uniform = uniform_opt.solve(
            bankroll=bankroll,
            max_total_exposure_fraction=total_exposure_cap,
            per_market_cap=per_market_cap,
        )

        multidim_staked = float(multidim.allocations.sum()) * bankroll
        uniform_staked = float(uniform.allocations.sum()) * bankroll
        capital_efficiency_gain = (
            (multidim_staked - uniform_staked) / max(uniform_staked, 1e-9)
        )

        return {
            "multidim": multidim,
            "uniform": uniform,
            "multidim_total_staked": round(multidim_staked, 2),
            "uniform_total_staked": round(uniform_staked, 2),
            "capital_efficiency_gain_pct": round(capital_efficiency_gain * 100, 1),
            "multidim_sharpe": multidim.sharpe_ratio,
            "uniform_sharpe": uniform.sharpe_ratio,
        }
