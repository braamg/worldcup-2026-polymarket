"""
cvar_manager.py — Conditional Value at Risk (CVaR / Expected Shortfall) portfolio manager.

CVaR-α = E[loss | loss > VaR-α] = average loss in the worst (1-α)% of simulated scenarios.

This replaces static per-market hard caps with a dynamic, portfolio-aware acceptance rule:
    - marginal_cvar(current + new) - marginal_cvar(current) < threshold → ACCEPT new position
    - A new bet that *diversifies* the portfolio may increase CVaR by less than a small
      same-edge bet that *concentrates* an already-loaded risk factor.

Reference: Rockafellar & Uryasev (2000). "Optimization of Conditional Value-at-Risk."
           Journal of Risk, 2(3), 21–41.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Position:
    """One open position in the portfolio."""
    market_id: str
    side: str               # 'YES' or 'NO'
    stake: float            # dollars at risk
    cost: float             # entry price per share (0, 1)
    model_prob: float       # model-implied win probability
    group: str = ""


class CVaRPortfolioManager:
    """Dynamic risk management for a portfolio of prediction-market positions.

    The manager holds the current portfolio state (list of open positions) and
    uses the tournament simulator's outcome paths to evaluate tail risk.

    Args:
        outcome_matrix: (n_sims × n_markets) binary outcome matrix from tournament sim.
        market_ids: list of market_id strings matching the column order.
        confidence_level: α for CVaR (default 0.95 → worst 5% of scenarios).
        bankroll: current bankroll for relative-risk computations.
    """

    def __init__(
        self,
        outcome_matrix: np.ndarray,
        market_ids: list[str],
        confidence_level: float = 0.95,
        bankroll: float = 1000.0,
    ):
        self.outcomes = np.asarray(outcome_matrix, dtype=float)
        self.market_ids = list(market_ids)
        self.id_to_col = {mid: i for i, mid in enumerate(self.market_ids)}
        self.confidence_level = confidence_level
        self.bankroll = bankroll
        self.n_sims = self.outcomes.shape[0]
        self._positions: list[Position] = []

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------
    def add_position(self, pos: Position) -> None:
        self._positions.append(pos)

    def remove_position(self, market_id: str) -> None:
        self._positions = [p for p in self._positions if p.market_id != market_id]

    def clear(self) -> None:
        self._positions = []

    # ------------------------------------------------------------------
    # P/L simulation
    # ------------------------------------------------------------------
    def _simulate_pnl(self, positions: list[Position]) -> np.ndarray:
        """Return per-simulation P/L vector (dollars) for a given position set.

        For each sim path and each position:
            P/L = stake * (1 - cost) / cost   if won
            P/L = -stake                       if lost
        """
        pnl = np.zeros(self.n_sims)
        for pos in positions:
            col = self.id_to_col.get(pos.market_id)
            if col is None:
                continue
            yes_resolves = self.outcomes[:, col]   # 1 = YES resolved, 0 = NO resolved
            won = yes_resolves if pos.side == "YES" else (1.0 - yes_resolves)
            profit_per_share = (1.0 - pos.cost) / max(pos.cost, 1e-9)
            n_shares = pos.stake / max(pos.cost, 1e-9)
            pnl += won * (n_shares * profit_per_share) + (1.0 - won) * (-pos.stake)
        return pnl

    # ------------------------------------------------------------------
    # CVaR computation
    # ------------------------------------------------------------------
    def compute_portfolio_cvar(
        self, positions: Optional[list[Position]] = None
    ) -> float:
        """Compute CVaR-α for the given (or current) position set.

        Returns the average loss (in dollars) in the worst (1-α) fraction of paths.
        A positive return means expected loss in the tail (losses are positive here).
        """
        positions = positions if positions is not None else self._positions
        if not positions:
            return 0.0
        pnl = self._simulate_pnl(positions)
        losses = -pnl                           # losses are positive
        cutoff = np.quantile(losses, self.confidence_level)
        tail_mask = losses >= cutoff
        cvar = float(losses[tail_mask].mean()) if tail_mask.any() else 0.0
        return round(cvar, 2)

    def marginal_cvar(self, new_position: Position) -> float:
        """CVaR increase from adding new_position to the current portfolio."""
        before = self.compute_portfolio_cvar(self._positions)
        after = self.compute_portfolio_cvar(self._positions + [new_position])
        return round(after - before, 2)

    def accept_position(self, new_position: Position, cvar_target: float) -> tuple[bool, float]:
        """Accept/reject a new position using the marginal CVaR rule.

        Decision rule:
            accept if  marginal_CVaR < cvar_target / 5

        This means a single position can increase tail loss by at most 20% of the
        portfolio's overall CVaR budget — a conservative, portfolio-scaling gate.

        Returns:
            (accepted: bool, marginal_cvar: float)
        """
        threshold = cvar_target / 5.0
        marginal = self.marginal_cvar(new_position)
        return (marginal < threshold), marginal

    # ------------------------------------------------------------------
    # Risk reporting
    # ------------------------------------------------------------------
    def risk_report(self) -> dict:
        """Human-readable portfolio risk dashboard."""
        if not self._positions:
            return {"status": "no_positions", "cvar": 0.0}

        pnl = self._simulate_pnl(self._positions)
        losses = -pnl

        cvar = float(losses[losses >= np.quantile(losses, self.confidence_level)].mean())
        var = float(np.quantile(losses, self.confidence_level))
        expected_pnl = float(pnl.mean())
        worst_loss = float(losses.max())
        best_gain = float(pnl.max())

        total_stake = sum(p.stake for p in self._positions)
        n_groups = len({p.group for p in self._positions if p.group})

        return {
            "n_positions": len(self._positions),
            "n_groups": n_groups,
            "total_staked": round(total_stake, 2),
            "expected_pnl": round(expected_pnl, 2),
            f"cvar_{int(self.confidence_level * 100)}": round(cvar, 2),
            f"var_{int(self.confidence_level * 100)}": round(var, 2),
            "worst_case_loss": round(worst_loss, 2),
            "best_case_gain": round(best_gain, 2),
            "cvar_as_pct_bankroll": round(cvar / max(self.bankroll, 1.0) * 100, 2),
        }

    def per_position_marginal_cvar(self) -> pd.DataFrame:
        """Marginal CVaR contribution for every position in the current portfolio.

        Useful for identifying which positions are driving tail risk.
        """
        rows = []
        for pos in self._positions:
            # CVaR without this position
            without = [p for p in self._positions if p.market_id != pos.market_id]
            cvar_without = self.compute_portfolio_cvar(without)
            cvar_total = self.compute_portfolio_cvar(self._positions)
            marginal = round(cvar_total - cvar_without, 2)
            rows.append({
                "market_id": pos.market_id,
                "group": pos.group,
                "side": pos.side,
                "stake": pos.stake,
                "marginal_cvar": marginal,
                "marginal_cvar_pct": round(marginal / max(cvar_total, 1e-9) * 100, 1),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("marginal_cvar", ascending=False).reset_index(drop=True)
        return df
