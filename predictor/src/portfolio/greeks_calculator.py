"""
greeks_calculator.py — Delta and Theta for World Cup tournament futures.

A tournament-winner future (e.g. "Brazil wins the championship") is a path-dependent
derivative: its value depends on the full sequence of match outcomes through the bracket,
not just one terminal event. This makes it analogous to an exotic option:
    - Delta: sensitivity to an individual match outcome (like an option's delta to spot)
    - Theta: probability decay as calendar time passes with no new match results

These Greeks enable delta-neutral hedging: hold the long futures position for its
tournament-edge while shorting the next group-stage match to neutralize directional risk
from that specific match's outcome.

Delta is computed via MC conditioning:
    Δ = P(futures | team wins match) - P(futures | team loses match)

Theta is computed from the expected probability change per day given the upcoming schedule:
    Θ ≈ E[ΔP(futures)] / days_to_next_match   (simplified; sign depends on direction)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FuturesGreeks:
    """Greeks for one tournament futures position."""
    futures_market_id: str
    futures_prob: float              # current P(futures resolves YES)
    delta: Optional[float] = None    # ΔP(futures) per match outcome change
    theta: Optional[float] = None    # ΔP(futures) per day of calendar time
    delta_match_id: Optional[str] = None     # match that this delta refers to
    hedge_direction: Optional[str] = None    # 'short_home' | 'short_away' | None
    hedge_notional: Optional[float] = None   # suggested dollar hedge size (per $1 of futures)


class SportsGreeksEngine:
    """Compute Delta and Theta for World Cup tournament futures.

    Args:
        outcome_matrix: (n_sims × n_markets) binary 0/1 outcome matrix.
        market_ids: list of market_id strings (column order of outcome_matrix).
        n_sims: number of MC paths (inferred from outcome_matrix if not provided).
    """

    def __init__(
        self,
        outcome_matrix: np.ndarray,
        market_ids: list[str],
    ):
        self.outcomes = np.asarray(outcome_matrix, dtype=float)
        self.market_ids = list(market_ids)
        self.id_to_col = {mid: i for i, mid in enumerate(self.market_ids)}
        self.n_sims = self.outcomes.shape[0]

    def compute_delta(
        self,
        futures_market_id: str,
        match_home_market_id: str,
        match_away_market_id: str,
    ) -> float:
        """Delta of a futures contract w.r.t. one upcoming match's outcome.

        Delta = P(futures=1 | home_wins) - P(futures=1 | home_loses)

        We condition on match outcomes using paths where the match resolved as
        home win vs. paths where it resolved as away win (draws excluded from
        this directional delta).

        Returns:
            delta ∈ (-1, 1). Positive → futures benefits from this team winning.
        """
        f_col = self.id_to_col.get(futures_market_id)
        h_col = self.id_to_col.get(match_home_market_id)
        a_col = self.id_to_col.get(match_away_market_id)

        if f_col is None:
            raise KeyError(f"Futures market not found: {futures_market_id}")

        futures_outcomes = self.outcomes[:, f_col]

        # Condition on home win paths
        if h_col is not None:
            home_win_mask = self.outcomes[:, h_col] == 1
            p_futures_given_home_win = (
                futures_outcomes[home_win_mask].mean()
                if home_win_mask.any() else 0.5
            )
        else:
            # No explicit home-win column; approximate from absence of away win
            if a_col is not None:
                home_win_mask = self.outcomes[:, a_col] == 0
                p_futures_given_home_win = (
                    futures_outcomes[home_win_mask].mean()
                    if home_win_mask.any() else 0.5
                )
            else:
                return 0.0

        # Condition on away win paths
        if a_col is not None:
            away_win_mask = self.outcomes[:, a_col] == 1
            p_futures_given_away_win = (
                futures_outcomes[away_win_mask].mean()
                if away_win_mask.any() else 0.5
            )
        else:
            if h_col is not None:
                away_win_mask = self.outcomes[:, h_col] == 0
                p_futures_given_away_win = (
                    futures_outcomes[away_win_mask].mean()
                    if away_win_mask.any() else 0.5
                )
            else:
                return 0.0

        return round(float(p_futures_given_home_win - p_futures_given_away_win), 4)

    def compute_theta(
        self,
        futures_market_id: str,
        days_until_next_match: float = 1.0,
    ) -> float:
        """Theta: expected daily probability decay for a futures contract.

        Approximation: as matches are played, the bracket narrows and probability mass
        concentrates — but on average across all paths, probabilities also drift toward
        resolution. We estimate theta as the cross-sim standard deviation of outcomes
        divided by the remaining days (a simple uncertainty-depletion model).

        This is intentionally simple; a full theta model needs the actual simulator
        integration with upcoming match schedules.

        Returns:
            theta ∈ (-1, 0). Negative = the futures decays in probability per day
            (true for underdogs as the bracket narrows; favorites tend to have positive theta
            initially and then negative as they face stronger opponents later).
        """
        f_col = self.id_to_col.get(futures_market_id)
        if f_col is None:
            raise KeyError(f"Futures market not found: {futures_market_id}")

        futures_outcomes = self.outcomes[:, f_col]
        current_prob = float(futures_outcomes.mean())
        std = float(futures_outcomes.std())

        # Rough model: uncertainty resolves at ~std/sqrt(days) per day
        if days_until_next_match <= 0:
            return 0.0
        theta = -std / np.sqrt(max(days_until_next_match, 1.0))

        # For favorites (prob > 0.5) theta is typically positive (more paths confirm them)
        # For underdogs (prob < 0.5) theta is typically negative (paths eliminate them)
        sign = 1.0 if current_prob > 0.5 else -1.0
        return round(sign * abs(theta), 4)

    def build_delta_hedge(
        self,
        futures_market_id: str,
        match_home_market_id: str,
        match_away_market_id: str,
        futures_position_size: float,
        futures_price: float,
    ) -> dict:
        """Recommend a match moneyline hedge to neutralize the futures delta.

        For a LONG futures position:
            If delta > 0 (futures benefits from home win):
                → SHORT the home moneyline (or equivalently, BUY away/draw side)
                → hedge_notional = |delta| * futures_position_size / futures_price
            If delta < 0:
                → SHORT the away moneyline

        Args:
            futures_position_size: dollar value of the futures position (stake / price)
            futures_price: entry price of the futures (0–1)

        Returns:
            dict with hedge recommendation fields.
        """
        delta = self.compute_delta(
            futures_market_id, match_home_market_id, match_away_market_id
        )

        if abs(delta) < 1e-4:
            return {
                "delta": delta,
                "hedge_direction": None,
                "hedge_notional": 0.0,
                "note": "Delta near zero; no hedge needed.",
            }

        hedge_notional = abs(delta) * futures_position_size

        if delta > 0:
            hedge_direction = "short_home"
            note = (f"Long futures benefits from home win (Δ={delta:+.4f}). "
                    f"Short home moneyline for ${hedge_notional:.2f} to neutralize.")
        else:
            hedge_direction = "short_away"
            note = (f"Long futures benefits from away win (Δ={delta:+.4f}). "
                    f"Short away moneyline for ${hedge_notional:.2f} to neutralize.")

        return {
            "futures_market_id": futures_market_id,
            "match_home_market_id": match_home_market_id,
            "match_away_market_id": match_away_market_id,
            "delta": delta,
            "hedge_direction": hedge_direction,
            "hedge_notional": round(hedge_notional, 2),
            "note": note,
        }

    def compute_greeks_snapshot(
        self,
        futures_market_id: str,
        match_home_market_id: Optional[str] = None,
        match_away_market_id: Optional[str] = None,
        days_until_next_match: float = 1.0,
    ) -> FuturesGreeks:
        """Compute a full Greeks snapshot for one futures contract.

        Returns a FuturesGreeks dataclass with delta, theta, and hedge recommendation.
        """
        f_col = self.id_to_col.get(futures_market_id)
        if f_col is None:
            raise KeyError(f"Futures market not found: {futures_market_id}")

        futures_prob = float(self.outcomes[:, f_col].mean())

        delta = None
        hedge_direction = None
        hedge_notional = None
        if match_home_market_id and match_away_market_id:
            delta = self.compute_delta(
                futures_market_id, match_home_market_id, match_away_market_id
            )
            if delta is not None and abs(delta) >= 1e-4:
                hedge_direction = "short_home" if delta > 0 else "short_away"
                hedge_notional = abs(delta)   # per $1 of futures

        theta = self.compute_theta(futures_market_id, days_until_next_match)

        return FuturesGreeks(
            futures_market_id=futures_market_id,
            futures_prob=round(futures_prob, 4),
            delta=delta,
            theta=theta,
            delta_match_id=match_home_market_id,
            hedge_direction=hedge_direction,
            hedge_notional=round(hedge_notional, 4) if hedge_notional is not None else None,
        )
