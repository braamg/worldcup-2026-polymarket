"""
Shared interface for all match outcome predictors.

Every model must implement fit() and predict_proba().
predict_proba() returns a (n, 3) float array: columns = [P(H), P(D), P(A)].
predict_match() is a convenience wrapper for single-match queries.

Result encoding used everywhere in this package:
    H = 0  (home win)
    D = 1  (draw)
    A = 2  (away win)
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from src.ratings.elo import DEFAULT_RATING, get_elo

RESULT_ENCODE: dict[str, int] = {'H': 0, 'D': 1, 'A': 2}
RESULT_DECODE: dict[int, str] = {0: 'H', 1: 'D', 2: 'A'}

# Home advantage in Elo points. MUST equal the value the historical ratings were built
# with (tuned to 125 in phase2_build_elo.py) so the elo_diff feature is computed
# identically at train time and predict time. (Reconciled 2026-06-14: was 100, which
# silently skewed predict-time elo_diff for non-neutral matches vs the 125 used in
# training; matches config.yaml elo.home_advantage.)
ELO_HOME_ADV: float = 125.0


class BasePredictor(ABC):

    def __init__(self) -> None:
        self._elo_ts: pd.DataFrame | None = None

    @abstractmethod
    def fit(self, df: pd.DataFrame, elo_ts: pd.DataFrame) -> 'BasePredictor':
        """
        Train on modern-era modeling table rows.

        Parameters
        ----------
        df      : modeling_table rows with is_wc2026=False already filtered out by caller.
        elo_ts  : elo_timeseries (team, date, elo_before, elo_after) — stored for
                  Elo lookup in predict_match on future fixtures.
        """

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Outcome probabilities for each row in df.

        Returns
        -------
        ndarray of shape (n, 3): columns = [P(H), P(D), P(A)]
        """

    # ------------------------------------------------------------------
    # predict_match — convenience wrapper, builds one-row df then calls predict_proba
    # ------------------------------------------------------------------

    def predict_match(
        self,
        home: str,
        away: str,
        date,
        neutral: bool = False,
        stage: str = 'group',
        log_ratio_mv: float | None = None,
        rest_days_home: float = 30.0,
        rest_days_away: float = 30.0,
        elo_home: float | None = None,
        elo_away: float | None = None,
    ) -> tuple[float, float, float]:
        """
        Predict a single match.

        Parameters
        ----------
        home, away    : team names (used for Elo lookup if elo_home/elo_away not supplied)
        date          : match date (for as-of Elo lookup)
        neutral       : True if played at a neutral venue
        stage         : 'group' or 'knockout' — models that use this (Dixon-Coles) apply
                        knockout goal-rate deflation
        log_ratio_mv  : log(mv_home / mv_away); None → has_market_value=False
        rest_days_*   : days since last match (default 30 = neutral imputation)
        elo_home/away : override Elo rating; if None, looked up from stored elo_timeseries

        Returns
        -------
        (P_home_win, P_draw, P_away_win)
        """
        date = pd.Timestamp(date)
        if elo_home is None:
            elo_home = self._get_elo(home, date)
        if elo_away is None:
            elo_away = self._get_elo(away, date)

        row = self._build_row(
            home=home, away=away, date=date, neutral=neutral, stage=stage,
            elo_home=elo_home, elo_away=elo_away,
            log_ratio_mv=log_ratio_mv,
            rest_days_home=rest_days_home,
            rest_days_away=rest_days_away,
        )
        proba = self.predict_proba(pd.DataFrame([row]))
        return (float(proba[0, 0]), float(proba[0, 1]), float(proba[0, 2]))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_elo(self, team: str, date: pd.Timestamp) -> float:
        if self._elo_ts is None:
            return DEFAULT_RATING
        return get_elo(team, date, self._elo_ts)

    def _build_row(
        self,
        home: str, away: str, date: pd.Timestamp,
        neutral: bool, stage: str,
        elo_home: float, elo_away: float,
        log_ratio_mv: float | None,
        rest_days_home: float, rest_days_away: float,
    ) -> dict:
        """Build a comprehensive feature dict usable by any model's predict_proba."""
        elo_diff = elo_home - elo_away + (0.0 if neutral else ELO_HOME_ADV)
        e_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        return {
            'home_team': home,
            'away_team': away,
            'date': date,
            'neutral_venue': neutral,
            'stage': stage,
            'elo_home_pre': elo_home,
            'elo_away_pre': elo_away,
            'elo_diff': elo_diff,
            'e_home': e_home,
            'log_ratio_mv': log_ratio_mv,
            'has_market_value': log_ratio_mv is not None and not np.isnan(log_ratio_mv),
            'cross_conf': False,
            'rest_days_home': rest_days_home,
            'rest_days_away': rest_days_away,
            'rest_days_home_missing': False,
            'rest_days_away_missing': False,
        }
