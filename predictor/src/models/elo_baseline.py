"""
Elo-only baseline predictor.

Maps elo_diff (pre-match Elo gap including home advantage for non-neutral venues)
to W/D/L probabilities via multinomial logistic regression.

This is the floor model: everything else must beat it.
Sample weights (training_weight) are applied so friendlies are down-weighted.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.base import BasePredictor, RESULT_ENCODE
from src.utils import get_logger

log = get_logger(__name__)

_FEATURE = 'elo_diff'


class EloBaseline(BasePredictor):
    """Multinomial logistic regression on a single feature: elo_diff."""

    def __init__(self) -> None:
        super().__init__()
        self.clf_: LogisticRegression | None = None

    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, elo_ts: pd.DataFrame) -> 'EloBaseline':
        self._elo_ts = elo_ts.sort_values(['team', 'date']).reset_index(drop=True)

        train = df.dropna(subset=[_FEATURE, 'result', 'training_weight'])
        X = train[[_FEATURE]].values
        y = train['result'].map(RESULT_ENCODE).values
        w = train['training_weight'].values

        # C=10 = weak regularisation; with 1 feature and ~29k samples, overfit risk is nil
        # multi_class removed in sklearn 1.7+ (default is multinomial for lbfgs)
        self.clf_ = LogisticRegression(
            solver='lbfgs',
            C=10.0,
            max_iter=500,
            random_state=42,
        )
        self.clf_.fit(X, y, sample_weight=w)

        log.info(
            "EloBaseline fitted on %d matches  classes=%s",
            len(train),
            list(self.clf_.classes_),
        )
        return self

    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns (n, 3) array [P(H), P(D), P(A)].
        Expects df to have an 'elo_diff' column (or 'elo_home_pre'+'elo_away_pre'+'neutral_venue').
        """
        if 'elo_diff' in df.columns:
            X = df[['elo_diff']].values
        else:
            # compute from raw Elo columns (predict_match path)
            from src.models.base import ELO_HOME_ADV
            neutral = df['neutral_venue'].values.astype(bool)
            elo_diff = (
                df['elo_home_pre'].values
                - df['elo_away_pre'].values
                + np.where(neutral, 0.0, ELO_HOME_ADV)
            )
            X = elo_diff.reshape(-1, 1)

        raw = self.clf_.predict_proba(X)  # shape (n, 3), columns in classes_ order

        # classes_ is sorted: [0, 1, 2] = [H, D, A] — already in correct order
        # Verify and reorder defensively
        cls = list(self.clf_.classes_)
        h_idx = cls.index(RESULT_ENCODE['H'])  # 0
        d_idx = cls.index(RESULT_ENCODE['D'])  # 1
        a_idx = cls.index(RESULT_ENCODE['A'])  # 2
        return raw[:, [h_idx, d_idx, a_idx]]

    # ------------------------------------------------------------------

    def summary(self) -> str:
        if self.clf_ is None:
            return "EloBaseline: not fitted"
        coef = self.clf_.coef_  # (3, 1): row per class [H, D, A]
        intercepts = self.clf_.intercept_
        lines = ["EloBaseline (multinomial logistic on elo_diff)"]
        for i, label in enumerate(['H', 'D', 'A']):
            lines.append(
                f"  class {label}: coef={coef[i, 0]:+.5f}  intercept={intercepts[i]:+.5f}"
            )
        return '\n'.join(lines)
