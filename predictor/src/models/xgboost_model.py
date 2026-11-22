"""
XGBoost multinomial classifier on the full Phase-5 feature set.

Features used (10 total):
  elo_diff               — Elo gap including home-advantage for non-neutral (continuous)
  e_home                 — Elo-implied home win probability (bounded 0-1)
  log_ratio_mv           — log(mv_home/mv_away); NaN for ~83% of rows (XGB handles native)
  has_market_value       — bool flag for log_ratio_mv presence
  neutral_venue          — bool
  cross_conf             — bool: home and away from different confederations
  rest_days_home         — days since home team's last match (imputed to 30)
  rest_days_away         — days since away team's last match (imputed to 30)
  rest_days_home_missing — bool: rest days were imputed
  rest_days_away_missing — bool: rest days were imputed

Target: H=0, D=1, A=2  (consistent with RESULT_ENCODE in base.py)

Validation for early stopping: last 20% of training rows by date (walk-forward,
never a random split per CLAUDE.md temporal-integrity requirement).

Sample weights (training_weight) applied to training set; val set is unweighted
so we measure true unweighted log-loss to guide early stopping.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.models.base import BasePredictor, RESULT_ENCODE, ELO_HOME_ADV
from src.utils import get_logger

log = get_logger(__name__)

FEATURES: list[str] = [
    'elo_diff',
    'e_home',
    'log_ratio_mv',
    'has_market_value',
    'neutral_venue',
    'cross_conf',
    'rest_days_home',
    'rest_days_away',
    'rest_days_home_missing',
    'rest_days_away_missing',
]

# Regularised params tuned for international football (sparse data, ~200 teams)
XGB_PARAMS: dict = dict(
    objective='multi:softprob',
    num_class=3,
    n_estimators=2000,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=50,    # strong — many class-pair cells have few samples
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=0.5,
    early_stopping_rounds=50,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
)

VAL_FRACTION: float = 0.20   # last 20% by date used as early-stopping validation


class XGBoostPredictor(BasePredictor):

    def __init__(self) -> None:
        super().__init__()
        self.clf_: XGBClassifier | None = None
        self.best_iteration_: int = 0
        self.val_logloss_: float = float('nan')

    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, elo_ts: pd.DataFrame) -> 'XGBoostPredictor':
        self._elo_ts = elo_ts.sort_values(['team', 'date']).reset_index(drop=True)

        train = df.dropna(subset=['result', 'training_weight']).copy()
        train = train.sort_values('date').reset_index(drop=True)

        y = train['result'].map(RESULT_ENCODE).values

        X = self._make_X(train)

        # Walk-forward split: last 20% of rows by date for early stopping
        cut = int(len(train) * (1 - VAL_FRACTION))
        X_tr, X_val = X[:cut], X[cut:]
        y_tr, y_val = y[:cut], y[cut:]
        w_tr = train['training_weight'].values[:cut]

        log.info(
            "XGBoost  train=%d  val=%d  features=%d",
            len(X_tr), len(X_val), X.shape[1],
        )

        self.clf_ = XGBClassifier(**XGB_PARAMS)
        self.clf_.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        self.best_iteration_ = self.clf_.best_iteration
        # Extract final val logloss
        evals = self.clf_.evals_result()
        if evals and 'validation_0' in evals:
            lls = evals['validation_0'].get('mlogloss', [float('nan')])
            self.val_logloss_ = float(lls[self.best_iteration_]) if lls else float('nan')

        log.info(
            "XGBoost fitted  best_iter=%d  val_logloss=%.5f",
            self.best_iteration_, self.val_logloss_,
        )
        return self

    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns (n, 3) array [P(H), P(D), P(A)].

        Builds the feature matrix from whatever columns are present.
        Missing feature columns get NaN (XGBoost handles natively).
        """
        X = self._make_X(df)
        raw = self.clf_.predict_proba(X)   # (n, 3): columns in 0,1,2 order = H,D,A ✓
        return raw

    # ------------------------------------------------------------------

    def _make_X(self, df: pd.DataFrame) -> np.ndarray:
        """Build (n, n_features) float array from df. Missing columns → NaN."""
        n = len(df)
        rows = np.full((n, len(FEATURES)), np.nan)
        for j, feat in enumerate(FEATURES):
            if feat in df.columns:
                col = df[feat].values
                # Cast booleans to float
                if col.dtype == bool or col.dtype == object:
                    col = col.astype(float)
                rows[:, j] = col.astype(float)

        # If elo_diff is missing but raw Elo columns are present, compute it
        if 'elo_diff' not in df.columns and 'elo_home_pre' in df.columns:
            neutral = df['neutral_venue'].values.astype(bool)
            rows[:, FEATURES.index('elo_diff')] = (
                df['elo_home_pre'].values
                - df['elo_away_pre'].values
                + np.where(neutral, 0.0, ELO_HOME_ADV)
            )
            if 'e_home' not in df.columns:
                elo_diff = rows[:, FEATURES.index('elo_diff')]
                rows[:, FEATURES.index('e_home')] = (
                    1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
                )

        return rows

    # ------------------------------------------------------------------

    def feature_importances(self) -> pd.Series:
        if self.clf_ is None:
            return pd.Series(dtype=float)
        imp = self.clf_.feature_importances_
        return pd.Series(imp, index=FEATURES, name='importance').sort_values(ascending=False)

    def summary(self) -> str:
        if self.clf_ is None:
            return "XGBoostPredictor: not fitted"
        lines = [
            f"XGBoostPredictor  best_iter={self.best_iteration_}  val_logloss={self.val_logloss_:.5f}",
            f"  features: {FEATURES}",
            "  feature importances:",
        ]
        for feat, imp in self.feature_importances().items():
            lines.append(f"    {feat:<35} {imp:.4f}")
        return '\n'.join(lines)
