"""
Dixon-Coles bivariate Poisson model.

Models P(home_goals=i, away_goals=j) as independent Poisson processes multiplied
by a low-scoring correction term tau(i, j, rho) (Dixon & Coles 1997).

Parameters per team:
  attack[team]   — log-scale attacking strength
  defense[team]  — log-scale defensive vulnerability
Global:
  home_adv       — home venue bonus in log-goal space
  rho            — Dixon-Coles low-score correlation parameter

Lambda equations:
  lambda_h = exp(attack[home] + defense[away] + home_adv * (1 - neutral))
  lambda_a = exp(attack[away] + defense[home])

Training uses exponential time-decay * match-importance weights combined.
L-BFGS-B optimization with analytical gradient for speed.

predict_proba(df) columns: home_team, away_team, neutral_venue, [stage]
  → (n, 3) float array [P(H), P(D), P(A)]

Knockout deflation: multiply both lambdas by ko_deflation (from config) when
stage == 'knockout'. Applied at prediction time only, not training time.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from src.models.base import BasePredictor, RESULT_ENCODE
from src.utils import get_logger, load_config

log = get_logger(__name__)

MAX_GOALS: int = 10          # scoreline grid: 0..10 each axis
DECAY_RATE: float = 0.002    # per day; half-life ≈ 347 days
RHO_BOUND: float = 0.15      # |rho| upper bound keeps tau(0,0) > 0 for lambda<=6.7

# ---------------------------------------------------------------------------
# Regularisation (empirical-Bayes shrinkage)
# ---------------------------------------------------------------------------
# Two ridge terms on attack/defense:
#   l2_global : weak anchor toward 0. Pins the additive gauge (attack+k / defense-k
#               leaves every lambda unchanged), so the optimiser cannot drift.
#   l2_conf   : the real workhorse. Shrinks each team toward its CONFEDERATION mean
#               attack/defense. Data-poor / outlier teams (e.g. Morocco's 2022
#               defense) are pulled toward their continental peers instead of being
#               trusted at face value from a handful of matches.
# EMPIRICAL FINDING (2026-06-14 sweep, scripts/_sweep_dc_shrinkage.py):
# confederation-mean shrinkage helps DC *standalone* RPS slightly (0.18233→0.18128)
# but HURTS the dc_xgb_avg ensemble at every strength (0.16968→0.16990+), because
# shrinking DC toward the mean makes it more redundant with XGBoost and destroys the
# error-diversity that is the entire reason for the ensemble. Defaults therefore
# reproduce the validated optimum: l2_global=0.01 (== old single L2_REG), l2_conf=0.
# The knob is retained, configurable, and validated-neutral — do not raise it without
# re-running the sweep and confirming the ENSEMBLE (not DC alone) improves.
L2_GLOBAL_DEFAULT: float = 0.01
L2_CONF_DEFAULT:   float = 0.0


# ---------------------------------------------------------------------------
# Vectorised negative log-likelihood + analytical gradient
# ---------------------------------------------------------------------------

def _nll_and_grad(
    params: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    goals_h: np.ndarray,
    goals_a: np.ndarray,
    is_neutral: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    conf_idx: np.ndarray,
    n_groups: int,
    l2_global: float,
    l2_conf: float,
) -> tuple[float, np.ndarray]:

    attack    = params[:n_teams]
    defense   = params[n_teams:2 * n_teams]
    home_adv  = params[2 * n_teams]
    rho       = params[2 * n_teams + 1]

    ha = np.where(is_neutral, 0.0, home_adv)
    lh = np.exp(attack[home_idx] + defense[away_idx] + ha)
    la = np.exp(attack[away_idx] + defense[home_idx])

    # ---------- DC tau and its partial derivatives ----------
    tau          = np.ones(len(goals_h))
    dtau_dlh     = np.zeros(len(goals_h))
    dtau_dla     = np.zeros(len(goals_h))
    dtau_drho    = np.zeros(len(goals_h))

    m00 = (goals_h == 0) & (goals_a == 0)
    m01 = (goals_h == 0) & (goals_a == 1)
    m10 = (goals_h == 1) & (goals_a == 0)
    m11 = (goals_h == 1) & (goals_a == 1)

    tau[m00]       = 1 - lh[m00] * la[m00] * rho
    tau[m01]       = 1 + lh[m01] * rho
    tau[m10]       = 1 + la[m10] * rho
    tau[m11]       = 1 - rho

    dtau_dlh[m00]  = -la[m00] * rho
    dtau_dlh[m01]  = rho

    dtau_dla[m00]  = -lh[m00] * rho
    dtau_dla[m10]  = rho

    dtau_drho[m00] = -lh[m00] * la[m00]
    dtau_drho[m01] = lh[m01]
    dtau_drho[m10] = la[m10]
    dtau_drho[m11] = -1.0

    # Guard: if any tau <= 0 we're in an infeasible region
    if np.any(tau <= 0):
        return 1e10, np.zeros_like(params)

    # ---------- log-likelihood ----------
    ll = (
        goals_h * np.log(lh) - lh - gammaln(goals_h + 1)
        + goals_a * np.log(la) - la - gammaln(goals_a + 1)
        + np.log(tau)
    )
    # ---------- empirical-Bayes ridge: global anchor + confederation-mean shrink ----------
    # Confederation means (per group), then each team's deviation from its group mean.
    counts      = np.bincount(conf_idx, minlength=n_groups).astype(float)
    counts_safe = np.maximum(counts, 1.0)
    mean_att = np.bincount(conf_idx, weights=attack,  minlength=n_groups) / counts_safe
    mean_def = np.bincount(conf_idx, weights=defense, minlength=n_groups) / counts_safe
    att_dev  = attack  - mean_att[conf_idx]
    def_dev  = defense - mean_def[conf_idx]

    reg = (
        l2_global * (np.dot(attack, attack) + np.dot(defense, defense))
        + l2_conf * (np.dot(att_dev, att_dev) + np.dot(def_dev, def_dev))
    )
    nll = -np.dot(weights, ll) + reg

    # ---------- gradient ----------
    # dh_m = d(LL_m)/d(log lambda_h_m) = (goals_h - lh) + lh * dtau_dlh / tau
    dh = (goals_h - lh) + lh * dtau_dlh / tau
    da = (goals_a - la) + la * dtau_dla / tau

    w_dh = weights * dh
    w_da = weights * da

    # attack[home] and defense[away] both affect lambda_h with the same coefficient
    grad_attack  = np.zeros(n_teams)
    grad_defense = np.zeros(n_teams)

    np.add.at(grad_attack,  home_idx, -w_dh)
    np.add.at(grad_attack,  away_idx, -w_da)
    np.add.at(grad_defense, away_idx, -w_dh)   # lambda_h = exp(... + defense[away])
    np.add.at(grad_defense, home_idx, -w_da)   # lambda_a = exp(... + defense[home])

    # Gradient of the ridge terms. The confederation-mean cross-term cancels
    # exactly (deviations from a group mean sum to zero), so the gradient is simply
    # 2*l2_conf*(param - group_mean) — see fit() docstring / project notes.
    grad_attack  += 2 * l2_global * attack  + 2 * l2_conf * att_dev
    grad_defense += 2 * l2_global * defense + 2 * l2_conf * def_dev

    grad_home_adv = -np.dot(w_dh, (1.0 - is_neutral.astype(float)))
    grad_rho      = -np.dot(weights, dtau_drho / tau)

    grad = np.concatenate([grad_attack, grad_defense, [grad_home_adv, grad_rho]])
    return float(nll), grad


# ---------------------------------------------------------------------------
# Scoreline → W/D/L probability (vectorised over a batch of matches)
# ---------------------------------------------------------------------------

def _outcomes_batch(lh_batch: np.ndarray, la_batch: np.ndarray, rho: float) -> np.ndarray:
    """
    Compute [P(H), P(D), P(A)] for each (lambda_h, lambda_a) pair.

    Returns ndarray of shape (n, 3).
    """
    from scipy.stats import poisson

    n = len(lh_batch)
    scores = np.arange(MAX_GOALS + 1)

    p_h = poisson.pmf(scores[None, :], lh_batch[:, None])   # (n, MAX+1)
    p_a = poisson.pmf(scores[None, :], la_batch[:, None])   # (n, MAX+1)

    # (n, MAX+1, MAX+1) — home goals on axis 1, away goals on axis 2
    pmat = p_h[:, :, None] * p_a[:, None, :]

    # Dixon-Coles low-score correction
    pmat[:, 0, 0] *= (1 - lh_batch * la_batch * rho)
    pmat[:, 0, 1] *= (1 + lh_batch * rho)
    pmat[:, 1, 0] *= (1 + la_batch * rho)
    pmat[:, 1, 1] *= (1 - rho)

    # Clip and normalise (correction can reduce total probability slightly)
    pmat = np.maximum(pmat, 0.0)
    total = pmat.sum(axis=(1, 2), keepdims=True)
    pmat /= np.maximum(total, 1e-12)

    # Boolean masks: home wins / draws / away wins
    j = np.arange(MAX_GOALS + 1)
    home_mask = j[:, None] > j[None, :]    # (MAX+1, MAX+1) home goals > away goals
    draw_mask = j[:, None] == j[None, :]
    away_mask = j[:, None] < j[None, :]

    ph  = pmat[:, home_mask].sum(axis=1)
    pd_ = pmat[:, draw_mask].sum(axis=1)
    pa  = pmat[:, away_mask].sum(axis=1)

    return np.stack([ph, pd_, pa], axis=1)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class DixonColes(BasePredictor):
    """
    Dixon-Coles bivariate Poisson model.

    After fitting:
      teams_      : list of team names (sorted)
      attack_     : ndarray (n_teams,)
      defense_    : ndarray (n_teams,)
      home_adv_   : float
      rho_        : float
      ko_deflation_ : float (from config, e.g. 0.85)
      converged_  : bool
      final_nll_  : float
    """

    def __init__(
        self,
        ko_deflation: float | None = None,
        l2_global: float | None = None,
        l2_conf: float | None = None,
    ) -> None:
        super().__init__()
        cfg = load_config()
        self.ko_deflation_: float = (
            ko_deflation
            if ko_deflation is not None
            else cfg.get('tournament', {}).get('knockout_goal_rate_deflation', 0.85)
        )
        mcfg = cfg.get('modeling', {})
        self.l2_global_: float = (
            l2_global if l2_global is not None
            else float(mcfg.get('dc_l2_global', L2_GLOBAL_DEFAULT))
        )
        self.l2_conf_: float = (
            l2_conf if l2_conf is not None
            else float(mcfg.get('dc_l2_conf', L2_CONF_DEFAULT))
        )
        self.teams_: list[str] = []
        self.team_idx_: dict[str, int] = {}
        self.attack_: np.ndarray = np.array([])
        self.defense_: np.ndarray = np.array([])
        self.home_adv_: float = 0.0
        self.rho_: float = 0.0
        self.converged_: bool = False
        self.final_nll_: float = float('inf')

    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, elo_ts: pd.DataFrame) -> 'DixonColes':
        self._elo_ts = elo_ts.sort_values(['team', 'date']).reset_index(drop=True)

        # Filter to matches with known scorelines
        train = df.dropna(subset=['home_score', 'away_score', 'training_weight']).copy()
        train['home_score'] = train['home_score'].astype(int)
        train['away_score'] = train['away_score'].astype(int)
        log.info("DixonColes training on %d matches", len(train))

        # Team roster
        all_teams = sorted(
            set(train['home_team'].unique()) | set(train['away_team'].unique())
        )
        self.teams_    = all_teams
        self.team_idx_ = {t: i for i, t in enumerate(all_teams)}
        N              = len(all_teams)

        # Confederation index per team (for confederation-mean shrinkage).
        # Unmapped / non-FIFA teams share an 'UNKNOWN' bucket.
        from src.data.constants import CONFEDERATION
        group_names = sorted(set(CONFEDERATION.get(t, 'UNKNOWN') for t in all_teams))
        group_id    = {g: i for i, g in enumerate(group_names)}
        conf_idx    = np.array([group_id[CONFEDERATION.get(t, 'UNKNOWN')] for t in all_teams])
        n_groups    = len(group_names)

        # Time-decay weight (reference = most recent training match)
        ref_date = train['date'].max()
        days_ago = (ref_date - train['date']).dt.days.values.astype(float)
        time_decay = np.exp(-DECAY_RATE * days_ago)

        # Combined weight
        weights = train['training_weight'].values * time_decay

        # Integer arrays for fast indexing
        home_idx  = train['home_team'].map(self.team_idx_).values
        away_idx  = train['away_team'].map(self.team_idx_).values
        goals_h   = train['home_score'].values.astype(int)
        goals_a   = train['away_score'].values.astype(int)
        is_neutral = train['neutral_venue'].values.astype(bool)

        # Initial parameters: all zeros + small home_adv + rho=0
        x0 = np.zeros(2 * N + 2)
        x0[2 * N]     = 0.3    # home_adv initial guess (log scale)
        x0[2 * N + 1] = 0.0    # rho

        bounds = (
            [(-5.0,  5.0)] * N        # attack
            + [(-5.0, 5.0)] * N       # defense
            + [(0.0,  2.0)]            # home_adv
            + [(-RHO_BOUND, RHO_BOUND)]  # rho
        )

        log.info(
            "Optimising %d parameters (L-BFGS-B)  l2_global=%.3f  l2_conf=%.3f  groups=%d …",
            2 * N + 2, self.l2_global_, self.l2_conf_, n_groups,
        )
        result = minimize(
            fun=_nll_and_grad,
            x0=x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            args=(home_idx, away_idx, goals_h, goals_a, is_neutral, weights, N,
                  conf_idx, n_groups, self.l2_global_, self.l2_conf_),
            options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-6},
        )

        self.attack_   = result.x[:N]
        self.defense_  = result.x[N:2 * N]
        self.home_adv_ = float(result.x[2 * N])
        self.rho_      = float(result.x[2 * N + 1])
        self.converged_ = bool(result.success)
        self.final_nll_ = float(result.fun)

        log.info(
            "DixonColes converged=%s  NLL=%.4f  home_adv=%.4f  rho=%.4f",
            self.converged_, self.final_nll_, self.home_adv_, self.rho_,
        )
        return self

    # ------------------------------------------------------------------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute [P(H), P(D), P(A)] for each row.

        Reads: home_team, away_team, neutral_venue, [stage] (defaults to 'group').
        Applies knockout deflation if stage == 'knockout'.
        """
        n = len(df)
        if n == 0:
            return np.empty((0, 3))

        # Look up attack/defense; unseen teams → 0 (league average)
        def _get_param(arr: np.ndarray, team: str) -> float:
            return float(arr[self.team_idx_[team]]) if team in self.team_idx_ else 0.0

        homes    = df['home_team'].values
        aways    = df['away_team'].values
        neutrals = df['neutral_venue'].values.astype(bool)
        stages   = df['stage'].values if 'stage' in df.columns else np.full(n, 'group')

        lh_list, la_list = [], []
        for i in range(n):
            ha = 0.0 if neutrals[i] else self.home_adv_
            lh_i = np.exp(_get_param(self.attack_,  homes[i])
                          + _get_param(self.defense_, aways[i])
                          + ha)
            la_i = np.exp(_get_param(self.attack_,  aways[i])
                          + _get_param(self.defense_, homes[i]))
            # Knockout goal-rate deflation
            if stages[i] == 'knockout':
                lh_i *= self.ko_deflation_
                la_i *= self.ko_deflation_
            lh_list.append(lh_i)
            la_list.append(la_i)

        lh_arr = np.array(lh_list)
        la_arr = np.array(la_list)

        return _outcomes_batch(lh_arr, la_arr, self.rho_)

    # ------------------------------------------------------------------

    def get_lambdas(
        self,
        home: str,
        away: str,
        is_neutral: bool = True,
        home_adv_override: float | None = None,
        stage: str = 'group',
    ) -> tuple[float, float]:
        """
        Return (lambda_home, lambda_away) expected goals.

        home_adv_override replaces self.home_adv_ when not None — used by
        WC2026Predictor to apply a separate host home-advantage parameter.
        """
        def _get(arr: np.ndarray, team: str) -> float:
            return float(arr[self.team_idx_[team]]) if team in self.team_idx_ else 0.0

        ha = 0.0 if is_neutral else (
            home_adv_override if home_adv_override is not None else self.home_adv_
        )
        lh = np.exp(_get(self.attack_, home) + _get(self.defense_, away) + ha)
        la = np.exp(_get(self.attack_, away) + _get(self.defense_, home))

        if stage == 'knockout':
            lh *= self.ko_deflation_
            la *= self.ko_deflation_

        return float(lh), float(la)

    def scoreline_pmf(
        self,
        home: str,
        away: str,
        is_neutral: bool = True,
        home_adv_override: float | None = None,
        stage: str = 'group',
    ) -> np.ndarray:
        """
        Return (MAX_GOALS+1, MAX_GOALS+1) probability matrix.

        pmat[i, j] = P(home scores i goals, away scores j goals).
        Dixon-Coles low-score correction applied; matrix sums to 1.
        """
        from scipy.stats import poisson as _poisson

        lh, la = self.get_lambdas(
            home, away, is_neutral=is_neutral,
            home_adv_override=home_adv_override, stage=stage,
        )
        scores = np.arange(MAX_GOALS + 1)
        ph = _poisson.pmf(scores, lh)   # (MAX+1,)
        pa = _poisson.pmf(scores, la)   # (MAX+1,)

        pmat = ph[:, None] * pa[None, :]   # (MAX+1, MAX+1): pmat[i,j] = P(h=i)*P(a=j)

        rho = self.rho_
        pmat[0, 0] *= (1 - lh * la * rho)
        pmat[0, 1] *= (1 + lh * rho)
        pmat[1, 0] *= (1 + la * rho)
        pmat[1, 1] *= (1 - rho)

        pmat = np.maximum(pmat, 0.0)
        pmat /= pmat.sum()
        return pmat

    def top_teams(self, n: int = 10) -> pd.DataFrame:
        """Return top-n teams by net strength (attack - defense mean)."""
        df = pd.DataFrame({
            'team':    self.teams_,
            'attack':  self.attack_,
            'defense': self.defense_,
        })
        df['strength'] = df['attack'] - df['defense'].mean() - df['defense']
        return df.sort_values('strength', ascending=False).head(n).reset_index(drop=True)

    def summary(self) -> str:
        lines = [
            f"DixonColes  converged={self.converged_}  NLL={self.final_nll_:.4f}",
            f"  home_adv={self.home_adv_:.4f}  rho={self.rho_:.4f}",
            f"  teams={len(self.teams_)}  decay_rate={DECAY_RATE}/day"
            f"  l2_global={self.l2_global_}  l2_conf={self.l2_conf_}",
            f"  ko_deflation={self.ko_deflation_}",
        ]
        return '\n'.join(lines)
