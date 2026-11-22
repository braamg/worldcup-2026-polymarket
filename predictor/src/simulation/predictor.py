"""
WC2026Predictor: clean match outcome predictor for World Cup 2026 matchups.

Wraps the dc_xgb_avg ensemble with:
  - Separate, configurable host home-advantage for USA / Mexico / Canada
    (host_home_advantage_elo and host_home_advantage_dc in config.yaml)
  - Cross-confederation uncertainty widening via entropy blend toward uniform
  - Optional temperature calibration (default: T from calibration_temperatures.json)
  - Scoreline distribution and top scorelines from Dixon-Coles

Usage
-----
predictor = WC2026Predictor(dc, xgb, elo_ts, calibration_T=1.04)
result = predictor.predict('USA', 'Germany', '2026-06-20', venue='home', stage='group')
print(result)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.data.constants import CONFEDERATION, TEAM_NAME_ALIASES
from src.features.market_value import get_mv_at_date
from src.models.base import ELO_HOME_ADV
from src.models.dixon_coles import DixonColes, MAX_GOALS
from src.models.xgboost_model import XGBoostPredictor
from src.ratings.elo import get_elo, DEFAULT_RATING
from src.utils import load_config


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class PredictionResult:
    """
    Match prediction from team_a's perspective.

    p_win / p_draw / p_loss sum to 1.
    scoreline_grid[i, j] = P(team_a scores i goals, team_b scores j goals).
    top_scorelines: list of (team_a_goals, team_b_goals, probability), ranked.
    """
    team_a: str
    team_b: str
    date: str
    venue: str            # 'home' | 'away' | 'neutral'
    stage: str

    p_win: float
    p_draw: float
    p_loss: float

    scoreline_grid: np.ndarray   # (MAX_GOALS+1, MAX_GOALS+1)
    top_scorelines: list         # [(a_goals, b_goals, prob), ...]

    is_cross_conf: bool
    conf_a: str
    conf_b: str
    entropy: float
    confidence_label: str        # 'High' | 'Medium' | 'Low'

    is_host_match: bool
    home_team: str | None        # None when neutral

    elo_a: float
    elo_b: float
    dc_lambda_a: float           # DC expected goals for team_a
    dc_lambda_b: float           # DC expected goals for team_b

    # ------------------------------------------------------------------

    def __str__(self) -> str:
        bar_w = 28

        def bar(p: float) -> str:
            filled = round(p * bar_w)
            return '█' * filled + '░' * (bar_w - filled)

        if self.venue == 'neutral':
            venue_tag = 'Neutral venue'
        elif self.is_host_match:
            venue_tag = f'HOME ({self.home_team} ★ HOST)'
        elif self.venue == 'home':
            venue_tag = f'Home ({self.home_team})'
        else:
            venue_tag = f'Away (at {self.home_team})'

        stage_tag = 'Group Stage' if self.stage == 'group' else 'Knockout'

        if self.is_cross_conf:
            conf_tag = f'{self.conf_a} vs {self.conf_b} — CROSS-CONFEDERATION  ⚡ uncertainty widened'
        else:
            conf_tag = f'{self.conf_a} (same confederation)'

        entropy_frac = self.entropy / np.log(3.0)

        sep = '═' * 65
        a = self.team_a
        b = self.team_b
        lines = [
            sep,
            f'  {a} vs {b}  │  {self.date}  │  {stage_tag}  │  {venue_tag}',
            f'  {conf_tag}',
            sep,
            f'  P({a:<15})  {self.p_win:>6.1%}  {bar(self.p_win)}',
            f'  P(Draw)            {self.p_draw:>6.1%}  {bar(self.p_draw)}',
            f'  P({b:<15})  {self.p_loss:>6.1%}  {bar(self.p_loss)}',
            '',
            f'  DC xG: {a} {self.dc_lambda_a:.2f}  ·  {b} {self.dc_lambda_b:.2f}',
        ]

        parts = [f'{ag}–{bg} ({p:.1%})' for ag, bg, p in self.top_scorelines]
        lines.append(f'  Most likely: {" | ".join(parts)}')
        lines.append('')
        lines.append(
            f'  Elo: {a} {self.elo_a:.0f}  ·  {b} {self.elo_b:.0f}'
            f'  │  Confidence: {self.confidence_label}'
            f' (entropy {entropy_frac:.0%} of max)'
        )
        lines.append(sep)
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main predictor
# ---------------------------------------------------------------------------

class WC2026Predictor:
    """
    Ensemble predictor for World Cup 2026 matchups.

    Combines Dixon-Coles and XGBoost (dc_xgb_avg) with:
      - A configurable host home-advantage separate from the regular-international
        home_advantage parameter (USA/Mexico/Canada playing in own stadiums).
      - Cross-confederation uncertainty widening: predictions for inter-confederation
        matchups are blended slightly toward uniform (1/3,1/3,1/3) to reflect
        the extra estimation uncertainty quantified in Phase 7 (cross-conf RPS 0.172
        vs same-conf 0.169).
      - Optional temperature calibration.

    Parameters
    ----------
    dc                       : fitted DixonColes model
    xgb                      : fitted XGBoostPredictor model
    elo_ts                   : Elo timeseries (team, date, elo_before, elo_after)
    calibration_T            : temperature scalar; 1.0 = no calibration
    host_home_advantage_elo  : Elo-point bonus for host nations at home
                               (overrides config tournament.host_home_advantage_elo)
    host_home_advantage_dc   : DC log-goal-space home_adv for host nations at home
                               (overrides config tournament.host_home_advantage_dc)
    cross_conf_entropy_blend : blend fraction toward uniform for cross-conf matches
                               (overrides config tournament.cross_conf_entropy_blend)
    """

    def __init__(
        self,
        dc: DixonColes,
        xgb: XGBoostPredictor,
        elo_ts: pd.DataFrame,
        calibration_T: float = 1.0,
        host_home_advantage_elo: float | None = None,
        host_home_advantage_dc: float | None = None,
        cross_conf_entropy_blend: float | None = None,
        squad_snapshots: pd.DataFrame | None = None,
    ) -> None:
        cfg = load_config()
        t = cfg.get('tournament', {})

        self.dc = dc
        self.xgb = xgb
        self.elo_ts = elo_ts.sort_values(['team', 'date']).reset_index(drop=True)
        self.calibration_T = calibration_T

        # Squad market values (as-of-date) so log_ratio_mv is LIVE for WC matches
        # instead of NaN. Auto-loads the default snapshot if not supplied. Most WC
        # teams have a Jan-2026 snapshot; the few missing fall back to NaN (XGB
        # handles natively, exactly as in training).
        if squad_snapshots is None:
            from pathlib import Path
            snap_path = Path(__file__).resolve().parents[2] / 'data' / 'interim' / 'squad_snapshots.parquet'
            try:
                squad_snapshots = pd.read_parquet(snap_path)
            except Exception:   # noqa: BLE001
                squad_snapshots = None
        if squad_snapshots is not None:
            squad_snapshots = squad_snapshots.sort_values(['team', 'valid_from']).reset_index(drop=True)
            squad_snapshots['valid_from'] = pd.to_datetime(squad_snapshots['valid_from'])
        self.squad_snapshots = squad_snapshots

        self.host_home_advantage_elo: float = (
            host_home_advantage_elo
            if host_home_advantage_elo is not None
            else float(t.get('host_home_advantage_elo', 100.0))
        )
        self.host_home_advantage_dc: float = (
            host_home_advantage_dc
            if host_home_advantage_dc is not None
            else float(t.get('host_home_advantage_dc', 0.23))
        )
        self.cross_conf_entropy_blend: float = (
            cross_conf_entropy_blend
            if cross_conf_entropy_blend is not None
            else float(t.get('cross_conf_entropy_blend', 0.0))   # 0.0: removed per phase10 backtest
        )
        self.hosts: set[str] = set(t.get('hosts', ['USA', 'Mexico', 'Canada']))

    # ------------------------------------------------------------------

    def _mv_log_ratio(self, home_team: str, away_team: str, date) -> tuple[float | None, bool]:
        """log(mv_home / mv_away) as-of-date; (None, False) if either team lacks a snapshot."""
        if self.squad_snapshots is None:
            return None, False
        mv_h = get_mv_at_date(home_team, pd.Timestamp(date), self.squad_snapshots)
        mv_a = get_mv_at_date(away_team, pd.Timestamp(date), self.squad_snapshots)
        if mv_h and mv_a and mv_h > 0 and mv_a > 0:
            return float(np.log(mv_h / mv_a)), True
        return None, False

    # ------------------------------------------------------------------

    def predict(
        self,
        team_a: str,
        team_b: str,
        date,
        venue: str = 'neutral',
        stage: str = 'group',
    ) -> PredictionResult:
        """
        Predict match outcome from team_a's perspective.

        Parameters
        ----------
        team_a, team_b : canonical team names (see src/data/constants.py)
        date           : match date — str 'YYYY-MM-DD' or pd.Timestamp
        venue          : 'neutral' — neither team designated as home
                         'home'    — team_a plays at home
                         'away'    — team_b plays at home, team_a is away
        stage          : 'group' | 'knockout'
                         knockout applies DC goal-rate deflation

        Returns
        -------
        PredictionResult with probs from team_a's perspective
        """
        # Normalize common name variants (e.g. "USA" → "United States")
        team_a = TEAM_NAME_ALIASES.get(team_a, team_a)
        team_b = TEAM_NAME_ALIASES.get(team_b, team_b)

        date = pd.Timestamp(date)
        is_neutral = (venue == 'neutral')

        # Orient home/away for model calls: models always see (home, away)
        if venue == 'away':
            home_team, away_team = team_b, team_a
        else:
            home_team, away_team = team_a, team_b

        # Host flag — only applies when a host nation plays in their own stadium
        is_host_home = (not is_neutral) and (home_team in self.hosts)

        # Elo ratings as-of-date (strictly before match)
        elo_home = get_elo(home_team, date, self.elo_ts)
        elo_away = get_elo(away_team, date, self.elo_ts)

        # Effective Elo home advantage for feature row
        if is_neutral:
            elo_adv = 0.0
        elif is_host_home:
            elo_adv = self.host_home_advantage_elo
        else:
            elo_adv = ELO_HOME_ADV   # standard 125-pt home advantage for non-host home games

        elo_diff = elo_home - elo_away + elo_adv
        e_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        # Squad market-value log-ratio, as-of-date (None → NaN, XGB handles it)
        log_ratio_mv, has_mv = self._mv_log_ratio(home_team, away_team, date)

        # Confederation membership
        conf_a = CONFEDERATION.get(team_a, 'UNKNOWN')
        conf_b = CONFEDERATION.get(team_b, 'UNKNOWN')
        is_cross_conf = (conf_a != conf_b) and ('UNKNOWN' not in (conf_a, conf_b))

        # Feature row for XGBoost — built manually so cross_conf is correct
        row = {
            'home_team':             home_team,
            'away_team':             away_team,
            'date':                  date,
            'neutral_venue':         is_neutral,
            'stage':                 stage,
            'elo_home_pre':          elo_home,
            'elo_away_pre':          elo_away,
            'elo_diff':              elo_diff,
            'e_home':                e_home,
            'log_ratio_mv':          log_ratio_mv,
            'has_market_value':      has_mv,
            'cross_conf':            is_cross_conf,
            'rest_days_home':        30.0,
            'rest_days_away':        30.0,
            'rest_days_home_missing': True,
            'rest_days_away_missing': True,
        }
        df_row = pd.DataFrame([row])

        # DC prediction via scoreline PMF
        dc_home_adv_override = self.host_home_advantage_dc if is_host_home else None
        dc_pmat = self.dc.scoreline_pmf(
            home_team, away_team,
            is_neutral=is_neutral,
            home_adv_override=dc_home_adv_override,
            stage=stage,
        )

        # W/D/L from the PMF (home_team perspective)
        idx = np.arange(MAX_GOALS + 1)
        home_mask = idx[:, None] > idx[None, :]
        draw_mask = idx[:, None] == idx[None, :]
        away_mask = idx[:, None] < idx[None, :]
        dc_proba = np.array([
            dc_pmat[home_mask].sum(),
            dc_pmat[draw_mask].sum(),
            dc_pmat[away_mask].sum(),
        ])

        # XGBoost prediction
        xgb_proba = self.xgb.predict_proba(df_row)[0]   # [P(H), P(D), P(A)]

        # Ensemble average
        ensemble = 0.5 * (dc_proba + xgb_proba)

        # Temperature calibration
        if self.calibration_T != 1.0:
            ensemble = softmax(np.log(np.clip(ensemble, 1e-15, 1.0)) / self.calibration_T)

        # Cross-confederation uncertainty widening
        if is_cross_conf and self.cross_conf_entropy_blend > 0.0:
            uniform = np.full(3, 1.0 / 3)
            alpha = self.cross_conf_entropy_blend
            ensemble = (1.0 - alpha) * ensemble + alpha * uniform

        # Flip to team_a perspective when team_b was the model's "home"
        if venue == 'away':
            # P(H)=P(b wins), P(A)=P(a wins) in the model call → flip
            p_a_win, p_draw, p_b_win = ensemble[2], ensemble[1], ensemble[0]
            dc_pmat_a = dc_pmat.T    # pmat_a[i,j] = P(a=i goals, b=j goals)
        else:
            p_a_win, p_draw, p_b_win = ensemble[0], ensemble[1], ensemble[2]
            dc_pmat_a = dc_pmat

        # Top 5 scorelines from team_a's perspective
        flat_idx = np.argsort(dc_pmat_a.ravel())[::-1][:5]
        top_scorelines = []
        for fi in flat_idx:
            i, j = np.unravel_index(fi, dc_pmat_a.shape)
            top_scorelines.append((int(i), int(j), float(dc_pmat_a[i, j])))

        # DC expected goals (team_a perspective)
        lh, la = self.dc.get_lambdas(
            home_team, away_team,
            is_neutral=is_neutral,
            home_adv_override=dc_home_adv_override,
            stage=stage,
        )
        if venue == 'away':
            dc_lambda_a, dc_lambda_b = la, lh
        else:
            dc_lambda_a, dc_lambda_b = lh, la

        # Elo ratings (team_a perspective)
        elo_a = elo_away if venue == 'away' else elo_home
        elo_b = elo_home if venue == 'away' else elo_away

        # Entropy-based confidence
        p_safe = np.clip(np.array([p_a_win, p_draw, p_b_win]), 1e-15, 1.0)
        entropy = float(-np.sum(p_safe * np.log(p_safe)))
        entropy_frac = entropy / np.log(3.0)
        if entropy_frac < 0.85:
            confidence_label = 'High'
        elif entropy_frac < 0.95:
            confidence_label = 'Medium'
        else:
            confidence_label = 'Low'

        return PredictionResult(
            team_a=team_a,
            team_b=team_b,
            date=str(date.date()),
            venue=venue,
            stage=stage,
            p_win=float(p_a_win),
            p_draw=float(p_draw),
            p_loss=float(p_b_win),
            scoreline_grid=dc_pmat_a,
            top_scorelines=top_scorelines,
            is_cross_conf=is_cross_conf,
            conf_a=conf_a,
            conf_b=conf_b,
            entropy=entropy,
            confidence_label=confidence_label,
            is_host_match=is_host_home,
            home_team=home_team if not is_neutral else None,
            elo_a=elo_a,
            elo_b=elo_b,
            dc_lambda_a=float(dc_lambda_a),
            dc_lambda_b=float(dc_lambda_b),
        )
