"""
WC2026 Monte Carlo Tournament Simulator.

Conditions on confirmed 2026 World Cup results (data/raw/wc2026_results.csv)
and simulates remaining matches using the dc_xgb_avg ensemble predictor.

Design notes:
- Group stage: DC PMFs pre-computed once per unplayed fixture; XGB re-run per
  simulation with jittered Elo features for parameter uncertainty propagation.
- Knockout stage: full predictor called per match with LRU-style dict cache
  (keyed on matchup without jitter) plus analytical Elo-jitter correction.
- Shootouts resolved ~50/50 (per CLAUDE.md: NEVER by team strength).
- Third-place R32 allocation via bipartite matching (scipy.linear_sum_assignment).

Update wc2026_results.csv after each matchday and re-run to refresh predictions.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax

from src.data.constants import CONFEDERATION
from src.models.base import ELO_HOME_ADV
from src.models.xgboost_model import FEATURES as XGB_FEATURES
from src.ratings.elo import DEFAULT_RATING
from src.utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Tournament structure
# ---------------------------------------------------------------------------

GROUPS: Dict[str, List[str]] = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czechia'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

ALL_WC_TEAMS: List[str] = [t for teams in GROUPS.values() for t in teams]
TEAM_TO_GROUP: Dict[str, str] = {t: g for g, ts in GROUPS.items() for t in ts}
HOST_TEAMS: frozenset = frozenset(['United States', 'Mexico', 'Canada'])

# ---------------------------------------------------------------------------
# Knockout venue → stadium country.
# A host nation gets home advantage in a knockout match ONLY when that match is
# actually played in its own country. The 2026 knockout venues are fixed by
# bracket slot (FIFA's published schedule), so they are knowable in advance.
#
# The full official 2026 schedule is in data/raw/wc2026_ko_venues.csv
# (match_num,country), which OVERRIDES these defaults. Only 3 R32 + 3 R16 matches
# are outside the USA: Mexico hosts 75 (Monterrey), 79 (Mexico City), 92 (Mexico
# City); Canada hosts 83 (Toronto), 85 (Vancouver), 96 (Vancouver). All QF/SF/Final
# are in the USA. A host gets home advantage ONLY when it actually plays in its own
# country's stadium — replacing the old "host plays every knockout at home", which
# compounded a 100-Elo bonus through up to 7 rounds and overrated the hosts (Mexico
# was at 3.45x the bookmaker title price). The defaults below are a safe fallback if
# the CSV is missing; any match still unlisted is treated as NEUTRAL.
FINAL_MATCH_NUM: int = 104
KO_VENUE_COUNTRY_DEFAULT: Dict[int, str] = {
    101: 'United States',   # Semifinal — Dallas
    102: 'United States',   # Semifinal — Atlanta
    104: 'United States',   # Final     — New Jersey (MetLife)
}

# ---------------------------------------------------------------------------
# R32 bracket constants
# ---------------------------------------------------------------------------

# Fixed R32 slots: match_num -> (slot_type_a, group_a, slot_type_b, group_b)
# slot_type: 'winner' | 'runner_up'
R32_FIXED: Dict[int, Tuple] = {
    73: ('runner_up', 'A', 'runner_up', 'B'),
    75: ('winner',    'F', 'runner_up', 'C'),
    76: ('winner',    'C', 'runner_up', 'F'),
    78: ('runner_up', 'E', 'runner_up', 'I'),
    83: ('runner_up', 'K', 'runner_up', 'L'),
    84: ('winner',    'H', 'runner_up', 'J'),
    86: ('winner',    'J', 'runner_up', 'H'),
    88: ('runner_up', 'D', 'runner_up', 'G'),
}

# Third-place R32 slots: match_num -> (group_winner, compatible_groups_for_3rd)
R32_THIRD: Dict[int, Tuple] = {
    74: ('E', frozenset('ABCDF')),
    77: ('I', frozenset('CDFGH')),
    79: ('A', frozenset('CEFHI')),
    80: ('L', frozenset('EHIJK')),
    81: ('D', frozenset('BEFIJ')),
    82: ('G', frozenset('AEHIJ')),
    85: ('B', frozenset('EFGIJ')),
    87: ('K', frozenset('DEIJL')),
}

# Knockout bracket: match_num -> (feed_match_a, feed_match_b)
R16_BRACKET: Dict[int, Tuple[int, int]] = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),
}
QF_BRACKET: Dict[int, Tuple[int, int]] = {
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),
}
SF_BRACKET: Dict[int, Tuple[int, int]] = {
    101: (97, 98), 102: (99, 100),
}
# R32 match numbers in order
ALL_R32 = sorted(list(R32_FIXED.keys()) + list(R32_THIRD.keys()))

STAGE_EXIT_LABELS = {
    'r32': 'r32_exit', 'r16': 'r16_exit', 'qf': 'qf_exit',
    'sf': 'sf_exit', 'final': 'final_loss', 'champion': 'champion',
}

# ---------------------------------------------------------------------------
# Group standings helpers
# ---------------------------------------------------------------------------

def _pts(gf: int, ga: int) -> int:
    if gf > ga: return 3
    if gf == ga: return 1
    return 0


def _standing_key(rec: dict) -> tuple:
    return (rec['pts'], rec['gd'], rec['gf'])


def _h2h_key(team: str, tied_teams: List[str], match_results: List[dict]) -> tuple:
    """Head-to-head points/GD/GF among a subset of tied teams."""
    pts = gd = gf = 0
    for r in match_results:
        a, b = r['team_a'], r['team_b']
        if a not in tied_teams or b not in tied_teams:
            continue
        if a == team:
            p = _pts(r['gf_a'], r['gf_b'])
            pts += p; gd += r['gf_a'] - r['gf_b']; gf += r['gf_a']
        elif b == team:
            p = _pts(r['gf_b'], r['gf_a'])
            pts += p; gd += r['gf_b'] - r['gf_a']; gf += r['gf_b']
    return (pts, gd, gf)


def rank_group(teams: List[str], match_results: List[dict]) -> List[str]:
    """
    Rank 4 group teams by FIFA tiebreaker chain (pts, GD, GF, H2H, random).
    match_results: list of dicts with team_a, team_b, gf_a, gf_b keys.
    Returns list from 1st to 4th place.
    """
    rec: Dict[str, dict] = {t: {'pts': 0, 'gd': 0, 'gf': 0} for t in teams}
    for r in match_results:
        a, b, ga, gb = r['team_a'], r['team_b'], r['gf_a'], r['gf_b']
        rec[a]['pts'] += _pts(ga, gb)
        rec[a]['gd']  += ga - gb
        rec[a]['gf']  += ga
        rec[b]['pts'] += _pts(gb, ga)
        rec[b]['gd']  += gb - ga
        rec[b]['gf']  += gb

    sorted_teams = sorted(teams, key=lambda t: _standing_key(rec[t]), reverse=True)

    # Re-rank tied clusters by H2H then random
    final: List[str] = []
    i = 0
    while i < len(sorted_teams):
        j = i + 1
        while j < len(sorted_teams) and (
            _standing_key(rec[sorted_teams[i]]) == _standing_key(rec[sorted_teams[j]])
        ):
            j += 1
        tied = sorted_teams[i:j]
        if len(tied) > 1:
            h2h_results = [r for r in match_results
                           if r['team_a'] in tied and r['team_b'] in tied]
            tied = sorted(tied, key=lambda t: _h2h_key(t, tied, h2h_results), reverse=True)
        final.extend(tied)
        i = j
    return final


def rank_third_place_teams(thirds: List[dict]) -> List[dict]:
    """
    Rank third-place teams across all groups by FIFA criteria:
    pts → GD → GF → (random for remaining ties).
    Each dict: {team, group, pts, gd, gf}
    Returns sorted list best-first.
    """
    return sorted(thirds, key=lambda r: (r['pts'], r['gd'], r['gf']), reverse=True)


# ---------------------------------------------------------------------------
# Third-place R32 allocation
# ---------------------------------------------------------------------------

THIRD_PLACE_SLOTS: Dict[int, frozenset] = {
    match: compat for match, (_, compat) in R32_THIRD.items()
}


def allocate_third_place(qualifying_groups: List[str]) -> Dict[str, int]:
    """
    Assign the 8 qualifying third-place group letters to R32 match slots.
    Uses bipartite matching to find a valid assignment.
    Returns {group_letter: r32_match_number}.

    Each slot has a pre-defined set of compatible groups; the assignment must
    give each slot exactly one team from its compatible set. With 495 valid
    group combinations (C(12,8)), a unique perfect matching always exists.
    """
    slots = sorted(THIRD_PLACE_SLOTS.keys())   # [74, 77, 79, 80, 81, 82, 85, 87]
    n = len(qualifying_groups)
    if n != 8:
        raise ValueError(f"Expected 8 qualifying groups, got {n}: {qualifying_groups}")

    INF = 1_000_000.0
    cost = np.full((n, n), INF)
    for i, grp in enumerate(qualifying_groups):
        for j, slot in enumerate(slots):
            if grp in THIRD_PLACE_SLOTS[slot]:
                cost[i, j] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)
    total_cost = cost[row_ind, col_ind].sum()
    if total_cost > 0:
        raise ValueError(
            f"No valid R32 third-place allocation for groups {qualifying_groups}. "
            "This combination should not arise in the 2026 WC format."
        )

    return {qualifying_groups[i]: slots[col_ind[i]] for i in range(n)}


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

class WC2026Simulator:
    """
    Monte Carlo simulator for the 2026 FIFA World Cup.

    Confirmed results are treated as facts. Remaining matches are simulated
    using the dc_xgb_avg ensemble with per-simulation Elo jitter to propagate
    parameter uncertainty (wider for cross-confederation teams per Phase 7).

    Parameters
    ----------
    predictor         : WC2026Predictor (Phase 8)
    fixtures_df       : 72-row DataFrame from wc2026_fixtures.csv
    results_df        : confirmed results from wc2026_results.csv
    n_simulations     : Monte Carlo draws
    elo_sigma_base    : 1-sigma Elo uncertainty in points (default 50)
    elo_sigma_cross   : multiplier for cross-confederation teams (default 1.3)
    seed              : RNG seed for reproducibility
    """

    def __init__(
        self,
        predictor,
        fixtures_df: pd.DataFrame,
        results_df: pd.DataFrame,
        n_simulations: int = 10_000,
        elo_sigma_base: float = 50.0,
        elo_sigma_cross: float = 1.3,
        seed: int = 42,
    ) -> None:
        self.predictor = predictor
        self.n_simulations = n_simulations
        self.elo_sigma_base = elo_sigma_base
        self.elo_sigma_cross = elo_sigma_cross
        self.rng = np.random.default_rng(seed)

        # ------------------------------------------------------------------
        # Build fixture table with played/score info
        # ------------------------------------------------------------------
        fix = fixtures_df.copy()
        fix['date'] = pd.to_datetime(fix['date'])
        fix['host_team'] = fix['host_team'].fillna('').astype(str)

        res = results_df.copy()
        res['date'] = pd.to_datetime(res['date'])

        played_keys = set(zip(res['group'], res['team_a'], res['team_b']))
        fix['played'] = fix.apply(
            lambda r: (r['group'], r['team_a'], r['team_b']) in played_keys, axis=1
        )
        fix = fix.merge(
            res.rename(columns={'goals_a': '_ga', 'goals_b': '_gb'})[
                ['group', 'team_a', 'team_b', '_ga', '_gb']
            ],
            on=['group', 'team_a', 'team_b'], how='left',
        )
        self.fixtures = fix

        # ------------------------------------------------------------------
        # Pre-compute current Elo ratings for all WC teams
        # ------------------------------------------------------------------
        self._base_elos = self._load_elos()

        # ------------------------------------------------------------------
        # Knockout venue table (stadium country per KO match number)
        # ------------------------------------------------------------------
        self._ko_venue_country = self._load_ko_venues()

        # ------------------------------------------------------------------
        # Per-team Elo jitter sigma — scaled by how DATA-POOR each team is.
        # A team's Elo is more uncertain the fewer recent internationals it has
        # played: rating SE scales ~1/sqrt(n). We count matches in the 4 years
        # before the tournament and set sigma = base * clip(sqrt(n_ref/n), lo, hi),
        # where n_ref is the median recent-match count across the 48 WC teams.
        # This subsumes (and is more principled than) the old flat 1.3x multiplier
        # for non-UEFA/CONMEBOL teams: minnows with few matches naturally get a
        # wider sigma, well-covered European/South-American sides a tighter one.
        # ------------------------------------------------------------------
        recent = self._recent_match_counts(years=4)
        counts = np.array([recent.get(t, 0) for t in ALL_WC_TEAMS], dtype=float)
        n_ref  = max(np.median(counts[counts > 0]) if (counts > 0).any() else 1.0, 1.0)
        scale  = np.sqrt(n_ref / np.clip(counts, 3.0, None))
        scale  = np.clip(scale, 0.7, 2.0)
        self._jitter_sigmas = elo_sigma_base * scale
        self._recent_counts = recent

        # ------------------------------------------------------------------
        # Pre-compute DC PMFs + DC W/D/L probabilities for unplayed fixtures
        # (DC parameters don't change with Elo jitter → compute once)
        # ------------------------------------------------------------------
        self._dc_pmfs: Dict[tuple, np.ndarray] = {}
        self._dc_proba: Dict[tuple, np.ndarray] = {}
        self._precompute_group_dc()

        # Cache for knockout predictions (keyed without jitter; jitter applied analytically)
        self._ko_cache: Dict[tuple, object] = {}

        # Vectorised batch arrays for group-stage XGB predictions (built once)
        self._precompute_group_batch_meta()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _load_elos(self) -> Dict[str, float]:
        from src.ratings.elo import get_elo
        # Use a date after the final so we always return the most up-to-date
        # Elo for each team (the caller should have already extended elo_ts with
        # confirmed WC results via extend_elo_with_wc_results before building
        # the simulator).
        as_of = pd.Timestamp('2026-07-20')
        return {team: get_elo(team, as_of, self.predictor.elo_ts)
                for team in ALL_WC_TEAMS}

    def _load_ko_venues(self) -> Dict[int, str]:
        """
        Knockout venue countries: defaults + optional override CSV
        (data/raw/wc2026_ko_venues.csv, columns: match_num,country).
        Returns {match_num: country}; missing matches are treated as neutral.
        """
        from pathlib import Path
        venues = dict(KO_VENUE_COUNTRY_DEFAULT)
        csv_path = Path(__file__).resolve().parents[2] / 'data' / 'raw' / 'wc2026_ko_venues.csv'
        if csv_path.exists():
            try:
                vdf = pd.read_csv(csv_path)
                for _, r in vdf.iterrows():
                    country = str(r['country']).strip()
                    if country and country.lower() not in ('neutral', 'nan', ''):
                        venues[int(r['match_num'])] = country
            except Exception as exc:   # noqa: BLE001
                log.warning("Could not read KO venue CSV (%s); using defaults", exc)
        return venues

    def _recent_match_counts(self, years: int = 4) -> Dict[str, int]:
        """Count each WC team's matches in the `years` before the tournament start."""
        ts = self.predictor.elo_ts
        cutoff_hi = pd.Timestamp('2026-06-11')
        cutoff_lo = cutoff_hi - pd.DateOffset(years=years)
        recent = ts[(ts['date'] >= cutoff_lo) & (ts['date'] < cutoff_hi)]
        vc = recent['team'].value_counts()
        return {t: int(vc.get(t, 0)) for t in ALL_WC_TEAMS}

    def _precompute_group_dc(self) -> None:
        """Pre-compute DC scoreline PMFs for all unplayed group fixtures (done once)."""
        unplayed = self.fixtures[~self.fixtures['played']]
        idx = np.arange(11)
        hw = idx[:, None] > idx[None, :]
        dw = idx[:, None] == idx[None, :]
        aw = idx[:, None] < idx[None, :]

        for _, row in unplayed.iterrows():
            venue = self._row_venue(row)
            key = (str(row.team_a), str(row.team_b), venue)
            if key in self._dc_pmfs:
                continue
            result = self.predictor.predict(
                str(row.team_a), str(row.team_b),
                row.date, venue=venue, stage='group',
            )
            pmat = result.scoreline_grid
            self._dc_pmfs[key]  = pmat
            self._dc_proba[key] = np.array([
                pmat[hw].sum(), pmat[dw].sum(), pmat[aw].sum()
            ])
        log.debug("Pre-computed DC PMFs for %d unplayed group fixtures", len(self._dc_pmfs))

    @staticmethod
    def _row_venue(row) -> str:
        ht = str(row.host_team).strip()
        if not ht:
            return 'neutral'
        return 'home' if ht == str(row.team_a).strip() else 'away'

    # ------------------------------------------------------------------
    # Elo jitter
    # ------------------------------------------------------------------

    def _sample_jitter(self) -> Dict[str, float]:
        """Draw per-team Elo perturbation for one simulation."""
        raw = self.rng.normal(0.0, 1.0, len(ALL_WC_TEAMS)) * self._jitter_sigmas
        return {team: float(raw[i]) for i, team in enumerate(ALL_WC_TEAMS)}

    def _jelo(self, team: str, jitter: Dict[str, float]) -> float:
        return self._base_elos.get(team, DEFAULT_RATING) + jitter.get(team, 0.0)

    # ------------------------------------------------------------------
    # Vectorised batch metadata for group-stage (built once at init)
    # ------------------------------------------------------------------

    def _precompute_group_batch_meta(self) -> None:
        """
        Pre-compute all static arrays needed for fast per-simulation group-stage
        predictions. Called once in __init__.

        Per-simulation cost: one batch XGB call on all 67 unplayed fixtures (one
        predict_proba call, not 67 individual calls), plus vectorised goal sampling
        using pre-computed conditional CDFs.
        """
        unplayed = self.fixtures[~self.fixtures['played']]

        home_teams, away_teams = [], []
        elo_advs, is_cc_list, is_neutral_list, is_away_list = [], [], [], []
        dc_proba_list, dc_pmf_list = [], []
        mv_list, has_mv_list = [], []          # log_ratio_mv per fixture (static)
        group_keys = []   # (group, team_a, team_b) for lookup by _simulate_one

        for _, row in unplayed.iterrows():
            venue = self._row_venue(row)
            ta, tb = str(row.team_a), str(row.team_b)
            is_neutral = (venue == 'neutral')
            is_away    = (venue == 'away')
            home_t = tb if is_away else ta
            away_t = ta if is_away else tb

            # Squad market-value log-ratio (home perspective), as-of the fixture date
            lr_mv, has_mv = self.predictor._mv_log_ratio(home_t, away_t, row.date)
            mv_list.append(np.nan if lr_mv is None else lr_mv)
            has_mv_list.append(1.0 if has_mv else 0.0)

            is_host = (not is_neutral) and (home_t in HOST_TEAMS)
            if is_neutral:
                elo_adv = 0.0
            elif is_host:
                elo_adv = self.predictor.host_home_advantage_elo
            else:
                elo_adv = ELO_HOME_ADV

            conf_a = CONFEDERATION.get(ta, 'UNKNOWN')
            conf_b = CONFEDERATION.get(tb, 'UNKNOWN')
            is_cc  = (conf_a != conf_b) and 'UNKNOWN' not in (conf_a, conf_b)

            key = (ta, tb, venue)
            home_teams.append(home_t)
            away_teams.append(away_t)
            elo_advs.append(elo_adv)
            is_cc_list.append(is_cc)
            is_neutral_list.append(is_neutral)
            is_away_list.append(is_away)
            dc_proba_list.append(self._dc_proba[key])   # (3,) from team_a perspective
            dc_pmf_list.append(self._dc_pmfs[key])
            group_keys.append((str(row['group']), ta, tb))

        n = len(group_keys)

        # ----------------------------------------------------------------
        # Team index arrays for vectorised jitter lookup
        # ----------------------------------------------------------------
        all_teams = list(self._base_elos.keys())
        t2i = {t: i for i, t in enumerate(all_teams)}
        self._gf_teams     = all_teams
        self._gf_elo_base  = np.array([self._base_elos[t] for t in all_teams])
        self._gf_home_idx  = np.array([t2i.get(t, 0) for t in home_teams])
        self._gf_away_idx  = np.array([t2i.get(t, 0) for t in away_teams])
        self._gf_elo_advs  = np.array(elo_advs)
        self._gf_is_cc     = np.array(is_cc_list, dtype=bool)
        self._gf_is_away   = np.array(is_away_list, dtype=bool)
        self._gf_dc_proba  = np.array(dc_proba_list)   # (n, 3) team_a perspective
        self._gf_dc_pmfs   = dc_pmf_list
        self._gf_keys      = group_keys
        self._gf_n         = n

        # ----------------------------------------------------------------
        # XGB feature matrix template (constant columns; elo_diff and e_home
        # filled fresh each simulation with jittered Elo values).
        # ----------------------------------------------------------------
        eidx = XGB_FEATURES.index('elo_diff')
        hidx = XGB_FEATURES.index('e_home')
        feat_tmpl = np.zeros((n, len(XGB_FEATURES)), dtype=np.float64)
        if n > 0:
            feat_tmpl[:, XGB_FEATURES.index('log_ratio_mv')]          = np.array(mv_list,     dtype=float)
            feat_tmpl[:, XGB_FEATURES.index('has_market_value')]       = np.array(has_mv_list, dtype=float)
            feat_tmpl[:, XGB_FEATURES.index('neutral_venue')]          = np.array(is_neutral_list, dtype=float)
            feat_tmpl[:, XGB_FEATURES.index('cross_conf')]             = np.array(is_cc_list,      dtype=float)
            feat_tmpl[:, XGB_FEATURES.index('rest_days_home')]         = 30.0
            feat_tmpl[:, XGB_FEATURES.index('rest_days_away')]         = 30.0
            feat_tmpl[:, XGB_FEATURES.index('rest_days_home_missing')] = 1.0
            feat_tmpl[:, XGB_FEATURES.index('rest_days_away_missing')] = 1.0

        self._gf_base_feat = feat_tmpl   # (n, 10) template; elo cols overwritten per sim
        self._gf_eidx      = eidx
        self._gf_hidx      = hidx

        if n > 0:

            # Pre-compute conditional CDFs for vectorised goal sampling
            # _gf_dc_cond_cdf[i, o, :] = CDF over 121 scorelines for fixture i, outcome o
            idx11 = np.arange(11)
            hw = idx11[:, None] > idx11[None, :]
            dw = idx11[:, None] == idx11[None, :]
            aw = idx11[:, None] < idx11[None, :]
            masks = (hw, dw, aw)

            cond_cdf = np.zeros((n, 3, 121), dtype=np.float64)
            for i, pmf in enumerate(dc_pmf_list):
                for o, mask in enumerate(masks):
                    cond = pmf * mask
                    total = cond.sum()
                    if total < 1e-15:
                        cond = mask.astype(np.float64) / max(mask.sum(), 1)
                    else:
                        cond = cond / total
                    cond_cdf[i, o] = np.cumsum(cond.ravel())
            self._gf_dc_cond_cdf = cond_cdf   # (n, 3, 121)
        else:
            self._gf_dc_cond_cdf = np.empty((0, 3, 121))

        # ----------------------------------------------------------------
        # Pre-parsed fixture lists per group (avoid DataFrame ops in hot loop)
        # (group -> list of (ta, tb, played, ga, gb, unplayed_idx))
        # ----------------------------------------------------------------
        key_to_idx = {k: i for i, k in enumerate(group_keys)}
        self._group_fixture_lists: Dict[str, list] = {}
        for grp in GROUPS:
            grp_fix = self.fixtures[self.fixtures['group'] == grp]
            flist = []
            for _, row in grp_fix.iterrows():
                ta, tb = str(row.team_a), str(row.team_b)
                if row.played:
                    flist.append((ta, tb, True, int(row._ga), int(row._gb), -1))
                else:
                    venue = self._row_venue(row)
                    unpl_key = (grp, ta, tb)
                    flist.append((ta, tb, False, 0, 0, key_to_idx[unpl_key]))
            self._group_fixture_lists[grp] = flist

        log.debug("Group batch meta: %d unplayed, base ensemble + CDFs computed", n)

    def _batch_predict_group(self, jitter: Dict[str, float]) -> np.ndarray:
        """
        Compute full ensemble predictions for all unplayed group fixtures.
        Uses ONE batch XGBoost call per simulation (67-row matrix) with jittered
        Elo features, then averages with pre-computed DC probabilities.
        Returns (n_unplayed, 3) array from team_a perspective.
        """
        if self._gf_n == 0:
            return np.empty((0, 3))

        # Vectorised jittered Elo → elo_diff and e_home for each fixture
        jitter_arr = np.empty(len(self._gf_teams))
        for i, t in enumerate(self._gf_teams):
            jitter_arr[i] = jitter.get(t, 0.0)

        jelo      = self._gf_elo_base + jitter_arr
        elo_home  = jelo[self._gf_home_idx]                              # (n,)
        elo_away  = jelo[self._gf_away_idx]                              # (n,)
        elo_diff  = elo_home - elo_away + self._gf_elo_advs              # (n,)
        e_home    = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))           # (n,)

        # Single batch XGB call with jittered features
        feat = self._gf_base_feat.copy()
        feat[:, self._gf_eidx] = elo_diff
        feat[:, self._gf_hidx] = e_home
        xgb_all = self.predictor.xgb.clf_.predict_proba(feat)           # (n, 3) home-team persp.

        # Flip venue='away' rows from home_t perspective to team_a perspective
        xgb_ta = xgb_all.copy()
        if self._gf_is_away.any():
            xgb_ta[self._gf_is_away] = xgb_all[self._gf_is_away][:, [2, 1, 0]]

        # Average with DC (both now in team_a perspective)
        ensemble = 0.5 * (self._gf_dc_proba + xgb_ta)                  # (n, 3)

        # Temperature calibration
        T = self.predictor.calibration_T
        if T != 1.0:
            log_e = np.log(np.clip(ensemble, 1e-15, 1.0)) / T
            log_e -= log_e.max(axis=1, keepdims=True)
            exp_e  = np.exp(log_e)
            ensemble = exp_e / exp_e.sum(axis=1, keepdims=True)

        # Cross-conf entropy blend (widen uncertainty for inter-confederation matchups)
        alpha = self.predictor.cross_conf_entropy_blend
        if alpha > 0 and self._gf_is_cc.any():
            ensemble = np.where(
                self._gf_is_cc[:, None],
                (1 - alpha) * ensemble + alpha / 3.0,
                ensemble,
            )

        return ensemble   # (n, 3) from team_a perspective

    # ------------------------------------------------------------------
    # Group-stage match prediction (scalar; kept for ad-hoc use)
    # ------------------------------------------------------------------

    def _predict_group(
        self, team_a: str, team_b: str, venue: str, jitter: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (ensemble_proba, dc_pmf) for a group-stage match.
        ensemble_proba is [P(team_a wins), P(draw), P(team_b wins)].
        XGB is re-run with jittered Elo features; DC PMF is pre-computed.
        """
        key = (team_a, team_b, venue)
        dc_proba = self._dc_proba[key]
        dc_pmf   = self._dc_pmfs[key]

        is_neutral = (venue == 'neutral')
        home_t = team_a if venue != 'away' else team_b
        away_t = team_b if venue != 'away' else team_a

        is_host = (not is_neutral) and (home_t in HOST_TEAMS)
        elo_home = self._jelo(home_t, jitter)
        elo_away = self._jelo(away_t, jitter)

        if is_neutral:
            elo_adv = 0.0
        elif is_host:
            elo_adv = self.predictor.host_home_advantage_elo
        else:
            elo_adv = ELO_HOME_ADV

        elo_diff = elo_home - elo_away + elo_adv
        e_home   = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        conf_a = CONFEDERATION.get(team_a, 'UNKNOWN')
        conf_b = CONFEDERATION.get(team_b, 'UNKNOWN')
        is_cc  = (conf_a != conf_b) and 'UNKNOWN' not in (conf_a, conf_b)

        row = pd.DataFrame([{
            'elo_diff': elo_diff, 'e_home': e_home,
            'log_ratio_mv': None, 'has_market_value': False,
            'neutral_venue': is_neutral, 'cross_conf': is_cc,
            'rest_days_home': 30.0, 'rest_days_away': 30.0,
            'rest_days_home_missing': True, 'rest_days_away_missing': True,
        }])
        xgb_proba = self.predictor.xgb.predict_proba(row)[0]

        # Ensemble, calibration, cross-conf blend — from team_a perspective
        ensemble = 0.5 * (dc_proba + xgb_proba)
        T = self.predictor.calibration_T
        if T != 1.0:
            ensemble = softmax(np.log(np.clip(ensemble, 1e-15, 1.0)) / T)
        if is_cc:
            alpha = self.predictor.cross_conf_entropy_blend
            ensemble = (1 - alpha) * ensemble + alpha * np.full(3, 1/3)

        if venue == 'away':
            ensemble = ensemble[[2, 1, 0]]  # flip to team_a perspective

        return ensemble, dc_pmf

    def _sample_goals(
        self, dc_pmf: np.ndarray, outcome_idx: int,
    ) -> Tuple[int, int]:
        """Sample (goals_a, goals_b) from DC PMF conditional on outcome."""
        n = dc_pmf.shape[0]
        idx = np.arange(n)
        if outcome_idx == 0:
            mask = idx[:, None] > idx[None, :]
        elif outcome_idx == 1:
            mask = idx[:, None] == idx[None, :]
        else:
            mask = idx[:, None] < idx[None, :]

        cond = dc_pmf * mask
        total = cond.sum()
        if total < 1e-15:
            cond = mask.astype(float)
            total = cond.sum()
        cond = (cond / total).ravel()

        flat = int(self.rng.choice(dc_pmf.size, p=cond))
        ga, gb = np.unravel_index(flat, dc_pmf.shape)
        return int(ga), int(gb)

    # ------------------------------------------------------------------
    # Knockout-stage match prediction (with analytical Elo-jitter correction)
    # ------------------------------------------------------------------

    def _predict_ko(
        self, team_a: str, team_b: str, venue: str, jitter: Dict[str, float],
    ) -> np.ndarray:
        """
        Return [P(team_a wins), P(draw), P(team_b wins)] for a knockout match.
        Uses cached full predictor + analytical Elo-jitter correction on logit.
        """
        cache_key = (team_a, team_b, venue)
        if cache_key not in self._ko_cache:
            result = self.predictor.predict(
                team_a, team_b, '2026-07-01', venue=venue, stage='knockout',
            )
            self._ko_cache[cache_key] = result

        res = self._ko_cache[cache_key]
        base = np.array([res.p_win, res.p_draw, res.p_loss])

        # Analytical Elo-jitter correction: shift log-odds of W vs L
        delta_elo = jitter.get(team_a, 0.0) - jitter.get(team_b, 0.0)
        if abs(delta_elo) > 0.5:
            logit_shift = delta_elo * np.log(10.0) / 400.0
            log_p = np.log(np.clip(base, 1e-15, 1.0))
            log_p[0] += logit_shift
            log_p[2] -= logit_shift
            base = softmax(log_p)

        return base

    def _ko_venue(self, team_a: str, team_b: str, match_num: int) -> str:
        """
        Venue for a knockout match from team_a's perspective.

        A host gets 'home'/'away' ONLY when this specific match is played in its
        own country (per the KO venue table). Otherwise neutral — a host playing
        a knockout in a different host country gets no advantage.
        """
        country = self._ko_venue_country.get(match_num, '')
        if not country:
            return 'neutral'
        a_home = (team_a == country)
        b_home = (team_b == country)
        if a_home and not b_home:
            return 'home'
        if b_home and not a_home:
            return 'away'
        return 'neutral'

    def _play_ko_match(
        self, team_a: str, team_b: str, jitter: Dict[str, float], match_num: int,
    ) -> str:
        """Simulate one knockout match and return the winner."""
        venue  = self._ko_venue(team_a, team_b, match_num)
        proba  = self._predict_ko(team_a, team_b, venue, jitter)
        outcome = int(self.rng.choice(3, p=proba))
        if outcome == 0:
            return team_a
        if outcome == 2:
            return team_b
        # Draw → penalty shootout (~50/50)
        return team_a if self.rng.random() < 0.5 else team_b

    # ------------------------------------------------------------------
    # One full simulation
    # ------------------------------------------------------------------

    def _simulate_one(self) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        Run one complete tournament simulation.
        Returns (outcomes, group_ranks) where:
          outcomes    : {team: stage}  stage ∈ 'group_exit'|'r32_exit'|...|'champion'
          group_ranks : {team: 1|2|3|4}  finishing position in group
        """
        jitter      = self._sample_jitter()
        outcomes    = {}   # team -> final stage
        group_ranks = {}   # team -> 1-4 group finish
        standings   = {}   # group -> [1st, 2nd, 3rd, 4th]
        thirds      = []   # list of {team, group, pts, gd, gf}

        # ----------------------------------------------------------------
        # Group stage: batch XGB ensemble (one 67-row predict_proba call)
        # ----------------------------------------------------------------
        ensemble_shifted = self._batch_predict_group(jitter)  # (n_unplayed, 3)

        # Vectorised goal sampling for all unplayed fixtures at once
        if self._gf_n > 0:
            # Sample outcome index for each unplayed fixture
            cdf_outcomes = np.cumsum(ensemble_shifted, axis=1)          # (n, 3)
            u_outcomes   = self.rng.random(self._gf_n)                  # (n,)
            oidx_all     = (u_outcomes[:, None] >= cdf_outcomes).sum(axis=1)  # (n,)
            oidx_all     = np.clip(oidx_all, 0, 2)

            # Vectorised goal sampling using pre-computed conditional CDFs
            fi = np.arange(self._gf_n)
            sel_cdf  = self._gf_dc_cond_cdf[fi, oidx_all, :]  # (n, 121) CDF per fixture
            u_goals  = self.rng.random(self._gf_n)             # (n,)
            gf_flat  = (u_goals[:, None] >= sel_cdf).sum(axis=1)  # (n,) flat scoreline idx
            ga_all, gb_all = np.unravel_index(gf_flat, (11, 11))

        for grp, teams in GROUPS.items():
            match_results = []

            for ta, tb, played, ga_fixed, gb_fixed, unpl_idx in self._group_fixture_lists[grp]:
                if played:
                    ga, gb = ga_fixed, gb_fixed
                else:
                    ga = int(ga_all[unpl_idx])
                    gb = int(gb_all[unpl_idx])

                match_results.append({'team_a': ta, 'team_b': tb, 'gf_a': ga, 'gf_b': gb})

            ranked = rank_group(teams, match_results)
            standings[grp] = ranked

            # Record finishing position in group
            for pos, team in enumerate(ranked, 1):
                group_ranks[team] = pos

            # Accumulate third-place team records
            third = ranked[2]
            rec   = {t: {'pts': 0, 'gd': 0, 'gf': 0} for t in teams}
            for r in match_results:
                a, b, ga, gb = r['team_a'], r['team_b'], r['gf_a'], r['gf_b']
                rec[a]['pts'] += _pts(ga, gb); rec[a]['gd'] += ga - gb; rec[a]['gf'] += ga
                rec[b]['pts'] += _pts(gb, ga); rec[b]['gd'] += gb - ga; rec[b]['gf'] += gb
            thirds.append({'team': third, 'group': grp,
                           'pts': rec[third]['pts'],
                           'gd': rec[third]['gd'],
                           'gf': rec[third]['gf']})

            # 4th-place teams exit in group stage
            outcomes[ranked[3]] = 'group_exit'

        # ----------------------------------------------------------------
        # Third-place ranking — select 8 best from 12
        # ----------------------------------------------------------------
        thirds_ranked  = rank_third_place_teams(thirds)
        qualifiers_3rd = thirds_ranked[:8]
        non_qualifiers = thirds_ranked[8:]
        for r in non_qualifiers:
            outcomes[r['team']] = 'group_exit'

        qual_groups   = [r['group'] for r in qualifiers_3rd]
        slot_map      = allocate_third_place(qual_groups)   # {group: r32_match_num}
        third_by_slot = {slot_map[r['group']]: r['team'] for r in qualifiers_3rd}

        # ----------------------------------------------------------------
        # Build R32 bracket
        # ----------------------------------------------------------------
        r32_bracket: Dict[int, Tuple[str, str]] = {}

        for match, (slot_a, grp_a, slot_b, grp_b) in R32_FIXED.items():
            team_a = standings[grp_a][0 if slot_a == 'winner' else 1]
            team_b = standings[grp_b][0 if slot_b == 'winner' else 1]
            r32_bracket[match] = (team_a, team_b)

        for match, (winner_grp, _compat) in R32_THIRD.items():
            team_a = standings[winner_grp][0]   # group winner
            team_b = third_by_slot[match]        # 3rd-place qualifier
            r32_bracket[match] = (team_a, team_b)

        # ----------------------------------------------------------------
        # Knockout rounds
        # ----------------------------------------------------------------
        ko_winners: Dict[int, str] = {}

        def play_round(bracket: Dict[int, Tuple[int, int]], exit_label: str) -> None:
            for match, (feed_a, feed_b) in bracket.items():
                ta = ko_winners[feed_a]
                tb = ko_winners[feed_b]
                winner = self._play_ko_match(ta, tb, jitter, match)
                loser  = tb if winner == ta else ta
                outcomes[loser]    = exit_label
                ko_winners[match]  = winner

        # R32
        for match in ALL_R32:
            ta, tb = r32_bracket[match]
            winner = self._play_ko_match(ta, tb, jitter, match)
            loser  = tb if winner == ta else ta
            outcomes[loser]   = 'r32_exit'
            ko_winners[match] = winner

        play_round(R16_BRACKET, 'r16_exit')
        play_round(QF_BRACKET,  'qf_exit')
        play_round(SF_BRACKET,  'sf_exit')

        # Final
        sf_a = ko_winners[101]
        sf_b = ko_winners[102]
        champion = self._play_ko_match(sf_a, sf_b, jitter, FINAL_MATCH_NUM)
        runner   = sf_b if champion == sf_a else sf_a
        outcomes[runner]   = 'final_loss'
        outcomes[champion] = 'champion'

        return outcomes, group_ranks

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    # Column order for the outcome matrix, fixed so portfolio modules can
    # interpret columns consistently across different simulation runs.
    OUTCOME_MATRIX_STAGES = [
        'reach_r32', 'reach_r16', 'reach_qf', 'reach_sf', 'reach_final', 'champion',
    ]

    @property
    def outcome_matrix_market_ids(self) -> list:
        """Canonical market_id list matching outcome_matrix columns.

        Format: '<team>:<stage>'  e.g. 'Brazil:champion', 'France:reach_sf'
        Alphabetically sorted by team then stage for deterministic column order.
        """
        ids = []
        for team in sorted(ALL_WC_TEAMS):
            for stage in self.OUTCOME_MATRIX_STAGES:
                ids.append(f'{team}:{stage}')
        return ids

    def run(self, return_outcome_matrix: bool = False):
        """
        Run N simulations.

        Args:
            return_outcome_matrix: if True, also return the raw
                (n_simulations × n_markets) binary outcome matrix needed
                by MCCovarianceExtractor for portfolio-aware Kelly sizing.

        Returns:
            probs — {team: {stage: probability}} dict (always).
            outcome_matrix — np.ndarray shape (n_sims, n_markets), dtype float32,
                only returned when return_outcome_matrix=True.

        Stages tracked per team:
            group_exit, r32_exit, r16_exit, qf_exit, sf_exit, final_loss, champion
        Cumulative probabilities (e.g. P(reach R16)) can be computed as
            1 - P(group_exit) - P(r32_exit).
        """
        import numpy as np

        stage_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        group_rank_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # Build column index for outcome matrix (only when needed — avoid
        # allocating 100k × 288 float32 array if caller doesn't want it).
        if return_outcome_matrix:
            market_ids = self.outcome_matrix_market_ids
            mid_to_col = {mid: i for i, mid in enumerate(market_ids)}
            n_cols = len(market_ids)
            outcome_rows = np.zeros((self.n_simulations, n_cols), dtype=np.float32)

        log.info("Running %d simulations ...", self.n_simulations)
        print_every = max(1, self.n_simulations // 10)

        for i in range(self.n_simulations):
            if (i + 1) % print_every == 0:
                log.info("  %6d / %d  (%.0f%%)", i + 1, self.n_simulations,
                         100 * (i + 1) / self.n_simulations)

            sim, g_ranks = self._simulate_one()
            for team, stage in sim.items():
                stage_counts[team][stage] += 1
            for team, rank in g_ranks.items():
                group_rank_counts[team][rank] += 1

            if return_outcome_matrix:
                # Translate per-team stage outcome into binary reach columns.
                # A team that exits at 'r16_exit' reached r32 and r16 but not qf.
                _stage_order = {
                    'group_exit': 0, 'r32_exit': 1, 'r16_exit': 2,
                    'qf_exit': 3, 'sf_exit': 4, 'final_loss': 5, 'champion': 6,
                }
                _reach_threshold = {
                    'reach_r32': 1, 'reach_r16': 2, 'reach_qf': 3,
                    'reach_sf': 4, 'reach_final': 5, 'champion': 6,
                }
                for team, stage in sim.items():
                    exit_level = _stage_order.get(stage, 0)
                    for reach_stage, threshold in _reach_threshold.items():
                        if exit_level >= threshold:
                            col = mid_to_col.get(f'{team}:{reach_stage}')
                            if col is not None:
                                outcome_rows[i, col] = 1.0

        n = self.n_simulations
        probs: Dict[str, Dict[str, float]] = {}
        for team in ALL_WC_TEAMS:
            cnt = stage_counts[team]
            p   = {stage: cnt.get(stage, 0) / n for stage in
                   ['group_exit', 'r32_exit', 'r16_exit', 'qf_exit',
                    'sf_exit', 'final_loss', 'champion']}
            p['reach_r32'] = 1.0 - p['group_exit']
            p['reach_r16'] = p['reach_r32'] - p['r32_exit']
            p['reach_qf']  = p['reach_r16'] - p['r16_exit']
            p['reach_sf']  = p['reach_qf']  - p['qf_exit']
            p['reach_final'] = p['reach_sf'] - p['sf_exit']
            gc = group_rank_counts[team]
            p['group_rank_1'] = gc.get(1, 0) / n
            p['group_rank_2'] = gc.get(2, 0) / n
            p['group_rank_3'] = gc.get(3, 0) / n
            p['group_rank_4'] = gc.get(4, 0) / n
            probs[team] = p

        if return_outcome_matrix:
            return probs, outcome_rows, market_ids
        return probs
