"""
Live Elo updater for the 2026 World Cup.

Extends the predictor's in-memory Elo timeseries with confirmed WC 2026 results
so that all subsequent predict() calls use WC-updated team strengths.

Does NOT write to disk — the on-disk Elo timeseries (used for model training)
is never touched.  Call extend_elo_with_wc_results() once after loading the
predictor; the simulator and matchday script then pick up updated Elos
automatically via get_elo().
"""

from __future__ import annotations

from typing import Dict
import pandas as pd

from src.ratings.elo import expected_home, k_effective, get_elo

WC_IMPORTANCE = 8   # FIFA World Cup group-stage importance weight


def extend_elo_with_wc_results(
    predictor,
    results_df: pd.DataFrame,
    fixtures_df: pd.DataFrame,
    k_base: float = 20.0,
    home_advantage: float = 100.0,
    mov_factor: float = 0.1,
) -> Dict[str, float]:
    """
    Extend predictor.elo_ts in memory with confirmed WC 2026 results.

    Processes results in chronological order, applying the same Elo update
    formula used during the historical build (run_elo in src/ratings/elo.py).

    Returns the current Elo ratings for all teams that have played at least
    one confirmed WC match (keyed by canonical team name).

    Parameters
    ----------
    predictor      : WC2026Predictor — elo_ts is extended in-place
    results_df     : confirmed WC results (group, team_a, team_b, date, goals_a, goals_b)
    fixtures_df    : full fixture list  (group, team_a, team_b, date, host_team)
    k_base         : Elo K-factor base  (config.yaml  elo.k_base)
    home_advantage : home-team Elo advantage in points (config.yaml elo.home_advantage)
    mov_factor     : margin-of-victory log-scaling exponent (config.yaml elo.mov_factor)
    """
    if results_df.empty:
        return {}

    # Build host-team lookup keyed on (team_a, team_b) from the fixture list
    fix_host: Dict[tuple, str] = {}
    for _, row in fixtures_df.iterrows():
        ht = str(row.host_team).strip() if pd.notna(row.host_team) and str(row.host_team).strip() else ''
        fix_host[(str(row.team_a), str(row.team_b))] = ht

    ratings: Dict[str, float] = {}   # running Elo state during processing
    new_rows: list[dict] = []

    for _, res in results_df.sort_values('date').iterrows():
        ta, tb   = str(res.team_a), str(res.team_b)
        date     = pd.Timestamp(res.date)
        ga, gb   = int(res.goals_a), int(res.goals_b)

        # Pre-match Elo: use the running dict for teams that already played a
        # WC game, otherwise fall back to the historical timeseries.
        r_a = ratings[ta] if ta in ratings else get_elo(ta, date, predictor.elo_ts)
        r_b = ratings[tb] if tb in ratings else get_elo(tb, date, predictor.elo_ts)

        # Determine venue so we can apply home advantage correctly
        ht        = fix_host.get((ta, tb), '')
        is_neutral = not ht
        home_is_a  = not is_neutral and ht == ta   # ta plays at home

        if is_neutral:
            r_home, r_away, gd, ha, home_is_a = r_a, r_b, ga - gb, 0.0, True
        elif home_is_a:
            r_home, r_away, gd, ha = r_a, r_b, ga - gb, home_advantage
        else:   # tb is home
            r_home, r_away, gd, ha, home_is_a = r_b, r_a, gb - ga, home_advantage, False

        e_h   = expected_home(r_home, r_away, ha, is_neutral)
        s_h   = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        k     = k_effective(k_base, WC_IMPORTANCE, abs(gd), mov_factor)
        delta = k * (s_h - e_h)

        r_home_new = r_home + delta
        r_away_new = r_away - delta

        if home_is_a:
            ratings[ta], ratings[tb] = r_home_new, r_away_new
            new_rows += [
                {'date': date, 'team': ta, 'elo_before': r_home, 'elo_after': r_home_new},
                {'date': date, 'team': tb, 'elo_before': r_away, 'elo_after': r_away_new},
            ]
        else:
            ratings[ta], ratings[tb] = r_away_new, r_home_new
            new_rows += [
                {'date': date, 'team': ta, 'elo_before': r_away, 'elo_after': r_away_new},
                {'date': date, 'team': tb, 'elo_before': r_home, 'elo_after': r_home_new},
            ]

    if new_rows:
        new_ts = pd.DataFrame(new_rows)
        predictor.elo_ts = (
            pd.concat([predictor.elo_ts, new_ts], ignore_index=True)
            .sort_values(['team', 'date'])
            .reset_index(drop=True)
        )

    return ratings


def print_elo_changes(ratings: Dict[str, float], predictor, results_df: pd.DataFrame) -> None:
    """Print a compact summary of Elo changes from confirmed WC results."""
    if not ratings:
        print("  No confirmed WC results yet — Elo unchanged.")
        return

    from src.ratings.elo import get_elo
    cutoff = pd.Timestamp('2026-06-11')   # day before tournament started

    print(f"\n  Elo changes from {len(results_df)} confirmed WC result(s):")
    print(f"  {'Team':<24}  {'Pre-WC':>7}  {'Current':>7}  {'Change':>7}")
    print(f"  {'─'*24}  {'─'*7}  {'─'*7}  {'─'*7}")

    for team in sorted(ratings):
        pre  = get_elo(team, cutoff, predictor.elo_ts)
        post = ratings[team]
        sign = '+' if post - pre >= 0 else ''
        print(f"  {team:<24}  {pre:>7.1f}  {post:>7.1f}  {sign}{post-pre:>6.1f}")
