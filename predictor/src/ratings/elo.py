"""
Football Elo engine with margin-of-victory and match-importance weighting.

Formula
-------
  E_home  = 1 / (1 + 10^((R_away - R_home - HA) / 400))
            where HA = home_advantage if not neutral, else 0.

  K_eff   = k_base * importance * mov_multiplier
  mov_mult= 1 + mov_factor * ln(|goal_diff| + 1)   (= 1 for draws)

  delta   = K_eff * (S_home - E_home)   ;  S_home ∈ {1, 0.5, 0}

Pre-match Elo (elo_home_pre / elo_away_pre) is the AS-OF-DATE snapshot
stored before each update — the feature that downstream models will use.
"""

import math
from dataclasses import dataclass

import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)

DEFAULT_RATING: float = 1500.0
_SCALE: float = 400.0


@dataclass
class EloParams:
    k_base:         float = 20.0
    home_advantage: float = 100.0
    mov_factor:     float = 0.5
    default_rating: float = DEFAULT_RATING


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def expected_home(r_home: float, r_away: float, home_adv: float, neutral: bool) -> float:
    adv = 0.0 if neutral else home_adv
    return 1.0 / (1.0 + 10.0 ** ((r_away - r_home - adv) / _SCALE))


def k_effective(k_base: float, importance: int, goal_diff: int, mov_factor: float) -> float:
    mov = 1.0 if goal_diff == 0 else 1.0 + mov_factor * math.log(abs(goal_diff) + 1)
    return k_base * importance * mov


# ---------------------------------------------------------------------------
# Main Elo runner
# ---------------------------------------------------------------------------

def run_elo(matches: pd.DataFrame, params: EloParams) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Process played matches in chronological order (matches must be pre-sorted).

    Returns
    -------
    ratings  : {team -> current_elo}  — state after the last processed match
    records  : DataFrame with one row per played match; includes pre-match Elos
               (the leak-free as-of-date snapshot) and predicted probabilities.
    """
    ratings: dict[str, float] = {}
    rows: list[dict] = []

    for row in matches.itertuples(index=False):
        if not row.is_played:
            continue

        # Safely unpack nullable Int64 goal_diff
        gd_raw = row.goal_diff
        if pd.isna(gd_raw):
            continue
        gd = int(gd_raw)

        home, away = row.home_team, row.away_team
        imp = int(row.importance)
        neu = bool(row.neutral_venue)

        r_h = ratings.get(home, params.default_rating)
        r_a = ratings.get(away, params.default_rating)

        e_h = expected_home(r_h, r_a, params.home_advantage, neu)
        s_h = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        k   = k_effective(params.k_base, imp, gd, params.mov_factor)
        d   = k * (s_h - e_h)

        rows.append({
            "date":           row.date,
            "home_team":      home,
            "away_team":      away,
            "elo_home_pre":   r_h,
            "elo_away_pre":   r_a,
            # elo_diff = effective Elo advantage for the home side (including HA for non-neutral)
            "elo_diff":       r_h - r_a + (0.0 if neu else params.home_advantage),
            "e_home":         round(e_h, 6),
            "s_home":         s_h,
            "k":              round(k, 4),
            "delta":          round(d, 4),
            "elo_home_post":  r_h + d,
            "elo_away_post":  r_a - d,
            "result":         row.result,
            "importance":     imp,
            "neutral_venue":  neu,
        })

        ratings[home] = r_h + d
        ratings[away] = r_a - d

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return ratings, df


# ---------------------------------------------------------------------------
# Time-series helpers
# ---------------------------------------------------------------------------

def make_team_series(records: pd.DataFrame) -> pd.DataFrame:
    """
    Long-format Elo time series from match records.
    One row per team-match appearance; sorted by (team, date).
    """
    home = records[["date", "home_team", "elo_home_pre", "elo_home_post"]].rename(
        columns={"home_team": "team", "elo_home_pre": "elo_before", "elo_home_post": "elo_after"}
    )
    away = records[["date", "away_team", "elo_away_pre", "elo_away_post"]].rename(
        columns={"away_team": "team", "elo_away_pre": "elo_before", "elo_away_post": "elo_after"}
    )
    series = (
        pd.concat([home, away], ignore_index=True)
        .sort_values(["team", "date"])
        .reset_index(drop=True)
    )
    return series


def get_elo(team: str, date, series: pd.DataFrame, default: float = DEFAULT_RATING) -> float:
    """
    Return the Elo rating for *team* as of just BEFORE *date*.
    Uses the last elo_after value strictly preceding the given date.
    """
    date = pd.Timestamp(date)
    mask = (series["team"] == team) & (series["date"] < date)
    rows = series.loc[mask]
    if rows.empty:
        return default
    # series is sorted by (team, date) so iloc[-1] is the most recent prior match
    return float(rows.iloc[-1]["elo_after"])


# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------

def brier_score(records: pd.DataFrame, from_date: str) -> float:
    """Mean squared error of (s_home − e_home) on matches on/after from_date."""
    subset = records[records["date"] >= pd.Timestamp(from_date)]
    if subset.empty:
        return float("inf")
    return float(((subset["s_home"] - subset["e_home"]) ** 2).mean())
