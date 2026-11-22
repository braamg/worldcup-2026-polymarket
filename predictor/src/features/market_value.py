"""
Squad market value feature: leak-free lookup from a table of squad-value snapshots.

Public-repo note
----------------
The original project built the snapshot table by scraping national-team squad pages from
Transfermarkt. Transfermarkt's terms forbid scraping, so the scraper, the HTML parser and
the team-ID map are NOT included here. This module only contains the feature logic, which
works on any snapshot table you supply with these columns:

    team             canonical team name (matching src/data/constants.py)
    saison_id        season start year (season Y/Y+1)
    squad_value_eur  total squad market value in EUR
    valid_from       pd.Timestamp; January 1 of (saison_id + 1)

Design
------
- Snapshot valid_from = January 1 of (saison_id + 1), so a match in any month of year Y
  uses the saison_id=Y-1 snapshot. This is strictly leak-free.
- Feature: log(mv_home / mv_away), an era-safe log-ratio.
- If either team lacks a snapshot, has_market_value = False and the ratio is NaN.
"""

import math

import pandas as pd

SNAPSHOT_COLUMNS = ["team", "saison_id", "squad_value_eur", "valid_from"]


def empty_snapshots() -> pd.DataFrame:
    """An empty snapshot table (every match then gets has_market_value=False)."""
    return pd.DataFrame(columns=SNAPSHOT_COLUMNS)


# ---------------------------------------------------------------------------
# Feature lookup
# ---------------------------------------------------------------------------

def get_mv_at_date(team: str, date: pd.Timestamp, snapshots: pd.DataFrame) -> float | None:
    """
    Return the squad market value (EUR) for *team* as of just BEFORE *date*.
    Uses the most recent snapshot with valid_from <= date.
    Returns None if no snapshot exists.
    """
    mask = (snapshots["team"] == team) & (snapshots["valid_from"] <= date)
    rows = snapshots.loc[mask]
    if rows.empty:
        return None
    return float(rows.iloc[-1]["squad_value_eur"])


def build_mv_features(
    matches: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add squad market value features to each played match.

    New columns
    -----------
    mv_home          : squad value EUR for home team (None if unavailable)
    mv_away          : squad value EUR for away team (None if unavailable)
    log_ratio_mv     : log(mv_home / mv_away), era-safe; NaN if missing
    has_market_value : bool, True if both teams have a valid snapshot
    mv_snapshot_season_home : saison_id used for home team (for staleness audit)
    mv_snapshot_season_away : saison_id used for away team
    """
    # Sort snapshots for correct iloc[-1] lookup
    snapshots = snapshots.sort_values(["team", "valid_from"]).reset_index(drop=True)

    mv_home_list, mv_away_list = [], []
    snap_home_list, snap_away_list = [], []

    for row in matches.itertuples(index=False):
        date = pd.Timestamp(row.date)
        mv_h = get_mv_at_date(row.home_team, date, snapshots)
        mv_a = get_mv_at_date(row.away_team, date, snapshots)
        mv_home_list.append(mv_h)
        mv_away_list.append(mv_a)
        # Record which saison was used
        snap_h = _snapshot_season_for(row.home_team, date, snapshots)
        snap_a = _snapshot_season_for(row.away_team, date, snapshots)
        snap_home_list.append(snap_h)
        snap_away_list.append(snap_a)

    out = matches.copy()
    out["mv_home"] = mv_home_list
    out["mv_away"] = mv_away_list
    out["mv_snapshot_season_home"] = snap_home_list
    out["mv_snapshot_season_away"] = snap_away_list

    # Log-ratio (era-safe): log(mv_home / mv_away)
    log_ratio = []
    for h, a in zip(mv_home_list, mv_away_list):
        if h is not None and a is not None and h > 0 and a > 0:
            log_ratio.append(math.log(h / a))
        else:
            log_ratio.append(float("nan"))
    out["log_ratio_mv"] = log_ratio

    both_present = [(h is not None) and (a is not None) for h, a in zip(mv_home_list, mv_away_list)]
    out["has_market_value"] = both_present

    return out


def _snapshot_season_for(team: str, date: pd.Timestamp, snapshots: pd.DataFrame) -> int | None:
    mask = (snapshots["team"] == team) & (snapshots["valid_from"] <= date)
    rows = snapshots.loc[mask]
    if rows.empty:
        return None
    return int(rows.iloc[-1]["saison_id"])
