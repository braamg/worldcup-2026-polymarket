"""
Match feature table builder (Phase 3).

Assembles per-match feature rows from:
  - Phase 1: match metadata (matches.parquet)
  - Phase 2: as-of-date Elo snapshots (elo_history.parquet)
  - Phase 3: squad market value snapshots (squad_snapshots.parquet)
  + computed: rest_days_home, rest_days_away

Output: data/interim/match_features.parquet
"""

import pandas as pd

from src.features.market_value import build_mv_features
from src.utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rest-days computation
# ---------------------------------------------------------------------------

def _add_rest_days(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized: for each played match, compute days since each team's prior match.
    NaN for the first appearance of a team.
    """
    played = matches[matches["is_played"]].copy()
    played = played.sort_values("date").reset_index(drop=True)

    # Long-format with explicit side label
    home_apps = played[["date", "home_team"]].rename(columns={"home_team": "team"})
    home_apps["side"] = "home"
    home_apps["match_idx"] = played.index

    away_apps = played[["date", "away_team"]].rename(columns={"away_team": "team"})
    away_apps["side"] = "away"
    away_apps["match_idx"] = played.index

    all_apps = pd.concat([home_apps, away_apps], ignore_index=True)
    all_apps = all_apps.sort_values(["team", "date"]).reset_index(drop=True)

    # Previous match date per team (vectorized groupby shift)
    all_apps["prev_date"] = all_apps.groupby("team")["date"].shift(1)
    all_apps["rest_days"] = (all_apps["date"] - all_apps["prev_date"]).dt.days

    # Pivot back to per-match home/away rest
    rest_wide = (
        all_apps[["match_idx", "side", "rest_days"]]
        .pivot_table(index="match_idx", columns="side", values="rest_days", aggfunc="first")
    )
    rest_wide.columns.name = None
    rest_wide = rest_wide.rename(columns={"home": "rest_days_home", "away": "rest_days_away"})

    played = played.join(rest_wide)
    return played


# ---------------------------------------------------------------------------
# Elo feature join
# ---------------------------------------------------------------------------

def _join_elo(matches: pd.DataFrame, elo_history: pd.DataFrame) -> pd.DataFrame:
    """
    Merge pre-match Elo features from elo_history onto the match table.
    elo_history has one row per played match (same key: date, home_team, away_team).
    """
    elo_cols = [
        "date", "home_team", "away_team",
        "elo_home_pre", "elo_away_pre", "elo_diff", "e_home",
    ]
    elo = elo_history[elo_cols].copy()
    elo["date"] = pd.to_datetime(elo["date"])
    matches["date"] = pd.to_datetime(matches["date"])

    merged = matches.merge(elo, on=["date", "home_team", "away_team"], how="left")
    missing = merged["elo_home_pre"].isna().sum()
    if missing > 0:
        log.warning("%d matches have no Elo snapshot (unplayed or pre-history)", missing)
    return merged


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_match_features(
    matches: pd.DataFrame,
    elo_history: pd.DataFrame,
    squad_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble the canonical match feature table.

    Parameters
    ----------
    matches         : full match table (includes both played and unplayed)
    elo_history     : one row per played match with elo_home_pre etc.
    squad_snapshots : per-team-season market value snapshots

    Returns
    -------
    DataFrame (played matches only) with all features + target (result).
    Unplayed/future fixtures are excluded.
    """
    played = matches[matches["is_played"]].copy()
    log.info("Building features for %d played matches", len(played))

    # 1. Rest days
    played_with_rest = _add_rest_days(played)

    # 2. Elo features
    with_elo = _join_elo(played_with_rest, elo_history)

    # 3. Market value features
    with_mv = build_mv_features(with_elo, squad_snapshots)

    # 4. Select and reorder final columns
    core_cols = [
        "date", "home_team", "away_team", "tournament", "neutral_venue",
        "importance", "home_conf", "away_conf", "cross_conf", "shootout_winner",
    ]
    elo_feature_cols = ["elo_home_pre", "elo_away_pre", "elo_diff", "e_home"]
    mv_feature_cols  = [
        "mv_home", "mv_away", "log_ratio_mv", "has_market_value",
        "mv_snapshot_season_home", "mv_snapshot_season_away",
    ]
    rest_cols  = ["rest_days_home", "rest_days_away"]
    target_col = ["result", "home_score", "away_score", "goal_diff", "is_played"]

    keep = core_cols + elo_feature_cols + mv_feature_cols + rest_cols + target_col
    keep = [c for c in keep if c in with_mv.columns]  # guard against missing cols

    features = with_mv[keep].sort_values("date").reset_index(drop=True)
    log.info("Feature table: %d rows × %d columns", len(features), len(features.columns))
    return features
