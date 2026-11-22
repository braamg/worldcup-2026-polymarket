"""Build the canonical international match table (Phase 1 pipeline)."""
from pathlib import Path

import pandas as pd

from src.data.constants import (
    CONFEDERATION,
    DEFAULT_IMPORTANCE,
    IMPORTANCE_PATTERNS,
    TEAM_NAME_ALIASES,
)
from src.data.ingestion import fetch_results, fetch_shootouts
from src.utils import get_logger, load_config, write_parquet

log = get_logger(__name__)


def _assign_importance(tournament: str) -> int:
    t = str(tournament)
    for pattern, weight in IMPORTANCE_PATTERNS:
        if pattern.lower() in t.lower():
            return weight
    return DEFAULT_IMPORTANCE


def build_match_table(config: dict | None = None) -> pd.DataFrame:
    if config is None:
        config = load_config()

    raw_dir     = Path(config["paths"]["data_raw"])
    interim_dir = Path(config["paths"]["data_interim"])

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    results   = fetch_results(raw_dir)
    shootouts = fetch_shootouts(raw_dir)

    log.info("Raw results:   %d rows", len(results))
    log.info("Raw shootouts: %d rows", len(shootouts))

    # ------------------------------------------------------------------
    # 2. Capture pre-alias team names for the index file
    # ------------------------------------------------------------------
    raw_home_teams = set(results["home_team"].unique())
    raw_away_teams = set(results["away_team"].unique())
    all_raw_teams  = sorted(raw_home_teams | raw_away_teams)

    # ------------------------------------------------------------------
    # 3. Normalize team names
    # ------------------------------------------------------------------
    for df in (results, shootouts):
        for col in ("home_team", "away_team"):
            if col in df.columns:
                df[col] = df[col].replace(TEAM_NAME_ALIASES)
        if "winner" in df.columns:
            df["winner"] = df["winner"].replace(TEAM_NAME_ALIASES)

    # ------------------------------------------------------------------
    # 4. Type coercion
    # ------------------------------------------------------------------
    results["date"]       = pd.to_datetime(results["date"])
    results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce").astype("Int64")
    results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce").astype("Int64")

    shootouts["date"] = pd.to_datetime(shootouts["date"])

    # ------------------------------------------------------------------
    # 4b. Deduplication (keep first occurrence, log count)
    # ------------------------------------------------------------------
    n_before = len(results)
    results = results.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
    n_dupes = n_before - len(results)
    if n_dupes:
        log.warning("Dropped %d duplicate rows (same date + teams)", n_dupes)

    # ------------------------------------------------------------------
    # 5. Derived columns
    # ------------------------------------------------------------------
    results["goal_diff"] = (results["home_score"] - results["away_score"]).astype("Int64")
    # is_played = False for future fixtures (null scores) or abandoned matches
    results["is_played"] = results["home_score"].notna() & results["away_score"].notna()
    # result is null for unplayed/abandoned fixtures; never default to "D"
    results["result"] = None
    played = results["is_played"]
    results.loc[played & (results["goal_diff"] > 0), "result"] = "H"
    results.loc[played & (results["goal_diff"] < 0), "result"] = "A"
    results.loc[played & (results["goal_diff"] == 0), "result"] = "D"
    # Rename neutral → neutral_venue; handle bool/string variants
    results["neutral_venue"] = results["neutral"].astype(bool)

    # ------------------------------------------------------------------
    # 6. Match importance weight
    # ------------------------------------------------------------------
    results["importance"] = results["tournament"].apply(_assign_importance)

    # ------------------------------------------------------------------
    # 7. Join shootout winner
    # ------------------------------------------------------------------
    join_cols = ["date", "home_team", "away_team"]
    shootout_slim = (
        shootouts[join_cols + ["winner"]]
        .rename(columns={"winner": "shootout_winner"})
    )
    matches = results.merge(shootout_slim, on=join_cols, how="left")

    # ------------------------------------------------------------------
    # 8. Confederation tags
    # ------------------------------------------------------------------
    matches["home_conf"] = matches["home_team"].map(CONFEDERATION)
    matches["away_conf"] = matches["away_team"].map(CONFEDERATION)
    matches["cross_conf"] = (
        matches["home_conf"].notna()
        & matches["away_conf"].notna()
        & (matches["home_conf"] != matches["away_conf"])
    )

    # ------------------------------------------------------------------
    # 9. Final column order
    # ------------------------------------------------------------------
    keep = [
        "date", "home_team", "away_team",
        "home_score", "away_score", "goal_diff", "result",
        "is_played",
        "tournament", "city", "country", "neutral_venue",
        "importance", "shootout_winner",
        "home_conf", "away_conf", "cross_conf",
    ]
    matches = matches[keep].sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 10. Save match table
    # ------------------------------------------------------------------
    out_path = interim_dir / "matches.parquet"
    write_parquet(matches, out_path)
    log.info("Saved match table → %s  (%d rows)", out_path, len(matches))

    # ------------------------------------------------------------------
    # 11. Save team index (name mapping + confederation)
    # ------------------------------------------------------------------
    index_rows = []
    for raw_name in all_raw_teams:
        canonical = TEAM_NAME_ALIASES.get(raw_name, raw_name)
        conf      = CONFEDERATION.get(canonical)
        index_rows.append({"raw_name": raw_name, "canonical_name": canonical, "confederation": conf})
    team_index = pd.DataFrame(index_rows).sort_values("canonical_name").reset_index(drop=True)
    write_parquet(team_index, interim_dir / "team_index.parquet")
    log.info("Saved team index → %d teams", len(team_index))

    return matches
