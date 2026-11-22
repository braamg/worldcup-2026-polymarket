"""
Leakage audit — verifies that every feature uses only data strictly before the match date.

Checks performed on a 200-match random sample:

  1. Elo provenance
     elo_home_pre / elo_away_pre must equal the rating reconstructed from
     elo_timeseries as of just before the match date.  The elo_timeseries
     stores elo_after for each match; the rating going into match M on date D
     is the elo_after of the team's most recent prior row where date < D.
     A team with no prior history must show DEFAULT_RATING (1500).

  2. Market value temporality
     mv_snapshot_season_home / _away must equal match.year - 1.
     That snapshot reflects data Transfermarkt published ~August of (year-1),
     which is strictly before any match in year (earliest would be January 1).

Assumptions:
  - International teams cannot play two matches on the same day (true for FIFA
    members), so same-date sequencing in the Elo engine never creates apparent
    Elo mismatches.
  - The modeling_table.parquet must exist (run phase5_build_modeling_table.py first).
"""

from pathlib import Path

import pandas as pd
import pytest

from src.ratings.elo import DEFAULT_RATING
from src.utils import load_config, read_parquet

_ELO_TOLERANCE = 0.02   # rounding buffer for floating-point comparisons
_N_SAMPLE       = 200
_RANDOM_STATE   = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _elo_before(team: str, before_date: pd.Timestamp, ts_team: pd.DataFrame) -> float:
    """
    Return the Elo rating for `team` just before `before_date`.

    `ts_team` is the elo_timeseries slice already filtered to this team and
    sorted by date ascending.  We take the elo_after of the last row whose
    date is strictly less than before_date.
    """
    prior = ts_team[ts_team["date"] < before_date]
    if prior.empty:
        return DEFAULT_RATING
    return float(prior.iloc[-1]["elo_after"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def modeling_table(config):
    path = Path(config["paths"]["data_processed"]) / "modeling_table.parquet"
    if not path.exists():
        pytest.skip(
            "modeling_table.parquet not found — run scripts/phase5_build_modeling_table.py first"
        )
    df = read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def elo_timeseries_by_team(config):
    """Pre-index elo_timeseries by team for fast per-team lookups."""
    path = Path(config["paths"]["data_interim"]) / "elo_timeseries.parquet"
    ts = read_parquet(path)
    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.sort_values(["team", "date"]).reset_index(drop=True)
    return {team: grp.reset_index(drop=True) for team, grp in ts.groupby("team")}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_future_features_in_historical_matches(modeling_table, elo_timeseries_by_team):
    """
    Verify temporal integrity for 200 randomly sampled matches.
    One violation causes the test to fail with a descriptive message.
    """
    sample = modeling_table.sample(n=min(_N_SAMPLE, len(modeling_table)), random_state=_RANDOM_STATE)

    violations: list[str] = []
    empty_ts = pd.DataFrame(columns=["date", "team", "elo_before", "elo_after"])

    for _, row in sample.iterrows():
        d  = pd.Timestamp(row["date"])
        ht = row["home_team"]
        at = row["away_team"]

        # ── 1. Elo provenance ─────────────────────────────────────────────
        ts_h = elo_timeseries_by_team.get(ht, empty_ts)
        ts_a = elo_timeseries_by_team.get(at, empty_ts)

        expected_h = _elo_before(ht, d, ts_h)
        expected_a = _elo_before(at, d, ts_a)

        if abs(expected_h - row["elo_home_pre"]) > _ELO_TOLERANCE:
            violations.append(
                f"Elo home mismatch: {ht} vs {at} on {d.date()} — "
                f"reconstructed={expected_h:.3f}, stored={row['elo_home_pre']:.3f}"
            )
        if abs(expected_a - row["elo_away_pre"]) > _ELO_TOLERANCE:
            violations.append(
                f"Elo away mismatch: {ht} vs {at} on {d.date()} — "
                f"reconstructed={expected_a:.3f}, stored={row['elo_away_pre']:.3f}"
            )

        # ── 2. Market value temporality ───────────────────────────────────
        if row["has_market_value"]:
            expected_season = d.year - 1
            got_h = row.get("mv_snapshot_season_home")
            got_a = row.get("mv_snapshot_season_away")

            if got_h is not None and pd.notna(got_h) and int(got_h) != expected_season:
                violations.append(
                    f"MV home season: {ht} on {d.date()} — "
                    f"expected saison_id={expected_season}, got {got_h}"
                )
            if got_a is not None and pd.notna(got_a) and int(got_a) != expected_season:
                violations.append(
                    f"MV away season: {at} on {d.date()} — "
                    f"expected saison_id={expected_season}, got {got_a}"
                )

    assert not violations, (
        f"{len(violations)} leakage violation(s) in {_N_SAMPLE}-match audit:\n"
        + "\n".join(f"  • {v}" for v in violations[:20])
        + (f"\n  … and {len(violations) - 20} more" if len(violations) > 20 else "")
    )


def test_elo_features_never_null(modeling_table):
    """Elo is initialised at DEFAULT_RATING so pre-match values are always present."""
    assert modeling_table["elo_home_pre"].notna().all(), "elo_home_pre has nulls"
    assert modeling_table["elo_away_pre"].notna().all(), "elo_away_pre has nulls"
    assert modeling_table["elo_diff"].notna().all(),     "elo_diff has nulls"
    assert modeling_table["e_home"].notna().all(),       "e_home has nulls"


def test_result_never_null(modeling_table):
    """All rows are played matches; result must be H, D, or A."""
    assert modeling_table["result"].notna().all(), "result has nulls in modeling table"
    assert set(modeling_table["result"].unique()).issubset({"H", "D", "A"})


def test_wc2026_flag_consistent(modeling_table):
    """WC 2026 rows must have importance=8 and date in 2026."""
    wc26 = modeling_table[modeling_table["is_wc2026"]]
    if wc26.empty:
        pytest.skip("No WC 2026 rows present")
    assert (wc26["importance"] == 8).all(), "is_wc2026 rows have importance != 8"
    assert (wc26["date"].dt.year == 2026).all(), "is_wc2026 rows are not in 2026"


def test_training_weight_range(modeling_table):
    """Training weights must be positive and within the declared schedule."""
    assert (modeling_table["training_weight"] > 0).all()
    assert (modeling_table["training_weight"] <= 3.01).all(), "weight exceeds max (3.0)"


def test_rest_days_not_null_after_impute(modeling_table):
    """After imputation rest_days must be fully non-null."""
    assert modeling_table["rest_days_home"].notna().all(), "rest_days_home still has nulls"
    assert modeling_table["rest_days_away"].notna().all(), "rest_days_away still has nulls"


def test_mv_snapshot_no_future_data(modeling_table):
    """
    For all rows with has_market_value=True, the MV snapshot's valid_from date
    must be strictly before the match date.

    valid_from = January 1 of (snapshot_season + 1).

    Note: the snapshot season may be older than match.year - 1 when TM lacks recent
    data for a team (e.g. Australia stops at 2018 in the scrape window).  That is
    stale but not a leak — the data was still published before the match.
    Staleness is separately reported in the missingness output.
    """
    has_mv = modeling_table[modeling_table["has_market_value"]].copy()
    if has_mv.empty:
        pytest.skip("No rows with market value data")

    # valid_from = Jan 1 of (snapshot_season + 1)
    valid_from_h = pd.to_datetime(
        (has_mv["mv_snapshot_season_home"].astype(int) + 1).astype(str) + "-01-01"
    )
    valid_from_a = pd.to_datetime(
        (has_mv["mv_snapshot_season_away"].astype(int) + 1).astype(str) + "-01-01"
    )

    # valid_from must be strictly before the match date
    bad_h = (valid_from_h >= has_mv["date"]).sum()
    bad_a = (valid_from_a >= has_mv["date"]).sum()

    assert bad_h == 0, (
        f"{bad_h} home MV snapshots have valid_from >= match_date (temporal leak)"
    )
    assert bad_a == 0, (
        f"{bad_a} away MV snapshots have valid_from >= match_date (temporal leak)"
    )
