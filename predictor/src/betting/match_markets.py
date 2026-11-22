"""
match_markets.py
================

The model side of single-MATCH betting (3-way moneyline: team_a win / draw / team_b win).

Unlike the tournament futures (model_markets.py, fed by the simulator's aggregated probs),
match markets use the predictor's per-match W/D/L directly — the layer that was actually
calibration-validated walk-forward (Phase 7) and on the WC slice (Phase 10). This is the
honest place to look for a real, measurable edge.

Both this module and the Polymarket match fetcher key every outcome on a CANONICAL match
key (teams sorted alphabetically), so the two tables join without needing a team-code map
or exact date match — a pair of group opponents meets exactly once.

    canonical_match_key("Argentina", "Algeria") -> ("Algeria", "Argentina", "match:Algeria|Argentina")
    market_ids -> {"Algeria": "...:Algeria", "draw": "...:draw", "Argentina": "...:Argentina"}
"""

from __future__ import annotations

import pandas as pd


def canonical_match_key(a: str, b: str) -> tuple[str, str, str]:
    """Return (t1, t2, group_id) with t1 < t2 alphabetically."""
    t1, t2 = sorted([a, b])
    return t1, t2, f"match:{t1}|{t2}"


def match_market_ids(a: str, b: str) -> dict[str, str]:
    """Map each outcome of a match to its stable market_id (same on model + market sides)."""
    t1, t2, grp = canonical_match_key(a, b)
    return {"group": grp, t1: f"{grp}:{t1}", "draw": f"{grp}:draw", t2: f"{grp}:{t2}"}


def _fixture_venue(row) -> str:
    """neutral / home (team_a hosts) / away (team_b hosts), from the host_team column."""
    ht = str(row.host_team).strip() if pd.notna(row.host_team) else ""
    if not ht:
        return "neutral"
    return "home" if ht == str(row.team_a).strip() else "away"


def unplayed_group_fixtures(fixtures_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """Group fixtures with no confirmed result yet, sorted by date."""
    played = set(zip(results_df["group"], results_df["team_a"], results_df["team_b"]))
    fx = fixtures_df.copy()
    fx["date"] = pd.to_datetime(fx["date"])
    mask = ~fx.apply(lambda r: (r["group"], r["team_a"], r["team_b"]) in played, axis=1)
    return fx[mask].sort_values("date").reset_index(drop=True)


def build_match_market_table(
    predictor,
    fixtures_df: pd.DataFrame,
    results_df: pd.DataFrame,
    days: int | None = None,
    today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Build the model's 3-way moneyline table for every unplayed group fixture.

    Parameters
    ----------
    predictor : a WC2026Predictor whose elo_ts has already been extended with WC results.
    days      : if set, only fixtures within `days` days of `today` (default: all unplayed).

    Returns
    -------
    DataFrame[market_id, market_type='match', team(or 'draw'), description, model_prob,
              group, short_horizon=True]  — three rows per fixture.
    """
    fx = unplayed_group_fixtures(fixtures_df, results_df)
    if days is not None:
        today = today or pd.Timestamp("today").normalize()
        fx = fx[fx["date"] <= today + pd.Timedelta(days=days)]

    rows = []
    for _, r in fx.iterrows():
        ta, tb = str(r.team_a), str(r.team_b)
        venue = _fixture_venue(r)
        res = predictor.predict(ta, tb, r["date"], venue=venue, stage="group")
        # res is from team_a's perspective: p_win=P(ta), p_draw, p_loss=P(tb)
        prob = {ta: res.p_win, "draw": res.p_draw, tb: res.p_loss}

        ids = match_market_ids(ta, tb)
        t1, t2, grp = canonical_match_key(ta, tb)
        for outcome in (t1, "draw", t2):
            desc = (f"{ta} vs {tb} — draw" if outcome == "draw"
                    else f"{outcome} to beat {t2 if outcome == t1 else t1}")
            rows.append({
                "market_id":     ids[outcome],
                "market_type":   "match",
                "team":          outcome,
                "description":   desc,
                "model_prob":    round(float(prob[outcome]), 6),
                "group":         grp,
                "short_horizon": True,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("market_id").reset_index(drop=True)
    return df
