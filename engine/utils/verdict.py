"""
utils/verdict.py
================
The scientific scoreboard. Answers two PRE-SPECIFIED questions with PRE-SPECIFIED tests, so the
conclusion is honest rather than a post-hoc story.

Q1 — Is the Polymarket WC2026 market inefficient in detectable ways?
     Answerable NOW from price history + results. Does not use our model.
       • Favourite–longshot bias (binned price vs observed hit rate)
       • Draw mispricing (opening price vs realised draw rate)
       • Open→close drift toward the outcome (Moskowitz momentum)
       • Market calibration / Brier vs a naive 1/3 baseline

Q2 — Did OUR model beat the market?
     Accumulates forward from the forecast log (no look-ahead). Needs resolved matches.
       • Brier skill (market Brier − model Brier), paired
       • Closing-line value on backable markets (model > market), with a t-test
       • Reliability (model predicted % vs observed %)
       • Paper-trading ROI of "bet every market where model − market ≥ threshold"

The final verdict combines these into one of: EDGE FOUND / PROMISING / NO EDGE / INCONCLUSIVE,
with a confidence level driven by sample size and significance. "Inconclusive but the process is
sound" is a legitimate, honest outcome when the tournament runs out of matches first.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

_RESULTS_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "predictor" / "data" / "raw" / "wc2026_results.csv",
    Path(__file__).resolve().parents[1] / "data" / "raw" / "wc2026_results.csv",
]
_PRICE_HISTORY_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "predictor" / "data" / "processed" / "price_history.csv",
]

_ALIASES = {"Turkiye": "Turkey", "Türkiye": "Turkey", "Curacao": "Curaçao", "USA": "United States",
            "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast", "IR Iran": "Iran",
            "Korea Republic": "South Korea"}

# Group stage: top 2 of each group + 8 best third-placed teams advance (48-team format).
_N_BEST_THIRDS = 8


def _norm(t: str) -> str:
    return _ALIASES.get(str(t).strip(), str(t).strip())


# ── stats helpers (no scipy) ────────────────────────────────────────────────
def _norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _two_sided_p(z: float) -> float:
    return float(2 * (1 - _norm_cdf(abs(z))))


def prop_test(k: int, n: int, p0: float) -> dict | None:
    """Observed proportion k/n vs hypothesised p0 (normal approx)."""
    if n == 0:
        return None
    phat = k / n
    se = math.sqrt(p0 * (1 - p0) / n) if 0 < p0 < 1 else 0.0
    z = (phat - p0) / se if se > 0 else 0.0
    return {"phat": phat, "n": n, "z": z, "p": _two_sided_p(z)}


def ttest_mean(values, mu0: float = 0.0) -> dict | None:
    """One-sample test that mean(values) != mu0 (normal approx of t)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return None
    m = float(v.mean())
    se = float(v.std(ddof=1) / math.sqrt(n))
    z = (m - mu0) / se if se > 0 else 0.0
    return {"mean": m, "se": se, "n": n, "z": z, "p": _two_sided_p(z)}


def _brier(p, y) -> float:
    p = np.asarray(p, float); y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


# ── data loaders + result mapping ───────────────────────────────────────────
def load_results() -> pd.DataFrame:
    for p in _RESULTS_CANDIDATES:
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()


def load_price_history() -> pd.DataFrame:
    for p in _PRICE_HISTORY_CANDIDATES:
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()


def match_outcome_map(results: pd.DataFrame) -> dict:
    """{market_id -> (outcome 0/1, match_date)} for every resolved 3-way market."""
    omap = {}
    for _, r in results.iterrows():
        ta, tb = _norm(r["team_a"]), _norm(r["team_b"])
        t1, t2 = sorted([ta, tb])
        date = r.get("date", "")
        if r["goals_a"] > r["goals_b"]:
            winner = ta
        elif r["goals_b"] > r["goals_a"]:
            winner = tb
        else:
            winner = "draw"
        omap[f"match:{t1}|{t2}:{t1}"]   = (1.0 if winner == t1 else 0.0, date)
        omap[f"match:{t1}|{t2}:{t2}"]   = (1.0 if winner == t2 else 0.0, date)
        omap[f"match:{t1}|{t2}:draw"]   = (1.0 if winner == "draw" else 0.0, date)
    return omap


def compute_standings(results: pd.DataFrame) -> dict:
    """
    Compute group standings and who advanced (top-2 + 8 best thirds). Returns
    {team -> 1.0/0.0} ONLY for groups where all 6 games are played; a group still in progress
    yields no entries (its 'advance' bets stay pending). Empty if group column missing.
    """
    if results.empty or "group" not in results.columns:
        return {}
    rows = []
    for _, r in results.iterrows():
        ta, tb, ga, gb = _norm(r.team_a), _norm(r.team_b), r.goals_a, r.goals_b
        for team, gf, gally in [(ta, ga, gb), (tb, gb, ga)]:
            pts = 3 if gf > gally else (1 if gf == gally else 0)
            rows.append({"group": r["group"], "team": team, "pts": pts,
                         "gd": gf - gally, "gf": gf})
    tbl = pd.DataFrame(rows).groupby(["group", "team"]).agg(
        played=("pts", "size"), pts=("pts", "sum"),
        gd=("gd", "sum"), gf=("gf", "sum")).reset_index()

    advanced = {}
    thirds = []
    for g, gdf in tbl.groupby("group"):
        if gdf["played"].sum() < 12:        # 4 teams × 3 games each = 12 team-games per group
            continue                         # group incomplete → leave pending
        ranked = gdf.sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True)
        for i, row in ranked.iterrows():
            if i < 2:
                advanced[row.team] = 1.0
            elif i == 2:
                thirds.append((row.team, row.pts, row.gd, row.gf))
            else:
                advanced[row.team] = 0.0
    # rank third-placed teams across completed groups; best 8 advance (only valid once all done)
    if thirds:
        thirds.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        n_groups_done = tbl.groupby("group")["played"].sum().ge(12).sum()
        all_groups_done = n_groups_done == tbl["group"].nunique() == 12
        for i, (team, *_rest) in enumerate(thirds):
            if all_groups_done:
                advanced[team] = 1.0 if i < _N_BEST_THIRDS else 0.0
            # else: thirds remain pending until every group finishes
    return advanced


# ── Q1: market efficiency ───────────────────────────────────────────────────
def market_efficiency_report(price_history: pd.DataFrame, results: pd.DataFrame) -> dict:
    omap = match_outcome_map(results)
    if not omap or price_history.empty:
        return {}

    ph = price_history[price_history["market_id"].isin(omap)].copy()
    if ph.empty:
        return {}
    ph["ts"] = pd.to_datetime(ph["ts"], errors="coerce")
    ph = ph.dropna(subset=["ts"]).sort_values("ts")
    ph["mdate"] = ph["market_id"].map(lambda m: pd.to_datetime(omap[m][1], errors="coerce"))

    opens = ph.groupby("market_id")["market_mid"].first()
    pre = ph[ph["ts"] < ph["mdate"]]
    closes_pre = pre.groupby("market_id")["market_mid"].last()
    # fallback close: last mid-range price if no strictly-pre-match snapshot
    midr = ph[(ph["market_mid"] > 0.03) & (ph["market_mid"] < 0.97)]
    closes_fb = midr.groupby("market_id")["market_mid"].last()

    rows = []
    for mid, (y, mdate) in omap.items():
        if mid not in opens.index:
            continue
        op = float(opens[mid])
        cl = float(closes_pre[mid]) if mid in closes_pre.index else \
             (float(closes_fb[mid]) if mid in closes_fb.index else op)
        rows.append({"market_id": mid, "y": y, "open": op, "close": cl,
                     "is_draw": mid.endswith(":draw")})
    df = pd.DataFrame(rows)
    if df.empty:
        return {}

    y = df["y"].to_numpy()
    brier_close = _brier(df["close"], y)
    brier_open = _brier(df["open"], y)
    brier_naive = _brier(np.full(len(df), 1/3), y)

    # Favourite–longshot bias: bin closing price, compare observed vs implied
    bands = [(0, .10), (.10, .20), (.20, .30), (.30, .40), (.40, .60), (.60, 1.01)]
    band_rows = []
    for lo, hi in bands:
        m = (df["close"] >= lo) & (df["close"] < hi)
        if not m.any():
            continue
        band_rows.append({
            "Price band": f"{int(lo*100)}–{int(hi*100)}¢",
            "Implied %":  round(float(df.loc[m, "close"].mean()) * 100, 1),
            "Observed %": round(float(df.loc[m, "y"].mean()) * 100, 1),
            "n":          int(m.sum()),
        })
    band_df = pd.DataFrame(band_rows)

    # Draw mispricing: opening price vs realised draw rate
    draws = df[df["is_draw"]]
    draw_test = None
    if len(draws) >= 5:
        k = int(draws["y"].sum())
        mean_open = float(draws["open"].mean())
        draw_test = {
            "n": len(draws), "actual_rate": float(draws["y"].mean()),
            "market_open": mean_open,
            **(prop_test(k, len(draws), mean_open) or {}),
        }

    # Open→close drift toward outcome (Moskowitz): YES markets should drift up, NO down
    df["drift"] = df["close"] - df["open"]
    drift_yes = ttest_mean(df.loc[df["y"] == 1, "drift"].to_numpy())
    drift_no  = ttest_mean(df.loc[df["y"] == 0, "drift"].to_numpy())

    return {
        "n_markets": len(df), "n_matches": len(results),
        "brier_close": round(brier_close, 4), "brier_open": round(brier_open, 4),
        "brier_naive": round(brier_naive, 4),
        "band_df": band_df, "draw_test": draw_test,
        "drift_yes": drift_yes, "drift_no": drift_no,
    }


# ── Q2: model edge ──────────────────────────────────────────────────────────
def resolved_observations(forecast_log: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """
    Clean resolved (model, market, close, outcome) rows from the forward forecast log.
    Each market's FIRST snapshot is the pre-match forecast/entry; the LAST is the close proxy.
    Only markets whose outcome is known (match W/D/L, or group-advance once the group completes)
    are returned. This is the contamination-free spine for both the verdict and trust calibration.
    """
    if forecast_log is None or forecast_log.empty:
        return pd.DataFrame()
    omap = match_outcome_map(results)
    advanced = compute_standings(results)
    fl = forecast_log.copy()
    fl["snapshot_date"] = pd.to_datetime(fl["snapshot_date"], errors="coerce")

    rows = []
    for mid, g in fl.groupby("market_id"):
        g = g.sort_values("snapshot_date")
        y = None
        if mid in omap:
            y = omap[mid][0]
        elif str(mid).startswith("advance:"):
            y = advanced.get(_norm(str(mid).split(":", 1)[1]))
        if y is None:
            continue
        rows.append({
            "market_id": mid, "bet_type": g["bet_type"].iloc[0], "y": float(y),
            "model": float(g["model_prob"].iloc[0]), "market": float(g["market_price"].iloc[0]),
            "close": float(g["market_price"].iloc[-1]),
        })
    return pd.DataFrame(rows)


def model_edge_report(forecast_log: pd.DataFrame, results: pd.DataFrame,
                      back_threshold: float = 0.05) -> dict:
    if forecast_log is None or forecast_log.empty:
        return {"n_resolved": 0}

    d = resolved_observations(forecast_log, results)
    n_logged = int(forecast_log["market_id"].nunique())
    if d.empty:
        return {"n_resolved": 0, "n_logged": n_logged}
    y = d["y"].to_numpy()
    brier_model  = _brier(d["model"], y)
    brier_market = _brier(d["market"], y)

    # CLV on backable markets (model says buy: model − market ≥ threshold)
    backable = d[d["model"] - d["market"] >= back_threshold].copy()
    clv_test = None
    paper = None
    if len(backable) >= 2:
        backable["clv_cents"] = (backable["close"] - backable["market"]) * 100
        clv_test = ttest_mean(backable["clv_cents"].to_numpy(), 0.0)
        # paper-trade: R1 on each backable market at entry price, settle at outcome
        stake = 1.0
        ret = np.where(backable["y"].to_numpy() == 1,
                       stake / backable["market"].to_numpy() - stake, -stake)
        paper = {"n": len(backable), "roi": float(ret.sum() / (stake * len(backable))),
                 "wins": int(backable["y"].sum()), "net": float(ret.sum())}

    # reliability table
    rel_bins = [(0, .2), (.2, .35), (.35, .5), (.5, .7), (.7, 1.01)]
    rel_rows = []
    for lo, hi in rel_bins:
        m = (d["model"] >= lo) & (d["model"] < hi)
        if not m.any():
            continue
        rel_rows.append({"Model bucket": f"{int(lo*100)}–{int(hi*100)}%",
                         "Predicted %": round(float(d.loc[m, "model"].mean()) * 100, 1),
                         "Observed %":  round(float(d.loc[m, "y"].mean()) * 100, 1),
                         "n": int(m.sum())})

    return {
        "n_resolved": len(d), "n_logged": n_logged,
        "n_backable": int(len(backable)),
        "brier_model": round(brier_model, 4), "brier_market": round(brier_market, 4),
        "brier_skill": round(brier_market - brier_model, 4),
        "clv_test": clv_test, "paper": paper,
        "reliability": pd.DataFrame(rel_rows),
    }


# ── Final verdict synthesis ─────────────────────────────────────────────────
def final_verdict(q1: dict, q2: dict, n_min: int = 15, n_strong: int = 30) -> dict:
    n = q2.get("n_resolved", 0) if q2 else 0
    skill = q2.get("brier_skill", 0.0) if q2 else 0.0
    clv = q2.get("clv_test") if q2 else None
    clv_mean = clv["mean"] if clv else None
    clv_p = clv["p"] if clv else None

    if n < n_min:
        verdict = "INCONCLUSIVE"
        conf = "Insufficient data"
        detail = (f"Only {n} resolved forecast(s) so far — need ≥{n_min} for any read, ≥{n_strong} "
                  f"for confidence. The forecast log accumulates daily; check back as matches resolve.")
    else:
        # Gate each signal on MEANINGFUL evidence, not just sign — a +0.06¢ CLV at p=0.70 is not
        # a "win", and a Brier skill inside noise is not a win.
        beats_brier = skill > 0.002
        beats_clv = (clv_mean is not None and clv_mean > 0 and clv_p is not None and clv_p < 0.20)
        sig_clv = clv_p is not None and clv_p < 0.10
        # Is the model clearly LOSING? (market sharper on Brier by a real margin)
        losing_brier = skill < -0.002
        if beats_brier and beats_clv and sig_clv:
            verdict = "EDGE FOUND"
            conf = "High" if (n >= n_strong and clv_p < 0.05) else "Medium"
            detail = ("Model beats the market on Brier AND shows positive, significant closing-line "
                      "value. This is the signature of genuine edge.")
        elif beats_brier or beats_clv:
            verdict = "PROMISING"
            conf = "Low–Medium"
            detail = ("Mixed evidence: the model shows a real positive signal on one of {Brier, CLV} "
                      "but not both. Suggestive of edge but not conclusive — keep gathering data.")
        else:
            verdict = "NO EDGE"
            conf = "Medium" if n >= n_strong else "Low"
            if losing_brier:
                clv_str = f"p={clv_p:.2f}" if clv_p is not None else "n/a"
                detail = (f"On {n} resolved forecasts the **market is sharper than the model** "
                          f"(Brier skill {skill:+.3f}) and closing-line value is flat ({clv_str}). "
                          "No demonstrated edge — the picks are the model disagreeing with a market "
                          "that has been more right.")
            else:
                detail = ("On the resolved sample the model does not beat the market. Either the model "
                          "lacks edge or the market is efficient on these contracts.")

    return {"verdict": verdict, "confidence": conf, "detail": detail,
            "n_resolved": n, "n_min": n_min, "n_strong": n_strong,
            "brier_skill": skill, "clv_mean": clv_mean, "clv_p": clv_p}
