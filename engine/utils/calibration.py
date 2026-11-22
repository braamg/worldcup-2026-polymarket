"""
utils/calibration.py
===================
Model validation against the market — the P0 of STRATEGY_REVIEW.md.

The strategy's entire premise is that the model's probabilities are sharper than the
market's prices. That is a testable claim. Given settled bets (each with a model
probability, the market-implied probability at bet time, and a binary outcome), we can
score BOTH and compare:

  - Brier score   = mean( (p − outcome)^2 )           lower is better
  - Log loss      = −mean( o·ln p + (1−o)·ln(1−p) )   lower is better
  - Brier skill   = brier_market − brier_model        >0 means the model beat the market

If, over a meaningful sample, the model cannot beat the market's Brier/log-loss (and cannot
produce positive closing-line value), the strategy has no demonstrated edge. This module is
the scoreboard for that question. It also buckets by price band to surface the
favourite–longshot bias specifically within the <40¢ zone the strategy targets.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

_EPS = 1e-9
_RESULTS_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "predictor" / "data" / "raw" / "wc2026_results.csv",
    Path(__file__).resolve().parents[1] / "data" / "raw" / "wc2026_results.csv",
]
_SHARP_FAIR_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "predictor" / "data" / "processed" / "sharp_fair.json",
]


def brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.clip(probs, _EPS, 1 - _EPS)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def reliability_table(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """Bin predictions and compare predicted probability to observed frequency."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "Bucket":      f"{edges[b]*100:.0f}–{edges[b+1]*100:.0f}%",
            "Predicted":   round(float(probs[mask].mean()) * 100, 1),
            "Observed":    round(float(outcomes[mask].mean()) * 100, 1),
            "Count":       int(mask.sum()),
        })
    return pd.DataFrame(rows)


def calibration_report(settled: list[dict]) -> dict:
    """
    Score the model and the market on settled bets.

    Each settled entry must carry: model_p (0–1), market_p (0–1), outcome ('Won'/'Lost').
    Returns scores, a model reliability table, and a price-bucket breakdown
    (favourite–longshot check). Returns {} if there is nothing scoreable.
    """
    rows = [
        e for e in settled
        if e.get("model_p") is not None
        and e.get("market_p") is not None
        and e.get("outcome") in ("Won", "Lost")
    ]
    if not rows:
        return {}

    model_p  = np.array([e["model_p"] for e in rows], dtype=float)
    market_p = np.array([e["market_p"] for e in rows], dtype=float)
    y        = np.array([1.0 if e["outcome"] == "Won" else 0.0 for e in rows])

    brier_model, brier_market = brier(model_p, y), brier(market_p, y)

    # Favourite–longshot check: model calibration within price bands.
    band_edges = [0, 0.10, 0.20, 0.30, 0.40, 1.01]
    band_labels = ["<10¢", "10–20¢", "20–30¢", "30–40¢", "≥40¢"]
    band_idx = np.clip(np.digitize(market_p, band_edges) - 1, 0, len(band_labels) - 1)
    band_rows = []
    for b, label in enumerate(band_labels):
        mask = band_idx == b
        if not mask.any():
            continue
        band_rows.append({
            "Price band":     label,
            "Model pred %":   round(float(model_p[mask].mean()) * 100, 1),
            "Observed %":     round(float(y[mask].mean()) * 100, 1),
            "Bets":           int(mask.sum()),
        })

    return {
        "n":             len(rows),
        "brier_model":   round(brier_model, 4),
        "brier_market":  round(brier_market, 4),
        "brier_skill":   round(brier_market - brier_model, 4),  # >0 → model beats market
        "logloss_model": round(log_loss(model_p, y), 4),
        "logloss_market":round(log_loss(market_p, y), 4),
        "reliability":   reliability_table(model_p, y),
        "by_price_band": pd.DataFrame(band_rows),
    }


# ---------------------------------------------------------------------------
# Trust calibration — find model_trust that minimises Brier on resolved data
# ---------------------------------------------------------------------------

def _load_match_results() -> pd.DataFrame:
    for p in _RESULTS_CANDIDATES:
        if p.exists():
            return pd.read_csv(p)
    return pd.DataFrame()


def load_sharp_fair() -> dict:
    """Load the model's current fair-value dict {market_id: prob} from sharp_fair.json."""
    for p in _SHARP_FAIR_CANDIDATES:
        if p.exists():
            try:
                with open(p) as f:
                    d = json.load(f)
                return d.get("fair", {})
            except Exception:
                pass
    return {}


def _build_calibration_dataset(match_probs_df: pd.DataFrame,
                                market_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Join match model probabilities with actual results to build a calibration dataset.

    Returns DataFrame[model_prob, market_prob, outcome, stage] where outcome is 0/1.
    market_prob is from `market_df` if supplied (live Polymarket prices) else NaN.
    stage: 'match' for 3-way moneylines, 'advance' for group-stage futures.
    """
    results = _load_match_results()
    if results.empty or match_probs_df.empty:
        return pd.DataFrame()

    # Canonical group_id: sorted team names, matching market_id prefix in match_probabilities
    results = results.copy()
    results["t1"] = results.apply(lambda r: min(r.team_a, r.team_b), axis=1)
    results["t2"] = results.apply(lambda r: max(r.team_a, r.team_b), axis=1)
    results["group_id"] = "match:" + results["t1"] + "|" + results["t2"]
    results["winner"] = results.apply(
        lambda r: r.team_a if r.goals_a > r.goals_b
                  else (r.team_b if r.goals_b > r.goals_a else "draw"), axis=1)

    rows = []
    for _, res in results.iterrows():
        gid = res["group_id"]
        block = match_probs_df[match_probs_df["group_id"] == gid]
        if block.empty:
            continue
        for _, mp in block.iterrows():
            sel = mp["selection"]
            # derive the team/outcome this row predicts
            if " to beat " in sel:
                predicted_winner = sel.split(" to beat ")[0].strip()
                outcome = 1.0 if res["winner"] == predicted_winner else 0.0
            elif "draw" in sel.lower():
                outcome = 1.0 if res["winner"] == "draw" else 0.0
            else:
                continue

            # market price if we have live data
            mkt_p = np.nan
            if market_df is not None and not market_df.empty and "market_id" in market_df.columns:
                mrow = market_df[market_df["market_id"] == mp["market_id"]]
                if not mrow.empty:
                    mkt_p = float(mrow["price"].iloc[0])

            rows.append({
                "selection":  sel,
                "model_prob": mp["model_prob"],
                "market_prob": mkt_p,
                "outcome":    outcome,
                "stage":      "match",
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def model_vs_market_positioning(
    match_probs_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Current snapshot: for each live market, compare model fair value vs Polymarket price.
    Returns a DataFrame useful for displaying systematic model biases (over/under confidence).
    """
    if match_probs_df is None or match_probs_df.empty:
        return pd.DataFrame()
    if market_df is None or market_df.empty:
        return pd.DataFrame()
    merged = match_probs_df.merge(
        market_df[["market_id", "price"]].rename(columns={"price": "market_price"}),
        on="market_id", how="inner",
    )
    if merged.empty:
        return pd.DataFrame()
    merged["model_edge"] = merged["model_prob"] - merged["market_price"]
    merged["type"] = merged["selection"].apply(
        lambda s: "draw" if "draw" in s.lower() else ("home/away win"))
    return merged[["selection", "model_prob", "market_price", "model_edge", "type",
                   "match_date"]].sort_values("model_edge", ascending=False)


_MIN_BRIER_GAIN = 0.003   # below this, the curve is effectively flat → don't recommend a change
_CURRENT_TRUST = 0.35


def calibrate_trust(
    observations: pd.DataFrame | None = None,
    ledger_settled: list[dict] | None = None,
    grid: np.ndarray | None = None,
    selection_haircut: float = 0.25,
) -> dict:
    """
    Sweep model_trust ∈ [0, 1] and find the value minimising Brier on CLEAN resolved data.

    Clean means each observation has a REAL, distinct market price captured pre-match (so the
    model-vs-market comparison is meaningful and contamination-free). Two sources:
      • `observations`: resolved rows from the forward forecast log — DataFrame with
        columns model_prob, market_prob, outcome (0/1). This is the contamination-free spine.
      • `ledger_settled`: settled ledger bets (model_p, market_p, outcome Won/Lost).

    Critically, rows where market_prob is missing OR equal to model_prob are DROPPED — they
    carry zero information about trust (the blend doesn't move with the trust knob), and counting
    them would both inflate n and bias the optimum. A near-flat curve (Brier gain below
    `_MIN_BRIER_GAIN`) is reported as "keep the default", never as a spurious raise/lower.

    Returns: trust_optimal, n_total, n_forecast, n_ledger, brier_curve, recommendation,
    reliable (n ≥ 15), brier_at_current, brier_at_optimal.
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 41)

    rows = []
    n_forecast = n_ledger = 0

    def _add(mp, kp, y):
        if mp is None or kp is None or y is None:
            return False
        try:
            mp, kp, y = float(mp), float(kp), float(y)
        except (TypeError, ValueError):
            return False
        if not (np.isfinite(mp) and np.isfinite(kp)):
            return False
        if abs(mp - kp) < 1e-6:            # market == model → trust knob is inert here
            return False
        rows.append({"model_prob": mp, "market_prob": kp, "outcome": y})
        return True

    if observations is not None and not observations.empty:
        for _, r in observations.iterrows():
            if _add(r.get("model_prob"), r.get("market_prob"), r.get("outcome")):
                n_forecast += 1

    if ledger_settled:
        for e in ledger_settled:
            if e.get("outcome") not in ("Won", "Lost"):
                continue
            y = 1.0 if e["outcome"] == "Won" else 0.0
            if _add(e.get("model_p"), e.get("market_p"), y):
                n_ledger += 1

    n_total = len(rows)
    if n_total == 0:
        return {
            "trust_optimal": _CURRENT_TRUST, "n_total": 0, "n_forecast": 0, "n_ledger": 0,
            "reliable": False, "brier_curve": None,
            "recommendation": (
                "No clean resolved comparisons yet (need a real pre-match market price AND the "
                "outcome). The forward forecast log supplies these as matches resolve — until then, "
                f"keep MODEL_TRUST = {_CURRENT_TRUST:.2f}. This is the contamination-free path: "
                "already-played matches are excluded because the live market price is gone and the "
                "model has seen the result."
            ),
        }

    df = pd.DataFrame(rows)
    model_p  = df["model_prob"].to_numpy(float)
    market_p = df["market_prob"].to_numpy(float)
    y        = df["outcome"].to_numpy(float)

    factor = 1.0 - selection_haircut
    curve_rows = []
    for t in grid:
        blend = np.clip(market_p + t * factor * (model_p - market_p), 0.0, 1.0)
        curve_rows.append({
            "trust":        round(float(t), 3),
            "brier_blend":  round(float(np.mean((blend - y) ** 2)), 5),
            "brier_market": round(float(np.mean((market_p - y) ** 2)), 5),
            "brier_model":  round(float(np.mean((model_p - y) ** 2)), 5),
        })
    curve = pd.DataFrame(curve_rows)
    best_trust = float(curve.loc[curve["brier_blend"].idxmin(), "trust"])
    brier_cur = float(curve.loc[(curve["trust"] - _CURRENT_TRUST).abs().idxmin(), "brier_blend"])
    brier_opt = float(curve["brier_blend"].min())
    gain = brier_cur - brier_opt
    reliable = n_total >= 15

    if not reliable:
        advice = (
            f"**{n_total} clean comparison(s)** so far — need ≥15 for a stable estimate. "
            f"Keep MODEL_TRUST = {_CURRENT_TRUST:.2f}. Early-look optimum {best_trust:.2f} is noise "
            "at this n; ignore it until the forecast log fills."
        )
    elif gain < _MIN_BRIER_GAIN:
        advice = (
            f"Curve is essentially flat over {n_total} comparisons (Brier gain only {gain:+.4f} at "
            f"the optimum). The data can't distinguish trust levels — **keep MODEL_TRUST = "
            f"{_CURRENT_TRUST:.2f}**. A boundary optimum like {best_trust:.2f} on a flat curve is an "
            "artifact, not a signal."
        )
    else:
        diff = best_trust - _CURRENT_TRUST
        if abs(diff) < 0.05:
            advice = (f"Calibration on {n_total} comparisons confirms the default: optimal trust ≈ "
                      f"**{best_trust:.2f}**. No change needed.")
        elif diff > 0:
            advice = (f"Over {n_total} comparisons the model genuinely beats the market (Brier gain "
                      f"{gain:.4f}). Optimal trust ≈ **{best_trust:.2f}** — 0.35 may be too "
                      f"conservative. Consider raising MODEL_TRUST.")
        else:
            advice = (f"Over {n_total} comparisons the market beats the model (Brier gain {gain:.4f}). "
                      f"Optimal trust ≈ **{best_trust:.2f}** — 0.35 over-trusts the model. "
                      f"Consider lowering MODEL_TRUST.")

    return {
        "trust_optimal": round(best_trust, 3),
        "n_total": n_total, "n_forecast": n_forecast, "n_ledger": n_ledger,
        "brier_curve": curve, "recommendation": advice, "reliable": reliable,
        "brier_at_current": brier_cur, "brier_at_optimal": brier_opt,
    }
