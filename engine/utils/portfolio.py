"""
utils/portfolio.py
==================
Correlation-aware bundle analysis for the portfolio simulator, plus CLV helpers.

Why this exists
---------------
The old simulator computed "P(all win)" as the product of marginal probabilities, which
assumes the legs are independent. Tournament outcomes are NOT independent:

  - Nested legs on one team are perfectly nested: a team cannot win without reaching the
    final, cannot reach the final without reaching the semis, etc. Betting "Morocco to win"
    and "Morocco to reach SF" together is one leveraged Morocco view, not diversification.

We don't have the simulator's joint distribution exposed here (sim_probabilities.json holds
only marginals), but we can still model intra-team correlation *exactly* from the marginals
using one shared uniform draw per team:

    For team T draw u ~ U(0,1). Leg L on team T wins iff u < P(L).

Because deeper stages have smaller probabilities, u < P(win) automatically implies
u < P(reach final) — the nesting is enforced and every leg's marginal is preserved exactly.
Different teams are still treated as independent (a documented approximation: it ignores
mutual exclusivity in winner-style markets, which the marginals alone cannot recover).

This yields the full P&L distribution of the bundle, so we can report the *median* outcome
and the chance of profit — not just the rosy "if ALL win" figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ZAR_USD_DEFAULT = 18.5


def bounded_kelly_weights(
    kelly_fracs,
    kelly_tilt: float = 0.5,
    cap: float = 0.35,
) -> np.ndarray:
    """
    Stake weights blending an equal split with a Kelly-proportional split, capped.

        w = (1 − tilt)·(1/N)  +  tilt·(kelly_i / Σkelly)

    `kelly_tilt` 0 → pure equal (1/N); 1 → fully Kelly-proportional. The result is then
    iteratively capped at `cap` per leg (and renormalised) so a single overstated edge can't
    dominate the book — important because the largest apparent edges are the least trustworthy.
    Returns weights summing to 1.

    Rationale (see STRATEGY_REVIEW.md): for a small, non-replenishable bankroll the median and
    variance matter more than the mean. Kelly naturally down-weights extreme longshots, which
    raises the typical outcome and cuts volatility vs. an equal split; the cap + blend keep it
    robust to model error.
    """
    k = np.clip(np.asarray(kelly_fracs, dtype=float), 0.0, None)
    n = len(k)
    if n == 0:
        return np.array([])
    w_eq = np.full(n, 1.0 / n)
    w_k = k / k.sum() if k.sum() > 0 else w_eq.copy()
    w = (1.0 - kelly_tilt) * w_eq + kelly_tilt * w_k

    # Iteratively cap (only meaningful when the cap is below an equal slice).
    if cap * n > 1.0:
        for _ in range(20):
            over = w > cap
            if not over.any():
                break
            excess = (w[over] - cap).sum()
            w[over] = cap
            under = ~over
            if w[under].sum() <= 0:
                break
            w[under] += excess * w[under] / w[under].sum()
    return w / w.sum()


def find_redundant_legs(legs_df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Return {team: [selections]} for any team that appears in more than one leg.
    These legs are correlated (often nested) — flagged so the user knows the bundle
    is not as diversified as the leg count suggests.
    """
    teams = legs_df["selection"].map(lambda s: s.split(" to ")[0].strip())
    grouped: dict[str, list[str]] = {}
    for team, sel in zip(teams, legs_df["selection"]):
        grouped.setdefault(team, []).append(sel)
    return {t: sels for t, sels in grouped.items() if len(sels) > 1}


def simulate_bundle(
    legs_df: pd.DataFrame,
    stake_per_leg_usd: float,
    zar_usd: float = ZAR_USD_DEFAULT,
    prob_col: str = "model_prob",
    n_sims: int = 20_000,
    seed: int = 42,
    win_matrix: np.ndarray | None = None,
) -> dict:
    """
    Compute the P&L distribution of a bundle.

    Two modes:
      - EXACT (`win_matrix` provided): a boolean (n_sims, n_legs) array of the actual
        per-simulation outcomes from the joint Monte Carlo matrix. Full cross-team and
        intra-team correlation; no resampling. `prob_col` is ignored.
      - APPROXIMATE (default): resample with one shared uniform per team, which reproduces
        intra-team nesting exactly but treats different teams as independent.

    `stake_per_leg_usd` may be a scalar (equal split) or a per-leg array (e.g. the bounded
    Kelly blend). Each leg's stake is always spent; a winning leg returns
    stake × payout_multiplier. Returns a summary dict in ZAR plus raw profit samples.
    """
    if legs_df.empty:
        return {}

    n_legs = len(legs_df)
    teams  = legs_df["selection"].map(lambda s: s.split(" to ")[0].strip()).tolist()
    unique_teams = list(dict.fromkeys(teams))

    if win_matrix is not None:
        exact = True
        wins = win_matrix.astype(bool)
        n_sims = wins.shape[0]
    else:
        exact = False
        team_idx = {t: i for i, t in enumerate(unique_teams)}
        rng = np.random.default_rng(seed)
        u = rng.random((n_sims, len(unique_teams)))   # one uniform per team per trial
        wins = np.zeros((n_sims, n_legs), dtype=bool)
        for j in range(n_legs):
            p = float(legs_df.iloc[j][prob_col])
            wins[:, j] = u[:, team_idx[teams[j]]] < p

    stakes = np.asarray(stake_per_leg_usd, dtype=float)
    if stakes.ndim == 0:
        stakes = np.full(n_legs, float(stakes))
    payouts = legs_df["payout_multiplier"].to_numpy(dtype=float)
    total_stake_usd = float(stakes.sum())

    profit_usd = -total_stake_usd + (wins * (stakes * payouts)).sum(axis=1)
    profit_zar = profit_usd * zar_usd
    n_wins = wins.sum(axis=1)

    return {
        "exact":            exact,
        "n_sims":           int(n_sims),
        "n_legs":           n_legs,
        "n_teams":          len(unique_teams),
        "total_stake_zar":  round(total_stake_usd * zar_usd, 2),
        "mean_profit_zar":  round(float(profit_zar.mean()), 2),
        "median_profit_zar":round(float(np.median(profit_zar)), 2),
        "p10_profit_zar":   round(float(np.percentile(profit_zar, 10)), 2),
        "p90_profit_zar":   round(float(np.percentile(profit_zar, 90)), 2),
        "p_profit":         round(float((profit_zar > 0).mean()), 4),
        "p_all_win":        round(float((n_wins == n_legs).mean()), 4),
        "p_total_loss":     round(float((n_wins == 0).mean()), 4),
        "profit_samples_zar": profit_zar,
    }


def simulate_blend(
    fut_df: pd.DataFrame,
    mat_df: pd.DataFrame,
    fut_stakes,
    mat_stakes,
    fut_win_matrix: "np.ndarray | None" = None,
    n_sims: int = 200_000,
    seed: int = 42,
    prob_col: str = "model_shrunk",
    shrink_futures: bool = True,
) -> dict:
    """
    Full P&L distribution for a blend of FUTURES + single-MATCH bets, with PER-LEG stakes.

    `fut_stakes` / `mat_stakes` are ZAR stake arrays aligned to the rows of `fut_df` / `mat_df`
    (a scalar broadcasts to every leg). This lets the caller weight the bankroll any way it likes
    — equal, probability-weighted, Kelly — and see the distribution change accordingly.

    Futures use the joint tournament outcome matrix when supplied (`fut_win_matrix`, shape
    (N, n_fut), columns in fut_df row order) — so their cross-correlation is exact. Match legs
    are drawn as independent Bernoulli(used prob). If no futures matrix is given, futures are
    drawn independently too.

    CONSISTENCY: the matrix encodes the *raw* model's win frequencies, which are more optimistic
    than the shrunk ("used") probabilities the rest of the app bets on. With `shrink_futures=True`
    (default) each futures leg's matrix wins are *thinned* by ratio = p_shrunk / p_raw so its
    marginal matches the used probability — while the joint co-movement (nesting, same-group
    anti-correlation) is preserved. Set False to see the rosier raw-model distribution.

    Returns the key probabilities (lose everything / break even+ / max payout), summary stats,
    and the raw profit samples (ZAR) for a histogram.
    """
    n_fut, n_mat = len(fut_df), len(mat_df)
    n = n_fut + n_mat
    if n == 0:
        return {}

    fut_stakes = np.asarray(fut_stakes, dtype=float).reshape(-1)
    mat_stakes = np.asarray(mat_stakes, dtype=float).reshape(-1)

    N = fut_win_matrix.shape[0] if fut_win_matrix is not None else n_sims
    rng = np.random.default_rng(seed)
    gross = np.zeros(N)
    n_wins = np.zeros(N, dtype=int)
    max_gross = 0.0

    if n_fut:
        if fut_win_matrix is not None:
            fw = fut_win_matrix.astype(bool)
            if shrink_futures and "model_prob" in fut_df.columns:
                # Thin matrix wins so each leg's marginal matches the shrunk (used) probability,
                # preserving the joint correlation. ratio = p_shrunk / p_raw, clipped to (0, 1].
                p_raw = fut_df["model_prob"].to_numpy(dtype=float)
                p_shr = fut_df[prob_col].to_numpy(dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(p_raw > 0, p_shr / p_raw, 1.0)
                ratio = np.clip(ratio, 0.0, 1.0)
                fw = fw & (rng.random((N, n_fut)) < ratio)
        else:
            fw = rng.random((N, n_fut)) < fut_df[prob_col].to_numpy(dtype=float)
        payout_f = fut_df["payout_exec"].to_numpy(dtype=float)
        gross += (fw * (fut_stakes * payout_f)).sum(axis=1)
        n_wins += fw.sum(axis=1)
        max_gross += float((fut_stakes * payout_f).sum())

    if n_mat:
        mw = rng.random((N, n_mat)) < mat_df[prob_col].to_numpy(dtype=float)
        payout_m = mat_df["payout_exec"].to_numpy(dtype=float)
        gross += (mw * (mat_stakes * payout_m)).sum(axis=1)
        n_wins += mw.sum(axis=1)
        max_gross += float((mat_stakes * payout_m).sum())

    total_stake = float(fut_stakes.sum() + mat_stakes.sum())
    profit = gross - total_stake
    return {
        "n": n, "n_sims": int(N), "total_stake": round(total_stake, 2),
        "p_lose_all":   round(float((n_wins == 0).mean()), 4),
        "p_break_even": round(float((profit >= 0).mean()), 4),
        "p_profit":     round(float((profit > 0).mean()), 4),
        "p_max":        round(float((n_wins == n).mean()), 4),
        "max_profit":   round(max_gross - total_stake, 2),
        "mean":   round(float(profit.mean()), 2),
        "median": round(float(np.median(profit)), 2),
        "p10":    round(float(np.percentile(profit, 10)), 2),
        "p90":    round(float(np.percentile(profit, 90)), 2),
        "samples": profit,
    }


def clv_cents(entry_cents: float, closing_cents: float) -> float:
    """
    Closing-Line Value for a YES buy, in cents.

    Positive = the market moved toward your position after you bet (the price rose),
    which is the single most reliable signal that a bet had genuine edge.
    """
    return round(closing_cents - entry_cents, 2)


def clv_summary(settled: list[dict]) -> dict:
    """Aggregate CLV across settled bets that have a recorded closing price."""
    vals = [e["clv_cents"] for e in settled if e.get("clv_cents") is not None]
    if not vals:
        return {}
    arr = np.array(vals, dtype=float)
    return {
        "n":          len(arr),
        "mean_clv":   round(float(arr.mean()), 2),
        "pct_beat":   round(float((arr > 0).mean()) * 100, 1),
    }
