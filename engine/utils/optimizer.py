"""
utils/optimizer.py
==================
Correlation-aware blend optimiser.

The manual blend lets you weight stakes by hand (equal / probability / Kelly). Those heuristics
sit *inside* the achievable frontier — there are allocations that beat them on BOTH chance-of-profit
and expected return at once (verified empirically). This module searches the stake space directly,
over the SAME distribution the dashboard shows (futures from the thinned+correlated joint matrix,
matches independent on the shrunk probability), and returns the allocation that best meets a chosen
objective — doing both selection (which legs) and sizing (how much) in one step.

Approach: build a fixed per-R1 return matrix once (n_sims × n_legs), generate many candidate stake
vectors (dense Dirichlet + sparse subsets + the heuristics as anchors), and score every candidate by
matrix-multiply. Cheap metrics (chance of profit, expected return) are computed for all candidates;
the expensive ones (median, ruin) only for the winner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _return_matrix(picks: pd.DataFrame, fut_win_matrix, n_sims: int, seed: int,
                   shrink_futures: bool = True) -> np.ndarray:
    """Per-R1 payout matrix (n_sims × n_legs): payout_exec if the leg wins in that sim, else 0."""
    rng = np.random.default_rng(seed)
    n = len(picks)
    is_fut = (picks["bet_type"] == "Future").to_numpy()
    fut_pos = np.where(is_fut)[0]
    mat_pos = np.where(~is_fut)[0]

    if fut_win_matrix is not None and len(fut_pos):
        full = fut_win_matrix.shape[0]
        NS = min(n_sims, full)
        idx = rng.choice(full, size=NS, replace=False) if NS < full else np.arange(full)
    else:
        NS = n_sims

    wins = np.zeros((NS, n), dtype=bool)

    if len(fut_pos):
        p_shr = picks.loc[is_fut, "model_shrunk"].to_numpy(float)
        if fut_win_matrix is not None:
            fw = fut_win_matrix[idx].astype(bool)
            if shrink_futures and "model_prob" in picks.columns:
                p_raw = picks.loc[is_fut, "model_prob"].to_numpy(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(p_raw > 0, p_shr / p_raw, 1.0)
                fw = fw & (rng.random((NS, len(fut_pos))) < np.clip(ratio, 0.0, 1.0))
            wins[:, fut_pos] = fw
        else:
            wins[:, fut_pos] = rng.random((NS, len(fut_pos))) < p_shr

    if len(mat_pos):
        p_shr = picks.loc[~is_fut, "model_shrunk"].to_numpy(float)
        wins[:, mat_pos] = rng.random((NS, len(mat_pos))) < p_shr

    return wins * picks["payout_exec"].to_numpy(float)


def _pareto_upper(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices on the upper-right Pareto frontier of (x, y) — both maximised."""
    order = np.argsort(-x, kind="stable")
    keep, ymax = [], -np.inf
    for i in order:
        if y[i] >= ymax:
            keep.append(i)
            ymax = y[i]
    return np.array(keep, dtype=int)


def _cap_weights(W: np.ndarray, cap: float, iters: int = 4) -> np.ndarray:
    """Clip each weight to `cap` and renormalise (a few passes ≈ water-filling)."""
    if cap >= 1.0:
        return W
    for _ in range(iters):
        W = np.minimum(W, cap)
        s = W.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        W = W / s
    return W


def optimize_blend(
    picks: pd.DataFrame,
    fut_win_matrix,
    bankroll: float,
    objective: str = "max_pprofit",       # max_pprofit | max_return_floor | max_median
    min_pprofit: float = 0.60,
    max_leg_frac: float = 0.60,
    n_sims: int = 25_000,
    n_candidates: int = 6_000,
    seed: int = 42,
) -> dict:
    """
    Search stake vectors to optimise `objective`. Returns the chosen allocation, its metrics,
    the Pareto frontier, and a candidate cloud (expected-return %, chance-of-profit %) for plotting.
    """
    picks = picks.reset_index(drop=True)
    n = len(picks)
    if n == 0:
        return {}

    ret1 = _return_matrix(picks, fut_win_matrix, n_sims, seed)
    NS = ret1.shape[0]
    ev = picks["model_shrunk"].to_numpy(float) * picks["payout_exec"].to_numpy(float) - 1.0

    rng = np.random.default_rng(seed + 1)
    half = max(1, n_candidates // 2)
    # sparse candidates need enough legs for the per-leg cap to be satisfiable
    kmin = min(n, max(1, int(np.ceil(1.0 / max_leg_frac)))) if max_leg_frac > 0 else 1
    dense = rng.dirichlet(np.ones(n), size=half)
    sparse = np.zeros((half, n))
    for i in range(half):
        k = int(rng.integers(kmin, n + 1)) if kmin < n else n
        cols = rng.choice(n, size=k, replace=False)
        sparse[i, cols] = rng.dirichlet(np.ones(k))
    # heuristic anchors: equal, probability-weighted, EV-weighted
    anchors = [np.ones(n) / n, picks["model_shrunk"].to_numpy(float)]
    evp = np.clip(ev, 0, None)
    anchors.append(evp / evp.sum() if evp.sum() > 0 else np.ones(n) / n)
    anchors = np.array([a / a.sum() for a in anchors])

    W = _cap_weights(np.vstack([dense, sparse, anchors]), max_leg_frac)
    if max_leg_frac < 1.0:                       # drop candidates the cap can't satisfy (e.g. 1-hot)
        keep = W.max(axis=1) <= max_leg_frac + 1e-9
        if keep.sum() >= 50:
            W = W[keep]
    stakes = bankroll * W                       # K × n
    gross = ret1 @ stakes.T                      # NS × K
    profit = gross - bankroll
    pprofit = (profit >= 0).mean(axis=0)         # cheap
    expret = W @ ev                              # cheap (fraction of bankroll)

    feasible_any = True
    if objective == "max_return_floor":
        feasible = pprofit >= min_pprofit
        if feasible.any():
            ci = np.where(feasible)[0]
            best = ci[np.argmax(expret[ci])]
        else:
            feasible_any = False
            best = int(np.argmax(pprofit))       # can't meet floor → show closest
    elif objective == "max_median":
        sub = profit[:min(6000, NS)]             # rank on a sims-subsample (median is slow on full)
        best = int(np.argmax(np.median(sub, axis=0)))
    else:  # max_pprofit (tiny return tie-break)
        best = int(np.argmax(pprofit + 1e-4 * expret))

    bg = gross[:, best] - bankroll
    chosen = {
        "weights": W[best], "stakes": stakes[best],
        "pprofit": float(pprofit[best]), "expret": float(expret[best]),
        "median": float(np.median(bg)), "ploseall": float((gross[:, best] <= 1e-9).mean()),
        "max_profit": float((stakes[best] * picks["payout_exec"].to_numpy(float)).sum() - bankroll),
    }

    front = _pareto_upper(expret, pprofit)
    cloud_n = min(1500, len(expret))
    cidx = rng.choice(len(expret), size=cloud_n, replace=False)
    return {
        "picks": picks, "chosen": chosen, "n_sims": NS, "feasible_any": feasible_any,
        "frontier": {"return_pct": expret[front] * 100, "pprofit_pct": pprofit[front] * 100},
        "cloud": {"return_pct": expret[cidx] * 100, "pprofit_pct": pprofit[cidx] * 100},
    }
