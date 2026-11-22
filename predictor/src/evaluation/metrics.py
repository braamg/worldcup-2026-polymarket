"""
Evaluation metrics for W/D/L outcome models.

All functions accept:
  proba   : (n, 3) float array — columns [P(H), P(D), P(A)]
  results : (n,) string array of 'H', 'D', 'A'

Lower is better for RPS, log-loss, Brier, and ECE.
"""

import numpy as np
import pandas as pd

_EPS = 1e-15    # clip floor for log-loss computation
RESULT_ENCODE = {'H': 0, 'D': 1, 'A': 2}


# ---------------------------------------------------------------------------
# Result encoding helpers
# ---------------------------------------------------------------------------

def _one_hot(results: np.ndarray) -> np.ndarray:
    """(n,) str array → (n, 3) one-hot: columns = [H, D, A]."""
    y = np.array([RESULT_ENCODE[r] for r in results])
    oh = np.zeros((len(y), 3), dtype=float)
    oh[np.arange(len(y)), y] = 1.0
    return oh


# ---------------------------------------------------------------------------
# Ranked Probability Score  (primary metric)
# ---------------------------------------------------------------------------

def rps_per_match(proba: np.ndarray, results: np.ndarray) -> np.ndarray:
    """Per-match Ranked Probability Score (n,). Mean of this is rps()."""
    oh = _one_hot(results)
    cum_pred   = np.cumsum(proba[:, :2], axis=1)   # (n, 2): [P(H), P(H)+P(D)]
    cum_actual = np.cumsum(oh[:, :2], axis=1)
    return 0.5 * np.sum((cum_pred - cum_actual) ** 2, axis=1)


def rps(proba: np.ndarray, results: np.ndarray) -> float:
    """
    Mean Ranked Probability Score for W/D/L ordered outcomes.

    RPS_i = 0.5 * [(cumP_H - cumA_H)^2 + (cumP_HD - cumA_HD)^2]

    where cumA = cumulative actual distribution for match i.
    """
    return float(np.mean(rps_per_match(proba, results)))


def paired_bootstrap_rps(
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    results: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap on per-match RPS to test whether model A really beats model B.

    The two models are scored on the SAME matches, so we resample match indices
    (not models) — this controls for shared match difficulty and is far more
    powerful than comparing independent mean RPS values.

    Returns a dict with:
      rps_a, rps_b   : point-estimate mean RPS for each model
      mean_diff      : mean(RPS_a - RPS_b). NEGATIVE ⇒ A better (lower RPS).
      ci95_low/high  : 95% bootstrap CI on mean_diff
      p_two_sided    : fraction of bootstrap diffs on the opposite side of 0
                       (≈ two-sided p-value that the true diff is 0)
      prob_a_better  : fraction of bootstrap resamples where A had lower RPS
    """
    ra = rps_per_match(proba_a, results)
    rb = rps_per_match(proba_b, results)
    diff = ra - rb
    n = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    mean_diff = float(diff.mean())
    frac_neg = float((boot_means < 0).mean())   # A better
    p_two_sided = float(2.0 * min(frac_neg, 1.0 - frac_neg))
    return {
        'rps_a':         float(ra.mean()),
        'rps_b':         float(rb.mean()),
        'mean_diff':     mean_diff,
        'ci95_low':      float(np.percentile(boot_means, 2.5)),
        'ci95_high':     float(np.percentile(boot_means, 97.5)),
        'p_two_sided':   p_two_sided,
        'prob_a_better': frac_neg,
    }


# ---------------------------------------------------------------------------
# Log-loss
# ---------------------------------------------------------------------------

def logloss(proba: np.ndarray, results: np.ndarray) -> float:
    """Mean negative log-likelihood of the actual outcome."""
    oh = _one_hot(results)
    p_clipped = np.clip(proba, _EPS, 1.0)
    return float(-np.mean(np.sum(oh * np.log(p_clipped), axis=1)))


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def brier(proba: np.ndarray, results: np.ndarray) -> float:
    """Mean sum-of-squared-errors across K=3 classes per match."""
    oh = _one_hot(results)
    return float(np.mean(np.sum((proba - oh) ** 2, axis=1)))


# ---------------------------------------------------------------------------
# Reliability curve  (for calibration / reliability diagrams)
# ---------------------------------------------------------------------------

def reliability_curve(
    p_pred: np.ndarray,
    y_binary: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute reliability curve data for one outcome class.

    Parameters
    ----------
    p_pred   : (n,) predicted probability for the class
    y_binary : (n,) binary: 1 if outcome == class, 0 otherwise
    n_bins   : number of equal-width probability bins

    Returns
    -------
    DataFrame with columns: bin_center, mean_pred, frac_pos, count
    """
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        rows.append({
            'bin_center': round((lo + hi) / 2, 2),
            'mean_pred':  float(p_pred[mask].mean()),
            'frac_pos':   float(y_binary[mask].mean()),
            'count':      cnt,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

def ece(proba: np.ndarray, results: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error over all three outcome classes.
    Weighted average of |mean_pred - frac_pos| within bins.
    """
    oh = _one_hot(results)
    total_ece = 0.0
    for k in range(3):
        rc = reliability_curve(proba[:, k], oh[:, k], n_bins=n_bins)
        if rc.empty:
            continue
        n_total = rc['count'].sum()
        total_ece += (rc['count'] / n_total * np.abs(rc['mean_pred'] - rc['frac_pos'])).sum()
    return float(total_ece / 3)


# ---------------------------------------------------------------------------
# Composite metrics report
# ---------------------------------------------------------------------------

def compute_metrics(proba: np.ndarray, results: np.ndarray) -> dict:
    return {
        'n':       len(results),
        'rps':     rps(proba, results),
        'logloss': logloss(proba, results),
        'brier':   brier(proba, results),
        'ece':     ece(proba, results),
    }


# ---------------------------------------------------------------------------
# Devig bookmaker odds  (used only when raw odds are available)
# ---------------------------------------------------------------------------

def devig_proba(raw_probs: np.ndarray) -> np.ndarray:
    """
    Remove overround from bookmaker implied probabilities.

    raw_probs : (n, 3) array of implied probs from raw decimal odds (1/odds).
    Returns   : (n, 3) normalised to sum to 1.
    """
    totals = raw_probs.sum(axis=1, keepdims=True)
    return raw_probs / totals
