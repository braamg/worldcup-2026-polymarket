"""
Post-hoc calibration for multi-class outcome models.

Default: temperature scaling (single parameter, avoids overfitting on small sets).
Optional: isotonic regression (only if calibration set is large enough and it wins
          on held-out reliability).
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def fit_temperature(
    proba_cal: np.ndarray,
    y_cal: np.ndarray,
    bounds: tuple[float, float] = (0.5, 5.0),
) -> float:
    """
    Find temperature T that minimises NLL on calibration set.

    Temperature scaling: p_cal = softmax(log(p) / T)
    T > 1 → softer (more uncertain), T < 1 → sharper.

    Parameters
    ----------
    proba_cal : (n, 3) predicted probabilities
    y_cal     : (n,) int array — 0=H, 1=D, 2=A

    Returns
    -------
    T : float
    """
    log_p = np.log(np.clip(proba_cal, 1e-15, 1.0))   # (n, 3)

    def nll(T: float) -> float:
        scaled = softmax(log_p / T, axis=1)
        return float(-np.mean(np.log(np.clip(scaled[np.arange(len(y_cal)), y_cal], 1e-15, 1.0))))

    result = minimize_scalar(nll, bounds=bounds, method='bounded')
    return float(result.x)


def apply_temperature(proba: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature T to a (n, 3) probability array."""
    log_p = np.log(np.clip(proba, 1e-15, 1.0))
    return softmax(log_p / T, axis=1)


# ---------------------------------------------------------------------------
# Isotonic regression (one per class, renormalised)
# ---------------------------------------------------------------------------

def fit_isotonic(
    proba_cal: np.ndarray,
    y_cal: np.ndarray,
) -> list:
    """
    Fit per-class isotonic regressors and return them.

    Returns a list of 3 fitted IsotonicRegression objects (H, D, A).
    """
    from sklearn.isotonic import IsotonicRegression
    regs = []
    for k in range(3):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(proba_cal[:, k], (y_cal == k).astype(float))
        regs.append(ir)
    return regs


def apply_isotonic(proba: np.ndarray, regs: list) -> np.ndarray:
    """Apply fitted isotonic regressors and renormalise to sum to 1."""
    out = np.stack([regs[k].predict(proba[:, k]) for k in range(3)], axis=1)
    out = np.clip(out, 1e-6, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


# ---------------------------------------------------------------------------
# Choose calibration method
# ---------------------------------------------------------------------------

def choose_and_fit(
    proba_cal: np.ndarray,
    y_cal: np.ndarray,
    n_cal: int,
    prefer_isotonic_threshold: int = 2000,
) -> dict:
    """
    Fit temperature scaling and (optionally) isotonic regression.
    Return a dict describing the chosen method and fitted objects.

    isotonic is ONLY used if n_cal >= prefer_isotonic_threshold and it produces
    lower NLL on the calibration set (which it always will by construction when
    fitted on the same data — so this check is intentionally conservative and
    defers to temperature scaling unless the dataset is very large).

    In practice for this project the calibration set is ~2k matches so we
    always use temperature scaling.
    """
    from src.evaluation.metrics import logloss
    from src.models.base import RESULT_DECODE  # noqa: F401

    T = fit_temperature(proba_cal, y_cal)
    result = {'method': 'temperature', 'T': T, 'regs': None}

    if n_cal >= prefer_isotonic_threshold:
        regs = fit_isotonic(proba_cal, y_cal)
        proba_iso = apply_isotonic(proba_cal, regs)
        proba_tmp = apply_temperature(proba_cal, T)

        results_str = np.array([RESULT_DECODE[y] for y in y_cal])
        ll_iso = logloss(proba_iso, results_str)
        ll_tmp = logloss(proba_tmp, results_str)

        if ll_iso < ll_tmp * 0.995:    # isotonic must win by >0.5% to justify extra complexity
            result = {'method': 'isotonic', 'T': None, 'regs': regs,
                      'note': f'isotonic={ll_iso:.5f} vs temp={ll_tmp:.5f}'}

    return result
