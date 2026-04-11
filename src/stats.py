"""Statistics: bootstrap CIs, monotonicity tests, permutation nulls,
multiple-comparisons corrections, and circular distribution diagnostics."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy import stats as sps


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_bootstrap: int,
    ci: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return (point_estimate, lo, hi) for a 1-D array."""
    point = statistic(values)
    n = len(values)
    if n == 0:
        return point, math.nan, math.nan
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = statistic(values[idx])
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1.0 - alpha))
    return float(point), lo, hi


def linear_slope(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """OLS slope and intercept of ys on xs."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    res = sps.linregress(xs, ys)
    return float(res.slope), float(res.intercept)


def monotone_score(xs: np.ndarray, ys: np.ndarray) -> float:
    """Spearman correlation between xs and ys; +1 = perfectly monotone increasing."""
    if len(xs) < 2:
        return float("nan")
    return float(sps.spearmanr(xs, ys).statistic)


def cosine_fit_r2(thetas_deg: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Fit y_i = A * cos(theta_i - phi) + C and return (R2, A, phi_deg).

    Implementation: re-parameterize as y = a*cos(theta) + b*sin(theta) + c,
    fit by OLS, then recover A = sqrt(a^2+b^2), phi = atan2(b, a).
    """
    thetas = np.radians(np.asarray(thetas_deg, dtype=np.float64))
    ys = np.asarray(ys, dtype=np.float64)
    X = np.column_stack([np.cos(thetas), np.sin(thetas), np.ones_like(thetas)])
    coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
    a, b, c = coef
    pred = X @ coef
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    A = float(math.hypot(a, b))
    phi = float(math.degrees(math.atan2(b, a)) % 360.0)
    return r2, A, phi


def permutation_p_value(
    thetas_deg: np.ndarray,
    ys: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, np.ndarray]:
    """Compare observed cosine-fit R^2 against the null distribution from
    randomly shuffling the angle labels. Returns (observed_r2, p_value, null_r2_distribution).
    """
    obs_r2, _, _ = cosine_fit_r2(thetas_deg, ys)
    null_r2 = np.empty(n_permutations, dtype=np.float64)
    n = len(thetas_deg)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        null_r2[i], _, _ = cosine_fit_r2(thetas_deg[perm], ys)
    # One-sided: how often does the null match or exceed the observed fit?
    p = float((np.sum(null_r2 >= obs_r2) + 1) / (n_permutations + 1))
    return obs_r2, p, null_r2


# --- Multiple comparisons ---------------------------------------------------

def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction. Returns a boolean list: True if the
    corresponding test rejects at family-wise alpha."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejects = [False] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (m - rank)
        if p <= threshold:
            rejects[orig_idx] = True
        else:
            break  # step-down: once we fail to reject, stop
    return rejects


# --- Random baseline for subspace projection --------------------------------

def random_projection_null(
    hidden_dim: int,
    basis_rows: np.ndarray,
    n_random: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Project `n_random` random unit vectors onto the subspace spanned by
    `basis_rows` and return the distribution of ||proj||/||v|| ratios.

    This establishes the null: what projection ratio would a random direction
    (unrelated to values) achieve?"""
    from .vectors import projection_onto_subspace
    ratios = np.empty(n_random, dtype=np.float64)
    for i in range(n_random):
        v = rng.standard_normal(hidden_dim).astype(np.float64)
        v /= np.linalg.norm(v)
        _, ratios[i] = projection_onto_subspace(v, basis_rows)
    return ratios


# --- Circular distribution diagnostics --------------------------------------

def circular_variance(angles_deg: np.ndarray) -> float:
    """Circular variance in [0, 1]. 0 = all angles identical, 1 = maximally spread.
    For a perfect circumplex (uniformly spaced), variance should be high (~1)."""
    thetas = np.radians(np.asarray(angles_deg, dtype=np.float64))
    R = abs(np.mean(np.exp(1j * thetas)))
    return float(1.0 - R)


def angle_gap_uniformity(angles_deg: np.ndarray) -> dict:
    """Diagnose whether empirical angles are uniformly spaced around the circle
    (as a circumplex predicts) or clustered.

    Returns:
      - gap_cv: coefficient of variation of the angular gaps. CV=0 means
        perfectly uniform spacing; high CV means clustered.
      - n_clusters: number of clusters found by a simple gap-based heuristic
        (gaps > 2 * median gap are cluster boundaries).
      - max_gap_deg: largest angular gap.
    """
    sorted_a = np.sort(np.asarray(angles_deg, dtype=np.float64) % 360.0)
    n = len(sorted_a)
    gaps = np.empty(n)
    for i in range(n - 1):
        gaps[i] = sorted_a[i + 1] - sorted_a[i]
    gaps[n - 1] = (360.0 - sorted_a[-1]) + sorted_a[0]  # wrap-around gap

    mean_gap = gaps.mean()
    std_gap = gaps.std()
    cv = float(std_gap / mean_gap) if mean_gap > 0 else 0.0
    median_gap = float(np.median(gaps))
    n_clusters = int(np.sum(gaps > 2 * median_gap)) + 1 if median_gap > 0 else 1
    return {
        "gap_cv": cv,
        "n_clusters": n_clusters,
        "max_gap_deg": float(gaps.max()),
        "min_gap_deg": float(gaps.min()),
        "mean_gap_deg": float(mean_gap),
        "expected_gap_deg": 360.0 / n,
    }
