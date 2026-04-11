"""Unit tests that run without loading any real model.

Run with:
  python -m pytest tests/ -v
or simply:
  python tests/test_all.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Make src importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D
from src import stats as S
from src import vectors as V


# --- math: angular distance ----------------------------------------------

def test_angular_distance():
    assert D.angular_distance_deg(0, 0) == 0.0
    assert D.angular_distance_deg(0, 180) == 180.0
    assert D.angular_distance_deg(350, 10) == 20.0
    assert D.angular_distance_deg(10, 350) == 20.0
    assert D.angular_distance_deg(0, 360) == 0.0
    assert D.angular_distance_deg(720, 45) == 45.0
    print("PASS test_angular_distance")


def test_schwartz_angles():
    angles = D.schwartz_angles_deg()
    assert len(angles) == 19
    vals = list(angles.values())
    step = 360.0 / 19
    for i in range(1, 19):
        assert abs(vals[i] - vals[i - 1] - step) < 1e-9
    print("PASS test_schwartz_angles")


# --- math: cosine fit on synthetic perfect data ---------------------------

def test_cosine_fit_recovers_amplitude_and_phase():
    n = 19
    thetas = np.linspace(0, 360, n, endpoint=False)
    true_A, true_phi, true_C = 2.5, 73.0, -0.4
    ys = true_A * np.cos(np.radians(thetas - true_phi)) + true_C
    r2, A, phi = S.cosine_fit_r2(thetas, ys)
    assert abs(r2 - 1.0) < 1e-9
    assert abs(A - true_A) < 1e-6
    # phi recovery should be close (modulo numerical error around the wrap point)
    phi_err = D.angular_distance_deg(phi, true_phi)
    assert phi_err < 1e-3, f"phi error {phi_err}"
    print("PASS test_cosine_fit_recovers_amplitude_and_phase")


def test_cosine_fit_zero_on_pure_noise():
    rng = np.random.default_rng(0)
    thetas = np.linspace(0, 360, 19, endpoint=False)
    ys = rng.normal(size=19)
    r2, _, _ = S.cosine_fit_r2(thetas, ys)
    # On pure noise the R^2 should be small (definitely < 0.5).
    assert r2 < 0.5, f"r2 on noise = {r2}"
    print("PASS test_cosine_fit_zero_on_pure_noise")


def test_permutation_null_calibrated():
    """If we feed the permutation test pure noise, the p-value distribution
    should be roughly uniform on (0, 1). We just check it's not extreme."""
    rng = np.random.default_rng(0)
    thetas = np.linspace(0, 360, 19, endpoint=False)
    p_values = []
    for _ in range(50):
        ys = rng.normal(size=19)
        _, p, _ = S.permutation_p_value(thetas, ys, 200, rng)
        p_values.append(p)
    mean_p = float(np.mean(p_values))
    # Under the null, expected p ≈ 0.5; allow generous margin.
    assert 0.3 < mean_p < 0.7, f"mean p under null = {mean_p}"
    print(f"PASS test_permutation_null_calibrated (mean p = {mean_p:.3f})")


# --- math: subspace projection -------------------------------------------

def test_projection_onto_subspace_basic():
    rng = np.random.default_rng(0)
    d = 64
    # Build a 5-dim subspace via random orthonormal basis.
    basis_full = rng.normal(size=(5, d)).astype(np.float64)
    Q, _ = np.linalg.qr(basis_full.T)
    basis = Q.T[:5]                                          # (5, d), orthonormal rows

    # Vector entirely inside the subspace.
    coeffs = rng.normal(size=5)
    v_in = (coeffs[:, None] * basis).sum(axis=0)
    proj, frac = V.projection_onto_subspace(v_in, basis)
    assert abs(frac - 1.0) < 1e-6, f"in-subspace fraction {frac}"
    assert np.linalg.norm(proj - v_in) < 1e-6

    # Vector entirely orthogonal to the subspace.
    rand = rng.normal(size=d)
    v_out = rand - (basis.T @ (basis @ rand))                # Gram-Schmidt
    proj, frac = V.projection_onto_subspace(v_out, basis)
    assert frac < 1e-6, f"orth fraction {frac}"

    # 50/50 mix.
    v_mix = v_in / np.linalg.norm(v_in) + v_out / np.linalg.norm(v_out)
    _, frac = V.projection_onto_subspace(v_mix, basis)
    # Norm of mix is sqrt(2); norm of in-component is 1; ratio = 1/sqrt(2) ≈ 0.707
    assert abs(frac - 1.0 / math.sqrt(2)) < 1e-6
    print("PASS test_projection_onto_subspace_basic")


def test_circumplex_2d_synthetic_circle():
    """If the value vectors lie exactly on a 2D circle in some plane, recover it."""
    rng = np.random.default_rng(0)
    d = 32
    # Pick a random 2D plane.
    Q, _ = np.linalg.qr(rng.normal(size=(d, 2)))
    plane = Q[:, :2]                                         # (d, 2)
    n = 19
    angles_true = np.linspace(0, 360, n, endpoint=False)
    coords_2d = np.column_stack([
        np.cos(np.radians(angles_true)),
        np.sin(np.radians(angles_true)),
    ])
    vectors = coords_2d @ plane.T                            # (n, d)
    coords, angles = V.circumplex_2d(vectors)
    # The recovered angles should be a rotated/reflected copy of the true angles.
    # The pairwise angular gaps should still be ~ 360/19.
    gaps = np.diff(np.sort(angles))
    expected_gap = 360.0 / n
    assert np.allclose(gaps, expected_gap, atol=1e-6), f"gaps {gaps}"
    print("PASS test_circumplex_2d_synthetic_circle")


# --- bootstrap ------------------------------------------------------------

def test_bootstrap_ci_contains_true_mean():
    rng = np.random.default_rng(0)
    n_inside = 0
    n_trials = 100
    for _ in range(n_trials):
        x = rng.normal(loc=2.0, scale=1.0, size=200)
        point, lo, hi = S.bootstrap_ci(x, np.mean, n_bootstrap=200, ci=0.95, rng=rng)
        if lo <= 2.0 <= hi:
            n_inside += 1
    # Coverage should be near 0.95; allow generous margin.
    assert n_inside / n_trials >= 0.85, f"coverage {n_inside}/{n_trials}"
    print(f"PASS test_bootstrap_ci_contains_true_mean (coverage = {n_inside}/{n_trials})")


# --- steering hook (no transformers, stub model) -------------------------

def test_steering_hook_adds_exact_amount():
    """The steering hook must add alpha*v to the residual stream at every position,
    AND nothing else."""
    if not HAS_TORCH:
        print("SKIP test_steering_hook_adds_exact_amount (torch not installed)")
        return

    from src.steering import steer_layer
    from src.models import LoadedModel

    class _StubLayer(nn.Module):
        """Identity layer that returns a tuple, like a real decoder layer.
        Has a dummy parameter so layer_device() (which queries .parameters())
        can find a device."""
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))
        def forward(self, x):
            return (x,)

    d = 16
    layers = nn.ModuleList([_StubLayer(), _StubLayer(), _StubLayer()])

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = layers
            self.dummy = nn.Parameter(torch.zeros(1))

    wrapper = Wrapper()
    fake = LoadedModel(
        name="stub", model=wrapper, tokenizer=None, is_instruct=False,
        layers=wrapper.layers, hidden_size=d,
        input_device=torch.device("cpu"),
    )

    x = torch.randn(2, 5, d)                                 # (B=2, S=5, H=d)
    v = np.arange(d, dtype=np.float32) - d / 2               # nontrivial vector
    alpha = 1.5

    # Run forward through layer 1 with steering installed.
    with steer_layer(fake, layer_idx=1, vector=v, alpha=alpha):
        out_tuple = fake.layers[1](x)
    out = out_tuple[0]

    expected = x + alpha * torch.from_numpy(v).view(1, 1, d).to(x.dtype)
    assert torch.allclose(out, expected, atol=1e-6), (
        f"steering hook didn't add the expected amount\n"
        f"  max diff = {(out - expected).abs().max().item()}"
    )

    # And after exiting the context, the hook should be removed: another forward
    # should produce the unmodified input.
    out2 = fake.layers[1](x)[0]
    assert torch.allclose(out2, x, atol=1e-6), "hook was not removed on context exit"

    print("PASS test_steering_hook_adds_exact_amount")


# --- ValueEval label parity -----------------------------------------------

def test_pvq_first_person_coverage():
    assert set(D.PVQ_ITEMS_FIRST_PERSON.keys()) == set(D.SCHWARTZ_19)
    for v, items in D.PVQ_ITEMS_FIRST_PERSON.items():
        assert len(items) == 3, f"{v} has {len(items)} items"
        for it in items:
            tokens = it.lower().split()
            assert "she" not in tokens, f"3p pronoun in {it!r}"
            assert "her" not in tokens, f"3p pronoun in {it!r}"
            assert "herself" not in tokens, f"3p pronoun in {it!r}"
    print("PASS test_pvq_first_person_coverage")


# --- Holm-Bonferroni correction -------------------------------------------

def test_holm_bonferroni_all_reject():
    """All p-values well below alpha should all reject."""
    pvals = [0.001, 0.002, 0.003, 0.004, 0.005]
    rejects = S.holm_bonferroni(pvals, alpha=0.05)
    assert all(rejects), f"Expected all rejections: {rejects}"
    print("PASS test_holm_bonferroni_all_reject")


def test_holm_bonferroni_none_reject():
    """All p-values above alpha should all fail to reject."""
    pvals = [0.1, 0.2, 0.3, 0.4, 0.5]
    rejects = S.holm_bonferroni(pvals, alpha=0.05)
    assert not any(rejects), f"Expected no rejections: {rejects}"
    print("PASS test_holm_bonferroni_none_reject")


def test_holm_bonferroni_partial_reject():
    """Mixed p-values: only the smallest should survive correction."""
    pvals = [0.005, 0.01, 0.03, 0.06, 0.1]
    rejects = S.holm_bonferroni(pvals, alpha=0.05)
    # Sorted: 0.005 < 0.05/5=0.01 ✓, 0.01 < 0.05/4=0.0125 ✓,
    # 0.03 < 0.05/3=0.0167? No → stop
    assert rejects[0] and rejects[1], f"First two should reject: {rejects}"
    assert not rejects[2] and not rejects[3] and not rejects[4]
    print("PASS test_holm_bonferroni_partial_reject")


# --- Random baseline for projection ratio ----------------------------------

def test_random_projection_null_reasonable():
    """Random vectors projected onto a k-dim subspace of R^d should have
    expected ratio ~ sqrt(k/d) for orthonormal basis."""
    rng = np.random.default_rng(42)
    d = 1000
    k = 19
    # Build orthonormal basis
    Q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    basis = Q.T[:k]  # (k, d)
    ratios = S.random_projection_null(d, basis, n_random=500, rng=rng)
    expected = math.sqrt(k / d)  # ~ 0.138
    mean_ratio = float(ratios.mean())
    assert abs(mean_ratio - expected) < 0.03, (
        f"Mean random ratio {mean_ratio:.4f} far from expected {expected:.4f}"
    )
    print(f"PASS test_random_projection_null_reasonable (mean={mean_ratio:.4f}, "
          f"expected~{expected:.4f})")


# --- Circular diagnostics ---------------------------------------------------

def test_circular_variance_uniform():
    """Uniformly spaced angles should have high circular variance."""
    angles = np.linspace(0, 360, 19, endpoint=False)
    cv = S.circular_variance(angles)
    assert cv > 0.95, f"Uniform angles should have circ_var > 0.95, got {cv}"
    print(f"PASS test_circular_variance_uniform (cv={cv:.4f})")


def test_circular_variance_clustered():
    """All angles near 0 should have low circular variance."""
    angles = np.array([0, 1, 2, 3, 4, 355, 356, 357, 358, 359], dtype=float)
    cv = S.circular_variance(angles)
    assert cv < 0.1, f"Clustered angles should have circ_var < 0.1, got {cv}"
    print(f"PASS test_circular_variance_clustered (cv={cv:.4f})")


def test_angle_gap_uniformity_detects_clusters():
    """Bimodal distribution (0° and 180°) should detect few clusters with high CV."""
    angles = np.array([0, 1, 2, 3, 4, 180, 181, 182, 183, 184], dtype=float)
    info = S.angle_gap_uniformity(angles)
    assert info["n_clusters"] <= 4, f"Expected few clusters, got {info['n_clusters']}"
    assert info["gap_cv"] > 1.0, f"Expected high gap CV, got {info['gap_cv']}"
    # Key test: max gap should be >> expected gap (36 deg for 10 points)
    assert info["max_gap_deg"] > 100, f"Expected large max gap, got {info['max_gap_deg']}"
    print(f"PASS test_angle_gap_uniformity_detects_clusters (n={info['n_clusters']}, "
          f"cv={info['gap_cv']:.2f}, max_gap={info['max_gap_deg']:.0f}°)")


def test_checkpoint_roundtrip_and_resume():
    import os, tempfile
    from src.checkpoint import JSONLCheckpoint, save_array_atomic, load_array_if_exists

    with tempfile.TemporaryDirectory() as d:
        # Append + in-memory membership + items()
        c = JSONLCheckpoint(os.path.join(d, "test.jsonl"))
        c.append(["a", 1, 0.5], {"score": 2.5})
        c.append(["b", 2, 1.0], {"score": 3.0})
        assert ["a", 1, 0.5] in c
        assert ["nope"] not in c

        # Resume: a fresh JSONLCheckpoint on the same file sees both records.
        c2 = JSONLCheckpoint(os.path.join(d, "test.jsonl"))
        assert len(c2) == 2
        items = list(c2.items())
        assert items[0][1]["score"] == 2.5
        assert items[1][1]["score"] == 3.0

        # Partial-line recovery: append a garbage line and confirm loader truncates.
        with open(os.path.join(d, "test.jsonl"), "a") as f:
            f.write('{"key":"bad"')   # truncated JSON
        c3 = JSONLCheckpoint(os.path.join(d, "test.jsonl"))
        assert len(c3) == 2, f"truncation recovery failed: len={len(c3)}"

        # Atomic .npy save/load including stale-tmp handling.
        p = os.path.join(d, "arr.npy")
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        # Drop a stale .tmp before calling — should be overwritten cleanly.
        with open(p + ".tmp", "w") as f:
            f.write("garbage")
        save_array_atomic(p, arr)
        loaded = load_array_if_exists(p)
        assert np.array_equal(arr, loaded)
        assert load_array_if_exists(os.path.join(d, "nope.npy")) is None

    print("PASS test_checkpoint_roundtrip_and_resume")


if __name__ == "__main__":
    test_angular_distance()
    test_schwartz_angles()
    test_cosine_fit_recovers_amplitude_and_phase()
    test_cosine_fit_zero_on_pure_noise()
    test_permutation_null_calibrated()
    test_projection_onto_subspace_basic()
    test_circumplex_2d_synthetic_circle()
    test_bootstrap_ci_contains_true_mean()
    test_steering_hook_adds_exact_amount()
    test_pvq_first_person_coverage()
    test_holm_bonferroni_all_reject()
    test_holm_bonferroni_none_reject()
    test_holm_bonferroni_partial_reject()
    test_random_projection_null_reasonable()
    test_circular_variance_uniform()
    test_circular_variance_clustered()
    test_angle_gap_uniformity_detects_clusters()
    test_checkpoint_roundtrip_and_resume()
    print("\nAll tests passed.")
