"""
Tests for MHD solver: convergence guarantees, normalization, and physics hierarchy.

Test 1 (Exact Convergence):
    step_layered with ALL patches at max_depth (local_factor=1)
    must produce the SAME result as step_full, up to float precision.

Test 2 (Normalization):
    The patch cost metric (area / local_factor^2) must give sensible values.

Test 3 (Physics Hierarchy):
    Deeper patches (higher depth) must give more accurate physics than shallower ones.

Test 4 (compute_local_factor unit tests):
    The shared helper must behave correctly at all edge cases.
"""

import sys
import os
from math import log

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.utils import compute_local_factor


# ═══════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════

def _make_full_coverage_patches(N, depth, target_dim=2):
    """
    Generate non-overlapping patches that tile the full NxN domain,
    all at the given depth. Patch size is chosen to be compatible
    with the local_factor at that depth.
    """
    min_size = 6  # Minimum patch size to avoid degenerate cases
    max_depth = int(log(N/min_size) / log(target_dim)) + 1
    lf = compute_local_factor(N, N, depth, max_depth, target_dim)

    # Patch size must be divisible by lf and tile the domain
    # Use a reasonable size: at least 2*lf to have meaningful computation
    patch_size = max(lf * 4, lf)
    # Ensure it divides N
    while N % patch_size != 0 and patch_size < N:
        patch_size += lf
    if N % patch_size != 0:
        patch_size = N

    patches = []
    for y0 in range(0, N, patch_size):
        for x0 in range(0, N, patch_size):
            y1 = min(y0 + patch_size, N)
            x1 = min(x0 + patch_size, N)
            patches.append({
                'bounds': (y0, y1, x0, x1),
                'depth': depth,
                'type': 'leaf_depth'
            })

    return patches, max_depth


def _make_all_coarse_leaf_patches(N, target_dim=2):
    """All patches are coarse_leaf (calm regions)."""
    min_size = 6
    max_depth = int(log(N/min_size) / log(target_dim)) + 1
    step = N // target_dim
    patches = []
    for y0 in range(0, N, step):
        for x0 in range(0, N, step):
            patches.append({
                'bounds': (y0, min(y0 + step, N), x0, min(x0 + step, N)),
                'depth': 1,
                'type': 'coarse_leaf'
            })
    return patches, max_depth


def _run_comparison(N, patches, max_depth, n_steps=1, target_dim=2, ):
    """
    Run both step_full and step_layered for n_steps on identical KH initial conditions.
    Returns (sim_full, sim_layered).
    """
    grid = PeriodicGrid(resolution_N=N)

    sim_full = MHDSolver(grid, dt=1e-4, Re=100, Rm=100)
    sim_full.init_kelvin_helmholtz()

    sim_layered = MHDSolver(grid, dt=1e-4, Re=100, Rm=100)
    sim_layered.init_kelvin_helmholtz()

    for _ in range(n_steps):
        sim_full.step_full(record_stats=False)
        sim_layered.step_layered(
            patches, max_depth=max_depth,
            target_dim=target_dim
        )

    return sim_full, sim_layered


def _max_field_diff(sim_a, sim_b):
    """Max absolute difference across all 4 MHD fields."""
    diffs = []
    for field in ['vx', 'vy', 'Bx', 'By']:
        diffs.append(np.max(np.abs(getattr(sim_a, field) - getattr(sim_b, field))))
    return max(diffs)


def _l2_error(sim_full, sim_layered):
    """Total relative L2 error across all fields."""
    total = 0.0
    for field in ['vx', 'vy', 'Bx', 'By']:
        f = getattr(sim_full, field).flatten()
        l = getattr(sim_layered, field).flatten()
        ref = np.linalg.norm(f)
        total += np.linalg.norm(f - l) / (ref + 1e-12)
    return total


# ═══════════════════════════════════════════════════
#  Test 1: Exact Convergence
# ═══════════════════════════════════════════════════

class TestExactConvergence:
    """
    When ALL patches are active at max_depth (local_factor=1),
    step_layered must produce EXACTLY the same result as step_full.
    """

    def test_single_step_convergence(self):
        """KH instability: step_layered == step_full after 1 step."""
        N = 256
        target_dim = 2
        min_size = 6
        
        max_depth = int(log(N/min_size) / log(target_dim)) + 1

        patches, max_depth = _make_full_coverage_patches(
            N, depth=max_depth, target_dim=target_dim
        )
        # Verify all patches have local_factor=1
        for p in patches:
            y0, y1, x0, x1 = p['bounds']
            lf = compute_local_factor(y1 - y0, x1 - x0, p['depth'], max_depth,
                                      target_dim)
            assert lf == 1, f"Expected lf=1 at max_depth, got {lf}"

        sim_full, sim_layered = _run_comparison(
            N, patches, max_depth, n_steps=1, target_dim=target_dim
        )

        tol = 1e-10
        for field in ['vx', 'vy', 'Bx', 'By']:
            f_full = getattr(sim_full, field)
            f_lay = getattr(sim_layered, field)
            diff = np.max(np.abs(f_full - f_lay))
            assert diff < tol, \
                f"{field} max diff = {diff:.2e} (tol={tol:.0e})"

    def test_multi_step_convergence(self):
        """Convergence holds over 5 time steps (accumulated float error)."""
        N = 256
        target_dim = 2
        min_size = 6
        
        n_steps = 5
        max_depth = int(log(N/min_size) / log(target_dim)) + 1

        patches, max_depth = _make_full_coverage_patches(
            N, depth=max_depth, target_dim=target_dim
        )

        sim_full, sim_layered = _run_comparison(
            N, patches, max_depth, n_steps=n_steps, target_dim=target_dim
        )

        tol = 1e-8
        for field in ['vx', 'vy', 'Bx', 'By']:
            f_full = getattr(sim_full, field)
            f_lay = getattr(sim_layered, field)
            diff = np.max(np.abs(f_full - f_lay))
            assert diff < tol, \
                f"{field} max diff after {n_steps} steps = {diff:.2e} (tol={tol:.0e})"

    def test_convergence_larger_grid(self):
        """Convergence on a 64x64 grid."""
        N = 64
        target_dim = 2
        min_size = 6
        max_depth = int(log(N/min_size) / log(target_dim)) + 1

        patches, max_depth = _make_full_coverage_patches(
            N, depth=max_depth, target_dim=target_dim
        )

        sim_full, sim_layered = _run_comparison(
            N, patches, max_depth, n_steps=1, target_dim=target_dim
        )

        tol = 1e-10
        diff = _max_field_diff(sim_full, sim_layered)
        assert diff < tol, f"Max diff on 64x64 = {diff:.2e}"


# ═══════════════════════════════════════════════════
#  Test 2: Normalization
# ═══════════════════════════════════════════════════

class TestNormalization:
    """Verify patch cost metric produces correct values."""

    def _compute_ratio(self, N, patches, max_depth, target_dim=2, ):
        total = 0.0
        for p in patches:
            y0, y1, x0, x1 = p['bounds']
            H, W = y1 - y0, x1 - x0
            lf = compute_local_factor(H, W, p['depth'], max_depth, target_dim)
            total += (H * W) / (lf ** 2)
        return total / (N ** 2)

    def test_full_coverage_max_depth(self):
        """All patches at max_depth (lf=1): ratio = 1.0."""
        N = 256
        min_size = 6
        max_depth = int(log(N/min_size) / log(2)) + 1
        patches, max_depth = _make_full_coverage_patches(N, depth=max_depth)
        ratio = self._compute_ratio(N, patches, max_depth)
        assert abs(ratio - 1.0) < 1e-10, f"Expected 1.0, got {ratio}"


    def test_coarse_leaf_patches_have_cost(self):
        """
        Coarse_leaf patches DO have a computational cost
        (they are computed in the solver at their depth's resolution).
        """
        N = 256
        patches, max_depth = _make_all_coarse_leaf_patches(N)
        ratio = self._compute_ratio(N, patches, max_depth)
        assert ratio > 0, f"Expected positive ratio, got {ratio}"

# ═══════════════════════════════════════════════════
#  Test 3: Physics Hierarchy
# ═══════════════════════════════════════════════════

class TestPhysicsHierarchy:
    """Deeper patches give more accurate physics than shallower ones."""

    def test_deeper_reduces_error(self):
        """
        Compare step_layered at different depths against step_full (DNS).
        Error should decrease (or stay equal) as depth increases.
        """
        N = 256
        target_dim = 2
        min_size = 6
        
        n_steps = 3
        max_depth = int(log(N/min_size) / log(target_dim)) + 1

        # Test depths that give distinct local_factors
        # With N=256, max_depth=6:
        #   depth 0-4: lf=4, depth 5: lf=2, depth 6: lf=1
        errors = {}
        for depth in [0, max_depth - 1, max_depth]:
            patches, _ = _make_full_coverage_patches(
                N, depth=depth, target_dim=target_dim
            )
            sim_full, sim_layered = _run_comparison(
                N, patches, max_depth, n_steps=n_steps,
                target_dim=target_dim
            )
            errors[depth] = _l2_error(sim_full, sim_layered)

        # Monotonic decrease: deeper should be at least as good
        depths_sorted = sorted(errors.keys())
        for i in range(len(depths_sorted) - 1):
            d_shallow = depths_sorted[i]
            d_deep = depths_sorted[i + 1]
            assert errors[d_deep] <= errors[d_shallow] + 1e-12, \
                f"Error did not decrease: depth {d_shallow}={errors[d_shallow]:.2e}, " \
                f"depth {d_deep}={errors[d_deep]:.2e}"

        # Error at max_depth should be near zero (convergence guarantee)
        assert errors[max_depth] < 1e-8, \
            f"Error at max_depth should be ~0, got {errors[max_depth]:.2e}"

    def test_max_depth_is_exact(self):
        """Error at max_depth must be essentially zero."""
        N = 256
        target_dim = 2
        min_size = 6
        max_depth = int(log(N/min_size) / log(target_dim)) + 1

        patches, _ = _make_full_coverage_patches(
            N, depth=max_depth, target_dim=target_dim
        )
        sim_full, sim_layered = _run_comparison(
            N, patches, max_depth, n_steps=1, target_dim=target_dim
        )
        err = _l2_error(sim_full, sim_layered)
        assert err < 1e-10, f"Max-depth error = {err:.2e}, expected ~0"


# ═══════════════════════════════════════════════════
#  Test 4: compute_local_factor
# ═══════════════════════════════════════════════════

class TestComputeLocalFactor:
    """Unit tests for the shared helper."""

    def test_max_depth_gives_1(self):
        assert compute_local_factor(256, 256, depth=6, max_depth=6,
                                    target_dim=2) == 1

    def test_intermediate_depth(self):
        """Depth max_depth-1 with target_dim=2 gives lf=2."""
        lf = compute_local_factor(256, 256, depth=5, max_depth=6,
                                  target_dim=2)
        assert lf == 2

    def test_minimum_is_1(self):
        """local_factor never goes below 1."""
        lf = compute_local_factor(3, 3, depth=0, max_depth=5,
                                  target_dim=2)
        assert lf >= 1

    def test_monotonic_with_depth(self):
        """local_factor is non-increasing with depth."""
        max_depth = 6
        factors = []
        for d in range(max_depth + 1):
            lf = compute_local_factor(256, 256, depth=d, max_depth=max_depth,
                                      target_dim=2)
            factors.append(lf)

        for i in range(len(factors) - 1):
            assert factors[i] >= factors[i + 1], \
                f"Not monotonic: depth {i}→{factors[i]}, depth {i+1}→{factors[i+1]}"


# ═══════════════════════════════════════════════════
#  Test 5: Corrected max_depth formula
# ═══════════════════════════════════════════════════

def _make_pipeline_patches_recursive(y0, y1, x0, x1, depth, VQA_N, max_depth, min_size):
    """Simulate the VQA recursive subdivision with threshold=0 (all active)."""
    patches = []
    H, W = y1 - y0, x1 - x0
    if H < min_size or W < min_size:
        patches.append({'bounds': (y0, y1, x0, x1), 'depth': depth, 'type': 'leaf_limit'})
        return patches
    if depth >= max_depth:
        patches.append({'bounds': (y0, y1, x0, x1), 'depth': depth, 'type': 'leaf_depth'})
        return patches
    step_y = H // VQA_N
    step_x = W // VQA_N
    for i in range(VQA_N):
        for j in range(VQA_N):
            sy = y0 + i * step_y
            ey = y0 + (i + 1) * step_y if i < VQA_N - 1 else y1
            sx = x0 + j * step_x
            ex = x0 + (j + 1) * step_x if j < VQA_N - 1 else x1
            patches.extend(_make_pipeline_patches_recursive(
                sy, ey, sx, ex, depth + 1, VQA_N, max_depth, min_size))
    return patches


class TestCorrectedMaxDepth:
    """
    The corrected max_depth = int(log(N/min_size)/log(VQA_N)) ensures that
    the deepest patches from the VQA recursion have local_factor=1 (DNS).
    """

    def test_deepest_patches_have_lf1(self):
        """With corrected max_depth, deepest pipeline patches get local_factor=1."""
        for N in [256, 512, 1024]:
            VQA_N = 2
            min_size = 6
            
            max_depth = max(1, int(log(N / min_size) / log(VQA_N)))
            patches = _make_pipeline_patches_recursive(0, N, 0, N, 0, VQA_N, max_depth, min_size)

            deepest = max(p['depth'] for p in patches)
            for p in patches:
                if p['depth'] == deepest:
                    y0, y1, x0, x1 = p['bounds']
                    lf = compute_local_factor(y1-y0, x1-x0, p['depth'], max_depth, VQA_N)
                    assert lf == 1, \
                        f"N={N}: deepest patch at depth={deepest} has lf={lf}, expected 1"

    def test_pipeline_convergence_n256(self):
        """Pipeline patches (threshold=0) with corrected max_depth converge to step_full."""
        N = 256
        VQA_N = 2
        
        min_size = 6
        max_depth = max(1, int(log(N / min_size) / log(VQA_N)))
        patches = _make_pipeline_patches_recursive(0, N, 0, N, 0, VQA_N, max_depth, min_size)

        sim_full, sim_layered = _run_comparison(
            N, patches, max_depth, n_steps=5, target_dim=VQA_N)

        tol = 1e-8
        diff = _max_field_diff(sim_full, sim_layered)
        assert diff < tol, f"Pipeline convergence failed: max diff={diff:.2e}"

    def test_pipeline_convergence_n64(self):
        """Pipeline patches (threshold=0) with corrected max_depth converge on 64x64."""
        N = 64
        VQA_N = 2
        
        min_size = 6
        max_depth = max(1, int(log(N / min_size) / log(VQA_N)))
        patches = _make_pipeline_patches_recursive(0, N, 0, N, 0, VQA_N, max_depth, min_size)

        sim_full, sim_layered = _run_comparison(
            N, patches, max_depth, n_steps=3, target_dim=VQA_N)

        tol = 1e-10
        diff = _max_field_diff(sim_full, sim_layered)
        assert diff < tol, f"N=64 pipeline convergence failed: max diff={diff:.2e}"


# ═══════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════

if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
