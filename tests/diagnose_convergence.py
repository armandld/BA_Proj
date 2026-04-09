"""
Diagnostic: Does step_layered converge to step_full when given ideal patches?
Tests the tau-correction solver and the max_depth calibration.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from math import log
from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.utils import compute_local_factor


def run_comparison(N, target_dim, patches, max_depth, n_steps=1, label=""):
    """Run both solvers n_steps and compare."""
    grid = PeriodicGrid(resolution_N=N)

    sim_full = MHDSolver(grid, dt=1e-4, Re=200, Rm=200)
    sim_full.init_kelvin_helmholtz()

    sim_layered = MHDSolver(grid, dt=1e-4, Re=200, Rm=200)
    sim_layered.init_kelvin_helmholtz()

    assert np.array_equal(sim_full.vx, sim_layered.vx), "Initial vx mismatch!"

    for step in range(n_steps):
        sim_full.step_full(record_stats=False)
        sim_layered.step_layered(patches, max_depth=max_depth,
                                  target_dim=target_dim)

    fields = ['vx', 'vy', 'Bx', 'By']
    max_diffs = {}
    for f in fields:
        diff = np.max(np.abs(getattr(sim_full, f) - getattr(sim_layered, f)))
        ref = np.max(np.abs(getattr(sim_full, f)))
        max_diffs[f] = (diff, diff / (ref + 1e-15))

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  N={N}, target_dim={target_dim}, max_depth={max_depth}")
    print(f"  n_patches={len(patches)}, n_steps={n_steps}")
    if patches:
        p0 = patches[0]
        lf = compute_local_factor(
            p0['bounds'][1]-p0['bounds'][0],
            p0['bounds'][3]-p0['bounds'][2],
            p0['depth'], max_depth, target_dim)
        print(f"  patch depth={p0['depth']}, local_factor={lf}")
        print(f"  patch size={p0['bounds'][1]-p0['bounds'][0]}x{p0['bounds'][3]-p0['bounds'][2]}")
    print(f"  {'Field':<6} {'Max Abs Diff':<20} {'Relative Diff':<20}")
    print(f"  {'-'*46}")
    all_ok = True
    for f in fields:
        abs_d, rel_d = max_diffs[f]
        status = "OK" if abs_d < 1e-10 else "MISMATCH"
        if abs_d >= 1e-10:
            all_ok = False
        print(f"  {f:<6} {abs_d:<20.2e} {rel_d:<20.2e} {status}")

    return all_ok, max_diffs


def make_pipeline_patches(N, VQA_N, max_depth, min_size):
    """Simulate what the pipeline produces with threshold=0 (all active)."""
    patches = []
    _recurse(patches, 0, N, 0, N, 0, VQA_N, max_depth, min_size)
    return patches


def _recurse(patches, y0, y1, x0, x1, depth, VQA_N, max_depth, min_size):
    H, W = y1 - y0, x1 - x0
    if H < min_size or W < min_size:
        patches.append({'bounds': (y0, y1, x0, x1), 'depth': depth, 'type': 'leaf_limit'})
        return
    if depth >= max_depth:
        patches.append({'bounds': (y0, y1, x0, x1), 'depth': depth, 'type': 'leaf_depth'})
        return
    step_y = H // VQA_N
    step_x = W // VQA_N
    for i in range(VQA_N):
        for j in range(VQA_N):
            sy = y0 + i * step_y
            ey = y0 + (i + 1) * step_y if i < VQA_N - 1 else y1
            sx = x0 + j * step_x
            ex = x0 + (j + 1) * step_x if j < VQA_N - 1 else x1
            _recurse(patches, sy, ey, sx, ex, depth + 1, VQA_N, max_depth, min_size)


def show_max_depth_comparison(N, VQA_N, min_size):
    old_md = int(log(N) / log(VQA_N)) + 1
    new_md = max(1, int(log(N / min_size) / log(VQA_N)))

    print(f"\n{'#'*60}")
    print(f"# max_depth comparison for N={N}, VQA_N={VQA_N}, min_size={min_size}")
    print(f"{'#'*60}")
    print(f"  Old max_depth = int(log({N})/log({VQA_N})) + 1 = {old_md}")
    print(f"  New max_depth = max(1, int(log({N}/{min_size})/log({VQA_N}))) = {new_md}")
    print()

    # Show patch info for both
    for label, md in [("OLD", old_md), ("NEW", new_md)]:
        patches = make_pipeline_patches(N, VQA_N, md, min_size)
        depths = [p['depth'] for p in patches]
        max_d = max(depths)
        p_at_max = [p for p in patches if p['depth'] == max_d]
        H = p_at_max[0]['bounds'][1] - p_at_max[0]['bounds'][0]
        W = p_at_max[0]['bounds'][3] - p_at_max[0]['bounds'][2]
        lf = compute_local_factor(H, W, max_d, md, VQA_N)
        print(f"  {label} (max_depth={md}):")
        print(f"    Total patches: {len(patches)}")
        print(f"    Deepest patches: depth={max_d}, size={H}x{W}, local_factor={lf}")
        print(f"    Resolution hierarchy:")
        seen = set()
        for d in sorted(set(depths)):
            ps = [p for p in patches if p['depth'] == d]
            H = ps[0]['bounds'][1] - ps[0]['bounds'][0]
            W = ps[0]['bounds'][3] - ps[0]['bounds'][2]
            lf = compute_local_factor(H, W, d, md, VQA_N)
            if lf not in seen:
                seen.add(lf)
            print(f"      depth {d}: {len(ps)} patches of {H}x{W}, lf={lf}")
        print()

    return new_md


# ================================================================
#  Part A: Show the max_depth problem and fix
# ================================================================
for N in [256, 512, 1024]:
    show_max_depth_comparison(N, 2, 6)


# ================================================================
#  Part B: Convergence tests with CORRECTED max_depth
# ================================================================
print("\n" + "="*60)
print("  CONVERGENCE TESTS WITH CORRECTED max_depth")
print("="*60)

N = 256
VQA_N = 2
min_size = 6
new_md = max(1, int(log(N / min_size) / log(VQA_N)))

# Test 1: Pipeline-like patches with corrected max_depth (all active, threshold=0)
print(f"\n{'#'*60}")
print(f"# Test B1: Pipeline patches with corrected max_depth={new_md}")
print(f"{'#'*60}")
patches = make_pipeline_patches(N, VQA_N, new_md, min_size)
print(f"  Generated {len(patches)} patches")
for d in sorted(set(p['depth'] for p in patches)):
    ps = [p for p in patches if p['depth'] == d]
    H = ps[0]['bounds'][1] - ps[0]['bounds'][0]
    lf = compute_local_factor(H, H, d, new_md, VQA_N)
    print(f"  depth {d}: {len(ps)} patches, size={H}x{H}, lf={lf}")
ok1, _ = run_comparison(N, VQA_N, patches, new_md, n_steps=1,
                         label="Pipeline patches (corrected max_depth), 1 step")

# Test 2: 10 steps
print(f"\n{'#'*60}")
print(f"# Test B2: Pipeline patches, corrected max_depth, 10 steps")
print(f"{'#'*60}")
ok2, _ = run_comparison(N, VQA_N, patches, new_md, n_steps=10,
                         label="Pipeline patches (corrected max_depth), 10 steps")

# Test 3: N=256
print(f"\n{'#'*60}")
print(f"# Test B3: N=256 pipeline patches, corrected max_depth")
print(f"{'#'*60}")
N256 = 256
md256 = max(1, int(log(N256 / min_size) / log(VQA_N)))
patches256 = make_pipeline_patches(N256, VQA_N, md256, min_size)
print(f"  N=256, max_depth={md256}, {len(patches256)} patches")
ok3, _ = run_comparison(N256, VQA_N, patches256, md256, n_steps=5,
                         label="N=256 pipeline patches, 5 steps")

# Test 4: Hierarchy test — deeper = more accurate
print(f"\n{'#'*60}")
print(f"# Test B4: Physics hierarchy — deeper patches reduce error")
print(f"{'#'*60}")
N = 256
new_md = max(1, int(log(N / min_size) / log(VQA_N)))
errors_by_depth = {}

for test_depth in sorted(set(p['depth'] for p in make_pipeline_patches(N, VQA_N, new_md, min_size))):
    # Make all patches at this depth
    ps = []
    size = N // (VQA_N ** test_depth)
    if size < 1:
        continue
    for y0 in range(0, N, size):
        for x0 in range(0, N, size):
            ps.append({
                'bounds': (y0, min(y0+size, N), x0, min(x0+size, N)),
                'depth': test_depth,
                'type': 'leaf_depth'
            })
    H = ps[0]['bounds'][1] - ps[0]['bounds'][0]
    lf = compute_local_factor(H, H, test_depth, new_md, VQA_N)

    grid = PeriodicGrid(resolution_N=N)
    sf = MHDSolver(grid, dt=1e-4, Re=200, Rm=200); sf.init_kelvin_helmholtz()
    sl = MHDSolver(grid, dt=1e-4, Re=200, Rm=200); sl.init_kelvin_helmholtz()
    for _ in range(5):
        sf.step_full(record_stats=False)
        sl.step_layered(ps, max_depth=new_md, target_dim=VQA_N)
    err = sum(np.linalg.norm(getattr(sf, f) - getattr(sl, f)) for f in ['vx','vy','Bx','By'])
    errors_by_depth[test_depth] = (err, lf)

print(f"\n  Depth → Error (5 steps), N={N}, max_depth={new_md}")
print(f"  {'Depth':<8} {'lf':<6} {'L2 Error':<15} {'Status'}")
print(f"  {'-'*45}")
prev_err = float('inf')
hierarchy_ok = True
for d in sorted(errors_by_depth.keys()):
    err, lf = errors_by_depth[d]
    status = "OK" if err <= prev_err + 1e-12 else "REGRESSION"
    if err > prev_err + 1e-12:
        hierarchy_ok = False
    print(f"  {d:<8} {lf:<6} {err:<15.2e} {status}")
    prev_err = err

# Summary
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
tests = [("B1: 1-step convergence", ok1),
         ("B2: 10-step convergence", ok2),
         ("B3: N=256 convergence", ok3),
         ("B4: Hierarchy monotonicity", hierarchy_ok)]
for name, ok in tests:
    print(f"  {name}: {'PASS' if ok else 'FAIL'}")
