"""
Diagnostic: measure actual Hamiltonian coefficient magnitudes on Orszag-Tang
to understand the balance between Z, ZZ, and ZZZZ terms.

Reports energy decomposition at full resolution and after downsampling
to the VQA grid size, including the K_xpoint (X-point reconnection) term.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from Simulation.RescaleArrays import get_adaptive_flux

N = 256
L = 2 * np.pi
RE = 1000
RM = 1000
THRESHOLD = 0.2
VQA_N = 2

# Run OT to t=2.0
grid = PeriodicGrid(N, L)
nu = L / RE
eta = L / RM
sim = MHDSolver(grid, dt=1e-3, Re=RE, Rm=RM)
sim.init_orszag_tang()

for _ in range(2000):
    sim.step_full(record_stats=False)

fields = sim.get_fluxes()
mapper = AngleMapper()
phi_dict = mapper.compute_stress_flux(fields)

phi_h = phi_dict['phi_horizontal']
phi_v = phi_dict['phi_vertical']
avg_phi = (np.mean(phi_h) + np.mean(phi_v)) / 2

print(f"{'='*70}")
print(f"DIAGNOSTIC: Hamiltonian Coefficients on Orszag-Tang (t=2.0)")
print(f"{'='*70}")
print(f"Grid: {N}x{N}, Re={RE}, Rm={RM}")
print(f"threshold_amr={THRESHOLD}")
print(f"avg_phi = {avg_phi:.6f}")
print()

records = []

# Test with different beta (threshold-contrast sensitivity) values
for beta_mic in [0.2, 0.5, 1.0, 1.5, 2.0]:
    phys_mapper = PhysicalMapper(cs=1.0, nu=nu, eta_mhd=eta,
                                  beta_curl=beta_mic, beta_xpoint=beta_mic,
                                  dx=grid.dx)

    score = phys_mapper.physical_score(fields)

    hp = phys_mapper.compute_coefficients(
        sim, score, fields, THRESHOLD,
        advanced_anomalies_enabled=True
    )

    H_h, H_v = hp['H_edges']
    C_h, C_v = hp['C_edges']
    K = hp['K_plaquettes']
    Kx = hp.get('K_xpoint', np.zeros_like(K))

    # Count non-zero entries
    n_total = H_h.size
    eps = 1e-6

    pct_H = np.sum(np.abs(H_h) > eps) / n_total * 100
    pct_C = np.sum(np.abs(C_h) > eps) / n_total * 100
    pct_K = np.sum(np.abs(K) > eps) / n_total * 100
    pct_Kx = np.sum(np.abs(Kx) > eps) / n_total * 100

    # Energy contributions (sum of absolute values)
    E_Z = np.sum(np.abs(H_h)) + np.sum(np.abs(H_v))
    E_ZZ = np.sum(np.abs(C_h)) + np.sum(np.abs(C_v))
    E_ZZZZ_plaq = np.sum(np.abs(K))
    E_ZZZZ_xpoint = np.sum(np.abs(Kx))
    E_total = E_Z + E_ZZ + E_ZZZZ_plaq + E_ZZZZ_xpoint

    print(f"--- beta = {beta_mic:.1f} ---")
    print(f"  Z bias (1-body):    mean|H|={np.mean(np.abs(H_h)):.4f}, max={np.max(np.abs(H_h)):.4f}, {pct_H:.0f}% non-zero, E_share={E_Z/E_total*100:.1f}%")
    print(f"  ZZ coupling (2-body): mean|C|={np.mean(np.abs(C_h)):.4f}, max={np.max(np.abs(C_h)):.4f}, {pct_C:.0f}% non-zero, E_share={E_ZZ/E_total*100:.1f}%")
    print(f"  ZZZZ plaquette:     mean|K|={np.mean(np.abs(K)):.4f}, max={np.max(np.abs(K)):.4f}, {pct_K:.0f}% non-zero, E_share={E_ZZZZ_plaq/E_total*100:.1f}%")
    print(f"  ZZZZ X-point:       mean|Kx|={np.mean(np.abs(Kx)):.4f}, max={np.max(np.abs(Kx)):.4f}, {pct_Kx:.0f}% non-zero, E_share={E_ZZZZ_xpoint/E_total*100:.1f}%")
    print(f"  Total energy: {E_total:.2f}")
    print()

    # After downsampling to 2x2 (what the QAOA actually sees at depth 0)
    mini_h, mini_v, mini_hp, mini_score = get_adaptive_flux(
        phi_h, phi_v, None, None, score, hp, target_dim=VQA_N, type_filter=True
    )

    mH_h, mH_v = mini_hp['H_edges']
    mC_h, mC_v = mini_hp['C_edges']
    mK = mini_hp['K_plaquettes']
    mKx = mini_hp.get('K_xpoint', np.zeros_like(mK))

    print(f"  After downsampling to {VQA_N}x{VQA_N} (what QAOA sees at depth 0):")
    print(f"    H_h (Z bias):   {mH_h}")
    print(f"    C_h (ZZ):       {mC_h}")
    print(f"    K (ZZZZ plaq):  {mK}")
    if np.any(np.abs(mKx) > eps):
        print(f"    Kx (X-point):   {mKx}")

    # Ratio of multi-body to single-body energy
    E_single = np.sum(np.abs(mH_h)) + np.sum(np.abs(mH_v))
    E_multi = np.sum(np.abs(mC_h)) + np.sum(np.abs(mC_v)) + np.sum(np.abs(mK)) + np.sum(np.abs(mKx))
    ratio = E_multi / max(E_single, 1e-10)
    print(f"    Multi-body / Single-body energy ratio: {ratio:.4f}")
    if ratio < 0.1:
        print(f"    ⚠ QAOA is essentially a SINGLE-BODY problem (no quantum advantage)")
    elif ratio < 0.3:
        print(f"    ⚠ Multi-body terms are weak — limited quantum correlations")
    else:
        print(f"    ✓ Meaningful multi-body terms — quantum correlations active")
    print()

    records.append({
        'beta': beta_mic,
        'ratio': ratio,
        'max_H': float(max(np.max(np.abs(mH_h)), np.max(np.abs(mH_v)))),
        'max_C': float(max(np.max(np.abs(mC_h)), np.max(np.abs(mC_v)))),
        'max_K': float(np.max(np.abs(mK))),
        'C_h': np.array(mC_h, dtype=float),
    })


# ══════════════════════════════════════════════════════════════════════
#  ACCEPTANCE — what V1 is expected to do
# ══════════════════════════════════════════════════════════════════════
#
# Reference run (N=256 Orszag-Tang at t=2.0, threshold_amr=0.2, VQA 2x2):
#
#   beta        0.2      0.5      1.0      1.5      2.0
#   ratio    11876.7  13527.6  16279.2  19030.7      ...
#   max|H| ~ 9.6e-05 at every beta, max|C| ~ 1.0031 at every beta, max|K| = 0
#
# Three facts are pinned, and none of them is what the printed verdict
# ("meaningful multi-body terms") suggests:
#
#   1. the downsampled ZZ block does not depend on beta_curl / beta_xpoint
#      at all — those hyperparameters reach the curl and X-point channels
#      only, so sweeping them cannot change the gradient coupling;
#   2. no ZZZZ plaquette survives downsampling (max|K| = 0 exactly);
#   3. the multi-body / single-body ratio is large because the Z bias is
#      ~1e-4, not because the couplings are strong. The ratio verdict is
#      therefore not evidence of "quantum correlations active".
MIN_RATIO = 1e3
MAX_Z_OVER_C = 1e-3


def check_expected_behaviour(records):
    """Fail the run if the measured structure departs from the recorded one.

    Without this the script prints a verdict line and returns 0 whatever it
    measured, so `run_tests.sh` reports the stage as PASSED.
    """
    assert len(records) == 5, f"expected 5 beta values, got {len(records)}"

    ref = records[0]['C_h']
    for r in records[1:]:
        assert np.allclose(r['C_h'], ref, rtol=0, atol=0), (
            f"the downsampled ZZ block changed with beta ({r['beta']}); it "
            "must not — beta_curl/beta_xpoint do not reach the gradient "
            "coupling"
        )

    for r in records:
        assert r['max_K'] == 0.0, (
            f"beta={r['beta']}: a ZZZZ plaquette survived downsampling "
            f"(max|K| = {r['max_K']:.3e}); the recorded behaviour is exactly 0"
        )
        assert r['max_C'] > 0.1, (
            f"beta={r['beta']}: the ZZ coupling vanished (max|C| = "
            f"{r['max_C']:.3e}); at threshold_amr=0.2 part of the field sits "
            "inside the uncertainty window and the coupling must survive"
        )
        z_over_c = r['max_H'] / r['max_C']
        assert z_over_c < MAX_Z_OVER_C, (
            f"beta={r['beta']}: Z/ZZ magnitude ratio {z_over_c:.3e} is no "
            f"longer below {MAX_Z_OVER_C:g} — the recorded imbalance changed"
        )
        assert r['ratio'] > MIN_RATIO, (
            f"beta={r['beta']}: multi/single ratio {r['ratio']:.1f} fell "
            f"below {MIN_RATIO:g}"
        )

    print(f"{'='*70}")
    print("  [ACCEPTANCE] ZZ block invariant under beta; no ZZZZ survives "
          "downsampling;")
    print(f"  Z/ZZ magnitude ratio < {MAX_Z_OVER_C:g} at every beta -> OK")
    print(f"{'='*70}")


check_expected_behaviour(records)
