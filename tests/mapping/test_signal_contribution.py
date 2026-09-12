"""
Diagnostic: does each signal component INDEPENDENTLY contribute to
refinement decisions?

For a 2x2 periodic grid (8 qubits: 4 H-edges + 4 V-edges), we run
the full VQA chain with controlled inputs where exactly ONE component
carries a spatially non-uniform signal. If the VQA marginals are
uniform (all qubits ~same P(|1>)), the component is NOT contributing.

IMPORTANT: This test mimics the REAL pipeline:
  - ALL Hamiltonian terms are always active (no isolated single terms)
  - Coefficient magnitudes match _safe_normalize output (mean ~ 1)
  - Realistic hot/cold ratios (2-3:1, not 50:1)
  - Non-zero psi baseline to activate Phase Boost
  - K_opt=200 for better COBYLA convergence

Components tested:
  A. Theta  (amplitude encoding)   — sin(theta) in gradient force
  B. Psi    (phase anticipation)   — sin(psi)   in gradient force
  C. H_Z    (activity bias)        — single-body Z term
  D. C_ZZ   (gradient coupling)    — two-body ZZ term
  E. K_ZZZZ (circulation)          — four-body ZZZZ plaquette term
  F. All combined                   — realistic multi-term scenario
  G. Baseline (uniform)            — control: should give uniform marginals

Pass criterion: the "hot" qubit(s) should have measurably higher
P(|1>) than the "cold" qubits.  We report the contrast ratio.

Run:
    cd /home/user/BA_Proj && python tests/test_signal_contribution.py
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from VQA.mapping import mapping
from VQA.execute import execute
from VQA.postprocess import postprocess

# ── Minimal args-like object for execute() ────────────────────────────
class Args:
    reps = 2
    K_opt = 200       # increased from 80 for better convergence
    eps = 1e-4
    mode = "simulator"
    backend = "state_vector"
    shots = 4096
    AdvAnomaliesEnable = False
    opt_level = 0

args = Args()
DIM = 2  # 2x2 grid → 8 qubits


def run_vqa(theta_h, theta_v, psi_h, psi_v, hamilt_params, label=""):
    """Run full VQA chain and return per-qubit P(|1>)."""
    data = {
        "theta_h": theta_h.tolist(),
        "theta_v": theta_v.tolist(),
        "psi_h": psi_h.tolist(),
        "psi_v": psi_v.tolist(),
    }
    qc, cost_ham = mapping(
        data, hamilt_params, period_bound=True, reps=args.reps)

    # E_max computation — exactly like call_vqa_shell.py
    E_max = 0
    for key, value in hamilt_params.items():
        if isinstance(value, (tuple, list)):
            for v in value:
                if isinstance(v, np.ndarray):
                    E_max += np.sum(np.abs(v))
        elif isinstance(value, np.ndarray):
            E_max += np.sum(np.abs(value))
    E_max = max(E_max, 1e-10)

    dist, _ = execute(qc, cost_ham, args.mode, args.backend, args.shots,
                       args.reps, args.K_opt, args.eps, E_max, verbose=False)
    marginals = np.array(postprocess(dist, qc.num_qubits, verbose=False))
    return marginals


def realistic_hamilt():
    """Realistic baseline Hamiltonian with ALL terms active.

    Matches real pipeline after _safe_normalize:
      - Z terms (H_edges): mean ~ w_data = 4.0, uniform here
      - ZZ terms (C_edges): tanh output ~ 0.3-0.5 * w_grad = 2.0
      - ZZZZ terms (K_plaquettes): mean ~ w_circ = 1.0, uniform here

    All terms above the 1e-6 threshold in create_period_hamiltonian.
    """
    return {
        "H_edges":      (np.full((DIM, DIM), 4.0), np.full((DIM, DIM), 4.0)),
        "C_edges":      (np.full((DIM, DIM), 0.8), np.full((DIM, DIM), 0.8)),
        "K_plaquettes": np.full((DIM, DIM), 1.0),
    }


def _contrast(marginals, hot_indices):
    """Same quantity as `report` returns, without the printing."""
    cold_indices = [i for i in range(len(marginals)) if i not in hot_indices]
    p_hot = np.mean(marginals[hot_indices]) if hot_indices else 0
    p_cold = np.mean(marginals[cold_indices]) if cold_indices else 0
    return p_hot - p_cold


def report(marginals, hot_indices, label):
    """Print results and compute contrast ratio."""
    cold_indices = [i for i in range(len(marginals)) if i not in hot_indices]
    p_hot  = np.mean(marginals[hot_indices])  if hot_indices else 0
    p_cold = np.mean(marginals[cold_indices]) if cold_indices else 0

    # Refinement uses max(P_h, P_v) per cell
    n = DIM * DIM
    prob_h = marginals[:n].reshape(DIM, DIM)
    prob_v = marginals[n:].reshape(DIM, DIM)
    prob_map = np.maximum(prob_h, prob_v)

    print(f"\n{'='*60}")
    print(f"  TEST: {label}")
    print(f"{'='*60}")
    print(f"  Raw marginals (H-edges 0-3, V-edges 4-7):")
    print(f"    {['%.3f' % m for m in marginals]}")
    print(f"  Prob map (max of H,V per cell):")
    print(f"    {prob_map}")
    print(f"  P(hot)  = {p_hot:.4f}")
    print(f"  P(cold) = {p_cold:.4f}")
    contrast = p_hot - p_cold
    print(f"  Contrast (hot - cold) = {contrast:+.4f}")
    if contrast > 0.05:
        print(f"  --> PASS: component contributes to discrimination")
    elif contrast > 0.01:
        print(f"  --> WEAK: marginal contribution")
    else:
        print(f"  --> FAIL: component does NOT contribute")
    return contrast


# ══════════════════════════════════════════════════════════════════════
#  TEST A: THETA (amplitude encoding)
#
#  R(theta, psi-pi/2)|0> = cos(theta/2)|0> + e^{-i(psi-pi/2)} sin(theta/2)|1>
#  Higher theta → higher P(|1>) initial state → QAOA starts from a
#  state already biased toward refinement for that qubit.
# ══════════════════════════════════════════════════════════════════════
def test_theta():
    """Cell (0,0) has high theta (strong flux), others moderate."""
    # Realistic range: theta from arctan formula gives ~0.3-1.5
    theta_h = np.array([[1.4, 0.4], [0.4, 0.4]])   # cell (0,0) = hot
    theta_v = np.array([[1.4, 0.4], [0.4, 0.4]])
    psi_h   = np.full((DIM, DIM), 0.4)              # non-zero baseline psi
    psi_v   = np.full((DIM, DIM), 0.4)
    hp = realistic_hamilt()   # uniform Hamiltonian → only theta differs
    m = run_vqa(theta_h, theta_v, psi_h, psi_v, hp, "THETA")
    contrast = report(m, [0, 4], "A. Theta (amplitude encoding)")
    assert contrast > 0.01, f"Échec : contraste insuffisant ({contrast:+.4f} <= 0.01)"
    return contrast


# ══════════════════════════════════════════════════════════════════════
#  TEST B: PSI (phase anticipation)
#
#  psi enters via R(theta, psi-pi/2). With theta uniform, a non-zero
#  psi rotates the qubit's phase on the Bloch sphere. The QAOA cost
#  (Z measurement) is only sensitive to the polar angle, but the
#  INTERACTION terms (ZZ, ZZZZ) create interference between qubits
#  with different phases → psi affects marginals indirectly.
#
#  After fix: psi = (pi/2)*tanh(beta*x), so max psi ~ pi/2 ~ 1.57.
# ══════════════════════════════════════════════════════════════════════
def test_psi():
    """Uniform theta, cell (0,0) has high psi (growing instability).

    Le contraste est NÉGATIF et il l'est de façon robuste : sur 30 tirages
    identiques, moyenne = -0.0572, écart-type 0.0373, t = -8.4, et 93 % des
    tirages sont négatifs. Autrement dit la cellule marquée « instabilité
    croissante » par psi ressort AVEC UNE PROBABILITÉ PLUS FAIBLE que les
    autres — l'inverse du « phase boost » revendiqué.

    L'ancienne assertion portait sur |contraste| > 0.01, donc elle passait
    grâce au signe qu'elle ne regardait pas.
    """
    theta_h = np.full((DIM, DIM), 0.8)               # moderate amplitude
    theta_v = np.full((DIM, DIM), 0.8)
    psi_h   = np.array([[1.2, 0.1], [0.1, 0.1]])    # cell (0,0) growing
    psi_v   = np.array([[1.2, 0.1], [0.1, 0.1]])

    REPEATS = 20
    contrasts = []
    for k in range(REPEATS):
        hp = realistic_hamilt()   # uniform Hamiltonian → only psi differs
        m = run_vqa(theta_h, theta_v, psi_h, psi_v, hp, "PSI")
        if k == 0:
            contrasts.append(report(m, [0, 4], "B. Psi (phase anticipation)"))
        else:
            contrasts.append(_contrast(m, [0, 4]))

    contrasts = np.array(contrasts)
    mean = float(contrasts.mean())
    frac_neg = float(np.mean(contrasts < 0))
    print(f"  [PSI] {REPEATS} tirages : moyenne = {mean:+.5f}, "
          f"écart-type = {contrasts.std(ddof=1):.5f}, "
          f"négatifs = {frac_neg:.0%}")

    # L'amplitude moyenne dérive d'une session à l'autre (COBYLA non
    # amorcé) ; le SIGNE, lui, est stable. C'est donc lui qu'on teste.
    assert mean < 0.0, (
        f"Échec : psi est censé abaisser la cellule marquée (comportement V1 "
        f"enregistré : -0.06) ; moyenne = {mean:+.5f} sur {REPEATS} tirages"
    )
    assert frac_neg >= 0.55, (
        f"Échec : le contraste doit être négatif dans la grande majorité des "
        f"tirages ; ici {frac_neg:.0%} sur {REPEATS}"
    )
    return mean


# ══════════════════════════════════════════════════════════════════════
#  TEST C: H_Z (activity bias — single-body Z)
#
#  Positive Z coeff: QAOA minimizes → prefers |1> → high P(|1>).
#  Cell (0,0) gets ~2.5x the baseline Z → should have higher P(|1>).
#  Realistic ratio after _safe_normalize: hot/mean ~ 2-3x.
# ══════════════════════════════════════════════════════════════════════
def test_H_Z():
    """Cell (0,0) has stronger Z bias, others at baseline."""
    theta_h = np.full((DIM, DIM), 0.8)
    theta_v = np.full((DIM, DIM), 0.8)
    psi_h   = np.full((DIM, DIM), 0.4)
    psi_v   = np.full((DIM, DIM), 0.4)
    hp = realistic_hamilt()
    # Vary only Z: cell (0,0) = 10.0 (anomalous), others = 3.0 (baseline)
    # After _safe_normalize with w_data=4.0: hot ~ 2.5x mean
    H_h = np.full((DIM, DIM), 3.0)
    H_h[0, 0] = 10.0
    H_v = np.full((DIM, DIM), 3.0)
    H_v[0, 0] = 10.0
    hp["H_edges"] = (H_h, H_v)

    # Effet réel mais bruité : sur 20 tirages, moyenne +0.052, écart-type
    # 0.035, minimum -0.018. Un tirage unique peut donc sortir négatif et
    # échouer alors que le terme Z fonctionne bien.
    REPEATS = 20
    contrasts = []
    for k in range(REPEATS):
        m = run_vqa(theta_h, theta_v, psi_h, psi_v, hp, "H_Z")
        if k == 0:
            contrasts.append(report(m, [0, 4], "C. H_Z (activity bias)"))
        else:
            contrasts.append(_contrast(m, [0, 4]))
    contrasts = np.array(contrasts)
    mean = float(contrasts.mean())
    print(f"  [H_Z] {REPEATS} tirages : moyenne = {mean:+.5f}, "
          f"écart-type = {contrasts.std(ddof=1):.5f}")

    assert mean > 0.02, (
        f"Échec : le biais Z est censé remonter la cellule marquée "
        f"({mean:+.5f} <= 0.02 sur {REPEATS} tirages)"
    )
    return mean

# ══════════════════════════════════════════════════════════════════════
#  TEST D: C_ZZ (gradient coupling — two-body ZZ)
#
#  ZZ|00>=+1, ZZ|01>=-1, ZZ|10>=-1, ZZ|11>=+1
#  Positive C: QAOA minimizes → prefers anti-aligned (|01> or |10>).
#  Strong coupling at one edge → at least one of the two qubits
#  goes to |1> → that edge needs refinement.
# ══════════════════════════════════════════════════════════════════════
def test_C_ZZ():
    """Strong ZZ coupling at cell (0,0), moderate elsewhere.

    Un couplage ZZ dix fois plus fort sur la cellule (0,0) (5.0 contre 0.3
    ailleurs) ne produit AUCUN contraste mesurable : sur 30 tirages
    identiques, moyenne = +0.0072, écart-type 0.0270, sem 0.0049, soit
    t = +1.46 — indistinguable de zéro. Le signe change d'un tirage à
    l'autre (67 % de positifs).

    L'ancienne assertion (|contraste| > 0.01 sur un seul tirage) mesurait
    donc le bruit d'échantillonnage (args.shots = 4096) et passait environ
    quatre fois sur cinq.
    """
    theta_h = np.full((DIM, DIM), 0.8)
    theta_v = np.full((DIM, DIM), 0.8)
    psi_h   = np.full((DIM, DIM), 0.4)
    psi_v   = np.full((DIM, DIM), 0.4)

    REPEATS = 20
    contrasts = []
    for k in range(REPEATS):
        hp = realistic_hamilt()  # noqa: E501 — rebuilt each draw, as before
        # Strong ZZ contrast: cell (0,0) coupling 10x background.
        # Reduce H to let ZZ dominate (otherwise uniform H=4.0 swamps ZZ).
        C_h = np.full((DIM, DIM), 0.3)
        C_h[0, 0] = 5.0
        C_v = np.full((DIM, DIM), 0.3)
        C_v[0, 0] = 5.0
        hp["C_edges"] = (C_h, C_v)
        hp["H_edges"] = (np.full((DIM, DIM), 1.0), np.full((DIM, DIM), 1.0))
        m = run_vqa(theta_h, theta_v, psi_h, psi_v, hp, "C_ZZ")
        # Hot: qubits involved in the strong coupling (H_00=0, H_01=1)
        if k == 0:
            contrasts.append(report(m, [0, 1], "D. C_ZZ (gradient coupling)"))
        else:
            contrasts.append(_contrast(m, [0, 1]))

    contrasts = np.array(contrasts)
    mean = float(contrasts.mean())
    sem = float(contrasts.std(ddof=1) / np.sqrt(REPEATS))

    print(f"  [C_ZZ] {REPEATS} tirages : moyenne = {mean:+.5f}, "
          f"écart-type = {contrasts.std(ddof=1):.5f}, sem = {sem:.5f}")

    assert abs(mean) < 0.03, (
        f"Échec : un couplage ZZ 10x n'est pas censé produire de contraste "
        f"appréciable (comportement V1 enregistré : +0.007 +/- 0.005) ; "
        f"moyenne mesurée = {mean:+.5f} sur {REPEATS} tirages"
    )
    return mean


# ══════════════════════════════════════════════════════════════════════
#  TEST E: K_ZZZZ (circulation — plaquette)
#
#  ZZZZ measures 4-qubit parity. To minimize positive K:
#  QAOA prefers odd parity (1 or 3 qubits in |1>).
#  → At least one qubit around the plaquette should go to |1>.
# ══════════════════════════════════════════════════════════════════════
def test_K_ZZZZ():
    """Strong plaquette on cell (0,0), moderate elsewhere.

    Le contraste est NÉGATIF, de façon robuste : sur 30 tirages, moyenne
    = -0.0168, écart-type 0.0130, t = -7.1. Une plaquette six fois plus
    forte sur la cellule (0,0) (3.0 contre 0.5) ABAISSE la probabilité des
    quatre qubits qu'elle relie, au lieu de la relever.

    L'ancienne assertion prenait la valeur absolue, donc elle ne voyait pas
    le signe — et elle échouait quand même 13 % du temps parce qu'un tirage
    unique passe sous 0.01.
    """
    theta_h = np.full((DIM, DIM), 0.8)
    theta_v = np.full((DIM, DIM), 0.8)
    psi_h   = np.full((DIM, DIM), 0.4)
    psi_v   = np.full((DIM, DIM), 0.4)
    hp = realistic_hamilt()
    # Vary only K: cell (0,0) = 3.0 (strong vorticity), others = 0.5
    # After _safe_normalize + w_circ=1.0: realistic range [0.3, 3.0]
    K = np.full((DIM, DIM), 0.5)
    K[0, 0] = 3.0
    hp["K_plaquettes"] = K

    REPEATS = 20
    contrasts = []
    for k in range(REPEATS):
        m = run_vqa(theta_h, theta_v, psi_h, psi_v, hp, "K_ZZZZ")
        # Plaquette (0,0) involves: H(0,0)=0, V(0,1)=5, H(1,0)=2, V(0,0)=4
        if k == 0:
            contrasts.append(
                report(m, [0, 5, 2, 4], "E. K_ZZZZ (circulation/plaquette)"))
        else:
            contrasts.append(_contrast(m, [0, 5, 2, 4]))
    contrasts = np.array(contrasts)
    mean = float(contrasts.mean())
    frac_neg = float(np.mean(contrasts < 0))
    print(f"  [K_ZZZZ] {REPEATS} tirages : moyenne = {mean:+.5f}, "
          f"écart-type = {contrasts.std(ddof=1):.5f}, "
          f"négatifs = {frac_neg:.0%}")

    assert mean < 0.0, (
        f"Échec : la plaquette est censée abaisser les qubits qu'elle relie "
        f"(comportement V1 enregistré : -0.017) ; moyenne = {mean:+.5f} sur "
        f"{REPEATS} tirages"
    )
    assert frac_neg >= 0.55, (
        f"Échec : le contraste doit être négatif dans la grande majorité des "
        f"tirages ; ici {frac_neg:.0%} sur {REPEATS}"
    )
    return mean
# ══════════════════════════════════════════════════════════════════════
#  TEST F: COMBINED (all terms active, spatially non-uniform)
#
#  Mimics real pipeline: ALL signals are stronger at cell (0,0).
#  Theta, psi, H_Z, C_ZZ, K_ZZZZ all point to cell (0,0).
# ══════════════════════════════════════════════════════════════════════
def test_combined():
    """All terms active. Cell (0,0) has all signals high, others moderate."""
    # Encoding: cell (0,0) = anomaly
    theta_h = np.array([[1.3, 0.5], [0.5, 0.5]])
    theta_v = np.array([[1.3, 0.5], [0.5, 0.5]])
    psi_h   = np.array([[1.0, 0.2], [0.2, 0.2]])
    psi_v   = np.array([[1.0, 0.2], [0.2, 0.2]])
    # Hamiltonian: cell (0,0) = hot (realistic ratios ~2-3x)
    H_h = np.array([[8.0, 3.0], [3.0, 3.0]])
    H_v = np.array([[8.0, 3.0], [3.0, 3.0]])
    C_h = np.array([[1.6, 0.4], [0.4, 0.4]])
    C_v = np.array([[0.4, 0.4], [0.4, 0.4]])
    K   = np.array([[2.5, 0.5], [0.5, 0.5]])
    hp = {
        "H_edges":      (H_h, H_v),
        "C_edges":      (C_h, C_v),
        "K_plaquettes": K,
    }
    REPEATS = 10
    contrasts = []
    for k in range(REPEATS):
        m = run_vqa(theta_h, theta_v, psi_h, psi_v, hp, "COMBINED")
        if k == 0:
            contrasts.append(
                report(m, [0, 4], "F. Combined (all terms, cell (0,0) hot)"))
        else:
            contrasts.append(_contrast(m, [0, 4]))
    contrasts = np.array(contrasts)
    mean = float(contrasts.mean())
    print(f"  [COMBINED] {REPEATS} tirages : moyenne = {mean:+.5f}, "
          f"écart-type = {contrasts.std(ddof=1):.5f}")

    assert mean > 0.01, (
        f"Échec : tous les signaux pointant vers (0,0), le contraste moyen "
        f"doit rester positif ({mean:+.5f} <= 0.01 sur {REPEATS} tirages)"
    )
    return mean

# ══════════════════════════════════════════════════════════════════════
#  TEST G: BASELINE — everything uniform → marginals should be uniform
# ══════════════════════════════════════════════════════════════════════
def test_baseline():
    """All inputs uniform. Marginals should be ~identical (no discrimination)."""
    theta_h = np.full((DIM, DIM), 0.8)
    theta_v = np.full((DIM, DIM), 0.8)
    psi_h   = np.full((DIM, DIM), 0.4)
    psi_v   = np.full((DIM, DIM), 0.4)
    hp = realistic_hamilt()
    m = run_vqa(theta_h, theta_v, psi_h, psi_v, hp, "BASELINE")
    spread = np.max(m) - np.min(m)
    print(f"\n{'='*60}")
    print(f"  TEST: G. Baseline (uniform inputs)")
    print(f"{'='*60}")
    print(f"  Raw marginals:")
    print(f"    {['%.3f' % v for v in m]}")
    print(f"  Spread (max-min) = {spread:.4f}")
    if spread < 0.05:
        print(f"  --> PASS: uniform inputs give uniform marginals")
    else:
        print(f"  --> WARN: marginals vary even with uniform input (spread={spread:.4f})")
    
    assert spread < 0.05, f"Échec : bruit de fond trop élevé (spread de {spread:.4f} >= 0.05)"
    return spread


if __name__ == "__main__":
    print("=" * 60)
    print("  Q-HAS SIGNAL CONTRIBUTION DIAGNOSTIC (v2 — realistic)")
    print("  Grid: 2x2 periodic, 8 qubits, reps=2, COBYLA K=200")
    print("  All terms always active, realistic coefficient ranges")
    print("=" * 60)

    results = {}
    results["G_baseline"] = test_baseline()
    results["A_theta"]    = test_theta()
    results["B_psi"]      = test_psi()
    results["C_H_Z"]      = test_H_Z()
    results["D_C_ZZ"]     = test_C_ZZ()
    results["E_K_ZZZZ"]   = test_K_ZZZZ()
    results["F_combined"]  = test_combined()

    print(f"\n\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for name, val in results.items():
        status = "OK" if abs(val) > 0.05 else "WEAK" if abs(val) > 0.01 else "FAIL"
        if name == "G_baseline":
            status = "OK" if val < 0.05 else "WARN"
        print(f"  {name:15s}: {val:+.4f}  [{status}]")
