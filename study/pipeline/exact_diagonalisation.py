#!/usr/bin/env python3
"""
Phase 4 - Exact diagonalization of the Hamiltonian on hard patches.

For each hard patch identified in Phase 2:
  1. Extract the patch fields at VQA resolution (n_patches x n_patches)
  2. Build the Hamiltonian (Z + ZZ + ZZZZ) via PhysicalMapper
  3. Exact diagonalization: H.to_matrix() -> numpy eigh
  4. Compare exact ground state decisions to the L2-based ground truth
  5. Report accuracy, overlap, and identify "promising" patches where
     the Hamiltonian correctly captures the refinement need

A patch is "promising" if the exact ground state of the Hamiltonian
agrees with the L2 ground truth on whether each cell should be refined.
Only these patches proceed to Phase 5 (QAOA).

Input:  results/dns_{scenario}_Re{Re}_N{N}.npz
        results/patches_{scenario}_Re{Re}_N{N}_dim{D}.npz
Output: results/exact_diag_{scenario}_Re{Re}_N{N}_dim{D}{sfx}.npz
        (sfx = "_v2" sous --v2, "" sinon)

Usage:
  python study/exact_diagonalisation.py
  python study/exact_diagonalisation.py --re 800 --dim 4
"""
import argparse, os, sys, time
import numpy as np

# --- chemins du dépôt (bloc unique, généré) -------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -------------------------------------------------------------------------
from config import (
    RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N, VQA_DIMS,
    TRAINED_SIGMA, TRAINED_BETA_CURL, TRAINED_BETA_XPOINT,
    TRAINED_W_Z_FRAC, TRAINED_THRESHOLD, TRAINED_GAMMA_HYDRO,
    TRAINED_GAMMA_MAG, TRAINED_KAPPA,
    V2_THRESHOLD,
)
from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.HamiltParams import PhysicalMapper
from Simulation.HamiltParams_v2 import PhysicalMapperV2
from Simulation.PhysToAngle import AngleMapper
from VQA.cost_hamiltonian import create_period_hamiltonian


# -------------------------------------------------------------------
# Build Hamiltonian for a single patch
# -------------------------------------------------------------------

def build_patch_hamiltonian(vx, vy, Bx, By, N, n_patches, Re,
                            threshold_amr, use_v2=False,
                            sigma=None, beta_curl=None,
                            beta_xpoint=None, w_z_frac=None,
                            gamma_hydro=None, gamma_mag=None, kappa=None,
                            c_bias=None):
    """
    Build the full Hamiltonian for the entire n_patches x n_patches grid.

    Returns:
      hamilt_params: dict with H_edges, C_edges, K_plaquettes
      score: (N, N) classical score
    """
    dx = 2 * np.pi / N
    nu = 1.0 / Re
    eta = 1.0 / Re

    if use_v2:
        mapper = PhysicalMapperV2(dx=dx, c_bias=c_bias)
    else:
        mapper = PhysicalMapper(
            cs=1.0, nu=nu, eta_mhd=eta, dx=dx,
            gamma_hydro=gamma_hydro, gamma_mag=gamma_mag,
            kappa=kappa, sigma=sigma,
            beta_curl=beta_curl, beta_xpoint=beta_xpoint,
            w_z_frac=w_z_frac,
        )

    # set up solver for gradient operators
    grid = PeriodicGrid(N)
    sim = MHDSolver(grid, dt=1e-4, Re=Re, Rm=Re)
    sim.vx, sim.vy, sim.Bx, sim.By = vx, vy, Bx, By

    fields = sim.get_fluxes()

    # full-resolution classical score (needs Jz)
    physics_state = {"vx": vx, "vy": vy, "Bx": Bx, "By": By,
                     "Jz": fields["Jz"], "dx": dx}
    full_score = AngleMapper.classical_score(physics_state)

    # downsample fields to VQA resolution for Hamiltonian construction
    patch_size = N // n_patches
    dx_vqa = dx * patch_size  # effective cell size at VQA resolution

    # block-average fields to VQA resolution
    def block_avg(f):
        return f.reshape(n_patches, patch_size, n_patches, patch_size).mean(axis=(1, 3))

    def block_max(f):
        return f.reshape(n_patches, patch_size, n_patches, patch_size).max(axis=(1, 3))

    vx_vqa = block_avg(vx)
    vy_vqa = block_avg(vy)
    Bx_vqa = block_avg(Bx)
    By_vqa = block_avg(By)
    score_vqa = block_max(full_score)

    # compute coefficients at VQA resolution
    grid_vqa = PeriodicGrid(n_patches, length_L=2*np.pi)
    sim_vqa = MHDSolver(grid_vqa, dt=1e-4, Re=Re, Rm=Re)
    sim_vqa.vx = vx_vqa
    sim_vqa.vy = vy_vqa
    sim_vqa.Bx = Bx_vqa
    sim_vqa.By = By_vqa

    fields_vqa = sim_vqa.get_fluxes()

    hamilt_params = mapper.compute_coefficients(
        sim_vqa, score_vqa, fields_vqa, threshold_amr,
        dx_override=dx_vqa, verbose=False,
    )

    return hamilt_params, score_vqa, full_score


# -------------------------------------------------------------------
# Exact diagonalization
# -------------------------------------------------------------------

def exact_diag(hamilt_params, dim):
    """
    Exact diagonalization of the Hamiltonian.

    Returns:
      ground_state: (2^n,) complex vector
      ground_energy: float
      all_energies: (2^n,) sorted eigenvalues
      gap: energy gap between ground and first excited state
    """
    n_qubits = 2 * dim * dim
    if n_qubits > 20:
        raise ValueError(
            f"Too many qubits ({n_qubits}) for exact diag. "
            f"Max supported: 20 (dim <= 3 for periodic)."
        )

    H_op = create_period_hamiltonian(hamilt_params, dim)
    H_mat = H_op.to_matrix()

    # hermitian eigendecomposition
    energies, states = np.linalg.eigh(H_mat.real)

    ground_energy = energies[0]
    ground_state = states[:, 0]
    gap = energies[1] - energies[0] if len(energies) > 1 else 0.0

    return ground_state, ground_energy, energies, gap


def ground_state_decisions(ground_state, dim):
    """
    Extract per-qubit refinement decisions from the exact ground state.

    For each qubit i, compute P(qi=1) = marginal probability of measuring |1>.
    Decision: refine if P(qi=1) > 0.5.

    Returns:
      marginals: (n_qubits,) array of P(qi=1)
      decisions_h: (dim, dim) bool array for horizontal links
      decisions_v: (dim, dim) bool array for vertical links
    """
    n_qubits = 2 * dim * dim
    probs = np.abs(ground_state) ** 2  # (2^n,)

    marginals = np.zeros(n_qubits)
    for qi in range(n_qubits):
        # sum probabilities of all basis states where qubit qi is |1>
        for basis_idx in range(len(probs)):
            if (basis_idx >> qi) & 1:
                marginals[qi] += probs[basis_idx]

    # split into horizontal and vertical
    n_cells = dim * dim
    marg_h = marginals[:n_cells].reshape(dim, dim)
    marg_v = marginals[n_cells:].reshape(dim, dim)

    # per-cell decision: refine if EITHER horizontal or vertical link says so
    decisions_h = marg_h > 0.5
    decisions_v = marg_v > 0.5

    return marginals, decisions_h, decisions_v


# -------------------------------------------------------------------
# Ground truth from L2 error
# -------------------------------------------------------------------

def l2_ground_truth(l2_errors, threshold):
    """
    Ground truth: which patches truly need refinement based on L2 error.
    """
    return l2_errors >= threshold


# -------------------------------------------------------------------
# Comparison metrics
# -------------------------------------------------------------------

def compare_decisions(exact_h, exact_v, gt_refine, score_patch, threshold_amr):
    """
    Compare exact diag decisions to L2 ground truth.

    exact_h, exact_v: (dim, dim) bool from exact ground state
    gt_refine: (dim, dim) bool from L2 error
    score_patch: (dim, dim) classical score per patch
    threshold_amr: threshold for classical decision

    Returns dict with accuracy, precision, recall, F1 for both
    exact diag and classical baseline.
    """
    # exact diag: refine if either H or V link says refine
    exact_refine = exact_h | exact_v

    # classical baseline: refine if score > threshold
    classical_refine = score_patch > threshold_amr

    def metrics(pred, gt):
        tp = np.sum(pred & gt)
        fp = np.sum(pred & ~gt)
        fn = np.sum(~pred & gt)
        tn = np.sum(~pred & ~gt)
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

    return {
        "exact": metrics(exact_refine, gt_refine),
        "classical": metrics(classical_refine, gt_refine),
        "exact_refine": exact_refine,
        "classical_refine": classical_refine,
    }


# -------------------------------------------------------------------
# Main analysis
# -------------------------------------------------------------------

def analyze_snapshot(vx, vy, Bx, By, N, n_patches, Re,
                     l2_errors, is_hard, l2_threshold,
                     use_v2=False):
    """
    Run exact diagonalization on one snapshot.

    Returns dict with results, or None if dim is too large.
    """
    n_qubits = 2 * n_patches * n_patches
    if n_qubits > 20:
        return None

    threshold_amr = V2_THRESHOLD if use_v2 else TRAINED_THRESHOLD

    # build Hamiltonian
    if use_v2:
        hamilt_params, score_vqa, full_score = build_patch_hamiltonian(
            vx, vy, Bx, By, N, n_patches, Re,
            threshold_amr=threshold_amr, use_v2=True,
        )
    else:
        hamilt_params, score_vqa, full_score = build_patch_hamiltonian(
            vx, vy, Bx, By, N, n_patches, Re,
            threshold_amr=threshold_amr,
            sigma=TRAINED_SIGMA,
            beta_curl=TRAINED_BETA_CURL,
            beta_xpoint=TRAINED_BETA_XPOINT,
            w_z_frac=TRAINED_W_Z_FRAC,
            gamma_hydro=TRAINED_GAMMA_HYDRO,
            gamma_mag=TRAINED_GAMMA_MAG,
            kappa=TRAINED_KAPPA,
        )

    # exact diag
    ground_state, ground_energy, energies, gap = exact_diag(
        hamilt_params, n_patches
    )

    # decisions from ground state
    marginals, decisions_h, decisions_v = ground_state_decisions(
        ground_state, n_patches
    )

    # ground truth from L2
    gt_refine = l2_ground_truth(l2_errors, l2_threshold)

    # compare
    comparison = compare_decisions(
        decisions_h, decisions_v, gt_refine,
        score_vqa, threshold_amr,
    )

    # is this snapshot "promising"?
    # promising = exact diag F1 > classical F1
    promising = comparison["exact"]["f1"] >= comparison["classical"]["f1"]

    # D-45 — `promising` seul ne dit pas si la comparaison a mesure quoi que
    # ce soit. Un predicteur CONSTANT (tout-raffiner, ou rien-raffiner) rend
    # un F1 parfaitement defini, dans le bon intervalle, et ne capture
    # aucune structure spatiale : compare a une ligne de base classique elle
    # aussi constante, il rend le MEME F1 par construction, donc `promising`
    # est vrai par le `>=` sans qu'aucun accord n'ait ete observe.
    #
    # Mesure (dim=2 — seule dimension executable, dim=4/8 depassent le
    # plafond de 20 qubits ; Re=400, N=256, 4 scenarios canoniques,
    # 40 snapshots) : decision exacte tout-a-1 40/40, ligne de base
    # classique tout-a-1 40/40, `exact_refine != classical_refine` 0/40,
    # F1 egaux 40/40 et jamais superieurs. `promising` valait donc True
    # 40/40 avec `>=` et aurait valu False 40/40 avec le `>` du commentaire
    # ci-dessus : la porte porte zero bit dans les deux sens.
    #
    # On n'y touche PAS le verdict — quel operateur `promising` doit porter
    # est une question ouverte pour l'humain (voir DEFAUTS.md D-47). On rend
    # la degenerescence VISIBLE, au lieu de la laisser lire comme un succes.
    exact_refine = comparison["exact_refine"]
    classical_refine = comparison["classical_refine"]
    degenerate_decision = bool(exact_refine.all() or (~exact_refine).all())
    degenerate_classical = bool(
        classical_refine.all() or (~classical_refine).all())
    f1_tie = bool(comparison["exact"]["f1"] == comparison["classical"]["f1"])

    return {
        "ground_energy": ground_energy,
        "gap": gap,
        "marginals": marginals,
        "decisions_h": decisions_h,
        "decisions_v": decisions_v,
        "gt_refine": gt_refine,
        "score_vqa": score_vqa,
        "comparison": comparison,
        "promising": promising,
        "degenerate_decision": degenerate_decision,
        "degenerate_classical": degenerate_classical,
        "f1_tie": f1_tie,
        # `promising` gagne par un vrai ecart, pas par une egalite entre deux
        # predicteurs constants. C'est ce chiffre-la qui selectionne.
        "promising_informative": bool(promising and not degenerate_decision
                                      and not f1_tie),
        "n_energies_below_gap": int(np.sum(energies < ground_energy + gap)),
    }


def run_phase4(dns_path, patches_path, n_patches, use_v2=False):
    """
    Run Phase 4 for one (scenario, Re) combination.
    """
    dns = np.load(dns_path)
    patches = np.load(patches_path)

    vx_all = dns["vx"]
    vy_all = dns["vy"]
    Bx_all = dns["Bx"]
    By_all = dns["By"]
    N = vx_all.shape[1]
    Re = int(dns.get("meta_Re", 800))
    scenario = str(dns.get("meta_scenario", "unknown"))

    is_hard_all = patches["is_hard"]
    l2_all = patches["l2_errors"]
    l2_threshold = float(patches["l2_threshold"])

    n_snaps = len(vx_all)
    n_qubits = 2 * n_patches * n_patches

    print(f"  {scenario} Re={Re} N={N}, dim={n_patches} ({n_qubits} qubits)")

    if n_qubits > 20:
        print(f"  SKIP: {n_qubits} qubits too large for exact diag (max 20)")
        return None, None

    # use a subset of snapshots for speed
    snap_indices = list(range(0, n_snaps, max(1, n_snaps // 10)))
    if len(snap_indices) < 3:
        snap_indices = list(range(n_snaps))

    print(f"  Analyzing {len(snap_indices)} snapshots...")

    all_results = []
    n_promising = 0
    n_informative = 0
    degenerate_snaps = []

    for si_idx, si in enumerate(snap_indices):
        t0 = time.time()
        vx = vx_all[si].astype(np.float64)
        vy = vy_all[si].astype(np.float64)
        Bx = Bx_all[si].astype(np.float64)
        By = By_all[si].astype(np.float64)

        result = analyze_snapshot(
            vx, vy, Bx, By, N, n_patches, Re,
            l2_all[si], is_hard_all[si], l2_threshold,
            use_v2=use_v2,
        )

        if result is None:
            continue

        all_results.append(result)
        if result["promising"]:
            n_promising += 1
        if result["promising_informative"]:
            n_informative += 1
        if result["degenerate_decision"]:
            degenerate_snaps.append(si)

        elapsed = time.time() - t0
        c = result["comparison"]
        # D-45 : un snapshot degenere ne s'annonce plus PROMISING tout court.
        if result["degenerate_decision"]:
            verdict = "DEGENERATE (constant decision)"
        elif result["promising_informative"]:
            verdict = "PROMISING"
        elif result["promising"]:
            verdict = "promising (tie only)"
        else:
            verdict = ""
        print(f"    snap {si:3d}: E0={result['ground_energy']:.4f} "
              f"gap={result['gap']:.4f} "
              f"exact_F1={c['exact']['f1']:.3f} "
              f"class_F1={c['classical']['f1']:.3f} "
              f"{verdict} "
              f"({elapsed:.1f}s)")

    if not all_results:
        print("  No results produced.")
        return None, None

    # summary
    exact_f1s = [r["comparison"]["exact"]["f1"] for r in all_results]
    class_f1s = [r["comparison"]["classical"]["f1"] for r in all_results]
    gaps = [r["gap"] for r in all_results]

    print(f"\n  Summary ({len(all_results)} snapshots):")
    print(f"    Exact diag F1:  mean={np.mean(exact_f1s):.3f} "
          f"std={np.std(exact_f1s):.3f} "
          f"[{np.min(exact_f1s):.3f}, {np.max(exact_f1s):.3f}]")
    print(f"    Classical F1:   mean={np.mean(class_f1s):.3f} "
          f"std={np.std(class_f1s):.3f} "
          f"[{np.min(class_f1s):.3f}, {np.max(class_f1s):.3f}]")
    print(f"    Energy gap:     mean={np.mean(gaps):.4f} "
          f"std={np.std(gaps):.4f}")
    print(f"    Promising:      {n_promising}/{len(all_results)} "
          f"({100*n_promising/len(all_results):.0f}%)")
    # D-45 : le chiffre qui selectionne vraiment, et ce qui a ete exclu.
    print(f"    ... informative: {n_informative}/{len(all_results)} "
          f"(ecart reel, hors egalites et decisions constantes)")
    if degenerate_snaps:
        print(f"    ... DEGENERATE: {len(degenerate_snaps)}/{len(all_results)}"
              f" — decision exacte constante sur les snapshots "
              f"{degenerate_snaps} : le fondamental ne distingue aucune "
              f"cellule, le F1 ne mesure pas un accord (DEFAUTS.md D-47)")

    meta = {
        "scenario": scenario, "Re": Re, "N": N,
        "n_patches": n_patches, "n_qubits": n_qubits,
        "snap_indices": np.array(snap_indices),
        "suffix": "_v2" if use_v2 else "",
    }

    return all_results, meta


def artifact_name(scenario, Re, N, n_patches, suffix=""):
    """
    Nom de l'artefact de la phase 4 — UNE seule source, ecrite ET relue.

    D-178 : le resume cross-Re de `main()` reconstruisait ce nom a la main,
    sans le suffixe. Sous `--v2` il relisait donc l'artefact **v1** homonyme
    (ou rien du tout), alors que `save_results` venait d'ecrire le `_v2`.
    `qaoa_inputs.py` (phase 5), l'autre lecteur, applique bien le suffixe :
    c'est le producteur qui divergeait de son propre consommateur.
    """
    return f"exact_diag_{scenario}_Re{Re}_N{N}_dim{n_patches}{suffix}.npz"


def save_results(all_results, meta, outdir=RESULTS_DIR):
    """Save Phase 4 results."""
    if all_results is None:
        return None

    fname = artifact_name(meta["scenario"], meta["Re"], meta["N"],
                          meta["n_patches"], meta.get("suffix", ""))
    path = os.path.join(outdir, fname)

    n = len(all_results)
    dim = meta["n_patches"]
    n_q = meta["n_qubits"]

    energies = np.array([r["ground_energy"] for r in all_results])
    gaps = np.array([r["gap"] for r in all_results])
    marginals = np.array([r["marginals"] for r in all_results])
    decisions_h = np.array([r["decisions_h"] for r in all_results])
    decisions_v = np.array([r["decisions_v"] for r in all_results])
    gt_refine = np.array([r["gt_refine"] for r in all_results])
    promising = np.array([r["promising"] for r in all_results])
    exact_f1 = np.array([r["comparison"]["exact"]["f1"] for r in all_results])
    class_f1 = np.array([r["comparison"]["classical"]["f1"] for r in all_results])
    # D-45 : sans ces trois colonnes, un artefact ou `promising` vaut 100 %
    # est indiscernable d'un artefact ou la porte n'a rien pu rejeter.
    degenerate = np.array([r["degenerate_decision"] for r in all_results])
    degenerate_cl = np.array([r["degenerate_classical"] for r in all_results])
    f1_tie = np.array([r["f1_tie"] for r in all_results])
    informative = np.array([r["promising_informative"] for r in all_results])

    np.savez_compressed(
        path,
        ground_energies=energies,
        gaps=gaps,
        marginals=marginals,
        decisions_h=decisions_h,
        decisions_v=decisions_v,
        gt_refine=gt_refine,
        promising=promising,
        degenerate_decision=degenerate,
        degenerate_classical=degenerate_cl,
        f1_tie=f1_tie,
        promising_informative=informative,
        exact_f1=exact_f1,
        classical_f1=class_f1,
        snap_indices=meta["snap_indices"],
        scenario=meta["scenario"],
        Re=meta["Re"],
        N=meta["N"],
        n_patches=meta["n_patches"],
    )
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {fname} ({size_kb:.0f} KB)")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Exact diagonalization on hard patches")
    parser.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    parser.add_argument("--scenario", nargs="+", default=SCENARIOS)
    parser.add_argument("--dim", nargs="+", type=int, default=VQA_DIMS)
    parser.add_argument("--N", type=int, default=DNS_N)
    parser.add_argument("--v2", action="store_true",
                        help="Use parameter-free v2 Hamiltonian")
    args = parser.parse_args()

    version = "v2" if args.v2 else "v1"
    print(f"Phase 4: Exact diagonalization ({version})")
    print(f"  Patch dims: {args.dim}")
    print()

    all_meta = {}

    for sc in args.scenario:
        for re in args.re:
            for dim in args.dim:
                n_qubits = 2 * dim * dim
                if n_qubits > 20:
                    print(f"  SKIP {sc} Re={re} dim={dim}: "
                          f"{n_qubits} qubits > 20")
                    continue

                dns_path = os.path.join(
                    RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
                patches_path = os.path.join(
                    RESULTS_DIR,
                    f"patches_{sc}_Re{re}_N{args.N}_dim{dim}.npz")

                if not os.path.exists(dns_path):
                    print(f"  SKIP: {dns_path} not found")
                    continue
                if not os.path.exists(patches_path):
                    print(f"  SKIP: {patches_path} not found")
                    continue

                print(f"[{sc} Re={re} dim={dim}]")
                results, meta = run_phase4(
                    dns_path, patches_path, dim, use_v2=args.v2)
                if results is not None:
                    save_results(results, meta)
                    all_meta[(sc, re, dim)] = meta
                print()

    if not all_meta:
        # D-148 : meme famille que D-55/D-56/D-75, sur la phase 4 — celle
        # que `BRIEF_REPRISE.md` §11 demande de RELANCER pour lever les
        # seuils perimes. Mesure : `--scenario no_such_scenario --N 64`
        # sortait avec le code 0 apres avoir imprime « Phase 4 complete. »,
        # sans ecrire d'artefact. Une relance qui ne mesure rien et laisse
        # en place les artefacts perimes se lit alors comme une relance
        # reussie — exactement ce que la famille interdit.
        raise RuntimeError(
            "balayage vide : aucun (scenario, Re, dim) n'a produit de "
            "resultat — entrees manquantes, ou toutes les dimensions "
            "au-dessus du plafond de 20 qubits. Le script sortait ici avec "
            "le code 0 et sans artefact (D-148).")

    # cross-Re summary
    if all_meta:
        print("=" * 60)
        print("PHASE 4 CROSS-Re SUMMARY")
        print("=" * 60)
        for (sc, re, dim), meta in sorted(all_meta.items()):
            ed_path = os.path.join(
                RESULTS_DIR,
                artifact_name(sc, re, meta["N"], dim, meta.get("suffix", "")))
            if os.path.exists(ed_path):
                d = np.load(ed_path)
                n_prom = np.sum(d["promising"])
                n_tot = len(d["promising"])
                # D-45 : ne jamais afficher `promising` sans le nombre de
                # decisions degenerees qui le composent.
                n_deg = (int(np.sum(d["degenerate_decision"]))
                         if "degenerate_decision" in d else -1)
                n_inf = (int(np.sum(d["promising_informative"]))
                         if "promising_informative" in d else -1)
                print(f"  {sc} Re={re} dim={dim}: "
                      f"exact_F1={np.mean(d['exact_f1']):.3f} "
                      f"class_F1={np.mean(d['classical_f1']):.3f} "
                      f"promising={n_prom}/{n_tot} "
                      f"(informative={n_inf}, degenerate={n_deg})")

    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()
