#!/usr/bin/env python3
"""Phase 7: simulated annealing on the Ising objective used by QAOA.

This module also contains the shared Ising-term builder and exact enumerator.
Decisions are compared descriptively to the L2-hard labels and the classical
threshold; inferential claims belong to the paired trajectory analysis.

Per spin convention: Z = +1 is "don't refine" (|0>), Z = -1 is "refine"
(|1>). So decision[i] = 1 iff spin[i] == -1. The Hamiltonian energy for
a classical spin configuration s ∈ {-1,+1}^N is:

    E(s) = sum_i  H_i  s_i
         + sum_ij C_ij s_i s_j
         + sum_pq K_p  s_i s_j s_k s_l  (plaquette 4-body)

SA minimises E(s) — the same objective QAOA chases.

Input:  results/dns_{scenario}_Re{Re}_N{N}.npz
        results/patches_{scenario}_Re{Re}_N{N}_dim{D}.npz
        (optionally) results/exact_diag_{scenario}_Re{Re}_N{N}_dim{D}{sfx}.npz
Output: results/sa_baseline_{scenario}_Re{Re}_N{N}_dim{D}{sfx}.npz

Usage:
  python study/common/ising_terms_and_annealing.py --dim 4
"""

# Must match ``VQA.cost_hamiltonian.COEFF_MIN`` so every solver sees the same
# Hamiltonian.
COEFF_MIN = 1e-6
TOL_E = 1e-9
MAX_ENUM_QUBITS = 22

import argparse, json, os, sys, time
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
    TRAINED_THRESHOLD, trained_mapper_params,
    V2_THRESHOLD,
)
from exact_diagonalisation import build_patch_hamiltonian
import provenance


# -------------------------------------------------------------------
# Ising energy for a single spin configuration
# -------------------------------------------------------------------

def _idx_H(y, x, dim):
    return (y % dim) * dim + (x % dim)

def _idx_V(y, x, dim):
    return dim * dim + (y % dim) * dim + (x % dim)


def build_ising_terms(hamilt_params, dim):
    """Flatten the Hamiltonian coefficients for classical energy evaluation.

    Returns
    -------
    Returns local biases and separate integer-index/float-coefficient tuples
    for the unique periodic ZZ links and all ZZZZ plaquette terms. At dim=2,
    opposite periodic traversals denote the same undirected ZZ link and are
    therefore emitted once, matching the circuit Hamiltonian.
    """
    n_q = 2 * dim * dim
    h_bias = np.zeros(n_q, dtype=np.float64)

    # H_edges: two (dim, dim) arrays for horizontal & vertical edge qubits
    H0, H1 = hamilt_params["H_edges"]
    for i in range(dim):
        for j in range(dim):
            if abs(H0[i, j]) > COEFF_MIN:
                h_bias[_idx_H(i, j, dim)] += float(H0[i, j])
            if abs(H1[i, j]) > COEFF_MIN:
                h_bias[_idx_V(i, j, dim)] += float(H1[i, j])

    # C_edges: ZZ interactions
    C0, C1 = hamilt_params["C_edges"]
    edge_idx = []
    edge_coef = []
    # Deduplicate undirected periodic links, including the dim=2 alias.
    _liens_zz_emis = set()

    def _lien_zz_neuf(a, b):
        cle = (a, b) if a <= b else (b, a)
        if cle in _liens_zz_emis:
            return False
        _liens_zz_emis.add(cle)
        return True

    for i in range(dim):
        for j in range(dim):
            if abs(C0[i, j]) > COEFF_MIN:
                a, b = _idx_H(i, j, dim), _idx_H(i, j + 1, dim)
                if _lien_zz_neuf(a, b):
                    edge_idx.append((a, b))
                    edge_coef.append(float(C0[i, j]))
            if abs(C1[i, j]) > COEFF_MIN:
                a, b = _idx_V(i, j, dim), _idx_V(i + 1, j, dim)
                if _lien_zz_neuf(a, b):
                    edge_idx.append((a, b))
                    edge_coef.append(float(C1[i, j]))
    edge_idx = np.asarray(edge_idx, dtype=np.int64).reshape(-1, 2)
    edge_coef = np.asarray(edge_coef, dtype=np.float64)

    # K_plaquettes: ZZZZ interactions
    plaq_idx = []
    plaq_coef = []
    K = hamilt_params.get("K_plaquettes")
    if K is not None:
        for i in range(dim):
            for j in range(dim):
                kv = float(K[i, j])
                if abs(kv) > COEFF_MIN:
                    plaq_idx.append((
                        _idx_H(i, j, dim),
                        _idx_V(i, j + 1, dim),
                        _idx_H(i + 1, j, dim),
                        _idx_V(i, j, dim),
                    ))
                    plaq_coef.append(kv)
    # K_xpoint is a second ZZZZ coefficient on the same plaquette topology.
    KX = hamilt_params.get("K_xpoint")
    if KX is not None:
        for i in range(dim):
            for j in range(dim):
                kv = float(KX[i, j])
                if abs(kv) > COEFF_MIN:
                    plaq_idx.append((
                        _idx_H(i, j, dim),
                        _idx_V(i, j + 1, dim),
                        _idx_H(i + 1, j, dim),
                        _idx_V(i, j, dim),
                    ))
                    plaq_coef.append(kv)

    plaq_idx = np.asarray(plaq_idx, dtype=np.int64).reshape(-1, 4)
    plaq_coef = np.asarray(plaq_coef, dtype=np.float64)

    return h_bias, (edge_idx, edge_coef), (plaq_idx, plaq_coef)


def total_energy(spins, h_bias, edges, plaqs):
    """Evaluate the full Ising + 4-body energy E(s). spins in {-1,+1}."""
    edge_idx, edge_coef = edges
    plaq_idx, plaq_coef = plaqs
    e = float(np.dot(h_bias, spins))
    if len(edge_coef):
        e += float(np.sum(edge_coef * spins[edge_idx[:, 0]]
                                      * spins[edge_idx[:, 1]]))
    if len(plaq_coef):
        e += float(np.sum(plaq_coef * spins[plaq_idx[:, 0]]
                                      * spins[plaq_idx[:, 1]]
                                      * spins[plaq_idx[:, 2]]
                                      * spins[plaq_idx[:, 3]]))
    return e


def _enumerated_energy_chunks(h_bias, edges, plaqs, n_q, chunk):
    """Yield ``(start, spins, energies)`` over the computational basis."""
    edge_idx, edge_coef = edges
    plaq_idx, plaq_coef = plaqs
    bit = 1 << np.arange(n_q, dtype=np.int64)
    for start in range(0, 1 << n_q, chunk):
        stop = min(start + chunk, 1 << n_q)
        indices = np.arange(start, stop, dtype=np.int64)[:, None]
        spins = np.where((indices & bit) > 0, -1.0, 1.0)
        energies = spins @ h_bias
        if len(edge_coef):
            energies += (
                spins[:, edge_idx[:, 0]] * spins[:, edge_idx[:, 1]]
            ) @ edge_coef
        if len(plaq_coef):
            energies += (
                spins[:, plaq_idx[:, 0]] * spins[:, plaq_idx[:, 1]]
                * spins[:, plaq_idx[:, 2]] * spins[:, plaq_idx[:, 3]]
            ) @ plaq_coef
        yield start, spins, energies


def exhaustive_ground_state(h_bias, edges, plaqs, n_q,
                            max_qubits=MAX_ENUM_QUBITS, chunk=1 << 16):
    """Return the exact minimizing spin state, energy and degeneracy."""
    if not isinstance(n_q, (int, np.integer)) or n_q < 1:
        raise ValueError("n_q must be a positive integer")
    if n_q > max_qubits:
        raise ValueError(
            f"exhaustive enumeration refused for {n_q} qubits "
            f"(max {max_qubits})")
    if not isinstance(chunk, (int, np.integer)) or chunk < 1:
        raise ValueError("chunk must be a positive integer")

    best_energy, best_spins = np.inf, None
    for _, spins, energies in _enumerated_energy_chunks(
            h_bias, edges, plaqs, n_q, int(chunk)):
        index = int(np.argmin(energies))
        if energies[index] < best_energy - TOL_E:
            best_energy = float(energies[index])
            best_spins = spins[index].astype(np.int8).copy()

    degeneracy = 0
    for _, _, energies in _enumerated_energy_chunks(
            h_bias, edges, plaqs, n_q, int(chunk)):
        degeneracy += int(np.sum(energies <= best_energy + TOL_E))
    if best_spins is None or degeneracy < 1:
        raise RuntimeError("exhaustive enumeration found no ground state")
    return best_spins, best_energy, degeneracy


def exact_ising_spectrum(h_bias, edges, plaqs, n_q,
                         max_qubits=20, chunk=1 << 16):
    """Return one exact ground state and the sorted diagonal spectrum."""
    if not isinstance(n_q, (int, np.integer)) or n_q < 1:
        raise ValueError("n_q must be a positive integer")
    if n_q > max_qubits:
        raise ValueError(
            f"exact spectrum refused for {n_q} qubits (max {max_qubits})")
    if not isinstance(chunk, (int, np.integer)) or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    spectrum = np.empty(1 << n_q, dtype=np.float64)
    best_energy, best_spins = np.inf, None
    for start, spins, energies in _enumerated_energy_chunks(
            h_bias, edges, plaqs, n_q, int(chunk)):
        spectrum[start:start + len(energies)] = energies
        index = int(np.argmin(energies))
        if energies[index] < best_energy - TOL_E:
            best_energy = float(energies[index])
            best_spins = spins[index].astype(np.int8).copy()
    spectrum.sort()
    if best_spins is None:
        raise RuntimeError("exact spectrum found no ground state")
    return best_spins, best_energy, spectrum


def delta_energy(spins, q, h_bias, edges, plaqs,
                 edges_by_q, plaqs_by_q):
    """Energy change from flipping spin q. O(neighbours of q)."""
    edge_idx, edge_coef = edges
    plaq_idx, plaq_coef = plaqs

    # h term: flipping flips sign → contribution changes by -2 h_q s_q
    dE = -2.0 * h_bias[q] * spins[q]

    # ZZ edges touching q
    for e_idx in edges_by_q[q]:
        i, j = edge_idx[e_idx]
        dE += -2.0 * edge_coef[e_idx] * spins[i] * spins[j]

    # ZZZZ plaquettes touching q
    for p_idx in plaqs_by_q[q]:
        i, j, k, l = plaq_idx[p_idx]
        dE += -2.0 * plaq_coef[p_idx] * spins[i] * spins[j] * spins[k] * spins[l]

    return dE


def coefficient_scale(h_bias, edges, plaqs):
    """Largest absolute Ising coefficient, used only to scale temperature."""
    arrays = [np.asarray(h_bias), np.asarray(edges[1]), np.asarray(plaqs[1])]
    maxima = [float(np.max(np.abs(a))) for a in arrays if a.size]
    scale = max(maxima, default=0.0)
    if not np.isfinite(scale):
        raise ValueError("Ising coefficients must be finite")
    return scale if scale > 0.0 else 1.0


def _build_incidence(n_q, edges, plaqs):
    edge_idx, _ = edges
    plaq_idx, _ = plaqs
    edges_by_q = [[] for _ in range(n_q)]
    plaqs_by_q = [[] for _ in range(n_q)]
    for e_idx, (i, j) in enumerate(edge_idx):
        edges_by_q[i].append(e_idx)
        edges_by_q[j].append(e_idx)
    for p_idx, (i, j, k, l) in enumerate(plaq_idx):
        for q in (i, j, k, l):
            plaqs_by_q[q].append(p_idx)
    return edges_by_q, plaqs_by_q


# -------------------------------------------------------------------
# Simulated annealing
# -------------------------------------------------------------------

def simulated_annealing(h_bias, edges, plaqs, n_q,
                        sweeps=2000, T_start=None, T_end=None,
                        rng=None, init_spins=None):
    """Metropolis SA with a geometric cooling schedule.

    Returns best spins, best_energy, trace (energy vs sweep)
    """
    if rng is None:
        rng = np.random.default_rng()
    scale = coefficient_scale(h_bias, edges, plaqs)
    T_start = scale if T_start is None else float(T_start)
    T_end = 0.005 * scale if T_end is None else float(T_end)
    if not (np.isfinite(T_start) and np.isfinite(T_end)
            and T_start > 0.0 and T_end > 0.0 and T_start >= T_end):
        raise ValueError("temperatures must be finite with T_start >= T_end > 0")
    edges_by_q, plaqs_by_q = _build_incidence(n_q, edges, plaqs)

    # initial configuration
    if init_spins is None:
        spins = rng.choice([-1, 1], size=n_q).astype(np.int8)
    else:
        spins = np.array(init_spins, dtype=np.int8).copy()

    current_E = total_energy(spins.astype(np.float64), h_bias, edges, plaqs)
    best_E = current_E
    best_spins = spins.copy()

    # geometric schedule
    if sweeps <= 1:
        Ts = np.array([T_start])
    else:
        Ts = T_start * (T_end / T_start) ** (np.arange(sweeps) / (sweeps - 1))

    trace = np.empty(sweeps, dtype=np.float64)

    spins_f = spins.astype(np.float64)
    for sweep, T in enumerate(Ts):
        # one sweep = one attempted flip per spin, in random order
        order = rng.permutation(n_q)
        for q in order:
            dE = delta_energy(spins_f, q, h_bias, edges, plaqs,
                              edges_by_q, plaqs_by_q)
            if dE <= 0.0 or rng.random() < np.exp(-dE / max(T, 1e-12)):
                spins_f[q] = -spins_f[q]
                current_E += dE
                if current_E < best_E:
                    best_E = current_E
                    best_spins = spins_f.astype(np.int8).copy()
        trace[sweep] = current_E

    return best_spins, best_E, trace


def sa_multi_restart(h_bias, edges, plaqs, n_q,
                     sweeps=2000, n_restarts=10,
                     T_start=None, T_end=None, rng=None,
                     classical_init=None):
    """Run SA n_restarts times; return the best solution across restarts.

    classical_init: optional initial spin array used as restart 0.
    """
    if rng is None:
        rng = np.random.default_rng()
    best_E = np.inf
    best_spins = None
    all_energies = []
    for r in range(n_restarts):
        init = classical_init if (r == 0 and classical_init is not None) else None
        spins, E, _ = simulated_annealing(
            h_bias, edges, plaqs, n_q,
            sweeps=sweeps, T_start=T_start, T_end=T_end,
            rng=rng, init_spins=init,
        )
        all_energies.append(E)
        if E < best_E:
            best_E = E
            best_spins = spins
    return best_spins, best_E, np.asarray(all_energies)


# -------------------------------------------------------------------
# Decision extraction + metrics
# -------------------------------------------------------------------

def spins_to_decisions(spins, dim):
    """Convert spin array to (decisions_h, decisions_v) bool arrays.
    Convention: Z = +1 -> no refine, Z = -1 -> refine.
    """
    n_cells = dim * dim
    refine = (spins == -1)
    dec_h = refine[:n_cells].reshape(dim, dim)
    dec_v = refine[n_cells:].reshape(dim, dim)
    return dec_h, dec_v


def _metrics(pred, gt):
    tp = np.sum(pred & gt);  fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt); tn = np.sum(~pred & ~gt)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-10)
    return {"accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


# -------------------------------------------------------------------
# Snapshot driver
# -------------------------------------------------------------------

def analyze_snapshot_sa(vx, vy, Bx, By, N, dim, Re,
                        l2_errors, l2_threshold,
                        use_v2=True, sweeps=2000, n_restarts=10,
                        classical_warm=False, rng=None):
    thr_amr = V2_THRESHOLD if use_v2 else TRAINED_THRESHOLD
    if use_v2:
        hp, score_vqa, _ = build_patch_hamiltonian(
            vx, vy, Bx, By, N, dim, Re,
            threshold_amr=thr_amr, use_v2=True,
        )
    else:
        hp, score_vqa, _ = build_patch_hamiltonian(
            vx, vy, Bx, By, N, dim, Re,
            threshold_amr=thr_amr,
            **trained_mapper_params(),
        )

    h_bias, edges, plaqs = build_ising_terms(hp, dim)
    n_q = 2 * dim * dim

    # optional classical warm start: build initial spin config from classical
    # decision (refine iff score > thr → spin = -1 else +1).
    classical_refine = score_vqa > thr_amr
    if classical_warm:
        init_spins = np.ones(n_q, dtype=np.int8)
        flat_h = classical_refine.flatten().astype(bool)
        flat_v = classical_refine.flatten().astype(bool)  # same per-cell decision
        init_spins[:dim*dim]   = np.where(flat_h, -1, 1)
        init_spins[dim*dim:]   = np.where(flat_v, -1, 1)
    else:
        init_spins = None

    t0 = time.time()
    best_spins, best_E, all_E = sa_multi_restart(
        h_bias, edges, plaqs, n_q,
        sweeps=sweeps, n_restarts=n_restarts, rng=rng,
        classical_init=init_spins,
    )
    wall = time.time() - t0

    dec_h, dec_v = spins_to_decisions(best_spins, dim)
    sa_refine = dec_h | dec_v
    gt_refine = l2_errors >= l2_threshold

    return {
        "best_E": float(best_E),
        "all_restart_E": all_E,
        "sa_refine": sa_refine,
        "classical_refine": classical_refine,
        "gt_refine": gt_refine,
        "metrics_sa": _metrics(sa_refine, gt_refine),
        "metrics_classical": _metrics(classical_refine, gt_refine),
        "wall_time": wall,
    }


def run_phase7(dns_path, patches_path, dim, *,
               use_v2=True, sweeps=2000, n_restarts=10,
               classical_warm=False, seed=0):
    dns = np.load(dns_path)
    patches = np.load(patches_path)

    vx_all = dns["vx"]; vy_all = dns["vy"]
    Bx_all = dns["Bx"]; By_all = dns["By"]
    N = vx_all.shape[1]
    Re = int(dns.get("meta_Re", 800))
    scenario = str(dns.get("meta_scenario", "unknown"))

    l2_all = patches["l2_errors"]
    l2_threshold = float(patches["l2_threshold"])

    n_snaps = len(vx_all)
    snap_indices = list(range(0, n_snaps, max(1, n_snaps // 10)))
    if len(snap_indices) < 3:
        snap_indices = list(range(n_snaps))

    n_q = 2 * dim * dim
    print(f"  {scenario} Re={Re} dim={dim} ({n_q} qubits), "
          f"{len(snap_indices)} snapshots, sweeps={sweeps}, restarts={n_restarts}")

    rng = np.random.default_rng(seed)
    all_results = []
    for si in snap_indices:
        r = analyze_snapshot_sa(
            vx_all[si].astype(np.float64),
            vy_all[si].astype(np.float64),
            Bx_all[si].astype(np.float64),
            By_all[si].astype(np.float64),
            N, dim, Re,
            l2_all[si], l2_threshold,
            use_v2=use_v2, sweeps=sweeps, n_restarts=n_restarts,
            classical_warm=classical_warm, rng=rng,
        )
        r["snap_idx"] = int(si)
        all_results.append(r)
        print(f"    snap {si:3d}: SA_E={r['best_E']:.4f} "
              f"SA_F1={r['metrics_sa']['f1']:.3f} "
              f"class_F1={r['metrics_classical']['f1']:.3f} "
              f"({r['wall_time']:.1f}s)")

    sa_f1s    = [r["metrics_sa"]["f1"]        for r in all_results]
    class_f1s = [r["metrics_classical"]["f1"] for r in all_results]
    print(f"\n  SA F1:         mean={np.mean(sa_f1s):.3f} std={np.std(sa_f1s):.3f}")
    print(f"  Classical F1:  mean={np.mean(class_f1s):.3f} std={np.std(class_f1s):.3f}")

    meta = {
        "scenario": scenario, "Re": Re, "N": N, "dim": dim,
        "snap_indices": np.array(snap_indices),
        "suffix": "_v2" if use_v2 else "",
        "sweeps": sweeps, "n_restarts": n_restarts,
        "classical_warm": classical_warm,
        "seed": int(seed),
    }
    return all_results, meta


def save_results(all_results, meta, outdir=RESULTS_DIR, *,
                 run_provenance=None, cli_args=None):
    if not all_results:
        return None
    variant = (f"_sweeps{meta['sweeps']}_r{meta['n_restarts']}"
               f"_seed{meta['seed']}"
               + ("_warm" if meta.get("classical_warm") else ""))
    suffix = variant + meta.get("suffix", "")
    fname = (f"sa_baseline_{meta['scenario']}_Re{meta['Re']}"
             f"_N{meta['N']}_dim{meta['dim']}{suffix}.npz")
    path = os.path.join(outdir, fname)

    best_E = np.array([r["best_E"] for r in all_results])
    sa_f1  = np.array([r["metrics_sa"]["f1"] for r in all_results])
    cl_f1  = np.array([r["metrics_classical"]["f1"] for r in all_results])
    wall   = np.array([r["wall_time"] for r in all_results])
    sa_refine = np.array([r["sa_refine"] for r in all_results])
    gt_refine = np.array([r["gt_refine"] for r in all_results])

    provenance_fields = (
        provenance.finish(run_provenance)
        if run_provenance is not None else {})
    np.savez_compressed(
        path,
        best_E=best_E,
        sa_f1=sa_f1, classical_f1=cl_f1,
        wall_times=wall,
        sa_refine=sa_refine,
        gt_refine=gt_refine,
        snap_indices=meta["snap_indices"],
        scenario=meta["scenario"],
        Re=meta["Re"], N=meta["N"], dim=meta["dim"],
        sweeps=meta["sweeps"], n_restarts=meta["n_restarts"],
        seed=meta["seed"],
        classical_warm=meta.get("classical_warm", False),
        cli_args="" if cli_args is None else cli_args,
        **provenance_fields,
    )
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {fname} ({size_kb:.0f} KB)")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: Simulated Annealing baseline on the same QUBO")
    parser.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    parser.add_argument("--scenario", nargs="+", default=SCENARIOS)
    parser.add_argument("--dim", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--N", type=int, default=DNS_N)
    parser.add_argument("--sweeps", type=int, default=2000)
    parser.add_argument("--n-restarts", type=int, default=10)
    parser.add_argument("--v1", action="store_true",
                        help="Use v1 trained Hamiltonian (default: v2)")
    parser.add_argument("--classical-warm", action="store_true",
                        help="Use classical AMR decision as SA restart 0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_provenance = provenance.start()
    cli_args = json.dumps(vars(args), sort_keys=True)

    use_v2 = not args.v1
    version = "v2" if use_v2 else "v1"
    print(f"Phase 7: SA baseline on the {version} Hamiltonian")
    print(f"  Dims: {args.dim}  sweeps={args.sweeps}  restarts={args.n_restarts}")
    if args.classical_warm:
        print(f"  Classical warm-start: ON")
    print()

    summary = {}
    for sc in args.scenario:
        for re in args.re:
            for dim in args.dim:
                dns_path = os.path.join(
                    RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
                patches_path = os.path.join(
                    RESULTS_DIR,
                    f"patches_{sc}_Re{re}_N{args.N}_dim{dim}.npz")
                if not (os.path.exists(dns_path)
                        and os.path.exists(patches_path)):
                    print(f"  SKIP {sc} Re={re} dim={dim}: missing input")
                    continue

                print(f"[{sc} Re={re} dim={dim}]")
                results, meta = run_phase7(
                    dns_path, patches_path, dim,
                    use_v2=use_v2, sweeps=args.sweeps,
                    n_restarts=args.n_restarts,
                    classical_warm=args.classical_warm,
                    seed=args.seed,
                )
                save_results(
                    results, meta,
                    run_provenance=run_provenance,
                    cli_args=cli_args)
                summary[(sc, re, dim)] = {
                    "sa_f1":    np.mean([r["metrics_sa"]["f1"]        for r in results]),
                    "class_f1": np.mean([r["metrics_classical"]["f1"] for r in results]),
                    "wall":     np.mean([r["wall_time"]               for r in results]),
                }
                print()

    if not summary:
        raise RuntimeError(
            "empty sweep: no (scenario, Re, dim) has both input artifacts")

    if summary:
        print("=" * 70)
        print("PHASE 7 SUMMARY (SA on the same Hamiltonian)")
        print("=" * 70)
        print(f"  {'scenario':<18} {'Re':>5} {'dim':>4} "
              f"{'SA_F1':>7} {'Class_F1':>9} {'wall(s)':>9}")
        for (sc, re, dim), s in sorted(summary.items()):
            print(f"  {sc:<18} {re:>5} {dim:>4} "
                  f"{s['sa_f1']:>7.3f} {s['class_f1']:>9.3f} "
                  f"{s['wall']:>9.2f}")

        mean_sa    = np.mean([s["sa_f1"]    for s in summary.values()])
        mean_class = np.mean([s["class_f1"] for s in summary.values()])
        print(f"\n  Overall SA F1:        {mean_sa:.3f}")
        print(f"  Overall Classical F1: {mean_class:.3f}")
        print(f"  Descriptive delta:      {mean_sa - mean_class:+.3f}")
        print("  Interpret with paired trajectory-level uncertainty; this phase "
              "alone does not establish superiority or equivalence.")

    print("\nPhase 7 complete.")


if __name__ == "__main__":
    main()
