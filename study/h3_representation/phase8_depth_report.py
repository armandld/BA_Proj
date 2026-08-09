#!/usr/bin/env python3
"""
Phase 8 - Circuit depth & pruning report.

For each (scenario, Re, dim), builds the v2 Hamiltonian, compiles the
QAOAAnsatz for reps=2, and reports:

  - # Z / ZZ / ZZZZ terms before and after coefficient pruning
  - compiled circuit depth (decomposed to 1q+2q gates)
  - compiled 2-qubit gate count
  - compiled overall gate count

This gives real citeable numbers for the thesis: "at dim=4 the v2
Hamiltonian compiles to D two-qubit gates per QAOA layer; pruning at
eps=0.1 drops this to D' with no measurable F1 loss."

Output: results/depth_report_N{N}_dim{D}{sfx}.csv (per-run rows)
        stdout summary table

Usage:
  python study/phase8_depth_report.py --dim 2 3 4
  python study/phase8_depth_report.py --dim 4 --prune-eps 0 0.05 0.1 0.2
"""
import argparse, csv, os, sys
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
from config import RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N

from phase4_exact_diag import build_patch_hamiltonian
from phase5_qaoa_eval import prune_hamilt_params
from VQA.cost_hamiltonian import create_period_hamiltonian

from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def _count_terms(spop):
    """Return (nZ, nZZ, nZZZZ) counts in a SparsePauliOp."""
    nZ = nZZ = nZZZZ = nOther = 0
    for pauli, _c in spop.to_list():
        w = pauli.count("Z")
        if   w == 1: nZ += 1
        elif w == 2: nZZ += 1
        elif w == 4: nZZZZ += 1
        else:        nOther += 1
    return nZ, nZZ, nZZZZ, nOther


def build_compiled_qaoa(hamilt_params, dim, reps, backend):
    """Build + fully decompose QAOAAnsatz, run transpile at opt_level=0."""
    H_op = create_period_hamiltonian(hamilt_params, dim)
    ansatz = QAOAAnsatz(cost_operator=H_op, reps=reps)
    qc = ansatz.decompose().decompose()
    pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
    qc_t = pm.run(qc)
    return qc_t, H_op


def report_row(scenario, Re, dim, N, reps, prune_eps, backend,
               vx_all, vy_all, Bx_all, By_all, snap_idx):
    vx = vx_all[snap_idx].astype(np.float64)
    vy = vy_all[snap_idx].astype(np.float64)
    Bx = Bx_all[snap_idx].astype(np.float64)
    By = By_all[snap_idx].astype(np.float64)

    hp, _, _ = build_patch_hamiltonian(
        vx, vy, Bx, By, N, dim, Re, threshold_amr=0.15, use_v2=True,
    )
    hp_pruned = prune_hamilt_params(hp, prune_eps) if prune_eps > 0 else hp

    # term count before compilation
    H_op_full = create_period_hamiltonian(hp, dim)
    H_op_prun = create_period_hamiltonian(hp_pruned, dim)
    nZ_f, nZZ_f, nZZZZ_f, _ = _count_terms(H_op_full)
    nZ_p, nZZ_p, nZZZZ_p, _ = _count_terms(H_op_prun)

    # compile
    qc_t, _ = build_compiled_qaoa(hp_pruned, dim, reps, backend)

    depth = qc_t.depth()
    gate_counts = qc_t.count_ops()
    total_gates = sum(gate_counts.values())
    # Aer's native 2q gate on statevector/MPS is typically cx / cz / rzz
    two_q_ops = sum(v for k, v in gate_counts.items()
                    if k in ("cx", "cz", "ecr", "cp", "rzz", "rzx", "swap"))

    return {
        "scenario": scenario, "Re": Re, "dim": dim, "N": N,
        "reps": reps, "prune_eps": prune_eps,
        "nZ_full": nZ_f, "nZZ_full": nZZ_f, "nZZZZ_full": nZZZZ_f,
        "nZ_pruned": nZ_p, "nZZ_pruned": nZZ_p, "nZZZZ_pruned": nZZZZ_p,
        "terms_dropped_frac": 1.0 - ((nZ_p + nZZ_p + nZZZZ_p)
                                     / max(nZ_f + nZZ_f + nZZZZ_f, 1)),
        "depth": int(depth),
        "two_q_gates": int(two_q_ops),
        "total_gates": int(total_gates),
        "num_qubits": qc_t.num_qubits,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8: circuit depth & pruning report")
    parser.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    parser.add_argument("--scenario", nargs="+", default=SCENARIOS)
    parser.add_argument("--dim", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--N", type=int, default=DNS_N)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--prune-eps", nargs="+", type=float,
                        default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--snap-idx", type=int, default=None,
                        help="Snapshot to profile (default = middle).")
    parser.add_argument("--out-csv", default=None,
                        help="Optional CSV output path (default: auto).")
    args = parser.parse_args()

    backend = AerSimulator(method='matrix_product_state')

    rows = []
    print("=" * 100)
    print("  Phase 8: QAOA compiled-circuit depth report (v2 Hamiltonian)")
    print("=" * 100)
    hdr = (f"  {'scenario':<18} {'Re':>5} {'dim':>4} {'eps':>5} "
           f"{'nZZ_f':>6} {'nZZ_p':>6} {'nK_f':>5} {'nK_p':>5} "
           f"{'drop%':>6} {'depth':>6} {'2q':>5} {'total':>7}")
    print(hdr)
    print("  " + "-" * 96)

    for sc in args.scenario:
        for re in args.re:
            dns_path = os.path.join(
                RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            if not os.path.exists(dns_path):
                continue
            dns = np.load(dns_path)
            vx_all = dns["vx"]; vy_all = dns["vy"]
            Bx_all = dns["Bx"]; By_all = dns["By"]
            n_snaps = len(vx_all)
            si = args.snap_idx if args.snap_idx is not None else n_snaps // 2

            for dim in args.dim:
                for eps in args.prune_eps:
                    try:
                        row = report_row(
                            sc, re, dim, args.N, args.reps, eps, backend,
                            vx_all, vy_all, Bx_all, By_all, si,
                        )
                    except Exception as e:
                        print(f"  FAIL {sc} Re={re} dim={dim} eps={eps}: {e}")
                        continue
                    rows.append(row)
                    print(f"  {sc:<18} {re:>5} {dim:>4} {eps:>5.2f} "
                          f"{row['nZZ_full']:>6} {row['nZZ_pruned']:>6} "
                          f"{row['nZZZZ_full']:>5} {row['nZZZZ_pruned']:>5} "
                          f"{100*row['terms_dropped_frac']:>5.1f} "
                          f"{row['depth']:>6} {row['two_q_gates']:>5} "
                          f"{row['total_gates']:>7}")

    # write csv
    if rows:
        out = args.out_csv or os.path.join(
            RESULTS_DIR, f"depth_report_N{args.N}.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  CSV: {out}")

    print("\nPhase 8 complete.")


if __name__ == "__main__":
    main()
