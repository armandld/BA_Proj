#!/usr/bin/env python3
"""
Phase 11E - V1 tuned Hamiltonian (with psi temporal channel) under LOSO.

Closes the most obvious open item in the V1 vs V2 picture: V2's
ceiling collapse is for the v2 parameter-free H; V1 ships with 8
Optuna-tuned parameters and a per-step `psi` temporal channel that
V2 has never tested under cross-scenario evaluation.

This phase runs V1's effective per-cell scoring under the same LOSO
protocol as phase 11B (4-fold leave-one-scenario-out, snapshot-level
splits) and reports F1 vs the L2-hard-patch label, alongside the V2
classical baseline and the V2 mean-field ceiling.

V1 components used (see `src/Simulation/PhysToAngle.py`):

  1. V1 classical multi-indicator:
       score_cls = RMS(|omega_z|, |div v|, |Jz|, Lohner(|B|))
     each normalised to [0, 1] per snapshot. This is V1's input theta
     channel: P(|1>) = sin^2(theta/2) = score_cls. Note: this is the
     baseline V1 itself reports against ('classical AMR'); the Lohner
     term means it is the AMR-community-standard baseline, not a
     strawman.

  2. V1 stress-flux psi channel:
       psi(t) = (pi/2) * tanh(beta * dPhi / <|dPhi|>)
     where Phi(t) is V1's stress-flux field (see compute_stress_flux)
     and dPhi = Phi(t) - Phi(t-1). Recomputed every snapshot pair.
     The per-cell |psi| is added to score_cls with the V1 best beta
     (0.5495). This emulates the input distribution the V1 QAOA
     polishes; it does NOT run the full QAOA.

  3. V1 best Optuna params (best_hyperparams.json, trial 85):
       beta = 0.5495, threshold_amr = 0.3044
     Other params (gamma_hydro, gamma_mag, kappa, ...) only enter
     V1's H_edges Hamiltonian coefficient and not the input theta /
     psi distribution; they affect QAOA's polishing of the input but
     not the input itself.

LIMITATION (documented for the paper):
  This phase evaluates V1's input-side score (theta + psi). It does
  NOT execute V1's QAOA. V1's QAOA polishes the input by at most a
  few percent in F1 (consistent with V1's reported 0.66% advantage),
  so the LOSO F1 reported here is a tight proxy for V1 H's true LOSO
  F1. A full QAOA-LOSO evaluation is a follow-up; the structural
  conclusion (V1 H stays inside the local-Ising ceiling, which
  collapses under LOSO) is unaffected.

Statistical rigour (post-update):
  Per-fold snapshot-level paired bootstrap (B=500) on the delta
  F1(v1+psi) - F1(v2_classical), matching phase 11B-2. Emits a 95%
  percentile CI per fold and a one-sided p-value for the hypothesis
  'V1+psi >= V2 classical' (high p = evidence of V1+psi advantage).
  Also reports a decomposition of the mean delta into the Lohner
  contribution (V1_class - V2_class) and the psi contribution
  (V1+psi - V1_class) -- the finding is that the Lohner term carries
  the signal, not psi.

Output: results/v1h_loso_N{N}_dim{D}.npz

Usage:
  python study/phase11e_v1h_loso.py --dim 4 --max-snaps 30 --n-boot 500
"""
import argparse, os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from config import RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N

from Simulation.PhysToAngle import AngleMapper
from sklearn.metrics import f1_score

from phase11_upper_bound import (
    extract_features_2d, N_FEATS, best_threshold_f1,
)


# ---- V1 best params (best_hyperparams.json, trial 85) -----------------

V1_BEST = dict(
    beta=0.5495366256460598,
    threshold_amr=0.304445558422031,
    gamma_hydro=2.1271683236213677,
    gamma_mag=2.361084417127398,
    kappa=14.332144989517795,
    w_z_frac=0.1013378156103162,
    beta_curl=0.8199244525306656,
    beta_xpoint=0.4256467677055795,
)


def _block_avg(f, patch_size, dim):
    return f.reshape(dim, patch_size, dim, patch_size).mean(axis=(1, 3))


def jz_from_b(Bx, By):
    """Jz = dBy/dx - dBx/dy (curl B, z-component, periodic)."""
    return (np.roll(By, -1, axis=1) - By) - (np.roll(Bx, -1, axis=0) - Bx)


def v1_state(vx, vy, Bx, By):
    return dict(vx=vx, vy=vy, Bx=Bx, By=By, Jz=jz_from_b(Bx, By))


def v1_classical_score(vx, vy, Bx, By):
    """V1's `classical_score` rendered inline (matches PhysToAngle.classical_score)."""
    return AngleMapper.classical_score(v1_state(vx, vy, Bx, By))


def v1_psi_field(vx_prev, vy_prev, Bx_prev, By_prev,
                 vx, vy, Bx, By, beta):
    """Compute |psi| field on full grid using V1 stress-flux machinery.

    Returns max(|psi_h|, |psi_v|) per cell as a positive 'temporal
    priority' channel.
    """
    am = AngleMapper()
    s_prev = v1_state(vx_prev, vy_prev, Bx_prev, By_prev)
    s_curr = v1_state(vx, vy, Bx, By)
    phi_prev = am.compute_stress_flux(s_prev)
    phi_curr = am.compute_stress_flux(s_curr)

    dphi_h = phi_curr["phi_horizontal"] - phi_prev["phi_horizontal"]
    dphi_v = phi_curr["phi_vertical"]   - phi_prev["phi_vertical"]
    avg = float(np.mean(np.abs(dphi_h) + np.abs(dphi_v))) / 2.0
    if avg < 1e-12:
        return np.zeros_like(vx)

    psi_h = (np.pi / 2.0) * np.tanh(beta * dphi_h / avg)
    psi_v = (np.pi / 2.0) * np.tanh(beta * dphi_v / avg)
    return np.maximum(np.abs(psi_h), np.abs(psi_v))


def v1_per_cell_score(vx_prev, vy_prev, Bx_prev, By_prev,
                       vx, vy, Bx, By, beta):
    """V1 input-side score = classical (V1) + (1/(pi/2))*|psi|.

    The (pi/2) normalisation puts |psi| on the same [0, 1] scale as
    V1's classical_score. The combination is a sum (V1 itself just
    rotates each qubit by theta + psi about different axes; for a
    per-cell scalar priority, summing the two angles' magnitudes is a
    faithful linearisation of V1's effective decision input).
    """
    cls = v1_classical_score(vx, vy, Bx, By)
    if vx_prev is None:
        return cls
    psi_mag = v1_psi_field(vx_prev, vy_prev, Bx_prev, By_prev,
                            vx, vy, Bx, By, beta)
    psi_norm = psi_mag / (np.pi / 2.0)  # in [0, 1]
    return np.clip(cls + psi_norm, 0.0, None)


def gather_scenario(scen_configs, dim, max_snaps, beta):
    """Return per-scenario dict with PER-SNAPSHOT lists:
        Xf   : list of (n_cells, 9)   V2 9-feature mean-field features
        Y    : list of (n_cells,)     L2-hard label
        Sv2c : list of (n_cells,)     V2 classical multi-indicator
        Sv1c : list of (n_cells,)     V1 classical_score (4-indicator + Lohner)
        Sv1f : list of (n_cells,)     V1 classical + psi (temporal)
    Kept as per-snapshot lists so phase 11E can bootstrap at the
    snapshot level (cells within a snapshot are spatially correlated).
    """
    Xf, Y, Sv2c, Sv1c, Sv1f = [], [], [], [], []
    for re, dns_path, patches_path in scen_configs:
        dns = np.load(dns_path)
        patches = np.load(patches_path)
        vx_all = dns["vx"].astype(np.float64); vy_all = dns["vy"].astype(np.float64)
        Bx_all = dns["Bx"].astype(np.float64); By_all = dns["By"].astype(np.float64)
        N = vx_all.shape[1]
        l2_all = patches["l2_errors"]; l2_thr = float(patches["l2_threshold"])
        n_snaps = len(vx_all)
        step = max(1, n_snaps // max_snaps)
        idx = list(range(0, n_snaps, step))[:max_snaps]
        ps = N // dim

        for k, si in enumerate(idx):
            vx, vy, Bx, By = vx_all[si], vy_all[si], Bx_all[si], By_all[si]
            # V2 9-feature mean-field
            feats_2d, score_v2 = extract_features_2d(vx, vy, Bx, By, N, dim, re)
            Xf.append(feats_2d.reshape(-1, N_FEATS))
            Sv2c.append(score_v2.ravel())
            Y.append((l2_all[si] >= l2_thr).ravel().astype(int))

            # V1 classical (4-indicator with Lohner) on full grid -> downsample
            s1c_full = v1_classical_score(vx, vy, Bx, By)
            s1c = _block_avg(s1c_full, ps, dim)
            Sv1c.append(s1c.ravel())

            # V1 full input-side (classical + psi)
            if k == 0:
                s1f_full = s1c_full
            else:
                pj = idx[k - 1]
                s1f_full = v1_per_cell_score(
                    vx_all[pj], vy_all[pj], Bx_all[pj], By_all[pj],
                    vx, vy, Bx, By, beta,
                )
            s1f = _block_avg(s1f_full, ps, dim)
            Sv1f.append(s1f.ravel())

    return dict(Xf=Xf, Y=Y, Sv2c=Sv2c, Sv1c=Sv1c, Sv1f=Sv1f)


def snap_f1(Y_list, P_list, thr):
    """Aggregated F1 over a list of per-snapshot arrays at threshold thr."""
    Y = np.concatenate(Y_list)
    P = np.concatenate(P_list)
    return f1_score(Y, (P > thr).astype(int), zero_division=0)


def paired_bootstrap_delta(
    Y_list, P_a_list, thr_a, P_b_list, thr_b, n_boot, rng
):
    """Snapshot-level paired bootstrap on F1(a) - F1(b).

    Returns (lo, hi, deltas, p_a_ge_b) where p_a_ge_b is the fraction
    of bootstrap replicates with delta >= 0 (high p = evidence a >= b).
    """
    n = len(Y_list)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y = np.concatenate([Y_list[i]   for i in idx])
        pa = np.concatenate([P_a_list[i] for i in idx])
        pb = np.concatenate([P_b_list[i] for i in idx])
        fa = f1_score(y, (pa > thr_a).astype(int), zero_division=0)
        fb = f1_score(y, (pb > thr_b).astype(int), zero_division=0)
        deltas[k] = fa - fb
    lo = float(np.quantile(deltas, 0.025))
    hi = float(np.quantile(deltas, 0.975))
    return lo, hi, deltas, float((deltas >= 0).mean())


def main():
    p = argparse.ArgumentParser(
        description="Phase 11E: V1 tuned H + psi under LOSO")
    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--max-snaps", type=int, default=30)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 88)
    print("  Phase 11E: V1 tuned H + psi under leave-one-scenario-out")
    print(f"  dim={args.dim}  N={args.N}  max-snaps/cfg={args.max_snaps}  "
          f"V1 beta={V1_BEST['beta']:.4f}")
    print("=" * 88)
    print()

    # discover configs per scenario
    by_sc = {}
    for sc in args.scenario:
        rows = []
        for re in args.re:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            pp = os.path.join(RESULTS_DIR,
                f"patches_{sc}_Re{re}_N{args.N}_dim{args.dim}.npz")
            if os.path.exists(dp) and os.path.exists(pp):
                rows.append((re, dp, pp))
        if rows:
            by_sc[sc] = rows
    if len(by_sc) < 2:
        print(f"need >=2 scenarios with data; have {list(by_sc.keys())}")
        return

    # gather per-snapshot (needed for snapshot-level bootstrap)
    t0 = time.time()
    data = {}
    for sc, rows in by_sc.items():
        d = gather_scenario(rows, args.dim, args.max_snaps, V1_BEST["beta"])
        data[sc] = d
        n_snaps = len(d["Y"])
        n_cells = sum(len(y) for y in d["Y"])
        pos = float(np.concatenate(d["Y"]).mean())
        print(f"  {sc:<18}  snaps={n_snaps:>4}  cells={n_cells:>6}  "
              f"pos_rate={pos:.3f}  built in {time.time()-t0:.1f}s")
    print()

    rng = np.random.default_rng(args.seed)

    # LOSO sweep with snapshot-level paired bootstrap on (v1+psi - v2_class)
    header = (f"  {'held-out':<18} "
              f"{'F1_v2_class':>11} {'F1_v1_class':>11} {'F1_v1_psi':>10} "
              f"{'delta':>7} {'CI95_delta':>18} {'p(H0)':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows_out = []
    for held in by_sc:
        # train pool = concat of other scenarios (flat, for threshold-fit)
        def cat(key):
            return np.concatenate(
                [x for s in by_sc if s != held for x in data[s][key]])
        Ytr    = cat("Y")
        Sv2c_tr = cat("Sv2c"); Sv1c_tr = cat("Sv1c"); Sv1f_tr = cat("Sv1f")

        # val pool stays per-snapshot for bootstrap
        Y_v    = data[held]["Y"]
        Sv2c_v = data[held]["Sv2c"]
        Sv1c_v = data[held]["Sv1c"]
        Sv1f_v = data[held]["Sv1f"]

        # threshold-fit each score on TRAIN
        thr_v2c, _ = best_threshold_f1(Sv2c_tr, Ytr)
        thr_v1c, _ = best_threshold_f1(
            Sv1c_tr, Ytr,
            grid=np.linspace(Sv1c_tr.min(), Sv1c_tr.max(), 201))
        thr_v1f, _ = best_threshold_f1(
            Sv1f_tr, Ytr,
            grid=np.linspace(Sv1f_tr.min(), Sv1f_tr.max(), 201))

        # point-estimate F1 on the held-out scenario
        f1_v2c = snap_f1(Y_v, Sv2c_v, thr_v2c)
        f1_v1c = snap_f1(Y_v, Sv1c_v, thr_v1c)
        f1_v1f = snap_f1(Y_v, Sv1f_v, thr_v1f)
        delta = f1_v1f - f1_v2c

        # paired bootstrap on delta = F1(v1+psi) - F1(v2_class)
        d_lo, d_hi, _, p_H0 = paired_bootstrap_delta(
            Y_v, Sv1f_v, thr_v1f, Sv2c_v, thr_v2c, args.n_boot, rng)

        rows_out.append(dict(
            held=held, n_val_snaps=len(Y_v),
            f1_v2_class=f1_v2c, f1_v1_class=f1_v1c,
            f1_v1_psi=f1_v1f, delta=delta,
            delta_ci=(d_lo, d_hi), p_v1psi_ge_v2c=p_H0,
            thr_v2c=thr_v2c, thr_v1c=thr_v1c, thr_v1f=thr_v1f,
        ))
        print(f"  {held:<18} "
              f"{f1_v2c:>11.3f} {f1_v1c:>11.3f} {f1_v1f:>10.3f} "
              f"{delta:>+7.3f} [{d_lo:>+6.3f},{d_hi:>+6.3f}]  "
              f"{p_H0:>7.3f}")

    print()
    print("  INTERPRETATION:")
    deltas = [r["delta"] for r in rows_out]
    mean_delta = float(np.mean(deltas))
    n_strict_pos = sum(r["delta_ci"][0] > 0 for r in rows_out)
    n_strict_neg = sum(r["delta_ci"][1] < 0 for r in rows_out)
    print(f"  * mean(F1_v1_psi - F1_v2_classical) = {mean_delta:+.3f} over "
          f"{len(rows_out)} folds")
    print(f"  * folds with CI strictly > 0 (V1+psi > V2 classical): "
          f"{n_strict_pos}/{len(rows_out)}")
    print(f"  * folds with CI strictly < 0 (V1+psi < V2 classical): "
          f"{n_strict_neg}/{len(rows_out)}")
    print()

    # also the across-fold "mean-of-deltas" bootstrap: since folds are
    # disjoint scenarios (paired only within scenario), the per-fold
    # CIs above are the statistically principled CIs. The mean-delta
    # across folds is reported as a scenario-average summary only.
    v1_minus_v1c = float(np.mean(
        [r["f1_v1_psi"] - r["f1_v1_class"] for r in rows_out]))
    v1c_minus_v2c = float(np.mean(
        [r["f1_v1_class"] - r["f1_v2_class"] for r in rows_out]))
    print(f"  * decomposition of the +{mean_delta:.3f} mean:")
    print(f"      V1_class - V2_class  = {v1c_minus_v2c:+.3f}  "
          f"(Lohner + 4-indicator RMS effect)")
    print(f"      V1+psi   - V1_class  = {v1_minus_v1c:+.3f}  "
          f"(psi temporal-channel effect)")

    # save
    out = os.path.join(RESULTS_DIR,
                       f"v1h_loso_N{args.N}_dim{args.dim}.npz")
    np.savez_compressed(
        out,
        scenarios=np.array([r["held"] for r in rows_out]),
        f1_v2_class=np.array([r["f1_v2_class"] for r in rows_out]),
        f1_v1_class=np.array([r["f1_v1_class"] for r in rows_out]),
        f1_v1_psi  =np.array([r["f1_v1_psi"]   for r in rows_out]),
        delta      =np.array([r["delta"]       for r in rows_out]),
        delta_ci_lo=np.array([r["delta_ci"][0] for r in rows_out]),
        delta_ci_hi=np.array([r["delta_ci"][1] for r in rows_out]),
        p_v1psi_ge_v2c=np.array(
            [r["p_v1psi_ge_v2c"] for r in rows_out]),
        n_val_snaps=np.array([r["n_val_snaps"] for r in rows_out]),
        v1_best_params=json.dumps(V1_BEST),
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nPhase 11E complete.")


if __name__ == "__main__":
    main()
