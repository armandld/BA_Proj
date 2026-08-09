#!/usr/bin/env python3
"""
Phase 10 - Closed-loop training of the v2 Hamiltonian.

Trains theta = (c_bias, thr_amr) against the L2-hard mask F1 by running
an actual optimiser in the loop:

    for step in range(n_steps):
        theta_k = optimiser.ask()
        H_k     = build_H(theta_k)          # v2: w_zz=2, w_zzzz=1 fixed
        spins_k = SA(H_k)
        F1_k    = score(spins_k vs GT)      # averaged over train batch
        optimiser.tell(theta_k, -F1_k)

Three training modes are run back-to-back and compared:

  (1) per-config   : one (c*, thr*) per (scenario, Re)   [N_s * N_Re runs]
  (2) per-scenario : one (c*, thr*) per scenario         [N_s runs]
  (3) joint        : one (c*, thr*) over ALL configs     [1 run]

Rationale: a joint fit may fail if the optimal (c*, thr*) varies across
scenarios (e.g. rotor vs Orszag-Tang have different circulation
topologies, so the right c_bias / thr_amr differs). Running all three
and reporting their spread is itself the scientific result:

  - per-config and per-scenario agree -> scenario-universal,
    joint training is valid.
  - per-config disagrees sharply      -> v2 ground state is
    scenario-specific, joint is too hard; thesis finding is that
    one Hamiltonian parameterisation cannot cover all MHD regimes.

Sanity checks printed at the end:
  - (c*, thr*) not hitting optimisation bounds
  - F1_val > classical baseline (delta > 0)
  - spread across training modes

w_zz = 2.0 and w_zzzz = 1.0 are fixed by physical reasoning (v2 design
choice) and are NOT trained.

Parameter vector (theta):
  theta[0] = log10(c_bias)   in [-1.0 ,  2.0 ]   -> c_bias in [0.1, 100]
  theta[1] = thr_amr         in [ 0.02,  0.60]

Input:  results/dns_{scenario}_Re{Re}_N{N}.npz
        results/patches_{scenario}_Re{Re}_N{N}_dim{D}.npz
Output: results/train_{tag}_N{N}_dim{D}.npz  (one per run)
        results/train_COMPARE_N{N}_dim{D}.npz (cross-mode summary)

Usage:
  python study/phase10_train_hamiltonian.py --dim 4
  python study/phase10_train_hamiltonian.py --dim 4 --modes per-scenario joint
  python study/phase10_train_hamiltonian.py --dim 4 --modes per-config
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
from config import RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N

from exact_diagonalisation import build_patch_hamiltonian
from ising_terms_and_annealing import (
    build_ising_terms, sa_multi_restart, spins_to_decisions, _metrics,
)

try:
    import cma           # noqa: F401
    HAS_CMA = True
except Exception:
    HAS_CMA = False

from scipy.optimize import minimize


# -------------------------------------------------------------------
# Parameter encoding
# -------------------------------------------------------------------

THETA_BOUNDS = np.array([
    [-1.0,  2.0],    # log10(c_bias)     -> [0.1, 100]
    [ 0.02, 0.60],   # thr_amr
])
THETA_DIM = len(THETA_BOUNDS)
THETA_INIT = np.array([np.log10(1.0),  # c_bias = 1.0
                       0.15])          # thr_amr = 0.15


def decode_theta(theta):
    theta = np.clip(theta, THETA_BOUNDS[:, 0], THETA_BOUNDS[:, 1])
    return float(10 ** theta[0]), float(theta[1])


def hits_bound(theta, rel_tol=0.02):
    """True if theta is within 2% of a bound (flags unreliable optimum)."""
    lb, ub = THETA_BOUNDS[:, 0], THETA_BOUNDS[:, 1]
    span = ub - lb
    return bool(np.any((theta - lb) / span < rel_tol) or
                np.any((ub - theta) / span < rel_tol))


# -------------------------------------------------------------------
# Snapshot loader
# -------------------------------------------------------------------

def load_snapshots(dns_path, patches_path):
    dns = np.load(dns_path)
    patches = np.load(patches_path)
    return dict(
        vx=dns["vx"].astype(np.float64),
        vy=dns["vy"].astype(np.float64),
        Bx=dns["Bx"].astype(np.float64),
        By=dns["By"].astype(np.float64),
        l2_errors=patches["l2_errors"],
        l2_threshold=float(patches["l2_threshold"]),
        N=int(dns["vx"].shape[1]),
        Re=int(dns.get("meta_Re", 800)),
        scenario=str(dns.get("meta_scenario", "unknown")),
    )


def f1_for_theta(theta, snaps, dim, snap_indices, sweeps, n_restarts, rng):
    c_bias, thr = decode_theta(theta)
    f1s, cfs = [], []
    for si in snap_indices:
        hp, score_vqa, _ = build_patch_hamiltonian(
            snaps["vx"][si], snaps["vy"][si],
            snaps["Bx"][si], snaps["By"][si],
            snaps["N"], dim, snaps["Re"],
            threshold_amr=thr, use_v2=True, c_bias=c_bias,
        )
        h_bias, edges, plaqs = build_ising_terms(hp, dim)
        n_q = 2 * dim * dim
        best_spins, _E, _ = sa_multi_restart(
            h_bias, edges, plaqs, n_q,
            sweeps=sweeps, n_restarts=n_restarts, rng=rng,
        )
        dec_h, dec_v = spins_to_decisions(best_spins, dim)
        sa_refine    = dec_h | dec_v
        gt_refine    = snaps["l2_errors"][si] >= snaps["l2_threshold"]
        class_refine = score_vqa > thr
        f1s.append(_metrics(sa_refine,    gt_refine)["f1"])
        cfs.append(_metrics(class_refine, gt_refine)["f1"])
    return float(np.mean(f1s)), float(np.mean(cfs))


# -------------------------------------------------------------------
# Core training loop (shared by all modes)
# -------------------------------------------------------------------

def train(snaps_list, dim, *, n_iters, sweeps, n_restarts,
          train_frac, seed, optimiser_name, max_batch, max_val,
          tag="", theta_init=None):
    """
    Optimise theta against mean F1 on a pooled train split.

    Key design choices (lessons from the first Result_phase10.txt run):
      - Val set is FROZEN (val_fixed) so F1_val is comparable across steps.
      - Train batches are resampled each call (noise is a feature of SGD).
      - Initial Nelder-Mead simplex spans a meaningful fraction of the
        search space (large perturbations in log10(c_bias) and thr_amr),
        otherwise NM collapses at x0 on noisy objectives.
      - CMA-ES uses sigma0=0.5 and popsize>=6 for noise tolerance.
      - Final best is chosen by RE-EVALUATING the top-K visited thetas
        on val_fixed, not by a single noisy mid-training val sample.
    """
    rng = np.random.default_rng(seed)

    pairs = [(ci, si) for ci, s in enumerate(snaps_list)
             for si in range(len(s["vx"]))]
    pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    n_tr = max(1, int(train_frac * len(pairs)))
    train_pairs = pairs[:n_tr]
    val_pairs   = pairs[n_tr:] if n_tr < len(pairs) else pairs[-1:]
    # deterministic, fixed val subset (same every time we eval val)
    val_fixed = val_pairs[:max_val]

    print(f"  [{tag}] dataset: {len(pairs)} pairs "
          f"-> train {len(train_pairs)}, val {len(val_pairs)} "
          f"(val_fixed={len(val_fixed)})")

    def _eval_on(theta, use_pairs, eval_rng):
        """Mean F1 (SA vs GT) + classical over a fixed pair list."""
        by_cfg = {}
        for ci, si in use_pairs:
            by_cfg.setdefault(ci, []).append(si)
        f1_all, cf_all = [], []
        for ci, sis in by_cfg.items():
            f1, cf = f1_for_theta(theta, snaps_list[ci], dim,
                                  sis, sweeps, n_restarts, eval_rng)
            f1_all.append(f1); cf_all.append(cf)
        return float(np.mean(f1_all)), float(np.mean(cf_all))

    def _train_batch(theta, eval_rng):
        if len(train_pairs) > max_batch:
            idx = eval_rng.choice(len(train_pairs), size=max_batch,
                                  replace=False)
            sub = [train_pairs[i] for i in idx]
        else:
            sub = train_pairs
        return _eval_on(theta, sub, eval_rng)

    history_theta, history_f1_tr, history_f1_val = [], [], []
    eval_rng = np.random.default_rng(seed + 1)

    def objective(theta_raw):
        t0 = time.time()
        f1_tr, _ = _train_batch(theta_raw, eval_rng)
        dt = time.time() - t0
        history_theta.append(np.array(theta_raw, copy=True))
        history_f1_tr.append(f1_tr)
        # val on frozen set, only every 5 steps (expensive)
        if len(history_theta) % 5 == 1:
            f1_v, _ = _eval_on(theta_raw, val_fixed, eval_rng)
        else:
            f1_v = history_f1_val[-1] if history_f1_val else f1_tr
        history_f1_val.append(f1_v)
        c, thr = decode_theta(theta_raw)
        print(f"    [{tag}] step {len(history_theta):>3}  "
              f"c={c:>7.3f}  thr={thr:.3f}  "
              f"F1_tr={f1_tr:.3f}  F1_val={f1_v:.3f}   [{dt:.1f}s]")
        return -f1_tr

    # ------------------------------------------------------------
    # Run optimiser
    # ------------------------------------------------------------
    x0 = (THETA_INIT.copy() if theta_init is None
          else np.clip(np.asarray(theta_init, dtype=float),
                       THETA_BOUNDS[:, 0], THETA_BOUNDS[:, 1]))
    c0_, thr0_ = decode_theta(x0)
    print(f"  [{tag}] x0: c_bias={c0_:.3f}  thr={thr0_:.3f}  "
          f"(source: {'analytical' if theta_init is not None else 'default'})")
    lb, ub = THETA_BOUNDS[:, 0], THETA_BOUNDS[:, 1]
    use_cma = (optimiser_name == "cma") and HAS_CMA
    if optimiser_name == "cma" and not HAS_CMA:
        print(f"  [{tag}][warn] cma not installed -> Nelder-Mead")

    if use_cma:
        # popsize>=6 and sigma0=0.5 give robust exploration on noisy F1
        print(f"  [{tag}] optimiser: CMA-ES, budget {n_iters}, "
              f"sigma0=0.5, popsize=6")
        es = cma.CMAEvolutionStrategy(
            x0, 0.5,
            {'bounds': [list(lb), list(ub)],
             'maxfevals': n_iters,
             'popsize': 6,
             'tolx': 1e-3,
             'tolfun': 1e-3,
             'verbose': -9,
             'seed': int(seed) + 2})
        while not es.stop() and len(history_theta) < n_iters:
            xs = es.ask()
            ys = [objective(x) for x in xs]
            es.tell(xs, ys)
    else:
        # Large initial simplex — otherwise NM collapses at x0 on noise.
        # Perturb log10(c_bias) by 1.0 (factor 10) and thr by 0.15.
        init_simplex = np.array([
            x0,
            x0 + np.array([ 1.0, 0.0 ]),   # c_bias = 10
            x0 + np.array([ 0.0, 0.15]),   # thr = 0.30
        ])
        # clip into bounds
        init_simplex = np.clip(init_simplex, lb, ub)
        print(f"  [{tag}] optimiser: Nelder-Mead (adaptive), "
              f"budget {n_iters}, wide simplex")
        minimize(objective, x0, method="Nelder-Mead",
                 options=dict(maxfev=n_iters,
                              xatol=1e-3, fatol=1e-3,
                              adaptive=True,
                              initial_simplex=init_simplex))

    # ------------------------------------------------------------
    # Robust best selection: reevaluate top-K thetas on val_fixed
    # using a FRESH rng, pick the highest mean F1.
    # ------------------------------------------------------------
    K = min(5, len(history_theta))
    # rank by training F1 (noisy) to pick candidates
    order = np.argsort(history_f1_tr)[::-1]
    # dedupe near-identical thetas (within 1e-3)
    uniq_idx = []
    seen = []
    for i in order:
        t = history_theta[i]
        if not any(np.linalg.norm(t - s) < 1e-3 for s in seen):
            uniq_idx.append(i); seen.append(t)
        if len(uniq_idx) >= K:
            break

    final_rng = np.random.default_rng(seed + 42)
    final_scores = []
    print(f"\n  [{tag}] re-evaluating top-{len(uniq_idx)} candidates "
          f"on val_fixed ({len(val_fixed)} pairs)...")
    for rank, idx in enumerate(uniq_idx, 1):
        th = history_theta[idx]
        f1v, cf = _eval_on(th, val_fixed, final_rng)
        c_, thr_ = decode_theta(th)
        final_scores.append((f1v, cf, th))
        print(f"    cand {rank}: c={c_:>7.3f} thr={thr_:.3f}  "
              f"val F1={f1v:.3f}  class={cf:.3f}")

    best_f1v, best_cf, best_theta = max(final_scores, key=lambda t: t[0])
    c, thr = decode_theta(best_theta)
    bnd = hits_bound(best_theta)
    print(f"\n  [{tag}] BEST: c_bias={c:.3f} thr={thr:.3f}  "
          f"F1_val={best_f1v:.3f}  classical={best_cf:.3f}  "
          f"delta={best_f1v - best_cf:+.3f}  "
          f"{'BOUND!' if bnd else 'OK'}")

    return dict(
        tag=tag,
        theta_history=np.array(history_theta),
        f1_train_history=np.array(history_f1_tr),
        f1_val_history=np.array(history_f1_val),
        best_theta=best_theta,
        best_c_bias=c, best_thr=thr,
        best_f1_val=best_f1v,
        classical_f1=best_cf,
        delta=best_f1v - best_cf,
        hits_bound=bnd,
        train_pairs=np.array(train_pairs, dtype=int),
        val_pairs=np.array(val_pairs, dtype=int),
        val_fixed=np.array(val_fixed, dtype=int),
        optimiser="cma" if use_cma else "nelder-mead",
        w_zz=2.0, w_zzzz=1.0,
    )


# -------------------------------------------------------------------
# Cross-mode comparison
# -------------------------------------------------------------------

def print_comparison(results):
    """Show how (c*, thr*) varies across training modes."""
    print("\n" + "=" * 88)
    print("  CROSS-MODE COMPARISON  (the scientific payload)")
    print("=" * 88)
    print(f"  {'tag':<28} {'c_bias*':>9} {'thr*':>7} "
          f"{'F1_val':>7} {'class':>6} {'delta':>7} {'bnd':>4}")
    for r in results:
        print(f"  {r['tag']:<28} {r['best_c_bias']:>9.3f} "
              f"{r['best_thr']:>7.3f} {r['best_f1_val']:>7.3f} "
              f"{r['classical_f1']:>6.3f} {r['delta']:>+7.3f} "
              f"{'YES' if r['hits_bound'] else '':>4}")

    # spread of optima
    cs  = np.array([r['best_c_bias'] for r in results])
    ths = np.array([r['best_thr']    for r in results])
    print(f"\n  c_bias*  range: [{cs.min():.3f}, {cs.max():.3f}]  "
          f"ratio max/min = {cs.max()/max(cs.min(), 1e-6):.2f}x")
    print(f"  thr*     range: [{ths.min():.3f}, {ths.max():.3f}]  "
          f"ratio max/min = {ths.max()/max(ths.min(), 1e-6):.2f}x")

    # verdict
    c_ratio  = cs.max() / max(cs.min(), 1e-6)
    th_ratio = ths.max() / max(ths.min(), 1e-6)
    joint = [r for r in results if r['tag'] == 'joint']
    per_scene = [r for r in results if r['tag'].startswith('scenario:')]

    print("\n  Interpretation:")
    if c_ratio < 2.0 and th_ratio < 1.5:
        print("    * (c*, thr*) ~ scenario-universal: joint training is valid.")
    else:
        print("    * (c*, thr*) VARIES across scenarios "
              f"(c spread {c_ratio:.1f}x, thr spread {th_ratio:.1f}x)")
        print("      -> v2 Hamiltonian is not scenario-universal; "
              "this is itself a thesis finding.")
    if joint and per_scene:
        mean_per = np.mean([r['best_f1_val'] for r in per_scene])
        print(f"    * mean F1_val per-scenario = {mean_per:.3f}  vs  "
              f"joint = {joint[0]['best_f1_val']:.3f}")
        if joint[0]['best_f1_val'] < mean_per - 0.02:
            print("      -> joint sacrifices F1; scenario-specific "
                  "(c, thr) would be needed in practice.")
    any_bnd = any(r['hits_bound'] for r in results)
    if any_bnd:
        print("    * WARNING: some optima hit the search bounds; "
              "the true optimum may lie outside. Re-run with wider bounds.")
    else:
        print("    * All optima are interior -> bounds are wide enough.")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Phase 10: train v2 (c_bias, thr_amr) in 3 modes")
    p.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--dim", type=int, default=4)
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--modes", nargs="+",
                   choices=["per-config", "per-scenario", "joint"],
                   default=["per-scenario", "joint"],
                   help="training modes to run; default: per-scenario + joint")
    p.add_argument("--n-iters", type=int, default=80,
                   help="max objective evaluations per training run")
    p.add_argument("--sweeps", type=int, default=600)
    p.add_argument("--n-restarts", type=int, default=2)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--max-batch", type=int, default=16,
                   help="train snaps sampled per eval (↑ = less noise)")
    p.add_argument("--max-val", type=int, default=24,
                   help="snaps in the frozen val_fixed set")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--optimiser", choices=["cma", "nelder-mead"],
                   default="cma")
    p.add_argument("--analytical-init", default="auto",
                   help="path to phase 10a analytical_*.npz; 'auto' looks "
                        "for results/analytical_N{N}_dim{D}.npz; 'none' "
                        "disables it.")
    args = p.parse_args()

    print("=" * 88)
    print("  Phase 10: training of v2 (c_bias, thr_amr)")
    print(f"  theta = (log10 c_bias, thr_amr) ; w_zz=2, w_zzzz=1 (fixed)")
    print(f"  modes: {args.modes}")
    print(f"  optimiser={args.optimiser}  budget/run={args.n_iters}")
    print("=" * 88)
    print()

    # collect available configs
    configs = []
    for sc in args.scenario:
        for re in args.re:
            dns_path = os.path.join(
                RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            patches_path = os.path.join(
                RESULTS_DIR,
                f"patches_{sc}_Re{re}_N{args.N}_dim{args.dim}.npz")
            if os.path.exists(dns_path) and os.path.exists(patches_path):
                configs.append((sc, re, dns_path, patches_path))
            else:
                print(f"  SKIP {sc} Re={re}: missing input")

    if not configs:
        print("no input found.")
        return

    # load once
    snaps_all = [load_snapshots(dp, pp) for _, _, dp, pp in configs]
    by_scene = {}
    for (sc, re, _, _), s in zip(configs, snaps_all):
        by_scene.setdefault(sc, []).append(s)

    # ------------------------------------------------------------
    # Optional analytical init (phase 10a). Maps tag -> theta_init.
    # ------------------------------------------------------------
    init_map = {}
    if args.analytical_init != "none":
        if args.analytical_init == "auto":
            ana_path = os.path.join(
                RESULTS_DIR,
                f"analytical_N{args.N}_dim{args.dim}.npz")
        else:
            ana_path = args.analytical_init
        if os.path.exists(ana_path):
            ana = np.load(ana_path, allow_pickle=False)
            tags = ana["tags"]; thr_s = ana["thr_star"]
            c_s = ana["c_bias_star"]
            for t, th, cb in zip(tags, thr_s, c_s):
                init_map[str(t)] = np.array(
                    [np.log10(max(float(cb), 0.1)), float(th)])
            print(f"  analytical init loaded from "
                  f"{os.path.basename(ana_path)}: {len(init_map)} entries\n")
        else:
            print(f"  no analytical init at {ana_path} -> "
                  f"using default x0\n")

    def _run(snaps_list, tag):
        t0 = time.time()
        theta_init = init_map.get(tag)
        res = train(
            snaps_list, args.dim,
            n_iters=args.n_iters, sweeps=args.sweeps,
            n_restarts=args.n_restarts, train_frac=args.train_frac,
            seed=args.seed, optimiser_name=args.optimiser,
            max_batch=args.max_batch, max_val=args.max_val, tag=tag,
            theta_init=theta_init,
        )
        print(f"  [{tag}] wall-time: {time.time()-t0:.0f}s\n")
        fname = f"train_{tag.replace(':', '-').replace(' ', '_')}_" \
                f"N{args.N}_dim{args.dim}.npz"
        np.savez_compressed(
            os.path.join(RESULTS_DIR, fname),
            **{k: v for k, v in res.items() if not isinstance(v, str)},
            tag_str=tag)
        print(f"  saved: {fname}")
        return res

    all_results = []

    if "per-config" in args.modes:
        print("\n### MODE: per-config ###")
        for (sc, re, _, _), s in zip(configs, snaps_all):
            tag = f"cfg:{sc}_Re{re}"
            all_results.append(_run([s], tag))

    if "per-scenario" in args.modes:
        print("\n### MODE: per-scenario ###")
        for sc, ss in by_scene.items():
            tag = f"scenario:{sc}"
            all_results.append(_run(ss, tag))

    if "joint" in args.modes:
        print("\n### MODE: joint ###")
        all_results.append(_run(snaps_all, "joint"))

    # cross-mode comparison
    print_comparison(all_results)

    # dump compare file
    compare_path = os.path.join(
        RESULTS_DIR, f"train_COMPARE_N{args.N}_dim{args.dim}.npz")
    np.savez_compressed(
        compare_path,
        tags=np.array([r["tag"] for r in all_results]),
        c_bias=np.array([r["best_c_bias"] for r in all_results]),
        thr=np.array([r["best_thr"] for r in all_results]),
        f1_val=np.array([r["best_f1_val"] for r in all_results]),
        classical_f1=np.array([r["classical_f1"] for r in all_results]),
        delta=np.array([r["delta"] for r in all_results]),
        hits_bound=np.array([r["hits_bound"] for r in all_results]),
    )
    print(f"\n  saved compare: {os.path.basename(compare_path)}")
    print("\nPhase 10 complete.")


if __name__ == "__main__":
    main()
