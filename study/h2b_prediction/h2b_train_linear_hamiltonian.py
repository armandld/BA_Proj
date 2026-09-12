#!/usr/bin/env python3
"""Phase 10: supervised rescue fit of the fixed-form V2 Hamiltonian.

``c_bias`` and the classical-score threshold are fitted against the L2-hard
mask. Each DNS trajectory is split chronologically into training, model
selection and final test blocks. The test block is evaluated once after
selection. Per-configuration, per-scenario and joint fits expose how much
specialisation the fixed Hamiltonian form requires.

This is a post-hoc rescue diagnostic. Phases 3--8 intentionally evaluate the
a-priori V2 constants and do not consume these fitted parameters.

Usage:
  python study/h2b_prediction/h2b_train_linear_hamiltonian.py --dim 4
"""
import argparse
import importlib.util
import json
import os
import sys
import time
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

HAS_CMA = importlib.util.find_spec("cma") is not None

from scipy.optimize import minimize
import provenance


# -------------------------------------------------------------------
# Parameter encoding
# -------------------------------------------------------------------

THETA_BOUNDS = np.array([
    [-1.0, 5.0],     # log10(c_bias) -> [0.1, 100000]
    [0.0, 1.0],      # score threshold
])
THETA_DIM = len(THETA_BOUNDS)
THETA_INIT = np.array([np.log10(1.0),  # c_bias = 1.0
                       0.15])          # thr_amr = 0.15


def decode_theta(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (THETA_DIM,) or not np.all(np.isfinite(theta)):
        raise ValueError(f"theta must be a finite vector of shape ({THETA_DIM},)")
    theta = np.clip(theta, THETA_BOUNDS[:, 0], THETA_BOUNDS[:, 1])
    return float(10 ** theta[0]), float(theta[1])


def hits_bound(theta, rel_tol=0.02):
    """True if theta is within 2% of a bound (flags unreliable optimum)."""
    theta = np.asarray(theta, dtype=float)
    if (theta.shape != (THETA_DIM,) or not np.all(np.isfinite(theta))
            or not np.isfinite(rel_tol) or rel_tol < 0.0):
        raise ValueError("theta and rel_tol must be finite and valid")
    lb, ub = THETA_BOUNDS[:, 0], THETA_BOUNDS[:, 1]
    span = ub - lb
    return bool(np.any((theta - lb) / span < rel_tol) or
                np.any((ub - theta) / span < rel_tol))


def chronological_split_indices(n_snapshots, train_frac=0.6, val_frac=0.2):
    """Split one trajectory into ordered train, validation and test blocks."""
    if not isinstance(n_snapshots, (int, np.integer)) or n_snapshots < 3:
        raise ValueError("each trajectory needs at least three snapshots")
    if not (np.isfinite(train_frac) and np.isfinite(val_frac)
            and 0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0
            and train_frac + val_frac < 1.0):
        raise ValueError("train_frac and val_frac must be positive and sum to < 1")
    n_train = min(max(1, int(np.floor(train_frac * n_snapshots))),
                  n_snapshots - 2)
    n_val = min(max(1, int(np.floor(val_frac * n_snapshots))),
                n_snapshots - n_train - 1)
    cut = n_train + n_val
    return (np.arange(0, n_train, dtype=int),
            np.arange(n_train, cut, dtype=int),
            np.arange(cut, n_snapshots, dtype=int))


def evenly_subsample(indices, limit):
    """Return at most ``limit`` ordered indices spanning the input."""
    indices = np.asarray(indices, dtype=int).ravel()
    if not isinstance(limit, (int, np.integer)) or limit < 1:
        raise ValueError("subsample limit must be a positive integer")
    if indices.size <= limit:
        return indices.copy()
    positions = np.linspace(0, indices.size - 1, int(limit), dtype=int)
    return indices[positions]


def split_snapshot_pairs(snaps_list, train_frac=0.6, val_frac=0.2):
    """Build aligned chronological split pairs for multiple trajectories."""
    train_pairs, val_pairs, test_pairs = [], [], []
    for config_index, snaps in enumerate(snaps_list):
        train_idx, val_idx, test_idx = chronological_split_indices(
            len(snaps["vx"]), train_frac, val_frac)
        train_pairs.extend((config_index, int(i)) for i in train_idx)
        val_pairs.extend((config_index, int(i)) for i in val_idx)
        test_pairs.extend((config_index, int(i)) for i in test_idx)
    return train_pairs, val_pairs, test_pairs


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


def _subsample_pairs_by_config(pairs, max_total):
    """Cap a split while retaining every represented configuration."""
    groups = {}
    for config_index, snapshot_index in pairs:
        groups.setdefault(config_index, []).append(snapshot_index)
    if len(pairs) <= max_total:
        return list(pairs)
    if max_total < len(groups):
        raise ValueError(
            "split cap must retain at least one snapshot per configuration")
    base, remainder = divmod(max_total, len(groups))
    selected = []
    for rank, (config_index, indices) in enumerate(sorted(groups.items())):
        limit = base + (rank < remainder)
        selected.extend(
            (config_index, int(i))
            for i in evenly_subsample(indices, limit)
        )
    return selected


def _sample_train_pairs(pairs, max_total, rng):
    """Draw a stratified training batch with all configurations represented."""
    groups = {}
    for config_index, snapshot_index in pairs:
        groups.setdefault(config_index, []).append(snapshot_index)
    if len(pairs) <= max_total:
        return list(pairs)
    if max_total < len(groups):
        raise ValueError(
            "max_batch must be at least the number of configurations")
    base, remainder = divmod(max_total, len(groups))
    selected = []
    for rank, (config_index, indices) in enumerate(sorted(groups.items())):
        limit = min(len(indices), base + (rank < remainder))
        picks = rng.choice(indices, size=limit, replace=False)
        selected.extend((config_index, int(i)) for i in picks)
    return selected


def _nelder_mead_simplex(x0):
    """Construct a non-degenerate bounded simplex around ``x0``."""
    x0 = np.asarray(x0, dtype=float)
    lb, ub = THETA_BOUNDS[:, 0], THETA_BOUNDS[:, 1]
    steps = np.array([1.0, 0.15])
    vertices = [x0.copy()]
    for axis, step in enumerate(steps):
        vertex = x0.copy()
        if x0[axis] + step <= ub[axis]:
            vertex[axis] += step
        elif x0[axis] - step >= lb[axis]:
            vertex[axis] -= step
        else:
            vertex[axis] = lb[axis] if x0[axis] != lb[axis] else ub[axis]
        vertices.append(vertex)
    return np.asarray(vertices)


def _load_cma():
    """Load the requested optimizer or fail before any objective evaluation."""
    if not HAS_CMA:
        raise RuntimeError(
            "CMA-ES was requested but package 'cma' is not installed")
    try:
        import cma
    except Exception as exc:
        raise RuntimeError("CMA-ES could not be imported") from exc
    return cma

def train(snaps_list, dim, *, n_iters, sweeps, n_restarts,
          train_frac, val_frac, seed, optimiser_name, max_batch, max_val,
          max_test,
          tag="", theta_init=None):
    """Fit on train, select on validation, then evaluate once on test."""
    if not snaps_list:
        raise ValueError("snaps_list must not be empty")
    if n_iters < 3 or sweeps < 1 or n_restarts < 1:
        raise ValueError("n_iters, sweeps and n_restarts must be positive")
    if max_batch < 1 or max_val < 1 or max_test < 1:
        raise ValueError("batch and split caps must be positive")
    if optimiser_name not in {"cma", "nelder-mead"}:
        raise ValueError("unknown optimizer")
    if optimiser_name == "cma" and n_iters < 4:
        raise ValueError("CMA-ES requires at least four objective evaluations")

    rng = np.random.default_rng(seed)
    train_pairs, val_pairs, test_pairs = split_snapshot_pairs(
        snaps_list, train_frac, val_frac)
    val_fixed = _subsample_pairs_by_config(val_pairs, max_val)
    test_fixed = _subsample_pairs_by_config(test_pairs, max_test)

    print(f"  [{tag}] chronological split: train={len(train_pairs)}, "
          f"validation={len(val_pairs)}, test={len(test_pairs)} "
          f"(evaluated: validation={len(val_fixed)}, test={len(test_fixed)})")

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
        sub = _sample_train_pairs(train_pairs, max_batch, eval_rng)
        return _eval_on(theta, sub, eval_rng)

    history_theta, history_f1_tr, history_f1_val = [], [], []
    eval_rng = np.random.default_rng(seed + 1)

    def objective(theta_raw):
        t0 = time.time()
        theta = np.clip(np.asarray(theta_raw, dtype=float),
                        THETA_BOUNDS[:, 0], THETA_BOUNDS[:, 1])
        f1_tr, _ = _train_batch(theta, eval_rng)
        dt = time.time() - t0
        history_theta.append(theta.copy())
        history_f1_tr.append(f1_tr)
        if len(history_theta) % 5 == 1:
            val_rng = np.random.default_rng(seed + 1000 + len(history_theta))
            f1_v, _ = _eval_on(theta, val_fixed, val_rng)
        else:
            f1_v = history_f1_val[-1] if history_f1_val else f1_tr
        history_f1_val.append(f1_v)
        c, thr = decode_theta(theta)
        print(f"    [{tag}] step {len(history_theta):>3}  "
              f"c={c:>7.3f}  thr={thr:.3f}  "
              f"F1_tr={f1_tr:.3f}  F1_val={f1_v:.3f}   [{dt:.1f}s]")
        return -f1_tr

    x0 = (THETA_INIT.copy() if theta_init is None
          else np.clip(np.asarray(theta_init, dtype=float),
                       THETA_BOUNDS[:, 0], THETA_BOUNDS[:, 1]))
    c0_, thr0_ = decode_theta(x0)
    x0_bnd = hits_bound(x0)
    print(f"  [{tag}] x0: c_bias={c0_:.3f}  thr={thr0_:.3f}  "
          f"(source: {'analytical' if theta_init is not None else 'default'})"
          f"{'  [BOUND]' if x0_bnd else ''}")
    lb, ub = THETA_BOUNDS[:, 0], THETA_BOUNDS[:, 1]
    if optimiser_name == "cma":
        cma = _load_cma()
        popsize = min(6, n_iters)
        print(f"  [{tag}] optimiser: CMA-ES, budget {n_iters}, "
              f"sigma0=0.5, popsize={popsize}")
        es = cma.CMAEvolutionStrategy(
            x0, 0.5,
            {'bounds': [list(lb), list(ub)],
             'maxfevals': n_iters,
             'popsize': popsize,
             'tolx': 1e-3,
             'tolfun': 1e-3,
             'verbose': -9,
             'seed': int(seed) + 2})
        while (not es.stop()
               and len(history_theta) + popsize <= n_iters):
            xs = es.ask()
            ys = [objective(x) for x in xs]
            es.tell(xs, ys)
    else:
        init_simplex = _nelder_mead_simplex(x0)
        print(f"  [{tag}] optimiser: Nelder-Mead (adaptive), "
              f"budget {n_iters}")
        minimize(objective, x0, method="Nelder-Mead",
                 options=dict(maxfev=n_iters,
                              xatol=1e-3, fatol=1e-3,
                              adaptive=True,
                              initial_simplex=init_simplex))

    if not history_theta:
        raise RuntimeError("optimizer produced no objective evaluation")
    K = min(5, len(history_theta))
    order = np.argsort(history_f1_tr)[::-1]
    uniq_idx = []
    seen = []
    for i in order:
        t = history_theta[i]
        if not any(np.linalg.norm(t - s) < 1e-3 for s in seen):
            uniq_idx.append(i); seen.append(t)
        if len(uniq_idx) >= K:
            break

    final_scores = []
    print(f"\n  [{tag}] re-evaluating top-{len(uniq_idx)} candidates "
          f"on val_fixed ({len(val_fixed)} pairs)...")
    for rank, idx in enumerate(uniq_idx, 1):
        th = history_theta[idx]
        # Reset to the same stream for every candidate (common random
        # numbers), so SA noise cannot favour one candidate by evaluation
        # order.
        candidate_rng = np.random.default_rng(seed + 10_000)
        f1v, cf = _eval_on(th, val_fixed, candidate_rng)
        c_, thr_ = decode_theta(th)
        final_scores.append((f1v, cf, th))
        print(f"    cand {rank}: c={c_:>7.3f} thr={thr_:.3f}  "
              f"val F1={f1v:.3f}  class={cf:.3f}")

    best_f1v, best_cf_val, best_theta = max(
        final_scores, key=lambda t: t[0])
    test_rng = np.random.default_rng(seed + 20_000)
    best_f1_test, best_cf_test = _eval_on(
        best_theta, test_fixed, test_rng)
    c, thr = decode_theta(best_theta)
    bnd = hits_bound(best_theta)
    print(f"\n  [{tag}] BEST: c_bias={c:.3f} thr={thr:.3f}  "
          f"F1_val={best_f1v:.3f}; "
          f"F1_test={best_f1_test:.3f}  classical_test={best_cf_test:.3f}  "
          f"delta_test={best_f1_test - best_cf_test:+.3f}  "
          f"{'BOUND!' if bnd else 'OK'}")

    return dict(
        tag=tag,
        theta_history=np.array(history_theta),
        f1_train_history=np.array(history_f1_tr),
        f1_val_history=np.array(history_f1_val),
        best_theta=best_theta,
        best_c_bias=c, best_thr=thr,
        best_f1_val=best_f1v,
        classical_f1_val=best_cf_val,
        delta_val=best_f1v - best_cf_val,
        best_f1_test=best_f1_test,
        classical_f1=best_cf_test,
        classical_f1_test=best_cf_test,
        delta=best_f1_test - best_cf_test,
        delta_test=best_f1_test - best_cf_test,
        hits_bound=bnd,
        x0_theta=np.asarray(x0, dtype=float),
        x0_hits_bound=x0_bnd,
        x0_from_analytical=bool(theta_init is not None),
        train_pairs=np.array(train_pairs, dtype=int),
        val_pairs=np.array(val_pairs, dtype=int),
        val_fixed=np.array(val_fixed, dtype=int),
        test_pairs=np.array(test_pairs, dtype=int),
        test_fixed=np.array(test_fixed, dtype=int),
        split_strategy="chronological_per_configuration",
        train_fraction=float(train_frac),
        validation_fraction=float(val_frac),
        optimiser=optimiser_name,
        w_zz=2.0, w_zzzz=1.0,
    )


# -------------------------------------------------------------------
# Cross-mode comparison
# -------------------------------------------------------------------

def print_comparison(results):
    """Compare fitted parameters and untouched test performance."""
    print("\n" + "=" * 88)
    print("  CROSS-MODE COMPARISON  (the scientific payload)")
    print("=" * 88)
    print(f"  {'tag':<28} {'c_bias*':>9} {'thr*':>7} "
          f"{'F1_test':>7} {'class':>6} {'delta':>7} {'bnd':>4}")
    for r in results:
        print(f"  {r['tag']:<28} {r['best_c_bias']:>9.3f} "
              f"{r['best_thr']:>7.3f} {r['best_f1_test']:>7.3f} "
              f"{r['classical_f1']:>6.3f} {r['delta']:>+7.3f} "
              f"{'YES' if r['hits_bound'] else '':>4}")

    cs  = np.array([r['best_c_bias'] for r in results])
    ths = np.array([r['best_thr']    for r in results])
    print(f"\n  c_bias*  range: [{cs.min():.3f}, {cs.max():.3f}]  "
          f"ratio max/min = {cs.max()/max(cs.min(), 1e-6):.2f}x")
    print(f"  thr*     range: [{ths.min():.3f}, {ths.max():.3f}]  "
          f"ratio max/min = {ths.max()/max(ths.min(), 1e-6):.2f}x")

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
        mean_per = np.mean([r['best_f1_test'] for r in per_scene])
        print(f"    * mean F1_test per-scenario = {mean_per:.3f}  vs  "
              f"joint = {joint[0]['best_f1_test']:.3f}")
        if joint[0]['best_f1_test'] < mean_per - 0.02:
            print("      -> joint sacrifices F1; scenario-specific "
                  "(c, thr) would be needed in practice.")
    any_bnd = any(r['hits_bound'] for r in results)
    if any_bnd:
        print("    * WARNING: some optima hit the search bounds; "
              "the true optimum may lie outside. Re-run with wider bounds.")
    else:
        print("    * All optima are interior -> bounds are wide enough.")


# -------------------------------------------------------------------
# Analytical init (phase 10a)
# -------------------------------------------------------------------

def build_init_map(tags, thr_star, c_bias_star, degenerate):
    """Build analytical initial points, omitting uninformative rows."""
    init_map, n_skipped = {}, 0
    for t, th, cb, dg in zip(tags, thr_star, c_bias_star, degenerate):
        if bool(dg) or not (np.isfinite(cb) and np.isfinite(th)):
            n_skipped += 1
            continue
        theta_raw = np.array([np.log10(max(float(cb), 0.1)), float(th)])
        theta_x0 = np.clip(theta_raw, THETA_BOUNDS[:, 0], THETA_BOUNDS[:, 1])
        if not np.allclose(theta_raw, theta_x0):
            print(f"  [{t}] analytical init clipped to optimizer bounds: "
                  f"(log10 c, thr) {theta_raw} -> {theta_x0}")
        if hits_bound(theta_x0):
            print(f"  [{t}] analytical init lies on an optimizer bound")
        init_map[str(t)] = theta_x0
    return init_map, n_skipped


def build_init_map_from_artifact(artifact, train_frac=0.6, val_frac=0.2):
    """Validate a phase-10a artifact before using it as optimizer input."""
    required = {
        "tags", "thr_star", "c_bias_star", "degenerate", "theta_bounds",
        "at_left_edge", "at_right_edge", "bias_only_limit",
        "split_strategy", "train_fraction", "validation_fraction",
    }
    available = set(artifact.files) if hasattr(artifact, "files") else set(artifact)
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(
            "analytical initialization artifact is obsolete; missing "
            f"{missing}. Re-run phase 10a before phase 10")
    bounds = np.asarray(artifact["theta_bounds"], dtype=float)
    if bounds.shape != THETA_BOUNDS.shape or not np.allclose(
            bounds, THETA_BOUNDS, rtol=0.0, atol=0.0):
        raise RuntimeError(
            "analytical initialization uses different search bounds; "
            "re-run phase 10a with the current code")
    if (str(np.asarray(artifact["split_strategy"]).item())
            != "chronological_per_configuration"
            or not np.isclose(float(artifact["train_fraction"]), train_frac)
            or not np.isclose(float(artifact["validation_fraction"]), val_frac)):
        raise RuntimeError(
            "analytical initialization uses a different temporal split; "
            "re-run phase 10a with matching --train-frac/--val-frac")
    degenerate = np.asarray(artifact["degenerate"], dtype=bool)
    left = np.asarray(artifact["at_left_edge"], dtype=bool)
    right = np.asarray(artifact["at_right_edge"], dtype=bool)
    bias_only = np.asarray(artifact["bias_only_limit"], dtype=bool)
    unresolved = (~degenerate) & (left | (right & ~bias_only))
    if np.any(unresolved):
        tags = np.asarray(artifact["tags"])[unresolved]
        raise RuntimeError(
            "analytical initialization contains unresolved edge sweeps for "
            f"{list(map(str, tags))}; widen the phase-10a grid")
    return build_init_map(
        artifact["tags"], artifact["thr_star"], artifact["c_bias_star"],
        degenerate)


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Phase 10: supervised V2 rescue fit")
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
    p.add_argument("--train-frac", type=float, default=0.6,
                   help="chronological training fraction per trajectory")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="chronological model-selection fraction")
    p.add_argument("--max-batch", type=int, default=16,
                   help="train snaps sampled per eval (↑ = less noise)")
    p.add_argument("--max-val", type=int, default=24,
                   help="maximum validation snapshots per fit")
    p.add_argument("--max-test", type=int, default=24,
                   help="maximum untouched test snapshots per fit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--optimiser", choices=["cma", "nelder-mead"],
                   default="cma")
    p.add_argument("--analytical-init", default="auto",
                   help="path to phase 10a analytical_*.npz; 'auto' looks "
                        "for results/analytical_N{N}_dim{D}.npz; 'none' "
                        "disables it.")
    args = p.parse_args()
    # Validate the split before loading inputs or starting an optimizer.
    try:
        chronological_split_indices(3, args.train_frac, args.val_frac)
    except ValueError as exc:
        p.error(str(exc))
    if args.optimiser == "cma" and not HAS_CMA:
        p.error("--optimiser cma requires the declared 'cma' dependency")
    run_provenance = provenance.start()

    print("=" * 88)
    print("  Phase 10: supervised V2 rescue fit")
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
        raise RuntimeError(
            "empty sweep: no configuration has both DNS and patch inputs")

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
            init_map, n_skip = build_init_map_from_artifact(
                ana, args.train_frac, args.val_frac)
            print(f"  analytical init loaded from "
                  f"{os.path.basename(ana_path)}: {len(init_map)} entries"
                  + (f", {n_skip} uninformative rows skipped"
                     if n_skip else "") + "\n")
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
            val_frac=args.val_frac,
            seed=args.seed, optimiser_name=args.optimiser,
            max_batch=args.max_batch, max_val=args.max_val,
            max_test=args.max_test, tag=tag,
            theta_init=theta_init,
        )
        print(f"  [{tag}] wall-time: {time.time()-t0:.0f}s\n")
        fname = f"train_{tag.replace(':', '-').replace(' ', '_')}_" \
                f"N{args.N}_dim{args.dim}.npz"
        np.savez_compressed(
            os.path.join(RESULTS_DIR, fname),
            **{k: v for k, v in res.items() if k != "tag"},
            tag_str=tag,
            cli_args=json.dumps(vars(args), sort_keys=True),
            seed=args.seed,
            **provenance.finish(run_provenance))
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
        classical_f1_val=np.array(
            [r["classical_f1_val"] for r in all_results]),
        delta_val=np.array([r["delta_val"] for r in all_results]),
        f1_test=np.array([r["best_f1_test"] for r in all_results]),
        classical_f1=np.array([r["classical_f1"] for r in all_results]),
        classical_f1_test=np.array(
            [r["classical_f1_test"] for r in all_results]),
        delta=np.array([r["delta"] for r in all_results]),
        delta_test=np.array([r["delta_test"] for r in all_results]),
        hits_bound=np.array([r["hits_bound"] for r in all_results]),
        optimiser=np.array([r["optimiser"] for r in all_results]),
        split_strategy="chronological_per_configuration",
        train_fraction=args.train_frac,
        validation_fraction=args.val_frac,
        cli_args=json.dumps(vars(args), sort_keys=True),
        seed=args.seed,
        **provenance.finish(run_provenance),
    )
    print(f"\n  saved compare: {os.path.basename(compare_path)}")
    print("\nPhase 10 complete.")


if __name__ == "__main__":
    main()
