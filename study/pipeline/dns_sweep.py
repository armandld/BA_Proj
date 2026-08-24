#!/usr/bin/env python3
"""Generate, label and validate the complete DNS campaign."""

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _path in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", name) for name in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

import provenance
from config import (
    DNS_N, DT_INIT, PHYSICS_NOISE_AMPLITUDE, PHYSICS_SEEDS, RESULTS_DIR,
    RE_VALUES, SCENARIOS, SCENARIO_CONFIG,
)
from h2b_feature_selection import git_commit_hash
from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver


def init_scenario(sim, scenario, phys_seed=0, noise_amplitude=None):
    """Initialise one scenario and apply the reproducible IC perturbation."""
    initialisers = {
        "orszag_tang": sim.init_orszag_tang,
        "harris_tearing": sim.init_harris_tearing,
        "kelvin_helmholtz": sim.init_kelvin_helmholtz,
        "mhd_rotor": sim.init_mhd_rotor,
        "lamb_oseen": sim.init_lamb_oseen_vortex,
        "island_coalescence": sim.init_island_coalescence,
        "double_tearing": sim.init_double_tearing,
        "magnetic_twist": sim.init_magnetic_twist,
    }
    if scenario not in initialisers:
        raise ValueError(f"unknown scenario: {scenario}")
    initialisers[scenario]()
    if noise_amplitude is None:
        noise_amplitude = PHYSICS_NOISE_AMPLITUDE[scenario]
    sim.apply_physics_perturbation(phys_seed, noise_amplitude)


def dns_path(results_dir, scenario, re, N, phys_seed):
    suffix = "" if phys_seed == 0 else f"_seed{phys_seed}"
    return os.path.join(
        results_dir, f"dns_{scenario}_Re{re}_N{N}{suffix}.npz")


def patches_path(results_dir, scenario, re, N, dim, phys_seed):
    suffix = "" if phys_seed == 0 else f"_seed{phys_seed}"
    return os.path.join(
        results_dir,
        f"patches_{scenario}_Re{re}_N{N}_dim{dim}{suffix}.npz")


def presence_matrix(results_dir, scenarios, re_values, N, physics_seeds):
    return {
        (scenario, seed): sum(
            os.path.exists(dns_path(results_dir, scenario, re, N, seed))
            for re in re_values)
        for scenario in scenarios for seed in physics_seeds
    }


def run_dns(scenario, re, N=DNS_N, dt_init=DT_INIT, phys_seed=0,
            noise_amplitude=None):
    """Integrate one trajectory and return immutable snapshot copies."""
    cfg = SCENARIO_CONFIG[scenario]
    grid = PeriodicGrid(N)
    sim = MHDSolver(grid, dt=dt_init, Re=re, Rm=re)
    if noise_amplitude is None:
        noise_amplitude = PHYSICS_NOISE_AMPLITUDE[scenario]
    init_scenario(sim, scenario, phys_seed, noise_amplitude)

    snapshots = []
    current_t = 0.0
    step = 0
    next_snapshot = 0.0
    started = time.time()
    while current_t < cfg["t_max"]:
        sim.dt = min(
            sim.adapt_dt(cfl_target=0.4), cfg["t_max"] - current_t)
        if current_t >= next_snapshot - 1e-10:
            snapshots.append({
                "vx": sim.vx.copy(), "vy": sim.vy.copy(),
                "Bx": sim.Bx.copy(), "By": sim.By.copy(),
                "t": current_t, "step": step, "dt": sim.dt,
            })
            next_snapshot += cfg["snapshot_dt"]
        sim.step_full(record_stats=False)
        current_t += sim.dt
        step += 1
        if sim.is_diverged():
            break

    metadata = {
        "scenario": scenario, "Re": re, "Rm": re, "N": N,
        "t_max": cfg["t_max"], "dt_init": dt_init,
        "snapshot_dt": cfg["snapshot_dt"],
        "n_snapshots": len(snapshots), "n_steps": step,
        "final_t": current_t, "wall_seconds": time.time() - started,
        "diverged": sim.is_diverged(), "phys_seed": int(phys_seed),
        "physics_noise_amplitude": float(noise_amplitude),
    }
    return snapshots, metadata


def save_dns(snapshots, metadata, outdir=RESULTS_DIR, cli_args=None):
    """Atomically save one trajectory with complete provenance."""
    os.makedirs(outdir, exist_ok=True)
    N = int(metadata["N"])
    arrays = {
        name: np.asarray([snapshot[name] for snapshot in snapshots],
                         dtype=np.float32)
        for name in ("vx", "vy", "Bx", "By")
    }
    arrays["t"] = np.asarray([snapshot["t"] for snapshot in snapshots])
    arrays["step"] = np.asarray(
        [snapshot["step"] for snapshot in snapshots], dtype=np.int32)
    arrays.update({
        f"meta_{key}": value for key, value in metadata.items()
        if isinstance(value, (int, float, str, bool))
    })
    arrays["git_hash"] = metadata.get(
        "git_hash_at_start", git_commit_hash())
    arrays["cli_args"] = json.dumps(cli_args or {}, sort_keys=True)

    output = dns_path(
        outdir, metadata["scenario"], metadata["Re"], N,
        int(metadata["phys_seed"]))
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{os.path.basename(output)}.", suffix=".npz",
        dir=outdir, delete=False)
    temporary = handle.name
    handle.close()
    try:
        np.savez_compressed(temporary, **arrays)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return output


def make_labels(path, dims, results_dir, phys_seed):
    """Atomically generate phase-2 labels for one saved trajectory."""
    from hard_patch_labels import analyze_dns_file

    results_by_dim, metadata = analyze_dns_file(path, dims)
    with np.load(path) as dns:
        source_git_hash = str(dns.get("git_hash", "unknown"))
    for dim, result in results_by_dim.items():
        output = patches_path(
            results_dir, metadata["scenario"], metadata["Re"],
            metadata["N"], dim, phys_seed)
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{os.path.basename(output)}.", suffix=".npz",
            dir=results_dir, delete=False)
        temporary = handle.name
        handle.close()
        try:
            np.savez_compressed(
                temporary, l2_errors=result["l2_errors"],
                classical_scores=result["classical_scores"],
                is_hard=result["is_hard"],
                l2_threshold=result["l2_threshold"], t=metadata["t"],
                scenario=metadata["scenario"], Re=metadata["Re"],
                N=metadata["N"], n_patches=dim, phys_seed=phys_seed,
                source_dns=os.path.basename(path),
                source_git_hash=source_git_hash)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _print_presence(args):
    matrix = presence_matrix(
        RESULTS_DIR, args.scenario, args.re, args.N, args.phys_seed)
    print("\nPresence matrix (files across requested Re values):")
    print(f"  {'scenario':<20} "
          + " ".join(f"seed{seed:>3}" for seed in args.phys_seed))
    for scenario in args.scenario:
        values = " ".join(
            f"{matrix[(scenario, seed)]:>7}" for seed in args.phys_seed)
        print(f"  {scenario:<20} {values}")
    return all(value == len(args.re) for value in matrix.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--re", nargs="+", type=int, default=list(RE_VALUES))
    parser.add_argument("--N", type=int, default=DNS_N)
    parser.add_argument(
        "--phys-seed", nargs="+", type=int, default=list(PHYSICS_SEEDS))
    parser.add_argument(
        "--noise-amplitude", type=float,
        help="override the registered scenario-specific amplitude")
    parser.add_argument("--labels-dim", nargs="+", type=int, default=[4])
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    unknown = sorted(set(args.scenario) - set(SCENARIOS))
    if unknown:
        parser.error(f"unknown scenario(s): {', '.join(unknown)}")
    if len(set(args.phys_seed)) != len(args.phys_seed):
        parser.error("--phys-seed contains duplicates")
    if any(seed < 0 for seed in args.phys_seed):
        parser.error("physics seeds must be non-negative")
    if args.noise_amplitude is not None and args.noise_amplitude < 0:
        parser.error("--noise-amplitude must be non-negative")
    if any(args.N % dim for dim in args.labels_dim):
        parser.error("each --labels-dim must divide N")

    campaign = provenance.start()
    if (campaign["dirty_at_start"] and not args.allow_dirty
            and not args.dry_run and not args.validate_only):
        raise RuntimeError("refusing a DNS campaign from a dirty tree")

    combinations = [
        (scenario, re, seed) for scenario in args.scenario
        for re in args.re for seed in args.phys_seed
    ]
    pending = [
        item for item in combinations
        if args.overwrite or not os.path.exists(
            dns_path(RESULTS_DIR, item[0], item[1], args.N, item[2]))
    ]
    print(f"DNS campaign: {len(combinations)} trajectories, "
          f"{len(pending)} to generate")
    if args.dry_run:
        for scenario, re, seed in pending:
            print(f"  {scenario} Re={re} seed={seed}")
        return

    if not args.validate_only:
        for index, (scenario, re, seed) in enumerate(pending, 1):
            print(f"[{index}/{len(pending)}] {scenario} Re={re} seed={seed}",
                  flush=True)
            snapshots, metadata = run_dns(
                scenario, re, N=args.N, phys_seed=seed,
                noise_amplitude=args.noise_amplitude)
            metadata.update(provenance.finish(campaign))
            path = save_dns(snapshots, metadata, RESULTS_DIR, vars(args))
            print(f"  saved {os.path.basename(path)}", flush=True)

    from dns_validation import validate_one

    failures = []
    for scenario, re, seed in combinations:
        path = dns_path(RESULTS_DIR, scenario, re, args.N, seed)
        if not os.path.exists(path):
            failures.append(f"missing {os.path.basename(path)}")
            continue
        if not args.validate_only and not args.skip_labels:
            make_labels(path, args.labels_dim, RESULTS_DIR, seed)
        current, log = validate_one(
            path, scenario, expected_re=re, expected_N=args.N,
            expected_seed=seed)
        failures.extend(current)
        status = "OK" if not current else "FAIL"
        print(f"[{status:>4}] {os.path.basename(path)}  " + "  ".join(log))

    complete = _print_presence(args)
    if failures:
        print(f"\nDNS validation: {len(failures)} failure(s)")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("\nDNS validation: clean")
    if failures or not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
