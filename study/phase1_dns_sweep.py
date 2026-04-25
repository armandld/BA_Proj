#!/usr/bin/env python3
"""
Phase 1 - DNS sweep at multiple Reynolds numbers.

Runs Orszag-Tang and Harris Tearing at N=256 for Re = 400, 800, 1200, 1600.
Saves full field snapshots at regular intervals for downstream analysis.

Output per (scenario, Re):
  results/dns_{scenario}_Re{Re}_N{N}.npz
  Contains: snapshots (list of dicts with vx,vy,Bx,By,t,step), metadata.

Usage:
  python study/phase1_dns_sweep.py                     # all combos
  python study/phase1_dns_sweep.py --re 800 1200       # specific Re values
  python study/phase1_dns_sweep.py --scenario orszag_tang --re 400
  python study/phase1_dns_sweep.py --N 512             # high-res run
"""
import argparse, time, os, sys
import numpy as np

# -- bootstrap imports --
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import (
    RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N, DT_INIT, SCENARIO_CONFIG,
)
from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver


def init_scenario(sim, scenario):
    """Initialize solver with the given scenario."""
    inits = {
        "orszag_tang": sim.init_orszag_tang,
        "harris_tearing": sim.init_harris_tearing,
        "kelvin_helmholtz": sim.init_kelvin_helmholtz,
        "mhd_rotor": sim.init_mhd_rotor,
    }
    if scenario not in inits:
        raise ValueError(f"Unknown scenario: {scenario}")
    inits[scenario]()


def run_dns(scenario, Re, N=DNS_N, dt_init=DT_INIT):
    """
    Run a single DNS simulation and return snapshots.

    Returns:
      snapshots: list of dicts {vx, vy, Bx, By, t, step, dt}
      metadata: dict with run parameters
    """
    cfg = SCENARIO_CONFIG[scenario]
    t_max = cfg["t_max"]
    snap_dt = cfg["snapshot_dt"]

    grid = PeriodicGrid(N)
    sim = MHDSolver(grid, dt=dt_init, Re=Re, Rm=Re)
    init_scenario(sim, scenario)

    print(f"  [{scenario} Re={Re} N={N}] Starting DNS, t_max={t_max}")

    snapshots = []
    t_current = 0.0
    step = 0
    next_snap_t = 0.0
    wall_start = time.time()

    while t_current < t_max:
        # adaptive dt
        sim.dt = sim.adapt_dt(cfl_target=0.4)

        # capture snapshot at regular intervals
        if t_current >= next_snap_t - 1e-10:
            snapshots.append({
                "vx": sim.vx.copy(),
                "vy": sim.vy.copy(),
                "Bx": sim.Bx.copy(),
                "By": sim.By.copy(),
                "t": t_current,
                "step": step,
                "dt": sim.dt,
            })
            next_snap_t += snap_dt
            if len(snapshots) % 5 == 0:
                wall_elapsed = time.time() - wall_start
                print(f"    t={t_current:.3f}/{t_max} step={step} "
                      f"dt={sim.dt:.2e} snapshots={len(snapshots)} "
                      f"wall={wall_elapsed:.0f}s")

        # advance
        sim.step_full(record_stats=False)
        t_current += sim.dt
        step += 1

        # divergence check
        if sim.is_diverged():
            print(f"  [DIVERGED] {scenario} Re={Re} at t={t_current:.4f} step={step}")
            break

    wall_total = time.time() - wall_start
    print(f"  [{scenario} Re={Re}] Done: {len(snapshots)} snapshots, "
          f"{step} steps, wall={wall_total:.0f}s")

    metadata = {
        "scenario": scenario,
        "Re": Re, "Rm": Re,
        "N": N,
        "t_max": t_max,
        "dt_init": dt_init,
        "snapshot_dt": snap_dt,
        "n_snapshots": len(snapshots),
        "n_steps": step,
        "final_t": t_current,
        "wall_seconds": wall_total,
        "diverged": sim.is_diverged(),
    }
    return snapshots, metadata


def save_dns(snapshots, metadata, outdir=RESULTS_DIR):
    """Save DNS results to compressed npz."""
    sc = metadata["scenario"]
    Re = metadata["Re"]
    N = metadata["N"]
    fname = f"dns_{sc}_Re{Re}_N{N}.npz"
    path = os.path.join(outdir, fname)

    # pack snapshots into arrays
    n = len(snapshots)
    Ng = metadata["N"]
    vx_all = np.zeros((n, Ng, Ng), dtype=np.float32)
    vy_all = np.zeros_like(vx_all)
    Bx_all = np.zeros_like(vx_all)
    By_all = np.zeros_like(vx_all)
    t_all = np.zeros(n)
    step_all = np.zeros(n, dtype=np.int32)

    for i, s in enumerate(snapshots):
        vx_all[i] = s["vx"].astype(np.float32)
        vy_all[i] = s["vy"].astype(np.float32)
        Bx_all[i] = s["Bx"].astype(np.float32)
        By_all[i] = s["By"].astype(np.float32)
        t_all[i] = s["t"]
        step_all[i] = s["step"]

    np.savez_compressed(
        path,
        vx=vx_all, vy=vy_all, Bx=Bx_all, By=By_all,
        t=t_all, step=step_all,
        **{f"meta_{k}": v for k, v in metadata.items()
           if isinstance(v, (int, float, str, bool))},
    )
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved: {path} ({size_mb:.1f} MB)")
    return path


def main():
    parser = argparse.ArgumentParser(description="Phase 1: DNS sweep")
    parser.add_argument("--scenario", nargs="+", default=SCENARIOS)
    parser.add_argument("--re", nargs="+", type=int, default=RE_VALUES)
    parser.add_argument("--N", type=int, default=DNS_N)
    args = parser.parse_args()

    combos = [(sc, re) for sc in args.scenario for re in args.re]
    print(f"Phase 1: DNS sweep - {len(combos)} runs")
    print(f"  Scenarios: {args.scenario}")
    print(f"  Re values: {args.re}")
    print(f"  Resolution: N={args.N}")
    print()

    results = {}
    for i, (sc, re) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] {sc} Re={re} N={args.N}")
        snapshots, metadata = run_dns(sc, re, N=args.N)
        path = save_dns(snapshots, metadata)
        results[(sc, re)] = {"path": path, "metadata": metadata}
        print()

    # summary
    print("=" * 60)
    print("Phase 1 complete.")
    for (sc, re), info in results.items():
        m = info["metadata"]
        status = "DIVERGED" if m["diverged"] else "OK"
        print(f"  {sc} Re={re}: {m['n_snapshots']} snaps, "
              f"{m['wall_seconds']:.0f}s [{status}]")


if __name__ == "__main__":
    main()
