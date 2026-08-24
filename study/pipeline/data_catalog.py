"""Resolve complete study panels without silently dropping trajectories."""

import argparse
import os

from dns_sweep import dns_path


def dns_trajectory_paths(results_dir, scenarios, re_values, N,
                         physics_seeds):
    """Return all requested DNS paths, or fail on the complete missing set."""
    rows = {scenario: [] for scenario in scenarios}
    missing = []
    for scenario in scenarios:
        for re in re_values:
            for seed in physics_seeds:
                path = dns_path(results_dir, scenario, re, N, seed)
                if os.path.exists(path):
                    rows[scenario].append((re, seed, path))
                else:
                    missing.append(path)
    if missing:
        preview = ", ".join(os.path.basename(path) for path in missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise FileNotFoundError(
            f"incomplete trajectory panel (DNS): "
            f"{len(missing)} missing artifact(s): "
            f"{preview}{suffix}")
    return rows


def labelled_trajectory_paths(results_dir, scenarios, re_values, N, dim,
                              physics_seeds, label_suffix=""):
    """Return ``{scenario: [(Re, seed, dns, labels), ...]}``.

    Every requested trajectory is mandatory. A missing input aborts before
    analysis so a partial campaign cannot masquerade as a complete panel.
    """
    dns_rows = dns_trajectory_paths(
        results_dir, scenarios, re_values, N, physics_seeds)
    rows = {scenario: [] for scenario in scenarios}
    missing = []
    for scenario, trajectories in dns_rows.items():
        for re, seed, dns in trajectories:
            seed_suffix = "" if seed == 0 else f"_seed{seed}"
            labels = os.path.join(
                results_dir,
                f"patches_{scenario}_Re{re}_N{N}_dim{dim}"
                f"{label_suffix}{seed_suffix}.npz")
            if os.path.exists(labels):
                rows[scenario].append((re, seed, dns, labels))
            else:
                missing.append(labels)
    if missing:
        preview = ", ".join(os.path.basename(path) for path in missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise FileNotFoundError(
            f"incomplete trajectory panel: {len(missing)} missing artifact(s): "
            f"{preview}{suffix}")
    return rows


def main():
    from config import DNS_N, PHYSICS_SEEDS, RESULTS_DIR, RE_VALUES, SCENARIOS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=DNS_N)
    parser.add_argument("--dim", nargs="+", type=int, default=[4])
    parser.add_argument("--scenario", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--re", nargs="+", type=int, default=list(RE_VALUES))
    parser.add_argument("--phys-seed", nargs="+", type=int,
                        default=list(PHYSICS_SEEDS))
    args = parser.parse_args()
    for dim in args.dim:
        panel = labelled_trajectory_paths(
            RESULTS_DIR, args.scenario, args.re, args.N, dim,
            args.phys_seed)
        count = sum(map(len, panel.values()))
        print(f"complete panel: dim={dim}, {count} trajectories")


if __name__ == "__main__":
    main()
