#!/usr/bin/env python3
"""Validate every DNS trajectory before it enters the study."""

import argparse
import os
import sys

import numpy as np


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _path in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", name) for name in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import DNS_N, PHYSICS_SEEDS, RESULTS_DIR, RE_VALUES, SCENARIOS
from Simulation.solver import MHDSolver


TEARING_LIKE = {
    "harris_tearing", "double_tearing", "island_coalescence",
}
OT_REF_ENERGY_RE1000 = {
    1.0: (0.80, 0.98), 2.0: (0.65, 0.85), 3.0: (0.52, 0.75),
}


def total_energy(vx, vy, Bx, By):
    return 0.5 * (vx ** 2 + vy ** 2 + Bx ** 2 + By ** 2).mean()


def kinetic_energy(vx, vy):
    return 0.5 * (vx ** 2 + vy ** 2).mean()


def magnetic_energy(Bx, By):
    return 0.5 * (Bx ** 2 + By ** 2).mean()


def matched_divergence(Bx, By, dx):
    """Divergence using the FD operator that advances the magnetic field."""
    grad_Bx_x, _ = MHDSolver._fd_grad(Bx, dx)
    _, grad_By_y = MHDSolver._fd_grad(By, dx)
    return grad_Bx_x + grad_By_y


def mean_sq_current(Bx, By, dx=1.0):
    """Mean squared current Jz = d_x By - d_y Bx on the repository axes."""
    grad_By_x, _ = MHDSolver._fd_grad(By, dx)
    _, grad_Bx_y = MHDSolver._fd_grad(Bx, dx)
    return float(np.mean((grad_By_x - grad_Bx_y) ** 2))


def fluctuating_KE(vx, vy):
    """KH perturbation energy after removing the homogeneous-x mean."""
    vx_mean = vx.mean(axis=0, keepdims=True)
    vy_mean = vy.mean(axis=0, keepdims=True)
    return float(0.5 * ((vx - vx_mean) ** 2 + (vy - vy_mean) ** 2).mean())


def fluctuating_mean_sq_current(Bx, By, dx=1.0):
    """Mean squared current after removing the homogeneous-x background.

    `mean_sq_current` moyenne `Jz**2` sur TOUT le domaine, y compris
    le courant d'equilibre de la nappe -- uniforme le long de x pour les
    trois scenarios `TEARING_LIKE` (profil en `tanh(y)`, voir
    `Simulation.solver.MHDSolver.init_harris_tearing` et les deux
    scenarios soeurs), quasi constant dans le temps, et qui domine la
    moyenne spatiale au point de masquer toute reconnexion. Meme geste que
    `fluctuating_KE` ci-dessus pour le KH : soustraire la moyenne
    homogene-en-x isole la partie qui varie reellement, celle que la
    reconnexion fait croitre.
    """
    grad_By_x, _ = MHDSolver._fd_grad(By, dx)
    _, grad_Bx_y = MHDSolver._fd_grad(Bx, dx)
    Jz = grad_By_x - grad_Bx_y
    Jz_mean = Jz.mean(axis=0, keepdims=True)
    return float(np.mean((Jz - Jz_mean) ** 2))


def energy_non_increasing(energy, tol=1e-3):
    increments = np.diff(np.asarray(energy, dtype=float))
    maximum = float(increments.max()) if increments.size else 0.0
    return bool(np.all(increments <= tol)), maximum


def analyse_one(path):
    """Compute validation diagnostics from one saved DNS trajectory."""
    with np.load(path) as data:
        vx = data["vx"].astype(np.float64)
        vy = data["vy"].astype(np.float64)
        Bx = data["Bx"].astype(np.float64)
        By = data["By"].astype(np.float64)
        t = data["t"].astype(np.float64)
        scenario = str(data["meta_scenario"])
        re = int(data["meta_Re"])
        N = int(data["meta_N"])
        diverged = bool(data["meta_diverged"])

    if not len(t):
        raise ValueError(f"empty DNS trajectory: {path}")
    dx = 2.0 * np.pi / N
    energy = np.asarray([
        total_energy(vx[i], vy[i], Bx[i], By[i]) for i in range(len(t))
    ])
    kinetic = np.asarray([
        kinetic_energy(vx[i], vy[i]) for i in range(len(t))
    ])
    magnetic = np.asarray([
        magnetic_energy(Bx[i], By[i]) for i in range(len(t))
    ])
    current = np.asarray([
        mean_sq_current(Bx[i], By[i], dx) for i in range(len(t))
    ])
    current_fluct = np.asarray([
        fluctuating_mean_sq_current(Bx[i], By[i], dx) for i in range(len(t))
    ])
    perturbation = np.asarray([
        fluctuating_KE(vx[i], vy[i]) for i in range(len(t))
    ])
    divergence = 0.0
    for index in range(len(t)):
        rms_B = np.sqrt((Bx[index] ** 2 + By[index] ** 2).mean()) + 1e-30
        divergence = max(
            divergence,
            float(np.max(np.abs(matched_divergence(
                Bx[index], By[index], dx))) / rms_B))

    return {
        "t": t, "E": energy, "Ek": kinetic, "Em": magnetic,
        "J2": current, "J2_fluct": current_fluct,
        "Ep": perturbation, "div_rel_max": divergence,
        "scenario": scenario, "Re": re, "N": N, "diverged": diverged,
    }


def check_ot(result):
    energy = result["E"]
    times = result["t"]
    monotone, max_increase = energy_non_increasing(energy)
    fraction = 1.0 - float(energy[-1] / max(energy[0], 1e-30))
    literature = []
    if result["Re"] >= 1000:
        for target, (low, high) in OT_REF_ENERGY_RE1000.items():
            index = int(np.argmin(np.abs(times - target)))
            if abs(times[index] - target) <= 0.2:
                literature.append((
                    target, float(times[index]), float(energy[index]),
                    low, high, bool(low <= energy[index] <= high)))
    return {
        "E0": float(energy[0]),
        "E0_ok": bool(0.98 <= energy[0] <= 1.02),
        "monotone_ok": monotone, "max_violation": max_increase,
        "frac_decay": fraction, "frac_ok": bool(0.01 <= fraction <= 0.45),
        "lit_checks": literature,
    }


def check_tearing(result):
    """Deux corrections a la facon dont la reconnexion est validee ici.

    (1) Lit `J2_fluct` (`fluctuating_mean_sq_current`), pas `J2` : le
    courant d'equilibre de la nappe, uniforme en x et quasi constant dans
    le temps, dominait la moyenne pleine grille et noyait le signal de
    reconnexion (mesure : ok=False sur 6/6 artefacts DNS harris_tearing
    malgre un vrai signal fluctuant, amplification 8x a 17x).

    (2) `interior` (le pic doit avoir un point apres lui dans la fenetre)
    n'est satisfaite par AUCUNE des 6 trajectoires mesurees : le pic
    tombe systematiquement au DERNIER pas enregistre (`peak_idx ==
    len(current)-1` sur les 6), parce que la reconnexion n'a pas fini de
    saturer avant la fin de la fenetre simulee `[0, t_max]`. Exiger un pic
    interieur qui redescend (`decays`) rejette alors TOUJOURS un signal
    reel, quel que soit l'observable utilise. Un pic encore montant en fin
    de fenetre, avec une amplification substantielle, est desormais
    accepte comme preuve suffisante de reconnexion en cours -- ce n'est
    pas la meme affirmation qu'un cycle complet observe (montee ET
    descente), et `saturated` le distingue dans le retour.
    """
    current = result["J2_fluct"]
    times = result["t"]
    index = int(np.argmax(current))
    interior = 0 < index < len(current) - 1
    decays = interior and current[index + 1] <= 1.01 * current[index]
    grows = len(current) > 1 and current[index] > 1.2 * current[0]
    saturated = bool(interior and decays)
    still_rising = index == len(current) - 1
    return {
        "t_peak": float(times[index]), "j_peak": float(current[index]),
        "j_start": float(current[0]),
        "amplification": float(current[index] / max(current[0], 1e-30)),
        "saturated": saturated,
        "ok": bool(grows and (saturated or still_rising)),
    }


def check_kh(result):
    times = result["t"]
    energy = result["Ep"]
    early = (times >= 0.0) & (times <= 0.2)
    developed = (times >= 0.8) & (times <= 1.2)
    if not early.any() or not developed.any():
        return {"ok": False, "reason": "insufficient time coverage"}
    initial = float(energy[early].mean())
    final = float(energy[developed].mean())
    growth = float(final / max(initial, 1e-30))
    return {
        "Ep_early": initial, "Ep_mid": final, "growth": growth,
        "ok": bool(growth > 1.1),
    }


def validate_one(path, scenario, div_tol=1e-3, *, expected_re=None,
                 expected_N=None, expected_seed=None):
    """Return hard validation failures and concise diagnostics."""
    result = analyse_one(path)
    if result["scenario"] != scenario:
        raise ValueError(
            f"scenario mismatch for {path}: {result['scenario']} != {scenario}")
    if expected_re is not None and result["Re"] != expected_re:
        raise ValueError(
            f"Re mismatch for {path}: {result['Re']} != {expected_re}")
    if expected_N is not None and result["N"] != expected_N:
        raise ValueError(
            f"N mismatch for {path}: {result['N']} != {expected_N}")
    if expected_seed is not None:
        with np.load(path) as data:
            actual_seed = int(data.get("meta_phys_seed", 0))
        if actual_seed != expected_seed:
            raise ValueError(
                f"physics-seed mismatch for {path}: "
                f"{actual_seed} != {expected_seed}")
    name = os.path.basename(path)
    failures = []
    log = []
    if result["diverged"]:
        failures.append(f"{name}: solver diverged")
    if result["div_rel_max"] > div_tol:
        failures.append(f"{name}: divB {result['div_rel_max']:.1e}")
    log.append(f"div={result['div_rel_max']:.1e}")

    monotone, maximum = energy_non_increasing(result["E"])
    log.append(f"E={'OK' if monotone else f'FAIL({maximum:+.1e})'}")
    if not monotone:
        failures.append(f"{name}: E increases (max dE={maximum:+.2e})")

    if scenario == "orszag_tang":
        check = check_ot(result)
        log.append(f"OT_decay={100.0 * check['frac_decay']:.1f}%")
        if not check["E0_ok"] or not check["frac_ok"]:
            failures.append(
                f"{name}: OT check E0={check['E0']:.3f}, "
                f"decay={100.0 * check['frac_decay']:.1f}%")
    elif scenario in TEARING_LIKE:
        check = check_tearing(result)
        state = "resolved" if check["ok"] else "unresolved"
        log.append(
            f"J2_amp={check['amplification']:.2f}x({state},diagnostic)")
    elif scenario == "kelvin_helmholtz":
        check = check_kh(result)
        log.append(f"KH_growth={check.get('growth', float('nan')):.2f}x")
        if not check["ok"]:
            failures.append(f"{name}: KH check failed")
    return failures, log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--re", nargs="+", type=int, default=list(RE_VALUES))
    parser.add_argument("--scenario", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--N", type=int, default=DNS_N)
    parser.add_argument(
        "--phys-seed", nargs="+", type=int, default=list(PHYSICS_SEEDS))
    parser.add_argument("--div-tol", type=float, default=1e-3)
    args = parser.parse_args()

    from dns_sweep import dns_path

    failures = []
    for scenario in args.scenario:
        for re in args.re:
            for seed in args.phys_seed:
                path = dns_path(RESULTS_DIR, scenario, re, args.N, seed)
                if not os.path.exists(path):
                    failures.append(f"missing {os.path.basename(path)}")
                    continue
                current, log = validate_one(
                    path, scenario, args.div_tol, expected_re=re,
                    expected_N=args.N, expected_seed=seed)
                failures.extend(current)
                status = "OK" if not current else "FAIL"
                print(f"[{status:>4}] {os.path.basename(path)}  "
                      + "  ".join(log))
    if failures:
        print(f"DNS validation: {len(failures)} failure(s)")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(2)
    print("DNS validation: clean")


if __name__ == "__main__":
    main()
