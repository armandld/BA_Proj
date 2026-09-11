#!/usr/bin/env python3
"""V2 (hamiltonien SANS PARAMETRE) contre GBT et contre lui-meme, LOSO,
sur DNS reelle -- sans aucun entrainement de V2 : ses poids sont figes,
D-22 (la campagne Optuna, jamais lancee -- mesuree cette session a
plusieurs semaines sur ce materiel, trop couteuse) ne le concerne pas.

Trois questions, un seul jeu d'instantanes, mainenant sur les 4 regimes
Re disponibles (400/800/1200/1600, meme discipline multi-Re que
`h2b_loso_transfer.py`) :

  1. GBT/classique vs QAOA ;
  2. H0a/H0b : QAOA atteint-il l'optimum exact de SON hamiltonien complet
     (`exact_refine`, enumeration exhaustive, dim=3 = 18 qubits,
     tractable) ; l'atteindre parfaitement fait-il mieux ou pire que
     QAOA sur la verite terrain ?
  3. H3 : le hamiltonien complet (biais Z + couplages ZZ/ZZZZ) fait-il
     mieux que le biais Z SEUL (couplages annules via
     `h3_term_ablation.zero_hamiltonian_terms`, meme mecanisme que T26),
     pour QAOA et pour l'optimum exact ?

dim=3 (18 qubits) : la seule taille certifiee non degeneree pour QAOA
dans ce depot (docs/EVALUATION.md -- dim=2 est degenere, D-45/D-47).
N=96 : la resolution DNS deja utilisee par `h0_optimiser_equivalence.py`
et `h2b_ceiling_random_split.py`. Les labels dim=3 a Re=800/1200/1600
n'existaient pas (seul Re=400 en avait) ; generes une fois via
`dns_sweep.make_labels` sur les DNS deja presentes -- aucun nouveau
calcul DNS, seulement les labels manquants.

Psi cable (with_psi=True, instantane precedent de la meme trajectoire
DNS) -- une DNS reelle a un instant precedent, contrairement au jouet
statique (`h3_toy_model_check.py`, psi=0 exact). Le biais Z seul
(ablation ZZ/ZZZZ) garde le meme psi -- psi module le biais Z, ce n'est
ni un couplage ni un terme ablatable ici.

Le GBT et le seuil classique SONT ajustes -- sur les scenarios
d'ENTRAINEMENT du pli LOSO (tous regimes Re confondus), jamais sur le
scenario tenu. QAOA et l'optimum exact n'ont rien a ajuster : evalues
une seule fois par instantane, la meme decision sert a chaque pli.

IC95 bootstrap (percentile, B=1000, resample par INSTANTANE -- les
cellules d'un meme instantane sont spatialement correlees) sur le tenu,
meme fonction que `h2b_v1_hamiltonian_loso.paired_bootstrap_delta`.

Usage:
  python study/h2b_prediction/h2b_v2_hamiltonian_vs_gbt_loso.py --n-snaps 10
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qaoa_inputs import prepare_qaoa_inputs, run_qaoa_on_snapshot
from h2b_ceiling_random_split import (
    extract_features_2d, N_FEATS, best_threshold_f1, make_model, fit_eval)
from h3_term_ablation import zero_hamiltonian_terms, ground_state_mask
from h2b_v1_hamiltonian_loso import paired_bootstrap_delta
from config import RESULTS_DIR, RE_VALUES
import provenance

SCENARIOS_4 = ("harris_tearing", "kelvin_helmholtz", "mhd_rotor", "orszag_tang")


def gather_scenario(sc, re, N, dim, n_snaps, k_opt, shots, seed):
    """Une ligne par instantane : score classique, verite terrain, features
    GBT, decisions QAOA/exactes (hamiltonien complet ET biais Z seul) --
    toutes calculees UNE fois, reutilisees dans chaque pli LOSO ou ce
    scenario sert de train ou de tenu."""
    dns = np.load(os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{N}.npz"))
    patches = np.load(
        os.path.join(RESULTS_DIR, f"patches_{sc}_Re{re}_N{N}_dim{dim}.npz"))
    vx_all = dns["vx"].astype(np.float64)
    vy_all = dns["vy"].astype(np.float64)
    Bx_all = dns["Bx"].astype(np.float64)
    By_all = dns["By"].astype(np.float64)
    l2_all = patches["l2_errors"]
    l2_thr = float(patches["l2_threshold"])

    n_total = len(vx_all)
    step = max(1, (n_total - 1) // n_snaps)
    idx = list(range(1, n_total, step))[:n_snaps]  # >=1 : instant precedent requis

    rows = []
    for si in idx:
        vx, vy, Bx, By = vx_all[si], vy_all[si], Bx_all[si], By_all[si]
        feats_2d, score = extract_features_2d(vx, vy, Bx, By, N, dim, re)
        gt = (l2_all[si] >= l2_thr).ravel().astype(int)

        prev_fields = dict(vx=vx_all[si - 1], vy=vy_all[si - 1],
                           Bx=Bx_all[si - 1], By=By_all[si - 1])
        data_in, hamilt_params, _ = prepare_qaoa_inputs(
            vx, vy, Bx, By, N=N, n_patches=dim, Re=re, use_v2=True,
            with_psi=True, prev_fields=prev_fields)
        t0 = time.time()
        _marg, qh, qv, _opt_p, _wall = run_qaoa_on_snapshot(
            data_in, hamilt_params, dim=dim, reps=2,
            K_opt=k_opt, shots=shots, seed=seed)
        qaoa_refine = (qh | qv).ravel().astype(int)

        exact_refine, exact_E, degeneracy, _unif = ground_state_mask(
            hamilt_params, dim)
        exact_refine = exact_refine.ravel().astype(int)

        hp_zonly = zero_hamiltonian_terms(hamilt_params, ("ZZ", "ZZZZ"))
        _marg_z, qh_z, qv_z, _opt_p_z, _wall_z = run_qaoa_on_snapshot(
            data_in, hp_zonly, dim=dim, reps=2,
            K_opt=k_opt, shots=shots, seed=seed)
        qaoa_refine_zonly = (qh_z | qv_z).ravel().astype(int)
        exact_refine_zonly, _E_z, _deg_z, _unif_z = ground_state_mask(
            hp_zonly, dim)
        exact_refine_zonly = exact_refine_zonly.ravel().astype(int)
        wall = time.time() - t0

        rows.append(dict(
            scenario=sc, re=re, snap=int(si), score=score.ravel(), gt=gt,
            feats_site=feats_2d.reshape(-1, N_FEATS),
            qaoa_refine=qaoa_refine, exact_refine=exact_refine,
            exact_E=exact_E, degeneracy=degeneracy,
            qaoa_refine_zonly=qaoa_refine_zonly,
            exact_refine_zonly=exact_refine_zonly, wall=wall))
    return rows


def _cat(rows, key):
    return np.concatenate([r[key] for r in rows])


def _ci(rows, key_a, key_b, thr_a, thr_b, n_boot, rng):
    """CI95 (percentile) sur F1(key_a) - F1(key_b), resample par
    instantane. `thr=0.5` sur un masque deja binaire (0/1) le laisse
    inchange -- reutilise `paired_bootstrap_delta` sans distinguer
    score continu (classique/GBT) et decision binaire (QAOA/exact)."""
    Y = [r["gt"] for r in rows]
    Pa = [r[key_a].astype(float) for r in rows]
    Pb = [r[key_b].astype(float) for r in rows]
    lo, hi, _deltas, p_a_ge_b = paired_bootstrap_delta(
        Y, Pa, thr_a, Pb, thr_b, n_boot, rng)
    return lo, hi, p_a_ge_b


def run(re_values, N, dim, n_snaps, k_opt, shots, seed, n_boot):
    started = provenance.start()
    t0 = time.time()
    by_sc_re = {}
    for sc in SCENARIOS_4:
        for re in re_values:
            by_sc_re[(sc, re)] = gather_scenario(
                sc, re, N, dim, n_snaps, k_opt, shots, seed)
            wall = sum(r["wall"] for r in by_sc_re[(sc, re)])
            print(f"  {sc:<18} Re={re:<5} {len(by_sc_re[(sc, re)])} "
                  f"instantanes  QAOA wall={wall:.0f}s  "
                  f"({time.time()-t0:.0f}s cumule)", flush=True)
    print()

    header = (f"  {'tenu a l ecart':<18} {'classique':>9} {'GBT':>6} "
              f"{'QAOA':>6} {'exact':>6} | {'QAOA Z':>7} {'exact Z':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    rng = np.random.default_rng(seed)
    rows_out = []
    for held in SCENARIOS_4:
        train_rows = [r for s in SCENARIOS_4 if s != held
                     for re in re_values for r in by_sc_re[(s, re)]]
        held_rows = [r for re in re_values for r in by_sc_re[(held, re)]]

        Str, Ytr = _cat(train_rows, "score"), _cat(train_rows, "gt")
        Sva, Yva = _cat(held_rows, "score"), _cat(held_rows, "gt")
        thr_cls, _ = best_threshold_f1(Str, Ytr)
        f1_classical = f1_score(Yva, (Sva > thr_cls).astype(int),
                                zero_division=0)

        Xtr = np.concatenate([r["feats_site"] for r in train_rows], axis=0)
        Xva = np.concatenate([r["feats_site"] for r in held_rows], axis=0)
        res_gbt = fit_eval(make_model("gbt", seed, early_stopping=True),
                           Xtr, Ytr, Xva, Yva)

        Qva = _cat(held_rows, "qaoa_refine")
        f1_qaoa = f1_score(Yva, Qva, zero_division=0)

        # H0a/H0b ici : l'optimum exact du hamiltonien COMPLET.
        Eva = _cat(held_rows, "exact_refine")
        f1_exact = f1_score(Yva, Eva, zero_division=0)
        agree_qaoa_exact = float(np.mean(Qva == Eva))

        # H3 ici : biais Z seul (couplages ZZ/ZZZZ annules).
        Qva_z = _cat(held_rows, "qaoa_refine_zonly")
        Eva_z = _cat(held_rows, "exact_refine_zonly")
        f1_qaoa_zonly = f1_score(Yva, Qva_z, zero_division=0)
        f1_exact_zonly = f1_score(Yva, Eva_z, zero_division=0)

        ci_qaoa_exact = _ci(held_rows, "qaoa_refine", "exact_refine",
                            0.5, 0.5, n_boot, rng)
        ci_qaoa_classical = _ci(
            held_rows, "qaoa_refine", "score", 0.5, thr_cls, n_boot, rng)
        ci_exact_full_zonly = _ci(
            held_rows, "exact_refine", "exact_refine_zonly",
            0.5, 0.5, n_boot, rng)
        ci_qaoa_full_zonly = _ci(
            held_rows, "qaoa_refine", "qaoa_refine_zonly",
            0.5, 0.5, n_boot, rng)

        rows_out.append(dict(
            held=held, n_val_snaps=len(held_rows),
            f1_classical=f1_classical, f1_gbt=res_gbt["f1"],
            f1_qaoa=f1_qaoa, thr_classical=thr_cls, f1_exact=f1_exact,
            agree_qaoa_exact=agree_qaoa_exact,
            f1_qaoa_zonly=f1_qaoa_zonly, f1_exact_zonly=f1_exact_zonly,
            ci_qaoa_exact=ci_qaoa_exact,
            ci_qaoa_classical=ci_qaoa_classical,
            ci_exact_full_zonly=ci_exact_full_zonly,
            ci_qaoa_full_zonly=ci_qaoa_full_zonly))
        print(f"  {held:<18} {f1_classical:>9.3f} {res_gbt['f1']:>6.3f} "
              f"{f1_qaoa:>6.3f} {f1_exact:>6.3f} | "
              f"{f1_qaoa_zonly:>7.3f} {f1_exact_zonly:>8.3f}"
              f"   (n={len(held_rows)})")
        print(f"      IC95 QAOA-exact       [{ci_qaoa_exact[0]:+.3f},"
              f"{ci_qaoa_exact[1]:+.3f}]  p(QAOA>=exact)={ci_qaoa_exact[2]:.3f}")
        print(f"      IC95 QAOA-classique   [{ci_qaoa_classical[0]:+.3f},"
              f"{ci_qaoa_classical[1]:+.3f}]  "
              f"p(QAOA>=classique)={ci_qaoa_classical[2]:.3f}")
        print(f"      IC95 complet-Zseul(exact) "
              f"[{ci_exact_full_zonly[0]:+.3f},{ci_exact_full_zonly[1]:+.3f}]  "
              f"p={ci_exact_full_zonly[2]:.3f}")
        print(f"      IC95 complet-Zseul(QAOA)  "
              f"[{ci_qaoa_full_zonly[0]:+.3f},{ci_qaoa_full_zonly[1]:+.3f}]  "
              f"p={ci_qaoa_full_zonly[2]:.3f}")

    print()
    keys = ("f1_classical", "f1_gbt", "f1_qaoa", "f1_exact",
            "f1_qaoa_zonly", "f1_exact_zonly", "agree_qaoa_exact")
    means = {k: float(np.mean([r[k] for r in rows_out])) for k in keys}
    print(f"  moyenne LOSO : classique={means['f1_classical']:.3f}  "
          f"gbt={means['f1_gbt']:.3f}  qaoa={means['f1_qaoa']:.3f}  "
          f"exact={means['f1_exact']:.3f}")
    print(f"  moyenne biais Z seul : qaoa_Z={means['f1_qaoa_zonly']:.3f}  "
          f"exact_Z={means['f1_exact_zonly']:.3f}")
    print(f"  accord QAOA/exact (hamiltonien complet) = "
          f"{means['agree_qaoa_exact']:.3f}")
    n_qaoa_beats_cls = sum(r["f1_qaoa"] > r["f1_classical"] for r in rows_out)
    n_qaoa_beats_gbt = sum(r["f1_qaoa"] > r["f1_gbt"] for r in rows_out)
    n_full_beats_zonly_qaoa = sum(
        r["f1_qaoa"] > r["f1_qaoa_zonly"] for r in rows_out)
    n_full_beats_zonly_exact = sum(
        r["f1_exact"] > r["f1_exact_zonly"] for r in rows_out)
    print(f"  QAOA bat classique sur {n_qaoa_beats_cls}/4 plis, "
          f"GBT sur {n_qaoa_beats_gbt}/4 plis")
    print(f"  hamiltonien complet bat le biais Z seul sur "
          f"{n_full_beats_zonly_qaoa}/4 plis (QAOA), "
          f"{n_full_beats_zonly_exact}/4 plis (exact)")

    def ci_arrays(name, i):
        return np.array([r[name][i] for r in rows_out])

    payload = dict(
        re_values=np.array(re_values), N=N, dim=dim, n_snaps=n_snaps,
        k_opt=k_opt, shots=shots, seed=seed, n_boot=n_boot,
        scenarios=np.array(SCENARIOS_4),
        held=np.array([r["held"] for r in rows_out]),
        n_val_snaps=np.array([r["n_val_snaps"] for r in rows_out]),
        f1_classical=np.array([r["f1_classical"] for r in rows_out]),
        f1_gbt=np.array([r["f1_gbt"] for r in rows_out]),
        f1_qaoa=np.array([r["f1_qaoa"] for r in rows_out]),
        f1_exact=np.array([r["f1_exact"] for r in rows_out]),
        agree_qaoa_exact=np.array(
            [r["agree_qaoa_exact"] for r in rows_out]),
        f1_qaoa_zonly=np.array([r["f1_qaoa_zonly"] for r in rows_out]),
        f1_exact_zonly=np.array([r["f1_exact_zonly"] for r in rows_out]),
        thr_classical=np.array([r["thr_classical"] for r in rows_out]),
        degeneracy=np.array([r["degeneracy"] for pair in by_sc_re.values()
                             for r in pair]),
        ci_qaoa_exact_lo=ci_arrays("ci_qaoa_exact", 0),
        ci_qaoa_exact_hi=ci_arrays("ci_qaoa_exact", 1),
        p_qaoa_ge_exact=ci_arrays("ci_qaoa_exact", 2),
        ci_qaoa_classical_lo=ci_arrays("ci_qaoa_classical", 0),
        ci_qaoa_classical_hi=ci_arrays("ci_qaoa_classical", 1),
        p_qaoa_ge_classical=ci_arrays("ci_qaoa_classical", 2),
        ci_exact_full_zonly_lo=ci_arrays("ci_exact_full_zonly", 0),
        ci_exact_full_zonly_hi=ci_arrays("ci_exact_full_zonly", 1),
        p_exact_full_ge_zonly=ci_arrays("ci_exact_full_zonly", 2),
        ci_qaoa_full_zonly_lo=ci_arrays("ci_qaoa_full_zonly", 0),
        ci_qaoa_full_zonly_hi=ci_arrays("ci_qaoa_full_zonly", 1),
        p_qaoa_full_ge_zonly=ci_arrays("ci_qaoa_full_zonly", 2),
        **provenance.finish(started),
    )
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--re", nargs="+", type=int, default=list(RE_VALUES))
    p.add_argument("--N", type=int, default=96)
    p.add_argument("--dim", type=int, default=3,
                   help="18 qubits a dim=3 : la seule taille QAOA "
                        "certifiee non degeneree ici (voir docstring)")
    p.add_argument("--n-snaps", type=int, default=10,
                   help="par (scenario, Re) -- 4 scenarios x len(--re) "
                        "regimes x n-snaps instantanes au total")
    p.add_argument("--k-opt", type=int, default=60)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    payload = run(args.re, args.N, args.dim, args.n_snaps, args.k_opt,
                 args.shots, args.seed, args.n_boot)
    out_path = os.path.join(
        RESULTS_DIR,
        f"h2b_v2_hamiltonian_vs_gbt_loso_N{args.N}_dim{args.dim}"
        f"_n{args.n_snaps}_multire.npz")
    np.savez(out_path, **payload)
    print(f"\nEcrit : {out_path}")


if __name__ == "__main__":
    main()
