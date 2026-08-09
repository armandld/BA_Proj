#!/usr/bin/env python3
"""
V4 Task 18 - Contrefactuel : les termes ZZ comptent-ils SANS la fenetre ?

QUESTION. T13 montre qu'annuler la famille ZZ ne change aucune decision.
T17 montre pourquoi : la fenetre gaussienne d'incertitude jette 88.6% a
99.99% du couplage physique, et le jette preferentiellement la ou il est
le plus grand. Reste la question que tout rapporteur posera, et dont
depend la portee de la conclusion :

  L'inertie causale des termes ZZ est-elle une propriete DE LA
  FORMULATION Ising, ou un artefact DE CETTE IMPLEMENTATION ?

Les deux lectures menent a des articles differents :
  - si, fenetre neutralisee, l'ablation ZZ continue de ne rien changer,
    alors le couplage multi-corps est inerte pour une raison plus
    profonde (le biais Z domine, l'etat fondamental reste uniforme) et la
    critique porte sur l'approche ;
  - si, fenetre neutralisee, l'ablation ZZ change des decisions, alors le
    signal existe et c'est le pipeline qui le detruit : le defaut est
    reparable et la critique porte sur l'implementation, pas sur l'idee.

PROTOCOLE. Pour chaque snapshot on construit DEUX Hamiltoniens avec la
meme physique et le meme mappeur deploye (v1) :
  A. `windowed`   — le pipeline tel qu'il tourne aujourd'hui ;
  B. `no_window`  — identique, sigma -> +inf, donc w == 1 partout.
La neutralisation se fait en substituant la constante de module utilisee
pour CONSTRUIRE le mappeur, puis en la restaurant : aucune ligne de V1
n'est modifiee, et la substitution est verifiee a posteriori (le couplage
sans fenetre doit dominer le couplage avec fenetre).

On rejoue ensuite, sur chacun des deux, les ablations de la tache 13
(`zero_hamiltonian_terms`, `ground_state_mask` sont importees, jamais
redefinies). L'ablation `full` reste le controle : elle doit donner
exactement zero changement dans les deux bras.

Sortie : results/t18_window_counterfactual_N{N}_dim{D}.npz
Usage :
  python study/v4/t18_window_counterfactual.py --N 256 --dim 2 --n-snaps 2
"""
import argparse, json, os, sys, time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# --- chemins du dépôt (bloc unique, généré) -------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# -------------------------------------------------------------------------

from t1_feature_selection import git_commit_hash
from t11_solver_attribution import f1_from_masks
from t13_term_ablation import (ABLATIONS, ground_state_mask,
                               zero_hamiltonian_terms)

# Les deux bras compares. `no_window` neutralise la gaussienne en portant
# sigma a une valeur ou exp(-((s-thr)/sigma)^2) vaut 1 en double precision.
HUGE_SIGMA = 1e9


def _c_amplitude(hp):
    """max |C_edges|, quel que soit le format (tuple h/v ou tableau)."""
    c = hp["C_edges"]
    if isinstance(c, (tuple, list)):
        return float(max(np.max(np.abs(np.asarray(x))) for x in c))
    return float(np.max(np.abs(np.asarray(c))))


def prepare_both_arms(vx, vy, Bx, By, N, dim, re):
    """Retourne (hp_windowed, hp_nowindow) pour le mappeur v1 deploye.

    La seule difference entre les deux appels est sigma. On substitue la
    constante de module utilisee a la construction du mappeur, puis on la
    restaure inconditionnellement.
    """
    import phase5_qaoa_eval as p5

    _, hp_w, _ = p5.prepare_qaoa_inputs(vx, vy, Bx, By, N, dim, re,
                                        use_v2=False)
    saved = p5.TRAINED_SIGMA
    try:
        p5.TRAINED_SIGMA = HUGE_SIGMA
        _, hp_nw, _ = p5.prepare_qaoa_inputs(vx, vy, Bx, By, N, dim, re,
                                             use_v2=False)
    finally:
        p5.TRAINED_SIGMA = saved
    assert p5.TRAINED_SIGMA == saved, "sigma not restored"

    # verification de la substitution : sans fenetre, le couplage ne peut
    # pas etre plus faible qu'avec. Si l'egalite est stricte, la fenetre
    # n'agissait deja pas et le contrefactuel serait vide de sens.
    a_w, a_nw = _c_amplitude(hp_w), _c_amplitude(hp_nw)
    assert a_nw >= a_w * (1.0 - 1e-9), (
        f"neutralisation failed: |C| no-window {a_nw:.3e} < windowed "
        f"{a_w:.3e}")
    return hp_w, hp_nw, a_w, a_nw


def ablate_all(hp, dim, gt):
    """Applique les ablations de la tache 13 a un Hamiltonien donne."""
    base_mask, base_E, _, base_uni = ground_state_mask(hp, dim)
    out = []
    for name, drop in ABLATIONS:
        hp_ab = zero_hamiltonian_terms(hp, drop)
        mask, E, n_opt, uni = ground_state_mask(hp_ab, dim)
        out.append(dict(
            ablation=name,
            changed=float(np.mean(mask != base_mask)),
            uniform=bool(uni), n_optima=n_opt,
            f1=f1_from_masks(mask, gt),
            refined=float(np.mean(mask)),
            dE=float(E - base_E)))
    return out, bool(base_uni)


def main():
    p = argparse.ArgumentParser(
        description="V4 Task 18: do ZZ terms matter without the window?")
    from config import RESULTS_DIR, SCENARIOS, DNS_N

    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--re", nargs="+", type=int, default=[400])
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--n-snaps", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    np.random.seed(args.seed)

    t0 = time.time()
    print("=" * 88)
    print("  V4 Task 18: counterfactual — are ZZ terms inert WITHOUT the "
          "uncertainty window?")
    print(f"  N={args.N}  dim={args.dim}  mapper=v1 (deployed)  "
          f"snaps/cfg={args.n_snaps}")
    print("  'full' is a control: it must give exactly zero change in BOTH "
          "arms.")
    print("=" * 88)
    print()

    rows, cross_rows = [], []
    for sc in args.scenario:
        for re in args.re:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            pp = os.path.join(
                RESULTS_DIR,
                f"patches_{sc}_Re{re}_N{args.N}_dim{args.dim}.npz")
            if not (os.path.exists(dp) and os.path.exists(pp)):
                print(f"  SKIP {sc} Re={re}: missing input")
                continue
            dns = np.load(dp)
            pat = np.load(pp)
            vx = dns["vx"].astype(np.float64)
            vy = dns["vy"].astype(np.float64)
            Bx = dns["Bx"].astype(np.float64)
            By = dns["By"].astype(np.float64)
            l2 = pat["l2_errors"]
            thr = float(pat["l2_threshold"])
            sel = sorted(set(int(round(i)) for i in np.linspace(
                0, len(vx) - 1, args.n_snaps + 1)[1:]))
            for si in sel:
                hp_w, hp_nw, a_w, a_nw = prepare_both_arms(
                    vx[si], vy[si], Bx[si], By[si], args.N, args.dim, re)
                gt = np.asarray(l2[si] >= thr)
                masks = {}
                for arm, hp, amp in (("windowed", hp_w, a_w),
                                     ("no_window", hp_nw, a_nw)):
                    res, base_uni = ablate_all(hp, args.dim, gt)
                    masks[arm] = ground_state_mask(hp, args.dim)[0]
                    for r in res:
                        r.update(scenario=sc, re=re, snap=si, arm=arm,
                                 c_amplitude=amp, base_uniform=base_uni)
                        rows.append(r)

                # Effet PROPRE de la fenetre, mesure ENTRE les deux bras
                # sur le Hamiltonien complet. C'est la question que la
                # comparaison intra-bras ne peut pas poser : les ablations
                # comparent chaque bras a lui-meme.
                cross = float(np.mean(masks["windowed"] != masks["no_window"]))
                cross_rows.append(dict(scenario=sc, re=re, snap=si,
                                       changed=cross, a_w=a_w, a_nw=a_nw))
                print(f"  {sc:<18} Re={re} snap={si:<3} "
                      f"max|C| windowed={a_w:.3e}  no-window={a_nw:.3e}  "
                      f"| window flips {cross:.4f} of decisions")

    if not rows:
        raise SystemExit("no input; run the DNS/patch phases first.")

    print("\n  " + "=" * 84)
    print(f"  {'arm':<12}{'ablation':<18}{'changed':>9}{'uniform':>9}"
          f"{'refined':>9}{'F1':>8}{'n_optima':>10}")
    print("  " + "-" * 84)
    summary = {}
    for arm in ("windowed", "no_window"):
        for n, _ in ABLATIONS:
            rs = [r for r in rows if r["arm"] == arm and r["ablation"] == n]
            if not rs:
                continue
            ch = float(np.mean([r["changed"] for r in rs]))
            summary[(arm, n)] = ch
            print(f"  {arm:<12}{n:<18}{ch:>9.4f}"
                  f"{np.mean([r['uniform'] for r in rs]):>9.3f}"
                  f"{np.mean([r['refined'] for r in rs]):>9.3f}"
                  f"{np.mean([r['f1'] for r in rs]):>8.3f}"
                  f"{np.mean([r['n_optima'] for r in rs]):>10.1f}")
        print("  " + "-" * 84)

    for arm in ("windowed", "no_window"):
        ctrl = summary.get((arm, "full"))
        if ctrl is not None and ctrl != 0.0:
            print(f"  WARNING: control non-zero in arm {arm}: {ctrl:.6f}")

    zz_w = summary.get(("windowed", "no_ZZ"))
    zz_nw = summary.get(("no_window", "no_ZZ"))
    z4_nw = summary.get(("no_window", "no_ZZZZ"))
    print()
    print("  READING")
    print(f"    ZZ ablation, windowed pipeline : {zz_w:.6f} decisions "
          f"changed")
    print(f"    ZZ ablation, window neutralised: {zz_nw:.6f} decisions "
          f"changed")
    if zz_nw is not None and zz_nw > 0.0:
        print("    => the ZZ signal EXISTS and the window destroys it: the "
              "defect is in\n       the implementation, and is repairable. "
              "The inertness reported in T13 is\n       NOT a property of "
              "the Ising formulation itself.")
    else:
        print("    => ZZ remains inert even at full coupling strength: the "
              "inertness is a\n       property of the formulation at this "
              "size, not an artefact of the window.")
    if z4_nw is not None:
        print(f"    ZZZZ ablation, window neutralised: {z4_nw:.6f} "
              f"(the window does not touch ZZZZ)")

    # L'effet propre de la fenetre ne passe PAS par le couplage en tant que
    # couplage : |C| entre dans `C_scale` (mediane des |C|,|K| non nuls),
    # qui fixe l'echelle alpha_z du biais Z. Eteindre C reechelonne donc le
    # biais Z, et c'est par la que la decision bouge.
    if cross_rows:
        cx = float(np.mean([r["changed"] for r in cross_rows]))
        print()
        print(f"    Window's OWN effect (full Hamiltonian, windowed vs "
              f"neutralised): {cx:.6f} of decisions flip")
        print("    Note this is not the coupling acting as coupling: |C| "
              "feeds C_scale, the\n    median that sets the Z-bias "
              "amplitude alpha_z, so suppressing C rescales\n    the Z bias "
              "itself.")

    out = os.path.join(
        RESULTS_DIR,
        f"t18_window_counterfactual_N{args.N}_dim{args.dim}.npz")
    np.savez_compressed(
        out,
        scenario=np.array([r["scenario"] for r in rows]),
        re=np.array([r["re"] for r in rows]),
        snap=np.array([r["snap"] for r in rows]),
        arm=np.array([r["arm"] for r in rows]),
        ablation=np.array([r["ablation"] for r in rows]),
        changed=np.array([r["changed"] for r in rows]),
        uniform=np.array([r["uniform"] for r in rows]),
        n_optima=np.array([r["n_optima"] for r in rows]),
        f1=np.array([r["f1"] for r in rows]),
        refined=np.array([r["refined"] for r in rows]),
        dE=np.array([r["dE"] for r in rows]),
        c_amplitude=np.array([r["c_amplitude"] for r in rows]),
        seed=args.seed, git_hash=git_commit_hash(),
        cli_args=json.dumps(vars(args)),
        wall_s=time.time() - t0,
        cross_scenario=np.array([r["scenario"] for r in cross_rows]),
        cross_snap=np.array([r["snap"] for r in cross_rows]),
        cross_changed=np.array([r["changed"] for r in cross_rows]),
        cross_a_windowed=np.array([r["a_w"] for r in cross_rows]),
        cross_a_nowindow=np.array([r["a_nw"] for r in cross_rows]),
    )
    print(f"\n  saved: {os.path.basename(out)}  ({time.time() - t0:.0f}s)")
    print("\nV4 Task 18 complete.")


if __name__ == "__main__":
    main()
