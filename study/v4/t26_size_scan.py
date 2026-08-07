#!/usr/bin/env python3
"""
V4 Tache 26 - L'inertie des couplages tient-elle quand la taille augmente ?

LA QUESTION, ET POURQUOI C'EST LA SEULE QUI RESTE OUVERTE
---------------------------------------------------------
T13 montre qu'ablater TOUS les ZZ et TOUS les ZZZZ change EXACTEMENT 0
decision. T18 montre que reparer la fenetre qui les supprimait n'y change
rien. Les deux resultats sont deterministes et exacts.

Mais ils sont mesures a `dim = 2`, soit `2*dim^2 = 8` qubits — precisement
le regime ou l'etat fondamental exact est un masque UNIFORME sur 100 % des
instantanes. L'objection evidente d'un rapporteur est donc : « a 8 qubits,
evidemment qu'il ne se passe rien ». Elle est juste, et c'est la faiblesse
centrale de toute l'etude.

Cette tache y repond en balayant la taille : 8 → 32 → 128 qubits.

  dim = 2  ->   8 qubits  (regime deploye)
  dim = 4  ->  32 qubits
  dim = 8  -> 128 qubits

Deux issues, toutes deux exploitables :

  - l'inertie PERSISTE : on ne ferme plus un cas, on ferme la famille de
    mappings sur toute la plage tractable ;
  - l'inertie CASSE a une taille donnee : on a la frontiere, c'est-a-dire
    l'endroit ou une structure combinatoire apparait et ou le quantique
    aurait quelque chose a resoudre.

LE PROBLEME DE SOLVEUR, ET COMMENT IL EST TRAITE HONNETEMENT
------------------------------------------------------------
`exhaustive_ground_state` refuse au-dela de 22 qubits (2^22 etats). On ne
peut donc pas obtenir l'etat fondamental EXACT a dim >= 4.

On utilise a la place la descente gloutonne demarree a chaud depuis la
decision classique — le meme point de depart que le pipeline deploye. Ce
n'est PAS l'optimum exact en general, et l'appeler ainsi serait exactement
le motif que cette campagne traque.

La parade : a `dim = 2`, ou l'exhaustif est possible, on VERIFIE que le
glouton rend la meme decision. Le taux d'accord est publie. Le proxy n'est
utilise aux grandes tailles qu'apres avoir ete valide la ou on peut.

Et la question posee au proxy est de toute facon la bonne : « ablater les
couplages change-t-il ce que le solveur DEPLOYE decide ? » Le pipeline
n'utilise pas l'exhaustif non plus.

Sortie : results/t26_size_scan_N{N}.npz
Usage :
  python study/v4/t26_size_scan.py --dims 2 4 8 --n-snaps 3
"""
import argparse, json, os, sys, time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "v3"))
sys.path.insert(0, _HERE)

import provenance
from phase7_sa_baseline import build_ising_terms, spins_to_decisions
from t11_solver_attribution import (classical_init_spins,
                                    exhaustive_ground_state,
                                    greedy_local_search)
from t11b_qaoa_displacement import mask_uniformity
from t13_term_ablation import zero_hamiltonian_terms
from phase5_qaoa_eval import prepare_qaoa_inputs

# Familles ablatees. On teste separement chaque famille de couplage et les
# deux ensemble ; `full` est le controle et doit rendre 0.
ABLATIONS = [("full", ()), ("no_ZZ", ("ZZ",)),
             ("no_ZZZZ", ("ZZZZ",)), ("Z_only", ("ZZ", "ZZZZ"))]

EXHAUSTIVE_MAX_Q = 22


def decide(hp, dim, use_exhaustive, init_spins):
    """(masque de decision, uniformite, methode) pour un jeu de parametres.

    `use_exhaustive` n'est vrai que sous la limite d'enumeration. Au-dela on
    prend la descente gloutonne a chaud, et la methode est RENDUE pour que
    l'artefact dise laquelle a servi plutot que de laisser deviner.
    """
    h, e, pq = build_ising_terms(hp, dim)
    n_q = 2 * dim * dim
    if use_exhaustive:
        gs, _, _ = exhaustive_ground_state(h, e, pq, n_q)
        method = "exhaustive"
    else:
        gs, _, _ = greedy_local_search(h, e, pq, n_q, init_spins)
        method = "greedy_warm"
    dh, dv = spins_to_decisions(np.asarray(gs), dim)
    return (dh | dv), mask_uniformity(gs), method, np.asarray(gs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    from config import (RESULTS_DIR, SCENARIOS, DNS_N,
                        TRAINED_THRESHOLD, V2_THRESHOLD)

    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--re", nargs="+", type=int, default=[400])
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--dims", nargs="+", type=int, default=[2, 4, 8])
    p.add_argument("--n-snaps", type=int, default=3)
    p.add_argument("--mapper", choices=["v1", "v2"], default="v1")
    p.add_argument("--force-greedy", action="store_true",
                   help="utiliser le glouton MEME quand l'exhaustif est "
                        "possible. CONTROLE DECISIF : si le glouton rend "
                        "des changements non nuls la ou l'exhaustif rend 0, "
                        "c'est le proxy qui les fabrique et le scan en "
                        "taille ne mesure rien.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    prov = provenance.start()
    t0 = time.time()

    print("=" * 88)
    print(f"  V4 Task 26: does coupling inertness survive a size scan?")
    print(f"  N={args.N}  dims={args.dims}  mapper={args.mapper}")
    print(f"  exhaustive ground state only below {EXHAUSTIVE_MAX_Q} qubits; "
          f"above it, warm-started greedy (validated at dim=2)")
    print("=" * 88, flush=True)

    rows = []
    for dim in args.dims:
        n_q = 2 * dim * dim
        exact_ok = (n_q <= EXHAUSTIVE_MAX_Q) and not args.force_greedy
        print(f"\n--- dim={dim}  ({n_q} qubits, "
              f"{'exhaustive' if exact_ok else 'greedy warm start'}) ---",
              flush=True)
        for sc in args.scenario:
            for re in args.re:
                dp = os.path.join(RESULTS_DIR,
                                  f"dns_{sc}_Re{re}_N{args.N}.npz")
                pp = os.path.join(
                    RESULTS_DIR,
                    f"patches_{sc}_Re{re}_N{args.N}_dim{dim}.npz")
                if not (os.path.exists(dp) and os.path.exists(pp)):
                    print(f"  SKIP {sc} dim={dim}: missing input", flush=True)
                    continue
                dns = np.load(dp); pat = np.load(pp)
                vx = dns["vx"].astype(np.float64)
                vy = dns["vy"].astype(np.float64)
                Bx = dns["Bx"].astype(np.float64)
                By = dns["By"].astype(np.float64)
                l2 = pat["l2_errors"]; thr = float(pat["l2_threshold"])
                sel = sorted(set(int(round(i)) for i in np.linspace(
                    0, len(vx) - 1, args.n_snaps + 1)[1:]))
                for si in sel:
                    use_v2 = (args.mapper == "v2")
                    thr_amr = V2_THRESHOLD if use_v2 else TRAINED_THRESHOLD
                    _, hp, score = prepare_qaoa_inputs(
                        vx[si], vy[si], Bx[si], By[si], args.N, dim, re,
                        use_v2=use_v2)
                    init = classical_init_spins(score, thr_amr, dim)

                    base, base_uni, method, base_spins = decide(
                        hp, dim, exact_ok, init)

                    # validation du proxy la ou les deux sont possibles
                    agree = None
                    if exact_ok:
                        g, _, _, _ = decide(hp, dim, False, init)
                        agree = float(np.mean(g == base))

                    for name, drop in ABLATIONS:
                        hp2 = zero_hamiltonian_terms(hp, drop)
                        m, uni, _, _ = decide(hp2, dim, exact_ok, init)
                        changed = float(np.mean(m != base))
                        rows.append(dict(
                            dim=dim, n_qubits=n_q, scenario=sc, re=re,
                            snap=si, ablation=name, changed=changed,
                            uniform=float(uni), base_uniform=float(base_uni),
                            method=method,
                            greedy_agrees_with_exhaustive=agree))
                    print(f"    {sc:18s} snap {si:3d}  "
                          + "  ".join(
                              f"{r['ablation']}={r['changed']:.4f}"
                              for r in rows[-len(ABLATIONS):])
                          + f"   uniform={base_uni:.2f}", flush=True)

    if not rows:
        raise SystemExit("no input for any requested dim")

    print("\n" + "=" * 88)
    print("  SYNTHESIS — decisions changed by ablating the coupling families")
    print("=" * 88)
    print(f"  {'dim':>4s} {'qubits':>7s} {'method':>12s} "
          f"{'no_ZZ':>9s} {'no_ZZZZ':>9s} {'Z_only':>9s} "
          f"{'uniform':>8s} {'proxy ok':>9s}")
    summary = []
    for dim in args.dims:
        sub = [r for r in rows if r["dim"] == dim]
        if not sub:
            continue
        g = lambda a: float(np.mean([r["changed"] for r in sub
                                     if r["ablation"] == a]))
        uni = float(np.mean([r["base_uniform"] for r in sub]))
        ag = [r["greedy_agrees_with_exhaustive"] for r in sub
              if r["greedy_agrees_with_exhaustive"] is not None]
        agree = float(np.mean(ag)) if ag else None
        meth = sub[0]["method"]
        print(f"  {dim:4d} {sub[0]['n_qubits']:7d} {meth:>12s} "
              f"{g('no_ZZ'):9.4f} {g('no_ZZZZ'):9.4f} {g('Z_only'):9.4f} "
              f"{uni:8.2f} " + (f"{agree:9.4f}" if agree is not None
                                else f"{'—':>9s}"))
        summary.append(dict(dim=dim, n_qubits=sub[0]["n_qubits"],
                            method=meth, no_ZZ=g("no_ZZ"),
                            no_ZZZZ=g("no_ZZZZ"), Z_only=g("Z_only"),
                            full=g("full"), uniform=uni,
                            greedy_agreement=agree, n=len(sub)))

    broke = [s for s in summary if s["Z_only"] > 0]
    print()
    if broke:
        print(f"  INERTNESS BREAKS at dim="
              f"{', '.join(str(s['dim']) for s in broke)} — the couplings "
              f"start changing decisions there.")
        print("  That is a FRONTIER, not a closed door: it locates the size "
              "at which the\n  formulation acquires combinatorial structure.")
    else:
        print("  Inertness HOLDS at every size scanned. The couplings change "
              "no decision\n  from 8 to "
              f"{max(s['n_qubits'] for s in summary)} qubits.")
    ctrl = [s["full"] for s in summary]
    if any(c != 0 for c in ctrl):
        print(f"  WARNING: the `full` control is non-zero somewhere "
              f"({ctrl}) — the comparison itself is unsound and nothing "
              f"above should be read.")

    out = dict(rows=rows, summary=summary, cli_args=vars(args))
    out.update(provenance.finish(prov))
    out["wall_s"] = time.time() - t0
    # Le mode de controle DOIT porter un nom distinct : sans cela il
    # ecrase le scan qu'il sert a valider, et le defaut D9 recommence dans
    # la tache ecrite pour trancher la question centrale de l'etude.
    ctrl = "_forcegreedy" if args.force_greedy else ""
    op = os.path.join(
        RESULTS_DIR,
        f"t26_size_scan_N{args.N}{ctrl}_{args.mapper}.json")
    json.dump(out, open(op, "w"), indent=1, default=float)
    print(f"\n  saved: {os.path.basename(op)} ({time.time() - t0:.0f}s)")
    print("\nV4 Task 26 complete.")


if __name__ == "__main__":
    main()
