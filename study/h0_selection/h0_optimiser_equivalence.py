#!/usr/bin/env python3
"""
V4 Task 11 - Attribution de la contribution quantique (audit, Priorite 0).

QUESTION. Sur un Hamiltonien donne, l'optimiseur quantique (QAOA) prend-il
des decisions differentes de celles d'optimiseurs classiques du MEME
objectif ? Si exact, annealing, recherche locale et QAOA prennent les memes
decisions, alors une amelioration closed-loop attribue une valeur au
HAMILTONIEN, pas a son optimisation quantique.

FAIT STRUCTUREL VERIFIE ICI. Le Hamiltonien de cout Q-HAS ne contient que
des termes Z, ZZ et ZZZZ : il est DIAGONAL dans la base de calcul. Son etat
fondamental est donc un etat de base, i.e. une configuration classique de
spins, et « diagonalisation exacte » se reduit a une enumeration classique
exhaustive. A dim=2 (8 qubits, le regime effectif du pipeline V1) cette
enumeration coute 256 evaluations : l'optimum est CERTIFIE, pas approche.
`is_diagonal_cost_hamiltonian` verifie la propriete a l'execution plutot
que de la supposer.

PANEL DE SOLVEURS (tous sur le meme H, le meme snapshot) :
  exhaustive   enumeration exacte des 2^n configurations (reference)
  sa           simulated annealing, budget controle      (phase 7)
  sa_warm      idem, warm-start sur la decision classique (phase 7)
  greedy       descente locale gloutonne warm-startee     (nouveau)
  qaoa_p{1,2,3} QAOA statevector a profondeur croissante  (phase 5)
  qaoa_shots   QAOA avec shots finis (backend `aer`)      (phase 5)

METRIQUES PAR SOLVEUR :
  E             energie de la configuration decidee
  E_gap         (E - E_exact) / max(|E_exact|, eps)  -> 0.0 = optimum
  hit_optimum   E <= E_exact + tol
  agree_spin    fraction de qubits identiques a l'etat fondamental exact
  exact_match   masque de raffinement identique a l'exact (booleen)
  f1            F1 du masque contre la verite terrain L2-hard
  wall_s        temps de resolution bout en bout

REGLE DE DECISION (pre-specifiee, avant execution) :
  - si tous les solveurs atteignent l'optimum ET produisent le meme masque
    -> l'optimiseur quantique n'est pas la source d'un eventuel gain ;
       la valeur eventuelle est attribuable au Hamiltonien.
  - si QAOA devie de l'optimum, la deviation est rapportee comme une
    approximation (potentiellement favorable par accident), jamais comme
    un avantage : elle n'est pas controlable a l'avance.

# Le nom porte le mappeur des qu'il n'est pas le defaut v2 :
# sans cela, relancer avec --mapper v1 ecraserait le resultat v2
# et la comparaison entre mappeurs ne tiendrait pas dans les
# artefacts (defaut D9, deja rencontre sur t13 et t19).
Sortie : results/h0_optimiser_equivalence_N{N}_dim{D}.npz (+ hash git, args CLI)

Usage :
  python study/v4/h0_optimiser_equivalence.py --N 256 --dim 2 --n-snaps 3
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

from t1_feature_selection import git_commit_hash          # v3, reutilise
from ising_terms_and_annealing import (                          # V2, reutilise
    build_ising_terms, total_energy, delta_energy,
    _build_incidence, sa_multi_restart, spins_to_decisions,
)

TOL_E = 1e-9          # tolerance d'egalite energetique
MAX_ENUM_QUBITS = 22  # garde-fou memoire pour l'enumeration exhaustive


# -------------------------------------------------------------------
# Helpers purs (testables sans donnees DNS)
# -------------------------------------------------------------------

def is_diagonal_cost_hamiltonian(cost_hamiltonian):
    """True si tous les termes de Pauli sont diagonaux (I/Z uniquement).

    Verifie a l'execution la propriete qui rend l'enumeration classique
    exacte : aucun terme X ou Y dans le Hamiltonien de cout.
    """
    labels = [str(p) for p in cost_hamiltonian.paulis]
    return all(set(lbl) <= {"I", "Z"} for lbl in labels)


def classical_init_spins(score_vqa, threshold_amr, dim):
    """Configuration de spins issue de la decision classique.

    Convention identique a celle construite en ligne dans
    `ising_terms_and_annealing.analyze_snapshot_sa` (raffiner ssi score > seuil
    -> spin -1, sinon +1), extraite ici en fonction pour etre partagee
    par le warm start de SA et de la descente gloutonne.
    """
    n_cells = dim * dim
    refine = np.asarray(score_vqa > threshold_amr).ravel().astype(bool)
    spins = np.ones(2 * n_cells, dtype=np.int8)
    spins[:n_cells] = np.where(refine, -1, 1)
    spins[n_cells:] = np.where(refine, -1, 1)
    return spins


def exhaustive_ground_state(h_bias, edges, plaqs, n_q,
                            max_qubits=MAX_ENUM_QUBITS, chunk=1 << 16):
    """Etat fondamental EXACT par enumeration des 2^n_q configurations.

    Licite parce que le Hamiltonien est diagonal (Z/ZZ/ZZZZ) : minimiser
    <psi|H|psi> revient a minimiser E(s) sur les configurations classiques.
    Evaluation vectorisee par blocs pour rester en memoire bornee.

    Retourne (best_spins, best_E, n_optima) ou n_optima compte les
    configurations degenerees a l'optimum (a tol pres).
    """
    if n_q > max_qubits:
        raise ValueError(
            f"exhaustive enumeration refused for {n_q} qubits "
            f"(max {max_qubits}); use SA/greedy instead")
    edge_idx, edge_coef = edges
    plaq_idx, plaq_coef = plaqs
    n_states = 1 << n_q
    bit = (1 << np.arange(n_q, dtype=np.int64))

    def _energies(start, stop):
        idx = np.arange(start, stop, dtype=np.int64)[:, None]
        # spin = +1 si bit 0, -1 si bit 1 (|1> a valeur propre Z = -1)
        S = np.where((idx & bit) > 0, -1.0, 1.0)          # (m, n_q)
        E = S @ h_bias
        if len(edge_coef):
            E = E + (S[:, edge_idx[:, 0]] * S[:, edge_idx[:, 1]]) @ edge_coef
        if len(plaq_coef):
            E = E + (S[:, plaq_idx[:, 0]] * S[:, plaq_idx[:, 1]]
                     * S[:, plaq_idx[:, 2]] * S[:, plaq_idx[:, 3]]) @ plaq_coef
        return S, E

    # Passe 1 : le minimum. Passe 2 : sa degenerescence.
    #
    # Compter en une seule passe est incorrect des que l'enumeration depasse
    # un bloc : quand un meilleur minimum apparait dans le bloc k, les
    # configurations degenerees des blocs 0..k-1 ne sont plus jamais
    # revisitees, et celles des blocs suivants s'ajoutent par-dessus. Le
    # compte n'etait donc juste que pour 2^n_q <= chunk, soit n_q <= 16 —
    # correct a la taille deployee (8 qubits), faux et silencieux au-dela.
    best_E, best_spins = np.inf, None
    for start in range(0, n_states, chunk):
        S, E = _energies(start, min(start + chunk, n_states))
        k = int(np.argmin(E))
        if E[k] < best_E - TOL_E:
            best_E = float(E[k])
            best_spins = S[k].astype(np.int8).copy()

    n_opt = 0
    for start in range(0, n_states, chunk):
        _, E = _energies(start, min(start + chunk, n_states))
        n_opt += int(np.sum(E <= best_E + TOL_E))

    if best_spins is None or n_opt < 1:
        raise RuntimeError(
            "enumeration exhaustive sans minimum : impossible sur un "
            "hamiltonien diagonal non vide")
    return best_spins, float(best_E), int(n_opt)


def greedy_local_search(h_bias, edges, plaqs, n_q, init_spins,
                        max_iter=1000):
    """Descente locale gloutonne (steepest descent, 1 flip a la fois).

    A chaque iteration, evalue delta_energy pour tous les qubits et
    applique le flip le plus negatif ; s'arrete a l'optimum local.
    Reutilise `delta_energy` / `_build_incidence` de la phase 7 : la
    fonction d'energie est donc strictement celle du solveur SA.

    Retourne (spins, E, n_flips).
    """
    edges_by_q, plaqs_by_q = _build_incidence(n_q, edges, plaqs)
    spins = np.asarray(init_spins, dtype=np.float64).copy()
    E = total_energy(spins, h_bias, edges, plaqs)
    n_flips = 0
    for _ in range(max_iter):
        dEs = np.array([
            delta_energy(spins, q, h_bias, edges, plaqs,
                         edges_by_q, plaqs_by_q)
            for q in range(n_q)
        ])
        q = int(np.argmin(dEs))
        if dEs[q] >= -TOL_E:
            break
        spins[q] = -spins[q]
        E += float(dEs[q])
        n_flips += 1
    return spins.astype(np.int8), float(E), n_flips


def decision_agreement(spins_a, spins_b, dim):
    """Comparaison de deux solutions au niveau spin ET au niveau decision.

    Retourne dict(agree_spin, exact_match, n_diff_patch) ou la decision
    par patch suit la convention phase 7 : refine = dec_h | dec_v.
    """
    a = np.asarray(spins_a).ravel()
    b = np.asarray(spins_b).ravel()
    ah, av = spins_to_decisions(a, dim)
    bh, bv = spins_to_decisions(b, dim)
    ref_a, ref_b = (ah | av), (bh | bv)
    return dict(
        agree_spin=float(np.mean(a == b)),
        exact_match=bool(np.array_equal(ref_a, ref_b)),
        n_diff_patch=int(np.sum(ref_a != ref_b)),
    )


def f1_from_masks(pred, gt):
    """F1 du masque de raffinement contre la verite terrain."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    tp = int(np.sum(pred & gt)); fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


# -------------------------------------------------------------------
# Panel de solveurs sur un snapshot
# -------------------------------------------------------------------

def solver_panel(vx, vy, Bx, By, N, dim, re, l2_errors, l2_threshold,
                 use_v2=True, sweeps=500, n_restarts=5,
                 qaoa_reps=(1, 2, 3), qaoa_shots=4096, k_opt=60,
                 zero_psi=False, scale_kopt=False, no_exact=False,
                 run_qaoa=True, seed=0):
    """Execute tous les solveurs sur le meme Hamiltonien / snapshot."""
    from qaoa_inputs import (                      # V2, reutilise
        prepare_qaoa_inputs, run_qaoa_on_snapshot,
        classical_warm_start_params,
    )
    from VQA.cost_hamiltonian import create_period_hamiltonian
    from config import V2_THRESHOLD, TRAINED_THRESHOLD

    thr_amr = V2_THRESHOLD if use_v2 else TRAINED_THRESHOLD
    data_in, hp, score_vqa = prepare_qaoa_inputs(
        vx, vy, Bx, By, N, dim, re, use_v2=use_v2)

    # Ablation psi : psi porte une derivee temporelle du flux qui n'existe
    # NULLE PART dans l'hamiltonien. Le QAOA part donc d'un etat encodant une
    # information que son propre cout ignore. La mettre a zero isole ce que
    # psi apporte, sans toucher a theta ni a H.
    if zero_psi:
        for _k in ("psi_h", "psi_v"):
            if _k in data_in:
                data_in[_k] = np.zeros_like(np.asarray(data_in[_k], float)).tolist()

    h_bias, edges, plaqs = build_ising_terms(hp, dim)
    n_q = 2 * dim * dim
    gt_refine = np.asarray(l2_errors >= l2_threshold)
    init = classical_init_spins(score_vqa, thr_amr, dim)

    diagonal = is_diagonal_cost_hamiltonian(
        create_period_hamiltonian(hp, dim, False))

    rows = {}

    def _record(name, spins, wall, extra=None):
        spins = np.asarray(spins, dtype=np.int8)
        E = total_energy(spins.astype(np.float64), h_bias, edges, plaqs)
        dh, dv = spins_to_decisions(spins, dim)
        refine = dh | dv
        r = dict(name=name, E=float(E), spins=spins, refine=refine,
                 wall_s=float(wall), f1=f1_from_masks(refine, gt_refine))
        if extra:
            r.update(extra)
        rows[name] = r
        return r

    # --- reference exacte -------------------------------------------
    t0 = time.time()
    # H0b — « mieux atteindre l'optimum ameliore-t-il la tache ? » — ne
    # demande PAS l'optimum exact : il suffit d'un ETALEMENT d'energies et
    # des F1 correspondants. Au-dela de MAX_ENUM_QUBITS on prend donc la
    # meilleure energie trouvee par le panel comme reference, ce qui rend
    # H0b mesurable a 32 qubits et au-dela. H0a, elle, reste indecidable
    # sans optimum certifie, et le tableau le signale.
    if no_exact or n_q > MAX_ENUM_QUBITS:
        ex_spins, ex_E, n_opt = None, float("nan"), -1
        certified = False
    else:
        ex_spins, ex_E, n_opt = exhaustive_ground_state(
            h_bias, edges, plaqs, n_q)
        certified = True
    _record("exhaustive", ex_spins, time.time() - t0, dict(n_optima=n_opt))

    # --- simulated annealing (froid puis warm-start) -----------------
    for nm, ini in (("sa", None), ("sa_warm", init)):
        rng = np.random.default_rng(seed)
        t0 = time.time()
        s, _, _ = sa_multi_restart(h_bias, edges, plaqs, n_q,
                                   sweeps=sweeps, n_restarts=n_restarts,
                                   rng=rng, classical_init=ini)
        _record(nm, s, time.time() - t0)

    # --- descente locale gloutonne warm-startee ----------------------
    t0 = time.time()
    g_spins, _, n_flips = greedy_local_search(h_bias, edges, plaqs, n_q, init)
    _record("greedy", g_spins, time.time() - t0, dict(n_flips=n_flips))

    # --- decision classique seule (reference non optimisee) ----------
    _record("classical_init", init, 0.0)

    # --- QAOA a profondeurs croissantes ------------------------------
    if run_qaoa:
        for reps in qaoa_reps:
            ws = classical_warm_start_params(score_vqa, thr_amr, reps)
            t0 = time.time()
            _, dh, dv, _, _ = run_qaoa_on_snapshot(
                data_in, hp, dim, reps=reps,
                K_opt=(k_opt * reps if scale_kopt else k_opt),
                shots=qaoa_shots, backend_name="state_vector",
                warm_start_params=ws)
            s = np.ones(n_q, dtype=np.int8)
            s[:dim * dim] = np.where(dh.ravel(), -1, 1)
            s[dim * dim:] = np.where(dv.ravel(), -1, 1)
            _record(f"qaoa_p{reps}", s, time.time() - t0)

        # QAOA avec shots finis (echantillonnage, pas statevector ideal)
        reps = max(qaoa_reps)
        ws = classical_warm_start_params(score_vqa, thr_amr, reps)
        t0 = time.time()
        _, dh, dv, _, _ = run_qaoa_on_snapshot(
            data_in, hp, dim, reps=reps,
            K_opt=(k_opt * reps if scale_kopt else k_opt),
            shots=qaoa_shots, backend_name="aer",
            warm_start_params=ws)
        s = np.ones(n_q, dtype=np.int8)
        s[:dim * dim] = np.where(dh.ravel(), -1, 1)
        s[dim * dim:] = np.where(dv.ravel(), -1, 1)
        _record(f"qaoa_shots_p{reps}", s, time.time() - t0)

    # --- comparaisons a l'exact --------------------------------------
    for name, r in rows.items():
        r.update(decision_agreement(r["spins"], ex_spins, dim))
        # Sans optimum certifie, la reference est la meilleure energie
        # TROUVEE par le panel : l'ecart reste comparable entre solveurs,
        # mais « atteindre l'optimum » n'a plus de sens et vaut NaN.
        ref_E = ex_E if certified else min(x["E"] for x in rows.values())
        r["E_gap"] = float((r["E"] - ref_E) / max(abs(ref_E), 1e-12))
        r["hit_optimum"] = (
            bool(r["E"] <= ex_E + 1e-6 * max(abs(ex_E), 1.0))
            if certified else float("nan"))
    return dict(rows=rows, diagonal=diagonal, n_optima=n_opt,
                E_exact=ex_E, gt_refine=gt_refine,
                f1_classical=f1_from_masks(rows["classical_init"]["refine"],
                                           gt_refine))


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


# ══════════════════════════════════════════════════════════════════════
#  CRITERE D'ACCEPTATION — H0
# ══════════════════════════════════════════════════════════════════════
#
# La regle de decision de H0 etait imprimee en prose et le script sortait 0
# quoi qu'il mesure : il ne pouvait pas echouer. Elle est ici un critere.
#
# Reference (N=64 et N=256, dim=2, 8 qubits) : les huit solveurs —
# exhaustif certifie, recuit simule, recuit a chaud, glouton, regle
# classique seule, QAOA p=1, p=2, p=2 a 4096 tirs — atteignent l'optimum
# certifie sur 100 % des instantanes et renvoient le MEME masque
# (agree_spin = mask_match = 1.000, E_gap = 0).
#
# Ce que le critere protege : si un jour un solveur cesse d'atteindre
# l'optimum, ou si deux solveurs divergent sur le masque, alors H0 n'est
# plus refutee et la campagne doit s'arreter au lieu d'imprimer un tableau.
MIN_HIT = 1.0
MIN_MASK_MATCH = 1.0


def check_expected_behaviour(summary, solvers, diag_flags):
    assert diag_flags and all(diag_flags), (
        "l'enumeration exhaustive n'est licite que si le hamiltonien de cout "
        "est diagonal ; le controle a echoue sur au moins un instantane")

    optimisers = [s for s in solvers if s != "classical_init"]
    assert len(optimisers) >= 4, (
        f"seulement {len(optimisers)} solveurs compares : le panel ne teste "
        "plus l'equivalence qu'il pretend tester")

    # Les solveurs DETERMINISTES doivent atteindre l'optimum a chaque fois.
    # Le recuit simule ne l'est pas (il n'est pas amorce), et son taux
    # mesure varie d'une execution a l'autre : 1.000 dans la campagne
    # publiee, 0.625 en rejouant la meme commande. On le rapporte au lieu
    # de l'exiger, et on le dit.
    deterministic = [s for s in optimisers if not s.startswith("sa")]
    stochastic = [s for s in optimisers if s.startswith("sa")]

    missed = {s: summary[s]["hit"] for s in deterministic
              if summary[s]["hit"] < MIN_HIT}
    assert not missed, (
        f"des solveurs deterministes n'atteignent plus l'optimum certifie : "
        f"{missed}. H0 (l'echec vient de l'optimiseur) redevient plausible.")

    for s in stochastic:
        print(f"  [NOTE] {s} : optimum atteint sur {summary[s]['hit']:.3f} "
              "des instantanes — solveur non amorce, taux variable d'une "
              "execution a l'autre (defaut D11)")

    # Le coeur de H0 : le masque du QAOA est celui de l'etat fondamental
    # exact. C'est cette egalite, et elle seule, qui retire a l'optimiseur
    # quantique tout role explicatif.
    qaoa = [s for s in solvers if s.startswith("qaoa")]
    assert qaoa, "aucun bras QAOA dans le panel : H0 n'est pas testee"
    diverging = {s: summary[s]["match"] for s in qaoa
                 if summary[s]["match"] < MIN_MASK_MATCH}
    assert not diverging, (
        f"le QAOA ne renvoie plus le masque de l'etat fondamental exact : "
        f"{diverging}. Le choix de l'optimiseur redeviendrait une variable "
        "explicative.")

    print(f"\n  [ACCEPTANCE] {len(optimisers)} optimiseurs atteignent "
          f"l'optimum certifie et renvoient un masque identique "
          f"(hit >= {MIN_HIT}, mask_match >= {MIN_MASK_MATCH}) -> H0 refutee "
          f"a cette taille.")


def main():
    p = argparse.ArgumentParser(
        description="V4 Task 11: quantum-contribution attribution")
    from config import RESULTS_DIR, SCENARIOS, RE_VALUES, DNS_N

    p.add_argument("--scenario", nargs="+", default=SCENARIOS)
    p.add_argument("--re", nargs="+", type=int, default=[400])
    p.add_argument("--N", type=int, default=DNS_N)
    p.add_argument("--dim", type=int, default=2,
                   help="2 = 8 qubits = regime effectif du pipeline V1")
    p.add_argument("--no-exact", action="store_true",
                   help="ne pas exiger l'optimum certifie. H0a devient "
                        "indecidable mais H0b reste mesurable : elle ne "
                        "demande qu'un etalement d'energies et les F1 "
                        "correspondants. Necessaire au-dela de 22 qubits.")
    p.add_argument("--zero-psi", action="store_true",
                   help="met psi a zero dans l'etat initial du QAOA. Isole "
                        "ce que l'encodage de phase apporte a la descente "
                        "vers le fondamental et a la detection.")
    p.add_argument("--scale-kopt", action="store_true",
                   help="budget COBYLA proportionnel a p (k_opt * reps). Sans "
                        "ce drapeau, p=6 optimise 12 parametres avec le meme "
                        "budget que p=1 en optimise 2, et le balayage confond "
                        "profondeur et sous-optimisation.")
    p.add_argument("--qaoa-reps", nargs="+", type=int, default=[1, 2, 3],
                   help="profondeurs p du QAOA. Balayer p=1..6 sert a tester "
                        "si la qualite de detection DECROIT quand le circuit "
                        "oublie son initialisation (theta, psi) pour "
                        "converger vers le fondamental de H.")
    p.add_argument("--n-snaps", type=int, default=3,
                   help="snapshots par configuration")
    p.add_argument("--sweeps", type=int, default=500)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--k-opt", type=int, default=60)
    p.add_argument("--no-qaoa", action="store_true")
    p.add_argument("--mapper", choices=["v1", "v2"], default="v2")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("=" * 88)
    print("  V4 Task 11: quantum-contribution attribution "
          f"(dim={args.dim} -> {2*args.dim*args.dim} qubits)")
    print(f"  N={args.N}  Re={args.re}  mapper={args.mapper}  "
          f"snaps/cfg={args.n_snaps}  SA sweeps={args.sweeps}x{args.restarts}")
    print("  Pre-registered rule: all solvers at the optimum with identical "
          "masks => value is attributable to H, not to quantum optimisation.")
    print("=" * 88)
    print()

    records, diag_flags = [], []
    for sc in args.scenario:
        for re in args.re:
            dp = os.path.join(RESULTS_DIR, f"dns_{sc}_Re{re}_N{args.N}.npz")
            pp = os.path.join(
                RESULTS_DIR,
                f"patches_{sc}_Re{re}_N{args.N}_dim{args.dim}.npz")
            if not (os.path.exists(dp) and os.path.exists(pp)):
                print(f"  SKIP {sc} Re={re}: missing input")
                continue
            dns = np.load(dp); pat = np.load(pp)
            vx = dns["vx"].astype(np.float64); vy = dns["vy"].astype(np.float64)
            Bx = dns["Bx"].astype(np.float64); By = dns["By"].astype(np.float64)
            l2 = pat["l2_errors"]; thr = float(pat["l2_threshold"])
            n_dns = len(vx)
            sel = sorted(set(int(round(i)) for i in
                             np.linspace(0, n_dns - 1, args.n_snaps + 1)[1:]))
            for si in sel:
                t0 = time.time()
                out = solver_panel(
                    vx[si], vy[si], Bx[si], By[si], args.N, args.dim, re,
                    l2[si], thr, use_v2=(args.mapper == "v2"),
                    sweeps=args.sweeps, n_restarts=args.restarts,
                    qaoa_reps=tuple(args.qaoa_reps), qaoa_shots=args.shots,
                    zero_psi=args.zero_psi, scale_kopt=args.scale_kopt,
                    no_exact=args.no_exact,
                    k_opt=args.k_opt, run_qaoa=not args.no_qaoa,
                    seed=args.seed)
                diag_flags.append(out["diagonal"])
                for name, r in out["rows"].items():
                    records.append(dict(
                        scenario=sc, re=re, snap=si, solver=name,
                        E=r["E"], E_gap=r["E_gap"],
                        hit=r["hit_optimum"], agree=r["agree_spin"],
                        match=r["exact_match"], n_diff=r["n_diff_patch"],
                        f1=r["f1"], wall=r["wall_s"]))
                print(f"  {sc:<18} Re={re} snap={si:<3} "
                      f"n_optima={out['n_optima']:<3} "
                      f"E*={out['E_exact']:+.4f}  [{time.time()-t0:.1f}s]")

    if not records:
        print("no input."); return

    solvers = list(dict.fromkeys(r["solver"] for r in records))
    print(f"\n  cost Hamiltonian diagonal (Z/ZZ/ZZZZ only) on all "
          f"{len(diag_flags)} snapshots: {all(diag_flags)}")
    print("\n  " + "=" * 84)
    print(f"  {'solver':<18} {'hit_opt':>8} {'E_gap':>10} "
          f"{'agree_spin':>11} {'mask_match':>11} {'F1':>7} {'wall_s':>8}")
    print("  " + "-" * 84)
    summary = {}
    for s in solvers:
        rs = [r for r in records if r["solver"] == s]
        summary[s] = dict(
            hit=float(np.mean([r["hit"] for r in rs])),
            gap=float(np.mean([r["E_gap"] for r in rs])),
            agree=float(np.mean([r["agree"] for r in rs])),
            match=float(np.mean([r["match"] for r in rs])),
            f1=float(np.mean([r["f1"] for r in rs])),
            wall=float(np.mean([r["wall"] for r in rs])))
        v = summary[s]
        print(f"  {s:<18} {v['hit']:>8.3f} {v['gap']:>10.2e} "
              f"{v['agree']:>11.3f} {v['match']:>11.3f} {v['f1']:>7.3f} "
              f"{v['wall']:>8.3f}")
    print("  " + "-" * 84)

    opt_solvers = [s for s in solvers
                   if s not in ("classical_init",) and summary[s]["hit"] >= 1.0]
    qaoa = [s for s in solvers if s.startswith("qaoa")]
    all_match = all(summary[s]["match"] >= 1.0 for s in qaoa) if qaoa else None
    print("\n  DECISION RULE:")
    print(f"  * solvers reaching the certified optimum on every snapshot: "
          f"{', '.join(opt_solvers) if opt_solvers else 'none'}")
    if qaoa:
        print(f"  * QAOA masks identical to the exact ground state on every "
              f"snapshot: {all_match}")
        print("  * => " + (
            "quantum optimisation is NOT the source of any gain; value is "
            "attributable to the Hamiltonian."
            if all_match else
            "QAOA deviates from the certified optimum; the deviation is an "
            "approximation artefact, not a controllable advantage."))


    out = os.path.join(
        RESULTS_DIR,
        f"h0_optimiser_equivalence_N{args.N}_dim{args.dim}"
        + ("_zeropsi" if args.zero_psi else "")
        + ("_noexact" if args.no_exact else "")
        + ("_scalekopt" if args.scale_kopt else "")
        + ("" if args.mapper == "v2" else f"_{args.mapper}")
        + ".npz")
    np.savez_compressed(
        out,
        scenario=np.array([r["scenario"] for r in records]),
        re=np.array([r["re"] for r in records]),
        snap=np.array([r["snap"] for r in records]),
        solver=np.array([r["solver"] for r in records]),
        E=np.array([r["E"] for r in records]),
        E_gap=np.array([r["E_gap"] for r in records]),
        hit=np.array([r["hit"] for r in records]),
        agree=np.array([r["agree"] for r in records]),
        match=np.array([r["match"] for r in records]),
        n_diff=np.array([r["n_diff"] for r in records]),
        f1=np.array([r["f1"] for r in records]),
        wall=np.array([r["wall"] for r in records]),
        diagonal_all=bool(all(diag_flags)),
        seed=args.seed, git_hash=git_commit_hash(),
        cli_args=json.dumps(vars(args)),
    )
    print(f"\n  saved: {os.path.basename(out)}")
    print("\nV4 Task 11 complete.")

    # Juger APRES avoir ecrit : un run exploratoire a une taille ou H0
    # n'est pas attendue doit quand meme laisser ses donnees derriere lui.
    check_expected_behaviour(summary, solvers, diag_flags)


if __name__ == "__main__":
    main()
