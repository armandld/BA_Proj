"""D-45 : la porte `promising` de la phase 4 comparait deux predicteurs
CONSTANTS.

`exact_diagonalisation.analyze_snapshot` declare un snapshot `promising`
quand `f1_exact >= f1_classique`, et la docstring du module dit que ce sont
les seuls patchs qui passent en QAOA. Mesure sur les donnees reelles
(dim=2 — seule dimension executable, dim=4/8 demandent 32/128 qubits contre
un plafond de 20 ; Re=400, N=256, 4 scenarios canoniques, 40 snapshots) :

  decision exacte tout-a-1        40/40
  ligne de base classique tout-a-1 40/40
  exact_refine != classical_refine  0/40
  F1 egaux                         40/40  (jamais superieurs)
  promising avec `>=`              40/40
  promising avec le `>` du commentaire 0/40

Deux predicteurs constants rendent le MEME F1 par construction : la porte ne
peut ni retenir ni rejeter, elle porte zero bit. `promising` reste inchange
— quel operateur il doit porter est une decision humaine ouverte (D-46) —
mais la degenerescence est desormais annoncee.

Ces tests echouent sur la version d'avant : `degenerate_decision`,
`f1_tie` et `promising_informative` n'y existent pas.
"""
import glob
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exact_diagonalisation as ed
from config import TRAINED_THRESHOLD

RESULTS = os.path.join(_REPO_ROOT, "results")
DIM = 2                      # 8 qubits : la seule dimension sous le plafond
SCENARIOS = ["orszag_tang", "harris_tearing", "kelvin_helmholtz", "mhd_rotor"]


# ------------------------------------------------------------------
# 1. Le coeur du defaut, sur une entree synthetique qui SEPARE
# ------------------------------------------------------------------

def _fake_result(exact_h, exact_v, gt, score, thr):
    """Rejoue la fin d'`analyze_snapshot` sans diagonaliser quoi que ce
    soit : c'est le verdict qu'on teste, pas l'algebre lineaire."""
    cmp_ = ed.compare_decisions(exact_h, exact_v, gt, score, thr)
    exact_refine = cmp_["exact_refine"]
    classical_refine = cmp_["classical_refine"]
    return {
        "promising": cmp_["exact"]["f1"] >= cmp_["classical"]["f1"],
        "degenerate_decision": bool(exact_refine.all() or (~exact_refine).all()),
        "degenerate_classical": bool(
            classical_refine.all() or (~classical_refine).all()),
        "f1_tie": bool(cmp_["exact"]["f1"] == cmp_["classical"]["f1"]),
        "cmp": cmp_,
    }


def test_two_constant_predictors_tie_and_are_flagged():
    """Tout-raffiner contre tout-raffiner : F1 egaux par construction.

    L'ancienne version rendait `promising=True` sans rien d'autre. On
    EPINGLE ce True — la correction ne change pas le verdict, elle le
    qualifie — et on exige que la degenerescence soit annoncee.
    """
    T = np.ones((DIM, DIM), dtype=bool)
    gt = np.array([[True, False], [False, False]])
    score = np.full((DIM, DIM), 0.90)          # tres au-dessus du seuil

    r = _fake_result(T, T, gt, score, TRAINED_THRESHOLD)

    # l'ancien comportement, epingle
    assert bool(r["promising"]) is True
    assert r["cmp"]["exact"]["f1"] == r["cmp"]["classical"]["f1"]

    # ce que la correction ajoute
    assert r["degenerate_decision"] is True
    assert r["degenerate_classical"] is True
    assert r["f1_tie"] is True


def test_gate_cannot_reject_when_both_predictors_are_constant():
    """La porte ne porte aucun bit : elle rend le meme verdict quelle que
    soit la verite terrain. C'est cela qui la rend inutilisable."""
    T = np.ones((DIM, DIM), dtype=bool)
    score = np.full((DIM, DIM), 0.90)

    verdicts = set()
    for n_hard in range(DIM * DIM + 1):
        gt = np.zeros(DIM * DIM, dtype=bool)
        gt[:n_hard] = True
        r = _fake_result(T, T, gt.reshape(DIM, DIM), score, TRAINED_THRESHOLD)
        verdicts.add(bool(r["promising"]))
        assert r["f1_tie"] is True
        assert r["degenerate_decision"] is True

    # 5 verites terrain differentes, un seul verdict : zero bit.
    assert verdicts == {True}


def test_non_degenerate_case_is_not_flagged():
    """Le drapeau doit pouvoir valoir False, sinon il ne mesure rien.

    Decision exacte non constante ET differente du classique : c'est le cas
    que la phase 4 pretend selectionner.
    """
    exact_h = np.array([[True, False], [False, False]])
    exact_v = np.zeros((DIM, DIM), dtype=bool)
    gt = np.array([[True, False], [False, False]])
    # le classique rate le coin dur et en signale un autre
    score = np.array([[0.01, 0.90], [0.01, 0.01]])

    r = _fake_result(exact_h, exact_v, gt, score, TRAINED_THRESHOLD)

    assert r["degenerate_decision"] is False
    assert r["degenerate_classical"] is False
    assert r["f1_tie"] is False
    assert r["cmp"]["exact"]["f1"] > r["cmp"]["classical"]["f1"]
    assert bool(r["promising"]) is True    # ici, un vrai ecart


# ------------------------------------------------------------------
# 2. Les cles existent bel et bien dans le retour d'analyze_snapshot
# ------------------------------------------------------------------

def test_analyze_snapshot_publishes_the_flags():
    """Interroge le module, pas le texte du source."""
    sc = SCENARIOS[0]
    dns_path = os.path.join(RESULTS, f"dns_{sc}_Re400_N256.npz")
    patches_path = os.path.join(RESULTS, f"patches_{sc}_Re400_N256_dim{DIM}.npz")
    if not (os.path.exists(dns_path) and os.path.exists(patches_path)):
        pytest.skip("artefacts DNS absents")

    dns = np.load(dns_path)
    p = np.load(patches_path)
    si = 0
    res = ed.analyze_snapshot(
        dns["vx"][si].astype(float), dns["vy"][si].astype(float),
        dns["Bx"][si].astype(float), dns["By"][si].astype(float),
        dns["vx"].shape[1], DIM, int(dns.get("meta_Re", 400)),
        p["l2_errors"][si], p["is_hard"][si], float(p["l2_threshold"]),
    )

    for key in ("degenerate_decision", "degenerate_classical",
                "f1_tie", "promising_informative"):
        assert key in res, f"{key} absent du retour d'analyze_snapshot"

    # coherence interne du verdict compose
    assert res["promising_informative"] == (
        res["promising"] and not res["degenerate_decision"]
        and not res["f1_tie"])


# ------------------------------------------------------------------
# 3. Rejeu sur les donnees reelles — le nombre mesure, ecrit en clair
# ------------------------------------------------------------------

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_real_dns_first_snapshot_is_degenerate(scenario):
    """Sur le premier snapshot des 4 scenarios canoniques : decision exacte
    constante, F1 a egalite, `promising_informative` faux.

    Mesure du 2026-08-13 sur les 40 snapshots (10 par scenario) :
    40/40 degeneres, 40/40 a egalite, 0/40 informatifs. Ce test n'en
    rejoue qu'un par scenario ; le balayage complet est le test `slow`
    ci-dessous.
    """
    dns_path = os.path.join(RESULTS, f"dns_{scenario}_Re400_N256.npz")
    patches_path = os.path.join(
        RESULTS, f"patches_{scenario}_Re400_N256_dim{DIM}.npz")
    if not (os.path.exists(dns_path) and os.path.exists(patches_path)):
        pytest.skip(f"artefacts absents pour {scenario}")

    dns = np.load(dns_path)
    p = np.load(patches_path)
    si = 0
    res = ed.analyze_snapshot(
        dns["vx"][si].astype(float), dns["vy"][si].astype(float),
        dns["Bx"][si].astype(float), dns["By"][si].astype(float),
        dns["vx"].shape[1], DIM, int(dns.get("meta_Re", 400)),
        p["l2_errors"][si], p["is_hard"][si], float(p["l2_threshold"]),
    )

    assert res["degenerate_decision"] is True, (
        f"{scenario}: la decision exacte n'est plus constante — "
        "remesurer avant de toucher au seuil de ce test")
    assert res["f1_tie"] is True
    assert bool(res["promising"]) is True      # ancien comportement, epingle
    assert res["promising_informative"] is False


def test_real_dns_full_sweep_is_forty_over_forty():
    """Le balayage complet, avec le nombre mesure ecrit dans le test.

    Mesure a 25 s : sous le seuil du marqueur `slow` (« quelques minutes »),
    donc il tourne dans la suite standard — un test deselectionne par defaut
    finit par ne plus etre lu.

    Un balayage vide doit crier : on verifie le nombre de snapshots
    REELLEMENT traites, pas seulement les compteurs.
    """
    n = n_deg = n_tie = n_inf = 0
    for scenario in SCENARIOS:
        dns_path = os.path.join(RESULTS, f"dns_{scenario}_Re400_N256.npz")
        patches_path = os.path.join(
            RESULTS, f"patches_{scenario}_Re400_N256_dim{DIM}.npz")
        if not (os.path.exists(dns_path) and os.path.exists(patches_path)):
            continue
        dns = np.load(dns_path)
        p = np.load(patches_path)
        ns = len(dns["vx"])
        for si in range(0, ns, max(1, ns // 10)):
            res = ed.analyze_snapshot(
                dns["vx"][si].astype(float), dns["vy"][si].astype(float),
                dns["Bx"][si].astype(float), dns["By"][si].astype(float),
                dns["vx"].shape[1], DIM, int(dns.get("meta_Re", 400)),
                p["l2_errors"][si], p["is_hard"][si], float(p["l2_threshold"]),
            )
            n += 1
            n_deg += int(res["degenerate_decision"])
            n_tie += int(res["f1_tie"])
            n_inf += int(res["promising_informative"])

    assert n == 40, f"balayage attendu sur 40 snapshots, {n} traites"
    assert n_deg == 40, f"degeneres : 40 mesures le 2026-08-13, {n_deg} ici"
    assert n_tie == 40, f"egalites : 40 mesurees le 2026-08-13, {n_tie} ici"
    assert n_inf == 0, f"informatifs : 0 mesure le 2026-08-13, {n_inf} ici"


# ------------------------------------------------------------------
# 4. L'artefact porte les colonnes, sinon 100 % se relit comme un succes
# ------------------------------------------------------------------

def test_saved_artifact_carries_the_flags(tmp_path):
    fake = [{
        "ground_energy": -1.0, "gap": 0.5,
        "marginals": np.ones(2 * DIM * DIM),
        "decisions_h": np.ones((DIM, DIM), dtype=bool),
        "decisions_v": np.ones((DIM, DIM), dtype=bool),
        "gt_refine": np.zeros((DIM, DIM), dtype=bool),
        "promising": True, "degenerate_decision": True,
        "degenerate_classical": True, "f1_tie": True,
        "promising_informative": False,
        "comparison": {"exact": {"f1": 0.0}, "classical": {"f1": 0.0}},
    }]
    meta = {"scenario": "synthetic", "Re": 400, "N": 256,
            "n_patches": DIM, "n_qubits": 2 * DIM * DIM,
            "snap_indices": np.array([0]), "suffix": ""}

    path = ed.save_results(fake, meta, outdir=str(tmp_path))
    d = np.load(path)

    for key in ("degenerate_decision", "degenerate_classical",
                "f1_tie", "promising_informative"):
        assert key in d.files, f"{key} absent de l'artefact exact_diag_*"
    assert bool(d["promising"][0]) is True
    assert bool(d["degenerate_decision"][0]) is True
    assert bool(d["promising_informative"][0]) is False
