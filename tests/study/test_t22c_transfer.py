"""Tests V4 T22c : synthese du test de transfert.

L'enjeu est de ne PAS conclure a un effet de transfert quand l'ecart tient
dans le bruit d'echantillonnage de D11. Les tests portent donc surtout sur
le critere de separabilite et sur la propagation de l'incertitude.
"""
import json
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

_HERE = os.path.dirname(os.path.abspath(__file__))


def _study_file(name):
    """Chemin d'un module de study/ quel que soit son dossier d'hypothese."""
    for _d in ("pipeline", "h0_selection", "h1_solver", "h2b_prediction",
               "h3_representation", "h4_transfer", "closed_loop", "common"):
        _c = os.path.join(_REPO_ROOT, "study", _d, name)
        if os.path.exists(_c):
            return _c
    raise FileNotFoundError(name)

from h4_transfer_summary import analyse, load, ratio_sd


def _rec(qc, qu, cc, cu, sqc=0.0, squ=0.0, n=5, qp=0.9, cp=0.6,
         unseen_runs=None):
    runs = unseen_runs if unseen_runs is not None else [
        {"phys_score": qu, "patch_ratio": qp} for _ in range(n)]
    return {"fold": "x", "underpowered": False, "raw": {
        "classical_reference_source": "budget-matched",
        "arms": {
            "qhas": {"n_runs": n,
                     "canonical": {"phys_score": qc, "patch_ratio": qp},
                     "unseen": {"phys_score": qu, "patch_ratio": qp},
                     "canonical_phys_sd": sqc, "unseen_phys_sd": squ,
                     "unseen_runs": runs},
            "classical": {"n_runs": 2,
                          "canonical": {"phys_score": cc, "patch_ratio": cp},
                          "unseen": {"phys_score": cu, "patch_ratio": cp}},
        }}}


def test_ratio_sd_combines_both_terms():
    """Les deux moyennes sont bruitees ; ignorer l'une sous-estime."""
    sd = ratio_sd(2.0, 0.2, 4.0, 0.4)
    only_num = ratio_sd(2.0, 0.2, 4.0, 0.0)
    assert sd > only_num > 0


def test_ratio_sd_is_zero_without_noise():
    assert ratio_sd(2.0, 0.0, 4.0, 0.0) == pytest.approx(0.0)


def test_large_spread_makes_the_difference_inseparable():
    """Degradations 0.8 vs 1.2 mais un ecart-type de 0.5 : rien n'est
    etabli. C'est exactement le cas de la passe a un seul tirage."""
    r = analyse(_rec(qc=1.0, qu=0.8, cc=1.0, cu=1.2, sqc=0.4, squ=0.4))
    assert r["separable"] is False
    assert r["separation_z"] < 2.0


def test_small_spread_makes_the_difference_separable():
    r = analyse(_rec(qc=1.0, qu=0.8, cc=1.0, cu=1.2, sqc=0.01, squ=0.01))
    assert r["separable"] is True
    assert r["separation_z"] > 2.0


def test_dominance_counted_per_draw_not_on_the_mean():
    """Un fold ou 3 tirages sur 5 depassent le classique ne doit pas etre
    compte 5/5 sous pretexte que la moyenne depasse."""
    runs = [{"phys_score": v, "patch_ratio": 0.9}
            for v in (0.5, 0.5, 2.0, 2.0, 2.0)]
    r = analyse(_rec(qc=1.0, qu=1.4, cc=1.0, cu=1.0, n=5, unseen_runs=runs))
    assert r["n_worse_on_unseen"] == 3
    assert r["n_dominated_on_unseen"] == 3


def test_cheaper_qhas_is_not_dominated():
    runs = [{"phys_score": 2.0, "patch_ratio": 0.4} for _ in range(5)]
    r = analyse(_rec(qc=1.0, qu=2.0, cc=1.0, cu=1.0, cp=0.6, n=5,
                     unseen_runs=runs))
    assert r["n_worse_on_unseen"] == 5
    assert r["n_costlier_on_unseen"] == 0
    assert r["n_dominated_on_unseen"] == 0


def test_single_run_output_is_flagged_underpowered(tmp_path):
    d = _rec(1.0, 1.0, 1.0, 1.0, n=1)["raw"]
    d["arms"]["qhas"]["n_runs"] = 1
    json.dump(d, open(os.path.join(
        str(tmp_path), "t22_unseen_unseen-ic_kh.json"), "w"))
    got = load(str(tmp_path), "kh")
    assert got is not None and got["underpowered"] is True


def test_missing_fold_returns_none(tmp_path):
    assert load(str(tmp_path), "nope") is None


def test_leak_free_really_replaces_the_leaked_threshold():
    """D-134 : le debranchement de la fuite D13 mesure, pas lu dans le source.

    `test_no_leak_mode_is_gone_and_leak_free_is_wired` ci-dessous cherche
    QUATRE chaines dans `h4_unseen_conditions.py`, dont
    `hp_q["threshold_amr"] = leak_free_thr` sous le message *« leak-free
    does not actually change the QAOA threshold »*. C'est un COMPORTEMENT.
    Mesure par mutation : une ligne
    `hp_q["threshold_amr"] = rec["qaoa_params"]["threshold_amr"]` ajoutee
    JUSTE APRES, les quatre chaines intactes -- le bras QAOA repart au seuil
    fuyant, et `test_t22c_transfer.py` + `test_t24_leak_free.py` restent
    **35 passed**, sous un artefact nomme `leak-free`.

    L'entree qui SEPARE : un `rec` dont le seuil classique DIFFERE du seuil
    fuyant. Si les deux coincidaient, appliquer le mode et ne pas
    l'appliquer rendraient la meme chose et le test ne mesurerait rien.

    Le calcul a ete extrait de `main()` en `apply_leak_free_threshold` pour
    etre appelable sans rejouer les heures de DNS d'un fold ; son corps est
    inchange.
    """
    import importlib

    huc = importlib.import_module("h4_unseen_conditions")

    #  Le seuil classique du fold, choisi loin du seuil fuyant.
    classical_thr = 0.31337
    assert abs(classical_thr - huc.LEAKED_THRESHOLD) > 0.1, (
        "le champ d'essai ne separe pas les deux seuils")

    hp_q = {"threshold_amr": huc.LEAKED_THRESHOLD, "beta": 1.0}
    rec = {"classical_params": {"threshold_amr": classical_thr},
           "qaoa_params": {"threshold_amr": huc.LEAKED_THRESHOLD}}

    huc.apply_leak_free_threshold(hp_q, rec)

    assert hp_q["threshold_amr"] == pytest.approx(classical_thr, abs=1e-12), (
        f"le bras QAOA tourne a {hp_q['threshold_amr']} et non au seuil "
        "classique du fold : la fuite D13 est de retour sous un artefact "
        "nomme leak-free")
    assert hp_q["threshold_amr"] != huc.LEAKED_THRESHOLD
    #  Le reste du dict n'est pas touche.
    assert hp_q["beta"] == 1.0

    #  Et le garde d'entree doit refuser un bras qui n'etait PAS au seuil
    #  fuyant : sinon le mode s'appliquerait a un fold deja corrige.
    with pytest.raises(AssertionError):
        huc.apply_leak_free_threshold(
            {"threshold_amr": 0.9}, dict(rec))


def test_no_leak_mode_is_gone_and_leak_free_is_wired(tmp_path):
    """Un mode accepte mais non implemente est un piege : `--mode no-leak`
    ne changeait que le nom du fichier de sortie, produisant un artefact
    nomme comme si la fuite D13 avait ete supprimee alors que le calcul
    etait identique. Il doit avoir disparu, et son remplacant doit etre
    REELLEMENT branche sur le seuil."""
    src = open(_study_file(
                            "h4_unseen_conditions.py")).read()
    assert '"no-leak"' not in src, "the unimplemented mode is back"
    assert '"leak-free"' in src
    # le mode doit modifier le seuil, pas seulement le nom de fichier
    assert 'hp_q["threshold_amr"] = leak_free_thr' in src, \
        "leak-free does not actually change the QAOA threshold"
    assert 'rec["classical_params"]["threshold_amr"]' in src, \
        "leak-free must take the threshold tuned on training classes only"
