"""Un F1 nul dit-il « pas de signal » ou « seuil non transféré » ?

Les deux rendent la même chose — prédiction constante, F1 au plancher — et
le protocole §1.3-B3 les exclut tous deux des décomptes. Mais ce ne sont pas
le même fait, et les confondre a coûté une lecture publiée : le pli
`harris_tearing` a été rapporté comme « 0,000 à tous les k », lu comme un
pli mort, et porté dans deux versions de `RESULTS.md`.

Mesuré (LOSO, dim=16, N=96, 20 instantanés) :

    probabilités du GBT sur harris : max = 0,1243
    seuil ajusté sur les 3 autres  : 0,4000
    -> 0 positif prédit sur 20 480 -> F1 = 0,000
    mais AUC = 0,908 et F1 à budget apparié = 0,659

Le classement est bon. C'est l'opérateur de décision qui ne traverse pas la
frontière de scénario. `threshold_transfer_flag` sépare désormais les deux
cas, et ces tests vérifient qu'il les sépare **sur des entrées où ils
divergent** — un discriminateur qui rendrait le même verdict partout ne
discriminerait rien.
"""
import os
import sys

import numpy as np
import pytest

_RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_RACINE, "src"),
           os.path.join(_RACINE, "study", "common"),
           os.path.join(_RACINE, "study", "pipeline"),
           os.path.join(_RACINE, "study", "h2b_prediction")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metrics import threshold_transfer_flag, degeneracy_flag   # noqa: E402


def _etiquettes(n=2000, prevalence=0.25, graine=0):
    rng = np.random.default_rng(graine)
    return (rng.random(n) < prevalence).astype(int), rng


# ------------------------------------------------------------------
#  1. les trois verdicts, sur des entrées qui les séparent
# ------------------------------------------------------------------
def test_un_classement_bon_a_probas_basses_est_un_defaut_de_seuil():
    """Le cas harris : le modèle classe bien mais ne franchit aucun seuil."""
    y, rng = _etiquettes()
    proba = np.clip(0.05 * y + 0.02 * rng.random(y.size), 0, 1)   # max ~0.07
    r = threshold_transfer_flag(y, proba, threshold=0.40)
    assert r["verdict"] == "seuil_non_transfere"
    assert r["degenerate"] and r["auc"] > 0.9
    assert r["proba_max"] < r["threshold"], "le cas testé n'est pas celui décrit"


def test_un_bruit_pur_est_une_absence_de_signal():
    """Le champ qui SÉPARE : mêmes probas basses, mais sans classement."""
    y, rng = _etiquettes()
    proba = rng.random(y.size) * 0.1                               # max ~0.10
    r = threshold_transfer_flag(y, proba, threshold=0.40)
    assert r["verdict"] == "aucun_signal"
    assert r["degenerate"] and r["auc"] < 0.60


def test_une_prediction_non_constante_passe():
    y, rng = _etiquettes()
    proba = np.clip(0.7 * y + 0.3 * rng.random(y.size), 0, 1)
    r = threshold_transfer_flag(y, proba, threshold=0.40)
    assert r["verdict"] == "ok" and not r["degenerate"]


def test_le_verdict_ne_depend_pas_du_seul_taux_de_positifs():
    """Sans l'AUC, les deux premiers cas seraient indiscernables.

    `degeneracy_flag` — la règle §1.3-B3 seule — rend le MÊME verdict sur
    les deux, ce qui est précisément la confusion qu'on corrige.
    """
    y, rng = _etiquettes()
    bon = np.clip(0.05 * y + 0.02 * rng.random(y.size), 0, 1)
    bruit = rng.random(y.size) * 0.1
    for p in (bon, bruit):
        assert degeneracy_flag((p > 0.40).astype(int), y.mean()) is True
    assert (threshold_transfer_flag(y, bon, 0.40)["verdict"]
            != threshold_transfer_flag(y, bruit, 0.40)["verdict"])


def test_tout_positif_est_aussi_degenere():
    """L'autre plancher : refine-all."""
    y, rng = _etiquettes()
    proba = 0.9 + 0.05 * rng.random(y.size)
    r = threshold_transfer_flag(y, proba, threshold=0.40)
    assert r["degenerate"] and r["positive_rate"] > 0.99


def test_une_seule_classe_ne_fait_pas_planter():
    y = np.ones(100, dtype=int)
    r = threshold_transfer_flag(y, np.full(100, 0.01), threshold=0.4)
    assert np.isnan(r["auc"]) and r["verdict"] == "aucun_signal"


def test_des_formes_incompatibles_sont_refusees():
    with pytest.raises(ValueError, match="coincident"):
        threshold_transfer_flag(np.zeros(10), np.zeros(11), 0.5)


# ------------------------------------------------------------------
#  2. le cas réel, épinglé
# ------------------------------------------------------------------
@pytest.mark.slow
def test_le_pli_harris_est_un_defaut_de_seuil_et_non_un_pli_mort():
    """Reproduit le pli LOSO `harris_tearing` et exige le bon verdict.

    Sur quelle entrée ce test échoue : le jour où le GBT cesse de classer
    harris — l'AUC tomberait sous 0,70 et le verdict deviendrait
    `aucun_signal`, ce qui serait alors un VRAI pli mort et changerait la
    lecture. Il échoue aussi si les artefacts d'entrée bougent.
    """
    from h2b_ceiling_random_split import (extract_features_2d, make_model,
                                          best_threshold_f1)
    SC = ["orszag_tang", "harris_tearing", "kelvin_helmholtz", "mhd_rotor"]
    RE, DIM, N, MS = [400, 800, 1200, 1600], 16, 96, 20
    res = os.path.join(_RACINE, "results")

    def charge(sc):
        X, Y = [], []
        for re in RE:
            dp = os.path.join(res, f"dns_{sc}_Re{re}_N{N}.npz")
            pp = os.path.join(res, f"patches_{sc}_Re{re}_N{N}_dim{DIM}.npz")
            if not (os.path.exists(dp) and os.path.exists(pp)):
                pytest.skip(f"artefacts absents : {sc} Re={re}")
            d, pat = np.load(dp), np.load(pp)
            l2, thr = pat["l2_errors"], float(pat["l2_threshold"])
            n = len(d["vx"])
            for si in list(range(0, n, max(1, n // MS)))[:MS]:
                f2, _ = extract_features_2d(
                    d["vx"][si].astype(float), d["vy"][si].astype(float),
                    d["Bx"][si].astype(float), d["By"][si].astype(float),
                    N, DIM, re)
                X.append(f2.reshape(-1, f2.shape[-1]))
                Y.append((l2[si] >= thr).ravel().astype(int))
        return np.concatenate(X), np.concatenate(Y)

    data = {s: charge(s) for s in SC}
    held = "harris_tearing"
    Xtr = np.concatenate([data[s][0] for s in SC if s != held])
    Ytr = np.concatenate([data[s][1] for s in SC if s != held])
    Xva, Yva = data[held]

    m = make_model("gbt", 0)
    m.fit(Xtr, Ytr)
    t, _ = best_threshold_f1(m.predict_proba(Xtr)[:, 1], Ytr,
                             grid=np.linspace(0.05, 0.95, 91))
    r = threshold_transfer_flag(Yva, m.predict_proba(Xva)[:, 1], t)

    assert r["verdict"] == "seuil_non_transfere", (
        f"verdict={r['verdict']}, auc={r['auc']:.3f} : si l'AUC est tombée, "
        "harris est devenu un vrai pli mort et la lecture change")
    assert r["proba_max"] < r["threshold"], (
        f"proba_max={r['proba_max']:.4f} >= seuil={r['threshold']:.4f} : le "
        "mécanisme décrit n'est plus celui qui produit le zéro")
    assert r["auc"] > 0.80, f"AUC={r['auc']:.3f} — le classement s'est dégradé"
