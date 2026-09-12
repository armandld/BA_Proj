"""D-89 : `h4_unseen_floor.py` levait `KeyError: 'canonical'`, sans artefact
ecrit, des qu'un bras de `t22_unseen_{mode}_{fold}.json` avait le statut
`total_abort` — le cas que `h4_unseen_conditions.py` documente et gere
explicitement comme « un resultat, pas une panne » (voir son propre bloc
`dead = [...]`). Le dict d'un bras `total_abort` ne porte aucune sous-cle
`"canonical"`/`"unseen"`, seulement les `*_runs` bruts.

Le calcul de ratio a ete extrait de `main()` en `floor_ratios()` pour etre
testable sans rejouer les ~4h de DNS d'un fold.
"""
import json
import math
import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_H4_DIR = os.path.join(_REPO_ROOT, "study", "h4_transfer")
_RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
if _H4_DIR not in sys.path:
    sys.path.insert(0, _H4_DIR)

from h4_unseen_floor import floor_ratios  # noqa: E402


_FLOOR_CAN = {"phys_score": 0.005}
_FLOOR_UNS = {"phys_score": 0.006}


def test_total_abort_arm_does_not_raise_and_is_flagged():
    """Mesure avant (D-89) : cette meme entree levait `KeyError: 'canonical'`
    sur l'ancienne version en ligne dans `main()`. Mesure apres : pas
    d'exception, le bras mort est identifie, son ratio est NaN plutot
    qu'un nombre reconstitue."""
    t22_arms = {
        "qhas": {
            "n_runs": 3, "n_completed_canonical": 3, "n_completed_unseen": 0,
            "n_aborted": 3, "canonical_runs": [], "unseen_runs": [],
            "status": "total_abort", "degradation_ratio": float("nan"),
        },
        "classical": {
            "n_runs": 2, "n_completed_canonical": 2, "n_completed_unseen": 2,
            "canonical_runs": [], "unseen_runs": [],
            "canonical": {"phys_score": 0.01}, "unseen": {"phys_score": 0.02},
            "status": "completed", "degradation_ratio": 2.0,
        },
    }
    ratios, dead = floor_ratios(t22_arms, _FLOOR_CAN, _FLOOR_UNS)
    assert dead == ["qhas"]
    assert math.isnan(ratios["qhas"]["canonical_over_floor"])
    assert math.isnan(ratios["qhas"]["unseen_over_floor"])
    # le bras vivant n'est pas touche par la mort de l'autre
    assert ratios["classical"]["canonical_over_floor"] == 2.0
    assert abs(ratios["classical"]["unseen_over_floor"] - 0.02 / 0.006) < 1e-12


def test_both_arms_completed_matches_direct_computation():
    """Chemin non-degenere : le ratio est bien phys/phys_floor, sans
    detour par le drapeau `dead`."""
    t22_arms = {
        "qhas": {"canonical": {"phys_score": 0.02},
                 "unseen": {"phys_score": 0.03}, "status": "completed"},
        "classical": {"canonical": {"phys_score": 0.01},
                      "unseen": {"phys_score": 0.02}, "status": "completed"},
    }
    ratios, dead = floor_ratios(t22_arms, _FLOOR_CAN, _FLOOR_UNS)
    assert dead == []
    assert abs(ratios["qhas"]["canonical_over_floor"] - 0.02 / 0.005) < 1e-12
    assert abs(ratios["qhas"]["unseen_over_floor"] - 0.03 / 0.006) < 1e-12
    assert abs(ratios["classical"]["canonical_over_floor"]
               - 0.01 / 0.005) < 1e-12


def test_pre_status_field_artifact_is_not_treated_as_dead():
    """Un dict de bras qui porte `"canonical"`/`"unseen"` mais AUCUNE cle
    `"status"` (schema d'avant son introduction) ne doit pas etre traite
    comme mort. Verifier sur la presence des cles, pas sur `status`, est ce
    qui a evite de regresser les artefacts reels ci-dessous."""
    t22_arms = {
        "qhas": {"canonical": {"phys_score": 0.04},
                 "unseen": {"phys_score": 0.05}},
        "classical": {"canonical": {"phys_score": 0.01},
                      "unseen": {"phys_score": 0.02}},
    }
    ratios, dead = floor_ratios(t22_arms, _FLOOR_CAN, _FLOOR_UNS)
    assert dead == []
    assert not math.isnan(ratios["qhas"]["unseen_over_floor"])


def test_published_t22d_artifacts_reproduce_bit_for_bit():
    """Les 4 artefacts `t22_unseen_unseen-ic_*.json` deja committes datent
    d'avant l'ajout du champ `status` (mesure : absent des deux bras, des 4
    folds). `floor_ratios` doit rendre exactement les 8 ratios deja publies
    dans `results/t22d_unseen_floor_*.json` -- sinon la correction changerait
    un nombre publie, ce que `VIGIL.md` interdit sans le signaler."""
    for fold in ("ot", "kh", "rotor", "tearing"):
        t22_path = os.path.join(_RESULTS_DIR,
                                f"t22_unseen_unseen-ic_{fold}.json")
        pub_path = os.path.join(_RESULTS_DIR, f"t22d_unseen_floor_{fold}.json")
        assert os.path.exists(t22_path) and os.path.exists(pub_path), (
            f"artefact publie manquant pour {fold} ; ce test suppose les "
            "artefacts deja commites dans results/")
        t22 = json.load(open(t22_path))
        published = json.load(open(pub_path))
        assert "status" not in t22["arms"]["qhas"], (
            "cet artefact porte desormais 'status' -- le pretexte de ce "
            "test (schema d'avant son introduction) n'est plus vrai, "
            "et test_pre_status_field_artifact_is_not_treated_as_dead "
            "devient le seul a couvrir ce cas")
        ratios, dead = floor_ratios(
            t22["arms"], published["floor_canonical"],
            published["floor_unseen"])
        assert dead == [], (fold, dead)
        for arm in ("qhas", "classical"):
            for key in ("canonical_over_floor", "unseen_over_floor"):
                got = ratios[arm][key]
                want = published["arms"][arm][key]
                assert abs(got - want) < 1e-9, (fold, arm, key, got, want)
