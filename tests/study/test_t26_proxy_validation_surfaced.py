"""D-57 : T26 calculait la validation de son proxy, la rangeait dans le
JSON, et ne l'imprimait ni ne la controlait jamais.

`h3_size_scan.py` remplace l'etat fondamental exact par une descente
gloutonne des que `2*dim^2 > 22`, et son en-tete annonce « warm-started
greedy (**validated at dim=2**) ». La validation existait bien —
`greedy_agrees_with_exhaustive` — mais elle n'apparaissait ni dans la table
de synthese ni dans aucun avertissement : elle finissait dans le JSON.

Mesure (N=96 et N=256, 4 scenarios canoniques, 12 instantanes) :

    --mapper v1  (le DEFAUT de la tache)   dim=2 -> 0,7500
    --mapper v2                            dim=2 -> 1,0000

Le 0,75 figure deja dans `results/t26_size_scan_N256_v1.json`, ou personne
ne le lit. C'est ce nombre, et lui seul, qui autorise a lire `dim = 4` et
`dim = 8`, ou seul le glouton tourne.

**Ce que ce defaut n'est PAS.** La conclusion de T26 n'est pas contaminee :
le controle `--force-greedy` du module a ete rejoue a dim=2, mappeurs v1 et
v2, et le glouton rend `changed = 0,0000` sur les quatre ablations, comme
l'exhaustif. Le proxy ne fabrique pas les changements qu'il rapporte. Le
risque etait reel, il ne s'est pas realise — et rien ne le disait.
"""
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from h3_size_scan import proxy_validation_message      # noqa: E402

#: Le resume que la tache produit reellement a `--mapper v1 --dims 2 4 8`,
#: avec le nombre mesure.
V1_SUMMARY = [
    dict(dim=2, n_qubits=8, greedy_agreement=0.75, method="exhaustive"),
    dict(dim=4, n_qubits=32, greedy_agreement=None, method="greedy_warm"),
    dict(dim=8, n_qubits=128, greedy_agreement=None, method="greedy_warm"),
]
V2_SUMMARY = [
    dict(dim=2, n_qubits=8, greedy_agreement=1.0, method="exhaustive"),
    dict(dim=4, n_qubits=32, greedy_agreement=None, method="greedy_warm"),
]


def test_imperfect_agreement_is_surfaced_and_warned():
    """Le coeur de D-57 : 0,75 doit se voir, et contredire explicitement
    l'en-tete qui dit « validated at dim=2 »."""
    msg = proxy_validation_message(V1_SUMMARY)
    assert "0.7500" in msg
    assert "WARNING" in msg
    assert "is not" in msg


def test_dimensions_carried_by_the_proxy_are_named():
    """Un lecteur doit savoir QUELLES lignes reposent sur le proxy."""
    msg = proxy_validation_message(V1_SUMMARY)
    assert "dim=4 (32 q)" in msg and "dim=8 (128 q)" in msg


def test_perfect_agreement_does_not_cry_wolf():
    """Un avertissement qui se declenche toujours cesse d'etre lu."""
    msg = proxy_validation_message(V2_SUMMARY)
    assert "1.0000" in msg
    assert "WARNING" not in msg


def test_a_scan_with_no_validated_dimension_says_so():
    """`--dims 4 8` ne peut valider nulle part : le silence serait pire que
    l'avertissement."""
    msg = proxy_validation_message(
        [dict(dim=4, n_qubits=32, greedy_agreement=None, method="greedy_warm")])
    assert "UNVALIDATED" in msg


def test_the_published_artifact_still_carries_the_measured_075():
    """Epingle la valeur la ou elle vit deja. Si l'artefact est rejoue et
    que le nombre bouge, D-57 est a relire."""
    import json
    path = os.path.join(_REPO_ROOT, "results", "t26_size_scan_N256_v1.json")
    if not os.path.exists(path):
        pytest.skip("artefact t26 absent")
    with open(path, encoding="utf-8") as fh:
        summary = json.load(fh)["summary"]
    checked = [s for s in summary if s.get("greedy_agreement") is not None]
    assert checked, "plus aucune dimension validee dans l'artefact publie"
    assert checked[0]["greedy_agreement"] == pytest.approx(0.75, abs=5e-3)
    # ... et le message le montrerait desormais
    assert "0.7500" in proxy_validation_message(summary)
