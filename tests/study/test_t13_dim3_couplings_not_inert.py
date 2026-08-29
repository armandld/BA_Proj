"""H3 / T13, refait a `dim = 3` (la seule taille certifiee non degeneree,
D-53) apres que D-58 a retire l'explication qui accompagnait la lecture
`dim = 2` (« la fenetre d'incertitude tue le couplage ZZ »).

A `dim = 2`, retirer les couplages ZZ/ZZZZ ne changeait AUCUNE decision
(`RESULTS.md`, T13) — mais `dim = 2` est la taille ou l'etat fondamental
exact vaut « raffiner partout » quel que soit l'hamiltonien (D-45/D-47) :
l'inertie mesuree pouvait donc etre un artefact de cette degenerescence,
pas une propriete du formalisme Ising.

Rejoue a `dim = 3` (`results/t13_term_ablation_N96_dim3_{v1,v2}.npz`,
memes scenarios et artefacts DNS que D-53/D-200) : les couplages NE sont
PLUS inertes (5,6 % a 16,7 % des decisions changent). Et la ou ils
changent quelque chose, ils degradent le F1 — le Hamiltonien complet fait
pire que son seul biais Z + couplages retires. Meme sens que H0b (D-53,
D-200) : plus de structure Ising, pire decision.

Ces tests EPINGLENT les artefacts commis. Deviation tests, pas des tests
de regression : ils cassent le jour ou l'un des deux `.npz` est remplace
ou regenere differemment — c'est le signal qu'il faut relire ce fichier.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RESULTS = os.path.join(_REPO_ROOT, "results")

V2 = os.path.join(RESULTS, "t13_term_ablation_N96_dim3_v2.npz")
V1 = os.path.join(RESULTS, "t13_term_ablation_N96_dim3_v1.npz")


def _by_ablation(path):
    if not os.path.exists(path):
        pytest.skip(f"artefact absent : {os.path.basename(path)}")
    d = np.load(path, allow_pickle=True)
    ablation = d["ablation"]
    out = {}
    for name in dict.fromkeys(ablation.tolist()):
        mask = ablation == name
        out[name] = dict(
            changed=float(np.mean(d["changed"][mask])),
            removed_max=float(np.max(d["removed_max"][mask])),
            f1=float(np.mean(d["f1"][mask])),
        )
    return out, d


@pytest.mark.parametrize("path", [V2, V1], ids=["mapper_v2", "mapper_v1"])
def test_the_control_still_reads_zero_at_dim3(path):
    rows, _ = _by_ablation(path)
    assert rows["full"]["changed"] == 0.0


@pytest.mark.parametrize("path", [V2, V1], ids=["mapper_v2", "mapper_v1"])
def test_couplings_are_not_causally_inert_at_dim3(path):
    """Le coeur de la refutation de la lecture `dim = 2` : a `dim = 3`,
    ablater ZZ ou ZZZZ change reellement des decisions, et l'ablation a
    bien retire quelque chose (`removed_max > 0`, sinon `changed = 0`
    ne dirait rien -- D-51/D-54)."""
    rows, _ = _by_ablation(path)
    for name in ("no_ZZ", "no_ZZZZ", "Z_only"):
        assert rows[name]["removed_max"] > 0.0, (
            f"{name} n'a rien retire de l'operateur — changed=0 ne "
            f"prouverait rien (D-51/D-54)")
        assert rows[name]["changed"] > 0.0, (
            f"{name} reste causalement inerte a dim=3 : la degenerescence "
            f"dim=2 (D-45/D-47) ne serait alors pas l'explication")


@pytest.mark.parametrize("path", [V2, V1], ids=["mapper_v2", "mapper_v1"])
def test_removing_the_couplings_improves_rather_than_hurts_f1(path):
    """La ou les couplages changent une decision, ils la degradent : le
    F1 de `Z_only` (biais Z seul) depasse celui du hamiltonien complet.
    Meme sens que H0b : plus de structure Ising, pire decision AMR."""
    rows, _ = _by_ablation(path)
    assert rows["Z_only"]["f1"] > rows["full"]["f1"]


def test_v1_and_v2_agree_on_the_qualitative_reading():
    """Les deux mappeurs racontent la meme histoire — ce n'est pas un
    artefact du mappeur sans parametre V2 (comme D-200 l'a deja verifie
    pour H0b)."""
    rows_v2, meta_v2 = _by_ablation(V2)
    rows_v1, meta_v1 = _by_ablation(V1)
    for name in ("no_ZZ", "no_ZZZZ", "Z_only"):
        assert rows_v2[name]["changed"] > 0.0
        assert rows_v1[name]["changed"] > 0.0
    assert rows_v2["Z_only"]["f1"] > rows_v2["full"]["f1"]
    assert rows_v1["Z_only"]["f1"] > rows_v1["full"]["f1"]
    assert "\"dim\": 3" in str(meta_v2["cli_args"])
    assert "\"dim\": 3" in str(meta_v1["cli_args"])
