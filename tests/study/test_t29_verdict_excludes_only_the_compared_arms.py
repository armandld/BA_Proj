"""D-79 : `verdict()` de T29 ecartait un fold parce qu'un predicteur ETRANGER
a la quantite comparee s'etait effondre sur une constante.

La quantite votee est `F1(stencil) - F1(site)`. Le predicteur classique
n'y entre pas — il est calcule et imprime pour situer les deux modeles, rien
de plus. Il figurait pourtant dans la meme liste `constant`, et cette liste
decide qui vote : un fold dont les DEUX bras compares etaient sains pouvait
etre ecarte parce qu'un troisieme, hors comparaison, etait constant.

Mesure — rejeu de T29 `--dim 4 --bootstrap 500` avec le `src/` d'avant D-1
(la configuration qui a produit l'artefact publie) :

    fold                F1_cls  F1_site  F1_sten   d_sten-site [IC95]      etat
    kelvin_helmholtz     0.400    0.353    0.326   -0.027 [-0.050,-0.001]  constant: cls
    harris_tearing       0.400    0.404    0.407   +0.003 [-0.099,+0.134]  constant: cls

    avant : « folds retenus : 2/4 » — `kelvin_helmholtz` ECARTE alors que son
            IC95 exclut zero, donc qu'il tranchait
    apres : « folds retenus : 4/4 » — les deux bras compares sont sains dans
            les quatre folds

Le verdict de cette configuration reste « indecidable » (les folds retenus ne
s'accordent pas) : la correction ne change pas la conclusion la, elle rend au
vote un fold decisif qu'il jetait. C'est dit plutot qu'embelli.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (os.path.join(_REPO_ROOT, "src"),
           os.path.join(_REPO_ROOT, "study", "pipeline"),
           os.path.join(_REPO_ROOT, "study", "common"),
           os.path.join(_REPO_ROOT, "study", "h2b_prediction")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from h2b_loso_delta_ci import constant_predictor, verdict  # noqa: E402


def _row(held, ci_low, ci_high, constant_compared, constant):
    return dict(held=held, ci_low=ci_low, ci_high=ci_high,
                constant=constant, constant_compared=constant_compared)


def test_a_fold_whose_two_compared_arms_are_sound_still_votes():
    """Le cas mesure, choisi pour qu'il SEPARE les deux regles.

    `kelvin_helmholtz` : `cls` constant, `site` et `sten` sains, IC95
    entierement negatif — decisif. `harris_tearing` : un bras compare
    (`sten`) effondre, donc ecarte par les deux regles.

    ancienne regle : plus aucun fold ne vote -> « indecidable »
    nouvelle regle : `kelvin_helmholtz` vote et tranche -> « nuisent »
    """
    rows = [
        _row("kelvin_helmholtz", -0.050, -0.001, "", "cls"),
        _row("harris_tearing", -0.400, +0.100, "sten", "sten,cls"),
    ]
    assert verdict(rows) == "nuisent", (
        "un fold dont les deux bras compares sont sains doit voter meme si "
        "le predicteur classique est constant : il n'entre pas dans "
        "F1(sten) - F1(site)")


def test_the_classical_collapse_does_not_manufacture_a_verdict_either():
    """L'autre sens, sur un champ qui separe aussi : rendre son vote a un
    fold peut RETIRER un verdict, pas seulement en donner un. Ancienne
    regle : un seul fold vote et tranche « nuisent ». Nouvelle : les deux
    votent, ils ne s'accordent pas -> « indecidable »."""
    rows = [
        _row("mhd_rotor", -0.050, -0.010, "", ""),
        _row("orszag_tang", -0.020, +0.030, "", "cls"),
    ]
    assert verdict(rows) == "indecidable"


def test_a_fold_whose_compared_arm_collapsed_still_does_not_vote():
    """La regle qu'on garde : si `site` ou `sten` est constant, son F1 ne
    mesure pas un modele et le fold ne vote pas."""
    rows = [
        _row("harris_tearing", -0.400, -0.100, "sten", "sten,cls"),
        _row("orszag_tang", -0.045, +0.004, "", ""),
    ]
    assert verdict(rows) == "indecidable", (
        "un fold dont un bras COMPARE s'est effondre ne doit pas voter")


def test_the_old_rule_would_have_silenced_the_measured_fold():
    """Epingle l'ancien comportement. Si cette assertion cesse de tenir,
    c'est que `constant` a cesse de porter `cls` — et le motif d'exclusion
    d'avant D-79 serait revenu sans qu'on le voie."""
    r = _row("kelvin_helmholtz", -0.050, -0.001, "", "cls")
    old_rule_votes = not r["constant"]          # la regle d'avant D-79
    new_rule_votes = not r["constant_compared"]  # celle d'apres
    assert not old_rule_votes and new_rule_votes, (
        "ce test ne separe plus les deux regles : il ne mesure rien")


def test_all_folds_degenerate_is_still_undecidable():
    rows = [_row("a", -0.1, -0.01, "site", "site,cls"),
            _row("b", +0.01, +0.1, "sten", "sten")]
    assert verdict(rows) == "indecidable"


def test_constant_predictor_detects_one_class_only():
    assert constant_predictor(np.zeros(10, dtype=int))
    assert constant_predictor(np.ones(10, dtype=int))
    assert not constant_predictor(np.array([0, 1, 0, 1]))


def test_the_script_records_both_lists_in_its_artefact():
    """`constant` (information) et `constant_compared` (ce qui decide le
    vote) doivent tous deux etre ecrits : sans le second, un relecteur ne
    peut pas refaire le compte des folds retenus."""
    import ast
    src = os.path.join(_REPO_ROOT, "study", "h2b_prediction",
                       "h2b_loso_delta_ci.py")
    with open(src, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    saved = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "savez_compressed"):
            saved.update(kw.arg for kw in node.keywords)
    assert {"constant", "constant_compared"} <= saved, (
        f"l'artefact T29 n'ecrit que {sorted(saved & {'constant', 'constant_compared'})}")
