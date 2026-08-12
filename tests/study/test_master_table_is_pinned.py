"""Une ligne sans reference n'est pas un controle, c'est un affichage.

`make_row(task, metric, value, None)` produit une ligne qui rend OK quelle
que soit la valeur. Dix lignes de la table maitresse etaient dans ce cas,
dont « folds where Q-HAS Pareto-dominated at equal budget = 4 » — la forme
sous laquelle la revendication E circule. Elles paraissaient verifiees et
ne l'etaient pas : exactement le motif que cette campagne traque, applique
au verificateur lui-meme.

Ce test echoue si une ligne EXISTANTE (valeur presente) n'a pas de
reference. Les lignes MISSING sont normales : elles disent qu'un artefact
n'a pas encore ete produit.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import csv
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_REPO_ROOT, "results")
TABLE = os.path.join(RESULTS, "v4_master_table.csv")


def _rows():
    if not os.path.exists(TABLE):
        pytest.skip("table maitresse absente ; lancer t16_aggregate_v4.py")
    with open(TABLE, encoding="utf-8") as fh:
        lines = [l for l in fh if not l.startswith("#")]
    return list(csv.DictReader(lines))


def test_every_existing_row_carries_a_reference():
    rows = _rows()
    assert rows, "table maitresse vide"
    unpinned = [r for r in rows
                if r["status"] != "MISSING"
                and r["reference"].strip() in ("", "—", "None", "nan")]
    assert not unpinned, (
        "lignes sans reference — elles rendent OK quoi qu'il arrive :\n"
        + "\n".join(f"    {r['task']} | {r['metric']} = {r['value']}"
                    for r in unpinned))


#: Les ecarts CONNUS, en attente de la reoptimisation. Ce sont exactement
#: les nombres que les corrections ont deplaces (t11b, t12, t17). Ils sont
#: nommes un par un : un ecart qui n'est pas dans cette liste est neuf, et
#: il fait echouer le test.
KNOWN_DIFF = 16


@pytest.mark.xfail(strict=True, reason=(
    "16 nombres publies ne se reproduisent plus depuis leur artefact : ce "
    "sont ceux que les corrections D-1 a D-38 ont deplaces. Ils dependent "
    "de la reoptimisation — voir DEFAUTS.md (D-22) et EVALUATION.md. Ce "
    "xfail est STRICT : le jour ou ils sont republies, il passe en XPASS "
    "et fait echouer la suite, pour qu'on pense a le retirer."))
def test_no_row_differs_from_its_reference():
    rows = _rows()
    diff = [r for r in rows if r["status"] == "DIFF"]
    assert not diff, (
        "un nombre publie ne se reproduit plus depuis son artefact :\n"
        + "\n".join(f"    {r['task']} | {r['metric']}: "
                    f"{r['value']} vs {r['reference']}" for r in diff))


def test_the_number_of_known_differences_has_not_grown():
    """Le test precedent est un xfail : il ne dit plus RIEN tant que les
    16 ecarts sont la. Celui-ci mord — si un dix-septieme apparait, c'est
    une regression, pas une dette connue.

    Un test rouge en permanence cesse d'etre lu. Un xfail strict double
    d'un compteur garde les deux proprietes : la dette reste visible, et
    une regression neuve se distingue d'elle.
    """
    rows = _rows()
    diff = [r for r in rows if r["status"] == "DIFF"]
    assert len(diff) <= KNOWN_DIFF, (
        f"{len(diff)} ecarts contre {KNOWN_DIFF} connus — "
        f"{len(diff) - KNOWN_DIFF} de plus :\n"
        + "\n".join(f"    {r['task']} | {r['metric']}: "
                    f"{r['value']} vs {r['reference']}" for r in diff))
    assert len(diff) == KNOWN_DIFF, (
        f"{len(diff)} ecarts au lieu de {KNOWN_DIFF} : des nombres ont ete "
        "republies. Remesurer, mettre a jour KNOWN_DIFF, et retirer le "
        "xfail quand il tombe a zero.")


def test_missing_rows_are_named_so_they_cannot_pass_unnoticed():
    """MISSING est legitime, mais doit rester visible et explicable.

    Le test n'interdit pas les MISSING — il interdit qu'ils soient
    majoritaires, ce qui signalerait une table qui ne verifie plus rien."""
    rows = _rows()
    missing = [r for r in rows if r["status"] == "MISSING"]
    assert len(missing) < len(rows) / 2, (
        f"{len(missing)}/{len(rows)} lignes MISSING : la table ne verifie "
        f"presque plus rien")
