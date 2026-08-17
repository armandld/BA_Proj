"""D-47 — la phase 5 ne filtre plus sur `promising`.

La porte de la phase 4 vaut
`promising = comparison["exact"]["f1"] >= comparison["classical"]["f1"]`
(`study/pipeline/exact_diagonalisation.py`). La phase 5
(`study/common/qaoa_inputs.py`) l'utilisait pour ÉCARTER des instantanés.

Mesure qui a motivé le changement, 40 instantanés (dim=2, Re=400, N=256,
4 scénarios canoniques) :

    décision exacte tout-à-1        40/40
    ligne de base classique tout-à-1 40/40
    exact_refine != classical_refine  0/40
    F1 égaux                        40/40, jamais supérieurs

Deux prédicteurs **constants identiques** rendent le même F1 par
construction. La porte valait donc `True` 40/40 avec `>=`, et aurait valu
`False` 40/40 avec le `>` écrit dans son propre commentaire : **zéro bit
d'information dans les deux sens.**

Cause mesurée : à résolution VQA le score est à **8,4 σ** du seuil, donc la
fenêtre gaussienne du couplage ZZ vaut au plus **1,15e−31** ; le biais Z,
positif partout, domine le ZZZZ d'un facteur **2,0 à 6,6**. Le fondamental
met tous les qubits à |1⟩ faute de terme portant une structure spatiale.

Décision de USER (option 1) : **documenter comme résultat.** C'est une
limite structurelle de l'hamiltonien v1 à cette résolution, pas un défaut
de la phase 4. `σ` et `w_z_frac` sont deux des neuf paramètres réoptimisés
— les régler à la main reviendrait à décider par avance ce que la campagne
existe pour mesurer.

Ce banc vérifie les deux moitiés de la décision : la phase 5 ne filtre
plus, et le diagnostic reste imprimé.
"""

import ast
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_QAOA_INPUTS = os.path.join(_REPO, "study", "common", "qaoa_inputs.py")
_EXACT_DIAG = os.path.join(_REPO, "study", "pipeline", "exact_diagonalisation.py")


def _fonction(chemin, nom):
    """L'AST de la fonction demandée. On interroge la STRUCTURE, pas le
    texte : un test qui cherche une chaîne passerait encore si le filtre
    revenait sous un autre nom de variable."""
    arbre = ast.parse(open(chemin, encoding="utf-8").read())
    for n in ast.walk(arbre):
        if isinstance(n, ast.FunctionDef) and n.name == nom:
            return n
    raise AssertionError(f"{nom} introuvable dans {chemin}")


def _noms_lus(noeud):
    return {n.id for n in ast.walk(noeud) if isinstance(n, ast.Name)}


# ══════════════════════════════════════════════════════════════════
#  1. Le filtre a disparu — vérifié sur la STRUCTURE
# ══════════════════════════════════════════════════════════════════

def test_aucun_continue_conditionne_par_promising():
    """Le cœur de D-47 : `if not promising[idx]: continue` écartait des
    instantanés sur un prédicteur constant.

    On cherche tout `continue` (ou `return`) gardé par une condition qui
    LIT `promising` — quelle que soit la forme de la condition.
    """
    src = open(_QAOA_INPUTS, encoding="utf-8").read()
    arbre = ast.parse(src)
    fautifs = []
    for noeud in ast.walk(arbre):
        if not isinstance(noeud, ast.If):
            continue
        if "promising" not in _noms_lus(noeud.test):
            continue
        for enfant in noeud.body:
            if isinstance(enfant, (ast.Continue, ast.Return)):
                fautifs.append(noeud.lineno)
    assert not fautifs, (
        f"lignes {fautifs} : la phase 5 saute encore des instantanés selon "
        f"`promising`, qui est un prédicteur constant (40/40). Filtrer "
        f"dessus, c'est sélectionner sur une constante — D-47.")


def test_la_boucle_de_phase5_parcourt_tous_les_instantanes():
    """La boucle doit itérer sur `range(len(snap_indices))` sans exclusion."""
    src = open(_QAOA_INPUTS, encoding="utf-8").read()
    assert "for idx in range(len(snap_indices)):" in src, (
        "la boucle de phase 5 a changé de forme : revérifier qu'elle ne "
        "réintroduit pas d'exclusion")


# ══════════════════════════════════════════════════════════════════
#  2. Le diagnostic, lui, doit rester
# ══════════════════════════════════════════════════════════════════

def test_le_compte_promising_reste_imprime():
    """Retirer le filtre ne doit pas retirer la MESURE.

    `promising` reste un diagnostic utile : le jour où la réoptimisation
    déplace `σ` ou `w_z_frac`, il redeviendra informatif, et on veut le
    voir dans les journaux de campagne.
    """
    src = open(_QAOA_INPUTS, encoding="utf-8").read()
    assert "n_promising" in src, "le compte a disparu avec le filtre"
    assert "diagnostic" in src, (
        "le compte est imprimé sans dire qu'il ne filtre plus : un lecteur "
        "de journal croira que des instantanés ont été écartés")


def test_la_porte_reste_calculee_en_phase4():
    """On retire l'USAGE, pas la mesure. `promising` doit continuer d'être
    calculé et sauvegardé par la phase 4, sinon on perd le diagnostic."""
    src = open(_EXACT_DIAG, encoding="utf-8").read()
    assert "promising" in src, (
        "la phase 4 ne calcule plus `promising` : le diagnostic est perdu")


# ══════════════════════════════════════════════════════════════════
#  3. Le garde-fou : ce banc peut-il encore échouer ?
# ══════════════════════════════════════════════════════════════════

def test_le_detecteur_de_filtre_mord_vraiment():
    """Sans ceci, `test_aucun_continue_conditionne_par_promising` pourrait
    devenir incapable d'échouer (motif AST qui ne correspond plus à rien)
    et passerait au vert sur un filtre réintroduit."""
    faux = ast.parse(
        "def f():\n"
        "    for idx in range(3):\n"
        "        if not promising[idx]:\n"
        "            continue\n")
    trouve = []
    for noeud in ast.walk(faux):
        if isinstance(noeud, ast.If) and "promising" in _noms_lus(noeud.test):
            for enfant in noeud.body:
                if isinstance(enfant, (ast.Continue, ast.Return)):
                    trouve.append(noeud.lineno)
    assert trouve, (
        "le motif AST ne reconnaît plus un filtre `promising` explicite : "
        "le test principal ne pourrait plus échouer")
