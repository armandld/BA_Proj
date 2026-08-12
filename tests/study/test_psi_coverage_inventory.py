"""Quels points d'entree de l'etude tournent encore avec psi = 0 ?

`qaoa_inputs.prepare_qaoa_inputs` met psi a zero sauf si on lui passe
`with_psi=True` ET l'instantane precedent. Le pipeline deploye, lui, calcule
psi (`refinement.py:181`), et la campagne Optuna a regle les hyperparametres
avec psi actif (`train_hyperparams.py:70` importe `pipeline`). Tout script
de l'etude qui n'a pas rebranche psi evalue donc une variante du modele
amputee d'un de ses trois encodages.

Ce fichier tient l'inventaire. Il echoue dans LES DEUX SENS :

  - si un script rebranche psi sans que l'inventaire soit mis a jour, le
    test tombe et force a consigner le progres ;
  - si un script perd son cablage psi, il tombe aussi.

Un simple commentaire « TODO : activer psi » ne tomberait ni dans un cas ni
dans l'autre, et la dette resterait invisible jusqu'a la relecture du
manuscrit.

Pourquoi ce n'est pas un simple drapeau a propager
--------------------------------------------------
psi est une derivee temporelle : chaque appelant doit conserver l'instantane
precedent de SA trajectoire. Deux scripts demandent en plus une decision
scientifique :

  - `h3_equivariance` applique des transformations de symetrie au champ.
    L'instantane precedent doit subir LA MEME transformation, sinon psi
    casse l'equivariance testee pour une raison purement instrumentale et le
    script mesurerait son propre defaut de cablage.
  - `h3_size_scan` change la taille des patches entre passes ; psi doit etre
    recalcule a chaque taille, pas reutilise.
"""

import ast
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_STUDY = os.path.join(_REPO_ROOT, "study")

#  Scripts appelant prepare_qaoa_inputs SANS rebrancher psi, a ce jour.
#  Retirer une entree quand le cablage est fait — le test le verifiera.
PSI_STILL_ZERO = {
    "h0_qaoa_displacement.py",
    "h1_curl_convention_gap.py",
    "h3_equivariance.py",
    "h3_size_scan.py",
    "h3_term_ablation.py",
    "h3_window_counterfactual.py",
}

#  Scripts ou psi est rebranche.
PSI_WIRED = {
    "h0_optimiser_equivalence.py",
}


def _called_name(func):
    """Nom appele, que l'appel soit `f(...)` ou `mod.f(...)`."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _callers():
    """Modules de study/ qui appellent prepare_qaoa_inputs, hors la def."""
    found = {}
    for dirpath, _dirs, names in os.walk(_STUDY):
        if "__pycache__" in dirpath:
            continue
        for n in names:
            if not n.endswith(".py"):
                continue
            path = os.path.join(dirpath, n)
            src = open(path, encoding="utf-8").read()
            tree = ast.parse(src, filename=path)
            # Les deux formes d'appel doivent etre vues : `f(...)` et
            # `module.f(...)`. Ne reconnaitre que la premiere laisserait
            # h3_window_counterfactual (qui appelle `p5.prepare_qaoa_inputs`)
            # hors du balayage, sans que rien ne le signale.
            calls = [node for node in ast.walk(tree)
                     if isinstance(node, ast.Call)
                     and _called_name(node.func) == "prepare_qaoa_inputs"]
            if calls:
                found[n] = calls
    #  qaoa_inputs.py definit la fonction ; il n'est pas un appelant.
    found.pop("qaoa_inputs.py", None)
    return found


def _passes_with_psi(calls):
    return any(kw.arg == "with_psi" for c in calls for kw in c.keywords)


def test_the_inventory_lists_exactly_the_callers():
    """L'inventaire doit couvrir tous les appelants, ni plus ni moins.

    Un nouveau script qui appellerait prepare_qaoa_inputs sans figurer ici
    passerait sous le radar : c'est precisement ce qu'on veut empecher.
    """
    actual = set(_callers())
    declared = PSI_STILL_ZERO | PSI_WIRED
    assert actual == declared, (
        f"inventaire desynchronise.\n"
        f"  appelants non declares : {sorted(actual - declared)}\n"
        f"  declares mais absents  : {sorted(declared - actual)}")


def test_the_wired_scripts_really_pass_with_psi():
    callers = _callers()
    for name in sorted(PSI_WIRED):
        assert _passes_with_psi(callers[name]), (
            f"{name} est declare comme rebranche mais n'passe jamais "
            "with_psi a prepare_qaoa_inputs")


def test_the_unwired_scripts_really_run_with_psi_zero():
    """Le sens qui compte : quand un script est cable, ce test tombe.

    Il faut alors deplacer son nom de PSI_STILL_ZERO vers PSI_WIRED — ce qui
    force a constater le progres au lieu de l'oublier.
    """
    callers = _callers()
    for name in sorted(PSI_STILL_ZERO):
        assert not _passes_with_psi(callers[name]), (
            f"{name} passe desormais with_psi : le deplacer de "
            "PSI_STILL_ZERO vers PSI_WIRED dans cet inventaire, et "
            "consigner le changement dans docs/RESULTS.md")


def test_the_debt_is_not_silently_growing():
    """Un garde-fou grossier sur la taille de la dette."""
    assert len(PSI_STILL_ZERO) <= 6, (
        f"{len(PSI_STILL_ZERO)} scripts tournent encore sans psi ; la dette "
        "augmente au lieu de diminuer")
    assert PSI_WIRED, "aucun script ne rebranche psi : le cablage a disparu"
