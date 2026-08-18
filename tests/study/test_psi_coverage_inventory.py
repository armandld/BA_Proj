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

import pytest

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


def _alias_locaux(tree):
    """Noms sous lesquels CE fichier a lie `prepare_qaoa_inputs`.

    D-155 : `from study.common.qaoa_inputs import prepare_qaoa_inputs as prep`
    puis `prep(...)` echappait au balayage — un script pouvait donc sortir de
    l'inventaire sans que rien ne le dise."""
    noms = {"prepare_qaoa_inputs"}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name == "prepare_qaoa_inputs" and a.asname:
                    noms.add(a.asname)
    return noms


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
            alias = _alias_locaux(tree)
            calls = [node for node in ast.walk(tree)
                     if isinstance(node, ast.Call)
                     and _called_name(node.func) in alias]
            if calls:
                found[n] = calls
    #  qaoa_inputs.py definit la fonction ; il n'est pas un appelant.
    found.pop("qaoa_inputs.py", None)
    return found


def _etat_psi(calls):
    """L'etat REEL du cablage psi d'un script — pas la presence du mot-cle.

    D-155. `_passes_with_psi` rendait `True` des qu'un appel portait un
    mot-cle nomme `with_psi`, quelle que soit sa valeur. Mesure du 18 aout
    2026 : le seul script declare cable (`h0_optimiser_equivalence.py`) mis
    a `with_psi=False` en dur — psi mort dans TOUT `study/` — laissait ce
    fichier a **4 passed**. C'est la lecon de `assert len(params) == 4` :
    l'assertion doit porter sur la garantie annoncee, pas sur la forme.

    Rend l'un de :
      "absent"   aucun appel ne passe with_psi        -> psi = 0
      "faux"     tous les appels passent le litteral False -> psi = 0
      "cable"    au moins un appel passe True, ou une expression qui peut
                 valoir True (un drapeau CLI, par exemple)

    `**kwargs` LEVE : un balayage statique ne peut pas voir a travers, et un
    balayage qui ne voit pas doit crier plutot que conclure (D-145)."""
    vus = []
    for c in calls:
        for kw in c.keywords:
            if kw.arg is None:
                raise AssertionError(
                    "un appel passe **kwargs : la valeur de with_psi n'est "
                    "pas lisible statiquement. Ecrire le mot-cle en clair, "
                    "ou deplacer ce script hors de l'inventaire en le "
                    "disant dans docs/RESULTS.md")
            if kw.arg == "with_psi":
                vus.append(kw.value)
    if not vus:
        return "absent"
    if all(isinstance(v, ast.Constant) and v.value is False for v in vus):
        return "faux"
    return "cable"


def _passe_l_instantane_precedent(calls):
    """La seconde moitie du cablage : `with_psi=True` sans `prev_fields`
    leve dans `prepare_qaoa_inputs`. Un script declare cable doit donc
    passer les deux — et pas `prev_fields=None`."""
    for c in calls:
        for kw in c.keywords:
            if kw.arg == "prev_fields" and not (
                    isinstance(kw.value, ast.Constant) and kw.value.value is None):
                return True
    return False


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
    """D-155 : la VALEUR, pas la presence du mot-cle."""
    callers = _callers()
    for name in sorted(PSI_WIRED):
        etat = _etat_psi(callers[name])
        assert etat == "cable", (
            f"{name} est declare comme rebranche mais son cablage psi est "
            f"« {etat} » : "
            + {"absent": "aucun appel ne passe with_psi",
               "faux": "tous les appels passent with_psi=False EN DUR — psi "
                       "vaut zero, l'inventaire annonce l'inverse"}[etat])
        assert _passe_l_instantane_precedent(callers[name]), (
            f"{name} passe with_psi mais jamais d'instantane precedent "
            "utilisable : `prepare_qaoa_inputs` leve sur with_psi=True et "
            "prev_fields=None — le cablage est incomplet")


def test_the_unwired_scripts_really_run_with_psi_zero():
    """Le sens qui compte : quand un script est cable, ce test tombe.

    Il faut alors deplacer son nom de PSI_STILL_ZERO vers PSI_WIRED — ce qui
    force a constater le progres au lieu de l'oublier.
    """
    callers = _callers()
    for name in sorted(PSI_STILL_ZERO):
        etat = _etat_psi(callers[name])
        assert etat in ("absent", "faux"), (
            f"{name} rebranche desormais psi (etat « {etat} ») : le "
            "deplacer de PSI_STILL_ZERO vers PSI_WIRED dans cet inventaire, "
            "et consigner le changement dans docs/RESULTS.md")


def test_the_debt_is_not_silently_growing():
    """Un garde-fou grossier sur la taille de la dette."""
    assert len(PSI_STILL_ZERO) <= 6, (
        f"{len(PSI_STILL_ZERO)} scripts tournent encore sans psi ; la dette "
        "augmente au lieu de diminuer")
    assert PSI_WIRED, "aucun script ne rebranche psi : le cablage a disparu"


def test_le_detecteur_separe_la_presence_du_mot_cle_de_sa_valeur():
    """Epingle l'ancien comportement (D-155).

    Sur quelle entree l'ancien critere echouerait-il ? Aucune : il rendait
    `True` des qu'un mot-cle nomme `with_psi` existait. Les quatre cas
    ci-dessous se distinguent, et le troisieme — `with_psi=False` en dur —
    est celui qui laissait l'inventaire mentir."""
    def _calls(src):
        arbre = ast.parse(src)
        alias = _alias_locaux(arbre)
        return [n for n in ast.walk(arbre)
                if isinstance(n, ast.Call) and _called_name(n.func) in alias]

    ancien = lambda calls: any(kw.arg == "with_psi"          # noqa: E731
                               for c in calls for kw in c.keywords)

    absent = _calls("prepare_qaoa_inputs(a, b)")
    faux = _calls("prepare_qaoa_inputs(a, b, with_psi=False)")
    vrai = _calls("prepare_qaoa_inputs(a, b, with_psi=True, prev_fields=prev)")
    variable = _calls("p5.prepare_qaoa_inputs(a, b, with_psi=flag, prev_fields=prev)")

    assert _etat_psi(absent) == "absent"
    assert _etat_psi(faux) == "faux"
    assert _etat_psi(vrai) == "cable"
    assert _etat_psi(variable) == "cable"

    #  Le point du defaut : l'ancien critere confond les deux derniers avec
    #  le deuxieme.
    assert ancien(faux) and _etat_psi(faux) != "cable", (
        "l'ancien critere declarait `with_psi=False` comme un cablage ; le "
        "nouveau doit le refuser, sinon D-155 est rouvert")

    #  L'alias : sans lui, un script sortait de l'inventaire en silence.
    alias = _calls("from study.common.qaoa_inputs import prepare_qaoa_inputs as prep\n"
                   "prep(a, b, with_psi=True, prev_fields=prev)")
    assert len(alias) == 1 and _etat_psi(alias) == "cable", (
        "un appel par alias echappe au balayage")

    #  Un balayage qui ne peut pas lire doit crier, pas conclure (D-145).
    with pytest.raises(AssertionError, match="kwargs"):
        _etat_psi(_calls("prepare_qaoa_inputs(a, b, **opts)"))


def test_le_balayage_des_appelants_n_est_pas_vide():
    """Un balayage vide doit crier — y compris celui-ci.

    Mesure du 18 aout 2026 : 66 fichiers dans `study/`, 7 appelants.

    D-170 : le plancher a 40 ne detectait plus rien — meme quantite que
    D-167 (`STUDY_FILES`), un troisieme site independant qui la recalcule
    par `os.walk` plutot que `glob`."""
    fichiers = [os.path.join(d, n)
                for d, _s, ns in os.walk(_STUDY) if "__pycache__" not in d
                for n in ns if n.endswith(".py")]
    assert len(fichiers) >= 66, (
        f"{len(fichiers)} fichiers de study/ balayes ; 66 mesures a "
        "`3d4f095` (18 aout 2026)")
    assert len(_callers()) >= 7, (
        f"{len(_callers())} appelants trouves — mesure du 18 aout : 7. Le "
        "balayage a perdu une forme d'appel")
