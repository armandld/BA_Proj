"""D-51 rouvert — aucun chemin de `study/` ne PRODUIT `K_xpoint`.

`test_xpoint_term_absent_from_study.py`, voisin de ce fichier, épingle le
**consommateur** : il fabrique lui-même un `hamilt_params` portant
`"K_xpoint"` et regarde qui le lit. Ce fichier-ci épingle le **producteur**,
que personne n'avait interrogé : la clé que ces contrôles fabriquent, le
chemin réel ne l'écrit jamais.

Mesuré, `N=32 Re=400`, champ MHD analytique, mappeur v1 : passer
`advanced_anomalies_enabled=True` à `create_period_hamiltonian` rend un
opérateur **identique bit à bit** à `False` — 48 termes des deux côtés,
`max|coeff(H_on − H_off)| = 0,0` — parce que `prepare_qaoa_inputs` appelle
`compute_coefficients` sans le kwarg, qui vaut `False` par défaut.

**Épinglage de déviation, pas verrouillage d'une correction.** Rien n'est
corrigé : activer le producteur changerait l'hamiltonien de toutes les
mesures de `study/`, donc des nombres publiés. Ces tests font échouer la
suite le jour où quelqu'un active le producteur — ce qui est précisément le
moment où il faut remesurer phase 4, T13 et T26, pas glisser.

Sur quelle entrée chacun échoue est écrit dans sa docstring.
"""
import ast
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src"), _REPO_ROOT] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h3_representation", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _champ_mhd(N=32):
    """Champ analytique portant une nappe de courant — il SÉPARE.

    `K_xpoint` dérive de `det(J_B) < 0` : un champ sans point X rendrait
    zéro partout et ne distinguerait pas un producteur actif d'un
    producteur muet. Celui-ci rend un terme non nul dès `dim = 4`.
    """
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return (np.sin(X) * np.cos(Y), -np.cos(X) * np.sin(Y),
            np.sin(2 * Y), np.sin(X))


def test_prepare_qaoa_inputs_ne_produit_pas_la_cle():
    """Échoue le jour où `prepare_qaoa_inputs` demande le terme de point X.

    Sur le champ ci-dessus à `dim = 4`, un producteur actif écrit un
    `K_xpoint` de `max|.| = 0,0885` (2 cellules sur 16). Le producteur
    courant n'écrit pas la clé du tout.
    """
    from qaoa_inputs import prepare_qaoa_inputs

    vx, vy, Bx, By = _champ_mhd()
    for use_v2 in (False, True):
        _, hp, _ = prepare_qaoa_inputs(vx, vy, Bx, By, 32, 4, 400,
                                       use_v2=use_v2)
        assert "K_xpoint" not in hp, (
            f"use_v2={use_v2} : `prepare_qaoa_inputs` produit desormais "
            "`K_xpoint`. C'est la levee de D-51, pas un detail : "
            "l'hamiltonien de TOUTES les mesures de study/ change. "
            "Remesurer phase 4, T13 et T26 AVANT de mettre ce test a jour.")


def test_le_drapeau_du_consommateur_est_sans_effet_sur_le_chemin_reel():
    """L'opérateur est identique bit à bit avec et sans le drapeau.

    Échoue si le producteur est activé (l'opérateur gagne alors 2 termes à
    `dim = 4`, écart `0,0885`), ou si `create_period_hamiltonian` se met à
    fabriquer un terme de point X en l'absence de la clé.
    """
    from qaoa_inputs import prepare_qaoa_inputs
    from VQA.cost_hamiltonian import create_period_hamiltonian

    vx, vy, Bx, By = _champ_mhd()
    _, hp, _ = prepare_qaoa_inputs(vx, vy, Bx, By, 32, 4, 400, use_v2=False)

    h_off = create_period_hamiltonian(hp, 4, advanced_anomalies_enabled=False)
    h_on = create_period_hamiltonian(hp, 4, advanced_anomalies_enabled=True)

    assert len(h_on) == len(h_off) == 48
    ecart = (h_on - h_off).simplify()
    ecart_max = float(np.max(np.abs(ecart.coeffs))) if len(ecart.coeffs) else 0.0
    assert ecart_max == 0.0, (
        f"max|coeff(H_on - H_off)| = {ecart_max} au lieu de 0,0 : le terme "
        "de point X entre desormais dans l'hamiltonien de study/. Voir "
        "D-51 dans docs/DEFAUTS.md avant de toucher a ce test.")


def _sites_compute_coefficients():
    """Chaque appel `.compute_coefficients(...)` de `study/`, par l'AST.

    Par l'AST et non par une chaîne de caractères : un test qui grep le
    source teste sa mise en forme (règle de `VIGIL.md`, trois fois payée
    ici).
    """
    sites = []
    for racine, _, fichiers in os.walk(os.path.join(_REPO_ROOT, "study")):
        if "__pycache__" in racine:
            continue
        for nom in sorted(fichiers):
            if not nom.endswith(".py"):
                continue
            chemin = os.path.join(racine, nom)
            with open(chemin, encoding="utf-8") as fh:
                arbre = ast.parse(fh.read(), filename=chemin)
            for noeud in ast.walk(arbre):
                if (isinstance(noeud, ast.Call)
                        and isinstance(noeud.func, ast.Attribute)
                        and noeud.func.attr == "compute_coefficients"):
                    passe = any(kw.arg == "advanced_anomalies_enabled"
                                for kw in noeud.keywords)
                    sites.append((os.path.relpath(chemin, _REPO_ROOT),
                                  noeud.lineno, passe))
    return sites


def test_un_seul_site_de_study_demande_les_anomalies_avancees():
    """Le compte des producteurs actifs, épinglé.

    Sept sites appellent `compute_coefficients` ; un seul passe le kwarg, et
    c'est `preflight_coefficients.py`, un diagnostic — aucun chemin de
    mesure. Échoue dès qu'un site change de côté, dans un sens comme dans
    l'autre.
    """
    sites = _sites_compute_coefficients()
    assert len(sites) >= 7, (
        f"{len(sites)} sites trouves, au moins 7 attendus : le balayage AST "
        "ne trouve plus ce qu'il comptait. Un balayage vide doit crier.")

    actifs = sorted(f"{f}:{l}" for f, l, passe in sites if passe)
    assert actifs == ["study/common/preflight_coefficients.py:64"], (
        f"producteurs demandant le terme de point X : {actifs}. Un seul est "
        "attendu, et c'est un diagnostic. Tout changement ici leve ou "
        "aggrave D-51 — remesurer, ne pas mettre ce test a jour.")
