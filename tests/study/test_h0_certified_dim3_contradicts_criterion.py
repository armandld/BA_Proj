"""D-53 : la seule taille a la fois CERTIFIEE et NON DEGENEREE jamais
executee contredit le critere d'acceptation de H0, et son resultat n'est
ecrit dans aucun document.

`RESULTS.md` T11 publie `dim = 2` (8 qubits) et conclut « Pre-registered
rule fires: quantum optimisation is not the source of any gain » ; `dim = 2`
est aussi la taille ou l'etat fondamental exact vaut « raffiner partout »
(D-45 / D-47), donc ou l'accord entre solveurs est acquis d'avance.

Trois artefacts `dim = 3` (18 qubits, optimum enumere exactement) existent
dans `results/` et disent l'inverse. Ces tests EPINGLENT leurs nombres. Ils
sont deliberement des tests de DEVIATION, pas de regression : ils ne
pouvaient pas echouer sur un commit anterieur, ils echouent le jour ou l'un
des trois artefacts est remplace ou ou le critere est requalifie -- ce qui
est precisement le moment ou D-53 doit etre relu.

Aucun nombre publie n'en depend : `aggregate_master_table.collect` lit
`h0_optimiser_equivalence_N{N}_dim{dim}` a `N=256, dim=2`.
"""
import io
import contextlib
import os
import re
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from h0_optimiser_equivalence import (            # noqa: E402
    MIN_HIT, MIN_MASK_MATCH, check_expected_behaviour,
)

RESULTS = os.path.join(_REPO_ROOT, "results")
DIM3 = os.path.join(RESULTS, "h0_optimiser_equivalence_N96_dim3.npz")
DIM3_KOPT = os.path.join(RESULTS, "h0_optimiser_equivalence_N96_dim3_scalekopt.npz")
DIM2 = os.path.join(RESULTS, "h0_optimiser_equivalence_N256_dim2.npz")


def _summary(path):
    """(summary, solvers, diag_flags) reconstruits depuis un artefact, dans
    la forme exacte que `main()` passe au critere."""
    if not os.path.exists(path):
        pytest.skip(f"artefact absent : {os.path.basename(path)}")
    d = np.load(path, allow_pickle=True)
    sol = d["solver"]
    solvers = list(dict.fromkeys(sol.tolist()))
    summary = {
        s: dict(hit=float(np.mean(d["hit"][sol == s])),
                match=float(np.mean(d["match"][sol == s])),
                agree=float(np.mean(d["agree"][sol == s])),
                gap=float(np.mean(d["E_gap"][sol == s])),
                f1=float(np.mean(d["f1"][sol == s])),
                wall=0.0)
        for s in solvers}
    return summary, solvers, [bool(d["diagonal_all"])]


def test_dim2_artifact_satisfies_the_criterion():
    """La reference du critere, verifiee et non supposee : a la taille
    deployee il passe bien."""
    summary, solvers, diag = _summary(DIM2)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        check_expected_behaviour(summary, solvers, diag)
    assert "[ACCEPTANCE]" in buf.getvalue()


def test_dim3_artifact_makes_the_criterion_raise():
    """Le coeur de D-53 : sur la seule taille certifiee non degeneree, le
    critere du module leve — et son propre message dit « H0 redevient
    plausible »."""
    summary, solvers, diag = _summary(DIM3)
    with pytest.raises(AssertionError) as exc:
        check_expected_behaviour(summary, solvers, diag)
    assert "redevient plausible" in str(exc.value)
    assert "qaoa_p1" in str(exc.value)


def test_dim3_qaoa_lands_further_from_the_optimum_than_the_classical_rule():
    """Mesure epinglee : le QAOA n'est pas seulement sous le critere, il est
    sous la regle classique dont il part."""
    summary, _, _ = _summary(DIM3)
    classical = summary["classical_init"]["match"]
    assert classical == pytest.approx(0.500, abs=5e-3)
    for depth in (1, 2, 3, 4, 5, 6):
        m = summary[f"qaoa_p{depth}"]["match"]
        assert m < classical, (
            f"qaoa_p{depth} match={m:.3f} n'est plus sous la regle classique "
            f"({classical:.3f}) : D-53 est a relire")
    assert summary["qaoa_p1"]["match"] == pytest.approx(0.15625, abs=5e-3)
    assert summary["qaoa_p6"]["match"] == pytest.approx(0.21875, abs=5e-3)


def test_dim3_qaoa_hit_rates_are_pinned():
    """Les taux eux-memes, pour qu'une derive se voie. 0,062–0,156 contre le
    1,000 exige."""
    summary, _, _ = _summary(DIM3)
    expected = {"qaoa_p1": 0.15625, "qaoa_p2": 0.15625, "qaoa_p3": 0.125,
                "qaoa_p4": 0.09375, "qaoa_p5": 0.0625, "qaoa_p6": 0.0625}
    for name, value in expected.items():
        assert summary[name]["hit"] == pytest.approx(value, abs=5e-3)
        assert summary[name]["hit"] < MIN_HIT
        assert summary[name]["match"] < MIN_MASK_MATCH


def test_scaled_budget_does_not_rescue_the_qaoa_arm():
    """L'objection « c'est un budget COBYLA trop court » est celle que
    `--scale-kopt` existe pour lever : avec le budget proportionnel a p, le
    bras QAOA tombe a 0,000 sur les quatre profondeurs, greedy restant a
    1,000."""
    summary, solvers, diag = _summary(DIM3_KOPT)
    for s in [s for s in solvers if s.startswith("qaoa")]:
        assert summary[s]["hit"] == pytest.approx(0.0, abs=1e-9)
    assert summary["greedy"]["hit"] == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(AssertionError, match="redevient plausible"):
        check_expected_behaviour(summary, solvers, diag)


# ══════════════════════════════════════════════════════════════════
#  D-146 — la decision de ne pas corriger doit rester ECRITE
# ══════════════════════════════════════════════════════════════════
#
# L'ancien garde faisait `assert "D-53" in docs/DEFAUTS.md`. Quatre
# caracteres, cherches dans 1 361 lignes : n'importe quelle reference
# croisee le satisfait. Ce n'est pas theorique — la decision a DEJA quitte
# `DEFAUTS.md` (D-53 est clos, son entree vit sous `# D-53` dans
# `RESULTS.md`) et le garde n'a rien vu, parce qu'un tableau de synthese
# de l'entree D-132 porte encore une ligne `| D-53 | ... |`.
#
# Ce que le garde doit distinguer : une ENTREE (un titre qui nomme le
# defaut, ou une ligne du registre des corriges) d'une REFERENCE CROISEE
# (le numero cite dans la prose d'une autre entree). Et il doit exiger que
# l'entree porte encore les nombres qui la rendent lisible : un titre nu ne
# consigne aucune decision.
#
# Mesure (voir `RESULTS.md`, ligne D-146) :
#   A' — les deux sections `# ...D-53...` retirees de `RESULTS.md`, toutes
#        les references croisees laissees en place, dans les deux fichiers
#        ancien garde : 6 passed  |  nouveau : 1 failed
#   B  — l'entree DEPLACEE de `RESULTS.md` vers `DEFAUTS.md` en `## D-53`
#        (changement voulu, la decision reste ecrite)
#        nouveau : pas de faux rouge

_REGISTRES = ("DEFAUTS.md", "RESULTS.md")

# Le titre du registre des defauts corriges : la seule table dont une ligne
# `| D-N | ... |` EST l'entree du defaut N. Ailleurs, une telle ligne est un
# tableau de synthese range dans l'entree d'un AUTRE defaut.
_TITRE_DU_REGISTRE = re.compile(r"d[ée]fauts\s+corrig[ée]s", re.IGNORECASE)


def _entrees_du_defaut(texte, numero):
    """Les entrees du defaut `numero` dans un document markdown.

    Rend une liste de corps (str). Une entree est :

      - une SECTION dont le titre nomme le defaut (`# D-53 — ...`), corps
        = jusqu'au titre suivant de niveau inferieur ou egal, donc les
        sous-titres de la section en font partie ; ou
      - une LIGNE DU REGISTRE (`| D-53 | ... |`) placee sous le titre du
        registre des defauts corriges.

    N'est PAS une entree : le numero cite dans la prose, ni une ligne
    `| D-53 | ... |` rangee sous le titre d'un autre defaut.
    """
    jeton = re.compile(r"\bD-%d\b" % numero)
    titre = re.compile(r"^(#{1,6})\s+(.*)$")
    ligne_registre = re.compile(r"^\|\s*D-%d\s*\|" % numero)

    lignes = texte.split("\n")
    # (niveau, titre) de la section courante, du plus englobant au plus fin
    pile = []
    entrees = []
    ouverte = None          # (niveau, [lignes de corps])

    for ligne in lignes:
        m = titre.match(ligne)
        if m:
            niveau, intitule = len(m.group(1)), m.group(2)
            if ouverte is not None and niveau <= ouverte[0]:
                entrees.append("\n".join(ouverte[1]))
                ouverte = None
            while pile and pile[-1][0] >= niveau:
                pile.pop()
            pile.append((niveau, intitule))
            if jeton.search(intitule) and ouverte is None:
                ouverte = (niveau, [ligne])
            continue
        if ouverte is not None:
            ouverte[1].append(ligne)
        if ligne_registre.match(ligne) and any(
                _TITRE_DU_REGISTRE.search(t) for _, t in pile):
            entrees.append(ligne)

    if ouverte is not None:
        entrees.append("\n".join(ouverte[1]))
    return entrees


# Les nombres SANS lesquels l'entree ne consigne plus rien : le taux
# atteint par le bras QAOA a dim=3, sa borne basse, et le 1,000 exige.
# Virgule ou point : la mise en forme n'est pas ce qui est garde.
_NOMBRES_DE_D53 = (r"0[.,]156", r"0[.,]062", r"1[.,]000")


def test_la_decision_de_ne_pas_corriger_D53_reste_ecrite():
    """`VIGIL.md` : une deviation connue mais non consignee se fait
    recorriger par erreur ; toute decision de ne pas corriger s'ecrit, et un
    test verifie que la mention y reste.

    Volontairement pas « dans DEFAUTS.md » : un defaut clos SORT de
    `DEFAUTS.md` et entre dans `RESULTS.md` — c'est la regle des six
    documents, et l'exiger dans un fichier nomme ferait rougir le test sur
    ce mouvement-la, qui est voulu. On exige l'entree dans l'un OU l'autre
    registre, et on exige qu'elle porte encore ses nombres.
    """
    trouvees = []
    for nom in _REGISTRES:
        with open(os.path.join(_REPO_ROOT, "docs", nom), encoding="utf-8") as fh:
            trouvees += [(nom, corps) for corps in _entrees_du_defaut(fh.read(), 53)]

    assert trouvees, (
        "aucune ENTREE D-53 dans " + " ni ".join(_REGISTRES) + " : la decision "
        "de ne pas corriger a disparu des registres. Sans elle, le critere "
        "MIN_HIT=1.0 se relit comme valide a toute taille. (Une reference "
        "croisee dans la prose d'un autre defaut ne compte pas : c'est "
        "exactement ce qui a laisse l'ancien garde vert, D-146.)")

    porteuses = [nom for nom, corps in trouvees
                 if all(re.search(n, corps) for n in _NOMBRES_DE_D53)]
    assert porteuses, (
        f"{len(trouvees)} entree(s) D-53 trouvee(s), aucune ne porte encore "
        f"les trois nombres qui la rendent lisible "
        f"({', '.join(_NOMBRES_DE_D53)}) : un titre nu ne consigne pas une "
        f"decision.")


def test_le_detecteur_dentree_ne_confond_pas_une_reference_croisee():
    """Auto-test du detecteur — sans lui, on ne saurait pas dire sur quelle
    entree le garde ci-dessus echouerait.

    Les trois premiers cas sont ceux qui laissaient l'ancien garde vert.
    """
    croisee = (
        "## D-132 — le bras QAOA ne classe plus\n\n"
        "Voir la ligne D-53 de RESULTS.md.\n\n"
        "| | |\n|---|---|\n"
        "| D-53 | optimum atteint 0,062-0,156 contre 1,000 exige |\n")
    assert "D-53" in croisee                     # l'ancien garde : VERT
    assert _entrees_du_defaut(croisee, 53) == [], (
        "une reference croisee, et une ligne de synthese rangee sous le "
        "titre d'un AUTRE defaut, ne sont pas des entrees")

    prose = "D-53 est le plus lourd de la liste et se lit en premier.\n"
    assert "D-53" in prose                       # l'ancien garde : VERT
    assert _entrees_du_defaut(prose, 53) == []

    section = ("# D-53 — la seule taille certifiee non degeneree\n\n"
               "Le QAOA atteint 0,156 puis 0,062 la ou 1,000 est exige.\n\n"
               "## Ce que dit dim = 3\n\nsuite de la meme section\n\n"
               "# D-54 — autre chose\n\ncorps de D-54\n")
    corps = _entrees_du_defaut(section, 53)
    assert len(corps) == 1, corps
    assert "suite de la meme section" in corps[0], (
        "les sous-titres appartiennent a la section : le corps doit aller "
        "jusqu'au titre suivant de niveau INFERIEUR ou egal")
    assert "corps de D-54" not in corps[0]

    registre = ("## Les 64 defauts corriges\n\n"
                "| # | ce qui etait faux | verifier |\n|---|---|---|\n"
                "| D-53 | 0,156 -> 0,062 contre 1,000 exige | pytest ... |\n")
    assert len(_entrees_du_defaut(registre, 53)) == 1, (
        "une ligne du registre des corriges EST une entree")

    assert _entrees_du_defaut(section, 5) == [], (
        "`D-5` ne doit pas etre trouve dans `D-53` : le jeton est borne")
