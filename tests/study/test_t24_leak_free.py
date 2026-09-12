"""T24 — la comparaison sans fuite ne doit pas confondre regle et budget.

En mode `leak-free` les deux bras tournent a des seuils DIFFERENTS (le bras
QAOA au seuil classique regle du fold, le controle au seuil budget-apparie
force par `--matched-reference`). Sur `rotor` : 0.5864 contre 0.0969.

Mon propre code affichait « at the SAME operating point the classical arm
completed » quand le bras Q-HAS mourait. C'etait faux, et c'est le motif de
la campagne dans sa forme la plus pure : une ligne de sortie qui ne decrit
pas le calcul qu'elle accompagne.
"""
import ast
import json
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


def _study_file(name):
    """Chemin d'un module de study/ quel que soit son dossier d'hypothese."""
    for _d in ("pipeline", "h0_selection", "h1_solver", "h2b_prediction",
               "h3_representation", "h4_transfer", "closed_loop", "common"):
        _c = os.path.join(_REPO_ROOT, "study", _d, name)
        if os.path.exists(_c):
            return _c
    raise FileNotFoundError(name)

_HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.join(_REPO_ROOT, "study")
RESULTS = os.path.join(_REPO_ROOT, "results")

from closed_loop_leak_free_summary import analyse, frontier, frontier_at

FOLDS = ("ot", "kh", "rotor", "tearing")


def _have(fold):
    """Artefact leak-free COMPLET. Un point de reprise n'en est pas un.

    Sans cette distinction les tests ci-dessous se mettaient a echouer des
    qu'une execution etait en cours — et un test qui echoue parce qu'un
    calcul tourne n'apprend rien sur le code."""
    p = os.path.join(RESULTS, f"t22_unseen_leak-free_{fold}.json")
    if not os.path.exists(p):
        return False
    try:
        return json.load(open(p)).get("status") != "partial"
    except ValueError:
        return False


def test_no_claim_of_a_shared_operating_point():
    """t22 ne doit plus affirmer un point de fonctionnement commun.

    En mode leak-free il n'y en a pas ; l'affirmer effacait la seule
    reserve qui empeche de lire l'avortement de Q-HAS comme une
    instabilite propre au bras a budget egal."""
    src = open(_study_file("h4_unseen_conditions.py"),
               encoding="utf-8").read()
    assert "at the SAME operating point" not in src, (
        "t22 affirme un point de fonctionnement commun que le mode "
        "leak-free n'a pas")
    assert "DIFFERENT operating points" in src


@pytest.mark.parametrize("fold", FOLDS)
def test_thresholds_are_recorded_and_differ(fold):
    """Les deux seuils doivent etre lisibles dans le resume, pas devines."""
    if not _have(fold):
        pytest.skip("artefact leak-free absent")
    r = analyse(RESULTS, fold)
    assert r["qaoa_threshold"] is not None
    assert r["classical_threshold"] is not None
    # en leak-free ils different par construction ; si un jour ils
    # coincidaient, le drapeau doit le dire plutot que de rester implicite
    assert r["thresholds_match"] is not None


def test_frontier_refuses_to_extrapolate():
    """Hors de la plage balayee, aucun rapport — pas une valeur de bord.

    `np.interp` rend silencieusement l'extremite : un nombre d'apparence
    normale pour une comparaison qui n'existe pas. C'est precisement le
    motif traque."""
    front = ([0.2, 0.5, 0.9], [1.0, 0.5, 0.1])
    assert frontier_at(front, 0.05) is None      # sous la plage
    assert frontier_at(front, 0.95) is None      # au-dessus
    assert frontier_at(front, 0.5) == pytest.approx(0.5)


@pytest.mark.parametrize("fold", FOLDS)
def test_out_of_range_budgets_carry_no_ratio(fold):
    """Un budget hors plage ne doit jamais porter de rapport."""
    if not _have(fold):
        pytest.skip("artefact leak-free absent")
    r = analyse(RESULTS, fold)
    for cond, rec in r["conditions"].items():
        if rec.get("out_of_swept_range"):
            assert rec["ratio_vs_frontier"] is None, (
                f"{fold}/{cond}: budget hors de la frontiere balayee mais "
                f"un rapport est publie")


@pytest.mark.parametrize("fold", FOLDS)
def test_aborted_draws_excluded_from_the_leak_free_means(fold):
    if not _have(fold):
        pytest.skip("artefact leak-free absent")
    d = json.load(open(os.path.join(
        RESULTS, f"t22_unseen_leak-free_{fold}.json")))
    r = analyse(RESULTS, fold)
    for cond in ("canonical", "unseen"):
        runs = d["arms"]["qhas"].get(f"{cond}_runs", [])
        n_ok = sum(1 for x in runs if x["completed"])
        assert r["conditions"][cond]["n_completed"] == n_ok
        if n_ok == 0:
            assert "qhas_phys" not in r["conditions"][cond], (
                f"{fold}/{cond}: une moyenne est publiee alors qu'aucun "
                f"tirage n'a abouti")


def test_summary_script_declares_it_is_only_a_bound():
    """Le mode ne re-regle pas le bras QAOA : le module doit le dire.

    Sans cette reserve, « la fuite retiree, Q-HAS empire » se lirait comme
    le test definitif, alors que le reglage n'a pas ete refait."""
    src = open(_study_file("closed_loop_leak_free_summary.py"),
               encoding="utf-8").read()
    assert "BORNE" in src and "Optuna" in src


def test_partial_checkpoints_are_never_analysed():
    """Un point de reprise ne doit jamais etre lu comme un resultat.

    Les executions leak-free durent plus longtemps que la duree de vie du
    conteneur, d'ou l'ecriture d'un etat partiel apres chaque condition.
    Cette mesure de sauvegarde INTRODUIRAIT le motif de la campagne si un
    artefact partiel etait indiscernable d'un artefact complet : ses
    moyennes portent sur les tirages faits jusque-la."""
    src = open(_study_file("h4_unseen_conditions.py"),
               encoding="utf-8").read()
    assert '"partial"' in src and "partial_warning" in src, (
        "t22 ecrit des points de reprise sans les marquer")
    for consumer in ("closed_loop_leak_free_summary.py", "h4_transfer_summary.py"):
        cs = open(_study_file(consumer), encoding="utf-8").read()
        assert '== "partial"' in cs, (
            f"{consumer} ne filtre pas les artefacts partiels — il "
            f"publierait des moyennes sur une execution interrompue")


def test_a_partial_record_is_rejected_by_the_summary(tmp_path):
    """Verification fonctionnelle, pas seulement textuelle."""
    import json as _json
    import closed_loop_leak_free_summary as t24
    d = tmp_path / "res"
    d.mkdir()
    (d / "t22_unseen_leak-free_kh.json").write_text(_json.dumps({
        "fold": "kh", "status": "partial", "partial_stage": "qhas/canonical",
        "arms": {"qhas": {"canonical_runs": [
            {"phys_score": 1.0, "patch_ratio": 0.5, "completed": True}]}},
    }))
    r = t24.analyse(str(d), "kh")
    assert r["status"] == "partial"
    assert r["conditions"] == {}, (
        "un enregistrement partiel a produit des statistiques de condition")


def test_resume_reuses_only_matching_configurations():
    """Reprendre sous une AUTRE configuration melangerait des tirages
    incomparables. Le code doit refuser plutot que deviner."""
    src = open(_study_file("h4_unseen_conditions.py"),
               encoding="utf-8").read()
    for guard in ('prev.get("fold") == args.fold',
                  'prev.get("mode") == args.mode',
                  '"repeats"',
                  '"matched_reference"'):
        assert guard in src, (
            f"la reprise ne verifie pas {guard} — elle pourrait melanger "
            f"des tirages issus d'une autre configuration")
    assert 'prev.get("status") == "partial"' in src, (
        "la reprise devrait ne repartir que d'un point de sauvegarde")


def test_resume_is_recorded_never_silent():
    """Des tirages venus d'un autre processus doivent etre declares.

    C'est sans effet statistique (bras non deterministe, tirages i.i.d.)
    mais l'invisibilite serait le motif : un artefact qui ne dit pas d'ou
    viennent ses donnees."""
    src = open(_study_file("h4_unseen_conditions.py"),
               encoding="utf-8").read()
    assert "resumed_from_checkpoint" in src and "n_runs_resumed" in src


def test_resume_truncates_to_the_requested_count():
    """`--repeats 3` apres un point a 5 tirages ne doit pas en rendre 5."""
    src = open(_study_file("h4_unseen_conditions.py"),
               encoding="utf-8").read()
    assert "return got[:n]" in src, (
        "les tirages repris ne sont pas tronques a n")


# ---------------------------------------------------------------- T25
# La robustesse physique ne doit pas se compter sur des conditions qui ne
# deplacent pas la trajectoire.

def test_t25_verifies_the_condition_actually_moved_the_physics():
    """Une condition qui ne bouge rien ne teste rien.

    T25 existe pour montrer que la direction ne tient pas a un etat initial
    arbitraire. Une condition dont la trajectoire est identique a la
    canonique donnerait un resultat indiscernable d'un vrai test de
    robustesse — le motif de la campagne, applique au controle lui-meme."""
    src = open(_study_file("h4_physics_robustness.py"),
               encoding="utf-8").read()
    assert "dns_relative_shift" in src and "condition_is_weak" in src, (
        "t25 ne mesure pas le deplacement de trajectoire de ses conditions")
    # D-136. La 3e assertion de ce test cherchait la chaine
    # `not c.get("condition_is_weak")` dans le source. Retiree, pas
    # affaiblie : elle rougissait sur la reecriture EQUIVALENTE
    # `c.get("condition_is_weak", False)` (faux rouge sur un changement
    # voulu) et restait verte quand le filtre disparaissait du chemin
    # principal (faux vert, 26 passed) -- la chaine existe deux fois. Les
    # deux tests ci-dessous la remplacent, l'un par le comportement,
    # l'autre par l'AST.


# D-136. Le garde ci-dessus cherche une chaine qui existe DEUX fois dans
# `h4_physics_robustness.py` -- une par chemin de comptage -- donc une seule
# suffit a le satisfaire. Mesure : filtre retire du chemin PRINCIPAL (celui
# qui ecrit l'artefact publie), chaine intacte sur l'autre -> 26 passed.
# Les deux tests qui suivent mesurent le comportement au lieu de le lire.

def test_t25_recompute_excludes_the_vacuous_conditions_from_the_count(
        tmp_path, monkeypatch):
    """Une condition qui ne deplace pas la trajectoire sort du decompte.

    L'ENTREE QUI SEPARE : un artefact portant une condition vacue
    DECIDABLE et une condition franche, les deux avec un verdict rendu.
    Des conditions toutes franches rendraient « exclure » et « ne pas
    exclure » indiscernables — le decompte vaudrait 2 dans les deux cas.

    `--recompute` est le seul des deux chemins de comptage executable sans
    rejouer des heures de DNS ; l'autre (fin de `main`) est garde par la
    structure dans le test suivant.
    """
    import config
    import h4_physics_robustness as M

    def _cond(tag, weak):
        # Frontiere LOCALEMENT SAINE, pour que `frontier_verdict` rende un
        # nombre plutot qu'un refus : decroissante, ecart 0.10 (< 0.12) et
        # rapport 2.0 (< 5.0). Budget Q-HAS 0.40, dans [0.35, 0.45].
        return {"tag": tag, "condition_is_weak": weak,
                "classical_frontier": [
                    {"patch_ratio": 0.35, "phys_score": 0.4,
                     "completed": True},
                    {"patch_ratio": 0.45, "phys_score": 0.2,
                     "completed": True}],
                "qhas_runs": [
                    {"patch_ratio": 0.40, "phys_score": 0.5,
                     "completed": True}]}

    op = tmp_path / "t25_physics_robustness_ot.json"
    op.write_text(json.dumps({"fold": "ot", "conditions": [
        _cond("vacuous", True), _cond("franche", False)]}))
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["h4_physics_robustness",
                                      "--fold", "ot", "--recompute"])
    with pytest.raises(SystemExit) as exc:
        M.main()
    assert exc.value.code == 0

    d = json.loads(op.read_text())
    # Les DEUX conditions sont decidables : le verdict est rendu pour
    # chacune. Seule l'exclusion des vacues peut donc separer 1 de 2.
    assert [c["qhas_worse"] for c in d["conditions"]] == [True, True], (
        "champ d'essai invalide : les deux conditions doivent etre "
        "decidables pour que le decompte mesure l'exclusion")
    assert d["n_decidable"] == 1, (
        "t25 compte une condition vacue dans sa direction : le decompte "
        "'k/n conditions decidables' est gonfle par des conditions qui ne "
        "peuvent ni confirmer ni infirmer")
    assert d["n_qhas_worse"] == 1


def test_t25_both_direction_counts_exclude_the_vacuous_conditions():
    """Les DEUX chemins de comptage portent le filtre, pas seulement un.

    L'AST delimite par la STRUCTURE — la liaison du nom `dec` a une
    comprehension — jamais par une distance ni par un comptage
    d'occurrences de texte. Une reecriture equivalente
    (`c.get("condition_is_weak", False)`) le laisse vert, a raison ; le
    retrait du filtre sur l'un OU l'autre chemin le fait rougir.
    """
    tree = ast.parse(open(_study_file("h4_physics_robustness.py"),
                          encoding="utf-8").read())
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.ListComp):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "dec"
                   for t in node.targets):
            continue
        keys = {c.args[0].value
                for gen in node.value.generators for cond in gen.ifs
                for c in ast.walk(cond)
                if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute) and c.func.attr == "get"
                and c.args and isinstance(c.args[0], ast.Constant)}
        sites.append((node.lineno, keys))

    # Un balayage vide doit crier : le nombre mesure est ecrit ici.
    assert len(sites) >= 2, (
        f"{len(sites)} decompte(s) de direction trouve(s), 2 attendus "
        f"(--recompute et la fin de main) — balayage vide ou renomme")
    for lineno, keys in sites:
        assert "condition_is_weak" in keys, (
            f"le decompte de direction ligne {lineno} compte des conditions "
            f"vacues")
        assert "qhas_worse" in keys, (
            f"le decompte ligne {lineno} ne filtre plus sur un verdict rendu")


# D-137. `test_t25_never_extrapolates_its_frontier` cherchait la garde
# `xs[0] <= qp <= xs[-1]` dans le SOURCE. Retiree, pas affaiblie : elle
# restait verte quand la garde etait neutralisee en code mort (faux vert,
# 28 passed, un budget hors plage rendant alors 0.700) et rougissait sur la
# reecriture EQUIVALENTE `min(xs) <= qp <= max(xs)` (faux rouge sur un
# changement voulu). Le test ci-dessous la remplace par le comportement.

def test_t25_refuses_a_budget_outside_the_swept_frontier():
    """Hors de la plage balayee, `frontier_verdict` refuse — il n'extrapole pas.

    D-136/D-137 : le test ci-dessus cherche la garde dans le SOURCE.
    Neutralisee en code mort (chaine intacte), elle rend sur la frontiere
    ci-dessous :

        budget 0.20 -> 0.700        budget 0.05 -> 1.000

    finis, positifs, dans l'intervalle d'un `phys_score`, sans refus ni
    plantage. Et le biais a un sens : l'erreur classique inventee CROIT
    quand le budget decroit, donc `ratio_vs_frontier = qe / ref` diminue
    et le bras Q-HAS parait meilleur qu'il n'est.

    L'ENTREE QUI SEPARE est le budget SOUS la plage. Au-dessus, la garde
    retiree fait lever `StopIteration` — un plantage, qui se voit. En
    dessous, elle rend le nombre plausible, qui ne se voit pas. Les deux
    sont couverts ici ; seul le second est silencieux.

    Les nombres mesures sont ecrits pour qu'une derive se voie.
    """
    from h4_physics_robustness import frontier_verdict
    f = lambda pts: [{"patch_ratio": p, "phys_score": e, "completed": True}
                     for p, e in pts]
    pts = f([(0.35, 0.4), (0.45, 0.2)])

    # temoin : a l'interieur, le verdict est rendu -- sans quoi le test
    # ci-dessous passerait sur une frontiere qui refuse tout.
    ref, why = frontier_verdict(pts, 0.40, 1.0)
    assert ref == pytest.approx(0.3) and why is None

    for qp, extrapole in ((0.20, 0.700), (0.05, 1.000)):
        ref, why = frontier_verdict(pts, qp, 1.0)
        assert ref is None, (
            f"budget {qp} SOUS la plage balayee [0.35, 0.45] : "
            f"extrapolation rendue au lieu d'un refus "
            f"(la garde retiree rend {extrapole:.3f})")
        assert "outside the swept range" in why

    ref, why = frontier_verdict(pts, 0.50, 1.0)
    assert ref is None and "outside the swept range" in why


def test_t25_refuses_a_non_monotone_bracketing_interval():
    """Interpoler « l'erreur atteignable » sur une frontiere non monotone
    rend un nombre d'apparence normale qui ne mesure rien.

    Sur `tearing_b`, raffiner de 0.625 a 0.874 fait passer l'erreur de
    0.012 a 1.289 — trente fois PIRE. `np.interp` y repondait sans
    broncher, et 1.28x avait deja ete affiche comme un resultat."""
    from h4_physics_robustness import frontier_verdict
    f = lambda pts: [{"patch_ratio": p, "phys_score": e, "completed": True}
                     for p, e in pts]
    # anti-monotone dans l'intervalle encadrant -> refus motive
    ref, why = frontier_verdict(f([(0.625, 0.012), (0.874, 1.289)]), 0.75, 1.0)
    assert ref is None and "not monotone" in why.lower()
    # trop raide -> refus motive (ecart de budget volontairement etroit
    # pour isoler le critere de raideur du critere de convergence)
    ref, why = frontier_verdict(f([(0.35, 10.0), (0.45, 0.1)]), 0.4, 1.0)
    assert ref is None and "steep" in why.lower()
    # bissection non convergee -> refus motive
    ref, why = frontier_verdict(f([(0.30, 0.4), (0.60, 0.2)]), 0.4, 1.0)
    assert ref is None and "did not converge" in why.lower()
    # sain -> verdict rendu
    ref, why = frontier_verdict(f([(0.35, 0.4), (0.45, 0.2)]), 0.4, 1.0)
    assert ref is not None and why is None
    assert ref == pytest.approx(0.3)


def test_t25_can_recompute_verdicts_without_simulating():
    """Quand la regle de verdict change, les tirages restent valables :
    seule leur lecture evolue. Sans ce mode il faudrait re-simuler des
    heures pour corriger une interpretation."""
    src = open(_study_file("h4_physics_robustness.py"),
               encoding="utf-8").read()
    assert "--recompute" in src and "verdicts_recomputed" in src


def test_t25_marks_reynolds_as_not_an_ic_variation():
    """`ot` n'a aucun parametre : son levier Reynolds n'est pas une
    variation de condition initiale et ne doit pas etre compte comme telle."""
    src = open(_study_file("h4_physics_robustness.py"),
               encoding="utf-8").read()
    assert 'is_ic_variation' in src


def test_t25_rng_override_changes_the_draw_and_restores_numpy():
    """La seule vraie graine de la suite doit etre reellement substituee,
    et numpy rendu intact — sinon la substitution contaminerait tout ce qui
    suit dans le meme processus."""
    import numpy as np
    from h4_physics_robustness import rng_override
    base = np.random.default_rng(42).standard_normal(8)
    with rng_override(7):
        alt = np.random.default_rng(42).standard_normal(8)
    after = np.random.default_rng(42).standard_normal(8)
    assert not np.allclose(base, alt), "la substitution de graine n'a rien changé"
    assert np.allclose(base, after), "numpy n'a pas ete restaure"
