"""Audit de contrat de `src/recompute_lambda_scores.py`.

Ce script relit les bases Optuna gelees et recalcule le score combine avec
un autre `lambda_cost`. Il ne produit aucun nombre publie -- il decide
comment on LIT les nombres publies, ce qui revient au meme si personne ne
le verifie.

Les cinq questions, dans l'ordre ou elles ont paye ici :

  Q4 (deux chemins censes coincider) -- `recompute_score` a `lambda = 0.4`,
     la valeur de la campagne, doit rendre exactement `trial.value`. Mesure
     sur les 303 essais reels : ecart max 2.2e-16, classement identique.

  Q5 (un test traverse-t-il cette configuration ?) -- le chemin d'ECHEC
     n'etait traverse par rien. Il rendait 0. Voir D-49.

  Q3 (consomme-t-elle ce que sa signature annonce ?) -- `recompute_score`
     et `build_trial_table` detectent les scenarios par deux chemins
     differents. Piege arme, non declenche sur les donnees gelees.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO, "src")
_DB_DIR = os.path.join(_REPO, "results", "hyperparams", "optuna_studies")
_SCRIPT = os.path.join(_SRC, "recompute_lambda_scores.py")

#: `LAMBDA_COST_SOFT` de `train_hyperparams` — la valeur avec laquelle les
#: bases gelees ont ete produites.
LAMBDA_CAMPAGNE = 0.4

#: Les deux seules bases gelees qui contiennent une etude. Les huit autres
#: sont vides — fait deja etabli dans `results/hyperparams/PROVENANCE.md`.
BASES_REELLES = [
    ("classical_v2_phase1", 125),
    ("q_has_v2_phase1", 178),
]


def _charge(nom):
    optuna = pytest.importorskip("optuna")
    chemin = os.path.join(_DB_DIR, f"{nom}.db")
    if not os.path.exists(chemin):
        pytest.skip(f"base gelee absente : {chemin}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    etude = optuna.load_study(
        study_name=nom, storage=f"sqlite:///{os.path.abspath(chemin)}")
    finis = [t for t in etude.trials
             if t.state == optuna.trial.TrialState.COMPLETE
             and t.value is not None and t.value < float("inf")]
    return finis


@pytest.fixture(scope="module")
def module_rescore():
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    return pytest.importorskip("recompute_lambda_scores")


# ══════════════════════════════════════════════════════════════════════
#  Q4 — le recalcul doit reproduire le score d'origine a lambda egal
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("nom,attendus", BASES_REELLES)
def test_le_rescore_reproduit_le_score_dorigine(module_rescore, nom, attendus):
    """A `lambda = 0.4`, `recompute_score` doit rendre `trial.value`.

    C'est le seul controle qui atteste que le script recalcule bien LA
    fonction objectif de la campagne, et pas une autre qui lui ressemble.
    Un facteur oublie, une normalisation differente, une moyenne sur le
    mauvais ensemble de scenarios : tout cela se verrait ici, et nulle
    part ailleurs.
    """
    finis = _charge(nom)
    assert len(finis) == attendus, (
        f"{nom} : {len(finis)} essais finis, {attendus} attendus — la base "
        f"gelee a change, ou le filtre COMPLETE/fini a change de sens")

    ecarts = np.array([
        abs(float(module_rescore.recompute_score(t, LAMBDA_CAMPAGNE)) - float(t.value))
        for t in finis
    ])
    assert ecarts.max() < 1e-12, (
        f"{nom} : ecart max {ecarts.max():.3e} entre le score recalcule a "
        f"lambda={LAMBDA_CAMPAGNE} et le score d'origine. Mesure de "
        f"reference : 2.2e-16")


@pytest.mark.parametrize("nom,_", BASES_REELLES)
def test_le_rescore_preserve_le_classement_a_lambda_egal(module_rescore, nom, _):
    """Meme controle, sur la grandeur qui sert reellement : l'ORDRE.

    Les conclusions de ce depot reposent sur des classements, pas sur des
    valeurs — le bras QAOA n'est pas deterministe en valeur.
    """
    finis = _charge(nom)
    ordre_origine = [t.number for t in sorted(finis, key=lambda x: x.value)]
    ordre_recalcule = [
        t.number for t in
        sorted(finis, key=lambda x: module_rescore.recompute_score(x, LAMBDA_CAMPAGNE))
    ]
    assert ordre_origine == ordre_recalcule, (
        f"{nom} : le classement change alors que lambda est inchange")


def test_lambda_zero_ne_garde_que_la_physique(module_rescore):
    """A `lambda = 0`, le score doit valoir la physique seule.

    Champ qui SEPARE : a lambda=0 le terme de cout disparait, donc
    `new_score == phys`. Toute ponderation residuelle apparait ici.
    """
    finis = _charge("classical_v2_phase1")
    cles = module_rescore._detect_scenario_keys(finis)
    assert cles, "aucune cle de scenario detectee — le balayage serait vide"

    for t in finis[:20]:
        phys = [t.user_attrs[f"phys_{k}"] for k in cles
                if f"phys_{k}" in t.user_attrs]
        attendu = float(np.mean(phys))
        obtenu = float(module_rescore.recompute_score(t, 0.0))
        assert obtenu == pytest.approx(attendu, abs=1e-12), (
            f"essai {t.number} : a lambda=0 le score vaut {obtenu}, la "
            f"physique seule vaut {attendu}")


def test_lambda_croissant_deplace_le_score_vers_le_cout(module_rescore):
    """Le sens du compromis doit etre le bon.

    Sur un essai dont `patch > phys`, augmenter lambda doit AUGMENTER le
    score ; sur un essai dont `patch < phys`, le diminuer. Un signe
    inverse rendrait le rescore exactement contraire a son propos.
    """
    finis = _charge("classical_v2_phase1")
    cles = module_rescore._detect_scenario_keys(finis)

    vus_au_dessus = vus_en_dessous = 0
    for t in finis:
        phys = np.mean([t.user_attrs[f"phys_{k}"] for k in cles])
        patch = np.mean([t.user_attrs[f"patch_{k}"] for k in cles])
        bas = module_rescore.recompute_score(t, 0.1)
        haut = module_rescore.recompute_score(t, 2.0)
        if patch > phys:
            assert haut > bas, (
                f"essai {t.number} : patch={patch:.4f} > phys={phys:.4f}, "
                f"le score devrait monter avec lambda ({bas:.4f} -> {haut:.4f})")
            vus_au_dessus += 1
        elif patch < phys:
            assert haut < bas, (
                f"essai {t.number} : patch={patch:.4f} < phys={phys:.4f}, "
                f"le score devrait baisser avec lambda ({bas:.4f} -> {haut:.4f})")
            vus_en_dessous += 1

    # Un balayage vide doit crier : les deux cas doivent exister.
    assert vus_au_dessus > 0 and vus_en_dessous > 0, (
        f"balayage degenere — {vus_au_dessus} essais au-dessus, "
        f"{vus_en_dessous} en dessous : le test ne separe rien")


# ══════════════════════════════════════════════════════════════════════
#  D-49 — le chemin d'echec rendait 0
# ══════════════════════════════════════════════════════════════════════

def _lance(args):
    return subprocess.run([sys.executable, _SCRIPT] + args,
                          capture_output=True, text=True, timeout=900)


def test_d49_base_introuvable_sort_en_erreur(tmp_path):
    """Une base qui n'existe pas doit rendre un code non nul.

    Avant D-49, un `except Exception` unique enveloppait tout le corps de
    `main`, imprimait « Erreur lors du chargement » et laissait la
    fonction rendre la main : **code de retour 0**. Un script de campagne
    qui teste `$?` voyait un succes.
    """
    r = _lance(["--db-path", str(tmp_path / "nexistepas.db"),
                "--study-name", "bidon", "--lambda-cost", "0.4"])
    assert r.returncode != 0, (
        f"base absente : code {r.returncode}, attendu non nul.\n"
        f"stdout={r.stdout}\nstderr={r.stderr}")


def test_d49_echec_apres_chargement_sort_en_erreur(tmp_path):
    """Une panne SURVENUE APRES le chargement doit aussi sortir non nul.

    C'est la moitie du defaut qu'un test « base absente » ne voit pas :
    l'etude se chargeait (125 essais annonces), puis l'ecriture echouait,
    et le message accusait quand meme le chargement — avec un code 0.
    """
    chemin = os.path.join(_DB_DIR, "classical_v2_phase1.db")
    if not os.path.exists(chemin):
        pytest.skip("base gelee absente")

    r = _lance(["--db-path", chemin, "--study-name", "classical_v2_phase1",
                "--lambda-cost", "0.4", "--output-dir", "/proc/impossible"])
    assert r.returncode != 0, (
        f"repertoire non ecrivable : code {r.returncode}, attendu non nul.\n"
        f"stdout={r.stdout}\nstderr={r.stderr}")
    assert "chargement" not in r.stderr.lower() or "FileNotFound" in r.stderr, (
        "la panne d'ecriture ne doit plus etre annoncee comme une panne de "
        f"chargement.\nstderr={r.stderr}")


def test_d49_le_chemin_nominal_rend_toujours_zero(tmp_path):
    """Garde-fou : rendre non nul en cas d'echec ne doit pas casser le
    succes. Sans ce test, « sortir en erreur » pourrait devenir « sortir
    en erreur tout le temps » sans que rien ne le signale."""
    chemin = os.path.join(_DB_DIR, "classical_v2_phase1.db")
    if not os.path.exists(chemin):
        pytest.skip("base gelee absente")

    r = _lance(["--db-path", chemin, "--study-name", "classical_v2_phase1",
                "--lambda-cost", "0.4", "--output-dir", str(tmp_path)])
    assert r.returncode == 0, (
        f"chemin nominal : code {r.returncode}\nstderr={r.stderr}")

    produit = os.path.join(tmp_path,
                           "rescore_classical_v2_phase1_lambda0.4000")
    attendus = ["trials_lambda0.4000.csv", "summary_lambda0.4000.txt"]
    for nom in attendus:
        assert os.path.exists(os.path.join(produit, nom)), (
            f"{nom} absent — le script a annonce un succes sans le produire")


# ══════════════════════════════════════════════════════════════════════
#  Q3 — deux detections de scenarios, un piege arme
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("nom,_", BASES_REELLES)
def test_les_deux_detections_de_scenarios_coincident(module_rescore, nom, _):
    """`recompute_score` detecte les scenarios PAR ESSAI ;
    `build_trial_table` les detecte sur l'ENSEMBLE, et seulement sur les
    dix premiers essais.

    Si un essai portait un jeu de cles different des autres, son
    `new_score` serait moyenne sur un denominateur different de celui de
    ses voisins — puis compare a eux dans le meme classement, sans le
    moindre signalement.

    Mesure sur les bases gelees : les deux chemins coincident, sur 100 %
    des essais. Le piege est **arme mais non declenche**. Ce test le fige :
    si une campagne future produit des essais heterogenes, il tombe ici et
    non dans un classement silencieusement fausse.
    """
    finis = _charge(nom)
    ensemble = module_rescore._detect_scenario_keys(finis)
    assert ensemble, "aucune cle detectee — balayage vide"

    divergents = [t.number for t in finis
                  if module_rescore._detect_scenario_keys([t]) != ensemble]
    assert not divergents, (
        f"{nom} : {len(divergents)} essais ont un jeu de scenarios different "
        f"de {ensemble} — leur score est moyenne sur un autre denominateur "
        f"que celui de leurs voisins. Essais : {divergents[:10]}")


def test_la_detection_ne_regarde_que_les_dix_premiers_essais(module_rescore):
    """Fige la limite connue de `_detect_scenario_keys`, pour qu'on ne la
    redecouvre pas par surprise.

    L'echantillon est `completed[:10]`. Une cle qui n'apparaitrait qu'a
    partir du onzieme essai serait invisible pour tout le tableau. Ce test
    ne demande pas de corriger : il exige que le comportement reste
    CONNU."""
    class FauxEssai:
        def __init__(self, attrs):
            self.user_attrs = attrs

    tardif = [FauxEssai({"phys_kh": 1.0, "patch_kh": 0.5}) for _ in range(10)]
    tardif.append(FauxEssai({"phys_ot": 1.0, "patch_ot": 0.5}))

    vu = module_rescore._detect_scenario_keys(tardif)
    assert vu == ["kh"], (
        f"detection sur {len(tardif)} essais : {vu}. Si 'ot' y figure "
        f"desormais, l'echantillon n'est plus limite aux dix premiers — "
        f"c'est une amelioration, remesurer et reecrire ce test.")


def test_le_balayage_de_lambda_produit_ses_trois_sorties(tmp_path):
    """`--lambda-sweep` est un axe de configuration distinct de
    `--lambda-cost`, et rien ne le traversait (question 5).

    Il emprunte `plot_lambda_sweep`, un repertoire `_sweep` separe, et le
    JSON de comparaison — aucun de ces trois chemins n'etait exerce.
    """
    import json

    chemin = os.path.join(_DB_DIR, "classical_v2_phase1.db")
    if not os.path.exists(chemin):
        pytest.skip("base gelee absente")

    r = _lance(["--db-path", chemin, "--study-name", "classical_v2_phase1",
                "--lambda-sweep", "0.0", "0.4", "1.0",
                "--output-dir", str(tmp_path)])
    assert r.returncode == 0, f"balayage : code {r.returncode}\n{r.stderr}"

    # Un repertoire par lambda, plus celui du balayage.
    for lam in ("0.0000", "0.4000", "1.0000"):
        d = tmp_path / f"rescore_classical_v2_phase1_lambda{lam}"
        assert d.is_dir(), f"repertoire manquant pour lambda={lam}"

    resume = tmp_path / "rescore_classical_v2_phase1_sweep" / "lambda_sweep_results.json"
    assert resume.exists(), "lambda_sweep_results.json absent"

    data = json.loads(resume.read_text())
    assert data["lambdas"] == [0.0, 0.4, 1.0]

    # Le score du meilleur essai doit CROITRE avec lambda : le terme de cout
    # s'ajoute a une physique deja minimisee. Un balayage qui rendrait trois
    # fois la meme valeur ne mesurerait rien.
    scores = data["best_scores"]
    assert scores == sorted(scores), (
        f"les meilleurs scores ne croissent pas avec lambda : {scores}")
    assert len(set(data["best_trials"])) > 1, (
        f"le meilleur essai est le meme a tout lambda ({data['best_trials']}) "
        f"— le rescore ne reclasserait rien, et ce test ne separerait rien")
