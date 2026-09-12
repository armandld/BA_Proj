"""Audit de contrat de `src/analyze_hyperparams.py`.

Deuxieme des cinq fichiers « jamais audites » de V1. Il ne produit aucun
nombre publie : il produit le resume et les seize figures a partir
desquels on DECIDE. Un diagnostic faux y coute autant qu'un nombre faux
ailleurs.

Deux trouvailles, de nature differente :

  D-50  -- le chemin d'echec rendait 0, et les deux gestionnaires
           designaient la mauvaise cause (« Neon » pour un fichier local
           absent ; « etude introuvable » pour un KeyError de figure).

  `_detect_scenario_keys` -- annoncait SEPT scenarios la ou les donnees en
           portent QUATRE. Aucune sortie fausse : les quatre appelants
           filtrent. Piege arme, corrige, sortie prouvee identique.
"""

import os
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO, "src")
_DB_DIR = os.path.join(_REPO, "results", "hyperparams", "optuna_studies")
_SCRIPT = os.path.join(_SRC, "analyze_hyperparams.py")

BASES_REELLES = ["classical_v2_phase1", "q_has_v2_phase1"]

#: Les scenarios que les deux bases gelees portent reellement.
SCENARIOS_REELS = ["kh", "tearing", "ot", "rotor"]


@pytest.fixture(scope="module")
def analyze():
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    return pytest.importorskip("analyze_hyperparams")


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
    return etude, finis


def _lance(args):
    return subprocess.run([sys.executable, _SCRIPT] + args,
                          capture_output=True, text=True, timeout=1800)


def test_loads_the_rented_campaign_journal(analyze, tmp_path):
    optuna = pytest.importorskip("optuna")
    from optuna.storages.journal import (JournalFileBackend,
                                         JournalFileOpenLock)
    path = tmp_path / "campaign.log"
    storage = optuna.storages.JournalStorage(JournalFileBackend(
        str(path), lock_obj=JournalFileOpenLock(str(path))))
    study = optuna.create_study(study_name="journal", storage=storage)
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0),
                   n_trials=2)

    loaded, completed = analyze.load_journal(str(path), "journal")
    assert loaded.study_name == "journal"
    assert len(completed) == 2


# ══════════════════════════════════════════════════════════════════════
#  `_detect_scenario_keys` — n'annoncer que ce qui existe
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("nom", BASES_REELLES)
def test_ne_declare_que_les_scenarios_presents(analyze, nom):
    """La fonction rendait toute la famille des sept des qu'UNE cle etait
    trouvee. Sur des donnees a quatre scenarios, elle en annoncait sept —
    inventant `vortex`, `coalescence` et `gt`."""
    _, finis = _charge(nom)
    cles = analyze._detect_scenario_keys(finis)

    assert cles == SCENARIOS_REELS, (
        f"{nom} : cles annoncees {cles}, attendues {SCENARIOS_REELS}")

    for fantome in ("vortex", "coalescence", "gt"):
        assert fantome not in cles, (
            f"{nom} : '{fantome}' annonce alors qu'aucun essai ne le porte")


@pytest.mark.parametrize("nom", BASES_REELLES)
def test_chaque_cle_annoncee_existe_vraiment(analyze, nom):
    """Formulation independante de la liste attendue : quelle que soit la
    base, toute cle rendue doit avoir son attribut dans les donnees."""
    _, finis = _charge(nom)
    cles = analyze._detect_scenario_keys(finis)
    assert cles, "aucune cle detectee — le balayage serait vide"

    for k in cles:
        porteurs = sum(1 for t in finis if f"loss_{k}" in t.user_attrs)
        assert porteurs > 0, (
            f"{nom} : '{k}' est annonce mais aucun des {len(finis)} essais "
            f"ne porte 'loss_{k}'")


def test_les_deux_copies_de_la_detection_saccordent(analyze):
    """`analyze_hyperparams` et `recompute_lambda_scores` portent chacun
    leur `_detect_scenario_keys`. Les deux avaient DIVERGE — prefixes
    differents (`loss_` contre `phys_`) et semantiques differentes.

    Elles lisent la meme base : elles doivent en tirer les memes
    scenarios, sans quoi les deux analyses d'une meme campagne ne parlent
    pas du meme sous-ensemble.
    """
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    recompute = pytest.importorskip("recompute_lambda_scores")

    for nom in BASES_REELLES:
        _, finis = _charge(nom)
        a = analyze._detect_scenario_keys(finis)
        r = recompute._detect_scenario_keys(finis)
        assert a == r, (
            f"{nom} : analyze rend {a}, recompute rend {r} — les deux "
            f"scripts n'analysent pas les memes scenarios")


def test_la_detection_reste_bornee_aux_dix_premiers_essais(analyze):
    """Fige la limite connue de l'echantillon, comme pour la copie de
    `recompute_lambda_scores`. Ce test n'exige pas de la corriger : il
    exige qu'elle reste CONNUE."""
    class FauxEssai:
        def __init__(self, attrs):
            self.user_attrs = attrs

    essais = [FauxEssai({"loss_kh": 1.0}) for _ in range(10)]
    essais.append(FauxEssai({"loss_ot": 1.0}))

    vu = analyze._detect_scenario_keys(essais)
    assert vu == ["kh"], (
        f"detection sur {len(essais)} essais : {vu}. Si 'ot' y figure "
        f"desormais, l'echantillon n'est plus borne aux dix premiers — "
        f"remesurer et reecrire ce test.")


# ══════════════════════════════════════════════════════════════════════
#  D-50 — le chemin d'echec rendait 0, en accusant la mauvaise cause
# ══════════════════════════════════════════════════════════════════════

def test_d50_base_introuvable_sort_en_erreur(tmp_path):
    """Avant D-50 : code 0, et le message accusait **Neon** — une base
    distante qui n'intervient pas — pour un fichier local absent."""
    r = _lance(["--db-path", str(tmp_path / "nexistepas.db"),
                "--study-name", "bidon",
                "--output-dir", str(tmp_path / "sortie")])
    assert r.returncode != 0, (
        f"base absente : code {r.returncode}, attendu non nul.\n"
        f"stdout={r.stdout}\nstderr={r.stderr}")


def test_d50_le_message_ne_parle_plus_de_neon(tmp_path):
    """Le diagnostic doit designer la cause reelle.

    « Study does not exist on Neon yet » pour un `.db` local manquant
    envoie chercher la panne au mauvais endroit — et ce message
    s'imprimait aussi pour un `KeyError` leve par n'importe laquelle des
    treize figures.
    """
    r = _lance(["--db-path", str(tmp_path / "nexistepas.db"),
                "--study-name", "bidon",
                "--output-dir", str(tmp_path / "sortie")])
    sortie = (r.stdout + r.stderr).lower()
    assert "neon" not in sortie, (
        f"le message accuse encore Neon :\n{r.stdout}\n{r.stderr}")
    assert "chargement" in sortie or "erreur" in sortie, (
        f"le message ne nomme pas la panne :\n{r.stdout}\n{r.stderr}")


def test_d50_le_chemin_nominal_rend_toujours_zero(tmp_path):
    """Garde-fou : sortir non nul en cas d'echec ne doit pas transformer
    le succes en echec. Sans ce test, la correction pourrait devenir
    « echouer toujours » sans que rien ne le signale."""
    chemin = os.path.join(_DB_DIR, "classical_v2_phase1.db")
    if not os.path.exists(chemin):
        pytest.skip("base gelee absente")

    r = _lance(["--db-path", chemin, "--study-name", "classical_v2_phase1",
                "--output-dir", str(tmp_path)])
    assert r.returncode == 0, (
        f"chemin nominal : code {r.returncode}\nstderr={r.stderr}")
    assert (tmp_path / "summary.txt").exists(), (
        "summary.txt absent — le script a annonce un succes sans le produire")


@pytest.mark.parametrize("nom", BASES_REELLES)
def test_le_resume_ne_liste_que_les_scenarios_reels(analyze, nom, tmp_path):
    """Controle de bout en bout, sur la sortie que l'humain lit.

    C'est ce test qui atteste que la correction de `_detect_scenario_keys`
    n'a rien change : le resume listait deja quatre scenarios, parce que
    chaque appelant filtrait. Il doit continuer d'en lister quatre.
    """
    import contextlib
    import io

    etude, finis = _charge(nom)
    tampon = io.StringIO()
    with contextlib.redirect_stdout(tampon):
        analyze.generate_summary(etude, finis,
                                 analyze.get_param_names(finis), str(tmp_path))
    texte = tampon.getvalue()

    assert "Per-scenario breakdown" in texte, (
        "le resume ne contient pas la ventilation par scenario — le test "
        "ne verifierait rien")
    for etiquette in ("Kelvin-Helmholtz", "Harris Tearing",
                      "Orszag-Tang", "MHD Rotor"):
        assert etiquette in texte, f"{nom} : '{etiquette}' absent du resume"
    for fantome in ("Lamb-Oseen", "Island Coalescence", "Ghost Twisting"):
        assert fantome not in texte, (
            f"{nom} : '{fantome}' apparait dans le resume alors qu'aucun "
            f"essai ne le porte")
