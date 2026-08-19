"""Le mode d'emploi promet-il des choses qui existent ?

Ecrit apres avoir lu `MODE_EMPLOI_CAMPAGNE.md` §4, qui demandait
`pip install -r requirements.txt` alors que **ce fichier n'existait pas**.
La mise en place echouait donc sur une machine deja payee, apres
l'allocation — meme famille que D-136, un mode annonce qui s'effondre au
lancement sur des coeurs factures.

Un document qui promet un fichier absent est un faux vert d'un genre
particulier : rien ne le teste, et on ne le decouvre qu'au moment ou ca
coute.
"""
import os
import re
import subprocess
import sys

import pytest

_RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MANUEL = os.path.join(_RACINE, "docs", "MODE_EMPLOI_CAMPAGNE.md")
_SOUMETTRE = os.path.join(_RACINE, "scripts", "soumettre_campagne.sh")


def _texte_manuel():
    return open(_MANUEL, encoding="utf-8").read()


# ------------------------------------------------------------------
#  1. les fichiers que le manuel promet
# ------------------------------------------------------------------
def test_requirements_existe_et_couvre_la_pile():
    """Le fichier que §4 demande d'installer, et son contenu minimal."""
    chemin = os.path.join(_RACINE, "requirements.txt")
    assert os.path.exists(chemin), (
        "MODE_EMPLOI §4 demande `pip install -r requirements.txt` : "
        "le fichier doit exister, sinon la mise en place echoue sur des "
        "coeurs deja payes")
    contenu = open(chemin, encoding="utf-8").read().lower()
    # Les paquets sans lesquels la campagne ne demarre pas, ou sans
    # lesquels la suite se tait au lieu d'echouer.
    for paquet in ("optuna", "qiskit", "qiskit-aer", "numpy", "scipy",
                   "scikit-learn", "cma", "qiskit-machine-learning"):
        assert paquet in contenu, f"{paquet} absent de requirements.txt"


#: Ce que le manuel cite mais que le depot ne versionne pas : des SORTIES,
#: regenerees par une commande. Meme distinction que le `_SORTIES` de
#: `test_scripts_point_somewhere.py` — une sortie absente est normale, une
#: ENTREE absente est une promesse non tenue.
_SORTIES_MANUEL = ("scripts/job_campagne_",)


def test_tout_script_cite_par_le_manuel_existe():
    """Un lanceur cite mais absent est une promesse non tenue."""
    cites = set(re.findall(r"scripts/[A-Za-z0-9_]+\.(?:sh|py)", _texte_manuel()))
    assert cites, "aucun script cite : le balayage ne prouve rien"
    manquants = [c for c in sorted(cites)
                 if not c.startswith(_SORTIES_MANUEL)
                 and not os.path.exists(os.path.join(_RACINE, c))]
    assert not manquants, f"cites par MODE_EMPLOI mais absents : {manquants}"


def test_les_sorties_exemptees_sont_bien_produites_par_un_script():
    """Une exemption qui ne correspond a rien devient un trou permanent.

    Sur quelle entree ce test echoue : si `job_campagne_` cesse d'etre
    ecrit par `soumettre_campagne.sh`, l'exemption couvrirait alors un
    fichier que plus personne ne produit — et le garde ci-dessus se
    tairait sur une vraie promesse non tenue.
    """
    source = open(_SOUMETTRE, encoding="utf-8").read()
    for sortie in _SORTIES_MANUEL:
        nom = sortie.split("/")[-1]
        assert nom in source, (
            f"'{sortie}' est exempte mais {os.path.basename(_SOUMETTRE)} "
            "ne l'ecrit plus : retirer l'exemption")


def test_le_manuel_ne_cite_plus_l_ancien_nom_de_base():
    """`q_has_v3.db` contenait l'etude `q_has_v2_phase1` : nom retire."""
    assert "q_has_v3.db" not in _texte_manuel(), (
        "le manuel cite encore `q_has_v3.db` ; la base porte desormais le "
        "nom de son etude")


# ------------------------------------------------------------------
#  2. le generateur de job
# ------------------------------------------------------------------
@pytest.mark.parametrize("ordonnanceur,directive", [("pbs", "#PBS"), ("slurm", "#SBATCH")])
def test_le_job_genere_est_du_bash_valide(tmp_path, ordonnanceur, directive):
    """Genere, puis passe a `bash -n` : un job qui ne parse pas se decouvre
    dans la file d'attente, des heures plus tard."""
    out = subprocess.run(["bash", _SOUMETTRE, ordonnanceur, "4", "6", "50"],
                         capture_output=True, text=True, timeout=120, cwd=_RACINE)
    assert out.returncode == 0, out.stderr
    job = os.path.join(_RACINE, "scripts", f"job_campagne_{ordonnanceur}.sh")
    try:
        assert os.path.exists(job)
        lignes = open(job, encoding="utf-8").read().splitlines()
        verif = subprocess.run(["bash", "-n", job], capture_output=True, text=True)
        assert verif.returncode == 0, verif.stderr

        # Les directives doivent suivre le shebang sans ligne vide : les
        # deux ordonnanceurs cessent de lire l'entete a la premiere ligne
        # qui n'est ni commentaire ni vide, et une entete decollee est un
        # job qui part avec les mauvaises ressources.
        assert lignes[0].startswith("#!")
        assert lignes[1].startswith(directive), (
            f"ligne 2 = {lignes[1]!r} ; l'entete {directive} doit coller "
            "au shebang")

        # Pas de variable runtime laissee litterale par un echappement de trop.
        texte = "\n".join(lignes)
        assert "\\$" not in texte, (
            "une variable est echappee en trop : elle serait litterale a "
            "l'execution")
    finally:
        if os.path.exists(job):
            os.remove(job)


def test_le_generateur_refuse_un_ordonnanceur_inconnu():
    out = subprocess.run(["bash", _SOUMETTRE, "lsf"],
                         capture_output=True, text=True, timeout=60, cwd=_RACINE)
    assert out.returncode == 2, "un ordonnanceur inconnu doit etre refuse"


def test_le_generateur_previent_quand_la_capacite_manque():
    """8 workers x 6 h = 40 essais < 200 : il doit le dire, pas se taire."""
    out = subprocess.run(["bash", _SOUMETTRE, "pbs", "8", "6", "200"],
                         capture_output=True, text=True, timeout=120, cwd=_RACINE)
    job = os.path.join(_RACINE, "scripts", "job_campagne_pbs.sh")
    try:
        assert out.returncode == 0, out.stderr
        assert "ATTENTION" in out.stdout, (
            "capacite insuffisante annoncee sans avertissement : "
            "on croirait la campagne dimensionnee")
    finally:
        if os.path.exists(job):
            os.remove(job)
