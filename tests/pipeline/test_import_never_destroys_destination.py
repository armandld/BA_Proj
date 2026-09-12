"""D-64 — plus aucun code du depot ne supprime une etude Optuna.

Historique. `src/import_Neon_data_to_local.py` etait le SEUL code du depot
qui appelait `optuna.delete_study` (deux sites). Sa boucle faisait, dans cet
ordre : supprimer l'etude de destination, puis charger la source, puis
copier. Toute erreur — etude absente de la source, reseau, ecriture — etait
rattrapee par un `except Exception` qui imprimait ❌ et laissait le processus
sortir a **0**. La destination, elle, etait deja detruite. Mesure d'alors :
5 essais dans la destination, source ne portant pas l'etude ->
`KeyError 'Record does not exist.'`, **code 0**, etude locale disparue.

C'est l'empreinte que portent 8 des 10 bases de
`results/hyperparams/optuna_studies/` : schema Optuna complet, **zero
ligne**, et pour `classical_v2_phase2` / `classical_v2_phase3` **274 ko /
299 ko** la ou un schema neuf pese **114 688 octets** — des pages liberees,
donc des lignes ecrites puis supprimees.

**Le fichier est supprime depuis** (`fdc7b03`, decision de USER,
`docs/RESULTS.md` § « Architecture Neon supprimée »). La porte que D-64
fermait n'existe plus : ce test ne peut donc plus exercer le script, et il
verifie desormais la propriete qui a remplace la correction — **aucun
fichier suivi n'appelle `delete_study`**. Un test dont l'assertion est une
absence peut passer pour de mauvaises raisons : `test_the_scan_can_fire`
prouve, sur un mini-depot synthetique, que le balayage trouve bien ce qu'il
cherche quand il y a quelque chose a trouver. Sans lui, l'absence ne
prouverait rien.

Sur quelle entree ces tests echouent-ils ? `test_no_tracked_code_deletes_a_study`
echoue des qu'un fichier suivi hors `tests/` reintroduit `delete_study` —
verifie en ecrivant la ligne dans un fichier suivi. `test_the_scan_can_fire`
echouait sur la version d'avant D-103, ou `_delete_study_hits` balayait le
disque au lieu des fichiers suivis.
"""

import os
import subprocess


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


def _delete_study_hits(root):
    """Cherche `delete_study` dans les fichiers SUIVIS par git de `root`.

    D-103 — PAS `grep -rn` sur le systeme de fichiers : ce depot est cense
    tourner dans un environnement Python installe a cote (conda,
    `environment.yaml`), mais rien ne l'empeche de vivre DANS `root` — le
    propre `.gitignore` du depot anticipe `.venv/`, `.venv_vigil/`, `env/`.
    `grep -rn` les traverse quand meme : `optuna` (dependance installee)
    definit et appelle `delete_study` a une douzaine d'endroits dans son
    propre code, tous remontes comme si le depot les avait ecrits. `git
    grep` ne regarde que les fichiers SUIVIS — c'est exactement « le depot »
    que ce test veut dire, et il ignore de lui-meme tout ce que `.gitignore`
    exclut, sans avoir a enumerer les noms de dossiers un par un.
    """
    out = subprocess.run(
        ["git", "grep", "-n", "delete_study", "--", "*.py"],
        cwd=root, capture_output=True, text=True)
    return out.stdout.splitlines()


def _repo_hits():
    hits = _delete_study_hits(_repo_root())
    return [h for h in hits
            if "/tests/" not in h and not h.startswith("tests/")
            and "__pycache__" not in h]


def test_no_tracked_code_deletes_a_study():
    """La porte de D-64 est retiree, pas seulement fermee.

    L'assertion est une absence ; `test_the_scan_can_fire` est ce qui la rend
    signifiante.
    """
    assert _repo_hits() == [], (
        "un fichier suivi rappelle `delete_study` : la destruction d'etude "
        "etait le chemin de D-64, elle ne revient pas sans decision "
        "explicite — voir docs/RESULTS.md")


def test_the_scan_can_fire(tmp_path):
    """Le balayage trouve un fichier SUIVI et ignore un `.venv/` local.

    Epingle a la fois D-103 (le balayage ne doit pas remonter les
    dependances installees sur place) et la validite du test ci-dessus : si
    `_delete_study_hits` ne rendait jamais rien, l'absence mesuree la-bas ne
    voudrait rien dire.
    """
    root = tmp_path
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)

    tracked = root / "un_importateur.py"
    tracked.write_text("optuna.delete_study(study_name='x')\n")
    subprocess.run(["git", "add", "un_importateur.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True)

    venv_pkg = root / ".venv" / "site-packages" / "optuna_stub.py"
    venv_pkg.parent.mkdir(parents=True)
    venv_pkg.write_text("def delete_study(): ...\n")

    hits = _delete_study_hits(root)
    assert hits, "le fichier suivi doit etre trouve : sinon le balayage est mort"
    assert all(".venv" not in h for h in hits), (
        f"le .venv non suivi ne doit jamais apparaitre : {hits}")
    assert all("un_importateur.py" in h for h in hits), hits
