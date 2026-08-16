"""D-64 — l'import supprimait la destination avant d'avoir lu la source.

`src/import_Neon_data_to_local.py` est le SEUL code du depot qui supprime une
etude Optuna (`optuna.delete_study`, deux sites). Sa boucle faisait, dans cet
ordre : supprimer l'etude de destination, puis charger la source, puis copier.
Toute erreur — etude absente de la source, reseau, ecriture — etait rattrapee
par un `except Exception` qui imprimait ❌ et laissait le processus sortir a
**0**. La destination, elle, etait deja detruite.

Mesure avant correction : 5 essais dans la destination, source ne portant pas
l'etude -> `KeyError 'Record does not exist.'`, **code 0**, et l'etude locale
n'existe plus.

C'est l'empreinte que portent 8 des 10 bases de
`results/hyperparams/optuna_studies/` : schema Optuna complet, **zero ligne**,
et pour `classical_v2_phase2` / `classical_v2_phase3` **274 ko / 299 ko** la
ou un schema neuf pese **114 688 octets** — des pages liberees, donc des
lignes ecrites puis supprimees. Ce test ne prouve pas que ce script les a
videes ; il ferme le chemin qui le fait.

Sur quelle entree ces tests echouent-ils ? Sur la version d'avant D-64 :
`test_missing_source_leaves_destination_intact` y trouve la destination
detruite, et `test_real_failure_exits_nonzero` y lit un code 0.
"""

import os
import subprocess
import sys

import pytest

optuna = pytest.importorskip("optuna")


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_SCRIPT = os.path.join(_repo_root(), "src", "import_Neon_data_to_local.py")

#: L'une des dix etudes que le script parcourt en dur.
_STUDY = "q_has_v2_phase1"


def _make_study(path, n_trials, name=_STUDY):
    study = optuna.create_study(study_name=name, storage="sqlite:///" + path)
    dist = optuna.distributions.FloatDistribution(0, 10)
    for i in range(n_trials):
        study.add_trial(optuna.trial.create_trial(
            params={"x": float(i)}, distributions={"x": dist},
            value=float(i)))
    return study


def _n_trials(path, name=_STUDY):
    return len(optuna.load_study(study_name=name,
                                 storage="sqlite:///" + path).trials)


def _run(train_dir, source_url):
    """Les deux extremites sont des SQLite : l'`--in-url` remplace Neon."""
    return subprocess.run(
        [sys.executable, _SCRIPT, "--train-dir", str(train_dir),
         "--in-url", source_url],
        capture_output=True, text=True, timeout=900)


def test_missing_source_leaves_destination_intact(tmp_path):
    dest = str(tmp_path / (_STUDY + ".db"))
    source = str(tmp_path / "source.db")
    _make_study(dest, 5)
    optuna.create_study(study_name="une_autre_etude",
                        storage="sqlite:///" + source)

    out = _run(tmp_path, "sqlite:///" + source)

    assert _n_trials(dest) == 5, (
        "la destination a ete detruite alors que la source n'avait rien "
        "a donner")
    assert out.returncode == 0, "une etude absente de la source n'est pas un echec"
    assert "destination laissee intacte" in out.stdout


def test_real_import_copies_and_exits_zero(tmp_path):
    """Épingle le chemin qui marche : la copie doit toujours se faire."""
    dest = str(tmp_path / (_STUDY + ".db"))
    source = str(tmp_path / "source.db")
    _make_study(dest, 2)
    _make_study(source, 7)

    out = _run(tmp_path, "sqlite:///" + source)

    assert out.returncode == 0, out.stderr[-2000:]
    assert _n_trials(dest) == 7


def test_real_failure_exits_nonzero(tmp_path):
    """Un import qui n'a rien importe doit crier.

    L'echec est force apres la lecture de la source : la destination est un
    dossier, donc SQLite ne peut pas l'ouvrir.
    """
    source = str(tmp_path / "source.db")
    _make_study(source, 3)
    os.mkdir(str(tmp_path / (_STUDY + ".db")))

    out = _run(tmp_path, "sqlite:///" + source)

    assert out.returncode != 0
    assert "etudes en echec" in out.stderr


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


def test_deletion_happens_only_in_this_script():
    """Une seule porte, et elle est fermee : le verifier, pas le supposer."""
    root = _repo_root()
    hits = _delete_study_hits(root)
    hits = [h for h in hits
            if "/tests/" not in h and not h.startswith("tests/")
            and "__pycache__" not in h]
    assert hits, "le motif ne correspond a rien : balayage vide"
    assert all("import_Neon_data_to_local.py" in h for h in hits), hits


def test_the_scan_ignores_untracked_local_directories(tmp_path):
    """Epingle D-103 : un `.venv/` local ne doit plus faire echouer le test.

    Avant la correction, `grep -rn` sur `root` remontait tout fichier
    portant `delete_study` sur le disque, suivi ou non par git — y compris
    une dependance installee localement dans `root/.venv/`. Reproduit ici
    sur un mini-depot synthetique : un fichier SUIVI qui a le droit
    d'appeler `delete_study`, et un fichier NON SUIVI (dans un `.venv/`
    factice, comme le ferait une dependance installee sur place) qui ne
    devrait jamais compter.
    """
    root = tmp_path
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=root, check=True)

    tracked = root / "import_Neon_data_to_local.py"
    tracked.write_text("optuna.delete_study(study_name='x')\n")
    subprocess.run(["git", "add", "import_Neon_data_to_local.py"],
                   cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=root, check=True)

    venv_pkg = root / ".venv" / "site-packages" / "optuna_stub.py"
    venv_pkg.parent.mkdir(parents=True)
    venv_pkg.write_text("def delete_study(): ...\n")

    hits = _delete_study_hits(root)
    assert hits, "le fichier suivi doit toujours etre trouve"
    assert all(".venv" not in h for h in hits), (
        f"le .venv non suivi ne doit jamais apparaitre : {hits}")
    assert all("import_Neon_data_to_local.py" in h for h in hits), hits
