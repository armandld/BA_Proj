"""Aucun identifiant de base ne doit revenir dans `src/`.

`import_Neon_data_to_local.py` portait l'URL Neon complete — mot de passe
compris — en valeur par defaut. Le depot est **public**. Le retrait de cette
valeur ne desamorce pas ce qui est deja dans l'historique git : seule une
rotation du mot de passe le fait (`docs/DEFAUTS.md`). Ce test empeche le
suivant d'etre publie a son tour.

L'assertion porte sur ce que le module FAIT — il refuse de demarrer sans URL
— et sur la forme d'une URL portant un mot de passe, pas sur une chaine
precise : un identifiant different, mais code en dur, doit tomber lui aussi.
"""

import os
import re
import subprocess
import sys

import pytest


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_ROOT = _repo_root()
_SRC = os.path.join(_ROOT, "src")

#: `schema://utilisateur:motdepasse@hote` — la forme qui porte un secret.
_URL_WITH_PASSWORD = re.compile(
    r"[a-z][a-z0-9+.\-]*://[^\s/:@\"']+:[^\s/:@\"']+@[^\s\"']+")


def _python_files():
    out = []
    for dirpath, _dirs, names in os.walk(_SRC):
        if "__pycache__" in dirpath:
            continue
        for n in names:
            if n.endswith(".py"):
                out.append(os.path.join(dirpath, n))
    return out


def test_no_hardcoded_credential_url_in_src():
    files = _python_files()
    assert files, "balayage vide : aucun fichier .py trouve dans src/"

    offenders = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if line.lstrip().startswith("#"):
                    continue          # un commentaire qui documente la fuite
                if _URL_WITH_PASSWORD.search(line):
                    offenders.append("%s:%d" % (os.path.relpath(path, _ROOT),
                                                lineno))
    assert offenders == [], (
        "URL portant un mot de passe dans du code source public : %s"
        % offenders)


def test_the_pattern_can_fire(tmp_path):
    """Un test qui ne peut pas echouer est un defaut : on le prouve ici."""
    assert _URL_WITH_PASSWORD.search(
        'url = "postgresql://user:secret@host.example/db?sslmode=require"')
    assert not _URL_WITH_PASSWORD.search('url = "sqlite:///local.db"')


def test_script_refuses_to_run_without_a_url(tmp_path):
    """Le comportement, pas le texte : sans URL, le script s'arrete."""
    env = dict(os.environ)
    env.pop("NEON_DB_URL", None)
    out = subprocess.run(
        [sys.executable, os.path.join(_SRC, "import_Neon_data_to_local.py"),
         "--train-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode != 0
    assert "NEON_DB_URL" in out.stderr
