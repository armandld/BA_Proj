"""Provenance d'une execution longue — le hash pris AU BON MOMENT.

DEFAUT D15
----------
`git_commit_hash()` etait appele au moment de SAUVEGARDER l'artefact. Une
execution d'une heure se retrouvait donc estampillee avec ce qui avait ete
commite pendant qu'elle tournait — c'est-a-dire, potentiellement, avec du
code qu'elle n'a jamais execute.

Ce n'est pas theorique. Les artefacts T20 de `ot` et `kh` portent un hash
POSTERIEUR au commit qui a introduit `always_matched=True` dans leur propre
controle classique, alors qu'ils ont execute la version d'avant. Python
charge le module a l'import : editer le fichier pendant l'execution ne
change rien au code qui tourne, mais change le hash qui sera ecrit. Le
tampon de provenance pointait activement a cote de la verite.

CLAUDE.md exige le hash du commit dans chaque sortie. Ce module rend cette
exigence utile pour les taches longues :

  - `hash_at_start` : l'etat du depot quand le calcul a COMMENCE ;
  - `hash_at_save`  : l'etat au moment d'ecrire ;
  - `head_moved_during_run` : vrai s'ils different, c'est-a-dire si un
    commit est tombe pendant l'execution. Dans ce cas `hash_at_start` est
    le seul des deux qui decrive le code execute ;
  - `dirty_at_start` : vrai si l'arbre de travail portait des modifications
    non commitees au demarrage — alors AUCUN hash ne decrit exactement ce
    qui a tourne, et il faut le dire plutot que de le laisser deviner.
"""
import os
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))


def _git(*argv):
    try:
        return subprocess.check_output(
            ["git", *argv], cwd=_HERE,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _head():
    return _git("rev-parse", "HEAD") or "unknown"


def _dirty():
    """Arbre de travail modifie ? None si git est indisponible."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=_HERE,
                             capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def start():
    """A appeler AVANT le calcul. Retourne l'etat de depart."""
    return {"git_hash_at_start": _head(), "dirty_at_start": _dirty()}


def finish(started):
    """A appeler au moment de sauvegarder. Retourne le bloc complet.

    `git_hash` reste present et vaut le hash de DEPART : c'est celui qui
    decrit le code execute, et c'est la cle que lisent les agregateurs.
    """
    end = _head()
    out = dict(started)
    out["git_hash_at_save"] = end
    out["head_moved_during_run"] = (
        started["git_hash_at_start"] != end
        and "unknown" not in (started["git_hash_at_start"], end))
    out["git_hash"] = started["git_hash_at_start"]
    return out
