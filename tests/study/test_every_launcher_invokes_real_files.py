"""D-76 : deux lanceurs de campagne sur cinq invoquaient des chemins que le
depot n'a plus, et mouraient sur leur PREMIER appel Python.

D-71 avait corrige exactement ce defaut — la reorganisation `17d983d` a
deplace ET renomme chaque script — mais son test nomme les lanceurs un par
un (`scripts/run_fold.sh`, `scripts/run_leak_free_campaign.sh`). Les trois
autres n'ont jamais ete regardes. Meme forme que D-75 : le detecteur d'une
correction ne voyait que les sites que cette correction avait touches.

Mesure avant / apres :

    bash scripts/run_study_v2_phases.sh 2
      avant : code 2, "python: can't open file '.../study/hard_patch_labels.py'"
      apres : code 0, phase 2 executee de bout en bout

    ls -lh study/results/*.npz   (derniere ligne des deux lanceurs)
      avant : "(no results yet)"  alors que `results/` porte 224 .npz
      apres : la liste reelle

Ce test-ci ne nomme aucun lanceur : il balaie `scripts/*.sh` et les `.sh` de
la racine. Un lanceur volontairement mort — `run_study_v3.sh`, gele par
D-49 — est exempte, mais l'exemption est verifiee : le fichier doit porter
son avertissement, sinon elle tombe.
"""
import glob
import os
import re

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

#: Lanceurs volontairement morts, avec la raison ecrite DANS le fichier.
#: L'exemption n'est pas un nom sur une liste : le test relit la raison.
_FROZEN = {
    "scripts/run_study_v3.sh": ("NE FONCTIONNE PLUS", "D-49"),
}

#: Les lanceurs passent leur cible a une fonction (`run_phase 2 study/...py`),
#: pas directement a `python` : chercher le mot-cle `python` sur la ligne ne
#: capturerait presque rien — et un balayage vide passerait au vert. Le motif
#: capture donc tout chemin de script du depot cite hors commentaire.
_INVOKE_RE = re.compile(r"((?:study|scripts|figures|src)/[A-Za-z0-9_./-]+\.(?:py|sh))")


def _launchers():
    out = sorted(glob.glob(os.path.join(_REPO_ROOT, "scripts", "*.sh")))
    out += sorted(glob.glob(os.path.join(_REPO_ROOT, "*.sh")))
    return [os.path.relpath(p, _REPO_ROOT) for p in out]


LAUNCHERS = _launchers()


def _invoked_paths(relpath):
    """Chemins reellement invoques, commentaires exclus."""
    invoked = set()
    with open(os.path.join(_REPO_ROOT, relpath), encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("#"):
                continue
            invoked.update(_INVOKE_RE.findall(line))
    return invoked


@pytest.mark.parametrize("relpath", LAUNCHERS, ids=LAUNCHERS)
def test_launcher_invokes_only_files_that_exist(relpath):
    if relpath in _FROZEN:
        pytest.skip(f"{relpath} : gele et documente comme tel, voir "
                    f"test_a_frozen_launcher_says_so_in_its_own_header")
    missing = sorted(p for p in _invoked_paths(relpath)
                     if not os.path.exists(os.path.join(_REPO_ROOT, p)))
    assert not missing, (
        f"{relpath} invoque des fichiers absents : {missing}. Le lanceur "
        "meurt sur le premier appel — la campagne qu'il documente n'est plus "
        "rejouable. Voir D-71, D-76.")


@pytest.mark.parametrize("relpath", sorted(_FROZEN), ids=sorted(_FROZEN))
def test_a_frozen_launcher_says_so_in_its_own_header(relpath):
    """Une deviation connue non consignee LA OU ELLE VIT se fait recorriger
    par erreur. Si l'avertissement quitte le fichier, l'exemption tombe et le
    test precedent redevient exigeant."""
    with open(os.path.join(_REPO_ROOT, relpath), encoding="utf-8") as fh:
        head = "".join(fh.readlines()[:40])
    for fragment in _FROZEN[relpath]:
        assert fragment in head, (
            f"{relpath} n'explique plus pourquoi il est mort ({fragment!r} "
            "absent de son en-tete) : soit le remettre en etat, soit y "
            "remettre la raison — l'exemption de ce test en depend")


@pytest.mark.parametrize("relpath", LAUNCHERS, ids=LAUNCHERS)
def test_no_launcher_lists_the_flattened_results_directory(relpath):
    """`study/results/` a ete aplati vers `results/` a la meme reorganisation.
    Un lanceur qui le liste encore annonce « (no results yet) » sur un depot
    qui en porte 224."""
    if relpath in _FROZEN:
        pytest.skip(f"{relpath} : gele et documente comme tel")
    with open(os.path.join(_REPO_ROOT, relpath), encoding="utf-8") as fh:
        hits = [ln.strip() for ln in fh
                if "study/results/" in ln and not ln.strip().startswith("#")]
    assert not hits, f"{relpath} pointe encore sur study/results/ : {hits}"


def test_the_sweep_itself_is_not_empty():
    """Un balayage vide doit crier — y compris celui-ci. Sans cela, un motif
    d'invocation qui cesse de correspondre rendrait tout vert."""
    assert len(LAUNCHERS) >= 5, f"seulement {LAUNCHERS} collectes"
    total = sum(len(_invoked_paths(p)) for p in LAUNCHERS)
    assert total >= 25, (
        f"le motif d'invocation ne trouve que {total} cibles dans "
        f"{len(LAUNCHERS)} lanceurs : c'est le motif qui a cesse de "
        "correspondre, pas les lanceurs qui ont cesse d'invoquer")
