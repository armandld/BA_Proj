"""D-71 — les commandes de reproduction de `RESULTS.md` pointaient sur des
fichiers que le depot n'a plus.

`17d983d` (« reorganise the repository around the hypotheses ») a deplace ET
renomme tous les scripts de `study/v4/tNN_xxx.py` vers
`study/<module>/<module>_xxx.py`, sans toucher `docs/RESULTS.md` ni les
docstrings d'usage des scripts eux-memes, ni les deux lanceurs de
`scripts/`. Aucun nombre n'etait faux ; la commande donnee pour le
reobtenir ne s'executait plus (`FileNotFoundError` immediat), ce qui est
exactement le defaut que `RESULTS.md` lui-meme s'interdit : *« un resultat
qu'on ne sait pas refaire n'est pas un resultat »*.

Corrige (D-71) : chaque chemin `study/v4/...` cite comme commande executable
a ete remplace par son chemin reel actuel, verifie fichier par fichier.
Les citations PUREMENT HISTORIQUES (narrations de campagnes passees, ex.
`tests/study/test_silent_failure_sweep.py`, le paragraphe « Trap sweep » de
`RESULTS.md`) et les mentions qui NOMMENT le defaut pour l'expliquer (les
lignes de registre D-71 elles-memes, les commentaires laisses dans les deux
scripts corriges) sont laissees telles quelles : elles decrivent un fait,
pas une commande a rejouer aujourd'hui.

Ce test balaie `docs/RESULTS.md` et les deux lanceurs de `scripts/` pour
tout chemin `study/...`, `scripts/...` ou `figures/...` qui ressemble a un
fichier executable, et verifie qu'il existe. Il epingle aussi l'absence du
prefixe mort `study/v4/` dans les fichiers vivants (hors narration
historique).
"""

import os
import re
import subprocess
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

_PATH = r"((?:study|scripts|figures)/[A-Za-z0-9_./-]+\.(?:py|sh))"
# une commande de reproduction : soit une ligne markdown ENTIEREMENT en
# code inline (`` `path --flags` ``), soit une ligne "python "/"bash " a
# l'interieur d'un bloc de code cloture. Les deux formes couvrent tout ce
# que RESULTS.md utilise reellement pour ses commandes rejouables — une
# citation narrative (« `study/phase0_sanity_check.py:95` », prose au fil
# du texte) ne correspond a aucune des deux.
_INLINE_CMD_RE = re.compile(r"^`" + _PATH + r"(?:\s[^`]*)?`$")
_FENCED_CMD_RE = re.compile(r"^(?:nohup\s+)?(?:python|bash)\s+" + _PATH + r"\b")

# lignes deliberement historiques : narration d'une campagne passee, pas une
# commande a rejouer. Identifiees par leur fichier + un fragment stable.
_HISTORICAL_EXCEPTIONS = {
    ("docs/RESULTS.md", "trap-sweep"),  # paragraphe "Trap sweep", cite l'etat historique
    ("docs/RESULTS.md", "d71-entry"),  # la ligne de registre D-71 elle-meme, decrit le defaut
    ("docs/COUVERTURE.md", "d71-entry"),  # idem, section h1_solver_convergence.py
    ("scripts/run_fold.sh", "d71-comment"),  # commentaire D-71 : explique la cause, ne l'invoque pas
    ("scripts/run_leak_free_campaign.sh", "d71-comment"),  # idem
}


def _paths_referenced(text):
    """Chemins cites comme COMMANDE rejouable, pas comme prose ou table."""
    found = set()
    in_fence = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        m = _INLINE_CMD_RE.match(line)
        if m:
            found.add(m.group(1))
            continue
        if in_fence:
            m = _FENCED_CMD_RE.match(line)
            if m:
                found.add(m.group(1))
    return found


@pytest.fixture(scope="module")
def results_md():
    with open(os.path.join(_REPO_ROOT, "docs", "RESULTS.md"), encoding="utf-8") as f:
        return f.read()


def test_every_repro_command_in_results_md_points_to_a_real_file(results_md):
    referenced = _paths_referenced(results_md)
    assert len(referenced) > 10, (
        "le balayage n'a presque rien trouve dans RESULTS.md : le motif a "
        "probablement cesse de correspondre, pas le depot qui n'a plus de "
        "commandes")
    missing = sorted(
        p for p in referenced
        if not os.path.exists(os.path.join(_REPO_ROOT, p)))
    assert not missing, (
        "commandes de reproduction dans RESULTS.md pointant sur des "
        f"fichiers absents : {missing}")


# Les deux lanceurs shell resolvent leur cible via une variable ($ROOT/...,
# $HERE/...), pas un chemin litteral sur la ligne "python " : ce motif
# capture le SUFFIXE study/... qu'importe le prefixe qui le precede.
_SHELL_INVOKE_CAPTURE_RE = re.compile(r"python\s+\S*?" + _PATH)


@pytest.mark.parametrize("relpath,expected_targets", [
    ("scripts/run_fold.sh", [
        "study/closed_loop/closed_loop_campaign.py",
        "study/closed_loop/closed_loop_budget_matched.py",
    ]),
    ("scripts/run_leak_free_campaign.sh", [
        "study/h4_transfer/h4_unseen_conditions.py",
    ]),
])
def test_launcher_scripts_invoke_real_files(relpath, expected_targets):
    with open(os.path.join(_REPO_ROOT, relpath), encoding="utf-8") as f:
        lines = f.readlines()
    invoked = set()
    for raw in lines:
        line = raw.strip()
        if line.startswith("#"):
            continue
        m = _SHELL_INVOKE_CAPTURE_RE.search(line)
        if m:
            invoked.add(m.group(1))
    # le script doit reellement invoquer ce qu'on attend de lui — pas juste
    # "un chemin qui existe quelque part" : un test qui ne verifierait que
    # l'existence laisserait passer une cible plausible mais fausse.
    assert invoked == set(expected_targets), (
        f"{relpath} invoque {sorted(invoked)}, attendu {sorted(expected_targets)}")
    missing = sorted(
        p for p in invoked if not os.path.exists(os.path.join(_REPO_ROOT, p)))
    assert not missing, f"{relpath} invoque des fichiers absents : {missing}"


# ── D-140 : le chemin existe, l'option non ────────────────────────────
#
# Les tests ci-dessus verifient que le FICHIER cite existe. Rien ne
# verifiait que les OPTIONS citees existent : la commande publiee pour
# verifier D-53 — le resultat le plus fort du depot — portait `--check`,
# que son script ne declare pas. Elle rendait `error: unrecognized
# arguments` et sortait en **2**, sous un test vert.
#
# L'assertion porte sur le COMPORTEMENT : on interroge le parseur du
# script par son propre `--help`, on ne cherche pas la chaine dans le
# source. ~20 s pour l'ensemble, les imports lourds etant mis en cache
# par script.

_PY_CMD_RE = re.compile(
    r"python\s+((?:study|src|scripts|figures)/[A-Za-z0-9_./-]+\.py)([^\n`|]*)")
_LONG_OPT_RE = re.compile(r"(?<![\w-])(--[a-zA-Z][a-zA-Z0-9-]*)")


def _commands_with_options(text):
    """(script, options) pour chaque commande `python <script> --opt …`.

    Le texte est aplati d'abord : une commande de `RESULTS.md` peut etre
    coupee en deux lignes a l'interieur d'un meme span de code inline.
    """
    flat = re.sub(r"`([^`]*)`", lambda m: "`" + m.group(1).replace("\n", " ") + "`",
                  text)
    out = set()
    for m in _PY_CMD_RE.finditer(flat):
        opts = frozenset(_LONG_OPT_RE.findall(m.group(2)))
        if opts:
            out.add((m.group(1), opts))
    return sorted(out)


@pytest.fixture(scope="module")
def _declared_options():
    """Options longues que chaque script declare, lues a son `--help`."""
    cache = {}

    def get(script):
        if script not in cache:
            r = subprocess.run(
                [sys.executable, os.path.join(_REPO_ROOT, script), "--help"],
                capture_output=True, text=True, timeout=300)
            cache[script] = (set(_LONG_OPT_RE.findall(r.stdout + r.stderr))
                             if r.returncode == 0 else None)
        return cache[script]

    return get


def test_every_repro_command_uses_options_its_script_declares(
        results_md, _declared_options):
    commands = _commands_with_options(results_md)
    # Balayage vide : sans ce garde, un motif qui cesse de correspondre
    # rendrait ce test vert sans rien verifier. Mesure du jour : 16.
    assert len(commands) >= 10, (
        f"le balayage n'a trouve que {len(commands)} commande(s) a options "
        "dans RESULTS.md : c'est le motif qui a cesse de correspondre, pas "
        "le depot qui n'a plus de commandes")
    faulty = []
    for script, opts in commands:
        if not os.path.exists(os.path.join(_REPO_ROOT, script)):
            continue                      # couvert par le test des chemins
        declared = _declared_options(script)
        if declared is None:
            continue                      # `--help` ne rend pas 0 : hors portee
        missing = sorted(o for o in opts if o not in declared)
        if missing:
            faulty.append(f"{script} : {' '.join(missing)}")
    assert not faulty, (
        "commandes de RESULTS.md citant une option que leur script ne "
        f"declare pas — elles sortent en 2 sans rien mesurer : {faulty}")


@pytest.mark.parametrize("relpath", [
    "docs/RESULTS.md",
    "docs/DEFAUTS.md",
    "docs/COUVERTURE.md",
    "docs/EVALUATION.md",
    "scripts/run_fold.sh",
    "scripts/run_leak_free_campaign.sh",
])
def test_no_dead_v4_prefix_outside_documented_history(relpath):
    """`study/v4/` n'existe plus : toute occurrence vivante est une commande
    cassee. Les exceptions historiques sont nommees, pas devinees."""
    path = os.path.join(_REPO_ROOT, relpath)
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    allowed = sum(1 for fname, _frag in _HISTORICAL_EXCEPTIONS if fname == relpath)
    hits = [ln for ln in lines if "study/v4/" in ln]
    assert len(hits) <= allowed, (
        f"{relpath} porte {len(hits)} occurrence(s) de 'study/v4/' non "
        f"documentee(s) comme historique (autorise : {allowed}) : {hits}")
