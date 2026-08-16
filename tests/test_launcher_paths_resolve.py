"""D-110 — les lanceurs `.sh` se testent aussi : leurs chemins existent-ils ?

Même panne que celle qui a donné `test_suite_integrity.py`, un cran plus
haut. La réorganisation de `tests/` (`17d983d`) a déplacé les fichiers vers
`solver/`, `mapping/`, `quantum/`, `amr/`, `pipeline/`, `study/`, `tools/`.
Les imports croisés ont été rattrapés (D-71, puis `test_suite_integrity.py`) ;
**les lanceurs `.sh`, non**. `run_tests.sh` — celui que `README.md` documente
(`bash run_tests.sh  # Run full test suite`) et que
`docs/protocol_v3_evaluation.md` érige en critère d'acceptation (« `bash
run_tests.sh` must pass unchanged after every task ») — désignait encore
l'arborescence plate.

Mesure, avant correction : **17 commandes d'étage sur 17** échouaient, toutes
sur `file or directory not found` (rc 4 pour les appels `pytest`, rc 2 pour
les 5 scripts autonomes). `run_stage` sort au premier code non nul
(`run_tests.sh:154`) et le script est `set -e` : il s'arrêtait à l'étage 1.
**0 test atteignable.** Après : **168 tests sélectionnés** sur 12 étages
`pytest`, et les 5 scripts autonomes résolvent.

Sur quelle entrée ce test échoue
--------------------------------
Sur tout lanceur qui invoque un fichier absent. Il aurait rougi dès le
commit de réorganisation. Il rougira au prochain déplacement — c'est son
seul objet.

Il porte sur le **comportement** (le fichier désigné existe-t-il ?), pas sur
la mise en forme du source : la ligne peut être réécrite comme on veut tant
que la cible résout.
"""
import os
import re

import pytest


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_ROOT = _repo_root()

# Chemins connus pour ne plus exister, avec leur raison. Une exemption sans
# raison se fait oublier ; une exemption perimee doit crier (test plus bas).
#
# Les chemins sont donnes TELS QUE LE LANCEUR LES CALCULE — pour
# `scripts/run_study_v3.sh` cela inclut son `../`, qui est lui-meme mesure et
# volontairement NON corrige : son en-tete dit « ne pas debugger les chemins
# en pensant le remettre en etat sans lire D-49 d'abord ». Un gel documente ne
# se corrige pas au passage.
_EXEMPTIONS = {
    # D-49, entree ouverte : les 9 generateurs v3 n'existent plus sous aucun
    # chemin. Leur sort (archiver le module, ou les reconstruire) est une
    # decision de USER, pas une correction a faire ici.
    "../study/v3/t1_feature_selection.py": "D-49 — generateur v3 supprime",
    "../study/v3/t1b_cone_curve.py": "D-49 — generateur v3 supprime",
    "../study/v3/t4_blocked_split.py": "D-49 — generateur v3 supprime",
    "../study/v3/t5_v1_psi_loso.py": "D-49 — generateur v3 supprime",
    "../study/v3/t6_dynamic_gt.py": "D-49 — generateur v3 supprime",
    "../study/v3/t7_horizon.py": "D-49 — generateur v3 supprime",
    "../study/v3/t9_prop2_check.py": "D-49 — generateur v3 supprime",
    # Celui-ci n'est pas invoque mais IMPRIME par un `echo` de conseil
    # (run_study_v3.sh:80) : meme cause, meme decision en attente.
    "study/v3/dns_extension.py":
        "D-49 — cite dans un message d'aide, generateur v3 supprime",
    "../study/v3/aggregate_v3.py":
        "D-49 — le chemin reel est `study/common/aggregate_v3.py` ; le "
        "lanceur porte deja l'avertissement",
    "../study/phase11_upper_bound.py":
        "D-49 — generateur v3 supprime (chemin reel : "
        "`study/h2b_prediction/phase11_upper_bound.py`, autre module)",
    "../study/phase11b_loso.py": "D-49 — generateur v3 supprime",
}

# Variables de chemin, LUES dans le lanceur au lieu d'etre recopiees ici.
# Une table ecrite a la main resterait juste le jour ou le lanceur change de
# racine : le test ne pourrait plus echouer sur exactement le defaut qu'il
# garde (D-111). On evalue donc les deux formes que les lanceurs emploient :
#
#     VAR="$(cd "<expr>" && pwd)"      et      VAR="$AUTRE"
#
# avec `SCRIPT_DIR` / `HERE` = le dossier du lanceur. Sans cette resolution le
# balayage rate 12 invocations sur 45 — et un balayage qui ne voit rien sort
# en vert.
def _expand(chunk, variables):
    """Remplace `$VAR` / `${VAR}` par la valeur que le lanceur lui donne."""
    for name, value in variables.items():
        chunk = chunk.replace("${" + name + "}", value).replace("$" + name,
                                                                value)
    return chunk


_DEF_CD = re.compile(r'^\s*(\w+)="\$\(cd\s+"([^"]+)"\s*&&\s*pwd\)"')
_DEF_ALIAS = re.compile(r'^\s*(\w+)="\$\{?(\w+)\}?"\s*$')


def _launcher_vars(path):
    """Les variables de chemin que CE lanceur se donne, evaluees."""
    here = os.path.dirname(path)
    variables = {"SCRIPT_DIR": here, "HERE": here}
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.lstrip().startswith("#"):
            continue
        m = _DEF_CD.match(line)
        if m:
            name, expr = m.group(1), _expand(m.group(2), variables)
            if "$" not in expr:
                variables[name] = os.path.normpath(expr)
            continue
        m = _DEF_ALIAS.match(line)
        if m and m.group(2) in variables:
            variables[m.group(1)] = variables[m.group(2)]
    return variables


# `python <fichier>`, `python -m pytest <fichier>`, `bash <fichier>`,
# et la forme `cd <dossier> && python <fichier>` des scripts autonomes.
_INVOKE = re.compile(
    r"(?:cd\s+([\w./${}-]+)\s*&&\s*)?"
    r"(?:python3?\s+(?:-m\s+pytest\s+)?|bash\s+|sh\s+)"
    r"\"?([\w./${}-]+\.(?:py|sh))"
)

# Les lanceurs passent aussi par une fonction d'enrobage — `run_phase 2
# study/pipeline/hard_patch_labels.py`, `run_step t1 python …` — ou le
# `python` est DANS la fonction, pas sur la ligne. Sans ce second motif, les
# 5 etages de `run_study_v2_phases.sh` ne sont vus par personne : le balayage
# passe au vert sans les avoir regardes.
_WRAPPED = re.compile(
    r"run_(?:phase|step|stage)\s+\S+\s+(?:python3?\s+(?:-m\s+pytest\s+)?)?"
    r"\"?([\w./${}-]+\.(?:py|sh))"
)


def _launchers():
    out = []
    for base in (_ROOT, os.path.join(_ROOT, "scripts")):
        if not os.path.isdir(base):
            continue
        out.extend(os.path.join(base, f) for f in sorted(os.listdir(base))
                   if f.endswith(".sh"))
    return sorted(out)


def _invocations(path):
    """(chemin_relatif_au_depot, numero_de_ligne) de chaque fichier invoque."""
    variables = _launcher_vars(path)
    out = []
    for lineno, line in enumerate(open(path, encoding="utf-8",
                                       errors="replace"), start=1):
        if line.lstrip().startswith("#"):
            continue
        trouves = [(cwd, tgt) for cwd, tgt in _INVOKE.findall(line)]
        trouves += [("", tgt) for tgt in _WRAPPED.findall(line)
                    if not any(tgt == t for _, t in trouves)]
        for cwd, target in trouves:
            cwd = _expand(cwd, variables)
            target = _expand(target, variables)
            if "$" in target or "$" in cwd or target.startswith("-"):
                continue          # chemin encore construit a l'execution
            base = cwd if os.path.isabs(cwd) else os.path.join(
                os.path.dirname(path) if cwd else _ROOT, cwd)
            full = target if os.path.isabs(target) else os.path.join(base,
                                                                     target)
            out.append((os.path.relpath(os.path.normpath(full), _ROOT),
                        lineno))
    return out


_ALL = [(lch, tgt, ln)
        for lch in _launchers()
        for tgt, ln in _invocations(lch)]


def test_the_sweep_is_not_empty():
    """Un balayage vide sort en vert et ne prouve rien."""
    assert len(_launchers()) >= 6, f"{len(_launchers())} lanceurs trouves"
    assert len(_ALL) >= 45, f"{len(_ALL)} invocations trouvees"


@pytest.mark.parametrize(
    "launcher,target,lineno", _ALL,
    ids=[f"{os.path.relpath(l, _ROOT)}:{n}:{t}" for l, t, n in _ALL])
def test_every_file_a_launcher_invokes_exists(launcher, target, lineno):
    if target in _EXEMPTIONS:
        pytest.skip(f"exemption documentee : {_EXEMPTIONS[target]}")
    assert os.path.exists(os.path.join(_ROOT, target)), (
        f"{os.path.relpath(launcher, _ROOT)}:{lineno} invoque `{target}`, "
        f"qui n'existe pas")


def test_run_tests_reaches_every_one_of_its_seventeen_stages():
    """Le lanceur du dépôt, nommement — c'est lui que D-110 a trouvé mort.

    Mesure épinglée : 17 commandes d'étage distinctes, 32 appels `run_stage`.
    Si ces nombres bougent, la mesure de D-110 est à refaire.
    """
    path = os.path.join(_ROOT, "run_tests.sh")
    txt = open(path, encoding="utf-8").read()
    calls = re.findall(r'run_stage\s+"([^"]+)"\s+(.*)', txt)
    assert len(calls) == 32, f"{len(calls)} appels run_stage (32 mesures)"
    assert len({c for c in calls}) == 17, "17 commandes distinctes attendues"

    manquants = [(n, t) for lch, t, _ in _ALL
                 for n in [os.path.relpath(lch, _ROOT)]
                 if n == "run_tests.sh"
                 and not os.path.exists(os.path.join(_ROOT, t))]
    assert manquants == [], (
        "run_tests.sh ne peut pas atteindre ces etages : " + str(manquants))


def test_the_undecided_remainder_of_D111_stays_written_where_it_lives():
    """Une deviation connue mais non consignee *la ou elle vit* se fait
    recorriger par erreur.

    D-111 corrige le `ROOT_DIR` de `scripts/generate_figures_v1.sh`, et
    s'arrete la : trois cibles (`figures_code/`, `Train_results/`, le
    `best_hyperparams.json` racine) n'existent plus sous aucune racine, et les
    rebrancher est une DECISION — la derniere ecraserait une entree gelee.
    Ce test verifie que la note le disant reste dans le fichier.
    """
    txt = open(os.path.join(_ROOT, "scripts", "generate_figures_v1.sh"),
               encoding="utf-8").read()
    assert "D-111" in txt, "la note de D-111 a disparu du lanceur"
    for cible in ("figures_code/", "Train_results/", "best_hyperparams.json"):
        assert cible in txt, f"la note ne mentionne plus {cible}"
    # Et la mesure qui la motive tient toujours : ces cibles n'existent pas.
    for cible in ("figures_code", "Train_results", "best_hyperparams.json"):
        assert not os.path.exists(os.path.join(_ROOT, cible)), (
            f"{cible} existe de nouveau a la racine : la note de D-111 est "
            "perimee, la retirer et rebrancher le lanceur")


def test_each_exemption_still_names_a_real_dead_path():
    """Une exemption perimee est un mensonge qui dort. Le jour ou l'un de ces
    chemins revient, l'exemption doit tomber avec lui."""
    assert _EXEMPTIONS, "liste d'exemptions vide : retirer le test"
    for target, raison in _EXEMPTIONS.items():
        assert raison.strip(), f"{target} exempte sans raison"
        assert not os.path.exists(os.path.join(_ROOT, target)), (
            f"{target} existe de nouveau : retirer son exemption")
        assert any(t == target for _, t, _ in _ALL), (
            f"{target} n'est plus invoque par aucun lanceur : "
            "retirer son exemption")
