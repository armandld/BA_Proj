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
# SEUIL PERIME, REMESURE (D-111, section « remesure ») — la liste est VIDE
# depuis la fusion `766d289` de la branche vive.
# Elle portait 11 entrees, toutes des cibles de `scripts/run_study_v3.sh`
# tenues pour mortes par D-49 (« les 9 generateurs v3 n'existent plus sous
# aucun chemin »). D-116, sur la branche vive, a repointe ce lanceur : les
# generateurs existent, RENOMMES, dans `study/h2b_prediction/`
# (t1_feature_selection -> h2b_feature_selection, t4_blocked_split ->
# h2b_blocked_split, …) — la meme table de renommage que D-76 avait deja
# etablie pour `run_study_v2*.sh`. Mesure, sur ce fichier, meme commande
# (`pytest tests/test_launcher_paths_resolve.py -q`) :
#
#   avant (bff6bd3) : 45 invocations balayees, 11 exemptions, 11/11 invoquees
#   apres (766d289) : 79 invocations balayees,  0 exemption,  0 cible morte
#
# La liste reste en place, vide : elle est le point d'accroche du jour ou un
# lanceur redesignera une cible morte pour une raison decidee.
_EXEMPTIONS = {}

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


#  `$(dirname "${BASH_SOURCE[0]}")` et `$(dirname "$0")` — les deux facons
#  dont un lanceur de ce depot nomme son propre dossier. Sans cette
#  substitution, `_DEF_CD` bute sur les guillemets internes et la variable
#  reste non resolue : `run_fold.sh` definit ainsi `root`, et ses deux
#  invocations disparaissaient du balayage (mesure D-151 : 83 -> 81).
_DIRNAME_DE_SOI = re.compile(
    r'\$\(dirname\s+"(?:\$\{BASH_SOURCE\[0\]\}|\$0)"\)')


def _launcher_vars(path):
    """Les variables de chemin que CE lanceur se donne, evaluees."""
    here = os.path.dirname(path)
    variables = {"SCRIPT_DIR": here, "HERE": here}
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.lstrip().startswith("#"):
            continue
        line = _DIRNAME_DE_SOI.sub(here, line)
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
#
# `"$PYTHON_BIN"` — la variable que les lanceurs recents (`run_study_v3.sh`,
# `run_reoptimisation.sh`, `run_fold.sh`, `repetition_campagne.sh`,
# `run_dns_campaign.sh`, `run_confirmatory_campaign.sh`,
# `run_rented_campaign.sh`) resolvent vers `.venv/bin/python` si present,
# `python3` sinon (portabilite) — ne porte litteralement aucun jeton
# `python` : sans cette alternative le motif ne matche rien sur ces lignes,
# la cible n'est jamais extraite, et `_expand` ne voit meme pas passer un
# `$` a rejeter. Mesure D-194 : balayage a 35 invocations sans cette
# alternative (six lanceurs a 0 ou 1, alors qu'ils en portent des dizaines),
# 83 avec — la meme famille de defaut que D-151 (le `cd` isole) et
# `_DIRNAME_DE_SOI` : un style d'ecriture legitime, plus recent que le
# motif, invisible pour lui.
_INVOKE = re.compile(
    r"(?:cd\s+([\w./${}-]+)\s*&&\s*)?"
    r"(?:\"?(?:python3?|\$\{?PYTHON_BIN\}?)\"?\s+(?:-m\s+pytest\s+)?|bash\s+|sh\s+)"
    r"\"?([\w./${}-]+\.(?:py|sh))"
)

# Les lanceurs passent aussi par une fonction d'enrobage — `run_phase 2
# study/pipeline/hard_patch_labels.py`, `run_step t1 "$PYTHON_BIN" …` — ou le
# `python`/`$PYTHON_BIN` est DANS la fonction, pas sur la ligne. Sans ce
# second motif, les 5 etages de `run_study_v2_phases.sh` ne sont vus par
# personne : le balayage passe au vert sans les avoir regardes. Meme
# alternative `$PYTHON_BIN` que `_INVOKE`, meme raison (D-194).
_WRAPPED = re.compile(
    r"run_(?:phase|step|stage)\s+\S+\s+"
    r"(?:\"?(?:python3?|\$\{?PYTHON_BIN\}?)\"?\s+(?:-m\s+pytest\s+)?)?"
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


#  D-151 — un `cd` sur sa propre ligne vaut pour toutes les lignes suivantes.
#
#  `_INVOKE` ne reconnaît `cd X && python Y` que sur UNE ligne. Or les
#  lanceurs de ce depot font le `cd` a part (`run_reoptimisation.sh:69`,
#  `run_fold.sh:27`, `run_study_v2_phases.sh:27`) : le dossier courant valait
#  alors la racine du depot pour tout ce qui suit. Deux consequences, dans les
#  deux sens :
#
#    - FAUX ROUGE : `cd "$ROOT_DIR/src"` puis `python train_hyperparams.py`
#      etait resolu en `train_hyperparams.py` a la racine, absent — le garde
#      rougissait sur un lanceur JUSTE (mesure : `run_reoptimisation.sh:72`) ;
#    - FAUX VERT : la meme resolution valide un HOMONYME. Un lanceur qui,
#      depuis `scripts/`, invoquerait `run_tests.sh` — qui n'y existe pas —
#      etait valide par le `run_tests.sh` de la racine.
#
#  On ne retient que le `cd` seul sur sa ligne : les `$(cd … && pwd)` des
#  definitions de variables sont deja traites par `_launcher_vars`, et un
#  `cd … && …` inline reste du ressort de `_INVOKE`.
_CD_SEUL = re.compile(r'^\s*cd\s+"?([^"&|;#]+?)"?\s*(?:#.*)?$')


def _invocations(path):
    """(chemin_relatif_au_depot, numero_de_ligne) de chaque fichier invoque."""
    variables = _launcher_vars(path)
    out = []
    cwd_courant = ""          # "" = racine du depot ; None = inconnu
    for lineno, line in enumerate(open(path, encoding="utf-8",
                                       errors="replace"), start=1):
        if line.lstrip().startswith("#"):
            continue
        if line.lstrip().startswith("echo"):
            #  Un `echo` n'invoque rien : il donne au lecteur une commande a
            #  taper, et le lecteur la tape depuis la racine du depot. La
            #  cible se resout donc contre la racine, quel que soit le `cd`
            #  du lanceur (`run_reoptimisation.sh:76`).
            for cwd, target in _INVOKE.findall(line):
                target = _expand(target, variables)
                if "$" in target or target.startswith("-"):
                    continue
                out.append((os.path.relpath(
                    os.path.normpath(os.path.join(_ROOT, _expand(cwd, variables),
                                                  target)), _ROOT), lineno))
            continue
        m = _CD_SEUL.match(line.rstrip("\n"))
        if m:
            cible = _expand(m.group(1).strip(), variables)
            #  Un `cd` qu'on ne sait pas resoudre rend le dossier courant
            #  INCONNU : mieux vaut ne rien affirmer sur les lignes qui
            #  suivent que les resoudre contre la racine, ce qui est
            #  precisement le defaut corrige ici.
            cwd_courant = cible if "$" not in cible else None
            continue
        trouves = [(cwd, tgt) for cwd, tgt in _INVOKE.findall(line)]
        trouves += [("", tgt) for tgt in _WRAPPED.findall(line)
                    if not any(tgt == t for _, t in trouves)]
        for cwd, target in trouves:
            cwd = _expand(cwd, variables)
            target = _expand(target, variables)
            if not cwd:
                if cwd_courant is None:
                    continue      # `cd` non resolu : on n'affirme rien
                cwd = cwd_courant
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


def test_the_figure_launcher_never_writes_over_the_frozen_hyperparams():
    """Ce qui RESTE de D-111 apres que D-117 a tranche le reste.

    D-111 laissait trois cibles en suspens (`figures_code/`, `Train_results/`,
    `best_hyperparams.json`) parce que les rebrancher etait une DECISION.
    D-117 l'a prise, et bien : les deux premieres pointent desormais vers
    `figures/v1_legacy/` et `results/hyperparams/optuna_studies/`, et la
    troisieme — la seule qui portait un risque — ecrit dans un fichier
    `.regenerated.json` distinct.

    Ce qui n'a pas change, c'est la RAISON : `results/hyperparams/` est la
    seule entree du depot que `CLAUDE.md` declare gelee et non reproductible
    par une commande (son `PROVENANCE.md`, et D-22). La deviation vit donc
    toujours, et se garde ici — sur le COMPORTEMENT du lanceur, pas sur le
    texte de sa note (D-114).

    Sur quelle entree ce test echoue
    --------------------------------
    Sur un `--output` du lanceur repointe vers le fichier gele, quelle que
    soit la mise en forme de la ligne. Verifie en editant `--output` de
    `scripts/generate_figures_v1.sh` : VERT -> ROUGE.
    """
    path = os.path.join(_ROOT, "scripts", "generate_figures_v1.sh")
    txt = open(path, encoding="utf-8").read()
    variables = _launcher_vars(path)

    sorties = [_expand(m.group(1), variables)
               for m in re.finditer(r'--output\s+"([^"]+)"', txt)]
    assert sorties, ("aucun `--output` lu dans le lanceur : balayage vide, "
                     "le test ne prouve rien")

    gele = os.path.normpath(
        os.path.join(_ROOT, "results", "hyperparams", "best_hyperparams.json"))
    for cible in sorties:
        assert os.path.normpath(cible) != gele, (
            "le lanceur ecrirait par-dessus l'entree GELEE "
            "results/hyperparams/best_hyperparams.json (PROVENANCE.md, D-22, "
            "D-111) : ecrire a cote, comme le fait `.regenerated.json`")


def test_each_exemption_still_names_a_real_dead_path():
    """Une exemption perimee est un mensonge qui dort. Le jour ou l'un de ces
    chemins revient, l'exemption doit tomber avec lui.

    La liste est vide depuis la remesure en tete de fichier ;
    le balayage, lui, ne doit jamais l'etre — une table vide et un balayage
    vide se ressemblent, et l'un des deux est un defaut.

    D-194 : le plancher de 79 (mesure a `766d289`, avant la consolidation
    single-machine de `CODE_REVIEW.md`) datait d'un jeu de lanceurs qui
    n'existe plus — 4 supprimes (`run_leak_free_campaign.sh`,
    `run_study_v2_phases.sh`, `run_study_v2b.sh`, `soumettre_campagne.sh`),
    remplaces par 3 plus petits (`run_confirmatory_campaign.sh`,
    `run_dns_campaign.sh`, `run_rented_campaign.sh`). Rejoue tel quel contre
    le nouveau jeu, le balayage tombait a 35 — mais pas par perte : les
    lanceurs survivants et les trois nouveaux resolvent tous leur python via
    une variable `$PYTHON_BIN` (portabilite `.venv`/`python3`) que `_INVOKE`/
    `_WRAPPED` ne reconnaissaient pas, une categorie de defaut deja vue deux
    fois dans ce fichier (D-151, `_DIRNAME_DE_SOI`). Corrige : 35 -> 61.
    Verifie site par site sur `run_study_v3.sh` (le plus gros ecart)
    qu'aucune invocation reelle ne manque encore : 10 lignes hors
    commentaire portent `.py`/`.sh`, 10 sont vues. 61 est donc un plancher
    mesure sur le jeu ACTUEL, pas une valeur abaissee sans le savoir.
    """
    assert len(_ALL) >= 61, (
        f"{len(_ALL)} invocations balayees, 61 mesurees le 25 aout apres "
        "correction D-194 (`$PYTHON_BIN` non reconnu) : le balayage a "
        "retreci, il ne prouve plus ce qu'il prouvait")
    for target, raison in _EXEMPTIONS.items():
        assert raison.strip(), f"{target} exempte sans raison"
        assert not os.path.exists(os.path.join(_ROOT, target)), (
            f"{target} existe de nouveau : retirer son exemption")
        assert any(t == target for _, t, _ in _ALL), (
            f"{target} n'est plus invoque par aucun lanceur : "
            "retirer son exemption")


# ══════════════════════════════════════════════════════════════════
#  D-151 — le dossier courant d'un `cd` pose sur sa propre ligne
# ══════════════════════════════════════════════════════════════════

def test_a_cd_on_its_own_line_moves_the_targets_that_follow():
    """A standalone `cd` changes how subsequent targets are resolved."""
    import tempfile
    with tempfile.NamedTemporaryFile(
            "w", suffix=".sh", dir=_ROOT, delete=False) as fh:
        fh.write("cd src\npython train_hyperparams.py\n")
        path = fh.name
    try:
        assert _invocations(path) == [("src/train_hyperparams.py", 2)]
    finally:
        os.remove(path)


def test_a_cd_inside_a_command_substitution_does_not_move_what_follows():
    """L'autre moitie de D-151, exercee depuis le 18 aout par le lanceur.

    `ETUDE="$(cd "$ROOT_DIR/src" && python -c ...)"` contient un `cd`, mais
    dans un sous-shell : il ne deplace PAS le dossier courant des lignes
    suivantes. Un parseur qui le prendrait pour un `cd` reel resoudrait
    toutes les cibles suivantes sous `src/` et les declarerait absentes.

    Sur quelle entree ce test echoue : sur un parseur qui traite n'importe
    quel `cd` de la ligne comme un changement de dossier persistant.
    """
    lanceur = os.path.join(_ROOT, "scripts", "run_reoptimisation.sh")
    lignes = open(lanceur, encoding="utf-8").read().splitlines()
    sub_ln = next((i + 1 for i, l in enumerate(lignes)
                   if l.startswith("ETUDE=") and "cd " in l), None)
    if sub_ln is None:
        pytest.skip("le lanceur ne contient plus de `cd` en substitution")

    cibles = dict((ln, t) for t, ln in _invocations(lanceur))
    suivantes = {ln: t for ln, t in cibles.items() if ln > sub_ln}
    assert suivantes, "aucune invocation apres la substitution : rien a prouver"
    for ln, cible in sorted(suivantes.items()):
        assert os.path.exists(os.path.join(_ROOT, cible)), (
            f"ligne {ln} resolue en {cible!r}, qui n'existe pas : le `cd` "
            f"en sous-shell de la ligne {sub_ln} a ete pris pour un `cd` reel")


def test_a_homonym_at_the_repository_root_no_longer_validates_the_target(tmp_path):
    """L'autre moitie du defaut, celle qui rendait le garde COMPLAISANT.

    Resoudre contre la racine ne rate pas seulement une cible : elle en
    valide une autre. Un lanceur qui, apres `cd`, invoque un nom qui
    n'existe QUE a la racine etait declare sain par l'homonyme.

    Champ d'essai qui SEPARE : `run_tests.sh` existe a la racine du depot et
    nulle part ailleurs.
    """
    lanceur = tmp_path / "faux_lanceur.sh"
    lanceur.write_text(
        'root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'cd "$root/sous_dossier"\n'
        'bash run_tests.sh\n')
    (tmp_path / "sous_dossier").mkdir()

    cibles = [t for t, _ in _invocations(str(lanceur))]
    assert cibles, "balayage vide : le test ne prouve rien"
    assert "run_tests.sh" not in cibles, (
        "la cible est resolue contre la racine du depot : le `run_tests.sh` "
        "de la racine valide un fichier que le lanceur n'atteindrait jamais")
    assert not any(os.path.exists(os.path.join(_ROOT, t)) for t in cibles), (
        f"{cibles} : aucune de ces cibles ne devrait exister")


def test_an_unresolved_cd_drops_the_lines_that_follow_rather_than_guessing():
    """Decision ecrite la ou elle vit : `cd` inconnu -> on n'affirme rien.

    Le silence a un cout — ces lignes sortent du balayage — donc il est
    borne par `test_each_exemption_still_names_a_real_dead_path`, qui exige
    un plancher d'invocations, et par le test ci-dessous qui epingle le
    total mesure.
    """
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write('cd "$UNE_VARIABLE_QUE_RIEN_NE_DEFINIT"\n'
                 'python study/common/provenance.py\n')
        chemin = fh.name
    try:
        assert _invocations(chemin) == []
    finally:
        os.remove(chemin)


def test_the_sweep_still_sees_every_invocation_it_saw_before():
    """Un parseur plus fin peut aussi voir MOINS : ce test l'interdit.

    83 invocations mesurees a `f8edebf`, 80 apres retrait de trois
    utilitaires de stockage obsoletes : c'etait la mesure sur le jeu de
    lanceurs d'alors. D-194 (25 aout) l'a remesuree sur le jeu ACTUEL,
    apres la consolidation single-machine qui a supprime 4 lanceurs et en
    a ajoute 3 differents (voir `test_each_exemption_still_names_a_real_
    dead_path`, qui documente le detail) : 61, plancher verifie complet,
    pas simplement abaisse pour que ce test passe.
    """
    assert len(_ALL) >= 61, (
        f"{len(_ALL)} invocations balayees, 61 attendues apres la "
        "correction D-194 du non-reconnu `$PYTHON_BIN`")
