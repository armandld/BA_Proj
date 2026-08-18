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

#  D-160 — la commande citee AU FIL DU TEXTE.
#
#  `_INLINE_CMD_RE` exige que la ligne soit ENTIEREMENT un span de code et
#  que le chemin y vienne en premier. La forme la plus naturelle de citer
#  une commande en prose — « reproduire : `python study/x.py --dim 16` » —
#  n'y repond ni par sa position dans la ligne, ni par son prefixe `python`.
#  Mesure du 18 aout 2026 : **10 commandes de `RESULTS.md`** ecrites ainsi,
#  invisibles au balayage des chemins. Toutes vivantes — le trou etait reel
#  et sans consequence, il est ferme pendant qu'il l'est.
#
#  Question 4 : `_commands_with_options` (D-140) les voit et verifie leurs
#  OPTIONS ; le balayage des chemins, non. Deux balayages du meme document,
#  l'un voit une commande que l'autre ignore.
#
#  Ce motif-ci ne peut pas attraper une citation narrative
#  (« `study/phase0_sanity_check.py:95` ») : il exige le prefixe `python` /
#  `bash`, qu'aucune citation de ce genre ne porte.
_INLINE_INVOKE_RE = re.compile(r"`(?:nohup\s+)?(?:python|bash)\s+" + _PATH + r"[^`]*`")

#  D-160 — lignes deliberement historiques : narration d'une campagne
#  passee, pas une commande a rejouer.
#
#  L'ancienne version listait des couples (fichier, jeton) et n'en gardait
#  que le COMPTE : `allowed = sum(1 for f, _frag in … if f == relpath)`. Le
#  jeton — `"d71-entry"`, `"trap-sweep"` — n'etait **jamais confronte au
#  fichier**, et n'y figurait d'ailleurs pas. Le test verifiait donc « pas
#  plus de deux occurrences », pas « ces deux occurrences-la ». Mesure :
#  remplacer une mention historique par `python study/v4/t31_….py --dim 16`
#  laisse le compte a 2 et le fichier **vert** — c'est la forme de D-136
#  (une chaine presente deux fois, dont une seule suffit).
#
#  Chaque exemption porte desormais un fragment qui doit se trouver SUR LA
#  LIGNE qui cite `study/v4/`, et chaque fragment declare doit encore
#  correspondre a une ligne — une exemption qui pourrit est pire que pas
#  d'exemption.
_HISTORICAL_EXCEPTIONS = {
    #  la ligne de registre D-71 elle-meme : elle DECRIT le deplacement
    ("docs/RESULTS.md", "| D-71 |"),
    #  le balayage des sites `run_arm`, narration d'un etat passe
    ("docs/RESULTS.md", "run_arm"),
    #  meme entree cote couverture, section h1_solver_convergence.py
    ("docs/COUVERTURE.md", "D-71"),
    #  commentaire D-71 : explique la cause, n'invoque rien
    ("scripts/run_fold.sh", "# D-71"),
    ("scripts/run_leak_free_campaign.sh", "# D-71"),
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
        #  D-160 : une commande citee au fil du texte, ou qu'elle soit
        #  dans la ligne. Le prefixe `python`/`bash` la distingue d'une
        #  citation narrative.
        found.update(m.group(1) for m in _INLINE_INVOKE_RE.finditer(raw))
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


# ── D-142 : la moitie `tests/` n'etait pas balayee du tout ────────────
#
# `_PATH` ci-dessus couvre `study|scripts|figures`. Les commandes de
# reproduction les plus nombreuses de `RESULTS.md` sont des `pytest
# tests/...` : elles n'etaient regardees par rien. Dix chemins y etaient
# restes a leur emplacement d'avant `17d983d` — meme cause que D-71, sur
# la moitie que son garde ne voyait pas. Deux blocs de recette entiers
# sortaient en **4**, `file or directory not found`.
#
# Deux differences avec le balayage de D-71, imposees par la forme reelle
# de ces commandes :
#   - une commande `pytest` s'etale sur plusieurs lignes, par `\` dans un
#     bloc cloture ou par simple retour a la ligne dans un span de code
#     inline : il faut suivre le CONTEXTE, pas juger ligne a ligne ;
#   - les lignes de TABLE (`| ... |`) citent des chemins historiques
#     (inventaire de la suite QAOA d'avant la reorganisation) et ne sont
#     pas des commandes : elles sont exclues par leur forme, pas par une
#     liste d'exceptions a tenir a la main.

_TEST_PATH_RE = re.compile(r"(tests/[A-Za-z0-9_./-]+\.py)")
_CMD_START_RE = re.compile(r"(?:^|`)(?:python -m |nohup )?pytest\b|^(?:python|bash)\s")


def _test_paths_in_commands(text):
    """Chemins `tests/...py` cites dans une COMMANDE, continuations comprises."""
    found, in_fence, suite = set(), False, False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_fence, suite = not in_fence, False
            continue
        if line.startswith("|"):              # table : inventaire, pas commande
            continue
        debut = bool(_CMD_START_RE.search(line))
        if in_fence:
            if debut:
                suite = True
            elif not (line.startswith("tests/") or line.startswith("-")):
                suite = False
        else:
            if not (debut or suite):
                continue
            # un span de code inline non ferme continue a la ligne suivante
            suite = (line.count("`") % 2 == 1) or (suite and "`" not in line)
        if debut or suite:
            found.update(_TEST_PATH_RE.findall(line))
    return found


def test_every_pytest_command_in_results_md_points_to_a_real_file(results_md):
    referenced = _test_paths_in_commands(results_md)
    # Anti-balayage-vide : mesure du jour, 29 chemins distincts.
    assert len(referenced) >= 20, (
        f"le balayage n'a trouve que {len(referenced)} chemin(s) `tests/` "
        "dans les commandes de RESULTS.md : c'est le motif qui a cesse de "
        "correspondre, pas le depot qui n'a plus de commandes pytest")
    missing = sorted(p for p in referenced
                     if not os.path.exists(os.path.join(_REPO_ROOT, p)))
    assert not missing, (
        "commandes `pytest` de RESULTS.md pointant sur des fichiers absents "
        f"— elles sortent en 4 sans rien mesurer : {missing}")


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

    D-156 — pourquoi ce n'est plus un aplatissement global. L'ancienne
    version faisait `re.sub(r"`([^`]*)`", …)` sur tout le document pour
    recoller les spans de code inline coupes en deux lignes. Un bloc
    ``` ``` ``` est fait de backquotes lui aussi : le motif appariait le
    DERNIER backquote de la cloture ouvrante avec le PREMIER de la
    cloture fermante, et aplatissait le bloc entier sur une ligne. Deux
    consequences, mesurees sur `RESULTS.md` au 18 aout 2026 :

      - `_PY_CMD_RE` consomme `[^\\n`|]*` : la PREMIERE commande du bloc
        avalait les options de toutes les suivantes — `--force-greedy`,
        option d'une seconde commande `h3_size_scan.py`, etait rapportee
        sur la premiere ;
      - `finditer` reprend apres ce qui a ete consomme : **les commandes
        suivantes du bloc n'etaient jamais vues**. 14 commandes lues sur
        21 reellement presentes.

    Le bloc cloture se lit donc ligne par ligne, avec ses continuations
    `\\`, et le span inline reste recolle — mais seulement lui.
    """
    out = set()
    lignes = text.splitlines()
    dans_bloc = False
    i = 0
    while i < len(lignes):
        ligne = lignes[i].strip()
        if ligne.startswith("```"):
            dans_bloc = not dans_bloc
            i += 1
            continue
        if dans_bloc:
            cmd = ligne
            while cmd.endswith("\\") and i + 1 < len(lignes):
                i += 1
                cmd = cmd[:-1] + " " + lignes[i].strip()
            #  le commentaire de fin de ligne n'est pas une option
            out |= _options_de(cmd.split("#")[0])
        i += 1
    #  Spans de code inline : eux se recollent, c'est leur forme. Les
    #  clotures sont neutralisees d'abord pour que le motif ne les apparie
    #  pas — c'est exactement le defaut D-156.
    sans_bloc = text.replace("```", "\x00")
    for m in re.finditer(r"`([^`]*)`", sans_bloc):
        out |= _options_de(m.group(1).replace("\n", " "))
    return sorted(out)


def _options_de(fragment):
    """{(script, options)} pour UNE commande deja isolee."""
    out = set()
    for m in _PY_CMD_RE.finditer(fragment):
        opts = frozenset(_LONG_OPT_RE.findall(m.group(2)))
        if opts:
            out.add((m.group(1), opts))
    return out


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


#: D-156 — perimetre. La verification des options ne lisait que
#: `RESULTS.md`. Les commandes que quelqu'un va reellement retaper vivent
#: aussi ailleurs : `MODE_EMPLOI_CAMPAGNE.md` est le mode d'emploi de la
#: campagne de ~224 h CPU, `BRIEF_REPRISE.md` la recette de reprise,
#: `README.md` la page d'accueil du depot. Mesure du 18 aout 2026 sur les
#: cinq : **25 commandes a options, aucune option non declaree**. Le trou
#: de perimetre etait donc reel et sans consequence vivante — il est ferme
#: pendant qu'il l'est.
_DOCS_A_COMMANDES = (
    "docs/RESULTS.md",
    "docs/DEFAUTS.md",
    "docs/BRIEF_REPRISE.md",
    "docs/MODE_EMPLOI_CAMPAGNE.md",
    "README.md",
)


@pytest.mark.parametrize("relpath", _DOCS_A_COMMANDES)
def test_every_repro_command_uses_options_its_script_declares(
        relpath, _declared_options):
    with open(os.path.join(_REPO_ROOT, relpath), encoding="utf-8") as f:
        commands = _commands_with_options(f.read())
    # Balayage vide : sans ce garde, un motif qui cesse de correspondre
    # rendrait ce test vert sans rien verifier. Mesure du 18 aout, par
    # document : RESULTS 16, DEFAUTS 2, BRIEF_REPRISE 1, MODE_EMPLOI 1,
    # README 5.
    plancher = {"docs/RESULTS.md": 12}.get(relpath, 1)
    assert len(commands) >= plancher, (
        f"le balayage n'a trouve que {len(commands)} commande(s) a options "
        f"dans {relpath} : c'est le motif qui a cesse de correspondre, pas "
        "le depot qui n'a plus de commandes")
    faulty = []
    for script, opts in commands:
        if not os.path.exists(os.path.join(_REPO_ROOT, script)):
            faulty.append(f"{script} : le fichier n'existe pas")
            continue
        declared = _declared_options(script)
        if declared is None:
            continue                      # `--help` ne rend pas 0 : hors portee
        missing = sorted(o for o in opts if o not in declared)
        if missing:
            faulty.append(f"{script} : {' '.join(missing)}")
    assert not faulty, (
        f"commandes de {relpath} citant une option que leur script ne "
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
    cassee. Les exceptions historiques sont **identifiees**, pas comptees.

    D-160 : compter les occurrences autorise n'importe laquelle. Chaque
    ligne qui cite `study/v4/` doit porter le fragment d'une exemption
    declaree pour ce fichier."""
    path = os.path.join(_REPO_ROOT, relpath)
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    fragments = [frag for fname, frag in _HISTORICAL_EXCEPTIONS
                 if fname == relpath]
    hits = [ln for ln in lines if "study/v4/" in ln]
    non_documentees = [ln.strip()[:120] for ln in hits
                       if not any(frag in ln for frag in fragments)]
    assert not non_documentees, (
        f"{relpath} cite 'study/v4/' — qui n'existe plus — sur une ligne "
        f"qu'aucune exemption ne designe : {non_documentees}. Si c'est de "
        "la narration historique, l'ajouter a `_HISTORICAL_EXCEPTIONS` avec "
        "un fragment de CETTE ligne ; sinon, c'est une commande morte")
    #  Une exemption qui ne correspond plus a rien pourrit : elle
    #  autoriserait demain une ligne qu'elle n'a jamais decrite.
    orphelines = [frag for frag in fragments
                  if not any(frag in ln for ln in hits)]
    assert not orphelines, (
        f"{relpath} : exemption(s) qui ne designent plus aucune ligne "
        f"'study/v4/' — les retirer : {orphelines}")


def test_un_bloc_cloture_n_est_pas_un_span_de_code_inline():
    """Epingle D-156 : sur quelle entree l'ancien parseur echouait-il ?

    Celle-ci — deux commandes dans un meme bloc ``` ```. L'ancien
    aplatissait le bloc, la premiere commande avalait les options de la
    seconde, et la seconde n'etait jamais vue."""
    doc = (
        "texte avant\n"
        "```bash\n"
        "python study/h3_representation/h3_size_scan.py --dims 2 --mapper v2\n"
        "python study/h4_transfer/h4_physics_robustness.py --fold rotor --recompute\n"
        "```\n"
        "texte apres, et un span inline coupe en deux :\n"
        "`python study/h0_selection/h0_qaoa_displacement.py --N 64\n"
        " --dim 2`\n"
    )
    vues = dict(_commands_with_options(doc))
    assert set(vues) == {
        "study/h3_representation/h3_size_scan.py",
        "study/h4_transfer/h4_physics_robustness.py",
        "study/h0_selection/h0_qaoa_displacement.py",
    }, f"les trois commandes doivent etre vues separement : {sorted(vues)}"
    assert vues["study/h3_representation/h3_size_scan.py"] == {"--dims", "--mapper"}, (
        "les options de la SECONDE commande du bloc sont attribuees a la "
        "premiere : c'est exactement D-156")
    assert vues["study/h4_transfer/h4_physics_robustness.py"] == {"--fold", "--recompute"}
    #  le span inline coupe en deux lignes se recolle toujours — c'est ce
    #  que l'aplatissement d'origine servait a faire, et il faut le garder
    assert vues["study/h0_selection/h0_qaoa_displacement.py"] == {"--N", "--dim"}

    #  L'ancien comportement, reproduit ici : s'il redevenait vrai, la
    #  premiere assertion tomberait. Ce bloc dit POURQUOI.
    ancien = {}
    flat = re.sub(r"`([^`]*)`",
                  lambda m: "`" + m.group(1).replace("\n", " ") + "`", doc)
    for m in _PY_CMD_RE.finditer(flat):
        opts = frozenset(_LONG_OPT_RE.findall(m.group(2)))
        if opts:
            ancien[m.group(1)] = opts
    assert "study/h4_transfer/h4_physics_robustness.py" not in ancien, (
        "l'ancien parseur voyait la seconde commande du bloc : D-156 "
        "n'aurait alors jamais existe — verifier ce test avant de le croire")
    assert ancien["study/h3_representation/h3_size_scan.py"] > {"--dims", "--mapper"}, (
        "l'ancien parseur fusionnait les options des deux commandes")


def test_le_commentaire_de_fin_de_ligne_n_est_pas_une_option():
    """`python x.py --dim 2   # --force-greedy a ete essaye` : le
    commentaire n'est pas une option citee. Sans quoi le nouveau parseur
    fabriquerait un faux rouge la ou l'ancien n'en avait pas."""
    doc = ("```bash\n"
           "python study/h3_representation/h3_size_scan.py --dims 2  "
           "# --option-inexistante, essayee puis abandonnee\n"
           "```\n")
    vues = dict(_commands_with_options(doc))
    assert vues == {"study/h3_representation/h3_size_scan.py": frozenset({"--dims"})}, vues


def test_une_commande_citee_en_prose_est_vue_et_une_citation_narrative_ne_l_est_pas():
    """Epingle D-160, les deux sens.

    Sur quelle entree l'ancien motif echouait-il ? Sur la forme la plus
    naturelle de citer une commande — au milieu d'une phrase, prefixee de
    `python`. Et il ne doit toujours PAS attraper une citation narrative,
    sans quoi on fabrique un faux rouge."""
    doc = ("reproduire : `python study/h3_representation/h3_size_scan.py --dims 2`\n"
           "le defaut vit dans `study/pipeline/dns_validation.py:95`, ligne 95\n"
           "`study/common/metrics.py --json`\n")
    vus = _paths_referenced(doc)
    assert "study/h3_representation/h3_size_scan.py" in vus, (
        "une commande citee au fil du texte doit etre vue : c'est D-160")
    assert "study/common/metrics.py" in vus, (
        "la forme historique — span seul sur sa ligne, chemin en tete — "
        "doit continuer d'etre vue")
    assert "study/pipeline/dns_validation.py" not in vus, (
        "une citation narrative avec un numero de ligne n'est pas une "
        "commande : l'attraper fabriquerait un faux rouge")

    #  L'ancien motif, rejoue : il ne voyait pas la premiere.
    ancien = [m.group(1) for ligne in doc.splitlines()
              for m in [_INLINE_CMD_RE.match(ligne.strip())] if m]
    assert "study/h3_representation/h3_size_scan.py" not in ancien, (
        "l'ancien motif voyait deja la commande en prose : D-160 n'aurait "
        "alors jamais existe — verifier ce test avant de le croire")


def test_l_exemption_historique_designe_une_ligne_et_non_un_compte(tmp_path):
    """Epingle D-160 : compter les lignes autorise n'importe laquelle.

    Champ qui SEPARE : deux documents portant le MEME nombre d'occurrences
    du prefixe mort, l'un narratif, l'autre une commande a rejouer. Un
    critere par compte les declare identiques."""
    fragments = ["| D-71 |"]
    narratif = ["| D-71 | la reorganisation a deplace study/v4/tNN_xxx.py\n"]
    commande = ["reproduire : `python study/v4/t31_axis.py --dim 16`\n"]

    def non_documentees(lignes):
        hits = [ln for ln in lignes if "study/v4/" in ln]
        return [ln for ln in hits
                if not any(f in ln for f in fragments)]

    assert non_documentees(narratif) == []
    assert non_documentees(commande), (
        "une commande vivante citant le prefixe mort doit etre signalee, "
        "meme quand le NOMBRE d'occurrences n'a pas bouge — c'est la forme "
        "de D-136, une chaine presente deux fois dont une seule suffit")
    #  le critere par compte, rejoue : il ne separe pas
    assert len([ln for ln in narratif if "study/v4/" in ln]) \
        == len([ln for ln in commande if "study/v4/" in ln]), (
        "les deux documents doivent porter le meme compte, sinon ce test "
        "ne montre pas ce qu'il annonce")
