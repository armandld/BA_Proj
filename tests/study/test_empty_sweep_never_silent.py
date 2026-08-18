"""D-56 : huit modules de `study/` sortaient avec le code 0 sur un balayage
vide, sans ecrire d'artefact — donc en laissant en place celui de la
campagne precedente.

`CLAUDE.md` : « Un test qui ne peut pas echouer est un defaut. Tout script de
`study/` ou de `tests/` porte une assertion, et un balayage vide doit
crier. » Onze modules levaient deja ; huit imprimaient `no input.` et
rendaient la main. D-55 en avait corrige un neuvieme
(`h3_term_ablation.py`) ; ceci ferme la famille.

Mesure avant / apres, meme commande
(`--scenario no_such_scenario --N 64`) :

    h3_locality_proposition.py   code 0 -> 1
    h3_equivariance.py           code 0 -> 1
    h2b_learned_meanfield_h.py   code 0 -> 1
"""
import ast
import glob
import os
import re
import subprocess
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

#: Noms d'accumulateurs derriere lesquels un `if not <x>:` garde la fin
#: d'un balayage. Tenu a la main : c'est la liste des formes reellement
#: rencontrees dans `study/`.
ACCUMULATORS = {"rows", "records", "results", "configs", "by_scene",
                "per_cfg", "out_rows", "all_rows", "entries"}


def _silent_empty_sweeps(path):
    """Gardes `if not <accumulateur>:` dont le corps ne fait que rendre la
    main. Interroge l'AST, pas le texte du source : un test qui cherche la
    chaine « no input. » casserait sur une reformulation sans qu'aucun
    defaut n'existe."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If)
                and isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and getattr(node.test.operand, "id", None) in ACCUMULATORS):
            continue
        raises = any(isinstance(n, ast.Raise) for n in ast.walk(node))
        returns = any(isinstance(n, ast.Return) for n in ast.walk(node))
        if returns and not raises:
            out.append((node.test.operand.id, node.lineno))
    return out


STUDY_FILES = sorted(glob.glob(os.path.join(_REPO_ROOT, "study", "**", "*.py"),
                               recursive=True))


@pytest.mark.parametrize("path", STUDY_FILES,
                         ids=[os.path.relpath(p, _REPO_ROOT) for p in STUDY_FILES])
def test_no_study_module_returns_silently_on_an_empty_sweep(path):
    silent = _silent_empty_sweeps(path)
    assert not silent, (
        f"{os.path.relpath(path, _REPO_ROOT)} rend la main sans lever sur un "
        f"balayage vide (accumulateur/ligne : {silent}). Le script sortirait "
        "avec le code 0 sans ecrire d'artefact, en laissant celui de la "
        "campagne precedente en place. Voir D-55 / D-56.")


def test_the_guard_actually_bites_on_a_real_module():
    """L'AST dit ce que le code CONTIENT ; ceci verifie ce qu'il FAIT.
    `h3_locality_proposition` rendait 0 sur ce meme appel."""
    r = subprocess.run(
        [sys.executable,
         os.path.join(_REPO_ROOT, "study", "h3_representation",
                      "h3_locality_proposition.py"),
         "--scenario", "no_such_scenario", "--N", "64"],
        capture_output=True, text=True, cwd=_REPO_ROOT, timeout=600)
    assert r.returncode != 0, (
        "le balayage vide est redevenu silencieux : code de sortie 0")
    assert "balayage vide" in (r.stderr + r.stdout)


# ══════════════════════════════════════════════════════════════════
#  D-148 — le meme garde, par le COMPORTEMENT
# ══════════════════════════════════════════════════════════════════
#
# Le detecteur AST ci-dessus cherche UNE forme (`if not <nom>: ... return`)
# avec une liste de noms tenue a la main. Balayage generalise a tout nom et
# a toute forme : 30 sites de `study/` repondent, AUCUN n'est dans
# `ACCUMULATORS` — le detecteur en voyait zero. Et six modules sortaient
# reellement avec le code 0 sans rien ecrire, dont les phases 2, 3, 4, 5, 7
# et 8 du pipeline, alors que l'en-tete de ce fichier annonce « ceci ferme
# la famille ».
#
# La lecon est celle de D-56 elle-meme, d'un cran plus haut : D-56 avait
# trouve trois modules par l'AST que la recherche de la chaine « no input. »
# aurait manques, parce que leur MESSAGE differait. Ici c'est la FORME qui
# differe — `if <accumulateur>:` autour du resume, puis on tombe en bas de
# `main()`. Aucune recherche de forme ne ferme une famille ; seul le
# comportement le fait.
#
# Ce test ne cherche donc aucune forme : il execute chaque module sur une
# demande qui ne correspond a rien et exige un code de sortie non nul.
# ~2 min pour 61 modules.

#: Modules exemptes, avec leur raison. Une exemption sans raison ecrite se
#: fait etendre par erreur.
_EXEMPTIONS = {
    "study/common/aggregate_master_table.py": (
        "D-158 — RAPPORT SEUL, en attente de decision. Sur une "
        "configuration qui ne correspond a rien (`--N 7 --dim 99`) il sort "
        "avec le code 0 ET REECRIT les artefacts publies : 180 -> 161 "
        "lignes, OK 176 -> 113, MISSING 0 -> 48, 136 lignes supprimees de "
        "`results/v4_master_table.csv`. Le lancer ici detruirait la table de "
        "non-regression du depot ; c'est pourquoi il n'est pas lance."),
    "study/common/aggregate_v3.py": (
        "D-158, meme forme : `--N 7 --dim 99` sort 0 et ecrit "
        "`results/v3_master_table.csv/.md` et `v3_master_N7.npz`. Non lance "
        "ici pour la meme raison — la mesure est dans `RESULTS.md`."),
    "study/common/aggregate_v2.py": (
        "D-158, meme forme, en moins grave : `--N 7 --dim 99` sort 0 mais "
        "ecrit `SUMMARY_N7_dim99.csv` — le nom porte la configuration, donc "
        "il n'ecrase rien de publie. Non lance ici par symetrie avec ses "
        "deux jumeaux, et parce que le verdict lui revient aussi."),
    "study/closed_loop/closed_loop_status.py": (
        "Sort 0 sur `--folds no_such_fold` en imprimant "
        "`no_such_fold [---] no-trials-yet`. C'est un rapporteur d'ETAT : "
        "« rien n'a tourne » est un etat valide, pas un balayage vide "
        "silencieux. Ecrit ici pour que la decision ne se reprenne pas "
        "chaque passe — voir D-158."),
    "study/common/preflight_coefficients.py": (
        "Aucun selecteur de donnees dans sa CLI (`--json` seulement) : on ne "
        "peut pas lui demander quelque chose qui ne correspond a rien. Le "
        "lancer sans argument lancerait sa vraie mesure, ce qui n'est pas ce "
        "que ce balayage teste. Non couvert, et dit."),
    "study/common/rho_gap_f1.py": (
        "Meme raison que `preflight_coefficients.py` : `--json` est sa seule "
        "option, aucune demande ne peut etre rendue vide. Non couvert, et "
        "dit plutot que devine."),
    "study/pipeline/dns_validation.py": (
        "GELE — ses artefacts sont publies, et une correction y a deja ete "
        "annulee. Mesure D-148 : il sort bien avec le code 0 sur "
        "`--scenario no_such_scenario`, MAIS il ECRIT "
        "`results/dns_validation_N64.npz` — sa sortie n'est donc pas "
        "indiscernable d'une campagne reussie faute d'artefact, elle EST "
        "une campagne qui a produit quelque chose. Non corrige, "
        "volontairement."),
}


def _modules_lancables():
    out = []
    for p in STUDY_FILES:
        src = open(p, encoding="utf-8").read()
        if "__main__" in src and "argparse" in src:
            out.append(p)
    return out


_LANCABLES = _modules_lancables()


def test_le_balayage_comportemental_nest_pas_vide():
    """Un balayage vide doit crier — y compris celui-ci."""
    assert len(_LANCABLES) > 50, (
        f"{len(_LANCABLES)} modules lancables collectes : le balayage de "
        "D-148 est vide ou tronque, et les tests ci-dessous ne prouveraient "
        "rien")


def test_chaque_exemption_de_D148_porte_sa_raison_et_existe_encore():
    """D-163 : une exemption doit encore SUPPRIMER quelque chose.

    Ce controle ne verifiait que l'existence du fichier. Or ce que fait une
    entree de `_EXEMPTIONS`, c'est retirer un module du balayage
    comportemental — et elle ne retire quelque chose que si le module y
    serait, c'est-a-dire s'il est encore dans `_LANCABLES`. Un module qui
    perd son `argparse` ou son bloc `__main__` sort du balayage tout seul :
    son exemption devient alors une permission dormante, prete a dispenser
    le jour ou il y reviendra. Meme forme que D-161.
    """
    lancables = {os.path.relpath(p, _REPO_ROOT) for p in _LANCABLES}
    for rel, raison in _EXEMPTIONS.items():
        assert os.path.exists(os.path.join(_REPO_ROOT, rel)), (
            f"{rel} est exempte mais n'existe plus : retirer l'exemption")
        assert len(raison) > 80, f"{rel} : exemption sans raison ecrite"
        assert rel in lancables, (
            f"{rel} est exempte du balayage comportemental, mais il n'y "
            "entrerait plus de toute facon (il a quitte `_LANCABLES`). "
            "L'exemption ne supprime plus rien et dispenserait sans le dire "
            "le jour ou le module y reviendrait : la retirer (D-163).")


def test_le_controle_de_peremption_des_exemptions_peut_echouer():
    """Un balayage vide doit crier — y compris ce controle-ci.

    Epingle D-163 : sur quelle entree l'ancien controle echouait-il ? Sur
    aucune — `os.path.exists` reste vrai pour un module qui a quitte
    `_LANCABLES`. Le critere qui mord est l'appartenance au balayage, et il
    doit pouvoir rendre faux.
    """
    lancables = {os.path.relpath(p, _REPO_ROOT) for p in _LANCABLES}
    #  un fichier de `study/` qui existe mais n'est pas lancable : le
    #  critere doit le refuser. S'il n'en existe aucun, le dire plutot que
    #  de conclure.
    non_lancables = [os.path.relpath(p, _REPO_ROOT) for p in STUDY_FILES
                     if os.path.relpath(p, _REPO_ROOT) not in lancables]
    assert non_lancables, (
        "tous les fichiers de study/ sont lancables : ce controle ne peut "
        "plus distinguer « exemption portante » de « exemption morte »")
    for rel in non_lancables:
        assert os.path.exists(os.path.join(_REPO_ROOT, rel))   # ancien critere : vert
        assert rel not in lancables                            # nouveau : rouge


# ══════════════════════════════════════════════════════════════════
#  D-157 — l'invocation elle-meme n'etait pas verifiee
# ══════════════════════════════════════════════════════════════════
#
# Le balayage ci-dessus envoyait `--scenario no_such_scenario --N 64` aux
# 60 modules, quels que soient les arguments qu'ils DECLARENT. Mesure du
# 18 aout 2026 : **21 des 60 mouraient dans argparse** — `unrecognized
# arguments` ou `the following arguments are required`, code de sortie 2 —
# sans executer une seule ligne de leur garde. Le test n'exigeait qu'un
# code non nul : il passait, sur un tiers du perimetre, pour une raison
# qui n'a rien a voir avec un balayage vide.
#
# C'est le piege du balayage vide, dans le fichier ecrit pour le fermer —
# et c'est la lecon de D-140 (« le chemin existe, l'option non ») retournee
# contre la suite : une commande qu'un script refuse ne mesure rien.
#
# Corrige en deux points :
#   - l'invocation est construite a partir des options que le module
#     DECLARE, lues a son propre `--help` — l'operateur assorti ;
#   - le test refuse desormais une mort dans argparse. Un code non nul ne
#     suffit plus : il doit venir du garde, pas du parseur.

_ERREURS_D_USAGE = ("unrecognized arguments",
                    "the following arguments are required",
                    "invalid choice")

_OPTIONS_CACHE = {}


def _options_declarees(path):
    """Options longues que le module declare, lues a son `--help`."""
    if path not in _OPTIONS_CACHE:
        r = subprocess.run([sys.executable, path, "--help"],
                           capture_output=True, text=True,
                           cwd=_REPO_ROOT, timeout=300)
        _OPTIONS_CACHE[path] = (set(re.findall(r"(?<![\w-])(--[a-zA-Z][a-zA-Z0-9-]*)",
                                               r.stdout + r.stderr))
                                if r.returncode == 0 else None)
    return _OPTIONS_CACHE[path]


def invocation_sans_correspondance(declarees):
    """Une demande que le module ACCEPTE et a laquelle rien ne repond.

    Ordre des selecteurs : du plus specifique au plus large. `None` quand
    le module n'expose aucun selecteur de donnees — c'est une absence de
    couverture, elle se declare (voir `_EXEMPTIONS`), elle ne se devine
    pas."""
    if declarees is None:
        return None
    for opt, valeur in (("--scenario", "no_such_scenario"),
                        ("--scenarios", "no_such_scenario"),
                        ("--fold", "no_such_fold"),
                        ("--folds", "no_such_fold")):
        if opt in declarees:
            args = [opt, valeur]
            if "--N" in declarees:
                args += ["--N", "64"]
            return args
    if "--N" in declarees and "--dim" in declarees:
        #  aucune campagne n'a jamais tourne a cette taille
        return ["--N", "7", "--dim", "99"]
    return None


@pytest.mark.parametrize(
    "path", _LANCABLES,
    ids=[os.path.relpath(p, _REPO_ROOT) for p in _LANCABLES])
def test_aucun_module_de_study_ne_sort_zero_sur_un_balayage_vide(path):
    """Une demande a laquelle rien ne repond : un module qui rend 0
    la-dessus est indiscernable d'une campagne reussie."""
    rel = os.path.relpath(path, _REPO_ROOT)
    if rel in _EXEMPTIONS:
        pytest.skip(_EXEMPTIONS[rel])
    args = invocation_sans_correspondance(_options_declarees(path))
    assert args is not None, (
        f"{rel} n'expose aucun selecteur de donnees : aucune demande ne "
        "peut lui etre faite qui ne corresponde a rien. Ce module n'est pas "
        "couvert par ce balayage — l'ecrire dans `_EXEMPTIONS` avec sa "
        "raison, ne pas le laisser passer en silence")
    r = subprocess.run([sys.executable, path] + args,
                       capture_output=True, text=True,
                       cwd=_REPO_ROOT, timeout=300)
    sortie = r.stdout + r.stderr
    usage = [u for u in _ERREURS_D_USAGE if u in sortie]
    assert not usage, (
        f"{rel} refuse l'invocation `{' '.join(args)}` dans son parseur "
        f"({usage[0]}) : le module n'a execute aucune ligne, et son code de "
        "sortie 2 ne dit RIEN de son garde de balayage vide. C'est D-157 — "
        "un test qui passe pour la mauvaise raison. Dernieres lignes :\n"
        + "\n".join(sortie.strip().split("\n")[-3:]))
    assert r.returncode != 0, (
        f"{rel} sort avec le code 0 sur un balayage qui ne correspond a "
        f"rien, sans ecrire d'artefact : il laisse en place ceux de la "
        f"campagne precedente et ne se distingue pas d'une campagne "
        f"reussie. Dernieres lignes :\n"
        + "\n".join(sortie.strip().split("\n")[-5:]))


def test_the_detector_itself_can_fail():
    """Un balayage vide doit crier — y compris celui-ci. Si `_silent_empty_sweeps`
    ne detectait plus rien, les tests ci-dessus passeraient sans mesurer quoi
    que ce soit (le piege du balayage vide, dans le fichier cense le
    detecter)."""
    assert len(STUDY_FILES) > 40, (
        f"seulement {len(STUDY_FILES)} modules de study/ collectes : le "
        "balayage du detecteur est vide ou tronque")

    import tempfile
    src = "def main():\n    rows = []\n    if not rows:\n        print('x')\n        return\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        tmp = fh.name
    try:
        assert _silent_empty_sweeps(tmp) == [("rows", 3)]
    finally:
        os.unlink(tmp)


def test_le_selecteur_choisit_une_option_que_le_module_declare():
    """Epingle D-157 : sur quelle entree l'ancienne invocation echouait-elle ?

    Sur tout module qui ne declare pas `--scenario`. Le selecteur ne doit
    proposer QUE des options declarees — sinon on remesure argparse."""
    #  un module qui ne connaît que --fold : ni --scenario, ni --N
    assert invocation_sans_correspondance({"--fold", "--threshold", "--help"}) \
        == ["--fold", "no_such_fold"]
    #  --N n'est ajoute que s'il est declare
    assert invocation_sans_correspondance({"--scenario", "--seed"}) \
        == ["--scenario", "no_such_scenario"]
    assert invocation_sans_correspondance({"--scenario", "--N"}) \
        == ["--scenario", "no_such_scenario", "--N", "64"]
    #  le repli par la taille, pour les agregateurs
    assert invocation_sans_correspondance({"--N", "--dim", "--strict"}) \
        == ["--N", "7", "--dim", "99"]
    #  aucun selecteur : on ne devine pas, on rend None et l'appelant crie
    assert invocation_sans_correspondance({"--json", "--help"}) is None
    assert invocation_sans_correspondance(None) is None

    #  L'ancienne invocation, rejouee : elle envoie --N a un module qui ne
    #  le declare pas. C'est la mesure de D-157, gardee ici pour que la
    #  regression se voie.
    ancienne = ["--scenario", "no_such_scenario", "--N", "64"]
    declarees = {"--scenario", "--grids", "--seed"}     # h1_solver_convergence
    assert any(a.startswith("--") and a not in declarees for a in ancienne), (
        "l'ancienne invocation passait --N a des modules qui ne le "
        "declarent pas : c'est ce qui les faisait mourir dans argparse")


def test_le_balayage_comportemental_couvre_ce_qu_il_annonce():
    """Combien de modules recoivent une invocation qu'ils ACCEPTENT ?

    Mesure du 18 aout 2026 : 61 modules lancables, 6 exemptes avec leur
    raison, **55 couverts**. Avant D-157, 21 des 60 lances mouraient dans
    argparse — le test passait sans qu'ils executent une ligne."""
    couverts, sans_selecteur = [], []
    for p in _LANCABLES:
        rel = os.path.relpath(p, _REPO_ROOT)
        if rel in _EXEMPTIONS:
            continue
        if invocation_sans_correspondance(_options_declarees(p)) is None:
            sans_selecteur.append(rel)
        else:
            couverts.append(rel)
    assert not sans_selecteur, (
        "modules sans selecteur de donnees et sans exemption ecrite : "
        f"{sans_selecteur}")
    assert len(couverts) >= 50, (
        f"{len(couverts)} modules couverts — mesure du 18 aout : 55. "
        "Le balayage s'est vide sans que personne ne le dise")
