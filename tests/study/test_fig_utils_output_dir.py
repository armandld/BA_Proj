"""D-93 — `figures/v1_legacy/fig_utils.py` : ou les 17 scripts de figures V1 ecrivent.

L'ancre `_PROJECT_ROOT` ne montait que d'un niveau. Juste tant que le fichier
vivait dans `figures_code/` a la racine ; faux depuis que la reorganisation
`17d983d` l'a descendu dans `figures/v1_legacy/` en ne reecrivant que l'autre
ancre du meme fichier (`_REPO_ROOT`, deux niveaux). FIG_DIR valait alors
`figures/figures/`, cree en silence a l'import, dans l'arborescence du code.

Ces tests echouent tous sur la version d'avant la correction.

Le champ qui SEPARE : la profondeur de l'ancre. Un test qui verifierait
seulement « FIG_DIR finit par figures » passe sur les deux versions —
`figures/figures` finit par `figures`. Ce qui separe les deux hypotheses,
c'est la position ABSOLUE du dossier, et le fait qu'il contienne (ou non)
les figures que le depot publie deja.
"""
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_V1_LEGACY = os.path.join(_REPO_ROOT, "figures", "v1_legacy")

# Une figure reellement committee dans le depot. Sert d'ancre de comportement :
# le dossier de sortie des figures est celui qui porte les figures publiees,
# pas celui dont le nom ressemble.
_PUBLISHED_FIGURE = "fig1_ceiling_bar.png"


def _fig_dir(env_extra=None):
    """Rend FIG_DIR (chemin absolu) tel que fig_utils le calcule a l'import.

    `os.makedirs` est neutralise dans le sous-processus : mesurer ne doit rien
    creer dans le depot — ni le bon dossier, ni le mauvais. Un sous-processus
    est necessaire parce que FIG_DIR est fige a l'import et depend de
    l'environnement (FIGURE_PHASE).
    """
    code = (
        "import os, sys\n"
        "os.makedirs = lambda *a, **k: None\n"
        "sys.path.insert(0, %r)\n"
        "import fig_utils as F\n"
        "print(os.path.abspath(F.FIG_DIR))\n" % _V1_LEGACY
    )
    env = dict(os.environ)
    env.pop("FIGURE_PHASE", None)
    if env_extra:
        env.update(env_extra)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env,
        cwd=_REPO_ROOT,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout.strip().splitlines()[-1]


@pytest.fixture(scope="module")
def fig_dir_default():
    return _fig_dir()


def test_fig_dir_est_le_dossier_de_sortie_du_depot(fig_dir_default):
    """FIG_DIR == <racine>/results/figures.

    Avant D-93 : <racine>/figures/figures.
    """
    attendu = os.path.join(_REPO_ROOT, "results", "figures")
    assert fig_dir_default == attendu


def test_fig_dir_porte_les_figures_deja_publiees(fig_dir_default):
    """L'assertion de comportement : le dossier de sortie est celui ou vivent
    les figures que le depot publie deja (`figures/result_figs.py` y ecrit).

    Deux chemins censes coincider — question 4 de VIGIL.md : `result_figs.py`
    (reste dans `figures/`, ancre a un niveau, JUSTE) et `fig_utils.py`
    (descendu dans `figures/v1_legacy/`, ancre a un niveau, FAUX) doivent
    designer le meme dossier. Avant D-93 ils en designaient deux.
    """
    assert os.path.isfile(os.path.join(fig_dir_default, _PUBLISHED_FIGURE)), (
        "FIG_DIR ne porte pas %s : les figures V1 n'atterrissent pas la ou le "
        "depot publie les siennes (FIG_DIR = %s)" % (_PUBLISHED_FIGURE, fig_dir_default)
    )


def test_d93_epingle_lancien_comportement(fig_dir_default):
    """Epingle la valeur FAUSSE, pour que la correction ne se defasse pas en silence.

    L'ancienne valeur mesuree etait exactement <racine>/figures/figures.
    """
    ancienne = os.path.join(_REPO_ROOT, "figures", "figures")
    assert fig_dir_default != ancienne
    # ... et, plus generalement, aucune sortie ne doit atterrir dans
    # l'arborescence du CODE des figures.
    source_tree = os.path.join(_REPO_ROOT, "figures") + os.sep
    assert not fig_dir_default.startswith(source_tree), (
        "FIG_DIR ecrit dans l'arborescence du code des figures : %s" % fig_dir_default
    )


def test_fig_dir_suit_figure_phase():
    """L'axe FIGURE_PHASE, parcouru des DEUX cotes (absent ci-dessus, present ici).

    Avant D-93 : <racine>/figures/figures/phase1.
    """
    mesure = _fig_dir({"FIGURE_PHASE": "1"})
    attendu = os.path.join(_REPO_ROOT, "results", "figures", "phase1")
    assert mesure == attendu
    assert mesure != os.path.join(_REPO_ROOT, "figures", "figures", "phase1")


def test_les_deux_ancres_du_fichier_coincident():
    """`_REPO_ROOT` et `_PROJECT_ROOT` designent la meme racine.

    C'est la cause du defaut : deux ancres pour la meme chose dans un seul
    fichier, dont une seule a suivi le deplacement. Interroge le module, pas
    le texte du source.
    """
    code = (
        "import os, sys\n"
        "os.makedirs = lambda *a, **k: None\n"
        "sys.path.insert(0, %r)\n"
        "import fig_utils as F\n"
        "print(os.path.abspath(F._REPO_ROOT))\n"
        "print(os.path.abspath(F._PROJECT_ROOT))\n" % _V1_LEGACY
    )
    env = dict(os.environ)
    env.pop("FIGURE_PHASE", None)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, cwd=_REPO_ROOT)
    assert out.returncode == 0, out.stderr[-3000:]
    repo_root, project_root = out.stdout.strip().splitlines()[-2:]
    assert repo_root == project_root == _REPO_ROOT


def test_importer_fig_utils_ne_cree_pas_de_dossier_parasite(tmp_path):
    """`os.makedirs` a l'import ne doit creer que le dossier de sortie legitime.

    Le defaut etait invisible precisement parce que l'import creait sa cible
    sans rien dire. On capture les appels plutot que de les laisser agir.
    """
    code = (
        "import os, sys\n"
        "faits = []\n"
        "os.makedirs = lambda p, **k: faits.append(os.path.abspath(p))\n"
        "sys.path.insert(0, %r)\n"
        "import fig_utils\n"
        "print('\\n'.join(faits))\n" % _V1_LEGACY
    )
    env = dict(os.environ)
    env.pop("FIGURE_PHASE", None)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, cwd=str(tmp_path))
    assert out.returncode == 0, out.stderr[-3000:]
    faits = [l for l in out.stdout.strip().splitlines() if l.startswith(os.sep)]
    assert faits, "l'import ne cree plus rien : ce test ne mesure plus rien"
    source_tree = os.path.join(_REPO_ROOT, "figures") + os.sep
    for d in faits:
        assert not d.startswith(source_tree), (
            "l'import de fig_utils cree %s dans l'arborescence du code" % d
        )
