"""Chemins d'import, une fois pour toutes.

Les fichiers de test etaient repartis a plat et calculaient chacun leur
chemin vers `src/` par un nombre de `dirname` egal a leur profondeur. Tout
deplacement cassait donc silencieusement les imports — ou, pire, faisait
lire un fichier a la mauvaise place.

Ce conftest resout les chemins depuis la racine du depot, quelle que soit la
profondeur du fichier de test. Il rend aussi les constantes disponibles :

    from conftest import REPO_ROOT, SRC
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")

#: Les paquets de `study/` que les tests importent par leur nom de module.
_STUDY_PKGS = (
    "study/pipeline", "study/common", "study/h0_selection",
    "study/h1_solver", "study/h2b_prediction", "study/h3_representation",
    "study/h4_transfer", "study/closed_loop",
)

for _p in (SRC, REPO_ROOT, *(os.path.join(REPO_ROOT, *d.split("/"))
                             for d in _STUDY_PKGS)):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
