"""D-96 — regression de D-37 dans `instrumented_bfs`/`instrumented_bfs_hamilt`.

`figures/v1_legacy/fig15_decision_flip_analysis.py`,
`fig16_decision_landscape.py` et `fig17_topological_attribution.py` portent
chacun leur propre reimplementation de la traversee BFS de l'AMR, plutot que
d'appeler celle de `src/Simulation/refinement.py`. Les trois copiaient le
meme appel fautif :

    score_map_padded = _process_score(
        local_score_raw, is_periodic,
        target_dim + 2 * pad if pad > 0 else target_dim)

`_process_score(arr, False, t_dim)` emprunte `_resize_padded_maxpool`, dont
le contrat est « entree (N+2, M+2) -> sortie (t_dim+2, t_dim+2) » : le halo
est deja ajoute par la fonction. Demander `target_dim + 2*pad` (donc
`t_dim=4` pour `target_dim=2`) fait rendre un coeur (6, 6) plutot que (4, 4).
Le code trimme ensuite `[1:-1, 1:-1]` en pensant obtenir un coeur
`target_dim x target_dim`, mais recupere un `(4, 4)` — dont la boucle
`for i in range(target_dim)` ne lit que le quart HAUT-GAUCHE : `score_map`
ne decrit alors qu'un sous-coin du patch, pas les 4 quadrants que `sub_bounds`
et `qaoa_prob` (qui LUI est correctement dimensionne en `target_dim`) disent
couvrir. `classical_score[i,j]` et `qaoa_prob[i,j]` finissent par decrire
deux regions differentes du meme patch a `depth > 0`.

C'est exactement D-37 (voir `tests/amr/test_patch_encoding_shapes.py`,
`refinement.py:180-193`), deja mesure et corrige dans le chemin canonique
(`_run_level`/`_run_level_classical`), jamais applique a ces trois copies.

Mesure sur `init_harris_tearing`, N=256, 30 pas, patch `depth=1`
(bounds=(0,128,0,128), pad=1) : ecart max **0.525** entre l'ancien et le
correct sur des scores dont l'echelle max vaut **0.656** (80 %). Avec
`threshold_amr=0.3228` (`load_hyperparams(method='classical')`), 2 des 4
decisions binaires `score >= threshold_amr` basculaient
(cellules (0,0) et (1,0) : 0.062/0.047 -> pas de raffinement, contre
0.541/0.572 -> raffinement) — exactement le type de comptage que
`fig15_decision_flip_analysis.py` existe pour produire.

Ces tests n'importent pas les modules `fig15`/`fig16`/`fig17` eux-memes :
les trois executent leur campagne complete (plusieurs scenarios, QAOA) a
l'IMPORT, sans garde `if __name__ == "__main__"` — les importer ferait
tourner l'analyse entiere. Ils epinglent donc l'appel exact que les trois
fichiers font au meme trio de fonctions (`get_periodic_patch`,
`_process_score`), sur les memes donnees, comme `tests/amr/test_patch_encoding_shapes.py`
le fait deja pour D-37 dans `refinement.py`.
"""
import os
import re
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from Simulation.PhysToAngle import AngleMapper
from Simulation.RescaleArrays import _process_score
from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.utils import get_periodic_patch

TARGET_DIM = 2
_V1_LEGACY = os.path.join(_REPO_ROOT, "figures", "v1_legacy")
_FIXED_FILES = [
    "fig15_decision_flip_analysis.py",
    "fig16_decision_landscape.py",
    "fig17_topological_attribution.py",
]


@pytest.fixture(scope="module")
def flow():
    """Harris tearing developpe : les deux quadrants du patch depth=1 ont
    des scores nettement differents, le champ qui SEPARE ancien et correct."""
    grid = PeriodicGrid(resolution_N=256)
    sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
    sim.init_harris_tearing()
    for _ in range(30):
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)
    return sim


def _score_map(sim, bounds, pad, t_dim_requested):
    y_s, y_e, x_s, x_e = bounds
    full_score = AngleMapper.classical_score(sim.get_fluxes())
    local_score_raw = get_periodic_patch(full_score, y_s, y_e, x_s, x_e, pad=pad)
    padded = _process_score(local_score_raw, pad == 0, t_dim_requested)
    return padded[1:-1, 1:-1] if pad > 0 else padded


# ══════════════════════════════════════════════════════════════════
#  1. Le code source ne porte plus l'appel fautif
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("fname", _FIXED_FILES)
def test_source_no_longer_asks_for_the_halo_twice(fname):
    src = open(os.path.join(_V1_LEGACY, fname), encoding="utf-8").read()
    assert "target_dim + 2 * pad" not in src, (
        f"{fname} redemande encore le halo que _process_score ajoute deja")


# ══════════════════════════════════════════════════════════════════
#  2. La mesure du defaut, figee (epingle l'ancien comportement)
# ══════════════════════════════════════════════════════════════════

def test_the_old_call_produced_a_wrong_shaped_core(flow):
    """Ancien appel : `_process_score(local, False, target_dim + 2*pad)`
    rend un coeur (4, 4), pas le (2, 2) que `for i in range(target_dim)`
    suppose lire."""
    bounds, pad = (0, 128, 0, 128), 1
    ancien = _score_map(flow, bounds, pad, TARGET_DIM + 2 * pad)
    assert ancien.shape == (4, 4)
    assert ancien.shape != (TARGET_DIM, TARGET_DIM)


def test_the_fixed_call_produces_the_canonical_two_by_two_grid(flow):
    """Nouvel appel (celui que fig15/16/17 font desormais) : (2, 2), et
    identique a ce que `_run_level_classical` (chemin canonique, deja
    corrige par D-37) calculerait sur le meme patch."""
    bounds, pad = (0, 128, 0, 128), 1
    correct = _score_map(flow, bounds, pad, TARGET_DIM)
    assert correct.shape == (TARGET_DIM, TARGET_DIM)


def test_the_wrong_grid_changed_classical_score_by_a_large_fraction(flow):
    """Mesure : jusqu'a 0.525 d'ecart, echelle max 0.656 (80 %). Le seuil
    est mis a 20 % du plus grand coefficient, comme D-37 (test_patch_
    encoding_shapes.py) : si le defaut redevient benin, remesurer plutot
    que d'abaisser le seuil."""
    bounds, pad = (0, 128, 0, 128), 1
    ancien_4x4 = _score_map(flow, bounds, pad, TARGET_DIM + 2 * pad)
    ancien = ancien_4x4[:TARGET_DIM, :TARGET_DIM]  # ce que la boucle lisait
    correct = _score_map(flow, bounds, pad, TARGET_DIM)

    ecart = float(np.max(np.abs(ancien - correct)))
    echelle = float(np.max(np.abs(correct)))
    assert ecart > 0.2 * echelle, (
        f"ecart {ecart:.5f} pour une echelle {echelle:.5f} : si le defaut "
        "est devenu benin, remesurer plutot que d'abaisser le seuil")


def test_the_wrong_grid_flipped_refine_decisions(flow):
    """Consequence directe pour fig15 : la decision binaire
    `classical_score >= threshold_amr` bascule sur au moins une cellule
    entre l'ancien et le correct calcul, sur les memes donnees."""
    bounds, pad = (0, 128, 0, 128), 1
    threshold_amr = 0.3228115420561065  # load_hyperparams(method='classical')

    ancien_4x4 = _score_map(flow, bounds, pad, TARGET_DIM + 2 * pad)
    ancien = ancien_4x4[:TARGET_DIM, :TARGET_DIM]
    correct = _score_map(flow, bounds, pad, TARGET_DIM)

    dec_ancien = ancien >= threshold_amr
    dec_correct = correct >= threshold_amr
    n_flipped = int(np.sum(dec_ancien != dec_correct))
    assert n_flipped >= 1, (
        "le defaut ne changeait aucune decision de raffinement sur ce "
        "patch : remesurer sur un autre patch/scenario avant de conclure "
        "qu'il est benin")
