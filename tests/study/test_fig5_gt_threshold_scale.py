"""D-101 — `fig5_qaoa_detailed_analysis.py::_gt_quadrant_above_threshold`
comparait une grandeur brute a un seuil calibre pour une autre.

`gt` vient de `ground_truth_errors()` (`fig_utils.py`) : magnitude de
gradient + laplacien sur les 4 champs MHD, **non normalisee**, dont
l'echelle depend du scenario (le fichier imprime lui-meme `gt.min()`/
`gt.max()` par scenario, preuve qu'elle n'est bornee a rien de fixe).

`threshold_amr` (~0,30) est calibre exclusivement contre
`AngleMapper.classical_score`, normalise au max du domaine et clippe dans
[0,1] (`PhysToAngle.py`, docstring : « Each indicator is normalized to
[0,1] by its domain-wide max »). Chaque autre usage de `threshold_amr`
dans le depot (`HamiltParams.py`, `refinement.py`, `cost_hamiltonian.py`)
le compare a ce score normalise, jamais a un champ d'erreur brut.

Consequence mesuree : sur `init_harris_tearing` (N=256, 150 pas),
`gt.max()` sur TOUT le domaine vaut 0,183 -- **jamais** superieur a
`threshold_amr` (~0,30). `gt_above` valait donc FAUX partout,
systematiquement, quelle que soit la structure reelle du champ : le
diagnostic « Decision quality » (TP/FP/FN) qui en decoule ne pouvait
JAMAIS afficher un vrai positif, sur aucun scenario -- un test qui ne
peut pas echouer, au sens de VIGIL.md, mais pour un printout diagnostique
plutot qu'une assertion.

Correction : `gt` est desormais compare a SA PROPRE moyenne
(`gt.mean()`), comme le fait deja `_gt_error_share` (meme fichier) et
`pixel_precision`/`pixel_recall` de `fig4_comprehensive_comparison.py`
(question 4 de VIGIL.md : deux fonctions du meme depot, memes donnees,
deux chemins qui doivent utiliser la meme convention).

Portee : diagnostic imprime sur la console (`print(...)`), pas ecrit dans
un fichier -- aucun nombre publie n'en depend, `figures/v1_legacy/`
n'ecrivant aucune figure committee dans ce depot.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from hyperparams_loader import load_hyperparams

TARGET_DIM = 2
N = 256


@pytest.fixture(scope="module")
def gt_field():
    """Champ d'erreur GT reel, sur un scenario asymetrique en croix (deux
    quadrants riches en structure, deux calmes) -- le champ qui SEPARE :
    un test qui thresholderait sur une constante globale-uniforme ne
    verrait aucune difference entre les deux formules."""
    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
    sim.init_harris_tearing()
    for _ in range(150):
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)
    state = sim.get_fluxes()
    total = np.zeros((N, N))
    for key in ['vx', 'vy', 'Bx', 'By']:
        f = state[key]
        grad_x = np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)
        grad_y = np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)
        grad_mag = grad_x**2 + grad_y**2
        fpp_xx = np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)
        fpp_yy = np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)
        lap_mag = fpp_xx**2 + fpp_yy**2
        total += grad_mag + lap_mag
    return np.sqrt(total)


def _threshold_amr():
    return load_hyperparams()['threshold_amr']


def _old_above(gt, target_dim, threshold):
    bk = gt.shape[0] // target_dim
    above = np.zeros((target_dim, target_dim), dtype=bool)
    for i in range(target_dim):
        for j in range(target_dim):
            above[i, j] = gt[i*bk:(i+1)*bk, j*bk:(j+1)*bk].mean() > threshold
    return above


# ══════════════════════════════════════════════════════════════════
#  1. Les deux echelles sont bien incommensurables (precondition)
# ══════════════════════════════════════════════════════════════════

def test_gt_scale_never_reaches_threshold_amr(gt_field):
    """Si gt.max() depassait un jour threshold_amr, l'ancien code ne
    serait plus degenere partout -- remesurer plutot que supposer."""
    assert float(gt_field.max()) < _threshold_amr()


def test_old_formula_was_always_false_on_this_field(gt_field):
    above = _old_above(gt_field, TARGET_DIM, _threshold_amr())
    assert not above.any(), (
        "l'ancienne comparaison rendait FAUX partout -- si elle rend "
        "maintenant VRAI quelque part, remesurer avant de conclure")


# ══════════════════════════════════════════════════════════════════
#  2. Le fichier committe, tel qu'il est
# ══════════════════════════════════════════════════════════════════

def _load_gt_quadrant_above_threshold():
    """Charge uniquement `_gt_quadrant_above_threshold` de fig5, sans
    importer le module entier (il execute sa campagne d'analyse a
    l'import, sans garde `__main__`)."""
    import ast
    import types

    path = os.path.join(_REPO_ROOT, "figures", "v1_legacy",
                         "fig5_qaoa_detailed_analysis.py")
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src, filename=path)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "_gt_quadrant_above_threshold")
    module_ast = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module_ast)
    g = {"np": np}
    exec(compile(module_ast, path, "exec"), g)
    return g["_gt_quadrant_above_threshold"]


def test_committed_function_no_longer_takes_a_classical_threshold():
    """Le defaut etait de comparer gt a un seuil d'une autre echelle : la
    fonction corrigee ne prend plus ce seuil du tout, elle se compare a
    elle-meme. Un appel avec l'ancienne signature doit echouer."""
    fn = _load_gt_quadrant_above_threshold()
    with pytest.raises(TypeError):
        fn(np.ones((4, 4)), TARGET_DIM, 0.3)


def test_committed_function_is_not_degenerate_on_a_real_field(gt_field):
    fn = _load_gt_quadrant_above_threshold()
    above = fn(gt_field, TARGET_DIM)
    assert above.any() and not above.all(), (
        f"gt_above devrait separer les quadrants riches des calmes sur ce "
        f"champ (croix asymetrique) : obtenu {above}")
