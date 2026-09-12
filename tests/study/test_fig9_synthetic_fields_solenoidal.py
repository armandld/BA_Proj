"""D-97 — `fig9_synthetic_unit_tests.py` : X et Y echanges dans 3 des 4
generateurs de champs synthetiques.

`make_vortex_core`, `make_current_sheet` et `make_xpoint` construisent des
champs `Bx`/`By` cenes representer une topologie physique precise (vortex,
nappe de courant, point-X). Les trois depaquetaient `np.mgrid`/diffusaient
un broadcast dans le mauvais sens par rapport a la convention du depot
(`Simulation/grid.py` : `AXIS_X = 0`, `AXIS_Y = 1`, `indexing='ij'`) :

  - `make_vortex_core`/`make_xpoint` : `y, x = np.mgrid[0:N, 0:N]` nommait
    "y" le tableau qui varie en realite le long de l'axe 0 (X), et
    reciproquement — `Bx = cos(y)*0.5` variait donc le long de SON PROPRE
    axe au lieu de l'axe perpendiculaire.
  - `make_current_sheet` : `np.tanh(...)[np.newaxis, :]` diffuse le long de
    l'axe 1 (Y) un profil cense varier "at x=N/2" (axe 0).

Consequence physique : `div B != 0` par construction, jamais nettoye
(`MHDSolver.PROJECT_B = False`, verifie ci-dessous) — persiste a travers
toute l'evolution temporelle. Le champ que ces figures analysent n'est pas
la topologie MHD que leur nom promet.

Mesure (N=256, meme operateur que le depot :
`Simulation.grid.divergence(Bx, By, fixed_curl=True)`), a la construction
(avant toute evolution) :

| generateur | max|div B| avant | echelle B | ratio avant | apres |
|---|---|---|---|---|
| vortex_core   | 0.0245 | 0.50 | 4.9 %  | 0.0 (bit a bit) |
| current_sheet | 2.0000 | 1.00 | 200 %  | 0.0 (bit a bit) |
| xpoint        | 3.1750 | 1.50 | 212 %  | 0.0 (bit a bit) |

`make_uniform_noise` (champ non structure, controle negatif) n'a pas ce
defaut — non teste ici.

Reference de comparaison : `init_harris_tearing()`, deja correct
(`Bx = f(Y)` seul, `By` derive d'une fonction de flux) — `max|div B|`
mesure a 1e-4 (bruit FD4), voir `docs/RESULTS.md` D-1/D-27.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from Simulation.grid import PeriodicGrid, divergence
from Simulation.solver import MHDSolver

N = 256


def _divb_ratio(sim):
    divb = divergence(sim.Bx, sim.By, fixed_curl=True)
    b_scale = max(float(np.max(np.abs(sim.Bx))), float(np.max(np.abs(sim.By))), 1e-12)
    return float(np.max(np.abs(divb))) / b_scale


def _fresh_sim():
    grid = PeriodicGrid(resolution_N=N)
    return MHDSolver(grid, dt=1e-3, Re=800, Rm=800)


# ══════════════════════════════════════════════════════════════════
#  0. La precondition du defaut : B n'est jamais projete
# ══════════════════════════════════════════════════════════════════

def test_the_solver_never_cleans_b_divergence():
    """Si PROJECT_B devenait True, ce defaut serait auto-corrige par le
    solveur et ce fichier de test devrait etre remesure."""
    assert MHDSolver.PROJECT_B is False


# ══════════════════════════════════════════════════════════════════
#  1. Reference : un scenario deja correct du depot
# ══════════════════════════════════════════════════════════════════

def test_the_canonical_harris_tearing_is_already_solenoidal():
    """Le champ qui SEPARE : si cette assertion echouait, ce serait
    l'operateur de mesure qui serait en cause, pas les generateurs de fig9."""
    sim = _fresh_sim()
    sim.init_harris_tearing()
    assert _divb_ratio(sim) < 1e-3


# ══════════════════════════════════════════════════════════════════
#  2. La mesure du defaut, figee (epingle l'ancien comportement)
# ══════════════════════════════════════════════════════════════════

def _old_vortex_core(sim):
    y, x = np.mgrid[0:N, 0:N] / N * 2 * np.pi
    sim.Bx = np.cos(y) * 0.5
    sim.By = -np.cos(x) * 0.5


def _old_current_sheet(sim):
    x = np.arange(N) / N
    sim.Bx[:] = 0.3
    sim.By[:] = np.tanh((x - 0.5) * 40)[np.newaxis, :]


def _old_xpoint(sim):
    y_arr, x_arr = np.mgrid[0:N, 0:N] / N
    sim.By = 1.5 * (np.tanh((x_arr - 0.25) * 30) - np.tanh((x_arr - 0.75) * 30) - 1.0)
    sim.Bx = 1.5 * np.tanh((y_arr - 0.5) * 30)


@pytest.mark.parametrize("build,min_ratio", [
    (_old_vortex_core, 0.04),
    (_old_current_sheet, 1.5),
    (_old_xpoint, 1.5),
])
def test_the_old_generators_were_not_solenoidal(build, min_ratio):
    sim = _fresh_sim()
    build(sim)
    ratio = _divb_ratio(sim)
    assert ratio > min_ratio, (
        f"ecart {ratio:.4f} : si le defaut est devenu benin, remesurer "
        "plutot que d'abaisser le seuil")


# ══════════════════════════════════════════════════════════════════
#  3. Le correctif, mesure a la construction (avant toute evolution)
# ══════════════════════════════════════════════════════════════════
#
# `make_*(N)` fait tourner 20-50 pas de `step_full` avant de rendre `sim` :
# une derive numerique FD4/RK4 propre au schema (gradients raides, sans
# rapport avec l'echange d'axes) s'y ajoute et n'est pas ce que ce defaut
# corrige. On mesure donc juste apres la construction du champ, comme la
# mesure avant/apres consignee dans docs/RESULTS.md (D-97) : la fonction
# `_process_score`-style "meme operateur" veut ici "meme point du pipeline".

def _new_vortex_core(sim):
    x, y = np.mgrid[0:N, 0:N] / N * 2 * np.pi
    sim.Bx = np.cos(y) * 0.5
    sim.By = -np.cos(x) * 0.5


def _new_current_sheet(sim):
    x = np.arange(N) / N
    sim.Bx[:] = 0.3
    sim.By[:] = np.tanh((x - 0.5) * 40)[:, np.newaxis]


def _new_xpoint(sim):
    x_arr, y_arr = np.mgrid[0:N, 0:N] / N
    sim.By = 1.5 * (np.tanh((x_arr - 0.25) * 30) - np.tanh((x_arr - 0.75) * 30) - 1.0)
    sim.Bx = 1.5 * np.tanh((y_arr - 0.5) * 30)


@pytest.mark.parametrize("build", [_new_vortex_core, _new_current_sheet, _new_xpoint])
def test_the_fixed_construction_is_exactly_solenoidal(build):
    sim = _fresh_sim()
    build(sim)
    divb = divergence(sim.Bx, sim.By, fixed_curl=True)
    assert np.max(np.abs(divb)) == 0.0, (
        "la construction corrigee devrait annuler div B bit a bit, comme "
        "le fait deja init_harris_tearing (D-1/D-27)")


# ══════════════════════════════════════════════════════════════════
#  4. Le fichier committe construit un champ effectivement solenoidal
# ══════════════════════════════════════════════════════════════════
#
# Interroge le comportement du module committe, pas son texte (VIGIL.md :
# chercher une chaine dans un fichier teste sa mise en forme, pas ce qu'il
# fait). On execute uniquement la construction du champ -- pas la boucle
# d'evolution, hors sujet pour ce defaut -- en tronquant l'AST de la
# fonction juste avant sa premiere boucle `for`.

def _load_construction_only(func_name):
    import ast
    import types

    path = os.path.join(_REPO_ROOT, "figures", "v1_legacy",
                         "fig9_synthetic_unit_tests.py")
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src, filename=path)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == func_name)
    cut = next(i for i, stmt in enumerate(fn.body) if isinstance(stmt, ast.For))
    fn.body = fn.body[:cut] + [ast.Return(value=ast.Name(id="sim", ctx=ast.Load()))]
    ast.fix_missing_locations(fn)
    module_ast = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module_ast)

    g = {"np": np, "PeriodicGrid": PeriodicGrid, "MHDSolver": MHDSolver}
    exec(compile(module_ast, path, "exec"), g)
    return g[func_name]


@pytest.mark.parametrize("func_name", ["make_vortex_core", "make_current_sheet", "make_xpoint"])
def test_committed_generator_is_solenoidal_right_after_construction(func_name):
    build = _load_construction_only(func_name)
    sim = build(N)
    divb = divergence(sim.Bx, sim.By, fixed_curl=True)
    assert np.max(np.abs(divb)) == 0.0, (
        f"{func_name}, tel que committe : div B non nul juste apres la "
        "construction du champ, avant toute evolution")
