"""D-84 : la phase 11E attribuait à la physique un écart de **réduction**.

`h2b_v1_hamiltonian_loso.py` imprime, sous le titre *decomposition* :

    V1_class - V2_class  = +0.145   (Lohner + 4-indicator RMS effect)

Ses deux colonnes « classiques » sortent pourtant de la **même** fonction,
`AngleMapper.classical_score` :

* `Sv2c` — le score fin réduit par `block_max`, via `build_patch_hamiltonian` ;
* `Sv1c` — le **même** score fin réduit par `block_avg`.

La seule autre différence est l'opérateur `Jz` : différences **avant** non
divisées (`grid.forward_curl_z`) ici, différences **centrées** divisées par
`2dx` (`solver.get_fluxes`) là-bas. Le terme de Löhner, la moyenne
quadratique des quatre indicateurs, la normalisation : identiques des deux
côtés. Il n'y a donc aucun « effet Löhner » à mesurer entre ces deux
colonnes.

Mesuré (`--dim 4 --N 256 --max-snaps 30 --seed 0`, Re=400, 4 scénarios) :

    écart publié      V1_class(avg) - V2_class(max)   = +0,145
    à réduction égale V1_class(max) - V2_class(max)   = +0,051
    la réduction en porte donc                          ~ +0,094

et, avec une grille de seuil unique pour les quatre colonnes (mesure
indépendante, hors script) : réduction seule sur le champ **identique**
(`V2` moyenné contre `V2` maxé) **+0,149**, écart d'opérateur seul à
réduction égale en moyenne **−0,001** (en moyenne) et **+0,050** (en max).

Ce qui reste après retrait de la réduction n'est donc pas un effet « Löhner
+ RMS » non plus : c'est l'opérateur `Jz`. Aucune part des +0,145 ne mesure
ce que la ligne annonçait — les deux colonnes portent le **même** terme de
Löhner et la **même** moyenne quadratique.

Ce que cela déplace : rien de publié — `results/` ne contient aucun
`v1h_loso_*.npz`. Ce que cela déplace quand même : la lecture que la phase
imprime. Sur le seul fold dont l'IC exclut zéro (`mhd_rotor`), l'écart des
deux colonnes classiques passe de +0,335 à +0,206 à réduction égale — il en
perd 38 %, pas la totalité : hypothèse « il s'effondre » mesurée et écartée.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (os.path.join(_REPO_ROOT, "src"),
           os.path.join(_REPO_ROOT, "study", "pipeline"),
           os.path.join(_REPO_ROOT, "study", "common"),
           os.path.join(_REPO_ROOT, "study", "h2b_prediction")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.PhysToAngle import AngleMapper  # noqa: E402
from Simulation.grid import PeriodicGrid  # noqa: E402
from Simulation.solver import MHDSolver  # noqa: E402
from exact_diagonalisation import build_patch_hamiltonian  # noqa: E402

import h2b_v1_hamiltonian_loso as phase11e  # noqa: E402

_SCENARIOS = ("orszag_tang", "harris_tearing", "kelvin_helmholtz", "mhd_rotor")


def _inputs(dim=4, N=256, re=400):
    files = []
    for sc in _SCENARIOS:
        dns = os.path.join(_REPO_ROOT, "results", f"dns_{sc}_Re{re}_N{N}.npz")
        pat = os.path.join(_REPO_ROOT, "results",
                           f"patches_{sc}_Re{re}_N{N}_dim{dim}.npz")
        if not (os.path.exists(dns) and os.path.exists(pat)):
            return None
        files += [dns, pat]
    return files


def test_the_two_classical_columns_are_the_same_function():
    """La preuve directe : à `Jz` identique, les deux champs fins coïncident.

    C'est l'entrée qui SÉPARE la bonne explication de la mauvaise. Si l'écart
    venait du terme de Löhner ou de la moyenne quadratique, forcer le même
    `Jz` n'y changerait rien ; il l'annule exactement.
    """
    files = _inputs()
    if files is None:
        pytest.skip("artefacts d'entrée N=256 dim=4 absents de ce checkout")
    dns = np.load(files[0])            # orszag_tang
    vx = dns["vx"][0].astype(np.float64); vy = dns["vy"][0].astype(np.float64)
    Bx = dns["Bx"][0].astype(np.float64); By = dns["By"][0].astype(np.float64)
    N = vx.shape[0]

    _, _, v2_full = build_patch_hamiltonian(
        vx, vy, Bx, By, N, 4, 400, threshold_amr=0.15, use_v2=True, c_bias=1.0)
    v1_full = phase11e.v1_classical_score(vx, vy, Bx, By)

    # le Jz de la colonne V2 : différences centrées / 2dx
    sim = MHDSolver(PeriodicGrid(N), dt=1e-4, Re=400, Rm=400)
    sim.vx, sim.vy, sim.Bx, sim.By = vx, vy, Bx, By
    jz_centre = sim.get_fluxes()["Jz"]
    v1_with_v2_jz = AngleMapper.classical_score(
        dict(vx=vx, vy=vy, Bx=Bx, By=By, Jz=jz_centre))

    assert np.array_equal(v1_with_v2_jz, v2_full), (
        "à Jz identique les deux colonnes devraient être le MÊME tableau : "
        "elles appellent la même fonction")
    # et sans cette substitution, elles diffèrent — sinon le test ne dirait rien
    assert not np.array_equal(v1_full, v2_full)
    # l'écart d'opérateur Jz, mesuré sur CE snapshot (orszag_tang, t0) ;
    # sur les 120 snapshots des 4 scénarios : médiane 0,015, max 0,164
    assert float(np.max(np.abs(v1_full - v2_full))) == pytest.approx(
        0.0097, abs=5e-4)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """La configuration mesurée, jouée une fois — sortie hors du dépôt."""
    files = _inputs()
    if files is None:
        pytest.skip("artefacts d'entrée N=256 dim=4 absents de ce checkout")
    out_dir = tmp_path_factory.mktemp("phase11e")
    for src in files:
        os.symlink(src, out_dir / os.path.basename(src))

    old_dir, old_argv = phase11e.RESULTS_DIR, sys.argv
    phase11e.RESULTS_DIR = str(out_dir)
    sys.argv = ["h2b_v1_hamiltonian_loso.py", "--dim", "4", "--N", "256",
                "--max-snaps", "30", "--n-boot", "500", "--seed", "0",
                "--re", "400"]
    try:
        phase11e.main()
    finally:
        phase11e.RESULTS_DIR, sys.argv = old_dir, old_argv
    return np.load(out_dir / "v1h_loso_N256_dim4.npz", allow_pickle=True)


def test_the_published_gap_is_mostly_the_reduction(run):
    """+0,145 publié, +0,051 à réduction égale : la réduction en porte ~+0,094."""
    v2 = run["f1_v2_class"]
    publie = float(np.mean(run["f1_v1_class"] - v2))
    egal = float(np.mean(run["f1_v1_class_maxpool"] - v2))

    assert publie == pytest.approx(0.145, abs=5e-3)
    assert egal == pytest.approx(0.051, abs=5e-3)
    assert publie - egal == pytest.approx(0.094, abs=1e-2), (
        "la part de l'écart imputable à la seule réduction block_avg/block_max")


def test_the_one_decisive_fold_loses_a_third_of_its_gap(run):
    """`mhd_rotor` est le seul fold dont l'IC exclut zéro.

    Première hypothèse, mesurée et **fausse** : « il s'effondre à réduction
    égale ». Il n'en perd que 38 %. La réduction en porte +0,129 sur +0,335 ;
    le reste vient de l'opérateur `Jz`. Dit tel quel — la conclusion tient
    sans lui : aucune part de cet écart n'est un effet « Löhner + RMS »,
    puisque les deux colonnes contiennent le même terme de Löhner.
    """
    i = list(run["scenarios"]).index("mhd_rotor")
    publie = float(run["f1_v1_class"][i]) - float(run["f1_v2_class"][i])
    egal = float(run["f1_v1_class_maxpool"][i]) - float(run["f1_v2_class"][i])
    assert publie == pytest.approx(0.335, abs=5e-3)
    assert egal == pytest.approx(0.206, abs=5e-3)
    assert publie - egal == pytest.approx(0.129, abs=1e-2)


def test_the_control_column_stays_in_the_artefact(run):
    """Épingle la correction : sans cette colonne, la mauvaise lecture revient."""
    assert "f1_v1_class_maxpool" in run.files
    assert run["f1_v1_class_maxpool"].shape == run["f1_v1_class"].shape
