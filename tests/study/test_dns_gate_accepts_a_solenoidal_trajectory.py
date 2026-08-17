"""D-73 — le portail de validation DNS doit juger la contrainte, pas l'ecart
entre deux stencils.

`validate_one` (`study/pipeline/dns_extension.py`) est la porte que franchit
CHAQUE trajectoire DNS nouvellement generee : elle rejette la trajectoire si
`div_rel_max > 1e-3`. Cette valeur venait de `analyse_one`, dans le fichier
GELE `dns_validation.py`, qui la calcule au SPECTRAL — et dont le commentaire
porte la condition devenue fausse : « should be O(eps_machine) WHEN THE FFT
PROJECTION IS APPLIED ».

Depuis D-25 elle ne l'est plus pour B : `PROJECT_B = False`, B est solenoidal
AUX DIFFERENCES FINIES par construction. Le portail rejetait donc des
trajectoires saines — mesure de bout en bout, DNS generee a HEAD
(harris_tearing, Re=400, N=64, seed=0) : `divB 1.6205e-02` -> FAIL, contre
`5.0573e-06` -> OK avec l'operateur assorti.

Ces tests n'ecrivent rien dans le depot : la trajectoire est construite avec
le solveur puis ecrite dans le `tmp_path` de pytest.
"""

import importlib.util
import os

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_DNS_EXT = os.path.join(_REPO_ROOT, "study", "pipeline", "dns_extension.py")

_DIV_TOL = 1e-3          # le seuil de validate_one lui-meme


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def dns_ext():
    return _load(_DNS_EXT, "dns_extension_d73_mod")


@pytest.fixture(scope="module")
def head_trajectory(tmp_path_factory):
    """Une trajectoire produite par le solveur TEL QU'IL EST, hors du depot.

    Meme empaquetage que `save_seeded` — dont le float32 est reproduit, car
    c'est lui qui fixe le plancher de ce qu'on peut mesurer sur un artefact.
    """
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver

    N, n_snap = 64, 6
    sim = MHDSolver(PeriodicGrid(N), dt=1e-3, Re=400, Rm=400)
    sim.init_harris_tearing()
    vx, vy, Bx, By, t = [], [], [], [], []
    for i in range(n_snap):
        for _ in range(4):
            sim.adapt_dt(cfl_target=0.4)
            sim.step_full(record_stats=False)
        vx.append(sim.vx.astype(np.float32))
        vy.append(sim.vy.astype(np.float32))
        Bx.append(sim.Bx.astype(np.float32))
        By.append(sim.By.astype(np.float32))
        t.append(float(i))

    path = str(tmp_path_factory.mktemp("dns_d73") / "dns_head.npz")
    np.savez_compressed(
        path, vx=np.array(vx), vy=np.array(vy),
        Bx=np.array(Bx), By=np.array(By),
        t=np.array(t), step=np.arange(n_snap, dtype=np.int32),
        meta_scenario="harris_tearing", meta_Re=400, meta_N=N,
        meta_diverged=False)
    return path


def test_the_gate_accepts_a_trajectory_whose_constraint_holds(
        dns_ext, head_trajectory):
    """Le critere du portail, sur une trajectoire produite par le solveur.

    Echoue sur la version d'avant D-73 : la valeur spectrale du fichier gele
    depasse le seuil de plus d'un ordre de grandeur, sur une trajectoire dont
    la contrainte est respectee.
    """
    div_rel = dns_ext.div_rel_max_fixed(head_trajectory)
    assert div_rel <= _DIV_TOL, (
        f"le portail rejette une trajectoire saine : {div_rel:.4e} > "
        f"{_DIV_TOL:.0e}. La contrainte que le solveur garantit est mesuree "
        "par l'operateur FD4, pas par le spectral (D-73)")


def test_validate_one_no_longer_fails_on_the_divergence_check(
        dns_ext, head_trajectory):
    """La garantie annoncee, pas l'absence de plantage.

    On interroge `validate_one` elle-meme et on regarde s'il reste un echec
    PORTANT SUR divB. Les autres verdicts de la fonction (tearing, energie)
    ne sont pas le sujet de D-73 et peuvent legitimement echouer ici — D-39
    est ouvert sur exactement ce point.
    """
    fails, log = dns_ext.validate_one(head_trajectory, "harris_tearing")
    div_fails = [f for f in fails if "divB" in f]
    assert not div_fails, (
        f"le check de divergence rejette encore la trajectoire : {div_fails}")
    assert any("div=" in entry for entry in log), (
        "le journal ne porte plus la divergence : un balayage muet")


def test_the_frozen_spectral_value_still_disagrees_and_is_still_logged(
        dns_ext, head_trajectory):
    """Epingle l'ANCIEN comportement, et le garde visible.

    Deux choses a la fois, parce qu'elles se defont ensemble : l'ecart entre
    les deux operateurs (sans lui, ce test ne separerait rien) et le fait que
    la valeur gelee reste JOURNALISEE a cote de la valeur retenue — une
    correction qui l'effacerait rendrait l'ecart invisible.

    Mord aussi si `PROJECT_B` repasse a True : B redevient spectralement
    solenoidal, l'ecart se referme, et le choix d'operateur doit etre
    retranche.
    """
    from dns_validation import analyse_one
    from Simulation.solver import MHDSolver

    assert MHDSolver.PROJECT_B is False, (
        "PROJECT_B est repasse a True : le choix d'operateur de D-73 doit "
        "etre retranche")

    spectral = float(analyse_one(head_trajectory)["div_rel_max"])
    matched = dns_ext.div_rel_max_fixed(head_trajectory)

    assert spectral > _DIV_TOL, (
        f"la valeur gelee vaut {spectral:.4e}, sous le seuil de "
        f"{_DIV_TOL:.0e} : elle valait 1.6205e-02 a la mesure de D-73. "
        "L'ecart entre les deux operateurs a change, D-73 doit etre remesure")
    assert spectral > 100 * matched, (
        f"spectral {spectral:.4e} contre assorti {matched:.4e} : si les deux "
        "coincidaient, ce test ne separerait rien")

    _fails, log = dns_ext.validate_one(head_trajectory, "harris_tearing")
    assert any("spectral gele" in entry for entry in log), (
        "la valeur spectrale gelee n'est plus journalisee : l'ecart entre "
        "les deux operateurs redevient invisible")
