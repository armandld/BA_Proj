"""psi est nul dans l'etude, actif dans le pipeline deploye.

`qaoa_inputs.prepare_qaoa_inputs` met psi a zero en dur
(« no temporal flux in study »), alors que le pipeline V1 le calcule
(`refinement.py:181`) et que la campagne Optuna a regle les hyperparametres
en le faisant tourner (`train_hyperparams.py:70` importe `pipeline`).

L'etude evalue donc une variante du modele amputee d'un de ses trois
encodages. Ces tests epinglent le fait, et verifient que la variante
`with_psi=True` rebranche EXACTEMENT le psi du pipeline plutot qu'un psi
vraisemblable : un psi fabrique serait indiscernable du vrai, ce qui est le
defaut que cette etude traque.
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

N = 32
DIM = 2
RE = 400


def _two_snapshots():
    """Deux instantanes consecutifs d'un meme ecoulement."""
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver

    sim = MHDSolver(PeriodicGrid(N), dt=1e-3, Re=RE, Rm=RE)
    sim.init_kelvin_helmholtz()
    for _ in range(20):
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)
    prev = {k: np.array(v) for k, v in sim.get_fluxes().items()}
    for _ in range(5):
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)
    cur = sim.get_fluxes()
    return cur, prev


def test_the_study_path_zeroes_psi():
    """Le defaut lui-meme : sans with_psi, psi est identiquement nul."""
    from qaoa_inputs import prepare_qaoa_inputs

    cur, _prev = _two_snapshots()
    data_in, _hp, _score = prepare_qaoa_inputs(
        cur["vx"], cur["vy"], cur["Bx"], cur["By"], N, DIM, RE)

    assert np.all(np.asarray(data_in["psi_h"]) == 0.0)
    assert np.all(np.asarray(data_in["psi_v"]) == 0.0)


def test_with_psi_refuses_a_lone_snapshot():
    """psi est une derivee temporelle : sans instantane precedent, on leve.

    Renvoyer des zeros silencieusement rendrait `with_psi=True` indiscernable
    de `with_psi=False`.
    """
    from qaoa_inputs import prepare_qaoa_inputs

    cur, _prev = _two_snapshots()
    with pytest.raises(ValueError, match="prev_fields"):
        prepare_qaoa_inputs(cur["vx"], cur["vy"], cur["Bx"], cur["By"],
                            N, DIM, RE, with_psi=True)


def test_with_psi_produces_a_nonzero_psi():
    """Rebranche, psi doit etre effectivement non nul."""
    from qaoa_inputs import prepare_qaoa_inputs

    cur, prev = _two_snapshots()
    data_in, _hp, _score = prepare_qaoa_inputs(
        cur["vx"], cur["vy"], cur["Bx"], cur["By"], N, DIM, RE,
        prev_fields=prev, with_psi=True)

    psi = np.concatenate([np.asarray(data_in["psi_h"]).ravel(),
                          np.asarray(data_in["psi_v"]).ravel()])
    assert np.max(np.abs(psi)) > 1e-9, (
        f"psi rebranche mais nul (max |psi| = {np.max(np.abs(psi)):.3e}) : "
        "la variante ne teste alors rien")


def test_with_psi_matches_the_deployed_encoder_bit_for_bit():
    """LE test qui compte : le psi obtenu doit etre CELUI du pipeline.

    On appelle directement `refinement._prepare_vqa_input` avec les memes
    entrees et on exige l'egalite exacte. Un psi qui ressemblerait au vrai
    sans l'etre passerait tous les autres tests.
    """
    from types import SimpleNamespace

    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver
    from Simulation.PhysToAngle import AngleMapper
    from Simulation.refinement import _prepare_vqa_input
    from qaoa_inputs import prepare_qaoa_inputs, TRAINED_THRESHOLD
    from Simulation.HamiltParams import PhysicalMapper

    cur, prev = _two_snapshots()

    data_in, _hp, _score = prepare_qaoa_inputs(
        cur["vx"], cur["vy"], cur["Bx"], cur["By"], N, DIM, RE,
        prev_fields=prev, with_psi=True)

    # Reference : le chemin deploye, reconstruit a la main.
    grid = PeriodicGrid(N)
    sim = MHDSolver(grid, dt=1e-4, Re=RE, Rm=RE)
    sim.vx, sim.vy = cur["vx"], cur["vy"]
    sim.Bx, sim.By = cur["Bx"], cur["By"]
    physics_state = sim.get_fluxes()

    am = AngleMapper()
    Phi = am.compute_stress_flux(physics_state)
    Phi_prev = am.compute_stress_flux(prev)
    fh, fv = Phi["phi_horizontal"], Phi["phi_vertical"]
    ph, pv = Phi_prev["phi_horizontal"], Phi_prev["phi_vertical"]
    apd = 0.5 * (np.mean(np.abs(fh - ph)) + np.mean(np.abs(fv - pv)))

    dx = 2 * np.pi / N
    hm = PhysicalMapper(cs=1.0, nu=1.0 / RE, eta_mhd=1.0 / RE, dx=dx)

    ref = _prepare_vqa_input(
        fh, fv, ph, pv, AngleMapper.classical_score(physics_state),
        physics_state, (0, N, 0, N), 0, am,
        SimpleNamespace(AdvAnomaliesEnable=False),
        apd, 1.0, DIM, HamiltMapper=hm, sim=sim,
        threshold_amr=TRAINED_THRESHOLD)
    assert ref is not None
    (_th, _tv, ref_psi_h, ref_psi_v), _, _ = ref

    np.testing.assert_allclose(
        np.asarray(data_in["psi_h"]), ref_psi_h, rtol=0, atol=0,
        err_msg="psi de l'etude != psi du pipeline deploye")
    np.testing.assert_allclose(
        np.asarray(data_in["psi_v"]), ref_psi_v, rtol=0, atol=0,
        err_msg="psi de l'etude != psi du pipeline deploye")
