"""Tests V3 Task 8 : extension DNS (8 scenarios x graines physiques).

Helpers purs + initialisation reelle des 8 scenarios sur un mini-solveur
N=16 (module scenario V1 importe, jamais re-implemente). Les heures de
DNS et la validation 1b complete sont validees par l'execution reelle.
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

_HERE = os.path.dirname(os.path.abspath(__file__))

from t8_dns_extension import (
    EXTRA_SCENARIOS,
    V3_SCENARIO_CONFIG,
    _extended_init,
    energy_non_increasing,
    perturb_fields,
    presence_matrix,
    seeded_dns_path,
    seeded_patches_path,
)

ALL_8 = ["orszag_tang", "harris_tearing", "kelvin_helmholtz",
         "mhd_rotor"] + EXTRA_SCENARIOS


# ------------------------------ nommage -------------------------------

def test_seeded_paths_seed0_matches_v2_convention():
    assert seeded_dns_path("/r", "orszag_tang", 400, 256, 0) == \
        "/r/dns_orszag_tang_Re400_N256.npz"
    assert seeded_patches_path("/r", "orszag_tang", 400, 256, 4, 0) == \
        "/r/patches_orszag_tang_Re400_N256_dim4.npz"


def test_seeded_paths_seedk_suffix():
    assert seeded_dns_path("/r", "lamb_oseen", 800, 256, 3) == \
        "/r/dns_lamb_oseen_Re800_N256_seed3.npz"
    assert seeded_patches_path("/r", "lamb_oseen", 800, 256, 4, 3) == \
        "/r/patches_lamb_oseen_Re800_N256_dim4_seed3.npz"


# ---------------------------- perturbation ----------------------------

def test_perturb_seed0_is_identity():
    vx = np.random.default_rng(0).normal(size=(8, 8))
    vy = np.random.default_rng(1).normal(size=(8, 8))
    px, py = perturb_fields(vx, vy, seed=0, amplitude=0.1)
    assert px is vx and py is vy   # aucune copie, aucune perturbation


def test_perturb_deterministic_and_amplitude():
    vx = np.zeros((64, 64)); vy = np.zeros((64, 64))
    p1x, p1y = perturb_fields(vx, vy, seed=1, amplitude=0.1)
    p1x_b, _ = perturb_fields(vx, vy, seed=1, amplitude=0.1)
    p2x, _ = perturb_fields(vx, vy, seed=2, amplitude=0.1)
    np.testing.assert_array_equal(p1x, p1x_b)      # deterministe
    assert not np.array_equal(p1x, p2x)            # graines distinctes
    assert p1x.std() == pytest.approx(0.1, rel=1e-6)  # amplitude V1
    np.testing.assert_array_equal(vx, 0.0)         # entree non mutee


def test_perturbation_is_band_limited():
    # aucune energie au-dela de k_cut : compatible avec le projecteur
    # spectral (exact hors Nyquist) et avec la perturbation V1
    # grande echelle
    from t8_dns_extension import PERT_K_CUT
    N = 64
    p, _ = perturb_fields(np.zeros((N, N)), np.zeros((N, N)),
                          seed=3, amplitude=0.1)
    ph = np.fft.fft2(p)
    k = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(k, k, indexing="ij")
    high = np.sqrt(KX ** 2 + KY ** 2) > PERT_K_CUT
    assert np.abs(ph[high]).max() < 1e-10 * np.abs(ph).max()


# ------------------------------ energie -------------------------------

def test_energy_non_increasing():
    ok, _ = energy_non_increasing([1.0, 0.9, 0.85, 0.85])
    assert ok
    ok, mx = energy_non_increasing([1.0, 0.9, 0.95])
    assert not ok and mx == pytest.approx(0.05)
    ok, _ = energy_non_increasing([1.0, 1.0005], tol=1e-3)  # tolerance RK
    assert ok


# --------------------------- matrice presence -------------------------

def test_presence_matrix(tmp_path):
    d = str(tmp_path)
    open(seeded_dns_path(d, "orszag_tang", 400, 256, 0), "w").close()
    open(seeded_dns_path(d, "orszag_tang", 800, 256, 0), "w").close()
    open(seeded_dns_path(d, "orszag_tang", 400, 256, 1), "w").close()
    pm = presence_matrix(d, ["orszag_tang", "lamb_oseen"],
                         [400, 800], 256, [0, 1])
    assert pm[("orszag_tang", 0)] == 2
    assert pm[("orszag_tang", 1)] == 1
    assert pm[("lamb_oseen", 0)] == 0


# ---------------------- config des 4 nouveaux -------------------------

def test_v3_config_covers_exactly_the_extra_scenarios():
    assert set(V3_SCENARIO_CONFIG) == set(EXTRA_SCENARIOS)
    for cfg in V3_SCENARIO_CONFIG.values():
        assert {"warmup_steps", "t_max", "snapshot_dt"} <= set(cfg)
        assert cfg["t_max"] / cfg["snapshot_dt"] >= 19  # >= ~20 snaps


# ---------------------- init reelle des 8 scenarios -------------------

def _fresh_sim(N=16):
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver
    return MHDSolver(PeriodicGrid(N), dt=1e-4, Re=400, Rm=400)


@pytest.mark.parametrize("sc", ALL_8)
def test_all_8_scenarios_initialise(sc):
    sim = _fresh_sim()
    _extended_init(sim, sc, seed=0, amplitude=0.1)
    for f in (sim.vx, sim.vy, sim.Bx, sim.By):
        assert f.shape == (16, 16)
        assert np.all(np.isfinite(f))


def test_seed0_matches_plain_v1_init():
    sim_a = _fresh_sim(); sim_a.init_lamb_oseen_vortex()
    sim_b = _fresh_sim()
    _extended_init(sim_b, "lamb_oseen", seed=0, amplitude=0.1)
    np.testing.assert_array_equal(sim_a.vx, sim_b.vx)
    np.testing.assert_array_equal(sim_a.By, sim_b.By)


def test_seeded_init_differs_and_velocity_stays_div_free():
    from phase1b_dns_validation import div_B
    sim0 = _fresh_sim()
    _extended_init(sim0, "orszag_tang", seed=0, amplitude=0.1)
    sim1 = _fresh_sim()
    _extended_init(sim1, "orszag_tang", seed=1, amplitude=0.1)
    assert not np.array_equal(sim0.vx, sim1.vx)     # IC perturbees
    # B non perturbe ; seul l'aller-retour FFT de la projection le
    # touche, a l'epsilon machine pres
    np.testing.assert_allclose(sim0.Bx, sim1.Bx, rtol=0, atol=1e-14)
    # la reprojection restaure div v = 0 (spectral, eps machine)
    dv = div_B(sim1.vx, sim1.vy, 2 * np.pi / 16)
    rms = np.sqrt((sim1.vx ** 2 + sim1.vy ** 2).mean())
    assert np.abs(dv).max() / rms < 1e-10


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        _extended_init(_fresh_sim(), "nope", seed=0, amplitude=0.1)


# ------------------ observable KH corrigee (D2) ------------------------

def _kh_like_fields(N=32, amp=0.0):
    """Flot de base v_flow(Y) (variant sur l'axe 1) + perturbation
    sin(X) d'amplitude amp (variant sur l'axe 0)."""
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    vx = np.tanh((Y - np.pi) / 0.5)
    vy = amp * np.sin(X)
    return vx, vy


def test_fixed_observable_removes_base_flow_exactly():
    from t8_dns_extension import fluctuating_ke_fixed
    vx, vy = _kh_like_fields(amp=0.0)
    # flot de base retire (a l'epsilon machine de la moyenne pres)
    assert fluctuating_ke_fixed(vx, vy) < 1e-30
    vx2, vy2 = _kh_like_fields(amp=0.1)
    # seule la perturbation reste : 0.5 * <(0.1 sin X)^2> = 0.0025
    assert fluctuating_ke_fixed(vx2, vy2) == pytest.approx(0.0025)


def test_phase1b_observable_is_contaminated_by_base_flow():
    # documente le bug D2 : la version 1b (moyenne sur l'axe 1) laisse
    # le profil de base dans Ep — c'est la justification de la copie
    # corrigee cote v3 (phase 1b reste intouchee)
    from phase1b_dns_validation import fluctuating_KE
    vx, vy = _kh_like_fields(amp=0.0)
    assert fluctuating_KE(vx, vy) > 0.1          # ~variance du profil


def test_check_kh_fixed_detects_growth(tmp_path):
    from t8_dns_extension import check_kh_fixed
    N, n_snaps = 32, 13
    t = np.linspace(0.0, 1.2, n_snaps)
    vx = np.zeros((n_snaps, N, N), dtype=np.float32)
    vy = np.zeros_like(vx)
    for i, amp in enumerate(0.01 * np.exp(t)):   # croissance exp
        fx, fy = _kh_like_fields(N, amp)
        vx[i], vy[i] = fx, fy
    path = str(tmp_path / "dns_kh.npz")
    np.savez(path, t=t, vx=vx, vy=vy)
    chk = check_kh_fixed(path)
    assert chk["ok"] and chk["growth"] > 1.1
    # perturbation constante -> pas de croissance
    for i in range(n_snaps):
        fx, fy = _kh_like_fields(N, 0.01)
        vx[i], vy[i] = fx, fy
    np.savez(path, t=t, vx=vx, vy=vy)
    chk = check_kh_fixed(path)
    assert not chk["ok"]
    assert chk["growth"] == pytest.approx(1.0, abs=1e-6)
