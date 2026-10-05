"""Lecture du QAOA et seuil du biais : le constat de la revue d'octobre.

Le panel H0 trouve rho(E_gap, F1) > 0 : mieux resoudre le Hamiltonien
degrade la decision. La revue a identifie deux faits de code qui
l'expliquent, et ce fichier les epingle :

  * le biais est centre sur le seuil AMR deploye (0,15), tres en dessous du
    seuil qui maximise F1 contre le label statique (0,51 a 0,60) : l'etat
    fondamental raffine presque tout ;
  * l'etude lit le QAOA par vote majoritaire (P(1) > 0,5 par qubit), le
    solveur deploye par moyenne des deux marginales d'une cellule comparee
    au seuil. Lu par vote majoritaire, le circuit NON optimise rend
    s > 0,5, pas la decision classique ; sur le panel V1 le F1 du QAOA egale
    celui de ce masque sur 11 instances sur 12.

Il epingle aussi les deux outils ajoutes pour le recentrage :
`qaoa_inputs.prepare_qaoa_inputs(threshold_amr=...)`,
`qaoa_inputs.deployed_readout` (prouve identique a la decision du solveur
deploye) et les options `--bias-threshold` / `--readout` du panel H0, dont
la calibration ne voit jamais le scenario evalue.
"""
import importlib.util
import inspect
import os
import subprocess
import sys

import numpy as np
import pytest


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_REPO_ROOT = _repo_root()
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import warnings  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    import qaoa_inputs  # noqa: E402
    from config import RESULTS_DIR, V2_THRESHOLD  # noqa: E402

from qaoa_inputs import (  # noqa: E402
    READOUTS, deployed_readout, majority_readout, prepare_qaoa_inputs,
    run_qaoa_on_snapshot)

_PANEL = os.path.join(_REPO_ROOT, "study", "h0_selection",
                      "h0_optimiser_equivalence.py")
_DIAG = os.path.join(_REPO_ROOT, "study", "h0_selection",
                     "h0_readout_threshold_diagnostic.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def panel():
    return _load(_PANEL, "h0panel_readout")


@pytest.fixture(scope="module")
def diag():
    return _load(_DIAG, "h0_readout_diag")


def _snapshot(scenario="orszag_tang", re=400, N=96, si=10):
    d = np.load(os.path.join(RESULTS_DIR, f"dns_{scenario}_Re{re}_N{N}.npz"))
    return [d[k][si].astype(np.float64) for k in ("vx", "vy", "Bx", "By")]


# ── les deux regles marginales -> decision ────────────────────────────

def test_the_two_readouts_are_the_ones_announced():
    """Vote majoritaire par qubit contre moyenne de cellule au seuil."""
    assert READOUTS == ("majority", "deployed")
    dim = 2
    marg = np.array([0.90, 0.20, 0.55, 0.05,     # aretes horizontales
                     0.20, 0.20, 0.55, 0.05])    # aretes verticales
    mh, mv = majority_readout(marg, dim)
    assert (mh | mv).ravel().tolist() == [True, False, True, False]
    # moyennes de cellule : 0,55 0,20 0,55 0,05
    dh, dv = deployed_readout(marg, dim, 0.6)
    assert not (dh | dv).any()
    dh, dv = deployed_readout(marg, dim, 0.15)
    assert (dh | dv).ravel().tolist() == [True, True, True, False]
    np.testing.assert_array_equal(dh, dv)


def test_unoptimised_circuit_deployed_is_classical_majority_is_half():
    """P(1) = s sur les deux aretes : la lecture deployee rend s >= seuil
    (la decision classique), la lecture majoritaire rend s > 0,5."""
    rng = np.random.default_rng(3)
    for dim in (2, 3, 4):
        for _ in range(20):
            s = rng.uniform(0, 1, dim * dim)
            thr = float(rng.uniform(0.05, 0.95))
            marg = np.concatenate([s, s])
            dh, dv = deployed_readout(marg, dim, thr)
            np.testing.assert_array_equal((dh | dv).ravel(), s >= thr)
            mh, mv = majority_readout(marg, dim)
            np.testing.assert_array_equal((mh | mv).ravel(), s > 0.5)


def test_readouts_refuse_malformed_inputs():
    with pytest.raises(ValueError):
        majority_readout(np.zeros(7), 2)
    with pytest.raises(ValueError):
        deployed_readout(np.zeros(8), 2, None)


def test_run_qaoa_routes_the_readout(monkeypatch):
    """Le choix de lecture atteint la decision ; le defaut reste historique."""
    assert inspect.signature(run_qaoa_on_snapshot).parameters[
        "readout"].default == "majority"
    marg = [0.90, 0.20, 0.55, 0.05, 0.20, 0.20, 0.55, 0.05]

    class _H:
        coeffs = np.array([1.0])

    monkeypatch.setattr(qaoa_inputs, "mapping",
                        lambda *a, **k: (object(), _H()))
    monkeypatch.setattr(qaoa_inputs, "execute",
                        lambda *a, **k: ({}, np.zeros(2)))
    monkeypatch.setattr(qaoa_inputs, "postprocess", lambda *a, **k: marg)

    _, dh, dv, _, _ = run_qaoa_on_snapshot({}, {}, 2, reps=1)
    assert (dh | dv).ravel().tolist() == [True, False, True, False]
    _, dh, dv, _, _ = run_qaoa_on_snapshot({}, {}, 2, reps=1,
                                           readout="deployed",
                                           threshold_amr=0.15)
    assert (dh | dv).ravel().tolist() == [True, True, True, False]
    with pytest.raises(ValueError):
        run_qaoa_on_snapshot({}, {}, 2, reps=1, readout="deployed")
    with pytest.raises(ValueError):
        run_qaoa_on_snapshot({}, {}, 2, reps=1, readout="argmax")


def test_deployed_readout_is_the_deployed_solver_decision(monkeypatch):
    """`deployed_readout` rend EXACTEMENT les sous-patches que
    `Simulation.refinement._run_level` raffine pour les memes marginales.

    Le VQA est remplace par des marginales imposees ; toutes les moyennes
    de cellule evitent la bande de sondage de bord [seuil/2, seuil[, ou le
    solveur deploye peut raffiner sur un autre critere."""
    from types import SimpleNamespace

    import Simulation.refinement as refinement
    from Simulation.grid import PeriodicGrid
    from Simulation.HamiltParams_v2 import PhysicalMapperV2
    from Simulation.PhysToAngle import AngleMapper
    from Simulation.solver import MHDSolver

    N, dim, thr = 96, 3, 0.55
    cur, prev = _snapshot(si=10), _snapshot(si=9)

    def state(fields):
        sim = MHDSolver(PeriodicGrid(N), dt=1e-4, Re=400, Rm=400)
        sim.vx, sim.vy, sim.Bx, sim.By = fields
        return sim, sim.get_fluxes()

    sim, phys = state(cur)
    _, phys_prev = state(prev)
    angle = AngleMapper()
    phi = angle.compute_stress_flux(phys)
    phi_prev = angle.compute_stress_flux(phys_prev)
    avg_dev = 0.5 * (
        np.mean(np.abs(phi["phi_horizontal"] - phi_prev["phi_horizontal"]))
        + np.mean(np.abs(phi["phi_vertical"] - phi_prev["phi_vertical"])))
    score = AngleMapper.classical_score(phys)

    # moyennes de cellule : >= 0,55 ou < 0,275, jamais dans la bande ; la
    # derniere cellule (0,52 / 0,02) separe les deux lectures
    h = np.array([0.95, 0.10, 0.60, 0.20, 0.90, 0.05, 0.70, 0.30, 0.52])
    v = np.array([0.40, 0.20, 0.60, 0.10, 0.30, 0.20, 0.50, 0.10, 0.02])
    marg = np.concatenate([h, v])
    means = 0.5 * (h + v)
    assert not np.any((means >= thr / 2) & (means < thr))

    monkeypatch.setattr(refinement, "call_vqa_shell",
                        lambda *a, **k: (marg.copy(), np.zeros(2)))
    active = []
    nxt = refinement._run_level(
        [(0, N, 0, N)], 0,
        phi["phi_horizontal"], phi["phi_vertical"],
        phi_prev["phi_horizontal"], phi_prev["phi_vertical"],
        score, phys, angle, SimpleNamespace(AdvAnomaliesEnable=True),
        avg_dev, 1.0, dim, 2, 4, thr, active,
        HamiltMapper=PhysicalMapperV2(dx=2 * np.pi / N), sim=sim)
    step = N // dim
    refined = {(b[0] // step, b[2] // step) for b in nxt}
    dh, dv = deployed_readout(marg, dim, thr)
    expected = {(i, j) for i in range(dim) for j in range(dim)
                if (dh | dv)[i, j]}
    assert expected, "le cas teste doit raffiner au moins une cellule"
    assert refined == expected
    mh, mv = majority_readout(marg, dim)
    assert {(i, j) for i in range(dim) for j in range(dim)
            if (mh | mv)[i, j]} != expected, (
        "le cas doit separer les deux lectures, sinon il ne prouve rien")


# ── le recentrage du biais ─────────────────────────────────────────────

@pytest.mark.parametrize("use_v2", [True, False])
def test_prepare_qaoa_inputs_recentres_the_bias(use_v2):
    """Le signe du biais suit (s - seuil) pour le seuil demande ; theta ne
    depend pas du seuil."""
    a = _snapshot()
    ref, _hp0, s0 = prepare_qaoa_inputs(*a, 96, 3, 400, use_v2=use_v2)
    for thr in (None, 0.30, 0.55):
        data_in, hp, s = prepare_qaoa_inputs(*a, 96, 3, 400, use_v2=use_v2,
                                             threshold_amr=thr)
        eff = (V2_THRESHOLD if use_v2 else qaoa_inputs.TRAINED_THRESHOLD) \
            if thr is None else thr
        H = np.asarray(hp["H_edges"])
        sc = np.asarray(s)
        assert H.shape == (2, 3, 3)
        for fam in range(2):
            np.testing.assert_array_equal(np.sign(H[fam]),
                                          np.sign(sc - eff))
        np.testing.assert_array_equal(np.asarray(data_in["theta_h"]),
                                      np.asarray(ref["theta_h"]))
        np.testing.assert_array_equal(sc, np.asarray(s0))


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_a_threshold_outside_the_score_range_is_refused(bad):
    with pytest.raises(ValueError):
        prepare_qaoa_inputs(*_snapshot(), 96, 3, 400, use_v2=True,
                            threshold_amr=bad)


def test_loso_thresholds_reproduce_the_confirmatory_classical_arm():
    """Meme selection, meme score, meme recherche : les seuils sont ceux du
    bras classique de la replication confirmatoire, au bit pres."""
    from bias_threshold import loso_f1_thresholds
    conf = np.load(os.path.join(
        RESULTS_DIR, "h2b_v2_hamiltonian_vs_gbt_loso_N96_dim3_n10_multire.npz"))
    expected = {str(h): float(t)
                for h, t in zip(conf["held"], conf["thr_classical"])}
    got = loso_f1_thresholds(RESULTS_DIR)
    assert {k: v[0] for k, v in got.items()} == expected
    assert all(0.51 - 1e-9 <= t <= 0.60 + 1e-9 for t in expected.values())


def test_the_calibration_never_sees_the_held_out_scenario(monkeypatch):
    """Changer les labels du scenario tenu ne change pas SON seuil."""
    import bias_threshold
    real = bias_threshold.block_scores_and_labels
    base = bias_threshold.loso_f1_thresholds(RESULTS_DIR)

    def flipped(scenario, *a, **k):
        s, y, idx = real(scenario, *a, **k)
        if scenario == "mhd_rotor":
            y = ~y
        return s, y, idx

    monkeypatch.setattr(bias_threshold, "block_scores_and_labels", flipped)
    moved = bias_threshold.loso_f1_thresholds(RESULTS_DIR)
    assert moved["mhd_rotor"] == base["mhd_rotor"]
    assert any(moved[k] != base[k] for k in base if k != "mhd_rotor"), (
        "le controle doit etre capable d'echouer : les autres plis lisent "
        "bien les labels du rotor")


def test_a_missing_calibration_input_is_an_error(tmp_path):
    from bias_threshold import loso_f1_thresholds
    with pytest.raises(FileNotFoundError):
        loso_f1_thresholds(str(tmp_path))


# ── le panel H0 : options et noms d'artefacts ──────────────────────────

def _name(panel, **kw):
    class A:
        pass
    a = A()
    base = dict(N=96, dim=3, legacy_curl=False, zero_psi=False,
                no_exact=False, backend="state_vector", scale_kopt=False,
                mapper="v2",
                scenario=["harris_tearing", "kelvin_helmholtz", "mhd_rotor",
                          "orszag_tang"])
    base.update(kw)
    for k, v in base.items():
        setattr(a, k, v)
    return os.path.basename(panel._output_path(a))


def test_options_name_their_artifacts(panel):
    stem = ("h0_optimiser_equivalence_N96_dim3_harris_tearing-"
            "kelvin_helmholtz-mhd_rotor-orszag_tang")
    assert _name(panel) == stem + ".npz"
    assert _name(panel, bias_threshold="deployed",
                 readout="majority") == stem + ".npz"
    assert _name(panel, bias_threshold="loso-f1") == stem + "_thrlosof1.npz"
    assert _name(panel, readout="deployed") == stem + "_readdeployed.npz"
    assert _name(panel, bias_threshold="loso-f1", readout="deployed",
                 mapper="v1") == stem + "_thrlosof1_readdeployed_v1.npz"
    combos = {_name(panel, bias_threshold=b, readout=r, mapper=m)
              for b in ("deployed", "loso-f1")
              for r in ("majority", "deployed") for m in ("v1", "v2")}
    assert len(combos) == 8
    # un plan d'experience qui rejoue une configuration publiee ne doit pas
    # ecraser l'artefact publie
    assert _name(panel, mapper="v1", tag="recentrage") == \
        stem + "_v1_recentrage.npz"
    assert _name(panel, mapper="v1", tag=None) == stem + "_v1.npz"


def test_the_cli_exposes_both_options():
    out = subprocess.run([sys.executable, _PANEL, "--help"],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "--bias-threshold {deployed,loso-f1}" in out.stdout
    assert "--readout {majority,deployed}" in out.stdout
    assert "--tag TAG" in out.stdout


# ── les nombres du constat ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def numbers(diag):
    return diag.diagnose(RESULTS_DIR, V2_THRESHOLD, calibration=False)


def test_the_bias_threshold_marks_most_cells_and_the_readouts_disagree(numbers):
    assert numbers["n_instances"] == 12 and numbers["n_cells"] == 108
    assert numbers["n_refine_classical"] == 88
    assert numbers["n_refine_majority_unoptimised"] == 54
    assert numbers["n_cells_classical_ne_majority"] == 34
    assert numbers["cells_classical_ne_majority_by_scenario"] == {
        "harris_tearing": 0, "kelvin_helmholtz": 9, "mhd_rotor": 11,
        "orszag_tang": 14}
    assert all(r["deployed_equals_classical"] for r in numbers["rows"])


def test_mean_f1_of_the_three_masks(numbers):
    f = numbers["mean_f1"]
    assert f["classical"] == pytest.approx(0.468, abs=5e-4)
    assert f["majority_unoptimised"] == pytest.approx(0.507, abs=5e-4)
    assert f["refine_all"] == pytest.approx(0.386, abs=5e-4)
    assert f["deployed_unoptimised"] == f["classical"]


def test_qaoa_f1_is_the_majority_readout_of_the_unoptimised_circuit(numbers):
    assert len(numbers["informative_instances"]) == 7
    v1 = numbers["panels"]["v1_28aug"]
    assert v1["classical_equal_recomputed"] == 12, (
        "le recalcul doit reproduire la decision classique de l'artefact, "
        "sinon les comparaisons suivantes ne portent pas sur les memes "
        "masques")
    for solver, q in v1["qaoa"].items():
        assert q["equal_majority_unoptimised"] == 11, solver
        assert q["equal_on_informative"] == 6, solver
    v2 = numbers["panels"]["v2_16aug"]
    assert [v2["qaoa"][s]["equal_majority_unoptimised"]
            for s in ("qaoa_p1", "qaoa_p2", "qaoa_p3")] == [9, 8, 8]


def test_the_ground_state_refines_nearly_everything(numbers):
    assert numbers["panels"]["v2_16aug"]["exhaustive_equal_refine_all"] == 12
    assert numbers["panels"]["v1_28aug"]["exhaustive_equal_refine_all"] == 9


def test_the_f1_optimal_threshold_is_far_above_the_bias_threshold(numbers):
    thr = numbers["confirmatory_thresholds"]
    assert set(thr) == {"harris_tearing", "kelvin_helmholtz", "mhd_rotor",
                        "orszag_tang"}
    assert min(thr.values()) > 3 * V2_THRESHOLD
    # 0,5099999... sur Orszag-Tang : bornes au milliardieme pres
    assert all(0.51 - 1e-9 <= t <= 0.60 + 1e-9 for t in thr.values())
