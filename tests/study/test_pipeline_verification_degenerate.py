"""D-40 : l'energie hamiltonienne v1 peut etre identiquement nulle sur
toute une simulation (aucun coefficient ne franchit son seuil critique).
AUC/F1 tombent alors a leur valeur de hasard par construction (0.5/0.0),
indiscernable a la lecture d'une vraie mesure au hasard. `analyze()` doit
le signaler (`degenerate_E`), et `split_degenerate` doit l'exclure des
moyennes agregees plutot que de laisser une ligne degeneree tirer le
verdict PASS/FAIL vers le bas en silence.
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

import pipeline_verification as pv


def _write_pair(results_dir, scenario, Re, N, dim, E, l2_errors, is_hard,
                 classical_scores):
    coef_path = os.path.join(
        results_dir, f"coefficients_{scenario}_Re{Re}_N{N}_dim{dim}.npz")
    np.savez_compressed(coef_path, **{"s0.023_E": E})

    patch_path = os.path.join(
        results_dir, f"patches_{scenario}_Re{Re}_N{N}_dim{dim}.npz")
    np.savez_compressed(
        patch_path,
        l2_errors=l2_errors,
        is_hard=is_hard,
        classical_scores=classical_scores,
        l2_threshold=0.1,
        t=np.arange(l2_errors.shape[0], dtype=float),
        scenario=scenario, Re=Re, N=N, n_patches=dim,
    )


def _synthetic_patches(rng, n_snaps, dim):
    l2 = rng.random((n_snaps, dim, dim))
    is_hard = l2 >= np.percentile(l2, 75)
    classical = rng.random((n_snaps, dim, dim))
    return l2, is_hard, classical


def test_analyze_flags_constant_E_as_degenerate(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "RESULTS_DIR", str(tmp_path))
    rng = np.random.default_rng(0)
    n_snaps, dim = 6, 4
    l2, is_hard, classical = _synthetic_patches(rng, n_snaps, dim)
    E_zero = np.zeros((n_snaps, dim, dim))
    _write_pair(tmp_path, "toy_scenario", 400, 256, dim,
                E_zero, l2, is_hard, classical)

    result = pv.analyze("toy_scenario", 400, dim, 256, use_v2=False)

    assert result["degenerate_E"] is True
    assert result["auc_E"] == pytest.approx(0.5)
    assert result["f1_E"] == pytest.approx(0.0)


def test_analyze_does_not_flag_varying_E(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "RESULTS_DIR", str(tmp_path))
    rng = np.random.default_rng(1)
    n_snaps, dim = 6, 4
    l2, is_hard, classical = _synthetic_patches(rng, n_snaps, dim)
    # E correlated with l2 so the row also carries real signal, not just
    # non-degeneracy.
    E_varying = l2 + 0.01 * rng.random((n_snaps, dim, dim))
    _write_pair(tmp_path, "toy_scenario", 400, 256, dim,
                E_varying, l2, is_hard, classical)

    result = pv.analyze("toy_scenario", 400, dim, 256, use_v2=False)

    assert result["degenerate_E"] is False


def test_split_degenerate_excludes_only_flagged_rows():
    rows = [
        {"scenario": "a", "degenerate_E": False, "auc_E": 0.9},
        {"scenario": "b", "degenerate_E": True, "auc_E": 0.5},
        {"scenario": "c", "degenerate_E": False, "auc_E": 0.7},
    ]
    clean, degenerate = pv.split_degenerate(rows)
    assert [r["scenario"] for r in clean] == ["a", "c"]
    assert [r["scenario"] for r in degenerate] == ["b"]


def test_aggregate_verdict_unpolluted_by_degenerate_row():
    """
    Pin the repo's D-40 measurement: on the four canonical scenarios at
    Re=400/N=256/dim=4 (real results/ artifacts), harris_tearing and
    kelvin_helmholtz have a constant v1 Hamiltonian energy (E == 0
    everywhere). Averaging them in with mhd_rotor/orszag_tang used to
    pull the mean AUC(E) down to 0.687 and F1(E) to 0.364 (below the
    classical F1 of 0.603 -- a WARN). Excluding them raises AUC(E) to
    0.874 and F1(E) to 0.729, now above the classical baseline -- a
    PASS. This test locks that the aggregate uses only clean rows, not
    the specific PASS/WARN wording.
    """
    rows = [
        {"scenario": "harris_tearing", "degenerate_E": True,
         "auc_E": 0.500, "f1_E": 0.000, "auc_c": 0.780, "f1_c": 0.629},
        {"scenario": "kelvin_helmholtz", "degenerate_E": True,
         "auc_E": 0.500, "f1_E": 0.000, "auc_c": 0.585, "f1_c": 0.474},
        {"scenario": "mhd_rotor", "degenerate_E": False,
         "auc_E": 0.985, "f1_E": 0.938, "auc_c": 0.998, "f1_c": 0.949},
        {"scenario": "orszag_tang", "degenerate_E": False,
         "auc_E": 0.762, "f1_E": 0.519, "auc_c": 0.337, "f1_c": 0.360},
    ]
    clean, degenerate = pv.split_degenerate(rows)
    assert len(degenerate) == 2

    mean_auc_E = np.mean([r["auc_E"] for r in clean])
    mean_f1_E = np.mean([r["f1_E"] for r in clean])
    mean_f1_c = np.mean([r["f1_c"] for r in clean])

    assert mean_auc_E == pytest.approx(0.8735, abs=1e-3)
    assert mean_f1_E == pytest.approx(0.7285, abs=1e-3)
    # before the fix this comparison read F1(E)=0.364 < F1(c)=0.603 (WARN);
    # on the clean rows F1(E) now exceeds the classical baseline (PASS).
    assert mean_f1_E > mean_f1_c
