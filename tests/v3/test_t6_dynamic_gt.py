"""Tests V3 Task 6 : verite terrain dynamique d_i.

Helpers purs + test d'integration sur un mini-solveur reel (N=16,
dim=2, delta_t=0.02) : le solveur V1 est importe, jamais re-implemente.
Le pilote chiffre (N=128, wall-clock, Spearman global) est valide par
l'execution sur les vraies donnees.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "study", "v3"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "study"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

from t6_dynamic_gt import (
    coarsen_one_patch,
    coarsen_patch_window,
    downsample_fields,
    dynamic_gt_snapshot,
    reference_evolution,
    rel_l2_diff,
    select_snapshots,
)


# --------------------------- helpers purs -----------------------------

def test_coarsen_patch_window_mean_and_no_mutation():
    f = np.arange(16, dtype=float).reshape(4, 4)
    f0 = f.copy()
    out = coarsen_patch_window(f, 0, 2, 2, 4)
    np.testing.assert_array_equal(f, f0)              # pas de mutation
    assert np.all(out[0:2, 2:4] == f[0:2, 2:4].mean())
    out2 = out.copy(); out2[0:2, 2:4] = f[0:2, 2:4]
    np.testing.assert_array_equal(out2, f)            # exterieur intact


def test_coarsen_patch_window_matches_phase2_coarsen_field():
    # replique de coarsen_field (phase 2) : moyenne par blocs +
    # prolongation constante ; restreinte a une fenetre = notre helper
    rng = np.random.default_rng(0)
    f = rng.normal(size=(8, 8))
    ps = 4
    coarse = f.reshape(2, ps, 2, ps).mean(axis=(1, 3))
    prolonged = np.repeat(np.repeat(coarse, ps, axis=0), ps, axis=1)
    out = coarsen_patch_window(f, 0, 4, 4, 8)         # patch (0, 1)
    np.testing.assert_allclose(out[0:4, 4:8], prolonged[0:4, 4:8])
    np.testing.assert_array_equal(out[4:8, :], f[4:8, :])


def test_rel_l2_diff_zero_hand_value_and_scale_invariance():
    z = np.zeros((4, 4))
    ones = np.ones((4, 4))
    ref = (ones, z, z, z)
    assert rel_l2_diff(ref, ref) == 0.0
    var = (ones * 2.0, z, z, z)   # diff^2 = 1 partout, rms(ref) = 1
    assert rel_l2_diff(ref, var) == pytest.approx(1.0)
    ref3 = tuple(3.0 * f for f in ref)
    var3 = tuple(3.0 * f for f in var)
    assert rel_l2_diff(ref3, var3) == pytest.approx(1.0)


def test_downsample_fields_block_mean():
    arr = np.arange(16, dtype=float).reshape(1, 4, 4)
    out = downsample_fields(arr, 2)
    np.testing.assert_allclose(out[0], [[2.5, 4.5], [10.5, 12.5]])


def test_select_snapshots_excludes_initial_condition():
    sel = select_snapshots(120, 2)
    assert 0 not in sel
    assert len(sel) == 2 and sel == sorted(sel)
    assert all(0 < s <= 119 for s in sel)
    assert select_snapshots(120, 10)[-1] == 119


# ----------------------- integration mini-solveur ---------------------

def _toy_fields(N=16):
    """Champs nuls sauf le patch (1, 1) (vx structure). Les patchs
    (0,0), (0,1), (1,0) sont exactement nuls champ par champ : leur
    coarsening (moyenne = 0.0 bit-exact) est l'identite."""
    vx = np.zeros((N, N))
    h = N // 2
    x = np.linspace(0, 2 * np.pi, h, endpoint=False)
    vx[h:, h:] = 0.2 * np.outer(np.sin(2 * x), np.cos(3 * x))
    vy = np.zeros((N, N))
    Bx = np.zeros((N, N))
    By = np.zeros((N, N))
    return vx, vy, Bx, By


def test_reference_dts_sum_to_delta_t():
    fields = _toy_fields()
    _, dts = reference_evolution(fields, 16, 400, delta_t=0.02, cfl=0.4)
    assert len(dts) >= 1
    assert sum(dts) == pytest.approx(0.02)


def test_dynamic_gt_constant_patch_is_exactly_zero():
    fields = _toy_fields()
    d, n_steps = dynamic_gt_snapshot(fields, 16, 2, 400,
                                     delta_t=0.02, cfl=0.4)
    assert d.shape == (2, 2) and n_steps >= 1
    assert np.all(d >= 0.0)
    # patches a champs constants : coarsening = identite -> d = 0 exact
    assert d[0, 0] == 0.0 and d[0, 1] == 0.0 and d[1, 0] == 0.0
    # le patch structure porte toute l'erreur dynamique
    assert d[1, 1] > 0.0


def test_dynamic_gt_consistent_with_static_e():
    # argmax(d) doit coincider avec argmax(e_i) de phase 2 sur ce cas
    from phase2_hard_patches import patch_l2_errors
    fields = _toy_fields()
    d, _ = dynamic_gt_snapshot(fields, 16, 2, 400,
                               delta_t=0.02, cfl=0.4)
    e = patch_l2_errors(*fields, 2)
    assert np.unravel_index(np.argmax(d), d.shape) == (1, 1)
    assert np.unravel_index(np.argmax(e), e.shape) == (1, 1)


def test_coarsen_one_patch_only_touches_one_window():
    fields = _toy_fields()
    var = coarsen_one_patch(fields, 1, 1, 8)
    for orig, mod in zip(fields, var):
        np.testing.assert_array_equal(mod[:8, :], orig[:8, :])
        np.testing.assert_array_equal(mod[8:, :8], orig[8:, :8])
        assert np.all(mod[8:, 8:] == orig[8:, 8:].mean())
