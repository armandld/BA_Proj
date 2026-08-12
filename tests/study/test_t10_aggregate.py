"""Tests V3 Task 10 : extracteurs de la table maitresse.

Les extracteurs prennent des dicts (charges depuis les .npz) : on les
teste avec des donnees synthetiques + le comportement MISSING/DIFF.
La reproduction des valeurs reelles est le critere d'acceptation
(execution de run_study_v3.sh sur les vraies donnees, --strict).
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

from aggregate_v3 import (
    load_npz,
    make_row,
    rows_t1,
    rows_t1b,
    rows_t4,
    rows_t9,
    status_of,
    to_csv,
    to_markdown,
)


# --------------------------- statuts ----------------------------------

def test_status_of_tolerance():
    assert status_of(0.4341, 0.434) == "OK"      # dans la tolerance
    assert status_of(0.440, 0.434) == "DIFF"     # hors tolerance
    assert status_of(None, 0.434) == "MISSING"
    assert status_of(float("nan"), 0.434) == "MISSING"
    assert status_of(1.234, None) == "OK"        # informatif sans ref


def test_make_row_fields():
    r = make_row("tX", "m", 0.5, 0.5)
    assert r == dict(task="tX", metric="m", value=0.5, ref=0.5,
                     status="OK")


# ------------------------- extracteurs --------------------------------

def test_rows_t1_mean_and_match():
    d = dict(f1_classical=np.array([0.264, 0.400, 0.400, 0.672]),
             f1_b5=np.array([0.227, 0.340, 0.290, 0.165]),
             f1_full9=np.array([0.327, 0.000, 0.344, 0.084]))
    rows = rows_t1(d)
    assert [r["status"] for r in rows] == ["OK", "OK", "OK"]
    assert rows[2]["value"] == pytest.approx(0.18875, abs=1e-6)


def test_rows_t1_missing():
    rows = rows_t1(None)
    assert len(rows) == 3
    assert all(r["status"] == "MISSING" for r in rows)


def test_rows_t1b_uses_k_values_order():
    d = dict(k_values=np.array([0, 1, 2, 3]),
             f1_loso_mean=np.array([0.189, 0.215, 0.140, 0.140]),
             f1_blocked=np.array([0.581, 0.733, 0.816, 0.816]))
    rows = rows_t1b(d)
    assert len(rows) == 8
    assert all(r["status"] == "OK" for r in rows)
    # une valeur deviee est signalee
    d2 = dict(d, f1_loso_mean=np.array([0.189, 0.300, 0.140, 0.140]))
    rows2 = rows_t1b(d2)
    assert rows2[1]["status"] == "DIFF"


def test_rows_t4_lookup_by_method_name_and_gap():
    names = ["B1 classical (block_avg)", "B2 classical (block_max)",
             "B4 gbt-9 (max)", "B4 gbt-9 (avg)"]
    d = dict(
        names=np.array(names),
        random_f1=np.array([0.492, 0.475, 0.980, 0.975]),
        blocked_f1=np.array([0.579, 0.517, 0.581, 0.738]),
        blocked_rho=np.array([0.767, 0.365, 0.640, 0.694]),
        blocked_auc=np.array([0.595, 0.582, 0.580, 0.587]),
    )
    rows = rows_t4(d)
    assert all(r["status"] == "OK" for r in rows)
    gap = next(r for r in rows if "leakage" in r["metric"])
    assert gap["value"] == pytest.approx(0.980 - 0.581)


def test_rows_t9_grouped_means():
    d = dict(
        mapper=np.array(["v1", "v1", "v2", "v2", "v1", "v2"]),
        dim=np.array([2, 2, 2, 2, 4, 4]),
        frac=np.array([0.008, 0.008, 0.034, 0.034, 0.221, 0.000]),
    )
    rows = rows_t9(d)
    by = {r["metric"]: r for r in rows}
    assert by["mean frac v1 dim=2"]["value"] == pytest.approx(0.008)
    assert by["mean frac v2 dim=4"]["value"] == pytest.approx(0.000)
    assert all(r["status"] == "OK" for r in rows)


# --------------------------- sorties ----------------------------------

def test_outputs_render_and_missing_marker():
    rows = [make_row("tX", "a", 0.5, 0.5), make_row("tX", "b", None)]
    md = to_markdown(rows, "abcdef123456")
    assert "| tX | a | 0.500 | 0.500 | OK |" in md
    assert "| tX | b | — | — | MISSING |" in md
    csv = to_csv(rows, "abcdef123456", {"N": 256})
    assert csv.splitlines()[0].startswith("# git_hash=")
    assert "tX,a,0.500000,0.500000,OK" in csv
    assert "tX,b,,,MISSING" in csv


def test_load_npz_roundtrip_and_none(tmp_path):
    path = str(tmp_path / "x.npz")
    assert load_npz(path) is None
    np.savez(path, a=np.array([1.0, 2.0]))
    d = load_npz(path)
    np.testing.assert_array_equal(d["a"], [1.0, 2.0])
