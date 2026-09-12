"""The DNS gate measures the divergence guaranteed by the solver."""

import os
import sys

import numpy as np
import pytest


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _path in (os.path.join(_REPO_ROOT, "src"),
              os.path.join(_REPO_ROOT, "study", "pipeline")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dns_validation import analyse_one, validate_one


@pytest.fixture(scope="module")
def head_trajectory(tmp_path_factory):
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver

    N = 64
    sim = MHDSolver(PeriodicGrid(N), dt=1e-3, Re=400, Rm=400)
    sim.init_harris_tearing()
    fields = {name: [] for name in ("vx", "vy", "Bx", "By")}
    times = []
    for index in range(6):
        for _ in range(4):
            sim.dt = sim.adapt_dt(cfl_target=0.4)
            sim.step_full(record_stats=False)
        for name in fields:
            fields[name].append(getattr(sim, name).astype(np.float32))
        times.append(float(index))

    path = str(tmp_path_factory.mktemp("dns_gate") / "trajectory.npz")
    np.savez_compressed(
        path, **{name: np.asarray(value) for name, value in fields.items()},
        t=np.asarray(times), step=np.arange(6, dtype=np.int32),
        meta_scenario="harris_tearing", meta_Re=400, meta_N=N,
        meta_diverged=False)
    return path


def test_matched_divergence_accepts_solver_trajectory(head_trajectory):
    result = analyse_one(head_trajectory)
    assert result["div_rel_max"] <= 1e-3


def test_validation_log_records_the_gate_value(head_trajectory):
    failures, log = validate_one(head_trajectory, "harris_tearing")
    assert not [failure for failure in failures if "divB" in failure]
    assert any(entry.startswith("div=") for entry in log)
