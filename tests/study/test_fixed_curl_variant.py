"""The deployed axis convention is the default; legacy is an ablation."""

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "study" / "common"
H0 = ROOT / "study" / "h0_selection"
for path in (ROOT / "src", COMMON, H0):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from h0_optimiser_equivalence import solver_panel  # noqa: E402
from qaoa_inputs import prepare_qaoa_inputs  # noqa: E402


def _fields(N=16):
    x = np.arange(N)[:, None] * 2 * np.pi / N
    y = np.arange(N)[None, :] * 2 * np.pi / N
    return (
        -np.sin(y) + 0 * x,
        np.sin(x) + 0 * y,
        0.4 * np.cos(y) + 0 * x,
        0.6 * np.cos(x) + 0 * y,
    )


def test_current_axis_convention_is_the_function_default():
    assert inspect.signature(prepare_qaoa_inputs).parameters[
        "fixed_curl"].default is True
    assert inspect.signature(solver_panel).parameters[
        "fixed_curl"].default is True


def test_legacy_ablation_changes_the_encoded_problem():
    fields = _fields()
    current = prepare_qaoa_inputs(
        *fields, 16, 2, 400, use_v2=False, fixed_curl=True)
    legacy = prepare_qaoa_inputs(
        *fields, 16, 2, 400, use_v2=False, fixed_curl=False)
    score_gap = np.max(np.abs(current[2] - legacy[2]))
    coefficient_gap = max(
        np.max(np.abs(np.asarray(current[1][key])
                      - np.asarray(legacy[1][key])))
        for key in current[1]
        if isinstance(current[1][key], np.ndarray)
    )
    assert score_gap > 1e-6
    assert coefficient_gap > 1e-6


def test_cli_names_only_the_legacy_ablation():
    result = subprocess.run(
        [sys.executable, str(H0 / "h0_optimiser_equivalence.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    assert "--legacy-curl" in result.stdout
    assert "--fixed-curl" not in result.stdout
