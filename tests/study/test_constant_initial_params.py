"""Contrats de l'initialisation déterministe des paramètres QAOA."""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (os.path.join(_REPO_ROOT, "src"),
           os.path.join(_REPO_ROOT, "study", "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.mark.parametrize("reps", [1, 2, 3, 4])
def test_constant_schedule_values(reps):
    from qaoa_inputs import constant_initial_params

    out = constant_initial_params(reps)
    assert out.shape == (2 * reps,)
    np.testing.assert_array_equal(out[:reps], np.full(reps, 0.05))
    np.testing.assert_allclose(
        out[reps:], 0.15 / np.arange(1, reps + 1), rtol=0, atol=0)


@pytest.mark.parametrize("reps", [0, -1])
def test_constant_schedule_rejects_invalid_depth(reps):
    from qaoa_inputs import constant_initial_params

    with pytest.raises(ValueError, match="reps"):
        constant_initial_params(reps)


def test_api_does_not_claim_a_classical_warm_start():
    import inspect

    from qaoa_inputs import constant_initial_params

    assert tuple(inspect.signature(constant_initial_params).parameters) == ("reps",)
    assert "classical_warm_start_params" not in vars(
        sys.modules[constant_initial_params.__module__])
