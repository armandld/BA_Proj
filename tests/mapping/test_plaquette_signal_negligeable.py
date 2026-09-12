"""Round-off handling for the dimensionless Hamiltonian mapper."""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from Simulation.HamiltParams_v2 import PhysicalMapperV2  # noqa: E402
from Simulation.grid import curl_z  # noqa: E402


N = 16
DX = 2 * np.pi / N
X, Y = np.meshgrid(
    np.linspace(0, 2 * np.pi, N, endpoint=False),
    np.linspace(0, 2 * np.pi, N, endpoint=False),
    indexing="ij",
)
ZERO = np.zeros((N, N))
SCORE = np.full((N, N), 0.5)


def _coefficients(vx=ZERO, vy=ZERO, Bx=ZERO, By=ZERO, xpoint=False):
    return PhysicalMapperV2(dx=DX, norm="max").compute_coefficients(
        None,
        SCORE,
        {"vx": vx, "vy": vy, "Bx": Bx, "By": By, "Jz": ZERO},
        0.15,
        advanced_anomalies_enabled=xpoint,
    )


@pytest.mark.parametrize("family", ["velocity", "magnetic"])
def test_roundoff_around_a_large_offset_is_not_promoted(family):
    dust = 1.0 + 1e-14 * np.sin(Y)
    kwargs = {"vx": dust} if family == "velocity" else {"Bx": dust}

    coeffs = _coefficients(**kwargs)

    assert np.max(np.abs(coeffs["C_edges"][0])) == 0.0
    assert np.max(np.abs(coeffs["C_edges"][1])) == 0.0
    assert np.max(np.abs(coeffs["K_plaquettes"])) == 0.0


@pytest.mark.parametrize("amplitude", [1.0, 1e-14])
def test_a_small_but_resolved_structure_keeps_full_relative_weight(amplitude):
    coeffs = _coefficients(vx=amplitude * np.sin(Y))

    assert np.max(np.abs(coeffs["C_edges"][0])) == pytest.approx(2.0)
    assert np.max(np.abs(coeffs["K_plaquettes"])) == pytest.approx(1.0)


def test_xpoint_roundoff_is_removed_but_a_resolved_perturbation_survives():
    def xpoint(amplitude):
        return _coefficients(
            Bx=1.0 + amplitude * np.sin(X),
            By=1.0 - amplitude * np.sin(Y),
            xpoint=True,
        )["K_xpoint"]

    assert np.max(np.abs(xpoint(1e-14))) == 0.0
    assert np.max(np.abs(xpoint(1e-12))) == pytest.approx(1.0)


def test_bias_scale_uses_the_effective_plaquette_operator():
    score = np.linspace(0.0, 1.0, N * N).reshape(N, N)
    mapper = PhysicalMapperV2(dx=DX, norm="max", c_bias=0.3)
    coeffs = mapper.compute_coefficients(
        None,
        score,
        {
            "vx": ZERO,
            "vy": ZERO,
            "Bx": np.sin(X),
            "By": -np.sin(Y),
            "Jz": ZERO,
        },
        0.4,
        advanced_anomalies_enabled=True,
    )
    effective_k = coeffs["K_plaquettes"] + coeffs["K_xpoint"]
    coupling_scale = max(
        np.max(np.abs(coeffs["C_edges"][0])),
        np.max(np.abs(coeffs["C_edges"][1])),
        np.max(np.abs(effective_k)),
    )
    expected_h_peak = 0.3 * coupling_scale * np.max(np.abs(score - 0.4))
    assert np.max(np.abs(coeffs["H_edges"][0])) == pytest.approx(
        expected_h_peak)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"dx": 0.0}, "dx"),
        ({"c_bias": -1.0}, "c_bias"),
        ({"w_zzzz": np.nan}, "w_zzzz"),
    ],
)
def test_nonphysical_mapper_parameters_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PhysicalMapperV2(**kwargs)


def test_legacy_mode_keeps_its_frozen_additive_guard():
    vx = 1e-9 * np.sin(Y)
    coeffs = PhysicalMapperV2(dx=DX, norm="legacy").compute_coefficients(
        None,
        SCORE,
        {"vx": vx, "vy": ZERO, "Bx": ZERO, "By": ZERO, "Jz": ZERO},
        0.15,
    )
    omega_peak = float(np.max(np.abs(curl_z(vx, ZERO))))
    expected_peak = omega_peak / (omega_peak + PhysicalMapperV2.EPS)
    assert np.max(np.abs(coeffs["K_plaquettes"])) == pytest.approx(expected_peak)
