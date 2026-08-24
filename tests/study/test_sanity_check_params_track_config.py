"""Le sanity check consomme directement les paramètres déployés."""

import importlib
import os
import sys

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (os.path.join(_REPO_ROOT, "src"),
           os.path.join(_REPO_ROOT, "study", "pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402
import sanity_check  # noqa: E402


_MAPPER = {
    "sigma": "TRAINED_SIGMA",
    "beta_curl": "TRAINED_BETA_CURL",
    "beta_xpoint": "TRAINED_BETA_XPOINT",
    "w_z_frac": "TRAINED_W_Z_FRAC",
    "gamma_hydro": "TRAINED_GAMMA_HYDRO",
    "gamma_mag": "TRAINED_GAMMA_MAG",
    "kappa": "TRAINED_KAPPA",
    "relative_percentile": "TRAINED_RELATIVE_PERCENTILE",
}


def test_mapper_parameters_come_from_config():
    assert set(sanity_check.V1_PARAMS) == set(_MAPPER)
    for key, config_name in _MAPPER.items():
        assert sanity_check.V1_PARAMS[key] == getattr(config, config_name)


def test_thresholds_come_from_config():
    assert sanity_check.V1_THRESHOLD == config.TRAINED_THRESHOLD
    assert sanity_check.V2_THRESHOLD == config.V2_THRESHOLD


def test_a_config_update_reaches_the_sanity_check(monkeypatch):
    value = config.TRAINED_KAPPA + 0.375
    monkeypatch.setattr(config, "TRAINED_KAPPA", value)
    reloaded = importlib.reload(sanity_check)
    try:
        assert reloaded.V1_PARAMS["kappa"] == value
    finally:
        monkeypatch.undo()
        importlib.reload(sanity_check)
