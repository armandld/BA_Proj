"""Contrats de production du terme de point X dans ``study/``."""

import ast
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src"), _REPO_ROOT] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h3_representation", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _field(n=32):
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return (np.sin(xx) * np.cos(yy), -np.cos(xx) * np.sin(yy),
            np.sin(xx), -np.sin(yy))


def test_prepare_qaoa_inputs_produces_xpoint():
    from qaoa_inputs import prepare_qaoa_inputs

    vx, vy, bx, by = _field()
    for use_v2 in (False, True):
        _, params, _ = prepare_qaoa_inputs(
            vx, vy, bx, by, 32, 4, 400, use_v2=use_v2)
        assert "K_xpoint" in params
        assert np.max(np.abs(params["K_xpoint"])) > 1e-6


def _coefficient_sites():
    sites = []
    study_root = os.path.join(_REPO_ROOT, "study")
    for root, _, files in os.walk(study_root):
        if "__pycache__" in root or os.sep + "results" in root:
            continue
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as stream:
                tree = ast.parse(stream.read(), filename=path)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "compute_coefficients"):
                    values = [kw.value for kw in node.keywords
                              if kw.arg == "advanced_anomalies_enabled"]
                    sites.append((os.path.relpath(path, _REPO_ROOT),
                                  node.lineno, values))
    return sites


def test_every_study_producer_explicitly_enables_xpoint():
    sites = _coefficient_sites()
    assert len(sites) >= 7
    for path, line, values in sites:
        assert len(values) == 1, (
            f"{path}:{line} doit choisir explicitement le terme X-point")
        assert isinstance(values[0], ast.Constant) and values[0].value is True, (
            f"{path}:{line} doit activer advanced_anomalies_enabled=True")
