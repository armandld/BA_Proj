"""D-46 : label_percentile_sensitivity.py imprimait « ROBUST ... fails for
ANY reasonable hard-patch definition » des que max(deltas) < 0.05, alors que
le docstring du module definit la robustesse comme « le gap ne devient
jamais positif » (delta < 0). Une marge positive jusqu'a 0.05 pouvait donc
etre lue comme une robustesse totale alors qu'a ce percentile le site model
bat deja le classique.
"""
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from label_percentile_sensitivity import interpretation_message


def _rows(deltas, percentiles):
    return [dict(percentile=p, delta=d) for p, d in zip(percentiles, deltas)]


def test_positive_delta_under_005_is_reported_sensitive_not_robust():
    """Pins the pre-fix behaviour: with the old 0.05 margin, a +0.03 delta
    (the site model beating classical at p=75) was swallowed into the
    'robust for ANY definition' verdict -- `max(deltas) < 0.05` was True
    for this input. Fixed threshold (`< 0`) reports it SENSITIVE instead.
    """
    deltas = [-0.10, -0.20, 0.03, -0.15]
    percentiles = [60, 70, 75, 80]
    msg = interpretation_message(_rows(deltas, percentiles), percentiles)
    assert "SENSITIVE" in msg
    assert "ROBUST" not in msg
    assert "p=75" in msg
    assert "+0.030" in msg


def test_all_negative_deltas_still_report_robust():
    deltas = [-0.35, -0.32, -0.15, -0.25, -0.22, -0.20]
    percentiles = [60, 70, 75, 80, 85, 90]
    msg = interpretation_message(_rows(deltas, percentiles), percentiles)
    assert "ROBUST" in msg
    assert "SENSITIVE" not in msg


def test_real_artifact_delta_stays_robust_under_the_tightened_threshold():
    """Measured on results/dns_*_Re400_N256.npz + patches_*_dim4.npz
    (the only default-args config available), dim=4, seed=0:
    max(delta) = -0.154 at p=75 -- below both the old (0.05) and the
    fixed (0) threshold, so this real run's verdict is unchanged by the
    fix. Locks the actual measured number so a future artifact change
    is visible here.
    """
    out = os.path.join(_REPO_ROOT, "results", "percentile_sensitivity_N256_dim4.npz")
    if not os.path.exists(out):
        import pytest
        pytest.skip(f"{out} not present -- run label_percentile_sensitivity.py first")
    d = np.load(out)
    deltas = d["deltas"]
    percentiles = d["percentiles"]
    assert np.isfinite(deltas).all()
    np.testing.assert_allclose(max(deltas), -0.154, atol=0.02)
    rows = [dict(percentile=p, delta=dl) for p, dl in zip(percentiles, deltas)]
    msg = interpretation_message(rows, list(percentiles))
    assert "ROBUST" in msg
