"""Tests V4 T19 : detection des bras Level-3 qui n'ont pas fini.

Le risque que cette tache couvre est asymetrique : ne PAS detecter un
avortement transforme le plantage d'un bras en victoire de l'autre. Les
tests portent donc d'abord sur la detection (aucun faux negatif sur les
marques reellement emises par V1), puis sur la propagation de l'exclusion
dans la synthese inter-folds.
"""
import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("v4", "v3", ""):
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "study", sub))

from t15c_fold_synthesis import load_divergence_audit
from t19_arm_divergence_audit import DIVERGENCE_PENALTY, parse_abort

# Trace reelle emise par V1 sur le bras classique du fold `rotor`.
REAL_ABORT = (
    "[WARNING] Violation CFL detectee : 1.21 > 1.0.\n"
    "[ABORT] Divergence detected at step 208 (t=0.2739)\n"
    "[DIVERGE] Partial score: combined=0.9294, diverged_fields=0/5\n"
)


def test_detects_the_real_rotor_abort():
    ab = parse_abort(REAL_ABORT)
    assert ab is not None
    assert ab["abort_step"] == 208
    assert ab["abort_time"] == pytest.approx(0.2739)
    assert ab["partial_combined"] == pytest.approx(0.9294)
    assert ab["diverged_fields"] == 0 and ab["n_fields"] == 5


def test_completed_run_yields_no_abort():
    txt = ("Hot-Start captured at t=0.9046\n"
           "step 1200 done\n"
           "error_vx.......... 0.001234\n")
    assert parse_abort(txt) is None


def test_cfl_warning_alone_is_not_an_abort():
    """V1 emet des avertissements CFL sans interrompre : les confondre
    avec un arret exclurait des folds parfaitement valides."""
    txt = ("[WARNING] Violation CFL detectee : 1.03 > 1.0.\n"
           "[WARNING] Violation CFL detectee : 1.21 > 1.0.\n")
    assert parse_abort(txt) is None


def test_abort_without_partial_line_still_detected():
    """Le garde-fou peut sortir par son `except` sans emettre [DIVERGE].
    L'arret doit rester detecte."""
    ab = parse_abort("[ABORT] Divergence detected at step 7 (t=0.0100)\n")
    assert ab is not None and ab["abort_step"] == 7
    assert "partial_combined" not in ab


def test_scientific_notation_in_abort_time():
    ab = parse_abort("[ABORT] Divergence detected at step 3 (t=1.5e-03)\n")
    assert ab is not None and ab["abort_time"] == pytest.approx(1.5e-3)


def test_flat_penalty_constant_matches_v1():
    """Si V1 change sa penalite forfaitaire, le drapeau `flat_penalty`
    devient faux sans prevenir."""
    import pipeline
    src = open(pipeline.__file__).read()
    assert f"DIVERGENCE_PENALTY = {DIVERGENCE_PENALTY}" in src


def test_audit_file_absent_returns_none(tmp_path):
    """Absence d'audit != tout va bien : la synthese doit pouvoir le
    distinguer."""
    assert load_divergence_audit(str(tmp_path)) is None


def test_audit_maps_folds_to_usability(tmp_path):
    payload = {"results": [
        {"fold": "ot", "fold_usable": True},
        {"fold": "rotor", "fold_usable": False},
    ]}
    json.dump(payload, open(os.path.join(
        str(tmp_path), "t19_arm_divergence_audit.json"), "w"))
    got = load_divergence_audit(str(tmp_path))
    assert got == {"ot": True, "rotor": False}
