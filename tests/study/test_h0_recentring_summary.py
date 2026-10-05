"""Synthese du plan de recentrage (`h0_recentring_summary.py`).

Le plan compte huit artefacts : deux mappeurs x deux seuils du biais
(deploye, `loso-f1`) x deux lectures du QAOA (majoritaire, deployee). La
synthese ne doit jamais repondre sur un plan incomplet, ni melanger des
artefacts de deux commits, ni accepter un artefact produit depuis un arbre
modifie : chacun de ces cas changerait la question sans le dire.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_REPO_ROOT = _repo_root()
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SUMMARY = os.path.join(_REPO_ROOT, "study", "h0_selection",
                        "h0_recentring_summary.py")
_PANEL = os.path.join(_REPO_ROOT, "study", "h0_selection",
                      "h0_optimiser_equivalence.py")


def _load(path, name):
    import warnings
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def summary():
    return _load(_SUMMARY, "h0_recentring_summary_t")


def _fake(path, git_hash="abc", dirty=False, f1_shift=0.0):
    solvers = ["exhaustive", "sa", "sa_warm", "greedy", "classical_init",
               "qaoa_p1", "qaoa_p2", "qaoa_p3", "qaoa_shots_p3"]
    gaps = np.linspace(0.0, 0.4, len(solvers))
    f1s = np.linspace(0.40, 0.52, len(solvers)) + f1_shift
    rows = [(s, g, f) for s, g, f in zip(solvers, gaps, f1s)]
    np.savez(
        path,
        solver=np.array([r[0] for r in rows]),
        E_gap=np.array([r[1] for r in rows]),
        f1=np.array([r[2] for r in rows]),
        hit=np.array([r[1] == 0.0 for r in rows]),
        match=np.array([r[1] == 0.0 for r in rows]),
        scenario=np.array(["harris_tearing"] * len(rows)),
        snap=np.array([6] * len(rows)),
        git_hash=git_hash, dirty_at_start=dirty,
        bias_threshold_mode="deployed", readout="majority",
        bias_threshold_scenarios=np.array(["harris_tearing"]),
        bias_threshold_values=np.array([0.15]))


def _full_plan(summary, d, **kw):
    for m in summary.MAPPERS:
        for b in summary.BIAS:
            for r in summary.READOUT:
                _fake(os.path.join(d, summary.artifact_name(m, b, r)), **kw)


def test_the_names_are_the_ones_the_panel_writes(summary):
    panel = _load(_PANEL, "h0panel_summary_t")

    class A:
        pass
    for m in summary.MAPPERS:
        for b in summary.BIAS:
            for r in summary.READOUT:
                a = A()
                for k, v in dict(
                        N=96, dim=3, legacy_curl=False, zero_psi=False,
                        no_exact=False, backend="state_vector",
                        scale_kopt=False, mapper=m, bias_threshold=b,
                        readout=r, tag=summary.TAG,
                        scenario=["harris_tearing", "kelvin_helmholtz",
                                  "mhd_rotor", "orszag_tang"]).items():
                    setattr(a, k, v)
                assert os.path.basename(panel._output_path(a)) == \
                    summary.artifact_name(m, b, r)


def test_a_complete_plan_is_summarised(summary, tmp_path):
    _full_plan(summary, str(tmp_path))
    # les faux artefacts n'ont pas de DNS derriere eux : la structure de
    # l'optimum (recalculee sur DNS) est testee sur les vrais artefacts
    cells = summary.collect(str(tmp_path), structure=False)
    assert len(cells) == 8
    c = cells["v1|loso-f1|deployed"]
    assert c["rho"] == pytest.approx(1.0)
    assert c["n_solvers"] == 9
    assert c["f1_exact_minus_classical"] == pytest.approx(-0.06)
    assert c["trajectories_exact_vs_classical"] == {
        "better": 0, "equal": 0, "worse": 1}
    assert set(c["by_trajectory"]) == {"harris_tearing"}


def test_an_incomplete_plan_is_refused(summary, tmp_path):
    _full_plan(summary, str(tmp_path))
    os.remove(os.path.join(str(tmp_path),
                           summary.artifact_name("v2", "loso-f1",
                                                 "majority")))
    with pytest.raises(FileNotFoundError):
        summary.collect(str(tmp_path))


def test_a_plan_from_two_commits_is_refused(summary, tmp_path):
    _full_plan(summary, str(tmp_path))
    _fake(os.path.join(str(tmp_path),
                       summary.artifact_name("v1", "deployed", "deployed")),
          git_hash="def")
    with pytest.raises(ValueError):
        summary.collect(str(tmp_path))


def test_a_dirty_artifact_is_refused(summary, tmp_path):
    _full_plan(summary, str(tmp_path))
    _fake(os.path.join(str(tmp_path),
                       summary.artifact_name("v2", "deployed", "majority")),
          dirty=True)
    with pytest.raises(ValueError):
        summary.collect(str(tmp_path))
