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


# ── les nombres du plan, sur les vrais artefacts ──────────────────────
#
# Plan lance depuis 420404a (arbre propre), synthese
# results/h0_recentring_summary.json. Voir RESULTS.md, « Recentrage du
# biais ».

_STEM = ("h0_optimiser_equivalence_N96_dim3_harris_tearing-kelvin_helmholtz-"
         "mhd_rotor-orszag_tang")


def _results_dir():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        from config import RESULTS_DIR
    return RESULTS_DIR


@pytest.fixture(scope="module")
def plan(summary):
    return summary.collect(_results_dir())


def test_the_v1_baseline_reproduces_the_published_panel_bit_for_bit():
    """La reference V1 du plan rejoue le panel publie du 28 aout sur le code
    courant : memes 108 lignes, memes F1, memes energies, au bit pres."""
    def rows(name):
        d = np.load(os.path.join(_results_dir(), name), allow_pickle=True)
        return {(str(s), int(n), str(v)): (float(e), float(f))
                for s, n, v, e, f in zip(d["scenario"], d["snap"],
                                         d["solver"], d["E"], d["f1"])}
    published = rows(_STEM + "_v1.npz")
    replay = rows(_STEM + "_v1_recentrage.npz")
    assert len(published) == 108
    assert replay == published


def test_rho_never_turns_negative(plan):
    """Le critere pre-enregistre (rho negatif : le reglage suffit) n'est
    atteint dans aucune des huit cellules, seuil recentre compris."""
    expected = {
        "v1|deployed|majority": +0.891, "v1|deployed|deployed": +0.632,
        "v1|loso-f1|majority": +0.863, "v1|loso-f1|deployed": +0.863,
        "v2|deployed|majority": -0.148, "v2|deployed|deployed": -0.130,
        "v2|loso-f1|majority": +0.979, "v2|loso-f1|deployed": +0.828,
    }
    for key, rho in expected.items():
        assert plan[key]["rho"] == pytest.approx(rho, abs=5e-4), key
    assert not any(c["rho"] < 0 and c["p"] < 0.05 for c in plan.values())
    for key in ("v1|loso-f1|majority", "v1|loso-f1|deployed",
                "v2|loso-f1|majority", "v2|loso-f1|deployed"):
        assert plan[key]["rho"] > 0.8 and plan[key]["p"] < 0.01, key
    # V2 courant au seuil deploye : plus de rho positif (artefact du 16 aout,
    # normalisation historique : +0,870)
    for key in ("v2|deployed|majority", "v2|deployed|deployed"):
        assert plan[key]["p"] > 0.5, key


def test_the_exact_optimum_never_decides_better_on_average(plan):
    f1 = {k: (c["per_solver"]["exhaustive"]["f1"],
              c["per_solver"]["classical_init"]["f1"])
          for k, c in plan.items() if k.endswith("|majority")}
    expected = {"v1|deployed|majority": (0.437, 0.468),
                "v1|loso-f1|majority": (0.405, 0.479),
                "v2|deployed|majority": (0.386, 0.468),
                "v2|loso-f1|majority": (0.108, 0.479)}
    for key, (exh, cls) in expected.items():
        assert f1[key][0] == pytest.approx(exh, abs=5e-4), key
        assert f1[key][1] == pytest.approx(cls, abs=5e-4), key
        assert f1[key][0] < f1[key][1], key
    # par trajectoire (l'unite d'inference) : 16 comparaisons, une seule
    # gagnee par l'optimum (V1 recentre, Orszag-Tang)
    better = {k: c["trajectories_exact_vs_classical"]["better"]
              for k, c in plan.items() if k.endswith("|majority")}
    assert sum(better.values()) == 1
    assert better["v1|loso-f1|majority"] == 1
    assert plan["v1|loso-f1|majority"]["by_trajectory"]["orszag_tang"][
        "exhaustive"] > plan["v1|loso-f1|majority"]["by_trajectory"][
        "orszag_tang"]["classical_init"]


def test_the_v2_ground_state_is_always_a_uniform_mask(plan):
    """Le seuil ne fait que choisir QUEL masque uniforme : tout raffiner a
    0,15, rien raffiner sur 8 instances sur 12 une fois recentre."""
    for bias, (n_all, n_none) in {"deployed": (12, 0),
                                  "loso-f1": (4, 8)}.items():
        g = plan[f"v2|{bias}|majority"]["ground_state"]
        assert (g["n_refine_all"], g["n_refine_none"]) == (n_all, n_none)
        assert g["n_unique_optimum"] == g["n_instances"] == 12
    assert plan["v2|loso-f1|majority"]["ground_state"][
        "n_equals_classical"] == 0


def test_the_v1_ground_state(plan):
    g0 = plan["v1|deployed|majority"]["ground_state"]
    g1 = plan["v1|loso-f1|majority"]["ground_state"]
    assert (g0["n_refine_all"], g0["n_refine_none"],
            g0["n_equals_classical"]) == (9, 0, 6)
    assert (g1["n_refine_all"], g1["n_refine_none"],
            g1["n_equals_classical"]) == (1, 2, 4)


def test_the_readout_decides_how_often_qaoa_reaches_the_optimum(plan):
    """Lu par vote majoritaire, le QAOA n'atteint presque jamais l'optimum
    (H0a) ; lu comme le solveur deploye au seuil 0,15, il part de la
    decision classique et l'atteint aussi souvent qu'elle sur V1."""
    hits = {k: max(c["per_solver"][s]["hit"]
                   for s in ("qaoa_p1", "qaoa_p2", "qaoa_p3"))
            for k, c in plan.items()}
    assert hits["v1|deployed|majority"] == 0.0
    assert hits["v1|loso-f1|majority"] == 0.0
    assert hits["v2|deployed|majority"] == pytest.approx(2 / 12)
    v1d = plan["v1|deployed|deployed"]["per_solver"]
    assert v1d["qaoa_p1"]["hit"] == v1d["classical_init"]["hit"] == 0.5
    assert v1d["qaoa_p1"]["f1"] == v1d["classical_init"]["f1"]


def test_the_committed_summary_matches_the_recomputation(plan):
    import json
    path = os.path.join(_results_dir(), "h0_recentring_summary.json")
    with open(path, encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["dirty_at_start"] is False
    for key, c in plan.items():
        s = saved["cells"][key]
        assert s["rho"] == pytest.approx(c["rho"], abs=1e-12), key
        assert s["ground_state"] == c["ground_state"], key
        assert s["trajectories_exact_vs_classical"] == \
            c["trajectories_exact_vs_classical"], key
