"""Every pipeline score must compare states at the same physical time."""

import warnings

import numpy as np
import pytest

import pipeline as P
import train_hyperparams as TH
from Simulation.pre_compute_dns import precompute_dns
from Simulation.solver import MHDSolver


FIELDS = ("vx", "vy", "Bx", "By", "Jz")


def _synthetic_precomputed_run(dt=0.1, with_final_flux=True):
    zeros = np.zeros((4, 4))
    hot = {
        "vx": zeros, "vy": zeros, "Bx": zeros, "By": zeros,
        "t_current": 0.0, "step": 0,
    }
    entry = {"dt": dt}
    if with_final_flux:
        entry["fluxes"] = {field: zeros for field in FIELDS}
    return {0: entry}, hot


def test_precomputed_run_must_cover_the_requested_time_exactly():
    trace, hot = _synthetic_precomputed_run()
    P.validate_precomputed_run(trace, hot, N=4, T_MAX=0.1)
    with pytest.raises(ValueError, match="covers"):
        P.validate_precomputed_run(trace, hot, N=4, T_MAX=0.2)


def test_precomputed_run_requires_a_final_reference_snapshot():
    trace, hot = _synthetic_precomputed_run(with_final_flux=False)
    with pytest.raises(ValueError, match="final DNS"):
        P.validate_precomputed_run(trace, hot, N=4, T_MAX=0.1)


class RecordingTrial:
    def __init__(self):
        self.reports = []
        self.attrs = {}

    def report(self, value, step):
        self.reports.append((int(step), float(value)))

    def should_prune(self):
        return False

    def set_user_attr(self, key, value):
        self.attrs[key] = value


def _max_gap(left, right):
    return max(float(np.max(np.abs(left[key] - right[key])))
               for key in FIELDS)


def _case():
    config = {
        **TH.SCENARIO_KH,
        "N": 32,
        "T_START": 0.9,
        "T_MAX": 1.2,
        "HYBRID_DT": 0.02,
        "K_opt": 4,
        "shots": 32,
        "max_depth_override": 1,
        "study_name": "dns_kh",
    }
    trace, hot = precompute_dns(config)
    hyperparams = {
        **{name: (low + high) / 2
           for name, (low, high, _scale) in TH.SEARCH_SPACE.items()},
        **TH.FIXED_PARAMS,
    }
    return config, trace, hot, hyperparams


@pytest.fixture(scope="module")
def scored_run():
    config, trace, hot, hyperparams = _case()
    calls = []
    real_score = P.score

    def recording_score(candidate, reference, *args, **kwargs):
        calls.append((
            {key: value.copy() for key, value in candidate.items()},
            {key: value.copy() for key, value in reference.items()},
        ))
        return real_score(candidate, reference, *args, **kwargs)

    trial = RecordingTrial()
    P.score = recording_score
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final = P.pipeline(
                N=config["N"], VQA_N=2, T_MAX=config["T_MAX"],
                DT=config["DT"],
                HYBRID=int(config["HYBRID_DT"] / config["DT"]),
                verbose=False, argus=TH.create_argus(config),
                hyperparams=hyperparams, lambda_cost=0.4, trial=trial,
                dns_trace=trace, hot_start_state=hot,
                max_depth_override=config["max_depth_override"],
                scenario=config["scenario"], return_details=True,
                classical_only=True,
            )
    finally:
        P.score = real_score
    return trace, trial, calls, final


def test_intermediate_scores_use_the_snapshot_after_the_completed_step(
        scored_run):
    trace, trial, calls, _final = scored_run
    assert len(trial.reports) >= 3
    intermediate_calls = calls[:len(trial.reports)]
    for (reported_step, _value), (candidate, reference) in zip(
            trial.reports, intermediate_calls):
        completed_step = reported_step - 1
        expected = trace[completed_step]["fluxes"]
        assert _max_gap(reference, expected) == 0.0
        assert _max_gap(candidate, expected) < 1e-12


def test_intermediate_physics_error_is_zero_for_an_exact_full_grid_arm(
        scored_run):
    _trace, trial, _calls, final = scored_run
    compute_floor = 0.4 / 1.4
    assert trial.reports
    for _step, combined in trial.reports:
        assert combined == pytest.approx(compute_floor, abs=1e-9)
    assert final["patch_ratio"] == pytest.approx(1.0)
    assert final["phys_score"] < 1e-9


def _aborted_run():
    config, trace, hot, hyperparams = _case()
    count = {"calls": 0}
    references = []
    candidates = []
    real_diverged = MHDSolver.is_diverged
    real_map = P.instability_weight_map
    real_error = P.weighted_relative_error

    def diverges_on_first_pipeline_check(self, max_value=1e8):
        count["calls"] += 1
        return count["calls"] > 1

    def recording_map(reference):
        references.append({key: value.copy()
                           for key, value in reference.items()})
        return real_map(reference)

    def recording_error(candidate, reference, weights, weight_sum):
        candidates.append(candidate.copy())
        return real_error(candidate, reference, weights, weight_sum)

    MHDSolver.is_diverged = diverges_on_first_pipeline_check
    P.instability_weight_map = recording_map
    P.weighted_relative_error = recording_error
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = P.pipeline(
                N=config["N"], VQA_N=2, T_MAX=config["T_MAX"],
                DT=config["DT"],
                HYBRID=int(config["HYBRID_DT"] / config["DT"]),
                verbose=False, argus=TH.create_argus(config),
                hyperparams=hyperparams, lambda_cost=0.4, trial=None,
                dns_trace=trace, hot_start_state=hot,
                max_depth_override=config["max_depth_override"],
                scenario=config["scenario"], return_details=True,
                classical_only=True,
            )
    finally:
        MHDSolver.is_diverged = real_diverged
        P.instability_weight_map = real_map
        P.weighted_relative_error = real_error

    candidate = dict(zip(FIELDS, candidates[-len(FIELDS):]))
    return references[-1], candidate, result


def test_divergence_score_uses_a_time_aligned_reference():
    reference, candidate, result = _aborted_run()
    assert _max_gap(candidate, reference) < 1e-12
    assert result["phys_score"] < 1e-9
    assert result["scoring_error"] is None
    assert result["completed"] is False
    assert result["abort"]["kind"] == "numerical_divergence"
