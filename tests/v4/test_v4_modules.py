"""Tests V4 : reponse experimentale a l'audit.

Fonctions pures uniquement (aucune donnee DNS requise). Les nombres
scientifiques sont valides par les criteres d'acceptation des scripts
eux-memes ; ici on verifie la CORRECTION des briques de calcul.
"""
import os
import sys

import numpy as np
import pytest


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_FIGURES = os.path.join(_REPO_ROOT, "figures")
if _FIGURES not in sys.path:
    sys.path.insert(0, _FIGURES)

_HERE = os.path.dirname(os.path.abspath(__file__))

from t11_solver_attribution import (
    classical_init_spins, decision_agreement, exhaustive_ground_state,
    f1_from_masks, greedy_local_search,
)
from t11b_qaoa_displacement import (
    ground_state_marginals, mask_uniformity, theta_marginals,
    variational_progress,
)
from t12_equivariance import (
    SYMMETRY_OPS, apply_symmetry, apply_symmetry_mask, orbit_error,
)
from t13_term_ablation import zero_hamiltonian_terms
from t14_numerical_validation import observed_order, relative_l2
from stats_confirmatory import (
    holm_correction, hierarchical_bootstrap, paired_hierarchical_delta,
    tost_equivalence,
)

EMPTY_E = (np.zeros((0, 2), dtype=np.int64), np.zeros(0))
EMPTY_P = (np.zeros((0, 4), dtype=np.int64), np.zeros(0))


def _naive_ground_state(h, edges, plaqs, n_q):
    """Reference independante : boucle python sur toutes les configurations."""
    ei, ec = edges; pi, pc = plaqs
    best = (None, np.inf)
    for b in range(1 << n_q):
        s = np.array([1.0 if not (b >> q) & 1 else -1.0 for q in range(n_q)])
        E = float(np.dot(h, s))
        for (i, j), c in zip(ei, ec):
            E += c * s[i] * s[j]
        for (i, j, k, l), c in zip(pi, pc):
            E += c * s[i] * s[j] * s[k] * s[l]
        if E < best[1]:
            best = (s.copy(), E)
    return best


# ----------------------- enumeration exacte ---------------------------

def test_exhaustive_single_site_follows_sign_of_h():
    for hv, expect in ((+1.0, -1), (-1.0, +1)):
        s, E, n = exhaustive_ground_state(np.array([hv]), EMPTY_E, EMPTY_P, 1)
        assert s[0] == expect and E == pytest.approx(-abs(hv)) and n == 1


def test_exhaustive_matches_naive_reference_on_random_instances():
    rng = np.random.default_rng(0)
    n_q = 6
    for _ in range(5):
        h = rng.normal(size=n_q)
        ei = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int64)
        ec = rng.normal(size=3)
        pi = np.array([[0, 1, 2, 3]], dtype=np.int64)
        pc = rng.normal(size=1)
        got_s, got_E, _ = exhaustive_ground_state(h, (ei, ec), (pi, pc), n_q)
        ref_s, ref_E = _naive_ground_state(h, (ei, ec), (pi, pc), n_q)
        assert got_E == pytest.approx(ref_E)
        np.testing.assert_array_equal(got_s, ref_s.astype(np.int8))


def test_exhaustive_ferromagnetic_degeneracy_is_counted():
    # couplage ferromagnetique pur, aucun biais -> deux etats fondamentaux
    ei = np.array([[0, 1]], dtype=np.int64)
    s, E, n = exhaustive_ground_state(
        np.zeros(2), (ei, np.array([-1.0])), EMPTY_P, 2)
    assert E == pytest.approx(-1.0) and n >= 2
    assert mask_uniformity(s)


def test_exhaustive_refuses_oversized_problems():
    with pytest.raises(ValueError):
        exhaustive_ground_state(np.zeros(30), EMPTY_E, EMPTY_P, 30,
                                max_qubits=22)


def test_exhaustive_chunking_is_transparent():
    rng = np.random.default_rng(3)
    h = rng.normal(size=8)
    a = exhaustive_ground_state(h, EMPTY_E, EMPTY_P, 8, chunk=4)
    b = exhaustive_ground_state(h, EMPTY_E, EMPTY_P, 8, chunk=1 << 16)
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == pytest.approx(b[1])


# --------------------------- greedy ----------------------------------

def test_greedy_reaches_exact_optimum_on_separable_problem():
    rng = np.random.default_rng(1)
    h = rng.normal(size=8)
    gs, E, _ = exhaustive_ground_state(h, EMPTY_E, EMPTY_P, 8)
    init = np.ones(8, dtype=np.int8)
    g, Eg, n_flips = greedy_local_search(h, EMPTY_E, EMPTY_P, 8, init)
    assert Eg == pytest.approx(E)
    np.testing.assert_array_equal(g, gs)
    assert n_flips == int(np.sum(gs != init))


def test_greedy_is_stationary_at_the_optimum():
    h = np.array([1.0, -2.0, 0.5])
    gs, E, _ = exhaustive_ground_state(h, EMPTY_E, EMPTY_P, 3)
    g, Eg, n_flips = greedy_local_search(h, EMPTY_E, EMPTY_P, 3, gs)
    assert n_flips == 0 and Eg == pytest.approx(E)


# ----------------------- accord des decisions -------------------------

def test_decision_agreement_identity_and_full_disagreement():
    dim = 2
    a = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int8)
    same = decision_agreement(a, a, dim)
    assert same["agree_spin"] == 1.0 and same["exact_match"]
    assert same["n_diff_patch"] == 0
    opp = decision_agreement(a, -a, dim)
    assert opp["agree_spin"] == 0.0 and not opp["exact_match"]


def test_classical_init_spins_follows_threshold():
    score = np.array([[0.9, 0.1], [0.2, 0.8]])
    s = classical_init_spins(score, 0.5, 2)
    # raffiner (spin -1) ssi score > seuil, replique sur les blocs H et V
    np.testing.assert_array_equal(s[:4], [-1, 1, 1, -1])
    np.testing.assert_array_equal(s[4:], [-1, 1, 1, -1])


def test_f1_from_masks_hand_computed():
    gt = np.array([[True, True], [False, False]])
    assert f1_from_masks(gt, gt) == pytest.approx(1.0)
    assert f1_from_masks(np.ones((2, 2), bool), gt) == pytest.approx(2 / 3)
    assert f1_from_masks(np.zeros((2, 2), bool), gt) == 0.0


# ------------------- deplacement variationnel -------------------------

def test_theta_marginals_recover_the_encoded_score():
    score = np.array([[0.25, 0.64], [0.0, 1.0]])
    th = 2.0 * np.arcsin(np.sqrt(score))
    m = theta_marginals({"theta_h": th, "theta_v": th})
    np.testing.assert_allclose(m[:4], score.ravel(), atol=1e-12)
    np.testing.assert_allclose(m[4:], score.ravel(), atol=1e-12)


def test_ground_state_marginals_are_binary():
    m = ground_state_marginals(np.array([1, -1, -1, 1]))
    np.testing.assert_array_equal(m, [0.0, 1.0, 1.0, 0.0])


def test_variational_progress_endpoints_and_orthogonality():
    m_theta = np.array([0.5, 0.5]); m_gs = np.array([1.0, 1.0])
    assert variational_progress(m_theta, m_theta, m_gs)["progress"] == \
        pytest.approx(0.0)
    assert variational_progress(m_theta, m_gs, m_gs)["progress"] == \
        pytest.approx(1.0)
    half = np.array([0.75, 0.75])
    assert variational_progress(m_theta, half, m_gs)["progress"] == \
        pytest.approx(0.5)
    # deplacement orthogonal a la direction utile -> progression nulle
    orth = np.array([0.6, 0.4])
    assert variational_progress(m_theta, orth, m_gs)["progress"] == \
        pytest.approx(0.0, abs=1e-12)


def test_mask_uniformity():
    assert mask_uniformity(np.array([-1, -1, -1]))
    assert mask_uniformity(np.array([1, 1]))
    assert not mask_uniformity(np.array([1, -1]))


# --------------------------- symetries --------------------------------

def test_symmetry_ops_are_involutive_or_of_order_four():
    rng = np.random.default_rng(2)
    f = [rng.normal(size=(8, 8)) for _ in range(4)]
    for op in ("flip0", "flip1", "rot180"):
        twice = apply_symmetry(*apply_symmetry(*f, op=op), op=op)
        for a, b in zip(twice, f):
            np.testing.assert_allclose(a, b, atol=1e-12)
    g = f
    for _ in range(4):
        g = apply_symmetry(*g, op="rot90")
    for a, b in zip(g, f):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_symmetry_preserves_field_energy():
    rng = np.random.default_rng(4)
    f = [rng.normal(size=(6, 6)) for _ in range(4)]
    e0 = sum(np.sum(x ** 2) for x in f)
    for op in SYMMETRY_OPS:
        e1 = sum(np.sum(x ** 2) for x in apply_symmetry(*f, op=op))
        assert e1 == pytest.approx(e0)


def test_axial_convention_flips_B_sign_on_reflections_only():
    ones = np.ones((4, 4))
    z = np.zeros((4, 4))
    _, _, Bx_ax, _ = apply_symmetry(z, z, ones, z, op="flip1", axial_B=True)
    _, _, Bx_pol, _ = apply_symmetry(z, z, ones, z, op="flip1", axial_B=False)
    assert np.allclose(Bx_ax, -Bx_pol)          # reflexion : signe oppose
    _, _, Bx_r, _ = apply_symmetry(z, z, ones, z, op="rot180", axial_B=True)
    _, _, Bx_r2, _ = apply_symmetry(z, z, ones, z, op="rot180", axial_B=False)
    np.testing.assert_allclose(Bx_r, Bx_r2)     # rotation : identique


def test_orbit_error_bounds_and_mask_transform():
    m = np.array([[True, False], [False, False]])
    assert orbit_error(m, m) == 0.0
    assert orbit_error(m, ~m) == 1.0
    np.testing.assert_array_equal(apply_symmetry_mask(m, "flip0"),
                                  np.flip(m, axis=0))
    with pytest.raises(ValueError):
        orbit_error(m, np.zeros((3, 3), bool))


# --------------------------- ablations --------------------------------

def test_zero_hamiltonian_terms_zeroes_only_the_requested_family():
    hp = {"H_edges": (np.ones((2, 2)), np.ones((2, 2))),
          "C_edges": (np.full((2, 2), 2.0), np.full((2, 2), 3.0)),
          "K_plaquettes": np.full((2, 2), 4.0)}
    out = zero_hamiltonian_terms(hp, ("ZZ",))
    assert np.all(out["C_edges"][0] == 0) and np.all(out["C_edges"][1] == 0)
    assert np.all(out["H_edges"][0] == 1) and np.all(out["K_plaquettes"] == 4)
    # l'entree n'est jamais mutee
    assert np.all(hp["C_edges"][0] == 2.0)


def test_zero_hamiltonian_terms_empty_drop_is_identity():
    hp = {"H_edges": (np.ones((2, 2)), np.ones((2, 2))),
          "C_edges": (np.ones((2, 2)), np.ones((2, 2))),
          "K_plaquettes": np.ones((2, 2))}
    out = zero_hamiltonian_terms(hp, ())
    for k in hp:
        a, b = hp[k], out[k]
        if isinstance(a, tuple):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
        else:
            np.testing.assert_array_equal(a, b)


# ------------------- validation numerique (helpers) -------------------

def test_relative_l2_and_observed_order():
    a = [np.ones((4, 4))] * 4
    b = [np.ones((4, 4))] * 4
    assert relative_l2(a, b) == 0.0
    assert observed_order(4.0, 1.0) == pytest.approx(2.0)
    assert observed_order(2.0, 1.0) == pytest.approx(1.0)
    assert np.isnan(observed_order(1.0, 0.0))


# ------------------ statistiques confirmatoires -----------------------

def test_holm_is_monotone_and_more_conservative_than_raw():
    p = np.array([0.001, 0.02, 0.03, 0.5])
    r = holm_correction(p, alpha=0.05)
    adj = r["p_adjusted"]
    assert np.all(adj >= p - 1e-12)                  # jamais plus permissif
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)   # monotone
    assert adj[0] == pytest.approx(4 * 0.001)
    assert r["reject"][0] and not r["reject"][-1]


def test_holm_single_and_empty():
    assert holm_correction(np.array([0.04]))["p_adjusted"][0] == \
        pytest.approx(0.04)
    assert len(holm_correction([])["p_adjusted"]) == 0


def test_hierarchical_bootstrap_covers_the_mean_and_is_deterministic():
    rng = np.random.default_rng(5)
    vals, cls, reg = [], [], []
    for c in range(4):
        for r in range(3):
            vals += list(rng.normal(loc=0.0, scale=0.1, size=5))
            cls += [c] * 5; reg += [f"{c}_{r}"] * 5
    a = hierarchical_bootstrap(vals, cls, reg, B=200, seed=0)
    b = hierarchical_bootstrap(vals, cls, reg, B=200, seed=0)
    np.testing.assert_array_equal(a["boot"], b["boot"])
    assert a["ci_low"] <= a["estimate"] <= a["ci_high"]
    assert a["n_class"] == 4


def test_hierarchical_ci_is_wider_than_naive_when_class_effects_dominate():
    # effet constant par classe : la variance vit au niveau classe
    from stats import bootstrap_by_trajectory          # v3, reutilise
    effects = [-1.0, -0.5, 0.5, 1.0]
    vals, cls, reg, traj = [], [], [], []
    for c, e in enumerate(effects):
        for r in range(3):
            vals += [e] * 4
            cls += [c] * 4; reg += [f"{c}_{r}"] * 4; traj += [f"{c}_{r}"] * 4
    h = hierarchical_bootstrap(vals, cls, reg, B=400, seed=1)
    t = bootstrap_by_trajectory(vals, traj, B=400, seed=1)
    assert (h["ci_high"] - h["ci_low"]) > 0.9 * (t["ci_high"] - t["ci_low"])


def test_paired_hierarchical_delta_constant_shift():
    vals = list(np.arange(12, dtype=float))
    cls = [0] * 6 + [1] * 6
    reg = [f"{c}_{i//3}" for c, i in zip(cls, range(12))]
    r = paired_hierarchical_delta([v + 1.0 for v in vals], vals, cls, reg,
                                  B=200, seed=0)
    assert r["estimate"] == pytest.approx(1.0)
    assert r["frac_positive"] == 1.0


def test_tost_detects_equivalence_and_refuses_it_when_far():
    rng = np.random.default_rng(6)
    a = rng.normal(0.0, 0.01, size=40)
    b = a + 0.001
    r = tost_equivalence(a, b, margin=0.05)
    assert r["equivalent"] and r["p_tost"] < 0.05
    far = tost_equivalence(a, a + 0.5, margin=0.05)
    assert not far["equivalent"]


def test_tost_requires_a_positive_margin_fixed_in_advance():
    with pytest.raises(ValueError):
        tost_equivalence([1.0, 2.0], [1.0, 2.0], margin=0.0)


def test_tost_unpaired_path_runs():
    rng = np.random.default_rng(7)
    a = rng.normal(0, 0.01, size=30); b = rng.normal(0, 0.01, size=25)
    r = tost_equivalence(a, b, margin=0.05, paired=False)
    assert r["equivalent"] and r["df"] > 1


# ------------------- Level-3 driver (pure helpers) --------------------

def test_fold_scenarios_deduplicates_the_v1_scenario_list():
    """Defaut V1 : SCENARIOS_ALL liste ot et rotor deux fois. Sans
    deduplication, un fold LOSO garderait la classe tenue dans
    l'entrainement (fuite)."""
    from types import SimpleNamespace
    from t15_level3_closed_loop import fold_scenarios
    cfg = lambda name: {"scenario": name, "N": 256, "T_MAX": 1.0, "Re": 800}
    T = SimpleNamespace(SCENARIOS_ALL=[
        ("kh", cfg("kelvin_helmholtz")), ("ot", cfg("orszag_tang")),
        ("tearing", cfg("harris_tearing")), ("rotor", cfg("mhd_rotor")),
        ("ot", cfg("orszag_tang")), ("rotor", cfg("mhd_rotor")),
    ])
    scen = fold_scenarios(T, warn=False)
    keys = [k for k, _ in scen]
    assert keys == ["kh", "ot", "tearing", "rotor"]
    assert len(keys) == len(set(keys))


def test_fold_scenarios_filter_by_key_or_scenario_name():
    from types import SimpleNamespace
    from t15_level3_closed_loop import fold_scenarios
    T = SimpleNamespace(SCENARIOS_ALL=[
        ("kh", {"scenario": "kelvin_helmholtz"}),
        ("ot", {"scenario": "orszag_tang"})])
    assert [k for k, _ in fold_scenarios(T, ["ot"], warn=False)] == ["ot"]
    assert [k for k, _ in
            fold_scenarios(T, ["kelvin_helmholtz"], warn=False)] == ["kh"]


def test_summarise_pairs_arms_and_counts_wins():
    from t15_level3_closed_loop import summarise
    recs = [
        dict(qhas={"combined": 0.20, "phys_score": 0.10, "patch_ratio": 0.6},
             classical={"combined": 0.25, "phys_score": 0.12,
                        "patch_ratio": 0.5}),
        dict(qhas={"combined": 0.30, "phys_score": 0.15, "patch_ratio": 0.7},
             classical={"combined": 0.28, "phys_score": 0.14,
                        "patch_ratio": 0.6}),
    ]
    s = summarise(recs)
    assert s["combined"]["mean_delta"] == pytest.approx((-0.05 + 0.02) / 2)
    assert s["combined"]["n_qhas_better"] == 1 and s["combined"]["n"] == 2
    # le cout est plus eleve pour Q-HAS sur les deux folds
    assert s["patch_ratio"]["n_qhas_better"] == 0


# ------------------ agregation V4 et figure Pareto --------------------

def test_t16_extractors_flag_missing_and_diff():
    from t16_aggregate_v4 import rows_t11b, rows_t14
    missing = rows_t11b(None)
    assert missing and all(r["status"] == "MISSING" for r in missing)
    ok = rows_t11b(dict(frac_uniform=1.0, mean_progress=0.0854,
                        reps=np.array([1, 4]),
                        progress=np.array([0.1588, -0.0132])))
    assert all(r["status"] == "OK" for r in ok)
    drifted = rows_t11b(dict(frac_uniform=1.0, mean_progress=0.20,
                             reps=np.array([1, 4]),
                             progress=np.array([0.1588, -0.0132])))
    assert any(r["status"] == "DIFF" for r in drifted)


def test_t16_convergence_order_is_derived_from_the_error_pair():
    from t16_aggregate_v4 import rows_t14
    d = dict(conv_err=np.array([4.0e-2, 2.0e-2]),
             split_with=np.array([[32, 1e-3, 1.12]]),
             split_without=np.array([[32, 1e-8, 4.00]]),
             cons_divB=np.array([1e-14]), all_checks_pass=True)
    by = {r["metric"]: r for r in rows_t14(d)}
    assert by["self-convergence order"]["value"] == pytest.approx(1.0)
    assert by["self-convergence order"]["status"] == "OK"
    assert by["temporal order without projection"]["value"] == \
        pytest.approx(4.0)


def test_t16_level3_rows_are_missing_until_the_fold_runs(tmp_path):
    from t16_aggregate_v4 import rows_level3
    rows = rows_level3(str(tmp_path), ["ot"])
    assert rows and all(r["status"] == "MISSING" for r in rows)
    import json as _json
    _json.dump(dict(qhas={"phys_score": 0.1940, "patch_ratio": 0.6797},
                    classical={"phys_score": 0.4845}),
               open(os.path.join(str(tmp_path),
                                 "t15_level3_fold_ot.json"), "w"))
    rows = rows_level3(str(tmp_path), ["ot"])
    by = {(r["task"], r["metric"]): r for r in rows}
    assert by[("t15/ot", "Q-HAS phys")]["status"] == "OK"
    assert by[("t15b/ot", "budget-matched patch")]["status"] == "MISSING"


def test_pareto_frontier_interpolation_and_dominance():
    from pareto_frontier import interp_frontier
    front = [{"patch": 0.20, "phys": 0.40}, {"patch": 0.60, "phys": 0.10},
             {"patch": 0.90, "phys": 0.02}]
    # au milieu du premier segment
    assert interp_frontier(front, 0.40) == pytest.approx(0.25)
    # aux noeuds
    assert interp_frontier(front, 0.60) == pytest.approx(0.10)
    # au-dela des bornes : plateau (np.interp)
    assert interp_frontier(front, 1.50) == pytest.approx(0.02)
