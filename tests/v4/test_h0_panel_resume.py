"""Reprise du panel H0, et le defaut qui rendait `--no-exact` inutilisable.

D-3  `--no-exact` levait un KeyError APRES le calcul.
     Quand l'optimum n'est pas certifie, la branche de repli ecrivait
     `mask_match` — que personne ne lit — et omettait `n_diff_patch`, alors
     que la boucle d'enregistrement lit `exact_match` et `n_diff_patch`.
     L'erreur ne survenait donc qu'au moment de consigner le PREMIER
     instantane, c'est-a-dire apres des heures de calcul a 32 qubits. La
     campagne H0b tournait dans cet etat.

D-4  Aucun point de reprise.
     Une campagne complete dure des heures ; le processus peut mourir. Sans
     reprise, tout est a refaire. Le piege associe est plus subtil que
     l'absence de reprise : reprendre un point de reprise produit sous
     D'AUTRES reglages melangerait deux campagnes dans un artefact
     parfaitement bien forme. La signature couvre donc tout ce qui change
     les nombres, et une reprise incompatible est refusee.
"""

import importlib.util
import json
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_PANEL = os.path.join(_REPO_ROOT, "study", "h0_selection",
                      "h0_optimiser_equivalence.py")


@pytest.fixture(scope="module")
def panel():
    spec = importlib.util.spec_from_file_location("h0panel", _PANEL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Args:
    pass


def _args(**kw):
    a = _Args()
    base = dict(N=96, dim=3, re=[400], scenario=["orszag_tang"],
                with_psi=True, fixed_curl=False, zero_psi=False,
                no_exact=False, backend="state_vector", scale_kopt=True,
                mapper="v2", qaoa_reps=[1, 3, 6], n_snaps=6, sweeps=500,
                restarts=5, shots=4096, k_opt=60, no_qaoa=False, seed=0,
                no_resume=False)
    base.update(kw)
    for k, v in base.items():
        setattr(a, k, v)
    return a


# ── D-3 : les cles du repli doivent etre celles que l'on lit ──────────

def test_the_uncertified_fallback_uses_the_same_keys(panel):
    """Sans optimum certifie, les trois cles doivent exister quand meme.

    On compare l'ensemble des cles produites par `decision_agreement` a
    celles que la branche de repli ecrit : toute divergence rejaillit en
    KeyError apres le calcul.
    """
    import numpy as np

    spins = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int8)
    produced = set(panel.decision_agreement(spins, spins, 2))
    assert {"agree_spin", "exact_match", "n_diff_patch"} <= produced

    src = open(_PANEL, encoding="utf-8").read()
    i = src.index('r.update(dict(agree_spin=float("nan")')
    fallback = src[i:i + 300]
    for key in ("exact_match", "n_diff_patch"):
        assert key in fallback, (
            f"la branche non certifiee n'ecrit pas {key} ; "
            "--no-exact levera un KeyError apres le calcul")


def test_the_dead_key_is_gone(panel):
    """`mask_match` etait ecrit et jamais lu : le repli mentait sur son nom."""
    src = open(_PANEL, encoding="utf-8").read()
    assert 'mask_match=float("nan")' not in src


# ── D-4 : la signature de reprise ────────────────────────────────────

def test_resume_round_trips(panel, tmp_path):
    a = _args()
    ckpt = str(tmp_path / "c.jsonl")
    panel._append_checkpoint(ckpt, a, "orszag_tang", 400, 3,
                             [{"solver": "sa", "E": 1.0}], True)
    panel._append_checkpoint(ckpt, a, "orszag_tang", 400, 7,
                             [{"solver": "sa", "E": 2.0}], False)
    recs, diags, done = panel._load_checkpoint(ckpt, a)
    assert len(recs) == 2
    assert diags == [True, False]
    assert done == {("orszag_tang", 400, 3), ("orszag_tang", 400, 7)}


@pytest.mark.parametrize("changed", [
    {"seed": 7}, {"dim": 4}, {"N": 128}, {"with_psi": False},
    {"fixed_curl": True}, {"zero_psi": True}, {"qaoa_reps": [1]},
    {"scale_kopt": False}, {"mapper": "v1"}, {"shots": 1024},
    {"backend": "matrix_product_state"}, {"no_exact": True},
])
def test_a_different_setting_refuses_the_checkpoint(panel, tmp_path, changed):
    """Chaque reglage qui change les nombres doit invalider la reprise.

    Sans cela, l'artefact final serait un panachage de deux campagnes,
    impossible a distinguer d'une campagne coherente.
    """
    ckpt = str(tmp_path / "c.jsonl")
    panel._append_checkpoint(ckpt, _args(), "orszag_tang", 400, 3,
                             [{"solver": "sa", "E": 1.0}], True)
    with pytest.raises(SystemExit, match="reglages differents"):
        panel._load_checkpoint(ckpt, _args(**changed))


def test_the_scenario_is_not_part_of_the_signature(panel, tmp_path):
    """Le scenario est deja dans le NOM du point de reprise.

    L'inclure dans la signature rendrait toute reprise impossible des qu'on
    relance scenario par scenario, ce qui est justement le cas d'usage.
    """
    ckpt = str(tmp_path / "c.jsonl")
    panel._append_checkpoint(ckpt, _args(), "orszag_tang", 400, 3,
                             [{"solver": "sa", "E": 1.0}], True)
    recs, _d, _done = panel._load_checkpoint(ckpt, _args(scenario=["mhd_rotor"]))
    assert len(recs) == 1


def test_no_resume_ignores_the_checkpoint(panel, tmp_path):
    ckpt = str(tmp_path / "c.jsonl")
    panel._append_checkpoint(ckpt, _args(), "orszag_tang", 400, 3,
                             [{"solver": "sa", "E": 1.0}], True)
    recs, diags, done = panel._load_checkpoint(ckpt, _args(no_resume=True))
    assert (recs, diags, done) == ([], [], set())


def test_a_truncated_last_line_is_tolerated(panel, tmp_path):
    """Une mort brutale peut couper la derniere ligne en deux.

    Elle doit etre ignoree — l'instantane sera recalcule — et surtout ne pas
    faire echouer toute la reprise.
    """
    ckpt = str(tmp_path / "c.jsonl")
    panel._append_checkpoint(ckpt, _args(), "orszag_tang", 400, 3,
                             [{"solver": "sa", "E": 1.0}], True)
    with open(ckpt, "a", encoding="utf-8") as fh:
        fh.write('{"scenario": "orszag_ta')
    recs, _d, done = panel._load_checkpoint(ckpt, _args())
    assert len(recs) == 1 and done == {("orszag_tang", 400, 3)}


def test_a_missing_checkpoint_is_not_an_error(panel, tmp_path):
    recs, diags, done = panel._load_checkpoint(
        str(tmp_path / "absent.jsonl"), _args())
    assert (recs, diags, done) == ([], [], set())


def test_the_checkpoint_name_follows_the_artefact_name(panel):
    """Les deux derivent de la meme fonction, donc ne peuvent pas diverger."""
    a = _args()
    art = os.path.basename(panel._output_path(a))
    ck = os.path.basename(panel._checkpoint_path(a))
    assert art.endswith(".npz") and ck.endswith(".jsonl")
    assert art[:-len(".npz")] == ck[:-len(".jsonl")]


def test_each_line_is_flushed_to_disk(panel):
    """Sans fsync, une mort brutale perd des instantanes deja annonces."""
    src = open(_PANEL, encoding="utf-8").read()
    i = src.index("def _append_checkpoint")
    body = src[i:i + 900]
    assert "fh.flush()" in body and "os.fsync(fh.fileno())" in body
