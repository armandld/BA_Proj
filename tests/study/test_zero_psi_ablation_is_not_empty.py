"""D-122 — `--zero-psi` sans `--with-psi` etait une ablation VIDE.

`prepare_qaoa_inputs` pose psi_h = psi_v = 0 EXACTEMENT tant que with_psi est
faux (« no temporal flux in study »). Le bloc d'ablation de `solver_panel`
reecrivait alors des zeros par des zeros : bit a bit, l'artefact `_zeropsi`
etait le meme balayage que son jumeau sans suffixe, sous un autre nom.

`results/h0_optimiser_equivalence_N96_dim3_zeropsi_scalekopt.npz` est publie
avec ce nom. Son `cli_args` porte `zero_psi: true` et aucun `with_psi` : c'est
le cas vide. Ses cinq solveurs deterministes sont bit a bit ceux de
`..._N96_dim3_scalekopt.npz` — meme `git_hash`, meme `seed`.

Ce que ces tests verrouillent :

* la RAISON de la garde — le no-op est reel, pas suppose ;
* le champ qui SEPARE — sous `with_psi`, psi est non nul, donc l'ablation
  mord : la garde n'interdit pas l'ablation, elle interdit sa version vide ;
* la GARDE elle-meme — sur l'ancienne version le panel partait en campagne
  sans rien ablater et sortait avec le code 0 ;
* le FAIT PUBLIE — pour qu'une relecture ne redecouvre pas la meme chose.
"""

import json
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

N = 32
DIM = 2
RE = 400

_RESULTS = os.path.join(_REPO_ROOT, "results")
_ZEROPSI = os.path.join(
    _RESULTS, "h0_optimiser_equivalence_N96_dim3_zeropsi_scalekopt.npz")
_TWIN = os.path.join(
    _RESULTS, "h0_optimiser_equivalence_N96_dim3_scalekopt.npz")

#: Les solveurs qui ne tirent aucun nombre au hasard. Le bras QAOA, lui,
#: disperse de 1,79e-1 a 3,61e-1 sur des appels identiques (fiche du depot) :
#: une egalite exacte n'y voudrait rien dire, ni dans un sens ni dans l'autre.
_DETERMINISTIC = ("exhaustive", "sa", "sa_warm", "greedy", "classical_init")


def _two_snapshots():
    """Deux instantanes consecutifs d'un meme ecoulement."""
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver

    sim = MHDSolver(PeriodicGrid(N), dt=1e-3, Re=RE, Rm=RE)
    sim.init_kelvin_helmholtz()
    for _ in range(20):
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)
    prev = {k: np.array(v) for k, v in sim.get_fluxes().items()}
    for _ in range(5):
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)
    return sim.get_fluxes(), prev


def _psi(data_in):
    return np.concatenate([np.asarray(data_in["psi_h"], float).ravel(),
                           np.asarray(data_in["psi_v"], float).ravel()])


# ── la raison de la garde : le no-op est mesure, pas suppose ─────────

def test_without_with_psi_the_ablation_rewrites_zeros_by_zeros():
    """psi est deja nul EXACTEMENT : l'ablation ne retire rien.

    Ce test epingle l'ANCIEN comportement. Il ne doit pas etre supprime avec
    la garde : c'est lui qui dit pourquoi elle existe. S'il rougit un jour,
    c'est que psi n'est plus nul sur le chemin de l'etude — et la garde de
    `solver_panel` doit alors etre reexaminee, pas contournee.
    """
    from qaoa_inputs import prepare_qaoa_inputs

    cur, _prev = _two_snapshots()
    data_in, _hp, _s = prepare_qaoa_inputs(
        cur["vx"], cur["vy"], cur["Bx"], cur["By"], N, DIM, RE, use_v2=True)

    psi_avant = _psi(data_in)
    assert np.max(np.abs(psi_avant)) == 0.0, (
        "psi n'est plus identiquement nul sans with_psi "
        f"(max |psi| = {np.max(np.abs(psi_avant)):.3e})")

    # le bloc d'ablation, tel quel
    psi_apres = np.zeros_like(psi_avant)
    assert np.array_equal(psi_avant, psi_apres), (
        "l'ablation modifie psi hors with_psi : la garde D-122 n'a plus lieu "
        "d'etre")


def test_with_psi_the_ablation_actually_bites():
    """Le champ qui SEPARE : sous with_psi, psi est non nul.

    Sans ce test, la garde pourrait etre lue comme « --zero-psi ne sert a
    rien » — ce qui est faux. Elle n'interdit que sa version vide.
    """
    from qaoa_inputs import prepare_qaoa_inputs

    cur, prev = _two_snapshots()
    data_in, _hp, _s = prepare_qaoa_inputs(
        cur["vx"], cur["vy"], cur["Bx"], cur["By"], N, DIM, RE, use_v2=True,
        prev_fields=prev, with_psi=True)

    psi = _psi(data_in)
    assert np.max(np.abs(psi)) > 1e-9, (
        f"psi rebranche mais nul (max |psi| = {np.max(np.abs(psi)):.3e}) : "
        "l'ablation serait vide des deux cotes et la garde ne separerait rien")


# ── la garde ─────────────────────────────────────────────────────────

def test_the_panel_refuses_an_empty_psi_ablation():
    """LA garde. Sur l'ancienne version, le panel partait en campagne.

    Il ecrivait un artefact suffixe `_zeropsi` sans avoir rien ablate, et
    sortait avec le code 0 : indiscernable d'une vraie ablation.
    """
    import h0_optimiser_equivalence as panel

    cur, _prev = _two_snapshots()
    l2 = np.zeros((DIM, DIM))

    with pytest.raises(SystemExit, match="ablation VIDE"):
        panel.solver_panel(
            cur["vx"], cur["vy"], cur["Bx"], cur["By"], N, DIM, RE,
            l2, 1.0, use_v2=True, zero_psi=True, with_psi=False,
            prev_fields=None, run_qaoa=False, no_exact=True,
            sweeps=1, n_restarts=1)


def test_the_guard_leaves_the_ordinary_path_alone():
    """Sans --zero-psi, rien ne change : la garde ne mord que sur le cas vide."""
    import h0_optimiser_equivalence as panel

    cur, _prev = _two_snapshots()
    l2 = np.zeros((DIM, DIM))

    out = panel.solver_panel(
        cur["vx"], cur["vy"], cur["Bx"], cur["By"], N, DIM, RE,
        l2, 1.0, use_v2=True, zero_psi=False, with_psi=False,
        prev_fields=None, run_qaoa=False, no_exact=True,
        sweeps=1, n_restarts=1)
    assert "rows" in out and out["rows"], "le chemin ordinaire ne rend rien"


# ── le fait publie ───────────────────────────────────────────────────

def _cli_args(path):
    return json.loads(str(np.load(path, allow_pickle=True)["cli_args"]))


@pytest.mark.skipif(not (os.path.exists(_ZEROPSI) and os.path.exists(_TWIN)),
                    reason="artefacts dim3 absents du checkout")
def test_the_published_zeropsi_artifact_is_the_empty_case():
    """L'artefact publie a bien ete produit dans le regime vide."""
    a = _cli_args(_ZEROPSI)
    assert a.get("zero_psi") is True
    assert not a.get("with_psi", False), (
        "l'artefact _zeropsi a ete produit AVEC with_psi : son ablation "
        "n'etait pas vide et l'entree D-122 doit etre remesuree")


@pytest.mark.skipif(not (os.path.exists(_ZEROPSI) and os.path.exists(_TWIN)),
                    reason="artefacts dim3 absents du checkout")
def test_the_two_published_artifacts_are_the_same_sweep():
    """Les deux artefacts ne different QUE par le bras stochastique.

    C'est la mesure qui fonde D-122 : si un jour ces deux fichiers different
    sur un solveur deterministe, l'ablation n'etait pas vide et la lecture de
    D-53 (« deux artefacts ») redevient une replication.
    """
    a = np.load(_ZEROPSI, allow_pickle=True)
    b = np.load(_TWIN, allow_pickle=True)

    assert str(a["git_hash"]) == str(b["git_hash"]), \
        "artefacts produits a des commits differents : la comparaison ne vaut"

    ca, cb = _cli_args(_ZEROPSI), _cli_args(_TWIN)
    diff = {k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
    assert diff == {"zero_psi"}, (
        f"les deux artefacts different aussi par {sorted(diff - {'zero_psi'})}"
        " : ce ne sont plus deux fois la meme campagne")

    sol = np.asarray(a["solver"], dtype=str)
    mask = np.isin(sol, _DETERMINISTIC)
    assert mask.sum() == 30, f"attendu 30 lignes deterministes, vu {mask.sum()}"

    d = np.abs(np.asarray(a["E"], float)[mask] - np.asarray(b["E"], float)[mask])
    assert float(d.max()) == 0.0, (
        f"les solveurs deterministes different (max |dE| = {d.max():.6g}) : "
        "l'ablation _zeropsi n'etait donc PAS vide")
