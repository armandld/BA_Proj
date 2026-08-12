"""D-23 — la trace DNS de reference finissait a un autre temps que ses deux bras.

`precompute_dns` construit la trajectoire de REFERENCE contre laquelle les
deux bras sont notes. Deux choses y etaient tacites, et l'une etait fausse.

DEFAUT : `dt = min(sim_dns.adapt_dt(...), T_MAX - t_current)`. `adapt_dt`
FIXE `sim_dns.dt` et le rend ; le `min` ne creait qu'une variable locale.
`step_full` integrait donc avec le dt NON borne pendant que `t_current`
avancait du dt borne — et c'est le dt borne qui etait ecrit dans la trace.

Mesure, orszag_tang N=32, T_MAX=0.05, dernier pas :

  integre par le solveur   0.037997804
  ecrit dans la trace      0.010730092

La reference finissait a t ~ 0.077 quand la trace annoncait 0.050. Le
pipeline, qui avance ses deux bras avec `dns_trace[step]['dt']`, les
comparait donc a une verite terrain prise 3.5 fois plus loin dans le temps.

Rien ne pouvait le signaler : la trace est bien formee, ses champs sont
finis, son `dt` est plausible. Seul un rejeu — refaire la trajectoire avec
les dt de la trace et comparer au dernier instantane — le montre. C'est ce
que fait le premier test.

CONVENTION, tacite mais voulue : les entrees portent l'etat AVANT leur pas,
sauf la derniere qui porte l'etat APRES. Le pipeline a besoin des deux.
Elle est desormais ecrite dans la docstring, et figee ici.
"""

import os
import sys

import numpy as np
import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from Simulation.grid import PeriodicGrid  # noqa: E402
from Simulation.pre_compute_dns import _init_dns_scenario, precompute_dns  # noqa: E402
from Simulation.solver import MHDSolver  # noqa: E402


def _cfg(**kw):
    c = dict(N=32, DT=1e-3, Re=400, Rm=400, T_MAX=0.05, T_START=0.0,
             HYBRID_DT=0.01, scenario="orszag_tang")
    c.update(kw)
    return c


def _replay(cfg, trace):
    """Refait la trajectoire avec les dt DE LA TRACE."""
    s = MHDSolver(PeriodicGrid(cfg["N"]), dt=cfg["DT"],
                  Re=cfg["Re"], Rm=cfg["Rm"])
    _init_dns_scenario(s, cfg["scenario"])
    for k in sorted(trace):
        s.dt = trace[k]["dt"]
        s.step_full(record_stats=False)
    return s


# ── D-23 : le rejeu doit retomber exactement sur la reference ─────────

def test_replaying_the_trace_reproduces_the_final_ground_truth_exactly():
    """LE test que le defaut ne passait pas.

    Si le solveur integre un dt different de celui qu'il ecrit, le rejeu
    diverge. Exactement zero est la seule reponse acceptable : ce sont les
    memes operations dans le meme ordre.
    """
    cfg = _cfg()
    trace, _ = precompute_dns(cfg)
    s = _replay(cfg, trace)
    ref = trace[max(trace)]["fluxes"]
    for key in ("vx", "vy", "Bx", "By"):
        assert np.max(np.abs(getattr(s, key) - ref[key])) == 0.0, (
            f"{key} diverge au rejeu : le dt integre n'est pas le dt ecrit")


@pytest.mark.parametrize("t_max", [0.02, 0.05, 0.11])
def test_the_recorded_timesteps_sum_to_the_requested_horizon(t_max):
    """La somme des dt doit valoir T_MAX. Elle le depassait, parce que le
    dernier pas etait ecrit borne et integre non borne."""
    trace, _ = precompute_dns(_cfg(T_MAX=t_max))
    total = sum(v["dt"] for v in trace.values())
    assert total == pytest.approx(t_max, rel=1e-12)


def test_no_recorded_timestep_exceeds_the_remaining_horizon():
    cfg = _cfg()
    trace, _ = precompute_dns(cfg)
    t = 0.0
    for k in sorted(trace):
        dt = trace[k]["dt"]
        assert dt <= cfg["T_MAX"] - t + 1e-15
        assert dt > 0.0, "un pas nul ferait boucler indefiniment"
        t += dt


def test_the_last_step_is_the_one_that_used_to_be_clamped():
    """Le defaut ne frappait que le dernier pas : c'est le seul ou le clamp
    mord. Un test qui ne regarderait que les pas intermediaires passerait."""
    cfg = _cfg()
    trace, _ = precompute_dns(cfg)
    ks = sorted(trace)
    s = MHDSolver(PeriodicGrid(cfg["N"]), dt=cfg["DT"], Re=cfg["Re"], Rm=cfg["Rm"])
    _init_dns_scenario(s, cfg["scenario"])
    clamped = []
    t = 0.0
    for k in ks:
        raw = s.adapt_dt(cft := 0.4) if False else s.adapt_dt(cfl_target=0.4)
        if raw > cfg["T_MAX"] - t + 1e-15:
            clamped.append(k)
        s.dt = trace[k]["dt"]
        s.step_full(record_stats=False)
        t += trace[k]["dt"]
    assert clamped, "aucun pas n'est borne : le test ne mesure rien"
    assert clamped == [ks[-1]], f"pas bornes : {clamped}"


# ── La convention de temps des instantanes ───────────────────────────

def test_intermediate_snapshots_hold_the_state_before_their_step():
    cfg = _cfg(T_MAX=0.11, HYBRID_DT=0.02)
    trace, _ = precompute_dns(cfg)
    snaps = sorted(k for k, v in trace.items() if "fluxes" in v)
    assert len(snaps) >= 3, "pas assez d'instantanes pour tester la convention"
    s = MHDSolver(PeriodicGrid(cfg["N"]), dt=cfg["DT"], Re=cfg["Re"], Rm=cfg["Rm"])
    _init_dns_scenario(s, cfg["scenario"])
    before = {}
    for k in sorted(trace):
        before[k] = s.vx.copy()
        s.dt = trace[k]["dt"]
        s.step_full(record_stats=False)
    for k in snaps[:-1]:
        assert np.max(np.abs(trace[k]["fluxes"]["vx"] - before[k])) == 0.0, (
            f"l'instantane {k} ne porte pas l'etat d'AVANT son pas")


def test_the_last_snapshot_holds_the_state_after_its_step():
    """C'est ce que le pipeline consomme comme verite terrain finale."""
    cfg = _cfg()
    trace, _ = precompute_dns(cfg)
    s = _replay(cfg, trace)
    assert np.max(np.abs(trace[max(trace)]["fluxes"]["vx"] - s.vx)) == 0.0


def test_the_two_conventions_are_written_in_the_docstring():
    """Une convention tacite se fait prendre pour l'autre."""
    doc = precompute_dns.__doc__
    assert "CONVENTION DE TEMPS" in doc
    assert "AVANT" in doc and "APRES" in doc


# ── Structure et gardes ───────────────────────────────────────────────

def test_every_step_carries_a_timestep():
    trace, _ = precompute_dns(_cfg())
    assert trace, "trace vide"
    assert sorted(trace) == list(range(len(trace))), "pas de numerotation continue"
    for k, v in trace.items():
        assert "dt" in v and np.isfinite(v["dt"])


def test_the_final_step_always_carries_fluxes():
    for t_max in (0.02, 0.05, 0.11):
        trace, _ = precompute_dns(_cfg(T_MAX=t_max))
        assert "fluxes" in trace[max(trace)], (
            "sans verite terrain finale, le pipeline n'a rien a comparer")


def test_the_flux_snapshots_carry_every_field_the_pipeline_reads():
    trace, _ = precompute_dns(_cfg())
    for v in trace.values():
        if "fluxes" in v:
            assert set(v["fluxes"]) >= {"vx", "vy", "Bx", "By", "Jz"}
            for a in v["fluxes"].values():
                assert np.all(np.isfinite(a))


def test_the_hot_start_state_is_captured_and_carries_its_time():
    _, hot = precompute_dns(_cfg(T_START=0.0))
    assert hot is not None
    assert set(hot) >= {"vx", "vy", "Bx", "By", "t_current", "step"}


def test_an_unreachable_hot_start_returns_none_rather_than_a_wrong_state():
    """T_START au-dela de l'horizon : mieux vaut None qu'un etat arbitraire."""
    _, hot = precompute_dns(_cfg(T_MAX=0.02, T_START=10.0))
    assert hot is None


def test_an_unknown_scenario_names_the_valid_ones():
    """Un KeyError nu fait relire le code source pour trouver les noms."""
    s = MHDSolver(PeriodicGrid(16), dt=1e-3, Re=400, Rm=400)
    with pytest.raises(ValueError, match="scenario inconnu"):
        _init_dns_scenario(s, "un scenario qui n'existe pas")
    try:
        _init_dns_scenario(s, "zzz")
    except ValueError as e:
        assert "orszag_tang" in str(e) and "mhd_rotor" in str(e)


@pytest.mark.parametrize("scenario", ["orszag_tang", "kelvin_helmholtz",
                                      "mhd_rotor", "harris_tearing"])
def test_every_deployed_scenario_can_be_precomputed(scenario):
    trace, _ = precompute_dns(_cfg(T_MAX=0.02, scenario=scenario))
    assert trace and "fluxes" in trace[max(trace)]


def test_a_diverging_dns_raises_instead_of_returning_garbage():
    """Une trace divergente empoisonnerait tous les essais qui la lisent."""
    import inspect
    src = inspect.getsource(precompute_dns)
    assert "is_diverged()" in src
    assert "raise RuntimeError" in src
