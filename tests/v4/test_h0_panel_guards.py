"""Deux defauts du panel H0, trouves en le verifiant au lieu de l'affirmer.

D-1  Un balayage vide sortait avec le code 0.
     `--N 32` (aucun artefact d'entree a cette resolution) affichait
     « SKIP … missing input » puis « no input. » et rendait la main sans
     erreur. Une campagne qui n'avait rien mesure etait donc indiscernable
     d'une campagne reussie — exactement ce que le depot s'interdit.

D-2  Le nom de l'artefact ne contenait pas le scenario.
     Lancer quatre processus en parallele, un par scenario — ce qui est la
     facon evidente de paralleliser une campagne — les faisait tous ecrire
     dans le MEME fichier. Le dernier ecrasait les trois autres, et
     l'artefact restant ressemblait trait pour trait a une campagne
     complete : meme nom, meme structure, quatre fois moins de donnees.

Le second est le plus dangereux des deux : il ne laisse aucune trace.
"""

import ast
import os
import subprocess
import sys
import textwrap

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_PANEL = os.path.join(_REPO_ROOT, "study", "h0_selection",
                      "h0_optimiser_equivalence.py")


# ── D-1 : un balayage vide doit crier ─────────────────────────────────

def test_an_empty_sweep_exits_nonzero():
    """N=32 n'a pas d'artefacts d'entree : le panel doit echouer, pas finir."""
    r = subprocess.run(
        [sys.executable, _PANEL, "--N", "32", "--dim", "2", "--n-snaps", "1",
         "--qaoa-reps", "1", "--scenario", "orszag_tang"],
        capture_output=True, text=True, cwd=_REPO_ROOT, timeout=900)
    assert r.returncode != 0, (
        "le panel a rendu la main avec le code 0 sans avoir mesure quoi que "
        "ce soit ; une campagne vide doit etre discernable d'une campagne "
        "reussie")
    assert "balayage vide" in (r.stderr + r.stdout)


def test_the_empty_sweep_message_names_what_is_missing():
    """Un message d'erreur qui ne dit pas quoi chercher fait perdre l'heure
    suivante."""
    src = open(_PANEL, encoding="utf-8").read()
    i = src.index("balayage vide")
    msg = src[i:i + 600]
    for token in ("args.scenario", "args.N", "args.dim", "dns_", "patches_"):
        assert token in msg, (
            f"le message d'erreur du balayage vide ne mentionne pas {token}")


def test_the_panel_never_returns_silently_on_no_records():
    """Le `return` nu doit avoir disparu du chemin « aucun enregistrement »."""
    src = open(_PANEL, encoding="utf-8").read()
    assert 'print("no input."); return' not in src
    assert 'print("no input.")' not in src


# ── D-2 : le scenario doit entrer dans le nom ─────────────────────────

def _out_name(**kw):
    """Reconstruit le nom de sortie du panel pour un jeu d'arguments."""
    src = open(_PANEL, encoding="utf-8").read()
    start = src.index("    _full_sweep = set(args.scenario)")
    end = src.index('+ ".npz")', start) + len('+ ".npz")')
    snippet = textwrap.dedent(src[start:end])

    class A:
        pass
    args = A()
    for k, v in kw.items():
        setattr(args, k, v)
    ns = {"args": args, "os": os, "RESULTS_DIR": "",
          "SCENARIOS": ("orszag_tang", "kelvin_helmholtz",
                        "mhd_rotor", "harris_tearing")}
    exec(compile(ast.parse(snippet), "<panel>", "exec"), ns)
    return os.path.basename(ns["out"])


_BASE = dict(N=96, dim=3, with_psi=False, fixed_curl=False, zero_psi=False,
             no_exact=False, backend="state_vector", scale_kopt=False,
             mapper="v2")
_ALL = ["orszag_tang", "kelvin_helmholtz", "mhd_rotor", "harris_tearing"]


def test_a_full_sweep_keeps_the_historical_name():
    """Les artefacts deja publies ne doivent pas etre renommes."""
    assert _out_name(scenario=_ALL, **_BASE) == \
        "h0_optimiser_equivalence_N96_dim3.npz"


def test_the_full_sweep_name_is_order_independent():
    """Passer les memes scenarios dans un autre ordre est le meme balayage."""
    assert _out_name(scenario=list(reversed(_ALL)), **_BASE) == \
        _out_name(scenario=_ALL, **_BASE)


@pytest.mark.parametrize("scen", _ALL)
def test_a_single_scenario_gets_its_own_file(scen):
    name = _out_name(scenario=[scen], **_BASE)
    assert scen in name, (
        f"le nom {name} ne porte pas le scenario : quatre processus "
        "paralleles s'ecraseraient mutuellement")


def test_four_parallel_runs_would_not_collide():
    """Le defaut lui-meme : les quatre noms doivent etre deux a deux distincts."""
    names = [_out_name(scenario=[s], **_BASE) for s in _ALL]
    assert len(set(names)) == 4, f"collision entre {names}"


def test_a_partial_sweep_is_distinct_from_the_full_one():
    """Trois scenarios sur quatre n'est pas la campagne complete."""
    partial = _out_name(scenario=_ALL[:3], **_BASE)
    full = _out_name(scenario=_ALL, **_BASE)
    assert partial != full


def test_the_scenario_tag_composes_with_the_other_flags():
    """Le suffixe de scenario ne doit pas manger les autres variantes."""
    kw = dict(_BASE)
    kw.update(with_psi=True, scale_kopt=True)
    name = _out_name(scenario=["mhd_rotor"], **kw)
    assert "mhd_rotor" in name and "_withpsi" in name and "_scalekopt" in name
