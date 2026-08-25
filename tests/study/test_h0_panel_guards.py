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



def _repo_root():
    """Racine du depot : on remonte jusqu'au dossier qui contient `src/`.

    Un calcul par `dirname` repete depend de la profondeur du fichier et
    casse au premier deplacement — souvent en silence, en pointant vers un
    chemin qui n'existe pas.
    """
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_REPO_ROOT = _repo_root()
_PANEL = os.path.join(_REPO_ROOT, "study", "h0_selection",
                      "h0_optimiser_equivalence.py")


# ── D-1 : un balayage vide doit crier ─────────────────────────────────

@pytest.fixture(scope="module")
def empty_sweep():
    """Le panel lance sur une taille sans artefacts d'entree.

    Partage par les deux tests ci-dessous : le sous-processus est la SEULE
    facon de voir le message tel qu'il est reellement emis.
    """
    return subprocess.run(
        [sys.executable, _PANEL, "--N", "32", "--dim", "2", "--n-snaps", "1",
         "--qaoa-reps", "1", "--scenario", "orszag_tang"],
        capture_output=True, text=True, cwd=_REPO_ROOT, timeout=900)


def test_an_empty_sweep_exits_nonzero(empty_sweep):
    """N=32 n'a pas d'artefacts d'entree : le panel doit echouer, pas finir."""
    assert empty_sweep.returncode != 0, (
        "le panel a rendu la main avec le code 0 sans avoir mesure quoi que "
        "ce soit ; une campagne vide doit etre discernable d'une campagne "
        "reussie")
    assert "balayage vide" in (empty_sweep.stderr + empty_sweep.stdout)


def test_the_empty_sweep_message_names_what_is_missing(empty_sweep):
    """Un message d'erreur qui ne dit pas quoi chercher fait perdre l'heure
    suivante.

    D-126 — ce test lisait le SOURCE : il prenait la PREMIERE occurrence de
    « balayage vide » dans le fichier et exigeait `args.scenario`, `args.N`,
    `args.dim` dans les 600 caracteres suivants. Deux defauts en un.

    D'abord l'ancre n'est pas unique : le jour ou un commentaire a mentionne
    la regle « un balayage vide doit crier » PLUS HAUT dans le fichier (D-122),
    la fenetre est tombee sur le commentaire et le test a rougi sans qu'aucun
    defaut n'existe — 5e faux rouge de cette forme dans ce depot.

    Ensuite il ne mesurait pas ce qu'il annonce : `args.scenario` dans le
    source est le nom du CODE, pas ce que l'utilisateur lit. Un message qui
    n'interpolerait rien le contiendrait aussi. On verifie donc les VALEURS
    dans la sortie reelle : c'est ce qui fait perdre, ou gagner, l'heure
    suivante.
    """
    sortie = empty_sweep.stderr + empty_sweep.stdout
    for token in ("orszag_tang", "N=32", "dim=2", "dns_", "patches_"):
        assert token in sortie, (
            f"le message du balayage vide ne nomme pas {token} — il ne dit "
            f"pas quoi chercher.\nSortie :\n{sortie[-1500:]}")


def test_the_panel_never_returns_silently_on_no_records():
    """Le `return` nu doit avoir disparu du chemin « aucun enregistrement »."""
    src = open(_PANEL, encoding="utf-8").read()
    assert 'print("no input."); return' not in src
    assert 'print("no input.")' not in src


# ── D-2 : le scenario doit entrer dans le nom ─────────────────────────

_PANEL_MOD = None


def _panel():
    """Charge le module du panel une fois, avec les chemins du depot."""
    global _PANEL_MOD
    if _PANEL_MOD is None:
        import importlib.util
        for _d in ("src", "study/pipeline", "study/common", "study/h0_selection",
                   "study/h2b_prediction"):
            _p = os.path.join(_REPO_ROOT, *_d.split("/"))
            if _p not in sys.path:
                sys.path.insert(0, _p)
        spec = importlib.util.spec_from_file_location("h0panel", _PANEL)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _PANEL_MOD = m
    return _PANEL_MOD


def _out_name(**kw):
    """Nom de sortie du panel, via la VRAIE fonction `_output_path`.

    L'ancienne version rejouait un extrait de source ; elle a cesse de
    fonctionner des que la logique a ete factorisee en fonction. Appeler
    le code reel teste ce qui tourne, pas une copie.
    """
    class A:
        pass
    args = A()
    for k, v in kw.items():
        setattr(args, k, v)
    return os.path.basename(_panel()._output_path(args))


_BASE = dict(N=96, dim=3, legacy_curl=False, zero_psi=False,
             no_exact=False, backend="state_vector", scale_kopt=False,
             mapper="v2")
_ALL = [
    "orszag_tang", "harris_tearing", "kelvin_helmholtz", "mhd_rotor",
    "lamb_oseen", "island_coalescence", "double_tearing", "magnetic_twist",
]


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
    """Le defaut lui-meme : les huit noms doivent etre deux a deux distincts.

    `_ALL` porte les 8 scenarios du protocole elargi (config.py, voir
    `docs/COUVERTURE.md` -- "protocole s'est elargi") depuis le 25 aout ;
    seul le nombre attendu ici (4 -> 8) n'avait pas suivi."""
    names = [_out_name(scenario=[s], **_BASE) for s in _ALL]
    assert len(set(names)) == 8, f"collision entre {names}"


def test_a_partial_sweep_is_distinct_from_the_full_one():
    """Trois scenarios sur quatre n'est pas la campagne complete."""
    partial = _out_name(scenario=_ALL[:3], **_BASE)
    full = _out_name(scenario=_ALL, **_BASE)
    assert partial != full


def test_the_scenario_tag_composes_with_the_other_flags():
    """Le suffixe de scenario ne doit pas manger les autres variantes."""
    kw = dict(_BASE)
    kw.update(legacy_curl=True, scale_kopt=True)
    name = _out_name(scenario=["mhd_rotor"], **kw)
    assert "mhd_rotor" in name and "_legacycurl" in name and "_scalekopt" in name
