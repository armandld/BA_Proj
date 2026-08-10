"""Le bras QAOA est echantillonne, et de combien.

Six assertions a tirage unique sur ce bras avaient deja ete corrigees. Une
septieme est passee au travers : `check_sweep_behaviour` exigeait
`abs(delta) <= 1e-9`, c'est-a-dire que QAOA selectionne EXACTEMENT les memes
2 blocs sur 9 que le classique, au moins une fois sur douze combinaisons
d'hyperparametres. Elle avait ete calibree sur une execution.

La cause est structurelle, pas accidentelle :

  - `src/VQA/execute.py` construit sa distribution finale a partir de
    `sampler.run(...)` : `final_distribution = counts / total_shots` ;
  - aucune graine n'est fixee dans `src/VQA/` — ni `seed_simulator`, ni
    `np.random.seed`, ni graine passee au sampler.

Ce fichier mesure l'amplitude de cette variation au lieu de la supposer, et
verrouille les deux proprietes qui empechent de recommencer :

  1. le bras VARIE d'un appel a l'autre a entree identique — donc toute
     assertion d'egalite exacte sur ses sorties est un coup de des ;
  2. il varie assez pour que le seuil `MAX_CLEAN_ADVANTAGE = 1e-9` soit
     hors de portee de plusieurs ordres de grandeur.

Si un jour une graine est fixee dans `src/VQA/`, le premier test tombera :
c'est voulu. Il faudra alors retablir les assertions exactes, et le dire.
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (os.path.join(_REPO_ROOT, "src"), _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

N = 64
N_BLOCKS = 3
N_REPEATS = 3


@pytest.fixture(scope="module")
def rotor_state():
    """Le meme etat que test_hyperparameter_sweep : mhd_rotor, Re=800, 200 pas."""
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver
    from Simulation.PhysToAngle import AngleMapper

    grid = PeriodicGrid(resolution_N=N)
    sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
    sim.init_mhd_rotor()
    mapper = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    phi_prev = None
    for i in range(200):
        if i == 199:
            phi_prev = mapper.compute_stress_flux(sim.get_fluxes())
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)
    return sim, phi_prev


@pytest.fixture(scope="module")
def repeated_scores(rotor_state):
    """N_REPEATS appels STRICTEMENT identiques."""
    from tests.test_qaoa_scaling_and_hparams import qaoa_block_scores

    sim, phi_prev = rotor_state
    return [np.asarray(qaoa_block_scores(sim, N, N_BLOCKS, 0.3, 0.10,
                                         Phi_prev=phi_prev), dtype=float)
            for _ in range(N_REPEATS)]


def test_identical_inputs_give_different_outputs(repeated_scores):
    """Le fait lui-meme. Si ce test tombe, une graine a ete fixee quelque
    part et les assertions exactes redeviennent legitimes."""
    a = repeated_scores[0]
    assert any(not np.array_equal(a, b) for b in repeated_scores[1:]), (
        "les appels repetes donnent des sorties identiques : le bras QAOA "
        "est devenu deterministe. Retablir alors les assertions d'egalite "
        "exacte, et consigner le changement dans docs/RESULTS_V4.md")


def test_the_spread_dwarfs_the_exact_tie_tolerance(repeated_scores):
    """L'amplitude mesuree, confrontee au seuil qui exigeait une egalite.

    Mesure : ecarts max de 9.58e-2 et 8.70e-2 entre appels identiques, sur
    des scores dans [0, 1]. Le seuil `MAX_CLEAN_ADVANTAGE` vaut 1e-9.
    """
    from tests.test_qaoa_scaling_and_hparams import MAX_CLEAN_ADVANTAGE

    a = repeated_scores[0]
    spread = max(float(np.max(np.abs(a - b))) for b in repeated_scores[1:])
    assert spread > 1e-3, (
        f"dispersion mesuree {spread:.3e} : plus faible qu'attendu, "
        "verifier si le bras a change")
    assert spread > 1e6 * MAX_CLEAN_ADVANTAGE, (
        f"dispersion {spread:.3e} contre tolerance {MAX_CLEAN_ADVANTAGE:.1e} : "
        "une assertion d'egalite exacte sur ce bras resterait un coup de des")


def test_the_ranking_survives_the_sampling(repeated_scores):
    """La dispersion ne doit pas non plus tout emporter.

    Si le classement des blocs changeait completement d'un appel a l'autre,
    le bras ne mesurerait rien du tout et le plafond du balayage ne
    prouverait rien. On verifie qu'il reste fortement correle a lui-meme.
    """
    from scipy.stats import spearmanr

    a = repeated_scores[0]
    rhos = [float(spearmanr(a, b).statistic) for b in repeated_scores[1:]]
    assert min(rhos) > 0.5, (
        f"auto-correlation de rang {rhos} : le bras ne classe pas de facon "
        "stable, ses sorties ne portent alors aucune information")


def test_no_seed_is_fixed_anywhere_in_the_vqa_stack():
    """La cause, verifiee a la source plutot que deduite du symptome."""
    import glob

    hits = []
    for path in glob.glob(os.path.join(_REPO_ROOT, "src", "VQA", "*.py")):
        src = open(path, encoding="utf-8").read()
        for token in ("seed_simulator", "np.random.seed", "default_rng",
                      "set_seed", "seed_transpiler"):
            if token in src:
                hits.append((os.path.basename(path), token))
    assert not hits, (
        f"une graine est desormais fixee dans src/VQA/ : {hits}. Le bras "
        "peut etre devenu reproductible ; reevaluer les assertions qui "
        "avaient ete relachees pour cause d'echantillonnage")


def test_the_sweep_check_no_longer_asserts_an_exact_tie():
    """Le correctif lui-meme, verrouille contre une reintroduction."""
    import inspect

    from tests.test_qaoa_scaling_and_hparams import check_sweep_behaviour

    src = inspect.getsource(check_sweep_behaviour)
    assert "assert ties" not in src, (
        "l'assertion d'egalite exacte a ete reintroduite sur un bras "
        "echantillonne")
    # Le plafond, lui, doit rester : c'est l'enonce du test.
    assert "best['delta'] <= MAX_CLEAN_ADVANTAGE" in src
