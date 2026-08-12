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
for _p in (os.path.join(_REPO_ROOT, "src"), _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

N = 64
N_BLOCKS = 3
#  Cinq appels -> dix paires. Trois appels n'en donnent que trois, ce qui
#  ne suffit pas a estimer une mediane : la premiere version de ce fichier
#  assertait le MINIMUM sur trois paires, statistique dont la valeur depend
#  presque entierement du tirage.
N_REPEATS = 5

#  Mesure de reference, 10 appels identiques -> 45 paires (mhd_rotor,
#  Re=800, N=64, 3x3 blocs, w_z_frac=0.10, threshold=0.3) :
#
#    auto-correlation de rang : min 0.350  med 0.933  max 1.000
#    dispersion par appel     : ptp de 1.79e-1 a 3.61e-1, 9 valeurs
#                               distinctes a chaque fois (jamais constant)
#    appels degeneres         : 0 / 10
#
#  Un premier sondage a 6 appels (15 paires) donnait min 0.550 : la queue
#  descend plus bas que ce que 15 paires laissaient voir. C'est le chiffre
#  a 45 paires qui fait foi, et les seuils ci-dessous portent sur la
#  MEDIANE, jamais sur le minimum.


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
    #  `.ravel()` est indispensable : les scores sortent en (3, 3), et
    #  `spearmanr` sur des entrees 2-D traite les COLONNES comme des
    #  variables et renvoie une matrice de correlation, pas un scalaire.
    #  Sans aplatissement, `float(...)` leve un TypeError — et deux tests
    #  qui tombent sur une exception ressemblent a deux assertions fausses.
    return [np.asarray(qaoa_block_scores(sim, N, N_BLOCKS, 0.3, 0.10,
                                         Phi_prev=phi_prev),
                       dtype=float).ravel()
            for _ in range(N_REPEATS)]


def test_the_fixture_hands_over_flat_vectors(repeated_scores):
    """Garde de forme, appris a mes depens.

    `qaoa_block_scores` rend un (3, 3). Passe tel quel a `spearmanr`, celui-ci
    traite les COLONNES comme des variables et renvoie une matrice : `float()`
    leve alors un TypeError. Deux tests qui tombent sur une exception se
    lisent comme deux assertions fausses, et l'on se met a chercher une
    explication dans les donnees plutot que dans la forme des tableaux.
    """
    for i, s in enumerate(repeated_scores):
        assert s.ndim == 1, f"appel {i} : forme {s.shape}, attendu un vecteur"
        assert s.size == N_BLOCKS * N_BLOCKS


def test_identical_inputs_give_different_outputs(repeated_scores):
    """Le fait lui-meme. Si ce test tombe, une graine a ete fixee quelque
    part et les assertions exactes redeviennent legitimes."""
    a = repeated_scores[0]
    assert any(not np.array_equal(a, b) for b in repeated_scores[1:]), (
        "les appels repetes donnent des sorties identiques : le bras QAOA "
        "est devenu deterministe. Retablir alors les assertions d'egalite "
        "exacte, et consigner le changement dans docs/RESULTS.md")


def test_the_spread_dwarfs_the_exact_tie_tolerance(repeated_scores):
    """L'amplitude mesuree, confrontee au seuil qui exigeait une egalite.

    Sur 15 paires : dispersion mediane 1.50e-1, maximum 2.15e-1, sur des
    scores dans [0, 1]. Le seuil `MAX_CLEAN_ADVANTAGE` vaut 1e-9.

    Un premier sondage a trois appels avait donne 9.58e-2 et 8.70e-2 ; ces
    valeurs sont exactes pour ce tirage mais SOUS-ESTIMENT la dispersion,
    trois paires ne suffisant pas a en voir la queue. C'est le chiffre a
    15 paires qui fait foi.
    """
    import itertools

    from tests.test_qaoa_scaling_and_hparams import MAX_CLEAN_ADVANTAGE

    spread = max(float(np.max(np.abs(a - b)))
                 for a, b in itertools.combinations(repeated_scores, 2))
    assert spread > 1e-3, (
        f"dispersion mesuree {spread:.3e} : plus faible qu'attendu, "
        "verifier si le bras a change")
    assert spread > 1e6 * MAX_CLEAN_ADVANTAGE, (
        f"dispersion {spread:.3e} contre tolerance {MAX_CLEAN_ADVANTAGE:.1e} : "
        "une assertion d'egalite exacte sur ce bras resterait un coup de des")


def test_the_ranking_survives_the_sampling(repeated_scores):
    """La dispersion ne doit pas non plus tout emporter.

    Si le classement des blocs changeait completement d'un appel a l'autre,
    le bras ne mesurerait rien et le plafond du balayage ne prouverait rien.

    On asserte la MEDIANE sur toutes les paires, pas le minimum. La mesure
    de reference donne min 0.550 et mediane 0.883 sur 15 paires : un seuil
    pose sur le minimum tombe exactement dans la queue de la distribution
    et echoue au hasard des tirages — ce qu'a fait la premiere version de
    ce test, qui assertait `min > 0.5` sur trois paires seulement.
    """
    import itertools

    import numpy as np
    from scipy.stats import spearmanr

    rhos = [float(spearmanr(a, b).statistic)
            for a, b in itertools.combinations(repeated_scores, 2)]
    med = float(np.median(rhos))
    assert med > 0.6, (
        f"auto-correlation de rang mediane {med:.3f} sur {len(rhos)} paires "
        f"(mesure de reference 0.883) : le bras ne classe plus de facon "
        f"stable, ses sorties ne portent alors aucune information. "
        f"Valeurs : {np.round(rhos, 3).tolist()}")


def test_the_ranking_is_nonetheless_visibly_perturbed(repeated_scores):
    """L'autre bord : le classement doit VRAIMENT bouger quelque part.

    Sans cette borne, un bras devenu deterministe passerait le test
    precedent haut la main et la dispersion mesuree plus haut n'aurait plus
    de traduction sur les decisions.
    """
    import itertools

    from scipy.stats import spearmanr

    rhos = [float(spearmanr(a, b).statistic)
            for a, b in itertools.combinations(repeated_scores, 2)]
    assert min(rhos) < 1.0, (
        "toutes les paires ont un classement identique : le bras est "
        "devenu reproductible, reevaluer les assertions relachees")


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
