"""D-117 — `relative_percentile` entre dans `SEARCH_SPACE` : la chaine complete.

Ajouter un nom a `SEARCH_SPACE` ne cable rien. D-31 est exactement cette
faute : `beta_michelson` etait propose a Optuna, echantillonne a chaque
essai, consigne dans la base -- et `pipeline.py` ne le lisait nulle part.
La campagne a optimise un parametre sans effet pendant 202 essais.

Ce fichier interdit la recidive, de deux facons independantes :

  - un **balayage generique** : TOUT nom de `SEARCH_SPACE` doit apparaitre
    comme cle d'un `hp.get(...)` dans le code VIVANT de `pipeline.py`.
    L'analyse passe par l'AST, donc le bloc mort de `pipeline.py` -- un
    litteral triple-quote qui contient un `hp.get` par parametre -- ne
    compte pas. Ce n'est pas une precaution theorique : la premiere
    insertion a vise ce bloc-la, et `relative_percentile` n'etait
    reference nulle part ailleurs que dans l'appel au mappeur. La
    pipeline aurait leve `NameError` au premier essai.

  - une **mutation mesuree** : deux mappeurs qui ne different que par
    `relative_percentile` doivent produire des coefficients differents.
    Un test qui verifie seulement que l'argument est accepte passerait
    encore si `__init__` le rangeait dans un attribut que personne ne lit.

Mesure de reference, faite a travers la VRAIE `compute_coefficients`
(reseau de tourbillons periodique, N=64, Re=Rm=800, hyperparametres
deployes, `advanced_anomalies_enabled=True`) :

    percentile 50 -> K_plaquettes non nuls : 2040 / 4096   |K| max 5.879e+00
    percentile 90 -> K_plaquettes non nuls :  404 / 4096   |K| max 4.254e-01
    percentile 99 -> K_plaquettes non nuls :   32 / 4096   |K| max 1.726e-02

C'est la quantite que le parametre pilote : combien de cellules le critere
designe, et avec quelle amplitude. Deux ordres de grandeur sur `|K| max`
entre les bornes de l'espace de recherche -- ce n'est pas un reglage
cosmetique, c'est le nombre de patchs que l'AMR ouvrira.

Note sur ce champ : |omega| max = 2.00e+00 pour `omega_crit = 8.15e-01`,
donc le canal FLUIDE est au-dessus de son critere absolu ; c'est le canal
MAGNETIQUE (B uniforme, donc sans courant) qui passe par le relatif. Le
percentile agit malgre tout sur `K_plaquettes`, qui combine les deux --
raison pour laquelle la garde `test_au_dessus_du_critere_absolu...`
s'exerce sur `_effective_crit` canal par canal, ou l'invariant est exact.
"""

import ast
import os
import sys

import numpy as np
import pytest
from types import SimpleNamespace

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import train_hyperparams as TH                       # noqa: E402
from Simulation.HamiltParams import PhysicalMapper   # noqa: E402


# ══════════════════════════════════════════════════════════════════
#  1. L'espace de recherche le declare
# ══════════════════════════════════════════════════════════════════

def test_relative_percentile_est_dans_lespace_de_recherche():
    assert "relative_percentile" in TH.SEARCH_SPACE
    lo, hi, log = TH.SEARCH_SPACE["relative_percentile"]
    assert 0.0 < lo < hi < 100.0, (
        f"un percentile vit dans ]0, 100[ : bornes ({lo}, {hi})")
    assert log is False, "un rang n'est pas une amplitude : echelle lineaire"


def test_la_graine_de_phase1_reproduit_le_comportement_actuel():
    """Le premier essai doit refaire EXACTEMENT ce que fait le depot
    aujourd'hui, sans quoi tout ecart mesure ensuite melange
    l'exploration et le changement de point de depart."""
    seeds = TH.phase1_seeds()
    assert seeds, "grille de graines vide"
    for seed in seeds:
        assert seed["relative_percentile"] == PhysicalMapper.RELATIVE_PERCENTILE


def test_optuna_lechantillonne_vraiment():
    """`trial.distributions` est la seule preuve qu'Optuna a bien recu une
    distribution : une cle absente de `SEARCH_SPACE` serait ignoree en
    silence."""
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    vu = {}

    def objective(trial):
        vu["hp"] = TH.suggest_hyperparams(trial)
        vu["dists"] = dict(trial.distributions)
        return 0.0

    optuna.create_study().optimize(objective, n_trials=1)
    assert "relative_percentile" in vu["dists"]
    lo, hi, _ = TH.SEARCH_SPACE["relative_percentile"]
    assert lo <= vu["hp"]["relative_percentile"] <= hi


# ══════════════════════════════════════════════════════════════════
#  2. Le balayage anti-D-31 : rien d'optimise qui ne soit lu
# ══════════════════════════════════════════════════════════════════

def _cles_hp_get_vivantes():
    """Les cles litterales de tout `<x>.get('...')` du code VIVANT.

    L'AST ne descend pas dans les litteraux de chaine : le bloc mort de
    `pipeline.py` n'est pas parcouru. C'est voulu -- une insertion faite
    dans ce bloc-la ne doit PAS satisfaire ce test.
    """
    tree = ast.parse(open(os.path.join(_SRC, "pipeline.py")).read())
    cles = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            cles.add(node.args[0].value)
    return cles


def test_le_bloc_mort_ne_compte_pas():
    """Garde-fou du garde-fou. Si le bloc commente de `pipeline.py`
    disparait ou devient du code, ce test le dit -- sinon le balayage
    ci-dessous pourrait etre satisfait par du texte."""
    source = open(os.path.join(_SRC, "pipeline.py")).read()
    occurrences = source.count("w_z_frac    = hp.get(")
    assert occurrences == 2, (
        f"{occurrences} occurrences de l'affectation de w_z_frac : le bloc "
        f"mort a change de forme, revoir `_cles_hp_get_vivantes`.")
    assert "w_z_frac" in _cles_hp_get_vivantes()


@pytest.mark.parametrize("nom", sorted(TH.SEARCH_SPACE))
def test_chaque_parametre_optimise_est_lu_par_la_pipeline(nom):
    """D-31 generique. Un parametre propose a Optuna que `pipeline.py`
    ne lit pas coute le prix plein de la campagne et ne change rien."""
    assert nom in _cles_hp_get_vivantes(), (
        f"`{nom}` est dans SEARCH_SPACE mais aucun `hp.get('{nom}')` vivant "
        f"dans pipeline.py : Optuna l'optimiserait sans effet (D-31).")


def test_la_valeur_lue_est_transmise_au_mappeur():
    """Lire `hp` ne suffit pas : la valeur doit atteindre le constructeur.

    On exige la forme `relative_percentile=relative_percentile` dans
    l'appel a `PhysicalMapper(...)` -- une variable locale calculee puis
    jamais passee serait la meme faute d'un cran plus loin.
    """
    tree = ast.parse(open(os.path.join(_SRC, "pipeline.py")).read())
    appels = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name)
              and n.func.id == "PhysicalMapper"]
    assert appels, "aucun appel a PhysicalMapper dans pipeline.py"
    for appel in appels:
        noms = {kw.arg for kw in appel.keywords}
        assert "relative_percentile" in noms, (
            "un appel a PhysicalMapper sans relative_percentile : "
            "l'essai serait echantillonne puis jete")


def test_la_pipeline_ne_leve_pas_sur_le_nom_quelle_vient_dajouter():
    """La premiere insertion a atterri dans le bloc mort : le nom n'etait
    defini nulle part, et `PhysicalMapper(relative_percentile=...)`
    aurait leve `NameError` au premier essai de la campagne. On compile
    le module et on verifie que le nom est BIEN une variable locale de
    `pipeline`, pas un fantome."""
    tree = ast.parse(open(os.path.join(_SRC, "pipeline.py")).read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "pipeline")
    assignes = {t.id for n in ast.walk(fn) if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)}
    assert "relative_percentile" in assignes, (
        "`relative_percentile` est passe a PhysicalMapper sans jamais etre "
        "affecte dans `pipeline` : NameError au premier essai.")


# ══════════════════════════════════════════════════════════════════
#  3. La mutation : le parametre change le resultat
# ══════════════════════════════════════════════════════════════════

def test_le_defaut_est_la_constante_de_classe():
    """`None` doit rendre le comportement d'avant, a l'identique."""
    assert PhysicalMapper().relative_percentile == \
        PhysicalMapper.RELATIVE_PERCENTILE
    assert PhysicalMapper(relative_percentile=55.0).relative_percentile == 55.0


def test_le_percentile_change_le_seuil_effectif():
    """Sous le critere absolu, `_effective_crit` rend le percentile
    demande -- et il est strictement croissant en `relative_percentile`."""
    signal = np.linspace(0.0, 1.0, 1001)
    crit_absolu = 10.0            # hors de portee : le relatif decide
    p50 = PhysicalMapper(relative_percentile=50.0)._effective_crit(
        signal, crit_absolu)
    p90 = PhysicalMapper(relative_percentile=90.0)._effective_crit(
        signal, crit_absolu)
    p99 = PhysicalMapper(relative_percentile=99.0)._effective_crit(
        signal, crit_absolu)
    assert p50 == pytest.approx(0.50, abs=1e-3)
    assert p90 == pytest.approx(0.90, abs=1e-3)
    assert p99 == pytest.approx(0.99, abs=1e-3)
    assert p50 < p90 < p99


def test_le_percentile_change_les_coefficients_deployes():
    """La mutation vue par la fonction que la pipeline appelle vraiment.

    Le test precedent interroge `_effective_crit` seul : il passerait
    encore si `compute_coefficients` cessait de s'en servir. Celui-ci
    part des champs et lit `K_plaquettes`, donc il voit `src/` changer.
    """
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver
    from Simulation.PhysToAngle import AngleMapper

    N, RE, RM = 64, 800, 800
    HP = dict(gamma_hydro=2.1272, gamma_mag=2.3611, kappa=14.3321,
              sigma=0.05, beta_curl=0.8199, beta_xpoint=0.4256,
              w_z_frac=0.1013)
    grid = PeriodicGrid(N)
    sim = MHDSolver(grid, dt=1e-3, Re=RE, Rm=RM)
    x = np.arange(N) * grid.dx
    X, Y = np.meshgrid(x, x, indexing="ij")       # AXIS_X=0, AXIS_Y=1
    k = 2.0 * np.pi / grid.L                      # reseau periodique
    sim.vx = -np.cos(k * X) * np.sin(k * Y)
    sim.vy = np.sin(k * X) * np.cos(k * Y)
    sim.Bx = np.ones_like(X)
    sim.By = np.zeros_like(X)
    etat = sim.get_fluxes()
    score = AngleMapper.classical_score(etat)

    compte, ampli = {}, {}
    for p in (50.0, 90.0, 99.0):
        mapper = PhysicalMapper(cs=1.0, nu=grid.L / RE, eta_mhd=grid.L / RM,
                                dx=grid.dx, relative_percentile=p, **HP)
        coeffs = mapper.compute_coefficients(
            sim, score, etat, threshold_amr=0.5,
            advanced_anomalies_enabled=True)
        K = np.asarray(coeffs["K_plaquettes"])
        compte[p] = int(np.count_nonzero(np.abs(K) > 1e-12))
        ampli[p] = float(np.abs(K).max())

    assert compte[50.0] > compte[90.0] > compte[99.0] > 0, (
        f"le percentile ne trie plus : {compte}")
    assert ampli[50.0] > ampli[90.0] > ampli[99.0] > 0.0, (
        f"le percentile ne module plus l'amplitude : {ampli}")
    # Epinglage des valeurs mesurees (voir l'en-tete du fichier).
    assert (compte[50.0], compte[90.0], compte[99.0]) == (2040, 404, 32)
    assert ampli[50.0] == pytest.approx(5.8788, rel=1e-3)
    assert ampli[99.0] == pytest.approx(1.7256e-2, rel=1e-3)


def test_au_dessus_du_critere_absolu_le_percentile_ne_sert_a_rien():
    """L'invariant qui rend ce parametre sur : des qu'une cellule franchit
    le critere physique, le comportement d'origine est conserve A
    L'IDENTIQUE, quelle que soit la valeur entrainee."""
    signal = np.linspace(0.0, 100.0, 1001)
    crit_absolu = 10.0            # atteint : l'absolu tire
    for p in (50.0, 75.0, 90.0, 99.0):
        assert PhysicalMapper(relative_percentile=p)._effective_crit(
            signal, crit_absolu) == crit_absolu


def test_le_defaut_est_un_NO_OP_bit_a_bit():
    """Le chemin par defaut doit etre INCHANGE, pas « equivalent ».

    `_effective_crit` est passee de `@classmethod` a methode d'instance :
    tout appelant qui ne passe rien doit obtenir exactement ce qu'il
    obtenait avant. Trois tests de la suite QAOA ont echoue lors du
    passage de recette suivant cette modification ; ils passent tous a la
    reexecution, et le bras QAOA n'est seme NULLE PART dans `src/VQA/`
    (`tests/quantum/test_qaoa_arm_is_sampled.py` l'epingle). Ce test
    ferme l'autre explication -- que la valeur par defaut ait bouge --
    par une comparaison bit-a-bit plutot que par un raisonnement.
    """
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver
    from Simulation.PhysToAngle import AngleMapper

    N, RE, RM = 64, 800, 800
    HP = dict(gamma_hydro=2.1272, gamma_mag=2.3611, kappa=14.3321,
              sigma=0.05, beta_curl=0.8199, beta_xpoint=0.4256,
              w_z_frac=0.1013)
    grid = PeriodicGrid(N)
    sim = MHDSolver(grid, dt=1e-3, Re=RE, Rm=RM)
    x = np.arange(N) * grid.dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    k = 2.0 * np.pi / grid.L
    sim.vx = -np.cos(k * X) * np.sin(k * Y)
    sim.vy = np.sin(k * X) * np.cos(k * Y)
    sim.Bx = np.ones_like(X)
    sim.By = np.zeros_like(X)
    etat = sim.get_fluxes()
    score = AngleMapper.classical_score(etat)

    def coeffs(**extra):
        m = PhysicalMapper(cs=1.0, nu=grid.L / RE, eta_mhd=grid.L / RM,
                           dx=grid.dx, **HP, **extra)
        return m.compute_coefficients(sim, score, etat, threshold_amr=0.5,
                                      advanced_anomalies_enabled=True)

    defaut = coeffs()                                          # rien de passe
    explicite = coeffs(relative_percentile=                    # l'ancienne
                       PhysicalMapper.RELATIVE_PERCENTILE)     # constante

    assert set(defaut) == set(explicite)
    for cle in defaut:
        a, b = np.asarray(defaut[cle]), np.asarray(explicite[cle])
        assert a.shape == b.shape, cle
        assert np.array_equal(a, b), (
            f"`{cle}` differe entre le defaut et la constante de classe : "
            f"le chemin par defaut n'est PAS un no-op.")


# ══════════════════════════════════════════════════════════════════
#  4. Le garde COMPORTEMENTAL — ce que l'AST ne peut pas voir
# ══════════════════════════════════════════════════════════════════

class _SentinelleMappeur(Exception):
    """Interrompt `pipeline` des la construction du mappeur."""
    def __init__(self, kwargs):
        self.kwargs = kwargs
        super().__init__("sentinelle")


def test_la_valeur_ECHANTILLONNEE_atteint_vraiment_le_mappeur(monkeypatch):
    """Le seul garde de ce fichier que la mutation A' ne survit pas.

    Les gardes AST ci-dessus verifient que le NOM est lu et que
    l'argument est PRESENT. Ils ne verifient pas que la VALEUR passee
    est celle qui a ete lue. Mesure, mutation appliquee a `pipeline.py` :

        PhysicalMapper(..., relative_percentile=90.0)   # valeur figee
        -> pytest tests/pipeline/test_relative_percentile_is_trainable.py
        -> 20 passed

    Vingt tests verts pendant qu'Optuna echantillonne un parametre que la
    pipeline remplace par une constante : D-31 exactement, sous la forme
    meme que ce fichier existe pour empecher. C'est la famille de faux
    vert que la branche `vigil/…` mesure en D-123 a D-131 (« garde par une
    chaine », « mutation A' reste VERTE ») ; le remede est le meme —
    interroger le COMPORTEMENT, pas le texte.

    Ici : on remplace `PhysicalMapper` dans l'espace de noms de `pipeline`
    par une sentinelle qui capture ses arguments et leve. La valeur
    capturee doit etre celle passee dans `hyperparams`.
    """
    import pipeline as P

    vu = {}

    def _faux_mappeur(*a, **kw):
        vu.update(kw)
        raise _SentinelleMappeur(kw)

    monkeypatch.setattr(P, "PhysicalMapper", _faux_mappeur)

    argus = SimpleNamespace(
        eta=0.001, Bz_guide=0.1, c_s=1.0, Re=800, Rm=800, shots=64,
        mode="simulator", backend="state_vector", method="COBYLA",
        opt_level=1, AdvAnomaliesEnable=True, K_opt=2, eps=1e-2, reps=1)

    SENTINELLE = 63.5          # valeur qu'aucun defaut du depot ne porte
    with pytest.raises(_SentinelleMappeur):
        P.pipeline(N=8, VQA_N=2, T_MAX=0.002, DT=1e-3, HYBRID=1,
                   verbose=False, argus=argus,
                   hyperparams={"relative_percentile": SENTINELLE},
                   scenario="harris_tearing", max_depth_override=1)

    assert "relative_percentile" in vu, (
        "PhysicalMapper construit sans relative_percentile : la valeur "
        "echantillonnee n'atteint pas le mappeur.")
    assert vu["relative_percentile"] == SENTINELLE, (
        f"la pipeline a passe {vu['relative_percentile']!r} au lieu de "
        f"{SENTINELLE!r} : la valeur echantillonnee est REMPLACEE en "
        f"chemin. Optuna optimiserait un parametre que rien ne lit (D-31).")
