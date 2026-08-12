"""La variante `--fixed-curl` doit etre une variante, pas une correction.

Le rotationnel et la divergence des mappeurs sont ecrits sous la convention
indexing='xy' alors que `grid.py` declare indexing='ij'. Corriger le chemin
par defaut invaliderait la campagne d'hyperparametres (une semaine de calcul
Optuna, `results/hyperparams/PROVENANCE.md`), qui a regle ses valeurs sur le
chemin historique. Le drapeau permet donc de MESURER l'ecart sans le
supprimer.

Ces tests verrouillent les trois proprietes qui rendent la mesure honnete :

  1. sans le drapeau, la chaine d'entree du QAOA est bit-a-bit celle
     d'avant ;
  2. avec le drapeau, elle change reellement — un drapeau accepte puis
     ignore serait indiscernable de son absence, et c'est exactement le
     defaut que cette etude traque ;
  3. l'artefact produit porte un nom distinct, donc les deux variantes ne
     peuvent pas s'ecraser l'une l'autre.
"""

import os
import subprocess
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
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

N = 32
DIM = 2
RE = 400

_PANEL = os.path.join(_REPO_ROOT, "study", "h0_selection",
                      "h0_optimiser_equivalence.py")


@pytest.fixture(scope="module")
def snapshot():
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver

    sim = MHDSolver(PeriodicGrid(N), dt=1e-3, Re=RE, Rm=RE)
    sim.init_orszag_tang()
    for _ in range(20):
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)
    prev = {k: np.array(v) for k, v in sim.get_fluxes().items()}
    for _ in range(5):
        sim.adapt_dt(cfl_target=0.4)
        sim.step_full(record_stats=False)
    return sim.get_fluxes(), prev


def _inputs(cur, fixed_curl, prev=None, with_psi=False):
    from qaoa_inputs import prepare_qaoa_inputs
    return prepare_qaoa_inputs(
        cur["vx"], cur["vy"], cur["Bx"], cur["By"], N, DIM, RE,
        prev_fields=prev, with_psi=with_psi, fixed_curl=fixed_curl)


def test_the_flag_defaults_to_off(snapshot):
    """Le defaut doit rester le chemin historique."""
    import inspect

    from qaoa_inputs import prepare_qaoa_inputs
    sig = inspect.signature(prepare_qaoa_inputs)
    assert sig.parameters["fixed_curl"].default is False

    cur, _prev = snapshot
    a_in, a_hp, a_sc = _inputs(cur, fixed_curl=False)
    b_in, b_hp, b_sc = prepare_qaoa_inputs(
        cur["vx"], cur["vy"], cur["Bx"], cur["By"], N, DIM, RE)
    np.testing.assert_array_equal(a_sc, b_sc)
    for k in a_in:
        np.testing.assert_array_equal(np.asarray(a_in[k]), np.asarray(b_in[k]))


def test_the_flag_changes_the_classical_score(snapshot):
    """Le score classique passe par la vorticite : il doit bouger."""
    cur, _prev = snapshot
    _a, _ahp, sc_off = _inputs(cur, fixed_curl=False)
    _b, _bhp, sc_on = _inputs(cur, fixed_curl=True)
    assert not np.array_equal(sc_off, sc_on)
    assert np.max(np.abs(np.asarray(sc_off) - np.asarray(sc_on))) > 1e-3, (
        "le drapeau ne deplace pratiquement pas le score : il ne branche rien")


def test_the_flag_changes_the_hamiltonian_coefficients(snapshot):
    """K_plaquettes est le canal ou entre le rotationnel."""
    cur, _prev = snapshot
    _a, hp_off, _ = _inputs(cur, fixed_curl=False)
    _b, hp_on, _ = _inputs(cur, fixed_curl=True)
    assert "K_plaquettes" in hp_off
    k_off = np.asarray(hp_off["K_plaquettes"], dtype=float)
    k_on = np.asarray(hp_on["K_plaquettes"], dtype=float)
    assert k_off.shape == k_on.shape
    assert not np.array_equal(k_off, k_on)


def test_the_flag_reaches_theta_through_the_pipeline_encoder(snapshot):
    """Les deux variantes se composent, et chacune touche ce qu'elle doit.

    Avec --with-psi, les angles ne sont plus calcules ici mais delegues a
    `refinement._prepare_vqa_input`. Il faut donc verifier que fixed_curl
    traverse cette delegation : theta derive du score classique, qui passe
    par la vorticite, donc theta doit bouger.

    psi, lui, ne doit PAS bouger : il encode la derivee temporelle du flux
    de contrainte, que `_compute_filtered_flux` forme a partir des sauts de
    champ entre cellules voisines (compression, cisaillement, saut de Jz).
    Aucun rotationnel n'y entre. Un psi qui changerait avec la convention
    d'axes signalerait que le drapeau fuit dans un canal ou il n'a rien a
    faire.
    """
    cur, prev = snapshot
    off, _, _ = _inputs(cur, fixed_curl=False, prev=prev, with_psi=True)
    on, _, _ = _inputs(cur, fixed_curl=True, prev=prev, with_psi=True)

    th_off = np.asarray(off["theta_h"], dtype=float)
    th_on = np.asarray(on["theta_h"], dtype=float)
    assert not np.array_equal(th_off, th_on), (
        "fixed_curl n'atteint pas l'encodeur du pipeline : theta est "
        "identique dans les deux conventions")
    assert np.max(np.abs(th_off - th_on)) > 1e-3

    psi_off = np.asarray(off["psi_h"], dtype=float)
    psi_on = np.asarray(on["psi_h"], dtype=float)
    assert np.max(np.abs(psi_off)) > 1e-12, "psi doit etre non nul avec --with-psi"
    np.testing.assert_array_equal(psi_off, psi_on)


def test_the_stress_flux_really_is_curl_free(snapshot):
    """Justifie l'invariance de psi ci-dessus au lieu de la postuler.

    Si `compute_stress_flux` venait a dependre du rotationnel, le test
    precedent deviendrait faux sans que rien ne le signale.
    """
    from Simulation.PhysToAngle import AngleMapper

    cur, _prev = snapshot
    mapper = AngleMapper()
    phi = mapper.compute_stress_flux(cur)
    # Le flux de contrainte ne lit que les champs, pas leur rotationnel :
    # une rotation solide superposee ne doit rien changer aux SAUTS entre
    # cellules... mais elle change les champs. On verifie donc directement
    # que la source ne mentionne aucun operateur de rotationnel.
    import inspect
    src = inspect.getsource(AngleMapper.compute_stress_flux)
    src += inspect.getsource(AngleMapper._compute_filtered_flux)
    for token in ("curl_z", "forward_curl", "legacy_forward_curl"):
        assert token not in src, (
            f"compute_stress_flux utilise {token} : psi depend desormais de "
            "la convention d'axes et le test d'invariance ci-dessus ment")
    assert set(phi) >= {"phi_horizontal", "phi_vertical"}


def test_the_panel_exposes_the_flag_and_suffixes_its_artefact():
    """Sans suffixe distinct, les deux variantes s'ecrasent (defaut D9)."""
    src = open(_PANEL, encoding="utf-8").read()
    assert '"--fixed-curl", action="store_true"' in src
    assert '+ ("_fixedcurl" if args.fixed_curl else "")' in src
    assert "fixed_curl=args.fixed_curl" in src, (
        "le drapeau doit atteindre solver_panel, sinon il ne fait que "
        "renommer le fichier de sortie")


def test_the_panel_accepts_the_flag_end_to_end(tmp_path):
    """Le drapeau doit survivre a un vrai passage en ligne de commande."""
    r = subprocess.run(
        [sys.executable, _PANEL, "--help"],
        capture_output=True, text=True, cwd=_REPO_ROOT, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "--fixed-curl" in r.stdout
