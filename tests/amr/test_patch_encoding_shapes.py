"""D-37 — le biais Z et les couplages decrivaient des grilles differentes.

A depth > 0, `_prepare_vqa_input` demandait a `_process_score` une taille
`target_dim + 2 * pad`. Or `_process_score` emprunte alors
`_resize_padded_maxpool`, dont le contrat est « entree (N+2, M+2) ->
sortie (t_dim+2, t_dim+2) » : le halo est deja ajoute par la fonction.
L'appelant l'ajoutait une SECONDE fois.

Consequence : pour un coeur 2x2, le score rendait (6, 6) la ou les champs
rendaient (4, 4). `H_edges` (biais Z, bati sur le score) et `C_edges` /
`K_plaquettes` (batis sur les champs) ne decrivaient plus le meme patch.
`create_bounded_hamiltonian(dim=2)` lisait le coin superieur gauche du
(6, 6) : le biais Z venait du quart haut-gauche du patch, plus un halo
situe deux cellules trop loin.

Present depuis le premier commit du fichier. Tous les niveaux de
raffinement SAUF le premier passent par la.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from Simulation.RescaleArrays import _process_score, _resize_padded_maxpool
from Simulation.grid import PeriodicGrid
from Simulation.HamiltParams import PhysicalMapper
from Simulation.PhysToAngle import AngleMapper
from Simulation.refinement import _downsample_fields, run_adaptive_vqa
from Simulation.solver import MHDSolver
from Simulation.utils import get_periodic_patch


TARGET_DIM = 2


@pytest.fixture(scope="module")
def flow():
    """Un ecoulement developpe : les coefficients doivent etre non nuls."""
    grid = PeriodicGrid(resolution_N=64)
    sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
    sim.init_orszag_tang()
    for _ in range(40):
        sim.step_full()
    return sim


def _mapper(sim):
    return PhysicalMapper(
        cs=1.0, nu=sim.grid.L / 800, eta_mhd=sim.grid.L / 800, dx=sim.grid.dx,
        gamma_hydro=2.0, gamma_mag=0.5, kappa=10.0, sigma=0.05,
        beta_curl=1.0, beta_xpoint=1.0, w_z_frac=0.15)


# ══════════════════════════════════════════════════════════════════
#  1. Le contrat de la reduction, epingle
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("t_dim", [2, 3, 4])
def test_the_bounded_reduction_adds_the_halo_itself(t_dim):
    """C'est LA propriete que l'appelant avait oubliee."""
    arr = np.arange((t_dim * 3 + 2) ** 2, dtype=float).reshape(
        t_dim * 3 + 2, t_dim * 3 + 2)
    out = _process_score(arr, False, t_dim)
    assert out.shape == (t_dim + 2, t_dim + 2)
    assert _resize_padded_maxpool(arr, t_dim).shape == (t_dim + 2, t_dim + 2)


def test_the_periodic_reduction_adds_no_halo():
    arr = np.arange(64.0).reshape(8, 8)
    assert _process_score(arr, True, TARGET_DIM).shape == (TARGET_DIM, TARGET_DIM)


def test_asking_for_the_halo_twice_gives_a_different_grid():
    """Epinglage de l'ancien appel. S'il redevenait correct, la
    distinction n'aurait plus lieu d'etre et ce test doit etre remesure."""
    arr = np.arange(100.0).reshape(10, 10)
    pad = 1
    ancien = _process_score(arr, False, TARGET_DIM + 2 * pad)
    correct = _process_score(arr, False, TARGET_DIM)
    assert ancien.shape == (6, 6)
    assert correct.shape == (4, 4)
    assert ancien.shape != correct.shape


# ══════════════════════════════════════════════════════════════════
#  2. Les deux moities de l'Hamiltonien decrivent le meme patch
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("depth,bounds", [
    (0, (0, 64, 0, 64)),
    (1, (0, 32, 0, 32)),
    (2, (8, 24, 8, 24)),
])
def test_the_z_bias_and_the_couplings_share_one_grid(flow, depth, bounds):
    """Question 4, dans sa forme la plus directe : deux moities d'un meme
    Hamiltonien, batis par deux chemins, sur la meme grille ?

    `H_edges` vient du SCORE, `C_edges` et `K_plaquettes` viennent des
    CHAMPS. Rien en aval ne peut detecter qu'ils divergent : le
    Hamiltonien reste parfaitement valide, calcule sur la mauvaise
    portion du patch.
    """
    sim = flow
    pad = 1 if depth > 0 else 0
    score = AngleMapper.classical_score(sim.get_fluxes())
    local = get_periodic_patch(score, *bounds, pad)

    mini_fields = _downsample_fields(sim.get_fluxes(), *bounds, TARGET_DIM, pad=pad)
    mini_score = _process_score(local, depth == 0, TARGET_DIM)

    assert mini_score.shape == np.shape(mini_fields["vx"]), (
        f"depth={depth} : score {mini_score.shape} contre champs "
        f"{np.shape(mini_fields['vx'])}")

    hp = _mapper(sim).compute_coefficients(
        sim, mini_score, mini_fields, 0.15,
        advanced_anomalies_enabled=True,
        dx_override=(bounds[1] - bounds[0]) / 64 * sim.grid.L / TARGET_DIM)

    expected = (TARGET_DIM + 2 * pad, TARGET_DIM + 2 * pad)
    for name in ("C_edges", "H_edges"):
        for k in (0, 1):
            assert np.shape(hp[name][k]) == expected, f"{name}[{k}] @ depth={depth}"
    for name in ("K_plaquettes", "K_xpoint"):
        if hp.get(name) is not None:
            assert np.shape(hp[name]) == expected, f"{name} @ depth={depth}"


def test_the_old_call_produced_two_incompatible_halves(flow):
    """La mesure du defaut, figee. Sur `orszag_tang` apres 40 pas :
    `H_edges` (6, 6) contre `C_edges` (4, 4)."""
    sim = flow
    bounds, pad = (0, 32, 0, 32), 1
    score = AngleMapper.classical_score(sim.get_fluxes())
    local = get_periodic_patch(score, *bounds, pad)
    mini_fields = _downsample_fields(sim.get_fluxes(), *bounds, TARGET_DIM, pad=pad)

    ancien = _mapper(sim).compute_coefficients(
        sim, _process_score(local, False, TARGET_DIM + 2 * pad), mini_fields,
        0.15, advanced_anomalies_enabled=True, dx_override=sim.grid.dx)

    assert np.shape(ancien["H_edges"][0]) == (6, 6)
    assert np.shape(ancien["C_edges"][0]) == (4, 4)


def test_the_wrong_grid_changed_the_z_bias_by_a_large_fraction(flow):
    """Ce n'etait pas une difference de bord : jusqu'a 41 % du plus grand
    coefficient."""
    sim = flow
    bounds, pad = (0, 32, 0, 32), 1
    score = AngleMapper.classical_score(sim.get_fluxes())
    local = get_periodic_patch(score, *bounds, pad)
    mini_fields = _downsample_fields(sim.get_fluxes(), *bounds, TARGET_DIM, pad=pad)
    M = _mapper(sim)

    lu = M.compute_coefficients(
        sim, _process_score(local, False, TARGET_DIM + 2 * pad), mini_fields,
        0.15, advanced_anomalies_enabled=True,
        dx_override=sim.grid.dx)["H_edges"][0][:4, :4]
    correct = M.compute_coefficients(
        sim, _process_score(local, False, TARGET_DIM), mini_fields,
        0.15, advanced_anomalies_enabled=True,
        dx_override=sim.grid.dx)["H_edges"][0]

    ecart = float(np.max(np.abs(lu - correct)))
    echelle = float(np.max(np.abs(correct)))
    assert ecart > 0.2 * echelle, (
        f"ecart {ecart:.5f} pour une echelle {echelle:.5f} : si le defaut "
        "est devenu benin, remesurer plutot que d'abaisser le seuil")


# ══════════════════════════════════════════════════════════════════
#  3. La regression : le chemin borne s'execute
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("max_depth", [1, 2, 3])
def test_the_vqa_scan_reaches_every_depth(flow, max_depth):
    """A `max_depth=1` le VQA ne descend jamais sous depth 0, donc le
    chemin borne n'est jamais emprunte. C'est pourquoi le defaut a
    survecu : les configurations rapides ne le traversent pas.
    """
    from types import SimpleNamespace
    sim = flow
    args = SimpleNamespace(reps=2, mode="simulator", backend="state_vector",
                           shots=64, method="COBYLA", opt_level=1,
                           AdvAnomaliesEnable=True, K_opt=6, eps=1e-2,
                           eta=0.001, Bz_guide=0.1, c_s=1.0, Re=800, Rm=800)
    patches, _, _ = run_adaptive_vqa(
        sim, AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0),
        _mapper(sim), args, None, verbose=False, beta=1.0, threshold_amr=0.15,
        target_dim=TARGET_DIM, max_depth=max_depth, min_size=4)

    assert patches, "balayage vide"
    depths = {p["depth"] for p in patches}
    assert max(depths) <= max_depth
    total = sum((p["bounds"][1] - p["bounds"][0]) * (p["bounds"][3] - p["bounds"][2])
                for p in patches)
    assert total == 64 * 64, (
        f"le pavage couvre {total} cellules sur {64 * 64} : trou ou recouvrement")


# ══════════════════════════════════════════════════════════════════
#  4. `compute_coefficients` porte SA PROPRE version de D-37 — inerte
#     aujourd'hui, jamais mesuree
# ══════════════════════════════════════════════════════════════════

def test_the_resize_branch_reproduces_D37_one_layer_deeper(flow):
    """`HamiltParams.compute_coefficients` (src/Simulation/HamiltParams.py,
    ~ligne 536) accepte `score.shape != field_shape` et resize `score` par
    `scipy.ndimage.zoom` avant de calculer la fenetre d'incertitude — mais
    `H_edges` est alloue via `N_field = score.shape[0]`, AVANT ce resize,
    et rempli plus tard avec le `score` NON resize (`z_bias = alpha_z *
    (score - threshold_amr)`). Le resize corrige `C_edges`/`K_plaquettes`
    (bases sur les champs) mais pas `H_edges` (base sur le score) : c'est
    exactement D-37 (`test_the_old_call_produced_two_incompatible_halves`
    ci-dessus), une couche plus bas, a l'interieur meme de la fonction que
    D-37 avait corrigee cote appelant.

    INERTE AUJOURD'HUI, verifie ici et non dans `DEFAUTS.md` (regle
    d'arret du fichier) : les 7 sites d'appel du depot (`refinement.py`,
    et les 6 sous `study/`) construisent toujours `score` et `fields` a la
    MEME resolution — `test_the_z_bias_and_the_couplings_share_one_grid`
    ci-dessus le garantit deja pour `refinement.py`, le seul chemin
    deploye. Ce test pique la seule autre garantie : SI ce garde-fou
    sautait un jour (nouvel appelant, resolution du score decouplee de
    celle des champs), la branche de secours de `compute_coefficients` ne
    rattraperait rien — elle produirait un Hamiltonien dont `H_edges` et
    `C_edges`/`K_plaquettes` decrivent deux grilles differentes, en
    silence.
    """
    sim = flow
    bounds, pad, t_dim = (0, 32, 0, 32), 1, TARGET_DIM
    score = AngleMapper.classical_score(sim.get_fluxes())
    local = get_periodic_patch(score, *bounds, pad)
    mini_fields = _downsample_fields(sim.get_fluxes(), *bounds, t_dim, pad=pad)
    # score au format AVEC halo (6, 6) ; les champs sont sans lui (4, 4) --
    # exactement l'ecart d'avant D-37, mais fourni directement en entree
    # de compute_coefficients plutot que rattrape en amont par l'appelant.
    mismatched_score = _process_score(local, False, t_dim + 2 * pad)
    assert mismatched_score.shape == (6, 6)
    assert mini_fields["vx"].shape == (4, 4)

    hp = _mapper(sim).compute_coefficients(
        sim, mismatched_score, mini_fields, 0.15,
        advanced_anomalies_enabled=True, dx_override=sim.grid.dx)

    assert np.shape(hp["H_edges"][0]) == (6, 6), (
        "H_edges suit le score non resize : si ce nombre bouge, le "
        "resize interne a change de comportement, remesurer plutot que "
        "de mettre a jour l'attendu en aveugle")
    assert np.shape(hp["C_edges"][0]) == (4, 4)
    assert np.shape(hp["K_plaquettes"]) == (4, 4)
    assert np.shape(hp["H_edges"][0]) != np.shape(hp["C_edges"][0]), (
        "H_edges et C_edges decrivent deux grilles differentes : "
        "un Hamiltonien assemble a partir de ce retour lirait le coin "
        "d'un patch pour l'un et un autre patch pour l'autre, comme D-37"
    )
