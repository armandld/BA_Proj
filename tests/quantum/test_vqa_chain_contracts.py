"""Audit de contrat de la chaine quantique : l'indice d'un qubit designe-t-il
toujours la meme cellule d'un bout a l'autre ?

Quatre etapes se passent un indice de qubit sans que personne ne verifie
l'accord :

  score (i,j)  --init_qbits_state-->  qubit k
  qubit k      --cost_hamiltonian-->  terme de Pauli en position k
  qubit k      --Statevector-->       caractere k de la chaine, DEPUIS LA DROITE
  chaine       --postprocess-->       marginales[k]

Si l'une des quatre conventions se retourne, la carte de decision revient
SPATIALEMENT MIROIR. Elle reste une carte plausible : meme taille, memes
valeurs, meme fraction raffinee. Aucun test de valeur ne peut la distinguer
d'une carte juste — seule une trace de bout en bout le peut.

Ces tests posent la trace : on force un qubit connu et on verifie qu'il
ressort a la bonne place, en passant par le VRAI chemin.

DEFAUT TROUVE ICI : `postprocess` acceptait n'importe quoi. Des comptes
bruts au lieu d'une distribution auraient donne des « marginales » de
l'ordre du millier, que toute comparaison au seuil (~0.15) aurait declarees
actives — un domaine entierement raffine, indiscernable d'une detection.
Une chaine multi-registres aurait decale toutes les positions apres
l'espace. Les deux sont desormais refuses.
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


_SRC = os.path.join(_repo_root(), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

from VQA.init_qbits_state import init_qbits_state  # noqa: E402
from VQA.postprocess import postprocess  # noqa: E402


def _marginals(qc):
    d = Statevector.from_instruction(qc).probabilities_dict()
    return np.array(postprocess(d, qc.num_qubits, False))


# ======================================================================
#  1. La trace de bout en bout : le qubit k ressort en position k
# ======================================================================

@pytest.mark.parametrize("k", range(8))
def test_a_single_excited_qubit_comes_back_at_its_own_index(k):
    """Le test qui attraperait une inversion de convention de bits."""
    n = 8
    qc = QuantumCircuit(n)
    qc.x(k)
    m = _marginals(qc)
    assert m[k] == pytest.approx(1.0)
    assert np.sum(m) == pytest.approx(1.0), (
        f"un seul qubit excite mais {np.sum(m)} marginales actives : "
        "la convention de bits ne colle pas")


def test_the_bit_order_is_qiskit_little_endian_not_the_reverse():
    """Qiskit ecrit le qubit 0 A DROITE. Le commentaire du code disait
    l'inverse ; c'est le code qui a raison."""
    assert postprocess({"0001": 1.0}, 4, False) == [1.0, 0.0, 0.0, 0.0]
    assert postprocess({"1000": 1.0}, 4, False) == [0.0, 0.0, 0.0, 1.0]


def test_marginals_are_probabilities_and_stay_in_the_unit_interval():
    n = 6
    qc = QuantumCircuit(n)
    for k in range(n):
        qc.ry(0.7 * (k + 1), k)
    m = _marginals(qc)
    assert np.all(m >= 0.0) and np.all(m <= 1.0)


def test_a_uniform_superposition_gives_one_half_everywhere():
    n = 5
    qc = QuantumCircuit(n)
    for k in range(n):
        qc.h(k)
    assert np.allclose(_marginals(qc), 0.5, atol=1e-12)


def test_the_ground_state_gives_zero_everywhere():
    assert np.allclose(_marginals(QuantumCircuit(4)), 0.0)


# ======================================================================
#  2. L'encodage d'angle : theta doit rendre EXACTEMENT le score
# ======================================================================

def test_the_initial_state_reproduces_the_classical_score_exactly():
    """P(|1>) = sin^2(theta/2) = score. C'est la promesse de l'architecture :
    le QAOA part de la meme decision que le detecteur classique."""
    score = np.array([[0.0, 0.13], [0.5, 1.0]])
    th = 2.0 * np.arcsin(np.sqrt(score))
    z = np.zeros_like(th)
    qc = init_qbits_state(th, th.copy(), z, z.copy())
    m = _marginals(qc)
    assert np.allclose(m[:4], score.ravel(), atol=1e-12)
    assert np.allclose(m[4:], score.ravel(), atol=1e-12)


@pytest.mark.parametrize("psi_val", [0.0, 0.7, -1.3, np.pi / 2])
def test_psi_changes_the_phase_and_never_the_probability(psi_val):
    """psi encode une phase. S'il deplacait P(|1>), il deplacerait la
    decision — et l'ablation psi ne mesurerait plus ce qu'elle croit."""
    score = np.array([[0.1, 0.9], [0.5, 0.25]])
    th = 2.0 * np.arcsin(np.sqrt(score))
    psi = np.full_like(th, psi_val)
    qc = init_qbits_state(th, th.copy(), psi, psi.copy())
    assert np.allclose(_marginals(qc)[:4], score.ravel(), atol=1e-12)


def test_psi_does_change_the_state_even_though_it_leaves_probabilities_alone():
    """Sinon psi serait purement decoratif AVANT meme le QAOA, et
    l'ablation ne pourrait rien mesurer."""
    score = np.full((2, 2), 0.5)
    th = 2.0 * np.arcsin(np.sqrt(score))
    z = np.zeros_like(th)
    a = Statevector.from_instruction(init_qbits_state(th, th.copy(), z, z.copy()))
    b = Statevector.from_instruction(
        init_qbits_state(th, th.copy(), np.full_like(th, 1.1),
                         np.full_like(th, 1.1)))
    assert not np.allclose(a.data, b.data, atol=1e-9)


def test_the_two_qubit_families_occupy_disjoint_index_ranges():
    """La famille h occupe [0, dim^2), la famille v [dim^2, 2 dim^2).
    Un chevauchement ferait ecrire deux cellules sur le meme qubit."""
    dim = 3
    hot = np.zeros((dim, dim))
    hot[1, 2] = 1.0
    cold = np.zeros((dim, dim))
    th_h = 2.0 * np.arcsin(np.sqrt(hot))
    th_v = 2.0 * np.arcsin(np.sqrt(cold))
    z = np.zeros((dim, dim))
    m = _marginals(init_qbits_state(th_h, th_v, z, z.copy()))
    assert m[1 * dim + 2] == pytest.approx(1.0)
    assert np.allclose(np.delete(m, 1 * dim + 2), 0.0, atol=1e-12)


def test_the_circuit_has_exactly_two_qubits_per_cell():
    for dim in (2, 3, 4):
        z = np.zeros((dim, dim))
        qc = init_qbits_state(z, z.copy(), z.copy(), z.copy())
        assert qc.num_qubits == 2 * dim * dim


def test_the_row_major_flattening_is_the_one_the_hamiltonian_assumes():
    """`idx_H(i,j) = i*dim + j` cote Hamiltonien ; `flatten()` cote circuit.
    Les deux doivent designer la meme cellule."""
    dim = 3
    for i in range(dim):
        for j in range(dim):
            hot = np.zeros((dim, dim))
            hot[i, j] = 1.0
            z = np.zeros((dim, dim))
            m = _marginals(init_qbits_state(2.0 * np.arcsin(np.sqrt(hot)),
                                            z, z.copy(), z.copy()))
            assert m[i * dim + j] == pytest.approx(1.0)


# ======================================================================
#  3. Le garde de postprocess
# ======================================================================

def test_raw_counts_are_refused_instead_of_becoming_giant_marginals():
    """Le defaut : des comptes bruts passaient pour des probabilites."""
    with pytest.raises(ValueError, match="normalisee"):
        postprocess({"01": 512, "10": 512}, 2, False)


def test_the_error_says_what_the_sum_was():
    try:
        postprocess({"01": 512, "10": 512}, 2, False)
    except ValueError as e:
        assert "1024" in str(e)


def test_a_multi_register_bitstring_is_refused():
    with pytest.raises(ValueError, match="multi-registres"):
        postprocess({"01 10": 1.0}, 4, False)


def test_a_bitstring_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="longueur"):
        postprocess({"010": 1.0}, 4, False)


def test_an_empty_distribution_is_refused():
    with pytest.raises(ValueError, match="vide"):
        postprocess({}, 4, False)


def test_a_normalised_distribution_still_goes_through():
    """Un garde qui refuse tout serait aussi inutile qu'un garde absent."""
    m = postprocess({"01": 0.25, "10": 0.75}, 2, False)
    assert m == [0.25, 0.75]


def test_the_marginals_of_a_mixture_are_the_weighted_sums():
    d = {"00": 0.1, "01": 0.2, "10": 0.3, "11": 0.4}
    m = postprocess(d, 2, False)
    assert m[0] == pytest.approx(0.2 + 0.4)
    assert m[1] == pytest.approx(0.3 + 0.4)


def test_the_tolerance_accepts_ordinary_floating_point_drift():
    """1e-9 de derive sur une somme de 2^12 termes ne doit pas crier."""
    n = 12
    d = {format(k, f"0{n}b"): 1.0 / 2 ** n for k in range(2 ** n)}
    m = postprocess(d, n, False)
    assert np.allclose(m, 0.5, atol=1e-9)


# ======================================================================
#  4. optimize() : le niveau demande est-il celui applique ?
# ======================================================================

def test_the_state_vector_backend_silently_forces_optimisation_level_zero():
    """Fige un comportement non documente : un appelant qui demande 3
    obtient 0. Ce n'est pas faux — un statevector n'a rien a transpiler —
    mais cela doit etre visible plutot que devine.

    D-174 : la version precedente cherchait les chaines litterales
    "opt_level = 0" et "state_vector" dans le SOURCE de `optimize()`, pas
    son comportement. Une mutation qui rend le forcage mort (`if False:
    opt_level = 0`, texte inchange) la laissait verte — voir
    `docs/RESULTS.md`. Celle-ci appelle `optimize()` pour de vrai et lit le
    `optimization_level` REELLEMENT transmis au pass manager.
    """
    from VQA import optimize as opt_mod

    captured = {}

    def _fake_pass_manager(optimization_level, backend):
        captured["level"] = optimization_level

        class _NoOpPM:
            def run(self, qc):
                return qc
        return _NoOpPM()

    orig = opt_mod.generate_preset_pass_manager
    opt_mod.generate_preset_pass_manager = _fake_pass_manager
    try:
        opt_mod.optimize(QuantumCircuit(1), "state_vector", 3, False)
    finally:
        opt_mod.generate_preset_pass_manager = orig

    assert captured["level"] == 0, (
        f"state_vector a transmis optimization_level={captured['level']!r} "
        "au pass manager alors que 0 est attendu, quel que soit le niveau "
        "demande par l'appelant"
    )


def test_a_non_state_vector_backend_keeps_the_requested_optimisation_level():
    """Champ qui SEPARE : `aer` ne doit PAS subir le meme forcage que
    `state_vector`, sinon le test precedent ne distinguerait rien."""
    from VQA import optimize as opt_mod

    captured = {}

    def _fake_pass_manager(optimization_level, backend):
        captured["level"] = optimization_level

        class _NoOpPM:
            def run(self, qc):
                return qc
        return _NoOpPM()

    orig = opt_mod.generate_preset_pass_manager
    opt_mod.generate_preset_pass_manager = _fake_pass_manager
    try:
        opt_mod.optimize(QuantumCircuit(1), "aer", 3, False)
    finally:
        opt_mod.generate_preset_pass_manager = orig

    assert captured["level"] == 3, (
        f"aer a transmis optimization_level={captured['level']!r} : le "
        "niveau demande par l'appelant doit passer inchange sur ce backend"
    )


def test_an_unknown_backend_is_refused_instead_of_silently_defaulting():
    from VQA.optimize import optimize
    with pytest.raises(ValueError, match="Unsupported backend"):
        optimize(QuantumCircuit(1), "un backend qui n'existe pas", 0, False)


# ======================================================================
#  5. Ce que le circuit PEUT deplacer, et par quel canal
# ======================================================================
#
# La couche de cout exp(-i gamma H) est DIAGONALE : elle n'ajoute que des
# phases, elle ne peut changer aucune probabilite de mesure. Seul le mixeur
# exp(-i beta sum X) deplace P(|1>). Et beta est borne a pi/(4 reps).
#
# Consequence structurelle : tout ce que l'Hamiltonien apporte a la DECISION
# passe par son interaction avec le mixeur. Ces tests le figent, et bornent
# de combien.

_REPS = 2
_BETA_MAX = np.pi / (4 * _REPS)


def _deployed_circuit(reps=_REPS, dim=2, seed=0):
    from Simulation.HamiltParams_v2 import PhysicalMapperV2
    from Simulation.PhysToAngle import AngleMapper
    from Simulation.grid import curl_z
    from VQA.mapping import mapping

    p = dim + 2
    rng = np.random.default_rng(seed)
    f = {k: rng.normal(size=(p, p)) for k in ("vx", "vy", "Bx", "By")}
    f["Jz"] = curl_z(f["Bx"], f["By"], True)
    sc = AngleMapper.classical_score(f)
    params = PhysicalMapperV2(dx=0.02).compute_coefficients(None, sc, f, 0.1496)
    th = 2.0 * np.arcsin(np.sqrt(np.clip(sc, 0.0, 1.0)))
    z = np.zeros((p, p))
    qc, _ = mapping({"theta_h": th, "theta_v": th.copy(),
                     "psi_h": z, "psi_v": z.copy()},
                    params, False, period_bound=False, reps=reps)
    return qc, sc


def _marg_of(qc, x):
    return np.array(postprocess(
        Statevector.from_instruction(qc.assign_parameters(x)).probabilities_dict(),
        qc.num_qubits, False))


def test_the_qaoa_parameter_order_is_all_betas_then_all_gammas():
    """Contrat inter-bibliotheque : `execute` construit x0 comme
    [zeros(reps), rampe_gamma]. Si une version de Qiskit reordonnait les
    parametres de QAOAAnsatz, la rampe s'appliquerait au MIXEUR et la borne
    beta au terme de cout — sans la moindre erreur."""
    for reps in (1, 2, 3):
        qc, _ = _deployed_circuit(reps=reps)
        names = [q.name for q in qc.parameters]
        assert names == [f"β[{i}]" for i in range(reps)] + \
                        [f"γ[{i}]" for i in range(reps)], names


def test_zero_parameters_reproduce_the_classical_initialisation_exactly():
    """C'est la porte de sortie du raccourci « Hamiltonien nul » : rendre
    les marginales de theta-init. Elle doit etre exacte, pas approchee."""
    qc, sc = _deployed_circuit()
    m = _marg_of(qc, np.zeros(2 * _REPS))
    assert np.allclose(m[:4], sc[1:-1, 1:-1].ravel(), atol=1e-12)


@pytest.mark.parametrize("gamma", [0.3, 1.0, 3.0, 10.0, 2 * np.pi])
def test_gamma_alone_moves_no_probability_at_all(gamma):
    """L'Hamiltonien est diagonal : sa couche n'ajoute que des phases."""
    qc, _ = _deployed_circuit()
    base = _marg_of(qc, np.zeros(2 * _REPS))
    x = np.concatenate([np.zeros(_REPS), np.full(_REPS, gamma)])
    assert np.max(np.abs(_marg_of(qc, x) - base)) < 1e-12, (
        "gamma a deplace une probabilite : la couche de cout ne serait plus "
        "diagonale, et l'enumeration exhaustive de l'etat fondamental — sur "
        "laquelle repose toute la campagne — cesserait d'etre valide")


def test_the_mixer_is_the_only_channel_that_moves_the_decision():
    """Meme constat, formule dans l'autre sens : a beta fixe, balayer gamma
    de 0 a 2 pi ne bouge rien tant que beta vaut zero."""
    qc, _ = _deployed_circuit(seed=3)
    base = _marg_of(qc, np.zeros(2 * _REPS))
    worst = max(
        np.max(np.abs(_marg_of(qc, np.concatenate(
            [np.zeros(_REPS), np.full(_REPS, g)])) - base))
        for g in np.linspace(0.0, 2 * np.pi, 25))
    assert worst < 1e-12


def test_beta_is_bounded_and_the_bound_is_the_documented_one():
    import inspect

    from VQA import execute as ex
    src = inspect.getsource(ex.execute)
    assert "beta_max = np.pi / (4 * reps)" in src
    assert np.pi / (4 * 2) == pytest.approx(0.3927, abs=1e-4)


def test_the_hamiltonian_can_only_act_through_the_mixer_and_by_how_much():
    """Borne superieure de ce que le circuit peut deplacer, et part
    attribuable a l'Hamiltonien.

    On balaie toute la grille admissible (beta borne, gamma libre) — c'est
    donc ce qu'un optimiseur PARFAIT atteindrait, pas ce que COBYLA trouve.

    Trois quantites : le mixeur seul (gamma=0), le mixeur avec
    l'Hamiltonien, et la difference. Si la difference tombait a zero,
    l'Hamiltonien serait inerte et la campagne mesurerait une rotation de
    mixeur.
    """
    qc, _ = _deployed_circuit(seed=5)
    base = _marg_of(qc, np.zeros(2 * _REPS))
    betas = np.linspace(-_BETA_MAX, _BETA_MAX, 13)
    gammas = np.linspace(0.0, 2 * np.pi, 17)
    mixer_only = max(
        np.max(np.abs(_marg_of(qc, np.concatenate(
            [np.full(_REPS, b), np.zeros(_REPS)])) - base)) for b in betas)
    both = 0.0
    for b in betas:
        for g in gammas:
            both = max(both, np.max(np.abs(_marg_of(qc, np.concatenate(
                [np.full(_REPS, b), np.full(_REPS, g)])) - base)))
    assert both >= mixer_only - 1e-12
    assert mixer_only > 0.01, "le mixeur borne ne deplace rien du tout"
    assert both - mixer_only > 0.01, (
        "l'Hamiltonien n'apporte rien au-dela d'une rotation de mixeur : "
        "le circuit mesurerait alors le mixeur, pas la physique")
    assert both < 1.0, "le deplacement ne peut pas exceder une probabilite"
