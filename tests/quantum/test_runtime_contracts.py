"""Audit de contrat de `VQARuntime` : le contexte d'execution partage.

Un objet construit une fois et passe a chaque appel VQA. Deux defauts y
dormaient, tous deux de la meme famille : une valeur inutilisable qui se
laisse produire sans bruit.

D-19  Un `backend_name` inconnu laissait `_backend`, `_estimator` et
      `_sampler` a None, et le constructeur rendait la main SANS ERREUR. La
      panne ne surgissait que bien plus loin, dans `execute`, sous la forme
      d'un AttributeError sur NoneType — a des dizaines de lignes de sa
      cause. `execute` et `optimize` levent tous deux ValueError pour la
      meme valeur : les trois sites disaient trois choses differentes.

D-20  Le cache d'ansatz etait indexe sur `(num_qubits, period_bound, reps)`.
      Or l'ansatz QAOA encode `exp(-i gamma H)` : il depend de l'Hamiltonien
      TERME PAR TERME. Deux patchs de meme taille aux coefficients
      differents collisionnaient donc, et le second recevait l'ansatz
      construit pour le premier — il se voyait optimise contre la physique
      d'un autre patch, sans le moindre signal.

      `get_ansatz` n'est appele par aucun code du depot. C'etait un piege
      arme, pret a se declencher au premier branchement — la raison d'etre
      d'un audit de contrat plutot que d'un audit de couverture.
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

from qiskit.quantum_info import SparsePauliOp  # noqa: E402

from VQA.runtime import VQARuntime, _hamiltonian_fingerprint  # noqa: E402

_H_A = SparsePauliOp.from_list([("ZZ", -1.0), ("ZI", 0.5)])
_H_B = SparsePauliOp.from_list([("ZZ", 7.0), ("IZ", -3.0)])


def _rt(backend="state_vector"):
    return VQARuntime(backend, "simulator", 1024, 0)


# ── D-19 : un backend inconnu doit crier tout de suite ────────────────

def test_an_unknown_backend_is_refused_at_construction():
    with pytest.raises(ValueError, match="Unsupported backend"):
        VQARuntime("un backend qui n'existe pas", "simulator", 1024, 0)


def test_the_error_lists_the_backends_that_do_exist():
    """Un message qui ne dit pas quoi ecrire fait perdre l'heure suivante."""
    try:
        VQARuntime("inconnu", "simulator", 1024, 0)
    except ValueError as e:
        for name in ("state_vector", "matrix_product_state", "aer",
                     "estimator"):
            assert name in str(e)


@pytest.mark.parametrize("backend", ["state_vector", "aer"])
def test_a_valid_backend_still_builds_a_usable_runtime(backend):
    """Un garde qui refuse tout serait aussi inutile qu'un garde absent."""
    r = _rt(backend)
    assert r._backend is not None
    assert r.estimator is not None and r.sampler is not None


def test_the_primitives_carry_the_requested_shot_count():
    r = VQARuntime("state_vector", "simulator", 777, 0)
    assert r.estimator.options.default_shots == 777
    assert r.sampler.options.default_shots == 777


def test_no_valid_backend_ever_leaves_a_primitive_at_none():
    for b in ("state_vector", "matrix_product_state", "aer"):
        r = _rt(b)
        assert None not in (r._backend, r._estimator, r._sampler), b


# ── D-20 : le cache doit distinguer deux Hamiltoniens ─────────────────

def test_two_different_hamiltonians_do_not_share_an_ansatz():
    """Le defaut : le second patch heritait de l'ansatz du premier."""
    r = _rt()
    a = r.get_ansatz(_H_A, 2, 2, False)
    b = r.get_ansatz(_H_B, 2, 2, False)
    assert a is not b, (
        "meme ansatz pour deux Hamiltoniens differents : le second patch "
        "serait optimise contre la physique du premier")


def test_the_same_hamiltonian_still_hits_the_cache():
    """Sinon le cache ne servirait a rien et la course serait perdue."""
    r = _rt()
    assert r.get_ansatz(_H_A, 2, 2, False) is r.get_ansatz(_H_A, 2, 2, False)


def test_a_coefficient_change_alone_is_enough_to_miss_the_cache():
    """Meme support de Pauli, un seul coefficient different."""
    r = _rt()
    h1 = SparsePauliOp.from_list([("ZZ", -1.0)])
    h2 = SparsePauliOp.from_list([("ZZ", -1.5)])
    assert r.get_ansatz(h1, 2, 2, False) is not r.get_ansatz(h2, 2, 2, False)


@pytest.mark.parametrize("field,a,b", [
    ("reps", 1, 2),
    ("period_bound", True, False),
])
def test_the_other_key_components_still_separate(field, a, b):
    r = _rt()
    if field == "reps":
        x, y = r.get_ansatz(_H_A, a, 2, False), r.get_ansatz(_H_A, b, 2, False)
    else:
        x, y = r.get_ansatz(_H_A, 2, 2, a), r.get_ansatz(_H_A, 2, 2, b)
    assert x is not y


def test_invalidating_the_cache_really_empties_it():
    r = _rt()
    first = r.get_ansatz(_H_A, 2, 2, False)
    r.invalidate_ansatz_cache()
    assert r.get_ansatz(_H_A, 2, 2, False) is not first


def test_the_cache_does_not_grow_on_repeated_identical_calls():
    r = _rt()
    for _ in range(5):
        r.get_ansatz(_H_A, 2, 2, False)
    assert len(r._ansatz_cache) == 1


# ── L'empreinte elle-meme ─────────────────────────────────────────────

def test_the_fingerprint_is_hashable_and_order_independent():
    a = SparsePauliOp.from_list([("ZZ", 1.0), ("ZI", 2.0)])
    b = SparsePauliOp.from_list([("ZI", 2.0), ("ZZ", 1.0)])
    assert hash(_hamiltonian_fingerprint(a))
    assert _hamiltonian_fingerprint(a) == _hamiltonian_fingerprint(b), (
        "l'ordre des termes n'est pas une difference physique")


def test_the_fingerprint_separates_a_sign_flip():
    a = SparsePauliOp.from_list([("ZZ", 1.0)])
    b = SparsePauliOp.from_list([("ZZ", -1.0)])
    assert _hamiltonian_fingerprint(a) != _hamiltonian_fingerprint(b)


def test_the_fingerprint_tolerates_last_bit_noise():
    """Sinon le cache exploserait sur des Hamiltoniens physiquement egaux."""
    a = SparsePauliOp.from_list([("ZZ", 1.0)])
    b = SparsePauliOp.from_list([("ZZ", 1.0 + 1e-15)])
    assert _hamiltonian_fingerprint(a) == _hamiltonian_fingerprint(b)


def test_the_fingerprint_separates_a_physically_meaningful_gap():
    a = SparsePauliOp.from_list([("ZZ", 1.0)])
    b = SparsePauliOp.from_list([("ZZ", 1.0 + 1e-9)])
    assert _hamiltonian_fingerprint(a) != _hamiltonian_fingerprint(b)


def test_the_fingerprint_separates_different_pauli_supports():
    a = SparsePauliOp.from_list([("ZZ", 1.0)])
    b = SparsePauliOp.from_list([("ZI", 1.0)])
    assert _hamiltonian_fingerprint(a) != _hamiltonian_fingerprint(b)


# ── Transpilation : le niveau demande est-il celui applique ? ─────────

def test_an_ideal_simulator_forces_optimisation_level_zero():
    """Fige un comportement non documente cote appelant : un opt_level=3
    demande est ignore pour les simulateurs ideaux. Ce n'est pas faux — il
    n'y a ni routage ni couplage a resoudre — mais cela doit etre visible."""
    import inspect
    src = inspect.getsource(VQARuntime.transpile)
    assert "state_vector" in src and "matrix_product_state" in src
    assert "level = 0" in src


def test_transpiling_preserves_the_qubit_count():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(4)
    for k in range(4):
        qc.ry(0.3 * (k + 1), k)
    out = _rt().transpile(qc)
    assert out.num_qubits == 4
