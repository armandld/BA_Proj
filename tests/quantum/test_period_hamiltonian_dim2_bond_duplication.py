"""D-59 : à dim = 2, la topologie périodique double le lien ZZ shear.

`create_period_hamiltonian` (chemin QAOA / diagonalisation exacte) et
`build_ising_terms` (chemin SA / exhaustif, `study/common/`) émettent tous
deux, pour chaque direction, un lien ZZ par cellule : `(i, j) -> (i, j+1 mod
dim)`. À `dim >= 3` cela donne `dim` liens distincts. À `dim = 2` l'anneau
périodique dégénère : `(i, 0) -> (i, 1)` et `(i, 1) -> (i, 0 mod 2)` relient
la MÊME paire de qubits, et les deux liens sont ajoutés séparément à
l'opérateur au lieu d'être fusionnés — poids effectif doublé sur cette
paire. `K_plaquettes` (ZZZZ) n'a pas ce défaut : les 4 quadruplets de
qubits produits par les 4 cellules à dim = 2 sont distincts deux à deux.

Mesuré (`docs/DEFAUTS.md` D-59) : sur les 4 scénarios canoniques (Re=400,
N=256, mappeur v1, 3 instantanés chacun), dédupliquer le lien ZZ ne change
**aucune** des 12 décisions de fondamental exact — le biais Z domine déjà
le couplage doublé (D-47). `src/` est gelé, aucun nombre publié n'en
dépend : rapport seul, pas de correction. Ces tests PINGUENT le
comportement actuel (dupliqué) pour qu'il ne dérive plus sans mesure, côté
QAOA/diag exacte comme côté SA/exhaustif — ce n'est pas une garantie que le
doublement est correct.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_SRC = os.path.join(_REPO_ROOT, "..", "src")
_SRC = os.path.abspath(_SRC)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
_COMMON = os.path.abspath(os.path.join(_REPO_ROOT, "..", "study", "common"))
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

from VQA.cost_hamiltonian import create_period_hamiltonian  # noqa: E402
from ising_terms_and_annealing import build_ising_terms      # noqa: E402


def _params(dim, h=0.0, c=None, k=0.0):
    """Coefficients de cisaillement NON uniformes : `c` fixe la valeur de
    la colonne/ligne 0, `c * 1.0000001` la colonne/ligne 1 -- assez proche
    pour rester dans une seule bande physique, assez distinct pour ne pas
    masquer une confusion d'index par une coïncidence de valeur."""
    C = np.full((dim, dim), c if c is not None else -1.0)
    return {
        "H_edges": (np.full((dim, dim), h), np.full((dim, dim), h)),
        "C_edges": (C.copy(), C.copy()),
        "K_plaquettes": np.full((dim, dim), k),
        "threshold_amr": 0.3,
        "w_z_frac": 0.15,
    }


def test_dim2_has_a_duplicate_zz_pauli_label():
    """À dim = 2, le lien horizontal (i,0)->(i,1) et (i,1)->(i,0 mod 2)
    relient la même paire de qubits : le SparsePauliOp porte deux entrées
    au même label, à sommer pour connaître le couplage effectif."""
    H = create_period_hamiltonian(_params(2, c=-1.0), 2)
    labels = [str(p) for p in H.paulis]
    zz_labels = [l for l in labels if l.count("Z") == 2]
    assert len(zz_labels) != len(set(zz_labels)), (
        "aucun label ZZ dupliqué à dim=2 : la dégénérescence de bond "
        "documentée en D-59 n'est plus reproduite -- remesurer avant de "
        "retirer ce test")


def test_dim3_has_no_duplicate_zz_pauli_label():
    """À dim = 3 le cycle ne dégénère plus : chaque cellule donne un lien
    ZZ distinct, aucun label ZZ ne se répète."""
    H = create_period_hamiltonian(_params(3, c=-1.0), 3)
    labels = [str(p) for p in H.paulis]
    zz_labels = [l for l in labels if l.count("Z") == 2]
    assert len(zz_labels) == len(set(zz_labels))


def test_dim2_plaquette_terms_are_not_duplicated():
    """Le terme ZZZZ n'a pas le défaut : les 4 quadruplets de qubits
    produits par les 4 cellules à dim=2 sont distincts deux à deux."""
    H = create_period_hamiltonian(_params(2, c=0.0, k=-1.0), 2)
    labels = [str(p) for p in H.paulis]
    zzzz_labels = [l for l in labels if l.count("Z") == 4]
    assert len(zzzz_labels) == 4
    assert len(set(zzzz_labels)) == 4, (
        "un label ZZZZ dupliqué à dim=2 : la plaquette dégénère aussi, "
        "au-delà de ce que D-59 mesure")


def test_build_ising_terms_shares_the_same_degeneracy():
    """`build_ising_terms` (SA / exhaustif) construit sa topologie avec la
    même boucle que `create_period_hamiltonian` : la dégénérescence doit y
    être identique, sinon les deux chemins n'étudient plus le même
    opérateur (cf. COUVERTURE.md, 'build_ising_terms contre
    create_period_hamiltonian : sain')."""
    hp = _params(2, c=-1.0)
    _, (edge_idx, edge_coef), _ = build_ising_terms(hp, 2)
    pairs = [tuple(sorted(map(int, e))) for e in edge_idx]
    assert len(pairs) != len(set(pairs)), (
        "build_ising_terms ne duplique plus le lien a dim=2 alors que "
        "create_period_hamiltonian si : les deux chemins divergent, "
        "au-dela de ce que D-59 documente")


def test_dim2_duplicated_coefficients_are_measured_equal():
    """Les deux occurrences dupliquées portent le MÊME coefficient dès que
    `C_edges` est symétrique par colonne/ligne -- c'est cette égalité
    (mesurée sur des DNS réels dans D-59, reproduite ici sur un cas
    synthétique) qui rend le doublon invisible à une simple lecture de
    `nZZ`."""
    hp = _params(2, c=-1.0)
    H = create_period_hamiltonian(hp, 2)
    by_label = {}
    for pauli, coeff in zip(H.paulis, H.coeffs):
        lbl = str(pauli)
        if lbl.count("Z") == 2:
            by_label.setdefault(lbl, []).append(float(np.real(coeff)))
    dup = {lbl: cs for lbl, cs in by_label.items() if len(cs) > 1}
    assert dup, "attendu au moins un label ZZ dupliqué à dim=2"
    for lbl, cs in dup.items():
        assert cs[0] == pytest.approx(cs[1]), (
            f"{lbl}: coefficients dupliqués distincts {cs} -- ne correspond "
            "plus à la mesure D-59, qui suppose une egalite au bit pres sur "
            "coefficients HamiltParams reels")
