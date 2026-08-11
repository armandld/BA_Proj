"""La pile VQA, module par module, sur des entrées dont on connaît la sortie.

`init_qbits_state`, `postprocess`, `mapping` et les index de
`cost_hamiltonian` décident de la correspondance entre les qubits et la
grille. Une erreur y est invisible : le circuit tourne, la distribution
sort, et la décision porte simplement sur les mauvaises cellules.

Ces tests vérifient les contrats que rien ne vérifiait :
  - un angle θ = 0 doit laisser le qubit dans |0⟩, θ = π l'amener dans |1⟩ ;
  - ψ est un axe de rotation, donc il ne doit PAS changer P(|1⟩) ;
  - les marginales doivent se lire dans le bon ordre de bits ;
  - le hamiltonien de coût doit être diagonal, et son état fondamental
    calculable à la main sur des cas simples.
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from VQA.cost_hamiltonian import (          # noqa: E402
    COEFF_MIN, NullHamiltonianError, create_bounded_hamiltonian,
    create_period_hamiltonian, get_expected_Z,
)
from VQA.init_qbits_state import init_qbits_state   # noqa: E402
from VQA.postprocess import postprocess             # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
#  1. Préparation d'état : θ = amplitude, ψ = axe
# ═══════════════════════════════════════════════════════════════════════

def _probs(qc):
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(qc)
    n = qc.num_qubits
    p = np.abs(sv.data) ** 2
    #  P(qubit i = 1), convention little-endian de Qiskit
    return np.array([sum(p[j] for j in range(len(p)) if (j >> i) & 1)
                     for i in range(n)])


def _z(dim):
    return np.zeros((dim, dim))


def test_the_circuit_has_two_qubits_per_cell():
    """2 * dim^2 qubits : un par arete horizontale, un par verticale."""
    for dim in (2, 3, 4):
        qc = init_qbits_state(_z(dim), _z(dim), _z(dim), _z(dim))
        assert qc.num_qubits == 2 * dim * dim


def test_theta_zero_leaves_every_qubit_in_the_ground_state():
    qc = init_qbits_state(_z(2), _z(2), _z(2), _z(2))
    np.testing.assert_allclose(_probs(qc), 0.0, atol=1e-12)


def test_theta_pi_flips_every_qubit():
    dim = 2
    pi = np.full((dim, dim), np.pi)
    qc = init_qbits_state(pi, pi, _z(dim), _z(dim))
    np.testing.assert_allclose(_probs(qc), 1.0, atol=1e-12)


@pytest.mark.parametrize("score", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_theta_encodes_the_score_as_a_probability(score):
    """L'encodage revendique P(|1>) = score avec theta = 2 arcsin(sqrt(score)).

    C'est le contrat central de la preparation d'etat ; il n'etait teste
    nulle part.
    """
    dim = 2
    theta = np.full((dim, dim), 2.0 * np.arcsin(np.sqrt(score)))
    qc = init_qbits_state(theta, theta, _z(dim), _z(dim))
    np.testing.assert_allclose(_probs(qc), score, atol=1e-10)


def test_psi_is_a_rotation_axis_and_leaves_the_probability_alone():
    """psi change la PHASE, pas l'amplitude.

    Si psi deplacait P(|1>), il ferait double emploi avec theta et
    l'ablation psi ne mesurerait pas ce qu'elle croit.
    """
    dim = 2
    theta = np.full((dim, dim), 2.0 * np.arcsin(np.sqrt(0.3)))
    ref = _probs(init_qbits_state(theta, theta, _z(dim), _z(dim)))
    for psi_val in (0.5, 1.0, -2.0, np.pi):
        psi = np.full((dim, dim), psi_val)
        got = _probs(init_qbits_state(theta, theta, psi, psi))
        np.testing.assert_allclose(got, ref, atol=1e-10,
                                   err_msg=f"psi={psi_val} deplace P(|1>)")


def test_psi_does_change_the_state_itself():
    """Controle oppose : psi ne doit pas etre inerte non plus."""
    from qiskit.quantum_info import Statevector

    dim = 2
    theta = np.full((dim, dim), 2.0 * np.arcsin(np.sqrt(0.3)))
    a = Statevector.from_instruction(
        init_qbits_state(theta, theta, _z(dim), _z(dim))).data
    b = Statevector.from_instruction(
        init_qbits_state(theta, theta, np.full((dim, dim), 1.0),
                         _z(dim))).data
    assert not np.allclose(a, b), "psi n'a aucun effet sur l'etat"


def test_horizontal_and_vertical_angles_reach_different_qubits():
    """Les dim^2 premiers qubits portent l'horizontal, les suivants le
    vertical. Une confusion ferait raffiner selon la mauvaise orientation.
    """
    dim = 2
    hot = np.full((dim, dim), np.pi)
    p_h = _probs(init_qbits_state(hot, _z(dim), _z(dim), _z(dim)))
    p_v = _probs(init_qbits_state(_z(dim), hot, _z(dim), _z(dim)))
    np.testing.assert_allclose(p_h[:dim * dim], 1.0, atol=1e-12)
    np.testing.assert_allclose(p_h[dim * dim:], 0.0, atol=1e-12)
    np.testing.assert_allclose(p_v[:dim * dim], 0.0, atol=1e-12)
    np.testing.assert_allclose(p_v[dim * dim:], 1.0, atol=1e-12)


def test_each_cell_maps_to_its_own_qubit():
    """Bijection cellule -> qubit : allumer une cellule n'en allume qu'un."""
    dim = 3
    for i in range(dim):
        for j in range(dim):
            th = _z(dim)
            th[i, j] = np.pi
            p = _probs(init_qbits_state(th, _z(dim), _z(dim), _z(dim)))
            assert p.sum() == pytest.approx(1.0, abs=1e-10)
            assert p[i * dim + j] == pytest.approx(1.0, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════
#  2. Post-traitement des distributions
# ═══════════════════════════════════════════════════════════════════════

def test_postprocess_reads_the_bitstring_right_to_left():
    """Convention : le bit le PLUS A DROITE est le qubit 0.

    Une lecture inversee ferait raffiner l'image miroir de la grille.
    """
    assert postprocess({"0001": 10}, 4, False) == [10, 0, 0, 0]
    assert postprocess({"1000": 10}, 4, False) == [0, 0, 0, 10]


def test_postprocess_sums_the_counts():
    got = postprocess({"01": 3, "11": 7}, 2, False)
    assert got == [10, 7]


def test_postprocess_returns_zero_on_an_empty_distribution():
    assert postprocess({}, 3, False) == [0, 0, 0]


def test_postprocess_ignores_bits_beyond_the_register():
    """Une chaine plus longue que le registre ne doit pas deborder."""
    assert postprocess({"111111": 5}, 2, False) == [5, 5]


def test_postprocess_never_exceeds_the_total_count():
    d = {"00": 1, "01": 2, "10": 3, "11": 4}
    got = postprocess(d, 2, False)
    assert max(got) <= sum(d.values())
    assert got == [2 + 4, 3 + 4]


# ═══════════════════════════════════════════════════════════════════════
#  3. Valeur propre attendue
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("theta,expected", [
    (0.0, 1.0), (np.pi, -1.0), (np.pi / 2, 0.0), (np.pi / 3, 0.5),
])
def test_expected_z_matches_cosine(theta, expected):
    assert get_expected_Z(theta) == pytest.approx(expected, abs=1e-12)


def test_expected_z_is_consistent_with_the_state_preparation():
    """<Z> = 1 - 2 P(|1>) : les deux descriptions doivent coincider."""
    for score in (0.0, 0.2, 0.5, 0.8, 1.0):
        theta = 2.0 * np.arcsin(np.sqrt(score))
        assert get_expected_Z(theta) == pytest.approx(1.0 - 2.0 * score,
                                                      abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════
#  4. Hamiltonien de coût
# ═══════════════════════════════════════════════════════════════════════

def _params(dim, h=0.0, c=0.0, k=0.0):
    return {
        "H_edges": (np.full((dim, dim), h), np.full((dim, dim), h)),
        "C_edges": (np.full((dim, dim), c), np.full((dim, dim), c)),
        "K_plaquettes": np.full((dim, dim), k),
        "threshold_amr": 0.3,
        "w_z_frac": 0.15,
    }


def _angles(dim):
    z = np.zeros((dim, dim))
    return z, z, z, z


def _params_halo(dim, h=0.0, c=0.0, k=0.0):
    """Variante bornee : coefficients ET angles portent un halo d'un pixel."""
    m = dim + 2
    return {
        "H_edges": (np.full((m, m), h), np.full((m, m), h)),
        "C_edges": (np.full((m, m), c), np.full((m, m), c)),
        "K_plaquettes": np.full((m, m), k),
        "threshold_amr": 0.3,
        "w_z_frac": 0.15,
    }


def test_the_cost_hamiltonian_is_diagonal():
    """Z, ZZ et ZZZZ seulement : aucun terme X ou Y.

    C'est la propriete qui rend l'enumeration classique exacte, et sur
    laquelle repose tout le panel H0.
    """
    for dim in (2, 3):
        th, tv, ph, pv = _angles(dim)
        H = create_period_hamiltonian(_params(dim, h=0.5, c=-1.0, k=-0.5), dim)
        labels = [str(p) for p in H.paulis]
        assert labels, "hamiltonien vide"
        for lbl in labels:
            assert set(lbl) <= {"I", "Z"}, f"terme non diagonal : {lbl}"


def test_the_cost_hamiltonian_spans_the_whole_register():
    for dim in (2, 3):
        th, tv, ph, pv = _angles(dim)
        H = create_period_hamiltonian(_params(dim, h=0.5, c=-1.0, k=-0.5), dim)
        assert H.num_qubits == 2 * dim * dim


def test_an_all_zero_hamiltonian_raises_instead_of_being_faked():
    """Un operateur vide ne doit pas etre remplace par un terme bidon.

    C'etait le defaut historique : `("Z", [0], 1e-3)` rendait un
    hamiltonien nul indiscernable d'un hamiltonien reel.
    """
    dim = 2
    with pytest.raises(NullHamiltonianError):
        create_period_hamiltonian(_params(dim, h=0.0, c=0.0, k=0.0), dim)
    #  la variante bornee attend des angles AVEC halo : (dim+2, dim+2)
    halo = np.zeros((dim + 2, dim + 2))
    with pytest.raises(NullHamiltonianError):
        create_bounded_hamiltonian(_params_halo(dim, h=0.0, c=0.0, k=0.0),
                                   dim, halo, halo, halo, halo)


def test_the_bounded_hamiltonian_builds_with_a_halo():
    """Variante a bord ouvert : angles (dim+2, dim+2), operateur diagonal."""
    dim = 2
    halo = np.zeros((dim + 2, dim + 2))
    H, cth, ctv, cph, cpv = create_bounded_hamiltonian(
        _params_halo(dim, h=0.4, c=-1.0, k=-0.5), dim, halo, halo, halo, halo)
    assert H.num_qubits == 2 * dim * dim
    #  la variante bornee rend aussi les angles du COEUR, sans le halo
    for arr in (cth, ctv, cph, cpv):
        assert arr.shape == (dim, dim)
    for lbl in (str(p) for p in H.paulis):
        assert set(lbl) <= {"I", "Z"}, f"terme non diagonal : {lbl}"


def test_the_halo_values_reach_the_bounded_hamiltonian():
    """Le halo porte la condition de bord : le changer doit changer H.

    Un halo ignore ferait traiter tout patch comme isole, et la
    decomposition multi-niveaux perdrait son couplage entre patches.
    """
    dim = 2
    zero = np.zeros((dim + 2, dim + 2))
    hot = np.zeros((dim + 2, dim + 2))
    #  Le halo HORIZONTAL est lu dans les COLONNES 0 et -1 (lignes 1:-1) :
    #  `z_halo_left = theta_h_full[1:-1, 0]`.
    hot[1:-1, 0] = np.pi
    a = create_bounded_hamiltonian(_params_halo(dim, h=0.4, c=-1.0, k=-0.5),
                                   dim, zero, zero, zero, zero)[0]
    b = create_bounded_hamiltonian(_params_halo(dim, h=0.4, c=-1.0, k=-0.5),
                                   dim, hot, zero, zero, zero)[0]
    same = ([str(p) for p in a.paulis] == [str(p) for p in b.paulis]
            and np.allclose(np.real(a.coeffs), np.real(b.coeffs)))
    assert not same, "le halo n'influence pas l'hamiltonien borne"


def test_coefficients_below_the_pruning_floor_are_dropped():
    """COEFF_MIN doit MORDRE : un coefficient sous le seuil disparait."""
    dim = 2
    th, tv, ph, pv = _angles(dim)
    with pytest.raises(NullHamiltonianError):
        create_period_hamiltonian(_params(dim, c=COEFF_MIN / 100.0), dim)
    #  juste au-dessus, l'operateur existe
    H = create_period_hamiltonian(_params(dim, c=-100.0 * COEFF_MIN), dim)
    assert len(H.paulis) > 0


def test_a_single_nonzero_channel_is_enough_to_build_the_operator():
    dim = 2
    th, tv, ph, pv = _angles(dim)
    for kw in ({"h": 0.7}, {"c": -0.7}, {"k": -0.7}):
        H = create_period_hamiltonian(_params(dim, **kw), dim)
        assert len(H.paulis) > 0, f"canal {kw} n'a produit aucun terme"


def _ground_state_energy(H):
    """Enumeration exhaustive — licite parce que H est diagonal."""
    n = H.num_qubits
    diag = np.zeros(1 << n)
    for pauli, coeff in zip(H.paulis, H.coeffs):
        lbl = str(pauli)[::-1]                 # little-endian
        qubits = [i for i, c in enumerate(lbl) if c == "Z"]
        if not qubits:
            diag += np.real(coeff)
            continue
        idx = np.arange(1 << n)
        sign = np.ones(1 << n)
        for q in qubits:
            sign = sign * np.where((idx >> q) & 1, -1.0, 1.0)
        diag += np.real(coeff) * sign
    return diag


def test_a_ferromagnetic_coupling_prefers_aligned_neighbours():
    """C < 0 : les voisins doivent s'ALIGNER a l'etat fondamental.

    On enumere les 2^n configurations — le hamiltonien etant diagonal,
    c'est exact — et on verifie que le minimum est atteint sur un etat
    uniforme.
    """
    dim = 2
    th, tv, ph, pv = _angles(dim)
    H = create_period_hamiltonian(_params(dim, c=-1.0), dim)
    diag = _ground_state_energy(H)
    best = int(np.argmin(diag))
    n = H.num_qubits
    bits = [(best >> i) & 1 for i in range(n)]
    assert len(set(bits)) == 1, (
        f"l'etat fondamental n'est pas uniforme : {bits}")


def test_a_positive_z_bias_pushes_towards_refinement():
    """h > 0 doit favoriser |1> ; h < 0 doit favoriser |0>.

    Le signe du biais decide du sens de la decision : une inversion ferait
    raffiner exactement le complementaire, sans rien casser.
    """
    dim = 2
    th, tv, ph, pv = _angles(dim)
    for h, expected_bit in ((0.8, 1), (-0.8, 0)):
        H = create_period_hamiltonian(_params(dim, h=h), dim)
        diag = _ground_state_energy(H)
        best = int(np.argmin(diag))
        bits = [(best >> i) & 1 for i in range(H.num_qubits)]
        assert set(bits) == {expected_bit}, (
            f"h={h} donne {bits}, attendu tous a {expected_bit}")


def test_the_hamiltonian_is_deterministic():
    dim = 2
    th, tv, ph, pv = _angles(dim)
    a = create_period_hamiltonian(_params(dim, h=0.3, c=-1.0, k=-0.5), dim)
    b = create_period_hamiltonian(_params(dim, h=0.3, c=-1.0, k=-0.5), dim)
    assert [str(p) for p in a.paulis] == [str(p) for p in b.paulis]
    np.testing.assert_allclose(np.real(a.coeffs), np.real(b.coeffs))


def test_scaling_every_coefficient_scales_the_energy():
    """H est lineaire en ses coefficients : doubler les doit doubler E."""
    dim = 2
    th, tv, ph, pv = _angles(dim)
    a = _ground_state_energy(
        create_period_hamiltonian(_params(dim, c=-1.0), dim))
    b = _ground_state_energy(
        create_period_hamiltonian(_params(dim, c=-2.0), dim))
    np.testing.assert_allclose(b, 2.0 * a, rtol=1e-9)


# ═══════════════════════════════════════════════════════════════════════
#  5. Construction du circuit complet
# ═══════════════════════════════════════════════════════════════════════

def test_mapping_builds_a_circuit_over_the_full_register():
    from VQA.mapping import mapping

    dim = 2
    th, tv, ph, pv = _angles(dim)
    data_in = {"theta_h": th, "theta_v": tv, "psi_h": ph, "psi_v": pv}
    qc, H = mapping(data_in, _params(dim, h=0.3, c=-1.0, k=-0.5),
                    period_bound=True, reps=2)
    assert qc.num_qubits == 2 * dim * dim
    assert H.num_qubits == 2 * dim * dim
    assert qc.num_parameters > 0, "l'ansatz n'a aucun parametre a optimiser"


def test_mapping_depth_grows_with_reps():
    """Plus de couches QAOA doit donner un circuit plus profond."""
    from VQA.mapping import mapping

    dim = 2
    th, tv, ph, pv = _angles(dim)
    data_in = {"theta_h": th, "theta_v": tv, "psi_h": ph, "psi_v": pv}
    depths = []
    for reps in (1, 2, 3):
        qc, _ = mapping(data_in, _params(dim, h=0.3, c=-1.0, k=-0.5),
                        period_bound=True, reps=reps)
        depths.append(qc.depth())
        assert qc.num_parameters == 2 * reps
    assert depths[0] < depths[1] < depths[2], f"profondeurs {depths}"


def test_the_bounded_hamiltonian_never_emits_a_null_coefficient():
    """D-8 : les quatre contractions de halo n'etaient PAS elaguees.

    Avec des coefficients tous nuls, elles remplissaient `sparse_list` de
    termes de valeur exactement 0.0 : la liste n'etait donc pas vide,
    `NullHamiltonianError` ne se declenchait pas, et un operateur nul
    repartait vers l'aval comme s'il etait reel — le defaut meme que cette
    exception devait empecher.
    """
    dim = 2
    halo = np.zeros((dim + 2, dim + 2))
    H = create_bounded_hamiltonian(_params_halo(dim, h=0.4, c=-1.0, k=-0.5),
                                   dim, halo, halo, halo, halo)[0]
    coeffs = np.real(H.coeffs)
    assert len(coeffs) > 0
    assert np.all(np.abs(coeffs) > COEFF_MIN), (
        f"{int(np.sum(np.abs(coeffs) <= COEFF_MIN))} coefficient(s) sous le "
        "seuil ont ete encodes")


def test_the_bounded_hamiltonian_prunes_like_the_periodic_one():
    """Les deux variantes doivent appliquer le MEME seuil.

    Un seuil applique d'un cote seulement rendrait les deux chemins
    incomparables sans que rien ne le signale.
    """
    dim = 2
    halo = np.zeros((dim + 2, dim + 2))
    tiny = COEFF_MIN / 100.0
    with pytest.raises(NullHamiltonianError):
        create_bounded_hamiltonian(_params_halo(dim, c=tiny), dim,
                                   halo, halo, halo, halo)
    with pytest.raises(NullHamiltonianError):
        create_period_hamiltonian(_params(dim, c=tiny), dim)


def test_only_the_documented_halo_cells_are_read():
    """Quelles cellules du halo comptent, mesure plutot que suppose.

    `theta_h_full` n'est lu qu'en colonnes 0 et -1 (lignes 1:-1) ;
    `theta_v_full` qu'en lignes 0 et -1 (colonnes 1:-1). Les quatre coins
    ne sont JAMAIS lus. Une confusion ligne/colonne ici ferait porter la
    condition de bord par le mauvais cote du patch.
    """
    dim = 2
    m = dim + 2
    zero = np.zeros((m, m))
    params = _params_halo(dim, h=0.4, c=-1.0, k=-0.5)

    def coeffs(th, tv):
        H = create_bounded_hamiltonian(params, dim, th, tv, zero, zero)[0]
        return {str(p): float(np.real(c)) for p, c in zip(H.paulis, H.coeffs)}

    base = coeffs(zero, zero)

    #  cellules qui DOIVENT compter
    for name, build in (
            ("theta_h colonne 0", lambda: (_hot(m, (slice(1, -1), 0)), zero)),
            ("theta_h colonne -1", lambda: (_hot(m, (slice(1, -1), -1)), zero)),
            ("theta_v ligne 0", lambda: (zero, _hot(m, (0, slice(1, -1))))),
            ("theta_v ligne -1", lambda: (zero, _hot(m, (-1, slice(1, -1))))),
    ):
        th, tv = build()
        assert coeffs(th, tv) != base, f"{name} est ignoree"

    #  les quatre coins ne doivent RIEN changer
    for corner in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        th = _hot(m, corner)
        assert coeffs(th, zero) == base, (
            f"le coin {corner} de theta_h influence l'hamiltonien")


def _hot(m, where, value=np.pi):
    a = np.zeros((m, m))
    a[where] = value
    return a
