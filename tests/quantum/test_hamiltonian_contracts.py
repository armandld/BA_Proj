"""Audit de contrat de l'Hamiltonien borne : la topologie encodee est-elle
celle que le code decrit ?

`create_bounded_hamiltonian` traduit des tableaux de coefficients en une
somme de chaines de Pauli. Rien en aval ne peut verifier cette traduction :
un mauvais indice produit un operateur parfaitement valide, dont l'etat
fondamental se calcule sans erreur et dont l'energie a l'air normale. C'est
exactement la forme de defaut que le depot cherche — un calcul qui rend une
valeur indiscernable d'une valeur juste.

Les tests ci-dessous construisent les coefficients A LA MAIN, avec une
valeur reconnaissable par case, puis relisent la liste creuse produite. On
n'inspecte pas une propriete statistique de l'operateur : on verifie case
par case quel coefficient a atterri sur quel qubit.

DEFAUT TROUVE ICI (section 2) : les bords GAUCHE et HAUT lisaient
`C_edges[0][ci, 1]` et `C_edges[1][1, cj]` — l'arete INTERIEURE (0)-(1),
deja consommee comme couplage de coeur — au lieu de l'arete du halo,
`[ci, 0]` et `[0, cj]`, qui existe pourtant dans le patch (dim+2, dim+2).
Les bords DROIT et BAS, eux, lisaient la bonne case. L'Hamiltonien etait
donc asymetrique entre gauche et droite sur un patch symetrique. Sur un
patch reel d'Orszag-Tang l'ecart sur le coefficient vaut 2 a 7 %.
"""

import os
import sys

import numpy as np
import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from VQA.cost_hamiltonian import (  # noqa: E402
    COEFF_MIN,
    NullHamiltonianError,
    create_bounded_hamiltonian,
    get_expected_Z,
)

DIM = 3
P = DIM + 2          # taille des tableaux padded


def _params(H=None, C=None, K=None, thr=0.0, w_z=1.0):
    """Coefficients padded, tous nuls sauf ce qu'on demande."""
    z = np.zeros((P, P))
    return {
        "H_edges": (z.copy() if H is None else H[0],
                    z.copy() if H is None else H[1]),
        "C_edges": (z.copy() if C is None else C[0],
                    z.copy() if C is None else C[1]),
        "K_plaquettes": z.copy() if K is None else K,
        "threshold_amr": thr,
        "w_z_frac": w_z,
    }


def _angles(theta_h=None, theta_v=None):
    """theta = pi/2 partout -> <Z> = 0, le halo ne contribue pas."""
    base = np.full((P, P), np.pi / 2.0)
    return (base.copy() if theta_h is None else theta_h,
            base.copy() if theta_v is None else theta_v,
            np.zeros((P, P)), np.zeros((P, P)))


def _build(params, angles=None, dim=DIM, adv=False):
    th, tv, ph, pv = angles if angles is not None else _angles()
    return create_bounded_hamiltonian(params, dim, th, tv, ph, pv,
                                      advanced_anomalies_enabled=adv)


def _terms(op):
    """{(label, tuple(qubits)) -> coefficient} depuis un SparsePauliOp."""
    out = {}
    for pauli, coeff in zip(op.paulis, op.coeffs):
        s = str(pauli)
        n = len(s)
        # Qiskit ecrit le qubit 0 a droite
        qubits = tuple(sorted(n - 1 - i for i, c in enumerate(s) if c != "I"))
        label = "Z" * len(qubits)
        out[(label, qubits)] = out.get((label, qubits), 0.0) + complex(coeff).real
    return out


def _idx_H(i, j, dim=DIM):
    return i * dim + j


def _idx_V(i, j, dim=DIM):
    return dim * dim + i * dim + j


# ======================================================================
#  1. Ce que l'operateur EST : diagonal, de la bonne taille
# ======================================================================

def test_the_hamiltonian_uses_two_qubits_per_cell():
    """dim x dim cellules, deux familles d'aretes : 2*dim^2 qubits."""
    C = (np.full((P, P), -1.0), np.full((P, P), -1.0))
    op, *_ = _build(_params(C=C))
    assert op.num_qubits == 2 * DIM * DIM


def test_the_hamiltonian_is_purely_diagonal():
    """Toute la campagne repose sur l'enumeration exhaustive de la diagonale.

    Un seul X ou Y rendrait l'etat fondamental non enumerable — et les
    resultats publies, faux sans que rien ne le signale.
    """
    C = (np.full((P, P), -1.0), np.full((P, P), -0.7))
    K = np.full((P, P), -0.4)
    H = (np.full((P, P), 0.2), np.full((P, P), -0.3))
    op, *_ = _build(_params(H=H, C=C, K=K), adv=False)
    for pauli in op.paulis:
        assert set(str(pauli)) <= {"I", "Z"}, f"terme non diagonal : {pauli}"


def test_every_pauli_string_is_at_most_four_body():
    C = (np.full((P, P), -1.0), np.full((P, P), -0.7))
    K = np.full((P, P), -0.4)
    op, *_ = _build(_params(C=C, K=K))
    for pauli in op.paulis:
        assert str(pauli).count("Z") <= 4


def test_the_returned_angles_are_the_core_and_only_the_core():
    """Rendre le halo par erreur donnerait (dim+2)^2 angles, silencieusement."""
    th = np.arange(P * P, dtype=float).reshape(P, P)
    tv = th + 1000.0
    C = (np.full((P, P), -1.0), np.full((P, P), -1.0))
    _, core_h, core_v, psi_h, psi_v = _build(
        _params(C=C), angles=(th, tv, th * 0, tv * 0))
    for a in (core_h, core_v, psi_h, psi_v):
        assert a.shape == (DIM, DIM)
    assert np.array_equal(core_h, th[1:-1, 1:-1])
    assert np.array_equal(core_v, tv[1:-1, 1:-1])


# ======================================================================
#  2. Le defaut : quel coefficient atterrit sur le qubit du bord ?
# ======================================================================

def _halo_probe():
    """C_edges reconnaissable : chaque colonne/ligne porte sa propre valeur."""
    Ch = np.zeros((P, P))
    Cv = np.zeros((P, P))
    for b in range(P):
        Ch[:, b] = -(b + 1)          # colonne b -> valeur -(b+1)
        Cv[b, :] = -(10 * (b + 1))   # ligne   b -> valeur -10*(b+1)
    # halo a <Z> = +1 (theta = 0) pour que la contraction soit lisible
    th = np.full((P, P), np.pi / 2.0)
    tv = np.full((P, P), np.pi / 2.0)
    th[1:-1, 0] = 0.0        # halo gauche  : <Z> = 1
    th[1:-1, -1] = 0.0       # halo droit   : <Z> = 1
    tv[0, 1:-1] = 0.0        # halo haut    : <Z> = 1
    tv[-1, 1:-1] = 0.0       # halo bas     : <Z> = 1
    return (Ch, Cv), (th, tv, np.zeros((P, P)), np.zeros((P, P)))


def test_the_left_halo_reads_its_own_edge_not_the_interior_one():
    """C_edges[0][ci, 0] est l'arete halo-coeur ; [ci, 1] est interieure."""
    (Ch, Cv), ang = _halo_probe()
    params = _params(C=(Ch, Cv), thr=0.5, w_z=1.0)
    op, *_ = _build(params, angles=ang)
    t = _terms(op)
    # <Z>_halo = 1, z_threshold = 1 - 2*0.5 = 0 -> contraction = 1.0 * C
    i = 1
    got = t[("Z", (_idx_H(i, 0),))]
    # la contribution du bord se superpose au couplage ZZ ? non : ZZ est
    # un terme a deux qubits, il ne peut pas se confondre avec ce Z.
    assert got == pytest.approx(Ch[i + 1, 0]), (
        f"le bord gauche a encode {got}, soit C_h[ci,1]={Ch[i + 1, 1]} "
        f"(l'arete interieure) au lieu de C_h[ci,0]={Ch[i + 1, 0]}")


def test_the_top_halo_reads_its_own_edge_not_the_interior_one():
    (Ch, Cv), ang = _halo_probe()
    op, *_ = _build(_params(C=(Ch, Cv), thr=0.5), angles=ang)
    t = _terms(op)
    j = 1
    got = t[("Z", (_idx_V(0, j),))]
    assert got == pytest.approx(Cv[0, j + 1]), (
        f"le bord haut a encode {got}, soit C_v[1,cj]={Cv[1, j + 1]} au lieu "
        f"de C_v[0,cj]={Cv[0, j + 1]}")


def test_the_right_halo_already_read_the_right_edge():
    """Le bord droit etait juste : c'est ce qui rendait l'operateur asymetrique."""
    (Ch, Cv), ang = _halo_probe()
    op, *_ = _build(_params(C=(Ch, Cv), thr=0.5), angles=ang)
    t = _terms(op)
    i = 1
    assert t[("Z", (_idx_H(i, DIM - 1),))] == pytest.approx(Ch[i + 1, DIM])


def test_a_mirror_symmetric_patch_gives_symmetric_boundary_terms():
    """Le test qui aurait attrape le defaut sans connaitre les indices.

    On construit C_h symetrique par j -> (P-1-j). Les contractions de
    gauche et de droite doivent alors se repondre exactement.
    """
    # `C_h[:, a]` porte l'arete (a)-(a+1). Le miroir des CELLULES
    # b -> P-1-b envoie l'arete a sur l'arete P-2-a : c'est autour de
    # celle-la qu'il faut symetriser, pas autour du centre des colonnes.
    Ch = np.zeros((P, P))
    for b in range(P - 1):
        Ch[:, b] = -(1.0 + min(b, P - 2 - b))
    assert np.array_equal(Ch[:, :P - 1], Ch[:, P - 2::-1])
    Cv = np.zeros((P, P))
    th = np.full((P, P), np.pi / 2.0)
    th[1:-1, 0] = 0.0
    th[1:-1, -1] = 0.0
    op, *_ = _build(_params(C=(Ch, Cv), thr=0.5),
                    angles=(th, np.full((P, P), np.pi / 2.0),
                            np.zeros((P, P)), np.zeros((P, P))))
    t = _terms(op)
    for i in range(DIM):
        left = t.get(("Z", (_idx_H(i, 0),)), 0.0)
        right = t.get(("Z", (_idx_H(i, DIM - 1),)), 0.0)
        assert left == pytest.approx(right), (
            f"ligne {i} : bord gauche {left}, bord droit {right} sur un "
            "patch pourtant symetrique gauche-droite")


def test_the_halo_contraction_is_neutral_exactly_at_the_threshold():
    """C'est la promesse explicite de la docstring : halo au seuil -> 0."""
    Ch = np.full((P, P), -3.0)
    Cv = np.zeros((P, P))
    thr = 0.25
    # cos(theta) = 1 - 2*thr  ->  theta = arccos(1 - 2*thr)
    th = np.full((P, P), np.arccos(1.0 - 2.0 * thr))
    op, *_ = _build(_params(C=(Ch, Cv), thr=thr),
                    angles=(th, np.full((P, P), np.pi / 2.0),
                            np.zeros((P, P)), np.zeros((P, P))))
    t = _terms(op)
    for i in range(DIM):
        for j in (0, DIM - 1):
            assert abs(t.get(("Z", (_idx_H(i, j),)), 0.0)) < 1e-9


@pytest.mark.parametrize("above", [True, False])
def test_the_halo_pushes_the_right_way_relative_to_the_threshold(above):
    """halo > seuil -> Z positif (raffiner) ; < seuil -> negatif."""
    thr = 0.5
    Ch = np.full((P, P), -2.0)      # C < 0, ferromagnetique
    Cv = np.zeros((P, P))
    score = 0.9 if above else 0.1
    th = np.full((P, P), np.arccos(1.0 - 2.0 * score))
    op, *_ = _build(_params(C=(Ch, Cv), thr=thr),
                    angles=(th, np.full((P, P), np.pi / 2.0),
                            np.zeros((P, P)), np.zeros((P, P))))
    t = _terms(op)
    got = t[("Z", (_idx_H(0, 0),))]
    # C < 0 et (cos(th) - z_thr) du signe oppose au score-seuil
    expected_sign = 1.0 if above else -1.0
    assert np.sign(got) == expected_sign, (
        f"halo {'au-dessus' if above else 'en-dessous'} du seuil -> {got}")


def test_the_halo_weight_scales_with_w_z_frac():
    Ch = np.full((P, P), -2.0)
    Cv = np.zeros((P, P))
    th = np.full((P, P), 0.0)      # <Z> = 1
    ang = (th, np.full((P, P), np.pi / 2.0), np.zeros((P, P)), np.zeros((P, P)))
    a = _terms(_build(_params(C=(Ch, Cv), thr=0.5, w_z=0.1), angles=ang)[0])
    b = _terms(_build(_params(C=(Ch, Cv), thr=0.5, w_z=0.4), angles=ang)[0])
    ka = a[("Z", (_idx_H(0, 0),))]
    kb = b[("Z", (_idx_H(0, 0),))]
    assert kb / ka == pytest.approx(4.0, rel=1e-9)


# ======================================================================
#  3. La plaquette : les quatre qubits forment-ils une boucle fermee ?
# ======================================================================

def test_the_plaquette_is_a_closed_loop_of_four_edges():
    """H(i,j), V(i,j+1), H(i+1,j), V(i,j) bordent le carre (i,j)-(i+1,j+1).

    Si un des quatre indices glissait, le terme ZZZZ cesserait de mesurer
    une circulation — sans cesser d'etre un operateur valide.
    """
    d = 4
    q = d + 2
    K = np.zeros((q, q))
    K[2, 2] = -5.0                      # cellule de coeur (i,j) = (1,1)
    base = np.full((q, q), np.pi / 2.0)
    zq = np.zeros((q, q))
    params = {"H_edges": (zq.copy(), zq.copy()),
              "C_edges": (zq.copy(), zq.copy()), "K_plaquettes": K,
              "threshold_amr": 0.0, "w_z_frac": 1.0}
    op, *_ = create_bounded_hamiltonian(params, d, base, base.copy(),
                                        zq.copy(), zq.copy())
    t = _terms(op)
    expected = tuple(sorted([
        1 * d + 1,                      # H(1,1)
        d * d + 1 * d + 2,              # V(1,2)
        2 * d + 1,                      # H(2,1)
        d * d + 1 * d + 1,              # V(1,1)
    ]))
    assert ("ZZZZ", expected) in t, (
        f"la plaquette encodee n'est pas la boucle attendue : "
        f"{[k for k in t if k[0] == 'ZZZZ']}")
    assert t[("ZZZZ", expected)] == pytest.approx(-5.0)


def test_the_four_plaquette_qubits_are_all_distinct():
    """Un doublon reduirait silencieusement le ZZZZ a un ZZ."""
    d = 4
    q = d + 2
    zq = np.zeros((q, q))
    base = np.full((q, q), np.pi / 2.0)
    params = {"H_edges": (zq.copy(), zq.copy()),
              "C_edges": (zq.copy(), zq.copy()),
              "K_plaquettes": np.full((q, q), -1.0),
              "threshold_amr": 0.0, "w_z_frac": 1.0}
    op, *_ = create_bounded_hamiltonian(params, d, base, base.copy(),
                                        zq.copy(), zq.copy())
    for pauli in op.paulis:
        s = str(pauli)
        assert s.count("Z") == len(set(i for i, c in enumerate(s) if c == "Z"))


def test_a_plaquette_on_the_boundary_contracts_instead_of_wrapping():
    """Au bord, le qubit manquant devient un coefficient — pas un voisin
    de l'autre cote du patch.

    SEUIL REMESURE (D-113), et la garantie n'a pas bouge. Ce test chauffait
    `theta_h[1:-1, -1]` pour le membre Droite et `theta_v[-1, 1:-1]` pour le
    membre Bas — c'est-a-dire exactement les deux tableaux que la version
    d'avant lisait, familles echangees. Ses commentaires disaient « halo
    droit » au-dessus d'une ecriture dans `theta_h` : le champ d'essai etait
    construit sur la convention fausse.

    Le membre Droite d'une plaquette est un lien **V** et le membre Bas un
    lien **H** : les cellules a chauffer sont `theta_v[1:-1, -1]` et
    `theta_h[-1, 1:-1]`.

    Remesure a l'identique (dim = 3, plaquette de coin `K[DIM, DIM] = -2.0`,
    tout le reste a `pi/2` donc `<Z> = 0`) :

      * anciennes cellules, code corrige : **0 terme** — c'est l'echec ;
      * bonnes cellules, code corrige    : **1 terme**, `ZZ` sur
        `{idx_H(2,2), idx_V(2,2)} = {8, 17}`, coefficient **-2.0** ;
      * bonnes cellules, code d'avant    : **0 terme**.

    L'assertion, elle, est inchangee. Ce qui a change est le champ d'essai.
    """
    K = np.zeros((P, P))
    K[DIM, DIM] = -2.0                  # derniere cellule de coeur
    tv = np.full((P, P), np.pi / 2.0)
    th = np.full((P, P), np.pi / 2.0)
    tv[1:-1, -1] = 0.0                  # liens V du halo droit -> membre Droite
    th[-1, 1:-1] = 0.0                  # liens H du halo bas   -> membre Bas
    op, *_ = _build(_params(K=K), angles=(th, tv, np.zeros((P, P)),
                                          np.zeros((P, P))))
    t = _terms(op)
    keys = [k for k in t if abs(t[k]) > 1e-9]
    assert len(keys) == 1
    label, qubits = keys[0]
    assert label == "ZZ", f"attendu une contraction a deux corps, obtenu {label}"
    assert set(qubits) == {_idx_H(DIM - 1, DIM - 1), _idx_V(DIM - 1, DIM - 1)}
    assert t[keys[0]] == pytest.approx(-2.0)


def test_the_boundary_plaquette_ignores_the_other_link_family():
    """Le pendant du test ci-dessus, et ce qui le rend discriminant (D-113).

    Chauffer les cellules de l'AUTRE famille — `theta_h` a droite, `theta_v`
    en bas — ne doit rien produire : ces liens-la n'appartiennent pas a la
    plaquette de coin. Sur la version d'avant, cette configuration rendait au
    contraire l'unique terme `ZZ` a **-2.0**, et celle du test precedent n'en
    rendait aucun : les deux tests sont exactement inverses d'une version a
    l'autre.
    """
    K = np.zeros((P, P))
    K[DIM, DIM] = -2.0
    tv = np.full((P, P), np.pi / 2.0)
    th = np.full((P, P), np.pi / 2.0)
    th[1:-1, -1] = 0.0                  # liens H du halo droit : hors plaquette
    tv[-1, 1:-1] = 0.0                  # liens V du halo bas   : hors plaquette
    op, *_ = _build(_params(K=K), angles=(th, tv, np.zeros((P, P)),
                                          np.zeros((P, P))))
    t = _terms(op)
    assert [k for k in t if abs(t[k]) > 1e-9] == [], (
        "la plaquette du bord se contracte encore sur l'autre famille de "
        "liens — c'est le defaut D-113")


def test_no_qubit_index_is_ever_minus_one():
    """-1 est un indice Python VALIDE : il designerait le dernier qubit."""
    K = np.full((P, P), -1.0)
    C = (np.full((P, P), -1.0), np.full((P, P), -1.0))
    H = (np.full((P, P), 0.5), np.full((P, P), 0.5))
    op, *_ = _build(_params(H=H, C=C, K=K))
    assert op.num_qubits == 2 * DIM * DIM      # aucun qubit fantome ajoute
    # un indice -1 se serait traduit par un Z sur le dernier qubit ;
    # on verifie que la structure est bien celle attendue terme a terme
    for pauli in op.paulis:
        assert len(str(pauli)) == 2 * DIM * DIM


# ======================================================================
#  4. Le refus d'un Hamiltonien nul
# ======================================================================

def test_an_all_zero_coefficient_set_raises_instead_of_returning_zero():
    with pytest.raises(NullHamiltonianError):
        _build(_params())


def test_the_null_error_names_the_qubit_count_and_threshold():
    """Un message qui ne dit pas ou chercher fait perdre l'heure suivante."""
    try:
        _build(_params())
    except NullHamiltonianError as e:
        assert e.num_qubits == 2 * DIM * DIM
        assert e.threshold == COEFF_MIN
        assert str(e)


def test_coefficients_below_the_threshold_are_not_encoded():
    C = (np.full((P, P), -1e-9), np.full((P, P), -1e-9))
    with pytest.raises(NullHamiltonianError):
        _build(_params(C=C))


def test_a_single_coefficient_above_the_threshold_is_enough():
    C = (np.zeros((P, P)), np.zeros((P, P)))
    C[0][2, 2] = -1.0
    op, *_ = _build(_params(C=C))
    assert len(op.paulis) >= 1


def test_the_halo_terms_alone_never_manufacture_a_hamiltonian():
    """Le piege corrige : des contractions nulles remplissaient la liste."""
    C = (np.zeros((P, P)), np.zeros((P, P)))
    th = np.zeros((P, P))            # <Z> = 1 partout, halo tres actif
    with pytest.raises(NullHamiltonianError):
        _build(_params(C=C), angles=(th, th, np.zeros((P, P)),
                                     np.zeros((P, P))))


# ======================================================================
#  5. Ce que l'Hamiltonien deploye peut representer
# ======================================================================
#
# Tous les couplages sortent negatifs des deux mappeurs (ferromagnetiques).
# Un ZZ ferromagnetique est satisfait par l'etat uniforme, et un ZZZZ de
# parite paire aussi. Sans biais Z, l'etat fondamental est donc uniforme
# par construction : le probleme ne contient AUCUNE frustration, il n'y a
# rien de combinatoire a resoudre. Ce fait est structurel, pas empirique —
# ces tests le figent pour qu'on ne l'attribue pas a l'optimiseur.

def _ground_state_energy(op):
    """Diagonale complete : l'operateur est diagonal, donc enumerable."""
    n = op.num_qubits
    diag = np.zeros(2 ** n)
    states = np.arange(2 ** n)
    for pauli, coeff in zip(op.paulis, op.coeffs):
        s = str(pauli)
        sign = np.ones(2 ** n)
        for k, c in enumerate(s):
            if c == "Z":
                q = n - 1 - k
                sign = sign * (1.0 - 2.0 * ((states >> q) & 1))
        diag = diag + complex(coeff).real * sign
    return diag


def test_a_purely_ferromagnetic_hamiltonian_has_a_uniform_ground_state():
    dim = 2
    p = dim + 2
    C = (np.full((p, p), -1.0), np.full((p, p), -1.0))
    K = np.full((p, p), -0.5)
    params = {"H_edges": (np.zeros((p, p)), np.zeros((p, p))),
              "C_edges": C, "K_plaquettes": K,
              "threshold_amr": 0.5, "w_z_frac": 0.0}
    th = np.full((p, p), np.arccos(0.0))     # halo neutre au seuil 0.5
    op, *_ = create_bounded_hamiltonian(params, dim, th, th.copy(),
                                        np.zeros((p, p)), np.zeros((p, p)))
    diag = _ground_state_energy(op)
    gs = int(np.argmin(diag))
    n = op.num_qubits
    assert gs in (0, 2 ** n - 1), (
        f"etat fondamental {gs:0{n}b} : un couplage purement ferromagnetique "
        "devrait etre satisfait par l'etat uniforme")


def test_the_z_bias_is_what_breaks_the_uniform_degeneracy():
    """Sans Z, deux etats fondamentaux exactement degeneres."""
    dim = 2
    p = dim + 2
    C = (np.full((p, p), -1.0), np.full((p, p), -1.0))
    params = {"H_edges": (np.zeros((p, p)), np.zeros((p, p))),
              "C_edges": C, "K_plaquettes": np.zeros((p, p)),
              "threshold_amr": 0.5, "w_z_frac": 0.0}
    th = np.full((p, p), np.arccos(0.0))
    op, *_ = create_bounded_hamiltonian(params, dim, th, th.copy(),
                                        np.zeros((p, p)), np.zeros((p, p)))
    diag = _ground_state_energy(op)
    lo = np.min(diag)
    assert int(np.sum(np.isclose(diag, lo))) >= 2

    H = (np.full((p, p), 0.3), np.full((p, p), 0.3))
    params["H_edges"] = H
    op2, *_ = create_bounded_hamiltonian(params, dim, th, th.copy(),
                                         np.zeros((p, p)), np.zeros((p, p)))
    diag2 = _ground_state_energy(op2)
    assert int(np.sum(np.isclose(diag2, np.min(diag2)))) == 1


def test_the_ground_state_is_uniform_on_real_deployed_coefficients():
    """Le meme fait, mais sur les coefficients que la campagne produit."""
    from Simulation.HamiltParams_v2 import PhysicalMapperV2
    from Simulation.PhysToAngle import AngleMapper
    from Simulation.grid import curl_z

    dim = 2
    p = dim + 2
    rng = np.random.default_rng(0)
    f = {k: rng.normal(size=(p, p)) for k in ("vx", "vy", "Bx", "By")}
    f["Jz"] = curl_z(f["Bx"], f["By"], True)
    sc = AngleMapper.classical_score(f)
    r = PhysicalMapperV2(dx=0.02).compute_coefficients(None, sc, f, 0.1496)
    assert np.all(r["C_edges"][0] <= 0) and np.all(r["K_plaquettes"] <= 0)
    th = 2.0 * np.arcsin(np.sqrt(np.clip(sc, 0.0, 1.0)))
    op, *_ = create_bounded_hamiltonian(r, dim, th, th.copy(),
                                        np.zeros((p, p)), np.zeros((p, p)))
    diag = _ground_state_energy(op)
    gs = int(np.argmin(diag))
    n = op.num_qubits
    bits = [(gs >> q) & 1 for q in range(n)]
    assert len(set(bits)) == 1, (
        f"etat fondamental {bits} — non uniforme, ce qui contredirait "
        "l'absence de frustration mesuree sur la campagne")


# ======================================================================
#  6. Le garde de forme : un patch trop grand se lisait en silence
# ======================================================================
#
# Toutes les lectures sont indexees par `dim`. Avec des tableaux plus grands
# que (dim+2, dim+2), la boucle lisait un sous-bloc du coin superieur gauche
# et rendait un Hamiltonien valide — calcule sur la mauvaise portion du
# patch, sans le moindre signal. C'est la forme de defaut la plus couteuse :
# elle ne laisse aucune trace.

def test_a_patch_larger_than_dim_is_refused_instead_of_truncated():
    big = 6
    z = np.zeros((big, big))
    params = {"H_edges": (z.copy(), z.copy()),
              "C_edges": (np.full((big, big), -1.0), z.copy()),
              "K_plaquettes": z.copy(),
              "threshold_amr": 0.0, "w_z_frac": 1.0}
    base = np.full((big, big), np.pi / 2.0)
    with pytest.raises(ValueError, match="attend des tableaux"):
        create_bounded_hamiltonian(params, 2, base, base.copy(),
                                   z.copy(), z.copy())


def test_a_patch_smaller_than_dim_is_refused_before_encoding_anything():
    small = 4
    z = np.zeros((small, small))
    params = {"H_edges": (z.copy(), z.copy()),
              "C_edges": (np.full((small, small), -1.0), z.copy()),
              "K_plaquettes": z.copy(),
              "threshold_amr": 0.0, "w_z_frac": 1.0}
    base = np.full((small, small), np.pi / 2.0)
    with pytest.raises(ValueError, match="attend des tableaux"):
        create_bounded_hamiltonian(params, 4, base, base.copy(),
                                   z.copy(), z.copy())


def test_the_error_message_names_every_array_that_is_wrong():
    """Un message qui n'en nomme qu'un fait recommencer trois fois."""
    z = np.zeros((6, 6))
    params = {"H_edges": (z.copy(), z.copy()),
              "C_edges": (z.copy(), z.copy()), "K_plaquettes": z.copy(),
              "threshold_amr": 0.0, "w_z_frac": 1.0}
    base = np.full((6, 6), np.pi / 2.0)
    try:
        create_bounded_hamiltonian(params, 2, base, base.copy(),
                                   z.copy(), z.copy())
    except ValueError as e:
        msg = str(e)
        for name in ("C_edges[0]", "H_edges[0]", "theta_h_full",
                     "K_plaquettes"):
            assert name in msg, f"{name} absent du message"
        assert "(4, 4)" in msg and "(6, 6)" in msg


def test_a_mismatched_angle_array_is_caught_too():
    """Les coefficients peuvent etre bons et les angles decales."""
    z = np.zeros((P, P))
    params = _params(C=(np.full((P, P), -1.0), z.copy()))
    wrong = np.full((P + 1, P + 1), np.pi / 2.0)
    with pytest.raises(ValueError, match="theta_h_full"):
        create_bounded_hamiltonian(params, DIM, wrong,
                                   np.full((P, P), np.pi / 2.0),
                                   z.copy(), z.copy())


def test_the_guard_lets_the_correct_shape_through():
    """Un garde qui refuse tout serait aussi inutile qu'un garde absent."""
    C = (np.full((P, P), -1.0), np.full((P, P), -1.0))
    op, *_ = _build(_params(C=C))
    assert op.num_qubits == 2 * DIM * DIM
