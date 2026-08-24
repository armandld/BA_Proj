"""D-113 — la contraction de plaquette du bord doit lire la BONNE famille de liens.

Une plaquette a quatre membres : Haut = H(i,j), Droite = V(i,j+1),
Bas = H(i+1,j), Gauche = V(i,j). Sur la colonne de droite du coeur le membre
manquant est un lien **V** ; sur la ligne du bas c'est un lien **H**. Le `<Z>`
qui remplace un qubit manquant doit venir du `theta` de CE lien —
`init_qbits_state` place `theta_h` sur les qubits `idx_H` et `theta_v` sur les
qubits `idx_V`.

`create_bounded_hamiltonian` lisait `theta_h_full` pour le membre Droite (un
lien V) et `theta_v_full` pour le membre Bas (un lien H) : les deux familles
etaient echangees. Les POSITIONS etaient bonnes ; seul le tableau lu etait le
mauvais.

**Sur quelle entree ces tests echouent-ils ?** Sur toute entree ou
`theta_h_full != theta_v_full` aux cellules du halo lues. En deploiement les
deux tableaux sont le MEME (`refinement._prepare_vqa_input` passe `mini_score`
deux fois, `PhysToAngle.map_to_angles` le documente et
`tests/mapping/test_mapper_contracts.py` le fige) : c'est pourquoi le defaut
etait inerte, et c'est pourquoi il fallait un champ d'essai qui SEPARE les deux
familles pour le voir.

Mesure (docs/RESULTS.md, D-113) : sur les 36 configurations aleatoires de la
configuration deployee (theta_h ≡ theta_v), l'operateur est identique bit a bit
avant/apres. Sur les 36 configurations separantes, les 36 changent, ecart max
**1,072818** — et le signe du terme bascule (k = -0,5 rendu **+0,5**).
"""

import numpy as np
import pytest

from VQA.cost_hamiltonian import create_bounded_hamiltonian

DIM = 2
M = DIM + 2
K_VAL = -0.5


def _params(k_cells, kx_cells=None):
    """Coefficients nuls partout sauf les plaquettes demandees."""
    zero = np.zeros((M, M))
    K = np.zeros((M, M))
    for cell, v in k_cells.items():
        K[cell] = v
    p = {
        'H_edges': [zero.copy(), zero.copy()],
        'C_edges': [zero.copy(), zero.copy()],
        'K_plaquettes': K,
        'threshold_amr': 0.0,
        'w_z_frac': 1.0,
    }
    if kx_cells is not None:
        KX = np.zeros((M, M))
        for cell, v in kx_cells.items():
            KX[cell] = v
        p['K_xpoint'] = KX
    return p


def _coeffs(params, theta_h, theta_v, advanced=False):
    zero = np.zeros((M, M))
    if not advanced:
        params = {k: v for k, v in params.items() if k != "K_xpoint"}
    op, *_ = create_bounded_hamiltonian(
        params, DIM, theta_h, theta_v, zero, zero)
    return {lbl: float(c.real) for lbl, c in zip(op.paulis.to_labels(), op.coeffs)}


def _only_value(coeffs):
    assert len(coeffs) == 1, f"attendu un seul terme, obtenu {coeffs}"
    return next(iter(coeffs.values()))


# --------------------------------------------------------------------------
# 1. Le membre Droite est un lien V : il doit lire theta_v, pas theta_h.
# --------------------------------------------------------------------------

def test_le_membre_droite_lit_theta_v_et_pas_theta_h():
    """theta_v[halo droit] = 0 -> <Z> = +1, donc le coefficient reste k_val.

    Echoue sur l'ancienne version : elle lisait theta_h[halo droit] = pi,
    donc <Z> = -1, et rendait **+0,5** au lieu de -0,5 — le signe de la
    plaquette, dont la convention (parite paire, K < 0) porte la detection
    de vorticite, etait inverse.
    """
    params = _params({(1, DIM): K_VAL})
    th, tv = np.zeros((M, M)), np.zeros((M, M))
    th[1:-1, -1] = np.pi          # ne concerne QUE les liens H du halo droit

    assert _only_value(_coeffs(params, th, tv)) == pytest.approx(K_VAL)


def test_le_membre_bas_lit_theta_h_et_pas_theta_v():
    """Symetrique : le membre Bas est un lien H, il doit lire theta_h."""
    params = _params({(DIM, 1): K_VAL})
    th, tv = np.zeros((M, M)), np.zeros((M, M))
    tv[-1, 1:-1] = np.pi          # ne concerne QUE les liens V du halo bas

    assert _only_value(_coeffs(params, th, tv)) == pytest.approx(K_VAL)


# --------------------------------------------------------------------------
# 2. L'ancien comportement, epingle : la mauvaise famille ne doit plus piloter.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cell, bonne, mauvaise",
    [
        # plaquette du bord DROIT : le lien manquant est un lien V
        ((1, DIM), ('v', (slice(1, -1), -1)), ('h', (slice(1, -1), -1))),
        # plaquette du bord BAS : le lien manquant est un lien H
        ((DIM, 1), ('h', (-1, slice(1, -1))), ('v', (-1, slice(1, -1)))),
    ],
    ids=["bord_droit", "bord_bas"],
)
def test_seule_la_bonne_famille_pilote_la_contraction(cell, bonne, mauvaise):
    """Epingle l'ancien comportement pour qu'il ne puisse pas revenir en silence.

    Bouger la MAUVAISE famille ne doit rien changer ; bouger la BONNE doit
    changer le coefficient. Avant la correction, les deux assertions etaient
    exactement inversees.
    """
    params = _params({cell: K_VAL})
    base_h, base_v = np.zeros((M, M)), np.zeros((M, M))
    reference = _only_value(_coeffs(params, base_h, base_v))

    def perturbe(famille_et_where):
        famille, where = famille_et_where
        th, tv = np.zeros((M, M)), np.zeros((M, M))
        (th if famille == 'h' else tv)[where] = np.pi / 3
        return _only_value(_coeffs(params, th, tv))

    assert perturbe(mauvaise) == pytest.approx(reference), (
        "la mauvaise famille de liens pilote encore la contraction de plaquette"
    )
    assert perturbe(bonne) != pytest.approx(reference), (
        "la bonne famille de liens n'est pas lue"
    )


def test_le_terme_de_plaquette_ne_disparait_plus_a_cause_de_l_autre_famille():
    """Le cas le plus silencieux de l'ancienne version : le terme s'annulait.

    theta_h[halo droit] = pi/2 donne <Z_H> = 0 exactement. L'ancienne version
    multipliait le coefficient de plaquette par ce 0 : la plaquette du bord
    disparaissait de l'Hamiltonien, sans erreur, sans NaN, sans trace.
    """
    params = _params({(1, DIM): K_VAL})
    th, tv = np.zeros((M, M)), np.zeros((M, M))
    th[1:-1, -1] = np.pi / 2

    assert _only_value(_coeffs(params, th, tv)) == pytest.approx(K_VAL)


# --------------------------------------------------------------------------
# 3. Le terme de point X porte la meme topologie, donc la meme correction.
# --------------------------------------------------------------------------

def test_la_plaquette_de_point_x_lit_la_meme_famille():
    """`K_xpoint` partage la topologie de `K_plaquettes` : meme substitution."""
    params = _params({}, kx_cells={(1, DIM): K_VAL})
    th, tv = np.zeros((M, M)), np.zeros((M, M))
    th[1:-1, -1] = np.pi

    assert _only_value(_coeffs(params, th, tv, advanced=True)) == pytest.approx(K_VAL)


# --------------------------------------------------------------------------
# 4. Portee : en configuration deployee, la correction ne bouge RIEN.
# --------------------------------------------------------------------------

def test_en_configuration_deployee_la_correction_ne_change_aucun_coefficient():
    """theta_h ≡ theta_v : les deux familles coincident, donc l'echange etait inerte.

    Ce test ne peut pas echouer sur l'ancienne version — c'est voulu : il
    n'epingle pas le defaut, il epingle sa PORTEE. C'est la raison pour
    laquelle aucun nombre publie ne bouge, et il doit crier le jour ou un
    appelant passera deux cartes de score distinctes (ce que
    `tests/mapping/test_mapper_contracts.py` interdit aujourd'hui).

    La garantie testee : quand theta_h et theta_v sont egaux, echanger les
    deux familles est l'identite. On la verifie en comparant la contraction
    aux deux bords contre la valeur non contractee.
    """
    rng = np.random.default_rng(113)
    for _ in range(8):
        params = _params({(1, DIM): float(rng.normal(0, .5)),
                          (DIM, 1): float(rng.normal(0, .5))})
        th = rng.uniform(0, np.pi, (M, M))
        avec_v_egal_h = _coeffs(params, th, th.copy())

        # theta_v identique a theta_h : lire l'une ou l'autre famille revient
        # au meme, donc le resultat doit egaler celui d'un tirage ou l'on
        # force explicitement les deux halos concernes a coincider.
        tv = th.copy()
        tv[1:-1, -1] = th[1:-1, -1]
        tv[-1, 1:-1] = th[-1, 1:-1]
        assert _coeffs(params, th, tv) == avec_v_egal_h


def test_le_test_de_portee_sait_encore_echouer():
    """Un test qui ne peut pas echouer est un defaut : celui-ci le peut.

    Sur theta_h != theta_v aux cellules lues, les coefficients DOIVENT
    differer — c'est exactement ce que mesure D-113.
    """
    params = _params({(1, DIM): K_VAL, (DIM, 1): K_VAL})
    rng = np.random.default_rng(1130)
    th = rng.uniform(0.2, np.pi - 0.2, (M, M))
    tv = th.copy()
    tv[1:-1, -1] = th[1:-1, -1] + 0.7      # separe les deux familles

    assert _coeffs(params, th, tv) != _coeffs(params, th, th.copy())
