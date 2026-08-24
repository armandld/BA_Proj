"""Le chemin Study et l'opérateur QAOA encodent le même Hamiltonien."""

import itertools
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_REPO, "src"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _coeffs(dim, graine=0, avec_xpoint=True):
    rng = np.random.default_rng(graine)
    hp = {
        "H_edges": (rng.normal(size=(dim, dim)) * 0.1,
                    rng.normal(size=(dim, dim)) * 0.1),
        "C_edges": (-np.abs(rng.normal(size=(dim, dim))),
                    -np.abs(rng.normal(size=(dim, dim)))),
        "K_plaquettes": -np.abs(rng.normal(size=(dim, dim))),
    }
    if avec_xpoint:
        hp["K_xpoint"] = -np.abs(rng.normal(size=(dim, dim)))
    return hp


def test_build_ising_terms_lit_desormais_K_xpoint():
    """La cle etait ignoree : la liste de plaquettes ne bougeait pas."""
    from study.common.ising_terms_and_annealing import build_ising_terms

    dim = 2
    sans = _coeffs(dim, avec_xpoint=False)
    avec = {**sans, "K_xpoint": np.full((dim, dim), -0.3)}

    _, _, p_sans = build_ising_terms(sans, dim)
    _, _, p_avec = build_ising_terms(avec, dim)

    assert p_avec[0].shape[0] == p_sans[0].shape[0] + dim * dim, (
        f"{p_avec[0].shape[0]} plaquettes avec K_xpoint contre "
        f"{p_sans[0].shape[0]} sans : la cle est toujours ignoree")
    assert p_avec[1].sum() == pytest.approx(p_sans[1].sum() - 0.3 * dim * dim), (
        "les coefficients de point X ne s'ajoutent pas")


@pytest.mark.parametrize("graine", [0, 1, 2])
def test_study_et_le_chemin_deploye_coincident(graine):
    """LE test qui compte : meme energie sur les 256 etats.

    Convention de spins : le bit de poids FAIBLE de l'index correspond au
    DERNIER qubit (Qiskit est little-endian), et 0 -> +1. Avec la
    convention naive (bits directs), l'ecart max vaut 1.88e+01 et la
    correlation 1.0e-04 — j'ai d'abord cru a une divergence des deux
    chemins avant de verifier ma propre convention. C'est la quatrieme
    fois de cette campagne qu'une reproduction incorrecte accuse du code
    juste ; le commentaire reste ici pour la cinquieme.
    """
    from study.common.ising_terms_and_annealing import build_ising_terms, total_energy
    from VQA.cost_hamiltonian import create_period_hamiltonian

    dim = 2
    nq = 2 * dim * dim
    hp = _coeffs(dim, graine)

    diag = np.real(np.diag(create_period_hamiltonian(hp, dim).to_matrix()))

    h, e, p = build_ising_terms(hp, dim)
    energies = np.array([
        total_energy(np.array([1 - 2 * x for x in bits[::-1]]), h, e, p)
        for bits in itertools.product([0, 1], repeat=nq)
    ])

    assert np.ptp(diag) > 1e-6, (
        "l'hamiltonien deploye est constant — le test ne separerait rien")
    ecart = np.abs(energies - diag).max()
    assert ecart < 1e-9, (
        f"graine {graine} : ecart max {ecart:.3e} entre le chemin study/ et "
        f"le chemin deploye. Mesure de reference : 5.3e-15.")


def test_le_producteur_xpoint_est_actif_dans_study():
    import ast
    import pathlib

    src = pathlib.Path(os.path.join(_REPO, "study", "common",
                                    "qaoa_inputs.py")).read_text()
    arbre = ast.parse(src)
    valeurs = []
    for n in ast.walk(arbre):
        if isinstance(n, ast.keyword) and n.arg == "advanced_anomalies_enabled":
            if isinstance(n.value, ast.Constant):
                valeurs.append(n.value.value)
    assert valeurs, "aucun `advanced_anomalies_enabled=` litteral trouve"
    assert all(v is True for v in valeurs), (
        f"`qaoa_inputs` code encore {valeurs} : study/ resterait aveugle au "
        f"terme de point X que l'entrainement active")
