"""D-195 : la troisieme piste (« l'optimum exact du hamiltonien coincide
avec la decision classique sur cette configuration ») est refutee par
enumeration exhaustive, pas seulement plausible.

`h3_noise_robustness_exact_check.py` reconstruit EXACTEMENT le
hamiltonien de `test_qaoa_noise_and_early.py::test_noise_robustness`
(Orszag-Tang, sans bruit) et calcule son etat fondamental par
enumeration (memes fonctions que T13/D-53, rien de reimplemente).
L'etat fondamental exact est UNIFORME (aucune cellule raffinee) alors
que la decision classique en selectionne 2 sur 9 : il ne peut donc pas
y avoir coincidence entre les deux, il n'y a pas de decision non
triviale du cote de l'optimum exact.

Ce que « QAOA egale le classique » explique alors : le mecanisme deja
etabli par T11b et H0a (D-53/D-200) -- QAOA, a profondeur/budget
limites, reste proche de l'encodage initial derive du score classique
(theta = 2*arcsin(sqrt(score))) plutot que d'atteindre l'optimum reel
de son hamiltonien. Ici, l'optimum reel est degenere et sans interet
(raffiner nulle part) ; QAOA n'a donc ni la possibilite ni la raison
de s'en approcher, et son resultat continu ressemble au score classique
parce que c'est de la que son etat initial part.

Deviation test : casse si l'artefact est remplace ou si la configuration
de `test_noise_robustness` change.
"""
import os

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ARTIFACT = os.path.join(_REPO_ROOT, "results", "h3_noise_robustness_exact_check.npz")


@pytest.fixture(scope="module")
def artifact():
    if not os.path.exists(ARTIFACT):
        pytest.skip(f"artefact absent : {os.path.basename(ARTIFACT)}")
    return np.load(ARTIFACT, allow_pickle=True)


def test_the_classical_captured_fraction_matches_the_cited_measurement(artifact):
    """Verifie que ce script rejoue bien LA MEME configuration que
    `test_noise_robustness` -- pas une config voisine qui ressemblerait.
    DEFAUTS.md D-195 cite frac_cl=0,3189 pour Orszag-Tang sans bruit."""
    assert float(artifact["classical_captured_fraction"]) == pytest.approx(0.3189, abs=5e-5)


def test_the_exact_ground_state_is_uniform(artifact):
    assert bool(artifact["exact_ground_state_uniform"]) is True
    assert float(artifact["exact_fraction_refined"]) == 0.0


def test_the_exact_ground_state_cannot_coincide_with_a_nontrivial_classical_selection(
        artifact):
    """Le coeur de la refutation : un optimum uniforme ne peut pas
    coincider avec une selection classique non triviale."""
    budget = int(artifact["budget"])
    n_cells = int(artifact["n_blocks"]) ** 2
    assert 0 < budget < n_cells, "la selection classique doit etre non triviale"
    assert float(artifact["exact_fraction_refined"]) in (0.0, 1.0)
    assert budget / n_cells not in (0.0, 1.0)


def test_the_ground_state_is_degenerate(artifact):
    """Signale la degenerescence pour qu'un futur lecteur sache qu'un
    SEUL representant a ete inspecte, pas les n_optima etats a egalite —
    une limite de cette mesure, pas une erreur."""
    assert int(artifact["n_optima"]) > 1
