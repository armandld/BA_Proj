"""D-48 — le « warm start classique » du QAOA est un schedule CONSTANT.

Ces tests **épinglent une déviation connue**, ils ne verrouillent pas une
correction : le comportement de `classical_warm_start_params` est inchangé,
bit-à-bit, par la passe qui a ouvert D-48. Ils ne pouvaient donc pas échouer
sur la version précédente, et c'est délibéré — `VIGIL.md`, « Ne jamais
laisser une déviation connue non écrite » : la décision de ne pas corriger
s'écrit là où elle vit, avec sa mesure, et un test vérifie qu'elle y reste.

Ce qu'ils font échouer : le jour où quelqu'un lie le schedule au score
classique — ce qui serait le comportement que le nom annonce — le premier
test tombe. C'est voulu : ce changement déplace `progress` (T11b,
`docs/RESULTS.md`), donc il doit être mesuré et tranché, pas glissé.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# Les six entrées de la mesure D-48, choisies pour SÉPARER : si le schedule
# dépendait du score ou du seuil de quelque façon que ce soit, deux de ces
# six au moins donneraient des sorties différentes.
_CASES = [
    ("score nul", np.zeros((2, 2)), 0.15),
    ("score unité", np.ones((2, 2)), 0.15),
    ("score aléatoire", np.random.default_rng(0).random((2, 2)), 0.15),
    ("seuil 0", np.ones((2, 2)), 0.0),
    ("seuil 1", np.ones((2, 2)), 1.0),
    ("seuil 1e9", np.ones((2, 2)), 1e9),
]


@pytest.mark.parametrize("reps", [1, 2, 3, 4])
def test_schedule_ignores_score_and_threshold(reps):
    """Sortie identique BIT-À-BIT sur les six entrées — écart mesuré 0,0e+00.

    Tombe si le schedule devient dépendant du score : c'est exactement le
    changement qui demande une remesure de `progress` avant d'être adopté.
    """
    from qaoa_inputs import classical_warm_start_params

    outs = [classical_warm_start_params(sc, thr, reps) for _, sc, thr in _CASES]
    ref = outs[0]
    for (name, _, _), out in zip(_CASES, outs):
        assert np.array_equal(out, ref), (
            f"D-48 : le schedule dépend maintenant de l'entrée « {name} ». "
            f"Ce n'est pas forcément une régression — c'est peut-être la "
            f"correction attendue — mais elle déplace `progress` (T11b) : "
            f"remesurer et consigner l'ancienne et la nouvelle valeur avant "
            f"de mettre ce test à jour.")


@pytest.mark.parametrize("reps", [1, 2, 3, 4])
def test_schedule_values_are_the_published_ones(reps):
    """β = 0,05 partout, γ = 0,15/k : les valeurs sur lesquelles T11 et T11b
    ont été obtenus. Écrites en clair pour qu'une dérive se voie."""
    from qaoa_inputs import classical_warm_start_params

    out = classical_warm_start_params(np.ones((2, 2)), 0.15, reps)
    assert out.shape == (2 * reps,)
    np.testing.assert_array_equal(out[:reps], np.full(reps, 0.05))
    np.testing.assert_allclose(out[reps:], 0.15 / np.arange(1, reps + 1),
                               rtol=0, atol=0)


def test_mixer_start_diverges_from_execute_cold_path():
    """Les deux chemins d'initialisation du dépôt ne s'accordent pas sur β₀.

    `execute()` démarre le mixer à β = 0 — « Beta/Omega (mixer) — must start
    at zero » — et `RESULTS.md` chiffre ce que coûte un β non nul sans coût
    pour le justifier : marginales 0,7000 → 0,5535 sur un hamiltonien nul.
    Le schedule D-48 démarre à β = 0,05. L'écart est petit et mesuré sans
    conséquence sur la décision (0 différence sur 4 scénarios canoniques,
    dim=2, reps=2), mais il est réel : on l'épingle plutôt que de le laisser
    se redécouvrir.
    """
    from qaoa_inputs import classical_warm_start_params

    reps = 2
    beta_warm = classical_warm_start_params(np.ones((2, 2)), 0.15, reps)[:reps]
    assert np.all(beta_warm != 0.0)
    # et il reste sous la borne que `execute()` impose au mixer
    beta_max = np.pi / (4 * reps)
    assert np.all(np.abs(beta_warm) <= beta_max)


def test_deviation_stays_written_where_it_lives():
    """La mention D-48 reste dans la docstring de la fonction concernée.

    Une déviation connue mais non écrite *là où elle vit* se fait recorriger
    par erreur — c'est déjà arrivé sur `dns_validation` dans ce dépôt. On
    interroge le module, pas le texte du fichier.
    """
    import inspect

    from qaoa_inputs import classical_warm_start_params

    doc = inspect.getdoc(classical_warm_start_params) or ""
    assert "D-48" in doc, (
        "la docstring ne porte plus le renvoi D-48 : sans lui, la prochaine "
        "lecture croira que le warm start dérive du score classique")
