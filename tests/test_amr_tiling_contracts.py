"""Audit de contrat de l'arbre AMR : la liste de patchs pave-t-elle le domaine ?

C'est l'invariant le plus elementaire d'un AMR — chaque cellule appartient a
exactement une feuille — et rien ne le verifiait. Une liste qui recouvre
deux fois la meme region reste une liste de patchs parfaitement plausible :
bornes valides, profondeurs coherentes, scores dans [0, 1]. Seule une somme
de couverture la distingue d'une liste juste.

DEFAUT TROUVE ICI : le sondage de bord etait un bloc SEPARE, execute apres
la ventilation. Quand il se declenchait, le sous-patch avait deja ete
enregistre comme feuille non raffinee par la branche `else`, et il etait
en plus pousse au niveau suivant. La meme region etait donc comptee deux
fois : une fois comme feuille grossiere, une fois redecoupee.

Mesure sur Orszag-Tang, Kelvin-Helmholtz, rotor et Harris (N=256, dim=2,
max_depth=3, 6 instantanes chacun) :

| seuil | configurations avec recouvrement | pire cas |
|---|---|---|
| **0.1496 (deploye)** | 2 / 24 | **25.0 %** du domaine |
| 0.30 | 6 / 24 | 25.0 % |
| 0.40 | 9 / 24 | 28.1 % |
| 0.50 | 12 / 24 | 20.3 % |

Toute metrique de budget ou de couverture lue sur la liste finale
surcomptait d'autant. Le balayage de seuils de la frontiere de Pareto passe
exactement dans la zone ou le defaut est le plus frequent.

La correction fond le sondage dans le meme if/elif/else : un sous-patch est
soit raffine, soit feuille, jamais les deux.
"""

import os
import sys

import numpy as np
import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from Simulation.PhysToAngle import AngleMapper  # noqa: E402
from Simulation.grid import PeriodicGrid  # noqa: E402
from Simulation.refinement import (  # noqa: E402
    _boundary_activation,
    run_adaptive_classical,
)
from Simulation.solver import MHDSolver  # noqa: E402

_REPO = os.path.dirname(_SRC)
N = 256


def _sim_at(scenario, snap):
    path = os.path.join(_REPO, "results", f"dns_{scenario}_Re400_N{N}.npz")
    if not os.path.exists(path):
        pytest.skip(f"artefact absent : {os.path.basename(path)}")
    d = np.load(path)
    ns = d["vx"].shape[0]
    idx = min(snap, ns - 1)
    sim = MHDSolver(PeriodicGrid(N), Re=400, Rm=400)
    for k in ("vx", "vy", "Bx", "By"):
        setattr(sim, k, d[k][idx].copy())
    return sim


def _coverage(patches):
    c = np.zeros((N, N), dtype=int)
    for p in patches:
        y_s, y_e, x_s, x_e = p["bounds"]
        c[y_s:y_e, x_s:x_e] += 1
    return c


def _run(scenario, snap, thr, target_dim=2, max_depth=3):
    patches, _ = run_adaptive_classical(
        _sim_at(scenario, snap), AngleMapper(), threshold_amr=thr,
        target_dim=target_dim, max_depth=max_depth, min_size=4, verbose=False)
    return patches


_SCEN = ["orszag_tang", "kelvin_helmholtz", "mhd_rotor", "harris_tearing"]


# ======================================================================
#  1. Le pavage : ni recouvrement, ni trou
# ======================================================================

@pytest.mark.parametrize("scenario", _SCEN)
@pytest.mark.parametrize("thr", [0.1496, 0.30, 0.50])
def test_the_patch_list_tiles_the_domain_exactly(scenario, thr):
    """Chaque cellule appartient a exactement une feuille."""
    c = _coverage(_run(scenario, 10, thr))
    over = int(np.sum(c > 1))
    hole = int(np.sum(c == 0))
    assert over == 0, (
        f"{over} cellules ({over / N ** 2:.2%}) couvertes par plusieurs "
        f"patchs : toute metrique de budget lue sur cette liste surcompte")
    assert hole == 0, (
        f"{hole} cellules ({hole / N ** 2:.2%}) ne sont couvertes par aucun "
        "patch : elles n'entrent dans aucune decision")


@pytest.mark.parametrize("thr", [0.1496, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65])
def test_the_tiling_holds_across_the_whole_pareto_threshold_range(thr):
    """Le balayage de seuils passe par la zone ou le defaut etait frequent."""
    c = _coverage(_run("orszag_tang", 0, thr))
    assert c.max() == 1 and c.min() == 1


@pytest.mark.parametrize("target_dim", [2, 4])
@pytest.mark.parametrize("max_depth", [1, 2, 3])
def test_the_tiling_holds_at_every_patch_size_and_depth(target_dim, max_depth):
    c = _coverage(_run("orszag_tang", 0, 0.55, target_dim=target_dim,
                       max_depth=max_depth))
    assert c.max() == 1 and c.min() == 1


def test_the_covered_area_equals_the_domain_area():
    """Verification independante du recomptage cellule par cellule."""
    patches = _run("mhd_rotor", 5, 0.30)
    area = sum((p["bounds"][1] - p["bounds"][0]) * (p["bounds"][3] - p["bounds"][2])
               for p in patches)
    assert area == N * N


def test_no_two_patches_share_the_same_bounds():
    """Un doublon exact ferait raffiner deux fois la meme region."""
    for thr in (0.1496, 0.40, 0.65):
        patches = _run("harris_tearing", 5, thr)
        bounds = [p["bounds"] for p in patches]
        assert len(bounds) == len(set(bounds)), "bornes dupliquees"


# ======================================================================
#  2. Ce que chaque patch PRETEND porter
# ======================================================================

def test_every_patch_carries_the_keys_the_pipeline_reads():
    for p in _run("orszag_tang", 10, 0.1496):
        assert set(p) >= {"bounds", "depth", "type"}
        assert len(p["bounds"]) == 4
        y_s, y_e, x_s, x_e = p["bounds"]
        assert 0 <= y_s < y_e <= N and 0 <= x_s < x_e <= N


def test_every_patch_type_is_one_of_the_two_declared_kinds():
    types = {p["type"] for p in _run("kelvin_helmholtz", 10, 0.30)}
    assert types <= {"coarse_leaf", "leaf_depth"}, types


def test_patch_scores_stay_in_the_unit_interval():
    for p in _run("orszag_tang", 10, 0.30):
        if "score" in p:
            assert 0.0 <= float(p["score"]) <= 1.0


def test_a_leaf_at_max_depth_is_never_deeper_than_max_depth():
    max_depth = 3
    for p in _run("orszag_tang", 10, 0.30, max_depth=max_depth):
        assert p["depth"] <= max_depth


# ======================================================================
#  3. Monotonie : un seuil plus haut ne peut pas raffiner davantage
# ======================================================================

def test_raising_the_threshold_never_increases_the_number_of_patches():
    """Un critere qui raffine PLUS quand on lui demande MOINS serait casse."""
    counts = [len(_run("orszag_tang", 0, t)) for t in
              (0.10, 0.20, 0.30, 0.45, 0.60, 0.80)]
    for a, b in zip(counts, counts[1:]):
        assert b <= a, f"le nombre de patchs remonte : {counts}"


def test_an_impossible_threshold_refines_nothing_beyond_the_first_split():
    """Aucun score ne depasse 1, donc aucune cellule ne passe le seuil.

    Le BFS decoupe toujours la racine une fois avant de tester — on obtient
    donc dim^2 feuilles grossieres a la profondeur 0, et rien de plus. Une
    seule descente supplementaire signifierait qu'un score a franchi un
    seuil qu'aucun score ne peut atteindre.
    """
    patches = _run("orszag_tang", 10, 1.01)
    assert len(patches) == 2 ** 2
    assert all(p["type"] == "coarse_leaf" and p["depth"] == 0 for p in patches)
    assert _coverage(patches).max() == 1


def test_a_zero_threshold_refines_everywhere():
    """Tout score est >= 0 : l'arbre doit descendre au bout partout."""
    patches = _run("orszag_tang", 10, 0.0, max_depth=2)
    assert all(p["type"] == "leaf_depth" for p in patches)
    assert len(patches) == 4 ** 2      # dim=2, deux descentes


# ======================================================================
#  4. _boundary_activation, la porte qui declenchait le sondage
# ======================================================================

def test_a_uniformly_active_patch_raises_no_boundary_flag():
    """Rien ne 'touche le bord' si tout est actif : ce serait un faux positif
    generalise, et c'est le sondage qui doublait les patchs."""
    assert _boundary_activation(np.full((4, 4), 0.9), 4) == {}


def test_a_uniformly_calm_patch_raises_no_flag():
    assert _boundary_activation(np.zeros((4, 4)), 4) == {}


@pytest.mark.parametrize("side,expected", [
    ("top", {"top"}), ("bottom", {"bottom"}),
    ("left", {"left"}), ("right", {"right"}),
])
def test_each_border_is_recognised_for_itself(side, expected):
    m = np.zeros((4, 4))
    {"top": lambda: m.__setitem__((0, slice(None)), 1.0),
     "bottom": lambda: m.__setitem__((-1, slice(None)), 1.0),
     "left": lambda: m.__setitem__((slice(None), 0), 1.0),
     "right": lambda: m.__setitem__((slice(None), -1), 1.0)}[side]()
    assert set(_boundary_activation(m, 4)) == expected


def test_an_interior_anomaly_raises_no_boundary_flag():
    m = np.zeros((5, 5))
    m[2, 2] = 1.0
    assert _boundary_activation(m, 5) == {}


def test_a_patch_too_small_to_have_an_interior_returns_no_flag():
    assert _boundary_activation(np.full((1, 1), 1.0), 1) == {}


# ======================================================================
#  5. Les deux bras doivent rester structurellement identiques
# ======================================================================

def test_both_arms_share_the_same_probe_structure():
    """Une divergence de code entre les deux bras ferait mesurer a leur
    comparaison la difference d'implementation autant que celle du critere."""
    import inspect

    from Simulation import refinement as R
    for fn in (R._run_level, R._run_level_classical):
        src = inspect.getsource(fn)
        assert "should_probe = (" in src, (
            f"{fn.__name__} n'a pas la forme corrigee du sondage")
        assert src.count("elif should_probe:") == 1, (
            f"{fn.__name__} : le sondage doit etre une branche du meme "
            "if/elif/else, pas un bloc separe qui rajoute apres coup")


def test_the_logged_threshold_is_the_one_actually_applied():
    """Le journal annonçait une rampe en profondeur que le code n'applique
    plus : toute lecture des decisions dans le journal etait fausse."""
    import inspect

    from Simulation import refinement as R
    src = inspect.getsource(R._run_level)
    assert "effective_thr = threshold_amr\n" in src
    assert "effective_thr = threshold_amr + (1.0 - threshold_amr)" not in src
