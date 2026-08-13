"""D-44 : `sanity_check.run_qaoa` decidait la convergence du QAOA sur
`np.std(marg) > 0.01` — la DISPERSION des marginales — alors que le critere
annonce par son propre commentaire est « marginals should not all be 0.5 »,
c'est-a-dire la DISTANCE a 0.5.

Les deux ne coincident pas, et le verdict s'inversait aux deux extremes.
Mesure sur les defauts du script (Re=400, N=32, dim=2, 4 scenarios) :

    scenario          bras  std(marg)  max|m-0.5|  ancien verdict
    harris_tearing    v1      0.0019      0.4800   NOT converged (flat)
    kelvin_helmholtz  v1      0.0014      0.2393   NOT converged (flat)
    orszag_tang       v1      0.0585      0.1769   converged

Les deux runs declares « flat » sont les deux plus tranches des huit
(marginales 0.976-0.980 pour harris_tearing) ; celui declare converge porte
une marginale a 0.517, soit 0.0169 de 0.5.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sanity_check as sc


# Les marginales reellement mesurees, relevees du run des defauts du script.
HARRIS_V1 = np.array([0.97618841, 0.97995398, 0.98002064, 0.97612489,
                      0.97615670, 0.97998737, 0.97998725, 0.97615661])


def test_unanimous_confident_marginals_are_converged():
    """
    Le champ qui SEPARE : des marginales toutes proches de 0.98 ont un
    ecart-type minuscule (1.9e-03) ET une distance a 0.5 maximale (0.480).
    Les deux hypotheses y donnent des reponses opposees.
    """
    converged, dist_max, dist_min, spread = sc.marginals_converged(HARRIS_V1)

    assert spread == pytest.approx(0.0019, abs=5e-4), \
        "l'ancien critere lisait bien un ecart-type sous sa tolerance de 0.01"
    assert spread < 0.01, "l'ancienne version declarait ce run NOT converged"

    assert dist_max == pytest.approx(0.4800, abs=1e-3)
    assert dist_min == pytest.approx(0.4761, abs=1e-3)
    assert converged is True


def test_all_marginals_at_half_are_not_converged():
    """La seule chose que le critere annonce doit refuser."""
    flat = np.full(8, 0.5)
    converged, dist_max, dist_min, spread = sc.marginals_converged(flat)
    assert dist_max == 0.0
    assert converged is False


def test_dispersed_but_undecided_is_still_undecided():
    """
    L'autre sens de l'erreur : des marginales dispersees mais toutes a
    moins de 0.005 de 0.5. L'ancien critere, s'il avait vu cette entree
    avec sa tolerance, aurait pu la declarer convergee sur la dispersion
    seule ; la distance a 0.5 tranche correctement.
    """
    near_half = np.array([0.4960, 0.5040, 0.4955, 0.5045,
                          0.4958, 0.5042, 0.4962, 0.5038])
    converged, dist_max, dist_min, spread = sc.marginals_converged(near_half)
    assert dist_max < 0.01
    assert converged is False


def test_criterion_reads_distance_not_dispersion():
    """
    Deux entrees de meme ecart-type, l'une tranchee l'autre non : le
    verdict doit differer. Sans cela le test ne distingue pas les deux
    grandeurs.
    """
    decided = np.array([0.90, 0.92, 0.90, 0.92])
    undecided = decided - 0.41          # meme std, centre sur 0.5
    assert np.std(decided) == pytest.approx(np.std(undecided), abs=1e-12)

    assert sc.marginals_converged(decided)[0] is True
    assert sc.marginals_converged(undecided, tol=0.02)[0] is False
