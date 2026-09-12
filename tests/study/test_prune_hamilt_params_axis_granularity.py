"""File ouverte de `COUVERTURE.md`, item 1/7 (`h3_depth_report.py --prune-eps`).

`prune_hamilt_params` (`study/common/qaoa_inputs.py`) porte cette docstring :

    Zero out coefficients with |coeff| < eps * max(|coeff|) in each block.
    [...] We prune each block independently so a very strong H_i doesn't
    kill all C/K terms (they live on different scales).

Lue naivement, « block » designe H / C / K — trois groupes. Le CODE va plus
loin : il normalise H0 separement de H1, et C0 separement de C1 (chacun des
deux tableaux d'un `H_edges`/`C_edges` a SON PROPRE eps*max). C'est une
granularite plus fine que ce que la docstring decrit litteralement.

Sur le champ synthetique analytique deja mesure pour `qaoa_inputs.py
--prune-eps` (COUVERTURE.md, ligne « traite — rien trouve »), les deux
lectures COINCIDENT : max|H0| = max|H1| = 330.5 par construction du champ.
Ce fichier mesure le meme code sur le champ REEL que consomme
`h3_depth_report.py --prune-eps` en production (`scripts=1`) et montre que
la coincidence NE SE GENERALISE PAS : sur `harris_tearing`, un scenario a
fort cisaillement anisotrope, `max|C1|` est 423 a 660x plus petit que
`max|C0|` (dim=2..32, mesure au snapshot median de
`results/dns_harris_tearing_Re400_N256.npz`). A eps=0.05 (le premier eps non
nul des defauts CLI de `h3_depth_report.py`), dim=4 :

    per-array (le code)     : C0 survit 8/16, C1 survit 16/16 -> 24 survivants
    per-block combine (la docstring lue au pied de la lettre) : C0 survit
    8/16, C1 survit 0/16 -> 8 survivants

Un facteur 3x sur le nombre de termes C survivants, donc sur la profondeur
de circuit compilee que `h3_depth_report.py` rapporte pour ce scenario.

Ce n'est PAS un defaut au sens de VIGIL.md : le comportement du CODE sert le
but enonce par la docstring ("un H_i fort ne doit pas tuer tous les termes
C/K") au moins aussi bien que la lecture "par bloc" — il protege MEME contre
la domination d'un axe sur l'autre a l'interieur du meme bloc C, ce que la
lecture "par bloc" ne fait pas. Aucun nombre publie n'en depend
(`depth_report.csv` n'est cite dans aucun document de `docs/`, verifie par
grep). Consigne dans COUVERTURE.md, pas DEFAUTS.md (regle d'arret : ni
lecture publiee, ni blocage de la reoptimisation).

Ce test epingle la granularite REELLE (per-array) pour que la mesure ne se
perime pas en silence si `prune_hamilt_params` change un jour de
granularite sans que quiconque ne remarque l'ecart avec la docstring.
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

_RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
_DNS_PATH = os.path.join(_RESULTS_DIR, "dns_harris_tearing_Re400_N256.npz")

pytestmark = pytest.mark.skipif(
    not os.path.exists(_DNS_PATH),
    reason=f"artefact DNS manquant : {_DNS_PATH}",
)

from qaoa_inputs import prune_hamilt_params  # noqa: E402
from exact_diagonalisation import build_patch_hamiltonian  # noqa: E402


def _hp_harris_dim4():
    dns = np.load(_DNS_PATH)
    si = len(dns["t"]) // 2
    vx = dns["vx"][si].astype(np.float64)
    vy = dns["vy"][si].astype(np.float64)
    Bx = dns["Bx"][si].astype(np.float64)
    By = dns["By"][si].astype(np.float64)
    hp, _, _ = build_patch_hamiltonian(
        vx, vy, Bx, By, 256, 4, 400, threshold_amr=0.15, use_v2=True)
    return hp


def _block_combined_prune(hp, eps):
    """La lecture 'par bloc' de la docstring : un seul eps*max pour C0 ET
    C1 ensemble, au lieu d'un eps*max par tableau (ce que le code fait
    reellement)."""
    import copy
    hp2 = copy.deepcopy(hp)
    C0, C1 = (np.asarray(a, dtype=float).copy() for a in hp2["C_edges"])
    m = max(float(np.max(np.abs(C0))), float(np.max(np.abs(C1))))
    if m > 0:
        C0[np.abs(C0) < eps * m] = 0.0
        C1[np.abs(C1) < eps * m] = 0.0
    hp2["C_edges"] = (C0, C1)
    return hp2


def test_harris_tearing_c_edges_axes_have_very_different_scale():
    """Precondition du reste du fichier : verifie que le champ choisi
    SEPARE bien les deux lectures (VIGIL.md, 'choisir un champ qui separe').
    Mesure : maxC0/maxC1 = 579.08 sur ce champ, dim=4, snapshot median."""
    hp = _hp_harris_dim4()
    C0, C1 = hp["C_edges"]
    ratio = float(np.max(np.abs(C0))) / float(np.max(np.abs(C1)))
    assert ratio > 100.0, (
        f"le champ ne separe plus les deux lectures (ratio mesure {ratio:.2f}, "
        "579.08 attendu a 778255d) -- ne pas conclure sur ce champ")


def test_code_prunes_per_array_not_per_block():
    """Epingle la granularite REELLE de prune_hamilt_params : C0 et C1 sont
    normalises chacun par SON PROPRE max, pas par le max combine du bloc C.
    eps=0.05, dim=4, harris_tearing Re400 N256, snapshot median."""
    hp = _hp_harris_dim4()

    hp_code = prune_hamilt_params(hp, 0.05)
    c0_code, c1_code = hp_code["C_edges"]
    nnz_c0_code = int(np.count_nonzero(c0_code))
    nnz_c1_code = int(np.count_nonzero(c1_code))

    hp_block = _block_combined_prune(hp, 0.05)
    c0_block, c1_block = hp_block["C_edges"]
    nnz_c0_block = int(np.count_nonzero(c0_block))
    nnz_c1_block = int(np.count_nonzero(c1_block))

    # Mesure a 778255d (cette passe) : per-array garde C1 quasi intact
    # (16/16, son propre eps*max ne coupe rien), per-bloc le viderait (0/16,
    # ecrase par l'echelle de C0).
    assert nnz_c1_code == 16, (
        f"C1 (per-array) attendu 16/16 survivants a eps=0.05, mesure {nnz_c1_code}")
    assert nnz_c1_block == 0, (
        f"C1 (per-bloc combine) attendu 0/16 survivants a eps=0.05, mesure {nnz_c1_block}")
    assert nnz_c0_code == nnz_c0_block == 8, (
        "C0 doit coincider entre les deux lectures (son propre max domine "
        f"deja le bloc combine) : code={nnz_c0_code} bloc={nnz_c0_block}")

    total_code = nnz_c0_code + nnz_c1_code
    total_block = nnz_c0_block + nnz_c1_block
    assert total_code == 24 and total_block == 8, (
        f"total survivants : per-array={total_code} (attendu 24), "
        f"per-bloc={total_block} (attendu 8) -- facteur mesure "
        f"{total_code / total_block:.2f}x")
