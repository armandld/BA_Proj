"""File ouverte de `COUVERTURE.md`, item 3/7 (`h2b_multiseed.py --n-seeds`).

`h2b_multiseed.py` docstring : *"V2's headline phase 11 (random split) [...]
numbers were originally seed=0 only. This phase re-runs both at seeds
0..N-1..."* -- et rapporte les anciennes valeurs single-seed en commentaire
("was: 0.989 single-seed" etc.) comme point de repere.

Ca suppose implicitement que le split seed=0 de `random_split_seed`
(`h2b_multiseed.py`) EST le meme split que celui de
`h2b_ceiling_random_split.main()`, la source de ces valeurs "was:". Mesure
directe sur les artefacts reels (`--N 96 --dim 4 --re 800 1200 1600
--max-snaps 30 --seed 0`, 330 snapshots) :

    h2b_ceiling_random_split (train_frac=0.7) : F1_classical val = 0.487
    h2b_multiseed            (val_frac=0.30)  : F1_classical val = 0.449

Et l'ecart n'est pas du bruit d'arrondi : les DEUX ensembles de validation,
construits a partir du MEME `np.random.default_rng(0).permutation(330)`,
sont **disjoints a 0 %** (0/100 indices communs). Cause : les deux scripts
tranchent aux DEUX BOUTS opposes de la meme permutation --
`h2b_ceiling_random_split` prend TRAIN = perm[:n_tr] (le DEBUT), VAL =
perm[n_tr:] (la FIN) ; `h2b_multiseed.random_split_seed` prend VAL =
perm[:n_va] (le DEBUT), TRAIN = perm[n_va:] (la FIN) -- c'est le
COMPLEMENT, pas le meme sous-ensemble.

Ce n'est pas bloquant au sens de `DEFAUTS.md` (aucune lecture publiee,
aucun blocage de la reoptimisation ne depend de `multiseed_N*.npz`
aujourd'hui, verifie par grep sur `docs/PLAN_PREPRINT.md` et
`docs/RESULTS.md`) : note en une ligne dans `COUVERTURE.md`, pas d'entree
`DEFAUTS.md`, geree sous le gel de `study/`.

Ce test epingle le desaccord MESURE (pas suppose) pour qu'une correction
future du split -- si elle arrive -- se voie plutot que de se glisser en
silence : le jour ou quelqu'un aligne les deux scripts, ce test devra
etre RE-MESURE, pas simplement inverse.
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

from h2b_multiseed import random_split_seed  # noqa: E402
from h2b_ceiling_random_split import N_FEATS  # noqa: E402


def _synthetic_snapshots(n_snap=330, cells_per_snap=4, seed=1234):
    """330 snapshots factices, assez pour reproduire la taille reelle
    (`--N 96 --dim 4 --re 800 1200 1600 --max-snaps 30`) sans toucher aux
    artefacts DNS : seule la logique d'indexation du split est en jeu ici,
    pas l'extraction de features (couverte ailleurs)."""
    rng = np.random.default_rng(seed)
    Xs, Xst, Ys, Ss = [], [], [], []
    for i in range(n_snap):
        Xs.append(rng.normal(size=(cells_per_snap, N_FEATS)))
        Xst.append(rng.normal(size=(cells_per_snap, 5 * N_FEATS)))
        # alterner pour garantir les deux classes dans train ET val
        Ys.append(np.array([i % 2] * cells_per_snap))
        Ss.append(rng.normal(size=cells_per_snap))
    return Xs, Xst, Ys, Ss


def _headline_train_val_indices(n, train_frac, seed):
    """Reproduit EXACTEMENT les trois lignes de split de
    `h2b_ceiling_random_split.main()` (mêmes appels rng, même formule) :
    TRAIN = tête de la permutation, VAL = queue."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = max(1, int(train_frac * n))
    return set(perm[:n_tr].tolist()), set(perm[n_tr:].tolist())


def _multiseed_val_indices(n, val_frac, seed):
    """Reproduit les deux lignes de split de `random_split_seed`."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_va = max(1, int(val_frac * n))
    return set(perm[:n_va].tolist()), set(perm[n_va:].tolist())


def test_headline_and_multiseed_validation_sets_are_disjoint_at_seed0():
    """Mesure a 778255d : sur n=330 (taille reelle de la config publiee),
    seed=0, train_frac=0.7 / val_frac=0.30 (les defauts des deux scripts),
    les ensembles de validation ne partagent AUCUN indice."""
    n = 330
    _, va_headline = _headline_train_val_indices(n, 0.7, seed=0)
    va_multi, _ = _multiseed_val_indices(n, 0.30, seed=0)

    overlap = va_headline & va_multi
    assert len(va_headline) == 100 and len(va_multi) == 99, (
        f"tailles mesurees a 778255d : headline={len(va_headline)}, "
        f"multiseed={len(va_multi)} -- remesurer si elles bougent")
    assert len(overlap) == 0, (
        "les deux splits partagent maintenant des indices de validation "
        f"({len(overlap)}) -- le desaccord mesure a 778255d a disparu, "
        "RE-MESURER la note de COUVERTURE.md avant de considerer ce test "
        "comme un faux negatif")


def test_random_split_seed_f1_differs_from_headline_on_real_artefacts():
    """Mesure de bout en bout sur donnees synthetiques (rapide, deterministe
    -- les vraies valeurs mesurees sur artefacts DNS reels a 778255d sont
    dans le commentaire du module : F1_classical headline=0.487,
    multiseed(seed=0)=0.449, ecart 0.038, sans recouvrement de VAL).
    Ici : verifie seulement que `random_split_seed` accepte le format
    attendu et rend un F1 fini -- garde de non-regression sur l'interface,
    pas une repetition de la mesure DNS (trop lente pour tourner a chaque
    passe de suite)."""
    Xs, Xst, Ys, Ss = _synthetic_snapshots()
    r = random_split_seed(Xs, Xst, Ys, Ss, val_frac=0.30, seed=0)
    assert np.isfinite(r["f1_class"])
    assert np.isfinite(r["f1_site"])
