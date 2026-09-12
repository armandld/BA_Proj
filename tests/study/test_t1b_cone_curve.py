"""Tests V3 Task 1b : voisinages k-hop, split bloque, courbe LOSO.

Modele leger injecte (regression logistique) : ni qiskit ni donnees DNS.
La reproduction exacte du stencil publie (k=1 LOSO = 0.215) est validee
par le critere d'acceptation du protocole sur les vraies donnees.
"""

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from h2b_feature_selection import loso_f1_subset
from h2b_neighbour_cone_curve import (

    blocked_split_indices,
    capped_model_factory,
    khop_distinct_footprint,
    khop_features,
    khop_offsets,
)


# ----------------------------- offsets ------------------------------

def test_khop_offsets_counts_match_protocol():
    # 9, 45, 225, 441 features pour F=9 -> 1, 5, 25, 49 cellules
    assert [len(khop_offsets(k)) for k in (0, 1, 2, 3)] == [1, 5, 25, 49]


def test_khop_offsets_self_first_and_unique():
    for k in (0, 1, 2, 3):
        offs = khop_offsets(k)
        assert offs[0] == (0, 0)
        assert len(set(offs)) == len(offs)


def test_khop_offsets_k1_matches_phase11_stencil_order():
    # phase 11 : [self, N, S, E, W] avec N = roll(-1, axis=0) = source (i+1, j)
    assert khop_offsets(1) == [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


# ----------------------------- features -----------------------------

def test_khop_features_shapes():
    feats = np.random.default_rng(0).normal(size=(4, 4, 9))
    for k, nf in zip((0, 1, 2, 3), (9, 45, 225, 441)):
        assert khop_features(feats, k).shape == (16, nf)


def test_khop_features_k0_is_identity():
    feats = np.random.default_rng(1).normal(size=(4, 4, 3))
    np.testing.assert_array_equal(khop_features(feats, 0),
                                  feats.reshape(-1, 3))


def test_khop_features_k1_periodic_neighbours():
    dim = 4
    v = np.arange(dim * dim, dtype=float).reshape(dim, dim, 1)  # v[i,j]=4i+j
    X = khop_features(v, 1)
    for i in range(dim):
        for j in range(dim):
            row = X[i * dim + j]
            expected = [
                v[i, j, 0],
                v[(i + 1) % dim, j, 0],   # N (roll -1 axe 0)
                v[(i - 1) % dim, j, 0],   # S
                v[i, (j + 1) % dim, 0],   # E
                v[i, (j - 1) % dim, 0],   # W
            ]
            np.testing.assert_array_equal(row, expected)


def test_khop_features_k1_equals_phase11_stencil_construction():
    # reproduction de stencil_features sans importer la chaine qiskit :
    # memes np.roll, meme ordre de concatenation
    feats = np.random.default_rng(2).normal(size=(4, 4, 9))
    f_n = np.roll(feats, -1, axis=0)
    f_s = np.roll(feats, +1, axis=0)
    f_e = np.roll(feats, -1, axis=1)
    f_w = np.roll(feats, +1, axis=1)
    ref = np.concatenate([feats, f_n, f_s, f_e, f_w], axis=-1).reshape(-1, 45)
    np.testing.assert_array_equal(khop_features(feats, 1), ref)


# ----------------- empreinte reelle sur grille periodique (D-88) -----------
#
# Reserve initialement : "a dim=4, le voisinage k=2 couvre deja toute la
# grille periodique, donc k=3 n'est pas un voisinage plus grand -- et
# `n_feats` compte les colonnes dupliquees". Mesure, cf.
# `khop_distinct_footprint` : a dim=4, footprint(2) = footprint(3) = 16 = 4*4
# (toute la grille), alors que les comptes NOMINAUX de `khop_offsets`
# (25 puis 49, deja pins par le test ci-dessus) suggeraient une croissance.
# Avant ce module, rien dans le fichier ne distinguait les deux -- le
# tableau imprime et l'artefact sauvegarde ne portaient que le compte
# nominal, plausible (une grille de features 15x15... 21x21 existe bien en
# theorie) mais faux sur CETTE grille periodique.

def test_khop_distinct_footprint_k2_covers_whole_dim4_grid():
    dim = 4
    assert khop_distinct_footprint(2, dim) == dim * dim


def test_khop_distinct_footprint_k3_is_not_larger_than_k2_at_dim4():
    dim = 4
    # sanity : le compte NOMINAL grandit bien (c'est lui qui trompe)
    assert len(khop_offsets(3)) > len(khop_offsets(2))
    # le compte REEL, lui, ne grandit pas : k=3 n'est pas un voisinage
    # plus grand que k=2 sur une grille periodique 4x4
    assert khop_distinct_footprint(3, dim) == khop_distinct_footprint(2, dim)


def test_khop_distinct_footprint_matches_actual_unique_columns():
    """Mesure a l'operateur assorti : comparaison directe aux colonnes
    REELLEMENT produites par `khop_features` (pas une reimplementation
    parallele des decalages)."""
    dim = 4
    feats = np.random.default_rng(7).normal(size=(dim, dim, 9))
    for k in (0, 1, 2, 3):
        X = khop_features(feats, k)
        n_unique = np.unique(X, axis=1).shape[1]
        assert n_unique == khop_distinct_footprint(k, dim) * 9


def test_khop_distinct_footprint_duplicated_columns_at_dim4():
    # n_feats "nominal" (celui qu'affichait le script avant D-88) contre le
    # nombre reel de colonnes distinctes : l'ecart EST le defaut.
    dim = 4
    nf_nominal = {k: len(khop_offsets(k)) * 9 for k in (0, 1, 2, 3)}
    nf_distinct = {k: khop_distinct_footprint(k, dim) * 9 for k in (0, 1, 2, 3)}
    assert nf_nominal == {0: 9, 1: 45, 2: 225, 3: 441}
    assert nf_distinct == {0: 9, 1: 45, 2: 144, 3: 144}
    # k=0 et k=1 n'ont pas encore de doublons (2k+1 < dim=4)
    assert nf_nominal[0] == nf_distinct[0]
    assert nf_nominal[1] == nf_distinct[1]
    # k=2 et k=3 en ont, et k=3 n'ajoute AUCUNE colonne distincte de plus
    assert nf_nominal[2] > nf_distinct[2]
    assert nf_nominal[3] > nf_distinct[3]
    assert nf_distinct[2] == nf_distinct[3]


def test_khop_distinct_footprint_does_not_saturate_on_a_larger_grid():
    """Champ qui separe : sur une grille assez grande (dim=8), 2k+1 <= 7 < 8
    pour tout k <= 3, donc AUCUNE saturation -- le defaut D-88 est propre a
    dim=4, pas un artefact du calcul lui-meme."""
    dim = 8
    assert [khop_distinct_footprint(k, dim) for k in (0, 1, 2, 3)] == \
        [len(khop_offsets(k)) for k in (0, 1, 2, 3)]


# --------------------------- split bloque ----------------------------

def test_blocked_split_indices_60_40():
    tr, va = blocked_split_indices(30, 0.6)
    assert tr == list(range(18))
    assert va == list(range(18, 30))


def test_blocked_split_indices_temporal_order_and_edges():
    tr, va = blocked_split_indices(2, 0.6)
    assert tr == [0] and va == [1]          # jamais de val vide
    tr, va = blocked_split_indices(10, 0.95)
    assert va == [9]                        # train plafonne a n-1
    assert max(tr) < min(va)                # train strictement avant val


# --------------------------- modele plafonne -------------------------

def test_capped_model_factory_sets_sqrt_fraction():
    base = lambda s: HistGradientBoostingClassifier(random_state=s)
    m = capped_model_factory(441, 0, base_factory=base)
    if m is None:  # sklearn < 1.4 : indisponible, signale par le script
        return
    assert abs(m.get_params()["max_features"] - np.sqrt(441) / 441) < 1e-12


# ----------------------- bout-en-bout synthetique --------------------

def _fit_fn(model, Xtr, Ytr, Xva, Yva):
    model.fit(Xtr, Ytr)
    p = model.predict_proba(Xva)[:, 1]
    return dict(f1=f1_score(Yva, (p > 0.5).astype(int), zero_division=0))


def _factory(seed):
    return LogisticRegression(max_iter=500, random_state=seed)


def test_neighbour_label_needs_k1():
    """Label = signe de la feature du voisin nord -> k=0 aveugle, k=1 voit."""
    rng = np.random.default_rng(3)
    dim, n_snaps = 4, 60
    data = {0: {}, 1: {}}
    for sc in ("sc_a", "sc_b"):
        Xs = {0: [], 1: []}
        Ys = []
        for _ in range(n_snaps):
            f = rng.normal(size=(dim, dim, 1))
            y = (np.roll(f[:, :, 0], -1, axis=0) > 0).ravel().astype(int)
            for k in (0, 1):
                Xs[k].append(khop_features(f, k))
            Ys.append(y)
        for k in (0, 1):
            data[k][sc] = dict(X_site=np.concatenate(Xs[k]),
                               Y=np.concatenate(Ys))

    scenarios = ["sc_a", "sc_b"]
    f1_k0 = np.mean(list(loso_f1_subset(
        data[0], scenarios, [0], seed=0,
        model_factory=_factory, fit_fn=_fit_fn).values()))
    f1_k1 = np.mean(list(loso_f1_subset(
        data[1], scenarios, list(range(5)), seed=0,
        model_factory=_factory, fit_fn=_fit_fn).values()))
    assert f1_k1 > 0.95
    assert f1_k1 - f1_k0 > 0.2
