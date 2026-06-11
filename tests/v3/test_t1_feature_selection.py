"""Tests V3 Task 1 : logique de selection LOSO (sans donnees DNS).

On injecte un modele leger (regression logistique) et un fit_fn minimal :
la pile qiskit / les .npz de phase 1-2 ne sont pas necessaires. Le chemin
reel (GBT de phase 11, folds phase 11b) est valide par le critere
d'acceptation du protocole (reproduction du 0.189 publie).
"""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "study", "v3"))
from t1_feature_selection import (
    classical_loso_f1,
    forward_selection,
    loso_f1_subset,
    select_columns,
)

SCENARIOS = ["sc_a", "sc_b", "sc_c"]
N_FEATS = 4
GOOD = 2  # seule feature dont le lien avec Y est stable entre scenarios


def _fit_fn(model, Xtr, Ytr, Xva, Yva):
    model.fit(Xtr, Ytr)
    p = model.predict_proba(Xva)[:, 1]
    return dict(f1=f1_score(Yva, (p > 0.5).astype(int), zero_division=0))


def _factory(seed):
    return LogisticRegression(max_iter=500, random_state=seed)


def _make_data(seed=0, n=400):
    """3 scenarios; la feature GOOD transfere, les autres sont du bruit
    dont la distribution depend du scenario (non transferables)."""
    rng = np.random.default_rng(seed)
    data = {}
    for k, sc in enumerate(SCENARIOS):
        y = rng.integers(0, 2, size=n)
        X = rng.normal(loc=3.0 * k, scale=1.0, size=(n, N_FEATS))
        X[:, GOOD] = y + rng.normal(scale=0.05, size=n)
        s = y + rng.normal(scale=0.10, size=n)  # score classique correle
        data[sc] = dict(X_site=X, Y=y, S=s)
    return data


def test_select_columns_sorts_to_canonical_order():
    X = np.arange(12).reshape(3, 4)
    out = select_columns(X, [3, 0])
    np.testing.assert_array_equal(out, X[:, [0, 3]])
    # l'ordre demande ne doit pas influer
    np.testing.assert_array_equal(out, select_columns(X, [0, 3]))


def test_loso_f1_good_feature_transfers():
    data = _make_data()
    pf = loso_f1_subset(data, SCENARIOS, [GOOD], seed=0,
                        model_factory=_factory, fit_fn=_fit_fn)
    assert set(pf) == set(SCENARIOS)
    assert all(v > 0.9 for v in pf.values())


def test_loso_f1_noise_features_do_not_transfer():
    data = _make_data()
    noise = [i for i in range(N_FEATS) if i != GOOD]
    pf = loso_f1_subset(data, SCENARIOS, noise, seed=0,
                        model_factory=_factory, fit_fn=_fit_fn)
    assert np.mean(list(pf.values())) < 0.75


def test_forward_selection_picks_good_feature_first():
    data = _make_data()
    path = forward_selection(data, SCENARIOS, N_FEATS, seed=0,
                             model_factory=_factory, fit_fn=_fit_fn)
    assert len(path) == N_FEATS
    assert path[0]["added"] == GOOD
    assert path[0]["mean"] > 0.9
    # chaque etape ajoute exactement une feature, sans repetition
    seen = [s["added"] for s in path]
    assert sorted(seen) == list(range(N_FEATS))
    assert path[-1]["selected"] == seen


def test_classical_loso_uses_train_threshold():
    data = _make_data()

    def thr_fn(scores, gt, grid=None):
        # seuil fixe connu : S ~ y + bruit(0.10), 0.5 separe parfaitement
        return 0.5, 1.0

    pf = classical_loso_f1(data, SCENARIOS, thr_fn=thr_fn)
    assert set(pf) == set(SCENARIOS)
    assert all(v > 0.95 for v in pf.values())
