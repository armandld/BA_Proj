"""D-198/D-199 (USER 26 aout, "il faut etre absolument surs qu'ils ne
surapprennent pas") : `make_model("gbt", seed)` n'avait AUCUNE protection
reelle contre le surapprentissage sur la taille reelle d'un fold LOSO.

`HistGradientBoostingClassifier(early_stopping="auto")` (le defaut
sklearn, jamais explicite avant ce fichier) ne declenche l'arret anticipe
que si `n_samples > 10000` -- verifie directement sur un fold reel
(`n_train=1280`, `h2b_loso_transfer.py`) : `do_early_stopping_` reste
Faux, le modele va au bout de ses 300 iterations avec `l2_
regularization=0.0`. `make_model` accepte maintenant `early_stopping`
(defaut `"auto"`, litteralement l'ancien defaut sklearn -- comportement
bit-a-bit inchange pour les 11 AUTRES consommateurs de cette fonction,
non revisites ici) ; `early_stopping=True` impose l'arret anticipe et une
L2 non nulle plutot que de compter sur un seuil qui ne mord jamais aux
tailles de ce depot.

Champ d'essai synthetique (`make_classification`, ~1200 lignes, sous le
seuil de 10000) : rapide, et dans le meme regime de taille que les vrais
folds -- un jeu de 20000+ lignes ne separerait rien, "auto" y
declencherait deja l'arret anticipe des les deux bras.
"""
import pytest

from h2b_ceiling_random_split import make_model


@pytest.fixture(scope="module")
def small_fold_data():
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1200, n_features=9, n_informative=5, n_redundant=2,
        weights=[0.75, 0.25], random_state=0)
    return X, y


def test_default_matches_the_untouched_sklearn_default():
    """Les 11 autres appelants de `make_model` ne passent jamais
    `early_stopping` : leur construction doit rester identique."""
    m = make_model("gbt", 0)
    assert m.early_stopping == "auto"
    assert m.l2_regularization == 0.0


def test_auto_default_does_not_actually_stop_early_on_a_fold_sized_dataset(
        small_fold_data):
    """Le defaut nomme "auto" laisse croire a une protection qui n'agit
    pas a cette taille -- le test qui aurait du exister avant D-198."""
    X, y = small_fold_data
    m = make_model("gbt", 0)
    m.fit(X, y)
    assert m.do_early_stopping_ is False, (
        "si ceci devient Vrai, sklearn a change son seuil implicite : "
        "revisiter le commentaire de make_model")
    assert m.n_iter_ == m.max_iter


def test_opt_in_actually_regularises_on_the_same_data(small_fold_data):
    """Le contraste qui separe : meme donnee, seul `early_stopping`
    change, et ca doit se voir dans le modele ajuste, pas seulement dans
    le parametre demande."""
    X, y = small_fold_data
    m = make_model("gbt", 0, early_stopping=True)
    m.fit(X, y)
    assert m.do_early_stopping_ is True
    assert m.n_iter_ < m.max_iter, (
        "early_stopping=True n'a produit aucun arret anticipe reel sur "
        "ce jeu : la protection demandee n'a pas d'effet mesurable")
    assert m.l2_regularization > 0.0


def test_other_model_names_ignore_the_early_stopping_argument():
    """`early_stopping` n'a de sens que pour "gbt" : les autres noms ne
    doivent pas planter ni changer de comportement quand on le passe."""
    lr_default = make_model("lr", 0)
    lr_opt_in = make_model("lr", 0, early_stopping=True)
    assert type(lr_default) is type(lr_opt_in)

    rf_default = make_model("rf", 0)
    rf_opt_in = make_model("rf", 0, early_stopping=True)
    assert rf_default.get_params() == rf_opt_in.get_params()
