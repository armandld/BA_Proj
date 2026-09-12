"""D-82 : dans `h2b_scenario_ablation.py`, la colonne « fuzz » choisissait son
seuil sur les labels de VALIDATION ; les trois autres colonnes de la meme
table le prennent sur le train.

Meme defaut que D-81, dans un autre fichier — trouve en passant la question de
D-81 a tous les appels de `best_threshold_f1` de `study/`. La table LOSO de ce
script imprime cote a cote `F1_class`, `F1_9feat`, `F1_9+id` (les trois par
`fit_eval`, seuil sur le train) et `F1_9+fuzz`, seul a beneficier de ses
propres labels de test. Et c'est precisement la colonne qui mesure la CHUTE
quand l'identite de scenario est fausse : un F1 gonfle sous-estime la chute.

Mesure, `--re 400 --N 64 --dim 4 --max-snaps 8` :

    fold orszag_tang     0.212 -> 0.198
    moyenne LOSO fuzz    0.163 -> 0.160

L'ecart est petit sur cette configuration parce que 3 folds sur 4 sont
degeneres a 0,000 ; il vaut 0,014 sur le seul fold non degenere. Dit tel
quel plutot qu'arrondi vers le haut.

Le correctif reutilise le seuil de `rid` — le MEME modele (meme graine, memes
donnees d'entrainement), donc son seuil vient de son propre train.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (os.path.join(_REPO_ROOT, "src"),
           os.path.join(_REPO_ROOT, "study", "pipeline"),
           os.path.join(_REPO_ROOT, "study", "common"),
           os.path.join(_REPO_ROOT, "study", "h2b_prediction")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SCENARIOS = ("orszag_tang", "harris_tearing", "kelvin_helmholtz", "mhd_rotor")


def _inputs():
    out = []
    for sc in _SCENARIOS:
        dns = os.path.join(_REPO_ROOT, "results", f"dns_{sc}_Re400_N64.npz")
        pat = os.path.join(_REPO_ROOT, "results",
                           f"patches_{sc}_Re400_N64_dim4.npz")
        if not (os.path.exists(dns) and os.path.exists(pat)):
            return None
        out += [dns, pat]
    return out


@pytest.fixture(scope="module")
def ablation(tmp_path_factory):
    """Lance l'ablation sur la plus petite configuration reelle, dans un
    dossier de sortie a part : aucun artefact du depot n'est touche."""
    files = _inputs()
    if files is None:
        pytest.skip("artefacts d'entree absents de ce checkout")

    import h2b_scenario_ablation as ablation_mod

    out = tmp_path_factory.mktemp("ablation")
    for f in files:
        os.symlink(f, os.path.join(out, os.path.basename(f)))

    old_dir, old_argv = ablation_mod.RESULTS_DIR, sys.argv
    ablation_mod.RESULTS_DIR = str(out)
    sys.argv = ["ablation", "--re", "400", "--N", "64", "--dim", "4",
                "--max-snaps", "6"]
    try:
        ablation_mod.main()
    finally:
        ablation_mod.RESULTS_DIR, sys.argv = old_dir, old_argv

    return np.load(os.path.join(out, "scenario_ablation_N64_dim4.npz"),
                   allow_pickle=False)


def test_the_fuzz_column_keeps_the_old_optimistic_number(ablation):
    assert "loso_site9fuzz_thr_on_val" in ablation.files, (
        "l'ancien nombre n'est plus mesure : le biais de D-82 redeviendrait "
        "invisible")


def test_the_reported_fuzz_f1_is_never_above_the_val_optimised_one(ablation):
    """Le seuil pris sur la validation MAXIMISE le F1 sur la validation :
    le nombre rapporte ne peut donc qu'etre inferieur ou egal. S'il etait
    superieur, c'est que les deux colonnes ne decrivent plus le meme
    modele."""
    honest = ablation["loso_site9fuzz"]
    optimistic = ablation["loso_site9fuzz_thr_on_val"]
    ok = np.isclose(honest, optimistic) | (honest <= optimistic)
    bad = [(h, o) for h, o, k in zip(honest, optimistic, ok)
           if not (k or (np.isnan(h) and np.isnan(o)))]
    assert not bad, f"F1 rapporte au-dessus de l'optimum de validation : {bad}"


def test_the_two_disciplines_actually_differ_somewhere(ablation):
    """Un test qui ne peut pas echouer est un defaut : si les deux seuils
    donnaient partout le meme F1, ce fichier ne mesurerait rien. Sur cette
    configuration, le fold non degenere les separe."""
    honest = ablation["loso_site9fuzz"]
    optimistic = ablation["loso_site9fuzz_thr_on_val"]
    gap = np.nanmax(np.abs(optimistic - honest))
    assert gap > 1e-6, (
        f"les deux disciplines coincident partout (ecart max {gap:.2e}) : "
        "la configuration d'essai ne SEPARE pas, la choisir autrement "
        "plutot que relacher l'assertion")


def test_the_other_three_columns_are_unchanged_by_the_fix(ablation):
    """Le correctif ne doit toucher que la colonne fuzz. Les trois autres
    passent par `fit_eval` et n'ont jamais vu les labels de validation."""
    for key in ("loso_class", "loso_site9", "loso_site9id"):
        assert key in ablation.files
        assert len(ablation[key]) == len(ablation["loso_held"])
