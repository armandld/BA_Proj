"""D-83 : la phase 11H comparait deux bras appris, seuillés sur la VALIDATION,
à un bras classique seuillé sur le train.

Troisième site de la famille D-81 (phase 12) / D-82 (phase 11G). Dans
`h2b_random_split_bootstrap.py`, `thr_cls` vient de `(Str, Ytr)` — le train —
tandis que `thr_site` et `thr_sten` venaient de
`best_threshold_f1(concat(P_*_list), concat(Yv_list))`, c'est-à-dire des
probabilités et des labels de validation, sous un commentaire qui annonçait
« same protocol as fit_eval grid search ». `fit_eval` prend le sien sur
`(p_tr, Ytr)` : le commentaire disait le contraire de ce que le code faisait.

Ce que la discipline touche ici n'est pas décoratif : `delta site-cls`, son IC
bootstrap et `p(site <= class)` comparent les deux bras entre eux, et l'IC
rééchantillonne l'ensemble même qui a servi à fixer le seuil.

**Le biais est positif par construction** — un seuil qui maximise le F1 sur la
validation ne peut pas y faire moins bien que celui du train. Seule sa taille
est empirique, et elle est petite ici :

    --dim 4 --N 256 --max-snaps 80 --n-boot 500 --seed 0
        F1_site        0,937 -> 0,931
        delta site-cls +0,460 -> +0,454   IC [+0,371, +0,547] -> [+0,371, +0,534]
        F1_stencil     0,973 -> 0,973     (les deux seuils coïncident, 0,050)

    cinq configurations (dim 4/16/32, graines 0/1/2) : biais de +0,0004 à
    +0,0057 sur F1_site, aucun verdict imprimé ne change.

C'est dit tel quel plutôt qu'arrondi vers le haut : c'est la discipline qui est
en cause, pas la taille de cet écart-ci.

Les tests rejouent `main()` une fois, entrées liées symboliquement depuis
`results/`, sortie dans un répertoire temporaire — aucun artefact du dépôt
n'est touché. Ils n'importent rien de neuf : ils lisent l'artefact, donc ils
échouent tels quels sur la version d'avant D-83.
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

import h2b_random_split_bootstrap as phase11h  # noqa: E402

_SCENARIOS = ("orszag_tang", "harris_tearing", "kelvin_helmholtz", "mhd_rotor")


def _inputs():
    files = []
    for sc in _SCENARIOS:
        dns = os.path.join(_REPO_ROOT, "results", f"dns_{sc}_Re400_N256.npz")
        pat = os.path.join(_REPO_ROOT, "results",
                           f"patches_{sc}_Re400_N256_dim4.npz")
        if not (os.path.exists(dns) and os.path.exists(pat)):
            return None
        files += [dns, pat]
    return files


@pytest.fixture(scope="module")
def run(tmp_path_factory, request):
    """La configuration mesurée, jouée une fois pour tout le fichier."""
    files = _inputs()
    if files is None:
        pytest.skip("artefacts d'entrée N=256 dim=4 absents de ce checkout")
    out_dir = tmp_path_factory.mktemp("phase11h")
    for src in files:
        os.symlink(src, out_dir / os.path.basename(src))

    old_dir, old_argv = phase11h.RESULTS_DIR, sys.argv
    phase11h.RESULTS_DIR = str(out_dir)
    sys.argv = ["h2b_random_split_bootstrap.py", "--dim", "4", "--N", "256",
                "--max-snaps", "80", "--n-boot", "500", "--seed", "0"]
    try:
        phase11h.main()
    finally:
        phase11h.RESULTS_DIR, sys.argv = old_dir, old_argv
    return np.load(out_dir / "random_split_bootstrap_N256_dim4.npz",
                   allow_pickle=True)


def test_the_two_disciplines_differ_on_this_configuration(run):
    """Sans cet écart, le fichier entier ne mesurerait rien."""
    assert float(run["thr_site"]) != float(run["thr_site_on_val"]), (
        "les deux seuils du bras site coïncident : cette configuration ne "
        "sépare pas les deux disciplines et ne prouverait rien")
    assert float(run["thr_site"]) == pytest.approx(0.050, abs=1e-9)
    assert float(run["thr_site_on_val"]) == pytest.approx(0.140, abs=1e-9)


def test_the_reported_site_f1_is_the_train_thresholded_one(run):
    """0,931 et non 0,937 : c'est l'assertion qui tombe sur l'ancienne version."""
    assert float(run["f1_site"]) == pytest.approx(0.931, abs=5e-4)


def test_the_old_optimistic_number_is_kept_and_is_higher(run):
    """Épingle l'ancien comportement : il reste calculé, à côté du bon.

    Si le seuil de validation revient dans `f1_site`, les deux nombres
    deviennent égaux et ce test tombe.
    """
    biaise = float(run["f1_site_thr_on_val"])
    assert biaise == pytest.approx(0.937, abs=5e-4)
    assert biaise > float(run["f1_site"])
    assert biaise - float(run["f1_site"]) == pytest.approx(0.006, abs=1e-3)


def test_the_compared_delta_and_its_ci_move_with_it(run):
    """La quantité réellement comparée, celle qui porte le verdict."""
    assert float(run["delta_site_class"]) == pytest.approx(0.454, abs=5e-4)
    lo, hi = (float(v) for v in run["delta_site_class_ci"])
    assert (lo, hi) == (pytest.approx(0.371, abs=5e-4),
                        pytest.approx(0.534, abs=5e-4))
    # le verdict imprimé ne change pas à cette configuration : dit, pas caché
    assert float(run["p_site_le_class"]) == 0.0


def test_the_classical_arm_is_untouched(run):
    """Il prenait déjà son seuil sur le train : rien ne doit y bouger."""
    assert float(run["thr_class"]) == pytest.approx(0.150, abs=1e-9)
    assert float(run["f1_class"]) == pytest.approx(0.476, abs=5e-4)


def test_the_stencil_arm_is_unchanged_here_and_that_is_measured(run):
    """Sur cette configuration les deux seuils du bras stencil coïncident.

    Ce test ne prouve donc rien sur la discipline — il est là pour que
    l'égalité soit une mesure consignée et non une supposition, et pour
    crier si elle cesse d'être vraie.
    """
    assert float(run["thr_sten"]) == float(run["thr_sten_on_val"])
    assert float(run["f1_sten"]) == pytest.approx(
        float(run["f1_sten_thr_on_val"]), abs=1e-12)
    assert float(run["f1_sten"]) == pytest.approx(0.973, abs=5e-4)
