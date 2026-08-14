"""D-80 : l'artefact de la phase 10 laissait tomber le champ `optimiser` — la
seule trace du repli CMA-ES -> Nelder-Mead.

`train()` prend soin de rendre `optimiser="cma" if use_cma else "nelder-mead"`.
L'ecriture le jetait :

    **{k: v for k, v in res.items() if not isinstance(v, str)}, tag_str=tag

`res` ne porte que deux chaines — `tag`, reajoutee juste apres sous `tag_str`,
et `optimiser`, qui ne l'etait pas. Quand `cma` n'est pas installe (paquet pip,
absent de bien des environnements), le script previent sur **une** ligne parmi
des centaines puis tourne en Nelder-Mead : l'artefact etait alors indiscernable
d'un vrai run CMA-ES, et `(c_bias*, thr_amr*)` s'y lit sans qu'on sache quel
optimiseur les a produits.

Mesure, meme commande, conteneur sans `cma`
(`--modes joint --n-iters 3 --sweeps 50 --n-restarts 1 --dim 2 --N 64`) :

    avant : cles = best_c_bias, best_f1_val, best_theta, best_thr,
            classical_f1, delta, f1_train_history, f1_val_history,
            hits_bound, tag_str, theta_history, train_pairs, val_fixed,
            val_pairs, w_zz, w_zzzz          <- aucun `optimiser`
    apres : optimiser='nelder-mead'  optimiser_requested='cma'
            cma_available=False      (et la meme colonne dans train_COMPARE)
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

_DNS = os.path.join(_REPO_ROOT, "results", "dns_orszag_tang_Re400_N64.npz")
_PATCHES = os.path.join(_REPO_ROOT, "results",
                        "patches_orszag_tang_Re400_N64_dim2.npz")


@pytest.fixture(scope="module")
def artefacts(tmp_path_factory):
    """Lance la phase 10 sur la plus petite configuration reelle, dans un
    dossier de sortie a part : aucun artefact du depot n'est touche."""
    if not (os.path.exists(_DNS) and os.path.exists(_PATCHES)):
        pytest.skip("artefacts d'entree absents de ce checkout")

    import h2b_train_linear_hamiltonian as phase10

    out = tmp_path_factory.mktemp("phase10")
    for src in (_DNS, _PATCHES):
        os.symlink(src, os.path.join(out, os.path.basename(src)))

    old_dir, old_argv = phase10.RESULTS_DIR, sys.argv
    phase10.RESULTS_DIR = str(out)
    sys.argv = ["phase10", "--modes", "joint", "--n-iters", "3",
                "--sweeps", "50", "--n-restarts", "1", "--dim", "2",
                "--N", "64", "--max-batch", "2", "--max-val", "2",
                "--scenario", "orszag_tang", "--re", "400",
                "--analytical-init", "none"]
    try:
        phase10.main()
    finally:
        phase10.RESULTS_DIR, sys.argv = old_dir, old_argv

    return (np.load(os.path.join(out, "train_joint_N64_dim2.npz"),
                    allow_pickle=False),
            np.load(os.path.join(out, "train_COMPARE_N64_dim2.npz"),
                    allow_pickle=False),
            phase10)


def test_the_run_artefact_names_the_optimiser_that_actually_ran(artefacts):
    run, _, phase10 = artefacts
    assert "optimiser" in run.files, (
        "l'artefact ne dit pas quel optimiseur l'a produit : un repli "
        "CMA-ES -> Nelder-Mead y est invisible (D-80)")
    expected = "cma" if phase10.HAS_CMA else "nelder-mead"
    assert str(run["optimiser"]) == expected, (
        f"l'artefact annonce {run['optimiser']!r} alors que `cma` est "
        f"{'disponible' if phase10.HAS_CMA else 'absent'} de cet "
        "environnement")


def test_the_artefact_distinguishes_what_was_asked_from_what_ran(artefacts):
    """Sans les deux, un repli est indiscernable d'un choix : `cma` demande
    et `nelder-mead` execute doivent se lire tous les deux."""
    run, _, phase10 = artefacts
    assert str(run["optimiser_requested"]) == "cma"
    assert bool(run["cma_available"]) is bool(phase10.HAS_CMA)
    if not phase10.HAS_CMA:
        assert str(run["optimiser"]) != str(run["optimiser_requested"]), (
            "sur un environnement sans `cma`, l'artefact doit montrer "
            "l'ecart entre demande et execution")


def test_the_cross_mode_table_carries_the_same_column(artefacts):
    """`train_COMPARE` croise les modes : deux lignes optimisees par deux
    optimiseurs differents y seraient indiscernables sans cette colonne."""
    _, compare, phase10 = artefacts
    assert "optimiser" in compare.files
    assert len(compare["optimiser"]) == len(compare["tags"])
    expected = "cma" if phase10.HAS_CMA else "nelder-mead"
    assert set(map(str, compare["optimiser"])) == {expected}


def test_the_fallback_is_reachable_at_all(artefacts):
    """Un test qui ne peut pas echouer est un defaut : si `cma` etait
    toujours present, ce fichier ne mesurerait que la branche facile. On le
    dit plutot que de le supposer."""
    _, _, phase10 = artefacts
    assert isinstance(phase10.HAS_CMA, bool)
    if phase10.HAS_CMA:
        pytest.skip("`cma` present ici : la branche de repli n'est pas "
                    "exercee par cette execution (elle l'a ete a la mesure "
                    "de D-80, sur un conteneur sans `cma`)")
