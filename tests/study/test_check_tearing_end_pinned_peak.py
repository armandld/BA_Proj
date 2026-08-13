"""D-42 : `check_tearing` exige un pic strictement a l'interieur de la
trace ("pas a t=0, pas a la fin", cf. sa docstring). La clause "pas a t=0"
etait bien verifiee, mais la clause "pas a la fin" comparait
`j[i_peak]` a `j[min(i_peak+1, len(j)-1)]` : quand le pic tombe sur le
dernier echantillon, `min(...)` retombe sur `i_peak` lui-meme et la
comparaison devient toujours vraie (`j[i_peak] <= j[i_peak]*1.01`). Une
trace qui croit encore strictement a la fin de la fenetre simulee (donc
jamais observee redescendre : ce n'est pas un pic, c'est une croissance non
bornee dans la fenetre) etait quand meme acceptee.

Mesure sur les 6 fichiers DNS harris_tearing reels de results/ (voir
docs/RESULTS.md, D-42) : cablage gele (`mean_sq_current`), le pic tombe sur
le dernier echantillon (i_peak=19/20) sur 6/6 fichiers ; `check_tearing`
rendait `ok=True` (amplification 1.53-2.65x) sur les 6, uniquement a cause
de ce defaut.
"""
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dns_validation import check_tearing


def _old_check_tearing(res):
    """Epingle l'ancien comportement (avant D-42), pour que la correction
    ne puisse pas etre defaite en silence."""
    j = res["J2"]
    t = res["t"]
    i_peak = int(np.argmax(j))
    growing = (j[min(i_peak + 1, len(j) - 1)] <= j[i_peak] * 1.01)
    growing_from_start = (j[i_peak] > j[0] * 1.2) if len(j) > 1 else False
    return dict(t_peak=float(t[i_peak]),
                amplification=float(j[i_peak] / max(j[0], 1e-30)),
                ok=bool(growing_from_start and growing))


def test_old_behaviour_accepted_a_peak_pinned_at_the_last_sample():
    """Epingle le defaut : une trace strictement croissante jusqu'au
    dernier point (jamais retombee, donc jamais un pic observe) passait
    quand meme sous l'ancienne clause."""
    t = np.linspace(0, 2, 20)
    j_monotone_runaway = np.linspace(1.0, 5.0, 20)
    res = dict(t=t, J2=j_monotone_runaway)

    old = _old_check_tearing(res)
    assert old["ok"] is True
    assert np.argmax(j_monotone_runaway) == len(j_monotone_runaway) - 1


def test_peak_pinned_at_the_last_sample_is_rejected():
    """Correction : la meme trace, jamais retombee dans la fenetre
    simulee, doit maintenant echouer -- ce n'est pas un pic observe."""
    t = np.linspace(0, 2, 20)
    j_monotone_runaway = np.linspace(1.0, 5.0, 20)
    res = dict(t=t, J2=j_monotone_runaway)

    new = check_tearing(res)
    assert new["ok"] is False


def test_genuine_interior_peak_still_accepted():
    """Un vrai pic -- croissance puis decroissance nette avant la fin de
    la fenetre -- doit continuer a passer : la correction ne doit pas
    durcir le critere au-dela de ce que sa docstring promet."""
    t = np.linspace(0, 2, 20)
    j_peaked = np.concatenate([np.linspace(1.0, 5.0, 10),
                               np.linspace(5.0, 2.0, 10)])
    res = dict(t=t, J2=j_peaked)

    old = _old_check_tearing(res)
    new = check_tearing(res)
    assert old["ok"] is True
    assert new["ok"] is True


def test_real_harris_tearing_files_flip_with_frozen_wiring():
    """Rejoue D-42 sur les 6 fichiers DNS harris_tearing reels de
    results/, avec l'observable GELEE (mean_sq_current) -- celle dont le
    tableau D-39 rapportait ok=True. Les 6 doivent desormais rendre
    ok=False : leur pic tombe sur le dernier echantillon, ce n'etait pas
    un pic observe."""
    import glob

    from dns_validation import mean_sq_current

    results_dir = os.path.join(_REPO_ROOT, "results")
    paths = sorted(glob.glob(
        os.path.join(results_dir, "dns_harris_tearing_Re*_N*.npz")))
    assert len(paths) >= 6, (
        f"attendu au moins 6 fichiers harris_tearing dans {results_dir}, "
        f"trouve {len(paths)} -- balayage vide, rien n'est verifie")

    for path in paths:
        d = np.load(path)
        Bx = d["Bx"].astype(np.float64)
        By = d["By"].astype(np.float64)
        t = d["t"].astype(np.float64)
        n = Bx.shape[0]
        J2 = np.array([mean_sq_current(Bx[i], By[i]) for i in range(n)])
        res = dict(t=t, J2=J2)

        assert int(np.argmax(J2)) == n - 1, (
            f"{os.path.basename(path)}: attendu un pic au dernier "
            f"echantillon (regression du fichier DNS ou de la mesure)")

        old = _old_check_tearing(res)
        new = check_tearing(res)
        assert old["ok"] is True, os.path.basename(path)
        assert new["ok"] is False, os.path.basename(path)
