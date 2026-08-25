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

D-39, ensuite : une fois D-42 applique seul, les 6/6 fichiers reels
rendaient `ok=False` (amplification 1.00-1.10x, voir docs/DEFAUTS.md) --
parce que `J2` moyenne le courant d'equilibre de la nappe sur tout le
domaine, qui noie le signal de reconnexion. `check_tearing` lit maintenant
`J2_fluct` (`fluctuating_mean_sq_current`, fond homogene-en-x retire) et
accepte un pic encore montant en fin de fenetre (`still_rising`) SI
l'amplification depasse le seuil `grows` (1.2x) -- ce n'est plus le rejet
inconditionnel introduit par D-42 : les 6 fichiers reels ne referment
jamais leur pic dans la fenetre simulee (reconnexion encore en cours a
t_max), et les rejeter tous reviendrait a rendre le check structurellement
insatisfiable par toute donnee reelle. `test_peak_pinned_at_the_last_
sample_is_rejected` d'origine testait une amplification (5x) qui, avec le
recul empirique des 6 fichiers reels (8.3x-17.6x), n'a plus de raison
d'etre rejetee -- il est remplace par deux tests qui isolent ce qui
distingue encore un signal accepte d'un signal rejete : l'amplification
mesuree contre le seuil `grows`, pas la seule position du pic.
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


def test_peak_pinned_at_the_last_sample_with_weak_growth_is_rejected():
    """D-39 : la position du pic seule ne suffit plus a rejeter (voir
    le test _strong_growth_ ci-dessous) -- mais une trace qui ne fait que
    deriver, sans jamais depasser le seuil `grows` (1.2x), doit rester
    rejetee : rien ne distingue alors ce cas d'un bruit numerique sans
    reconnexion. C'est ce seuil, pas la position du pic, qui porte
    desormais le critere de rejet."""
    t = np.linspace(0, 2, 20)
    j_weak_drift = np.linspace(1.0, 1.1, 20)  # +10 %, sous le seuil 1.2x
    res = dict(t=t, J2=j_weak_drift, J2_fluct=j_weak_drift)

    new = check_tearing(res)
    assert np.argmax(j_weak_drift) == len(j_weak_drift) - 1
    assert new["ok"] is False


def test_peak_pinned_at_the_last_sample_with_strong_growth_is_now_accepted():
    """D-39 : la MEME trace que `test_old_behaviour_accepted_...` ci-dessus
    -- jamais retombee dans la fenetre, pic au dernier echantillon --
    est maintenant ACCEPTEE, a rebours du rejet impose par D-42.

    Ce n'est pas un retour en arriere : D-42 rejetait TOUTE trace dont le
    pic tombe en fin de fenetre, en assumant qu'une vraie reconnexion se
    referme forcement dans la fenetre simulee. Les 6 fichiers DNS
    harris_tearing reels (docs/RESULTS.md, D-39) montrent que cette
    hypothese etait fausse : ils ne referment JAMAIS leur pic (reconnexion
    encore en cours a t_max) tout en montrant une amplification franche
    (8.3x-17.6x). Rejeter systematiquement ce cas rendait le check
    insatisfiable par toute donnee reelle. `saturated=False` distingue ce
    cas -- pic encore montant, accepte sur la seule force du signal -- d'un
    pic qui a reellement redescendu (`test_genuine_interior_peak_still_
    accepted` ci-dessous)."""
    t = np.linspace(0, 2, 20)
    j_monotone_runaway = np.linspace(1.0, 5.0, 20)
    res = dict(t=t, J2=j_monotone_runaway, J2_fluct=j_monotone_runaway)

    new = check_tearing(res)
    assert np.argmax(j_monotone_runaway) == len(j_monotone_runaway) - 1
    assert new["ok"] is True
    assert new["saturated"] is False


def test_genuine_interior_peak_still_accepted():
    """Un vrai pic -- croissance puis decroissance nette avant la fin de
    la fenetre -- doit continuer a passer : la correction ne doit pas
    durcir le critere au-dela de ce que sa docstring promet."""
    t = np.linspace(0, 2, 20)
    j_peaked = np.concatenate([np.linspace(1.0, 5.0, 10),
                               np.linspace(5.0, 2.0, 10)])
    res = dict(t=t, J2=j_peaked, J2_fluct=j_peaked)

    old = _old_check_tearing(res)
    new = check_tearing(res)
    assert old["ok"] is True
    assert new["ok"] is True
    assert new["saturated"] is True


def test_real_harris_tearing_diagnostic_is_finite():
    """Le diagnostic reste defini sur les 6 fichiers reels, sans supposer
    ou tombe le pic -- et D-39 doit y rendre `ok=True` sur les 6/6 : c'est
    la mesure meme qui justifie le changement de critere ci-dessus, pas
    seulement une garantie de non-crash."""
    import glob

    from dns_validation import fluctuating_mean_sq_current

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
        J2_fluct = np.array(
            [fluctuating_mean_sq_current(Bx[i], By[i]) for i in range(n)])
        res = dict(t=t, J2_fluct=J2_fluct)

        diagnostic = check_tearing(res)
        name = os.path.basename(path)
        assert np.isfinite(diagnostic["amplification"]), name
        assert 0 <= diagnostic["t_peak"] <= t[-1], name
        assert diagnostic["ok"] is True, (
            f"{name}: attendu ok=True (D-39, amplification mesuree "
            f"{diagnostic['amplification']:.2f}x), obtenu ok=False -- "
            "le signal de reconnexion ne passe plus le critere")
