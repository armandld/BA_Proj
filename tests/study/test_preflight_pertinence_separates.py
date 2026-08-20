"""D-141 — le controle PERTINENCE du preflight ne separe pas le coefficient
d'un champ physique nu, ni de la baseline classique.

`study/common/preflight_coefficients.py` est la porte de la campagne : il
imprime « les coefficients font leur travail. Campagne possible. » avant
~224 h CPU. Son 4e controle affirme *« le coefficient correle avec l'erreur
REELLE DNS-vs-grossier, rho = 0.798 »* et accepte des que rho > 0.6.

Mesure : cinq grandeurs qui ne portent AUCUN coefficient, AUCUN
hyperparametre et ne passent PAS par `PhysicalMapper` franchissent le meme
seuil sur le meme etat DNS. Le score classique — la baseline meme que le
bras quantique doit battre — le franchit **plus haut** que le coefficient.

Le controle n'est pas vide pour autant : le bruit blanc le rate. Ce qu'il
mesure est « quelque chose se concentre dans la nappe », pas « le
coefficient fait son travail ».

Ces tests sont des tests de DEVIATION, comme ceux de D-53 : ils ne
pouvaient pas echouer sur un commit anterieur, ils echouent le jour ou le
controle gagne un critere de discrimination — c'est-a-dire le jour ou D-141
est tranche, et ou ce fichier doit etre relu.

Nombres mesures (2 executions, identiques au dernier chiffre) :

    K_plaquettes (ce que le controle regarde)   +0.7977
    score classique (la baseline)               +0.8137
    |Jz| courant                                +0.7429
    |v| module de la vitesse                    +0.7247
    |grad |B||                                  +0.6764
    K_xpoint                                    +0.4345
    bruit blanc (controle negatif)              -0.0401

Les assertions portent sur des ORDRES, pas sur ces valeurs : le depot
n'epingle aucune version de `numpy`/`scipy`, et un rang est robuste la ou
une valeur ne l'est pas. Les valeurs restent ecrites ci-dessus pour qu'une
derive se voie a la lecture.
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (os.path.join(_REPO_ROOT, "src"),
           os.path.join(_REPO_ROOT, "study", "common"),
           _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import preflight_coefficients as P                    # noqa: E402

SEUIL = 0.6                                            # celui du controle


@pytest.fixture(scope="module")
def rhos():
    """Rejoue le calcul du controle, puis substitue d'autres grandeurs.

    Le premier assert du premier test verifie que cette replique EST le
    controle — sans quoi on mesurerait un autre operateur et la comparaison
    ne dirait rien.
    """
    from scipy.stats import spearmanr

    nb, NF, NC, pas = 8, 128, 32, 200
    gf = P.PeriodicGrid(NF)
    sf = P.MHDSolver(gf, dt=1e-3, Re=P.RE, Rm=P.RM)
    sf.init_harris_tearing()
    gc = P.PeriodicGrid(NC)
    sc = P.MHDSolver(gc, dt=1e-3, Re=P.RE, Rm=P.RM)
    sc.init_harris_tearing()
    for _ in range(pas):
        sf.step_full()
    for _ in range(pas):
        sc.step_full()

    def bm(a):
        return a.reshape(nb, a.shape[0] // nb, nb, a.shape[0] // nb).mean(axis=(1, 3))

    ff, fc = sf.get_fluxes(), sc.get_fluxes()
    err = np.zeros((nb, nb))
    for v in ("vx", "vy", "Bx", "By"):
        d = bm(ff[v])
        c_ = bm(np.repeat(np.repeat(fc[v], NF // NC, 0), NF // NC, 1))
        err += np.abs(d - c_) / (np.abs(d).mean() + 1e-12)

    coeffs, score = P._coeffs(sf, gf)

    def rho(field):
        kb = bm(np.abs(np.asarray(field)))
        if np.ptp(kb) == 0:
            return float("nan")
        return float(spearmanr(kb.ravel(), err.ravel()).statistic)

    gx, gy = np.gradient(np.hypot(ff["Bx"], ff["By"]))
    rng = np.random.default_rng(0)
    return {
        "K_plaquettes": rho(coeffs["K_plaquettes"]),
        "K_xpoint": rho(coeffs["K_xpoint"]),
        "score_classique": rho(score),
        "Jz": rho(ff["Jz"]),
        "v_module": rho(np.hypot(ff["vx"], ff["vy"])),
        "grad_B": rho(np.hypot(gx, gy)),
        "bruit_blanc": rho(rng.random((NF, NF))),
    }


def test_the_replica_is_the_control_itself(rhos):
    """Operateur assorti : sans ceci, tout ce fichier mesure autre chose."""
    ok, mesures = P.controle_pertinence()
    assert ok, "le controle lui-meme echoue — D-141 est ecrit contre un OK"
    assert mesures["rho"] == pytest.approx(rhos["K_plaquettes"], abs=1e-12), (
        f"la replique rend {rhos['K_plaquettes']:+.6f}, le controle "
        f"{mesures['rho']:+.6f} : ce n'est pas le meme calcul")


def test_the_control_rejects_pure_noise(rhos):
    """Controle positif de ce fichier : le seuil n'est pas vide."""
    assert rhos["bruit_blanc"] < SEUIL, (
        f"le bruit blanc passe a {rhos['bruit_blanc']:+.4f} : le seuil "
        "n'ecarte plus rien, D-141 devient bien pire que decrit")


def test_bare_physical_fields_clear_the_same_threshold(rhos):
    """Le coeur de D-141 : aucun coefficient, aucun hyperparametre, et ca passe."""
    nus = {k: rhos[k] for k in ("Jz", "v_module", "grad_B")}
    passent = {k: v for k, v in nus.items() if v > SEUIL}
    assert len(passent) >= 3, (
        "D-141 ne se reproduit plus : les champs nus ne franchissent plus le "
        f"seuil du controle ({nus}). Le controle a peut-etre gagne un critere "
        "de discrimination — relire D-141 avant de supprimer ce test")


def test_the_classical_baseline_clears_it_better_than_the_coefficient(rhos):
    """Le point qui decide : la baseline que la campagne doit battre passe
    la porte de la campagne MIEUX que le coefficient qu'elle regle."""
    assert rhos["score_classique"] > SEUIL
    assert rhos["score_classique"] >= rhos["K_plaquettes"], (
        f"score classique {rhos['score_classique']:+.4f} contre K_plaquettes "
        f"{rhos['K_plaquettes']:+.4f} : l'ordre s'est inverse. C'est un "
        "resultat, pas une reparation — remesurer et relire D-141")


def test_only_one_of_the_four_channels_is_looked_at(rhos):
    """`K_xpoint` est un coefficient a part entiere et n'atteint pas le
    seuil. Le controle ne regarde que `K_plaquettes` : le nommer evite de
    lire « les coefficients » la ou un seul est mesure."""
    assert rhos["K_xpoint"] < SEUIL, (
        f"K_xpoint passe desormais a {rhos['K_xpoint']:+.4f} — remesurer "
        "D-141, sa portee a change")


# ── D-141, portee : lesquels des cinq controles voient la STRUCTURE ? ──
#
# La porte rend OK/ECHEC. Un controle qui reste vert sur un mappeur
# manifestement faux ne garde rien. On mute la SORTIE de
# `PhysicalMapper.compute_coefficients` -- jamais le controle -- et on
# regarde lesquels mordent.
#
# Matrice mesuree (deux executions identiques), `coincidence` exclu : il
# n'appelle pas `PhysicalMapper` du tout, il compare deux chemins de calcul
# d'energie sur des coefficients tires au hasard.
#
#   mutation                          specificite equilibre vivant pertinence
#   aucune (reference)                     OK        OK       OK       OK
#   axes transposes                        OK        OK       OK     ECHEC
#   K_plaq <-> K_xpoint                  ECHEC     ECHEC      OK     ECHEC
#   tout x1000                             OK        OK       OK       OK
#   K_xpoint mis a zero                    OK        OK     ECHEC      OK
#   coefficient = bruit                  ECHEC       OK       OK     ECHEC
#   melange spatial (meme distribution)    OK        OK       OK     ECHEC
#
# Lecture : `pertinence` est le SEUL des quatre a voir ou le coefficient
# met sa masse. Les trois autres sont des controles d'amplitude, et leurs
# docstrings ne promettent rien d'autre -- ce n'est pas un defaut de leur
# part. Ce qui compte pour D-141 est la conjonction : le seul controle
# sensible a la structure est aussi celui que la baseline franchit mieux.
#
# `tout x1000` passe les cinq, et c'est JUSTE : l'etat fondamental d'un
# Ising est invariant par mise a l'echelle positive uniforme des
# couplages. Ce n'est pas un trou, c'est une symetrie.

def _mute(transformation):
    """Applique une mutation a la sortie du mappeur, rend {controle: ok}."""
    import contextlib
    import io

    from Simulation.HamiltParams import PhysicalMapper

    vrai = PhysicalMapper.compute_coefficients
    PhysicalMapper.compute_coefficients = (
        lambda self, *a, **k: transformation(vrai(self, *a, **k)))
    try:
        out = {}
        for nom, fn, _desc in P.CONTROLES:
            with contextlib.redirect_stdout(io.StringIO()):
                out[nom] = bool(fn()[0])
        return out
    finally:
        PhysicalMapper.compute_coefficients = vrai


def _melange_spatial(coeffs):
    """Meme distribution de valeurs, structure spatiale detruite.

    C'est le champ d'essai qui SEPARE « le coefficient porte de
    l'information » de « le coefficient a la bonne amplitude » : les deux
    hypotheses donnent des reponses differentes ici, et seulement ici.
    """
    rng = np.random.default_rng(0)
    out = dict(coeffs)
    for k in ("K_plaquettes", "K_xpoint"):
        a = np.asarray(out[k]).copy()
        plat = a.ravel()
        rng.shuffle(plat)
        out[k] = plat.reshape(a.shape)
    return out


def test_the_cheap_controls_are_blind_to_a_spatial_shuffle():
    """`specificite` et `equilibre` restent verts sur un coefficient dont
    la structure spatiale est detruite. Les deux controles bon marche de la
    porte ne regardent que des amplitudes."""
    res = _mute(_melange_spatial)
    assert res["specificite"] is True, (
        "`specificite` mord desormais sur le melange spatial — la porte a "
        "gagne en portee, remesurer la matrice de D-141")
    assert res["equilibre"] is True, (
        "`equilibre` mord desormais sur le melange spatial — idem")


@pytest.mark.slow
def test_the_full_mutation_matrix_of_the_gate():
    """La matrice entiere, ~2 min : quatre mutations, quatre controles.

    Marque `slow` parce qu'elle rejoue `vivant` (200 pas a N=256) et
    `pertinence` (deux simulations) une fois par mutation. La commande
    existe pour que la matrice citee dans D-141 soit refaisable — un
    resultat qu'on ne sait pas refaire n'est pas un resultat.
    """
    def transposer(c):
        out = dict(c)
        for k in ("K_plaquettes", "K_xpoint"):
            out[k] = np.asarray(out[k]).T
        for k in ("C_edges", "H_edges"):
            out[k] = tuple(np.asarray(x).T for x in out[k])
        return out

    def eteindre_kxpoint(c):
        out = dict(c)
        out["K_xpoint"] = np.zeros_like(np.asarray(out["K_xpoint"]))
        return out

    attendu = {
        "aucune": ({}, dict(specificite=True, equilibre=True,
                            vivant=True, pertinence=True)),
        "transposee": (transposer, dict(specificite=True, equilibre=True,
                                        vivant=True, pertinence=False)),
        "kxpoint_zero": (eteindre_kxpoint, dict(specificite=True, equilibre=True,
                                                vivant=False, pertinence=True)),
        "melange": (_melange_spatial, dict(specificite=True, equilibre=True,
                                           vivant=True, pertinence=False)),
    }
    ecarts = []
    for nom, (mut, att) in attendu.items():
        res = _mute(mut if mut else (lambda c: c))
        for controle, valeur in att.items():
            if res[controle] is not valeur:
                ecarts.append(f"{nom}/{controle} : mesure {res[controle]}, "
                              f"matrice de D-141 {valeur}")
    assert not ecarts, (
        "la matrice de D-141 ne se reproduit plus — c'est un RESULTAT, pas "
        f"une reparation : remesurer et relire D-141. {ecarts}")
