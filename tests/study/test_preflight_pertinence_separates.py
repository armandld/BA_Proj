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
