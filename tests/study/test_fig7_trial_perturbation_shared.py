"""D-104 — `fig7_physical_fidelity.py` perturbait ses trois simulations avec
trois tirages DIFFERENTS.

fig7 construit trois simulations explicitement identiques ("Create 3
identical sims") : `dns`, `qaoa`, `classical`. Tout ce que la figure
rapporte — `field_l2_error(sims['qaoa'], sims['dns'])`, l'energie cinetique
et l'enstrophie des trois bras — suppose que le seul ecart entre elles est
la decision d'AMR.

Le bloc « Add tiny perturbation for trial independence » creait `rng` une
fois puis le consommait dans la boucle `for lbl in sims` : chaque
simulation recevait donc un bruit different. Des `trial = 1`, les trois ne
partaient plus du meme etat, et la courbe « Rel. L2 Error » mesurait la
divergence de deux conditions initiales, pas l'erreur de l'AMR.

Mesure (`init_harris_tearing`, N=256, warmup=80, 3 pas d'AMR, trial=1) :

| grandeur | avant | apres | reference trial=0 |
|---|---|---|---|
| L2(qaoa, dns) a t=0, avant tout AMR | 1,4122e-05 | 0,0 | 0,0 |
| L2(classical, dns) a t=0 | 1,4127e-05 | 0,0 | 0,0 |
| L2(qaoa, dns) apres warmup, avant AMR | 2,020e-06 | 0,0 | 0,0 |
| L2(qaoa, dns) apres 3 pas d'AMR | 1,6795e-05 | 8,695e-07 | 8,182e-07 |
| L2(classical, dns) apres 3 pas d'AMR | 2,104e-06 | 8,185e-07 | 8,182e-07 |
| rapport Q-HAS / classique | **7,98** | 1,06 | 1,000 |

Soit x20,5 sur l'erreur annoncee, et un ecart x8,0 entre deux bras qui, a
`trial = 0`, rendent des valeurs bit-a-bit identiques.

`N_TRIALS = 1` dans le fichier committe : le bloc fautif ne s'executait
jamais. C'est la question 5 de `VIGIL.md` — une configuration qu'aucun
essai n'emprunte — et la question 1 : un piege arme, invisible a tout audit
de couverture.

Aucun nombre publie ne bouge : aucune figure `results/figures/fig7_*` n'est
committee dans ce depot.

Les tests portent sur le comportement du fichier committe (fonction
extraite par AST puis executee), pas sur son texte source.
"""
import ast
import os
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_V1_LEGACY = os.path.join(_REPO_ROOT, "figures", "v1_legacy")
_FIG7 = os.path.join(_V1_LEGACY, "fig7_physical_fidelity.py")
for _p in (_V1_LEGACY, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIELDS = ("vx", "vy", "Bx", "By")


class _FakeSim:
    """Porteur de champs — `perturb_trial` ne touche que des attributs."""

    def __init__(self, arrays):
        for name, arr in arrays.items():
            setattr(self, name, arr.copy())


def _base_arrays(n=16, seed=7):
    rng = np.random.default_rng(seed)
    return {fn: rng.standard_normal((n, n)) for fn in FIELDS}


def _three_identical_sims(n=16, seed=7):
    base = _base_arrays(n, seed)
    return {lbl: _FakeSim(base) for lbl in ("dns", "qaoa", "classical")}, base


def _load_perturb_trial():
    """Extrait `perturb_trial` du fichier committe, sans executer le script.

    fig7 produit sa figure a l'import : on ne peut pas l'importer. On
    compile la seule definition de fonction, et on execute celle-la.
    """
    with open(_FIG7, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=_FIG7)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "perturb_trial"),
              None)
    if fn is None:                                    # pragma: no cover
        pytest.fail(
            "fig7_physical_fidelity.py n'expose plus `perturb_trial` : "
            "la perturbation d'essai est redevenue un bloc inline, donc "
            "non testable — c'est l'etat d'avant D-104.")
    module_ast = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module_ast)
    g = {"np": np}
    exec(compile(module_ast, _FIG7, "exec"), g)       # noqa: S102
    return g["perturb_trial"]


def _old_perturbation(sims, trial, eps=1e-5):
    """Le bloc d'avant D-104, reproduit mot pour mot.

    Un seul `rng`, consomme dans la boucle `for lbl in sims`.
    """
    if trial > 0:
        rng = np.random.default_rng(trial)
        for lbl in sims:
            for fn in FIELDS:
                f = getattr(sims[lbl], fn)
                rms = max(np.std(f), 1e-10)
                setattr(sims[lbl], fn,
                        f + eps * rms * rng.standard_normal(f.shape))


def _max_rel_gap(sims, a="qaoa", b="dns"):
    """Ecart relatif max entre deux sims, sur les quatre champs."""
    worst = 0.0
    for fn in FIELDS:
        fa, fb = getattr(sims[a], fn), getattr(sims[b], fn)
        scale = max(float(np.max(np.abs(fb))), 1e-30)
        worst = max(worst, float(np.max(np.abs(fa - fb))) / scale)
    return worst


# ══════════════════════════════════════════════════════════════════
#  1. L'ancien comportement, epingle — la correction ne peut pas
#     etre defaite en silence
# ══════════════════════════════════════════════════════════════════

def test_lancien_bloc_rend_les_trois_sims_differentes():
    """Le defaut lui-meme, sur le bloc d'avant D-104.

    Si un jour ce test passe, c'est que la boucle a ete rendue partagee
    ailleurs — et il faudra le dire, pas le laisser verdir.
    """
    sims, _ = _three_identical_sims()
    _old_perturbation(sims, trial=1)
    assert _max_rel_gap(sims, "qaoa", "dns") > 0.0, (
        "l'ancien bloc rendait bien trois etats differents")
    assert _max_rel_gap(sims, "classical", "dns") > 0.0
    assert _max_rel_gap(sims, "classical", "qaoa") > 0.0


def test_lancien_bloc_a_trial_zero_ne_perturbe_rien():
    """La raison pour laquelle personne ne l'a vu : `N_TRIALS = 1`."""
    sims, base = _three_identical_sims()
    _old_perturbation(sims, trial=0)
    for lbl in sims:
        for fn in FIELDS:
            assert np.array_equal(getattr(sims[lbl], fn), base[fn])


# ══════════════════════════════════════════════════════════════════
#  2. La garantie annoncee : les trois sims restent identiques
# ══════════════════════════════════════════════════════════════════

def test_les_trois_sims_restent_bit_a_bit_identiques():
    """La garantie que « Create 3 identical sims » promet, a trial > 0."""
    perturb_trial = _load_perturb_trial()
    for trial in (1, 2, 5):
        sims, _ = _three_identical_sims()
        perturb_trial(sims, trial)
        for fn in FIELDS:
            ref = getattr(sims["dns"], fn)
            for lbl in ("qaoa", "classical"):
                assert np.array_equal(getattr(sims[lbl], fn), ref), (
                    f"trial={trial}, champ {fn} : les sims divergent avant "
                    "tout pas d'AMR")


def test_la_perturbation_est_bien_appliquee_et_a_la_bonne_echelle():
    """« Identiques » ne doit pas etre obtenu en ne perturbant plus rien.

    L'assertion porte sur la garantie annoncee (un decalage d'ordre
    `eps * rms`), pas sur l'absence de plantage.
    """
    perturb_trial = _load_perturb_trial()
    sims, base = _three_identical_sims(n=64, seed=3)
    eps = 1e-5
    perturb_trial(sims, 1, eps=eps)
    for fn in FIELDS:
        delta = getattr(sims["dns"], fn) - base[fn]
        assert not np.array_equal(delta, np.zeros_like(delta)), (
            f"{fn} n'a pas ete perturbe du tout")
        rms = max(float(np.std(base[fn])), 1e-10)
        observed = float(np.std(delta)) / (eps * rms)
        # ecart-type d'un tirage normal reduit : 1, a la variance
        # d'echantillonnage pres (64x64 tirages)
        assert 0.9 < observed < 1.1, (
            f"{fn} : echelle de perturbation {observed:.3f} au lieu de ~1")


def test_les_essais_restent_independants_entre_eux():
    """Partager le tirage ENTRE les sims ne doit pas le partager entre
    les essais : c'est ce que `trial independence` demande."""
    perturb_trial = _load_perturb_trial()
    sims1, base = _three_identical_sims()
    sims2, _ = _three_identical_sims()
    perturb_trial(sims1, 1)
    perturb_trial(sims2, 2)
    for fn in FIELDS:
        assert not np.array_equal(getattr(sims1["dns"], fn),
                                  getattr(sims2["dns"], fn)), (
            f"{fn} : deux essais recoivent le meme tirage")


def test_trial_zero_ne_perturbe_toujours_rien():
    """Le seul essai que le fichier committe execute aujourd'hui."""
    perturb_trial = _load_perturb_trial()
    sims, base = _three_identical_sims()
    perturb_trial(sims, 0)
    for lbl in sims:
        for fn in FIELDS:
            assert np.array_equal(getattr(sims[lbl], fn), base[fn])


# ══════════════════════════════════════════════════════════════════
#  3. Sur le vrai etat MHD : le nombre mesure, ecrit pour qu'une
#     derive se voie
# ══════════════════════════════════════════════════════════════════

def test_sur_le_vrai_champ_lancien_bloc_decale_les_sims_de_1_4e_05():
    """`init_harris_tearing`, N=256, a la construction (aucun pas de temps).

    Nombre mesure et consigne : 1,4122e-05 pour (qaoa, dns) — l'ecart que
    la courbe L2 de fig7 rapportait comme une erreur d'AMR alors qu'aucun
    pas d'AMR n'avait encore eu lieu. La reference sans perturbation vaut
    8,182e-07 apres 3 pas d'AMR : le decalage initial est 17 fois plus
    grand que ce que la figure mesure.
    """
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver
    from fig_utils import field_l2_error

    def _fresh():
        sims = {}
        for lbl in ("dns", "qaoa", "classical"):
            s = MHDSolver(PeriodicGrid(resolution_N=256), dt=1e-3,
                          Re=800, Rm=800)
            s.init_harris_tearing()
            sims[lbl] = s
        return sims

    old = _fresh()
    _old_perturbation(old, trial=1)
    l2_old = field_l2_error(old["qaoa"], old["dns"])
    assert l2_old == pytest.approx(1.4122e-05, rel=2e-3), (
        f"mesure de reference D-104 derivee : {l2_old:.6e} au lieu de "
        "1,4122e-05 — remesurer, ne pas ajuster le seuil")

    new = _fresh()
    _load_perturb_trial()(new, 1)
    assert field_l2_error(new["qaoa"], new["dns"]) == 0.0
    assert field_l2_error(new["classical"], new["dns"]) == 0.0
