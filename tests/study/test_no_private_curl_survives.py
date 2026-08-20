"""D-17 — l'operateur « legacy » n'est pas un rotationnel de signe oppose.

D-1 avait corrige le rotationnel dans `src/`. Le balayage s'etait arrete la.
Trois sites hors de `src/Simulation/` reimplementaient encore leur propre
rotationnel sous la convention inverse (axis=1 lu comme x) :

  study/h2b_prediction/h2b_v1_hamiltonian_loso.py   jz_from_b
  study/h2b_prediction/h2b_ceiling_random_split.py  omega_z, J_z
  figures/v1_legacy/fig_utils.py                    compute_enstrophy

Ce que fait reellement cet operateur :

    correct : (roll(fy,-1,AXIS_X) - fy) - (roll(fx,-1,AXIS_Y) - fx) = dfy/dx - dfx/dy
    legacy  : (roll(fy,-1,AXIS_Y) - fy) - (roll(fx,-1,AXIS_X) - fx) = dfy/dy - dfx/dx

Ce n'est PAS un rotationnel de signe oppose — auquel cas prendre la valeur
absolue ou le carre aurait tout rattrape. C'est son COMPLEMENTAIRE : une
combinaison de deformation, qui vaut exactement zero la ou le rotationnel
est maximal, et qui est maximale la ou le rotationnel s'annule.

  | champ                    | rotationnel | operateur legacy |
  |--------------------------|-------------|------------------|
  | rotation solide          |   +0.392699 |         0.000000 |
  | cisaillement pur         |   -0.196350 |         0.000000 |
  | compression pure         |    0.000000 |        -0.392699 |

Consequence mesuree sur `compute_enstrophy`, cisaillement pur periodique
`vx = sin y` dont l'enstrophie exacte vaut 2 pi^2 = 19.7392 :

  version corrigee : 19.7352  (0.02 % de la valeur exacte, l'erreur de la
                               difference centree)
  version ancienne :  0.0000  (0 % — l'« enstrophie » tracee etait nulle)

Attention au choix du champ de validation : sur Taylor-Green les deux
versions rendent la MEME integrale, par symetrie de leurs carres. Un test
ecrit sur Taylor-Green aurait donc passe sans rien verifier.
"""

import importlib.util
import os
import re
import sys

import numpy as np
import pytest



def _repo_root():
    """Racine du depot : on remonte jusqu'au dossier qui contient `src/`.

    Un calcul par `dirname` repete depend de la profondeur du fichier et
    casse au premier deplacement — souvent en silence, en pointant vers un
    chemin qui n'existe pas.
    """
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_REPO = _repo_root()
for _d in ("src", "study/h2b_prediction", "study/pipeline", "study/common"):
    _p = os.path.join(_REPO, *_d.split("/"))
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.grid import (  # noqa: E402
    AXIS_X,
    AXIS_Y,
    PeriodicGrid,
    forward_curl_z,
    legacy_forward_curl_z,
)
from Simulation.solver import MHDSolver  # noqa: E402

N = 64
L = 2.0 * np.pi


def _coords(n=N):
    c = np.arange(n) * L / n
    return np.meshgrid(c, c, indexing="ij")


def _load(rel, name):
    path = os.path.join(_REPO, *rel.split("/"))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ======================================================================
#  1. Ce que l'operateur legacy calcule vraiment
# ======================================================================

def test_the_legacy_operator_is_not_the_curl_with_the_opposite_sign():
    """Si ce n'etait qu'un signe, `abs` et le carre auraient tout rattrape."""
    X, Y = _coords()
    vx, vy = -(Y - np.pi), (X - np.pi)
    a, b = forward_curl_z(vx, vy), legacy_forward_curl_z(vx, vy)
    assert not np.allclose(a, -b), (
        "l'operateur legacy serait alors inoffensif sous valeur absolue ; "
        "il ne l'est pas")


def test_the_legacy_operator_is_blind_to_a_solid_rotation():
    X, Y = _coords()
    vx, vy = -(Y - np.pi), (X - np.pi)
    assert abs(np.median(legacy_forward_curl_z(vx, vy))) < 1e-12
    assert np.median(forward_curl_z(vx, vy)) > 0.1


def test_the_legacy_operator_is_blind_to_a_pure_shear():
    """Champ PERIODIQUE : une rampe (Y - pi) est une dent de scie sur le
    tore, et son raccord fabrique un pic qui masque ce qu'on mesure."""
    X, Y = _coords()
    vx, vy = np.sin(Y), np.zeros_like(X)
    assert np.max(np.abs(legacy_forward_curl_z(vx, vy))) < 1e-12
    assert np.max(np.abs(forward_curl_z(vx, vy))) > 0.05


def test_the_legacy_operator_sees_exactly_what_the_curl_does_not():
    """Compression pure : rotationnel nul, operateur legacy maximal."""
    X, Y = _coords()
    vx, vy = (X - np.pi), -(Y - np.pi)
    assert abs(np.median(forward_curl_z(vx, vy))) < 1e-12
    assert abs(np.median(legacy_forward_curl_z(vx, vy))) > 0.1


# ======================================================================
#  2. Les trois sites corriges
# ======================================================================

def test_jz_from_b_now_calls_the_shared_operator():
    """Reutiliser avant de reecrire : c'est la divergence des copies privees
    qui a produit D-1."""
    m = _load("study/h2b_prediction/h2b_v1_hamiltonian_loso.py", "loso_d17")
    rng = np.random.default_rng(0)
    Bx, By = rng.normal(size=(N, N)), rng.normal(size=(N, N))
    assert np.array_equal(m.jz_from_b(Bx, By), forward_curl_z(Bx, By))


def test_jz_from_b_sees_a_solid_rotation():
    m = _load("study/h2b_prediction/h2b_v1_hamiltonian_loso.py", "loso_d17b")
    X, Y = _coords()
    got = np.median(m.jz_from_b(-(Y - np.pi), (X - np.pi)))
    assert got > 0.1, f"jz_from_b rend {got} sur une rotation solide"


def test_the_h2b_ceiling_features_use_the_repo_axis_constants():
    """Le test lit la SOURCE : les huit differences doivent passer par
    AXIS_X / AXIS_Y, pour qu'un futur editeur ne puisse plus se tromper
    d'axe sans le voir."""
    path = os.path.join(_REPO, "study", "h2b_prediction",
                        "h2b_ceiling_random_split.py")
    src = open(path, encoding="utf-8").read()
    raw = src[src.index("    dxvy ="):src.index("    omega_z =")]
    # on ne juge que le CODE : un commentaire a le droit de citer axis=1
    block = "\n".join(l for l in raw.splitlines()
                      if not l.lstrip().startswith("#"))
    assert "axis=1" not in block and "axis=0" not in block, (
        "une difference du bloc de features est encore ecrite avec un axe "
        f"numerique nu :\n{block}")
    for name, axis in (("dxvy", "AXIS_X"), ("dyvx", "AXIS_Y"),
                       ("dxBy", "AXIS_X"), ("dyBx", "AXIS_Y"),
                       ("dxvx", "AXIS_X"), ("dyvy", "AXIS_Y"),
                       ("dxBx", "AXIS_X"), ("dyBy", "AXIS_Y")):
        assert re.search(rf"{name} = np\.roll\([^,]+, -1, axis={axis}\)", block), (
            f"{name} ne derive pas selon {axis}")


def test_the_h2b_ceiling_omega_matches_the_shared_curl():
    """Reconstruit omega_z depuis la source corrigee et le compare."""
    X, Y = _coords()
    vx, vy = -(Y - np.pi), (X - np.pi)
    dxvy = np.roll(vy, -1, axis=AXIS_X) - vy
    dyvx = np.roll(vx, -1, axis=AXIS_Y) - vx
    assert np.allclose(dxvy - dyvx, forward_curl_z(vx, vy), atol=1e-13)


# ======================================================================
#  3. L'enstrophie des figures
# ======================================================================

def _enstrophy_sim(vx, vy, n):
    g = PeriodicGrid(n, L)
    sim = MHDSolver(g, Re=400, Rm=400)
    sim.vx, sim.vy = vx, vy
    sim.Bx = np.zeros((n, n))
    sim.By = np.zeros((n, n))
    return sim


def test_the_enstrophy_matches_its_analytic_value_on_a_pure_shear():
    """vx = sin y, vy = 0  ->  omega = -cos y  ->  integrale = 2 pi^2.

    Champ choisi EXPRES : sur Taylor-Green les deux conventions rendent la
    meme integrale et le test n'aurait rien verifie.
    """
    n = 256
    c = np.arange(n) * L / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    fu = _load("figures/v1_legacy/fig_utils.py", "fu_d17")
    got = fu.compute_enstrophy(_enstrophy_sim(np.sin(Y), np.zeros_like(X), n))
    exact = 2.0 * np.pi ** 2
    assert got == pytest.approx(exact, rel=1e-3), (
        f"enstrophie {got:.6f} contre {exact:.6f} exact")


def test_the_old_enstrophy_convention_returned_exactly_zero_there():
    """Le defaut lui-meme : la grandeur tracee valait 0 % de sa valeur."""
    n = 256
    c = np.arange(n) * L / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    dx = L / n
    vx, vy = np.sin(Y), np.zeros_like(X)
    old = ((np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2 * dx)
           - (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * dx))
    assert np.sum(old ** 2) * dx ** 2 < 1e-20


def test_taylor_green_cannot_separate_the_two_conventions():
    """Fige le piege : un test ecrit sur ce champ aurait passe pour rien."""
    n = 128
    c = np.arange(n) * L / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    dx = L / n
    vx, vy = np.sin(X) * np.cos(Y), -np.cos(X) * np.sin(Y)
    new = ((np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2 * dx)
           - (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2 * dx))
    old = ((np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2 * dx)
           - (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * dx))
    assert np.sum(new ** 2) == pytest.approx(np.sum(old ** 2), rel=1e-9)


def test_the_enstrophy_is_positive_and_scales_as_the_square():
    n = 128
    c = np.arange(n) * L / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    fu = _load("figures/v1_legacy/fig_utils.py", "fu_d17b")
    a = fu.compute_enstrophy(_enstrophy_sim(np.sin(Y), np.zeros_like(X), n))
    b = fu.compute_enstrophy(_enstrophy_sim(3.0 * np.sin(Y),
                                            np.zeros_like(X), n))
    assert a > 0
    assert b / a == pytest.approx(9.0, rel=1e-9)


def test_a_uniform_flow_carries_no_enstrophy():
    n = 64
    fu = _load("figures/v1_legacy/fig_utils.py", "fu_d17c")
    ones = np.ones((n, n))
    assert fu.compute_enstrophy(_enstrophy_sim(ones, 2 * ones, n)) < 1e-20


# ======================================================================
#  4. Le balayage : plus aucune copie privee ne doit apparaitre
# ======================================================================

# Chaque entree a ete VERIFIEE a la main : la derivee selon x roule bien
# l'axe 0 et celle selon y l'axe 1. Elles gardent des axes numeriques nus
# pour rester bit-a-bit identiques a ce qui a produit les artefacts publies.
# Ajouter une ligne ici sans l'avoir verifiee reviendrait a desarmer le
# balayage.
_ALLOWED = {
    # grid.py EST l'implementation de reference
    os.path.join("src", "Simulation", "grid.py"),
    # get_fluxes utilise volontairement une difference CENTREE, pas avant
    os.path.join("src", "Simulation", "solver.py"),
    # les mappeurs v1/v2 appellent curl_z ; leurs jacobiens sont centres
    os.path.join("src", "Simulation", "HamiltParams.py"),
    os.path.join("src", "Simulation", "HamiltParams_v2.py"),
    os.path.join("src", "Simulation", "PhysToAngle.py"),
    # le pipeline utilise deja AXIS_X / AXIS_Y explicitement
    os.path.join("src", "pipeline.py"),
    # verifie : grad_By_x roule l'axe 0, grad_Bx_y l'axe 1 -> correct
    os.path.join("study", "pipeline", "hard_patch_labels.py"),
    os.path.join("study", "common", "qaoa_inputs.py"),
    # verifie : dvydx roule l'axe 0, dvxdy l'axe 1 -> correct depuis D-17
    os.path.join("figures", "v1_legacy", "fig_utils.py"),
}


def _sources():
    for root in ("src", "study", "figures"):
        base = os.path.join(_REPO, root)
        for dirpath, _, files in os.walk(base):
            for fn in files:
                if fn.endswith(".py"):
                    full = os.path.join(dirpath, fn)
                    yield full, os.path.relpath(full, _REPO)


def test_no_new_hand_rolled_curl_uses_a_bare_axis_number():
    """Un rotationnel ecrit a la main avec `axis=0`/`axis=1` nus est
    indistinguable d'un rotationnel juste tant qu'on ne le teste pas sur
    une rotation solide. On exige donc AXIS_X / AXIS_Y, qui rendent l'erreur
    visible a la lecture."""
    offenders = []
    for full, rel in _sources():
        if rel in _ALLOWED:
            continue
        lines = open(full, encoding="utf-8").read().splitlines()
        for i, line in enumerate(lines):
            if "np.roll(" not in line or "axis=" not in line:
                continue
            if "AXIS_X" in line or "AXIS_Y" in line:
                continue
            ctx = "\n".join(lines[max(0, i - 4):i + 4]).lower()
            if any(k in ctx for k in ("curl", "jz =", "j_z", "omega_z",
                                      "vortic", "enstroph")):
                offenders.append(f"{rel}:{i + 1}  {line.strip()[:80]}")
    assert not offenders, (
        "rotationnel ecrit a la main avec un axe numerique nu :\n  "
        + "\n  ".join(offenders))


def test_the_repo_axis_constants_are_the_ones_these_tests_assume():
    assert (AXIS_X, AXIS_Y) == (0, 1)


# ======================================================================
#  5. D-2 / D-3 — le gel de phase 1b, et les copies corrigees
# ======================================================================
#
# CORRECTION A UNE AFFIRMATION PRECEDENTE : le defaut d'axe de
# `dns_validation.fluctuating_KE` n'est PAS une trouvaille de cet audit. Il
# etait deja connu, consigne comme « deviation D2 », et la decision prise
# alors etait explicite — `dns_extension.py:85` : « phase 1b reste
# intouchee, reparation cote v3 par copie ». Un test de tests/v3 epinglait
# meme la contamination. La corriger dans `dns_validation.py` aurait casse
# la reproductibilite des artefacts de phase 1b.
#
# Ce que l'audit apporte reellement ici :
#   - la MESURE de l'ampleur de D2, qui n'etait chiffree nulle part ;
#   - une deuxieme deviation dans le meme fichier, `mean_sq_current`, qui
#     porte la meme inversion d'axes et qu'aucun test n'epinglait (D3) ;
#   - une copie corrigee `mean_sq_current_fixed`, sur le modele de
#     `fluctuating_ke_fixed` qui existait deja pour D2.
#
# Les tests ci-dessous verrouillent les DEUX cotes : la version gelee doit
# rester fausse a l'identique, la copie corrigee doit etre juste.

def _dv():
    return _load("study/pipeline/dns_validation.py", "dns_val_d18")


def _dx():
    return _load("study/pipeline/dns_extension.py", "dns_ext_d18")


def _kh(noise, n=128):
    sim = MHDSolver(PeriodicGrid(n, L), Re=400, Rm=400)
    sim.init_kelvin_helmholtz(noise_amplitude=noise)
    return sim


# ── D2 : la version gelee doit rester gelee ───────────────────────────

def test_the_frozen_observable_still_reports_the_base_flow():
    """Le gel est une decision, pas un oubli : si cette valeur changeait,
    les artefacts de phase 1b cesseraient d'etre reproductibles."""
    sim = _kh(0.0)
    got = _dv().fluctuating_KE(sim.vx, sim.vy)
    total = 0.5 * np.mean(sim.vx ** 2 + sim.vy ** 2)
    assert got > 0.1, "la version de phase 1b a ete corrigee : le gel est rompu"
    assert got / total > 0.5


def test_the_measured_size_of_deviation_d2():
    """Chiffre l'ampleur, qui n'etait consignee nulle part : la grandeur ne
    bouge que de 0.02 % quand on allume la perturbation."""
    dv = _dv()
    base = dv.fluctuating_KE(*(lambda s: (s.vx, s.vy))(_kh(0.0)))
    pert = dv.fluctuating_KE(*(lambda s: (s.vx, s.vy))(_kh(0.1)))
    assert pert / base == pytest.approx(1.0002, abs=5e-4)


# ── D2 : la copie corrigee doit etre juste ────────────────────────────

def test_the_fixed_observable_removes_the_base_flow_exactly():
    sim = _kh(0.0)
    assert _dx().fluctuating_ke_fixed(sim.vx, sim.vy) < 1e-20


def test_the_fixed_observable_responds_to_the_perturbation():
    dx = _dx()
    base = dx.fluctuating_ke_fixed(*(lambda s: (s.vx, s.vy))(_kh(0.0)))
    pert = dx.fluctuating_ke_fixed(*(lambda s: (s.vx, s.vy))(_kh(0.1)))
    assert pert > 1e6 * max(base, 1e-30)


def test_the_fixed_observable_is_quadratic_in_the_amplitude():
    dx = _dx()
    a = dx.fluctuating_ke_fixed(*(lambda s: (s.vx, s.vy))(_kh(0.05)))
    b = dx.fluctuating_ke_fixed(*(lambda s: (s.vx, s.vy))(_kh(0.10)))
    assert b / a == pytest.approx(4.0, rel=0.05)


# ── D3 : meme structure, deviation trouvee par cet audit ──────────────

def test_the_frozen_current_still_carries_the_axis_inversion():
    """Sur un cisaillement magnetique pur, la version gelee doit rendre
    zero — c'est la signature de l'inversion d'axes."""
    n = 128
    c = np.arange(n) * L / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    Bx, By = -np.sin(Y), np.zeros_like(X)      # Jz = -dBx/dy = cos y
    assert _dv().mean_sq_current(Bx, By) < 1e-25, (
        "la version de phase 1b a ete corrigee : le gel est rompu")


def test_the_fixed_current_sees_the_current_sheet():
    n = 128
    c = np.arange(n) * L / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    Bx, By = -np.sin(Y), np.zeros_like(X)
    got = _dx().mean_sq_current_fixed(Bx, By)
    assert got == pytest.approx(np.mean(forward_curl_z(Bx, By) ** 2), rel=0.05)
    assert got > 0.0


def test_the_fixed_current_is_zero_on_a_potential_field():
    """B = grad(phi) porte un courant nul : le controle negatif."""
    n = 128
    c = np.arange(n) * L / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    assert _dx().mean_sq_current_fixed(-np.sin(X), -np.sin(Y)) < 1e-25


def test_the_two_current_versions_really_differ():
    """Sinon la copie corrigee serait decorative."""
    n = 128
    c = np.arange(n) * L / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    Bx, By = -np.sin(Y), np.zeros_like(X)
    assert _dx().mean_sq_current_fixed(Bx, By) > 1e6 * max(
        _dv().mean_sq_current(Bx, By), 1e-30)


def test_the_fixed_current_scales_as_the_square_of_the_field():
    n = 128
    c = np.arange(n) * L / n
    X, Y = np.meshgrid(c, c, indexing="ij")
    dx = _dx()
    a = dx.mean_sq_current_fixed(-np.sin(Y), np.zeros_like(X))
    b = dx.mean_sq_current_fixed(-3.0 * np.sin(Y), np.zeros_like(X))
    assert b / a == pytest.approx(9.0, rel=1e-9)


def test_both_frozen_deviations_are_written_down_where_they_live():
    """Une deviation connue mais non ecrite se refait corriger par erreur —
    c'est exactement ce qui a failli arriver ici."""
    src = open(os.path.join(_REPO, "study", "pipeline",
                            "dns_validation.py"), encoding="utf-8").read().lower()
    for marker in ("deviation d2", "deviation d3", "phase 1b reste intouchee"):
        assert marker in src, f"{marker} n'est pas consigne dans le fichier gele"


# ======================================================================
#  D-153 — le rotationnel se reconnaît a sa FORME, pas a son nom
# ======================================================================
#
#  L'ancien balayage cherchait `np.roll(` + `axis=` sur UNE ligne, puis
#  decidait par une fenetre de +/-4 lignes contenant l'un des mots
#  `curl`, `jz =`, `j_z`, `omega_z`, `vortic`, `enstroph`. Trois trous,
#  tous mesures :
#
#    1. **le nom** — un rotationnel appele `rot_z`, `w_z` ou `dvydx`
#       n'emploie aucun de ces mots : invisible. C'est le defaut de D-148
#       (un detecteur de forme pilote par une liste tenue a la main) sur la
#       famille la plus dangereuse de ce depot ;
#    2. **la syntaxe** — `np.roll(By, -1, 0)` passe l'axe en POSITIONNEL et
#       ne contient donc pas `axis=`. Mesure : 2 sites du depot l'ecrivaient
#       ainsi, tous deux dans `tests/` ;
#    3. **le perimetre** — `src`, `study`, `figures` seulement. Un test qui
#       calcule sa propre reference avec le rotationnel inverse mesure autre
#       chose que ce qu'il annonce, et rien ne le voyait. C'est la regle
#       « mesurer avec l'operateur assorti », retournee contre la suite.
#
#  Le detecteur ci-dessous ne connaît aucun nom : il cherche la SIGNATURE
#  d'un rotationnel discret — une soustraction dont les deux cotes roulent
#  DEUX tableaux differents sur DEUX axes differents. Un laplacien (meme
#  tableau) et un gradient (un seul roll) n'y repondent pas.

import ast as _ast

_RACINES_BALAYEES = ("src", "study", "figures", "tests", "scripts")

#: Fichiers autorises a ecrire un axe nu, avec leur raison. Une exemption
#: perimee est un mensonge qui dort : le test plus bas exige que chacune
#: designe un fichier qui existe et qui porte encore un site.
_AXE_NU_AUTORISE = {
    os.path.join("tests", "study", "test_no_private_curl_survives.py"):
        "ce fichier REPRODUIT les deux conventions pour les mesurer : "
        "l'ancienne y est ecrite exprès (sections 3 et 4)",
}

#: Plancher mesure a `38389e5`. Un balayage qui retrecit ne prouve rien.
_PLANCHER_FICHIERS_PY = 200


def _axe_de(appel):
    """L'expression d'axe d'un `np.roll`, mot-cle OU positionnel."""
    for kw in appel.keywords:
        if kw.arg == "axis":
            return _ast.unparse(kw.value)
    if len(appel.args) >= 3:
        return _ast.unparse(appel.args[2])
    return None


def _rolls_de(noeud):
    out = []
    for n in _ast.walk(noeud):
        if isinstance(n, _ast.Call) and getattr(n.func, "attr", None) == "roll":
            if not n.args:
                continue
            axe = _axe_de(n)
            if axe is not None:
                out.append((_ast.unparse(n.args[0]), axe))
    return out


def _sites_de_forme_rotationnelle(arbre):
    """(ligne, description, axes) des expressions en forme de rotationnel.

    Signature : `A - B` ou A roule un tableau sur un axe et B un AUTRE
    tableau sur un AUTRE axe. C'est ce que fait tout rotationnel discret, et
    ce que ne fait ni un laplacien ni un gradient.
    """
    sites = {}
    for n in _ast.walk(arbre):
        if not (isinstance(n, _ast.BinOp) and isinstance(n.op, _ast.Sub)):
            continue
        for arr_g, axe_g in _rolls_de(n.left):
            for arr_d, axe_d in _rolls_de(n.right):
                if arr_g != arr_d and axe_g != axe_d:
                    sites[n.lineno] = (f"{arr_g}@{axe_g} - {arr_d}@{axe_d}",
                                       (axe_g, axe_d))
    return [(l, d, a) for l, (d, a) in sorted(sites.items())]


def _fichiers_python_balayes():
    out = []
    for racine in _RACINES_BALAYEES:
        base = os.path.join(_REPO, racine)
        for dirpath, _dirs, files in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for fn in files:
                if fn.endswith(".py"):
                    out.append(os.path.join(dirpath, fn))
    return sorted(out)


def _axe_est_nomme(expr):
    return "AXIS_X" in expr or "AXIS_Y" in expr


def test_no_hand_rolled_curl_uses_a_bare_axis_anywhere():
    """Tout rotationnel ecrit a la main nomme ses axes — partout, `tests/`
    compris.

    Sur quelle entree ce test echoue : sur `w = np.roll(vy,-1,1) -
    np.roll(vx,-1,0)`, un rotationnel INVERSE dont aucun nom ne dit qu'il en
    est un. L'ancien garde restait vert dessus (mesure D-153).
    """
    fichiers = _fichiers_python_balayes()
    assert len(fichiers) >= _PLANCHER_FICHIERS_PY, (
        f"{len(fichiers)} fichiers balayes, {_PLANCHER_FICHIERS_PY} exiges "
        "(mesure a `38389e5`) : le balayage a retreci")

    offenders = []
    for chemin in fichiers:
        rel = os.path.relpath(chemin, _REPO)
        if rel in _AXE_NU_AUTORISE:
            continue
        try:
            arbre = _ast.parse(open(chemin, encoding="utf-8").read())
        except SyntaxError:                               # pragma: no cover
            continue
        for lineno, description, axes in _sites_de_forme_rotationnelle(arbre):
            if all(_axe_est_nomme(a) for a in axes):
                continue
            offenders.append(f"{rel}:{lineno}  {description}")
    assert not offenders, (
        "rotationnel ecrit a la main avec un axe numerique nu — `grid.py` "
        "fait foi (AXIS_X = 0, AXIS_Y = 1) :\n  " + "\n  ".join(offenders))


def test_the_shape_detector_knows_a_curl_from_a_laplacian():
    """Le detecteur doit trouver ce qu'il annonce, et RIEN d'autre.

    Les quatre cas qui l'ont ecrit — dont les deux qui doivent rester
    muets, sans quoi le balayage crierait sur tout le solveur.
    """
    def _sites(code):
        return _sites_de_forme_rotationnelle(_ast.parse(code))

    #  un rotationnel, axes nus : vu
    assert len(_sites("w = np.roll(vy,-1,1) - np.roll(vx,-1,0)")) == 1
    #  le meme, axe passe par mot-cle : vu aussi (le premier cas ci-dessus
    #  est la forme POSITIONNELLE, celle qui ne contient pas `axis=`)
    assert len(_sites("w = np.roll(vy,-1,axis=1) - np.roll(vx,-1,axis=0)")) == 1
    #  un laplacien : MEME tableau des deux cotes -> muet
    assert _sites("l = np.roll(f,-1,0) - np.roll(f,1,0)") == []
    #  deux tableaux mais le MEME axe (une divergence partielle) -> muet
    assert _sites("d = np.roll(vx,-1,0) - np.roll(vy,-1,0)") == []
    #  un seul roll -> muet
    assert _sites("g = np.roll(f,-1,0) - f") == []
    #  la forme nommee, celle qu'on exige : vue comme site, axes nommes
    sites = _sites("w = np.roll(vy,-1,AXIS_X) - np.roll(vx,-1,AXIS_Y)")
    assert len(sites) == 1 and all(_axe_est_nomme(a) for a in sites[0][2])


def test_every_bare_axis_exemption_still_names_a_real_site():
    """Une exemption perimee doit crier."""
    for rel, raison in _AXE_NU_AUTORISE.items():
        assert raison.strip(), f"{rel} exempte sans raison"
        chemin = os.path.join(_REPO, rel)
        assert os.path.exists(chemin), (
            f"{rel} n'existe plus : retirer son exemption")
        arbre = _ast.parse(open(chemin, encoding="utf-8").read())
        nus = [s for s in _sites_de_forme_rotationnelle(arbre)
               if not all(_axe_est_nomme(a) for a in s[2])]
        assert nus, (
            f"{rel} ne porte plus aucun axe nu : son exemption ne couvre "
            "plus rien, la retirer")


def test_the_sweep_sees_the_shapes_the_repository_actually_contains():
    """Un balayage qui ne trouve AUCUN rotationnel ne prouve rien.

    Mesure a `38389e5` : **12 sites** de forme rotationnelle dans les cinq
    racines, dont 3 dans ce fichier (les deux conventions reproduites).
    """
    total = 0
    for chemin in _fichiers_python_balayes():
        try:
            arbre = _ast.parse(open(chemin, encoding="utf-8").read())
        except SyntaxError:                               # pragma: no cover
            continue
        total += len(_sites_de_forme_rotationnelle(arbre))
    assert total >= 10, (
        f"{total} sites de forme rotationnelle trouves, 12 mesures a "
        "`38389e5` : le detecteur ne reconnaît plus la forme")
