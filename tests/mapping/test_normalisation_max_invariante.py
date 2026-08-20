"""La normalisation `max` rend-elle l'EQUILIBRE des termes independant de `dim` ?

Pourquoi ce fichier existe
--------------------------
Les trois termes du v2 `legacy` utilisent trois normalisateurs differents :
la MOYENNE des sauts pour ZZ, la somme de DEUX maxima pour ZZZZ, la MEDIANE
des couplages pour le biais Z. Or moyenner par blocs a la resolution `dim`
est un filtre passe-bas : monter `dim` resout des echelles plus fines et
dissymetrise la distribution des sauts. Le pic suit la queue, la mediane
non — donc le rapport biais/couplage DERIVE avec `dim`, par construction.

Consequence pratique : un reglage d'hyperparametres obtenu a une taille ne
transfere pas a une autre taille, et un balayage en `dim` mesure alors deux
choses a la fois.

`norm="max"` accroche les trois termes au meme genre de statistique :

    max|C| == w_zz            exactement
    max|K| == w_zzzz          exactement
    max|h| / max(|C|,|K|) == c_bias * max|s - thr|

Ce que ces tests prouvent : ces trois egalites tiennent a `dim` variable.
Ce qu'ils ne pretendent PAS : que les coefficients soient identiques d'un
`dim` a l'autre. Ils ne peuvent pas l'etre — le champ d'entree lui-meme
change. Seul l'EQUILIBRE entre les termes devient invariant.
"""
import os
import sys

import numpy as np
import pytest

_RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_RACINE, "src"), os.path.join(_RACINE, "study", "pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Simulation.HamiltParams_v2 import PhysicalMapperV2   # noqa: E402


# ------------------------------------------------------------------
#  champ d'essai : intermittent, donc separant
# ------------------------------------------------------------------
def _champ(n, graine=0):
    """Champ periodique a plusieurs echelles PLUS une structure fine.

    Un champ mono-echelle ne separerait rien : c'est justement la presence
    de structure sous la coupure du filtre qui fait deriver le rapport
    pic/mediane quand `dim` monte. La nappe etroite joue le role du feuillet
    de courant.
    """
    x = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    rng = np.random.default_rng(graine)
    nappe = np.exp(-((Y - np.pi) ** 2) / (2 * (2 * np.pi / n * 3) ** 2))
    return {
        "vx": np.sin(Y) + 0.3 * np.sin(5 * X) + 0.05 * rng.standard_normal((n, n)),
        "vy": np.sin(X) + 0.3 * np.cos(7 * Y),
        "Bx": np.cos(Y) + 2.0 * nappe,
        "By": np.sin(2 * X) + 0.2 * np.cos(11 * X),
        "Jz": np.zeros((n, n)),
    }


def _coeffs(n, norm, c_bias=0.1, thr=0.15, echelle_champ=1.0, dx=None, graine=0):
    champs = {k: v * echelle_champ for k, v in _champ(n, graine).items()}
    score = np.linspace(0.0, 1.0, n * n).reshape(n, n)
    m = PhysicalMapperV2(dx=(dx if dx is not None else 2 * np.pi / n),
                         c_bias=c_bias, norm=norm)
    return m.compute_coefficients(None, score, champs, thr)


def _max_couplage(hp):
    Ch, Cv = hp["C_edges"]
    return max(float(np.max(np.abs(Ch))), float(np.max(np.abs(Cv)))), \
        float(np.max(np.abs(hp["K_plaquettes"])))


DIMS = [4, 8, 16, 32]


# ------------------------------------------------------------------
#  1. ce que `max` garantit
# ------------------------------------------------------------------
@pytest.mark.parametrize("n", DIMS)
def test_max_C_vaut_exactement_w_zz(n):
    maxC, _ = _max_couplage(_coeffs(n, "max"))
    assert maxC == pytest.approx(PhysicalMapperV2.W_ZZ, rel=1e-12)


@pytest.mark.parametrize("n", DIMS)
def test_max_K_vaut_exactement_w_zzzz(n):
    _, maxK = _max_couplage(_coeffs(n, "max"))
    assert maxK == pytest.approx(PhysicalMapperV2.W_ZZZZ, rel=1e-12)


@pytest.mark.parametrize("n", DIMS)
def test_le_rapport_biais_couplage_vaut_c_bias(n):
    """C'est l'invariant qui rend les hyperparametres transferables."""
    c_bias, thr = 0.1, 0.15
    hp = _coeffs(n, "max", c_bias=c_bias, thr=thr)
    maxC, maxK = _max_couplage(hp)
    Hh, _ = hp["H_edges"]
    score = np.linspace(0.0, 1.0, n * n).reshape(n, n)
    attendu = c_bias * max(maxC, maxK) * np.max(np.abs(score - thr))
    assert float(np.max(np.abs(Hh))) == pytest.approx(attendu, rel=1e-12)


def test_l_equilibre_ne_derive_pas_avec_dim():
    """max|C|, max|K| et max|h| : identiques aux quatre tailles."""
    mesures = []
    for n in DIMS:
        hp = _coeffs(n, "max")
        maxC, maxK = _max_couplage(hp)
        mesures.append((maxC, maxK, float(np.max(np.abs(hp["H_edges"][0])))))
    for i, nom in enumerate(("max|C|", "max|K|", "max|h|")):
        valeurs = [m[i] for m in mesures]
        assert max(valeurs) == pytest.approx(min(valeurs), rel=1e-12), (
            f"{nom} varie avec dim : {dict(zip(DIMS, valeurs))}")


def test_legacy_lui_fait_deriver_l_equilibre():
    """Le champ qui SEPARE : sans ce test, l'invariance de `max` ne prouve
    rien — elle pourrait tenir pour les deux normalisations."""
    rapports = []
    for n in DIMS:
        hp = _coeffs(n, "legacy")
        maxC, maxK = _max_couplage(hp)
        rapports.append(float(np.max(np.abs(hp["H_edges"][0]))) / max(maxC, maxK))
    etalement = max(rapports) / min(rapports)
    assert etalement > 1.5, (
        f"`legacy` ne derive pas sur ce champ (etalement {etalement:.2f}x) : "
        "le champ d'essai ne separe pas les deux normalisations, le rendre "
        "plus intermittent")


# ------------------------------------------------------------------
#  2. les invariances deja revendiquees, sous `max`
# ------------------------------------------------------------------
@pytest.mark.parametrize("norm", ["legacy", "max"])
def test_invariance_en_dx(norm):
    """`dx` ne doit pas entrer : les differences ne sont pas divisees."""
    a = _coeffs(8, norm, dx=1.0)
    b = _coeffs(8, norm, dx=1e-3)
    np.testing.assert_allclose(a["C_edges"][0], b["C_edges"][0], rtol=0, atol=0)
    np.testing.assert_allclose(a["K_plaquettes"], b["K_plaquettes"], rtol=0, atol=0)


#: Ecart relatif MESURE (pas suppose) sur l'invariance d'amplitude, a
#: `graine=0`, n=8 : `legacy` porte un garde ADDITIF `+ EPS` qui decale
#: legerement l'echelle, `max` porte un garde multiplicatif et l'invariance
#: y est exacte a la precision machine.
_TOLERANCE_AMPLITUDE = {"legacy": 1e-9, "max": 1e-14}


@pytest.mark.parametrize("norm", ["legacy", "max"])
def test_invariance_en_amplitude_des_champs(norm):
    """Multiplier v et B par 10 laisse les coefficients inchanges.

    Les deux tolerances different parce que les deux gardes different, et
    l'ecart est mesure : 9,8e-11 (`legacy`) contre 4,8e-16 (`max`). Une
    tolerance unique et lache cacherait que `max` est exact.
    """
    tol = _TOLERANCE_AMPLITUDE[norm]
    a = _coeffs(8, norm, echelle_champ=1.0)
    b = _coeffs(8, norm, echelle_champ=10.0)
    np.testing.assert_allclose(a["C_edges"][0], b["C_edges"][0], rtol=tol)
    np.testing.assert_allclose(a["K_plaquettes"], b["K_plaquettes"], rtol=tol)


def test_max_est_strictement_plus_invariant_que_legacy():
    """Le garde additif de `legacy` est bien la cause, pas une coincidence.

    Sur quelle entree ce test echoue : si `legacy` devenait exact (garde
    rendu multiplicatif la aussi), ou si `max` cessait de l'etre.
    """
    def ecart(norm):
        a = _coeffs(8, norm, echelle_champ=1.0)["C_edges"][0]
        b = _coeffs(8, norm, echelle_champ=10.0)["C_edges"][0]
        return float(np.max(np.abs(a - b) / (np.abs(a) + 1e-30)))
    e_legacy, e_max = ecart("legacy"), ecart("max")
    assert e_max < 1e-14 < e_legacy, (
        f"legacy={e_legacy:.2e}, max={e_max:.2e} : l'ecart d'exactitude "
        "entre les deux gardes a disparu")


# ------------------------------------------------------------------
#  3. le chemin historique n'a pas bouge
# ------------------------------------------------------------------
def test_legacy_est_le_defaut():
    assert PhysicalMapperV2().norm == "legacy", (
        "changer le defaut est un changement de comportement scientifique : "
        "il se decide, il ne se subit pas")


def test_legacy_reproduit_la_formule_historique():
    """Recalcul independant des trois normalisateurs d'origine."""
    n = 8
    champs = _champ(n)
    score = np.linspace(0.0, 1.0, n * n).reshape(n, n)
    hp = _coeffs(n, "legacy")

    def saut(axis):
        d = [champs[k] - np.roll(champs[k], -1, axis=axis)
             for k in ("vx", "vy", "Bx", "By")]
        return np.sqrt(sum(x ** 2 for x in d))

    jh, jv = saut(1), saut(0)
    attendu = -PhysicalMapperV2.W_ZZ * jh / (0.5 * (jh.mean() + jv.mean())
                                             + PhysicalMapperV2.EPS)
    np.testing.assert_allclose(hp["C_edges"][0], attendu, rtol=1e-12)


def test_les_deux_normalisations_different_vraiment():
    """Une option qui ne change rien est une option qui ment."""
    a = _coeffs(8, "legacy")
    b = _coeffs(8, "max")
    assert not np.allclose(a["C_edges"][0], b["C_edges"][0]), \
        "`max` rend le meme C que `legacy` : l'option ne fait rien"


def test_une_normalisation_inconnue_est_refusee():
    with pytest.raises(ValueError, match="inconnue"):
        PhysicalMapperV2(norm="mediane")


# ------------------------------------------------------------------
#  4. ce que `max` ne pretend PAS
# ------------------------------------------------------------------
def test_le_motif_spatial_depend_encore_de_dim():
    """Garde-fou contre une sur-lecture de ces tests.

    L'equilibre des termes devient invariant ; le MOTIF ne peut pas l'etre,
    puisque le champ d'entree lui-meme change avec la coupure du filtre. Si
    ce test venait a passer, c'est que le champ d'essai est devenu
    auto-similaire et que les tests d'invariance ci-dessus ne prouvent plus
    ce qu'ils annoncent.
    """
    a = _coeffs(8, "max")["C_edges"][0]
    b = _coeffs(16, "max")["C_edges"][0]
    # meme grille pour comparer : on sous-echantillonne la plus fine
    b_reduit = b.reshape(8, 2, 8, 2).mean(axis=(1, 3))
    assert not np.allclose(a, b_reduit, atol=1e-3), (
        "le motif de C est devenu independant de dim : le champ d'essai "
        "n'a plus de structure sous la coupure, ces tests ne separent plus")
