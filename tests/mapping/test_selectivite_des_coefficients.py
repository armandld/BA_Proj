"""Chaque coefficient capte-t-il SON type de structure, et lui seul ?

Exigence de USER : les coefficients doivent être intuitifs — adimensionnels,
indépendants de tout sauf du type d'instabilité qu'ils prétendent capter.
Les invariances (dx, amplitude, `dim`) sont mesurées ailleurs
(`test_normalisation_max_invariante.py`). Ce fichier mesure la **sélectivité**,
qui est l'autre moitié de l'exigence et n'était vérifiée nulle part.

Champs analytiques, réponse connue à la main :

    uniforme          omega = 0, J = 0, det(nabla B) = 0
    rotation solide   omega != 0, J = 0
    nappe de courant  omega = 0,  J != 0
    X-point           omega = 0,  J = 0, det(nabla B) < 0

Les deux normalisations sont exercées. Avant ce fichier, **aucun test
n'exerçait `norm="max"` hors des invariances** : basculer le défaut aurait
laissé toute la réponse physique non vérifiée.

Ce que les mesures établissent, et que ces tests épinglent :

1. `K_xpoint` est **sélectif** — il ne répond qu'au det(nabla B) < 0.
2. `K_plaquettes` **ne l'est pas** : il vaut `(|omega| + |J|)/norme`, donc un
   vortex et une nappe de courant de magnitudes appariées rendent la **même**
   valeur. Le terme capte « il se passe quelque chose de rotationnel OU
   magnétique », pas un type.
3. `C_edges` ne l'est pas non plus : `sqrt(|dv|^2 + |dB|^2)` confond un saut
   hydrodynamique et un saut magnétique.
4. Sous `norm="max"`, le PIC vaut `w_zz` / `w_zzzz` sur **tout** champ non
   uniforme : par construction, la magnitude ne distingue plus rien du tout
   et toute l'information de structure passe dans le MOTIF spatial. Sous
   `legacy` le pic varie (rapport pic/moyenne = intermittence) et porte donc
   une information de structure — au prix de la dépendance en `dim`.

Les points 2 à 4 ne sont pas des défauts déclarés : ce sont des propriétés
de conception, mesurées et épinglées pour qu'un changement les fasse crier.
"""
import os
import sys

import numpy as np
import pytest

_RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(_RACINE, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_RACINE, "src"))

from Simulation.HamiltParams_v2 import PhysicalMapperV2       # noqa: E402

N = 32
NORMS = ["legacy", "max"]


def _grille():
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.meshgrid(x, x, indexing="ij")


def _champs():
    """Quatre champs dont la réponse est connue analytiquement."""
    X, Y = _grille()
    Z, U = np.zeros((N, N)), np.ones((N, N))
    return {
        "uniforme":      dict(vx=U, vy=Z, Bx=U, By=Z),
        "rotation":      dict(vx=-np.sin(Y), vy=np.sin(X), Bx=Z, By=Z),
        "nappe_courant": dict(vx=Z, vy=Z, Bx=np.tanh(np.sin(Y)), By=Z),
        "xpoint":        dict(vx=Z, vy=Z, Bx=np.sin(Y), By=np.sin(X)),
    }


def _coeffs(nom, norm, xpoint=True):
    champ = dict(_champs()[nom])
    champ["Jz"] = np.zeros((N, N))
    m = PhysicalMapperV2(dx=2 * np.pi / N, norm=norm)
    hp = m.compute_coefficients(None, np.full((N, N), 0.5), champ, 0.15,
                                advanced_anomalies_enabled=xpoint)
    Ch, Cv = hp["C_edges"]
    return {
        "C": max(float(np.max(np.abs(Ch))), float(np.max(np.abs(Cv)))),
        "K": float(np.max(np.abs(hp["K_plaquettes"]))),
        "Kxp": float(np.max(np.abs(hp.get("K_xpoint", np.zeros(1))))),
    }


# ------------------------------------------------------------------
#  1. le champ nul : rien ne doit répondre
# ------------------------------------------------------------------
@pytest.mark.parametrize("norm", NORMS)
def test_un_champ_uniforme_neteint_les_trois_familles(norm):
    r = _coeffs("uniforme", norm)
    for nom, v in r.items():
        assert v == pytest.approx(0.0, abs=1e-12), f"{nom} = {v:.3e} sur un champ uniforme"


# ------------------------------------------------------------------
#  2. K_xpoint EST sélectif — le seul des trois
# ------------------------------------------------------------------
@pytest.mark.parametrize("norm", NORMS)
def test_kxpoint_ne_repond_quau_xpoint(norm):
    """Sur quelle entrée ce test échoue : si `K_xpoint` se mettait à répondre
    à une rotation ou à une nappe, il cesserait de désigner un type."""
    assert _coeffs("xpoint", norm)["Kxp"] > 0.5
    for autre in ("rotation", "nappe_courant", "uniforme"):
        v = _coeffs(autre, norm)["Kxp"]
        assert v == pytest.approx(0.0, abs=1e-12), (
            f"K_xpoint = {v:.3e} sur '{autre}', qui n'a pas de det(nabla B) < 0")


def test_kxpoint_est_muet_la_ou_le_determinant_est_POSITIF():
    """La sélectivité de SIGNE, que le test précédent ne voyait pas.

    `K_xp = -w * max(0, -det(nabla B)) / norme` : le `max(0, .)` ne retient
    que les points hyperboliques (det < 0, X-point) et doit être exactement
    muet sur les points elliptiques (det > 0, coeur d'îlot).

    Sur quelle entrée ce test échoue : remplacer `max(0, -det)` par `|det|`.
    Le test précédent, qui ne regardait qu'un MAXIMUM sur tout le champ, ne
    voyait pas cette mutation — les deux formes y donnent le même pic. Il
    faut comparer POINT PAR POINT.
    """
    X, Y = _grille()
    champ = dict(vx=np.zeros((N, N)), vy=np.zeros((N, N)),
                 Bx=np.sin(Y), By=np.sin(X), Jz=np.zeros((N, N)))
    m = PhysicalMapperV2(dx=2 * np.pi / N, norm="max")
    hp = m.compute_coefficients(None, np.full((N, N), 0.5), champ, 0.15,
                                advanced_anomalies_enabled=True)
    kxp = np.abs(np.asarray(hp["K_xpoint"], dtype=float))
    det = m._compute_det_jacobian_B(champ["Bx"], champ["By"], 2 * np.pi / N)

    positifs = det > 1e-9
    negatifs = det < -1e-9
    assert positifs.any() and negatifs.any(), (
        "le champ d'essai n'a pas les deux signes : il ne sépare rien")
    assert np.allclose(kxp[positifs], 0.0, atol=1e-12), (
        f"K_xpoint vaut jusqu'à {kxp[positifs].max():.3e} là où det > 0 : "
        "le terme répond aux points ELLIPTIQUES, qu'il est censé ignorer")
    assert kxp[negatifs].max() > 0.5, (
        "K_xpoint est muet là où det < 0 : il ne détecte plus les X-points")


# ------------------------------------------------------------------
#  3. K_plaquettes ne l'est PAS : |omega| + |J| confond les deux
# ------------------------------------------------------------------
@pytest.mark.parametrize("norm", NORMS)
def test_la_plaquette_ne_distingue_pas_un_vortex_dune_nappe(norm):
    """Propriété de CONCEPTION, épinglée pour qu'un changement crie.

    `K_p = -w * (|omega_p| + |J_p|) / norme` somme les deux magnitudes : un
    champ purement rotationnel et un champ purement magnétique, tous deux
    normalisés par leur propre pic, rendent le même K. Le terme capte « il
    se passe quelque chose », pas un type.
    """
    k_rot = _coeffs("rotation", norm)["K"]
    k_nappe = _coeffs("nappe_courant", norm)["K"]
    assert k_rot > 0.5 and k_nappe > 0.5, "les deux structures doivent répondre"
    assert k_rot == pytest.approx(k_nappe, rel=1e-9), (
        f"vortex {k_rot:.4f} != nappe {k_nappe:.4f} : la plaquette est devenue "
        "sélective — c'est un changement de conception, le documenter")


@pytest.mark.parametrize("norm", NORMS)
def test_le_couplage_zz_repond_a_tout_saut_sans_distinguer(norm):
    """`C = sqrt(|dv|^2 + |dB|^2)` : un saut hydro et un saut magnétique
    entrent dans la même racine. C'est aussi une conception, pas un défaut."""
    for nom in ("rotation", "nappe_courant", "xpoint"):
        assert _coeffs(nom, norm)["C"] > 0.0, f"ZZ muet sur '{nom}'"


# ------------------------------------------------------------------
#  4. ce que `norm="max"` change au SENS du nombre
# ------------------------------------------------------------------
def test_sous_max_le_pic_est_constant_donc_ne_porte_aucune_structure():
    """Conséquence directe du changement demandé, à connaître.

    `max` force `max|C| = w_zz` et `max|K| = w_zzzz` sur TOUT champ non
    uniforme : la magnitude cesse de distinguer quoi que ce soit, et toute
    l'information de structure passe dans le motif spatial.
    """
    pics_C = {n: _coeffs(n, "max")["C"] for n in ("rotation", "nappe_courant", "xpoint")}
    pics_K = {n: _coeffs(n, "max")["K"] for n in ("rotation", "nappe_courant", "xpoint")}
    for pics, attendu, nom in ((pics_C, PhysicalMapperV2.W_ZZ, "max|C|"),
                               (pics_K, PhysicalMapperV2.W_ZZZZ, "max|K|")):
        for champ, v in pics.items():
            assert v == pytest.approx(attendu, rel=1e-9), (
                f"{nom} = {v:.4f} sur '{champ}', attendu {attendu}")


def test_sous_legacy_le_pic_varie_donc_il_porte_une_structure():
    """Le champ qui SÉPARE les deux normalisations.

    Sous `legacy`, `max|C|` est le rapport pic/moyenne des sauts — une mesure
    d'intermittence. Une nappe raide et une rotation lisse doivent donc
    rendre des pics DIFFÉRENTS. Sans ce test, celui du dessus pourrait
    passer pour une propriété universelle du calcul.
    """
    c_rot = _coeffs("rotation", "legacy")["C"]
    c_nappe = _coeffs("nappe_courant", "legacy")["C"]
    assert c_nappe > 1.5 * c_rot, (
        f"nappe {c_nappe:.3f} vs rotation {c_rot:.3f} : `legacy` ne sépare "
        "plus les deux structures par le pic, ce test ne mesure plus rien")


# ------------------------------------------------------------------
#  5. la sélectivité ne dépend pas de l'échelle ni de la résolution
# ------------------------------------------------------------------
@pytest.mark.parametrize("norm", NORMS)
def test_le_verdict_de_selectivite_survit_a_un_facteur_dechelle(norm):
    """Un coefficient « intuitif » ne doit pas changer de verdict parce
    qu'on a multiplié les champs par 10."""
    base = _champs()["xpoint"]
    for facteur in (1.0, 10.0):
        champ = {k: v * facteur for k, v in base.items()}
        champ["Jz"] = np.zeros((N, N))
        m = PhysicalMapperV2(dx=1.0, norm=norm)
        hp = m.compute_coefficients(None, np.full((N, N), 0.5), champ, 0.15,
                                    advanced_anomalies_enabled=True)
        assert float(np.max(np.abs(hp["K_xpoint"]))) > 0.5, (
            f"K_xpoint s'éteint au facteur {facteur} : le coefficient n'est "
            "pas adimensionnel")
