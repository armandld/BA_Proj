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
    mixte             omega != 0 ET J != 0, pics de rapport > 2
    mixte_desequilibre  idem, mais |J| CENT fois |omega|

Les deux normalisations sont exercées. Avant ce fichier, **aucun test
n'exerçait `norm="max"` hors des invariances** : basculer le défaut aurait
laissé toute la réponse physique non vérifiée.

Ce que les mesures établissent, et que ces tests épinglent :

1. `K_xpoint` est **sélectif** — il ne répond qu'au det(nabla B) < 0.
2. `K_plaquettes` **ne l'est pas**, dans les deux modes : c'est une somme
   `|omega| + |J|`, donc un vortex et une nappe de courant appariés rendent
   la **même** valeur. Le terme capte « il se passe quelque chose de
   rotationnel OU magnétique », pas un type — et c'est assumé : distinguer
   les deux demanderait deux familles de portes ZZZZ pour la même
   information.
3. `C_edges` ne l'est pas non plus : `sqrt(|dv|^2 + |dB|^2)` confond un saut
   hydrodynamique et un saut magnétique.
4. **Ce que `norm="max"` change à la plaquette, et qui est l'objet principal
   de ce fichier.** `legacy` somme les magnitudes BRUTES sous un dénominateur
   COMMUN : le signal le plus fort écrase l'autre en proportion de son
   amplitude. `max` rend chaque magnitude adimensionnelle par son PROPRE
   maximum avant de sommer, si bien que les deux structures pèsent 1/2
   chacune quel que soit leur rapport d'amplitude.

   Ce n'est pas un raffinement théorique. Mesuré sur les quatre scénarios
   canoniques à N=256, poids effectif de la vorticité sous `legacy` :

       harris_tearing     0,000 - 0,003 - 0,006   -> la VORTICITE est morte
       kelvin_helmholtz   0,975 - 0,993 - 1,000   -> le COURANT est mort
       mhd_rotor          0,193 - 0,391 - 1,000
       orszag_tang        0,212 - 0,278 - 0,400

   Sur **deux scénarios sur quatre**, l'une des deux structures que le terme
   prétend détecter ne contribue pas.

5. Sous `norm="max"`, le PIC vaut `w_zz` / `w_zzzz` sur **tout** champ non
   uniforme : la magnitude ne distingue plus rien et toute l'information de
   structure passe dans le MOTIF spatial.

Sur le point 5, une version antérieure de ce fichier concluait à un
« arbitrage » : `legacy` porterait dans son pic une information de structure
que `max` retirerait. C'est **faux**, et la mesure le dit. Comme `max|K|`
vaut 1 sous `max`, `max|C|` EST le poids de la famille ZZ contre la famille
ZZZZ ; sous `legacy` il passe de 3,121 (rotation lisse) à 8,095 (nappe
raide), un facteur 2,59 sur l'équilibre de deux familles de termes, décidé
par la forme du champ au lieu de la conception. Ce n'est pas une information,
c'est un couplage parasite. Il n'y a donc pas d'arbitrage.

Les points 2, 3 et 5 ne sont pas des défauts déclarés : ce sont des
propriétés de conception, mesurées et épinglées pour qu'un changement les
fasse crier.
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


def _bosse(Y, centre, sigma=0.45):
    """Structure localisée en `centre`, décroissante à 5 sigma du bord."""
    return np.exp(-((Y - centre) ** 2) / (2 * sigma ** 2))


def _champs():
    """Champs dont la réponse est connue analytiquement."""
    X, Y = _grille()
    Z, U = np.zeros((N, N)), np.ones((N, N))
    return {
        "uniforme":      dict(vx=U, vy=Z, Bx=U, By=Z),
        "rotation":      dict(vx=-np.sin(Y), vy=np.sin(X), Bx=Z, By=Z),
        "nappe_courant": dict(vx=Z, vy=Z, Bx=np.tanh(np.sin(Y)), By=Z),
        "xpoint":        dict(vx=Z, vy=Z, Bx=np.sin(Y), By=np.sin(X)),
        # MIXTE : un tourbillon et une nappe de courant SPATIALEMENT
        # SEPARES — le tourbillon en y = pi/2, la nappe en y = 3pi/2,
        # recouvrement 3,7e-07. La separation est indispensable : sur un
        # champ ou les deux structures culminent AU MEME POINT, aucune ne
        # peut en ecraser une autre et les deux formules coincident. C'est
        # l'erreur qu'une premiere version de ce fichier a commise.
        # Amplitudes appariees : max|omega| = max|J| a 1e-3 pres.
        "mixte":         dict(vx=-_bosse(Y, np.pi / 2), vy=Z,
                              Bx=_bosse(Y, 3 * np.pi / 2), By=Z),
        # MIXTE DESEQUILIBRE : meme geometrie, mais |J| exactement CENT fois
        # |omega|. Reproduit en laboratoire ce que les champs MHD reels font
        # (facteur 179 sur harris, 84 sur KH).
        "mixte_desequilibre": dict(vx=-_bosse(Y, np.pi / 2), vy=Z,
                                   Bx=100.0 * _bosse(Y, 3 * np.pi / 2), By=Z),
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
        "K_carte": np.abs(np.asarray(hp["K_plaquettes"], dtype=float)),
    }


def _pics_des_deux_signaux(nom):
    """max|omega| et max|J| du champ, calculés par le même opérateur que
    le mappeur — sinon on mesure l'écart entre deux stencils."""
    from Simulation.grid import curl_z
    c = _champs()[nom]
    return (float(np.max(np.abs(curl_z(c["vx"], c["vy"], True)))),
            float(np.max(np.abs(curl_z(c["Bx"], c["By"], True)))))


# ------------------------------------------------------------------
#  1. le champ nul : rien ne doit répondre
# ------------------------------------------------------------------
@pytest.mark.parametrize("norm", NORMS)
def test_un_champ_uniforme_neteint_les_trois_familles(norm):
    r = _coeffs("uniforme", norm)
    for nom in ("C", "K", "Kxp"):
        assert r[nom] == pytest.approx(0.0, abs=1e-12), \
            f"{nom} = {r[nom]:.3e} sur un champ uniforme"


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

    positifs, negatifs = det > 1e-9, det < -1e-9
    assert positifs.any() and negatifs.any(), (
        "le champ d'essai n'a pas les deux signes : il ne sépare rien")
    assert np.allclose(kxp[positifs], 0.0, atol=1e-12), (
        f"K_xpoint vaut jusqu'à {kxp[positifs].max():.3e} là où det > 0 : "
        "le terme répond aux points ELLIPTIQUES, qu'il est censé ignorer")
    assert kxp[negatifs].max() > 0.5, (
        "K_xpoint est muet là où det < 0 : il ne détecte plus les X-points")


# ------------------------------------------------------------------
#  3. la plaquette ne distingue PAS un type — assumé, dans les deux modes
# ------------------------------------------------------------------
@pytest.mark.parametrize("norm", NORMS)
def test_la_plaquette_ne_distingue_pas_un_vortex_dune_nappe(norm):
    """Propriété de CONCEPTION, épinglée pour qu'un changement crie.

    La plaquette est une SOMME : un champ purement rotationnel et un champ
    purement magnétique, chacun normalisé, rendent le même K. Le terme capte
    « il se passe quelque chose », pas un type.

    Distinguer les deux demanderait **deux familles de portes ZZZZ** pour la
    même information — décision de USER : trop cher pour ce que ça rapporte.
    Ce test est donc un épinglage définitif, pas une lacune à combler.
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
#  4. LE POINT PRINCIPAL : sous `max`, la structure faible n'est plus
#     écrasée par la forte
# ------------------------------------------------------------------
def _contribution_de_la_vorticite(norm):
    """Valeur de la plaquette là où omega culmine, RAPPORTEE à son pic.

    Sur `mixte_desequilibre`, |J| vaut ~100 fois |omega|. Si la formule
    somme les magnitudes brutes sous un dénominateur commun, la vorticité
    ne peut pas dépasser ~1/100 du pic. Si chaque signal est normalisé par
    son propre maximum, elle atteint son plein poids.
    """
    from Simulation.grid import curl_z
    c = _champs()["mixte_desequilibre"]
    om = np.abs(curl_z(c["vx"], c["vy"], True))
    carte = _coeffs("mixte_desequilibre", norm)["K_carte"]
    ou = np.unravel_index(np.argmax(om), om.shape)
    return float(carte[ou]) / float(carte.max())


def test_sous_legacy_la_structure_faible_est_ecrasee_par_la_forte():
    """Le défaut que `max` corrige, mesuré en laboratoire.

    Sur quelle entrée ce test échoue : si `legacy` cessait de partager un
    dénominateur entre les deux magnitudes — auquel cas il n'y aurait plus
    rien à corriger et ce fichier n'aurait plus d'objet.
    """
    pic_om, pic_jz = _pics_des_deux_signaux("mixte_desequilibre")
    assert pic_jz / pic_om > 50, (
        f"le champ d'essai n'est plus déséquilibré (rapport {pic_jz/pic_om:.1f}) : "
        "il ne peut plus montrer l'écrasement")
    part = _contribution_de_la_vorticite("legacy")
    assert part < 0.05, (
        f"la vorticité atteint {part:.3f} du pic sous `legacy` alors que son "
        "amplitude est 100 fois plus faible : l'écrasement a disparu")


def test_sous_max_la_structure_faible_pese_autant_que_la_forte():
    """Ce que la modification achète — et l'objet de la demande de USER.

    Sur quelle entrée ce test échoue : si `max` revenait à un dénominateur
    commun, la vorticité retomberait à ~1/100 et ce test rougirait.
    """
    part = _contribution_de_la_vorticite("max")
    assert part > 0.4, (
        f"la vorticité n'atteint que {part:.3f} du pic sous `max` : les deux "
        "magnitudes ne sont plus rendues adimensionnelles séparément")


def test_les_deux_modes_different_bien_sur_ce_champ():
    """Le garde du garde : les deux tests ci-dessus doivent porter sur des
    valeurs RÉELLEMENT différentes, sinon ils ne mesurent qu'un seuil."""
    l, m = (_contribution_de_la_vorticite(n) for n in ("legacy", "max"))
    assert m > 10 * l, (
        f"legacy {l:.4f} contre max {m:.4f} : les deux formules ne se "
        "séparent plus sur ce champ")


@pytest.mark.parametrize("norm", NORMS)
def test_les_deux_signaux_contribuent_encore_sur_un_champ_equilibre(norm):
    """L'autre côté : la correction ne doit pas ÉTEINDRE le cas déjà sain.

    Sur un champ dont les deux pics sont du même ordre, les deux modes
    doivent tous deux laisser la vorticité peser. Sans ce test, une formule
    qui écraserait l'autre signal passerait les deux tests précédents.
    """
    from Simulation.grid import curl_z
    c = _champs()["mixte"]
    carte = _coeffs("mixte", norm)["K_carte"]
    for etiquette, signal in (("vorticité", curl_z(c["vx"], c["vy"], True)),
                              ("courant", curl_z(c["Bx"], c["By"], True))):
        ou = np.unravel_index(np.argmax(np.abs(signal)), signal.shape)
        part = float(carte[ou]) / float(carte.max())
        assert part > 0.8, (
            f"la {etiquette} ne pèse que {part:.3f} du pic sur un champ "
            "pourtant apparié : la correction a éteint le cas déjà sain")


def test_sous_max_le_pic_de_la_plaquette_vaut_exactement_son_poids():
    """L'invariance qui rend le terme comparable d'un `dim` à l'autre.

    Elle survit au changement de formule : normaliser chaque signal PUIS
    diviser la somme par son propre max laisse `max|K| == w_zzzz`.
    """
    for nom in ("rotation", "nappe_courant", "xpoint", "mixte",
                "mixte_desequilibre"):
        k = _coeffs(nom, "max")["K"]
        assert k == pytest.approx(PhysicalMapperV2.W_ZZZZ, rel=1e-9), (
            f"max|K| = {k:.6f} sur '{nom}', attendu {PhysicalMapperV2.W_ZZZZ}")


def test_sous_legacy_le_pic_de_la_plaquette_reste_SOUS_son_poids():
    """Le champ qui SÉPARE : `legacy` prend DEUX maxima en des points
    possiblement différents, donc `max|K| < w_zzzz` dès qu'ils ne coïncident
    pas — d'une quantité qui dépend du champ. Sans ce test, celui du dessus
    passerait pour une propriété universelle du calcul."""
    k = _coeffs("mixte", "legacy")["K"]
    assert k < 0.95 * PhysicalMapperV2.W_ZZZZ, (
        f"max|K| = {k:.4f} sous `legacy` : le mode ne sous-estime plus le "
        "pic, les deux modes sont devenus indiscernables sur ce point")


# ------------------------------------------------------------------
#  5. l'ENSEMBLE des clés rendues fait partie du contrat
# ------------------------------------------------------------------
#: Clés rendues sans le terme X-point, mesuré le 21 août 2026. Liste FERMÉE.
_CLES_SANS_XPOINT = {"H_edges", "C_edges", "K_plaquettes",
                     "threshold_amr", "w_z_frac"}


@pytest.mark.parametrize("norm", NORMS)
def test_lensemble_des_cles_rendues_est_ferme(norm):
    """Pourquoi les valeurs ne suffisent pas.

    `src/call_vqa_shell.py` ne consomme pas le dictionnaire clé par clé : il
    somme `|coeff|` sur TOUTES les clés tableau, sans liste blanche, pour
    former `E_max`. **Ajouter une clé est donc déjà un changement de
    comportement**, même quand aucune valeur partagée ne bouge d'un bit —
    mesuré à +15,9 % (`legacy`) et +33,6 % (`max`) pour deux clés de plus.
    `RescaleArrays.py` itère lui aussi sur toutes les clés.

    Ce test existe parce que ce défaut a été livré une fois, annoncé comme
    son contraire sur la foi d'une comparaison bit à bit des seules valeurs
    partagées.
    """
    champ = dict(_champs()["mixte"]); champ["Jz"] = np.zeros((N, N))
    m = PhysicalMapperV2(dx=2 * np.pi / N, norm=norm)
    hp = m.compute_coefficients(None, np.full((N, N), 0.5), champ, 0.15)
    assert set(hp) == _CLES_SANS_XPOINT, (
        f"clés en trop : {sorted(set(hp) - _CLES_SANS_XPOINT)}, "
        f"manquantes : {sorted(_CLES_SANS_XPOINT - set(hp))}. Tout ajout "
        "déplace `E_max` chez tout consommateur qui somme sur les clés.")

    hp_x = m.compute_coefficients(None, np.full((N, N), 0.5), champ, 0.15,
                                  advanced_anomalies_enabled=True)
    assert set(hp_x) == _CLES_SANS_XPOINT | {"K_xpoint"}, (
        "le drapeau X-point n'ajoute plus exactement une clé")


def test_la_formule_legacy_de_la_plaquette_est_inchangee():
    """`legacy` doit rester la reproduction EXACTE du chemin historique —
    c'est tout ce qui lui reste comme raison d'être. Recalcul indépendant.

    Sur quelle entrée ce test échoue : le jour où quelqu'un « améliore »
    aussi `legacy`, auquel cas plus aucun mode ne reproduit les artefacts
    gelés et toute comparaison avant/après devient impossible.
    """
    from Simulation.grid import curl_z
    champ = dict(_champs()["mixte"]); champ["Jz"] = np.zeros((N, N))
    m = PhysicalMapperV2(dx=2 * np.pi / N, norm="legacy")
    hp = m.compute_coefficients(None, np.full((N, N), 0.5), champ, 0.15)

    omega = curl_z(champ["vx"], champ["vy"], True)
    jz = curl_z(champ["Bx"], champ["By"], True)
    attendu = -PhysicalMapperV2.W_ZZZZ * (np.abs(omega) + np.abs(jz)) / (
        np.max(np.abs(omega)) + np.max(np.abs(jz)) + PhysicalMapperV2.EPS)
    np.testing.assert_allclose(hp["K_plaquettes"], attendu, rtol=1e-12)


# ------------------------------------------------------------------
#  6. ce que `norm="max"` change au SENS du nombre
# ------------------------------------------------------------------
def test_sous_max_le_pic_est_constant_donc_ne_porte_aucune_structure():
    """Conséquence directe du changement demandé, à connaître.

    `max` force `max|C| = w_zz` et `max|K| = w_zzzz` sur TOUT champ non
    uniforme : la magnitude cesse de distinguer quoi que ce soit, et toute
    l'information de structure passe dans le motif spatial.
    """
    for champ in ("rotation", "nappe_courant", "xpoint"):
        r = _coeffs(champ, "max")
        assert r["C"] == pytest.approx(PhysicalMapperV2.W_ZZ, rel=1e-9), \
            f"max|C| = {r['C']:.4f} sur '{champ}'"
        assert r["K"] == pytest.approx(PhysicalMapperV2.W_ZZZZ, rel=1e-9), \
            f"max|K| = {r['K']:.4f} sur '{champ}'"


def test_sous_legacy_le_poids_relatif_des_familles_derive_avec_le_champ():
    """Le champ qui SÉPARE les deux normalisations — et le vrai enjeu.

    Sous `legacy`, `max|C|` est le rapport pic/moyenne des sauts. Comme
    `max|K|` reste de l'ordre de 1, ce pic EST le poids de la famille ZZ
    relativement à la famille ZZZZ dans l'hamiltonien. Il dérive donc avec la
    seule « spikiness » de l'entrée :

        rotation lisse  ZZ:ZZZZ = 3,121
        nappe raide     ZZ:ZZZZ = 8,095      -> facteur 2,59

    Ce n'est pas une information de structure qu'on perdrait en passant à
    `max` : c'est l'équilibre de deux familles de termes décidé par la forme
    du champ au lieu de la conception. C'est la réponse à « où est
    l'arbitrage » : il n'y en a pas.
    """
    rot, nappe = _coeffs("rotation", "legacy"), _coeffs("nappe_courant", "legacy")
    r_rot, r_nappe = rot["C"] / rot["K"], nappe["C"] / nappe["K"]
    assert r_nappe > 1.5 * r_rot, (
        f"ZZ:ZZZZ nappe {r_nappe:.3f} vs rotation {r_rot:.3f} : `legacy` ne "
        "fait plus dériver le poids relatif, ce test ne mesure plus rien")


def test_sous_max_le_poids_relatif_des_familles_est_celui_de_la_conception():
    """Ce que `max` achète, et que le pic seul ne montrait pas.

    Le rapport ZZ:ZZZZ vaut `W_ZZ / W_ZZZZ` sur TOUT champ non uniforme :
    l'équilibre des deux familles redevient un paramètre de conception, pas
    une propriété de l'instantané. Ce test est l'argument pour `max`.
    """
    attendu = PhysicalMapperV2.W_ZZ / PhysicalMapperV2.W_ZZZZ
    for nom in ("rotation", "nappe_courant", "xpoint", "mixte"):
        r = _coeffs(nom, "max")
        assert r["C"] / r["K"] == pytest.approx(attendu, rel=1e-9), (
            f"ZZ:ZZZZ = {r['C'] / r['K']:.4f} sur '{nom}', attendu {attendu} "
            "— `max` ne borne plus les deux familles au même endroit")


# ------------------------------------------------------------------
#  7. la sélectivité ne dépend ni de l'échelle ni de la résolution
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


def test_la_plaquette_max_est_invariante_a_lechelle_de_chaque_champ():
    """Le corollaire de la normalisation séparée : multiplier B seul par 10
    ne doit RIEN changer sous `max` — et changer beaucoup sous `legacy`."""
    base = dict(_champs()["mixte"])
    def carte(norm, facteur):
        champ = dict(base); champ["Bx"] = base["Bx"] * facteur
        champ["Jz"] = np.zeros((N, N))
        m = PhysicalMapperV2(dx=2 * np.pi / N, norm=norm)
        hp = m.compute_coefficients(None, np.full((N, N), 0.5), champ, 0.15)
        return np.abs(np.asarray(hp["K_plaquettes"], dtype=float))

    np.testing.assert_allclose(carte("max", 1.0), carte("max", 10.0), rtol=1e-9,
                               err_msg="`max` n'est plus invariant a l'echelle "
                                       "de B seul : la normalisation separee a saute")
    a, b = carte("legacy", 1.0), carte("legacy", 10.0)
    assert not np.allclose(a, b, rtol=1e-3), (
        "`legacy` est devenu invariant lui aussi : les deux modes ne se "
        "separent plus, ce test ne mesure plus rien")


# ------------------------------------------------------------------
#  8. le fait qui motive tout le changement, sur les VRAIS champs
# ------------------------------------------------------------------
@pytest.mark.slow
def test_sur_les_champs_reels_legacy_eteint_une_structure_sur_deux_scenarios():
    """Ce que la mesure de laboratoire prétend reproduire, sur le corpus.

    Sur quelle entrée ce test échoue : le jour où les DNS du dépôt cessent
    d'être dominés par une seule structure — auquel cas la justification du
    changement de formule tombe et il faut la réécrire, pas retoucher le
    seuil.
    """
    import glob
    from Simulation.grid import curl_z
    attendus = {"harris_tearing": "vorticite", "kelvin_helmholtz": "courant"}
    vus = {}
    for f in sorted(glob.glob(os.path.join(_RACINE, "results",
                                           "dns_*_Re400_N256.npz"))):
        nom = os.path.basename(f)[4:-16]
        if nom not in attendus:
            continue
        d = np.load(f)
        si = len(d["vx"]) // 2
        om = curl_z(d["vx"][si].astype(float), d["vy"][si].astype(float), True)
        jz = curl_z(d["Bx"][si].astype(float), d["By"][si].astype(float), True)
        a, b = float(np.max(np.abs(om))), float(np.max(np.abs(jz)))
        vus[nom] = a / (a + b)

    if len(vus) < 2:
        pytest.skip("artefacts DNS N=256 absents")

    assert vus["harris_tearing"] < 0.05, (
        f"harris : la vorticité pèse {vus['harris_tearing']:.4f} sous `legacy`, "
        "elle n'est plus écrasée — la justification du changement a bougé")
    assert vus["kelvin_helmholtz"] > 0.95, (
        f"kh : le courant pèse {1 - vus['kelvin_helmholtz']:.4f}, il n'est plus "
        "écrasé — même remarque")
