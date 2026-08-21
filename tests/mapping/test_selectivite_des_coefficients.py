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
4. `K_vorticity` et `K_current` — la plaquette **scindée** — SONT sélectifs,
   chacun sur son signal, et rendent des réponses opposées sur la paire de
   champs que la somme confond. `K_plaquettes` est laissé inchangé bit à bit.
5. Sous `norm="max"`, le PIC vaut `w_zz` / `w_zzzz` sur **tout** champ non
   uniforme : la magnitude ne distingue plus rien et toute l'information de
   structure passe dans le MOTIF spatial.

Sur le point 5, une version antérieure de ce fichier concluait à un
« arbitrage » : `legacy` porterait dans son pic une information de structure
que `max` retirerait. C'est **faux**, et la mesure le dit. Comme `max|K|`
vaut 1 dans les deux modes, `max|C|` EST le poids de la famille ZZ contre la
famille ZZZZ ; sous `legacy` il passe de 3,121 (rotation lisse) à 8,095
(nappe raide), un facteur 2,59 sur l'équilibre de deux familles de termes,
décidé par la forme du champ au lieu de la conception. Ce n'est pas une
information, c'est un couplage parasite. Il n'y a donc pas d'arbitrage :
`max` le retire, et les deux tests de la section 4 épinglent les deux côtés.

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


def _champs():
    """Quatre champs dont la réponse est connue analytiquement."""
    X, Y = _grille()
    Z, U = np.zeros((N, N)), np.ones((N, N))
    return {
        "uniforme":      dict(vx=U, vy=Z, Bx=U, By=Z),
        "rotation":      dict(vx=-np.sin(Y), vy=np.sin(X), Bx=Z, By=Z),
        "nappe_courant": dict(vx=Z, vy=Z, Bx=np.tanh(np.sin(Y)), By=Z),
        "xpoint":        dict(vx=Z, vy=Z, Bx=np.sin(Y), By=np.sin(X)),
        # MIXTE : omega != 0 ET J != 0, avec des pics DIFFERENTS (facteur ~4
        # sur l'amplitude magnetique). C'est le seul champ du jeu qui peut
        # separer une normalisation par signal d'une normalisation commune —
        # sur un champ pur, les deux coincident exactement.
        "mixte":         dict(vx=-np.sin(Y), vy=np.sin(X),
                              Bx=0.25 * np.tanh(np.sin(Y)), By=Z),
    }


def _coeffs(nom, norm, xpoint=True):
    champ = dict(_champs()[nom])
    champ["Jz"] = np.zeros((N, N))
    m = PhysicalMapperV2(dx=2 * np.pi / N, norm=norm)
    hp = m.compute_coefficients(None, np.full((N, N), 0.5), champ, 0.15,
                                advanced_anomalies_enabled=xpoint,
                                split_plaquette=True)
    Ch, Cv = hp["C_edges"]
    return {
        "C": max(float(np.max(np.abs(Ch))), float(np.max(np.abs(Cv)))),
        "K": float(np.max(np.abs(hp["K_plaquettes"]))),
        "Kvort": float(np.max(np.abs(hp["K_vorticity"]))),
        "Kcurr": float(np.max(np.abs(hp["K_current"]))),
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
#  3 bis. la plaquette SCINDEE, elle, est selective
# ------------------------------------------------------------------
@pytest.mark.parametrize("norm", NORMS)
def test_k_vorticity_ne_repond_quau_rotationnel(norm):
    """Ce que `K_plaquettes` ne sait pas faire, et pourquoi le scindement.

    Sur quelle entree ce test echoue : si `K_vorticity` se remettait a lire
    `J_z`, il redeviendrait aveugle au type.
    """
    assert _coeffs("rotation", norm)["Kvort"] > 0.5
    for muet in ("nappe_courant", "uniforme"):
        v = _coeffs(muet, norm)["Kvort"]
        assert v == pytest.approx(0.0, abs=1e-12), (
            f"K_vorticity = {v:.3e} sur '{muet}', ou omega_z = 0")


@pytest.mark.parametrize("norm", NORMS)
def test_k_current_ne_repond_quau_courant(norm):
    assert _coeffs("nappe_courant", norm)["Kcurr"] > 0.5
    for muet in ("rotation", "uniforme"):
        v = _coeffs(muet, norm)["Kcurr"]
        assert v == pytest.approx(0.0, abs=1e-12), (
            f"K_current = {v:.3e} sur '{muet}', ou J_z = 0")


@pytest.mark.parametrize("norm", NORMS)
def test_les_deux_termes_scindes_separent_ce_que_la_somme_confond(norm):
    """Le contraste, mis cote a cote — c'est l'objet du scindement.

    Meme paire de champs : la somme rend la MEME valeur, les deux termes
    scindes rendent des reponses opposees.
    """
    rot, nappe = _coeffs("rotation", norm), _coeffs("nappe_courant", norm)
    assert rot["K"] == pytest.approx(nappe["K"], rel=1e-9), (
        "la somme ne confond plus les deux : le scindement perd son motif")
    assert rot["Kvort"] > 0.5 and rot["Kcurr"] == pytest.approx(0.0, abs=1e-12)
    assert nappe["Kcurr"] > 0.5 and nappe["Kvort"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("norm", NORMS)
def test_les_termes_scindes_sont_bornes_par_leur_poids(norm):
    """Normalises par le max de LEUR signal dans les deux modes : le pic
    vaut w_zzzz des qu'il y a du signal, sans dependre de `norm`."""
    for champ, cle in (("rotation", "Kvort"), ("nappe_courant", "Kcurr")):
        assert _coeffs(champ, norm)[cle] == pytest.approx(
            PhysicalMapperV2.W_ZZZZ, rel=1e-9)


@pytest.mark.parametrize("norm", NORMS)
def test_chaque_terme_scinde_est_normalise_par_SON_signal(norm):
    """Le champ MIXTE, sans lequel ce choix de conception n'est pas testable.

    Les deux termes scindés divisent chacun par le max de son propre signal.
    L'alternative — une normalisation COMMUNE, `max|omega| + max|J|`, celle
    de `K_plaquettes` — réintroduirait exactement ce qu'on retire : le poids
    de chaque famille dépendrait du pic de l'AUTRE, donc de la forme du
    champ. Ici B est 4 fois plus faible que v ; sous normalisation commune
    `K_vorticity` culminerait à ~0,8 et `K_current` à ~0,2 au lieu de 1 et 1.

    Ce test a été écrit APRÈS avoir constaté que la mutation « normalisation
    commune » survivait à tout le fichier : les quatre champs d'origine sont
    PURS (omega = 0 ou J = 0), et sur un champ pur les deux normalisations
    coïncident au bit près. Un test qui ne peut pas échouer est un défaut.
    """
    r = _coeffs("mixte", norm)
    assert r["Kvort"] == pytest.approx(PhysicalMapperV2.W_ZZZZ, rel=1e-9), (
        f"K_vorticity culmine à {r['Kvort']:.4f} et non {PhysicalMapperV2.W_ZZZZ} "
        "sur un champ où omega ET J sont actifs : la normalisation n'est plus "
        "celle de son propre signal")
    assert r["Kcurr"] == pytest.approx(PhysicalMapperV2.W_ZZZZ, rel=1e-9), (
        f"K_current culmine à {r['Kcurr']:.4f} et non {PhysicalMapperV2.W_ZZZZ} "
        "sur un champ mixte : même défaut, côté courant")


def test_le_champ_mixte_a_bien_les_deux_signaux_a_des_pics_differents():
    """Le garde du garde : si le champ mixte cessait d'être mixte, ou si ses
    deux pics devenaient égaux, le test ci-dessus redeviendrait incapable de
    distinguer les deux normalisations sans que rien ne crie."""
    from Simulation.grid import curl_z
    c = _champs()["mixte"]
    omega = curl_z(c["vx"], c["vy"], True)
    jz = curl_z(c["Bx"], c["By"], True)
    p_om, p_jz = float(np.max(np.abs(omega))), float(np.max(np.abs(jz)))
    assert p_om > 1e-6 and p_jz > 1e-6, (
        f"le champ mixte n'est plus mixte : max|omega|={p_om:.3e}, "
        f"max|J|={p_jz:.3e}")
    rapport = max(p_om, p_jz) / min(p_om, p_jz)
    assert rapport > 2.0, (
        f"pics trop proches (rapport {rapport:.2f}) : une normalisation "
        "commune donnerait presque le même résultat, le test ne sépare plus")


# ------------------------------------------------------------------
#  3 ter. le scindement est OPT-IN, et voici pourquoi
# ------------------------------------------------------------------
#: L'ensemble EXACT des clés rendues par défaut, mesuré le 21 août 2026.
#: Fermé : une clé de plus fait rougir ce fichier, et c'est le but.
_CLES_PAR_DEFAUT = {"H_edges", "C_edges", "K_plaquettes",
                    "threshold_amr", "w_z_frac"}


def _emax_facon_call_vqa_shell(hp):
    """Reproduit le calcul de `src/call_vqa_shell.py` : somme de |coeff| sur
    TOUTES les clés tableau, sans liste blanche."""
    total = 0.0
    for v in hp.values():
        for a in (v if isinstance(v, (tuple, list)) else (v,)):
            if isinstance(a, np.ndarray):
                total += float(np.sum(np.abs(a)))
    return total


@pytest.mark.parametrize("norm", NORMS)
def test_par_defaut_lensemble_des_cles_rendues_est_inchange(norm):
    """L'ENSEMBLE des clés fait partie du contrat, pas seulement les valeurs.

    `src/call_vqa_shell.py` consomme le dictionnaire comme un TOUT : il somme
    `|coeff|` sur toutes les clés tableau pour former `E_max`, sans liste
    blanche. **Ajouter une clé est donc déjà un changement de comportement**,
    même quand aucune valeur partagée ne bouge d'un bit.
    """
    champ = dict(_champs()["mixte"]); champ["Jz"] = np.zeros((N, N))
    m = PhysicalMapperV2(dx=2 * np.pi / N, norm=norm)
    hp = m.compute_coefficients(None, np.full((N, N), 0.5), champ, 0.15)
    assert set(hp) == _CLES_PAR_DEFAUT, (
        f"clés en trop : {sorted(set(hp) - _CLES_PAR_DEFAUT)}, "
        f"manquantes : {sorted(_CLES_PAR_DEFAUT - set(hp))}. Tout ajout "
        "déplace `E_max` chez tout consommateur qui somme sur les clés.")


@pytest.mark.parametrize("norm", NORMS)
def test_le_scindement_deplace_E_max_ce_qui_est_la_raison_du_opt_in(norm):
    """Le champ qui SÉPARE — et la mesure qui justifie le défaut à False.

    Sans ce test, `split_plaquette=False` passerait pour une précaution
    décorative. Il chiffre ce que le drapeau évite : +15,9 % (`legacy`) et
    +34,2 % (`max`) sur `E_max`, mesuré le 21 août 2026 sur un champ bruité.

    Sur quelle entrée ce test échoue : si le scindement devenait le défaut,
    ou si les deux termes cessaient de contribuer à la somme.
    """
    champ = dict(_champs()["mixte"]); champ["Jz"] = np.zeros((N, N))
    m = PhysicalMapperV2(dx=2 * np.pi / N, norm=norm)
    sans = m.compute_coefficients(None, np.full((N, N), 0.5), champ, 0.15)
    avec = m.compute_coefficients(None, np.full((N, N), 0.5), champ, 0.15,
                                  split_plaquette=True)

    for cle in sans:
        a, b = sans[cle], avec[cle]
        if isinstance(a, tuple):
            assert all(np.array_equal(p, q) for p, q in zip(a, b)), cle
        elif isinstance(a, np.ndarray):
            assert np.array_equal(a, b), f"{cle} bouge : le scindement n'est plus inerte"
        else:
            assert a == b, cle

    e_sans, e_avec = _emax_facon_call_vqa_shell(sans), _emax_facon_call_vqa_shell(avec)
    assert e_avec > 1.05 * e_sans, (
        f"E_max {e_sans:.2f} -> {e_avec:.2f} : les deux termes scindés ne "
        "pèsent plus dans la somme, ce test ne justifie plus le opt-in")


def test_le_scindement_ne_touche_pas_les_cles_preexistantes():
    """`K_plaquettes` doit rester ce qu'il etait — sinon des nombres publies
    bougeraient. Recalcul independant de la formule d'origine."""
    X, Y = _grille()
    champ = dict(_champs()["xpoint"]); champ["Jz"] = np.zeros((N, N))
    m = PhysicalMapperV2(dx=2 * np.pi / N, norm="legacy")
    hp = m.compute_coefficients(None, np.full((N, N), 0.5), champ, 0.15)
    from Simulation.grid import curl_z
    omega = curl_z(champ["vx"], champ["vy"], True)
    jz = curl_z(champ["Bx"], champ["By"], True)
    attendu = -PhysicalMapperV2.W_ZZZZ * (np.abs(omega) + np.abs(jz)) / (
        np.max(np.abs(omega)) + np.max(np.abs(jz)) + PhysicalMapperV2.EPS)
    np.testing.assert_allclose(hp["K_plaquettes"], attendu, rtol=1e-12)


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


def test_sous_legacy_le_poids_relatif_des_familles_derive_avec_le_champ():
    """Le champ qui SÉPARE les deux normalisations — et le vrai enjeu.

    Sous `legacy`, `max|C|` est le rapport pic/moyenne des sauts. Comme
    `max|K|` vaut 1 dans les deux modes, ce pic EST le poids de la famille ZZ
    relativement à la famille ZZZZ dans l'hamiltonien. Il dérive donc avec la
    seule « spikiness » de l'entrée :

        rotation lisse  ZZ:ZZZZ = 3,121
        nappe raide     ZZ:ZZZZ = 8,095      -> facteur 2,59

    Ce n'est pas une information de structure qu'on perdrait en passant à
    `max` : c'est l'équilibre de deux familles de termes décidé par la forme
    du champ au lieu de la conception. C'est la réponse à « où est
    l'arbitrage » : il n'y en a pas.

    Sur quelle entrée ce test échoue : le jour où `legacy` cesse de diviser
    par une moyenne, les deux modes deviennent indiscernables et le test
    ci-dessus ne mesure plus rien.
    """
    rot = _coeffs("rotation", "legacy")
    nappe = _coeffs("nappe_courant", "legacy")
    r_rot, r_nappe = rot["C"] / rot["K"], nappe["C"] / nappe["K"]
    assert r_nappe > 1.5 * r_rot, (
        f"ZZ:ZZZZ nappe {r_nappe:.3f} vs rotation {r_rot:.3f} : `legacy` ne "
        "fait plus dériver le poids relatif, ce test ne mesure plus rien")


def test_sous_max_le_poids_relatif_des_familles_est_celui_de_la_conception():
    """Ce que `max` achète, et que le pic seul ne montrait pas.

    Le rapport ZZ:ZZZZ vaut `W_ZZ / W_ZZZZ` sur TOUT champ non uniforme :
    l'équilibre des deux familles redevient un paramètre de conception, pas
    une propriété de l'instantané. Ce test est l'argument pour `max` ; sans
    lui, le passage à `max` ne serait justifié que par l'invariance en `dim`.
    """
    attendu = PhysicalMapperV2.W_ZZ / PhysicalMapperV2.W_ZZZZ
    for nom in ("rotation", "nappe_courant", "xpoint"):
        r = _coeffs(nom, "max")
        assert r["C"] / r["K"] == pytest.approx(attendu, rel=1e-9), (
            f"ZZ:ZZZZ = {r['C'] / r['K']:.4f} sur '{nom}', attendu {attendu} "
            "— `max` ne borne plus les deux familles au même endroit")


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
