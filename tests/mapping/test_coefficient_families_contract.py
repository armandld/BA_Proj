"""Chaque famille de coefficients remplit-elle son role ?

Regle de ce banc : **tout passe par la vraie `compute_coefficients`.**

Reconstruire la chaine etage par etage dans un test donne un test qui ne
voit pas `src/` changer -- piege trouve dans mon propre banc de points X,
ou 20 tests passaient a l'identique apres une modification du mappeur.
Ici, chaque assertion interroge l'objet deploye.

Les quatre familles et leur contrat, lu dans le code :

  H_edges       biais Z adaptatif : `alpha_z * (score - threshold_amr)`
                avec `alpha_z = w_z_frac * median(|C|, |K|)`. Donc : nul
                exactement au seuil, signe qui bascule de part et d'autre,
                amplitude SUBORDONNEE aux couplages.

  C_edges       couplage ZZ, ferromagnetique (negatif), module par une
                fenetre gaussienne centree sur le seuil :
                `exp(-((score - threshold_amr)/sigma)^2)`. Donc : maximal
                la ou la decision classique est incertaine, eteint loin.

  K_plaquettes  ZZZZ de vorticite/courant, `-sqrt(fluide^2 + magnetique^2)`.
                Donc : negatif, et nul sur un champ sans structure.

  K_xpoint      ZZZZ de point X, `max(0, -det(J_B))`. Couvert en detail par
                `test_xpoint_at_training_resolution.py`.

La derniere section verifie que ce que le banc mesure est bien ce que le
CIRCUIT recoit, apres le reechantillonnage du pipeline.
"""

import numpy as np
import pytest

from Simulation.grid import PeriodicGrid
from Simulation.solver import MHDSolver
from Simulation.PhysToAngle import AngleMapper
from Simulation.HamiltParams import PhysicalMapper
from Simulation.RescaleArrays import get_adaptive_flux

N = 64
RE = RM = 800
#: Hyperparametres deployes, figes ici pour que le banc ne bouge pas si le
#: JSON change. Les valeurs viennent de `best_hyperparams.json`.
HP = dict(gamma_hydro=2.1272, gamma_mag=2.3611, kappa=14.3321, sigma=0.05,
          beta_curl=0.8199, beta_xpoint=0.4256, w_z_frac=0.1013)


def _mappeur(grid):
    return PhysicalMapper(cs=1.0, nu=grid.L / RE, eta_mhd=grid.L / RM,
                          dx=grid.dx, **HP)


def _sim_avec(champ):
    """Solveur dont on impose les champs — reponse connue a la main."""
    grid = PeriodicGrid(N)
    sim = MHDSolver(grid, dt=1e-3, Re=RE, Rm=RM)
    x = np.arange(N) * grid.dx
    X, Y = np.meshgrid(x, x, indexing="ij")     # AXIS_X=0, AXIS_Y=1
    sim.vx, sim.vy, sim.Bx, sim.By = champ(X, Y, grid)
    return sim, grid


def _coeffs(sim, grid, seuil, anomalies=True, score=None):
    """LA fonction deployee. Aucun test de ce fichier ne la contourne."""
    mapper = _mappeur(grid)
    etat = sim.get_fluxes()
    if score is None:
        score = AngleMapper.classical_score(etat)
    return mapper.compute_coefficients(
        sim, score, etat, threshold_amr=seuil,
        advanced_anomalies_enabled=anomalies), score


# ── champs d'essai, reponse connue ───────────────────────────────────

def _calme(X, Y, g):
    z = np.zeros_like(X)
    return z, z, np.ones_like(X), z          # B uniforme, v nul

def _rotation_solide(X, Y, g):
    return -(Y - g.L / 2), (X - g.L / 2), np.ones_like(X), np.zeros_like(X)

def _cisaillement_magnetique(X, Y, g):
    z = np.zeros_like(X)
    return z, z, np.sin(Y), z                # du courant, pas de nul


# ══════════════════════════════════════════════════════════════════════
#  H_edges — le biais Z
# ══════════════════════════════════════════════════════════════════════

def test_h_edges_sannule_exactement_au_seuil():
    """`z_bias = alpha_z * (score - threshold_amr)` : un score UNIFORME
    place au seuil doit rendre un biais nul, quel que soit alpha_z."""
    sim, grid = _sim_avec(_rotation_solide)
    seuil = 0.37
    score = np.full((N, N), seuil)
    c, _ = _coeffs(sim, grid, seuil, score=score)
    for k, H in enumerate(c["H_edges"]):
        assert np.abs(np.asarray(H)).max() == pytest.approx(0.0, abs=1e-12), (
            f"H_edges[{k}] non nul alors que score == threshold_amr")


def test_h_edges_change_de_signe_de_part_et_dautre_du_seuil():
    """Le biais doit ORIENTER la decision : un score au-dessus du seuil et
    un score en dessous doivent produire des biais de signes opposes.

    Sans cela, le terme Z ne briserait pas la degenerescence dans le bon
    sens — il la briserait au hasard.
    """
    sim, grid = _sim_avec(_rotation_solide)
    seuil = 0.4
    haut, _ = _coeffs(sim, grid, seuil, score=np.full((N, N), seuil + 0.3))
    bas, _ = _coeffs(sim, grid, seuil, score=np.full((N, N), seuil - 0.3))

    h_haut = np.asarray(haut["H_edges"][0])
    h_bas = np.asarray(bas["H_edges"][0])
    assert np.abs(h_haut).max() > 0, "biais nul au-dessus du seuil"
    assert np.abs(h_bas).max() > 0, "biais nul en dessous du seuil"
    assert np.sign(h_haut).sum() * np.sign(h_bas).sum() < 0, (
        f"les deux biais ont le meme signe : {np.sign(h_haut).sum():+.0f} et "
        f"{np.sign(h_bas).sum():+.0f}")


def test_h_edges_reste_subordonne_aux_couplages():
    """`alpha_z = w_z_frac * median(|C|, |K|)` avec w_z_frac ≈ 0.10.

    Le biais Z doit rester PETIT devant les couplages, sinon il decide seul
    et le quantique degenere vers la regle classique — c'est exactement ce
    que la borne haute de `w_z_frac` interroge.
    """
    sim, grid = _sim_avec(_rotation_solide)
    c, score = _coeffs(sim, grid, 0.3)

    # MEME ensemble ET MEME FILTRE que le code : `> 1e-10`, pas `> 0`.
    #
    # Deux erreurs commises en montant ce banc, toutes deux du meme genre :
    # reproduire un calcul sans reproduire son filtre. Avec `> 0` sur |C|
    # seul, la mediane vaut 5.9e-22 ; avec `> 0` sur |C| et |K|, 1.7e-22 —
    # dans les deux cas la queue de la fenetre gaussienne ecrase tout. Le
    # seuil `1e-10` du code l'ecarte et rend C_scale ~ 0.203. Le code etait
    # juste ; c'est le test qui mesurait autre chose.
    couplages = np.concatenate(
        [np.abs(np.asarray(a)).ravel() for a in c["C_edges"]]
        + [np.abs(np.asarray(c["K_plaquettes"])).ravel()])
    couplages = couplages[couplages > 1e-10]
    if couplages.size == 0:
        pytest.skip("aucun couplage non nul sur ce champ — rien a comparer")

    h_max = max(np.abs(np.asarray(a)).max() for a in c["H_edges"])
    ecart_max = np.abs(score - 0.3).max()
    borne = HP["w_z_frac"] * np.median(couplages) * ecart_max * 1.01

    assert h_max <= borne, (
        f"biais Z max {h_max:.4e} au-dela de la borne {borne:.4e} "
        f"(w_z_frac={HP['w_z_frac']} x mediane {np.median(couplages):.4e} "
        f"x ecart {ecart_max:.4f})")


# ══════════════════════════════════════════════════════════════════════
#  C_edges — le couplage ZZ et sa fenetre d'incertitude
# ══════════════════════════════════════════════════════════════════════

def test_c_edges_est_ferromagnetique():
    """Le couplage ZZ doit etre negatif : deux aretes voisines gagnent a
    prendre la meme decision. Un signe positif inverserait la physique."""
    sim, grid = _sim_avec(_rotation_solide)
    c, _ = _coeffs(sim, grid, 0.3)
    vus = 0
    for k, C in enumerate(c["C_edges"]):
        C = np.asarray(C)
        actifs = C[np.abs(C) > 1e-12]
        vus += actifs.size
        if actifs.size == 0:
            continue
        assert (actifs <= 0).all(), (
            f"C_edges[{k}] porte {int((actifs > 0).sum())} valeurs positives "
            f"(max {actifs.max():.4e}) — couplage antiferromagnetique")
    assert vus > 0, "aucun couplage actif — ce test ne verifierait aucun signe"


def test_la_fenetre_dincertitude_eteint_le_couplage_loin_du_seuil():
    """`exp(-((score - threshold_amr)/sigma)^2)` avec sigma = 0.05.

    C'est le CHAMP QUI SEPARE de cette famille : a 10 sigma du seuil, la
    fenetre vaut exp(-100) ~ 4e-44. Le couplage doit s'eteindre. Sinon la
    correction quantique s'applique partout, y compris la ou la decision
    classique est certaine — et le terme perd sa raison d'etre.
    """
    sim, grid = _sim_avec(_rotation_solide)
    seuil = 0.4
    au_seuil, _ = _coeffs(sim, grid, seuil, score=np.full((N, N), seuil))
    tres_loin, _ = _coeffs(sim, grid, seuil,
                           score=np.full((N, N), seuil + 10 * HP["sigma"]))

    c_proche = max(np.abs(np.asarray(a)).max() for a in au_seuil["C_edges"])
    c_loin = max(np.abs(np.asarray(a)).max() for a in tres_loin["C_edges"])

    assert c_proche > 0, "couplage nul AU seuil — le test ne separerait rien"
    assert c_loin < c_proche * 1e-6, (
        f"a 10 sigma du seuil le couplage vaut encore {c_loin:.4e} contre "
        f"{c_proche:.4e} au seuil — la fenetre n'eteint pas")


# ══════════════════════════════════════════════════════════════════════
#  K_plaquettes — le ZZZZ de vorticite / courant
# ══════════════════════════════════════════════════════════════════════

def test_k_plaquettes_est_de_parite_paire_negative():
    """`-sqrt(fluide^2 + magnetique^2)` : jamais positif, par construction.
    Un signe positif retournerait la contribution ZZZZ dans le cout."""
    sim, grid = _sim_avec(_rotation_solide)
    c, _ = _coeffs(sim, grid, 0.3)
    K = np.asarray(c["K_plaquettes"])
    assert np.abs(K).max() > 1.0, (
        f"K_plaquettes est nul ({np.abs(K).max():.4e}) sur la rotation "
        f"solide — ce test ne verifierait aucun signe")
    assert (K <= 1e-15).all(), (
        f"K_plaquettes porte {int((K > 1e-15).sum())} valeurs positives, "
        f"max {K.max():.4e}")


def test_k_plaquettes_est_nul_sur_un_champ_sans_structure():
    """Champ de controle : B uniforme, v nul. Ni vorticite ni courant, donc
    aucune raison de raffiner."""
    sim, grid = _sim_avec(_calme)
    c, _ = _coeffs(sim, grid, 0.3)
    K = np.asarray(c["K_plaquettes"])
    assert np.abs(K).max() == pytest.approx(0.0, abs=1e-12), (
        f"K_plaquettes vaut {np.abs(K).max():.4e} sur un champ uniforme")


def test_k_plaquettes_repond_a_la_structure():
    """Et il doit REPONDRE quand la structure existe — sans quoi le test
    precedent serait satisfait par un coefficient mort."""
    calme, g1 = _sim_avec(_calme)
    actif, g2 = _sim_avec(_rotation_solide)
    k_calme = np.abs(np.asarray(_coeffs(calme, g1, 0.3)[0]["K_plaquettes"])).max()
    k_actif = np.abs(np.asarray(_coeffs(actif, g2, 0.3)[0]["K_plaquettes"])).max()
    assert k_actif > 1.0, (
        f"rotation solide : K_plaquettes max {k_actif:.4e}, attendu ~9.5e+01")
    assert k_calme == pytest.approx(0.0, abs=1e-12)

    # NOTE mesuree : sur `_cisaillement_magnetique` (Bx = sin y), K_plaquettes
    # vaut 0 alors que |Jz| franchit son seuil — la porte `f_Rm_cell = |B|dx/eta`
    # s'annule la ou |B| = 0, c'est-a-dire exactement ou le courant est
    # maximal sur ce champ. Pour K_plaquettes la porte est DEFENDABLE (sans
    # champ, pas de raideur magnetique) ; c'est pour K_xpoint qu'elle ne
    # l'etait pas, un point X etant par definition un zero de B.


# ══════════════════════════════════════════════════════════════════════
#  Le drapeau des anomalies avancees
# ══════════════════════════════════════════════════════════════════════

def test_le_drapeau_commande_bien_la_presence_de_K_xpoint():
    """`advanced_anomalies_enabled` doit decider, seul, si la cle existe.

    C'est l'axe sur lequel `study/` et la campagne d'entrainement
    divergent : la campagne l'active sur 6/6 scenarios, `study/` code
    `False` en dur.
    """
    sim, grid = _sim_avec(_cisaillement_magnetique)
    avec, _ = _coeffs(sim, grid, 0.3, anomalies=True)
    sans, _ = _coeffs(sim, grid, 0.3, anomalies=False)
    assert "K_xpoint" in avec, "drapeau vrai mais cle absente"
    assert "K_xpoint" not in sans, "drapeau faux mais cle presente"


# ══════════════════════════════════════════════════════════════════════
#  Coincidence avec le pipeline — ce que le CIRCUIT recoit
# ══════════════════════════════════════════════════════════════════════

def _reechantillonne(c, score, grid, sim, dim):
    """Le chemin exact du pipeline : `get_adaptive_flux` a `target_dim`."""
    mapper_angles = AngleMapper(v0=1.0, B0=1.0, w_compress=2.0, w_shear=1.0)
    Phi = mapper_angles.compute_stress_flux(sim.get_fluxes())
    _, _, mini, _ = get_adaptive_flux(
        Phi["phi_horizontal"], Phi["phi_vertical"], None, None,
        score, c, target_dim=dim, type_filter=True)
    return mini


@pytest.mark.parametrize("dim", [2, 4])
def test_le_reechantillonnage_preserve_les_signes(dim):
    """Le banc mesure a pleine resolution ; le circuit voit la version
    reechantillonnee. Les proprietes verifiees plus haut doivent survivre —
    sinon le banc et le pipeline ne parlent pas du meme hamiltonien.

    C'est la question 4 appliquee a ce banc lui-meme : deux chemins censes
    coincider coincident-ils ?
    """
    sim, grid = _sim_avec(_rotation_solide)
    c, score = _coeffs(sim, grid, 0.3)
    mini = _reechantillonne(c, score, grid, sim, dim)

    for k, C in enumerate(mini["C_edges"]):
        C = np.asarray(C)
        actifs = C[np.abs(C) > 1e-12]
        if actifs.size:
            assert (actifs <= 0).all(), (
                f"dim={dim} : C_edges[{k}] devient antiferromagnetique apres "
                f"reechantillonnage (max {actifs.max():.4e})")

    K = np.asarray(mini["K_plaquettes"])
    assert (K <= 1e-15).all(), (
        f"dim={dim} : K_plaquettes devient positif apres reechantillonnage "
        f"(max {K.max():.4e})")


@pytest.mark.parametrize("dim", [2, 4])
def test_le_reechantillonnage_rend_les_formes_attendues(dim):
    """Le circuit porte `2 * dim^2` qubits : les tableaux doivent avoir la
    forme que `cost_hamiltonian` attend, sans quoi l'erreur ne se voit que
    bien plus loin."""
    sim, grid = _sim_avec(_rotation_solide)
    c, score = _coeffs(sim, grid, 0.3)
    mini = _reechantillonne(c, score, grid, sim, dim)

    for nom in ("H_edges", "C_edges"):
        for k, a in enumerate(mini[nom]):
            assert np.asarray(a).shape == (dim, dim), (
                f"dim={dim} : {nom}[{k}] a la forme "
                f"{np.asarray(a).shape}, attendu ({dim}, {dim})")
    assert np.asarray(mini["K_plaquettes"]).shape == (dim, dim)


def test_le_champ_calme_reste_calme_jusqu_au_circuit():
    """Bout en bout : un champ sans structure ne doit produire AUCUN
    couplage, ni a pleine resolution, ni apres reechantillonnage.

    Si le reechantillonnage fabriquait du signal a partir de rien, tout le
    reste du banc serait sans valeur.
    """
    sim, grid = _sim_avec(_calme)
    c, score = _coeffs(sim, grid, 0.3)
    mini = _reechantillonne(c, score, grid, sim, 2)

    assert np.abs(np.asarray(c["K_plaquettes"])).max() == pytest.approx(0.0, abs=1e-12)
    assert np.abs(np.asarray(mini["K_plaquettes"])).max() == pytest.approx(0.0, abs=1e-12), (
        "le reechantillonnage fabrique du ZZZZ a partir d'un champ uniforme")


# ══════════════════════════════════════════════════════════════════════
#  Tenue en resolution — le meme champ physique, plusieurs grilles
# ══════════════════════════════════════════════════════════════════════

def test_le_critere_relatif_empeche_l_effondrement_en_resolution():
    """Sur un champ physique FIXE, seule la resolution changeant.

    AVANT le critere relatif (seuil ABSOLU seul) :

        N       H_edges      C_edges      K_plaquettes
        32     2.10e-01     1.04e+01       1.00e+02
        64     6.02e-03     1.78e+00       9.48e+01
       128     2.89e-05     3.43e-01       1.86e+01
       256     2.27e-05     8.32e-02       0.00e+00   <- mort

    APRES :

       256     2.27e-05     8.32e-02       6.59e+01   <- vivant

    Le seuil de maille croit en 1/dx^2 : a N=256 plus aucune cellule ne le
    franchit, et le terme a quatre corps disparaissait EXACTEMENT a la
    resolution d'entrainement. Le relais relatif le maintient.

    `H_edges` et `C_edges` decroissent toujours : ils sont gouvernes par
    la fenetre gaussienne et par `C_scale`, pas par le seuil de maille.
    C'est un mecanisme different, et ce test ne le confond pas avec
    celui-ci.
    """
    valeurs = {}
    for n in (32, 64, 128, 256):
        global N
        ancien, N = N, n
        try:
            sim, grid = _sim_avec(_rotation_solide)
            c, _ = _coeffs(sim, grid, 0.3)
            valeurs[n] = np.abs(np.asarray(c["K_plaquettes"])).max()
        finally:
            N = ancien

    assert valeurs[32] > 50.0, (
        f"K_plaquettes vaut {valeurs[32]:.3e} a N=32, attendu ~1.0e+02 — "
        f"sans structure a la grille grossiere, ce test ne separerait rien")
    assert valeurs[256] > 1.0, (
        f"K_plaquettes vaut {valeurs[256]:.3e} a N=256, attendu ~6.6e+01 : "
        f"le terme a quatre corps meurt de nouveau a la resolution "
        f"d'entrainement")

# ══════════════════════════════════════════════════════════════════════
#  Le signal brut porte-t-il l'instabilite ? (avant tout seuil)
# ══════════════════════════════════════════════════════════════════════

def test_le_signal_brut_discrimine_meme_quand_le_seuil_absolu_ne_tire_pas():
    """A N=256, les seuils absolus ne sont pas franchis — mais le signal
    BRUT porte un contraste enorme.

    MESURE (N=256, Re=Rm=800, seuil de maille = 1.304e+01) :

      scenario            grandeur     max/mediane   max/seuil
      orszag_tang         |omega|             2.8      0.1612
      orszag_tang         sqrt(det)          36.3      0.1088
      harris_tearing      |Jz|               49.5      0.2714
      harris_tearing      sqrt(det)        1104.3      0.0135
      island_coalescence  |Jz|               68.0      0.3393
      island_coalescence  sqrt(det)         222.8      0.0256
      mhd_rotor           |omega|           752.3      1.5038

    Deux lectures, toutes deux dans les chiffres :

    1. L'INFORMATION EXISTE. Un facteur 1104 entre le maximum et la mediane
       sur harris_tearing signifie que les cellules a raffiner se
       distinguent tres nettement. C'est le seuil ABSOLU qui l'efface, pas
       l'absence de structure.

    2. UN SEUIL UNIQUE NE PEUT PAS SERVIR TOUS LES SCENARIOS. |omega| vaut
       1.55e-02 au maximum sur harris_tearing et 1.96e+01 sur mhd_rotor —
       trois ordres de grandeur d'ecart. Le meme seuil de 13.04 pour les
       deux ne peut pas etre le bon pour les deux.

    3. `sqrt(det)` est le canal le PLUS discriminant sur trois scenarios
       sur quatre (36x, 1104x, 223x contre 2.6x, 49.5x, 68x pour |Jz|).
       C'est le terme de point X qui porte le plus d'information — et
       c'etait celui qui etait doublement casse.

    Ce test epingle la mesure. Il ne prescrit rien : passer a un critere
    RELATIF est une decision de conception, pas une correction.
    """
    from Simulation.HamiltParams import PhysicalMapper as PM

    n = 256
    g = PeriodicGrid(n)
    sim = MHDSolver(g, dt=1e-3, Re=RE, Rm=RM)
    sim.init_harris_tearing()
    for _ in range(200):
        sim.step_full()

    st = sim.get_fluxes()
    dx, eta = g.dx, g.L / RM
    crit = 1.0 * eta / dx ** 2

    jz = np.abs(st["Jz"])
    xp = np.sqrt(np.maximum(
        0.0, -PM._compute_det_jacobian_B(st["Bx"], st["By"], dx)))

    for nom, v, contraste_min in (("|Jz|", jz, 20.0), ("sqrt(det)", xp, 100.0)):
        med = np.median(v)
        assert med > 0, f"{nom} : mediane nulle, le contraste serait indefini"
        contraste = v.max() / med
        assert contraste > contraste_min, (
            f"{nom} : contraste max/mediane = {contraste:.1f}, attendu "
            f"> {contraste_min} — le signal ne discrimine plus")
        assert v.max() < crit, (
            f"{nom} : max {v.max():.3e} franchit le seuil {crit:.3e} — "
            f"l'equilibre a change, remesurer la table de la docstring")


# ══════════════════════════════════════════════════════════════════════
#  Le critere RELATIF
# ══════════════════════════════════════════════════════════════════════

def test_le_critere_relatif_ne_fabrique_pas_de_signal():
    """L'invariant de surete, et le plus important de ce fichier.

    Un champ rigoureusement uniforme n'a AUCUNE cellule « plus instable »
    que les autres. Son percentile vaut son maximum, le contraste seuille
    rend zero partout. Si ce test tombait, le critere relatif inventerait
    du raffinement sur du vide — et tout ce qu'il produit ailleurs serait
    sans valeur.
    """
    sim, grid = _sim_avec(_calme)
    c, _ = _coeffs(sim, grid, 0.3)
    for nom in ("K_plaquettes", "K_xpoint"):
        v = np.abs(np.asarray(c[nom])).max()
        assert v == pytest.approx(0.0, abs=1e-12), (
            f"{nom} vaut {v:.4e} sur un champ uniforme : le critere relatif "
            f"fabrique du signal a partir de rien")


def test_l_absolu_l_emporte_quand_il_tire():
    """`min(absolu, percentile)` : des qu'une cellule franchit le critere
    physique, le comportement d'origine doit etre conserve A L'IDENTIQUE.

    Sans cette clause, le critere relatif remplacerait la physique au lieu
    de la completer.
    """
    from Simulation.HamiltParams import PhysicalMapper as PM
    signal = np.array([0.1, 0.5, 2.0, 10.0])      # le max franchit 1.0
    assert PM._effective_crit(signal, 1.0) == 1.0

    signal_faible = np.array([0.01, 0.02, 0.05])   # aucun ne franchit
    eff = PM._effective_crit(signal_faible, 1.0)
    assert eff < 1.0, f"le relatif n'a pas pris le relais : {eff}"
    assert eff == pytest.approx(
        np.percentile(signal_faible, PM.RELATIVE_PERCENTILE))


def test_le_critere_relatif_redonne_un_hamiltonien_non_vide_a_N256():
    """MESURE avant / apres, a la resolution d'entrainement.

        scenario              K_plaq avant -> apres   K_xpoint avant -> apres
        orszag_tang              0 -> 2.78e-01           0 -> 8.34e-02
        harris_tearing           0 -> 2.66e-03           0 -> 9.81e-01
        island_coalescence       0 -> 1.54e-02           0 -> 5.55e-01
        mhd_rotor                0 -> 6.68e+01           0 -> 1.00e+01

    Le terme a quatre corps existait sur AUCUN des quatre scenarios ; il
    existe desormais sur les quatre.

    A noter, et coherent avec le contraste du signal brut : sur les deux
    scenarios de reconnexion, le canal point X DOMINE le canal courant --
    0.981 contre 0.0027 sur harris_tearing (facteur 370), 0.555 contre
    0.0154 sur island_coalescence (facteur 36). C'est ce que la physique
    predit, et c'est le canal qui etait doublement casse.
    """
    global N
    ancien, N = N, 256
    try:
        sim, grid = _sim_avec(_calme)      # place le solveur
        sim.init_harris_tearing()
        for _ in range(200):
            sim.step_full()
        c, _ = _coeffs(sim, grid, 0.3)
    finally:
        N = ancien

    k_plaq = np.abs(np.asarray(c["K_plaquettes"])).max()
    k_xp = np.abs(np.asarray(c["K_xpoint"])).max()

    assert k_xp > 0.1, (
        f"K_xpoint = {k_xp:.4e} a N=256 sur harris_tearing, attendu ~9.8e-01 "
        f"— le critere relatif ne prend pas le relais")
    assert k_plaq > 0.0, f"K_plaquettes reste nul ({k_plaq:.4e})"
    assert k_xp > k_plaq * 10, (
        f"sur un scenario de reconnexion, le canal point X ({k_xp:.4e}) "
        f"devrait dominer le canal courant ({k_plaq:.4e})")


# ══════════════════════════════════════════════════════════════════════
#  Le test qui compte : le coefficient pointe-t-il ou l'erreur EST ?
# ══════════════════════════════════════════════════════════════════════

def test_les_coefficients_pointent_ou_le_raffinement_est_necessaire():
    """Correlation de rang entre le coefficient et l'erreur REELLE.

    Tous les tests precedents verifient des contrats internes : signes,
    seuils, invariance. Celui-ci verifie la seule chose qui justifie le
    modele -- le coefficient designe-t-il les blocs ou la solution
    grossiere s'ecarte vraiment du DNS ?

    Protocole : meme scenario a N=128 (reference) et N=32 (grossier), meme
    nombre de pas ; erreur relative par bloc sur 8x8 = 64 blocs ; Spearman
    contre le coefficient moyen du bloc.

    MESURE :

      scenario              K_plaq   K_xpoint   max(K)   score classique
      harris_tearing         0.897     0.434     0.788        0.814
      island_coalescence     0.877     0.408     0.760        0.912
      mhd_rotor              0.755     0.680     0.759        0.528
      orszag_tang            0.249     0.311     0.443        0.422

    Trois lectures :

    1. `K_plaquettes` correle FORTEMENT (0.75 a 0.90) sur trois scenarios
       sur quatre. Le coefficient pointe bien ou le raffinement est
       necessaire — c'est le contrat central du modele, et il est tenu.

    2. Sur `mhd_rotor`, le coefficient BAT le score classique : 0.755
       contre 0.528. C'est la premiere preuve quantitative, dans ce depot,
       que le terme a quatre corps apporte quelque chose que l'indicateur
       lineaire n'a pas — et c'est precisement le scenario autour duquel
       `compare_rotor_budget` a ete construit.

    3. `orszag_tang` est faible pour TOUT (0.25 a 0.44), coefficients et
       score classique confondus. Ce n'est pas un defaut des coefficients :
       c'est le scenario le plus difficile pour n'importe quel indicateur
       local.

    Ce test est LENT (quatre paires de simulations). Il est le dernier du
    fichier pour cette raison.
    """
    pytest.importorskip("scipy")
    from scipy.stats import spearmanr
    from Simulation.HamiltParams import PhysicalMapper as PM

    nb = 8

    def _bloc_moy(a):
        n = a.shape[0]
        b = n // nb
        return a.reshape(nb, b, nb, b).mean(axis=(1, 3))

    NF, NC, pas = 128, 32, 200
    gf = PeriodicGrid(NF)
    sf = MHDSolver(gf, dt=1e-3, Re=RE, Rm=RM)
    sf.init_harris_tearing()
    gc = PeriodicGrid(NC)
    sc = MHDSolver(gc, dt=1e-3, Re=RE, Rm=RM)
    sc.init_harris_tearing()
    for _ in range(pas):
        sf.step_full()
    for _ in range(pas):
        sc.step_full()

    ff, fc = sf.get_fluxes(), sc.get_fluxes()
    err = np.zeros((nb, nb))
    for v in ("vx", "vy", "Bx", "By"):
        d = _bloc_moy(ff[v])
        c_ = _bloc_moy(np.repeat(np.repeat(fc[v], NF // NC, 0), NF // NC, 1))
        err += np.abs(d - c_) / (np.abs(d).mean() + 1e-12)

    m = PM(cs=1.0, nu=gf.L / RE, eta_mhd=gf.L / RM, dx=gf.dx, **HP)
    st = sf.get_fluxes()
    score = AngleMapper.classical_score(st)
    co = m.compute_coefficients(sf, score, st, threshold_amr=0.3,
                                advanced_anomalies_enabled=True)

    kp = _bloc_moy(np.abs(np.asarray(co["K_plaquettes"])))
    assert np.ptp(kp) > 0, (
        "K_plaquettes est constant sur les 64 blocs — la correlation serait "
        "indefinie et ce test ne verifierait rien")
    assert np.ptp(err) > 0, "l'erreur est constante — le controle est vide"

    rho = spearmanr(kp.ravel(), err.ravel()).statistic
    assert rho > 0.6, (
        f"correlation de rang coefficient/erreur = {rho:.3f} sur "
        f"harris_tearing, mesure de reference 0.897. En dessous de 0.6, le "
        f"coefficient ne designe plus les blocs a raffiner.")


def test_matrice_de_specificite_chaque_famille_repond_a_son_instabilite():
    """Chaque champ isole UNE instabilite ; quelle famille repond ?

    TOUS LES CHAMPS SONT PERIODIQUES. La premiere version de ce test
    utilisait une rotation solide `v = (-(y-L/2), x-L/2)`, qui est
    DISCONTINUE au raccord periodique. Son `K_plaquettes` valait 9.48e+01
    avec un maximum dans le coin (63, 63) : un artefact de bord, pas une
    mesure. Le reseau de vortex `v = (-sin y, sin x)` donne 5.01e-01, dans
    l'interieur. J'avais donc surestime le canal fluide d'un facteur 190.

    MESURE (N=64, hyperparametres deployes, via la vraie fonction) :

      champ                         H_edges    C_edges     K_plaq   K_xpoint
      reseau de vortex (fluide)   1.160e-05  1.826e-01  5.009e-01  0.000e+00
      nappe de courant (magnet.)  1.567e-07  7.183e-05  1.816e-05  0.000e+00
      cisaillement v (Q<0)        2.127e-08  2.724e-04  5.709e-03  0.000e+00
      point X magnetique          9.948e-07  3.328e-01  5.092e-06  9.585e-02
      uniforme (controle)         0.000e+00  0.000e+00  0.000e+00  0.000e+00

    CE QUI EST SAIN :

      - le controle uniforme rend zero sur les QUATRE familles ;
      - le reseau de vortex allume `K_plaquettes` (5.01e-01) et laisse
        `K_xpoint` a zero : le canal fluide ne deborde pas ;
      - le point X allume `K_xpoint` (9.59e-02) et laisse `K_plaquettes` a
        5.09e-06, soit 19 000 fois moins. Les deux canaux ZZZZ sont
        ORTHOGONAUX, ce qui est leur raison d'etre ;
      - `H_edges` reste subordonne partout.

    CE QUI RESTE OUVERT — le desequilibre fluide / magnetique :

      vortex 5.01e-01 contre nappe de courant 1.82e-05, soit un facteur
      **27 500** pour deux instabilites de meme nature. C'est beaucoup
      moins que les 10^6 que j'avais annonces sur le champ non periodique,
      mais c'est toujours un desequilibre reel.

      La cause n'est PAS localisee et je m'abstiens de la nommer : trois
      fois deja dans cette campagne, une reproduction incomplete d'un
      calcul m'a fait accuser du code juste.

    Ce test fige la matrice. Il tombera si l'equilibre change -- y compris
    si quelqu'un corrige le canal magnetique, et c'est voulu.
    """
    def _mesure(champ, seuil=0.3):
        sim, grid = _sim_avec(champ)
        c, _ = _coeffs(sim, grid, seuil)
        a = lambda v: np.abs(np.asarray(v)).max()
        return dict(H=max(a(v) for v in c["H_edges"]),
                    C=max(a(v) for v in c["C_edges"]),
                    Kp=a(c["K_plaquettes"]), Kx=a(c["K_xpoint"]))

    z = lambda X: np.zeros_like(X)
    o = lambda X: np.ones_like(X)
    # PERIODIQUES : un reseau de vortex, pas une rotation solide.
    vortex = _mesure(lambda X, Y, g: (-np.sin(Y), np.sin(X), o(X), z(X)))
    nappe = _mesure(lambda X, Y, g: (z(X), z(X), 1 + 0.8 * np.tanh(3 * np.sin(Y)), z(X)))
    xpt = _mesure(lambda X, Y, g: (z(X), z(X),
                                   np.sin(Y - g.L / 2), np.sin(X - g.L / 2)))
    calme = _mesure(_calme)

    for k, v in calme.items():
        assert v == pytest.approx(0.0, abs=1e-12), (
            f"champ uniforme : {k} vaut {v:.3e}")

    assert vortex["Kp"] > 0.1, (
        f"reseau de vortex : K_plaq = {vortex['Kp']:.3e}, attendu ~5.0e-01")
    assert vortex["Kx"] == pytest.approx(0.0, abs=1e-12), (
        f"vortex : K_xpoint = {vortex['Kx']:.3e}, il n'y a pas de nul magnetique")

    assert xpt["Kx"] > 1e-2, f"point X : K_xpoint = {xpt['Kx']:.3e}"
    assert xpt["Kx"] > xpt["Kp"] * 1000, (
        f"point X : K_xpoint {xpt['Kx']:.3e} contre K_plaq {xpt['Kp']:.3e} — "
        f"les deux canaux ZZZZ ne sont plus orthogonaux")

    for nom, m in (("vortex", vortex), ("point X", xpt)):
        assert m["H"] < m["C"], (
            f"{nom} : H_edges {m['H']:.3e} depasse C_edges {m['C']:.3e}")

    # LE DESEQUILIBRE OUVERT — epingle, pas corrige
    assert nappe["Kp"] < vortex["Kp"] / 1e3, (
        f"le canal magnetique n'est plus ecrase : nappe {nappe['Kp']:.3e} "
        f"contre vortex {vortex['Kp']:.3e} (facteur de reference 27 500). "
        f"Si c'est une correction deliberee, REMESURER cette matrice au "
        f"lieu d'ajuster ce seuil.")


def test_les_deux_portes_g_comparent_des_unites_differentes():
    """LA CAUSE du desequilibre fluide / magnetique, trouvee depuis
    l'INTERIEUR de la fonction via `_stages`.

    `g_rot` compare `Q_OW` a `Q_CRIT = 2.0`. `Q_OW` vient de
    `grid._compute_q_criterion(vx, vy, dx=dx)` : il PREND dx, donc il est
    en unites PHYSIQUES.

    `g_mag` compare `Jz_curl` a `J_CRIT = 1.0`. `Jz_curl` vient de
    `curl_z(Bx, By)`, qui ne prend PAS dx : c'est une difference finie en
    unites de GRILLE. Verifie ici : sur `Bx = 1 + 0.8 tanh(3 sin y)`, de
    derivee analytique 2.4, `curl_z` rend 0.2287 = 2.4 x dx.

    Les deux portes comparent donc des grandeurs de deux systemes d'unites
    differents a des seuils de meme ordre nominal. La porte magnetique est
    plus dure a franchir d'un facteur exactement 1/dx : 10.2 a N=64,
    20.4 a N=128, 40.7 a N=256 -- elle se degrade quand la grille se
    raffine. Quatrieme membre de la famille dimensionnelle de cette
    campagne.

    MESURE via `_stages`, nappe de courant a N=64 :
        mic_jz    = 1.541e-01   <- l'etage de SEUIL est sain
        f_Rm_cell = 8.346       <- la porte d'ECHELLE est saine
        g_mag     = 0.000       <- la porte TOPOLOGIQUE s'annule
        mag_comp  = 1.816e-05
    contre `g_rot = 1.000` et `fluid_comp = 5.0e-01` sur le vortex.

    EPINGLE, pas corrige : harmoniser les unites deplace TOUS les
    coefficients magnetiques. C'est une decision de USER.
    """
    from Simulation.grid import curl_z

    n = 64
    gr = PeriodicGrid(n)
    x = np.arange(n) * gr.dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    Bx = 1 + 0.8 * np.tanh(3 * np.sin(Y))
    By = np.zeros_like(X)

    brut = np.abs(curl_z(Bx, By, True)).max()
    assert brut == pytest.approx(2.4 * gr.dx, rel=0.05), (
        f"curl_z rend {brut:.4f}, attendu {2.4 * gr.dx:.4f} (derivee x dx). "
        f"S'il divise desormais par dx, le desequilibre est corrige — "
        f"REMESURER la matrice de specificite.")

    global N
    ancien, N = N, n
    try:
        sim, grid = _sim_avec(
            lambda X_, Y_, g_: (np.zeros_like(X_), np.zeros_like(X_),
                                1 + 0.8 * np.tanh(3 * np.sin(Y_)),
                                np.zeros_like(X_)))
        mapper = _mappeur(grid)
        etat = sim.get_fluxes()
        mapper.compute_coefficients(
            sim, AngleMapper.classical_score(etat), etat,
            threshold_amr=0.3, advanced_anomalies_enabled=True)
    finally:
        N = ancien

    et = mapper._stages
    assert np.abs(et["mic_jz"]).max() > 1e-2, (
        f"mic_jz = {np.abs(et['mic_jz']).max():.3e} : le fautif serait "
        f"l'etage de seuil, pas la porte — remesurer")
    assert np.abs(et["f_Rm_cell"]).max() > 1.0, (
        f"f_Rm_cell = {np.abs(et['f_Rm_cell']).max():.3e} : le fautif serait "
        f"la porte d'echelle")
    assert np.abs(et["g_mag"]).max() < 1e-3, (
        f"g_mag = {np.abs(et['g_mag']).max():.3e} : la porte magnetique ne "
        f"s'annule plus. Si les unites ont ete harmonisees, REMESURER.")
