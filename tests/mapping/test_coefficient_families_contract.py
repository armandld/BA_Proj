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

def test_les_coefficients_s_effondrent_quand_la_grille_se_raffine():
    """Sur un champ physique FIXE, seule la resolution changeant.

    MESURE (rotation solide, hyperparametres deployes) :

        N       H_edges      C_edges      K_plaquettes
        32     2.10e-01     1.04e+01       1.00e+02
        64     6.02e-03     1.78e+00       9.48e+01
       128     2.89e-05     3.43e-01       1.86e+01
       256     2.27e-05     8.32e-02       0.00e+00

    Les trois familles s'effondrent, et `K_plaquettes` atteint **exactement
    zero a N=256** — la resolution d'entrainement.

    Ce n'est PAS un defaut d'implementation : les seuils sont des Reynolds
    de maille (`omega_crit = RE_CRIT nu / (dx^2 v0)`), qui croissent en
    1/dx^2. Une grille plus fine resout mieux, donc demande moins de
    raffinement. La logique AMR est correcte.

    Mais la CONSEQUENCE doit rester visible : a la configuration
    d'entrainement (N=256, Re=Rm=800), l'hamiltonien perd sa structure ZZZZ
    sur les champs canoniques. C'est le meme fait que le verdict deja
    publie « le terme ZZZZ etait numeriquement mort », vu ici au niveau du
    coefficient plutot que du circuit.

    Ce test EPINGLE la mesure. Il tombera si un changement de
    normalisation, de seuil ou d'hyperparametres deplace cet equilibre —
    et il faudra alors remesurer, pas ajuster.
    """
    valeurs = {}
    for n in (32, 64, 128, 256):
        global N
        ancien, N = N, n
        try:
            sim, grid = _sim_avec(_rotation_solide)
            c, _ = _coeffs(sim, grid, 0.3)
            valeurs[n] = (
                max(np.abs(np.asarray(a)).max() for a in c["H_edges"]),
                max(np.abs(np.asarray(a)).max() for a in c["C_edges"]),
                np.abs(np.asarray(c["K_plaquettes"])).max(),
            )
        finally:
            N = ancien

    # decroissance monotone des trois familles
    for i, nom in enumerate(("H_edges", "C_edges", "K_plaquettes")):
        suite = [valeurs[n][i] for n in (32, 64, 128, 256)]
        assert suite == sorted(suite, reverse=True), (
            f"{nom} n'est plus monotone decroissant en resolution : {suite}")

    assert valeurs[32][2] > 50.0, (
        f"K_plaquettes vaut {valeurs[32][2]:.3e} a N=32, attendu ~1.0e+02 — "
        f"sans structure a la grille grossiere, ce test ne separerait rien")
    assert valeurs[256][2] == pytest.approx(0.0, abs=1e-12), (
        f"K_plaquettes vaut {valeurs[256][2]:.3e} a N=256 : l'equilibre a "
        f"change, remesurer la table de la docstring")


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
