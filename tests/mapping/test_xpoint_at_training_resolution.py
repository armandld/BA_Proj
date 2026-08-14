"""Le coefficient de point X **a la resolution d'entrainement**.

`test_beta_xpoint.py` valide deja la forme a N = 8. A cette taille, `dx`
est si grand que le seuil est trivialement franchi : le test ne pouvait
pas voir ce qui suit. La campagne, elle, tourne a **N = 256**.

Banc analytique — des champs dont la bonne reponse est connue a la main.
Un point X magnetique est un nul HYPERBOLIQUE de B (det(J_B) < 0), un
O-point un nul ELLIPTIQUE (det > 0). Le discriminant topologique est donc
le signe du determinant jacobien.

Ce que ce banc etablit, mesure par mesure, a N = 256 :

  1. la FORME `max(0, -det(J_B))` est saine — elle tire sur un point X,
     rend zero sur un O-point, et zero sur un cisaillement magnetique pur
     (le champ qui SEPARE : il porte du courant sans porter de point X) ;

  2. le SEUIL est sain — il ne tire que sur une nappe plus fine que
     ~3 cellules, c'est-a-dire reellement sous-resolue. C'est la bonne
     logique AMR, et elle explique que le terme se taise sur un point X
     lisse : a Rm = 800 et N = 256, un tel point X **est** resolu ;

  3. la PORTE `f_Rm_cell` eteint le coefficient EXACTEMENT au point X.
     Elle vaut `f_gate(|B| dx / eta)` — et un point X est par definition
     un zero de B. Le detecteur marque l'anneau autour du point X, jamais
     le point X lui-meme. Sur une grille grossiere (dim = 2 a 8), la
     cellule du point X est precisement celle qu'on veut voir signalee.

Le commentaire du code dit deja *« No separate g-gate needed (signal is
intrinsically localized) »* : le commentaire et le code se contredisent.
"""

import numpy as np
import pytest

from Simulation.HamiltParams import PhysicalMapper as PM

L = 2 * np.pi
B0 = 1.0
ETA = L / 800.0          # eta_mhd de la campagne (Rm = 800)
BETA_XPOINT = 0.4256     # valeur deployee
GAMMA_MAG = 2.36         # valeur deployee
N_ENTRAINEMENT = 256     # N_TRAINING de pipeline.PHASE


def _champ(N, a, genre):
    """Nul localise d'echelle `a`, amplitude O(B0) : gradient ~ B0/a.

    Garder l'amplitude fixe et retrecir `a` fabrique une vraie nappe fine.
    (Multiplier l'amplitude par `a` garderait le gradient constant et ne
    fabriquerait rien du tout — erreur commise puis corrigee en montant ce
    banc.)
    """
    dx = L / N
    x = np.arange(N) * dx - L / 2
    X, Y = np.meshgrid(x, x, indexing="ij")     # AXIS_X=0, AXIS_Y=1
    env = np.exp(-(X**2 + Y**2) / (2 * (L / 6) ** 2))
    if genre == "X":              # nul hyperbolique
        return dx, B0 * np.sin(Y / a) * env, B0 * np.sin(X / a) * env
    if genre == "O":              # nul elliptique
        return dx, -B0 * np.sin(Y / a) * env, B0 * np.sin(X / a) * env
    if genre == "cisaillement":   # du COURANT, aucun nul hyperbolique
        return dx, B0 * np.sin(Y / a) * env, np.zeros_like(X)
    raise ValueError(genre)


def _signal(Bx, By, dx):
    return np.maximum(0.0, -PM._compute_det_jacobian_B(Bx, By, dx))


def _mic(sig, dx):
    """L'etage de seuil, tel que `compute_coefficients` le construit."""
    return PM._threshold_contrast(
        sig / (B0**2 / dx**2), (1.0 * ETA / (dx * B0)) ** 2, BETA_XPOINT)


def _porte(Bx, By, dx):
    """`f_Rm_cell`, tel que `compute_coefficients` le construit."""
    return PM._f_gate(np.hypot(Bx, By) * dx / ETA, 1.0, GAMMA_MAG)


# ══════════════════════════════════════════════════════════════════════
#  1. La forme discrimine la topologie
# ══════════════════════════════════════════════════════════════════════

def test_le_determinant_separe_point_X_et_point_O():
    """Sur psi = cos(x)cos(y) : O-points en (0,0) et (pi,pi), X-points en
    (pi/2,pi/2) et (pi/2,3pi/2). Reponse connue a la main."""
    N = 128
    dx = L / N
    x = np.arange(N) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    Bx, By = -np.cos(X) * np.sin(Y), np.sin(X) * np.cos(Y)
    det = PM._compute_det_jacobian_B(Bx, By, dx)

    for (i, j), attendu in (((0, 0), "O"), ((N // 4, N // 4), "X"),
                            ((N // 2, N // 2), "O"), ((N // 4, 3 * N // 4), "X")):
        assert np.hypot(Bx[i, j], By[i, j]) < 1e-12, "ce point n'est pas un nul"
        lu = "X" if det[i, j] < 0 else "O"
        assert lu == attendu, (
            f"nul en ({i},{j}) : det={det[i, j]:+.4f} lu comme {lu}, "
            f"attendu {attendu}")


def test_le_signal_est_orthogonal_au_courant():
    """Le champ qui SEPARE : un cisaillement magnetique pur porte du
    courant et aucun point X.

    C'est tout l'interet du terme ZZZZ de point X face au ZZZZ de
    vorticite : un detecteur qui confondrait les deux tirerait ici.
    """
    N = N_ENTRAINEMENT
    dx, Bx, By = _champ(N, 2 * L / N, "cisaillement")

    Jz = (0.5 * (np.roll(By, -1, 0) - np.roll(By, 1, 0))
          - 0.5 * (np.roll(Bx, -1, 1) - np.roll(Bx, 1, 1))) / dx
    assert np.abs(Jz).max() > 1.0, (
        f"le controle ne porte pas de courant (|Jz|max={np.abs(Jz).max():.3e}) "
        f"— il ne separerait rien")
    assert _signal(Bx, By, dx).max() == pytest.approx(0.0, abs=1e-9), (
        "le signal de point X tire sur un cisaillement magnetique pur")


def test_le_signal_ignore_les_points_O():
    """Un O-point est un nul, mais elliptique : det > 0, signal nul."""
    N = N_ENTRAINEMENT
    dx, Bx, By = _champ(N, 2 * L / N, "O")
    i0 = N // 2
    assert np.hypot(Bx[i0, i0], By[i0, i0]) < 1e-12, "le centre n'est pas un nul"
    assert _signal(Bx, By, dx)[i0, i0] == pytest.approx(0.0, abs=1e-9)


# ══════════════════════════════════════════════════════════════════════
#  2. Le seuil implemente « nappe sous-resolue » — et c'est correct
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("epaisseur,doit_tirer",
                         [(8.0, False), (4.0, False), (2.0, True), (1.0, True)])
def test_le_seuil_ne_tire_que_sur_une_nappe_sous_resolue(epaisseur, doit_tirer):
    """A N = 256, une nappe de 8 ou 4 cellules est resolue : aucun
    raffinement requis. A 2 cellules et moins elle ne l'est plus.

    Ce n'est pas un defaut : c'est la bonne logique AMR. Elle explique
    pourquoi le terme se tait sur un point X lisse a la resolution
    d'entrainement.
    """
    N = N_ENTRAINEMENT
    dx, Bx, By = _champ(N, epaisseur * L / N, "X")
    mic_max = _mic(_signal(Bx, By, dx), dx).max()
    if doit_tirer:
        assert mic_max > 1e-3, (
            f"nappe de {epaisseur} cellules : seuil non franchi "
            f"({mic_max:.3e}) alors qu'elle est sous-resolue")
    else:
        assert mic_max == pytest.approx(0.0, abs=1e-12), (
            f"nappe de {epaisseur} cellules : le seuil tire ({mic_max:.3e}) "
            f"alors qu'elle est resolue")


# ══════════════════════════════════════════════════════════════════════
#  3. La porte eteint le coefficient au point X — comportement epingle
# ══════════════════════════════════════════════════════════════════════

def test_la_porte_sannule_exactement_au_nul_magnetique():
    """`f_Rm_cell = f_gate(|B| dx / eta)` et |B| = 0 a un point X."""
    N = N_ENTRAINEMENT
    dx, Bx, By = _champ(N, 2 * L / N, "X")
    i0 = N // 2
    porte = _porte(Bx, By, dx)

    assert np.hypot(Bx[i0, i0], By[i0, i0]) < 1e-12, "le centre n'est pas un nul"
    assert porte[i0, i0] == pytest.approx(0.0, abs=1e-12), (
        f"la porte vaut {porte[i0, i0]:.3e} au nul")
    assert porte.max() > 1.0, (
        f"la porte est faible partout (max={porte.max():.3e}) — ce test ne "
        f"montrerait pas qu'elle s'annule SPECIFIQUEMENT au nul")


def test_le_vrai_mappeur_marque_le_point_X(monkeypatch):
    """Le controle qui compte : il appelle `compute_coefficients` LUI-MEME.

    Les tests ci-dessus reconstruisent la chaine etage par etage pour
    l'inspecter — ils ne verraient donc pas un changement dans `src/`.
    Celui-ci passe par la vraie fonction, et c'est lui qui verrouille le
    retrait de la porte.
    """
    from Simulation.grid import PeriodicGrid
    from Simulation.solver import MHDSolver
    from Simulation.PhysToAngle import AngleMapper

    N = 64
    a = 2 * L / N
    grid = PeriodicGrid(N)
    sim = MHDSolver(grid, dt=1e-3, Re=800, Rm=800)
    dx, Bx, By = _champ(N, a, "X")
    # champ impose directement : on veut un point X franc, pas une evolution
    sim.Bx, sim.By = Bx, By
    sim.vx = np.zeros_like(Bx)
    sim.vy = np.zeros_like(Bx)

    mapper = PM(cs=1.0, nu=grid.L / 800, eta_mhd=ETA, dx=grid.dx,
                gamma_hydro=2.13, gamma_mag=GAMMA_MAG, kappa=14.33,
                sigma=0.05, beta_curl=0.82, beta_xpoint=BETA_XPOINT,
                w_z_frac=0.10)
    etat = sim.get_fluxes()
    score = AngleMapper.classical_score(etat)
    coeffs = mapper.compute_coefficients(
        sim, score, etat, threshold_amr=0.0, advanced_anomalies_enabled=True)

    assert "K_xpoint" in coeffs, "le drapeau est vrai, la cle doit exister"
    K = np.asarray(coeffs["K_xpoint"])
    i0 = N // 2

    assert np.abs(K).max() > 0.0, (
        "K_xpoint est nul partout — le seuil n'est pas atteint sur ce champ, "
        "le test ne verifierait rien")
    assert abs(K[i0, i0]) == pytest.approx(np.abs(K).max(), rel=1e-9), (
        f"le coefficient vaut {abs(K[i0, i0]):.4e} AU point X pour un maximum "
        f"de {np.abs(K).max():.4e} ailleurs : la porte |B| est-elle revenue ?")


def test_le_coefficient_est_nul_au_point_X_mais_pas_autour():
    """La consequence : le detecteur marque l'anneau, jamais le centre.

    EPINGLE l'ETAT ANTERIEUR, reconstruit ici etage par etage : la porte
    `f_Rm_cell` a ete RETIREE de `K_xpoint` dans `src/`. Ce test conserve
    la mesure qui a justifie ce retrait ; c'est
    `test_le_vrai_mappeur_marque_le_point_X` qui verifie le code actuel.
    """
    N = N_ENTRAINEMENT
    dx, Bx, By = _champ(N, 2 * L / N, "X")
    i0 = N // 2
    mic = _mic(_signal(Bx, By, dx), dx)
    K = _porte(Bx, By, dx) * mic

    assert mic[i0, i0] > 1e-2, (
        f"le seuil ne tire pas au point X ({mic[i0, i0]:.3e}) — ce test "
        f"n'isolerait pas l'effet de la porte")
    assert K[i0, i0] == pytest.approx(0.0, abs=1e-12), (
        f"|K_xpoint| au point X = {K[i0, i0]:.3e}, attendu 0 (porte)")
    assert np.abs(K).max() > 0.1, (
        f"|K_xpoint| est nul partout ({np.abs(K).max():.3e}) : le "
        f"coefficient ne marque meme pas l'anneau")


def test_sans_la_porte_le_coefficient_marque_le_point_X():
    """La forme candidate : meme signal, meme seuil, **sans la porte**.

    Elle rend le coefficient maximal AU point X, et reste nulle sur
    l'O-point comme sur le cisaillement pur — les deux controles qui
    portent du courant. C'est la mesure qui justifie de retirer la porte.
    """
    N = N_ENTRAINEMENT
    i0 = N // 2
    valeurs = {}
    for genre in ("X", "O", "cisaillement"):
        dx, Bx, By = _champ(N, 2 * L / N, genre)
        valeurs[genre] = _mic(_signal(Bx, By, dx), dx)[i0, i0]

    assert valeurs["X"] > 0.1, (
        f"sans porte, le point X rend {valeurs['X']:.3e} — trop faible pour "
        f"peser dans l'hamiltonien")
    assert valeurs["O"] == pytest.approx(0.0, abs=1e-12), (
        f"sans porte, l'O-point tire : {valeurs['O']:.3e}")
    assert valeurs["cisaillement"] == pytest.approx(0.0, abs=1e-12), (
        f"sans porte, le cisaillement pur tire : {valeurs['cisaillement']:.3e}")
