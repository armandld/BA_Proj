"""Le rééchantillonnage AMR : restriction, prolongation, détection de bord.

Ces fonctions déplacent les champs entre niveaux de raffinement. Une erreur
n'y produit pas de plantage : elle produit un champ décalé, lissé ou tronqué,
que la suite du pipeline traite comme s'il était juste.

Trois promesses écrites dans les docstrings sont vérifiées ici :

  - `_downsample_local` : « Restriction conservative (block-average) —
    préserve l'intégrale » ;
  - `_upsample_global` : « respecte la topologie torique » — donc un champ
    périodique doit rester périodique, sans couture au raccord ;
  - restriction ∘ prolongation doit être l'identité sur les champs que la
    restriction peut représenter.
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from Simulation.grid import PeriodicGrid          # noqa: E402
from Simulation.refinement import (               # noqa: E402
    _boundary_activation, _downsample_fields,
)
from Simulation.solver import MHDSolver           # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
#  1. Restriction — la promesse de conservation
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("factor", [2, 4, 8])
def test_downsampling_preserves_the_mean_exactly(factor):
    """« Préserve l'intégrale » : la moyenne globale ne doit pas bouger."""
    rng = np.random.default_rng(0)
    f = rng.standard_normal((64, 64))
    coarse = MHDSolver._downsample_local(f, factor)
    assert coarse.shape == (64 // factor, 64 // factor)
    assert coarse.mean() == pytest.approx(f.mean(), rel=1e-12)


def test_downsampling_by_one_is_the_identity():
    rng = np.random.default_rng(1)
    f = rng.standard_normal((8, 8))
    assert MHDSolver._downsample_local(f, 1) is f


def test_downsampling_is_a_block_average_cell_by_cell():
    """Valeur connue : chaque cellule grossiere est la moyenne de son bloc."""
    f = np.arange(16, dtype=float).reshape(4, 4)
    coarse = MHDSolver._downsample_local(f, 2)
    assert coarse[0, 0] == pytest.approx(np.mean([0, 1, 4, 5]))
    assert coarse[1, 1] == pytest.approx(np.mean([10, 11, 14, 15]))


def test_downsampling_a_uniform_field_is_exact():
    f = np.full((32, 32), -2.5)
    np.testing.assert_allclose(MHDSolver._downsample_local(f, 4),
                               np.full((8, 8), -2.5), rtol=0, atol=1e-15)


def test_downsampling_dilutes_a_spike_as_area_averaging_must():
    """Contraste avec le max-abs de RescaleArrays : ici le pic DOIT se
    diluer, parce que ces champs sont physiques et non des indicateurs."""
    f = np.zeros((16, 16))
    f[3, 3] = 1.0
    assert MHDSolver._downsample_local(f, 4).max() == pytest.approx(1.0 / 16)


# ═══════════════════════════════════════════════════════════════════════
#  2. Prolongation
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("factor", [2, 4])
def test_upsampling_a_uniform_field_stays_uniform(factor):
    """Le test le plus simple, et celui qui attrape les bords casses."""
    for fn in (MHDSolver._upsample_local, MHDSolver._upsample_global):
        out = fn(np.full((8, 8), 3.0), factor)
        assert out.shape == (8 * factor, 8 * factor)
        np.testing.assert_allclose(out, 3.0, rtol=0, atol=1e-9)


def test_upsampling_by_one_is_the_identity():
    rng = np.random.default_rng(2)
    f = rng.standard_normal((8, 8))
    assert MHDSolver._upsample_local(f, 1) is f
    assert MHDSolver._upsample_global(f, 1) is f


def test_global_upsampling_has_no_seam_at_the_periodic_join():
    """`mode='wrap'` doit rendre le raccord invisible.

    Sur un champ periodique lisse, l'erreur au bord ne doit pas depasser
    l'erreur a l'interieur. Une prolongation non periodique produit
    typiquement un pic d'erreur sur la premiere et la derniere ligne.
    """
    Nc, factor = 32, 4
    g_c = PeriodicGrid(Nc)
    coarse = np.sin(g_c.X) * np.cos(g_c.Y)
    fine = MHDSolver._upsample_global(coarse, factor)

    g_f = PeriodicGrid(Nc * factor)
    exact = np.sin(g_f.X) * np.cos(g_f.Y)
    err = np.abs(fine - exact)

    edge = max(err[0, :].max(), err[-1, :].max(),
               err[:, 0].max(), err[:, -1].max())
    interior = err[2:-2, 2:-2].max()
    assert edge <= 3.0 * interior + 1e-12, (
        f"couture au raccord periodique : erreur bord {edge:.3e} contre "
        f"interieur {interior:.3e}")


def test_global_upsampling_reaches_cubic_accuracy():
    """D-2 corrige : la prolongation atteint enfin la precision de l'ordre 3.

    Deux conventions ont ete alignees sur celles de `PeriodicGrid` :
    echantillonnage aux NOEUDS (`j / factor`, et non `(j+0.5)/factor - 0.5`)
    et `mode='grid-wrap'`, le seul enroulement reellement periodique depuis
    scipy 1.6.

    Mesure sur sin(x)cos(y), 32 -> 128 : 7.74e-6, contre 2.49e-1 avant.
    """
    Nc, factor = 32, 4
    g_c = PeriodicGrid(Nc)
    g_f = PeriodicGrid(Nc * factor)
    coarse = np.sin(g_c.X) * np.cos(g_c.Y)
    exact = np.sin(g_f.X) * np.cos(g_f.Y)
    err = float(np.max(np.abs(
        MHDSolver._upsample_global(coarse, factor) - exact)))
    assert err < 1e-4, f"erreur {err:.3e}, attendu < 1e-4"


def test_global_upsampling_converges_at_third_order_or_better():
    """L'ordre annonce par `order=3` doit se voir dans les chiffres."""
    errs = []
    for Nc in (16, 32, 64):
        g_c = PeriodicGrid(Nc)
        coarse = np.sin(g_c.X) * np.cos(g_c.Y)
        fine = MHDSolver._upsample_global(coarse, 2)
        g_f = PeriodicGrid(2 * Nc)
        errs.append(float(np.max(np.abs(
            fine - np.sin(g_f.X) * np.cos(g_f.Y)))))
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    assert all(o > 2.5 for o in orders), (
        f"ordre {orders} (erreurs {errs}), attendu > 2.5")


def test_the_global_prolongation_no_longer_shifts_the_field():
    """Le decalage de -0.375 cellule grossiere a disparu.

    Un decalage residuel se lit comme une correlation entre l'erreur et la
    DERIVEE du champ.
    """
    Nc, factor = 32, 4
    g_c = PeriodicGrid(Nc)
    g_f = PeriodicGrid(Nc * factor)
    fine = MHDSolver._upsample_global(np.sin(g_c.X), factor)
    bias = float(np.mean((fine - np.sin(g_f.X)) * np.cos(g_f.X)))
    assert abs(bias) < 1e-6, f"biais de decalage residuel {bias:.3e}"


def test_restriction_after_prolongation_recovers_a_uniform_field():
    """R o P = identite sur ce que la grille grossiere peut representer."""
    coarse = np.full((8, 8), 1.25)
    back = MHDSolver._downsample_local(
        MHDSolver._upsample_global(coarse, 4), 4)
    np.testing.assert_allclose(back, coarse, rtol=0, atol=1e-9)


def test_restriction_and_prolongation_are_not_adjoint_and_that_is_expected():
    """R o P n'est PAS l'identite sur un champ lisse — et ne peut pas l'etre.

    Les deux operateurs ne parlent pas de la meme chose :

      - `_downsample_local` fait une MOYENNE DE BLOC, semantique
        volume-fini (valeur moyenne sur la cellule) ;
      - `_upsample_global` interpole aux NOEUDS, semantique valeur
        ponctuelle — celle du solveur, dont les champs sont des valeurs
        aux points `PeriodicGrid.X`.

    Moyenner les 4 points fins qui suivent un noeud grossier ne redonne pas
    la valeur en ce noeud : cela la decale d'environ 3h/8 et la lisse.
    L'ecart mesure est de 0.149 sur un champ d'amplitude 1 a facteur 4.

    L'ancienne prolongation, elle, echantillonnait au centre des cellules,
    donc APPARIEE a la moyenne de bloc : son aller-retour revenait mieux.
    Mais elle etait desaccordee avec la grille du solveur, et la mesure
    tranche en faveur de la convention noeud — voir
    `test_the_corrected_prolongation_tracks_the_reference_better`.
    """
    g = PeriodicGrid(16)
    coarse = np.sin(g.X) * np.cos(g.Y)
    back = MHDSolver._downsample_local(
        MHDSolver._upsample_global(coarse, 4), 4)
    err = float(np.max(np.abs(back - coarse)))
    assert 0.05 < err < 0.5, (
        f"aller-retour a {err:.3e} : si les deux operateurs ont ete "
        "reaccordes, mettre a jour ce test")


def test_the_corrected_prolongation_tracks_the_reference_better():
    """La mesure qui justifie D-2 : le chemin AMR suit mieux `step_full`.

    C'est le seul juge valable — l'aller-retour R o P favorise la
    convention centre-cellule, mais ce n'est pas ce que le solveur demande.
    Ecart relatif a `step_full` apres 15 pas, N=64, max_depth=2 :

        couverture      ancienne prolongation   corrigee
        aucun patch     1.1896 %                1.1465 %
        un quart        1.3879 %                1.1197 %

    Le residu ~1.1 % est l'erreur intrinseque de la resolution grossiere,
    que la prolongation ne peut pas retirer.
    """
    from scipy.ndimage import map_coordinates

    def old_upsample(field, factor):
        if factor == 1:
            return field
        Nc = field.shape[0]
        pos = (np.arange(Nc * factor) + 0.5) / factor - 0.5
        A, B = np.meshgrid(pos, pos, indexing="ij")
        return map_coordinates(field, [A, B], order=3, mode="wrap")

    N = 64
    patches = [{"bounds": (0, N // 2, 0, N // 2), "depth": 2}]

    def run(kind):
        g = PeriodicGrid(N)
        s = MHDSolver(g, dt=1e-3, Re=400, Rm=400)
        s.init_orszag_tang()
        for _ in range(20):
            s.adapt_dt(cfl_target=0.4)
            s.step_full(record_stats=False)
        if kind == "old":
            s._upsample_global = old_upsample
        s.dt = 1e-3
        for _ in range(15):
            if kind == "full":
                s.step_full(record_stats=False)
            else:
                s.step_layered(active_patches=patches, max_depth=2,
                               target_dim=2)
        return np.array([s.vx, s.vy, s.Bx, s.By])

    ref = run("full")
    amp = np.max(np.abs(ref))
    err_old = float(np.max(np.abs(run("old") - ref)) / amp)
    err_new = float(np.max(np.abs(run("new") - ref)) / amp)
    assert err_new < err_old, (
        f"la prolongation corrigee ({err_new:.4%}) ne fait pas mieux que "
        f"l'ancienne ({err_old:.4%})")


# ═══════════════════════════════════════════════════════════════════════
#  3. Restriction des champs physiques (refinement)
# ═══════════════════════════════════════════════════════════════════════

def _fields(N, value=None):
    rng = np.random.default_rng(3)
    keys = ('vx', 'vy', 'Bx', 'By', 'Jz')
    if value is not None:
        return {k: np.full((N, N), value) for k in keys}
    return {k: rng.standard_normal((N, N)) for k in keys}


def test_field_downsampling_returns_every_key_at_the_target_size():
    out = _downsample_fields(_fields(64), 0, 64, 0, 64, 8)
    assert set(out) == {'vx', 'vy', 'Bx', 'By', 'Jz'}
    for k, v in out.items():
        assert v.shape == (8, 8), f"{k} a la forme {v.shape}"


def test_field_downsampling_preserves_the_mean_of_each_field():
    f = _fields(64)
    out = _downsample_fields(f, 0, 64, 0, 64, 8)
    for k in out:
        assert out[k].mean() == pytest.approx(f[k].mean(), rel=1e-12), (
            f"{k} : la moyenne n'est pas preservee")


def test_field_downsampling_is_exact_on_uniform_fields():
    out = _downsample_fields(_fields(32, value=2.5), 0, 32, 0, 32, 4)
    for v in out.values():
        np.testing.assert_allclose(v, 2.5, rtol=0, atol=1e-15)


def test_field_downsampling_with_halo_enlarges_the_output():
    """pad=1 doit donner target_dim + 2 de cote, halo compris."""
    #  `pad` compte en cellules FINES de chaque cote, et la sortie vaut
    #  target_dim + 2*pad — pas target_dim + 2.
    out = _downsample_fields(_fields(64), 8, 40, 8, 40, 4, pad=1)
    for v in out.values():
        assert v.shape == (6, 6)
    out4 = _downsample_fields(_fields(64), 8, 40, 8, 40, 4, pad=4)
    for v in out4.values():
        assert v.shape == (12, 12)


def test_field_downsampling_wraps_around_the_torus():
    """Un patch a cheval sur le bord doit lire de l'autre cote.

    On marque le coin oppose : s'il ressort, l'enroulement fonctionne.
    """
    f = _fields(16, value=0.0)
    f['vx'][15, 15] = 10.0
    out = _downsample_fields(f, 0, 4, 0, 4, 2, pad=1)
    assert out['vx'].max() > 0.0, (
        "le patch borde n'a pas lu la cellule (15, 15) par periodicite")


# ═══════════════════════════════════════════════════════════════════════
#  4. Détection d'anomalie au bord du patch
# ═══════════════════════════════════════════════════════════════════════

def test_boundary_activation_is_empty_below_two_cells():
    assert _boundary_activation(np.ones((1, 1)), 1) == {}


def test_boundary_activation_stays_silent_on_a_quiet_patch():
    assert _boundary_activation(np.zeros((4, 4)), 4) == {}


def test_boundary_activation_names_the_right_edge():
    """Controle directionnel : chaque bord doit etre reconnu pour lui-meme.

    Une confusion haut/bas ou gauche/droite ferait etendre le raffinement
    du mauvais cote, sans rien casser.
    """
    for edge, sl in (('top', (0, slice(None))),
                     ('bottom', (-1, slice(None))),
                     ('left', (slice(None), 0)),
                     ('right', (slice(None), -1))):
        p = np.zeros((5, 5))
        p[sl] = 1.0
        assert _boundary_activation(p, 5) == {edge: True}, (
            f"bord {edge} mal identifie")


def test_boundary_activation_ignores_a_purely_interior_anomaly():
    """Une anomalie au centre ne doit declencher aucun drapeau."""
    p = np.zeros((5, 5))
    p[2, 2] = 1.0
    assert _boundary_activation(p, 5) == {}


def test_boundary_activation_needs_a_contrast_not_just_a_level():
    """Un patch uniformement actif n'a pas d'anomalie AU BORD.

    Le seuil vaut max(moyenne interieure + 0.1, 0.3) : un patch entierement
    a 1.0 ne doit rien declencher, sinon tout patch actif demanderait une
    extension.
    """
    assert _boundary_activation(np.ones((5, 5)), 5) == {}


def test_boundary_activation_can_flag_several_edges_at_once():
    p = np.zeros((5, 5))
    p[0, :] = 1.0
    p[:, 0] = 1.0
    assert set(_boundary_activation(p, 5)) == {'top', 'left'}


# ═══════════════════════════════════════════════════════════════════════
#  5. Ce que le défaut de prolongation coûte au chemin AMR
# ═══════════════════════════════════════════════════════════════════════

def _fixed_upsample(field, factor):
    """`_upsample_global` avec les deux conventions corrigees."""
    from scipy.ndimage import map_coordinates
    if factor == 1:
        return field
    Nc = field.shape[0]
    idx = np.arange(Nc * factor) / factor          # noeuds, pas centres
    A, B = np.meshgrid(idx, idx, indexing="ij")
    return map_coordinates(field, [A, B], order=3, mode="grid-wrap")


def _march(patched, patches, depth, N=64, nsteps=15):
    g = PeriodicGrid(N)
    s = MHDSolver(g, dt=1e-3, Re=400, Rm=400)
    s.init_orszag_tang()
    for _ in range(20):
        s.adapt_dt(cfl_target=0.4)
        s.step_full(record_stats=False)
    if patched:
        s._upsample_global = _fixed_upsample
    s.dt = 1e-3
    for _ in range(nsteps):
        s.step_layered(active_patches=patches, max_depth=depth, target_dim=2)
    return np.array([s.vx, s.vy, s.Bx, s.By])


def test_layered_with_no_patches_equals_step_full():
    """La garantie annoncee ligne 561 : a max_depth=0, resultat IDENTIQUE.

    Mesure : ecart exactement 0.0. C'est une garantie forte, et elle tient.
    """
    N = 64
    def run(layered):
        g = PeriodicGrid(N)
        s = MHDSolver(g, dt=1e-3, Re=400, Rm=400)
        s.init_orszag_tang()
        for _ in range(20):
            s.adapt_dt(cfl_target=0.4)
            s.step_full(record_stats=False)
        s.dt = 1e-3
        if layered:
            s.step_layered(active_patches=[], max_depth=0, target_dim=2)
        else:
            s.step_full(record_stats=False)
        return np.array([s.vx, s.vy, s.Bx, s.By])
    assert np.max(np.abs(run(False) - run(True))) == 0.0


def test_the_tau_correction_cancels_the_prolongation_error_under_patches():
    """Sous un patch actif, l'erreur de prolongation s'annule exactement.

    Phase 1 ajoute le delta grossier prolonge ; Phase 2 retranche ce meme
    delta et lui substitue le delta fin. La prolongation disparait donc de
    la difference — c'est le principe de la correction tau, et il tient a
    l'arrondi machine.
    """
    N = 64
    full = [{"bounds": (0, N, 0, N), "depth": 2}]
    err = float(np.max(np.abs(_march(False, full, 2) - _march(True, full, 2))))
    assert err < 1e-12, (
        f"la correction tau ne compense plus la prolongation : {err:.3e}")


@pytest.mark.parametrize("patches,label", [([], "aucun patch"), (None, "quart")])
def test_the_prolongation_no_longer_biases_the_unrefined_background(patches, label):
    """D-2 corrige : le fond non raffine ne porte plus l'erreur.

    Avant correction, l'ecart entre le code et la variante aux bonnes
    conventions valait 1.79 % sans patch et 1.67 % au quart de couverture,
    apres 15 pas. Il doit maintenant etre negligeable, puisque le code EST
    la variante corrigee.
    """
    N = 64
    if patches is None:
        patches = [{"bounds": (0, N // 2, 0, N // 2), "depth": 2}]
    a = _march(False, patches, 2, N=N)
    b = _march(True, patches, 2, N=N)
    rel = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
    assert rel < 1e-12, (
        f"{label} : ecart relatif {rel:.3e} — `_upsample_global` ne coincide "
        "plus avec la reference corrigee")


def test_the_amr_background_now_tracks_the_reference_solver():
    """Consequence physique de D-2 : le fond grossier suit mieux step_full.

    Sans patch actif, `step_layered` applique la correction grossiere
    prolongee sur tout le domaine. Elle ne peut pas egaler `step_full` — la
    resolution grossiere perd de l'information — mais l'ecart ne doit plus
    porter la composante de prolongation.
    """
    N = 64
    def run(layered):
        g = PeriodicGrid(N)
        s = MHDSolver(g, dt=1e-3, Re=400, Rm=400)
        s.init_orszag_tang()
        for _ in range(20):
            s.adapt_dt(cfl_target=0.4)
            s.step_full(record_stats=False)
        s.dt = 1e-3
        for _ in range(10):
            if layered:
                s.step_layered(active_patches=[], max_depth=2, target_dim=2)
            else:
                s.step_full(record_stats=False)
        return np.array([s.vx, s.vy, s.Bx, s.By])
    gap = float(np.max(np.abs(run(True) - run(False))))
    assert np.isfinite(gap)
    assert gap < 0.5, f"le fond AMR diverge de la reference : {gap:.3e}"
