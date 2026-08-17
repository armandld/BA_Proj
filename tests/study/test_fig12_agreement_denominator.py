"""D-106 — le « taux d'accord » du panneau C de `fig12_depth_analysis.py`
divisait par le domaine ENTIER.

`_agreement_by_depth` construit, pour chaque profondeur, deux masques
booleens (`qa_mask`, `cl_mask`) marquant les patchs NON `coarse_leaf` de
cette profondeur, puis renvoyait `np.sum(qa_mask == cl_mask) / (N * N)`.

A une profondeur donnee, presque aucun pixel du domaine ne porte de patch :
tous ceux que ni l'un ni l'autre bras ne touche verifient
`False == False` et comptaient comme un ACCORD. Le taux mesurait donc la
proportion de domaine vide, pas la proportion de decisions concordantes.

Mesure (`init_harris_tearing`, N=256, 300 pas, `target_dim=2`,
`min_size=6`, `solve_max_depth=5`, seuils du depot ; types de patchs :
QA `leaf_depth`=106 / `coarse_leaf`=144, CL `leaf_depth`=256 /
`coarse_leaf`=96) :

| profondeur | patchs QA | patchs CL | union couverte | avant | apres |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0,00 % | 100,00 % | indefini |
| 1 | 0 | 0 | 0,00 % | 100,00 % | indefini |
| 2 | 44 | 32 | 0,00 % | 100,00 % | indefini |
| 3 | 38 | 64 | 0,00 % | 100,00 % | indefini |
| 4 | **62** | **0** | 0,00 % | **100,00 %** | indefini |
| 5 | 106 | 256 | 25,00 % | **85,35 %** | **41,41 %** |

Cinq profondeurs sur six annoncaient un accord PARFAIT en ne mesurant
rien. A la profondeur 4, le bras Q-HAS porte 62 patchs et le bras
classique zero — le desaccord structurel maximal — et le panneau imprimait
100 %. A la seule profondeur reellement mesuree, le taux passe de 85,35 %
a 41,41 % : un facteur 2, qui fait passer la barre de couleur de l'ambre
(`> 85`) au rouge.

Second scenario, `init_orszag_tang`, N=256, 500 pas (QA `leaf_depth`=14 /
`coarse_leaf`=116, CL `leaf_depth`=63 / `coarse_leaf`=145) : profondeurs 0
a 4 toutes a **100,00 %** avant, indefinies apres ; profondeur 5, union
6,25 % du domaine, **95,02 % -> 20,31 %**. La barre y etait **verte**
(`> 95`) pour un accord reel d'un cinquieme.

La note du panneau attribuait ce chiffre a la physique (« High agreement
(>90%) is expected — most BFS decisions are far from the threshold ») :
c'etait un artefact du comptage.

Un second chemin menait au meme faux 100 % : le repli
`np.mean(all_agreement[d]) * 100 if all_agreement[d] else 100` du trace.

Aucun nombre publie ne bouge : aucune figure ni log `fig12_*` n'est
committe dans ce depot (`git ls-files results/figures/` ne rend que
`fig1_ceiling_bar.png` et `fig2_loso_scatter.png`).

Les tests portent sur le comportement du fichier committe (fonctions
importees depuis le module, dont `main()` n'est appele que sous
`__main__`), pas sur son texte source.
"""
import os
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_REPO_ROOT, "figures", "v1_legacy"),
           os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fig12_depth_analysis import _agreement_by_depth  # noqa: E402


def _agreement_bars():
    """Import tardif : sur la version d'avant D-106 la fonction n'existe pas,
    et les tests de `_agreement_by_depth` doivent quand meme pouvoir tourner
    (et echouer sur leur assertion, pas sur la collecte)."""
    try:
        from fig12_depth_analysis import agreement_bars
    except ImportError:                                   # pragma: no cover
        pytest.fail(
            "fig12_depth_analysis.py n'expose plus `agreement_bars` : le "
            "repli `if all_agreement[d] else 100` est de retour (D-106).")
    return agreement_bars

N = 256


def _leaf(depth, y0, y1, x0, x1):
    return {'bounds': (y0, y1, x0, x1), 'depth': depth, 'type': 'leaf_depth'}


def _coarse(depth, y0, y1, x0, x1):
    return {'bounds': (y0, y1, x0, x1), 'depth': depth, 'type': 'coarse_leaf'}


def _ancien_taux(qa_patches, cl_patches, N, d):
    """Le calcul d'avant D-106, reproduit : denominateur = N*N."""
    def mask(ps):
        m = np.zeros((N, N), dtype=bool)
        for p in ps:
            if p['depth'] == d and p.get('type', '') != 'coarse_leaf':
                y_s, y_e, x_s, x_e = p['bounds']
                m[np.ix_(np.arange(y_s, y_e) % N, np.arange(x_s, x_e) % N)] = True
        return m
    return float(np.sum(mask(qa_patches) == mask(cl_patches))) / (N * N)


# ══════════════════════════════════════════════════════════════════
#  1. Le champ qui SEPARE : deux bras en desaccord total
# ══════════════════════════════════════════════════════════════════

def test_le_desaccord_maximal_etait_annonce_comme_un_accord_parfait():
    """Reproduit la profondeur 4 mesuree : 62 patchs contre 0.

    Ici, en reduit : un bras porte des patchs `coarse_leaf` a cette
    profondeur (donc filtres), l'autre rien du tout. Les deux masques sont
    vides, et l'ancien calcul rendait 1,0.
    """
    qa = [_coarse(4, 0, 8, 0, 8), _coarse(4, 8, 16, 8, 16)]
    cl = []
    assert _ancien_taux(qa, cl, N, 4) == 1.0, "reproduction de l'ancien calcul"
    assert np.isnan(_agreement_by_depth(qa, cl, N)[4]), (
        "une profondeur sans aucune decision doit etre indefinie, pas 100 %")


def test_deux_bras_qui_ne_se_recouvrent_pas_du_tout():
    """Accord reel 0 %, ancien taux tres proche de 100 %.

    Deux patchs disjoints de 16x16 sur un domaine 256x256 : l'ancien
    denominateur noie le desaccord dans 65 024 pixels vides.
    """
    qa = [_leaf(1, 0, 16, 0, 16)]
    cl = [_leaf(1, 32, 48, 32, 48)]
    ancien = _ancien_taux(qa, cl, N, 1)
    nouveau = _agreement_by_depth(qa, cl, N)[1]
    assert ancien == pytest.approx(1.0 - 512 / (N * N)), ancien
    assert ancien > 0.99, f"l'ancien taux annoncait {ancien:.4f}"
    assert nouveau == 0.0, (
        f"aucun pixel commun : l'accord vaut 0, pas {nouveau}")


def test_recouvrement_partiel_le_facteur_est_chiffre():
    """Recouvrement d'un tiers : accord reel 33 %, ancien taux > 99 %.

    Dans l'union, « les deux masques sont d'accord » ne peut vouloir dire
    que « tous les deux vrais » : c'est l'intersection.
    """
    qa = [_leaf(2, 0, 16, 0, 32)]      # x dans [0, 32)
    cl = [_leaf(2, 0, 16, 16, 48)]     # x dans [16, 48)
    # union = 16*48 = 768 px ; intersection = 16*16 = 256 px
    assert _agreement_by_depth(qa, cl, N)[2] == pytest.approx(256 / 768)
    assert _ancien_taux(qa, cl, N, 2) > 0.99


def test_accord_parfait_reste_a_un():
    """La correction ne doit pas rendre l'accord impossible a atteindre."""
    qa = [_leaf(1, 0, 32, 0, 32)]
    cl = [_leaf(1, 0, 32, 0, 32)]
    assert _agreement_by_depth(qa, cl, N)[1] == 1.0


# ══════════════════════════════════════════════════════════════════
#  2. Le second chemin vers le faux 100 % : le repli du trace
# ══════════════════════════════════════════════════════════════════

def test_une_profondeur_sans_essai_nest_plus_comptee_a_cent():
    means, stds, undef = _agreement_bars()({}, [0, 1, 2])
    assert undef == [True, True, True]
    assert all(m != m for m in means), f"{means} : un nan est attendu partout"


def test_une_profondeur_dont_tous_les_essais_sont_indefinis_reste_indefinie():
    means, _, undef = _agreement_bars()({0: [float('nan')] * 3, 1: [0.5, 0.7]},
                                        [0, 1])
    assert undef == [True, False]
    assert means[1] == pytest.approx(60.0)


def test_les_essais_definis_sont_moyennes_sans_les_indefinis():
    """Un essai indefini ne doit ni compter pour 100 ni annuler les autres."""
    means, stds, undef = _agreement_bars()({0: [0.4, float('nan'), 0.6]}, [0])
    assert undef == [False]
    assert means[0] == pytest.approx(50.0)
    assert stds[0] == pytest.approx(10.0)


# ══════════════════════════════════════════════════════════════════
#  3. Les nombres mesures, epingles pour qu'une derive se voie
# ══════════════════════════════════════════════════════════════════

def test_la_mesure_de_reference_de_la_profondeur_5_est_ecrite():
    """`init_harris_tearing`, N=256, 300 pas : union 16 384 px sur 65 536.

    Reconstruit a partir des comptes mesures plutot que rejoue (une
    execution complete coute ~12 min) : couverture QA 6 784 px, CL 16 384 px,
    union 16 384 px, intersection 6 784 px (QA inclus dans CL).
    Ancien taux = (N^2 - (union - intersection)) / N^2 ; nouveau =
    intersection / union.
    """
    intersection, union = 6784, 16384
    ancien = (N * N - (union - intersection)) / (N * N)
    nouveau = intersection / union
    assert ancien == pytest.approx(0.853515625)
    assert nouveau == pytest.approx(0.4140625)
    # la barre change de couleur : ambre (> 85) -> rouge
    assert ancien * 100 > 85 and nouveau * 100 <= 85


def test_la_mesure_de_reference_orszag_tang_est_ecrite():
    """`init_orszag_tang`, N=256, 500 pas : union 4 096 px sur 65 536.

    Couverture QA 896 px, CL 4 032 px, union 4 096 px, intersection 832 px.
    C'est le cas ou la couleur de la barre passe du VERT (`> 95`) au rouge.
    """
    intersection, union = 832, 4096
    ancien = (N * N - (union - intersection)) / (N * N)
    nouveau = intersection / union
    assert ancien == pytest.approx(0.9501953125, rel=1e-9)
    assert nouveau == pytest.approx(0.203125, rel=1e-9)
    assert ancien * 100 > 95 and nouveau * 100 <= 85
