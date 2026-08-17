"""D-107 — `fig5_qaoa_detailed_analysis.py` : la profondeur 0 de sa courbe
de coefficients empruntait un `dx` 128 fois plus petit que toutes les
autres, et rendait un Hamiltonien identiquement nul.

`analyze_vqa_at_patch` calcule les coefficients d'un patch avec
`HamiltMapper.compute_coefficients(..., dx_override=dx_eff)`. Pour un patch
(`bounds is not None`) il passait la taille de cellule VQA,
`dx_eff = (patch_size / N) * L / target_dim` — la meme formule que
`refinement._run_level` (`dx_eff = patch_phys_size / target_dim`). Pour le
domaine complet (`bounds is None`, la profondeur 0 de la courbe) il passait
`None`, donc `grid.dx = L / N` : le pas de la grille FINE.

A N=256 et `target_dim=2`, c'est un facteur **128** : 0,024544 contre
3,141593. `compute_coefficients` en tire `Re_cell = |v| * dx / nu` ; a
`dx = grid.dx` aucun `Re_cell` n'atteint `RE_CRIT`, donc aucune anomalie
n'est detectee et les trois blocs sortent a zero.

Mesure (`init_harris_tearing`, N=256, 300 pas, `target_dim=2`,
`threshold_amr` du depot, `advanced_anomalies_enabled=True`) :

| profondeur | dx | somme|H_edges| | somme|C_edges| | somme|K_plaquettes| |
|---|---|---|---|---|
| 0 **avant** | 0,024544 | **0,000000** | **0,000e+00** | **0,000e+00** |
| 0 **apres** | 3,141593 | **0,238940** | **1,6579e+05** | **8,3461e+04** |
| 1 | 1,570796 | 0,038764 | 2,1036e+04 | 1,4071e+04 |
| 2 | 0,785398 | 0,003722 | 4,4755e+03 | 3,1925e+03 |
| 3 | 0,392699 | 0,001872 | 5,0236e+02 | 1,4181e+03 |

La lecture s'inverse : la courbe passe d'« un zero a la profondeur 0 puis
une bosse » a une **decroissance monotone**, ou la profondeur 0 porte les
coefficients les PLUS forts. La docstring du module attribuait ce zero a la
physique (« deeper patches where the effective dx is small enough to trigger
physical thresholds ») — `dx_eff` DECROIT avec la profondeur, donc les
patchs profonds declenchent MOINS, pas plus. C'est le recit qui cachait le
defaut.

Aucun nombre publie ne bouge : aucune figure `results/figures/fig5_*` n'est
committee dans ce depot. La sortie de `fig5` elle-meme change (son panneau
de coefficients et sa correction QAOA a la profondeur 0), ce qui est
l'objet de la correction.

Les tests portent sur le comportement du fichier committe (fonction
extraite par AST puis executee sur des doubles), pas sur son texte source.
"""
import ast
import os
import sys

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_FIG5 = os.path.join(_REPO_ROOT, "figures", "v1_legacy",
                     "fig5_qaoa_detailed_analysis.py")
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

N = 256
TARGET_DIM = 2
L = 2.0 * np.pi


class _DxRecorded(Exception):
    """Sort de `analyze_vqa_at_patch` des que `dx_override` est connu."""

    def __init__(self, dx):
        self.dx = dx
        super().__init__(str(dx))


class _Grid:
    def __init__(self):
        self.L = L
        self.dx = L / N
        self.resolution_N = N


class _Sim:
    def __init__(self):
        self.grid = _Grid()

    def get_fluxes(self):
        z = np.zeros((N, N))
        return {'vx': z.copy(), 'vy': z.copy(), 'Bx': z.copy(), 'By': z.copy()}


class _Mapper:
    """Double de `PhysicalMapper` : enregistre le `dx_override` recu."""

    def __init__(self, **kwargs):
        pass

    def compute_coefficients(self, sim, score, state, threshold, **kwargs):
        raise _DxRecorded(kwargs.get('dx_override'))


class _AngleMapper:
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def classical_score(state):
        return np.zeros((N, N))

    def compute_stress_flux(self, state):
        return {'phi_horizontal': np.zeros((N, N)),
                'phi_vertical': np.zeros((N, N))}


def _load_analyze():
    """Extrait `analyze_vqa_at_patch` du fichier committe.

    fig5 lance l'analyse a l'import : on ne peut pas l'importer. On compile
    la seule definition de fonction et on lui donne des doubles pour ses
    globales, de sorte que le `dx_override` reellement passe soit observe.
    """
    with open(_FIG5, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=_FIG5)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef)
               and n.name == "analyze_vqa_at_patch"), None)
    if fn is None:                                        # pragma: no cover
        pytest.fail("fig5 n'expose plus `analyze_vqa_at_patch`")
    module_ast = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module_ast)
    g = {
        "np": np,
        "N": N,
        "AngleMapper": _AngleMapper,
        "PhysicalMapper": _Mapper,
        "_hamilt_mapper_kwargs": lambda grid: {},
        "_safe_energy": lambda v: 0.0,
    }
    exec(compile(module_ast, _FIG5, "exec"), g)           # noqa: S102
    return g["analyze_vqa_at_patch"]


def _dx_for(bounds):
    analyze = _load_analyze()
    try:
        analyze(_Sim(), N, None, 0.3, bounds=bounds, target_dim=TARGET_DIM)
    except _DxRecorded as rec:
        return rec.dx
    raise AssertionError("compute_coefficients n'a pas ete appele")


def _dx_refinement(patch_span):
    """La convention de `refinement._run_level`, ecrite ici comme reference.

    `patch_phys_size = (y_e - y_s) / N * L` puis `/ target_dim`.
    """
    return (patch_span / N) * L / TARGET_DIM


# ══════════════════════════════════════════════════════════════════
#  1. Les deux chemins qui devaient coincider
# ══════════════════════════════════════════════════════════════════

def test_la_profondeur_zero_prend_la_cellule_vqa_pas_le_pas_fin():
    """Le coeur de D-107. Avant : `None` -> `grid.dx` = L/256."""
    dx0 = _dx_for(None)
    assert dx0 is not None, (
        "la profondeur 0 repasse `dx_override=None` : `compute_coefficients` "
        "retombe sur `grid.dx`, le pas de la grille fine (D-107)")
    assert dx0 == pytest.approx(L / TARGET_DIM), (
        f"dx de la profondeur 0 = {dx0}, attendu L/target_dim = {L/TARGET_DIM}")
    # l'ecart avec l'ancienne valeur, chiffre
    assert dx0 / (L / N) == pytest.approx(N / TARGET_DIM)   # 128
    assert dx0 / (L / N) == pytest.approx(128.0)


def test_toutes_les_profondeurs_suivent_la_convention_de_refinement():
    """Question 4 : fig5 et `refinement._run_level` doivent coincider."""
    cas = [(None, N), ((0, 128, 0, 128), 128),
           ((0, 64, 0, 64), 64), ((0, 32, 0, 32), 32)]
    for bounds, span in cas:
        assert _dx_for(bounds) == pytest.approx(_dx_refinement(span)), (
            f"bounds={bounds} : fig5 s'ecarte de la convention de refinement.py")


def test_le_dx_decroit_avec_la_profondeur():
    """La docstring disait l'inverse : les patchs profonds declencheraient
    PLUS les seuils. `Re_cell = |v|*dx/nu` croit avec `dx`, et `dx` decroit."""
    dxs = [_dx_for(None), _dx_for((0, 128, 0, 128)),
           _dx_for((0, 64, 0, 64)), _dx_for((0, 32, 0, 32))]
    assert dxs == sorted(dxs, reverse=True), dxs
    assert dxs == pytest.approx([3.14159265, 1.57079633, 0.78539816, 0.39269908])


# ══════════════════════════════════════════════════════════════════
#  2. L'ancien comportement, epingle par sa mesure
# ══════════════════════════════════════════════════════════════════

def test_la_mesure_avant_apres_est_ecrite():
    """`init_harris_tearing`, N=256, 300 pas. Les nombres sont ici pour
    qu'une derive se voie ; ils ne sont pas rejoues (un run coute ~10 min).

    Le point qui compte : a la profondeur 0 les trois blocs valaient
    EXACTEMENT zero, et valent maintenant les plus grandes valeurs de la
    courbe — la lecture « les coefficients apparaissent avec la
    profondeur » s'inverse en une decroissance monotone.
    """
    avant = {'dx': 0.024544, 'H': 0.0, 'C': 0.0, 'K': 0.0}
    apres = {'dx': 3.141593, 'H': 0.238940, 'C': 1.65790e5, 'K': 8.34611e4}
    depth1 = {'dx': 1.570796, 'H': 0.038764, 'C': 2.103649e4, 'K': 1.407063e4}
    depth2 = {'dx': 0.785398, 'H': 0.003722, 'C': 4.475491e3, 'K': 3.192463e3}
    depth3 = {'dx': 0.392699, 'H': 0.001872, 'C': 5.023601e2, 'K': 1.418127e3}

    assert avant['H'] == avant['C'] == avant['K'] == 0.0
    assert apres['dx'] / avant['dx'] == pytest.approx(128.0, rel=1e-4)
    for bloc in ('H', 'C', 'K'):
        serie = [apres[bloc], depth1[bloc], depth2[bloc], depth3[bloc]]
        assert serie == sorted(serie, reverse=True), (
            f"{bloc} : la serie corrigee n'est plus monotone decroissante "
            f"— remesurer, ne pas retoucher les nombres. {serie}")
        assert apres[bloc] > depth1[bloc], bloc


def test_la_docstring_du_module_ne_reattribue_plus_le_zero_a_la_physique():
    """La deviation est ecrite la ou elle vit (`VIGIL.md`).

    Ce test porte sur le contenu du contrat, pas sur sa mise en forme : il
    exige que le numero de defaut soit cite dans la docstring du module,
    pour que la phrase corrigee ne redevienne pas la phrase d'origine sans
    que personne ne le remarque.
    """
    with open(_FIG5, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=_FIG5)
    doc = ast.get_docstring(tree) or ""
    assert "D-107" in doc, (
        "la docstring du module ne porte plus la trace de D-107")
