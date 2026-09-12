"""D-119 — l'axe « optimiseur » traverse, du cote qui ne l'avait jamais ete.

`VIGIL_BA_Proj.md` demande que chaque axe soit emprunte des deux cotes. La
passe precedente a compte, sur la suite entiere : COBYLA **317 appels**,
Powell **1**, L-BFGS-B **1**, Nelder-Mead **1**. Ce fichier prend ce
desequilibre. Il rend deux faits, tous deux mesures, et **ne corrige rien** :
ce qu'il faudrait faire est une decision, pas une correction de chemin.

**1. `K_opt` n'achete pas le meme budget selon l'optimiseur.**
`execute` passe `options={'maxiter': K_opt}` aux trois methodes. Pour
COBYLA, scipy traduit `maxiter` en nombre d'EVALUATIONS de la fonction ;
pour Powell et L-BFGS-B, en nombre d'ITERATIONS, chacune valant plusieurs
evaluations (recherche lineaire, gradient par differences finies). Compte
des appels a l'estimateur, `K_opt = 20`, 6 tirages chacun, a `766d289` :

    COBYLA     20  20  20  20  20  20        exactement K_opt, 6 fois sur 6
    L-BFGS-B   50  60  85  95 115  90        x2,5 a x5,8
    Powell    187 377 328 176 251 265        x8,8 a x18,9

Les trois intervalles sont **disjoints** : la dispersion ne l'explique pas.
Un meme `K_opt` achete donc jusqu'a **dix-neuf fois** plus de circuits selon
la methode.

**2. Le `K_opt` gele a ete regle sous COBYLA ; le CLI deploie L-BFGS-B.**
`train_hyperparams.create_argus` — l'objectif que la campagne Optuna a
optimise, celle dont `results/hyperparams/best_hyperparams.json` est
l'artefact gele — code `method="COBYLA"` en dur. `src/pipeline.py` offre
`--method` avec pour defaut **`L-BFGS-B`**, et **aucun lanceur du depot ne
passe `--method`** : tout run de `pipeline.py` prend ce defaut. Les neuf
hyperparametres de `SEARCH_SPACE` ont donc ete selectionnes sous un
optimiseur variationnel, et sont deployes par defaut sous un autre — qui ne
consomme pas le meme budget au meme `K_opt`. Present depuis le premier
commit (`cf93ba3`) : le defaut vaut `L-BFGS-B` des l'origine.

**Ce que la mesure NE dit PAS — et c'est la moitie du resultat.**
La question naturelle est : les deux optimiseurs rendent-ils une decision
differente ? **A la dispersion du bras QAOA, cette mesure ne tranche pas.**
Patch `dim = 2` deploye, `K_opt = 30`, 6 tirages par methode :

    ecart des MOYENNES COBYLA vs L-BFGS-B, par qubit : max **0,0867**
    dispersion INTRA-methode (max - min par qubit)   : **0,200** (COBYLA)
                                                       **0,240** (L-BFGS-B)

L'ecart cherche est trois fois plus petit que le bruit d'execution de
chacune des deux references. `VIGIL.md` : « si les deux references different
de plus que l'effet cherche, la grandeur ne tranche rien : le dire, et ne
pas conclure. » C'est dit. Trancher demanderait une campagne, pas une passe
de relecture — et c'est precisement pourquoi rien n'est corrige ici.

Sur quelle entree ces tests echouent
------------------------------------
Le premier, si `maxiter` se met a signifier la meme chose pour les trois
methodes (ou si un `maxfev`/`maxfun` est ajoute). Le second, s'il se met a
PASSER — c'est le sens de `strict=True` : le jour ou l'entrainement et le
deploiement nomment le meme optimiseur, la dette est payee et la suite doit
le dire.
"""

import ast
import os
import sys
import warnings

import numpy as np
import pytest

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_ROOT = _repo_root()
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

#: Mesure a `766d289`, `K_opt = 20`, 6 tirages. Marges prises LARGES sous le
#: minimum mesure : ce test doit epingler l'ordre de grandeur, pas devenir un
#: seuil calibre sur le tirage du jour.
_K_OPT = 20
_MIN_RATIO = {"COBYLA": None,       # <= K_opt, garantie dure de scipy
              "L-BFGS-B": 2.0,      # mesure 2,5 a 5,8
              "Powell": 5.0}        # mesure 8,8 a 18,9


class _CountingEstimator:
    """Compte les appels a l'estimateur sans changer ce qu'il rend."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def run(self, pubs):
        self.calls += len(pubs)
        return self._inner.run(pubs)

    @property
    def options(self):
        return self._inner.options


def _count_evaluations(method, K_opt=_K_OPT):
    from VQA.execute import execute
    from VQA.runtime import VQARuntime

    op = SparsePauliOp.from_list([("ZZ", -5.0), ("ZI", -2.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qc = QAOAAnsatz(cost_operator=op, reps=2).decompose(reps=3)
        runtime = VQARuntime(backend_name="state_vector", mode="simulator",
                             shots=1024, opt_level=1)
        counter = _CountingEstimator(runtime._estimator)
        runtime._estimator = counter
        execute(qc, op, "simulator", "state_vector", 1024, 2, K_opt, 1e-3,
                1.0, False, vqa_runtime=runtime, method=method)
    return counter.calls


def test_cobyla_spends_exactly_the_budget_k_opt_names():
    """Pour COBYLA seul, `maxiter` EST le nombre d'evaluations."""
    n = _count_evaluations("COBYLA")
    assert n <= _K_OPT, (
        f"COBYLA a evalue {n} fois pour K_opt = {_K_OPT} : `maxiter` ne "
        "borne plus les evaluations, la mesure de D-119 est a refaire")
    assert n >= _K_OPT // 2, (
        f"COBYLA n'a evalue que {n} fois : il converge avant le budget, "
        "la comparaison ci-dessous ne mesure plus la meme chose")


@pytest.mark.parametrize("method", ["L-BFGS-B", "Powell"])
def test_the_other_optimisers_spend_a_multiple_of_that_budget(method):
    """`K_opt` n'achete pas le meme nombre de circuits selon la methode.

    C'est le fait de D-119 : un budget nomme une fois, honore de trois
    facons. Les seuils sont des ordres de grandeur pris tres en dessous du
    minimum mesure sur 6 tirages, pas des seuils du jour.
    """
    n = _count_evaluations(method)
    plancher = _MIN_RATIO[method] * _K_OPT
    assert n > plancher, (
        f"{method} a evalue {n} fois pour K_opt = {_K_OPT} (plancher "
        f"{plancher:.0f}, mesure {_MIN_RATIO[method]}x a `766d289`) : "
        "`maxiter` est peut-etre devenu un budget d'evaluations pour cette "
        "methode aussi — remesurer D-119")


def _cli_method_default():
    """Le `default` de `--method`, lu dans l'AST de `src/pipeline.py`.

    Par l'AST et non par une recherche de chaine : un test qui lit le texte
    du source teste sa mise en forme (D-114, D-115).
    """
    src = open(os.path.join(_SRC, "pipeline.py"), encoding="utf-8").read()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--method"):
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                return ast.literal_eval(kw.value)
    return None


def _training_method():
    """L'optimiseur que l'objectif d'entrainement emploie REELLEMENT.

    Obtenu en appelant `create_argus`, pas en relisant le fichier : c'est
    l'objet que la campagne Optuna a consomme.
    """
    import train_hyperparams as th
    return th.create_argus(th.SCENARIO_OT).method


def test_no_launcher_of_the_repository_pins_the_optimiser():
    """Si un lanceur passait `--method`, le defaut ne servirait a rien.

    C'est la mesure qui rend le second fait de D-119 consequent : rien ne
    pose l'optimiseur, donc tout run de `pipeline.py` prend le defaut.
    """
    lanceurs = []
    for base in (_ROOT, os.path.join(_ROOT, "scripts")):
        if os.path.isdir(base):
            lanceurs += [os.path.join(base, f) for f in sorted(os.listdir(base))
                         if f.endswith(".sh")]
    assert len(lanceurs) >= 6, (
        f"{len(lanceurs)} lanceurs balayes : balayage vide, rien de prouve")
    porteurs = [os.path.relpath(p, _ROOT) for p in lanceurs
                if "--method" in open(p, encoding="utf-8",
                                      errors="replace").read()]
    assert porteurs == [], (
        f"{porteurs} pose(nt) desormais `--method` : le defaut du CLI n'est "
        "plus ce qui tourne, remesurer D-119")


def test_training_and_deployment_agree_on_the_optimiser():
    """Ce que les deux chemins PROMETTENT : optimiser la meme chose.

    L'entrainement choisit les hyperparametres en minimisant un objectif que
    COBYLA explore ; le deploiement les consomme en laissant L-BFGS-B
    explorer. Les deux devraient coincider — question 4 de `VIGIL.md`.
    """
    entraine = _training_method()
    deploye = _cli_method_default()
    assert deploye is not None, (
        "`--method` n'a plus de `default` dans src/pipeline.py : le test ne "
        "mesure plus rien")
    assert entraine == deploye, (
        f"entrainement sous {entraine!r}, deploiement par defaut sous "
        f"{deploye!r}")


def test_the_gap_between_the_two_optimisers_is_smaller_than_the_qaoa_spread():
    """La moitie honnete du resultat : cette mesure ne tranche pas.

    Un test qui epingle une NON-conclusion. Il rougirait le jour ou le bras
    QAOA deviendrait assez reproductible pour que la comparaison decide —
    et ce jour-la, il faudrait la refaire pour de bon.

    Mesure a `766d289`, `dim = 2`, `K_opt = 30`, 6 tirages par methode :
    ecart des moyennes **0,0867**, dispersion intra **0,200** / **0,240**.
    Ce test rejoue une version courte (3 tirages) et n'exige que l'ordre.
    """
    from types import SimpleNamespace

    from Simulation.HamiltParams_v2 import PhysicalMapperV2
    from Simulation.PhysToAngle import AngleMapper
    from Simulation.grid import curl_z
    from VQA.runtime import VQARuntime
    from call_vqa_shell import call_vqa_shell

    dim, reps, tirages = 2, 2, 3
    p = dim + 2
    rng = np.random.default_rng(3)
    f = {k: rng.normal(size=(p, p)) for k in ("vx", "vy", "Bx", "By")}
    f["Jz"] = curl_z(f["Bx"], f["By"], True)
    score = AngleMapper.classical_score(f)
    params = PhysicalMapperV2(dx=0.02).compute_coefficients(None, score, f,
                                                            0.1496)
    theta = 2.0 * np.arcsin(np.sqrt(np.clip(score, 0.0, 1.0)))
    zero = np.zeros((p, p))
    angles = (theta, theta.copy(), zero, zero.copy())

    def marginales(method):
        args = SimpleNamespace(reps=reps, mode="simulator",
                               backend="state_vector", shots=1024,
                               method=method, opt_level=1,
                               AdvAnomaliesEnable=False, K_opt=30, eps=1e-2)
        runtime = VQARuntime(backend_name="state_vector", mode="simulator",
                             shots=1024, opt_level=1)
        out, _ = call_vqa_shell(angles, params, False, args,
                                period_bound=False, vqa_runtime=runtime)
        return np.array(out)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = np.array([marginales("COBYLA") for _ in range(tirages)])
        b = np.array([marginales("L-BFGS-B") for _ in range(tirages)])

    ecart = float(np.max(np.abs(a.mean(0) - b.mean(0))))
    intra = max(float(np.max(a.max(0) - a.min(0))),
                float(np.max(b.max(0) - b.min(0))))

    assert intra > 0.0, "le bras QAOA est devenu deterministe : remesurer"
    assert ecart < intra, (
        f"ecart inter-methodes {ecart:.4f} au-dela de la dispersion "
        f"intra-methode {intra:.4f} : la comparaison tranche desormais, "
        "refaire la mesure de D-119 pour de bon (campagne, pas 3 tirages)")
