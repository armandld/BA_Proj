"""D-118 — l'axe « backend » traverse, du cote qui ne l'avait jamais ete.

`VIGIL_BA_Proj.md` liste sept axes de configuration et demande que chacun
soit emprunte des DEUX cotes. La passe precedente a compte, sur la suite
entiere : backend `state_vector` **320 appels**, backend echantillonne
**0**. Ce fichier prend ce zero.

Ce que la traversee a rendu, en deux temps.

**1. `aer` (echantillonne) et `state_vector` (exact) COINCIDENT** — mesure a
parametres FIXES, pour separer le backend de l'optimiseur (qui, lui, est
stochastique dans les deux cas : `EstimatorV2` tire `default_shots` meme
sous `AerSimulator(method='statevector')`). Ecart maximal sur les huit
marginales d'un patch `dim = 2` deploye, contre le statevector exact :

    shots =  1 024 : 0.0205   (bruit de tir attendu ~0.0312)
    shots =  8 192 : 0.0102   (~0.0110)
    shots = 65 536 : 0.0037   (~0.0039)

Les deux chemins de `execute` — `Statevector.probabilities_dict()` et
`sampler.run(...).get_counts()` — rendent donc la meme distribution, a la
racine de N pres. **Sain.**

**2. `estimator` (le troisieme choix du CLI) ne rend RIEN, jamais.**
`src/pipeline.py` l'offre : `--backend {aer, estimator, state_vector}`.
Il resout `FakeFez`, un modele de machine reelle a 156 qubits. La
transpilation etale le circuit LOGIQUE sur ces 156 qubits, `measure_all()`
cree 156 bits classiques, et le simulateur refuse : 2^156 ne tient pas en
memoire. La panne remonte sous la forme

    ValueError: could not broadcast input array from shape (0,20) into
                shape (1024,20)

— 20 octets, soit les 156 bits empaquetes — qui ne nomme ni le backend, ni
la memoire, ni la transpilation. Mesure : identique a **2 qubits logiques**
et a **8**, donc independante de la taille du probleme. Aucun nombre publie
n'en depend, et ne PEUT en dependre : ce chemin n'a jamais pu rendre une
valeur.

Pourquoi ce fichier ne corrige rien
-----------------------------------
Parce que la panne visible en cache une seconde, qui ne le serait pas.
`call_vqa_shell` appelle `postprocess(probs, qc.num_qubits)` avec le
nombre de qubits du circuit TRANSPILE. Si l'on rendait seulement la memoire
au simulateur, `postprocess` recevrait 156 == 156, son garde de longueur ne
verrait rien, et il rendrait **156 marginales indexees par qubit PHYSIQUE**
la ou l'appelant en attend 8, indexees par qubit logique. Mesure du
placement, `dim = 2` (4 qubits logiques) :

    final_index_layout() = [136, 142, 141, 143]

La marginale du qubit logique 0 se lit a l'indice 136. Une correction qui
ne rend que la memoire transforme donc un plantage — qui se voit — en une
valeur plausible et fausse — qui ne se voit pas. C'est exactement la classe
que `VIGIL.md` dit etre la seule qui compte.

Trancher entre « retirer `estimator` des choix » et « le cabler pour de
bon, layout compris » est une DECISION, pas une correction de chemin :
`VIGIL.md` dit alors de mesurer, documenter, ne pas corriger, et demander.
C'est ce que fait ce fichier — le `xfail(strict=True)` ci-dessous fera
rougir la suite le jour ou la panne sera levee, ce qui obligera a regarder
le placement dans le meme geste.

Sur quelle entree ces tests echouent
------------------------------------
Le premier, si le CLI cesse d'offrir `estimator` (la dette est payee, ce
fichier doit partir). Le deuxieme, si la transpilation cesse d'etaler le
circuit ou si le placement redevient l'identite. Le troisieme, s'il se met
a PASSER — c'est le sens de `strict=True`.
"""

import ast
import os
import sys
import warnings

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

from VQA.runtime import VQARuntime  # noqa: E402

#: Le circuit le plus petit que le depot sache construire — la mesure ne
#: depend pas de la taille, et ce fichier ne doit pas couter une minute.
_OP_2Q = SparsePauliOp.from_list([("ZZ", -1.0)])
_OP_4Q = SparsePauliOp.from_list([("ZZII", -1.0), ("IZZI", 0.7),
                                  ("IIZZ", -0.3), ("ZIIZ", 0.2)])

#: Mesure, `766d289` : FakeFez expose 156 qubits, quel que soit le circuit.
_FAKEFEZ_QUBITS = 156


def _ansatz(op, reps=1):
    return QAOAAnsatz(cost_operator=op, reps=reps).decompose(reps=3)


def _cli_backend_choices():
    """Les `choices` de `--backend`, lus dans l'AST de `src/pipeline.py`.

    Par l'AST et non par une recherche de chaine : un test qui lit le texte
    du source teste sa mise en forme, pas son comportement (D-114, D-115).
    """
    src = open(os.path.join(_SRC, "pipeline.py"), encoding="utf-8").read()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--backend"):
            continue
        for kw in node.keywords:
            if kw.arg == "choices":
                return [ast.literal_eval(e) for e in kw.value.elts]
    return None


def _deployed_patch(dim=2, reps=2, seed=0):
    """Un patch `dim x dim` DEPLOYE — le meme geste que `refinement`."""
    import numpy as np
    from Simulation.HamiltParams_v2 import PhysicalMapperV2
    from Simulation.PhysToAngle import AngleMapper
    from Simulation.grid import curl_z
    from VQA.mapping import mapping

    p = dim + 2
    rng = np.random.default_rng(seed)
    f = {k: rng.normal(size=(p, p)) for k in ("vx", "vy", "Bx", "By")}
    f["Jz"] = curl_z(f["Bx"], f["By"], True)
    score = AngleMapper.classical_score(f)
    params = PhysicalMapperV2(dx=0.02).compute_coefficients(None, score, f,
                                                            0.1496)
    theta = 2.0 * np.arcsin(np.sqrt(np.clip(score, 0.0, 1.0)))
    zero = np.zeros((p, p))
    qc, _ = mapping({"theta_h": theta, "theta_v": theta.copy(),
                     "psi_h": zero, "psi_v": zero.copy()},
                    params, False, period_bound=False, reps=reps)
    return qc


def test_the_sampled_backend_agrees_with_the_exact_one_to_shot_noise():
    """L'axe backend, cote sain : les deux chemins finaux de `execute`.

    A parametres FIXES — sinon on mesure l'optimiseur, qui est stochastique
    dans les deux cas (`EstimatorV2` tire `default_shots` meme sous
    `AerSimulator(method='statevector')`), et non le backend.

    Le seuil n'est PAS calibre sur la mesure du jour : c'est 4 sigma
    binomiaux a p = 0.5, soit `2/sqrt(shots)` = 0.0221 a 8 192 tirs. Les
    cinq tirages de reference mesures a `766d289` valent 0.00538, 0.00561,
    0.00591, 0.00686 et 0.01122 — tous sous la moitie du seuil. A 65 536
    tirs l'ecart tombe a 0.0037 pour un bruit attendu de 0.0039.

    Sur quelle entree ce test echoue
    --------------------------------
    Sur une divergence de convention entre les deux chemins — ordre des
    bits, largeur de chaine, normalisation par le nombre de tirs — qui
    ne se verrait sur aucune des deux prises isolement.
    """
    import numpy as np
    from qiskit.quantum_info import Statevector
    from VQA.postprocess import postprocess

    shots = 8192
    qc = _deployed_patch()
    x = np.array([0.10, -0.05, 0.7, 1.3])

    exact = np.array(postprocess(
        Statevector.from_instruction(qc.assign_parameters(x))
        .probabilities_dict(), qc.num_qubits, False))

    rt = VQARuntime(backend_name="aer", mode="simulator", shots=shots,
                    opt_level=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qct = rt.transpile(qc)
    circuit = qct.assign_parameters(x)
    circuit.measure_all()
    counts = rt.sampler.run([(circuit,)]).result()[0].data.meas.get_counts()
    total = sum(counts.values())
    assert total == shots, f"{total} tirs pour {shots} demandes"
    sampled = np.array(postprocess({k: v / total for k, v in counts.items()},
                                   qct.num_qubits, False))

    assert len(sampled) == len(exact) == 8, (
        f"{len(sampled)} / {len(exact)} marginales pour 8 attendues "
        "(2 familles de liens sur un patch dim = 2)")
    ecart = float(np.max(np.abs(sampled - exact)))
    assert ecart < 2.0 / np.sqrt(shots), (
        f"ecart max {ecart:.5f} au-dela de 4 sigma binomiaux "
        f"({2.0 / np.sqrt(shots):.5f}) : les deux chemins de `execute` ne "
        "rendent plus la meme distribution")


def test_the_cli_still_offers_the_backend_that_cannot_run():
    """La dette n'existe que tant que le CLI l'offre.

    Le jour ou `estimator` sort des `choices`, la decision a ete prise et
    ce fichier entier doit etre relu — c'est ce que ce test force.
    """
    choices = _cli_backend_choices()
    assert choices is not None, (
        "`--backend` n'a plus de `choices` dans src/pipeline.py : "
        "le balayage ne prouve plus rien")
    assert "estimator" in choices, (
        f"`estimator` a quitte les choix du CLI ({choices}) : la dette de "
        "D-118 est tranchee, retirer ce fichier et sa ligne de registre")
    assert "state_vector" in choices and "aer" in choices, (
        f"les deux backends sains ont bouge : {choices}")


@pytest.mark.parametrize("op,logiques", [(_OP_2Q, 2), (_OP_4Q, 4)])
def test_the_fake_device_spreads_the_circuit_over_every_physical_qubit(op,
                                                                      logiques):
    """La cause, mesuree : la transpilation etale, elle ne restreint pas.

    Independante de la taille du probleme — c'est ce que la
    parametrisation 2 / 4 qubits epingle.
    """
    rt = VQARuntime(backend_name="estimator", mode="simulator",
                    shots=256, opt_level=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qct = rt.transpile(_ansatz(op))

    assert qct.num_qubits == _FAKEFEZ_QUBITS, (
        f"{qct.num_qubits} qubits physiques, {_FAKEFEZ_QUBITS} mesures a "
        "`766d289` : la mesure de D-118 est a refaire")
    assert qct.num_qubits > logiques, "le circuit n'est plus etale"
    assert qct.layout is not None, (
        "plus de layout : la seconde moitie de D-118 — les marginales lues "
        "au mauvais indice — est peut-etre levee, remesurer")


def test_the_logical_qubits_do_not_land_on_their_own_indices():
    """La SECONDE panne, celle qui ne se verrait pas.

    `call_vqa_shell` passe `qc.num_qubits` (transpile) a `postprocess`. Si
    la memoire seule etait rendue, le garde de longueur verrait 156 == 156
    et laisserait passer des marginales indexees par qubit physique.
    """
    rt = VQARuntime(backend_name="estimator", mode="simulator",
                    shots=256, opt_level=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qct = rt.transpile(_ansatz(_OP_4Q, reps=2))

    placement = list(qct.layout.final_index_layout())
    assert len(placement) == 4, f"{len(placement)} qubits logiques places"
    assert placement != list(range(4)), (
        "le placement est redevenu l'identite : la marginale du qubit "
        "logique i se lirait bien a l'indice i, remesurer D-118")
    assert max(placement) >= 4, (
        f"placement {placement} entierement dans les 4 premiers indices : "
        "remesurer")


@pytest.mark.xfail(strict=True, reason=(
    "D-118 — dette declaree, non corrigee. `--backend estimator` transpile "
    "sur les 156 qubits de FakeFez ; `measure_all()` cree 156 bits et le "
    "simulateur depasse `max_memory_mb`. La levee ne nomme ni le backend "
    "ni la memoire. NE PAS lever ce xfail en rendant seulement la memoire : "
    "il resterait des marginales indexees par qubit PHYSIQUE, plausibles et "
    "fausses (voir le test de placement ci-dessus). La decision — retirer "
    "le choix, ou cabler le layout de bout en bout — est pour USER."))
def test_the_estimator_backend_can_produce_a_final_distribution():
    """Ce que le backend PROMET : une distribution finale. Il ne la rend pas.

    Ce test porte sur la garantie annoncee, pas sur l'absence de plantage :
    il exige des comptes, et c'est en les exigeant qu'il rougit.
    """
    rt = VQARuntime(backend_name="estimator", mode="simulator",
                    shots=256, opt_level=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qct = rt.transpile(_ansatz(_OP_2Q))
    circuit = qct.assign_parameters([0.1, 0.2])
    circuit.measure_all()

    counts = rt.sampler.run([(circuit,)]).result()[0].data.meas.get_counts()

    assert counts, "aucun compte rendu"
    assert sum(counts.values()) == 256
