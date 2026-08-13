"""Les quatre poches partielles de V1, auditées avant la réoptimisation.

`COUVERTURE.md` listait quatre modules « partiellement audités » : des
fonctions jamais soumises aux quatre questions, dans du code par ailleurs
relu. Trois d'entre elles décident ce que la campagne va mesurer.

Ce fichier porte ce qui en est sorti :

  - **D-48**, la seule trouvaille : `mode="hardware"` s'exécutait sur un
    simulateur sans le dire ;
  - trois vérifications **saines**, épinglées pour ne pas être refaites —
    la mémoire TTL, le bras `classical_only`, et le mode Colab.

Une vérification saine est un résultat : sans elle, la prochaine passe
relit le même code. Chaque test dit donc aussi **quels axes ont été
empruntés**.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))


# ══════════════════════════════════════════════════════════════════
#  D-48 — `mode="hardware"` tournait sur simulateur sans le signaler
# ══════════════════════════════════════════════════════════════════

def test_the_runtime_only_ever_built_simulators():
    """La mesure qui fonde D-48, épinglée.

    `_init_backend` ne dispatche que sur `backend_name` : il rendait le
    MÊME objet pour `mode='simulator'` et `mode='hardware'`. Ce test
    reconstruit cette mesure sur le seul mode encore acceptable, pour
    que le jour où un backend matériel serait câblé, elle soit refaite
    plutôt que supposée.
    """
    from VQA.runtime import VQARuntime
    attendu = {
        "state_vector": "AerSimulator",
        "matrix_product_state": "AerSimulator",
        "aer": "AerSimulator",
        "estimator": "FakeFez",
    }
    for name, cls in attendu.items():
        r = VQARuntime(backend_name=name, mode="simulator", shots=64, opt_level=1)
        assert type(r._backend).__name__ == cls, name


@pytest.mark.parametrize("mode", ["hardware", "ibm", "real", ""])
def test_a_non_simulator_mode_is_refused_at_construction(mode):
    """Le refus tombe au constructeur, pas au milieu d'une campagne."""
    from VQA.runtime import VQARuntime
    with pytest.raises(ValueError, match="non supporte"):
        VQARuntime(backend_name="state_vector", mode=mode, shots=64, opt_level=1)


def test_the_mode_parameter_is_now_read():
    """Épinglage : `mode` était stocké et lu NULLE PART dans `src/`.

    Un paramètre qu'aucun code ne lit ressemble à un réglage. Celui-ci
    en avait l'air pendant que le chemin matériel rendait des nombres de
    simulateur.
    """
    import ast
    src = open(os.path.join(_REPO_ROOT, "src", "VQA", "runtime.py")).read()
    tree = ast.parse(src)
    lectures = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr == "mode"
        and isinstance(n.value, ast.Name) and n.value.id == "self"
        and isinstance(n.ctx, ast.Load)
    ]
    assert lectures, "`self.mode` n'est toujours lu nulle part"


def test_execute_refuses_the_hardware_path_too():
    """`VQARuntime` refuse à la construction ; `execute` couvre le chemin
    hérité où `vqa_runtime is None`."""
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp
    from VQA.execute import execute
    from VQA.init_qbits_state import init_qbits_state

    th = np.full((2, 2), 1.0)
    qc0 = init_qbits_state(th, th, np.zeros((2, 2)), np.zeros((2, 2)))
    n = qc0.num_qubits
    H = SparsePauliOp.from_list([("Z" + "I" * (n - 1), -0.5)])
    qc = QAOAAnsatz(cost_operator=H, reps=2, initial_state=qc0).decompose().decompose()

    with pytest.raises(ValueError, match="non supporte"):
        execute(qc, H, "hardware", "state_vector", 64, 2, 4, 1e-2, 1.0, False,
                vqa_runtime=None, method="COBYLA")


def test_the_cli_no_longer_advertises_a_mode_it_cannot_honour():
    """`--mode hardware` figurait dans les choix de `pipeline.main()`. Une
    option annoncée dans l'aide est une promesse."""
    import ast
    tree = ast.parse(open(os.path.join(_REPO_ROOT, "src", "pipeline.py")).read())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "add_argument"):
            continue
        if not any(isinstance(a, ast.Constant) and a.value == "--mode"
                   for a in node.args):
            continue
        for kw in node.keywords:
            if kw.arg == "choices":
                choix = [e.value for e in kw.value.elts]
                assert choix == ["simulator"], choix
                return
    pytest.fail("l'argument --mode a disparu de pipeline.main()")


# ══════════════════════════════════════════════════════════════════
#  Vérifié et trouvé sain — la mémoire TTL
#  Axes empruntés : détection fraîche / signal perdu / plusieurs pas
#  hybrides d'affilée / les deux bras.
# ══════════════════════════════════════════════════════════════════

_TTL_N, _TTL_DIM, _TTL_THR = 8, 2, 0.5


def _balayage_classique(score, ttl_map, max_depth=2):
    """Un « pas hybride » du détecteur classique, TTL persistante."""
    from Simulation.refinement import _run_level_classical
    actives, pending, depth = [], [(0, _TTL_N, 0, _TTL_N)], 0
    while pending and depth <= max_depth:
        pending = _run_level_classical(
            pending, depth, full_score=score, target_dim=_TTL_DIM,
            max_depth=max_depth, min_size=2, threshold_amr=_TTL_THR,
            active_patches=actives, verbose=False,
            solve_max_depth=max_depth, ttl_map=ttl_map)
        depth += 1
    return actives


def test_the_ttl_grants_exactly_one_grace_step_then_expires():
    """Le contrat annoncé : « survit 1 pas hybride après la dernière
    détection ». Mesuré chaud puis froid, sur tout l'arbre."""
    from Simulation.refinement import DEFAULT_TTL
    chaud = np.full((_TTL_N, _TTL_N), 0.9)
    froid = np.full((_TTL_N, _TTL_N), 0.1)
    ttl = {}

    _balayage_classique(chaud, ttl)
    assert ttl and set(ttl.values()) == {DEFAULT_TTL}, ttl

    _balayage_classique(froid, ttl)
    assert set(ttl.values()) == {0}, "le sursis n'a pas été consommé"

    _balayage_classique(froid, ttl)
    assert set(ttl.values()) == {0}, "une TTL expirée s'est réarmée"


def test_an_expired_ttl_never_forces_refinement_again():
    """Le sursis ne se réarme que sur une DÉTECTION, pas sur une visite."""
    chaud = np.full((_TTL_N, _TTL_N), 0.9)
    froid = np.full((_TTL_N, _TTL_N), 0.1)
    ttl = {}
    _balayage_classique(chaud, ttl)
    apres_sursis = _balayage_classique(froid, ttl)
    apres_expiration = _balayage_classique(froid, ttl)

    profondeur_max = max(p["depth"] for p in apres_expiration)
    assert profondeur_max <= max(p["depth"] for p in apres_sursis)


def test_the_ttl_hypothesis_that_a_stale_entry_survives_is_false():
    """Hypothèse mesurée et RÉFUTÉE, consignée pour ne pas la reformer.

    On soupçonnait qu'un patch dont le parent cesse d'être raffiné ne
    serait jamais visité, donc jamais décrémenté — sa TTL survivrait un
    nombre arbitraire de pas hybrides. Mesure : la TTL du **parent** le
    maintient dans `next_level`, donc l'enfant EST visité et décrémenté.
    Tout l'arbre passe de 1 à 0 au même pas.
    """
    chaud = np.full((_TTL_N, _TTL_N), 0.9)
    froid = np.full((_TTL_N, _TTL_N), 0.1)
    ttl = {}
    _balayage_classique(chaud, ttl)
    profondeurs = {len([c for c in k]) for k in ttl}
    assert len(ttl) > 4, "l'arbre n'a pas plus d'un niveau, le test ne sépare rien"

    _balayage_classique(froid, ttl)
    assert set(ttl.values()) == {0}, (
        "des entrées survivent au pas froid : l'hypothèse redevient "
        "plausible, la remesurer plutôt que d'ajuster ce test")


def test_the_ttl_map_cannot_grow_without_bound():
    """Les clés sont les bornes du pavage, déterministes : le nombre
    d'entrées est borné par l'arbre, pas par le nombre de pas."""
    chaud = np.full((_TTL_N, _TTL_N), 0.9)
    ttl = {}
    tailles = []
    for _ in range(5):
        _balayage_classique(chaud, ttl)
        tailles.append(len(ttl))
    assert tailles[0] == tailles[-1], f"la mémoire TTL croît : {tailles}"


# ══════════════════════════════════════════════════════════════════
#  Vérifié et trouvé sain — le bras `classical_only`
#  Axes empruntés : classical_only seul / avec classic_AMR_comp /
#  deux exécutions identiques.
# ══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def petit_run():
    import warnings
    import train_hyperparams as TH
    from Simulation.pre_compute_dns import precompute_dns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = {**TH.SCENARIO_KH, "N": 32,
               "T_MAX": TH.SCENARIO_KH["T_START"] + 0.1, "K_opt": 4,
               "shots": 32, "max_depth_override": 1, "study_name": "dns_kh"}
        trace, hot = precompute_dns(cfg)
    hp = {**{n: (lo + hi) / 2 for n, (lo, hi, _) in TH.SEARCH_SPACE.items()},
          **TH.FIXED_PARAMS}
    return dict(N=cfg["N"], VQA_N=2, T_MAX=cfg["T_MAX"], DT=cfg["DT"],
                HYBRID=int(cfg["HYBRID_DT"] / cfg["DT"]), verbose=False,
                argus=TH.create_argus(cfg), hyperparams=hp, lambda_cost=0.4,
                trial=None, dns_trace=trace, hot_start_state=hot,
                max_depth_override=cfg["max_depth_override"],
                scenario=cfg["scenario"], return_details=True)


def test_the_classical_arm_is_deterministic(petit_run):
    """Le bras de comparaison ne doit rien devoir au hasard — sinon un
    écart entre les deux bras ne se distingue pas d'un tirage."""
    import warnings
    import pipeline as P
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = P.pipeline(**petit_run, classical_only=True)
        b = P.pipeline(**petit_run, classical_only=True)
    assert a["combined"] == b["combined"]
    assert a["patch_ratio"] == b["patch_ratio"]


def test_the_two_classical_call_sites_do_not_interfere(petit_run):
    """`classical_only` et `classic_AMR_comp` appellent le MÊME détecteur
    depuis deux endroits, avec deux mémoires TTL distinctes
    (`ttl_map` / `ttl_map_classical`). Question 4 : le résultat du
    premier change-t-il quand le second tourne aussi ?"""
    import warnings
    import pipeline as P
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seul = P.pipeline(**petit_run, classical_only=True)
        avec = P.pipeline(**petit_run, classical_only=True,
                          classic_AMR_comp=True)
    assert avec["combined"] == seul["combined"]


def test_the_classical_arm_carries_the_sigma_provenance_too(petit_run):
    """D-36 portait sur les quatre sorties détaillées ; le bras classique
    en emprunte une. Elle porte la provenance comme les autres."""
    import warnings
    import pipeline as P
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = P.pipeline(**petit_run, classical_only=True)
    assert out["sigma_source"] == "loaded"


# ══════════════════════════════════════════════════════════════════
#  Vérifié et trouvé sain — le mode Colab
#  Axe NON empruntable ici : `google.colab` n'est pas importable.
#  Ce qui est vérifiable l'est ; le reste est nommé.
# ══════════════════════════════════════════════════════════════════

def test_outside_colab_no_drive_path_is_ever_constructed():
    import train_hyperparams as TH
    assert TH.IN_COLAB is False
    assert TH.drive_dir is None and TH.local_dir is None
    assert TH.data_dir.endswith("Train_results")


def test_ensure_dirs_is_idempotent_and_silent(tmp_path, monkeypatch, capsys):
    import train_hyperparams as TH
    monkeypatch.setattr(TH, "data_dir", str(tmp_path / "sortie"))
    monkeypatch.setattr(TH, "_DIRS_READY", False)
    assert TH.ensure_dirs() == TH.ensure_dirs() == str(tmp_path / "sortie")
    assert os.path.isdir(tmp_path / "sortie")
    assert capsys.readouterr().out == ""


def test_every_drive_copy_sits_under_an_in_colab_guard():
    """Les recopies Drive touchent `drive_dir`, qui vaut None hors Colab :
    une seule non gardée lèverait `TypeError` sur une machine ordinaire.

    Le contrôle marche sur l'AST — il vérifie que chaque appel est
    DANS un `if IN_COLAB`, pas qu'il en soit voisin dans le texte.
    """
    import ast
    tree = ast.parse(open(os.path.join(_REPO_ROOT, "src",
                                       "train_hyperparams.py")).read())

    def copies_drive(noeud):
        return [ast.unparse(n)[:60] for n in ast.walk(noeud)
                if isinstance(n, ast.Call)
                and getattr(n.func, "attr", None) in ("copy2", "copytree")
                and "drive_dir" in ast.unparse(n)]

    toutes = copies_drive(tree)
    assert toutes, "aucune recopie Drive : le balayage ne prouve rien"

    gardees = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "IN_COLAB" in ast.unparse(node.test):
            gardees.extend(copies_drive(node))

    assert set(toutes) == set(gardees), (
        f"recopies Drive hors garde IN_COLAB : "
        f"{sorted(set(toutes) - set(gardees))}")
