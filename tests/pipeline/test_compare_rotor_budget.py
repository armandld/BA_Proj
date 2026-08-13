"""`src/compare_rotor_budget.py` — D-10 et ses defauts de configuration.

Ce script est la demonstration d'avantage quantique sous budget contraint :
quand seuls K blocs sur n^2 peuvent etre raffines, l'indicateur lineaire
classique ne distingue pas « forte vorticite sans Jz » (coeur du rotor,
lisse) de « forte vorticite ET fort Jz » (gaine magnetique, a raffiner).

Il ecrit un `.npz` — c'est donc un PRODUCTEUR de resultat, pas un
analyseur. Il n'avait aucun test, et il n'avait jamais tourne :

  D-10  `PhysicalMapper(..., beta=0.5, ...)` levait `TypeError` a l'etape
        4 sur 5, apres avoir paye le DNS. `beta` a quitte le constructeur
        du mapper pour devenir un argument de `run_adaptive_vqa`.

  defauts impossibles : `--n-blocks 4` demande 2*4^2 = 32 qubits, soit
        69 Go de statevector ; et une fois n_blocks ramene a 3, la
        resolution 128 n'est plus divisible par 3.
"""

import ast
import os
import pathlib
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO, "src")
_SCRIPT = os.path.join(_SRC, "compare_rotor_budget.py")


@pytest.fixture(scope="module")
def crb():
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    import matplotlib
    matplotlib.use("Agg")
    return pytest.importorskip("compare_rotor_budget")


def _mots_cles_appel_mapper():
    """Mots-cles passes a `PhysicalMapper(...)` dans `qhas_block_scores`.

    Lu sur l'AST : le texte du fichier contient aussi le commentaire qui
    documente D-10, et une recherche par chaine y trouve un faux `beta=`.
    """
    arbre = ast.parse(pathlib.Path(_SCRIPT).read_text())
    fn = next(n for n in ast.walk(arbre)
              if isinstance(n, ast.FunctionDef) and n.name == "qhas_block_scores")
    appels = [n for n in ast.walk(fn)
              if isinstance(n, ast.Call)
              and getattr(n.func, "id", None) == "PhysicalMapper"]
    assert len(appels) == 1, f"{len(appels)} appels a PhysicalMapper, 1 attendu"
    return {kw.arg for kw in appels[0].keywords}


def _defauts():
    """Valeurs par defaut declarees par la CLI du script."""
    arbre = ast.parse(pathlib.Path(_SCRIPT).read_text())
    fn = next(n for n in ast.walk(arbre)
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    out = {}
    for n in ast.walk(fn):
        if (isinstance(n, ast.Call)
                and getattr(n.func, "attr", None) == "add_argument" and n.args):
            nom = getattr(n.args[0], "value", None)
            for kw in n.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    out[nom] = kw.value.value
    return out


# ══════════════════════════════════════════════════════════════════════
#  D-10 — le script doit pouvoir se construire
# ══════════════════════════════════════════════════════════════════════

def test_d10_le_mapper_se_construit(crb):
    """`qhas_block_scores` construisait `PhysicalMapper(beta=...)`, un
    mot-cle retire de la signature."""
    import inspect
    from Simulation.HamiltParams import PhysicalMapper

    params = inspect.signature(PhysicalMapper.__init__).parameters
    assert "beta" not in params, (
        "`beta` est revenu dans la signature — verifier que ce test est "
        "encore utile")
    mots = _mots_cles_appel_mapper()
    assert "beta" not in mots, (
        f"`beta=` est repasse dans l'appel a PhysicalMapper : {sorted(mots)}")


def test_le_mapper_recoit_les_hyperparametres_deployes(crb):
    """Les constantes en dur `gamma_hydro=0.5, gamma_mag=0.5, kappa=5.0`
    n'etaient celles d'aucune campagne."""
    import inspect
    mots = _mots_cles_appel_mapper()
    for cle in ("gamma_hydro", "gamma_mag", "kappa", "sigma",
                "beta_curl", "beta_xpoint", "w_z_frac"):
        assert cle in mots, f"{cle} absent de l'appel a PhysicalMapper"
    assert "load_hyperparams" in inspect.getsource(crb.qhas_block_scores), (
        "les hyperparametres ne sont plus charges depuis le fichier deploye")


# ══════════════════════════════════════════════════════════════════════
#  La taille du circuit — refuser AVANT de payer le DNS
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("n_blocks,attendu", [(2, 8), (3, 18), (4, 32)])
def test_le_nombre_de_qubits_est_bien_2n2(crb, n_blocks, attendu):
    """Une arete par qubit, horizontales et verticales : 2*n^2."""
    assert crb.qubits_requis(n_blocks) == attendu


def test_la_garde_refuse_une_taille_hors_memoire(crb):
    """32 qubits = 69 Go. Mesure : qiskit-aer levait `Insufficient
    memory ... Required memory: 65536M` — a l'etape 4 sur 5."""
    with pytest.raises(ValueError, match="qubits"):
        crb.verifier_taille_circuit(4, "state_vector")


def test_la_garde_laisse_passer_ce_qui_tient(crb):
    """Garde-fou : refuser l'impossible ne doit pas refuser le possible."""
    crb.verifier_taille_circuit(2, "state_vector")
    crb.verifier_taille_circuit(3, "state_vector")


def test_les_defauts_forment_une_configuration_executable(crb):
    """Le couple d'origine (resolution 128, n_blocks 4) violait deux
    contraintes a la fois. Les defauts d'un script doivent tourner."""
    d = _defauts()
    N, n_blocks, budget = d["--resolution"], d["--n-blocks"], d["--budget"]

    assert N % n_blocks == 0, (
        f"defauts incoherents : resolution {N} non divisible par "
        f"n_blocks {n_blocks}")
    assert budget <= n_blocks ** 2, f"budget {budget} > {n_blocks**2} blocs"
    crb.verifier_taille_circuit(n_blocks, d["--backend"])

    # Le budget doit CONTRAINDRE : raffiner tous les blocs ne compare rien.
    assert budget < n_blocks ** 2, (
        f"budget {budget} = tous les {n_blocks**2} blocs : la contrainte "
        f"serait vide, et le test ne separerait rien")


def test_la_divisibilite_est_refusee_avec_une_suggestion():
    """Un message d'erreur qui ne dit pas quoi faire fait perdre un tour."""
    r = subprocess.run(
        [sys.executable, _SCRIPT, "--resolution", "128", "--n-blocks", "3"],
        capture_output=True, text=True, timeout=300)
    assert r.returncode != 0
    assert "--resolution" in r.stderr, (
        f"le message doit suggerer une resolution valide :\n{r.stderr}")


# ══════════════════════════════════════════════════════════════════════
#  Les fonctions pures
# ══════════════════════════════════════════════════════════════════════

def test_select_top_k_prend_bien_les_plus_grands(crb):
    import numpy as np
    scores = np.array([[0.1, 0.9], [0.5, 0.3]])
    assert crb.select_top_k(scores, 1) == [(0, 1)]
    assert set(crb.select_top_k(scores, 2)) == {(0, 1), (1, 0)}
    assert len(crb.select_top_k(scores, 4)) == 4


def test_compute_solution_error_est_nulle_sur_identite(crb):
    import numpy as np
    champs = {k: np.random.RandomState(0).rand(8, 8)
              for k in ("vx", "vy", "Bx", "By", "Jz")}
    err = crb.compute_solution_error(champs, champs)
    assert err == pytest.approx(0.0, abs=1e-12), (
        f"un champ compare a lui-meme doit donner 0, obtenu {err}")


def test_compute_solution_error_croit_avec_l_ecart(crb):
    """Champ qui SEPARE : sans cela, une erreur constante passerait."""
    import numpy as np
    base = {k: np.ones((8, 8)) for k in ("vx", "vy", "Bx", "By", "Jz")}
    proche = {k: v * 1.01 for k, v in base.items()}
    loin = {k: v * 1.5 for k, v in base.items()}
    e1 = crb.compute_solution_error(proche, base)
    e2 = crb.compute_solution_error(loin, base)
    assert 0 < e1 < e2, f"erreurs non ordonnees : {e1} puis {e2}"
