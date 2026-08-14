"""Le point d'entree en ligne de commande : D-66 et D-67.

`python src/pipeline.py` est le seul moyen de lancer une simulation
complete a la main. Rien dans le depot n'importe `main` -- tout importe la
FONCTION `pipeline`. C'est exactement pourquoi personne ne voyait que la
CLI ne calculait rien.

Mesure avant correction, invocation par defaut :

    combined 0.333333   phys_score 0.000000   patch_ratio 1.0000
    erreur exactement nulle sur les cinq champs, code de retour 0

Le hot start place `t_current` a T_START = 2.3 ; `T_MAX` valait 1.0, le
defaut de la CLI. `while t_current < T_MAX` etait faux des l'entree, la
boucle ne tournait pas, l'etat final restait l'etat DNS.

Mesure apres correction, meme invocation :

    Q-HAS      combined 0.228928  phys 0.140052  patch 0.4067
    Classique  combined 0.212591  phys 0.117626  patch 0.4025
"""

import ast
import os
import pathlib
import re
import subprocess
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO, "src")
_SCRIPT = os.path.join(_SRC, "pipeline.py")


@pytest.fixture(scope="module")
def P():
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    return pytest.importorskip("pipeline")


# ══════════════════════════════════════════════════════════════════════
#  La table de scenarios — l'invariant dont la violation faisait D-66
# ══════════════════════════════════════════════════════════════════════

def test_chaque_scenario_a_un_horizon_non_vide(P):
    """`T_MAX > T_START`, sans quoi la boucle hybride ne tourne pas."""
    assert P.PHASE, "PHASE est vide — le balayage ne verifierait rien"
    for nom, cfg in P.PHASE.items():
        t0, t1 = cfg.get("T_START", 0.0), cfg["T_MAX"]
        assert t1 > t0, (
            f"{nom} : T_MAX={t1} <= T_START={t0}. La boucle "
            f"`while t_current < T_MAX` serait fausse des l'entree.")


def test_chaque_scenario_declare_ce_dont_main_a_besoin(P):
    """Une cle absente se replie silencieusement sur un defaut de CLI —
    c'est le mecanisme de D-66 et de D-33."""
    requis = ("scenario", "N", "T_MAX", "T_START", "DT", "HYBRID_DT",
              "K_opt", "Re", "Rm", "shots", "AdvAnomaliesEnable",
              "max_depth_override")
    for nom, cfg in P.PHASE.items():
        manquantes = [k for k in requis if k not in cfg]
        assert not manquantes, f"{nom} : cles manquantes {manquantes}"


def test_le_pas_hybride_est_un_multiple_entier_du_pas_de_temps(P):
    """`HYBRID = int(HYBRID_DT / DT)` : un rapport non entier decalerait
    silencieusement la frequence d'hybridation."""
    for nom, cfg in P.PHASE.items():
        rapport = cfg["HYBRID_DT"] / cfg["DT"]
        assert abs(rapport - round(rapport)) < 1e-9, (
            f"{nom} : HYBRID_DT/DT = {rapport}, non entier")
        assert round(rapport) >= 1, f"{nom} : rapport {rapport} < 1"


def test_la_cli_n_annonce_que_des_scenarios_configures(P):
    """Une option affichee dans l'aide est une promesse.

    La liste ecrite a la main annoncait dix scenarios pour sept entrees de
    `PHASE` : trois etaient acceptes puis levaient `KeyError`.
    """
    r = subprocess.run([sys.executable, _SCRIPT, "--scenario", "magnetic_twist"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0, "un scenario absent de PHASE doit etre refuse"
    assert "invalid choice" in r.stderr, r.stderr

    aide = subprocess.run([sys.executable, _SCRIPT, "--help"],
                          capture_output=True, text=True, timeout=300).stdout
    m = re.search(r"--scenario \{([^}]*)\}", aide)
    assert m, f"choix de --scenario illisibles dans l'aide :\n{aide}"
    annonces = set(m.group(1).split(","))
    assert annonces == set(P.PHASE), (
        f"CLI annonce {sorted(annonces)}, PHASE porte {sorted(P.PHASE)}")


# ══════════════════════════════════════════════════════════════════════
#  D-67 — un run vide ne doit pas se noter
# ══════════════════════════════════════════════════════════════════════

def test_d67_score_refuse_zero_pas(P):
    """`total_steps == 0` se repliait sur `patch_ratio = 1.0`, donc sur
    `combined = lambda/(1+lambda)` — 0.333333 a lambda=0.5."""
    import numpy as np
    champs = {k: np.ones((8, 8)) for k in ("vx", "vy", "Bx", "By", "Jz")}
    with pytest.raises(ValueError, match="total_steps"):
        P.score(champs, champs, 0.5, 0, 0, 64)


def test_d67_score_fonctionne_avec_des_pas(P):
    """Garde-fou : refuser zero pas ne doit pas casser le cas nominal."""
    import numpy as np
    champs = {k: np.ones((8, 8)) for k in ("vx", "vy", "Bx", "By", "Jz")}
    out = P.score(champs, champs, 0.5, 32, 2, 64)
    assert out["phys_score"] == pytest.approx(0.0, abs=1e-12)
    assert out["patch_ratio"] == pytest.approx(0.25), (
        f"32 pixels sur 2 pas de 64 -> 0.25, obtenu {out['patch_ratio']}")
    assert out["combined"] == pytest.approx((0.0 + 0.5 * 0.25) / 1.5)


# ══════════════════════════════════════════════════════════════════════
#  D-66 — la configuration du scenario fait foi
# ══════════════════════════════════════════════════════════════════════

def test_d66_les_options_temporelles_n_ont_plus_de_defaut_de_cli():
    """Le mecanisme meme de D-66 : un defaut de CLI non nul ECRASE la
    valeur du scenario. Ces options doivent valoir `None`."""
    arbre = ast.parse(pathlib.Path(_SCRIPT).read_text())
    fn = next(n for n in arbre.body
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    defauts = {}
    for n in ast.walk(fn):
        if (isinstance(n, ast.Call)
                and getattr(n.func, "attr", None) == "add_argument" and n.args):
            nom = getattr(n.args[0], "value", None)
            for kw in n.keywords:
                if kw.arg == "default":
                    defauts[nom] = kw.value

    for opt in ("--t-max", "--dt", "--hybrid-dt", "--dns-resolution",
                "--shots", "--K-opt", "--AdvAnomaliesEnable"):
        assert opt in defauts, f"{opt} n'a plus de `default=` explicite"
        v = defauts[opt]
        assert isinstance(v, ast.Constant) and v.value is None, (
            f"{opt} a pour defaut {ast.dump(v)} au lieu de None : il "
            f"ecraserait la valeur du scenario (D-66)")


def test_d66_un_horizon_anterieur_au_depart_est_refuse():
    """Le garde-fou explicite : meme si quelqu'un passe `--t-max` avant
    `T_START`, le run doit lever au lieu de rendre un score vide."""
    r = subprocess.run(
        [sys.executable, _SCRIPT, "--scenario", "orszag_tang",
         "--t-max", "1.0", "--out-dir", "/tmp/d66"],
        capture_output=True, text=True, timeout=1200)
    assert r.returncode != 0, (
        f"T_MAX=1.0 < T_START=2.3 doit lever, code {r.returncode}")
    assert "T_START" in (r.stderr + r.stdout), (
        f"le message doit nommer T_START :\n{r.stderr[-2000:]}")
