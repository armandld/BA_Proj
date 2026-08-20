"""D-178 : le resume cross-Re de la phase 4 relisait un autre artefact que
celui qu'elle venait d'ecrire.

`save_results` nomme l'artefact `exact_diag_..._dim{D}{sfx}.npz`, avec
`sfx = "_v2"` sous `--v2`. Le resume cross-Re de `main()` reconstruisait ce
nom a la main, **sans le suffixe**. Deux consequences, mesurees a `c7292a1`
sur `harris_tearing Re=400 N=256 dim=2` :

  relance fraiche, `--v2` seul        resume cross-Re : 0 ligne, code 0
  artefact v1 present, puis `--v2`    resume cross-Re : la ligne du v1

Le premier cas est l'ordre exact de `scripts/run_study_v2b.sh` phase 4, qui
lance `--v2` d'abord : la relance que `BRIEF_REPRISE.md` §8 demande imprime
son en-tete « PHASE 4 CROSS-Re SUMMARY » puis rien, et sort avec le code 0
— la famille D-55/D-56/D-75/D-148, dans la fonction meme qui leve deja pour
un balayage vide douze lignes plus haut.

Le second est le plus insidieux : la ligne imprimee est celle d'un autre
hamiltonien. Elle ne se voit pas *aujourd'hui* — D-47 rend les deux
predicteurs constants, donc le F1 ne depend que de la verite terrain et
sort identique en v1 et en v2 sur les quatre scenarios canoniques (0.3333,
0.3333, 0.3333, 0.3733). Les deux artefacts sont pourtant des objets
distincts : E0 moyen **-2039.70** (v1) contre **-11.15** (v2) sur
`orszag_tang`. Le jour ou D-47 est leve, le F1 cesse d'etre insensible au
hamiltonien et la ligne devient fausse.

`study/common/qaoa_inputs.py` (phase 5), l'autre lecteur de ces artefacts,
applique bien le suffixe : c'est le producteur qui divergeait de son propre
consommateur.

Le test 2 EPINGLE l'ancien comportement : il plante un artefact v1 portant
`exact_f1 = 0.999`, une valeur qu'aucun calcul ne produit ici, et exige que
le resume `--v2` ne l'imprime jamais.
"""
import contextlib
import io
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exact_diagonalisation as ed

SC, RE, N, DIM = "synthetic", 400, 8, 2   # 8 qubits, sous le plafond de 20
N_SNAPS = 4
PLANTED_F1 = 0.999                        # aucun calcul de ce test ne le rend


def _make_inputs(outdir):
    """Entrees minimales de la phase 4 : un DNS lisse et des erreurs L2."""
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")

    def field(phase):
        return np.stack([np.sin(X + phase) * np.cos(Y)] * N_SNAPS)

    np.savez(os.path.join(outdir, f"dns_{SC}_Re{RE}_N{N}.npz"),
             vx=field(0.1), vy=field(0.2), Bx=field(0.3), By=field(0.4),
             meta_Re=RE, meta_scenario=SC)

    l2 = np.linspace(0.0, 1.0, N_SNAPS * DIM * DIM).reshape(N_SNAPS, DIM, DIM)
    np.savez(os.path.join(outdir, f"patches_{SC}_Re{RE}_N{N}_dim{DIM}.npz"),
             l2_errors=l2, is_hard=l2 >= 0.5, l2_threshold=0.5)


def _run_v2(outdir):
    """Lance `main()` en `--v2` sur `outdir`, rend les lignes du resume."""
    old_dir, old_argv = ed.RESULTS_DIR, sys.argv
    old_defaults = ed.save_results.__defaults__
    ed.RESULTS_DIR = outdir
    ed.save_results.__defaults__ = (outdir,)
    sys.argv = ["exact_diagonalisation.py", "--scenario", SC, "--re", str(RE),
                "--N", str(N), "--dim", str(DIM), "--v2"]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            ed.main()
    finally:
        ed.RESULTS_DIR, sys.argv = old_dir, old_argv
        ed.save_results.__defaults__ = old_defaults

    out = buf.getvalue()
    rows = [l.strip() for l in out.splitlines()
            if "exact_F1=" in l and "promising=" in l]
    return out, rows


# ------------------------------------------------------------------
# 1. Une relance fraiche en --v2 ne rend pas un resume vide
# ------------------------------------------------------------------

def test_fresh_v2_relaunch_prints_its_summary_row(tmp_path):
    """Avant : 0 ligne sous l'en-tete, code 0. Le balayage vide silencieux."""
    _make_inputs(str(tmp_path))
    out, rows = _run_v2(str(tmp_path))

    assert "PHASE 4 CROSS-Re SUMMARY" in out, "en-tete du resume absent"
    assert len(rows) == 1, (
        f"resume cross-Re : {len(rows)} ligne(s) pour 1 attendue — un resume "
        f"vide sous un en-tete se lit comme une relance reussie (D-148)")
    assert f"{SC} Re={RE} dim={DIM}" in rows[0]

    written = [f for f in os.listdir(str(tmp_path))
               if f.startswith("exact_diag")]
    assert written == [f"exact_diag_{SC}_Re{RE}_N{N}_dim{DIM}_v2.npz"], written


# ------------------------------------------------------------------
# 2. Le resume --v2 ne relit jamais l'artefact v1 homonyme
# ------------------------------------------------------------------

def test_v2_summary_never_reports_the_v1_artifact(tmp_path):
    """Epingle l'ancien comportement : la ligne v1 plantee ne doit pas sortir."""
    _make_inputs(str(tmp_path))

    n = N_SNAPS
    np.savez(os.path.join(str(tmp_path),
                          f"exact_diag_{SC}_Re{RE}_N{N}_dim{DIM}.npz"),
             exact_f1=np.full(n, PLANTED_F1),
             classical_f1=np.full(n, PLANTED_F1),
             promising=np.ones(n, dtype=bool),
             degenerate_decision=np.zeros(n, dtype=bool),
             promising_informative=np.ones(n, dtype=bool))

    out, rows = _run_v2(str(tmp_path))

    assert len(rows) == 1, f"resume cross-Re : {len(rows)} ligne(s)"
    assert f"{PLANTED_F1:.3f}" not in rows[0], (
        f"le resume --v2 a imprime l'artefact v1 : {rows[0]!r}")

    v2 = np.load(os.path.join(str(tmp_path),
                              f"exact_diag_{SC}_Re{RE}_N{N}_dim{DIM}_v2.npz"))
    assert f"exact_F1={np.mean(v2['exact_f1']):.3f}" in rows[0], (
        f"le resume n'a pas imprime l'artefact v2 : {rows[0]!r}")


# ------------------------------------------------------------------
# 3. Un seul constructeur de nom, pour l'ecriture comme pour la relecture
# ------------------------------------------------------------------

def test_artifact_name_is_the_only_source_of_the_filename(tmp_path):
    meta = {"scenario": SC, "Re": RE, "N": N, "n_patches": DIM,
            "n_qubits": 2 * DIM * DIM,
            "snap_indices": np.array([0]), "suffix": "_v2"}
    fake = [{
        "ground_energy": -1.0, "gap": 0.5,
        "marginals": np.ones(2 * DIM * DIM),
        "decisions_h": np.ones((DIM, DIM), dtype=bool),
        "decisions_v": np.ones((DIM, DIM), dtype=bool),
        "gt_refine": np.zeros((DIM, DIM), dtype=bool),
        "promising": True, "degenerate_decision": True,
        "degenerate_classical": True, "f1_tie": True,
        "promising_informative": False,
        "comparison": {"exact": {"f1": 0.0}, "classical": {"f1": 0.0}},
    }]

    path = ed.save_results(fake, meta, outdir=str(tmp_path))
    assert os.path.basename(path) == ed.artifact_name(SC, RE, N, DIM, "_v2")
    assert ed.artifact_name(SC, RE, N, DIM, "_v2") != \
        ed.artifact_name(SC, RE, N, DIM, "")


# ------------------------------------------------------------------
# 4. D-179 — le contrat de `build_patch_hamiltonian`, verifie point par point
# ------------------------------------------------------------------

def test_build_patch_hamiltonian_returns_what_its_docstring_says():
    """Deux scores classiques de meme type et de meme intervalle, sur des
    grilles differentes : la docstring n'en annoncait qu'un, avec la forme
    de l'autre. Les 15 sites d'appel prennent tous le bon, mais rien ne le
    verifiait — l'assertion porte sur les FORMES, pas sur le texte."""
    n_full = 32
    x = np.linspace(0, 2 * np.pi, n_full, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    vx, vy, Bx, By = [np.sin(X + k) * np.cos(Y) for k in (0.1, 0.2, 0.3, 0.4)]

    out = ed.build_patch_hamiltonian(
        vx, vy, Bx, By, n_full, DIM, RE, threshold_amr=0.15, use_v2=True)

    assert len(out) == 3, f"{len(out)} valeurs rendues, 3 documentees"
    hamilt_params, score_vqa, full_score = out

    for key in ("H_edges", "C_edges", "K_plaquettes"):
        assert key in hamilt_params, f"{key} absent de hamilt_params"

    assert score_vqa.shape == (DIM, DIM), (
        f"score_vqa : {score_vqa.shape}, attendu {(DIM, DIM)} — c'est le "
        f"score REDUIT, pas celui a pleine resolution")
    assert full_score.shape == (n_full, n_full), (
        f"full_score : {full_score.shape}, attendu {(n_full, n_full)}")

    # Le champ qui SEPARE : block_max sur un score non constant rend une
    # reduction strictement au-dessus de la moyenne des blocs, donc les deux
    # grandeurs ne peuvent pas etre confondues par egalite numerique.
    patch = n_full // DIM
    blocks = full_score.reshape(DIM, patch, DIM, patch)
    assert np.allclose(score_vqa, blocks.max(axis=(1, 3))), (
        "score_vqa n'est pas le block_max de full_score")
