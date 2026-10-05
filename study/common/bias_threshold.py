"""Seuil AMR calibre en F1, tenu a l'ecart du scenario evalue.

Le biais des deux mappeurs est centre sur `threshold_amr` : h_i est
proportionnel a (s_i - seuil). Le seuil deploye (0,15 pour V2, 0,1496 pour
V1) vient de l'entrainement classique historique, qui optimisait une perte
composite en boucle fermee, pas le F1 contre le label statique sur lequel
l'etude note les decisions. Sur les scores-blocs de `dim = 3` il marque 88
cellules sur 108 a raffiner, alors que le seuil qui maximise F1 sur les
scenarios d'entrainement de la replication confirmatoire vaut 0,51 a 0,60
(`study/h0_selection/h0_readout_threshold_diagnostic.py`).

Ce module recalcule ce seuil EXACTEMENT comme le bras classique de
`study/h2b_prediction/h2b_v2_hamiltonian_vs_gbt_loso.py` : meme selection
d'instantanes, meme score (maximum par bloc du score classique, chemin
`build_patch_hamiltonian`), meme label (L2 >= seuil du fichier de patches),
meme recherche `best_threshold_f1`, ajustee sur les trois AUTRES scenarios
canoniques, tous Reynolds confondus. Le scenario evalue n'est jamais vu :
c'est l'invariant de CLAUDE.md (aucun reglage ne voit le scenario tenu ni
ses labels). Un fichier d'entree manquant leve une erreur ; rien n'est saute
en silence.

Le score par bloc ne depend ni du mappeur ni du seuil : il est identique,
au bit pres, a `score_vqa` de `qaoa_inputs.prepare_qaoa_inputs` (verifie
sur 48 instantanes des 16 trajectoires, ecart maximal 0).
"""
import os

import numpy as np

#: Les quatre scenarios du protocole confirmatoire (`SCENARIOS_4`).
CANONICAL_SCENARIOS = ("harris_tearing", "kelvin_helmholtz", "mhd_rotor",
                       "orszag_tang")
#: Reynolds et nombre d'instantanes de la replication confirmatoire.
CALIBRATION_RE = (400, 800, 1200, 1600)
CALIBRATION_N_SNAPS = 10


def confirmatory_snapshot_indices(n_total, n_snaps):
    """Selection de `h2b_v2_hamiltonian_vs_gbt_loso.collect_rows`.

    `n_snaps` instantanes regulierement espaces a partir de l'indice 1
    (l'instantane 0 n'a pas de predecesseur pour psi).
    """
    step = max(1, (n_total - 1) // n_snaps)
    return list(range(1, n_total, step))[:n_snaps]


def _input_paths(scenario, re, N, dim, results_dir):
    dp = os.path.join(results_dir, f"dns_{scenario}_Re{re}_N{N}.npz")
    pp = os.path.join(results_dir,
                      f"patches_{scenario}_Re{re}_N{N}_dim{dim}.npz")
    missing = [p for p in (dp, pp) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"entrees absentes pour {scenario} Re={re} : "
            + ", ".join(os.path.basename(p) for p in missing)
            + ". Rien n'est saute en silence : un seuil ou un panel calcule "
            "sur un sous-ensemble des trajectoires ne serait plus celui "
            "qu'il annonce.")
    return dp, pp


def block_scores_and_labels(scenario, re, N, dim, results_dir, snap_indices):
    """Score-bloc classique et label statique d'une trajectoire.

    `snap_indices` est une liste d'indices, ou une fonction
    `n_total -> indices`. Retourne `(scores, labels, indices)` : deux
    tableaux `(len(indices), dim*dim)` (float, bool) et les indices retenus.
    """
    from config import V2_THRESHOLD
    from exact_diagonalisation import build_patch_hamiltonian

    dp, pp = _input_paths(scenario, re, N, dim, results_dir)
    dns = np.load(dp)
    pat = np.load(pp)
    n_total = len(dns["vx"])
    idx = (list(snap_indices(n_total)) if callable(snap_indices)
           else [int(i) for i in snap_indices])
    if not idx:
        raise ValueError(f"aucun instantane retenu pour {scenario} Re={re}")
    l2 = pat["l2_errors"]
    l2_thr = float(pat["l2_threshold"])
    scores, labels = [], []
    for si in idx:
        _coef, score_vqa, _full = build_patch_hamiltonian(
            dns["vx"][si].astype(np.float64), dns["vy"][si].astype(np.float64),
            dns["Bx"][si].astype(np.float64), dns["By"][si].astype(np.float64),
            N, dim, re, threshold_amr=V2_THRESHOLD, use_v2=True)
        scores.append(np.asarray(score_vqa, dtype=float).ravel())
        labels.append(np.asarray(l2[si] >= l2_thr, dtype=bool).ravel())
    return np.array(scores), np.array(labels, dtype=bool), idx


def loso_f1_thresholds(results_dir, N=96, dim=3,
                       scenarios=CANONICAL_SCENARIOS,
                       re_values=CALIBRATION_RE,
                       n_snaps=CALIBRATION_N_SNAPS):
    """Seuil F1-optimal du score classique, scenario par scenario tenu.

    Retourne `{scenario_tenu: (seuil, F1_entrainement)}`, chaque seuil etant
    ajuste par `best_threshold_f1` sur les seuls AUTRES scenarios de
    `scenarios`, tous `re_values` confondus.
    """
    from h2b_ceiling_random_split import best_threshold_f1

    scenarios = tuple(scenarios)
    if len(scenarios) < 2:
        raise ValueError("une calibration tenue a l'ecart exige au moins "
                         "deux scenarios")
    data = {}
    for sc in scenarios:
        for re in re_values:
            s, y, _idx = block_scores_and_labels(
                sc, re, N, dim, results_dir,
                lambda n: confirmatory_snapshot_indices(n, n_snaps))
            data[(sc, re)] = (s, y)
    out = {}
    for held in scenarios:
        train = [(s, re) for s in scenarios if s != held for re in re_values]
        S = np.concatenate([data[k][0].ravel() for k in train])
        Y = np.concatenate([data[k][1].ravel().astype(int) for k in train])
        thr, f1 = best_threshold_f1(S, Y)
        out[held] = (float(thr), float(f1))
    return out
