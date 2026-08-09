#!/usr/bin/env python3
"""
V3 Task 3 - Bootstrap par blocs au niveau trajectoire (protocole v3,
section 1.5).

L'unite de replication statistique est la TRAJECTOIRE (un scenario x Re
x graine physique), jamais le snapshot : le re-echantillonnage au niveau
snapshot est interdit pour les chiffres titres (autocorrelation
temporelle). Fonctions pures (numpy uniquement) :

  - bootstrap_by_trajectory(values, traj_ids, B=1000, ...) :
      re-echantillonne les trajectoires avec remise, recalcule la
      statistique sur les valeurs regroupees de chaque tirage, CI par
      percentiles.
  - paired_delta_bootstrap(values_a, values_b, traj_ids, B=1000, ...) :
      variante appariee (§1.5) : delta par trajectoire
      delta_t = stat(a_t) - stat(b_t), puis bootstrap des trajectoires ;
      rapporte le delta moyen, la CI 95 % par percentiles et la fraction
      de trajectoires avec delta > 0.
"""
import numpy as np


def _group_by_trajectory(values, traj_ids):
    values = np.asarray(values, dtype=float).ravel()
    traj_ids = np.asarray(traj_ids).ravel()
    if len(values) != len(traj_ids):
        raise ValueError("values and traj_ids must have the same length")
    if len(values) == 0:
        raise ValueError("empty input")
    uids = np.unique(traj_ids)
    return uids, [values[traj_ids == u] for u in uids]


def bootstrap_by_trajectory(values, traj_ids, B=1000, statistic=np.mean,
                            seed=0, ci=95.0):
    """Bootstrap par blocs trajectoire (§1.5), CI par percentiles.

    A chaque tirage, on echantillonne T identifiants de trajectoire avec
    remise et la statistique est recalculee sur la concatenation des
    valeurs des trajectoires tirees (avec multiplicite).

    Retourne dict(estimate, ci_low, ci_high, boot, n_traj).
    """
    uids, groups = _group_by_trajectory(values, traj_ids)
    T = len(uids)
    rng = np.random.default_rng(seed)
    boot = np.empty(B, dtype=float)
    for b in range(B):
        pick = rng.integers(0, T, size=T)
        boot[b] = statistic(np.concatenate([groups[i] for i in pick]))
    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(boot, [alpha, 100.0 - alpha])
    return dict(
        estimate=float(statistic(np.concatenate(groups))),
        ci_low=float(lo), ci_high=float(hi),
        boot=boot, n_traj=T,
    )


def paired_delta_bootstrap(values_a, values_b, traj_ids, B=1000,
                           statistic=np.mean, seed=0, ci=95.0):
    """Comparaison appariee au niveau trajectoire (§1.5).

    delta_t = statistic(a_t) - statistic(b_t) pour chaque trajectoire t
    (convention : methode A moins methode B). Le bootstrap
    re-echantillonne les identifiants de trajectoire avec remise et
    recalcule la moyenne des deltas ; CI 95 % par percentiles.

    Retourne dict(mean_delta, ci_low, ci_high, frac_positive, deltas,
    boot, n_traj). `frac_positive` = fraction des trajectoires avec
    delta > 0 (a rapporter avec le delta moyen et la CI).
    """
    _, groups_a = _group_by_trajectory(values_a, traj_ids)
    _, groups_b = _group_by_trajectory(values_b, traj_ids)
    deltas = np.array([statistic(ga) - statistic(gb)
                       for ga, gb in zip(groups_a, groups_b)])
    T = len(deltas)
    rng = np.random.default_rng(seed)
    boot = np.empty(B, dtype=float)
    for b in range(B):
        boot[b] = deltas[rng.integers(0, T, size=T)].mean()
    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(boot, [alpha, 100.0 - alpha])
    return dict(
        mean_delta=float(deltas.mean()),
        ci_low=float(lo), ci_high=float(hi),
        frac_positive=float((deltas > 0).mean()),
        deltas=deltas, boot=boot, n_traj=T,
    )
