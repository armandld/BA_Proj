#!/usr/bin/env python3
"""
V4 - Statistiques confirmatoires (audit, Priorite 0).

Le protocole v3 fournit deja le bootstrap par blocs au niveau trajectoire
(`study/v3/stats.py` : bootstrap_by_trajectory, paired_delta_bootstrap).
L'audit demande trois briques supplementaires, ajoutees ici SANS redefinir
les fonctions v3 (elles sont importees et reutilisees telles quelles) :

  1. bootstrap HIERARCHIQUE classe -> regime -> seed, qui respecte la
     structure d'echantillonnage a trois niveaux (une trajectoire n'est pas
     un tirage independant a l'interieur d'une classe d'instabilite) ;
  2. correction de HOLM-Bonferroni pour la multiplicite (scenarios et
     endpoints secondaires), avec la garantie FWER ;
  3. test d'EQUIVALENCE (TOST) pour les claims de parite : une difference
     non significative ne prouve pas l'equivalence ; il faut une marge
     definie avant calcul.

Toutes les fonctions sont pures (numpy) et testables sans donnees DNS.
"""
import numpy as np

__all__ = [
    "hierarchical_bootstrap",
    "holm_correction",
    "tost_equivalence",
    "paired_hierarchical_delta",
]


def _resample_nested(rng, groups):
    """Tire un echantillon hierarchique : niveaux imbriques avec remise.

    `groups` est un dict {cle_niveau1: {cle_niveau2: [valeurs...]}}.
    On tire les cles de niveau 1 avec remise, puis, pour chaque cle tiree,
    ses cles de niveau 2 avec remise, puis les valeurs avec remise.
    """
    lvl1 = list(groups.keys())
    out = []
    for k1 in rng.choice(len(lvl1), size=len(lvl1), replace=True):
        sub = groups[lvl1[k1]]
        lvl2 = list(sub.keys())
        for k2 in rng.choice(len(lvl2), size=len(lvl2), replace=True):
            vals = np.asarray(sub[lvl2[k2]], dtype=float)
            if len(vals) == 0:
                continue
            idx = rng.integers(0, len(vals), size=len(vals))
            out.append(vals[idx])
    return np.concatenate(out) if out else np.array([])


def _build_nested(values, class_ids, regime_ids):
    groups = {}
    values = np.asarray(values, dtype=float).ravel()
    class_ids = np.asarray(class_ids).ravel()
    regime_ids = np.asarray(regime_ids).ravel()
    if not (len(values) == len(class_ids) == len(regime_ids)):
        raise ValueError("values, class_ids and regime_ids must align")
    if len(values) == 0:
        raise ValueError("empty input")
    for v, c, r in zip(values, class_ids, regime_ids):
        groups.setdefault(c, {}).setdefault(r, []).append(v)
    return groups


def hierarchical_bootstrap(values, class_ids, regime_ids, B=1000,
                           statistic=np.mean, seed=0, ci=95.0):
    """Bootstrap hierarchique classe -> regime -> (seed/trajectoire).

    Respecte la structure demandee par l'audit : la variance inter-classes
    domine et ne doit pas etre diluee en traitant chaque trajectoire comme
    un tirage independant.

    Retourne dict(estimate, ci_low, ci_high, boot, n_class, n_traj).
    """
    groups = _build_nested(values, class_ids, regime_ids)
    rng = np.random.default_rng(seed)
    boot = np.empty(B, dtype=float)
    for b in range(B):
        sample = _resample_nested(rng, groups)
        boot[b] = statistic(sample) if len(sample) else np.nan
    alpha = (100.0 - ci) / 2.0
    lo, hi = np.nanpercentile(boot, [alpha, 100.0 - alpha])
    return dict(
        estimate=float(statistic(np.asarray(values, dtype=float).ravel())),
        ci_low=float(lo), ci_high=float(hi), boot=boot,
        n_class=len(groups),
        n_traj=int(sum(len(v) for s in groups.values() for v in s.values())),
    )


def paired_hierarchical_delta(values_a, values_b, class_ids, regime_ids,
                              B=1000, statistic=np.mean, seed=0, ci=95.0):
    """Variante appariee : delta = a - b, meme structure hierarchique.

    L'appariement est par observation (meme classe, meme regime, meme seed),
    conformement a la regle d'appariement de l'audit.
    """
    a = np.asarray(values_a, dtype=float).ravel()
    b = np.asarray(values_b, dtype=float).ravel()
    if len(a) != len(b):
        raise ValueError("values_a and values_b must align")
    res = hierarchical_bootstrap(a - b, class_ids, regime_ids, B=B,
                                 statistic=statistic, seed=seed, ci=ci)
    res["frac_positive"] = float(np.mean((a - b) > 0))
    return res


def holm_correction(pvalues, alpha=0.05):
    """Procedure descendante de Holm-Bonferroni (controle du FWER).

    Retourne dict(p_adjusted, reject, alpha). Les p ajustes sont rendus
    monotones (cumulative max) comme l'exige la procedure.
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    m = len(p)
    if m == 0:
        return dict(p_adjusted=np.array([]), reject=np.array([], dtype=bool),
                    alpha=alpha)
    order = np.argsort(p)
    p_sorted = p[order]
    adj_sorted = np.maximum.accumulate(
        (m - np.arange(m)) * p_sorted)
    adj_sorted = np.minimum(adj_sorted, 1.0)
    p_adj = np.empty(m, dtype=float)
    p_adj[order] = adj_sorted
    return dict(p_adjusted=p_adj, reject=p_adj <= alpha, alpha=alpha)


def tost_equivalence(a, b, margin, paired=True, alpha=0.05):
    """Two One-Sided Tests : teste l'EQUIVALENCE de a et b a +/- margin.

    H0 : |mu_a - mu_b| >= margin  (non-equivalence)
    H1 : |mu_a - mu_b| <  margin  (equivalence)
    Le rejet de H0 (p_tost <= alpha) est la seule preuve valide de parite ;
    un test de difference non significatif ne prouve rien.

    Retourne dict(diff, p_lower, p_upper, p_tost, equivalent, margin, df).
    """
    from scipy import stats

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if margin <= 0:
        raise ValueError("margin must be > 0 and fixed before computation")
    if paired:
        if len(a) != len(b):
            raise ValueError("paired TOST requires equal lengths")
        d = a - b
        n = len(d)
        if n < 2:
            raise ValueError("need at least 2 paired observations")
        diff = float(np.mean(d))
        se = float(np.std(d, ddof=1) / np.sqrt(n))
        df = n - 1
    else:
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            raise ValueError("need at least 2 observations per group")
        diff = float(np.mean(a) - np.mean(b))
        va, vb = np.var(a, ddof=1) / na, np.var(b, ddof=1) / nb
        se = float(np.sqrt(va + vb))
        df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    if se == 0.0:
        equivalent = abs(diff) < margin
        return dict(diff=diff, p_lower=0.0 if equivalent else 1.0,
                    p_upper=0.0 if equivalent else 1.0,
                    p_tost=0.0 if equivalent else 1.0,
                    equivalent=bool(equivalent), margin=float(margin), df=df)
    t_lower = (diff + margin) / se     # H0: diff <= -margin
    t_upper = (diff - margin) / se     # H0: diff >= +margin
    p_lower = float(stats.t.sf(t_lower, df))
    p_upper = float(stats.t.cdf(t_upper, df))
    p_tost = max(p_lower, p_upper)
    return dict(diff=diff, p_lower=p_lower, p_upper=p_upper, p_tost=p_tost,
                equivalent=bool(p_tost <= alpha), margin=float(margin), df=df)
