#!/usr/bin/env python3
"""
V3 Task 2 - Module de metriques (protocole v3, section 1.4 et 1.3-B3).

Fonctions pures (numpy/scipy/sklearn uniquement, pas de pile qiskit) :

  - captured_error_at_budget(scores, e, budgets) :
      metrique primaire CE(b). On classe les patches par score
      decroissant, on raffine les top-ceil(b*n), et
      CE(b) = somme des e_i raffines / somme totale des e_i.
      Retourne aussi l'AUC de la courbe CE(b) complete (trapezes sur
      [0, 1] avec le point CE(0) = 0).
  - ce_curve(scores, e) : la courbe complete, CE(k/n) pour k = 1..n.
  - spearman(scores, e) : rho de Spearman (metrique secondaire).
  - degeneracy_floors(prevalence) : planchers de la section 1.3-B3 :
      refine-all  F1 = 2p/(1+p)   (tout predire positif)
      refine-none F1 = 0          (ne rien predire positif)
  - degeneracy_flag(pred, prevalence, tol=0.005, gt=None) :
      drapeau 1.3-B3 "methode a moins de tol d'un plancher".
      Avec gt : distance du F1 realise aux planchers.
      Sans gt : prediction (quasi) constante, dont le F1 est au
      plancher par construction.
"""
import numpy as np

# np.trapz retire dans numpy 2.0 (renomme trapezoid)
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def ce_curve(scores, e):
    """Courbe CE complete : CE(k/n) pour k = 1..n.

    Classement par score decroissant (argsort stable : les ex aequo
    sont departages par l'indice, deterministe). Si la somme des e est
    nulle, retourne un tableau de NaN.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    e = np.asarray(e, dtype=float).ravel()
    if scores.shape != e.shape:
        raise ValueError("scores and e must have the same length")
    order = np.argsort(-scores, kind="stable")
    total = e.sum()
    if total <= 0:
        return np.full(len(e), np.nan)
    return np.cumsum(e[order]) / total


def captured_error_at_budget(scores, e, budgets=(0.10, 0.25, 0.50)):
    """Metrique primaire (section 1.4). Retourne ({b: CE(b)}, auc).

    CE(b) = somme des e des top-ceil(b*n) patches (par score) / somme
    totale. AUC = integrale par trapezes de la courbe CE(b) sur [0, 1],
    avec CE(0) = 0 prefixe.
    """
    n = len(np.asarray(e).ravel())
    curve = ce_curve(scores, e)
    ce = {}
    for b in budgets:
        if not 0.0 < b <= 1.0:
            raise ValueError(f"budget must be in (0, 1], got {b}")
        k = min(n, int(np.ceil(b * n)))
        ce[float(b)] = float(curve[k - 1])
    x = np.concatenate([[0.0], np.arange(1, n + 1) / n])
    y = np.concatenate([[0.0], curve])
    auc = float(_trapz(y, x))
    return ce, auc


def spearman(scores, e):
    """Rho de Spearman entre le score de la methode et e_i (ou d_i)."""
    from scipy.stats import spearmanr
    return float(spearmanr(np.asarray(scores).ravel(),
                           np.asarray(e).ravel())[0])


def degeneracy_floors(prevalence):
    """Planchers 1.3-B3 pour une prevalence p du label binaire."""
    p = float(prevalence)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"prevalence must be in [0, 1], got {p}")
    return {"refine_all": 2.0 * p / (1.0 + p), "refine_none": 0.0}


def degeneracy_flag(pred, prevalence, tol=0.005, gt=None):
    """Drapeau 1.3-B3 : la methode opere-t-elle a un plancher ?

    - gt fourni : True si le F1 realise est a moins de tol d'un des deux
      planchers (la methode est alors exclue des decomptes win/loss).
    - gt absent : True si la prediction est (quasi) constante - tout
      positif ou tout negatif a tol pres - son F1 etant alors au
      plancher par construction.
    """
    pred = np.asarray(pred).astype(int).ravel()
    floors = degeneracy_floors(prevalence)
    if gt is not None:
        from sklearn.metrics import f1_score
        f1 = f1_score(np.asarray(gt).astype(int).ravel(), pred,
                      zero_division=0)
        return bool(min(abs(f1 - floors["refine_all"]),
                        abs(f1 - floors["refine_none"])) <= tol)
    rate = pred.mean()
    return bool(rate <= tol or rate >= 1.0 - tol)


def threshold_transfer_flag(gt, proba, threshold, auc_floor=0.70, tol=0.005):
    """Un F1 nul vient-il d'une ABSENCE DE SIGNAL ou d'un SEUIL NON TRANSFERE ?

    Les deux rendent la meme chose — une prediction constante et un F1 au
    plancher — et le protocole §1.3-B3 les traite deja de la meme facon :
    `degeneracy_flag` les exclut des decomptes. Mais ce ne sont pas le meme
    fait, et les confondre a coute une lecture.

    Cas mesure (pli `harris_tearing`, LOSO, dim=16) : les probabilites du
    GBT y plafonnent a 0.124 tandis que le seuil ajuste sur les scenarios
    d'entrainement vaut 0.400. Aucun positif n'est predit sur 20 480
    cellules, donc F1 = 0.000 — alors que l'AUC vaut 0.908 et que le F1 a
    budget appaire vaut 0.659. Le CLASSEMENT est bon ; c'est l'operateur de
    decision qui ne traverse pas la frontiere de scenario.

    Renvoie un dict :
      degenerate            la prediction est-elle constante ?
      auc                   qualite du CLASSEMENT, independante du seuil
      verdict               'ok' | 'aucun_signal' | 'seuil_non_transfere'
      proba_max, threshold  les deux nombres qui expliquent le verdict

    'seuil_non_transfere' n'est PAS un resultat sur la tache : c'est un
    defaut de l'operateur de mesure, et le nombre a citer est alors le F1 a
    budget appaire, pas le F1 au seuil.
    """
    from sklearn.metrics import roc_auc_score

    gt = np.asarray(gt).astype(int).ravel()
    proba = np.asarray(proba, dtype=float).ravel()
    if gt.shape != proba.shape:
        raise ValueError(f"gt {gt.shape} et proba {proba.shape} ne coincident pas")

    pred = (proba > float(threshold)).astype(int)
    rate = pred.mean()
    degenerate = bool(rate <= tol or rate >= 1.0 - tol)

    try:
        auc = float(roc_auc_score(gt, proba))
    except ValueError:          # une seule classe dans gt
        auc = float("nan")

    if not degenerate:
        verdict = "ok"
    elif np.isnan(auc) or auc < auc_floor:
        verdict = "aucun_signal"
    else:
        verdict = "seuil_non_transfere"

    return {"degenerate": degenerate, "auc": auc, "verdict": verdict,
            "proba_max": float(proba.max()) if proba.size else float("nan"),
            "threshold": float(threshold), "positive_rate": float(rate)}
