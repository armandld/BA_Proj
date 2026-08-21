"""D-181 -- `bisect_threshold_for_budget` (`closed_loop_budget_matched.py`,
T15b) pique une entree `NaN` de la trace comme "meilleure" au lieu de
continuer a chercher, si l'evaluation au bord BAS de la fourchette
(`lo=0.05`) echoue.

Cause : `min(trace, key=lambda r: abs(r["patch_ratio"] - target_patch))`.
Les comparaisons de flottants avec NaN sont TOUJOURS fausses en Python :
`x < nan` et `nan < x` valent False quel que soit `x`. `min()` construit
son candidat en gardant le PREMIER element et ne le remplace que sur une
comparaison strictement VRAIE. Si l'evaluation `_eval(lo)` -- la toute
premiere du trace, toujours executee en premier -- rend un `patch_ratio`
NaN (le solveur diverge, `run_arm` echoue), ce candidat NaN reste "best"
JUSQU'A LA FIN, quel que soit le nombre d'iterations de bissection qui
suivent et aussi bon que soit leur resultat.

Mesure sur les 4 artefacts publies (`results/t15b_budget_matched_{ot,kh,
rotor,tearing}.json`, 778255d) : aucun `patch_ratio` n'y est NaN
aujourd'hui -- le defaut est LATENT, pas manifeste dans les lectures
publiees actuelles. Mais ces memes artefacts sont la reference
"budget-matched" citee dans `docs/RESULTS.md` a travers T15b, T19, T20,
T23 et `figures/pareto_frontier.py` : un futur re-run (campagne en cours,
`armandld/desire#38`) qui rencontre UNE SEULE evaluation solveur divergente
au bord bas de la fourchette produirait un artefact dont `matched_classical`
et `delta_phys_matched` seraient silencieusement des NaN, avec une phrase
`READING` imprimee comme si la comparaison avait reussi (`nan < -0.01` vaut
False, donc la branche "recovers the fidelity" s'imprime).

Rapport seul dans `docs/DEFAUTS.md` (D-181) : la correction touche
`study/closed_loop/closed_loop_budget_matched.py`, gele pendant la
campagne. Ce test EPINGLE le comportement actuel (faux) pour qu'une
correction future soit mesuree et pas glissee en silence -- il doit
ECHOUER le jour ou quelqu'un filtre les NaN avant le `min()`.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import closed_loop_budget_matched as m  # noqa: E402


def _fake_run_arm_nan_at_lo(patch_at_lo, patch_slope):
    """Simule un `run_arm` ou l'evaluation au bord bas (thr==lo) echoue
    (NaN), et le reste de la fourchette repond normalement, decroissant en
    threshold (comme le vrai `patch_ratio`)."""
    def _run(T, key, cfg, dns_held, hp, is_classical,
             lambda_cost=None, verbose=False):
        thr = hp["threshold_amr"]
        if abs(thr - 0.05) < 1e-12:
            return dict(patch_ratio=float("nan"), phys_score=float("nan"),
                        combined=float("nan"), wall_s=0.0)
        pr = max(0.0, min(1.0, patch_slope(thr)))
        return dict(patch_ratio=pr, phys_score=1.0 - pr, combined=0.0,
                    wall_s=0.0)
    return _run


def test_nan_at_low_bracket_is_pinned_as_best_today(monkeypatch):
    """Comportement ACTUEL (defaut) : une seule evaluation NaN au bord bas
    de la fourchette contamine tout le resultat, meme apres 4 iterations
    de bissection convergeant proprement ailleurs."""
    monkeypatch.setattr(
        m, "run_arm",
        _fake_run_arm_nan_at_lo(None, lambda thr: 1.0 - thr))

    best, trace = m.bisect_threshold_for_budget(
        None, "x", None, None, {}, target_patch=0.5,
        lo=0.05, hi=0.80, max_iter=4, tol=0.02)

    # Le trace CONTIENT une entree quasi-parfaite (thr proche de 0.5,
    # patch proche de 0.5) -- la bissection a bien converge.
    finite = [r for r in trace if np.isfinite(r["patch_ratio"])]
    best_finite = min(finite, key=lambda r: abs(r["patch_ratio"] - 0.5))
    assert abs(best_finite["patch_ratio"] - 0.5) <= 0.02, (
        "precondition : la bissection doit converger sur les evaluations "
        f"finies pour que ce test separe -- trouve {best_finite}")

    # Comportement mesure aujourd'hui : `best` est l'entree NaN du bord
    # bas, PAS l'entree convergee. C'est le defaut D-181.
    assert np.isnan(best["patch_ratio"]), (
        "le defaut D-181 semble corrige (min() ignore desormais le NaN) -- "
        "RE-MESURER et deplacer l'entree D-181 vers docs/RESULTS.md avant "
        "de considerer ce test comme un faux negatif")
    assert best["threshold"] == pytest.approx(0.05)
    assert best is not best_finite


def test_delta_phys_and_reading_are_corrupted_by_the_nan_best(monkeypatch):
    """Consequence bout-en-bout : `delta_phys_matched` (le nombre ecrit
    dans le json publie) et la phrase READING derivent du `best` NaN, pas
    de la bonne evaluation -- silencieusement, sans avertissement affiche."""
    monkeypatch.setattr(
        m, "run_arm",
        _fake_run_arm_nan_at_lo(None, lambda thr: 1.0 - thr))

    best, trace = m.bisect_threshold_for_budget(
        None, "x", None, None, {}, target_patch=0.5,
        lo=0.05, hi=0.80, max_iter=4, tol=0.02)

    qhas_phys_score = 0.42  # valeur arbitraire, seul le calcul compte
    d_phys = qhas_phys_score - best["phys_score"]
    assert np.isnan(d_phys), (
        "delta_phys_matched (le champ ecrit dans "
        "results/t15b_budget_matched_{fold}.json) devient NaN des que la "
        "premiere evaluation echoue -- silencieusement, sans warning "
        "imprime au-dela du controle de bracket initial")

    # La phrase READING imprimee par main() teste `d_phys < -0.01` : NaN
    # comparee a quoi que ce soit vaut toujours False, donc la branche
    # "recovers the fidelity" s'imprime comme si la comparaison avait
    # reussi, sans jamais mentionner NaN.
    reading_says_recovers = not (d_phys < -0.01)
    assert reading_says_recovers, (
        "la phrase READING imprimerait desormais le bon message d'echec -- "
        "remesurer D-181")
