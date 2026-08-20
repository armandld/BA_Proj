"""D-144 — la décision D-47 gardée par le COMPORTEMENT de `run_phase5`.

`tests/study/test_phase5_ne_filtre_plus_sur_promising.py` garde les deux
moitiés de la décision de USER sur D-47 — la phase 5 ne filtre plus sur
`promising`, et le compte reste imprimé comme diagnostic — en lisant le
SOURCE de `study/common/qaoa_inputs.py`. Deux de ses quatre assertions ne
peuvent pas rougir sur le comportement qu'elles annoncent garder :

  A'-1  le filtre revient AVANT la boucle (`snap_indices` réduit aux
        indices prometteurs), sans `if ... promising ...: continue` et sans
        toucher la ligne `for idx in range(len(snap_indices)):`
                                          -> le fichier reste **5 passed**
  A'-2  le `print(...)` du diagnostic est SUPPRIMÉ ; les deux jetons
        cherchés (`n_promising`, `diagnostic`) survivent dans le commentaire
        D-47 juste au-dessus
                                          -> le fichier reste **5 passed**

Le détecteur AST de ce fichier n'est pas en cause : il reconnaît bien la
forme qu'il cherche (son auto-test le vérifie). Ce qu'il ne voit pas, c'est
que **filtrer ne demande pas un `if`** — réduire l'itérable suffit.

Ce banc-ci exécute `run_phase5` de bout en bout sur des artefacts
synthétiques, avec les trois fonctions coûteuses remplacées : ce qui est
mesuré est la SÉLECTION des instantanés, pas le QAOA. ~0,1 s.

Le champ d'essai SÉPARE : `promising = [True, False, True]`. Sur un
`promising` tout-à-True — la valeur mesurée en production, 40/40 — un
filtre réintroduit serait invisible, et ce banc ne mesurerait rien.
"""

import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# Import direct, jamais `importorskip` : la règle écrite du dépôt
# (en-tête de `tests/study/test_fig0_pareto_paths.py`, D-94) est qu'un module
# qu'on ne peut pas importer doit rendre la suite ROUGE, pas verte-avec-skip.
from study.common import qaoa_inputs


_N = 8              # grille DNS ; aucun calcul physique n'est fait dessus
_DIM = 2            # patches par côté
_N_SNAP = 3
_PROMISING = [True, False, True]


@pytest.fixture
def artefacts(tmp_path):
    """Trois instantanés, dont **un seul** est `promising=False`.

    C'est l'entrée qui SÉPARE : elle distingue « la phase 5 traite tous les
    instantanés » de « la phase 5 traite les prometteurs ». Un `promising`
    constant ne distinguerait rien.
    """
    assert not all(_PROMISING), (
        "champ d'essai vide : avec un `promising` constant, un filtre "
        "réintroduit rendrait exactement le même résultat")

    rng = np.random.default_rng(0)
    champ = rng.standard_normal((_N_SNAP, _N, _N))

    dns = tmp_path / "dns.npz"
    np.savez(dns, vx=champ, vy=champ, Bx=champ, By=champ,
             meta_Re=400, meta_scenario="banc_d144")

    patches = tmp_path / "patches.npz"
    np.savez(patches,
             l2_errors=rng.random((_N_SNAP, _DIM, _DIM)),
             l2_threshold=0.5,
             is_hard=np.zeros((_N_SNAP, _DIM, _DIM), dtype=bool))

    ed = tmp_path / "ed.npz"
    np.savez(ed,
             promising=np.array(_PROMISING),
             snap_indices=np.arange(_N_SNAP),
             decisions_h=np.zeros((_N_SNAP, _DIM, _DIM), dtype=bool),
             decisions_v=np.zeros((_N_SNAP, _DIM, _DIM), dtype=bool),
             gt_refine=np.zeros((_N_SNAP, _DIM, _DIM), dtype=bool))

    return str(dns), str(patches), str(ed)


@pytest.fixture
def traces(monkeypatch):
    """Les trois fonctions coûteuses remplacées, et ce qu'elles ont vu.

    On ne remplace RIEN de la sélection : la boucle, `promising`, et les
    tableaux qu'elle indexe restent ceux du module.
    """
    vus = {"prepare": 0, "qaoa": 0, "comparaisons": 0}

    def faux_prepare(vx, vy, Bx, By, N, n_patches, Re, use_v2=False):
        vus["prepare"] += 1
        return {"score_grid": np.zeros((n_patches, n_patches))}, {}, \
            np.zeros((n_patches, n_patches))

    def faux_qaoa(data_in, hamilt_params, n_patches, **kwargs):
        vus["qaoa"] += 1
        faux = np.zeros((n_patches, n_patches), dtype=bool)
        return np.zeros(n_patches * n_patches), faux, faux, None, 0.0

    def fausse_comparaison(*args, **kwargs):
        vus["comparaisons"] += 1
        m = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {"qaoa": m, "exact": m, "classical": m,
                "qaoa_exact_agreement": 1.0}

    monkeypatch.setattr(qaoa_inputs, "prepare_qaoa_inputs", faux_prepare)
    monkeypatch.setattr(qaoa_inputs, "run_qaoa_on_snapshot", faux_qaoa)
    monkeypatch.setattr(qaoa_inputs, "full_comparison", fausse_comparaison)
    return vus


# ══════════════════════════════════════════════════════════════════
#  1. La phase 5 traite TOUS les instantanés — mesuré, pas lu
# ══════════════════════════════════════════════════════════════════

def test_les_trois_instantanes_sont_traites(artefacts, traces):
    """Rougit sur A'-1, quelle que soit la FORME du filtre réintroduit.

    Ce n'est pas un `if` qu'on cherche : c'est le nombre d'instantanés qui
    ressortent. Réduire `snap_indices`, filtrer par compréhension, sauter
    par `continue` — les trois donnent 2 au lieu de 3.
    """
    dns, patches, ed = artefacts
    resultats, meta = qaoa_inputs.run_phase5(dns, patches, ed, _DIM)

    assert [r["snap_idx"] for r in resultats] == [0, 1, 2], (
        f"la phase 5 a rendu {[r['snap_idx'] for r in resultats]} au lieu "
        f"des trois instantanés : l'instantané 1 est le seul dont "
        f"`promising` est faux, donc il a été ÉCARTÉ. C'est D-47, dont la "
        f"décision de USER est que la phase 5 les traite tous")
    assert traces["qaoa"] == _N_SNAP, (
        f"{traces['qaoa']} appels QAOA pour {_N_SNAP} instantanés")
    assert meta["scenario"] == "banc_d144"


def test_le_diagnostic_est_imprime_avec_son_compte(artefacts, traces, capsys):
    """Rougit sur A'-2 : le `print` supprimé, les jetons cherchés survivant
    dans le commentaire au-dessus.

    L'assertion porte sur ce qui SORT, jamais sur ce qui est écrit dans le
    fichier — c'est toute la différence avec le garde qu'elle complète.
    """
    dns, patches, ed = artefacts
    qaoa_inputs.run_phase5(dns, patches, ed, _DIM)
    sortie = capsys.readouterr().out

    attendu = f"{sum(_PROMISING)}/{_N_SNAP} snapshots"
    assert attendu in sortie, (
        f"le diagnostic `{attendu}` n'est pas imprimé. Retirer le filtre "
        f"sans garder la MESURE perd le seul chiffre qui dira, après la "
        f"réoptimisation, que `promising` est redevenu informatif.\n"
        f"sortie :\n{sortie}")
    assert "D-47" in sortie, (
        "le compte est imprimé sans dire qu'il ne filtre plus : un lecteur "
        "de journal croira que des instantanés ont été écartés")


# ══════════════════════════════════════════════════════════════════
#  2. Le banc peut-il échouer ? — la question de VIGIL.md
# ══════════════════════════════════════════════════════════════════

def test_le_banc_rougit_si_un_instantane_est_ecarte(artefacts, traces,
                                                    monkeypatch):
    """Épingle le comportement d'AVANT la décision D-47, pour que la
    correction ne puisse pas être défaite en silence.

    On rejoue la boucle de phase 5 sur `snap_indices` réduit aux indices
    prometteurs — la forme exacte de la mutation A'-1 — et on exige que
    l'assertion du premier test tombe. Un banc dont on ne sait pas dire sur
    quelle entrée il échoue n'a rien prouvé.
    """
    dns, patches, ed = artefacts
    vrai_load = np.load

    def load_filtre(chemin, *a, **kw):
        brut = dict(vrai_load(chemin, *a, **kw))
        if "promising" in brut:
            garder = np.flatnonzero(brut["promising"])
            for cle in ("snap_indices", "decisions_h", "decisions_v",
                        "gt_refine"):
                brut[cle] = brut[cle][garder]
            brut["promising"] = brut["promising"][garder]
        return brut

    monkeypatch.setattr(np, "load", load_filtre)
    resultats, _ = qaoa_inputs.run_phase5(dns, patches, ed, _DIM)

    indices = [r["snap_idx"] for r in resultats]
    assert indices == [0, 2], (
        f"la mutation n'a pas écarté l'instantané non prometteur "
        f"({indices}) : le banc ne mesure donc pas ce qu'il annonce")
    assert traces["qaoa"] == 2
