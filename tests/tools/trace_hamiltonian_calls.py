"""Plugin pytest : compte les appels REELS aux deux constructeurs d'Hamiltonien.

Repond a la cinquieme question de `VIGIL.md` — *un test emprunte-t-il cette
configuration ?* — pour l'axe **bord du patch** de `VIGIL_BA_Proj.md`, qui a
deux valeurs servies par deux constructeurs distincts :

    create_period_hamiltonian   <- period_bound=True,  depth == 0
    create_bounded_hamiltonian  <- period_bound=False, depth  > 0

Lire les tests ne repond pas : `mapping()` choisit le constructeur a
l'execution, depuis `depth`. Il faut compter.

Ce n'est PAS un test : rien ici ne peut echouer. C'est un instrument de
mesure, range dans `tests/tools/` avec les autres (`diag_*.py`) — dossier que
`test_suite_integrity.py` exclut explicitement de ses balayages.

Usage :

    TRACE_OUT=/tmp/trace.json \\
      python -m pytest tests/ -q -m "not slow" -p trace_hamiltonian_calls

    # necessite que `tests/tools/` soit sur le PYTHONPATH, ou :
    python -m pytest tests/ -q -p tests.tools.trace_hamiltonian_calls

Le fichier JSON rend, pour chaque constructeur, le nombre d'appels par `dim` :
un `dim` absent est un `dim` qu'aucun test n'a traverse.

Mesure du 16 aout (`tests/pipeline tests/amr`, `-m "not slow"`) :
`{"bounded": {"2": 24}, "period": {"2": 60}}` — les deux cotes de l'axe sont
empruntes, mais a `dim = 2` seulement.
"""

import atexit
import collections
import json
import os

import VQA.cost_hamiltonian as ch
import VQA.mapping as mp

STATS = {'bounded': collections.Counter(), 'period': collections.Counter()}

_orig_bounded = ch.create_bounded_hamiltonian
_orig_period = ch.create_period_hamiltonian


def _wrap_bounded(hamilt_params, dim, *args, **kwargs):
    STATS['bounded'][int(dim)] += 1
    return _orig_bounded(hamilt_params, dim, *args, **kwargs)


def _wrap_period(hamilt_params, dim, *args, **kwargs):
    STATS['period'][int(dim)] += 1
    return _orig_period(hamilt_params, dim, *args, **kwargs)


# Le plugin est charge AVANT la collecte, donc avant que les modules de test
# n'importent ces noms : un `from VQA.cost_hamiltonian import ...` dans un
# test recevra la version enrobee. `mapping` a deja importe les siens au
# chargement de ce fichier, d'ou la seconde paire d'affectations.
ch.create_bounded_hamiltonian = _wrap_bounded
ch.create_period_hamiltonian = _wrap_period
mp.create_bounded_hamiltonian = _wrap_bounded
mp.create_period_hamiltonian = _wrap_period


@atexit.register
def _dump():
    destination = os.environ.get('TRACE_OUT')
    if not destination:
        return
    with open(destination, 'w', encoding='utf-8') as fh:
        json.dump({famille: {str(dim): n for dim, n in compteur.items()}
                   for famille, compteur in STATS.items()}, fh, indent=1)
