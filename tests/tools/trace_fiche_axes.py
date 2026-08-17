"""Plugin pytest : mesure lesquels des axes de `VIGIL_BA_Proj.md` la suite emprunte.

La cinquieme question de `VIGIL.md` dit qu'un module n'est pas audite parce que
ses fonctions ont ete lues, mais quand un test traverse **chacune des
configurations que le code emprunte en production**. La fiche du depot liste
sept axes, chacun a deux valeurs. Lire les tests ne dit pas lesquelles sont
traversees : les deux cotes se choisissent a l'execution, depuis `depth`,
`backend_name`, `method`, `warm_start_params`, `classical_only`.

Ce plugin enrobe les points de decision et compte. Un cote a **0** est un cote
qu'aucun test n'emprunte — c'est la reponse, et elle ne se devine pas.

Ce n'est PAS un test : rien ici ne peut echouer. C'est un instrument, range
dans `tests/tools/` avec les autres (`diag_*.py`), dossier que
`test_suite_integrity.py` exclut de ses balayages.

Usage :

    TRACE_OUT=/tmp/axes.json PYTHONPATH=src:tests/tools \\
      python -m pytest tests/ -q -m "not slow" -p trace_fiche_axes
"""

import atexit
import collections
import json
import os

import numpy as np

import VQA.cost_hamiltonian as ch
import VQA.execute as ex
import VQA.mapping as mp

AXES = collections.defaultdict(collections.Counter)

_orig_bounded = ch.create_bounded_hamiltonian
_orig_period = ch.create_period_hamiltonian
_orig_execute = ex.execute


def _wrap_bounded(hamilt_params, dim, *args, **kwargs):
    AXES['bord_du_patch']['borne'] += 1
    AXES['dim'][str(int(dim))] += 1
    return _orig_bounded(hamilt_params, dim, *args, **kwargs)


def _wrap_period(hamilt_params, dim, *args, **kwargs):
    AXES['bord_du_patch']['periodique'] += 1
    AXES['dim'][str(int(dim))] += 1
    return _orig_period(hamilt_params, dim, *args, **kwargs)


def _wrap_execute(qc, cost_hamiltonian, mode, backend_name, shots, reps,
                  K_opt, eps, E_max, verbose, vqa_runtime=None,
                  method="COBYLA", warm_start_params=None):
    AXES['backend'][str(backend_name)] += 1
    AXES['optimiseur'][str(method)] += 1
    AXES['warm_start']['present' if warm_start_params is not None
                       else 'absent'] += 1
    try:
        nul = bool(np.allclose(np.abs(cost_hamiltonian.coeffs), 0.0))
    except Exception:                       # noqa: BLE001 - instrument, pas test
        nul = False
    AXES['hamiltonien']['nul' if nul else 'non_nul'] += 1
    return _orig_execute(qc, cost_hamiltonian, mode, backend_name, shots, reps,
                         K_opt, eps, E_max, verbose, vqa_runtime=vqa_runtime,
                         method=method, warm_start_params=warm_start_params)


ch.create_bounded_hamiltonian = _wrap_bounded
ch.create_period_hamiltonian = _wrap_period
mp.create_bounded_hamiltonian = _wrap_bounded
mp.create_period_hamiltonian = _wrap_period
ex.execute = _wrap_execute

# `pipeline` importe `execute` par son nom : il faut enrober la reference que
# le module appelant detient, pas seulement celle de `VQA.execute`.
try:
    import pipeline as _pipeline
    if hasattr(_pipeline, 'execute'):
        _pipeline.execute = _wrap_execute

    _orig_pipeline_fn = _pipeline.pipeline

    def _wrap_pipeline(*args, **kwargs):
        AXES['bras']['classique' if kwargs.get('classical_only')
                     else 'quantique'] += 1
        return _orig_pipeline_fn(*args, **kwargs)

    _pipeline.pipeline = _wrap_pipeline
except Exception:                           # noqa: BLE001 - instrument, pas test
    pass


@atexit.register
def _dump():
    destination = os.environ.get('TRACE_OUT')
    if not destination:
        return
    with open(destination, 'w', encoding='utf-8') as fh:
        json.dump({axe: dict(compteur) for axe, compteur in AXES.items()},
                  fh, indent=1, sort_keys=True)
