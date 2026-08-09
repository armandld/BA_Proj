"""Load trained hyperparameters from best_hyperparams.json.

Usage:
    from hyperparams_loader import load_hyperparams

    # Default quantum params (used by pipeline automatically)
    params = load_hyperparams()

    # Default classical params
    params = load_hyperparams(method='classical')

    # Best params for a specific scenario
    params = load_hyperparams(scenario='kelvin_helmholtz')

    # Best params for a scenario combo
    params = load_hyperparams(combo='simple')  # or 'complex'

    # Raw training data: specific phase/lambda/rank
    params = load_hyperparams(phase='phase1', lambda_cost='lambda_0.40', rank=0)
"""
import json
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Les hyperparametres sont une SORTIE d'entrainement, rangee avec les autres
# resultats. L'ancien emplacement racine reste accepte en repli.
_DEFAULT_PATH = os.path.join(_PROJECT_ROOT, 'results', 'hyperparams',
                             'best_hyperparams.json')
if not os.path.isfile(_DEFAULT_PATH):
    _DEFAULT_PATH = os.path.join(_PROJECT_ROOT, 'best_hyperparams.json')


def load_hyperparams(path=None, method='quantum', scenario=None, combo=None,
                     phase=None, lambda_cost=None, rank=0):
    """Return a dict of hyperparameters from best_hyperparams.json.

    Parameters
    ----------
    path : str or None
        Path to the JSON file. Defaults to <project_root>/best_hyperparams.json.
    method : str
        'quantum' or 'classical'. Selects the parameter set type.
    scenario : str or None
        Specific scenario name (e.g. 'kelvin_helmholtz'). Returns best params
        for that scenario.
    combo : str or None
        Scenario combo: 'simple' (4 isolated) or 'complex' (OT + Rotor).
    phase : str or None
        training data from that phase/lambda/rank.
    lambda_cost : str or None
        Lambda key (e.g. 'lambda_0.40'). Used with phase.
    rank : int
        0 = best trial, 1 = second best, etc. Used with phase.

    Returns
    -------
    dict : hyperparameter name -> value
    """
    path = path or _DEFAULT_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"best_hyperparams.json not found at {path}. "
            f"Run: ./ExtractBestHyperparams.sh"
        )

    with open(path, 'r') as f:
        data = json.load(f)
        
    if phase is not None and lambda_cost is None:
        bpp = data.get('best_per_phase', {}).get(method, {})
        if phase in bpp:
            return dict(bpp[phase]['params'])

    # ── New structure (has 'default.quantum' or 'default.classical') ──
    if 'default' in data and isinstance(data['default'], dict) and \
       ('quantum' in data['default'] or 'classical' in data['default']):
        return _load_new_format(data, method, scenario, combo, phase, lambda_cost, rank)

    # ── Legacy structure (has 'default.params') ──
    return _load_legacy_format(data, phase, lambda_cost, rank)


def _load_new_format(data, method, scenario, combo, phase, lambda_cost, rank):
    """Load from the new structured JSON format."""

    # Priority 1: specific phase/lambda/rank (raw training data)
    if phase is not None:
        return _load_from_training_phases(data, method, phase, lambda_cost, rank)

    # Priority 2: per-scenario best
    if scenario is not None:
        per_scenario = data.get('per_scenario', {})
        if scenario not in per_scenario:
            available = list(per_scenario.keys())
            raise KeyError(f"Scenario '{scenario}' not found. Available: {available}")
        entry = per_scenario[scenario].get(method)
        if entry is None:
            raise KeyError(f"No {method} params for scenario '{scenario}'")
        return dict(entry['params'])

    # Priority 3: scenario combo
    if combo is not None:
        combos = data.get('scenario_combos', {})
        if combo not in combos:
            available = list(combos.keys())
            raise KeyError(f"Combo '{combo}' not found. Available: {available}")
        entry = combos[combo].get(method)
        if entry is None:
            raise KeyError(f"No {method} params for combo '{combo}'")
        return dict(entry['params'])

    # Priority 4: default
    default = data.get('default', {})
    entry = default.get(method)
    if entry is None:
        # Fallback: try the other method
        for m in ['quantum', 'classical']:
            if m in default:
                entry = default[m]
                break
    if entry is None:
        raise KeyError(f"No default {method} params found in JSON")
    return dict(entry['params'])


def _load_from_training_phases(data, method, phase, lambda_cost, rank):
    """Load from training_phases section (raw phase/lambda/rank)."""
    tp = data.get('training_phases', {})
    method_phases = tp.get(method, {})
    if not method_phases:
        # Legacy: phases at top level
        method_phases = data.get('phases', {})
    if phase not in method_phases:
        available = list(method_phases.keys())
        raise KeyError(f"Phase '{phase}' not in {method} training. Available: {available}")

    lambdas = method_phases[phase]
    if lambda_cost is None:
        # Pick first available lambda
        lambda_cost = sorted(lambdas.keys())[0]
    if lambda_cost not in lambdas:
        available = list(lambdas.keys())
        raise KeyError(f"Lambda '{lambda_cost}' not in {phase}. Available: {available}")

    trials = lambdas[lambda_cost]
    if rank >= len(trials):
        raise IndexError(f"Rank {rank} requested but only {len(trials)} trials available")

    return dict(trials[rank]['params'])


def _load_legacy_format(data, phase, lambda_cost, rank):
    """Load from the legacy JSON format (backward compatibility)."""
    if phase is None and lambda_cost is None and rank == 0:
        return dict(data['default']['params'])

    phase = phase or data['default']['phase']
    lambda_cost = lambda_cost or data['default']['lambda_cost']

    phases = data.get('phases', {})
    if phase not in phases:
        available = list(phases.keys())
        raise KeyError(f"Phase '{phase}' not found. Available: {available}")

    lambdas = phases[phase]
    if lambda_cost not in lambdas:
        available = list(lambdas.keys())
        raise KeyError(f"Lambda '{lambda_cost}' not in {phase}. Available: {available}")

    trials = lambdas[lambda_cost]
    if rank >= len(trials):
        raise IndexError(f"Rank {rank} requested but only {len(trials)} trials available")

    return dict(trials[rank]['params'])
