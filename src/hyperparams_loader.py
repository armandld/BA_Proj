"""Load one explicit, validated hyperparameter artifact.

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

    # Raw archived training data: specific phase/lambda/rank
    params = load_hyperparams(phase='phase1', lambda_cost='lambda_0.40', rank=0)
"""
import json
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_PATH = os.path.join(_PROJECT_ROOT, 'results', 'hyperparams',
                             'best_hyperparams.json')
_ENV_PATH = "QHAS_HYPERPARAMS_PATH"
_REQUIRED_QUANTUM = {
    "beta", "w_z_frac", "sigma", "beta_curl", "beta_xpoint",
    "gamma_hydro", "gamma_mag", "kappa", "relative_percentile",
    "threshold_amr",
}


def resolve_hyperparams_path(path=None):
    """Return the configured artifact path without loading it."""
    selected = path or os.environ.get(_ENV_PATH) or _DEFAULT_PATH
    return os.path.abspath(os.path.expanduser(selected))


def _load_phase_candidate(data, method, selectors):
    """Load a completed phase-1 candidate produced by the campaign launcher."""
    if any(value is not None for value in selectors):
        raise ValueError(
            "scenario/combo/phase selectors do not apply to a phase candidate")
    if method != "quantum":
        raise KeyError("a phase candidate contains only the quantum arm")
    if data.get("status") != "complete":
        raise RuntimeError(
            f"campaign candidate is {data.get('status', 'unlabelled')!r}, "
            "not complete")
    if not data.get("campaign_contract_sha256") or not data.get(
            "campaign_contract"):
        raise RuntimeError("campaign candidate has no verified campaign contract")
    params = data.get("result", {}).get("best_params")
    if not isinstance(params, dict):
        raise RuntimeError("campaign candidate has no deployable best_params")
    missing = sorted(_REQUIRED_QUANTUM - set(params))
    extra = sorted(set(params) - _REQUIRED_QUANTUM)
    if missing or extra:
        raise RuntimeError(
            f"campaign candidate parameter mismatch: missing={missing}, "
            f"extra={extra}")
    return dict(params)


def _load_progressive_export(data, method, selectors):
    """Load the final export written after progressive phases 1--3."""
    if any(value is not None for value in selectors):
        raise ValueError(
            "scenario/combo/phase selectors do not apply to a deploy export")
    params = data.get("deploy", {}).get(method)
    if not isinstance(params, dict):
        raise KeyError(f"deploy export has no {method!r} parameter set")
    return dict(params)


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
    path = resolve_hyperparams_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"best_hyperparams.json not found at {path}. "
            f"Set {_ENV_PATH} or provide path= explicitly."
        )

    with open(path, 'r') as f:
        data = json.load(f)
        
    selectors = (scenario, combo, phase, lambda_cost)
    if data.get("artifact") == "phase_candidate":
        return _load_phase_candidate(data, method, selectors)
    if "deploy" in data:
        return _load_progressive_export(data, method, selectors)

    if phase is not None and lambda_cost is None:
        bpp = data.get('best_per_phase', {}).get(method, {})
        if phase in bpp:
            return dict(bpp[phase]['params'])

    # ── New structure (has 'default.quantum' or 'default.classical') ──
    if 'default' in data and isinstance(data['default'], dict) and \
       ('quantum' in data['default'] or 'classical' in data['default']):
        return _load_new_format(data, method, scenario, combo, phase, lambda_cost, rank)

    raise ValueError(
        "unsupported hyperparameter artifact schema; expected a campaign "
        "candidate, a progressive deploy export, or default.quantum/classical")


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
        # Ne PAS renvoyer les parametres de l'autre bras si celui demande
        # manque : demander 'quantum' et recevoir 'classical' sans le
        # savoir viderait la comparaison des deux bras de son sens.
        available = sorted(k for k in default if isinstance(default[k], dict))
        raise KeyError(
            f"No default '{method}' params in the hyperparameter file. "
            f"Available arms: {available}. Refusing to substitute another "
            f"arm's parameters: '{method}' and the substitute would be "
            f"indistinguishable downstream.")
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
        # Choisir le premier lambda par ordre alphabetique est arbitraire des
        # qu'il y en a plusieurs : l'appelant recevrait un jeu de parametres
        # sans savoir lequel. Tant qu'il n'y en a qu'un, le choix est force
        # et donc licite.
        if len(lambdas) != 1:
            raise KeyError(
                f"Phase '{phase}' ({method}) holds {len(lambdas)} cost "
                f"weights {sorted(lambdas)}; pass lambda_cost explicitly "
                f"rather than receiving an arbitrary one.")
        lambda_cost = next(iter(lambdas))
    if lambda_cost not in lambdas:
        available = list(lambdas.keys())
        raise KeyError(f"Lambda '{lambda_cost}' not in {phase}. Available: {available}")

    trials = lambdas[lambda_cost]
    if rank >= len(trials):
        raise IndexError(f"Rank {rank} requested but only {len(trials)} trials available")

    return dict(trials[rank]['params'])
