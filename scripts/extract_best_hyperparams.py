#!/usr/bin/env python3
"""
Extract top-K hyperparameters from each rescore CSV and write best_hyperparams.json.

Scans Train_results/rescore_*_lambda*/ directories for both quantum (q_has_v2)
and classical (classical_v2) training results.  Outputs a structured JSON with:

  - default.quantum / default.classical : best overall params for pipeline
  - scenario_combos.simple / complex    : best params for scenario groups
  - per_scenario.<name>                 : best params per individual scenario
  - training_phases                     : top-K trials per phase/lambda (raw data)

Usage:
    python scripts/extract_best_hyperparams.py [--train-dir Train_results] \
        [--output best_hyperparams.json] [--top-k 3] [--lambda-cost 0.40]
"""
import argparse
import csv
import json
import os
import re
import sys

# Quantum param columns — Phase 1 (shared beta) vs Phase 1b+ (split beta)
PARAM_COLS_SHARED = [
    'param_beta', 'param_threshold_amr', 'param_gamma_hydro',
    'param_gamma_mag', 'param_kappa', 'param_w_z_frac', 'param_beta_michelson',
]
PARAM_COLS_SPLIT = [
    'param_beta', 'param_threshold_amr', 'param_gamma_hydro',
    'param_gamma_mag', 'param_kappa', 'param_w_z_frac',
    'param_sigma', 'param_beta_curl', 'param_beta_xpoint',
]

# Classical param columns (only threshold_amr)
PARAM_COLS_CLASSICAL = ['param_threshold_amr']

SCORE_COLS = ['new_score', 'original_score', 'phys_score', 'patch_ratio']

# Per-scenario score columns — auto-detected from CSV header.
# Maps full scenario name -> (loss_col, phys_col, patch_col).
# Two naming conventions exist depending on the training phase:
#   Phase 1 (old): kh, vortex, tearing, coalescence
#   Phase 1 (new): kh, tearing, ot, rotor
SCENARIO_SCORE_COLS_OLD = {
    'kelvin_helmholtz':  ('new_loss_kh',          'phys_kh',          'patch_kh'),
    'lamb_oseen_vortex': ('new_loss_vortex',       'phys_vortex',      'patch_vortex'),
    'harris_tearing':    ('new_loss_tearing',       'phys_tearing',     'patch_tearing'),
    'island_coalescence':('new_loss_coalescence',   'phys_coalescence', 'patch_coalescence'),
}
SCENARIO_SCORE_COLS_NEW = {
    'kelvin_helmholtz':  ('new_loss_kh',      'phys_kh',      'patch_kh'),
    'harris_tearing':    ('new_loss_tearing',  'phys_tearing',  'patch_tearing'),
    'orszag_tang':       ('new_loss_ot',       'phys_ot',       'patch_ot'),
    'mhd_rotor':         ('new_loss_rotor',    'phys_rotor',    'patch_rotor'),
}

# Scenario groups
SIMPLE_SCENARIOS = ['kelvin_helmholtz', 'lamb_oseen_vortex', 'harris_tearing', 'island_coalescence']
COMPLEX_SCENARIOS = ['orszag_tang', 'mhd_rotor']

# Pattern for quantum rescore dirs
QUANTUM_DIR_PATTERN = re.compile(
    r'^rescore_q_has_v2_phase([\w]+)_lambda([\d.]+)$'
)
# Pattern for classical rescore dirs
CLASSICAL_DIR_PATTERN = re.compile(
    r'^rescore_classical_v2_phase([\w]+)_lambda([\d.]+)$'
)


def _detect_param_cols(header):
    """Auto-detect whether CSV has shared or split Michelson betas.

    D-108 — pourquoi l'ancienne version etait fausse. La branche testait
    `param_beta_grad` dans l'en-tete puis renvoyait `PARAM_COLS_SPLIT`, qui
    ne contient pas cette colonne : la generation de campagne que la fonction
    reconnait explicitement voyait son 9e parametre jete sans un mot, et le
    JSON ecrit portait 8 parametres la ou le CSV en portait 9. Trois
    generations de noms coexistent (`beta_michelson` -> `beta_grad` ->
    `sigma`) et chaque renommage rejouait le meme piege.

    Mesure (avant) sur `results/hyperparams/optuna_studies/GOOD_RESERVE/
    GOOD_reserve_v2/before_halo_fix/` : 4 campagnes quantiques a 9 colonnes
    `param_*`, 8 parametres extraits, **579 valeurs `beta_grad`
    echantillonnees jetees** (etendue 0,100000 a 2,000000).

    On garde l'ordre canonique et on AJOUTE toute colonne `param_*` presente
    dans l'en-tete qu'il ne nomme pas, en la signalant : une extraction ne
    peut pas perdre en silence ce que la campagne a echantillonne (meme
    famille que D-55/D-56, un balayage vide doit crier).
    """
    if 'param_sigma' in header or 'param_beta_grad' in header:
        cols = list(PARAM_COLS_SPLIT)
    elif 'param_beta_michelson' in header:
        cols = list(PARAM_COLS_SHARED)
    elif 'param_threshold_amr' in header:
        cols = list(PARAM_COLS_CLASSICAL)
    else:
        cols = list(PARAM_COLS_SHARED)

    extras = [c for c in header
              if c.startswith('param_') and c not in cols]
    if extras:
        print(f"  [extra] colonnes hors du jeu canonique, extraites telles "
              f"quelles : {', '.join(extras)}", file=sys.stderr)
        cols.extend(extras)
    return cols


def _detect_scenario_cols(header):
    """Auto-detect scenario column mapping based on CSV header."""
    # Check for new-style columns (ot, rotor) vs old-style (vortex, coalescence)
    if 'phys_ot' in header or 'phys_rotor' in header:
        return SCENARIO_SCORE_COLS_NEW
    return SCENARIO_SCORE_COLS_OLD


def parse_rescore_dir(dirpath, top_k=3):
    """Parse a rescore directory and return top-k trials with per-scenario data."""
    csvs = [f for f in os.listdir(dirpath) if f.endswith('.csv')]
    if not csvs:
        print(f"  SKIP {dirpath}: no CSV found", file=sys.stderr)
        return None

    csvpath = os.path.join(dirpath, csvs[0])
    trials = []
    with open(csvpath, 'r') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        param_cols = _detect_param_cols(header)
        param_names = [c.replace('param_', '') for c in param_cols]
        scenario_cols = _detect_scenario_cols(header)

        for row in reader:
            try:
                score = float(row['new_score'])
            except (ValueError, KeyError):
                continue

            params = {}
            for col, name in zip(param_cols, param_names):
                if col in row:
                    try:
                        params[name] = float(row[col])
                    except (ValueError, TypeError):
                        pass

            scores = {}
            for col in SCORE_COLS:
                if col in row:
                    try:
                        scores[col] = float(row[col])
                    except (ValueError, TypeError):
                        scores[col] = None

            # Per-scenario scores
            per_scenario = {}
            for sc_name, (loss_col, phys_col, patch_col) in scenario_cols.items():
                if phys_col in row:
                    try:
                        per_scenario[sc_name] = {
                            'loss': float(row[loss_col]) if loss_col in row else None,
                            'phys': float(row[phys_col]),
                            'patch': float(row[patch_col]) if patch_col in row else None,
                        }
                    except (ValueError, TypeError):
                        pass

            trials.append({
                'trial': int(row.get('trial', -1)),
                'new_score': score,
                'scores': scores,
                'params': params,
                'per_scenario': per_scenario,
            })

    trials.sort(key=lambda t: t['new_score'])
    return trials[:top_k]


def _pick_best_for_scenario(all_trials, scenario_name):
    """Among all trials, pick the one with lowest phys for a given scenario."""
    candidates = []
    for t in all_trials:
        sc = t.get('per_scenario', {}).get(scenario_name)
        if sc and sc.get('phys') is not None:
            candidates.append((sc['phys'], sc.get('patch', 1.0), t))
    if not candidates:
        return None
    # Sort by phys ascending, then patch ascending
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def _pick_best_for_group(all_trials, scenario_names):
    """Pick trial with lowest average phys across a group of scenarios."""
    candidates = []
    for t in all_trials:
        physes = []
        for sc_name in scenario_names:
            sc = t.get('per_scenario', {}).get(sc_name)
            if sc and sc.get('phys') is not None:
                physes.append(sc['phys'])
        if len(physes) == len(scenario_names):
            avg_phys = sum(physes) / len(physes)
            candidates.append((avg_phys, t))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _trial_to_entry(trial):
    """Convert a trial dict to a clean JSON entry."""
    entry = {
        'trial': trial['trial'],
        'new_score': trial['new_score'],
        'phys_score': trial['scores'].get('phys_score'),
        'patch_ratio': trial['scores'].get('patch_ratio'),
        'params': trial['params'],
    }
    if trial.get('per_scenario'):
        entry['per_scenario'] = trial['per_scenario']
    return entry

def get_best_in_phase(phases_dict, target_lambda, target_lambda_alt):
    res = {}
    for p_name, l_dict in phases_dict.items():
        trials = []
        for l_key, t_list in l_dict.items():
            l_val = l_key.replace('lambda_', '')
            if l_val == target_lambda or l_val == target_lambda_alt:
                trials.extend(t_list)
        if not trials:
            for t_list in l_dict.values():
                trials.extend(t_list)
        if trials:
            best = min(trials, key=lambda t: t['new_score'])
            res[p_name] = _trial_to_entry(best)
    return res

def main():
    parser = argparse.ArgumentParser(
        description='Extract best hyperparameters from rescore results')
    parser.add_argument('--train-dir', default='Train_results',
                        help='Path to Train_results directory')
    parser.add_argument('--output', default='best_hyperparams.json',
                        help='Output JSON file path')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of top trials per phase/lambda (default: 3)')
    parser.add_argument('--lambda-cost', type=float, default=0.40,
                        help='Lambda cost to use for default selection (default: 0.40)')
    parser.add_argument('--phase-filter', type=str, default=None,
                        help='Only include rescore dirs matching this phase suffix '
                             '(e.g. "phase3" to match q_has_v2_phase3 and classical_v2_phase3)')
    parser.add_argument('--quantum-phase-filter', type=str, default=None,
                        help='Override phase filter for quantum only '
                             '(e.g. "phase1b" to use split-beta results)')
    parser.add_argument('--classical-phase-filter', type=str, default=None,
                        help='Override phase filter for classical only '
                             '(e.g. "phase1" for classical Phase 1)')
    args = parser.parse_args()

    train_dir = args.train_dir
    if not os.path.isdir(train_dir):
        print(f"ERROR: {train_dir} not found", file=sys.stderr)
        sys.exit(1)

    target_lambda = f"{args.lambda_cost:.2f}"
    # Also accept 4-digit lambda format (e.g. 0.4000)
    target_lambda_alt = f"{args.lambda_cost:.4f}"

    quantum_phases = {}
    classical_phases = {}

    # Scan all rescore directories
    # Separate filters for quantum and classical (with fallback to shared filter)
    q_phase_filter = args.quantum_phase_filter or args.phase_filter
    c_phase_filter = args.classical_phase_filter or args.phase_filter
    if q_phase_filter:
        print(f"[FILTER] Quantum phase filter: {q_phase_filter}")
    if c_phase_filter:
        print(f"[FILTER] Classical phase filter: {c_phase_filter}")

    for entry in sorted(os.listdir(train_dir)):
        dirpath = os.path.join(train_dir, entry)
        if not os.path.isdir(dirpath):
            continue

        # Quantum
        m = QUANTUM_DIR_PATTERN.match(entry)
        if m:
            raw_phase = f"phase{m.group(1)}"
            lam = m.group(2)
            if q_phase_filter and raw_phase != q_phase_filter:
                continue
            print(f"[quantum] phase={raw_phase}, lambda={lam}")
            top = parse_rescore_dir(dirpath, top_k=args.top_k)
            if top is None:
                continue
            if raw_phase not in quantum_phases:
                quantum_phases[raw_phase] = {}
            quantum_phases[raw_phase][f"lambda_{lam}"] = top
            continue

        # Classical
        m = CLASSICAL_DIR_PATTERN.match(entry)
        if m:
            phase = f"phase{m.group(1)}"
            lam = m.group(2)
            if c_phase_filter and phase != c_phase_filter:
                continue
            print(f"[classical] phase={phase}, lambda={lam}")
            top = parse_rescore_dir(dirpath, top_k=args.top_k)
            if top is None:
                continue
            if phase not in classical_phases:
                classical_phases[phase] = {}
            classical_phases[phase][f"lambda_{lam}"] = top
            continue

    if not quantum_phases and not classical_phases:
        print("ERROR: no rescore directories found", file=sys.stderr)
        sys.exit(1)

    # ── Build structured output ──────────────────────────────────

    # Collect ALL quantum trials from target lambda for selection
    all_quantum_trials = []
    for phase_key in sorted(quantum_phases.keys()):
        lambdas = quantum_phases[phase_key]
        for lam_key, trials in lambdas.items():
            lam_val = lam_key.replace('lambda_', '')
            if lam_val == target_lambda or lam_val == target_lambda_alt:
                all_quantum_trials.extend(trials)

    # If no target lambda found, use all trials
    if not all_quantum_trials:
        for phase_key in sorted(quantum_phases.keys()):
            for trials in quantum_phases[phase_key].values():
                all_quantum_trials.extend(trials)

    # Same for classical
    all_classical_trials = []
    for phase_key in sorted(classical_phases.keys()):
        for lam_key, trials in classical_phases[phase_key].items():
            lam_val = lam_key.replace('lambda_', '')
            if lam_val == target_lambda or lam_val == target_lambda_alt:
                all_classical_trials.extend(trials)
    if not all_classical_trials:
        for phase_key in sorted(classical_phases.keys()):
            for trials in classical_phases[phase_key].values():
                all_classical_trials.extend(trials)

    # ── Defaults: best overall trial ──
    default_section = {}

    # Quantum default: best by new_score
    if all_quantum_trials:
        best_q = min(all_quantum_trials, key=lambda t: t['new_score'])
        default_section['quantum'] = _trial_to_entry(best_q)

    # Classical default: best by new_score
    if all_classical_trials:
        best_c = min(all_classical_trials, key=lambda t: t['new_score'])
        default_section['classical'] = _trial_to_entry(best_c)

    # ── Scenario combos: simple (4 scenarios) + complex (2 scenarios) ──
    scenario_combos = {}

    # Simple combo — quantum
    combo_entry = {'quantum': None, 'classical': None}
    if all_quantum_trials:
        best = _pick_best_for_group(all_quantum_trials, SIMPLE_SCENARIOS)
        if best:
            combo_entry['quantum'] = _trial_to_entry(best)
    if all_classical_trials:
        best = _pick_best_for_group(all_classical_trials, SIMPLE_SCENARIOS)
        if best:
            combo_entry['classical'] = _trial_to_entry(best)
    scenario_combos['simple'] = combo_entry

    # Complex combo — quantum
    combo_entry = {'quantum': None, 'classical': None}
    if all_quantum_trials:
        best = _pick_best_for_group(all_quantum_trials, COMPLEX_SCENARIOS)
        if best:
            combo_entry['quantum'] = _trial_to_entry(best)
    if all_classical_trials:
        best = _pick_best_for_group(all_classical_trials, COMPLEX_SCENARIOS)
        if best:
            combo_entry['classical'] = _trial_to_entry(best)
    scenario_combos['complex'] = combo_entry

    # ── Per-scenario: best trial for each scenario found in data ──
    all_scenarios = list(set(SIMPLE_SCENARIOS + COMPLEX_SCENARIOS))
    per_scenario = {}
    for sc_name in all_scenarios:
        sc_entry = {'quantum': None, 'classical': None}
        if all_quantum_trials:
            best = _pick_best_for_scenario(all_quantum_trials, sc_name)
            if best:
                sc_entry['quantum'] = _trial_to_entry(best)
        if all_classical_trials:
            best = _pick_best_for_scenario(all_classical_trials, sc_name)
            if best:
                sc_entry['classical'] = _trial_to_entry(best)
        per_scenario[sc_name] = sc_entry

    # ── Training phases: raw top-K data ──
    training_phases = {
        'quantum': quantum_phases,
        'classical': classical_phases,
    }

    # ── Best Per Phase ──
    best_per_phase = {
        'quantum': get_best_in_phase(quantum_phases, target_lambda, target_lambda_alt),
        'classical': get_best_in_phase(classical_phases, target_lambda, target_lambda_alt)
    }

    # ── Assemble final output ──
    output = {
        'default': default_section,
        'best_per_phase': best_per_phase,
        'scenario_combos': scenario_combos,
        'per_scenario': per_scenario,
        'training_phases': training_phases,
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"\nWritten to {args.output}")
    if 'quantum' in default_section:
        q = default_section['quantum']
        print(f"Default quantum: trial #{q['trial']} "
              f"(score={q['new_score']:.6f}, phys={q.get('phys_score', '?')}, "
              f"patch={q.get('patch_ratio', '?')})")
        for name, val in q['params'].items():
            print(f"  {name:>20s} = {val}")
    if 'classical' in default_section:
        c = default_section['classical']
        print(f"Default classical: trial #{c['trial']} "
              f"(score={c['new_score']:.6f})")


if __name__ == '__main__':
    main()
