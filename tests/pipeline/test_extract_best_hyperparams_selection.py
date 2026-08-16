"""D-109 — `per_scenario` annonçait un optimum et n'en cherchait pas un.

Ce que ces tests verrouillent
-----------------------------
`parse_rescore_dir` ne rendait que les `top_k` meilleurs essais au score
**agrégé**, et `main()` en faisait `all_quantum_trials` — le nom promettait
tous les essais et en portait trois. `_pick_best_for_scenario`, dont la
docstring dit « Among all trials », ne pouvait donc désigner l'optimum d'un
scénario que parmi les 3 meilleurs du score agrégé, sur 178. `--top-k` borne
la donnée brute écrite dans `training_phases` ; il n'a jamais eu vocation à
borner la sélection.

La valeur écrite restait plausible : un `phys` du bon ordre, du bon signe,
issu d'un vrai essai. Elle n'était simplement pas le minimum annoncé.

Mesure sur la campagne vive (`results/hyperparams/optuna_studies/
rescore_q_has_v2_phase1_lambda0.4000`, 178 essais) — `phys` écrit, avant →
après : `kelvin_helmholtz` 0,003604662 → 0,0013197164 (**×2,7**),
`harris_tearing` 0,0044288611 → 0,0024402795 (**×1,8**), `orszag_tang`
0,063719323 → 0,061306123. Côté classique, `mhd_rotor` 0,048986229 →
0,02698116 (**×1,8**). 6 entrées sur 8 changent ; `default`,
`best_per_phase` et `training_phases` ne bougent pas.

Sur quelle entrée ces tests échouent
------------------------------------
Sur toute campagne où l'optimum d'un scénario n'est pas dans le top-3 du
score agrégé — c'est le cas de la campagne vive du dépôt, sur 3 scénarios
quantiques sur 4. Sur une campagne où les deux coïncident (`mhd_rotor`
quantique ici), le test ne sépare rien : c'est pourquoi il porte sur les
quatre scénarios et les deux bras, pas sur un seul.

Ce que D-109 ne fait pas : il ne touche pas `best_hyperparams.json`, entrée
gelée. Aucun nombre publié ne bouge.
"""
import csv
import glob
import importlib.util
import json
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_ROOT, 'scripts', 'extract_best_hyperparams.py')
_LIVE = os.path.join(_ROOT, 'results', 'hyperparams', 'optuna_studies')
_Q_DIR = os.path.join(_LIVE, 'rescore_q_has_v2_phase1_lambda0.4000')

# scenario -> colonne `phys` du CSV (convention « nouvelle », cf. le script)
_PHYS_COL = {
    'kelvin_helmholtz': 'phys_kh',
    'harris_tearing': 'phys_tearing',
    'orszag_tang': 'phys_ot',
    'mhd_rotor': 'phys_rotor',
}

# Nombres MESURÉS avant/après la correction, écrits ici pour qu'une dérive se
# voie. Les « avant » épinglent l'ancien comportement : si l'un d'eux revient,
# la correction a été défaite.
_QUANTUM_BEFORE_AFTER = {
    'kelvin_helmholtz': (0.003604662, 0.0013197164),
    'harris_tearing': (0.0044288611, 0.0024402795),
    'orszag_tang': (0.063719323, 0.061306123),
    'mhd_rotor': (0.055825707, 0.055825707),   # ne sépare pas : coïncide
}


def _load_script():
    spec = importlib.util.spec_from_file_location('extract_best_hyperparams',
                                                  _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = _load_script()

_skip = pytest.mark.skipif(not os.path.isdir(_Q_DIR),
                           reason='campagne vive absente du checkout')


def _rows(dirpath):
    return [r for r in csv.DictReader(open(glob.glob(os.path.join(dirpath, '*.csv'))[0]))
            if r.get('new_score')]


@pytest.fixture(scope='module')
def extracted(tmp_path_factory):
    out = tmp_path_factory.mktemp('d109') / 'out.json'
    subprocess.run([sys.executable, _SCRIPT, '--train-dir', _LIVE,
                    '--output', str(out)], check=True,
                   capture_output=True, cwd=_ROOT)
    return json.loads(out.read_text())


# ── 1. Le contrat de `parse_rescore_dir` ──────────────────────────────

@_skip
def test_parse_returns_the_top_k_and_also_every_trial():
    """ROUGE avant : la fonction ne rendait qu'une liste de `top_k` essais.

    Les deux sorties ont deux usages disjoints — donnée brute bornée d'un
    côté, base de sélection complète de l'autre.
    """
    top, every = ex.parse_rescore_dir(_Q_DIR, top_k=3)
    assert len(top) == 3
    assert len(every) == len(_rows(_Q_DIR)) == 178, (
        "178 essais attendus dans la campagne vive : si ce nombre a changé, "
        "les mesures de D-109 sont à refaire")
    assert every[:3] == top, "le top-K doit être le préfixe du tri complet"
    scores = [t['new_score'] for t in every]
    assert scores == sorted(scores), "l'ordre croissant du score agrégé est perdu"


# ── 2. La garantie annoncée : un optimum EST un optimum ───────────────

@_skip
@pytest.mark.parametrize('scenario', sorted(_PHYS_COL))
@pytest.mark.parametrize('arm', ['quantum', 'classical'])
def test_the_written_per_scenario_entry_is_the_true_minimum(extracted, scenario, arm):
    """L'assertion porte sur la garantie annoncée (« best params per
    individual scenario »), pas sur le fait que la clé existe.

    ROUGE avant sur 6 des 8 combinaisons.
    """
    entry = extracted['per_scenario'][scenario][arm]
    if entry is None:
        pytest.skip(f'{scenario}/{arm} absent de cette campagne')
    written = entry['per_scenario'][scenario]['phys']

    dirs = glob.glob(os.path.join(_LIVE, f'rescore_{"q_has" if arm == "quantum" else "classical"}_v2_*'))
    col = _PHYS_COL[scenario]
    values = [float(r[col]) for d in dirs for r in _rows(d) if r.get(col)]
    assert values, f'aucune valeur {col} : le balayage est vide, il ne prouve rien'

    assert written == pytest.approx(min(values), rel=1e-9), (
        f"{scenario}/{arm} : le JSON écrit {written:.8g} alors que le minimum "
        f"de la campagne vaut {min(values):.8g} sur {len(values)} essais")


@_skip
@pytest.mark.parametrize('scenario,before,after',
                         [(k, v[0], v[1]) for k, v in _QUANTUM_BEFORE_AFTER.items()])
def test_pins_the_measured_numbers_before_and_after(extracted, scenario, before, after):
    """Épingle les nombres mesurés, pour que la correction ne soit pas défaite
    en silence et qu'une dérive se voie."""
    written = extracted['per_scenario'][scenario]['quantum']['per_scenario'][scenario]['phys']
    assert written == pytest.approx(after, rel=1e-6)
    if before != after:
        assert written != pytest.approx(before, rel=1e-6), (
            f"{scenario} est revenu à la valeur d'avant D-109 ({before}) : "
            "la sélection choisit de nouveau parmi les top-K")


# ── 3. Portée : ce que la correction NE change pas ────────────────────

@_skip
def test_the_default_block_is_untouched(extracted):
    """`default` est le seul bloc que le pipeline consomme sans argument.
    Il est déjà le minimum du score agrégé, donc il ne peut pas bouger —
    et s'il bouge, la correction déborde de son périmètre."""
    assert extracted['default']['quantum']['trial'] == 4
    assert extracted['default']['quantum']['new_score'] == pytest.approx(
        0.2134283135270856, rel=1e-12)
    assert extracted['default']['classical']['trial'] == 64


@_skip
def test_the_raw_block_stays_bounded_by_top_k(extracted):
    """`training_phases` reste la donnée brute bornée : la correction élargit
    la SÉLECTION, pas le fichier écrit."""
    lambdas = extracted['training_phases']['quantum']['phase1']
    for trials in lambdas.values():
        assert len(trials) == 3


def test_the_deployed_json_is_not_regenerated_here():
    """`results/hyperparams/` est gelé (son `PROVENANCE.md`). D-109 corrige
    l'extracteur ; le fichier déployé garde ses valeurs, et donc le symptôme
    qui a mis sur la piste : ses quatre optima « par scénario » quantiques
    sont un seul et même essai."""
    deployed = json.load(open(os.path.join(
        _ROOT, 'results', 'hyperparams', 'best_hyperparams.json')))
    trials = {name: block['quantum']['trial']
              for name, block in deployed['per_scenario'].items()
              if block['quantum'] is not None}
    assert set(trials.values()) == {85}, (
        "le JSON déployé a été régénéré : refaire les mesures de D-109 et "
        "relire D-22")
    assert deployed['default']['quantum']['trial'] == 85
