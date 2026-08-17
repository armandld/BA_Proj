"""D-108 — `scripts/extract_best_hyperparams.py` jetait le 9e parametre.

Ce que ces tests verrouillent
-----------------------------
`_detect_param_cols` reconnaissait `param_beta_grad` dans l'en-tete d'un CSV
de rescore, puis renvoyait `PARAM_COLS_SPLIT` — un jeu de colonnes qui ne
contient pas `param_beta_grad`. La generation de campagne que la fonction
detecte explicitement perdait donc sa 7e colonne, et le JSON ecrit portait 8
parametres la ou le CSV en portait 9. Silencieusement : ni exception, ni
avertissement, ni difference de forme — un JSON de 8 parametres est un JSON
parfaitement valide.

Mesure sur les artefacts gelés du depot
(`results/hyperparams/optuna_studies/GOOD_RESERVE/GOOD_reserve_v2/
before_halo_fix/`) : 4 campagnes quantiques a 9 colonnes `param_*`, **579
valeurs `beta_grad` echantillonnees jetees**, etendue 0,100000 a 2,000000.
Sur le meilleur essai de `phase1b` (essai 81), `beta_grad =
1.744060606058018`.

Sur quelle entree ces tests echouent
------------------------------------
Sur la version d'avant, tout en-tete portant `param_beta_grad` sans
`param_sigma`. Les tests `_pins_the_old_behaviour_*` sont ecrits pour rougir
si la correction est defaite en silence.

Trois generations de noms coexistent dans les artefacts — `beta_michelson`
-> `beta_grad` -> `sigma` — et chaque renommage rejouait le meme piege ; d'ou
le test de garantie generale sur une colonne inconnue.
"""
import csv
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_ROOT, 'scripts', 'extract_best_hyperparams.py')
_FROZEN = os.path.join(_ROOT, 'results', 'hyperparams', 'optuna_studies',
                       'GOOD_RESERVE', 'GOOD_reserve_v2', 'before_halo_fix')


def _load_script():
    spec = importlib.util.spec_from_file_location('extract_best_hyperparams',
                                                  _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ex = _load_script()

# En-tetes reels, releves dans les artefacts geles (pas inventes).
HDR_BETA_GRAD = [
    'trial', 'original_score', 'new_score', 'phys_score', 'patch_ratio',
    'param_beta', 'param_threshold_amr', 'param_gamma_hydro',
    'param_gamma_mag', 'param_kappa', 'param_w_z_frac', 'param_beta_grad',
    'param_beta_curl', 'param_beta_xpoint',
]
HDR_SIGMA = [
    'trial', 'new_score',
    'param_beta', 'param_w_z_frac', 'param_sigma', 'param_beta_curl',
    'param_beta_xpoint',
]
HDR_MICHELSON = [
    'trial', 'new_score',
    'param_beta', 'param_threshold_amr', 'param_gamma_hydro',
    'param_gamma_mag', 'param_kappa', 'param_w_z_frac',
    'param_beta_michelson',
]
HDR_CLASSICAL = ['trial', 'new_score', 'param_threshold_amr']


# ── 1. La garantie : aucune colonne echantillonnee n'est perdue ────────

@pytest.mark.parametrize('header', [HDR_BETA_GRAD, HDR_SIGMA, HDR_MICHELSON,
                                    HDR_CLASSICAL],
                         ids=['beta_grad', 'sigma', 'michelson', 'classical'])
def test_every_param_column_of_the_header_survives_detection(header):
    """L'extraction ne peut pas perdre ce que la campagne a echantillonne.

    ROUGE sur la version d'avant pour l'en-tete `beta_grad` uniquement :
    `param_beta_grad` etait detecte puis absent du jeu renvoye.
    """
    detected = ex._detect_param_cols(header)
    missing = [c for c in header
               if c.startswith('param_') and c not in detected]
    assert missing == [], (
        f"colonnes echantillonnees jetees par la detection : {missing}")


def test_pins_the_old_behaviour_beta_grad_is_not_in_the_canonical_split():
    """Epingle la CAUSE, pour que la correction ne soit pas defaite.

    `PARAM_COLS_SPLIT` est le jeu de la generation `sigma`. Il ne contient
    pas `param_beta_grad` — c'est correct, et c'est precisement pourquoi la
    detection ne doit pas s'y limiter quand l'en-tete porte `beta_grad`.
    """
    assert 'param_beta_grad' not in ex.PARAM_COLS_SPLIT
    assert 'param_sigma' in ex.PARAM_COLS_SPLIT
    # ... et la detection doit donc rendre STRICTEMENT plus que ce jeu.
    detected = ex._detect_param_cols(HDR_BETA_GRAD)
    assert 'param_beta_grad' in detected
    assert len(detected) > len(ex.PARAM_COLS_SPLIT)


def test_a_column_no_generation_has_ever_carried_is_not_dropped_either():
    """La garantie porte sur la classe, pas sur `beta_grad`.

    Trois renommages ont deja eu lieu ; le quatrieme ne doit pas se perdre
    en silence.
    """
    header = HDR_SIGMA + ['param_futur_inconnu']
    assert 'param_futur_inconnu' in ex._detect_param_cols(header)


def test_the_extra_column_is_announced_on_stderr(capsys):
    """Un ajout hors jeu canonique se voit : un balayage muet est un defaut."""
    ex._detect_param_cols(HDR_BETA_GRAD)
    err = capsys.readouterr().err
    assert 'param_beta_grad' in err


def test_detection_invents_nothing_for_the_classical_arm():
    """Le bras classique n'echantillonne QUE le seuil (D-22) : la correction
    ne doit pas lui inventer de parametres."""
    assert ex._detect_param_cols(HDR_CLASSICAL) == list(ex.PARAM_COLS_CLASSICAL)


# ── 2. Sur les artefacts geles, bout en bout ──────────────────────────

def _nine_column_dirs():
    out = []
    for d in sorted(glob.glob(os.path.join(_FROZEN, 'rescore_q_has_*'))):
        csvs = glob.glob(os.path.join(d, '*.csv'))
        if not csvs:
            continue
        with open(csvs[0]) as f:
            header = next(csv.reader(f))
        if 'param_beta_grad' in header:
            out.append(d)
    return out


@pytest.mark.skipif(not os.path.isdir(_FROZEN),
                    reason='artefacts geles absents du checkout')
def test_the_frozen_campaign_really_carries_nine_sampled_columns():
    """La mesure d'ou vient D-108 : 4 campagnes, 9 colonnes, 579 valeurs.

    Si ce test rougit, ce n'est pas le code qui a bouge, ce sont les
    artefacts — et alors les nombres de D-108 sont a remesurer.
    """
    dirs = _nine_column_dirs()
    assert len(dirs) == 4, f"attendu 4 campagnes a 9 colonnes, vu {len(dirs)}"
    total = 0
    lo, hi = float('inf'), float('-inf')
    for d in dirs:
        rows = list(csv.DictReader(open(glob.glob(os.path.join(d, '*.csv'))[0])))
        pcols = [c for c in rows[0] if c.startswith('param_')]
        assert len(pcols) == 9, f"{d} : {len(pcols)} colonnes param_"
        vals = [float(r['param_beta_grad']) for r in rows if r['param_beta_grad']]
        total += len(vals)
        lo, hi = min(lo, min(vals)), max(hi, max(vals))
    assert total == 579, f"579 valeurs beta_grad attendues, vu {total}"
    assert lo == pytest.approx(0.100000, abs=1e-6)
    assert hi == pytest.approx(2.000000, abs=1e-6)


@pytest.mark.skipif(not os.path.isdir(_FROZEN),
                    reason='artefacts geles absents du checkout')
def test_extraction_of_the_frozen_phase1b_keeps_beta_grad(tmp_path):
    """Bout en bout, sur le vrai fichier, avec le vrai script.

    ROUGE sur la version d'avant : le JSON portait 8 parametres et pas
    `beta_grad`.
    """
    src = os.path.join(_FROZEN, 'rescore_q_has_v2_phase1b_lambda0.4000')
    train_dir = tmp_path / 'train'
    train_dir.mkdir()
    shutil.copytree(src, train_dir / os.path.basename(src))
    out = tmp_path / 'out.json'

    subprocess.run([sys.executable, _SCRIPT, '--train-dir', str(train_dir),
                    '--output', str(out)], check=True,
                   capture_output=True, cwd=_ROOT)

    entry = json.loads(out.read_text())['default']['quantum']
    assert entry['trial'] == 81, "l'essai retenu a change : remesurer D-108"
    assert len(entry['params']) == 9, (
        f"9 parametres attendus, {len(entry['params'])} ecrits — "
        f"{sorted(entry['params'])}")
    assert entry['params']['beta_grad'] == pytest.approx(1.744060606058018,
                                                         abs=1e-12)

    # La valeur ecrite est bien celle de la ligne source, pas une valeur repliee.
    rows = list(csv.DictReader(open(glob.glob(os.path.join(src, '*.csv'))[0])))
    row = next(r for r in rows if int(r['trial']) == 81)
    for name, value in entry['params'].items():
        assert float(row['param_' + name]) == pytest.approx(value, abs=1e-12)


@pytest.mark.skipif(not os.path.isdir(_FROZEN),
                    reason='artefacts geles absents du checkout')
def test_the_correction_changes_nothing_for_the_sigma_era_campaign(tmp_path):
    """Portee de la correction : elle n'ajoute que ce qui est dans le CSV.

    La campagne vive (`results/hyperparams/optuna_studies/`) est de la
    generation `sigma` : sa sortie doit rester identique au bit pres, sinon
    la correction changerait la science en meme temps que le code.
    """
    live = os.path.join(_ROOT, 'results', 'hyperparams', 'optuna_studies')
    out = tmp_path / 'live.json'
    subprocess.run([sys.executable, _SCRIPT, '--train-dir', live,
                    '--output', str(out)], check=True,
                   capture_output=True, cwd=_ROOT)
    q = json.loads(out.read_text())['default']['quantum']['params']
    assert sorted(q) == ['beta', 'beta_curl', 'beta_xpoint', 'sigma',
                         'w_z_frac']
    assert 'beta_grad' not in q


# ── 3. Le fichier deploye n'est pas touche ────────────────────────────

def test_the_deployed_json_is_still_the_frozen_orphan_of_D22():
    """D-108 corrige l'extracteur, PAS le fichier deploye.

    `results/hyperparams/` est une entree gelee (voir son `PROVENANCE.md`).
    Ce test dit ce que D-108 ne fait pas : il ne restaure aucune provenance,
    et `sigma` reste absent — c'est D-22 qui reste ouvert.
    """
    sys.path.insert(0, os.path.join(_ROOT, 'src'))
    from hyperparams_loader import load_hyperparams
    deployed = load_hyperparams()
    assert 'sigma' not in deployed
    assert 'beta_grad' not in deployed
    assert len(deployed) == 8, (
        "le JSON deploye porte les 8 parametres qui SURVIVENT a l'ancienne "
        "extraction ; s'il en porte 9, il a ete regenere — et D-22 est a "
        "relire")
