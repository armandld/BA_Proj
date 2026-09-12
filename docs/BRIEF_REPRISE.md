# Brief de reprise

## Objectif

Évaluer l’intérêt de QAOA pour la décision AMR en MHD 2-D, puis produire des
résultats directement exploitables dans un préprint, y compris si la réponse
est négative.

## Protocole actuel

- huit scénarios indépendants ;
- Re = Rm ∈ {400, 800, 1200, 1600} ;
- cinq graines de condition initiale ;
- DNS principale N=256 ;
- LOSO par scénario ;
- trois trajectoires physiques appariées par fold confirmatoire ;
- 170 essais Optuna Q-HAS et 85 essais classiques par fold ;
- bootstrap hiérarchique par trajectoire et correction de Holm.

## Ordre d’exécution

```bash
# Vérification
.venv/bin/python -m pytest tests -q -m "not slow"
.venv/bin/python -m pytest tests -q -m slow
bash scripts/repetition_campagne.sh

# Entraînement global sur la machine louée
bash scripts/run_rented_campaign.sh 16 600 0

# Données et analyses statiques
bash scripts/run_dns_campaign.sh 8 256
QHAS_HYPERPARAMS_PATH=results/hyperparams/reoptimisation/candidate_phase1.json \
  bash scripts/run_study_v3.sh --all

# Comparaison confirmatoire
bash scripts/run_confirmatory_campaign.sh 4 170 85
```

Avant chaque campagne longue : commit propre, espace disque contrôlé et copie
du dossier de résultats après arrêt des processus.

Les trois derniers lanceurs partagent par défaut
`results/campaigns/current/`. Pour nommer explicitement une campagne :

```bash
export QHAS_RESULTS_DIR="$PWD/results/campaigns/preprint_01"
```

## Sorties décisives

- `candidate_phase1.json` : candidat complet et traçable ;
- `v3_master_table.md` : analyses statiques, sans ligne manquante ;
- `t20_qhas_run_variance_<fold>.json` : trajectoires appariées ;
- `t23_headline_counts.json` : inférence confirmatoire ;
- `t15c_fold_synthesis.md` : synthèse par fold.

Ne pas citer les nombres déjà présents dans `results/` avant leur régénération
sur le commit de campagne.
