# Campagne sur une machine louée

La phase 1 utilise une seule machine, plusieurs processus Python et un journal
Optuna local. Tous les workers partagent la même cible globale : demander 600
essais avec 16 workers produit 600 essais au total, pas 600 par worker.

## Installation

Depuis la racine du dépôt :

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

La campagne refuse un arbre Git modifié. Le commit doit donc contenir exactement
le code qui sera exécuté.

## Répétition rapide

Avant de louer la machine :

```bash
bash scripts/repetition_campagne.sh
```

Ce test exerce le journal réel, deux workers concurrents, la cible globale et la
reprise d'un essai interrompu. Il ne valide pas la physique, couverte par les
tests et le contrôle des coefficients.

## Lancement de la phase 1

```bash
bash scripts/run_rented_campaign.sh 16 600 0
```

Les arguments sont, dans l'ordre :

1. nombre de workers ;
2. cible globale d'essais ;
3. graine du premier worker.

Sans argument, le lanceur utilise tous les cœurs détectés, la cible du protocole
(600) et la graine 0. Les bibliothèques numériques restent limitées à un thread
par worker afin d'éviter la surallocation des cœurs.

Le lanceur exécute le contrôle des coefficients, affiche le périmètre entraîné,
prépare le journal, lance les workers, attend leur terminaison puis valide le
candidat final.

## Sorties et suivi

Les fichiers de travail vivent dans :

```text
results/hyperparams/reoptimisation/
├── journal/q_has_v2_phase1.log
├── logs/<date>/worker_<id>.log
└── candidate_phase1.json
```

Suivre un worker :

```bash
tail -f results/hyperparams/reoptimisation/logs/<date>/worker_0.log
```

`candidate_phase1.json` n'est exploitable que si `status` vaut `complete`. Le
lanceur vérifie cette condition et sort en erreur si elle n'est pas satisfaite.

Produire les diagnostics Optuna :

```bash
.venv/bin/python src/analyze_hyperparams.py \
  --journal-path results/hyperparams/reoptimisation/journal/q_has_v2_phase1.log \
  --study-name q_has_v2_phase1
```

## Interruption et reprise

Arrêter le lanceur interrompt ses workers. Pour reprendre, relancer exactement
la même commande depuis le même commit et avec la même cible :

```bash
bash scripts/run_rented_campaign.sh 16 600 0
```

La préparation marque les anciens essais restés `RUNNING` comme échoués et les
exclut du budget ; leur place est donc recalculée. Un changement de commit, de
cible ou de contrat scientifique est refusé au lieu d'être mélangé au journal
existant.

Pour conserver une campagne, sauvegarder tout le dossier
`results/hyperparams/reoptimisation/` lorsque les workers sont arrêtés.

## Diagnostic

- Échec avant le premier worker : corriger le contrôle affiché ; ne pas forcer.
- Échec d'un worker : lire son fichier dans `logs/<date>/` puis relancer la même
  commande après correction et nouveau commit. Un contrat modifié exige un
  nouveau dossier de campagne.
- Candidat `partial` : au moins un essai manque ou reste actif ; ne pas l'utiliser
  dans `study`.

## Panel DNS et étude

Une fois le candidat complet, générer les 160 trajectoires DNS en exploitant
les cœurs loués, puis lancer les analyses statiques :

```bash
bash scripts/run_dns_campaign.sh 8 256
QHAS_HYPERPARAMS_PATH=results/hyperparams/reoptimisation/candidate_phase1.json \
  bash scripts/run_study_v3.sh --all
```

Le premier argument du lanceur DNS est le parallélisme maximal. Chaque tâche
écrit un fichier distinct, puis une validation globale exige le panel complet.
Par défaut, DNS, labels, analyses et folds vivent ensemble dans
`results/campaigns/current/`, dossier ignoré par Git. Définir
`QHAS_RESULTS_DIR` avant les trois lanceurs permet de choisir un autre dossier
persistant sans mélanger deux campagnes.

## Campagne confirmatoire

Les huit folds sont indépendants et peuvent être lancés en parallèle :

```bash
bash scripts/run_confirmatory_campaign.sh 4 170 85
```

Les arguments sont le nombre maximal de folds simultanés, le budget Optuna
Q-HAS par fold et le budget classique. Le lanceur produit ensuite les comptes
confirmatoires et la synthèse. Tous ces runs refusent un arbre Git modifié.
