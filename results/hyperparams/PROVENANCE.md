# Hyperparamètres de V1 — provenance

`best_hyperparams.json` et `optuna_studies/` sont des **entrées gelées** de
cette étude, pas des résultats qu'elle produit. Ils viennent d'une campagne
Optuna d'environ une semaine sur la machine de l'auteur, antérieure au
travail de falsification.

## Ce qu'ils sont

| chemin | contenu |
|---|---|
| `best_hyperparams.json` | le fichier que `src/hyperparams_loader.py` consomme : blocs `default`, `best_per_phase`, `scenario_combos`, `per_scenario`, `training_phases` |
| `optuna_studies/*.db` | les bases SQLite des études Optuna (`q_has_v2_phase*`, `classical_v2_phase*`, variantes `_agr`, `rescore_*`) |
| `optuna_studies/analysis_*` | les tableaux d'analyse par phase produits à l'époque |
| `optuna_studies/GOOD_RESERVE` | une copie de sauvegarde des meilleures études |

Les scripts d'entraînement correspondants sont `src/TrainHyperParam_v1.py`
à `_v4.py`. Leurs phases déclarent 200 à 600 essais chacune.

## Pourquoi ils ne sont pas régénérés ici

Relancer la campagne coûte environ une semaine de calcul et, l'optimiseur
étant stochastique, elle ne redonnerait pas les mêmes valeurs — seulement,
au mieux, des valeurs équivalentes. Le coût est sans rapport avec ce que la
vérification apporterait : aucune conclusion de cette étude ne dépend de la
valeur exacte des hyperparamètres, seulement du fait que V1 tourne avec
**ceux qui ont été retenus à l'époque**, qui sont ici.

## Ce qu'il faut en dire dans le manuscrit

Deux points, tous deux vérifiables dans les fichiers ci-dessus :

1. **Le seuil `threshold_amr` fait partie des paramètres optimisés.** C'est
   la source du défaut D13 : le bras Q-HAS arrivait réglé sur les scénarios
   contre lesquels il était ensuite comparé. La campagne sans fuite
   (`study/h4_transfer/t22_unseen_conditions.py`, mode leak-free) est celle
   qui retire ce réglage ; c'est elle qui fait foi.
2. **L'optimisation a porté sur les 4 scénarios simultanément**, ce qui rend
   les hyperparamètres « par scénario » non indépendants entre eux.

## Reproductibilité

Ce dossier est le seul du dépôt qui ne soit pas reproductible par une
commande. Tout le reste de `results/` se recalcule depuis `study/`, et
`study/common/t16_aggregate_v4.py` le vérifie ligne par ligne.
