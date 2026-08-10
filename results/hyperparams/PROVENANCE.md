# Hyperparamètres de V1 — provenance

`best_hyperparams.json` et `optuna_studies/` sont des **entrées gelées** de
cette étude, pas des résultats qu'elle produit. Ils viennent d'une campagne
Optuna antérieure au travail de falsification, menée sur la machine de
l'auteur.

## Ce que les bases contiennent réellement

Ce document annonçait « environ une semaine » de calcul. Les bases ne le
soutiennent pas. Compté directement dans les SQLite
(`tests/v4/test_hyperparams_provenance.py` refait le calcul) :

| base | essais | états | début → fin | mur |
|---|---|---|---|---|
| `classical_v2_phase1.db` | 143 | 125 COMPLETE, 18 RUNNING | 2026-04-03 20:13 → 2026-04-04 12:51 | **16.6 h** |
| `q_has_v2_phase1.db` | 202 | 178 COMPLETE, 24 RUNNING | 2026-04-04 13:50 → 2026-04-05 20:14 | **30.4 h** |
| les 8 autres bases | **0** | — | — | — |

**345 essais, ~47 h de mur.** Mais le mur n'est pas le coût : les essais ont
tourné en parallèle (jusqu'à 9 simultanés, 3.9× en moyenne sur le bras
classique, 5.3× sur le bras quantique). En temps processeur :

| bras | essais | CPU | mur | parallélisme |
|---|---|---|---|---|
| classique | 125 complets | 64.6 h | 16.6 h | 3.9× |
| quantique | 178 complets | 159.8 h | 30.4 h | 5.3× |
| **total** | **303** | **224.4 h** | **47.0 h** | — |

**224 h CPU = 9.3 jours mono-cœur.** L'annonce « environ une semaine » était
donc juste en temps processeur, et c'est le chiffre qui compte pour estimer
une relance. Coût médian d'un essai : **35 min** (classique), **56 min**
(quantique).

Trois faits de plus, tous vérifiables dans les fichiers :

1. **Seule `phase1` a tourné.** `TrainHyperParam_v2.PHASES` déclare 600 /
   600 / 400 essais pour `phase1_composite`, `phase2_complex` et
   `phase3_validation` ; les bases phase2 et phase3 sont vides.
   `best_hyperparams.json` le confirme : `best_per_phase` ne contient que
   `phase1` pour les deux bras.
2. **Aucune phase n'a atteint son quota.** 143 et 202 essais contre 600
   déclarés.
3. **Les deux campagnes ont été interrompues**, pas menées à terme : 18 et
   24 essais restent à l'état `RUNNING`.

Les hyperparamètres publiés viennent donc d'**une seule phase, incomplète**.
C'est ce qu'il faut écrire, parce que c'est ce qu'on peut vérifier.

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

L'optimiseur étant stochastique, une relance ne redonnerait pas les mêmes
valeurs — seulement, au mieux, des valeurs équivalentes. Et aucune conclusion
de cette étude ne dépend de la valeur exacte des hyperparamètres, seulement
du fait que V1 tourne avec **ceux qui ont été retenus à l'époque**, qui sont
ici.

Le coût d'une relance ciblée se calcule à partir du coût médian par essai,
pas du temps de mur de la campagne d'origine. Un essai du bras quantique
coûte **56 min de CPU**. Une réoptimisation limitée aux trois paramètres qui
touchent le canal du rotationnel (`beta_curl`, `kappa`, `threshold_amr`)
demande moins d'essais qu'un espace à 7–9 dimensions, mais chaque essai
coûte le même prix :

| budget d'essais | CPU | mur sur 4 cœurs | mur sur 32 cœurs |
|---|---|---|---|
| 30 | 28 h | ~7 h | ~1 h |
| 60 | 56 h | ~14 h | ~2 h |
| 100 | 93 h | ~23 h | ~3 h |

C'est donc **une nuit sur une machine ordinaire**, ou une heure sur une
machine louée à 32 cœurs — pas « quelques heures » comme annoncé d'abord.
Le nombre d'essais nécessaire en dimension 3 reste une **hypothèse**, non
mesurée : c'est la partie molle de cette estimation.

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
