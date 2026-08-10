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

**345 essais, ~47 h de mur** — deux jours, pas une semaine. Trois faits en
découlent, tous vérifiables dans les fichiers :

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

Le coût, en revanche, n'est pas l'obstacle qu'on croyait : ~47 h de mur pour
345 essais sur deux bras. Une réoptimisation **ciblée** — les seuls
paramètres qui touchent le canal du rotationnel (`beta_curl`, `kappa`,
`threshold_amr`), sur un bras, à budget réduit — est donc de l'ordre de
quelques heures, pas d'une semaine. C'est ce qui rend la question T31
(`docs/RESULTS_V4.md`) tranchable au lieu de rester une conjecture.

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
