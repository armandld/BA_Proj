# Synthèse de revue du code

## Décisions appliquées

- exécution sur une seule machine louée ; suppression des chemins HPC et des
  stockages distants ;
- journal Optuna local partagé avec cible globale et reprise contractuelle ;
- huit scénarios canoniques, quatre Reynolds et cinq graines physiques ;
- portail DNS unique, validation FD4 assortie et catalogue de données strict ;
- réglages LOSO sans fuite et comparaison fermée à budget apparié ;
- inférence sur trajectoires physiques, pas sur patches ou instantanés ;
- graine QAOA fixe pour isoler la variance physique ;
- suppression des lanceurs V2 et du bootstrap par instantané remplacés ;
- provenance et écritures atomiques sur les sorties critiques.

## Défauts corrigés pendant la dernière passe

- scénarios supplémentaires absents du panel DNS et de plusieurs analyses ;
- données manquantes silencieusement ignorées ;
- graine VQC déclarée mais non consommée ;
- coefficients ZZZZ superposés écrasés dans le contrôle d’ablation ;
- observable KH et courant calculés selon un axe incohérent ;
- pic de courant tearing utilisé à tort comme porte universelle ;
- comptes confirmatoires limités à quatre folds et sans réplication physique ;
- agrégation de schémas historiques à la place des sorties courantes.

## Limites assumées

Le travail utilise des simulateurs quantiques, pas du matériel réel. Le
solveur, les labels et le coût AMR définissent un banc d’essai précis, pas une
preuve universelle sur l’AMR ou QAOA. Les modules exploratoires qui ne figurent
pas dans le protocole ne peuvent pas soutenir l’affirmation principale.

La qualité finale reste conditionnée aux contrôles de `DEFAUTS.md` et à
la régénération complète des résultats.
