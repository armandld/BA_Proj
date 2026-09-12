# Protocole d’évaluation Q-HAS

Ce document est la spécification de la prochaine campagne. Toute modification
après le premier calcul scientifique doit être inscrite dans
`protocol_deviations.md` avant l’analyse des résultats.

## 1. Question et comparateurs

Question principale : à budget de raffinement égal, Q-HAS réduit-il davantage
l’erreur physique que le critère AMR classique ?

Les comparateurs obligatoires sont :

1. critère classique, seuil réglé sans le scénario tenu ;
2. Q-HAS, hyperparamètres réglés sans le scénario tenu ;
3. référence classique ajustée au budget réellement consommé par Q-HAS.

Les plafonds GBT, solutions exactes et ablations expliquent le mécanisme ; ils
ne remplacent pas la comparaison fermée.

## 2. Panel physique

Scénarios : Kelvin–Helmholtz, vortex de Lamb–Oseen, tearing de Harris,
coalescence d’îlots, double tearing, torsion magnétique, Orszag–Tang et rotor
MHD. Leurs clés et paramètres vivent dans `src/train_hyperparams.py` et
`study/pipeline/config.py`; les deux listes doivent coïncider.

- Re = Rm : 400, 800, 1200, 1600 ;
- résolution DNS : N=256 ;
- graines physiques : 0, 1, 2, 3, 4 ;
- labels principaux : top 25 % de l’erreur L2, seuil calculé par trajectoire ;
- grille de patches principale : dim=4.

Chaque combinaison demandée est obligatoire. Une DNS ou un label absent fait
échouer l’analyse avant le calcul.

## 3. Validation DNS

Portes dures : champs finis, solveur non divergent, divergence magnétique FD4
relative ≤ 10⁻³ et énergie non croissante à 10⁻³ près. Orszag–Tang doit avoir
une énergie initiale normalisée et une décroissance plausible ;
Kelvin–Helmholtz doit montrer la croissance de l’énergie de perturbation après
retrait du profil moyen. Le pic de courant des cas tearing est rapporté comme
diagnostic, pas utilisé comme invariant universel.

## 4. Analyses statiques

- comparaison split aléatoire / split bloqué pour quantifier la fuite ;
- LOSO sur les huit scénarios ;
- sélection de features et courbe du cône de voisinage ;
- ablation des features ψ et horizon de prédiction ;
- condition locale de l’Hamiltonien ;
- écart stencil–site avec IC95 bootstrap par trajectoire.

Le split aléatoire est un plafond descriptif. Seuls le split bloqué et LOSO
peuvent soutenir une affirmation de transfert.

## 5. Analyse confirmatoire fermée

Chaque fold tient un scénario entier hors de tout réglage. Le fold utilise 170
essais Optuna Q-HAS et 85 essais classiques. Trois graines physiques distinctes
sont évaluées avec une graine QAOA fixe. Les deux bras partagent trace DNS,
hot start, solveur, temps simulé et profondeur AMR.

Grandeur principale :

`delta_phys = erreur_physique(Q-HAS) - erreur_physique(classique apparié)`.

Une valeur négative favorise Q-HAS. Chaque fold rapporte moyenne, IC95
bootstrap par trajectoire et test exact de signe. Les huit tests de fold sont
corrigés par Holm. L’estimation globale utilise un bootstrap hiérarchique
fold → trajectoire.

Règle confirmatoire : une supériorité classique est établie si au moins six
folds sur huit ont un IC95 strictement positif après validation du panel.
L’inverse définit une supériorité Q-HAS. Toute autre configuration est
inconclusive. L’équivalence n’est affirmée qu’avec une marge fixée avant le
calcul et un TOST valide.

## 6. Provenance et reprise

Un run scientifique exige un arbre propre. Contrat, commit, arguments, graines,
budgets et états d’essais sont enregistrés. La reprise est autorisée seulement
si le hash du contrat coïncide. Les écritures critiques sont atomiques.

## 7. Ordre gelé

1. tests rapides et lents ;
2. préflight des coefficients ;
3. campagne Optuna globale ;
4. génération et validation du panel DNS/labels ;
5. analyses statiques ;
6. huit folds confirmatoires ;
7. agrégation stricte ;
8. figures et rédaction.
