# Plan du préprint

## Titre de travail

**Can local QAOA decisions improve adaptive mesh refinement in 2-D MHD? A
leak-free, budget-matched evaluation**

Le titre définitif dépend du résultat confirmatoire. Ne pas annoncer un
avantage, un échec ou une équivalence avant l’agrégation stricte.

## 1. Question

Présenter l’AMR comme une décision structurée locale et expliquer pourquoi un
Hamiltonien d’Ising pourrait représenter les interactions entre patches. La
question n’est pas de montrer que QAOA optimise un Hamiltonien, mais que cette
optimisation améliore l’erreur physique à coût de raffinement égal.

## 2. Méthodes

- solveur MHD FD4/RK4 et huit conditions initiales ;
- score classique et mapping physique–Ising ;
- circuit QAOA, optimisation et règle de décision ;
- panel 8 scénarios × 4 Re × 5 graines ;
- labels, splits bloqués et LOSO ;
- comparaison fermée et appariement du budget ;
- bootstrap hiérarchique et correction de Holm.

## 3. Validation du banc

Rapporter convergence du solveur, divergence magnétique, énergie, contrôles KH
et Orszag–Tang, exactitude de l’opérateur Ising et dégénérescence. Séparer les
diagnostics des critères d’exclusion.

## 4. Résultats

1. fuite du split aléatoire par rapport au split bloqué ;
2. transfert LOSO du score, du modèle site et du stencil ;
3. structure, localité et ablations du Hamiltonien ;
4. capacité du solveur QAOA à atteindre l’objectif ;
5. résultat fermé Q-HAS contre classique à budget égal ;
6. hétérogénéité par scénario et coût de calcul.

## 5. Discussion

Si Q-HAS gagne : borner l’affirmation au panel et distinguer gain du mapping,
gain de l’optimiseur et coût de simulation. S’il perd : distinguer échec de la
représentation, de l’optimisation et de la spécification des labels. Si le
résultat est inconclusif : publier les bornes et la puissance obtenue sans
transformer l’absence de preuve en équivalence.

## Figures minimales

1. schéma du pipeline et des unités de réplication ;
2. fuite : split aléatoire contre split bloqué/LOSO ;
3. courbe du cône de voisinage ;
4. deltas physiques appariés avec IC95 par fold ;
5. synthèse globale et coût de raffinement.

## Tableaux minimaux

- panel physique et contrôles DNS ;
- paramètres, budgets et versions ;
- résultats par fold ;
- ablations et limites.

Chaque nombre du texte doit pointer vers une clé d’artefact et une commande de
reproduction.
