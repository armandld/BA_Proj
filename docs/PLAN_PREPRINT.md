# Plan du preprint

Structure mère du manuscrit. **On ne s'étale ni sur les défauts ni sur les
résultats** : on dit ce qu'on a, et on renvoie.

| fichier | contenu |
|---|---|
| **`PLAN_PREPRINT.md`** (ce fichier) | la structure |
| `DEFAUTS.md` | les défauts, ce qui les a révélés, comment les retester |
| `RESULTS.md` | les résultats, comment ils ont été obtenus, comment les réobtenir |

---

## 1. Histoire

D'où vient l'idée, et pourquoi elle est séduisante : mapper les instabilités
d'une grille MHD dans un hamiltonien d'Ising, et laisser un solveur quantique
arbitrer le raffinement.

## 2. Objectif

Décider si un critère de raffinement fondé sur un Ising quantique résolu par
QAOA local, avec un léger cône d'information sur les voisins, a une valeur
au-delà de la baseline classique.

L'attente qui motive cette famille d'approches : le quantique traite mieux
les problèmes combinatoires — ici l'interaction entre voisins — donc ajouter
cette information devrait rendre l'AMR plus performante, tout en restant
calculable sur matériel quantique (ce travail n'utilise que des simulations).

Si aucun avantage n'est trouvé, déterminer **ce qui échoue** : la sélection,
la représentation, la forme du modèle, la spécification de la tâche, ou le
fait même de faire du ML.

**Préalable.** Cette question ne peut être posée qu'à un modèle dont on sait
qu'il calcule ce que sa documentation annonce. Établir cela a occupé une part
substantielle du travail, n'était pas prévu, et constitue une contribution à
part entière. → `DEFAUTS.md`

## 3. Mise en place des hypothèses

**H0 — l'échec vient de la sélection.** Qualité de l'optimisation
variationnelle à profondeur p.
- **H0a** — l'optimiseur atteint-il l'optimum de son propre hamiltonien ?
- **H0b** — mieux l'atteindre améliore-t-il la tâche ?

**H1 — les défauts d'autre origine (solveur, numérique) sont secondaires.**

**H2 — l'échec vient de la forme du modèle.**
- **QH2a** — existe-t-il un modèle restrictif *autre* que V1 qui batte la
  baseline ?
- **H2b** — le modèle est-il simplement trop restrictif ?

**H3 — l'information des voisins.**
- **H3a** — le cône apporte-t-il un gain en distribution ?
- **H3b** — en apporte-t-il un sous transfert ?

**H4 — l'échec vient de ce qu'on fait du ML, quantique ou non.**

**H5 — l'échec vient de la spécification de la tâche.** Objectif
d'entraînement, label, score de référence.

## 4. Comment V1 marche — et pourquoi ça, intuitivement

L'instinct de départ : les instabilités MHD ont une **structure locale
d'interaction**, donc elles se prêtent à un hamiltonien de spins sur les
arêtes de la grille.

- l'encodage : score classique → θ, flux de contrainte → ψ ;
- les trois termes — biais Z, couplage ZZ de gradient, plaquette ZZZZ de
  circulation — et ce que chacun est censé détecter ;
- les portes physiques qui les modulent (Reynolds, Okubo-Weiss, activité
  magnétique) ;
- la forme volontairement restreinte, et pourquoi elle l'est.

Deux faits structurels à énoncer ici, parce qu'ils conditionnent la lecture
de H0b et de H3 :

- la couche de coût est **diagonale** — seul le mixeur déplace une
  probabilité de mesure, et il est borné ;
- les portes `g_strain` et `g_rot` somment à **1 exactement** — ZZ et ZZZZ
  partitionnent un unique scalaire, ils ne sont pas deux détecteurs
  indépendants.

## 5. Comment le GBT fonctionne, à partir de quoi *(court)*

Features locales contre features en cône, le protocole d'entraînement, et le
rôle de témoin qu'il joue vis-à-vis de V1.

## 6. Étude des deux modèles

**L'approche.** Comment on étudie un modèle qu'on soupçonne :

- **auditer les contrats, pas les valeurs** — pourquoi une fonction existe,
  ce qu'elle promet, ce qu'elle consomme, et si deux chemins censés
  coïncider coïncident encore ;
- **un test doit pouvoir échouer** — pas de seuil calibré sur la mesure du
  jour, un balayage vide doit crier ;
- **toute conclusion porte un intervalle** — bootstrap par trajectoire, bloc
  = instantané ; refus de conclure quand l'intervalle contient zéro ;
- **un prédicteur constant ne vote pas** ;
- **le split aléatoire ne mesure pas le transfert** — il ne vaut que comme
  plafond, et l'argument devient *a fortiori* ;
- **chaque nombre publié se recalcule depuis son artefact.**

**Les graphes qui en résultent.** → `RESULTS.md`

## 7. Discussion — affirmation, réfutation, ce qui reste ouvert

Verdict par hypothèse, avec sa portée. Certaines restent ouvertes ; on le
dit.

Ce qui doit être mis en avant : **H0b ferme l'approche plus directement que
H3**. Le pari de départ est que le quantique optimise mieux le combinatoire ;
H0b montre que mieux optimiser n'améliore pas la tâche. C'est la valeur de
l'optimisation qui est attaquée — précisément ce qu'on paierait en qubits.

Sur H3, l'énoncé défendable est **économique**, pas un argument d'inutilité.
La courbe de cône a désormais son premier artefact (`dim = 8`, → `RESULTS.md`)
et va dans ce sens : gain faible en distribution, aucun sous transfert. Deux
choses manquent encore pour que la clause « décroît quand on affine » soit
mesurée et non postulée — le balayage en `dim`, et l'explication du pli
`harris_tearing`, mort à tous les k :

> Le gain apporté par l'information des voisins est réel mais petit, il
> décroît quand on affine la grille, et il ne justifie pas le coût d'un
> dispositif quantique.

Les limites qui bornent ces conclusions — un seul solveur, quatre scénarios,
8 qubits en déploiement, baseline partagée par les deux bras, non-déterminisme
du bras QAOA, chute d'ordre du solveur commune aux deux bras — sont énoncées
ici, chiffrées dans `RESULTS.md`.

## 8. Conclusion

Ce que le travail tranche, ce qu'il laisse ouvert, et les questions qu'il
formule pour la suite.

## 9. Bibliographie

---

## Appendice A — état du chantier

*Transitoire, disparaîtra du manuscrit.*

Les campagnes n'ont pas été relancées depuis les corrections. Ordre contraint,
chaque étape conditionnant la suivante :

1. **Réoptimisation.** Les hyperparamètres déployés ne correspondent à aucune
   base du dépôt, et trois d'entre eux n'ont jamais été échantillonnés →
   `DEFAUTS.md`, D-22.
2. **Relance des campagnes** sur le code corrigé.
3. **Republication ou justification** des 16 lignes de la table maître qui ne
   se recalculent plus.
4. **Ajout du témoin « mixeur seul »** aux campagnes H0b — sans lui, l'apport
   de l'hamiltonien n'est pas séparable d'une rotation de mixeur.

**Une conclusion est désormais invalidée, pas seulement en attente.** À toute
profondeur de raffinement supérieure à la première, le biais Z de
l'hamiltonien et ses couplages décrivaient deux grilles différentes : le biais
d'un patch venait du quart haut-gauche de ce patch (D-37, écart 41 % du plus
grand coefficient, présent depuis le premier commit). À `max_depth = 4`,
réglage de toutes les campagnes, trois niveaux sur quatre passaient par là.

Le bras classique n'est pas touché — il ne construit aucun hamiltonien. La
comparaison des deux bras était donc biaisée dans un sens connu.

Le reste est **en attente de confirmation** sur le code corrigé, ce qui n'est
pas la même chose qu'invalidé → `EVALUATION.md`.
