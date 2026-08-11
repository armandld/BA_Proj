# Plan du preprint

Squelette du manuscrit. Contenu durable uniquement : ce qui figure ici est
destiné à se retrouver dans le papier. L'état d'avancement du chantier — ce
qui reste à relancer, à republier, à réoptimiser — est en **appendice A**, et
disparaîtra.

---

## Structure

1. Histoire
2. Objectif
3. Travaux liés
4. Mise en place des hypothèses
5. Comment V1 marche, et pourquoi cette forme
6. Comment le GBT fonctionne, et à partir de quoi *(court)*
7. Méthode : comment on étudie un modèle qu'on soupçonne
8. Étude des deux modèles — résultats et figures
9. Discussion : affirmation / réfutation
10. Limites et menaces à la validité
11. Reproductibilité
12. Conclusion — ce qui reste ouvert
13. Bibliographie

---

## 2. Objectif

Décider si un critère de raffinement fondé sur un Ising quantique résolu par
QAOA local, avec un léger cône d'information sur les voisins, a une valeur
au-delà de la baseline classique.

L'attente qui motive cette famille d'approches est que le quantique traite
mieux les problèmes combinatoires — ici l'interaction entre voisins — donc
qu'ajouter cette information devrait rendre l'AMR plus performante, tout en
restant calculable sur matériel quantique (ce travail n'utilise que des
simulations).

Si aucun avantage n'est trouvé, déterminer **ce qui échoue** : la sélection,
la représentation, la forme du modèle, la spécification de la tâche, ou le
fait même de faire du ML.

**Préalable méthodologique.** Cette question ne peut être posée qu'à un
modèle dont on sait qu'il calcule ce que sa documentation annonce. Une part
substantielle de ce travail a consisté à établir cela. Elle n'était pas
prévue, et elle constitue une contribution à part entière.

---

## 4. Mise en place des hypothèses

### Le point de départ

Un instinct de physicien : mapper les instabilités de la grille dans un
hamiltonien. C'est l'idée centrale du papier. V1 lui donne une forme très
restreinte mais intuitive, dictée par la nature des instabilités. Ses
hyperparamètres sont optimisés sur quatre scénarios MHD, le seuil de la
baseline classique étant gelé pour que la comparaison porte sur ce que le
quantique ajoute et non sur un réglage différent.

Les performances mesurées ne sont pas bonnes. On veut comprendre pourquoi.

### Le solveur : ce qui est établi

- Ordre 4 sur le second membre seul (contre une évaluation spectrale :
  3,97 / 4,02 / 3,99).
- Solveur complet : ordre ~1,2.
- La chute vient de la projection d'incompressibilité, par un **splitting de
  Lie d'ordre 1** — `step_full` applique RK4 puis projette. Mesuré à grille
  **fixe** (N=128), en ne raffinant que le pas de temps :

| | erreur à 128 pas | ordre observé |
|---|---|---|
| avec projection | 3,27e−4 | **1,12** |
| sans projection | 9,17e−11 | **4,00** |

Sept ordres de grandeur : à grille fixe, seule l'erreur temporelle varie,
l'expérience est discriminante.

**La correction n'est pas un splitting de Strang.** Un splitting symétrique
suppose deux *flots* qu'on peut découper en demi-pas ; la projection n'en est
pas un, c'est un **projecteur idempotent**, et « P^(1/2) » n'a pas de sens.
Vérifié : `P ∘ RK4 ∘ P` rend des erreurs **identiques** à `P ∘ RK4` — après
le premier pas l'état est déjà dans le sous-espace, donc la projection
initiale est l'identité.

Le système est **différentiel-algébrique** : l'ordre chute parce que la
contrainte est imposée *après* un pas RK4 non contraint. Les deux corrections
qui tiennent sont de projeter le **second membre** à chaque étage — le champ
intégré est alors à divergence nulle par construction — ou de passer à une
formulation à pression.

Candidat non testé : la projection est spectrale alors que le second membre
est aux différences finies d'ordre 4 — deux opérateurs de divergence
incompatibles.

### Vingt et un défauts, chacun mesuré avant et après

C'est le matériau le plus solide du papier.

**Conventions et opérateurs**

| # | défaut | avant | après |
|---|---|---|---|
| D-1 | les mappeurs forment leur rotationnel sous `indexing='xy'` quand la grille déclare `'ij'` | 0,0 sur rotation solide | **+2,0** |
| D-3 | la fonction objectif pondère par cette même vorticité fausse | 0,0 | **+2,0** |
| D-11 | la diode de choc s'applique au **cisaillement** : normal et tangentiel échangés | rapport compression/cisaillement **0,500** ; diode inerte | **2,0** ; diode vivante |
| D-17 | trois sites hors de `src/` gardent la convention pré-D-1 | enstrophie tracée = **0 %** de sa valeur | **0,02 %** d'écart |
| — | le critère Q pondère la déformation de moitié et compte sa partie isotrope | cisaillement +0,25 | **0** |

L'opérateur fautif n'est pas un rotationnel de signe opposé — `abs` et le
carré n'auraient rien rattrapé. Il vaut `∂fy/∂y − ∂fx/∂x`, le
**complémentaire** du rotationnel : nul là où celui-ci est maximal.

**Numérique et rééchantillonnage**

| # | défaut | avant | après |
|---|---|---|---|
| D-2 | la prolongation AMR échantillonne au centre des cellules et utilise `mode='wrap'`, non périodique | 2,49e−1 | **7,74e−6** |
| D-7 | la projection ignore le mode de Nyquist : ni exacte ni idempotente | 0,378 | **1,1e−14** |
| D-14 | la réduction des champs tronque là où celle du score couvre tout | 94,1 % de couverture à la profondeur 3 | **100 %** |
| D-21 | le flux descend par un lissage puis un échantillonnage bilinéaire | pic conservé à **38 %** | **100 %** |

**Encodage et décision**

| # | défaut | avant | après |
|---|---|---|---|
| D-8 | l'hamiltonien borné encode des coefficients exactement nuls sans lever | non détecté | **lève** |
| D-13 | les bords gauche et haut lisent l'arête **intérieure** | asymétrie de 1,2 à 7,0 % sur patch symétrique | **symétrique** |
| D-15 | `postprocess` accepte des comptes bruts pour des probabilités | marginales ~1000, tout au-dessus du seuil | **refusé** |
| D-16 | la liste de patchs AMR **se recouvre elle-même** | jusqu'à 25 % du domaine compté deux fois | **0 %, sans trou** |
| D-19 | un backend inconnu construit un contexte mort sans erreur | panne loin de sa cause | **lève** |
| D-20 | le cache d'ansatz confond deux hamiltoniens | même objet pour des coefficients disjoints | **séparés** |

**Mesure et documentation**

| # | défaut | avant | après |
|---|---|---|---|
| D-4 | la documentation de la pondération annonce le double du facteur appliqué | ×2 | aligné |
| D-5 | le chemin de divergence note avec une formule non pondérée, sous la même clé | 1,8 % | **0** |
| D-6 | `init_magnetic_twist` ne pose aucune torsion | 6,4e−7 rad | **π/2 exact** |
| D-9 | l'ablation ψ mesure la fenêtre sur le mauvais score | « annihilation sur 42 ordres » | **ZZ domine K de 1,5 à 8,2×** |
| D-12 | le mappeur `study/` annonce que ν, η et dx influencent sa sortie | trois des quatre sans effet | documenté |

### Ce que l'audit de contrat apporte comme méthode

Douze des vingt et un ont été trouvés en changeant de question. Au lieu de
vérifier des valeurs, on demande à chaque fonction : **pourquoi existe-t-elle,
que promet-elle, consomme-t-elle les entrées que sa signature annonce, rend-
elle la forme et le domaine promis, et deux chemins censés coïncider
coïncident-ils encore ?**

Cette dernière question a produit la moitié des trouvailles — D-11 (la diode
contre sa propre docstring), D-13 (le bord gauche contre le bord droit),
D-14 et D-21 (la réduction des champs contre celle du score), D-9 (l'ablation
contre le pipeline). Aucun test de valeur ne pouvait les voir : **tous
rendent un résultat plausible**, indiscernable d'un résultat juste.

### Deux lectures antérieures renversées

**Le mécanisme du couplage ZZ.** L'explication publiée — « la fenêtre
gaussienne annihile le ZZ, donc l'ablater ne change rien » — était mesurée
sur le mauvais score. Le vrai mécanisme : l'état fondamental est **uniforme
sur 100 %** des instantanés, `no_ZZ` change la décision de 0,000 et `no_Z`
de 0,750. Un couplage ferromagnétique est trivialement satisfait par un état
uniforme : **le problème ne contient aucune frustration à la taille
déployée**, il n'y a rien de combinatoire à résoudre. C'est un argument plus
fort que l'annihilation, et il renforce H0b.

**Le terme ZZZZ était aveugle aux vortex.** Deux tests affirmaient qu'un
vortex de Lamb-Oseen ne gagne aucun contraste spatial. Ils mesuraient juste ;
la cause n'était pas le QAOA mais D-1. Attribution sur le même vortex,
16 tirages par ligne, tout le reste égal :

| convention du rotationnel | contraste | σ | max\|K\| |
|---|---|---|---|
| historique | −0,00725 | −3,4 | 0,0553 |
| **corrigée** | **+0,05672** | **+5,7** | **1,2545** |

Le coefficient de plaquette est **23× plus grand** dès que le rotationnel
voit la rotation. Le terme dont la seule raison d'être est de détecter une
circulation était numériquement mort sur un vortex pur.

### Quel mappeur a produit le nombre ?

Le dépôt contient **deux** mappeurs, et ils ne sont pas interchangeables.

| | v1 | v2 |
|---|---|---|
| utilisé par | **la boucle fermée** — niveau 3, Pareto | les analyses de `study/` |
| hyperparamètres | σ, β_curl, γ_hydro, γ_mag, κ, w_z_frac — **entraînés** | aucun |
| ν, η, dx | entrent via `Re = v_jump·dx/ν` | **n'entrent pas** |
| échelle des champs | agit | **sans effet** (×10 → sortie identique) |

Le v2 est **adimensionnel** : dx de 1,0 à 0,001 laisse les coefficients
bit-à-bit identiques, ν et η sont absents du fichier. Conséquence directe
pour **H4** : un transfert entre nombres de Reynolds est *trivialement*
satisfait par les coefficients du v2, puisque Re n'y entre pas — toute
dépendance en Re ne peut venir que du score externe. Chaque verdict doit dire
lequel des deux l'a produit.

### Ce que le circuit peut déplacer, et par quel canal

La couche de coût `exp(−iγH)` est **diagonale** : elle n'ajoute que des
phases et ne peut déplacer **aucune** probabilité de mesure — mesuré en
balayant γ de 0 à 2π, écart maximal **4,4e−16**. Seul le mixeur
`exp(−iβ ΣXᵢ)` déplace `P(|1⟩)`, et β est borné par construction à
`π/(4·reps) = 0,393 rad`.

Tout ce que l'hamiltonien apporte à la décision passe donc par son
interaction avec le mixeur. En balayant toute la grille admissible — ce qu'un
optimiseur *parfait* atteindrait :

| | médiane sur 5 patchs réels |
|---|---|
| mixeur seul | 0,254 |
| mixeur + hamiltonien | 0,490 |
| **apport de l'hamiltonien** | **0,236** |

L'hamiltonien n'est pas inerte. Mais **le témoin correct pour mesurer son
apport est le mixeur seul, pas le score classique**. « Le QAOA déplace-t-il
la décision ? » ne distingue pas « le mixeur la déplace » de « la physique la
déplace ».

### Les hypothèses

**H0 — le problème vient de la sélection** (qualité de l'optimisation
variationnelle à profondeur p).

- **H0a** *(non tranchée)* — l'optimiseur atteint-il l'optimum de son propre
  hamiltonien ? À 8 qubits, huit solveurs y arrivent et rendent le même
  masque — mais l'état fondamental y est uniforme sur 100 % des instantanés,
  donc l'accord porte sur un problème sans structure. À 18 qubits, le QAOA
  n'atteint l'optimum que sur 6 à 16 % des instantanés.
- **H0b** *(réfutée à 18 qubits ; non testable à 8, faute de variation
  d'énergie)* — mieux atteindre l'état fondamental améliore-t-il la tâche ?
  **Non** : l'énumération exacte détecte moins bien (F1 0,391) que le QAOA
  qui n'y arrive presque jamais (0,491 à 0,539), et que la règle classique
  (0,471). C'est H0b, et non H0a, qui retire à l'optimiseur son rôle
  explicatif.

**H1 — les défauts d'autre origine sont secondaires** *(mise en difficulté)*.
Les mesures penchent contre : la prolongation AMR introduisait 1,7 % d'erreur
sur le fond non raffiné ; la projection n'était pas idempotente ; le score
classique n'a pas de zéro absolu et est utilisé par les **deux** bras ; deux
formules de score divergeaient de 1,8 % dans la même campagne. H1 doit être
reformulée en question quantitative — *quelle part de l'écart à la référence
est imputable au numérique ?* — et non en présupposé.

**H2 — l'échec vient de la forme du modèle.**

- **QH2a** *(ouverte, hors de portée)* — l'idée de restreindre est bonne,
  mais les restrictions de V1 sont peut-être au mauvais endroit. Existe-t-il
  un modèle restrictif autre que V1 qui batte la baseline ?
- **H2b** *(non répondue)* — le modèle est-il trop restrictif ? L'ancienne
  réfutation ne tient plus depuis la correction du label.

**H3 — l'information des voisins.**

- **H3a** *(confirmée, en distribution seulement)* — le gain d'un cône de
  45 features sur 9 features locales est faible et **décroît** quand on
  affine : +0,018 / +0,015 / +0,010 à 16×16, 8×8, 4×4, avec 16× plus
  d'échantillons à chaque pas.
- **H3b** *(réfutée)* — sous transfert (LOSO), le GBT **s'améliore** avec
  l'information des voisins : +0,019 / +0,255 / +0,030 (dim 16) et
  +0,048 / +0,064 (dim 64), tous positifs, IC excluant zéro.

*Réserve structurelle sur toute ablation ZZ / ZZZZ.* Les deux portes qui
pilotent ces termes, `g_strain` et `g_rot`, somment à **1 exactement** pour
tout Q — identité algébrique, `1/(1+e^x) + 1/(1+e^−x) = 1`. Elles ne peuvent
jamais être actives ensemble ni inactives ensemble. Le ZZ et le ZZZZ sont une
**partition d'un unique scalaire d'Okubo-Weiss**, pas deux détecteurs
indépendants : une ablation déplace le poids d'un côté à l'autre du même
signal, elle ne retire pas une source. *(Même famille : dans la branche
hydrodynamique, `f_Re` et `mic_v` sont deux reparamétrages monotones du même
scalaire, égaux à 1e−12 près.)*

**H4 — l'échec vient de ce qu'on fait du ML, quantique ou non**
*(aperçu seulement)*. Un seul cas de non-transfert subsiste après correction
du label, `harris_tearing`, non expliqué. L'argument « deux modèles ne
transfèrent pas » ne tient plus. Affaibli en outre par l'adimensionnalité du
v2 (voir plus haut).

**H5 — l'échec vient de la spécification de la tâche**
*(partiellement établie)*. Objectif d'entraînement, label, score de
référence :

- la fonction objectif pondérait par une vorticité nulle sur toute rotation
  solide (D-3) ;
- le label était un rang intra-scénario : 25 % de patchs durs partout par
  construction, seuils variant d'un facteur 2,8 ;
- le score classique n'a **pas de zéro absolu** — champ uniforme + bruit
  1e−12 donne une médiane de 0,2372 et un maximum de 0,6571 — et il est **non
  monotone** en quantité de structure. Il est utilisé par les deux bras.

Sans H5, « le modèle échoue » n'est pas séparable de « la cible était mal
posée ».

### Tableau des verdicts

| hypothèse | verdict | portée |
|---|---|---|
| H0a | **non tranchée** | trivial à 8 qubits, non atteint à 18 |
| H0b | **réfutée** | à 18 qubits |
| H1 | **mise en difficulté** | quantifiée, pas close |
| H2b | **non répondue** | l'ancienne réfutation est caduque |
| QH2a | **ouverte** | hors de portée |
| H3a | **confirmée** | en distribution seulement |
| H3b | **réfutée** | GBT, LOSO, quatre scénarios |
| H4 | **aperçu seulement** | un cas inexpliqué |
| H5 | **partiellement établie** | trois défauts mesurés |

### Ce que le papier peut et ne peut pas conclure

Sur H3, l'énoncé défendable est :

> Le gain apporté par l'information des voisins est **réel mais petit**, il
> **décroît** quand on affine la grille, et il ne justifie pas le coût d'un
> dispositif quantique.

C'est un argument **économique**, pas un argument d'inutilité. Plus faible
que « les voisins ne servent à rien », mais vrai.

**Le résultat qui ferme le plus directement l'approche est H0b, pas H3.** Le
pari de départ est que le quantique optimise mieux le combinatoire. H0b
montre que **mieux optimiser n'améliore pas la tâche** : l'énergie est
décorrélée de l'objectif. C'est plus direct que H3, qui attaque la valeur de
l'information des voisins ; H0b attaque la valeur de l'optimisation, qui est
précisément ce qu'on paierait en qubits. H0b doit être le résultat mis en
avant.

---

## 7. Méthode : comment on étudie un modèle qu'on soupçonne

Cette section est une contribution, pas un préambule.

- **Auditer les contrats, pas les valeurs.** Pour chaque fonction : que
  promet-elle, consomme-t-elle ce que sa signature annonce, rend-elle la
  forme et le domaine promis, et deux chemins censés coïncider coïncident-ils
  encore ? Douze des vingt et un défauts viennent de là.
- **Un test doit pouvoir échouer.** Une assertion à seuil calibrée sur la
  mesure du jour ne mesure rien ; un balayage vide doit crier ; un script qui
  n'a rien mesuré doit être discernable d'un script qui a réussi.
- **Choisir le champ de validation qui sépare.** Sur Taylor-Green, les deux
  conventions de rotationnel rendent la **même** enstrophie, par symétrie de
  leurs carrés. Un test écrit sur ce champ passe sans rien vérifier.
- **Toute conclusion porte un intervalle.** Bootstrap par trajectoire, le
  bloc étant l'instantané et non le patch. Refus explicite de conclure quand
  l'intervalle contient zéro.
- **Un prédicteur constant ne vote pas.** Les folds où un bras prédit
  toujours la même classe sont écartés, et leur nombre est publié.
- **Le split aléatoire ne mesure pas le transfert** — des patchs du même
  instantané se retrouvent des deux côtés. Il ne vaut que comme **plafond**,
  et l'argument devient *a fortiori*.
- **Chaque nombre publié est recalculé depuis son artefact** par un
  agrégateur.

---

## 10. Limites et menaces à la validité

- **Un seul solveur, quatre scénarios, une résolution principale.** Rien ne
  garantit la généralité.
- **`harris_tearing` dégénère dans toutes les configurations testées**, sans
  explication.
- **Le bras QAOA n'est pas déterministe.** Sur 45 paires d'appels
  identiques : dispersion par appel de 1,79e−1 à 3,61e−1, auto-corrélation de
  rang médiane 0,933, minimum 0,350. Les **valeurs** bougent, le
  **classement** tient — les conclusions fondées sur un ordre (budget
  apparié, top-k) sont robustes, celles qui reposeraient sur une valeur ne le
  seraient pas.
- **La baseline classique est elle-même défectueuse** (score sans zéro
  absolu) et partagée par les deux bras. La comparaison est appariée, mais
  sur une référence discutable.
- **8 qubits en déploiement** : conclusions extrapolées depuis un régime où
  l'instance est souvent triviale — et où l'on sait maintenant qu'elle est
  sans frustration.
- **Les hyperparamètres viennent d'une campagne incomplète** : 345 essais,
  phase 1 seulement, interrompue, jugée par l'objectif défectueux D-3.
- **Le témoin « mixeur seul » n'est pas dans les campagnes.** Sans lui,
  l'apport de l'hamiltonien n'est pas séparable d'une rotation de mixeur.

---

## 11. Reproductibilité

- Chaque `.npz` porte le hash git et les arguments CLI complets.
- L'agrégateur recalcule les nombres publiés depuis les artefacts.
- La campagne Optuna est une **entrée gelée**, documentée avec ses chiffres
  vérifiés : 345 essais, 224 h CPU, 47 h de mur, phase 1 seule, interrompue.
- Les campagnes longues sont reprenables, avec refus de reprise si les
  réglages ont changé.
- Les corrections qui changent un nombre publié sont réversibles par drapeau
  (`fixed_curl`, `fixed_flux`), pour que l'ancien chemin reste reproductible
  bit à bit.

---

# Appendice A — état du chantier

*Cette section décrit un état transitoire et disparaîtra du manuscrit.*

Les campagnes n'ont pas été relancées depuis les corrections. Ordre contraint,
chaque étape conditionnant la suivante :

1. **Réoptimisation ciblée** de `beta_curl`, `kappa`, `threshold_amr`
   (~60 essais). Les hyperparamètres actuels ont été ajustés contre un
   pipeline portant D-1, D-11, D-13, D-14, D-16 et D-21 actifs. D-1 change
   `Q_OW`, `ω_z` et `J_z`, donc les trois portes que κ pilote ; D-14 et D-21
   changent les champs et le flux grossiers ; D-16 change la liste de patchs,
   donc le coût contre lequel λ arbitre. Ce sont les optima d'un autre
   problème. *Réserve : `g_strain + g_rot ≡ 1` signifie que κ ne contrôle
   qu'un seul degré de liberté ; l'espace est plus petit qu'il n'en a l'air.*
2. **Relance des campagnes** sur le code corrigé. En particulier, le terme
   ZZZZ n'était pas visible côté v1 : sa remise en état peut déplacer H3.
3. **Republication ou justification** des seize lignes de la table maître qui
   ne se recalculent plus depuis leurs artefacts (t17, t11b — exactement les
   nombres que les corrections ont déplacés). Publier une valeur qu'aucun
   artefact ne produit est ce que le dépôt s'interdit.
4. **Ajout du témoin « mixeur seul »** aux campagnes H0b.

Aucune conclusion actuelle n'est *invalidée* par ces étapes : elles sont **en
attente de confirmation** sur le code corrigé, ce qui n'est pas la même
chose. Le papier peut d'ores et déjà défendre la **méthode** et les **vingt et
un défauts mesurés**, qui n'en dépendent pas.
