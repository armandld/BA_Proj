# Plan du preprint — version révisée

> Les modifications par rapport au plan initial sont marquées `[M-n]` et
> récapitulées en fin de document.

---

## Structure

1. Histoire
2. Objectif
3. **Travaux liés** `[M-1]`
4. Mise en place des hypothèses
5. Comment V1 marche ? Pourquoi ça, intuitivement ?
6. Comment GBT fonctionne ? À partir de quoi ? *(bien plus court)*
7. **Méthode : comment on étudie un modèle qu'on soupçonne** `[M-2]`
8. Étude des deux modèles — résultats et figures
9. Discussion : affirmation / réfutation
10. **Limites et menaces à la validité** `[M-3]`
11. **Reproductibilité** `[M-4]`
12. Conclusion — ce qui reste ouvert
13. Bibliographie

---

## 2. Objectif

On veut décider si un critère de raffinement fondé sur un Ising quantique
résolu par QAOA local, avec un léger cône d'information sur les voisins, a
une valeur au-delà de la baseline classique.

L'attente qui motive cette famille d'approches est que le quantique traite
mieux les problèmes combinatoires — ici, l'interaction entre voisins — donc
qu'ajouter cette information devrait rendre l'AMR plus performante, tout en
restant calculable sur matériel quantique (ce travail n'utilise que des
simulations).

Si aucun avantage n'est trouvé, on veut déterminer **ce qui échoue** :
la sélection, la représentation, la forme du modèle, la spécification de la
tâche, ou le fait même de faire du ML.

`[M-5]` **Ajout d'un préalable méthodologique.** Cette question ne peut être
posée qu'à un modèle dont on sait qu'il calcule ce que sa documentation
annonce. Une part substantielle de ce travail a donc consisté à établir
cela, et n'était pas prévue au départ. Neuf défauts ont été trouvés et
corrigés dans V1, dont un dans la fonction objectif elle-même. Le papier
doit dire que **les résultats de performance publiés avant ces corrections
ne sont pas ceux du modèle corrigé** — c'est la raison pour laquelle
l'étude est refaite.

---

## 4. Mise en place des hypothèses

### Le point de départ

On a suivi un instinct de physicien : mapper les instabilités de la grille
dans un hamiltonien. C'est l'idée centrale du papier. Pour V1, on a donné à
cet hamiltonien une forme très restreinte mais intuitive, dictée par la
nature des instabilités. Ses hyperparamètres ont été optimisés sur quatre
scénarios MHD typiques — le seuil de la baseline classique étant gelé, pour
que la comparaison porte sur ce que le quantique ajoute et non sur un
réglage différent.

Les performances mesurées ne sont pas bonnes. On veut comprendre pourquoi.

### Défauts connus au départ `[M-6]`

*(section entièrement réécrite : deux entrées devenues trois catégories)*

**Établi et mesuré.**

- Le solveur est d'ordre 4 sur le second membre seul (vérifié contre une
  évaluation spectrale : ordres 3,97 / 4,02 / 3,99).
- Le solveur complet converge à l'ordre ~1,2.
- `grid.py` emploie l'ordre 2 là où `solver.py` emploie l'ordre 4.

**Attribution CONFIRMÉE** `[M-6b]` — *correction d'une correction.* La
chute d'ordre vient bien de la projection d'incompressibilité, et le
mécanisme est un **splitting de Lie d'ordre 1** : `step_full` applique RK4
puis projette. Mesuré à grille **fixe** (N=128), en ne raffinant que le pas
de temps :

| | erreur à 128 pas | ordre observé |
|---|---|---|
| avec projection | 3,27e−4 | **1,12** |
| sans projection | 9,17e−11 | **4,00** |

Sept ordres de grandeur d'écart. C'est l'expérience discriminante propre :
à grille fixe, seule l'erreur temporelle varie.

*Note d'honnêteté.* Cette attribution avait été retirée du plan sur la foi
d'une sonde que j'avais écrite et qui était invalide — Orszag-Tang à N=16,
très sous-résolu, avec une référence trop proche, et un raffinement
simultané en espace et en temps. Elle ne mesurait aucun ordre asymptotique.
Le dépôt contenait déjà la bonne mesure, en section `[D]` de
`h1_solver_convergence.py`. Le plan initial avait raison.

**Conséquence pour le papier.** Le facteur limitant du solveur est
identifié et corrigeable : un splitting de Strang (demi-pas, projection,
demi-pas) rendrait l'ordre 2, et une formulation à pression rendrait
l'ordre du schéma. Ce n'est pas une limite de principe.

**Candidat non testé.** La projection est spectrale alors que le second
membre est aux différences finies d'ordre 4 : deux opérateurs de divergence
incompatibles. Mécanisme plausible, non vérifié.

**Choix de discrétisation.** Différences finies là où des volumes finis
auraient été préférables pour une loi de conservation.

### Défauts trouvés pendant l'étude `[M-7]`

*(section entièrement nouvelle — c'est le matériau le plus solide du
papier, et il n'y figurait pas)*

Neuf défauts, chacun mesuré avant et après correction :

| # | défaut | avant | après |
|---|---|---|---|
| D-1 | les mappeurs forment leur rotationnel sous `indexing='xy'` alors que la grille déclare `'ij'` — leur « vorticité » est en fait une différence de déformations normales | 0,0 sur rotation solide | **+2,0** |
| D-2 | la prolongation AMR échantillonne au centre des cellules et utilise `mode='wrap'`, qui n'est pas périodique | 2,49e−1 | **7,74e−6** |
| D-3 | **la fonction objectif pondère par la même vorticité fausse** | 0,0 | **+2,0** |
| D-4 | la documentation de la pondération annonce le double du facteur appliqué | ×2 | aligné |
| D-5 | le chemin de divergence note avec une formule non pondérée, sous la même clé | 1,8 % | **0** |
| D-6 | `init_magnetic_twist` ne pose aucune torsion : la projection annule sa composante transverse | 6,4e−7 rad | **π/2 exact** |
| D-7 | la projection ignore le mode de Nyquist : elle n'est ni exacte ni idempotente sur champ bruité | 0,378 | **1,1e−14** |
| D-8 | le hamiltonien borné encode des coefficients exactement nuls sans lever | non détecté | **lève** |
| — | le critère Q pondère la déformation de moitié et compte sa partie isotrope | cisaillement +0,25 | **0** |
| D-9 | l'ablation ψ mesure la fenêtre sur `physical_score` là où le pipeline fournit `classical_score` | « annihilation sur 42 ordres » | **ZZ domine K de 1,5 à 8,2×** |
| D-11 | la diode de choc s'applique au **cisaillement** : les rôles normal/tangentiel sont échangés | rapport compression/cisaillement **0,500** ; diode inerte | **2,0** ; diode vivante |
| D-12 | `PhysicalMapperV2` annonce que ν, η et dx influencent sa sortie | trois des quatre sans effet | docstring alignée |
| D-13 | les bords gauche et haut de l'hamiltonien lisent l'arête **intérieure** | asymétrie de 1,2 à 7,0 % sur patch symétrique | **symétrique** |
| D-14 | la réduction des champs tronque là où celle du score couvre tout | 94,1 % de couverture à la profondeur 3 | **100 %** |
| D-15 | `postprocess` accepte des comptes bruts pour des probabilités | marginales ~1000, tout au-dessus du seuil | **refusé** |
| D-16 | la liste de patchs AMR **se recouvre elle-même** | jusqu'à 25 % du domaine compté deux fois | **0 %, sans trou** |
| D-17 | trois sites hors de `src/` gardent la convention pré-D-1 | enstrophie tracée = **0 %** de sa valeur | **0,02 %** d'écart |
| D-19 | un backend inconnu construit un `VQARuntime` mort sans erreur | panne à des dizaines de lignes de sa cause | **lève** |
| D-20 | le cache d'ansatz confond deux hamiltoniens | même objet pour des coefficients disjoints | **séparés** |
| D-21 | le flux descend par un lissage puis un échantillonnage bilinéaire | pic conservé à **38 %** (orszag_tang) | **100 %** |

**D-3 est le plus lourd pour l'interprétation.** Les 345 essais Optuna ont
été jugés par une fonction objectif dont la pondération était aveugle à
toute rotation solide. Aucun résultat de performance antérieur n'est donc
attribuable au modèle seul.

`[M-14]` **La table passe de neuf à vingt et un défauts.** Les douze
nouveaux viennent d'un **audit de contrat** : au lieu de vérifier des
valeurs, on demande à chaque fonction ce qu'elle prétend faire, si elle
consomme les entrées que sa signature annonce, si elle rend la forme et le
domaine promis, et si deux chemins censés coïncider coïncident encore.

Cette dernière question a produit la moitié des trouvailles — D-11 (la diode
contre sa propre docstring), D-13 (le bord gauche contre le bord droit),
D-14 et D-21 (la réduction des champs contre celle du score). Aucun test de
valeur ne pouvait les voir : **tous rendent un résultat plausible**.

`[M-15]` **Deux d'entre eux renversent une lecture publiée.**

**D-9** — le mécanisme annoncé (« la fenêtre gaussienne annihile le
couplage ZZ, donc l'ablater ne change rien ») est **faux**. Il avait été
mesuré sur le mauvais score. Le vrai mécanisme, lu sur le chemin correct :
l'état fondamental est **uniforme sur 100 %** des instantanés, `no_ZZ`
change la décision de 0,000 et `no_Z` de 0,750. Un couplage ferromagnétique
est trivialement satisfait par un état uniforme : **le problème ne contient
aucune frustration à la taille déployée**, il n'y a rien de combinatoire à
résoudre. C'est un argument *plus fort* que l'annihilation, et il renforce
H0b.

**D-1 rendait le terme ZZZZ aveugle aux vortex.** Deux tests QAOA
affirmaient qu'un vortex de Lamb-Oseen ne gagne aucun contraste spatial. Ils
mesuraient juste, mais la cause n'était pas le QAOA. Attribution sur le même
vortex, 16 tirages par ligne, tout le reste égal :

| `fixed_curl` | contraste | σ | max\|K\| |
|---|---|---|---|
| False (historique) | −0,00725 | −3,4 | 0,0553 |
| **True** | **+0,05672** | **+5,7** | **1,2545** |

Le coefficient de plaquette est **23× plus grand** dès que le rotationnel
voit la rotation. Le terme dont la seule raison d'être est de détecter une
circulation était numériquement mort sur un vortex pur. Réserve : mesuré sur
le harnais **v1** ; sur le v2 l'effet est une redistribution, pas une
amplification uniforme.

### Quel mappeur a produit le nombre ? `[M-16]`

*(section nouvelle — cette distinction n'apparaissait nulle part, et elle
change la portée de plusieurs verdicts)*

Le dépôt contient **deux** mappeurs d'hamiltonien, et ils ne sont pas
interchangeables :

| | `PhysicalMapper` (v1) | `PhysicalMapperV2` |
|---|---|---|
| utilisé par | **`src/pipeline.py`** — la boucle fermée, niveau 3, Pareto | les scripts de `study/` |
| hyperparamètres | σ, β_curl, γ_hydro, γ_mag, κ, w_z_frac — **entraînés** | aucun (poids fixes) |
| ν, η, dx | entrent via `Re_h = v_jump·dx/ν` | **n'entrent pas du tout** |
| échelle des champs | agit | **sans effet** (×10 → sortie identique) |

Le v2 est **adimensionnel** : mesuré, dx de 1,0 à 0,001 laisse `C`, `K` et
`H` bit-à-bit identiques, et ν et η sont absents du fichier. Deux
conséquences pour la lecture :

- **H4 (transfert)** — un transfert entre nombres de Reynolds est
  *trivialement* satisfait par les coefficients du v2, puisque Re n'y entre
  pas. Toute dépendance en Re ne peut venir que du score externe.
- Chaque verdict doit dire **lequel des deux** l'a produit. Un résultat v2
  ne se transporte pas tel quel à la boucle fermée, qui tourne sur le v1.

### Ce que le circuit peut déplacer, et par quel canal `[M-17]`

*(section nouvelle — c'est le contrôle qui manque à H0b)*

La couche de coût `exp(−iγH)` est **diagonale** : elle n'ajoute que des
phases et ne peut déplacer **aucune** probabilité de mesure. Mesuré en
balayant γ de 0 à 2π : écart maximal **4,4e−16**. Seul le mixeur
`exp(−iβ ΣXᵢ)` déplace `P(|1⟩)`, et β est borné par construction à
`π/(4·reps) = 0,393 rad`.

Tout ce que l'hamiltonien apporte à la décision passe donc par son
interaction avec le mixeur. En balayant toute la grille admissible — ce
qu'un optimiseur *parfait* atteindrait, pas ce que COBYLA trouve :

| | médiane sur 5 patchs réels |
|---|---|
| mixeur seul | 0,254 |
| mixeur + hamiltonien | 0,490 |
| **apport de l'hamiltonien** | **0,236** |

L'hamiltonien n'est pas inerte. Mais **le témoin correct pour mesurer son
apport est le mixeur seul, pas le score classique** — et aucune campagne du
dépôt n'utilise ce témoin. « Le QAOA déplace-t-il la décision ? » ne
distingue pas « le mixeur la déplace » de « la physique la déplace ».

### Les hypothèses

`[M-8]` **Ajout de H5** — le plan initial mentionnait « définition de la
cible » dans l'objectif mais n'avait aucune hypothèse pour la porter, alors
que c'est là que les défauts les plus lourds ont été trouvés.

**H0 — le problème vient de la sélection** (qualité de l'optimisation
variationnelle du QAOA à profondeur p).

- **H0a** *(non tranchée)* — l'optimiseur atteint-il l'optimum de son propre
  hamiltonien ? À 8 qubits, huit solveurs y arrivent et rendent le même
  masque — mais l'état fondamental y est uniforme sur 100 % des instantanés,
  donc l'accord porte sur un problème sans structure. À 18 qubits, où la
  structure apparaît, le QAOA n'atteint l'optimum que sur 6 à 16 % des
  instantanés.
- **H0b** *(réfutée à 18 qubits ; non testable à 8, faute de variation
  d'énergie)* — mieux atteindre l'état fondamental améliore-t-il la tâche ?
  **Non** : l'énumération exacte détecte moins bien (F1 0,391) que le QAOA
  qui n'y arrive presque jamais (0,491 à 0,539), et que la règle classique
  (0,471). C'est H0b, et non H0a, qui retire à l'optimiseur son rôle
  explicatif.

**H1 — les défauts d'autre origine (solveur, numérique) sont secondaires.**

`[M-9]` *Statut révisé : de « non testée » à « mise en difficulté ».* Les
mesures penchent contre : la prolongation AMR introduisait 1,7 % d'erreur
sur le fond non raffiné ; la projection n'était pas idempotente sur champ
bruité ; le score classique n'a pas de zéro absolu et est utilisé par les
**deux** bras ; deux formules de score divergeaient de 1,8 % dans la même
campagne. H1 doit être reformulée en question quantitative — *quelle part de
l'écart à la référence est imputable au numérique ?* — et non en
présupposé.

**H2 — l'échec vient de la forme du modèle.**

- **QH2a** *(question ouverte, non vérifiable ici)* — l'idée de restreindre
  est bonne, mais les restrictions de V1 sont peut-être au mauvais endroit.
  Existe-t-il un modèle restrictif autre que celui de V1 qui batte la
  baseline ?
- **H2b** *(non répondue)* — le modèle est-il simplement trop restrictif,
  faudrait-il un ML plus libre ? L'ancienne réfutation ne tient plus depuis
  la correction du label, et aucune conclusion positive n'est établie.

**H3 — l'information des voisins.**

`[M-18]` *Réserve structurelle sur toute ablation ZZ / ZZZZ.* Les deux
portes qui pilotent ces termes, `g_strain` et `g_rot`, somment à **1
exactement** pour tout Q — c'est une identité algébrique,
`1/(1+e^x) + 1/(1+e^−x) = 1`. Elles ne peuvent jamais être actives
ensemble, ni inactives ensemble.

Le ZZ et le ZZZZ sont donc une **partition d'un unique scalaire
d'Okubo-Weiss**, pas deux détecteurs indépendants. Retirer le ZZ ne retire
pas une source d'information distincte du ZZZZ : cela déplace le poids d'un
côté à l'autre du même signal. Toute phrase du papier les présentant comme
deux apports séparables est à réécrire.

Même famille, mesuré au passage : dans la branche hydrodynamique, `f_Re` et
`mic_v` sont deux reparamétrages **monotones du même scalaire** (égaux à
1e−12 près). Le coefficient affiche deux facteurs physiques là où il n'y a
qu'une variable.

- **H3a** *(confirmée, en distribution seulement)* — le gain apporté par un
  cône de 45 features sur les 9 features locales est faible et **décroît**
  quand on affine : +0,018 / +0,015 / +0,010 à 16×16, 8×8, 4×4 cellules,
  avec 16× plus d'échantillons à chaque pas.
- **H3b** *(réfutée)* — sous transfert (LOSO), le GBT **s'améliore** avec
  l'information des voisins : +0,019 / +0,255 / +0,030 (dim 16) et
  +0,048 / +0,064 (dim 64), tous positifs, IC excluant zéro.

**H4 — l'échec vient de ce qu'on fait du ML, quantique ou non.**
*(non vérifiable ici)* — un seul cas de non-transfert subsiste après
correction du label, `harris_tearing`, et il n'est pas expliqué. L'argument
« deux modèles ne transfèrent pas » ne tient plus.

**H5 — l'échec vient de la spécification de la tâche** `[M-8]`
*(nouvelle ; partiellement établie)* — objectif d'entraînement, label, score
de référence. Éléments déjà mesurés :

- la fonction objectif pondérait par une vorticité nulle sur toute rotation
  solide (D-3) ;
- le label était un rang intra-scénario : 25 % de patchs durs partout par
  construction, avec des seuils variant d'un facteur 2,8 ;
- le score classique n'a **pas de zéro absolu** — champ uniforme + bruit
  1e−12 donne une médiane de 0,2372 et un maximum de 0,6571, et il est
  **non monotone** en quantité de structure. Il est utilisé par les deux
  bras.

Sans H5, « le modèle échoue » n'est pas séparable de « la cible était mal
posée ».

### Ce que ce travail tranche `[M-10]`

*(le plan disait « H0, H3a, H3b » ; H0a n'est pas tranchée)*

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

### Ce que le papier peut et ne peut pas conclure `[M-11]`

*(correction d'une contradiction interne du plan initial)*

Le plan disait que « prouver H3 est le cœur du papier », et que H3 montre
que l'information des voisins est **inutile**. Ses propres entrées disent
l'inverse : H3b est **réfutée**, les voisins **aident** sous transfert.

L'énoncé défendable est :

> Le gain apporté par l'information des voisins est **réel mais petit**, il
> **décroît** quand on affine la grille, et il ne justifie pas le coût d'un
> dispositif quantique.

C'est un argument **économique**, pas un argument d'inutilité. Il est plus
faible que « les voisins ne servent à rien », mais il est vrai.

`[M-12]` **Le résultat qui ferme le plus directement l'approche est H0b, pas
H3.** Le pari de départ est que le quantique optimise mieux le combinatoire.
H0b montre que **mieux optimiser n'améliore pas la tâche** — l'énergie est
décorrélée de l'objectif. C'est plus direct que H3 : H3 attaque la valeur de
l'information des voisins, H0b attaque la valeur de l'optimisation, qui est
précisément ce qu'on paierait en qubits. H0b devrait donc être le résultat
mis en avant.

---

## 7. Méthode : comment on étudie un modèle qu'on soupçonne `[M-2]`

*(section nouvelle — le plan passait directement des hypothèses aux
résultats, sans exposer la méthode, qui est pourtant une contribution)*

- **Toute conclusion porte un intervalle.** Bootstrap par trajectoire, le
  bloc étant l'instantané et non le patch. Refus explicite de conclure quand
  l'intervalle contient zéro.
- **Un prédicteur constant ne vote pas.** Les folds où un bras prédit
  toujours la même classe sont écartés, et leur nombre est publié.
- **Le split aléatoire ne mesure pas le transfert** — des patchs du même
  instantané se retrouvent des deux côtés. Il ne vaut que comme **plafond**,
  et l'argument devient *a fortiori* : dans les conditions les plus
  favorables, le gain des voisins reste ≤ +0,018.
- **Chaque nombre publié est recalculé depuis son artefact** par un
  agrégateur (180 lignes, 0 écart, 0 manquant).
- **Un balayage vide doit crier.** Un script qui n'a rien mesuré doit être
  discernable d'un script qui a réussi.

---

## 10. Limites et menaces à la validité `[M-3]`

*(section nouvelle — indispensable pour un papier de falsification)*

- **Un seul solveur, quatre scénarios, une résolution principale.** Rien ne
  garantit la généralité.
- **`harris_tearing` dégénère dans toutes les configurations testées**, sans
  explication.
- **Le bras QAOA n'est pas déterministe.** Mesuré sur 45 paires d'appels
  identiques : dispersion par appel de 1,79e−1 à 3,61e−1, auto-corrélation
  de rang médiane 0,933, minimum 0,350. `[M-13]` *Les **valeurs** bougent, le
  **classement** tient* — donc les conclusions fondées sur un ordre (budget
  apparié, top-k) sont robustes, celles qui reposeraient sur une valeur ne le
  seraient pas.
- **La baseline classique est elle-même défectueuse** (score sans zéro
  absolu) et partagée par les deux bras. La comparaison est donc appariée,
  mais sur une référence discutable.
- **8 qubits en déploiement** : conclusions extrapolées à partir d'un régime
  où l'instance est souvent triviale.
- **Les hyperparamètres viennent d'une campagne incomplète** : phase1
  seulement, 345 essais, interrompue, jugée par l'objectif défectueux D-3.
- `[M-19]` **Les hyperparamètres optimisent un problème qui n'existe plus.**
  Ils ont été ajustés contre un pipeline portant D-1, D-11, D-13, D-14, D-16
  et D-21 actifs. Trois de ces défauts touchent directement ce que le
  mappeur v1 consomme : D-1 change `Q_OW`, `ω_z` et `J_z`, donc les **trois
  portes** que κ pilote ; D-14 et D-21 changent les champs et le flux
  grossiers qu'il reçoit ; D-16 change la liste de patchs, donc le coût
  contre lequel λ arbitre. Les valeurs de σ, β_curl, κ et θ_amr sont des
  optima **d'un autre problème**.

  Une réoptimisation ciblée s'impose donc avant toute reprise des chiffres
  de performance. Réserve : `g_strain + g_rot ≡ 1` signifie que κ ne
  contrôle **qu'un seul** degré de liberté ; l'espace à réoptimiser est plus
  petit qu'il n'en a l'air.
- `[M-20]` **Seize des cent quatre-vingts nombres publiés ne se recalculent
  plus depuis leurs artefacts.** L'agrégateur reste à 0 manquant, mais seize
  lignes diffèrent de leur référence — ce sont exactement les nombres que
  les corrections ont déplacés (t17, t11b). Ils doivent être **republiés ou
  justifiés ligne par ligne** avant soumission ; les laisser en écart
  reviendrait à publier des valeurs qu'aucun artefact ne produit.

---

## 11. Reproductibilité `[M-4]`

*(section nouvelle — c'est un atout du dépôt, absent du plan)*

- Chaque `.npz` porte le hash git et les arguments CLI complets.
- L'agrégateur recalcule les 180 nombres publiés depuis les artefacts.
- La campagne Optuna est une **entrée gelée**, documentée dans
  `PROVENANCE.md` avec ses chiffres vérifiés : 345 essais, 224 h CPU,
  47 h de mur, phase1 seule, interrompue.
- Les campagnes longues sont reprenables, avec refus de reprise si les
  réglages ont changé.

---

## Récapitulatif des modifications

| | modification | pourquoi |
|---|---|---|
| M-1 | ajout d'une section « travaux liés » | aucune mise en contexte n'était prévue |
| M-2 | ajout d'une section « méthode » | la méthode d'audit est une contribution, elle n'apparaissait pas |
| M-3 | ajout de « limites et menaces à la validité » | indispensable à un papier de falsification |
| M-4 | ajout de « reproductibilité » | atout réel du dépôt, non exposé |
| M-5 | préalable méthodologique dans l'objectif | les résultats antérieurs aux corrections ne sont pas ceux du modèle corrigé |
| M-6 | « défauts connus » réécrite en trois catégories | séparer le mesuré, l'attribué et le non testé |
| M-6b | **attribution de la chute d'ordre CONFIRMÉE** | splitting de Lie mesuré à grille fixe : ordre 1,12 avec projection contre 4,00 sans. J'avais retiré cette attribution sur la foi d'une sonde invalide ; le plan initial avait raison |
| M-7 | section « défauts trouvés pendant l'étude » | neuf défauts mesurés, absents du plan |
| M-8 | ajout de **H5** (spécification de la tâche) | l'objectif la mentionnait sans hypothèse pour la porter |
| M-9 | H1 passe de « non testée » à « mise en difficulté » | les mesures penchent contre |
| M-10 | tableau des verdicts ; H0a n'est pas tranchée | le plan disait « répond à H0 » |
| M-11 | correction de la contradiction sur H3 | le plan concluait « les voisins sont inutiles » alors que H3b est réfutée |
| M-12 | H0b mis en avant devant H3 | H0b ferme l'approche plus directement |
| M-13 | chiffres de non-déterminisme QAOA harmonisés | valeurs mesurées sur 45 paires, avec la distinction valeur/classement |
| M-14 | table des défauts portée de 9 à **21** | douze défauts trouvés par audit de contrat ; la moitié par la question « deux chemins censés coïncider coïncident-ils ? », qu'aucun test de valeur ne pose |
| M-15 | **deux lectures publiées renversées** | D-9 : le mécanisme n'est pas l'annihilation du ZZ mais l'**absence de frustration** — argument plus fort, qui renforce H0b. D-1 : le terme ZZZZ était **numériquement mort sur un vortex**, coefficient 23× plus petit |
| M-16 | section « quel mappeur a produit le nombre ? » | le dépôt en a **deux** et ils ne sont pas interchangeables : la boucle fermée tourne sur v1 (entraîné, dimensionnel), `study/` sur v2 (sans paramètre, **adimensionnel**). H4 s'en trouve affaiblie |
| M-17 | section « ce que le circuit peut déplacer » | γ seul ne déplace **rien** (4,4e−16) ; seul le mixeur agit, borné à 0,393 rad. Le témoin correct pour H0b est **le mixeur seul**, et aucune campagne ne l'utilise |
| M-18 | réserve structurelle sur toute ablation ZZ / ZZZZ | `g_strain + g_rot ≡ 1` : ce sont deux faces d'un **unique** scalaire, pas deux détecteurs. Une ablation déplace le poids, elle ne retire pas une source |
| M-19 | la réoptimisation devient un préalable, pas une option | les hyperparamètres optimisent un problème que six corrections ont modifié |
| M-20 | les 16 écarts de la table maître deviennent une dette explicite | publier une valeur qu'aucun artefact ne recalcule est exactement ce que le dépôt s'interdit |

---

## Ce qu'il reste à faire, dans l'ordre `[M-19]` `[M-20]`

Trois des sections ci-dessus décrivent un état qui n'existe pas encore : les
campagnes n'ont pas été relancées depuis les corrections. L'ordre contraint
est le suivant, chaque étape conditionnant la suivante :

1. **Réoptimisation ciblée** de `beta_curl`, `kappa`, `threshold_amr`
   (~60 essais). Sans elle, tout chiffre de performance mesure un réglage
   ajusté pour un pipeline différent.
2. **Relance des campagnes** sur le code corrigé — c'est la seule façon de
   savoir si les verdicts de H0b, H3 et H4 tiennent. En particulier : le
   terme ZZZZ n'était pas visible du côté v1 ; sa remise en état peut
   déplacer H3.
3. **Republication ou justification** des seize lignes en écart.
4. **Ajout du témoin « mixeur seul »** aux campagnes H0b — sans lui, l'apport
   de l'hamiltonien n'est pas séparable d'une rotation de mixeur.

Aucune des conclusions actuelles n'est invalidée par ces étapes ; elles sont
**en attente de confirmation** sur le code corrigé, ce qui n'est pas la même
chose. Le papier peut d'ores et déjà défendre la **méthode** et les
**vingt et un défauts mesurés**, qui ne dépendent d'aucune de ces relances.
