# Plan du preprint

Structure mère du manuscrit. **On ne s'étale ni sur les défauts ni sur les
résultats** : on dit ce qu'on a, et on renvoie.

| fichier | contenu |
|---|---|
| **`PLAN_PREPRINT.md`** (ce fichier) | la structure |
| `DEFAUTS.md` | les défauts, ce qui les a révélés, comment les retester |
| `RESULTS.md` | les résultats, comment ils ont été obtenus, comment les réobtenir |

---

## Reconstruction du 26 août — pourquoi ce fichier a une nouvelle histoire

Le commit `d3d7573` (« improvements and corrections of the tests, some
corrections on the src code », 24 août) a remplacé ce fichier — alors
**283 lignes, 9 sections + un appendice, la structure hypothèses/histoire/
discussion ci-dessous** — par un squelette de plan de papier à 5 sections
(Titre, Question, Méthodes, Validation du banc, Résultats, Discussion,
Figures, Tableaux) sans hypothèses `H0`–`H5`, sans verdict, sans discussion :
59 lignes neuves, 224 supprimées, dans le même geste qui a vidé
`DEFAUTS.md`, `COUVERTURE.md`, `RESULTS.md`, `EVALUATION.md` et
`CODE_REVIEW.md` sans rien archiver.

**Ce fichier est la source mère** (`CLAUDE.md` : « objectif, hypothèses, ce
qu'on peut prouver ou non ») — celui dont dépendent le titre, le §7 que
`RESULTS.md` cite déjà verbatim (« l'argument de fermeture »), et la
question que toute campagne future doit continuer à poser dans les mêmes
termes. `DEFAUTS.md`, `COUVERTURE.md` et `RESULTS.md` ont été restaurés
avec pleine rigueur le 25 août ; celui-ci et `EVALUATION.md` ne l'avaient
pas encore été — l'écart a été signalé par USER, pas trouvé en interne.

**Méthode de restauration**, identique à celle des trois premiers
documents : le texte original (`git show d047015:docs/PLAN_PREPRINT.md`)
est restauré section par section, puis chaque affirmation est revérifiée
contre `RESULTS.md`/`DEFAUTS.md` **tels qu'ils sont aujourd'hui** — pas
seulement contre leur état à `d047015`, qui datait déjà de 39 commits et
d'une journée entière de corrections (25 août : D-39, D-50, D-98, D-100,
D-158, D-191, D-187, D-195). Ce qui a bougé est marqué **« mis à jour le
26 août »** ; ce qui n'a pas bougé est laissé tel quel, sans faux
renouvellement de date.

**Une chose que la restauration a trouvée et qu'il faut dire tout de
suite** : le texte de `d047015` lui-même n'était pas parfaitement à jour
avec `RESULTS.md` à cette même date — son §7 ne mentionnait pas D-58 (la
rétractation « ZZ is numerically dead », déjà écrite dans `RESULTS.md` à
`d047015`) ni la question rouverte de T13/T11b. Ce document a donc
toujours eu tendance à prendre du retard sur `RESULTS.md` ; le corriger ici
ne suffira pas à empêcher que ça se reproduise — seule la discipline de
CLAUDE.md (« les tenir à jour fait partie de chaque tâche ») le peut.

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

**Mis à jour le 26 août.** Ce préalable n'est toujours pas clos : il l'est
moins qu'il ne le semblait à `d047015`. Le 25 août a rouvert et refermé six
défauts de plus (D-39, D-50, D-98, D-100, D-158, D-191), en a laissé un
nouveau ouvert (D-195, une corrélation de rang QAOA/vérité négative et une
égalité QAOA=classique sans bruit, toutes deux stables sous deux tirages
indépendants — cause non élucidée), et a confirmé que deux défauts restent
bloqués sur une décision humaine, pas sur du code (D-22, D-188). Un
troisième, D-189, a été retiré une seconde fois le 26 août : la
description restaurée depuis `d047015` décrivait un défaut que `d3d7573`
avait déjà corrigé le 24, dans le même geste que la suppression des
documents — erreur de vérification (le test qui l'entourait avait
lui-même été réécrit pour vérifier le nouveau comportement, sa réussite
ne distinguait donc plus « corrigé » de « pas encore corrigé »), pas une
nouvelle correction de code. Le préalable est une contribution qui
continue, pas un socle acquis une fois pour toutes.

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

**État courant, mis à jour le 26 août** (détail et mesures → §7 et
`RESULTS.md`) :

| hypothèse | verdict | qualificatif nécessaire |
|---|---|---|
| H0a | **NON** — 0,062–0,156 contre 1,000 exigé | à `dim = 3`, la seule taille certifiée non dégénérée (D-53) ; réfutation antérieure invalide, mesurée à `dim = 2` où le problème est dégénéré (D-45/D-47) |
| H0b | **NON** — ρ(E_gap, F1) = +0,870 : mieux résoudre H dégrade la décision | mesuré à `dim = 3`, 9 solveurs (D-53) |
| H1 | **PARTIEL** | les défauts numériques comptent, mais ne suffisent pas seuls à expliquer l'écart |
| QH2a / H2b | **RÉFUTÉ** | modèle libre testé (`study/h2b_prediction/`), ne bat pas la baseline |
| H3a / H3b | **À REPRENDRE** | D-58 a retiré l'explication causale des ablations nulles de T13 et de la stagnation de T11b (« ZZ is numerically dead » était faux) ; la courbe de cône elle-même (T1b, `dim = 8`/`16`) n'est pas retirée par cette rétractation mais reste bornée par un pli dégénéré non expliqué (`harris_tearing`) |
| H4 | **CONJECTURE** | pas d'expérience dédiée qui l'isole du reste |
| H5 | **NON RÉPARÉ PAR LE LABEL DYNAMIQUE** | `ρ(d, e) ≥ 0,98` sur les quatre scénarios à l'horizon du protocole (D-188) — changer de label ne suffit pas |

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

**À ajouter ici, mis à jour le 26 août — le témoin qui manque à H0b.**
Mesuré depuis (`RESULTS.md`, « Ce que le circuit peut déplacer, et par quel
canal ») : le mixeur seul, sans hamiltonien, déplace déjà une probabilité
médiane de 0,254 sur les patches réels ; l'hamiltonien en ajoute 0,236, sur
un canal borné à `β ≤ π/(4·reps) = 0,393 rad`. **Aucune campagne du dépôt
n'utilise ce témoin** — voir Appendice A, item 4, toujours ouvert. Sans lui,
« le QAOA déplace la décision » ne distingue pas l'apport de la physique de
celui d'une simple rotation de mixeur.

Un troisième, mesuré depuis, qui appartient à la même section : **aucun des
deux couplages ne désigne un type d'instabilité.** La plaquette vaut
`(|ω| + |J|)/norme` — un vortex pur et une nappe de courant pure y rendent la
même valeur — et le couplage ZZ fait entrer un saut hydrodynamique et un saut
magnétique dans la même racine. Seul `K_xpoint` est sélectif. C'est une
propriété de la **forme choisie**, pas un défaut d'implémentation, et elle se
dit dans cette section : l'hamiltonien détecte « il se passe quelque chose »
localement, pas « quoi ».

Et un fait qui appartient au manuscrit, pas seulement au journal de bord :
sous la normalisation historique, **la moitié du terme ZZZZ était
numériquement morte sur deux scénarios canoniques sur quatre**. La plaquette
sommait `|ω|` et `|J|` sous un dénominateur commun, si bien que le signal le
plus faible disparaissait en proportion de son amplitude — rapport 179 sur
`harris_tearing`, 84 sur `kelvin_helmholtz`. Chaque scénario est dominé par un
type de structure, et le dénominateur commun transformait ce fait physique en
effacement de l'autre. Corrigé en rendant les deux magnitudes adimensionnelles
séparément avant la somme, **sans ajouter de porte**.

Ce fait a sa place dans la section : il montre qu'un coefficient peut être
*bien formé, borné, adimensionnel* et pourtant ne mesurer qu'une moitié de ce
qu'il annonce — et que seule une mesure sur les champs réels le révèle.

**Ne pas confondre avec, mis à jour le 26 août.** Ce ZZZZ-là (le poids de
plaquette, mapper v2, corrigé ci-dessus) est un défaut **distinct** de celui
que D-58 retire plus loin (« ZZ is numerically dead », la fenêtre
d'incertitude de la campagne confirmatoire, T17/T18). Les deux portent sur
« un coefficient ZZ/ZZZZ semble mort » et se ressemblent assez pour être
confondus dans une lecture rapide du manuscrit — ils ne le sont pas : l'un
est un défaut de normalisation déjà corrigé, l'autre une lecture publiée puis
rétractée. Le distinguer explicitement dans le texte du papier évite qu'un
correctif applique à tort la leçon de l'un à l'autre.

### La spécification de la tâche — H5, et ce qu'une vérité terrain dynamique en dit

Le label de la phase 2, `e_i`, est l'écart intra-patch à la moyenne : une
mesure de non-lissité, instantanée et confinée au patch. Ce n'est pas ce que
l'AMR cherche à contrôler, et l'AUC du score classique seul contre `e_i` —
**1,000** (harris), **0,997** (KH), **0,948** (rotor), 0,592 (OT) — dit que
sur trois scénarios sur quatre la tâche est quasi gratuite.

La vérité terrain **dynamique** `d_i` du protocole §1.2 existe désormais, et
sa mesure appartient au manuscrit :

- à l'horizon que le protocole impose (δt = 0,1), **ρ(d, e) ≥ 0,98 sur les
  quatre scénarios** : le label dynamique est une renumérotation monotone du
  statique, et le contrôle d'acceptation du protocole (« Spearman > 0 ») le
  laisse passer ;
- la raison est physique et se calcule : à cet horizon la perturbation
  parcourt **0,11 à 0,25** d'une largeur de patch — il n'y a rien à propager ;
- à δt = 2,0, un seul scénario décolle (`orszag_tang`, ρ = 0,596) — le seul
  dont la perturbation **amplifie** (1,38×), et le seul où la tâche statique
  n'était pas déjà résolue.

Ce que la section doit dire : **changer de label ne suffit pas à réparer la
spécification de la tâche.** Là où la tâche était triviale, elle le reste ;
elle ne cesse de l'être que là où l'écoulement est turbulent. C'est une
contrainte sur ce que ce corpus peut établir, et elle se dit avant les
résultats, pas après.

**Mis à jour le 26 août.** Ce constat est toujours l'état de l'art (D-188,
`docs/DEFAUTS.md`) : re-vérifié le 25 août, rien ne l'a fait bouger depuis
`d047015`. Ce qui reste ouvert n'est pas la mesure elle-même mais la
décision de campagne qu'elle appelle (fixer l'horizon sur `t_x`, pas sur un
compte de pas hybrides) — non tranchée.

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

**Mis à jour le 26 août — cet argument est désormais mesuré, pas seulement
anticipé.** Ce paragraphe, écrit à `d047015`, formulait un pari sur ce que la
discussion devrait dire. `D-53` (`RESULTS.md`) l'a depuis mesuré à la seule
taille certifiée non dégénérée (`dim = 3`, 18 qubits, l'optimum y est
énuméré exactement) :

| | question | verdict |
|---|---|---|
| **H0a** | l'optimiseur atteint-il l'optimum de son propre hamiltonien ? | **NON** — 0,062–0,156 contre 1,000 exigé (la règle classique dont il part atteint déjà 0,500) |
| **H0b** | mieux l'atteindre améliorerait-il la tâche ? | **NON** — ρ(E_gap, F1) = +0,870 sur 9 solveurs : mieux résoudre H **dégrade** la décision AMR |

**Ce que ceci corrige par rapport au texte publié avant D-53** :
`CLAUDE.md` portait « H0 → RÉFUTÉ » sans qualificatif, et un test T11
concluait que l'optimisation quantique n'était la source d'aucun gain — les
deux mesurés **entièrement à `dim = 2`**, où l'état fondamental exact est le
prédicteur constant « tout raffiner » sur 40 instantanés sur 40 (D-45/D-47) :
tous les solveurs atteignent l'optimum parce qu'il n'y a rien à départager.
Réfuter H0 là-dessus, c'était la réfuter sur un problème vide. La lecture
juste sépare H0a de H0b et les mesure là où le problème est certifié non
dégénéré : **l'optimiseur échoue vraiment, et le réparer ne servirait à
rien.**

Sur H3, l'énoncé **doit être réécrit** : la courbe de cône a maintenant deux
artefacts (`dim = 8` et `dim = 16`, → `RESULTS.md`) et ils vont **contre** la
formulation ci-dessous.

- Le cône **n'est pas plat** : écarts par saut +0,123 / −0,076 / +0,100 à
  `dim = 16`, contre un seuil de retrait pré-enregistré de 0,01.
- Hors pli dégénéré, à la seule taille où les quatre k sont des voisinages,
  un saut fait passer de 0,429 à 0,593 et le cône **dépasse** le classique.
- Le gain **croît** de `dim = 8` à `dim = 16` — la clause « décroît quand on
  affine » est contredite par les deux seuls points mesurés.

Ce qui borne cette lecture : `harris_tearing` rend 0,000 à tous les k, et la
conclusion change de signe selon qu'on compte ce pli ou non. Rien n'est
tranché tant qu'il n'est pas expliqué.

**Mis à jour le 26 août — cette lecture tient toujours, mais elle est
maintenant elle-même bornée par un défaut différent.** `RESULTS.md`
confirme les nombres ci-dessus tels quels (rejoués, `git ef5f0a4`) : la
courbe de cône n'est **pas** retirée par ce qui suit. Mais `D-58`, déjà
présent dans `RESULTS.md` à `d047015` sans que ce paragraphe en tienne
compte, a depuis rétracté une affirmation voisine et plus lourde :
*« the deployed pipeline discards ~99 % of [ZZ] before the QAOA ever sees
it, which is a sufficient explanation for T13's null ablations and for
T11b's near-zero variational progress »* était **faux** — la fenêtre
conserve en fait 3,3 %–12,1 % de la masse ZZ en boucle ouverte et
33,8 %–59,4 % au réglage Level-3 ; aucune classe n'est numériquement morte.
**L'explication causale des ablations nulles de T13 et de la stagnation de
T11b tombe avec la rétractation : ces deux résultats restent à expliquer.**
Ce n'est pas la courbe de cône (T1b) qui est en cause — c'est un pan
différent de H3 (T13, T11b) qui se rouvre sans explication de repli.

**Conséquence de structure** : la fermeture ne peut pas reposer sur H3. Elle
repose sur **H0b**, qui n'en dépend pas. L'ancienne formulation, conservée
ici pour mémoire et à ne plus citer telle quelle :

> Le gain apporté par l'information des voisins est réel mais petit, il
> décroît quand on affine la grille, et il ne justifie pas le coût d'un
> dispositif quantique.

Les limites qui bornent ces conclusions — un seul solveur, quatre scénarios,
8 qubits en déploiement, baseline partagée par les deux bras, non-déterminisme
du bras QAOA, chute d'ordre du solveur commune aux deux bras — sont énoncées
ici, chiffrées dans `RESULTS.md`.

**Mis à jour le 26 août.** Le protocole s'est élargi depuis (`COUVERTURE.md`) :
8 scénarios canoniques et 5 graines physiques, contre les 4 scénarios et la
graine implicite que cette liste de limites décrivait encore. « Un seul
solveur, quatre scénarios » est donc devenu **plus étroit que le protocole
actuel** sans être faux pour ce qui a été mesuré jusqu'ici — aucune campagne
confirmatoire n'a encore tourné sur les 8 scénarios (bloquée par D-22).
« Non-déterminisme du bras QAOA » doit aussi se relire à la lumière de D-191
(25 août) : la graine QAOA était par défaut fixée à `0` avant ce correctif,
ce qui masquait la dispersion réelle sur plusieurs tests et scripts hors du
protocole confirmatoire — corrigé, mais aucune mesure de dispersion publiée
avant le 25 août ne doit être citée sans vérifier de quel côté du correctif
elle a été produite.

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

**Mis à jour le 26 août — l'ordre contraint tient toujours, item par
item :**

1. **D-22 : toujours bloquant, mécanisme en place, campagne non lancée.**
   Re-vérifié le 25 août : `w_z_frac` reste borné à `[0.1, 1000.0]` (log)
   dans `train_hyperparams.py --print-space`, la borne haute que D-22
   signalait n'est toujours pas tranchée. Le JSON déployé
   (`best_hyperparams.json`) ne correspond toujours à **aucune** ligne des
   CSV Optuna du dépôt (perte 0,3213 dans la base contre 0,2215 annoncée).
   Ce qui a changé : `_save_results` écrit désormais le jeu complet de
   paramètres, le hash du commit et `sys.argv` — le mécanisme de provenance
   existe — mais il ne s'applique qu'à une campagne qui n'a pas encore
   tourné. Item 1 reste ouvert.
2. **Non lancé.** Attend l'item 1.
3. **Le compte a bougé plusieurs fois et doit être remesuré avant citation,
   pas recopié d'ici.** « 16 lignes en écart » date de `d047015`. Mesuré
   depuis, à des moments différents et sur des périmètres différents (le
   protocole s'est élargi à 8 scénarios entre-temps) :
   `study/common/aggregate_master_table.py` a rendu tour à tour
   **180 lignes, 176 OK / 4 DIFF / 0 MISSING** (après la clôture de D-58,
   périmètre à 4 scénarios — ce compte revient identique des dizaines de
   fois dans `RESULTS.md`, c'est le plus stable des trois),
   puis **268 lignes, 142 OK / 6 DIFF / 120 MISSING** (25 août, après la correction de
   D-158, `--allow-missing`, périmètre élargi à 8 scénarios — les MISSING
   sont les scénarios de la campagne confirmatoire qui n'a pas encore
   tourné, pas une régression). **Aucun de ces trois nombres n'est le bon
   à citer dans un manuscrit** : chacun est vrai pour son jour et son
   périmètre, aucun n'est stable. La commande qui les produit doit être
   rejouée au moment de rédiger, pas avant.
   ```bash
   python study/common/aggregate_master_table.py --allow-missing
   ```
4. **Toujours pas ajouté.** Confirmé le 26 août : `RESULTS.md` (« Ce que le
   circuit peut déplacer, et par quel canal ») mesure et motive ce témoin en
   détail — mixeur seul 0,254 de médiane, mixeur+H 0,490 — mais conclut
   lui-même : « Aucune campagne du dépôt n'utilise ce témoin. » Item 4 reste
   ouvert, à la fois dans l'appendice et dans `RESULTS.md`.

**Une conclusion est désormais invalidée, pas seulement en attente.** À toute
profondeur de raffinement supérieure à la première, le biais Z de
l'hamiltonien et ses couplages décrivaient deux grilles différentes : le biais
d'un patch venait du quart haut-gauche de ce patch (D-37, écart 41 % du plus
grand coefficient, présent depuis le premier commit). À `max_depth = 4`,
réglage de toutes les campagnes, trois niveaux sur quatre passaient par là.

Le bras classique n'est pas touché — il ne construit aucun hamiltonien. La
comparaison des deux bras était donc biaisée dans un sens connu.

**Mis à jour le 26 août — D-37 est corrigé, la portée de « invalidé »
change de forme sans disparaître.** `RESULTS.md` (« D-37 — le biais Z et les
couplages décrivaient deux grilles différentes ») documente la correction :
une ligne, `_process_score(local_score, depth == 0, target_dim)`, le halo
n'est plus ajouté deux fois ; `H_edges` passe de (6,6) à (4,4) à `depth > 0`,
et le pipeline s'exécute désormais à `max_depth = 2` là où il levait une
`ValueError` avant. **Tout nombre Q-HAS mesuré avant cette correction, à une
profondeur > 1, reste invalidé — mais rien mesuré après ne l'est plus pour
cette raison.** Distinguer les deux dans le manuscrit demande de savoir de
quel côté du commit `git log -S "target_dim + 2 * pad" -- src/Simulation/refinement.py`
chaque artefact cité a été produit — non fait ici, à faire avant de citer un
nombre Q-HAS à profondeur > 1.

Le reste est **en attente de confirmation** sur le code corrigé, ce qui n'est
pas la même chose qu'invalidé → `EVALUATION.md`.
