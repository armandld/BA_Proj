# Défauts

**Où ça bloque.** Uniquement ce qui n'est pas résolu : ce qui empêche
d'avancer, comment on est tombé dessus, et où on en est pour le lever.

Ce qui est corrigé n'est **pas** ici — c'est un résultat, il vit dans
`RESULTS.md` avec sa mesure et sa commande de vérification.

| | |
|---|---|
| **ouverts** — décision ou campagne requise | **5** |
| **gelés** volontairement | 2 |

Un seul demande la campagne elle-même (D-22) ; les trois autres sont des
décisions, dont deux à prendre **avant** de lancer. D-27, D-37 et D-48 sont
sortis d'ici : corrigés, mesurés, verrouillés — ils vivent dans `RESULTS.md`.

**Un point bloque de nouveau la réoptimisation côté code** : D-118
ci-dessous — le bras QAOA a cessé de classer les blocs mieux que le hasard
sur une partie de l'espace d'hyperparamètres. Bisection en cours.

---

## D-22 — les hyperparamètres déployés n'ont aucune provenance

**Où ça bloque.** Réoptimiser demande de savoir d'où l'on part. Aucun chiffre
de performance n'est attribuable à un réglage dont on ignore l'origine.

**Comment on est tombé dessus.** En auditant `train_hyperparams`, quatre
paramètres écrits comme conditionnels se sont révélés être des constantes. En
remontant aux bases Optuna pour voir lesquels avaient réellement été
échantillonnés, le JSON déployé ne correspondait à aucune.

**Ce qui est établi.**

| | échantillonné par la campagne gelée |
|---|---|
| `beta`, `beta_curl`, `beta_xpoint`, `w_z_frac` | oui, étude quantique |
| `sigma` | oui — mais **absent du JSON**, donc repli sur 0,05 |
| `threshold_amr` | seulement dans l'étude **classique** |
| `gamma_hydro`, `gamma_mag`, `kappa` | **jamais, nulle part** |

L'essai 85 que le JSON déclare a une perte de **0,3213** dans la base contre
**0,2215** annoncée, et **aucun** de ses quatre paramètres communs ne
coïncide. Le code d'entraînement, lui, est cohérent avec les bases : il fixe
`threshold_amr` à 0,14959824837662078, exactement le meilleur essai
classique. **C'est le JSON qui est orphelin.**

**Où on en est.** Ne se corrige pas par du code seul. Seule la réoptimisation
le règle. Trois paramètres n'ayant jamais été échantillonnés, ce sera pour eux
une *première*, pas une reprise.

**Périmètre : tranché — les 8.** `beta`, `w_z_frac`, `sigma`, `beta_curl`,
`beta_xpoint`, `gamma_hydro`, `gamma_mag`, `kappa`. `threshold_amr` reste gelé
au meilleur essai classique, pour que la comparaison porte sur ce que le
quantique ajoute. Nuance qui allège : `g_strain + g_rot ≡ 1`, donc `kappa` ne
contrôle **qu'un** degré de liberté.

**Le mécanisme est fermé, la campagne reste à faire.** D-35 identifiait
pourquoi le fichier déployé ne correspondait à aucune base : `_save_results`
n'écrivait que `study.best_params`, c'est-à-dire les seuls paramètres
*échantillonnés*. Le JSON écrit désormais le jeu complet, l'espace de
recherche avec ses bornes, le hash du commit et `sys.argv`. Une campagne
lancée aujourd'hui produira donc un fichier traçable — mais le fichier
**actuellement déployé** reste orphelin jusqu'à ce qu'elle tourne.

**Reste à trancher avant de lancer.** La borne haute de `w_z_frac` vaut 1000
alors que le paramètre est documenté comme une *fraction*. Elle vient de la
campagne gelée (graine 500). Conservée telle quelle pour ne pas changer la
science en même temps que le code.

```bash
python src/train_hyperparams.py --print-space          # l'espace réel
pytest tests/pipeline/test_hyperparams_provenance_break.py
pytest tests/pipeline/test_train_hyperparams_contracts.py
```

Le dernier test est le **critère d'acceptation** : `xfail` aujourd'hui, il
passera sans modification le jour où chaque valeur déployée sera traçable.

---

## D-24 — le solveur est d'ordre 1,2, la correction n'est pas applicable

**Où ça bloque.** Nulle part en toute rigueur : la chute est **commune aux
deux bras**, donc elle ne biaise pas leur comparaison. Elle comprime la plage
dans laquelle un meilleur critère pourrait se distinguer.

**Comment on est tombé dessus.** Le solveur revendique l'ordre 4 pour ses
dérivées spatiales — vérifié, 3,97 / 4,02 / 3,99. Le solveur complet converge
à 1,2. À grille fixe, en ne raffinant que le pas de temps : avec projection
1,12, sans projection 4,00. Sept ordres de grandeur d'écart.

**Ce qui a été essayé, et mesuré.**

| tentative | résultat |
|---|---|
| splitting de Strang | **erreurs identiques** — un projecteur idempotent n'a pas de demi-pas |
| projection du second membre, `step_full` | ordre **4,00**, erreur 52 000× plus petite |
| la même, `step_layered` | **1,94e−02, indépendant du pas** — plancher créé par la projection par patch |
| Van Kan, pression incrémentale | ordre inchangé, gain 2,4 % |

La projection spectrale est **non locale** : la phase 2 la calcule sur un
patch avec halo traité comme périodique. Ce n'est pas la projection globale
restreinte, et aucun découpage ne peut la reproduire. Projeter le second
membre casserait la garantie « à `max_depth`, `step_layered` ≡ `step_full` »,
qui tient aujourd'hui à **3,331e−16**.

**Où on en est.** `PROJECT_RHS = False`, raison écrite dans le code. Ce n'est
pas un bug d'implémentation : une méthode de projection à la Chorin **est**
d'ordre 1.

**Décision attendue**, par coût croissant :

1. **Laisser en l'état**, documenter comme limite. *(Recommandé.)*
2. Contrainte **locale** — volumes finis, transport contraint pour B, Poisson
   multigrille pour v. C'est ce que font FLASH, Athena++, AMReX. Ordre 2
   réaliste ; l'ordre 4 avec AMR est un sujet de recherche.
3. Formulation à pression — réécriture du cœur.

Les options 2 et 3 réécrivent `src/Simulation/` et invalident tout nombre
publié.

```bash
pytest tests/solver/test_solver_convergence.py -m slow     # ~10 min
```

---

## La borne haute de `w_z_frac` — décision avant de lancer

**Où ça bloque.** Nulle part techniquement : la campagne partirait. Mais elle
explorerait un domaine dont la moitié haute n'a peut-être aucun sens.

**Comment on est tombé dessus.** En déclarant l'espace de recherche pour
l'audit du script d'entraînement (D-35). Écrire les bornes comme des données
oblige à les lire.

**Ce qui est établi.** `w_z_frac` est documenté comme la **fraction** de la
médiane des couplages qui donne son poids au biais Z :
`alpha_z = w_z_frac × median(|C|, |K|)`. Sa borne haute vaut **1000**. Une
fraction de 1000 signifie un biais Z mille fois plus grand que les couplages,
c'est-à-dire un Hamiltonien où les termes ZZ et ZZZZ ne pèsent plus rien — le
quantique dégénère vers la décision classique.

La borne vient de la campagne gelée, dont la graine valait 500. Conservée
telle quelle pour ne pas changer la science en même temps que le code.

**Décision attendue.** Resserrer à un intervalle où le mot « fraction » a un
sens (par exemple 0,01–10, en log), ou garder 1000 et documenter que la
partie haute du domaine teste la dégénérescence vers le classique. La
seconde option est défendable — mais alors il faut le dire.

```bash
python src/train_hyperparams.py --print-space
```

---

## La persistance Colab — décision avant de lancer

**Où ça bloque.** Nulle part si la campagne tourne sur cœurs loués avec un
stockage Optuna distant. Sur Colab **non distribué**, une déconnexion perd
jusqu'à **9 essais** — sur une campagne d'une semaine, c'est du temps payé.

**Comment on est tombé dessus.** En auditant la poche « mode Colab », le
dernier axe de configuration qu'aucun test ne traversait.

**Ce qui est établi.** Trois copies vers Drive, toutes sous `if IN_COLAB`
(vérifié sur l'AST, `train_hyperparams.py:145, 457, 1164`) :

| site | quand |
|---|---|
| `ensure_dirs` | au démarrage — rapatrie les `.db` du Drive |
| `callback_save` | `IN_COLAB and not DISTRIBUTED and trial.number % 10 == 0` |
| fin de `_save_results` | une fois, à la toute fin |

La base vit sur `/content/Train_results_local`, éphémère. En mode
**distribué**, `db_path` vaut `None` et il n'y a **aucune** synchronisation
intermédiaire — sans conséquence, puisque les essais vivent alors dans le
stockage distant. C'est le chemin **non distribué** qui expose les 9 essais.

**Décision attendue.** Soit lancer en distribué avec stockage distant — ce
que le périmètre « cœurs loués » implique de toute façon — soit descendre le
pas de synchronisation à 1 si la campagne doit tourner sur Colab seul. Ne
rien changer est défendable, mais alors il faut savoir ce qu'on risque.

```bash
pytest tests/pipeline/test_v1_partial_pockets.py -k colab
```

---

## Gelés volontairement — ne pas corriger

`study/pipeline/dns_validation.py` est le code de phase 1b. Ses artefacts
sont publiés ; le corriger casse leur reproductibilité. Décision antérieure,
documentée dans le fichier lui-même.

| | défaut | version correcte |
|---|---|---|
| **D2** | `fluctuating_KE` moyenne à travers la couche de cisaillement : sur le profil de base seul elle lit **73 %** de l'énergie totale, et n'évolue que de **0,02 %** quand on allume la perturbation | `dns_extension.fluctuating_ke_fixed` |
| **D3** | `mean_sq_current` porte la même inversion d'axes | `dns_extension.mean_sq_current_fixed` |

`analyse_one` utilise désormais les versions corrigées : **le gel porte sur
les fonctions, pas sur l'analyse qui les appelle.**

Une correction a déjà été annulée ici après qu'un test a rappelé la décision.
Une déviation connue mais non écrite *là où elle vit* se fait recorriger par
erreur.

```bash
pytest tests/study/test_no_private_curl_survives.py tests/study/test_t8_dns_extension.py
```

---

## Ajouter une entrée

Un défaut n'entre ici que s'il **bloque**. Une fois corrigé, il sort d'ici et
entre dans `RESULTS.md`.

Chaque entrée porte : **où ça bloque**, **comment on est tombé dessus**, **ce
qui est établi** (chiffré), **où on en est**, et la commande qui vérifie
l'état.

Un défaut sans mesure est une suspicion. Un défaut sans commande de
vérification n'a pas sa place ici.


---

## D-118 — le bras QAOA ne classe plus, sur une partie de l'espace

**Où ça bloque.** Une campagne Optuna explore l'espace d'hyperparamètres.
Si le bras quantique n'y porte aucun signal sur une partie de cet espace,
la campagne y optimise du bruit — et elle y passera du temps de calcul
payé. À trancher **avant** de louer des cœurs.

**Comment on est tombé dessus.** Passage de recette complet après l'ajout
du 9ᵉ paramètre : `2006 passed, 3 failed` en 1 h 32. Les trois échecs sont
dans la suite QAOA.

**Ce qui est établi.**

| | |
|---|---|
| `test_hyperparameter_sweep` | **échoue**, reproductible (3 exécutions) |
| `test_noise_robustness` | **échoue**, reproductible (2 exécutions) |
| `test_the_ranking_survives_the_sampling` | **passe** à la réexécution — celui-là est bien un tirage |

L'assertion qui tombe dans le balayage n'est **pas** le plafond
`MAX_CLEAN_ADVANTAGE`, qui passe. C'est la garde de vivacité une ligne
plus bas :

```
AssertionError: correlation de rang QAOA/verite negative (-0.467) :
le bras ne classe plus rien, le plafond ci-dessus ne prouve alors rien
assert -0.467 > 0.0
  where -0.467 = min([-0.467, -0.467, 0.75, 0.95, -0.467, 0.933, ...])
```

Trois des douze combinaisons d'hyperparamètres donnent une corrélation de
rang **négative** avec la vérité terrain ; d'autres donnent +0,95. Le
second échec dit la même chose autrement :

```
Orszag-Tang: without noise the QAOA arm is expected to lose by more than
0.09 captured fraction, measured gap +0.0000
```

Un écart **exactement nul** : les deux bras sélectionnent les mêmes blocs.
C'est ce qu'on verrait si l'hamiltonien était devenu inerte dans cette
configuration et que QAOA retombait sur le biais classique.

**Ce n'est pas l'ajout de `relative_percentile`.** Le chemin par défaut est
un no-op **bit-à-bit**, épinglé par
`tests/pipeline/test_relative_percentile_is_trainable.py::test_le_defaut_est_un_NO_OP_bit_a_bit`.

**Bisection.**

| commit | verdict | durée |
|---|---|---|
| `d978539` — naissance de la garde | **passe** | 46 min |
| `403240b` — juste avant les corrections de coefficients | *en cours* | — |
| `5bdcf80` = `235dbbf~1` — après elles | **échoue** (−0,467) | 13 min |

La garde a donc passé à sa naissance : ce n'est pas un test commité rouge,
c'est une régression réelle, apparue dans un intervalle de 26 commits
touchant `src/`. Les trois en tête de cet intervalle sont les corrections
de coefficients (`e8a9455` porte de `K_xpoint`, `10180da` dimensionnement
des quatre familles, `5bdcf80` critère relatif). L'exécution à `403240b`
sépare les deux hypothèses.

La chute de durée **46 min → 13 min** à configuration égale est un indice
corroborant : les circuits construits ne sont pas seulement notés
différemment, ils sont différents.

**Ce qu'il faut noter sur ces deux tests.** Tous deux encodent d'anciens
**résultats** comme assertions — « QAOA perd d'au moins 0,09 », « QAOA
classe positivement ». Un changement de physique s'y manifeste donc en
rouge, pas en résultat. Ne pas déplacer les seuils avant de savoir ce que
la physique a fait : ce serait effacer la mesure au lieu de la lire.

**Ce que ça n'invalide pas.** `preflight_coefficients.py` passe 5/5, mais
il vérifie que les coefficients corrèlent avec le **besoin de
raffinement** — pas que le bras quantique classe mieux que le hasard.
Deux affirmations distinctes ; seule la seconde échoue.
