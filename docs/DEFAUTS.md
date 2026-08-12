# Défauts

**Où ça bloque.** Uniquement ce qui n'est pas résolu : ce qui empêche
d'avancer, comment on est tombé dessus, et où on en est pour le lever.

Ce qui est corrigé n'est **pas** ici — c'est un résultat, il vit dans
`RESULTS.md` avec sa mesure et sa commande de vérification.

| | |
|---|---|
| **ouverts** — décision ou campagne requise | **3** |
| **gelés** volontairement | 2 |

Avant la réoptimisation, les trois ouverts doivent être tranchés : deux
demandent une décision, un demande la campagne elle-même.

---

## D-22 — les hyperparamètres déployés n'ont aucune provenance

**Où ça bloque.** Réoptimiser demande de savoir d'où l'on part. Aucun chiffre
de performance n'est attribuable à un réglage dont on ignore l'origine.

**Comment on est tombé dessus.** En auditant `TrainHyperParam_v2`, quatre
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

**Où on en est.** Ne se corrige pas par du code. Seule la réoptimisation le
règle. Trois paramètres n'ayant jamais été échantillonnés, ce sera pour eux
une *première*, pas une reprise.

**Décision attendue** : périmètre — les 5 déjà échantillonnés, ou les 8.
Nuance qui allège : `g_strain + g_rot ≡ 1`, donc `kappa` ne contrôle **qu'un**
degré de liberté.

```bash
pytest tests/pipeline/test_hyperparams_provenance_break.py
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

## D-27 — la projection amputait la perturbation de quatre scénarios

**Où ça bloque.** `harris_tearing` est un des quatre scénarios **déployés**.
Réoptimiser avant de trancher, c'est ajuster contre une instabilité amorcée
au quart de son amplitude.

**Comment on est tombé dessus.** En retirant la projection de B — elle
dégradait sa divergence de huit ordres — quatre scénarios se sont révélés non
solénoïdaux à l'initialisation. Ils ne l'avaient jamais été ; la projection
les rattrapait, et personne ne l'avait vu parce qu'elle masquait la
divergence.

| scénario | div B relative | perturbation conservée |
|---|---|---|
| **`harris_tearing`** *(déployé)* | 2,801e−03 | **27,5 %** |
| `island_coalescence` | 1,400e−02 | **27,5 %** |
| `noisy_uniform` | 4,947e−01 | 55,7 % |
| `double_tearing` | 9,062e−04 | 77,3 % |

`harris_tearing` amorce son mode de déchirement par `δBy = ε cos(kx) sech²`.
**La projection en retirait 72,5 %.**

Le plan note que ce scénario « dégénère dans toutes les configurations
testées, sans explication ». Ceci en est peut-être une part — **non
affirmé** : seule l'amplitude initiale est mesurée, pas le comportement de
l'instabilité.

**Où on en est.** L'état actuel est figé par des tests, valeurs mesurées
écrites, pour qu'une dérive se voie.

**Décision attendue.** La correction propre est d'écrire la perturbation
comme le rotationnel d'une fonction de flux, `δB = ∇×(ψ ẑ)` — solénoïdale par
construction, comme cela a déjà été fait pour `magnetic_twist` et
`ghost_twisting`. Elle change les champs de quatre scénarios, donc tout
nombre publié qui les traverse.

```bash
pytest tests/solver/test_scenarios_analytic.py -k "solenoidal or amputates"
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
