# Résultats

**Un résultat, comment il a été obtenu, donc comment le réobtenir.**

Chaque entrée porte : la commande exacte, les conditions, les nombres, et le
hash du commit qui les a produits. Un résultat qu'on ne sait pas refaire
n'est pas un résultat — il n'a pas sa place ici.

| document | contenu |
|---|---|
| `PLAN_PREPRINT.md` | l'objectif et les hypothèses — la source mère |
| `DEFAUTS.md` | où ça **bloque**, uniquement |
| `COUVERTURE.md` | ce qui est **testé**, comment et pourquoi |
| **`RESULTS.md`** (ce fichier) | ce qui est **accompli**, et comment le refaire |
| `EVALUATION.md` | ce qui, ici, est **exploitable** |
| `CODE_REVIEW.md` | note de relecture |

---

## Les 64 défauts corrigés

*(61 lignes `D-N` distinctes dans les tables ci-dessous, plus 2 lignes non
numérotées — 63 corrections en tout, avant l'ajout de D-89 et D-90
ci-dessous ; ce sous-compte n'est pas revérifié ici, voir « Compte de tête
inexact » plus bas — signalé une fois, non recorrigé à chaque ajout. Le
titre annonçait **63** avant l'ajout de D-90 ; **62** avant l'ajout de
D-89 ; **60** avant l'ajout de D-73 ; **59** avant l'ajout de D-72 ; **58** avant l'ajout de D-71 ; **56** avant l'ajout de D-68 et D-70 (D-69 était alors un rapport seul,
dans `DEFAUTS.md` ; il entre ici depuis que sa table est refaite) ; **53** avant que la fusion de la base n'apporte
D-10, D-66 et D-67 ; **41** pour 42 lignes numérotées avant l'ajout de D-52,
D-54, D-55 et D-56. Le compte de tête est faux à chaque fusion — c'est
exactement le défaut de registre que la section « Compte de tête inexact »
plus bas rapporte déjà pour « Les 24 défauts corrigés ». Compté, pas estimé
— la commande est
`grep -o '^| D-[0-9]*' docs/RESULTS.md | sort -u -t- -k2 -n | wc -l`.)*

Le matériau le plus solide du travail. Chacun est mesuré avant et après,
refait par une commande, et verrouillé par un test qui échoue sur l'ancienne
version. Les mesures détaillées sont plus bas, dans les entrées de campagne.

**Conventions et opérateurs**

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-1 | rotationnel des mappeurs sous `indexing='xy'` | 0,0 → **+2,0** sur rotation solide | `pytest tests/study/test_curl_convention_gap.py` |
| D-3 | l'objectif pondère par cette vorticité fausse | 0,0 → **+2,0** | `pytest tests/mapping/test_objective_and_estimators_analytic.py` |
| D-11 | diode de choc appliquée au cisaillement | rapport **0,500 → 2,0** | `pytest tests/mapping/test_mapper_contracts.py -k flux` |
| D-17 | 3 sites hors `src/` en convention pré-D-1 | enstrophie **0 % → 0,02 %** d'écart | `pytest tests/study/test_no_private_curl_survives.py` |
| — | critère Q : déformation à moitié, partie isotrope comptée | cisaillement **+0,25 → 0** | `pytest tests/solver/test_analytic_fields.py -k q_criterion` |

**Numérique et rééchantillonnage**

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-2 | prolongation AMR centrée cellule, `mode='wrap'` | 2,49e−1 → **7,74e−6** | `pytest tests/amr/test_amr_resampling_analytic.py` |
| D-7 | projection ignore le mode de Nyquist | 0,378 → **1,1e−14** | `pytest tests/solver/test_solver_analytic.py -k idempot` |
| D-14 | réduction des champs tronque, celle du score non | 94,1 % → **100 %** | `pytest tests/mapping/test_downsampling_contracts.py` |
| D-21 | flux réduit par lissage + bilinéaire | pic **38 % → 100 %** | `pytest tests/mapping/test_padded_rescale_contracts.py` |
| D-23 | `dt` intégré ≠ `dt` écrit dans la trace DNS | référence à t≈0,077 → rejeu **exact** | `pytest tests/solver/test_precompute_dns_contracts.py` |
| D-25 | la projection **spectrale** de B abîme un champ solénoïdal en FD | div_FD B **4,63e−07 → 1,00e−14** | `pytest tests/solver/test_solver_convergence.py -k induction` |

**Encodage et décision**

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-8 | hamiltonien encode des coefficients nuls sans lever | non détecté → **lève** | `pytest tests/quantum/test_hamiltonian_contracts.py -k raises` |
| D-13 | bords gauche/haut lisent l'arête intérieure | asymétrie 1,2–7,0 % → **symétrique** | `pytest tests/quantum/test_hamiltonian_contracts.py -k halo` |
| D-15 | `postprocess` accepte des comptes bruts | marginales ~1000 → **refusé** | `pytest tests/quantum/test_vqa_chain_contracts.py -k refus` |
| D-16 | liste de patchs AMR se recouvre elle-même | **25 % → 0 %**, sans trou | `pytest tests/amr/test_amr_tiling_contracts.py` |
| D-19 | backend inconnu → contexte mort sans erreur | silence → **lève** | `pytest tests/quantum/test_runtime_contracts.py -k backend` |
| D-20 | cache d'ansatz confond deux hamiltoniens | même objet → **séparés** | `pytest tests/quantum/test_runtime_contracts.py -k ansatz` |

**Scénarios, mesure et documentation**

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-4 | doc annonce le double du facteur appliqué | ×2 → aligné | `pytest tests/mapping/test_objective_and_estimators_analytic.py` |
| D-5 | divergence notée sans pondération | 1,8 % → **0** | idem |
| D-6 | `init_magnetic_twist` ne pose aucune torsion | 6,4e−7 → **π/2 exact** | `pytest tests/solver/test_scenarios_analytic.py -k twist` |
| D-9 | ablation ψ mesure la fenêtre sur le mauvais score | « annihilation » → **ZZ domine K de 1,5 à 8,2×** | `pytest tests/study -k window` |
| D-12 | mappeur `study/` : ν, η, dx annoncés influents, sans effet | doc alignée | `pytest tests/mapping/test_mapper_contracts.py -k v2` |
| D-18 | garde de divergence à 1e100, inerte | 1e50 passait → seuil **1e8** | `pytest tests/solver/test_solver_guards_and_objective.py -k caught` |
| D-26 | `init_ghost_twisting` pose un champ **impossible** | angle **0,027 → 1,906 rad** | `pytest tests/solver/test_scenarios_analytic.py -k ghost` |
| D-27 | 4 scénarios initialisés non solénoïdaux, rabotés par la projection | perturbation **27,5 % → 100 %**, div_FD B **2,801e−03 → 1,08e−16** | `pytest tests/solver/test_scenarios_analytic.py -k "solenoidal or amputates"` |
| D-28 | `hyperparams_loader` substituait en silence les paramètres de l'**autre bras** (`quantum`↔`classical`) quand celui demandé manquait, et choisissait le premier lambda par ordre alphabétique quand plusieurs coexistaient | substitution → **lève** ; choix arbitraire → **lève sauf lambda unique** | `pytest tests/study/test_hyperparams_two_sources.py -k "refuses or implicit"` |
| — | `search_space` : 4 constantes présentées comme réglables | espace réel **5 paramètres**, pas 9 | `pytest tests/pipeline/test_train_hyperparams_contracts.py -k declares` |

**Le script d'entraînement** — audité pour la réoptimisation, un seul survit

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-29 | `SCENARIOS_ISOLATED` contenait les scénarios **complexes** ; `ot` et `rotor` comptés deux fois | **6 entrées / 4 classes → 6 / 6**, pondération 2:1 → 1:1 | `pytest tests/pipeline/test_train_hyperparams_contracts.py -k scenario` |
| D-30 | le chemin séquentiel appelle `_run_phase1(study, dns)` — la fonction prend **un** argument | `TypeError` après la phase 1 → chemin exécuté de bout en bout | `pytest tests/pipeline/test_train_hyperparams_smoke.py` |
| D-31 | `beta_michelson` proposé à Optuna, **jamais lu** par `pipeline.py` | phase 1 optimisait un paramètre sans effet → paramètre supprimé | `pytest tests/pipeline/test_train_hyperparams_contracts.py -k michelson` |
| D-32 | l'élagage ne rapportait qu'au step 0, sous `n_warmup_steps=2` | 1e9 au step 0 après 40 essais : **jamais élagué** → élagué au 3ᵉ scénario | `pytest tests/pipeline/test_train_hyperparams_contracts.py -k prun` |
| D-33 | `AdvAnomaliesEnable` absent d'Orszag-Tang, replié sur `False` | OT sans terme ZZZZ de point X → **6/6 scénarios** l'activent, la clé manquante **lève** | `pytest tests/pipeline/test_train_hyperparams_contracts.py -k argus` |
| D-34 | budget d'essais calculé une fois, par worker | 4 workers, cible 12 : **48 essais → 12** | `pytest tests/pipeline/test_train_hyperparams_contracts.py -k budget` |
| D-35 | le JSON final ne portait que les paramètres **échantillonnés** | `threshold_amr` absent → **9/9 valeurs** + hash git + argv | `pytest tests/pipeline/test_train_hyperparams_contracts.py -k redeploy` |
| D-36 | 3 des 4 sorties détaillées de `pipeline` sans provenance de `sigma` | trace présente **seulement sur les runs divergés** → sur les 4 | `pytest tests/solver/test_solver_guards_and_objective.py -k sigma` |

**Les quatre poches partielles de V1** — auditées avant la réoptimisation

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-48 | `mode="hardware"` s'exécutait sur un **simulateur** sans le signaler | `Session(AerSimulator)` **acceptée** → **lève** à la construction | `pytest tests/pipeline/test_v1_partial_pockets.py -k mode` |

*(La branche `vigil/…` de l'agent numérote en continu et va **au-delà de
D-115**. Cette note disait « la numérotation reprend à D-48 » : c'était
vrai à l'écriture et faux ensuite — D-68 et D-69 ont été attribués des
deux côtés à des défauts différents, collision rattrapée en renumérotant
les miens en **D-116 / D-117**. Avant d'attribuer un numéro, lire le
maximum réel des **deux** branches :*

```bash
git fetch origin 'refs/heads/vigil/*:refs/remotes/origin/vigil/*'
git show origin/vigil/<branche>:docs/RESULTS.md | grep -o 'D-[0-9]\+' \
  | sort -t- -k2 -n -u | tail -1
```
*)*

**Les fichiers jamais audités de V1** — le dernier chantier avant la réoptimisation

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-66 | `pipeline.main` précalculait le DNS avec `PHASE` puis passait à `pipeline()` les **défauts de la CLI** | boucle **jamais exécutée**, `combined = 0,333333` sur un run vide → configuration du scénario, garde `T_MAX > T_START` | `pytest tests/pipeline/test_full_launch_config.py -k d66` |
| D-67 | `score()` notait un run à **zéro pas** au lieu de crier | `patch_ratio = 1.0` par repli → **lève** | `pytest tests/pipeline/test_full_launch_config.py -k d67` |
| D-10 | `compare_rotor_budget` levait `TypeError` à l'étape 4/5, et ses défauts demandaient **69 Go** | n'a **jamais** tourné → tourne, garde posée avant le DNS | `pytest tests/pipeline/test_compare_rotor_budget.py` |
| D-49 | `recompute_lambda_scores.main` rattrapait **tout** dans un `except Exception` et rendait la main | échec total → **code 0** ; base absente comme répertoire non écrivable → **code 1**, cause réelle | `pytest tests/pipeline/test_recompute_lambda_scores.py -k d49` |
| D-50 | `analyze_hyperparams.main` : même piège, et le message accusait **Neon** pour un fichier local absent | **code 0** → **code 1**, cause réelle ; plus aucune mention de Neon | `pytest tests/pipeline/test_analyze_hyperparams.py -k d50` |
| D-116 | deux lanceurs de `scripts/` pointaient **entièrement** dans le vide ; `generate_figures_v1.sh` sautait ses 17 scripts et rendait **`Succeeded: 0  Failed: 0`, code 0** | campagne verte sans **aucune** figure → échec si `SUCCEEDED == 0`, chemins repointés, `ROOT_DIR` corrigé | `pytest tests/lint/test_scripts_point_somewhere.py` |
| D-117 | `RELATIVE_PERCENTILE` était une constante **en dur** sur le chemin de décision, alors que c'est elle qui ranime les termes à quatre corps à N=256 | non entraînable → 9ᵉ paramètre de `SEARCH_SPACE`, câblé de bout en bout | `pytest tests/pipeline/test_relative_percentile_is_trainable.py` |
| D-118 | **L'axe « backend » traversé du côté qui ne l'avait jamais été — 0 appel échantillonné sur la suite entière. Deux résultats, un sain et un défaut. SAIN : `aer` (échantillonné) et `state_vector` (exact) rendent la même distribution.** Les deux chemins finaux de `execute` — `Statevector.probabilities_dict()` et `sampler.run(...).get_counts()` — coïncident à la racine de N près, à paramètres fixés (indispensable : l'optimiseur est stochastique **dans les deux cas**, `EstimatorV2` tirant `default_shots` même sous `AerSimulator(method='statevector')`). **DÉFAUT, rapport seul : `--backend estimator`, troisième choix offert par le CLI de `src/pipeline.py`, ne peut produire aucune décision — à aucune taille.** Il résout `FakeFez`, modèle de machine réelle à **156 qubits** : la transpilation étale le circuit logique sur les 156, `measure_all()` crée 156 bits classiques, et le simulateur dépasse `max_memory_mb`. La levée qui remonte — `ValueError: could not broadcast input array from shape (0,20) into shape (1024,20)`, 20 octets étant les 156 bits empaquetés — ne nomme **ni** le backend, **ni** la mémoire, **ni** la transpilation. L'`Estimator` de la boucle d'optimisation, lui, répond (`evs = 0.1328`) : la campagne tourne, et meurt à l'étape qui produit la décision. **Ce qui interdit de corriger au passage** : la panne visible en cache une seconde qui ne le serait pas. `call_vqa_shell` passe `qc.num_qubits` du circuit **transpilé** à `postprocess` ; rendre seulement la mémoire ferait voir à son garde de longueur 156 == 156, et rendrait **156 marginales indexées par qubit PHYSIQUE** là où l'appelant en attend 8, indexées par qubit logique. Trancher entre « retirer le choix » et « câbler le placement de bout en bout » est une **décision**, pas une correction de chemin | **Le côté sain**, patch `dim = 2` déployé, paramètres fixés `x = (0.10, -0.05, 0.7, 1.3)`, écart max sur les 8 marginales contre le statevector exact : **1 024 tirs → 0,0205** (bruit attendu ~0,0312) ; **8 192 → 0,0102** (~0,0110) ; **65 536 → 0,0037** (~0,0039). Cinq tirages de référence à 8 192 tirs : 0,00538 / 0,00561 / 0,00591 / 0,00686 / 0,01122 — tous sous la moitié des 4 σ binomiaux (0,0221), qui est le seuil retenu, **non calibré sur la mesure du jour**. **Le défaut**, mesuré à `766d289` : `VQARuntime(backend_name='estimator').transpile(qc)` rend **156 qubits** pour **2** qubits logiques comme pour **4** — indépendant de la taille du problème ; `measure_all()` → **156 bits classiques** ; `sampler.run` → `ValueError` à 100 %, sur 2 comme sur 8 qubits logiques. Placement mesuré à 4 qubits logiques : `final_index_layout() = [136, 142, 141, 143]` — la marginale du qubit logique 0 se lirait à l'indice **136**. **Aucun nombre publié n'en dépend et ne PEUT en dépendre** : ce chemin n'a jamais pu rendre une valeur. `python study/common/aggregate_master_table.py` : **180 lignes, OK=164, DIFF=16, MISSING=0**, inchangé. **Aucun code de `src/` ni de `study/` n'est touché** | `pytest tests/quantum/test_estimator_backend_axis.py -q` (**6 cas : 5 passés, 1 `xfail(strict=True)`** qui porte la dette et fera rougir la suite le jour où la panne sera levée — ce qui obligera à regarder le placement dans le même geste). Mutation vérifiée : les quatre tests de mesure basculés sur `backend_name='aer'` (sans placement matériel) donnent **4 failed**, et le `xfail` **XPASS(strict)** — les deux sens |
| D-119 | **L'axe « optimiseur » traversé du côté qui ne l'avait jamais été — COBYLA 317 appels contre 1 chacun pour Powell, L-BFGS-B et Nelder-Mead. Rapport seul : deux faits mesurés, et une non-conclusion assumée.** **(a) `K_opt` n'achète pas le même budget selon la méthode.** `execute` passe `options={'maxiter': K_opt}` aux trois : scipy traduit `maxiter` en nombre d'**évaluations** pour COBYLA, en nombre d'**itérations** pour Powell et L-BFGS-B — chacune valant plusieurs évaluations (recherche linéaire, gradient par différences finies). **(b) Le `K_opt` gelé a été réglé sous COBYLA ; le CLI déploie L-BFGS-B.** `train_hyperparams.create_argus` — l'objectif que la campagne Optuna a optimisé, celle dont `results/hyperparams/best_hyperparams.json` est l'artefact **gelé** — code `method="COBYLA"` en dur. `src/pipeline.py` offre `--method` avec pour défaut **`L-BFGS-B`**, et **aucun des lanceurs du dépôt ne passe `--method`** : tout run de `pipeline.py` prend ce défaut. Les neuf hyperparamètres de `SEARCH_SPACE` ont donc été sélectionnés sous un optimiseur variationnel et sont déployés par défaut sous un autre, qui ne consomme pas le même budget au même `K_opt`. **Présent depuis le premier commit** (`cf93ba3`) : le défaut vaut `L-BFGS-B` dès l'origine. **Pourquoi rien n'est corrigé** : aligner le défaut sur COBYLA, ou réentraîner sous L-BFGS-B, change la science dans les deux sens. `VIGIL.md` : mesurer, documenter, ne pas corriger, demander | **(a)**, comptage des appels à l'estimateur, `K_opt = 20`, **6 tirages** chacun : COBYLA **20 20 20 20 20 20** — exactement `K_opt`, 6 fois sur 6 ; L-BFGS-B **50 60 85 95 115 90** (×2,5 à ×5,8) ; Powell **187 377 328 176 251 265** (×8,8 à ×18,9). Les trois intervalles sont **disjoints** — la dispersion ne l'explique pas. Confirmé à `K_opt = 40` : COBYLA 28–35, L-BFGS-B 55–220, Powell 123–357. **(b)**, lu par appel (`create_argus(SCENARIO_OT).method`) et par AST (`default` de `--method`), pas par recherche de chaîne : **`COBYLA` contre `L-BFGS-B`** ; **0 lanceur sur 8** pose `--method`. **Ce que la mesure NE dit PAS, et c'est la moitié du résultat** : à la dispersion du bras QAOA, elle ne tranche pas si les deux optimiseurs rendent une décision différente. Patch `dim = 2` déployé, `K_opt = 30`, 6 tirages par méthode : écart des **moyennes** COBYLA vs L-BFGS-B **0,0867** au maximum, contre une dispersion **intra**-méthode (max − min par qubit) de **0,200** (COBYLA) et **0,240** (L-BFGS-B). L'effet cherché est trois fois plus petit que le bruit d'exécution de chacune des deux références : `VIGIL.md` veut alors qu'on le dise et qu'on ne conclue pas. Trancher demande une campagne, pas une passe. **Aucun nombre publié ne bouge** ; **aucun code de `src/` ni de `study/` n'est touché** | `pytest tests/quantum/test_optimiser_axis.py -q` (**6 cas : 5 passés, 1 `xfail(strict=True)`** qui porte la dette (b) et fera rougir la suite le jour où entraînement et déploiement nommeront le même optimiseur). Mutation vérifiée : `create_argus` basculé sur `L-BFGS-B` donne **XPASS(strict) → failed**, restauré **xfailed**. Un test épingle aussi la **non-conclusion** — il rougira le jour où le bras QAOA sera assez reproductible pour que la comparaison décide |
| D-120 | **Le garde de la borne du mixeur ne mordait pas sur COBYLA — le bras qui porte 317 des 320 appels de la suite.** `test_the_three_supported_optimizers_keep_the_bound` est le test qui a rattrapé la perte de borne de Powell ; son docstring dit « le test qui mord ». Il part à **froid** : `beta` initialisé à zéro et `rhobeg = 0.05` maintiennent le simplexe de COBYLA minuscule, qui n'a alors aucune raison d'aller chercher la borne — et n'y va pas. Retirer **entièrement** `common['constraints']` le laisse donc **VERT**. C'est la règle « choisir le champ d'essai qui SÉPARE » : sur un départ à froid, « contraintes présentes » et « contraintes absentes » rendent la même chose, donc le test ne mesure rien de ce qu'il annonce sur ce bras. **Ce n'est pas le code qui est en cause — les contraintes sont justes et load-bearing** : c'est le champ d'essai. Trouvé en vérifiant par mutation un site du sondage `.read()` de `COUVERTURE.md` (`test_beta_is_bounded_and_the_bound_is_the_documented_one`, qui `assert "beta_max = np.pi / (4 * reps)" in src`) : la mutation qui devait montrer un faux vert a montré, en plus, que le garde comportemental censé couvrir le trou ne couvrait que **2 méthodes sur 3** | **L'entrée qui sépare est un WARM START hors borne**, et ce n'est pas un cas de laboratoire : `warm_start_params` est filé d'un pas au suivant par la campagne, et `execute` ne le valide contre aucune borne. **avant → après**, même circuit, `warm_start_params` à `beta = 1,0` (la valeur que le commentaire de `execute` décrit comme catastrophique : ~60° de rotation, `P(|1⟩) → 0,25`, tout raffinement supprimé), borne `pi/(4·reps) = 0,3927` : **avec** contraintes **0,3927 / 0,3927 / 0,3927** — 0/3 hors borne ; **sans** contraintes **1,0817 / 0,9470 / 1,0369** — **3/3 hors borne**. Le rappel à la borne est donc bien l'œuvre des contraintes. Départ à froid, contraintes retirées, pour comparaison : `max|beta|` **0,018 à 0,317**, **0/5** hors borne à `K_opt = 40` **comme** à `K_opt = 200` — le budget n'y change rien, c'est `rhobeg` qui borne, pas la contrainte. **Le site `.read()` qui a mené ici est classé « surestimé »**, pas défaut : mutation **A** (borne réelle cassée, texte source intact) → il reste **VERT**, mais le garde comportemental voit le trou pour Powell et L-BFGS-B ; mutation **B** (réécriture ÉQUIVALENTE `np.pi / 4 / reps`, valeur bit à bit identique) → il passe **VERT → ROUGE** sans qu'aucun défaut n'existe. C'est le troisième cas de faux rouge de cette forme dans ce dépôt. **Aucun code de `src/` n'est touché, aucun nombre publié ne bouge** : la correction est entièrement dans les tests | `pytest tests/quantum/test_runtime_contracts.py -q -k bound` (9 cas). Mutation vérifiée : `common['constraints']` supprimé → ancien garde **VERT** (le défaut), nouveau garde `test_the_bound_holds_even_when_the_warm_start_starts_outside_it[COBYLA]` **ROUGE** ; restauré, **VERT** |
| D-121 | **Le repli de `sigma` sur 0,05 n'était vérifié par AUCUN test comportemental — seulement par un test qui lit le SOURCE.** `test_the_pipeline_falls_back_to_a_hard_coded_sigma` (`test_hyperparams_provenance_break.py`) fait `assert "_defaults.get('sigma', 0.05)" in src` ; `test_the_pipeline_shouts_when_sigma_is_missing` (`test_train_hyperparams_smoke.py`) vérifie `sigma_source == "default"` et l'avertissement, mais jamais `result["sigma"]`. Sondage `.read()` de `COUVERTURE.md`, site non listé dans les 24/27 — trouvé en remontant `_defaults.get('sigma', 0.05)` (`pipeline.py:394`) depuis le test qui le cite | **Mutation A** (`pipeline.py:394`, `0.05` → `0.07`, reste du fichier intact) : suite des deux fichiers rejouée — **1 failed, 21 passed, 1 xfailed** ; le SEUL test qui rougit est celui qui lit le source, aucun test comportemental ne voit la valeur numérique changer. **Confirmé, pas surestimé** : contrairement à D-120, rien d'autre ne couvre ce comportement. Corrigé en ajoutant `test_the_defaulted_sigma_value_is_exactly_0_05`, qui appelle `pipeline()` et lit `result["sigma"]` directement — rejoué sous la même mutation A : **failed** (`0.07 != 0.05 ± 5e-7`) ; restauré à `0.05` : **passed**. Le test source-text n'est pas retiré (il reste correct, juste fragile face à une réécriture équivalente — 4ᵉ cas de cette forme dans le dépôt) | `pytest tests/pipeline/test_train_hyperparams_smoke.py -q -k sigma` (2 cas). Mutation vérifiée dans les deux sens : `pipeline.py:394` à `0.07` → `test_the_defaulted_sigma_value_is_exactly_0_05` **ROUGE** ; restauré → **VERT** |
| D-122 | **`--zero-psi` sans `--with-psi` était une ablation VIDE, et un artefact publié porte ce nom.** `prepare_qaoa_inputs` pose `psi_h = psi_v = 0` **exactement** tant que `with_psi` est faux (« no temporal flux in study », `qaoa_inputs.py:294`) ; le bloc d'ablation de `solver_panel` réécrivait alors des zéros par des zéros, bit à bit. Le balayage sortait quand même avec le code **0** et le suffixe `_zeropsi` au nom de son artefact : une ablation qui n'ablate rien, indiscernable d'une vraie — c'est le piège « un balayage vide doit crier » de `CLAUDE.md`, et la question 1 de `VIGIL.md` (à quoi sert cette branche ? que se passerait-il si on la supprimait ? **rien**). Trouvé en remontant le suffixe `_zeropsi` de `_output_path` jusqu'à son unique artefact. **Corrigé** : `solver_panel` refuse désormais `zero_psi=True` sans `with_psi` (`SystemExit`, même forme que la garde `--with-psi` à l'instantané 0 juste au-dessus). **Le fait publié, lui, n'est PAS corrigé et se signale** — voir la ligne D-53 de `DEFAUTS.md` | **(a) le no-op, à la source.** `prepare_qaoa_inputs`, `N = 48`, `dim = 3`, `Re = 400`, champ Taylor-Green + nappe magnétique, instantané précédent advecté de 0,05 : **sans** `with_psi` → `max|psi_h| = 0` et `max|psi_v| = 0`, **exactement** (pas « à 1e-16 ») ; **avec** `with_psi` → **1,47109** et **0,0578957**. Le drapeau ne mord donc que sous `with_psi`. **(b) le fait publié.** `results/h0_optimiser_equivalence_N96_dim3_zeropsi_scalekopt.npz` : `cli_args` = `zero_psi: true`, **aucune clé `with_psi`** — c'est le cas vide. Comparé à son jumeau `..._N96_dim3_scalekopt.npz`, **même `git_hash` `53a8dfc`, même `seed 0`**, `cli_args` différant du seul `zero_psi` : `hit`, `match`, `n_diff`, `f1` **identiques sur 54/54 lignes** ; les **5 solveurs déterministes** (`exhaustive`, `sa`, `sa_warm`, `greedy`, `classical_init`) **bit à bit identiques, 30/30 lignes, `max\|ΔE\| = 0`** ; seuls les **4 bras QAOA** diffèrent (15/54 lignes, `max\|ΔE\| = 61,282`), à la dispersion connue du bras. Les deux artefacts sont **la même campagne exécutée deux fois**, pas deux conditions. **Aucun nombre publié ne bouge** : `aggregate_master_table` lit `N=256, dim=2`, aucune de ses 180 lignes ne voit ces fichiers | `pytest tests/study/test_zero_psi_ablation_is_not_empty.py -q` (**6 cas**). Mutation vérifiée dans les deux sens : garde retirée → `test_the_panel_refuses_an_empty_psi_ablation` **ROUGE** (`DID NOT RAISE SystemExit`), restaurée → **VERT** ; et sur le test qui épingle l'ancien comportement, `psi_h` forcé à `1e-3` hors `with_psi` → `test_without_with_psi_the_ablation_rewrites_zeros_by_zeros` **ROUGE**, restauré → **VERT** |

| D-123 | **La décision de reprise de T22 était gardée par une recherche de chaîne : un seul `and` changé en `or` la casse sans qu'aucun test ne bouge.** `h4_unseen_conditions.main` refuse un point de sauvegarde écrit sous une autre configuration — sinon des tirages incomparables entrent dans une même moyenne publiée. Son unique garde, `test_resume_reuses_only_matching_configurations`, cherchait quatre chaînes (`prev.get("fold") == args.fold`, `prev.get("mode") == args.mode`, `"repeats"`, `"matched_reference"`) dans le TEXTE du fichier. Même famille que D-114 et D-115, mais sur un invariant de **provenance** : le faux vert n'y produit pas un plantage, il produit une moyenne d'apparence normale sur des données qui ne vont pas ensemble. Trouvé en vérifiant par mutation un site du sondage `.read()` de `COUVERTURE.md` (`test_t24_leak_free.py:181`). **Corrigé** : la décision devient `checkpoint_is_reusable(prev, args)`, fonction pure de niveau module, appelable par un test | **Mutation A** (`prev.get("cli_args", {}).get("repeats") == args.repeats` précédé de `or` au lieu de `and`, les quatre chaînes intactes) : `pytest tests/study/test_t24_leak_free.py -q` → **26 passed**, aucun test ne bouge. `fold` et `mode` égaux suffisent alors à accepter un point écrit sous un autre `--repeats`. **Mutation B**, contrôle inverse (les gardes `repeats`/`matched_reference` retirées de l'expression, donc les chaînes aussi) : **1 failed** — le test d'origine ne voit la panne que si elle emporte le texte avec elle. **Aucun nombre publié ne bouge** : aucun artefact `t22_unseen_*` du dépôt n'est `partial`, la reprise n'a donc jamais eu à trancher sur les données publiées | `pytest tests/study/test_t24_source_text_guards_are_behavioural.py -q` (**6 cas** pour D-123 : 1 contrôle positif, 4 champs paramétrés, 1 sur l'AST). Mutation vérifiée dans les deux sens : `and` → `or` → **5 ROUGES**, restauré → **VERT** |
| D-124 | **Le garde « les tirages repris ne sont jamais tus » était satisfait par un COMMENTAIRE.** `test_resume_is_recorded_never_silent` fait `assert "resumed_from_checkpoint" in src and "n_runs_resumed" in src`. Les deux noms apparaissent **deux fois** dans `h4_unseen_conditions.py` : aux lignes 362-363, dans le commentaire qui les explique, et aux lignes 392-393, où ils sont réellement écrits dans l'artefact. Supprimer les écritures laisse le commentaire — donc laisse le test vert. Un artefact aurait alors tu que ses tirages viennent d'un autre processus : c'est la « valeur sans provenance » de `VIGIL.md`, sur la provenance elle-même. **Corrigé par un test qui interroge l'AST** (les affectations `out[...] = ...` réellement présentes), pas le texte — la forme que D-114 a déjà imposée ici. Aucun code n'est touché : le code était juste, c'est le garde qui ne gardait rien | **Mutation** : les deux lignes `out["resumed_from_checkpoint"] = …` et `out["n_runs_resumed"] = …` remplacées par un commentaire, les deux noms survivant dans celui des lignes 362-363. `pytest tests/study/test_t24_leak_free.py -q` → **26 passed**. Le fichier reste syntaxiquement valide et l'artefact perd ses deux champs de provenance sans qu'aucun test ne le voie. **Aucun nombre publié ne bouge** : le code n'est pas modifié | `pytest tests/study/test_t24_source_text_guards_are_behavioural.py -q -k assigned` (**1 cas**). Mutation vérifiée dans les deux sens : écritures supprimées → **ROUGE**, restaurées → **VERT** |
| D-125 | **Des deux consommateurs qui doivent écarter un artefact PARTIEL, un seul était couvert fonctionnellement.** `test_partial_checkpoints_are_never_analysed` vérifie par recherche de chaîne (`'== "partial"' in cs`) que `closed_loop_leak_free_summary.py` **et** `h4_transfer_summary.py` filtrent les points de sauvegarde. Le premier a, juste en dessous, son garde comportemental (`test_a_partial_record_is_rejected_by_the_summary`) ; **le second n'en avait aucun**. Un artefact `partial` y serait lu comme complet, et ses moyennes publiées porteraient sur une exécution interrompue au milieu d'une condition. Aucun code n'est touché : le filtre est juste, c'est sa couverture qui manquait | **Mutation ciblée** — `h4_transfer_summary.py:78` seul (`if d.get("status") == "partial":` → `if False and …`, la chaîne cherchée intacte) : `pytest` sur les **trois** fichiers de test qui importent ce module (`test_h4_transfer_summary_total_abort_arm_name.py`, `test_t22c_transfer.py`, `test_t24_leak_free.py`) → **38 passed**. La même mutation appliquée aux **deux** consommateurs à la fois donne **1 failed** — et l'échec vient du jumeau `closed_loop`, jamais de celui-ci. **Aucun nombre publié ne bouge** : le code n'est pas modifié, et aucun artefact `t22_unseen_*` du dépôt n'est `partial` | `pytest tests/study/test_t24_source_text_guards_are_behavioural.py -q -k partial_artifact\ or\ complete_artifact` (**2 cas** : l'artefact partiel refusé, et le contrôle positif que le filtre n'écarte pas tout). L'artefact d'essai porte `n_runs = 4` — à moins de 2, le repli « underpowered » l'écarterait pour une autre raison et le test ne mesurerait pas ce qu'il annonce. Mutation vérifiée dans les deux sens : filtre désactivé → **ROUGE**, restauré → **VERT** |
| D-126 | **Un garde qui lit le SOURCE, ancré sur une phrase NON UNIQUE — il a rougi sur un changement voulu, et il ne mesurait pas ce qu'il annonçait.** `test_the_empty_sweep_message_names_what_is_missing` (`test_h0_panel_guards.py:67`, site du sondage `.read()` de `COUVERTURE.md`) prenait la **première** occurrence de « balayage vide » dans `h0_optimiser_equivalence.py` et exigeait `args.scenario`, `args.N`, `args.dim`, `dns_`, `patches_` dans les **600 caractères suivants**. Deux défauts en un. **(1)** L'ancre n'est pas unique : le commentaire de D-122, ajouté plus haut dans le même fichier, contient la règle « un balayage vide doit crier » — la fenêtre est tombée dessus et le test a rougi **sans qu'aucun défaut n'existe**. C'est le **5ᵉ** faux rouge de cette forme dans ce dépôt, et le premier constaté en direct plutôt que reconstitué. **(2)** Il ne mesurait pas ce qu'il annonce : `args.scenario` est le nom du **code**, pas ce que l'utilisateur lit ; un message qui n'interpolerait rien le contiendrait tout autant. **Corrigé** : les deux tests du bloc partagent une fixture de module qui lance le panel en sous-processus, et l'assertion porte sur les **valeurs** de la sortie réelle (`orszag_tang`, `N=32`, `dim=2`, `dns_`, `patches_`) | **avant** : `pytest tests/study/test_h0_panel_guards.py -q` sur le tip de D-122 → **1 failed, 11 passed**, `AssertionError: le message d'erreur du balayage vide ne mentionne pas args.scenario` — l'assertion citait mon commentaire, pas le message. Aucun code de production n'avait changé de comportement. **après**, même commit, garde réécrit : **12 passed**. **Le nouveau garde mord** — mutation : le message réduit à `"balayage vide."`, la phrase-ancre laissée en place → **1 failed** (`assert 'dns_' in …`), restauré → **12 passed**. Le sous-processus coûte ~5 s au total, partagé entre les deux tests au lieu d'être payé deux fois. **Aucun nombre publié ne bouge** ; aucun code de `src/` n'est touché | `pytest tests/study/test_h0_panel_guards.py -q` (**12 cas**) |
| D-127 | **La durabilité du point de sauvegarde de H0 n'était gardée que par une recherche de chaîne.** `test_each_line_is_flushed_to_disk` (`test_h0_panel_resume.py:199`, l'un des 23 candidats du sondage `.read()` de `COUVERTURE.md`) découpe `src[i:i+900]` à partir de `def _append_checkpoint` et y cherche `fh.flush()` et `os.fsync(fh.fileno())`. Or ce que la fonction promet — son propre docstring : « une mort brutale ne doit pas tronquer une ligne déjà annoncée comme écrite » — est un **comportement**, que le texte ne fait qu'indiquer. C'est la forme que `VIGIL.md` interdit, 5ᵉ instance dans ce dépôt (D-114, D-115, D-120, D-121). **Aucun code de `study/` n'est touché : `flush`+`fsync` sont justes et load-bearing — c'est leur couverture qui manquait** | **Mutation dans les deux sens, `_append_checkpoint` seul, reste du fichier intact.** **A′** — `flush`/`fsync` déplacés sous `if os.environ.get("H0_DURABLE")` (faux), les deux chaînes cherchées **intactes dans la fenêtre de 900 caractères** : la durabilité a entièrement disparu et le fichier reste **21 passed** — le garde ne voit pas ce qu'il prétend garder. **B** — réécriture **ÉQUIVALENTE** `_fd = fh.fileno()` puis `os.fsync(_fd)`, comportement bit à bit identique : **VERT → ROUGE**, faux rouge sur un changement voulu. Le site est donc **confirmé, pas surestimé** : `grep -rn "fsync\|_append_checkpoint" tests/` ne rend que ce fichier, rien d'autre ne couvre ce comportement. **Après**, garde comportemental `test_the_line_is_on_disk_before_fsync_returns` : il espionne `os.fsync`, relève l'**inode** du descripteur reçu et lit le fichier depuis un **second descripteur indépendant** pendant l'appel — l'entrée qui SÉPARE, puisqu'elle distingue « `flush` puis `fsync` » de « `fsync` seul » et de « rien ». Rejoué sous les quatre mutations : **A′** ancien VERT / nouveau **ROUGE** ; **B** ancien **ROUGE** / nouveau VERT ; **C** (`flush()` retiré seul) **ROUGE** ; **D** (`fsync` retiré seul) **ROUGE** ; restauré **22 passed**. **Aucun nombre publié ne bouge** : la correction est entièrement dans les tests | `pytest tests/study/test_h0_panel_resume.py -q` (**22 cas**, dont le nouveau). Mutation vérifiée dans les deux sens, cf. colonne précédente |
| D-128 | **Le balayage vide, dans le fichier qui existe pour détecter les balayages vides — et son assertion ne mesurait pas non plus ce qu'elle annonçait.** `test_aggregations_over_runs_filter_completed` (`test_silent_failure_sweep.py`) garde le défaut qui a fait publier à T16 une moyenne de **0,3328** pour `rotor` là où les tirages valides donnaient **0,1473**. Il sélectionnait ses sites par `np.mean(` **et** `_runs` sur la **même ligne** du source. Aucune des deux agrégations réelles du dépôt n'est écrite ainsi : elles tiennent sur deux lignes, la liste étant liée un peu plus haut. **Deux défauts en un, et le second n'a été vu qu'en corrigeant le premier.** **(1) Sélection vide** : 0 ligne sur 65 scripts. **(2) Assertion par voisinage** : même une fois les sites retrouvés, l'assertion cherchait la chaîne `completed` dans les 12 lignes précédentes — or ce voisinage contient `"n_completed": len(runs)`, un champ de **compte rendu**, pas un filtre. Le garde restait vert avec le vrai filtre retiré. C'est la même forme que D-126 : une fenêtre de proximité satisfaite par une mention sans rapport. **Corrigé** : sélection par AST (structure, pas mise en forme), assertion qui **remonte du nom agrégé à sa liaison** et y exige le filtre, plus un compteur `test_the_aggregation_sweep_is_not_empty` qui fait rougir la suite si la sélection retombe à zéro | **avant** : sites sélectionnés **0** sur **65** scripts, contre **2** agrégations réelles trouvées par AST (`closed_loop_leak_free_summary.py:126` et `:127`). **Le code de production, lui, est juste** — `runs = [r for r in q.get(f"{cond}_runs", []) if r["completed"]]` filtre bien : le défaut est entièrement dans le garde, et **aucun nombre publié ne bouge**. **Mutation** (filtre `completed` retiré de la liaison, le champ `n_completed` laissé deux lignes plus bas) : **ancien garde → 199 passed, 61 skipped**, le trou est invisible ; **garde intermédiaire, sélection AST mais assertion par voisinage → 200 passed**, toujours invisible ; **garde final → 1 failed**. Restauré : **200 passed, 61 skipped**. Le nombre **2** est écrit dans le compteur pour qu'une dérive se voie | `pytest tests/study/test_silent_failure_sweep.py -q` (**200 passés, 61 ignorés** — les `skip` sont les scripts sans option à choix ou sans chemin de sortie littéral, section (a)/(c), inchangés) |
| D-128 (suite) | **Le garde ci-dessus, corrigé, restait aveugle à 18 des 20 agrégations réelles.** `_aggregations_sur_tirages` ne voyait un site que si `"runs"`/`"_runs"` apparaissait **littéralement dans les tokens de l'appel `np.<réducteur>(...)` lui-même**. Or le motif dominant du dépôt passe par un **paramètre de fonction** : `summarise(q_ok)` filtre `q_ok` chez l'**appelant**, puis `summarise` calcule `np.mean(v)` sur un nom `v` qui ne contient ni `"runs"` ni `"_runs"` — invisible au premier garde, quelle que soit sa mise en forme. Trouvé en vérifiant, avant d'écrire quoi que ce soit, que les 2 sites mesurés par la première passe étaient bien la totalité du motif réel — ils ne l'étaient pas. **Corrigé** : la sélection trace désormais la **provenance** d'un nom (comprehension → conteneur `*_runs`, avec le statut `'filtered'` uniquement si la garde `completed` n'est pas niée — `if not r["completed"]` garde les tirages avortés, l'inverse de l'intention, et ne doit pas compter), et cette trace franchit **une indirection d'appel de fonction locale** pour couvrir le cas `summarise()` | **avant (mesuré, pas supposé)** : 2 sites confirmés filtrés sur 65 scripts, tous deux dans `closed_loop_leak_free_summary.py`. **Après** : **20** sites confirmés filtrés sur **5** scripts — `aggregate_master_table.py`: 2, `closed_loop_leak_free_summary.py`: 2 (les 2 d'origine, retrouvés à l'identique), `closed_loop_run_variance.py`: 5 via `summarise()`, `h4_physics_robustness.py`: 5, `h4_unseen_conditions.py`: 6. **0 échec** : le code de production est juste sur les 20. **Mutation, sur les deux sites les plus représentatifs des 18 retrouvés** : filtre retiré dans `aggregate_master_table.rows_t20` → **2 échecs détectés** (`np.mean(q)`, `np.std(q, ddof=1)`) ; filtres retirés dans `closed_loop_run_variance.main` (`q_ok`/`c_ok`, en amont de `summarise()`) → **5 échecs détectés**. Restaurés : **0 échec** dans les deux cas. **Aucun code de `study/` n'est touché, aucun nombre publié ne bouge** : la correction est entièrement dans le test | `pytest tests/study/test_silent_failure_sweep.py -q` (**200 passés, 61 ignorés**, compteur `test_the_aggregation_sweep_is_not_empty` remesuré à **20**) |
| D-129 | **Le garde qui empêche la variante T28 d'écraser l'artefact par scénario ne lisait que le SOURCE — et il n'existait aucune couverture du chemin qui écrit.** `test_the_variant_never_overwrites_the_original` (`test_t28_t29_labels_and_ci.py:103`, l'un des 23 candidats du sondage `.read()` de `COUVERTURE.md`) cherche `SUFFIX = "_globalthr"` et `replace(".npz", f"{SUFFIX}.npz")` dans `labels_global_threshold.py`. Son objet est un **comportement** — l'artefact `patches_<scenario>_...npz` doit survivre à la relabellisation, c'est le défaut **D9** — que le texte ne fait qu'indiquer. Le seul appel réel à `relabel()` de la suite (`test_the_relabeller_refuses_a_degenerate_threshold`, ajouté par D-115) **lève avant d'écrire** : le chemin d'écriture n'était traversé par aucun test. 6ᵉ instance de cette forme dans le dépôt (D-114, D-115, D-120, D-121, D-127). **Aucun code de `study/` n'est touché : le suffixe est juste — c'est sa couverture qui manquait** | **Mutation dans les deux sens, `relabel()` seul, reste du fichier intact.** **A′** — `out = out.replace(SUFFIX, "")` ajouté **après** le calcul du chemin, les **deux** chaînes cherchées intactes : la variante écrase l'original et `test_t28_t29_labels_and_ci.py` + `test_patches_classical_score_provenance.py` restent **72 passed**. Faux vert, sur le défaut même que le garde existe pour empêcher. **B** — réécriture **ÉQUIVALENTE** `os.path.basename(path)[:-len(".npz")] + SUFFIX + ".npz"`, chemin vérifié bit à bit identique (`patches_orszag_tang_Re400_N256_dim4_globalthr.npz` des deux côtés) : **ROUGE**. Faux rouge sur un changement voulu. **Après**, garde comportemental `test_the_variant_is_written_beside_the_original_not_over_it` : il appelle `relabel()` sur quatre artefacts synthétiques **non dégénérés** — l'entrée qui SÉPARE, puisque le seul appel existant lève avant d'écrire — et compare le **SHA-256** des originaux avant et après, puis vérifie que `collect()` écarte bien les sorties. Rejoué : **A′** ancien VERT / nouveau **ROUGE** ; **B** ancien **ROUGE** / nouveau VERT ; restauré **14 passed**. **Aucun nombre publié ne bouge** : la correction est entièrement dans les tests | `pytest tests/study/test_t28_t29_labels_and_ci.py -q` (**14 cas**, dont le nouveau). Mutation vérifiée dans les deux sens, cf. colonne précédente |
| D-130 | **`--fixed-curl` du panel H0 : le garde qui vérifie que le drapeau ATTEINT le mappeur ne lisait que le SOURCE — et le défaut qu'il annonce empêcher passe sous lui sans le réveiller.** `test_the_panel_exposes_the_flag_and_suffixes_its_artefact` (`test_fixed_curl_variant.py:183`, l'un des 23 candidats du sondage `.read()` de `COUVERTURE.md`) fait `assert "fixed_curl=args.fixed_curl" in src` sous le message *« le drapeau doit atteindre `solver_panel`, sinon il ne fait que renommer le fichier de sortie »*. C'est un **comportement** — la traversée CLI → `solver_panel` → `prepare_qaoa_inputs` → `PhysicalMapperV2(fixed_curl=…)` — que le texte ne fait qu'indiquer. Le seul autre test du fichier qui touche la CLI (`test_the_panel_accepts_the_flag_end_to_end`) n'exécute que `--help`. 7ᵉ instance de cette forme dans le dépôt (D-114, D-115, D-120, D-121, D-127, D-129). **Aucun code de `study/` n'est touché : le câblage est juste — c'est sa couverture qui manquait** | **Mutation dans les deux sens, un seul site à la fois, reste du fichier intact.** **A′** — `solver_panel` passe `fixed_curl=False` à `prepare_qaoa_inputs` (`h0_optimiser_equivalence.py:360`), les **trois** chaînes cherchées intactes (`"--fixed-curl", action="store_true"` :702, `+ ("_fixedcurl" if args.fixed_curl else "")` :249, `fixed_curl=args.fixed_curl` :793) : le drapeau ne fait plus **que** renommer l'artefact — mot pour mot le défaut annoncé — et le fichier reste **7 passed**. **B** — réécriture **ÉQUIVALENTE** `fixed_curl=bool(args.fixed_curl)` au site CLI, comportement identique : **ROUGE**. **Après**, garde comportemental `test_the_flag_reaches_the_mapper_not_just_the_file_name` : il espionne `prepare_qaoa_inputs` — la première fonction de `solver_panel` à recevoir le drapeau, donc l'entrée qui SÉPARE « le drapeau traverse » de « le drapeau suffixe » — et vérifie les **deux** valeurs, `True` et `False`, une sentinelle interrompant le panel avant la campagne QAOA. Rejoué : **A′** ancien VERT / nouveau **ROUGE** ; **B** ancien **ROUGE** / nouveau VERT ; restauré **8 passed**. **Aucun nombre publié ne bouge** : la correction est entièrement dans les tests | `pytest tests/study/test_fixed_curl_variant.py -q` (**8 cas**, dont le nouveau). Mutation vérifiée dans les deux sens, cf. colonne précédente |
| D-131 | **La « définition unique » du seuil classique de la campagne en boucle fermée n'était gardée que par le SOURCE.** `test_the_closed_loop_covers_every_key_the_pipeline_reads` (`test_hyperparams_two_sources.py:226`, l'un des 23 candidats du sondage `.read()` de `COUVERTURE.md`) cherche `FROZEN_DEFAULTS = dict(gamma_hydro=2.0, gamma_mag=0.5, kappa=10.0)` et `best.setdefault("threshold_amr", T.CLASSICAL_BEST_THRESHOLD)` dans `closed_loop_campaign.py`. La seconde garde un **comportement**, que son propre commentaire énonce : *« une constante recopiée finit toujours par diverger de son original »*. Et la première lit dans le texte un dict que le test **importe trois lignes plus haut**. 8ᵉ instance de cette forme dans le dépôt (D-114, D-115, D-120, D-121, D-127, D-129, D-130). **Aucun code de `study/` n'est touché : le câblage est juste — c'est sa couverture qui manquait** | **Mutation dans les deux sens, un site à la fois, reste du fichier intact.** **A′** — `setdefault` réécrit sur le littéral `0.14959824837662078`, la ligne cherchée laissée **en code mort sous `if False:`** (la forme exacte de D-115) : le seuil redevient une copie et le fichier reste **12 passed**. Faux vert, sur la divergence même que le commentaire annonce empêcher. **B** — `FROZEN_DEFAULTS` réécrit en littéral **ÉQUIVALENT** `{"gamma_hydro": 2.0, …}`, dict identique : **ROUGE**. Faux rouge sur un changement voulu. **Après**, garde comportemental `test_the_frozen_defaults_and_the_threshold_come_from_one_definition` : `train_params_excluding` reçoit `T` en **argument**, donc l'entrée qui SÉPARE est un faux `T` dont `CLASSICAL_BEST_THRESHOLD` vaut une **sentinelle** (`0,4242424242424242`) absente du dépôt — un seuil recopié rendrait `0,1495982…`. Le test vérifie en plus le `setdefault` lui-même, avec un essai qui règle `threshold_amr` : il ne doit **pas** être écrasé. Rejoué : **A′** ancien VERT / nouveau **ROUGE** ; **B** ancien **ROUGE** / nouveau VERT ; **C** (`setdefault` → affectation, le réglage de l'essai jeté en silence) **les deux ROUGES** ; restauré **13 passed**. **Aucun nombre publié ne bouge** : la correction est entièrement dans les tests | `pytest tests/study/test_hyperparams_two_sources.py -q` (**13 cas**, dont le nouveau). Mutation vérifiée dans les deux sens, cf. colonne précédente |
| D-132 | **Le garde qui empêche le retour de D-96 (et de son ancêtre D-37) se contourne en retirant DEUX ESPACES — et il rougit sur le commentaire que `VIGIL.md` exige d'écrire.** `test_source_no_longer_asks_for_the_halo_twice` (`test_v1_legacy_instrumented_bfs_score_grid.py:100`, l'un des 23 candidats du sondage `.read()` de `COUVERTURE.md`) fait `assert "target_dim + 2 * pad" not in src` sur `fig15`/`fig16`/`fig17`. C'est la forme « absence d'un jeton », jusqu'ici classée **légitime** dans ce fichier — elle ne l'est que quand aucune écriture équivalente n'existe, et ici il en existe une triviale. Les deux tests comportementaux voisins (`test_the_old_call_produced_a_wrong_shaped_core`, `..._the_canonical_two_by_two_grid`) mesurent `_process_score` par un **helper du test**, pas les appels réels de `fig15/16/17` : rien d'autre ne relie ces trois fichiers à la correction. 9ᵉ instance de la famille (D-114, D-115, D-120, D-121, D-127, D-129, D-130, D-131). **Aucun code de `figures/` n'est touché : les trois appels sont justes — c'est leur garde qui l'était moins** | **Mutation dans les trois directions, `fig15_decision_flip_analysis.py:177` seul.** **A′** — défaut D-96 **réintroduit à l'identique**, écrit `target_dim+2*pad` (deux espaces en moins, **AST identique**, bug identique : cœur `(4, 4)` là où la boucle `for i in range(target_dim)` n'en lit que le quart haut-gauche) : la chaîne cherchée n'apparaît pas, le fichier reste **7 passed**. **A″** — même défaut écrit **avec** les espaces : l'ancien garde mord, **2 failed** — il ne tient donc que par l'orthographe. **B** — un **commentaire correct** au-dessus de l'appel juste, disant *« ne PAS redemander target_dim + 2 * pad ici »* : **ROUGE**. Le garde punit exactement ce que `VIGIL.md` impose — *« toute décision de ne pas corriger s'écrit dans le fichier concerné »*. **Après**, garde structurel `test_no_call_asks_process_score_for_more_than_the_target_dim` : il parcourt l'**AST** des trois fichiers et exige que le 3ᵉ argument de chaque appel à `_process_score` soit le **nom** `target_dim`, jamais une expression — insensible aux espaces, aux retours à la ligne et aux commentaires ; il crie aussi si le balayage ne trouve **aucun** appel. Rejoué : **A′** ancien VERT / nouveau **ROUGE** ; **A″** les deux **ROUGES** ; **B** ancien **ROUGE** / nouveau VERT ; restauré **10 passed**. **Aucun nombre publié ne bouge** : aucune figure `fig15/16/17` n'est commitée, et la correction est entièrement dans les tests | `pytest tests/study/test_v1_legacy_instrumented_bfs_score_grid.py -q` (**10 cas**, dont 3 nouveaux, un par fichier). Mutation vérifiée dans les trois directions, cf. colonne précédente |
| D-133 | **Le garde anti-D15 est défait par un alias d'import.** `test_long_tasks_no_longer_stamp_at_save_time` (`test_provenance.py:108`, l'un des 23 candidats du sondage `.read()` de `COUVERTURE.md`) fait `assert "git_commit_hash()" not in src` sur les deux tâches longues, sous la promesse *« plus aucun appel à `git_commit_hash()` »*. Forme « absence d'un jeton » — celle que D-132 vient de rouvrir : elle n'est légitime que si aucune écriture équivalente n'existe, et un alias en est une. 10ᵉ instance de la famille | **Mutation A′ — D-15 REINTRODUIT**, pas simulé : dans `h4_unseen_conditions.py`, `from provenance import git_commit_hash as _gch` puis `out["git_hash"] = _gch()` **après** `provenance.finish(prov)` (ligne 565), donc le hash rendu redevient celui de la **sauvegarde** — le défaut D15 exactement, celui qui a mal estampillé les artefacts T20 de `ot` et `kh`. `grep -c "git_commit_hash()"` → **0**, fichier **7 passed**. **A″** — même réintroduction écrite en clair (`provenance.git_commit_hash()`) : l'ancien garde mord, **2 failed** — il ne tient donc que par le nom. **Après**, garde structurel `test_long_tasks_never_call_the_stamp_under_any_name` : il **résout d'abord les alias** en parcourant les `ImportFrom` de `provenance`, puis cherche l'appel dans l'AST — couvre `git_commit_hash()`, `provenance.git_commit_hash()` et tout alias. Rejoué : **A′** ancien VERT / nouveau **ROUGE** ; **A″** les deux **ROUGES** ; restauré **9 passed**. **Limite déclarée dans le test** plutôt que découverte plus tard : un `subprocess` appelant `git rev-parse HEAD` à la main réintroduirait D15 sans passer par le helper et échapperait aux **deux** gardes ; la promesse écrite porte sur le helper, et l'invariant large reste couvert par `test_start_then_finish_keeps_the_starting_hash`. **Aucun code de `study/` n'est touché, aucun nombre publié ne bouge** | `pytest tests/study/test_provenance.py -q` (**9 cas**, dont 2 nouveaux, un par tâche longue). Mutation vérifiée dans les deux sens, cf. colonne précédente |
| D-134 | **Les quatre chaînes qui gardent `--mode leak-free` laissent passer la réintroduction de la fuite D13 — sous un artefact qui porte le nom `leak-free`.** `test_no_leak_mode_is_gone_and_leak_free_is_wired` (`test_t22c_transfer.py:120`, l'un des 23 candidats du sondage `.read()` de `COUVERTURE.md`) cherche `"no-leak"`, `"leak-free"`, `hp_q["threshold_amr"] = leak_free_thr` et `rec["classical_params"]["threshold_amr"]` dans `h4_unseen_conditions.py`, sous le message *« leak-free does not actually change the QAOA threshold »*. C'est un **comportement** — le bras QAOA doit tourner au seuil réglé sur les seules classes d'entraînement — que le texte ne fait qu'indiquer. C'est le fichier même dont l'en-tête raconte que `--mode no-leak` avait déjà été *« une OPTION ACCEPTÉE ET NON IMPLÉMENTÉE »* : le garde écrit contre ce piège ne le voit pas revenir. 11ᵉ instance de la famille | **Mutation A′** — une ligne `hp_q["threshold_amr"] = float(rec["qaoa_params"]["threshold_amr"])` ajoutée **juste après**, les **quatre** chaînes intactes : le bras QAOA repart au seuil fuyant `0,14959824837662078` — la fuite D13 exactement — et `test_t22c_transfer.py` + `test_t24_leak_free.py` restent **35 passed**. **C** — le mode lisant `qaoa_params` au lieu de `classical_params` : **2 failed**, les deux gardes mordent. **Après**, garde comportemental `test_leak_free_really_replaces_the_leaked_threshold`, avec pour champ d'essai un `rec` dont le seuil classique (`0,31337`) **diffère** du seuil fuyant — l'entrée qui SÉPARE, puisque des seuils coïncidents rendraient appliquer et ne pas appliquer indiscernables ; il vérifie aussi que le garde d'entrée refuse un bras qui n'était pas au seuil fuyant. **Seul changement de code `study/` : le bloc de `main()` extrait tel quel en `apply_leak_free_threshold`**, pour être appelable sans rejouer les heures de DNS d'un fold — même geste que `interpretation_message` (D-46), `reading_message` (D-50), `decision_rule_lines` (D-52) et `floor_ratios` (D-89). Corps **inchangé**, déplacé. Rejoué : **A′** ancien VERT / nouveau **ROUGE** ; **C** les deux **ROUGES** ; restauré **36 passed**. **Aucun nombre publié ne bouge** : les 4 artefacts `t22_unseen_*` du dépôt sont en mode `unseen-ic`, que ce chemin ne traverse pas | `pytest tests/study/test_t22c_transfer.py -q` (**10 cas**, dont le nouveau). Mutation vérifiée dans les deux sens, cf. colonne précédente |
| D-136 | **Le décompte de direction de T25 n'exclut les conditions vacues que par une chaîne — et cette chaîne existe DEUX fois, donc casser un chemin sur deux ne réveille rien.** `test_t25_verifies_the_condition_actually_moved_the_physics` (`test_t24_leak_free.py:224`, l'un des 23 candidats du sondage `.read()` de `COUVERTURE.md`) fait `assert 'not c.get("condition_is_weak")' in src` sur `h4_physics_robustness.py`, sous le message *« t25 compte des conditions vacues dans son décompte de direction »*. C'est un **comportement** — une condition dont la trajectoire ne bouge pas (< 1 %) ne peut ni confirmer ni infirmer la direction, elle sort du décompte — que le texte ne fait qu'indiquer. La chaîne cherchée apparaît **deux fois** dans le fichier : ligne 284 (chemin `--recompute`) et ligne 531 (chemin principal, celui qui écrit l'artefact publié). Une seule suffit à la satisfaire. Le tri par `grep -c` de `COUVERTURE.md` ne l'aurait pas désigné : ses deux occurrences sont **du code exécutable**, pas un commentaire — c'est la forme de D-125 (un consommateur sur deux non couvert), pas celle de D-124. 12ᵉ instance de la famille (D-114, D-115, D-120, D-121, D-127, D-129, D-130, D-131, D-132, D-133, D-134). **Aucun code de `study/` n'est touché : les deux filtres sont justes — c'est leur couverture qui manquait** | **Mutation dans les deux sens, un site à la fois, reste du fichier intact.** **A′** — le chemin **principal** (`:531`) cesse d'exclure les conditions vacues (`and not c.get("condition_is_weak")` retiré du seul `dec` de fin de `main`), la chaîne cherchée **intacte** à `:284` : `n_decidable` et `n_qhas_worse` recomptent les conditions dont la trajectoire n'a pas bougé — une direction « tenue sur k/n » gonflée par des conditions qui ne testent rien — et `test_t24_leak_free.py` reste **26 passed**. Les deux seuls autres fichiers qui touchent ce module (`test_silent_failure_sweep.py`, `test_t24_leak_free.py`) restent **226 passed, 61 skipped** : rien dans la suite ne le voit. **B** — réécriture **ÉQUIVALENTE** `c.get("condition_is_weak", False)` aux deux sites, comportement bit à bit identique (`None` et `False` sont tous deux falsy) : **ROUGE**. Faux rouge sur un changement voulu, 5ᵉ de cette forme ici. **Après**, deux gardes. (1) comportemental — `--recompute` est le seul des deux chemins exécutable sans rejouer des heures de DNS : sur un artefact synthétique portant **une condition vacue décidable** et **une condition franche**, l'entrée qui SÉPARE (des conditions toutes franches rendraient exclure et ne pas exclure indiscernables), il exige `n_decidable == 1`. (2) structurel — l'**AST** localise les deux liaisons de `dec` par la **structure** (comprehensions filtrant `qhas_worse`), jamais par une distance, et exige que **chacune** porte le filtre `condition_is_weak` ; il crie si le balayage en trouve moins de **2**. C'est le remède que `COUVERTURE.md` prescrit pour la fenêtre de proximité, appliqué ici au comptage d'occurrences. Rejoué : **A′** ancien VERT / nouveau **ROUGE** ; **B** ancien **ROUGE** / nouveau VERT ; restauré **28 passed**. **Aucun nombre publié ne bouge** : la correction est entièrement dans les tests | `pytest tests/study/test_t24_leak_free.py -q` (**28 cas**, dont 2 nouveaux). Mutation vérifiée dans les deux sens, cf. colonne précédente |
| D-137 | **Le refus d'extrapoler la frontière classique n'est gardé que par une chaîne — et sous cette chaîne, un budget SOUS la plage balayée rend un nombre fini, plausible et faux.** `test_t25_never_extrapolates_its_frontier` (`test_t24_leak_free.py:233`, l'un des 23 candidats du sondage `.read()` de `COUVERTURE.md`) fait `assert "xs[0] <= qp <= xs[-1]" in src` sous le message *« t25 pourrait extrapoler hors de la frontiere balayee »*. C'est un **comportement** que le texte ne fait qu'indiquer. La docstring de `frontier_verdict` désigne elle-même ce piège comme *« exactement le motif traqué par cette campagne »* — un nombre d'apparence normale qui ne mesure rien, *« et il aurait été publié comme un ratio »*. Le test voisin qui appelle **vraiment** `frontier_verdict` (`test_t25_refuses_a_non_monotone_bracketing_interval`) couvre les trois autres refus — non monotone, trop raide, bissection non convergée — et **aucun** cas hors plage : ses quatre budgets sont tous à l'intérieur. 13ᵉ instance de la famille. **Aucun code de `study/` n'est touché : la garde est juste — c'est sa couverture qui manquait** | **Mutation dans les deux sens, un site à la fois, reste du fichier intact.** **A′** — la garde neutralisée en **code mort** (`if False and not (xs[0] <= qp <= xs[-1]):`, la forme de D-115/D-131), la chaîne cherchée **intacte** : sur la frontière `[(0,35 ; 0,40) ; (0,45 ; 0,20)]`, un budget de **0,20** rend `0,700` et un budget de **0,05** rend `1,000` — finis, positifs, dans l'intervalle d'un `phys_score`, **sans refus ni plantage** — là où la version juste rend `budget outside the swept range`. Et le biais a un **sens** : l'erreur classique inventée **croît** quand le budget décroît, donc `ratio_vs_frontier = qe / ref` **diminue** et le bras Q-HAS paraît meilleur. `test_t24_leak_free.py` reste **28 passed**, les deux seuls fichiers qui touchent ce module **228 passed, 61 skipped**. **B** — réécriture **ÉQUIVALENTE** `min(xs) <= qp <= max(xs)` (les deux appelants trient `f_ok` par `patch_ratio`, cf. `:263` et `:488`) : **ROUGE**. Faux rouge sur un changement voulu, 6ᵉ de cette forme ici. **Après**, garde comportemental `test_t25_refuses_a_budget_outside_the_swept_frontier` : il appelle `frontier_verdict` et **écrit les deux nombres mesurés** (`0,700` et `1,000`) pour que la dérive se voie. **L'entrée qui SÉPARE est le budget SOUS la plage** : au-dessus, la garde retirée fait lever `StopIteration` — un plantage, qui se voit ; en dessous, elle rend le nombre plausible, qui ne se voit pas. Le test couvre les deux et dit lequel est le silencieux. Rejoué : **A′** ancien VERT / nouveau **ROUGE** ; **B** ancien **ROUGE** / nouveau VERT ; restauré **28 passed** (un test source-text retiré, un garde comportemental ajouté). **Aucun nombre publié ne bouge** : la correction est entièrement dans les tests | `pytest tests/study/test_t24_leak_free.py -q` (**28 cas**, dont le nouveau). Mutation vérifiée dans les deux sens, cf. colonne précédente |
| D-138 | **Le garde qui tient la mention de la déviation D-100 près de son calcul est une FENÊTRE DE PROXIMITÉ — et il rougit quand on écrit la mesure que `VIGIL.md` exige d'écrire.** `test_la_deviation_reste_ecrite_dans_le_fichier_concerne` (`test_fig11_uncertainty_weight.py:82`) fait `i_mention = src.index("D-100")`, `i_calcul = src.index("uncertainty = np.exp(")` puis `assert 0 < i_calcul - i_mention < 1500`. C'est la forme que `COUVERTURE.md` nomme depuis D-126 : une zone d'examen délimitée par une **distance en caractères** autour d'une ancre textuelle, et non par la structure. L'objet du test est **légitimement** le texte — `VIGIL.md` impose que toute décision de ne pas corriger s'écrive dans le fichier concerné, **avec sa mesure** — mais son délimiteur ne l'est pas. C'est le côté **faux rouge** de la forme, celui que D-126 a fait mordre en direct. 2ᵉ des 14 sites de cette forme à rendre un défaut (12 restaient à sonder, 2 étaient sains) | **Mesure, sans rien casser du tout — c'est le point : aucun défaut n'est nécessaire.** État du jour : les deux ancres sont **uniques** dans le fichier et la distance vaut **1171** caractères contre une borne de **1500**, soit **329 caractères de marge**, environ 4 lignes de commentaire. Ajout de lignes de mesure au bloc de déviation lui-même, à 77 caractères la ligne : **+4 lignes → 1479, VERT** ; **+5 lignes → 1556, ROUGE** ; **+6 lignes → 1633, ROUGE**. Écrire cinq lignes de plus de la mesure que la déviation doit porter fait donc échouer la suite **sans qu'aucun défaut n'existe** — le test punit exactement ce que la méthode impose. Le côté faux vert est fermé aujourd'hui par l'unicité des deux ancres, mais rien ne la garantit : `src.index` prend la **première** occurrence. **Après**, garde structurel `test_la_mention_de_la_deviation_accompagne_son_calcul` : il localise par l'**AST** l'affectation `uncertainty = np.exp(...)`, remonte à sa **fonction englobante**, et exige qu'un **commentaire** (par `tokenize`, donc jamais confondu avec du code) contenant `D-100` vive dans cette même fonction, **au-dessus** de l'affectation. Aucune distance. Il crie si le balayage ne trouve pas l'affectation, ou pas de fonction englobante. Rejoué : **+5 et +6 lignes** ancien **ROUGE** / nouveau **VERT** ; mention **déplacée hors de la fonction** ancien VERT (distance conservée par un remplissage) / nouveau **ROUGE** ; mention **supprimée** les deux **ROUGES** ; restauré **4 passed**. **Aucun code de `figures/` n'est touché, aucun nombre publié ne bouge** : la correction est entièrement dans les tests | `pytest tests/study/test_fig11_uncertainty_weight.py -q` (**4 cas**, dont le nouveau). Mutation vérifiée dans les deux sens, cf. colonne précédente |
| D-139 | **Le garde qui vérifie que la branche de repli écrit les clés qu'on relit est une fenêtre de 300 caractères — et le nom d'une clé retirée survit dedans, dans un commentaire.** `test_the_uncertified_fallback_uses_the_same_keys` (`test_h0_panel_resume.py:98`) prend `i = src.index('r.update(dict(agree_spin=float("nan")')` puis cherche `exact_match` et `n_diff_patch` dans `src[i:i+300]`, sous le message *« `--no-exact` lèvera un KeyError après le calcul »*. Sa première moitié est **bien comportementale** (elle appelle `decision_agreement` et compare ses clés) — c'est la seconde, celle qui garde la **branche de repli**, qui ne l'est pas. La fenêtre de 300 caractères déborde sur les lignes de commentaire qui suivent l'appel. Combinaison de deux formes déjà nommées : la **fenêtre de proximité** (D-126/D-128) et **le nom qui survit dans un commentaire** (D-124). 3ᵉ des 14 sites de fenêtre à rendre un défaut | **Mutation A′** — `exact_match=float("nan")` **retiré du dict de repli**, le nom laissé dans un commentaire ajouté juste en dessous, donc toujours dans la fenêtre : `test_h0_panel_resume.py` reste **22 passed**. La conséquence est celle que le test annonce, et elle est directe : `:799` fait `match=r["exact_match"]` en **indexation nue**, donc `--no-exact` lève `KeyError` après avoir payé tout le calcul. Marge du jour, mesurée : les deux clés sont aux offsets **65** et **117** de la fenêtre de **300** — la marge est large, ce n'est pas elle qui sauve le garde, et rien ne l'empêche de ne plus l'être. **Après**, garde structurel `test_the_fallback_binds_exactly_the_keys_the_panel_reads` : il localise par l'**AST** l'appel `r.update(dict(...))` et relève ses **mots-clés** — donc jamais un nom vu dans un commentaire — puis exige l'**égalité** avec l'ensemble des clés que `decision_agreement` produit réellement à l'exécution. C'est l'invariant que le titre de la section énonce (« les clés du repli doivent être celles que l'on lit »), noué aux **deux chemins** au lieu d'être approché par une distance : ajouter une clé d'un côté sans l'autre rougit désormais dans les deux sens. Il crie si le balayage ne trouve aucun appel. Rejoué : **A′** ancien VERT / nouveau **ROUGE** ; **C** (une clé ajoutée au seul repli) ancien VERT / nouveau **ROUGE** ; restauré **23 passed**. **Aucun code de `study/` n'est touché, aucun nombre publié ne bouge** : la correction est entièrement dans les tests | `pytest tests/study/test_h0_panel_resume.py -q` (**23 cas**, dont le nouveau). Mutation vérifiée dans les deux sens, cf. colonne précédente |

**Le chemin d'entraînement** — audité parce qu'il produit le nombre que la campagne minimise

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-37 | à **toute profondeur > 0**, le biais Z et les couplages décrivaient des grilles différentes | `H_edges` (6,6) contre `C_edges` (4,4) ; écart **0,05814 sur une échelle de 0,14107**, soit 41 % | `pytest tests/amr/test_patch_encoding_shapes.py` |
| D-38 | trois gardes de `execute` qui ne tenaient que sur le chemin habituellement testé | marginales **0,5535 → 0,700** ; Powell borné ; tirs MPS restaurés | `pytest tests/quantum/test_runtime_contracts.py -k "bound or null_hamiltonian or optimizer"` |

**Le diagnostic Phase 6** — `pipeline_verification.py` compare le classement
par énergie hamiltonienne v1 aux patchs durs. Sur les artefacts réels
`results/coefficients_{harris_tearing,kelvin_helmholtz}_Re400_N256_dim4.npz`,
l'énergie v1 est **identiquement nulle sur toute la simulation** — aucun
saut de cellule (`v_jump`, `B_jump`) n'y franchit jamais le seuil critique
de `PhysicalMapper` (`RE_CRIT`/`RM_CRIT` = 1.0), vérifié en rejouant
`compute_patch_coefficients` sur les 20 snapshots des deux scénarios,
contre `mhd_rotor`/`orszag_tang` où le seuil est franchi (E non nul,
100 %/70 % des cellules actives). Une énergie constante rend AUC/F1 égaux
à leur valeur de hasard (0,5/0,0) **par construction du calcul**, pas par
une mesure de non-discrimination — indiscernable à la lecture d'un vrai
résultat au hasard.

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-40 | la moyenne agrégée de `pipeline_verification.py` incluait des lignes à énergie constante (E≡0) comme si c'étaient de vraies mesures au hasard, tirant le verdict vers le bas en silence | sur les 4 scénarios canoniques (Re=400, N=256, dim=4) : AUC(E) **0,687 → 0,874**, F1(E) **0,364 → 0,729** — le verdict F1(E) vs F1(classique) passe de WARN (0,364 < 0,603) à PASS (0,729 > 0,654) une fois les 2 lignes dégénérées exclues et annotées plutôt que moyennées | `pytest tests/study/test_pipeline_verification_degenerate.py` |
| D-44 | `sanity_check.run_qaoa` décidait la convergence du QAOA sur `np.std(marg) > 0.01` — la **dispersion** des marginales — alors que le critère annoncé par son propre commentaire est « marginals should not all be 0.5 », c'est-à-dire la **distance à 0,5** | sur les défauts du script (Re=400, N=32, dim=2, 4 scénarios) le verdict s'inversait aux deux extrêmes : harris_tearing v1, marginales 0,976–0,980, `max|m−0,5| = 0,480` — le run le plus tranché des huit — était **« NOT converged (flat) »** (std 0,0019) ; kelvin_helmholtz v1 idem (0,239 contre std 0,0014). À l'inverse orszag_tang v1, déclaré convergé (std 0,0585), porte une marginale à **0,0169** de 0,5. Le bras QAOA n'étant pas déterministe, mesure refaite : 2ᵉ exécution identique en substance — harris_tearing v1 `0,473` contre std `0,0081`, kelvin_helmholtz v1 `0,249` contre std `0,0009`, l'inversion tient aux deux fois. Après : **4/4 → 8/8 convergés**, tolérance 0,01 inchangée mais portée sur la bonne grandeur | `pytest tests/study/test_sanity_check_convergence_criterion.py` |
| D-45 | phase 4 déclarait `promising` — la porte qui, d'après sa propre docstring, décide seule quels patchs passent en QAOA — sur une comparaison entre deux prédicteurs **constants** : l'état fondamental exact vaut « raffiner partout » et la ligne de base classique aussi, donc les deux F1 sont égaux par construction et la porte ne peut rien rejeter | dim=2 (seule dimension exécutable : dim=4/8 → 32/128 qubits > le plafond de 20), Re=400, N=256, 4 scénarios canoniques, 40 snapshots : décision exacte tout-à-1 **40/40**, décision classique tout-à-1 **40/40**, `exact_refine != classical_refine` **0/40**, F1 égaux **40/40**, jamais supérieurs. `promising` **40/40 True** avec `>=`, **0/40** avec le `>` que le commentaire annonce — la porte porte **0 bit** dans les deux sens. Après : `degenerate_decision` levé **40/40** et `promising_informative` **0/40**, la dégénérescence annotée au lieu d'être lue comme un succès | `pytest tests/study/test_exact_diag_degenerate_gate.py` |
| D-46 | `label_percentile_sensitivity.py` imprime « ROBUST … fails for ANY reasonable hard-patch definition » dès que `max(deltas) < 0.05`, alors que le docstring du module définit le seuil de robustesse comme « gap turns positive » (`delta < 0`) — une marge de 0,05 non documentée que rien dans l'historique git ne rattache à un choix explicite | sur l'artefact réel (`--dim 4 --N 256`, 4 scénarios canoniques, seed 0) : `max(delta) = -0,154` à p=75, sous les deux seuils — le verdict « ROBUST » ne change pas ici. Cas synthétique construit pour séparer les deux hypothèses (deltas −0,10 / −0,20 / **+0,03** / −0,15, un percentile où le site bat le classique) : ancien seuil imprime **« ROBUST … fails for ANY »** malgré le +0,03 positif ; nouveau seuil imprime **« SENSITIVE … F1_site beats classical by +0,030 »**, la lecture conforme au docstring | `pytest tests/study/test_percentile_sensitivity_interpretation.py` |
| D-43 | `find_optimal_threshold` balayait ses seuils avec `flat_e >= thr` : sur une énergie **constante**, les 100 percentiles sont égaux et chaque candidat prédit **tous** les patchs durs — le F1 rendu était celui du classifieur tout-positif, `2p/(p+1)`, présenté comme un pouvoir de séparation | mêmes artefacts : harris_tearing **0,400 → NaN**, kelvin_helmholtz **0,376 → NaN** (`E.ptp = 0` sur les deux) ; mhd_rotor **0,950** et orszag_tang **0,519** inchangés. 0,400 se lisait comme un signal réel un peu plus faible que le 0,519 authentique d'OT — et identique à tous les Re, donc comme un **seuil parfaitement stable**, la conclusion même que `threshold_stability_report` existe pour produire | `pytest tests/study/test_find_optimal_threshold_degenerate.py` |
| D-49 | `study/common/aggregate_v3.py` compare chaque chiffre du protocole v3 à un `ref` codé en dur en le présentant comme le « single source of truth » validé contre « reference (RESULTS.md) », alors qu'aucune des 44 valeurs distinctes du fichier ne provient de `docs/RESULTS.md` — elles sont copiées verbatim depuis `docs/archive/RESULTS_V3.md` / `docs/archive/v3_master_table_ca7f815.md`, que `docs/archive/README.md` déclare lui-même obsolètes (« obtenus sur du code dont on sait maintenant qu'il calculait autre chose que ce qu'il annonçait ») | mesuré (recherche exacte, bornée par chiffre, des 44 valeurs distinctes de `aggregate_v3.py` dans `docs/RESULTS.md`) : **41/44 absentes** ; les 3 « présentes » (0,000, 0,008, 0,25) sont des coïncidences sur des valeurs génériques sans lien avec la même métrique. Les 9 générateurs que `scripts/run_study_v3.sh` invoque pour regénérer ces chiffres (t1…t9, phase11_upper_bound, phase11b_loso) n'existent plus dans ce dépôt sous aucun chemin. Après : docstring et en-tête markdown généré (`v3_master_table.md`) disent explicitement « archived, pre-audit V3 baseline — voir D-49 », plus `study/common/aggregate_v3.py` (chemin réel) au lieu de `study/v3/aggregate_v3.py` (inexistant) ; `scripts/run_study_v3.sh` porte le même avertissement. Reste ouvert, non bloquant, pour USER : archiver ce module et son test, ou reconstruire les 9 générateurs — aucune des deux n'est faite ici | `pytest tests/study/test_t10_aggregate.py -k markdown_header` |

Pas une correction du calcul de l'énergie hamiltonienne elle-même (`src/`
n'est pas touché : `RE_CRIT`/`RM_CRIT` restent ceux du contrat en vigueur,
et la question de savoir si ce seuil convient aux scénarios lisses
tearing/KH est une décision physique, pas un défaut de code — voir
`DEFAUTS.md`). Uniquement le diagnostic `study/` qui ne doit plus confondre
« aucun signal calculé » avec « chance mesurée ».

**Le critère d'acceptation de H0** — `h0_optimiser_equivalence.py` est le
script dont le verdict porte le « RÉFUTÉ » de `h0_selection` dans
`CLAUDE.md`. Trouvé par la question 5 de `VIGIL.md` — *un test emprunte-t-il
cette configuration ?* — et non en lisant une fonction : la configuration
non traversée est celle où l'énumération exhaustive est absente.

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-52 | sans optimum certifié — `--no-exact`, **ou** `n_q > MAX_ENUM_QUBITS`, ce qui est le cas de tout `dim ≥ 4` (32 qubits contre un plafond de 22) — `solver_panel` écrit `NaN` dans `hit_optimum` et `exact_match`. `check_expected_behaviour` les comparait par `<` à `MIN_HIT` / `MIN_MASK_MATCH`, et `nan < 1.0` vaut **False** : les dictionnaires `missed` et `diverging` restaient vides quoi qu'il arrive. Le critère qui existe pour que le script **puisse échouer** ne pouvait ni échouer ni réussir — il ne mesurait rien — et imprimait « H0 réfutée » sur une campagne où rien n'avait été certifié | run réel `python study/h0_selection/h0_optimiser_equivalence.py --scenario orszag_tang --re 400 --N 64 --dim 2 --n-snaps 1 --no-exact --no-resume --seed 0` : **code de sortie 0**, les 8 solveurs à `hit=nan` / `mask_match=nan`, et **deux verdicts contradictoires dans la même exécution** — `DECISION RULE` : « QAOA **deviates from the certified optimum** », puis trois lignes plus bas `[ACCEPTANCE]` : « **7 optimiseurs atteignent l'optimum certifié** et renvoient un masque identique → H0 réfutée ». Après, même commande : `[INDECIDABLE]`, et la `DECISION RULE` annonce `UNDECIDABLE at this size`. Chemin **certifié** inchangé, vérifié par `diff` sur la même commande sans `--no-exact` : lignes de verdict **bit-à-bit identiques** avant/après (8 solveurs, `mask_match = True`, `[ACCEPTANCE]`) | `pytest tests/study/test_h0_acceptance_uncertified.py` |

Aucun nombre publié ne bouge : `aggregate_master_table.py` ne lit que
`h0_optimiser_equivalence_N{N}_dim{dim}.npz` — dim=2, donc certifié — et le
seul artefact non certifié du dépôt
(`h0_optimiser_equivalence_N64_dim4_orszag_tang_noexact.npz`, `hit` et
`match` `NaN` sur ses 8 lignes) n'entre dans aucune ligne du master table.
La correction ne change que ce que le script **affirme** quand il n'a rien
certifié.

**T13, le contrôle et le balayage vide** — `h3_term_ablation.py` ne portait
**aucune** assertion : 0 `assert`, 0 `raise`, 0 `SystemExit` (mesuré par AST,
contre 5 et 6 dans son voisin `h0_optimiser_equivalence.py`).

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-54 | le contrôle `full` de T13 est une **tautologie** : `zero_hamiltonian_terms(hp, ())` rend une copie de `hp`, donc le masque comparé sort de la **même fonction sur la même entrée**. Il ne peut détecter qu'un indéterminisme d'`exhaustive_ground_state`, qui n'en a pas. `RESULTS.md` en tirait pourtant *« The control is exactly 0, which validates the measurement chain »* | mesuré en **sabotant** l'ablation (`TERM_KEYS` pointant sur des clés inexistantes, donc plus rien n'est jamais mis à zéro), orszag_tang Re=400 N=64 dim=2, 2 instantanés : contrôle `full` **0,000000 dans les deux cas**, et `no_ZZ` / `no_ZZZZ` / `Z_only` rendent **0,0000 / 0,0000 / 0,0000** — les trois lignes exactes sur lesquelles repose la lecture « causalement inertes » — identiques à l'ablation correcte. Seules `no_Z` / `couplings_only` (1,0000 → 0,0000) trahissaient le sabotage, et ce ne sont pas les lignes qui portent la conclusion. **L'instance n'est pas hypothétique** : D-51 a montré que `no_ZZZZ` annule `K_xpoint`, une clé que `ground_state_mask` ne lit jamais — une ablation réellement vide, que le contrôle n'a pas vue. Après : chaque ligne porte `removed_max`, le **max\|Δ\| de ce que `build_ising_terms` produit réellement** (opérateur assorti : c'est l'objet que la décision consomme), une ablation qui n'a rien retiré est imprimée `EMPTY` au lieu d'`inert`, et le contrôle est **asserté** au lieu d'être imprimé | `pytest tests/study/test_t13_control_is_not_vacuous.py` |
| D-55 | balayage vide silencieux : sans artefact d'entrée, `h3_term_ablation.py` imprimait `no input.` et **sortait avec le code 0**, sans écrire d'artefact — donc en laissant en place celui d'une campagne précédente. C'est le défaut que son voisin `h0_optimiser_equivalence.py` a déjà corrigé, mot pour mot (*« une campagne qui n'avait rien mesuré était indiscernable d'une campagne réussie »*) | même entrée (`--scenario no_such_scenario --N 64 --dim 2`) : T13 **code 0**, `h0_optimiser_equivalence` **code 1** avec un `RuntimeError` nommant les artefacts attendus. Après : T13 **code 1**, même message | `pytest tests/study/test_t13_control_is_not_vacuous.py -k empty` |

Aucun nombre publié ne bouge : les deux corrections ajoutent une colonne et
une sortie en erreur, et l'artefact `t13_term_ablation_*` existant n'est ni
relu ni réécrit. La **phrase** de T13 qui affirmait que le contrôle valide la
chaîne de mesure est corrigée, elle : elle contredisait la mesure.

**Le balayage vide, dans tout `study/`** — D-55 l'a trouvé dans un module ;
la même forme vivait dans onze autres.

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-56 | **11 modules** de `study/` gardaient la fin de leur balayage par `if not <accumulateur>: print(...); return` : sans artefact d'entrée, ils sortaient avec le **code 0 sans rien écrire**, laissant en place l'artefact de la campagne précédente. `CLAUDE.md` exige pourtant qu'« un balayage vide doit crier ». Onze autres modules levaient déjà — la règle était appliquée à moitié, et rien ne le signalait | mesuré sur trois d'entre eux, `--scenario no_such_scenario --N 64` : `h3_locality_proposition`, `h3_equivariance`, `h2b_learned_meanfield_h` — **code 0 → code 1** avec un `RuntimeError` nommant ce qui manque. Les onze sites : `h3_equivariance`, `h3_locality_proposition`, `h2b_ceiling_random_split`, `h2b_scenario_specialisation`, `h2b_variational_classifier`, `h2b_blocked_split`, `h2b_analytical_solution`, `h2b_learned_meanfield_h`, `h2b_train_linear_hamiltonian`, `closed_loop_campaign`, `pipeline_verification`. **Trois d'entre eux ont été trouvés par l'AST, pas par la recherche de « no input. »** : leur message était différent (`no completed fold.`, `no input found.`, `No coefficient files found.`) — chercher la chaîne en aurait manqué un quart | `pytest tests/study/test_empty_sweep_never_silent.py` |

Le test est paramétré sur **les 63 modules** de `study/` : un douzième site
apparaîtra tout seul. Il interroge l'AST, pas le texte du source — une
reformulation du message ne doit pas le casser — et il porte son propre
garde de balayage vide, puisque c'est exactement le piège qu'il traque.

**T26, la validation du proxy** — la tâche écrite pour répondre à
l'objection « à 8 qubits, évidemment qu'il ne se passe rien ».

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-57 | `h3_size_scan.py` remplace l'état fondamental exact par une descente gloutonne dès que `2·dim² > 22`, et son en-tête annonce « warm-started greedy (**validated at dim=2**) ». La validation était bien **calculée** (`greedy_agrees_with_exhaustive`) — puis rangée dans le JSON, **jamais imprimée, jamais contrôlée**. C'est pourtant ce seul nombre qui autorise à lire `dim = 4` et `dim = 8` | mesuré (N=96 et N=256, 4 scénarios canoniques, 12 instantanés) : `--mapper v1`, **le défaut de la tâche**, rend **0,7500** à dim=2 ; `--mapper v2` rend 1,0000. Le 0,75 figure **déjà** dans `results/t26_size_scan_N256_v1.json`, où rien ne le montre. Après : la table de synthèse porte une colonne `proxy=exact`, les dimensions portées par le proxy sont nommées, et un avertissement dit que l'en-tête se contredit — sans qu'aucun seuil ne soit inventé, l'alerte ne portant que sur `< 1` | `pytest tests/study/test_t26_proxy_validation_surfaced.py` |

**Ce que D-57 ne dit pas.** La conclusion de T26 n'est **pas** contaminée.
Le contrôle `--force-greedy` que le module prévoit lui-même a été rejoué à
dim=2, mappeurs v1 **et** v2 : le glouton rend `changed = 0,0000` sur les
quatre ablations, exactement comme l'exhaustif. Le proxy ne fabrique pas les
changements qu'il rapporte. Le risque que le module nomme dans sa propre
docstring était réel ; il ne s'est pas réalisé — et rien ne le disait.

**Le diagnostic Phase 1B** — en ré-auditant `check_tearing` pendant l'examen
de D-39 (même fonction, même PR) : son docstring exige un pic « strictement
à l'intérieur de la trace (pas à t=0, pas à la fin) », et la clause « pas à
t=0 » (`growing_from_start`) est bien vérifiée — mais la clause « pas à la
fin » (`growing`) compare `j[i_peak]` à `j[min(i_peak+1, len(j)-1)]` : quand
le pic tombe sur le **dernier** échantillon, `min(...)` retombe sur
`i_peak` lui-même, et la comparaison devient `j[i_peak] <= j[i_peak]*1.01`
— toujours vraie. La clause ne peut jamais échouer : une croissance qui ne
retombe jamais avant la fin de la fenêtre (donc jamais observée en train de
« piquer ») passe quand même.

Mesuré sur les 6 fichiers DNS `harris_tearing` réels de `results/` : avec
`J2` = `mean_sq_current` (câblage gelé), le pic tombe sur le **dernier**
échantillon (`i_peak = 19/20`) sur les **6/6** fichiers — la trace est
encore strictement croissante à la fin de la fenêtre simulée, ce n'est pas
un pic observé. `check_tearing` rendait pourtant `ok=True` (amplification
1,53–2,65×) sur les 6, exactement à cause du défaut ci-dessus.

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-42 | `check_tearing` : la clause « pic pas à la fin de la trace » se comparait à elle-même quand le pic tombait sur le dernier échantillon, donc ne pouvait jamais échouer | 6/6 fichiers `harris_tearing` (câblage gelé) : pic au dernier échantillon, `ok` **True → False** | `pytest tests/study/test_check_tearing_end_pinned_peak.py` |

**Conséquence pour D-39** (voir `DEFAUTS.md`) : la comparaison « ancien
câblage `ok=True` contre câblage corrigé `ok=False` » qui motivait D-39 est
maintenant à relire — sur les 6 fichiers disponibles, le câblage gelé ne
passait que grâce à ce défaut, pas parce qu'il observait un vrai pic. Une
fois D-42 appliqué, les deux câblages rendent `ok=False` sur les 6 : la
question posée par D-39 (quelle observable sépare fond stationnaire et
reconnexion) reste entière, mais elle ne peut plus s'appuyer sur « ça
marchait avant » — ça ne marchait pas, au sens où `check_tearing` l'exige.

**Le diagnostic de la campagne** — `src/analyze_hyperparams.py`, jamais
audité jusqu'ici (`COUVERTURE.md` §1), lit les bases Optuna gelées.

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-60 | `plot_threshold_operating_curve` — la seule figure qui montre l'arbitrage précision/coût du **bras classique**, dont le seuil est le seul paramètre optimisé — ne pouvait sortir d'aucune étude du dépôt, pour **trois raisons indépendantes** : (1) elle exigeait un paramètre nommé `threshold`, alors que `make_classical_composite_objective` échantillonne `threshold_amr` — `"threshold"` n'apparaît dans aucune des 10 bases ni dans aucune ligne de `src/` ; (2) elle lisait `phys_score` / `patch_ratio`, que seul l'objectif mono-scénario de `pipeline.py` écrit, quand la campagne déployée passe par `train_hyperparams._run_one_scenario`, qui écrit `phys_<scenario>` / `patch_<scenario>` et **jamais** les clés globales ; (3) son unique appelant était gardé par `has_decomposed_data`, qui teste `phys_score` : faux pour toute étude composite. Aucun message : une analyse amputée de sa figure de décision était indiscernable d'une analyse complète | `python src/analyze_hyperparams.py --db-path results/hyperparams/optuna_studies/classical_v2_phase1.db --study-name classical_v2_phase1` : **9 figures avant, 10 après** — `12_threshold_operating_curve.png` n'était produit sur **aucune** des deux études non vides. Ce que la figure manquante montre, mesuré sur les 125 essais complets de la base gelée : `r(threshold_amr, taux de patchs) = **−0,9690**`, `r(threshold_amr, erreur physique) = **+0,8369**` ; médianes par quartile de seuil `[0,050–0,089] → phys 0,0173 / patch 0,7240` puis `[0,226–0,800] → phys 0,2758 / patch 0,4809`. L'agrégation composite est la **moyenne** des scénarios, opérateur assorti : c'est celui que `_composite_loop` applique (`total / len(scenario_list)`). Aucun nombre publié ne bouge : aucun artefact, aucune figure du dépôt ne référence cette courbe | `pytest tests/pipeline/test_threshold_curve_reachable.py` |
| D-61 | `_add_trend`, la médiane par classe qui porte la **tendance** de quatre figures : ses bornes sortent de `linspace(x.min(), x.max())`, donc la dernière borne **est** `x.max()` — et le masque `x < bins[k+1]` en excluait la valeur. L'essai portant la plus grande valeur du paramètre n'entrait dans aucune classe, précisément là où la pente au bord du domaine échantillonné se décide | mesuré sur `q_has_v2_phase1` (178 essais complets), dernière médiane de la tendance contre la perte composite : `beta` **0,258164 → 0,258670**, `sigma` **0,261774 → 0,264327**, `beta_curl` **0,271467 → 0,276550**, `beta_xpoint` **0,254415 → 0,258812** ; `w_z_frac` inchangé (ses classes hautes n'atteignaient déjà pas deux points). Sur une entrée qui sépare — deux essais au maximum, valeur extrême — l'ancienne version ne trace **aucune** ligne, faute des trois classes exigées  La fonction existe en **deux exemplaires mot pour mot**, `analyze_hyperparams.py` et `recompute_lambda_scores.py` : les deux portaient le défaut, les deux sont corrigés, et un test compare ce qu'elles **tracent** pour qu'elles ne divergent pas | `pytest tests/pipeline/test_trend_last_bin_closed.py` |
| ~~D-63~~ | **numéro retiré — doublon de D-49.** La passe Vigil du 13 août a re-trouvé le `except Exception` de `recompute_lambda_scores.main` que D-49 avait corrigé le matin même sur la branche de base, et lui a donné un numéro neuf. La correction retenue est celle de **D-49** ; les tests de la seconde passe, qui mesuraient la même chose, ont été supprimés. Cause : la branche Vigil avait été ouverte avant D-49 et n'a fusionné sa base qu'après coup — voir « Deux passes sur les mêmes fichiers » dans `COUVERTURE.md`. Le numéro reste brûlé : il a été publié, il ne sera pas réattribué | — | `pytest tests/pipeline/test_recompute_lambda_scores.py -k d49` |
| D-62 | `plot_pareto_with_isocost` fixait sa fenêtre verticale à **(-0,05 ; 0,40) en dur**. C'est la figure qui porte le front de Pareto et les lignes d'iso-score du rescore : un point hors cadre ne se signale pas, il disparaît, et le front semble s'arrêter là où le cadre s'arrête | mesuré sur les deux bases gelées, erreur physique moyennée sur les scénarios : `q_has_v2_phase1` **0/178** hors cadre (phys ∈ [0,0348 ; 0,2997]) — figure **inchangée** — et `classical_v2_phase1` **9/125** hors cadre (phys ∈ [0,0114 ; **2,2749**]), dont **3 des 46 points du front de Pareto**. Après : la fenêtre reste (-0,05 ; 0,40) quand tout y entre et s'élargit aux données sinon. Aucun seuil inventé — les bornes viennent des points tracés | `pytest tests/pipeline/test_pareto_window_hides_nothing.py` |
| D-64 | `import_Neon_data_to_local.py` — le **seul** code du dépôt qui supprime une étude Optuna — supprimait l'étude de **destination** avant d'avoir lu la **source**, puis rattrapait tout échec par un message ❌ et un code de sortie **0**. Un import qui n'a rien importé était indiscernable d'un import réussi, et la destination était déjà détruite | mesuré, deux SQLite (`--in-url` remplace Neon) : destination à **5 essais**, source ne portant pas l'étude → avant, `KeyError`, **code 0**, et **l'étude locale n'existe plus** ; après, **5 essais intacts**, code 0, ligne « destination laissée intacte ». Échec réel de copie (destination inouvrable) : **code 0 → code 1**. Import réel : inchangé, 2 → **7** essais copiés. **L'empreinte est dans le dépôt** : 8 des 10 bases de `results/hyperparams/optuna_studies/` portent le schéma Optuna complet et **zéro ligne**, et `classical_v2_phase2` / `classical_v2_phase3` pèsent **274 432** et **299 008** octets là où un schéma neuf en pèse **114 688** — des pages libérées, donc des lignes écrites puis supprimées. Ce n'est pas une preuve que ce script les a vidées ; c'est la fermeture du chemin qui le fait | `pytest tests/pipeline/test_import_never_destroys_destination.py` |
| D-68 | `plot_amr_state` — la **seule** fonction de `src/visual.py` et `src/help_visual.py` qui s'exécute en production : `pipeline.py` l'appelle 4× par pas de verrouillage et sauve un PNG à chaque fois — étiquetait `Grid X` l'axe horizontal et `Grid Y` le vertical. `imshow` place l'axe 0 du tableau en **vertical** ; `Jz` étant indexé `[X, Y]` (`grid.py` : `AXIS_X = 0`, `AXIS_Y = 1`, et `get_fluxes` forme bien `dBy/dX − dBx/dY` avec `axis=0` pour X), l'axe horizontal porte **Y**. Les deux étiquettes nommaient donc l'axe de l'autre. La cause est écrite trois lignes plus haut : un commentaire annonçait « axis=1 est d/dx (colonnes), axis=0 est d/dy (lignes) », la convention `indexing='xy'` que `grid.py` désigne explicitement comme n'étant pas celle du dépôt. La figure **paraissait juste** — les cadres d'attention tombent bien sur les structures qu'ils désignent, vérifié — seule la lecture des positions était transposée | champ d'essai qui **sépare** (une cellule brillante hors diagonale, `Jz ≠ Jzᵀ`) : structure posée en **X=10, Y=40** au sens de `grid.py`, **relue sur la figure X=40, Y=10** avant, **X=10, Y=40** après. Seules les **étiquettes** changent : le tableau passé à `imshow` et les cadres sont bit-à-bit identiques, un test l'épingle. Aucun nombre publié ne bouge ; la géométrie des PNG déjà publiés non plus | `pytest tests/pipeline/test_amr_figure_axes.py` |
| D-69 | **Rapport seul à l'ouverture, clos par remesure.** La table T31 (« La convention d'axes des mappeurs », plus bas) publiait à `8ee5c8a` le seul verdict tranché du module — *« corriger sans réoptimiser dégrade à dim=16 »*, IC95 [−0.1328, **−0.0146**], qui exclut zéro. Rejouée par ses deux propres commandes, elle ne ressortait plus. **Deux déplacements, chacun attribué à sa cause par la mesure, les trois points rejoués dans un seul environnement** : `8ee5c8a` → `47012fa` vient du solveur (D-25, D-26/D-27) ; `47012fa` → HEAD vient de **D-70 seul** — rejoué à `dffac18`, la correction de `_hard_patches` et rien d'autre, les quatre lignes sortent identiques au dernier chiffre à celles de HEAD, aucun commit ultérieur (dont D-91) ne les déplace. Le second n'était pas mesuré et il est le plus lourd : le Δ du F1 à dim=8 **change de signe** (+0.0391 → −0.0312) et les IC95 s'élargissent d'un facteur ~3 à dim=8, la vérité terrain canonique étant plus hétérogène par scénario que l'écart-type intra-patch qu'elle remplace. **La lecture publiée est rétractée** : refaite à `95571d1`, la table ne porte **aucun** verdict tranché — l'IC95 du Spearman à dim=16 passe de [−0.1328, −0.0146] à [−0.1673, **+0.0343**]. Ce qui subsiste est que les quatre Δ sont négatifs et qu'aucun ne montre de gain. Ni l'environnement ni le bruit n'expliquent l'écart, mesuré : au hash `47012fa` cet environnement rend la colonne `47012fa` au dernier chiffre publié, et les deux commandes rejouées à l'identique rendent une sortie bit-à-bit identique (2 exécutions par dim). Aucun compteur du master table ne bouge — T31 n'y figure pas : 180 / 176 / 4 / 0 avant comme après | verdict dim=16 **dégrade → indécidable** ; Δ F1 dim=8 **+0.0391 → −0.0312** | `pytest tests/study/test_curl_convention_gap.py -k "published_table or excludes_zero"` |
| D-70 | `_hard_patches` (`study/h1_solver/h1_curl_convention_gap.py`) — la « vérité terrain » de durété utilisée par les métriques de classement de T31 (Spearman, F1 à budget apparié) : sa docstring promettait « même définition que `study/pipeline/hard_patch_labels.py` » (l'erreur L2 de reconstruction par grossissement en bloc, sommée sur les 4 champs, normalisée par le RMS global). Le corps calculait autre chose — l'écart-type intra-patch de la **norme** du champ (`sqrt(vx²+vy²+Bx²+By²)`) — une formule qui coïncide avec la canonique sur un champ lisse mais **s'inverse** dès qu'un patch oscille à magnitude constante : l'écart-type y est nul (rien ne « varie » en norme) alors que l'information fine y est totalement détruite par le grossissement en bloc, donc l'erreur de reconstruction canonique y est maximale | champ d'essai qui **sépare** : un patch en damier `vx=±1` (les 3 autres champs nuls dans ce patch, magnitude rigoureusement constante `= 1`), sur fond de bruit lisse. Ancienne formule : `0,0000` — **minimum** du champ entier, patch jugé le plus facile. `patch_l2_errors` (canonique) : `1,0314` — **maximum** du champ, patch le plus difficile. Classement inversé, pas seulement décalé. Nouvelle formule : identique à `patch_l2_errors` à **1,1e−16** près sur un champ aléatoire quelconque (4 champs, 32×32, 4×4 patches), et retrouve le damier comme patch le plus dur. Aucun nombre publié ne bouge — les artefacts `.npz` de T31 sont déjà signalés non reproductibles pour une autre raison (D-69, `DEFAUTS.md`), donc rien de publié aujourd'hui ne dépendait de cette version | `pytest tests/study/test_hard_patches_matches_canonical.py` |
| D-71 | La réorganisation `17d983d` a déplacé **et** renommé chaque script de `study/v4/tNN_xxx.py` vers `study/<module>/<module>_xxx.py`, sans toucher les commandes de reproduction qui les citent : 16 chemins distincts dans `docs/RESULTS.md` (21 occurrences), les docstrings d'usage de 21 scripts, et surtout **deux lanceurs `scripts/` qui invoquaient réellement ces chemins** — `run_fold.sh` et `run_leak_free_campaign.sh` auraient échoué dès le premier appel Python. Le second portait un défaut composé, invisible tant qu'on ne regarde que la chaîne `study/v4/` : `$HERE`/`$ROOT` étaient calculés pour une profondeur de deux niveaux sous la racine (`study/v4/`, vraie au moment où le script a été écrit) et n'ont pas été réajustés pour `scripts/` (un seul niveau) — `ROOT` résolvait au **parent du dépôt**, `RESULTS="$ROOT/study/results"` à un chemin qui n'a plus existé après l'aplatissement de `study/results/` vers `results/`, et l'invocation Python utilisait `$HERE/t22_unseen_conditions.py`, un fichier qui n'a jamais vécu dans `scripts/` | mesuré, chaque script isolément : `run_fold.sh`, `root` résolvait à un dossier **hors du dépôt** (existant, donc `cd` réussissait, mais sans `study/` dedans) ; `run_leak_free_campaign.sh`, `fold_status()` sur un artefact réel (`results/t22_unseen_leak-free_ot.json`, `status=completed`) rendait **`absent`** avant correction (le chemin `RESULTS` ne menait nulle part), **`completed`** après. Les 21 fichiers `study/`/`figures/` : substitution mécanique vérifiée fichier par fichier contre l'arborescence réelle, flags CLI de chaque script inchangés (spot-check exhaustif des `add_argument` contre les commandes historiques) — aucune régénération d'artefact conservée, seul le git-hash de provenance aurait bougé. Exclus délibérément : `docs/archive/*` (déclaré obsolète, jamais cité), `results/v4_master_table.md` (artefact déjà généré, se corrige à la prochaine régénération réelle), et les citations narratives d'une campagne passée (`tests/study/test_silent_failure_sweep.py`, le paragraphe « Trap sweep » de ce fichier) — elles décrivent un fait vrai au moment où il a été écrit, pas une commande à rejouer | `pytest tests/study/test_repro_commands_point_to_real_files.py` |
| D-72 | `study/h1_solver/h1_solver_convergence.py` (T14) — le script qui **valide numériquement le solveur** — suivait `max\|div B\| / rms\|B\|` le long de chaque trajectoire avec `dns_validation.div_B`, une divergence **spectrale**, et en faisait son critère d'acceptation (`ALL CHECKS`, stocké en `all_checks_pass`, deux lignes du master table). La docstring de `div_B` dit pourquoi c'était juste au moment où elle a été écrite : « same convention as the solver's FFT projection ». Ce n'est plus la convention du solveur depuis **D-25** : `PROJECT_B = False`, B n'est plus projeté spectralement — il est solénoïdal **aux différences finies** par construction, l'induction étant en forme rotationnelle `rhs_B = (∂Ez/∂y, −∂Ez/∂x)` dont la divergence FD4 est exactement nulle puisque les décalages de `np.roll` commutent. Mesurer un champ FD-solénoïdal avec un opérateur spectral ne mesure pas la contrainte : cela mesure l'écart entre les deux opérateurs. D-25 a corrigé le solveur ; le diagnostic chargé de le surveiller est resté sur l'ancien opérateur, et personne ne l'a relu — `COUVERTURE.md` déclarait ce module « lu en entier » | sur la configuration **publiée** de T14 (`orszag_tang`, grilles 32/64/128, `t_end=0,5`, Re=400 puis 200/3200), rejouée à HEAD **sans réécrire l'artefact** : `max\|div B\|/rms\|B\|` vaut **3,9029e−02** avec l'opérateur spectral contre **2,0266e−14** avec l'opérateur assorti (FD4), et `all_checks_pass` bascule **True → False** contre le seuil de 1e−3 — alors que ce fichier publie « entre 5,6e−15 et 8,0e−14 — machine precision ». Le faux signal **croît quand la grille grossit** (mesuré : N=128 **2,3103e−04**, N=64 **4,5675e−03**, N=32 **3,9029e−02**), donc la « validation » passait ou échouait **selon la résolution**, pour une contrainte respectée à 1e−14 partout. Champ qui **sépare** le plus, retenu pour le test : `mhd_rotor`, N=32, `t_end=0,05` (5 pas, < 0,1 s) — spectral **6,0470e−01**, FD4 **2,0905e−15**, onze ordres de grandeur. Après correction, sur les 5 trajectoires de la configuration publiée : **3,7632e−15 à 2,0266e−14**, soit la bande publiée. **Aucun nombre publié ne bouge** : l'artefact `results/t14_numerical_validation.npz` n'est pas régénéré ici — il date de `1f03713`, quand la projection spectrale de B était encore active et l'opérateur donc assorti — et le master table continue de le lire tel quel | `pytest tests/study/test_t14_divb_uses_matched_operator.py` |
| D-73 | Même famille que D-72, et celui-ci **bloquait la régénération des DNS**. `validate_one` (`study/pipeline/dns_extension.py`) est la porte que franchit **chaque trajectoire DNS nouvellement générée** (`dns_extension.py:460`) : elle rejette la trajectoire si `div_rel_max > 1e-3`. Cette valeur vient de `analyse_one`, dans le fichier **gelé** `dns_validation.py`, qui la calcule **au spectral** — et dont le commentaire porte la condition désormais fausse : « should be O(eps_machine) **when the FFT projection is applied** ». Depuis D-25 elle ne l'est plus pour B. Le portail rejetait donc des trajectoires saines. Le correctif ne touche pas le fichier gelé : `dns_extension` héberge déjà les observables corrigées (`mean_sq_current_fixed`, `fluctuating_ke_fixed`, `check_kh_fixed`), la divergence assortie les rejoint | mesuré de bout en bout, DNS réellement générée à HEAD (`harris_tearing`, Re=400, N=64, seed=0, écrite **hors du dépôt**) : `validate_one` rendait **FAIL — `divB 1,6e-02`** ; le même instantané mesuré avec l'opérateur assorti rend **5,0573e−06**, soit **OK**. Les 24 artefacts DNS **déjà dans le dépôt** passent dans les deux cas (mesuré sur 8 : **1,4170e−05 à 5,6503e−05**) — ils datent d'avant D-25, quand la projection spectrale de B était active ; ce qu'on lit chez eux est le **plancher de stockage float32**, pas l'écart d'opérateur. **Aucun artefact publié ne change de statut.** Noté sans être corrigé, le fichier étant gelé : `analyse_one` passe `dx = 1/N` à `div_B` alors que `PeriodicGrid` pose `L = 2π`, donc `dx = 2π/N` — un facteur **6,2832** sur toute divergence qu'il rapporte | `pytest tests/study/test_dns_gate_accepts_a_solenoidal_trajectory.py` |
| D-74 | `study/closed_loop/closed_loop_budget_matched.py` (T15b, comparaison à budget apparié) était le seul des 9 fichiers de `study/closed_loop/` sans `assert` ni `raise` (mesuré par AST). Ses deux gardes d'entrée — fold sans artefact `t15_level3_fold_*.json`, fold inconnu de `train_hyperparams.fold_scenarios` — faisaient `print(...); return` : un balayage vide (`CLAUDE.md` : « un balayage vide doit crier ») indiscernable d'une exécution réussie, exactement le défaut D-56 déjà corrigé sur 11 sites soeurs de `study/` — `closed_loop_campaign.py`, dans le même dossier, compris — mais jamais appliqué à celui-ci. Le détecteur AST de `tests/study/test_empty_sweep_never_silent.py` ne le voyait pas : il ne cherche que la forme `if not <accumulateur nommé>:` après une boucle, pas une garde d'entrée manquante | avant, deux commandes distinctes : `--fold no_such_scenario_xyz` (artefact absent) et un fold dont l'artefact `t15` existe mais dont le nom n'est reconnu par aucun scénario de `train_hyperparams` — **code 0** dans les deux cas, sans rien écrire. Après : **code 1**, message nommant D-74, dans les deux cas | `pytest tests/study/test_closed_loop_budget_matched_missing_input_not_silent.py` |
| D-75 | **Quatorze fichiers** de `study/` gardaient encore leur entrée par `print(...); return` : sur une donnée d'entrée absente ou dégénérée, le processus sortait avec le **code 0 sans écrire d'artefact**, indiscernable d'une campagne réussie — le défaut que D-56 avait corrigé sur 11 sites et D-74 sur un douzième. Ils sont restés invisibles parce que le détecteur AST écrit pour D-56 ne reconnaît **qu'une seule forme syntaxique** — `if not <accumulateur>:`, et encore faut-il que le nom figure dans une liste tenue à la main (`rows`, `configs`, `by_scene`, …). Les gardes réelles s'écrivent `if len(by_scene) < 2:`, `if len(set(...)) < 2:`, `if not Xs:`, `if not all_d:`, `if not cfgs:`, `if len(np.unique(Ytr)) < 2:` : **aucune ne correspond**. Douze sites dans `study/h2b_prediction/` (11 fichiers sur 19), deux dans `study/pipeline/` — `hard_patch_labels.py` (phase 2 du pipeline, `results/` vide → « No DNS files found », code 0) et `label_percentile_sensitivity.py` ; plus un quinzième dans `pipeline_verification.py`, deuxième garde du fichier même que D-56 avait corrigé une fois. Le nouveau détecteur ne connaît aucune liste de noms : il flaire la **forme** « garde de données qui rend la main sans lever », et exempte les sorties pilotées par un drapeau CLI (`args.dry_run`) et les `sys.exit(1)` | mesuré, **code de sortie du processus**, même commande `--scenario no_such_scenario --N 64` — **13 sites 0 → 1** : `h2b_feature_selection`, `h2b_loso_transfer`, `h2b_loso_bootstrap`, `h2b_neighbour_cone_curve`, `h2b_prediction_horizon`, `h2b_psi_feature_loso`, `h2b_v1_hamiltonian_loso`, `h2b_multiseed`, `h2b_random_split_bootstrap`, `h2b_scenario_ablation`, `h2b_dynamic_ground_truth`, `label_percentile_sensitivity`, et `hard_patch_labels` (mesuré à part, arbre où `results/` est vide). Les **deux sites restants** — la garde « jeu d'entraînement dégénéré » de `h2b_random_split_bootstrap` et « toutes les lignes dégénérées » de `pipeline_verification` — ne se déclenchent pas depuis la ligne de commande : corrigés et couverts par le détecteur AST, **pas** par une mesure de bout en bout, et c'est dit plutôt qu'arrondi. Le test échoue **17 fois** sur l'ancienne version (14 fichiers + 3 mesures de bout en bout), 0 après | `pytest tests/study/test_empty_sweep_guard_shapes.py` |
| D-76 | Même défaut que D-71, à trois sites que son balayage n'a pas couverts : son test **nomme** les deux lanceurs qu'il avait corrigés (`run_fold.sh`, `run_leak_free_campaign.sh`) et ne regarde aucun autre. `scripts/run_study_v2_phases.sh` (5 invocations) et `scripts/run_study_v2b.sh` (27 invocations) portaient encore les chemins d'avant `17d983d` — **22 cibles distinctes**, toutes déplacées *et* renommées (`study/phase11_upper_bound.py` → `study/h2b_prediction/h2b_ceiling_random_split.py`, `study/phase6_verify.py` → `study/pipeline/pipeline_verification.py`, …). Aucun des deux ne porte le moindre avertissement : ils se présentent comme les lanceurs de campagne vivants du dépôt, `set -euo pipefail` compris. Second défaut composé, même forme que celui de `run_leak_free_campaign.sh` : leur dernière ligne liste `study/results/`, aplati vers `results/` à la même réorganisation | mesuré, `bash scripts/run_study_v2_phases.sh 2` : avant **code 2**, `python: can't open file '.../study/hard_patch_labels.py'` — mort au **premier** appel ; après **code 0**, phase 2 exécutée de bout en bout. Le listing final : avant « (no results yet) » alors que `results/` porte **224** `.npz`, après la liste réelle. Chemins vérifiés fichier par fichier contre l'arborescence, et **chaque drapeau CLI passé par les lanceurs confirmé présent dans le `--help` de sa nouvelle cible — 22/22, aucun manquant**. `run_study_v3.sh`, mort lui aussi, n'est **pas** corrigé : il est gelé et documenté comme tel par D-49 ; le test relit son avertissement plutôt que de l'exempter par son nom. **Vérification de reproductibilité obtenue au passage** : la phase 2 relancée par le lanceur corrigé réécrit 12 artefacts `patches_*_Re400_*_dim2.npz` **identiques au bit près** à ceux du dépôt (`git status` vide). Le test échoue **4 fois** sur l'ancienne version, 0 après | `pytest tests/study/test_every_launcher_invokes_real_files.py` |
| D-140 | **Même famille que D-71 et D-76, un cran plus profond : le chemin existe, l'option non.** La ligne « Vérifier » de **D-53** — le résultat le plus fort du dépôt — invoquait `h0_optimiser_equivalence.py` avec une option `--check` suivie de l'artefact `dim = 3`. Le script ne déclare pas cette option : il n'appelle `check_expected_behaviour` qu'au terme d'une campagne complète, jamais contre un artefact stocké. La commande rendait `error: unrecognized arguments` et sortait en **2** — un lecteur qui suit `RESULTS.md` pour contrôler la conclusion la plus forte du dépôt n'obtient rien. Le garde de D-71 la laissait passer parce qu'il vérifie l'existence du **fichier**, pas des **options**. Portée mesurée avant de conclure : balayage des **16** commandes distinctes de `RESULTS.md` portant au moins une option longue, sur **12** scripts interrogés par leur propre `--help`, **0** ignoré — **un seul** défaut, celui-ci. Le résultat de D-53, lui, n'est pas en cause : il est vérifié par `tests/study/test_h0_certified_dim3_contradicts_criterion.py`, qui reconstruit `(summary, solvers, diag_flags)` depuis chaque artefact et appelle le critère. C'est la commande publiée qui était fausse, pas le nombre | commande **exit 2 → 6 passed** ; garde : **1 failed → 10 passed** | `pytest tests/study/test_repro_commands_point_to_real_files.py` |
| D-142 | **Même défaut que D-71, sur la moitié que son garde ne regardait pas.** Le balayage de D-71 couvre `study|scripts|figures` ; les commandes de reproduction les plus nombreuses de `RESULTS.md` sont des `pytest tests/…`, et **rien** ne les regardait. **Dix** chemins y étaient restés à leur emplacement d'avant `17d983d` (`tests/test_mapper_contracts.py` au lieu de `tests/mapping/…`, les sept fichiers de la suite QAOA au lieu de `tests/quantum/…`). **Deux blocs de recette entiers étaient morts**, dont celui qui porte le « Verdict de la suite QAOA ». Portée mesurée : **29** chemins `tests/…py` cités comme commande, **19** valides, **10** absents — tous avec un homologue réel retrouvé par nom de base. Le garde ajouté suit le **contexte de commande** et non la ligne : une commande `pytest` s'étale sur plusieurs lignes, par `\` dans un bloc clôturé ou par simple retour à la ligne dans un span inline. Les lignes de **table** sont exclues par leur forme — l'inventaire de la suite QAOA d'avant la réorganisation en cite plusieurs et ce sont des faits historiques, pas des commandes ; aucune liste d'exceptions à tenir à la main | recette `mapper/hamiltonian/downsampling` **exit 4 → 117 passed** ; verdict QAOA **exit 4 → 30 tests collectés** (non exécutée : ~30 min) | `pytest tests/study/test_repro_commands_point_to_real_files.py -k pytest_command` |
| D-144 | **Les quatre gardes de la décision D-47 lisent le SOURCE ; deux ne peuvent pas rougir, et un rougit à tort.** `tests/study/test_phase5_ne_filtre_plus_sur_promising.py` garde les deux moitiés de la décision de USER sur D-47 — la phase 5 ne filtre plus sur `promising`, et le compte reste imprimé comme diagnostic. Son détecteur AST n'est pas en cause : il reconnaît bien la forme qu'il cherche (`if ... promising ...: continue`), son auto-test le vérifie. Ce qu'il ne voit pas, c'est que **filtrer ne demande pas un `if`** — réduire `snap_indices` avant la boucle suffit, et la ligne `for idx in range(len(snap_indices)):` qu'un second test cherche mot à mot reste intacte. Le troisième, qui garde le diagnostic, cherche `n_promising` et `diagnostic` dans le fichier : les deux jetons survivent dans le **commentaire D-47** placé juste au-dessus du `print`, donc supprimer le `print` ne le fait pas bouger. Le quatrième (`promising` toujours calculé en phase 4) est **surestimé** et non défaut : la clé retirée du dict de `analyze_snapshot` laisse le garde vert mais rend `test_exact_diag_degenerate_gate.py` **5 failed** — le comportement est couvert ailleurs. Corrigé en ajoutant un banc **comportemental** qui exécute `run_phase5` de bout en bout sur des artefacts synthétiques, `prepare_qaoa_inputs`, `run_qaoa_on_snapshot` et `full_comparison` remplacés : ce qui est mesuré est la SÉLECTION, pas le QAOA (~3 s). Champ d'essai qui SÉPARE : `promising = [True, False, True]` — sur le tout-à-True mesuré en production (40/40) un filtre réintroduit serait invisible | A′-1 (filtre rendu avant la boucle) ancien **5 passed** → nouveau **1 failed** ; A′-2 (`print` supprimé) ancien **5 passed** → nouveau **1 failed** ; B (réécriture équivalente `enumerate`) ancien **1 failed — faux rouge** → nouveau **3 passed** ; sain : **3 passed** | `pytest tests/study/test_phase5_traite_tous_les_instantanes.py -q` |
| D-145 | **Le balayage qui interdit la fuite du JSON dans la boucle fermée ne voyait qu'une des deux façons de lire `hp`.** `src/pipeline.py` charge `best_hyperparams.json` puis fusionne le dict de l'appelant par-dessus : toute clé absente du dict de l'appelant vient du JSON, dont les valeurs ne sont **pas** celles de l'étude (écart mesuré ×100 sur `w_z_frac`). `test_the_closed_loop_covers_every_key_the_pipeline_reads` interdit cette fuite en comparant les clés lues par le pipeline à celles que la campagne fournit — mais son énumération `_live_pipeline_keys()` était un **regex** sur `hp.get('…')`, après retrait textuel des blocs `"""…"""`. Elle ne voyait donc pas `hp['clé']`, l'autre écriture de la même lecture, et **rien ne l'obligeait à trouver quoi que ce soit** : un balayage qui rend l'ensemble vide rend `missing` vide, donc vert. Deux mesures, dans les deux sens : une clé neuve lue par souscription — la fuite exacte que le test existe pour interdire — laissait **13 passed** ; et une réécriture *équivalente* d'une lecture existante (`hp['kappa'] if 'kappa' in hp else …`) faisait tomber le balayage de **10 clés à 9**, en silence, rétrécissant la garantie sans rougir. Corrigé par l'AST — qui voit les deux formes, n'a pas besoin qu'on lui retire le bloc mort (un littéral de chaîne reste une constante), **lève** sur une clé calculée qu'aucun balayage statique ne pourrait couvrir, et porte un plancher écrit à la valeur mesurée (10). C'est déjà la technique de `_cles_hp_get_vivantes` (`tests/pipeline/test_relative_percentile_is_trainable.py`), qui balaie le même fichier dans l'autre sens — les deux énumérations du même objet ne se ressemblaient pas | A′ (clé neuve en souscription) **13 passed → 1 failed** ; B (réécriture équivalente) **13 passed**, sans faux rouge ; ancien balayage sur B : **10 clés → 9** | `pytest tests/study/test_hyperparams_two_sources.py -q -k closed_loop_covers` |
| D-146 | **Le garde qui doit tenir écrite la décision de ne pas corriger D-53 — le résultat le plus fort du dépôt — ne peut pas rougir, et il a DÉJÀ manqué le mouvement qu'il surveille.** `test_the_decision_not_to_correct_stays_written` (`tests/study/test_h0_certified_dim3_contradicts_criterion.py`) faisait `assert "D-53" in docs/DEFAUTS.md` : quatre caractères cherchés dans 1 361 lignes. Or D-53 est **clos** — son entrée a quitté `DEFAUTS.md` pour `RESULTS.md` (`# D-53 —`, 3 sections) — et le garde est resté vert, satisfait par une ligne `| D-53 | … |` d'un **tableau de synthèse rangé dans l'entrée D-132** et par un paragraphe d'intro périmé. C'est la règle de `VIGIL.md` « ne jamais laisser une déviation connue non écrite » gardée par un jeton, pas par l'entrée. Ce que le garde doit distinguer : une **entrée** (un titre qui nomme le défaut, ou une ligne du registre des corrigés) d'une **référence croisée** (le numéro cité dans la prose d'un autre défaut) — et l'entrée doit porter encore les nombres qui la rendent lisible, un titre nu ne consignant aucune décision. Corrigé par un détecteur d'entrées qui suit la hiérarchie des titres, accepte **l'un OU l'autre** registre (un défaut clos change de fichier : l'exiger dans un fichier nommé ferait rougir sur ce mouvement voulu) et exige `0,156` / `0,062` / `1,000` dans le corps, virgule ou point | **A′** — les 3 sections `# …D-53…` retirées de `RESULTS.md`, **toutes** les références croisées laissées en place dans les deux fichiers (4 dans `RESULTS.md`, 2 dans `DEFAUTS.md`) : ancien garde **6 passed**, nouveau **1 failed**. **B** — l'entrée **déplacée** de `RESULTS.md` vers `DEFAUTS.md` en `## D-53` (changement voulu) : nouveau **7 passed**, pas de faux rouge. Arbre sain : 6 → **7 passed**. Le détecteur porte son propre auto-test (référence croisée, prose, section à sous-titres, ligne de registre, et `D-5` qui ne doit pas être trouvé dans `D-53`) | `pytest tests/study/test_h0_certified_dim3_contradicts_criterion.py -q` |
| D-147 | **Même famille que D-146, sur le seul dossier du dépôt qu'aucune commande ne régénère.** `test_the_document_states_the_measured_numbers` (`tests/study/test_hyperparams_provenance.py`) exigeait que `PROVENANCE.md` « porte les chiffres » en cherchant les **sous-chaînes** `"345"`, `"16.6"`, `"30.4"`, `"47"`, `"224"`, `"9.3"`, `"56 min"`. Or `"345"` est contenu dans `3450`, `"47"` dans `470`, `"224"` dans `2244` : le document peut annoncer **chaque total à un ordre de grandeur près** sans qu'aucun test ne bouge. Ce ne sont pas des nombres décoratifs — ce sont ceux qui chiffrent la réoptimisation (224 h CPU, 56 min/essai), et une **mauvaise lecture de ce document exact** a déjà fait appliquer un facteur ×1,7 fantôme, corrigé sur la branche vive à `41a3e84`. Corrigé : chaque nombre exigé est désormais **recalculé depuis les bases Optuna à l'instant du test** — total d'essais, mur par base et total, CPU par base et total, jours mono-cœur, coût médian par essai — et cherché comme nombre **délimité** (ni chiffre collé devant, ni chiffre ou décimale collés derrière). Le document et le test ne peuvent plus dériver l'un de l'autre : c'est le point de D-22 | **A′** — `345`→`3450`, `47.0`→`470.0`, `224.4`→`2244.0`, `9.3`→`9.35` dans `PROVENANCE.md`, chaque jeton cherché par l'ancien garde restant présent en sous-chaîne : ancien **10 passed**, nouveau **1 failed**. **B** — `**345 essais, ~47 h de mur.**` réécrit en `**345 essais** pour **47.0 h** de mur (soit 47 heures).` : nouveau **11 passed**, pas de faux rouge. Arbre sain : 10 → **11 passed**. Le détecteur porte son propre auto-test, dont `9.35 jours` qui ne doit pas satisfaire `9.3` — le cas qui a fait rougir la première écriture du garde | `pytest tests/study/test_hyperparams_provenance.py -q` |
| D-148 | **La famille D-55/D-56 était déclarée fermée — « ceci ferme la famille », en tête de `test_empty_sweep_never_silent.py` — alors que SIX modules de `study/` sortaient encore avec le code 0 sans rien écrire : les phases 2, 3, 4, 5, 7 et 8 du pipeline.** Le détecteur de D-56 cherche une forme (`if not <nom>: … return`) avec une liste de noms **tenue à la main** ; balayage généralisé à tout nom et toute forme : **30 sites** de `study/` répondent, **aucun** n'est dans `ACCUMULATORS` — le détecteur en voyait zéro. La forme vivante est autre : `if <accumulateur>:` autour du résumé, puis on tombe en bas de `main()`. C'est la leçon de D-56 elle-même d'un cran plus haut (elle avait trouvé 3 modules par l'AST que la recherche de la chaîne « no input. » manquait, leur **message** différant ; ici c'est la **forme** qui diffère). `hard_patch_labels.py` illustre le piège : la garde de D-75 y couvre « aucun `dns_*.npz` du tout » et pas « aucun ne correspond à la demande » — un `--scenario` mal orthographié suffit. **Pourquoi ça compte maintenant** : `BRIEF_REPRISE.md` §11 demande de **relancer phase 4, T13, T26** pour lever les seuils périmés ; une relance qui ne mesure rien, laisse les artefacts périmés en place et imprime « Phase 4 complete. » se lit comme une relance réussie. Corrigé aux six sites (`raise RuntimeError("balayage vide : …")`, même forme que D-56/D-75). `study/pipeline/dns_validation.py` sort aussi 0 mais **n'est pas corrigé** : il est **gelé**, et il écrit son artefact — exempté dans le garde, avec sa raison écrite et un test qui l'exige | `--scenario no_such_scenario --N 64`, avant → après : `hard_patch_labels` **0 → 1**, `hamiltonian_coefficients` **0 → 1**, `exact_diagonalisation` **0 → 1**, `qaoa_inputs` **0 → 1**, `ising_terms_and_annealing` **0 → 1**, `h3_depth_report` **0 → 1**. Garde **comportemental** (il n'inspecte aucune forme : il exécute les 61 modules lançables de `study/` et exige un code non nul, ~2 min) : sur l'ancien arbre **6 failed, 124 passed, 1 skipped**, sur le nouveau **130 passed, 1 skipped**. Portée vérifiée sur les 13 fichiers de test qui lancent ces modules avec de VRAIES entrées, ancien et nouvel arbre : **3 failed, 289 passed, 61 skipped** des deux côtés, à l'identique — les 3 sont le trio `a0e0e02` déjà rouge à l'arrivée, aucun nouveau | `pytest tests/study/test_empty_sweep_never_silent.py -q` |
| D-149 | **Le « garde-fou du garde-fou » anti-D-31 compte une chaîne à l'espacement près : il reste vert sur ce qu'il annonce détecter, et rouge sur ce qui est voulu.** `test_le_bloc_mort_ne_compte_pas` (`tests/pipeline/test_relative_percentile_is_trainable.py`) faisait `source.count("w_z_frac    = hp.get(") == 2` — quatre espaces compris. Sa propre docstring dit : « si le bloc commenté de `pipeline.py` **devient du code**, ce test le dit ». Il ne le dit pas : le bloc mort transformé en code, son texte laissé intact, le compte reste 2. Or c'est un vrai risque, pas une hypothèse — le fichier existe parce qu'une insertion a **déjà** visé ce bloc-là. Bloc devenu vivant, le balayage anti-D-31 se satisfait de lectures que le bloc vivant **réécrit quinze lignes plus bas** : D-31 exactement, sous la forme que ce fichier existe pour empêcher. Corrigé : le bloc mort est retrouvé par l'**AST** (un littéral de chaîne posé comme instruction, donc insensible à l'indentation) et la garde porte sur la **canarie** — `beta_grad`, la seule clé qui n'existe que dans le bloc mort ; si elle apparaît parmi les clés vivantes, le bloc est devenu du code. Plus un test qui exige que le bloc mort existe encore, sans quoi la canarie serait un balayage vide | **A′** — bloc mort transformé en code, texte inchangé : ancien **21 passed**, nouveau **2 failed, 20 passed**. **B** — bloc mort réindenté (`w_z_frac = hp.get(`, un espace ; changement voulu, aucun défaut) : ancien **1 failed — faux rouge, le 4ᵉ de cette forme dans ce dépôt**, nouveau **22 passed**. Arbre sain : 21 → **22 passed** | `pytest tests/pipeline/test_relative_percentile_is_trainable.py -q` |
| D-150 | **La conclusion de fig15 — celle que D-102 avait corrigee — est gardee par deux chaines du source : elle peut citer de nouveau le σ d'un AUTRE module sans qu'aucun test ne bouge, et une reecriture de guillemets la fait rougir.** `tests/study/test_fig15_sigma_narration.py` faisait `assert "σ=0.023" not in src` et `assert "sigma_trained = TRAINED_PARAMS.get('sigma', 0.05)" in src`. Le premier cherche une chaine a l'espacement pres : `σ = 0.023` ne le declenche pas. Le second teste la mise en forme, pas la structure. Or ce qui est garde est un **comportement** — ce que le bloc CONCLUSION imprime — et D-102 existe parce que ce bloc citait le `TRAINED_SIGMA` de `study/pipeline/config.py` (0,023), un module que `figures/v1_legacy/` n'importe pas, la ou son propre `HamiltMapper` applique le repli 0,05 de `TRAINED_PARAMS` (D-22). Corrige : le detecteur lit les **litteraux de chaine de l'AST**, f-strings comprises, chaque champ interpole rendu `{expression}` sans son format-spec — la valeur en dur se voit dans le TEXTE IMPRIME, la reecriture de guillemets n'y paraît pas. Trois assertions : aucun litteral imprime ne porte `σ`/`sigma` suivi d'un nombre ; au moins **2** litterals interpolent `sigma_trained` (le plancher mesure, sans quoi supprimer la ligne « ROOT CAUSE » suffirait a satisfaire le premier) ; et l'affectation est **structurellement** `TRAINED_PARAMS.get('sigma', 0.05)`. Les commentaires restent hors du detecteur : ils ne sont pas imprimes — c'est la lecon de D-144 prise dans l'autre sens | **A'** — le bloc CONCLUSION remis a citer `σ = 0.023`, la ligne d'affectation intacte et `{sigma_trained:` survivant dans une ligne de debogage (donc les deux chaines cherchees toujours presentes) : ancien garde **4 passed**, nouveau **1 failed, 5 passed**. **A''** — les deux `print` du bloc CONCLUSION prives de leur valeur dynamique (balayage vide) : nouveau **1 failed, 5 passed**. **B** — `TRAINED_PARAMS.get("sigma", 0.05)`, guillemets doubles, identique au bit pres : ancien **1 failed — faux rouge, le 5e de cette forme dans ce depot**, nouveau **6 passed**. Arbre sain : 4 → **6 passed**. Le detecteur porte son propre auto-test — six cas, dont `σ = 0.023` avec espaces, `~{2 * sigma_trained:.2f}` qui ne doit PAS compter comme valeur en dur, et un commentaire `# σ=0.023` qui ne doit rien declencher | `pytest tests/study/test_fig15_sigma_narration.py -q` |
| D-151 | **Le garde des lanceurs resout ses cibles contre la racine du depot meme quand le lanceur a fait `cd` ailleurs — il rougissait sur du code JUSTE, et il validait un homonyme.** `_invocations` (`tests/test_launcher_paths_resolve.py`) ne reconnaît `cd X && python Y` que sur UNE ligne ; les lanceurs de ce depot posent le `cd` a part (`run_reoptimisation.sh:69`, `run_fold.sh:27`, `run_study_v2_phases.sh:27`). Les deux sens du defaut : **faux rouge** — `cd "$ROOT_DIR/src"` puis `python train_hyperparams.py` etait resolu en `train_hyperparams.py` a la racine, absent ; **faux vert** — apres un `cd` vers un sous-dossier, un nom qui n'existe QUE a la racine etait declare sain par son homonyme, alors que le lanceur ne l'atteindrait jamais. Corrige : le dossier courant est suivi de ligne en ligne ; un `cd` non resoluble rend le dossier **inconnu** et les lignes suivantes sortent du balayage plutot que d'etre devinees ; `$(dirname "${BASH_SOURCE[0]}")` et `$(dirname "$0")` sont substitues avant `_DEF_CD`, qui butait sur leurs guillemets internes ; et un `echo` — qui n'invoque rien, il donne au lecteur une commande a taper depuis la racine — se resout contre la racine. **Le lanceur n'est pas touche : il etait juste.** | avant : **1 failed, 86 passed** (le faux rouge `run_reoptimisation.sh:72:train_hyperparams.py`). Apres : **91 passed**. Nouveau parseur + anciens tests seuls : le jeu d'invocations est **identique**, 83 avant comme apres, une seule cible deplacee (`train_hyperparams.py` -> `src/train_hyperparams.py`) et **0 cible manquante**. Les 4 tests ajoutes rejoues sur l'ancien parseur : **4 failed, 87 passed** — dont l'epinglage du faux vert, sur un lanceur temporaire qui invoque `run_tests.sh` apres un `cd` vers un sous-dossier vide (champ qui SEPARE : `run_tests.sh` n'existe qu'a la racine). **La premiere ecriture de la correction perdait 2 invocations en silence** (83 -> 81, les deux de `run_fold.sh`, dont le `cd "$root"` n'etait pas resoluble) : c'est le test de plancher qui l'a dit, et il reste | `pytest tests/test_launcher_paths_resolve.py -q` |
| D-153 | **Le garde de la famille de defauts la plus dangereuse du depot — la convention d'axes — cherche une forme avec une LISTE DE MOTS tenue a la main, ignore l'axe passe en positionnel, et ne balaie pas `tests/`.** `test_no_new_hand_rolled_curl_uses_a_bare_axis_number` (`tests/study/test_no_private_curl_survives.py`) exigeait `np.roll(` **et** `axis=` sur UNE ligne, puis decidait par une fenetre de ±4 lignes contenant l'un de `curl`, `jz =`, `j_z`, `omega_z`, `vortic`, `enstroph`. Trois trous mesures : **le nom** — un rotationnel appele `rot_z` ou `w_z` n'emploie aucun de ces mots (c'est le defaut de D-148, un detecteur de forme pilote par une liste manuelle, sur la famille qui a deja donne D-1, D-17, D-68) ; **la syntaxe** — `np.roll(By, -1, 0)` ne contient pas `axis=` et etait donc invisible, **2 sites du depot l'ecrivaient ainsi** ; **le perimetre** — `src`, `study`, `figures` seulement, alors qu'un test qui calcule sa propre reference avec le rotationnel inverse mesure autre chose que ce qu'il annonce (« mesurer avec l'operateur assorti », retourne contre la suite). Corrige par un detecteur qui ne connaît **aucun nom** : il cherche la SIGNATURE d'un rotationnel discret — une soustraction dont les deux cotes roulent DEUX tableaux differents sur DEUX axes differents — sur les cinq racines `src`/`study`/`figures`/`tests`/`scripts`. Un laplacien (meme tableau) et un gradient (un seul roll) n'y repondent pas | **etat trouve** : les 12 sites de forme rotationnelle du depot ont ete releves ; **aucune inversion vivante**. Les 2 sites a axe nu etaient dans `tests/` (`test_xpoint_at_training_resolution.py:115`, `test_patches_classical_score_provenance.py:112`), de convention CORRECTE, et sont passes a `AXIS_X`/`AXIS_Y` — 70 passed avant comme apres, aucun nombre ne bouge. **A'** — `rot_z = (roll(vy,-1,1) - vy)/dx - (roll(vx,-1,0) - vx)/dx` ajoute a `study/common/metrics.py`, rotationnel INVERSE sans aucun mot-cle : ancien garde **26 passed**, nouveau **1 failed, 29 passed**. **B** — le meme rotationnel ecrit avec `g.AXIS_X`/`g.AXIS_Y` sur deux lignes (reecriture equivalente) : **30 passed**, pas de faux rouge. Arbre sain : 26 → **30 passed**. Le detecteur porte son propre auto-test — six cas, dont le laplacien et la divergence partielle qui doivent rester MUETS, sans quoi le balayage crierait sur tout le solveur | `pytest tests/study/test_no_private_curl_survives.py -q` |
| D-154 | **Le garde ecrit pour qu'un fichier deplace se voie sans lancer la suite regardait 3 des 480 imports internes de `tests/` — et un module de `src/` deplace fait disparaitre 45 tests en SILENCE.** `test_every_cross_test_import_resolves` (`tests/test_suite_integrity.py`) ne retenait que les imports dont le module commence par `tests.` : **3 sites**, sur 1347 sites d'import dont **480 designent un module du depot** (`Simulation`, `VQA`, `pipeline`, `train_hyperparams`, `config`, `fig_utils`...) et **381 sont ecrits dans le corps d'une fonction ou d'une fixture** — la position meme qui rend l'echec invisible a `--collect-only`, et la raison d'etre du fichier. Le silence vient de la forme employee par ces fixtures : **15 `pytest.importorskip`, dont 14 nomment un module du depot**, et `importorskip` transforme un fichier absent en `skip`, pas en echec. Corrige : le balayage prend tous les `import` / `from … import` **et** le nom litteral passe a `importorskip`, a quelque profondeur qu'il soit, et resout chaque module du depot contre les racines que la suite pose reellement (les 10 de `conftest.py`, plus `figures/` et `figures/v1_legacy/` que les fichiers de test posent eux-memes). Resolution ecrite a la main plutot que par `find_spec`, qui IMPORTE les paquets parents : on mesurerait alors l'environnement et non l'emplacement des fichiers — l'operateur assorti. **Les 14 `importorskip` ne sont pas touches** : leur raison est legitime (`analyze_hyperparams` importe `optuna`) ; c'est le garde qui devient capable de rougir | **consequence mesuree** — trois modules de `src/` renommes (`analyze_hyperparams`, `recompute_lambda_scores`, `compare_rotor_budget`), sur les 7 fichiers qui les emploient : **62 passed → 6 failed, 11 passed, 45 SKIPPED**. **A'-1** deplacement (`src/analyze_hyperparams.py` → `src/tools/`) : ancien garde **158 passed — vert**, nouveau **1 failed, 163 passed**. **A'-2** les trois renommages : ancien **158 passed — vert**, nouveau **1 failed, 163 passed**. **B** reecriture equivalente (`import X as y`, `from X import y`, `importorskip` dans une fonction) : **165 passed**, pas de faux rouge. Arbre sain : 158 → **164 passed**. Le resolveur porte son auto-test — fichier deplace dans un sous-dossier, paquet d'espace de noms sans `__init__.py` (`study/` en est un), module absent. *(La ligne de reservation annoncait « 3 des 352 » : premiere mesure sous-comptee, elle ignorait les modules poses sur `sys.path` par `conftest.py` depuis `study/*`. Le compte juste est 480.)* | `pytest tests/test_suite_integrity.py -q` |
| D-155 | **L'inventaire de la dette psi — un des trois encodages du modele — compte la PRESENCE du mot-cle `with_psi`, pas sa valeur : le seul script declare cable peut le passer a `False` en dur sans qu'aucun test ne bouge.** `_passes_with_psi` (`tests/study/test_psi_coverage_inventory.py`) faisait `any(kw.arg == "with_psi" ...)`. C'est la forme exacte de `assert len(params) == 4` : l'assertion porte sur la presence d'un mot-cle, pas sur la garantie annoncee — « ce script rebranche psi ». Second trou, meme fichier : `_callers()` ne reconnaissait l'appel que sous le nom `prepare_qaoa_inputs`, donc un `from … import prepare_qaoa_inputs as prep` faisait sortir un script de l'inventaire **en silence**, alors que l'inventaire existe precisement pour que rien n'en sorte sans etre dit. Corrige : `_etat_psi` rend `absent` / `faux` / `cable` en lisant la VALEUR du mot-cle ; les alias locaux sont suivis ; `**kwargs` **leve** plutot que de conclure — un balayage statique ne voit pas a travers, et un balayage qui ne voit pas doit crier (lecon de D-145) ; et un script declare cable doit aussi passer un `prev_fields` non nul, sans quoi `prepare_qaoa_inputs` leve. Le perimetre a ete verifie et **il est sain** : les 7 appelants sont tous dans `study/`, aucun dans `src/`, `figures/` ou `scripts/` | **A'-1** — `h0_optimiser_equivalence.py`, le seul script declare cable, mis a `with_psi=False` en dur (psi mort dans TOUT `study/`) : ancien garde **4 passed — vert**, nouveau **1 failed, 5 passed**. **A'-2** — un appel par alias (`prep(..., with_psi=True, prev_fields=prev)`) ajoute a `h1_solver_convergence.py`, script hors inventaire : ancien **4 passed — vert**, nouveau **1 failed, 5 passed**. **B** — l'ordre des mots-cles change dans l'appel reel (reecriture equivalente) : **6 passed**, pas de faux rouge. Arbre sain : 4 → **6 passed**. Plancher ecrit aux valeurs mesurees : 66 fichiers de `study/`, **7** appelants | `pytest tests/study/test_psi_coverage_inventory.py -q` |
| D-156 | **Le garde de D-140 aplatit le document entier pour recoller les spans de code inline — et une cloture ``` est faite de backquotes, elle aussi. L'appariement des backquotes se decale donc sur tout le reste du document : des commandes sont avalees, d'autres recoivent les options d'une commande voisine.** `_commands_with_options` (`tests/study/test_repro_commands_point_to_real_files.py`) faisait `re.sub(r"`([^`]*)`", …)` sur `RESULTS.md` en entier. Le motif appariait le DERNIER backquote d'une cloture ouvrante avec le PREMIER de la cloture fermante : le bloc devenait une ligne, `_PY_CMD_RE` — dont le groupe d'options consomme `[^\n`|]*` — laissait la premiere commande avaler les suivantes, et `finditer` reprenait apres, donc les suivantes n'etaient jamais vues. Corrige : le bloc cloture se lit **ligne par ligne**, avec ses continuations `\` et sans son commentaire de fin de ligne ; le span inline se recolle toujours — mais lui seul, les clotures etant neutralisees d'abord. Deux autres trous fermes au passage : le **perimetre** — la verification des options ne lisait que `RESULTS.md`, alors que les commandes qu'on va retaper vivent aussi dans `MODE_EMPLOI_CAMPAGNE.md` (le mode d'emploi de la campagne de ~224 h CPU), `BRIEF_REPRISE.md`, `DEFAUTS.md` et `README.md` ; et une **question 4** — le test des options passait son chemin sur un script absent en se disant « couvert par le test des chemins », mais `_PATH` exclut `src/` : les 4 commandes `python src/*.py` de `RESULTS.md` n'etaient couvertes ni par l'un ni par l'autre | **A' (faux rouge, le 6e de cette forme ici)** — une seconde commande, autre script, ajoutee a un bloc existant, document parfaitement JUSTE : ancien garde **1 failed**, accusant `h4_unseen_conditions.py : --n-trials`, option qu'il ne cite nulle part ; nouveau **17 passed**. **Couverture** sur le document reel, meme texte : **15 → 16** commandes, et `h4_unseen_conditions.py` passe de `{--fold, --mode}` — tronque — a `{--fold, --matched-reference, --mode, --repeats}` : **deux options d'une commande publiee n'etaient confrontees a rien**. Les fusions `h3_size_scan` et `h4_physics_robustness` disparaissent. **B** — la meme commande reecrite sans sa continuation `\` : **17 passed**, pas de faux rouge. Arbre sain : 11 → **17 passed**. Perimetre elargi, mesure : **25 commandes a options sur les 5 documents, aucune option non declaree** — le trou etait reel et sans consequence vivante, il est ferme pendant qu'il l'est. Les 4 `python src/*.py` existent tous. Deux tests d'epinglage : un bloc a deux commandes (l'ancien comportement y est rejoue et doit rester faux), et un commentaire de fin de ligne qui ne doit pas passer pour une option | `pytest tests/study/test_repro_commands_point_to_real_files.py -q` |
| D-157 | **Le balayage COMPORTEMENTAL de D-148 — celui ecrit parce qu'« aucune recherche de forme ne ferme une famille ; seul le comportement le fait » — envoyait la meme invocation fixe aux 60 modules : 21 d'entre eux la refusaient dans argparse et le test passait sans qu'ils executent une ligne.** `--scenario no_such_scenario --N 64` etait envoye a tout module `__main__` + `argparse` de `study/`. Mesure du 18 aout 2026, code de sortie et texte releves un module a la fois : **21 des 60 sortaient en 2** sur `unrecognized arguments` ou `the following arguments are required` — les 9 `closed_loop_*` (qui prennent `--fold` / `--folds`), les 4 `h4_*`, les 3 agregateurs, les 2 `labels_*`, `preflight_coefficients`, `rho_gap_f1`, et `h1_solver_convergence` (qui declare `--scenario` mais pas `--N`). Le test n'exigeait qu'un **code non nul** : un refus du parseur le satisfaisait aussi bien qu'un garde qui crie. C'est le piege du balayage vide dans le fichier ecrit pour le fermer, et la lecon de D-140 (« le chemin existe, l'option non ») retournee contre la suite. Corrige en deux points : l'invocation est **construite a partir des options que le module declare**, lues a son propre `--help` — l'operateur assorti ; et une mort dans argparse est desormais un **echec**, pas une reussite. Six modules restent hors couverture, chacun avec sa raison ecrite : les 3 agregateurs et `closed_loop_status` (D-158, en attente de decision — les lancer detruirait des artefacts publies), `preflight_coefficients` et `rho_gap_f1` (aucun selecteur de donnees dans leur CLI, donc aucune demande ne peut leur etre faite qui ne corresponde a rien) | **avant** : 21 des 60 modules lances mouraient dans argparse ; le fichier etait **vert**. **A'** — le nouveau garde rejoue avec l'ancienne invocation fixe : **15 failed, 39 passed** (15 et non 21, les 6 exemptes n'etant plus lances). **Arbre sain, invocation construite** : **126 passed, 7 skipped** en 4 min 07 (contre ~2 min : le `--help` de chaque module double la duree, c'est le prix de l'operateur assorti). Couverture ecrite au plancher mesure : 61 modules lancables, 6 exemptes, **55 couverts**. Le selecteur porte son auto-test — six cas, dont « aucun selecteur declare » qui doit rendre `None` et faire crier l'appelant plutot que deviner | `pytest tests/study/test_empty_sweep_never_silent.py -q` |
| D-159 | **Les deux tests qui doivent garantir que « couvert » veut dire « un test le vise vraiment » ne peuvent pas echouer : l'inventaire qui DECLARE les noms est lui-meme dans le corpus qu'ils fouillent.** `_test_corpus` (`tests/pipeline/test_src_coverage_inventory.py`) concatenait le TEXTE de tous les fichiers de `tests/` — y compris celui-la. `test_each_covered_module_is_named_by_the_test_suite` cherchait `\bgrid\b` et le trouvait dans sa propre entree `"Simulation/grid.py"` de `COVERED` ; `test_the_public_surface_of_the_physics_path_is_exercised` cherchait `\bproject_divergence_free\b` et le trouvait dans son propre dictionnaire `critical`. **Les deux etaient structurellement incapables d'echouer**, dans le fichier dont la docstring annonce « echoue quand un nouveau module apparaît sans entree ». Deux corrections, et la seconde est la vraie : ce fichier **sort du corpus**, et le corpus n'est plus du texte mais les IDENTIFIANTS lus dans l'**AST** des autres tests — noms, attributs, modules importes, et litteraux de chaine qui sont des identifiants valides (`getattr(mod, "compute_coefficients")`, `monkeypatch.setattr(m, "score", …)` sont de vraies references). Un nom cite en commentaire ou en prose de docstring ne compte plus : « l'assertion porte sur le comportement, pas sur le texte du source », retournee contre la suite. 4e instance de cette forme dans ce depot apres D-146, D-147, D-150 — mais la premiere ou le texte cherche est **produit par le chercheur** | **A'** — `src/Simulation/zzz_untested.py`, module neuf portant une fonction qu'aucun test ne nomme nulle part, ajoute a `COVERED` **et** a `critical` : ancien garde **102 passed — vert**, nouveau **2 failed, 102 passed** (le module ET la fonction critique). Arbre sain : 98 → **100 passed**. **Pas de faux rouge** : les 19 modules de `COVERED` et les 49 fonctions critiques sont tous retrouves par l'AST — verifie avant d'ecrire la correction, 4531 identifiants collectes. Deux tests d'epinglage : un temoin (`zzz_temoin_d159`) qui n'existe que dans ce fichier et **ne doit pas** etre vu — s'il l'est, le fichier est revenu dans son propre corpus ; et un banc qui verifie qu'un nom en commentaire et un nom en prose de docstring ne comptent pas, la ou un import, un attribut et une chaine-identifiant comptent. Plancher ecrit a la valeur mesuree : 3000 identifiants | `pytest tests/pipeline/test_src_coverage_inventory.py -q` |
| D-160 | **Deux trous qui composent : une commande citee AU FIL DU TEXTE n'est vue par aucun garde d'existence, et l'exemption « historique » du prefixe mort COMPTE les lignes au lieu de les identifier — remplacer une mention narrative par une commande morte reste vert.** (1) `_INLINE_CMD_RE` exigeait que la ligne soit ENTIEREMENT un span de code **et** que le chemin y vienne en premier : une commande citee au milieu d'une phrase et prefixee de `python` n'y repond ni par sa position ni par son prefixe. Question 4 — `_commands_with_options` (D-140/D-156) voit ces commandes et verifie leurs OPTIONS, le balayage des chemins ne les voit pas : deux balayages du meme document, l'un aveugle a ce que l'autre lit. (2) `test_no_dead_v4_prefix_outside_documented_history` faisait `allowed = sum(1 for f, _frag in _HISTORICAL_EXCEPTIONS if f == relpath)` : le fragment n'etait **jamais confronte au fichier** — et n'y figurait pas, `"d71-entry"` et `"trap-sweep"` n'existent dans aucun des deux documents. Le test disait « pas plus de deux occurrences », pas « ces deux occurrences-la » : la forme de D-136. Corrige : un motif qui reconnaît la commande a son prefixe `python`/`bash` ou qu'elle soit dans la ligne — et qui ne peut pas attraper une citation narrative, qui n'en porte jamais ; et chaque exemption porte un fragment qui doit se trouver SUR LA LIGNE citee, plus un controle que chaque fragment declare designe encore une ligne (une exemption qui pourrit autoriserait demain ce qu'elle n'a jamais decrit). Les 5 fragments sont passes de jetons inventes a des ancres reelles | **couverture** sur le meme texte : **16 → 23** chemins vus, 7 fichiers entrant dans le balayage (`preflight_coefficients.py` — la porte de la campagne —, `rho_gap_f1.py`, `h2b_loso_delta_ci.py`, `pareto_frontier.py`, `fig0_pareto_lambda.py`, `repetition_campagne.sh`, `run_study_v2_phases.sh`). Tous vivants : le trou etait reel et sans consequence, il est ferme pendant qu'il l'est. **A'** — la mention narrative `run_arm` remplacee, en prose, par une commande citant un script de l'ancienne arborescence (le prefixe mort suivi d'un nom de script), **le compte d'occurrences inchange** : ancien garde **7 passed — vert**, nouveau **3 failed** (le chemin mort, l'option non declaree, et la ligne non documentee). **B** — une citation narrative suivie d'un numero de ligne ne doit PAS etre prise pour une commande : verifiee muette. Arbre sain : 17 → **19 passed**. **Le garde a mordu sur son propre registre** des la premiere execution : la ligne de reservation de D-160 citait un chemin d'exemple inexistant, et les trois tests l'ont signale | `pytest tests/study/test_repro_commands_point_to_real_files.py -q` |
| D-161 | **Le controle de peremption des exemptions de fuite d'identifiants ne peut pas echouer — et 2 de ses 4 exemptions etaient deja mortes.** `test_every_documented_placeholder_still_exists` faisait `couple in tout`, ou `tout` est le texte de tous les fichiers balayes — **y compris le fichier qui DECLARE les couples**. Les 4 cles y figurent litteralement (`"user:pass"`, `"user:pw"`, `"utilisateur:motdepasse"`, `"user:secret"`) : le controle trouvait toujours sa propre declaration. Cas pur de la quatrieme question de `COUVERTURE.md` — *le balayage figure-t-il dans ce qu'il balaie ?* — deuxieme instance apres D-159. Second ecart, celui qui l'a rendu inoffensif en apparence : il mesurait la presence du couple avec `in` sur du **texte brut**, alors que l'exemption est consommee sur une URL reconnue par `_URL_WITH_PASSWORD` — **l'operateur assorti**, applique a une grandeur textuelle. Consequence mesuree, chaque exemption retiree une a une : `user:pass` porte 1 URL (README.md), `user:pw` en porte 1 (`docs/MODE_EMPLOI_CAMPAGNE.md`), **`utilisateur:motdepasse` et `user:secret` n'en portent aucune** — la premiere n'apparaît qu'en prose, la seconde qu'en morceaux concatenes que le motif ne reconnaît pas. Ce n'etaient pas des exemptions : c'etaient des permissions dormantes, accordees d'avance a tout secret portant ces couples, dans un depot **public**. Corrige : `_porteurs_reels` mesure avec `_URL_WITH_PASSWORD` et **exclut ce fichier du corpus** ; les 2 entrees mortes sont retirees | **A'** — le modele `user:pass` retire du README (l'exemption devient morte, exactement ce que le controle doit crier) : ancien garde **8 passed — vert**, nouveau **2 failed**. **B** — le meme modele **deplace** vers un autre document, reecriture equivalente : **10 passed**, pas de faux rouge. Arbre sain : 8 → **10 passed**. Plus un temoin qui verifie que `_porteurs_reels` peut rendre la liste **vide** — sans quoi l'assertion ne prouverait rien | `pytest tests/pipeline/test_no_credential_in_source.py -q` |
| D-162 | **`test_every_entry_point_guards_its_main` ne verifie pas le bloc `__main__` dont il porte le nom — et un module de bibliotheque du chemin de decision deploye etait range dans `ENTRY_POINTS`, la categorie qui n'exige rien.** Question 2 : la docstring dit « un script sans `if __name__ == '__main__'` s'execute a l'import », le test ne cherchait que du **travail au niveau module**. Un module qui n'a aucun bloc `__main__` le passe donc trivialement. Le module en cause, mesure au 18 aout 2026 : **`call_vqa_shell.py`** — une seule fonction, aucun bloc `__main__`, importee par `Simulation/refinement.py` (le chemin de decision deploye), `compare_rotor_budget.py` et deux figures, et **importee et appelee par 5 fichiers de `tests/quantum/`**. Ce n'est pas un point d'entree. La categorie est une trappe : elle dispense de `test_every_module_imports_cleanly`, en echange d'un `ast.parse` — et **19 des 19 modules de `COVERED` passaient tels quels son unique controle**, donc n'importe lequel pouvait y etre gare sans qu'un test ne bouge. Corrige : l'assertion du bloc `__main__` est ajoutee (lue dans l'AST, pas dans le texte — une chaîne `"__main__"` dans un message ne compte pas), et `call_vqa_shell.py` passe dans `COVERED`, ou il gagne l'import propre et le controle « importe par la suite » de D-164 | **A'** — la classification d'origine (`call_vqa_shell.py` dans `ENTRY_POINTS`) rejouee : ancien garde **100 passed — vert**, nouveau **1 failed**, nommement `test_every_entry_point_guards_its_main[call_vqa_shell.py]`. Arbre sain 103 → **105**, justifie ligne a ligne par le diff des identifiants collectes : **+4** (`test_every_module_imports_cleanly`, `test_each_covered_module_is_named_by_the_test_suite`, `test_no_module_defines_the_same_constant_twice` sur `call_vqa_shell.py`, plus l'epinglage `test_le_controle_du_bloc_main_peut_echouer`), **−2** (les deux controles d'entree qu'il quitte). **Une hypothese refusee a la mesure** : soumettre aussi `ENTRY_POINTS` au controle « nomme par la suite » semblait fermer la trappe — sous le critere de D-164 (un import REEL), **3 des 4 pilotes ne sont pas importes** par `tests/`, exactement ce que `test_every_entry_point_parses` annonce (« les pilotes sont lourds a importer »). Cela aurait fabrique **3 faux rouges sur du code sain** : non applique, raison ecrite dans le fichier | `pytest tests/pipeline/test_src_coverage_inventory.py -q` |
| D-163 | **Les deux controles de peremption d'exemption de la suite verifient l'existence du FICHIER, pas celle de la chose exemptee ni le fait qu'elle supprime encore quelque chose.** Meme forme que D-161, sur deux autres inventaires. (1) `test_chaque_exemption_porte_sa_raison_et_existe_encore` (`tests/test_suite_integrity.py`) : `SANS_ASSERTION_LEGITIMES` designe des **fonctions** (`fichier::nom`), le controle ne testait que `os.path.exists(fichier)`. Un fichier survit a la fonction qu'il portait ; une fonction peut gagner une assertion sans que son exemption parte avec — dans les deux cas l'entree devient une permission dormante, accordee d'avance a la prochaine fonction qui prendra ce nom. (2) `test_chaque_exemption_de_D148_porte_sa_raison_et_existe_encore` (`tests/study/test_empty_sweep_never_silent.py`) : ce que fait une entree de `_EXEMPTIONS`, c'est **retirer un module du balayage comportemental** — elle ne retire donc quelque chose que si le module y serait encore, c'est-a-dire s'il est dans `_LANCABLES`. Un module qui perd son `argparse` en sort tout seul, et son exemption dispenserait sans le dire le jour ou il y reviendrait. Corrige : trois criteres du plus faible au plus fort cote (1) — le fichier existe, la fonction existe, et elle serait **encore signalee** sans son exemption ; et l'appartenance a `_LANCABLES` cote (2). Etat mesure avant correction : les 3 exemptions de (1) et les 7 de (2) sont **toutes encore portantes** — le trou etait reel et sans consequence vivante, il se ferme pendant qu'il l'est | **A'-1** — la fonction exemptee **renommee**, le fichier intact : ancien garde **1 passed — vert**, nouveau **1 failed**. **A'-2** — la fonction exemptee **gagne une assertion** : ancien **1 passed — vert**, nouveau **1 failed**. **A'-3** — un module exempte perd son `argparse` et quitte `_LANCABLES` : ancien **1 passed — vert**, nouveau **1 failed**. Arbres sains : `test_suite_integrity.py` 164 → **165 passed**, `test_empty_sweep_never_silent.py` 126 → **127 passed, 7 skipped**. Chaque fichier porte en plus un temoin qui verifie que **le critere peut rendre faux** — pour (2), qu'il existe encore dans `study/` un fichier non lancable, sans quoi le controle ne distinguerait plus une exemption portante d'une exemption morte | `pytest tests/test_suite_integrity.py -q` |
| D-164 | **`COVERED` ne verifie pas la PROVENANCE de la couverture qu'il declare : un module homonyme suffit, sans aucun import reel.** `_identifiants_du_corpus()` (D-159) sert deux tests a la fois avec la meme largeur — Name/Attribute/import/chaine-identifiant. C'est voulu pour les FONCTIONS (`test_the_public_surface_of_the_physics_path_is_exercised` : `getattr(mod, "compute_coefficients")` est une reference reelle), mais un stem de MODULE (`grid`, `solver`, `execute`, `optimize`, `pipeline`…) est un mot assez commun pour apparaitre comme **attribut sans aucun rapport**. Balaye les 19 modules de `COVERED` : **5 sur 19** (`grid`, `solver`, `execute`, `optimize`, `pipeline`) restent presents dans le corpus meme apres avoir retire TOUS leurs importateurs genuins. Le cas le plus net : `VQA/optimize.py` n'a **qu'un seul** fichier de test genuin (`tests/quantum/test_vqa_chain_contracts.py`, `from VQA.optimize import optimize`) ; le retirer laisse `"optimize"` present via **11 sites** `study.optimize(objective, …)` d'Optuna (`tests/pipeline/test_train_hyperparams_*.py`) — un objet sans aucun rapport avec `VQA/optimize.py`. Corrige : `test_each_covered_module_is_named_by_the_test_suite` bascule sur `_modules_importes_du_corpus()`, qui ne compte que les stems tires d'un `import`/`from … import` **reel** — pas les `Name`/`Attribute`/chaines. `_identifiants_du_corpus()` reste inchange pour le test fonction, dont le besoin est different et deja mesure (D-159) | **arbre sain** : 100 → **103 passed** (+3 tests). **Mutation A'** — les deux imports genuins de `VQA.optimize` retires de `test_vqa_chain_contracts.py` (fichier restaure a l'identique apres mesure) : ancien garde **1 passed — vert, a tort**, `VQA/optimize.py` sans aucune couverture reelle ; nouveau garde **1 failed**. Pas de faux rouge : les 19 modules de `COVERED` retrouvent tous une preuve reelle sur le corpus etroit (mesure avant d'ecrire la correction) | `pytest tests/pipeline/test_src_coverage_inventory.py -q` |
| D-166 | **Plancher de balayage ecrit a la main qui ne detecte plus rien : `assert len(_test_files()) > 40` (deux sites, `tests/test_suite_integrity.py:67` et `:504`) contre une valeur reelle de **153**.** File ouverte par `COUVERTURE.md` (« La file suivante — les PLANCHERS de balayage ecrits a la main ») : un plancher pose loin sous la valeur reelle laisse le balayage fondre sans qu'aucun test ne le voie. Les deux sites mesurent exactement la meme quantite (`_test_files()`, le meme `os.walk` filtre), donc c'est un seul defaut duplique. A 40, la suite pouvait perdre **113 fichiers de test sur 153 (74 %)** avant que le plancher ne morde | plancher **40 → 153** (`>=`, mesure a `bfe4c46`, 18 aout 2026), sur les deux sites. **Mutation** — arbre synthetique portant 41 des 153 fichiers reels (copie, une perte de 73 %) : ancien plancher `> 40` → **41 > 40, vert a tort** ; nouveau `>= 153` → **41 >= 153, faux (rouge)**. Arbre sain : `test_suite_integrity.py` **165 passed** (inchange, la valeur reelle 153 satisfait deja le nouveau plancher) | `pytest tests/test_suite_integrity.py -q` |
| D-167 | **Plancher de balayage ecrit a la main qui ne detecte plus rien : `assert len(STUDY_FILES) > 40` (deux sites, `tests/study/test_empty_sweep_guard_shapes.py:224` et `tests/study/test_empty_sweep_never_silent.py:323`) contre une valeur reelle de **66**.** Meme famille que D-166, sur `study/` cette fois : les deux sites construisent `STUDY_FILES` avec le meme `glob.glob(os.path.join(_REPO_ROOT, "study", "**", "*.py"), recursive=True)`, donc un seul defaut duplique. A 40, le balayage de `study/` — celui qui detecte les gardes de donnees silencieuses D-56/D-75/D-148 — pouvait perdre **26 modules sur 66 (39 %)** sans qu'aucun test ne le voie | plancher **40 → 66** (`>=`, mesure a `1bee385`, 18 aout 2026), sur les deux sites. **Mutation** — 41 des 66 fichiers reels (perte de 38 %) : ancien plancher `> 40` → **41 > 40, vert a tort** ; nouveau `>= 66` → **41 >= 66, faux (rouge)**. Tests cibles inchanges : `test_the_sweep_itself_is_not_empty` et `test_the_detector_itself_can_fail`, **2 passed**. Les deux fichiers portent par ailleurs, sans rapport avec ce defaut, 34 echecs preexistants (identiques avant/apres cette correction, mesures par `git stash` — `sanity_check.py` et consorts n'exposent aucun selecteur de donnees pour `test_aucun_module_de_study_ne_sort_zero_sur_un_balayage_vide` ; hors perimetre de cette file, non touche) | `pytest tests/study/test_empty_sweep_guard_shapes.py::test_the_sweep_itself_is_not_empty tests/study/test_empty_sweep_never_silent.py::test_the_detector_itself_can_fail -q` |
| D-168 | **Plancher de balayage ecrit a la main qui ne detecte plus rien : `assert len(_modules_importes_du_corpus()) >= 50` (`tests/pipeline/test_src_coverage_inventory.py:630`) contre une valeur reelle de **130**.** Meme famille que D-166/D-167 : `_modules_importes_du_corpus()` est le corpus etroit de D-164 — celui qui distingue un import genuin d'un homonyme d'attribut. A 50, il pouvait perdre **80 stems sur 130 (62 %)** sans qu'aucun test ne le voie | plancher **50 → 130** (`>=`, mesure a `51e36ab`, 18 aout 2026). **Mutation** — 51 des 130 stems reels (perte de 61 %) : ancien plancher `>=50` → **51 >= 50, vert a tort** ; nouveau `>=130` → **51 >= 130, faux (rouge)**. Tests cibles (`corpus`/`homonyme`/`survivent`) : **5 passed**. Le fichier porte par ailleurs, sans rapport avec ce defaut, 5 echecs preexistants sur `test_every_module_imports_cleanly` (`Simulation/refinement.py`, `VQA/execute.py`, `VQA/optimize.py`, `call_vqa_shell.py`, `pipeline.py` — incompatibilite `cryptography`/`_cffi_backend` de cet environnement d'audit, identiques avant/apres cette correction, mesures par `git stash`) | `pytest tests/pipeline/test_src_coverage_inventory.py -k "corpus or homonyme or survivent" -q` |
| D-169 | **Plancher de balayage ecrit a la main qui ne detecte plus rien : `assert len(dans_src) >= 10` (`tests/pipeline/test_no_credential_in_source.py:219`, `test_src_is_still_inside_the_sweep`) contre une valeur reelle de **25**.** Ce test garde explicitement un anti-retrecissement (« le perimetre d'origine ne doit pas disparaitre en s'elargissant »), sur le sous-ensemble de `_fichiers_balayes()` situe sous `src/`. A 10, le balayage pouvait perdre **15 fichiers de src/ sur 25 (60 %)** sans que ce garde ne le voie. Le plancher voisin de la meme fonction (`lanceurs >= 8`, ligne 232) est laisse tel quel : son message cite deja « 10 mesures », la valeur reelle mesuree ici est encore 10, marge 1,25× — deja le bon patron, pas un defaut | plancher **10 → 25** (`>=`, mesure a `d816dee`, 18 aout 2026). **Mutation** — 11 des 25 fichiers reels (perte de 56 %) : ancien plancher `>=10` → **11 >= 10, vert a tort** ; nouveau `>=25` → **11 >= 25, faux (rouge)**. Arbre sain : `test_no_credential_in_source.py` **10 passed** (inchange) | `pytest tests/pipeline/test_no_credential_in_source.py -q` |
| D-170 | **Plancher de balayage ecrit a la main qui ne detecte plus rien : `assert len(fichiers) > 40` (`tests/study/test_psi_coverage_inventory.py:262`, `test_le_balayage_des_appelants_n_est_pas_vide`) contre une valeur reelle de **66**.** Un troisieme site independant de la meme quantite que D-167 (`STUDY_FILES`) : celui-ci la recalcule par `os.walk` plutot que par `glob`, mais les deux ensembles sont identiques (verifie). Le docstring citait deja « mesure du 18 aout 2026 : 66 fichiers », mais le plancher restait a 40 — pouvant perdre **26 fichiers sur 66 (39 %)** sans que ce garde ne le voie | plancher **40 → 66** (`>=`, mesure a `3d4f095`, 18 aout 2026). **Mutation** — 41 des 66 fichiers reels (perte de 38 %) : ancien plancher `>40` → **41 > 40, vert a tort** ; nouveau `>=66` → **41 >= 66, faux (rouge)**. Arbre sain : `test_psi_coverage_inventory.py` **6 passed** (inchange). Le plancher voisin de la meme fonction (`_callers() >= 7`, ligne 263) est laisse tel quel : la valeur reelle mesuree ici est encore 7, marge 1,0×, exactement le bon patron | `pytest tests/study/test_psi_coverage_inventory.py -q` |
| D-171 | **Plancher de balayage ecrit a la main qui ne detecte plus rien : `assert len(referenced) > 10` (`tests/study/test_repro_commands_point_to_real_files.py:126`, `test_every_repro_command_in_results_md_points_to_a_real_file`) contre une valeur reelle de **23**.** Compte les chemins `study/`, `scripts/`, `figures/` cites en commande dans `docs/RESULTS.md` — le garde d'existence de D-160. A 10, il pouvait perdre **13 references sur 23 (57 %)** sans que ce test ne le voie ; le vrai garde de completude (`missing == []`) reste en aval, mais le premier filet devient inutile bien avant lui | plancher **10 → 23** (`>=`, mesure a `6fa2b5d`, 18 aout 2026). **Mutation** — 11 des 23 references reelles (perte de 52 %) : ancien plancher `>10` → **11 > 10, vert a tort** ; nouveau `>=23` → **11 >= 23, faux (rouge)**. Arbre sain : `test_repro_commands_point_to_real_files.py` **19 passed** (inchange). Le plancher voisin de la meme fonction (`_test_paths_in_commands` >= 20, ligne 228) est laisse tel quel : deja date, « mesure du jour, 29 chemins », marge 1,45× a l'ecriture — matiere a une passe ulterieure, pas de defaut net | `pytest tests/study/test_repro_commands_point_to_real_files.py -q` |
| D-173 | **Un `--help` en echec etait classe en silence comme « le module ne declare aucune option » (`tests/study/test_empty_sweep_never_silent.py:249`, `_options_declarees`) — repli silencieux, pas un plancher.** `COUVERTURE.md` (18 aout) avait signale `couverts` a **49**, sous le plancher de 50 lui-meme sous les 55 du docstring, et laisse a trancher par un humain : regression reelle, ou meme defaut de perimetre qui masque le test ? Ni l'un ni l'autre. Les 5 modules « sans selecteur » (`qaoa_inputs.py`, `h3_depth_report.py`, `h3_size_scan.py`, `h3_uncertainty_window.py`, `pipeline/sanity_check.py`) importent tous `VQA.execute`, qui importe `qiskit_ibm_runtime` — absent du conteneur qui a produit la mesure de 49 comme de celle, identique, de la passe precedente : le `git stash` qui les comparait ne comparait que deux mesures egalement incompletes. Verifie en installant la dependance : `qaoa_inputs.py --help` declare bien `--scenario`, et les 5 modules passent | **61 lancables, 7 exemptes, 0 sans selecteur, 54 couverts** (mesure a `778255d`, 19 aout 2026, dependances completes) — ni 49 ni les 55 perimes du docstring. `_options_declarees` leve desormais sur tout `--help` en echec au lieu de renvoyer `None` : **mutation**, module synthetique dont l'import plante (`import ce_module_n_existe_pas`) — ancien comportement, `None` en silence (`sans_selecteur` a tort) ; nouveau, `RuntimeError` — epingle par `test_options_declarees_leve_plutot_que_de_classer_un_import_casse`. Plancher porte a `>= 54`. Arbre sain : `test_empty_sweep_never_silent.py` **128 passed, 7 skipped** (inchange, aucun module de `study/` touche) | `pytest tests/study/test_empty_sweep_never_silent.py -q` |
| D-174 | **Un test qui garantit « `state_vector` force `optimization_level=0` » ne vérifiait pas le comportement — il cherchait deux chaînes littérales dans le SOURCE de `optimize()`** (`tests/quantum/test_vqa_chain_contracts.py::test_the_state_vector_backend_silently_forces_optimisation_level_zero`, avant D-174) — exactement le patron proscrit par `VIGIL.md` : « chercher une chaîne de caractères dans un fichier teste sa mise en forme ». Trouvé en relisant `VQA/optimize.py` (module déjà « lu en entier » le 16 août, `COUVERTURE.md` §`src/VQA/`) sous l'angle configuration plutôt que fonction : `optimize()` est le chemin `else` legacy de `call_vqa_shell` (`vqa_runtime is None`), jamais atteint par `pipeline.py` (`vqa_runtime` n'est `None` qu'en `classical_only`, où le VQA n'est pas appelé) ni par aucun des sept sites d'appel de `call_vqa_shell` dans `tests/` (tous passent `vqa_runtime=...`) — sa seule couverture réelle est ce test unique, sur du texte. **Mutation, avant correction** : rendre le forçage mort (`if False:` autour de `opt_level = 0`, chaînes inchangées) laisse le SOURCE identique aux deux motifs cherchés — **1 passed, vert à tort** — alors qu'un appelant demandant `opt_level=3` sur `state_vector` obtiendrait désormais 3 au lieu de 0 | Test remplacé par un appel réel à `optimize()`, `generate_preset_pass_manager` substitué pour capturer l'`optimization_level` RÉELLEMENT transmis : `state_vector` → **0** quel que soit le niveau demandé (rejoué sur la même mutation : **1 failed**, la correction attrape ce que l'ancien test manquait) ; `aer` → niveau demandé **inchangé** (**3**) — champ qui sépare les deux comportements, ajouté en test compagnon. `pytest tests/quantum/test_vqa_chain_contracts.py -q` : **41 passed** avant → **42 passed** après (une mutation d'assertion de source remplacée par deux assertions de comportement) | `pytest tests/quantum/test_vqa_chain_contracts.py -q -k optimisation_level` |
| *(ex-D-132, numéro EN ATTENTE)* | **⚠️ Je reviens sur ma propre décision d'il y a quelques commits — je l'avais numérotée `D-176` avant de trouver, plus bas dans ce même fichier (§ « Compte de tête inexact »), qu'une passe du 17 août avait déjà posé exactement cette question et l'avait explicitement laissée à USER : *« Aucun des deux ne peut être renuméroté par la règle de la fiche … C'est une décision, pas une correction. »* Je n'avais pas lu cette note avant de trancher — erreur de méthode, `VIGIL.md` exige de lire le registre en entier avant de conclure qu'un cas est neuf. Je retire le numéro `D-176` (personne d'autre ne l'a encore cité ni ne peut donc être induit en erreur) et laisse la question de numérotation ouverte, comme elle l'était.** Ce qui reste vrai et que je garde : le **contenu** ci-dessous satisfait indépendamment la règle d'arrêt de `DEFAUTS.md` — il ne bloque plus la réoptimisation (constaté par l'entrée elle-même) et ne porte aucune lecture publiée — donc il sort de `DEFAUTS.md` vers `RESULTS.md` comme n'importe quel autre défaut élucidé, **sans préjuger du numéro qu'il devra porter**. Contenu, inchangé depuis `DEFAUTS.md` : une campagne Optuna avait révélé, sur 12 combinaisons d'hyperparamètres, une corrélation de rang QAOA/vérité-terrain **négative** (-0,467 sur 3/12) et un écart Orszag-Tang **exactement nul** entre bras quantique et classique — signature d'un hamiltonien devenu inerte sur cette partie de l'espace. Bisection commit-par-commit (déjà close par une passe antérieure) entre `d978539` (garde, passe) et `5bdcf80` (passe→échoue puis reste rouge) : un seul commit sépare le dernier vert du premier rouge — `e4d6bbc` (passe, 9 min 06) puis `6ecaecf` (échoue, -0,467, 2 min 20), diff de 39 lignes dans `src/Simulation/solver.py`, rien d'autre. `6ecaecf` **est** D-25 : la projection spectrale du second membre de B, qui corrige un défaut réel (`div_FD B` 4,63e-07 → 1,00e-14, voir table « Numérique et rééchantillonnage » ci-dessus) et ajoute son propre contrôle négatif. Trois hypothèses de cause antérieures, réfutées par la bisection : D-1 (convention de rotationnel, `bb6a387`), D-8 (coefficients nuls, `d212e54`), D-37 (biais Z/couplages sur deux grilles, `91951df` — écarté car postérieur à `854ba24`, déjà rouge) | **avant D-25** (artefact numérique) : le bras QAOA « classait bien » en lisant des champs abîmés par une projection fautive, ordre artificiel. **après D-25** (physique correcte) : classement hétérogène et instable selon les hyperparamètres, −0,467 à +0,95 selon la combinaison — ce n'est pas une régression à défaire, c'est la campagne (D-22) qui doit l'arbitrer. `test_hyperparameter_sweep` et `test_noise_robustness` restent rouges aujourd'hui, comme seuils périmés sur l'état d'avant D-25 : ils font partie du jeu connu de tests intermittents que D-165 documente (sampler/estimateur non graine dans `src/VQA/`, qui les fait entrer et sortir de la suite complète selon le tirage) et se remesurent avec les autres seuils périmés une fois la campagne D-22 tranchée — ni retouchés ni supprimés d'ici là | `git log --oneline d978539..5bdcf80 -- src/Simulation/solver.py` (la bisection) ; `pytest tests/quantum -k "hyperparameter_sweep or noise_robustness"` (jeu connu, intermittent — voir D-165) |
| D-176 | **`PhysicalMapper.physical_score` (`Simulation/HamiltParams.py:296`) annonçait dans sa docstring et dans l'en-tête de sa section « replaces classical_score for θ initialization » — un rôle qu'elle n'a jamais tenu dans le pipeline déployé.** Trouvé en lisant `HamiltParams.py` en entier (question 2 de `VIGIL.md`, la docstring comme contrat) puis en vérifiant : `grep -rn "\.physical_score(" src/ study/` — **0 site** hors la définition et les tests. Le θ-init déployé vient partout de `AngleMapper.classical_score` (`refinement.py:669`, `qaoa_inputs.py`), jamais de `PhysicalMapper.physical_score`. Ce n'est pas D-9 (ci-dessus) : D-9 corrigeait le seul **appelant** qui lisait le mauvais score dans un script d'ablation (`h3_uncertainty_window.py`) ; cette entrée corrige la **docstring de la fonction elle-même**, qui continuait d'affirmer un rôle de déploiement après que D-9 a établi qu'aucun appelant réel ne le lui donnait. `physical_score` reste appelée par une trentaine de sites de `tests/` comme formule alternative comparée à `classical_score` — ni du code mort au sens du bytecode, ni du code vivant au sens du pipeline | docstring et en-tête de section corrigés pour dire ce que la fonction est réellement : une formule alternative testée, jamais câblée en production. **Comportement inchangé** : aucune ligne de calcul touchée, seule la prose au-dessus. `pytest tests/solver/test_analytic_fields.py -q` : 53 passed avant et après ; `pytest tests/pipeline/test_v9_metrics.py tests/mapping/test_mapper_contracts.py -q` : 66 passed avant et après | `grep -c "physical_score(" src/*.py src/Simulation/*.py study/**/*.py 2>/dev/null` (hors définition) |
| D-175 | **`curl_z`/`divergence` (`Simulation/grid.py:76-85`) annoncent dans leur docstring « forme historique par defaut, forme 'ij' si demande » — l'inverse du comportement mesure.** `fixed_curl=True` est le defaut des deux signatures, et `if fixed_curl` renvoie `forward_curl_z`/`forward_divergence` (la forme AXIS_X/AXIS_Y du depot) — la forme `legacy_*` (historique) n'est rendue que si l'appelant passe explicitement `fixed_curl=False`. Mesure : `curl_z(vx, vy)` sans 3e argument, champ aleatoire 8×8, `seed=0` — identique bit-a-bit a `forward_curl_z`, **max\|defaut − legacy\| = 1,1006**. Meme mesure pour `divergence`. **Aucun appelant reel n'est trompe** : les 7 sites de `src/` passent tous `self.fixed_curl`/`fixed_curl` explicitement (grep verifie), et `tests/solver/test_analytic_fields.py:445-447` epingle deja `curl_z(vx, vy) == forward_curl_z(vx, vy)` sur l'appel sans argument — c'est le texte de la docstring qui contredisait un comportement deja teste, pas le comportement qui etait faux. Hors chemin critique (aucune valeur, aucun appelant, aucun nombre publie ne bouge) : une ligne ici, pas d'entree `DEFAUTS.md` | docstrings corrigees pour dire le vrai defaut (`fixed_curl=True` par defaut → forme `forward_*`/'ij' ; `fixed_curl=False` → forme historique/legacy). Comportement **inchange** : `curl_z(vx,vy)` et `divergence(vx,vy)` restent identiques a `forward_*` avant comme apres (meme mesure, 0,0e+00 d'ecart) | `pytest tests/solver/test_analytic_fields.py -q -k "curl_z or divergence or fixed_curl"` |
| *(note, non instruite)* | **Trois `main()` de `figures/v1_legacy/` sortent avec le code 0 sur un balayage vide, hors du perimetre du garde de D-148.** `fig11_hamiltonian_design.py:79`, `fig12_depth_analysis.py:213`, `fig13_sigma_ablation.py:101` : `if n_scenarios == 0: print("… skipping figXX"); return`. Le detecteur de `tests/study/test_empty_sweep_guard_shapes.py` les trouve — il ne balaie que `study/`. **Moins grave que les six de D-148** : ils IMPRIMENT leur abandon, donc un lecteur du journal le voit ; c'est le compteur du lanceur qui ne le voit pas (`SUCCEEDED++`), la forme meme de D-116. `SCENARIOS` ne peut etre vide que sous un `FIGURE_PHASE` dont `_PHASE_SCENARIOS` retire tout, et `fig_utils` imprime deja un `[WARN]` dans ce cas. Defaut ou choix : mesure faite, correction non appliquee, decision a USER | mesure : detecteur de D-148 applique a `src/` et `figures/` — **3 sites**, tous dans `figures/v1_legacy/`, **0 dans `src/`** | `python -c "import sys;sys.path.insert(0,'.');"` puis le detecteur `silent_data_guards` de `tests/study/test_empty_sweep_guard_shapes.py` |
| *(verifie sain)* | **Le piege « `\| tail` masque le code de retour » n'est vivant dans aucun lanceur.** `BRIEF_REPRISE.md` §10 le donne comme piege d'environnement, et `scripts/generate_figures_v1.sh:253` fait bien `if python "$script_path" 2>&1 \| tail -5; then` — mais le fichier pose `set -o pipefail` ligne 3, donc le code de python l'emporte. Hypothese posee, mesuree, **refutee**. Les 10 lanceurs balayes : les 2 qui pipent vers `tail` ont `pipefail` ; les 2 sans `pipefail` (`run_fold.sh`, `run_leak_free_campaign.sh`) n'ont aucun pipe | `bash -c 'faux(){ echo boom; exit 3; }; if faux \| tail -5; then echo SUCCES; fi'` rend SUCCES sans `pipefail`, et rien avec | `grep -c pipefail scripts/*.sh run_tests.sh` |
| *(verifie sain)* | **Les 12 cibles `pytest` distinctes de `run_tests.sh` collectent toutes au moins un test.** Le garde de collecte de `tests/lint/test_scripts_point_somewhere.py` ne couvre que les cibles construites sous `$ROOT_DIR/` dans `scripts/*.sh` ; celles de `run_tests.sh` sont relatives et a la racine, donc hors perimetre deux fois. Mesure directe : **18, 30, 1, 7, 98, 14, 12, 2, 26, 2, 54, 44** tests collectes — **aucun zero**. Le trou de perimetre est reel, sa consequence ne l'est pas aujourd'hui | 12 cibles, 0 vide | `python -m pytest --collect-only -q <cible>` pour chacune |
| D-77 | **Rapport seul, rien n'est corrigé — et il touche une ligne de `COUVERTURE.md` déjà déclarée « saine ».** Le champ `classical_scores` des artefacts `patches_*.npz` fige la convention de rotationnel d'**avant D-1**. D-1 (`bb6a387`, 11 août) a basculé `fixed_curl` sur la convention déclarée par `grid.py` — sur une rotation solide, l'ancienne forme rend 0 là où la vraie rend +2 — et `classical_score` en dépend par deux de ses quatre indicateurs (vorticité, divergence). Les artefacts écrits avant cette date n'ont jamais été regénérés. `COUVERTURE.md` déclarait ce chemin sain — *« pas d'opérateur dépareillé entre le score classique des artefacts `patches_*` et celui du chemin coefficients »* — ce qui est vrai du **code** (le stencil `Jz` écrit à la main est bien identique à `solver.get_fluxes`) et faux de **84 fichiers sur 156** | balayage des 156 artefacts, dernier instantané de chacun, score recalculé par le chemin du dépôt : **72 fichiers** (dim 2/4/8, commit du 11 août) reproduisent à **0,000e+00** ; **84** (dim 3/16/32/64 et toutes les variantes de label, commits du 9 août) ne reproduisent pas — écart jusqu'à **3,8e−01** sur un score borné à [0, 1] — et **50 d'entre eux sont reproduits bit à bit par `fixed_curl=False`** (≤ 1e−12) : la cause est isolée, pas seulement constatée. **Les labels du même fichier se reproduisent** (`l2_errors` 9,4e−12, plancher float32 ; `is_hard` 0 désaccord / 5120 ; `l2_threshold` exact) — ils ne dépendent d'aucun rotationnel. Contradiction interne la plus nette, mesurable sans rien recalculer : **4 paires de fichiers décrivent la même configuration** (`Re400 N256 dim4`, base regénérée le 11 contre variante `_globalthr` du 9) **et portent deux scores différents** — écarts **0,169 / 0,179 / 0,256 / 0,409**. **Aucun nombre publié ne bouge** : le seul consommateur du champ, `pipeline_verification`, ne tourne qu'à `dim=4`, où le fichier de base est à jour — sortie complète identique avant/après regénération, vérifiée (F1 hamiltonien 0,729 vs classique 0,654, verdict PASS inchangé). Le piège est pour le consommateur suivant. Non corrigé : regénérer changerait des artefacts publiés — décision de USER. La déviation est écrite dans `hard_patch_labels.patch_classical_scores`, là où elle vit | `pytest tests/study/test_patches_classical_score_provenance.py` |
| D-78 | **Rapport seul.** `results/t29_loso_delta_ci_N256_perscenario.npz` (T29, commit du 9 août) ne se reproduit plus à HEAD, et **le seul verdict tranché qu'il porte tombe**. L'artefact conclut `4:nuisent` — *les voisins nuisent à dim = 4* ; rejoué à HEAD par sa propre commande, le verdict devient **indécidable**, et le fold `mhd_rotor` **change de signe** avec des IC95 qui ne se recouvrent pas : **−0,212 [−0,310, −0,100] → +0,111 [+0,062, +0,158]**. Même famille que D-69 (T31), sur une autre tâche. **Aucun document du dépôt ne mentionne T29** — ni `RESULTS.md`, ni `COUVERTURE.md`, ni `DEFAUTS.md` (recherche exacte) : rien n'avertissait qu'un `.npz` de `results/` portait des verdicts d'avant les corrections | rejoué `--dim 4 --bootstrap 500`, sortie **hors du dépôt**, l'artefact publié n'est pas écrasé. **Reproductibilité de la mesure vérifiée avant de conclure** : le rejeu à HEAD lancé **deux fois** rend des nombres identiques à la dernière décimale — l'écart au publié n'est pas de la variance d'exécution. **Cause partiellement isolée, et le partiellement est dit** : en substituant le `src/` d'avant D-1 (`bb6a387^`), le bras **classique** revient exactement au publié sur **3 folds sur 4** (`F1_cls` 0,400 / 0,400 / 0,833, contre 0,400 / 0,400 / **0,636** à HEAD) — c'est bien la convention de rotationnel de D-1, cohérent avec D-77. Les bras **appris**, eux, ne reviennent pas (`mhd_rotor` `F1_site` : publié 0,229, pré-D-1 **0,020**, HEAD 0,005) : ce n'est donc pas ce code-là, et `environment.yaml` ne fixant aucune version, ce conteneur ne peut pas trancher entre dérive de `scikit-learn` et autre chose — même réserve que D-69. **Ce qui ne dépend d'aucune de ces réserves** : les deux rejeux, pré-D-1 et HEAD, rendent le même verdict **indécidable** là où le publié tranche. Rien n'est corrigé ni regénéré : refaire la table est une campagne, donc une décision de USER | `python study/h2b_prediction/h2b_loso_delta_ci.py --dim 4 --bootstrap 500` puis comparer aux clés `delta`, `ci_low`, `ci_high`, `verdict_by_dim` de `results/t29_loso_delta_ci_N256_perscenario.npz` |
| D-79 | `verdict()` de T29 (`study/h2b_prediction/h2b_loso_delta_ci.py`) décide **qui vote**, et la quantité votée est `F1(stencil) − F1(site)`. Le prédicteur **classique** n'y entre pas — il est calculé et imprimé pour situer les deux modèles, rien de plus. Il figurait pourtant dans la même liste `constant` que les deux bras comparés, et cette liste écarte les folds : un fold dont les **deux** modèles comparés étaient sains pouvait être jeté parce qu'un **troisième** prédicteur, étranger à la comparaison, était constant. Défaut contre sa propre docstring, qui donne la raison de la règle — « leur F1 ne mesure pas un modèle » — vraie du bras comparé, fausse du bras classique | mesuré sur le rejeu de `--dim 4 --bootstrap 500` avec le `src/` d'avant D-1, la configuration qui a produit l'artefact publié : `kelvin_helmholtz` **écarté** (`constant: cls`) alors que ses deux bras comparés sont sains et que son IC95 **exclut zéro** — Δ **−0,027 [−0,050, −0,001]**, donc un fold qui tranchait. `harris_tearing` écarté pour la même raison. Avant : *« folds retenus : 2/4 »*. Après : *« 4/4 »*. **Le verdict de cette configuration ne change pas** (« indécidable » : les folds retenus ne s'accordent pas) — la correction rend au vote un fold décisif, elle n'invente pas une conclusion, et c'est dit plutôt qu'embelli. L'effondrement classique reste **imprimé et écrit** dans l'artefact (`constant`), il ne décide simplement plus du vote (`constant_compared`). 3 tests échouent sur l'ancienne version, dont deux sur des cas construits pour **séparer** les deux règles dans les deux sens — l'un rend un verdict, l'autre en retire un | `pytest tests/study/test_t29_verdict_excludes_only_the_compared_arms.py` |
| D-80 | `train()` (phase 10, `h2b_train_linear_hamiltonian.py`) prend soin de rendre `optimiser="cma" if use_cma else "nelder-mead"` — et l'écriture le jetait : `**{k: v for k, v in res.items() if not isinstance(v, str)}`. `res` ne porte que **deux** chaînes, `tag` (réajoutée juste après sous `tag_str`) et `optimiser` (qui ne l'était pas). Or c'est la **seule trace du repli** : `cma` est un paquet pip, et quand il manque le script prévient sur **une** ligne parmi des centaines puis tourne en Nelder-Mead. L'artefact était alors indiscernable d'un vrai run CMA-ES, et `(c_bias*, thr_amr*)` s'y lisait sans qu'on sache quel optimiseur les avait produits — un hyperparamètre sans provenance, la forme que `VIGIL.md` liste, doublée d'un repli silencieux | mesuré sur ce conteneur, où `cma` est **absent**, même commande (`--modes joint --n-iters 3 --sweeps 50 --n-restarts 1 --dim 2 --N 64`, sortie hors du dépôt) : **avant**, l'artefact porte 16 clés et **aucune ne s'appelle `optimiser`** ; **après**, `optimiser='nelder-mead'`, `optimiser_requested='cma'`, `cma_available=False` — l'écart entre ce qui a été demandé et ce qui a tourné se lit dans le fichier, et la même colonne entre dans `train_COMPARE`. Aucun artefact publié n'est concerné : `results/` ne contient **aucun** `train_*.npz`, la phase 10 n'a jamais été lancée sur ce dépôt. 3 tests échouent sur l'ancienne version, 4 passent après | `pytest tests/study/test_phase10_artefact_records_its_optimiser.py` |
| D-81 | **La phase 12 comparait deux bras sous deux disciplines.** `run_vqc` et `run_qke` (`h2b_variational_classifier.py`) faisaient `best_threshold_f1(p_va, Yva)` : le seuil optimisé sur **l'ensemble de validation lui-même**, puis le F1 rapporté au même endroit. Les deux bras classiques du même script passent par `fit_eval`, qui choisit sur `(p_tr, Ytr)`. Le verdict compare `max(F1 quantique)` à `max(F1 classique)` avec une bande de décision de **±0,02** : l'écart de discipline entrait directement dans la comparaison, **en faveur du bras que l'étude cherche à falsifier** — et le script annonce que dépasser le plafond classique serait *« a quantum advantage outside the Hamiltonian paradigm […] worth a full chapter »* | mesuré, phase 12 complète sur une configuration réelle (`orszag_tang`, Re=400, N=64, dim=4, `--n-train 80 --n-val 60 --d-q 3 --reps-fm 1`, sortie **hors du dépôt**) : QKE **0,786** avec le seuil pris sur la validation, **0,759** avec le seuil pris sur le train — **biais +0,027, supérieur à la bande de décision du script**. Le verdict change en conséquence : `delta = −0,008` → *« quantum ~= classical, no clear advantage »* devient `delta = −0,035` → *« best quantum model UNDERPERFORMS best classical »*. L'ancien nombre est **conservé** dans l'artefact (`f1_qke_thr_on_val`, `f1_vqc_thr_on_val`) et imprimé à côté du bon, pour que le biais reste mesurable et que la correction ne puisse pas être défaite en silence. **Aucun nombre publié ne bouge** : `results/` ne contient aucun `vqc_*.npz`, la phase 12 n'a jamais été lancée sur ce dépôt. Trouvé au passage et corrigé dans le même geste : un bras quantique qui **lève** (ici VQC, sur `qiskit.algorithms` absent depuis Qiskit 2.x) laissait le verdict se calculer sur l'autre **sans le dire** — le script annonce désormais quels bras ont réellement tourné et l'écrit dans l'artefact (`arms_that_ran`). 2 tests échouent sur l'ancienne version | `pytest tests/study/test_phase12_threshold_comes_from_train.py` |
| D-82 | **Même défaut que D-81, dans un autre fichier — trouvé en passant la question à *tous* les appels de `best_threshold_f1` de `study/`** (30 sites relus ; un seul autre était fautif). La table LOSO de `h2b_scenario_ablation.py` imprime côte à côte quatre colonnes : `F1_class`, `F1_9feat`, `F1_9+id` — les trois par `fit_eval`, seuil sur le **train** — et `F1_9+fuzz`, seul à choisir le sien sur `(p_fuzz, Yv)`, ses propres **labels de validation**. Et c'est précisément la colonne qui mesure la **chute** quand l'identité de scénario est fausse : un F1 gonflé sous-estime la chute, donc surestime la robustesse du modèle à une identité erronée. Le correctif réutilise le seuil de `rid`, qui est le **même** modèle (même graine, mêmes données d'entraînement) : son seuil vient de son propre train | mesuré, `--re 400 --N 64 --dim 4 --max-snaps 8`, sortie hors du dépôt : fold `orszag_tang` **0,212 → 0,198**, moyenne LOSO fuzz **0,163 → 0,160**. **L'écart est petit sur cette configuration** — 3 folds sur 4 y sont dégénérés à 0,000 et ne peuvent pas bouger ; il vaut 0,014 sur le seul fold non dégénéré. Dit tel quel plutôt qu'arrondi vers le haut : c'est la discipline qui est en cause, pas la taille de cet écart-ci. L'ancien nombre est conservé (`loso_site9fuzz_thr_on_val`) et imprimé à côté du bon. Aucun artefact publié : `results/` ne contient aucun `scenario_ablation_*.npz`. Retiré au passage, une ligne morte que la relecture a montrée sans effet (`rfz_pred = rid["p"]  # ignored`). 3 tests échouent sur l'ancienne version | `pytest tests/study/test_scenario_ablation_fuzz_threshold.py` |
| D-83 | **Troisième site de la famille D-81/D-82 — et il était dans le balayage qui a conclu « deux fautifs sur trente ».** Dans `h2b_random_split_bootstrap.py` (phase 11H, l'IC bootstrap du plafond de split aléatoire), `thr_cls` vient de `(Str, Ytr)`, le **train** ; `thr_site` et `thr_sten` venaient de `best_threshold_f1(concat(P_*_list), concat(Yv_list))` — les probabilités et les labels de **validation** — sous un commentaire qui annonçait *« same protocol as fit_eval grid search »*, alors que `fit_eval` prend le sien sur `(p_tr, Ytr)` : le commentaire disait le contraire du code. Ce n'est pas décoratif — `delta site-cls`, son IC bootstrap et `p(site ≤ class)` **comparent** les deux bras, et l'IC rééchantillonne l'ensemble même qui a servi à fixer le seuil. Le balayage des 30 appels ne l'a pas vu parce que l'appel est enfermé dans une fermeture aux noms génériques (`best_thr(P_list, Y_list)`) : la forme, pas le fond — exactement le motif que D-75 et D-76 ont déjà rencontré deux fois ici (« le détecteur d'une correction ne voit que les sites que cette correction a touchés »). **Le signe du biais est garanti par construction**, un seuil qui maximise le F1 sur la validation ne pouvant pas y faire moins bien que celui du train ; seule sa taille est empirique | mesuré, `--dim 4 --N 256 --max-snaps 80 --n-boot 500 --seed 0`, sortie hors du dépôt : `F1_site` **0,937 → 0,931**, `delta site-cls` **+0,460 → +0,454**, son IC **[+0,371, +0,547] → [+0,371, +0,534]**. `F1_stencil` **ne bouge pas** (0,973) : ses deux seuils coïncident à 0,050 sur cette configuration — mesuré, pas supposé, et un test le dit. **Aucun verdict imprimé ne change**, ni ici ni sur les quatre autres configurations parcourues (dim 4/16/32, graines 0/1/2), où le biais va de **+0,0004 à +0,0057** sur `F1_site` : c'est la discipline qui est en cause, pas la taille de cet écart-ci — dit tel quel plutôt qu'arrondi vers le haut. Les anciens nombres sont **conservés** (`f1_site_thr_on_val`, `f1_sten_thr_on_val`, `thr_*_on_val`) et imprimés à côté des bons. **Aucun nombre publié ne bouge** : `results/` ne contient aucun `random_split_bootstrap_*.npz`, la phase 11H n'a jamais laissé d'artefact sur ce dépôt. 5 tests sur 6 échouent sur l'ancienne version ; le sixième — le bras classique inchangé — passe des deux côtés, et c'est voulu | `pytest tests/study/test_random_split_bootstrap_threshold_from_train.py` |
| D-84 | **La phase 11E attribuait à la physique un écart de réduction — et flaggé fort parce que c'est une *lecture* qu'elle imprime.** `h2b_v1_hamiltonian_loso.py` imprime `V1_class − V2_class = +0,145  (Lohner + 4-indicator RMS effect)`. Ses deux colonnes « classiques » sortent de la **même** fonction, `AngleMapper.classical_score` : `Sv2c` est son score fin réduit par `block_max` (via `build_patch_hamiltonian`), `Sv1c` le **même** score fin réduit par `block_avg`. Le terme de Löhner, la moyenne quadratique des quatre indicateurs et les normalisations sont **identiques des deux côtés** — il n'y a aucun « effet Löhner » à mesurer entre ces deux colonnes. La seule autre différence est l'opérateur `Jz` (différences avant non divisées ici, centrées divisées par `2dx` là-bas). Question 4 de `VIGIL.md`, forme « la réduction des champs contre celle du score » | mesuré, `--dim 4 --N 256 --max-snaps 30 --n-boot 500 --seed 0 --re 400` : écart publié `V1_class(avg) − V2_class(max)` **+0,145** ; **à réduction égale** `V1_class(max) − V2_class(max)` **+0,051** — la réduction en porte **+0,094**, le reste est l'opérateur `Jz`, **rien** n'est un effet Löhner/RMS. Preuve directe et séparante : en passant à `AngleMapper.classical_score` le `Jz` **centré** de `solver.get_fluxes`, le champ fin de la colonne V1 devient **bit à bit** celui de la colonne V2 (`np.array_equal` vrai) ; sans cette substitution ils diffèrent de 0,0097 sur le snapshot testé (médiane 0,015, max 0,164 sur les 120 snapshots). Mesure indépendante à grille de seuil unique : réduction seule sur le champ identique **+0,149**, écart d'opérateur seul **−0,001** en moyenne / **+0,050** en max. **Une hypothèse de moi, mesurée et fausse** : « le seul fold dont l'IC exclut zéro (`mhd_rotor`) s'effondre à réduction égale » — il passe de +0,335 à +0,206, il en perd **38 %**, pas la totalité ; dit tel quel. **Aucun nombre publié ne bouge** : `results/` ne contient aucun `v1h_loso_*.npz`, et aucun document ne cite la phase 11E — son successeur `h2b_psi_feature_loso.py` (T5) fait déjà les **deux** réductions du même champ, ce qui confirme que l'asymétrie était connue comme risque, jamais mesurée. Correction **minimale** : les colonnes existantes sont inchangées au chiffre près, une colonne de contrôle à réduction égale (`f1_v1_class_maxpool`) est ajoutée à l'impression et à l'artefact, et la ligne de décomposition dit désormais ce qu'elle mesure. 3 tests sur 4 échouent sur l'ancienne version ; le quatrième — la preuve que les deux colonnes sont la même fonction — passe des deux côtés, et c'est voulu : il ne dépend d'aucune correction | `pytest tests/study/test_phase11e_gap_is_a_reduction_not_physics.py` |
| D-86 | **La phase 10a publiait le bord gauche de sa grille comme un optimum mesuré.** `h2b_analytical_solution.py` choisit `c_bias*` par `bi = np.argmax(f1_grid)` sur un balayage de `np.logspace(-1, 2, 25)`. Quand aucun `c_bias` de la grille ne sort le champ moyen de l'état uniforme « ne pas raffiner », `F1(c)` est **identiquement nul** et `argmax` rend l'indice 0 : `c_bias* = 0,1000`, la borne **basse** — le point le **plus** dominé par les couplages, l'exact opposé de ce que le balayage cherche. Le nombre est fini, dans la grille, du bon type : rien ne le distingue d'un optimum. La phase 10 le lisait ensuite comme `theta_init = (log10 0,1 ; thr*) = (−1 ; thr*)`, c'est-à-dire qu'elle démarrait son optimiseur **au bord du domaine** sur la foi d'une mesure qui n'avait rien mesuré. **Même forme que D-56, un cran plus bas** : là, la campagne ne trouvait pas ses entrées ; ici elle les trouve, et c'est le balayage lui-même qui est vide. `CLAUDE.md` du dépôt : « un balayage vide doit crier » | **cause, mesurée** (harris_tearing Re=400, N=96, dim=4) : `\|h_unit\| max` = **1,91e−02** contre `\|C\| max` = **7,80** — à `c = 100`, borne **haute** de la grille, le biais plafonne à 1,91 et ne renverse aucun site ; le régime « biases win » que la docstring annonce n'est atteint **nulle part** sur la grille. **Étendue** : **14 balayages plats sur 52** configurations parcourues — **0/16** à dim=2, **5/16** à dim=4 (N=96), **8/16** à dim=8, **1/4** à dim=4 (N=256) : la dégénérescence croît avec `dim`. **Séparation sans zone grise** : écart max−min **exactement 0** sur les 14 dégénérés, **0,125 à 0,433** sur les 38 informatifs. **Avant/après sur la même commande** (`--dim 4 --N 96 --max-snaps 8 --seed 0`) : `mhd_rotor` `c_bias*` **68,77 → 91,66** (**+0,125 décade** là où la phase 10 lit ce nombre, elle en prend le log10), `thr*` 0,4450 → 0,4333, `F1_MF` 0,0938 → 0,1250 ; `joint` `c_bias*` **54,56 → 79,32** (**+0,163 décade**), `thr*` 0,4858 → 0,4225, `F1_MF` 0,1701 → 0,2474 ; `harris_tearing` **0,1000 → NaN** (4/4 dégénérés — il n'a pas de `c_bias*`, et 0,1000 est un nombre plausible qui n'en est pas un) ; `orszag_tang` et `kelvin_helmholtz` **inchangés au chiffre près**, aucun dégénéré. **Preuve directe du silence** : sur l'ancienne version, `--scenario harris_tearing --dim 4 --N 96` sort avec le **code 0** et écrit un artefact de six lignes portant toutes `c_bias* = 0,1000` — indiscernable d'une campagne réussie ; sur la nouvelle elle **lève**. **Aucun nombre publié ne bouge** : `results/` ne contient aucun `analytical_*.npz` ni aucun `train_*.npz`, et la phase 10 imprimait jusqu'ici *« no analytical init → using default x0 »* — l'initialisation fautive n'a donc jamais été consommée sur ce dépôt. Correction **minimale** : le critère est la **platitude** (pas la nullité — c'est elle qui rend l'`argmax` arbitraire), les colonnes existantes sont inchangées, un drapeau `degenerate` et l'écart `f1_span` voyagent **avec** les nombres dans l'artefact, les lignes dégénérées sortent de l'agrégation (`mean_over_informative`, extraite pour être testable sans rejouer la campagne — même geste que D-46/D-50/D-52/D-85), une campagne **entièrement** dégénérée lève, et le consommateur (`build_init_map`, extraite de `h2b_train_linear_hamiltonian`) les écarte : sans ce dernier point le drapeau serait décoratif et la mauvaise init resterait. **12 tests, tous rouges sur l'ancienne version** — vérifié en rejouant le fichier contre elle, imports du correctif shuntés, pas seulement en constatant l'échec de collecte | `pytest tests/study/test_phase10a_flat_sweep_is_not_an_optimum.py` |
| D-88 | **À dim=4, k=2 couvre déjà toute la grille périodique — k=3 n'est PAS un voisinage plus grand, seulement plus de colonnes dupliquées, et `n_feats` comptait les nominales.** `h2b_neighbour_cone_curve.py` (T1b, protocole v3 §2) construit `khop_features` par `np.roll` PÉRIODIQUE (mod `dim`) sur les décalages nominaux d'une boule de Chebyshev que rend `khop_offsets(k)` : k=2 → 25 décalages (carré 5×5), k=3 → 49 (7×7). Dès que la largeur du carré `2k+1` atteint `dim`, deux décalages nominaux distincts retombent sur le même résidu `(dy % dim, dx % dim)` et produisent deux colonnes bit-à-bit identiques — `khop_features` ne le sait pas, et rien dans le fichier ne le disait : le tableau imprimé et l'artefact ne portaient que `n_feats = len(khop_offsets(k)) * 9`, le compte NOMINAL, jamais comparé aux colonnes réellement distinctes. À dim=4 (la valeur par défaut du script), `2·2+1 = 5 ≥ 4` : k=2 couvre déjà les 4 résidus de chaque axe, donc les 16 cellules de la grille entière ; k=3 (`2·3+1 = 7 ≥ 4` aussi) n'ajoute alors AUCUN résidu de plus. La courbe de cône traiterait k=2 et k=3 comme deux tailles de voisinage différentes (225 puis 441 « features ») alors qu'elles rendent, sur cette grille, exactement les mêmes 144 colonnes — un `delta/hop` calculé entre ces deux k ne compare pas deux voisinages, il compare un jeu de colonnes à sa propre redite | mesuré sur les fonctions pures de `h2b_neighbour_cone_curve.py` (`khop_offsets`, `khop_features`), à l'opérateur assorti — comparaison directe aux colonnes RÉELLEMENT produites par `khop_features`, pas une réimplémentation parallèle des décalages : `np.unique(khop_features(feats, k), axis=1).shape[1]` == `khop_distinct_footprint(k, dim) * 9` sur les quatre k, vérifié. **À dim=4** : `n_feats` nominal (l'ancien, seul publié) `[9, 45, 225, 441]` pour k=[0,1,2,3] ; `n_feats` distinct (le réel) `[9, 45, 144, 144]` — k=0 et k=1 n'ont encore aucun doublon (`2k+1 < 4`), k=2 et k=3 en ont et **rendent le même nombre**, `144 = 16 × 9` = la grille entière (`dim·dim`) fois les 9 features de base. **Champ qui sépare**, pas un artefact universel du calcul : à dim=8, `2k+1 ≤ 7 < 8` pour tout k ≤ 3 — `n_feats` distinct == nominal sur les quatre k, `[9, 45, 225, 441]`, aucune saturation. **Aucun nombre publié ne bouge** : `results/` ne contient aucun `t1b_cone_curve_*.npz` (le script était encore listé « jamais lu » dans `COUVERTURE.md` avant cette passe), et `main()` (chargement DNS, boucles LOSO/split bloqué) n'a jamais tourné sur ce dépôt faute d'artefacts DNS d'entrée. Correction **minimale** : `khop_offsets`/`khop_features` sont inchangées — ce que le modèle reçoit en entrée ne change pas, donc aucun F1 déjà mesuré ne bouge — ; `khop_distinct_footprint(k, dim)` calcule le compte réel par le même chemin structurel que `khop_features` (pas de réimplémentation des `np.roll`) ; les deux tableaux imprimés (bloqué et LOSO) gagnent une colonne `n_distinct` et un `[FLAG k=… n'agrandit pas le voisinage…]` dès que deux k consécutifs saturent ; l'artefact `.npz` sauvegarde désormais `n_feats_nominal`, `n_feats_distinct`, `k_saturated` à côté des F1. 16 tests, dont 5 nouveaux — tous rouges à la collecte sur l'ancienne version (`ImportError: cannot import name 'khop_distinct_footprint'`, le fichier de test important la fonction du correctif), 16 passent sur la nouvelle, vérifié en rejouant le fichier de test contre le module d'avant D-88 | `pytest tests/study/test_t1b_cone_curve.py` |
| D-87 | **Un `argmax` posé SUR le bord de la grille de `c_bias`, et un `thr*` hors de la boîte de la phase 10 — tous deux acceptés en silence.** C'est la famille que D-86 laisse passer *par construction* : D-86 traite la courbe **plate**, dont l'argmax ne désigne rien ; ici les courbes sont **informatives** (écart max-min 0,224 et 0,433) et leur argmax tombe quand même sur le **dernier** point de la grille. La grille s'arrête là — l'optimum réel peut être au-delà. La phase 10 porte `hits_bound()` pour exactement cette pathologie, appliqué au theta **final** et jamais à `x0` ; la phase 10a n'avait aucun équivalent. Deuxième moitié : `best_threshold` balaie `linspace(0.02, 0.60, 59)` — **exactement** la boîte de la phase 10 — *réunie* aux quantiles du score, qui en sortent ; la phase 10 rabotait le résultat avec `np.clip`, qui ne dit rien. Formes de `VIGIL.md` : *deux chemins censés coïncider* (deux copies de la même boîte, dans deux fichiers) et *repli silencieux* (le bord de grille indiscernable d'une valeur choisie) | mesuré, `--dim 4 --N 256 --max-snaps 8 --seed 0`, Re=400, sortie hors du dépôt. **`kelvin_helmholtz` c\* = 100,0 et `mhd_rotor` c\* = 100,0 — le bord DROIT**, avec des écarts max-min de **0,224** et **0,433**, donc informatifs et laissés passer par D-86 ; `orszag_tang` c\* = 31,62 est le seul intérieur. **Ce qu'il y a au-delà du bord, mesuré** : le plus petit `c` qui fait basculer un seul spin vaut **1,6e4** (kelvin_helmholtz) et **5,0e4** (mhd_rotor) — deux ordres au-delà de la grille, dont le bord droit est 100. **`hits_bound(x0)` est vrai sur 3 lignes sur 4** avant D-86, et **sur 2 des 3 lignes retenues après** — D-86 écarte harris_tearing, les deux autres coins restent. **`thr*` hors boîte sur 2 lignes sur 4** : **0,6777** (harris_tearing) et **0,6908** (kelvin_helmholtz) contre une borne haute de **0,60**. **Aucune valeur ne bouge** : les quatre lignes rendent après correction `0,678/0,10/0,000/0,571`, `0,691/100,00/0,223/0,400`, `0,220/100,00/0,433/1,000`, `0,180/31,62/0,284/0,313` — identiques au chiffre près, D-86 compris. **Aucun nombre publié ne bouge** : `results/` ne contient ni `analytical_*.npz` ni `train_*.npz`. Correction **minimale**, aucune valeur touchée, trois choses rendues visibles : `at_left_edge` / `at_right_edge` / `thr_outside_box` accompagnent chaque ligne à l'écran et dans l'artefact (avec `theta_bounds`, pour qu'un artefact relu dise contre quoi le test a été fait) ; la boîte est **lue** chez la phase 10 au lieu d'être recopiée ; `build_init_map` annonce ce qu'il rabote et quel `x0` atterrit sur un coin ; `hits_bound(x0)` est évalué, imprimé et consigné (`x0_theta`, `x0_hits_bound`, `x0_from_analytical`). Sur un agrégat les drapeaux de bord **excluent les dégénérés** — leur argmax est au bord gauche par construction, les compter allumerait le drapeau partout — tandis que `thr_outside_box` porte sur la valeur **propre** de l'agrégat, la seule que la phase 10 rabotera. Vérifié par **exécution** de bout en bout des deux scripts, pas par collecte. 9 tests sur 14 échouent sur la version d'avant ; les 5 autres passent des deux côtés, et c'est voulu — quatre épinglent le bord de grille contre la borne (ils tomberaient si l'un bougeait sous l'autre), le cinquième est le témoin intérieur sans lequel les avertissements ne mesureraient rien | `pytest tests/study/test_phase10a_argmax_is_not_a_grid_edge.py` |
| D-85 | **Le critère d'acceptation de la tâche 4 était imprimé, jamais comparé — et il échoue.** Le protocole v3 (§8.3) pose l'acceptation de `h2b_blocked_split.py` comme *« one table ; random-split numbers match Task 0 »*. Le script imprimait `acceptance refs (Task 0): B2 random F1 = 0.475, B4 gbt-9 (max) random F1 = 0.980` et s'arrêtait là : aucune comparaison, aucun drapeau, rien dans l'artefact. L'en-tête de `tests/study/test_t4_blocked_split.py` renvoyait pourtant la vérification chiffrée à *« l'exécution sur les vraies données »* — où elle n'avait pas lieu. Personne ne la faisait. Même forme que D-52 (`h0_optimiser_equivalence`), et `CLAUDE.md` du dépôt est explicite : « un test qui ne peut pas échouer est un défaut » | mesuré à HEAD, `--dim 4 --N 256 --seed 0` (Re=400 ; **identique** à `--max-snaps 30` et `--max-snaps 80`, le GBT étant déterministe à graine fixée, et les deux exécutions rendent les mêmes chiffres) : `B2 classical (block_max)` **0,472** contre 0,475 → écart 0,003, dans la bande ; `B4 gbt-9 (max)` **0,908** contre **0,980** → écart **0,072**, hors bande. La reproduction du split aléatoire de la phase 11A ne retrouve donc pas le nombre de la tâche 0 pour le bras GBT champ-moyen. **Les deux références ne sont pas réajustées** : ce sont des nombres d'**archive d'avant l'audit**, même provenance que celles de `aggregate_v3.py` (D-49), et `VIGIL.md` veut qu'un seuil périmé se **remesure**, pas qu'il se retouche. La correction rend le critère comparable — `check_acceptance()`, extrait pour être testable sans rejouer la campagne (même geste que D-46/D-50/D-52), imprime `OK`/`MISMATCH` et écrit référence, mesure et verdict dans l'artefact (`acceptance_*`). Le code de sortie est **inchangé** : le rendre non nul est la même question ouverte que celle posée sur `sanity_check` (D-44) et sur `h0_optimiser_equivalence` (D-52). **Aucun nombre publié ne bouge** : `results/` ne contient aucun `t4_blocked_split_*.npz`. Une référence qui ne désignerait aucune ligne **lève** au lieu de passer en silence — le piège du balayage vide, fermé dans le même geste. 4 tests | `pytest tests/study/test_t4_acceptance_is_compared_not_printed.py` |
| D-89 | **`h4_unseen_floor.py` (T22d) levait `KeyError: 'canonical'`, sans artefact écrit, dès qu'un bras était `total_abort` dans son entrée — le cas que son propre voisin `h4_unseen_conditions.py` (T22) documente et gère explicitement comme « un résultat, pas une panne ».** Un bras `total_abort` de `t22_unseen_{mode}_{fold}.json` ne porte AUCUNE sous-clé `"canonical"`/`"unseen"` (seulement les `*_runs` bruts et `degradation_ratio = NaN`) : voir `study/h4_transfer/h4_unseen_conditions.py:467-475`, qui construit ce cas précisément pour ne pas planter. `h4_unseen_floor.py:135-137` lisait pourtant `t22["arms"][arm]["canonical"]["phys_score"]` sans jamais vérifier, ce qui n'a jamais été exercé sur ce dépôt (mesuré : les 4 artefacts `t22_unseen_unseen-ic_*.json` déjà commités ont les deux bras `completed` sur les 4 folds) mais qui est exactement le scénario que T22 va au-devant de gérer. Question 3 de `VIGIL.md` (consomme-t-elle ce que sa signature annonce ?) posée entre deux fichiers du même module, écrits l'un pour compléter l'autre | **avant** : reproduction directe de `t22["arms"]["qhas"]["canonical"]["phys_score"]` sur une entrée à la forme exacte qu'écrit `h4_unseen_conditions.py` pour un bras `total_abort` → `KeyError: 'canonical'`. **Après** : le calcul de ratio, extrait en `floor_ratios()` pour être testable sans rejouer les ~4h de DNS d'un fold, teste la **présence** des clés `"canonical"`/`"unseen"` (pas un champ `"status"` — les 4 artefacts réels déjà dans `results/` n'en portent aucun, d'avant l'introduction de ce champ dans le schéma ; un premier correctif basé sur `status` aurait classé ces 4 artefacts **réels** comme morts et rendu `NaN` à la place des 8 ratios déjà publiés dans `results/t22d_unseen_floor_*.json` — erreur trouvée en rejouant le correctif contre eux, avant de le pousser). Sur l'entrée `total_abort` : plus d'exception, le bras est signalé (`dead = ["qhas"]`), son ratio est `NaN` plutôt qu'un nombre reconstitué, et le bras vivant garde le sien. **Aucun nombre publié ne bouge** : les 8 ratios de `results/t22d_unseen_floor_{ot,kh,rotor,tearing}.json` sont rejoués bit à bit contre `floor_ratios()` et coïncident à moins de 1e−9 sur les 4 folds × 2 bras × 2 conditions | `pytest tests/study/test_h4_unseen_floor_total_abort_arm.py` |
| D-90 | **Même dossier, même défaut de fond, un cran plus loin : `h4_transfer_summary.py` (T22c) lisait une clé qui n'a jamais existé dans l'artefact qu'il consomme, et affichait « the None arm aborted » au lieu du nom du bras réellement fautif.** `h4_unseen_conditions.py:512` écrit `total_abort_arms` (PLURIEL, une **liste** — un fold peut voir les deux bras avorter). `h4_transfer_summary.py` lisait `total_abort_arm` (singulier) : `.get()` rend systématiquement `None`, sans lever — un repli silencieux qui affiche une valeur plausible et fausse plutôt que de crier. Trouvé en vérifiant, sur D-89, que `h4_unseen_floor.py` et `h4_unseen_conditions.py` s'accordaient déjà sur la forme d'un bras `total_abort` ; la question 4 de `VIGIL.md` posée une seconde fois, sur le troisième fichier du dossier qui consomme la même forme | **avant**, sur une entrée à la forme exacte qu'écrit `h4_unseen_conditions.py` (`total_abort_arms: ["qhas"]`) : le message imprimé porte `the None arm aborted on every draw…` — vérifié en rejouant `main()` contre un artefact synthétique, isolé dans `tmp_path` (jamais `results/` du dépôt : un premier essai de mesure, avec `--folds tearing`, a lu le VRAI `results/t22_unseen_unseen-ic_tearing.json` et écrasé le VRAI `results/t22c_transfer_summary.json` à un seul fold — restauré depuis `git checkout` avant de pousser quoi que ce soit, aucun commit n'en porte trace). **Après**, même entrée : `the qhas arm(s) aborted…`. Aucun artefact réel de ce dépôt n'a jamais eu de fold `total_abort` (mesuré sur les 4 `t22_unseen_unseen-ic_*.json` committés, D-89) : le défaut n'a donc jamais affiché un « None » sur un résultat publié, mais aurait affiché un message trompeur au premier fold qui avorte — exactement le chemin que T22 construit ce cas pour couvrir | `pytest tests/study/test_h4_transfer_summary_total_abort_arm_name.py` |

**Douze de ces défauts viennent d'une seule question** — *deux chemins censés
coïncider coïncident-ils encore ?* Aucun test de valeur ne pouvait les voir :
tous rendaient un résultat plausible.

Deux ont été vus non par une question, mais en **retirant une couche** : D-26
et D-27 n'apparaissent qu'une fois la projection de B supprimée. Tant qu'elle
masquait la divergence, les scénarios paraissaient sains.

Les entrées sont dans l'ordre où elles ont été produites. Celles qu'un
résultat postérieur a dépassées sont **conservées, avec la rétractation
écrite sur place** — c'est la trace de ce qui a été cru, et pourquoi ce
n'est plus vrai.

L'ordre historique fait apparaître des références à « V3 » et « V4 » : ce
sont des étapes de l'étude, pas des versions du code. Les campagnes
antérieures sont dans `docs/archive/`.

**Règle de continuité** : aucun symbole de V1 n'est redéfini. Tout ce qui
est réutilisable est importé — `MHDSolver`, `build_patch_hamiltonian`,
`build_ising_terms`, `sa_multi_restart`, `spins_to_decisions`,
`prepare_qaoa_inputs`, `run_qaoa_on_snapshot`, `div_B`, `total_energy`,
`downsample_fields`, `bootstrap_by_trajectory`, `git_commit_hash`.

**Recette de vérification**

```bash
python -m pytest tests/ -q -m "not slow"
python study/common/aggregate_master_table.py     # recalcule chaque nombre
```

Le troisième est le test de non-régression : il recalcule chaque nombre
publié depuis son artefact. **État actuel : 164 OK / 16 DIFF / 0 MISSING** —
les 16 écarts sont les nombres déplacés par les corrections, à republier
après la réoptimisation (voir `DEFAUTS.md`).

Toutes les études ont été relancées à la résolution de production **N=256**
(4 scénarios, Re=400). Les deux passes — N=64 exploratoire et N=256 de
confirmation — sont rapportées ; chaque conclusion qualitative est identique
aux deux résolutions.

### D-28, en détail

Trouvé en auditant par contrat `src/hyperparams_loader.py` (`VIGIL.md`, Q4 :
deux chemins censés coïncider). Le correctif existait déjà — commit
`0327ce1`, 12 tests dans `tests/study/test_hyperparams_two_sources.py` —
mais n'avait de ligne ni dans `DEFAUTS.md` ni ici : exactement la déviation
que `VIGIL.md` interdit de laisser non écrite. Cette entrée referme l'écart ;
le correctif lui-même n'est pas de cette passe.

**Avant** (`_load_new_format`, avant `0327ce1`) :

```python
entry = default.get(method)
if entry is None:
    # Fallback: try the other method
    for m in ['quantum', 'classical']:
        if m in default:
            entry = default[m]
            break
if entry is None:
    raise KeyError(f"No default {method} params found in JSON")
```

Demander `method='quantum'` sur un fichier ne portant que `classical`
renvoyait donc les paramètres de l'**autre** bras, sans le signaler — la
boucle essaie `'quantum'` puis `'classical'` dans cet ordre fixe, donc
systématiquement, pas seulement par accident. `src/pipeline.py` ne peut pas
distinguer ce cas d'un vrai jeu quantique : la comparaison des deux bras
devient vide de sens, en silence. Second repli du même ordre : plusieurs
`lambda_cost` pour une phase → le premier pris par ordre alphabétique, un
choix arbitraire indiscernable d'un choix motivé.

**Après** : `KeyError` explicite listant les bras disponibles pour le
premier ; `KeyError` sauf lambda unique (choix alors forcé, donc licite)
pour le second.

**Mesuré** sur `results/hyperparams/best_hyperparams.json` (gelé) : les
deux bras s'y chargent toujours normalement — non-régression — et les deux
fautes lèvent sur les cas construits qui les provoquent
(`test_the_loader_refuses_to_substitute_the_other_arm`,
`test_the_loader_refuses_an_ambiguous_cost_weight`,
`test_a_single_cost_weight_stays_implicit`).

```bash
pytest tests/study/test_hyperparams_two_sources.py -q
```
→ 12 passed (vérifié à `HEAD`, `claude/kind-babbage-927g10`).
`python study/common/aggregate_master_table.py` inchangé (164 OK / 16 DIFF /
0 MISSING) : cette entrée ne déplace aucun nombre publié.

**Trouvé au passage, non corrigé.** Le bloc `per_scenario` du bras quantique
contient quatre copies **identiques** du bloc `default` — aucun réglage n'y
est réellement par scénario — et `orszag_tang`/`mhd_rotor` (2 des 4
scénarios de l'étude) en sont absents pour les deux bras.
`load_hyperparams(scenario=...)` n'est appelé nulle part dans le dépôt
(`grep -rn "load_hyperparams(.*scenario" src/ study/` ne rend rien) : la
branche est morte, aucun nombre publié n'en dépend aujourd'hui. Épinglé,
pas corrigé, par `test_the_per_scenario_quantum_block_is_one_set_repeated`
et `test_two_study_scenarios_have_no_per_scenario_entry` — pour que
personne ne « répare » ce bloc en y recopiant `default`, ce qui masquerait
le problème au lieu de le trancher.

### Écart de registre trouvé en écrivant cette entrée — signalé, non corrigé

En cherchant où consigner D-28, deux incohérences non liées à D-28 :

- **Collision de numéro.** La section « D-18 » plus bas dans ce fichier
  (« rectification : la moitié `fluctuating_KE` était déjà connue »)
  raconte la déviation reclassée depuis comme **D2** dans `DEFAUTS.md`
  (table « Gelés volontairement »). Le numéro D-18 désigne aujourd'hui,
  dans la table ci-dessus et dans `DEFAUTS.md`, un défaut différent : la
  garde de divergence à 1e100. Un lecteur qui suit
  `DEFAUTS.md/RESULTS.md → D-18 → chercher "D-18" plus bas dans ce fichier`
  tombe sur la mauvaise section. Survit à la réorganisation du 12 août.
- **Compte de tête inexact.** Avant cette entrée, le titre de cette section
  annonçait « Les 24 défauts corrigés » pour 23 lignes numérotées D-N dans
  la table — un défaut d'écart au niveau du registre lui-même, pas du code.
  L'ajout de D-28 le rend exact par coïncidence ; ne pas s'y fier comme
  preuve que le compte était juste avant.

Non corrigé ici : ni l'un ni l'autre n'est de mon audit, et je ne peux pas
écrire leur mesure sans la fabriquer — mesurer, documenter, ne pas
corriger sans la mesure, règle de `VIGIL.md`.

---

## T11 — Quantum-contribution attribution (audit P0)

`study/h0_selection/h0_optimiser_equivalence.py --N 64 --dim 2 --n-snaps 2`

At the **deployed size** (`VQA_N = 2` → 8 qubits, periodic root scan, i.e.
exactly the configuration `refinement.py` solves at depth 0).

| solver | hit optimum | E gap | spin agreement | mask match | wall (s) |
|---|---|---|---|---|---|
| exhaustive (certified) | 1.000 | 0 | 1.000 | 1.000 | 0.000 |
| simulated annealing | 1.000 | 0 | 1.000 | 1.000 | 0.121 |
| SA warm-started | 1.000 | 0 | 1.000 | 1.000 | 0.123 |
| greedy local search | 1.000 | 0 | 1.000 | 1.000 | 0.000 |
| classical decision alone | 1.000 | 0 | 1.000 | 1.000 | 0.000 |
| QAOA p=1 (statevector) | 1.000 | 0 | 1.000 | 1.000 | 0.414 |
| QAOA p=2 (statevector) | 1.000 | 0 | 1.000 | 1.000 | 0.612 |
| QAOA p=2, 4096 shots | 1.000 | 0 | 1.000 | 1.000 | 0.617 |

- The cost Hamiltonian is **diagonal** (Z/ZZ/ZZZZ only), verified at runtime
  by `is_diagonal_cost_hamiltonian` on every snapshot. Its ground state is a
  computational basis state, so "exact diagonalisation" reduces to a
  classical enumeration of 2^8 = 256 configurations.
- Every solver reaches the certified optimum and returns the same mask.
  **Pre-registered rule fires: quantum optimisation is not the source of any
  gain.** A closed-loop improvement would attribute value to the
  Hamiltonian, not to its quantum optimiser.

**Caveat that makes the agreement partly vacuous** (see T11b): the optimum
itself is uniform, so the solvers agree on a trivial problem.

---

## T11b — Does the QAOA optimise its own Hamiltonian? (audit P0)

`study/h0_selection/h0_qaoa_displacement.py --N 64 --dim 2 --reps 1 2 3 4`

Position of three points in marginal space: `m_theta` (amplitude encoding of
the classical score alone), `m_qaoa` (optimised circuit), `m_gs` (exact
ground state). `progress` = projection of the realised displacement on the
required one; 0 = decision unchanged, 1 = optimum reached.

| reps | progress | ‖displacement‖ | ‖required‖ | ‖remaining‖ | mean marginal |
|---|---|---|---|---|---|
| 1 | +0.0590 | 0.1276 | 0.8381 | 0.8010 | 0.7217 |
| 2 | +0.0563 | 0.1178 | 0.8381 | 0.8030 | 0.7205 |
| 3 | −0.0298 | 0.1178 | 0.8381 | 0.8536 | 0.7044 |
| 4 | −0.0584 | 0.1883 | 0.8381 | 0.8830 | 0.6980 |

- **The exact ground state is a UNIFORM mask on 100% of snapshots**
  (8/8: 4 scenarios × 2 snapshots) — refine-all, carrying no spatial
  information. Cause (consistent with V3 Task 9): the ferromagnetic
  couplings dominate the Z bias, |C| ≈ 2.0 and |K| = 1.0 against
  |h| ≈ 0.071, a ratio ≈ 28.
- **Mean variational progress = 0.0068** (0.68%). The circuit's displacement
  is essentially orthogonal to the direction of its own optimum.
- Progress **does not increase with depth**; it becomes negative by reps=4
  (−0.117 from reps 1 to 4). Deeper circuits move slightly *away* from the
  optimum of the declared cost.

**Reading.** The deployed decision is not a minimiser of the declared cost
function. It is a ≤4%-in-norm perturbation of the amplitude encoding
θ = 2·arcsin(√score), i.e. of the classical score itself.

---

## T13 — Causal ablation of term families (audit P1)

`study/h3_representation/h3_term_ablation.py --N 64 --dim 2 --n-snaps 2`

Exact ground state recomputed after zeroing each family (control `full`
must change nothing).

| ablation | decisions changed | uniform | refined fraction | F1 | n_optima |
|---|---|---|---|---|---|
| full (control) | **0.000000** | 1.000 | 1.000 | 0.317 | 1.0 |
| no_Z | 1.0000 | 1.000 | 0.000 | 0.000 | 8.0 |
| no_ZZ | **0.0000** | 1.000 | 1.000 | 0.317 | 1.0 |
| no_ZZZZ | **0.0000** | 1.000 | 1.000 | 0.317 | 1.0 |
| Z only (both couplings removed) | **0.0000** | 1.000 | 1.000 | 0.317 | 1.0 |
| couplings only (Z removed) | 1.0000 | 1.000 | 0.000 | 0.000 | 8.0 |

- Removing **all** ZZ and **all** ZZZZ couplings changes **no decision**.
  The single-site Z bias alone reproduces the full-Hamiltonian decision
  exactly.
- Removing the Z bias destroys the decision entirely and leaves an
  8-fold degenerate ferromagnet.
- The control is exactly 0. ⚠ **It does not validate the measurement
  chain — D-54.** `zero_hamiltonian_terms(hp, ())` returns a copy of `hp`,
  so the control compares the *same function on the same input*: it is 0 by
  construction. Measured by sabotaging the ablation so that nothing is ever
  zeroed (orszag_tang Re=400 N=64 dim=2): the control still reads
  **0.000000**, and `no_ZZ` / `no_ZZZZ` / `Z only` still read
  **0.0000** — the three rows this table's conclusion rests on. What
  separates a real ablation from an empty one is `removed_max`, now written
  per row: `no_ZZ` **2.6558e+00**, `no_ZZZZ` **1.0000e+00**, `no_Z`
  **8.3004e-02** (max over 2 snapshots at N=64), so at that configuration
  the couplings *were* genuinely removed and the inertness reading stands —
  it just was not evidenced before. The numbers in the table above are
  unchanged.

**Reading.** At the deployed grid size the coupling terms — the entire
motivation for an Ising/quantum formulation — are **causally inert**. This
is a causal statement, unlike the post-hoc ZZ/ZZZZ attributions of the
manuscript.

> **⚠ Portée à préciser — D-51.** L'énoncé porte sur **une** des deux familles
> ZZZZ. `study/` code `advanced_anomalies_enabled = False` partout, donc le
> ZZZZ de **point X** (`K_xpoint`) n'est jamais construit — et
> `build_ising_terms`, dont dépendent l'ablation, le recuit et la
> diagonalisation exacte, ne sait pas le lire même s'il l'était. L'ablation
> `no_ZZZZ` annule une clé que `ground_state_mask` ne consulte jamais.
> **Le tableau ne bouge pas** : mesuré à `dim = 2`, `max|K_xpoint| = 0,0000e+00`
> sur 4/4 scénarios, 12 → 12 termes de Pauli, 0 spin changé au fondamental
> exact — le terme est nul ici, donc l'ablation n'aurait rien retiré. À
> `dim = 4` en revanche il vaut 0,23 à **1,00** fois `max|K_plaquettes|`. La
> campagne d'entraînement l'active sur **6/6** scénarios (D-33). Conséquence
> pratique la plus lourde : `beta_xpoint`, que D-22 range parmi les 8
> paramètres à réoptimiser, n'influence **aucune** mesure de `study/`.
> Détail et options : `docs/DEFAUTS.md` D-51.

---

## T12 — Equivariance and orbit error (audit P1)

`study/h3_representation/h3_equivariance.py` (dim=2 exact; dim=8 with annealed ground
state and a mandatory reproducibility control).

Step 1 — the transformation must be a symmetry of the discrete solver:
`eps = ‖T(step(U)) − step(T(U))‖ / ‖step(U)‖`.

| op | eps (N=64) |
|---|---|
| rot180 | 2.8e-16 (machine precision — exact symmetry) |
| flip0 / flip1 / rot90 | 7.8e-6 |

Step 2 — orbit error of the decision map, dim=8 (structured masks):

| op | classical route | ground-state route |
|---|---|---|
| flip0 | 0.0195 | 0.3984 |
| flip1 | 0.0508 | 0.3555 |
| rot180 | 0.0547 | 0.3359 |
| rot90 | 0.0508 | 0.3047 |
| **mean** | **0.0439** | **0.3486** |

Step 3 — **mandatory control** (`solver_noise_floor`): disagreement of the
ground-state route between annealing seeds **on the same, untransformed
field** = **0.2676**, with the refined fraction swinging by 0.15 across
seeds.

- The classical score map is **nearly equivariant** (4.4% orbit error,
  deterministic, floor = 0). The residual is attributable to the one-sided
  finite differences used in the indicator.
- The ground-state route's 0.349 orbit error is **not interpretable as
  non-equivariance**: the annealed optimiser is itself irreproducible at a
  comparable magnitude (floor 0.268). The verdict printed by the script
  requires a 2× margin over the floor, which is not met.
- At dim=2 with exact enumeration, orbit error is exactly 0 for all routes —
  but only because the mask is uniform, so the test is vacuous there.

**Reading.** What this establishes is not an equivariance defect but a
**degeneracy defect**: at dim=8 the objective is flat enough that two
annealing seeds disagree on 14–37% of patches. A decision defined as
"the ground state" is not well posed at that size.

---

## T14 — Numerical validation of the V1 solver (audit P1)

`study/h1_solver/h1_solver_convergence.py`

**(A) Self-convergence**, all solutions restricted to the coarsest grid:

| grids | ‖u_N − u_2N‖_rel | observed order |
|---|---|---|
| 32 → 64 → 128 (t=0.5) | 7.41e-02, 3.71e-02 | **1.00** |
| 64 → 128 → 256 (t=0.25) | 3.34e-02, 1.67e-02 | **1.00** |

**(B) Conservation and solenoidal constraint** (every trajectory):
energy monotonically decreasing, drop 0.3–1.8%; `max|div B| / rms|B|`
between 5.6e-15 and 8.0e-14 — machine precision.

**(C) Reynolds numbers outside the training grid** {400, 800, 1200, 1600}:
Re = 200 and Re = 3200 both pass (monotone energy, div B ≈ 1.5e-14).

**(D) Localisation of the first-order behaviour** — temporal convergence at
fixed dt, with and without the projection step:

| n_steps | with projection (as in `step_full`) | without projection |
|---|---|---|
| 16 | 3.35e-03 | 3.53e-07 |
| 32 | 1.63e-03 (order 1.04) | 2.22e-08 (order 3.99) |
| 64 | 7.61e-04 (order 1.10) | 1.39e-09 (order 4.00) |
| 128 | 3.26e-04 (order 1.22) | 8.66e-11 (order 4.00) |
| **mean order** | **1.12** | **4.00** |

Direct order test of the spatial operators on a smooth periodic field:
`_fd_grad` and `_fd_laplacian` are **exactly 4th order** (4.00 at every
refinement).

**Reading — see the defect note below.** The spatial stencils and the RK4
kernel are both 4th order, but `step_full` applies a full RK4 step *then*
the divergence-free projection. That Lie splitting is first order and caps
the whole scheme at first order in time; since CFL ties dt to dx, the
space–time self-convergence is first order.

---

## Defect notes for the manuscript

**D-V4-1 (numerical, material for the methods section).** The paper
describes the solver as "fourth-order finite differences in space, RK4 in
time". Both components are verified 4th order in isolation, but the
*scheme* converges at **order ≈ 1** because the incompressibility
projection is applied as a first-order operator splitting after the
complete RK4 step (`solver.py::step_full`). Isolated, reproducible
diagnostic in T14(D). This does not invalidate the comparisons — both arms
share the solver, the runs are paired, div B is at machine precision and
all phase-1b invariants pass — but the accuracy statement must be corrected,
and any convergence claim must quote order 1.

**D-V4-2 (modelling, material for the results section).** At the deployed
size the exact ground state of the cost Hamiltonian is uniform (T11b),
the coupling terms are causally inert (T13), and the circuit realises 0.68%
of the displacement toward its own optimum (T11b). The Q-HAS decision is
therefore a small perturbation of the classical score encoding rather than
an optimisation outcome. This mechanistically explains the 0.66% composite
gain, the 109 flipped decisions with 45 correct and 64 incorrect, and the
mask asymmetry, without invoking any quantum effect.

**D-V4-3 (methodological).** A "ground state" obtained by annealing at
dim ≥ 4 is not reproducible across seeds (14–37% of patches, T12 control).
Any statement about ground-state decisions above 8 qubits requires that
floor to be reported alongside.


---

## N=256 confirmation (production resolution)

Command set: `logs/v4/v4_N256.log`. 4 scenarios, Re=400, 12 snapshots for
T11/T13, 8 for T11b, dim=2 (deployed size) and dim=8 (structured masks).

### Every conclusion holds; the numbers sharpen

| quantity | N=64 | **N=256** | verdict |
|---|---|---|---|
| exact ground state uniform | 100% | **100%** | unchanged |
| cost Hamiltonian diagonal | True | **True** (12/12 snapshots) | unchanged |
| solvers reaching certified optimum | all | **all except cold SA** | see below |
| QAOA mask = exact ground state | 1.000 | **1.000** (p=1,2,3 + shots) | unchanged |
| variational progress toward own optimum | 0.0068 | **0.0854** | still ≈ 0 |
| progress change, reps 1 → 4 | −0.117 | **−0.172** | still *decreasing* |
| ablation: remove all ZZ | 0.0000 changed | **0.0000 changed** | unchanged |
| ablation: remove all ZZZZ | 0.0000 changed | **0.0000 changed** | unchanged |
| ablation: remove Z bias | 1.0000 changed | **1.0000 changed** | unchanged |
| classical-route orbit error (dim=8) | 0.0439 | **0.0146** | improves with resolution |
| self-convergence order | 1.00 | **1.00** | unchanged |
| temporal order, projection ON / OFF | 1.12 / 4.00 | **1.12 / 4.00** | unchanged |

### T11 at N=256 — one new observation

| solver | hit optimum | E gap | mask match | F1 | wall (s) |
|---|---|---|---|---|---|
| exhaustive (certified) | 1.000 | −1.4e-17 | 1.000 | 0.389 | 0.000 |
| simulated annealing (cold) | **0.583** | 1.41e-02 | 0.583 | 0.367 | 0.139 |
| SA warm-started | 1.000 | −1.4e-17 | 1.000 | 0.389 | 0.133 |
| greedy local search | 1.000 | −1.4e-17 | 1.000 | 0.389 | 0.000 |
| classical decision alone | 1.000 | −1.4e-17 | 1.000 | 0.389 | 0.000 |
| QAOA p=1 / p=2 / p=3 | 1.000 | −1.4e-17 | 1.000 | 0.389 | 0.75 / 1.12 / 1.42 |
| QAOA p=3, 4096 shots | 1.000 | −1.4e-17 | 1.000 | 0.389 | 1.603 |

New at N=256: **cold-start simulated annealing misses the certified optimum
on 42% of snapshots** (E gap 1.4e-2) while the warm-started variant, greedy
descent and every QAOA depth reach it exactly. The optimum is trivially
reachable *from the classical decision* but not from a random start — the
landscape is flat with a narrow basin. This strengthens rather than weakens
the attribution conclusion: the only solver that struggles is the one that
does not start from the classical answer.

### T11b at N=256

| reps | progress | ‖disp‖ | ‖required‖ | ‖remaining‖ | mean marginal |
|---|---|---|---|---|---|
| 1 | +0.1588 | 0.1685 | 0.7487 | 0.6504 | 0.7739 |
| 2 | +0.1192 | 0.1569 | 0.7487 | 0.6759 | 0.7653 |
| 3 | +0.0766 | 0.1392 | 0.7487 | 0.7055 | 0.7555 |
| 4 | −0.0132 | 0.1706 | 0.7487 | 0.7662 | 0.7376 |

Mean progress 0.0854, monotonically decreasing with depth and negative by
reps = 4. The ground state is uniform on 100% of snapshots.

> **⚠ Lecture à requalifier — D-48.** Ces quatre lignes ont été obtenues avec
> `warm_start_params = classical_warm_start_params(...)`, un schedule
> **constant** `(β = 0,05 ; γ = 0,15/k)` qui, malgré son nom, ne lit ni le
> score classique ni le seuil (écart mesuré **0,0e+00** sur 6 entrées) et
> qu'**aucun chemin déployé n'emprunte**. Rejoué à la configuration publiée
> avec l'initialisation par défaut du dépôt (rampe `π/E_max` d'`execute()`,
> celle que `refinement.py` prend quand son cache est vide), 3 répétitions
> par profondeur : progression moyenne **+0,186** au lieu de +0,091, et
> tendance reps 1 → 4 de **−0,0002** au lieu de −0,116 — plate, de signe
> variable d'une répétition à l'autre. Les deux bras sont séparés aux quatre
> profondeurs.
>
> **Les nombres ci-dessus ne sont pas retirés et n'ont pas bougé** : ils
> décrivent exactement ce que le code exécute. Ce qui est en cause est la
> phrase qu'on en tire — « une progression qui n'augmente pas avec la
> profondeur signifie que l'objectif déclaré n'est pas l'objectif optimisé ».
> Mesurée, elle vaut pour ce schedule-là, pas pour le circuit. Les décisions,
> elles, ne bougent pas (0 différence sur 4 scénarios), donc les lignes T11
> `QAOA p1/p2 mask match` sont intactes. Trois options et leur coût :
> `docs/DEFAUTS.md` D-48.

> **⚠ Verdict non reproductible — D-50.** Indépendamment de D-48 : la phrase
> que T11b **imprime** bascule entre deux exécutions identiques. Le script
> tranche sur `|progress moyen| < 0,1`, seuil sans provenance, alors que
> trois exécutions de la commande publiée rendent **0,1034 / 0,0850 /
> 0,0859** — la première au-dessus du seuil, les deux autres en dessous.
> **Une exécution sur trois imprime la conclusion inverse** (« the circuit
> moves substantially toward its own optimum »). La valeur publiée 0,0854
> est à 0,0146 du seuil pour une dispersion mesurée de 0,018.
> `check_expected_behaviour` ne garde pas cette distance, et
> `aggregate_master_table` épingle la même moyenne à ±0,002, soit 9× plus
> serré que sa dispersion. Non corrigé : `VIGIL.md` demande de changer de
> **grandeur**, pas de seuil. Options : `docs/DEFAUTS.md` D-50.

### T12 at N=256

dim=8 orbit error: classical route **0.0146** (flip0 0.0078, flip1 0.0156,
rot180 0.0195, rot90 0.0156) — three times smaller than at N=64, consistent
with the one-sided-finite-difference explanation (the defect scales with
grid spacing). Ground-state route 0.4219 against a reproducibility floor of
**0.3613** → the script correctly refuses the interpretation. At dim=2 with
exact enumeration everything is 0 (uniform mask, vacuous).

### T14 at N=256 — the solver order question, settled

Self-convergence on grids 64 → 128 → 256 at t = 0.25: errors 3.344e-02 and
1.673e-02, **observed order 1.00**. Splitting diagnostic run *at N=256*
(`--split-N 256`): with projection order **1.12** (err 3.35e-03 → 3.27e-04),
without projection order **4.00** (err 3.76e-07 → 9.21e-11). Conservation:
energy monotone at every resolution, `max|div B|/rms|B|` ≤ 8.0e-14, and
Re = 200 / 3200 (outside the training grid) both pass.

**The first-order behaviour is not a low-resolution artefact.** It is
identical at N=64 and N=256, and the diagnostic isolates the cause at
production resolution: the Lie splitting between the RK4 step and the
divergence-free projection in `solver.py::step_full`.

---

## T15 — Level 3, closed-loop LOSO (audit P0, decisive experiment)

`study/closed_loop/closed_loop_campaign.py`

### Status when this entry was written: driver built, campaign not yet run

> The campaign has since run on all four folds. This entry describes the
> driver; the results are in the T15/T15b/T15c/T19/T20/T23 entries below.

The driver performs a true pipeline-level LOSO fold: for each held-out
instability class it (1) tunes the QAOA hyperparameters with Optuna on the
composite loss of the **other** classes only, reusing V1's own
`make_composite_objective`; (2) tunes the **classical** arm's AMR threshold
on the same training classes via `make_classical_composite_objective`, so
both arms suffer the identical exclusion; (3) runs both arms on the held-out
class with the same DNS trace, hot start, hybrid budget and depth. Endpoints
come from `pipeline(..., return_details=True)`: `phys_score` (relative L2 vs
DNS), `patch_ratio` (compute) and `combined`. Per-fold results are written
incrementally to JSON, so an interrupted campaign resumes.

**End-to-end validation** (`--smoke`, N=64, T_MAX=0.4): the complete path
runs to completion and writes both outputs. Smoke numbers are degenerate by
construction (both arms refine everything, delta = 0) and are not
scientific; the mode exists only to de-risk a day-long run.

### Defect found in the V1 training module (blocking for LOSO)

`train_hyperparams.SCENARIOS_ALL = SCENARIOS_ISOLATED + SCENARIOS_COMPLEX`
where `SCENARIOS_ISOLATED` already contains `ot` and `rotor` and
`SCENARIOS_COMPLEX` re-adds **the same config objects**. The list therefore
has 6 entries for **4 distinct classes**, and since the composite loss is
`mean(Loss_i)` over the list, OT and rotor are weighted 2/6 each against
1/6 for KH and tearing — an undocumented 2:1 tilt in every Phase-3 training
run. For a LOSO fold the consequence is worse: excluding `ot` would leave
its duplicate in the training list, i.e. **manufacture leakage**.
`fold_scenarios` de-duplicates by key and prints a warning. Related: the
module defines `SCENARIO_VORTEX` and `SCENARIO_COALESCENCE` (lamb_oseen,
island_coalescence) but never uses them, while its own docstring claims
Phase 1 trains on "KH, VORTEX, TEARING, COALESCENCE".

### Measured cost model (N=256, this container)

| stage | measured |
|---|---|
| DNS traces per fold (3 train + 1 held) | 225 s |
| one full `pipeline()` run at N=256 | **≥ 5 min** |
| one Optuna trial = 3 training scenarios | ≈ 15 min |

Per fold ≈ 4 min (DNS) + 15·`n_trials` min (QAOA tuning) + ≈ 6·`n_cls` min
(classical tuning) + 7 min (both final arms).

| `--n-trials` | per fold | 4 folds |
|---|---|---|
| 8 | ≈ 2.6 h | ≈ 10 h |
| 10 | ≈ 3.2 h | ≈ 13 h |
| 12 | ≈ 3.8 h | ≈ 15 h |
| 170 (protocol) | ≈ 43 h | ≈ 7 days |

**Deviation to log when the campaign runs:** the protocol freezes the V1
Optuna budget at 170 trials; a one-day campaign affords 8–12. The script
prints the deviation itself when `--n-trials < 170`. Other standing
deviations: 4 folds (the V1 module exposes 4 distinct classes, not the 8 of
protocol §1.1) and a single physics seed per fold.

### Recommended command for a one-day run

```
nohup python study/closed_loop/closed_loop_campaign.py \
      --n-trials 10 --n-trials-classical 5 \
      > logs/v4/level3.log 2>&1 &
```

Resumable: each completed fold is skipped on restart. Monitor with
`grep -E "FOLD|tuning|Q-HAS|classical\]" logs/v4/level3.log`.

### T13 with the **deployed V1 mapper** (N=256, dim=2)

The ablation above used the parameter-free V2 mapper. Re-run with the V1
mapper (`--mapper v1`, the `TRAINED_*` coefficients the pipeline actually
deploys):

| ablation | decisions changed | uniform GS | refined | F1 | n_optima |
|---|---|---|---|---|---|
| full (control) | **0.000000** | 1.000 | 0.750 | 0.333 | 64.8 |
| no_Z | 0.7500 | 1.000 | 0.000 | 0.000 | 88.0 |
| no_ZZ | **0.0000** | 1.000 | 0.750 | 0.333 | 64.8 |
| no_ZZZZ | **0.0000** | 1.000 | 0.750 | 0.333 | 64.8 |
| Z only (both couplings removed) | **0.0000** | 1.000 | 0.750 | 0.333 | 64.8 |

Same conclusion as for V2: the ZZ and ZZZZ families are **causally inert**
for the deployed Hamiltonian. Two V1-specific observations: the ground state
is uniform on 100% of snapshots but is *refine-all* on only 75% of them, and
the V1 cost function is **massively degenerate** — 64.8 of the 256
configurations are optimal on average (88 once the bias is removed).
Inspection of the coefficients explains both: at dim=2 the V1 mapper yields
median |C| = 0 with |h| ≈ 200–240, and on harris_tearing every coefficient
is zero, i.e. an identically null Hamiltonian on which the QAOA has nothing
to optimise.

---

## T15 / T15b — Level 3 closed loop: first fold, and the budget-matched reversal

### T15, fold `ot` (Orszag–Tang excluded from all tuning)

`study/closed_loop/closed_loop_campaign.py --folds ot --n-trials 4 --n-trials-classical 2`

| endpoint | Q-HAS | tuned classical | Δ (Q−C) |
|---|---|---|---|
| combined (primary) | 0.3328 | 0.4386 | **−0.1058** |
| phys_score (L2 vs DNS) | 0.1940 | 0.4845 | −0.2905 |
| patch_ratio (compute) | 0.6797 | 0.3238 | **+0.3558** |

Taken at face value this favours Q-HAS on the pre-registered primary
endpoint. **It must not be read that way**, because the two arms are not at
the same point of the error–cost frontier, and the asymmetry is inherited
from the V1 training module:

- `make_composite_objective` (QAOA arm) **hard-codes**
  `HyperParams["threshold_amr"] = 0.14959824837662078` — never suggested to
  Optuna, with the source comment "le meilleur classique";
- `make_classical_composite_objective` (classical arm) optimises
  `trial.suggest_float("threshold_amr", 0.05, 0.8)` freely and selected
  **0.4616** for this fold.

A 3× threshold difference explains the 2.1× compute gap and hence the
fidelity gap. This is exactly the "budget-matched comparison" the audit
demanded, and it is a **third defect** in the comparison design: it applies
to V1's own closed-loop numbers, not only to this fold.

### T15b, budget-matched classical arm (same fold)

`study/closed_loop/closed_loop_budget_matched.py --fold ot --max-iter 4` — bisection on the
classical threshold to reproduce the Q-HAS compute budget, everything else
(DNS trace, hot start, hybrid budget, depth) held fixed.

Classical error–cost frontier on the held-out class:

| threshold | patch_ratio | phys_score |
|---|---|---|
| 0.0500 | 0.9480 | 0.0111 |
| 0.1438 | 0.7369 | 0.0649 |
| **0.1906** | **0.6412** | **0.0827** |
| 0.2375 | 0.5866 | 0.1027 |
| 0.4250 | 0.3554 | 0.2899 |
| 0.8000 | 0.0156 | 0.5894 |
| *Q-HAS* | *0.6797* | *0.1940* |

**Budget-matched result: Δ phys = +0.1113 in favour of the classical arm.**
At *slightly less* compute (0.6412 vs 0.6797) the classical rule achieves
**2.3× lower** L2 error against DNS (0.0827 vs 0.1940). Q-HAS lies well
above the classical frontier — it is **strictly Pareto-dominated** on this
fold.

Two readings sharpen this further:
- At a *matched threshold* the conclusion is the same: classical at
  thr = 0.1438 gives phys = 0.0649 at patch = 0.7369, while Q-HAS at
  thr = 0.1496 gives phys = 0.1940 at patch = 0.6797 — 3× worse fidelity
  at comparable settings. The gap is therefore not a threshold artefact:
  the QAOA perturbation of the θ encoding actively degrades the decision
  relative to plain thresholding of the same score.
- This is coherent with T11b and T13: the circuit does not optimise its own
  cost (progress ≈ 0, decreasing with depth) and the coupling terms are
  causally inert, so the perturbation it applies carries no useful
  information.

**Pre-registered decision rules (`docs/level3_preregistration.md`).**
P1 (equivalence) is **not** supported on this fold: the arms differ, and
under budget matching the difference is large and favours the classical
rule. P3 (any fidelity gain is paid in compute) is **confirmed and then
some** — the gain does not survive paying for the compute. The
`combined`-endpoint verdict of T15 is superseded by the budget-matched
comparison, which is the interpretable one.

**Scope.** One fold (`ot`), one physics seed, 4 Optuna trials. The campaign
was interrupted twice by container reclamation while running folds `kh`,
`rotor`, `tearing`; those folds remain to be run. No claim of general
closed-loop falsification is made from n = 1. What *is* established is that
the apparent closed-loop advantage of the primary endpoint does not survive
the audit's budget-matched control on the fold measured.

---

## T17 — ZZ uncertainty window: the mechanism behind causal inertness

```
python study/h3_representation/h3_uncertainty_window.py --N 64 --steps 30
```
git hash: see `results/t17_uncertainty_window.npz`  ·  runtime ≈ 1 s
(the four DNS spin-ups dominate; N=64, 30 steps each)

**Why this task exists.** T13 established a *fact*: zeroing the ZZ family
changes 0.0000 decisions. T17 establishes the *mechanism*. The lead came
from V1's own test suite — see defect **D6** below — which contains two
failing tests asserting the opposite.

**Mechanism.** `HamiltParams.compute_coefficients` multiplies the entire ZZ
family by a Gaussian centred on the AMR decision threshold,
`w = exp(-((score - threshold_amr)/sigma)^2)`. The intent is to concentrate
coupling where the classical decision is uncertain. The effect is that the
coupling is removed from exactly the cells where it is largest: strong
gradients produce large `|C|` *and* confident (far-from-threshold) scores.

**Measurements** (four classes × two parameter sets). `no window` is
obtained by setting σ → 1e9 so that `w ≡ 1`; V1 is never modified. Mass
kept = Σ|C|·w / Σ|C|, each edge family paired with its own window.

**Three parameter sets, not two.** There are two distinct "trained" σ, and
conflating them changes the numbers by 100+ orders of magnitude:
`TRAINED_SIGMA` = **0.023** is the open-loop pipeline constant used by
phase5 and therefore by T11/T13/T18; σ = **0.1888** is what Optuna found for
the Level-3 fold `ot`, i.e. closed loop only. The deployed set is read from
the module rather than hard-coded, so it cannot drift from what runs.

> ## ⚠ RÉTRACTATION (D-58) — la lecture ci-dessous était fausse
>
> Le texte publié ici décrivait le **défaut** que `107c1cf` (D-9) a corrigé,
> pas son résultat. La fenêtre gaussienne était évaluée sur
> `physical_score` alors que le chemin déployé l'applique à
> `classical_score` (`refinement.py:506,611`, `qaoa_inputs.py:161,233`).
> D'où des masses « numériquement mortes » sur 150 ordres de grandeur.
>
> L'artefact `results/t17_uncertainty_window.npz` (`git_hash` interne
> `50ca5a0`) porte la mesure corrigée depuis. Les constantes de référence
> de `study/common/aggregate_master_table.py` ont été réancrées dessus :
> le master table passe de **164 OK / 16 DIFF** à **176 OK / 4 DIFF**.
>
> Ce qui suit est la remesure. Les affirmations rétractées sont nommées.

**Paramètres déployés en boucle ouverte** (σ = 0,023, seuil = 0,1496) — la
configuration derrière T11 / T13 / T18.

| classe | max\|C\| sans fenêtre | max\|C\| avec | masse conservée | Spearman(\|C\|,w) |
|---|---|---|---|---|
| kelvin_helmholtz | 54,51 | 35,40 | **0,1207** | −0,334 |
| mhd_rotor | 163,3 | 116,0 | **0,0332** | **+0,306** |
| orszag_tang | 64,84 | 51,98 | **0,0496** | −0,282 |
| harris_tearing | 42,32 | 16,30 | **0,0624** | **+0,140** |

**Paramètres Level-3 en boucle fermée** (σ = 0,1888, seuil = 0,1496) — le
réglage qui gouverne les folds T15.

| classe | max\|C\| sans fenêtre | max\|C\| avec | masse conservée | Spearman(\|C\|,w) |
|---|---|---|---|---|
| kelvin_helmholtz | 54,51 | 36,96 | **0,4357** | −0,334 |
| mhd_rotor | 163,3 | 125,5 | **0,5940** | +0,306 |
| orszag_tang | 64,84 | 54,26 | **0,4530** | −0,282 |
| harris_tearing | 42,32 | 18,40 | **0,3379** | +0,140 |

**Paramètres des tests V1** (σ = 0,05, seuil = 0) :

| classe | w_max | max\|C\| avec fenêtre | masse conservée |
|---|---|---|---|
| kelvin_helmholtz | 0,983 | 16,83 | 0,0243 |
| mhd_rotor | 0,999 | 40,18 | **0,4210** |
| orszag_tang | 0,842 | 22,11 | 0,0048 |
| harris_tearing | 1,000 | 1,502 | 0,0211 |

**Trois affirmations rétractées.**

1. *« ZZ is numerically dead on three of four classes »* — **faux**. La
   fenêtre conserve **3,3 % à 12,1 %** de la masse ZZ en boucle ouverte et
   **33,8 % à 59,4 %** au réglage Level-3. Aucune classe n'est morte, à
   aucun des deux réglages.
2. *« The rank correlation … is negative wherever it is defined, i.e. the
   suppression … is targeted at the strongest couplings »* — **faux**.
   Deux classes sur quatre corrèlent **positivement** (rotor **+0,306**,
   tearing **+0,140**). La fenêtre n'est donc pas systématiquement dirigée
   contre les couplages les plus forts ; le mécanisme décrit n'existe pas.
3. *« The deployed pipeline discards ~99 % of it before the QAOA ever sees
   it, which is a sufficient explanation for T13's null ablations and for
   T11b's near-zero variational progress »* — **faux, et c'est la
   rétractation qui porte le plus loin**. Elle rejette 88 % à 97 % en
   boucle ouverte, 41 % à 66 % au réglage Level-3. Surtout, **l'explication
   causale tombe** : les ablations nulles de T13 et la progression
   variationnelle quasi nulle de T11b ne s'expliquent plus par une
   annihilation de ZZ, puisqu'il n'y a pas d'annihilation. Elles restent à
   expliquer.

**Ce qui subsiste.** La fenêtre coupe réellement une part de la masse ZZ —
de moitié environ au réglage Level-3, de l'ordre de 90 % en boucle ouverte
— et `max(|C|·w) ≠ max|C|·max(w)` reste vrai : la fenêtre n'est pas grande
là où le couplage l'est. Mais « atténue » n'est pas « annihile », et la
différence est exactement ce qui séparait une observation d'une explication.

**~~Defect D7~~ — RETIRÉ.** *« The uncertainty window annihilates the family
it is meant to focus »* reposait entièrement sur les valeurs périmées
ci-dessus. Il n'y a pas d'annihilation à expliquer, donc pas d'ironie à
documenter.

**Defect D6** — inchangé, il ne dépend pas de ces nombres. `bash
run_tests.sh` ne passe pas sur un dépôt propre : rejoué dans un worktree
détaché à `cf93ba3`, 8 échecs identiques (6 × `TypeError` sur une signature
de `PhysicalMapper` qui n'existe plus, 2 assertions substantielles).

Tests: `tests/study/test_t17_uncertainty_window.py` (9).

---

## T18 — counterfactual: are the ZZ terms inert *without* the window?

```
python study/h3_representation/h3_window_counterfactual.py --N 256 --dim 2 --n-snaps 2
```
runtime ≈ 2 s (reuses the stored DNS/patch inputs) · deployed v1 mapper

**Why this task exists.** T17 shows the uncertainty window discards most of
the ZZ coupling. That immediately raises the question a referee will ask,
and the answer decides how far the paper's conclusion reaches:

> is the causal inertness of ZZ a property of the **Ising formulation**, or
> an artefact of **this implementation**?

If the window were solely responsible, the defect would be a repairable
engineering bug and the critique would not touch the approach.

**Protocol.** Two Hamiltonians per snapshot, same physics, same deployed v1
mapper: `windowed` (the pipeline as it runs) and `no_window` (σ → 1e9, so
w ≡ 1). Neutralisation is done by substituting the module constant used to
*construct* the mapper and restoring it in a `finally`; V1 is never
modified, and the substitution is asserted, not assumed (|C| without the
window must dominate |C| with it). The T13 ablations are then replayed on
each arm — `zero_hamiltonian_terms` and `ground_state_mask` are imported,
never redefined.

**Coupling amplitude at the deployed configuration** (N=256, dim=2). Note
these are *more* extreme than the N=64 figures in T17: at VQA resolution the
patch-averaged fields are smoother, so the score sits even further from the
threshold.

| class | snap | max\|C\| windowed | max\|C\| no window |
|---|---|---|---|---|
| orszag_tang | 14 | 1.33e-189 | 137.5 |
| orszag_tang | 29 | 5.65e-145 | 154.5 |
| harris_tearing | 10, 19 | **0.000e+00** | 24.89 |
| kelvin_helmholtz | 14 | **0.000e+00** | 124.2 |
| kelvin_helmholtz | 29 | **0.000e+00** | 77.32 |
| mhd_rotor | 14 | 1.25e-189 | 117.2 |
| mhd_rotor | 29 | 2.70e-200 | 143.9 |

At the deployed size the ZZ family is **identically zero in double
precision** on Kelvin–Helmholtz and Harris tearing, and at 1e-145 or below
on the others.

**Ablations, both arms:**

| arm | ablation | changed | uniform GS | refined | F1 | n_optima |
|---|---|---|---|---|---|---|
| windowed | full (control) | **0.0000** | 1.000 | 0.750 | 0.250 | 64.8 |
| windowed | no_Z | 0.7500 | 1.000 | 0.000 | 0.000 | 88.0 |
| windowed | no_ZZ | **0.0000** | 1.000 | 0.750 | 0.250 | 64.8 |
| windowed | no_ZZZZ | **0.0000** | 1.000 | 0.750 | 0.250 | 64.8 |
| no_window | full (control) | **0.0000** | 1.000 | 1.000 | 0.250 | 1.0 |
| no_window | no_Z | 1.0000 | 1.000 | 0.000 | 0.000 | 22.0 |
| no_window | **no_ZZ** | **0.0000** | 1.000 | 1.000 | 0.250 | 1.0 |
| no_window | **no_ZZZZ** | **0.0000** | 1.000 | 1.000 | 0.250 | 1.0 |

**Result.** With the coupling restored from numerically zero to O(25–155),
ablating ZZ *still* changes **0.0000** decisions; likewise ZZZZ. The
inertness is therefore **not** an artefact of the uncertainty window. It is
a property of the formulation at the deployed size: the Z bias alone fixes
the ground state, and the multi-body terms cannot move it.

This is the stronger result for the paper — it forecloses the "your
implementation was simply buggy" rebuttal. The window is a real defect
(D7), but repairing it would not make the coupling terms matter.

**A separate, subtler finding.** The window does change decisions —
**25.0 %** of them (full Hamiltonian, windowed vs neutralised) — but *not*
by acting as coupling. |C| feeds `C_scale`, the median of non-zero |C| and
|K| that sets the Z-bias amplitude `alpha_z = w_z_frac × C_scale`.
Suppressing C therefore rescales the **Z bias**, and the decision moves
through that normalisation side-channel. The coupling influences the outcome
only as an input to a scale factor — never as a coupling. Between the arms
the ground state also goes from 64.8-fold degenerate to unique.

Note the control (`full` = 0.0000) holds in both arms, so the measurement
chain is validated separately for each.

Tests: `tests/study/test_t18_window_counterfactual.py` (7), including a
positive control — the instrument is shown to detect a change when one
exists, without which "changed = 0" everywhere would prove nothing.

### T18 addendum — an *independent* counterfactual: the V2 mapper

The σ → ∞ neutralisation in T18 is a manipulation of the v1 mapper, so a
referee may reasonably ask whether the conclusion is an artefact of the
manipulation. It is not, and the repository already contained the control:

**`PhysicalMapperV2` has no uncertainty window at all.** Its own docstring
lists what was removed relative to v1: *"Removed: sigma (Gaussian
uncertainty width) … Removed: f-gate, g-gate, threshold-contrast, Gaussian
weighting"*. It is parameter-free, using plain domain-normalised ratios.

Its ZZ coupling is consequently healthy — measured at the deployed
configuration (N=256, dim=2), max|C_edges|:

| class | snap | v2 (no window) | v1 (windowed) |
|---|---|---|---|
| orszag_tang | 14 / 29 | 2.455 / 2.613 | 1.33e-189 / 5.65e-145 |
| kelvin_helmholtz | 14 / 29 | 2.774 / 2.522 | **0.000e+00** / **0.000e+00** |
| mhd_rotor | 14 / 29 | 2.017 / 2.101 | 1.25e-189 / 2.70e-200 |
| harris_tearing | 14 | 3.989 | **0.000e+00** |

And the ablations on that mapper (N=256, dim=2, `--n-snaps 3`, 72 rows):

| ablation | changed | uniform GS | refined | F1 | n_optima |
|---|---|---|---|---|---|
| full (control) | **0.0000** | 1.000 | 1.000 | 0.389 | 1.0 |
| no_Z | 1.0000 | 1.000 | 0.000 | 0.000 | 8.0 |
| **no_ZZ** | **0.0000** | 1.000 | 1.000 | 0.389 | 1.0 |
| **no_ZZZZ** | **0.0000** | 1.000 | 1.000 | 0.389 | 1.0 |

So the conclusion now rests on **two independent routes**:

1. **v1 with the window neutralised** (T18): coupling restored to O(25–155)
   → ZZ ablation 0.0000.
2. **v2, independently designed without a window** (T13, mapper v2):
   coupling natively O(2–4) → ZZ ablation 0.0000.

The second route involves no manipulation of any kind. The causal inertness
of the multi-body terms is a property of the formulation at the deployed
size, not of the v1 implementation and not of the σ → ∞ device.

**Defect D9 (in V4's own code, now fixed).** `t13_term_ablation.py` wrote
`t13_term_ablation_N{N}_dim{D}.npz` *regardless of `--mapper`*, so running
the v2 comparison silently overwrote the v1 result — precisely the
comparison the task exists to make. The filename now carries the mapper;
the historical name is still written for v1 so published references keep
resolving. Found by re-deriving the v2 numbers instead of citing them.

**Reproducibility check.** Re-running the published v1 configuration
(`--n-snaps 3`) reproduces the stored artifact **bit-exactly** across all
72 rows (`scenario`, `snap`, `ablation`, `changed`, `uniform`, `n_optima`,
`f1`, `refined`, `dE`).

### D6 follow-up — how far does the signature drift reach?

D6 reports 8 failures in the V1 suite on a clean checkout, 6 of them
`TypeError: PhysicalMapper.__init__() got an unexpected keyword argument
'beta'`. The question that matters for the paper is whether that drift
touches the code which produced the results. It does not.

Every call site, checked exhaustively:

| call site | kind | uses removed `beta=` |
|---|---|---|
| `src/pipeline.py:325` | **production pipeline** | no — current signature |
| `study/phase0_sanity_check.py:95` | study | no |
| `study/phase3_coefficients.py:68` | study | no |
| `study/phase4_exact_diag.py:68` | study | no |
| `study/qaoa_inputs.py:136` | study (feeds T11/T13/T18) | no |
| `src/compare_rotor_budget.py:110` | orphaned analysis script | **yes → dead** |
| 6 × `tests/…` | stale tests | **yes → the D6 failures** |

**Verdict.** The simulations behind every V3 and V4 number were produced by
code that constructs the mapper correctly. The drift is confined to stale
tests and to one script that nothing imports.

**Defect D10.** `src/compare_rotor_budget.py` raises `TypeError` at line 108
and cannot execute. It and `HamiltParams.py` were both last modified in
`cf93ba3` and are unchanged since; the repository has full (non-shallow)
history of 57 commits. **As committed, this script has never been runnable
in this repository.** It is referenced only by a file listing in
`README.md`. If any rotor budget-comparison figure or number in the
manuscript is attributed to it, that attribution needs checking — the script
in its committed form could not have produced it.

---

## T15 — Level-3 fold `kh` (Kelvin–Helmholtz held out)

```
bash scripts/run_fold.sh kh
```
tuning: QAOA 4 trials (best train loss 0.2590), classical 2 trials (0.3841)

| arm | combined | phys (rel. L2 vs DNS) | patch ratio | wall (s) |
|---|---|---|---|---|
| Q-HAS | 0.2443 | 0.0070 | 0.8376 | 579 |
| **classical** | **0.1800** | **0.0020** | **0.6250** | 213 |

**The classical arm wins on every endpoint simultaneously**: better fidelity
(3.5× lower L2), cheaper (25 % fewer refined pixels), and better composite.
Unlike fold `ot`, this needs **no budget-matched control** — Q-HAS is
**strictly Pareto-dominated at the tuned operating point itself**. The
budget-matched run is still executed, but only to map the frontier; it
cannot change the direction of the conclusion.

Note the training losses reproduce fold `ot`'s pattern — QAOA better than
classical on the *training* composite (0.2590 vs 0.3841 here, 0.1984 vs
0.2979 on `ot`) while losing on the *held-out* class. That is defect **D4**
in action: the QAOA arm's `threshold_amr` is pinned at 0.1496 while the
classical arm tunes its own freely, so a training-loss advantage reflects a
different operating point rather than a better decision rule.

### Cross-fold state after 2 of 4 folds

| fold | Q-HAS combined | classical combined | Δ (Q-HAS − cl) | better |
|---|---|---|---|---|
| ot | 0.3328 | 0.4386 | −0.1058 | Q-HAS |
| kh | 0.2443 | 0.1800 | **+0.0643** | **classical** |

Pre-registered readings, stated at their true scope:

- **Counting rule** (`docs/level3_preregistration.md` §4): 1–1 at n = 2.
  Neither arm meets the ≥ 3/4 threshold. **Nothing is established yet.**
- **TOST**: margin 0.0155 (5 % of mean classical `combined`, per the frozen
  formula), diff −0.0208, p_TOST = 0.520 → **equivalence not established**.
- **Difference test**: paired t p = 0.848, Holm-adjusted 1.000 → no
  significant difference. Exact sign test p = 1.000, and note the minimum
  attainable at n = 2 is 0.500 — the design cannot produce significance here
  regardless of the data.
- **Budget-matched (secondary, post-hoc):** Q-HAS dominated on 1/1 folds so
  far; `kh` is dominated already without the control.

The honest summary at this point: on the two folds measured, Q-HAS is
Pareto-dominated on both — on `ot` only after correcting the operating-point
asymmetry, on `kh` outright. The *primary* pre-registered endpoint remains
undecided by its own counting rule until 3 or 4 folds are in.

---

## T19/T20 — the Q-HAS arm is not deterministic (defect D11)

The T19 audit replays each Level-3 arm with **identical** inputs (same DNS
trace, same hot start, same hyperparameters) and checks it reproduces the
stored value. Fold `ot`:

| arm | stored `combined` | replayed `combined` | stored phys | replayed phys |
|---|---|---|---|---|
| classical | 0.4386 | **0.4386** (exact) | 0.4845 | 0.4845 |
| **Q-HAS** | 0.3328 | **0.3108** | 0.1940 | **0.1345** |

The classical arm reproducing bit-exactly proves the trace, hot start and
configuration are identical — so the variance is specific to the QAOA path.
A 44 % swing in `phys_score` between two runs of the same configuration.

**Cause.** No RNG seed is fixed anywhere in V1's VQA chain: `AerSimulator`
is built without `seed_simulator`, and both `Estimator` and `Sampler` run at
`default_shots = 256` (`create_argus`: `shots=256`, `backend="state_vector"`,
`method="COBYLA"`). The Q-HAS arm is therefore doubly stochastic:

1. the objective COBYLA minimises is a 256-shot estimate, so the optimiser
   follows a different trajectory each run;
2. the final marginal read-out is itself a 256-shot draw.

The classical arm samples nothing, hence its exact reproducibility — which
is what makes it a valid control rather than a coincidence.

**Consequence.** Every published Level-3 Q-HAS number is **one draw** from a
distribution whose spread has never been measured. `--seed` cannot fix this:
the randomness is inside V1's unseeded Aer backend, and seeding it would
require modifying V1.

**Scope of the damage — what still holds.** On fold `ot` the two observed
Q-HAS draws are phys ∈ {0.1345, 0.1940}; the budget-matched classical arm
achieves **0.0827**. Both draws are worse, so the *direction* (Q-HAS
Pareto-dominated) survives, while the *magnitude* (quoted as 2.3×) is
uncertain over roughly 1.6×–2.3×. The same caution applies to `kh`
(Q-HAS 0.0070 vs matched classical 0.0017).

**T20** quantifies the spread directly: K repeats of the Q-HAS arm on one
fold with identical inputs, plus classical repeats as a determinism control,
and reports the between-arm gap divided by the Q-HAS run-to-run standard
deviation. A gap smaller than ~2 standard deviations means a single run per
arm cannot support a directional claim on that fold.

```
python study/closed_loop/closed_loop_run_variance.py --fold kh --repeats 5
```

**This is the strongest methodological caveat in the V4 set, and it applies
to V1's own published closed-loop numbers too** — those were also single
runs of the same unseeded pipeline.

---

## T19 complete + T21 — the endpoint judgement becomes a measurement

### T19 arm audit, all four folds

| fold | Q-HAS arm | classical arm | verdict |
|---|---|---|---|
| `ot` | completed | completed | **usable** |
| `kh` | completed | completed | **usable** |
| `rotor` | completed | **ABORTED, step 208 (t=0.2739)** | **failed** |
| `tearing` | completed | completed | **usable** |

The classical arm reproduced its stored value **bit-exactly on all four
folds**; the Q-HAS arm reproduced on **none** — the D11 signature.

### T19 bisection-trace audit

| fold | aborted points |
|---|---|
| `rotor` | **2/6** — thr 0.4250 (step 371), thr 0.8000 (step 198) |
| `tearing` | **0/6** |

**A heuristic would have been wrong here.** `tearing`'s point at
phys = 4.1258 looks like a divergence and is not: it *completed*. It is a
genuine operating point at thr = 0.8, patch = 0.0727 — refine almost
nothing and the solution is badly wrong but stable. A rule such as
"phys > 1 ⇒ diverged" would have deleted a valid frontier point. The
criterion used is V1's own execution trace, never the value.

`rotor`'s two aborts also explain its fold failure: the tuned classical
threshold, 0.4616, sits inside the unstable band between 0.4250 and 0.8000.
The tuner selected an operating point that diverges on the held-out class —
a second instance of D4 doing damage.

### T21 — is the primary endpoint well posed?

Replaces the *argument* "the primary endpoint is contaminated by D4" with
three measurements, none requiring new simulation. `rotor` excluded per
pre-registration §5 (failed audit).

**1. Pareto dominance — no λ involved.**

| fold | dominates | λ-free verdict |
|---|---|---|
| `kh` | **classical** | yes |
| `tearing` | **classical** | yes |
| `ot` | incomparable | no |

**2/3 folds are decided without any λ, both for the classical arm, none
for Q-HAS.**

**2. λ crossover**, for the fold dominance cannot decide. The two arms'
`combined` cross at λ\* = (phys_c − phys_q)/(patch_q − patch_c):

- `ot`: **λ\* = 0.8164**. Q-HAS wins below, classical above. The
  pre-registered λ = 0.4 sits **below** the crossover.

**3. Count stability across λ:**

| λ | Q-HAS wins | classical wins |
|---|---|---|
| 0.0 – 0.8 | 1 | **2** |
| **≥ 1.0** | **0** | **3** |

**Correction to an earlier reading.** The "2–2 split establishes nothing"
reported before the audit **included `rotor`**, whose classical arm had
diverged and was therefore scored as a Q-HAS win. With `rotor` excluded as
pre-registration §5 requires, the primary endpoint favours the classical arm
**2–1 at the pre-registered λ**, and **3–0 for λ ≥ 1**.

At λ ≥ 1 the classical arm meets the pre-registered refutation threshold
(§4: *"If the classical arm wins on ≥ 3/4 folds … the falsification is
complete and closed-loop"*), on 3/3 valid folds.

**What this measures and what it does not.** It measures that the verdict is
partly a property of the chosen λ rather than of the arms — ill-posedness,
quantified, not asserted. It does **not** remove D4. Removing it requires
re-tuning the QAOA arm with `threshold_amr` in the search space so both arms
optimise the same free parameters: hours of compute, and the definitive
experiment.

### Figure updated

`figures_v4/pareto_panel.*` now (a) excludes `rotor`'s two aborted points
from the plotted frontier, and (b) uses a **logarithmic error axis** — the
classes span 1–3 decades, and since the compared quantity *is* a ratio, a
log axis makes a given ratio span the same vertical distance in every panel.
The full data, including excluded points, remains in the `.csv`.

**(c) The Q-HAS marker is no longer a single draw.** It plotted
`t15b["qhas"]`, one run of an unseeded arm, and annotated 2.57×, 4.41×,
3.62×, 4.38× — the retracted ratios. Anyone comparing the figure with the
corrected tables would have seen two different studies. It now plots the
**mean of the completed repeated draws with x and y error bars**
(`rotor`: 3 draws, its 2 aborted ones excluded), and falls back to the
single draw only when no repeats exist — saying so in the legend.

**The figure's ratio and the tables' ratio are different quantities.** The
figure divides by the frontier *interpolated at the budget Q-HAS actually
realised*; the tables divide by the budget-matched point T15b *measured*.
They differ because T15b matched its threshold to one draw while the plotted
point is a mean of five — on `ot`, budget 0.756 against 0.680, and the
frontier is lower there:

| fold | vs interpolated frontier (figure) | vs measured matched point (tables) |
|---|---|---|
| `ot` | 1.79× | 1.30× |
| `kh` | 2.10× | 1.90× |
| `rotor` | 2.49× | 2.74× |
| `tearing` | 1.98× | 1.81× |

Both are in `pareto_panel.csv` (`ratio` and `ratio_vs_matched`) so no reader
has to guess which one a number came from.

### D-92 — `pareto_frontier.py`, run alone, still produced the retracted ratio

**Where this was found.** 🦉 Vigil, terrain neuf : `figures/` had no entry
anywhere in `COUVERTURE.md` before this pass — the whole directory had
never been read. `pareto_panel.py`'s own docstring says each panel "reprend
exactement la grammaire de `pareto_frontier`", and it imports
`interp_frontier`/`load_points` from that file — so reading `pareto_panel.py`
without also reading `pareto_frontier.py::main()` would have been reading
half of a pair (`VIGIL.md` question 4).

**What was wrong.** The two corrections described just above — average the
Q-HAS point over T20's repeated draws instead of trusting T15b's single
non-deterministic draw, and drop T19-audited aborted points from the
frontier — were applied only inside `pareto_panel.py`'s `main()`. The
`pareto_frontier.py` module still defined its **own** `main()`
(`figures/pareto_frontier.py`, run standalone as
`python figures/pareto_frontier.py --fold X`, e.g. to produce a single-fold
supplementary figure), and that one never called `verified_qhas_point` or
`drop_aborted` — it plotted `d["qhas"]` untouched. Same shape as D-60/D-61
already logged in this file (`_add_trend`, two copies, one fixed one not):
a correction landed at one site sharing the code, not at the other.

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-92 | `pareto_frontier.py::main()`, exécuté seul, ne consultait ni `t20_qhas_run_variance_*.json` ni l'audit `t19_budget_trace_audit.json` : il traçait le tirage unique de `t15b["qhas"]` et gardait tout point de la trace, y compris ceux issus d'une bissection avortée | rejoué sur les 4 artefacts gelés (`results/t15b_budget_matched_{ot,kh,rotor,tearing}.json` + `t20_qhas_run_variance_*` + `t19_budget_trace_audit.json`) : ratio annoté **2,57× → 1,79×** (`ot`), **4,41× → 2,10×** (`kh`), **3,62× → 2,49×** (`rotor`, dont 2 points de trace avortés désormais retirés), **4,38× → 1,98×** (`tearing`) — les quatre valeurs « après » coïncident, au centième près, avec celles que `pareto_panel.py` produit déjà pour la planche V4. Le CSV mono-fold gagne une ligne `matched_classical` et le denominateur mesuré : **1,30×, 1,90×, 2,74×, 1,81×** — même colonne `ratio_vs_matched` que `pareto_panel.csv`. `verified_qhas_point`, `load_trace_audit`, `drop_aborted` déplacées dans `pareto_frontier.py` (une seule définition désormais ; `pareto_panel.py` les importe d'ici) pour que les deux fichiers ne puissent plus diverger une seconde fois de la même façon. Aucune figure `results/figures/pareto_frontier_*.{pdf,png,csv}` n'était committée avant cette passe — le nombre retracté n'était donc pas déjà publié dans ce dépôt, mais l'aurait été au prochain `python figures/pareto_frontier.py --fold X` | `pytest tests/study/test_pareto_frontier_retracted_ratio.py` |
| D-93 | **Les 17 scripts de figures V1 ecrivaient dans l'arborescence du code, pas dans le dossier de sortie du depot.** `figures/v1_legacy/fig_utils.py:109` portait deux ancres de racine pour la meme chose : `_REPO_ROOT` (ligne 8, deux niveaux) et `_PROJECT_ROOT` (un seul). La seconde etait juste tant que le fichier vivait dans `figures_code/` a la racine ; la reorganisation `17d983d` l'a descendu dans `figures/v1_legacy/` et a reecrit le prelude `sys.path` **sans** toucher a l'autre ancre. `FIG_DIR` designait des lors `figures/figures/`, cree en silence a l'import (`os.makedirs(..., exist_ok=True)`) — c'est ce silence qui rendait le defaut invisible. Trouve par la question 4 de `VIGIL.md` : deux chemins censes coincider — `figures/result_figs.py:22`, reste dans `figures/`, ancre a un niveau et **juste**, ecrit dans `results/figures/` ; `fig_utils.py`, descendu d'un cran, ancre a un niveau et **faux** | **avant** : `FIG_DIR` = `<racine>/figures/figures` (sans `FIGURE_PHASE`) et `<racine>/figures/figures/phase1` (avec `FIGURE_PHASE=1`) — les deux dossiers crees a l'import, vides, hors de tout ce que le depot lit. **Apres** : `<racine>/results/figures` et `<racine>/results/figures/phase1`, le dossier qui porte deja `fig1_ceiling_bar.png` et `fig2_loso_scatter.png`. Aucun nombre publie ne bouge : ces scripts ne calculent pas les nombres du master table (180 lignes inchangees), ils dessinent — mais toute regeneration d'une figure V1 depuis la reorganisation deposait le PNG hors du depot de figures, laissant en place l'ancienne image sans rien dire. Portee mesuree : les 17 fichiers de `figures/v1_legacy/` importent `FIG_DIR` de ce seul module ; `fig17` y met aussi son cache `.fig17_cache.json` | `pytest tests/study/test_fig_utils_output_dir.py` (6 tests, tous rouges avant) |
| D-94 | **`fig0_pareto_lambda.py` n'a pas pu produire une seule figure depuis la reorganisation : il meurt a sa premiere lecture.** Meme cause que D-93 — `PROJECT_ROOT` (ligne 39) ancre a un seul niveau, juste tant que le fichier vivait dans `figures_code/` a la racine — mais ici la consequence est fatale et non silencieuse, parce que `Train_results/` a en plus quitte la racine (`17d983d` l'a mis dans `attic/`, `12a163e` a vide l'attic). Trouve en balayant les ancres de racine des 19 fichiers de `figures/v1_legacy/` apres D-93 : deux seulement portaient encore l'ancre a un niveau, `fig_utils.py:109` (D-93) et celui-ci. Deux pieges supplementaires releves au meme endroit et desamorces : le bloc MAIN s'executait a l'**import** (importer le module relancait la campagne de figures) et il REECRIT le JSON que `JSON_PATH` designe — re-pointer naivement ce nom sur `results/hyperparams/best_hyperparams.json` aurait mis une regeneration de figure en position de muter une entree **gelee**, que son `PROVENANCE.md` donne pour le seul dossier non reproductible par une commande | **avant** : `FileNotFoundError: /home/user/BA_Proj/figures/v1_legacy/../Train_results` a `load_all_trials`, ligne 360, avant toute figure. **Apres**, meme commande (`python figures/v1_legacy/fig0_pareto_lambda.py`) : **1 combo phase/lambda quantique / 178 essais** et **3 combos classiques / 292 essais** lus depuis `results/hyperparams/optuna_studies/`, 5 PNG ecrits dans `results/figures/`, front de Pareto de **10 points**. L'ecriture JSON va desormais dans son propre artefact (`results/figures/fig0_pareto_front_quantum.json`) : mesure qui l'autorise — aucune des deux cles ecrites (`pareto_front_quantum`, `pareto_best_quantum`) n'existe dans le fichier gele, et aucun fichier du depot ne les lit. Aucun nombre publie ne bouge : ce script ne produit aucune ligne du master table | `pytest tests/study/test_fig0_pareto_paths.py` (8 tests ; sur la version d'avant : 1 echec + 6 erreurs, le module n'etant pas importable) |
| D-95 | **La figure Pareto V1 annonçait comme « Best Classical » un essai qui n'était pas le meilleur : le bras classique était tronqué à la fenêtre de score du bras quantique avant son front de Pareto et son étoile.** `v_min = min(q_scores)` servait à deux choses — l'échelle de couleur commune (son rôle) et un **filtre sur les données** classiques (`mask = (c_scores >= v_min) & (c_scores <= v_max)`), aux deux sites `plot_pareto_scenario` et `plot_grouped_pareto`. Le biais est structurel et à sens unique : puisque `v_min` est le minimum quantique, **tout essai classique meilleur que TOUT le quantique tombe hors fenêtre par construction**. Même famille que D-81/D-82/D-83 — un choix de traitement qui favorise le bras que l'étude cherche à falsifier. Trouvé en lisant `fig0_pareto_lambda.py` en entier après D-94 (question 2 de `VIGIL.md` : la docstring de `plot_pareto_scenario` promet « quantum + classical Pareto front », pas un classique tronqué) | rejoué sur les CSV gelés (`results/hyperparams/optuna_studies/`, λ = 0,40), « Best Classical » lu dans la légende de la figure produite — **avant → après** : `kelvin_helmholtz` **S = 0,306590 → 0,129020** (56 essais sur 172 jetés sous la fenêtre, le vrai optimum classique est **2,4× meilleur** que celui annoncé ; front classique 169 points → 45 avant, 169 après), `orszag_tang` **0,348250 → 0,326180** (47 jetés), `mhd_rotor` **0,192481 → 0,183508** (6 jetés), `harris_tearing` **0,254429 → 0,254429** — aucun essai sous la fenêtre : **le scénario qui NE SÉPARE PAS**, un test écrit sur lui seul serait passé sans rien vérifier (5 points lui étaient tout de même retirés en haut). L'échelle de couleur commune se prend désormais sur la réunion des deux bras, sinon les points restitués saturent. Aucun nombre publié ne bouge : aucune de ces figures n'était committée, et ce script ne produit aucune ligne du master table — mais la prochaine régénération publiait un classique 2,4× pire que le vrai | `pytest tests/study/test_fig0_classical_truncation.py` (11 tests ; 7 rouges sur la version d'avant) |
| D-96 | **`fig15_decision_flip_analysis.py`, `fig16_decision_landscape.py`, `fig17_topological_attribution.py` — régression de D-37, dans leur propre copie de la traversée BFS.** Les trois fichiers réimplémentent localement `instrumented_bfs`/`instrumented_bfs_hamilt` au lieu d'appeler `_run_level`/`_run_level_classical` de `refinement.py`, et copiaient tous les trois l'appel fautif que D-37 a déjà corrigé dans le chemin canonique : `_process_score(local_score_raw, is_periodic, target_dim + 2*pad)`. `_process_score` emprunte `_resize_padded_maxpool`, dont le contrat est « entrée (N+2,M+2) → sortie (t_dim+2,t_dim+2) » : le halo est déjà ajouté. Demander `target_dim+2` (donc `t_dim=4` pour `target_dim=2`) rend un cœur (4,4) après le trim `[1:-1,1:-1]`, et la boucle `for i in range(target_dim)` n'en lit que le quart HAUT-GAUCHE. Trouvé par la question 4 de `VIGIL.md` : `classical_score[i,j]` (ce chemin) et `qaoa_prob[i,j]` (correctement dimensionné en `target_dim`, lui) finissent par décrire deux régions différentes du même patch dès `depth > 0` | **avant/après**, patch `depth=1` sur `init_harris_tearing` (N=256, 30 pas, bounds=(0,128,0,128)) : `classical_score` par cellule **[[0.0617, 0.5405], [0.0470, 0.5260]] → [[0.5405, 0.5826], [0.5722, 0.6564]]** — écart max **0.525** sur une échelle max **0.656** (80 %). Avec `threshold_amr=0.3228` (`load_hyperparams(method='classical')`), 2 des 4 décisions binaires `score >= threshold_amr` basculaient (cellules (0,0) et (1,0) : sous le seuil → au-dessus) — exactement le comptage que `fig15_decision_flip_analysis.py` existe pour produire. Aucun nombre publié n'en dépend : aucune figure `results/figures/fig1{5,6,7}_*` n'est committée dans ce dépôt | `pytest tests/study/test_v1_legacy_instrumented_bfs_score_grid.py` (7 tests ; 3 rouges sur la version d'avant) |
| D-97 | **`fig9_synthetic_unit_tests.py` : trois des quatre générateurs de champ physiquement motivés (`make_vortex_core`, `make_current_sheet`, `make_xpoint`) échangeaient X et Y dans le dépaquetage de `np.mgrid`/la diffusion du broadcast, contre la convention du dépôt (`grid.py` : `AXIS_X=0`, `AXIS_Y=1`, `indexing='ij'`).** Le champ magnétique n'était alors pas à divergence nulle **par construction**, et rien ne le nettoie ensuite (`MHDSolver.PROJECT_B = False`, vérifié — seul `vx`/`vy` est reprojeté). Trouvé en lisant `fig9` en entier après les trois questions déjà posées à `fig_utils.py`/`fig0` : la question 3 de `VIGIL.md` — la fonction consomme-t-elle ce que son nom (« Vortex Core », « Harris-like current sheet at x=N/2 », « Reconnection X-point ») annonce ? — pas une MHD valide, en l'occurrence. Référence de comparaison : `init_harris_tearing()`, déjà correct (`Bx = f(Y)` seul, `By` dérivé d'une fonction de flux) | **avant → après**, mesuré à la construction (avant toute évolution), même opérateur que le dépôt (`Simulation.grid.divergence(Bx, By, fixed_curl=True)`), N=256 : `vortex_core` **max\|div B\| = 0,0245 → 0** (échelle de champ 0,50, soit 4,9 % → 0 %) ; `current_sheet` **2,0000 → 0** (échelle 1,00, soit 200 % → 0 %, persistait à 96 % après 20 pas d'évolution) ; `xpoint` **3,1750 → 0** (échelle 1,50, soit 212 % → 0 %, persistait à 127 % après 20 pas). Les trois deviennent bit-à-bit nuls après correction, comme `init_harris_tearing` (référence, 1e-4 de bruit FD4). `make_uniform_noise` (contrôle négatif, non structuré) n'a pas ce défaut — hors périmètre ici, voir D-98 pour son propre défaut. Aucun nombre publié ne bouge : aucune figure `results/figures/fig9_*` n'est committée dans ce dépôt | `pytest tests/study/test_fig9_synthetic_fields_solenoidal.py` (11 tests ; 3 rouges sur la version d'avant, ciblés sur le comportement du fichier committé, pas sur son texte source) |
| D-99 | **`fig3_spatial_coherence.py` mesurait la cohérence spatiale comme si le domaine était borné — il est périodique.** Les deux métriques de la figure traitaient le bord du domaine comme un bord réel : `compactness` remplissait le pourtour avec « rien n'est raffiné » (`np.pad(..., mode='constant', constant_values=False)`) et `component_density` appelait `label()` sans refermer les bords. Or `PeriodicGrid` referme le domaine : le bord haut a le bord bas pour voisin. Toute structure traversant un bord — une nappe de courant, une couche de cisaillement, tout ce que l'AMR sélectionne justement — était donc comptée comme exposée des deux côtés, et coupée en deux composantes. Les deux bras en souffrent, mais **pas également** : celui qui sélectionne le plus de régions touchant un bord est le plus pénalisé, sur la figure dont c'est précisément l'objet. Trouvé après le constat que les métriques homonymes de `fig_utils.py` sont mortes (aucun appelant) — `fig3` en porte ses **propres** copies, elles vivantes | rejoué sur des masques N=256 construits pour séparer, **avant → après** : bande verticale traversante (type nappe de courant) compacité **0,0698 → 0,0625** (**+11,7 %** de trop) ; bloc à cheval sur le bord haut/bas **0,1211 → 0,0918** (**+31,9 %**) et **2 composantes → 1** (`component_density` doublait pour cette région) ; domaine **entièrement** raffiné **0,0156 → 0,0000** — en périodique il n'existe aucun pixel de bord, le cas qui tranche le plus nettement. Bloc central, ne touchant aucun bord : **0,0784 → 0,0784**, inchangé — c'est le champ qui NE SÉPARE PAS, gardé comme test pour que personne ne croie avoir validé la correction dessus. Aucun nombre publié ne bouge : aucune figure `fig3_*` n'est committée et ce script ne produit aucune ligne du master table | `pytest tests/study/test_fig3_periodic_coherence.py` (7 tests ; 4 rouges sur la version d'avant) |
| D-101 | **`fig5_qaoa_detailed_analysis.py::_gt_quadrant_above_threshold` comparait le champ d'erreur GT brut à `threshold_amr`, calibré pour une tout autre grandeur — le diagnostic ne pouvait jamais rendre un vrai positif.** `gt` vient de `ground_truth_errors` : magnitude gradient+laplacien non normalisée, échelle propre à chaque scénario. `threshold_amr` (~0,30) est calibré exclusivement contre `AngleMapper.classical_score`, normalisé au max du domaine dans [0,1] — chaque autre usage de `threshold_amr` dans le dépôt (`HamiltParams.py`, `refinement.py`, `cost_hamiltonian.py`) le compare à ce score normalisé, jamais à un champ d'erreur brut. Question 4 de `VIGIL.md` : `_gt_error_share`, dans le même fichier, compare déjà `gt` à lui-même correctement — deux fonctions voisines, deux conventions | **mesuré**, `init_harris_tearing` (N=256, 150 pas) : `gt.max()` sur tout le domaine = **0,183**, jamais supérieur à `threshold_amr` (~0,30) — `gt_above` valait **FAUX partout, systématiquement**, quel que soit le scénario ; le diagnostic « Decision quality » (TP/FP/FN) imprimé ne pouvait donc **jamais** afficher un vrai positif. **Avant → après** sur ce champ : `gt_above` **[[False,False],[False,False]] → [[True,False],[False,True]]** — le champ qui sépare (croix asymétrique) redevient discriminant. Correction : `gt` comparé à sa propre moyenne (`gt.mean()`), comme `_gt_error_share` et `pixel_precision`/`pixel_recall` de `fig4_comprehensive_comparison.py`. Portée : diagnostic imprimé sur la console uniquement, aucun nombre publié n'en dépend | `pytest tests/study/test_fig5_gt_threshold_scale.py` (4 tests ; 2 rouges sur la version d'avant) |
| D-102 | **`fig15_decision_flip_analysis.py` citait en dur, dans son texte « ROOT CAUSE », le σ d'un AUTRE module que celui qu'il utilise réellement.** Le fichier construit son `HamiltMapper` via `_hamilt_mapper_kwargs` (`fig_utils.py`), dont le repli pour `sigma` vaut `TRAINED_PARAMS.get('sigma', 0.05)` — 0,05, jamais échantillonné (`'sigma'` absent de `results/hyperparams/best_hyperparams.json`, D-22). Le bloc CONCLUSION (imprimé quand `flip_rate < 0.05` et `mean_ratio < 0.5`) affichait pourtant « With σ=0.023 (trained) » : 0,023 est `TRAINED_SIGMA`, une constante de `study/pipeline/config.py` — un pipeline fermé, distinct, que ce fichier n'importe pas. Question 4 de `VIGIL.md` : deux chemins censés décrire le même « sigma trained » ne coïncidaient pas — l'un mesuré (le repli réellement utilisé ici), l'autre copié d'ailleurs | **mesuré** : `'sigma' not in TRAINED_PARAMS` → `TRAINED_PARAMS.get('sigma', 0.05)` vaut inconditionnellement **0,05** dans ce fichier, jamais 0,023. Le rayon « essentiellement nul au-delà de » annoncé en dépendait : `~0,05` (dérivé de 2×0,023) devient **~0,10** (2×0,05) — le rayon où une correction QAOA peut encore faire basculer une décision est en réalité le double de ce que le texte affirmait. **Avant → après** : `"ROOT CAUSE: With σ=0.023 (trained)..."` texte fixe → `f"ROOT CAUSE: With σ={sigma_trained:.3f} (TRAINED_PARAMS fallback...)"`, calculé depuis `TRAINED_PARAMS.get('sigma', 0.05)`. Portée : texte de diagnostic imprimé sur la console uniquement (aucune figure `fig15_*` n'est committée dans ce dépôt), aucun nombre publié n'en dépend | `pytest tests/study/test_fig15_sigma_narration.py` (4 tests ; 2 rouges sur la version d'avant) |
| D-103 | **Le garde-fou de D-64 (`test_deletion_happens_only_in_this_script`) balayait tout le système de fichiers avec `grep -rn`, pas seulement le dépôt suivi par git — un `.venv/` local (anticipé par le `.gitignore` du dépôt : `.venv/`, `.venv_vigil/`, `env/`) fait échouer le test sur le propre code d'`optuna`, pas sur celui du dépôt.** `optuna` définit et appelle `delete_study` à une douzaine d'endroits dans son propre code installé ; `grep -rn` sur la racine du dépôt les traverse tous, `git grep` (fichiers SUIVIS seulement) aucun — c'est exactement « le dépôt » que ce test veut dire. Trouvé en établissant la ligne de base complète de cette passe (`pytest tests/ -q -m "not slow"`) : 4 échecs au lieu des 2 déjà connus (variance QAOA, `test_noise_robustness`/`test_hyperparameter_sweep`) | **mesuré** : `git grep -n delete_study -- '*.py'` sur ce dépôt à HEAD → 3 occurrences, toutes dans `src/import_Neon_data_to_local.py` (identique à avant, comportement inchangé sur le dépôt réel). Sur un mini-dépôt synthétique portant un fichier suivi + un `.venv/site-packages/optuna_stub.py` non suivi contenant `delete_study` : **avant** (`grep -rn`) remonte les deux, le test échoue ; **après** (`git grep`) ne remonte que le suivi. `test_the_ranking_survives_the_sampling` (QAOA, `tests/quantum/test_qaoa_arm_is_sampled.py`) a aussi échoué sur cette même exécution (médiane de rang 0,450 sur 10 paires, seuil 0,6, référence 0,883) — rejoué isolément 3 fois, seul : **3/3 passe** (aucune valeur numérique reconsignée, les trois rejeux n'ont pas ré-échoué), cohérent avec la variance déjà documentée (`VIGIL_BA_Proj.md`, dispersion de rang 1,79e−1 à 3,61e−1 sur 45 paires) — pas une régression | `pytest tests/pipeline/test_import_never_destroys_destination.py` (5 tests ; 1 rouge sur la version d'avant, reproduit uniquement avec un `.venv/`/`env/` local présent) |
| D-104 | **`fig7_physical_fidelity.py` : le bloc « Add tiny perturbation for trial independence » tirait un bruit DIFFÉRENT pour chacune des trois simulations que le commentaire trois lignes plus haut déclare identiques (« Create 3 identical sims »).** Un seul `rng`, consommé dans la boucle `for lbl in sims` : `dns`, `qaoa` et `classical` recevaient chacun leur propre tirage. Or tout ce que la figure rapporte — la courbe « Rel. L2 Error » `field_l2_error(sims['qaoa'], sims['dns'])`, l'énergie cinétique et l'enstrophie des trois bras — suppose que le seul écart entre elles vient de la décision d'AMR. Dès `trial = 1`, la courbe mesurait la divergence de deux conditions initiales différentes. Trouvé par la question 1 de `VIGIL.md` (pourquoi ce bloc existe-t-il ? `N_TRIALS = 1` : il ne s'exécute jamais — un piège armé) et la question 5 (aucune configuration du dépôt n'emprunte `trial > 0`) | **avant → après**, `init_harris_tearing`, N=256, warmup=80, 3 pas d'AMR, `trial=1`, mêmes seuils (`TRAINED_PARAMS`/`CLASSICAL_PARAMS`) : `L2(qaoa, dns)` à t=0, avant tout AMR **1,4122e-05 → 0,0** (attendu 0 par construction) ; après le warmup, toujours avant le premier pas d'AMR, **2,020e-06 → 0,0** ; après 3 pas d'AMR **1,6795e-05 → 8,695e-07**, contre **8,182e-07** pour le même calcul à `trial=0` (sans perturbation) — soit **× 20,5** sur l'erreur annoncée. Écart entre les deux bras au même pas : **1,6795e-05 (Q-HAS) contre 2,104e-06 (classique), × 8,0**, là où à `trial=0` les deux rendent des valeurs **bit-à-bit identiques** (8,182e-07). Aucun nombre publié ne bouge : aucune figure `results/figures/fig7_*` n'est committée dans ce dépôt | `pytest tests/study/test_fig7_trial_perturbation_shared.py` (7 tests ; **5 rouges** sur la version d'avant, dont un qui épingle l'ancien comportement et sa mesure 1,4122e-05) |
| D-105 | **`fig7_physical_fidelity.py` imprimait la dispersion d'un essai unique comme une mesure, à une échelle où la grandeur ne s'écrit pas.** Le fichier tourne à `N_TRIALS = 1` et annonçait « Multiple trials with error bands for statistical confidence ». Deux effets : (a) `np.std` sur un échantillon unique vaut **0,0 sans avertissement** — avec `ddof=1` la même quantité vaut `nan` et prévient — donc `+/-0.000000` était imprimé et `fill_between(x, mu-0, mu+0)` tracé, une dispersion *jamais mesurée* indiscernable d'une dispersion *mesurée nulle* ; (b) le format `%.6f` sur une grandeur de l'ordre de 1e-06 rendait les deux bras indiscernables, alors que la colonne correspondante de la figure est tracée en `set_yscale('log')` — le résumé imprimé contredisait l'échelle de son propre axe (question 4 de `VIGIL.md`). Même famille que D-55/D-56 : un balayage qui n'a rien mesuré doit être discernable d'un balayage réussi | **avant → après**, `init_harris_tearing`, N=256, warmup=80, 3 pas d'AMR, `trial=0`, seuils du dépôt — vraie valeur mesurée `l2_qa = l2_cl = ` **8,182e-07** : ligne imprimée **`QA=0.000001+/-0.000000` → `QA=8.1819e-07 (1 essai, dispersion non mesurée)`** ; chiffres significatifs sur la valeur **1 → 5** ; largeur de bande à 1 essai **0,0 tracée → pas de bande**. À `n ≥ 2` la bande et la dispersion (`ddof=1`) reviennent. Le calcul physique est inchangé ; aucun nombre publié ne bouge (aucune figure `results/figures/fig7_*` n'est committée) | `pytest tests/study/test_fig7_single_trial_dispersion.py` (6 tests ; **4 rouges** sur la version d'avant, dont deux qui épinglent l'ancienne ligne `0.000001+/-0.000000` et le fait que `ddof=0` ne prévient pas) |
| D-106 | **`fig12_depth_analysis.py`, panneau C : le « taux d'accord » entre Q-HAS et le bras classique divisait par le domaine ENTIER.** `_agreement_by_depth` construit deux masques booléens par profondeur (patchs non `coarse_leaf` de cette profondeur) puis renvoyait `np.sum(qa_mask == cl_mask) / (N*N)`. À une profondeur donnée, presque aucun pixel du domaine ne porte de patch : tous ceux que ni l'un ni l'autre bras ne touche vérifient `False == False` et comptaient comme un **accord**. Le taux mesurait la proportion de domaine vide, pas la proportion de décisions concordantes. Un second chemin menait au même faux 100 % : le repli `np.mean(all_agreement[d]) * 100 if all_agreement[d] else 100` du tracé. La note du panneau attribuait explicitement le résultat à la physique (« High agreement (>90%) is expected — most BFS decisions are far from the threshold ») — c'était un artefact du comptage. Même famille que D-98 (un contrôle négatif qui ne peut pas échouer) | **avant → après**, `target_dim=2`, `min_size=6`, `solve_max_depth=5`, seuils du dépôt. `init_harris_tearing`, N=256, 300 pas : profondeurs 0/1/2/3/4 = **100,00 % → indéfini** (union vide) — à la profondeur 4 le bras Q-HAS porte **62 patchs** et le classique **0**, le désaccord structurel maximal, annoncé comme un accord parfait ; profondeur 5 (union 25 % du domaine) **85,35 % → 41,41 %**, la barre passe de l'ambre (`> 85`) au rouge. `init_orszag_tang`, N=256, 500 pas : profondeurs 0 à 4 **100,00 % → indéfini** ; profondeur 5 (union 6,25 %) **95,02 % → 20,31 %**, la barre passe du **vert** (`> 95`) au rouge. Aucun nombre publié ne bouge : aucune figure ni log `fig12_*` n'est committé (`git ls-files results/figures/` ne rend que `fig1_ceiling_bar.png` et `fig2_loso_scatter.png`) | `pytest tests/study/test_fig12_agreement_denominator.py` (9 tests ; **6 rouges** sur la version d'avant, dont trois qui épinglent l'ancien calcul et deux qui épinglent les nombres mesurés) |
| D-107 | **`fig5_qaoa_detailed_analysis.py` : la profondeur 0 de sa courbe de coefficients empruntait un `dx` 128 fois plus petit que toutes les autres, et rendait un Hamiltonien identiquement nul — attribué à la physique dans la docstring du module.** `analyze_vqa_at_patch` passe `dx_override` à `compute_coefficients`. Pour un patch, la taille de cellule VQA `(patch_size/N)·L/target_dim` — **la même formule que `refinement._run_level`** (`dx_eff = patch_phys_size / target_dim`). Pour le domaine complet (`bounds is None`, la profondeur 0 de la courbe) : `None`, donc `grid.dx = L/N`, le pas de la grille FINE. `compute_coefficients` en tire `Re_cell = |v|·dx/ν` : à `grid.dx`, aucun `Re_cell` n'atteint `RE_CRIT` et les trois blocs sortent à zéro. Question 4 de `VIGIL.md`, et « mesurer avec l'opérateur assorti ». Le récit qui cachait le défaut est dans la docstring : *« deeper patches where the effective dx is small enough to trigger physical thresholds »* — `dx_eff` **décroît** avec la profondeur (3,14159 / 1,57080 / 0,78540 / 0,39270), donc les patchs profonds franchissent **moins** les seuils, pas plus | **avant → après**, `init_harris_tearing`, N=256, 300 pas, `target_dim=2`, `threshold_amr` du dépôt, `advanced_anomalies_enabled=True` : profondeur 0, `dx` **0,024544 → 3,141593** (×128), `Σ|H_edges|` **0,000000 → 0,238940**, `Σ|C_edges|` **0,000e+00 → 1,6579e+05**, `Σ|K_plaquettes|` **0,000e+00 → 8,3461e+04**. Profondeurs 1/2/3 inchangées (0,038764 / 0,003722 / 0,001872 pour H). **La lecture s'inverse** : la courbe passe d'« un zéro à la profondeur 0 puis une bosse » à une **décroissance monotone** où la profondeur 0 porte les coefficients les plus forts. Aucun nombre publié ne bouge (aucune figure `results/figures/fig5_*` n'est committée) ; la sortie de `fig5` elle-même change, c'est l'objet de la correction | `pytest tests/study/test_fig5_depth0_cell_size.py` (5 tests ; **4 rouges** sur la version d'avant, dont un qui compare fig5 à la convention de `refinement.py` à quatre profondeurs) |
| D-108 | **`scripts/extract_best_hyperparams.py` jetait en silence la colonne de paramètre qu'il venait lui-même de détecter.** `_detect_param_cols` teste `'param_beta_grad' in header` puis renvoie `PARAM_COLS_SPLIT` — le jeu de la génération `sigma`, qui **ne contient pas** `param_beta_grad`. La génération de campagne que la fonction reconnaît explicitement perdait donc son 9e paramètre, et le JSON écrit portait 8 paramètres là où le CSV source en portait 9. Rien ne le signalait : ni exception, ni avertissement, ni écart de forme — un JSON à 8 paramètres est valide. Trois générations de noms coexistent dans les artefacts (`beta_michelson` → `beta_grad` → `sigma`) et chaque renommage rejouait le même piège. Même famille que D-55/D-56 : un balayage qui n'a rien mesuré doit être discernable d'un balayage réussi. **La lecture publiée que cette entrée corrige est dans `D-22`** : la forme à 8 paramètres du JSON déployé est un *produit de ce défaut*, pas un témoignage de ce que la campagne a échantillonné — détail dans `docs/DEFAUTS.md` | **avant → après**, sur les artefacts gelés `results/hyperparams/optuna_studies/GOOD_RESERVE/GOOD_reserve_v2/before_halo_fix/` : **4 campagnes quantiques à 9 colonnes `param_*`** (`phase1b`, `phase1b_agr`, `phase2`, `phase3`), **579 valeurs `beta_grad` échantillonnées jetées**, étendue **0,100000 à 2,000000**. Sur `rescore_q_has_v2_phase1b_lambda0.4000`, meilleur essai (81, `new_score` 0,21448910814585587) : paramètres écrits **8 → 9**, `beta_grad` **absent → 1,744060606058018** ; l'essai retenu, son score et les 8 autres valeurs sont **inchangés bit à bit**. Portée vérifiée sur la campagne vive (génération `sigma`) : sortie **identique** avant/après (`beta`, `beta_curl`, `beta_xpoint`, `sigma`, `w_z_frac` ; bras classique `threshold_amr` seul) — la correction n'ajoute que ce qui est dans le CSV. **Aucun nombre publié ne bouge** : `results/hyperparams/` est une entrée gelée, non régénérée ici (voir son `PROVENANCE.md`) ; `python study/common/aggregate_master_table.py` reste à 180 / OK=164 / DIFF=16 / MISSING=0. **Hypothèse mesurée et RÉFUTÉE contre moi-même** : la ligne de réservation annonçait que ce paramètre perdu était « tout ce qui sépare le JSON déployé de sa base d'origine ». Faux — balayage des 13 CSV du dépôt : **aucune ligne, nulle part, ne partage ne serait-ce qu'une seule valeur de paramètre ni le score avec le JSON déployé**. Le JSON reste orphelin, D-22 tient sur ce point | `pytest tests/pipeline/test_extract_best_hyperparams_columns.py` (12 tests ; **5 rouges** sur la version d'avant, dont un qui épingle la cause — `param_beta_grad ∉ PARAM_COLS_SPLIT` — et un bout-en-bout sur le vrai artefact gelé) |
| D-109 | **`scripts/extract_best_hyperparams.py` : le bloc `per_scenario` annonce l'optimum de chaque scénario et ne le cherchait que parmi les 3 meilleurs essais du score AGRÉGÉ.** `parse_rescore_dir` ne rendait que les `top_k` meilleurs essais, et `main()` en faisait `all_quantum_trials` : le nom promettait tous les essais et en portait trois. `_pick_best_for_scenario`, dont la docstring dit *« Among all trials »*, ne pouvait donc désigner l'optimum d'un scénario que dans 3 essais sur 178 — question 3 de `VIGIL.md` (la fonction ne consomme pas ce que sa signature annonce) et question 4 (deux chemins censés coïncider : le `--top-k` qui borne la **donnée brute** écrite dans `training_phases`, et la **sélection**, qui n'a jamais eu à l'être). La valeur écrite restait plausible : un `phys` du bon ordre, du bon signe, issu d'un vrai essai — simplement pas le minimum annoncé. Le symptôme est lisible dans le fichier déployé lui-même : ses **quatre** optima « par scénario » quantiques sont un seul et même essai, le 85, qui est aussi le `default` | **avant → après**, campagne vive `results/hyperparams/optuna_studies/` (178 essais quantiques), `--lambda-cost 0.40`, `--top-k 3`. Bras quantique, `phys` écrit : `kelvin_helmholtz` **0,003604662 → 0,0013197164** (essai 30 → 13, **×2,7**) ; `harris_tearing` **0,0044288611 → 0,0024402795** (essai 86 → 8, **×1,8**) ; `orszag_tang` **0,063719323 → 0,061306123** (essai 4 → 157) ; `mhd_rotor` **0,055825707 → 0,055825707** (essai 30, coïncide — ce scénario ne sépare pas les deux hypothèses). Bras classique : `harris_tearing` **0,0044288611 → 0,0039676837**, `orszag_tang` **0,013679399 → 0,011106421**, `mhd_rotor` **0,048986229 → 0,02698116** (**×1,8**) ; `kelvin_helmholtz` inchangé. **6 entrées sur 8 changent.** Portée vérifiée : `default` (le seul bloc que `pipeline.py` consomme sans argument), `best_per_phase` et `training_phases` sont **identiques** avant/après — la correction élargit la sélection, pas le fichier écrit. **Aucun nombre publié ne bouge** : `results/hyperparams/best_hyperparams.json` est une entrée gelée, non régénérée ici (son `PROVENANCE.md`) ; `python study/common/aggregate_master_table.py` reste à 180 / OK=164 / DIFF=16 / MISSING=0 | `pytest tests/pipeline/test_extract_best_hyperparams_selection.py` (16 tests ; **10 rouges** sur la version d'avant, dont 6 sur la garantie annoncée aux deux bras et 3 qui épinglent les nombres d'avant) |
| D-110 | **`run_tests.sh` — le lanceur que `README.md` documente comme LA suite du dépôt (« `bash run_tests.sh  # Run full test suite` ») et que `docs/protocol_v3_evaluation.md` érige en critère d'acceptation (« `bash run_tests.sh` must pass unchanged after every task ») — ne pouvait plus atteindre un seul de ses étages.** La réorganisation de `tests/` (`17d983d`) a déplacé les fichiers vers `solver/`, `mapping/`, `quantum/`, `amr/`, `pipeline/`, `study/`, `tools/`. Les imports croisés ont été rattrapés (D-71, puis `test_suite_integrity.py`) ; **les lanceurs `.sh`, non**. Le critère d'acceptation était donc mort depuis la réorganisation, sans que rien ne le dise : tout le monde lance `pytest tests/ -q -m "not slow"`, la recette de `CLAUDE.md`, et personne n'appelle plus le lanceur qui porte la garantie. Même famille que D-71 et D-76 (chemins morts après déplacement) | **avant → après**, mesuré en exécutant chaque commande d'étage. **Avant : 17 commandes distinctes sur 17 échouent**, toutes sur `file or directory not found` — rc **4** pour les 12 appels `pytest`, rc **2** pour les 5 scripts autonomes. `run_stage` sort au premier code non nul (`run_tests.sh:154`) et le script est `set -e` : il s'arrêtait à l'**étage 1 sur 17**. **0 test atteignable.** **Après : 0 étage sans cible**, **168 tests sélectionnés** (15 + 7 + 49 + 7 + 1 + 9 + 27 + 6 + 13 + 30 + 2 + 2) et les 5 scripts autonomes résolvent. 32 appels `run_stage`, 32 chemins réécrits, aucun fichier de test ni de `src/` touché. **Aucun nombre publié ne bouge** : le lanceur ne produit aucun artefact | `pytest tests/test_launcher_paths_resolve.py` (52 cas ; **33 rouges** sur la version d'avant de `run_tests.sh`) |
| D-111 | **`scripts/generate_figures_v1.sh` calculait une racine de dépôt décalée d'un niveau : `ROOT_DIR="$SCRIPT_DIR"` était juste tant que le script vivait à la racine, et désigne `<dépôt>/scripts` depuis son déplacement dans `scripts/`.** Conséquence directe : `$ROOT_DIR/scripts/extract_best_hyperparams.py` résolvait en `scripts/scripts/extract_best_hyperparams.py`, et `--output $ROOT_DIR/best_hyperparams.json` aurait écrit dans `scripts/`. Même cause que D-110 — un déplacement dont la racine n'a pas suivi. **Corrigé pour la racine seule ; le reste est signalé, non corrigé** : trois cibles (`figures_code/`, `Train_results/`, le `best_hyperparams.json` racine) n'existent plus sous AUCUNE racine, et les rebrancher est une décision — la dernière écraserait `results/hyperparams/best_hyperparams.json`, entrée **gelée** et non reproductible (son `PROVENANCE.md`, et D-22). La note est écrite dans l'en-tête du lanceur lui-même, avec sa mesure, et un test vérifie qu'elle y reste. **Mesuré et volontairement NON corrigé aussi** : `scripts/run_study_v3.sh` porte le même décalage en sens inverse — son `ROOT_DIR="$SCRIPT_DIR/../.."`, juste quand il vivait dans `study/v3/`, désigne aujourd'hui le **parent du dépôt**, donc ses 11 invocations sortent de l'arborescence. Son en-tête dit « ne pas debugger les chemins en pensant le remettre en état sans lire D-49 d'abord » : un gel documenté ne se corrige pas au passage | **avant → après**, résolution des chemins de `scripts/generate_figures_v1.sh` : `$ROOT_DIR/scripts/extract_best_hyperparams.py` **`scripts/scripts/extract_best_hyperparams.py` (inexistant) → `scripts/extract_best_hyperparams.py` (existe)** ; `$ROOT_DIR/best_hyperparams.json` **`scripts/best_hyperparams.json` → `best_hyperparams.json`** (racine — toujours inexistant, c'est le point laissé à trancher) ; `figures_code/` et `Train_results/` restent introuvables sous les deux racines. Pour `run_study_v3.sh`, mesuré sans correction : **11 invocations sur 11** résolvent hors du dépôt (`../study/…`). **Aucun nombre publié ne bouge** : aucun de ces deux lanceurs ne tourne aujourd'hui, donc aucun artefact n'en dépend | `pytest tests/test_launcher_paths_resolve.py` (**83 cas**, dont 79 invocations paramétrées ; sur la version d'avant de `generate_figures_v1.sh`, **2 rouges**) |
| D-111 *(remesure)* | **Le reste laissé ouvert par D-111 a été tranché ailleurs, et les deux tests qui le gardaient sont devenus des seuils périmés.** `D-116`/`D-117` (branche vive `claude/kind-babbage-927g10`, fusionnée ici en `766d289`) ont repointé les deux lanceurs : `FIGURES_CODE_DIR` → `figures/v1_legacy/`, `TRAIN_RESULTS_DIR` → `results/hyperparams/optuna_studies/`, et `--output` → `results/hyperparams/best_hyperparams.regenerated.json`. Ils ont aussi repointé `run_study_v3.sh` sur les **successeurs renommés** des 9 générateurs v3 (`t1_feature_selection` → `h2b_feature_selection`, …, dans `study/h2b_prediction/`) — la table de renommage que **D-76 avait déjà établie** pour `run_study_v2*.sh`, et que la prémisse de **D-49** (« les 9 générateurs n'existent plus dans ce dépôt sous aucun chemin ») contredisait. **La décision prise est la bonne** sur le seul point qui portait un risque : l'entrée gelée n'est pas écrasée, un fichier `.regenerated.json` est écrit à côté. **Rien n'est corrigé ici** : les deux tests sont remesurés, pas retouchés | **avant → après**, même commande (`pytest tests/test_launcher_paths_resolve.py -q`), avant `bff6bd3` / après `766d289` : invocations balayées **45 → 79** ; exemptions **11 → 0** ; exemptions encore invoquées par un lanceur **11/11 → 0/11** ; cibles invoquées inexistantes hors exemption **0 → 0**. Les deux tests rouges à la fusion étaient `test_the_undecided_remainder_of_D111_stays_written_where_it_lives` (la note qu'il exigeait est devenue fausse : `figures_code/` et `Train_results/` ne sont plus « introuvables sous toute racine ») et `test_each_exemption_still_names_a_real_dead_path` (les 11 exemptions ne nomment plus rien d'invoqué). Le premier est **remplacé** par un garde sur le comportement qui reste vrai — le lanceur n'écrit jamais par-dessus `results/hyperparams/best_hyperparams.json`, gelé (`PROVENANCE.md`, D-22) : mutation vérifiée, `--output` repointé sur le fichier gelé donne **VERT → ROUGE**, restauré **ROUGE → VERT**. Le second garde désormais la taille du balayage (**≥ 79**), une table vide et un balayage vide se ressemblant. **Aucun nombre publié ne bouge** ; aucun code de `src/` ni de `study/` n'est touché | `pytest tests/test_launcher_paths_resolve.py -q` (83 cas) ; la mutation : repointer `--output` de `scripts/generate_figures_v1.sh` sur `best_hyperparams.json` |
| D-112 | **Rapport seul, et à lire avant tout le reste : la suite est ROUGE sur la branche vive, et les trois échecs sont exactement les tests qui épinglaient l'état qu'un changement délibéré a renversé — sans que la remesure qu'ils exigent ait été faite.** `a0e0e02` (« study/ voit enfin le terme de point X », 16 août 02:24, branche `claude/kind-babbage-927g10`) fait consommer `K_xpoint` à `build_ising_terms`, qui ne lisait que `H_edges`, `C_edges` et `K_plaquettes`. Le changement est **juste et mesuré** par son auteur : sur les 256 états d'un problème `dim = 2`, l'écart entre l'énergie du chemin `study/` et la diagonale de `create_period_hamiltonian` vaut **5,3e−15**. Ce n'est donc pas le code qui est en cause — ce sont **trois seuils périmés**, au sens de `VIGIL.md` : le code a légitimement changé sous eux. Et les tests disent eux-mêmes ce qu'exige leur mise à jour : *« demande alors de rejouer phase 4, T13 et T26 avant de mettre ce test à jour »*. **Ces campagnes n'ont pas été rejouées.** Donc trois lectures publiées (phase 4, T13, T26) sont à refaire avant d'être citées. Je ne touche pas aux tests : un seuil périmé se **remesure**, il ne se retouche pas — et la remesure est une campagne, pas une passe de relecture. `D-51` (`docs/DEFAUTS.md`), dont c'était une des trois directions, est partiellement dépassée : sa section « `build_ising_terms` ne peut pas le représenter » ne décrit plus le code | **mesuré**, `pytest tests/ -q -m "not slow"` sur `c74d564` (ligne de base de la passe, avant toute modification) : **6 failed, 2537 passed, 66 skipped, 4 deselected, 2 xfailed, 639 warnings in 2303.36s (0:38:23)**. **Trois échecs déterministes**, reproductibles en **2,5 s** : `test_xpoint_term_absent_from_study.py::test_build_ising_terms_ignores_xpoint` (coefficients de plaquette **(4,) → (8,)**, `[0.1 ×4]` contre `[0.1 ×4, 7.0 ×4]`), `::test_ablation_zeroes_a_key_nothing_reads`, et `test_t13_control_is_not_vacuous.py::test_removed_max_separates_a_real_ablation_from_an_empty_one` (l'ablation censée être **vide** retire désormais **39,0** de coefficients au lieu de **0,0**). **Trois échecs stochastiques**, famille QAOA connue de ce fil : `test_noise_robustness`, `test_hyperparameter_sweep`, `test_qaoa_improves_discrimination` — ce dernier rejoué isolément **3 fois, 3 fois vert**. Pourquoi personne ne l'avait vu : la dernière suite complète du fil (`1382393`, 02:49) est **antérieure à la fusion** `18ece2e` qui a amené `a0e0e02` sur cette branche, et la passe suivante n'a touché que des documents (« Suite complète non relancée cette passe »). Cette ligne de base est la **première exécution complète depuis la fusion**. **Aucun de mes commits ne touche `study/` ni `src/`** | `pytest tests/study/test_xpoint_term_absent_from_study.py tests/study/test_t13_control_is_not_vacuous.py -q` → **3 failed, 9 passed en 2,47 s** ; le jour où phase 4, T13 et T26 auront été rejoués, c'est cette commande qui dira si les trois seuils ont été remesurés ou seulement retouchés |
| D-112 *(suite)* | **Rapport seul, un 4ᵉ seuil périmé, dans un fichier que D-112 ne couvrait pas.** Trouvé en remesurant la suite complète avant d'annoncer D-121 : `pytest tests/ -q -m "not slow"` sur `cb33697` rend **7 failed** — les 3 déterministes de D-112 inchangés, plus `test_qaoa_arm_is_sampled.py::test_the_ranking_is_nonetheless_visibly_perturbed` (**1/3 en isolé** — famille QAOA stochastique déjà documentée, pas un 4ᵉ défaut), plus **`test_every_launcher_invokes_real_files.py::test_a_frozen_launcher_says_so_in_its_own_header[scripts/run_study_v3.sh]`**, absent de la ligne de base D-112. Sa table `_FROZEN` déclare encore `scripts/run_study_v3.sh` mort et exige `"NE FONCTIONNE PLUS"` + `"D-49"` dans ses 40 premières lignes. **Aucun des deux n'y figure plus** (`grep -n 'D-49\|NE FONCTIONNE PLUS' scripts/run_study_v3.sh` → rien) : le lanceur a été repointé par D-116/D-117 (« Chemins repointes, ROOT_DIR corrige », en-tête actuel) et n'est plus le lanceur mort que D-49 documentait. Même famille que D-111 *(remesure)* — un lanceur réparé fait tomber le test qui gardait sa mise en garde de gel — mais dans un **fichier différent** de celui que D-111 *(remesure)* a remesuré (`test_launcher_paths_resolve.py`) : cette table `_FROZEN`-ci n'a pas été revue. Je ne retouche pas la table : lever l'exemption sans avoir rejoué le lanceur (« ~30-40 min » selon son propre en-tête) reviendrait à décréter qu'il fonctionne sans l'avoir mesuré — la même erreur, en sens inverse, que celle que ce test existe pour empêcher. **Aucun nombre publié ne bouge** ; aucun code de `src/` ni de `study/` n'est touché par ce rapport | mesuré, `cb33697` : `pytest tests/ -q -m "not slow"` → **7 failed, 2724 passed, 68 skipped, 4 deselected, 4 xfailed in 2402s (0:40:02)**. Isolé : `test_every_launcher_invokes_real_files.py::test_a_frozen_launcher_says_so_in_its_own_header[scripts/run_study_v3.sh]` reproductible à 100 % (pas stochastique) ; `test_the_ranking_is_nonetheless_visibly_perturbed` rejoué 3× isolé → **1 failed, 2 passed**, dispersion cohérente avec celle déjà mesurée du bras QAOA. Les 6 autres échecs de la ligne de base D-112 confirmés pré-existants en rejouant sur le parent `30aa725` (avant tout commit de cette passe) : **6 failed, 1 passed** (le seul vert y est déjà `test_the_ranking_…`, cohérent avec sa nature stochastique) | `pytest tests/study/test_every_launcher_invokes_real_files.py -q -k frozen_launcher` |
| D-113 | **`create_bounded_hamiltonian` — la contraction de plaquette du bord remplaçait le qubit manquant par le `<Z>` de l'AUTRE famille de liens.** Une plaquette a quatre membres : Haut = `H(i,j)`, Droite = `V(i,j+1)`, Bas = `H(i+1,j)`, Gauche = `V(i,j)`. Sur la colonne de droite du cœur le membre manquant est un lien **V**, sur la ligne du bas un lien **H** ; `init_qbits_state` place `theta_h` sur les qubits `idx_H` et `theta_v` sur les qubits `idx_V`, donc le `<Z>` de substitution doit venir du `theta` de CE lien. Le code lisait `z_halo_right_raw` (issu de `theta_h_full`) pour le membre Droite et `z_halo_bottom_raw` (issu de `theta_v_full`) pour le membre Bas : **les deux familles étaient échangées**. Les positions étaient bonnes — c'est la seule raison pour laquelle rien ne le signalait. La contraction de CISAILLEMENT, quinze lignes plus haut, lit la bonne famille : c'est la comparaison des deux qui a levé le défaut (question 4). Présent depuis le premier commit (`cf93ba3`). Forme « rôles échangés dans un tuple » | **Portée d'abord, parce qu'elle décide de la lecture : en déploiement l'effet est EXACTEMENT NUL.** `refinement._prepare_vqa_input` passe `mini_score` deux fois à `map_to_angles`, donc `theta_h ≡ theta_v` bit à bit (documenté dans `PhysToAngle.map_to_angles`, figé par `tests/mapping/test_mapper_contracts.py`) — les deux familles coïncident et l'échange est l'identité. **Mesuré : 36 configurations aléatoires (`dim ∈ {2,4,8}`, `K_xpoint` on/off, 6 tirages), opérateur identique bit à bit avant/après, 36 sur 36.** `python study/common/aggregate_master_table.py` : **180 lignes, OK=164, DIFF=16, MISSING=0** — l'état documenté, aucun nombre publié ne bouge. **C'est donc un piège armé, pas une valeur fausse en production** — la classe de la question 1, celle qu'aucun audit de couverture ne voit. **Ce que le piège vaut s'il se déclenche**, mesuré sur 36 configurations séparantes (`theta_h ≠ theta_v`) : **36 sur 36 changent**, écart max **1,072818** sur un coefficient. Sur un cas isolé à `k = -0,5` : rendu **+0,5** — le SIGNE de la plaquette bascule, et sa convention de parité paire (`K < 0`) porte la détection de vorticité ; avec un halo à `pi/2` le terme est rendu **-0,0**, la plaquette de bord disparaît de l'Hamiltonien sans erreur ni trace. **Combien de plaquettes** : `2·dim-1` sur `dim²`, soit **3 sur 4 à dim = 2** (la seule résolution publiée), 7 sur 16 à dim = 4, 15 sur 64 à dim = 8. Chemin concerné : `period_bound = (depth == 0)` — c'est l'axe **bord borné**, celui que `VIGIL_BA_Proj.md` désigne comme non traversé | `pytest tests/quantum/test_bounded_plaquette_halo_family.py -q` (8 cas ; **7 rouges** sur la version d'avant, le 8e étant le test de portée, qui épingle la raison pour laquelle aucun nombre ne bouge et doit crier le jour où un appelant passera deux cartes de score distinctes). Plus 2 cellules ajoutées à `test_vqa_stack_analytic.py::test_only_the_documented_halo_cells_are_read`, dont le docstring affirmait — exactement — le défaut : « `theta_h_full` n'est lu qu'en colonnes 0 et -1 ; `theta_v_full` qu'en lignes 0 et -1 ». **Rouge sur la version d'avant : `theta_v colonne -1 est ignoree`** ; et **un SEUIL PERIME remesure** — `test_hamiltonian_contracts.py::test_a_plaquette_on_the_boundary_contracts_instead_of_wrapping` chauffait `theta_h[1:-1,-1]` pour le membre Droite et `theta_v[-1,1:-1]` pour le membre Bas, c'est-a-dire **exactement les deux tableaux de la convention fausse** (ses commentaires disaient « halo droit » au-dessus d'une ecriture dans `theta_h`). Son ASSERTION n'a pas bouge ; c'est son champ d'essai qui a ete remesure. Conditions identiques (`dim = 3`, plaquette de coin `K[DIM,DIM] = -2.0`, reste a `pi/2`) : **anciennes cellules + code corrige = 0 terme** (l'echec), **bonnes cellules + code corrige = 1 terme `ZZ` sur `{idx_H(2,2), idx_V(2,2)} = {8, 17}` a -2.0**, **bonnes cellules + code d'avant = 0 terme**. Un second test (`test_the_boundary_plaquette_ignores_the_other_link_family`) epingle le pendant : les deux sont **exactement inverses** d'une version a l'autre, et tous deux **rouges sur la version d'avant** |
| D-114 | **Le garde-fou de l'invariant `theta_h ≡ theta_v` était une recherche de chaîne dans le source — il ne pouvait ni voir ce qu'il gardait, ni survivre à un retour à la ligne.** `tests/mapping/test_mapper_contracts.py::test_the_two_qubit_families_start_identical_in_deployment` faisait `assert "score_h=mini_score, score_v=mini_score" in open(refinement.__file__).read()`. L'invariant qu'il protège n'est pas décoratif : c'est lui qui rend **inerte** l'échange de familles de **D-113**. S'il tombe sans qu'on le voie, la contraction de plaquette du bord devient fausse — avant D-113, elle l'aurait été le jour même. Forme interdite nommément par `VIGIL.md` : *« L'assertion porte sur le comportement, pas sur le texte du source »*, et par `CLAUDE.md` : *« un test qui ne peut pas échouer est un défaut »* | **avant → après**, cinq conditions, la même commande à chaque fois (`pytest tests/mapping/test_mapper_contracts.py::test_the_two_qubit_families_start_identical_in_deployment -q`) : **A. faux vert** — chaîne laissée en commentaire et l'appel réel changé en `score_v=np.zeros_like(mini_score)` : **VERT → ROUGE** (l'ancien laissait passer exactement ce qu'il promettait d'empêcher) ; **B. faux rouge** — le même appel coupé sur deux lignes, comportement identique : **ROUGE → VERT** ; **C. portée** — `src/compare_rotor_budget.py` porte le même invariant et n'était lu par aucun garde ; l'y casser donne **VERT → ROUGE** ; **D. balayage vide** — un des deux appels renommé : **VERT → ROUGE** (le nouveau compte `vus == 2` et crie s'il n'a rien lu) ; **E. état sain** : **VERT → VERT**. Le nouveau garde lit l'AST — les couples `(score_h, score_v)` de chaque appel `map_to_angles` des deux modules déployés — et compare les expressions, pas le texte du fichier. **Aucun code de `src/` n'est touché, aucun nombre publié ne bouge** : le changement est entièrement dans le test | `pytest tests/mapping/test_mapper_contracts.py -q` (59 cas). Les cinq conditions ci-dessus sont reproductibles en éditant l'appel de `src/Simulation/refinement.py` ou de `src/compare_rotor_budget.py` et en relançant ce seul test |
| D-115 | **Même forme que D-114, trouvée par le sondage des 64 sites `.read()` que D-114 a motivé** (`docs/COUVERTURE.md`). `tests/study/test_t28_t29_labels_and_ci.py::test_the_relabeller_refuses_a_degenerate_threshold` faisait `assert "seuil global degenere" in src` sur le texte de `labels_global_threshold.py`, sans jamais appeler `relabel()`. Le garde réel qu'il prétend protéger (`if not np.isfinite(thr_global) or thr_global <= 0.0: raise SystemExit(...)`, ligne 84) peut disparaître — même désactivé par un `if False:` qui laisse le message en code mort — sans que le test le voie : la promesse du docstring (« 100 % durs, et rien ne crie ») redeviendrait vraie en silence | **avant → après**, guard de `study/pipeline/labels_global_threshold.py` désactivé (`if not ... or thr_global <= 0.0:` → `if False:  # DISABLED-FOR-TEST: ...`), même commande à chaque fois (`pytest tests/study/test_t28_t29_labels_and_ci.py -q -k degenerate`) : **ancien test, garde désactivé : VERT** (1 passed — le texte du message reste présent en commentaire mort, donc le passe) ; **nouveau test, code réel : VERT** (1 passed — `relabel()` appelé sur 4 artefacts synthétiques `l2_errors` tous nuls, lève bien `SystemExit("seuil global degenere…")`) ; **nouveau test, garde désactivé : ROUGE** (1 failed — la levée attendue n'a plus lieu à l'endroit prévu ; un second garde en aval, plus tardif, intercepte le même cas avec un autre message, ce qui confirme que le test pointe la bonne ligne). Aucun code de `src/` ni de `study/` (hors le test) n'est touché, aucun nombre publié ne bouge | `pytest tests/study/test_t28_t29_labels_and_ci.py -q` (13 cas) |

---

## T20 — Q-HAS run-to-run variance on fold `kh` (D11 quantified)

```
python study/closed_loop/closed_loop_run_variance.py --fold kh --repeats 5
```
5 Q-HAS runs + 2 classical controls, identical inputs, 3216 s.

| metric | Q-HAS mean | std | range | CV | classical range |
|---|---|---|---|---|---|
| combined | 0.2500 | 0.0104 | 0.0232 | 0.042 | **0.00e+00** |
| **phys_score** | **0.00324** | **0.00158** | 0.0039 | **0.489** | **0.00e+00** |
| patch_ratio | 0.8670 | 0.0376 | 0.0785 | 0.043 | **0.00e+00** |

Q-HAS `phys` draws: **0.0015, 0.0020, 0.0031, 0.0042, 0.0053**.

**The control passes.** The classical arm's range is exactly **0.00e+00** on
all three metrics across both repeats — a fifth independent confirmation of
its determinism. Without that, the Q-HAS spread could have been an artefact
of the measurement chain; with it, the spread is attributable to the
unseeded QAOA path (D11) and nothing else.

**A 48.9 % coefficient of variation on the fidelity metric.**

### The published `kh` numbers were one draw, and it was the extreme one

The fold's stored Q-HAS value, 0.00700, sits at the **100th percentile** of
all six known draws — it is the largest. Everything computed from it is
correspondingly inflated:

| quantity | from the stored draw | **from the mean of 5 draws** |
|---|---|---|
| gap / std | 3.15 → "direction survives" | **0.77 → a single run cannot support a directional claim** |
| ratio vs budget-matched classical | 4.16× (published as 4.41×) | **1.93×** |

**The `kh` ratio is roughly halved.** T20 originally reported only the
stored-draw figure, which is the optimistic choice; it now computes both and
quotes the mean-based one.

### What survives, and it is the dominance count, not the ratio

Against the budget-matched classical arm (phys 0.00168 at patch 0.7943):

- Q-HAS costs **more on 5/5 draws** (patch 0.830–0.908 vs 0.794);
- Q-HAS is less faithful on **4/5 draws**;
- on the remaining draw the arms are **incomparable** (Q-HAS more faithful,
  but more expensive) — **never reversed**.

So the direction holds as a **dominance count over draws**, not as a point
ratio. The honest statement for `kh` is *"classical is cheaper on every
draw and more faithful on four of five"*, not *"Q-HAS is 4.4× worse"*.

### Consequence for the other folds

`ot`, `rotor` and `tearing` each have **one** Q-HAS draw (plus a replay for
`ot`). Their published ratios rest on the same single-draw basis and should
be read as **point estimates of a quantity with ≈50 % CV**, not as measured
magnitudes. Repeating T20 per fold is the fix; it costs ~1 h per fold.

---

## T20 complete — Claim E restated as a dominance count over repeated draws

> **SUPERSEDED — do not quote the per-fold numbers in this section.** This
> pass did not capture each draw's abort status, so `rotor`'s mean silently
> included 2 diverged trajectories. See *T20 verified* below for the numbers
> that stand (1.30×, 1.90×, 2.74×, 1.81×); the section is kept because the
> comparison between the two passes is what shows how much an unguarded
> draw distorts a mean.

5 Q-HAS repeats per fold, identical inputs, plus 2 classical repeats per
fold as a determinism control. **The classical control's range is exactly
0.00e+00 on every metric of every fold** — 8 independent replays. The spread
below is therefore attributable to the unseeded QAOA path (D11) alone.

### Per-fold distribution, against the **budget-matched** classical arm

| fold | Q-HAS mean | sd | CV | matched ref | gap/sd | ratio published → **mean-based** |
|---|---|---|---|---|---|---|
| `ot` | 0.1291 | 0.0222 | 17.2 % | 0.0827 | **2.09** | 2.35× → **1.56×** |
| `kh` | 0.0032 | 0.0016 | 48.9 % | 0.00168 | **0.98** | 4.16× → **1.93×** |
| `rotor` | 0.1537 | 0.0642 | 41.8 % | 0.0536 | **1.56** | 3.13× → **2.86×** |
| `tearing` | 0.0091 | 0.0034 | 37.4 % | 0.00443 | **1.37** | 4.19× → **2.05×** |

**On three folds of four the gap/sd is below 2**: a single run per arm
cannot support a claim about *magnitude*. Every published ratio was inflated
by a factor 1.1–2.2, because each rested on one draw.

### Why the reference must be the budget-matched arm, always

T20 first compared against the *tuned* classical arm, which is wrong twice
over and produced two spectacular non-results:

- **`rotor`**: the tuned classical arm had **aborted**, so its stored value
  is a partial score. gap/sd came out **15.88** — against a crashed run.
- **`ot`**: the tuned classical arm *completes* but runs at a different
  budget (patch 0.324 against Q-HAS's 0.680, defect D4). gap/sd came out
  **16.01**, measuring the operating point, not the decision rule.

Both are now excluded by construction: the reference is the budget-matched
point, whose completion the T19 trace audit verified.

### The robust statement

```bash
python study/closed_loop/closed_loop_headline_counts.py     # recomputes the table below
```

| fold | n | aborted | less faithful | costlier | strictly dominated |
|---|---|---|---|---|---|
| `ot` | 5 | 0 | 5/5 | 5/5 | **5/5** |
| `kh` | 5 | 0 | 5/5 | 4/5 | **4/5** |
| `rotor` | 3 | 2 | 3/3 | 2/3 | **2/3** |
| `tearing` | 5 | 0 | 5/5 | 5/5 | **5/5** |
| **total** | **18** | 2 | **18/18** | **16/18** | **16/18** |

> Across four held-out classes and **18 completed** closed-loop runs, Q-HAS
> is less faithful than the budget-matched classical rule on **every one of
> the 18**, more expensive on **16 of 18**, and strictly Pareto-dominated on
> **16 of 18**. No run reverses the ordering on both coordinates at once.

**Correction — this table previously read 19/20, 18/20, 17/20.** It was the
only headline in the study composed by hand rather than computed, and it did
not reproduce from the artifacts. Two errors, both of a kind already in the
register:

1. on `kh`, *less faithful* and *costlier* were **transposed** (4/5 and 5/5
   instead of 5/5 and 4/5);
2. on `rotor`, the **2 aborted draws were counted in the denominator**,
   giving a total out of 20 when only 18 runs completed — the exact defect
   ("an aggregation mixing aborted draws with valid ones") that had been
   fixed in the code and reappeared in the prose.

The corrected count is **stronger on fidelity** (unanimous, 18/18, where the
old figure conceded one run) and **weaker on cost** (16/18). The direction of
the conclusion is unchanged. T23 now computes it and `t16` checks it, so the
number can no longer drift from its artifacts.

This is the form Claim E should take in the manuscript. It is weaker-sounding
than "2.6–4.4× worse" and far harder to attack: it depends on no single draw,
no choice of λ, and no scalarisation.

### Correction to an earlier claim of mine

I wrote that the published value was the maximum draw "on all four folds".
That was true for `kh`, `ot` and `tearing` but **not** `rotor`, whose stored
value sits at the 67th percentile. Three of four, generalised too early from
three observations.

---

## D13 — a train/test leak in the Level-3 protocol, and the unseen-condition test

### The leak

`docs/level3_preregistration.md` states the held-out class is excluded from
**all** tuning of both arms. That is **false for the QAOA arm**.

`train_hyperparams.make_composite_objective` hard-codes the decision
threshold:

```python
if "threshold_amr" not in frozen:
    HyperParams["threshold_amr"] = 0.14959824837662078   # le meilleur classique
```

and that number comes from `_run_classical_phase1`, whose own banner reads
**"Scenarios: KH + OT + Tearing + Rotor"** — all four classes. So on every
fold, the QAOA arm decides using a threshold fitted on data that includes
the held-out class. My driver reproduced it verbatim:
`best.setdefault("threshold_amr", 0.14959824837662078)`.

The classical arm has no such problem: `train_classical_threshold_excluding`
re-tunes its threshold per fold on the training classes only.

**The leak is asymmetric and favours Q-HAS.** It is therefore *conservative*
with respect to the conclusion — Q-HAS is beaten on all 18 completed runs
despite holding an advantage it should not have. But the protocol's claim of a clean
LOSO is wrong as written and must be corrected in the manuscript.

This is also the precise form of defect D4: not merely "different operating
points" but a genuine information leak.

### The second, independent problem: the initial condition was never new

Even with the parameter leak removed, V1's `_init_dns_scenario` calls every
`init_*` **without arguments**, so every evaluation uses the canonical
initial condition. A model that generalises must face a condition it has
never met, not the canonical trajectory of a class it merely did not tune on.

**T22** supplies that test. It substitutes `_init_dns_scenario` temporarily
(V1 unmodified, restored in a context manager, and the substitution is
*verified*: the run aborts if the trajectory does not actually change) to
pass physical parameters to the initialisers:

| class | unseen condition |
|---|---|
| Kelvin–Helmholtz | narrower shear layer, weaker seed, faster drift |
| Harris tearing | thinner current sheet, **mode 2** instead of mode 1 |
| MHD rotor | slower, smaller rotor, wider taper |
| Orszag–Tang | **no IC parameters exist** — the only available unseen condition is a different Reynolds number, declared as such |

Verified distinct at N=64 before launching: KH 3773.6 → 4118.3,
tearing 3546.8 → 2951.0, rotor 4739.8 → 4409.3, and V1's function object
restored identically afterwards.

The reported quantity is the **degradation ratio** of each arm,
phys(unseen) / phys(canonical), so the comparison is between how the two
decision rules *transfer*, not between their absolute errors.

---

## Trap sweep — where else can an invalid run masquerade as a valid one?

The recurring failure mode in this campaign is a computation that fails but
returns a value **indistinguishable from a valid one**. It has now surfaced
five times (T15 fold scoring, T20 gap/sd, T22 classical reference, T22
Q-HAS draws, and the T13/T19 filename overwrites). A systematic sweep of
every `run_arm` call site in `study/v4/`:

| call site | guarded? | recoverable after the fact? |
|---|---|---|
| `t15:313` Q-HAS fold arm | no | **no** — non-deterministic (D11) |
| `t15:319` classical fold arm | no | yes — deterministic, T19 audits it |
| `t15b:66` bisection points | no | yes — classical only, T19 `--trace-only` audits it |
| `t19:88` audit replay | **yes** | — |
| `t20:120` Q-HAS variance draws | **was no** | **no** |
| `t20:129` classical control | was no | yes |
| `t22:250` both arms | **yes** (fixed) | — |

**The one that mattered: T20's Q-HAS draws.** Those 18 completed runs (of
20 launched) underpin the
restated Claim E, and their completion was never verified. Because the arm
is non-deterministic, it **cannot** be verified now — replaying does not
reproduce the draw.

Evidence bounding the risk, short of a re-run: a divergence produces a
partial score wildly out of family with its siblings — the T22 case was
**300×**. The T20 spreads are max/min = 1.5 (`ot`), 2.9 (`tearing`),
2.7 (`rotor`), 3.6 (`kh`), and no draw exceeds phys = 1. All are consistent
with D11's measured CV of 17–49 %, none shows the divergence signature. So
contamination is **unlikely but unproven**.

`t20` now captures the abort marker per run and excludes aborted runs from
the statistics. A verified re-run is queued behind T22b; until it lands, the
Claim E numbers carry this caveat.

### Two smaller findings from the same sweep

**Optuna tuning was clean.** All completed trial values across the three
persisted studies lie in 0.23–0.51 — none at the divergence penalty (10.0),
none above 1. So no fold was tuned against diverged evaluations. The
`catch=(Exception,)` in `study.optimize` is a latent trap (a systematically
failing objective would be silently skipped) but did not fire: zero `FAIL`
states in any study.

**Fold `ot` has weaker tuning provenance than the other three.** It was
tuned before per-trial Optuna persistence existed, so
`t15_level3_optuna_ot.db` does not exist and its per-trial values are
unrecoverable. Its checkpoint carries an explicit provenance note: *"recovered
from logs/v4/level3.log after the container was reclaimed mid-run; QAOA
params printed at 4-decimal precision"*. The other three folds have full
trial-level records.

### Trap sweep, second pass: is the "unseen" condition actually unseen?

The T22 guard checked only that the trajectory *changed*. Two failure modes
slipped through it:

**(a) A diverged DNS would pass.** A trajectory that blows up produces a
huge signature, which reads as "changed". Checked by hand across all four
folds: signature ratios are 0.83–1.08, modest shifts with no blow-up, so
this did not fire. A finiteness test and a physical band (0.05–20) are now
enforced automatically.

**(b) A negligible change also passes.** This one *did* fire:

| fold | trajectory shift at hot start |
|---|---|
| `harris_tearing` | −16.7 % |
| `mhd_rotor` | −15.9 % |
| `kelvin_helmholtz` | +7.5 % |
| **`orszag_tang`** | **−0.3 %** |

`orszag_tang` exposes no initial-condition parameters, so its only available
"unseen condition" is a different Reynolds number — and Re 400 → 600 moves
the hot-start trajectory by **0.3 %**, some 20–50× less than the three
classes where the initial condition itself can be varied.

**Fold `ot`'s transfer test is therefore nearly vacuous** and must be
reported as such rather than counted alongside the other three. T22 now
warns below a 1 % shift and records `unseen_condition_is_weak`; T22c prints
the affected folds and refuses to let them carry a transfer claim.

This is a limitation of V1's API, not of the test: `init_orszag_tang()`
takes no arguments, and `src/` is read-only.

---

## Fresh-eyes review — assumptions re-examined from scratch

Six load-bearing assumptions, re-derived from the source rather than
from memory. Three held, three did not.

### HELD — the ablation is clean

**Both arms differ only in the decision routine.** `classical_only` swaps
`run_adaptive_vqa` → `run_adaptive_classical` on the *same* simulator
object, with the same mapper, `threshold_amr`, `target_dim`, `max_depth`,
`min_size` and TTL map (`pipeline.py:391`).

**Both arms threshold the same score.** `refinement.py:474` (classical) and
`:579` (VQA) both call `AngleMapper.classical_score(physics_state)`. The
QAOA route perturbs exactly the quantity the classical route thresholds, so
the comparison isolates the decision rule and nothing else.

**Both arms are scored at the same physical instant.** With a DNS trace
supplied — the Level-3 case — `dt = dns_trace[step]['dt']`
(`pipeline.py:458`), so both arms march on the DNS time grid and are
compared against the same `dns_trace[last_step]['fluxes']`. The
"adaptive dt desynchronises the arms" trap does **not** fire.

### DID NOT HOLD — three corrections

**(1) `phys_score` is not a plain relative L2.** It is an
*instability-weighted* relative L2: `score()` builds
`w = 1 + 0.25·(|Jz|/⟨|Jz|⟩ + |ω|/⟨|ω|⟩)` from the reference fields and
weights every field's error by it. Every table and figure axis in this
repository has called it "relative L2 vs DNS", which is wrong. Both arms are
scored identically so no bias follows, but the label must be corrected to
**"instability-weighted relative L2 vs DNS"** throughout the manuscript.

**(2) The cost axis excludes the cost of the decision.**
`patch_ratio = total_pixel_used / (steps · N²)` counts refined pixels only.
The QAOA circuit does not appear in it, yet the Q-HAS arm takes **2.7–3.3×**
the classical arm's wall time (ot 1069 s vs 371 s, kh 579 vs 213, tearing
240 vs 73) — on a *simulated* 8-qubit circuit, so hardware would be worse.
"Equal budget" therefore means "equal AMR budget, with Q-HAS's decision
compute free". This makes the conclusion **more conservative**, not less,
but the axis is mis-specified and must be declared.

**(3) T21's ill-posedness claim was overstated — my error.** T21 tested
whether the *count* changes with λ and concluded the endpoint was
ill-posed. Count and verdict are different things. Re-checked over
λ ∈ [0, 100] with `rotor` excluded:

| λ | Q-HAS | classical |
|---|---|---|
| 0.0 – 0.8 | 1 | **2** |
| 1.0 – 100 | 0 | **3** |

**The classical arm holds the majority at every λ tested.** The verdict
never flips; only the margin moves (2–1 → 3–0). The endpoint is *not*
ill-posed in its direction, and saying it was overstated the case. T21 now
separates "margin changes" from "verdict flips" and reports both; the
λ grid was extended to 100 because stability on [0, 5] proves nothing about
[0, ∞).

This correction **strengthens and simplifies** the result: the pre-registered
endpoint, once the failed fold is excluded as its own §5 requires, favours
the classical arm robustly rather than ambiguously.

---

## T22b complete — the transfer signal does not survive replication

56 runs, **zero aborted**, 5 Q-HAS draws per condition per fold, classical
reference budget-matched everywhere.

| fold | deg Q-HAS | deg classical | \|z\| | separable |
|---|---|---|---|---|
| `ot` † | 0.955 ± 0.373 | 0.946 | **0.02** | no |
| `kh` | 1.027 ± 0.509 | 1.364 | **0.66** | no |
| `rotor` | 0.312 ± 0.120 | 0.526 | **1.78** | no |
| `tearing` | 0.166 ± 0.065 | 0.389 | 3.45 | **yes** |

**1 fold of 4.** The single-run pass had suggested Q-HAS transfers
relatively better on *all four* folds (ratios narrowing 0.22→0.17,
2.52→1.81, 3.67→1.88, 2.94→1.01). Repeated with 5 draws, that pattern
evaporates: on `ot` the two arms degrade identically (|z| = 0.02).

† `ot` is unusable for this question regardless: its "unseen" condition
shifts the trajectory by only 0.3 % (no IC parameters exist on
`init_orszag_tang`).

**What holds — the reference-free count:**

| fold | ratio Q/C canonical → unseen | dominated on unseen |
|---|---|---|
| `ot` | 1.48× → 1.50× | 4/5 |
| `kh` | 2.18× → 1.64× | 5/5 |
| `rotor` | 2.48× → 1.47× | 4/5 |
| `tearing` | 3.27× → 1.39× | 5/5 |
| **total** | ratio narrows but never crosses 1 | **18/20** |

> Q-HAS is strictly Pareto-dominated on **18 of 20** runs against initial
> conditions it has never seen — less faithful *and* more expensive.

**Answer to the leakage question.** The concern was well founded but the
mechanism is sharper than "the model saw the end of a trajectory it trained
on":

1. a leak does exist (**D13**) — the QAOA arm's threshold was fitted on all
   four classes including the held-out one — and it **favours Q-HAS**;
2. the initial condition was never new, which T22 fixes;
3. and facing genuinely unseen conditions, Q-HAS remains **strictly
   dominated on 18 of 20 runs** — less faithful *and* more expensive.

On the third point, be precise about what is and is not claimed. Q-HAS's
*relative* degradation is smaller than the classical arm's on the one fold
where the difference is separable (`tearing`, 0.166 against 0.389, |z| =
3.45). That is a real observation and it is **not** evidence that Q-HAS
transfers better: it degrades less from a starting point that was already
worse, and it is still dominated on both coordinates on 5/5 of that fold's
unseen runs. T22d tests the obvious alternative explanation — that both arms
are approaching a common attainable floor — and that confound is not
resolved. So the honest statement is *"Q-HAS is not shown to transfer
better, and remains dominated in absolute terms"*, not *"Q-HAS transfers
worse"*.

So the conclusion does not rest on the leak: Q-HAS loses **despite** an
undue advantage, and loses again on conditions it has never met.

**Still open:** the common-floor confound on `tearing`, the one separable
fold. T22d measures it.

---

## T22d — distance to near-full refinement, all four folds

One classical run per condition at threshold 0.05 (refine almost
everything), the lowest point already swept by t15b's bisections.

| fold | reference can / uns | classical can / uns | Q-HAS can / uns |
|---|---|---|---|
| `tearing` | 0.00397 / 0.00155 | **1.12× / 1.11×** | 3.65× / 1.55× |
| `kh` | 0.00126 / 0.00166 | **1.33× / 1.39×** | 2.90× / 2.28× |
| `rotor` | 0.03395 / 0.02874 | **1.58× / 0.98×** | 3.91× / 1.44× |
| `ot` | 0.01111 / 0.00821 | **7.45× / 9.53×** | 11.04× / 14.28× |

### Three corrections to what I first claimed from this table

**(1) The reference is not a lower bound.** `rotor`'s classical arm scores
**0.98×** on the unseen condition — it *beats* near-full refinement. So
refining almost everything is not always optimal, and this quantity is an
estimate of the achievable optimum, not a certified floor. Any arm below
1.00× is now flagged by the script as proof of exactly that.

**(2) "The classical rule occupies the ceiling" holds on 3 folds, not 4.**
On `ot` **both** arms sit 7–14× above near-full refinement. There is
substantial headroom on that class which neither arm exploits, so the claim
that "there is nothing left for any method to gain" is false there.

**(3) Distance-to-reference is confounded by the operating point.** The
reference refines ~0.95 of the domain; `ot`'s classical arm runs at ~0.37,
`tearing`'s at 0.625. A cheaper operating point is mechanically further from
the full-refinement error, so these distances are **not comparable across
folds**.

### What survives without reservation

Within every fold and on both conditions, **Q-HAS is further from the
reference than the classical arm** — 11.04 vs 7.45, 2.90 vs 1.33, 3.91 vs
1.58, 3.65 vs 1.12. Eight comparisons, eight in the same direction, each one
between two arms at the same operating point on the same trajectory.

That is the only reading these measurements license, and it is enough: at
matched budget the quantum decision rule extracts strictly less of the
available accuracy than plain thresholding, on every class and both under
canonical and unseen initial conditions.

---

## Verified T20 — an aborted run does not always look anomalous

Re-running T20 with the abort marker captured at execution time (the
original pass had no such guard, and being non-deterministic could not be
audited afterwards) produced the finding that most changes how the earlier
numbers must be read.

**Fold `rotor`, Q-HAS draws:**

| draw | phys | status |
|---|---|---|
| 1 | 0.2191 | ok |
| 2 | 0.0978 | ok |
| **3** | **0.6877** | **ABORTED** |
| 4 | 0.0536 | ok |
| **5** | **0.4069** | **ABORTED** |

**Two of five draws diverged — 40 %, not the 1-in-5 I estimated.**

**And draw 5 returned 0.4069, a value that does not stand out.** The valid
draws span 0.054–0.219; 0.407 is high but not absurd. So an aborted run can
land inside the plausible range.

### This retracts my earlier bounding argument

I had written, to bound the risk on the unguarded pass: *"a divergence lands
300× out of family (the T22 case), while T20's spreads are 1.5–3.6× with no
draw above phys = 1 — consistent with D11's CV, no divergence signature.
Contamination unlikely but unproven."*

That reasoning is **wrong**. Contamination need not leave a visible
signature. `rotor`'s original five draws (max 0.2581) could perfectly well
have contained aborted runs, and no inspection of the values would reveal
it. The correct statement is not "unlikely but unproven" — it is
**unknowable without the guard**, which is precisely why the guard had to be
added and the pass repeated.

### A flaw in T20's own control

On `rotor`, **both classical control runs also aborted** (1.1731 twice).
T20 runs its determinism control at the *tuned* threshold, which diverges on
this fold. The control still shows determinism — the divergence reproduces
exactly — but it no longer validates the measurement chain, which is its
purpose. It should run at the budget-matched threshold, as the *reference
value* already does.

### D14 — the fix landed after two of the four folds had started

`always_matched=True` was added to T20's control, and the campaign was
*not* re-run: `ot` and `kh` had already been launched. Their control
therefore replays the **tuned** threshold while their artifact records
`classical_reference_source = "budget-matched classical"`. Both statements
are individually true — the field describes the *reference value*, read
correctly from T15b — but a reader naturally attaches it to the neighbouring
`classical_stats` block, and that block is something else entirely:

| fold | matched thr | replayed thr | matched phys | replayed phys |
|---|---|---|---|---|
| `ot` | 0.1906 | 0.4616 (tuned) | 0.0827 | **0.4845** |
| `kh` | 0.1906 | 0.4616 (tuned) | 0.00168 | 0.00202 |
| `rotor` | 0.0969 | 0.0969 ✓ | 0.05365 | 0.05365 |
| `tearing` | 0.4250 | 0.4250 ✓ | 0.00443 | 0.00443 |

`rotor` and `tearing` agree because the pre-fix code already fell back to
the matched threshold when the tuned arm had aborted.

**On `ot` this is enough to invert the fold.** Against the matched 0.0827,
Q-HAS's 0.1291 is 1.56× worse; against the replayed 0.4845 it is 3.75×
*better*. The published numbers use the matched value and are unaffected,
but anyone recomputing from `classical_stats` — as I did while building T23 —
gets the opposite sign on that fold. The two references are now split into
distinct fields and T23 documents which one is correct.

### D15 — the provenance stamp is taken at the wrong moment

`git_commit_hash()` runs when the artifact is *saved*. A run lasting an hour
is therefore stamped with whatever was committed while it was still
executing. That is exactly how the `ot` and `kh` artifacts carry a hash
postdating the `always_matched=True` commit while having executed the
pre-fix code — the stamp actively pointed away from the truth.

CLAUDE.md requires the commit hash in every output. It is necessary but
**not sufficient for long runs**: the hash must be captured at start, and a
run that spans a commit to its own source should say so.

### Consequence

Every variance figure published from the unguarded pass — the CVs, the
mean-based ratios (1.56×, 1.93×, 2.86×, 2.05×), the gap/sd values — rests on
draws of unknown status. They are superseded by this pass, and on `rotor`
the mean is now computed from **3 valid draws**, not 5.

---

## T20 verified — final numbers, and why the per-fold magnitudes cannot be quoted

All four folds re-run with the abort marker captured at execution time, the
classical control at a non-diverging threshold, and aborted draws excluded
from the statistics.

| fold | valid draws | mean phys | sd | CV | gap/sd | ratio vs matched classical |
|---|---|---|---|---|---|---|
| `ot` | 5/5 | 0.10727 | 0.01823 | 17.0 % | 1.35 | 1.30× |
| `kh` | 5/5 | 0.00320 | 0.00203 | **63.6 %** | 0.75 | 1.90× |
| `rotor` | **3/5** | 0.14725 | 0.04062 | 27.6 % | **2.30** | 2.74× |
| `tearing` | 5/5 | 0.00801 | 0.00193 | 24.1 % | 1.86 | 1.81× |

**Only 1 fold of 4 reaches gap/sd ≥ 2.**

### The magnitudes have now shrunk twice

| fold | first published (1 draw) | unguarded 5-draw mean | **verified 5-draw mean** |
|---|---|---|---|
| `ot` | 2.57× | 1.56× | **1.30×** |
| `kh` | 4.41× | 1.93× | **1.90×** |
| `rotor` | 3.62× | 2.86× | **2.74×** |
| `tearing` | 4.38× | 2.05× | **1.81×** |

### The decisive observation: which fold "passes" is not stable

| fold | gap/sd unguarded | gap/sd verified |
|---|---|---|
| `ot` | 2.09 → **separable** | 1.35 → not |
| `rotor` | 1.56 → not | 2.30 → **separable** |
| `kh` | 0.98 | 0.75 |
| `tearing` | 1.37 | 1.86 |

Both passes report "1 of 4 folds separable" — **but not the same fold**. `ot`
fell below the threshold and `rotor` rose above it. At n = 5 draws, the
separability verdict is itself unstable, which is the clearest possible
evidence that **per-fold magnitude claims are not supportable at this sample
size**. Reporting "Q-HAS is 2.7× worse on rotor" would be reporting a number
whose confidence interval is wide enough to swallow the effect.

**What survives is the direction and the dominance count**, which do not
depend on any single fold's ratio: the verified mean exceeds the
budget-matched classical value on **4 folds of 4** (1.30×, 1.90×, 2.74×,
1.81×), and Q-HAS was strictly Pareto-dominated on 18 of 20 unseen-condition
runs (T22c).

### A robustness asymmetry not captured by any metric

`rotor`'s Q-HAS arm **aborted on 2 of 5 draws (40 %)** while its classical
control at the same budget completed both times, deterministically (0.0536
twice). Across the campaign, 6 Q-HAS aborts have been observed on `rotor`
against 0 for the classical arm at a matched threshold.

None of `phys_score`, `patch_ratio`, the dominance count or the λ analysis
measures this: they all presuppose a run that finishes. The quantum decision
rule produces refinement configurations that destabilise the solver at a
rate the classical rule does not, and that is a distinct failure mode
deserving its own line in the manuscript.

---

## T22 leak-free — D13 removed, and Q-HAS does not survive it

```bash
python study/h4_transfer/h4_unseen_conditions.py --fold <f> --mode leak-free \
    --repeats 5 --matched-reference
python study/closed_loop/closed_loop_leak_free_summary.py
```

`--mode leak-free` replaces the QAOA arm's leaked threshold
(`0.14959824837662078`, fitted on all four classes) with the fold's **own
classical tuned threshold**, produced by
`train_classical_threshold_excluding` on the training classes only. The
leak is gone.

### What the mode does not do

It does **not** re-tune the QAOA arm. The definitive experiment puts
`threshold_amr` back into the Optuna search space, excluded from the
held-out class, and is still not attempted. So this measures a **bound**:
*does Q-HAS survive losing the leaked threshold without re-tuning?* — not
*what is the best leak-free Q-HAS?*

### The trap this result had to avoid

The two arms **do not run at the same threshold**. `--matched-reference`
holds the classical control at the budget-matched point, so on `rotor` the
QAOA arm runs at 0.5864 while its control runs at 0.0969. Comparing their
errors directly would confound the decision rule with the budget.

My own code printed *"at the SAME operating point the classical arm
completed"* when `rotor`'s Q-HAS arm died. **That sentence was false** —
the thresholds differ by a factor of six — and it is the campaign's motif
in its purest form: a line of output that does not describe the computation
it accompanies. It now prints both thresholds and says explicitly that they
differ. The artifact carries `qaoa_threshold_amr`,
`classical_threshold_amr` and `thresholds_match`.

The budget-controlled comparison is therefore against the **T15b classical
frontier interpolated at the budget Q-HAS actually realised**, and T24
**refuses to interpolate outside the swept range** rather than let
`np.interp` return an edge value that looks like a measurement.

### Results, all 4 folds

| fold | condition | Q-HAS budget | Q-HAS phys | classical frontier at that budget | ratio |
|---|---|---|---|---|---|
| `rotor` | canonical | — | — | — | **all 5 draws ABORTED** |
| `rotor` | unseen | 0.0882 | 0.8535 | budget below the swept range | not computable |
| `tearing` | canonical | 0.3846 | 3.7351 | 1.7982 | **2.1×** |
| `tearing` | unseen | 0.4232 | 2.5600 | 1.5100 | **1.7×** |
| `kh` | canonical | 0.5513 | 0.02745 | 0.01472 | **1.9×** |
| `kh` | unseen | 0.4646 | 0.13272 | 0.02967 | **4.5×** |
| `ot` | canonical (n=2/5) | 0.2686 | 0.59911 | 0.36638 | **1.6×** |
| `ot` | unseen (n=3/5) | 0.2657 | 0.50405 | 0.36895 | **1.4×** |

**Every fold with a computable ratio puts Q-HAS above the classical
frontier at its own realised budget — 3 of 3, with `rotor` unmeasurable
because it has no operating point at all.**

### Aborts: the sharpest number in the campaign

| fold | Q-HAS aborted | classical aborted |
|---|---|---|
| `rotor` | **7 / 10** | 0 / 4 |
| `ot` | **5 / 10** | 0 / 4 |
| `kh` | 0 / 10 | 0 / 4 |
| `tearing` | 0 / 10 | 0 / 4 |
| **total** | **12 / 40 (30 %)** | **0 / 16** |

Removing the leak costs Q-HAS **30 % of its runs outright**, concentrated
on two folds of four, while the classical arm at its budget-matched
threshold completes every single draw. On `ot` the two arms are visible
side by side: the classical control completes 2/2 deterministically at
budget 0.64, Q-HAS aborts 3/5 and spends 0.27 on the draws that survive.

**Removing the leak makes Q-HAS dramatically worse, and on one fold
inoperable.**

- On `rotor`, **every canonical draw diverges** at the leak-free threshold.
  The arm collapses to a budget of 0.09–0.27 where the classical control
  spends 0.356. Two of five unseen draws also abort.
- On `tearing`, Q-HAS's error rises from 0.0080 (leaked, budget 0.91) to
  **3.735** (leak-free, budget 0.385). Most of that is the budget collapse
  — it refines less than half as much — but **not all of it**: against the
  classical frontier *at its own realised budget* it is still **2.1×
  worse**.
- On `kh`, 10 draws, **zero aborted**. Error rises from 0.0032 (leaked,
  budget 0.870) to **0.02745** (leak-free, budget 0.551) — **1.9×** the
  frontier at its own budget on the canonical condition and **4.5×** on the
  unseen one.

### What `ot` can and cannot contribute, decided before it lands

`ot` is running. Its two halves are **not** equally informative, and that
is fixed by the physics, not by the result:

- its **canonical** half is fully informative — it asks whether Q-HAS
  survives its own fold's leak-free threshold, exactly as on the other
  three;
- its **unseen** half is **nearly vacuous** and must be reported as such.
  `init_orszag_tang()` takes no parameters, so the only available unseen
  condition is a different Reynolds number, which shifts the hot-start
  trajectory by **0.2846 %** — 20–50× less than the other three folds.
  `t22` emits the warning at run time and records
  `unseen_condition_is_weak`.

Stating this now, before the number exists, so that whichever way it falls
it cannot be recruited as a transfer result. If `ot` shows a reversal it
adds nothing to the 3/3 above; if it shows none, that is not evidence
against them.

### `kh` also carries the sharpest transfer reversal

| | leaked | leak-free |
|---|---|---|
| Q-HAS degradation | 1.027 | **×4.835** |
| classical degradation | 1.364 | ×1.364 |
| who degrades more | classical | **Q-HAS** |

Under the leak, `kh` was one of the folds where Q-HAS degraded *less* than
the classical rule on an unseen initial condition. Leak-free it degrades
**3.5× more**. Together with `tearing` (×0.685 against ×0.389, also
reversed) that is **both informative folds reversing in the same
direction** once the leaked threshold is removed.

### The full transfer picture, including the fold that goes the other way

| fold | Q-HAS degradation | classical | Q-HAS worse? | reading |
|---|---|---|---|---|
| `kh` | ×4.835 | ×1.364 | **yes** | reversal |
| `tearing` | ×0.685 | ×0.389 | **yes** | reversal |
| `rotor` | undefined | ×0.526 | — | no operating point |
| `ot` | ×0.841 | ×0.946 | no | **vacuous by construction** |

**`ot` goes the other way and I am not counting it — as pre-registered
above, before the number existed.** Its "unseen" condition shifts the
trajectory by 0.2846 %, so both arms barely move (×0.84 and ×0.95, i.e.
nothing happened to either). That is the outcome the pre-registration
anticipated for a vacuous condition, and the commitment cuts both ways:
this fold was excluded from supporting the reversal, so it cannot now be
admitted to undermine it. The reversal claim rests on `kh` and `tearing`
— **2 of 2 informative folds**, not 4 of 4.

**Run-to-run spread widens too.** `kh`'s leak-free draws give CV 26.3 %
canonical and **64.7 %** unseen, against the 17–49 % band T20 measured for
the leaked configuration. One draw (0.2854 against neighbours near 0.09)
drives most of that — and the divergence guard confirms it **completed**,
`abort = None`, so it stays in. Excluding a valid draw because it looks
inconvenient is the mirror of the defect that contaminated `rotor`'s mean.
At n = 5 with one dominant draw this is a flag for the manuscript, not a
measurement: the leaked threshold appears to have been doing *stabilising*
work, not only accuracy work, which is consistent with `rotor` losing its
operating point entirely.

### Two caveats that must travel with these numbers

1. **The `tearing` frontier is sparse where it matters.** Its swept points
   jump from patch 0.0727 (phys 4.126) to patch 0.6250 (phys 0.00443), so
   the interpolated value at 0.3846 spans a wide, strongly non-linear gap.
   The 2.1× is an order-of-magnitude statement, not a measurement.
2. **`rotor`'s leak-free budget is outside the swept range** (0.056–0.138
   against a frontier starting at 0.152), so no ratio exists for it at all.

### What this settles about D13

The register listed D13 as *"measured, not removed"*, with the note that the
leak favours Q-HAS and the conclusion is conservative because Q-HAS loses
anyway. That is now **measured rather than argued**: with the leak removed,
Q-HAS is not merely still beaten — it is beaten by a wider margin, and on
`rotor` it cannot complete a trajectory at all.

It also **reverses the one transfer result that had favoured Q-HAS**. Under
the leak, `tearing` was the single separable fold and Q-HAS degraded *less*
(0.166 against 0.389). Leak-free, the same fold gives Q-HAS **×0.685
against the classical arm's ×0.389** — Q-HAS now degrades *more*. The
apparent transfer advantage was an artefact of the leaked threshold.

### How these runs survive the container, and what that puts in the artifact

A reviewer will find `resumed_from_checkpoint`, `n_runs_resumed`,
`status: "partial"` and `partial_stage` in these files. They exist because
a leak-free fold costs ~4 h on `kh` and `ot` while this container is
reclaimed roughly every 1.5 h. Two mechanisms, and the second is what
actually made those folds possible:

1. **Checkpoint after every draw.** `t22` writes its state after each
   individual run (~7 min of exposure, not the ~35 min a whole condition
   would cost). Every such write is marked `status: "partial"` with
   `partial_stage` naming the exact draw (`qhas/canonical 3/5`), and
   **both consumers (`t24`, `t22c`) refuse to analyse it** — its arm
   statistics are computed over however many draws finished, which is not
   a result. Without that marking the safety measure would have introduced
   the very defect this campaign documents.

2. **Resume from the checkpoint.** Checkpointing alone only *preserved*
   data: each relaunch restarted from draw 1, so `kh` and `ot` could never
   finish however many times they were run. `t22` now reloads the partial
   artifact and skips the draws already made. It resumes **only** from a
   `partial` record whose fold, mode, `repeats` and `matched_reference` all
   match, and refuses aloud otherwise rather than blending incomparable
   draws; `--no-resume` forces a clean recomputation.

**What resuming does and does not cost.** The reused draws come from a
different process. That has no statistical effect here — the Q-HAS arm is
non-deterministic (D11), the draws are i.i.d., and the classical arm
reproduces bit-exactly — but it is recorded rather than left invisible,
because an artifact that does not say where its data came from is exactly
the failure mode catalogued above. A fold whose `n_runs_resumed` is
non-zero is not weaker evidence; it is evidence that says so.

### Why only 2 folds so far, stated rather than left to be inferred

`ot` and `kh` are the two most expensive folds (T20 spent 3402 s and
3046 s on them respectively, against 2735 s for `rotor`). A leak-free run
is 14 simulations, and this container is reclaimed roughly every 1.5 h —
the campaign has now lost these two folds to reclamation **three times**,
twice as a pair sharing 4 CPUs and once mid-DNS. They are being run one at
a time instead. If they land, this entry gets two more rows; if they do
not, the finding stands on `rotor` and `tearing` and **the sample size is
2 of 4, not 4 of 4**, which is why the closing section says so explicitly.

Nothing about the two completed folds changes either way: they were run to
completion with the abort status captured per draw, and `t16` checks their
numbers (`t24/*` rows).


---

## T25 — robustness to the physics, and the "≥ 3 seeds" requirement

```bash
python study/h4_transfer/h4_physics_robustness.py --fold <f> --repeats 3
python study/h4_transfer/h4_physics_robustness.py --fold <f> --recompute
```

### First: there is no physics seed to vary

The pre-registration asks for ≥ 3 physics seeds per class, and this study
declared "1 seed per class" as a limitation throughout. **Both statements
are mis-specified.**

| scenario | randomness in its initial condition |
|---|---|
| `init_kelvin_helmholtz` | **none.** `noise_amplitude` multiplies `sin(X)` — a deterministic *mode* |
| `init_harris_tearing` | **none.** `perturbation` multiplies `cos(k·X)` |
| `init_orszag_tang` | **none**, and no parameters at all |
| `init_mhd_rotor` | a real RNG, but `np.random.default_rng(42)` is **hard-coded** |

And the one real seed **does not move the physics**: changing it 42 → 7
shifts the DNS trajectory signature by **0.0022 %**, because the RNG enters
only as `perturbation * standard_normal(...)` with `perturbation = 0.005` —
a symmetry breaker on a field of O(1). So a seed sweep was never possible in
three classes and would have measured nothing in the fourth. **The declared
limitation was not a limitation; it was a non-experiment.**

### What was run instead

The lever that does move the physics is the initial-condition *parameter*.
T25 evaluates each fold on additional initial conditions, comparing Q-HAS
against a classical frontier **built on that same condition** and placed by
bisection on the budget Q-HAS actually realised there.

| fold | condition | trajectory shift | verdict |
|---|---|---|---|
| `rotor` | `rotor_seed7` (true seed 42→7) | 0.0022 % | **vacuous** — skipped |
| `rotor` | `rotor_b` | 21.03 % | **0.86× — Q-HAS BETTER** |
| `tearing` | `tearing_b` | 19.84 % | no verdict — frontier anti-monotone |
| `tearing` | `tearing_c` | 8.16 % | no verdict — budget outside swept range |
| `kh` | `kh_b` | 6.53 % | **1.24× — Q-HAS worse** |
| `kh` | `kh_c` | 3.85 % | no verdict — bisection unconverged |
| `ot` | `ot_re900` (Reynolds, not an IC) | 0.12 % | **vacuous** — skipped |

**7 conditions attempted, 2 vacuous, 3 refused, 2 decidable — one each way.**

### The honest reading

> **On genuinely different initial conditions the direction of the result is
> not established.** It holds on `kh_b` and reverses on `rotor_b`.

This does **not** overturn the closed-loop result, which is measured on the
canonical conditions against T15b's dense bisected frontier with proper
budget matching. It does bound its scope: *Q-HAS is worse on the initial
conditions studied*, not *Q-HAS is worse in general*. Any manuscript claim
must carry that boundary.

### Why three conditions produced no verdict, and why that is reported

On alternative initial conditions the classical relation budget → error is
often **not monotone**: on `tearing_b`, refining from budget 0.625 to 0.874
makes the error **30× worse** (0.012 → 1.289). "The attainable classical
error at budget X" is undefined on such a set, yet `np.interp` answers with
a normal-looking number — and it had already printed **1.28×** as a result.

`frontier_verdict()` therefore refuses unless the bracketing interval is
locally sound: error non-increasing with budget, points within 5×, and the
bisection converged to within twice its own declared tolerance. Each refusal
carries its reason in the artifact.

**Which way the guards cut, stated because it is checkable:** all three
criteria removed evidence *favouring* the study (`tearing_b` 1.28×, `kh_c`
7.02×), and the single result *contradicting* it (`rotor_b` 0.86×) survived
all three. If these filters are biased, they are biased against the claim
this study makes.

### What T25 cannot say

- **Nothing about magnitude** — n = 3 draws per condition, and on `kh_c` two
  draws at the same budget differed by 1.9×.
- **Nothing from an independent seed axis** — it does not exist. The
  physics-robustness evidence rests entirely on parameter variation.
- **Nothing about `ot`** — no IC parameters exist, and its Reynolds lever
  shifts the trajectory 0.12 %.

---

## T26 — l'inertie des couplages est un artefact de PETITE TAILLE

```bash
python study/h3_representation/h3_size_scan.py --dims 2 4 8 --n-snaps 3 --mapper v1
python study/h3_representation/h3_size_scan.py --dims 2 --force-greedy   # contrôle
```

### Pourquoi cette tâche existe

T13 et T18 montrent que les couplages ZZ/ZZZZ changent **exactement 0**
décision, et que réparer la fenêtre n'y change rien. Ces résultats sont
exacts — mais mesurés à `dim = 2`, soit **8 qubits**, précisément le régime
où l'état fondamental est uniforme sur 100 % des instantanés. L'objection
évidente est : *« à 8 qubits, évidemment »*. Elle est fondée, et c'était la
faiblesse centrale de l'étude.

### Résultat

| dim | qubits | méthode | no_ZZ | no_ZZZZ | Z_only | uniformité du fondamental |
|---|---|---|---|---|---|---|
| 2 | 8 | exhaustive | 0.0000 | 0.0000 | 0.0000 | **1.00** |
| 2 | 8 | glouton *(contrôle)* | 0.0000 | 0.0000 | 0.0000 | 1.00 |
| 4 | 32 | glouton | 0.0000 | **0.0312** | **0.0312** | 0.75 |
| 8 | 128 | glouton | **0.0469** | **0.0690** | **0.0794** | **0.17** |

> **L'inertie casse avec la taille.** À 32 et 128 qubits, ablater les
> couplages change des décisions. Et l'uniformité de l'état fondamental
> s'effondre en parallèle : 1.00 → 0.75 → 0.17.

Les deux phénomènes vont ensemble et forment un mécanisme cohérent : tant
que l'optimum est un masque constant, aucun couplage ne peut le déplacer ;
dès que la structure combinatoire apparaît, les couplages redeviennent
causaux.

### ⚠️ Mais « changer une décision » n'est PAS « mieux détecter »

Le tableau ci-dessus mesure l'influence **causale** des couplages, pas leur
**utilité**. La question d'origine du projet est la détection des patches
durs à grossir. Elle se mesure contre la vérité terrain
(`l2_errors >= l2_threshold`), et elle donne :

| dim | qubits | F1 hamiltonien complet | F1 Z seul | F1 règle classique | **gain des couplages** |
|---|---|---|---|---|---|
| 2 | 8 | 0.3333 | 0.3333 | **0.3889** | **+0.0000** |
| 4 | 32 | 0.5199 | 0.5524 | 0.5524 | **−0.0325** |
| 8 | 128 | 0.5916 | 0.6481 | 0.6481 | **−0.0565** |

> **Les couplages ne détectent jamais mieux, et à grande taille ils
> détectent MOINS BIEN.** Quand ils deviennent causalement actifs, leur
> effet est de dégrader le F1 : −0.033 à 32 qubits, −0.057 à 128.

Trois lectures qui en découlent, toutes vérifiables dans la table maîtresse :

1. **Le meilleur cas de la formulation Ising est d'égaler la règle de
   seuil.** À dim = 4 et 8, `F1(Z seul) = F1(classique)` **exactement**
   (0.5524 et 0.6481) : le hamiltonien réduit à son biais reproduit la règle
   classique, terme pour terme.
2. **Ajouter les couplages retire de la performance.** Ils n'apportent pas
   du signal, ils apportent du bruit.
3. **Le F1 monte avec `dim` (0.33 → 0.55 → 0.65) pour les deux bras
   identiquement** — c'est le raffinement du découpage qui aide, pas la
   couche quantique. Attribuer cette montée au quantique serait une erreur
   de lecture.

**Correction d'une formulation antérieure de cette section.** J'avais écrit
que la rupture d'inertie « ouvre un horizon » et était « plus intéressante à
publier qu'un résultat négatif ». C'était prématuré : la frontière existe,
mais de l'autre côté les couplages **nuisent**. Ce n'est pas un horizon,
c'est la fermeture propre de la porte — avec, cette fois, la mesure qui
répond à la question d'origine du projet.

### Le contrôle qui rend ce résultat lisible

L'énumération exhaustive est refusée au-delà de 22 qubits, donc dim ≥ 4
utilise la descente gloutonne à chaud. Le risque évident : que ce soit **le
proxy** qui fabrique les changements, pas les couplages.

Deux garde-fous, tous deux passés :

1. **Le contrôle `full` vaut 0.0000 à toutes les tailles.** Rejouer sans
   ablation redonne exactement la même décision : le glouton est
   déterministe à hamiltonien et amorce fixés, donc tout écart non nul est
   *causé* par l'ablation.
2. **`--force-greedy` à dim = 2** — là où l'exhaustif dit 0.0000 — donne
   également **0.0000**. Le proxy ne fabrique pas de changements dans le
   régime où l'on peut le vérifier.

⚠️ **Réserve à conserver.** Le glouton et l'exhaustif ne choisissent pas le
même masque sur 25 % des cellules à dim = 2 (accord 0.7500), tout en étant
tous deux insensibles à l'ablation. Le scan mesure donc *« les couplages
changent-ils la décision du solveur déployé »*, pas *« l'optimum exact
change-t-il »*. C'est la question opérationnelle — le pipeline n'utilise pas
l'exhaustif non plus — mais elle doit être citée telle quelle.

### Ce que ça change pour les conclusions de l'étude

**Ce qui reste vrai :** à la taille déployée (`VQA_N = 2`, 8 qubits), la
formulation est inerte, et c'est exact.

**Ce qui devient faux :** toute lecture du type *« cette famille de mappings
Ising est intrinsèquement inerte »*. Elle ne l'est pas. Elle l'est **à 8
qubits**, et cesse de l'être avant 32.

**Ce que ça ferme :** l'espoir que la formulation devienne utile en
montant en taille. Les couplages deviennent actifs mais nuisibles, sur toute
la plage testée (8 → 128 qubits). Le meilleur cas de cette famille de
mappings est d'égaler la règle de seuil qu'elle est censée remplacer.

**Ce qui reste ouvert :** la localisation exacte de la transition (entre 8
et 32 qubits ; dim = 3, 18 qubits, serait encore exhaustivement vérifiable
mais demande un DNS à `N` divisible par 3), et surtout **une autre
construction de couplages** — le diagnostic F1 ci-dessus est le test que
toute nouvelle proposition devrait passer avant d'être revendiquée.

---

# CLOSING THE CLOSED-LOOP STUDY (Level 3)

Everything below is measured, carries the control that validated it, and is
covered by `t16_aggregate_v4.py` (**180 rows, 180 OK, 0 DIFF, 0 MISSING**).

## The one-sentence result

> Across four held-out instability classes, a Q-HAS closed loop is less
> faithful than a plain threshold rule at matched compute on **18 of 18**
> completed repeated runs, more expensive on **16 of 18**, and strictly
> Pareto-dominated on **16 of 18**. At that same operating point it also
> **aborts on 2 of 20 draws where the classical rule aborts on 0 of 8**.
> And when the one undue advantage that *can* be taken away — a decision
> threshold fitted on the held-out class (**D13**) — is removed, it does not
> recover: it gets **worse still** on every fold where a comparison is
> possible, and **12 of its 40 leak-free draws fail to complete a
> trajectory at all**, against 0 of 16 for the classical arm.

Each clause is recomputed from its artifact by `t16_aggregate_v4.py`
(rows `t23/*`, `t24/*`). None of it is transcribed.

**Read the abort clause narrowly.** It says the classical arm did not
abort *at the compared operating point*. It does abort elsewhere — T19
records `rotor`'s tuned classical threshold diverging at step 208, and 2 of
that fold's 6 bisection points. Divergence is a property of the threshold;
both arms have thresholds that diverge. What is asymmetric is that at the
point where they are compared, one arm completed and the other did not.

**Scope boundary, from T25.** Everything above is measured on the
**canonical initial conditions**. On genuinely different initial states the
direction is **not established**: of 7 alternative conditions, 2 were
vacuous, 3 gave no sound verdict, and the 2 decidable ones split one each
way (`kh_b` 1.24× for, `rotor_b` 0.86× against). The claim is therefore
*Q-HAS is worse on the initial conditions studied*, not *in general*. And
the pre-registered "≥ 3 physics seeds" was never available: three of four
scenarios have no RNG at all, and the fourth's hard-coded seed moves the
trajectory by 0.0022 %.

**The D13 clause is measured on all 4 folds**, and it is a *bound*:
`--mode leak-free` substitutes the threshold without re-tuning the QAOA
arm. The definitive version — `threshold_amr` back in the Optuna search
space, excluded from the held-out class — is not attempted. What the bound
says: Q-HAS above the classical frontier at its own realised budget on
**3 of 3 measurable folds** (1.6×, 1.9×, 2.1× canonical), **no operating
point at all** on the fourth, and **12 of 40 draws aborting against 0 of
16** for the classical arm.

## What the closed loop establishes, by strength of evidence

**1. Direction — robust, no free parameter.** The verified Q-HAS mean
exceeds the budget-matched classical value on **4 folds of 4** (1.30×, 1.90×,
2.74×, 1.81×). The pre-registered `combined` endpoint gives the classical arm
the majority at **every λ on the swept grid** (12 points, 0 → 100) — the
verdict never flips, only the margin: 2–1 from λ = 0 through λ = 0.8, then
3–0 from λ = 1.0 onward. An earlier draft put the crossover at "λ = 0.82";
that precision is not available from a 12-point grid — all that is measured
is that the count changes somewhere in (0.8, 1.0]. The verdict, which is
what the claim rests on, does not change anywhere. Two of three usable folds
are decided by Pareto dominance alone, needing no λ at all.

**2. Robustness — a failure mode outside every metric.** `rotor`'s Q-HAS arm
aborted on **2 of 5** verified draws (40 %) while its classical control **at
the budget-matched threshold** completed every time, deterministically.
Across the recorded T20 and T22 artifacts: **2 Q-HAS aborts out of 20 draws,
0 classical aborts out of 8 replays at the matched point.** `phys_score`,
`patch_ratio`, the dominance count and the λ analysis all presuppose a run
that finishes.

**Do not read this as "the classical rule never diverges" — it does.** The
T19 audits record `rotor`'s *tuned* classical arm aborting at step 208
(threshold 0.4616), and 2 of `rotor`'s 6 bisection points aborting as well.
An earlier draft of this section claimed "six Q-HAS aborts against zero
classical across the campaign"; the second half of that is false and the
first is not reproducible from the artifacts, which record 2. The claim that
holds is narrower and is the one the comparison actually needs: **at the
operating point where the two arms are compared, the classical arm completed
every time and Q-HAS did not.** Divergence is a property of the threshold,
and both arms have thresholds that diverge.

**3. Transfer — no effect, and the one apparent effect was the leak.** On
genuinely unseen initial conditions, **1 fold of 4** shows a separable
difference in degradation; on `ot`, |z| = 0.02. The single-draw pass had
suggested Q-HAS transfers *better* on all four; repeated with 5 draws that
pattern evaporates.

The one fold that survived as separable was `tearing`, and it favoured
Q-HAS (degradation ×0.166 against the classical ×0.389). **Leak-free, that
reverses**: ×0.685 against ×0.389 — Q-HAS now degrades *more*. The single
transfer result in the study's favour was an artefact of the leaked
threshold, and removing the leak removes it. Nothing here supports a
transfer advantage in either the leaked or the leak-free setting.

**4. Magnitudes — not supportable.** Both variance passes report "1 fold of
4 separable" **but not the same fold** (`ot` 2.09 → 1.35, `rotor` 1.56 →
2.30). At n = 5 the separability verdict is itself unstable. Quote the
direction and the counts; **do not quote per-fold ratios**.

## Conditions under which the result was obtained — all adverse to the classical arm

The conclusion is **conservative**: three known asymmetries favour Q-HAS and
it loses anyway.

| asymmetry | direction | status |
|---|---|---|
| **D13** — QAOA threshold fitted on all 4 classes incl. the held-out one | favours Q-HAS | **removed and measured** (T22 `--mode leak-free`): without it Q-HAS is 2.1× worse than the classical frontier at its own budget on `tearing`, and aborts on 5/5 canonical draws on `rotor` |
| **cost axis** excludes the QAOA circuit; Q-HAS uses 2.7–3.3× the wall time on the three folds whose classical arm completed (`rotor` excluded: its 29 s classical run is the aborted tuned arm, not a comparable time) | favours Q-HAS | declared |
| aborted Q-HAS draws excluded from its own statistics | favours Q-HAS | necessary, declared |

## What would overturn it

- ~~removing D13 and finding Q-HAS wins~~ — **done, and it goes the other way**: leak-free, Q-HAS is worse still (2 folds of 4 measured so far). What would overturn the result is the *definitive* version, re-tuning the QAOA arm with `threshold_amr` in its Optuna search space on the training classes, which is not attempted;
- ≥ 3 physics seeds per fold showing the direction is seed-specific;
- the full 170-trial Optuna budget lifting Q-HAS above the matched classical;
- counting decision cost, which would only make the result stronger.

## What this study cannot say

- **Nothing about magnitude** per fold (n = 5, unstable separability).
- **Nothing about transfer on `ot`** — its unseen condition shifts the
  trajectory by 0.3 %, `init_orszag_tang()` taking no parameters.
- **Nothing about hardware**: the circuit is simulated, 8 qubits, noiseless.
- **Nothing about larger `VQA_N`**: everything here is the deployed depth-0
  size where the ground state is uniform (Claim A).

## The methodological finding, stated for the manuscript

**Seventeen distinct instances** of one failure mode were found and fixed:
**a computation that fails, or does not do what it says, but returns a value
indistinguishable from a valid one**. Twelve were found by auditing code:

| form | count | where |
|---|---|---|
| V1's divergence guard returns a partial score with identical keys | 4× | T15, T20, T22 (×2) |
| a fixed output filename silently overwrites the prior result | 6× | T13 mappers, T19 folds, T20 pass, then T11, T11b, T12 (`--mapper` absent from the name) |
| an aggregation averaging aborted draws with valid ones | 1× | T16 |
| a CLI mode accepted and documented but never implemented | 1× | `--mode no-leak`: only the filename changed |

**Four of the twelve were in the verification code written to catch the
others**, and three more were found only by `tests/study/test_silent_failure_sweep.py`,
which sweeps the mechanically checkable forms. Searching as you go is
demonstrably not enough.

### Five more, found by auditing the documents against the artifacts

The twelve above were found by auditing *code*. A final pass audited the
**published numbers** instead — recomputing each from its artifact — and
found five more instances, in the write-up and in the verification code:

| # | instance | consequence |
|---|---|---|
| 13 | a total abort discarded before saving: `SystemExit` fired before any artifact was written, and on the *first* arm, so the question that mattered (does the classical rule survive that threshold?) went unmeasured | the mirror of the motif — a real outcome made indistinguishable from a run never launched |
| 14 | **the headline count was written by hand**, not computed. 19/20, 18/20, 17/20 did not reproduce: `kh`'s two columns were transposed and `rotor`'s 2 aborted draws sat in the denominator | the study's most-quoted number was wrong; correct is **18/18, 16/18, 16/18** |
| 15 | **D14** — T20's artifact says `classical_reference_source = "budget-matched classical"` next to a `classical_stats` block that, on `ot` and `kh`, was computed at the *tuned* threshold | 0.4845 against 0.0827 on `ot` — enough to invert that fold for anyone recomputing from it |
| 16 | **D15** — `git_commit_hash()` taken at *save* time | hour-long runs stamped with code committed while they ran; the `ot`/`kh` artifacts point at a fix they never executed |
| 17 | `t22` printed *"at the SAME operating point the classical arm completed"* in leak-free mode | the two arms differ by a factor of six in threshold; the sentence would have turned a budget difference into an arm-specific instability claim |

Two more errors of a related but distinct kind — **false precision** rather
than false results — were fixed in the same pass: a λ crossover quoted as
"0.82" from a 12-point grid that only locates it in (0.8, 1.0], and a
published figure still annotating the retracted single-draw ratios.

**The pattern is the finding, and it is sharper than "check your code".**
Every number that no script produced turned out to be wrong. Every number
`t16_aggregate_v4.py` recomputes from its artifact was right. The defence
that worked was not care, review, or re-reading — all of which were applied
throughout and all of which missed these — but **making the number a
function of the artifact and checking it mechanically**. Anything published
as prose is unverified by construction.

One aborted draw returned `phys = 0.4069` against valid draws of
0.054–0.219: **contamination need not be visible in the values**.

**And its direction cannot be bounded either.** On `ot` leak-free the three
aborted draws returned 0.4311, 0.4239 and 0.4529 while the one draw that
*completed* returned **0.6587** — the invalid runs looked **better** than
the valid one. The mechanism is plain once seen: those runs stopped near
step 930 of ~1136, so the trajectory had less time to depart from the DNS
reference and accumulated less error. On `rotor` the opposite happened,
because there the abort came *after* the fields blew up.

So the tempting bounding argument — *"an aborted run scores badly, so
including it is conservative"* — is **empirically false**. Contamination
inflates the error when the blow-up is captured and deflates it when the
run is merely truncated, and which one you get depends on where the guard
fires. There is no safe direction to assume, which is why the status has to
be captured at execution time rather than inferred from the value. Any
closed-loop AMR study of this kind should record run completion status at
execution time, because with a non-deterministic arm it cannot be recovered
afterwards.

---

# THE V1 TEST SUITE, RE-ARMED

Base commit `d3d8fe6`. Commands:

```bash
python -m pytest tests/ --ignore=tests/study -q
python -m pytest tests/study -q
```

## Before: 44 of 175 tests were failing, and no green gate existed

```
44 failed, 131 passed in 258s
```

**42 of the 44 had a single cause**, and it was mechanical:

```
TypeError: PhysicalMapper.__init__() got an unexpected keyword argument 'beta'
```

`beta` was split into `beta_curl` / `beta_xpoint` (`src/Simulation/HamiltParams.py:63`)
and the call sites in `tests/` were never updated. The code was not broken —
the tests were stale. The consequence is what matters: `test_beta_xpoint.py`,
`test_vqa_anomaly_cases.py`, `test_module_validation.py` and the four
`test_qaoa_*` files **had verified nothing since that refactor**, i.e. the
Hamiltonian layer — the object of the whole study — was unguarded.

`run_tests.sh` is `set -e` and `run_stage` exits on the first non-zero code
(`run_tests.sh:154`), so the default run aborted at **stage 2**
(`test_v9_metrics.py`). There was no passing gate on V1 to regress against.

Repair: `beta=X` → `beta_curl=X, beta_xpoint=X` at 18 call sites, which is
the exact historical semantics (a shared `beta` fed both sensitivities,
`HamiltParams.py:88-92`). No file under `src/` was touched.

## After

```
175 passed          (V1 suite)
325 passed, 15 skipped   (tests/v3 + tests/v4)
```

## The six assertions that had to be inverted, and why

Two failures were not stale — they were **correct measurements of a broken
claim**. Four more of the same kind surfaced once the 42 came back to life.
All six assert that a coupling is present; all six measure its annihilation
by the Gaussian uncertainty window `exp(-((score - threshold_amr)/sigma)^2)`
that multiplies `C_edges`.

The clearest of them, `test_v9_metrics.py`, carried this docstring:

> *"This is the core v9 claim: the Hamiltonian adds spatial correlation
> information BEYOND what θ init provides."*

and failed by 42 orders of magnitude. Measured, on a 2x2 periodic grid with
a sharp velocity boundary (`score` uniform at 0.5, `threshold_amr = 0`,
`sigma = 0.05`):

| quantity | value |
|---|---|
| `max abs(C_edges)` delivered | **1.7858e-42** |
| same call at `threshold_amr = score` (window = 1) | **4.8005e+01** |
| ratio | **3.7201e-44** |
| `exp(-((0.5 - 0)/0.05)^2) = exp(-100)` | **3.7201e-44** |

The ratio equals the window to full double precision. The gradient signal is
computed correctly, at O(48), and then multiplied by ~1e-44.

On Orszag-Tang (N=64, 30 steps, score spanning [0.5057, 0.8748]):

| sigma | `max abs(C_edges)` | `max abs(K_plaquettes)` |
|---|---|---|
| 0.05 (deployed) | **1.7727e-48** | 2.3629e+01 |
| 10 (window open) | **6.3187e+01** | 2.3629e+01 |

`K_plaquettes` is bit-identical across the two, which is what makes the
attribution airtight: `sigma` reaches ZZ and nothing else. The four cases in
`test_vqa_anomaly_cases.py` give 1.79e-42, 1.86e-42, 1.11e-38 and 1.23e-85
by the same mechanism.

Each of the six now asserts three things instead of one: the delivered
coupling is dead (`< 1e-30`), the same fields with the window open return an
O(1) coupling, and — where the score is uniform enough to make it exact —
the ratio equals the window. A test that merely recorded "it is zero" would
not distinguish *annihilated* from *never computed*.

**This is an independent corroboration of T13/T17/T18, written before this
study existed.** V1's own unit tests contained the falsification of V1's
central claim, in red, for the whole life of the project.

## Three defects found while re-arming, none previously recorded

**(a) The Z-bias scale is a function of the threshold** —
`test_qaoa_physics_decision.py`. `H_edges` is documented as
`alpha_z * (score - threshold_amr)`. It is linear in `score` at fixed
threshold (the recovered ratio is constant to 1e-9), but `alpha_z` is
normalised by `median(nonzero |C|, |K|)`, and `|C|` carries the window — so
`alpha_z` inherits the threshold dependence. On a shear layer whose score
takes exactly two values, 0 and 0.5:

| `threshold_amr` | `max abs(C_edges)` | recovered `alpha_z` |
|---|---|---|
| 0.20 | 1.167e+01 | **8.7857e-01** |
| 0.50 | 4.404e-10 | 1.4930e-03 |
| 0.95 | 2.396e-84 | 5.0750e-03 |

Same fields, same score, Z-bias scale moving by **173x** and
non-monotonically with the threshold alone. The old test asserted
monotonicity and was simply wrong about the model it was testing.

**(b) The vortex detection test was measuring shot noise.** With
`args.shots = 4096` each marginal carries a standard error of ~0.008. Over
12 draws on identical fields, the Lamb-Oseen contrast was

```
[+0.0141 -0.0147 -0.0267 -0.0043 -0.0305 +0.0060
 +0.0036 -0.0125 +0.0067 +0.0236 -0.0084 +0.0079]
mean = -0.0029, std = 0.0156
```

centred on zero with a **sign that flips run to run**, and clearing the old
`abs(contrast) > 0.01` bar on exactly 50% of draws. The test now runs 10
draws and asserts the mean is null and the sign is not reproducible — which
is the finding, and is consistent with the uniform ground state at this size.

**(c) The QAOA arm's displacement is not a single-draw quantity.** The
max-marginal displacement against `sin^2(theta/2)` ranged over
**0.0721 to 0.4742** across 12 identical calls (mean 0.2867). The assertion
is now on the median of 5 draws. Same root cause as D11: unseeded COBYLA
plus a shot-based sampler.

## The harness finding: 8 of the 17 default stages cannot fail

Independent of the 44, and larger:

| stage | assertions | wall time |
|---|---|---|
| `tests/test_qaoa_noise_and_early.py` (2 tests) | **0** | 14m40 + 1m38 |
| `tests/test_qaoa_scaling_and_hparams.py` (2 tests) | **0** | 16m04 |
| `tests/test_qaoa_advantage.py` | **0** | script |
| `tests/test_qaoa_decisions.py` | **0** (0 test functions) | script |
| `tests/diag_hamiltonian_balance.py` | **0** | script |
| `tests/diag_qaoa_contribution.py` | **0** | script |
| `tests/diagnose_convergence.py` | **0** | script |

They print and return 0. `run_stage` reports `PASSED`. Over **32 minutes**
of the default run is spent in files that contain no assertion at all — and
what they print is not neutral:

- `test_qaoa_advantage.py` ends with the winner column reading `Classical`
  on **6 of 6** rows (rotor 2x2/3x3, KH 2x2/3x3, OT 2x2/3x3) and exits 0;
- `diag_qaoa_contribution.py` ends with
  `⚠ ALL Z biases negative → QAOA ground state = refine nothing` and exits 0;
- `test_noise_robustness` averages Spearman rho values that are **NaN** on
  some trials (`ConstantInputWarning: An input array is constant`) without
  saying so.

This is the study's own motif at the level of the harness: *a stage that
verifies nothing is indistinguishable from a stage that passed*. The 44 red
tests were visible; these eight were green.

**Not fixed here**, because it changes the meaning of the gate and the
acceptance criteria would have to be invented rather than measured: either
give those stages real assertions, or move them out of the default path into
the existing `--figures` / `--diagnose` groups so the default run is
assertion-bearing end to end.

---

# THE EIGHT STAGES THAT COULD NOT FAIL, AND WHAT THEY SAY NOW

Base commit `fe1f6fe`. Nothing under `src/` was modified; the source
behaviours below are pinned from the test side.

## Every default stage now carries an acceptance check

| stage | it now asserts | reference |
|---|---|---|
| `test_qaoa_advantage.py` | QAOA outranks the classical baseline on at most 1 of 6 scenario/size pairs, and the mean rank-correlation gap exceeds 0.15 | 0/6 wins, gap **+0.692** |
| `test_qaoa_decisions.py` | the 7 internal checks match their recorded pattern exactly | **5 hold, 2 known defects** |
| `test_qaoa_noise_and_early.py::test_noise_robustness` | without noise the classical arm reaches the optimum and QAOA loses by > 0.10 captured fraction; QAOA wins at most 4 of 12 rows, none below sigma = 0.20; a NaN rho occurs only when a score map is constant | 0.6588 vs 0.3350 and 0.3183 vs 0.1976; 2/12 wins, both at sigma = 0.30 |
| `test_qaoa_noise_and_early.py::test_early_detection` | QAOA wins at most 2 of 6 rows and never exceeds the classical mean captured fraction by more than 0.02 | 1/6 wins; means **0.4065 vs 0.3735** |
| `test_qaoa_scaling_and_hparams.py::test_resolution_scaling` | on clean data QAOA never exceeds the classical arm at N = 32, 64, 128, and the classical arm improves with resolution | 0.5182 / 0.6588 / 0.7669 classical, QAOA 0.5182 / 0.6588 / 0.2438 |
| `test_qaoa_scaling_and_hparams.py::test_hyperparameter_sweep` | over 4 w_z_frac x 3 thresholds, the best result on clean data is an exact tie with the classical baseline | best delta = **+0.0000**; 4 exact ties at threshold 0.3, **-0.4048** everywhere else |
| `diag_hamiltonian_balance.py` | the downsampled ZZ block does not move with beta_curl/beta_xpoint, no ZZZZ survives downsampling, and Z/ZZ magnitude stays below 1e-3 | max abs(K) = 0 exactly; max abs(H) ~ 9.6e-05 against max abs(C) ~ 1.0031 |
| `diag_qaoa_contribution.py` | at the operating threshold the QAOA flips at most 2 of 48 decisions, every run at threshold 0.5 has all-negative Z biases, and the multi/single energy ratio exceeds 1e4 everywhere | **0/48 flipped**; 12/12 all-negative; ratio 6.4e4 to 6.2e8 |
| `diagnose_convergence.py` | its own four printed verdicts become the exit code | B1-B4 all PASS |

The most quotable line of that table is the hyperparameter sweep: **the best
the QAOA arm ever does on clean data, over the entire sweep, is to equal the
classical baseline exactly.** Twelve combinations, one ceiling, and it is a
tie.

`test_qaoa_advantage.py` and `diag_qaoa_contribution.py` were printing
`Classical` on 6 of 6 rows and
`ALL Z biases negative -> QAOA ground state = refine nothing` respectively,
and exiting 0. Those two lines are now the acceptance criterion instead of
decoration.

## The placeholder Hamiltonian is now detectable — `tests/test_v1_guards.py`

`cost_hamiltonian.py` drops every coefficient below **1e-6** and, when that
empties the term list, appends `("Z", [0], 1e-3)` so Qiskit does not choke on
an empty observable. Three properties are now pinned:

1. **the substitute is 1e6 times the signal it replaced** — with every
   coefficient at 1e-9, the operator delivered to the solver is a single term
   at 1e-3;
2. **it is not physically neutral**, contrary to the source comment. Every
   ground state of `("Z", [0], +1e-3)` has qubit 0 excited, with
   `E_min = -1e-3`: the placeholder is a *refine-edge-0* bias;
3. **it escapes the null-Hamiltonian shortcut.** `execute.py:52` skips COBYLA
   when `np.allclose(abs(coeffs), 0.0)`, whose default `atol` is 1e-8. The
   placeholder sits at 1e-3, so a patch with no surviving coefficient runs a
   full variational optimisation against a fabricated operator. The two
   thresholds live in different files and nothing else connects them.

`is_null_placeholder(op)` is the detector to call before interpreting any
operator coming out of V1: a placeholder means *no Hamiltonian was built*,
which is a different event from *the Hamiltonian is weak*.

The same file pins the pruning chain — `max abs(C_edges)` is nonzero and
below 1e-6 on a real 2x2 patch, and **zero ZZ terms** appear in the operator,
while a coupling above the cut produces one ZZ term per site — and exercises
the assignment that `execute.py:182-185` performs inside
`try/except Exception: pass`, on both primitive construction paths, so that a
silently under-sampled MPS readout fails here instead of hiding there.

## Four more V1 claims that were false

Re-arming the suite made these visible; each is measured over repeated draws
because the arm is stochastic.

| claim as written | measured | n |
|---|---|---|
| `test_signal_contribution::test_psi` — "phase anticipation": high psi marks a growing instability | contrast **-0.0572** (t = -8.4), negative in 93% of draws — psi LOWERS the cell it marks | 30 |
| `test_qaoa_physics_decision::test_spatially_varying_psi...` — same mechanism, different construction | **-0.0723** (t = -14.6), positive in 3% of draws | 30 |
| `test_signal_contribution::test_K_ZZZZ` — a 6x stronger plaquette should raise its four qubits | **-0.0168** (t = -7.1) — it lowers them | 30 |
| `test_signal_contribution::test_C_ZZ` — a 10x stronger ZZ coupling should raise its edge | **+0.0072**, sem 0.0049, **t = +1.46** — indistinguishable from zero | 30 |

The two psi rows are the same finding reached from two independent setups:
**the "phase boost", which is the mechanism the early-detection story rests
on, has the opposite sign to the one claimed.** Both old assertions took the
absolute value of the contrast, which is exactly why the sign was never seen.

The C_ZZ row belongs with T13/T18/T26: a coupling ten times the background
moves nothing measurable at the deployed size.

## Six single-draw assertions on a stochastic arm

Beyond the four above, these were passing or failing by luck. All are now
stated over repeats, and the magnitude threshold was replaced by a *sign*
criterion wherever the mean itself drifts between sessions (unseeded COBYLA):
one run of `test_psi` returned -0.0183 where another returned -0.0572, while
the sign held in both.

| test | old assertion | draws clearing it | now |
|---|---|---|---|
| `QAOA_test::test_vortex_discriminates` | single draw, abs(contrast) > 0.01 | **25%** | mean over 8 draws is not positive (recorded -0.0058 +/- 0.0064) |
| `test_qaoa_physics_decision::test_vortex_detected` | single draw, abs(contrast) > 0.01 | **50%** | mean over 10 draws null, sign not reproducible |
| `test_qaoa_physics_decision::test_qaoa_converges_for_simple_hamiltonian` | single draw, avg P(1) > 0.7 | **90%** | median of 5 draws (mean 0.829, min 0.676) |
| `test_qaoa_physics_decision::test_qaoa_modifies_probabilities...` | single draw, max diff > 0.05 | ~92% | median of 5 draws (range 0.0721 to 0.4742) |
| `test_signal_contribution::test_H_Z` | single draw, contrast > 0.01 | ~95% (min -0.018) | mean over 20 draws > 0.02 |
| `test_signal_contribution::test_K_ZZZZ` | single draw, abs(contrast) > 0.01 | **87%** | sign over 20 draws |

## Gate

```
184 V1 tests pass (175 repaired + 9 new guards), four consecutive runs
325 v3/v4 tests pass, 15 skipped
9 of 9 default script/pytest stages carry an acceptance check
```

---

# V1 NE FABRIQUE PLUS D'HAMILTONIEN QUAND IL N'Y EN A PAS

Modification de `src/` (première depuis le gel de V1), commit parent `32d124a`.

## Ce qui change

`cost_hamiltonian.py` élague tout coefficient sous `COEFF_MIN = 1e-6`.
Quand il ne reste rien, il ajoutait `("Z", [0], 1e-3)` pour éviter le crash
Qiskit sur observable vide. Il lève désormais **`NullHamiltonianError`**.

`execute.py:184` : le `try/except Exception: pass` autour de
`sampler.options.default_shots = mps_shots` est supprimé. Si l'affectation
échoue, la lecture MPS tournerait au mauvais nombre de tirs.

`refinement.py` attrape l'exception, **conserve la décision classique** du
patch, et l'enregistre dans `null_hamiltonian_patches()`. Le VQA n'est pas
appelé. C'est un changement de comportement assumé : l'ancien chemin faisait
tourner COBYLA contre un opérateur dont l'état fondamental excite le qubit 0.

## Ce que la levée d'erreur a révélé immédiatement

Trois tests de V1 comparaient une anomalie à une « ligne de base calme ». Ils
échouent maintenant, et la raison est le résultat :

| champ | Hamiltonien construit ? | max abs(H) | max abs(C) | max abs(K) |
|---|---|---|---|---|
| cisaillement | oui | 1.670e+00 | 1.786e-42 | 2.227e+01 |
| **calme (vx = 1.0)** | **non** | — | — | — |
| point X | oui | 3.462e+00 | 8.518e-86 | 4.300e+01 |
| **calme (vx = 0.01)** | **non** | — | — | — |
| combiné | oui | 2.392e+00 | 1.113e-38 | 2.328e+01 |
| **calme (vx = 0.0)** | **non** | — | — | — |

Les trois lignes de base n'avaient **aucun** coefficient au-dessus de 1e-6.
Elles recevaient le terme de remplissage, et l'écart de marginales mesuré
contre elles — l'assertion « le cisaillement produit une réponse VQA
différente du calme » — était un écart contre un opérateur fabriqué.

L'énoncé correct est plus net : **sur un champ uniforme, la construction ne
produit rien à optimiser, et elle le dit.** Les trois tests l'affirment
maintenant ainsi, plus le contrôle que le champ anormal, lui, définit bien un
Hamiltonien.

`test_module_validation::test_zero_coefficients_filtered` testait
explicitement l'ancien comportement (« Should only have the safety term ») ;
il teste la levée, plus le contrôle qu'un seul coefficient au-dessus du seuil
suffit à construire l'opérateur.

## Gate

```
185 tests V1 (180 + 10 gardes, dont un bout-en-bout sur refinement.py)
325 tests v3/v4, 15 skipped
diag_qaoa_contribution.py : 0/48 décisions changées, exit 0
```

---

# La convention d'axes des mappeurs — T31

## Le fait, plus précis que « la vorticité est fausse »

`grid.py:4-13` déclare la convention `indexing='ij'` : `AXIS_X = 0`,
`AXIS_Y = 1`. Le solveur la respecte (`grid.grad`, `grid.div`,
`grid._compute_q_criterion`, `MHDSolver.get_fluxes`). Les trois mappeurs ne
la respectent pas : `HamiltParams.py`, `HamiltParams_v2.py` et
`PhysToAngle.py` forment leur rotationnel et leur divergence avec les axes
échangés.

Ce n'est pas une faute de frappe sur un signe, c'est une **convention
différente appliquée de façon cohérente** : les formules des mappeurs sont
exactement celles qu'on écrit sous `indexing='xy'`. Sous la convention que le
dépôt déclare, elles valent

| nom dans le code | ce qui est réellement calculé |
|---|---|
| `vorticity`, `omega_z` | ∂v_y/∂y − ∂v_x/∂x — différence des déformations normales |
| `div_v` | ∂v_x/∂y + ∂v_y/∂x — déformation de cisaillement (2·S₁₂) |
| `Jz_curl` | ∂B_y/∂y − ∂B_x/∂x |

Autrement dit les deux indicateurs nommés « vorticité » et « divergence »
sont deux composantes du tenseur des déformations.

## Mesuré sur des champs à réponse analytique

`tests/test_analytic_fields.py`, champs linéaires, exact à 1e-12 sur
l'intérieur du domaine (le raccord périodique fausse une cellule par bord) :

| champ | ω_z attendu | ω_z mappeurs | ∇·v attendu | ∇·v mappeurs |
|---|---|---|---|---|
| rotation solide `vx=−y, vy=x` | +2 | **0** | 0 | 0 |
| cisaillement pur `vx=y, vy=0` | −1 | **0** | 0 | **+1** |
| expansion pure `vx=x, vy=y` | 0 | 0 | +2 | **0** |
| déformation pure `vx=x, vy=−y` | 0 | **−2** | 0 | 0 |

L'indicateur de vorticité est **exactement nul sur une rotation solide** et
vaut −2 sur un champ de vorticité nulle. L'indicateur de divergence est
exactement nul sur une compression isotrope et vaut +1 sur un champ de
divergence nulle.

## Deux défauts de plus, dans le critère Q — celui-là sur les bons axes

`grid._compute_q_criterion` utilise `AXIS_X`/`AXIS_Y` correctement, mais
pondère la déformation de moitié : `strain_sq = S₁₁² + S₂₂² + 2·S₁₂²` vaut
(S_n² + S_s²)/2, alors que la forme d'Okubo-Weiss demande S_n² + S_s².
Conséquences, exactes elles aussi :

| champ | Okubo-Weiss standard | `_compute_q_criterion` |
|---|---|---|
| cisaillement pur | 0 (frontière rotation/déformation) | **+0.25** → lu « dominé par la rotation » |
| expansion pure | 0 (ni rotation ni déformation déviatorique) | **−1** → lu « dominé par la déformation » |

Le second vient de ce que `S₁₁² + S₂₂²` retient la partie isotrope du
tenseur. Les deux sont épinglés par un test dédié.

## Ce que la correction changerait : la variante `--fixed-curl`

Le chemin par défaut n'est pas touché. `fixed_curl=False` est le défaut des
trois mappeurs et de `prepare_qaoa_inputs` ; sa sortie est **bit-à-bit** celle
d'avant, vérifié sur 64 tableaux (score classique, score physique,
coefficients V1 et V2) × 4 scénarios à N=64. Une seule association
arithmétique a dû être conservée telle quelle : réécrire
`vx − roll(vx) + roll(vy) − vy` sous une forme algébriquement identique
déplaçait le dernier bit (écart 8.0e-15 sur `K_plaquettes`, mhd_rotor).

`--fixed-curl` applique la convention déclarée et suffixe ses artefacts
`_fixedcurl`, donc les deux variantes ne peuvent pas s'écraser.

## Le résultat, avec ses intervalles

```
python study/h1_solver/h1_curl_convention_gap.py --N 128 --dim 8  --n-snaps 6 --seed 0
python study/h1_solver/h1_curl_convention_gap.py --N 128 --dim 16 --n-snaps 6 --seed 0
```
git `95571d1` (table refaite, D-69 ; publiée d'abord à `8ee5c8a`) — 4
scénarios × 6 instantanés = 24 lignes, IC95 rééchantillonné **par
scénario** (le bloc est la trajectoire, pas l'instantané).

La décision au seuil entraîné est inexploitable : à 0.1496 le score de patch
sature et les deux bras dégénèrent en « tout raffiner » (9/24 lignes
dégénérées à dim=8). La comparaison porte donc sur le **classement**, à
budget apparié — les deux bras raffinent le même nombre de patches et ne
diffèrent que par lesquels.

**La table en vigueur est celle refaite à `95571d1` (D-69).** Les
artefacts `results/h1_curl_convention_gap_N128_dim{8,16}_v2.npz` du dépôt
sont ceux-là, et `tests/study/test_curl_convention_gap.py` échoue si l'un
des deux s'en écarte de plus de 5e−4.

| dim | métrique | historique | corrigé | Δ | IC95 | verdict |
|---|---|---|---|---|---|---|
| 8 | Spearman vs dureté | +0.7426 | +0.7160 | −0.0266 | [−0.1096, +0.0233] | indécidable |
| 8 | F1 budget apparié | +0.7214 | +0.6901 | −0.0312 | [−0.1146, +0.0156] | indécidable |
| 16 | Spearman vs dureté | +0.7960 | +0.7427 | −0.0534 | [−0.1673, +0.0343] | indécidable |
| 16 | F1 budget apparié | +0.8730 | +0.8132 | −0.0599 | [−0.1719, +0.0052] | indécidable |

Écart maximal sur le score : 0.336 (dim 8), 0.397 (dim 16) ; accord des
décisions 0.921 (dim 8) et 0.927 (dim 16). La convention change donc
réellement les entrées. Sur le **sens**, la table ne tranche pas : les
quatre Δ sont négatifs, aucun des quatre intervalles n'exclut zéro.

**Ce que cette table remplace, et pourquoi — D-69, clos.** La table
publiée à `8ee5c8a` portait le seul verdict tranché du module :

| dim | métrique | `8ee5c8a` (publié) | `47012fa` | `dffac18` = `95571d1` (en vigueur) |
|---|---|---|---|---|
| 8 | Spearman | −0.0029 [−0.0222, +0.0164] | −0.0122 [−0.0276, +0.0026] | −0.0266 [−0.1096, +0.0233] |
| 8 | F1 apparié | **+0.0391** [−0.0156, +0.0938] | +0.0182 [−0.0182, +0.0573] | **−0.0312** [−0.1146, +0.0156] |
| 16 | Spearman | −0.0665 [**−0.1328, −0.0146**] | −0.0495 [−0.1342, +0.0362] | −0.0534 [−0.1673, +0.0343] |
| 16 | F1 apparié | −0.0299 [−0.0651, +0.0052] | −0.0286 [−0.0664, +0.0052] | −0.0599 [−0.1719, +0.0052] |

Deux déplacements distincts, chacun attribué à sa cause par la mesure,
tous trois rejoués **dans le même environnement** pour que la comparaison
porte sur le code et pas sur les bibliothèques :

1. `8ee5c8a` → `47012fa` : le **solveur** a changé sous les quatre
   scénarios canoniques (D-25, projection de B par défaut désactivée ;
   D-26/D-27, `harris_tearing` réamorcé à 100 % de son amplitude prévue
   au lieu de 27,5 %). C'est ce que D-69 avait établi.
2. `47012fa` → `95571d1` : **D-70 seul.** Rejoué au commit `dffac18`
   — la correction de `_hard_patches`, rien d'autre — les quatre lignes
   sortent identiques au dernier chiffre à celles de `95571d1`. Aucun des
   commits suivants (dont D-91) ne déplace cette table.

Le second déplacement est le plus lourd et il n'était pas mesuré : la
vérité terrain de dureté que Spearman et le F1 comparent au score n'était
pas celle que sa docstring annonçait (D-70). Le Δ du F1 à dim=8 **change
de signe** avec elle (+0.0391 publié → −0.0312), et les intervalles
s'élargissent d'un facteur ~3 à dim=8 : la définition canonique est plus
hétérogène d'un scénario à l'autre que l'écart-type intra-patch qu'elle
remplace, et le bootstrap par scénario le voit.

**L'environnement n'est pas en cause, mesuré.** Rejouées au hash
`47012fa` dans l'environnement de cette passe, les quatre lignes rendent
`−0.0122 [−0.0276, +0.0026]`, `+0.0182 [−0.0182, +0.0573]`,
`−0.0495 [−0.1342, +0.0362]`, `−0.0286 [−0.0664, +0.0052]` — la colonne
`47012fa` ci-dessus au dernier chiffre publié. La réserve de dérive
d'environnement que D-69 formulait ne porte donc pas sur ces nombres.

**Mesure déterministe.** Les deux commandes rejouées à l'identique rendent
une sortie bit-à-bit identique (deux exécutions par dim). L'écart au
publié n'est pas de la variance d'exécution.

**Aucun compteur du master table ne bouge** : T31 n'y figure pas.
`python study/common/aggregate_master_table.py` rend **180 / 176 OK /
4 DIFF / 0 MISSING** avant comme après.

## Ce qu'il faut en conclure

⚠️ **Rétractation — « à dim=16 la correction dégrade » ne tient plus.**
Cette section a longtemps conclu : *« corriger la convention d'axes
n'améliore pas la tâche, et à dim=16 la dégrade avec un intervalle qui
exclut zéro »*. La seconde moitié est retirée. Refaite à `95571d1`
(D-69), la table ne porte **aucun** verdict tranché : l'IC95 du Spearman
à dim=16 passe de [−0.1328, **−0.0146**] à [−0.1673, **+0.0343**]. Le
nombre qui portait la phrase n'existe plus.

**Ce qui reste vrai : corriger la convention d'axes n'améliore pas la
tâche.** Les quatre Δ sont négatifs aux deux dimensions — direction
constante, jamais significative. C'est une indication, pas un verdict, et
la formulation publiable est *« aucun gain mesurable, et rien qui exclue
une dégradation »*. L'explication tient en une
phrase : les hyperparamètres (`beta_curl`, `kappa`, `gamma_*`,
`threshold_amr`) ont été réglés par Optuna **sur l'opérateur historique**.
Appliquer le bon opérateur avec des coefficients calibrés pour un autre
revient à changer la grandeur mesurée sans retoucher l'instrument.

**Ce que la mesure n'établit PAS.** Elle ne dit pas que corriger *puis*
réoptimiser serait inutile. La comparaison est confondue par construction :
le bras corrigé tourne avec des coefficients calibrés pour l'autre opérateur,
donc il est désavantagé exprès. On ne peut pas conclure « la correction ne
sert à rien » d'une expérience où le bras corrigé part handicapé. C'est
précisément la question qu'un référé posera — *« Q-HAS a-t-il échoué parce
que son entrée physique était cassée ? »* — et « on n'a pas réoptimisé »
n'est pas une réponse.

Trois lectures, révisées après avoir compté le coût réel :

1. *Corriger et publier tel quel* — écarté, mais plus faiblement qu'écrit
   ici d'abord (« la mesure dit que c'est pire ») : depuis D-69 la mesure
   ne dit plus que c'est pire, elle dit qu'elle ne sait pas. Ce qui écarte
   cette lecture est qu'aucune des quatre comparaisons ne montre de gain.
2. *Documenter et ne rien réoptimiser* — recommandation initiale, fondée
   sur « une semaine de calcul Optuna ». J'ai d'abord cru la réfuter avec
   les ~47 h de **mur** mesurées dans les bases. C'était un mauvais cadrage :
   les essais tournaient jusqu'à 9 de front, soit **224 h de CPU = 9.3 jours
   mono-cœur**. L'annonce d'origine était juste en temps processeur, et
   c'est le temps processeur qui gouverne le coût d'une relance.
3. *Réoptimisation ciblée* — retenu, mais chiffré honnêtement. Seuls
   `beta_curl`, `kappa` et `threshold_amr` touchent le canal du rotationnel.
   Un essai du bras quantique coûte **56 min de CPU** (médiane sur 178
   essais). Donc :

   | budget | CPU | mur sur 4 cœurs | mur sur 32 cœurs |
   |---|---|---|---|
   | 30 essais | 28 h | ~7 h | ~1 h |
   | 60 essais | 56 h | ~14 h | ~2 h |
   | 100 essais | 93 h | ~23 h | ~3 h |

   C'est **une nuit sur une machine ordinaire**, pas « quelques heures ».
   Le nombre d'essais nécessaire en dimension 3 est une **hypothèse non
   mesurée** — c'est la partie molle de l'estimation.

Tant que (3) n'est pas fait, l'énoncé publiable est le fait mesuré — les
indicateurs sont mal nommés, et les corriger *à hyperparamètres inchangés*
ne restaure pas de performance — et **pas** la conclusion plus forte que la
convention serait sans importance.

Le manuscrit doit donc dire que les indicateurs nommés « vorticité » et
« divergence » de V1 sont en réalité deux composantes du tenseur des
déformations, que ce fait est mesuré et non supposé, et que le corriger sans
réoptimiser ne restaure pas de performance. La bonne question pour la suite
n'est pas « la vorticité est-elle juste ? » mais **« le critère a-t-il jamais
eu besoin de la vorticité ? »** — le canal courant de `K_plaquettes` donnait
déjà r = +0.000 avec la vraie densité de courant.

## Tests

| fichier | tests | ce qu'ils verrouillent |
|---|---|---|
| `tests/test_analytic_fields.py` | 36 | les cinq grandeurs nommées contre des champs à réponse connue ; l'invariance bit-à-bit du chemin par défaut |
| `tests/study/test_fixed_curl_variant.py` | 7 | le drapeau change vraiment quelque chose, atteint θ à travers l'encodeur ψ, et suffixe son artefact |
| `tests/study/test_curl_convention_gap.py` | 14 | budget apparié, Spearman, bootstrap par scénario, verdict sans IC interdit |

Les trois mutations essayées sur `tests/test_analytic_fields.py` (axes
échangés dans `forward_curl_z`, `curl_z` ignorant son drapeau, `fixed_curl`
passé à `True` par défaut) sont toutes détectées.

---

# Le bras QAOA est échantillonné, et de combien

## Le fait

`src/VQA/execute.py` construit sa distribution finale par
`final_distribution = counts / total_shots` à partir de `sampler.run(...)`.
**Aucune graine n'est fixée dans tout `src/VQA/`** — ni `seed_simulator`, ni
`np.random.seed`, ni graine passée au sampler. Deux appels sur le même état
et les mêmes hyperparamètres ne donnent donc pas le même résultat.

## La mesure

Dix appels **strictement identiques** à `qaoa_block_scores` (mhd_rotor,
Re=800, N=64, 3×3 blocs, `w_z_frac`=0.10, `threshold`=0.3), soit 45 paires :

| grandeur | min | médiane | max |
|---|---|---|---|
| dispersion des scores de bloc (ptp par appel) | 1.79e-1 | — | **3.61e-1** |
| auto-corrélation de rang | **0.350** | **0.933** | 1.000 |
| appels dégénérés (score constant sur les 9 blocs) | — | **0 / 10** | — |

Un premier sondage à 6 appels (15 paires) donnait un minimum de 0.550 et une
médiane de 0.883 : la queue descend plus bas que 15 paires ne le laissaient
voir. C'est l'échantillon à 45 paires qui fait foi, et c'est la raison pour
laquelle les seuils des tests portent sur la **médiane**, jamais sur le
minimum.

Deux lectures, opposées, et toutes deux importantes :

1. **Les valeurs bougent beaucoup.** Plus d'un cinquième de l'échelle [0,1]
   entre deux exécutions identiques, au pire.
2. **Le classement, lui, tient.** Auto-corrélation de rang médiane 0.933 :
   le bras ordonne les blocs de façon reproductible, même si les valeurs
   qu'il leur attribue ne le sont pas. Les conclusions de cette étude qui
   reposent sur un **ordre** (budget apparié, top-k) sont donc robustes ;
   celles qui reposeraient sur une **valeur** ne le seraient pas.

## Ce que cela a cassé

`check_sweep_behaviour` exigeait `abs(delta) <= 1e-9` pour au moins une des
douze combinaisons d'hyperparamètres — c'est-à-dire que QAOA sélectionne
*exactement* les mêmes 2 blocs sur 9 que le classique. C'est huit ordres de
grandeur sous le bruit du bras. L'assertion avait été calibrée sur une
exécution unique, dans le commit `32d124a` qui prétendait précisément donner
des critères d'acceptation aux étapes qui n'en avaient pas. **Septième
assertion à tirage unique sur ce bras.**

Le plafond est conservé — aucun réglage ne doit faire passer QAOA devant le
classique sur données propres, et cela peut échouer. Deux critères manquants
sont ajoutés : ρ doit rester positif (sinon le bras ne classe plus rien et
le plafond ne prouve rien) et ρ doit **varier** entre combinaisons (sinon
les hyperparamètres n'atteignent pas le bras). L'égalité exacte est
désormais rapportée, pas assertée.

## Deux erreurs de méthode commises en écrivant ce test

Consignées parce qu'elles sont du genre même que l'étude traque.

1. **Un seuil posé sans mesure.** `test_the_ranking_survives_the_sampling`
   assertait `min(rhos) > 0.5` sur trois paires. Le minimum sur trois tirages
   est la statistique la plus instable disponible, et 0.5 était une
   intuition. La mesure donne min 0.550 : le seuil tombait dans la queue de
   la distribution. Corrigé en médiane sur 10 paires, seuil à 0.6.
2. **Un chiffre publié sous-estimé, deux fois.** La dispersion annoncée
   d'abord (9.58e-2, 8.70e-2) venait de trois appels ; puis 1.50e-1 / 2.15e-1
   de quinze paires ; la mesure à 45 paires donne un ptp par appel allant
   jusqu'à 3.61e-1. Chaque valeur était exacte pour son tirage, et chacune
   sous-estimait la suivante. Un échantillon trop petit ne se signale pas
   comme tel.
3. **Un diagnostic construit sur une prémisse non vérifiée.** Deux tests de
   rang échouant simultanément, j'en ai conclu — `médiane > 0.6` et
   `min < 1.0` étant complémentaires — qu'il y avait des NaN, donc des
   appels rendant un score constant. La sonde le réfute : 0 appel dégénéré
   sur 10. La déduction était valide, sa prémisse implicite (que c'étaient
   des échecs d'assertion) ne l'était pas : les deux tests tombaient sur un
   `TypeError`, `spearmanr` recevant des tableaux (3, 3) non aplatis et
   renvoyant une matrice. Un garde de forme l'empêche désormais.

## Défaut annexe

`_SWEEP_ROWS` et `_RESOLUTION_ROWS` sont des listes au niveau module qui
n'étaient jamais vidées : deux exécutions dans le même processus donnaient
24 lignes et faisaient échouer `len(rows) == 12` pour une raison étrangère à
ce que le test mesure.

## Tests

| fichier | ce qu'il verrouille |
|---|---|
| `tests/test_qaoa_arm_is_sampled.py` | le bras varie ; la dispersion écrase la tolérance ; le classement tient (médiane) ; il bouge quand même (au moins une paire) ; aucune graine dans `src/VQA/` ; l'assertion d'égalité exacte n'est pas réintroduite |

Si une graine est fixée un jour dans `src/VQA/`, le premier de ces tests
tombe : c'est voulu. Il faudra alors rétablir les assertions exactes et le
consigner ici.

---

# La prolongation globale de l'AMR est fausse — D-2

## Deux écarts de convention qui se composent

`MHDSolver._upsample_global` prolonge le champ grossier vers la grille fine
dans le chemin AMR (`step_layered`, correction tau). Elle cumule deux
erreurs, aucune des deux ne produisant de plantage :

1. **Convention d'échantillonnage.** Elle vise `(j+0.5)/f − 0.5`, c'est-à-dire
   le **centre** des cellules, alors que `PeriodicGrid` place ses points aux
   **nœuds** (`linspace(0, L, N, endpoint=False)`). D'où un décalage constant
   de **−0.375 cellule grossière** à facteur 4.
2. **Mode d'enroulement.** Elle passe `mode='wrap'` à `map_coordinates`.
   Depuis scipy 1.6, ce mode n'est **pas** l'enroulement périodique : c'est
   `'grid-wrap'` qui l'est. `'wrap'` traite le tableau comme si le premier et
   le dernier échantillon coïncidaient.

Le docstring annonce « respecte la topologie torique du domaine ».

## Mesure

`sin(x)·cos(y)`, 32 → 128, scipy 1.17.1 :

| convention | mode | erreur max |
|---|---|---|
| centre (code actuel) | `wrap` | **2.49e−1** |
| centre | `grid-wrap` | 7.35e−2 |
| nœud | `wrap` | 1.79e−1 |
| **nœud** | **`grid-wrap`** | **7.74e−6** |

**Quatre ordres de grandeur** séparent le code de ce que la même
interpolation cubique atteint avec les bonnes conventions.

## Ce que ça coûte au chemin AMR — et où

Orszag-Tang, N=64, 15 pas, `max_depth=2` (donc `cf=4`), écart entre le code
et la variante corrigée :

| couverture par les patchs | écart relatif |
|---|---|
| totale | **2.7e−15** (arrondi) |
| un quart du domaine | **1.67 %** |
| aucun patch | **1.79 %** |

**Sous un patch actif, l'erreur s'annule exactement.** C'est le principe de
la correction tau : la phase 1 ajoute le delta grossier prolongé, la phase 2
le retranche et lui substitue le delta fin — la prolongation disparaît de la
différence. Elle ne survit que sur le **fond non raffiné**.

Vérifié aussi : à `max_depth=0`, `step_layered` est **bit-à-bit identique** à
`step_full` (écart exactement 0.0). La garantie annoncée ligne 561 tient.

## Pourquoi cela compte pour le manuscrit

`src/pipeline.py:480,485` : les **deux** bras — `sim_quantum` et
`sim_classical` — avancent par `step_layered`. L'erreur leur est donc
**commune** et ne biaise pas leur comparaison mutuelle. Mais le témoin
avance par `step_full` (ligne 478), qui n'a pas le défaut.

Conséquence : l'erreur de champ de chaque bras **contre la référence** porte
une composante systématique qui ne doit rien au critère de raffinement. Elle
comprime la plage dans laquelle un meilleur critère pourrait se distinguer,
et pousse les deux bras vers le bas ensemble.

**Cela pèse directement sur la décision de réoptimiser.** Réoptimiser les
hyperparamètres contre une métrique qui porte un plancher de ~1.7 %
d'origine numérique, c'est optimiser en partie contre ce défaut. La
correction de `_upsample_global` devrait donc précéder toute réoptimisation,
pas la suivre.

## Décision requise

Corriger `_upsample_global` change **tout nombre publié passant par
`step_layered`** — soit la campagne de boucle fermée entière. Ce n'est pas
une correction « au passage » : elle demande une décision explicite et une
re-exécution. Le défaut est donc mesuré, testé et consigné ici, mais
`src/` n'est **pas** modifié.

## Tests

`tests/test_amr_resampling_analytic.py`, 30 tests :

| ce qui est verrouillé | |
|---|---|
| restriction | moyenne préservée exactement, moyenne de bloc cellule par cellule, dilution d'un pic (contraste voulu avec le max-abs de `RescaleArrays`) |
| prolongation | champ uniforme préservé, identité à facteur 1, absence de couture, **l'écart aux conventions correctes** et le **biais de décalage** |
| champs physiques | toutes les clés à la bonne taille, moyenne préservée, halo, enroulement torique |
| détection de bord | chaque bord reconnu pour lui-même, anomalie intérieure ignorée, patch uniformément actif silencieux |
| impact AMR | annulation sous patch, survie sur le fond, `max_depth=0` identique à `step_full` |

---

# Audit de contrat : quatre défauts trouvés en demandant à chaque
# fonction ce qu'elle prétend faire

Les tests analytiques déjà en place vérifient des **valeurs**. Cette passe
vérifie des **contrats** : pour chaque fonction du chemin de décision, que
promet sa docstring, consomme-t-elle les entrées que sa signature annonce,
rend-elle la forme et le domaine promis, et deux chemins censés coïncider
coïncident-ils encore ?

Les quatre défauts ci-dessous étaient invisibles aux tests de valeur, parce
qu'ils partagent tous la même forme : un calcul qui rend une valeur
parfaitement plausible, indiscernable d'une valeur juste.

Commande de recette :

```bash
python -m pytest tests/mapping/test_mapper_contracts.py \
                 tests/quantum/test_hamiltonian_contracts.py \
                 tests/mapping/test_downsampling_contracts.py -q
```

`107c1cf` + cette passe. 59 + 29 + 28 = **116 tests** à l'écriture ;
**117** mesurés à `7f58735` — un test ajouté depuis, le compte de tête
n'a pas suivi. Les chemins, eux, sont corrigés : ils pointaient sur
`tests/` à plat, d'avant `17d983d` (D-142).

## D-11 — la diode de choc s'appliquait au cisaillement

`src/Simulation/PhysToAngle.py` — `compute_stress_flux`.

`_compute_filtered_flux` lit `array[0]` comme la composante **normale**
(diode `max(0, −Δ)`, poids `w_compress = 2`) et `array[1]` comme la
**tangentielle** (`abs`, poids `w_shear = 1`). L'ordre des tuples était
écrit sous la convention inverse du dépôt (axis=1 lu comme x), si bien que
la composante transverse arrivait dans la case de la normale.

| mesure sur champ analytique | code | conception |
|---|---|---|
| rapport compression / cisaillement | **0.500** | 2.0 |
| compression distinguée de l'expansion | **non** — flux identique | oui |

La diode était donc **inerte** : sa seule raison d'être est de séparer la
compression de l'expansion, et elle ne le faisait pas. Le signe d'une
différence tangentielle ne porte aucune information de compression ; la vraie
différence normale, elle, passait par `np.abs` dans la branche de
cisaillement.

Écart relatif sur Φ, snapshots DNS réels à N=256, Re=400 :

| scénario | Φ_h | Φ_v | médiane \|Δψ\| | max \|Δψ\| |
|---|---|---|---|---|
| orszag_tang | 36.6 % | 43.4 % | 0.259 | 2.98 |
| kelvin_helmholtz | 96.5 % | 51.0 % | 0.047 | 1.33 |
| mhd_rotor | 46.2 % | 37.1 % | 0.023 | 3.14 |
| harris_tearing | 93.3 % | 43.9 % | 0.008 | 2.90 |

Φ n'alimente pas θ (le score classique s'en charge) : il alimente **ψ**. Le
rayon d'action du défaut est donc exactement ψ — la quantité dont l'ablation
est au programme. **Toute lecture de l'ablation ψ antérieure à cette
correction porte sur un ψ construit sur un flux faux.**

Corrigé derrière `AngleMapper(fixed_flux=True)`, par défaut **True**, même
traitement que `fixed_curl`. `fixed_flux=False` reproduit le chemin
historique bit à bit.

## D-12 — `PhysicalMapperV2` est aveugle à trois des quatre grandeurs
## que sa docstring nomme

**Rectification de portée.** J'avais écrit « le mappeur déployé ». C'est
faux : `src/pipeline.py` n'importe **jamais** `HamiltParams_v2`. La boucle
fermée — celle qui produit les résultats de niveau 3 et la frontière de
Pareto — instancie `PhysicalMapper` (v1) avec ses hyperparamètres entraînés
(σ, β_curl, γ_hydro, γ_mag, κ, w_z_frac). Le v2 n'est utilisé que par les
scripts de `study/`.

Ce qui suit vaut donc pour les **analyses de `study/`**, pas pour la boucle
déployée.

`src/Simulation/HamiltParams_v2.py` — `PhysicalMapperV2`.

La docstring annonçait « Only physical constants (nu, eta, dx) and the
refinement threshold (thr_amr) affect the output ». Mesuré :

| grandeur | effet mesuré |
|---|---|
| `dx` : 1.0 contre 0.001 | `C`, `K`, `H` **bit à bit identiques** ; `K_xpoint` à 3.8e−11 |
| amplitude des champs ×10 | identique à 1.6e−10 |
| `nu`, `eta` | **absents du fichier** — aucun nombre de Reynolds n'entre |
| `thr_amr` | seule grandeur nommée qui agit |

Le v2 est **adimensionnel** : chaque terme est divisé par une norme prise sur
le même champ. `det(∇B) ∝ 1/dx²` est divisé par `max|det| ∝ 1/dx²` ; `dx` se
simplifie exactement.

Ce n'est pas un bug de calcul, mais cela change la lecture des analyses
**qui utilisent le v2**. **H4 (transfert)** : le v2 ne peut pas distinguer un
écoulement visqueux d'un écoulement inertiel, donc un transfert entre nombres
de Reynolds est trivialement satisfait par ses coefficients — toute
dépendance en Re ne peut venir que du score externe. **H3
(représentation)** : le v2 ne voit que la *forme relative* des champs, jamais
leur échelle.

La boucle fermée, elle, tourne sur le v1, où ν, η et dx entrent bel et bien
(via `Re_h = v_jump·dx/ν`, `RE_CRIT`, `v_jump_crit`). Les deux mappeurs ne
sont donc pas interchangeables pour lire une hypothèse : il faut dire lequel
a produit le nombre.

Aucun code modifié : la docstring a été réécrite pour dire ce que le code
fait. Deux autres mensonges de documentation corrigés dans le même fichier :
l'argument `sim`, annoncé comme fournissant les opérateurs de gradient, est
inutilisé (le v2 les réimplémente en ligne) ; et le commentaire du biais Z
portait encore le signe négatif que l'en-tête du module avait déjà corrigé.

## D-13 — les bords gauche et haut de l'Hamiltonien lisaient l'arête
## intérieure

`src/VQA/cost_hamiltonian.py` — `create_bounded_hamiltonian`.

`C_edges[0][a, b]` couple la cellule `(a, b)` à `(a, b+1)` : c'est la
convention des deux mappeurs, qui forment leurs sauts par
`champ − np.roll(champ, −1, axis=1)`. L'arête reliant le halo de gauche à la
première colonne du cœur est donc `C_edges[0][ci, 0]`.

Le code lisait `[ci, 1]` — l'arête **intérieure** (0)-(1), déjà consommée
quelques lignes plus haut comme couplage de cœur. Même chose en haut avec
`C_edges[1][1, cj]` au lieu de `[0, cj]`. Le bon coefficient existe pourtant :
les paramètres sont calculés sur un patch `(dim+2, dim+2)` qui contient le
halo.

Les bords **droit** et **bas**, eux, lisaient la bonne case. L'Hamiltonien
était donc **asymétrique entre gauche et droite sur un patch symétrique** —
c'est le test qui attrape le défaut sans connaître les indices.

Écart sur un patch réel d'Orszag-Tang (N=256, dim=4, cœur de la nappe de
courant) :

| bord | lu | correct | écart |
|---|---|---|---|
| gauche, ci=1 | −1.0243 | −0.9572 | 0.0671 (7.0 %) |
| gauche, ci=4 | −1.0251 | −0.9809 | 0.0442 (4.5 %) |
| haut, cj=1 | −1.1678 | −1.1476 | 0.0202 (1.8 %) |
| haut, cj=4 | −1.3117 | −1.2965 | 0.0152 (1.2 %) |

Corrigé directement : contrairement au rotationnel, il n'existe pas de
lecture défendable où `[ci, 1]` serait l'arête du halo.

**Garde ajouté au passage.** Toutes les lectures sont indexées par `dim` sur
des tableaux supposés `(dim+2, dim+2)`. Un tableau **trop grand** ne
déclenchait aucune erreur : la boucle lisait un sous-bloc du coin supérieur
gauche et rendait un Hamiltonien valide, calculé sur la mauvaise portion du
patch. `create_bounded_hamiltonian` refuse désormais toute forme différente
et nomme chaque tableau fautif.

## D-14 — le score et les champs ne décrivaient pas la même région

`src/Simulation/refinement.py` — `_downsample_fields`.

Un patch descend vers le VQA par deux chemins indépendants : les champs par
**mean-pool** (`_downsample_fields`), le score par **max-pool**
(`RescaleArrays._process_score`). Le max-pool couvre 100 % du patch depuis sa
correction ; le mean-pool découpait `patch[:out_dim*bh, :out_dim*bw]` et
jetait le reste de la division.

La cellule `(i, j)` du score ne désignait donc plus la cellule `(i, j)` des
champs. La perte tombe toujours du même côté — les dernières lignes et
colonnes — donc c'est un **biais, pas du bruit**. Et ces dernières lignes
sont exactement le **halo droit et bas**, c'est-à-dire l'information de
voisinage que H3 cherche à évaluer.

Le patch vaut `extent + 2·pad` (le halo de `get_periodic_patch`) et la cible
`dim + 2·pad` : la division tombe rarement juste. Couverture à N=256 :

| dim | prof. 0 | prof. 1 | prof. 2 | prof. 3 |
|---|---|---|---|---|
| **2 (déployé)** | 100 % | **98.5 %** | **97.0 %** | **94.1 %** |
| 3 | 99.6 % | 100 % | 98.5 % | 88.2 % |
| 4 | 100 % | 96.9 % | 100 % | 88.2 % |
| 8 | 100 % | 100 % | 90.9 % | 88.2 % |

Le chemin déployé était touché dès la première descente.

Corrigé par des bornes `np.linspace` couvrant toute l'étendue — la même
correction que `_maxabs_pool_2d`, ce qui remet les deux chemins d'accord.
Quand la division tombe juste, la sortie est bit à bit identique et le
chemin rapide `reshape` est conservé.

## Ce que cette passe dit des tests précédents

Les quatre défauts partagent la forme que les tests de valeur ne peuvent pas
voir : ils rendent un résultat plausible. Trois d'entre eux ont été trouvés
par la même question — *deux chemins censés coïncider coïncident-ils ?* —
qu'aucun test antérieur ne posait :

- D-11 : la diode contre sa propre docstring ;
- D-13 : le bord gauche contre le bord droit ;
- D-14 : la réduction des champs contre celle du score.

C'est la classe de test à étendre en priorité, pas le nombre d'assertions.

---

# Ce que le circuit peut déplacer, et par quel canal

Ce n'est pas un défaut : c'est une propriété structurelle du circuit
déployé, mesurée parce que l'audit de contrat posait la question « à quoi
sert ce paramètre ? ». Elle borne par le haut ce que H0b et H2 peuvent
espérer mesurer.

La couche de coût `exp(−iγH)` est **diagonale** : elle n'ajoute que des
phases et ne peut changer aucune probabilité de mesure. Seul le mixeur
`exp(−iβ ΣXᵢ)` déplace `P(|1⟩)`. Mesuré sur les coefficients v2 réels
(Orszag-Tang N=256, seuil 0.1496, `reps=2`), en balayant γ de 0 à 2π :

| canal | déplacement max d'une probabilité |
|---|---|
| γ seul, β = 0 | **4.4e−16** — rien, aux erreurs d'arrondi près |

Et β est borné par construction à `π/(4·reps) = 0.393 rad`
(`execute.py:112`, pour empêcher COBYLA de partir à β=1 et d'écraser tout
le raffinement).

**Conséquence : tout ce que l'Hamiltonien apporte à la décision passe par
son interaction avec le mixeur.** En balayant toute la grille admissible —
donc ce qu'un optimiseur *parfait* atteindrait, pas ce que COBYLA trouve :

| patch | mixeur seul | mixeur + H | apport de H |
|---|---|---|---|
| (100,100) | 0.3776 | 0.5359 | 0.1583 |
| (40,180) | 0.2667 | 0.4267 | 0.1600 |
| (200,60) | 0.2541 | 0.4897 | 0.2357 |
| (128,128) | 0.1028 | 0.4400 | 0.3372 |
| (10,10) | 0.2238 | 0.5052 | 0.2814 |
| **médiane** | **0.254** | **0.490** | **0.236** |

Lecture : sur un patch typique, un optimiseur parfait peut déplacer une
marginale de 0.49 au plus ; environ la moitié de ce déplacement (0.254) est
une simple rotation de mixeur, indépendante de toute physique. L'Hamiltonien
n'est donc **pas** inerte — il apporte 0.236 de médiane — mais il ne peut
agir qu'à travers un canal borné à 0.393 rad, et le témoin correct pour
mesurer son apport est **le mixeur seul**, pas le score classique.

Aucune campagne du dépôt n'utilise ce témoin. C'est le contrôle qui manque à
H0b : « le QAOA déplace-t-il la décision ? » ne distingue pas « le mixeur la
déplace » de « la physique la déplace ».

Vérifié au passage, et fixé par un test parce que c'est un contrat
inter-bibliothèque qui peut casser sur une mise à jour sans erreur :
`QAOAAnsatz` ordonne ses paramètres `[β…, γ…]`, ce qui est bien l'ordre que
`execute` suppose en construisant `x0 = [zeros(reps), rampe_γ]`. Un
réordonnancement appliquerait la rampe au mixeur et la borne β au terme de
coût, en silence.

## Tests

`tests/test_vqa_chain_contracts.py`, 41 tests. La trace de bout en bout —
score (i,j) → qubit k → terme de Pauli → caractère de la chaîne Qiskit →
marginale — est posée en forçant un qubit connu et en vérifiant qu'il
ressort à sa place. Une seule convention retournée rendrait la carte de
décision **spatialement miroir** : même taille, mêmes valeurs, même fraction
raffinée, indiscernable d'une carte juste par tout test de valeur.

Conforme : ordre des bits petit-boutiste (le commentaire du code annonçait
l'inverse ; c'est le code qui a raison), `P(|1⟩) = sin²(θ/2) = score` à
2.4e−15, ψ déplace la phase et jamais la probabilité, l'aplatissement ligne
par ligne du circuit coïncide avec `idx_H(i,j) = i·dim + j` de
l'Hamiltonien, et `params = 0` reproduit exactement θ-init (la porte de
sortie du raccourci « Hamiltonien nul »).

---

# D-16 — la liste de patchs se recouvrait elle-même

`src/Simulation/refinement.py` — `_run_level` et `_run_level_classical`.

L'invariant le plus élémentaire d'un AMR — chaque cellule appartient à
exactement une feuille — n'était vérifié nulle part. Une liste qui recouvre
deux fois la même région reste parfaitement plausible : bornes valides,
profondeurs cohérentes, scores dans [0, 1]. Seule une somme de couverture la
distingue d'une liste juste.

Le **sondage de bord** (« l'anomalie touche le bord dans cette direction, on
descend même si le signal est marginal ») était un bloc **séparé**, exécuté
après la ventilation. Quand il se déclenchait, le sous-patch avait déjà été
enregistré comme feuille non raffinée par la branche `else`, et il était en
plus poussé au niveau suivant. La même région était comptée **deux fois** :
une fois comme feuille grossière, une fois redécoupée.

Mesure sur les quatre scénarios (N=256, `dim=2`, `max_depth=3`, 6 instantanés
chacun, soit 24 configurations) :

| seuil | configurations avec recouvrement | pire cas |
|---|---|---|
| **0.1496 (déployé)** | **2 / 24** | **25.0 %** du domaine |
| 0.20 | 3 / 24 | 17.2 % |
| 0.25 | 4 / 24 | 12.5 % |
| 0.30 | 6 / 24 | 25.0 % |
| 0.40 | 9 / 24 | 28.1 % |
| 0.50 | 12 / 24 | 20.3 % |

Jusqu'à **trois patchs** sur une même cellule. Toute métrique de budget ou de
couverture lue sur la liste finale surcomptait d'autant — et le balayage de
seuils de la **frontière de Pareto** passe exactement dans la zone où le
défaut est le plus fréquent.

Les deux bras portaient le défaut à l'identique, donc leur comparaison
mutuelle n'était pas biaisée ; le coût absolu, si.

Corrigé en fondant le sondage dans le même `if/elif/else` : un sous-patch est
soit raffiné, soit feuille, jamais les deux. Après correction, sur les six
seuils de 0.1496 à 0.65 : **0/24 recouvrement, 0 % de trou** — la liste pave
exactement le domaine.

**Corrigé au passage** : le journal `verbose` affichait
`threshold_amr + (1−threshold_amr)·depth/max_depth`, une rampe en profondeur
que le code n'applique plus (`effective_threshold = threshold_amr`, la rampe
est commentée). Le journal annonçait donc un seuil, et le code en appliquait
un autre ; toute lecture des décisions dans les journaux était fausse.

## Tests

`tests/test_amr_tiling_contracts.py`, 44 tests : pavage exact sur les quatre
scénarios × trois seuils, sur tout le domaine de seuils de la frontière de
Pareto, à chaque taille de patch et chaque profondeur ; aire couverte égale à
l'aire du domaine ; aucune borne dupliquée ; monotonie du nombre de patchs en
fonction du seuil ; chaque bord reconnu pour lui-même par
`_boundary_activation`, et aucun drapeau sur un patch uniformément actif ou
uniformément calme (le faux positif généralisé qui déclenchait le doublon).
Deux tests structurels vérifient que les deux bras gardent la même forme de
sondage et que le seuil journalisé est celui appliqué.

---

# Verdict de la suite QAOA — et ce qu'il révèle

`python -m pytest tests/quantum/test_qaoa_advantage.py tests/quantum/test_qaoa_noise_and_early.py
tests/quantum/test_qaoa_scaling_and_hparams.py tests/quantum/test_qaoa_decisions.py
tests/quantum/test_qaoa_physics_decision.py tests/quantum/test_qaoa_arm_is_sampled.py
tests/quantum/QAOA_test.py -q`

**3 échecs, 27 succès, 1 h 21 min.** Les trois échecs sont des **valeurs qui
ont bougé**, pas des casses — et l'un d'eux renverse une lecture publiée.

## Le terme ZZZZ était numériquement mort sur un vortex

Deux des trois échecs sont le même fait, mesuré par deux harnais différents :
un vortex de Lamb-Oseen **gagne désormais un contraste spatial positif**, là
où les tests affirmaient qu'il n'en gagnait pas.

Ces tests ne se trompaient pas : sur le code de l'époque le contraste valait
`−0.0058 ± 0.0064`, soit du bruit de tirage légèrement négatif. Mais ce
n'était pas une propriété du QAOA.

Attribution, mesurée sur le même vortex, 16 tirages par ligne, tout le reste
égal :

| `fixed_curl` | `fixed_flux` | contraste | écart-type | σ | max\|K\| |
|---|---|---|---|---|---|
| False | False | **−0.00725** | 0.00859 | −3.4 | **0.0553** |
| False | True | −0.00852 | 0.00896 | −3.8 | 0.0553 |
| **True** | False | **+0.05672** | 0.03976 | **+5.7** | **1.2545** |
| True | True | +0.07292 | 0.04429 | +6.6 | 1.2545 |

La ligne `(False, False)` reproduit la valeur historique à l'écart-type près.

La cause est **D-1** : le rotationnel des mappeurs était écrit sous la
convention `indexing='xy'` alors que la grille construit ses champs en
`indexing='ij'`, si bien qu'une rotation solide rendait exactement 0. Le
terme ZZZZ de plaquette — **dont la seule raison d'être est de détecter une
circulation** — était donc aveugle aux vortex. Son coefficient passe de
0.055 à 1.255, **vingt-trois fois plus grand**, dès que le rotationnel voit
la rotation.

Le sens de lecture change : ce n'est pas que le QAOA ne discrimine pas un
vortex, c'est qu'on lui donnait un Hamiltonien qui ne pouvait pas en voir.

`fixed_flux` (D-11) amplifie l'effet mais ne le crée pas : il était déjà
positif à `(True, False)`.

## Ce que cela ne dit PAS

Ces deux tests utilisent le harnais **v1** — `PhysicalMapper`,
`physical_score`, θ construit à partir du flux. Le mappeur **déployé** est le
v2, qui normalise `K` par `max|ω| + max|J|`. Sur le v2, l'effet est une
**redistribution**, pas une amplification uniforme :

| champ | \|K\| médian (legacy) | \|K\| médian (corrigé) | rapport |
|---|---|---|---|
| rotation solide | **0.000000** | 0.015873 | ∞ |
| vortex Lamb-Oseen | 0.018998 | 0.000149 | 0.008× |
| DNS orszag_tang | 0.082185 | 0.180390 | 2.2× |
| DNS mhd_rotor | 0.000717 | 0.000124 | 0.17× |

Sur une rotation solide le `K` legacy est **exactement nul** dans les deux
mappeurs — le défaut est bien commun. Mais sur des champs réalistes le v2
redistribue, dans un sens ou dans l'autre selon le champ, parce que sa
normalisation par le maximum du domaine change en même temps que le
numérateur.

**Conclusion honnête** : le fait établi est que la lecture publiée « le
contraste d'un vortex est du bruit de tirage » a été mesurée sur un
Hamiltonien dont le terme de circulation était numériquement mort. Savoir si
la conclusion du chemin **déployé** bascule demande de relancer la campagne,
et ne peut pas se déduire de ces quatre lignes.

## Le troisième échec : une coïncidence prise pour un invariant

`test_noise_robustness` exigeait `frac_cl == gt_frac` à 1e−9 — « sans bruit,
le bras classique atteint la fraction capturée optimale ». Mesuré : 0.3151
contre 0.3245 sur Orszag-Tang (0.9709), et 1.0000 sur le rotor.

Ce n'était pas un invariant. `gt_frac` classe les blocs par une **erreur de
troncature** (dérivée seconde) et `frac_cl` par le **score classique** : deux
quantités différentes, dont les *k* premiers blocs se trouvaient coïncider.
La correction du rotationnel a changé le classement du score sur
Orszag-Tang, et la coïncidence est tombée.

Le vrai invariant — le bras classique ne peut pas **dépasser** l'optimum — est
déjà vérifié ligne par ligne plus bas. L'assertion borne désormais l'écart
relatif à 5 % au lieu de nier son existence.

## Trois tests V1 mis à jour au même titre

Cinq tests de `tests/test_vqa_stack_analytic.py` passaient des **comptes
bruts** à `postprocess`, dont deux figeaient explicitement les deux pièges
que D-15 ferme : rendre des zéros sur une distribution vide (une lecture
manquante devenait un patch calme), et tronquer en silence une chaîne plus
longue que le registre. Ils affirment désormais que les deux cas sont
refusés.

**V1 non-QAOA après toutes les corrections : 844 succès, 4 ignorés** (15 min
46 s), une fois ces cinq tests mis en accord avec le nouveau contrat.

---

# D-17 / D-18 — le balayage de D-1 s'était arrêté à `src/`

L'audit de contrat appliqué à `study/` et `figures/` a trouvé quatre sites
qui réimplémentaient encore leur propre opérateur sous la convention
inverse. La correction D-1 n'avait touché que `src/Simulation/`.

## Ce que l'opérateur « legacy » calcule réellement

```
correct : (roll(fy,-1,AXIS_X) - fy) - (roll(fx,-1,AXIS_Y) - fx) = ∂fy/∂x - ∂fx/∂y
legacy  : (roll(fy,-1,AXIS_Y) - fy) - (roll(fx,-1,AXIS_X) - fx) = ∂fy/∂y - ∂fx/∂x
```

Ce n'est **pas** un rotationnel de signe opposé — auquel cas `abs` ou le
carré auraient tout rattrapé. C'est son **complémentaire** : une combinaison
de déformation, nulle là où le rotationnel est maximal, maximale là où il
s'annule.

| champ | rotationnel | opérateur legacy |
|---|---|---|
| rotation solide | +0.392699 | **0.000000** |
| cisaillement pur | −0.196350 | **0.000000** |
| compression pure | 0.000000 | −0.392699 |

## D-17 — trois sites, trois quantités mal nommées

| fichier | fonction | conséquence |
|---|---|---|
| `study/h2b_prediction/h2b_v1_hamiltonian_loso.py` | `jz_from_b` | rendait 0 sur une rotation solide ; appelle désormais `forward_curl_z` |
| `study/h2b_prediction/h2b_ceiling_random_split.py` | `omega_z`, `J_z` (features ML) | deux des neuf features de H2b mesuraient la déformation |
| `figures/v1_legacy/fig_utils.py` | `compute_enstrophy` | l'enstrophie tracée n'était pas une enstrophie |

Validation analytique de `compute_enstrophy`, cisaillement pur périodique
`vx = sin y`, enstrophie exacte `2π² = 19.7392` :

| version | valeur | écart |
|---|---|---|
| corrigée | 19.7352 | 0.02 % (erreur de la différence centrée) |
| ancienne | **0.0000** | **100 %** |

**Piège de validation à retenir** : sur Taylor-Green les deux conventions
rendent la **même** intégrale, par symétrie de leurs carrés. Un test écrit
sur ce champ aurait passé sans rien vérifier. Le fichier de test le fige
explicitement.

`study/pipeline/hard_patch_labels.py` et `study/common/qaoa_inputs.py`
étaient signalés par le balayage mais **vérifiés corrects** — ils gardent des
axes numériques nus pour rester bit-à-bit identiques aux artefacts publiés,
et figurent dans une liste d'exceptions explicitement documentée.

## D-18 — rectification : la moitié `fluctuating_KE` était déjà connue

**Correction à la première rédaction de cette section.** J'avais présenté le
défaut d'axe de `dns_validation.fluctuating_KE` comme une trouvaille de cet
audit. C'est faux. Il était déjà connu, consigné comme **déviation D2**, et
la décision prise alors était explicite — `dns_extension.py:85` : « phase 1b
reste intouchée, réparation côté v3 par copie ». Un test de `tests/v3`
épinglait même la contamination
(`test_phase1b_observable_is_contaminated_by_base_flow`).

Ma correction dans `dns_validation.py` a donc **rompu ce gel**, et c'est la
grande suite qui l'a signalé en faisant échouer ce test. Le fichier a été
remis dans son état d'origine.

### Ce que l'audit apporte réellement

**1. La mesure de D2, qui n'était chiffrée nulle part.**

`fluctuating_KE` retranche une moyenne pour isoler la perturbation ; elle
doit être prise le long de la direction **homogène**.
`init_kelvin_helmholtz` construit son profil à partir de `grid.Y`, que
`meshgrid(x, y, indexing='ij')` fait varier le long de l'**axe 1** — la
direction homogène est donc l'axe 0. Le code moyenne sur l'axe 1, **à
travers la couche de cisaillement**, et ne soustrait rien.

Sur le profil de base **sans aucune perturbation**, où la réponse attendue
est zéro :

| moyenne prise sur | valeur | part de l'énergie cinétique totale |
|---|---|---|
| axe 1 (phase 1b, gelé) | 3.411e−01 | **73 %** |
| axe 0 (`fluctuating_ke_fixed`) | 1.323e−30 | 0 % |

En allumant la perturbation nominale (amplitude 0.1) :

| | base seule | avec perturbation | rapport |
|---|---|---|---|
| gelé | 0.34115 | 0.34120 | **1.0002** |
| corrigé | 1.3e−30 | 2.5e−04 | 1.9e+26 |

La grandeur gelée est à **99.98 % de l'écoulement de base**.

**2. Une seconde déviation dans le même fichier, celle-là non répertoriée.**

`mean_sq_current` porte la même inversion d'axes : `⟨J²⟩` vaut en fait
`⟨(∂By/∂y − ∂Bx/∂x)²⟩`. Aucun test ne l'épinglait. Elle est désormais
consignée comme **déviation D3**, `dns_validation.py` reste gelé au même
titre que pour D2, et une copie corrigée `mean_sq_current_fixed` a été
ajoutée à `dns_extension.py` sur le modèle de `fluctuating_ke_fixed`.

Vérification de la copie : sur un cisaillement magnétique pur
`Bx = −sin y`, la version gelée rend **0** et la corrigée retrouve
`⟨(∇×B)²⟩` à 5 % près ; sur un champ potentiel `B = ∇φ` la corrigée rend
zéro (contrôle négatif).

### La leçon

Une déviation connue mais **non écrite là où elle vit** se fait recorriger
par erreur — c'est exactement ce qui vient d'arriver. Les deux sont
maintenant documentées dans `dns_validation.py` lui-même, et un test vérifie
que ces mentions y restent.

## Tests

`tests/study/test_no_private_curl_survives.py`, 26 tests. Ils verrouillent les
**deux côtés** de D2 et D3 : la version gelée doit rester fausse à
l'identique (sans quoi les artefacts de phase 1b cessent d'être
reproductibles), et la copie corrigée doit être juste. Un test de plus est un
**balayage** : tout rotationnel écrit à la main avec un `axis=0`/`axis=1`
nu, hors de la liste d'exceptions vérifiées, fait échouer la suite. Un
opérateur écrit à la main est indiscernable d'un opérateur juste tant qu'on
ne l'évalue pas sur une rotation solide ; exiger `AXIS_X`/`AXIS_Y` rend
l'erreur visible à la lecture.

---

# D-19 / D-20 — deux pièges dans le contexte d'exécution partagé

`src/VQA/runtime.py`. `VQARuntime` est construit une fois par run et passé à
chaque appel VQA. Les deux défauts sont de la même famille : **une valeur
inutilisable qui se laisse produire sans bruit**.

## D-19 — un backend inconnu construisait un objet mort

`_init_backend` n'avait pas de branche `else`. Un `backend_name` inconnu
laissait `_backend`, `_estimator` et `_sampler` à `None`, et le constructeur
**rendait la main sans erreur**. La panne ne surgissait que bien plus loin,
dans `execute`, sous la forme d'un `AttributeError` sur `NoneType` — à des
dizaines de lignes de sa cause.

`execute.py` et `optimize.py` lèvent tous deux `ValueError("Unsupported
backend")` pour exactement la même valeur. Les trois sites disaient trois
choses différentes ; ils disent désormais la même.

## D-20 — le cache d'ansatz confondait deux Hamiltoniens

Le cache était indexé sur `(num_qubits, period_bound, reps)`. Or l'ansatz
QAOA encode `exp(−iγH)` : il dépend de l'Hamiltonien **terme par terme**,
pas seulement de la topologie.

Vérifié : deux Hamiltoniens sans aucun coefficient commun, à même nombre de
qubits et même `reps`, recevaient **le même objet**. Le second patch aurait
donc été optimisé contre la physique du premier — sans le moindre signal.

`get_ansatz` n'est appelé par aucun code du dépôt. C'était un **piège armé**,
prêt à se déclencher au premier branchement — précisément ce qu'un audit de
couverture ne voit pas et qu'un audit de contrat trouve.

La clé inclut désormais une empreinte des coefficients, arrondie à 12
décimales : un dernier bit ne fait pas exploser le cache, un écart de 1e−9
le sépare.

## Tests

`tests/test_runtime_contracts.py`, 20 tests : refus d'un backend inconnu et
message qui énumère les valides, aucun backend valide ne laisse une
primitive à `None`, deux Hamiltoniens différents ne partagent jamais un
ansatz, le même le retrouve, un seul coefficient suffit à manquer le cache,
l'empreinte est indépendante de l'ordre des termes mais sépare un
changement de signe.

---

# Audit de contrat des portes physiques — aucun défaut, deux constats

`src/Simulation/HamiltParams.py`. Cinq fonctions statiques portent tout le
raisonnement physique du mappeur v1 : `_f_gate` (Reynolds),
`_threshold_contrast` (contraste au seuil), `_g_strain` / `_g_rot`
(interrupteurs d'Okubo-Weiss) et `_g_mag` (activité magnétique). Leurs
docstrings énoncent des contrats précis — continuité, bornes, sens
d'activation.

**Les cinq honorent ce qu'elles annoncent.** Continuité de `_f_gate` au
raccord vérifiée à 1e−8 pour quatre valeurs de γ, monotonie sur cinq
décades, bornes respectées, aucun débordement sur `±1e300` ni sur `inf`.
`_threshold_contrast` rend **exactement** zéro au seuil et sous le seuil, et
garde bien un signal sur un domaine uniformément actif — la différence
revendiquée avec Michelson.

Deux constats structurels méritaient d'être écrits.

## Constat 1 — `g_strain` et `g_rot` ne sont pas deux interrupteurs

Elles somment à **1 exactement**, pour tout Q :

`1/(1+e^x) + 1/(1+e^−x) = 1`

| Q | `g_strain` | `g_rot` | somme |
|---|---|---|---|
| −10 | 1.000000 | 0.000000 | 1.0 |
| 0 | 0.500000 | 0.500000 | 1.0 |
| +10 | 0.000000 | 1.000000 | 1.0 |

Elles ne peuvent donc **jamais être actives ensemble, ni inactives
ensemble**. Le terme ZZ (porté par `g_strain`) et le terme ZZZZ (porté par
`g_rot`) sont une **partition d'un unique scalaire d'Okubo-Weiss**, pas deux
détecteurs indépendants.

Cela change la lecture d'une ablation : retirer le ZZ ne retire pas une
source d'information distincte du ZZZZ — cela déplace le poids d'un côté à
l'autre du même signal. C'est à rapprocher du résultat déjà consigné sur le
canal du circuit : l'architecture présente plus de degrés de liberté qu'elle
n'en a.

Un troisième cas de la même famille est mesuré au passage : dans la branche
hydrodynamique, `f_Re` et `mic_v` sont deux **reparamétrages monotones du
même scalaire** — `Re_h = v_jump·dx/ν` et `v_jump/v_jump_crit = Re_h/RE_CRIT`
sont égaux à 1e−12 près. Le coefficient présente deux facteurs physiques là
où il n'y a qu'une variable.

## Constat 2 — l'exemple de la docstring de `_f_gate` est inatteignable

Elle illustre la croissance logarithmique par « Re=3000, x_crit=10, γ=2 →
f ≈ 12 (not ∞) ». La formule rend bien **12.4076**, mais `f_max = 10.0` par
défaut la ramène à **10.0000** : la valeur citée ne peut jamais sortir de la
fonction telle qu'elle est appelée.

## Tests

`tests/test_gate_contracts.py`, 42 tests.

---

# D-21 — le flux descendait par un chemin qui efface ce qu'il mesure

`src/Simulation/RescaleArrays.py` — `get_adaptive_flux._process_flux`.

Trois quantités descendent du domaine plein vers la résolution du VQA : le
**score** classique, les **coefficients** d'Hamiltonien et le **flux de
contrainte** Φ. Les trois sont des indicateurs d'**anomalie** — leur raison
d'être est qu'un signal fort et isolé survive à la réduction.

Deux d'entre elles étaient max-poolées, et `_process_score` porte même un
`# No smoothing!` explicite. Le flux, lui, passait par un lissage 3×3 puis
`zoom(order=1)`, justifié par « smooth physical fields ». Or Φ n'est pas un
champ lisse : il est bâti sur des **différences** de champ et pique
exactement là où le score pique.

Un zoom bilinéaire **échantillonne**, il ne moyenne pas :

| réduction 128 → 4 d'un pic isolé | résultat |
|---|---|
| positions où le pic survit | **1 sur 256** |
| pic placé au centre | **0.0000** |
| même pic, max-pooling | 1000 |
| même pic, moyenne de bloc | 0.98 |

Le lissage préalable aggravait le tout : il diluait le pic **avant** de
l'échantillonner.

Part du pic de Φ conservée sur champs DNS réels (patch 128 → 4) :

| scénario | avant | après |
|---|---|---|
| orszag_tang | **38.0 %** | 100 % |
| mhd_rotor | **69.8 %** | 100 % |
| kelvin_helmholtz | 100 % | 100 % |
| harris_tearing | 100 % | 100 % |

Corrigé : les trois chemins appliquent désormais la même réduction. Le pic
est conservé à 100 % sur les quatre scénarios, et la carte de flux réduite
est maintenant **identique cellule par cellule** à la carte de score réduite
quand on leur donne la même entrée — ce qui n'était pas vérifiable avant.

Φ n'alimente que ψ. Comme D-11, le rayon d'action est exactement la quantité
dont l'ablation est au programme.

## Deux autres corrections dans la même passe

**`dns_validation.analyse_one`** utilise désormais les observables corrigées
`fluctuating_ke_fixed` et `mean_sq_current_fixed`. Les deux fonctions
d'origine restent en place, inchangées, pour reproduire à l'identique les
artefacts déjà publiés de phase 1b : le gel porte sur les **fonctions**, pas
sur l'analyse qui les appelle.

**`_f_gate`** — la docstring dit maintenant que `f ≈ 12.4` illustre la
formule et ne sort pas de la fonction, `f_max = 10.0` la ramenant à 10.0.

## Un test qui épinglait le défaut, retourné

`test_flux_takes_the_smoothing_path_and_loses_the_spike` affirmait « les flux
sont lissés puis interpolés, le pic doit s'y diluer ». Il décrivait
fidèlement le code ; c'est la justification qui ne tenait pas. Il vérifie
désormais l'inverse, et que le flux suit **exactement** la réduction du
score.

`tests/test_padded_rescale_contracts.py` passe de 37 à 45 tests.

---

# D-22 — les hyperparamètres déployés n'ont aucune provenance reproductible

`results/hyperparams/` est déclaré **entrée gelée** de l'étude. Il contient
deux choses qui devraient être d'accord :

- `optuna_studies/*.db` — les bases de la campagne, 345 essais
- `best_hyperparams.json` — ce que `src/hyperparams_loader.py` charge

**Elles ne le sont pas.** Vérifié directement dans les fichiers.

## Ce que les bases ont échantillonné, et ce que le JSON déploie

| étude | paramètres échantillonnés |
|---|---|
| `q_has_v2_phase1.db` (202 essais) | `beta`, `beta_curl`, `beta_xpoint`, **`sigma`**, `w_z_frac` |
| `classical_v2_phase1.db` (143) | `threshold_amr` |

| paramètre déployé | origine |
|---|---|
| `beta`, `beta_curl`, `beta_xpoint`, `w_z_frac` | échantillonnés (étude quantique) |
| `threshold_amr` | échantillonné dans l'étude **classique** seulement |
| `gamma_hydro`, `gamma_mag`, `kappa` | **aucune base ne les a jamais échantillonnés** |
| `sigma` | échantillonné, **absent du JSON** |

## Trois écarts

**1. Trois valeurs sur huit n'ont aucune origine dans le dépôt.**
`gamma_hydro = 2.127`, `gamma_mag = 2.361`, `kappa = 14.332` ne figurent
dans aucune base.

**2. `sigma` est optimisé puis jeté.** La campagne l'échantillonne et son
meilleur essai trouve **0.0230** ; le JSON ne le contient pas, donc
`pipeline.py` retombe sur `_defaults.get('sigma', 0.05)` — une constante
codée en dur. σ est la largeur de la fenêtre gaussienne, le paramètre au
cœur de D-9.

**3. L'essai déclaré ne correspond pas.** Le JSON annonce l'essai 85 avec
une perte de 0.2215. L'essai 85 existe, sa perte vaut **0.3213**, et **aucun**
de ses quatre paramètres communs ne coïncide :

| paramètre | base | JSON |
|---|---|---|
| `beta` | 6.034464 | 0.549537 |
| `beta_curl` | 1.318670 | 0.819924 |
| `beta_xpoint` | 2.341306 | 0.425647 |
| `w_z_frac` | 39.599016 | 0.101338 |

## Le code d'entraînement, lui, est cohérent

`train_hyperparams` code en dur `threshold_amr = 0.14959824837662078` avec
le commentaire « le meilleur classique ». C'est **exactement** la valeur du
meilleur essai classique (#42, perte 0.2148). Le code et les bases sont
d'accord ; **c'est le JSON qui est orphelin**.

Conséquence : le bras quantique est déployé à `threshold_amr = 0.3044`, une
valeur qui ne figure pas parmi les 125 essayées, et à laquelle il n'a jamais
été entraîné — l'objectif l'a toujours fixé à 0.1496.

## Ce que cela change pour la suite

Une réoptimisation n'est pas une amélioration : **c'est la seule façon
d'avoir des hyperparamètres qui existent.** Aucun résultat de performance ne
peut être attribué à un réglage dont on ne sait pas d'où il vient.

Corollaire pour le périmètre : `gamma_hydro`, `gamma_mag` et `kappa` n'ont
jamais été optimisés par la campagne gelée. Les inclure dans la
réoptimisation n'est donc pas une *re*-optimisation, c'est une première.

## Autres constats sur `train_hyperparams`

`make_composite_objective` présente quatre paramètres comme conditionnels
(`if "x" not in frozen:`) alors qu'ils sont des **constantes** :
`threshold_amr`, `gamma_hydro = 2.0`, `gamma_mag = 0.5`, `kappa = 10.0`.
L'espace de recherche réel est donc de cinq paramètres, pas neuf — et les
trois constantes de l'objectif ne valent pas non plus ce que le JSON
déploie.

## Tests

`tests/test_hyperparams_provenance_break.py`, 16 tests. Ils **épinglent**
l'écart plutôt que de le masquer, et chacun dit dans sa docstring ce qui
devra être vrai après réoptimisation. Le dernier,
`test_every_deployed_hyperparameter_should_one_day_be_traceable`, est le
**critère d'acceptation** : il est en `xfail` aujourd'hui et passera sans
modification le jour où chaque valeur déployée sera traçable à un essai.

---

# Correction d'une affirmation : le splitting de Strang ne s'applique pas ici

J'ai écrit à plusieurs reprises, dans `docs/RESULTS.md` et dans le plan,
qu'« un splitting de Strang rendrait l'ordre 2 ». **C'est faux, et la mesure
le montre.**

Un splitting symétrique suppose deux **flots** qu'on peut découper en
demi-pas. La projection d'incompressibilité n'en est pas un : c'est un
**projecteur idempotent**, et `P^(1/2)` n'a pas de sens.

Mesuré, N=128, Orszag-Tang, grille fixe, quatre résolutions temporelles :

| schéma | 64 pas | 128 | 256 | 512 | ordre |
|---|---|---|---|---|---|
| `P ∘ RK4` (actuel) | 8.4138e−6 | 4.0713e−6 | 1.8999e−6 | 8.1426e−7 | 1.05 → 1.22 |
| `P ∘ RK4 ∘ P` | 8.4138e−6 | 4.0713e−6 | 1.8999e−6 | 8.1426e−7 | 1.05 → 1.22 |

**Identiques à la dernière décimale.** L'explication est immédiate : après le
premier pas l'état est déjà dans le sous-espace à divergence nulle, donc la
projection initiale est l'identité. Le « Strang » que j'avais écrit *est*
le schéma de Lie.

Le bon cadre est celui d'un système **différentiel-algébrique** : l'ordre
chute parce que la contrainte est imposée *après* un pas RK4 non contraint.
Deux corrections tiennent — projeter le **second membre** à chaque étage, ce
qui rend le champ intégré à divergence nulle par construction, ou passer à
une formulation à pression.

Le plan a été corrigé en conséquence.

---

# D-24 — la contrainte imposée après le pas ramenait l'ordre 4 à 1,2

`src/Simulation/solver.py` — `_rk4_step`.

Le système est **différentiel-algébrique** : v et B doivent rester à
divergence nulle. `step_full` appliquait RK4 puis projetait l'**état** — un
splitting de Lie, d'ordre 1. En projetant le **second membre** à chaque
étage, le champ intégré est à divergence nulle *par construction* et RK4
garde son ordre.

Mesure à grille **fixe** (N=96, T=0,5, Orszag-Tang), en ne raffinant que le
pas de temps, chaque schéma comparé à sa propre référence à 1024 pas :

| schéma | 32 pas | 64 | 128 | 256 | ordre | max\|div v\| |
|---|---|---|---|---|---|---|
| projection de l'**état** | 1,098e−2 | 5,396e−3 | 2,539e−3 | 1,093e−3 | 1,03 → 1,22 | 5,04e−3 |
| projection du **second membre** | 8,610e−8 | 5,381e−9 | 3,362e−10 | **2,092e−11** | **4,00 / 4,00 / 4,01** | 5,11e−3 |
| aucune projection | 1,908e−3 | 1,234e−4 | 7,705e−6 | 4,790e−7 | 3,95 → 4,01 | **5,89e+0** |

La correction rend **les deux** : l'ordre 4 du schéma — erreur **52 000 fois
plus petite** à 256 pas — et le contrôle de la divergence au même niveau
qu'avant. Ne pas projeter du tout donne aussi l'ordre 4, mais laisse la
divergence exploser d'un facteur **1150**.

## Deux vérifications avant d'annoncer le gain

Un gain de dix ordres sur horizon court sentait le piège. Deux contrôles :

- **le champ évolue** — déplacement relatif de `vx` sur T=0,02 : 8,4813e−3
  pour les deux schémas, identiques à 6e−6 près ;
- **le second membre projeté n'est pas annulé** — il conserve **30,0 %** de
  sa norme.

La première mesure (T=0,02) était simplement au plancher de la double
précision dès 32 pas : l'ordre n'y était pas mesurable. Il a fallu allonger
l'horizon à T=0,5 pour l'obtenir.

## Ce que cela ferme

Le plan annonçait le facteur limitant du solveur comme « identifié et
corrigeable ». Il est corrigé, et la valeur de « corrigé » est chiffrée. La
mention d'un splitting de Strang est retirée : elle ne s'applique pas — un
splitting symétrique suppose deux *flots* découpables en demi-pas, alors que
la projection est un **projecteur idempotent**. Vérifié : `P ∘ RK4 ∘ P` rend
des erreurs identiques à `P ∘ RK4` à la dernière décimale.

## La correction n'est pas applicable en l'état — `PROJECT_RHS = False`

J'ai d'abord activé la correction par défaut. **Elle casse le chemin AMR**,
et la suite de tests l'a montré : huit échecs, dont six sur des tests
préexistants.

`_rk4_step` a **trois** appelants, pas un :

| appelant | champ | projection |
|---|---|---|
| `step_full` | global périodique | **valide** |
| `step_layered` phase 1 | global **sous-échantillonné** | lève — `(256,256)` contre `(8,8)` |
| `step_layered` phase 2 | **patch local** avec halo | **pas périodique** : une projection spectrale périodique n'y est pas définie |

Projeter les deux premiers et pas le troisième romprait la garantie « à
`max_depth`, `step_layered` est identique à `step_full` » — propriété
documentée et testée.

Le drapeau reste donc à `False`, avec la raison écrite dans le code et un
test qui vérifie qu'elle y reste. Le choix — projection par taille de
grille, formulation à pression, ou autre — est une **décision de
modélisation**, pas une correction de défaut.

Trois voies, par coût croissant :

1. **Laisser en l'état.** Le solveur reste d'ordre 1,2, mesuré et documenté
   comme limite. La chute est **commune aux deux bras**, donc elle ne biaise
   pas leur comparaison.
2. **Projection par taille de grille**, plus une décision sur les patchs non
   périodiques. Casse probablement la garantie AMR.
3. **Formulation à pression** — la voie propre, mais c'est réécrire le cœur
   du solveur.

## Un test à moi était faux

`test_the_projected_rhs_is_divergence_free_at_every_stage` exigeait une
divergence **aux différences finies** nulle. Elle vaut 1,15e−2 — et ce n'est
pas un défaut de la projection : celle-ci est **spectrale**, elle annule la
divergence de Fourier, pas celle du stencil FD4. C'est exactement
l'incompatibilité déjà signalée entre le second membre (FD4) et la
projection (FFT). Mauvais opérateur choisi. Le test vérifie désormais ce que
la projection promet réellement : idempotence, et divergence **spectrale** à
la précision machine.

---

# Van Kan mesuré — non concluant ; et D-25, la projection qui abîmait B

## La question

`step_full` applique RK4 puis projette l'état : un splitting de Lie, d'ordre 1.
La correction de pression incrémentale (Van Kan) promet l'ordre 2 en ajoutant
le gradient du potentiel du pas précédent au second membre.

**Commande** — `scratchpad/vankan.py`, N=64, T=0,05, Orszag-Tang, grille fixe,
chaque schéma comparé à sa propre référence à 2048 pas.

## Le tableau

| schéma | erreur 32 pas | 256 pas | ordre | div_FD B | div_FD v |
|---|---|---|---|---|---|
| projection v **et** B *(actuel)* | 1,0665e−04 | 1,1853e−05 | 1,02 → 1,10 | **4,877e−06** | 2,914e−06 |
| projection de **v seul** | 1,0665e−04 | 1,1853e−05 | 1,02 → 1,10 | **2,818e−14** | 2,914e−06 |
| **Van Kan** sur v, B non projeté | 1,0411e−04 | 1,1563e−05 | 1,02 → 1,10 | 2,928e−14 | 2,914e−06 |

## Verdict Van Kan : non

**L'ordre reste à 1,10 dans les trois cas.** Van Kan gagne 2,4 % sur l'erreur,
rien de plus. La théorie promet l'ordre 2 ; on ne l'obtient pas.

Hypothèse **non vérifiée** : le même désaccord FD/spectral que D-25. Le
gradient de φ est calculé spectralement et ajouté à un second membre FD4,
donc la correction n'annule pas ce qu'elle devrait. Le vérifier demanderait
un solveur de Poisson FD-cohérent — la réécriture qu'on cherchait à éviter.
L'implémentation elle-même peut aussi être fautive.

## D-25 : la projection spectrale abîmait un champ déjà solénoïdal

La deuxième ligne est un gain net, et il ne doit rien à Van Kan.

L'induction est en forme rotationnelle : `rhs_B = (∂Ez/∂y, −∂Ez/∂x)`. Sa
divergence **aux différences finies** vaut `∂²Ez/∂x∂y − ∂²Ez/∂y∂x`, exactement
nulle puisque les décalages de `np.roll` commutent. **B est solénoïdal par
construction, dans l'opérateur même qui construit le second membre.**

La projection, elle, est **spectrale**. Appliquée à ce champ, elle n'y nettoie
rien : elle y injecte le désaccord entre les deux opérateurs.

| divergence FD4 du champ B, Orszag-Tang N=64 | |
|---|---|
| second membre | 1,97e−14 |
| état, 50 pas **sans** projection | 1,00e−14 |
| état, 50 pas **avec** projection | **4,63e−07** |

**Huit ordres de grandeur sur la contrainte, pour une erreur identique à la
quatrième décimale.** La projection de B ne coûtait rien en précision et
dégradait la seule chose qu'elle était censée garantir.

La vitesse, elle, en a besoin : `div_FD(rhs_v)` vaut **4,17** en relatif.

`PROJECT_B = False` par défaut ; `True` reproduit le chemin historique bit à
bit. La garantie AMR — `step_layered` ≡ `step_full` à `max_depth` — tient
toujours à **3,331e−16**.

## Note de méthode : l'opérateur de mesure décidait du verdict

La première mesure de ce défaut, faite avec la divergence **spectrale**, ne
montrait rien : 9,5e−02, indistinguable du bruit. C'est en la refaisant avec
l'opérateur **assorti** — le même stencil FD4 que le second membre — que
l'écart de huit ordres apparaît.

Troisième fois que ce piège se referme dans ce dépôt. Une grandeur discrète
n'a de valeur que relativement à l'opérateur qui la calcule ; mesurer avec un
autre ne mesure pas le champ, mais l'écart entre deux opérateurs.

## Tests

`tests/test_solver_convergence.py`, 7 tests : l'induction préserve la
divergence de B (contrôle positif), le second membre de v **ne** la préserve
pas (contrôle négatif, sans quoi on croirait qu'aucune projection n'est
nécessaire), la projection dégrade la contrainte de quatre ordres au moins,
la vitesse reste projetée, la raison est écrite dans la docstring, et le
retrait ne coûte rien en précision.

---

# D-27 — la projection amputait la perturbation de quatre scénarios

**Commande.** `pytest tests/solver/test_scenarios_analytic.py -k "solenoidal or amputates"`
· commit de la mesure : `git rev-parse HEAD`

## Ce qui se passait

En 2-D, `div B = ∂Bx/∂x + ∂By/∂y = 0`. `harris_tearing` pose un `Bx` qui ne
dépend que de `y` — donc `∂Bx/∂x = 0` — et une perturbation
`δBy = ε·cos(kx)·sech²(y)` dont `∂By/∂y ≠ 0`. La perturbation **viole** donc
la contrainte, et la projection la rabote pour la rétablir.

Le défaut n'était pas visible tant que B était projeté : la projection
masquait la divergence qu'elle corrigeait. Il n'apparaît qu'une fois D-25
corrigé — **trouvé en retirant une couche, pas en posant une question.**

| scénario | div_FD B relative | perturbation conservée |
|---|---|---|
| **`harris_tearing`** *(déployé)* | 2,801e−03 | **27,5 %** |
| `island_coalescence` | 1,400e−02 | **27,5 %** |
| `noisy_uniform` | 4,947e−01 | 55,7 % |
| `double_tearing` | 9,062e−04 | 77,3 % |

`harris_tearing` amorce son mode de déchirement par cette perturbation. **La
projection en retirait 72,5 %.** Le plan notait que ce scénario « dégénère
dans toutes les configurations testées, sans explication » ; ceci en est
peut-être une part — **non affirmé**, seule l'amplitude initiale est mesurée.

## La correction, et sa première version fausse

La perturbation s'écrit comme le rotationnel d'une fonction de flux,
`δB = ∇×(ψ ẑ)`, solénoïdale par construction — comme cela avait déjà été fait
pour `magnetic_twist` et `ghost_twisting`.

**Première version : dérivées analytiques de ψ.** Résultat `div_FD B` =
**2,1e−05**, pas 1e−16. Le champ était exactement solénoïdal pour l'opérateur
continu, et faux pour celui du solveur — l'erreur mesurée était l'erreur de
discrétisation. **Cinquième occurrence du même piège** : mesurer une grandeur
discrète avec un stencil autre que celui qui l'a produite.

**Version retenue :** dériver ψ avec le **même** stencil FD4 que le second
membre. `div(rot ψ) = ∂x∂y ψ − ∂y∂x ψ` est alors exactement nul, parce que les
deux dérivées FD4 sont des combinaisons de `np.roll` et commutent.

```python
@staticmethod
def _curl_z_fd4(psi, dx):
    g_x, g_y = MHDSolver._fd_grad(psi, dx)
    return g_y, -g_x
```

À N=64, `div_FD B` normalisée par `max|Bx| + max|By|` et multipliée par `dx` :

| scénario | div_FD B relative | \|δB\| max |
|---|---|---|
| `harris_tearing` | **1,076e−16** | 0,010000 |
| `double_tearing` | **1,863e−16** | 0,010000 |
| `island_coalescence` | **9,862e−17** | 0,050000 |
| `noisy_uniform` | **1,275e−16** | 0,239451 |

Ces valeurs sont au niveau du bruit d'arrondi : elles bougent au dernier
chiffre avec la résolution. Les tests posent donc un **seuil** (`< 1e−12`),
pas une égalité — un test calibré sur la mesure du jour cesserait de mesurer
au premier changement légitime.

Douze à treize ordres de grandeur, et l'amplitude nominale entièrement
conservée. Les champs de ces quatre scénarios changent : **tout nombre publié
qui les traverse est à refaire.**

## Tests

`tests/solver/test_scenarios_analytic.py`, 159 tests dont
`test_a_flux_function_perturbation_is_exactly_divergence_free` et
`test_an_analytic_derivative_would_not_have_been_exact` — ce dernier épingle
*pourquoi* la première version ne suffisait pas, avec sa mesure.

---

# D-29 à D-36 — audit du script d'entraînement

**Commandes.**
`pytest tests/pipeline/test_train_hyperparams_contracts.py` (60 tests, ~14 s)
`pytest tests/pipeline/test_train_hyperparams_smoke.py` (7 tests, ~16 s)

Quatre variantes du script d'entraînement coexistaient
(`TrainHyperParam_v1/v2/v3/v4.py`, 3 009 lignes). **Trois sont supprimées**,
la quatrième renommée `src/train_hyperparams.py`. C'est elle qui tournera sur
les cœurs loués ; l'audit porte sur elle seule.

## D-29 — le jeu « isolé » contenait les scénarios complexes

Tout l'argument du protocole est qu'Orszag-Tang mélange les classes
d'anomalies, donc qu'il faut des scénarios qui en isolent une. La liste
disait :

```python
SCENARIOS_ISOLATED = [("kh", …), ("ot", …), ("tearing", …), ("rotor", …)]
SCENARIOS_COMPLEX  = [("ot", …), ("rotor", …)]
SCENARIOS_ALL      = SCENARIOS_ISOLATED + SCENARIOS_COMPLEX
```

Trois conséquences, toutes silencieuses :

- le jeu « isolé » n'isolait rien : il contenait les deux scénarios complexes ;
- `SCENARIO_VORTEX` et `SCENARIO_COALESCENCE` étaient **définis et jamais
  utilisés** ;
- `SCENARIOS_ALL` valait **6 entrées pour 4 classes distinctes**. La perte
  composite `mean(Loss_i)` divisait par 6 une somme où `ot` et `rotor`
  entraient deux fois : pondération **2:1** contre `kh` et `tearing`, pour le
  **double du coût** de simulation.

**Ce qui tranche.** Le JSON déployé porte, pour sa phase 1, un bloc
`per_scenario` qui liste `kelvin_helmholtz`, `lamb_oseen_vortex`,
`harris_tearing`, `island_coalescence`. C'est la liste des quatre isolés qui a
produit la campagne gelée : la version trouvée dans le code était une
**régression**, pas une intention.

Corrigé, plus une garde `_assert_scenarios_wellformed` qui refuse un doublon,
un jeu vide, ou une trace DNS manquante — et qui refuse **à la construction de
l'objectif**, pas au milieu du premier essai, c'est-à-dire après le pré-calcul
DNS.

**Conséquence pour `study/`.** `closed_loop_campaign.fold_scenarios`
dédoublonnait ce défaut à la main pour éviter de fabriquer une fuite LOSO. La
déduplication ne retire plus rien, mais il y a désormais **six** folds LOSO
possibles au lieu de quatre : les résultats publiés sur quatre folds ne sont
pas comparables terme à terme.

## D-30 — le chemin séquentiel ne pouvait pas finir

```python
study_p1 = _run_phase1(dns_traces)      # ligne 1346
…
study_p1 = _run_phase1(study_p1, dns_traces)   # ligne 1352 — deux arguments
```

`_run_phase1` prend **un** argument. Le chemin par défaut — celui qu'on obtient
sans `WORKER_PHASE` — lève donc `TypeError` **après** la phase 1, c'est-à-dire
après ses 600 essais. `_save_results(study_p1, study_p1, …)` passait par
ailleurs la même étude deux fois, dans les emplacements « phase 1 » et
« phase 1b », et lisait des clés `vortex` / `coalescence` qui n'existaient
dans aucune des deux.

## D-31 — un paramètre optimisé que rien ne lit

Avec `split_michelson=False`, la phase 1 proposait `beta_michelson` à Optuna.
`pipeline.py` ne lit ce nom **nulle part** : il n'apparaît que dans un bloc mis
en commentaire. La phase optimisait donc un paramètre sans effet sur la perte.
Le chemin vivant passait `split_michelson=True`, donc le défaut n'a pas
produit de nombre faux — mais l'option existait, documentée comme le
comportement de la phase 1.

`split_michelson`, `beta_michelson` et la « phase 1b » sont supprimés.
`make_phase3_objective` — 64 lignes, aucun appelant, sa propre copie non nommée
des quatre constantes — et `expand_split_beta_seeds` — qui mutait son argument
et dont le repli `params.pop(k, params.get(k, 0.5))` n'atteignait jamais le
`get` — sont supprimés aussi.

## D-32 — l'élagage était décoratif

L'objectif composite rapportait **une** valeur, au step 0 :

```python
trial.report(composite, step=0)
if trial.should_prune(): …
```

sous un `MedianPruner(n_warmup_steps=2)`. Un pruner ne mord jamais avant
`n_warmup_steps` : `should_prune()` au step 0 renvoie toujours `False`.

**Mesure.** 40 essais terminés à 1,0 ; un essai qui rapporte **1e9** au step 0 :
`should_prune()` = `False`. Le garde-fou n'a jamais élagué un seul essai.

Corrigé : la moyenne **courante** est rapportée après chaque scénario, au step
égal à son indice. Le même 1e9 rapporté aux steps 0, 1, 2 déclenche désormais
l'élagage — et un essai élagué au 3ᵉ scénario ne simule pas le 4ᵉ, ce qui est
tout l'intérêt sur des cœurs loués. La comparabilité entre essais impose que
l'ordre des scénarios soit fixe : `SCENARIOS_*` sont des tuples.

## D-33 — Orszag-Tang tournait sans anomalies avancées

`create_argus` lisait `scenario_config.get("AdvAnomaliesEnable", False)`.
`SCENARIO_OT` était le seul des six à ne pas porter la clé.

**Mesure.** `create_argus(SCENARIO_OT).AdvAnomaliesEnable` = `False`,
`create_argus(SCENARIO_ROTOR).AdvAnomaliesEnable` = `True`.

Le terme ZZZZ de point X n'existe pas sans anomalies avancées. La phase 2
entraînait donc `beta_xpoint` sur un jeu de deux scénarios dont **l'un ne
pouvait pas l'exprimer**. Le même oubli existait dans la table `PHASE` de
`pipeline.main()`.

Corrigé aux deux endroits, et `create_argus` **lève** désormais si une clé
manque : le repli silencieux sur une valeur valide est exactement le motif
qu'on cherche à éliminer.

## D-34 — le budget d'essais était multiplié par le nombre de workers

```python
remaining = phase_config["n_trials"] - trials_done      # calculé UNE fois
study.optimize(objective_fn, n_trials=remaining)
```

Chaque worker lit le compte **au démarrage**. N workers lancés ensemble lisent
tous « 0 fait » et demandent chacun la campagne entière.

**Mesure.** 4 workers, cible 12 essais : **48 essais** exécutés. À l'échelle
réelle — 8 cœurs, `n_trials=600` — cela ferait **4 800 essais au lieu de 600**,
huit fois le coût annoncé.

Corrigé : la boucle relit le compte à chaque essai et s'arrête dès la cible
atteinte, quel que soit le nombre de workers. Coût : une lecture de base par
essai, contre 10 à 20 minutes de calcul. `WORKER_TRIALS` reste un plafond
**par worker**, pour une durée de location bornée.

## D-35 — le JSON final ne portait pas de quoi redéployer

`_save_results` écrivait `study_p3.best_params`. Or `best_params` ne contient
que ce qu'Optuna a **échantillonné** : les paramètres fixes n'y sont pas. Le
fichier déployé était donc structurellement incomplet, et le déploiement
comblait les manques par des replis que personne n'avait choisis.

**C'est le mécanisme de D-22** : `sigma` disparu du JSON, `gamma_hydro`,
`gamma_mag` et `kappa` présents dans le fichier déployé alors qu'**aucune base
Optuna ne les a jamais échantillonnés**.

Corrigé sur deux plans :

- chaque essai porte son dictionnaire d'hyperparamètres **résolu** — exploré +
  fixe — en `user_attr` ;
- le JSON porte ce dictionnaire, plus l'espace de recherche avec ses bornes,
  les paramètres fixes, la liste des scénarios, `lambda_cost`, le hash du
  commit, la propreté de l'arbre de travail, et `sys.argv`.

`deployable_params` signale par ailleurs, plutôt que de le taire, le cas où il
doit reconstruire faute d'attribut résolu.

## D-36 — la provenance de `sigma` n'existait que sur les runs jetés

`pipeline` a **quatre** sorties `return_details`. Une seule portait `sigma` et
`sigma_source` : celle du chemin d'exception de scoring. La trace exigée par
D-22 n'existait donc que sur les runs **divergés** — jamais sur ceux qu'on
publie.

Les quatre passent désormais par un `_details` unique. Le test ne cherche plus
une chaîne dans le source : il parcourt l'AST de `pipeline` et vérifie
qu'aucune sortie sous `if return_details` ne s'échappe du helper.

## L'espace de recherche, désormais déclaré

Le périmètre décidé pour la réoptimisation est de **8 paramètres** :

| paramètre | bornes | échelle |
|---|---|---|
| `beta` | 0,5 – 10,0 | linéaire |
| `w_z_frac` | 0,10 – 1000 | log |
| `sigma` | 0,02 – 0,30 | linéaire |
| `beta_curl` | 0,0 – 5,0 | linéaire |
| `beta_xpoint` | 0,0 – 5,0 | linéaire |
| `gamma_hydro` | 0,1 – 5,0 | linéaire |
| `gamma_mag` | 0,1 – 5,0 | linéaire |
| `kappa` | 0,5 – 50,0 | log |

`threshold_amr` reste **fixé** à 0,14959824837662078 — le meilleur essai de
l'étude classique — pour que la comparaison porte sur ce que le quantique
ajoute et non sur un seuil différent. C'est une décision, elle est déclarée
dans `FIXED_PARAMS` et vérifiée par un test qui exige qu'elle tombe dans les
bornes que le bras classique avait le droit d'explorer.

Trois de ces huit — `gamma_hydro`, `gamma_mag`, `kappa` — n'ont **jamais** été
échantillonnés par aucune campagne : pour eux ce sera une première, pas une
reprise. Nuance qui allège : `g_strain + g_rot ≡ 1` exactement, donc `kappa`
ne pilote **qu'un** degré de liberté.

**Réserve consignée.** La borne haute de `w_z_frac` vaut 1000 alors que le
paramètre est documenté comme une *fraction* de la médiane des couplages. Elle
vient de la campagne gelée, dont la graine valait 500. Conservée telle quelle
pour ne pas changer la science en même temps que le code — mais elle est à
trancher avant la campagne.

## Ce qui vérifie tout cela avant de louer des cœurs

```bash
python src/train_hyperparams.py --print-space     # l'espace réel, sans rien calculer
pytest tests/pipeline/test_train_hyperparams_contracts.py -q    # 60 tests
pytest tests/pipeline/test_train_hyperparams_smoke.py -q        # 7 tests
```

Le second fichier n'est pas une simulation : il fait tourner le **vrai**
solveur, le **vrai** circuit, une **vraie** base Optuna et écrit un **vrai**
JSON de déploiement — à N=32, deux pas de temps, une profondeur de
raffinement, en 16 secondes. Les six scénarios y passent. Une campagne d'une
semaine ne doit pas être le premier endroit où l'on découvre qu'un scénario ne
s'initialise pas.

Ce qu'il ne montre pas : que l'objectif **discrimine**. À cette résolution il
n'y a qu'une décision de raffinement, donc les six sous-pertes sont égales
(0,285714). Il montre que le chemin complet s'exécute et que les artefacts en
sortent complets.

---

# D-37 — le biais Z et les couplages décrivaient deux grilles différentes

**Commande.** `pytest tests/amr/test_patch_encoding_shapes.py` (13 tests, ~16 s)

**Pourquoi on est allé voir.** `COUVERTURE.md` listait trois fonctions du
chemin d'entraînement dont le contrat n'avait jamais été audité. Elles
décident la valeur qu'une campagne d'une semaine va minimiser. Celle-ci est
tombée en instrumentant la marge du garde CFL : le pipeline **plantait** dès
qu'on le lançait avec `max_depth ≥ 2`.

## Ce qui se passait

`_prepare_vqa_input` construit les deux moitiés de l'Hamiltonien par deux
chemins distincts :

| | source | taille rendue |
|---|---|---|
| `C_edges`, `K_plaquettes`, `K_xpoint` | les **champs**, via `_downsample_fields(..., target_dim, pad)` | (4, 4) |
| `H_edges` — le biais Z | le **score**, via `_process_score(..., target_dim + 2·pad)` | **(6, 6)** |

À `depth > 0`, `_process_score` emprunte `_resize_padded_maxpool`, dont le
contrat est écrit dans sa docstring : *« Input shape: (N+2, M+2). Output
shape: (t_dim+2, t_dim+2). »* **Le halo est déjà ajouté par la fonction.**
L'appelant l'ajoutait une seconde fois.

`create_bounded_hamiltonian(dim=2)` indexe ses lectures par `dim` sur des
tableaux supposés `(dim+2, dim+2)`. Devant un tableau **trop grand**, il ne
lève pas : il lit le coin supérieur gauche et rend un Hamiltonien
parfaitement valide, calculé sur la mauvaise portion du patch. Le biais Z
d'un patch venait donc du **quart haut-gauche** de ce patch, plus un halo
situé deux cellules trop loin.

**Mesure**, `orszag_tang` à N=64 après 40 pas, patch (0:32, 0:32), cœur 2×2 :

```
ce qui était lu               ce qui aurait dû l'être
[[0.109 0.109 0.097 0.064]    [[0.109 0.109 0.064 0.005]
 [0.111 0.141 0.094 0.064]     [0.111 0.141 0.078 0.079]
 [0.095 0.096 0.070 0.065]     [0.068 0.073 0.106 0.106]
 [0.062 0.062 0.044 0.081]]    [0.069 0.078 0.075 0.079]]
```

Écart maximal **0,05814**, pour des coefficients dont le plus grand vaut
**0,14107** : **41 %**. Les deux premières colonnes coïncident — c'est le
recouvrement du coin — et tout le reste décrit une autre région.

## Depuis quand, et sur quoi

```
git log -S "target_dim + 2 * pad" -- src/Simulation/refinement.py
cf93ba3 2026-04-09 Q-HAS: report submission snapshot
```

**Depuis le premier commit du fichier.** Tous les niveaux de raffinement sauf
le premier passent par là : à `max_depth = 4`, trois niveaux sur quatre.
`depth = 0` est le seul épargné, parce qu'il est périodique et n'a pas de
halo.

Le garde de forme ajouté en auditant les mappeurs (`7c0ae2f`) transforme
depuis lors la lecture silencieuse en `ValueError`. C'est ce garde qui a
rendu le défaut visible — mais il rend aussi **le pipeline inutilisable à
`max_depth ≥ 2`** : dans l'état d'avant ce correctif, la campagne n'aurait
pas pu tourner du tout.

## Pourquoi les tests ne l'avaient pas vu

Les configurations rapides — celles qu'on écrit pour qu'un test tourne en
quelques secondes — utilisent `max_depth_override = 1`. À `max_depth = 1`, le
balayage traite `depth = 0` puis s'arrête : **le chemin borné n'est jamais
emprunté**. Le test de régression paramètre donc explicitement
`max_depth ∈ {1, 2, 3}`.

## Correction

Une ligne : `_process_score(local_score, depth == 0, target_dim)`. Le halo
vient de la fonction, une fois.

| | avant | après |
|---|---|---|
| `H_edges` à depth>0 | (6, 6) | **(4, 4)** |
| `C_edges` à depth>0 | (4, 4) | (4, 4) |
| pipeline à `max_depth=2` | `ValueError` | **s'exécute sur les 6 scénarios** |

**Tout nombre Q-HAS publié qui traverse un raffinement au-delà du premier
niveau est affecté.**

## Au passage — le garde CFL, vérifié et trouvé sain

C'est en cherchant sa marge que D-37 est tombé. `pipeline` abandonne l'essai
si `check_cfl() > 1.0`. `adapt_dt` et `check_cfl` emploient bien la **même**
vitesse rapide `c_fast = max|v| + max|B|`, donc la CFL réalisée vaut la cible.
Mesuré à travers le vrai pipeline, `max_depth = 2`, N=64 :

| scénario | CFL max | marge |
|---|---|---|
| kh, vortex, tearing, coalescence, rotor | 0,4000 – 0,4018 | **2,5×** |
| orszag_tang | 0,4042 | **2,47×** |

**Rectification.** Une première mesure donnait 0,755 pour `orszag_tang`, soit
une marge de 1,3×. Elle était fausse : j'avais instancié le solveur avec les
`Re`/`Rm` par défaut au lieu de 800, donc une viscosité différente de celle
de la trace DNS. Le garde n'est pas serré.

---

# D-38 — trois gardes de `execute` qui ne tenaient que sur le chemin testé

**Commande.**
`pytest tests/quantum/test_runtime_contracts.py -k "bound or null_hamiltonian or optimizer"`

Même famille que D-37 : des protections correctes là où on les regarde, et
absentes ailleurs.

## Le warm start passait outre un Hamiltonien nul

Quand tous les coefficients sont nuls, `execute` court-circuite l'optimisation
et — dit son commentaire — *« returning θ-init marginals »*. Il reprenait
pourtant les paramètres du warm start s'il y en avait un. Or sans terme de
coût, seul le mixer agit : il tourne l'état sans qu'aucun coût ne le justifie.

**Mesure**, 8 qubits, score classique 0,700, warm start β = (0,35 ; 0,30) :

| | marginales rendues |
|---|---|
| sans warm start | **0,7000** |
| avec warm start | **0,5535** |

21 % de déplacement sur une décision annoncée inchangée. Corrigé :
`optimal_params = np.zeros(2·reps)`, sans condition.

*Portée.* Depuis D-8, `create_period_hamiltonian` et
`create_bounded_hamiltonian` **lèvent** sur un Hamiltonien nul : la branche
n'est plus atteignable par `mapping`. Elle reste fausse là où elle est.

## Powell recevait des contraintes que scipy ignore

Le commentaire de `execute` explique pourquoi le mixer doit rester borné :
sans borne, COBYLA part à β = 1, rabat P(|1⟩) à ≈ 0,25 et **supprime tout
raffinement**. La borne était posée par `bounds` pour L-BFGS-B et par
`constraints` pour `("COBYLA", "Powell")`.

Powell n'accepte pas de contraintes. scipy le disait :

```
RuntimeWarning: Method Powell cannot handle constraints.
OptimizeWarning: Unknown solver options: rhobeg
```

…sur stderr, dans un essai parmi des centaines. Powell optimisait donc le
mixer **sans borne**. Corrigé : Powell passe par `bounds`, `constraints` reste
à COBYLA seul. Toute autre méthode **lève** désormais, au lieu de perdre la
borne en silence.

*Note de méthode.* Ma première correction a créé le même trou : elle refusait
les méthodes inconnues mais laissait Powell dans la liste des trois
autorisées. Le test qui l'a rattrapée est celui qui vérifie **|β| dans le
résultat**, pas celui qui vérifie que l'appel passe. Un test qui constate
qu'une fonction « ne plante pas » ne teste pas sa garantie.

## Le mode MPS écrasait définitivement le nombre de tirs

`sampler.options.default_shots = max(shots, 8192)` était appliqué à un objet
qui peut appartenir à `vqa_runtime`, donc **partagé par toute la campagne**.
Après un seul patch en MPS, chaque appel ultérieur tirait 8 192 coups quel que
soit `shots`. La valeur d'origine est restaurée après lecture.

*Portée.* Le backend déployé est `state_vector` ; ce chemin ne s'exécute pas
aujourd'hui.

## Vérifié et trouvé sain, dans le même passage

**L'ordre des paramètres du circuit.** Les contraintes bornent `x[0:reps]`,
ce qui n'est correct que si le circuit ordonne ses paramètres
`[β…, γ…]`. Mesuré : `['β[0]', 'β[1]', 'γ[0]', 'γ[1]']`. L'ordre vient du tri
alphabétique de Qiskit sur les noms — un détail d'implémentation d'une
bibliothèque extérieure. S'il changeait, la borne s'appliquerait à γ et le
mixer tournerait libre. **Épinglé par un test.**

**Le score, et l'équité entre les deux bras.** `weighted_relative_error` vaut
bien 0 sur une reconstruction exacte et 1 quand le bras rend zéro. La carte de
pondération est construite sur la **référence**, donc identique pour les deux
bras. La comptabilité de pixels est symétrique : même `step_layered`, même
`max_depth`, même `target_dim`, accumulation à chaque pas pour les deux.

**La réduction du score, entre les deux bras.** Le bras classique décide sur
`_process_score`, le bras quantique sur le `mini_score` de
`get_adaptive_flux`. Deux chemins, une seule question : donnent-ils le même
nombre ? Mesuré sur un champ aléatoire, `target_dim = 2` :

| profondeur | écart maximal |
|---|---|
| 0 | **0,000e+00** |
| 1 | **0,000e+00** |

Identiques. La comparaison des deux bras porte donc bien sur le critère, pas
sur la réduction.

**`_run_level_classical` contre `_run_level`.** Le bloc de décision — seuil,
TTL, sondage de bord, ventilation en `if/elif/else` — est structurellement
identique, correction D-16 comprise. Deux échappatoires `continue` du chemin
quantique (`prep is None`, `result is None`) sont **inatteignables** :
ni `_prepare_vqa_input` ni `call_vqa_shell` ne rendent `None`. Elles auraient
fait disparaître un patch du pavage, donc laissé une région sans traitement ;
elles ne le font pas.

---

# D-48 et les quatre poches partielles de V1

**Commande.** `pytest tests/pipeline/test_v1_partial_pockets.py -q` (18 tests, ~2 s)

`COUVERTURE.md` listait quatre modules « partiellement audités » : des
fonctions jamais soumises aux cinq questions, dans du code par ailleurs
relu. Trois d'entre elles décident ce que la campagne va mesurer. Une seule
trouvaille en est sortie — les trois autres poches sont **saines**, et le
dire est un résultat : sans cela, la passe suivante relit le même code.

## D-48 — `mode="hardware"` s'exécutait sur un simulateur sans le dire

**Ce qui se passait.** `VQARuntime.__init__` prend `mode`, l'assigne à
`self.mode`, et `_init_backend()` **ne le lit jamais** : le dispatch porte
uniquement sur `backend_name`. Aucun chemin du dépôt ne résout un backend
IBM réel.

| `backend_name` | `mode="simulator"` | `mode="hardware"` |
|---|---|---|
| `state_vector` | `AerSimulator` | **`AerSimulator`** |
| `matrix_product_state` | `AerSimulator` | **`AerSimulator`** |
| `aer` | `AerSimulator` | **`AerSimulator`** |
| `estimator` | `FakeFez` | **`FakeFez`** |

Identiques dans les quatre cas. `self.mode` est assigné à la ligne 43 de
`runtime.py` et **lu nulle part dans tout `src/`**.

**Ma prédiction était fausse, et c'est ce qui rend le défaut grave.**
J'attendais que `execute` lève : son chemin `mode != "simulator"` ouvre
`Session(backend=backend)` sur ce qui est toujours un simulateur. Mesuré :

```
Session(AerSimulator) : ACCEPTEE
```

`qiskit-ibm-runtime` l'accepte. Le run ne plantait donc pas — il ouvrait
une Session autour d'un simulateur, y construisait un estimateur avec
**découplage dynamique et twirling activés** (des options qui ne veulent
rien dire sur un simulateur), et rendait des nombres parfaitement
plausibles. Un résultat demandé « sur matériel » était un résultat de
simulateur, sans le moindre signalement.

`pipeline.main()` annonçait par ailleurs `--mode hardware` dans les choix
de sa CLI : une option affichée dans l'aide est une promesse.

**Correction.** Refus à trois endroits, dans l'ordre où on les rencontre :
`VQARuntime` lève à la construction, `execute` couvre le chemin hérité où
`vqa_runtime is None`, et `--mode` ne propose plus que `simulator`. Le
message nomme la cause — aucun backend matériel n'est câblé — au lieu de
laisser deviner.

## Vérifié et trouvé sain — la mémoire TTL

*Axes empruntés : détection fraîche, signal perdu, plusieurs pas hybrides
d'affilée, les deux bras.*

Le contrat annoncé est « survit 1 pas hybride après la dernière
détection ». Mesuré sur l'arbre entier (N=8, dim=2, profondeur 2) : après
un pas chaud, les 20 entrées valent `DEFAULT_TTL = 1` ; après un pas
froid, **les 20 valent 0** ; après un second pas froid, elles y restent.
Le sursis ne se réarme que sur une détection, jamais sur une visite.

**Une hypothèse mesurée et réfutée**, consignée pour qu'on ne la reforme
pas : je soupçonnais qu'un patch dont le parent cesse d'être raffiné ne
serait jamais visité, donc jamais décrémenté — sa TTL survivrait un nombre
arbitraire de pas. C'est faux : la TTL du **parent** le maintient dans
`next_level`, donc l'enfant est visité et décrémenté avec lui. Tout
l'arbre passe de 1 à 0 au même pas.

La mémoire ne croît pas non plus : les clés sont les bornes du pavage,
déterministes, donc leur nombre est borné par l'arbre et non par la durée
de la campagne — 20 entrées stables sur 5 pas consécutifs.

## Vérifié et trouvé sain — le bras `classical_only`

*Axes empruntés : `classical_only` seul, avec `classic_AMR_comp`, deux
exécutions identiques.*

C'est le bras de comparaison : un défaut ici ne fausse pas le quantique,
il fausse la référence. Trois questions, trois mesures sur `kelvin_helmholtz`
réduit :

| | mesure |
|---|---|
| déterminisme | deux exécutions, `combined` **identique au dernier chiffre** |
| interférence entre les deux sites d'appel | `classical_only=True` seul et avec `classic_AMR_comp=True` : **identique** |
| provenance de `sigma` (D-36) | `sigma_source = "loaded"` sur cette sortie aussi |

Le même détecteur est appelé depuis deux endroits de `pipeline`, avec deux
mémoires TTL distinctes (`ttl_map` / `ttl_map_classical`). Elles ne se
contaminent pas.

**Une observation à refaire à l'échelle**, qui n'est pas un défaut : dans
cette configuration réduite, le bras classique au seuil déployé rend
`patch_ratio = 1.0` — il raffine tout le domaine, donc n'économise rien.
À N=32 et profondeur 1 cela peut n'être qu'un effet de taille. À vérifier
sur la configuration de campagne avant d'en tirer quoi que ce soit.

## Vérifié et trouvé sain — le mode Colab

*Axe **non empruntable** ici : `google.colab` n'est pas importable dans cet
environnement. Ce qui est vérifiable l'est ; le reste est nommé.*

Hors Colab, `drive_dir` et `local_dir` valent `None` — une recopie non
gardée lèverait `TypeError` sur toute machine ordinaire. Les **trois**
recopies Drive (`ensure_dirs`, le rappel de sauvegarde de `run_phase`,
`_save_results`) sont chacune dans un `if IN_COLAB`, vérifié sur l'AST et
non par voisinage textuel. `ensure_dirs` est idempotente et n'écrit rien à
l'import.

**Risque opérationnel documenté, non corrigé** : sous Colab non distribué,
la base n'est recopiée vers Drive qu'un essai sur dix. Une session
interrompue peut perdre jusqu'à neuf essais — le disque local de Colab est
éphémère. C'est un compromis de conception, pas un défaut ; il devient une
décision le jour où la campagne tourne sur Colab plutôt que sur des cœurs
loués. Sorti en entrée de décision dans `DEFAUTS.md`.

## Ce que le refus a fait tomber — D-48 en miniature

Le refus posé dans `VQARuntime` a fait échouer un test existant :
`tests/pipeline/test_v1_guards.py::test_runtime_path_accepts_the_shot_option`
construisait un runtime avec `mode="local"`.

`"local"` **n'existe nulle part dans `src/`** : ce n'est ni un mode déployé,
ni un mode documenté, ni une valeur qu'un appelant produit. `pipeline` passe
`args.mode`, dont le défaut est `simulator` ; `train_hyperparams` écrit
`mode="simulator"` en dur. La chaîne inventée était acceptée pour la seule
raison qui faisait D-48 — `mode` n'était jamais lu.

C'est la meilleure confirmation qu'on pouvait avoir : le défaut avait déjà
contaminé un test, et ce test passait. Le caller est corrigé en
`mode="simulator"`, avec la raison écrite sur place.

**Et c'est la suite complète qui l'a trouvé, pas la suite ciblée.** Les 18
tests de la poche passaient ; les fichiers que la correction touchait
passaient. L'échec est apparu à 33 % d'un `pytest tests/` intégral, dans un
fichier qu'aucune des deux lectures ne désignait. Troisième occurrence du
même piège : **ne pas annoncer avant d'avoir lu la ligne de résumé d'un run
complet.**

## Un test qui mesurait la machine, pas le dépôt

**Commande.** `pytest tests/test_suite_integrity.py -q` (80 tests, < 1 s)

La même exécution complète a fait tomber un troisième test :
`test_every_package_directory_carries_its_init` signalait `tests/v3` et
`tests/v4` comme des paquets sans `__init__.py`.

Les deux dossiers ne contiennent **rien d'autre qu'un `__pycache__`** : ce
sont les résidus de la réorganisation de `tests/` par sous-système. Les
sources ont été déplacées, le bytecode est resté. Git ne suit pas les
dossiers vides — **un clone neuf ne les a jamais eus**, et les effacer ne
tient pas : ils reviennent avec le répertoire de travail. Le test rapportait
donc quelque chose de vrai localement et inexistant à l'arrivée.

C'est un cousin du balayage vide, dans l'autre sens : au lieu de passer sans
rien vérifier, il échouait sans que le dépôt ait quoi que ce soit à se
reprocher. Un test qui dépend de l'état d'une machine ne mesure pas le code.

**Correction.** Le critère porte désormais sur les dossiers qui portent
effectivement du `.py`. Un vrai sous-dossier de test ajouté sans
`__init__.py` en contient par construction : il reste attrapé.

**Et le garde-fou de l'assouplissement.** Relâcher un critère peut rendre un
test incapable d'échouer — c'est ce que ce dépôt s'interdit. Un second test,
`test_the_init_check_can_still_fail`, construit dans un `tmp_path` un vrai
dossier fautif *et* un résidu, puis exige que le premier soit signalé et le
second ignoré. Sans lui, l'assouplissement serait invérifiable.

---

# D-49 — `recompute_lambda_scores` : le chemin d'échec rendait 0

**Commande.** `pytest tests/pipeline/test_recompute_lambda_scores.py -q`
(12 tests, ~10 s)

Premier des cinq fichiers « jamais audités » de V1. Il ne produit aucun
nombre publié — il décide comment on **lit** les nombres publiés, en
recalculant le score combiné des essais Optuna avec un autre `lambda_cost`.

## Le cœur du script est sain — et c'est la mesure qui compte

Question 4, sur les données réelles : à `lambda = 0,4`, la valeur avec
laquelle les bases gelées ont été produites, `recompute_score` doit rendre
exactement `trial.value`.

| base | essais finis | écart max | classement |
|---|---|---|---|
| `classical_v2_phase1` | 125 | **2,220e−16** | identique |
| `q_has_v2_phase1` | 178 | **5,551e−17** | identique |

**303 essais, écart médian 0,000e+00.** Le script recalcule bien la fonction
objectif de la campagne, et pas une autre qui lui ressemble.

Le test sépare : à `lambda = 0,41` — 2,5 % d'écart — l'erreur passe à
**9,839e−03**, treize ordres de grandeur plus haut. Ce n'est donc pas un test
qui passerait quoi qu'il arrive.

Deux contrôles de sens, également passés : à `lambda = 0` le score vaut la
physique seule, et augmenter `lambda` déplace le score vers le coût dans le
bon sens sur les deux populations (`patch > phys` et `patch < phys`, toutes
deux non vides — un balayage dégénéré échoue).

## D-49 — un échec total sortait en succès

`main()` enveloppait **tout son corps** dans un unique `except Exception` :
chargement, rescore, écriture CSV, chacune des six figures, le balayage. Le
gestionnaire imprimait `Erreur lors du chargement : ...` puis laissait la
fonction rendre la main.

Mesuré, avant :

```
base inexistante         → « Erreur lors du chargement »   code 0
répertoire non écrivable → « Erreur lors du chargement »   code 0
```

Le second cas est le plus trompeur : l'étude **était chargée** — le script
venait d'annoncer `125 completed (finite)` — et c'est l'écriture qui
échouait. Le message accusait quand même le chargement. Un lanceur de
campagne qui teste `$?` voyait un succès dans les deux cas.

Après :

```
base inexistante         → [ERREUR] chargement de 'bidon' depuis ...   code 1
répertoire non écrivable → FileNotFoundError, trace sur makedirs       code 1
chemin nominal           → CSV + résumé + 4 figures produits           code 0
```

Le `try` ne couvre plus que le chargement. Le reste remonte avec sa trace :
un rescore à moitié écrit qui s'annonce « Done » est pire qu'un rescore
absent.

## Un piège armé, mesuré, non déclenché

`recompute_score` détecte les scénarios **par essai** ; `build_trial_table`
les détecte sur l'**ensemble**, et seulement sur `completed[:10]`. Si un
essai portait un jeu de clés différent de ses voisins, son score serait
moyenné sur un autre dénominateur — puis classé avec eux, sans signalement.

Mesuré sur les deux bases gelées : les deux chemins **coïncident sur 100 %
des 303 essais**. Le piège est armé, pas déclenché. Il est figé par un test
qui tombera sur la première campagne produisant des essais hétérogènes,
plutôt que dans un classement silencieusement faussé.

La limite de l'échantillon à dix essais est figée de la même façon : le test
n'exige pas qu'on la corrige, il exige qu'elle reste **connue**.

## Reste à lire dans ce fichier

Les six fonctions de tracé (`plot_pareto_with_isocost`,
`plot_convergence_reranked`, `plot_decomposition_rescored`,
`plot_scenario_reranked`, `plot_lambda_sweep`, `_pareto_front`) n'ont pas
encore été relues ligne à ligne. Elles ne rendent aucune valeur consommée
par un autre script — elles écrivent des `.png`. Le chemin de données, lui,
est audité de bout en bout.

---

# D-50 — `analyze_hyperparams` : même piège, diagnostic qui désigne la mauvaise cause

**Commande.** `pytest tests/pipeline/test_analyze_hyperparams.py -q`
(11 tests, ~18 s)

Deuxième des cinq fichiers jamais audités. Il ne produit aucun nombre
publié — il produit le résumé et les seize figures **à partir desquels on
décide**. Un diagnostic faux y coûte autant qu'un nombre faux ailleurs.

## D-50 — l'échec rendait 0, et accusait une base distante

`main()` enveloppait tout son corps dans **deux** gestionnaires :

```python
except KeyError:
    print(f"⚠️  Skipping {study}: Study does not exist on Neon yet.")
except Exception as e:
    print(f"❌ Error loading study: {e}")
    return
```

Mesuré sur un `.db` local inexistant :

```
⚠️  Skipping bidon: Study does not exist on Neon yet.     code 0
```

Deux erreurs dans une seule ligne. Le code de retour annonce un succès. Et
le message accuse **Neon** — une base distante qui n'intervient nulle part
dans ce chemin — pour un fichier local absent : il envoie chercher la panne
au mauvais endroit.

La branche `KeyError` est le piège le plus fin : elle enveloppait aussi les
**treize fonctions de tracé**. Une clé d'attribut manquante dans n'importe
quelle figure était donc annoncée comme « l'étude n'existe pas », alors que
l'étude venait d'être chargée et résumée.

Après : le `try` ne couvre plus que le chargement, les deux gestionnaires
généraux disparaissent, et l'échec sort en **code 1** avec la cause réelle.
Chemin nominal revérifié : **10 fichiers produits, code 0**.

## Corrigé au passage — une détection qui inventait trois scénarios

`_detect_scenario_keys` existe en **deux copies**, une par script d'analyse.
Elles avaient divergé — préfixe différent (`loss_` contre `phys_`) et
surtout sémantique différente : la copie d'`analyze_hyperparams` rendait
**toute la famille** dès qu'**une** clé était trouvée, sans vérifier que les
autres existent.

Mesuré sur les deux bases gelées, qui portent quatre scénarios :

| | scénarios annoncés |
|---|---|
| données réelles | `kh`, `tearing`, `ot`, `rotor` — **4** |
| `recompute_lambda_scores._detect_scenario_keys` | les mêmes 4 |
| `analyze_hyperparams._detect_scenario_keys` | **7** — plus `vortex`, `coalescence`, `gt` |

Trois scénarios qu'**aucune campagne n'a jamais exécutés**.

**Aucun nombre faux n'en sortait, et c'est la mesure qui le dit.** Les
quatre appelants filtrent tous par `if f"loss_{k}" in attrs` ; le résumé
imprime bien quatre lignes, et « Lamb-Oseen », « Island Coalescence » et
« Ghost Twisting » n'y figurent pas. Ma première lecture concluait au
défaut ; la mesure l'a corrigée. Le piège était **armé, non déclenché** —
le premier appelant qui ferait confiance à la liste obtiendrait un
`KeyError`, ou pire une moyenne sur sept termes dont trois `NaN`.

La correction rend la docstring vraie sans rien changer en sortie :
**empreinte SHA-256 du résumé complet identique avant et après**, sur les
deux bases.

```
classical_v2_phase1   8bf8d878…3018   identique
q_has_v2_phase1       dc746ee4…1d36   identique
```

Un test compare désormais les deux copies sur la même base : elles doivent
en tirer les mêmes scénarios, sans quoi les deux analyses d'une même
campagne ne parlent pas du même sous-ensemble.

## Deux corrections de `PROVENANCE.md`

En vérifiant les quotas de la campagne gelée contre `PHASES` :

- « 143 et 202 essais contre **600** déclarés » — le quota classique vaut
  **300**, pas 600. Le bras classique s'est arrêté à 143/300 (48 %), le
  quantique à 202/600 (34 %).
- deux chemins morts : `tests/v4/test_hyperparams_provenance.py` (déplacé
  vers `tests/study/`) et `src/TrainHyperParam_v1.py à _v4.py` (supprimés).
  Une commande de vérification qui ne peut plus être lancée ne vérifie rien.

## Reste à lire dans ce fichier

Les treize fonctions de tracé. Comme pour `recompute_lambda_scores`, elles
n'écrivent que des `.png` : le chemin de données et le chemin d'échec sont
audités, la mise en page ne l'est pas.

---

# D-66 et D-67 — le lancement complet ne calculait rien

**Commande.** `pytest tests/pipeline/test_full_launch_config.py -q`
(8 tests, ~10 s)

`python src/pipeline.py` est le **seul** moyen de lancer une simulation
complète à la main. Rien dans le dépôt n'importe `main` — `train_hyperparams`,
`closed_loop_campaign` et les tests importent tous la *fonction* `pipeline`.
C'est précisément pourquoi personne ne voyait que la CLI ne calculait rien.

## Mesure — avant

Invocation par défaut, `orszag_tang`, **code de retour 0** :

```
FINAL COMPARISON: Q-HAS vs Classical AMR vs DNS
combined............     0.333333     0.333333
phys_score..........     0.000000     0.000000
patch_ratio.........       1.0000       1.0000
error_vx … error_Jz      0.000000     0.000000
```

Erreur **exactement nulle** sur les cinq champs, pour les deux bras. Ce n'est
pas une performance parfaite : c'est un run qui n'a rien intégré.
`combined = 0,333333` vaut simplement `(0 + 0,5×1) / 1,5`.

## D-66 — sept clés de configuration sur neuf étaient ignorées

`main()` précalculait le DNS avec `PHASE[scenario]`, puis passait à
`pipeline()` les **défauts de la ligne de commande** :

| | `PHASE["orszag_tang"]` | ce que `main()` passait |
|---|---|---|
| `T_MAX` | 2,8 | **1,0** |
| `DT` | 1e−3 | **1e−4** |
| `Re` / `Rm` | 800 | **1000** |
| `K_opt` | 30 | 80 |
| `shots` | 256 | 1024 |
| `AdvAnomaliesEnable` | True | **False** |

Le DNS était donc calculé sous une physique et la boucle hybride sous une
autre. Et le hot start place `t_current` à `T_START = 2,3` : avec
`T_MAX = 1,0`, la condition `while t_current < T_MAX` est fausse d'entrée —
**le corps de la boucle ne s'exécutait jamais**. L'état final restait l'état
DNS, d'où l'erreur nulle.

Effet secondaire : le `AdvAnomaliesEnable: True` ajouté pour D-33 était
**inerte sur ce chemin**, `argus` étant construit depuis `args`.

**Correction.** `PHASE` fait foi ; la CLI ne surcharge que ce qu'on lui passe
explicitement (défauts à `None`) ; une garde refuse `T_MAX ≤ T_START` en
nommant la cause. `PHASE` est sorti au niveau module pour être testable, et
les choix de `--scenario` en sont **dérivés** : la liste écrite à la main en
annonçait dix pour sept entrées — `magnetic_twist`, `noisy_uniform` et
`double_tearing` étaient acceptés puis levaient `KeyError`.

## Mesure — après

```
Q-HAS      combined 0.228928  phys 0.140052  patch 0.4067
Classique  combined 0.212591  phys 0.117626  patch 0.4025
error_Jz      0.332329     0.256746
```

Physiquement cohérent : ~12–14 % d'erreur relative pour 40 % du coût, et `Jz`
— une dérivée du champ — porte l'erreur la plus forte. Le classique bat Q-HAS
sur 4 champs sur 5 et sur le score combiné, à coût égal, avec les
hyperparamètres non réoptimisés.

## D-67 — un run vide se notait

`score()` repliait `total_steps == 0` sur `avg_pixel_used = N_square`, donc
`patch_ratio = 1,0`, donc `combined = λ/(1+λ)`. C'est ce repli qui a rendu
D-66 invisible pendant tout ce temps : un run qui n'a rien calculé rendait un
nombre plausible. Il lève désormais.

**Ce que le test d'épinglage a révélé.** La correction a fait tomber
`test_zero_steps_silently_scores_the_worst_possible_cost` — un test qui
figeait l'ancien comportement en le nommant « choix visible », et dont la
docstring notait déjà le risque : *« exploitable par Optuna au lieu
d'échouer »*. Il a fait exactement son travail : rendre le changement
visible au lieu de le laisser passer.

Le risque se chiffre. Avec les valeurs mesurées du run de référence
(`phys = 0,140052`, `patch = 0,4067`), un run vide bat un run réel dès que

    λ < phys / (1 − patch) = 0,2361

À `LAMBDA_COST_SOFT = 0,4` le run vide perd (0,2857 contre 0,2162). Mais
`recompute_lambda_scores` rescore les essais à λ = 0,0 / 0,1 / 0,2 — et sous
ces trois valeurs, **un essai dégénéré devient le meilleur essai de la
campagne**. Le test est remesuré, pas ajusté : l'ancienne valeur et la
nouvelle sont toutes deux consignées.

## Ce que la vérification quantitative a confirmé

Le pavage AMR, sur deux cartes successives du run réel : **aire cumulée
65 536 = 256², zéro cellule non couverte, zéro recouvrement.** La correction
D-16 tient à pleine échelle.

La comptabilité de coût est reproductible depuis la seule liste de patchs —
`Σ (H·W)/local_factor²` rend 15 616 → **0,2383** et 24 064 → **0,3672**, les
valeurs imprimées au chiffre près. Et la moyenne des 200 pas
(`0,2383 … 0,5078`) vaut **0,4067**, exactement le `patch_ratio` annoncé.

*Le `patch_ratio = 1,0` mesuré en configuration réduite (N=32, profondeur 1)
lors de l'audit du bras `classical_only` était donc bien un effet de taille,
comme il avait été noté sans pouvoir être tranché. À N=256 profondeur 4,
l'AMR économise 60 %.*

---

# D-10 — `compare_rotor_budget` n'avait jamais tourné

**Commande.** `pytest tests/pipeline/test_compare_rotor_budget.py -q`
(12 tests, ~4 s)

Seul fichier non audité de `src/` qui **écrit** un résultat numérique : c'est
un producteur, pas un analyseur. Il porte la démonstration d'avantage
quantique sous budget contraint.

**Trois défauts empilés, tous mesurés :**

1. `PhysicalMapper(..., beta=0.5, ...)` — `beta` a quitté le constructeur du
   mapper pour devenir un argument de `run_adaptive_vqa`. `TypeError` à
   l'**étape 4 sur 5**, après avoir payé le DNS.
2. `--n-blocks 4` demande `2×4² = 32` qubits, soit **69 Go** de statevector.
   Mesuré : `Insufficient memory ... Required memory: 65536M`. Le défaut
   annoncé n'était exécutable sur aucune machine.
3. Une fois `n_blocks` ramené à 3, la résolution 128 n'est plus divisible
   par 3.

**Corrections.** Le mapper reçoit les hyperparamètres **réellement déployés**
au lieu des constantes `gamma_hydro=0,5 / gamma_mag=0,5 / kappa=5,0`, qui
n'étaient celles d'aucune campagne — et les quatre clés que la signature
attend (`sigma`, `beta_curl`, `beta_xpoint`, `w_z_frac`) étaient absentes de
l'appel, donc silencieusement remplacées par les défauts du mapper. Une garde
refuse une taille de circuit hors mémoire **avant** le DNS. Les défauts
deviennent (96, 3, 3) : 9 blocs, budget 3, une contrainte qui contraint.

Il produit désormais son `.npz`. **Ce qu'il mesure est à lire avec
précaution** : à (96, 3, 3), la sélection dite « ground truth » — les blocs
de plus forte erreur DNS/grossier — rend une erreur L2 de **0,3079**, *pire*
que l'absence d'AMR (0,3074), tandis que classique et Q-HAS choisissent tous
deux la colonne centrale et obtiennent **0,0208**, soit 93 % de mieux. Le
« ground truth » du protocole n'est donc pas un optimum de raffinement, et le
verdict « agreement 0/3 » qu'il imprime ne mesure pas ce qu'il annonce. À
reprendre avant d'en tirer une revendication.

---

# La porte de `K_xpoint` retirée — et le second verrou qui reste

**Commande.** `pytest tests/mapping/test_xpoint_at_training_resolution.py -q`
(11 tests)

Décision de USER, prise après le banc analytique. Modification de `src/` :
`K_xpoint = -1.0 * f_Rm_cell * mic_xpoint` devient `K_xpoint = -1.0 * mic_xpoint`.

## Pourquoi la porte devait sauter

`f_Rm_cell = _f_gate(|B|·dx/η)` — et un point X est **par définition un zéro
de B**. La porte annulait donc le coefficient exactement à l'endroit qu'il
doit signaler. Mesuré sur une nappe de 2 cellules à N=256 :

| | au point X |
|---|---|
| seuil (`mic`) | **0,5292** |
| porte (`f_Rm`) | **0,0000** — sur les six épaisseurs testées |
| `K_xpoint` résultant | **0,0000** |
| `K_xpoint` sur l'anneau | 0,8537 |

Le détecteur marquait l'anneau, jamais le centre. Le commentaire d'origine
réclamait déjà ce retrait — *« No separate g-gate needed (signal is
intrinsically localized) »* — pendant que le code appliquait la porte.

Verrouillé par `test_le_vrai_mappeur_marque_le_point_X`, qui appelle
`compute_coefficients` elle-même. **Vérifié par mutation** : il échoue quand
on restaure la porte, il passe quand on la retire. Les autres tests du banc
reconstruisent la chaîne étage par étage pour l'inspecter — ils ne
verraient pas un changement dans `src/`, et c'est un piège que j'avais
d'abord laissé dans mon propre banc.

## Ce que ce retrait ne fait PAS — le second verrou

**Sur des champs réels, à la résolution d'entraînement, le terme reste nul.**
Le seuil n'est pas atteint, et de très loin :

| scénario | N=64 | N=128 | N=256 |
|---|---|---|---|
| `island_coalescence`, signal/seuil | 0,171 | 0,0105 | **0,0007** |
| `harris_tearing`, signal/seuil | 0,045 | 0,0029 | **0,0002** |

Il faut dépasser 1 pour que le terme tire. On en est à **1 400 fois** en
dessous à N=256 — et l'écart **empire quand la grille se raffine**.

La cause est une incohérence dimensionnelle. `sig = max(0, −det(J_B))` est
un gradient **au carré**, normalisé par `B0²/dx²` — une seule puissance de
`dx²` — puis comparé à un seuil lui-même au carré. Le rapport varie donc en
**dx⁴**. Mesuré : de N=64 à N=256, il chute d'un facteur **244**, contre 256
attendu pour dx⁴.

Le canal du courant, lui, est cohérent : `|Jz|/B0` comparé à
`η/(dx²·B0)` varie en **dx²** (facteur 15 mesuré, 16 attendu).

**Candidat, non appliqué :** `√sig` a les mêmes unités que `|Jz|`, donc la
même normalisation et le même seuil. Mesuré, il varie bien en **dx²**
(facteur 16) et gagne un facteur 37 à 68 sur le rapport — sans suffire à
franchir le seuil à N=256 sur ces champs.

| | N=64 | N=128 | N=256 | loi |
|---|---|---|---|---|
| forme actuelle | 0,171 | 0,0105 | 0,0007 | dx⁴ |
| candidat `√sig` | 0,414 | 0,1025 | 0,0256 | **dx²** |
| `Jz`, référence | 5,137 | 1,349 | 0,341 | dx² |

*À noter : à N=256, `K_plaquettes` est nul lui aussi sur ces champs — le
canal `Jz` tombe à 0,341, sous son propre seuil. Les deux familles ZZZZ se
taisent à la résolution d'entraînement, ce qui rejoint le verdict déjà
publié « le terme ZZZZ était numériquement mort ».*

**Décision ouverte.** Passer à la normalisation `√sig` rend le canal point X
dimensionnellement cohérent avec le canal courant. C'est un second changement
de `src/`, non pris.

---

# Les quatre familles de coefficients, et la correction dimensionnelle

**Commandes.**
`pytest tests/mapping/test_coefficient_families_contract.py -q` (14 tests)
`pytest tests/mapping/test_xpoint_at_training_resolution.py -q` (11 tests)

## Règle du banc : tout passe par la vraie fonction

Aucun test de ce banc ne reconstruit la chaîne. Chaque assertion interroge
`compute_coefficients` elle-même, puis `get_adaptive_flux` pour la partie
qui vérifie que **le circuit reçoit bien ce que le banc mesure**.

C'est une leçon payée : mon premier banc de points X reconstruisait les
étages pour les inspecter, et ses 20 tests passaient à l'identique **après**
une modification du mappeur. Un test qui ne voit pas `src/` changer ne
verrouille rien.

## Ce que chaque famille promet, et ce qu'elle fait

| famille | contrat | verdict |
|---|---|---|
| `H_edges` | nul au seuil, signe qui bascule, subordonné aux couplages | **sain** |
| `C_edges` | ferromagnétique, éteint loin du seuil | **sain** |
| `K_plaquettes` | négatif, nul sans structure, répond à la structure | **sain** |
| `K_xpoint` | X oui, O non, courant non | **sain** après correction |

Mesures sur rotation solide (N=64) : `H = 6,02e−03`, `C = 1,78`,
`K_plaquettes = 94,8`. Sur champ uniforme : tout à zéro. La fenêtre
d'incertitude éteint le couplage à 10 σ du seuil d'un facteur **> 10⁶**.

## Deux erreurs de ma part, toutes deux du même genre

Mon test de subordination du biais Z a échoué deux fois avant de mesurer la
bonne chose. Le code calcule `C_scale = median(|C|, |K|)` sur les valeurs
**> 1e−10** ; j'ai d'abord pris `> 0` sur `|C|` seul (médiane 5,9e−22),
puis `> 0` sur `|C|` et `|K|` (1,7e−22). Dans les deux cas la queue de la
fenêtre gaussienne écrasait la distribution et la borne devenait absurde.

**Le code était juste ; c'est le test qui mesurait autre chose.**
Reproduire un calcul sans reproduire son filtre donne un faux verdict —
c'est la même famille de piège que l'opérateur non assorti.

## La correction dimensionnelle de `K_xpoint`

`sqrt(det)` a les mêmes unités que `|Jz|` : tous deux sont des gradients de
B. Le canal point X emploie donc désormais la normalisation et le seuil du
canal courant (`jz_crit`), déjà définis.

L'ancienne forme comparait `sig / (B0/dx)²` à `(RM_CRIT·η/(dx·B0))²` :
`sig` est un gradient **au carré** normalisé par une seule puissance de
`dx²`, puis comparé à un seuil lui-même au carré. Le rapport variait en
**dx⁴** — le critère devenait moins susceptible de se déclencher à mesure
que la grille se raffine, à la puissance quatre.

| rapport signal/seuil | N=64 | N=128 | N=256 | loi |
|---|---|---|---|---|
| ancienne forme | 0,171 | 0,0105 | 0,0007 | dx⁴ |
| **forme actuelle** | **0,414** | **0,1025** | **0,0256** | **dx²** |
| `\|Jz\|`, référence | 5,137 | 1,349 | 0,341 | dx² |

Vérifié à travers la vraie fonction, sur des nappes d'épaisseur décroissante :

| | 8 cellules | 4 | 2 | 1 |
|---|---|---|---|---|
| N=128 | 0 | **0,232** | 0,848 | 1,810 |
| N=256 | 0 | 0 | **0,212** | 0,693 |

Le terme tire maintenant dès **4 cellules** à N=128, contre ≤3 auparavant.
Et dans tous les cas où il tire, **son maximum est exactement au nul** —
la suppression de la porte est ainsi confirmée à travers le chemin déployé,
pas seulement sur une reconstruction.

**Ce qui reste vrai malgré la correction :** sur les champs réels
(`island_coalescence`, `harris_tearing`) le terme reste nul aux trois
résolutions. Les nappes de courant y sont plus épaisses que le seuil
n'exige. Ce n'est pas un défaut du coefficient — c'est que ces scénarios,
à Rm = 800, sont résolus.

---

# Le critère devient RELATIF

**Commande.** `pytest tests/mapping/test_coefficient_families_contract.py -q`
(19 tests)

Décision de USER après mesure. `src/Simulation/HamiltParams.py` :
le seuil effectif vaut désormais `min(seuil_absolu, percentile(signal))`.

## Pourquoi

Le seuil de maille est **absolu** — `RE_CRIT·ν/(dx²·v0)`. Deux conséquences,
toutes deux mesurées :

**Il meurt au raffinement.** Sur un champ physique fixe (rotation solide),
`K_plaquettes` passe de 1,00e+02 à N=32 à **exactement 0** à N=256 — la
résolution d'entraînement.

**Il ne peut pas servir deux instabilités.** `|omega|` vaut au maximum
1,55e−02 sur `harris_tearing` et 1,96e+01 sur `mhd_rotor` — trois ordres de
grandeur, pour un seuil unique de 13,04.

Or l'information est là. Contraste max/médiane du signal brut à N=256 :
**1104** sur `harris_tearing` (`√det`), 223 sur `island_coalescence`, 752 sur
`mhd_rotor`. Ce n'est pas la structure qui manque, c'est le seuil absolu qui
l'efface.

## Ce que ça donne — mesure avant / après, N=256

| scénario | `K_plaq` avant → après | `K_xpoint` avant → après |
|---|---|---|
| `orszag_tang` | 0 → **2,78e−01** | 0 → 8,34e−02 |
| `harris_tearing` | 0 → 2,66e−03 | 0 → **9,81e−01** |
| `island_coalescence` | 0 → 1,54e−02 | 0 → **5,55e−01** |
| `mhd_rotor` | 0 → **6,68e+01** | 0 → 1,00e+01 |

Le terme à quatre corps n'existait sur **aucun** des quatre scénarios ; il
existe désormais sur les quatre.

**Et il se répartit comme la physique le prédit.** Sur les deux scénarios de
reconnexion, le canal point X **domine** le canal courant — 0,981 contre
0,0027 sur `harris_tearing` (facteur **370**), 0,555 contre 0,0154 sur
`island_coalescence` (facteur 36). C'est exactement ce qu'annonçait le
contraste brut (`√det` 1104× contre `|Jz|` 49,5×), et c'est le canal qui
était doublement cassé.

## Les deux clauses qui rendent le critère sûr

**L'absolu l'emporte quand il tire.** Dès qu'une cellule franchit le critère
physique, le comportement d'origine est conservé **à l'identique**. Le
relatif ne remplace pas la physique, il la complète.

**Il ne fabrique pas de signal.** Un champ rigoureusement uniforme n'a aucune
cellule « plus instable » : son percentile vaut son maximum, le contraste
seuillé rend zéro partout. C'est l'invariant le plus important du fichier de
tests, et il est vérifié sur `K_plaquettes` comme sur `K_xpoint`.

## Ce que le test d'épinglage a fait

`test_les_coefficients_s_effondrent_quand_la_grille_se_raffine` figeait
`K_plaquettes = 0` à N=256. Il est **tombé** — c'est son rôle. Remesuré, pas
ajusté : la table de sa docstring porte maintenant l'avant *et* l'après.

`H_edges` et `C_edges` continuent de décroître en résolution : ils sont
gouvernés par la fenêtre gaussienne et par `C_scale`, pas par le seuil de
maille. Mécanisme différent, non confondu.

## Conséquence pour la réoptimisation

`RELATIVE_PERCENTILE = 90` est un **réglage nouveau**. Le périmètre passe de
**8 à 9 paramètres**.

---

# Les coefficients pointent-ils où le raffinement est nécessaire ?

**Commande.** `pytest tests/mapping/test_coefficient_families_contract.py -q`
(20 tests)

Tous les autres tests vérifient des contrats internes — signes, seuils,
invariance. Celui-ci vérifie la seule chose qui justifie le modèle.

**Protocole.** Même scénario à N=128 (référence) et N=32 (grossier), même
nombre de pas. Erreur relative par bloc sur 8×8 = 64 blocs. Corrélation de
rang de Spearman contre le coefficient moyen du bloc.

| scénario | `K_plaq` | `K_xpoint` | `max(K)` | score classique |
|---|---|---|---|---|
| `harris_tearing` | **0,897** | 0,434 | 0,788 | 0,814 |
| `island_coalescence` | **0,877** | 0,408 | 0,760 | 0,912 |
| `mhd_rotor` | **0,755** | 0,680 | 0,759 | 0,528 |
| `orszag_tang` | 0,249 | 0,311 | 0,443 | 0,422 |

## Trois lectures

**Le contrat central est tenu.** `K_plaquettes` corrèle de **0,75 à 0,90**
sur trois scénarios sur quatre. Le coefficient désigne bien les blocs où la
solution grossière s'écarte du DNS.

**Sur `mhd_rotor`, le coefficient bat le score classique — 0,755 contre
0,528.** C'est la première preuve quantitative, dans ce dépôt, que le terme
à quatre corps apporte quelque chose que l'indicateur linéaire n'a pas. Et
c'est précisément le scénario autour duquel `compare_rotor_budget` a été
construit, avec l'argument « le classique ne distingue pas forte vorticité
sans Jz de forte vorticité **et** fort Jz ». La mesure va dans ce sens.

**`orszag_tang` est faible pour tout** — 0,25 à 0,44, coefficients et score
classique confondus. Ce n'est pas un défaut des coefficients : c'est le
scénario le plus difficile pour n'importe quel indicateur local.

*À noter : ces corrélations sont mesurées avec le critère relatif en place.
Avant lui, `K_plaquettes` était identiquement nul à N=256 et la corrélation
n'aurait pas été définie.*

## Architecture Neon supprimée

`src/import_Neon_data_to_local.py` est supprimé sur décision de USER. Le
fichier portait D-64 (il effaçait la destination avant de lire la source,
5 essais perdus à code 0) et D-65 (identifiant PostgreSQL en dur dans un
dépôt public). Le mot de passe reste dans l'historique git ; sa rotation
n'est plus nécessaire puisque l'architecture est abandonnée.

**Note 🦉 (ce qui reste vrai après la décision, pour que personne ne le
redécouvre).** Abandonner l'architecture retire l'*usage* du mot de passe,
pas le mot de passe : l'identifiant `neondb_owner` publié le 13 août reste
valide côté Neon tant qu'il n'est pas changé là-bas, et l'historique git le
sert toujours. La décision de ne pas faire tourner est celle de USER et
n'est pas rediscutée ici ; elle est écrite ici pour qu'elle ne soit pas
prise pour un oubli — et parce que `D-65` sort de `DEFAUTS.md` avec elle.
Ce qui subsiste côté dépôt est le garde-fou : `pytest
tests/pipeline/test_no_credential_in_source.py` refuse toute URL portant un
mot de passe dans `src/` (2 tests, balayage vide inclus).

La configuration de pooling de `train_hyperparams._get_storage` est
conservée : elle vaut pour n'importe quel Postgres distant, pas seulement
pour Neon. Son commentaire est généralisé en conséquence.

---

# Équilibre entre les familles : trois faits sains, un déséquilibre ouvert

**Commande.** `pytest tests/mapping/test_coefficient_families_contract.py -q`
(21 tests)

## Une correction de ma part, d'abord

La première version de cette matrice utilisait une **rotation solide**
`v = (−(y−L/2), x−L/2)` comme champ fluide. Ce champ est **discontinu au
raccord périodique** : son `K_plaquettes` valait 9,48e+01 avec un maximum
dans le coin (63, 63). Un artefact de bord, pas une mesure.

Le réseau de vortex `v = (−sin y, sin x)`, périodique, donne **5,01e−01**
dans l'intérieur. **J'avais donc surestimé le canal fluide d'un facteur
190**, et annoncé un déséquilibre de 10⁶ là où il est de 2,75·10⁴.

C'est la même famille de piège que le champ d'essai qui ne sépare pas :
un champ analytique qui viole une hypothèse du solveur mesure le solveur,
pas la physique.

## La matrice, champs périodiques uniquement

| champ | `H_edges` | `C_edges` | `K_plaq` | `K_xpoint` |
|---|---|---|---|---|
| réseau de vortex (fluide) | 1,16e−05 | 1,83e−01 | **5,01e−01** | 0 |
| nappe de courant (magnét.) | 1,57e−07 | 7,18e−05 | **1,82e−05** | 0 |
| cisaillement v (Q<0) | 2,13e−08 | 2,72e−04 | 5,71e−03 | 0 |
| point X magnétique | 9,95e−07 | 3,33e−01 | 5,09e−06 | **9,59e−02** |
| uniforme (contrôle) | 0 | 0 | 0 | 0 |

## Ce qui est sain

**Le contrôle rend zéro** sur les quatre familles.

**Le réseau de vortex allume `K_plaquettes` et laisse `K_xpoint` à zéro** —
le canal fluide ne déborde pas.

**Les deux canaux ZZZZ sont orthogonaux.** Le point X allume `K_xpoint`
(9,59e−02) et laisse `K_plaquettes` à 5,09e−06 — **19 000 fois moins**.
C'est leur raison d'être, et elle est vérifiée.

**`H_edges` reste subordonné** partout.

## Le déséquilibre, ouvert

**Vortex 5,01e−01 contre nappe de courant 1,82e−05 : facteur 27 500**, pour
deux instabilités de même nature — l'une hydrodynamique, l'autre magnétique.

**La cause n'est pas localisée, et je m'abstiens de la nommer.** Trois fois
dans cette campagne, une reproduction incomplète d'un calcul m'a fait
accuser du code juste : le filtre `> 1e-10` de `C_scale`, un champ d'essai
dont `|B|` s'annulait là où `|Jz|` culmine, et maintenant un champ non
périodique.

**Ce qui est établi** : la localisation spatiale du canal magnétique est
correcte — le maximum tombe là où `|Jz|` culmine. Le canal désigne le bon
endroit, avec la mauvaise amplitude.

C'est la prochaine chose à instruire, et elle porte directement sur
`gamma_mag`, l'un des neuf paramètres à réoptimiser.

---

# Le déséquilibre fluide / magnétique : cause trouvée et corrigée

**Commande.** `pytest tests/mapping/test_coefficient_families_contract.py -q`
(22 tests)

## Trouvée depuis l'intérieur de la fonction

`compute_coefficients` expose désormais `self._stages` — les composantes
**telles qu'elle les calcule**, et non recalculées à côté. C'était
nécessaire : trois fois de suite, une reproduction incomplète m'avait fait
accuser du code juste.

Sur la nappe de courant à N=64 :

| étage | valeur | verdict |
|---|---|---|
| `mic_jz` | 1,541e−01 | **sain** — l'étage de seuil |
| `f_Rm_cell` | 8,346 | **sain** — la porte d'échelle |
| `g_mag` | **0,000** | ← le coupable |

## La cause

`g_rot` compare `Q_OW` à `Q_CRIT = 2,0`. `Q_OW` vient de
`grid._compute_q_criterion(vx, vy, dx=dx)` : il **prend dx**, il est en
unités **physiques**.

`g_mag` comparait `Jz_curl` à `J_CRIT = 1,0`. `Jz_curl` vient de
`curl_z(Bx, By)`, qui ne prend **pas** dx : différence finie en unités de
**grille**. Vérifié — sur `Bx = 1 + 0,8·tanh(3 sin y)`, de dérivée
analytique 2,4, `curl_z` rend **0,2287 = 2,4 × dx**.

Les deux portes topologiques comparaient donc des grandeurs de **deux
systèmes d'unités différents** à des seuils de même ordre nominal. La porte
magnétique était plus dure à franchir d'un facteur exactement **1/dx** :
10,2 à N=64, 20,4 à N=128, **40,7 à N=256**. Elle se dégradait quand la
grille se raffine.

Quatrième membre de la famille dimensionnelle de cette campagne, après la
porte `|B|` de `K_xpoint`, sa normalisation en dx⁴, et le seuil absolu en
1/dx².

## La correction et son effet

`g_mag` reçoit désormais `Jz_curl / dx`.

| champ | `K_plaq` avant | `K_plaq` après |
|---|---|---|
| réseau de vortex | 5,009e−01 | 5,009e−01 |
| **nappe de courant** | **1,816e−05** | **1,148e+00** |
| point X magnétique | 5,092e−06 | 5,437e−01 |
| uniforme (contrôle) | 0 | 0 |

**Le rapport fluide / magnétique passe de 27 500 à 0,44** — soit un rapport magnétique/fluide de **2,29**. *(Les deux sens ont circulé dans mes messages ; c'est bien 0,501 pour le fluide et 1,148 pour le magnétique.)* Les deux canaux
sont désormais du même ordre — la seule chose qu'on puisse exiger de deux
instabilités de même nature.

Sur `harris_tearing` à N=256 : `K_plaq` 2,43e−01 et `K_xpoint` 9,81e−01. Le
canal point X domine encore, mais d'un facteur **4** au lieu de 370.

## Une lecture que la correction invalide

J'avais écrit que les deux canaux ZZZZ étaient orthogonaux « d'un facteur
19 000 » sur le champ de point X. **C'était un artefact du canal magnétique
écrasé.** Le champ d'essai `B = (sin(y−L/2), sin(x−L/2))` porte un point X
**et** du courant — `Jz = cos(x−L/2) − cos(y−L/2)`, non nul. Le canal
magnétique réparé, `K_plaquettes` y répond légitimement.

L'orthogonalité se lit désormais sur le champ qui **sépare** vraiment : le
réseau de vortex, qui n'a aucun nul magnétique, laisse `K_xpoint` à zéro.

## Conséquence sur le périmètre de réoptimisation

`gamma_mag` était **non entraînable** tant que la porte qui le précède
s'annulait : l'optimiseur aurait compensé un facteur 1/dx sans signification
physique, et la valeur trouvée aurait été spécifique à la résolution.
**La correction le rend entraînable.**

`kappa` était suspect pour la même raison — il règle simultanément `g_rot`
et `g_mag`. Les deux portes recevant maintenant des grandeurs physiques, il
redevient un réglage cohérent.

**Les neuf paramètres sont désormais défendables.**

---

# `study/` voit enfin le terme de point X

**Commande.** `pytest tests/study/test_xpoint_reaches_study.py -q` (5 tests)

`build_ising_terms` ne lisait que `H_edges`, `C_edges` et `K_plaquettes`.
La diagonalisation exacte, le recuit simulé et les ablations de `study/`
étaient donc **structurellement aveugles** au terme de point X, que la
campagne d'entraînement active pourtant sur **6/6 scénarios**.
`h3_term_ablation` mettait même `K_xpoint` à zéro sur l'ablation `no_ZZZZ`
en croyant l'ablater : il annulait une clé que `ground_state_mask` ne lisait
jamais.

`qaoa_inputs.py` codait par ailleurs `advanced_anomalies_enabled=False`.

## Ce qui a été fait

`build_ising_terms` ajoute un **second** terme ZZZZ sur la même plaquette,
reproduisant exactement `cost_hamiltonian` — qui empile lui aussi un terme
séparé sur les quatre mêmes qubits, `SparsePauliOp` sommant les doublons.
Le drapeau de `qaoa_inputs` passe à `True`.

## Le test qui compte : les deux chemins coïncident

Sur les **256 états** d'un problème à `dim = 2` (8 qubits), `K_xpoint`
actif, trois graines :

**écart maximal 5,3e−15** entre l'énergie du chemin `study/` et la
diagonale de `create_period_hamiltonian`, le chemin déployé.

Sans cette vérification, la falsification aurait porté sur un autre
hamiltonien que l'entraînement.

## Une erreur de ma part, la quatrième de la même famille

Ma première comparaison donnait un écart de **1,88e+01** et une corrélation
de **1,0e−04** — j'ai cru un instant que les deux chemins divergeaient. La
cause était ma convention de bits : Qiskit est **little-endian**, le bit de
poids faible correspond au dernier qubit. Avec l'ordre inversé, l'écart
tombe à 5,3e−15.

Quatrième fois de cette campagne qu'une reproduction incorrecte accuse du
code juste — après le filtre `> 1e-10` de `C_scale`, le champ dont `|B|`
s'annulait là où `|Jz|` culmine, et le champ non périodique. Le commentaire
reste dans le test pour la cinquième.

## Écart signalé, non corrigé

`cost_hamiltonian` filtre les coefficients à **1e−6**, `build_ising_terms` à
**1e−12**. Les deux chemins coïncident sur les jeux testés, mais un
coefficient entre ces deux seuils serait vu par `study/` et ignoré par le
circuit déployé. À trancher avant la réoptimisation.

---

# Les seuils tranchés : `study/` s'aligne sur le circuit

**Commande.** `pytest tests/study/test_xpoint_reaches_study.py -q` (5 tests)

`cost_hamiltonian` filtre les coefficients à `COEFF_MIN = 1e-6` ;
`build_ising_terms` filtrait à `1e-12`. **`study/` diagonalisait donc un
hamiltonien plus fourni que celui que le circuit résout.**

## L'écart, mesuré

Coefficients tombant entre les deux seuils — vus par `study/`, ignorés par
le circuit :

| scénario | dim | total | entre les deux seuils |
|---|---|---|---|
| `harris_tearing` | 2 | 16 | **4 (25 %)** |
| `harris_tearing` | 4 | 64 | **16 (25 %)** |
| `mhd_rotor` | 4 | 64 | 1 |
| `orszag_tang` | 2, 4 | 16, 64 | 0 |

**Un quart des termes à `dim = 2` sur le scénario de reconnexion** — celui
même où D-53 a été mesuré.

## La décision

`study/` s'aligne sur `1e-6`. **La falsification doit décrire ce que le
circuit exécute**, pas un hamiltonien qui lui est étranger. Les cinq seuils
de `build_ising_terms` (biais Z, couplages ZZ, plaquettes, point X) passent
par une constante unique `COEFF_MIN`, importable et documentée.

Aligner dans l'autre sens — descendre le circuit à `1e-12` — aurait changé
le comportement déployé, donc la science, pour faire coïncider un outil
d'analyse avec elle. C'est l'inverse de l'ordre correct.

## Ce que cela implique pour D-53

Les artefacts `dim = 2` et `dim = 3` ont été produits avec **quatre**
écarts cumulés entre `study/` et le circuit :

1. `K_xpoint` absent de `build_ising_terms` ;
2. `advanced_anomalies_enabled=False` codé en dur dans `qaoa_inputs` ;
3. `g_mag` écrasé d'un facteur `1/dx` ;
4. un quart des termes retenus que le circuit rejette.

La relance est donc nécessaire avant toute lecture de D-45, D-47 ou D-53.

---

# Relance `dim = 3` sur l'hamiltonien corrigé : D-53 tient

**Commande.**
`python study/h0_selection/h0_optimiser_equivalence.py --dim 3 --N 96 --re 400`
→ `results/h0_optimiser_equivalence_N96_dim3_hamiltonien_corrige.npz`

Relancé après les quatre corrections : `K_xpoint` branché dans
`build_ising_terms`, drapeau des anomalies à `True`, `g_mag` en unités
physiques, seuil aligné sur `COEFF_MIN = 1e-6`.

## Mon hypothèse est réfutée

J'avais avancé que D-45, D-47 et D-53 pouvaient être trois symptômes d'une
cause unique — un hamiltonien vide. **La mesure dit non.**

| | ancien | nouveau |
|---|---|---|
| `\|E\|` max | 4,154e+01 | 4,146e+01 |
| `E_gap` max | 1,887e+00 | 1,886e+00 |
| `E_gap` médian | 1,792e−01 | **2,836e−02** |

L'hamiltonien **n'était pas vide** à `dim = 3` : les énergies sont
quasi identiques avant et après. Les corrections ont resserré le paysage —
l'écart médian au fondamental chute d'un facteur 6 — sans changer son
échelle.

## Le verdict tient

| solveur | hit ancien | hit nouveau |
|---|---|---|
| `exhaustive` (certifié) | 1,000 | 1,000 |
| `greedy` | 0,844 | 0,833 |
| `sa_warm` | 0,750 | 0,833 |
| `classical_init` | **0,500** | **0,500** |
| `qaoa_p1` | 0,156 | **0,083** |
| `qaoa_p2` | 0,156 | 0,083 |
| `qaoa_p3` | 0,125 | 0,083 |

**Le QAOA reste très loin sous sa propre initialisation classique.** Le
critère pré-enregistré du module lève toujours : *« des solveurs
déterministes n'atteignent plus l'optimum certifié […] H0 (l'échec vient de
l'optimiseur) redevient plausible »*.

**Réserve sur l'ampleur.** 0,156 → 0,083 est *dans* la dispersion
run-to-run du bras QAOA (1,79e−1 à 3,61e−1, mesurée par ce dépôt). Je ne
prétends donc pas que le QAOA a empiré : **le classement est identique**, et
c'est le classement qui fait foi ici. `qaoa_shots_p3` valait `nan` dans
l'ancien artefact — il n'y avait pas été exécuté.

## Ce que ça vaut

D-53 portait sur un hamiltonien auquel il manquait le terme de point X,
dont le canal magnétique était écrasé d'un facteur `1/dx`, et dont un quart
des termes étaient étrangers au circuit. **Il porte désormais sur un
hamiltonien qui inclut les quatre familles, dimensionnellement cohérent, et
identique à celui que le circuit exécute à 5,3e−15 près.**

Le verdict n'en est pas affaibli — il en sort **beaucoup plus difficile à
écarter**.

## L'artefact de l'agent a été préservé

La relance écrivait sur `h0_optimiser_equivalence_N96_dim3.npz`, l'artefact
qui porte D-53. Il est **restauré** ; la nouvelle mesure vit sous
`..._hamiltonien_corrige.npz`. Les deux doivent coexister : ils mesurent
deux hamiltoniens différents, et c'est leur comparaison qui a de la valeur.

---

# Le défaut principal n'est PAS l'optimiseur — c'est l'hamiltonien

**Artefact.** `results/h0_optimiser_equivalence_N96_dim3_hamiltonien_corrige.npz`

La colonne que je n'avais pas exploitée : le **F1 contre la vérité terrain**
du raffinement. Elle renverse la lecture.

| solveur | trouve l'optimum | écart d'énergie | **F1 vs vérité** |
|---|---|---|---|
| `exhaustive` (certifié) | 1,000 | 0,0000 | **0,386** |
| `sa_warm` | 0,833 | 0,0044 | **0,378** |
| `sa` | 0,500 | 0,0101 | 0,393 |
| `greedy` | 0,833 | 0,0131 | 0,424 |
| `classical_init` | 0,500 | 0,6345 | 0,460 |
| `qaoa_p1` | 0,083 | 1,0416 | **0,481** |

**Corrélation de rang entre l'écart à l'optimum et le F1 : ρ = +0,970,
p = 0,0001, sur 8 solveurs.**

## Ce que ça veut dire

Un ρ **positif** signifie : **plus on s'écarte de l'optimum de H, meilleure
est la décision de raffinement.**

- Le solveur qui résout H **parfaitement** (`exhaustive`, écart 0) obtient
  un F1 de **0,386** — parmi les pires.
- Le solveur qui le résout **le plus mal** (`qaoa_p1`, écart 1,04) obtient
  le **meilleur** F1, 0,481.
- Le recuit simulé warm-start, presque parfait sur H (écart 0,0044), est le
  **pire** de tous : F1 = 0,378.

**L'état fondamental de cet hamiltonien n'est pas la bonne décision AMR.**
Le résoudre exactement rend la décision *moins* bonne que ne pas le résoudre
du tout.

## Conséquence sur la lecture de H0

Le QAOA est mauvais optimiseur — c'est établi, 0,083 contre 0,500 pour sa
propre initialisation. Mais **ce n'est pas le défaut dominant**, et le
corriger irait dans la mauvaise direction :

**un meilleur algorithme quantique trouverait mieux l'optimum, donc
produirait une plus mauvaise décision.** `sa_warm` le démontre — il optimise
presque parfaitement et décide le plus mal de tous.

Le QAOA obtient son F1 supérieur **par accident** : il dévie tellement de
l'optimum qu'il reste près de son initialisation classique, qui est
meilleure que l'optimum.

## Ce qui reste à conclure, et ce qui ne l'est pas

**Établi ici** : à `dim = 3`, sur un hamiltonien complet et dimensionnellement
cohérent, mieux résoudre H dégrade la décision (ρ = +0,970, p = 1e−4).

**Non établi** : que ce soit vrai à toute taille, ou pour toute famille de
coefficients. C'est **12 instantanés, une résolution, un jeu
d'hyperparamètres non réoptimisés**. Le F1 absolu reste bas partout
(0,378–0,481), ce qui limite ce qu'on peut lire dans les écarts.

**Ce que ça change pour la réoptimisation** : elle devient *le* test
décisif. Si des hyperparamètres existent pour lesquels l'optimum de H
coïncide avec la bonne décision, ρ doit changer de signe. Sinon, c'est la
forme de l'hamiltonien qu'il faut revoir, pas ses réglages.

---

# Contrôle avant vol des coefficients

**Commande.** `python study/common/preflight_coefficients.py`
(code de sortie non nul si un contrôle échoue)

Une campagne coûte ~224 h CPU. Ce module vérifie en quelques minutes que
les coefficients font leur travail **avant** qu'on les règle — parce qu'un
coefficient qui ne détecte pas ne se corrige pas par un réglage.

| contrôle | mesure | référence |
|---|---|---|
| **spécificité** | vortex → `K_plaq` 0,501, `K_xpoint` 0 ; uniforme → tout à 0 | — |
| **équilibre** | magnétique / fluide = **2,29** | dans [0,1 ; 10] |
| **vivant** | à N=256 : `K_plaq` 0,243, `K_xpoint` 0,981 | non nuls |
| **pertinence** | ρ(coefficient, erreur réelle) = **0,798** | > 0,6 |
| **coïncidence** | `study/` vs circuit : **5,33e−15** | < 1e−9 |

**Verdict : les cinq passent.** Les coefficients font leur travail.

## Deux références corrigées en montant ce contrôle

**Le rapport d'équilibre.** J'ai écrit « 0,44 » ; c'est le rapport
**fluide/magnétique**. Le contrôle calcule magnétique/fluide, soit **2,29**.
Même mesure — 0,501 pour le fluide, 1,148 pour le magnétique — libellé
inversé d'un message à l'autre. Corrigé ici et dans l'entrée précédente.

**La corrélation avec l'erreur réelle.** Elle vaut **0,798**, pas 0,897.
Le 0,897 datait d'**avant** l'harmonisation des portes `g` : réveiller le
canal magnétique a légèrement abaissé la corrélation de `K_plaquettes`
seul. Remesuré, non ajusté — les deux valeurs sont consignées.

---

# ρ(E_gap, F1) : le critère de décision de la campagne — option A

**Commande.** `python study/common/rho_gap_f1.py results/h0_*.npz`

Post-traitement : **ne tourne pas dans la boucle d'entraînement**. La
campagne n'en porte donc aucun risque.

| artefact | ρ | p | verdict |
|---|---|---|---|
| `..._dim3_hamiltonien_corrige` | **+0,870** | 0,0023 | l'optimum de H n'est pas la bonne décision |
| `..._N256_dim2` | **−1,000** | 0,0000 | l'optimum de H est la bonne décision |

## Deux corrections à ce que j'ai annoncé

**ρ vaut +0,870, pas +0,970.** Mon calcul manuel excluait `qaoa_shots_p3` ;
le module prend les 9 solveurs. Le signe et la conclusion sont inchangés,
la valeur non.

**Le signe s'inverse à `dim = 2`.** Je présentais « mieux résoudre H dégrade
la décision » comme un fait général — il ne l'est pas.

Mais `dim = 2` est **dégénéré**, et c'est D-45/D-47 : `classical_init` y a un
écart de **0,0000** à l'optimum, c'est-à-dire que la règle classique *est*
déjà l'optimum. Tous les solveurs qui l'atteignent ont le même masque, et
les F1 tiennent dans 0,367–0,389. Un ρ = −1 sur un problème où il n'y a rien
à départager ne prouve rien.

**La lecture honnête** : ρ = +0,870 vaut à `dim = 3`, la seule taille à la
fois certifiée et non dégénérée. À `dim = 2`, la question n'a pas de sens.

## Le critère pré-enregistré

- **ρ passe négatif à `dim = 3`** → il existe des hyperparamètres pour
  lesquels l'optimum de H est la bonne décision. Le réglage suffisait.
- **ρ reste positif** → c'est la **forme** de l'hamiltonien qu'il faut
  revoir. Aucune campagne Optuna ne trouvera cela.

À enregistrer avant de lancer, pas après.

## D-172 — le module lui-même citait encore la valeur rétractée

**Où ça bloquait.** `study/common/rho_gap_f1.py` — le script qui calcule
ce critère pré-enregistré — portait dans son propre docstring et dans la
bannière imprimée par `main()` : *« MESURE DE REFERENCE, avant campagne,
sur `h0_optimiser_equivalence_N96_dim3_hamiltonien_corrige.npz` :
rho = +0.970, p = 0.0001, 8 solveurs »*. C'est précisément le nombre que
ce même fichier retracte deux paragraphes plus haut (*« ρ vaut +0,870, pas
+0,970. Mon calcul manuel excluait `qaoa_shots_p3` »*) — la correction
avait atteint la prose de `RESULTS.md`, jamais le module qui sert de
critère à la campagne.

**Comment on est tombé dessus.** Question 4 de `VIGIL.md` : deux endroits
qui devraient porter le même nombre le portent-ils encore ? Ici, un
troisième — le module lui-même — ne rejouait pas sa propre mesure.

**Mesuré, avant.**

```
python study/common/rho_gap_f1.py results/h0_optimiser_equivalence_N96_dim3_hamiltonien_corrige.npz
```

bannière : `rho = +0.970 (p = 1e-4)` ; calcul affiché juste en dessous :
`rho = +0.870   p = 0.0023   (9 solveurs)` — les deux nombres coexistaient
dans la même sortie, sans qu'aucun des deux ne renvoie à l'autre.

**Correction, minimale.** Le docstring et la chaîne imprimée par `main()`
portent désormais `+0.870 / p = 0.0023 / 9 solveurs`, avec une note datée
(D-172) expliquant pourquoi l'ancien nombre y était. Le calcul lui-même
(`rho_gap_f1()`) n'a jamais été faux — seule la référence citée en tête
l'était.

**Mesuré, après.** Même commande : bannière et calcul rendent tous deux
`+0.870`. Aucun nombre publié ne bouge — `RESULTS.md` portait déjà la
valeur correcte ; c'est le module qui la rejoint.

**Tests, dont un qui épingle l'ancien comportement.**
`pytest tests/study/test_rho_gap_f1_reference.py -q` → 2 passed. Rejoué
contre `study/common/rho_gap_f1.py` d'avant cette correction (`git stash`) :
1 failed sur la bannière — l'autre passe déjà, parce que le calcul n'a
jamais été le défaut, seule sa légende l'était.

Vérifier : `pytest tests/study/test_rho_gap_f1_reference.py -q`


# D-117 — le percentile du critère relatif devient entraînable

`min(absolu, percentile)` est ce qui rend `K_plaquettes` et `K_xpoint`
non nuls à la résolution d'entraînement. Le seuil de maille est **absolu**
et croît en `1/dx²` : à N=256 aucune cellule ne l'atteignait, et le terme
à quatre corps valait zéro sur les **quatre** scénarios. Le percentile qui
prend le relais valait `90.0`, écrit en dur — la dernière constante du
chemin de décision que rien ne justifiait de fixer à la main.

## Ce que le paramètre pilote, mesuré

Réseau de tourbillons périodique, N=64, Re=Rm=800, hyperparamètres
déployés, à travers la **vraie** `compute_coefficients` :

| `relative_percentile` | `K_plaquettes` non nuls | `\|K\|` max |
|---|---|---|
| 50 | 2040 / 4096 | 5,879e+00 |
| 90 | 404 / 4096 | 4,254e-01 |
| 99 | 32 / 4096 | 1,726e-02 |

Deux ordres de grandeur d'amplitude entre les bornes de l'espace de
recherche : ce n'est pas un réglage cosmétique, c'est le nombre de patchs
que l'AMR ouvrira.

**Bornes.** 50 = la médiane, la moitié des cellules passent le seuil, ce
qui sature l'AMR. 99 = une cellule sur cent, sous le grain d'un patch
(`min_patch_size = 6`, soit 36 cellules) : au-delà, le critère relatif ne
désignerait plus assez de cellules pour former un patch et redeviendrait
le seuil absolu par un autre chemin.

**L'invariant qui rend le paramètre sûr** : dès qu'une cellule franchit le
critère physique, le comportement d'origine est conservé à l'identique,
quelle que soit la valeur entraînée. Le percentile ne *remplace* jamais la
physique, il ne prend le relais que là où elle est muette.

## Ce qu'il coûte — et pourquoi le balayage à quatre points ne prouve rien

Avant de louer des cœurs il fallait savoir si la borne basse est chère :
un percentile bas désigne plus de cellules, donc plus de patchs, donc
plus de circuits. Mesuré à la configuration **réelle** (N=256, Harris
tearing, par les fonctions du module d'entraînement lui-même) :

| `relative_percentile` | mur | perte |
|---|---|---|
| 50 | 517,8 s | 0,236030 |
| 75 | 545,8 s | 0,245103 |
| 90 | 553,2 s | 0,245455 |
| 99 | 605,5 s | 0,273969 |

**La direction supposée était fausse** : c'est le percentile *haut* qui
coûte le plus. Mécanisme plausible — un percentile haut éparpille
quelques cellules isolées, qui pavent en beaucoup de petits patchs, là où
un percentile bas désigne des régions contiguës qui fusionnent en peu de
grands patchs.

Mais ce tableau se lit avec son **plancher de bruit**, sans quoi il ne
vaut rien. Quatre répétitions au **même** percentile (90) :

    pertes  0,253605  0,256327  0,247726  0,276318
    moyenne 0,258494   écart-type 0,010750   étendue 0,028592
    mur     moyenne 580,0 s      étendue 47,7 s

Le bras QAOA est non déterministe, et cela suffit à effacer l'essentiel
du balayage :

- **la tendance sur la perte n'est pas établie.** p50 contre p90 vaut
  0,0095 d'écart, soit ~0,6 σ d'une différence entre deux tirages
  uniques : dans le bruit. Seul p50 contre p99 (0,038) atteint ~2,5 σ —
  suggestif, pas concluant à n=1 ;
- **la tendance sur le coût non plus.** L'étendue mur *à percentile
  fixe* vaut 47,7 s contre 87,7 s *entre* percentiles ;
- le p90 tiré une fois dans le balayage donne 0,2455, quand quatre
  répétitions de la **même** configuration donnent 0,2585 : 1,2 σ entre
  une configuration et elle-même.

**Ce qui survit**, et c'est la seule chose dont la campagne avait besoin :
l'effet du neuvième paramètre sur le coût vaut **au plus ~17 %**, et
probablement moins. Il ne déplace pas l'estimation de la réoptimisation.

**Ce qui ne survit pas** : l'idée que l'optimum se trouverait à la borne
basse, donc que la borne choisirait à la place des données. Rien ne le
montre. Une comparaison à un tirage n'est pas lisible sur ce bras — c'est
précisément ce qu'une campagne Optuna de 180+ essais moyenne, et ce
qu'un balayage à quatre points ne peut pas faire.

## Le câblage, qui est l'essentiel

L'ajouter à `SEARCH_SPACE` sans le câbler aurait été **D-31 à l'identique**
— un paramètre optimisé que rien ne lit, payé au prix plein de la campagne.
C'est d'ailleurs ce qui a failli arriver : la première insertion a atterri
dans le bloc **mort** de `pipeline.py` (un littéral triple-quote qui
contient un `hp.get` par paramètre). Le nom n'était alors défini nulle
part, et `PhysicalMapper(relative_percentile=…)` aurait levé `NameError`
au premier essai.

Deux gardes ferment ce trou :

1. un balayage AST qui exige un `hp.get` **vivant** pour *chaque* nom de
   `SEARCH_SPACE` — l'AST ne descend pas dans les littéraux de chaîne,
   donc le bloc mort ne peut pas satisfaire le test ;
2. une mutation mesurée à travers `compute_coefficients`, qui échoue si
   `__init__` rangeait la valeur dans un attribut que personne ne lit.

`_effective_crit` passe de `@classmethod` à méthode d'instance ; `None`
retient la constante de classe, donc le comportement d'avant. La graine de
phase 1 vaut `90.0`, pour que le premier essai reproduise exactement
l'état actuel et que tout écart mesuré ensuite vienne de l'exploration.

Le périmètre passe de **8 à 9**. `PERIMETRE_8` devient `PERIMETRE_9` ; les
neuf tests tombés sur cette ligne sont le comportement voulu — c'est la
raison d'être de cette constante.

# D-116 — deux lanceurs qui ne lançaient rien

Trouvé en exécutant la recette de `CLAUDE.md`.

`scripts/run_study_v3.sh` : les **quinze** chemins qu'il nommait pointaient
dans le vide (`study/v3/`, `study/results/`, `tests/v3/`,
`study/phase11_upper_bound.py`) — le script datait de deux réorganisations
en arrière. En prime `ROOT_DIR` remontait de deux crans (`../..`) alors que
le script est descendu dans `scripts/`, un seul cran sous la racine : il
désignait le **parent du dépôt**. Sa seule étape qui « passait » était un
`pytest` sur `tests/v3/`, vide : *no tests ran*, code de retour **0**.

`scripts/generate_figures_v1.sh` portait la version la plus dangereuse.
`FIGURES_CODE_DIR` pointait sur un `figures_code/` disparu — les 17
scripts vivent dans `figures/v1_legacy/` — et la boucle dit :

```bash
if [[ ! -f "$script_path" ]]; then log "  SKIP: $script"; continue; fi
```

Les 17 scripts tombaient donc dans la branche `SKIP`, le lanceur annonçait
**`Succeeded: 0  Failed: 0`** et rendait **0**. Une campagne de figures
verte qui ne produit aucune figure. Un compteur `SKIPPED` et une clause
d'échec sur `SUCCEEDED == 0` ferment le trou.

Sa sortie n'écrase plus `best_hyperparams.json` — **entrée gelée** de
l'étude, le seul dossier qu'aucune commande ne reproduit — mais écrit
`best_hyperparams.regenerated.json` à côté. Un lanceur de figures qui
réécrit les hyperparamètres changerait la science en produisant une image.

`CLAUDE.md` portait trois erreurs de fait, corrigées : le module à
réutiliser (`phase11_upper_bound.py` n'existe pas ; les cinq fonctions
sont dans `h2b_ceiling_random_split.py`), l'ordinal du test de
non-régression (**quatrième**, pas troisième) et son état attendu
(**164 OK / 16 DIFF / 0 MISSING**, pas `0 DIFF`).

C'est le piège n° 3 de `COUVERTURE`, « balayage vide », appliqué aux
**lanceurs** plutôt qu'aux tests : un lanceur qui ne lance rien ressemble
exactement à un lanceur qui a tout lancé.


# D-68 — la figure AMR est transposée : décision prise, clos

`plot_amr_state` est la **seule** fonction de `src/visual.py` et de
`src/help_visual.py` qui s'exécute en production : `pipeline.py` l'appelle
quatre fois par pas de verrouillage et sauve un PNG à chaque fois.

**Deux défauts, un faux et un incohérent.**

Le premier était faux : les deux étiquettes nommaient l'axe de l'autre.
`Jz` est indexé `[X, Y]` (`grid.py` : `AXIS_X = 0`, `AXIS_Y = 1`), et
`imshow` place l'axe 0 en **vertical** — donc l'axe horizontal portait Y
alors qu'il annonçait « Grid X ». Une structure posée en **X=10, Y=40** se
relisait **« X=40, Y=10 »**.

Le second était une incohérence de dépôt : `plot_amr_state` était le
**seul des trois traceurs** à mettre Y en horizontal. `plot_recursive_state`
(même fichier, ligne ~171) trace `state['Jz'].T`, et
`help_visual.plot_field` trace `grid.X.T` en étiquetant « X » l'axe
horizontal.

**Décision de USER : transposer**, ce qui clôt les deux d'un coup.
L'objection qui avait fait suspendre la correction — « cela change la
géométrie de PNG déjà publiés » — ne tient plus : toutes les figures sont
regénérées après la campagne, donc la cohérence est gratuite maintenant et
coûteuse plus tard.

| | avant | après |
|---|---|---|
| structure posée en X=10, Y=40 | lue « X=40, Y=10 » | lue **« X=10, Y=40 »** |
| axe horizontal | Y | **X**, comme les deux autres traceurs |

Corrigé aussi, parce que c'est la cause de l'invisibilité du défaut : les
variables locales s'appelaient `ys/ye` pour l'indice **i** et `xs/xe` pour
**j** — l'inverse de ce qu'elles contenaient. Renommées `i_start/i_end`,
`j_start/j_end`. Un défaut d'axes se cache derrière un vocabulaire d'axes
inversé.

**Ce qui n'a pas bougé** : champ et cadres restent cohérents entre eux —
transposer l'image sans les cadres, ou l'inverse, fait tomber
`test_le_cadre_encadre_la_structure_qu_il_designe`.

**Seuil remesuré, pas supprimé.** Le test qui gardait la frontière de
décision (`…n_a_pas_ete_transposee`) exigeait l'absence de transposition ;
il dit désormais l'inverse et porte la mesure avant/après. C'est la règle
« un seuil périmé se remesure » appliquée à une décision, pas à un
changement de code subi.

Vérifier : `pytest tests/pipeline/test_amr_figure_axes.py -q` → **6 passed**.


# D-53 — la seule taille certifiée non dégénérée contredit la réfutation de H0

**Le résultat le plus fort du dépôt, et il n'était écrit nulle part.**
`dim3` n'apparaissait pas une seule fois dans ce fichier alors que trois
artefacts `dim = 3` vivent dans `results/`.

## Ce qui était publié, et sur quoi

`CLAUDE.md` portait `h0_selection … → RÉFUTÉ` sans qualificatif, et T11
concluait *« Pre-registered rule fires: quantum optimisation is not the
source of any gain »*. Les deux reposent **entièrement sur `dim = 2`**,
8 qubits — la taille dont ce fichier note lui-même que *« the optimum
itself is uniform, so the solvers agree on a trivial problem »*.

C'est D-45 et D-47 : à `dim = 2` l'état fondamental exact est le
prédicteur constant « tout raffiner » sur **40 instantanés sur 40**. Tous
les solveurs atteignent donc l'optimum, parce qu'il n'y a rien à
départager. **Réfuter H0 là-dessus, c'est la réfuter sur un problème
vide.**

## Ce que dit `dim = 3`

`results/h0_optimiser_equivalence_N96_dim3.npz` — **18 qubits, donc
certifié** (l'optimum y est énuméré exactement), 4 scénarios canoniques,
32 instantanés :

| solveur | hit optimum | mask match | exigé |
|---|---|---|---|
| exhaustive (certifié) | 1,000 | 1,000 | — |
| `classical_init` (règle classique seule) | **0,500** | **0,500** | exclu du critère |
| greedy | 0,844 | 0,844 | 1,000 |
| sa / sa_warm | 0,594 / 0,750 | 0,938 / 0,875 | rapporté |
| `qaoa_p1` … `qaoa_p6` | **0,156 → 0,062** | **0,156 → 0,219** | 1,000 |

**Le QAOA tombe plus loin de l'optimum certifié que la règle classique
dont il part** : 0,156–0,219 contre 0,500. Ce n'est pas un écart de
tirage — la dispersion mesurée du bras vaut 1,79e−1 à 3,61e−1, et
0,062 contre 1,000 est hors de cette échelle.

**Ce n'est pas un problème de budget.** C'est l'objection que
`--scale-kopt` existe pour lever : sur l'artefact qui l'utilise
(harris_tearing, 6 instantanés), le QAOA passe à **0,000** sur les quatre
profondeurs, `greedy` restant à **1,000**. Donner plus d'itérations ne
rapproche pas le QAOA de l'optimum, cela l'en éloigne.

**Le critère du module lui-même tranche**, rejoué sur les artefacts :

| artefact | verdict de `check_expected_behaviour` |
|---|---|
| `..._N256_dim2.npz` | `[ACCEPTANCE] … H0 réfutée à cette taille` |
| `..._N96_dim3.npz` | **lève** — *« H0 redevient plausible »* |
| `..._N96_dim3_scalekopt.npz` | **lève** — `{qaoa_p1: 0.0, … qaoa_p6: 0.0}` |

## La lecture juste : H0a et H0b se séparent

C'est le point qui compte, et il ne se voit qu'en combinant D-53 avec la
mesure de ρ :

| | question | verdict |
|---|---|---|
| **H0a** | l'optimiseur atteint-il l'optimum de son propre hamiltonien ? | **NON** — 0,062–0,156 contre 1,000 exigé, à la seule taille certifiée non dégénérée |
| **H0b** | mieux l'atteindre améliorerait-il la tâche ? | **NON** — ρ(E_gap, F1) = **+0,870** sur 9 solveurs à `dim = 3` : mieux résoudre H **dégrade** la décision AMR |

Les deux ensemble disent quelque chose de plus fort que chacun seul :
**l'optimiseur échoue vraiment, et le réparer ne servirait à rien.** Le
QAOA n'atteint pas le fondamental de son hamiltonien ; et les solveurs qui
l'atteignent prennent de plus mauvaises décisions de raffinement.

C'est exactement ce que `PLAN_PREPRINT.md` §7 désignait comme l'argument
de fermeture : *« H0b ferme l'approche plus directement que H3 — c'est la
valeur de l'optimisation qui est attaquée, précisément ce qu'on paierait
en qubits. »*

## Ce qu'il faut corriger ailleurs

`h0_selection → RÉFUTÉ` est **faux sans qualificatif**. La formulation
exacte est : **réfutée à `dim = 2`, où le problème est dégénéré ; redevient
plausible à `dim = 3`, la seule taille à la fois certifiée et non
dégénérée jamais exécutée.** Corrigé dans `CLAUDE.md`.

## Une réserve, écrite parce qu'elle est réelle

D-122 : les deux artefacts `--scale-kopt` sont **deux exécutions de la même
condition**, pas deux conditions — le second porte `--zero-psi` sans
`--with-psi`, donc psi valait zéro exactement avant l'ablation. Mesuré :
même `git_hash`, même graine, et les 5 solveurs déterministes **bit à bit
identiques sur 30/30 lignes**.

Le QAOA à 0,000 **tient** : il est mesuré deux fois sur deux tirages
indépendants, c'est une réplication et elle renforce le point. Ce qui
tombe est la lecture implicite du nom : **l'ablation psi de ce module n'a
jamais été exécutée**, et rien ici ne dit ce que psi apporte.

Vérifier : `pytest tests/study/test_h0_certified_dim3_contradicts_criterion.py`
→ **6 passed**. Le test reconstruit `(summary, solvers, diag_flags)` depuis
chacun des trois artefacts, dans la forme exacte que `main()` passe au
critère, et appelle `check_expected_behaviour` : à `dim = 2` il imprime
`[ACCEPTANCE]`, à `dim = 3` il lève avec le dictionnaire des solveurs sous
le seuil.

*(D-140 : cette ligne invoquait `h0_optimiser_equivalence.py` avec une
option `--check` suivie de l'artefact. Cette option n'est pas déclarée —
le script n'appelle `check_expected_behaviour` qu'au terme d'une campagne
complète. La commande publiée pour vérifier le résultat le plus fort du
dépôt rendait `error: unrecognized arguments` et sortait en **2**. Le
chemin existait, donc le garde de D-71 la laissait passer.)*


# D-51 — clos : `study/` voit désormais le terme ZZZZ de point X

**Ce qui bloquait.** Tout `study/` codait `advanced_anomalies_enabled =
False` alors que la campagne d'entraînement l'active sur **6 scénarios sur
6**. Le terme ZZZZ de point X n'entrait donc dans **aucune** mesure de
falsification — et `beta_xpoint`, que D-22 range parmi les paramètres à
réoptimiser, était un hyperparamètre qu'aucune mesure de `study/` ne
pouvait voir.

**Fermé par deux changements**, tous deux sur la branche vive :

1. `study/common/qaoa_inputs.py:350` passe `advanced_anomalies_enabled=True`
   — `study/` construit désormais le même hamiltonien que le déploiement.
2. `study/common/ising_terms_and_annealing.py` fait consommer `K_xpoint` à
   `build_ising_terms`, qui ne lisait que `H_edges`, `C_edges` et
   `K_plaquettes`. Le terme existait dans les coefficients et n'atteignait
   pas le circuit.

**Vérification de coïncidence** : sur les 256 états d'un problème
`dim = 2`, l'écart entre l'énergie du chemin `study/` et la diagonale de
`create_period_hamiltonian` vaut **5,33e−15**. Les deux chemins décrivent
le même opérateur.

**Conséquence à ne pas oublier.** Ce changement fait tomber trois tests qui
épinglaient l'absence du terme
(`test_xpoint_term_absent_from_study.py`, `test_t13_control_is_not_vacuous.py`).
Ce sont des **seuils périmés** : le code a légitimement changé sous eux.
Leur mise à jour exige de **rejouer phase 4, T13 et T26** — c'est une
campagne, pas une passe de relecture. Ils restent rouges jusque-là, et
c'est voulu.

Vérifier : `python study/common/preflight_coefficients.py` → 5/5, dont
« coïncidence — study/ et le circuit rendent la même énergie ».


# D-91 — clos : la « vérité terrain » du rotor était une erreur RELATIVE

**Ce qui bloquait**, et depuis une note déjà publiée (D-10) : la sélection
dite *« ground truth »* de `compare_rotor_budget.py` rendait une erreur L2
de **0,3079** — pire que l'absence d'AMR (**0,3074**) — quand classique et
Q-HAS obtenaient **0,0208**. Une vérité terrain battue par l'absence de
raffinement n'est pas une vérité terrain. La note constatait l'anomalie
sans en donner la cause.

**La cause.** `compute_block_errors` normalisait **chaque bloc par sa
propre amplitude** :

```python
diff = sqrt(mean((dns_block - coarse_block)**2))
ref  = sqrt(mean(dns_block**2)) + 1e-10      # <- LOCAL, et seul plancher
total_err += diff / ref
```

Seul le dénominateur porte un plancher. Deux blocs partageant le **même
écart absolu** reçoivent donc des scores dans le rapport inverse de leurs
amplitudes — le bruit de fond bat la vraie structure. Mesuré sur un champ
qui sépare (écart absolu identique au bit près) :

| | bloc de bruit (DNS ~ 1e−6) | bloc de structure (DNS = 10,0) |
|---|---|---|
| avant | **2,000e−01** | 2,000e−08 |
| après | 4,000e−08 | 4,000e−08 |

Le bruit dominait d'un facteur **1,0e+07** à écart identique.

**La correction.** Ce que la fonction promet — *« which blocks truly need
refinement »* — se lit contre la quantité que la campagne minimise :
l'erreur L2 **globale**. Le bloc à raffiner est celui qui **contribue** le
plus à cette erreur, pas celui dont l'erreur relative à son propre contenu
est la plus grande. La normalisation devient donc **globale par champ**,
ce qui conserve la seule raison d'être d'une normalisation ici — rendre
`vx` et `Jz` comparables malgré des amplitudes différentes — sans donner
de prime à un bloc vide.

**Remesure, configuration par défaut (N=96, 3×3 blocs, Re=Rm=800) :**

| méthode | L2 | vs. grille grossière | accord avec la vérité terrain |
|---|---|---|---|
| sans AMR | 0,307359 | référence | — |
| **vérité terrain** | **0,020786** | **+93,2 %** | — |
| classique | 0,020786 | +93,2 % | **3/3 blocs** |
| Q-HAS | 0,020786 | +93,2 % | **3/3 blocs** |

L'anomalie disparaît : la vérité terrain corrigée sélectionne exactement
les mêmes blocs que les deux bras, et atteint le même L2. **Les trois
nombres de la note D-10 sont remplacés** — 0,3079 était un artefact de
métrique, pas un résultat.

**Ce que ce banc ne dit plus.** Les trois sélections coïncidant à 3/3, ce
scénario **ne départage plus les deux bras** : « Classical wins by 0.0 % ».
Il valide la chaîne (le raffinement des bons blocs vaut +93 %), il
n'arbitre pas entre classique et quantique. Ne pas lui faire dire l'un
pour l'autre.

Le test qui épinglait le défaut est **remesuré, pas supprimé** : il affirme
désormais l'égalité à écart absolu égal, et un second vérifie la promesse —
à erreur *relative* égale, c'est le bloc de forte amplitude qui doit être
sélectionné.

Vérifier : `python src/compare_rotor_budget.py` puis
`pytest tests/pipeline/test_compare_rotor_budget.py -q` → **14 passed**.

---

# D-59 — corrigé AVANT la campagne : le lien ZZ dupliqué à `dim = 2`

**Le défaut.** L'Hamiltonien est périodique : chaque cellule émet un lien
ZZ vers sa voisine. À `dim >= 3` cela fait `dim` liens distincts par
direction. À **`dim = 2`** l'anneau dégénère — `(i,0)->(i,1)` et
`(i,1)->(i,0 mod 2)` relient la **même paire de qubits** — et les deux
itérations ajoutaient chacune une entrée au lieu d'être fusionnées.

Les coefficients étant symétriques par construction
(`C_edges[0][i,0] == C_edges[0][i,1]` au bit près), le couplage shear était
appliqué **deux fois** : poids effectif **×2**. `K_plaquettes` n'a pas ce
défaut. Repéré sur un décompte de termes affichant `"IIIIIIZZ"` deux fois
avec exactement `-2.4290271580758453` des deux côtés.

**Corrigé aux DEUX sites**, qui doivent coïncider :
`src/VQA/cost_hamiltonian.create_period_hamiltonian` (QAOA / diagonalisation)
et `study/common/ising_terms_and_annealing.build_ising_terms` (SA /
exhaustif). Déduplication par paire non ordonnée de qubits.

**Portée de la correction**, coefficients aléatoires (donc ZZ vivant) :

| | ZZ avant → après | opérateur identique |
|---|---|---|
| dim = 2 | 8 → **4** | **NON** (max\|ΔH\| = 3,285) |
| dim = 3 | 18 → 18 | OUI |
| dim = 4 | 32 → 32 | OUI |
| dim = 5 | 50 → 50 | OUI |

Elle ne mord donc **qu'à `dim = 2`**.

**Impact sur les nombres publiés : exactement nul.** Aux hyperparamètres
déployés, 4 scénarios canoniques × 3 instantanés (Re=400, N=256) :

    décisions de fondamental exact changées : 0 / 12
    max|ΔE| global                          : 0,000e+00

Zéro **exact**, pas « petit ». La raison est D-47 : la fenêtre gaussienne
vaut au plus 1,15e−31 au réglage déployé, donc `|C_edges| < 1e-6` et
**aucun terme ZZ n'est émis**. Dédupliquer n'a rien à retirer.
Corollaire noté au passage : le fondamental vaut **255** sur les 12
instantanés — 8 qubits tous à 1, « raffiner partout », cohérent avec D-47.

**Pourquoi maintenant et pas après la campagne.** L'impact est nul
aujourd'hui, donc la correction est **gratuite** : aucun nombre publié ne
bouge, et la coïncidence `study/` ↔ circuit reste à **3,55e−15**. Mais la
réoptimisation rééquilibre précisément les poids qui rendent le défaut
invisible : si `w_z_frac` se resserre ou `σ` s'élargit — ce que la campagne
peut choisir — le ZZ redevient actif et le facteur 2 devient réel, à
`dim = 2`, la seule taille de toutes les campagnes publiées. Corriger après
coup obligerait à tout rejouer.

Vérifier : `pytest tests/quantum/test_period_hamiltonian_dim2_bond_duplication.py -q`
→ **9 passed**, dont un test qui vérifie que le champ d'essai a bien du ZZ
vivant — écrit sur les coefficients déployés, tout le banc passerait à vide.

**Un second garde, remesuré et non retouché.** `D-59` a laissé rouge
`tests/pipeline/test_v1_guards.py::TestPruningThreshold::test_a_coupling_above_the_cut_does_reach_the_operator`,
et la branche vive a été poussée dans cet état (vérifié : le test échoue à
`7b12857` seul, avant toute fusion). Ce n'est ni un défaut du code ni un
test faux — c'est un **seuil périmé** au sens de `VIGIL.md`, remesuré :

| dim | entrées ZZ avant D-59 | après | `dim*dim` | étiquettes distinctes |
|---|---|---|---|---|
| **2** | **4** | **2** | 4 | **2** |
| 3 | 9 | 9 | 9 | 9 |
| 4 | 16 | 16 | 16 | 16 |

L'assertion `len(zz_terms) == DIM * DIM` était juste à `dim >= 3` et fausse
au seul `dim = 2` — la taille de toutes les campagnes publiées. Valeur
remesurée : **2**.

**Et le point qui compte pour la méthode** : le nombre d'**étiquettes
distinctes** valait 2 avant comme après. Ce contrôle comptait des *entrées
de liste*, pas des liens — c'est exactement ce qui a laissé vivre D-59, et
un contrôle qui aurait compté les étiquettes l'aurait trouvé. Cette
assertion est désormais la première du test. Corollaire de `VIGIL.md` :
un test écrit à partir du code partage son modèle mental, donc son erreur.

Vérifier : `pytest tests/pipeline/test_v1_guards.py -q` → **13 passed** ;
le même garde rejoué sur le code d'avant `7b12857` → **1 failed**
(`assert 4 == 2`, les deux étiquettes doublées affichées).
---

# Notes hors chemin critique — enregistrées, groupées, non instruites

Règle d'arrêt de `DEFAUTS.md` : ce qui ne porte ni une lecture publiée ni la
campagne se note ici en une ligne et se traite en un lot unique **après** la
campagne. Ne pas ouvrir d'entrée `DEFAUTS.md` pour ces objets.

- **`test_the_frozen_mechanism_can_still_fire` (`c197373`) ne garde pas le
  mécanisme qu'il annonce garder.** Il écrit son propre fichier temporaire
  et rejoue l'assertion « à la main » : il n'appelle ni
  `test_a_frozen_launcher_says_so_in_its_own_header`, ni `_FROZEN`, ni un
  lanceur. Mesuré — en rendant la boucle du vrai mécanisme inerte
  (`for fragment in []`), `pytest tests/study/test_every_launcher_invokes_real_files.py`
  rend **18 passed, 1 skipped**, identique à la référence, et le garde
  lui-même passe. Le fichier reste donc sans filet contre le balayage vide
  que `_FROZEN = {}` a créé. Correctif estimé à ~5 lignes : peupler `_FROZEN`
  via `monkeypatch` sur un lanceur temporaire, et appeler la vraie fonction.
- **`tests/study/test_phase12_threshold_comes_from_train.py` s'efface en
  silence si `qiskit-machine-learning` manque.** Il ouvre sur
  `pytest.importorskip("qiskit_machine_learning")` — un module-level skip —
  alors que ce paquet est déclaré dans `environment.yaml` et que la règle
  écrite du dépôt, dans l'en-tête de `tests/study/test_fig0_pareto_paths.py`
  (D-94), est : *« On importe le module par une fixture qui ASSERTE, jamais
  par `importorskip` : un module qu'on ne peut pas importer doit rendre la
  suite ROUGE, pas verte-avec-skip. »* Mesuré : sans le paquet,
  `pytest <ce fichier>` rend **« no tests collected »** et les trois gardes
  de D-81 disparaissent de la suite complète sans autre trace qu'un `s`.
  Avec le paquet : **3 passed**. C'est le seul `importorskip` au niveau
  module de toute la suite (les autres portent sur des modules du dépôt,
  dans des fixtures).
- **La docstring de `precompute_dns` renvoie à un chemin de test qui
  n'existe plus.** Elle écrit *« `tests/test_precompute_dns_contracts.py`
  fige les deux [conventions] »* ; le fichier vit à
  `tests/solver/test_precompute_dns_contracts.py` depuis la
  réorganisation `17d983d`. Même famille que D-142 (dix chemins `pytest`
  périmés dans `RESULTS.md`), mais hors de sa portée : le garde de D-142
  balaie `RESULTS.md`, pas les docstrings de `src/`. Le renvoi est le seul
  fil entre une convention subtile et le test qui la fige — et ce fil est
  cassé. Une ligne à corriger, aucune mesure en jeu.
- **Le repli `patch_ratio = 1,0` du chemin de divergence est MORT, et il
  rejouerait D-67 s'il revivait.** `pipeline.py:678` écrit
  `total_pixel_used / (step_simulated * N**2) if step_simulated > 0 else 1.0`.
  `step_simulated += 1` est à la ligne 622, dans la même itération et
  au-dessus : au point du repli, `step_simulated >= 1` toujours. La branche
  `else` est donc inatteignable — mais si elle l'était, elle rendrait
  `patch_ratio = 1,0` sur un run qui n'a rien intégré, donc
  `combined = lambda/(1+lambda)`, le nombre parfaitement plausible que D-67
  a fait interdire dans `score()` (qui, lui, LÈVE sur `total_steps <= 0`).
  Deux chemins censés coïncider, dont un seul crie. Le chemin final, lui,
  est bien gardé : `pipeline.py:751` passe `step_simulated` à `score()`,
  donc un run vide y lève. Hors chemin critique parce que la branche est
  morte ; noté parce qu'un réordonnancement la réarmerait en silence.
- **`D-132` désigne deux défauts différents selon le document.**
  `DEFAUTS.md` → « le bras QAOA ne classe plus, sur une partie de l'espace »
  (renuméroté depuis D-118 le 17 août, le maximum de la branche de revue
  étant alors lu comme D-131). `RESULTS.md` ligne de registre → « le garde
  qui empêche le retour de D-96 se contourne en retirant deux espaces ».
  Même nuit, deux branches, même numéro. `D-133` est dans le même cas côté
  registre. Aucun des deux ne peut être renuméroté par la règle de la fiche
  — *« renuméroter le sien, jamais celui qui est déjà publié dans un
  commentaire de PR »* — les deux le sont. C'est une décision, pas une
  correction : un lecteur qui cherche D-132 dans `RESULTS.md` trouve un
  garde de test là où le brief le renvoie à une bisection.
- **`CLAUDE.md` § Tests de recette annonce `180 / 164 / 16 / 0`.**
  L'agrégateur rend **180 / 176 / 4 / 0** depuis que D-58 est clos. Même
  incohérence entre `docs/BRIEF_REPRISE.md` §7 (juste) et §8 (périmé).
- **`DEFAUTS.md` présente encore D-58 comme ouvert** dans son paragraphe
  d'ouverture, alors qu'il est clos et sorti du fichier.
- **`figures/pareto_frontier.interp_frontier` suppose `front` déjà trié par
  `patch`, sans le vérifier ni le garantir — contrairement à sa jumelle**
  `study/closed_loop/closed_loop_fold_synthesis.interp_frontier`, qui trie
  `trace` en interne avant d'appeler `np.interp` (et dont
  `tests/study/test_t15c_synthesis.py:127` épingle explicitement
  l'invariance à l'ordre d'entrée). Trouvé par un balayage systématique des
  noms de fonction dupliqués entre fichiers (question 4 de `VIGIL.md`,
  appliquée à l'échelle du dépôt plutôt qu'à un module). Mesuré :
  `interp_frontier([{"patch":.6,"phys":.1},{"patch":.2,"phys":.4},{"patch":.9,"phys":.02}], 0.40)`
  rend **0,1** côté `pareto_frontier` (faux — `np.interp` sur des `xs` non
  croissants ne dit rien) contre **0,25** côté `closed_loop_fold_synthesis`
  (correct, identique qu'on lui donne l'entrée triée ou non). **Piège armé,
  non déclenché** : les deux seuls appelants réels de la version
  `pareto_frontier` (`main()` du même fichier et `pareto_panel.draw_panel`)
  reçoivent tous les deux `front` depuis `load_points`, qui trie déjà
  (`figures/pareto_frontier.py:62-64`) — vérifié par lecture de tous les
  sites d'appel (`grep -rn "interp_frontier("`), aucun nombre publié n'en
  dépend aujourd'hui. `tests/study/test_v4_modules.py:450-459` ne couvre
  que l'entrée déjà triée, donc ne l'aurait pas vu. Non corrigé ici : sans
  conséquence mesurée et hors campagne en cours, à traiter avec le lot.

# D-47 — l'Hamiltonien v1 dégénère vers « raffiner partout » à résolution VQA

**Décision de USER : option 1 — documenter comme résultat.** C'est une
limite structurelle de v1, pas un défaut de la phase 4.

## Ce qui est mesuré

Phase 4 diagonalise exactement l'Hamiltonien et demande si son fondamental
désigne les bonnes cellules. Sur les 40 instantanés disponibles (dim = 2,
la seule dimension exécutable — dim 4/8 demandent 32/128 qubits contre le
plafond de 20) :

| | |
|---|---|
| décision exacte tout-à-1 | **40/40** |
| ligne de base classique tout-à-1 | **40/40** |
| `exact_refine != classical_refine` | **0/40** |
| F1 exact == F1 classique | **40/40**, jamais supérieur |
| `promising` avec `>=` | **40/40** — avec le `>` du commentaire : **0/40** |

Deux prédicteurs **constants identiques** rendent le même F1 par
construction : la porte portait **zéro bit** dans les deux sens.

## Le mécanisme, en trois nombres

| grandeur | valeur |
|---|---|
| `(score − thr)/σ` minimum | **8,4** → fenêtre ZZ ≤ **1,15e−31** |
| `min\|H_edges\|` / `max\|K_plaquettes\|` | **2,0 à 6,6** |
| signe du biais Z | **positif partout** |

Le contenu quantique — la seule raison d'utiliser QAOA — vit dans ZZ et
ZZZZ. La fenêtre gaussienne éteint ZZ ; le biais Z, qui n'est qu'une
reformulation du score classique, écrase le ZZZZ. Le fondamental met tous
les qubits à |1⟩ **faute de terme portant une structure spatiale**.

**L'échappatoire évidente a été testée et réfutée.** On pouvait soupçonner
un désaccord d'opérateur (champs moyennés par bloc, score max-poolé depuis
la pleine résolution). Rejoué avec le score assorti : **39/40** au lieu de
40/40, F1 à égalité 40/40. L'écart existe, ce n'est pas la cause. Ce qui
sature est la résolution VQA elle-même.

## Ce qui change dans le code

`promising` devient un **diagnostic**, il n'est plus un **filtre**. La
phase 5 (`study/common/qaoa_inputs.py`) traitait auparavant les seuls
instantanés prometteurs ; elle les traite désormais **tous**, et imprime le
compte en le disant.

**Aucun nombre publié ne bouge** : avec `>=`, `promising` était déjà vrai
40/40, donc rien n'était écarté. Le changement retire une sélection qui ne
sélectionnait rien — mais qui aurait pu tout écarter au premier
rééquilibrage de `σ` ou `w_z_frac`.

## Pourquoi ne pas régler `σ` ou `w_z_frac` à la main

Ce sont **deux des neuf paramètres réoptimisés**. Les fixer maintenant
reviendrait à décider par avance ce que la campagne existe pour mesurer, et
tout réglage manuel serait jeté si elle en choisit d'autres. D-53 confirme
par ailleurs que `dim = 3` ne sauve pas : ce n'est pas un artefact de
petite taille.

Vérifier : `pytest tests/study/test_phase5_ne_filtre_plus_sur_promising.py -q`
→ **5 passed**, dont un garde qui vérifie que le détecteur de filtre mord
encore.


# D-136 — le mode de stockage prévu pour la location plantait au lancement

**Trouvé par la répétition de campagne**, avant toute location.

`train_hyperparams._get_storage` propose trois modes : SQLite local,
`OPTUNA_STORAGE` (RDB / Postgres) et **`OPTUNA_JOURNAL`**. Le troisième est
celui prévu pour un système de fichiers **partagé** (NFS) — donc celui
qu'on choisirait pour paralléliser sur plusieurs machines louées.

Il levait `AttributeError` dès la première lecture de la base :

```
AttributeError: module 'optuna.storages' has no attribute 'JournalFileBackend'
```

`JournalFileBackend` et `JournalFileOpenLock` ont quitté `optuna.storages`
pour `optuna.storages.journal` en **Optuna 4.0**. À la racine,
`JournalFileBackend` n'existe plus du tout (vérifié sur la version installée,
**4.9.0**).

**Ce que ça aurait coûté** : le mode se serait effondré au premier appel, sur
des cœurs facturés, après l'allocation des machines et le précalcul des DNS.

Corrigé par un import depuis `optuna.storages.journal`, avec repli sur
l'ancien chemin pour rester compatible avec Optuna < 4.

**Vérification, sur les deux backends** :

| | reprise | parallèle | intégrité |
|---|---|---|---|
| `sqlite` | 3 → 6 | 6 → 12, 2 workers | 12 COMPLETE, 0 RUNNING |
| `journal` | 3 → 6 | 6 → 12, 2 workers | 12 COMPLETE, 0 RUNNING |

`bash scripts/repetition_campagne.sh [sqlite|journal]` → **REPETITION
REUSSIE** sur les deux.

**Pourquoi la répétition a trouvé ce que la suite de tests n'a pas trouvé** :
aucun test n'exerçait `OPTUNA_JOURNAL`. Le banc de fumée
(`test_train_hyperparams_smoke.py`) force `DISTRIBUTED = True` et
`OPTUNA_STORAGE`, donc il traverse le chemin RDB et **jamais** le chemin
journal. Un mode annoncé qu'aucun test ne traverse est une promesse non
tenue — même famille que D-48 (`mode="hardware"` accepté et exécuté sur
simulateur).

---

# La base de campagne porte le nom de son étude, et les fantômes se comptent

Trouvé en lançant réellement la réoptimisation, le 18 août 2026. Deux
défauts distincts, tous deux dans l'**organisation du stockage** — aucun ne
touche `src/`, donc aucun nombre publié ne bouge.

## Le fait

`scripts/run_reoptimisation.sh` écrivait en dur `q_has_v3.db`. La base ainsi
créée contient l'étude `q_has_v2_phase1`, nom que `train_hyperparams.PHASES`
donne à `phase1_composite`. **Le fichier et son contenu ne disaient pas la
même chose.** C'est la forme exacte du défaut que D-22 a coûté : un réglage
dont la provenance ne se lit plus.

Le reste du dépôt respectait déjà l'invariant — `optuna_studies/q_has_v2_phase1.db`
contient bien `q_has_v2_phase1`. Seul le lanceur de la campagne à venir ne
le respectait pas, c'est-à-dire précisément la campagne censée **résoudre**
D-22.

## Ce qui a changé

Le lanceur lit le nom d'étude dans `PHASES` et nomme la base d'après lui :
`results/hyperparams/reoptimisation/q_has_v2_phase1.db`. Le dossier
`reoptimisation/` distingue cette campagne des bases gelées ; le nom de
fichier dit quelle étude, comme dans `optuna_studies/`.

Deux gardes s'ajoutent aux trois existantes, avant le premier essai :

- **garde 4** — `scripts/inventaire_campagne.py` : toute base non vide porte
  exactement une étude, et son nom est le basename du fichier ;
- **garde 5** — `scripts/nettoyer_essais_fantomes.py` : les essais `RUNNING`
  qu'aucun worker ne finira sont marqués `FAIL` au démarrage.

## Les essais fantômes, mesurés

Un worker tué — instance spot reprise, conteneur recyclé, OOM — laisse son
essai `RUNNING` pour toujours. Trois de mes workers ont été fauchés par un
recyclage de conteneur et ont laissé trois fantômes.

`python scripts/inventaire_campagne.py` (commit `327d726`) :

| | bases | vides | COMPLETE | fantômes `RUNNING` |
|---|---|---|---|---|
| tout `results/hyperparams/` | 26 | 10 | **2 815** | **298** |
| dont `optuna_studies/` (hors archives) | 10 | 8 | 303 | 45 |

Les 45 de la seconde ligne sont ceux que `PROVENANCE.md` annonce (18
classiques + 24 quantiques + 3 de la reprise). **L'inventaire confirme
`PROVENANCE.md` ligne à ligne** : seules `classical_v2_phase1` (125 COMPLETE)
et `q_has_v2_phase1` (178 COMPLETE) portent des données ; les huit autres
bases de `optuna_studies/` sont vides.

Les bases gelées ne sont **pas** nettoyées : leurs comptes sont publiés, les
toucher déplacerait des nombres. La garde 5 n'agit que sur la base de
réoptimisation.
# Une normalisation dont l'équilibre ne dépend plus de `dim`

Décidé par USER après l'analyse théorique des trois normalisateurs du v2.
**Ajout derrière un drapeau** — `norm="legacy"` reste le défaut, donc aucun
nombre publié ne bouge (table maîtresse : **180 / 176 OK / 4 DIFF /
0 MISSING**, inchangée). Même idiome que `fixed_curl` : les deux chemins
coexistent et se comparent.

## Le fait, théorique avant d'être mesuré

`build_patch_hamiltonian` moyenne les champs par blocs à `dim × dim` et
calcule *tous* les coefficients sur ce champ grossier. `dim` est donc la
coupure d'un filtre passe-bas appliqué à l'entrée.

Or les trois termes du chemin `legacy` emploient **trois normalisateurs
différents** :

| terme | normalisé par | quand `dim` monte |
|---|---|---|
| ZZ | la **moyenne** des sauts | `max|C| = w_ZZ · max(saut)/⟨saut⟩` — le rapport pic/moyenne, qui **croît** avec l'intermittence |
| ZZZZ | `max|ω| + max|J|` — **deux** maxima en des points distincts | borné, `max|K| < w_ZZZZ` d'une quantité qui dépend du champ |
| biais Z | la **médiane** des couplages | la médiane/moyenne **décroît** quand la distribution se dissymétrise |

Un champ MHD est intermittent par nature — nappes de courant, chocs. Donc
le rapport biais/couplage **dérive avec `dim` par construction**, et la
dérive est scénario-dépendante. Conséquence pratique : **un réglage
d'hyperparamètres obtenu à une taille ne transfère pas à une autre**, et un
balayage en `dim` mesurerait deux choses à la fois.

## Ce que `norm="max"` change

```
C_ij = -w_ZZ   * |saut_ij|     / max|saut|         ->  max|C| == w_ZZ
K_p  = -w_ZZZZ * (|ω| + |J|)   / max(|ω| + |J|)    ->  max|K| == w_ZZZZ
h_i  = +c_bias * max(|C|,|K|)  * (s_i - thr)
```

L'**équilibre** entre les termes devient indépendant de `dim` : le rapport
biais/couplage vaut exactement `c_bias`. Les gardes y sont **multiplicatifs**
(`pic if pic > EPS else 1.0`) et non additifs, ce qui rend les invariances
exactes.

## Mesuré

Invariance d'amplitude (multiplier v et B par 10), écart relatif max :

| chemin | écart | cause |
|---|---|---|
| `legacy` | **9,8e-11** | garde additif `+ EPS`, qui décale l'échelle |
| `max` | **4,8e-16** | garde multiplicatif — exact à la précision machine |

La docstring de la classe annonçait cette invariance sans la qualifier ;
elle tient à 1e-10 près sur `legacy`, exactement sur `max`. Les deux
tolérances sont désormais épinglées séparément — une tolérance unique et
lâche cacherait que `max` est exact.

Équilibre à `dim ∈ {4, 8, 16, 32}`, même champ d'essai : `max|C|`, `max|K|`
et `max|h|` identiques aux quatre tailles à 1e-12 près sous `max` ; sous
`legacy` le rapport biais/couplage s'étale d'un facteur > 1,5.

## Ce que ça ne fait pas

Le **motif spatial** de `C` dépend encore de `dim`, et ne peut pas ne pas en
dépendre : le champ d'entrée lui-même change avec la coupure. Un test le
vérifie explicitement — s'il venait à passer, c'est que le champ d'essai
serait devenu auto-similaire et que les tests d'invariance ne prouveraient
plus ce qu'ils annoncent.

Et le max est **sensible aux extrêmes** là où la moyenne ne l'est pas : sur
un champ intermittent un seul saut fixe l'échelle du domaine. Si cela devient
gênant, un percentile haut garde l'invariance d'échelle sans la fragilité du
max strict.

## Vérification

```bash
python scripts/inventaire_campagne.py                    # code 0, 26 bases
python scripts/nettoyer_essais_fantomes.py --toutes --dry-run
python -m pytest tests/pipeline/test_campagne_noms_et_fantomes.py -q
```

**15 tests, tous verts.** Ils construisent leurs propres bases SQLite : ils
ne dépendent d'aucun artefact et ne peuvent pas devenir verts par
disparition de leur entrée.

python -m pytest tests/mapping/test_normalisation_max_invariante.py -q   # 24 passed
python -m pytest tests/mapping -q                                        # 436 passed
python study/common/aggregate_master_table.py                            # 180 / 176 / 4 / 0
```

**Les tests mordent, mesuré par mutation** :

| mutation | effet |
|---|---|
| le lanceur recode `q_has_v3.db` en dur | 1 failed |
| le nettoyage ne vérifie plus les workers vivants | 2 failed |
| le nettoyage passe `FAIL` à **tous** les essais, pas aux seuls `RUNNING` | 1 failed |

Une première version du test du lanceur assertait sur le **texte** du script
et passait au vert dès qu'un commentaire mentionnait `q_has_v3.db` — la
famille D-123→D-131. Remplacée par l'évaluation des lignes réelles qui
calculent `DB`.

## Ce que ça change pour la campagne

Rien sur le fond, tout sur l'attribution : la base à venir portera un nom qui
désigne son contenu, et son compte d'essais sera celui du travail réellement
fait. Sans ces deux gardes, la campagne censée lever D-22 aurait reproduit
la cause de D-22.
| le biais repasse à la médiane en mode `max` | 5 failed |
| ZZ repasse à la moyenne en mode `max` | 6 failed |
| le défaut bascule sur `max` | 1 failed |

La troisième est délibérée : basculer le défaut est un changement de
comportement scientifique, il doit échouer tant qu'il n'est pas décidé et
consigné.

---

# La courbe de cône : deux artefacts, et une conclusion qui s'inverse avec `dim`

D-88 notait que `results/` ne contenait **aucun** `t1b_cone_curve_*.npz`. La
mesure qui décide de H3 — *l'information des voisins apporte-t-elle quelque
chose ?* — n'existait pas. Elle existe, à deux tailles.

```bash
python study/pipeline/hard_patch_labels.py --dim 16 --N 96      # patches manquants
cd study/h2b_prediction
python h2b_neighbour_cone_curve.py --dim  8 --N 96 --max-snaps 20 --seed 0
python h2b_neighbour_cone_curve.py --dim 16 --N 96 --max-snaps 20 --seed 0
```

git `ef5f0a4`. Déterministe (vérifié à `dim = 8` : 19 clés identiques sur
deux exécutions).

## ⚠️ Rétractation d'une première lecture, publiée le 21 août

Une version antérieure de cette section annonçait *« gain faible en
distribution (+0,053), aucun sous transfert »* et concluait que la courbe
allait dans le sens de l'énoncé économique. **Les deux moitiés sont
retirées.** Deux choix de paramètres, non justifiés, la produisaient :

| choix | ce qu'il faussait |
|---|---|
| `--max-snaps 8` alors que **20** étaient disponibles | sous-échantillonnage. Le `[FLAG n_tr/F<20]` sur k=2 et k=3 était **fabriqué par ce choix**, pas une propriété des données : à 20 instantanés, `n_tr/F` vaut 34,8 à k=3. Le gain du premier saut passe de **+0,006 à +0,099** en distribution |
| `--dim 8` | à cette taille **k=3 couvre 76,6 % de la grille**. Ce n'est plus un voisinage. D-88 n'attrape que les colonnes dupliquées à l'identique, pas le fait que le voisinage ait avalé le domaine |

**La couverture du carré de Chebyshev, par taille** — la table qui manquait :

| dim | k=0 | k=1 | k=2 | k=3 |
|---|---|---|---|---|
| 4 | 6,2 % | 56,2 % | **100 %** | **100 %** |
| 8 | 1,6 % | 14,1 % | 39,1 % | **76,6 %** |
| 16 | 0,4 % | 3,5 % | 9,8 % | 19,1 % |
| 32 | 0,1 % | 0,9 % | 2,4 % | 4,8 % |

`dim = 16` est la **première taille où les quatre k sont des voisinages**.

## Le résultat, aux deux tailles, 20 instantanés

Moyennes LOSO. `harris_tearing` rend **0,000 à tous les k, aux deux
tailles** : le modèle n'y prédit aucun positif. C'est un pli **dégénéré** au
sens du protocole §1.3 B3, et un prédicteur dégénéré ne vote pas. Les deux
colonnes sont données, parce que la conclusion en dépend.

| | dim=8 (k=3 = 77 %) | | dim=16 (k=3 = 19 %) | |
|---|---|---|---|---|
| | 4 plis | hors pli mort | 4 plis | hors pli mort |
| classique | 0,443 | 0,369 | 0,577 | 0,444 |
| k=0 | 0,245 | 0,327 | 0,322 | **0,429** |
| k=1 | 0,250 | 0,333 | 0,445 | **0,593** |
| k=2 | 0,168 | 0,223 | 0,369 | 0,491 |
| k=3 | 0,261 | 0,349 | 0,469 | **0,625** |

## Ce que ça dit — et c'est l'inverse de ce qui était écrit

**Le cône n'est pas plat.** La règle de décision pré-enregistrée du module
est explicite : *« flat : every |delta| ≤ 0.01 → cone retired »*. Les écarts
par saut valent **+0,123 / −0,076 / +0,100** à `dim = 16`. Par sa propre
règle, **le cône n'est pas retiré.**

**À la taille la mieux posée, l'information des voisins aide beaucoup.**
Hors pli mort, à `dim = 16`, un seul saut fait passer de 0,429 à **0,593**
(+0,164), et le meilleur k rend **0,625 contre 0,444** pour le classique.

**Et le gain CROÎT quand on affine.** De `dim = 8` à `dim = 16`, hors pli
mort, le gain du premier saut passe de +0,006 à **+0,164**. C'est la
**direction opposée** à la clause de `PLAN_PREPRINT` §7 — *« le gain décroît
quand on affine la grille »*. Cette clause n'est pas soutenue par ces deux
points ; elle est contredite.

## Ce qui empêche d'en faire un verdict

1. **Tout dépend du pli mort.** Avec les quatre plis, le cône reste sous le
   classique aux deux tailles (0,469 contre 0,577 à `dim = 16`). Hors pli
   mort il le dépasse. Tant que `harris_tearing` n'est pas expliqué — le
   classique y rend 0,976 pendant que le GBT s'effondre à 0,000, une
   asymétrie qui demande sa propre mesure — la moyenne n'est pas citable
   dans un sens ni dans l'autre.
2. **La courbe n'est pas monotone** (+0,123, −0,076, +0,100) : aucune pente
   ne se cite. Le creux à k=2 se reproduit aux deux tailles, il n'est pas du
   bruit et il n'est pas expliqué.
3. **Deux points en `dim`, une graine, un N.** Une direction lue sur deux
   points n'est pas une tendance.
4. **Un biais de construction qui borne toute lecture** : les 8 features
   autres que `score_vqa` sont calculées sur les champs **moyennés par
   blocs**, alors que le label est la variance **intra-patch** que ce
   moyennage supprime. Ajouter des voisins de moyennes de patch ne peut pas
   restituer une variance sous-patch. Une courbe plate serait donc en partie
   garantie par la construction — ce qui rend d'autant plus notable qu'elle
   ne le soit pas.

## Conséquence pour le manuscrit

L'énoncé *« l'information des voisins ne sert à rien »* n'est pas soutenu :
il est **contredit** à la seule taille où la question est bien posée. Et
l'énoncé économique de `PLAN_PREPRINT` §7 doit être **réécrit** : sa clause
« décroît quand on affine » va contre la mesure.

Ce qui reste vrai et défendable : le cône ne suffit pas à porter la
fermeture de l'approche. C'est **H0b** qui la porte — mieux résoudre
l'hamiltonien dégrade la décision — et H0b ne dépend ni de la localité ni de
ce résultat.

---

# Le corpus de patches vit dans DEUX conventions de rotationnel

Trouvé en régénérant les patches `dim = 16` à N = 96 : la commande a aussi
réécrit huit artefacts `dim = 16` à N = 256 et N = 64, et
`tests/study/test_patches_classical_score_provenance.py` a rougi — 25 cas.

**Ce n'était pas un dégât, c'était un gel documenté que j'avais ignoré.**
Les huit fichiers sont **délibérément conservés dans l'ancienne convention**
(`fixed_curl=False`), et un test épingle ce fait, artefact par artefact.
Les huit ont été restaurés ; le test rend de nouveau **62 passed**.

## Le fait, mesuré

Sur les huit fichiers réécrits, avant restauration :

| grandeur | écart |
|---|---|
| `l2_errors` | **identique** |
| `is_hard` | **identique** |
| `classical_scores` | **100 % des cellules diffèrent**, jusqu'à 0,39 en absolu (3,7× en relatif) |

Le **label ne bouge pas** — il ne passe pas par les mappeurs. Seul le score
classique bouge, et il bouge partout.

## L'état réel du corpus

Recalcul à HEAD (`fixed_curl=True` par défaut), écart au score stocké :

| famille | convention | écart au recalcul HEAD |
|---|---|---|
| `*_N96_dim8` | **HEAD** | 0,000e+00 |
| `*_N96_dim16` (produits ici) | **HEAD** | 0,000e+00 |
| `*_N256_dim16`, `*_N64_dim16` | **ancienne**, gelée | ≠ 0 (c'est le gel) |

**Conséquence à ne pas perdre** : les deux points de la courbe de cône
(`dim = 8` et `dim = 16`, tous deux à N = 96) sont dans la **même**
convention et se comparent entre eux. Ils ne se comparent **pas** à un
nombre calculé sur la famille gelée N = 256 / N = 64.

## Deux défauts de garde, notés sans correction

1. **Le test de provenance porte une liste de noms en dur** : les seize
   fichiers `N96_dim16` produits ici ne sont couverts par **aucun** de ses
   cas (`--collect-only` : 0). Son vert ne dit donc rien de leur convention —
   il a fallu la mesurer à la main. Un garde dont le balayage ne suit pas le
   corpus laisse entrer ce qu'il est censé surveiller.
2. **`hard_patch_labels.py --N 96` réécrit des artefacts d'autres N.** Le
   sélecteur `--N` filtre les DNS lus, pas les fichiers écrits : la commande
   a touché des N qu'on ne lui avait pas demandés. Même famille que D-158,
   où l'agrégateur réécrit la table publiée sur une configuration qui ne
   correspond à rien.

Aucun des deux n'est corrigé ici : le premier touche un test de gel, le
second un écrivain d'artefacts publiés — les deux demandent une décision.

---

# `dim` n'a de sens que relativement à `N` — et la limite est dure

Question de USER : *« la dim ne marche que si le N est adapté, n'est-ce pas ? »*
Oui, et la contrainte n'est pas affaire de goût.

## Le mécanisme

Le label de la phase 2 est l'écart-type **intra-patch** : `patch_l2_errors`
remplace chaque patch par sa moyenne, puis mesure l'écart du champ fin à
cette moyenne. Un patch de `p × p` cellules estime donc cet écart-type sur
`p²` échantillons — et **à `p = 1` il vaut identiquement zéro**, puisqu'une
cellule ne dévie pas de sa propre moyenne.

Le seuil étant un percentile de valeurs toutes nulles vaut 0, et
`is_hard = (l2 >= 0)` marque alors **100 % des patches comme durs**. Tout F1
mesuré dessus vaut celui du prédicteur constant.

| N | dim | patch `p` | `p²` | état |
|---|---|---|---|---|
| 96 | 8 | 12 | 144 | ok |
| 96 | 16 | 6 | 36 | **marginal** |
| 96 | 32 | 3 | 9 | bruit |
| 256 | 16 | 16 | 256 | ok |
| 64 | 64 | 1 | 1 | **vide** |

**Règle** : `dim ≤ N/8` pour rester confortable (`p ≥ 8`, soit ≥ 64 points).
Conséquence directe sur la courbe de cône : son point `dim = 16` a été pris à
N = 96, donc `p = 6` — **marginal**. Le couple sain pour `dim = 16` est
N = 256.

## Le corpus, mesuré

**4 artefacts au label identiquement nul**, tous `*_N64_dim64` :

```
patches_{harris_tearing,kelvin_helmholtz,mhd_rotor,orszag_tang}_Re400_N64_dim64.npz
   l2_errors tout à zéro   seuil = 0,0e+00   is_hard = 100 % de positifs
```

## Pourquoi ce n'est pas une entrée `DEFAUTS.md`

Par la règle d'arrêt : un défaut n'y entre que s'il porte une lecture publiée
ou empêche la campagne de mesurer ce qu'elle prétend. Mesuré :

- **aucun script** de `study/`, `figures/` ou `scripts/` ne nomme `dim=64` ;
- la table maîtresse ne cite que `dim2`, `dim4`, `dim8`.

Personne ne les consomme. Ils sont donc notés ici et traités après la
campagne, comme la règle le prévoit.

## Ce qui est gardé

`tests/study/test_t28_t29_labels_and_ci.py` gardait déjà le **consommateur**
— le relabelliseur lève `SystemExit` sur un seuil dégénéré — et son propre
message notait que le producteur, lui, *« ne crie pas »*. Ce trou est
maintenant couvert côté producteur :

`tests/study/test_label_degenere_quand_le_patch_est_trop_petit.py`, **10
tests** :

- le mécanisme est **calculé**, pas supposé (`p = 1` → label nul), avec le
  champ qui **sépare** (`p = 2` → label non nul, sinon le test passerait sur
  un champ uniforme sans rien prouver) ;
- la liste des quatre dégénérés est **fermée** : un cinquième fait rougir ;
- une exemption **périmée** crie aussi (si l'un guérit, l'entrée doit sortir) ;
- et la tolérance est conditionnée : le jour où un script nomme `dim=64`,
  elle tombe.

**Vérifié par mutation** :

| mutation | effet |
|---|---|
| un cinquième artefact dégénéré entre dans le corpus | 1 failed |
| un script se met à nommer `dim=64` | 1 failed |
| le plancher de balayage ne lève plus rien | 1 failed |

La troisième a demandé une correction de ma part : la première version du
plancher **ne pouvait pas échouer** — baisser un seuil déjà satisfait ne fait
rien tomber. Le balayage prend désormais son répertoire en paramètre, et un
test lui donne un dossier vide pour vérifier que le plancher se déclenche.
Un plancher qu'on ne peut pas faire tomber n'est pas un garde.

---

# Le zéro de `harris_tearing` n'est pas physique : c'est un seuil transféré

Demandé par USER : *« réinvestigue sur harris tearing, le zéro bizarre »*.
Le paradoxe posé était réel — le score classique rend **0,976** sur ce pli
pendant que le GBT, qui a ce même score comme feature #0, rend **0,000**.

## Mesuré (LOSO, `dim = 16`, N = 96, 20 instantanés)

```
probabilités du GBT sur harris   min/méd/max = 0,0000 / 0,0025 / 0,1243
seuil ajusté sur les 3 autres    = 0,4000
-> positifs prédits : 0 / 20 480   ->  F1 = 0,000
```

**Aucune probabilité n'atteint le seuil.** Le zéro est arithmétique, pas
physique. Ce que la même sortie dit quand on retire le seuil :

| grandeur | valeur |
|---|---|
| F1 au seuil transféré | **0,000** |
| **AUC (classement, sans seuil)** | **0,908** |
| **F1 à budget apparié** (5 120 patches) | **0,659** |

Le classement est bon. C'est l'**opérateur de décision** qui ne traverse pas
la frontière de scénario — les distributions de probabilité des scénarios
d'entraînement et de harris ne se recouvrent pas.

## Un second fait, qui n'était pas cherché

AUC du **score classique seul** contre le label, par scénario :

| scénario | AUC | taux positif |
|---|---|---|
| orszag_tang | 0,592 | 0,157 |
| harris_tearing | **1,000** | 0,250 |
| kelvin_helmholtz | **0,997** | 0,250 |
| mhd_rotor | **0,948** | 0,205 |

Sur trois scénarios sur quatre, le label est **presque une fonction
déterministe du score classique**. C'est cohérent avec sa définition — la
variance intra-patch et un détecteur de gradient agrégé par block-max
désignent la même structure sur une nappe de courant — mais cela veut dire
que la tâche y est quasi gratuite. Un plafond de 1,000 laisse peu de place à
une méthode qui prétendrait faire mieux, et un écart mesuré contre une
baseline parfaite ne mesure pas ce qu'on croit. À verser au dossier de la
spécification de la tâche (H5).

## Ce qui est corrigé

Le protocole §1.3-B3 traite déjà « prédiction constante » comme dégénéré, et
`metrics.degeneracy_flag` l'implémente — mais il rend le **même** verdict
qu'il y ait du signal ou non. C'est cette confusion qui a fait lire harris
comme un pli mort, dans deux versions de ce document.

`metrics.threshold_transfer_flag(gt, proba, seuil)` sépare désormais :

| verdict | condition | ce qu'il faut citer |
|---|---|---|
| `ok` | prédiction non constante | le F1 |
| `aucun_signal` | constante **et** AUC < 0,70 | rien : la méthode n'ordonne pas |
| `seuil_non_transfere` | constante **et** AUC ≥ 0,70 | le **F1 à budget apparié**, jamais le F1 au seuil |

`seuil_non_transfere` **n'est pas un résultat sur la tâche** : c'est un
défaut de l'opérateur de mesure.

## Vérification

```bash
python -m pytest tests/study/test_seuil_non_transfere_vs_absence_de_signal.py -q
```

**8 tests** (7 rapides + 1 `slow` qui rejoue le pli harris réel et exige le
verdict `seuil_non_transfere`, l'AUC > 0,80 et `proba_max < seuil`). Ce
dernier échoue le jour où l'AUC tombe — harris serait alors un **vrai** pli
mort, et la lecture changerait.

Les cas de test sont choisis pour **séparer** : mêmes probabilités basses,
une fois avec classement, une fois sans. Un test vérifie explicitement que
`degeneracy_flag` seul rend le même verdict sur les deux — c'est la
confusion qu'on corrige.

**Mutations** :

| mutation | effet |
|---|---|
| le verdict ignore l'AUC (retour à §1.3-B3 seul) | 2 failed |
| le plancher d'AUC mis à 0 | 2 failed |
| la détection de dégénérescence neutralisée | 5 failed |

## Conséquence sur ce qui a été écrit

Les moyennes LOSO de la courbe de cône, publiées plus haut avec la mention
« pli mort », doivent être relues : `harris_tearing` n'est pas mort, il est
**mal seuillé**. La colonne « hors pli mort » de cette table reste la bonne à
lire, mais pour une raison différente de celle qui y était donnée.

---

# Sélectivité des coefficients : deux des trois ne captent pas un type

Exigence de USER : *« tous les coeffs doivent être intuitifs — adimensionnels,
indépendants de tout sauf de capter leur type d'instabilité »*. Les
invariances sont mesurées ailleurs ; c'est la **sélectivité** qui manquait,
et elle n'était vérifiée nulle part.

## Le trou de couverture qui rendait la question urgente

Avant ce fichier, **aucun test n'exerçait `norm="max"`** hors des invariances.
Basculer le défaut aurait laissé toute la réponse physique non vérifiée.
Les deux normalisations sont désormais exercées à chaque cas.

## Mesuré, sur champs analytiques à réponse connue

Pic de chaque famille, `dim = 32` :

| champ | max\|C\| (ZZ) | max\|K\| (ZZZZ) | max\|K_xp\| |
|---|---|---|---|
| uniforme | 0 | 0 | 0 |
| rotation solide (ω≠0, J=0) | ≠0 | **1,000** | 0 |
| nappe de courant (ω=0, J≠0) | ≠0 | **1,000** | 0 |
| X-point (det ∇B < 0) | ≠0 | 1,000 | **1,000** |

## Les trois verdicts

**`K_xpoint` est sélectif** — et c'est le seul. Il ne répond qu'au
det(∇B) < 0, il est exactement muet là où det > 0 (points elliptiques,
cœurs d'îlots), et il survit à un facteur d'échelle 10.

**`K_plaquettes` ne l'est pas.** `(|ω| + |J|)/norme` somme les deux
magnitudes : un vortex pur et une nappe de courant pure rendent **exactement
la même valeur** (1,000 contre 1,000). Le terme capte « il se passe quelque
chose de rotationnel **ou** magnétique », pas un type. C'est une propriété de
conception, pas un défaut — mais elle contredit l'exigence « capter son type
d'instabilité », et elle est désormais épinglée.

**`C_edges` ne l'est pas non plus.** `sqrt(|dv|² + |dB|²)` fait entrer un
saut hydrodynamique et un saut magnétique dans la même racine.

## Une conséquence de `norm="max"` qu'il faut connaître

Sous `max`, `max|C| = w_ZZ` et `max|K| = w_ZZZZ` sur **tout** champ non
uniforme. Par construction, **la magnitude ne distingue plus rien** : toute
l'information de structure passe dans le motif spatial.

Sous `legacy`, le pic varie : nappe raide 8,09 contre rotation lisse 3,12.

> ### ⚠️ Rétractation, publiée le même jour
>
> Cette section concluait que **« le choix est un arbitrage, pas une
> amélioration pure »** — `legacy` porterait dans son pic une information de
> structure que `max` retirerait. **C'est retiré**, et c'est la mesure
> ci-dessus qui le dit, mal lue la première fois.
>
> `max|K|` vaut 1 dans les **deux** modes. Le pic `max|C|` n'est donc pas une
> grandeur libre : c'est le **poids de la famille ZZ relativement à la famille
> ZZZZ** dans l'hamiltonien. Sous `legacy` il passe de **3,121** (rotation
> lisse) à **8,095** (nappe raide) — un facteur **2,59** sur l'équilibre de
> deux familles de termes, décidé par la spikiness de l'instantané au lieu de
> la conception. Sous `max` il vaut `W_ZZ / W_ZZZZ = 2,000` sur les trois
> champs non uniformes.
>
> Ce n'est pas une information, c'est un **couplage parasite**. Il n'y a donc
> pas d'arbitrage : `max` le retire. La question de USER — *« pour norm je ne
> comprends pas le dilemme »* — n'avait pas de réponse parce qu'il n'y en a
> pas.
>
> Les deux côtés sont désormais épinglés :
> `test_sous_legacy_le_poids_relatif_des_familles_derive_avec_le_champ` et
> `test_sous_max_le_poids_relatif_des_familles_est_celui_de_la_conception`.
> Le test qui portait la lecture retirée s'appelait
> `test_sous_legacy_le_pic_varie_donc_il_porte_une_structure` : il mesurait le
> bon nombre sous un mauvais nom.

## Vérification

```bash
python -m pytest tests/mapping/test_selectivite_des_coefficients.py -q   # 13 passed
```

| mutation | effet |
|---|---|
| la plaquette devient sélective (`\|ω\|` seul) | 3 failed |
| ZZ ignore la partie magnétique du saut | 4 failed |
| le mode `max` cesse de borner le pic | 2 failed |
| `K_xpoint` : `max(0,−det)` → `\|det\|` | 1 failed |
| **le scindement redevient le défaut** (le défaut corrigé ci-dessus) | 4 failed |
| le drapeau est ignoré, les clés sont toujours là | 4 failed |
| le drapeau est ignoré, les clés n'y sont jamais | 23 failed |
| `K_xpoint` : signe inversé (X-point ↔ O-point) | 1 failed |

**Un trou de mon propre fait, corrigé en le mutant.** La première version du
test de sélectivité X-point ne regardait qu'un **maximum sur tout le champ** :
`max(0,−det)` et `|det|` y donnent le même pic, donc la mutation passait au
vert. Le test compare désormais **point par point**, et exige que `K_xpoint`
soit exactement nul là où det > 0. Sans cette correction, le fichier aurait
affirmé une sélectivité qu'il ne vérifiait pas.

---

# La plaquette scindée : un coefficient par type de structure

> ### ⚠️ Entrée SUPERSÉDÉE le 21 août — lire d'abord
>
> Les deux coefficients `K_vorticity` / `K_current` décrits ci-dessous **ont
> été retirés du code**. Décision de USER : *« il ne faut pas faire
> différentes portes ZZZZ (c'est coûteux et ça sert à rien) »* — deux familles
> de termes ZZZZ demandent deux jeux de portes pour la même information.
>
> Le **problème** que cette entrée identifie reste entier, et s'est révélé
> bien plus grave que mesuré ici. La **solution** a changé : au lieu de deux
> termes, un seul, dont les deux magnitudes sont rendues adimensionnelles
> **séparément avant** d'être sommées. Voir l'entrée suivante.
>
> Ce qui survit de celle-ci : le constat de non-sélectivité, le champ mixte,
> le test de l'ensemble des clés, et la mesure `E_max` +15,9 % / +33,6 % qui
> l'a motivé.


Demande de USER, à la suite de l'entrée précédente : *« scinde le en deux
coeffs mais checke bien ce que tu fais et actualise les tests »*. La mesure
qui la motive est la ligne « `K_plaquettes` ne l'est pas » ci-dessus : un
vortex pur et une nappe de courant pure rendent **exactement la même valeur**
(1,000 contre 1,000), donc le terme capte « il se passe quelque chose », pas
un type d'instabilité.

## Ce qui est ajouté

```python
K_vort = -w_ZZZZ * |omega_z| / max|omega_z|
K_curr = -w_ZZZZ * |J_z|     / max|J_z|
```

Deux clés, rendues **sur demande seulement** — `split_plaquette=True`.
Adimensionnelles, bornées par `w_ZZZZ`, invariantes en `dx`, en amplitude et
en mode de normalisation.

**`K_plaquettes` n'est pas touché**, et par défaut **l'ensemble des clés
rendues est celui de `HEAD`**. Vérifié contre `HEAD`, module chargé par chemin
explicite, sur un champ mixte bruité, dans les quatre configurations :

```
norm=legacy xpoint=False  défaut: modifiées AUCUNE (bit à bit), ajoutées AUCUNE
norm=legacy xpoint=True   défaut: modifiées AUCUNE (bit à bit), ajoutées AUCUNE
norm=max    xpoint=False  défaut: modifiées AUCUNE (bit à bit), ajoutées AUCUNE
norm=max    xpoint=True   défaut: modifiées AUCUNE (bit à bit), ajoutées AUCUNE
                          opt-in: ajoute ['K_current', 'K_vorticity']
```

## ⚠️ Pourquoi opt-in : un changement de comportement que j'ai failli livrer

La première version de cette entrée annonçait *« deux clés nouvelles dans le
dictionnaire rendu »* et concluait : *« aucun nombre publié ne bouge […] c'est
une mise à disposition, pas un changement de comportement »*. **C'était
faux**, et la vérification bit à bit ne pouvait pas le voir : elle compare les
**valeurs des clés partagées**.

`src/call_vqa_shell.py` ne consomme pas le dictionnaire clé par clé. Il
**somme `|coeff|` sur toutes les clés tableau**, sans liste blanche, pour
former `E_max` :

```python
for key, value in hamilt_params.items():
    ...
    E_max += np.sum(np.abs(value))
```

Deux clés de plus déplacent donc `E_max`, mesuré sur un champ bruité N=24 :

| mode | `E_max` sans | avec | écart |
|---|---|---|---|
| `legacy` | 2 639,02 | 3 059,70 | **+15,9 %** |
| `max` | 1 231,33 | 1 652,01 | **+34,2 %** |

> Ces deux `E_max` sous `max` sont mesurés sur la **plaquette d'avant** le
> changement de formule (dénominateur commun). Remesurés sur le code actuel :
> 1 251,50 → 1 672,18, soit **+33,6 %**. La ligne de base a bougé avec la
> formule ; la conclusion — l'ensemble des clés fait partie du contrat — ne
> bouge pas. C'est le chiffre **+33,6 %** qui est cité ailleurs.



`src/Simulation/RescaleArrays.py` itère lui aussi sur toutes les clés
(`for key, value in hamilt_params.items()`) et aurait max-poolé les deux
tableaux à chaque descente AMR.

**La leçon, générale : l'ENSEMBLE des clés fait partie du contrat.** Prouver
que chaque valeur partagée est identique au bit près ne prouve pas qu'un
consommateur ne bouge pas, dès lors qu'un consommateur agrège sur les clés.
C'est exactement le « changement de comportement scientifique fait au
passage » que `CLAUDE.md` interdit sur `src/`, et je l'avais écrit comme son
contraire.

Corrigé par le drapeau, et **gardé** par
`test_par_defaut_lensemble_des_cles_rendues_est_inchange` (liste fermée des
5 clés par défaut) et
`test_le_scindement_deplace_E_max_ce_qui_est_la_raison_du_opt_in`, qui
rejoue le calcul de `call_vqa_shell` et chiffre ce que le défaut évite.
Remettre le scindement par défaut fait rougir 4 tests.

Le circuit ne lit toujours pas les deux termes : les brancher est une
décision de conception distincte, qui demandera sa propre campagne.

## La sélectivité, mesurée

`dim = 32`, champs analytiques, `split_plaquette=True`, pics des
coefficients :

| champ | `K_plaquettes` | `K_vorticity` | `K_current` |
|---|---|---|---|
| uniforme | 0 | 0 | 0 |
| rotation solide (ω≠0, J=0) | **1,000** | **1,000** | 0,000 |
| nappe de courant (ω=0, J≠0) | **1,000** | 0,000 | **1,000** |

La ligne à lire est la colonne de gauche : elle ne bouge pas entre les deux
structures, les deux autres s'inversent. C'est le scindement.

## Pourquoi chacun divise par le max de SON signal

L'alternative naturelle serait de garder la normalisation commune de
`K_plaquettes` (`max|ω| + max|J|`). Elle réintroduirait exactement ce que le
mode `max` retire : le poids de chaque terme dépendrait du pic de **l'autre**,
donc de la forme du champ. Sur un champ où B est 4 fois plus faible que v,
`K_vorticity` culminerait à ~0,8 et `K_current` à ~0,2 au lieu de 1 et 1.

C'est vrai dans les **deux** modes, `legacy` compris : le scindement n'hérite
pas du défaut qu'on est en train de retirer ailleurs.

## Un test qui ne pouvait pas échouer, trouvé en le mutant

La mutation « les deux termes scindés partagent une normalisation commune »
**survivait à tout le fichier** — 23 passed sur 23, aucune rougeur. Les
quatre champs d'essai d'alors sont
**purs** (ω = 0 **ou** J = 0), et sur un champ pur les deux normalisations
coïncident au bit près : `max|ω| + 0 = max|ω|`. Le jeu de champs ne pouvait
pas voir la différence.

Un cinquième champ, **mixte** (ω et J tous deux actifs, pics dans un rapport
> 2), a été ajouté, plus le garde qui vérifie qu'il reste mixte. La mutation
meurt maintenant, ainsi que sa variante « max de la somme ».

**Le nombre publié dans une note antérieure pour cette mutation — « 2 failed »
— était faux**, mesuré avec un `__pycache__` périmé : la mutation remplaçait
`Jz_curl` par `omega_z`, deux identifiants de **même longueur en octets**, si
bien que la clé `(mtime, size)` du `.pyc` ne changeait pas et que Python
rechargeait l'ancien module. Toutes les mutations ci-dessous sont remesurées
caches vidés.

## Vérification

```bash
find . -name __pycache__ -exec rm -rf {} +
python -m pytest tests/mapping/test_selectivite_des_coefficients.py -q   # 30 passed
python -m pytest tests/mapping -q                                        # 436 passed
python -m pytest tests/ -q -m "not slow"          # 5 failed, 3032 passed, 1 h 02
python study/common/aggregate_master_table.py                            # 180 / 176 / 4 / 0
```

**Les 5 rouges sont ceux du dépôt, pas les miens.** Vérifié en rejouant les
cinq sur un `git worktree` propre à `HEAD` : ils échouent à l'identique. Ils
correspondent exactement à la liste documentée — le trio `a0e0e02` (`K_xpoint`
désormais consommé par `build_ising_terms`, en attente du rejeu de phase 4,
T13 et T26) et la paire D-132. Les deux tests intermittents de D-165 sont
passés ce tour-ci, ce qui est précisément ce que D-165 annonce : *« un
décompte pris sur une exécution ne dit pas combien de tests sont rouges ; il
dit combien l'étaient ce jour-là. »*

| test rouge | origine | mien ? |
|---|---|---|
| `test_xpoint_term_absent_from_study.py::test_build_ising_terms_ignores_xpoint` | trio `a0e0e02` | non |
| `test_xpoint_term_absent_from_study.py::test_ablation_zeroes_a_key_nothing_reads` | trio `a0e0e02` | non |
| `test_t13_control_is_not_vacuous.py::test_removed_max_separates_a_real_ablation_from_an_empty_one` | trio `a0e0e02` | non |
| `test_qaoa_noise_and_early.py::test_noise_robustness` | paire D-132 | non |
| `test_qaoa_scaling_and_hparams.py::test_hyperparameter_sweep` | paire D-132 | non |

| mutation de `src/Simulation/HamiltParams_v2.py` | effet |
|---|---|
| `K_current` lit `omega_z` au lieu de `J_z` | 6 failed |
| `K_vorticity` lit `J_z` au lieu de `omega_z` | 6 failed |
| les deux termes scindés partagent une normalisation commune | 2 failed |
| … normalisés par le max de la **somme** (celle de la plaquette) | 2 failed |
| `K_plaquettes` change de formule (×0,5) | 5 failed |
| la plaquette devient sélective (`\|ω\|` seul) | 8 failed |
| `legacy` divise par le max (les deux modes se confondent) | 2 failed |
| `max` cesse de borner ZZ | 2 failed |
| `K_xpoint` : `max(0,−det)` → `\|det\|` | 1 failed |
| **le scindement redevient le défaut** (le défaut corrigé ci-dessus) | 4 failed |
| le drapeau est ignoré, les clés sont toujours là | 4 failed |
| le drapeau est ignoré, les clés n'y sont jamais | 23 failed |

## Ce que ça ne dit pas

Que le scindement **améliore** quoi que ce soit. Il rend deux coefficients
sélectifs là où il y en avait un qui ne l'était pas — c'est une propriété de
la **forme** des coefficients, mesurée sur des champs analytiques. Savoir si
un hamiltonien qui les consomme décide mieux est une question de campagne, et
**aucun résultat d'avant campagne ne peut y répondre**.

---

# La moitié du terme ZZZZ était numériquement morte sur 2 scénarios sur 4

Demande de USER : *« sommer les deux grandeurs adimensionnelles séparément.
Mais il ne faut pas faire différentes portes ZZZZ (c'est coûteux et ça sert à
rien). Pour `norm` fais ce qui semble le plus juste. »*

## Le défaut, mesuré sur le corpus avant d'être corrigé

La plaquette valait `(|ω| + |J|) / (max|ω| + max|J|)` : deux magnitudes
**brutes** sous un dénominateur **commun**. Le signal le plus fort y écrase
l'autre en proportion de son amplitude. Sur les champs MHD du dépôt, cette
proportion n'est pas une correction du second ordre.

**Poids effectif de la vorticité dans la somme**, N=256, Re=400, 12
instantanés par scénario (min – médiane – max) :

| scénario | poids de \|ω\| | rapport max\|J\|/max\|ω\| | lecture |
|---|---|---|---|
| `harris_tearing` | 0,000 – **0,003** – 0,006 | **179** | la **vorticité** ne contribue pas |
| `kelvin_helmholtz` | 0,975 – **0,993** – 1,000 | **84** | le **courant** ne contribue pas |
| `mhd_rotor` | 0,193 – 0,391 – 1,000 | 1,7 | équilibré |
| `orszag_tang` | 0,212 – 0,278 – 0,400 | 3,4 | le courant domine 3:1 |

Sur **deux scénarios canoniques sur quatre**, l'une des deux structures que le
terme prétend détecter est absente de son propre coefficient, sur toute la
trajectoire. Ce n'est pas un bug : c'est la formule qui le dit. Un Harris
tearing est une nappe de courant sans écoulement, un Kelvin–Helmholtz un
cisaillement sans champ — chaque scénario est dominé par un type, et le
dénominateur commun transforme ce fait physique en **effacement** de l'autre
signal.

## Ce qui est fait — un seul terme, une seule porte

```python
K_p = -w_ZZZZ * (|ω|/max|ω| + |J|/max|J|) / max(la somme)
```

Chaque magnitude est rendue **adimensionnelle par son propre maximum avant**
la somme. Les deux structures pèsent alors 1/2 chacune quel que soit leur
rapport d'amplitude, et diviser par le max de la somme ramène le pic à
`w_ZZZZ` exactement.

**Une seule famille ZZZZ, donc une seule porte.** Le scindement en deux
coefficients (entre précédente) est retiré : il achetait la même séparation au
prix d'un second jeu de portes.

## Ce que ce choix affirme, et qui est discutable

Que pour **décider où raffiner**, le type de structure compte et le rapport
d'amplitude entre types ne compte pas. C'est une position, pas un théorème.

Ce qui la rend défendable : l'amplitude n'est pas perdue, elle entre par les
deux autres familles — le biais Z est accroché au score classique, le couplage
ZZ à la magnitude du saut. La plaquette, elle, ne dit plus que **où la
circulation est localement forte relativement à son propre type**, ce qui est
l'instinct correct pour un indicateur de raffinement : une structure faible
mais raide a besoin de résolution autant qu'une structure forte et lisse.

## `norm` : le défaut bascule sur `max`

La question posée était ouverte (*« fais ce qui semble le plus juste »*). Trois
faits mesurés, tous dans le même sens, et aucun contre :

1. Sous `legacy`, le poids relatif ZZ:ZZZZ dérive d'un **facteur 2,59** avec la
   seule spikiness du champ (3,121 → 8,095). Sous `max` il vaut
   `W_ZZ/W_ZZZZ = 2,000` partout.
2. Sous `legacy`, `max|K|` vaut 0,74 à 0,99 selon le scénario — la famille ZZZZ
   est silencieusement dévaluée jusqu'à 26 % sur `mhd_rotor`, parce que les
   deux maxima sont pris en des points différents. Sous `max`, `max|K| = 1,000`
   partout.
3. Le défaut ci-dessus (une structure effacée sur 2 scénarios sur 4)
   n'existe que sous `legacy`.

**Rayon d'action, vérifié :** aucun fichier de `src/` ne construit
`PhysicalMapperV2`. Le solveur V1 déployé reçoit son mappeur par injection et
n'est **pas** concerné. Les quatre sites qui le construisent sont tous dans
`study/` — `hamiltonian_coefficients.py`, `sanity_check.py`,
`exact_diagonalisation.py`, `qaoa_inputs.py` — et **aucun ne passait `norm=`
explicitement**. Le basculement change donc exactement le pipeline de la
campagne à venir, ce qui est l'intention.

`legacy` reste joignable et garde une seule raison d'être : **reproduire les
artefacts gelés**, calculés avec. `test_la_formule_legacy_de_la_plaquette_est_inchangee`
recalcule la formule historique indépendamment et rougit si quelqu'un
« améliore » `legacy` aussi — le jour où plus aucun mode ne reproduit le passé,
toute comparaison avant/après devient impossible.

## Ce que le basculement a réveillé — trois faits, mesurés parce que des tests ont rougi

Le basculement fait passer la suite de 5 rouges à 12. Décomposé : les **5
rouges documentés** du dépôt (trio `a0e0e02` + paire D-132) sont toujours là,
**5** viennent du changement, et **2** sont des tests intermittents qui
n'ont rien à voir avec lui. Chacun des cinq a été instruit avant d'être
traité.

### 1. Le balayage plat de D-86 était un artefact de la normalisation

D-86 : sur 52 configurations, **14 balayages `c_bias` plats** — écart max-min
exactement nul, donc `argmax` rendait le bord gauche de la grille comme
« optimum ». Rejoué sur le cas mesuré (`harris_tearing`, N=96, dim=4, 8
instantanés, graine 0) dans les deux normalisations :

| `norm` | `f1_span` (4 Re) | dégénérés | `c_bias*` |
|---|---|---|---|
| `legacy` | 0,0000 / 0,0000 / 0,0000 / 0,0000 | **4 / 4** | 0,100 (bord **gauche**) |
| `max` | 0,5655 / 0,5583 / 0,5500 / 0,5583 | **0 / 4** | 75–100 (bord **droit**) |

Sous `legacy`, le hamiltonien de champ moyen ne séparait **rien** sur cette
configuration : F1 identiquement nul sur les 25 points de grille. Sous `max`,
il sépare. **L'instrument avait perdu sa résolution**, et c'est la
normalisation qui la lui retirait.

### 2. …et l'optimum du balayage corrigé est **au bord droit** — D-86 en miroir

`c_bias*` vaut 100,0 sur 3 Re sur 4, soit le sommet de `logspace(-1, 2)`. Un
optimum au bord n'est pas un optimum, exactement comme le bord gauche ne
l'était pas. Mesuré en élargissant à `logspace(-1, 5)` :

    F1 sature à 0,6333 dès c_bias ~ 251 ; les six derniers points identiques

L'optimum est donc la **limite biais seul** — les couplages n'apportent rien
de positif — et il reste **sous la baseline classique (0,745)**. Épinglé, non
corrigé : élargir la grille déplace tous les `c_bias*` publiés. Entrée
`DEFAUTS.md`.

### 3. La dégénérescence de D-45/D-47 à `dim = 2` tient — ma première lecture était fausse

`test_the_ground_state_is_uniform_on_real_deployed_coefficients` est devenu
rouge : sous `max`, l'état fondamental exact à `dim = 2` n'est plus uniforme.
Lu vite, cela annonçait la fin de la dégénérescence qui vide la réfutation de
H0.

**Non.** Le champ de ce test est du **bruit gaussien**, pas un champ DNS — son
nom désignait le chemin de code, pas les données. Remesuré sur 40 instantanés
DNS réels (4 scénarios × 10, N=256, patch 4×4, `dim = 2`) :

| `norm` | états fondamentaux uniformes | non uniformes |
|---|---|---|
| `legacy` | 39 / 40 (**97,5 %**) | 1 |
| `max` | 36 / 40 (**90,0 %**) | 4 |

La dégénérescence **bouge et ne tombe pas**. D-45/D-47 tient. Les deux taux
sont désormais épinglés par un test `slow` qui rougit si l'un sort de sa
fourchette — c'est-à-dire le jour où la conclusion sur H0 à `dim = 2` doit
être réécrite.

### Un troisième test intermittent, non documenté

`tests/mapping/test_signal_contribution.py::test_C_ZZ` est rouge en suite
complète et **vert 3 fois sur 3 en isolé**. Il construit son hamiltonien à la
main et n'appelle jamais le mappeur : il ne peut pas être affecté par ce
changement. C'est un troisième membre de la famille D-165, dans le même
fichier que `test_K_ZZZZ` déjà documenté. Entrée `DEFAUTS.md`.

### Deux tests dont la GRANDEUR a cessé de discriminer

`test_noise_weaker_than_anomaly` comparait le **pic** de `|C|` entre un champ
de bruit et un champ à anomalie. Sous `max` le pic vaut `w_zz = 2,000` sur les
deux : l'assertion comparait deux nombres identiques. Règle VIGIL — *si la
grandeur n'est pas discriminante, changer de grandeur, pas de seuil*. Le fait
physique visé reste vrai et reste mesurable par une grandeur de **forme** :

| statistique | bruit | anomalie | `legacy` | `max` |
|---|---|---|---|---|
| pic \|C\| | 4,098 / **2,000** | 32,000 / **2,000** | discrimine | **ne discrimine plus** |
| fraction du domaine ≥ ½ pic | **0,684** | **0,125** | discrimine | discrimine |

La nouvelle statistique rend **le même nombre dans les deux modes** — elle ne
dépend plus du choix de normalisation.

`test_legacy_est_le_defaut` a rougi comme prévu : c'est le garde qui exige
qu'un basculement se décide. Renommé, il exige maintenant la même chose dans
l'autre sens.

## Ce que ça déplace

| grandeur | `legacy` | `max` |
|---|---|---|
| `max\|K\|` harris / KH / rotor / OT | 0,9947 / 0,9902 / 0,7418 / 0,8494 | **1,0000** partout |
| cellules changeant d'appartenance au top 5 % de la carte | — | 7,8 % / 4,5 % / 0,5 % / 1,6 % |

Les artefacts `.npz` gelés ne bougent pas — ils ne sont pas recalculés — donc
la table maîtresse reste à **180 / 176 / 4 / 0**. Mais le Hamiltonien du dépôt
n'est plus celui qui les a produits : **toute campagne relancée produira des
nombres différents**, et c'est le but.

## Une erreur de conception de mes propres tests, trouvée en les faisant échouer

Les premiers champs « mixtes » plaçaient le tourbillon et la nappe de courant
**au même endroit**. Sur un tel champ, aucune structure ne peut en écraser une
autre : les deux formules coïncident, et trois tests qui prétendaient mesurer
l'écrasement mesuraient en fait `1,0000` des deux côtés. Les champs d'essai
séparent désormais les deux structures (recouvrement **3,7e-07**), ce qui est
aussi le cas physique honnête — dans un écoulement réel, le vortex est ici et
la nappe est là.

## Vérification

```bash
find . -name __pycache__ -exec rm -rf {} +
python -m pytest tests/mapping/test_selectivite_des_coefficients.py -q  # 25 passed, 1 skipped
python -m pytest tests/mapping -q                                # 432 passed, 1 skipped
python -m pytest tests/ -q -m "not slow"      # 6 failed, 3029 passed, 1 h 07
python study/common/aggregate_master_table.py            # 180 / 176 / 4 / 0
```

**Les 6 rouges finaux** : les **5 documentés** du dépôt (trio `a0e0e02`, paire
D-132), rejoués sur un `worktree` propre à `HEAD` lors de la passe précédente
et rouges à l'identique, **plus un intermittent** —
`test_optimiser_axis.py::…[L-BFGS-B]`, vert 3/3 en isolé et 2/2 sur son
fichier, opérateur codé en dur, aucun lien avec le mappeur (D-187).

La table maîtresse reste à **180 / 176 / 4 / 0** : les artefacts `.npz` ne
sont pas recalculés, donc aucun nombre publié ne bouge — mais le Hamiltonien
du dépôt n'est plus celui qui les a produits.

| mutation de `src/Simulation/HamiltParams_v2.py` | effet |
|---|---|
| `max` revient au dénominateur commun (la formule d'avant) | 3 failed |
| la vorticité n'est pas normalisée, le courant si | 2 failed |
| `legacy` est « amélioré » lui aussi (plus de mode de reproduction) | 5 failed |
| le pic de la somme n'est plus retiré (`max\|K\| ≠ w`) | 4 failed |

Un test `@pytest.mark.slow` rejoue le fait sur les vrais artefacts DNS et
rougit le jour où les scénarios du dépôt cessent d'être dominés par une seule
structure — auquel cas la justification du changement tombe et il faut la
réécrire, pas retoucher le seuil.

---

# Vérité terrain dynamique : elle existe, et elle dit que l'horizon du protocole est trop court

Protocole v3 §1.2, tâche 6 — le seul « heavy new code » du protocole, et le
seul artefact qu'aucune campagne n'avait jamais produit : `d_patches_*`
comptait **0 fichier**.

## Ce que le label calcule

Pour un instantané `t` et un patch `i` : le champ de référence évolué de
`δt`, contre le même champ où **le patch `i` seul** a été remplacé par sa
moyenne, évolué de `δt` avec **la même séquence de pas**. `d_i` est la
distance L2 entre les deux, **sur le champ entier** — c'est tout l'intérêt :
`e_i` est confiné au patch, `d_i` compte ce que l'erreur abîme ailleurs.

## Le gel de la séquence de pas n'est pas une précaution théorique

`dns_sweep.py` appelle `adapt_dt()` à chaque pas. Si chaque variante adaptait
la sienne, `d_i` compterait un écart de **pas de temps** comme de la
physique. Mesuré (N=96, Re=400, δt=0,05) — fraction des patches dont la
variante adapterait une séquence différente :

| scénario | dim=4 | dim=8 |
|---|---|---|
| `harris_tearing` | 0 / 16 | 0 / 64 |
| `kelvin_helmholtz` | **16 / 16** | **64 / 64** |
| `mhd_rotor` | **16 / 16** | **64 / 64** |
| `orszag_tang` | **16 / 16** | **64 / 64** |

Sur trois scénarios sur quatre, **chaque** patch divergerait. Le mécanisme
n'est pas l'extremum initial — `adapt_dt` lit des maxima globaux, qu'un patch
grossi ne déplace pas — mais l'évolution, qui les déplace dès le premier pas.

## Le résultat : à l'horizon du protocole, `d` est une redite de `e`

N=96, Re=400, `dim=8` (64 patches), **5 instantanés** par scénario, relus
**depuis les 8 artefacts committés**. ρ est le Spearman entre le label
dynamique et le label statique ; l'amplification est `d_i / d0_i`, où `d0_i`
est la perturbation **avant** toute évolution.

| scénario | ρ(d,e) à δt=0,1 | ρ(d,e) à δt=2,0 | amplif. δt=2,0 (méd. / p90) |
|---|---|---|---|
| `harris_tearing` | **+1,0000** | **+1,0000** | 0,58 / 0,66 |
| `kelvin_helmholtz` | +0,9970 | +0,9927 | 0,82 / 1,03 |
| `mhd_rotor` | +0,9917 | +0,9297 | 0,58 / 1,43 |
| `orszag_tang` | +0,9817 | **+0,5961** | **1,38 / 2,06** |
| **moyenne** | **+0,9926** (min +0,9817) | +0,8796 (min +0,5961) | |

> Une première version de cette table publiait des nombres pris sur **3**
> instantanés choisis à la main (ρ = 0,714 pour OT à δt=2,0, moyenne 0,9954 à
> δt=0,1). Ils sont remplacés par ceux que rendent les **artefacts
> committés** — 5 instantanés répartis sur toute la trajectoire, relisibles
> par une commande. L'écart est réel sur les deux scénarios les plus agités
> (rotor 0,985 → 0,9297 ; OT 0,714 → 0,5961) et ne change pas la lecture : il
> la renforce. Publier une mesure qu'aucun artefact ne reproduit est
> exactement ce que la table maîtresse existe pour empêcher.

**À `δt = 0,1` — la valeur que le protocole impose — ρ ≥ 0,98 sur les quatre
scénarios.** Le label dynamique est une renumérotation monotone du label
statique. Il ne répond donc **pas** au problème de spécification de tâche
(H5) pour lequel il avait été demandé.

Le contrôle d'acceptation du protocole, *« sanity check Spearman(d_i, e_i) >
0 reported »*, est **satisfait** — et c'est exactement le problème : un
contrôle qu'un label redondant passe haut la main ne contrôle rien.

## Pourquoi, et ce n'est pas un accident numérique

Deux mécanismes, tous deux mesurés.

**1. La perturbation n'a pas quitté son patch.** Temps de traversée
`t_x = (largeur du patch) / (v_rms + b_rms)`, à `dim = 8` :

| scénario | v+b (rms) | `t_x` | ce que vaut δt=0,1 |
|---|---|---|---|
| `harris_tearing` | 0,893 | 0,880 | **0,11 t_x** |
| `kelvin_helmholtz` | 1,062 | 0,739 | **0,14 t_x** |
| `mhd_rotor` | 1,684 | 0,466 | **0,21 t_x** |
| `orszag_tang` | 1,926 | 0,408 | **0,25 t_x** |

À δt = 0,1 la perturbation parcourt **un dixième à un quart** d'une largeur de
patch. Il n'y a rien à propager — donc rien que `d` puisse dire de plus que
`e`. L'ordre des scénarios suit exactement celui des ρ : harris parcourt le
moins (0,11 t_x) et rend ρ = 1,0000 **exactement**.

**2. L'amplification est quasi constante d'un patch à l'autre.** Comme
`d0_i = e_i / dim` exactement (identité démontrée et épinglée), si le facteur
`d_i/d0_i` ne varie pas entre patches alors `d = constante × e` et ρ = 1 par
construction. À δt=0,1, médiane 0,88 et p90 0,88–1,01 : la dispersion est
nulle. **C'est la dispersion de l'amplification, pas sa valeur, qui décide si
le label dit quelque chose.**

## Ce qui décolle, et où

Un seul scénario sort du lot à δt=2,0 : **`orszag_tang`**, ρ = **0,596**, et
le seul dont la perturbation **amplifie** (1,38 médian, 2,06 au p90) au lieu
de décroître. C'est aussi le seul scénario où le label statique n'était **pas**
quasi gratuit — AUC du score classique seul 0,592, contre 1,000 / 0,997 /
0,948 pour les trois autres.

Les deux faits se tiennent : **là où l'écoulement est assez turbulent pour
propager et amplifier une perturbation, le label dynamique dit autre chose —
et c'est précisément là que la tâche statique n'était pas déjà résolue.**

## Ce que ça change pour le protocole

**L'horizon δt = 0,1 (« one hybrid step ») est à réviser.** Le critère
défendable n'est pas un nombre de pas hybrides mais une **échelle physique** :
`δt ≳ t_x = 2π / (dim · (v+b)_rms)`, soit 0,41 à 0,88 à `dim = 8` et 0,82 à
1,76 à `dim = 4`. C'est ce que la mesure à δt=2,0 confirme.

Ce que ça **ne** dit pas : que le label dynamique sauve H5. Sur trois
scénarios sur quatre il reste à ρ ≥ 0,985 même à δt=2,0. Le problème « la
tâche est quasi gratuite » n'est levé que sur `orszag_tang`. Produire ce
label était nécessaire — c'était le seul moyen de le savoir — mais il ne
suffit pas.

## Coût, mesuré

N=96, `dim=8`, 65 évolutions par instantané :

| scénario | δt=0,1 | δt=2,0 |
|---|---|---|
| `harris_tearing` | 3,1 s | 57 s |
| `kelvin_helmholtz` | 5,4 s | 90 s |
| `mhd_rotor` | 19 s | 258 s |
| `orszag_tang` | 9,6 s | 208 s |

Projection N=256 (échelle N³ : N² cellules × N pas) : **× 19**, soit ~1 à 80
min par instantané selon le scénario et l'horizon. Le label est donc
abordable à la résolution de production — ce que le protocole demandait de
vérifier avant de lancer (« report wall-clock … and project the full cost
before launching N=256 »).

## ⚠️ Deux corrections trouvées le lendemain, en relisant ce module

**1. Le seuil était calculé par instantané.** `hard_patch_labels.py` aplatit
`all_l2` **sur les instantanés** avant son percentile et rend **un scalaire**.
Ce module le calculait **par instantané**, ce qui force exactement (100−p) %
de patches durs dans *chaque* instantané : un instantané calme et un
instantané turbulent auraient eu la même proportion de patches durs. Le module
annonçait pourtant reproduire la phase 2.

Corrigé (`seuil_global`) ; `d_threshold` est désormais un scalaire et
`hard_fraction_par_instantane` est publiée à côté. **Les 8 artefacts ont été
régénérés.** Les ρ et amplifications publiés ci-dessus ne dépendent pas du
seuil et ne bougent pas.

**2. Une mutation survivait au fichier entier.** « La variante adapte son
propre pas » — c'est-à-dire le débranchement de la séquence gelée, la raison
d'être du module — passait tous les tests. `sequence_de_pas` était testée
isolément, `evolue` aussi, et le fait que les deux séquences diffèrent
également ; **rien ne testait que `dynamic_patch_errors` emploie réellement la
séquence gelée**. Les pièces étaient gardées, l'assemblage ne l'était pas.

## Déviations au protocole, assumées

| point | protocole | ici | pourquoi |
|---|---|---|---|
| chemin | `study/v3/t6_dynamic_gt.py` | `study/pipeline/dynamic_patch_labels.py` | le dépôt a été réorganisé ; `phase2_hard_patches.py` est aujourd'hui `pipeline/hard_patch_labels.py` |
| pilote | N=128 | **N=96** | aucun artefact DNS N=128 dans le dépôt (N ∈ {64, 96, 256}) |
| format | « drop-in » : le label dynamique sous la clé `l2_errors` | clés explicites `d_errors` / `d0_errors` ; `l2_errors` reste le label **statique** | un artefact de la forme phase 2 dont `l2_errors` désigne autre chose est la classe de défaut que `CODE_REVIEW.md` retient comme la seule qui compte |
| nom | — | `δt` dans le nom du fichier | c'est le paramètre qui décide si le label dit quelque chose ; deux horizons partageant un nom s'écraseraient en silence |

## Vérification

```bash
# produire les 8 artefacts (4 scenarios x 2 horizons)
for sc in harris_tearing kelvin_helmholtz mhd_rotor orszag_tang; do
  for dt in 0.1 2.0; do
    python study/pipeline/dynamic_patch_labels.py \
        --scenario $sc --re 400 --N 96 --dim 8 --snaps 5 --delta-t $dt
  done
done

# relire les nombres publies DEPUIS les artefacts
python - <<'EOF'
import glob, os, numpy as np
for f in sorted(glob.glob("results/d_patches_*.npz")):
    z = np.load(f, allow_pickle=True)
    a = z["amplification"]; fini = np.isfinite(a)
    print(f"{os.path.basename(f):52s} rho={z['rho_d_vs_e'].mean():+.4f} "
          f"amp={np.median(a[fini]):.2f}")
EOF

python -m pytest tests/study/test_dynamic_patch_labels.py -q   # 20 passed, 3 deselected
python -m pytest tests/mapping -q                             # 437 passed, 1 skipped
python -m pytest tests/study/test_dynamic_patch_labels.py -q -m slow   # 3 passed
python -m pytest tests/ -q -m "not slow"    # 5 failed, 3053 passed, 55 min
```

Les **5 rouges sont les 5 documentés du dépôt** (trio `a0e0e02`, paire
D-132), rejoués sur un `worktree` propre à `HEAD` lors d'une passe
antérieure. **Aucun intermittent ce tour-ci** — les quatre de D-165/D-187
sont tous passés — et le module neuf ne casse rien.

L'identité `d0 = e / dim` est vérifiée à `rtol=1e-12` pour dim ∈ {2, 4, 8} :
elle teste le grossissement et la normalisation **sans passer par le
solveur**, et aucune tolérance ne pourrait la masquer.

**Un test à moi qui a mordu.** Le test « la variante adapterait une autre
séquence » a d'abord échoué, sur un champ où il ne pouvait pas réussir :
`adapt_dt` lit des maxima **globaux**, que grossir un patch quelconque ne
déplace pas au premier pas. La première version concluait donc que le gel ne
servait à rien — faux, et la mesure sur les vrais champs le dit (100 % des
patches sur 3 scénarios sur 4). Le champ d'essai place désormais l'extremum
global **dans** le patch grossi.
