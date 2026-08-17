# Couverture

**Ce qu'on teste, comment, et pourquoi.** Sert à savoir ce qui est
reproductible et validé — et ce qui ne l'est pas encore.

Deux mesures distinctes, à ne pas confondre :

| | ce que ça mesure | comment |
|---|---|---|
| **couverture de ligne** | les lignes jamais exécutées | `coverage`, automatique |
| **audit de contrat** | les fonctions dont on n'a jamais vérifié qu'elles font ce qu'elles annoncent | à la main, fonction par fonction |

**La seconde compte davantage.** Le module le plus défectueux de V1 était
couvert à **100 %** : ses tests vérifiaient des valeurs, partageaient le
modèle mental du code, donc son erreur. Un module à 95 % peut être un piège ;
un module à 60 % peut être sain.

**1 971 tests**, 77 fichiers. Commandes dans `tests/README.md`.

---

## 1. Ce qui n'est pas couvert — la liste qui dit quoi faire

### Lignes jamais exécutées

Remesuré le 13 août, **suites QAOA comprises** — seules les mesures `slow`
sont exclues. La mesure précédente les excluait aussi, ce qui rendait faux
par construction les chiffres de `pipeline.py`, `call_vqa_shell.py` et
`solver.py`. **Les deux séries ne sont pas comparables** : ce qui a bougé,
c'est le périmètre de mesure, pas la couverture. 36 min.

```bash
python -m coverage run --source=src -m pytest tests/ -q -m "not slow"
python -m coverage report --include="src/*"
```

**Le chemin scientifique déployé — 85 %** (2 658 instructions) :

| module | couverture | ce qui manque |
|---|---|---|
| `cost_hamiltonian`, `init_qbits_state`, `mapping`, `call_vqa_shell`, `HamiltParams_v2`, `PhysToAngle` | **100 %** | — |
| `Simulation/solver.py` | **99 %** | 2 instructions |
| `HamiltParams.py` | **99 %** | 2 instructions |
| `VQA/runtime.py` | **98 %** | 1 instruction |
| `pre_compute_dns.py` | **98 %** | 1 instruction |
| `RescaleArrays.py` | **97 %** | 3 instructions |
| `VQA/postprocess.py` | **95 %** | 1 instruction |
| `train_hyperparams.py` | **90 %** | le mode Colab (non exécutable ici), l'analyse des CSV de rescore en erreur |
| `Simulation/grid.py` | **90 %** | 13 instructions |
| `VQA/optimize.py` | **88 %** | les backends autres que `state_vector` |
| `Simulation/refinement.py` | **82 %** | le sondage de bord, la reprise de campagne |
| `Simulation/utils.py` | **65 %** | `slice_hamiltonian_params`, non appelé par le chemin déployé |
| `VQA/execute.py` | **64 %** | la boucle COBYLA à plusieurs redémarrages, les chemins hérités |
| `hyperparams_loader.py` | **55 %** | les sélecteurs par scénario, combo, phase, rang |
| `pipeline.py` | **52 %** | le corps de `main()` et sa CLI — le module s'exécute par sa bibliothèque, pas par sa ligne de commande |

**Le reste de `src/` — 0 à 12 %**, et ce sont exactement les cinq fichiers
jamais audités : `analyze_hyperparams.py`, `recompute_lambda_scores.py`,
`compare_rotor_budget.py` et `import_Neon_data_to_local.py` à **0 %**,
`visual.py` à 12 %, `help_visual.py` à 7 %.

*Chiffres déjà dépassés pour `recompute_lambda_scores.py` et
`analyze_hyperparams.py` : les 13 tests de D-49 et les 11 de D-50 sont
postérieurs à cette mesure. À remesurer à la prochaine passe — noté ici
plutôt que corrigé au jugé, parce qu'un nombre non mesuré n'a pas sa place
dans ce tableau.*

**Total sur tout `src/` : 56 %** (4 102 instructions). Ce chiffre unique
cachait deux populations. **1 429 des 1 822 instructions jamais exécutées —
78 % — sont dans les cinq fichiers jamais audités.** Les deux listes
coïncident : ce qui n'est pas relu n'est pas exécuté non plus. C'est la
raison de fond pour finir V1 avant de réoptimiser.

### Fonctions dont le contrat n'a jamais été audité

Pas mesurable automatiquement. Liste tenue à la main, depuis ce qui a été
effectivement relu fonction par fonction.

**Jamais audité** — aucune des cinq questions n'y a été posée :

Vide. `compare_rotor_budget.py` (481 lignes) figurait ici — « ne s'exécute
pas » était déjà faux au moment où c'était écrit (corrigé par D-10 le
13 août, avant cette note) : lu en entier cette passe, voir plus bas — D-91.

`visual.py` et `help_visual.py` figuraient ici (327 lignes, « figures ») :
lus en entier le 14 août, voir §1a quater — D-68.

`analyze_hyperparams.py`, `recompute_lambda_scores.py` et
`import_Neon_data_to_local.py` figuraient ici. Ils ont été audités **deux
fois le 13 août, par deux passes qui s'ignoraient** : D-49 et D-50 d'abord
(chemin de données et chemin d'échec), puis D-60 à D-65 (les figures et le
chemin de sortie). La seconde a re-trouvé D-49 sous le numéro D-63 — voir
« Deux passes sur les mêmes fichiers » ci-dessous.

**~810 lignes restantes**, en aval du chemin scientifique : elles lisent des
résultats, elles n'en produisent pas. Réserve apprise à la seconde passe :
« en aval » ne veut pas dire « sans conséquence ». `import_Neon_data_to_local.py`
fait 76 lignes, ne calcule rien — et c'est le seul code du dépôt qui peut
détruire les bases de la campagne (D-64).

Les deux fichiers qui **décident la lecture** des nombres publiés ont été
audités le 13 août — chemin de données et chemin d'échec, pas la mise en
page :

| fichier | lignes | verdict |
|---|---|---|
| `recompute_lambda_scores.py` | 717 | cœur **sain** — à `lambda` égal il reproduit `trial.value` à **2,2e−16** sur les 303 essais réels, classement identique. **D-49** : son chemin d'échec rendait 0 |
| `analyze_hyperparams.py` | 918 | **D-50** : chemin d'échec à 0, message accusant une base distante. Détection de scénarios qui en annonçait **7** pour **4** — corrigée, sortie prouvée identique au SHA-256 |

`TrainHyperParam_v1/v3/v4.py` (1 641 lignes) figuraient ici. **Supprimés** :
quatre variantes du même script d'entraînement coexistaient sans qu'aucune ne
soit désignée. `TrainHyperParam_v2.py` est renommé `train_hyperparams.py`,
audité fonction par fonction, et couvert par 67 tests — voir D-27 à D-36 dans
`RESULTS.md`.

**Partiellement audité** — le contrat a été vérifié sur une partie des
fonctions seulement :

| fichier | ce qui reste |
|---|---|
| `recompute_lambda_scores.py` | ses six fonctions de tracé — chemin de données audité, voir D-49 |
| `analyze_hyperparams.py` | ses treize fonctions de tracé — chemin de données audité, voir D-50 |
| `study/` | **en totalité** — c'est le chantier suivant |

Les quatre poches partielles de V1 (`Simulation/refinement.py`, le mode Colab
de `train_hyperparams.py`, les branches matériel de `VQA/execute.py`, le bras
`classical_only` de `pipeline.py`) ont été fermées le 13 août — section
suivante.

**Audité le 12 août, sur le chemin d'entraînement** — parce que ces
fonctions décident le nombre qu'une campagne d'une semaine minimise :

| fonction | verdict |
|---|---|
| `_prepare_vqa_input` | **D-37** : le biais Z et les couplages décrivaient deux grilles différentes à toute profondeur > 0 |
| `execute`, boucle et bornes | **D-38** : trois gardes qui ne tenaient que sur le chemin habituellement testé |
| `pipeline.score`, `weighted_relative_error` | **sain** — 0 sur une reconstruction exacte, 1 quand le bras rend zéro ; pondération construite sur la référence, donc identique aux deux bras |
| comptabilité de pixels des deux bras | **saine** — même `step_layered`, même profondeur, accumulation à chaque pas |
| réduction du score, classique contre quantique | **saine** — écart **0,000e+00** aux profondeurs 0 et 1 |
| `_run_level_classical` contre `_run_level` | **sain** — bloc de décision identique, correction D-16 comprise |
| garde CFL (`check_cfl > 1.0`) | **sain** — marge mesurée **2,5×** sur les six scénarios |

**Lu en entier le 13 août, `study/pipeline/`** — parce que c'est là que le
verdict « le hamiltonien trouve-t-il les patchs durs » se fabrique.

| fichier / fonction | verdict |
|---|---|
| `hamiltonian_coefficients.find_optimal_threshold` | **D-43** : sur une énergie constante, rendait le F1 tout-positif `2p/(p+1)` — 0,400 / 0,376 — comme un pouvoir de séparation |
| `sanity_check.run_qaoa`, drapeau de convergence | **D-44** : mesurait `np.std(marg)`, pas la distance à 0,5 que son commentaire annonce ; verdict inversé aux deux extrêmes |
| `hard_patch_labels.patch_classical_scores`, `Jz` écrit à la main | **sain dans le code, périmé sur 84 artefacts** — le stencil `Jz` et sa convention d'axes sont bien **identiques** à `solver.get_fluxes` (`AXIS_X=0`, centré, `/2dx`), vérifié à l'octet. Mais la conclusion « pas d'opérateur dépareillé entre le score classique des artefacts `patches_*` et celui du chemin coefficients » ne valait que du **code** : le rotationnel *interne* à `classical_score` a changé à D-1, et **84 des 156 fichiers `patches_*` n'ont pas été regénérés depuis** — leur `classical_scores` ne se reproduit plus (jusqu'à 3,8e−01), 50 d'entre eux sont reproduits bit à bit par `fixed_curl=False`. Les `l2_errors` du même fichier, eux, se reproduisent (9,4e−12). **D-77**, rapport seul — `pytest tests/study/test_patches_classical_score_provenance.py` |
| `hard_patch_labels.coarsen_field`, `patch_l2_errors` | **sain** — moyenne de bloc exacte, normalisation RMS interne à l'instantané |
| réduction en patchs : `E_all` par **moyenne**, `score_all` par **max** | **choix de conception, non corrigé** — le max reproduit la décision AMR de production (un patch chaud suffit), la moyenne une densité d'énergie. Écart non mesuré ; à trancher si une comparaison E-contre-score devient un résultat publié |
| `pipeline_verification.analyze`, reconstruction des `snap_indices` | **saine** — vérifié sur les 59 couples `dns_*` / `patches_*` de `results/` : `n_snaps` identique des deux côtés partout, donc E et `is_hard` restent appariés |
| `dns_extension.mean_sq_current_fixed` contre `dns_validation.mean_sq_current` | **sain** — les deux omettent `/dx` de la même façon : la correction porte sur la convention d'axes seule, et le `dx²` commun s'annule dans les rapports que les checks utilisent |
| `labels_global_threshold.py`, `labels_error_tolerance.py` | **sains** — les deux refusent explicitement un seuil ou une tolérance qui dégénère, et un balayage vide y crie |
| `exact_diagonalisation.py` | `analyze_snapshot` reçoit `is_hard` sans jamais s'en servir (recalculé à l'identique depuis `l2_threshold`) et `H_mat` est diagonale, donc `eigh` est un argmin coûteux — deux gaspillages, pas des valeurs fausses. La clause `promising = f1_exact >= f1_classique` (le commentaire au-dessus dit `>`) est désormais **mesurée** : phase 4 rejouée depuis les DNS et les `patches_*` faute d'artefact `exact_diag_*`, elle rend **40/40** avec `>=` et **0/40** avec `>` parce que les deux prédicteurs comparés sont **constants** — **D-45** (annonce de la dégénérescence, corrigé) et **D-47** (sa cause, ouvert) |
| `exact_diagonalisation.n_energies_below_gap` | **défaut sans conséquence, non corrigé** : `sum(energies < E0 + gap)` avec `gap = E1 − E0` vaut 1 quand le fondamental est non dégénéré et **0** quand il l'est — l'inverse de ce que le nom annonce. Aucun appelant ne le lit et il n'entre dans aucun artefact : signalé ici, pas corrigé, pour ne pas mélanger une correction sans mesure aux deux qui en ont une |
| `label_percentile_sensitivity.py`, message d'interprétation | **D-46** : imprimait « ROBUST … fails for ANY » dès `max(deltas) < 0,05`, alors que le docstring définit la robustesse comme « le gap ne devient jamais positif » (`delta < 0`) — mesuré sur l'artefact réel (dim=4, N=256, 4 scénarios, seed 0) `max(delta) = -0,154`, sous les deux seuils : ce run-ci n'était pas affecté, mais un cas construit (`+0,03` à un percentile) montrait le verdict « ROBUST » imprimé malgré un delta positif |
| `dns_extension.py`, seconde moitié (`_band_limited_noise`, `perturb_fields`, `energy_non_increasing`, `presence_matrix`, `validate_one`, `main`) | **sain** — filtre passe-bas spectral en convention `AXIS_X`/`AXIS_Y` correcte (`indexing='ij'`), tolérance de monotonie d'énergie (1e-3) identique à celle de `check_ot` côté `dns_validation.py`. `check_kh_fixed` (ici) et `dns_validation.check_kh(analyse_one(...))` sont **bit-à-bit identiques** sur `kelvin_helmholtz` Re400 N96 et N256 réels — redondant depuis le rebranchement D-21 d'`analyse_one` sur les observables corrigées, pas une divergence de valeur. Les docstrings de `fluctuating_ke_fixed`/`mean_sq_current_fixed` (« phase 1b reste intouchée, réparation côté v3 par copie ») décrivent l'état pré-D-21 et sont datées sans être fausses : les *fonctions* gelées `mean_sq_current`/`fluctuating_KE` restent inchangées, comme `DEFAUTS.md` le documente déjà |

**Ce module n'est pas « audité » au sens de la fiche.** Il a été **lu en
entier**, fonction par fonction ; aucun test ne traverse encore ses axes
(profondeur AMR, bord du patch, bras, backend, warm start, hamiltonien nul,
optimiseur). Lu en entier et non traversé : `dns_sweep.py`, `config.py`.
`label_percentile_sensitivity.py` a désormais un test sur son message
d'interprétation (D-46) mais pas sur `build_percentile_dataset` /
`loso_site_vs_class` eux-mêmes. `dns_extension.py` est maintenant lu en
entier. Non relu cette passe : `dns_validation.py` (passe concurrente,
D-42).

**Axes empruntés par `exact_diagonalisation.py`** (D-45,
`tests/study/test_exact_diag_degenerate_gate.py`) : bras **quantique**,
backend **state_vector** (diagonalisation exacte, aucun échantillonnage),
hamiltonien **non nul**, bord **périodique** (`create_period_hamiltonian`),
warm start **absent**, optimiseur **aucun** (pas de variationnel), AMR
**depth = 0**. Les côtés opposés de chaque axe restent non traversés — et
`dim` ne l'est que par **2** : `VQA_DIMS = [2, 4, 8]`, mais 4 et 8 demandent
32 et 128 qubits contre le plafond de 20 codé dans `exact_diag`, donc les
deux tiers de la configuration déclarée du module ne s'exécutent pas.

---

## 1a. `src/analyze_hyperparams.py` — lu en entier

918 lignes, jamais auditées jusqu'ici. C'est le seul module qui relit les
bases Optuna gelées : ce qu'il montre est ce qu'on sait de la campagne d'une
semaine qu'on ne relancera pas.

| fonction | verdict |
|---|---|
| `plot_threshold_operating_curve` | **D-60** — la figure d'arbitrage précision/coût du bras classique ne pouvait sortir d'aucune étude, pour trois raisons indépendantes (nom de paramètre, schéma d'attributs, garde de l'appelant), et sans un mot |
| `_add_trend` | **D-61** — dernière classe ouverte à droite : l'essai portant la plus grande valeur du paramètre n'entrait dans aucune médiane |
| `_pareto_front` | **sain** — croisé contre une implémentation de référence écrite séparément, **300/300** identiques sur des tirages arrondis à 1e−2, donc pleins d'ex aequo, le cas où la clause de non-domination stricte pourrait mordre |
| `load_study`, filtre des essais | **sain** — `t.value < inf` écarte aussi les `NaN` (`nan < inf` est `False`), ce que le nom ne dit pas mais que le contrat demande |
| `generate_summary`, perte composite | **sain** — la moyenne des `loss_<scenario>` qu'il imprime coïncide avec `t.value` à **5,6e−17** (quantique, 178 essais) et **2,2e−16** (classique, 125) : le résumé et l'objectif rendent bien la même grandeur |
| `plot_scenario_correlation_heatmap`, `plot_field_correlation_heatmap` | **sains** d'orientation — lignes = scénarios/champs, colonnes = paramètres, `ax.text(j, i)` assorti à `imshow`. Réserve non corrigée : un écart-type nul laisse la case à **0,00**, indiscernable d'une corrélation mesurée nulle. Aucun paramètre gelé n'apparaît dans `trial.params`, donc le cas ne se produit sur aucune base du dépôt |
| `plot_scenario_breakdown_bar` | **réserve, non corrigée** — `user_attrs.get(f"loss_{key}", 0)` : un scénario sans perte enregistrée s'empilerait à hauteur nulle, comme s'il n'avait rien coûté. Mesuré : les deux bases ne portent qu'**un seul jeu de clés** sur 303 essais, le repli n'est donc jamais emprunté |
| `plot_pareto_front`, `plot_score_decomposition`, `plot_per_field_sensitivity`, `plot_field_correlation_heatmap` | **même cécité de schéma que D-60**, non corrigée : elles lisent `phys_score` / `error_<champ>`, que seul l'objectif mono-scénario de `pipeline.py` écrit. Elles restent derrière `has_decomposed_data`, qui **imprime** une ligne `[INFO]` quand elle est fausse — l'absence est donc dite, contrairement à la courbe de seuil. Les rendre composites demanderait de choisir une agrégation par champ qu'aucun artefact ne réclame : mesurer et demander plutôt qu'inventer |
| `plot_optuna_builtins`, `plot_convergence`, `plot_2d_landscapes`, `plot_scenario_*` restants | **lus, aucun défaut de valeur** — interpolations et tracés ; `griddata(method="cubic")` peut dépasser les données aux bords, ce qui est une propriété de l'interpolant, pas un défaut du module |

**Axes empruntés.** Aucun de ceux de la fiche : ce module ne construit ni
hamiltonien ni patch, il relit des bases. Les axes qui le concernent sont
ceux de ses **entrées**, et les deux côtés sont traversés par les tests de
D-60 : schéma **mono-scénario** (`phys_score` / `patch_ratio`, écrit par
`pipeline.py`) **et** schéma **composite** (`phys_<scenario>`, écrit par
`train_hyperparams`) ; bras **quantique** (`q_has_v2_phase1`, 5 paramètres)
**et** bras **classique** (`classical_v2_phase1`, `threshold_amr` seul) ;
étude **avec** seuil et étude **sans**.

**Ce que la lecture a montré sans que ce soit un défaut** : sur les dix bases
de `results/hyperparams/optuna_studies/`, **deux seulement** portent une
étude — `q_has_v2_phase1` et `classical_v2_phase1`. Les huit autres, phases
1b, 2 et 3 comprises, sont vides. C'est cohérent avec D-22 et n'ajoute rien
à ce qu'il dit ; c'est noté ici pour que personne ne les relise en pensant y
trouver la campagne.

---

## 1a bis. `src/recompute_lambda_scores.py` — lu en entier

717 lignes. Il rejoue le score d'un essai sous un autre `lambda_cost` :
c'est le seul chemin par lequel un arbitrage précision/coût déjà payé peut
être relu sans relancer la campagne.

**Son contrat central tient**, et il a été mesuré **deux fois par deux
passes indépendantes, au même nombre** : à `LAMBDA_COST_SOFT = 0,4`,
`recompute_score` rend exactement ce que la campagne a mesuré — écart
maximal **5,551e−17** sur les 178 essais quantiques et **2,220e−16** sur les
125 classiques, **303/303**. La formule `(phys + λ·patch)/(1+λ)` et la
moyenne des sous-pertes coïncident des deux côtés (`pipeline.score` et
`_composite_loop` contre `recompute_score`). Verrouillé par
`tests/pipeline/test_recompute_lambda_scores.py` (D-49), qui sépare aussi à
λ = 0,41 : l'écart passe à 9,839e−03.

| fonction | verdict |
|---|---|
| `main`, rattrapage des erreurs | **D-49** — tout le corps dans un `except Exception` sans sortie non nulle. **Trouvé deux fois** : la seconde passe l'a re-signalé sous le numéro **D-63**, retiré depuis. La correction retenue est celle de D-49 |
| `plot_pareto_with_isocost`, fenêtre | **D-62** — `set_ylim(-0,05 ; 0,40)` en dur : 9 essais sur 125 et 3 des 46 points du front hors cadre sur l'étude classique |
| `_add_trend` | **D-61**, second site — copie mot pour mot de celle d'`analyze_hyperparams` |
| `recompute_score`, `_recompute_global`, `build_trial_table` | **sains** — voir le contrat ci-dessus. Le repli global→moyenne des scénarios est le même dans les trois (`_get_global_phys_patch`, `build_trial_table`, `recompute_score`), et rend la même valeur |
| `save_summary`, changements de rang | **sain** — rangs d'origine et rangs nouveaux calculés sur la même liste, `orig_rank − new_rank` positif = montée, cohérent avec l'étiquette `UP` |
| `plot_lambda_sweep` | **sain** — orientation de la carte `imshow` assortie à ses étiquettes et à `ax.text(j, i)` ; la ligne d'iso-score est coupée à `iso ≥ 0`, ce qui est voulu |
| `_pareto_front` | **sain** — copie identique à celle d'`analyze_hyperparams`, croisée contre une référence indépendante (300/300) |
| `save_summary`, `original_lambda` | **réserve, non corrigée** — le paramètre existe mais `run_single_lambda` ne le passe jamais : le résumé compare `new_score` à `original_score` sans jamais dire quel λ a produit le second. Rien de faux n'est imprimé ; il manque une ligne. Signalé plutôt que corrigé — écrire « 0,4 » ici serait le recopier depuis `train_hyperparams` sans que l'artefact le porte, exactement la valeur sans provenance que D-22 reproche au JSON déployé |
| `plot_scenario_reranked` | **réserve, non corrigée** — `user_attrs.get(f"phys_{key}", 0)` : un scénario sans mesure s'empilerait à zéro. Non emprunté sur les bases du dépôt (un seul jeu de clés, 303 essais) |

### Deux passes sur les mêmes fichiers, le même jour

`analyze_hyperparams.py` et `recompute_lambda_scores.py` ont été audités le
13 août par **deux passes qui ne se voyaient pas** : D-49/D-50 sur la
branche `claude/kind-babbage-927g10`, puis D-60 à D-63 sur la branche Vigil,
ouverte **avant** que la première ne soit poussée et jamais rebasée dessus.

Ce que ça a coûté : **un défaut sur quatre re-trouvé** (D-63 = D-49), et
deux jeux de tests qui mesurent la même chose — les doublons de la seconde
passe ont été supprimés, la version D-49 est celle qui reste.

Ce que ça a rapporté, involontairement : **le contrat central mesuré deux
fois, aux mêmes chiffres, par deux chemins écrits séparément** — 5,551e−17
et 2,220e−16. Une reproduction, pas une répétition.

**La règle qui manquait** : une branche Vigil se rebase (ou fusionne) sur sa
base **avant** de lire un fichier, pas seulement avant de pousser. Lire le
fil de la PR ne suffit pas : la première passe avait poussé son travail sans
encore le commenter.

**Axes empruntés** : schéma **composite** (`phys_<scenario>`) **et**
schéma **mono-scénario** (`phys_score`, par `_recompute_global`) ; bras
**quantique** et **classique** ; λ **de l'entraînement (0,4)**, λ **nul**
et λ **unitaire** ; sortie **saine** et sortie **en échec** (étude absente,
base absente, écriture impossible). Non traversé : le mode `--lambda-sweep`
à plus d'un λ n'est vérifié par aucun test — il écrit `lambda_sweep_results.json`
et deux figures de plus, tous produits par les mêmes fonctions déjà couvertes.

---

## 1a ter. `src/import_Neon_data_to_local.py` — lu en entier

76 lignes, et le seul code du dépôt qui **supprime** une étude Optuna.

| ligne | verdict |
|---|---|
| la boucle d'import | **D-64** — suppression de la destination avant lecture de la source, échec rattrapé en code 0 |
| l'URL par défaut | **D-65** — identifiant Neon complet publié dans un dépôt public (ouvert : rotation) |
| `--ResetNeon` contre `--ResetLocal` | **asymétrie non corrigée, mesurée** — `--ResetNeon` ne supprime que `q_has_v2_phase1` (garde codée en dur), `--ResetLocal` supprime les **dix**. Aucune des deux n'est documentée dans l'aide au-delà de « Reset … data ». Ce n'est pas une valeur fausse : c'est un drapeau destructeur dont la portée ne se lit que dans le code. Signalé, à trancher par USER |
| le message de fin | **réserve, non corrigée** — `Successfully uploaded {study} to Neon` est imprimé **dans les deux sens**, y compris quand la copie va de Neon vers le local, et le compte d'essais qu'il donne est celui de la **source**, pas de ce qui a été écrit |

**Axes empruntés** : les deux sens (`--LocalToNeon` et le défaut), source
**présente** et **absente**, destination **peuplée** et **absente**, échec
d'écriture réel. Tous traversés avec **deux SQLite** — `--in-url` accepte
n'importe quelle URL de stockage, donc le chemin « Neon » se teste hors
ligne. Non traversé : un vrai PostgreSQL.

## 1a quater. `src/visual.py` et `src/help_visual.py` — lus en entier

327 lignes, les deux derniers modules de `src/` que l'inventaire excluait,
avec pour raison « tracé matplotlib, aucune valeur numérique produite ».

**La raison d'exclusion était vraie et insuffisante.** Aucune de ces
fonctions ne rend de valeur — c'est exact, vérifié : toutes rendent `None`.
Mais une figure porte une **convention d'axes**, et celle de la seule
fonction qui s'exécute était fausse : **D-68**. « Ne produit aucun nombre »
n'implique pas « ne peut rien dire de faux ».

**Ce qui s'exécute, et ce qui ne s'exécute pas.** Mesuré, pas supposé — un
`grep` des appelants sur tout le dépôt, `docs/archive/` exclu :

| fonction | lignes | appelée par |
|---|---|---|
| `plot_amr_state` | 88 | `src/pipeline.py`, 4× par pas de verrouillage |
| `plot_recursive_state` | 48 | **personne** |
| `simple_hierarchical_plot` | 22 | **personne** |
| `plot_grid_topology` | 28 | **personne** |
| `plot_flux_on_edges` | 76 | **personne** |
| `visualize_vqa_step` | 51 | **personne** — mais *importée* par `refinement.py` |

**88 lignes sur 327 s'exécutent.** Les 239 autres sont mortes.

`visualize_vqa_step` mérite sa ligne : `src/Simulation/refinement.py:7` fait
`from help_visual import visualize_vqa_step` et ne l'appelle jamais. L'import
seul suffit à rendre `matplotlib` obligatoire pour importer le chemin AMR.
Ce n'est pas un défaut de calcul — rien n'en dépend numériquement — mais
c'est une dépendance que rien ne justifie, et elle est notée ici plutôt que
retirée : `src/` est l'objet d'étude, et retirer un import est un
changement de comportement à l'import qu'aucune mesure ne réclame
aujourd'hui.

**Vérifié et trouvé sain**, mesuré et non supposé :

- `_, _, _, _, Jz = sim.get_fluxes().values()` — le dépaquetage **positionnel**
  d'un dictionnaire. Fragile par construction, mais **correct** :
  `MHDSolver.get_fluxes` rend `{'vx', 'vy', 'Bx', 'By', 'Jz'}` dans cet
  ordre, et `Jz` y est bien le cinquième. Un `dict` Python conserve l'ordre
  d'insertion depuis 3.7 ; le contrat tient tant que l'ordre des clés de
  `get_fluxes` ne change pas. Aucun test ne le garde — noté, non corrigé ;
- `Jz` lui-même : `get_fluxes` forme `dBy/dX − dBx/dY` avec `axis=0` pour X,
  conforme à `AXIS_X = 0` / `AXIS_Y = 1`. Ce n'est **pas** un rotationnel
  privé au sens de `test_no_private_curl_survives.py` — il est du bon côté
  de la convention ;
- **les cadres d'attention tombent sur les structures qu'ils désignent** :
  `bounds = (a0_s, a0_e, a1_s, a1_e)` indexe l'axe 0 puis l'axe 1 — l'ordre
  que `get_periodic_patch` consomme — et `Rectangle((xs, ys), …)` place bien
  le cadre sur l'image non transposée. Vérifié sur un champ **asymétrique
  sous transposition**, seul champ qui sépare les deux hypothèses. C'est la
  moitié saine de `plot_amr_state`, et un test l'épingle pour qu'une
  « correction » de D-68 ne la casse pas ;
- le garde `if not verbose and save_dir is None: return` : la fonction ne
  fait rien quand la boucle fermée tourne sans figure. Il est placé **avant**
  la docstring, qui n'est donc pas une docstring mais une expression morte —
  `plot_amr_state.__doc__` vaut `None`. Sans effet, noté.

**Axes empruntés** (fiche `VIGIL_BA_Proj.md`) : `depth = 0` **et** `depth > 0`
— le cadre à `depth > 0` porte une annotation de zoom que `depth = 0` n'a
pas, les deux branches sont traversées ; patch **périodique** et **borné**
sans distinction, `plot_amr_state` ne lit pas la topologie ; bras
**quantique** et **classique** — `pipeline.py` appelle la même fonction pour
les quatre suffixes (`dns`, `quantum_amr`, `quantum_amr_wo_vqa`,
`classic_amr`), aucun chemin ne diffère. Les axes backend, warm start,
Hamiltonien nul et optimiseur **ne traversent pas ce module** : il ne voit
que `sim.get_fluxes()` et une liste de `bounds`.

**Ce que cette lecture ne couvre pas.** Les 239 lignes mortes ont été lues,
pas mesurées : `plot_recursive_state` contient une boucle dont le corps est
un `pass` et un commentaire annonçant une correction jamais faite, et
`simple_hierarchical_plot` construit un `depth_counts` qu'il n'utilise pas.
Rien n'en dépend. Elles ne sont pas auditées au sens de la fiche, et ne le
seront que si un appelant apparaît.

---

## `src/compare_rotor_budget.py` — lu en entier, tourne depuis D-10, un défaut ouvert (D-91)

Note obsolète : le `TypeError` décrit ci-dessous a été corrigé le
13 août 2026 (`403240b`, D-10/D-66/D-67) — **avant** que cette note ne soit
écrite, mais sans qu'elle soit mise à jour. Le script tourne, produit son
`.npz`, et est désormais lu en entier, fonction par fonction (question 2 de
`VIGIL.md` posée sur `compute_block_errors` : que promet-elle ?).

**D-91, ouvert** (`DEFAUTS.md`) : `compute_block_errors` divise par
`ref = sqrt(mean(dns_block**2)) + 1e-10`, plancher côté dénominateur
seul — deux blocs au même écart absolu reçoivent un score qui dépend de
l'amplitude du signal, pas de l'écart. Sur le rotor MHD réel, la sélection
« ground truth » qui en sort exclut le bloc central (celui qui porte la
vraie structure), au profit de coins de fond quasi vide. C'est la cause de
l'anomalie déjà notée dans `RESULTS.md` (D-10 : sélection ground truth
0,3079, à peine mieux que l'absence d'AMR à 0,3074, contre 0,0208 pour
classique/Q-HAS). Non corrigé : changer la métrique changerait ces deux
nombres déjà publiés.

Sain par ailleurs : `select_top_k` (testé, `argsort` décroissant correct),
`build_patches_from_selection` (même convention `bi,bj` que les trois
fonctions de score), `classical_block_scores`/`qhas_block_scores` (le
mapper reçoit désormais les hyperparamètres réellement déployés, D-10),
`compute_solution_error` (nulle sur un champ identique, croissante avec
l'écart, testé). Historique désormais dans le fichier lui-même : même
réparé, il comparerait un bras Q-HAS aux hyperparamètres écrits en dur
avant D-10 — ce n'est plus le cas, `qhas_block_scores` appelle
`load_hyperparams()`.

**Axes empruntés** : bras qhas/classique/ground-truth tous exercés, bord
périodique uniquement, `state_vector` uniquement (le seul backend rejoué
ici), warm start absent (`Phi_prev` fourni mais pas de cache warm start
inter-appel comme `refinement.py`), Hamiltonien non nul.

---

## 1b. `study/common/` — vérifié cette passe

Trois modules relus en entier et **croisés contre une référence
indépendante**, pas seulement lus :

| lu | verdict |
|---|---|
| `stats_confirmatory.holm_correction` | **sain** — multiplicateur `(m − j + 1)`, cumul monotone puis plafond à 1, dans cet ordre. Croisé sur **2 000** jeux de p-valeurs arrondies à 1e−3 (donc pleins d'ex aequo, le cas où un `argsort` non stable pourrait mordre) : **2 000/2 000** identiques à une implémentation de référence écrite séparément |
| `stats_confirmatory.tost_equivalence` | **sain** — les deux tests unilatéraux et le `max` des deux p. Apparié : **500/500** identiques à `scipy.stats.ttest_1samp(d, ∓margin, alternative=…)`. Non apparié : **500/500** identiques au `df` de Welch de `scipy.stats.ttest_ind(equal_var=False)` |
| `stats.paired_delta_bootstrap`, `stats.bootstrap_by_trajectory` | **sains** — `np.unique` trie les identifiants de la même façon des deux côtés, donc `groups_a[i]` et `groups_b[i]` décrivent bien la même trajectoire : l'appariement tient |
| `metrics.degeneracy_flag` et ses 2 appelants | **sain** — le piège attendu était une prévalence calculée sur un autre ensemble que le `gt` qui sert au F1 ; vérifié, les deux appelants passent `float(Yva.mean())` avec le même `Yva`. C'est la même famille que D-45 : le protocole v3 sait déjà nommer un plancher de dégénérescence |
| `provenance.py` | **sain** — `git_hash` reste le hash de **départ**, `head_moved_during_run` et `dirty_at_start` disent quand aucun hash ne décrit l'exécution |
| `ising_terms_and_annealing.build_ising_terms` contre `create_period_hamiltonian` | **sain** — la docstring promet « the EXACT SAME Hamiltonian that QAOA minimises », et les deux n'ont pourtant pas le même seuil d'encodage : **1e−12** côté SA, `COEFF_MIN = 1e−6` côté QAOA. Mesuré sur les 800 coefficients des 40 snapshots (dim=2, Re=400, N=256) : **0** tombe dans la bande `[1e−12, 1e−6)`, donc les deux encodent bien le même opérateur. Écart latent, pas un défaut mesuré — le rejouer si `sigma` ou `w_z_frac` bougent |
| `ising_terms_and_annealing`, convention de spin | **saine** — `Z = −1 → raffiner` côté SA et `P(q=1) > 0,5 → raffiner` côté diagonalisation exacte décrivent le même état ; topologies `_idx_H`/`_idx_V` identiques à celles de `create_period_hamiltonian` |

**Une piste écartée par la mesure, à ne pas re-suivre.** La phase 7
(`analyze_snapshot_sa`) compare SA à `score_vqa > thr_amr`, exactement la
forme de comparaison non gardée que D-45 a trouvée dégénérée en phase 4.
Vérifié avant d'accuser, à l'usage documenté (`--dim 4 --v2`, Re=400,
N=256, 3 snapshots × 4 scénarios, sweeps=2000, restarts=10) : décision SA
constante **9/12**, décision classique constante **10/12**, SA identique au
classique **7/12**, F1 à égalité **8/12**. Ce n'est **pas** la dégénérescence
totale de la phase 4 (40/40) : la phase 7 porte du signal, et il n'y a pas
de défaut ici. Elle gagnerait le même drapeau que D-45, mais c'est une
amélioration de rapport, pas une correction — non faite, pas mesurée comme
nécessaire.

`delta_energy` double la contribution d'une plaquette dont deux sommets
coïncident (`_build_incidence` ajoute l'index une fois par sommet) : cela
n'arrive qu'à **dim = 1**, absent de `VQA_DIMS = [2, 4, 8]`. Noté, non
corrigé — corriger un chemin que rien n'emprunte, c'est du risque sans
mesure.

Deux autres remarques du même module, ni l'une ni l'autre une valeur fausse :
`main()` prend `--dim [2, 3, 4]` par défaut alors qu'aucun
`patches_*_dim3.npz` n'existe — le cas part dans la branche `SKIP ... missing
input`, qui **crie**, donc pas de balayage muet ; et le verdict final
(« beats / worse / ties ») tranche sur une bande de **±0,02** codée en dur,
sans provenance écrite. À remesurer le jour où ce verdict devient un nombre
publié.


**Confirmation d'une observation déjà publiée, pas une trouvaille.**
`RESULTS.md` rapporte, via `diag_hamiltonian_balance.py`, `max|K| = 0
exactement` — « aucun ZZZZ ne survit au sous-échantillonnage ». Mesuré ici à
la résolution de **production** (`compute_patch_coefficients`, Re=400,
N=256, dim=4, 3 snapshots × 4 scénarios) : `max|K_cell| = 0` sur **12/12**,
donc le terme ZZZZ ne franchit son seuil ni après réduction ni à pleine
grille. Le ZZ, lui, vit : `max|C_cell|` vaut 26–73 sur `orszag_tang` et
`mhd_rotor`, 0 sur `harris_tearing`/`kelvin_helmholtz` (c'est D-41).
Vérifié contre le registre **avant** de l'écrire : rien de neuf à signaler,
la portée de l'énoncé publié s'élargit seulement.

Symétriquement, à la résolution VQA de la phase 4 (dim=2) c'est l'inverse :
`|K|` vaut jusqu'à 65,9 et `|C|` s'effondre sous 2,4e−120 — les deux termes
multi-corps ne sont jamais vivants **en même temps** aux deux résolutions
que le dépôt emprunte. Non mesuré : s'il existe une résolution où les deux
franchissent leur seuil ensemble.

`aggregate_master_table.py` rejoué sur cette branche :
**180 lignes, OK = 164, DIFF = 16, MISSING = 0** — exactement l'état
documenté, donc D-45 / D-46 / D-47 n'ont déplacé **aucun** nombre publié.

`ising_terms_and_annealing.py` était marqué « non lu » plus haut alors que
la section 1b le couvre déjà en entier — mention corrigée, pas une nouvelle
lecture.

### `qaoa_inputs.py` — lu en entier

Le module qui fabrique les entrées QAOA de la phase 5 et des études h0 / h3.
Deux passes concurrentes l'ont lu cette nuit ; verdicts fusionnés ici.

| lu | verdict |
|---|---|
| `classical_warm_start_params` | **D-48** — schedule **constant** : ni `score_vqa` ni `threshold_amr` n'entrent dans le résultat. Sortie identique bit-à-bit sur 6 entrées couvrant tout l'intervalle, écart **0,0e+00**. Le nom, la docstring et l'aide CLI annonçaient un warm start dérivé de la décision classique |
| `prepare_qaoa_inputs`, réduction `block_avg` des champs / `block_max` du score | **déjà mesuré par D-47, ne pas y revenir** — l'écart d'opérateur est réel et n'est *pas* la cause de la dégénérescence (39/40 contre 40/40 avec le score assorti) ; c'est le même calcul que `exact_diagonalisation.build_patch_hamiltonian` |
| `_psi_from_pipeline` | **sain** — délègue à `refinement._prepare_vqa_input`, l'encodeur réellement déployé, au lieu de le réimplémenter ; lève si l'encodeur refuse le patch plutôt que de fabriquer un psi. `angles` reçu est déjà le 4-tuple `(theta_h, theta_v, psi_h, psi_v)` (`PhysToAngle.map_to_angles`), donc son dépaquetage par l'appelant est correct malgré l'apparence d'un retour à une seule valeur |
| `prepare_qaoa_inputs`, garde `with_psi` sans `prev_fields` | **saine** — lève explicitement : psi est une dérivée temporelle, il ne peut pas naître d'un instantané isolé |
| `full_comparison.metrics` | **sain** — dénominateurs gardés par `max(·, 1)` ; sur un prédicteur vide `tp = 0` rend F1 = 0, pas une division par zéro |
| `run_phase5`, chaînage du warm start | **sain et conforme au déployé** — enchaîne `optimal_params` d'un instantané au suivant, comme `refinement.py` le fait via `warm_start_cache`. Le schedule constant de D-48 n'y sert **que** pour le premier instantané, et seulement sous `--warm-start`. C'est h0 / h3 qui l'appliquent à **chaque** appel |
| `run_phase5`, absence de patch prometteur | **crie** — `No promising patches -- skipping QAOA`, pas de balayage muet |
| `run_phase5`, `is_hard_all` chargé puis jamais lu | **gaspillage, pas une valeur fausse** — `gt_refine` recalcule `l2_all >= l2_threshold`, exactement la formule qui a produit `is_hard` dans `hard_patch_labels.py:209` |
| `prune_hamilt_params` | **incohérence docstring / code, sans conséquence mesurable** — la docstring annonce un élagage « par bloc » sur `H_edges`, `C_edges`, `K_plaquettes` (3 blocs) ; le code prend un maximum séparé pour `H0` et `H1`, `C0` et `C1` (5 groupes). Non corrigé : **aucun artefact `*depth*` n'existe dans `results/`** et aucune ligne de `RESULTS.md` ne cite l'élagage, donc aucun nombre publié n'en dépend — corriger sans mesure serait du risque sans gain |

### `study/h0_selection/h0_qaoa_displacement.py` — lu en entier

Rouvert par D-48, qui vient de son initialisation.

| lu | verdict |
|---|---|
| ligne `READING` de `main()` | **D-50** — tranche sur `\|progress\| < 0,1`, seuil sans provenance, sur une grandeur dont la dispersion mesurée (0,018) dépasse la marge au seuil (0,0146) : **1 exécution sur 3** imprime la conclusion inverse. Extrait en `reading_message()` pour être testable, texte et seuil **inchangés** |
| `variational_progress` | **sain** — projection scalaire correcte, `den > eps` sinon `NaN` : une progression indéfinie ne devient pas 0 |
| moyenne et pente, traitement des indéfinis | **sains, et c'est un point fin déjà réglé** — la pente est **appariée par instantané** entre `p_min` et `p_max`, avec un commentaire disant pourquoi deux `nanmean` indépendants compareraient deux populations différentes ; `n_undefined_progress` est écrit dans l'artefact au lieu d'être absorbé |
| `check_expected_behaviour` | **garde réelle mais incomplète** — `MAX_FRAC_UNDEFINED` et `MIN_PAIRED` mordent ; rien ne garde la distance au seuil du verdict (D-50) |
| `theta_marginals`, `ground_state_marginals`, `mask_uniformity` | **sains** — `P(\|1⟩) = sin²(θ/2) = score`, convention `Z = −1 → raffiner` identique à celle de `ising_terms_and_annealing` |
| balayage vide | **crie** — `raise SystemExit` explicite si aucun instantané n'est traité |
| provenance | **saine** — hash git, `--seed`, CLI complet écrits dans le `.npz` |

**Axes empruntés** : bras quantique, backend `state_vector`, hamiltonien non
nul, bord périodique, warm start **présent et absent** (mesure D-48),
optimiseur COBYLA, AMR `depth = 0`, `dim = 2`. Non traversés : `classical_only`,
backend échantillonné, bord borné, hamiltonien nul, autres optimiseurs,
`depth > 0`, `dim = 4 / 8`.

### `study/h0_selection/h0_optimiser_equivalence.py` — lu en entier

Le second et dernier fichier de `h0_selection` : **le module est clos**.
C'est le script dont le verdict porte le `RÉFUTÉ` de `h0_selection` dans
`CLAUDE.md`.

| lu | verdict |
|---|---|
| `check_expected_behaviour`, sans optimum certifié | **D-52** — `hit_optimum` et `exact_match` valent `NaN`, comparés par `<` à `MIN_HIT`/`MIN_MASK_MATCH` : `nan < 1.0` est **False**, donc `missed` et `diverging` restaient vides quoi qu'il arrive. Le critère qui existe pour que le script *puisse* échouer ne pouvait ni échouer ni réussir. Run réel `--N 64 --dim 2 --n-snaps 1 --no-exact` : code de sortie **0**, `[ACCEPTANCE] … H0 refutee` imprimé trois lignes sous une `DECISION RULE` disant *« QAOA deviates from the certified optimum »*. Corrigé : `[INDECIDABLE]`, chemin certifié bit-à-bit identique |
| `MIN_HIT` / `MIN_MASK_MATCH`, la référence de leur commentaire | **D-53** — *« les huit solveurs … atteignent l'optimum certifié sur 100 % des instantanés »* est vrai à `dim = 2` et **faux à `dim = 3`** : sur les 32 instantanés de `..._N96_dim3.npz` (18 qubits, donc certifié) le QAOA atteint l'optimum sur **0,062 à 0,156**, sous la règle classique elle-même (0,500). Le critère, rejoué dessus, lève : *« H0 redevient plausible »*. Rapport seul — la décision touche une lecture publiée |
| `exhaustive_ground_state`, comptage de `n_optima` en deux passes | **sain, et la correction qu'annonce son commentaire tient** — croisé contre une énumération de référence écrite séparément, **60** hamiltoniens à coefficients **entiers** (donc pleins d'ex aequo, le cas où le comptage en une passe mordait) × **4** tailles de bloc (`1<<16`, 64, 7, **1**, la dernière forçant un bloc par configuration) : **0 désaccord** sur `(E_min, n_optima)` |
| énergie de l'énumération contre `total_energy` | **saines et assorties** — c'est la question 4 sur ce module : l'optimum certifié et les énergies enregistrées par `_record` doivent sortir du même opérateur, sinon `hit_optimum` compare deux échelles. 400 configurations aléatoires, écart max **1,78e−15** ; et `E_exact` est **exactement** `min(total_energy)` sur les 2⁸ configurations (écart 0,0e+00, `n_optima` confirmé par un comptage indépendant) |
| `classical_init_spins` contre le bloc en ligne d'`analyze_snapshot_sa` | **sain** — la docstring annonce une convention « identique » ; vérifié sur données réelles, tableaux **égaux**. `.ravel()` ici, `.flatten()` là : même ordre C, même résultat |
| aller-retour spins → `(dh, dv)` → spins du bras QAOA | **sain** — `_record` reconstruit les spins depuis les décisions QAOA ; 200 tirages, **identique** à chaque fois : pas d'inversion d'ordre entre les deux blocs de qubits |
| `greedy_local_search`, énergie accumulée | **saine** — `E += dEs[q]` pourrait dériver de l'énergie vraie (forme « variable locale non réécrite ») ; mesuré, accumulée et recalculée coïncident à la 9ᵉ décimale. Sans conséquence de toute façon : `_record` recalcule par `total_energy` et ignore la valeur rendue |
| `_output_path` / `_checkpoint_path`, signature de reprise | **sains** — le point de reprise dérive du nom de l'artefact, donc les deux ne peuvent pas diverger ; `_run_signature` couvre tous les arguments sauf `scenario` (déjà dans le nom du fichier) et refuse explicitement une reprise issue d'autres réglages plutôt que de panacher deux campagnes |
| balayage vide | **crie** — `RuntimeError` explicite listant les artefacts attendus |
| `f1_from_masks`, `decision_agreement` | **sains** — F1 = `2tp/(2tp+fp+fn)`, dénominateur nul rendu 0 ; convention de décision `dec_h | dec_v` identique à la phase 7 |

**Trois remarques, aucune n'est une valeur fausse** — non corrigées faute de
conséquence mesurée :

- le critère range le **QAOA parmi les solveurs déterministes** et exige de
  lui `hit = 1,000`, alors que la fiche établit que le bras QAOA n'est pas
  déterministe. À `dim = 2` l'assertion ne tire jamais (le fondamental est
  constant, D-47) ; à `dim = 3` elle tire, et c'est D-53. Le classement
  lui-même n'a donc jamais été exercé là où il pourrait être faux ;
- la sélection d'instantanés `set(int(round(i)) for i in linspace(…))`
  **collapse silencieusement** quand `n_dns` est petit devant `--n-snaps` :
  on obtient moins d'instantanés que demandé, sans un mot. Aux tailles
  réelles (`n_dns` ≈ 20, `n_snaps` ≤ 3) le cas ne se produit pas, et le
  commentaire voisin — « la sélection commence à 1 » — n'est faux que dans
  ce même régime ;
- l'artefact **n'écrit pas** `certified` : la seule trace qu'une campagne
  n'a pas certifié son optimum est que `hit`/`match` y sont `NaN`. Dérivable,
  donc pas une valeur sans provenance — mais une colonne explicite coûterait
  une ligne. Non ajoutée : changer le schéma d'un artefact n'est pas une
  correction minimale.

**Axes empruntés.** Bras **quantique** ; backend **`state_vector` *et*
échantillonné** (`aer`, 4096 tirs — les deux côtés de cet axe, ce qu'aucun
module lu jusqu'ici ne faisait) ; warm start **présent *et* absent** (`sa`
froid contre `sa_warm`/`greedy` amorcés — mais **pas** côté QAOA, qui reçoit
le schedule constant de D-48) ; hamiltonien **non nul** ; bord
**périodique** ; optimiseur **COBYLA** ; AMR **`depth = 0`** ; `dim` **2 et
3** par les artefacts, **4** par la mesure de D-52. Non traversés :
`classical_only`, bord **borné**, hamiltonien **nul**, autres optimiseurs,
`depth > 0`, et l'axe des **anomalies avancées** (`False` codé en dur, D-51).

### `study/h1_solver/h1_curl_convention_gap.py` (T31) — lu en entier

Le script qui chiffre ce que la correction de convention d'axes des
mappeurs (D-1 à D-17) changerait pour la tâche de sélection de patches, et
sur lequel la conclusion « corriger sans réoptimiser dégrade à dim=16 »
s'appuie.

| lu | verdict |
|---|---|
| `_top_k`, budget apparié | **sain** — sélectionne exactement k patches, prend les k scores les plus hauts, `kind="stable"` (déterministe sur des ex æquo) |
| `_spearman` | **sain** — refuse un vecteur constant (`NaN` explicite plutôt qu'une corrélation indéfinie silencieuse) |
| `bootstrap_delta_ci` | **sain** — le bloc est le **scénario**, pas l'instantané (les instantanés d'une même trajectoire ne sont pas indépendants) ; refuse un seul scénario (`NaN`) plutôt qu'un intervalle vide de sens |
| `verdict` | **sain** — n'affirme un sens que si l'IC95 exclut zéro des deux côtés ; `indécidable` sinon |
| `_hard_patches` | **D-70** — docstring « même définition que `hard_patch_labels.py` », corps : une autre formule (écart-type intra-patch de la norme, pas l'erreur L2 de reconstruction par grossissement en bloc). Corrigé, `RESULTS.md` |
| **la table publiée elle-même**, rejouée par sa propre commande | **D-69** — ne se reproduit plus à HEAD ; le verdict « dégrade » à dim=16 devient « indécidable ». Cause identifiée : le solveur a changé sous les 4 scénarios canoniques après l'écriture de T31 (D-25, D-26/D-27). Rapport seul, `DEFAUTS.md` |
| balayage vide | **crie** — `RuntimeError` explicite |

**Axes empruntés** : bord **périodique** uniquement (le script n'instancie
que `PeriodicGrid`) ; hamiltonien **non nul** sur les 4 scénarios ;
mappeur **v1** (`use_v2=False` par défaut, `--mapper` accepte v2 mais aucun
artefact publié ne l'utilise) ; `dim` **8 et 16** ; drapeau `fixed_curl`
**les deux côtés**, dans le même appel. Non traversés : bord **borné**,
mappeur **v2** publié, bras **classical_only** (le script ne construit que
des scores, pas de décision QAOA), backend, warm start, optimiseur — aucun
de ces axes n'entre dans ce que ce script mesure.

### `study/h1_solver/h1_solver_convergence.py` (T14) — lu en entier

`study/h1_solver/` **est maintenant lu en entier**, les deux fichiers du
module couverts. T14 valide numériquement le solveur V1 : auto-convergence
en maillage, conservation/contrainte solénoïdale, comportement hors grille
d'entraînement, et localisation temporelle de l'ordre 1 (D-25).

| lu | verdict |
|---|---|
| `evolve_to` — `sim.dt = min(sim.adapt_dt(cfl_target=cfl), t_end - t)` puis `sim.step_full()` | **sain** — question 4, forme « variable locale non réécrite » vérifiée explicitement : `adapt_dt` écrit `self.dt` ET le retourne, la ré-affectation qui suit (clip sur `t_end - t`) est bien relue par `step_full`, qui lit `self.dt` (pas un argument séparé). Pas le défaut-modèle de `VIGIL.md` |
| `to_common_grid`, `relative_l2`, `observed_order` | **sains** — coarsening par bloc réutilisé de la phase 2 (pas réimplémenté), norme L2 relative symétrique, `log2` protégé contre une erreur nulle ou négative (`NaN` explicite) |
| `splitting_order_diagnostic` | **sain** — rejoue `_rk4_step` et `enforce_incompressibility` de V1 sans réimplémentation ; le diagnostic (ordre ≈4 sans projection, ≈1 avec) confirme mécaniquement D-25, déjà mesuré et corrigé ailleurs |
| **la commande de reproduction elle-même**, ainsi que celle de 15 autres scripts `study/` et 2 lanceurs `scripts/` | **D-71** — `study/v4/t14_numerical_validation.py` (et 15 chemins frères) n'existent plus depuis la réorganisation `17d983d` ; corrigé, `RESULTS.md` |
| l'**opérateur** qui mesure `max\|div B\|/rms\|B\|` dans `evolve_to` | **D-72** — `dns_validation.div_B` est spectrale ; depuis D-25 le solveur ne projette plus B et le garantit **en FD4**. La ligne avait été lue, l'opérateur non : mesuré 3,9029e−02 contre 2,0266e−14 sur la configuration publiée, `all_checks_pass` True → False. Corrigé, `RESULTS.md` |

**Vérifié et trouvé sain : le reste de T14 se reproduit à HEAD.** L'artefact
`t14_numerical_validation.npz` date de `1f03713` (2026-08-11), donc **d'avant
D-25** — même situation que la table T31 que D-69 a trouvée non
reproductible. La question a donc été posée à T14 aussi, et la réponse est
l'inverse : seule la divergence avait bougé. Rejoué à HEAD sur la
configuration publiée, sans réécrire l'artefact :

| section | rejoué à HEAD | publié | tol |
|---|---|---|---|
| (A) `‖u_N − u_2N‖_rel` | 7,4132e−02 · 3,7087e−02 | 7,41e−02 · 3,71e−02 | — |
| (A) ordre d'auto-convergence | **0,9992** | 1,00 | 0,05 |
| (D) ordre temporel **avec** projection | **1,1194** | 1,12 | 0,05 |
| (D) ordre temporel **sans** projection | **3,9978** | 4,00 | 0,05 |

Les trois ordres publiés tiennent. C'est ce qui rend D-72 imputable : tout le
reste de T14 se reproduit, seul le diagnostic de divergence ne le faisait
plus. Et le master table, régénéré, reste à **180 / OK=164 / DIFF=16 /
MISSING=0**, ses 180 lignes de données identiques à l'octet.

**La même question posée à tous les consommateurs de l'opérateur — D-73.**
D-72 trouvé, la question qui l'a produit a été passée à chaque site qui
mesure une divergence : *depuis D-25, qui mesure B avec un opérateur que le
solveur ne garantit plus ?* Six sites importent de `dns_validation` ; un
seul autre pesait sur une décision, et c'était le plus coûteux —
`validate_one`, la porte de **chaque DNS nouvellement générée**. Mesuré de
bout en bout : elle rejetait une trajectoire saine (`divB 1,6e-02` contre un
seuil de 1e−3 ; assorti **5,0573e−06**). Corrigé dans `dns_extension`, le
fichier gelé intact, `RESULTS.md`. Les deux autres sites sont sains :
`tests/study/test_t8_dns_extension.py` mesure la **vitesse**, qui est bien
projetée spectralement (opérateur assorti), et `dns_validation.main` est le
chemin gelé lui-même.

**Ce que la passe précédente avait manqué, et pourquoi.** La ligne du haut
déclarait `evolve_to` **sain** après avoir vérifié la forme « variable locale
non réécrite » — vérification juste, et toujours valable. Mais elle portait
sur le `dt`, pas sur l'**opérateur** de l'autre moitié de la même ligne. Un
module se relit aussi par les grandeurs qu'il mesure et par l'opérateur qui
les produit, pas seulement par ses fonctions : c'est exactement ce que la
fiche appelle regarder « par les configurations ». D-72 est le second défaut
de ce dépôt trouvé dans du code déclaré audité.

**Axes empruntés** : aucun de la fiche — ce module ne construit ni circuit
ni décision, il valide le solveur MHD lui-même sur `orszag_tang`, grilles
32/64/128 (et 64/128/256 à N=256), Re dans et hors grille d'entraînement
({200, 3200} contre {400, 800, 1200, 1600}). **Ajoutés par D-72** : l'axe
**projection de B** (`PROJECT_B` False — le défaut — et True, qui referme
l'écart entre les deux opérateurs et que le test d'épinglage surveille), et
le scénario `mhd_rotor`, qui sépare onze ordres de grandeur là où
`orszag_tang` n'en sépare que quatre à N=128.

### Un axe qui manquait à la fiche : les anomalies avancées — D-51

`VIGIL_BA_Proj.md` liste sept axes ; il en manque un huitième, et `study/`
n'en emprunte qu'un côté :

| axe | les deux valeurs | ce que `study/` emprunte |
|---|---|---|
| anomalies avancées | `False` **et** `True` — le second ajoute le ZZZZ de point X | **`False` uniquement**, aux deux seuls sites qui mentionnent le drapeau (`qaoa_inputs.py:191`, `:350`) |

La campagne d'entraînement, elle, l'active sur **6/6** scénarios (D-33,
`RESULTS.md`). Mesuré, mappeur v1 entraîné (`beta_xpoint = 2,39`), N=256,
Re=400, 4 scénarios, drapeau OFF contre ON — les trois blocs communs sont
identiques à l'octet dans les huit cas :

| | `dim = 2` (tous les nombres publiés) | `dim = 4` (déclarée dans `VQA_DIMS`, jamais exécutée) |
|---|---|---|
| `max\|K_xpoint\|` | **0,0000e+00 sur 4/4** | 1,24e+01 à 4,35e+01 |
| rapport à `max\|K_plaq\|` | — | **0,23 à 1,00** |
| termes de Pauli, OFF → ON | 12 → 12 sur 4/4 | 48 → 52…62 |
| fondamental exact | **0 spin changé** | non calculable (32 qubits > plafond 20) |

**Rien de publié n'en dépend, et rien ne pouvait en dépendre** : à `dim = 2`
le terme est exactement nul. Mais `build_ising_terms` ne lit pas `K_xpoint`
du tout, donc recuit, diagonalisation exacte et ablations y sont aveugles
même drapeau levé — et l'ablation `no_ZZZZ` de T13 annule une clé que son
propre `ground_state_mask` ne lit jamais. Détail, options et commande :
`DEFAUTS.md` D-51 ; épinglé par
`tests/study/test_xpoint_term_absent_from_study.py`.

### `study/h3_representation/h3_term_ablation.py` — lu en entier

Le module qui produit T13, dont la lecture publiée est que les termes de
couplage sont **causalement inertes**. Ouvert parce que D-51 y avait déjà
montré une ablation vide (`no_ZZZZ` annule `K_xpoint`, clé que
`ground_state_mask` ne lit jamais).

| lu | verdict |
|---|---|
| le contrôle `full` | **D-54** — tautologie : `zero_hamiltonian_terms(hp, ())` rend une copie de `hp`, donc le masque comparé sort de la **même fonction sur la même entrée**. Mesuré en sabotant `TERM_KEYS` : contrôle **0,000000 des deux côtés**, et `no_ZZ`/`no_ZZZZ`/`Z_only` **0,0000 des deux côtés** — les trois lignes qui portent la conclusion. Corrigé : colonne `removed_max` (max\|Δ\| de ce que `build_ising_terms` produit réellement), ablation vide imprimée `EMPTY`, contrôle **asserté** |
| balayage vide | **D-55** — imprimait `no input.` et sortait à **0**, sans artefact, donc en laissant en place celui de la campagne précédente. Le module voisin sortait à **1** sur la même entrée. Corrigé : même `RuntimeError`, même formulation |
| gardes du module, mesurées par **AST** | **0 `assert`, 0 `raise`, 0 `SystemExit`** avant cette passe — contre 5 et 6 dans `h0_optimiser_equivalence.py`. `CLAUDE.md` exige pourtant que tout script de `study/` porte une assertion. 3 gardes après |
| `zero_hamiltonian_terms`, non-mutation de l'entrée | **saine** — copie superficielle qui **remplace** les clés au lieu d'écrire dedans ; évidencé par la mesure elle-même, le contrôle restant à 0 après six ablations successives sur le même `hp` |
| `coefficients_removed`, comparaison des opérateurs | **écrite avec l'opérateur assorti** — `build_ising_terms` n'émet un terme que si \|coefficient\| > 1e−12, donc une ablation **raccourcit** les listes d'index au lieu d'y écrire des zéros. Comparer position par position n'a pas de sens ; la comparaison se fait indexée par le tuple de qubits, sur l'union des deux. Une première version rendait `inf` sur ce cas — c'était la **mesure** qui était fausse, pas le code mesuré |
| `ground_state_mask`, contrat `dim <= 3` | **sain** — au-delà, `exhaustive_ground_state` lève (32 qubits > le plafond 22) : refus bruyant, pas de repli |
| nom de sortie et copie « legacy » | **sains** — le mappeur entre dans le nom depuis D9 ; la copie sous le nom historique n'est faite que pour `v1`, le mappeur déployé, et elle est annoncée |

**Deux remarques, aucune n'est une valeur fausse.** `couplings_only` est
déclaré « alias lisible de `no_Z` » et l'est bien — mesuré, les deux lignes
sont identiques (1,0000 / 8,3004e−02) : le tableau affiche donc **6** lignes
pour **5** ablations distinctes, et `aggregate_master_table.rows_t13` n'en
attend que 5. Et la sélection d'instantanés
`set(int(round(i)) for i in linspace(…))` collapse silencieusement quand
`n_dns` est petit devant `--n-snaps` — même forme que dans
`h0_optimiser_equivalence`, même absence de conséquence aux tailles réelles.

**Ce que la mesure de D-54 dit *en faveur* de T13**, et qu'il faut écrire
aussi : à la configuration mesurée (orszag_tang, Re=400, N=64, dim=2), les
couplages ont bien été retirés — `removed_max` vaut **2,6558e+00** pour
`no_ZZ` et **1,0000e+00** pour `no_ZZZZ`, contre **8,3004e−02** pour le
biais Z qui, lui, renverse toute la décision. La lecture « causalement
inertes » **tient** ; elle n'était simplement étayée par rien avant cette
colonne.

**Axes empruntés** : bras **quantique** (état fondamental exact seulement —
la décision QAOA que le docstring annonce comme optionnelle n'est **pas**
implémentée dans ce module) ; backend **aucun** (énumération) ; hamiltonien
**non nul** *et* **nul** (`no_Z` sur un hamiltonien sans biais laisse un
ferromagnétique 8 fois dégénéré — c'est le seul module lu à traverser ce
côté-là de l'axe) ; bord **périodique** ; warm start **absent** ; optimiseur
**aucun** ; AMR `depth = 0` ; `dim = 2`. Non traversés : `classical_only`,
bord borné, backend échantillonné, `depth > 0`, `dim = 3 / 4`, et l'axe des
anomalies avancées (D-51).

### `study/h3_representation/h3_size_scan.py` — lu en entier

T26, la tâche écrite pour répondre à l'objection que sa propre docstring
nomme comme « la faiblesse centrale de toute l'étude » : *à 8 qubits,
évidemment qu'il ne se passe rien*. **Un module d'une honnêteté inhabituelle,
et un défaut de reporting.**

| lu | verdict |
|---|---|
| `greedy_agrees_with_exhaustive` | **D-57** — calculé, rangé dans le JSON, **jamais imprimé ni contrôlé**, alors que l'en-tête annonce « validated at dim=2 » et que c'est le seul nombre qui autorise à lire `dim = 4` et `dim = 8`. Mesuré : **0,7500** au mappeur **v1**, le défaut de la tâche ; 1,0000 en v2. Le 0,75 est **déjà** dans `t26_size_scan_N256_v1.json`. Corrigé : colonne `proxy=exact`, dimensions non validées nommées, avertissement sur `< 1` |
| le contrôle `--force-greedy` | **conçu correctement, et il passe** — c'est le module lui-même qui écrit que si le glouton rend des changements non nuls là où l'exhaustif rend 0, « le scan en taille ne mesure rien ». Rejoué à dim=2 sur les **deux** mappeurs : `changed = 0,0000` sur les quatre ablations des deux côtés. **Le proxy ne fabrique pas les changements qu'il rapporte** — donc D-57 est un défaut de reporting, pas une conclusion fausse |
| nommage de l'artefact de contrôle (`_forcegreedy`) | **sain** — le mode de contrôle porte un nom distinct, avec en commentaire la raison : sans cela il écraserait le scan qu'il sert à valider (D9) |
| distinction « changer une décision » / « mieux détecter » | **saine et explicite** — le F1 contre la vérité terrain L2 est calculé à côté de `changed`, et le module imprime le rappel lui-même |
| balayage vide | **crie déjà** — `SystemExit("no input for any requested dim")` |
| contrôle `full` non nul | **imprimé en WARNING**, non asserté — mais contrairement à T13 (D-54) il est ici comparé entre ablations et non à lui-même, donc il n'est pas tautologique |

**Le point qui relie T26 à D-53.** Les deux tâches bornent leur validation à
`dim = 2`, et `dim = 3` — 18 qubits, **encore énumérable** — n'est dans les
`--dims` par défaut d'aucune des deux. C'est pourtant là que le fondamental
exact cesse d'être uniforme : mesuré, **6/12** instantanés uniformes à
`dim = 3` en mappeur v1, contre **12/12** à `dim = 2`. `dim = 3` est la
mesure la moins chère du dépôt qui ferait bouger l'une ou l'autre question.

**Axes empruntés** : `dim` **2, 4, 8** (c'est l'objet même de la tâche) ;
mappeurs **v1 *et* v2** ; hamiltonien **non nul** ; bord **périodique** ;
warm start **présent** (le glouton part de la décision classique, comme le
pipeline déployé) ; solveur **exhaustif *et* glouton**, avec le contrôle qui
les compare. Non traversés : bras `classical_only`, backend échantillonné,
bord borné, `depth > 0`, anomalies avancées (D-51).

### `study/h3_representation/h3_locality_proposition.py` — lu en entier

Le **livrable théorique** du protocole v3 (Proposition 2). Un seul défaut, et
il appartient à la famille D-56 (balayage vide). La proposition elle-même a
été **vérifiée par la mesure**, pas seulement relue.

| lu | verdict |
|---|---|
| **la Proposition 2 elle-même** — si `Σ 2\|C_ij\| + Σ 4\|K_p\| < \|h_i\|` sur **tous** les sites, alors le fondamental exact vaut `s_i* = −sign(h_i)` | **vérifiée, 300/300** — hamiltoniens tirés au sort à `dim = 2` (8 qubits) jusqu'à en obtenir 300 satisfaisant la condition **stricte** partout, puis fondamental **énuméré** (`exhaustive_ground_state`) comparé au champ moyen : **0 désaccord**. Marge `min(\|h\| − lhs)` médiane 0,0855, minimum **6,08e−04** — la condition est donc testée près de sa frontière, pas seulement loin d'elle |
| le **contrôle** de cette vérification | **il sépare** — sans lui la mesure ne dirait rien. Sur 300 hamiltoniens **violant** la condition, le fondamental exact diffère du champ moyen dans **95 %** des cas. L'accord n'est donc pas automatique : c'est bien la condition qui le produit |
| topologie, « miroir exact de `create_period_hamiltonian` » | **cohérente avec `build_ising_terms`** — c'est ce que la vérification ci-dessus emprunte de bout en bout : `per_site_condition` et `build_ising_terms` indexent les mêmes qubits, sinon les 300 accords seraient impossibles |
| `include_xpoint=True` par défaut | **divergence réelle avec le reste de `study/`, sans conséquence publiée** — ce module **lit** `K_xpoint` ; `build_ising_terms` ne le lit pas (D-51). Deux chemins qui décrivent le même hamiltonien ne comptent donc pas les mêmes termes. À `dim = 2`, `max\|K_xpoint\| = 0` (mesuré en D-51), donc `lhs` est identique des deux côtés et rien de publié n'en dépend. À `dim = 4` les deux divergeraient. **Noté, pas corrigé** : c'est le même arbitrage que D-51, et il est déjà ouvert |
| balayage vide | **D-56** — `if not rows: print("no input."); return`, code 0. Corrigé |

**Axes empruntés** : hamiltonien **non nul**, bord **périodique**, `dim` **2
et 4** (les deux valeurs par défaut du CLI), mappeurs **v1 *et* v2** — le
seul module lu à traverser cet axe-là aux deux valeurs. Non traversés : bras
`classical_only`, backend échantillonné, bord borné, hamiltonien nul, warm
start, optimiseurs, `depth > 0`.

### `study/h3_representation/h3_window_counterfactual.py` (T18) — lu en entier

Le contrefactuel de T13/T17 : la fenêtre gaussienne neutralisée
(`sigma → 1e9`), l'ablation ZZ change-t-elle encore une décision ?

| lu | verdict |
|---|---|
| `prepare_both_arms`, substitution de `TRAINED_SIGMA` | **saine** — la valeur est relue depuis le module (`qaoa_inputs.py:244`, `sigma=TRAINED_SIGMA` à l'intérieur du corps de fonction, pas un défaut par défaut) à chaque appel, donc la substitution du module-global prend effet ; restauration **assertée**, pas seulement faite dans un `finally` |
| l'assertion de neutralisation (`a_nw >= a_w·(1 − 1e−9)`) | **saine, et elle mord** — vérifie que la substitution a réellement affaibli la fenêtre plutôt que de le supposer |
| le contrôle `full` (`ctrl != 0.0` → WARNING) | **compare le Hamiltonien à lui-même, comme D-54 avant sa correction** — mais `tests/study/test_t18_window_counterfactual.py` le documente explicitement comme un contrôle de chaîne de mesure (« il doit rendre exactement 0 »), pas comme une validation du mécanisme d'ablation, et porte un vrai contrôle positif (`test_ablation_detects_a_real_change`) à côté. Moins sévère que la forme originale de D-54 : signalé, pas rouvert |
| lecture finale (« ZZ existe et la fenêtre le détruit » / « ZZ reste inerte ») | **conforme à la mesure** — `zz_nw` (ZZ ablation, fenêtre neutralisée) tranche entre les deux branches sans ambiguïté |
| l'effet propre de la fenêtre (`cross`), mesuré entre les deux bras sur le Hamiltonien complet | **sain et bien distingué** — le commentaire explique correctement que `|C|` entre dans `C_scale`, qui fixe l'échelle du biais Z : éteindre `C` n'agit donc pas seulement « comme un couplage » |

**Axes empruntés** : bras quantique (état fondamental exact), hamiltonien
**non nul**, bord périodique, mappeur **v1** (déployé), fenêtre **présente
*et* neutralisée** — c'est l'axe que le module existe pour traverser. Non
traversés : `classical_only`, backend échantillonné, bord borné, `dim > 2`,
warm start, optimiseurs, anomalies avancées (D-51).

### `study/h3_representation/h3_depth_report.py` (phase 8) — lu en entier

Dernier fichier non lu de `study/h3_representation/` — **le module est
maintenant lu en entier**. Script de reporting pur (profondeur de circuit,
nombre de portes) pour le manuscrit ; jamais exécuté jusqu'à publication —
`results/depth_report_N*.csv` n'existe dans aucun commit, et aucune ligne de
`RESULTS.md`/`EVALUATION.md` ne cite une profondeur de circuit ou un compte
de portes.

| lu | verdict |
|---|---|
| `_count_terms` (Z/ZZ/ZZZZ) | **sain** — classification par poids du label Pauli, cohérente avec `create_period_hamiltonian` |
| `report_row`, comptage avant/après élagage | **sain sur les cas rejoués** (orszag_tang, Re=400, N=256, dim 2 et 4, eps 0 à 5,0) — `prune_hamilt_params` retire bien des termes à `eps` suffisant, l'opérateur lève `NullHamiltonianError` si tout est élagué |
| liste blanche des portes à 2 qubits (`cx,cz,ecr,cp,rzz,rzx,swap`) | **vérifiée exhaustive sur les cas rejoués** — la cible réelle du backend (`AerSimulator(method='matrix_product_state')`, `opt_level=0`) ne produit que `cx`/`rzz` en pratique sur ces Hamiltoniens diagonaux ; le compte `two_q_gates` coïncide avec `gate_counts` filtré à la main |
| **découverte en le rejouant, pas en le lisant** : labels ZZ dupliqués dans `create_period_hamiltonian` à `dim = 2` | **D-59** (`DEFAUTS.md`) — sans conséquence mesurée sur les décisions publiées, mais une topologie qui n'avait jamais été testée avec des coefficients non uniformes (`tests/quantum/test_vqa_stack_analytic.py` n'utilise que `np.full`, donc ne peut pas la révéler) |
| `threshold_amr=0.15` codé en dur dans `report_row` | **valeur sans provenance, sans conséquence** — proche de `TRAINED_THRESHOLD` (0,1496…) mais pas la valeur exacte ; sans conséquence puisque rien de ce script n'est publié |

**Axes empruntés** : hamiltonien non nul, bord périodique, mappeur **v2**
(imposé par le docstring), `dim` **2 et 4**, élagage **présent et absent**
(`prune_eps` de 0 à 5,0 — le seul module à balayer cet axe). Non traversés :
bras `classical_only`, backend échantillonné, hamiltonien nul, mappeur v1,
warm start, optimiseurs.

`study/h3_representation/` **est maintenant lu en entier**, les sept
fichiers du dossier couverts module par module.

### `study/h3_representation/h3_uncertainty_window.py` — lu en entier

Lu tout de suite après T13, parce que c'est lui qui donne le **mécanisme**
de l'inertie causale que T13 constate. **Aucun défaut.**

| lu | verdict |
|---|---|
| `uncertainty_window`, réimplémentation de la fenêtre de `src` | **sain à l'usage, avec une divergence latente mesurée** — la docstring annonce « exactement comme `HamiltParams.compute_coefficients` la calcule ». Vérifié sur `init_orszag_tang` N=32 : les formes du score et des champs **coïncident** (32,32), donc la branche de redimensionnement de `src` n'est **pas** empruntée, et les deux fenêtres sont identiques à **0,000e+00**. Mais `src` redimensionne le score (`scipy.ndimage.zoom`, ordre 1) quand les formes diffèrent, et la copie de `study/` **ne le fait pas** : sur un score à halo (34,34) elle rendrait une fenêtre (34,34) là où `src` en rendrait une (32,32). C'est la forme exacte de D-37. Aucune conséquence ici — aucun appelant ne lui passe un score à halo — donc **noté, pas corrigé** |
| appariement de chaque famille d'arêtes à **sa propre** fenêtre | **sain, et c'est un point fin déjà réglé** — `w_h` sur les arêtes horizontales, `w_v` sur les verticales, avec un commentaire disant pourquoi les apparier entre elles fausserait `zz_mass_kept` |
| neutralisation de la fenêtre pour isoler les portes amont | **saine et assertée** — `sigma → 1e9` donne `w ≡ 1`, vérifié par un `assert` qui mord (`window not neutralised`), et la comparaison sépare bien les deux causes possibles d'un ZZ nul |
| les deux `sigma` « entraînés » | **saine** — le module refuse de les confondre (0,023 du pipeline ouvert lu **depuis le module** pour qu'il ne puisse pas diverger, contre 0,1888 du fold Level-3) et rapporte les jeux de paramètres séparément, sans conclure d'un seul |
| balayage vide | **crie** — `SystemExit("no scenario produced a measurement")` |
| statistiques rapportées sur `w_h` seul | **choix documenté, sans conséquence** — `zz_mass_kept`, la grandeur qui porte la lecture, concatène bien les deux familles |

**Axes empruntés** : aucun du tableau de la fiche — ce module ne construit ni
circuit ni décision, il mesure un coefficient de `src` sur trois jeux de
paramètres (`v1_test_default`, `deployed_openloop`, `level3_trained`) et
quatre scénarios. C'est un diagnostic de mappeur, pas un chemin de décision.

### `study/h3_representation/h3_equivariance.py` — lu en entier

Troisième consommateur du schedule de D-48. Lu pour savoir si un nombre publié
en dépend — **non** : `aggregate_master_table.rows_t12` ne lit que les routes
`classical` et `ground_state`, jamais `qaoa`. La route QAOA est écrite dans
l'artefact mais aucune ligne du master table ne la cite.

| lu | verdict |
|---|---|
| `SYMMETRY_OPS`, cohérence tableau / composantes | **saine** — `flip0` = réflexion selon `AXIS_X`, matrice `diag(−1, +1)`, signe axial −1 : c'est bien `B_i → det(R)·R_ij·B_j` pour un pseudo-vecteur. `rot180` et `rot90` ont `det = +1`, donc signe axial +1. `rot90` : `np.rot90(f, k=1)` envoie l'ancien `(i,j)` en `(N−1−j, i)`, soit la rotation `(f_x,f_y) → (−f_y, f_x)` que la matrice applique |
| centre des transformations, sur grille périodique | **sain** — `np.flip`/`np.rot90` opèrent autour de `(N−1)/2`, pas autour de l'origine : c'est une symétrie **composée d'une translation**, et une translation est une symétrie exacte d'un opérateur FD périodique. Le module ne le suppose pas, il le **mesure** (`solver_commutation_defect`) |
| commutation avec la réduction en patchs | **saine** — `flip0` envoie le bloc `b` sur le bloc `n−1−b` exactement, et `rot90` `(⌊i/P⌋,⌊j/P⌋)` sur `(n−1−⌊j/P⌋,⌊i/P⌋)`, dès que `P` divise `N`. Les frontières de blocs sont préservées, donc décider-puis-transformer et transformer-puis-décider portent sur le même pavage |
| `solver_commutation_defect` | **sain** — `dt` figé une fois depuis l'état de référence et passé aux deux branches : le piège « le solveur ré-adapte son `dt` » (forme « variable locale non réécrite ») est évité explicitement |
| convention axiale de `B` | **non postulée, tranchée par la mesure** — les deux variantes sont implémentées et c'est le défaut de commutation qui départage |
| `solver_noise_floor` | **contrôle réel mais partiel, non mesuré ici** — le plancher d'irreproductibilité est calculé pour la route `ground_state` seulement. La route `qaoa`, que le docstring désigne comme « la route réellement déployée », est **elle aussi** stochastique et n'a pas de plancher. Le principe que le module énonce (« une erreur d'orbite sous le plancher ne mesure pas un défaut d'équivariance ») ne lui est donc pas appliqué. **Non mesuré, donc pas appelé défaut** — et sans conséquence publiée, la route QAOA n'entrant dans aucune ligne du master table |
| `orbit_error`, ordre des arguments | **sain** — `D(T(U))` puis `T(D(U))`, conforme au docstring, et la fonction est de toute façon symétrique |

### `aggregate_v2.py`, `aggregate_v3.py` — lus en entier

| lu | verdict |
|---|---|
| `aggregate_v3.status_of` / `make_row` / `collect` | **sains** — `None` devient MISSING, `--strict` sort non nul sur DIFF ou MISSING, provenance (hash git + CLI) écrite dans le `.npz`. Les extracteurs indexent par `names.index(...)`, qui lève sur un nom absent : pas de repli silencieux |
| `aggregate_v3.rows_t9`, sélection vide | **saine** — `mask.any()` faux donne `None`, donc MISSING, donc visible |
| `aggregate_v3.py`, les 44 valeurs `ref` codées en dur | **D-49** — copiées verbatim de `docs/archive/RESULTS_V3.md` / `v3_master_table_ca7f815.md` (obsolètes par déclaration du dépôt lui-même), présentées comme « single source of truth » contre `docs/RESULTS.md` : **41/44 absentes** du fichier courant. Les 9 générateurs que `scripts/run_study_v3.sh` invoque pour regénérer ces chiffres n'existent plus. Corrigé — étiquetage seul, aucun nombre touché, voir `RESULTS.md` |
| `aggregate_v2.py`, motifs `glob` de `p5_qaoa`/`p7_sa` | **sains** — vérifiés caractère par caractère contre les noms réellement écrits par `qaoa_inputs.save_results` / `ising_terms_and_annealing.save_results` ; aucun n'a jamais produit d'artefact (phase 5/7 jamais exécutées), donc jamais exercés en pratique, mais corrects |
| `aggregate_v2`, verdict « ZZ/ZZZZ add NO measurable value » | **bande codée en dur sans provenance** (`d_sten < 0.02`), même famille que le ±0,02 déjà noté dans `ising_terms_and_annealing`, et jamais cité dans `RESULTS.md`/`EVALUATION.md` (aucun `results/SUMMARY_*` n'existe — le script n'a jamais tourné jusqu'au bout). Le bloc entier est en outre imbriqué sous `d_site is not None` : une exécution où `d_sten` existe mais pas `d_site` n'imprime **aucun** verdict. Signalé, pas corrigé — ni l'un ni l'autre n'est une valeur fausse, et inventer une provenance serait pire que l'absence |

### `aggregate_master_table.py` — lu en entier

Dernier fichier non relu de `study/common/` ; exécuté à chaque passe
(`180 / 164 / 16 / 0`) mais jamais relu fonction par fonction avant cette
passe.

| lu | verdict |
|---|---|
| `_mean_where`, `collect`, `to_markdown`, sorties `.md`/`.csv`/`.npz` | **sains** — délèguent `status_of`/`make_row` à `aggregate_v3` (déjà audité), pas de logique de comparaison réimplémentée ici |
| `TOL = 0.002` en tête de fichier | **mort, sans conséquence** — jamais référencé ailleurs dans le module ; `make_row` importé utilise son propre défaut (`aggregate_v3.TOL`, également 0,002). Les deux valeurs coïncidant, aucune ligne n'en dépend ; noté, pas corrigé |
| les 12 lignes T17 (`spearman C/w`, `ZZ mass kept`), dictionnaires `ref` codés en dur | **D-58** — recopient le défaut que D-9 a corrigé, pas son résultat : la moitié des lignes `DIFF` du master table (12 sur 16) vient de ces deux dictionnaires, jamais mis à jour après `107c1cf` |
| les autres extracteurs (`rows_t11`, `rows_t11b`, `rows_t12`, `rows_t13`, `rows_t13_degeneracy`, `rows_t14`, `rows_level3`, `rows_t18`, `rows_t20`, `rows_t22`, `rows_t23`, `rows_t24`, `rows_t25`, `rows_t26`, `rows_t15c`) | **sains** — rejoués contre les artefacts réels de `results/`, chacun retombe sur la valeur affichée par le master table (164 OK sur 180, dont les 4 explicables par D-48/T12-dim8/D-58 restent DIFF pour la raison déjà connue) |

**Axes empruntés.** Aucun — ce module ne construit ni circuit ni décision,
il relit des artefacts déjà produits par 15 tâches différentes et compare à
des constantes. Le rejouer against `results/` (Re=400, N=256 et N=64 selon
la tâche, 4 folds Level-3, dim 2/4/8 selon `t26`) est la seule forme de test
qui s'applique ici, et c'est celle utilisée pour trouver D-58.

`study/common/` **est maintenant lu en entier**, module par module.

**Aucun de ces trois modules (`qaoa_inputs.py`, `aggregate_v2.py`,
`aggregate_v3.py`) n'est « audité » au sens de la fiche** : ils
ont été lus en entier et, pour `qaoa_inputs.py`, mesurés là où une mesure
tranchait — mais aucun test ne traverse leurs axes avec des données réelles
(seuls les extracteurs purs de `aggregate_v3.py` le sont, sur données
synthétiques, via `tests/study/test_t10_aggregate.py`). Axes empruntés par
les mesures de D-48 (`qaoa_inputs.py`) : bras **quantique**, backend
**state_vector**, hamiltonien **non nul**, bord **périodique**, warm start
**présent *et* absent** (c'est la mesure elle-même), optimiseur **COBYLA**,
AMR **depth = 0**, `dim = 2` seulement. Restent non traversés : le bras
`classical_only`, le backend échantillonné, le bord borné, l'hamiltonien
nul, les autres optimiseurs, `depth > 0`, et `dim = 4 / 8`. `aggregate_v3.py`
ne peut plus être traversé structurellement (D-49) au-delà de
`upper_bound_loso_*`, seul artefact d'entrée encore présent.
### Audité le 13 août — les quatre poches partielles de V1

Ces quatre-là restaient ouvertes parce qu'aucun test ne **traversait leur
configuration** : le mode matériel, le mode Colab, le bras `classical_only`,
la mémoire TTL. Un module dont chaque fonction a été relue reste partiellement
audité tant qu'un axe de configuration n'a jamais été exécuté.

| poche | verdict |
|---|---|
| branches matériel de `VQA/execute.py` | **D-48** — `mode="hardware"` s'exécutait sur un simulateur sans le signaler |
| mémoire TTL de `Simulation/refinement.py` | **vérifiée et trouvée saine** |
| bras `classical_only` de `pipeline.py` | **vérifié et trouvé sain** |
| mode Colab de `train_hyperparams.py` | **vérifié et trouvé sain** |

**D-48.** `self.mode` était **stocké et lu nulle part** : `_init_backend` ne
dispatche que sur `backend_name`. Mesuré, `mode="hardware"` rend
`AerSimulator` pour `state_vector` / `matrix_product_state` / `aer` et
`FakeFez` pour `estimator` — identique à `mode="simulator"` dans les quatre
cas. Le piège n'est pas que ça casse : `Session(backend=AerSimulator)` est
**acceptée** par qiskit-ibm-runtime. Un run demandé en matériel ouvrait donc
une session autour d'un simulateur, y construisait un estimateur avec
découplage dynamique et twirling, et rendait des nombres plausibles. Refus
posé aux trois sites (`VQARuntime._validate_mode`, `execute`, et les choix
`--mode` de `pipeline.main`).

**TTL** — contrat d'**un pas de grâce** confirmé sur l'arbre entier : 20
entrées passent 1 → 0 → 0. L'hypothèse « une entrée périmée survit parce que
le TTL du parent maintient ses enfants visités » a été **mesurée et
réfutée** ; elle est épinglée par un test, pas abandonnée en silence. Carte
bornée à 20 entrées sur 5 pas.

**`classical_only`** — déterministe jusqu'au dernier chiffre, insensible à
`classic_AMR_comp`, et porte `sigma_source = "loaded"` comme les autres
sorties (D-36). *Observation à refaire à l'échelle : dans la configuration
réduite, `patch_ratio = 1,0` au seuil déployé.*

**Colab** — les **trois** copies vers Drive sont sous `if IN_COLAB`, vérifié
sur l'**AST** et non par proximité de texte : la lecture par proximité en
comptait deux. `ensure_dirs` idempotent et silencieux. *Risque opérationnel,
sorti en décision dans `DEFAUTS.md` : hors mode distribué, la base Optuna
n'atteint Drive qu'un essai sur dix, et elle vit sur un disque éphémère.*

Couvert par `tests/pipeline/test_v1_partial_pockets.py` — **18 tests**.

---

## 2. Ce qui est couvert, et par quel type de test

### Les cinq familles

**Tests analytiques** — une entrée à réponse connue, une sortie exacte.
Champ construit pour que la bonne réponse soit calculable à la main.
*Exemple : une rotation solide doit donner ω = +2,0 ; l'enstrophie d'un
cisaillement `vx = sin y` vaut 2π².*

**Audits de contrat** — cinq questions, posées à chaque fonction :

1. pourquoi existe-t-elle ?
2. que promet sa docstring ?
3. consomme-t-elle ce que sa signature annonce ?
4. deux chemins censés coïncider coïncident-ils encore ?
5. **un test traverse-t-il cette configuration ?**

**Douze des 37 défauts viennent de la quatrième question** — c'est de loin la
plus rentable. La cinquième a été ajoutée le 13 août, après D-48 : un module
dont chaque fonction a été relue reste partiellement audité tant qu'un axe de
configuration n'a jamais été exécuté. Les axes de ce dépôt : profondeur AMR
0 / >0, patch périodique / borné, quantique / `classical_only`,
`state_vector` / échantillonné, warm start absent / présent, hamiltonien nul /
non nul, COBYLA / autre optimiseur.

**Tests de trace** — on force une valeur connue à l'entrée et on vérifie
qu'elle ressort à la bonne place, en passant par le vrai chemin. *Exemple :
le qubit k excité doit ressortir en position k, de `init_qbits_state`
jusqu'aux marginales.*

**Tests d'épinglage** — figent l'ancien comportement d'un défaut corrigé,
pour que la correction ne puisse pas être défaite en silence. *Un test qui
n'a jamais échoué n'a jamais rien prouvé.*

**Mesures d'ordre** — marquées `slow` : refont une trajectoire à plusieurs
résolutions temporelles. Quelques minutes chacune.

### Par sous-système

| dossier | modules | couverture de ligne | contrat audité |
|---|---|---|---|
| `tests/solver/` | `solver.py`, `grid.py`, `pre_compute_dns.py` | **99 / 90 / 98 %** | opérateurs, projection, scénarios, trace DNS |
| `tests/mapping/` | `PhysToAngle`, `HamiltParams`, `HamiltParams_v2`, `RescaleArrays` | **100 / 99 / 100 / 97 %** | **complet** |
| `tests/quantum/` | `VQA/*` | 88–100 % sauf `execute` (64 %) | hamiltonien, chaîne de décision, runtime, refus du mode matériel |
| `tests/amr/` | `refinement.py`, `utils.py` | 82 / 65 % | pavage, rééchantillonnage, encodage du patch, mémoire TTL |
| `tests/pipeline/` | `pipeline.py`, `hyperparams_loader.py`, `train_hyperparams.py` | 52 / 55 / **90 %** | provenance des hyperparamètres, espace de recherche, budget d'essais, routage des 8 phases, campagne miniature, bras `classical_only` |
| `tests/study/` | tout `study/` | non mesuré | **en cours** — l'agent tient `study/pipeline/` et `study/common/` |

---

## 3. Ce qui rend un test digne de confiance

Un test peut passer sans rien prouver. Six pièges rencontrés dans ce
dépôt, chacun ayant coûté du temps :

**Le champ qui ne sépare pas.** Sur Taylor-Green, deux conventions de
rotationnel opposées rendent la **même** enstrophie, par symétrie de leurs
carrés. Avant d'écrire un test : *sur quelle entrée les deux hypothèses
divergent-elles ?*

**L'opérateur non assorti.** Mesurer la divergence d'un champ avec un stencil
différent de celui qui l'a produit ne mesure pas le champ, mais l'écart entre
deux opérateurs. **Cinq occurrences ici**, dont une où un défaut de huit ordres
de grandeur restait invisible, et une où une correction *paraissait* fausse
(2,1e−05) alors que c'est la mesure qui l'était.

**Le balayage vide.** Une commande `pytest -k` dont le motif ne correspond à
rien sort en vert. Vérifier le **nombre de tests sélectionnés**, pas le code
de retour. *Trois commandes sur vingt-deux d'un registre de vérification ne
sélectionnaient rien — dans le fichier même censé détecter ce piège.*

**Le contrat lu à moitié.** Une fonction dont le contrat inclut déjà une
transformation, et un appelant qui l'applique une seconde fois. Testée seule,
la fonction est correcte ; testé seul, l'appelant est correct ; c'est leur
composition qui est fausse. *D-37 : `_process_score` ajoutait le halo, son
appelant le redemandait — écart de 41 % à toute profondeur > 0.* Le test qui
le voit exerce la **paire**, pas les deux moitiés.

**Le test qui demande seulement si ça passe.** Un appel qui n'explose pas ne
prouve rien sur ce qu'il a fait. *scipy accepte Powell puis jette
silencieusement ses `constraints` ; `Session(AerSimulator)` est acceptée par
qiskit-ibm-runtime.* Dans les deux cas l'exécution réussit et rend des
nombres plausibles. Il faut assertir la **grandeur** — `max|β| ≤ π/(4·reps)`,
le backend réellement résolu — pas le code de retour.

**Le seuil périmé.** Un test calibré sur la mesure du jour cesse de mesurer
dès que le code change légitimement. Il ne s'actualise pas : il se
**remesure**, avec l'ancienne et la nouvelle valeur consignées. Et si la
grandeur s'avère non reproductible, on change de **grandeur**, pas de seuil.

---

## 4. Ce qui est reproductible

| | état |
|---|---|
| tests | **1 971**, déterministes sauf les suites QAOA |
| nombres publiés recalculés depuis leur artefact | **164 / 180** |
| écarts en attente | **16** — les nombres déplacés par les corrections |
| artefacts portant hash git + arguments CLI | tous les `.npz` |
| entrée non reproductible par une commande | `results/hyperparams/` — et sa provenance est rompue, voir `DEFAUTS.md` |

**Le bras QAOA n'est pas déterministe.** Sur 45 paires d'appels identiques :
dispersion de 1,79e−1 à 3,61e−1, auto-corrélation de rang médiane 0,933.
Les **valeurs** bougent, le **classement** tient — les conclusions fondées
sur un ordre sont robustes, celles qui reposeraient sur une valeur ne le
seraient pas.

Conséquence pratique : **avant de conclure sur un écart, mesurer la variance
de la mesure elle-même.** Deux estimations du même contraste ont différé
d'un facteur 3,5.

---

## `study/closed_loop/` — lu en entier, terrain neuf

Les 9 fichiers (~2 500 lignes), jamais mentionnés ici ni dans `DEFAUTS.md`
avant cette passe. `h1_solver`/T14 venait de se fermer (D-72/D-73), sans
entrée ouverte sur ce module : terrain neuf, comme le prescrit l'ordre du
travail de la fiche.

Le module diffère du reste de `study/` sur un point de fond : il n'appelle
**aucun** des modules déjà audités (`h0_selection`, `h1_solver`,
`h3_representation`, `hard_patch_labels`, `dns_extension`,
`ising_terms_and_annealing`) — il pilote directement `src/pipeline.py` et
`src/train_hyperparams.py` (le chemin **déployé**), plus
`study/common/stats_confirmatory.py`, `study/common/provenance.py` et
`study/h2b_prediction/h2b_feature_selection.git_commit_hash` (déjà sains,
voir §1b). Les défauts historiques D-39/D-69/D-72/D-73 — un consommateur
resté sur l'ancien opérateur ou l'ancienne définition d'un module amont
corrigé sous lui — n'ont donc **pas de prise ici** : il n'y a pas de
duplication de calcul entre `closed_loop/` et `study/pipeline/`
ou `study/common/`, seulement des appels à la fonction vivante.

| fichier | verdict |
|---|---|
| `closed_loop_status.py` | **sain** — lecture Optuna en mode `?mode=ro` explicite (ne peut ni bloquer ni corrompre un writer concurrent), erreur SQLite consignée plutôt qu'avalée |
| `closed_loop_campaign.py` | **sain** — `fold_scenarios` dé-doublonne `SCENARIOS_ALL` par garde, mais la correction amont (D-33 lignée, `SCENARIOS_ISOLATED` restaurée à 4 scénarios distincts) fait qu'elle ne retire plus rien aujourd'hui ; vérifié en relisant `train_hyperparams.py:820-874` que les 6 clés sont bien distinctes. `FROZEN_DEFAULTS` (`gamma_hydro=2.0, gamma_mag=0.5, kappa=10.0`) vérifié **identique à l'octet** à `config.TRAINED_GAMMA_HYDRO/MAG/KAPPA` et à `PHASE1_SEED_GRID` — même « valeur V1 de référence » partout, pas de fourche silencieuse. `summarise()` construit `q`/`c` non filtrés et `delta` filtré par `np.isfinite` sous la même clé `s[k]` : question 4 posée explicitement (un appelant qui indexerait `delta[i]` en pensant lire `q[i]-c[i]` lirait la mauvaise paire dès qu'un NaN a été filtré) — vérifié, `main()` recalcule `q[i]-c[i]` directement pour l'affichage par fold et ne lit `delta` que pour la statistique globale, qui n'a pas besoin de l'alignement par fold |
| `closed_loop_budget_matched.py` | **D-74** — seul fichier des 9 sans `assert` ni `raise` ; ses deux gardes d'entrée rendaient la main code 0 sans artefact. Corrigé. Le reste est sain : la bissection suppose `patch_ratio` décroissant en `threshold_amr` — vérifié contre `refinement.py:369,401` (`local_prob >= effective_threshold` → raffiner, donc seuil haut ⇒ moins de raffinement) sur les deux chemins (`_run_level` VQA et `_run_level_classical`), même sens des deux côtés |
| `closed_loop_divergence_audit.py` | **sain** — `parse_abort` cherche la marque `[ABORT]` que `pipeline.py:621` émet uniquement `if verbose`, et `audit_arm` appelle bien `run_arm(..., verbose=True)` : pas de garde muette côté source. `DIVERGENCE_PENALTY = 10.0` recopié localement plutôt qu'importé de `src/pipeline.py` (repli sans provenance en puissance) — **vérifié identique** aux deux endroits aujourd'hui, donc pas un défaut mesuré ; à réimporter si un jour ça diverge. La fusion `merged[...]` en fin de script relit l'audit existant avant d'écrire, en commentaire explicite contre le défaut D9 (perte silencieuse d'un sous-ensemble déjà audité) |
| `closed_loop_endpoint_wellposedness.py` | **sain** — `crossover_lambda` vérifié analytiquement (`combined_q(λ)=combined_c(λ)` résolu en λ, formule assortie au code) ; `combined()` est affine en λ à `patch` et `phys` fixés, donc un seul point testé au-delà de `lambda*` suffit à trancher le signe pour tout λ plus grand — vérifié algébriquement, pas supposé |
| `closed_loop_fold_synthesis.py` | **sain** — `primary_analysis` relit le `combined` **stocké** par `t15` ; vérifié contre un recalcul indépendant `(phys+0,4·patch)/1,4` sur les 4 artefacts réels du dépôt (`ot/kh/rotor/tearing`, bras `qhas` et `classical`) : **8/8 identiques** à 1e−6. `load_divergence_audit` traite l'absence d'audit comme « inconnu », jamais comme « valide » |
| `closed_loop_headline_counts.py` | **sain** — recompte vérifié contre l'artefact réel `t23_headline_counts.json` committé : totaux 18 complétés / 18 moins fidèle / 16 plus coûteux / 16 dominé, conformes au docstring et recalculés à la main depuis `t20_qhas_run_variance_*.json` + `t15b_budget_matched_*.json` |
| `closed_loop_leak_free_summary.py` | **sain, une réserve non corrigée** — `ratio_vs_frontier = qe/ref if ref else None` traite un `ref` (phys_score classique interpolé) exactement nul comme absent : la forme « zéro confondu avec valeur manquante » que `D-46`/`h2b` ont déjà rencontrée ailleurs. Vérifié sur les 8 lignes de l'artefact `t24_leak_free_summary.json` réellement commité : aucune ne vaut 0 (0,015 à 1,80), donc **non emprunté** aujourd'hui — signalé, pas corrigé, comme le veut la règle « mesurer avant d'affirmer » |
| `closed_loop_run_variance.py` | **sain** — `guarded()` capture systématiquement le statut d'avortement (question 4 : un tirage QAOA divergent ne peut pas se détecter après coup en le rejouant vu la non-déterminisme, donc il faut le capturer *pendant*) ; la référence classique choisie est **toujours** le point budget-apparié quand il existe, avec la source imprimée (`ref_source`) qui nomme explicitement un repli sur le bras réglé quand `t15b` n'existe pas encore — écart avec le commentaire du fichier (« TOUJOURS le point budget-apparié ») qui n'est cependant jamais silencieux : la déviation s'auto-déclare dans la sortie |

**Axes empruntés** (table de `VIGIL_BA_Proj.md`, plus l'axe « anomalies
avancées » ajouté par D-51) : AMR **`depth > 0`** exclusivement — les 6
scénarios d'entraînement portent tous `max_depth_override = MAX_DEPTH_TRAINING
= 4` (`src/train_hyperparams.py:88`), c'est le **seul** module de `study/`
dont l'exécution réelle traverse ce côté-là de l'axe plutôt que `depth = 0`
(voir la note « aucun n'emprunte » de la fiche pour tous les modules relus
avant celui-ci) ; bras **quantique et `classical_only`**, les deux à chaque
fold ; backend **échantillonné** (`shots = 256`, `Estimator`/`Sampler` sans
`seed_simulator` — c'est l'objet même de D11/T20) ; warm start
**présent** — c'est le chaînage réel de `refinement.py` (`warm_start_cache`),
pas le schedule constant de D-48, qui ne vit que dans `qaoa_inputs.py` ;
optimiseur **COBYLA** (défaut du dépôt, jamais changé ici) ; hamiltonien
**non nul** sur les runs qui aboutissent ; **anomalies avancées `True` sur
6/6 scénarios** (`AdvAnomaliesEnable`) — le seul chemin de `study/` qui
emprunte ce côté de l'axe que D-51 documente comme jamais traversé ailleurs.
Non traversés par ce module : bord **borné** au sens Hamiltonien (le
solveur MHD est toujours périodique ici), `dim` autre que celui de la
production, `depth = 0`.

**Un défaut trouvé, corrigé** : D-74 (ci-dessus), `RESULTS.md`.

**Ce que la lecture n'a pas fait.** Aucune campagne n'a été relancée — les
9 fichiers ont été lus en entier et croisés contre les artefacts déjà commités
dans `results/` (`t15`, `t15b`, `t15c`, `t19`, `t20`, `t21`, `t23`, `t24`),
pas rejoués de bout en bout (`closed_loop_campaign.py --n-trials 40` coûte
des heures par fold, hors budget d'une passe Vigil). Les recalculs
indépendants ci-dessus (combined à 1e−6, totaux T23 à l'unité) portent donc
sur la cohérence du **code contre ses propres artefacts déjà écrits**, pas
sur une reproduction de la campagne depuis zéro.

`study/h4_transfer/` (consommé par `closed_loop_leak_free_summary.py` via
les artefacts `t22_unseen_*`) reste **non lu** : hors périmètre de cette
passe, qui portait sur `closed_loop/` seul.

## `study/h2b_prediction/` — passe du 14 août, lu en partie, et le partiel est dit

19 fichiers, **6 549 lignes**, terrain neuf : le module n'apparaissait ni ici
ni dans `DEFAUTS.md`. `closed_loop/` s'était fermé la veille sans entrée
ouverte, donc c'était le suivant dans l'ordre de la fiche.

**Lu en entier, fonction par fonction — 7 fichiers, ~2 460 lignes :**
`h2b_ceiling_random_split.py` (461), `h2b_loso_transfer.py` (239),
`h2b_loso_delta_ci.py` (240), `h2b_dynamic_ground_truth.py` (295),
`h2b_feature_selection.py` (270), `h2b_train_linear_hamiltonian.py` (545),
`h2b_variational_classifier.py` (362).

**Lu partiellement, et seulement autour du défaut trouvé** :
`h2b_scenario_ablation.py` (la boucle LOSO et ses quatre bras — D-82),
`h2b_psi_feature_loso.py` (les helpers purs `signed_combine`,
`psi_signed_v1`, `psi_abs_v1`, `block_agg` : vérifiés cohérents entre eux et
avec la formule que la docstring pré-enregistre ; la boucle principale non).

**Les 10 restants n'ont PAS été lus.** Ils n'ont été traversés que par trois
balayages mécaniques : l'audit AST des gardes d'entrée (D-75), la
vérification que chaque script accepte encore les drapeaux des lanceurs
(D-76), et la relecture des **30 appels de `best_threshold_f1` de tout
`study/`** déclenchée par D-81 — c'est ce dernier qui a donné D-82, et il
n'en a trouvé que deux fautifs sur trente. Dire « module audité » ici serait
faux : au sens de ce document, un fichier n'est audité que quand ses
contrats ont été lus et qu'un test emprunte ses configurations.

**Vérifié et trouvé sain**, mesuré et non supposé :

| ce qui a été vérifié | comment |
|---|---|
| alignement instantanés DNS ↔ `patches_*` | 20 instantanés de part et d'autre, vecteur `t` identique — un décalage aurait étiqueté chaque cellule avec le label d'un autre temps |
| ordre de raveling `patch_l2_errors` ↔ `_block_avg` | même ordre C sur `(dim, dim)` : les labels et les features décrivent la même cellule |
| `fit_eval` : seuil choisi sur **train**, appliqué sur val | pas de fuite du seuil ; la même discipline dans les quatre scripts qui l'importent |
| `bootstrap_by_trajectory` tel que T29 l'utilise | bloc = instantané ; les indices passés comme valeurs flottantes et re-castés dans la statistique — le contrat est respecté |
| `reference_evolution` (T6) contre le défaut-modèle de `VIGIL.md` | `dt = min(sim.dt, reste)` **puis `sim.dt = dt`** : le solveur relit bien la valeur tronquée, ce n'est pas la « variable locale non réécrite » |
| `coarsen_patch_window` (T6) ↔ `coarsen_field` de la phase 2 | moyenne de bloc restreinte à une fenêtre, identique |
| `Jz` écrit à la main dans `hard_patch_labels` ↔ `solver.get_fluxes` | stencils identiques à l'octet — c'est le rotationnel **interne** à `classical_score` qui diverge, voir D-77 |
| `fit_learned_h` : le champ en unités physiques ↔ le champ standardisé | `x·w_raw + b_raw` et `z·w_std + b_std` coïncident à **3,6e−15** sur 400 points ; la colonne à variance nulle (`Re` à un seul Reynolds) donne `scale_ = 1`, pas une division par zéro. **Réserve levée au passage** : la docstring annonçait `h = w·φ − b` alors que l'intercepte rendu est **additif** — appliquée telle quelle elle décale le champ de `2·b_raw` (mesuré : 2,01). Aucun consommateur du dépôt n'était touché (`h2b_blocked_split` et `h2b_scenario_specialisation` passent par `predict_h`) ; le contrat est aligné sur le calcul |
| `rng.shuffle(Xp[:, k])` de l'importance par permutation | une tranche de base est une **vue** : la colonne est bien permutée dans le parent, les huit autres intactes (vérifié) — pas le piège du `copy()` implicite |

**Configurations empruntées** — ce qui a réellement tourné pendant la passe :

| axe | emprunté | non emprunté |
|---|---|---|
| `dim` | 2, 4, 16 | 3, 8, 32, 64 |
| variante de label | par scénario | `_globalthr`, `tau*` |
| folds LOSO | les 4 scénarios canoniques | — |
| `src/` | HEAD **et** `bb6a387^` (avant D-1) | — |
| optimiseur de la phase 10 | la branche de **repli** Nelder-Mead (`cma` absent du conteneur) | la branche CMA-ES |
| balayage vide | les 13 scripts mesurés code 0 → 1 | les 2 gardes non atteignables en ligne de commande (D-75) |

**Ce que le module produit réellement** : sur 19 scripts, **deux seulement**
ont un artefact dans `results/` — `upper_bound_*` (8 fichiers) et `t29_*`
(4). Les tâches T1, T1b, T4, T5, T6, T7 n'ont laissé aucun `.npz` : leur code
existe, leur sortie non. À savoir avant de citer un de leurs nombres.

**Défauts trouvés** : D-75 (12 sites ici sur 15), D-77, D-78, D-79, D-80,
D-81, D-82 — détail et mesures dans `RESULTS.md`.

**Une discipline vérifiée sur tout `study/` — et mon premier balayage était
faux, il est corrigé ici.** J'avais écrit « 30 appels de `best_threshold_f1`,
deux exceptions ». Les deux chiffres étaient faux, et une session concurrente
l'a montré en réservant **D-83** sur un site que j'avais manqué.

La cause est exactement le piège que `VIGIL.md` décrit : j'avais balayé au
**`grep`**, qui coupe les appels écrits sur plusieurs lignes. Refait à
l'**AST**, en dépliant chaque appel et en lisant ses deux premiers arguments :

| | |
|---|---|
| appels de `best_threshold_f1` dans `study/` | **37**, pas 30 |
| seuil pris sur le **train** (la règle du dépôt, celle de `fit_eval`) | **32** |
| seuil pris sur la **validation**, avant cette passe | **3** — `h2b_variational_classifier` ×2 (D-81), `h2b_scenario_ablation` (D-82), `h2b_random_split_bootstrap` (**D-83**, session concurrente) |
| faux positifs de mon balayage `grep` | `h2b_learned_meanfield_h:245,247` — `Yt` y désigne les folds d'**entraînement** du LOSO, pas un jeu de test ; sains |
| appels sur validation **délibérés**, ajoutés par D-81/D-82 | 4, sous le nom `f1_*_thr_on_val` : ce sont les anciens nombres, gardés pour que le biais reste mesurable |

Les trois sites fautifs sont tous sur le bras dont le script cherche à
mesurer l'avantage ou la chute. **Un balayage au `grep` ne suffit pas pour
ce genre d'audit** — c'est la troisième fois que ce dépôt le paie (D-56 :
« trois des onze sites ont été trouvés par l'AST, pas par la recherche de
chaîne »).

**Reste à faire sur ce module**, dans cet ordre : les 13 fichiers non lus,
en commençant par ceux qui portent un artefact (`h2b_ceiling_random_split`
est fait, restent les consommateurs de `upper_bound_*`) ; puis la variante
`_globalthr` de T29, jamais rejouée ici.

---

## `study/h2b_prediction/` — seconde passe du 14 août, 4 fichiers de plus

Passe concurrente de la précédente sur le même module ; les deux se sont
croisées sur D-82 (voir le fil de la PR). Lus **en entier, fonction par
fonction** : `h2b_scenario_ablation.py` (264), `h2b_random_split_bootstrap.py`
(252), `h2b_v1_hamiltonian_loso.py` (412), `h2b_blocked_split.py` (405) —
environ 1 330 lignes. **Défauts trouvés** : D-83, D-84, D-85 (détail et
mesures dans `RESULTS.md`).

**Vérifié et trouvé sain**, mesuré et non supposé :

| ce qui a été vérifié | comment |
|---|---|
| appariement du bootstrap de la phase 11H | `bootstrap_ci` et `paired_delta_ci` tirent **un seul** jeu d'indices par réplique et l'appliquent aux deux bras : l'appariement est réel, pas nominal |
| `split_indices_random` (T4) contre la phase 11A | même convention à la ligne près (`tr = perm[:n_tr]`, `h2b_ceiling_random_split.py:308-310`) |
| garde de cohérence d'agrégation de T4 | traversée par **exécution** (`--dim 4 --N 256`) : `raise RuntimeError("score aggregation mismatch")` ne se déclenche pas — le canal `score` de `extract_features_2d` est bien `block_max(full_score)` |
| ordre des blocs de `ranking_metrics_per_snapshot` | `np.split(scores_va, len(va_idx))` suit l'ordre de `va_idx`, celui de `e_snaps` : chaque classement est comparé à l'erreur de **son** instantané |
| `AngleMapper.classical_score` et la clé `dx` absente | la fonction ne lit pas `dx` ; le `physics_state` de `v1_state`, qui ne la porte pas, ne déclenche aucun repli silencieux |

**Noté, non corrigé** — ni l'un ni l'autre ne change une valeur :

* 11G et 11H prennent la convention de split **complémentaire** de 11A
  (`va = perm[:n_va]` contre `tr = perm[:n_tr]`) : à graine et permutation
  identiques, leurs ensembles de validation sont **disjoints** de celui de
  11A. Aucune des trois ne prétend rejouer le split d'une autre, donc ce
  n'est pas un défaut — mais la docstring de 11H annonce un IC sur « le
  nombre de tête du split aléatoire », qui est un nombre d'archive (0,989,
  `docs/archive/`, obsolète par la règle du dépôt).
* `gap_hi = st_hi - s_lo` (11H) est calculé et jamais utilisé ; la ligne
  imprimée juste après lit `dst_hi`. Code mort, pas une valeur fausse.

**Configurations empruntées** — ce qui a réellement tourné :

| axe | emprunté | non emprunté |
|---|---|---|
| `dim` | 4 ; 16 et 32 au balayage de biais de D-83 | 2, 3, 8, 64 |
| `N` | 256 ; 64 pour la vérification de bout en bout de T4 | 512 |
| graine | 0, 1, 2 (D-83) | — |
| `Re` | 400 — le seul qui ait des artefacts à N=256 | 800, 1200, 1600 |
| réduction du score | `block_avg` **et** `block_max` (D-84) | — |
| discipline de seuil | train **et** validation, mesurées côte à côte (D-83) | — |

**Reste à faire sur ce module** : 7 fichiers encore non lus —
`h2b_analytical_solution`, `h2b_multiseed`,
`h2b_prediction_horizon`, `h2b_scenario_specialisation`, `h2b_loso_bootstrap`,
et les deux lus seulement en partie (`h2b_psi_feature_loso`,
`h2b_learned_meanfield_h`). Aucun n'a d'artefact dans `results/`.

---

## `study/h2b_prediction/h2b_neighbour_cone_curve.py` (T1b) — D-88

Lu en entier, fonction par fonction (~450 lignes). **Défaut trouvé** : D-88
(détail et mesures dans `RESULTS.md`) — `n_feats` comptait les colonnes
NOMINALES d'une boule de Chebyshev, pas les colonnes réellement distinctes
qu'`np.roll` périodique produit ; à dim=4 la courbe de cône traite k=2 et
k=3 comme deux voisinages de tailles différentes (225 puis 441 features)
alors qu'ils rendent tous deux les 144 mêmes colonnes.

**Configurations empruntées** — ce qui a réellement tourné :

| axe | emprunté | non emprunté |
|---|---|---|
| `dim` (empreinte périodique de `khop_offsets`/`khop_features`) | 4 (sature à k=2), 8 (champ séparateur : aucune saturation à k≤3) | 2, 3, 16, 32, 64 |
| `k` | 0, 1, 2, 3 (les quatre points de la courbe) | — |
| chemin bout-en-bout (`main`, LOSO/split bloqué sur DNS réelles) | non emprunté — `results/` ne porte aucun `dns_*`/`patches_*` pour ce script, ni `t1b_cone_curve_*.npz` | tout `main()` |

Les fonctions pures (`khop_offsets`, `khop_features`,
`khop_distinct_footprint`, `blocked_split_indices`, `capped_model_factory`)
sont vérifiées à l'opérateur assorti (mesure directe sur
`np.unique(khop_features(...), axis=1)`, pas de réimplémentation
parallèle). `main()` (chargement DNS, boucle LOSO/bloqué, sauvegarde) n'a
jamais tourné sur ce dépôt — aucune donnée d'entrée disponible — et reste
non audité par exécution.

---

## `study/h2b_prediction/` — passe du 15 août, dernier lot : le module est lu en entier

Les 7 items que la section précédente listait sous « reste à faire » : le
module n'avait pas d'entrée ouverte dans `DEFAUTS.md` (D-75 à D-85 et D-88
tous corrigés), donc terrain neuf au sens de l'ordre du travail de la fiche.
Ligne de base avant lecture, `tests/study -q -m "not slow"` :
**952 passed, 62 skipped, 1 xfailed** — verte, aucun échec préexistant.
Suite complète non relancée (hors budget d'une passe, comme pour
`closed_loop/` le 14 août) ; la ligne ci-dessus couvre les 65 fichiers de
`tests/study/`.

**Lus en entier, fonction par fonction — 4 fichiers** :
`h2b_multiseed.py` (232), `h2b_loso_bootstrap.py` (309),
`h2b_scenario_specialisation.py` (298), `h2b_learned_meanfield_h.py` (291,
déjà partiellement audité pour `fit_learned_h` — relu en entier ici, `main()`
compris). **Relu en entier après une première lecture partielle** :
`h2b_psi_feature_loso.py` (`_gather`/`main`, les helpers purs l'étaient
déjà) et `h2b_prediction_horizon.py` (611, lu en entier — c'est le plus
long fichier du module). **Revérifié, pas relu** : `h2b_analytical_solution.py`
— ses deux défauts (D-86, D-87) étaient déjà corrigés et testés
(`test_phase10a_argmax_is_not_a_grid_edge.py`,
`test_phase10a_flat_sweep_is_not_an_optimum.py`) ; la question 4 posée en
plus ici (ci-dessous) n'en rouvre aucun.

**Aucun défaut trouvé.** Vérifié et non supposé :

| ce qui a été vérifié | comment |
|---|---|
| `h2b_analytical_solution.mf_f1_curve`, « build avec c_bias=1, mise à l'échelle après coup » | **l'hypothèse tient** — remonté jusqu'à `HamiltParams_v2.compute_coefficients` (`HamiltParams_v2.py:203`) : `z_bias = c_bias * median_scale * (score - thr)` est bien la SEULE grandeur linéaire en `c_bias` ; `C_edges`/`K_plaquettes` (lignes 167-186) ne le lisent pas du tout. Construire une fois à `c_bias=1` puis mettre `h_unit` à l'échelle est donc exact, pas une approximation |
| discipline de seuil (train seul, jamais la validation) | **tenue dans les 4 fichiers lus en entier** — `h2b_multiseed.random_split_seed`/`loso_seed`, `h2b_loso_bootstrap.main`, `h2b_scenario_specialisation.main` (split interne ET matrice de transfert), `h2b_learned_meanfield_h.main` (split ET LOSO) : chaque appel de `best_threshold_f1`/`fit_eval` reçoit `(scores_train, Y_train)`, jamais la validation. Aucun des trois sites fautifs de D-81/D-82/D-83 (le balayage AST déjà fait sur tout `study/`) n'est dans ce lot |
| `h2b_prediction_horizon.horizon_pairs`/`blocked_pair_split` | **saines** — `horizon_pairs(n, h)` borne `t < n-h`, donc `t+h <= n-1` reste un indice valide ; `blocked_pair_split` classe une paire par le PIRE de ses deux bords (`th < t0` pour train, `t >= t0` pour val), donc une paire à cheval sur la frontière n'est ni l'une ni l'autre — anti-fuite conforme au docstring, vérifié sur les bornes plutôt que supposé |
| `h2b_prediction_horizon._assemble` | **saine** — la cible `Y`/`e` est lue à `t+h` (le futur), les features à `t` (le présent) ; le score brut des baselines classiques (`RAW_BASELINES`) est lui aussi lu à `t`, pas à `t+h` — la tâche est bien « prédire », pas « décrire » |
| convention de signe de `psi` entre `h2b_psi_feature_loso._gather` et `h2b_prediction_horizon._gather` | **identiques et correctes** — les deux calculent `dphi = phi[t] - phi[t-1]` à la main pour `psi_signed_v1`, et passent `(phi_prev, phi)` dans cet ordre à `compute_psi_v2`, dont le corps fait `delta = phi - phi_prev` (`HamiltParams_v2.py:274`) : même signe des deux côtés, question 4 posée entre les deux fichiers et entre chaque fichier et la fonction qu'il appelle |
| `h2b_prediction_horizon`, cône causal `k=2` à `dim=4` | **hérite de la saturation déjà mesurée en D-88** (`khop_distinct_footprint(2,4)` = 16 distinct sur 25 nominaux), pas un défaut nouveau — les colonnes dupliquées sont redondantes pour le GBT, pas fausses, et D-88 a déjà tranché de ne pas changer `khop_features`. Noté ici pour qu'un futur lecteur du tableau `k x h` sache pourquoi `k=2` ne gagne pas grand-chose sur `k=1` à cette taille |
| `h2b_scenario_specialisation`, diagonale de la matrice de transfert | **cohérente avec le calcul direct** — `T[i,i]` (recalculée via la boucle `[2]`) mesure la même paire (modèle entraîné sur `s`, évalué sur la validation de `s`) que `f1_spec` de la boucle `[1]`, sur les mêmes `Xva`/`Yva` |

**Configurations empruntées** (au sens de la fiche, plus l'axe « discipline
de seuil » et l'axe temporel propres à ce lot) :

| axe | emprunté | non emprunté |
|---|---|---|
| hamiltonien | non nul (`c_bias=1` et grille log, `h2b_analytical_solution`) | nul |
| bord | périodique (seul type construit par `build_patch_hamiltonian` dans ce lot) | borné |
| split | aléatoire par instantané, bloqué temporel (Task 4), LOSO — les trois, dans des fichiers différents | — |
| horizon temporel (`h2b_prediction_horizon`) | 1, 2, 4, 8 pas de la sous-séquence sous-échantillonnée | horizon = pas DNS bruts (le sous-échantillonnage `max_snaps` change l'unité) |
| modèle | régression logistique (`h2b_learned_meanfield_h`, `h2b_scenario_specialisation --model lr`) et GBT (les cinq autres) | — |

`study/h2b_prediction/` **est maintenant lu en entier, module par module,
19 fichiers sur 19** — dernier fichier restant avant cette passe :
aucun, la liste « reste à faire » de la section précédente est vide.
Non fait, comme déjà noté pour le lot du 14 août : aucune campagne
`main()` n'a été rejouée de bout en bout sur ces 4-7 fichiers (les
artefacts d'entrée `dns_*`/`patches_*` du dépôt le permettent pour
certains, pas pour tous — `d_patches_*` en particulier n'existe pour
aucune configuration, donc la branche « cible `d_i`
(Task 6) » de `h2b_prediction_horizon.main` n'a jamais tourné ici).

---

## `study/h4_transfer/` — lu en entier, terrain neuf, 2 défauts (D-89, D-90)

4 fichiers (~1 550 lignes), jamais mentionnés ici ni dans `DEFAUTS.md` avant
cette passe — `study/h2b_prediction/` venait de se clore sans entrée
ouverte, `study/h4_transfer/` était le terrain neuf suivant dans l'ordre de
la fiche (déjà signalé non lu par la passe du 14 août sur `closed_loop/`).

| fichier | verdict |
|---|---|
| `h4_unseen_conditions.py` (T22) | **sain** — construit et documente lui-même le cas `total_abort` (« un résultat, pas une panne ») qui a fait trébucher ses deux consommateurs ci-dessous ; `degradation_ratio = mu/mc` cohérent avec `phys_score` = erreur (plus bas est meilleur, `src/pipeline.py:859`) |
| `h4_unseen_floor.py` (T22d) | **D-89**, corrigé — plantait (`KeyError: 'canonical'`) sur un bras `total_abort` de T22, jamais exercé sur ce dépôt (les 4 artefacts réels ont les deux bras `completed`) |
| `h4_transfer_summary.py` (T22c) | **D-90**, corrigé — lisait `total_abort_arm` (singulier), clé absente de l'artefact réel (T22 écrit `total_abort_arms`, pluriel) : affichait « the None arm aborted » au lieu du nom du bras. Le reste du fichier (`ratio_sd`, `analyse`, dominance tirage par tirage) est **sain** — la note interne à `analyse()` (« shift/deg algébriquement identiques ») a été vérifiée à la main, elle tient |
| `h4_physics_robustness.py` (T25) | **sain** — bissection sur `patch_ratio` vérifiée dans le bon sens (seuil haut ⇒ patch bas, cohérent avec D-74) ; `frontier_verdict` refuse explicitement une interpolation sur une frontière non monotone ou mal encadrée plutôt que de rendre un ratio d'apparence normale (documenté dans le fichier lui-même comme le motif qu'il évite) ; `rng_override` ignore l'argument que l'appelant passe à `np.random.default_rng` et le remplace par la graine substituée — c'est voulu, `init_mhd_rotor` appelle `default_rng(42)` en dur |

**Ce que la lecture n'a pas fait** : aucune campagne `main()` rejouée de
bout en bout (chaque fold coûte des heures de DNS/QAOA) ; les 2 défauts
trouvés l'ont été par lecture de contrat (question 3, deux fichiers
consommant la même forme d'artefact que documente un troisième) et
reproduits par appel direct des fonctions extraites, pas par exécution de
`main()` sur données réelles.

**Axes empruntés** (au sens de la fiche) : aucun — ce module ne construit
ni circuit ni Hamiltonien, il pilote `src/pipeline.py`/`src/train_hyperparams.py`
(le chemin déployé) sur des conditions initiales alternatives, comme
`closed_loop/`. Bord périodique uniquement, bras `qhas` et `classical`
(`only=True`) tous deux exercés par construction.

`study/h4_transfer/` **est maintenant lu en entier**, les 4 fichiers du
dossier couverts.

---

## `figures/pareto_frontier.py` et `figures/pareto_panel.py` — terrain neuf, lus en entier

596 lignes (191 + 405). **`figures/` n'apparaissait nulle part dans ce
document avant cette passe** — ni dans « jamais audité », ni ailleurs :
`src/` et `study/` étant tous deux lus en entier (sections précédentes), le
dossier entier était le terrain neuf suivant au sens de l'ordre du travail
de la fiche. Ce sont les deux scripts qui produisent les figures V4
« Q-HAS contre la frontière classique » — `figures/v1_legacy/` (17 fichiers)
reste non lu, prochain terrain neuf.

**`figures/result_figs.py` — lu en entier, sain.** 133 lignes, génère les
deux figures de la falsification v1 (`fig1_ceiling_bar`,
`fig2_loso_scatter`) à partir de nombres **recopiés à la main** depuis les
logs de campagne (`results/logs_v2/Result_phase*.txt`) plutôt que calculés
— le risque propre à ce genre de fichier est une transcription silencieuse,
pas un calcul faux. Les 12 valeurs (6 barres + 4×3 points du nuage LOSO)
revérifiées une à une contre leur source : `bar_values`/`bar_errors`
coïncident avec `Result_phase7.txt:603-604` (0,409/0,336, classique/SA) et
`Result_phase13.txt:30-31` (0,989/0,991, plafonds split aléatoire) et
`Result_phase_end.txt:39` (0,191±0,152/0,215±0,142, plafonds LOSO) ;
`f1_class_loso`/`f1_site_loso`/`f1_sten_loso`/`f1_learn_loso` coïncident
avec les deux tableaux de `Result_phase_end.txt:34-37` et `:83-86`, dans
l'ordre des scénarios. Aucun écart. Réserve non corrigée, sans conséquence
sur les nombres : le commentaire d'en-tête attribue les plafonds « split
aléatoire » à `Result_phase11.txt`, qui n'en porte que le détail
par-scénario — l'agrégat qu'affiche la figure vient de `Result_phase13.txt`.

| fonction | verdict |
|---|---|
| `pareto_frontier.main()` | **D-92** — exécuté seul, reproduisait les rapports Q-HAS déjà retractés (voir `RESULTS.md`) : tirage `t15b` unique au lieu de la moyenne T20, aucun retrait des points de trace avortés (audit T19) |
| `pareto_frontier.load_points`, le triplet `tuned` | **réserve, non corrigée** — mélange `patch`/`phys` de `tuned_classical` et `thr` de `matched_classical`, deux runs différents, deux seuils différents. Sans conséquence mesurée : `tuned` n'est lu ni par `build_figure` ni par aucun appelant de `main()` dans les deux fichiers — un piège armé, non déclenché (question 1 de `VIGIL.md`), signalé plutôt que corrigé pour rester sur un seul défaut par commit (D-92 le touche déjà) |
| `pareto_frontier.interp_frontier` | **sain** — extrapolation plate hors domaine (comportement documenté de `np.interp`), jamais empruntée : sur les 4 folds réels le budget Q-HAS retombe toujours dans l'intervalle balayé par la trace |
| `pareto_frontier.build_figure` | **sain** — fenêtre verticale dynamique (`max(...) * 1.12`, pas de borne codée en dur comme D-62 dans `recompute_lambda_scores.py`) ; étiquette du point apparié indexée par `argmin` sur l'écart de budget, pas par position dans la trace |
| `pareto_panel.draw_panel`, `build_panel` | **lus, aucun défaut de valeur** — mise en page en pouces (marges, largeur de texte enveloppé) non revérifiée au pixel près : ce n'est pas une grandeur physique au sens de la mission de `VIGIL.md` |
| `pareto_panel.available_folds` | **sain** — filtre trivial sur l'existence du fichier, ordre demandé préservé |

**Ce que la lecture a montré sans que ce soit un défaut** : `pareto_panel.py`
tenait sa propre copie de `verified_qhas_point`/`load_trace_audit`/
`drop_aborted`, identique à celle qu'on aurait dû trouver dans
`pareto_frontier.py` — exactement la forme que ce dépôt a déjà rencontrée
deux fois (D-60/D-61, `_add_trend`). Déplacées dans `pareto_frontier.py`
avec D-92, `pareto_panel.py` les importe désormais au lieu d'en tenir sa
propre définition ; `tests/study/test_pareto_frontier_retracted_ratio.py`
verrouille l'identité des trois pour qu'elles ne puissent plus diverger une
seconde fois.

**Axes empruntés** : fold **ot**, **kh**, **tearing** (T20 disponible, 5
tirages, aucun point de trace avorté) et **rotor** (T20 disponible, 3
tirages sur 5 dont 2 avortés, 2 points de trace avortés retirés par
l'audit T19) — les quatre artefacts gelés du dépôt. Point Q-HAS **moyenné**
(T20 présent) et **tirage unique en repli** (synthétique, T20 absent, tests
D-92). Trace **avec** points avortés à retirer et **sans**. Non emprunté :
un vrai désaccord entre `t19_budget_trace_audit.json` et les seuils de la
trace (aucun test ne construit un seuil avorté qui ne serait PAS dans la
trace) ; l'option `--folds` de `pareto_panel.py` au-delà des 4 valeurs par
défaut.

---

## `figures/v1_legacy/fig_utils.py` et `fig0_pareto_lambda.py` — terrain neuf, lus en entier

1 396 lignes (962 + 434). `figures/v1_legacy/` était, après la passe
précédente, la seule partie non lue du dépôt. Trois défauts en sont sortis —
D-93, D-94, D-95, détaillés dans `RESULTS.md`.

**Ce qui les relie** : la réorganisation `17d983d` a réécrit le prélude
`sys.path` de chaque fichier (`_REPO_ROOT`, deux niveaux) sans toucher aux
ancres de racine **déjà présentes** dans le corps. Balayage fait sur les 19
fichiers du dossier : **deux** portaient encore une ancre à un seul niveau,
`fig_utils.py:109` (D-93) et `fig0_pareto_lambda.py:39` (D-94) — les 17 autres
n'ont que l'ancre générée. La liste est close, pas échantillonnée.

**Configurations traversées.** `FIGURE_PHASE` **absent et présent** — l'axe
propre à ce module, les deux côtés (`test_fig_utils_output_dir.py`). Les deux
motifs de `load_all_trials`, **quantique et classique**, sur les CSV gelés
(`test_fig0_pareto_paths.py`). Les **quatre** scénarios de `SCENARIOS_ALL`
pour D-95, dont `harris_tearing` qui **ne sépare pas** — aucun essai classique
sous la fenêtre quantique : un test écrit sur ce seul scénario serait passé
sans rien vérifier.

**Ce qui n'a PAS été traversé, et doit être dit** : la moitié « simulation »
de `fig_utils.py` — `qaoa_block_scores`, `run_hierarchical_comparison`,
`run_single_method`, `find_optimal_threshold`, `patches_to_metrics` — a été
**lue**, pas **exécutée**. Aucun test ne l'emprunte, et les axes de la fiche
(profondeur AMR, bord du patch, bras, backend, warm start, Hamiltonien,
optimiseur) passent tous par là. Au sens de la fiche, cette moitié n'est donc
**pas auditée** : elle est lue.

**Vérifié et trouvé sain, mesuré :**

| ce qui a été vérifié | verdict |
|---|---|
| `_hamilt_mapper_kwargs` code `nu = eta = grid.L / 800` alors que `make_sim` accepte `Re`/`Rm` | **sain** — soupçon de repli silencieux tué par la mesure : **aucun** des 17 scripts ne passe un `Re` ou `Rm` autre que 800 (balayage complet du dossier). La constante duplique le réglage, elle ne le contredit pas |
| `ground_truth_errors` lit `axis=1` pour `grad_x` et `axis=0` pour `grad_y`, à l'envers de la convention `grid.py` (`AXIS_X = 0`) | **sain** — les deux contributions sont **sommées au carré** : l'échange est exactement symétrique, la sortie est bit-à-bit la même. Ce n'est pas la forme de D-1 |
| `_patches_overlap_with_gt` prend le **max** des poids sur les patchs qui se recouvrent | **sain** — pas de double comptage : la somme des poids par pixel ne peut pas dépasser 1 |
| `interp`/`argmin`/`argmax` de `fig0` | **sain** — `extract_pareto_front` est une dominance stricte correcte ; `np.argmin(scores)` porte sur le tableau complet depuis D-95 |
| les 12 valeurs codées en dur | **sans objet** — `fig0` ne code aucune valeur : tout vient des CSV |

**Réserve mesurée, non corrigée — un piège armé, non déclenché.** Neuf
fonctions publiques de `fig_utils.py` n'ont **aucun appelant**, ni dans les 17
scripts, ni ailleurs dans le dépôt : `phase_allows_scenario`,
`make_sim_with_history`, `smoothed_classical_scores`, `compute_mean_jz_squared`,
`selection_to_mask`, `count_connected_components`, `compute_fragmentation`,
`compute_perimeter_area_ratio`, `selection_jaccard` (plus la constante
`LABELS`). Trois d'entre elles — les métriques de cohérence spatiale — **sont
fausses sur la grille de ce dépôt**, qui est périodique (`PeriodicGrid`), et
rendraient des valeurs plausibles :

| entrée | ce que la fonction rend | vérité périodique |
|---|---|---|
| `{(0,0), (3,0)}` sur 4×4, collés **à travers** le bord | 2 composantes, fragmentation **1,000** (« maximalement fragmenté ») | 1 composante, **0,500** |
| bande complète `mask[:,0]` — un cylindre fermé | périmètre/aire **2,500** | **2,000** (**+25 %**) |
| `{(1,1), (1,2)}`, au centre, ne touche aucun bord | 1 composante, 0,500, périmètre/aire 3,000 | **identique** |

La troisième ligne est le champ qui **ne sépare pas** : toute validation écrite
sur une sélection centrale passe sans rien vérifier. Non corrigé — aucun
consommateur, donc aucun avant/après à mesurer sur une sortie ; c'est à
trancher (supprimer, ou rendre périodique) le jour où une figure de cohérence
spatiale revient. Écrit ici pour que le trou ne soit pas re-trouvé une
troisième fois.

---

## `figures/v1_legacy/` — l'axe `FIGURE_PHASE` n'a qu'une valeur vivante

Mesuré en traversant l'axe des **quatre** côtés (absent, 1, 2, 3), sur
l'import de n'importe quel script du dossier :

| `FIGURE_PHASE` | ce qui se passe |
|---|---|
| absent | import OK — `FIG_DIR = results/figures` |
| `1` | import OK — `FIG_DIR = results/figures/phase1` |
| `2` | **meurt à l'import** : `KeyError: "Phase 'phase2' not in quantum training. Available: ['phase1']"`, `hyperparams_loader.py:139`, appelé par `fig_utils.py:327` |
| `3` | **meurt à l'import**, même erreur |

Les hyperparamètres déployés ne portent que `phase1`. Toute la machinerie de
phase de `fig_utils.py` — `SCENARIOS_PHASE2`, `SCENARIOS_PHASE3`,
`filter_scenarios`, `filter_scenarios_dict`, `phase_allows_scenario` — n'est
donc atteignable que dans sa branche `phase1`, où elle ne filtre rien (les 4
scénarios y sont tous). S'y ajoute que **plus rien ne pose `FIGURE_PHASE`** :
`generate_figures.sh`, que les commentaires du dossier citent comme le
poseur (`fig_utils.py:84`, `fig0:352`), a été supprimé par la
réorganisation. C'est une décision en attente — retirer la machinerie, ou
lui rendre un lanceur — pas un défaut à corriger au passage.

## `figures/v1_legacy/fig4_comprehensive_comparison.py` — lu en entier

189 lignes. Aucun défaut **vivant**. Deux réserves mesurées :

**Un piège armé, non déclenché — les étiquettes du graphe sont codées en
dur.** `short_names = ['KH', 'Tearing', 'Rotor', 'OT']` est tronqué par
`[:n_scen]`, tandis que `scen_names`, construit depuis la liste réellement
parcourue, existe dans le même fichier et sert au tableau texte. Les deux
chemins coïncident tant que `filter_scenarios` ne retire rien — c'est le cas
`FIGURE_PHASE` absent et `phase1`, les deux seuls exécutables. Ils
divergeraient sous `phase2`, dont l'ensemble est `{orszag_tang, mhd_rotor}` :
la figure étiquetterait « KH » et « Tearing » deux barres qui sont Rotor et
Orszag-Tang, pendant que le tableau texte de la même exécution imprimerait
les bons noms. Non corrigé : la configuration qui l'arme meurt d'abord à
l'import (voir ci-dessus), donc il n'y a aucun avant/après à mesurer sur une
sortie.

**Dérive de docstring, sans conséquence sur les valeurs.** L'en-tête annonce
quatre métriques dont « Efficiency = captured / compute » ; la figure trace
`['Captured Fraction', 'Precision', 'Recall', 'Compute Ratio']` — `Recall`
au lieu d'`Efficiency`. Les valeurs tracées sont justes et correctement
titrées sur les axes ; c'est l'en-tête qui est périmé.

**Vérifié et trouvé sain** : `pixel_precision` / `pixel_recall` gardent leurs
cas dégénérés (`n_refined == 0` → 0,0 ; `n_needs == 0` → 1,0) dans le bon
sens ; les barres `yerr` portent bien l'écart-type sur `N_TRIALS`, et le bras
classique étant déterministe, son `yerr` vaut 0 — ce n'est pas un défaut,
mais la barre d'erreur des deux bras ne mesure pas la même chose.

---

## `figures/v1_legacy/fig1_noise_robustness.py` — lu en entier

173 lignes. Aucun défaut de valeur. Trois observations, dont deux à retenir :

**Contrat faux, sans conséquence ici.** `inject_field_noise` annonce
*« Returns a copy of the sim with noisy fields »* — elle ne copie rien :
`setattr(sim, field_name, field + noise)` **modifie l'objet reçu**. Le script
ne s'en aperçoit pas parce que son appelant construit un `sim_noisy` neuf à
chaque essai et ignore la valeur de retour ; il repose donc sur la mutation,
c'est-à-dire sur l'inverse de ce que la docstring promet. Piège armé pour le
premier appelant qui croira la docstring : il verra son entrée corrompue.

**Une étape annoncée qui n'a pas lieu.** La bannière imprime *« Finding
optimal threshold for each method first... »* et le commentaire de l'étape 1
parle de *« GT and threshold optimization »*. Aucune optimisation de seuil
n'est faite : `best_qa_thr` / `best_cl_thr` sont lus tels quels dans
`TRAINED_PARAMS` / `CLASSICAL_PARAMS`, et `find_optimal_threshold` n'est
jamais appelée. Le journal fait donc croire à une recherche qui n'existe pas.

**Vérifié et trouvé sain** : les graines
`42 + trial + int(sigma * 1000)` ne collisionnent pas — les décalages de
`NOISE_LEVELS` (0, 50, 100, 200, 300, 500) sont tous séparés d'au moins 50
pour `trial` ≤ 4 (soupçon levé par le calcul, pas par l'inspection) ; le bruit
est bien mis à l'échelle du RMS de **chaque** champ, comme annoncé ; la vérité
terrain est prise sur le champ PROPRE et le raffinement sur le champ bruité,
ce qui est la bonne façon de mesurer une robustesse.

---

## `figures/v1_legacy/fig2_early_detection.py` — lu en entier

307 lignes. **Aucun défaut.** C'est le fichier le plus soigneux du dossier :
la détection précoce est mesurée aux temps courts **contre la vérité terrain
tardive**, ce que sa docstring annonce et ce qui est la bonne façon de poser
la question ; l'IoU consécutif est indexé juste (`si - 1` n'est écrit que
lorsque `prev` existe, `n-1` cases remplies, `mid_steps` de même longueur) ;
la référence relative `gt.mean()` y sert à comparer **deux bras sur le même
champ**, son rôle défendable.

Une seule réserve, non déclenchée : `qa_iou_all` / `cl_iou_all` sont
initialisés à zéro et ne seraient jamais écrits si `N_POINTS` retombait à un
seul pas après `np.unique` — les zéros entreraient alors dans la moyenne
comme des mesures. Avec les réglages du fichier, `n_steps_actual` vaut
toujours plus de 1.

## Reste après D-95 : la fenêtre verticale de `fig0`, mesurée et bornée

La correction D-95 retire la troncature des **données**. Il reste une
asymétrie de **cadre** : `y_max = max(percentile(q_phys, 95) * 1,3 ; 0,4)` est
fixé par le seul bras quantique et appliqué aux deux panneaux. Mesuré sur les
CSV gelés — part des essais classiques hors cadre, et visibilité du meilleur
d'entre eux :

| scénario | `y_max` | classiques hors cadre | `phys` du meilleur classique |
|---|---|---|---|
| `kelvin_helmholtz` | 0,400 | 0 / 172 (0 %) | 0,077 — **visible** |
| `harris_tearing` | 0,400 | 5 / 169 (3 %) | 0,004 — **visible** |
| `orszag_tang` | 0,400 | 35 / 292 (12 %) | 0,089 — **visible** |
| `mhd_rotor` | 0,932 | 9 / 292 (3 %) | 0,027 — **visible** |

Les points coupés sont tous du **mauvais** côté (erreur physique élevée), et
dans les quatre cas l'optimum classique reste dans le cadre : l'étoile que
D-95 rétablit est bien visible. Cosmétique, donc — écrit ici pour ne pas
être re-trouvé comme un défaut, et pour que la mesure existe si USER veut
malgré tout un cadre commun.

---

## `figures/v1_legacy/fig13_sigma_ablation.py` — lu en entier, deux pistes mesurées, aucune retenue

247 lignes. Deux soupçons plausibles, tous deux tués par la mesure — noté ici
pour qu'ils ne soient pas re-suspectés sans être re-mesurés.

**`active_frac` (panneau B) utilise le score BRUT, pas le score moyenné par
arête que `HamiltParams.compute_coefficients` applique réellement** (`score_avg_h
= 0,5·(score + roll(score,-1,axis=1))`, séparément pour h et v). Plausible à la
lecture — deux formes différentes de la même grandeur, exactement la question 4
de `VIGIL.md`. **Mesuré, sur données réelles** (`init_harris_tearing`,
`init_kelvin_helmholtz`, `init_orszag_tang`, `init_mhd_rotor`, N=256,
100-150 pas), aux 7 valeurs de σ du balayage : écart absolu maximal **0,0010**,
écart relatif maximal **2,7 %**, sur les 28 combinaisons scénario×σ testées.
Le moyennage par arête d'un champ physique lisse ne déplace quasiment aucune
cellule de part et d'autre du seuil `> 0.1` sur le poids gaussien — l'effet
existe mais ne sépare rien à l'échelle où la figure trace ses points.
**Non corrigé : mesuré et trouvé sans conséquence**, comme le splitting de
Strang de `VIGIL.md` (« une hypothèse plausible s'est révélée fausse à la
mesure »).

**`sigma_trained = TRAINED_PARAMS.get('sigma', 0.023)` n'est jamais réellement
« trained ».** Vérifié : `'sigma'` n'existe dans aucune entrée de
`results/hyperparams/best_hyperparams.json` (`grep -c '"sigma"'` → 0), donc
`sigma_trained` vaut **inconditionnellement** le repli codé en dur 0,023 —
jamais un chiffre échantillonné par la campagne Optuna gelée. La valeur
elle-même n'est pas fausse (elle sert de constante d'ablation cohérente dans
tout le fichier), mais la légende (`σ*=0,023`) et le log (`f"trained σ =
{sigma_trained}"`) la présentent comme si elle l'était. Repli silencieux
(forme connue de `VIGIL.md`), mais **choix de conception plutôt que défaut** :
`sigma` n'a jamais été un axe de la campagne d'entraînement gelée (voir
`results/hyperparams/PROVENANCE.md`), il n'y a donc rien à corriger côté
provenance — seulement une étiquette à ne pas prendre pour un résultat mesuré.
Non corrigé, signalé pour que la prochaine lecture ne le retrouve pas comme un
défaut de calcul.

**Vérifié et trouvé sain** : `zz_mean`/`z_mean` lisent `C_edges`/`H_edges`
directement (pas de reconstruction locale) ; `SIGMA_BASELINE = 100.0` désactive
bien le gating (`uncertainty → 1` partout à cette échelle) ; usage de
`AngleMapper.classical_score` (pas `physical_score`) correct, pas de confusion
D-9.

**Axes empruntés** : les 4 scénarios de `SCENARIOS`, les 7 valeurs de
`SIGMA_VALUES` plus la référence `SIGMA_BASELINE`.

---

## `figures/v1_legacy/fig10_grid_scaling.py` — lu en entier, **abandonné par décision**

218 lignes, dont 204 mortes **par déclaration** : le fichier s'arrête à un
`sys.exit(0)` ligne 14, et son en-tête dit pourquoi (« ABANDONED … N < 256
raffine tout à cause de `min_patch_size` ; N > 256 hors budget ; le panneau
de temps de décision n'a pas de sens sur un simulateur »). C'est un **choix de
conception documenté**, pas un défaut — la distinction que `VIGIL.md` demande
de faire avant d'accuser.

À signaler quand même, parce que c'est un piège si le fichier revit : lignes
99-100, le temps du bras classique n'est pas mesuré, il est **fabriqué** —
`all_cl_time[N].append(t_total * 0.3)`, avec pour seule justification le
commentaire *« classical is ~30% of total »*, tandis que `all_qa_time` reçoit
le total des **deux** bras. Le rapport de vitesse qui en sortirait vaudrait
3,33× par construction, quelle que soit la mesure — une valeur sans
provenance, au sens de la 8e forme de `VIGIL.md`.

**Ce qui sauve la figure** : ces deux dictionnaires sont écrits et **jamais
relus** (vérifié sur tout le dossier). Le panneau C trace `qa_times_separate`
et `cl_times_separate`, qui sont, eux, honnêtement chronométrés par deux
appels séparés à `run_single_method`. Le 0,3 est donc doublement mort — dans
un fichier désactivé, et dans une variable sans lecteur. Non corrigé pour
cette raison ; écrit ici pour que sa réanimation ne passe pas inaperçue.

---

## `figures/v1_legacy/fig9_synthetic_unit_tests.py`, `fig11_hamiltonian_design.py`, `fig3_spatial_coherence.py` — lus en entier

Aucun défaut neuf. `fig9` (D-97 corrigé, D-98 rapporté — déviation écrite
dans la docstring de `pixel_prf`, comme `VIGIL.md` l'exige), `fig11` (D-100
rapporté, écrit à côté du calcul concerné) et `fig3` (D-99 corrigé, les
deux métriques rendues périodiques) ne portent rien d'autre après lecture
complète des quatre questions — boucles principales, générateurs de champ,
annotations de figure compris. **Axes empruntés** : les 4 scénarios
canoniques (KH, Tearing, Rotor, OT pour fig11/fig3 ; les 4 motifs
synthétiques propres à fig9), bras Q-HAS et classique, N=256.

---

## `figures/v1_legacy/fig15_decision_flip_analysis.py` — lu en entier, D-102

Le bloc CONCLUSION citait en dur `σ=0.023 (trained)` — le `TRAINED_SIGMA`
de `study/pipeline/config.py`, un module que ce fichier n'importe pas.
Le repli réellement utilisé par son propre `HamiltMapper`
(`_hamilt_mapper_kwargs`, `fig_utils.py`) vaut 0,05, jamais 0,023 (`'sigma'`
absent de `results/hyperparams/best_hyperparams.json`, D-22). Détail
chiffré et correction dans `RESULTS.md` (D-102). Le reste du fichier — BFS
instrumenté (D-96, déjà corrigé), les quatre panneaux, le résumé texte —
ne porte rien d'autre. **Axes empruntés** : les 4 scénarios canoniques,
branche `flip_rate < 0.05 et mean_ratio < 0.5` du texte de conclusion
(celle qui portait le défaut) vérifiée par lecture directe du code, pas
par exécution (campagne VQA complète, hors budget d'une relecture).

---

## `figures/v1_legacy/fig16_decision_landscape.py` — lu en entier, sain

Sa propre copie de `instrumented_bfs` (D-96, déjà corrigée) est plus
pauvre que celle de `fig15`/`fig17` (pas de `gt_error_max`, `sub_area`, ni
décomposition ZZ) mais cohérente avec ce que `fig16` lui demande réellement
— aucun champ manquant n'est lu plus loin. La docstring du module
(« Reuses the instrumented_bfs() engine from fig15 ») est fausse au sens
strict (copie locale, pas un import) mais ne porte sur aucune valeur
calculée — hors périmètre de `VIGIL.md`, non corrigée. **Vérifié et trouvé
sain** : seuil, diagonale, comptages de quadrants et accord
`decision_qaoa`/`decision_classical` avec `should_refine` (moyenne du
`gt_error_mean` du journal) cohérents entre eux.

---

## `figures/v1_legacy/fig17_topological_attribution.py` — lu en entier, un point NON mesuré

Décomposition Hamiltonienne (ZZ/ZZZZ/X-point/Z) par cellule cohérente avec
`fig15` (même halo D-96 déjà corrigé, mêmes conventions de depad). Rien
trouvé de faux dans les quatre panneaux ni l'agrégation par quartile.

**Soupçon écrit mais NON mesuré, à vérifier au prochain passage** :
`CACHE_PATH = FIG_DIR/.fig17_cache.json`, et
`use_cache = os.path.exists(CACHE_PATH) and '--recompute' not in sys.argv`
— le cache ne porte aucune empreinte de configuration (pas de hash de
`TRAINED_PARAMS`/`threshold_amr`/`N`). Si le fichier a déjà tourné une fois
puis que les hyperparamètres déployés changent (campagne D-22), une
relecture réutiliserait un `decision_log` dont `decision_classical`/
`decision_qaoa` ont été calculés avec l'ANCIEN `threshold_amr`, alors que
l'en-tête du log imprime le seuil COURANT (`f"N={N}, threshold=
{threshold:.4f}"`) — deux grandeurs qui devraient coïncider (question 4)
et ne le feraient plus. **Non mesuré** : aucun `.fig17_cache.json` n'existe
dans ce dépôt (le script n'a jamais tourné jusqu'au bout ici), donc rien à
comparer avant/après pour l'instant. Ne pas le déclarer défaut avant
mesure — juste écrit pour que la prochaine passe qui fait tourner `fig17`
la première fois sache où regarder si elle le refait tourner une seconde
fois après un changement d'hyperparamètres.

## `figures/v1_legacy/` — passe du 16 août, dernier lot : fig5, fig6, fig7, fig8, fig12, fig14

Six fichiers, tous **lus en entier**. Ils ferment le module :
`fig_utils`, fig0 à fig4, fig9 à fig11, fig13 et fig15 à fig17 l'étaient
déjà. `fig5` portait **D-101** (corrigé par une passe antérieure) mais
n'avait jamais été déclaré lu en entier — il l'est ici, et il rendait
**D-107**.

**Axes traversés** (la fiche en liste sept) : profondeur AMR `depth = 0`
**et** `depth > 0` (`solve_max_depth = 5` mesuré à N=256, `min_size=6`) ;
bord de patch périodique **et** borné (les deux constructeurs, via
`run_adaptive_vqa`/`run_adaptive_classical`) ; bras quantique **et**
classique ; backend `state_vector` ; Hamiltonien non nul ; COBYLA ;
`AdvAnomaliesEnable = True` (`run_hierarchical_comparison`). **Non
traversés** : backend échantillonné, warm start, Hamiltonien nul,
optimiseurs autres que COBYLA — aucun de ces fichiers ne les emprunte.

| fichier | verdict |
|---|---|
| `fig5_qaoa_detailed_analysis.py` | **D-107** (corrigé) |
| `fig6_statistical_validation.py` | **sain sur ses valeurs**, un contrat inexact |
| `fig7_physical_fidelity.py` | **D-104** (corrigé) et **D-105** (corrigé) |
| `fig8_hierarchical_comparison.py` | **sain sur ses valeurs**, deux calculs morts |
| `fig12_depth_analysis.py` | **D-106** (corrigé) |
| `fig14_boundary_correction.py` | **abandonné par décision**, code mort |

### `fig5` — lu en entier, D-107, et deux notes

Au-delà de D-107 (le `dx` de la profondeur 0), deux choses relevées et
**non corrigées**, parce qu'aucune valeur n'en dépend :

- deux définitions du « meilleur patch de profondeur 1 » cohabitent dans le
  fichier — `analyze_hamiltonian_by_depth` le choisit par `gt[...].sum()`,
  le bloc de tracé par `np.max(gt[...])`. Elles alimentent deux panneaux
  différents, tous deux étiquetés « Depth 1 ». Pas un défaut de valeur,
  mais deux étiquettes identiques pour deux sélections distinctes ;
- la profondeur 0 est analysée **deux fois** (`analyze_hamiltonian_by_depth`
  puis `analysis_d0`), donc deux appels QAOA complets sur les mêmes
  entrées. Les énergies de coefficients, elles, sont déterministes à `dx`
  fixé : les deux chemins coïncident sur ce qui est tracé.

Vérifié sain : `_gt_quadrant_above_threshold` compare bien à `gt.mean()`
(la correction de D-101 est en place et sa raison est écrite à côté du
calcul) ; `_gt_error_share` normalise par la somme du domaine, pas par
quadrant.

### `fig6` — la question posée, et la réponse mesurée contre moi

Le module annonce *« Each seed creates a genuinely different simulation
(tiny initial perturbation) so the captured-fraction samples are
independent »*, et `perturb_sim` promet un décalage *« large enough to
produce different VQA decisions and different GT error maps »*. Deux
raisons de douter : la perturbation est appliquée **après** les `n_steps`,
donc ce n'est pas une perturbation *initiale* et la simulation n'est jamais
rejouée ; et son amplitude est de `1e-4 × rms`. Si les dix graines
rendaient la même valeur, l'IC bootstrap serait de largeur nulle et la
p-valeur de permutation vaudrait 2⁻¹⁰ ≈ 0,001 — « *** » — par construction.

**Mesuré** (`init_kelvin_helmholtz`, N=256, 400 pas, `TARGET_DIM=2`,
`MIN_SIZE=6`, `K_opt=40`, seuils du dépôt, 3 graines) : les trois graines
rendent **trois valeurs distinctes**. Q-HAS 0,9921 / 0,9882 / 0,9979
(étendue 9,7e-03), classique 0,9943 / 0,9949 / 0,9912 (étendue 3,7e-03) ;
déviation relative max de la carte GT 7,7e-03. **Le contrat tient** : les
échantillons ne sont pas dégénérés, la p-valeur n'est pas fabriquée.

Reste **inexact et non corrigé** (ni valeur ni décision n'en dépendent) :
la perturbation n'est pas *initiale*, et `make_sim` est rejoué à l'identique
à chaque graine (déterministe, aucun tirage) — dix simulations complètes
pour un état qui ne change pas. Écrit ici, pas dans le code.

Vérifié aussi : `permutation_pvalue` est bien le test de signe unilatéral
que sa docstring annonce ; `cohens_d` est le `d_z` apparié (`ddof=1`) avec
son garde `d_std > 1e-12` ; les seuils des deux bras sont **distincts**
(`TRAINED_PARAMS` pour Q-HAS, `CLASSICAL_PARAMS` pour le classique) — pas
le défaut de D-101.

### `fig8` — sain sur ses valeurs, deux calculs morts et deux commentaires faux

Aucun défaut de valeur. Le panneau D construit ses trois simulations
identiques **sans** perturbation d'essai : il n'a pas D-104.

Deux choses écrites ici plutôt que corrigées, parce qu'aucun nombre n'en
dépend :

- `# Find optimal thresholds via fine-grained search` est suivi de
  `sim0, Phi0 = make_sim(...)` et `gt0 = ground_truth_errors(...)`, **jamais
  utilisés** — une simulation MHD complète (80 à 120 pas à N=256) jetée par
  scénario — puis de deux constantes. Aucune recherche n'a lieu, alors que
  `find_optimal_threshold` existe dans `fig_utils`. Même forme au panneau D
  (`gt_fid`, également inutilisé). C'est le patron « repli silencieux » de
  `VIGIL.md` : le commentaire décrit un calibrage, le code lit une constante.
- Les barres d'erreur du bras **classique** (panneaux A) sont nulles par
  construction : `make_sim` est déterministe et `run_adaptive_classical`
  aussi, donc les `N_TRIALS = 2` essais sont identiques. Ce zéro-là est
  **vrai** (le bras n'a pas de dispersion), contrairement à celui de D-105.

### `fig14` — abandonné, et son code est inatteignable

`sys.exit(0)` à la ligne 18, avant tout import : les ~600 lignes qui
suivent ne s'exécutent jamais. Même statut que `fig10`. Une seule note :
la docstring d'abandon invoque *« σ=0.023 suppression »* — la valeur d'un
autre module, exactement l'erreur que **D-102** a corrigée dans `fig15`,
alors que le `SIGMA` du fichier vaut `TRAINED_PARAMS.get('sigma', 0.05)`.
Non corrigé : c'est du texte dans du code mort.

### Une piste mesurée et écartée — pour ne pas la re-chercher

`_patches_overlap_with_gt` (`fig_utils.py`) **infère** `target_dim` depuis
les patchs alors que son appelant `patches_to_metrics` le reçoit en
argument et le passe à l'autre moitié du calcul
(`_patches_total_fine_pixels`). Soupçon : le `break` sortirait de la boucle
au premier patch quel qu'il soit, laissant `target_dim = 2` par défaut.
**Mesuré** sur un arbre `target_dim = 4` construit dans les deux ordres
(patch `depth=0` en tête, puis en queue) : `captured_fraction` **identique**
(0,00781250, contre 0,01953125 si `target_dim = 2` était utilisé). Le
`break` est bien à l'intérieur du `if p['depth'] == 0`. **Pas un défaut.**

Reste une **asymétrie latente, non mesurée en production** : un patch de
type `fallback` (« l'AMR n'a rien trouvé, on repasse en complet ») est
ignoré par `_patches_overlap_with_gt` (poids 0) mais compté par
`_patches_total_fine_pixels`. Mesure structurelle sur un unique patch
`fallback` couvrant tout le domaine : `captured_fraction = 0,0` et
`compute_ratio = 0,25`. Les deux moitiés du même dictionnaire ne décrivent
pas le même patch. Aucun `fallback` n'est apparu dans les exécutions
réelles de cette passe (tearing 300 pas, OT 500 pas, KH 400 pas : types
observés `leaf_depth` et `coarse_leaf` uniquement), donc **latent, pas un
défaut** — à rouvrir si un jour un bras ne trouve rien.

Second point latent, même famille. `compute_local_factor` se déclare
« Shared between solver (physics) and pipeline (cost metric) to guarantee
consistency ». Le solveur reçoit le vrai `solve_max_depth`
(`step_layered(patches, max_depth=solve_md, ...)`) ; les deux métriques de
`fig_utils` lui passent `max(p['depth'] for p in patches)`, le maximum
**présent dans la liste**. Les deux ne coïncident que si un patch atteint
la profondeur de résolution. C'est le cas dès qu'il existe un patch
`leaf_depth`, puisque `refinement.py` les enregistre avec `_solve_depth`.
**Observé** sur les trois exécutions réelles de cette passe (tearing 300
pas, OT 500 pas, KH 400 pas, `min_size=6`, N=256) : `max(depth) = 5 =
solve_max_depth` dans les six listes de patchs. Non mesuré : le cas où
aucun `leaf_depth` n'existe, où le facteur local — donc `compute_ratio` et
`captured_fraction` — serait calculé sur une hiérarchie plus courte que
celle que le solveur applique. **Latent, pas un défaut.**

---

## `figures/v1_legacy/fig_utils.py` — la moitié « simulation », lue et croisée

Dernier point resté ouvert de la section précédente : `qaoa_block_scores`,
`run_hierarchical_comparison`, `run_single_method`, `find_optimal_threshold`,
`patches_to_metrics` avaient été **lues, pas croisées**. Relues fonction par
fonction, question 3 (consomment-elles ce que leur signature annonce ?) et
question 4 (deux chemins censés coïncider) :

| ce qui a été vérifié | verdict |
|---|---|
| `_hamilt_mapper_kwargs` (partagée par les cinq) tire `sigma`/`beta_curl`/`beta_xpoint`/`gamma_hydro`/`gamma_mag`/`kappa`/`w_z_frac` de `TRAINED_PARAMS`, comme le chemin déployé | **sain** — pas de repli silencieux propre à ces fonctions ; le repli `sigma=0,05` est le même que celui déjà documenté (D-22) |
| `patches_to_metrics` appelle `_patches_overlap_with_gt` sans lui passer `target_dim` (reçu en argument), et le passe seulement à `_patches_total_fine_pixels` | déjà **mesuré et écarté** dans la section précédente (« piste mesurée et écartée ») — pas re-testé ici ; confirmé qu'aucune des cinq fonctions n'appelle `patches_to_metrics` avec un `target_dim ≠ 2` |
| `find_optimal_threshold` — sélection du « genou » (candidats dans `capture_margin` du meilleur, puis min `compute_ratio`) | **sain** — la liste de candidats ne peut pas être vide (repli explicite sur `d['captured'] == best_cap`) ; le balayage grossier puis fin est dédoublonné par `already`, aucun point perdu ni compté deux fois |
| `run_hierarchical_comparison` / `run_single_method` calculent `nu = grid.L / 800` puis ne le relisent jamais (`HamiltMapper` reçoit ses kwargs de `_hamilt_mapper_kwargs`, pas de cette variable locale) | **gaspillage, pas une valeur fausse** — la variable n'est lue nulle part, donc rien d'aval n'en dépend |

**Ce que ça ne couvre pas.** Aucun test n'emprunte encore ces cinq
fonctions — elles ne s'exercent qu'en faisant tourner une figure pour de
vrai (DNS + VQA complets). La lecture croisée ci-dessus n'est pas un test
qui traverserait profondeur AMR / bord / bras / backend / warm start /
hamiltonien nul / optimiseur sur ces chemins ; ça reste à faire si l'un de
ces scripts redevient prioritaire.

`figures/v1_legacy/fig_utils.py` est maintenant lu **et** croisé en entier.
Seul reste ouvert le point non mesuré de `fig17` (cache sans empreinte de
configuration, ci-dessus) — qui demande de faire tourner le script pour de
vrai, pas une relecture.

---

## `scripts/` — module lu en entier (passe du 16 août)

Terrain neuf : `scripts/` n'avait jamais été déclaré lu. 7 fichiers,
1 443 lignes. **Quatre défauts**, tous poussés — D-108, D-109 (extraction
des hyperparamètres), D-110, D-111 (racines et chemins des lanceurs).

| ce qui a été vérifié | verdict |
|---|---|
| `extract_best_hyperparams.py::_detect_param_cols` — trois générations de noms (`beta_michelson` → `beta_grad` → `sigma`) | **D-108** — la branche détectait `param_beta_grad` puis renvoyait un jeu qui ne le contient pas ; 579 valeurs échantillonnées jetées. Corrigé, verrouillé |
| `extract_best_hyperparams.py::_pick_best_for_scenario` / `_pick_best_for_group`, et ce que `main()` leur donne | **D-109** — l'optimum « par scénario » était choisi parmi les 3 meilleurs du score agrégé, sur 178 essais ; 6 entrées sur 8 changent. Corrigé, verrouillé |
| `run_tests.sh`, ses 32 appels `run_stage` | **D-110** — 17 commandes d'étage sur 17 mortes depuis la réorganisation `17d983d` ; 0 test atteignable → 168. Corrigé, verrouillé |
| `generate_figures_v1.sh`, `run_study_v3.sh` — racine du dépôt calculée | **D-111** — décalées d'un niveau, dans les deux sens. La première corrigée ; la seconde **volontairement non corrigée** (son en-tête gèle ses chemins jusqu'à ce que D-49 soit tranché), déviation écrite dans le fichier et vérifiée par un test |
| `run_fold.sh`, `run_leak_free_campaign.sh` — mêmes racines | **sains** — déjà corrigés par D-71, et leur commentaire porte la mesure |
| `run_study_v2_phases.sh`, `run_study_v2b.sh` — chemins et drapeaux | **sains** — déjà vérifiés fichier par fichier par D-76 ; les 32 invocations enrobées (`run_phase`) résolvent toutes, mesuré |
| `extract_best_hyperparams.py` — sélection `default` / `best_per_phase` | **sains** — minimum du score agrégé, identique que la sélection porte sur le top-K ou sur tous les essais (mesuré avant/après D-109) |

**Trois observations mesurées, non corrigées** (elles ne rendent aucune
valeur fausse ; à trancher) :

1. **L'ordre des clés du JSON d'hyperparamètres n'est pas reproductible** —
   `all_scenarios = list(set(...))` : trois exécutions à `PYTHONHASHSEED`
   différents, mêmes entrées, **trois sha256 différents**. Les valeurs sont
   identiques ; seul un diff d'artefacts en souffre.
2. **Un scénario ou un combo sans résultat s'écrit `null` sans un mot** —
   sur la campagne vive, `scenario_combos.simple` est `null` aux deux bras
   et 2 entrées de `per_scenario` sur 6 le sont aussi, sans que le script le
   dise. Atténuation : `load_hyperparams(combo=…)` **lève** sur une entrée
   nulle.
3. **`run_study_v2b.sh` accepte une phase `9` qu'il ne sait pas exécuter** —
   le parseur admet `[1-9]`, la carte des phases va de 1 à 8 puis 10, et le
   `case` d'exécution n'a pas de branche `*)`. `bash scripts/run_study_v2b.sh 9`
   affiche sa bannière, ne lance rien et sort **0**. Un étage vide doit crier.

**Ce que ça ne couvre pas.** Les lanceurs ne sont pas *exécutés* par la
suite — le nouveau `tests/test_launcher_paths_resolve.py` vérifie que chaque
fichier invoqué existe (79 invocations, 7 lanceurs), pas que la campagne
tourne. Les axes de la fiche (profondeur AMR, bord, bras, backend, warm
start, hamiltonien nul, optimiseur) ne s'appliquent pas à `scripts/` : aucun
de ces fichiers ne calcule de physique. `extract_best_hyperparams.py`, lui,
a été traversé sur ses **trois** générations de campagne (`beta_michelson`,
`beta_grad`, `sigma`) et sur les **deux** bras (quantique, classique) — c'est
l'axe réel de ce module.

---

## `src/VQA/` — module lu en entier (passe du 16 août), un défaut

Les sept fichiers du dossier, **1 026 lignes** (`wc -l src/VQA/*.py`) :
`cost_hamiltonian.py` (414), `execute.py` (256), `runtime.py` (189),
`postprocess.py` (61), `mapping.py` (50), `optimize.py` (35),
`init_qbits_state.py` (22). Le
dossier figurait à **100 % de couverture de ligne** dans le tableau plus
haut — c'est exactement le cas que la mission de `VIGIL.md` décrit : la
couverture ne dit rien du contrat.

| ce qui a été vérifié | verdict |
|---|---|
| `create_bounded_hamiltonian` — contraction de plaquette au bord | **D-113** — le qubit manquant était remplacé par le `<Z>` de l'AUTRE famille de liens (Droite = lien V lisant `theta_h`, Bas = lien H lisant `theta_v`). Corrigé, verrouillé |
| `create_bounded_hamiltonian` — contraction de CISAILLEMENT aux quatre bords | **saine** — chaque bord lit sa propre famille (`theta_h` colonnes 0/-1 pour les liens H, `theta_v` lignes 0/-1 pour les liens V) et le bon coefficient de couplage (`C_edges[0][ci,0]` à gauche, correction antérieure). C'est la comparaison avec elle qui a levé D-113 |
| `create_bounded_hamiltonian` — centrage `z_threshold = 1 − 2·threshold_amr` | **sain** — `theta = 2·arcsin(√score)` donne `cos θ = 1 − 2·score`, donc le centrage est bien sur la frontière de décision, et le facteur `w_z_frac` n'est appliqué qu'aux contractions 1-corps, comme son commentaire l'annonce |
| `create_bounded_hamiltonian` — garde de forme A0 | **saine** — refuse tout tableau qui n'est pas `(dim+2, dim+2)`, dans les deux sens (trop grand comme trop petit) |
| `create_period_hamiltonian` | **sain sur ses valeurs**, déviation `dim = 2` documentée (D-59) et épinglée par un test. **N'a PAS l'équivalent de la garde A0** : un `hamilt_params` trop grand y serait lu par son coin haut-gauche sans erreur. Aucun appelant du dépôt ne le fait — `build_patch_hamiltonian` rend bien `(dim, dim)` — donc **observation, pas défaut** |
| `init_qbits_state` — quel `theta` va sur quel qubit | **sain**, et c'est le contrat qui tranche D-113 : qubits `0 … dim²−1` ← `theta_h`, qubits `dim² … 2dim²−1` ← `theta_v` |
| `execute` — bornes du mixeur par méthode | **sain** — `bounds` pour L-BFGS-B et Powell, `constraints` pour COBYLA, et un `raise` explicite pour toute autre méthode plutôt qu'une borne perdue en silence (correction antérieure, D-38) |
| `execute` — ordre des paramètres `[β…, γ…]` | **sain** — les bornes et les contraintes indexent `x[0:reps]` comme β, ce qui est l'ordre que `QAOAAnsatz` expose (β avant γ) |
| `execute` — restauration de `default_shots` après une lecture MPS | **saine** — le `sampler` peut être partagé par toute la campagne, la valeur d'origine est remise |
| `optimize` — les trois backends | **sain** — `state_vector` force `opt_level = 0`, et un backend inconnu lève |
| `postprocess` — convention de bits et contrat d'entrée | **sain** — parcourt `bitstring[::-1]` (qubit 0 à droite, la convention de `probabilities_dict()` et de `get_counts()`), et **refuse** trois entrées qui rendraient des marginales plausibles et fausses : distribution non normalisée, chaîne multi-registres (l'espace décalerait toutes les positions suivantes), longueur ≠ `num_qubits` |
| `runtime.VQARuntime` — mode, backend, cache d'ansatz | **sain** — `mode` hors `('simulator',)` lève (D-48 : `_init_backend` rend un simulateur quel que soit le mode, un run « hardware » tournait donc sur simulateur sans le dire), un `backend_name` inconnu lève au constructeur plutôt qu'un `AttributeError` cinquante lignes plus loin, et la clé du cache d'ansatz inclut une **empreinte des coefficients**, pas seulement la topologie |

**Axes traversés — mesurés sur la suite ENTIÈRE, pas supposés.** Un plugin de
trace enrobe les points de décision et compte les appels réels :

```bash
TRACE_OUT=/tmp/axes.json PYTHONPATH=src:tests/tools \
  python -m pytest tests/ -q -m "not slow" -p trace_fiche_axes    # ~1 h
```

Les sept axes de la fiche, et ce que la suite emprunte vraiment :

| axe | côté A | côté B | verdict |
|---|---|---|---|
| **bord du patch** | borné **113** | périodique **390** | **les deux** |
| **bras** | quantique **28** | classique **11** | **les deux** |
| **Hamiltonien** | non nul **318** | nul **2** | **les deux** |
| **warm start** | absent **315** | présent **5** | **les deux** |
| **profondeur AMR** | `depth = 0` | `depth > 0` | **les deux** — le bras borné n'est atteint que par `depth > 0` |
| **backend** | `state_vector` **320** | échantillonné **0** → **traversé depuis D-118** | `aer` **sain**, `estimator` **mort** — voir plus bas |
| **optimiseur** | COBYLA **317** | Powell **1**, L-BFGS-B **1**, Nelder-Mead **1** → **traversé depuis D-119** | budget **non comparable**, et l'entraînement n'emploie pas le défaut du CLI — voir plus bas |

Et l'axe qui n'est pas dans la fiche mais que le code emprunte, `dim` :
**2 → 367**, **3 → 133**, **4 → 3**, **8 → 0**. `VQA_DIMS = [2, 4, 8]` : la
plus grande taille déclarée n'est traversée par aucun test de la suite rapide.

**Deux axes restaient donc à traverser**, et ce sont des faits mesurés, pas
des impressions : le **backend échantillonné** (0 appel) et les
**optimiseurs autres que COBYLA** (1 appel chacun — assez pour qu'une erreur
de bornes y survive, c'est exactement ce qui s'était produit avec Powell).

**Le backend échantillonné est traversé depuis D-118** —
`tests/quantum/test_estimator_backend_axis.py`, 6 cas. Ce qu'il a rendu :

| côté de l'axe | verdict |
|---|---|
| `aer` (échantillonné, Aer idéal) | **sain** — les deux chemins finaux de `execute` coïncident à la racine de N près : écart max sur 8 marginales **0,0205 / 0,0102 / 0,0037** à 1 024 / 8 192 / 65 536 tirs, pour un bruit attendu de 0,0312 / 0,0110 / 0,0039. Mesuré à paramètres FIXÉS, sans quoi on mesure l'optimiseur |
| `estimator` (FakeFez) | **mort — D-118**, à 100 % et à toute taille : 156 qubits physiques pour 2 comme pour 4 logiques, 156 bits classiques, `max_memory_mb` dépassé. Et le placement n'est pas l'identité (`[136, 142, 141, 143]`), donc rendre la mémoire seule produirait des marginales indexées par qubit **physique** — plausibles et fausses. Dette portée par un `xfail(strict=True)` |

**Ce que la traversée a appris sur la méthode.** L'axe ne se traverse pas en
appelant les deux backends de bout en bout : le résultat diverge alors pour
une raison qui n'est pas le backend. `EstimatorV2` tire `default_shots`
**même** sous `AerSimulator(method='statevector')`, donc la boucle
d'optimisation est stochastique des deux côtés — deux exécutions du **même**
backend `state_vector` donnent des marginales écartées de **0,143** et des
paramètres de **0,989**. Comparer deux backends à paramètres libres, c'est
mesurer cette dispersion, pas l'axe. Le seul geste qui sépare est de figer
les paramètres.

**L'axe optimiseur est traversé depuis D-119** —
`tests/quantum/test_optimiser_axis.py`, 6 cas. Deux faits, et une
non-conclusion :

| ce qui est mesuré | verdict |
|---|---|
| budget acheté par `K_opt` | **non comparable** — `options={'maxiter': K_opt}` vaut des **évaluations** pour COBYLA (20/20 à `K_opt = 20`, 6 fois sur 6) et des **itérations** pour les deux autres (L-BFGS-B 50–115, Powell 176–377). Intervalles disjoints, jusqu'à **×18,9** |
| optimiseur de l'entraînement contre celui du déploiement | **ils diffèrent** — `create_argus` code `COBYLA` en dur, `--method` de `pipeline.py` a pour défaut `L-BFGS-B`, et **0 lanceur sur 8** ne le surcharge. Depuis `cf93ba3` |
| ce que cela déplace sur la **décision** | **non décidable à cette dispersion** — écart des moyennes **0,0867** contre un bruit intra-méthode de **0,200** / **0,240**. Dit, et non conclu |

**Ce que la traversée a appris sur la méthode.** Un axe « traversé » par un
appel ne l'est pas : le seul appel Powell et le seul appel L-BFGS-B de la
suite passaient tous deux, et aucun des deux ne comparait le budget consommé
à celui de COBYLA. Compter les appels dit qu'un chemin est **exécuté** ;
il faut encore demander ce que le chemin **promet** pour voir qu'il promet
autre chose que son voisin.

**Ce qui reste, et ce qui n'est PAS une file.** Le comptage donne `dim`
2 → 367, 3 → 133, 4 → 3, 8 → **0**. Ce zéro n'ouvre rien : il est **déjà
expliqué et consigné** plus haut dans ce fichier (§ `exact_diag`) et dans
`DEFAUTS.md` — `dim = 4` et `dim = 8` demandent 32 et 128 qubits contre le
plafond de **20** codé dans `exact_diagonalisation.py`, qui les saute
explicitement. Ce n'est pas un axe non traversé, c'est une limite déclarée.
Le noter ici évite d'y renvoyer une passe qui re-trouverait un fait connu —
le piège que `VIGIL.md` décrit sous « lire le registre avant la passe ».

**Les sept axes de la fiche sont donc traversés.** La file de la prochaine
passe ne vient plus des axes : elle vient de `DEFAUTS.md`.

**Ce que cette traversée ne pouvait pas voir, et pourquoi.** L'axe « bord
borné » était traversé, et D-113 y est resté invisible : en déploiement
`theta_h ≡ theta_v` (`refinement._prepare_vqa_input` passe `mini_score`
deux fois), donc échanger les deux familles est l'identité. **Un axe
traversé ne suffit pas quand les deux valeurs de l'axe coïncident dans la
configuration testée** — c'est la règle « choisir le champ d'essai qui
SÉPARE », appliquée aux axes eux-mêmes. À retenir pour la fiche.

**Et pire que « invisible » : le défaut avait colonisé ses propres témoins.**
Trois endroits de la suite l'écrivaient noir sur blanc — le docstring de
`test_only_the_documented_halo_cells_are_read` (« `theta_h_full` n'est lu
qu'en colonnes 0 et -1 »), le **champ d'essai** de
`test_a_plaquette_on_the_boundary_contracts_instead_of_wrapping` (il
chauffait exactement les deux tableaux de la convention fausse, sous un
commentaire disant « halo droit » au-dessus d'une écriture dans `theta_h`),
et le code lui-même. Un test écrit à partir du code partage son modèle
mental, donc son erreur : c'est le corollaire de `VIGIL.md` sur la
couverture, vérifié ici une fois de plus.

---

## Les tests qui lisent le SOURCE — sondage, pas balayage complet

D-114 a montré un garde-fou de **comportement** écrit comme une recherche de
chaîne. `VIGIL.md` interdit la forme, et elle s'est déjà produite trois fois
ici. La question suivante s'impose : combien d'autres ?

**Mesuré** : `grep -rn "\.read()" tests/ --include=*.py`, hors `tests/tools/`
et hors `ast.parse` — **64 sites, 41 fichiers**.

**Sondés à la main cette passe — 3 sites, et un seul est un défaut :**

| site | verdict |
|---|---|
| `tests/mapping/test_mapper_contracts.py` — invariant `theta_h ≡ theta_v` | **D-114**, corrigé : gardait un COMPORTEMENT par une chaîne |
| `tests/pipeline/test_amr_figure_axes.py` — `assert "D-68" in src` | **légitime** — garde une **déviation documentée**, ce que `VIGIL.md` exige explicitement (« un test vérifie que la mention y reste ») ; son propre docstring dit qu'il est le seul du fichier à lire le source et pourquoi |
| `tests/quantum/test_qaoa_arm_is_sampled.py` — aucun `seed_*` dans `src/VQA/` | **légitime** — cherche l'ABSENCE d'un jeton dans un dossier entier ; aucune formulation de code ne le contourne, et un faux positif y coûte un examen, pas une conclusion fausse |

**La distinction qui tranche, et qu'il faut appliquer aux 61 sites restants** :
lire le source est **juste** quand l'objet du test EST le texte (une mention de
déviation qui doit rester, un jeton qui ne doit pas apparaître) ; c'est **faux**
quand l'objet est un comportement que le texte ne fait qu'indiquer. D-114 était
du second type ; les deux autres sondés sont du premier.

**Passe suivante (16 août) — un premier tri automatisé, puis une vérification
individuelle qui le contredit largement.** Un agent a relu les 61 sites
restants contre le critère ci-dessus et en a classé 27 « suspects ». Trois de
ces 27 ont été vérifiés à la main cette passe, par mutation (désactiver le
comportement réel, garder le texte source intact, relancer le test) :

| site | verdict du tri automatisé | vérifié par mutation |
|---|---|---|
| `test_t19_divergence_audit.py:78` (`DIVERGENCE_PENALTY`) | suspect | **faux positif** — le tri a lu au présent une docstring qui parle au passé (« était redéfinie quatre fois ») — le défaut est déjà corrigé et déjà verrouillé ailleurs par `test_the_divergence_penalty_has_a_single_definition`, qui compte les définitions par regex) |
| `test_solver_guards_and_objective.py:138` (avertissement sigma D-22) | suspect | **surestimé** — le comportement réel (`sigma_source == "default"`) est déjà vérifié fonctionnellement par `test_train_hyperparams_smoke.py:203` ; seule la spécificité « lève bien un `RuntimeWarning` » (par opposition à seulement l'enregistrer) reste non exécutée — un écart réel mais mineur, pas la régression silencieuse annoncée |
| `test_t28_t29_labels_and_ci.py:118` (seuil dégénéré) | suspect | **confirmé** — `relabel()` n'était jamais appelé ; un garde désactivé (`if False:`) laissait le texte du message en code mort et le test restait vert. Corrigé, mesuré, verrouillé → **D-115** ci-dessus |

**Passe du 16 août (soir) — un 4ᵉ site vérifié, et ce qu'il a fait tomber
à côté.** `test_vqa_chain_contracts.py:349`
(`test_beta_is_bounded_and_the_bound_is_the_documented_one`, qui fait
`assert "beta_max = np.pi / (4 * reps)" in src`) — il ne figurait pas dans
les 27 du tri, il vient d'une lecture directe. Vérifié par mutation, dans
les deux sens :

| mutation | ce que fait le test qui lit le source |
|---|---|
| **A** — borne réelle cassée (`bounds_beta` et contraintes à ±10), texte source intact | reste **VERT** — il ne voit pas ce qu'il prétend garder |
| **B** — réécriture **équivalente** `np.pi / 4 / reps`, valeur bit à bit identique | passe **VERT → ROUGE** — faux rouge sur un changement voulu, le 3ᵉ de cette forme ici |

**Verdict : « surestimé », pas défaut** — le comportement est couvert
ailleurs, par `test_the_three_supported_optimizers_keep_the_bound`
(`test_runtime_contracts.py`), qui est bien comportemental.

**Mais la mutation A a montré autre chose, et c'est D-120** : ce garde
comportemental ne rougissait que pour **Powell et L-BFGS-B**. Sur **COBYLA**
— 317 des 320 appels de la suite, et l'optimiseur de la campagne
d'entraînement — il passait **avec les contraintes entièrement retirées**.
Son champ d'essai part à froid, où `rhobeg = 0.05` borne `beta` tout seul ;
l'entrée qui SÉPARE est un warm start hors borne. Voir `RESULTS.md`.

**La leçon, qui vaut pour les 58 sites restants** : la mutation qui infirme
un site en révèle souvent un autre. Vérifier un candidat « surestimé » n'est
pas du temps perdu — c'est la question « et qui couvre le comportement,
alors ? », et c'est elle qui a rendu D-120.

**Passe du 16 août (nuit) — un 5ᵉ site vérifié, et c'est un défaut.**
`test_hyperparams_provenance_break.py:171-172`
(`test_the_pipeline_falls_back_to_a_hard_coded_sigma`, qui fait
`assert "_defaults.get('sigma', 0.05)" in src`) — l'un des 27 du tri
automatisé. Vérifié par mutation A (`pipeline.py:394`, `0.05` → `0.07`,
reste du fichier intact) : suite des deux fichiers concernés rejouée,
**1 failed, 21 passed, 1 xfailed** — le seul test qui rougit est celui qui
lit le source. Contrairement à D-120, **rien d'autre ne couvre ce
comportement** :
`test_the_pipeline_shouts_when_sigma_is_missing` (`test_train_hyperparams_smoke.py`)
vérifie `sigma_source == "default"` et l'avertissement, jamais la valeur
numérique de `result["sigma"]`. **Verdict : confirmé, pas surestimé** —
D-121, voir `RESULTS.md`. Corrigé en ajoutant un test comportemental qui
appelle `pipeline()` et lit `result["sigma"]` directement ; le test
source-text reste en place (il n'est pas faux, seulement fragile face à
une réécriture équivalente — un 4ᵉ cas de cette forme dans ce dépôt,
non vérifié celui-ci faute de mutation B).

**Calibrage à retenir avant la prochaine passe** : sur cet échantillon de 3,
le tri automatisé s'est trompé ou a exagéré 2 fois sur 3. Ce n'est pas le
taux du dépôt (`CODE_REVIEW.md` : la majorité du code est juste, un défaut
sur trois sondés à la main) — c'est le taux d'un **premier passage
automatisé non vérifié**, et il ne suffit pas à promouvoir une entrée en
défaut. Chaque site restant doit repasser par la vérification par mutation
avant d'entrer dans `DEFAUTS.md`, pas seulement par la lecture qui l'a
signalé.

**Passe du 17 août (nuit) — 4 sites de plus, et 3 sont des défauts.** Tous
dans `test_t24_leak_free.py`, le plus gros bloc de la liste (9 sites). La
mutation employée est la seule qui tranche pour cette forme : **casser le
comportement en laissant la chaîne cherchée en place**.

| site | mutation | ancienne suite | verdict |
|---|---|---|---|
| `:181` `test_resume_reuses_only_matching_configurations` | un `and` de la décision de reprise devient `or` | **26 passed** | **confirmé → D-123** |
| `:200` `test_resume_is_recorded_never_silent` | les deux écritures `out[...]` supprimées ; les deux noms survivent **dans le commentaire** qui les explique | **26 passed** | **confirmé → D-124** |
| `:150,154` `test_partial_checkpoints_are_never_analysed` | `h4_transfer_summary.py` seul cesse de filtrer `status == "partial"` | **38 passed** (3 fichiers) | **confirmé → D-125** — le jumeau `closed_loop` est, lui, couvert fonctionnellement |
| `:68` `test_no_claim_of_a_shared_operating_point` | — | — | **légitime** — cherche l'**absence** d'un jeton, le type que le critère ci-dessus autorise |

**Ce que ce bloc apprend, et qui change le tri.** Les trois confirmés ne
gardent pas un calcul : ils gardent une **provenance** — ne pas mélanger des
tirages d'une autre configuration, ne pas taire qu'un tirage vient d'un autre
processus, ne pas publier une exécution interrompue comme complète. Un faux
vert n'y produit pas un plantage mais une moyenne d'apparence normale sur des
données qui ne vont pas ensemble. **Les sites qui gardent une provenance sont
donc à sonder avant ceux qui gardent un calcul** : un calcul faux finit par se
voir, une provenance perdue, non.

Deuxième leçon, sur la mutation elle-même : `:200` n'aurait **jamais** été vu
par une mutation qui supprime le code *et* son texte. Le faux vert y tient à
ce que le nom cherché existe **deux fois** dans le fichier, une fois en code
et une fois en commentaire. **Vérifier si la chaîne cherchée apparaît plus
d'une fois dans le fichier visé** est un tri à un coup de `grep -c`, et il
aurait désigné ce site sans aucune exécution.

**Deux défauts de la même nuit ne viennent PAS du sondage `.read()` mais de
ce qu'il a rendu visible en chemin** — et tous deux sont des fenêtres de
proximité :

* **D-126** — un garde prenait la **première** occurrence d'une phrase non
  unique et lisait les 600 caractères suivants ; un commentaire ajouté plus
  haut a fait tomber la fenêtre à côté, et le test a rougi sans qu'aucun
  défaut n'existe.
* **D-128** — un garde cherchait la chaîne `completed` dans les **12 lignes**
  précédant une agrégation ; ce voisinage contient `"n_completed": len(runs)`,
  un champ de compte rendu. Le vrai filtre retiré, le garde restait vert.

**La forme à chercher, maintenant nommée : la fenêtre de proximité.** Un garde
qui délimite sa zone d'examen par un décalage de lignes ou de caractères
autour d'une ancre textuelle. Elle échoue **des deux côtés** — faux rouge
quand l'ancre bouge, faux vert quand la fenêtre attrape autre chose. Le
remède est le même dans les deux cas : l'AST délimite par la **structure**
(la liaison d'un nom, le corps d'une fonction), jamais par une distance.

Le relevé de cette forme, fait la même nuit sur tout `tests/` : **14 sites**
(`src.index`, `splitlines()[…]`, tranches de `src`). Deux vérifiés,
**tous deux sains** — et ce qui les sauve mérite d'être noté, parce que c'est
la parade à écrire ailleurs :

| site | pourquoi il tient |
|---|---|
| `test_no_private_curl_survives.py:161` — fenêtre `src[index("    dxvy ="):index("    omega_z =")]` | la fenêtre **peut** devenir vide, mais 8 assertions **positives** (`re.search` d'un motif par nom) s'y appliquent : une fenêtre vide les fait toutes rougir. Un balayage vide y crie |
| `test_padded_rescale_contracts.py:437` — fenêtre entre deux `def` | mêmes deux ingrédients : une assertion positive dans la fenêtre, **et** un garde comportemental exact ailleurs (`test_a_peak_in_the_halo_of_the_flux_is_kept_too`, `max == 9.0`) — un lissage ajouté par un helper hors fenêtre le ferait rougir |

**Ce qui distingue les sains des défauts** n'est donc pas la fenêtre, c'est
ce qu'on y assied : une assertion **positive** dans la fenêtre transforme
« fenêtre décalée » en échec bruyant, là où une assertion **négative** seule
(`X not in bloc`) passe au vert sur une fenêtre vide. D-126 et D-128 n'avaient
que des assertions satisfaites par l'absence ou par le voisinage.

12 sites de cette forme restent à sonder.

**Un balayage latent, mesuré et NON promu en défaut.**
`test_hyperparams_two_sources.py::_live_pipeline_keys` retrouve par regex les
clés que `pipeline.py` lit (`hp.get('…')`), et `missing = clés − fournies`
doit être vide. Si la regex ne trouvait plus rien, `missing` serait vide lui
aussi : le test passerait sans rien couvrir — la forme de D-128, mais
**latente**. Mesuré aujourd'hui : la regex voit **10** clés, l'AST en voit
**10**, aucun accès `hp["…"]` ni `hp.get(variable)` n'existe dans
`pipeline.py`. **Le balayage est donc fidèle à cette date, et aucune mutation
réaliste ne le vide** — ce n'est pas un défaut, c'est une garde à ajouter
(`assert _live_pipeline_keys()`, avec le nombre 10 écrit dedans) le jour où
ce fichier sera rouvert. Il ne l'est pas ici : une seconde session Vigil y
travaille en parallèle (D-131), et deux mains dans le même fichier coûtent
plus qu'elles ne rapportent.

**Un 5ᵉ site du sondage `.read()` vérifié la même nuit, et il est sain** :
`test_provenance.py:108` (`test_long_tasks_no_longer_stamp_at_save_time`,
`assert "git_commit_hash()" not in src`). Assertion **négative** sur le texte,
donc a priori de la mauvaise forme — mais l'invariant qu'elle garde (D-15 : le
hash écrit est celui pris au DÉPART) est tenu à l'intérieur de
`provenance.finish`, pas aux sites d'appel, et il y est vérifié
fonctionnellement : `test_provenance.py:51-53` assertent
`out["git_hash"] == p["git_hash_at_start"]`. Le texte n'est qu'une ceinture
au-dessus des bretelles. **Verdict : surestimé, pas défaut** — 2ᵉ cas où la
question « et qui couvre le comportement, alors ? » sauve un site plutôt que
d'en condamner un.

**Le tri par `grep -c`, exécuté sur tout `tests/` la même nuit — et il ne rend
rien de plus.** Il fallait le mesurer plutôt que de l'annoncer : un script
apparie chaque `assert "…" in src` à sa cible, puis compte les occurrences de
la chaîne **hors code exécutable** (commentaires et docstrings, par
tokenisation et AST — pas par `grep`, qui ne distingue pas). Sur l'arbre
entier : **3 candidats**, et aucun n'est un défaut.

| candidat | pourquoi ce n'en est pas un |
|---|---|
| `test_t24_leak_free.py:278` → `is_ic_variation` | les 2 occurrences sont des **clés de dictionnaire** (`h4_physics_robustness.py:411,491`), donc du code exécutable — l'heuristique compte tout littéral de chaîne, elle a sur-signalé |
| `test_t24_leak_free.py:73` → `DIFFERENT operating points` | l'objet du test **EST** le texte : il garde une réserve documentée. Type légitime, celui que le critère de cette section autorise |
| `test_objective_and_estimators_analytic.py:335` → la formule `w = 1 + 0,25 × (…)` | idem : une formule citée dans un commentaire, gardée comme mention |

**Ce que ça vaut.** Le tri est bon marché et sans faux négatif sur sa propre
question, mais sa question est étroite : il n'aurait trouvé que D-124 des trois
défauts de cette passe. D-123 (`and` → `or`) et D-125 (un consommateur sur deux
non couvert) ne laissent **aucune trace textuelle** — seule la mutation les
voit. À garder comme premier filtre, jamais comme balayage.

**Calibrage cumulé** : 3 défauts sur 4 sites sondés cette passe, contre 1 sur
3 aux deux précédentes. Ce n'est pas une dérive du taux du dépôt — c'est que
le bloc `test_t24_leak_free.py` a été écrit d'un seul tenant, dans un style
qui garde tout par le texte. Les grappes se sondent ensemble.

**23 sites restent non vérifiés** (sur les 27 signalés par le tri, candidats
seulement — ne pas les citer comme défauts sans les rejouer ; `:171` de la
première entrée est sorti de cette liste, vérifié et devenu D-121) :
`test_hyperparams_provenance_break.py:211` ·
`test_objective_and_estimators_analytic.py:574` ·
`test_h0_panel_resume.py:97,108,199` ·
`test_provenance.py:108` ·
`test_t22c_transfer.py:120` ·
`test_fixed_curl_variant.py:183` ·
`test_silent_failure_sweep.py:82` (usages 141, 165, 193) ·
`test_hyperparams_two_sources.py:226` (usage 232) ·
`test_v1_legacy_instrumented_bfs_score_grid.py:100` ·
`test_t28_t29_labels_and_ci.py:103,179` ·
`test_t24_leak_free.py:207,224,233,269,277` (`:150,154,181,200` vérifiés
cette passe — D-123, D-124, D-125 ; `:207` reste : `assert "return got[:n]" in src`,
la chaîne EST le comportement sur une seule ligne, aucune mutation réaliste ne
la préserve en cassant la troncature — fragile mais honnête, non promu) ·
`test_h0_panel_guards.py:67`.

**Ce sondage n'est pas un balayage.** Avec les 3 vérifiés cette passe (6 au
total sur 64), 58 sites restent sans verdict fiable — 24 avec un tri qui
demande confirmation, 34 jamais relus. Terrain neuf de la prochaine passe,
dans cet ordre : d'abord confirmer ou infirmer les 24, puis lire les 34.

**Ce décompte n'a pas été retenu à jour aux deux passes suivantes** (16 août
soir — D-120 — et 16 août nuit — D-121) : chacune ajoute un site vérifié,
sans qu'on ait recompté ici plutôt que de risquer un chiffre approximatif —
le « compte de tête inexact » que ce document reproche déjà à `RESULTS.md`
ailleurs. Ce qui est sûr, compté à cette passe : la liste des 24 candidats
issus du tri n'en porte plus que **23** (`:171` de
`test_hyperparams_provenance_break.py` en est sorti, devenu D-121) ; les deux
autres sites vérifiés depuis (D-120, et les 3 + 3 d'avant) ne viennent pas de
cette liste. Un recompte complet de `grep -rn "\.read()" tests/` avant la
prochaine passe dirait le total exact plutôt que de le supposer inchangé.

---

## Passe du 17 août — six faux verts, et le critère « absence d'un jeton » rouvert

Six sites du sondage `.read()` vérifiés par mutation **dans les deux sens**,
disjoints de ceux de la passe concurrente de la même nuit. Détail chiffré et
commandes dans `RESULTS.md`, D-127 / D-129 / D-130 / D-131 / D-132 / D-133.

| site | ce qui était gardé par le texte | **A′** — comportement cassé, texte intact | **B** |
|---|---|---|---|
| `test_h0_panel_resume.py:199` | la durabilité du point de sauvegarde H0 | `flush`+`fsync` sous une condition fausse → **21 passed** | réécriture équivalente → ROUGE |
| `test_t28_t29_labels_and_ci.py:103` | le suffixe `_globalthr` (défaut **D9**) | suffixe retiré du chemin écrit → **72 passed** | chemin bit à bit identique → ROUGE |
| `test_fixed_curl_variant.py:183` | `--fixed-curl` atteint le mappeur | drapeau réduit à un renommage → **7 passed** | `bool(...)` → ROUGE |
| `test_hyperparams_two_sources.py:226` | une seule définition du seuil classique | seuil recopié, ligne cherchée en code mort → **12 passed** | littéral équivalent → ROUGE |
| `test_v1_legacy_...bfs_score_grid.py:100` | le non-retour de **D-96/D-37** | défaut réintroduit **sans les espaces** → **7 passed** | commentaire documentant la déviation → ROUGE |
| `test_provenance.py:108` | le non-retour de **D15** | D15 réintroduit via un **alias d'import** → **7 passed** | — |

**Le résultat de méthode de cette passe : « absence d'un jeton » n'est pas
une forme légitime en soi.** Ce document classait cette forme légitime au vu
de deux sondages (« aucune formulation de code ne le contourne »). D-132 et
D-133 la prennent en défaut, chacun par une écriture qu'un développeur
normal produirait : **retirer deux espaces**, et **aliaser un import**. Le
critère qui tient n'est pas *présence contre absence*, c'est :

> une recherche de texte est juste quand l'objet du test **est** ce texte, et
> qu'aucune écriture équivalente du même comportement ne lui échappe. La
> seconde moitié se **mesure**, elle ne se raisonne pas — les deux fois où on
> l'a raisonnée ici, on s'est trompé.

Le remède est le même que celui que la passe concurrente formule pour la
fenêtre de proximité : **l'AST délimite par la structure**. D-132 exige que
le 3ᵉ argument de `_process_score` soit le *nom* `target_dim` ; D-133 résout
d'abord les alias d'import, puis cherche l'appel.

**Trois sites re-jugés à ce critère, et un qui tient.**

| site | verdict antérieur | à ce critère |
|---|---|---|
| `test_amr_figure_axes.py` — `assert "D-68" in src` | légitime | **tient** — l'objet EST le texte : une mention de déviation qui doit rester, ce que `VIGIL.md` exige |
| `test_qaoa_arm_is_sampled.py` — aucun `seed_*` dans `src/VQA/` | légitime | **tient, et pour la bonne raison** : le comportement est mesuré **dans le même fichier** par `test_the_arm_is_not_reproducible`, qui rougirait si une graine était posée. Le balayage de jetons ne fait que **nommer la cause** ; il n'est pas le garde. C'est exactement la configuration qui rend la forme acceptable |
| `test_v1_legacy_...:100`, `test_provenance.py:108` | légitimes par la même règle | **faux** — D-132, D-133 |

**Le recomptage que ce document réclamait.** `grep -rn "\.read()" tests/
--include=*.py`, hors `tests/tools/` : **85 sites, 45 fichiers** — pas 64/41.
Le total a grandi avec la suite depuis le 15 août, donc « 58 restants » était
un plancher, jamais le total. Le nombre à citer est celui-ci, remesuré.

**Suite de la passe — trois sites de plus, et deux verdicts NÉGATIFS qui
comptent autant.** Le sondage a rendu deux défauts supplémentaires et trois
non-défauts, tous mesurés par mutation, pour que la prochaine passe ne les
rejoue pas :

| site | verdict |
|---|---|
| `test_t22c_transfer.py:120` — `--mode leak-free` | **D-134**, confirmé : la fuite **D13** réintroduite (seuil QAOA remis à `LEAKED_THRESHOLD` juste après), les QUATRE chaînes intactes → **35 passed**, sous un artefact nommé `leak-free` |
| `test_objective_and_estimators_analytic.py:574` — l'accord des deux chemins de score | **D-135**, confirmé mais **NON CORRIGÉ** : la L2 non pondérée de D-5 réintroduite sur le chemin de divergence, l'appel partagé laissé en place et son résultat réécrit → **46 passed**. C'est `src/pipeline.py`, le chemin déployé : mesuré, documenté, pas corrigé — deux directions dans `DEFAUTS.md` |
| `test_h0_panel_resume.py:108` — la clé morte `mask_match` a disparu | **surestimé** — évadable (`r["mask_match"] = float("nan")` → **22 passed**), mais la moitié qui porte une valeur (les trois clés que la boucle LIT) est couverte comportementalement au site `:97`, qui appelle `decision_agreement`. Réintroduire une clé morte ne fabrique pas une valeur plausible et fausse |
| `test_hyperparams_provenance_break.py:211` — `0.14959824837662078` en dur | **surestimé** — même forme que D-120 : `CLASSICAL_BEST_THRESHOLD` porté à `0,2` en gardant le littéral ailleurs dans le fichier laisse ce fichier à **15 passed**, mais `test_the_frozen_threshold_is_the_measured_classical_best` (`test_solver_guards_and_objective.py:126`) **rougit**. Le comportement est couvert |
| `test_h0_panel_resume.py:97`, `test_silent_failure_sweep.py:82`, `test_t28_t29_labels_and_ci.py:179` | **légitimes** — le premier appelle `decision_agreement` avant de lire le source ; les deux autres passent par `ast.parse` et par `importlib`, pas par le texte |

**Le calibrage tient, et il s'affine.** Sur les **11 sites** sondés à la main
depuis le 15 août : **7 défauts**, **3 surestimés**, **1 légitime confirmé**.
Le tri automatisé, lui, reste à ~1 sur 3 — c'est un premier filtre, jamais un
verdict. Et la règle qui a rendu le plus : **vérifier un « surestimé » n'est
pas du temps perdu** — D-120 est né de là, et les deux surestimés ci-dessus
ont chacun désigné le test qui couvre vraiment le comportement, ce qu'aucune
lecture n'aurait donné.

**Ce que cette passe n'a PAS fait**, écrit pour ne pas être supposé fait :
aucun code de `src/`, `study/` ou `figures/` n'est modifié — les six
corrections sont entièrement dans les tests, parce que dans les six cas le
code est juste et c'est sa couverture qui manquait. Aucun nombre publié ne
bouge. Les axes de la fiche ne sont pas re-traversés : ils l'étaient déjà
depuis D-119.

---

## Les commandes publiées — le chemin était testé, l'option non (D-140)

`tests/study/test_repro_commands_point_to_real_files.py` vérifiait depuis
D-71 que tout script cité par une commande de `RESULTS.md` **existe**. Rien
ne vérifiait que les **options** citées existent : une commande peut pointer
sur un fichier bien réel et sortir en 2 sur `unrecognized arguments`.

C'est ce qui est arrivé à la ligne « Vérifier » de **D-53**, le résultat le
plus fort du dépôt — voir `RESULTS.md`, D-140.

**Ce qui est testé maintenant.** `test_every_repro_command_uses_options_its_script_declares`
extrait chaque commande `python <script du dépôt> --…` de `RESULTS.md`, puis
interroge le **parseur** du script par son propre `--help`. L'assertion porte
sur le comportement déclaré, pas sur le texte du source : renommer une option
dans le code la fait rougir, reformater le fichier non.

**Ce qui est mesuré.** 16 commandes distinctes à options, 12 scripts
interrogés, 0 ignoré. Garde anti-balayage-vide : le test exige d'en trouver
au moins 10 et le dit dans son message. ~20 s.

**Ce que ce test ne couvre pas, écrit pour ne pas le croire couvert :**

- les options **courtes** (`-q`, `-k`) et les commandes `pytest`, hors motif ;
- les **valeurs** passées aux options — `--dim 3` est accepté ici quelle que
  soit la valeur, seul le nom de l'option est vérifié ;
- les commandes de `DEFAUTS.md` et de `COUVERTURE.md` : seul `RESULTS.md`
  est balayé, parce que c'est lui qui porte le contrat « un résultat qu'on ne
  sait pas refaire n'est pas un résultat » ;
- un script dont `--help` ne rend pas 0 est **ignoré**, pas signalé — aucun
  aujourd'hui, mais le jour où il y en aura un, il passera en silence.

## `preflight_coefficients.py` — la porte de la campagne, ses cinq contrôles sondés (D-141)

Module lu en entier. C'est la porte de la réoptimisation : il décide si
~224 h CPU partent. Rien ne le testait.

**Ce qui est vérifié maintenant** — `tests/study/test_preflight_pertinence_separates.py`,
5 tests, ~10 s, déterministe (deux exécutions identiques au dernier
chiffre) :

| test | ce qu'il tient |
|---|---|
| `the_replica_is_the_control_itself` | **opérateur assorti** : la réplique du calcul rend le rho du contrôle à **1e−12**. Sans lui, tout le reste du fichier mesurerait autre chose |
| `the_control_rejects_pure_noise` | contrôle positif : le seuil n'est pas vide, le bruit blanc rate à −0,0401 |
| `bare_physical_fields_clear_the_same_threshold` | le cœur de D-141 : \|Jz\|, \|v\| et \|∇\|B\|\| passent sans porter aucun coefficient |
| `the_classical_baseline_clears_it_better…` | le point qui décide : score classique **+0,8137** contre `K_plaquettes` **+0,7977** |
| `only_one_of_the_four_channels_is_looked_at` | `K_xpoint` (+0,4345) ne franchirait pas le seuil — « les coefficients » désigne un seul canal |

Les assertions portent sur des **ordres**, pas sur les valeurs : le dépôt
n'épingle aucune version de `numpy`/`scipy`. Les valeurs sont écrites dans
la docstring pour qu'une dérive se voie à la lecture.

**Ce sont des tests de déviation**, comme ceux de D-53 : ils échouent le
jour où le contrôle gagne un critère de discrimination, donc le jour où
D-141 est tranché. Vérifié en mutant le seuil du contrôle de 0,6 à 0,85 :
**2 failed** — ils suivent bien le critère et ne sont pas décoratifs.

**Ce que ce fichier ne couvre PAS**, écrit pour ne pas le croire couvert :
les contrôles `specificite`, `equilibre`, `vivant` et `coincidence` sont
**exécutés** par le preflight mais **aucun test ne les sonde** — on ne sait
pas, pour eux, sur quelle entrée ils échoueraient. Seul `pertinence` a été
pris pour cible, parce que c'est celui dont le libellé porte le verdict.

## Tenir ce document à jour

À chaque passe : ajouter ce qui vient d'être audité, retirer de la liste
« jamais audité ». Quand un module est fini, écrire qu'il a été **vérifié et
trouvé sain** — c'est un résultat, et cela évite de le relire deux fois.

Remesurer la couverture — **sur tout `src/`**, et sans exclure les suites
QAOA : les exclure fausse `pipeline.py`, `call_vqa_shell.py` et `solver.py`,
et c'est ce qui rendait la mesure précédente incomparable à celle-ci.

```bash
python -m coverage run --source=src -m pytest tests/ -q -m "not slow"   # ~36 min
python -m coverage report --include="src/*"
```

Quand un chiffre bouge, dire **ce qui a changé : le code ou le périmètre de
mesure.** Publier deux séries non comparables comme si elles l'étaient est
la façon la plus simple de fabriquer un progrès qui n'a pas eu lieu.
