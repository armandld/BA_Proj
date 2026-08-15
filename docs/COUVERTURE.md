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
