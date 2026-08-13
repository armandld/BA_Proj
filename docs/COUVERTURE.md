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

**1 827 tests**, 73 fichiers. Commandes dans `tests/README.md`.

---

## 1. Ce qui n'est pas couvert — la liste qui dit quoi faire

### Lignes jamais exécutées

Mesuré par `coverage`, suites QAOA et mesures `slow` exclues *(elles
exercent `pipeline.py`, `call_vqa_shell.py` et `hyperparams_loader.py` : les
chiffres de ces trois-là sont donc faux par défaut)*.

| module | couverture | ce qui manque |
|---|---|---|
| `pipeline.py` | **14 %** | la boucle fermée elle-même — exercée par les suites QAOA, non comptée ici |
| `train_hyperparams.py` | **89 %** | le mode Colab, l'analyse des CSV de rescore en erreur |
| `hyperparams_loader.py` | **28 %** | les sélecteurs par scénario, combo, phase, rang |
| `VQA/execute.py` | **49 %** | les branches matériel, MPS, et la boucle COBYLA |
| `Simulation/solver.py` | **52 %** | `step_layered` en profondeur, le sous-cyclage |
| `call_vqa_shell.py` | **63 %** | la normalisation des coefficients, le warm-start |
| `VQA/optimize.py` | **65 %** | les backends autres que `state_vector` |
| `Simulation/utils.py` | **65 %** | `slice_hamiltonian_params`, non appelé par le chemin déployé |
| `Simulation/refinement.py` | **73 %** | le TTL, le sondage de bord, la reprise |

**Total mesuré : 55 %** sur `src/`, chiffre à lire avec la réserve ci-dessus.

### Fonctions dont le contrat n'a jamais été audité

Pas mesurable automatiquement. Liste tenue à la main, depuis ce qui a été
effectivement relu fonction par fonction.

**Jamais audité** — aucune des quatre questions n'y a été posée :

| fichier | lignes | pourquoi ça compte |
|---|---|---|
| `analyze_hyperparams.py` | 918 | analyse de la campagne |
| `recompute_lambda_scores.py` | 717 | recalcul de scores publiés |
| `compare_rotor_budget.py` | 481 | comparaison de budget, utilise le pipeline |
| `visual.py`, `help_visual.py` | 327 | figures |
| `import_Neon_data_to_local.py` | 76 | import de données |

**~2 520 lignes**, toutes en aval du chemin scientifique : elles lisent des
résultats, elles n'en produisent pas.

`TrainHyperParam_v1/v3/v4.py` (1 641 lignes) figuraient ici. **Supprimés** :
quatre variantes du même script d'entraînement coexistaient sans qu'aucune ne
soit désignée. `TrainHyperParam_v2.py` est renommé `train_hyperparams.py`,
audité fonction par fonction, et couvert par 67 tests — voir D-27 à D-36 dans
`RESULTS.md`.

**Partiellement audité** — le contrat a été vérifié sur une partie des
fonctions seulement :

| fichier | ce qui reste |
|---|---|
| `Simulation/refinement.py` | le TTL, la reprise de campagne |
| `train_hyperparams.py` | le mode Colab (non testable ici) |
| `VQA/execute.py` | les branches **matériel** (`mode != "simulator"`, session IBM) |
| `pipeline.py` | le mode `classical_only` de bout en bout |
| `study/` | **en totalité** — c'est le chantier suivant |

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
| `hard_patch_labels.patch_classical_scores`, `Jz` écrit à la main | **sain** — stencil et convention d'axes **identiques** à `solver.get_fluxes` (`AXIS_X=0`, centré, `/2dx`) : pas d'opérateur dépareillé entre le score classique des artefacts `patches_*` et celui du chemin coefficients |
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

---

## 2. Ce qui est couvert, et par quel type de test

### Les cinq familles

**Tests analytiques** — une entrée à réponse connue, une sortie exacte.
Champ construit pour que la bonne réponse soit calculable à la main.
*Exemple : une rotation solide doit donner ω = +2,0 ; l'enstrophie d'un
cisaillement `vx = sin y` vaut 2π².*

**Audits de contrat** — pour chaque fonction : pourquoi existe-t-elle, que
promet sa docstring, consomme-t-elle ce que sa signature annonce, et deux
chemins censés coïncider coïncident-ils encore ? **Douze des 27 défauts
viennent de la quatrième question.**

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
| `tests/solver/` | `solver.py`, `grid.py`, `pre_compute_dns.py` | 52 / 90 / 98 % | opérateurs, projection, scénarios, trace DNS |
| `tests/mapping/` | `PhysToAngle`, `HamiltParams`, `HamiltParams_v2`, `RescaleArrays` | **100 / 98 / 100 / 97 %** | **complet** |
| `tests/quantum/` | `VQA/*` | 90–100 % sauf `execute` (49 %) | hamiltonien, chaîne de décision, runtime |
| `tests/amr/` | `refinement.py`, `utils.py` | 73 / 65 % | pavage, rééchantillonnage |
| `tests/pipeline/` | `pipeline.py`, `hyperparams_loader.py`, `train_hyperparams.py` | 14 / 28 / **89 %** | provenance des hyperparamètres, espace de recherche, budget d'essais, routage des 8 phases, campagne miniature de bout en bout |
| `tests/study/` | tout `study/` | non mesuré | **aucun** |

---

## 3. Ce qui rend un test digne de confiance

Un test peut passer sans rien prouver. Quatre pièges rencontrés dans ce
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

**Le seuil périmé.** Un test calibré sur la mesure du jour cesse de mesurer
dès que le code change légitimement. Il ne s'actualise pas : il se
**remesure**, avec l'ancienne et la nouvelle valeur consignées. Et si la
grandeur s'avère non reproductible, on change de **grandeur**, pas de seuil.

---

## 4. Ce qui est reproductible

| | état |
|---|---|
| tests | **1 827**, déterministes sauf les suites QAOA |
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

## Tenir ce document à jour

À chaque passe : ajouter ce qui vient d'être audité, retirer de la liste
« jamais audité ». Quand un module est fini, écrire qu'il a été **vérifié et
trouvé sain** — c'est un résultat, et cela évite de le relire deux fois.

Remesurer la couverture :

```bash
python -m coverage run --source=src -m pytest tests/ -q -m "not slow"
python -m coverage report --include="src/Simulation/*,src/VQA/*,src/pipeline.py"
```
