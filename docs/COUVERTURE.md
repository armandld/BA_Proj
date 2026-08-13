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

### `qaoa_inputs.py` — lu en entier

Le module qui fabrique les entrées QAOA de la phase 5 et des études h0 / h3.

| lu | verdict |
|---|---|
| `classical_warm_start_params` | **D-48** — schedule **constant** : ni `score_vqa` ni `threshold_amr` n'entrent dans le résultat. Sortie identique bit-à-bit sur 6 entrées couvrant tout l'intervalle, écart **0,0e+00**. Le nom, la docstring et l'aide CLI annonçaient un warm start dérivé de la décision classique |
| `prepare_qaoa_inputs`, réduction `block_avg` des champs / `block_max` du score | **déjà mesuré par D-47, ne pas y revenir** — l'écart d'opérateur est réel et n'est *pas* la cause de la dégénérescence (39/40 contre 40/40 avec le score assorti) |
| `_psi_from_pipeline` | **sain par construction** — délègue à `refinement._prepare_vqa_input`, l'encodeur réellement déployé, au lieu de le réimplémenter ; lève si l'encodeur refuse le patch plutôt que de fabriquer un psi |
| `prepare_qaoa_inputs`, garde `with_psi` sans `prev_fields` | **saine** — lève explicitement : psi est une dérivée temporelle, il ne peut pas naître d'un instantané isolé |
| `full_comparison.metrics` | **sain** — dénominateurs gardés par `max(·, 1)` ; sur un prédicteur vide `tp = 0` rend F1 = 0, pas une division par zéro |
| `run_phase5`, chaînage du warm start | **sain et conforme au déployé** — enchaîne `optimal_params` d'un instantané au suivant, comme `refinement.py` le fait via `warm_start_cache`. Le schedule constant de D-48 n'y sert **que** pour le premier instantané, et seulement sous `--warm-start`. C'est h0 / h3 qui l'appliquent à **chaque** appel |
| `run_phase5`, absence de patch prometteur | **crie** — `No promising patches -- skipping QAOA`, pas de balayage muet |
| `prune_hamilt_params` | **incohérence docstring / code, sans conséquence mesurable** — la docstring annonce un élagage « par bloc » sur `H_edges`, `C_edges`, `K_plaquettes` (3 blocs) ; le code prend un maximum séparé pour `H0` et `H1`, `C0` et `C1` (5 groupes). Non corrigé : **aucun artefact `*depth*` n'existe dans `results/`** et aucune ligne de `RESULTS.md` ne cite l'élagage, donc aucun nombre publié n'en dépend — corriger sans mesure serait du risque sans gain |

### `aggregate_v2.py`, `aggregate_v3.py` — lus en entier

| lu | verdict |
|---|---|
| `aggregate_v3.status_of` / `make_row` / `collect` | **sains** — `None` devient MISSING, `--strict` sort non nul sur DIFF ou MISSING, provenance (hash git + CLI) écrite dans le `.npz`. Les extracteurs indexent par `names.index(...)`, qui lève sur un nom absent : pas de repli silencieux |
| `aggregate_v3.rows_t9`, sélection vide | **saine** — `mask.any()` faux donne `None`, donc MISSING, donc visible |
| `aggregate_v2`, verdict « ZZ/ZZZZ add NO measurable value » | **bande codée en dur sans provenance** (`d_sten < 0.02`), même famille que le ±0,02 déjà noté dans `ising_terms_and_annealing`. Et le bloc entier est imbriqué sous `d_site is not None` : une exécution où `d_sten` existe mais pas `d_site` n'imprime **aucun** verdict. Ni l'un ni l'autre n'est une valeur fausse, et `SUMMARY_*` n'est cité par aucune ligne de `RESULTS.md` |

Il ne reste **plus qu'un** fichier non relu dans `study/common/` :
`aggregate_master_table.py` (exécuté à chaque passe, jamais relu
fonction par fonction).

**Aucun de ces trois modules n'est « audité » au sens de la fiche** : ils
ont été lus en entier et, pour `qaoa_inputs.py`, mesurés là où une mesure
tranchait — mais aucun test n'en traverse les axes. Axes empruntés par les
mesures de D-48 (`qaoa_inputs.py`) : bras **quantique**, backend
**state_vector**, hamiltonien **non nul**, bord **périodique**, warm start
**présent *et* absent** (c'est la mesure elle-même), optimiseur **COBYLA**,
AMR **depth = 0**, `dim = 2` seulement. Restent non traversés : le bras
`classical_only`, le backend échantillonné, le bord borné, l'hamiltonien
nul, les autres optimiseurs, `depth > 0`, et `dim = 4 / 8`.

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
