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
| `exact_diagonalisation.py` | lu, aucun défaut mesuré ; `analyze_snapshot` reçoit `is_hard` sans jamais s'en servir (recalculé à l'identique depuis `l2_threshold`) et `H_mat` est diagonale, donc `eigh` est un argmin coûteux — deux gaspillages, pas des valeurs fausses. **Aucun artefact `exact_diag_*` dans `results/`** : rien à rejouer, la clause `promising = f1_exact >= f1_classique` (le commentaire au-dessus dit `>`) reste **non mesurée** |
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
