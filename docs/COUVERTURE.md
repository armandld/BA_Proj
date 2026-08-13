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

**1 944 tests**, 75 fichiers. Commandes dans `tests/README.md`.

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

**Jamais audité** — aucune des cinq questions n'y a été posée :

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
| `tests/solver/` | `solver.py`, `grid.py`, `pre_compute_dns.py` | 52 / 90 / 98 % | opérateurs, projection, scénarios, trace DNS |
| `tests/mapping/` | `PhysToAngle`, `HamiltParams`, `HamiltParams_v2`, `RescaleArrays` | **100 / 98 / 100 / 97 %** | **complet** |
| `tests/quantum/` | `VQA/*` | 90–100 % sauf `execute` (49 %) | hamiltonien, chaîne de décision, runtime |
| `tests/amr/` | `refinement.py`, `utils.py` | 73 / 65 % | pavage, rééchantillonnage |
| `tests/pipeline/` | `pipeline.py`, `hyperparams_loader.py`, `train_hyperparams.py` | 14 / 28 / **89 %** | provenance des hyperparamètres, espace de recherche, budget d'essais, routage des 8 phases, campagne miniature de bout en bout |
| `tests/study/` | tout `study/` | non mesuré | **aucun** |

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
| tests | **1 944**, déterministes sauf les suites QAOA |
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
