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

**1 771 tests**, 71 fichiers. Commandes dans `tests/README.md`.

---

## 1. Ce qui n'est pas couvert — la liste qui dit quoi faire

### Lignes jamais exécutées

Mesuré par `coverage`, suites QAOA et mesures `slow` exclues *(elles
exercent `pipeline.py`, `call_vqa_shell.py` et `hyperparams_loader.py` : les
chiffres de ces trois-là sont donc faux par défaut)*.

| module | couverture | ce qui manque |
|---|---|---|
| `pipeline.py` | **14 %** | la boucle fermée elle-même — exercée par les suites QAOA, non comptée ici |
| `TrainHyperParam_v2.py` | **14 %** | les phases d'entraînement ; seul `search_space` et l'objectif sont testés |
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
| `TrainHyperParam_v1/v3/v4.py` | 1 641 | trois variantes de l'objectif d'entraînement ; c'est dans `v2` que vivait D-3 |
| `analyze_hyperparams.py` | 918 | analyse de la campagne |
| `recompute_lambda_scores.py` | 717 | recalcul de scores publiés |
| `compare_rotor_budget.py` | 481 | comparaison de budget, utilise le pipeline |
| `visual.py`, `help_visual.py` | 327 | figures |
| `import_Neon_data_to_local.py` | 76 | import de données |

**~4 160 lignes.** `TrainHyperParam_v1/v3/v4` est la seule entrée de cette
liste qui touche le chemin scientifique : à auditer **avant** toute
réoptimisation qui en utiliserait une.

**Partiellement audité** — le contrat a été vérifié sur une partie des
fonctions seulement :

| fichier | ce qui reste |
|---|---|
| `Simulation/refinement.py` | `_run_level_classical`, TTL, reprise de campagne |
| `VQA/execute.py` | la boucle COBYLA, les contraintes sur β, les branches matériel |
| `pipeline.py` | le calcul de score, la garde de divergence, le mode `classical_only` |
| `study/` | **en totalité** — c'est le chantier suivant |

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
| `tests/pipeline/` | `pipeline.py`, `hyperparams_loader.py` | 14 / 28 % | provenance des hyperparamètres |
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
deux opérateurs. **Trois occurrences ici**, dont une où un défaut de huit
ordres de grandeur restait invisible.

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
| tests | **1 771**, déterministes sauf les suites QAOA |
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
