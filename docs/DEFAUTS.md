# Registre des défauts

Index de référence. Une ligne par défaut, avec **la commande qui vérifie son
état**. Pour savoir où on en est, lancer la commande — pas relire un rapport.

- Les **mesures détaillées** (avant/après, conditions) sont dans
  `docs/RESULTS_V4.md`.
- Le **contenu destiné au papier** est dans `docs/PLAN_PREPRINT.md`.
- Ce fichier ne contient ni l'un ni l'autre : seulement l'état et la preuve.

**Vérification globale**

```bash
python -m pytest tests/ --ignore=tests/v3 --ignore=tests/v4 -q -m "not slow"
python -m pytest tests/v3 tests/v4 -q
python study/common/aggregate_master_table.py     # 180 lignes, 0 DIFF attendu
```

---

## Résumé

| état | nombre |
|---|---|
| **corrigés et verrouillés par un test** | 20 |
| **gelés volontairement** (reproductibilité) | 2 |
| **ouverts** — décision ou campagne requise | 2 |
| **total** | **24** |

---

## Corrigés

Chacun a un test qui **échoue sur l'ancienne version**. La commande donnée
vérifie la correction.

### Conventions et opérateurs

| # | défaut | avant → après | vérifier |
|---|---|---|---|
| **D-1** | rotationnel des mappeurs sous `indexing='xy'` | 0,0 → **+2,0** sur rotation solide | `pytest tests/v4/test_curl_convention_gap.py tests/v4/test_fixed_curl_variant.py` |
| **D-3** | l'objectif pondère par cette vorticité fausse | 0,0 → **+2,0** | `pytest tests/test_objective_and_estimators_analytic.py` |
| **D-11** | diode de choc appliquée au cisaillement | rapport **0,500 → 2,0** ; diode inerte → vivante | `pytest tests/test_mapper_contracts.py -k flux` |
| **D-17** | 3 sites hors `src/` en convention pré-D-1 | enstrophie **0 % → 0,02 %** d'écart | `pytest tests/v4/test_no_private_curl_survives.py` |
| — | critère Q : déformation à moitié, partie isotrope comptée | cisaillement **+0,25 → 0** | `pytest tests/test_analytic_fields.py -k q_criterion` |

> L'opérateur fautif n'est **pas** un rotationnel de signe opposé — `abs` et
> le carré n'auraient rien rattrapé. Il vaut `∂fy/∂y − ∂fx/∂x`, le
> *complémentaire* : nul là où le rotationnel est maximal.

### Numérique et rééchantillonnage

| # | défaut | avant → après | vérifier |
|---|---|---|---|
| **D-2** | prolongation AMR au centre des cellules, `mode='wrap'` | 2,49e−1 → **7,74e−6** | `pytest tests/test_amr_resampling_analytic.py` |
| **D-7** | projection ignore le mode de Nyquist | 0,378 → **1,1e−14** | `pytest tests/test_solver_analytic.py -k idempot` |
| **D-14** | réduction des champs tronque, celle du score non | 94,1 % → **100 %** de couverture | `pytest tests/test_downsampling_contracts.py` |
| **D-21** | flux réduit par lissage + interpolation bilinéaire | pic **38 % → 100 %** | `pytest tests/test_padded_rescale_contracts.py` |
| **D-23** | `dt` intégré ≠ `dt` écrit dans la trace DNS | référence à t≈0,077 au lieu de 0,050 → rejeu **exact** | `pytest tests/test_precompute_dns_contracts.py` |

### Encodage et décision

| # | défaut | avant → après | vérifier |
|---|---|---|---|
| **D-8** | hamiltonien encode des coefficients nuls sans lever | non détecté → **lève** | `pytest tests/test_hamiltonian_contracts.py -k raises` |
| **D-13** | bords gauche/haut lisent l'arête intérieure | asymétrie 1,2–7,0 % → **symétrique** | `pytest tests/test_hamiltonian_contracts.py -k halo` |
| **D-15** | `postprocess` accepte des comptes bruts | marginales ~1000 → **refusé** | `pytest tests/test_vqa_chain_contracts.py -k refus` |
| **D-16** | liste de patchs AMR se recouvre elle-même | **25 % → 0 %**, sans trou | `pytest tests/test_amr_tiling_contracts.py` |
| **D-19** | backend inconnu → contexte mort sans erreur | silence → **lève** | `pytest tests/test_runtime_contracts.py -k backend` |
| **D-20** | cache d'ansatz confond deux hamiltoniens | même objet → **séparés** | `pytest tests/test_runtime_contracts.py -k ansatz` |

### Mesure, gardes et documentation

| # | défaut | avant → après | vérifier |
|---|---|---|---|
| **D-4** | doc annonce le double du facteur appliqué | ×2 → aligné | `pytest tests/test_objective_and_estimators_analytic.py` |
| **D-5** | chemin de divergence noté sans pondération | 1,8 % → **0** | `pytest tests/test_objective_and_estimators_analytic.py` |
| **D-6** | `init_magnetic_twist` ne pose aucune torsion | 6,4e−7 rad → **π/2 exact** | `pytest tests/test_scenarios_analytic.py -k twist` |
| **D-9** | ablation ψ mesure la fenêtre sur le mauvais score | « annihilation » → **ZZ domine K de 1,5 à 8,2×** | `pytest tests/v4 -k window` |
| **D-12** | mappeur `study/` : ν, η, dx annoncés influents, sans effet | doc alignée sur le code | `pytest tests/test_mapper_contracts.py -k v2` |
| **D-18** | garde de divergence à 1e100, inerte | 1e50 passait → seuil **1e8** | `pytest tests/test_solver_guards_and_objective.py -k caught` |
| — | `search_space` : 4 constantes présentées comme réglables | espace réel **5 paramètres**, pas 9 | `pytest tests/test_solver_guards_and_objective.py -k search` |
| — | `sigma` : repli silencieux sur 0,05 | **avertit** et consigne l'origine | `pytest tests/test_solver_guards_and_objective.py -k sigma` |

---

## Gelés volontairement

Ne pas corriger. Les originaux reproduisent les artefacts publiés de
phase 1b ; les versions correctes vivent à côté.

| # | défaut | où | correction disponible |
|---|---|---|---|
| **D2** | `fluctuating_KE` moyenne à travers la couche de cisaillement — sur le profil de base seul elle lit **73 %** de l'énergie totale | `study/pipeline/dns_validation.py` | `dns_extension.fluctuating_ke_fixed` |
| **D3** | `mean_sq_current` porte la même inversion d'axes | idem | `dns_extension.mean_sq_current_fixed` |

`analyse_one` utilise désormais les versions corrigées. Le gel porte sur les
**fonctions**, pas sur l'analyse qui les appelle.

```bash
pytest tests/v4/test_no_private_curl_survives.py tests/v3/test_t8_dns_extension.py
```

---

## Ouverts

### D-22 — les hyperparamètres déployés n'ont aucune provenance

`best_hyperparams.json` ne correspond à **aucune** base Optuna du dépôt :

- `gamma_hydro`, `gamma_mag`, `kappa` — jamais échantillonnés nulle part ;
- `sigma` — échantillonné (meilleur essai 0,0230), **absent** du JSON, donc
  repli sur 0,05 ;
- l'essai 85 déclaré a une perte de 0,3213 dans la base contre 0,2215
  annoncée, et **aucun** de ses quatre paramètres communs ne coïncide.

Le bras quantique tourne à `threshold_amr = 0,3044`, valeur absente des 125
essayées, alors que l'objectif fixe 0,1496.

**Ne se corrige pas par du code.** Seule la réoptimisation le règle. Trois
paramètres n'ayant jamais été échantillonnés, ce sera pour eux une
*première*, pas une reprise.

```bash
pytest tests/test_hyperparams_provenance_break.py
```

Le dernier test du fichier est le **critère d'acceptation** : `xfail`
aujourd'hui, il passera sans modification le jour où chaque valeur déployée
sera traçable.

### D-24 — chute d'ordre du solveur, correction mesurée non applicable

Mesuré à grille fixe (N=96, T=0,5), chaque schéma contre sa propre référence :

| schéma | erreur à 256 pas | ordre | max\|div v\| |
|---|---|---|---|
| projection de l'état (actuel) | 1,093e−3 | **1,22** | 5,04e−3 |
| projection du second membre | **2,092e−11** | **4,00** | 5,11e−3 |
| aucune projection | 4,790e−7 | 4,01 | **5,89e+0** |

La correction rend l'ordre 4 **et** garde la divergence au même niveau. Mais
elle n'est valide que sur `step_full` : `step_layered` appelle le même
intégrateur sur un champ global sous-échantillonné (autre taille de grille,
lève) et sur des **patchs locaux non périodiques**, où une projection
spectrale périodique n'est pas définie. L'appliquer aux uns et pas aux autres
romprait la garantie « à `max_depth`, `step_layered` ≡ `step_full` ».

`MHDSolver.PROJECT_RHS = False`, raison écrite dans le code.

**Décision attendue**, par coût croissant :

1. **Laisser en l'état.** La chute est commune aux deux bras : elle ne biaise
   pas leur comparaison, elle comprime la plage où un meilleur critère
   pourrait se distinguer. *(Recommandé pour ce papier.)*
2. Projection par taille de grille + décision sur les patchs non périodiques.
   Casse probablement la garantie AMR.
3. Formulation à pression — réécriture du cœur du solveur.

```bash
pytest tests/test_solver_convergence.py -m slow     # ~10 min
```

---

## Dette de reproductibilité

`python study/common/aggregate_master_table.py` → **164 OK / 16 DIFF /
0 MISSING**.

Les 16 écarts sont exactement les nombres déplacés par les corrections
(t17, t11b). Ils doivent être **republiés ou justifiés ligne par ligne** —
publier une valeur qu'aucun artefact ne recalcule est ce que ce dépôt
s'interdit.

**Dépend de D-22** : ne pas s'y attaquer avant la réoptimisation.

---

## Comment ajouter une entrée

1. La mesure **avant**, avec sa commande.
2. La correction, minimale.
3. La mesure **après**, mêmes conditions.
4. Un test qui échoue sur l'ancienne version — et, quand c'est possible, un
   test qui **épingle l'ancien comportement**, pour que la correction ne
   puisse pas être défaite en silence.
5. Une ligne ici, avec la commande de vérification.
6. Le détail dans `docs/RESULTS_V4.md`.

Un défaut sans mesure n'est pas un défaut : c'est une suspicion. Un défaut
sans commande de vérification n'a pas sa place dans ce fichier.
