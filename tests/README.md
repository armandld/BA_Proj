# Tests

Organisés par **sous-système testé**, en miroir de `src/`. Le découpage
précédent — tout à plat, plus `v3/` et `v4/` — mélangeait l'objet testé et
l'époque de la campagne.

| dossier | ce qui est testé |
|---|---|
| `solver/` | `Simulation/solver.py`, `grid.py`, `pre_compute_dns.py` — intégrateur, opérateurs de grille, scénarios, trace DNS |
| `mapping/` | `PhysToAngle.py`, `HamiltParams*.py`, `RescaleArrays.py` — score, angles, coefficients, réduction |
| `quantum/` | `VQA/*` — hamiltonien, circuit, chaîne de décision, suites QAOA |
| `amr/` | `refinement.py`, `utils.py` — pavage des patchs, rééchantillonnage |
| `pipeline/` | `pipeline.py`, `hyperparams_loader.py`, entraînement, inventaire de couverture |
| `study/` | tout `study/` — anciennement `v3/` et `v4/` |
| `tools/` | diagnostics, **pas** des tests : non collectés |

## Lancer

```bash
pytest tests/ -q -m "not slow"          # tout, sauf les mesures longues
pytest tests/solver -q                  # un sous-système
pytest tests/ -q -m slow                # ordre de convergence, ~10 min
```

Les suites QAOA de `quantum/` prennent environ une heure : les lancer en
fond.

**Avant de lancer une campagne d'entraînement** — c'est-à-dire avant de louer
des cœurs :

```bash
python src/train_hyperparams.py --print-space              # l'espace réel
pytest tests/pipeline/test_train_hyperparams_contracts.py -q   # 60 tests, 14 s
pytest tests/pipeline/test_train_hyperparams_smoke.py -q       #  7 tests, 16 s
pytest tests/pipeline/test_v1_partial_pockets.py -q            # 18 tests,  2 s
```

Le deuxième fichier ne simule rien : vrai solveur, vrai circuit, vraie base
Optuna, vrai JSON de déploiement, sur les six scénarios — à N=32.

Le troisième vérifie les axes de configuration qu'aucun test ne traversait :
refus du mode matériel (D-48), sémantique du TTL, déterminisme du bras
`classical_only`, gardes Colab.

## Chemins

`conftest.py` résout `src/` et les paquets de `study/` depuis la racine du
dépôt, **quelle que soit la profondeur** du fichier de test. Ne pas
réintroduire de calcul par `dirname` répété : il casse au premier
déplacement, et souvent en silence.
