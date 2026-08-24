# Résultats actuels

## État de la revue

Le chemin de campagne a été consolidé autour d’une machine multi-cœur locale :
journal Optuna partagé, budget global, reprise contrôlée et sorties atomiques.
Les chemins HPC, bases distantes et anciens lanceurs V2 ont été retirés.

Les corrections scientifiques principales sont en place : conventions d’axes
du rotationnel et de la divergence, point X dans l’opérateur, coefficients
adimensionnels, warm starts, paramètres réellement explorés, seuils réglés hors
fold, budget apparié, graines QAOA explicites, exact diagonalisation avec
dégénérescence, panels complets et bootstrap par trajectoire.

La campagne finale n’a pas encore été exécutée. Il n’existe donc aucun résultat
numérique actuel autorisant une affirmation sur l’avantage quantique.

## Vérifications reproductibles

```bash
.venv/bin/python -m pytest tests -q -m "not slow"
.venv/bin/python -m pytest tests -q -m slow
bash scripts/repetition_campagne.sh
.venv/bin/python study/common/preflight_coefficients.py
.venv/bin/python study/pipeline/dns_sweep.py --dry-run
```

## Résultats à inscrire après campagne

1. nombre d’essais Optuna complets, élagués et échoués ;
2. matrice DNS/labels 8 × 4 × 5 et diagnostics physiques ;
3. résultats statiques LOSO et split bloqué ;
4. pour chaque fold : trois deltas physiques appariés, IC95 et p corrigée ;
5. bootstrap hiérarchique global ;
6. coût CPU, mémoire, taux de raffinement et sensibilité aux scénarios ;
7. hash des artefacts finaux et commande exacte.

Les anciennes mesures restent accessibles dans l’historique Git et
`docs/archive/`; elles ne sont pas recopiées ici.
