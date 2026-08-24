# Couverture et contrôles

## Suites obligatoires

```bash
# Contrats logiciels et scientifiques rapides
.venv/bin/python -m pytest tests -q -m "not slow"

# Calculs numériques ou quantiques longs
.venv/bin/python -m pytest tests -q -m slow

# Syntaxe des lanceurs
bash -n scripts/run_rented_campaign.sh
bash -n scripts/run_confirmatory_campaign.sh
bash -n scripts/run_study_v3.sh
```

## Matrice couverte

- solveur : champs finis, énergie, divergence, convergence et scénarios ;
- mapping : axes, normalisation, bords, point X et familles Z/ZZ/ZZZZ ;
- VQA : Hamiltonien, optimiseur, graines, shots, warm start et post-traitement ;
- entraînement : espace de recherche, budget global, concurrence, reprise,
  contrat et candidat complet ;
- étude : panel complet, absence de fuite, exact, ablations, splits, bootstrap
  par trajectoire et agrégateurs ;
- lanceurs : chemins existants, refus d’un arbre modifié et sorties distinctes.

## Contrôles avant publication

- exécuter la suite entière dans l’environnement exact de la campagne ;
- conserver le résumé pytest et `pip freeze` avec les artefacts ;
- vérifier `git diff --check` et la syntaxe de tous les scripts ;
- exécuter les agrégateurs en mode strict ;
- reconstruire tableaux et figures uniquement depuis les nouveaux artefacts.

La couverture de ligne n’est pas une preuve scientifique. Les tests doivent
séparer des implémentations physiquement différentes et échouer sur une entrée
manquante ou dégénérée.
