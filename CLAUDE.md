# BA_Proj — contexte de travail

Q-HAS étudie si une décision locale produite par un Hamiltonien d’Ising et
QAOA améliore l’AMR d’un solveur MHD 2-D. Le résultat négatif est recevable :
le dépôt doit mesurer un avantage, une équivalence ou un désavantage avec la
même rigueur.

## Sources de vérité

- `docs/protocol_v3_evaluation.md` : protocole gelé de la prochaine campagne ;
- `docs/MODE_EMPLOI_CAMPAGNE.md` : commandes d’exécution et de reprise ;
- `docs/DEFAUTS.md` : seuls les blocages actuels ;
- `docs/RESULTS.md` : résultats produits par le code actuel ;
- `docs/EVALUATION.md` : règles d’admissibilité pour le préprint ;
- `docs/PLAN_PREPRINT.md` : plan et formulations autorisées.

`docs/archive/` et les anciens artefacts de `results/` sont historiques. Ils
ne justifient aucun nombre du futur manuscrit.

## Chemin scientifique actuel

1. `src/` : solveur FD4/RK4, raffinement, mapping physique–Ising, QAOA et
   entraînement Optuna.
2. `study/pipeline/` : huit scénarios, quatre Reynolds, cinq graines
   physiques, DNS validées et labels.
3. `study/h2b_prediction/` et `study/h3_representation/` : plafonds,
   transfert LOSO, fuite par split et structure du Hamiltonien.
4. `study/closed_loop/` : comparaison confirmatoire appariée Q-HAS/classique.
5. `study/common/` : statistiques, provenance et agrégation.

## Invariants

- Aucun résultat scientifique depuis un arbre Git modifié.
- Aucun fichier manquant n’est ignoré silencieusement.
- Aucun réglage ne voit le scénario tenu ou les labels d’évaluation.
- L’unité d’inférence est la trajectoire physique, jamais le patch ou
  l’instantané.
- Les bras comparés partagent DNS, état initial, budget de raffinement et
  configuration du solveur ; seule la règle de décision diffère.
- Les graines physiques varient, la graine QAOA reste fixe dans l’analyse
  confirmatoire.
- Tout artefact porte le commit, les arguments et le contrat de campagne.
- Un artefact ancien n’est jamais repris sous un contrat différent.

## Vérification minimale

```bash
.venv/bin/python -m pytest tests -q -m "not slow"
.venv/bin/python -m pytest tests -q -m slow
bash scripts/repetition_campagne.sh
.venv/bin/python study/common/preflight_coefficients.py
```

Ne jamais qualifier `src` ou `study` de « parfait » sans ces quatre contrôles
sur le commit destiné à la campagne.
