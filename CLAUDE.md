# BA_Proj — working context

Spécification complète : `docs/protocol_v3_evaluation.md`. Résultats publiés :
`docs/RESULTS_V3.md`, `docs/RESULTS_V4.md`, `docs/FINDINGS_V2.md`.

## Contexte

Q-HAS : cadre hybride quantique-classique d'AMR pour la MHD 2-D. Le dépôt
sépare l'**objet d'étude** (`src/`, gelé) du **travail de falsification**
(`study/`), organisé par hypothèse.

## Arborescence

```
src/                    V1 — solveur MHD (FD4/RK4), grille, raffinement,
                        PhysToAngle, HamiltParams, pile VQA/QAOA.
                        C'est l'objet étudié, pas une dépendance à améliorer.
tests/                  tests de src/ ; tests/v3/ et tests/v4/ testent study/
study/
  pipeline/             DNS → patches durs → coefficients → diagonalisation
                        (config.py, phase0–phase4, t8_dns_extension)
  h0_selection/         l'échec vient-il de l'optimiseur variationnel ?
                        (t11, t11b, phase5–phase7)          → RÉFUTÉ
  h1_solver/            les défauts numériques sont-ils secondaires ?
                        (t14)                                → NON TESTÉ
  h2b_prediction/       un modèle ML libre s'en sortirait-il ?
                        (phase10–phase12, t1, t1b, t4–t7)    → RÉFUTÉ
  h3_representation/    l'information des voisins est-elle inutile ?
                        (t12, t13, t17, t18, t26, t9, phase8) → ÉTABLI
  h4_transfer/          transfert sur conditions inédites
                        (t22, t22c, t22d, t25)               → CONJECTURE
  closed_loop/          campagne niveau 3 (t15, t15b, t15c, t19–t21, t23,
                        t24, level3_status)
  common/               provenance, statistiques, agrégateurs
                        (provenance.py, stats.py, metrics.py,
                        stats_confirmatory.py, t16_aggregate_v4.py)
results/                sorties .npz/.json de study/ ; results/figures/ pour
                        les images produites par figures/
figures/                code produisant les figures À PARTIR de results/
                        (figures/v1_legacy/ = figures de l'ère V1)
scripts/                lanceurs de campagne (.sh)
docs/                   protocole, résultats, critique d'évaluation
results/hyperparams/    entrées gelées : campagne Optuna (~1 semaine) et
                        best_hyperparams.json — voir son PROVENANCE.md,
                        seul dossier non reproductible par une commande
results/v1_runs/        sorties de l'ère V1
results/logs_v2/        journaux de la campagne V2 (lus par figures/)
```

## Garde-fous

- **`src/` est l'objet d'étude.** Toute modification est un changement de
  comportement scientifique : elle doit être justifiée, testée et consignée
  dans `docs/RESULTS_V4.md`. Elle n'est jamais faite « au passage ».
- **Un test qui ne peut pas échouer est un défaut.** Tout script de `study/`
  ou de `tests/` porte une assertion, et un balayage vide doit crier.
- Chaque tâche livre : code + un pytest sous `tests/v3/` ou `tests/v4/` +
  une entrée dans `docs/RESULTS_V4.md` (commande, hash git, nombres).
- Déterminisme : tout script accepte `--seed` et écrit le hash du commit et
  les arguments CLI complets dans ses `.npz`.
- Réutiliser avant de réécrire : importer `build_dataset`,
  `extract_features_2d`, `make_model`, `fit_eval`, `best_threshold_f1` depuis
  `study/h2b_prediction/phase11_upper_bound.py` ; importer le solveur, ne
  jamais réimplémenter la numérique.
- Commentaires en français acceptés ; identifiants de code en anglais.

## Tests de recette

```bash
python -m pytest tests/ --ignore=tests/v3 --ignore=tests/v4 -q   # V1
python -m pytest tests/v3 tests/v4 -q                            # study
python study/common/aggregate_master_table.py                          # 180 lignes,
                                                                 # 0 DIFF, 0 MISSING
```

Le troisième est le test de non-régression du dépôt : il recalcule chaque
nombre publié à partir de son artefact. S'il reste à 180 / 0 / 0 après un
déplacement de fichiers, le déplacement n'a rien cassé.
