# BA_Proj — working context

Six documents portent l'état du projet, avec des rôles disjoints. Les tenir
à jour fait partie de chaque tâche.

| document | contenu |
|---|---|
| `docs/PLAN_PREPRINT.md` | **la source mère** : objectif, hypothèses, ce qu'on peut prouver ou non |
| `docs/DEFAUTS.md` | où ça **bloque** — uniquement ce qui n'est pas résolu |
| `docs/COUVERTURE.md` | ce qui est **testé**, comment et pourquoi ; ce qui ne l'est pas |
| `docs/RESULTS.md` | ce qui est **accompli**, et la commande pour le refaire |
| `docs/EVALUATION.md` | ce qui, dans RESULTS, est **exploitable** — et ce qui ne l'est pas |
| `docs/CODE_REVIEW.md` | note de relecture |

**Les nombres des documents archivés sont obsolètes** : obtenus sur du code
dont on sait qu'il calculait autre chose que ce qu'il annonçait. Voir
`docs/archive/README.md`.

Spécification du protocole : `docs/protocol_v3_evaluation.md`. Campagnes
antérieures : `docs/archive/`.

## Contexte

Q-HAS : cadre hybride quantique-classique d'AMR pour la MHD 2-D. Le dépôt
sépare l'**objet d'étude** (`src/`, gelé) du **travail de falsification**
(`study/`), organisé par hypothèse.

## Arborescence

```
src/                    V1 — solveur MHD (FD4/RK4), grille, raffinement,
                        PhysToAngle, HamiltParams, pile VQA/QAOA.
                        C'est l'objet étudié, pas une dépendance à améliorer.
tests/                  tests de src/ ; tests/study/ et tests/study/ testent study/
study/
  pipeline/             DNS → patches durs → coefficients → diagonalisation
                        (config, dns_*, hard_patch_labels,
                         exact_diagonalisation, sanity_check…)
  h0_selection/         l'échec vient-il de l'optimiseur variationnel ?
                        (h0_optimiser_equivalence,
                         h0_qaoa_displacement)   → voir ci-dessous, NUANCÉ
  h1_solver/            les défauts numériques sont-ils secondaires ?
                        (h1_solver_convergence,
                         h1_curl_convention_gap)             → PARTIEL
  h2b_prediction/       un modèle ML libre s'en sortirait-il ?
                        (h2b_*, 19 scripts)                  → RÉFUTÉ
  h3_representation/    l'information des voisins est-elle inutile ?
                        (h3_*, 7 scripts)          → À REPRENDRE (D-58)
  h4_transfer/          transfert sur conditions inédites
                        (h4_*, 4 scripts)                    → CONJECTURE
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

## État des hypothèses — à lire avant de citer un verdict

**H0 n'est PAS réfutée sans qualificatif.** La réfutation publiée reposait
entièrement sur `dim = 2`, où l'état fondamental exact est le prédicteur
constant « tout raffiner » sur 40 instantanés sur 40 (D-45 / D-47) : tous
les solveurs y atteignent l'optimum parce qu'il n'y a rien à départager.
À `dim = 3` — 18 qubits, la seule taille à la fois **certifiée et non
dégénérée** jamais exécutée — le critère d'acceptation du module **lève** :
le QAOA atteint l'optimum sur **0,062–0,156** des instantanés contre 1,000
exigé, **sous** la règle classique dont il part (0,500). Voir D-53.

La formulation juste sépare les deux sous-questions :

| | verdict |
|---|---|
| **H0a** — l'optimiseur atteint-il l'optimum de son hamiltonien ? | **NON** à `dim = 3` |
| **H0b** — mieux l'atteindre aiderait-il ? | **NON** — ρ(E_gap, F1) = +0,870 : mieux résoudre H *dégrade* la décision |

Soit : **l'optimiseur échoue vraiment, et le réparer ne servirait à rien.**

**H3 est à reprendre.** D-58 a rétracté la lecture publiée « ZZ is
numerically dead on three of four classes » : la fenêtre conserve en fait
3–12 % de la masse ZZ en boucle ouverte et 34–59 % au réglage Level-3.
L'explication causale des ablations nulles de T13 et de la progression
quasi nulle de T11b **tombe avec elle** — ces deux résultats restent à
expliquer.

## Garde-fous

- **`src/` est l'objet d'étude.** Toute modification est un changement de
  comportement scientifique : elle doit être justifiée, testée et consignée
  dans `docs/RESULTS.md`. Elle n'est jamais faite « au passage ».
- **Un test qui ne peut pas échouer est un défaut.** Tout script de `study/`
  ou de `tests/` porte une assertion, et un balayage vide doit crier.
- Chaque tâche livre : code + un pytest dans le dossier de `tests/`
  correspondant au sous-système touché (voir `tests/README.md`) +
  une entrée dans `docs/RESULTS.md` (commande, hash git, nombres).
- Déterminisme : tout script accepte `--seed` et écrit le hash du commit et
  les arguments CLI complets dans ses `.npz`.
- Réutiliser avant de réécrire : importer `build_dataset`,
  `extract_features_2d`, `make_model`, `fit_eval`, `best_threshold_f1` depuis
  `study/h2b_prediction/h2b_ceiling_random_split.py` ; importer le solveur, ne
  jamais réimplémenter la numérique.
- Commentaires en français acceptés ; identifiants de code en anglais.

## Tests de recette

```bash
python -m pytest tests/ -q -m "not slow"        # tout, hors mesures longues
python -m pytest tests/solver -q                # un sous-système
python -m pytest tests/ -q -m slow              # ordre de convergence, ~10 min
python study/common/aggregate_master_table.py   # 180 lignes,
                                                # 164 OK / 16 DIFF / 0 MISSING
```

Le **quatrième** est le test de non-régression du dépôt : il recalcule
chaque nombre publié à partir de son artefact. Son état attendu n'est pas
`0 DIFF` : les **16 DIFF** sont les nombres déplacés par les corrections
de `src/`, à republier après la réoptimisation (voir `docs/RESULTS.md`).
Ce qui compte est que les trois compteurs ne bougent pas : s'il reste à
**180 / 164 / 16 / 0** après un déplacement de fichiers, le déplacement
n'a rien cassé. Un `MISSING` non nul, lui, est toujours une régression.
