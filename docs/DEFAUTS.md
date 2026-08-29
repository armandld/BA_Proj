# Défauts

**Où ça bloque.** Uniquement ce qui n'est pas résolu : ce qui empêche
d'avancer, comment on est tombé dessus, et où on en est pour le lever.

Ce qui est corrigé n'est **pas** ici — c'est un résultat, il vit dans
`RESULTS.md` avec sa mesure et sa commande de vérification.

---

## D-22 — la campagne à venir doit encore tourner

**Ne se corrige pas par du code seul. Seule la campagne le règle.**

Le mécanisme de provenance est en place : `_save_results` écrit un JSON
traçable (jeu complet de paramètres, hash du commit, `sys.argv`) et
`_deploy`, appelée automatiquement en fin de `--phase all`, le copie vers
le chemin exact que `pipeline.py`/`study/` lisent par défaut
(`RESULTS.md`, « le résultat d'une campagne ne rejoignait jamais
`study/` »). La sélection finale de la phase 3 est protégée par un
damier de validation tenu à l'écart (`HOLDOUT_GRID`, 6 régimes physiques,
sélection par perte moyenne) ; l'entraînement lui-même diversifie ses
régimes physiques par essai (`TRAINING_REGIME_GRID`, 4 régimes, coût par
essai inchangé). Rien ne manque au code.

**Périmètre.** 8 hyperparamètres à réoptimiser : `beta`, `w_z_frac`,
`sigma`, `beta_curl`, `beta_xpoint`, `gamma_hydro`, `gamma_mag`, `kappa`.
`threshold_amr` reste gelé au meilleur essai classique.

**Vérification minimale (`CLAUDE.md`) passée sur `c52c1de`** : `pytest
-m "not slow"` (3108 passed, 2 échecs connus — voir ci-dessous),
`pytest -m slow` (10 passed, 1 skipped), `scripts/repetition_campagne.sh`
(journal concurrent, budget global, reprise — vert), `study/common/
preflight_coefficients.py` (5 contrôles physiques, « campagne
possible »). Les 2 échecs restants, `test_hyperparameter_sweep` et
`test_noise_robustness`, sont rouges par construction depuis D-195 : ils
épinglent des valeurs (`min(rho)=-0,467` ; écart nul sur Orszag-Tang)
qui SONT la preuve de H0a, pas une régression — confirmé en rejouant les
deux tests, identiques à la décimale, sur le commit d'avant cette passe
de synthèse de commentaires. Aucune autre ligne rouge à ce jour.

```bash
python src/train_hyperparams.py --print-space   # verifie l'espace, ne calcule rien
python src/train_hyperparams.py --phase all --seed <graine>   # la campagne elle-meme
```

C'est un blocage de **campagne** (plusieurs jours de calcul), pas un
défaut de code : il se ferme quand la campagne tourne, pas avant.

---

## D-197 — la campagne LOSO du niveau 3 (H4) n'a que 4 des 8 folds requis

H4 (transfert sur conditions inédites) répond exclusivement sur
`study/closed_loop/` — structurellement **séparé** de
`results/hyperparams/best_hyperparams.json` : `closed_loop_campaign.py`
mène sa propre recherche Optuna LOSO par fold et n'importe jamais le JSON
de la campagne D-22. Ce n'est pas un défaut de câblage — corriger D-22 ne
fait rien pour H4.

**Les artefacts réels sont périmés.** Seuls 4 des 8 folds attendus
existent (`kh`, `ot`, `rotor`, `tearing` ; manquent `vortex`,
`coalescence`, `double_tearing`, `magnetic_twist` — `FOLD_KEYS` dans
`study/pipeline/config.py`), à `n_trials=4` (échelle fumée, pas les
170 essais/fold requis) et sans `campaign_contract_sha256`.
`closed_loop_campaign.py` lève `RuntimeError` si ce champ est absent ou
ne correspond pas au contrat courant : ces 4 folds ne peuvent pas être
complétés par une reprise incrémentale, les 8 doivent être (re)joués sous
le contrat actuel.

**Conséquence sur la table maître.** `t15c | folds completed = 4/8`,
`budget-matched folds = 4/8`, et les 3 lignes de verdict (`folds where
Q-HAS better`, `Pareto-dominated`, `mean delta phys`) sont `MISSING`
plutôt que d'afficher un nombre non représentatif. Le protocole L3 exige
`>= 3/4` folds gagnants pour trancher — avec 4/8, la règle de décision
pré-enregistrée ne peut être appliquée sans biais de sélection sur QUELS
folds ont tourné.

**Coût, même ordre que D-22.** Compléter les 4 folds manquants demande
~170 essais Optuna chacun — un ordre de grandeur d'heures par fold, du
même ordre que la campagne d'hyperparamètres mais structurellement
distinct d'elle. À lancer sur la même machine, avec elle.

```bash
ls results/t15_level3_fold_*.json                 # 4 presents, 4 manquants
python study/common/aggregate_master_table.py --allow-missing | grep t15c
```

---

## D-198 — le plafond GBT sous LOSO (H2b) est saboté par un signe qui s'inverse d'un scénario à l'autre

**Sans conséquence sur le verdict H2b** (qui ne repose pas sur cette
seule comparaison, voir les 19 scripts de `study/h2b_prediction/`), mais
non résolu : la comparaison T5 spécifiquement ne doit pas être citée
comme preuve que la physique bat le ML.

`score_classical` est littéralement la feature n°0 des 9 du GBT
(`np.allclose(X_site[:,0], S)`, vrai sur les 4 scénarios). Sur le fold
`mhd_rotor` tenu (celui qui porte presque tout l'avantage classique), un
seuil brut sur cette seule feature fait 0,636 de F1 ; le GBT sur la MÊME
feature seule fait 0,163. Ce n'est pas un excès de features qui nuit,
c'est le mécanisme d'apprentissage.

**La cause : le signe de la relation score→label s'inverse par
scénario.** Moyenne du score classique par classe :

| scénario | classe positive | classe négative |
|---|---|---|
| harris_tearing | 0,677 | 0,649 (quasi égal) |
| kelvin_helmholtz | 0,732 | **0,740 (inversé)** |
| mhd_rotor | 0,647 | 0,057 (séparation nette) |
| orszag_tang | 0,381 | **0,485 (inversé)** |

Sur 3 scénarios sur 4, un score plus haut ne prédit pas mieux « à
raffiner » — sur deux, c'est inversé. Un GBT qui **apprend** cette
relation sur 3 pools d'entraînement LOSO la transfère mal au 4ᵉ, où elle
est forte et positive.

**Une cause contributive corrigée.** `make_model("gbt", seed)` utilisait
`early_stopping="auto"`, qui ne se déclenche que si `n_samples > 10000` —
jamais sur un fold LOSO réel (`n_train=1280`). `early_stopping=True`
explicite (10 % de validation interne, L2=1,0) est maintenant l'option
utilisée par `h2b_loso_transfer.py` ; effet mesuré modeste (`f1_site`
moyen 0,278), `mhd_rotor` reste catastrophique (0,013).

**La cause dominante — testée, pas réparée.** Normaliser chaque scénario
par ses propres statistiques avant le pool LOSO
(`normalise_per_scenario()`, `--normalize-per-scenario`) **aggrave** le
plafond moyen :

| tenu à l'écart | F1_site (brut) | F1_site (normalisé/scénario) |
|---|---|---|
| harris_tearing | 0,404 | **0,000** |
| mhd_rotor | 0,013 | **0,483** |
| **moyenne** | **0,278** | **0,165** |

Ça répare `mhd_rotor` mais casse `harris_tearing` : l'échelle absolue du
score porte elle-même du signal réel selon le scénario, que la
normalisation efface avec le bruit. Deux pistes restent non essayées :
modèle non-monotone, calibration du seuil de label.

```bash
python study/h2b_prediction/h2b_loso_transfer.py --dim 4
python study/h2b_prediction/h2b_loso_transfer.py --dim 4 --normalize-per-scenario
```
