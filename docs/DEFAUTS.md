# Défauts

**Où ça bloque.** Uniquement ce qui n'est pas résolu : ce qui empêche
d'avancer, comment on est tombé dessus, et où on en est pour le lever.

Ce qui est corrigé n'est **pas** ici — c'est un résultat, il vit dans
`RESULTS.md` avec sa mesure et sa commande de vérification.

## Reconstruction du 25 août — pourquoi ce fichier a une nouvelle histoire

Le commit `d3d7573` (« improvements and corrections of the tests, some
corrections on the src code ») a remplacé ce fichier — alors 2194 lignes,
17 entrées — par trois paragraphes génériques, sans rien déplacer vers
`RESULTS.md` ni vers `docs/archive/`. Le contenu perdu n'est reconstructible
qu'en lisant l'historique Git (`git show d047015:docs/DEFAUTS.md`), ce que
la discipline du dépôt ne prévoit pas : ces six documents sont censés
porter l'état **lisible**, pas un état qu'il faut fouiller.

Chacune des 17 entrées a été relue et **revérifiée contre le code présent
sur cette branche** (pas contre le texte d'origine). Le verdict :

| statut | nombre | quoi |
|---|---|---|
| toujours ouvertes, restaurées ci-dessous | **9** | D-22, D-39, D-50, D-98, D-100, D-158, D-187, D-188, D-189 |
| résolues entre-temps, non restaurées | 6 | D-41, D-48, D-135, D-141, D-143, D-186 — vérifiées aujourd'hui, mécanisme changé, plus de symptôme |
| décidées (limite assumée) | 1 | D-24 |
| déjà marquée corrigée avant la suppression | 1 | D-190 |

Les 5 « résolues entre-temps » n'ont **aucune trace écrite** de leur
correction — ni mesure avant/après, ni commande, ni date. C'est exactement
la dette que `CLAUDE.md` interdit : un résultat sans sa mesure. Quiconque
veut s'appuyer dessus doit remesurer avant de citer.

**Seconde passe, même jour** : la suite complète (`-m "not slow"`, 3102
tests) a été rejouée pour remesurer la couverture de `COUVERTURE.md`. Elle
n'est pas verte — **20 failed**, dont 19 préexistants sur cette branche
(diff exact contre une exécution capturée avant cette reconstruction) et 1
introduit par elle-même (corrigé, voir `COUVERTURE.md`). Sur les 19,
**7 étaient des tests devenus obsolètes** (chemin renommé, constante
non remise à jour après l'élargissement à 8 scénarios, canarie dont le
bloc mort a été nettoyé, import fantôme) — corrigés directement dans
`tests/`, aucun ne touchait `src/`. Les **12 restants dépassent la
correction mécanique** et entrent ici : D-191 (5 sites), D-192 (3 sites),
D-193 (1), D-194 (3). Deux des cinq sites de D-191
(`test_signal_contribution.py::test_C_ZZ`,
`test_qaoa_scaling_and_hparams.py::test_hyperparameter_sweep`) n'ont été
attribués à ce défaut qu'à une passe ultérieure, le 25 août également —
consignés ici pour ne pas laisser croire que le premier passage les avait
tranchés.

## Règle d'arrêt — ce qui entre dans ce fichier

Écrite parce que le taux de découverte a dépassé le taux de résolution.
Un défaut qui ne touche ni un nombre du papier ni un chemin déployé reste
noté ici tant qu'il n'est pas tranché — mais la barre pour y entrer est
haute : un rapport, pas une inquiétude.

---

## D-22 — les hyperparamètres déployés n'ont aucune provenance

**Ne se corrige pas par du code seul. Seule la campagne le règle.**

**Où ça bloque.** Réoptimiser demande de savoir d'où l'on part. Aucun
chiffre de performance n'est attribuable à un réglage dont on ignore
l'origine. Le JSON déployé (`best_hyperparams.json`) ne correspond à
**aucune** ligne des 13 CSV Optuna du dépôt — l'essai qu'il déclare a une
perte de 0,3213 dans la base contre 0,2215 annoncée, et aucun de ses
paramètres communs ne coïncide.

**État vérifié le 25 août.** `w_z_frac` reste borné à `[0.1, 1000.0]` (log)
dans `train_hyperparams.py --print-space` — la borne haute jamais tranchée
que D-22 signalait. Le mécanisme qui produit un JSON traçable existe
(`_save_results` écrit désormais le jeu complet, le hash du commit et
`sys.argv`), mais **le fichier actuellement déployé reste orphelin** : ce
mécanisme ne s'applique qu'à une campagne qui n'a pas encore tourné.

**Périmètre tranché.** 8 paramètres à réoptimiser : `beta`, `w_z_frac`,
`sigma`, `beta_curl`, `beta_xpoint`, `gamma_hydro`, `gamma_mag`, `kappa`.
`threshold_amr` reste gelé au meilleur essai classique.

```bash
python src/train_hyperparams.py --print-space
```

C'est un blocage de **campagne**, pas un défaut de code : il se ferme
quand la campagne tourne et écrit un JSON traçable, pas avant.

---

## D-39 — le check de tearing n'a plus de signal

**Où ça bloque.** `check_tearing` classe **harris_tearing comme `ok=False`
sur les 6/6 artefacts DNS** (Re 400/800/1200/1600, N 64/96/256). C'est le
seul des trois checks de validation phase 1b qui porte sur la reconnexion
magnétique.

**Vérifié le 25 août**, rejoué de bout en bout sur le code actuel :

```
Re1200_N96  ok=False  amplification=1.086
Re1600_N96  ok=False  amplification=1.103
Re400_N256  ok=False  amplification=1.000
Re400_N64   ok=False  amplification=1.016
Re400_N96   ok=False  amplification=1.000
Re800_N96   ok=False  amplification=1.077
```

Toujours le même symptôme qu'à la découverte : `mean_sq_current` moyenne
`J_z²` sur **tout le domaine**, y compris le courant d'équilibre de la
nappe (uniforme, non nul dès t=0), qui domine la moyenne spatiale et
n'évolue presque pas. Le module a été réécrit depuis (`mean_sq_current_fixed`
n'existe plus, remplacé par une seule `mean_sq_current`), mais la
composante fluctuante (retrait du fond stationnaire) n'a pas été branchée :
un test exploratoire antérieur avait montré qu'elle récupère un vrai signal
de reconnexion (amplification creux→fin de 8,3× à 17,6× sur les 6 fichiers)
mais ne suffit pas seule — la fenêtre temporelle ne referme jamais le pic.

**Deux corrections à composer, aucune appliquée** : (a) observable
fluctuante plutôt que moyenne pleine grille, **et** (b) revoir la fenêtre
`[0, t_max]` ou le critère de pic. Ne pas rebrancher sur les fonctions
gelées d'avant D-21 : ce serait réintroduire le défaut D3 pour faire
retomber `ok=True` sans corriger l'observable.

```bash
python3 -c "
import sys, glob, re, numpy as np
sys.path.insert(0, 'study/pipeline'); sys.path.insert(0, 'src')
import dns_validation as dv
for path in sorted(glob.glob('results/dns_harris_tearing_Re*_N*.npz')):
    res = dv.analyse_one(path)
    print(re.search(r'Re(\d+)_N(\d+)', path).group(0), dv.check_tearing(res))
"
```

---

## D-50 — le verdict imprimé de T11b bascule d'une exécution à l'autre

**Où ça bloque.** `h0_qaoa_displacement.main()` imprime l'une de deux
phrases opposées selon `abs(progress_moyen) < READING_THRESHOLD (0,1)` — un
seuil codé en dur appliqué à une grandeur mesurée non reproductible à cette
précision (dispersion inter-exécutions 0,018, contre 0,0146 de marge au
seuil).

**Vérifié le 25 août** — le mécanisme est identique à la découverte :

```python
READING_FLAT = "the circuit stays at the classical encoding; ..."
READING_MOVES = "the circuit moves substantially toward its own optimum."
READING_THRESHOLD = 0.1
def reading_message(prog_all):
    return READING_FLAT if abs(prog_all) < READING_THRESHOLD else READING_MOVES
```

(`study/h0_selection/h0_qaoa_displacement.py`, lignes 133-162). Toujours un
seul tirage comparé à un seuil ponctuel, aucune des trois options
envisagées (répéter et publier une dispersion ; changer pour une grandeur
déterministe comme `slope_paired` ; retirer la phrase) n'est appliquée.
Composé avec D-48 (résolu depuis, voir ci-dessous) : le renommage de
`classical_warm_start_params` en `constant_initial_params` clarifie ce que
le schedule fait, mais ne touche pas à la fragilité du seuil de lecture.

```bash
python study/h0_selection/h0_qaoa_displacement.py --N 256 --dim 2 --n-snaps 2
# a relancer plusieurs fois : la ligne READING n'est pas stable
```

---

## D-98 — le « contrôle négatif » de la figure 9 ne peut pas rendre de faux positif

**Où ça bloque.** `figures/v1_legacy/fig9_synthetic_unit_tests.py` construit
un motif de bruit uniforme annoncé comme contrôle négatif (« false positive
rate »), mais `pixel_prf` définit sa vérité terrain **relativement au champ
testé** (`gt > gt.mean()`) : le contrôle négatif déclare mécaniquement
~47 % du domaine « à raffiner », quelle que soit l'absence de structure.
Aucune valeur de `gt` ne peut y faire échouer le contrôle.

**Vérifié le 25 août** : le fichier et son test de déviation existent
toujours à l'identique, `pytest tests/study/test_fig9_negative_control.py`
passe (7 tests) — c'est-à-dire que la déviation reste correctement
**épinglée**, pas corrigée. Deux options non tranchées : seuil absolu
commun pour `needs`, ou retirer la 4ᵉ ligne de la figure.

```bash
pytest tests/study/test_fig9_negative_control.py
```

Aucune figure `results/figures/fig9_*` n'est committée dans ce dépôt :
aucun nombre publié n'en dépend.

---

## D-100 — le panneau « Uncertainty w(s) » de la figure 11 n'affiche pas le poids que le hamiltonien applique

**Où ça bloque.** `fig11_hamiltonian_design.py` recalcule le poids
d'incertitude à partir du score **par cellule** ; le mappeur réel
(`HamiltParams.py`) le calcule sur le score **moyenné par arête** et produit
**deux** champs (horizontal/vertical), pas un. Sur `harris_tearing`, l'écart
atteint +167 % (arêtes horizontales 4,3× plus actives que ce que le panneau
montre) — l'anisotropie que le hamiltonien voit n'apparaît pas du tout sur
un panneau unique.

**Vérifié le 25 août** : fichier et test de déviation inchangés,
`pytest tests/study/test_fig11_uncertainty_weight.py` passe. Choix de
présentation non tranché (afficher `w_h`, `w_v`, leur moyenne, ou les deux
cartes séparément).

```bash
pytest tests/study/test_fig11_uncertainty_weight.py
```

---

## D-158 — l'agrégateur de la table maîtresse n'est plus fiable, et c'est pire qu'à sa découverte

**Où ça bloque.** `python study/common/aggregate_master_table.py` est le
**quatrième test de recette** de `CLAUDE.md` : le test de non-régression du
dépôt. `CLAUDE.md` écrit : *« Un `MISSING` non nul, lui, est toujours une
régression. »*

**Défaut d'origine, non revérifié aujourd'hui** : donné une taille pour
laquelle aucune campagne n'a tourné, l'agrégateur trouvait `MISSING`,
écrivait quand même par-dessus les artefacts publiés, et sortait en code 0.

**Défaut nouveau, trouvé le 22 août, confirmé le 25 : le chemin canonique
— sans aucun argument — ne rend plus 180/176/4/0, il PLANTE.**

```
RuntimeError: results/t20_qhas_run_variance_kh.json uses an obsolete
schema without per-run budget matches
```

Les **4** artefacts `t20_qhas_run_variance_{kh,ot,rotor,tearing}.json` du
dépôt portent tous `schema=None`, alors que `closed_loop_headline_counts.py`
exige désormais `schema == 2`. **Conséquence directe : personne ne peut
aujourd'hui vérifier qu'un seul nombre publié n'a pas bougé**, puisque
l'outil qui les recalcule tous ne s'exécute plus jusqu'au bout. C'est plus
grave que le défaut d'origine (qui écrivait une table fausse en silence) :
celui-ci empêche même de savoir si la table actuelle est juste.

```bash
python study/common/aggregate_master_table.py
```

**Où on en est.** Ni l'ancien défaut (silent overwrite) ni le nouveau
(crash sur schéma obsolète) ne sont corrigés. Le second bloque tout : il
doit être traité avant de faire confiance à quoi que ce soit d'autre dans
`results/`.

---

## D-187 — trois tests stochastiques restent rouges par intermittence

**Rapport seul.** Même famille de défaut, même remède à trancher.

**Mise à jour du 25 août.** Le fichier `tests/quantum/test_qaoa_arm_is_sampled.py`
a été réécrit autour de contrats de graines explicites ; les deux tests de
stabilité de classement qui y vivaient (`test_the_ranking_is_nonetheless_
visibly_perturbed`, `test_the_ranking_survives_the_sampling`) ont disparu
avec l'ancienne version du fichier. **Non vérifié : si la réécriture a réglé
l'instabilité de fond (le bras QAOA échantillonné) ou si elle a simplement
retiré les deux tests qui la donnaient à voir.** Les trois tests restants
sont, eux, confirmés vivants (le premier a échoué dans la suite complète du
25 août) :

| test | fichier |
|---|---|
| `test_K_ZZZZ` | `tests/mapping/test_signal_contribution.py` |
| `test_C_ZZ` | `tests/mapping/test_signal_contribution.py` |
| `test_the_other_optimisers_spend_a_multiple_of_that_budget[L-BFGS-B]` | `tests/quantum/test_optimiser_axis.py` |

Aucun n'est lié au mappeur ni à un nombre publié — leur décompte dans une
exécution de la suite ne peut pas servir de comparaison d'une passe à
l'autre. **Trois options, aucune appliquée** : changer de grandeur (asserter
le coefficient déterministe sous-jacent plutôt que la moyenne d'un tirage) ;
fixer la graine ; ou consigner et ignorer explicitement ces identifiants
dans le décompte de la suite.

---

## D-188 — le critère d'acceptation de la tâche 6 (vérité terrain dynamique) est passé par un label redondant

**Rapport seul. Rien n'est corrigé** — changer l'horizon du protocole est
une décision de campagne.

**Où ça bloque.** Le protocole v3 §1.2 fixe `δt = 0,1` (« one hybrid step »)
et pose comme seul critère d'acceptation *« Spearman(d_i, e_i) > 0 »*.
Mesuré (N=96, `dim=8`, 5 instantanés/scénario, relu depuis les 8 artefacts
`d_patches_*.npz`) : **ρ ≥ 0,98 sur les quatre scénarios** à cet horizon —
le label dynamique est une renumérotation monotone du label statique, et le
critère du protocole le laisse passer sans rien détecter.

À `δt = 2,0` un seul scénario décolle (`orszag_tang`, ρ = 0,596, le seul
dont la perturbation **amplifie** au lieu de décroître) — cohérent avec le
fait que c'est aussi le seul scénario où le label statique n'était pas déjà
quasi gratuit (AUC du score classique seul : 0,592 contre 1,000/0,997/0,948).

**Ce qui bloque** : la tâche 7 du protocole prévoit de consommer `d_i(t+h)`
comme cible — à l'horizon prescrit, elle mesurerait deux fois la même chose
que `e_i(t+h)`. Toute tâche consommant `d_i` doit d'abord fixer son horizon
sur `t_x = 2π/(dim·(v+b)_rms)`, pas sur un nombre de pas hybrides.

```bash
pytest tests/study/test_dynamic_patch_labels.py -q -m slow
```

---

## D-189 — sous `norm="max"`, `EPS` sert de seuil physique et peut promouvoir la poussière numérique

**Rapport seul. Rien n'est corrigé** — le corpus n'entre pas dans la bande
dangereuse aujourd'hui, et choisir un plancher physique est une décision de
conception sur `src/`.

**Où ça bloque.** La plaquette (`HamiltParams_v2.py`, `norm="max"`) divise
chaque magnitude (vorticité, courant, point X) par **son propre** maximum.
Le seul garde est `EPS = 1e-10`, un garde de division par zéro, pas un
seuil physique : une vorticité de 1e-9 pèse alors **autant** qu'une
vorticité de 1,0 (marche mesurée : 0,000000 sous `EPS`, 0,999998 juste
au-dessus). C'est le revers exact de la correction du 21 août : ce qui
protégeait `legacy` de la poussière numérique était le dénominateur commun
qu'on a précisément retiré.

**Pourquoi ça ne bloque pas aujourd'hui** : balayage des 24 artefacts DNS
(480 instantanés) — aucun `max|ω|` ni `max|J|` ne tombe dans `(1e-10,
1e-6)`. Les valeurs sont soit exactement nulles, soit ≥ 4,9e-02.

**Épinglé par** `tests/mapping/test_plaquette_signal_negligeable.py` (5
tests) — la marche dans les deux modes, un balayage du corpus qui fait
rougir la suite si un futur artefact entre dans la bande, et la vérification
que les pics nuls du corpus sont exactement nuls.

```bash
pytest tests/mapping/test_plaquette_signal_negligeable.py -q -m slow
```

---

## D-191 — le bras QAOA est désormais déterministe PAR DÉFAUT partout, pas seulement où le protocole le documente

**Rapport seul. Décision requise, rien n'est corrigé.**

**Où ça bloque.** `VQARuntime.__init__` et `VQA/execute.execute()` prennent
désormais un paramètre `seed`, **par défaut `0`** — pas `None`, pas un
tirage. Ce défaut est threadé à `AerSimulator(seed_simulator=seed)`,
`estimator.options.simulator.seed_simulator`,
`sampler.options.simulator.seed_simulator` et
`seed_transpiler=self.seed`. Tous les appelants réels
(`src/pipeline.py:484`, `src/call_vqa_shell.py:93,102`) le lisent via
`getattr(argus, "seed", 0)` — **un seul `--seed` CLI pour toute une
campagne**, transmis identique à chaque appel QAOA, quel que soit le
patch, le pas de temps ou le scénario. Cinq tests dont l'unique raison
d'être est de mesurer la dispersion propre du bras QAOA (l'échantillonnage
shot-à-shot, indépendamment des conditions physiques) rougissent parce que
cette dispersion est tombée à zéro **exactement** :

| test | mesure | avant (attendu) | maintenant |
|---|---|---|---|
| `test_optimiser_axis.py::test_the_gap_between_the_two_optimisers_is_smaller_than_the_qaoa_spread` | écart intra-méthode sur 3 tirages | `> 0.0` | **`0.0` exactement** |
| `test_qaoa_noise_and_early.py::test_noise_robustness` | écart QAOA/classique sans bruit, Orszag-Tang | doit perdre de `> 0,09` | **`0,0000` exactement** — QAOA égale le classique au bit |
| `test_qaoa_physics_decision.py::test_the_vortex_contrast_is_not_reproducible_enough_to_conclude` | écart-type de 10 tirages répétés | `> 1e-6` | **`0,0` exactement** — les 10 valeurs sont identiques à la dernière décimale |
| `test_signal_contribution.py::test_C_ZZ` | 20 tirages, même hamiltonien `np.full(...)` (aucune source de bruit dans l'entrée) | écart-type non nul, **0,0270** mesuré et écrit dans la docstring | **écart-type `0,00000` exactement** — les 20 tirages sont un seul tirage recopié. La moyenne (`+0,03159`) franchit son seuil (`< 0,03`), mais n'est plus une moyenne de 20 échantillons : c'est un point unique, pas de sens statistique à en tirer |
| `test_qaoa_scaling_and_hparams.py::test_hyperparameter_sweep` | corrélation de rang QAOA/vérité sur 12 combinaisons `w_z_frac`×`threshold`, un seul tirage chacune | `min(rho) > 0.0` | `min(rho) = -0,467` — négatif sur 8 des 12 combinaisons. Peut-être un vrai effet du seuil ; ne peut pas être distingué d'un artefact de CE tirage précis sans en rejouer un second à une autre graine |

Les commentaires que ces tests portent encore affirment le contraire du
code actuel : « `aucune graine n'est fixée dans src/VQA/` » (les trois
premiers). Ce ne sont pas des faux positifs — c'est le contrat qui a
changé sous eux sans que le texte qui l'annonce ait suivi, exactement la
forme que ce dépôt appelle question 4. Les deux derniers ne portent pas
cette phrase, mais souffrent du même mécanisme : sans graine qui varie, un
« balayage » ou une moyenne sur *n* tirages n'en couvre plus qu'un seul —
`test_hyperparameter_sweep` ne peut aujourd'hui pas dire si le rho négatif
mesuré est un fait sur `w_z_frac`/`threshold` ou un accident du tirage à
`seed=0`.

**Ce n'est pas une régression accidentelle.** `docs/protocol_v3_evaluation.md:61-62` :
*« Trois graines physiques distinctes sont évaluées avec une graine QAOA
fixe »* — c'est le design délibéré du protocole confirmatoire
(`study/closed_loop/closed_loop_run_variance.py`, docstring : « the QAOA
seed remains fixed, so the statistical unit is the trajectory »), pour
isoler la variance due aux conditions physiques de celle due à
l'échantillonnage QAOA. `docs/RESULTS.md` liste « graines QAOA explicites »
parmi les corrections en place, et `docs/CODE_REVIEW.md` (§ « Décisions
appliquées ») : « graine QAOA fixe **pour isoler la variance physique** » —
trois sources concordantes, aucune ne dit que le défaut vaut hors de ce
cadre.

**Ce qui n'est écrit nulle part : que ce défaut vaut aussi pour TOUT le
reste du dépôt, pas seulement pour le protocole confirmatoire qui le
motive.** `docs/protocol_deviations.md` — le registre formel des écarts
autorisés — ne mentionne que l'amplitude de perturbation Kelvin-Helmholtz
et dit explicitement *« Aucun autre écart n'est autorisé à ce stade »*.
Un `--seed` par défaut à `0`, partagé par construction entre toutes les
questions qui consomment `execute()`/`VQARuntime` sans le surcharger
explicitement — les cinq tests ci-dessus, mais aussi potentiellement
`figures/`, `study/h0_selection/`, `study/h3_representation/`, tout ce que
`COUVERTURE.md` documente comme mesurant la dispersion QAOA — n'y figure
pas. La revendication centrale du dépôt (`COUVERTURE.md` § 4, « Le bras
QAOA n'est pas déterministe », dispersion 1,79e-1 à 3,61e-1) a été mesurée
sous l'ancien régime non-seedé ; personne n'a vérifié si elle tient encore
sous celui-ci pour les chemins qui ne sont pas le protocole confirmatoire.

**Où on en est.** Deux lectures possibles, aucune tranchée :

1. Le défaut par `seed=0` n'aurait dû s'appliquer qu'au chemin
   confirmatoire (`closed_loop_run_variance.py`, qui passe déjà
   `--qaoa-seed` explicitement) — tout le reste devrait recevoir une
   graine `None`/tirée par défaut, et les cinq tests ci-dessus sont
   corrects tels quels.
2. Le défaut déterministe est voulu partout pour la reproductibilité, et
   ce sont les cinq tests (et toute mesure de dispersion QAOA hors
   protocole confirmatoire) qui doivent désormais fixer explicitement des
   graines **distinctes** par tirage pour continuer à mesurer ce qu'ils
   prétendent mesurer.

Aucune des deux n'est appliquée ici : c'est un changement de comportement
de `src/`, non consigné dans `RESULTS.md` au-delà d'une ligne sans portée,
et il touche une affirmation déjà publiée.

```bash
git diff d047015..HEAD -- src/VQA/execute.py src/VQA/runtime.py
pytest tests/quantum/test_optimiser_axis.py::test_the_gap_between_the_two_optimisers_is_smaller_than_the_qaoa_spread \
       tests/mapping/test_signal_contribution.py::test_C_ZZ \
       tests/quantum/test_qaoa_scaling_and_hparams.py::test_hyperparameter_sweep -q
```

---

## D-192 — un nettoyage de commentaires a fait disparaître des mesures et des renvois D-NNN de `src/`, sondage seulement

**Rapport seul. Décision requise : restaurer la provenance perdue, ou
adapter les gardes qui la lisent.**

**Où ça bloque.** Le même commit `d3d7573` a raccourci de nombreux
commentaires de `src/` — comparé au français détaillé d'avant (mesures,
renvois `D-NNN`), plusieurs sont devenus une phrase anglaise sans nombre.
Deux sites trouvés par les tests qui les gardaient, pas par balayage
complet :

| fichier | avant (`d047015`) | maintenant | test qui le voit |
|---|---|---|---|
| `Simulation/solver.py`, `enforce_incompressibility` | docstring citant `rotationnelle`, `SPECTRALE`, et les deux mesures **4,63e-07** / **2,818e-14** qui ont motivé le choix | `"Project velocity; rotational induction preserves FD4 div(B)=0."` — aucun nombre | `test_solver_convergence.py::test_the_corrected_path_is_the_default_and_the_reason_is_written` |
| `Simulation/solver.py`, `PROJECT_RHS` | commentaire citant `step_full`, `step_layered`, `patch LOCAL`, `pas periodique` | `"RHS projection is valid only for global periodic fields. It remains disabled because layered patch updates are local and non-periodic."` — sens conservé, jeton `step_full` perdu | `test_solver_convergence.py::test_the_correction_is_off_by_default_and_the_reason_is_written` |
| `pipeline.py`, garde `_sigma_defaulted` | renvoi `D-22` dans le commentaire | renvoi disparu, comportement (le `RuntimeWarning`) intact | `test_solver_guards_and_objective.py::test_the_pipeline_warns_when_sigma_has_to_be_defaulted` |

**Ce que ça n'est PAS.** Aucune des trois valeurs/comportements gardés
n'a changé — `PROJECT_RHS`/`PROJECT_B` restent `False`, le `RuntimeWarning`
se lève toujours. Ce n'est pas un défaut de calcul.

**Ce que c'est.** Une perte de provenance : les deux mesures spectrales
(4,63e-07, 2,818e-14) qui justifiaient `enforce_incompressibility` ne sont
plus écrites nulle part dans le code qu'elles justifient, et le renvoi
`D-22` qui reliait le garde sigma à sa découverte a disparu. C'est
exactement la règle que `VIGIL.md`/ce dépôt appliquent partout ailleurs
(« un résultat sans sa mesure ») — retournée contre le dépôt lui-même.

**Portée non mesurée.** Deux sites trouvés parce qu'un test les gardait ;
combien d'autres commentaires de `src/` ont perdu un chiffre ou un
`D-NNN` sans qu'aucun test ne le lise n'est pas su — sondage, pas
balayage.

```bash
git diff d047015..HEAD -- src/Simulation/solver.py | grep -B3 -A1 "^-.*D-\|^-.*[0-9]e[+-][0-9]"
pytest tests/solver/test_solver_convergence.py -q -k "reason_is_written"
```

---

## D-193 — le résultat central de H0a n'existe dans aucun document vivant

**Rapport seul. Décision requise : où republier ce défaut.**

*(Ce titre évite volontairement d'écrire le numéro du défaut discuté :
`test_la_decision_de_ne_pas_corriger_D53_reste_ecrite` reconnaît une
entrée à son TITRE, et un renvoi dans la prose d'un autre défaut ne doit
pas compter comme l'entrée elle-même — exactement ce que D-146 a déjà
appris à ce dépôt. Le numéro est cité une fois, en corps de texte, dans
le paragraphe « Où ça bloque » ci-dessous.)*

**Où ça bloque.** `CLAUDE.md` — le document que toute session lit en
premier — écrit noir sur blanc : *« À dim = 3 [...] le QAOA atteint
l'optimum sur 0,062–0,156 des instantanés contre 1,000 exigé [...] Voir
D-53. »* C'est la mesure sur laquelle repose tout H0a. **`D-53` n'existe
dans aucun fichier lisible du dépôt** — ni `DEFAUTS.md` (vérifié : absent
aussi de la version pré-suppression, `d047015`, donc ce n'est pas un oubli
de cette reconstruction), ni `RESULTS.md` (39 lignes actuelles, aucune
mention).

**Comment on est tombé dessus.**
`tests/study/test_h0_certified_dim3_contradicts_criterion.py::test_la_decision_de_ne_pas_corriger_D53_reste_ecrite`
rougit : *« aucune ENTREE D-53 dans DEFAUTS.md ni RESULTS.md »*. Son propre
docstring explique pourquoi il cherche dans les deux : *« un défaut clos
SORT de `DEFAUTS.md` et entre dans `RESULTS.md` — c'est la règle des six
documents »*. D-53 est donc censé vivre dans `RESULTS.md`, pas dans
`DEFAUTS.md` — cohérent avec le fait qu'il n'était déjà plus dans
`DEFAUTS.md` avant la suppression du 24 août.

**Où c'est réellement passé.** `RESULTS.md` a subi la même suppression du
24 août que `DEFAUTS.md`/`COUVERTURE.md` (voir leurs sections de
reconstruction respectives) et n'a, à ce jour, **pas** été restauré —
seul un état courant à 39 lignes existe, sans historique. Le texte complet
de D-53 (mesure, seuils, méthode) reste lisible dans l'historique Git via
`git show d047015:docs/COUVERTURE.md`, section `h0_optimiser_equivalence.py`.

**Ce que ça coûte.** Le test lui-même le dit : sans une entrée D-53 quelque
part, *« le critère MIN_HIT=1.0 se relit comme valide à toute taille »* —
c'est-à-dire que rien n'empêche une lecture future de croire que le critère
d'acceptation de `h0_optimiser_equivalence.py` est toujours satisfait
partout, alors que la décision documentée de ne PAS le corriger à `dim=3`
(parce que le classement des solveurs n'y a jamais été exercé avant que
D-53 ne le fasse lever) a disparu du seul endroit où un lecteur la
trouverait.

**Où on en est.** Non corrigé : republier D-53 demande de restaurer
`RESULTS.md` (hors périmètre de cette passe, qui portait sur
`COUVERTURE.md`), pas seulement d'ajouter une ligne.

```bash
pytest tests/study/test_h0_certified_dim3_contradicts_criterion.py -q
git show d047015:docs/COUVERTURE.md | grep -n "D-53" 
```

---

## D-194 — le balayage des invocations de lanceurs a perdu plus de la moitié de sa surface, et personne n'a dit si c'est voulu

**Rapport seul. Décision requise : le plancher a-t-il raison de rester
haut, ou la perte est-elle acceptée ?**

**Où ça bloque.** `tests/test_launcher_paths_resolve.py` porte trois
planchers datés, chacun avec le hash du commit qui l'a mesuré
(`COUVERTURE.md` les cite déjà comme « le bon patron » : daté, marge
mesurée). Les trois rougissent aujourd'hui :

| plancher | mesuré à | aujourd'hui |
|---|---|---|
| `>= 45` | — | **35** |
| `>= 79` (`766d289`) | 79 | **35** |
| `>= 80` (`f8edebf`) | 80 | **35** |

**Ce que la mesure elle-même dit être correcte** : ces trois gardes
existent précisément pour détecter ce cas — *« le balayage a rétréci, il
ne prouve plus ce qu'il prouvait »*. Ils font leur travail.

**Cause identifiée, cohérente avec `COUVERTURE.md`** : six des neuf
scripts supprimés dans `d3d7573` (voir `COUVERTURE.md`, section
« Reconstruction ») étaient des `.sh` — `inventaire_campagne.py` et
`nettoyer_essais_fantomes.py` mis à part, `run_leak_free_campaign.sh`,
`run_study_v2_phases.sh`, `run_study_v2b.sh`, `soumettre_campagne.sh`
portaient chacun un nombre substantiel d'invocations (12 à 54 selon la
mesure du 18 août citée dans `COUVERTURE.md`). Leur suppression réduit
mécaniquement ce que le balayage peut voir.

**Ce qui n'est PAS établi** : que la surface perdue était redondante. Les
scripts supprimés ont-ils été consolidés dans les quatre nouveaux
(`run_confirmatory_campaign.sh`, `run_dns_campaign.sh`,
`run_rented_campaign.sh` plus les lanceurs existants), auquel cas les
mêmes invocations survivent sous une autre plume et les planchers doivent
simplement être abaissés à la nouvelle valeur mesurée — ou une partie du
comportement qu'ils enrobaient a disparu avec eux sans que rien ne le
remplace. Non tranché ici : abaisser un plancher sans le savoir serait
exactement la faute que ce fichier a déjà commise une fois (D-151) et que
`COUVERTURE.md` documente comme corrigée.

```bash
pytest tests/test_launcher_paths_resolve.py -q -k "sweep or exemption"
```

---

## Résolus depuis la dernière version de ce fichier — vérifiés le 25 août, non restaurés en tant que défauts

Ce qui suit n'est **pas** un blocage : c'est noté ici une seule fois, pour
qu'un nombre publié qui s'appuierait sur l'ancien comportement soit
signalé, puis ce paragraphe doit être retiré au prochain nettoyage.

- **D-41** (hamiltonien v1 identiquement nul sur harris/KH) — `E_patch.max()`
  n'est plus nul (2,76 sur harris, 1,27 sur KH, mesuré avec les mêmes
  paramètres et le même artefact que la découverte). Mécanisme du
  changement non tracé.
- **D-48** (le warm start ne lisait pas la décision classique) —
  `classical_warm_start_params` a été retiré ; `constant_initial_params`
  porte désormais un nom honnête. Le schedule reste constant (ce n'est pas
  l'option 3 — dériver réellement du score — qui a été choisie), mais la
  déception de nommage a disparu.
- **D-135** (deux chemins de score divergents dans `pipeline()`) — le
  chemin de divergence appelle désormais `instability_weight_map()`, la
  même fonction que `score()`.
- **D-141** (la porte de campagne franchie plus haut par la baseline que
  par le coefficient) — `relevance_is_sufficient` exige maintenant
  `rho_best - rho_classical > margin` : la comparaison relative que D-141
  réclamait comme option 2 est en place.
- **D-143** (référence DNS lue un cran trop tôt sur le chemin de
  divergence) — `dns_trace[step - 1]` n'existe plus dans `pipeline.py` ;
  les deux sites lisent `dns_trace[step]`.
- **D-186** (l'optimum du balayage `c_bias` tombait au bord de la grille) —
  **résolu avec soin.** `h2b_analytical_solution.py` porte désormais
  `c_bias_grid` par défaut sur `[0,1 ; 1e5]` (contre `[0,1 ; 100]` à la
  découverte) et une fonction `require_interior_optima` qui distingue un
  bord non résolu (lève `RuntimeError`, refuse de produire un artefact) d'un
  plateau **biais seul** authentique (`bias_only_limit=True`, `c_bias_
  identifiable=False`, exempté du refus). Remesuré le 25 août sur
  `harris_tearing` Re400 N96 dim4 : `at_right_edge=True`, `bias_only_limit=
  True`, F1 sature à **0,7405**, sous la baseline classique (0,830). Le
  fichier de test qui portait l'ancien nom a été réécrit (voir D-187) ; les
  52 configurations de D-86 n'ont toujours pas toutes été rejouées sous
  cette version, mais le mécanisme qui les rendrait lisibles existe.

**Aucun des cinq premiers n'a de mesure avant/après écrite quelque part** —
ni date, ni commande, ni chiffre publié dans `RESULTS.md`. C'est la dette
que la suppression du 24 août a créée : le prochain qui doit s'appuyer sur
l'un de ces faits doit d'abord le remesurer lui-même. D-186 fait exception :
sa mesure de confirmation est ci-dessus, avec sa commande.
