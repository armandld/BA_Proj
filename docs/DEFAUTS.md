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

**D-158, D-98, D-100, D-50, D-39, D-187 et D-189, sept des 9 « toujours
ouvertes » ci-dessus, ont eux aussi été refermés — six le 25 août, D-189
le 26 après une correction d'erreur.** D-158 : la cause exacte du plantage était une exception non
attrapée à un seul site d'appel (`aggregate_master_table.py::rows_t23`),
corrigée sans toucher au contrat de la fonction levée ; la table maîtresse
tourne désormais jusqu'au bout (268 lignes, 142 OK / 6 DIFF / 120 MISSING
— les MISSING sont attendus tant que la campagne confirmatoire élargie à
8 scénarios n'a pas tourné, pas une régression). D-98 : le contrôle
négatif de fig9 utilise désormais une fraction de pixels marqués, sans
référence relative au champ, au lieu d'un P/R/F1 qui ne pouvait pas
échouer. D-100 : fig11 affiche désormais les deux poids d'incertitude par
arête (h et v) que le mappeur calcule réellement, au lieu d'un panneau
unique par cellule qui masquait leur anisotropie. D-50 : décision USER —
le verdict imprimé de T11b lit désormais `slope_paired` (une pente déjà
calculée) plutôt que `prog_all` (un tirage unique mesuré instable :
0,1034/0,0850/0,0859 sur trois exécutions identiques, deux conclusions
opposées). D-39 : `check_tearing` lit désormais une observable fluctuante
(`J2_fluct`, fond homogène-en-x retiré) et accepte un pic encore montant
en fin de fenêtre simulée si l'amplification dépasse le seuil `grows` —
les 6/6 fichiers DNS `harris_tearing` réels passent maintenant `ok=True`
(8,1×–17,5×), contre `ok=False` sur les 6 avant (1,0×–1,1×, uniquement le
courant d'équilibre de la nappe, qui ne referme jamais son pic dans la
fenêtre `[0, t_max]`). D-187 : les trois tests qu'il listait passent tous,
rejoués plusieurs fois chacun (3/3, 3/3, 4/4) — voir `docs/RESULTS.md`
pour la réserve sur la taille de cet échantillon. Voir `docs/RESULTS.md`
pour les six mesures complètes.

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
tranchés. **D-192, D-193 et D-194, eux, ont été refermés le même jour** —
D-193 par la restauration de `RESULTS.md` qui manquait précisément pour le
lever, D-192 par un balayage complet qui a remplacé les 3 sites du sondage
par 37 sites réels restaurés, D-194 en trouvant que la « perte » de
couverture était un parseur périmé (variable `$PYTHON_BIN` non reconnue)
et non une régression réelle (voir la liste des résolus, plus bas, et
`docs/RESULTS.md`). **D-191, lui, est maintenant décidé, implémenté et
intégralement vérifié** : les 5 sites cassés ont tous été rejoués — 3
confirment la dispersion restaurée (`test_C_ZZ`,
`test_the_gap_between_the_two_optimisers_is_smaller_than_the_qaoa_spread`,
`TestFullPipelineVortex::test_the_vortex_contrast_is_not_reproducible_
enough_to_conclude`), 2 (`test_hyperparameter_sweep`,
`test_noise_robustness`) restent rouges mais rendent une valeur identique
à la décimale près sous deux tirages indépendants confirmés aléatoires —
la preuve que ce n'est plus D-191 (dispersion nulle) qui les fait échouer.
Ces deux-là rouvrent sous **D-195**, cause distincte non élucidée. Il ne
reste donc plus **aucune** entrée ouverte directement issue de cette
seconde passe : les 5 sites de D-191 sont clos, D-195 est neuf.

## D-198 — le plafond GBT sous LOSO est saboté par un signe qui s'inverse d'un scénario à l'autre, pas par la physique

**Trouvé en réagissant à un doute USER** (« cette histoire de F1 reste
étrange ») sur T5 (`docs/RESULTS.md`) : le score classique domine
largement (F1 0,52–0,55) le plafond GBT (0,29–0,32) sous LOSO apparié.
Le titre est correct comme mesure ; l'interprétation « la physique bat le
ML » ne l'est pas.

**Vérifié directement, pas supposé.** `score_classical` est littéralement
la feature n°0 des 9 du GBT (`FEATURE_NAMES[0]`,
`h2b_ceiling_random_split.py`) — vérifié `np.allclose(X_site[:,0], S)`
sur les 4 scénarios, vrai partout. GBT devrait donc au pire égaler un
simple seuil sur cette feature. Sur le fold `mhd_rotor` tenu (celui qui
porte presque tout l'avantage classique) :

| méthode | F1 |
|---|---|
| seuil brut sur le score classique | 0,636 |
| GBT (HistGradientBoosting), les 9 features | 0,005 |
| GBT, **la même feature seule** | 0,163 |

Même restreint à LA MÊME feature que le seuil, GBT reste ~4× pire. Ce
n'est pas un excès de features qui nuit — c'est le mécanisme
d'apprentissage.

**La cause, mesurée** : la moyenne du score classique par classe,
scénario par scénario (`--re 400 --N 256 --dim 4`, 30 instantanés/config) :

| scénario | moyenne, classe positive | moyenne, classe négative |
|---|---|---|
| harris_tearing | 0,677 | 0,649 (quasi égal) |
| kelvin_helmholtz | 0,732 | **0,740 (inversé)** |
| mhd_rotor | 0,647 | 0,057 (séparation nette) |
| orszag_tang | 0,381 | **0,485 (inversé)** |

Sur 3 scénarios sur 4, un score classique plus haut ne prédit **pas**
mieux « à raffiner » — sur deux, la relation est inversée. Seul
`mhd_rotor` sépare franchement les deux classes. Un seuil brut,
peu sensible à un entraînement bruité, transfère quand même
raisonnablement sur ce dernier. Un GBT, qui **apprend** la relation
score→probabilité sur les scénarios d'entraînement, apprend une relation
plate ou inversée sur 3 des 4 pools d'entraînement LOSO et la transfère
mal au 4ᵉ, où la vraie relation est forte et positive.

**Conséquence pour H2b** : le verdict RÉFUTÉ ne repose pas sur cette seule
comparaison (voir les 19 scripts de `study/h2b_prediction/`), donc il
n'est pas remis en cause en bloc. Mais **la comparaison T5 spécifiquement
ne doit pas être citée comme preuve que la physique bat le ML** — elle
mesure surtout que ce GBT particulier généralise mal à travers un
changement de signe scénario-à-scénario, pas que l'information n'est pas
apprenable.

**Une cause contributive corrigée, la dominante non.** USER (26 août) :
« il faut être absolument sûrs qu'ils ne surapprennent pas ». Vérifié :
`make_model("gbt", seed)` utilisait `early_stopping="auto"` (le défaut
sklearn, jamais explicité) — qui ne se déclenche que si `n_samples >
10000`. Sur un fold LOSO réel (`n_train=1280`), `do_early_stopping_`
reste **Faux** et le modèle va au bout de ses 300 itérations avec
`l2_regularization=0.0` : aucune protection réelle, quel que soit le nom
du paramètre. `make_model` accepte maintenant `early_stopping` (défaut
`"auto"`, bit-à-bit inchangé pour les 11 autres consommateurs de la
fonction) ; `h2b_loso_transfer.py` passe désormais `early_stopping=True`
(10 % de validation interne, 10 itérations sans progrès, L2=1,0).

**Effet mesuré, modeste** : `upper_bound_loso_N256_dim4.npz` régénéré —
`f1_site` moyen 0,278 (était 0,316), `f1_sten` 0,307 (était 0,287) ;
`f1_class` moyen 0,403 (était 0,465 — variation qui ne vient PAS de ce
correctif, `f1_class` ne passe jamais par `make_model` ; l'ancien
artefact datait d'une version antérieure du dépôt, provenance non
tracée). Sur `mhd_rotor` spécifiquement, `f1_site` reste catastrophique
(0,013) : la régularisation ne répare pas la panne dominante décrite
ci-dessus. **La normalisation par scénario, le modèle non-monotone, et
la calibration du seuil de label restent tous les trois non corrigés** —
trancher la cause du changement de signe demanderait l'un des trois.

```bash
python -c "
import sys, os
sys.path.insert(0, 'src')
for d in ('pipeline','h2b_prediction','common'):
    sys.path.insert(0, os.path.join('study', d))
from h2b_loso_transfer import _gather_scenario
from config import RESULTS_DIR
for sc in ('harris_tearing','kelvin_helmholtz','mhd_rotor','orszag_tang'):
    dp = os.path.join(RESULTS_DIR, f'dns_{sc}_Re400_N256.npz')
    pp = os.path.join(RESULTS_DIR, f'patches_{sc}_Re400_N256_dim4.npz')
    _, _, Y, S = _gather_scenario([(400, dp, pp)], 4, 30)
    print(sc, S[Y==1].mean(), S[Y==0].mean())
"
```

## D-197 — H4 n'est répondable qu'à moitié : la campagne LOSO niveau 3 n'a que 4 des 8 folds

**Trouvé en auditant si H1/H3/H4 sont bien implémentées dans `study/`
contre la sortie de la campagne** (26 août, sur demande USER). H4
(transfert sur conditions inédites) répond exclusivement sur
`study/closed_loop/` — structurellement **séparé** de
`results/hyperparams/best_hyperparams.json` : `closed_loop_campaign.py`
mène sa propre recherche Optuna LOSO par fold (voir
`test_the_closed_loop_covers_every_key_the_pipeline_reads`,
`docs/RESULTS.md`) et n'importe jamais le JSON de la campagne D-22. Ce
n'est donc pas un défaut de câblage — corriger D-22 ne fait rien pour H4.

**Les artefacts réels sont périmés.** Seuls 4 des 8 folds attendus
existent (`kh`, `ot`, `rotor`, `tearing` ; manquent `vortex`,
`coalescence`, `double_tearing`, `magnetic_twist` — `FOLD_KEYS` dans
`study/pipeline/config.py`), et les 4 présents datent de `17d983d`
(« reorganise the repository around the hypotheses », une réorganisation
de fichiers, pas une campagne) : `n_trials=4` (échelle fumée, pas les
170 essais/fold requis) et aucun `campaign_contract_sha256`. Le code actuel
de `closed_loop_campaign.py` (lignes 275–278, 509–510) **lève
`RuntimeError`** si ce champ est absent ou ne correspond pas au contrat
courant — ces 4 folds ne peuvent donc pas être complétés par une reprise
incrémentale : les 8 doivent être (re)joués sous le contrat actuel.

**Conséquence directe sur `docs/v4_master_table.csv`** (voir D-196) :
`t15c | folds completed = 4/8`, `budget-matched folds = 4/8`, et les 3
lignes de verdict (`folds where Q-HAS better`, `Pareto-dominated`, `mean
delta phys`) sont désormais `MISSING` plutôt que d'afficher un nombre non
représentatif. Le protocole L3 exige `>= 3/4` folds gagnants pour trancher
— avec 4/8 (dont la moitié seulement dans l'échantillon actuel), la règle
de décision pré-enregistrée ne peut être appliquée sans un biais de
sélection sur QUELS folds ont tourné.

**Pas corrigé ici** : compléter les 4 folds manquants demande de relancer
`closed_loop_campaign.py` pour chacun — un ordre de grandeur d'heures par
fold (170 essais Optuna), du même ordre que la campagne D-22 mais
structurellement distinct d'elle. Décision USER nécessaire avant de
lancer : ce n'est pas un correctif de code autonome.

```bash
ls results/t15_level3_fold_*.json                 # 4 presents, 4 manquants
python -c "import json; d=json.load(open('results/t15_level3_fold_kh.json')); \
print(d['n_trials'], 'campaign_contract_sha256' in d)"   # 4 False
python study/common/aggregate_master_table.py --allow-missing | grep t15c
```

## Règle d'arrêt — ce qui entre dans ce fichier

Écrite parce que le taux de découverte a dépassé le taux de résolution.
Un défaut qui ne touche ni un nombre du papier ni un chemin déployé reste
noté ici tant qu'il n'est pas tranché — mais la barre pour y entrer est
haute : un rapport, pas une inquiétude.

---

## D-22 — la campagne à venir doit encore tourner

**Ne se corrige pas par du code seul. Seule la campagne le règle.**

**Décision USER du 26 août : la provenance du fichier ACTUELLEMENT
déployé n'est plus la question.** Les hyperparamètres vont être
réentraînés — savoir d'où vient l'ancien essai (perte 0,3213 dans la base
contre 0,2215 annoncée, aucun paramètre commun) ne change plus rien à ce
qu'il faut faire. Ce qui compte : que le résultat de la **prochaine**
campagne soit conservé et effectivement utilisé par `study/` et par le
papier. Ce dernier point est **réglé** (voir `docs/RESULTS.md`,
« D-22 — le résultat d'une campagne ne rejoignait jamais `study/` ») —
`_save_results` écrit un JSON traçable (jeu complet, hash du commit,
`sys.argv`) et `_deploy`, appelée automatiquement en fin de `--phase all`,
le copie vers le chemin exact que `pipeline.py`/`study/` lisent par
défaut. Ce que ce défaut recouvre encore, c'est uniquement le fait que la
campagne **n'a pas tourné** — pas un manque de code.

**Périmètre tranché.** 8 paramètres à réoptimiser : `beta`, `w_z_frac`,
`sigma`, `beta_curl`, `beta_xpoint`, `gamma_hydro`, `gamma_mag`, `kappa`.
`threshold_amr` reste gelé au meilleur essai classique. `w_z_frac` reste
borné à `[0.1, 1000.0]` (log) dans `train_hyperparams.py --print-space` —
vérifié le 25 août, la borne haute jamais tranchée que D-22 signalait à
l'origine.

```bash
python src/train_hyperparams.py --print-space   # verifie l'espace, ne calcule rien
python src/train_hyperparams.py --phase all --seed <graine>   # la campagne elle-meme
```

C'est un blocage de **campagne** (jours de calcul), pas un défaut de
code : il se ferme quand la campagne tourne, pas avant.

---

## D-188 — la tâche 7 doit lire `t_x` par scénario, pas un horizon unique

**Décidé et mesuré le 26 août** (`docs/RESULTS.md`, « D-188 — vérité
terrain dynamique remesurée à l'horizon `t_x`, verdict mixte »). L'horizon
`δt = 0,1` du protocole v3 §1.2 rendait `ρ(d_i, e_i) ≥ 0,98` sur les
quatre scénarios — un label dynamique redondant avec le statique. Les 8
artefacts `d_patches_*.npz` déjà présents n'utilisaient que `δt = 0,1` et
`δt = 2,0` (ablations explicites) ; l'horizon par défaut du script
lui-même (`t_x`, déjà implémenté, jamais utilisé) n'avait jamais tourné.

**Régénéré aux 4 scénarios canoniques, N=96, dim=8, 5 instantanés, à
`t_x`** (0,41 à 0,88 selon le scénario — 4 à 9× `δt = 0,1`) :

| scénario | ρ(d, e) à `t_x` |
|---|---|
| harris_tearing | 0,97–1,00 — reste redondant |
| kelvin_helmholtz | 0,995–0,998 — reste redondant |
| mhd_rotor | 0,82–1,00 — 1 instantané sous le seuil de redondance (0,95) |
| orszag_tang | 0,66–0,97 — 3 des 5 instantanés sous le seuil |

**Verdict mixte, pas un renversement.** Corriger l'horizon expose un vrai
signal sur `mhd_rotor` et `orszag_tang` (2 scénarios sur 4), surtout à
l'instantané le plus précoce, mais `harris_tearing`/`kelvin_helmholtz`
restent essentiellement une renumérotation du label statique même à
l'horizon physique. **Toute tâche future consommant `d_i` (tâche 7) doit
fixer son horizon sur `t_x` — confirmé nécessaire — mais ne doit pas
présumer que le label devient informatif partout : il ne l'est, par le
classement, que sur la moitié des scénarios canoniques.**

```bash
python study/pipeline/dynamic_patch_labels.py --scenario <s> --re 400 --N 96 \
    --dim 8 --snaps 5 --seed 0 --allow-redundant   # les 4 scenarios
pytest tests/study/test_dynamic_patch_labels.py -q -m "not slow"   # 25 passed
```

---

## D-195 — `test_noise_robustness` (Orszag-Tang) : deux causes éliminées, pas de troisième confirmée

**`test_hyperparameter_sweep` (MHD Rotor) est refermé le 26 août**
(`docs/RESULTS.md`, « D-195 — une moitié expliquée et confirmée ») : ce
n'était pas un défaut distinct — augmenter `K_opt` de 80 (déployé) à 800
sur la même cellule fait passer `captured` de 0,1769 à 0,6033 (correct),
sans toucher au Hamiltonien. C'est H0a (D-53 : l'optimiseur variationnel
n'atteint pas l'optimum de son propre Hamiltonien), observé une troisième
fois et confirmé par la même méthode (rejouer à budget plus grand).

**`test_qaoa_noise_and_early.py::test_noise_robustness` reste ouvert.**
QAOA égale exactement le classique sur Orszag-Tang sans bruit
(`frac_qa=frac_cl=0,3189`), stable sous deux tirages indépendants
confirmés aléatoires. Deux causes plausibles, **éliminées par la
mesure** :
1. Budget d'optimiseur insuffisant — `K_opt=80` contre `800` : `captured`
   reste exactement `0,3189` aux deux budgets (contrairement au rotor).
2. Fenêtre d'incertitude (mécanisme D-58/T17) qui noierait le couplage
   ZZ/ZZZZ — calculée directement sur les scores classiques de cette
   configuration (`threshold=0,3`, `σ=0,05`) : `0,68`–`0,93` sur 6 des 9
   cellules, pas près de zéro.

**Piste restante, non vérifiée** : le score QAOA continu suit presque
cellule à cellule le score classique (écarts 0,01–0,05), cohérent avec un
optimum du Hamiltonien qui coïncide réellement avec la décision classique
sur cette configuration précise — pas une panne de l'optimiseur ni un
couplage inerte, juste un cas où la physique ne change rien. Trancher
demanderait une énumération exhaustive du Hamiltonien de cette
configuration (méthode de D-53), pas faite ici.

```bash
pytest tests/quantum/test_qaoa_noise_and_early.py::test_noise_robustness -q
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
- **D-192** (nettoyage de commentaires ayant fait disparaître des mesures
  et des renvois D-NNN de `src/`, connu à 3 sites par sondage) — un
  balayage complet du diff `d047015..HEAD` sur `src/` (pas seulement le
  sondage d'origine) a trouvé **37 sites** distincts sur 8 fichiers, la
  plupart avec un renvoi `D-NNN` et une mesure chiffrée disparus. Restaurés
  verbatim, sauf 4 sites vérifiés comme n'étant PAS des pertes (le code
  qu'ils décrivaient a réellement changé — restaurer le texte l'aurait
  rendu faux, pas seulement muet ; voir `docs/RESULTS.md`, entrée du
  25 août, pour la liste exacte et pourquoi). Mesure de confirmation
  ci-dessous.
- **D-193** (le résultat central de H0a, D-53, n'existait dans aucun
  document vivant) — résolu par la restauration de `docs/RESULTS.md` le
  25 août : les trois sections `# D-53 — …` sont de retour, avec leur
  mesure complète. `pytest tests/study/test_h0_certified_dim3_contradicts_criterion.py`
  passe (7 tests), y compris `test_la_decision_de_ne_pas_corriger_D53_reste_ecrite`
  qui avait détecté l'absence. Contrairement aux cinq entrées ci-dessous,
  celui-ci porte sa mesure de confirmation ci-dessus (commande, résultat) —
  pas seulement une affirmation.
- **D-194** (le balayage des invocations de lanceurs était tombé à 35,
  contre 79-80 mesurés sur l'ancien jeu de scripts — perte réelle ou
  parseur périmé, non tranché) — **parseur périmé, pas perte.** Les
  lanceurs actuels (`run_study_v3.sh`, `run_reoptimisation.sh`,
  `run_fold.sh`, et les 3 nouveaux `run_confirmatory_campaign.sh`/
  `run_dns_campaign.sh`/`run_rented_campaign.sh`) résolvent tous leur
  interpréteur via une variable `$PYTHON_BIN` (portabilité `.venv`/
  `python3`) que `_INVOKE`/`_WRAPPED` ne reconnaissaient pas — même
  catégorie de défaut que D-151 et `_DIRNAME_DE_SOI`, déjà documentée
  deux fois dans le même fichier. `tests/test_launcher_paths_resolve.py`
  corrigé : 35 → **61** invocations balayées, vérifié complet ligne à
  ligne sur `run_study_v3.sh` (le plus gros écart : 10 lignes hors
  commentaire portant `.py`/`.sh`, 10 vues). Les trois planchers datés du
  fichier mis à jour avec la nouvelle mesure et sa justification, pas
  simplement abaissés.
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
l'un de ces faits doit d'abord le remesurer lui-même. D-192, D-193, D-194 et
D-186 font exception : chacun porte sa mesure de confirmation (ci-dessus,
ou dans `docs/RESULTS.md` pour D-192, trop volumineuse pour une puce),
avec sa commande.
