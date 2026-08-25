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

**D-158, D-98, D-100, D-50, D-39 et D-187, six des 9 « toujours ouvertes »
ci-dessus, ont eux aussi été refermés le 25 août** — même jour, passes
séparées. D-158 : la cause exacte du plantage était une exception non
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

**Re-vérifié le 25 août, décision toujours en attente.** Rien dans les
passes de ce jour n'a touché `dynamic_patch_labels.py` ni régénéré les
artefacts `d_patches_*.npz` dont ce défaut dépend ; le test `-m slow` n'a
pas été rejoué (coût, sans changement de code source à vérifier). Reste
un blocage de décision de campagne, pas de code.

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

**Re-vérifié le 25 août** (partiellement) : la garde non-`slow`
(`pytest tests/mapping/test_plaquette_signal_negligeable.py -q -m "not slow"`,
10 tests) passe toujours. Le balayage complet des 24 artefacts DNS
(`-m slow`) n'a pas été rejoué ; rien dans les passes de ce jour n'a
touché `HamiltParams_v2.py` ni régénéré de DNS. Reste un blocage de
décision de conception sur `src/`, pas un défaut de code.

---

## D-195 — deux tests QAOA restent rouges après D-191, cause distincte non élucidée

**Rapport seul.** Trouvé en vérifiant D-191 (`docs/RESULTS.md`) : ces deux
tests figuraient parmi ses cinq symptômes, mais rejoués sous une graine
confirmée aléatoire (la correction de D-191), ils rendent une valeur
identique à la décimale près à celle mesurée sous l'ancien `seed=0` fixe.
`test_C_ZZ`, qui isole la dispersion pure de l'échantillonnage QAOA sur
un hamiltonien constant, la mesure bien restaurée — donc l'absence de
changement sur ces deux tests n'est pas un signe que la graine reste
fixe : c'est le signe que D-191 n'est pas leur cause. Encore un défaut,
pas encore identifié.

| test | assertion | mesuré |
|---|---|---|
| `test_qaoa_scaling_and_hparams.py::test_hyperparameter_sweep` | `min(rho) > 0.0` sur 12 combinaisons `w_z_frac`×`threshold`, MHD Rotor | `-0,4667` — négatif sur 8/12, identique à la décimale près sous deux tirages indépendants |
| `test_qaoa_noise_and_early.py::test_noise_robustness` | `frac_cl - frac_qa > 0,09` sans bruit, Orszag-Tang | `0,0000` exactement — QAOA égale le classique au bit, identique sous deux tirages indépendants |

**Piste, non vérifiée.** Un commentaire déjà présent dans
`test_qaoa_noise_and_early.py` (section « quiet ») documente que la
correction du rotationnel D-1 a changé le classement du score sur
Orszag-Tang et fait tomber une égalité classique/optimum qui n'était
qu'une coïncidence (`frac_cl`/`gt_frac` mesuré alors : 0,9709, pas 1,0).
Si ce même changement de classement rend, sur ces deux configurations
précises, la sélection de blocs de QAOA insensible à son propre
échantillonnage (signal trop net pour que le bruit de tirage change quels
blocs sont choisis), cela expliquerait la stabilité au tirage des deux
mesures sans contredire `test_C_ZZ` — qui mesure la dispersion des scores
eux-mêmes, pas de la sélection discrète qui en découle. Hypothèse posée,
pas vérifiée en isolant `w_z_frac`/`threshold`/le scénario un par un.

```bash
pytest tests/quantum/test_qaoa_scaling_and_hparams.py::test_hyperparameter_sweep -q
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
