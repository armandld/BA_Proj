# Résultats

**Un résultat, comment il a été obtenu, donc comment le réobtenir.**

Chaque entrée porte : la commande exacte, les conditions, les nombres, et le
hash du commit qui les a produits. Un résultat qu'on ne sait pas refaire
n'est pas un résultat — il n'a pas sa place ici.

| document | contenu |
|---|---|
| `PLAN_PREPRINT.md` | l'objectif et les hypothèses — la source mère |
| `DEFAUTS.md` | où ça **bloque**, uniquement |
| `COUVERTURE.md` | ce qui est **testé**, comment et pourquoi |
| **`RESULTS.md`** (ce fichier) | ce qui est **accompli**, et comment le refaire |
| `EVALUATION.md` | ce qui, ici, est **exploitable** |
| `CODE_REVIEW.md` | note de relecture |

---

## Les 41 défauts corrigés

Le matériau le plus solide du travail. Chacun est mesuré avant et après,
refait par une commande, et verrouillé par un test qui échoue sur l'ancienne
version. Les mesures détaillées sont plus bas, dans les entrées de campagne.

**Conventions et opérateurs**

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-1 | rotationnel des mappeurs sous `indexing='xy'` | 0,0 → **+2,0** sur rotation solide | `pytest tests/study/test_curl_convention_gap.py` |
| D-3 | l'objectif pondère par cette vorticité fausse | 0,0 → **+2,0** | `pytest tests/mapping/test_objective_and_estimators_analytic.py` |
| D-11 | diode de choc appliquée au cisaillement | rapport **0,500 → 2,0** | `pytest tests/mapping/test_mapper_contracts.py -k flux` |
| D-17 | 3 sites hors `src/` en convention pré-D-1 | enstrophie **0 % → 0,02 %** d'écart | `pytest tests/study/test_no_private_curl_survives.py` |
| — | critère Q : déformation à moitié, partie isotrope comptée | cisaillement **+0,25 → 0** | `pytest tests/solver/test_analytic_fields.py -k q_criterion` |

**Numérique et rééchantillonnage**

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-2 | prolongation AMR centrée cellule, `mode='wrap'` | 2,49e−1 → **7,74e−6** | `pytest tests/amr/test_amr_resampling_analytic.py` |
| D-7 | projection ignore le mode de Nyquist | 0,378 → **1,1e−14** | `pytest tests/solver/test_solver_analytic.py -k idempot` |
| D-14 | réduction des champs tronque, celle du score non | 94,1 % → **100 %** | `pytest tests/mapping/test_downsampling_contracts.py` |
| D-21 | flux réduit par lissage + bilinéaire | pic **38 % → 100 %** | `pytest tests/mapping/test_padded_rescale_contracts.py` |
| D-23 | `dt` intégré ≠ `dt` écrit dans la trace DNS | référence à t≈0,077 → rejeu **exact** | `pytest tests/solver/test_precompute_dns_contracts.py` |
| D-25 | la projection **spectrale** de B abîme un champ solénoïdal en FD | div_FD B **4,63e−07 → 1,00e−14** | `pytest tests/solver/test_solver_convergence.py -k induction` |

**Encodage et décision**

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-8 | hamiltonien encode des coefficients nuls sans lever | non détecté → **lève** | `pytest tests/quantum/test_hamiltonian_contracts.py -k raises` |
| D-13 | bords gauche/haut lisent l'arête intérieure | asymétrie 1,2–7,0 % → **symétrique** | `pytest tests/quantum/test_hamiltonian_contracts.py -k halo` |
| D-15 | `postprocess` accepte des comptes bruts | marginales ~1000 → **refusé** | `pytest tests/quantum/test_vqa_chain_contracts.py -k refus` |
| D-16 | liste de patchs AMR se recouvre elle-même | **25 % → 0 %**, sans trou | `pytest tests/amr/test_amr_tiling_contracts.py` |
| D-19 | backend inconnu → contexte mort sans erreur | silence → **lève** | `pytest tests/quantum/test_runtime_contracts.py -k backend` |
| D-20 | cache d'ansatz confond deux hamiltoniens | même objet → **séparés** | `pytest tests/quantum/test_runtime_contracts.py -k ansatz` |

**Scénarios, mesure et documentation**

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-4 | doc annonce le double du facteur appliqué | ×2 → aligné | `pytest tests/mapping/test_objective_and_estimators_analytic.py` |
| D-5 | divergence notée sans pondération | 1,8 % → **0** | idem |
| D-6 | `init_magnetic_twist` ne pose aucune torsion | 6,4e−7 → **π/2 exact** | `pytest tests/solver/test_scenarios_analytic.py -k twist` |
| D-9 | ablation ψ mesure la fenêtre sur le mauvais score | « annihilation » → **ZZ domine K de 1,5 à 8,2×** | `pytest tests/study -k window` |
| D-12 | mappeur `study/` : ν, η, dx annoncés influents, sans effet | doc alignée | `pytest tests/mapping/test_mapper_contracts.py -k v2` |
| D-18 | garde de divergence à 1e100, inerte | 1e50 passait → seuil **1e8** | `pytest tests/solver/test_solver_guards_and_objective.py -k caught` |
| D-26 | `init_ghost_twisting` pose un champ **impossible** | angle **0,027 → 1,906 rad** | `pytest tests/solver/test_scenarios_analytic.py -k ghost` |
| D-27 | 4 scénarios initialisés non solénoïdaux, rabotés par la projection | perturbation **27,5 % → 100 %**, div_FD B **2,801e−03 → 1,08e−16** | `pytest tests/solver/test_scenarios_analytic.py -k "solenoidal or amputates"` |
| D-28 | `hyperparams_loader` substituait en silence les paramètres de l'**autre bras** (`quantum`↔`classical`) quand celui demandé manquait, et choisissait le premier lambda par ordre alphabétique quand plusieurs coexistaient | substitution → **lève** ; choix arbitraire → **lève sauf lambda unique** | `pytest tests/study/test_hyperparams_two_sources.py -k "refuses or implicit"` |
| — | `search_space` : 4 constantes présentées comme réglables | espace réel **5 paramètres**, pas 9 | `pytest tests/pipeline/test_train_hyperparams_contracts.py -k declares` |

**Le script d'entraînement** — audité pour la réoptimisation, un seul survit

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-29 | `SCENARIOS_ISOLATED` contenait les scénarios **complexes** ; `ot` et `rotor` comptés deux fois | **6 entrées / 4 classes → 6 / 6**, pondération 2:1 → 1:1 | `pytest tests/pipeline/test_train_hyperparams_contracts.py -k scenario` |
| D-30 | le chemin séquentiel appelle `_run_phase1(study, dns)` — la fonction prend **un** argument | `TypeError` après la phase 1 → chemin exécuté de bout en bout | `pytest tests/pipeline/test_train_hyperparams_smoke.py` |
| D-31 | `beta_michelson` proposé à Optuna, **jamais lu** par `pipeline.py` | phase 1 optimisait un paramètre sans effet → paramètre supprimé | `pytest tests/pipeline/test_train_hyperparams_contracts.py -k michelson` |
| D-32 | l'élagage ne rapportait qu'au step 0, sous `n_warmup_steps=2` | 1e9 au step 0 après 40 essais : **jamais élagué** → élagué au 3ᵉ scénario | `pytest tests/pipeline/test_train_hyperparams_contracts.py -k prun` |
| D-33 | `AdvAnomaliesEnable` absent d'Orszag-Tang, replié sur `False` | OT sans terme ZZZZ de point X → **6/6 scénarios** l'activent, la clé manquante **lève** | `pytest tests/pipeline/test_train_hyperparams_contracts.py -k argus` |
| D-34 | budget d'essais calculé une fois, par worker | 4 workers, cible 12 : **48 essais → 12** | `pytest tests/pipeline/test_train_hyperparams_contracts.py -k budget` |
| D-35 | le JSON final ne portait que les paramètres **échantillonnés** | `threshold_amr` absent → **9/9 valeurs** + hash git + argv | `pytest tests/pipeline/test_train_hyperparams_contracts.py -k redeploy` |
| D-36 | 3 des 4 sorties détaillées de `pipeline` sans provenance de `sigma` | trace présente **seulement sur les runs divergés** → sur les 4 | `pytest tests/solver/test_solver_guards_and_objective.py -k sigma` |

**Le chemin d'entraînement** — audité parce qu'il produit le nombre que la campagne minimise

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-37 | à **toute profondeur > 0**, le biais Z et les couplages décrivaient des grilles différentes | `H_edges` (6,6) contre `C_edges` (4,4) ; écart **0,05814 sur une échelle de 0,14107**, soit 41 % | `pytest tests/amr/test_patch_encoding_shapes.py` |
| D-38 | trois gardes de `execute` qui ne tenaient que sur le chemin habituellement testé | marginales **0,5535 → 0,700** ; Powell borné ; tirs MPS restaurés | `pytest tests/quantum/test_runtime_contracts.py -k "bound or null_hamiltonian or optimizer"` |

**Le diagnostic Phase 6** — `pipeline_verification.py` compare le classement
par énergie hamiltonienne v1 aux patchs durs. Sur les artefacts réels
`results/coefficients_{harris_tearing,kelvin_helmholtz}_Re400_N256_dim4.npz`,
l'énergie v1 est **identiquement nulle sur toute la simulation** — aucun
saut de cellule (`v_jump`, `B_jump`) n'y franchit jamais le seuil critique
de `PhysicalMapper` (`RE_CRIT`/`RM_CRIT` = 1.0), vérifié en rejouant
`compute_patch_coefficients` sur les 20 snapshots des deux scénarios,
contre `mhd_rotor`/`orszag_tang` où le seuil est franchi (E non nul,
100 %/70 % des cellules actives). Une énergie constante rend AUC/F1 égaux
à leur valeur de hasard (0,5/0,0) **par construction du calcul**, pas par
une mesure de non-discrimination — indiscernable à la lecture d'un vrai
résultat au hasard.

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-40 | la moyenne agrégée de `pipeline_verification.py` incluait des lignes à énergie constante (E≡0) comme si c'étaient de vraies mesures au hasard, tirant le verdict vers le bas en silence | sur les 4 scénarios canoniques (Re=400, N=256, dim=4) : AUC(E) **0,687 → 0,874**, F1(E) **0,364 → 0,729** — le verdict F1(E) vs F1(classique) passe de WARN (0,364 < 0,603) à PASS (0,729 > 0,654) une fois les 2 lignes dégénérées exclues et annotées plutôt que moyennées | `pytest tests/study/test_pipeline_verification_degenerate.py` |
| D-44 | `sanity_check.run_qaoa` décidait la convergence du QAOA sur `np.std(marg) > 0.01` — la **dispersion** des marginales — alors que le critère annoncé par son propre commentaire est « marginals should not all be 0.5 », c'est-à-dire la **distance à 0,5** | sur les défauts du script (Re=400, N=32, dim=2, 4 scénarios) le verdict s'inversait aux deux extrêmes : harris_tearing v1, marginales 0,976–0,980, `max|m−0,5| = 0,480` — le run le plus tranché des huit — était **« NOT converged (flat) »** (std 0,0019) ; kelvin_helmholtz v1 idem (0,239 contre std 0,0014). À l'inverse orszag_tang v1, déclaré convergé (std 0,0585), porte une marginale à **0,0169** de 0,5. Le bras QAOA n'étant pas déterministe, mesure refaite : 2ᵉ exécution identique en substance — harris_tearing v1 `0,473` contre std `0,0081`, kelvin_helmholtz v1 `0,249` contre std `0,0009`, l'inversion tient aux deux fois. Après : **4/4 → 8/8 convergés**, tolérance 0,01 inchangée mais portée sur la bonne grandeur | `pytest tests/study/test_sanity_check_convergence_criterion.py` |
| D-45 | phase 4 déclarait `promising` — la porte qui, d'après sa propre docstring, décide seule quels patchs passent en QAOA — sur une comparaison entre deux prédicteurs **constants** : l'état fondamental exact vaut « raffiner partout » et la ligne de base classique aussi, donc les deux F1 sont égaux par construction et la porte ne peut rien rejeter | dim=2 (seule dimension exécutable : dim=4/8 → 32/128 qubits > le plafond de 20), Re=400, N=256, 4 scénarios canoniques, 40 snapshots : décision exacte tout-à-1 **40/40**, décision classique tout-à-1 **40/40**, `exact_refine != classical_refine` **0/40**, F1 égaux **40/40**, jamais supérieurs. `promising` **40/40 True** avec `>=`, **0/40** avec le `>` que le commentaire annonce — la porte porte **0 bit** dans les deux sens. Après : `degenerate_decision` levé **40/40** et `promising_informative` **0/40**, la dégénérescence annotée au lieu d'être lue comme un succès | `pytest tests/study/test_exact_diag_degenerate_gate.py` |
| D-46 | `label_percentile_sensitivity.py` imprime « ROBUST … fails for ANY reasonable hard-patch definition » dès que `max(deltas) < 0.05`, alors que le docstring du module définit le seuil de robustesse comme « gap turns positive » (`delta < 0`) — une marge de 0,05 non documentée que rien dans l'historique git ne rattache à un choix explicite | sur l'artefact réel (`--dim 4 --N 256`, 4 scénarios canoniques, seed 0) : `max(delta) = -0,154` à p=75, sous les deux seuils — le verdict « ROBUST » ne change pas ici. Cas synthétique construit pour séparer les deux hypothèses (deltas −0,10 / −0,20 / **+0,03** / −0,15, un percentile où le site bat le classique) : ancien seuil imprime **« ROBUST … fails for ANY »** malgré le +0,03 positif ; nouveau seuil imprime **« SENSITIVE … F1_site beats classical by +0,030 »**, la lecture conforme au docstring | `pytest tests/study/test_percentile_sensitivity_interpretation.py` |
| D-43 | `find_optimal_threshold` balayait ses seuils avec `flat_e >= thr` : sur une énergie **constante**, les 100 percentiles sont égaux et chaque candidat prédit **tous** les patchs durs — le F1 rendu était celui du classifieur tout-positif, `2p/(p+1)`, présenté comme un pouvoir de séparation | mêmes artefacts : harris_tearing **0,400 → NaN**, kelvin_helmholtz **0,376 → NaN** (`E.ptp = 0` sur les deux) ; mhd_rotor **0,950** et orszag_tang **0,519** inchangés. 0,400 se lisait comme un signal réel un peu plus faible que le 0,519 authentique d'OT — et identique à tous les Re, donc comme un **seuil parfaitement stable**, la conclusion même que `threshold_stability_report` existe pour produire | `pytest tests/study/test_find_optimal_threshold_degenerate.py` |

Pas une correction du calcul de l'énergie hamiltonienne elle-même (`src/`
n'est pas touché : `RE_CRIT`/`RM_CRIT` restent ceux du contrat en vigueur,
et la question de savoir si ce seuil convient aux scénarios lisses
tearing/KH est une décision physique, pas un défaut de code — voir
`DEFAUTS.md`). Uniquement le diagnostic `study/` qui ne doit plus confondre
« aucun signal calculé » avec « chance mesurée ».

**Le diagnostic Phase 1B** — en ré-auditant `check_tearing` pendant l'examen
de D-39 (même fonction, même PR) : son docstring exige un pic « strictement
à l'intérieur de la trace (pas à t=0, pas à la fin) », et la clause « pas à
t=0 » (`growing_from_start`) est bien vérifiée — mais la clause « pas à la
fin » (`growing`) compare `j[i_peak]` à `j[min(i_peak+1, len(j)-1)]` : quand
le pic tombe sur le **dernier** échantillon, `min(...)` retombe sur
`i_peak` lui-même, et la comparaison devient `j[i_peak] <= j[i_peak]*1.01`
— toujours vraie. La clause ne peut jamais échouer : une croissance qui ne
retombe jamais avant la fin de la fenêtre (donc jamais observée en train de
« piquer ») passe quand même.

Mesuré sur les 6 fichiers DNS `harris_tearing` réels de `results/` : avec
`J2` = `mean_sq_current` (câblage gelé), le pic tombe sur le **dernier**
échantillon (`i_peak = 19/20`) sur les **6/6** fichiers — la trace est
encore strictement croissante à la fin de la fenêtre simulée, ce n'est pas
un pic observé. `check_tearing` rendait pourtant `ok=True` (amplification
1,53–2,65×) sur les 6, exactement à cause du défaut ci-dessus.

| # | ce qui était faux | avant → après | vérifier |
|---|---|---|---|
| D-42 | `check_tearing` : la clause « pic pas à la fin de la trace » se comparait à elle-même quand le pic tombait sur le dernier échantillon, donc ne pouvait jamais échouer | 6/6 fichiers `harris_tearing` (câblage gelé) : pic au dernier échantillon, `ok` **True → False** | `pytest tests/study/test_check_tearing_end_pinned_peak.py` |

**Conséquence pour D-39** (voir `DEFAUTS.md`) : la comparaison « ancien
câblage `ok=True` contre câblage corrigé `ok=False` » qui motivait D-39 est
maintenant à relire — sur les 6 fichiers disponibles, le câblage gelé ne
passait que grâce à ce défaut, pas parce qu'il observait un vrai pic. Une
fois D-42 appliqué, les deux câblages rendent `ok=False` sur les 6 : la
question posée par D-39 (quelle observable sépare fond stationnaire et
reconnexion) reste entière, mais elle ne peut plus s'appuyer sur « ça
marchait avant » — ça ne marchait pas, au sens où `check_tearing` l'exige.

**Douze de ces défauts viennent d'une seule question** — *deux chemins censés
coïncider coïncident-ils encore ?* Aucun test de valeur ne pouvait les voir :
tous rendaient un résultat plausible.

Deux ont été vus non par une question, mais en **retirant une couche** : D-26
et D-27 n'apparaissent qu'une fois la projection de B supprimée. Tant qu'elle
masquait la divergence, les scénarios paraissaient sains.

Les entrées sont dans l'ordre où elles ont été produites. Celles qu'un
résultat postérieur a dépassées sont **conservées, avec la rétractation
écrite sur place** — c'est la trace de ce qui a été cru, et pourquoi ce
n'est plus vrai.

L'ordre historique fait apparaître des références à « V3 » et « V4 » : ce
sont des étapes de l'étude, pas des versions du code. Les campagnes
antérieures sont dans `docs/archive/`.

**Règle de continuité** : aucun symbole de V1 n'est redéfini. Tout ce qui
est réutilisable est importé — `MHDSolver`, `build_patch_hamiltonian`,
`build_ising_terms`, `sa_multi_restart`, `spins_to_decisions`,
`prepare_qaoa_inputs`, `run_qaoa_on_snapshot`, `div_B`, `total_energy`,
`downsample_fields`, `bootstrap_by_trajectory`, `git_commit_hash`.

**Recette de vérification**

```bash
python -m pytest tests/ -q -m "not slow"
python study/common/aggregate_master_table.py     # recalcule chaque nombre
```

Le troisième est le test de non-régression : il recalcule chaque nombre
publié depuis son artefact. **État actuel : 164 OK / 16 DIFF / 0 MISSING** —
les 16 écarts sont les nombres déplacés par les corrections, à republier
après la réoptimisation (voir `DEFAUTS.md`).

Toutes les études ont été relancées à la résolution de production **N=256**
(4 scénarios, Re=400). Les deux passes — N=64 exploratoire et N=256 de
confirmation — sont rapportées ; chaque conclusion qualitative est identique
aux deux résolutions.

### D-28, en détail

Trouvé en auditant par contrat `src/hyperparams_loader.py` (`VIGIL.md`, Q4 :
deux chemins censés coïncider). Le correctif existait déjà — commit
`0327ce1`, 12 tests dans `tests/study/test_hyperparams_two_sources.py` —
mais n'avait de ligne ni dans `DEFAUTS.md` ni ici : exactement la déviation
que `VIGIL.md` interdit de laisser non écrite. Cette entrée referme l'écart ;
le correctif lui-même n'est pas de cette passe.

**Avant** (`_load_new_format`, avant `0327ce1`) :

```python
entry = default.get(method)
if entry is None:
    # Fallback: try the other method
    for m in ['quantum', 'classical']:
        if m in default:
            entry = default[m]
            break
if entry is None:
    raise KeyError(f"No default {method} params found in JSON")
```

Demander `method='quantum'` sur un fichier ne portant que `classical`
renvoyait donc les paramètres de l'**autre** bras, sans le signaler — la
boucle essaie `'quantum'` puis `'classical'` dans cet ordre fixe, donc
systématiquement, pas seulement par accident. `src/pipeline.py` ne peut pas
distinguer ce cas d'un vrai jeu quantique : la comparaison des deux bras
devient vide de sens, en silence. Second repli du même ordre : plusieurs
`lambda_cost` pour une phase → le premier pris par ordre alphabétique, un
choix arbitraire indiscernable d'un choix motivé.

**Après** : `KeyError` explicite listant les bras disponibles pour le
premier ; `KeyError` sauf lambda unique (choix alors forcé, donc licite)
pour le second.

**Mesuré** sur `results/hyperparams/best_hyperparams.json` (gelé) : les
deux bras s'y chargent toujours normalement — non-régression — et les deux
fautes lèvent sur les cas construits qui les provoquent
(`test_the_loader_refuses_to_substitute_the_other_arm`,
`test_the_loader_refuses_an_ambiguous_cost_weight`,
`test_a_single_cost_weight_stays_implicit`).

```bash
pytest tests/study/test_hyperparams_two_sources.py -q
```
→ 12 passed (vérifié à `HEAD`, `claude/kind-babbage-927g10`).
`python study/common/aggregate_master_table.py` inchangé (164 OK / 16 DIFF /
0 MISSING) : cette entrée ne déplace aucun nombre publié.

**Trouvé au passage, non corrigé.** Le bloc `per_scenario` du bras quantique
contient quatre copies **identiques** du bloc `default` — aucun réglage n'y
est réellement par scénario — et `orszag_tang`/`mhd_rotor` (2 des 4
scénarios de l'étude) en sont absents pour les deux bras.
`load_hyperparams(scenario=...)` n'est appelé nulle part dans le dépôt
(`grep -rn "load_hyperparams(.*scenario" src/ study/` ne rend rien) : la
branche est morte, aucun nombre publié n'en dépend aujourd'hui. Épinglé,
pas corrigé, par `test_the_per_scenario_quantum_block_is_one_set_repeated`
et `test_two_study_scenarios_have_no_per_scenario_entry` — pour que
personne ne « répare » ce bloc en y recopiant `default`, ce qui masquerait
le problème au lieu de le trancher.

### Écart de registre trouvé en écrivant cette entrée — signalé, non corrigé

En cherchant où consigner D-28, deux incohérences non liées à D-28 :

- **Collision de numéro.** La section « D-18 » plus bas dans ce fichier
  (« rectification : la moitié `fluctuating_KE` était déjà connue »)
  raconte la déviation reclassée depuis comme **D2** dans `DEFAUTS.md`
  (table « Gelés volontairement »). Le numéro D-18 désigne aujourd'hui,
  dans la table ci-dessus et dans `DEFAUTS.md`, un défaut différent : la
  garde de divergence à 1e100. Un lecteur qui suit
  `DEFAUTS.md/RESULTS.md → D-18 → chercher "D-18" plus bas dans ce fichier`
  tombe sur la mauvaise section. Survit à la réorganisation du 12 août.
- **Compte de tête inexact.** Avant cette entrée, le titre de cette section
  annonçait « Les 24 défauts corrigés » pour 23 lignes numérotées D-N dans
  la table — un défaut d'écart au niveau du registre lui-même, pas du code.
  L'ajout de D-28 le rend exact par coïncidence ; ne pas s'y fier comme
  preuve que le compte était juste avant.

Non corrigé ici : ni l'un ni l'autre n'est de mon audit, et je ne peux pas
écrire leur mesure sans la fabriquer — mesurer, documenter, ne pas
corriger sans la mesure, règle de `VIGIL.md`.

---

## T11 — Quantum-contribution attribution (audit P0)

`study/v4/h0_optimiser_equivalence.py --N 64 --dim 2 --n-snaps 2`

At the **deployed size** (`VQA_N = 2` → 8 qubits, periodic root scan, i.e.
exactly the configuration `refinement.py` solves at depth 0).

| solver | hit optimum | E gap | spin agreement | mask match | wall (s) |
|---|---|---|---|---|---|
| exhaustive (certified) | 1.000 | 0 | 1.000 | 1.000 | 0.000 |
| simulated annealing | 1.000 | 0 | 1.000 | 1.000 | 0.121 |
| SA warm-started | 1.000 | 0 | 1.000 | 1.000 | 0.123 |
| greedy local search | 1.000 | 0 | 1.000 | 1.000 | 0.000 |
| classical decision alone | 1.000 | 0 | 1.000 | 1.000 | 0.000 |
| QAOA p=1 (statevector) | 1.000 | 0 | 1.000 | 1.000 | 0.414 |
| QAOA p=2 (statevector) | 1.000 | 0 | 1.000 | 1.000 | 0.612 |
| QAOA p=2, 4096 shots | 1.000 | 0 | 1.000 | 1.000 | 0.617 |

- The cost Hamiltonian is **diagonal** (Z/ZZ/ZZZZ only), verified at runtime
  by `is_diagonal_cost_hamiltonian` on every snapshot. Its ground state is a
  computational basis state, so "exact diagonalisation" reduces to a
  classical enumeration of 2^8 = 256 configurations.
- Every solver reaches the certified optimum and returns the same mask.
  **Pre-registered rule fires: quantum optimisation is not the source of any
  gain.** A closed-loop improvement would attribute value to the
  Hamiltonian, not to its quantum optimiser.

**Caveat that makes the agreement partly vacuous** (see T11b): the optimum
itself is uniform, so the solvers agree on a trivial problem.

---

## T11b — Does the QAOA optimise its own Hamiltonian? (audit P0)

`study/v4/h0_qaoa_displacement.py --N 64 --dim 2 --reps 1 2 3 4`

Position of three points in marginal space: `m_theta` (amplitude encoding of
the classical score alone), `m_qaoa` (optimised circuit), `m_gs` (exact
ground state). `progress` = projection of the realised displacement on the
required one; 0 = decision unchanged, 1 = optimum reached.

| reps | progress | ‖displacement‖ | ‖required‖ | ‖remaining‖ | mean marginal |
|---|---|---|---|---|---|
| 1 | +0.0590 | 0.1276 | 0.8381 | 0.8010 | 0.7217 |
| 2 | +0.0563 | 0.1178 | 0.8381 | 0.8030 | 0.7205 |
| 3 | −0.0298 | 0.1178 | 0.8381 | 0.8536 | 0.7044 |
| 4 | −0.0584 | 0.1883 | 0.8381 | 0.8830 | 0.6980 |

- **The exact ground state is a UNIFORM mask on 100% of snapshots**
  (8/8: 4 scenarios × 2 snapshots) — refine-all, carrying no spatial
  information. Cause (consistent with V3 Task 9): the ferromagnetic
  couplings dominate the Z bias, |C| ≈ 2.0 and |K| = 1.0 against
  |h| ≈ 0.071, a ratio ≈ 28.
- **Mean variational progress = 0.0068** (0.68%). The circuit's displacement
  is essentially orthogonal to the direction of its own optimum.
- Progress **does not increase with depth**; it becomes negative by reps=4
  (−0.117 from reps 1 to 4). Deeper circuits move slightly *away* from the
  optimum of the declared cost.

**Reading.** The deployed decision is not a minimiser of the declared cost
function. It is a ≤4%-in-norm perturbation of the amplitude encoding
θ = 2·arcsin(√score), i.e. of the classical score itself.

---

## T13 — Causal ablation of term families (audit P1)

`study/v4/t13_term_ablation.py --N 64 --dim 2 --n-snaps 2`

Exact ground state recomputed after zeroing each family (control `full`
must change nothing).

| ablation | decisions changed | uniform | refined fraction | F1 | n_optima |
|---|---|---|---|---|---|
| full (control) | **0.000000** | 1.000 | 1.000 | 0.317 | 1.0 |
| no_Z | 1.0000 | 1.000 | 0.000 | 0.000 | 8.0 |
| no_ZZ | **0.0000** | 1.000 | 1.000 | 0.317 | 1.0 |
| no_ZZZZ | **0.0000** | 1.000 | 1.000 | 0.317 | 1.0 |
| Z only (both couplings removed) | **0.0000** | 1.000 | 1.000 | 0.317 | 1.0 |
| couplings only (Z removed) | 1.0000 | 1.000 | 0.000 | 0.000 | 8.0 |

- Removing **all** ZZ and **all** ZZZZ couplings changes **no decision**.
  The single-site Z bias alone reproduces the full-Hamiltonian decision
  exactly.
- Removing the Z bias destroys the decision entirely and leaves an
  8-fold degenerate ferromagnet.
- The control is exactly 0, which validates the measurement chain.

**Reading.** At the deployed grid size the coupling terms — the entire
motivation for an Ising/quantum formulation — are **causally inert**. This
is a causal statement, unlike the post-hoc ZZ/ZZZZ attributions of the
manuscript.

---

## T12 — Equivariance and orbit error (audit P1)

`study/v4/t12_equivariance.py` (dim=2 exact; dim=8 with annealed ground
state and a mandatory reproducibility control).

Step 1 — the transformation must be a symmetry of the discrete solver:
`eps = ‖T(step(U)) − step(T(U))‖ / ‖step(U)‖`.

| op | eps (N=64) |
|---|---|
| rot180 | 2.8e-16 (machine precision — exact symmetry) |
| flip0 / flip1 / rot90 | 7.8e-6 |

Step 2 — orbit error of the decision map, dim=8 (structured masks):

| op | classical route | ground-state route |
|---|---|---|
| flip0 | 0.0195 | 0.3984 |
| flip1 | 0.0508 | 0.3555 |
| rot180 | 0.0547 | 0.3359 |
| rot90 | 0.0508 | 0.3047 |
| **mean** | **0.0439** | **0.3486** |

Step 3 — **mandatory control** (`solver_noise_floor`): disagreement of the
ground-state route between annealing seeds **on the same, untransformed
field** = **0.2676**, with the refined fraction swinging by 0.15 across
seeds.

- The classical score map is **nearly equivariant** (4.4% orbit error,
  deterministic, floor = 0). The residual is attributable to the one-sided
  finite differences used in the indicator.
- The ground-state route's 0.349 orbit error is **not interpretable as
  non-equivariance**: the annealed optimiser is itself irreproducible at a
  comparable magnitude (floor 0.268). The verdict printed by the script
  requires a 2× margin over the floor, which is not met.
- At dim=2 with exact enumeration, orbit error is exactly 0 for all routes —
  but only because the mask is uniform, so the test is vacuous there.

**Reading.** What this establishes is not an equivariance defect but a
**degeneracy defect**: at dim=8 the objective is flat enough that two
annealing seeds disagree on 14–37% of patches. A decision defined as
"the ground state" is not well posed at that size.

---

## T14 — Numerical validation of the V1 solver (audit P1)

`study/v4/t14_numerical_validation.py`

**(A) Self-convergence**, all solutions restricted to the coarsest grid:

| grids | ‖u_N − u_2N‖_rel | observed order |
|---|---|---|
| 32 → 64 → 128 (t=0.5) | 7.41e-02, 3.71e-02 | **1.00** |
| 64 → 128 → 256 (t=0.25) | 3.34e-02, 1.67e-02 | **1.00** |

**(B) Conservation and solenoidal constraint** (every trajectory):
energy monotonically decreasing, drop 0.3–1.8%; `max|div B| / rms|B|`
between 5.6e-15 and 8.0e-14 — machine precision.

**(C) Reynolds numbers outside the training grid** {400, 800, 1200, 1600}:
Re = 200 and Re = 3200 both pass (monotone energy, div B ≈ 1.5e-14).

**(D) Localisation of the first-order behaviour** — temporal convergence at
fixed dt, with and without the projection step:

| n_steps | with projection (as in `step_full`) | without projection |
|---|---|---|
| 16 | 3.35e-03 | 3.53e-07 |
| 32 | 1.63e-03 (order 1.04) | 2.22e-08 (order 3.99) |
| 64 | 7.61e-04 (order 1.10) | 1.39e-09 (order 4.00) |
| 128 | 3.26e-04 (order 1.22) | 8.66e-11 (order 4.00) |
| **mean order** | **1.12** | **4.00** |

Direct order test of the spatial operators on a smooth periodic field:
`_fd_grad` and `_fd_laplacian` are **exactly 4th order** (4.00 at every
refinement).

**Reading — see the defect note below.** The spatial stencils and the RK4
kernel are both 4th order, but `step_full` applies a full RK4 step *then*
the divergence-free projection. That Lie splitting is first order and caps
the whole scheme at first order in time; since CFL ties dt to dx, the
space–time self-convergence is first order.

---

## Defect notes for the manuscript

**D-V4-1 (numerical, material for the methods section).** The paper
describes the solver as "fourth-order finite differences in space, RK4 in
time". Both components are verified 4th order in isolation, but the
*scheme* converges at **order ≈ 1** because the incompressibility
projection is applied as a first-order operator splitting after the
complete RK4 step (`solver.py::step_full`). Isolated, reproducible
diagnostic in T14(D). This does not invalidate the comparisons — both arms
share the solver, the runs are paired, div B is at machine precision and
all phase-1b invariants pass — but the accuracy statement must be corrected,
and any convergence claim must quote order 1.

**D-V4-2 (modelling, material for the results section).** At the deployed
size the exact ground state of the cost Hamiltonian is uniform (T11b),
the coupling terms are causally inert (T13), and the circuit realises 0.68%
of the displacement toward its own optimum (T11b). The Q-HAS decision is
therefore a small perturbation of the classical score encoding rather than
an optimisation outcome. This mechanistically explains the 0.66% composite
gain, the 109 flipped decisions with 45 correct and 64 incorrect, and the
mask asymmetry, without invoking any quantum effect.

**D-V4-3 (methodological).** A "ground state" obtained by annealing at
dim ≥ 4 is not reproducible across seeds (14–37% of patches, T12 control).
Any statement about ground-state decisions above 8 qubits requires that
floor to be reported alongside.


---

## N=256 confirmation (production resolution)

Command set: `logs/v4/v4_N256.log`. 4 scenarios, Re=400, 12 snapshots for
T11/T13, 8 for T11b, dim=2 (deployed size) and dim=8 (structured masks).

### Every conclusion holds; the numbers sharpen

| quantity | N=64 | **N=256** | verdict |
|---|---|---|---|
| exact ground state uniform | 100% | **100%** | unchanged |
| cost Hamiltonian diagonal | True | **True** (12/12 snapshots) | unchanged |
| solvers reaching certified optimum | all | **all except cold SA** | see below |
| QAOA mask = exact ground state | 1.000 | **1.000** (p=1,2,3 + shots) | unchanged |
| variational progress toward own optimum | 0.0068 | **0.0854** | still ≈ 0 |
| progress change, reps 1 → 4 | −0.117 | **−0.172** | still *decreasing* |
| ablation: remove all ZZ | 0.0000 changed | **0.0000 changed** | unchanged |
| ablation: remove all ZZZZ | 0.0000 changed | **0.0000 changed** | unchanged |
| ablation: remove Z bias | 1.0000 changed | **1.0000 changed** | unchanged |
| classical-route orbit error (dim=8) | 0.0439 | **0.0146** | improves with resolution |
| self-convergence order | 1.00 | **1.00** | unchanged |
| temporal order, projection ON / OFF | 1.12 / 4.00 | **1.12 / 4.00** | unchanged |

### T11 at N=256 — one new observation

| solver | hit optimum | E gap | mask match | F1 | wall (s) |
|---|---|---|---|---|---|
| exhaustive (certified) | 1.000 | −1.4e-17 | 1.000 | 0.389 | 0.000 |
| simulated annealing (cold) | **0.583** | 1.41e-02 | 0.583 | 0.367 | 0.139 |
| SA warm-started | 1.000 | −1.4e-17 | 1.000 | 0.389 | 0.133 |
| greedy local search | 1.000 | −1.4e-17 | 1.000 | 0.389 | 0.000 |
| classical decision alone | 1.000 | −1.4e-17 | 1.000 | 0.389 | 0.000 |
| QAOA p=1 / p=2 / p=3 | 1.000 | −1.4e-17 | 1.000 | 0.389 | 0.75 / 1.12 / 1.42 |
| QAOA p=3, 4096 shots | 1.000 | −1.4e-17 | 1.000 | 0.389 | 1.603 |

New at N=256: **cold-start simulated annealing misses the certified optimum
on 42% of snapshots** (E gap 1.4e-2) while the warm-started variant, greedy
descent and every QAOA depth reach it exactly. The optimum is trivially
reachable *from the classical decision* but not from a random start — the
landscape is flat with a narrow basin. This strengthens rather than weakens
the attribution conclusion: the only solver that struggles is the one that
does not start from the classical answer.

### T11b at N=256

| reps | progress | ‖disp‖ | ‖required‖ | ‖remaining‖ | mean marginal |
|---|---|---|---|---|---|
| 1 | +0.1588 | 0.1685 | 0.7487 | 0.6504 | 0.7739 |
| 2 | +0.1192 | 0.1569 | 0.7487 | 0.6759 | 0.7653 |
| 3 | +0.0766 | 0.1392 | 0.7487 | 0.7055 | 0.7555 |
| 4 | −0.0132 | 0.1706 | 0.7487 | 0.7662 | 0.7376 |

Mean progress 0.0854, monotonically decreasing with depth and negative by
reps = 4. The ground state is uniform on 100% of snapshots.

> **⚠ Lecture à requalifier — D-48.** Ces quatre lignes ont été obtenues avec
> `warm_start_params = classical_warm_start_params(...)`, un schedule
> **constant** `(β = 0,05 ; γ = 0,15/k)` qui, malgré son nom, ne lit ni le
> score classique ni le seuil (écart mesuré **0,0e+00** sur 6 entrées) et
> qu'**aucun chemin déployé n'emprunte**. Rejoué à la configuration publiée
> avec l'initialisation par défaut du dépôt (rampe `π/E_max` d'`execute()`,
> celle que `refinement.py` prend quand son cache est vide), 3 répétitions
> par profondeur : progression moyenne **+0,186** au lieu de +0,091, et
> tendance reps 1 → 4 de **−0,0002** au lieu de −0,116 — plate, de signe
> variable d'une répétition à l'autre. Les deux bras sont séparés aux quatre
> profondeurs.
>
> **Les nombres ci-dessus ne sont pas retirés et n'ont pas bougé** : ils
> décrivent exactement ce que le code exécute. Ce qui est en cause est la
> phrase qu'on en tire — « une progression qui n'augmente pas avec la
> profondeur signifie que l'objectif déclaré n'est pas l'objectif optimisé ».
> Mesurée, elle vaut pour ce schedule-là, pas pour le circuit. Les décisions,
> elles, ne bougent pas (0 différence sur 4 scénarios), donc les lignes T11
> `QAOA p1/p2 mask match` sont intactes. Trois options et leur coût :
> `docs/DEFAUTS.md` D-48.

### T12 at N=256

dim=8 orbit error: classical route **0.0146** (flip0 0.0078, flip1 0.0156,
rot180 0.0195, rot90 0.0156) — three times smaller than at N=64, consistent
with the one-sided-finite-difference explanation (the defect scales with
grid spacing). Ground-state route 0.4219 against a reproducibility floor of
**0.3613** → the script correctly refuses the interpretation. At dim=2 with
exact enumeration everything is 0 (uniform mask, vacuous).

### T14 at N=256 — the solver order question, settled

Self-convergence on grids 64 → 128 → 256 at t = 0.25: errors 3.344e-02 and
1.673e-02, **observed order 1.00**. Splitting diagnostic run *at N=256*
(`--split-N 256`): with projection order **1.12** (err 3.35e-03 → 3.27e-04),
without projection order **4.00** (err 3.76e-07 → 9.21e-11). Conservation:
energy monotone at every resolution, `max|div B|/rms|B|` ≤ 8.0e-14, and
Re = 200 / 3200 (outside the training grid) both pass.

**The first-order behaviour is not a low-resolution artefact.** It is
identical at N=64 and N=256, and the diagnostic isolates the cause at
production resolution: the Lie splitting between the RK4 step and the
divergence-free projection in `solver.py::step_full`.

---

## T15 — Level 3, closed-loop LOSO (audit P0, decisive experiment)

`study/v4/t15_level3_closed_loop.py`

### Status when this entry was written: driver built, campaign not yet run

> The campaign has since run on all four folds. This entry describes the
> driver; the results are in the T15/T15b/T15c/T19/T20/T23 entries below.

The driver performs a true pipeline-level LOSO fold: for each held-out
instability class it (1) tunes the QAOA hyperparameters with Optuna on the
composite loss of the **other** classes only, reusing V1's own
`make_composite_objective`; (2) tunes the **classical** arm's AMR threshold
on the same training classes via `make_classical_composite_objective`, so
both arms suffer the identical exclusion; (3) runs both arms on the held-out
class with the same DNS trace, hot start, hybrid budget and depth. Endpoints
come from `pipeline(..., return_details=True)`: `phys_score` (relative L2 vs
DNS), `patch_ratio` (compute) and `combined`. Per-fold results are written
incrementally to JSON, so an interrupted campaign resumes.

**End-to-end validation** (`--smoke`, N=64, T_MAX=0.4): the complete path
runs to completion and writes both outputs. Smoke numbers are degenerate by
construction (both arms refine everything, delta = 0) and are not
scientific; the mode exists only to de-risk a day-long run.

### Defect found in the V1 training module (blocking for LOSO)

`train_hyperparams.SCENARIOS_ALL = SCENARIOS_ISOLATED + SCENARIOS_COMPLEX`
where `SCENARIOS_ISOLATED` already contains `ot` and `rotor` and
`SCENARIOS_COMPLEX` re-adds **the same config objects**. The list therefore
has 6 entries for **4 distinct classes**, and since the composite loss is
`mean(Loss_i)` over the list, OT and rotor are weighted 2/6 each against
1/6 for KH and tearing — an undocumented 2:1 tilt in every Phase-3 training
run. For a LOSO fold the consequence is worse: excluding `ot` would leave
its duplicate in the training list, i.e. **manufacture leakage**.
`fold_scenarios` de-duplicates by key and prints a warning. Related: the
module defines `SCENARIO_VORTEX` and `SCENARIO_COALESCENCE` (lamb_oseen,
island_coalescence) but never uses them, while its own docstring claims
Phase 1 trains on "KH, VORTEX, TEARING, COALESCENCE".

### Measured cost model (N=256, this container)

| stage | measured |
|---|---|
| DNS traces per fold (3 train + 1 held) | 225 s |
| one full `pipeline()` run at N=256 | **≥ 5 min** |
| one Optuna trial = 3 training scenarios | ≈ 15 min |

Per fold ≈ 4 min (DNS) + 15·`n_trials` min (QAOA tuning) + ≈ 6·`n_cls` min
(classical tuning) + 7 min (both final arms).

| `--n-trials` | per fold | 4 folds |
|---|---|---|
| 8 | ≈ 2.6 h | ≈ 10 h |
| 10 | ≈ 3.2 h | ≈ 13 h |
| 12 | ≈ 3.8 h | ≈ 15 h |
| 170 (protocol) | ≈ 43 h | ≈ 7 days |

**Deviation to log when the campaign runs:** the protocol freezes the V1
Optuna budget at 170 trials; a one-day campaign affords 8–12. The script
prints the deviation itself when `--n-trials < 170`. Other standing
deviations: 4 folds (the V1 module exposes 4 distinct classes, not the 8 of
protocol §1.1) and a single physics seed per fold.

### Recommended command for a one-day run

```
nohup python study/v4/t15_level3_closed_loop.py \
      --n-trials 10 --n-trials-classical 5 \
      > logs/v4/level3.log 2>&1 &
```

Resumable: each completed fold is skipped on restart. Monitor with
`grep -E "FOLD|tuning|Q-HAS|classical\]" logs/v4/level3.log`.

### T13 with the **deployed V1 mapper** (N=256, dim=2)

The ablation above used the parameter-free V2 mapper. Re-run with the V1
mapper (`--mapper v1`, the `TRAINED_*` coefficients the pipeline actually
deploys):

| ablation | decisions changed | uniform GS | refined | F1 | n_optima |
|---|---|---|---|---|---|
| full (control) | **0.000000** | 1.000 | 0.750 | 0.333 | 64.8 |
| no_Z | 0.7500 | 1.000 | 0.000 | 0.000 | 88.0 |
| no_ZZ | **0.0000** | 1.000 | 0.750 | 0.333 | 64.8 |
| no_ZZZZ | **0.0000** | 1.000 | 0.750 | 0.333 | 64.8 |
| Z only (both couplings removed) | **0.0000** | 1.000 | 0.750 | 0.333 | 64.8 |

Same conclusion as for V2: the ZZ and ZZZZ families are **causally inert**
for the deployed Hamiltonian. Two V1-specific observations: the ground state
is uniform on 100% of snapshots but is *refine-all* on only 75% of them, and
the V1 cost function is **massively degenerate** — 64.8 of the 256
configurations are optimal on average (88 once the bias is removed).
Inspection of the coefficients explains both: at dim=2 the V1 mapper yields
median |C| = 0 with |h| ≈ 200–240, and on harris_tearing every coefficient
is zero, i.e. an identically null Hamiltonian on which the QAOA has nothing
to optimise.

---

## T15 / T15b — Level 3 closed loop: first fold, and the budget-matched reversal

### T15, fold `ot` (Orszag–Tang excluded from all tuning)

`study/v4/t15_level3_closed_loop.py --folds ot --n-trials 4 --n-trials-classical 2`

| endpoint | Q-HAS | tuned classical | Δ (Q−C) |
|---|---|---|---|
| combined (primary) | 0.3328 | 0.4386 | **−0.1058** |
| phys_score (L2 vs DNS) | 0.1940 | 0.4845 | −0.2905 |
| patch_ratio (compute) | 0.6797 | 0.3238 | **+0.3558** |

Taken at face value this favours Q-HAS on the pre-registered primary
endpoint. **It must not be read that way**, because the two arms are not at
the same point of the error–cost frontier, and the asymmetry is inherited
from the V1 training module:

- `make_composite_objective` (QAOA arm) **hard-codes**
  `HyperParams["threshold_amr"] = 0.14959824837662078` — never suggested to
  Optuna, with the source comment "le meilleur classique";
- `make_classical_composite_objective` (classical arm) optimises
  `trial.suggest_float("threshold_amr", 0.05, 0.8)` freely and selected
  **0.4616** for this fold.

A 3× threshold difference explains the 2.1× compute gap and hence the
fidelity gap. This is exactly the "budget-matched comparison" the audit
demanded, and it is a **third defect** in the comparison design: it applies
to V1's own closed-loop numbers, not only to this fold.

### T15b, budget-matched classical arm (same fold)

`study/v4/t15b_budget_matched.py --fold ot --max-iter 4` — bisection on the
classical threshold to reproduce the Q-HAS compute budget, everything else
(DNS trace, hot start, hybrid budget, depth) held fixed.

Classical error–cost frontier on the held-out class:

| threshold | patch_ratio | phys_score |
|---|---|---|
| 0.0500 | 0.9480 | 0.0111 |
| 0.1438 | 0.7369 | 0.0649 |
| **0.1906** | **0.6412** | **0.0827** |
| 0.2375 | 0.5866 | 0.1027 |
| 0.4250 | 0.3554 | 0.2899 |
| 0.8000 | 0.0156 | 0.5894 |
| *Q-HAS* | *0.6797* | *0.1940* |

**Budget-matched result: Δ phys = +0.1113 in favour of the classical arm.**
At *slightly less* compute (0.6412 vs 0.6797) the classical rule achieves
**2.3× lower** L2 error against DNS (0.0827 vs 0.1940). Q-HAS lies well
above the classical frontier — it is **strictly Pareto-dominated** on this
fold.

Two readings sharpen this further:
- At a *matched threshold* the conclusion is the same: classical at
  thr = 0.1438 gives phys = 0.0649 at patch = 0.7369, while Q-HAS at
  thr = 0.1496 gives phys = 0.1940 at patch = 0.6797 — 3× worse fidelity
  at comparable settings. The gap is therefore not a threshold artefact:
  the QAOA perturbation of the θ encoding actively degrades the decision
  relative to plain thresholding of the same score.
- This is coherent with T11b and T13: the circuit does not optimise its own
  cost (progress ≈ 0, decreasing with depth) and the coupling terms are
  causally inert, so the perturbation it applies carries no useful
  information.

**Pre-registered decision rules (`docs/level3_preregistration.md`).**
P1 (equivalence) is **not** supported on this fold: the arms differ, and
under budget matching the difference is large and favours the classical
rule. P3 (any fidelity gain is paid in compute) is **confirmed and then
some** — the gain does not survive paying for the compute. The
`combined`-endpoint verdict of T15 is superseded by the budget-matched
comparison, which is the interpretable one.

**Scope.** One fold (`ot`), one physics seed, 4 Optuna trials. The campaign
was interrupted twice by container reclamation while running folds `kh`,
`rotor`, `tearing`; those folds remain to be run. No claim of general
closed-loop falsification is made from n = 1. What *is* established is that
the apparent closed-loop advantage of the primary endpoint does not survive
the audit's budget-matched control on the fold measured.

---

## T17 — ZZ uncertainty window: the mechanism behind causal inertness

```
python study/v4/t17_uncertainty_window.py --N 64 --steps 30
```
git hash: see `results/t17_uncertainty_window.npz`  ·  runtime ≈ 1 s
(the four DNS spin-ups dominate; N=64, 30 steps each)

**Why this task exists.** T13 established a *fact*: zeroing the ZZ family
changes 0.0000 decisions. T17 establishes the *mechanism*. The lead came
from V1's own test suite — see defect **D6** below — which contains two
failing tests asserting the opposite.

**Mechanism.** `HamiltParams.compute_coefficients` multiplies the entire ZZ
family by a Gaussian centred on the AMR decision threshold,
`w = exp(-((score - threshold_amr)/sigma)^2)`. The intent is to concentrate
coupling where the classical decision is uncertain. The effect is that the
coupling is removed from exactly the cells where it is largest: strong
gradients produce large `|C|` *and* confident (far-from-threshold) scores.

**Measurements** (four classes × two parameter sets). `no window` is
obtained by setting σ → 1e9 so that `w ≡ 1`; V1 is never modified. Mass
kept = Σ|C|·w / Σ|C|, each edge family paired with its own window.

**Three parameter sets, not two.** There are two distinct "trained" σ, and
conflating them changes the numbers by 100+ orders of magnitude:
`TRAINED_SIGMA` = **0.023** is the open-loop pipeline constant used by
phase5 and therefore by T11/T13/T18; σ = **0.1888** is what Optuna found for
the Level-3 fold `ot`, i.e. closed loop only. The deployed set is read from
the module rather than hard-coded, so it cannot drift from what runs.

Deployed **open-loop** parameters — the configuration behind T11/T13/T18
(σ = 0.023, threshold = 0.1496). This is the harshest case:

| class | max\|C\| no window | mass kept | Spearman(\|C\|,w) |
|---|---|---|---|
| kelvin_helmholtz | 53.92 | 1.319e-02 | −0.372 |
| mhd_rotor | 136.0 | 7.652e-28 | −0.400 |
| orszag_tang | 63.59 | 4.187e-125 | −0.012 (degenerate) |
| harris_tearing | 42.32 | 3.855e-154 | −0.502 |

ZZ is **numerically dead on three of four classes** at the deployed
open-loop setting, and retains 1.3 % on the fourth.

Level-3 **closed-loop** parameters (σ = 0.1888, threshold = 0.1496) — the
most permissive setting, and the one governing the T15 folds:

| class | max\|C\| no window | max\|C\| with window | mass kept | Spearman(\|C\|,w) |
|---|---|---|---|---|
| kelvin_helmholtz | 53.92 | 36.71 | 1.142e-01 | −0.372 |
| harris_tearing | 42.32 | 0.0935 | 1.990e-03 | −0.502 |
| mhd_rotor | 136.0 | 1.331 | 3.951e-04 | −0.460 |
| orszag_tang | 63.59 | 0.6955 | 9.679e-05 | −0.008 (degenerate) |

Parameters of the failing V1 tests (σ = 0.05, threshold = 0):

| class | w_max | max\|C\| with window | mass kept |
|---|---|---|---|
| kelvin_helmholtz | 9.964e-01 | 19.60 | 7.449e-03 |
| harris_tearing | 2.626e-01 | 2.626e-58 | 2.537e-60 |
| mhd_rotor | 1.010e-19 | 9.943e-18 | 9.547e-23 |
| orszag_tang | 4.228e-50 | 1.773e-48 | 1.314e-53 |

**Reading.** Before the window the coupling is healthy on *every* class
(40–136). After it, three of four classes retain under 0.2 % of the
coupling mass, and the best case retains 11.4 %. The rank correlation
between coupling magnitude and window weight is negative wherever it is
defined, i.e. the suppression is not uniform noise — it is targeted at the
strongest couplings. At the tests' parameters the window underflows
outright (4e-50 on OT, 1e-19 on rotor).

Note `harris_tearing` under the test parameters: `w_max` = 0.26 looks
healthy, yet mass kept = 2.5e-60. `max(|C|·w) ≠ max|C|·max(w)` — the window
is large only where the coupling is not. This is the anti-correlation in its
starkest form and is why the window's effect cannot be judged from `w_max`.

**Consequence.** The Ising formulation's rationale is the multi-body
coupling. The deployed pipeline discards ~99 % of it before the QAOA ever
sees it, which is a sufficient explanation for T13's null ablations and for
T11b's near-zero variational progress.

**Defect D6.** `bash run_tests.sh` does **not** pass on a clean checkout.
Re-running the V1 suite in a detached worktree at `cf93ba3` (the last commit
touching `src/` or `tests/`, well before any V3/V4 work) reproduces an
identical set of 8 failures:

- 6 × `TypeError: PhysicalMapper.__init__() got an unexpected keyword
  argument 'beta'` — the tests call a signature that no longer exists.
- 2 × substantive assertions:
  `test_coefficients_survive_orszag_tang` ("Orszag-Tang should produce
  significant C_edges", actual 1.77e-48) and
  `test_hamiltonian_carries_spatial_info_beyond_score` ("C_edges should be
  nonzero at velocity boundary", actual 1.79e-42).

The two substantive failures are the V1 author's own guard against exactly
the failure mode T17 characterises. They have been failing, not passing.

**Defect D7.** The uncertainty window annihilates the family it is meant to
focus (numbers above). Documented irony: V1 replaced Michelson
normalisation because it *"kills the signal when the domain is uniformly
active"*; the uncertainty window reintroduces that failure mode at the score
level.

Tests: `tests/study/test_t17_uncertainty_window.py` (9).

---

## T18 — counterfactual: are the ZZ terms inert *without* the window?

```
python study/v4/t18_window_counterfactual.py --N 256 --dim 2 --n-snaps 2
```
runtime ≈ 2 s (reuses the stored DNS/patch inputs) · deployed v1 mapper

**Why this task exists.** T17 shows the uncertainty window discards most of
the ZZ coupling. That immediately raises the question a referee will ask,
and the answer decides how far the paper's conclusion reaches:

> is the causal inertness of ZZ a property of the **Ising formulation**, or
> an artefact of **this implementation**?

If the window were solely responsible, the defect would be a repairable
engineering bug and the critique would not touch the approach.

**Protocol.** Two Hamiltonians per snapshot, same physics, same deployed v1
mapper: `windowed` (the pipeline as it runs) and `no_window` (σ → 1e9, so
w ≡ 1). Neutralisation is done by substituting the module constant used to
*construct* the mapper and restoring it in a `finally`; V1 is never
modified, and the substitution is asserted, not assumed (|C| without the
window must dominate |C| with it). The T13 ablations are then replayed on
each arm — `zero_hamiltonian_terms` and `ground_state_mask` are imported,
never redefined.

**Coupling amplitude at the deployed configuration** (N=256, dim=2). Note
these are *more* extreme than the N=64 figures in T17: at VQA resolution the
patch-averaged fields are smoother, so the score sits even further from the
threshold.

| class | snap | max\|C\| windowed | max\|C\| no window |
|---|---|---|---|---|
| orszag_tang | 14 | 1.33e-189 | 137.5 |
| orszag_tang | 29 | 5.65e-145 | 154.5 |
| harris_tearing | 10, 19 | **0.000e+00** | 24.89 |
| kelvin_helmholtz | 14 | **0.000e+00** | 124.2 |
| kelvin_helmholtz | 29 | **0.000e+00** | 77.32 |
| mhd_rotor | 14 | 1.25e-189 | 117.2 |
| mhd_rotor | 29 | 2.70e-200 | 143.9 |

At the deployed size the ZZ family is **identically zero in double
precision** on Kelvin–Helmholtz and Harris tearing, and at 1e-145 or below
on the others.

**Ablations, both arms:**

| arm | ablation | changed | uniform GS | refined | F1 | n_optima |
|---|---|---|---|---|---|---|
| windowed | full (control) | **0.0000** | 1.000 | 0.750 | 0.250 | 64.8 |
| windowed | no_Z | 0.7500 | 1.000 | 0.000 | 0.000 | 88.0 |
| windowed | no_ZZ | **0.0000** | 1.000 | 0.750 | 0.250 | 64.8 |
| windowed | no_ZZZZ | **0.0000** | 1.000 | 0.750 | 0.250 | 64.8 |
| no_window | full (control) | **0.0000** | 1.000 | 1.000 | 0.250 | 1.0 |
| no_window | no_Z | 1.0000 | 1.000 | 0.000 | 0.000 | 22.0 |
| no_window | **no_ZZ** | **0.0000** | 1.000 | 1.000 | 0.250 | 1.0 |
| no_window | **no_ZZZZ** | **0.0000** | 1.000 | 1.000 | 0.250 | 1.0 |

**Result.** With the coupling restored from numerically zero to O(25–155),
ablating ZZ *still* changes **0.0000** decisions; likewise ZZZZ. The
inertness is therefore **not** an artefact of the uncertainty window. It is
a property of the formulation at the deployed size: the Z bias alone fixes
the ground state, and the multi-body terms cannot move it.

This is the stronger result for the paper — it forecloses the "your
implementation was simply buggy" rebuttal. The window is a real defect
(D7), but repairing it would not make the coupling terms matter.

**A separate, subtler finding.** The window does change decisions —
**25.0 %** of them (full Hamiltonian, windowed vs neutralised) — but *not*
by acting as coupling. |C| feeds `C_scale`, the median of non-zero |C| and
|K| that sets the Z-bias amplitude `alpha_z = w_z_frac × C_scale`.
Suppressing C therefore rescales the **Z bias**, and the decision moves
through that normalisation side-channel. The coupling influences the outcome
only as an input to a scale factor — never as a coupling. Between the arms
the ground state also goes from 64.8-fold degenerate to unique.

Note the control (`full` = 0.0000) holds in both arms, so the measurement
chain is validated separately for each.

Tests: `tests/study/test_t18_window_counterfactual.py` (7), including a
positive control — the instrument is shown to detect a change when one
exists, without which "changed = 0" everywhere would prove nothing.

### T18 addendum — an *independent* counterfactual: the V2 mapper

The σ → ∞ neutralisation in T18 is a manipulation of the v1 mapper, so a
referee may reasonably ask whether the conclusion is an artefact of the
manipulation. It is not, and the repository already contained the control:

**`PhysicalMapperV2` has no uncertainty window at all.** Its own docstring
lists what was removed relative to v1: *"Removed: sigma (Gaussian
uncertainty width) … Removed: f-gate, g-gate, threshold-contrast, Gaussian
weighting"*. It is parameter-free, using plain domain-normalised ratios.

Its ZZ coupling is consequently healthy — measured at the deployed
configuration (N=256, dim=2), max|C_edges|:

| class | snap | v2 (no window) | v1 (windowed) |
|---|---|---|---|
| orszag_tang | 14 / 29 | 2.455 / 2.613 | 1.33e-189 / 5.65e-145 |
| kelvin_helmholtz | 14 / 29 | 2.774 / 2.522 | **0.000e+00** / **0.000e+00** |
| mhd_rotor | 14 / 29 | 2.017 / 2.101 | 1.25e-189 / 2.70e-200 |
| harris_tearing | 14 | 3.989 | **0.000e+00** |

And the ablations on that mapper (N=256, dim=2, `--n-snaps 3`, 72 rows):

| ablation | changed | uniform GS | refined | F1 | n_optima |
|---|---|---|---|---|---|
| full (control) | **0.0000** | 1.000 | 1.000 | 0.389 | 1.0 |
| no_Z | 1.0000 | 1.000 | 0.000 | 0.000 | 8.0 |
| **no_ZZ** | **0.0000** | 1.000 | 1.000 | 0.389 | 1.0 |
| **no_ZZZZ** | **0.0000** | 1.000 | 1.000 | 0.389 | 1.0 |

So the conclusion now rests on **two independent routes**:

1. **v1 with the window neutralised** (T18): coupling restored to O(25–155)
   → ZZ ablation 0.0000.
2. **v2, independently designed without a window** (T13, mapper v2):
   coupling natively O(2–4) → ZZ ablation 0.0000.

The second route involves no manipulation of any kind. The causal inertness
of the multi-body terms is a property of the formulation at the deployed
size, not of the v1 implementation and not of the σ → ∞ device.

**Defect D9 (in V4's own code, now fixed).** `t13_term_ablation.py` wrote
`t13_term_ablation_N{N}_dim{D}.npz` *regardless of `--mapper`*, so running
the v2 comparison silently overwrote the v1 result — precisely the
comparison the task exists to make. The filename now carries the mapper;
the historical name is still written for v1 so published references keep
resolving. Found by re-deriving the v2 numbers instead of citing them.

**Reproducibility check.** Re-running the published v1 configuration
(`--n-snaps 3`) reproduces the stored artifact **bit-exactly** across all
72 rows (`scenario`, `snap`, `ablation`, `changed`, `uniform`, `n_optima`,
`f1`, `refined`, `dE`).

### D6 follow-up — how far does the signature drift reach?

D6 reports 8 failures in the V1 suite on a clean checkout, 6 of them
`TypeError: PhysicalMapper.__init__() got an unexpected keyword argument
'beta'`. The question that matters for the paper is whether that drift
touches the code which produced the results. It does not.

Every call site, checked exhaustively:

| call site | kind | uses removed `beta=` |
|---|---|---|
| `src/pipeline.py:325` | **production pipeline** | no — current signature |
| `study/phase0_sanity_check.py:95` | study | no |
| `study/phase3_coefficients.py:68` | study | no |
| `study/phase4_exact_diag.py:68` | study | no |
| `study/qaoa_inputs.py:136` | study (feeds T11/T13/T18) | no |
| `src/compare_rotor_budget.py:110` | orphaned analysis script | **yes → dead** |
| 6 × `tests/…` | stale tests | **yes → the D6 failures** |

**Verdict.** The simulations behind every V3 and V4 number were produced by
code that constructs the mapper correctly. The drift is confined to stale
tests and to one script that nothing imports.

**Defect D10.** `src/compare_rotor_budget.py` raises `TypeError` at line 108
and cannot execute. It and `HamiltParams.py` were both last modified in
`cf93ba3` and are unchanged since; the repository has full (non-shallow)
history of 57 commits. **As committed, this script has never been runnable
in this repository.** It is referenced only by a file listing in
`README.md`. If any rotor budget-comparison figure or number in the
manuscript is attributed to it, that attribution needs checking — the script
in its committed form could not have produced it.

---

## T15 — Level-3 fold `kh` (Kelvin–Helmholtz held out)

```
bash study/v4/run_fold.sh kh
```
tuning: QAOA 4 trials (best train loss 0.2590), classical 2 trials (0.3841)

| arm | combined | phys (rel. L2 vs DNS) | patch ratio | wall (s) |
|---|---|---|---|---|
| Q-HAS | 0.2443 | 0.0070 | 0.8376 | 579 |
| **classical** | **0.1800** | **0.0020** | **0.6250** | 213 |

**The classical arm wins on every endpoint simultaneously**: better fidelity
(3.5× lower L2), cheaper (25 % fewer refined pixels), and better composite.
Unlike fold `ot`, this needs **no budget-matched control** — Q-HAS is
**strictly Pareto-dominated at the tuned operating point itself**. The
budget-matched run is still executed, but only to map the frontier; it
cannot change the direction of the conclusion.

Note the training losses reproduce fold `ot`'s pattern — QAOA better than
classical on the *training* composite (0.2590 vs 0.3841 here, 0.1984 vs
0.2979 on `ot`) while losing on the *held-out* class. That is defect **D4**
in action: the QAOA arm's `threshold_amr` is pinned at 0.1496 while the
classical arm tunes its own freely, so a training-loss advantage reflects a
different operating point rather than a better decision rule.

### Cross-fold state after 2 of 4 folds

| fold | Q-HAS combined | classical combined | Δ (Q-HAS − cl) | better |
|---|---|---|---|---|
| ot | 0.3328 | 0.4386 | −0.1058 | Q-HAS |
| kh | 0.2443 | 0.1800 | **+0.0643** | **classical** |

Pre-registered readings, stated at their true scope:

- **Counting rule** (`docs/level3_preregistration.md` §4): 1–1 at n = 2.
  Neither arm meets the ≥ 3/4 threshold. **Nothing is established yet.**
- **TOST**: margin 0.0155 (5 % of mean classical `combined`, per the frozen
  formula), diff −0.0208, p_TOST = 0.520 → **equivalence not established**.
- **Difference test**: paired t p = 0.848, Holm-adjusted 1.000 → no
  significant difference. Exact sign test p = 1.000, and note the minimum
  attainable at n = 2 is 0.500 — the design cannot produce significance here
  regardless of the data.
- **Budget-matched (secondary, post-hoc):** Q-HAS dominated on 1/1 folds so
  far; `kh` is dominated already without the control.

The honest summary at this point: on the two folds measured, Q-HAS is
Pareto-dominated on both — on `ot` only after correcting the operating-point
asymmetry, on `kh` outright. The *primary* pre-registered endpoint remains
undecided by its own counting rule until 3 or 4 folds are in.

---

## T19/T20 — the Q-HAS arm is not deterministic (defect D11)

The T19 audit replays each Level-3 arm with **identical** inputs (same DNS
trace, same hot start, same hyperparameters) and checks it reproduces the
stored value. Fold `ot`:

| arm | stored `combined` | replayed `combined` | stored phys | replayed phys |
|---|---|---|---|---|
| classical | 0.4386 | **0.4386** (exact) | 0.4845 | 0.4845 |
| **Q-HAS** | 0.3328 | **0.3108** | 0.1940 | **0.1345** |

The classical arm reproducing bit-exactly proves the trace, hot start and
configuration are identical — so the variance is specific to the QAOA path.
A 44 % swing in `phys_score` between two runs of the same configuration.

**Cause.** No RNG seed is fixed anywhere in V1's VQA chain: `AerSimulator`
is built without `seed_simulator`, and both `Estimator` and `Sampler` run at
`default_shots = 256` (`create_argus`: `shots=256`, `backend="state_vector"`,
`method="COBYLA"`). The Q-HAS arm is therefore doubly stochastic:

1. the objective COBYLA minimises is a 256-shot estimate, so the optimiser
   follows a different trajectory each run;
2. the final marginal read-out is itself a 256-shot draw.

The classical arm samples nothing, hence its exact reproducibility — which
is what makes it a valid control rather than a coincidence.

**Consequence.** Every published Level-3 Q-HAS number is **one draw** from a
distribution whose spread has never been measured. `--seed` cannot fix this:
the randomness is inside V1's unseeded Aer backend, and seeding it would
require modifying V1.

**Scope of the damage — what still holds.** On fold `ot` the two observed
Q-HAS draws are phys ∈ {0.1345, 0.1940}; the budget-matched classical arm
achieves **0.0827**. Both draws are worse, so the *direction* (Q-HAS
Pareto-dominated) survives, while the *magnitude* (quoted as 2.3×) is
uncertain over roughly 1.6×–2.3×. The same caution applies to `kh`
(Q-HAS 0.0070 vs matched classical 0.0017).

**T20** quantifies the spread directly: K repeats of the Q-HAS arm on one
fold with identical inputs, plus classical repeats as a determinism control,
and reports the between-arm gap divided by the Q-HAS run-to-run standard
deviation. A gap smaller than ~2 standard deviations means a single run per
arm cannot support a directional claim on that fold.

```
python study/v4/t20_qhas_run_variance.py --fold kh --repeats 5
```

**This is the strongest methodological caveat in the V4 set, and it applies
to V1's own published closed-loop numbers too** — those were also single
runs of the same unseeded pipeline.

---

## T19 complete + T21 — the endpoint judgement becomes a measurement

### T19 arm audit, all four folds

| fold | Q-HAS arm | classical arm | verdict |
|---|---|---|---|
| `ot` | completed | completed | **usable** |
| `kh` | completed | completed | **usable** |
| `rotor` | completed | **ABORTED, step 208 (t=0.2739)** | **failed** |
| `tearing` | completed | completed | **usable** |

The classical arm reproduced its stored value **bit-exactly on all four
folds**; the Q-HAS arm reproduced on **none** — the D11 signature.

### T19 bisection-trace audit

| fold | aborted points |
|---|---|
| `rotor` | **2/6** — thr 0.4250 (step 371), thr 0.8000 (step 198) |
| `tearing` | **0/6** |

**A heuristic would have been wrong here.** `tearing`'s point at
phys = 4.1258 looks like a divergence and is not: it *completed*. It is a
genuine operating point at thr = 0.8, patch = 0.0727 — refine almost
nothing and the solution is badly wrong but stable. A rule such as
"phys > 1 ⇒ diverged" would have deleted a valid frontier point. The
criterion used is V1's own execution trace, never the value.

`rotor`'s two aborts also explain its fold failure: the tuned classical
threshold, 0.4616, sits inside the unstable band between 0.4250 and 0.8000.
The tuner selected an operating point that diverges on the held-out class —
a second instance of D4 doing damage.

### T21 — is the primary endpoint well posed?

Replaces the *argument* "the primary endpoint is contaminated by D4" with
three measurements, none requiring new simulation. `rotor` excluded per
pre-registration §5 (failed audit).

**1. Pareto dominance — no λ involved.**

| fold | dominates | λ-free verdict |
|---|---|---|
| `kh` | **classical** | yes |
| `tearing` | **classical** | yes |
| `ot` | incomparable | no |

**2/3 folds are decided without any λ, both for the classical arm, none
for Q-HAS.**

**2. λ crossover**, for the fold dominance cannot decide. The two arms'
`combined` cross at λ\* = (phys_c − phys_q)/(patch_q − patch_c):

- `ot`: **λ\* = 0.8164**. Q-HAS wins below, classical above. The
  pre-registered λ = 0.4 sits **below** the crossover.

**3. Count stability across λ:**

| λ | Q-HAS wins | classical wins |
|---|---|---|
| 0.0 – 0.8 | 1 | **2** |
| **≥ 1.0** | **0** | **3** |

**Correction to an earlier reading.** The "2–2 split establishes nothing"
reported before the audit **included `rotor`**, whose classical arm had
diverged and was therefore scored as a Q-HAS win. With `rotor` excluded as
pre-registration §5 requires, the primary endpoint favours the classical arm
**2–1 at the pre-registered λ**, and **3–0 for λ ≥ 1**.

At λ ≥ 1 the classical arm meets the pre-registered refutation threshold
(§4: *"If the classical arm wins on ≥ 3/4 folds … the falsification is
complete and closed-loop"*), on 3/3 valid folds.

**What this measures and what it does not.** It measures that the verdict is
partly a property of the chosen λ rather than of the arms — ill-posedness,
quantified, not asserted. It does **not** remove D4. Removing it requires
re-tuning the QAOA arm with `threshold_amr` in the search space so both arms
optimise the same free parameters: hours of compute, and the definitive
experiment.

### Figure updated

`figures_v4/pareto_panel.*` now (a) excludes `rotor`'s two aborted points
from the plotted frontier, and (b) uses a **logarithmic error axis** — the
classes span 1–3 decades, and since the compared quantity *is* a ratio, a
log axis makes a given ratio span the same vertical distance in every panel.
The full data, including excluded points, remains in the `.csv`.

**(c) The Q-HAS marker is no longer a single draw.** It plotted
`t15b["qhas"]`, one run of an unseeded arm, and annotated 2.57×, 4.41×,
3.62×, 4.38× — the retracted ratios. Anyone comparing the figure with the
corrected tables would have seen two different studies. It now plots the
**mean of the completed repeated draws with x and y error bars**
(`rotor`: 3 draws, its 2 aborted ones excluded), and falls back to the
single draw only when no repeats exist — saying so in the legend.

**The figure's ratio and the tables' ratio are different quantities.** The
figure divides by the frontier *interpolated at the budget Q-HAS actually
realised*; the tables divide by the budget-matched point T15b *measured*.
They differ because T15b matched its threshold to one draw while the plotted
point is a mean of five — on `ot`, budget 0.756 against 0.680, and the
frontier is lower there:

| fold | vs interpolated frontier (figure) | vs measured matched point (tables) |
|---|---|---|
| `ot` | 1.79× | 1.30× |
| `kh` | 2.10× | 1.90× |
| `rotor` | 2.49× | 2.74× |
| `tearing` | 1.98× | 1.81× |

Both are in `pareto_panel.csv` (`ratio` and `ratio_vs_matched`) so no reader
has to guess which one a number came from.

---

## T20 — Q-HAS run-to-run variance on fold `kh` (D11 quantified)

```
python study/v4/t20_qhas_run_variance.py --fold kh --repeats 5
```
5 Q-HAS runs + 2 classical controls, identical inputs, 3216 s.

| metric | Q-HAS mean | std | range | CV | classical range |
|---|---|---|---|---|---|
| combined | 0.2500 | 0.0104 | 0.0232 | 0.042 | **0.00e+00** |
| **phys_score** | **0.00324** | **0.00158** | 0.0039 | **0.489** | **0.00e+00** |
| patch_ratio | 0.8670 | 0.0376 | 0.0785 | 0.043 | **0.00e+00** |

Q-HAS `phys` draws: **0.0015, 0.0020, 0.0031, 0.0042, 0.0053**.

**The control passes.** The classical arm's range is exactly **0.00e+00** on
all three metrics across both repeats — a fifth independent confirmation of
its determinism. Without that, the Q-HAS spread could have been an artefact
of the measurement chain; with it, the spread is attributable to the
unseeded QAOA path (D11) and nothing else.

**A 48.9 % coefficient of variation on the fidelity metric.**

### The published `kh` numbers were one draw, and it was the extreme one

The fold's stored Q-HAS value, 0.00700, sits at the **100th percentile** of
all six known draws — it is the largest. Everything computed from it is
correspondingly inflated:

| quantity | from the stored draw | **from the mean of 5 draws** |
|---|---|---|
| gap / std | 3.15 → "direction survives" | **0.77 → a single run cannot support a directional claim** |
| ratio vs budget-matched classical | 4.16× (published as 4.41×) | **1.93×** |

**The `kh` ratio is roughly halved.** T20 originally reported only the
stored-draw figure, which is the optimistic choice; it now computes both and
quotes the mean-based one.

### What survives, and it is the dominance count, not the ratio

Against the budget-matched classical arm (phys 0.00168 at patch 0.7943):

- Q-HAS costs **more on 5/5 draws** (patch 0.830–0.908 vs 0.794);
- Q-HAS is less faithful on **4/5 draws**;
- on the remaining draw the arms are **incomparable** (Q-HAS more faithful,
  but more expensive) — **never reversed**.

So the direction holds as a **dominance count over draws**, not as a point
ratio. The honest statement for `kh` is *"classical is cheaper on every
draw and more faithful on four of five"*, not *"Q-HAS is 4.4× worse"*.

### Consequence for the other folds

`ot`, `rotor` and `tearing` each have **one** Q-HAS draw (plus a replay for
`ot`). Their published ratios rest on the same single-draw basis and should
be read as **point estimates of a quantity with ≈50 % CV**, not as measured
magnitudes. Repeating T20 per fold is the fix; it costs ~1 h per fold.

---

## T20 complete — Claim E restated as a dominance count over repeated draws

> **SUPERSEDED — do not quote the per-fold numbers in this section.** This
> pass did not capture each draw's abort status, so `rotor`'s mean silently
> included 2 diverged trajectories. See *T20 verified* below for the numbers
> that stand (1.30×, 1.90×, 2.74×, 1.81×); the section is kept because the
> comparison between the two passes is what shows how much an unguarded
> draw distorts a mean.

5 Q-HAS repeats per fold, identical inputs, plus 2 classical repeats per
fold as a determinism control. **The classical control's range is exactly
0.00e+00 on every metric of every fold** — 8 independent replays. The spread
below is therefore attributable to the unseeded QAOA path (D11) alone.

### Per-fold distribution, against the **budget-matched** classical arm

| fold | Q-HAS mean | sd | CV | matched ref | gap/sd | ratio published → **mean-based** |
|---|---|---|---|---|---|---|
| `ot` | 0.1291 | 0.0222 | 17.2 % | 0.0827 | **2.09** | 2.35× → **1.56×** |
| `kh` | 0.0032 | 0.0016 | 48.9 % | 0.00168 | **0.98** | 4.16× → **1.93×** |
| `rotor` | 0.1537 | 0.0642 | 41.8 % | 0.0536 | **1.56** | 3.13× → **2.86×** |
| `tearing` | 0.0091 | 0.0034 | 37.4 % | 0.00443 | **1.37** | 4.19× → **2.05×** |

**On three folds of four the gap/sd is below 2**: a single run per arm
cannot support a claim about *magnitude*. Every published ratio was inflated
by a factor 1.1–2.2, because each rested on one draw.

### Why the reference must be the budget-matched arm, always

T20 first compared against the *tuned* classical arm, which is wrong twice
over and produced two spectacular non-results:

- **`rotor`**: the tuned classical arm had **aborted**, so its stored value
  is a partial score. gap/sd came out **15.88** — against a crashed run.
- **`ot`**: the tuned classical arm *completes* but runs at a different
  budget (patch 0.324 against Q-HAS's 0.680, defect D4). gap/sd came out
  **16.01**, measuring the operating point, not the decision rule.

Both are now excluded by construction: the reference is the budget-matched
point, whose completion the T19 trace audit verified.

### The robust statement

```bash
python study/v4/t23_headline_counts.py     # recomputes the table below
```

| fold | n | aborted | less faithful | costlier | strictly dominated |
|---|---|---|---|---|---|
| `ot` | 5 | 0 | 5/5 | 5/5 | **5/5** |
| `kh` | 5 | 0 | 5/5 | 4/5 | **4/5** |
| `rotor` | 3 | 2 | 3/3 | 2/3 | **2/3** |
| `tearing` | 5 | 0 | 5/5 | 5/5 | **5/5** |
| **total** | **18** | 2 | **18/18** | **16/18** | **16/18** |

> Across four held-out classes and **18 completed** closed-loop runs, Q-HAS
> is less faithful than the budget-matched classical rule on **every one of
> the 18**, more expensive on **16 of 18**, and strictly Pareto-dominated on
> **16 of 18**. No run reverses the ordering on both coordinates at once.

**Correction — this table previously read 19/20, 18/20, 17/20.** It was the
only headline in the study composed by hand rather than computed, and it did
not reproduce from the artifacts. Two errors, both of a kind already in the
register:

1. on `kh`, *less faithful* and *costlier* were **transposed** (4/5 and 5/5
   instead of 5/5 and 4/5);
2. on `rotor`, the **2 aborted draws were counted in the denominator**,
   giving a total out of 20 when only 18 runs completed — the exact defect
   ("an aggregation mixing aborted draws with valid ones") that had been
   fixed in the code and reappeared in the prose.

The corrected count is **stronger on fidelity** (unanimous, 18/18, where the
old figure conceded one run) and **weaker on cost** (16/18). The direction of
the conclusion is unchanged. T23 now computes it and `t16` checks it, so the
number can no longer drift from its artifacts.

This is the form Claim E should take in the manuscript. It is weaker-sounding
than "2.6–4.4× worse" and far harder to attack: it depends on no single draw,
no choice of λ, and no scalarisation.

### Correction to an earlier claim of mine

I wrote that the published value was the maximum draw "on all four folds".
That was true for `kh`, `ot` and `tearing` but **not** `rotor`, whose stored
value sits at the 67th percentile. Three of four, generalised too early from
three observations.

---

## D13 — a train/test leak in the Level-3 protocol, and the unseen-condition test

### The leak

`docs/level3_preregistration.md` states the held-out class is excluded from
**all** tuning of both arms. That is **false for the QAOA arm**.

`train_hyperparams.make_composite_objective` hard-codes the decision
threshold:

```python
if "threshold_amr" not in frozen:
    HyperParams["threshold_amr"] = 0.14959824837662078   # le meilleur classique
```

and that number comes from `_run_classical_phase1`, whose own banner reads
**"Scenarios: KH + OT + Tearing + Rotor"** — all four classes. So on every
fold, the QAOA arm decides using a threshold fitted on data that includes
the held-out class. My driver reproduced it verbatim:
`best.setdefault("threshold_amr", 0.14959824837662078)`.

The classical arm has no such problem: `train_classical_threshold_excluding`
re-tunes its threshold per fold on the training classes only.

**The leak is asymmetric and favours Q-HAS.** It is therefore *conservative*
with respect to the conclusion — Q-HAS is beaten on all 18 completed runs
despite holding an advantage it should not have. But the protocol's claim of a clean
LOSO is wrong as written and must be corrected in the manuscript.

This is also the precise form of defect D4: not merely "different operating
points" but a genuine information leak.

### The second, independent problem: the initial condition was never new

Even with the parameter leak removed, V1's `_init_dns_scenario` calls every
`init_*` **without arguments**, so every evaluation uses the canonical
initial condition. A model that generalises must face a condition it has
never met, not the canonical trajectory of a class it merely did not tune on.

**T22** supplies that test. It substitutes `_init_dns_scenario` temporarily
(V1 unmodified, restored in a context manager, and the substitution is
*verified*: the run aborts if the trajectory does not actually change) to
pass physical parameters to the initialisers:

| class | unseen condition |
|---|---|
| Kelvin–Helmholtz | narrower shear layer, weaker seed, faster drift |
| Harris tearing | thinner current sheet, **mode 2** instead of mode 1 |
| MHD rotor | slower, smaller rotor, wider taper |
| Orszag–Tang | **no IC parameters exist** — the only available unseen condition is a different Reynolds number, declared as such |

Verified distinct at N=64 before launching: KH 3773.6 → 4118.3,
tearing 3546.8 → 2951.0, rotor 4739.8 → 4409.3, and V1's function object
restored identically afterwards.

The reported quantity is the **degradation ratio** of each arm,
phys(unseen) / phys(canonical), so the comparison is between how the two
decision rules *transfer*, not between their absolute errors.

---

## Trap sweep — where else can an invalid run masquerade as a valid one?

The recurring failure mode in this campaign is a computation that fails but
returns a value **indistinguishable from a valid one**. It has now surfaced
five times (T15 fold scoring, T20 gap/sd, T22 classical reference, T22
Q-HAS draws, and the T13/T19 filename overwrites). A systematic sweep of
every `run_arm` call site in `study/v4/`:

| call site | guarded? | recoverable after the fact? |
|---|---|---|
| `t15:313` Q-HAS fold arm | no | **no** — non-deterministic (D11) |
| `t15:319` classical fold arm | no | yes — deterministic, T19 audits it |
| `t15b:66` bisection points | no | yes — classical only, T19 `--trace-only` audits it |
| `t19:88` audit replay | **yes** | — |
| `t20:120` Q-HAS variance draws | **was no** | **no** |
| `t20:129` classical control | was no | yes |
| `t22:250` both arms | **yes** (fixed) | — |

**The one that mattered: T20's Q-HAS draws.** Those 18 completed runs (of
20 launched) underpin the
restated Claim E, and their completion was never verified. Because the arm
is non-deterministic, it **cannot** be verified now — replaying does not
reproduce the draw.

Evidence bounding the risk, short of a re-run: a divergence produces a
partial score wildly out of family with its siblings — the T22 case was
**300×**. The T20 spreads are max/min = 1.5 (`ot`), 2.9 (`tearing`),
2.7 (`rotor`), 3.6 (`kh`), and no draw exceeds phys = 1. All are consistent
with D11's measured CV of 17–49 %, none shows the divergence signature. So
contamination is **unlikely but unproven**.

`t20` now captures the abort marker per run and excludes aborted runs from
the statistics. A verified re-run is queued behind T22b; until it lands, the
Claim E numbers carry this caveat.

### Two smaller findings from the same sweep

**Optuna tuning was clean.** All completed trial values across the three
persisted studies lie in 0.23–0.51 — none at the divergence penalty (10.0),
none above 1. So no fold was tuned against diverged evaluations. The
`catch=(Exception,)` in `study.optimize` is a latent trap (a systematically
failing objective would be silently skipped) but did not fire: zero `FAIL`
states in any study.

**Fold `ot` has weaker tuning provenance than the other three.** It was
tuned before per-trial Optuna persistence existed, so
`t15_level3_optuna_ot.db` does not exist and its per-trial values are
unrecoverable. Its checkpoint carries an explicit provenance note: *"recovered
from logs/v4/level3.log after the container was reclaimed mid-run; QAOA
params printed at 4-decimal precision"*. The other three folds have full
trial-level records.

### Trap sweep, second pass: is the "unseen" condition actually unseen?

The T22 guard checked only that the trajectory *changed*. Two failure modes
slipped through it:

**(a) A diverged DNS would pass.** A trajectory that blows up produces a
huge signature, which reads as "changed". Checked by hand across all four
folds: signature ratios are 0.83–1.08, modest shifts with no blow-up, so
this did not fire. A finiteness test and a physical band (0.05–20) are now
enforced automatically.

**(b) A negligible change also passes.** This one *did* fire:

| fold | trajectory shift at hot start |
|---|---|
| `harris_tearing` | −16.7 % |
| `mhd_rotor` | −15.9 % |
| `kelvin_helmholtz` | +7.5 % |
| **`orszag_tang`** | **−0.3 %** |

`orszag_tang` exposes no initial-condition parameters, so its only available
"unseen condition" is a different Reynolds number — and Re 400 → 600 moves
the hot-start trajectory by **0.3 %**, some 20–50× less than the three
classes where the initial condition itself can be varied.

**Fold `ot`'s transfer test is therefore nearly vacuous** and must be
reported as such rather than counted alongside the other three. T22 now
warns below a 1 % shift and records `unseen_condition_is_weak`; T22c prints
the affected folds and refuses to let them carry a transfer claim.

This is a limitation of V1's API, not of the test: `init_orszag_tang()`
takes no arguments, and `src/` is read-only.

---

## Fresh-eyes review — assumptions re-examined from scratch

Six load-bearing assumptions, re-derived from the source rather than
from memory. Three held, three did not.

### HELD — the ablation is clean

**Both arms differ only in the decision routine.** `classical_only` swaps
`run_adaptive_vqa` → `run_adaptive_classical` on the *same* simulator
object, with the same mapper, `threshold_amr`, `target_dim`, `max_depth`,
`min_size` and TTL map (`pipeline.py:391`).

**Both arms threshold the same score.** `refinement.py:474` (classical) and
`:579` (VQA) both call `AngleMapper.classical_score(physics_state)`. The
QAOA route perturbs exactly the quantity the classical route thresholds, so
the comparison isolates the decision rule and nothing else.

**Both arms are scored at the same physical instant.** With a DNS trace
supplied — the Level-3 case — `dt = dns_trace[step]['dt']`
(`pipeline.py:458`), so both arms march on the DNS time grid and are
compared against the same `dns_trace[last_step]['fluxes']`. The
"adaptive dt desynchronises the arms" trap does **not** fire.

### DID NOT HOLD — three corrections

**(1) `phys_score` is not a plain relative L2.** It is an
*instability-weighted* relative L2: `score()` builds
`w = 1 + 0.25·(|Jz|/⟨|Jz|⟩ + |ω|/⟨|ω|⟩)` from the reference fields and
weights every field's error by it. Every table and figure axis in this
repository has called it "relative L2 vs DNS", which is wrong. Both arms are
scored identically so no bias follows, but the label must be corrected to
**"instability-weighted relative L2 vs DNS"** throughout the manuscript.

**(2) The cost axis excludes the cost of the decision.**
`patch_ratio = total_pixel_used / (steps · N²)` counts refined pixels only.
The QAOA circuit does not appear in it, yet the Q-HAS arm takes **2.7–3.3×**
the classical arm's wall time (ot 1069 s vs 371 s, kh 579 vs 213, tearing
240 vs 73) — on a *simulated* 8-qubit circuit, so hardware would be worse.
"Equal budget" therefore means "equal AMR budget, with Q-HAS's decision
compute free". This makes the conclusion **more conservative**, not less,
but the axis is mis-specified and must be declared.

**(3) T21's ill-posedness claim was overstated — my error.** T21 tested
whether the *count* changes with λ and concluded the endpoint was
ill-posed. Count and verdict are different things. Re-checked over
λ ∈ [0, 100] with `rotor` excluded:

| λ | Q-HAS | classical |
|---|---|---|
| 0.0 – 0.8 | 1 | **2** |
| 1.0 – 100 | 0 | **3** |

**The classical arm holds the majority at every λ tested.** The verdict
never flips; only the margin moves (2–1 → 3–0). The endpoint is *not*
ill-posed in its direction, and saying it was overstated the case. T21 now
separates "margin changes" from "verdict flips" and reports both; the
λ grid was extended to 100 because stability on [0, 5] proves nothing about
[0, ∞).

This correction **strengthens and simplifies** the result: the pre-registered
endpoint, once the failed fold is excluded as its own §5 requires, favours
the classical arm robustly rather than ambiguously.

---

## T22b complete — the transfer signal does not survive replication

56 runs, **zero aborted**, 5 Q-HAS draws per condition per fold, classical
reference budget-matched everywhere.

| fold | deg Q-HAS | deg classical | \|z\| | separable |
|---|---|---|---|---|
| `ot` † | 0.955 ± 0.373 | 0.946 | **0.02** | no |
| `kh` | 1.027 ± 0.509 | 1.364 | **0.66** | no |
| `rotor` | 0.312 ± 0.120 | 0.526 | **1.78** | no |
| `tearing` | 0.166 ± 0.065 | 0.389 | 3.45 | **yes** |

**1 fold of 4.** The single-run pass had suggested Q-HAS transfers
relatively better on *all four* folds (ratios narrowing 0.22→0.17,
2.52→1.81, 3.67→1.88, 2.94→1.01). Repeated with 5 draws, that pattern
evaporates: on `ot` the two arms degrade identically (|z| = 0.02).

† `ot` is unusable for this question regardless: its "unseen" condition
shifts the trajectory by only 0.3 % (no IC parameters exist on
`init_orszag_tang`).

**What holds — the reference-free count:**

| fold | ratio Q/C canonical → unseen | dominated on unseen |
|---|---|---|
| `ot` | 1.48× → 1.50× | 4/5 |
| `kh` | 2.18× → 1.64× | 5/5 |
| `rotor` | 2.48× → 1.47× | 4/5 |
| `tearing` | 3.27× → 1.39× | 5/5 |
| **total** | ratio narrows but never crosses 1 | **18/20** |

> Q-HAS is strictly Pareto-dominated on **18 of 20** runs against initial
> conditions it has never seen — less faithful *and* more expensive.

**Answer to the leakage question.** The concern was well founded but the
mechanism is sharper than "the model saw the end of a trajectory it trained
on":

1. a leak does exist (**D13**) — the QAOA arm's threshold was fitted on all
   four classes including the held-out one — and it **favours Q-HAS**;
2. the initial condition was never new, which T22 fixes;
3. and facing genuinely unseen conditions, Q-HAS remains **strictly
   dominated on 18 of 20 runs** — less faithful *and* more expensive.

On the third point, be precise about what is and is not claimed. Q-HAS's
*relative* degradation is smaller than the classical arm's on the one fold
where the difference is separable (`tearing`, 0.166 against 0.389, |z| =
3.45). That is a real observation and it is **not** evidence that Q-HAS
transfers better: it degrades less from a starting point that was already
worse, and it is still dominated on both coordinates on 5/5 of that fold's
unseen runs. T22d tests the obvious alternative explanation — that both arms
are approaching a common attainable floor — and that confound is not
resolved. So the honest statement is *"Q-HAS is not shown to transfer
better, and remains dominated in absolute terms"*, not *"Q-HAS transfers
worse"*.

So the conclusion does not rest on the leak: Q-HAS loses **despite** an
undue advantage, and loses again on conditions it has never met.

**Still open:** the common-floor confound on `tearing`, the one separable
fold. T22d measures it.

---

## T22d — distance to near-full refinement, all four folds

One classical run per condition at threshold 0.05 (refine almost
everything), the lowest point already swept by t15b's bisections.

| fold | reference can / uns | classical can / uns | Q-HAS can / uns |
|---|---|---|---|
| `tearing` | 0.00397 / 0.00155 | **1.12× / 1.11×** | 3.65× / 1.55× |
| `kh` | 0.00126 / 0.00166 | **1.33× / 1.39×** | 2.90× / 2.28× |
| `rotor` | 0.03395 / 0.02874 | **1.58× / 0.98×** | 3.91× / 1.44× |
| `ot` | 0.01111 / 0.00821 | **7.45× / 9.53×** | 11.04× / 14.28× |

### Three corrections to what I first claimed from this table

**(1) The reference is not a lower bound.** `rotor`'s classical arm scores
**0.98×** on the unseen condition — it *beats* near-full refinement. So
refining almost everything is not always optimal, and this quantity is an
estimate of the achievable optimum, not a certified floor. Any arm below
1.00× is now flagged by the script as proof of exactly that.

**(2) "The classical rule occupies the ceiling" holds on 3 folds, not 4.**
On `ot` **both** arms sit 7–14× above near-full refinement. There is
substantial headroom on that class which neither arm exploits, so the claim
that "there is nothing left for any method to gain" is false there.

**(3) Distance-to-reference is confounded by the operating point.** The
reference refines ~0.95 of the domain; `ot`'s classical arm runs at ~0.37,
`tearing`'s at 0.625. A cheaper operating point is mechanically further from
the full-refinement error, so these distances are **not comparable across
folds**.

### What survives without reservation

Within every fold and on both conditions, **Q-HAS is further from the
reference than the classical arm** — 11.04 vs 7.45, 2.90 vs 1.33, 3.91 vs
1.58, 3.65 vs 1.12. Eight comparisons, eight in the same direction, each one
between two arms at the same operating point on the same trajectory.

That is the only reading these measurements license, and it is enough: at
matched budget the quantum decision rule extracts strictly less of the
available accuracy than plain thresholding, on every class and both under
canonical and unseen initial conditions.

---

## Verified T20 — an aborted run does not always look anomalous

Re-running T20 with the abort marker captured at execution time (the
original pass had no such guard, and being non-deterministic could not be
audited afterwards) produced the finding that most changes how the earlier
numbers must be read.

**Fold `rotor`, Q-HAS draws:**

| draw | phys | status |
|---|---|---|
| 1 | 0.2191 | ok |
| 2 | 0.0978 | ok |
| **3** | **0.6877** | **ABORTED** |
| 4 | 0.0536 | ok |
| **5** | **0.4069** | **ABORTED** |

**Two of five draws diverged — 40 %, not the 1-in-5 I estimated.**

**And draw 5 returned 0.4069, a value that does not stand out.** The valid
draws span 0.054–0.219; 0.407 is high but not absurd. So an aborted run can
land inside the plausible range.

### This retracts my earlier bounding argument

I had written, to bound the risk on the unguarded pass: *"a divergence lands
300× out of family (the T22 case), while T20's spreads are 1.5–3.6× with no
draw above phys = 1 — consistent with D11's CV, no divergence signature.
Contamination unlikely but unproven."*

That reasoning is **wrong**. Contamination need not leave a visible
signature. `rotor`'s original five draws (max 0.2581) could perfectly well
have contained aborted runs, and no inspection of the values would reveal
it. The correct statement is not "unlikely but unproven" — it is
**unknowable without the guard**, which is precisely why the guard had to be
added and the pass repeated.

### A flaw in T20's own control

On `rotor`, **both classical control runs also aborted** (1.1731 twice).
T20 runs its determinism control at the *tuned* threshold, which diverges on
this fold. The control still shows determinism — the divergence reproduces
exactly — but it no longer validates the measurement chain, which is its
purpose. It should run at the budget-matched threshold, as the *reference
value* already does.

### D14 — the fix landed after two of the four folds had started

`always_matched=True` was added to T20's control, and the campaign was
*not* re-run: `ot` and `kh` had already been launched. Their control
therefore replays the **tuned** threshold while their artifact records
`classical_reference_source = "budget-matched classical"`. Both statements
are individually true — the field describes the *reference value*, read
correctly from T15b — but a reader naturally attaches it to the neighbouring
`classical_stats` block, and that block is something else entirely:

| fold | matched thr | replayed thr | matched phys | replayed phys |
|---|---|---|---|---|
| `ot` | 0.1906 | 0.4616 (tuned) | 0.0827 | **0.4845** |
| `kh` | 0.1906 | 0.4616 (tuned) | 0.00168 | 0.00202 |
| `rotor` | 0.0969 | 0.0969 ✓ | 0.05365 | 0.05365 |
| `tearing` | 0.4250 | 0.4250 ✓ | 0.00443 | 0.00443 |

`rotor` and `tearing` agree because the pre-fix code already fell back to
the matched threshold when the tuned arm had aborted.

**On `ot` this is enough to invert the fold.** Against the matched 0.0827,
Q-HAS's 0.1291 is 1.56× worse; against the replayed 0.4845 it is 3.75×
*better*. The published numbers use the matched value and are unaffected,
but anyone recomputing from `classical_stats` — as I did while building T23 —
gets the opposite sign on that fold. The two references are now split into
distinct fields and T23 documents which one is correct.

### D15 — the provenance stamp is taken at the wrong moment

`git_commit_hash()` runs when the artifact is *saved*. A run lasting an hour
is therefore stamped with whatever was committed while it was still
executing. That is exactly how the `ot` and `kh` artifacts carry a hash
postdating the `always_matched=True` commit while having executed the
pre-fix code — the stamp actively pointed away from the truth.

CLAUDE.md requires the commit hash in every output. It is necessary but
**not sufficient for long runs**: the hash must be captured at start, and a
run that spans a commit to its own source should say so.

### Consequence

Every variance figure published from the unguarded pass — the CVs, the
mean-based ratios (1.56×, 1.93×, 2.86×, 2.05×), the gap/sd values — rests on
draws of unknown status. They are superseded by this pass, and on `rotor`
the mean is now computed from **3 valid draws**, not 5.

---

## T20 verified — final numbers, and why the per-fold magnitudes cannot be quoted

All four folds re-run with the abort marker captured at execution time, the
classical control at a non-diverging threshold, and aborted draws excluded
from the statistics.

| fold | valid draws | mean phys | sd | CV | gap/sd | ratio vs matched classical |
|---|---|---|---|---|---|---|
| `ot` | 5/5 | 0.10727 | 0.01823 | 17.0 % | 1.35 | 1.30× |
| `kh` | 5/5 | 0.00320 | 0.00203 | **63.6 %** | 0.75 | 1.90× |
| `rotor` | **3/5** | 0.14725 | 0.04062 | 27.6 % | **2.30** | 2.74× |
| `tearing` | 5/5 | 0.00801 | 0.00193 | 24.1 % | 1.86 | 1.81× |

**Only 1 fold of 4 reaches gap/sd ≥ 2.**

### The magnitudes have now shrunk twice

| fold | first published (1 draw) | unguarded 5-draw mean | **verified 5-draw mean** |
|---|---|---|---|
| `ot` | 2.57× | 1.56× | **1.30×** |
| `kh` | 4.41× | 1.93× | **1.90×** |
| `rotor` | 3.62× | 2.86× | **2.74×** |
| `tearing` | 4.38× | 2.05× | **1.81×** |

### The decisive observation: which fold "passes" is not stable

| fold | gap/sd unguarded | gap/sd verified |
|---|---|---|
| `ot` | 2.09 → **separable** | 1.35 → not |
| `rotor` | 1.56 → not | 2.30 → **separable** |
| `kh` | 0.98 | 0.75 |
| `tearing` | 1.37 | 1.86 |

Both passes report "1 of 4 folds separable" — **but not the same fold**. `ot`
fell below the threshold and `rotor` rose above it. At n = 5 draws, the
separability verdict is itself unstable, which is the clearest possible
evidence that **per-fold magnitude claims are not supportable at this sample
size**. Reporting "Q-HAS is 2.7× worse on rotor" would be reporting a number
whose confidence interval is wide enough to swallow the effect.

**What survives is the direction and the dominance count**, which do not
depend on any single fold's ratio: the verified mean exceeds the
budget-matched classical value on **4 folds of 4** (1.30×, 1.90×, 2.74×,
1.81×), and Q-HAS was strictly Pareto-dominated on 18 of 20 unseen-condition
runs (T22c).

### A robustness asymmetry not captured by any metric

`rotor`'s Q-HAS arm **aborted on 2 of 5 draws (40 %)** while its classical
control at the same budget completed both times, deterministically (0.0536
twice). Across the campaign, 6 Q-HAS aborts have been observed on `rotor`
against 0 for the classical arm at a matched threshold.

None of `phys_score`, `patch_ratio`, the dominance count or the λ analysis
measures this: they all presuppose a run that finishes. The quantum decision
rule produces refinement configurations that destabilise the solver at a
rate the classical rule does not, and that is a distinct failure mode
deserving its own line in the manuscript.

---

## T22 leak-free — D13 removed, and Q-HAS does not survive it

```bash
python study/v4/t22_unseen_conditions.py --fold <f> --mode leak-free \
    --repeats 5 --matched-reference
python study/v4/t24_leak_free_summary.py
```

`--mode leak-free` replaces the QAOA arm's leaked threshold
(`0.14959824837662078`, fitted on all four classes) with the fold's **own
classical tuned threshold**, produced by
`train_classical_threshold_excluding` on the training classes only. The
leak is gone.

### What the mode does not do

It does **not** re-tune the QAOA arm. The definitive experiment puts
`threshold_amr` back into the Optuna search space, excluded from the
held-out class, and is still not attempted. So this measures a **bound**:
*does Q-HAS survive losing the leaked threshold without re-tuning?* — not
*what is the best leak-free Q-HAS?*

### The trap this result had to avoid

The two arms **do not run at the same threshold**. `--matched-reference`
holds the classical control at the budget-matched point, so on `rotor` the
QAOA arm runs at 0.5864 while its control runs at 0.0969. Comparing their
errors directly would confound the decision rule with the budget.

My own code printed *"at the SAME operating point the classical arm
completed"* when `rotor`'s Q-HAS arm died. **That sentence was false** —
the thresholds differ by a factor of six — and it is the campaign's motif
in its purest form: a line of output that does not describe the computation
it accompanies. It now prints both thresholds and says explicitly that they
differ. The artifact carries `qaoa_threshold_amr`,
`classical_threshold_amr` and `thresholds_match`.

The budget-controlled comparison is therefore against the **T15b classical
frontier interpolated at the budget Q-HAS actually realised**, and T24
**refuses to interpolate outside the swept range** rather than let
`np.interp` return an edge value that looks like a measurement.

### Results, all 4 folds

| fold | condition | Q-HAS budget | Q-HAS phys | classical frontier at that budget | ratio |
|---|---|---|---|---|---|
| `rotor` | canonical | — | — | — | **all 5 draws ABORTED** |
| `rotor` | unseen | 0.0882 | 0.8535 | budget below the swept range | not computable |
| `tearing` | canonical | 0.3846 | 3.7351 | 1.7982 | **2.1×** |
| `tearing` | unseen | 0.4232 | 2.5600 | 1.5100 | **1.7×** |
| `kh` | canonical | 0.5513 | 0.02745 | 0.01472 | **1.9×** |
| `kh` | unseen | 0.4646 | 0.13272 | 0.02967 | **4.5×** |
| `ot` | canonical (n=2/5) | 0.2686 | 0.59911 | 0.36638 | **1.6×** |
| `ot` | unseen (n=3/5) | 0.2657 | 0.50405 | 0.36895 | **1.4×** |

**Every fold with a computable ratio puts Q-HAS above the classical
frontier at its own realised budget — 3 of 3, with `rotor` unmeasurable
because it has no operating point at all.**

### Aborts: the sharpest number in the campaign

| fold | Q-HAS aborted | classical aborted |
|---|---|---|
| `rotor` | **7 / 10** | 0 / 4 |
| `ot` | **5 / 10** | 0 / 4 |
| `kh` | 0 / 10 | 0 / 4 |
| `tearing` | 0 / 10 | 0 / 4 |
| **total** | **12 / 40 (30 %)** | **0 / 16** |

Removing the leak costs Q-HAS **30 % of its runs outright**, concentrated
on two folds of four, while the classical arm at its budget-matched
threshold completes every single draw. On `ot` the two arms are visible
side by side: the classical control completes 2/2 deterministically at
budget 0.64, Q-HAS aborts 3/5 and spends 0.27 on the draws that survive.

**Removing the leak makes Q-HAS dramatically worse, and on one fold
inoperable.**

- On `rotor`, **every canonical draw diverges** at the leak-free threshold.
  The arm collapses to a budget of 0.09–0.27 where the classical control
  spends 0.356. Two of five unseen draws also abort.
- On `tearing`, Q-HAS's error rises from 0.0080 (leaked, budget 0.91) to
  **3.735** (leak-free, budget 0.385). Most of that is the budget collapse
  — it refines less than half as much — but **not all of it**: against the
  classical frontier *at its own realised budget* it is still **2.1×
  worse**.
- On `kh`, 10 draws, **zero aborted**. Error rises from 0.0032 (leaked,
  budget 0.870) to **0.02745** (leak-free, budget 0.551) — **1.9×** the
  frontier at its own budget on the canonical condition and **4.5×** on the
  unseen one.

### What `ot` can and cannot contribute, decided before it lands

`ot` is running. Its two halves are **not** equally informative, and that
is fixed by the physics, not by the result:

- its **canonical** half is fully informative — it asks whether Q-HAS
  survives its own fold's leak-free threshold, exactly as on the other
  three;
- its **unseen** half is **nearly vacuous** and must be reported as such.
  `init_orszag_tang()` takes no parameters, so the only available unseen
  condition is a different Reynolds number, which shifts the hot-start
  trajectory by **0.2846 %** — 20–50× less than the other three folds.
  `t22` emits the warning at run time and records
  `unseen_condition_is_weak`.

Stating this now, before the number exists, so that whichever way it falls
it cannot be recruited as a transfer result. If `ot` shows a reversal it
adds nothing to the 3/3 above; if it shows none, that is not evidence
against them.

### `kh` also carries the sharpest transfer reversal

| | leaked | leak-free |
|---|---|---|
| Q-HAS degradation | 1.027 | **×4.835** |
| classical degradation | 1.364 | ×1.364 |
| who degrades more | classical | **Q-HAS** |

Under the leak, `kh` was one of the folds where Q-HAS degraded *less* than
the classical rule on an unseen initial condition. Leak-free it degrades
**3.5× more**. Together with `tearing` (×0.685 against ×0.389, also
reversed) that is **both informative folds reversing in the same
direction** once the leaked threshold is removed.

### The full transfer picture, including the fold that goes the other way

| fold | Q-HAS degradation | classical | Q-HAS worse? | reading |
|---|---|---|---|---|
| `kh` | ×4.835 | ×1.364 | **yes** | reversal |
| `tearing` | ×0.685 | ×0.389 | **yes** | reversal |
| `rotor` | undefined | ×0.526 | — | no operating point |
| `ot` | ×0.841 | ×0.946 | no | **vacuous by construction** |

**`ot` goes the other way and I am not counting it — as pre-registered
above, before the number existed.** Its "unseen" condition shifts the
trajectory by 0.2846 %, so both arms barely move (×0.84 and ×0.95, i.e.
nothing happened to either). That is the outcome the pre-registration
anticipated for a vacuous condition, and the commitment cuts both ways:
this fold was excluded from supporting the reversal, so it cannot now be
admitted to undermine it. The reversal claim rests on `kh` and `tearing`
— **2 of 2 informative folds**, not 4 of 4.

**Run-to-run spread widens too.** `kh`'s leak-free draws give CV 26.3 %
canonical and **64.7 %** unseen, against the 17–49 % band T20 measured for
the leaked configuration. One draw (0.2854 against neighbours near 0.09)
drives most of that — and the divergence guard confirms it **completed**,
`abort = None`, so it stays in. Excluding a valid draw because it looks
inconvenient is the mirror of the defect that contaminated `rotor`'s mean.
At n = 5 with one dominant draw this is a flag for the manuscript, not a
measurement: the leaked threshold appears to have been doing *stabilising*
work, not only accuracy work, which is consistent with `rotor` losing its
operating point entirely.

### Two caveats that must travel with these numbers

1. **The `tearing` frontier is sparse where it matters.** Its swept points
   jump from patch 0.0727 (phys 4.126) to patch 0.6250 (phys 0.00443), so
   the interpolated value at 0.3846 spans a wide, strongly non-linear gap.
   The 2.1× is an order-of-magnitude statement, not a measurement.
2. **`rotor`'s leak-free budget is outside the swept range** (0.056–0.138
   against a frontier starting at 0.152), so no ratio exists for it at all.

### What this settles about D13

The register listed D13 as *"measured, not removed"*, with the note that the
leak favours Q-HAS and the conclusion is conservative because Q-HAS loses
anyway. That is now **measured rather than argued**: with the leak removed,
Q-HAS is not merely still beaten — it is beaten by a wider margin, and on
`rotor` it cannot complete a trajectory at all.

It also **reverses the one transfer result that had favoured Q-HAS**. Under
the leak, `tearing` was the single separable fold and Q-HAS degraded *less*
(0.166 against 0.389). Leak-free, the same fold gives Q-HAS **×0.685
against the classical arm's ×0.389** — Q-HAS now degrades *more*. The
apparent transfer advantage was an artefact of the leaked threshold.

### How these runs survive the container, and what that puts in the artifact

A reviewer will find `resumed_from_checkpoint`, `n_runs_resumed`,
`status: "partial"` and `partial_stage` in these files. They exist because
a leak-free fold costs ~4 h on `kh` and `ot` while this container is
reclaimed roughly every 1.5 h. Two mechanisms, and the second is what
actually made those folds possible:

1. **Checkpoint after every draw.** `t22` writes its state after each
   individual run (~7 min of exposure, not the ~35 min a whole condition
   would cost). Every such write is marked `status: "partial"` with
   `partial_stage` naming the exact draw (`qhas/canonical 3/5`), and
   **both consumers (`t24`, `t22c`) refuse to analyse it** — its arm
   statistics are computed over however many draws finished, which is not
   a result. Without that marking the safety measure would have introduced
   the very defect this campaign documents.

2. **Resume from the checkpoint.** Checkpointing alone only *preserved*
   data: each relaunch restarted from draw 1, so `kh` and `ot` could never
   finish however many times they were run. `t22` now reloads the partial
   artifact and skips the draws already made. It resumes **only** from a
   `partial` record whose fold, mode, `repeats` and `matched_reference` all
   match, and refuses aloud otherwise rather than blending incomparable
   draws; `--no-resume` forces a clean recomputation.

**What resuming does and does not cost.** The reused draws come from a
different process. That has no statistical effect here — the Q-HAS arm is
non-deterministic (D11), the draws are i.i.d., and the classical arm
reproduces bit-exactly — but it is recorded rather than left invisible,
because an artifact that does not say where its data came from is exactly
the failure mode catalogued above. A fold whose `n_runs_resumed` is
non-zero is not weaker evidence; it is evidence that says so.

### Why only 2 folds so far, stated rather than left to be inferred

`ot` and `kh` are the two most expensive folds (T20 spent 3402 s and
3046 s on them respectively, against 2735 s for `rotor`). A leak-free run
is 14 simulations, and this container is reclaimed roughly every 1.5 h —
the campaign has now lost these two folds to reclamation **three times**,
twice as a pair sharing 4 CPUs and once mid-DNS. They are being run one at
a time instead. If they land, this entry gets two more rows; if they do
not, the finding stands on `rotor` and `tearing` and **the sample size is
2 of 4, not 4 of 4**, which is why the closing section says so explicitly.

Nothing about the two completed folds changes either way: they were run to
completion with the abort status captured per draw, and `t16` checks their
numbers (`t24/*` rows).


---

## T25 — robustness to the physics, and the "≥ 3 seeds" requirement

```bash
python study/v4/t25_physics_robustness.py --fold <f> --repeats 3
python study/v4/t25_physics_robustness.py --fold <f> --recompute
```

### First: there is no physics seed to vary

The pre-registration asks for ≥ 3 physics seeds per class, and this study
declared "1 seed per class" as a limitation throughout. **Both statements
are mis-specified.**

| scenario | randomness in its initial condition |
|---|---|
| `init_kelvin_helmholtz` | **none.** `noise_amplitude` multiplies `sin(X)` — a deterministic *mode* |
| `init_harris_tearing` | **none.** `perturbation` multiplies `cos(k·X)` |
| `init_orszag_tang` | **none**, and no parameters at all |
| `init_mhd_rotor` | a real RNG, but `np.random.default_rng(42)` is **hard-coded** |

And the one real seed **does not move the physics**: changing it 42 → 7
shifts the DNS trajectory signature by **0.0022 %**, because the RNG enters
only as `perturbation * standard_normal(...)` with `perturbation = 0.005` —
a symmetry breaker on a field of O(1). So a seed sweep was never possible in
three classes and would have measured nothing in the fourth. **The declared
limitation was not a limitation; it was a non-experiment.**

### What was run instead

The lever that does move the physics is the initial-condition *parameter*.
T25 evaluates each fold on additional initial conditions, comparing Q-HAS
against a classical frontier **built on that same condition** and placed by
bisection on the budget Q-HAS actually realised there.

| fold | condition | trajectory shift | verdict |
|---|---|---|---|
| `rotor` | `rotor_seed7` (true seed 42→7) | 0.0022 % | **vacuous** — skipped |
| `rotor` | `rotor_b` | 21.03 % | **0.86× — Q-HAS BETTER** |
| `tearing` | `tearing_b` | 19.84 % | no verdict — frontier anti-monotone |
| `tearing` | `tearing_c` | 8.16 % | no verdict — budget outside swept range |
| `kh` | `kh_b` | 6.53 % | **1.24× — Q-HAS worse** |
| `kh` | `kh_c` | 3.85 % | no verdict — bisection unconverged |
| `ot` | `ot_re900` (Reynolds, not an IC) | 0.12 % | **vacuous** — skipped |

**7 conditions attempted, 2 vacuous, 3 refused, 2 decidable — one each way.**

### The honest reading

> **On genuinely different initial conditions the direction of the result is
> not established.** It holds on `kh_b` and reverses on `rotor_b`.

This does **not** overturn the closed-loop result, which is measured on the
canonical conditions against T15b's dense bisected frontier with proper
budget matching. It does bound its scope: *Q-HAS is worse on the initial
conditions studied*, not *Q-HAS is worse in general*. Any manuscript claim
must carry that boundary.

### Why three conditions produced no verdict, and why that is reported

On alternative initial conditions the classical relation budget → error is
often **not monotone**: on `tearing_b`, refining from budget 0.625 to 0.874
makes the error **30× worse** (0.012 → 1.289). "The attainable classical
error at budget X" is undefined on such a set, yet `np.interp` answers with
a normal-looking number — and it had already printed **1.28×** as a result.

`frontier_verdict()` therefore refuses unless the bracketing interval is
locally sound: error non-increasing with budget, points within 5×, and the
bisection converged to within twice its own declared tolerance. Each refusal
carries its reason in the artifact.

**Which way the guards cut, stated because it is checkable:** all three
criteria removed evidence *favouring* the study (`tearing_b` 1.28×, `kh_c`
7.02×), and the single result *contradicting* it (`rotor_b` 0.86×) survived
all three. If these filters are biased, they are biased against the claim
this study makes.

### What T25 cannot say

- **Nothing about magnitude** — n = 3 draws per condition, and on `kh_c` two
  draws at the same budget differed by 1.9×.
- **Nothing from an independent seed axis** — it does not exist. The
  physics-robustness evidence rests entirely on parameter variation.
- **Nothing about `ot`** — no IC parameters exist, and its Reynolds lever
  shifts the trajectory 0.12 %.

---

## T26 — l'inertie des couplages est un artefact de PETITE TAILLE

```bash
python study/v4/t26_size_scan.py --dims 2 4 8 --n-snaps 3 --mapper v1
python study/v4/t26_size_scan.py --dims 2 --force-greedy   # contrôle
```

### Pourquoi cette tâche existe

T13 et T18 montrent que les couplages ZZ/ZZZZ changent **exactement 0**
décision, et que réparer la fenêtre n'y change rien. Ces résultats sont
exacts — mais mesurés à `dim = 2`, soit **8 qubits**, précisément le régime
où l'état fondamental est uniforme sur 100 % des instantanés. L'objection
évidente est : *« à 8 qubits, évidemment »*. Elle est fondée, et c'était la
faiblesse centrale de l'étude.

### Résultat

| dim | qubits | méthode | no_ZZ | no_ZZZZ | Z_only | uniformité du fondamental |
|---|---|---|---|---|---|---|
| 2 | 8 | exhaustive | 0.0000 | 0.0000 | 0.0000 | **1.00** |
| 2 | 8 | glouton *(contrôle)* | 0.0000 | 0.0000 | 0.0000 | 1.00 |
| 4 | 32 | glouton | 0.0000 | **0.0312** | **0.0312** | 0.75 |
| 8 | 128 | glouton | **0.0469** | **0.0690** | **0.0794** | **0.17** |

> **L'inertie casse avec la taille.** À 32 et 128 qubits, ablater les
> couplages change des décisions. Et l'uniformité de l'état fondamental
> s'effondre en parallèle : 1.00 → 0.75 → 0.17.

Les deux phénomènes vont ensemble et forment un mécanisme cohérent : tant
que l'optimum est un masque constant, aucun couplage ne peut le déplacer ;
dès que la structure combinatoire apparaît, les couplages redeviennent
causaux.

### ⚠️ Mais « changer une décision » n'est PAS « mieux détecter »

Le tableau ci-dessus mesure l'influence **causale** des couplages, pas leur
**utilité**. La question d'origine du projet est la détection des patches
durs à grossir. Elle se mesure contre la vérité terrain
(`l2_errors >= l2_threshold`), et elle donne :

| dim | qubits | F1 hamiltonien complet | F1 Z seul | F1 règle classique | **gain des couplages** |
|---|---|---|---|---|---|
| 2 | 8 | 0.3333 | 0.3333 | **0.3889** | **+0.0000** |
| 4 | 32 | 0.5199 | 0.5524 | 0.5524 | **−0.0325** |
| 8 | 128 | 0.5916 | 0.6481 | 0.6481 | **−0.0565** |

> **Les couplages ne détectent jamais mieux, et à grande taille ils
> détectent MOINS BIEN.** Quand ils deviennent causalement actifs, leur
> effet est de dégrader le F1 : −0.033 à 32 qubits, −0.057 à 128.

Trois lectures qui en découlent, toutes vérifiables dans la table maîtresse :

1. **Le meilleur cas de la formulation Ising est d'égaler la règle de
   seuil.** À dim = 4 et 8, `F1(Z seul) = F1(classique)` **exactement**
   (0.5524 et 0.6481) : le hamiltonien réduit à son biais reproduit la règle
   classique, terme pour terme.
2. **Ajouter les couplages retire de la performance.** Ils n'apportent pas
   du signal, ils apportent du bruit.
3. **Le F1 monte avec `dim` (0.33 → 0.55 → 0.65) pour les deux bras
   identiquement** — c'est le raffinement du découpage qui aide, pas la
   couche quantique. Attribuer cette montée au quantique serait une erreur
   de lecture.

**Correction d'une formulation antérieure de cette section.** J'avais écrit
que la rupture d'inertie « ouvre un horizon » et était « plus intéressante à
publier qu'un résultat négatif ». C'était prématuré : la frontière existe,
mais de l'autre côté les couplages **nuisent**. Ce n'est pas un horizon,
c'est la fermeture propre de la porte — avec, cette fois, la mesure qui
répond à la question d'origine du projet.

### Le contrôle qui rend ce résultat lisible

L'énumération exhaustive est refusée au-delà de 22 qubits, donc dim ≥ 4
utilise la descente gloutonne à chaud. Le risque évident : que ce soit **le
proxy** qui fabrique les changements, pas les couplages.

Deux garde-fous, tous deux passés :

1. **Le contrôle `full` vaut 0.0000 à toutes les tailles.** Rejouer sans
   ablation redonne exactement la même décision : le glouton est
   déterministe à hamiltonien et amorce fixés, donc tout écart non nul est
   *causé* par l'ablation.
2. **`--force-greedy` à dim = 2** — là où l'exhaustif dit 0.0000 — donne
   également **0.0000**. Le proxy ne fabrique pas de changements dans le
   régime où l'on peut le vérifier.

⚠️ **Réserve à conserver.** Le glouton et l'exhaustif ne choisissent pas le
même masque sur 25 % des cellules à dim = 2 (accord 0.7500), tout en étant
tous deux insensibles à l'ablation. Le scan mesure donc *« les couplages
changent-ils la décision du solveur déployé »*, pas *« l'optimum exact
change-t-il »*. C'est la question opérationnelle — le pipeline n'utilise pas
l'exhaustif non plus — mais elle doit être citée telle quelle.

### Ce que ça change pour les conclusions de l'étude

**Ce qui reste vrai :** à la taille déployée (`VQA_N = 2`, 8 qubits), la
formulation est inerte, et c'est exact.

**Ce qui devient faux :** toute lecture du type *« cette famille de mappings
Ising est intrinsèquement inerte »*. Elle ne l'est pas. Elle l'est **à 8
qubits**, et cesse de l'être avant 32.

**Ce que ça ferme :** l'espoir que la formulation devienne utile en
montant en taille. Les couplages deviennent actifs mais nuisibles, sur toute
la plage testée (8 → 128 qubits). Le meilleur cas de cette famille de
mappings est d'égaler la règle de seuil qu'elle est censée remplacer.

**Ce qui reste ouvert :** la localisation exacte de la transition (entre 8
et 32 qubits ; dim = 3, 18 qubits, serait encore exhaustivement vérifiable
mais demande un DNS à `N` divisible par 3), et surtout **une autre
construction de couplages** — le diagnostic F1 ci-dessus est le test que
toute nouvelle proposition devrait passer avant d'être revendiquée.

---

# CLOSING THE CLOSED-LOOP STUDY (Level 3)

Everything below is measured, carries the control that validated it, and is
covered by `t16_aggregate_v4.py` (**180 rows, 180 OK, 0 DIFF, 0 MISSING**).

## The one-sentence result

> Across four held-out instability classes, a Q-HAS closed loop is less
> faithful than a plain threshold rule at matched compute on **18 of 18**
> completed repeated runs, more expensive on **16 of 18**, and strictly
> Pareto-dominated on **16 of 18**. At that same operating point it also
> **aborts on 2 of 20 draws where the classical rule aborts on 0 of 8**.
> And when the one undue advantage that *can* be taken away — a decision
> threshold fitted on the held-out class (**D13**) — is removed, it does not
> recover: it gets **worse still** on every fold where a comparison is
> possible, and **12 of its 40 leak-free draws fail to complete a
> trajectory at all**, against 0 of 16 for the classical arm.

Each clause is recomputed from its artifact by `t16_aggregate_v4.py`
(rows `t23/*`, `t24/*`). None of it is transcribed.

**Read the abort clause narrowly.** It says the classical arm did not
abort *at the compared operating point*. It does abort elsewhere — T19
records `rotor`'s tuned classical threshold diverging at step 208, and 2 of
that fold's 6 bisection points. Divergence is a property of the threshold;
both arms have thresholds that diverge. What is asymmetric is that at the
point where they are compared, one arm completed and the other did not.

**Scope boundary, from T25.** Everything above is measured on the
**canonical initial conditions**. On genuinely different initial states the
direction is **not established**: of 7 alternative conditions, 2 were
vacuous, 3 gave no sound verdict, and the 2 decidable ones split one each
way (`kh_b` 1.24× for, `rotor_b` 0.86× against). The claim is therefore
*Q-HAS is worse on the initial conditions studied*, not *in general*. And
the pre-registered "≥ 3 physics seeds" was never available: three of four
scenarios have no RNG at all, and the fourth's hard-coded seed moves the
trajectory by 0.0022 %.

**The D13 clause is measured on all 4 folds**, and it is a *bound*:
`--mode leak-free` substitutes the threshold without re-tuning the QAOA
arm. The definitive version — `threshold_amr` back in the Optuna search
space, excluded from the held-out class — is not attempted. What the bound
says: Q-HAS above the classical frontier at its own realised budget on
**3 of 3 measurable folds** (1.6×, 1.9×, 2.1× canonical), **no operating
point at all** on the fourth, and **12 of 40 draws aborting against 0 of
16** for the classical arm.

## What the closed loop establishes, by strength of evidence

**1. Direction — robust, no free parameter.** The verified Q-HAS mean
exceeds the budget-matched classical value on **4 folds of 4** (1.30×, 1.90×,
2.74×, 1.81×). The pre-registered `combined` endpoint gives the classical arm
the majority at **every λ on the swept grid** (12 points, 0 → 100) — the
verdict never flips, only the margin: 2–1 from λ = 0 through λ = 0.8, then
3–0 from λ = 1.0 onward. An earlier draft put the crossover at "λ = 0.82";
that precision is not available from a 12-point grid — all that is measured
is that the count changes somewhere in (0.8, 1.0]. The verdict, which is
what the claim rests on, does not change anywhere. Two of three usable folds
are decided by Pareto dominance alone, needing no λ at all.

**2. Robustness — a failure mode outside every metric.** `rotor`'s Q-HAS arm
aborted on **2 of 5** verified draws (40 %) while its classical control **at
the budget-matched threshold** completed every time, deterministically.
Across the recorded T20 and T22 artifacts: **2 Q-HAS aborts out of 20 draws,
0 classical aborts out of 8 replays at the matched point.** `phys_score`,
`patch_ratio`, the dominance count and the λ analysis all presuppose a run
that finishes.

**Do not read this as "the classical rule never diverges" — it does.** The
T19 audits record `rotor`'s *tuned* classical arm aborting at step 208
(threshold 0.4616), and 2 of `rotor`'s 6 bisection points aborting as well.
An earlier draft of this section claimed "six Q-HAS aborts against zero
classical across the campaign"; the second half of that is false and the
first is not reproducible from the artifacts, which record 2. The claim that
holds is narrower and is the one the comparison actually needs: **at the
operating point where the two arms are compared, the classical arm completed
every time and Q-HAS did not.** Divergence is a property of the threshold,
and both arms have thresholds that diverge.

**3. Transfer — no effect, and the one apparent effect was the leak.** On
genuinely unseen initial conditions, **1 fold of 4** shows a separable
difference in degradation; on `ot`, |z| = 0.02. The single-draw pass had
suggested Q-HAS transfers *better* on all four; repeated with 5 draws that
pattern evaporates.

The one fold that survived as separable was `tearing`, and it favoured
Q-HAS (degradation ×0.166 against the classical ×0.389). **Leak-free, that
reverses**: ×0.685 against ×0.389 — Q-HAS now degrades *more*. The single
transfer result in the study's favour was an artefact of the leaked
threshold, and removing the leak removes it. Nothing here supports a
transfer advantage in either the leaked or the leak-free setting.

**4. Magnitudes — not supportable.** Both variance passes report "1 fold of
4 separable" **but not the same fold** (`ot` 2.09 → 1.35, `rotor` 1.56 →
2.30). At n = 5 the separability verdict is itself unstable. Quote the
direction and the counts; **do not quote per-fold ratios**.

## Conditions under which the result was obtained — all adverse to the classical arm

The conclusion is **conservative**: three known asymmetries favour Q-HAS and
it loses anyway.

| asymmetry | direction | status |
|---|---|---|
| **D13** — QAOA threshold fitted on all 4 classes incl. the held-out one | favours Q-HAS | **removed and measured** (T22 `--mode leak-free`): without it Q-HAS is 2.1× worse than the classical frontier at its own budget on `tearing`, and aborts on 5/5 canonical draws on `rotor` |
| **cost axis** excludes the QAOA circuit; Q-HAS uses 2.7–3.3× the wall time on the three folds whose classical arm completed (`rotor` excluded: its 29 s classical run is the aborted tuned arm, not a comparable time) | favours Q-HAS | declared |
| aborted Q-HAS draws excluded from its own statistics | favours Q-HAS | necessary, declared |

## What would overturn it

- ~~removing D13 and finding Q-HAS wins~~ — **done, and it goes the other way**: leak-free, Q-HAS is worse still (2 folds of 4 measured so far). What would overturn the result is the *definitive* version, re-tuning the QAOA arm with `threshold_amr` in its Optuna search space on the training classes, which is not attempted;
- ≥ 3 physics seeds per fold showing the direction is seed-specific;
- the full 170-trial Optuna budget lifting Q-HAS above the matched classical;
- counting decision cost, which would only make the result stronger.

## What this study cannot say

- **Nothing about magnitude** per fold (n = 5, unstable separability).
- **Nothing about transfer on `ot`** — its unseen condition shifts the
  trajectory by 0.3 %, `init_orszag_tang()` taking no parameters.
- **Nothing about hardware**: the circuit is simulated, 8 qubits, noiseless.
- **Nothing about larger `VQA_N`**: everything here is the deployed depth-0
  size where the ground state is uniform (Claim A).

## The methodological finding, stated for the manuscript

**Seventeen distinct instances** of one failure mode were found and fixed:
**a computation that fails, or does not do what it says, but returns a value
indistinguishable from a valid one**. Twelve were found by auditing code:

| form | count | where |
|---|---|---|
| V1's divergence guard returns a partial score with identical keys | 4× | T15, T20, T22 (×2) |
| a fixed output filename silently overwrites the prior result | 6× | T13 mappers, T19 folds, T20 pass, then T11, T11b, T12 (`--mapper` absent from the name) |
| an aggregation averaging aborted draws with valid ones | 1× | T16 |
| a CLI mode accepted and documented but never implemented | 1× | `--mode no-leak`: only the filename changed |

**Four of the twelve were in the verification code written to catch the
others**, and three more were found only by `tests/study/test_silent_failure_sweep.py`,
which sweeps the mechanically checkable forms. Searching as you go is
demonstrably not enough.

### Five more, found by auditing the documents against the artifacts

The twelve above were found by auditing *code*. A final pass audited the
**published numbers** instead — recomputing each from its artifact — and
found five more instances, in the write-up and in the verification code:

| # | instance | consequence |
|---|---|---|
| 13 | a total abort discarded before saving: `SystemExit` fired before any artifact was written, and on the *first* arm, so the question that mattered (does the classical rule survive that threshold?) went unmeasured | the mirror of the motif — a real outcome made indistinguishable from a run never launched |
| 14 | **the headline count was written by hand**, not computed. 19/20, 18/20, 17/20 did not reproduce: `kh`'s two columns were transposed and `rotor`'s 2 aborted draws sat in the denominator | the study's most-quoted number was wrong; correct is **18/18, 16/18, 16/18** |
| 15 | **D14** — T20's artifact says `classical_reference_source = "budget-matched classical"` next to a `classical_stats` block that, on `ot` and `kh`, was computed at the *tuned* threshold | 0.4845 against 0.0827 on `ot` — enough to invert that fold for anyone recomputing from it |
| 16 | **D15** — `git_commit_hash()` taken at *save* time | hour-long runs stamped with code committed while they ran; the `ot`/`kh` artifacts point at a fix they never executed |
| 17 | `t22` printed *"at the SAME operating point the classical arm completed"* in leak-free mode | the two arms differ by a factor of six in threshold; the sentence would have turned a budget difference into an arm-specific instability claim |

Two more errors of a related but distinct kind — **false precision** rather
than false results — were fixed in the same pass: a λ crossover quoted as
"0.82" from a 12-point grid that only locates it in (0.8, 1.0], and a
published figure still annotating the retracted single-draw ratios.

**The pattern is the finding, and it is sharper than "check your code".**
Every number that no script produced turned out to be wrong. Every number
`t16_aggregate_v4.py` recomputes from its artifact was right. The defence
that worked was not care, review, or re-reading — all of which were applied
throughout and all of which missed these — but **making the number a
function of the artifact and checking it mechanically**. Anything published
as prose is unverified by construction.

One aborted draw returned `phys = 0.4069` against valid draws of
0.054–0.219: **contamination need not be visible in the values**.

**And its direction cannot be bounded either.** On `ot` leak-free the three
aborted draws returned 0.4311, 0.4239 and 0.4529 while the one draw that
*completed* returned **0.6587** — the invalid runs looked **better** than
the valid one. The mechanism is plain once seen: those runs stopped near
step 930 of ~1136, so the trajectory had less time to depart from the DNS
reference and accumulated less error. On `rotor` the opposite happened,
because there the abort came *after* the fields blew up.

So the tempting bounding argument — *"an aborted run scores badly, so
including it is conservative"* — is **empirically false**. Contamination
inflates the error when the blow-up is captured and deflates it when the
run is merely truncated, and which one you get depends on where the guard
fires. There is no safe direction to assume, which is why the status has to
be captured at execution time rather than inferred from the value. Any
closed-loop AMR study of this kind should record run completion status at
execution time, because with a non-deterministic arm it cannot be recovered
afterwards.

---

# THE V1 TEST SUITE, RE-ARMED

Base commit `d3d8fe6`. Commands:

```bash
python -m pytest tests/ --ignore=tests/study -q
python -m pytest tests/study -q
```

## Before: 44 of 175 tests were failing, and no green gate existed

```
44 failed, 131 passed in 258s
```

**42 of the 44 had a single cause**, and it was mechanical:

```
TypeError: PhysicalMapper.__init__() got an unexpected keyword argument 'beta'
```

`beta` was split into `beta_curl` / `beta_xpoint` (`src/Simulation/HamiltParams.py:63`)
and the call sites in `tests/` were never updated. The code was not broken —
the tests were stale. The consequence is what matters: `test_beta_xpoint.py`,
`test_vqa_anomaly_cases.py`, `test_module_validation.py` and the four
`test_qaoa_*` files **had verified nothing since that refactor**, i.e. the
Hamiltonian layer — the object of the whole study — was unguarded.

`run_tests.sh` is `set -e` and `run_stage` exits on the first non-zero code
(`run_tests.sh:154`), so the default run aborted at **stage 2**
(`test_v9_metrics.py`). There was no passing gate on V1 to regress against.

Repair: `beta=X` → `beta_curl=X, beta_xpoint=X` at 18 call sites, which is
the exact historical semantics (a shared `beta` fed both sensitivities,
`HamiltParams.py:88-92`). No file under `src/` was touched.

## After

```
175 passed          (V1 suite)
325 passed, 15 skipped   (tests/v3 + tests/v4)
```

## The six assertions that had to be inverted, and why

Two failures were not stale — they were **correct measurements of a broken
claim**. Four more of the same kind surfaced once the 42 came back to life.
All six assert that a coupling is present; all six measure its annihilation
by the Gaussian uncertainty window `exp(-((score - threshold_amr)/sigma)^2)`
that multiplies `C_edges`.

The clearest of them, `test_v9_metrics.py`, carried this docstring:

> *"This is the core v9 claim: the Hamiltonian adds spatial correlation
> information BEYOND what θ init provides."*

and failed by 42 orders of magnitude. Measured, on a 2x2 periodic grid with
a sharp velocity boundary (`score` uniform at 0.5, `threshold_amr = 0`,
`sigma = 0.05`):

| quantity | value |
|---|---|
| `max abs(C_edges)` delivered | **1.7858e-42** |
| same call at `threshold_amr = score` (window = 1) | **4.8005e+01** |
| ratio | **3.7201e-44** |
| `exp(-((0.5 - 0)/0.05)^2) = exp(-100)` | **3.7201e-44** |

The ratio equals the window to full double precision. The gradient signal is
computed correctly, at O(48), and then multiplied by ~1e-44.

On Orszag-Tang (N=64, 30 steps, score spanning [0.5057, 0.8748]):

| sigma | `max abs(C_edges)` | `max abs(K_plaquettes)` |
|---|---|---|
| 0.05 (deployed) | **1.7727e-48** | 2.3629e+01 |
| 10 (window open) | **6.3187e+01** | 2.3629e+01 |

`K_plaquettes` is bit-identical across the two, which is what makes the
attribution airtight: `sigma` reaches ZZ and nothing else. The four cases in
`test_vqa_anomaly_cases.py` give 1.79e-42, 1.86e-42, 1.11e-38 and 1.23e-85
by the same mechanism.

Each of the six now asserts three things instead of one: the delivered
coupling is dead (`< 1e-30`), the same fields with the window open return an
O(1) coupling, and — where the score is uniform enough to make it exact —
the ratio equals the window. A test that merely recorded "it is zero" would
not distinguish *annihilated* from *never computed*.

**This is an independent corroboration of T13/T17/T18, written before this
study existed.** V1's own unit tests contained the falsification of V1's
central claim, in red, for the whole life of the project.

## Three defects found while re-arming, none previously recorded

**(a) The Z-bias scale is a function of the threshold** —
`test_qaoa_physics_decision.py`. `H_edges` is documented as
`alpha_z * (score - threshold_amr)`. It is linear in `score` at fixed
threshold (the recovered ratio is constant to 1e-9), but `alpha_z` is
normalised by `median(nonzero |C|, |K|)`, and `|C|` carries the window — so
`alpha_z` inherits the threshold dependence. On a shear layer whose score
takes exactly two values, 0 and 0.5:

| `threshold_amr` | `max abs(C_edges)` | recovered `alpha_z` |
|---|---|---|
| 0.20 | 1.167e+01 | **8.7857e-01** |
| 0.50 | 4.404e-10 | 1.4930e-03 |
| 0.95 | 2.396e-84 | 5.0750e-03 |

Same fields, same score, Z-bias scale moving by **173x** and
non-monotonically with the threshold alone. The old test asserted
monotonicity and was simply wrong about the model it was testing.

**(b) The vortex detection test was measuring shot noise.** With
`args.shots = 4096` each marginal carries a standard error of ~0.008. Over
12 draws on identical fields, the Lamb-Oseen contrast was

```
[+0.0141 -0.0147 -0.0267 -0.0043 -0.0305 +0.0060
 +0.0036 -0.0125 +0.0067 +0.0236 -0.0084 +0.0079]
mean = -0.0029, std = 0.0156
```

centred on zero with a **sign that flips run to run**, and clearing the old
`abs(contrast) > 0.01` bar on exactly 50% of draws. The test now runs 10
draws and asserts the mean is null and the sign is not reproducible — which
is the finding, and is consistent with the uniform ground state at this size.

**(c) The QAOA arm's displacement is not a single-draw quantity.** The
max-marginal displacement against `sin^2(theta/2)` ranged over
**0.0721 to 0.4742** across 12 identical calls (mean 0.2867). The assertion
is now on the median of 5 draws. Same root cause as D11: unseeded COBYLA
plus a shot-based sampler.

## The harness finding: 8 of the 17 default stages cannot fail

Independent of the 44, and larger:

| stage | assertions | wall time |
|---|---|---|
| `tests/test_qaoa_noise_and_early.py` (2 tests) | **0** | 14m40 + 1m38 |
| `tests/test_qaoa_scaling_and_hparams.py` (2 tests) | **0** | 16m04 |
| `tests/test_qaoa_advantage.py` | **0** | script |
| `tests/test_qaoa_decisions.py` | **0** (0 test functions) | script |
| `tests/diag_hamiltonian_balance.py` | **0** | script |
| `tests/diag_qaoa_contribution.py` | **0** | script |
| `tests/diagnose_convergence.py` | **0** | script |

They print and return 0. `run_stage` reports `PASSED`. Over **32 minutes**
of the default run is spent in files that contain no assertion at all — and
what they print is not neutral:

- `test_qaoa_advantage.py` ends with the winner column reading `Classical`
  on **6 of 6** rows (rotor 2x2/3x3, KH 2x2/3x3, OT 2x2/3x3) and exits 0;
- `diag_qaoa_contribution.py` ends with
  `⚠ ALL Z biases negative → QAOA ground state = refine nothing` and exits 0;
- `test_noise_robustness` averages Spearman rho values that are **NaN** on
  some trials (`ConstantInputWarning: An input array is constant`) without
  saying so.

This is the study's own motif at the level of the harness: *a stage that
verifies nothing is indistinguishable from a stage that passed*. The 44 red
tests were visible; these eight were green.

**Not fixed here**, because it changes the meaning of the gate and the
acceptance criteria would have to be invented rather than measured: either
give those stages real assertions, or move them out of the default path into
the existing `--figures` / `--diagnose` groups so the default run is
assertion-bearing end to end.

---

# THE EIGHT STAGES THAT COULD NOT FAIL, AND WHAT THEY SAY NOW

Base commit `fe1f6fe`. Nothing under `src/` was modified; the source
behaviours below are pinned from the test side.

## Every default stage now carries an acceptance check

| stage | it now asserts | reference |
|---|---|---|
| `test_qaoa_advantage.py` | QAOA outranks the classical baseline on at most 1 of 6 scenario/size pairs, and the mean rank-correlation gap exceeds 0.15 | 0/6 wins, gap **+0.692** |
| `test_qaoa_decisions.py` | the 7 internal checks match their recorded pattern exactly | **5 hold, 2 known defects** |
| `test_qaoa_noise_and_early.py::test_noise_robustness` | without noise the classical arm reaches the optimum and QAOA loses by > 0.10 captured fraction; QAOA wins at most 4 of 12 rows, none below sigma = 0.20; a NaN rho occurs only when a score map is constant | 0.6588 vs 0.3350 and 0.3183 vs 0.1976; 2/12 wins, both at sigma = 0.30 |
| `test_qaoa_noise_and_early.py::test_early_detection` | QAOA wins at most 2 of 6 rows and never exceeds the classical mean captured fraction by more than 0.02 | 1/6 wins; means **0.4065 vs 0.3735** |
| `test_qaoa_scaling_and_hparams.py::test_resolution_scaling` | on clean data QAOA never exceeds the classical arm at N = 32, 64, 128, and the classical arm improves with resolution | 0.5182 / 0.6588 / 0.7669 classical, QAOA 0.5182 / 0.6588 / 0.2438 |
| `test_qaoa_scaling_and_hparams.py::test_hyperparameter_sweep` | over 4 w_z_frac x 3 thresholds, the best result on clean data is an exact tie with the classical baseline | best delta = **+0.0000**; 4 exact ties at threshold 0.3, **-0.4048** everywhere else |
| `diag_hamiltonian_balance.py` | the downsampled ZZ block does not move with beta_curl/beta_xpoint, no ZZZZ survives downsampling, and Z/ZZ magnitude stays below 1e-3 | max abs(K) = 0 exactly; max abs(H) ~ 9.6e-05 against max abs(C) ~ 1.0031 |
| `diag_qaoa_contribution.py` | at the operating threshold the QAOA flips at most 2 of 48 decisions, every run at threshold 0.5 has all-negative Z biases, and the multi/single energy ratio exceeds 1e4 everywhere | **0/48 flipped**; 12/12 all-negative; ratio 6.4e4 to 6.2e8 |
| `diagnose_convergence.py` | its own four printed verdicts become the exit code | B1-B4 all PASS |

The most quotable line of that table is the hyperparameter sweep: **the best
the QAOA arm ever does on clean data, over the entire sweep, is to equal the
classical baseline exactly.** Twelve combinations, one ceiling, and it is a
tie.

`test_qaoa_advantage.py` and `diag_qaoa_contribution.py` were printing
`Classical` on 6 of 6 rows and
`ALL Z biases negative -> QAOA ground state = refine nothing` respectively,
and exiting 0. Those two lines are now the acceptance criterion instead of
decoration.

## The placeholder Hamiltonian is now detectable — `tests/test_v1_guards.py`

`cost_hamiltonian.py` drops every coefficient below **1e-6** and, when that
empties the term list, appends `("Z", [0], 1e-3)` so Qiskit does not choke on
an empty observable. Three properties are now pinned:

1. **the substitute is 1e6 times the signal it replaced** — with every
   coefficient at 1e-9, the operator delivered to the solver is a single term
   at 1e-3;
2. **it is not physically neutral**, contrary to the source comment. Every
   ground state of `("Z", [0], +1e-3)` has qubit 0 excited, with
   `E_min = -1e-3`: the placeholder is a *refine-edge-0* bias;
3. **it escapes the null-Hamiltonian shortcut.** `execute.py:52` skips COBYLA
   when `np.allclose(abs(coeffs), 0.0)`, whose default `atol` is 1e-8. The
   placeholder sits at 1e-3, so a patch with no surviving coefficient runs a
   full variational optimisation against a fabricated operator. The two
   thresholds live in different files and nothing else connects them.

`is_null_placeholder(op)` is the detector to call before interpreting any
operator coming out of V1: a placeholder means *no Hamiltonian was built*,
which is a different event from *the Hamiltonian is weak*.

The same file pins the pruning chain — `max abs(C_edges)` is nonzero and
below 1e-6 on a real 2x2 patch, and **zero ZZ terms** appear in the operator,
while a coupling above the cut produces one ZZ term per site — and exercises
the assignment that `execute.py:182-185` performs inside
`try/except Exception: pass`, on both primitive construction paths, so that a
silently under-sampled MPS readout fails here instead of hiding there.

## Four more V1 claims that were false

Re-arming the suite made these visible; each is measured over repeated draws
because the arm is stochastic.

| claim as written | measured | n |
|---|---|---|
| `test_signal_contribution::test_psi` — "phase anticipation": high psi marks a growing instability | contrast **-0.0572** (t = -8.4), negative in 93% of draws — psi LOWERS the cell it marks | 30 |
| `test_qaoa_physics_decision::test_spatially_varying_psi...` — same mechanism, different construction | **-0.0723** (t = -14.6), positive in 3% of draws | 30 |
| `test_signal_contribution::test_K_ZZZZ` — a 6x stronger plaquette should raise its four qubits | **-0.0168** (t = -7.1) — it lowers them | 30 |
| `test_signal_contribution::test_C_ZZ` — a 10x stronger ZZ coupling should raise its edge | **+0.0072**, sem 0.0049, **t = +1.46** — indistinguishable from zero | 30 |

The two psi rows are the same finding reached from two independent setups:
**the "phase boost", which is the mechanism the early-detection story rests
on, has the opposite sign to the one claimed.** Both old assertions took the
absolute value of the contrast, which is exactly why the sign was never seen.

The C_ZZ row belongs with T13/T18/T26: a coupling ten times the background
moves nothing measurable at the deployed size.

## Six single-draw assertions on a stochastic arm

Beyond the four above, these were passing or failing by luck. All are now
stated over repeats, and the magnitude threshold was replaced by a *sign*
criterion wherever the mean itself drifts between sessions (unseeded COBYLA):
one run of `test_psi` returned -0.0183 where another returned -0.0572, while
the sign held in both.

| test | old assertion | draws clearing it | now |
|---|---|---|---|
| `QAOA_test::test_vortex_discriminates` | single draw, abs(contrast) > 0.01 | **25%** | mean over 8 draws is not positive (recorded -0.0058 +/- 0.0064) |
| `test_qaoa_physics_decision::test_vortex_detected` | single draw, abs(contrast) > 0.01 | **50%** | mean over 10 draws null, sign not reproducible |
| `test_qaoa_physics_decision::test_qaoa_converges_for_simple_hamiltonian` | single draw, avg P(1) > 0.7 | **90%** | median of 5 draws (mean 0.829, min 0.676) |
| `test_qaoa_physics_decision::test_qaoa_modifies_probabilities...` | single draw, max diff > 0.05 | ~92% | median of 5 draws (range 0.0721 to 0.4742) |
| `test_signal_contribution::test_H_Z` | single draw, contrast > 0.01 | ~95% (min -0.018) | mean over 20 draws > 0.02 |
| `test_signal_contribution::test_K_ZZZZ` | single draw, abs(contrast) > 0.01 | **87%** | sign over 20 draws |

## Gate

```
184 V1 tests pass (175 repaired + 9 new guards), four consecutive runs
325 v3/v4 tests pass, 15 skipped
9 of 9 default script/pytest stages carry an acceptance check
```

---

# V1 NE FABRIQUE PLUS D'HAMILTONIEN QUAND IL N'Y EN A PAS

Modification de `src/` (première depuis le gel de V1), commit parent `32d124a`.

## Ce qui change

`cost_hamiltonian.py` élague tout coefficient sous `COEFF_MIN = 1e-6`.
Quand il ne reste rien, il ajoutait `("Z", [0], 1e-3)` pour éviter le crash
Qiskit sur observable vide. Il lève désormais **`NullHamiltonianError`**.

`execute.py:184` : le `try/except Exception: pass` autour de
`sampler.options.default_shots = mps_shots` est supprimé. Si l'affectation
échoue, la lecture MPS tournerait au mauvais nombre de tirs.

`refinement.py` attrape l'exception, **conserve la décision classique** du
patch, et l'enregistre dans `null_hamiltonian_patches()`. Le VQA n'est pas
appelé. C'est un changement de comportement assumé : l'ancien chemin faisait
tourner COBYLA contre un opérateur dont l'état fondamental excite le qubit 0.

## Ce que la levée d'erreur a révélé immédiatement

Trois tests de V1 comparaient une anomalie à une « ligne de base calme ». Ils
échouent maintenant, et la raison est le résultat :

| champ | Hamiltonien construit ? | max abs(H) | max abs(C) | max abs(K) |
|---|---|---|---|---|
| cisaillement | oui | 1.670e+00 | 1.786e-42 | 2.227e+01 |
| **calme (vx = 1.0)** | **non** | — | — | — |
| point X | oui | 3.462e+00 | 8.518e-86 | 4.300e+01 |
| **calme (vx = 0.01)** | **non** | — | — | — |
| combiné | oui | 2.392e+00 | 1.113e-38 | 2.328e+01 |
| **calme (vx = 0.0)** | **non** | — | — | — |

Les trois lignes de base n'avaient **aucun** coefficient au-dessus de 1e-6.
Elles recevaient le terme de remplissage, et l'écart de marginales mesuré
contre elles — l'assertion « le cisaillement produit une réponse VQA
différente du calme » — était un écart contre un opérateur fabriqué.

L'énoncé correct est plus net : **sur un champ uniforme, la construction ne
produit rien à optimiser, et elle le dit.** Les trois tests l'affirment
maintenant ainsi, plus le contrôle que le champ anormal, lui, définit bien un
Hamiltonien.

`test_module_validation::test_zero_coefficients_filtered` testait
explicitement l'ancien comportement (« Should only have the safety term ») ;
il teste la levée, plus le contrôle qu'un seul coefficient au-dessus du seuil
suffit à construire l'opérateur.

## Gate

```
185 tests V1 (180 + 10 gardes, dont un bout-en-bout sur refinement.py)
325 tests v3/v4, 15 skipped
diag_qaoa_contribution.py : 0/48 décisions changées, exit 0
```

---

# La convention d'axes des mappeurs — T31

## Le fait, plus précis que « la vorticité est fausse »

`grid.py:4-13` déclare la convention `indexing='ij'` : `AXIS_X = 0`,
`AXIS_Y = 1`. Le solveur la respecte (`grid.grad`, `grid.div`,
`grid._compute_q_criterion`, `MHDSolver.get_fluxes`). Les trois mappeurs ne
la respectent pas : `HamiltParams.py`, `HamiltParams_v2.py` et
`PhysToAngle.py` forment leur rotationnel et leur divergence avec les axes
échangés.

Ce n'est pas une faute de frappe sur un signe, c'est une **convention
différente appliquée de façon cohérente** : les formules des mappeurs sont
exactement celles qu'on écrit sous `indexing='xy'`. Sous la convention que le
dépôt déclare, elles valent

| nom dans le code | ce qui est réellement calculé |
|---|---|
| `vorticity`, `omega_z` | ∂v_y/∂y − ∂v_x/∂x — différence des déformations normales |
| `div_v` | ∂v_x/∂y + ∂v_y/∂x — déformation de cisaillement (2·S₁₂) |
| `Jz_curl` | ∂B_y/∂y − ∂B_x/∂x |

Autrement dit les deux indicateurs nommés « vorticité » et « divergence »
sont deux composantes du tenseur des déformations.

## Mesuré sur des champs à réponse analytique

`tests/test_analytic_fields.py`, champs linéaires, exact à 1e-12 sur
l'intérieur du domaine (le raccord périodique fausse une cellule par bord) :

| champ | ω_z attendu | ω_z mappeurs | ∇·v attendu | ∇·v mappeurs |
|---|---|---|---|---|
| rotation solide `vx=−y, vy=x` | +2 | **0** | 0 | 0 |
| cisaillement pur `vx=y, vy=0` | −1 | **0** | 0 | **+1** |
| expansion pure `vx=x, vy=y` | 0 | 0 | +2 | **0** |
| déformation pure `vx=x, vy=−y` | 0 | **−2** | 0 | 0 |

L'indicateur de vorticité est **exactement nul sur une rotation solide** et
vaut −2 sur un champ de vorticité nulle. L'indicateur de divergence est
exactement nul sur une compression isotrope et vaut +1 sur un champ de
divergence nulle.

## Deux défauts de plus, dans le critère Q — celui-là sur les bons axes

`grid._compute_q_criterion` utilise `AXIS_X`/`AXIS_Y` correctement, mais
pondère la déformation de moitié : `strain_sq = S₁₁² + S₂₂² + 2·S₁₂²` vaut
(S_n² + S_s²)/2, alors que la forme d'Okubo-Weiss demande S_n² + S_s².
Conséquences, exactes elles aussi :

| champ | Okubo-Weiss standard | `_compute_q_criterion` |
|---|---|---|
| cisaillement pur | 0 (frontière rotation/déformation) | **+0.25** → lu « dominé par la rotation » |
| expansion pure | 0 (ni rotation ni déformation déviatorique) | **−1** → lu « dominé par la déformation » |

Le second vient de ce que `S₁₁² + S₂₂²` retient la partie isotrope du
tenseur. Les deux sont épinglés par un test dédié.

## Ce que la correction changerait : la variante `--fixed-curl`

Le chemin par défaut n'est pas touché. `fixed_curl=False` est le défaut des
trois mappeurs et de `prepare_qaoa_inputs` ; sa sortie est **bit-à-bit** celle
d'avant, vérifié sur 64 tableaux (score classique, score physique,
coefficients V1 et V2) × 4 scénarios à N=64. Une seule association
arithmétique a dû être conservée telle quelle : réécrire
`vx − roll(vx) + roll(vy) − vy` sous une forme algébriquement identique
déplaçait le dernier bit (écart 8.0e-15 sur `K_plaquettes`, mhd_rotor).

`--fixed-curl` applique la convention déclarée et suffixe ses artefacts
`_fixedcurl`, donc les deux variantes ne peuvent pas s'écraser.

## Le résultat, avec ses intervalles

```
python study/h1_solver/h1_curl_convention_gap.py --N 128 --dim 8  --n-snaps 6 --seed 0
python study/h1_solver/h1_curl_convention_gap.py --N 128 --dim 16 --n-snaps 6 --seed 0
```
git 8ee5c8a — 4 scénarios × 6 instantanés = 24 lignes, IC95 rééchantillonné
**par scénario** (le bloc est la trajectoire, pas l'instantané).

La décision au seuil entraîné est inexploitable : à 0.1496 le score de patch
sature et les deux bras dégénèrent en « tout raffiner » (9/24 lignes
dégénérées à dim=8). La comparaison porte donc sur le **classement**, à
budget apparié — les deux bras raffinent le même nombre de patches et ne
diffèrent que par lesquels.

| dim | métrique | historique | corrigé | Δ | IC95 | verdict |
|---|---|---|---|---|---|---|
| 8 | Spearman vs dureté | +0.7266 | +0.7237 | −0.0029 | [−0.0222, +0.0164] | indécidable |
| 8 | F1 budget apparié | 0.7396 | 0.7786 | +0.0391 | [−0.0156, +0.0938] | indécidable |
| 16 | Spearman vs dureté | +0.7896 | +0.7231 | −0.0665 | [−0.1328, −0.0146] | **dégrade** |
| 16 | F1 budget apparié | 0.8522 | 0.8223 | −0.0299 | [−0.0651, +0.0052] | indécidable |

Écart maximal sur le score : 0.318 (dim 8), 0.397 (dim 16) ; accord des
décisions 0.921 dans les deux cas. La convention change donc réellement les
entrées — mais pas dans le bon sens.

## Ce qu'il faut en conclure

**Corriger la convention d'axes n'améliore pas la tâche, et à dim=16 la
dégrade avec un intervalle qui exclut zéro.** L'explication tient en une
phrase : les hyperparamètres (`beta_curl`, `kappa`, `gamma_*`,
`threshold_amr`) ont été réglés par Optuna **sur l'opérateur historique**.
Appliquer le bon opérateur avec des coefficients calibrés pour un autre
revient à changer la grandeur mesurée sans retoucher l'instrument.

**Ce que la mesure n'établit PAS.** Elle ne dit pas que corriger *puis*
réoptimiser serait inutile. La comparaison est confondue par construction :
le bras corrigé tourne avec des coefficients calibrés pour l'autre opérateur,
donc il est désavantagé exprès. On ne peut pas conclure « la correction ne
sert à rien » d'une expérience où le bras corrigé part handicapé. C'est
précisément la question qu'un référé posera — *« Q-HAS a-t-il échoué parce
que son entrée physique était cassée ? »* — et « on n'a pas réoptimisé »
n'est pas une réponse.

Trois lectures, révisées après avoir compté le coût réel :

1. *Corriger et publier tel quel* — exclu : la mesure dit que c'est pire.
2. *Documenter et ne rien réoptimiser* — recommandation initiale, fondée
   sur « une semaine de calcul Optuna ». J'ai d'abord cru la réfuter avec
   les ~47 h de **mur** mesurées dans les bases. C'était un mauvais cadrage :
   les essais tournaient jusqu'à 9 de front, soit **224 h de CPU = 9.3 jours
   mono-cœur**. L'annonce d'origine était juste en temps processeur, et
   c'est le temps processeur qui gouverne le coût d'une relance.
3. *Réoptimisation ciblée* — retenu, mais chiffré honnêtement. Seuls
   `beta_curl`, `kappa` et `threshold_amr` touchent le canal du rotationnel.
   Un essai du bras quantique coûte **56 min de CPU** (médiane sur 178
   essais). Donc :

   | budget | CPU | mur sur 4 cœurs | mur sur 32 cœurs |
   |---|---|---|---|
   | 30 essais | 28 h | ~7 h | ~1 h |
   | 60 essais | 56 h | ~14 h | ~2 h |
   | 100 essais | 93 h | ~23 h | ~3 h |

   C'est **une nuit sur une machine ordinaire**, pas « quelques heures ».
   Le nombre d'essais nécessaire en dimension 3 est une **hypothèse non
   mesurée** — c'est la partie molle de l'estimation.

Tant que (3) n'est pas fait, l'énoncé publiable est le fait mesuré — les
indicateurs sont mal nommés, et les corriger *à hyperparamètres inchangés*
ne restaure pas de performance — et **pas** la conclusion plus forte que la
convention serait sans importance.

Le manuscrit doit donc dire que les indicateurs nommés « vorticité » et
« divergence » de V1 sont en réalité deux composantes du tenseur des
déformations, que ce fait est mesuré et non supposé, et que le corriger sans
réoptimiser ne restaure pas de performance. La bonne question pour la suite
n'est pas « la vorticité est-elle juste ? » mais **« le critère a-t-il jamais
eu besoin de la vorticité ? »** — le canal courant de `K_plaquettes` donnait
déjà r = +0.000 avec la vraie densité de courant.

## Tests

| fichier | tests | ce qu'ils verrouillent |
|---|---|---|
| `tests/test_analytic_fields.py` | 36 | les cinq grandeurs nommées contre des champs à réponse connue ; l'invariance bit-à-bit du chemin par défaut |
| `tests/study/test_fixed_curl_variant.py` | 7 | le drapeau change vraiment quelque chose, atteint θ à travers l'encodeur ψ, et suffixe son artefact |
| `tests/study/test_curl_convention_gap.py` | 14 | budget apparié, Spearman, bootstrap par scénario, verdict sans IC interdit |

Les trois mutations essayées sur `tests/test_analytic_fields.py` (axes
échangés dans `forward_curl_z`, `curl_z` ignorant son drapeau, `fixed_curl`
passé à `True` par défaut) sont toutes détectées.

---

# Le bras QAOA est échantillonné, et de combien

## Le fait

`src/VQA/execute.py` construit sa distribution finale par
`final_distribution = counts / total_shots` à partir de `sampler.run(...)`.
**Aucune graine n'est fixée dans tout `src/VQA/`** — ni `seed_simulator`, ni
`np.random.seed`, ni graine passée au sampler. Deux appels sur le même état
et les mêmes hyperparamètres ne donnent donc pas le même résultat.

## La mesure

Dix appels **strictement identiques** à `qaoa_block_scores` (mhd_rotor,
Re=800, N=64, 3×3 blocs, `w_z_frac`=0.10, `threshold`=0.3), soit 45 paires :

| grandeur | min | médiane | max |
|---|---|---|---|
| dispersion des scores de bloc (ptp par appel) | 1.79e-1 | — | **3.61e-1** |
| auto-corrélation de rang | **0.350** | **0.933** | 1.000 |
| appels dégénérés (score constant sur les 9 blocs) | — | **0 / 10** | — |

Un premier sondage à 6 appels (15 paires) donnait un minimum de 0.550 et une
médiane de 0.883 : la queue descend plus bas que 15 paires ne le laissaient
voir. C'est l'échantillon à 45 paires qui fait foi, et c'est la raison pour
laquelle les seuils des tests portent sur la **médiane**, jamais sur le
minimum.

Deux lectures, opposées, et toutes deux importantes :

1. **Les valeurs bougent beaucoup.** Plus d'un cinquième de l'échelle [0,1]
   entre deux exécutions identiques, au pire.
2. **Le classement, lui, tient.** Auto-corrélation de rang médiane 0.933 :
   le bras ordonne les blocs de façon reproductible, même si les valeurs
   qu'il leur attribue ne le sont pas. Les conclusions de cette étude qui
   reposent sur un **ordre** (budget apparié, top-k) sont donc robustes ;
   celles qui reposeraient sur une **valeur** ne le seraient pas.

## Ce que cela a cassé

`check_sweep_behaviour` exigeait `abs(delta) <= 1e-9` pour au moins une des
douze combinaisons d'hyperparamètres — c'est-à-dire que QAOA sélectionne
*exactement* les mêmes 2 blocs sur 9 que le classique. C'est huit ordres de
grandeur sous le bruit du bras. L'assertion avait été calibrée sur une
exécution unique, dans le commit `32d124a` qui prétendait précisément donner
des critères d'acceptation aux étapes qui n'en avaient pas. **Septième
assertion à tirage unique sur ce bras.**

Le plafond est conservé — aucun réglage ne doit faire passer QAOA devant le
classique sur données propres, et cela peut échouer. Deux critères manquants
sont ajoutés : ρ doit rester positif (sinon le bras ne classe plus rien et
le plafond ne prouve rien) et ρ doit **varier** entre combinaisons (sinon
les hyperparamètres n'atteignent pas le bras). L'égalité exacte est
désormais rapportée, pas assertée.

## Deux erreurs de méthode commises en écrivant ce test

Consignées parce qu'elles sont du genre même que l'étude traque.

1. **Un seuil posé sans mesure.** `test_the_ranking_survives_the_sampling`
   assertait `min(rhos) > 0.5` sur trois paires. Le minimum sur trois tirages
   est la statistique la plus instable disponible, et 0.5 était une
   intuition. La mesure donne min 0.550 : le seuil tombait dans la queue de
   la distribution. Corrigé en médiane sur 10 paires, seuil à 0.6.
2. **Un chiffre publié sous-estimé, deux fois.** La dispersion annoncée
   d'abord (9.58e-2, 8.70e-2) venait de trois appels ; puis 1.50e-1 / 2.15e-1
   de quinze paires ; la mesure à 45 paires donne un ptp par appel allant
   jusqu'à 3.61e-1. Chaque valeur était exacte pour son tirage, et chacune
   sous-estimait la suivante. Un échantillon trop petit ne se signale pas
   comme tel.
3. **Un diagnostic construit sur une prémisse non vérifiée.** Deux tests de
   rang échouant simultanément, j'en ai conclu — `médiane > 0.6` et
   `min < 1.0` étant complémentaires — qu'il y avait des NaN, donc des
   appels rendant un score constant. La sonde le réfute : 0 appel dégénéré
   sur 10. La déduction était valide, sa prémisse implicite (que c'étaient
   des échecs d'assertion) ne l'était pas : les deux tests tombaient sur un
   `TypeError`, `spearmanr` recevant des tableaux (3, 3) non aplatis et
   renvoyant une matrice. Un garde de forme l'empêche désormais.

## Défaut annexe

`_SWEEP_ROWS` et `_RESOLUTION_ROWS` sont des listes au niveau module qui
n'étaient jamais vidées : deux exécutions dans le même processus donnaient
24 lignes et faisaient échouer `len(rows) == 12` pour une raison étrangère à
ce que le test mesure.

## Tests

| fichier | ce qu'il verrouille |
|---|---|
| `tests/test_qaoa_arm_is_sampled.py` | le bras varie ; la dispersion écrase la tolérance ; le classement tient (médiane) ; il bouge quand même (au moins une paire) ; aucune graine dans `src/VQA/` ; l'assertion d'égalité exacte n'est pas réintroduite |

Si une graine est fixée un jour dans `src/VQA/`, le premier de ces tests
tombe : c'est voulu. Il faudra alors rétablir les assertions exactes et le
consigner ici.

---

# La prolongation globale de l'AMR est fausse — D-2

## Deux écarts de convention qui se composent

`MHDSolver._upsample_global` prolonge le champ grossier vers la grille fine
dans le chemin AMR (`step_layered`, correction tau). Elle cumule deux
erreurs, aucune des deux ne produisant de plantage :

1. **Convention d'échantillonnage.** Elle vise `(j+0.5)/f − 0.5`, c'est-à-dire
   le **centre** des cellules, alors que `PeriodicGrid` place ses points aux
   **nœuds** (`linspace(0, L, N, endpoint=False)`). D'où un décalage constant
   de **−0.375 cellule grossière** à facteur 4.
2. **Mode d'enroulement.** Elle passe `mode='wrap'` à `map_coordinates`.
   Depuis scipy 1.6, ce mode n'est **pas** l'enroulement périodique : c'est
   `'grid-wrap'` qui l'est. `'wrap'` traite le tableau comme si le premier et
   le dernier échantillon coïncidaient.

Le docstring annonce « respecte la topologie torique du domaine ».

## Mesure

`sin(x)·cos(y)`, 32 → 128, scipy 1.17.1 :

| convention | mode | erreur max |
|---|---|---|
| centre (code actuel) | `wrap` | **2.49e−1** |
| centre | `grid-wrap` | 7.35e−2 |
| nœud | `wrap` | 1.79e−1 |
| **nœud** | **`grid-wrap`** | **7.74e−6** |

**Quatre ordres de grandeur** séparent le code de ce que la même
interpolation cubique atteint avec les bonnes conventions.

## Ce que ça coûte au chemin AMR — et où

Orszag-Tang, N=64, 15 pas, `max_depth=2` (donc `cf=4`), écart entre le code
et la variante corrigée :

| couverture par les patchs | écart relatif |
|---|---|
| totale | **2.7e−15** (arrondi) |
| un quart du domaine | **1.67 %** |
| aucun patch | **1.79 %** |

**Sous un patch actif, l'erreur s'annule exactement.** C'est le principe de
la correction tau : la phase 1 ajoute le delta grossier prolongé, la phase 2
le retranche et lui substitue le delta fin — la prolongation disparaît de la
différence. Elle ne survit que sur le **fond non raffiné**.

Vérifié aussi : à `max_depth=0`, `step_layered` est **bit-à-bit identique** à
`step_full` (écart exactement 0.0). La garantie annoncée ligne 561 tient.

## Pourquoi cela compte pour le manuscrit

`src/pipeline.py:480,485` : les **deux** bras — `sim_quantum` et
`sim_classical` — avancent par `step_layered`. L'erreur leur est donc
**commune** et ne biaise pas leur comparaison mutuelle. Mais le témoin
avance par `step_full` (ligne 478), qui n'a pas le défaut.

Conséquence : l'erreur de champ de chaque bras **contre la référence** porte
une composante systématique qui ne doit rien au critère de raffinement. Elle
comprime la plage dans laquelle un meilleur critère pourrait se distinguer,
et pousse les deux bras vers le bas ensemble.

**Cela pèse directement sur la décision de réoptimiser.** Réoptimiser les
hyperparamètres contre une métrique qui porte un plancher de ~1.7 %
d'origine numérique, c'est optimiser en partie contre ce défaut. La
correction de `_upsample_global` devrait donc précéder toute réoptimisation,
pas la suivre.

## Décision requise

Corriger `_upsample_global` change **tout nombre publié passant par
`step_layered`** — soit la campagne de boucle fermée entière. Ce n'est pas
une correction « au passage » : elle demande une décision explicite et une
re-exécution. Le défaut est donc mesuré, testé et consigné ici, mais
`src/` n'est **pas** modifié.

## Tests

`tests/test_amr_resampling_analytic.py`, 30 tests :

| ce qui est verrouillé | |
|---|---|
| restriction | moyenne préservée exactement, moyenne de bloc cellule par cellule, dilution d'un pic (contraste voulu avec le max-abs de `RescaleArrays`) |
| prolongation | champ uniforme préservé, identité à facteur 1, absence de couture, **l'écart aux conventions correctes** et le **biais de décalage** |
| champs physiques | toutes les clés à la bonne taille, moyenne préservée, halo, enroulement torique |
| détection de bord | chaque bord reconnu pour lui-même, anomalie intérieure ignorée, patch uniformément actif silencieux |
| impact AMR | annulation sous patch, survie sur le fond, `max_depth=0` identique à `step_full` |

---

# Audit de contrat : quatre défauts trouvés en demandant à chaque
# fonction ce qu'elle prétend faire

Les tests analytiques déjà en place vérifient des **valeurs**. Cette passe
vérifie des **contrats** : pour chaque fonction du chemin de décision, que
promet sa docstring, consomme-t-elle les entrées que sa signature annonce,
rend-elle la forme et le domaine promis, et deux chemins censés coïncider
coïncident-ils encore ?

Les quatre défauts ci-dessous étaient invisibles aux tests de valeur, parce
qu'ils partagent tous la même forme : un calcul qui rend une valeur
parfaitement plausible, indiscernable d'une valeur juste.

Commande de recette :

```bash
python -m pytest tests/test_mapper_contracts.py \
                 tests/test_hamiltonian_contracts.py \
                 tests/test_downsampling_contracts.py -q
```

`107c1cf` + cette passe. 59 + 29 + 28 = **116 tests**.

## D-11 — la diode de choc s'appliquait au cisaillement

`src/Simulation/PhysToAngle.py` — `compute_stress_flux`.

`_compute_filtered_flux` lit `array[0]` comme la composante **normale**
(diode `max(0, −Δ)`, poids `w_compress = 2`) et `array[1]` comme la
**tangentielle** (`abs`, poids `w_shear = 1`). L'ordre des tuples était
écrit sous la convention inverse du dépôt (axis=1 lu comme x), si bien que
la composante transverse arrivait dans la case de la normale.

| mesure sur champ analytique | code | conception |
|---|---|---|
| rapport compression / cisaillement | **0.500** | 2.0 |
| compression distinguée de l'expansion | **non** — flux identique | oui |

La diode était donc **inerte** : sa seule raison d'être est de séparer la
compression de l'expansion, et elle ne le faisait pas. Le signe d'une
différence tangentielle ne porte aucune information de compression ; la vraie
différence normale, elle, passait par `np.abs` dans la branche de
cisaillement.

Écart relatif sur Φ, snapshots DNS réels à N=256, Re=400 :

| scénario | Φ_h | Φ_v | médiane \|Δψ\| | max \|Δψ\| |
|---|---|---|---|---|
| orszag_tang | 36.6 % | 43.4 % | 0.259 | 2.98 |
| kelvin_helmholtz | 96.5 % | 51.0 % | 0.047 | 1.33 |
| mhd_rotor | 46.2 % | 37.1 % | 0.023 | 3.14 |
| harris_tearing | 93.3 % | 43.9 % | 0.008 | 2.90 |

Φ n'alimente pas θ (le score classique s'en charge) : il alimente **ψ**. Le
rayon d'action du défaut est donc exactement ψ — la quantité dont l'ablation
est au programme. **Toute lecture de l'ablation ψ antérieure à cette
correction porte sur un ψ construit sur un flux faux.**

Corrigé derrière `AngleMapper(fixed_flux=True)`, par défaut **True**, même
traitement que `fixed_curl`. `fixed_flux=False` reproduit le chemin
historique bit à bit.

## D-12 — `PhysicalMapperV2` est aveugle à trois des quatre grandeurs
## que sa docstring nomme

**Rectification de portée.** J'avais écrit « le mappeur déployé ». C'est
faux : `src/pipeline.py` n'importe **jamais** `HamiltParams_v2`. La boucle
fermée — celle qui produit les résultats de niveau 3 et la frontière de
Pareto — instancie `PhysicalMapper` (v1) avec ses hyperparamètres entraînés
(σ, β_curl, γ_hydro, γ_mag, κ, w_z_frac). Le v2 n'est utilisé que par les
scripts de `study/`.

Ce qui suit vaut donc pour les **analyses de `study/`**, pas pour la boucle
déployée.

`src/Simulation/HamiltParams_v2.py` — `PhysicalMapperV2`.

La docstring annonçait « Only physical constants (nu, eta, dx) and the
refinement threshold (thr_amr) affect the output ». Mesuré :

| grandeur | effet mesuré |
|---|---|
| `dx` : 1.0 contre 0.001 | `C`, `K`, `H` **bit à bit identiques** ; `K_xpoint` à 3.8e−11 |
| amplitude des champs ×10 | identique à 1.6e−10 |
| `nu`, `eta` | **absents du fichier** — aucun nombre de Reynolds n'entre |
| `thr_amr` | seule grandeur nommée qui agit |

Le v2 est **adimensionnel** : chaque terme est divisé par une norme prise sur
le même champ. `det(∇B) ∝ 1/dx²` est divisé par `max|det| ∝ 1/dx²` ; `dx` se
simplifie exactement.

Ce n'est pas un bug de calcul, mais cela change la lecture des analyses
**qui utilisent le v2**. **H4 (transfert)** : le v2 ne peut pas distinguer un
écoulement visqueux d'un écoulement inertiel, donc un transfert entre nombres
de Reynolds est trivialement satisfait par ses coefficients — toute
dépendance en Re ne peut venir que du score externe. **H3
(représentation)** : le v2 ne voit que la *forme relative* des champs, jamais
leur échelle.

La boucle fermée, elle, tourne sur le v1, où ν, η et dx entrent bel et bien
(via `Re_h = v_jump·dx/ν`, `RE_CRIT`, `v_jump_crit`). Les deux mappeurs ne
sont donc pas interchangeables pour lire une hypothèse : il faut dire lequel
a produit le nombre.

Aucun code modifié : la docstring a été réécrite pour dire ce que le code
fait. Deux autres mensonges de documentation corrigés dans le même fichier :
l'argument `sim`, annoncé comme fournissant les opérateurs de gradient, est
inutilisé (le v2 les réimplémente en ligne) ; et le commentaire du biais Z
portait encore le signe négatif que l'en-tête du module avait déjà corrigé.

## D-13 — les bords gauche et haut de l'Hamiltonien lisaient l'arête
## intérieure

`src/VQA/cost_hamiltonian.py` — `create_bounded_hamiltonian`.

`C_edges[0][a, b]` couple la cellule `(a, b)` à `(a, b+1)` : c'est la
convention des deux mappeurs, qui forment leurs sauts par
`champ − np.roll(champ, −1, axis=1)`. L'arête reliant le halo de gauche à la
première colonne du cœur est donc `C_edges[0][ci, 0]`.

Le code lisait `[ci, 1]` — l'arête **intérieure** (0)-(1), déjà consommée
quelques lignes plus haut comme couplage de cœur. Même chose en haut avec
`C_edges[1][1, cj]` au lieu de `[0, cj]`. Le bon coefficient existe pourtant :
les paramètres sont calculés sur un patch `(dim+2, dim+2)` qui contient le
halo.

Les bords **droit** et **bas**, eux, lisaient la bonne case. L'Hamiltonien
était donc **asymétrique entre gauche et droite sur un patch symétrique** —
c'est le test qui attrape le défaut sans connaître les indices.

Écart sur un patch réel d'Orszag-Tang (N=256, dim=4, cœur de la nappe de
courant) :

| bord | lu | correct | écart |
|---|---|---|---|
| gauche, ci=1 | −1.0243 | −0.9572 | 0.0671 (7.0 %) |
| gauche, ci=4 | −1.0251 | −0.9809 | 0.0442 (4.5 %) |
| haut, cj=1 | −1.1678 | −1.1476 | 0.0202 (1.8 %) |
| haut, cj=4 | −1.3117 | −1.2965 | 0.0152 (1.2 %) |

Corrigé directement : contrairement au rotationnel, il n'existe pas de
lecture défendable où `[ci, 1]` serait l'arête du halo.

**Garde ajouté au passage.** Toutes les lectures sont indexées par `dim` sur
des tableaux supposés `(dim+2, dim+2)`. Un tableau **trop grand** ne
déclenchait aucune erreur : la boucle lisait un sous-bloc du coin supérieur
gauche et rendait un Hamiltonien valide, calculé sur la mauvaise portion du
patch. `create_bounded_hamiltonian` refuse désormais toute forme différente
et nomme chaque tableau fautif.

## D-14 — le score et les champs ne décrivaient pas la même région

`src/Simulation/refinement.py` — `_downsample_fields`.

Un patch descend vers le VQA par deux chemins indépendants : les champs par
**mean-pool** (`_downsample_fields`), le score par **max-pool**
(`RescaleArrays._process_score`). Le max-pool couvre 100 % du patch depuis sa
correction ; le mean-pool découpait `patch[:out_dim*bh, :out_dim*bw]` et
jetait le reste de la division.

La cellule `(i, j)` du score ne désignait donc plus la cellule `(i, j)` des
champs. La perte tombe toujours du même côté — les dernières lignes et
colonnes — donc c'est un **biais, pas du bruit**. Et ces dernières lignes
sont exactement le **halo droit et bas**, c'est-à-dire l'information de
voisinage que H3 cherche à évaluer.

Le patch vaut `extent + 2·pad` (le halo de `get_periodic_patch`) et la cible
`dim + 2·pad` : la division tombe rarement juste. Couverture à N=256 :

| dim | prof. 0 | prof. 1 | prof. 2 | prof. 3 |
|---|---|---|---|---|
| **2 (déployé)** | 100 % | **98.5 %** | **97.0 %** | **94.1 %** |
| 3 | 99.6 % | 100 % | 98.5 % | 88.2 % |
| 4 | 100 % | 96.9 % | 100 % | 88.2 % |
| 8 | 100 % | 100 % | 90.9 % | 88.2 % |

Le chemin déployé était touché dès la première descente.

Corrigé par des bornes `np.linspace` couvrant toute l'étendue — la même
correction que `_maxabs_pool_2d`, ce qui remet les deux chemins d'accord.
Quand la division tombe juste, la sortie est bit à bit identique et le
chemin rapide `reshape` est conservé.

## Ce que cette passe dit des tests précédents

Les quatre défauts partagent la forme que les tests de valeur ne peuvent pas
voir : ils rendent un résultat plausible. Trois d'entre eux ont été trouvés
par la même question — *deux chemins censés coïncider coïncident-ils ?* —
qu'aucun test antérieur ne posait :

- D-11 : la diode contre sa propre docstring ;
- D-13 : le bord gauche contre le bord droit ;
- D-14 : la réduction des champs contre celle du score.

C'est la classe de test à étendre en priorité, pas le nombre d'assertions.

---

# Ce que le circuit peut déplacer, et par quel canal

Ce n'est pas un défaut : c'est une propriété structurelle du circuit
déployé, mesurée parce que l'audit de contrat posait la question « à quoi
sert ce paramètre ? ». Elle borne par le haut ce que H0b et H2 peuvent
espérer mesurer.

La couche de coût `exp(−iγH)` est **diagonale** : elle n'ajoute que des
phases et ne peut changer aucune probabilité de mesure. Seul le mixeur
`exp(−iβ ΣXᵢ)` déplace `P(|1⟩)`. Mesuré sur les coefficients v2 réels
(Orszag-Tang N=256, seuil 0.1496, `reps=2`), en balayant γ de 0 à 2π :

| canal | déplacement max d'une probabilité |
|---|---|
| γ seul, β = 0 | **4.4e−16** — rien, aux erreurs d'arrondi près |

Et β est borné par construction à `π/(4·reps) = 0.393 rad`
(`execute.py:112`, pour empêcher COBYLA de partir à β=1 et d'écraser tout
le raffinement).

**Conséquence : tout ce que l'Hamiltonien apporte à la décision passe par
son interaction avec le mixeur.** En balayant toute la grille admissible —
donc ce qu'un optimiseur *parfait* atteindrait, pas ce que COBYLA trouve :

| patch | mixeur seul | mixeur + H | apport de H |
|---|---|---|---|
| (100,100) | 0.3776 | 0.5359 | 0.1583 |
| (40,180) | 0.2667 | 0.4267 | 0.1600 |
| (200,60) | 0.2541 | 0.4897 | 0.2357 |
| (128,128) | 0.1028 | 0.4400 | 0.3372 |
| (10,10) | 0.2238 | 0.5052 | 0.2814 |
| **médiane** | **0.254** | **0.490** | **0.236** |

Lecture : sur un patch typique, un optimiseur parfait peut déplacer une
marginale de 0.49 au plus ; environ la moitié de ce déplacement (0.254) est
une simple rotation de mixeur, indépendante de toute physique. L'Hamiltonien
n'est donc **pas** inerte — il apporte 0.236 de médiane — mais il ne peut
agir qu'à travers un canal borné à 0.393 rad, et le témoin correct pour
mesurer son apport est **le mixeur seul**, pas le score classique.

Aucune campagne du dépôt n'utilise ce témoin. C'est le contrôle qui manque à
H0b : « le QAOA déplace-t-il la décision ? » ne distingue pas « le mixeur la
déplace » de « la physique la déplace ».

Vérifié au passage, et fixé par un test parce que c'est un contrat
inter-bibliothèque qui peut casser sur une mise à jour sans erreur :
`QAOAAnsatz` ordonne ses paramètres `[β…, γ…]`, ce qui est bien l'ordre que
`execute` suppose en construisant `x0 = [zeros(reps), rampe_γ]`. Un
réordonnancement appliquerait la rampe au mixeur et la borne β au terme de
coût, en silence.

## Tests

`tests/test_vqa_chain_contracts.py`, 41 tests. La trace de bout en bout —
score (i,j) → qubit k → terme de Pauli → caractère de la chaîne Qiskit →
marginale — est posée en forçant un qubit connu et en vérifiant qu'il
ressort à sa place. Une seule convention retournée rendrait la carte de
décision **spatialement miroir** : même taille, mêmes valeurs, même fraction
raffinée, indiscernable d'une carte juste par tout test de valeur.

Conforme : ordre des bits petit-boutiste (le commentaire du code annonçait
l'inverse ; c'est le code qui a raison), `P(|1⟩) = sin²(θ/2) = score` à
2.4e−15, ψ déplace la phase et jamais la probabilité, l'aplatissement ligne
par ligne du circuit coïncide avec `idx_H(i,j) = i·dim + j` de
l'Hamiltonien, et `params = 0` reproduit exactement θ-init (la porte de
sortie du raccourci « Hamiltonien nul »).

---

# D-16 — la liste de patchs se recouvrait elle-même

`src/Simulation/refinement.py` — `_run_level` et `_run_level_classical`.

L'invariant le plus élémentaire d'un AMR — chaque cellule appartient à
exactement une feuille — n'était vérifié nulle part. Une liste qui recouvre
deux fois la même région reste parfaitement plausible : bornes valides,
profondeurs cohérentes, scores dans [0, 1]. Seule une somme de couverture la
distingue d'une liste juste.

Le **sondage de bord** (« l'anomalie touche le bord dans cette direction, on
descend même si le signal est marginal ») était un bloc **séparé**, exécuté
après la ventilation. Quand il se déclenchait, le sous-patch avait déjà été
enregistré comme feuille non raffinée par la branche `else`, et il était en
plus poussé au niveau suivant. La même région était comptée **deux fois** :
une fois comme feuille grossière, une fois redécoupée.

Mesure sur les quatre scénarios (N=256, `dim=2`, `max_depth=3`, 6 instantanés
chacun, soit 24 configurations) :

| seuil | configurations avec recouvrement | pire cas |
|---|---|---|
| **0.1496 (déployé)** | **2 / 24** | **25.0 %** du domaine |
| 0.20 | 3 / 24 | 17.2 % |
| 0.25 | 4 / 24 | 12.5 % |
| 0.30 | 6 / 24 | 25.0 % |
| 0.40 | 9 / 24 | 28.1 % |
| 0.50 | 12 / 24 | 20.3 % |

Jusqu'à **trois patchs** sur une même cellule. Toute métrique de budget ou de
couverture lue sur la liste finale surcomptait d'autant — et le balayage de
seuils de la **frontière de Pareto** passe exactement dans la zone où le
défaut est le plus fréquent.

Les deux bras portaient le défaut à l'identique, donc leur comparaison
mutuelle n'était pas biaisée ; le coût absolu, si.

Corrigé en fondant le sondage dans le même `if/elif/else` : un sous-patch est
soit raffiné, soit feuille, jamais les deux. Après correction, sur les six
seuils de 0.1496 à 0.65 : **0/24 recouvrement, 0 % de trou** — la liste pave
exactement le domaine.

**Corrigé au passage** : le journal `verbose` affichait
`threshold_amr + (1−threshold_amr)·depth/max_depth`, une rampe en profondeur
que le code n'applique plus (`effective_threshold = threshold_amr`, la rampe
est commentée). Le journal annonçait donc un seuil, et le code en appliquait
un autre ; toute lecture des décisions dans les journaux était fausse.

## Tests

`tests/test_amr_tiling_contracts.py`, 44 tests : pavage exact sur les quatre
scénarios × trois seuils, sur tout le domaine de seuils de la frontière de
Pareto, à chaque taille de patch et chaque profondeur ; aire couverte égale à
l'aire du domaine ; aucune borne dupliquée ; monotonie du nombre de patchs en
fonction du seuil ; chaque bord reconnu pour lui-même par
`_boundary_activation`, et aucun drapeau sur un patch uniformément actif ou
uniformément calme (le faux positif généralisé qui déclenchait le doublon).
Deux tests structurels vérifient que les deux bras gardent la même forme de
sondage et que le seuil journalisé est celui appliqué.

---

# Verdict de la suite QAOA — et ce qu'il révèle

`python -m pytest tests/test_qaoa_advantage.py tests/test_qaoa_noise_and_early.py
tests/test_qaoa_scaling_and_hparams.py tests/test_qaoa_decisions.py
tests/test_qaoa_physics_decision.py tests/test_qaoa_arm_is_sampled.py
tests/QAOA_test.py -q`

**3 échecs, 27 succès, 1 h 21 min.** Les trois échecs sont des **valeurs qui
ont bougé**, pas des casses — et l'un d'eux renverse une lecture publiée.

## Le terme ZZZZ était numériquement mort sur un vortex

Deux des trois échecs sont le même fait, mesuré par deux harnais différents :
un vortex de Lamb-Oseen **gagne désormais un contraste spatial positif**, là
où les tests affirmaient qu'il n'en gagnait pas.

Ces tests ne se trompaient pas : sur le code de l'époque le contraste valait
`−0.0058 ± 0.0064`, soit du bruit de tirage légèrement négatif. Mais ce
n'était pas une propriété du QAOA.

Attribution, mesurée sur le même vortex, 16 tirages par ligne, tout le reste
égal :

| `fixed_curl` | `fixed_flux` | contraste | écart-type | σ | max\|K\| |
|---|---|---|---|---|---|
| False | False | **−0.00725** | 0.00859 | −3.4 | **0.0553** |
| False | True | −0.00852 | 0.00896 | −3.8 | 0.0553 |
| **True** | False | **+0.05672** | 0.03976 | **+5.7** | **1.2545** |
| True | True | +0.07292 | 0.04429 | +6.6 | 1.2545 |

La ligne `(False, False)` reproduit la valeur historique à l'écart-type près.

La cause est **D-1** : le rotationnel des mappeurs était écrit sous la
convention `indexing='xy'` alors que la grille construit ses champs en
`indexing='ij'`, si bien qu'une rotation solide rendait exactement 0. Le
terme ZZZZ de plaquette — **dont la seule raison d'être est de détecter une
circulation** — était donc aveugle aux vortex. Son coefficient passe de
0.055 à 1.255, **vingt-trois fois plus grand**, dès que le rotationnel voit
la rotation.

Le sens de lecture change : ce n'est pas que le QAOA ne discrimine pas un
vortex, c'est qu'on lui donnait un Hamiltonien qui ne pouvait pas en voir.

`fixed_flux` (D-11) amplifie l'effet mais ne le crée pas : il était déjà
positif à `(True, False)`.

## Ce que cela ne dit PAS

Ces deux tests utilisent le harnais **v1** — `PhysicalMapper`,
`physical_score`, θ construit à partir du flux. Le mappeur **déployé** est le
v2, qui normalise `K` par `max|ω| + max|J|`. Sur le v2, l'effet est une
**redistribution**, pas une amplification uniforme :

| champ | \|K\| médian (legacy) | \|K\| médian (corrigé) | rapport |
|---|---|---|---|
| rotation solide | **0.000000** | 0.015873 | ∞ |
| vortex Lamb-Oseen | 0.018998 | 0.000149 | 0.008× |
| DNS orszag_tang | 0.082185 | 0.180390 | 2.2× |
| DNS mhd_rotor | 0.000717 | 0.000124 | 0.17× |

Sur une rotation solide le `K` legacy est **exactement nul** dans les deux
mappeurs — le défaut est bien commun. Mais sur des champs réalistes le v2
redistribue, dans un sens ou dans l'autre selon le champ, parce que sa
normalisation par le maximum du domaine change en même temps que le
numérateur.

**Conclusion honnête** : le fait établi est que la lecture publiée « le
contraste d'un vortex est du bruit de tirage » a été mesurée sur un
Hamiltonien dont le terme de circulation était numériquement mort. Savoir si
la conclusion du chemin **déployé** bascule demande de relancer la campagne,
et ne peut pas se déduire de ces quatre lignes.

## Le troisième échec : une coïncidence prise pour un invariant

`test_noise_robustness` exigeait `frac_cl == gt_frac` à 1e−9 — « sans bruit,
le bras classique atteint la fraction capturée optimale ». Mesuré : 0.3151
contre 0.3245 sur Orszag-Tang (0.9709), et 1.0000 sur le rotor.

Ce n'était pas un invariant. `gt_frac` classe les blocs par une **erreur de
troncature** (dérivée seconde) et `frac_cl` par le **score classique** : deux
quantités différentes, dont les *k* premiers blocs se trouvaient coïncider.
La correction du rotationnel a changé le classement du score sur
Orszag-Tang, et la coïncidence est tombée.

Le vrai invariant — le bras classique ne peut pas **dépasser** l'optimum — est
déjà vérifié ligne par ligne plus bas. L'assertion borne désormais l'écart
relatif à 5 % au lieu de nier son existence.

## Trois tests V1 mis à jour au même titre

Cinq tests de `tests/test_vqa_stack_analytic.py` passaient des **comptes
bruts** à `postprocess`, dont deux figeaient explicitement les deux pièges
que D-15 ferme : rendre des zéros sur une distribution vide (une lecture
manquante devenait un patch calme), et tronquer en silence une chaîne plus
longue que le registre. Ils affirment désormais que les deux cas sont
refusés.

**V1 non-QAOA après toutes les corrections : 844 succès, 4 ignorés** (15 min
46 s), une fois ces cinq tests mis en accord avec le nouveau contrat.

---

# D-17 / D-18 — le balayage de D-1 s'était arrêté à `src/`

L'audit de contrat appliqué à `study/` et `figures/` a trouvé quatre sites
qui réimplémentaient encore leur propre opérateur sous la convention
inverse. La correction D-1 n'avait touché que `src/Simulation/`.

## Ce que l'opérateur « legacy » calcule réellement

```
correct : (roll(fy,-1,AXIS_X) - fy) - (roll(fx,-1,AXIS_Y) - fx) = ∂fy/∂x - ∂fx/∂y
legacy  : (roll(fy,-1,AXIS_Y) - fy) - (roll(fx,-1,AXIS_X) - fx) = ∂fy/∂y - ∂fx/∂x
```

Ce n'est **pas** un rotationnel de signe opposé — auquel cas `abs` ou le
carré auraient tout rattrapé. C'est son **complémentaire** : une combinaison
de déformation, nulle là où le rotationnel est maximal, maximale là où il
s'annule.

| champ | rotationnel | opérateur legacy |
|---|---|---|
| rotation solide | +0.392699 | **0.000000** |
| cisaillement pur | −0.196350 | **0.000000** |
| compression pure | 0.000000 | −0.392699 |

## D-17 — trois sites, trois quantités mal nommées

| fichier | fonction | conséquence |
|---|---|---|
| `study/h2b_prediction/h2b_v1_hamiltonian_loso.py` | `jz_from_b` | rendait 0 sur une rotation solide ; appelle désormais `forward_curl_z` |
| `study/h2b_prediction/h2b_ceiling_random_split.py` | `omega_z`, `J_z` (features ML) | deux des neuf features de H2b mesuraient la déformation |
| `figures/v1_legacy/fig_utils.py` | `compute_enstrophy` | l'enstrophie tracée n'était pas une enstrophie |

Validation analytique de `compute_enstrophy`, cisaillement pur périodique
`vx = sin y`, enstrophie exacte `2π² = 19.7392` :

| version | valeur | écart |
|---|---|---|
| corrigée | 19.7352 | 0.02 % (erreur de la différence centrée) |
| ancienne | **0.0000** | **100 %** |

**Piège de validation à retenir** : sur Taylor-Green les deux conventions
rendent la **même** intégrale, par symétrie de leurs carrés. Un test écrit
sur ce champ aurait passé sans rien vérifier. Le fichier de test le fige
explicitement.

`study/pipeline/hard_patch_labels.py` et `study/common/qaoa_inputs.py`
étaient signalés par le balayage mais **vérifiés corrects** — ils gardent des
axes numériques nus pour rester bit-à-bit identiques aux artefacts publiés,
et figurent dans une liste d'exceptions explicitement documentée.

## D-18 — rectification : la moitié `fluctuating_KE` était déjà connue

**Correction à la première rédaction de cette section.** J'avais présenté le
défaut d'axe de `dns_validation.fluctuating_KE` comme une trouvaille de cet
audit. C'est faux. Il était déjà connu, consigné comme **déviation D2**, et
la décision prise alors était explicite — `dns_extension.py:85` : « phase 1b
reste intouchée, réparation côté v3 par copie ». Un test de `tests/v3`
épinglait même la contamination
(`test_phase1b_observable_is_contaminated_by_base_flow`).

Ma correction dans `dns_validation.py` a donc **rompu ce gel**, et c'est la
grande suite qui l'a signalé en faisant échouer ce test. Le fichier a été
remis dans son état d'origine.

### Ce que l'audit apporte réellement

**1. La mesure de D2, qui n'était chiffrée nulle part.**

`fluctuating_KE` retranche une moyenne pour isoler la perturbation ; elle
doit être prise le long de la direction **homogène**.
`init_kelvin_helmholtz` construit son profil à partir de `grid.Y`, que
`meshgrid(x, y, indexing='ij')` fait varier le long de l'**axe 1** — la
direction homogène est donc l'axe 0. Le code moyenne sur l'axe 1, **à
travers la couche de cisaillement**, et ne soustrait rien.

Sur le profil de base **sans aucune perturbation**, où la réponse attendue
est zéro :

| moyenne prise sur | valeur | part de l'énergie cinétique totale |
|---|---|---|
| axe 1 (phase 1b, gelé) | 3.411e−01 | **73 %** |
| axe 0 (`fluctuating_ke_fixed`) | 1.323e−30 | 0 % |

En allumant la perturbation nominale (amplitude 0.1) :

| | base seule | avec perturbation | rapport |
|---|---|---|---|
| gelé | 0.34115 | 0.34120 | **1.0002** |
| corrigé | 1.3e−30 | 2.5e−04 | 1.9e+26 |

La grandeur gelée est à **99.98 % de l'écoulement de base**.

**2. Une seconde déviation dans le même fichier, celle-là non répertoriée.**

`mean_sq_current` porte la même inversion d'axes : `⟨J²⟩` vaut en fait
`⟨(∂By/∂y − ∂Bx/∂x)²⟩`. Aucun test ne l'épinglait. Elle est désormais
consignée comme **déviation D3**, `dns_validation.py` reste gelé au même
titre que pour D2, et une copie corrigée `mean_sq_current_fixed` a été
ajoutée à `dns_extension.py` sur le modèle de `fluctuating_ke_fixed`.

Vérification de la copie : sur un cisaillement magnétique pur
`Bx = −sin y`, la version gelée rend **0** et la corrigée retrouve
`⟨(∇×B)²⟩` à 5 % près ; sur un champ potentiel `B = ∇φ` la corrigée rend
zéro (contrôle négatif).

### La leçon

Une déviation connue mais **non écrite là où elle vit** se fait recorriger
par erreur — c'est exactement ce qui vient d'arriver. Les deux sont
maintenant documentées dans `dns_validation.py` lui-même, et un test vérifie
que ces mentions y restent.

## Tests

`tests/study/test_no_private_curl_survives.py`, 26 tests. Ils verrouillent les
**deux côtés** de D2 et D3 : la version gelée doit rester fausse à
l'identique (sans quoi les artefacts de phase 1b cessent d'être
reproductibles), et la copie corrigée doit être juste. Un test de plus est un
**balayage** : tout rotationnel écrit à la main avec un `axis=0`/`axis=1`
nu, hors de la liste d'exceptions vérifiées, fait échouer la suite. Un
opérateur écrit à la main est indiscernable d'un opérateur juste tant qu'on
ne l'évalue pas sur une rotation solide ; exiger `AXIS_X`/`AXIS_Y` rend
l'erreur visible à la lecture.

---

# D-19 / D-20 — deux pièges dans le contexte d'exécution partagé

`src/VQA/runtime.py`. `VQARuntime` est construit une fois par run et passé à
chaque appel VQA. Les deux défauts sont de la même famille : **une valeur
inutilisable qui se laisse produire sans bruit**.

## D-19 — un backend inconnu construisait un objet mort

`_init_backend` n'avait pas de branche `else`. Un `backend_name` inconnu
laissait `_backend`, `_estimator` et `_sampler` à `None`, et le constructeur
**rendait la main sans erreur**. La panne ne surgissait que bien plus loin,
dans `execute`, sous la forme d'un `AttributeError` sur `NoneType` — à des
dizaines de lignes de sa cause.

`execute.py` et `optimize.py` lèvent tous deux `ValueError("Unsupported
backend")` pour exactement la même valeur. Les trois sites disaient trois
choses différentes ; ils disent désormais la même.

## D-20 — le cache d'ansatz confondait deux Hamiltoniens

Le cache était indexé sur `(num_qubits, period_bound, reps)`. Or l'ansatz
QAOA encode `exp(−iγH)` : il dépend de l'Hamiltonien **terme par terme**,
pas seulement de la topologie.

Vérifié : deux Hamiltoniens sans aucun coefficient commun, à même nombre de
qubits et même `reps`, recevaient **le même objet**. Le second patch aurait
donc été optimisé contre la physique du premier — sans le moindre signal.

`get_ansatz` n'est appelé par aucun code du dépôt. C'était un **piège armé**,
prêt à se déclencher au premier branchement — précisément ce qu'un audit de
couverture ne voit pas et qu'un audit de contrat trouve.

La clé inclut désormais une empreinte des coefficients, arrondie à 12
décimales : un dernier bit ne fait pas exploser le cache, un écart de 1e−9
le sépare.

## Tests

`tests/test_runtime_contracts.py`, 20 tests : refus d'un backend inconnu et
message qui énumère les valides, aucun backend valide ne laisse une
primitive à `None`, deux Hamiltoniens différents ne partagent jamais un
ansatz, le même le retrouve, un seul coefficient suffit à manquer le cache,
l'empreinte est indépendante de l'ordre des termes mais sépare un
changement de signe.

---

# Audit de contrat des portes physiques — aucun défaut, deux constats

`src/Simulation/HamiltParams.py`. Cinq fonctions statiques portent tout le
raisonnement physique du mappeur v1 : `_f_gate` (Reynolds),
`_threshold_contrast` (contraste au seuil), `_g_strain` / `_g_rot`
(interrupteurs d'Okubo-Weiss) et `_g_mag` (activité magnétique). Leurs
docstrings énoncent des contrats précis — continuité, bornes, sens
d'activation.

**Les cinq honorent ce qu'elles annoncent.** Continuité de `_f_gate` au
raccord vérifiée à 1e−8 pour quatre valeurs de γ, monotonie sur cinq
décades, bornes respectées, aucun débordement sur `±1e300` ni sur `inf`.
`_threshold_contrast` rend **exactement** zéro au seuil et sous le seuil, et
garde bien un signal sur un domaine uniformément actif — la différence
revendiquée avec Michelson.

Deux constats structurels méritaient d'être écrits.

## Constat 1 — `g_strain` et `g_rot` ne sont pas deux interrupteurs

Elles somment à **1 exactement**, pour tout Q :

`1/(1+e^x) + 1/(1+e^−x) = 1`

| Q | `g_strain` | `g_rot` | somme |
|---|---|---|---|
| −10 | 1.000000 | 0.000000 | 1.0 |
| 0 | 0.500000 | 0.500000 | 1.0 |
| +10 | 0.000000 | 1.000000 | 1.0 |

Elles ne peuvent donc **jamais être actives ensemble, ni inactives
ensemble**. Le terme ZZ (porté par `g_strain`) et le terme ZZZZ (porté par
`g_rot`) sont une **partition d'un unique scalaire d'Okubo-Weiss**, pas deux
détecteurs indépendants.

Cela change la lecture d'une ablation : retirer le ZZ ne retire pas une
source d'information distincte du ZZZZ — cela déplace le poids d'un côté à
l'autre du même signal. C'est à rapprocher du résultat déjà consigné sur le
canal du circuit : l'architecture présente plus de degrés de liberté qu'elle
n'en a.

Un troisième cas de la même famille est mesuré au passage : dans la branche
hydrodynamique, `f_Re` et `mic_v` sont deux **reparamétrages monotones du
même scalaire** — `Re_h = v_jump·dx/ν` et `v_jump/v_jump_crit = Re_h/RE_CRIT`
sont égaux à 1e−12 près. Le coefficient présente deux facteurs physiques là
où il n'y a qu'une variable.

## Constat 2 — l'exemple de la docstring de `_f_gate` est inatteignable

Elle illustre la croissance logarithmique par « Re=3000, x_crit=10, γ=2 →
f ≈ 12 (not ∞) ». La formule rend bien **12.4076**, mais `f_max = 10.0` par
défaut la ramène à **10.0000** : la valeur citée ne peut jamais sortir de la
fonction telle qu'elle est appelée.

## Tests

`tests/test_gate_contracts.py`, 42 tests.

---

# D-21 — le flux descendait par un chemin qui efface ce qu'il mesure

`src/Simulation/RescaleArrays.py` — `get_adaptive_flux._process_flux`.

Trois quantités descendent du domaine plein vers la résolution du VQA : le
**score** classique, les **coefficients** d'Hamiltonien et le **flux de
contrainte** Φ. Les trois sont des indicateurs d'**anomalie** — leur raison
d'être est qu'un signal fort et isolé survive à la réduction.

Deux d'entre elles étaient max-poolées, et `_process_score` porte même un
`# No smoothing!` explicite. Le flux, lui, passait par un lissage 3×3 puis
`zoom(order=1)`, justifié par « smooth physical fields ». Or Φ n'est pas un
champ lisse : il est bâti sur des **différences** de champ et pique
exactement là où le score pique.

Un zoom bilinéaire **échantillonne**, il ne moyenne pas :

| réduction 128 → 4 d'un pic isolé | résultat |
|---|---|
| positions où le pic survit | **1 sur 256** |
| pic placé au centre | **0.0000** |
| même pic, max-pooling | 1000 |
| même pic, moyenne de bloc | 0.98 |

Le lissage préalable aggravait le tout : il diluait le pic **avant** de
l'échantillonner.

Part du pic de Φ conservée sur champs DNS réels (patch 128 → 4) :

| scénario | avant | après |
|---|---|---|
| orszag_tang | **38.0 %** | 100 % |
| mhd_rotor | **69.8 %** | 100 % |
| kelvin_helmholtz | 100 % | 100 % |
| harris_tearing | 100 % | 100 % |

Corrigé : les trois chemins appliquent désormais la même réduction. Le pic
est conservé à 100 % sur les quatre scénarios, et la carte de flux réduite
est maintenant **identique cellule par cellule** à la carte de score réduite
quand on leur donne la même entrée — ce qui n'était pas vérifiable avant.

Φ n'alimente que ψ. Comme D-11, le rayon d'action est exactement la quantité
dont l'ablation est au programme.

## Deux autres corrections dans la même passe

**`dns_validation.analyse_one`** utilise désormais les observables corrigées
`fluctuating_ke_fixed` et `mean_sq_current_fixed`. Les deux fonctions
d'origine restent en place, inchangées, pour reproduire à l'identique les
artefacts déjà publiés de phase 1b : le gel porte sur les **fonctions**, pas
sur l'analyse qui les appelle.

**`_f_gate`** — la docstring dit maintenant que `f ≈ 12.4` illustre la
formule et ne sort pas de la fonction, `f_max = 10.0` la ramenant à 10.0.

## Un test qui épinglait le défaut, retourné

`test_flux_takes_the_smoothing_path_and_loses_the_spike` affirmait « les flux
sont lissés puis interpolés, le pic doit s'y diluer ». Il décrivait
fidèlement le code ; c'est la justification qui ne tenait pas. Il vérifie
désormais l'inverse, et que le flux suit **exactement** la réduction du
score.

`tests/test_padded_rescale_contracts.py` passe de 37 à 45 tests.

---

# D-22 — les hyperparamètres déployés n'ont aucune provenance reproductible

`results/hyperparams/` est déclaré **entrée gelée** de l'étude. Il contient
deux choses qui devraient être d'accord :

- `optuna_studies/*.db` — les bases de la campagne, 345 essais
- `best_hyperparams.json` — ce que `src/hyperparams_loader.py` charge

**Elles ne le sont pas.** Vérifié directement dans les fichiers.

## Ce que les bases ont échantillonné, et ce que le JSON déploie

| étude | paramètres échantillonnés |
|---|---|
| `q_has_v2_phase1.db` (202 essais) | `beta`, `beta_curl`, `beta_xpoint`, **`sigma`**, `w_z_frac` |
| `classical_v2_phase1.db` (143) | `threshold_amr` |

| paramètre déployé | origine |
|---|---|
| `beta`, `beta_curl`, `beta_xpoint`, `w_z_frac` | échantillonnés (étude quantique) |
| `threshold_amr` | échantillonné dans l'étude **classique** seulement |
| `gamma_hydro`, `gamma_mag`, `kappa` | **aucune base ne les a jamais échantillonnés** |
| `sigma` | échantillonné, **absent du JSON** |

## Trois écarts

**1. Trois valeurs sur huit n'ont aucune origine dans le dépôt.**
`gamma_hydro = 2.127`, `gamma_mag = 2.361`, `kappa = 14.332` ne figurent
dans aucune base.

**2. `sigma` est optimisé puis jeté.** La campagne l'échantillonne et son
meilleur essai trouve **0.0230** ; le JSON ne le contient pas, donc
`pipeline.py` retombe sur `_defaults.get('sigma', 0.05)` — une constante
codée en dur. σ est la largeur de la fenêtre gaussienne, le paramètre au
cœur de D-9.

**3. L'essai déclaré ne correspond pas.** Le JSON annonce l'essai 85 avec
une perte de 0.2215. L'essai 85 existe, sa perte vaut **0.3213**, et **aucun**
de ses quatre paramètres communs ne coïncide :

| paramètre | base | JSON |
|---|---|---|
| `beta` | 6.034464 | 0.549537 |
| `beta_curl` | 1.318670 | 0.819924 |
| `beta_xpoint` | 2.341306 | 0.425647 |
| `w_z_frac` | 39.599016 | 0.101338 |

## Le code d'entraînement, lui, est cohérent

`train_hyperparams` code en dur `threshold_amr = 0.14959824837662078` avec
le commentaire « le meilleur classique ». C'est **exactement** la valeur du
meilleur essai classique (#42, perte 0.2148). Le code et les bases sont
d'accord ; **c'est le JSON qui est orphelin**.

Conséquence : le bras quantique est déployé à `threshold_amr = 0.3044`, une
valeur qui ne figure pas parmi les 125 essayées, et à laquelle il n'a jamais
été entraîné — l'objectif l'a toujours fixé à 0.1496.

## Ce que cela change pour la suite

Une réoptimisation n'est pas une amélioration : **c'est la seule façon
d'avoir des hyperparamètres qui existent.** Aucun résultat de performance ne
peut être attribué à un réglage dont on ne sait pas d'où il vient.

Corollaire pour le périmètre : `gamma_hydro`, `gamma_mag` et `kappa` n'ont
jamais été optimisés par la campagne gelée. Les inclure dans la
réoptimisation n'est donc pas une *re*-optimisation, c'est une première.

## Autres constats sur `train_hyperparams`

`make_composite_objective` présente quatre paramètres comme conditionnels
(`if "x" not in frozen:`) alors qu'ils sont des **constantes** :
`threshold_amr`, `gamma_hydro = 2.0`, `gamma_mag = 0.5`, `kappa = 10.0`.
L'espace de recherche réel est donc de cinq paramètres, pas neuf — et les
trois constantes de l'objectif ne valent pas non plus ce que le JSON
déploie.

## Tests

`tests/test_hyperparams_provenance_break.py`, 16 tests. Ils **épinglent**
l'écart plutôt que de le masquer, et chacun dit dans sa docstring ce qui
devra être vrai après réoptimisation. Le dernier,
`test_every_deployed_hyperparameter_should_one_day_be_traceable`, est le
**critère d'acceptation** : il est en `xfail` aujourd'hui et passera sans
modification le jour où chaque valeur déployée sera traçable à un essai.

---

# Correction d'une affirmation : le splitting de Strang ne s'applique pas ici

J'ai écrit à plusieurs reprises, dans `docs/RESULTS.md` et dans le plan,
qu'« un splitting de Strang rendrait l'ordre 2 ». **C'est faux, et la mesure
le montre.**

Un splitting symétrique suppose deux **flots** qu'on peut découper en
demi-pas. La projection d'incompressibilité n'en est pas un : c'est un
**projecteur idempotent**, et `P^(1/2)` n'a pas de sens.

Mesuré, N=128, Orszag-Tang, grille fixe, quatre résolutions temporelles :

| schéma | 64 pas | 128 | 256 | 512 | ordre |
|---|---|---|---|---|---|
| `P ∘ RK4` (actuel) | 8.4138e−6 | 4.0713e−6 | 1.8999e−6 | 8.1426e−7 | 1.05 → 1.22 |
| `P ∘ RK4 ∘ P` | 8.4138e−6 | 4.0713e−6 | 1.8999e−6 | 8.1426e−7 | 1.05 → 1.22 |

**Identiques à la dernière décimale.** L'explication est immédiate : après le
premier pas l'état est déjà dans le sous-espace à divergence nulle, donc la
projection initiale est l'identité. Le « Strang » que j'avais écrit *est*
le schéma de Lie.

Le bon cadre est celui d'un système **différentiel-algébrique** : l'ordre
chute parce que la contrainte est imposée *après* un pas RK4 non contraint.
Deux corrections tiennent — projeter le **second membre** à chaque étage, ce
qui rend le champ intégré à divergence nulle par construction, ou passer à
une formulation à pression.

Le plan a été corrigé en conséquence.

---

# D-24 — la contrainte imposée après le pas ramenait l'ordre 4 à 1,2

`src/Simulation/solver.py` — `_rk4_step`.

Le système est **différentiel-algébrique** : v et B doivent rester à
divergence nulle. `step_full` appliquait RK4 puis projetait l'**état** — un
splitting de Lie, d'ordre 1. En projetant le **second membre** à chaque
étage, le champ intégré est à divergence nulle *par construction* et RK4
garde son ordre.

Mesure à grille **fixe** (N=96, T=0,5, Orszag-Tang), en ne raffinant que le
pas de temps, chaque schéma comparé à sa propre référence à 1024 pas :

| schéma | 32 pas | 64 | 128 | 256 | ordre | max\|div v\| |
|---|---|---|---|---|---|---|
| projection de l'**état** | 1,098e−2 | 5,396e−3 | 2,539e−3 | 1,093e−3 | 1,03 → 1,22 | 5,04e−3 |
| projection du **second membre** | 8,610e−8 | 5,381e−9 | 3,362e−10 | **2,092e−11** | **4,00 / 4,00 / 4,01** | 5,11e−3 |
| aucune projection | 1,908e−3 | 1,234e−4 | 7,705e−6 | 4,790e−7 | 3,95 → 4,01 | **5,89e+0** |

La correction rend **les deux** : l'ordre 4 du schéma — erreur **52 000 fois
plus petite** à 256 pas — et le contrôle de la divergence au même niveau
qu'avant. Ne pas projeter du tout donne aussi l'ordre 4, mais laisse la
divergence exploser d'un facteur **1150**.

## Deux vérifications avant d'annoncer le gain

Un gain de dix ordres sur horizon court sentait le piège. Deux contrôles :

- **le champ évolue** — déplacement relatif de `vx` sur T=0,02 : 8,4813e−3
  pour les deux schémas, identiques à 6e−6 près ;
- **le second membre projeté n'est pas annulé** — il conserve **30,0 %** de
  sa norme.

La première mesure (T=0,02) était simplement au plancher de la double
précision dès 32 pas : l'ordre n'y était pas mesurable. Il a fallu allonger
l'horizon à T=0,5 pour l'obtenir.

## Ce que cela ferme

Le plan annonçait le facteur limitant du solveur comme « identifié et
corrigeable ». Il est corrigé, et la valeur de « corrigé » est chiffrée. La
mention d'un splitting de Strang est retirée : elle ne s'applique pas — un
splitting symétrique suppose deux *flots* découpables en demi-pas, alors que
la projection est un **projecteur idempotent**. Vérifié : `P ∘ RK4 ∘ P` rend
des erreurs identiques à `P ∘ RK4` à la dernière décimale.

## La correction n'est pas applicable en l'état — `PROJECT_RHS = False`

J'ai d'abord activé la correction par défaut. **Elle casse le chemin AMR**,
et la suite de tests l'a montré : huit échecs, dont six sur des tests
préexistants.

`_rk4_step` a **trois** appelants, pas un :

| appelant | champ | projection |
|---|---|---|
| `step_full` | global périodique | **valide** |
| `step_layered` phase 1 | global **sous-échantillonné** | lève — `(256,256)` contre `(8,8)` |
| `step_layered` phase 2 | **patch local** avec halo | **pas périodique** : une projection spectrale périodique n'y est pas définie |

Projeter les deux premiers et pas le troisième romprait la garantie « à
`max_depth`, `step_layered` est identique à `step_full` » — propriété
documentée et testée.

Le drapeau reste donc à `False`, avec la raison écrite dans le code et un
test qui vérifie qu'elle y reste. Le choix — projection par taille de
grille, formulation à pression, ou autre — est une **décision de
modélisation**, pas une correction de défaut.

Trois voies, par coût croissant :

1. **Laisser en l'état.** Le solveur reste d'ordre 1,2, mesuré et documenté
   comme limite. La chute est **commune aux deux bras**, donc elle ne biaise
   pas leur comparaison.
2. **Projection par taille de grille**, plus une décision sur les patchs non
   périodiques. Casse probablement la garantie AMR.
3. **Formulation à pression** — la voie propre, mais c'est réécrire le cœur
   du solveur.

## Un test à moi était faux

`test_the_projected_rhs_is_divergence_free_at_every_stage` exigeait une
divergence **aux différences finies** nulle. Elle vaut 1,15e−2 — et ce n'est
pas un défaut de la projection : celle-ci est **spectrale**, elle annule la
divergence de Fourier, pas celle du stencil FD4. C'est exactement
l'incompatibilité déjà signalée entre le second membre (FD4) et la
projection (FFT). Mauvais opérateur choisi. Le test vérifie désormais ce que
la projection promet réellement : idempotence, et divergence **spectrale** à
la précision machine.

---

# Van Kan mesuré — non concluant ; et D-25, la projection qui abîmait B

## La question

`step_full` applique RK4 puis projette l'état : un splitting de Lie, d'ordre 1.
La correction de pression incrémentale (Van Kan) promet l'ordre 2 en ajoutant
le gradient du potentiel du pas précédent au second membre.

**Commande** — `scratchpad/vankan.py`, N=64, T=0,05, Orszag-Tang, grille fixe,
chaque schéma comparé à sa propre référence à 2048 pas.

## Le tableau

| schéma | erreur 32 pas | 256 pas | ordre | div_FD B | div_FD v |
|---|---|---|---|---|---|
| projection v **et** B *(actuel)* | 1,0665e−04 | 1,1853e−05 | 1,02 → 1,10 | **4,877e−06** | 2,914e−06 |
| projection de **v seul** | 1,0665e−04 | 1,1853e−05 | 1,02 → 1,10 | **2,818e−14** | 2,914e−06 |
| **Van Kan** sur v, B non projeté | 1,0411e−04 | 1,1563e−05 | 1,02 → 1,10 | 2,928e−14 | 2,914e−06 |

## Verdict Van Kan : non

**L'ordre reste à 1,10 dans les trois cas.** Van Kan gagne 2,4 % sur l'erreur,
rien de plus. La théorie promet l'ordre 2 ; on ne l'obtient pas.

Hypothèse **non vérifiée** : le même désaccord FD/spectral que D-25. Le
gradient de φ est calculé spectralement et ajouté à un second membre FD4,
donc la correction n'annule pas ce qu'elle devrait. Le vérifier demanderait
un solveur de Poisson FD-cohérent — la réécriture qu'on cherchait à éviter.
L'implémentation elle-même peut aussi être fautive.

## D-25 : la projection spectrale abîmait un champ déjà solénoïdal

La deuxième ligne est un gain net, et il ne doit rien à Van Kan.

L'induction est en forme rotationnelle : `rhs_B = (∂Ez/∂y, −∂Ez/∂x)`. Sa
divergence **aux différences finies** vaut `∂²Ez/∂x∂y − ∂²Ez/∂y∂x`, exactement
nulle puisque les décalages de `np.roll` commutent. **B est solénoïdal par
construction, dans l'opérateur même qui construit le second membre.**

La projection, elle, est **spectrale**. Appliquée à ce champ, elle n'y nettoie
rien : elle y injecte le désaccord entre les deux opérateurs.

| divergence FD4 du champ B, Orszag-Tang N=64 | |
|---|---|
| second membre | 1,97e−14 |
| état, 50 pas **sans** projection | 1,00e−14 |
| état, 50 pas **avec** projection | **4,63e−07** |

**Huit ordres de grandeur sur la contrainte, pour une erreur identique à la
quatrième décimale.** La projection de B ne coûtait rien en précision et
dégradait la seule chose qu'elle était censée garantir.

La vitesse, elle, en a besoin : `div_FD(rhs_v)` vaut **4,17** en relatif.

`PROJECT_B = False` par défaut ; `True` reproduit le chemin historique bit à
bit. La garantie AMR — `step_layered` ≡ `step_full` à `max_depth` — tient
toujours à **3,331e−16**.

## Note de méthode : l'opérateur de mesure décidait du verdict

La première mesure de ce défaut, faite avec la divergence **spectrale**, ne
montrait rien : 9,5e−02, indistinguable du bruit. C'est en la refaisant avec
l'opérateur **assorti** — le même stencil FD4 que le second membre — que
l'écart de huit ordres apparaît.

Troisième fois que ce piège se referme dans ce dépôt. Une grandeur discrète
n'a de valeur que relativement à l'opérateur qui la calcule ; mesurer avec un
autre ne mesure pas le champ, mais l'écart entre deux opérateurs.

## Tests

`tests/test_solver_convergence.py`, 7 tests : l'induction préserve la
divergence de B (contrôle positif), le second membre de v **ne** la préserve
pas (contrôle négatif, sans quoi on croirait qu'aucune projection n'est
nécessaire), la projection dégrade la contrainte de quatre ordres au moins,
la vitesse reste projetée, la raison est écrite dans la docstring, et le
retrait ne coûte rien en précision.

---

# D-27 — la projection amputait la perturbation de quatre scénarios

**Commande.** `pytest tests/solver/test_scenarios_analytic.py -k "solenoidal or amputates"`
· commit de la mesure : `git rev-parse HEAD`

## Ce qui se passait

En 2-D, `div B = ∂Bx/∂x + ∂By/∂y = 0`. `harris_tearing` pose un `Bx` qui ne
dépend que de `y` — donc `∂Bx/∂x = 0` — et une perturbation
`δBy = ε·cos(kx)·sech²(y)` dont `∂By/∂y ≠ 0`. La perturbation **viole** donc
la contrainte, et la projection la rabote pour la rétablir.

Le défaut n'était pas visible tant que B était projeté : la projection
masquait la divergence qu'elle corrigeait. Il n'apparaît qu'une fois D-25
corrigé — **trouvé en retirant une couche, pas en posant une question.**

| scénario | div_FD B relative | perturbation conservée |
|---|---|---|
| **`harris_tearing`** *(déployé)* | 2,801e−03 | **27,5 %** |
| `island_coalescence` | 1,400e−02 | **27,5 %** |
| `noisy_uniform` | 4,947e−01 | 55,7 % |
| `double_tearing` | 9,062e−04 | 77,3 % |

`harris_tearing` amorce son mode de déchirement par cette perturbation. **La
projection en retirait 72,5 %.** Le plan notait que ce scénario « dégénère
dans toutes les configurations testées, sans explication » ; ceci en est
peut-être une part — **non affirmé**, seule l'amplitude initiale est mesurée.

## La correction, et sa première version fausse

La perturbation s'écrit comme le rotationnel d'une fonction de flux,
`δB = ∇×(ψ ẑ)`, solénoïdale par construction — comme cela avait déjà été fait
pour `magnetic_twist` et `ghost_twisting`.

**Première version : dérivées analytiques de ψ.** Résultat `div_FD B` =
**2,1e−05**, pas 1e−16. Le champ était exactement solénoïdal pour l'opérateur
continu, et faux pour celui du solveur — l'erreur mesurée était l'erreur de
discrétisation. **Cinquième occurrence du même piège** : mesurer une grandeur
discrète avec un stencil autre que celui qui l'a produite.

**Version retenue :** dériver ψ avec le **même** stencil FD4 que le second
membre. `div(rot ψ) = ∂x∂y ψ − ∂y∂x ψ` est alors exactement nul, parce que les
deux dérivées FD4 sont des combinaisons de `np.roll` et commutent.

```python
@staticmethod
def _curl_z_fd4(psi, dx):
    g_x, g_y = MHDSolver._fd_grad(psi, dx)
    return g_y, -g_x
```

À N=64, `div_FD B` normalisée par `max|Bx| + max|By|` et multipliée par `dx` :

| scénario | div_FD B relative | \|δB\| max |
|---|---|---|
| `harris_tearing` | **1,076e−16** | 0,010000 |
| `double_tearing` | **1,863e−16** | 0,010000 |
| `island_coalescence` | **9,862e−17** | 0,050000 |
| `noisy_uniform` | **1,275e−16** | 0,239451 |

Ces valeurs sont au niveau du bruit d'arrondi : elles bougent au dernier
chiffre avec la résolution. Les tests posent donc un **seuil** (`< 1e−12`),
pas une égalité — un test calibré sur la mesure du jour cesserait de mesurer
au premier changement légitime.

Douze à treize ordres de grandeur, et l'amplitude nominale entièrement
conservée. Les champs de ces quatre scénarios changent : **tout nombre publié
qui les traverse est à refaire.**

## Tests

`tests/solver/test_scenarios_analytic.py`, 159 tests dont
`test_a_flux_function_perturbation_is_exactly_divergence_free` et
`test_an_analytic_derivative_would_not_have_been_exact` — ce dernier épingle
*pourquoi* la première version ne suffisait pas, avec sa mesure.

---

# D-29 à D-36 — audit du script d'entraînement

**Commandes.**
`pytest tests/pipeline/test_train_hyperparams_contracts.py` (60 tests, ~14 s)
`pytest tests/pipeline/test_train_hyperparams_smoke.py` (7 tests, ~16 s)

Quatre variantes du script d'entraînement coexistaient
(`TrainHyperParam_v1/v2/v3/v4.py`, 3 009 lignes). **Trois sont supprimées**,
la quatrième renommée `src/train_hyperparams.py`. C'est elle qui tournera sur
les cœurs loués ; l'audit porte sur elle seule.

## D-29 — le jeu « isolé » contenait les scénarios complexes

Tout l'argument du protocole est qu'Orszag-Tang mélange les classes
d'anomalies, donc qu'il faut des scénarios qui en isolent une. La liste
disait :

```python
SCENARIOS_ISOLATED = [("kh", …), ("ot", …), ("tearing", …), ("rotor", …)]
SCENARIOS_COMPLEX  = [("ot", …), ("rotor", …)]
SCENARIOS_ALL      = SCENARIOS_ISOLATED + SCENARIOS_COMPLEX
```

Trois conséquences, toutes silencieuses :

- le jeu « isolé » n'isolait rien : il contenait les deux scénarios complexes ;
- `SCENARIO_VORTEX` et `SCENARIO_COALESCENCE` étaient **définis et jamais
  utilisés** ;
- `SCENARIOS_ALL` valait **6 entrées pour 4 classes distinctes**. La perte
  composite `mean(Loss_i)` divisait par 6 une somme où `ot` et `rotor`
  entraient deux fois : pondération **2:1** contre `kh` et `tearing`, pour le
  **double du coût** de simulation.

**Ce qui tranche.** Le JSON déployé porte, pour sa phase 1, un bloc
`per_scenario` qui liste `kelvin_helmholtz`, `lamb_oseen_vortex`,
`harris_tearing`, `island_coalescence`. C'est la liste des quatre isolés qui a
produit la campagne gelée : la version trouvée dans le code était une
**régression**, pas une intention.

Corrigé, plus une garde `_assert_scenarios_wellformed` qui refuse un doublon,
un jeu vide, ou une trace DNS manquante — et qui refuse **à la construction de
l'objectif**, pas au milieu du premier essai, c'est-à-dire après le pré-calcul
DNS.

**Conséquence pour `study/`.** `closed_loop_campaign.fold_scenarios`
dédoublonnait ce défaut à la main pour éviter de fabriquer une fuite LOSO. La
déduplication ne retire plus rien, mais il y a désormais **six** folds LOSO
possibles au lieu de quatre : les résultats publiés sur quatre folds ne sont
pas comparables terme à terme.

## D-30 — le chemin séquentiel ne pouvait pas finir

```python
study_p1 = _run_phase1(dns_traces)      # ligne 1346
…
study_p1 = _run_phase1(study_p1, dns_traces)   # ligne 1352 — deux arguments
```

`_run_phase1` prend **un** argument. Le chemin par défaut — celui qu'on obtient
sans `WORKER_PHASE` — lève donc `TypeError` **après** la phase 1, c'est-à-dire
après ses 600 essais. `_save_results(study_p1, study_p1, …)` passait par
ailleurs la même étude deux fois, dans les emplacements « phase 1 » et
« phase 1b », et lisait des clés `vortex` / `coalescence` qui n'existaient
dans aucune des deux.

## D-31 — un paramètre optimisé que rien ne lit

Avec `split_michelson=False`, la phase 1 proposait `beta_michelson` à Optuna.
`pipeline.py` ne lit ce nom **nulle part** : il n'apparaît que dans un bloc mis
en commentaire. La phase optimisait donc un paramètre sans effet sur la perte.
Le chemin vivant passait `split_michelson=True`, donc le défaut n'a pas
produit de nombre faux — mais l'option existait, documentée comme le
comportement de la phase 1.

`split_michelson`, `beta_michelson` et la « phase 1b » sont supprimés.
`make_phase3_objective` — 64 lignes, aucun appelant, sa propre copie non nommée
des quatre constantes — et `expand_split_beta_seeds` — qui mutait son argument
et dont le repli `params.pop(k, params.get(k, 0.5))` n'atteignait jamais le
`get` — sont supprimés aussi.

## D-32 — l'élagage était décoratif

L'objectif composite rapportait **une** valeur, au step 0 :

```python
trial.report(composite, step=0)
if trial.should_prune(): …
```

sous un `MedianPruner(n_warmup_steps=2)`. Un pruner ne mord jamais avant
`n_warmup_steps` : `should_prune()` au step 0 renvoie toujours `False`.

**Mesure.** 40 essais terminés à 1,0 ; un essai qui rapporte **1e9** au step 0 :
`should_prune()` = `False`. Le garde-fou n'a jamais élagué un seul essai.

Corrigé : la moyenne **courante** est rapportée après chaque scénario, au step
égal à son indice. Le même 1e9 rapporté aux steps 0, 1, 2 déclenche désormais
l'élagage — et un essai élagué au 3ᵉ scénario ne simule pas le 4ᵉ, ce qui est
tout l'intérêt sur des cœurs loués. La comparabilité entre essais impose que
l'ordre des scénarios soit fixe : `SCENARIOS_*` sont des tuples.

## D-33 — Orszag-Tang tournait sans anomalies avancées

`create_argus` lisait `scenario_config.get("AdvAnomaliesEnable", False)`.
`SCENARIO_OT` était le seul des six à ne pas porter la clé.

**Mesure.** `create_argus(SCENARIO_OT).AdvAnomaliesEnable` = `False`,
`create_argus(SCENARIO_ROTOR).AdvAnomaliesEnable` = `True`.

Le terme ZZZZ de point X n'existe pas sans anomalies avancées. La phase 2
entraînait donc `beta_xpoint` sur un jeu de deux scénarios dont **l'un ne
pouvait pas l'exprimer**. Le même oubli existait dans la table `PHASE` de
`pipeline.main()`.

Corrigé aux deux endroits, et `create_argus` **lève** désormais si une clé
manque : le repli silencieux sur une valeur valide est exactement le motif
qu'on cherche à éliminer.

## D-34 — le budget d'essais était multiplié par le nombre de workers

```python
remaining = phase_config["n_trials"] - trials_done      # calculé UNE fois
study.optimize(objective_fn, n_trials=remaining)
```

Chaque worker lit le compte **au démarrage**. N workers lancés ensemble lisent
tous « 0 fait » et demandent chacun la campagne entière.

**Mesure.** 4 workers, cible 12 essais : **48 essais** exécutés. À l'échelle
réelle — 8 cœurs, `n_trials=600` — cela ferait **4 800 essais au lieu de 600**,
huit fois le coût annoncé.

Corrigé : la boucle relit le compte à chaque essai et s'arrête dès la cible
atteinte, quel que soit le nombre de workers. Coût : une lecture de base par
essai, contre 10 à 20 minutes de calcul. `WORKER_TRIALS` reste un plafond
**par worker**, pour une durée de location bornée.

## D-35 — le JSON final ne portait pas de quoi redéployer

`_save_results` écrivait `study_p3.best_params`. Or `best_params` ne contient
que ce qu'Optuna a **échantillonné** : les paramètres fixes n'y sont pas. Le
fichier déployé était donc structurellement incomplet, et le déploiement
comblait les manques par des replis que personne n'avait choisis.

**C'est le mécanisme de D-22** : `sigma` disparu du JSON, `gamma_hydro`,
`gamma_mag` et `kappa` présents dans le fichier déployé alors qu'**aucune base
Optuna ne les a jamais échantillonnés**.

Corrigé sur deux plans :

- chaque essai porte son dictionnaire d'hyperparamètres **résolu** — exploré +
  fixe — en `user_attr` ;
- le JSON porte ce dictionnaire, plus l'espace de recherche avec ses bornes,
  les paramètres fixes, la liste des scénarios, `lambda_cost`, le hash du
  commit, la propreté de l'arbre de travail, et `sys.argv`.

`deployable_params` signale par ailleurs, plutôt que de le taire, le cas où il
doit reconstruire faute d'attribut résolu.

## D-36 — la provenance de `sigma` n'existait que sur les runs jetés

`pipeline` a **quatre** sorties `return_details`. Une seule portait `sigma` et
`sigma_source` : celle du chemin d'exception de scoring. La trace exigée par
D-22 n'existait donc que sur les runs **divergés** — jamais sur ceux qu'on
publie.

Les quatre passent désormais par un `_details` unique. Le test ne cherche plus
une chaîne dans le source : il parcourt l'AST de `pipeline` et vérifie
qu'aucune sortie sous `if return_details` ne s'échappe du helper.

## L'espace de recherche, désormais déclaré

Le périmètre décidé pour la réoptimisation est de **8 paramètres** :

| paramètre | bornes | échelle |
|---|---|---|
| `beta` | 0,5 – 10,0 | linéaire |
| `w_z_frac` | 0,10 – 1000 | log |
| `sigma` | 0,02 – 0,30 | linéaire |
| `beta_curl` | 0,0 – 5,0 | linéaire |
| `beta_xpoint` | 0,0 – 5,0 | linéaire |
| `gamma_hydro` | 0,1 – 5,0 | linéaire |
| `gamma_mag` | 0,1 – 5,0 | linéaire |
| `kappa` | 0,5 – 50,0 | log |

`threshold_amr` reste **fixé** à 0,14959824837662078 — le meilleur essai de
l'étude classique — pour que la comparaison porte sur ce que le quantique
ajoute et non sur un seuil différent. C'est une décision, elle est déclarée
dans `FIXED_PARAMS` et vérifiée par un test qui exige qu'elle tombe dans les
bornes que le bras classique avait le droit d'explorer.

Trois de ces huit — `gamma_hydro`, `gamma_mag`, `kappa` — n'ont **jamais** été
échantillonnés par aucune campagne : pour eux ce sera une première, pas une
reprise. Nuance qui allège : `g_strain + g_rot ≡ 1` exactement, donc `kappa`
ne pilote **qu'un** degré de liberté.

**Réserve consignée.** La borne haute de `w_z_frac` vaut 1000 alors que le
paramètre est documenté comme une *fraction* de la médiane des couplages. Elle
vient de la campagne gelée, dont la graine valait 500. Conservée telle quelle
pour ne pas changer la science en même temps que le code — mais elle est à
trancher avant la campagne.

## Ce qui vérifie tout cela avant de louer des cœurs

```bash
python src/train_hyperparams.py --print-space     # l'espace réel, sans rien calculer
pytest tests/pipeline/test_train_hyperparams_contracts.py -q    # 60 tests
pytest tests/pipeline/test_train_hyperparams_smoke.py -q        # 7 tests
```

Le second fichier n'est pas une simulation : il fait tourner le **vrai**
solveur, le **vrai** circuit, une **vraie** base Optuna et écrit un **vrai**
JSON de déploiement — à N=32, deux pas de temps, une profondeur de
raffinement, en 16 secondes. Les six scénarios y passent. Une campagne d'une
semaine ne doit pas être le premier endroit où l'on découvre qu'un scénario ne
s'initialise pas.

Ce qu'il ne montre pas : que l'objectif **discrimine**. À cette résolution il
n'y a qu'une décision de raffinement, donc les six sous-pertes sont égales
(0,285714). Il montre que le chemin complet s'exécute et que les artefacts en
sortent complets.

---

# D-37 — le biais Z et les couplages décrivaient deux grilles différentes

**Commande.** `pytest tests/amr/test_patch_encoding_shapes.py` (13 tests, ~16 s)

**Pourquoi on est allé voir.** `COUVERTURE.md` listait trois fonctions du
chemin d'entraînement dont le contrat n'avait jamais été audité. Elles
décident la valeur qu'une campagne d'une semaine va minimiser. Celle-ci est
tombée en instrumentant la marge du garde CFL : le pipeline **plantait** dès
qu'on le lançait avec `max_depth ≥ 2`.

## Ce qui se passait

`_prepare_vqa_input` construit les deux moitiés de l'Hamiltonien par deux
chemins distincts :

| | source | taille rendue |
|---|---|---|
| `C_edges`, `K_plaquettes`, `K_xpoint` | les **champs**, via `_downsample_fields(..., target_dim, pad)` | (4, 4) |
| `H_edges` — le biais Z | le **score**, via `_process_score(..., target_dim + 2·pad)` | **(6, 6)** |

À `depth > 0`, `_process_score` emprunte `_resize_padded_maxpool`, dont le
contrat est écrit dans sa docstring : *« Input shape: (N+2, M+2). Output
shape: (t_dim+2, t_dim+2). »* **Le halo est déjà ajouté par la fonction.**
L'appelant l'ajoutait une seconde fois.

`create_bounded_hamiltonian(dim=2)` indexe ses lectures par `dim` sur des
tableaux supposés `(dim+2, dim+2)`. Devant un tableau **trop grand**, il ne
lève pas : il lit le coin supérieur gauche et rend un Hamiltonien
parfaitement valide, calculé sur la mauvaise portion du patch. Le biais Z
d'un patch venait donc du **quart haut-gauche** de ce patch, plus un halo
situé deux cellules trop loin.

**Mesure**, `orszag_tang` à N=64 après 40 pas, patch (0:32, 0:32), cœur 2×2 :

```
ce qui était lu               ce qui aurait dû l'être
[[0.109 0.109 0.097 0.064]    [[0.109 0.109 0.064 0.005]
 [0.111 0.141 0.094 0.064]     [0.111 0.141 0.078 0.079]
 [0.095 0.096 0.070 0.065]     [0.068 0.073 0.106 0.106]
 [0.062 0.062 0.044 0.081]]    [0.069 0.078 0.075 0.079]]
```

Écart maximal **0,05814**, pour des coefficients dont le plus grand vaut
**0,14107** : **41 %**. Les deux premières colonnes coïncident — c'est le
recouvrement du coin — et tout le reste décrit une autre région.

## Depuis quand, et sur quoi

```
git log -S "target_dim + 2 * pad" -- src/Simulation/refinement.py
cf93ba3 2026-04-09 Q-HAS: report submission snapshot
```

**Depuis le premier commit du fichier.** Tous les niveaux de raffinement sauf
le premier passent par là : à `max_depth = 4`, trois niveaux sur quatre.
`depth = 0` est le seul épargné, parce qu'il est périodique et n'a pas de
halo.

Le garde de forme ajouté en auditant les mappeurs (`7c0ae2f`) transforme
depuis lors la lecture silencieuse en `ValueError`. C'est ce garde qui a
rendu le défaut visible — mais il rend aussi **le pipeline inutilisable à
`max_depth ≥ 2`** : dans l'état d'avant ce correctif, la campagne n'aurait
pas pu tourner du tout.

## Pourquoi les tests ne l'avaient pas vu

Les configurations rapides — celles qu'on écrit pour qu'un test tourne en
quelques secondes — utilisent `max_depth_override = 1`. À `max_depth = 1`, le
balayage traite `depth = 0` puis s'arrête : **le chemin borné n'est jamais
emprunté**. Le test de régression paramètre donc explicitement
`max_depth ∈ {1, 2, 3}`.

## Correction

Une ligne : `_process_score(local_score, depth == 0, target_dim)`. Le halo
vient de la fonction, une fois.

| | avant | après |
|---|---|---|
| `H_edges` à depth>0 | (6, 6) | **(4, 4)** |
| `C_edges` à depth>0 | (4, 4) | (4, 4) |
| pipeline à `max_depth=2` | `ValueError` | **s'exécute sur les 6 scénarios** |

**Tout nombre Q-HAS publié qui traverse un raffinement au-delà du premier
niveau est affecté.**

## Au passage — le garde CFL, vérifié et trouvé sain

C'est en cherchant sa marge que D-37 est tombé. `pipeline` abandonne l'essai
si `check_cfl() > 1.0`. `adapt_dt` et `check_cfl` emploient bien la **même**
vitesse rapide `c_fast = max|v| + max|B|`, donc la CFL réalisée vaut la cible.
Mesuré à travers le vrai pipeline, `max_depth = 2`, N=64 :

| scénario | CFL max | marge |
|---|---|---|
| kh, vortex, tearing, coalescence, rotor | 0,4000 – 0,4018 | **2,5×** |
| orszag_tang | 0,4042 | **2,47×** |

**Rectification.** Une première mesure donnait 0,755 pour `orszag_tang`, soit
une marge de 1,3×. Elle était fausse : j'avais instancié le solveur avec les
`Re`/`Rm` par défaut au lieu de 800, donc une viscosité différente de celle
de la trace DNS. Le garde n'est pas serré.

---

# D-38 — trois gardes de `execute` qui ne tenaient que sur le chemin testé

**Commande.**
`pytest tests/quantum/test_runtime_contracts.py -k "bound or null_hamiltonian or optimizer"`

Même famille que D-37 : des protections correctes là où on les regarde, et
absentes ailleurs.

## Le warm start passait outre un Hamiltonien nul

Quand tous les coefficients sont nuls, `execute` court-circuite l'optimisation
et — dit son commentaire — *« returning θ-init marginals »*. Il reprenait
pourtant les paramètres du warm start s'il y en avait un. Or sans terme de
coût, seul le mixer agit : il tourne l'état sans qu'aucun coût ne le justifie.

**Mesure**, 8 qubits, score classique 0,700, warm start β = (0,35 ; 0,30) :

| | marginales rendues |
|---|---|
| sans warm start | **0,7000** |
| avec warm start | **0,5535** |

21 % de déplacement sur une décision annoncée inchangée. Corrigé :
`optimal_params = np.zeros(2·reps)`, sans condition.

*Portée.* Depuis D-8, `create_period_hamiltonian` et
`create_bounded_hamiltonian` **lèvent** sur un Hamiltonien nul : la branche
n'est plus atteignable par `mapping`. Elle reste fausse là où elle est.

## Powell recevait des contraintes que scipy ignore

Le commentaire de `execute` explique pourquoi le mixer doit rester borné :
sans borne, COBYLA part à β = 1, rabat P(|1⟩) à ≈ 0,25 et **supprime tout
raffinement**. La borne était posée par `bounds` pour L-BFGS-B et par
`constraints` pour `("COBYLA", "Powell")`.

Powell n'accepte pas de contraintes. scipy le disait :

```
RuntimeWarning: Method Powell cannot handle constraints.
OptimizeWarning: Unknown solver options: rhobeg
```

…sur stderr, dans un essai parmi des centaines. Powell optimisait donc le
mixer **sans borne**. Corrigé : Powell passe par `bounds`, `constraints` reste
à COBYLA seul. Toute autre méthode **lève** désormais, au lieu de perdre la
borne en silence.

*Note de méthode.* Ma première correction a créé le même trou : elle refusait
les méthodes inconnues mais laissait Powell dans la liste des trois
autorisées. Le test qui l'a rattrapée est celui qui vérifie **|β| dans le
résultat**, pas celui qui vérifie que l'appel passe. Un test qui constate
qu'une fonction « ne plante pas » ne teste pas sa garantie.

## Le mode MPS écrasait définitivement le nombre de tirs

`sampler.options.default_shots = max(shots, 8192)` était appliqué à un objet
qui peut appartenir à `vqa_runtime`, donc **partagé par toute la campagne**.
Après un seul patch en MPS, chaque appel ultérieur tirait 8 192 coups quel que
soit `shots`. La valeur d'origine est restaurée après lecture.

*Portée.* Le backend déployé est `state_vector` ; ce chemin ne s'exécute pas
aujourd'hui.

## Vérifié et trouvé sain, dans le même passage

**L'ordre des paramètres du circuit.** Les contraintes bornent `x[0:reps]`,
ce qui n'est correct que si le circuit ordonne ses paramètres
`[β…, γ…]`. Mesuré : `['β[0]', 'β[1]', 'γ[0]', 'γ[1]']`. L'ordre vient du tri
alphabétique de Qiskit sur les noms — un détail d'implémentation d'une
bibliothèque extérieure. S'il changeait, la borne s'appliquerait à γ et le
mixer tournerait libre. **Épinglé par un test.**

**Le score, et l'équité entre les deux bras.** `weighted_relative_error` vaut
bien 0 sur une reconstruction exacte et 1 quand le bras rend zéro. La carte de
pondération est construite sur la **référence**, donc identique pour les deux
bras. La comptabilité de pixels est symétrique : même `step_layered`, même
`max_depth`, même `target_dim`, accumulation à chaque pas pour les deux.

**La réduction du score, entre les deux bras.** Le bras classique décide sur
`_process_score`, le bras quantique sur le `mini_score` de
`get_adaptive_flux`. Deux chemins, une seule question : donnent-ils le même
nombre ? Mesuré sur un champ aléatoire, `target_dim = 2` :

| profondeur | écart maximal |
|---|---|
| 0 | **0,000e+00** |
| 1 | **0,000e+00** |

Identiques. La comparaison des deux bras porte donc bien sur le critère, pas
sur la réduction.

**`_run_level_classical` contre `_run_level`.** Le bloc de décision — seuil,
TTL, sondage de bord, ventilation en `if/elif/else` — est structurellement
identique, correction D-16 comprise. Deux échappatoires `continue` du chemin
quantique (`prep is None`, `result is None`) sont **inatteignables** :
ni `_prepare_vqa_input` ni `call_vqa_shell` ne rendent `None`. Elles auraient
fait disparaître un patch du pavage, donc laissé une région sans traitement ;
elles ne le font pas.
