# Évaluation

**Ce qui, dans `RESULTS.md`, est exploitable — et ce qui ne l'est pas.**

Un résultat peut être correctement obtenu et rester inutilisable : mesuré sur
du code depuis corrigé, non reproductible d'une exécution à l'autre, ou
dépendant d'un réglage sans provenance. Ce document trie.

## Les quatre niveaux

| niveau | signification |
|---|---|
| **A — exploitable** | reproductible, mesuré sur le code actuel, verrouillé par un test |
| **B — en attente de confirmation** | correct quand il a été obtenu, mais le code a changé depuis ; à refaire |
| **C — non concluant** | la mesure ne tranche pas — sa variance dépasse l'effet cherché |
| **D — obsolète** | obtenu sur du code dont on sait maintenant qu'il était faux |

---

## A — Exploitable

Ce qui tient aujourd'hui, et sur quoi le papier peut s'appuyer.

**H0a et H0b, à `dim = 3`, sur les deux mappeurs — c'est le résultat le plus
fort du dépôt.** Seule taille certifiée non dégénérée (18 qubits, l'optimum
y est énuméré exactement — `dim = 2` est dégénéré, D-45/D-47, prédicteur
constant optimal sur 40/40 instantanés) :

| | verdict | mesure (V2 / V1) |
|---|---|---|
| H0a | **NON** | QAOA atteint l'optimum sur 0,062–0,156 (V2) / 0/12 (V1) des instantanés, contre 1,000 exigé |
| H0b | **NON** | ρ(E_gap, F1) = +0,870 (V2) / +0,891, p=0,0013 (V1) sur 9 solveurs : mieux résoudre l'hamiltonien **dégrade** la décision AMR |

V2 est le mappeur sans paramètre (poids figés, hors de portée de
`train_hyperparams.py`) ; V1 est celui que la campagne règle et que le
pipeline déploie, mesuré aux hyperparamètres de référence (la campagne n'a
pas encore tourné). Les deux mappeurs donnent le même verdict.

Verrouillé par `pytest tests/study/test_h0_certified_dim3_contradicts_criterion.py`
(7 tests, artefacts V2), reproductible depuis
`results/h0_optimiser_equivalence_N96_dim3.npz` (V2) et
`results/h0_optimiser_equivalence_N96_dim3_..._v1.npz` (V1 — pas encore
verrouillé par un test dédié). Une limite demeure : un seul point du domaine
de recherche à 9 dimensions a été mesuré pour V1, pas un balayage.

**H3 — les couplages ZZ/ZZZZ n'aident jamais, et dégradent la décision dès
qu'ils cessent d'être inertes.** Balayage causal (ablation + F1 contre
vérité terrain) de `dim = 2` à `dim = 8` :

| dim | qubits | méthode | couplages inertes ? | F1 hamiltonien complet | F1 biais Z seul |
|---|---|---|---|---|---|
| 2 | 8 | exhaustive | oui (0 décision changée) | 0,333 | 0,333 |
| 3 | 18 | exhaustive | **non** (6,9–15,3 % changées) | 0,405–0,386 | **0,451** |
| 4 | 32 | glouton, contrôlé contre l'exhaustif | non | 0,520 | **0,552** |
| 8 | 128 | glouton | non | 0,592 | **0,648** |

À `dim = 2` l'inertie est un artefact de taille (l'optimum exact y est
uniforme quel que soit le hamiltonien, D-45/D-47), pas une propriété du
formalisme Ising. Dès que les couplages cessent d'être inertes, ils ne font
jamais mieux que le biais Z seul, et le dépassent en médiocrité de 0,03 à
0,06 de F1 à `dim ≥ 4`. Verrouillé par
`pytest tests/study/test_t13_dim3_couplings_not_inert.py` (7 tests) pour le
point `dim = 3` ; le reste de la table vient de `h3_size_scan.py`
(`RESULTS.md`, T26), avec ses propres contrôles (glouton validé contre
l'exhaustif à `dim = 2`, `--force-greedy`).

**Les défauts corrigés, chacun mesuré avant/après.** Le matériau le plus
solide du travail : chaque mesure est déterministe, refaite par une
commande, verrouillée par un test qui échoue sur l'ancienne version.
→ `RESULTS.md`, `COUVERTURE.md`

**Les faits structurels sur le circuit.** Mesurés, déterministes,
indépendants de tout réglage :

- la couche de coût est **diagonale** — γ seul ne déplace aucune probabilité
  de mesure (4,4e−16) ; seul le mixeur agit, borné à `π/(4·reps) = 0,393` ;
- `g_strain + g_rot ≡ 1` par identité algébrique — ZZ et ZZZZ partitionnent
  un unique scalaire d'Okubo-Weiss, ils ne sont pas deux détecteurs
  indépendants ;
- `PhysicalMapperV2` est **adimensionnel** — dx de 1,0 à 0,001 laisse les
  coefficients bit à bit identiques, ν et η n'y entrent pas ;
- le mixeur seul (balayage exhaustif de β, γ, sans hamiltonien) déplace déjà
  une probabilité médiane de 0,254 sur des patches réels ; l'hamiltonien en
  ajoute 0,236, sur un canal borné à 0,393 rad. Ce n'est **pas** encore un
  solveur comparable aux autres dans le panel H0b (l'y intégrer demande de
  faire tourner QAOA sur un hamiltonien de coût nul, ce que l'optimiseur ne
  peut pas faire faute de signal — `PLAN_PREPRINT.md`, Appendice A, item 4) :
  c'est une borne supérieure mesurée, pas une comparaison faite.

**Les mesures d'ordre du solveur.** Grille fixe, quatre résolutions
temporelles, chaque schéma contre sa propre référence. Reproductible.

**La méthode d'audit elle-même.** Cinq questions, huit patrons de défaut,
plus de 190 défauts de contrat trouvés et fermés — pas une couverture de
ligne, un audit de contrat. C'est une contribution à part entière, et elle
ne dépend d'aucune campagne.

---

## A bis — Ce qui vient d'entrer en A

**La courbe de cône d'information, `dim = 8` et `dim = 16`.** Deux artefacts
(`t1b_cone_curve_N96_dim{8,16}.npz`), déterministes, reproductibles par une
commande, mesurés sur le code actuel, entourés par
`tests/study/test_t1b_cone_curve.py`.

**Exploitable** : les courbes, leurs `n_distinct`, la table de couverture du
carré de Chebyshev par taille (`dim = 16` est la première où les quatre k
sont des voisinages). **Exploitable aussi** : le cône **n'est pas plat** —
écarts par saut à `dim = 16` : +0,123 / −0,076 / +0,100, contre le seuil de
retrait pré-enregistré de 0,01.

**NON exploitable, et cela borne le reste** : la *moyenne* LOSO, dans un sens
comme dans l'autre. `harris_tearing` rend 0,000 à tous les k et aux deux
tailles — pli dégénéré (protocole §1.3 B3), cause non expliquée. Avec ce
pli, le cône reste sous le classique ; sans lui, il le dépasse (0,625 contre
0,444 à `dim = 16`). La conclusion change de signe selon qu'on le compte ou
non. Ce résultat n'est pas touché par la rétractation qui a rouvert T13/T11b
(portée disjointe — `RESULTS.md` le dit explicitement) ; H3 (ci-dessus,
catégorie A) ferme la question qui comptait pour le manuscrit (« les
couplages aident-ils la décision ») indépendamment de ce pli.

**Corpus dans deux conventions de rotationnel.** Huit artefacts `dim = 16` à
N=256/N=64 sont gelés dans l'ancienne convention (`fixed_curl=False`) ; les
artefacts N=96 (ceux que la courbe de cône utilise) sont dans la convention
actuelle (`fixed_curl=True`). `classical_scores` diffère de 100 % des
cellules, jusqu'à 3,7× en relatif, entre les deux conventions — vérifier
avant de mélanger deux artefacts `dim = 16`.

---

## B — En attente de confirmation

Correctement obtenus, mais **sur du code depuis corrigé** — ou sur une
campagne qui n'a pas encore tourné. Pas invalidés : à refaire ou à
compléter.

| hypothèse | où elle vit | pourquoi |
|---|---|---|
| H0a, H0b | **en A** | remesurées à `dim = 3`, verrouillées par un test |
| H2b | réfuté, hors de ce tableau | modèle libre testé, ne bat pas la baseline |
| H3 | **en A** | balayage causal `dim = 2` à `dim = 8`, T26 |
| H1 | reste en B | partiel — les défauts numériques comptent, rien ne dit qu'ils suffisent seuls |
| H4 | reste en B, au sens faible | aucune expérience dédiée ne l'isole ; conjecture. La campagne LOSO qui répondrait n'a que 4 des 8 folds requis (`DEFAUTS.md`, D-197) |
| H5 | mixte, hors de ce tableau | à l'horizon `t_x` : redondant sur harris_tearing/KH (ρ≈1,0), informatif sur mhd_rotor/orszag_tang (ρ jusqu'à 0,66) |

**Les lignes de la table maître qui ne se recalculent plus.** 268 lignes,
**139 OK / 6 DIFF / 123 MISSING** (`aggregate_master_table.py
--allow-missing`) — les MISSING sont les scénarios de la campagne
confirmatoire qui n'a pas encore tourné, pas une régression. Ce compte doit
être recalculé au moment de rédiger, pas recopié d'ici.

**Ce qui bloque leur confirmation** : la campagne d'hyperparamètres, prête
à lancer (mécanisme de diversité d'entraînement et de validation en place,
provenance tracée), qui n'a simplement pas encore tourné → `DEFAUTS.md`,
D-22.

---

## C — Non concluant

La mesure existe, elle est correcte, et **elle ne tranche pas**.

**Le contraste de décision sur un vortex.** Deux estimations de la même
grandeur, même configuration : +0,0186 ± 0,0067 (16 tirages) et
+0,0053 ± 0,0029 (8 tirages) — facteur 3,5 entre deux exécutions, du même
ordre que l'effet cherché. Les deux estimations n'ont pas été retracées à
une exécution précise : avant de les citer, vérifier de quel côté du
correctif de graine QAOA (`RESULTS.md`, D-191) chacune a été produite, et
remesurer si ce n'est pas déterminable.

Ce qui **est** concluant sur le même sujet : le coefficient de plaquette,
déterministe à l'écart nul, passe de 0,055 à 1,255 selon la convention de
rotationnel — facteur **22,7**.

**Leçon générale, vérifiée directement** : une grandeur issue d'un tirage
stochastique demande qu'on mesure d'abord la variance de la mesure. Le bras
QAOA a une dispersion par appel de 1,79e−1 à 3,61e−1. Les conclusions
fondées sur un **classement** tiennent (auto-corrélation de rang médiane
0,933) ; celles qui reposeraient sur une **valeur** ne tiennent pas.
`test_C_ZZ` isole la dispersion pure de l'échantillonnage QAOA sur un
hamiltonien constant et la confirme restaurée sous graine aléatoire (contre
écart-type nul sous l'ancien défaut de graine fixe) ; `test_hyperparameter_sweep`,
rejoué sous une graine indépendante, rend une corrélation identique à la
décimale près à celle mesurée sous l'ancien défaut — pas un artefact de
tirage, un vrai effet (expliqué depuis : une instance de plus de H0a, un
budget d'optimiseur 10× plus grand répare la sélection sans toucher au
Hamiltonien).

---

## D — Obsolète

**Tout nombre Q-HAS obtenu à une profondeur de raffinement supérieure à 1,
avant la correction de `_process_score`** (`refinement.py`). Le biais Z de
l'hamiltonien et ses couplages décrivaient deux grilles différentes à toute
profondeur > 0 (le biais d'un patch venait du quart haut-gauche de ce
patch, écart 41 % du plus grand coefficient). À `max_depth = 4`, réglage de
toutes les campagnes historiques, trois niveaux sur quatre passaient par
là. Le bras classique n'est pas touché (il ne construit aucun
hamiltonien) : la comparaison des deux bras était biaisée dans un sens
connu, avant la correction. Après (`H_edges` passe de (6,6) à (4,4) à
`depth > 0`), plus aucun nombre n'est concerné pour cette raison. Situer un
artefact par rapport au correctif : `git log -S "target_dim + 2 * pad" --
src/Simulation/refinement.py`.

**Tous les nombres publiés dans les documents historiques** —
`docs/archive/`, `v3_master_table_ca7f815.md`, `v3_preprint_description.md`,
`v4_final_results_for_paper.md`, `review_phases_1_to_11c.md`,
`level3_preregistration.md`, `ceiling_proposition.md`, `v1_vs_study.md`.
Ils documentent l'histoire du projet, pas son état ; tout nombre à réutiliser
doit être remesuré par la commande qui le produit. `protocol_v3_evaluation.md`
et `protocol_deviations.md` restent valides — ils décrivent un protocole,
pas des résultats.

---

## Ce qui n'est pas un résultat, et n'entre donc nulle part

**La sélectivité des coefficients** est une propriété de la forme des
coefficients, mesurée sur des champs analytiques à réponse connue. Elle ne
dit rien sur la qualité de la décision d'AMR.

**Le terme ZZZZ à moitié mort, historique.** La mesure « la vorticité pèse
0,003 sur `harris_tearing`, le courant 0,007 sur `kelvin_helmholtz` » décrit
un défaut de l'instrument (sous l'ancienne normalisation de plaquette),
corrigé depuis (`h2b_analytical_solution.py`, `c_bias_grid` élargi,
`require_interior_optima` qui distingue un bord non résolu d'un plateau
biais-seul authentique). Sur `harris_tearing` Re400 N96 dim4 sous le
mécanisme corrigé : F1 sature à 0,7405, sous la baseline classique (0,830).
Les 52 configurations de D-86 n'ont pas toutes été rejouées sous cette
version — le mécanisme qui les rendrait lisibles existe, la campagne pour
les produire non.

**La vérité terrain dynamique.** `d_patches_*` (8 scénarios × 2 horizons,
N=96, dim=8) est une mesure sur les champs réels, pas sur l'objet : c'est
une mesure sur le protocole lui-même. Résumé : voir §H5 en catégorie A/B —
à l'horizon `t_x`, informative sur la moitié du panel canonique.

---

## Ce qu'il faut vérifier avant de faire entrer un résultat en A

1. La commande qui le produit tourne aujourd'hui et rend la même valeur.
2. Un test l'entoure, et ce test **peut** échouer.
3. La grandeur est reproductible — si elle est stochastique, la variance de
   la mesure a été mesurée et elle est plus petite que l'effet.
4. Il ne dépend d'aucun réglage sans provenance.
5. L'opérateur de mesure est **assorti** à celui qui a produit la grandeur.
6. **Le test qui l'entoure emprunte-t-il le chemin réel ?** Une
   configuration rapide peut éviter le chemin qu'elle prétend tester.
7. **La vérification porte-t-elle sur le bon objet ?** Une comparaison
   bit-à-bit correcte peut porter sur un sous-ensemble de clés qui n'est
   pas ce que le consommateur réel agrège.
8. **Le test peut-il échouer aujourd'hui, ou seulement épingler un défaut
   déjà connu ?** Un test qui n'asserte que la persistance d'un bug déjà
   identifié est vert par construction et ne peut signaler aucune
   régression une fois le bug corrigé.

Le point 5 a coûté cinq erreurs dans ce dépôt, dont une où un défaut de huit
ordres de grandeur restait invisible, et une où une correction *correcte*
paraissait fausse.

---

## Où ça mène

**H0a, H0b et H3 ferment l'approche par deux chemins indépendants qui
s'accordent** — optimisation (H0b : même un optimiseur parfait donnerait
une mauvaise décision) et représentation (H3 : les couplages n'apportent
jamais un gain, et nuisent dès qu'ils comptent). La liste A/A bis
ci-dessus est ce sur quoi le manuscrit peut déjà s'appuyer sans attendre.
Il ne manque que la campagne d'hyperparamètres pour redire ces mêmes
verdicts avec des poids réellement entraînés plutôt qu'avec le point de
départ → `PLAN_PREPRINT.md`, Appendice A.
