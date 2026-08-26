# Évaluation

**Ce qui, dans `RESULTS.md`, est exploitable — et ce qui ne l'est pas.**

Un résultat peut être correctement obtenu et rester inutilisable : mesuré sur
du code depuis corrigé, non reproductible d'une exécution à l'autre, ou
dépendant d'un réglage sans provenance. Ce document trie.

---

## Reconstruction du 26 août — pourquoi ce fichier a une nouvelle histoire

Le commit `d3d7573` (24 août) a remplacé ce fichier — alors **278 lignes**,
les quatre niveaux A/B/C/D remplis avec des résultats nominatifs — par un
gabarit générique de 32 lignes (« Admissibilité des résultats », des
catégories sans contenu) : 246 lignes supprimées, 32 nouvelles, aucune trace
de ce qui avait été classé où. Même geste que pour `DEFAUTS.md`,
`COUVERTURE.md`, `RESULTS.md` et `PLAN_PREPRINT.md` — sauf que ces deux
derniers n'avaient, jusqu'au 26 août, jamais été restaurés : l'écart a été
relevé par USER (« l'agent a l'air complètement perdu »), pas trouvé en
interne, après que `DEFAUTS.md`/`COUVERTURE.md`/`RESULTS.md` eurent déjà
reçu une reconstruction complète le 25 août.

**Ce que ça veut dire concrètement** : entre le 25 et le 26 août, ce dépôt
avait un `DEFAUTS.md` dense et daté (321 lignes) à côté d'un `EVALUATION.md`
de 32 lignes sans un seul résultat nommé — alors que c'est précisément ce
second fichier qui est censé répondre à « qu'est-ce qui, dans tout ce
travail, peut entrer dans le papier ? ». Sans lui, cette question n'avait
pas de réponse écrite : il fallait la refaire de tête à chaque fois.

**Méthode** : texte original (`git show d047015:docs/EVALUATION.md`)
restauré niveau par niveau, chaque entrée revérifiée contre `RESULTS.md` et
`DEFAUTS.md` **actuels** — pas seulement contre leur état à `d047015`.
Marqué **« mis à jour le 26 août »** ce qui a bougé depuis.

---

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

**H0a et H0b, à `dim = 3` (D-53). Ajouté le 26 août — c'est la promotion la
plus importante de cette reconstruction.** À `d047015`, ces deux verdicts
vivaient en catégorie B (« établis avant D-1…D-25, à refaire »). Ils ont
depuis été refaits, à la seule taille certifiée non dégénérée (`dim = 3`,
18 qubits, l'optimum y est énuméré exactement — `dim = 2` est dégénéré,
D-45/D-47, prédicteur constant optimal sur 40/40 instantanés) :

| | verdict | mesure |
|---|---|---|
| H0a | **NON** | QAOA atteint l'optimum sur 0,062–0,156 des instantanés, contre 1,000 exigé et 0,500 pour la règle classique dont il part |
| H0b | **NON** | ρ(E_gap, F1) = +0,870 sur 9 solveurs : mieux résoudre l'hamiltonien **dégrade** la décision AMR |

Verrouillé par `pytest tests/study/test_h0_certified_dim3_contradicts_criterion.py`
(7 tests), reproductible depuis `results/h0_optimiser_equivalence_N96_dim3.npz`
et sa variante `--scale-kopt`. C'est le résultat le plus fort du dépôt : il
n'était écrit dans aucun document vivant avant le 25 août (D-193).

**Les défauts corrigés, chacun mesuré avant/après.** C'est le matériau le
plus solide du travail. Chaque mesure est déterministe, refaite par une
commande, et verrouillée par un test qui échoue sur l'ancienne version.
→ `RESULTS.md`, `COUVERTURE.md`

**Mis à jour le 26 août — le compte de « 37 » ne tient plus, et ne doit pas
être recopié tel quel.** `RESULTS.md` porte aujourd'hui 32 sections `# D-NNN`
et 159 lignes de table `| D-NNN`, sur un historique qui inclut au moins 6
défauts supplémentaires refermés le seul 25 août (D-39, D-50, D-98, D-100,
D-158, D-191) plus un nouveau laissé ouvert (D-195). Le total exact —
distinct du compte de lignes, puisque certaines entrées partagent un numéro
ou se recoupent (ex-D-132, D-192 à 37 sites) — n'a pas été recalculé ici. Ne
pas citer « 37 » dans le manuscrit ; recompter au moment de rédiger.

**Les faits structurels sur le circuit.** Mesurés, déterministes,
indépendants de tout réglage :

- la couche de coût est **diagonale** — γ seul ne déplace aucune probabilité
  de mesure (4,4e−16) ; seul le mixeur agit, borné à `π/(4·reps) = 0,393` ;
- `g_strain + g_rot ≡ 1` par identité algébrique — ZZ et ZZZZ partitionnent
  un unique scalaire d'Okubo-Weiss, ils ne sont pas deux détecteurs
  indépendants ;
- `PhysicalMapperV2` est **adimensionnel** — dx de 1,0 à 0,001 laisse les
  coefficients bit à bit identiques, ν et η n'y entrent pas.

**Ajouté le 26 août, appartient à cette liste — le témoin qui manque
encore.** Le mixeur seul, sans hamiltonien, déplace déjà une probabilité
médiane de 0,254 sur des patches réels (Orszag-Tang N=256, balayage complet
de γ et β) ; l'hamiltonien en ajoute 0,236, sur un canal borné à 0,393 rad.
Mesuré, déterministe (balayage exhaustif, pas un tirage), verrouillé par le
test de contrat d'ordre des paramètres `QAOAAnsatz`. **Non exploitable
directement pour trancher H0b** : aucune campagne du dépôt ne l'utilise
encore comme témoin (`PLAN_PREPRINT.md`, Appendice A, item 4) — c'est une
borne supérieure mesurée, pas une comparaison faite.

**Les mesures d'ordre du solveur.** Grille fixe, quatre résolutions
temporelles, chaque schéma contre sa propre référence. Reproductible.

**La méthode d'audit elle-même.** Les cinq questions, les huit patrons, les
proportions — 12 défauts sur 37 par une seule question, deux trouvés
en retirant une couche plutôt qu'en posant une question, et un (D-48) par la
cinquième question seule, celle qui demande si un test **traverse** la
configuration. C'est une contribution à part entière, et elle ne dépend
d'aucune campagne.

**Mis à jour le 26 août.** La méthode continue de rapporter : les six
défauts refermés le 25 août (D-39, D-50, D-98, D-100, D-158, D-191) et le
nouveau trouvé en vérifiant l'un d'eux (D-195) suivent le même schéma —
audit de contrat, pas de couverture de ligne. Le ratio « 12 sur 37 » cité
ci-dessus est à recalculer avec le compte total (voir plus haut) avant
d'être republié.

---

## A bis — Ce qui vient d'entrer en A

**La courbe de cône d'information, `dim = 8` et `dim = 16`.** Deux artefacts
(`t1b_cone_curve_N96_dim{8,16}.npz`, git `ef5f0a4`), déterministes.
Reproductibles par une commande, mesurés sur le code actuel, entourés par
`tests/study/test_t1b_cone_curve.py`.

**Exploitable** : les courbes, leurs `n_distinct`, et la table de couverture
du carré de Chebyshev par taille — `dim = 16` est la première où les quatre
k sont des voisinages (k=3 y couvre 19 % de la grille, contre 77 % à
`dim = 8` et 100 % à `dim = 4`).

**Exploitable aussi, et c'est le fait qui compte** : le cône **n'est pas
plat**. Écarts par saut à `dim = 16` : +0,123 / −0,076 / +0,100, contre le
seuil de retrait pré-enregistré de 0,01.

**NON exploitable, et cela borne tout le reste** : la *moyenne* LOSO, dans un
sens comme dans l'autre. `harris_tearing` rend 0,000 à tous les k et aux deux
tailles — pli dégénéré au sens du protocole §1.3 B3. Avec ce pli, le cône
reste sous le classique ; sans lui, il le dépasse (0,625 contre 0,444 à
`dim = 16`). **La conclusion change de signe selon qu'on le compte ou non**,
et rien ne tranche tant que le pli n'est pas expliqué : le classique y rend
0,976 pendant que le GBT s'effondre.

**Rétractation.** Une première lecture, publiée puis retirée le 21 août,
annonçait « gain faible, aucun sous transfert ». Elle venait de deux
paramètres non justifiés — 8 instantanés sur 20 disponibles, et `dim = 8` où
k=3 n'est plus un voisinage. → `RESULTS.md`.

**Mis à jour le 26 août — deux choses de plus, aucune ne fait bouger le
verdict ci-dessus, toutes deux à savoir avant de citer ces artefacts.**

1. **Ce résultat n'est PAS touché par la rétractation D-58** (ci-dessous,
   catégorie B) : D-58 porte sur T17/T18 — la fenêtre d'incertitude de la
   campagne confirmatoire et son effet sur le couplage ZZ — pas sur T1b, la
   courbe de cône. `RESULTS.md` le dit explicitement : « ce n'est pas la
   courbe de cône qui est en cause ». Les deux se ressemblent (tous deux
   « un signal ZZ semblait mort ») sans être le même défaut ; ne pas les
   fusionner dans le manuscrit.
2. **Le corpus de patches vit dans deux conventions de rotationnel.** Huit
   artefacts `dim = 16` à N = 256/N = 64 sont **délibérément gelés** dans
   l'ancienne convention (`fixed_curl=False`) ; les artefacts N = 96 (ceux
   que la courbe de cône utilise, `dim = 8` et `dim = 16`) sont tous les
   deux dans la convention actuelle (`fixed_curl=True`) et se comparent
   entre eux sans problème. Mais **tout autre nombre** tiré d'un artefact
   `dim = 16` doit d'abord vérifier de quel côté du gel il se trouve —
   `classical_scores` diffère de 100 % des cellules, jusqu'à 3,7× en
   relatif, entre les deux conventions ; `l2_errors` et `is_hard`
   (le label) ne bougent pas.

---

## B — En attente de confirmation

Correctement obtenus, mais **sur du code depuis corrigé**. Ils ne sont pas
invalidés : ils sont à refaire.

**Tous les verdicts d'hypothèse. H0a, H0b, H1, H2b, H3a, H3b, H4, H5 ont
été établis avant D-1, D-9, D-11, D-13, D-14, D-16, D-21, D-25.** Chacune de
ces corrections touche ce que le modèle consomme.

**Mis à jour le 26 août — cette entrée ne décrit plus l'état du dépôt telle
quelle ; elle est remplacée par la liste ci-dessous, hypothèse par
hypothèse, plutôt que reconduite en bloc.**

| hypothèse | où elle vit maintenant | pourquoi |
|---|---|---|
| H0a, H0b | **promues en A** ci-dessus | remesurées à `dim = 3` (D-53), verrouillées par un test, après D-1…D-25 |
| H2b | **réfuté, hors de ce tableau** | modèle libre testé (`study/h2b_prediction/`), ne bat pas la baseline — verdict stable, à confirmer sur le code post-25-août mais sans signe qu'il ait bougé |
| H3a, H3b | **rétrogradées en dessous de B, voir note** | D-58 (déjà connu à `d047015`, jamais reporté ici) retire l'explication causale de T13/T11b ; ces deux résultats restent à expliquer, pas seulement à reconfirmer. La courbe de cône (T1b) reste en A bis, non touchée |
| H1 | reste en B | partiel — les défauts numériques comptent, mais rien ne dit qu'ils suffisent seuls |
| H4 | reste en B, au sens faible | aucune expérience dédiée ne l'isole ; conjecture, pas mesure en attente d'un refresh |
| H5 | **mixte, remesuré à `t_x`, hors de ce tableau** | D-188 (26 août) — redondant sur harris_tearing/KH (ρ≈1,0), informatif sur mhd_rotor/orszag_tang (ρ jusqu'à 0,66) à l'horizon physique |

**Note sur H3 — pourquoi « à reprendre » n'est pas la même chose que « en
attente ».** Les autres lignes de B attendent une réexécution sur du code
corrigé : la question posée reste la même, seule la réponse doit être
rafraîchie. H3a/H3b sont dans une situation différente et plus grave :
l'**explication** qui accompagnait la lecture publiée de T13 (ablations
nulles) et T11b (progression variationnelle quasi nulle) s'est effondrée
avec D-58, sans explication de repli. Ces deux résultats ne sont pas « à
refaire » — ils sont « à comprendre à nouveau ».

**Les lignes de la table maître qui ne se recalculent plus.** Ce sont
exactement les nombres déplacés par les corrections. Ils doivent être
republiés ou justifiés ligne par ligne : publier une valeur qu'aucun
artefact ne recalcule est ce que ce dépôt s'interdit.

**Mis à jour le 26 août — le compte a bougé plusieurs fois, voir
`PLAN_PREPRINT.md` Appendice A item 3 pour l'historique complet.** « 16
lignes » date de `d047015`. Repris identique des dizaines de fois dans
`RESULTS.md` après la clôture de D-58 : **180 lignes, 176 OK / 4 DIFF /
0 MISSING** (périmètre à 4 scénarios). Le 25 août, après la correction de
D-158 et sur le périmètre élargi à 8 scénarios : **268 lignes, 142 OK /
6 DIFF / 120 MISSING** (`--allow-missing`). N'utiliser aucun de ces trois
comptes sans le recalculer d'abord — chacun décrit un jour et un périmètre
différents.

**Ce qui bloque leur confirmation** : la réoptimisation, elle-même bloquée
par une décision → `DEFAUTS.md`, D-22.

**Mis à jour le 26 août.** D-22 reste bloquant : re-vérifié le 25 août, le
JSON déployé ne correspond toujours à aucune ligne des CSV Optuna du dépôt.
Le mécanisme de provenance existe désormais (`_save_results` écrit le jeu
complet de paramètres, le hash du commit, `sys.argv`) mais ne s'applique
qu'à une campagne qui n'a pas encore tourné.

---

## C — Non concluant

La mesure existe, elle est correcte, et **elle ne tranche pas**.

**Le contraste de décision sur un vortex.** Deux estimations de la même
grandeur, même configuration : +0,0186 ± 0,0067 (16 tirages) et
+0,0053 ± 0,0029 (8 tirages). Un facteur 3,5 entre deux exécutions. L'effet
cherché est du même ordre que la variation d'exécution.

**Mis à jour le 26 août — ces deux nombres eux-mêmes doivent être
suspectés, pas seulement le contraste qu'ils mesurent.** D-191 (25 août) a
trouvé que `VQARuntime`/`execute()` fixaient par défaut `seed=0` **partout**
hors du protocole confirmatoire — masquant, dans certains chemins, la
dispersion réelle du bras QAOA jusqu'à zéro exactement. Le test qui vit sur
cette exacte question
(`test_qaoa_physics_decision.py::TestFullPipelineVortex::
test_the_vortex_contrast_is_not_reproducible_enough_to_conclude`) faisait
partie des cinq tests cassés par ce défaut, et **passe de nouveau** depuis
la correction — dispersion restaurée. Les deux estimations « +0,0186 » et
« +0,0053 » citées ci-dessus n'ont pas été retracées à une exécution
antérieure ou postérieure à D-191 : avant de les citer dans un manuscrit,
vérifier de quel côté du correctif chacune a été produite, et remesurer si
ce n'est pas déterminable.

Ce qui **est** concluant sur le même sujet : le coefficient de plaquette,
déterministe à l'écart nul, passe de 0,055 à 1,255 selon la convention de
rotationnel — facteur **22,7**. C'est là-dessus que le test porte désormais.

**Leçon générale** : une grandeur issue d'un tirage stochastique demande
qu'on mesure d'abord la variance de la mesure. Le bras QAOA a une dispersion
par appel de 1,79e−1 à 3,61e−1. Les conclusions fondées sur un **classement**
tiennent (auto-corrélation de rang médiane 0,933) ; celles qui reposeraient
sur une **valeur** ne tiennent pas.

**Mis à jour le 26 août — cette leçon est maintenant mieux vérifiée qu'à
`d047015`, et elle a produit un défaut nouveau plutôt qu'une confirmation
simple.** `test_C_ZZ` (`tests/mapping/test_signal_contribution.py`) isole
la dispersion pure de l'échantillonnage QAOA sur un hamiltonien constant
(aucune source de bruit hors QAOA) : sous l'ancien `seed=0`, écart-type
0,0 exactement ; sous une graine confirmée aléatoire (D-191), la dispersion
est bien restaurée. Ce contrôle a servi à trancher un cas ambigu :
`test_hyperparameter_sweep`, rejoué sous une graine indépendante confirmée
aléatoire, rend une corrélation de rang **identique à la décimale près** à
celle mesurée sous l'ancien `seed=0` — donc **pas** un artefact de tirage,
un vrai effet, stable, non expliqué (D-195, `DEFAUTS.md`). La leçon
« mesurer la variance avant de conclure » a donc, une fois appliquée pour de
bon, produit un nouveau résultat non concluant plutôt qu'd'effacer les
anciens.

---

## D — Obsolète

**Tout nombre Q-HAS obtenu à une profondeur de raffinement supérieure à 1**
(D-37). Le biais Z et les couplages de l'Hamiltonien décrivaient deux grilles
différentes à toute profondeur > 0 : le biais d'un patch venait du quart
haut-gauche de ce patch. Écart mesuré 41 % du plus grand coefficient. Présent
depuis le premier commit du fichier.

Portée : `depth = 0` est épargné — il est périodique et n'a pas de halo. À
`max_depth = 4`, réglage de toutes les campagnes, **trois niveaux sur quatre**
passaient par là. Cela ne se répare pas en reclassant : il faut refaire les
mesures.

Ce qui **n'est pas** touché : le bras classique, qui ne construit aucun
Hamiltonien. La comparaison des deux bras est donc biaisée dans un sens
connu — le bras quantique décidait sur un biais Z lu au mauvais endroit.

**Mis à jour le 26 août — D-37 est corrigé ; cette entrée décrit maintenant
une fenêtre de temps, pas un état permanent du code.** `RESULTS.md`
documente la correction (une ligne, `_process_score(local_score,
depth == 0, target_dim)` ; `H_edges` passe de (6,6) à (4,4) à `depth > 0` ;
le pipeline s'exécute désormais à `max_depth = 2` là où il levait une
`ValueError` avant). **Tout nombre produit avant ce commit, à profondeur
> 1, reste obsolète au sens de ce niveau — tout nombre produit après ne
l'est plus pour cette raison.** Le déterminer demande de situer l'artefact
cité par rapport à `git log -S "target_dim + 2 * pad" --
src/Simulation/refinement.py` — non fait ici pour les artefacts existants,
à faire avant de promouvoir l'un d'eux hors de D.

**Tous les nombres publiés dans les documents antérieurs à cet audit.** Ils
ont été obtenus sur du code dont on sait maintenant qu'il calculait autre
chose que ce qu'il annonçait, et dont le code d'étude n'était pas testé.

Concernés : `docs/archive/`, et les documents de campagne conservés pour
mémoire — `v3_master_table_ca7f815.md`, `v3_preprint_description.md`,
`v4_final_results_for_paper.md`, `review_phases_1_to_11c.md`,
`level3_preregistration.md`, `ceiling_proposition.md`, `v1_vs_study.md`.

**Ne pas les citer.** Ils documentent l'histoire du projet, pas son état.
Tout nombre qu'on veut réutiliser doit être **remesuré** par la commande qui
le produit.

Cas particulier : `protocol_v3_evaluation.md` et `protocol_deviations.md`
restent valides — ils décrivent un **protocole**, pas des résultats.

**Ajouté le 26 août.** `docs/DEFAUTS.md`, `docs/COUVERTURE.md`,
`docs/RESULTS.md`, ce fichier et `docs/PLAN_PREPRINT.md` ont chacun un
« avant » (avant le commit `d3d7573` du 24 août, qui les a vidés sans rien
archiver) et un « après » (restauré, vérifié contre le code actuel). Le
même principe qui interdit de citer `docs/archive/` interdit de citer une
version d'un de ces six documents antérieure à sa propre reconstruction —
même règle, appliquée aux documents qui portent les résultats plutôt
qu'aux résultats eux-mêmes.

---

## Ce qu'il faut vérifier avant de faire entrer un résultat en A

1. La commande qui le produit tourne aujourd'hui et rend la même valeur.
2. Un test l'entoure, et ce test **peut** échouer.
3. La grandeur est reproductible — si elle est stochastique, la variance de
   la mesure a été mesurée et elle est plus petite que l'effet.
4. Il ne dépend d'aucun réglage sans provenance.
5. L'opérateur de mesure est **assorti** à celui qui a produit la grandeur.

Le point 5 a coûté cinq erreurs dans ce dépôt, dont une où un défaut de huit
ordres de grandeur restait invisible, et une où une correction *correcte*
paraissait fausse.

6. **Le test qui l'entoure emprunte-t-il le chemin réel ?** D-37 a survécu à
   toute la suite parce que les configurations rapides utilisent
   `max_depth = 1`, profondeur à laquelle le chemin borné n'est jamais
   emprunté. Un test qui ne descend pas là où le code vit ne le teste pas.

7. **La vérification porte-t-elle sur le bon objet ?** Une vérification
   juste, correctement exécutée, peut mesurer autre chose que ce qu'on
   croit. La plaquette scindée a été annoncée « sans changement de
   comportement » sur la foi d'une comparaison bit à bit des **valeurs**
   des clés partagées — vraie, reproductible, et sans rapport avec la
   question, puisque `call_vqa_shell.py` agrège sur l'**ensemble** des clés
   et voyait donc son `E_max` bouger de +15,9 % à +33,6 %. Avant de conclure
   « rien ne bouge », demander : *qu'est-ce qui, chez le consommateur réel,
   pourrait bouger sans que ma mesure le voie ?*

**Ajouté le 26 août, une huitième question — trouvée en vérifiant ce
document lui-même :**

8. **Le test peut-il échouer aujourd'hui, ou seulement épingler un défaut
   déjà connu ?** Cinq tests de ce dépôt (D-98, D-100, D-194, D-50, D-39)
   ont été trouvés le 25 août à n'asserter que la persistance d'un bug
   déjà identifié — donc verts par construction, incapables de signaler une
   régression une fois le bug corrigé. Chacun a été réécrit en test de
   correction (il vérifie maintenant ce que le code doit faire, pas ce
   qu'il ne doit plus faire) au moment même où le défaut sous-jacent a été
   corrigé. Un résultat entouré d'un test de ce type ne satisfait pas le
   point 2 ci-dessus, même s'il a l'air de le faire.

---

## Ce qui n'est pas un résultat, et n'entre donc nulle part

**La sélectivité des coefficients** (RESULTS, 21 août) est une propriété de la
**forme** des coefficients, mesurée sur des champs analytiques à réponse
connue. Elle ne dit rien sur la qualité de la décision d'AMR : ce n'est pas un
résultat au sens de ce document, et elle n'a pas de niveau A/B/C/D.

**Cas à part — le terme ZZZZ à moitié mort.** La mesure « la vorticité pèse
0,003 sur `harris_tearing`, le courant 0,007 sur `kelvin_helmholtz` » est,
elle, une mesure sur les **champs réels du corpus**, reproductible par une
commande, et elle ne dépend d'aucun modèle entraîné. Elle décrit néanmoins un
**défaut de l'instrument**, pas un résultat sur l'objet : elle dit ce que
l'hamiltonien ne mesurait pas, pas ce que le QAOA sait faire.

Sa conséquence est en revanche directe et lourde : **toute campagne relancée
tourne sur un hamiltonien différent** de celui qui a produit les artefacts
gelés. Les nombres publiés d'avant ne sont pas invalidés — ils décrivent
fidèlement le Hamiltonien d'alors — mais ils ne sont plus **comparables** à ce
qui sortira ensuite. Toute comparaison avant/après doit passer par
`norm="legacy"`, seul mode qui reproduit le passé.

**Un instrument qui avait perdu sa résolution.** Le fait le plus lourd du
21 août n'est pas la formule : c'est que sous `legacy`, sur la configuration
`harris_tearing` N=96 dim=4, le balayage `c_bias` rendait **F1 = 0 sur les 25
points de grille**, aux quatre Re. Le hamiltonien de champ moyen n'y séparait
rien. Sous `max`, `f1_span` vaut 0,55 à 0,57 et aucun balayage n'est dégénéré.

Conséquence pour ce document : **les 14 balayages plats que D-86 recense
mesuraient peut-être l'instrument, pas l'objet.** Aucun résultat n'est
rétracté sur cette base — un seul cas a été rejoué — mais tout niveau qui
s'appuie sur un balayage `c_bias` est désormais **suspendu** en attendant que
les 52 configurations soient rejouées.

**Et ce que le balayage corrigé montre, il faut aussi le dire :** F1 sature à
**0,6333** dans la limite biais seul, contre **0,745** pour la baseline
classique sur la même configuration. L'instrument retrouve sa résolution, et
ce qu'il résout ne va pas dans le sens de Q-HAS. Une configuration ne fait pas
un verdict — mais elle interdit de présenter le basculement comme une bonne
nouvelle pour l'hypothèse.

**Mis à jour le 26 août — un second cas a depuis été rejoué (D-186,
`DEFAUTS.md`), avec un mécanisme plus soigné et des nombres différents.**
`h2b_analytical_solution.py` porte désormais `c_bias_grid` par défaut sur
`[0,1 ; 1e5]` (contre `[0,1 ; 100]` à la découverte, la grille était trop
courte) et une fonction `require_interior_optima` qui **distingue** un bord
non résolu (lève `RuntimeError`, refuse de produire un artefact) d'un
plateau **biais seul authentique** (`bias_only_limit=True`,
`c_bias_identifiable=False`, exempté du refus — l'optimum est légitimement
au bord parce que le biais seul n'identifie rien au-delà). Remesuré le
25 août sur `harris_tearing` Re400 N96 dim4 : `at_right_edge=True`,
`bias_only_limit=True`, **F1 sature à 0,7405**, toujours **sous** la
baseline classique (0,830) — un nombre différent du 0,6333/0,745 ci-dessus
(grille et mécanisme de détection différents), mais la même conclusion :
l'instrument retrouve sa résolution, et Q-HAS n'en bénéficie pas. **Les 52
configurations de D-86 n'ont toujours pas toutes été rejouées sous cette
version** — le mécanisme qui les rendrait lisibles existe maintenant, la
campagne pour les produire non.

**La vérité terrain dynamique, et ce qu'elle vaut.** `d_patches_*` existe
désormais (4 scénarios × 2 horizons, N=96, dim=8). Le résultat est une mesure
sur les champs réels, reproductible par une commande, et il est **négatif au
sens utile** : à l'horizon du protocole, ρ(d, e) ≥ 0,98 — le label dynamique
ne répare pas H5. Cela ne se range dans aucun niveau A/B/C/D parce que ce
n'est pas un résultat sur l'objet : c'est une mesure **sur le protocole
lui-même**, qui invalide son critère d'acceptation (« Spearman > 0 ») et son
horizon (δt = 0,1).

Conséquence pratique : toute tâche du protocole qui devait consommer `d_i`
comme label alternatif doit d'abord fixer l'horizon sur `t_x`, le temps de
traversée d'un patch. Sans quoi elle mesurerait deux fois la même chose.

**Mis à jour le 26 août — tranché et remesuré, plus une décision en
attente.** USER a confirmé le principe (fixer `t_x`) et demandé la
remesure. Régénéré aux 4 scénarios canoniques à `t_x` (`docs/RESULTS.md`,
D-188) : `ρ(d, e)` reste ≥ 0,97 pour harris_tearing/kelvin_helmholtz
(toujours redondant), mais tombe à 0,66–0,92 pour orszag_tang et
0,82–0,99 pour mhd_rotor sur certains instantanés — **sous le seuil de
redondance du module (0,95) pour la moitié du panel**. Le label dynamique
ne répare donc pas H5 partout, mais il n'échoue plus partout non plus :
c'est une mesure plus fine que « ρ ≥ 0,98 uniforme », pas la même
conclusion redite. `--allow-redundant` a permis d'obtenir la mesure même
pour les scénarios redondants, plutôt que le refus silencieux que le
garde-fou du script aurait produit par défaut.

Ce que tout ceci autorise : poser proprement la question « un hamiltonien dont
les deux structures pèsent également décide-t-il mieux ? ». **Cette question
est ouverte, et seule une campagne peut y répondre** — les mesures d'avant
campagne ne le peuvent pas, par construction.

**Mis à jour le 26 août, pour clore.** Cette phrase de fermeture, écrite à
`d047015`, reste la bonne réponse à la question que USER a posée le 26 août
(« la campagne peut être lancée, les résultats interprétés, je peux écrire
le papier ? ») : **non**, pour trois raisons indépendantes, pas une seule —
(1) D-22 bloque toujours le lancement de la campagne elle-même ; (2) H3
(T13/T11b) est à reprendre, pas seulement à reconfirmer, depuis D-58 ; (3)
D-195, trouvé le 25 août en vérifiant D-191, est un effet réel et non
expliqué qui n'a pas encore de niveau A/B/C/D parce qu'il n'a pas encore
d'explication à évaluer. Ce que ce document permet, en revanche, c'est de
savoir exactement *quoi* faire tourner et *quoi* laisser de côté quand la
campagne partira : la liste A/A bis ci-dessus est ce sur quoi le papier peut
déjà s'appuyer sans attendre.
