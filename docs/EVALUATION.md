# Évaluation

**Ce qui, dans `RESULTS.md`, est exploitable — et ce qui ne l'est pas.**

Un résultat peut être correctement obtenu et rester inutilisable : mesuré sur
du code depuis corrigé, non reproductible d'une exécution à l'autre, ou
dépendant d'un réglage sans provenance. Ce document trie.

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

**Les 37 défauts corrigés, chacun mesuré avant/après.** C'est le matériau le
plus solide du travail. Chaque mesure est déterministe, refaite par une
commande, et verrouillée par un test qui échoue sur l'ancienne version.
→ `RESULTS.md`, `COUVERTURE.md`

**Les faits structurels sur le circuit.** Mesurés, déterministes,
indépendants de tout réglage :

- la couche de coût est **diagonale** — γ seul ne déplace aucune probabilité
  de mesure (4,4e−16) ; seul le mixeur agit, borné à `π/(4·reps) = 0,393` ;
- `g_strain + g_rot ≡ 1` par identité algébrique — ZZ et ZZZZ partitionnent
  un unique scalaire d'Okubo-Weiss, ils ne sont pas deux détecteurs
  indépendants ;
- `PhysicalMapperV2` est **adimensionnel** — dx de 1,0 à 0,001 laisse les
  coefficients bit à bit identiques, ν et η n'y entrent pas.

**Les mesures d'ordre du solveur.** Grille fixe, quatre résolutions
temporelles, chaque schéma contre sa propre référence. Reproductible.

**La méthode d'audit elle-même.** Les cinq questions, les huit patrons, les
proportions — 12 défauts sur 37 par une seule question, deux trouvés
en retirant une couche plutôt qu'en posant une question, et un (D-48) par la
cinquième question seule, celle qui demande si un test **traverse** la
configuration. C'est une contribution à part entière, et elle ne dépend
d'aucune campagne.

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

---

## B — En attente de confirmation

Correctement obtenus, mais **sur du code depuis corrigé**. Ils ne sont pas
invalidés : ils sont à refaire.

**Tous les verdicts d'hypothèse.** H0a, H0b, H1, H2b, H3a, H3b, H4, H5 ont
été établis avant D-1, D-9, D-11, D-13, D-14, D-16, D-21, D-25. Chacune de
ces corrections touche ce que le modèle consomme.

**Les 16 lignes de la table maître en écart.** Ce sont exactement les nombres
déplacés par les corrections. Ils doivent être republiés ou justifiés ligne
par ligne : publier une valeur qu'aucun artefact ne recalcule est ce que ce
dépôt s'interdit.

**Ce qui bloque leur confirmation** : la réoptimisation, elle-même bloquée
par trois décisions → `DEFAUTS.md`.

---

## C — Non concluant

La mesure existe, elle est correcte, et **elle ne tranche pas**.

**Le contraste de décision sur un vortex.** Deux estimations de la même
grandeur, même configuration : +0,0186 ± 0,0067 (16 tirages) et
+0,0053 ± 0,0029 (8 tirages). Un facteur 3,5 entre deux exécutions. L'effet
cherché est du même ordre que la variation d'exécution.

Ce qui **est** concluant sur le même sujet : le coefficient de plaquette,
déterministe à l'écart nul, passe de 0,055 à 1,255 selon la convention de
rotationnel — facteur **22,7**. C'est là-dessus que le test porte désormais.

**Leçon générale** : une grandeur issue d'un tirage stochastique demande
qu'on mesure d'abord la variance de la mesure. Le bras QAOA a une dispersion
par appel de 1,79e−1 à 3,61e−1. Les conclusions fondées sur un **classement**
tiennent (auto-corrélation de rang médiane 0,933) ; celles qui reposeraient
sur une **valeur** ne tiennent pas.

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
