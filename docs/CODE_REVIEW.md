# Note de revue

Note personnelle sur la façon de relire ce dépôt. Pas un protocole imposé —
ce qui a marché, et ce qui m'a coûté du temps.

---

## Ce que je cherche

Un calcul qui **rend une valeur plausible mais fausse**. C'est la seule
classe qui compte : un plantage se voit, un NaN se voit. Ce qui ne se voit
pas, c'est un tableau de la bonne forme, aux valeurs finies, dans le bon
intervalle — et faux.

Les 27 défauts de ce dépôt appartiennent tous à cette classe.

## Les quatre questions, par rentabilité mesurée

1. **Pourquoi cette fonction existe-t-elle ?** Que se passerait-il si on la
   supprimait ?
2. **Que promet-elle ?** Lire la docstring comme un contrat.
3. **Consomme-t-elle ce que sa signature annonce ?**
4. **Deux chemins censés coïncider coïncident-ils encore ?**
   → **12 défauts sur 27.** Commencer par là.

## Ce qui m'a pris du temps

**L'opérateur non assorti — quatre fois.** Mesurer une grandeur avec un
stencil différent de celui qui l'a produite ne mesure pas la grandeur, mais
l'écart entre deux opérateurs. Une fois, un défaut de huit ordres restait
invisible au spectral et sautait aux yeux en FD4.

**Annoncer avant d'avoir lancé la suite complète.** Deux fois. Une correction
juste sur `step_full` cassait l'AMR : huit tests, dont six préexistants.

**Calibrer un seuil sur la mesure du jour.** Mes tests de vortex étaient
périmés trois tours plus tard. La grandeur elle-même n'était pas
reproductible ; aucun seuil n'était le bon.

**Corriger un gel documenté.** Une décision antérieure, écrite dans le
fichier, que je n'avais pas lue. C'est un test qui me l'a rappelée.

## Ce qui marche

- **Mesurer avant d'affirmer, y compris contre soi.** Le splitting de Strang
  et la projection du second membre étaient tous deux plausibles et tous deux
  faux, chacun démonté par une mesure.
- **Choisir le champ qui sépare.** Sur Taylor-Green, deux conventions
  opposées rendent la même enstrophie. Avant d'écrire un test : *sur quelle
  entrée les deux hypothèses divergent-elles ?*
- **Un test qui épingle l'ancien comportement.** Sans lui, une correction se
  défait en silence.
- **Retirer une couche peut révéler ce qu'elle cachait.** D-26 et D-27 n'ont
  été vus qu'en cessant de projeter B. Aucune des quatre questions ne les
  aurait trouvés.

## L'ordre dans lequel je relis

Un module, les quatre questions, **jusqu'au bout**. Mieux vaut un module
épuisé que dix survolés. Quand il est fini, écrire ce qui a été vérifié et
trouvé sain — c'est un résultat, et cela évite de le relire deux fois.

## Le réglage à ne pas rater

**La majorité du code est juste** : 164 des 180 nombres publiés inchangés
après 27 défauts trouvés. Un relecteur qui rapporte un défaut par fonction se
trompe, et un faux positif coûte plus cher qu'un défaut manqué — il envoie
corriger du code correct.

## Où j'écris quoi

| document | ce que j'y mets |
|---|---|
| `PLAN_PREPRINT.md` | la structure et les hypothèses — **ni défaut ni mesure** |
| `DEFAUTS.md` | ce qui **bloque**, uniquement |
| `RESULTS.md` | ce qui est **accompli**, avec la commande pour le refaire |
| `COUVERTURE.md` | ce qui est **testé**, et ce qui ne l'est pas |
| `EVALUATION.md` | ce qui, dans `RESULTS`, est **exploitable** |
| `CODE_REVIEW.md` | ce fichier |
