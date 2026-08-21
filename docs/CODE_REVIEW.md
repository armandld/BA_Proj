# Note de revue

Note personnelle sur la façon de relire ce dépôt. Pas un protocole imposé —
ce qui a marché, et ce qui m'a coûté du temps.

---

## Ce que je cherche

Un calcul qui **rend une valeur plausible mais fausse**. C'est la seule
classe qui compte : un plantage se voit, un NaN se voit. Ce qui ne se voit
pas, c'est un tableau de la bonne forme, aux valeurs finies, dans le bon
intervalle — et faux.

Les 37 défauts de ce dépôt appartiennent tous à cette classe.

## Les cinq questions, par rentabilité mesurée

1. **Pourquoi cette fonction existe-t-elle ?** Que se passerait-il si on la
   supprimait ?
2. **Que promet-elle ?** Lire la docstring comme un contrat.
3. **Consomme-t-elle ce que sa signature annonce ?**
4. **Deux chemins censés coïncider coïncident-ils encore ?**
   → **12 défauts sur 37.** Commencer par là.
5. **Un test traverse-t-il cette configuration ?** Relire chaque fonction ne
   suffit pas : un module reste partiellement audité tant qu'un axe de
   configuration n'a jamais été exécuté. Les axes d'ici : profondeur AMR
   0 / >0, patch périodique / borné, quantique / `classical_only`,
   `state_vector` / échantillonné, warm start absent / présent, hamiltonien
   nul / non nul, COBYLA / autre optimiseur.
   → **D-48**, trouvé par cette question seule : `mode="hardware"` tournait
   sur un simulateur sans le dire, parce qu'aucun test n'avait jamais
   demandé ce mode.

## Ce qui m'a pris du temps

**L'opérateur non assorti — cinq fois.** Mesurer une grandeur avec un
stencil différent de celui qui l'a produite ne mesure pas la grandeur, mais
l'écart entre deux opérateurs. Une fois, un défaut de huit ordres restait
invisible au spectral et sautait aux yeux en FD4. Une autre fois, c'est une
**correction juste** qui paraissait fausse (2,1e−05) : la mesure employait
des dérivées analytiques là où le champ venait d'un stencil FD4 — assortie,
elle rend 1,08e−16.

**Croire qu'un appel réussi prouve quelque chose.** scipy accepte Powell puis
jette silencieusement ses `constraints` ; qiskit-ibm-runtime accepte
`Session(backend=AerSimulator)`. Dans les deux cas le run va au bout et rend
des nombres plausibles. Assertir la **grandeur**, jamais le code de retour.

**Annoncer avant d'avoir lancé la suite complète.** Trois fois. Une correction
juste sur `step_full` cassait l'AMR : huit tests, dont six préexistants. Le
refus de D-48 a fait tomber un test de `test_v1_guards.py` qui passait un mode
inventé — invisible aux 18 tests de la poche comme aux fichiers touchés, il
n'est apparu qu'à 33 % d'un `pytest tests/` intégral.

**Calibrer un seuil sur la mesure du jour.** Mes tests de vortex étaient
périmés trois tours plus tard. La grandeur elle-même n'était pas
reproductible ; aucun seuil n'était le bon.

**Corriger un gel documenté.** Une décision antérieure, écrite dans le
fichier, que je n'avais pas lue. C'est un test qui me l'a rappelée.

**Muter sans vider `__pycache__`.** Une mutation qui remplace un identifiant
par un autre de **même longueur en octets** (`Jz_curl` → `omega_z`) laisse la
clé d'invalidation du `.pyc` — `(mtime, size)` — inchangée si l'écriture tombe
dans la même seconde : Python recharge l'**ancien** module et la campagne
rapporte « mutation tuée » sur du code jamais exécuté. C'est le pire des
faux positifs, puisqu'il certifie qu'un test mord exactement là où il ne mord
pas. `find . -name __pycache__ -exec rm -rf {} +` entre chaque variante.

**Croire qu'un jeu de champs analytiques couvre parce qu'il est varié.**
Quatre champs, quatre structures différentes — et tous **purs** (un seul
signal actif à la fois). Deux normalisations distinctes y rendaient le même
nombre au bit près, et la mutation survivait au fichier entier. La bonne
question à un corpus d'essai n'est pas « couvre-t-il les cas ? » mais **« deux
implémentations différentes peuvent-elles y rendre le même nombre ? »**.

**Le même piège, une couche plus bas.** Le champ mixte ajouté pour corriger
ce qui précède plaçait les deux structures **au même point**. Il excitait bien
les deux modes — et ne séparait toujours rien, parce que deux formules ne
divergent que là où elles font **interagir** les signaux. Un champ d'essai
doit reproduire la **géométrie** du phénomène, pas seulement sa liste
d'ingrédients. Coût : trois tests rouges pour s'en apercevoir, et ils étaient
rouges dans le bon sens.

**Mesurer sur le corpus AVANT de corriger.** Le vrai poids de ce défaut
(facteur 179, une structure morte sur 2 scénarios sur 4) n'est apparu qu'en
mesurant sur les DNS réels, pas sur les champs analytiques. La mesure de
laboratoire dit *si* un mécanisme existe ; seule la mesure sur le corpus dit
s'il **compte**.

**Lire un test rouge trop vite, dans le sens qui m'arrange.**
`test_the_ground_state_is_uniform_on_real_deployed_coefficients` est devenu
rouge après un changement à moi, et j'y ai lu « la dégénérescence D-45/D-47
est corrigée » — le résultat le plus spectaculaire possible. Le champ de ce
test est du **bruit gaussien** ; son nom dit « real deployed coefficients »
pour désigner le chemin de code, pas les données. Remesuré sur 40 instantanés
DNS : 97,5 % → 90 % d'états uniformes. La dégénérescence bouge et ne tombe
pas. **Un test rouge dit qu'une valeur a changé, jamais pourquoi ni combien
ça compte** — et un nom de test n'est pas une garantie de portée.

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
  été vus qu'en cessant de projeter B. Aucune des cinq questions ne les
  aurait trouvés.

## L'ordre dans lequel je relis

Un module, les cinq questions, **jusqu'au bout**. Mieux vaut un module
épuisé que dix survolés. Quand il est fini, écrire ce qui a été vérifié et
trouvé sain — c'est un résultat, et cela évite de le relire deux fois.

## Le réglage à ne pas rater

**La majorité du code est juste** : 164 des 180 nombres publiés inchangés
après 37 défauts trouvés. Un relecteur qui rapporte un défaut par fonction se
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
