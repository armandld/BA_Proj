# Défauts

**Où ça bloque.** Uniquement ce qui n'est pas résolu : ce qui empêche
d'avancer, comment on est tombé dessus, et où on en est pour le lever.

Ce qui est corrigé n'est **pas** ici — c'est un résultat, il vit dans
`RESULTS.md` avec sa mesure et sa commande de vérification.

## Règle d'arrêt — ce qui entre dans ce fichier

Écrite parce que le taux de découverte a dépassé le taux de résolution.
Comptés sur les défauts D-39 à D-131 des deux branches :

| zone | commits |
|---|---|
| `src/` + `study/` — le chemin scientifique | 98 |
| figures et lanceurs | 42 |
| gardes de test (faux verts, mutations) | 37 |

**Près de la moitié de l'effort récent ne touche ni un nombre du papier ni
la campagne.** Le travail est juste — les faux verts de D-123 à D-131 sont
de vrais défauts — mais sa valeur marginale s'est effondrée, et c'est ce
qui donne l'impression d'avancer de plus en plus lentement : on découvre
plus vite qu'on ne ferme, sur des objets de moins en moins décisifs.

Un défaut est **bloquant**, et n'entre ici, que s'il satisfait l'un des
deux critères :

1. **il porte une lecture publiée** — un nombre ou une phrase que le
   manuscrit contiendra ; ou
2. **il empêche la réoptimisation** de mesurer ce qu'elle prétend mesurer.

Tout le reste — rendu des figures, hygiène des tests, lanteurs hors
chemin — est **enregistré et groupé**, traité en un lot unique APRÈS la
campagne. Un défaut hors chemin critique se note en une ligne dans
`RESULTS.md` et ne s'ouvre pas ici.

Ce n'est pas un abandon de rigueur : c'est le même principe que
« mesurer avant d'affirmer », appliqué à l'allocation du temps. Un audit
qui ne finit jamais ne protège aucune conclusion, parce qu'il n'y a pas de
conclusion.


| | |
|---|---|
| **ouverts** — décision ou campagne requise | **14** |
| **gelés** volontairement | 2 |

*(compté, pas estimé : `grep -c '^## D-' docs/DEFAUTS.md` → **14** au 18 août
(nuit) — le paragraphe ci-dessous s'arrêtait à 13, avant que **D-165** ne
soit entré (rapport seul, décision requise) sans que le compte de tête ne
soit mis à jour. C'est exactement le travers que ce paragraphe se reproche
déjà à lui-même deux lignes plus bas : recompter, ne pas estimer, à chaque
ajout, pas seulement à chaque fusion.
12 après la fusion du 17 août au soir, puis **D-158 est entré** — rapport
seul, décision requise). D-69 est sorti au profit de `RESULTS.md`, sa
table étant refaite ; D-141 est entré. Puis **D-59 est sorti** — corrigé sur
la branche vive à `7b12857`, il vit désormais dans `RESULTS.md` — et
**D-143 est entré** ; côté branche vive, **D-47 est sorti** — décision de
USER, option 1, il vit dans `RESULTS.md` comme résultat. Les
deux côtés de la fusion annonçaient 13 et 11 : ni l'un ni l'autre n'était le
compte de l'arbre fusionné. Le compte de tête de `RESULTS.md` porte le même
travers et le dit lui-même : recompter, ne pas estimer, à chaque fusion.)*

**D-58, ajouté cette passe, touche lui aussi une lecture publiée — et depuis
plus longtemps que les autres.** La narration T17 (« ZZ est numériquement
mort sur trois classes sur quatre ») date d'avant D-9, qui a corrigé le
défaut exact que T17 décrit et a démontré l'inverse (« ZZ domine K de 1,5 à
8,2× ») — sans que la prose de T17 ni les constantes de référence de
`aggregate_master_table.py` ne soient mises à jour. 12 des 16 lignes DIFF du
master table viennent de là, pas d'un nouveau bruit.

**D-53 est le plus lourd de la liste et se lit en premier.** Les trois
artefacts `dim = 3` du dépôt — la seule taille à la fois **certifiée** et
**non dégénérée** jamais exécutée — contredisent le critère d'acceptation de
H0 : le QAOA y atteint l'optimum certifié sur **0,062 à 0,156** des
instantanés là où le critère exige 1,000, et tombe plus loin de l'optimum
que la règle classique dont il part (0,500). Rejoué sur ces artefacts, le
critère du module lève : *« H0 redevient plausible »*. Aucun nombre publié
n'en dépend — `dim3` n'apparaît dans aucun document et dans aucune des 180
lignes du master table. C'est la lecture, pas le nombre, qui est en jeu.

Des deux qui restent, l'un demande la campagne elle-même, l'autre une décision
qu'on peut prendre après. D-27 et D-37 sont sortis d'ici : corrigés, mesurés,
verrouillés — ils vivent dans `RESULTS.md`.

D-47 bloque la phase 4 et tout ce qui s'appuie sur sa sélection, sans bloquer
la réoptimisation.

**D-48 et D-50 sont les derniers arrivés, et ce sont les seuls qui touchent
une lecture publiée.** D-48 : la tendance décroissante de T11b est mesurée
comme une propriété du schedule d'initialisation, pas du circuit. D-50 : la
phrase de conclusion que T11b imprime bascule d'une exécution à l'autre de la
même commande — une sur trois donne l'inverse. Aucun nombre publié n'a bougé —
le master table reste à 180 / 164 / 16 / 0 — mais leur lecture demande une
décision. Les deux se composent et **ne se confondent pas** : D-50 tient même
si D-48 est tranché en ne changeant rien.

**D-51 est à lire avant de lancer la campagne de D-22.** Tout `study/` code
`advanced_anomalies_enabled = False` alors que la campagne d'entraînement
l'active sur 6/6 scénarios : le terme ZZZZ de point X n'entre dans aucune
mesure de falsification. Mesuré nul à `dim = 2`, donc **aucun nombre publié
n'en dépend** — mais `beta_xpoint`, que D-22 range parmi les 8 paramètres à
réoptimiser, est un hyperparamètre qu'aucune mesure de `study/` ne peut voir.

**D-69 est sorti d'ici : la table T31 est refaite, il vit dans
`RESULTS.md`.** Et il emporte une lecture publiée avec lui — « corriger la
convention d'axes dégrade la tâche à dim=16 » est **rétractée** : refaite à
`95571d1`, la table ne porte plus aucun verdict tranché (IC95 du Spearman à
dim=16 : [−0.1328, −0.0146] → [−0.1673, **+0.0343**]). Deux causes
mesurées, pas une : le solveur (D-25, D-26/D-27), puis **D-70 seul** pour
tout le reste du déplacement. Ce qui subsiste est plus faible que ce qui
était écrit : les quatre Δ sont négatifs, aucun n'est significatif.

**Rien ne bloque plus la réoptimisation côté code.** D-132 (bras QAOA
instable selon les hyperparamètres) est **élucidé** : la bisection nomme
`6ecaecf` — D-25, la projection spectrale — dont la correction a retiré
l'artefact sur lequel reposait le classement. Ce n'est pas une régression
à défaire, et l'instabilité résiduelle est précisément ce que la campagne
arbitre.

---

## D-22 — les hyperparamètres déployés n'ont aucune provenance

**Où ça bloque.** Réoptimiser demande de savoir d'où l'on part. Aucun chiffre
de performance n'est attribuable à un réglage dont on ignore l'origine.

**Comment on est tombé dessus.** En auditant `train_hyperparams`, quatre
paramètres écrits comme conditionnels se sont révélés être des constantes. En
remontant aux bases Optuna pour voir lesquels avaient réellement été
échantillonnés, le JSON déployé ne correspondait à aucune.

**Ce qui est établi.**

| | échantillonné par la campagne gelée |
|---|---|
| `beta`, `beta_curl`, `beta_xpoint`, `w_z_frac` | oui, étude quantique |
| `sigma` | oui — mais **absent du JSON**, donc repli sur 0,05 |
| `threshold_amr` | seulement dans l'étude **classique** |
| `gamma_hydro`, `gamma_mag`, `kappa` | **jamais, nulle part** |

L'essai 85 que le JSON déclare a une perte de **0,3213** dans la base contre
**0,2215** annoncée, et **aucun** de ses quatre paramètres communs ne
coïncide. Le code d'entraînement, lui, est cohérent avec les bases : il fixe
`threshold_amr` à 0,14959824837662078, exactement le meilleur essai
classique. **C'est le JSON qui est orphelin.**

**Précision mesurée (D-108) : la ligne `sigma` ci-dessus ne dit pas ce
qu'elle semble dire.** `sigma` est bien échantillonné par `q_has_v2_phase1.db`
et bien absent du JSON déployé — les deux faits tiennent. Mais l'absence
n'est pas un abandon au déploiement : le JSON **ne vient pas de cette
base**. Il porte les clés de scénario de l'ancienne génération
(`lamb_oseen_vortex`, `island_coalescence`), que les CSV de la campagne vive
n'ont plus (`ot`, `rotor`), et ses 8 paramètres sont **exactement ceux qui
survivent au défaut D-108** — l'extracteur jetait `param_beta_grad`, le
prédécesseur de `sigma`, sur toute campagne de cette génération (579 valeurs
jetées, mesuré). La forme à 8 paramètres est donc un **produit de
l'extracteur**, pas un témoignage sur ce qui a été échantillonné, et elle ne
peut pas servir d'argument sur le sort de `sigma`.

Ce que la mesure **ne** change **pas** : le JSON reste orphelin. Balayage des
13 CSV du dépôt — aucune ligne ne partage une seule valeur de paramètre ni le
score avec lui. Le repli de `pipeline.py` sur `sigma = 0,05`, valeur
qu'aucun essai n'a choisie, reste donc entier.

**Où on en est.** Ne se corrige pas par du code seul. Seule la réoptimisation
le règle. Trois paramètres n'ayant jamais été échantillonnés, ce sera pour eux
une *première*, pas une reprise.

**Périmètre : tranché — les 8.** `beta`, `w_z_frac`, `sigma`, `beta_curl`,
`beta_xpoint`, `gamma_hydro`, `gamma_mag`, `kappa`. `threshold_amr` reste gelé
au meilleur essai classique, pour que la comparaison porte sur ce que le
quantique ajoute. Nuance qui allège : `g_strain + g_rot ≡ 1`, donc `kappa` ne
contrôle **qu'un** degré de liberté.

**Le mécanisme est fermé, la campagne reste à faire.** D-35 identifiait
pourquoi le fichier déployé ne correspondait à aucune base : `_save_results`
n'écrivait que `study.best_params`, c'est-à-dire les seuls paramètres
*échantillonnés*. Le JSON écrit désormais le jeu complet, l'espace de
recherche avec ses bornes, le hash du commit et `sys.argv`. Une campagne
lancée aujourd'hui produira donc un fichier traçable — mais le fichier
**actuellement déployé** reste orphelin jusqu'à ce qu'elle tourne.

**Reste à trancher avant de lancer.** La borne haute de `w_z_frac` vaut 1000
alors que le paramètre est documenté comme une *fraction*. Elle vient de la
campagne gelée (graine 500). Conservée telle quelle pour ne pas changer la
science en même temps que le code.

```bash
python src/train_hyperparams.py --print-space          # l'espace réel
pytest tests/pipeline/test_hyperparams_provenance_break.py
pytest tests/pipeline/test_train_hyperparams_contracts.py
```

Le dernier test est le **critère d'acceptation** : `xfail` aujourd'hui, il
passera sans modification le jour où chaque valeur déployée sera traçable.

---

## D-24 — le solveur est d'ordre 1,2 — **DÉCIDÉ : assumé**

**Décision de USER : assumé, et écrit comme une limite.** La correction
(projection du second membre) restaure l'ordre **4,00 à divergence égale**,
mais n'est valide que sur `step_full` — la grille globale périodique. Les
patchs AMR de `step_layered` sont **locaux et non périodiques**, et la
projection spectrale suppose la périodicité : l'appliquer là serait faux
d'une façon plus difficile à voir que l'erreur actuelle.

Ce qui rend l'acceptation défendable : la chute d'ordre frappe **les deux
bras à l'identique** — Q-HAS et l'AMR classique tournent sur le même
solveur, les mêmes patchs. Elle gonfle l'erreur absolue de tous les
nombres, mais elle ne peut pas fabriquer un écart entre les deux bras, et
toutes les affirmations du papier sont comparatives. C'est donc une limite
à énoncer, pas un biais à retirer. Déjà listée dans `PLAN_PREPRINT.md` §7.

Reste ici, et non dans `RESULTS.md`, parce que rien n'est corrigé : c'est
une limite acceptée, pas un défaut fermé.

**Où ça bloque.** Nulle part en toute rigueur : la chute est **commune aux
deux bras**, donc elle ne biaise pas leur comparaison. Elle comprime la plage
dans laquelle un meilleur critère pourrait se distinguer.

**Comment on est tombé dessus.** Le solveur revendique l'ordre 4 pour ses
dérivées spatiales — vérifié, 3,97 / 4,02 / 3,99. Le solveur complet converge
à 1,2. À grille fixe, en ne raffinant que le pas de temps : avec projection
1,12, sans projection 4,00. Sept ordres de grandeur d'écart.

**Ce qui a été essayé, et mesuré.**

| tentative | résultat |
|---|---|
| splitting de Strang | **erreurs identiques** — un projecteur idempotent n'a pas de demi-pas |
| projection du second membre, `step_full` | ordre **4,00**, erreur 52 000× plus petite |
| la même, `step_layered` | **1,94e−02, indépendant du pas** — plancher créé par la projection par patch |
| Van Kan, pression incrémentale | ordre inchangé, gain 2,4 % |

La projection spectrale est **non locale** : la phase 2 la calcule sur un
patch avec halo traité comme périodique. Ce n'est pas la projection globale
restreinte, et aucun découpage ne peut la reproduire. Projeter le second
membre casserait la garantie « à `max_depth`, `step_layered` ≡ `step_full` »,
qui tient aujourd'hui à **3,331e−16**.

**Où on en est.** `PROJECT_RHS = False`, raison écrite dans le code. Ce n'est
pas un bug d'implémentation : une méthode de projection à la Chorin **est**
d'ordre 1.

**Décision attendue**, par coût croissant :

1. **Laisser en l'état**, documenter comme limite. *(Recommandé.)*
2. Contrainte **locale** — volumes finis, transport contraint pour B, Poisson
   multigrille pour v. C'est ce que font FLASH, Athena++, AMReX. Ordre 2
   réaliste ; l'ordre 4 avec AMR est un sujet de recherche.
3. Formulation à pression — réécriture du cœur.

Les options 2 et 3 réécrivent `src/Simulation/` et invalident tout nombre
publié.

```bash
pytest tests/solver/test_solver_convergence.py -m slow     # ~10 min
```

---

## La borne haute de `w_z_frac` — décision avant de lancer

**Où ça bloque.** Nulle part techniquement : la campagne partirait. Mais elle
explorerait un domaine dont la moitié haute n'a peut-être aucun sens.

**Comment on est tombé dessus.** En déclarant l'espace de recherche pour
l'audit du script d'entraînement (D-35). Écrire les bornes comme des données
oblige à les lire.

**Ce qui est établi.** `w_z_frac` est documenté comme la **fraction** de la
médiane des couplages qui donne son poids au biais Z :
`alpha_z = w_z_frac × median(|C|, |K|)`. Sa borne haute vaut **1000**. Une
fraction de 1000 signifie un biais Z mille fois plus grand que les couplages,
c'est-à-dire un Hamiltonien où les termes ZZ et ZZZZ ne pèsent plus rien — le
quantique dégénère vers la décision classique.

La borne vient de la campagne gelée, dont la graine valait 500. Conservée
telle quelle pour ne pas changer la science en même temps que le code.

**Décision attendue.** Resserrer à un intervalle où le mot « fraction » a un
sens (par exemple 0,01–10, en log), ou garder 1000 et documenter que la
partie haute du domaine teste la dégénérescence vers le classique. La
seconde option est défendable — mais alors il faut le dire.

```bash
python src/train_hyperparams.py --print-space
```

---

## D-39 — le check de tearing n'a plus de signal avec l'observable corrigée

**Où ça bloque.** `dns_validation.main()` classe **harris_tearing comme
WEAK sur les 4 Re canoniques** (400/800/1200/1600) et sur les deux
résolutions supplémentaires présentes dans `results/` (N64, N96). C'est le
seul des trois checks 1b (OT, KH, tearing) qui porte sur la reconnexion :
le pipeline de validation phase 1b a perdu son seul signal qualitatif de
reconnexion magnétique, silencieusement, depuis D-21.

**Comment on est tombé dessus.** `memory#3` (revue d'Evening) signale que
`claude/kind-babbage-927g10` aurait « cassé son propre gel de
reproductibilité » : `analyse_one` appelle désormais
`mean_sq_current_fixed` au lieu de `mean_sq_current` (`be72b519`, D-21),
quatre commits après que `2b4f400` (D-18) a restauré les FONCTIONS gelées.

Vérification indépendante du texte du gel d'abord : `2b4f400` et `be72b51`
sont explicites — « le gel porte sur les FONCTIONS, pas sur l'analyse »
(commentaire dans `dns_validation.py:176-183`, repris dans `DEFAUTS.md`
lui-même, section « Gelés volontairement » ci-dessous). Rien n'a été
« cassé » au sens d'une décision annulée : `analyse_one` n'a jamais été
gelée, seules `mean_sq_current`/`fluctuating_KE` le sont, et la reproduire
en l'état reproduirait le défaut D3 documenté.

Mais l'**effet** qu'Evening rapporte est réel — vérifié indépendamment en
rejouant `check_tearing` sur les 6 fichiers DNS `harris_tearing` présents
dans `results/`, avec `J2` calculé soit par `mean_sq_current` (brut, gelé,
= l'ancien câblage d'`analyse_one` avant D-21) soit par
`mean_sq_current_fixed` (corrigé, câblage actuel) :

| Re | N | amplification ANCIEN (brut) | ok | amplification ACTUEL (corrigé) | ok |
|---|---|---|---|---|---|
| 400 | 64 | 1.53× | **True** | 1.000× | **False** |
| 400 | 96 | 1.96× | **True** | 1.000× | **False** |
| 400 | 256 | 2.65× | **True** | 1.000× | **False** |
| 800 | 96 | 2.27× | **True** | 1.039× | **False** |
| 1200 | 96 | 2.35× | **True** | 1.063× | **False** |
| 1600 | 96 | 2.40× | **True** | 1.077× | **False** |

**Hypothèse de cause, non tranchée.** `mean_sq_current(_fixed)` moyenne
`J_z²` sur **tout le domaine**, y compris le courant d'équilibre de la
nappe de Harris, `J_z0(y)`, uniforme le long de `AXIS_X` (= axe 0) et donc
non nul dès `t=0`. Ce fond domine la moyenne spatiale et ne croît presque
pas pendant la simulation (amplification 1,00–1,08×). La version brute
(défaut D3 : `∂By/∂y − ∂Bx/∂x`, une combinaison de DÉFORMATION) est nulle
par construction sur l'équilibre de la nappe (cisaillement pur) — elle
ignore donc *par accident* le fond stationnaire et ne répond qu'à la
composante non homogène en x qui grandit réellement pendant la
reconnexion. Le check qualitatif marchait avec le défaut D3 en jouant, sans
le vouloir, le rôle d'un filtre passe-haut que la version corrigée n'a pas.

Comparer aux séries temporelles : sur `Re400_N256`, `J2` (corrigé)
DÉCROÎT de façon monotone de `t=0` à `t=1.91` (0.000851 → 0.000802) ; `J2`
(brut) décroît puis remonte d'un facteur 3,5× après `t≈0.9` — le profil
qualitatif attendu d'une instabilité de reconnexion.

**Ce que ce n'est pas.** Ni un bug de `mean_sq_current_fixed` (verrouillée
par 22+26 tests dans `test_t8_dns_extension.py` /
`test_no_private_curl_survives.py`, formule conforme à la convention
`AXIS_X=0`/`AXIS_Y=1`), ni le gel remis en cause (les fonctions gelées
restent inchangées). C'est `check_tearing` — ou le choix d'observable
qu'`analyse_one` lui fournit — qui ne sépare pas fond stationnaire et
croissance de la reconnexion.

**Où on en est.** Pas corrigé. Deux issues possibles, à trancher : (a)
`check_tearing` compare une composante **fluctuante** de `J²` — soustraire
la moyenne le long de `AXIS_X`, comme `fluctuating_ke_fixed` le fait déjà
pour `Ep` — plutôt que la moyenne pleine grille ; (b) la fenêtre
`[0, t_max]` actuelle de `harris_tearing` ne développe simplement pas assez
la reconnexion pour que le signal dépasse le fond dans une moyenne globale,
et c'est la scène (ou `t_max`) qu'il faut revoir. Ne pas rebrancher
`analyse_one` sur les fonctions gelées : ce serait réintroduire D3 pour
faire retomber `ok=True`, exactement la « recorrection par erreur » que ce
fichier existe pour empêcher.

**Mise à jour (D-42, `RESULTS.md`).** Le tableau ci-dessus, colonne
« ancien (brut) », doit se relire : `check_tearing` portait un second
défaut indépendant — sa clause « pic pas à la fin de la trace » se
comparait à elle-même quand le pic tombait sur le dernier échantillon, donc
ne pouvait jamais échouer. Sur les 6 fichiers ci-dessus, le pic (câblage
brut comme câblage corrigé) tombe justement sur le dernier échantillon —
une croissance qui ne retombe jamais dans la fenêtre simulée, pas un pic
observé. Une fois D-42 appliqué, les deux câblages rendent `ok=False` sur
les 6/6 : le câblage brut ne « marchait » que grâce à ce défaut, pas parce
qu'il observait un vrai pic. La question posée ici (quelle observable
sépare fond stationnaire et reconnexion) reste ouverte, mais sans l'appui
de « l'ancien câblage passait ».

```bash
python3 -c "
import sys, glob, re, numpy as np
sys.path.insert(0, 'study/pipeline'); sys.path.insert(0, 'src')
import dns_validation as dv
for path in sorted(glob.glob('results/dns_harris_tearing_Re*_N*.npz')):
    d = np.load(path)
    vx=d['vx'].astype(float); vy=d['vy'].astype(float)
    Bx=d['Bx'].astype(float); By=d['By'].astype(float); t=d['t'].astype(float)
    n = vx.shape[0]
    J2f = np.array([dv.mean_sq_current_fixed(Bx[i],By[i]) for i in range(n)])
    J2r = np.array([dv.mean_sq_current(Bx[i],By[i]) for i in range(n)])
    cf = dv.check_tearing({'t':t,'J2':J2f}); cr = dv.check_tearing({'t':t,'J2':J2r})
    m = re.search(r'Re(\d+)_N(\d+)', path)
    print(m.group(0), 'fixed', cf['ok'], round(cf['amplification'],3),
          '| raw', cr['ok'], round(cr['amplification'],3))
"
```

**Addendum du 19 août (Vigil) — l'hypothèse (a) testée, elle ne tranche pas
seule.** Mesure directe sur les 6 fichiers DNS `harris_tearing` : `J_z`
recalculé avec `mean_sq_current_fixed` (opérateur assorti), puis sa
composante fluctuante isolée par `J - J.mean(axis=AXIS_X, keepdims=True)` —
retrait du fond stationnaire `J_z0(y)`, uniforme le long de `AXIS_X` comme
D-39 le nomme déjà.

| Re | N | creux (indice) | amplification creux→fin | pic dans la fenêtre ? |
|---|---|---|---|---|
| 400 | 64 | i=7 | **8,32×** | non — dernier échantillon |
| 400 | 96 | i=7 | **11,86×** | non |
| 400 | 256 | i=6 | **14,27×** | non |
| 800 | 96 | i=7 | **15,53×** | non |
| 1200 | 96 | i=7 | **16,87×** | non |
| 1600 | 96 | i=7 | **17,62×** | non |

Sur `Re400_N256`, la trace complète de la composante fluctuante : décroît
légèrement de `t=0` à `t≈0,6` (3,83e−09 → 3,12e−09), puis croît **de façon
monotone et sans retomber** jusqu'à `t=1,905`, fin de fenêtre (3,12e−09 →
4,45e−08). C'est le profil qualitatif attendu d'une instabilité de
reconnexion — croissance franche, 6/6 fichiers, facteur cohérent avec
l'ordre de grandeur de Re — mais `check_tearing.ok` reste **`False` sur les
6/6** : `i_peak` tombe sur le dernier échantillon dans les six cas, donc la
clause `not_pinned_at_end` (D-42) échoue.

**Ce que ça établit : (a) et (b) se composent, elles ne s'excluent pas.**
(a) seule — changer d'observable pour la composante fluctuante — **récupère
un vrai signal de reconnexion**, invisible aux deux câblages actuels (brut
comme corrigé). Mais (b) reste vraie simultanément : aucune des 6 traces ne
redescend dans `[0, t_max]`, donc même l'observable corrigée ne ferait pas
lever `check_tearing.ok` sans toucher aussi au critère de pic ou à la
fenêtre temporelle. Choisir (a) seule, tel qu'énoncé, ne suffit pas à faire
passer le check — ce n'était pas mesuré avant cette passe.

```bash
python3 -c "
import sys, glob, re, numpy as np
sys.path.insert(0, 'study/pipeline'); sys.path.insert(0, 'src')
import dns_validation as dv
from Simulation.grid import AXIS_X, AXIS_Y
def mean_sq_current_fluct(Bx, By):
    dxBy = (np.roll(By, -1, axis=AXIS_X) - np.roll(By, 1, axis=AXIS_X)) * 0.5
    dyBx = (np.roll(Bx, -1, axis=AXIS_Y) - np.roll(Bx, 1, axis=AXIS_Y)) * 0.5
    J = dxBy - dyBx
    return (J - J.mean(axis=AXIS_X, keepdims=True))**2
for path in sorted(glob.glob('results/dns_harris_tearing_Re*_N*.npz')):
    d = np.load(path)
    Bx=d['Bx'].astype(float); By=d['By'].astype(float); t=d['t'].astype(float)
    n = Bx.shape[0]
    J2 = np.array([mean_sq_current_fluct(Bx[i],By[i]).mean() for i in range(n)])
    c = dv.check_tearing({'t':t,'J2':J2})
    m = re.search(r'Re(\d+)_N(\d+)', path)
    i_min = int(np.argmin(J2))
    print(m.group(0), 'ok', c['ok'], 'amp(creux->fin)',
          round(J2[-1]/max(J2[i_min],1e-30), 3), 'i_peak==last?', c['t_peak']==t[-1])
"
```

---

## D-41 — le seuil critique du hamiltonien v1 n'est jamais franchi sur 2 scénarios canoniques / 4

**Où ça bloque.** Toute comparaison "hamiltonien quantique vs score
classique" sur `harris_tearing`/`kelvin_helmholtz` porte en réalité sur un
hamiltonien identiquement nul (voir D-40, `RESULTS.md`) : `PhysicalMapper`
(v1) ne produit **aucun** coefficient non nul sur ces deux scénarios, à
Re=400 comme à N=256, du premier au dernier snapshot. Toute campagne future
qui réoptimiserait ou publierait un résultat comparatif sur ces scénarios à
cette résolution comparerait en réalité "hasard construit" contre le score
classique, pas un vrai hamiltonien.

**Comment on est tombé dessus.** En auditant `study/pipeline/hamiltonian_coefficients.py`
(FOCUS `pipeline/`) : les 4 fichiers `coefficients_*_Re400_N256_dim4.npz`
existants montrent `E_patch` non nul pour `mhd_rotor` (100 % des cellules
actives) et `orszag_tang` (70 %), mais **0 %** pour `harris_tearing` et
`kelvin_helmholtz` — rejoué indépendamment avec le code actuel
(`compute_patch_coefficients`), pas seulement lu depuis l'artefact.

**Ce qui est établi.** Dans `PhysicalMapper.compute_coefficients`
(`src/Simulation/HamiltParams.py`), le terme ZZ (`C_edges`) est nul dès que
`v_jump`/`B_jump` (saut de cellule entre voisins) reste sous
`v_jump_crit = RE_CRIT·ν/dx` / `B_jump_crit = RM_CRIT·η/dx`. Mesuré sur
`dns_harris_tearing_Re400_N256.npz`, snapshot médian (t≈1,0) :

| | v_jump max | v_jump_crit | rapport |
|---|---|---|---|
| harris_tearing Re400 N256 | 4,56e−4 | 0,102 | **0,004** |
| B_jump max | 0,085 | 0,102 | **0,83** (proche, jamais franchi) |

Le terme ZZZZ (`K_plaquettes`) dépend du même type de seuil sur la
vorticité/le courant ; `omega_mag`/`jz_mag` y restent aussi, mesuré, sous
`omega_crit`/`jz_crit` sur toute la trace (`v0` s'annule algébriquement du
rapport — vérifié, ce n'est pas un artefact de normalisation par `v0`).

**Ce que ce n'est probablement pas** : le commentaire du fichier explique
que `RE_CRIT` est passé de 10 à 1,0 précisément pour éviter « un hamiltonien
vide pour la plupart des grilles de résolution de simulation » — mesuré ici,
le seuil à 1,0 laisse malgré tout `harris_tearing`/`kelvin_helmholtz`
complètement vides à la résolution de production (N=256). Les deux
scénarios à discontinuités (`mhd_rotor`, `orszag_tang`) franchissent le
seuil sans peine.

**Hypothèse de cause, non tranchée.** `harris_tearing`/`kelvin_helmholtz`
sont des instabilités **lisses** (pas de choc) : leurs sauts inter-cellules
restent proportionnels à `gradient physique × dx`, qui s'écrase avec la
résolution, alors que `v_jump_crit` croît en `1/dx`. `mhd_rotor`/
`orszag_tang` contiennent de vrais chocs (saut ≈ constant, indépendant de
`dx`), d'où un rapport qui ne s'écrase pas à N=256. Si c'est le cas,
`RE_CRIT`/`RM_CRIT` mesurent une vraie propriété physique (pas de structure
à l'échelle de la cellule) et le hamiltonien vide serait un résultat
correct, pas un défaut — mais alors les scénarios lisses ne peuvent
structurellement jamais bénéficier du terme ZZ/ZZZZ v1 à résolution DNS,
ce qui n'est écrit nulle part et devrait l'être avant toute réoptimisation
qui inclurait ces scénarios.

**Où on en est.** Pas corrigé : changer `RE_CRIT`/`RM_CRIT` change un
paramètre physique du hamiltonien étudié dans `src/`, donc un nombre
publié — décision humaine, pas un défaut de code à corriger seul. Deux
issues possibles, à trancher : (a) documenter que le hamiltonien v1 est
structurellement vide sur les scénarios lisses à résolution DNS, et exclure
`harris_tearing`/`kelvin_helmholtz` de toute comparaison quantique/classique
tant que ce n'est pas vrai ; (b) abaisser encore `RE_CRIT`/`RM_CRIT`, ou
changer leur normalisation pour qu'ils réagissent au gradient plutôt qu'au
saut brut, et remesurer si cela casse la calibration sur `mhd_rotor`/
`orszag_tang`.

```bash
python3 -c "
import sys, numpy as np
sys.path.insert(0, 'src'); sys.path.insert(0, 'study/pipeline')
from hamiltonian_coefficients import compute_patch_coefficients
from config import (TRAINED_THRESHOLD, TRAINED_SIGMA, TRAINED_BETA_CURL,
    TRAINED_BETA_XPOINT, TRAINED_W_Z_FRAC, TRAINED_GAMMA_HYDRO,
    TRAINED_GAMMA_MAG, TRAINED_KAPPA)
for sc in ['harris_tearing', 'kelvin_helmholtz', 'mhd_rotor', 'orszag_tang']:
    dns = np.load(f'results/dns_{sc}_Re400_N256.npz')
    si = len(dns['t']) // 2
    vx=dns['vx'][si].astype(float); vy=dns['vy'][si].astype(float)
    Bx=dns['Bx'][si].astype(float); By=dns['By'][si].astype(float)
    N = vx.shape[0]
    res = compute_patch_coefficients(vx, vy, Bx, By, N, 4, 400, TRAINED_THRESHOLD,
        sigma=TRAINED_SIGMA, beta_curl=TRAINED_BETA_CURL, beta_xpoint=TRAINED_BETA_XPOINT,
        w_z_frac=TRAINED_W_Z_FRAC, gamma_hydro=TRAINED_GAMMA_HYDRO,
        gamma_mag=TRAINED_GAMMA_MAG, kappa=TRAINED_KAPPA)
    print(sc, 'E max =', res['E_patch'].max())
"
```

**Addendum du 19 août (Vigil) — l'hypothèse de cause testée par résolution
croisée, à Re=400 fixe.** Les DNS existantes couvrent N=64/96/256 pour les
quatre scénarios canoniques à Re constant : de quoi observer directement
comment `v_jump_max` (numérateur du rapport) réagit à `dx`, plutôt que de
le supposer. Snapshot médian, même opérateur que `HamiltParams`
(`_get_vector_jump`) :

| scénario | v_jump·N (N=64) | v_jump·N (N=96) | v_jump·N (N=256) | variation N=64→256 |
|---|---|---|---|---|
| harris_tearing | 0,1634 | 0,1573 | 0,1168 | **−29 %** |
| kelvin_helmholtz | 12,355 | 12,428 | 12,491 | **+1 %** (plat) |
| mhd_rotor | 160,65 | 208,55 | 294,28 | **+83 %**, croissant |
| orszag_tang | 57,37 | 67,26 | 62,20 | +8 %, non monotone |

Si `v_jump` suit un gradient physique résolu (champ lisse), `v_jump ∝ dx`
et `v_jump·N` doit rester **constant** quand `N` varie à `Re` fixé — c'est
l'hypothèse (a) de cause. Si `v_jump` porte un vrai saut de grille (choc),
il ne s'écrase pas avec `dx`, et `v_jump·N` doit **croître** avec `N`.

**Ce qui est établi.** Les deux scénarios que D-41 dit lisses sont bien
ceux dont `v_jump·N` varie le moins (≤ 29 % sur un facteur 4 en résolution,
contre un facteur ~22 pour le rapport `v_jump/v_jump_crit` lui-même, qui
lui s'écrase en `dx²` comme prédit) — `kelvin_helmholtz` est quasiment
invariant (**+1 %**). `mhd_rotor`, le scénario à choc le plus net dans
D-41 (100 % des cellules franchissent le seuil), est celui dont
`v_jump·N` **croît** le plus (+83 %) : le saut ne s'écrase pas, il se
précise à mesure que le maillage se resserre — signature inverse de
celle des deux scénarios lisses. `orszag_tang`, que D-41 mesure
intermédiaire (70 % des cellules), a un `v_jump·N` non monotone, cohérent
avec un mélange de région lisse et de région choquée.

**Ce que ça ne tranche toujours pas.** La mesure resserre l'hypothèse de
cause — elle n'était pas mesurée en croisant les résolutions avant cette
passe — mais elle ne dit pas laquelle des options (a)/(b) choisir : `(a)`
reste défendable (le hamiltonien v1 est structurellement aveugle aux
scénarios lisses à résolution DNS, par construction et non par bug), et
`(b)` n'est ni confirmée ni écartée par cette mesure seule. Décision
toujours à USER.

```bash
python3 -c "
import sys, numpy as np
sys.path.insert(0, 'src')
def get_vector_jump(f_x, f_y, axis):
    df_x = f_x - np.roll(f_x, -1, axis=axis)
    df_y = f_y - np.roll(f_y, -1, axis=axis)
    return np.sqrt(df_x**2 + df_y**2)
Re = 400
for sc in ['harris_tearing', 'kelvin_helmholtz', 'mhd_rotor', 'orszag_tang']:
    print(sc)
    for N in [64, 96, 256]:
        d = np.load(f'results/dns_{sc}_Re{Re}_N{N}.npz')
        vx = d['vx'].astype(float); vy = d['vy'].astype(float)
        si = len(d['t']) // 2
        vj = max(get_vector_jump(vx[si], vy[si], 1).max(),
                 get_vector_jump(vx[si], vy[si], 0).max())
        print(f'  N={N} v_jump*N={vj*N:.4f}')
"
```

---


## D-48 — le « warm start classique » du QAOA ne lit pas la décision classique

**Où ça bloque.** La lecture publiée de T11b — *« mean progress 0.0854,
monotonically decreasing with depth and negative by reps = 4 »* — et
l'interprétation pré-spécifiée que le module en tire (*« une progression qui
n'augmente pas avec la profondeur signifie que l'objectif déclaré n'est pas
l'objectif optimisé »*) sont mesurées ici comme une propriété du **schedule
constant**, pas du bras QAOA. Sous l'initialisation par défaut du dépôt — la
rampe en `π/E_max` d'`execute()`, celle que `refinement.py` emprunte tant que
son `warm_start_cache` est vide — la progression moyenne double et la
tendance en profondeur s'annule. Tant que la question n'est pas tranchée, on
ne sait pas si T11b décrit le circuit ou son point de départ.

**Comment on est tombé dessus.** Question 3 de `VIGIL.md`, en lisant
`study/common/qaoa_inputs.py` en entier : `classical_warm_start_params(score_vqa,
threshold_amr, reps)` ne consomme aucun de ses deux premiers arguments.

**Ce qui est établi — le contrat.** Six entrées couvrant tout l'intervalle
(score nul, score unité, score aléatoire, seuil 0 / 1 / 1e9), `reps = 2` :

| | |
|---|---|
| sorties identiques bit-à-bit | **6/6** |
| écart maximal | **0,0e+00** |
| valeur rendue | `beta = (0,05 ; 0,05)`, `gamma = (0,15 ; 0,075)` |

Les deux arguments sont morts. Le nom, la docstring (« from the classical AMR
decision ») et l'aide CLI de `--warm-start` (« classical-score-derived »)
annonçaient l'inverse. Ils sont corrigés dans le code ; **le schedule, lui,
est inchangé bit-à-bit.**

Ce qui aggrave la lecture : les appelants passent ces deux arguments **à côté
de warm starts qui, eux, sont réels**. Dans `h0_optimiser_equivalence.solver_panel`,
`sa_warm` et `greedy` démarrent sur `classical_init_spins(score_vqa, thr_amr, dim)`.
Le bras QAOA, à trois lignes de là, reçoit un appel de forme identique qui ne
lit rien. Et **aucun chemin déployé n'utilise ce schedule** : `refinement.py`
enchaîne `warm_start_cache` (les paramètres optimaux du patch précédent), à
défaut `None` → rampe `E_max`. La phase 5 (`run_phase5`) fait de même et ne
sert le schedule constant qu'au **premier** instantané, sous `--warm-start`.
Ce sont les études h0 / h3 qui l'appliquent à **chaque** appel.

**Ce qui est établi — la conséquence sur un nombre publié.** Configuration
publiée de `h0_qaoa_displacement` : `N=256`, `dim=2`, mapper v2, `k_opt=100`,
`shots=4096`, `--n-snaps 2` (8 instantanés, 4 scénarios), reps 1 à 4. Le bras
QAOA n'étant pas déterministe, **trois répétitions par bras et par
profondeur** ; « séparé » veut dire que les enveloppes des trois répétitions
ne se recouvrent pas.

| reps | warm — schedule constant (ce que le code fait) | cold — rampe `E_max` (défaut du dépôt) | séparé ? |
|---|---|---|---|
| 1 | +0,1674  +0,1670  +0,1515 | +0,1884  +0,1775  +0,1836 | **oui** |
| 2 | +0,1320  +0,0881  +0,1304 | +0,1939  +0,1983  +0,1876 | **oui** |
| 3 | +0,0670  +0,0320  +0,0226 | +0,1689  +0,1980  +0,1856 | **oui** |
| 4 | +0,0472  +0,0527  +0,0391 | +0,1934  +0,1697  +0,1858 | **oui** |

| | warm | cold | publié |
|---|---|---|---|
| moyenne sur reps 1–4 | **+0,0914** | **+0,1859** | +0,0854 |
| tendance reps 1 → 4 | −0,1202 / −0,1143 / −0,1124 → **−0,116** | +0,0050 / −0,0078 / +0,0022 → **−0,0002** | −0,172 |
| décroît sur les 3 répétitions ? | **oui, 3/3** | **non** — deux tendances sur trois sont *positives* | « still decreasing » |

Le bras warm **reproduit** la lecture publiée : moyenne du même ordre,
tendance négative aux trois répétitions. Le bras cold ne la reproduit pas :
progression **2,0×** plus haute et tendance nulle à ±0,006, de signe variable.
L'écart entre bras dépasse la dispersion intra-bras aux **quatre**
profondeurs.

**Ce que cela ne dit pas.** Les décisions, elles, ne bougent pas : mesurées
sur les 4 scénarios canoniques (`N=256`, `dim=2`, `reps=2`, 2 répétitions par
bras), **0 différence de décision** entre warm et cold, masques tous à
« raffiner partout » — c'est-à-dire la dégénérescence de D-45 / D-47. Les
lignes T11 du master table (`QAOA p1/p2 mask match`, référence 1,000) sont
donc **insensibles** à ce choix. Seules les **trois lignes T11b** en
dépendent. Le master table reste à **180 / OK 164 / DIFF 16 / MISSING 0**.

**Conséquence secondaire, mesurée, non numérotée.** `aggregate_master_table.rows_t11b`
épingle `mean variational progress` à `0,0854` avec la tolérance globale
`TOL = 0,002`. Les trois répétitions du bras warm rendent **0,1034 / 0,0850 /
0,0859** — une dispersion de **0,018, soit 9× la tolérance**. Cette ligne ne
peut pas discriminer à sa propre précision : son statut OK/DIFF est décidé
par le tirage. C'est le piège du seuil calibré sur une grandeur non
reproductible que `VIGIL.md` décrit ; la règle y répond « changer de
**grandeur**, pas de seuil ». Non fait : choisir la grandeur de remplacement
est une décision, pas une correction mécanique.

**Où on en est — trois options, aucune appliquée.**

1. **Laisser tel quel, en le nommant.** Le schedule reste constant, T11b garde
   ses nombres, et la lecture publiée est requalifiée : elle décrit le circuit
   *partant d'un schedule fixe*, pas le circuit en général. Coût : la phrase
   « l'objectif déclaré n'est pas l'objectif optimisé » perd sa portée
   générale.
2. **Rejouer T11b sur la rampe `E_max`**, c'est-à-dire l'initialisation que le
   pipeline déployé emprunte réellement. Déplace les trois lignes T11b
   (+0,0854 → ≈ +0,186 ; −0,172 → ≈ 0). C'est la seule option qui fait dire à
   T11b ce que son titre annonce — « où se situe la décision réellement prise
   par le pipeline ».
3. **Rendre le warm start réellement dérivé du score**, ce que son nom promet.
   Nouvelle mesure à faire de bout en bout ; ni les nombres de l'option 1 ni
   ceux de l'option 2 ne s'y transportent.

Rien n'est appliqué : les trois déplacent ou requalifient un nombre publié, et
`VIGIL.md` réserve cela à USER. Le code porte désormais la déviation là où
elle vit, et un test l'épingle.

```bash
pytest tests/study/test_warm_start_is_constant.py     # 10 tests
python study/common/aggregate_master_table.py         # 180 / 164 / 16 / 0
```

---

## D-50 — le verdict imprimé de T11b bascule d'une exécution à l'autre de la même commande

**Où ça bloque.** `h0_qaoa_displacement.main()` imprime l'une de **deux
phrases opposées** selon que `|progress moyen| < 0,1` :

```python
print("\n  READING: " + (
    "the circuit stays at the classical encoding; the deployed decision "
    "is not a minimiser of its declared cost."
    if abs(prog_all) < 0.1 else
    "the circuit moves substantially toward its own optimum."))
```

C'est la conclusion de T11b, celle qui alimente le verdict **RÉFUTÉ** de
`h0_selection` dans `CLAUDE.md`. Elle repose sur un seuil codé en dur, sans
provenance, appliqué à une grandeur **non reproductible à cette précision**.

**Comment on est tombé dessus.** En relisant `h0_qaoa_displacement.py` en
entier après D-48. Les trois répétitions du bras publié avaient déjà été
mesurées ; il a suffi de les comparer au seuil.

**Ce qui est établi.** Configuration publiée (N=256, dim=2, mapper v2,
`k_opt=100`, `shots=4096`, `--n-snaps 2`, reps 1–4), trois exécutions de la
**même** commande, bras warm — celui du code :

| répétition | progression moyenne | `abs(·) < 0,1` | phrase imprimée |
|---|---|---|---|
| 1 | **0,1034** | **non** | *« the circuit moves substantially toward its own optimum »* |
| 2 | 0,0850 | oui | *« the circuit stays at the classical encoding »* |
| 3 | 0,0859 | oui | *« the circuit stays at the classical encoding »* |

**Une exécution sur trois imprime la conclusion inverse.** La valeur publiée,
0,0854, est à **0,0146** du seuil, pour une dispersion inter-exécutions
mesurée à **0,018** — plus large que la marge. Le verdict est décidé par le
tirage, pas par le code.

Le changement d'initialisation de D-48 le fait basculer *systématiquement* :
bras cold, moyenne **+0,186**, la phrase devient « moves substantially » aux
trois répétitions. Les deux défauts se composent, ils ne se confondent pas :
**D-50 tient même si D-48 est tranché par l'option 1** (ne rien changer).

**Ce que les garde-fous existants ne couvrent pas.**
`check_expected_behaviour` vérifie deux choses réelles — la fraction
d'instantanés à progression indéfinie (`MAX_FRAC_UNDEFINED`) et le nombre de
paires pour la pente (`MIN_PAIRED`) — mais **rien** sur la distance du
résultat au seuil qui décide de la phrase. Le script sort donc en `[ACCEPTANCE]
… nombres publiables` en imprimant l'une ou l'autre conclusion.

`aggregate_master_table.rows_t11b` épingle par ailleurs cette même moyenne à
`0,0854` avec `TOL = 0,002`, soit **9× plus serré** que la dispersion mesurée :
le statut OK/DIFF de cette ligne se joue aussi au tirage. C'est une seconde
face du même problème, pas un défaut distinct.

**Où on en est — non corrigé, décision requise.** `VIGIL.md` est explicite :
quand une grandeur s'avère non reproductible à la précision d'un seuil, on
change de **grandeur**, pas de seuil. Retoucher 0,1 ou `TOL` ferait passer
la suite sans rien mesurer de plus. Trois directions, aucune appliquée :

1. **Répéter et publier une dispersion.** N exécutions, moyenne ± écart-type,
   et le verdict n'est imprimé que si l'intervalle **entier** tombe d'un
   côté du seuil ; sinon la phrase dit qu'on ne tranche pas. C'est la seule
   option qui garde la grandeur actuelle.
2. **Changer de grandeur** pour une déterministe. La pente appariée
   `slope_paired` est déjà calculée et déjà appariée par instantané ; sa
   dispersion mesurée est **0,008** sur trois répétitions (−0,1202 / −0,1143 /
   −0,1124), contre 0,018 pour la moyenne. Elle discrimine mieux — mais elle
   dépend de l'initialisation exactement comme D-48 le décrit.
3. **Retirer la phrase.** Publier les nombres et leur dispersion, laisser la
   lecture au texte. Le moins coûteux, et cohérent avec ce que le module
   annonce déjà en tête (« une progression qui n'augmente pas avec la
   profondeur est **rapportée comme telle** »).

Aucune n'est appliquée : toutes changent ce que le script publie.

```bash
python study/h0_selection/h0_qaoa_displacement.py --N 256 --dim 2 --n-snaps 2
# a relancer 3 fois : la ligne READING n'est pas stable
```

---



## D-98 — le « contrôle négatif » de la figure 9 ne peut pas rendre de faux positif

**Où ça bloque.** `figures/v1_legacy/fig9_synthetic_unit_tests.py` construit
quatre motifs synthétiques, dont le quatrième est annoncé par sa propre
docstring comme *« Uniform noise : negative control → false positive rate »*.
Il ne borne aucun taux de faux positifs. Tant que la référence n'est pas
tranchée, la 4e ligne de la figure ne s'interprète pas — et ses barres se
lisent à côté de celles des trois lignes à signal, qui, elles, sont valides.

**Comment on est tombé dessus.** Question 2 de `VIGIL.md` — lire la docstring
comme un contrat : `pixel_prf` promet une précision/rappel « against GT >
mean ». La promesse du motif 4 (« taux de faux positifs ») et celle de la
métrique (« référence = la moyenne du champ lui-même ») ne peuvent pas être
vraies en même temps.

**Ce qui est établi.**

`needs = gt > gt.mean()` est une référence **relative au champ mesuré**. Elle
ne porte aucune information absolue : `gt` multiplié par 1000, ou décalé d'une
constante, laisse `needs` **bit-à-bit identique** (vérifié sans solveur ni
tirage).

Sur le champ sans anomalie du contrôle négatif — bruit gaussien d'écart-type
0,01 sur un `Bx` uniforme, 50 pas :

| champ | fraction des pixels déclarée « à raffiner » | `gt` max |
|---|---|---|
| **Uniform Noise** — contrôle NÉGATIF, aucune anomalie | **46,6 %** (N=256) / 47,1 % (N=64) | 8,93e−03 |
| Vortex Core — signal | 58,1 % | 3,47e−02 |

Le contrôle négatif déclare donc la moitié du domaine « à raffiner », par
construction : il n'existe pas de faux positif à compter quand la vérité
terrain est définie sur le champ qu'on teste. L'information qui séparerait
les deux lignes existe pourtant — les `gt` max diffèrent d'un facteur ~4 —
mais `pixel_prf` la normalise entièrement.

**Ce qui n'est PAS touché.** Les lignes 1 à 3. Une référence relative y sert à
comparer **deux bras sur le même champ**, où elle est défendable : les deux
voient le même `needs`. Même remarque pour `fig2_early_detection.py:102` et
`fig4_comprehensive_comparison.py:74,84`, qui partagent la référence relative
dans ce rôle-là — aucun d'eux ne s'annonce comme contrôle négatif.

**Où on en est.** Rapport seul, non corrigé : trancher demande de choisir, et
tout choix change les quatre lignes de la figure. Deux directions :

1. **Seuil absolu commun** pour `needs` (par ex. un plancher sur `gt` calibré
   hors du champ testé), ce qui rend le contrôle négatif capable d'échouer ;
2. **Retirer la 4e ligne** et cesser d'annoncer un taux de faux positifs que
   la métrique ne peut pas produire.

La déviation est écrite dans la docstring de `pixel_prf`, là où elle vit, et
un test vérifie qu'elle y reste.

```bash
pytest tests/study/test_fig9_negative_control.py
```

## D-100 — le panneau « Uncertainty w(s) » de la figure 11 n'affiche pas le poids que le hamiltonien applique

**Où ça bloque.** La figure s'intitule *« Hamiltonian Design:
Uncertainty-Weighted ZZ Coupling »* et son 4e panneau annonce `Uncertainty
w(s)`, avec une annotation chiffrée (« X % of cells w > 0.1 »). Tant que la
présentation n'est pas tranchée, ce panneau et son pourcentage ne décrivent
pas le hamiltonien qu'ils illustrent.

**Comment on est tombé dessus.** Question 4 de `VIGIL.md` — deux chemins
censés coïncider. `fig11_hamiltonian_design.py:102` recalcule le poids
localement à partir du score **par cellule** ; `HamiltParams.py:469-473`, le
seul endroit où ce poids est réellement appliqué, le calcule sur le score
**moyenné par arête** (`0.5 * (s + roll(s, -1, axis))`) et en produit **deux**
champs distincts, horizontal et vertical.

**Ce qui est établi.** Part des cellules à `w > 0,1`, mesurée à N=64 avec les
paramètres déployés (`sigma = 0,0500`, `threshold_amr = 0,3044`) :

| scénario | panneau D (par cellule) | ce qu'applique le mappeur (arêtes h / v) | écart |
|---|---|---|---|
| Kelvin-Helmholtz | 9,89 % | 10,40 % / 9,91 % | +3 % |
| **Harris Tearing** | **1,27 %** | **5,52 %** / 1,27 % | **+167 %** |

Sur la nappe de tearing, les arêtes **horizontales** sont **4,3×** plus
actives que ce que le panneau affiche. Et l'anisotropie que le hamiltonien
voit — `h` ≠ `v`, d'un facteur 4,3 ici — n'apparaît pas du tout sur un
panneau unique. Le champ qui sépare est celui dont le score varie le long
d'**un seul** axe ; sur un score constant, les trois versions coïncident.

**Ce qui n'est PAS touché.** L'annotation « ZZ reduced by X % » du panneau C
vient bien du mappeur réel (`compute_zz_maps` appelle
`compute_coefficients`) : elle n'est pas concernée.

**Où on en est.** Rapport seul. Afficher `w_h`, `w_v`, leur moyenne ou leur
max est un **choix de présentation** — et montrer deux cartes au lieu d'une
change la mise en page de la figure. À trancher. La déviation est écrite à
côté du calcul concerné, et un test vérifie qu'elle y reste.

```bash
pytest tests/study/test_fig11_uncertainty_weight.py
```

## D-135 — l'accord des deux chemins de score n'est gardé que par une chaîne, dans `src/`

**Où ça bloque.** `pipeline()` calcule le score de deux façons selon la
branche prise. D-5 avait corrigé le chemin de divergence, qui utilisait une
L2 **non pondérée** là où `score()` pondère par la carte d'instabilité :
deux formules partaient vers Optuna sous la même clé `combined`, écart
mesuré **1,8 %** sur un champ à nappe de courant. Rien ne garde aujourd'hui
cette correction sinon une recherche de texte.

**Comment on est tombé dessus.** Sondage `.read()` de `COUVERTURE.md`, site
`tests/mapping/test_objective_and_estimators_analytic.py:574`.
`test_both_scoring_paths_now_use_the_same_formula` fait
`assert "field_errors[var] = weighted_relative_error(" in _PIPELINE_SRC`.
La vérification numérique qui suit dans le même test compare `score()` à
`weighted_relative_error` — elle mesure le chemin **`score()`**, jamais le
chemin de divergence, qui vit dans le corps de `pipeline()`.

**Ce qui est établi.** Mutation A′, `src/pipeline.py:675`, reste du fichier
intact : l'appel partagé est **laissé en place** et son résultat réécrit une
ligne plus bas par `float(np.sqrt(np.mean((arr_q - arr_r)**2)))` — la L2 non
pondérée de D-5, exactement.

| | |
|---|---|
| chaîne cherchée par le test | **toujours présente** |
| `pytest tests/mapping/test_objective_and_estimators_analytic.py` | **46 passed** |
| ce que la mutation rétablit | le défaut D-5, sur le chemin de divergence |

**Où on en est. Rapport seul, non corrigé — et c'est délibéré.**
`src/pipeline.py` est le chemin scientifique **déployé** : c'est le fichier
qui a produit les nombres publiés. Les corrections de cette famille (D-127,
D-129 à D-134) sont toutes entièrement dans les tests parce que le code y
était juste ; ici aussi le code est juste, mais le rendre *mesurable*
demanderait d'extraire la boucle par champ de `pipeline()` en une fonction
appelable — le geste fait ailleurs (`interpretation_message` D-46,
`floor_ratios` D-89, `apply_leak_free_threshold` D-134), mais jamais encore
dans `src/`. La fiche du dépôt est explicite : dans le doute entre défaut et
choix de conception, **mesurer, documenter, ne pas corriger, demander**.

Deux directions, à trancher :

1. **extraire** la boucle par champ en `field_errors_for(q_fluxes,
   ref_fluxes, variables)` dans `src/pipeline.py`, corps inchangé, et
   l'éprouver contre `score()` sur un champ à nappe de courant — le seul
   qui SÉPARE les deux formules, l'écart y valant 1,8 % ;
2. **atteindre la branche par `pipeline()` lui-même**, avec une entrée qui
   force la divergence d'un champ, sans toucher à `src/` — plus fidèle,
   mais l'entrée reste à construire et son coût n'est pas mesuré.

**La direction 2 est FAITE — le trou de garde est fermé, `src/` intact.**
L'entrée qui restait à construire : rendre `MHDSolver.is_diverged` vrai au
n-ième appel. Le vrai bloc `--- Divergence guard ---` s'exécute alors, sur
de vrais champs, en **~1,5 s**. Le champ qui SÉPARE une L2 pondérée d'une
L2 nue n'est pas la nappe de courant mais la **carte de poids elle-même** :
on la remplace par une carte uniforme et on exige que le résultat bouge.

| | `phys_score` |
|---|---|
| carte d'instabilité réelle | **3,27453290e−02** |
| carte uniforme | 3,31748727e−02 |
| écart | **−1,2948 %** |

Même ordre que les 1,8 % que D-5 avait mesurés. Rejoué sous la **mutation
A′** de cette entrée — la L2 non pondérée réintroduite juste après l'appel
partagé, la chaîne cherchée laissée en place :

| | après A′ |
|---|---|
| `test_objective_and_estimators_analytic.py` (l'ancien garde) | **46 passed** — le faux vert de D-135, inchangé |
| `test_intermediate_score_time_alignment.py` (le nouveau) | **1 failed** |

**Ce qui reste ouvert : la direction 1 seule.** Extraire la boucle par
champ de `pipeline()` reste une modification de `src/`, donc une décision.
Ce qui change, c'est qu'elle n'est plus urgente : le défaut ne peut plus
revenir en silence.

```bash
pytest tests/mapping/test_objective_and_estimators_analytic.py -q -k both_scoring_paths
pytest tests/pipeline/test_intermediate_score_time_alignment.py -q -k weights_by_the_instability_map
```

## Ajouter une entrée

Un défaut n'entre ici que s'il **bloque**. Une fois corrigé, il sort d'ici et
entre dans `RESULTS.md`.

Chaque entrée porte : **où ça bloque**, **comment on est tombé dessus**, **ce
qui est établi** (chiffré), **où on en est**, et la commande qui vérifie
l'état.

Un défaut sans mesure est une suspicion. Un défaut sans commande de
vérification n'a pas sa place ici.


---

## D-132 — le bras QAOA ne classe plus, sur une partie de l'espace

**Où ça bloque.** Une campagne Optuna explore l'espace d'hyperparamètres.
Si le bras quantique n'y porte aucun signal sur une partie de cet espace,
la campagne y optimise du bruit — et elle y passera du temps de calcul
payé. À trancher **avant** de louer des cœurs.

**Comment on est tombé dessus.** Passage de recette complet après l'ajout
du 9ᵉ paramètre : `2006 passed, 3 failed` en 1 h 32. Les trois échecs sont
dans la suite QAOA.

**Ce qui est établi.**

| | |
|---|---|
| `test_hyperparameter_sweep` | **échoue**, reproductible (3 exécutions) |
| `test_noise_robustness` | **échoue**, reproductible (2 exécutions) |
| `test_the_ranking_survives_the_sampling` | **passe** à la réexécution — celui-là est bien un tirage |

L'assertion qui tombe dans le balayage n'est **pas** le plafond
`MAX_CLEAN_ADVANTAGE`, qui passe. C'est la garde de vivacité une ligne
plus bas :

```
AssertionError: correlation de rang QAOA/verite negative (-0.467) :
le bras ne classe plus rien, le plafond ci-dessus ne prouve alors rien
assert -0.467 > 0.0
  where -0.467 = min([-0.467, -0.467, 0.75, 0.95, -0.467, 0.933, ...])
```

Trois des douze combinaisons d'hyperparamètres donnent une corrélation de
rang **négative** avec la vérité terrain ; d'autres donnent +0,95. Le
second échec dit la même chose autrement :

```
Orszag-Tang: without noise the QAOA arm is expected to lose by more than
0.09 captured fraction, measured gap +0.0000
```

Un écart **exactement nul** : les deux bras sélectionnent les mêmes blocs.
C'est ce qu'on verrait si l'hamiltonien était devenu inerte dans cette
configuration et que QAOA retombait sur le biais classique.

**Ce n'est pas l'ajout de `relative_percentile`.** Le chemin par défaut est
un no-op **bit-à-bit**, épinglé par
`tests/pipeline/test_relative_percentile_is_trainable.py::test_le_defaut_est_un_NO_OP_bit_a_bit`.

**Bisection — CLOS. Le coupable est `6ecaecf` (D-25).**

`e4d6bbc` est son **parent direct** et passe ; `6ecaecf` échoue. Un seul
commit les sépare : c'est établi, pas inféré.

| commit | verdict | durée |
|---|---|---|
| `d978539` — naissance de la garde | passe | 46 min |
| `d212e54` — D-8, hamiltonien borné | passe | 10 min 43 |
| `6de5fbf` — D-24, `PROJECT_RHS` à False | passe | 9 min 15 |
| `e4d6bbc` — projection indépendante de la taille | passe | 9 min 06 |
| **`6ecaecf` — D-25, projection spectrale** | **ÉCHOUE** (−0,467) | **2 min 20** |
| `854ba24` — pénalité de divergence partagée | échoue | 10 min 32 |
| `91951df` — D-37 / D-38 | *(après un rouge : non concluant)* | — |
| `403240b` — avant les corrections de coefficients | échoue | 2 min 55 |
| `5bdcf80` — après elles | échoue | 13 min |

Diff de `6ecaecf` : **39 lignes dans `src/Simulation/solver.py`**, rien
d'autre.

**Trois hypothèses successives réfutées par cette bisection**, toutes
plausibles et toutes fausses : **D-1** (bascule de convention du
rotationnel, `bb6a387`), **D-8** (coefficients exactement nuls,
`d212e54`), **D-37** (biais Z et couplages sur deux grilles, `91951df` —
écarté parce qu'il vient *après* `854ba24`, déjà rouge).

**Ce que D-25 a corrigé** : *« la projection spectrale abîmait un champ
déjà solénoïdal »*. Le commit ajoute un contrôle **négatif** — vérifier
que le second membre de `v` ne préserve pas la divergence — précisément
pour qu'on ne puisse pas croire qu'aucune projection n'est nécessaire.

**Ce n'est donc pas une régression à défaire.** D-25 est une correction
juste, mesurée, avec son contrôle négatif. Avant elle, le bras QAOA
classait « bien » parce qu'il lisait des champs **abîmés par une
projection fautive** : l'ordre des blocs qu'il produisait tenait à un
artefact numérique, pas à la physique.

**Ce que cela ajoute à la figure d'ensemble.** C'est le quatrième point,
et il en change le statut :

| | |
|---|---|
| D-47 | le fondamental exact = « tout raffiner », 40/40 |
| D-53 | optimum atteint 0,062–0,156 contre 1,000 exigé, **sous** le classique |
| ρ(E_gap, F1) | +0,870 — mieux résoudre H dégrade la décision |
| **D-132 / D-25** | le classement du bras reposait sur une projection fautive |

Chaque correction qui retire un artefact rend le bras quantique **moins**
bon. Ce n'est plus « quatre mesures concordantes » mais une **direction
systématique**, ce qui est un argument plus fort.

**La nuance à ne pas perdre** : les corrélations restent hétérogènes —
−0,467 sur 3 combinaisons, +0,95 sur d'autres. Le bras n'est pas mort
partout, il est devenu **instable selon les hyperparamètres**. C'est
exactement ce que la réoptimisation arbitre : **D-132 ne bloque donc plus
la campagne**, il en change la lecture attendue.

**Conséquence sur les deux tests rouges.** Ils épinglent l'état d'avant
D-25 : ce sont des **seuils périmés**, le code a légitimement changé sous
eux. Ils se remesurent après la campagne, avec les autres. Ils ne se
retouchent pas et ne se suppriment pas.

**Ce qu'il faut noter sur ces deux tests.** Tous deux encodent d'anciens
**résultats** comme assertions — « QAOA perd d'au moins 0,09 », « QAOA
classe positivement ». Un changement de physique s'y manifeste donc en
rouge, pas en résultat. Ne pas déplacer les seuils avant de savoir ce que
la physique a fait : ce serait effacer la mesure au lieu de la lire.

**Ce que ça n'invalide pas.** `preflight_coefficients.py` passe 5/5, mais
il vérifie que les coefficients corrèlent avec le **besoin de
raffinement** — pas que le bras quantique classe mieux que le hasard.
Deux affirmations distinctes ; seule la seconde échoue.

---

## D-141 — la porte de la campagne est franchie plus haut par la baseline que par le coefficient

**Rapport seul. Décision requise, rien n'est corrigé.**

**Où ça bloque.** `study/common/preflight_coefficients.py` est la porte de
la réoptimisation : il imprime « *les coefficients font leur travail.
Campagne possible.* » avant ~224 h CPU. Son 4ᵉ contrôle, `pertinence`,
annonce *« le coefficient corrèle avec l'erreur RÉELLE DNS-vs-grossier,
rho = 0,798 »* et accepte dès `rho > 0,6`. Ce seuil ne sépare pas : des
grandeurs qui ne portent **aucun** coefficient et **aucun**
hyperparamètre le franchissent sur le même état DNS — dont le **score
classique**, la baseline même que le bras quantique doit battre, et il le
franchit **plus haut** que le coefficient.

**Comment on est tombé dessus.** Question 3 de `VIGIL.md` appliquée à un
contrôle plutôt qu'à une fonction : *consomme-t-il ce que son nom
annonce ?* Puis la règle du champ d'essai qui **sépare** — sur quelle
entrée les deux hypothèses (« le coefficient porte l'information » /
« quelque chose se concentre dans la nappe ») donnent-elles des réponses
différentes ?

**Ce qui est établi.** Même état, mêmes 8×8 blocs, même erreur de
référence (DNS `N=128` contre run grossier `N=32`, 200 pas,
`harris_tearing`), même opérateur — la réplique est vérifiée identique au
contrôle à **1e−12** avant toute comparaison :

| grandeur | porte un coefficient ? | rho | franchit 0,6 ? |
|---|---|---|---|
| **score classique** — la baseline | **non** | **+0,8137** | **oui** |
| `K_plaquettes` — ce que le contrôle regarde | oui | +0,7977 | oui |
| \|Jz\| — courant | non | +0,7429 | oui |
| \|v\| — module de la vitesse | non | +0,7247 | oui |
| \|∇\|B\|\| — gradient brut | non | +0,6764 | oui |
| `K_xpoint` — un coefficient à part entière | oui | +0,4345 | **non** |
| `H_edges[0]` | oui | −0,6288 | non |
| bruit blanc — contrôle négatif | non | −0,0401 | non |

Mesure déterministe : deux exécutions identiques au dernier chiffre.

**Trois lectures, et seule la troisième est en cause.**

1. Le contrôle **n'est pas vide** : le bruit blanc le rate. Il mesure
   quelque chose de réel.
2. La **fonction honore sa docstring** : elle promet que le coefficient
   corrèle avec l'erreur, et il corrèle. Ce n'est pas un défaut de code.
3. C'est le **verdict** qui sur-conclut. « Les coefficients font leur
   travail » ne se déduit pas d'un seuil que la baseline franchit mieux,
   et que trois champs nus franchissent aussi. Le contrôle ne distingue
   pas *« le coefficient porte l'information »* de *« quelque chose se
   concentre dans la nappe »*.

Deux précisions qui limitent la portée, écrites pour ne pas la surestimer :
`K_plaquettes` est **calculé à partir** du score classique
(`compute_coefficients(sim, score, …)`), donc sa corrélation en hérite en
partie — ce n'est pas un hasard, c'est une dépendance. Et le contrôle ne
regarde qu'**un** des quatre canaux : `K_xpoint` et `H_edges` ne
franchiraient pas le seuil.

**Ce que ça ne dit pas.** Rien ici ne dit que la campagne est inutile, ni
qu'un nombre publié est faux. Aucun nombre publié ne dépend de ce module :
c'est un contrôle avant vol, pas une mesure. D-132 notait déjà que le
preflight « ne vérifie pas que le bras quantique classe mieux que le
hasard » ; D-141 est un cran en deçà — il ne vérifie pas non plus que le
coefficient fasse mieux que la baseline sur ce que le contrôle mesure.

**Pourquoi rien n'est corrigé.** Le correctif naturel — exiger
`rho(coefficient) > rho(score classique)` — **changerait le verdict de la
porte**, aujourd'hui `OK`, en `ÉCHEC` : +0,7977 contre +0,8137. Faire
passer au rouge la porte d'une campagne de 224 h CPU est une décision, pas
une correction mécanique. `VIGIL.md` : *mesurer, documenter, ne pas
corriger, demander.*

**Trois options, aucune appliquée.**

1. **Requalifier le libellé, ne rien changer au code.** Le contrôle dit
   alors ce qu'il mesure : « le coefficient se concentre là où le run
   grossier se trompe » — vrai, vérifié, et sans prétention de
   discrimination. Coût : le verdict « les coefficients font leur travail »
   doit être réécrit.
2. **Ajouter un critère relatif** (`rho(coef) > rho(score classique)`, ou
   un écart minimal). Le plus informatif, et la porte passe au rouge
   aujourd'hui — c'est précisément ce qui en fait une décision.
3. **Changer de grandeur de référence.** L'erreur est ici une différence
   de moyennes par bloc ; la structure sous-bloc, celle que le
   raffinement récupère, n'y entre pas. Une erreur au sens de
   `patch_l2_errors` mesurerait autre chose. Nouvelle mesure de bout en
   bout ; ni les nombres de 1 ni ceux de 2 ne s'y transportent.

**Portée mesurée : lequel des cinq contrôles voit la structure ?** On mute
la **sortie** de `PhysicalMapper.compute_coefficients` — jamais le contrôle
— et on regarde lesquels mordent. `coincidence` est exclu : il n'appelle
pas le mappeur, il compare deux chemins de calcul d'énergie sur des
coefficients tirés au hasard. Deux exécutions, matrice identique.

| mutation | `specificite` | `equilibre` | `vivant` | `pertinence` |
|---|---|---|---|---|
| aucune (référence) | OK | OK | OK | OK |
| **axes transposés** | OK | OK | OK | **ÉCHEC** |
| `K_plaq` ↔ `K_xpoint` | ÉCHEC | ÉCHEC | OK | ÉCHEC |
| tout ×1000 | OK | OK | OK | OK |
| `K_xpoint` mis à zéro | OK | OK | ÉCHEC | OK |
| coefficient = bruit | ÉCHEC | OK | OK | ÉCHEC |
| **mélange spatial** (même distribution) | OK | OK | OK | **ÉCHEC** |

**`pertinence` est le seul des quatre à voir OÙ le coefficient met sa
masse.** Les trois autres sont des contrôles d'amplitude — et leurs
docstrings ne promettent rien d'autre, donc ce n'est pas un défaut de leur
part. Ce qui compte pour D-141 est la **conjonction** : le seul contrôle
sensible à la structure est aussi celui que la baseline franchit mieux. Une
transposition d'axes — la famille de défauts la plus fréquente de ce dépôt,
D-1, D-17, T31 — n'est vue que par lui.

**`tout ×1000` passe les cinq, et c'est juste** : l'état fondamental d'un
Ising est invariant par mise à l'échelle positive uniforme des couplages.
Ce n'est pas un trou, c'est une symétrie — noté pour qu'il ne soit pas
« corrigé » par erreur.

```bash
python study/common/preflight_coefficients.py          # 5/5, rho = 0.798
pytest tests/study/test_preflight_pertinence_separates.py -m "not slow"
                                                        # 6 passed, ~29 s
pytest tests/study/test_preflight_pertinence_separates.py -m slow
                                                        # 1 passed, ~73 s
```

Le second est un test de **déviation** : il échoue le jour où le contrôle
gagne un critère de discrimination — c'est-à-dire le jour où D-141 est
tranché, et où il doit être relu. Mesuré : seuil porté à 0,85 → **2 failed**.

---

## D-143 — deux des trois notations de `pipeline()` lisent la référence un cran trop tôt

**Rapport seul. Décision requise, rien n'est corrigé.**

### ⚠️ Portée corrigée après première publication — lire ceci d'abord

La première rédaction de cette entrée annonçait que le décalage touchait
« le chemin qui décide quels essais survivent à ~224 h CPU ». **C'est faux
pour le premier site, et je le corrige avant qu'il ne serve.**

Mesuré, en cherchant les appelants plutôt qu'en le supposant : **les trois
appelants de production de `pipeline()` passent `trial=None`.**

| appelant | `trial` |
|---|---|
| `src/train_hyperparams.py:695` — la campagne Optuna | **`None`**, sous le commentaire *« elagage gere ici, pas dedans »* |
| `study/closed_loop/closed_loop_campaign.py:220` | **`None`** |
| `src/pipeline.py:271` — la CLI | **`None`** |

L'élagage de la campagne vit à `train_hyperparams.py:752`, **par scénario**
(`trial.report(total / (i + 1), step=i)`), et ne passe pas par le bloc
intra-run. Donc :

| site | état |
|---|---|
| **1. score intermédiaire** (`:718`) | **inatteignable aujourd'hui** — aucun appelant de production ne passe de `trial`. C'est un **piège armé**, pas un défaut vivant : la signature invite à passer un essai, et quiconque le fera recevra un signal d'élagage dominé par l'évolution propre de la DNS |
| **2. chemin de divergence** (`:639`) | **VIVANT, sur le chemin de la campagne.** `pipeline()` y passe dès qu'un essai diverge, et le score partiel qu'il rend **est** l'objectif que la campagne remonte à Optuna |

C'est le site 2 qui compte, et il est arrivé second dans mon enquête. Les
mesures ci-dessous restent valables telles quelles — elles portent sur la
même arithmétique aux deux sites — mais la **conséquence** est celle du
site 2 : un essai qui diverge est noté contre une référence périmée d'un
pas, et un bras parfait y prend **3,27e−02** au lieu de zéro.

Ce que cela ne change pas : aucun nombre publié n'en dépend, ni par le
site 1 (mort) ni par le site 2 (les 180 lignes du master table viennent de
runs qui n'ont pas divergé). Ce que cela change : la décision porte
d'abord sur le site 2, et le site 1 peut être traité comme de l'hygiène.

**Où ça bloque.** `src/pipeline.py:715` calcule, à chaque pas hybride, le
score rapporté à Optuna pour l'élagage :

```python
        t_current += dt
        step += 1                                    # ← le compteur avance
        step_simulated += 1
        ...
        if trial is not None and did_hybrid and steps_hybrid_count > 1:
            dns_entry = dns_trace.get(step - 1, {}) if dns_presence else {}
```

À ce point, les deux bras ont été intégrés jusqu'à `t_step` et `step` a
déjà été incrémenté ; l'instantané DNS qui leur correspond est donc
`dns_trace[step]`. Le code lit `dns_trace[step - 1]` : l'état DNS du
**début** du pas qu'on vient d'intégrer. La valeur rapportée à
`trial.report(...)` puis à `should_prune()` mesure donc, pour l'essentiel,
l'évolution propre de la DNS sur un pas de temps — une grandeur qui ne
dépend d'aucun hyperparamètre de l'essai.

**Le producteur, lui, documente et TESTE la convention que les
consommateurs lisent de travers.** La docstring de `precompute_dns` écrit
noir sur blanc : *« `dns_trace[k]['fluxes']` l'état AVANT le pas k, pour
tout k sauf le dernier »*, et
`tests/solver/test_precompute_dns_contracts.py::test_intermediate_snapshots_hold_the_state_before_their_step`
la fige. Ce n'est donc pas une convention ambiguë sur laquelle deux
lectures se vaudraient : l'entrée `k` est l'état à `t_k`, le bras après
`step += 1` est à `t_step`, et l'index aligné est `step`. Le producteur
est juste ; ce sont ses deux consommateurs qui lisent un cran trop tôt.

**Comment on est tombé dessus.** Question 4 de `VIGIL.md` : deux chemins
censés coïncider coïncident-ils encore ? Le score **final**, vingt lignes
plus bas, choisit son index autrement :

```python
        last_step = step if step in dns_trace else step - 1
```

Il préfère `step` et ne retombe sur `step - 1` qu'à défaut. Deux lectures
du même `dns_trace`, dans la même fonction, avec deux conventions
d'alignement.

**Ce qui est établi.** Champ d'essai choisi pour **séparer** : une
configuration où le bras reproduit la DNS *exactement*
(`patch_ratio = 1,0`, donc raffinement total), si bien que toute erreur
rapportée non nulle ne peut venir que de la référence. Kelvin-Helmholtz,
`N = 32`, `T_START = 0,9`, `T_MAX = 1,2`, `HYBRID_DT = 0,02`,
`DT = 1e-3`, `max_depth_override = 1`, `classical_only`, trace DNS de 25
pas / 7 instantanés, départ à chaud au pas 19. Deux exécutions identiques
au dernier chiffre.

| rapport | `phys_score` rapporté | `phys_score` aligné | rapport |
|---|---|---|---|
| step 21 | **3,274533e−02** | 6,142937e−16 | 5,33e+13 |
| step 22 | **3,197032e−02** | 1,201023e−15 | 2,66e+13 |
| step 23 | **3,127157e−02** | 1,885542e−15 | 1,66e+13 |
| step 25 (dernier) | 3,055743e−15 | — | **juste** |

Le score **final** du même run vaut `phys_score = 3,055743e−15` : le bras
est exact, et les trois premiers rapports intermédiaires annoncent une
erreur **treize ordres de grandeur** trop grande. Pour situer, l'évolution
propre de la DNS entre deux instantanés consécutifs, mesurée avec
l'opérateur assorti (`score` lui-même), vaut **3,457654e−02** — c'est bien
elle que les rapports mesurent.

**Le décalage vaut exactement un cran**, prouvé par identité des tableaux
et non par ressemblance des nombres :

| rapport | réf. utilisée vs `trace[k−1]` | réf. utilisée vs `trace[k]` | bras vs `trace[k]` |
|---|---|---|---|
| 21 | **0,000e+00** | 1,789e−03 | 4,441e−16 |
| 22 | **0,000e+00** | 1,802e−03 | 6,661e−16 |
| 23 | **0,000e+00** | 1,825e−03 | 8,882e−16 |

La référence consommée est **bit-à-bit** `dns_trace[step-1]['fluxes']`, et
le bras est **bit-à-bit** `dns_trace[step]['fluxes']` à l'epsilon machine.

**Pourquoi le dernier rapport, lui, est juste.** `pre_compute_dns.py:126`
réécrit `dns_trace[step-1]['fluxes']` après la boucle avec l'état de
**fin** de run (« AJOUT CRITIQUE »). Cette double convention est
**voulue, écrite et testée** — elle sert le score final, qui compare les
deux bras une fois tous les pas faits, et
`test_the_last_snapshot_holds_the_state_after_its_step` la fige. Ne pas
la « corriger ». Ce qui est accidentel, c'est que le **dernier rapport
intermédiaire** en bénéficie : il lit cette entrée-là et tombe donc
aligné, sans que rien ne l'ait voulu. Un point sur cinq.

**Le pas 24 est écarté de la mesure, et non tu.** C'est l'entrée réécrite :
le bras y diffère de `trace[24]` de **6,964e−04**, donc `trace[24]` n'est
pas une référence propre à cet instant et la colonne « aligné » n'y
voudrait rien dire. Trois rapports propres suffisent.

**Ce que ça ne dit pas.** Aucun nombre publié n'en dépend : les 180 lignes
du master table viennent du score **final**, dont l'alignement est juste.
Et surtout — **on n'a pas mesuré si le classement des essais survit**. Le
terme parasite est commun à tous les essais d'une même trace DNS, donc un
élagueur qui compare un essai à la médiane des autres au même pas peut le
voir s'annuler en grande partie. Trancher demanderait deux essais
d'hyperparamètres différents avec un bras **non exact** ; ce n'est pas
fait, et rien ici ne permet de conclure que la campagne est faussée. Ce
qui est établi est plus étroit et suffit à entrer ici : **le signal
d'élagage est dominé par une grandeur qui n'appartient pas à l'essai.**

La configuration de campagne (`N = 256`, `HYBRID_DT = 0,10`) n'est pas
rejouée — coût. La mesure porte sur `N = 32`.

**Un DEUXIÈME site, trouvé après coup.** Le chemin de divergence
(`pipeline.py:639`) écrit `last_ok = step - 1` puis remonte tant qu'il ne
trouve pas de `'fluxes'`. Le bras y est à `t_step` : la recherche devrait
partir de `step`. Mesuré sur la même configuration, en forçant
l'avortement par `MHDSolver.is_diverged` :

| avortement | référence consommée | l'instantané aligné existe-t-il ? |
|---|---|---|
| bras à `trace[21]` | `trace[20]` | **oui** — la trace en porte à 18…24 |
| bras à `trace[22]` | `trace[21]` | oui |
| bras à `trace[23]` | `trace[22]` | oui |

Identification par identité de tableaux. Le `phys_score` que ce chemin
rend au premier avortement vaut **3,27453290e−02** — le **même nombre**
que le rapport intermédiaire du pas 21, ce qui est attendu : c'est la même
comparaison décalée.

**Deux des trois sites de notation partent donc de `step - 1`** ; seul le
score final préfère `step`. Sur ce second site le correctif est **sans
risque** — la boucle de remontée existe déjà et gère l'absence
d'instantané, donc partir de `step` ne peut que trouver une référence
meilleure ou égale. Il n'est pas appliqué malgré tout : il change un
nombre que la campagne consomme, et la décision se prend en une fois avec
le premier site.

**Pourquoi le code lit `step - 1`, et pourquoi ce n'est pas une excuse.**
L'instantané DNS est posé **à** la frontière hybride, et la notation a
lieu **après** que le pas franchissant cette frontière a été intégré.
`step - 1` est donc le seul index qui porte des `'fluxes'` : le décalage
n'est pas une faute de frappe, c'est la conséquence d'avoir noté après le
pas alors que la référence est posée avant. Cela explique le défaut ; cela
ne le rend pas inoffensif, puisque la valeur rapportée reste dominée par
un terme étranger à l'essai.

**Pourquoi rien n'est corrigé — mesuré, et non déduit.** Le correctif
tient en un caractère (`step - 1` → `step`), et c'est ce qui le rend
trompeur. Appliqué seul dans la configuration ci-dessus, il rend :

| rapport | avant | après le correctif d'un caractère |
|---|---|---|
| 21, 22, 23 | 0,2857143 + 3,1e−02 de parasite | **0,2857143** — juste |
| 24 | idem | 0,2936544 — **toujours faux**, c'est l'entrée réécrite |
| 25 (dernier) | 0,2857143 — juste | **10,0 = `DIVERGENCE_PENALTY`** |

Le dernier rapport bascule sur la pénalité de divergence d'un run
parfaitement sain. Mécanisme vérifié dans le code : `dns_trace[step]`
n'existe pas au dernier pas, donc `dns_entry` est vide ; et le repli
`elif sim_temoin is not None` est **mort** dès qu'une trace DNS est
fournie — `pipeline.py:327` et `:338` laissent `sim_temoin = None`
précisément dans ce cas (« *No need for `sim_temoin` — `dns_trace` is the
reference* »). Il reste `score_result = None`, donc `nan`, donc
`DIVERGENCE_PENALTY`, rapporté à `should_prune()`.

C'est le piège que `VIGIL.md` appelle *vérifier la portée d'une correction
avant de l'activer* : un correctif juste sur trois pas sur cinq en casse
un cinquième, sur le chemin qui décide quels essais survivent à une
campagne de ~224 h CPU. `VIGIL.md` : *mesurer, documenter, ne pas
corriger, demander.*

**Trois options, aucune appliquée.**

1. **Noter avant d'intégrer.** Déplacer le bloc de notation au-dessus de
   `step_layered`, là où le bras est encore à l'instant de la frontière
   hybride et où l'instantané existe. Aucun index à changer, aucune
   référence à changer de nature — mais le rapport `k` porte alors le pas
   `k−1`, ce qui décale la courbe vue par l'élagueur d'un cran.
2. **Remonter au dernier instantané disponible**, comme le fait déjà le
   chemin de divergence (`while last_ok >= 0 and 'fluxes' not in …`), et
   **rapporter l'écart de temps** avec le score. Garde la DNS pour
   référence, rend le décalage visible au lieu de le taire — mais ne le
   supprime pas.
3. **Poser un instantané DNS au pas qui suit chaque frontière hybride**,
   dans `pre_compute_dns.py`. Supprime le décalage à la racine et rend le
   correctif d'un caractère sûr ; coûte un `get_fluxes()` de plus par
   frontière et change la trace, donc à remesurer de bout en bout.

Le correctif d'un caractère seul n'est **pas** une option : mesuré
ci-dessus, il fabrique un `DIVERGENCE_PENALTY` sur un run sain.

```bash
pytest tests/pipeline/test_intermediate_score_time_alignment.py -q
```

C'est un test de **déviation**, comme ceux de D-141 : il épingle le
décalage mesuré et rougit le jour où D-143 est tranché — c'est-à-dire le
jour où il doit être relu.

## D-158 — l'agrégateur du master table réécrit la table publiée sur une configuration qui ne correspond à rien, et sort avec le code 0

**Rapport seul. Décision requise, rien n'est corrigé** — la correction
changerait le comportement d'une commande qui écrit des artefacts publiés.

**Où ça bloque.** `python study/common/aggregate_master_table.py` est le
**quatrième test de recette** de `CLAUDE.md`, le test de non-régression du
dépôt : il recalcule chaque nombre publié à partir de son artefact.
`CLAUDE.md` écrit, sur ce même paragraphe : *« Un `MISSING` non nul, lui,
est toujours une régression. »*

Donné une taille pour laquelle **aucune campagne n'a jamais tourné**, il
ne trouve rien, **écrit quand même** par-dessus les trois artefacts
publiés, et **sort avec le code 0**.

**Comment on est tombé dessus.** En construisant, pour D-157, une
invocation que chaque module de `study/` **accepte** et à laquelle rien ne
répond. Les trois agrégateurs n'exposent ni `--scenario` ni `--fold` : le
seul sélecteur qu'ils déclarent est la taille, `--N` / `--dim`.

**Mesuré**, même arbre, deux exécutions successives depuis un état propre :

| | `aggregate_master_table.py` (canonique, sans argument) | `--N 7 --dim 99` |
|---|---|---|
| lignes | **180** | **161** |
| OK | **176** | **113** |
| DIFF | **4** | **0** |
| MISSING | **0** | **48** |
| code de sortie | 0 | **0** |
| dernière ligne imprimée | `V4 Task 16 complete.` | `V4 Task 16 complete.` |
| `results/v4_master_table.csv` | réécrit à l'identique | **136 lignes supprimées, 98 ajoutées** |

Les 19 lignes qui **disparaissent** ne sont pas comptées `MISSING` : elles
sortent de la table. Les trois compteurs que `CLAUDE.md` demande de
surveiller bougent tous les trois, et la commande annonce sa réussite.

`results/v4_master.npz` et `results/v4_master_table.md` sont réécrits de
même. Un `git checkout` est la seule chose entre une faute de frappe sur
`--N` et la table de référence du dépôt.

**Ce qui n'est pas en cause.** `--strict` fait ce qu'il annonce : son aide
dit *« exit non-zero on any DIFF (MISSING is tolerated) »*, et tolérer
`MISSING` est un **choix écrit** — une ligne de niveau 3 est `MISSING`
tant que le fold n'a pas tourné. Le désaccord porte sur le fait que
`CLAUDE.md` dit l'inverse (« toujours une régression ») : deux documents,
deux règles. Ce n'est pas au relecteur de trancher.

**Deux autres sites, même forme, mesurés au même moment :**

| module | `--N 7 --dim 99` | ce qu'il écrit |
|---|---|---|
| `aggregate_v3.py` | code **0** | `v3_master_table.csv`, `.md`, `v3_master_N7.npz` |
| `aggregate_v2.py` | code **0** | `SUMMARY_N7_dim99.csv` — **le nom porte la configuration**, donc rien de publié n'est écrasé |

**C'est `aggregate_v2.py` qui donne la réponse la plus simple** : dans le
même dossier, un agrégateur nomme sa sortie par la configuration
demandée, les deux autres écrivent toujours dans le même fichier. Deux
chemins censés coïncider — question 4.

**Un quatrième site, et je le crois sain** : `closed_loop_status.py`
`--folds no_such_fold` sort **0** en imprimant
`no_such_fold  [---]  no-trials-yet`. C'est un rapporteur d'**état** :
« rien n'a tourné » est un état, pas un balayage vide silencieux. Écrit
ici pour que la décision ne se reprenne pas à chaque passe.

**Trois options, à trancher — aucune appliquée.**

1. **Refuser d'écrire quand rien ne correspond.** Sortie non nulle dès que
   `MISSING == len(rows)` sur une taille demandée explicitement. Le plus
   proche de `CLAUDE.md`, et ça ne touche pas le chemin canonique.
2. **Nommer la sortie par la configuration**, comme `aggregate_v2.py` déjà
   le fait — `v4_master_table_N7_dim99.csv`. Rien de publié n'est plus
   atteignable par accident ; en échange le chemin canonique change de nom
   ou garde un cas particulier.
3. **Ne rien changer au code, corriger `CLAUDE.md`** : y écrire que
   `MISSING` est toléré et que la commande écrit toujours. Le moins cher,
   et le moins protecteur.

**Ce que ça ne dit pas.** Aucun nombre publié n'est faux aujourd'hui : le
chemin canonique rend **180 / 176 / 4 / 0**, mesuré ici, et les artefacts
du dépôt sont ceux-là. Ce qui est en jeu est la **destructibilité** de la
référence, pas son contenu. Le garde de D-157
(`tests/study/test_empty_sweep_never_silent.py`) porte les trois modules en
exemption écrite, avec cette mesure, précisément pour qu'il ne les lance
pas et ne détruise rien.

---

## D-165 — deux tests stochastiques rouges par intermittence : le décompte « cinq rouges connus » est pris sur des exécutions où ils sont passés par chance

**Rapport seul. Décision requise, rien n'est corrigé** — corriger demande
de choisir *quelle autre grandeur* asserter, et c'est une décision
scientifique, pas une retouche de seuil.

**Où ça bloque.** La suite complète à l'arrivée de cette passe rend **7
échecs**, pas 5. Les cinq connus sont là (les deux de D-132, le trio
`a0e0e02`). Les deux autres ne figurent **ni dans `BRIEF_REPRISE.md` §
« Quatre tests rouges connus », ni dans aucun commentaire de PR** :

| test | régime |
|---|---|
| `tests/mapping/test_signal_contribution.py::test_K_ZZZZ` | 20 tirages, assertion sur le **signe d'une moyenne** |
| `tests/quantum/test_qaoa_arm_is_sampled.py::test_the_ranking_is_nonetheless_visibly_perturbed` | assertion sur `min(rho de Spearman) < 1,0` entre tirages répétés |

**Ce ne sont pas des régressions de cette passe.** Les deux fichiers datent
de `b60dc39` (réorganisation des tests) ; aucune ligne de `src/` n'a été
touchée par cette passe, et les deux échouent sur l'arbre d'**arrivée**.

### `test_K_ZZZZ` — la dispersion varie d'un facteur 15,6 entre exécutions de la même chose

La docstring du test enregistre : *« sur 30 tirages, moyenne = -0,0168,
écart-type 0,0130, t = -7,1 »*, et note que l'assertion **précédente**
« échouait quand même 13 % du temps ». Remesuré, cinq exécutions de la
commande identique :

| exécution | moyenne | écart-type | négatifs | verdict |
|---|---|---|---|---|
| suite complète | **+0,02955** | **0,14325** | 85 % | **échec** |
| isolée 1 | −0,02745 | 0,00915 | 100 % | passe |
| isolée 2 | −0,02018 | 0,02463 | 95 % | passe |
| isolée 3 | −0,01355 | 0,07484 | 95 % | passe |
| isolée 4 | −0,00999 | 0,07948 | 95 % | passe |

**L'écart-type va de 0,00915 à 0,14325 — un facteur 15,6** entre deux
exécutions de la même chose, et la moyenne assertie **change de signe**.
L'écart-type enregistré dans la docstring (0,0130) n'est reproduit que par
l'exécution la plus favorable des cinq ; le `t = -7,1` qui l'accompagne
n'est donc pas une propriété de la grandeur.

C'est mot pour mot le piège que la fiche du dépôt consigne déjà — *« un
contraste asserti à 2σ variait d'un facteur 3,5 entre deux exécutions ; le
test a été déplacé sur le coefficient de plaquette, déterministe »*. Ici le
facteur est 15,6, et la règle de `VIGIL.md` s'applique telle quelle : **si
la grandeur n'est pas reproductible à cette précision, changer de
grandeur — pas de seuil.**

Noter que **le signe reste juste** : 85 à 100 % des tirages sont négatifs à
chaque exécution. Le résultat physique enregistré (une plaquette forte
*abaisse* les qubits qu'elle relie) n'est **pas** remis en cause. C'est sa
**mesure par la moyenne de 20 tirages** qui ne tranche pas.

### `test_the_ranking_is_nonetheless_visibly_perturbed` — échoue ~1 fois sur 4

Il exige l'inverse du précédent : que le classement du bras QAOA **bouge**
(`min(rho) < 1,0`). Il échoue quand toutes les paires rendent
`rho = 1,000` — le bras a été reproductible ce tour-là. Mesuré :

    pytest tests/quantum/test_qaoa_arm_is_sampled.py -q
    # 4 exécutions isolées : 7 passed / 1 failed / 7 passed / 7 passed
    # plus l'échec de la suite complète

Soit **2 échecs sur 5 exécutions**. Sur les 9 scores comparés, les valeurs
bougent bien (0,40049 contre 0,39659 ; 0,03925 contre 0,05164) mais l'ordre
ne bouge pas — c'est exactement le fait consigné dans la fiche du dépôt :
*« Les valeurs bougent, le classement tient »*, auto-corrélation de rang
médiane **0,933**. Une médiane de 0,933 sur 9 éléments veut dire qu'une
partie appréciable des tirages rend 1,000 : **le test contredit le fait
mesuré du dépôt**, il ne le garde pas.

### Ce que ça change pour le décompte

`BRIEF_REPRISE.md` annonce « quatre tests rouges connus » ; les
commentaires de PR des deux dernières passes ont corrigé en cinq, puis six,
puis « c'est cinq ». **Aucun de ces décomptes n'est stable**, parce que deux
tests intermittents entrent et sortent du total selon le tirage. Un
décompte pris sur une exécution ne dit pas combien de tests sont rouges ; il
dit combien l'étaient ce jour-là. Tant que ces deux tests restent en l'état,
la ligne de résumé de la suite **ne peut pas** servir de comparaison d'une
passe à l'autre.

### Trois options, à trancher — aucune appliquée

1. **Changer de grandeur**, comme cela a déjà été fait une fois ici :
   asserter le coefficient sous-jacent, déterministe, plutôt que la moyenne
   d'un tirage. C'est la remédiation que la fiche enregistre comme ayant
   marché (0,0553 → 1,2545, ×22,7). Demande de choisir la grandeur.
2. **Fixer la graine** du bras échantillonné dans ces deux tests. Rend les
   deux reproductibles, mais `test_the_ranking_is_nonetheless_visibly_perturbed`
   perd son objet : il existe précisément pour constater que le bras n'est
   pas déterministe.
3. **Déclarer la dette** : `xfail(strict=False)` avec sa raison et un
   compteur qui mord, le temps que 1 ou 2 soit fait. Rend le décompte de la
   suite de nouveau comparable d'une passe à l'autre, sans rien prétendre
   résoudre.

**Ce que ça ne dit pas.** Aucun nombre publié n'en dépend : ces deux tests
sont des gardes, pas des producteurs d'artefact. Et la mesure a été faite
dans un conteneur neuf, dont les versions de `qiskit` ne sont pas
nécessairement celles qui ont produit l'écart-type de 0,0130 enregistré
dans la docstring — **l'écart entre 0,0130 et les cinq mesures ci-dessus
n'est donc pas attribué**, seulement constaté. Ce qui est attribué, et
suffit, est le facteur 15,6 **entre les cinq mesures d'aujourd'hui**, faites
dans le même conteneur.

### Addendum du 19 août — la même famille mord sur DEUX tests de plus, non prévus par le décompte ci-dessus

Suite complète de cette passe (`163bd38`..`5b8c99c`, conteneur neuf,
dépendances complètes) : **7 échecs**, pas 5. Les cinq connus (D-132, trio
`a0e0e02`) plus **deux nouveaux, ni l'un ni l'autre dans le tableau
ci-dessus** :

| test | assertion | isolé (2 exécutions) |
|---|---|---|
| `tests/quantum/test_qaoa_arm_is_sampled.py::test_the_ranking_survives_the_sampling` | `médiane(rho) > 0,6` sur 15 paires (référence 0,883) | **2/2 passed** |
| `tests/quantum/test_qaoa_scaling_and_hparams.py::test_resolution_scaling` | `qa_frac <= cl_frac + 1e-9` à N=32/64/128, données propres (σ=0) | **2/2 passed** — l'échec de la suite : à N=128, `qa_frac=0,7087` contre `cl_frac=0,6790`, le bras QAOA dépasse le classique par hasard |

Les deux sont bâtis sur `backend="state_vector"` (expectation exacte, pas
d'échantillonnage) — la variabilité ne vient donc pas de bruit de mesure
quantique, mais de l'optimiseur COBYLA lui-même, dont le point de départ
n'est fixé par aucune graine dans ces deux fichiers : c'est la même source
que celle déjà nommée par la fiche du dépôt (« le bras QAOA n'est pas
déterministe… dispersion 1,79e−1 à 3,61e−1 »), pas un nouveau mécanisme.
`test_resolution_scaling` en particulier n'avait encore jamais été vu rouge
dans aucun commentaire de PR — sa docstring ne porte d'ailleurs aucune
mention de variance, contrairement aux deux tests déjà dans le tableau.

**Ce que ça change** : le décompte « 5 connus » sous-compte désormais
**quatre** tests intermittents de la même famille, pas deux — les deux
d'origine plus ces deux-ci. Aucune option nouvelle n'est nécessaire : les
trois listées ci-dessus s'appliquent verbatim à `test_resolution_scaling`
et `test_the_ranking_survives_the_sampling` — même défaut de graine, même
disproportion entre l'effort de calcul (chacun relance un solveur MHD
complet, ~5 min isolé) et l'information que le test en tire. Rien de plus
corrigé ici : ce sont des mesures qui étendent le rapport existant, pas un
nouveau défaut.
