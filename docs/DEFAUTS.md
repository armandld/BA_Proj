# Défauts

**Où ça bloque.** Uniquement ce qui n'est pas résolu : ce qui empêche
d'avancer, comment on est tombé dessus, et où on en est pour le lever.

Ce qui est corrigé n'est **pas** ici — c'est un résultat, il vit dans
`RESULTS.md` avec sa mesure et sa commande de vérification.

| | |
|---|---|
| **ouverts** — décision ou campagne requise | **14** |
| **gelés** volontairement | 2 |

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

**D-69, ajouté cette passe, touche une lecture publiée plus directement que
D-48/D-50 : il en change le verdict, pas seulement la lecture.** La table
T31 (« corriger la convention d'axes dégrade la tâche à dim=16 ») ne se
reproduit plus à HEAD — rejouée, l'IC95 qui excluait zéro l'inclut
désormais. Le solveur sous-jacent a changé (D-25, D-26/D-27) après
l'écriture de T31, pour de bonnes raisons ailleurs ; la table, elle, ne
l'a pas suivi. Table refaite requise avant de citer cette conclusion.

**Rien ne bloque plus la réoptimisation côté code.** Ce qui la conditionne
encore est une décision, pas un défaut : voir les deux entrées ci-dessous.

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

## D-24 — le solveur est d'ordre 1,2, la correction n'est pas applicable

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

---

## D-47 — l'état fondamental exact vaut « raffiner partout » sur 40 snapshots / 40

**Où ça bloque.** La phase 4 (`exact_diagonalisation.py`) est censée dire si
l'Hamiltonien v1 capte le besoin de raffinement, et sa porte `promising`
décide seule quels patchs passent en QAOA (phase 5). Mesuré : l'état
fondamental exact est le prédicteur **constant tout-raffiner** sur la
totalité des snapshots disponibles. Une comparaison quantique/classique en
phase 4, et toute campagne phase 5 qui s'appuierait sur sa sélection,
porteraient sur un fondamental qui ne distingue aucune cellule. C'est le
mécanisme derrière D-45 (`RESULTS.md`) : D-45 rend la dégénérescence
visible, il ne la lève pas.

**Comment on est tombé dessus.** En mesurant la clause laissée « non
mesurée » par la passe précédente (`COUVERTURE.md`, `analyze_snapshot` :
`promising = f1_exact >= f1_classique` alors que le commentaire dit `>`).
Il n'existe **aucun artefact `exact_diag_*` dans `results/`** : la phase 4
a été rejouée depuis les DNS et les `patches_*` existants.

**Ce qui est établi.** dim=2 — la seule dimension exécutable, `VQA_DIMS`
valant `[2, 4, 8]` et dim=4/8 demandant 32/128 qubits contre le plafond de
20 codé dans `exact_diag`. Re=400, N=256, 4 scénarios canoniques,
40 snapshots (10 par scénario) :

| | |
|---|---|
| décision exacte tout-à-1 | **40/40** (les 8 marginales valent 1,000) |
| ligne de base classique tout-à-1 | **40/40** |
| `exact_refine != classical_refine` | **0/40** |
| F1 exact == F1 classique | **40/40**, jamais supérieur |
| `promising` avec `>=` | **40/40** — avec le `>` du commentaire : **0/40** |

Le mécanisme, mesuré sur les mêmes 40 snapshots :

| grandeur | valeur |
|---|---|
| `(score − thr)/σ` minimum | **8,4** — donc la fenêtre `exp(−x²)` du ZZ vaut au plus **1,15e−31** |
| `max\|C_edges\|` (ZZ) | **≤ 2,41e−120**, nul sur certains snapshots |
| `max\|K_plaquettes\|` (ZZZZ) | 1,14e−3 à 65,9 |
| `min\|H_edges\|` (biais Z) | 6,2e−3 à 332,1 |
| rapport `min\|H\| / max\|K\|` | **2,02 à 6,64** — le biais Z domine sur **40/40** |
| signe du biais Z | **positif partout** (`z_bias = α_z·(score − thr)`, score > thr partout) |

À résolution VQA le score est loin du seuil de décision (0,344 à 0,869
contre `thr = 0,1496`), donc la fenêtre gaussienne `exp(−((score−thr)/σ)²)`
avec `σ = 0,023` éteint le couplage ZZ. Restent le ZZZZ et un biais Z
valant `α_z = w_z_frac × median(|C|,|K|)` avec `w_z_frac = 10,40` : il
domine d'un facteur 2 à 6,6 et il est positif partout, donc le fondamental
met tous les qubits à |1⟩ sans que la structure ZZ/ZZZZ n'entre dans la
décision.

**Une hypothèse plausible, mesurée et FAUSSE.** `build_patch_hamiltonian`
fournit au mappeur VQA des champs **moyennés par bloc** mais un score
**max-poolé** depuis la pleine résolution — deux opérateurs différents dans
le même appel. Hypothèse : c'est le max-pool qui sature le score et tue le
ZZ. Rejoué avec le score **assorti** (recalculé par `AngleMapper.classical_score`
sur les champs moyennés, donc l'opérateur qui a construit `sim_vqa`/
`fields_vqa`) : décision constante **39/40** au lieu de 40/40, F1 à égalité
**40/40**, écart strict **0/40**. Le score assorti vaut 0,4996 à 0,8660 —
lui aussi très au-dessus du seuil. **L'écart d'opérateur est réel mais
n'est pas la cause** : ne pas y retourner. Ce qui sature, c'est la
résolution VQA elle-même, pas la façon de la réduire.

**Où on en est.** Pas corrigé, et pas corrigeable ici : `σ`, `w_z_frac` et
la normalisation du biais Z sont des paramètres physiques de
`src/Simulation/HamiltParams.py`, l'objet d'étude. Trois issues possibles,
à trancher :

1. **Documenter comme résultat.** À résolution VQA grossière, l'Hamiltonien
   v1 dégénère vers « raffiner partout » : c'est une limite structurelle de
   v1, pas un défaut de la phase 4. Alors la phase 4 telle quelle ne peut
   rien sélectionner et la phase 5 ne doit pas s'appuyer dessus.
2. **Élargir `σ`** pour que la fenêtre ZZ survive à un score loin du seuil,
   et remesurer si cela casse la calibration à pleine résolution.
3. **Resserrer `w_z_frac`** pour que le biais Z ne domine plus les
   couplages — c'est la même grandeur que la décision déjà ouverte sur sa
   **borne haute** (section « La borne haute de `w_z_frac` » ci-dessus) :
   la valeur déployée, 10,40, produit déjà la dégénérescence que la borne à
   1000 était soupçonnée de produire. Les deux décisions se prennent
   ensemble.

Voisin de D-41, sans être le même : D-41 constate un Hamiltonien
**identiquement nul** à pleine résolution sur les scénarios lisses ; ici
l'Hamiltonien est non nul mais son fondamental est **constant**, et cela
sur les 4 scénarios, `mhd_rotor` et `orszag_tang` compris.

```bash
python3 -c "
import sys, numpy as np
sys.path.insert(0, 'src'); sys.path.insert(0, 'study/pipeline')
import exact_diagonalisation as ed
for sc in ['orszag_tang','harris_tearing','kelvin_helmholtz','mhd_rotor']:
    d = np.load(f'results/dns_{sc}_Re400_N256.npz')
    p = np.load(f'results/patches_{sc}_Re400_N256_dim2.npz')
    n = len(d['t']); N = d['vx'].shape[1]
    for si in range(0, n, max(1, n // 10)):
        r = ed.analyze_snapshot(d['vx'][si].astype(float), d['vy'][si].astype(float),
            d['Bx'][si].astype(float), d['By'][si].astype(float), N, 2, 400,
            p['l2_errors'][si], p['is_hard'][si], float(p['l2_threshold']))
        print(sc, si, 'degenere', r['degenerate_decision'],
              'egalite', r['f1_tie'], 'informatif', r['promising_informative'])
"
pytest tests/study/test_exact_diag_degenerate_gate.py
```

---

## Gelés volontairement — ne pas corriger

`study/pipeline/dns_validation.py` est le code de phase 1b. Ses artefacts
sont publiés ; le corriger casse leur reproductibilité. Décision antérieure,
documentée dans le fichier lui-même.

| | défaut | version correcte |
|---|---|---|
| **D2** | `fluctuating_KE` moyenne à travers la couche de cisaillement : sur le profil de base seul elle lit **73 %** de l'énergie totale, et n'évolue que de **0,02 %** quand on allume la perturbation | `dns_extension.fluctuating_ke_fixed` |
| **D3** | `mean_sq_current` porte la même inversion d'axes | `dns_extension.mean_sq_current_fixed` |

`analyse_one` utilise désormais les versions corrigées : **le gel porte sur
les fonctions, pas sur l'analyse qui les appelle.**

Une correction a déjà été annulée ici après qu'un test a rappelé la décision.
Une déviation connue mais non écrite *là où elle vit* se fait recorriger par
erreur.

```bash
pytest tests/study/test_no_private_curl_survives.py tests/study/test_t8_dns_extension.py
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

## D-51 — `study/` teste un Hamiltonien amputé du terme ZZZZ de point X

**Où ça bloque.** Pas sur les nombres publiés — la mesure ci-dessous montre
que le terme est **identiquement nul** à `dim = 2`, la seule résolution d'où
sortent les nombres publiés. Ça bloque sur la **suite** : `VQA_DIMS = [2, 4, 8]`,
et à `dim = 4` le terme pèse autant que le ZZZZ déjà présent. Et ça bloque sur
D-22 : la campagne de réoptimisation prévoit de régler `beta_xpoint`, un
paramètre qu'**aucune mesure de `study/` ne peut voir**.

**Comment on est tombé dessus.** Question 5 de `VIGIL.md` — *un test emprunte-t-il
cette configuration ?* — en relisant `h3_term_ablation.zero_hamiltonian_terms`,
qui met `K_xpoint` à zéro sur l'ablation `no_ZZZZ`. Remonter la chaîne : rien
ne lit jamais `K_xpoint`.

**Ce qui est établi — l'axe et de quel côté chacun est.**

| | `advanced_anomalies_enabled` |
|---|---|
| `PhysicalMapper.compute_coefficients` / `..._v2` | défaut **`False`** |
| `qaoa_inputs.prepare_qaoa_inputs` | ne passe **jamais** l'argument → `False` |
| `qaoa_inputs.run_qaoa_on_snapshot` → `mapping(...)` | **`False`** codé en dur |
| `h0_optimiser_equivalence`, `create_period_hamiltonian(hp, dim, False)` | **`False`** codé en dur |
| **campagne d'entraînement**, `src/train_hyperparams.py` | **`True`** sur **6/6** scénarios — c'est D-33, déjà corrigé et publié dans `RESULTS.md` |

Ce sont les **deux seuls** sites de `study/` qui mentionnent le drapeau
(`qaoa_inputs.py:191` et `:350`) ; tout le reste hérite du défaut. Et
`K_xpoint` n'est produit par le mappeur **que** si le drapeau est vrai : dans
`study/`, la clé n'existe même pas.

**Ce qui est établi — `build_ising_terms` ne peut pas le représenter.**
`ising_terms_and_annealing.build_ising_terms` lit `H_edges`, `C_edges`,
`K_plaquettes` — et rien d'autre : `K_xpoint` n'apparaît pas dans son source.
Le recuit simulé, la diagonalisation exacte et les ablations sont donc
**structurellement aveugles** à cette famille, drapeau ou pas. Et
`h3_term_ablation` met `K_xpoint` à zéro sur `no_ZZZZ` en croyant l'ablater :
il annule une clé que son propre `ground_state_mask` ne lit jamais.

**Ce qui est établi — la mesure.** Mappeur v1 entraîné (`beta_xpoint = 2,39`),
N=256, Re=400, dernier instantané des 4 scénarios canoniques, drapeau OFF
contre ON. Les trois blocs communs sont **identiques à l'octet** dans les huit
cas : le drapeau n'ajoute que le terme de point X, il n'en modifie aucun autre.

**À `dim = 2` — la résolution de tous les nombres publiés :**

| scénario | `max\|K_xpoint\|` | `max\|K_plaq\|` | termes Pauli OFF → ON | spins changés au fondamental exact |
|---|---|---|---|---|
| harris_tearing | **0,0000e+00** | 1,14e−03 | 12 → 12 | **0** |
| kelvin_helmholtz | **0,0000e+00** | 5,00e+01 | 12 → 12 | **0** |
| mhd_rotor | **0,0000e+00** | 4,78e+01 | 12 → 12 | **0** |
| orszag_tang | **0,0000e+00** | 6,60e+01 | 12 → 12 | **0** |

Somme des `|coefficients|` identique dans les quatre cas. **Aucun nombre
publié ne bouge, et aucun ne pouvait bouger.**

**À `dim = 4` — déclarée dans `VQA_DIMS`, jamais exécutée en phase 4 :**

| scénario | `max\|K_xpoint\|` | `max\|K_plaq\|` | rapport | termes Pauli OFF → ON | Σ\|coeffs\| |
|---|---|---|---|---|---|
| harris_tearing | **4,15e+01** | 4,15e+01 | **1,00** | 48 → 56 | 3 997 → 4 329 (+8,3 %) |
| kelvin_helmholtz | 1,24e+01 | 5,45e+01 | 0,23 | 48 → 52 | 9 922 → 9 944 (+0,2 %) |
| mhd_rotor | **4,30e+01** | 5,04e+01 | **0,85** | 48 → 58 | 3 932 → 4 356 (+10,8 %) |
| orszag_tang | **4,35e+01** | 6,96e+01 | **0,63** | 52 → 62 | 5 238 → 5 653 (+7,9 %) |

Le terme de point X est du **même ordre** que le ZZZZ déjà compté, et il
ajoute jusqu'à 10 termes de Pauli sur 48. Il est absent par construction de
tout ce que `study/` mesure.

**Ce que cela veut dire, précisément.**

1. Rien de publié n'est faux de ce fait : à `dim = 2` le terme vaut zéro,
   mesuré, sur 4/4.
2. La conclusion T13 — *« les couplages ZZ/ZZZZ n'ajoutent aucune valeur
   mesurable »* — porte, sans le dire, sur **une** des deux familles ZZZZ.
   L'ablation `no_ZZZZ` ne peut pas ablater ce qui n'a jamais été construit.
3. `beta_xpoint` est un hyperparamètre **entraîné** (2,39, et 2,341306 dans la
   base gelée) dont **aucune** mesure de `study/` ne dépend. D-22 le range
   parmi les 8 à réoptimiser : la campagne réglerait un paramètre que
   l'étude de falsification ne peut pas percevoir. **C'est le point à trancher
   avant de la lancer**, pas après.
4. La fiche `VIGIL_BA_Proj.md` ne liste pas cet axe. Il s'ajoute aux sept.

**Où on en est — non corrigé.** Activer le drapeau dans `study/` déplacerait
tout à `dim ≥ 4` et demanderait d'implémenter `K_xpoint` dans
`build_ising_terms` : ce n'est pas une correction, c'est une campagne. Trois
directions :

1. **Documenter et borner** : écrire que `study/` mesure le Hamiltonien
   *sans anomalies avancées*, et que la conclusion T13 vaut pour la famille
   ZZZZ de vorticité seulement. Coût nul, et honnête.
2. **Implémenter `K_xpoint` dans `build_ising_terms`** puis rejouer à
   `dim = 2` : le résultat est prévisible — inchangé, le terme y est nul —
   mais cela rend l'axe traversable et ferme le trou pour `dim ≥ 4`.
3. **Rejouer la phase 4 et T13 à `dim = 4` drapeau activé.** C'est là que le
   terme vit. Demande de lever le plafond de 20 qubits de `exact_diag`
   (32 qubits) ou de passer au recuit.

Rien n'est appliqué : les trois changent ce que l'étude mesure.

```bash
pytest tests/study/test_xpoint_term_absent_from_study.py
```

---

## D-53 — la seule taille certifiée **non dégénérée** jamais exécutée contredit le critère de H0, et son résultat n'est écrit nulle part

**Où ça bloque.** Sur une **lecture publiée**. `CLAUDE.md` porte
`h0_selection … → RÉFUTÉ` sans qualificatif, et `RESULTS.md` T11 conclut
*« Pre-registered rule fires: quantum optimisation is not the source of any
gain »*. Les deux reposent sur `dim = 2` — 8 qubits, la taille où
`RESULTS.md` note lui-même que *« the optimum itself is uniform, so the
solvers agree on a trivial problem »* (c'est D-45 / D-47). Trois artefacts à
`dim = 3` existent dans `results/`, disent l'inverse, et **`dim3`
n'apparaît pas une seule fois dans `docs/RESULTS.md`**.

**Comment on est tombé dessus.** En vérifiant le commentaire de référence de
`MIN_HIT` / `MIN_MASK_MATCH` — *« les huit solveurs … atteignent l'optimum
certifié sur 100 % des instantanés »* — contre les artefacts réellement
stockés (question 2 de `VIGIL.md` : lire le contrat, puis le vérifier point
par point). Il est vrai à `dim = 2`. À `dim = 3` il est faux.

**Ce qui est établi.** `results/h0_optimiser_equivalence_N96_dim3.npz`,
18 qubits — donc **certifié**, l'optimum y est énuméré exactement — 4
scénarios canoniques, 32 instantanés :

| solveur | hit optimum | mask match | exigé par le critère |
|---|---|---|---|
| exhaustive (certifié) | 1,000 | 1,000 | — |
| `classical_init` (règle classique seule) | **0,500** | **0,500** | exclu du critère |
| greedy | 0,844 | 0,844 | 1,000 |
| sa / sa_warm | 0,594 / 0,750 | 0,938 / 0,875 | rapporté, pas exigé |
| `qaoa_p1` … `qaoa_p6` | **0,156 → 0,062** | **0,156 → 0,219** | 1,000 |
| `qaoa_shots_p6` | 0,062 | 0,219 | 1,000 |

Deux lectures, toutes deux directement dans l'artefact :

1. **Le QAOA tombe plus loin de l'optimum certifié que la règle classique
   dont il part** — 0,156–0,219 contre 0,500 de `classical_init`. Ce n'est
   pas un écart de tirage : le bras QAOA est stochastique (dispersion
   1,79e−1 à 3,61e−1, fiche), 0,062 contre 1,000 ne l'est pas.
2. **Ce n'est pas un problème de budget.** C'est l'objection que
   `--scale-kopt` existe pour lever ; sur les deux artefacts qui l'utilisent
   (`_scalekopt`, `_zeropsi_scalekopt`, harris_tearing, 6 instantanés), le
   QAOA passe à **0,000** sur les quatre profondeurs, `greedy` restant à
   1,000.

**Le critère du module lui-même tranche.** `check_expected_behaviour`,
rejoué sur ces artefacts :

| artefact | verdict |
|---|---|
| `..._N256_dim2.npz` | `[ACCEPTANCE] … H0 refutee a cette taille` |
| `..._N96_dim3.npz` | **lève** — *« des solveurs deterministes n'atteignent plus l'optimum certifie : {greedy: 0.844, qaoa_p1: 0.156, … qaoa_p6: 0.062}. H0 (l'echec vient de l'optimiseur) **redevient plausible**. »* |
| `..._N96_dim3_scalekopt.npz` | **lève** — `{qaoa_p1: 0.0, qaoa_p3: 0.0, qaoa_p6: 0.0, qaoa_shots_p6: 0.0}` |

Le critère existait **avant** ce balayage : il est arrivé en `70a3306`,
le balayage `dim = 3` en `a334712`, et le `git_hash` inscrit dans l'artefact
(`d99b9a6`) descend de `70a3306`. Le script écrit son artefact **avant** de
juger (choix délibéré et commenté). La campagne a donc levé, l'artefact a
été commité, et le résultat n'est entré dans aucun document.

`aggregate_master_table.collect` lit `h0_optimiser_equivalence_N{N}_dim{dim}`
avec `N=256, dim=2` par défaut : les 44 instantanés à `dim = 3` ne sont dans
aucune des 180 lignes. Rien de publié ne bouge — **rien de publié ne les
voit**.

**Ce que ça ne dit pas.** `dim = 3` n'est pas la taille déployée : la boucle
fermée tourne à `dim = 2`. La phrase que le script imprime est d'ailleurs
correctement bornée — *« H0 refutee **a cette taille** »*. Ce qui perd la
borne, c'est le `RÉFUTÉ` non qualifié de `CLAUDE.md` et la conclusion de T11.
Et la règle pré-enregistrée du module couvre le cas : *« si QAOA dévie de
l'optimum, la déviation est rapportée comme une approximation … jamais comme
un avantage »*. Elle n'a simplement jamais été rapportée.

**Où on en est — non corrigé, et ce n'est pas un défaut de code.** Trois
directions, la moins chère d'abord :

1. **Publier les 44 instantanés `dim = 3` dans `RESULTS.md`** et borner le
   `RÉFUTÉ` de `CLAUDE.md` à `dim = 2`. Aucun calcul, aucun nombre déplacé.
2. **Rejouer `dim = 3` sur le tip courant** (le balayage date de `d99b9a6`,
   avant D-37/D-38/D-45/D-46/D-52) avant d'en conclure quoi que ce soit —
   c'est la seule direction qui demande du calcul, ~44 instantanés.
3. **Requalifier le critère** : `MIN_HIT`/`MIN_MASK_MATCH` = 1,000 est
   calibré sur la seule taille dégénérée. Soit il est borné à `dim = 2` dans
   le code, soit il devient une mesure rapportée comme celle du recuit. C'est
   le même arbitrage que D-50 : `VIGIL.md` dit de changer la **grandeur**,
   pas le seuil.

Rien n'est appliqué : les trois touchent une lecture publiée.

```bash
pytest tests/study/test_h0_certified_dim3_contradicts_criterion.py
```

## D-58 — la narration T17 (« ZZ numériquement mort ») décrit le défaut que D-9 a déjà corrigé, pas son résultat

**Où ça bloque.** Sur une **lecture publiée**, et depuis plus longtemps que
les autres entrées de ce fichier. `docs/RESULTS.md` §T17 affirme : *« ZZ is
numerically dead on three of four classes at the deployed open-loop
setting, and retains 1.3 % on the fourth »*, avec un tableau donnant, pour
`deployed_openloop`, `mass kept` = 1,319e−02 / 7,652e−28 / 4,187e−125 /
3,855e−154 (kelvin_helmholtz / mhd_rotor / orszag_tang / harris_tearing).
**Defect D7**, juste après, généralise : « the uncertainty window
annihilates the family it is meant to focus ». Les deux sont faux depuis
`107c1cf` (D-9).

**Comment on est tombé dessus.** En auditant `study/common/`
(`aggregate_master_table.py`, le seul fichier du module encore non lu
fonction par fonction — voir `COUVERTURE.md`), les 16 lignes `DIFF` du
master table ont été prises une à une plutôt que lues comme un total connu.
3 sont expliquées par D-48 (T11b, non reproductible). 1 (`t12/dim8`) tient
dans le plancher de reproductibilité publié (0,3613). Les **12 restantes**
— les 4 `spearman C/w` et les 8 `ZZ mass kept` de T17 — n'avaient aucune
entrée dans `DEFAUTS.md`.

`git log -- results/t17_uncertainty_window.npz` montre trois commits ; le
plus récent, `107c1cf` (« D-9 : le mécanisme d'inertie du ZZ était mesuré
sur le mauvais score »), corrige `h3_uncertainty_window.py` : la fenêtre
gaussienne était mesurée sur `physical_score` alors que le chemin déployé
l'applique à `classical_score` (`refinement.py:506,611`,
`qaoa_inputs.py:161,233`). Le message de ce commit conclut lui-même : *« Le
mécanisme publié — la fenêtre annihile ZZ, donc l'ablater ne change rien —
est donc faux »*, et se termine par *« Agrégateur : 164 OK, 16 DIFF, 0
MISSING »* — l'auteur du commit avait donc déjà sous les yeux le compte de
lignes DIFF que cette entrée explique, sans l'écrire nulle part.

**Ce qui est établi.** L'artefact `results/t17_uncertainty_window.npz`
actuellement commité (`git_hash` interne `50ca5a0`, un ancêtre du tip) porte
déjà les valeurs corrigées :

| classe | `deployed_openloop` publié | `deployed_openloop` commité | `level3_trained` publié | `level3_trained` commité |
|---|---|---|---|---|
| kelvin_helmholtz | 1,319e−02 | **0,1207** | 1,142e−01 | **0,4357** |
| orszag_tang | 4,187e−125 | **0,0496** | 9,679e−05 | **0,4530** |
| mhd_rotor | 7,652e−28 | **0,0332** | 3,951e−04 | **0,5940** |
| harris_tearing | 3,855e−154 | **0,0624** | 1,990e−03 | **0,3379** |

Rejoué indépendamment (`python3 study/h3_representation/h3_uncertainty_window.py
--N 64 --steps 30 --seed 0`, sur le tip courant) : mêmes ordres de grandeur
(0,024–0,591 selon classe et jeu de paramètres), et le script imprime
lui-même `ZZ numerically dead on: none` pour les **trois** jeux de
paramètres, y compris `deployed_openloop`. Petit écart mesuré entre le rejeu
et l'artefact commité (jusqu'à ~0,08 en absolu sur `spearman`,
vraisemblablement une dérive de version d'environnement plutôt qu'une
véritable non-reproductibilité — non tranché, sans conséquence sur la
lecture qualitative ci-dessous, qui tient dans les deux cas).

Les deux références — le tableau `docs/RESULTS.md` §T17 et le dictionnaire
`ref` de `rows_t17` / `rows_t17_spearman` dans `aggregate_master_table.py`
(qui recopie les mêmes nombres) — sont donc toutes les deux des lectures du
défaut D-9 **avant** sa correction, jamais mises à jour après. `DEFAUTS.md`
D-9 le dit déjà correctement, en une ligne : *« annihilation » → ZZ domine K
de 1,5 à 8,2×* — mais rien n'a propagé cette ligne dans la prose de T17, ni
dans D7, ni dans les constantes que `aggregate_master_table.py` compare.

**Collision de numéro trouvée en chemin.** `docs/RESULTS.md:1111` porte un
second paragraphe intitulé **« Defect D9 »** (sans tiret), pour un défaut
sans rapport : `t13_term_ablation.py` écrivait son artefact sous un nom qui
ignorait `--mapper`, écrasant le résultat v1 en rejouant v2. Cette
occurrence n'a jamais de ligne `| D-N |` dans le tableau de tête — elle
n'était donc pas réservée au sens de ce fichier — mais elle partage le
libellé du vrai D-9 (`docs/RESULTS.md:73`), exactement la forme de collision
déjà documentée pour D-18 et D-28. Non renommée ici : c'est un simple
toilettage de texte, mais il touche `RESULTS.md`, réservé à `RESULTS.md`
lui-même — signalé, laissé à la prochaine passe qui y touche.

**Ce que ça ne dit pas.** La conclusion « causalement inerte » de T13
elle-même **ne dépend pas** de cette narration : T18 la reconfirme
séparément, fenêtre neutralisée (`σ → 1e9`), couplage restauré à
O(25–155), ablation ZZ toujours à 0,0000 — et son propre addendum recoupe
via le mappeur v2, sans fenêtre du tout. Rien de la conclusion scientifique
ne bouge. Ce qui bouge, c'est le **mécanisme** que T17/D7 racontent pour
l'expliquer.

**Où on en est — non corrigé, ça touche une lecture publiée.** Trois
directions, la moins chère d'abord :

1. **Réécrire §T17 et D7 de `RESULTS.md`** avec les nombres déjà commités
   (tableau ci-dessus) et mettre à jour le dictionnaire `ref` de
   `aggregate_master_table.py` en conséquence — aucun calcul nouveau, D-9 a
   déjà produit les deux. Coût : straightforward, mais c'est réécrire la
   description du mécanisme central de la section H3, pas juste un nombre.
2. **Rejouer T17 proprement avant de réécrire**, pour trancher l'écart de
   ~0,08 observé entre l'artefact commité et le rejeu de cette passe (piste
   « dérive d'environnement », non mesurée jusqu'au bout).
3. **Laisser tel quel et l'annoter seulement** — ajouter une note en tête de
   §T17 pointant vers D-9 et cette entrée, sans toucher aux nombres publiés
   ni aux constantes de l'agrégateur.

Rien n'est appliqué : les trois déplacent une lecture publiée, et
`VIGIL.md` réserve cela à USER.

```bash
git log --oneline -- results/t17_uncertainty_window.npz
python3 study/common/aggregate_master_table.py | grep DIFF   # 16 lignes, 12 = T17
python3 study/h3_representation/h3_uncertainty_window.py --N 64 --steps 30 --seed 0
```

## D-59 — à dim = 2, la topologie périodique double le lien ZZ shear (pas le ZZZZ), sans conséquence mesurée sur les décisions publiées

**Où ça bloque.** Pas sur une décision publiée aujourd'hui — mesuré
ci-dessous sur les 4 scénarios canoniques. Ça bloque une lecture **future** :
la campagne de réoptimisation D-22 rééquilibre les poids qui font
aujourd'hui du biais Z le terme dominant (facteur 2 à 6,6×, D-47). Si ce
rapport se resserre, ce doublement structurel n'a plus de raison de rester
invisible, et rien ne le surveille.

**Comment on est tombé dessus.** Question 4 de `VIGIL.md`, en lisant
`study/h3_representation/h3_depth_report.py` (dernier fichier non lu de
`study/h3_representation/`, avec `h3_window_counterfactual.py` — voir
`COUVERTURE.md`) : le décompte des termes du `SparsePauliOp` produit par
`create_period_hamiltonian` affichait des libellés Pauli **dupliqués**, à
coefficient identique au bit près — `"IIIIIIZZ"` apparaissait deux fois avec
exactement `-2.4290271580758453` dans les deux cas (orszag_tang, Re=400,
N=256, mappeur v1, instantané médian).

**Ce qui est établi.** `create_period_hamiltonian`
(`src/VQA/cost_hamiltonian.py:309`, chemin QAOA/diagonalisation exacte) et
`build_ising_terms` (`study/common/ising_terms_and_annealing.py:65`, chemin
SA/exhaustif — déjà vérifiés « identiques entre eux » dans `COUVERTURE.md`)
itèrent tous deux sur les `dim` cellules pour émettre un lien ZZ horizontal
`(i, j) → (i, j+1 mod dim)` et vertical `(i, j) → (i+1 mod dim, j)`. À
`dim ≥ 3` cela produit `dim` liens **distincts** par direction (vérifié à
`dim = 3` : aucun label ZZ ne se répète). À **`dim = 2`** l'anneau périodique
dégénère : `(i, 0) → (i, 1)` et `(i, 1) → (i, 0 mod 2)` relient la **même**
paire de qubits, et les deux itérations ajoutent chacune une entrée ZZ à
l'opérateur au lieu d'être fusionnées.

Vérifié sur les coefficients réels (4 scénarios canoniques, Re=400, N=256,
mappeur v1) : `C_edges[0][i,0] == C_edges[0][i,1]` **au bit près** pour
chaque ligne `i`, de même pour `C_edges[1]` par colonne — la formule de
`HamiltParams` (symétrique par construction du saut de cellule) rend donc
les deux entrées dupliquées **identiques**, et l'opérateur applique deux
fois le même couplage shear entre la même paire de qubits : poids effectif
**×2**. `K_plaquettes` (ZZZZ) **n'a pas** ce défaut — vérifié, les 4
quadruplets de qubits produits par les 4 cellules à `dim = 2` sont distincts
deux à deux (la topologie de plaquette ne dégénère pas de la même façon que
celle des liens).

**Mesure d'impact.** État fondamental exact (énumération, même méthode que
D-47), 4 scénarios canoniques × 3 instantanés (Re=400, N=256, dim=2,
mappeur v1, `threshold_amr=0,15`), comparant l'Hamiltonien tel quel contre
une version diagnostique où la seconde occurrence du lien dupliqué est mise
à zéro (retire le doublon, ne touche à rien d'autre) :

| | |
|---|---|
| décisions changées | **0 / 12** |
| fondamental (les deux versions) | `raffiner partout` (biais Z dominant, D-47) |
| `max\|C\|` mesuré | 2,04 à 3,99 selon scénario/instantané |

**Où on en est — non corrigé, rapport seul.** `src/` est l'objet d'étude
gelé (`CLAUDE.md`) ; corriger la topologie périodique changerait la
définition de l'Hamiltonien à `dim = 2` — la **seule** résolution dont sort
un nombre publié — donc tout nombre qui en dépend. Aucun nombre publié ne
bouge (mesuré 0/12), donc rien n'exige de correction aujourd'hui. La
déviation est écrite dans les deux fichiers qui la partagent
(`cost_hamiltonian.py`, `ising_terms_and_annealing.py`) pour ne pas se faire
« recorriger » sans mesure, et pinguée par un test qui échouerait si la
dégénérescence disparaissait sans que ce fichier soit mis à jour. À
surveiller explicitement **avant** de publier tout nombre issu de la
campagne D-22 : si le biais Z retrouvé pèse moins que le couplage ZZ
doublé, le fondamental peut cesser d'être insensible à ce doublon.

```bash
pytest tests/quantum/test_period_hamiltonian_dim2_bond_duplication.py
```

## D-65 — l'identifiant Neon est publié dans un dépôt public, et l'historique le garde

**Où ça bloque.** Rien de calculatoire. Ce qui bloque est une action que le
code ne peut pas faire à la place de USER : **changer le mot de passe côté
Neon**. Tant qu'il n'est pas changé, la base de la campagne est ouverte à
qui lit le dépôt.

**Comment on est tombé dessus.** En lisant `src/import_Neon_data_to_local.py`
en entier pour la première fois (76 lignes, jamais auditées), pendant la
passe qui a produit D-64 dans le même fichier.

**Ce qui est établi.** La ligne 16 portait
`postgresql://neondb_owner:<mot de passe>@ep-patient-hall-abitnl4g-pooler.eu-west-2.aws.neon.tech/neondb`
comme valeur **par défaut** de `--in-url`. `armandld/BA_Proj` est **public**
(`visibility: public`, vérifié par l'API le 13 août 2026). La valeur est
entrée dans l'historique git : elle y reste, quel que soit le commit qui la
retire du fichier.

**Ce qui est fait.** La valeur par défaut est supprimée : le script lit
`--in-url`, sinon la variable d'environnement `NEON_DB_URL`, sinon il
s'arrête en le disant. Un test refuse toute URL portant un mot de passe dans
`src/`. **Cela n'annule pas la publication** — cela empêche la suivante.

**Où on en est.** Deux gestes restent, et ils sont pour USER :

1. faire tourner le mot de passe `neondb_owner` sur Neon — c'est le seul qui
   ferme l'accès déjà publié ;
2. décider si l'historique doit être réécrit. Réécrire casse tous les hashes
   déjà cités dans `RESULTS.md` — le remède peut coûter plus que le mal une
   fois le mot de passe changé. Mesurer avant de trancher : ce document ne
   recommande rien ici.

```bash
pytest tests/pipeline/test_no_credential_in_source.py
```

## D-68 — l'image de la figure AMR est transposée par rapport au reste du dépôt : décision

**Où ça bloque.** Nulle part sur un nombre. Ce qui reste ouvert est un
**choix de présentation** que Vigil ne tranche pas seul, parce qu'il change
la géométrie d'images déjà publiées.

**Comment on est tombé dessus.** En lisant `src/visual.py` et
`src/help_visual.py` en entier — les deux seuls modules que
`tests/pipeline/test_src_coverage_inventory.py` excluait de l'inventaire,
avec pour raison « tracé matplotlib, aucune valeur numérique produite ». La
raison est vraie au sens strict (aucune valeur ne ressort de ces fonctions)
et c'est précisément ce qui les avait mises hors de portée : la figure, elle,
porte une convention d'axes, et elle était fausse (D-68, corrigé).

**Ce qui est établi.** Des trois fonctions du dépôt qui affichent `Jz` :

| fonction | ce qu'elle passe à l'afficheur | axe horizontal | appelée par |
|---|---|---|---|
| `plot_amr_state` | `Jz` **tel quel** | Y (axe 1) | `pipeline.py`, 4×/pas de verrouillage |
| `plot_recursive_state` | `Jz.T` | X (axe 0) | **personne** |
| `simple_hierarchical_plot` | `Jz.T` | X (axe 0) | **personne** |

La seule qui s'exécute est donc la seule à mettre X en vertical. Les deux
autres — mortes — suivent la convention du reste du dépôt. Mesuré : une
structure posée en `X=10, Y=40` au sens de `grid.py` apparaît en
horizontal = 40, vertical = 10.

Les étiquettes ont été corrigées pour nommer l'axe qu'elles portent (D-68
dans `RESULTS.md`) : la figure ne ment plus. Le **champ et les cadres n'ont
pas bougé d'un pixel**, et ils sont cohérents entre eux — le cadre
d'attention tombe bien sur la structure qu'il désigne, vérifié sur un champ
asymétrique sous transposition.

**Où on en est.** La question ouverte, pour USER :

- **laisser tel quel** — les PNG déjà produits restent lisibles à
  l'identique, au prix d'une figure dont l'orientation diffère de celle du
  reste du dépôt ;
- **transposer** `Jz` et les cadres pour mettre X en horizontal, comme
  partout ailleurs — au prix de PNG dont la géométrie ne correspond plus à
  ceux déjà publiés.

Vigil ne recommande rien ici : les deux côtés se défendent, et le coût est
un coût de reproductibilité, pas de justesse.
`tests/pipeline/test_amr_figure_axes.py::test_la_geometrie_du_champ_n_a_pas_ete_transposee`
tombe le jour où la seconde branche est prise — pour qu'elle ne se prenne
pas en silence.

```bash
pytest tests/pipeline/test_amr_figure_axes.py
```

## D-69 — la table T31 n'est plus reproductible à HEAD, et son verdict le plus fort en dépend

**Où ça bloque.** Sur une lecture publiée : la phrase de conclusion de T31
(`RESULTS.md`, « La convention d'axes des mappeurs ») — *« corriger sans
réoptimiser dégrade la tâche à dim=16, avec un intervalle qui exclut
zéro »* — ne se reproduit plus en rejouant la commande publiée telle
quelle. Le verdict passe de **dégrade** à **indécidable**.

**Comment on est tombé dessus.** Prochaine étape du plan de passe
(`h1_solver`, après `visual.py`/`help_visual.py`, D-68) : rejouer les deux
commandes que T31 donne comme sa propre preuve de reproductibilité, avant
de lire le reste du module.

```bash
python study/h1_solver/h1_curl_convention_gap.py --N 128 --dim 8  --n-snaps 6 --seed 0
python study/h1_solver/h1_curl_convention_gap.py --N 128 --dim 16 --n-snaps 6 --seed 0
```

**Ce qui est établi.**

| dim | métrique | publié (`8ee5c8a`) | rejoué à HEAD (`47012fa`) | verdict publié | verdict à HEAD |
|---|---|---|---|---|---|
| 8 | Spearman vs dureté | Δ −0,0029 IC95 [−0,0222, +0,0164] | Δ −0,0122 IC95 [−0,0276, +0,0026] | indécidable | indécidable |
| 8 | F1 budget apparié | Δ +0,0391 IC95 [−0,0156, +0,0938] | Δ +0,0182 IC95 [−0,0182, +0,0573] | indécidable | indécidable |
| 16 | Spearman vs dureté | Δ −0,0665 IC95 [**−0,1328, −0,0146**] | Δ −0,0495 IC95 [**−0,1342, +0,0362**] | **dégrade** | **indécidable** |
| 16 | F1 budget apparié | Δ −0,0299 IC95 [−0,0651, +0,0052] | Δ −0,0286 IC95 [−0,0664, +0,0052] | indécidable | indécidable |

Le seul verdict publié qui n'était pas déjà « indécidable » ne l'est plus :
la borne haute de l'IC95 passe de −0,0146 (exclut zéro) à +0,0362 (l'inclut).

**Cause identifiée, pas seulement constatée.** Le script se déclare
lui-même entièrement déterministe (aucun tirage hors provenance). En
substituant `src/Simulation/solver.py` et `src/Simulation/grid.py` tels
qu'ils étaient au hash `31d5727` (l'écriture de T31) dans un rejeu à
dim=8 : le F1 budget apparié redevient **identique au bit près** au publié
(Δ +0,0391, IC95 [−0,0156, +0,0938]) — confirmant que le solveur, pas le
script T31 lui-même, porte l'écart. Trois commits, tous postérieurs à
l'écriture de T31 et tous corrigeant des défauts mesurés et acceptés (pas
des régressions) :

- **D-25** (`6ecaecf`) : projection spectrale de B désactivée par défaut
  (`PROJECT_B = False`, était `True`) ;
- **D-26/D-27** (`e4d6bbc`, `7e6f1d4`) : `harris_tearing` — l'un des 4
  scénarios canoniques que T31 balaie — réamorcé à **100 %** de son
  amplitude de perturbation prévue, contre **27,5 %** avant correction
  (`div_FD B` : 2,801e−03 → 1,208e−16).

Les trajectoires DNS que T31 échantillonne ont donc changé sous lui entre
la publication et aujourd'hui — légitimement, ce sont des corrections
mesurées et déjà verrouillées par leurs propres tests, pas un nouveau
défaut de leur part.

**Écart résiduel non expliqué, mesuré séparément.** Même avec le solveur de
`31d5727` restauré, le Spearman ne matche pas au bit près (dim=8 :
historique publié +0,7266, rejoué +0,7320 — écart ~0,005, dans le même
IC95, aucun verdict n'en dépend). `environment.yaml` ne fixe aucune version
(`numpy`, `scipy`, `qiskit` non épinglés) : cet écart est cohérent avec une
dérive d'environnement (FFT/BLAS) et ne change aucune conclusion mesurée
ici — noté pour ne pas être reconfondu avec la cause ci-dessus si quelqu'un
le retrouve.

**Second facteur, distinct et cumulatif : D-70.** Indépendamment du
solveur, le label de « dureté » que Spearman et le F1 apparié comparent au
score des mappeurs (`_hard_patches`) ne calculait pas la définition que sa
docstring revendiquait — voir D-70, `RESULTS.md`, corrigé cette même passe.
Une table refaite aujourd'hui à l'identique de la commande publiée
utiliserait déjà la définition corrigée, donc ne serait de toute façon pas
comparable point par point aux nombres publiés sans le dire.

**Où on en est — non corrigé, décision ou campagne requise.** Il n'y a rien
à corriger dans le *code* de T31 lui-même (D-70 mis à part, déjà traité) :
le script fait ce qu'il annonce, le solveur sous-jacent a changé pour de
bonnes raisons ailleurs. Ce qui manque est une **table refaite** — mêmes
commandes, HEAD actuel, avec D-70 appliqué — avant que la phrase « dégrade
à dim=16 » puisse être citée. Deux options, laissées à USER :

- refaire uniquement les deux commandes ci-dessus (quelques minutes,
  mêmes 4 scénarios × 6 instantanés) et republier la table et sa
  conclusion ;
- ou traiter ceci comme faisant partie de la campagne de réoptimisation
  D-22 déjà identifiée par T31 lui-même, puisque le solveur qui a bougé
  est le même dont la campagne recalibrerait les hyperparamètres.

Pas de test qui « épingle » un ancien comportement ici : il n'y a pas de
code corrigé à protéger d'une régression, seulement un narratif dont la
fraîcheur doit être vérifiée à la demande.

```bash
python study/h1_solver/h1_curl_convention_gap.py --N 128 --dim 8  --n-snaps 6 --seed 0
python study/h1_solver/h1_curl_convention_gap.py --N 128 --dim 16 --n-snaps 6 --seed 0
# comparer results/h1_curl_convention_gap_N128_dim{8,16}_v2.npz['verdicts']
# a la table publiee dans RESULTS.md (git hash 8ee5c8a)
```

## Ajouter une entrée

Un défaut n'entre ici que s'il **bloque**. Une fois corrigé, il sort d'ici et
entre dans `RESULTS.md`.

Chaque entrée porte : **où ça bloque**, **comment on est tombé dessus**, **ce
qui est établi** (chiffré), **où on en est**, et la commande qui vérifie
l'état.

Un défaut sans mesure est une suspicion. Un défaut sans commande de
vérification n'a pas sa place ici.
