# Défauts

**Où ça bloque.** Uniquement ce qui n'est pas résolu : ce qui empêche
d'avancer, comment on est tombé dessus, et où on en est pour le lever.

Ce qui est corrigé n'est **pas** ici — c'est un résultat, il vit dans
`RESULTS.md` avec sa mesure et sa commande de vérification.

| | |
|---|---|
| **ouverts** — décision ou campagne requise | **7** |
| **gelés** volontairement | 2 |

Des deux qui restent, l'un demande la campagne elle-même, l'autre une décision
qu'on peut prendre après. D-27 et D-37 sont sortis d'ici : corrigés, mesurés,
verrouillés — ils vivent dans `RESULTS.md`.

D-47 bloque la phase 4 et tout ce qui s'appuie sur sa sélection, sans bloquer
la réoptimisation.

**D-48 est le dernier arrivé, et c'est le seul qui touche une lecture
publiée** : la tendance décroissante de T11b est mesurée comme une propriété
du schedule d'initialisation, pas du circuit. Aucun nombre publié n'a bougé —
le master table reste à 180 / 164 / 16 / 0 — mais leur lecture demande une
décision.

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

## D-49 — `aggregate_v3.py` valide contre le baseline V3 archivé, pas contre `RESULTS.md`

**Réservation du numéro.** Entrée en cours d'écriture — mesure faite, rédaction
en cours. Ne pas reprendre D-49.

---

## D-50 — le verdict imprimé de T11b bascule d'une exécution à l'autre de la même commande

**Réservation du numéro.** Entrée en cours d'écriture — mesure faite, rédaction
en cours. Ne pas reprendre D-50.

---

## Ajouter une entrée

Un défaut n'entre ici que s'il **bloque**. Une fois corrigé, il sort d'ici et
entre dans `RESULTS.md`.

Chaque entrée porte : **où ça bloque**, **comment on est tombé dessus**, **ce
qui est établi** (chiffré), **où on en est**, et la commande qui vérifie
l'état.

Un défaut sans mesure est une suspicion. Un défaut sans commande de
vérification n'a pas sa place ici.
