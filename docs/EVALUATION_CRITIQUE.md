# Ce qui est réellement exploitable dans ces résultats

Document de triage. Il sépare trois choses que la masse des chiffres a
tendance à confondre :

- **ce qui est incontestable** et peut porter un argument seul ;
- **ce qui est réel mais borné**, et ne vaut qu'avec sa réserve attachée ;
- **ce qui ne doit pas être cité**, parce qu'une petite erreur y fausserait
  tout le raisonnement en aval.

**Critère utilisé pour « incontestable ».** Un résultat entre en catégorie A
s'il satisfait les quatre conditions suivantes :

1. il est **déterministe** — pas de tirage, pas de moyenne, pas d'intervalle ;
2. il ne dépend **d'aucun appariement de budget** ni d'aucune interpolation ;
3. il est **exactement** ce qu'il prétend être (un zéro est un zéro, pas un
   petit nombre) ;
4. il est **recalculé par `t16_aggregate_v4.py`** depuis son artefact, donc
   vérifiable en une commande.

Tout le reste descend d'un cran. Cette hiérarchie est délibérément sévère :
la campagne a produit 17 instances d'un même défaut — *un calcul qui échoue,
ou ne fait pas ce qu'il annonce, et rend une valeur indiscernable d'une
valeur valide* — et **tout chiffre qu'aucun script ne produisait s'est révélé
faux**. La catégorie A ne contient donc que ce qui survivrait à un
adversaire.

---

## A. Solide — exploitable tel quel

### A1. Le hamiltonien de coût est diagonal, donc il n'y a rien à optimiser quantiquement

| mesure | valeur | source |
|---|---|---|
| hamiltonien diagonal | **1.0000** (100 % des instantanés) | `t11` |
| état fondamental = masque uniforme | **1.0000** (100 %) | `t11b` |
| solveur exhaustif atteint l'optimum | **1.0000** | `t11` |
| glouton atteint l'optimum | **1.0000** | `t11` |
| QAOA p1 reproduit le masque | **1.0000** | `t11` |
| QAOA p2 reproduit le masque | **1.0000** | `t11` |

Un hamiltonien diagonal a son état fondamental dans la base de calcul. Il
n'y a **pas de superposition à exploiter** : le problème est un simple
argmin sur des scalaires. Le fait que le solveur exhaustif, le glouton et
QAOA donnent tous la même réponse n'est pas une coïncidence statistique,
c'est une conséquence structurelle.

**Ce que ça permet d'affirmer :** à la taille déployée, la formulation Ising
ne pose pas un problème que le quantique puisse résoudre mieux qu'un tri.
C'est vrai par construction, pas en moyenne.

**Le seul solveur qui échoue** est le recuit simulé *à froid* (0.5833), le
seul qui ne soit pas initialisé depuis la réponse classique — ce qui indique
que la performance des autres vient de leur initialisation, pas de leur
recherche.

### A2. Les couplages ZZ et ZZZZ ne changent **exactement aucune** décision

| ablation | décisions changées (mappeur v1) | (mappeur v2) |
|---|---|---|
| contrôle (rien enlevé) | 0.0000 | 0.0000 |
| **tout ZZ enlevé** | **0.0000** | **0.0000** |
| **tout ZZZZ enlevé** | **0.0000** | **0.0000** |
| ZZ *et* ZZZZ enlevés | **0.0000** | **0.0000** |
| biais Z enlevé | **0.7500** | — |

C'est un **zéro exact**, pas un petit nombre. Les termes à deux corps et à
quatre corps — c'est-à-dire toute la structure qui justifie une formulation
quantique — peuvent être supprimés sans qu'une seule décision de raffinement
change. Seul le biais à un corps (`Z`) porte la décision, et le retirer la
détruit entièrement.

**Reproduit sur deux mappeurs indépendants**, dont le v2 qui n'a pas le
défaut de fenêtre du v1.

### A3. Réparer le défaut qui supprimait les couplages ne les ressuscite pas

C'est la pièce maîtresse, parce qu'elle ferme la réfutation évidente
(« votre implémentation est buggée, corrigez-la »).

| configuration | ZZ enlevé → décisions changées |
|---|---|
| fenêtre active (déployée) | 0.0000 |
| **fenêtre neutralisée**, couplage restauré à O(25–155) | **0.0000** |
| ZZZZ enlevé, fenêtre neutralisée | **0.0000** |

L'inertie survit à la réparation complète du défaut qu'elle avait révélé.
Le mappeur v2, qui n'a aucune fenêtre par construction, est tout aussi
inerte.

**Ce que ça permet d'affirmer :** l'inertie des couplages n'est pas un bug
d'implémentation. C'est une propriété de la formulation à cette taille.

### A4. Le mécanisme est identifié et mesuré

La fenêtre gaussienne centrée sur le seuil AMR détruit la masse de couplage
ZZ :

| scénario | masse ZZ conservée (config Level-3) | (config déployée open-loop) |
|---|---|---|
| Kelvin–Helmholtz | 0.1142 | 0.0132 |
| Harris tearing | 0.0020 | 0.0000 |
| MHD rotor | 0.0004 | 0.0000 |
| Orszag–Tang | 0.0001 | 0.0000 |

De **88,6 % jusqu'à effectivement 100 %** du couplage est jeté (la masse
conservée descend à 4×10⁻¹⁵⁴), et **préférentiellement là où le couplage est
le plus fort** : Spearman −0.37 / −0.46 / −0.50 sur KH, rotor et tearing.

⚠️ Sur `ot` le Spearman vaut −0.008, pas −0.37. La raison est légitime — la
fenêtre n'y laisse aucune masse ZZ, donc il n'y a rien à corréler — mais
**l'intervalle « −0.37 à −0.50 » exclut ce fold sans le dire** si on ne
l'écrit pas. Citer les quatre valeurs, pas l'intervalle.

### A5. Le solveur converge à l'ordre 1, pas 4

| mesure | valeur |
|---|---|
| RK4 **sans** projection | **3.9977** (≈ 4, conforme) |
| RK4 **avec** projection d'incompressibilité | **1.1194** |
| auto-convergence | 0.9994 |

L'enchaînement RK4 → projection est un splitting de Lie, donc d'ordre 1. Le
schéma annoncé comme d'ordre 4 est d'ordre 1. Étude de convergence
déterministe, reproductible en une minute.

**C'est un défaut de V1, pas de Q-HAS** — mais il affecte toute
interprétation quantitative de la fidélité, dans les deux bras également.

### A6. Le circuit QAOA se déplace de 0–8,5 % vers son propre optimum, et **moins** quand on l'approfondit

| profondeur | déplacement variationnel |
|---|---|
| moyenne | **0.0854** |
| p = 1 | 0.1588 |
| p = 4 | **−0.0132** (négatif) |

Approfondir le circuit le fait *reculer*. Combiné à A1 (l'optimum est
trivial et déjà atteint), cela dit que la couche variationnelle n'apporte
rien et se dégrade avec la profondeur.

### A7. Il n'existe pas de graine physique dans cette suite

Fait de code, vérifiable par lecture :

| scénario | aléa dans la condition initiale |
|---|---|
| `init_kelvin_helmholtz` | **aucun** — `noise_amplitude` multiplie `sin(X)`, un mode déterministe |
| `init_harris_tearing` | **aucun** — `perturbation` multiplie `cos(k·X)` |
| `init_orszag_tang` | **aucun**, et aucun paramètre |
| `init_mhd_rotor` | un vrai RNG, mais `default_rng(42)` **écrit en dur** |

Et la seule graine réelle **ne déplace pas la physique** : 42 → 7 déplace la
signature de trajectoire de **0,0022 %**, parce que le RNG n'entre que comme
`perturbation * standard_normal(...)` avec `perturbation = 0.005`.

**Conséquence directe :** l'exigence pré-enregistrée « ≥ 3 graines physiques
par classe » et la limite déclarée « 1 seed par classe » sont **toutes deux
vides**. Aucune expérience de ce type n'était possible ni informative. Il
faut corriger le protocole, pas s'excuser de ne pas l'avoir suivi.

---

## B. Réel mais borné — ne jamais citer sans sa réserve

### B1. Le décompte de dominance en boucle fermée

> Sur **18 exécutions abouties**, Q-HAS est moins fidèle que la règle de
> seuil appariée en budget sur **18/18**, plus coûteux sur **16/18**,
> strictement Pareto-dominé sur **16/18**.

**Réserves obligatoires :**

- **Le critère primaire pré-enregistré ne dit pas ça.** Sur l'endpoint
  `combined` figé avant l'expérience, Q-HAS gagne **2 folds sur 4** — un
  partage 2–2 qui n'établit rien sous sa propre règle des ≥3/4. La
  conclusion repose sur la comparaison **budget-appariée**, qui est
  **post-hoc**, ajoutée après avoir vu le premier fold. C'est la
  vulnérabilité interprétative principale de toute l'étude et elle doit être
  déclarée à chaque citation.
- n = 5 tirages par fold, **1 seule condition initiale par classe**.
- Le bras Q-HAS n'est pas déterministe (aucune graine dans `src/VQA/`), CV
  17–49 %.
- L'axe de coût **exclut le circuit QAOA**, qui prend 2,7–3,3× le temps mur
  du bras classique sur les trois folds dont le bras classique a abouti.

### B2. Le retrait de la fuite D13 aggrave Q-HAS

Mesuré sur les 4 folds. Contre la frontière classique **au budget que Q-HAS
réalise réellement** : `ot` 1.6×, `kh` 1.9×, `tearing` 2.1× ; `rotor` n'a
**aucun point de fonctionnement** (5/5 tirages canoniques avortent).
Avortements : **12/40 pour Q-HAS contre 0/16 pour le classique**.

**Réserves obligatoires :**

- C'est une **borne**, pas l'expérience définitive : le mode substitue le
  seuil **sans re-régler** le bras QAOA. Le test propre remettrait
  `threshold_amr` dans l'espace Optuna hors classe tenue. Non tenté.
- **Les deux bras ne tournent pas au même seuil** (sur `rotor` : 0.5864
  contre 0.0969). Lire les avortements comme une instabilité propre au bras
  *à budget égal* est faux — c'est l'erreur que mon propre code affichait
  avant correction.
- Le ratio 2.1× de `tearing` traverse un intervalle de frontière large et
  fortement non linéaire : ordre de grandeur, pas mesure.

### B3. L'avantage de transfert apparent était la fuite

Sous la fuite, `tearing` et `kh` montraient Q-HAS se dégradant *moins* sur
condition inédite. Sans la fuite, les deux s'inversent (`kh` ×4.835 contre
×1.364 ; `tearing` ×0.685 contre ×0.389).

**Réserve :** 2 folds informatifs sur 4. `rotor` n'a pas de ratio défini,
et la condition « inédite » de `ot` déplace la trajectoire de **0,28 %** —
elle est vide et exclue par pré-enregistrement, dans les deux sens.

---

## C. À ne pas exploiter

### C1. Toute magnitude par fold

Les ratios 1.30× / 1.90× / 2.74× / 1.81× sont des moyennes d'une quantité à
CV 17–49 %, et **gap/sd < 2 sur trois folds sur quatre**. Les deux passes de
variance rapportent « 1 fold sur 4 séparable » **mais pas le même fold**
(`ot` 2.09 → 1.35, `rotor` 1.56 → 2.30) : la séparabilité elle-même est
instable à n = 5.

Historique qui justifie la sévérité : ces ratios ont été publiés à
2.57–4.41× (un seul tirage), puis 1.56–2.86× (passe non gardée), puis
1.30–2.74× (passe vérifiée). **Deux corrections successives, toutes deux à
la baisse.**

### C2. La robustesse de la direction à la physique

**Elle n'est pas établie.** Sur 7 conditions initiales alternatives : 2
vides, 3 sans verdict solide, **2 décidables — une dans chaque sens**
(`rotor_b` 0.86× *contre* la thèse, `kh_b` 1.24× *pour*).

La portée correcte est donc : *« Q-HAS est moins bon sur les conditions
initiales étudiées »*, **pas** *« en général »*.

### C3. La « frontière atteignable » sur conditions alternatives

Sur `tearing_b`, raffiner de 0.625 à 0.874 rend l'erreur **30× pire**
(0.012 → 1.289). La relation budget → erreur n'est **pas monotone**, donc
« l'erreur classique atteignable à ce budget » n'y est souvent pas définie.
`np.interp` répond quand même, et avait déjà produit **1.28×** comme
résultat.

Toute comparaison budget-appariée sur ces conditions doit passer par un
garde-fou (monotonie locale, raideur bornée, bissection convergée) ou être
refusée.

### C4. Toute extrapolation hors du régime testé

- **taille** : `VQA_N = 2`, 8 qubits, profondeur 0. C'est exactement le
  régime où l'état fondamental est uniforme (A1). Rien ici ne dit quoi que
  ce soit d'une taille plus grande, où la dégénérescence pourrait se lever.
- **matériel** : circuit simulé, sans bruit.
- **réglage** : 4 essais Optuna au lieu de 170. « Q-HAS perd » est en partie
  « Q-HAS a été à peine réglé » — le contrôle budget-apparié atténue cela
  sans le supprimer.

### C5. Le socle V1 n'est pas lui-même validé

`bash run_tests.sh` échoue sur **8 tests en checkout propre**, reproduit à
`cf93ba3`, avant tout ce travail (défaut D6, dont 2 substantiels). Toute
conclusion quantitative hérite de cette incertitude, dans les deux bras.

Par ailleurs `phys_score` est une **L2 relative pondérée par l'instabilité**
(`w = 1 + 0.25(|Jz|/⟨|Jz|⟩ + |ω|/⟨|ω|⟩)`), pas une L2 relative simple. Les
deux bras sont notés identiquement donc aucun biais n'en découle, mais
l'étiquette doit être corrigée partout dans le manuscrit.

---

## D. Les pièges d'interprétation, nommément

| affirmation tentante | pourquoi elle est fausse ou excessive |
|---|---|
| « le quantique n'aide pas pour l'AMR » | l'étude teste **une** formulation à **une** taille minuscule. A1–A3 sont solides *à VQA_N = 2*. |
| « les couplages sont inutiles » | ils sont **inertes dans cette configuration**. A3 renforce beaucoup, mais reste à 8 qubits. |
| « Q-HAS est instable » | 12/40 avortements en mode leak-free, mais **à un seuil différent de celui du contrôle**. Ce n'est pas une instabilité à budget égal. |
| « le classique ne diverge jamais » | faux. T19 enregistre le seuil réglé de `rotor` avortant au pas 208, et 2 de ses 6 points de bissection. |
| « l'étude est pré-enregistrée donc confirmatoire » | l'endpoint primaire donne **2–2**. La conclusion vient d'une analyse **post-hoc**. |
| « les tirages avortés sont pessimistes, donc les inclure serait conservateur » | **empiriquement faux.** Sur `ot`, les avortés paraissaient *meilleurs* (0.42–0.45) que le valide (0.66) ; sur `rotor`, l'inverse. Le sens dépend de l'instant où la garde se déclenche. |

---

## E. Le résultat méthodologique, qui est peut-être le plus publiable

**17 instances d'un même mode de défaillance** ont été trouvées et
corrigées : *un calcul qui échoue, ou ne fait pas ce qu'il annonce, et rend
une valeur indiscernable d'une valeur valide.*

Ce qui généralise n'est pas la liste, c'est ceci :

> **Tout nombre qu'aucun script ne produisait s'est révélé faux. Tout nombre
> que l'agrégateur recalcule depuis son artefact s'est révélé juste.**

Le décompte de tête (19/20 → 18/18), les ratios de la figure, « six
avortements contre zéro », λ = 0.82, la dégénérescence attribuée à la
mauvaise tâche, dix lignes de table qui ne vérifiaient rien : tous des
nombres non épinglés. La relecture attentive a été appliquée partout et n'en
a attrapé aucun. **Ce qui a marché, c'est de faire du nombre une fonction de
l'artefact et de la vérifier mécaniquement.**

Corollaire opérationnel pour n'importe quelle étude de ce type : le statut
d'avortement doit être capturé **au moment de l'exécution**, jamais déduit
de la valeur — avec un bras non déterministe, un rejeu ne reproduit pas le
tirage fautif.

---

## F. Ce qu'il faudrait pour durcir les conclusions

Par rapport coût / gain :

| priorité | expérience | ce que ça débloquerait |
|---|---|---|
| **1** | Re-régler le bras QAOA avec `threshold_amr` dans l'espace Optuna, hors classe tenue | transforme B2 d'une **borne** en test définitif de D13 |
| **2** | 170 essais Optuna au lieu de 4 | retire « Q-HAS a été à peine réglé » |
| **3** | ≥ 10 tirages par fold | rendrait les magnitudes citables (actuellement C1) |
| **4** | ≥ 5 conditions initiales par classe avec frontière dense | trancherait C2, aujourd'hui à une chacune |
| **5** | `VQA_N` plus grand | teste si A1 (fondamental uniforme) se lève — c'est **la** question ouverte |
| — | graines physiques | **inutile**, cf. A7 : rien à faire varier |

**La priorité 5 est la seule qui pourrait renverser A1–A3.** Les autres
consolident ; celle-là teste la limite du régime.

---

## G. Résumé en une page

**À utiliser sans réserve :** le hamiltonien est diagonal et son fondamental
uniforme (100 %) ; les couplages ZZ/ZZZZ changent **exactement 0** décision
sur deux mappeurs ; réparer la fenêtre ne change rien ; la fenêtre jette
88,6 %–~100 % du couplage préférentiellement là où il est le plus fort ; le
solveur est d'ordre 1 ; le circuit se déplace de 0–8,5 % et recule avec la
profondeur ; il n'existe aucune graine physique à faire varier.

**À utiliser avec réserve attachée :** la dominance 18/18 · 16/18 · 16/18
(endpoint primaire 2–2, analyse post-hoc) ; le retrait de D13 aggravant
Q-HAS (borne, seuils différents entre bras).

**À ne pas utiliser :** les magnitudes par fold ; toute affirmation de
robustesse à la physique ; toute extrapolation en taille, en matériel ou en
budget de réglage.

**La conclusion défendable** n'est pas « Q-HAS perd la boucle fermée » — qui
est vraie mais bornée et post-hoc. C'est la chaîne mécanistique : *à la
taille déployée, la formulation Ising rend un problème dont la partie
quantique est structurellement inerte, et cette inertie survit à la
réparation du défaut qui l'expliquait.* Elle ne dépend d'aucun n, d'aucun
appariement de budget, d'aucune interpolation, et d'aucune des statistiques
contestées.

---

*Chaque chiffre de ce document est recalculé depuis son artefact par
`python study/v4/t16_aggregate_v4.py` (152 lignes, 0 DIFF, 0 MISSING),
à l'exception des faits de code (A7, C5), vérifiables par lecture directe
des fichiers cités.*
