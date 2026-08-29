# Plan du preprint

Structure mère du manuscrit. **On ne s'étale ni sur les défauts ni sur les
résultats** : on dit ce qu'on a, et on renvoie.

| fichier | contenu |
|---|---|
| **`PLAN_PREPRINT.md`** (ce fichier) | la structure, les verdicts, ce qui reste à faire |
| `DEFAUTS.md` | uniquement ce qui bloque encore |
| `RESULTS.md` | les résultats, comment ils ont été obtenus, comment les réobtenir |
| `EVALUATION.md` | ce qui, dans `RESULTS.md`, est exploitable dans le manuscrit |
| `COUVERTURE.md` | l'audit de couverture du code |

---

## 1. Histoire

D'où vient l'idée, et pourquoi elle est séduisante : mapper les instabilités
d'une grille MHD dans un hamiltonien d'Ising, et laisser un solveur quantique
arbitrer le raffinement.

## 2. Objectif

Décider si un critère de raffinement fondé sur un Ising quantique résolu par
QAOA local, avec un léger cône d'information sur les voisins, a une valeur
au-delà de la baseline classique. Si aucun avantage n'est trouvé, déterminer
**ce qui échoue** : la sélection, la représentation, la forme du modèle, la
spécification de la tâche, ou le fait même de faire du ML.

**Ce que ce n'est pas.** Ni à `dim = 2` (256 états) ni à `dim = 3`
(262 144 états) l'espace n'est intraitable classiquement — les deux se
diagonalisent exactement en quelques secondes sur une machine classique.
Ce travail ne teste donc aucun avantage quantique au sens de la
complexité, et ne prétend jamais le contraire. La question testée est
plus étroite : un solveur NISQ (QAOA, peu de répétitions), utilisé comme
règle de décision, fait-il mieux qu'une heuristique classique bon marché
sur cette tâche précise ? C'est une question empirique légitime même
quand le problème sous-jacent est classique — mais seulement celle-là.

**Préalable.** Cette question ne peut être posée qu'à un modèle dont on sait
qu'il calcule ce que sa documentation annonce. Le vérifier a occupé une part
substantielle du travail et reste une contribution à part entière : plus de
190 défauts de contrat ont été trouvés et fermés par un audit systématique
(cinq questions, huit patrons de défaut — méthode ci-dessous, §6) plutôt que
par une relecture ligne à ligne. → `DEFAUTS.md` pour ce qui reste ouvert,
`RESULTS.md` pour ce qui est fermé.

## 3. Hypothèses et verdicts

**H0 — l'échec vient de la sélection.**
- **H0a** — l'optimiseur atteint-il l'optimum de son propre hamiltonien ?
- **H0b** — mieux l'atteindre améliore-t-il la tâche ?

**H1 — les défauts d'autre origine (solveur, numérique) sont secondaires.**

**H2 — l'échec vient de la forme du modèle.**
- **QH2a** — existe-t-il un modèle restrictif *autre* que V1 qui batte la
  baseline ?
- **H2b** — le modèle est-il simplement trop restrictif ?

**H3 — l'information des voisins (les couplages ZZ/ZZZZ) aide-t-elle ?**

**H4 — l'échec vient de ce qu'on fait du ML, quantique ou non.**

**H5 — l'échec vient de la spécification de la tâche.**

| hypothèse | verdict | portée |
|---|---|---|
| H0a | **NON** — QAOA atteint l'optimum sur 0,000–0,156 des instantanés, contre 1,000 exigé | à `dim = 3` (18 qubits), la seule taille certifiée non dégénérée, sur les deux mappeurs (V2 : D-53 ; V1, celui que la campagne règle : D-200, 0/12) |
| H0b | **NON** — ρ(E_gap, F1) = +0,87 à +0,89 : mieux résoudre H **dégrade** la décision | même protocole, 9 solveurs, V2 et V1 (D-53, D-200) — mesuré aux hyperparamètres de référence, avant toute campagne |
| H1 | **PARTIEL** | les défauts numériques comptent, ne suffisent pas seuls |
| QH2a / H2b | **RÉFUTÉ** | modèle libre testé (`study/h2b_prediction/`), ne bat pas la baseline |
| H3 | **NON** — les couplages ne détectent jamais mieux, et dégradent le F1 dès qu'ils cessent d'être inertes | balayage exhaustif/glouton `dim = 2` à `dim = 8` (T26) ; à `dim = 2` (8 qubits) l'optimum exact est le prédicteur constant, donc rien n'y peut jamais paraître causal — dès `dim = 3` (exhaustif) les couplages changent 6,9–15,3 % des décisions et le F1 baisse de 0,033 à 0,057 par rapport au biais Z seul |
| H4 | **CONJECTURE** | pas d'expérience dédiée qui l'isole du reste |
| H5 | **MIXTE** | à l'horizon physique `t_x` : `harris_tearing`/`kelvin_helmholtz` restent redondants avec le label statique (ρ ≈ 1,0), `mhd_rotor`/`orszag_tang` divergent réellement (ρ jusqu'à 0,66) — corriger l'horizon expose un signal sur la moitié du panel canonique, pas sur tout |

## 4. Comment V1 marche — et pourquoi ça, intuitivement

L'instinct de départ : les instabilités MHD ont une **structure locale
d'interaction**, donc elles se prêtent à un hamiltonien de spins sur les
arêtes de la grille.

- l'encodage : score classique → θ, flux de contrainte → ψ ;
- les trois termes — biais Z, couplage ZZ de gradient, plaquette ZZZZ de
  circulation — et ce que chacun est censé détecter ;
- les portes physiques qui les modulent (Reynolds, Okubo-Weiss, activité
  magnétique) ;
- la forme volontairement restreinte, et pourquoi elle l'est.

Quatre faits structurels conditionnent la lecture de H0b et de H3 :

- la couche de coût est **diagonale** — seul le mixeur déplace une
  probabilité de mesure (γ seul, β = 0 : déplacement 4,4e−16), et il est
  borné à `β ≤ π/(4·reps) = 0,393 rad` ;
- sur un balayage exhaustif de la grille admissible, le mixeur seul
  déplace déjà une probabilité médiane de 0,254 ; l'hamiltonien en ajoute
  0,236 — l'apport de la physique est réel mais partage le même canal
  borné que le mixeur (`RESULTS.md`, « Ce que le circuit peut
  déplacer ») ; ce témoin n'est pas encore intégré comme solveur du panel
  H0b lui-même — l'intégrer demanderait de faire tourner QAOA sur un
  hamiltonien de coût nul, ce que COBYLA ne peut pas optimiser (aucun
  signal) et que le garde-fou `NullHamiltonianError` refuse à raison ; la
  mesure par balayage direct reste la bonne méthode pour cette question ;
- les portes `g_strain` et `g_rot` somment à **1 exactement** — ZZ et ZZZZ
  partitionnent un unique scalaire d'Okubo-Weiss, ils ne sont pas deux
  détecteurs indépendants ;
- aucun des deux couplages ne désigne un TYPE d'instabilité : la
  plaquette vaut `(|ω| + |J|)/norme` (un vortex pur et une nappe de
  courant pure y rendent la même valeur) et le couplage ZZ mélange saut
  hydrodynamique et saut magnétique sous la même racine. Seul `K_xpoint`
  est sélectif. C'est une propriété de la forme choisie, pas un défaut :
  l'hamiltonien détecte « il se passe quelque chose » localement, pas
  « quoi ».

**Une leçon de mesure qui a sa place ici.** La plaquette combine `|ω|` et
`|J|` ; sous un dénominateur commun mal choisi, le signal le plus faible
des deux pouvait numériquement disparaître (rapport 179 sur
`harris_tearing`, 84 sur `kelvin_helmholtz`, historique). Corrigé en
adimensionnalisant les deux magnitudes séparément avant la somme, sans
ajouter de porte. Elle illustre qu'un coefficient bien formé, borné et
adimensionnel peut ne mesurer qu'une moitié de ce qu'il annonce — visible
seulement sur des champs réels, jamais sur un cas synthétique.

### La spécification de la tâche — H5

Le label de la phase 2, `e_i`, est l'écart intra-patch à la moyenne : une
mesure de non-lissité, instantanée et confinée au patch. L'AUC du score
classique seul contre `e_i` — 1,000 (harris), 0,997 (KH), 0,948 (rotor),
0,592 (OT) — dit que sur trois scénarios sur quatre la tâche est quasi
gratuite.

La vérité terrain dynamique `d_i` (protocole §1.2), mesurée à son horizon
physique `t_x = 2π/(dim·(v+b)_rms)` (le temps de traversée réel d'un
patch, 4 à 9× l'horizon `δt = 0,1` du protocole) plutôt qu'à `δt = 0,1`
(où la perturbation ne parcourt que 0,11–0,25 d'une largeur de patch, et
`ρ(d,e) ≥ 0,98` uniformément — rien à distinguer) : `harris_tearing` et
`kelvin_helmholtz` restent redondants avec le label statique même à `t_x`
(ρ ≈ 1,0) ; `mhd_rotor` et `orszag_tang` divergent réellement à certains
instantanés (ρ jusqu'à 0,66 pour Orszag-Tang, 0,82 pour le rotor, au
premier instantané). **Corriger l'horizon est nécessaire et expose un
vrai signal sur la moitié du panel canonique — pas sur tout.** Toute
tâche future consommant `d_i` doit fixer son horizon sur `t_x`.

## 5. Comment le GBT fonctionne, à partir de quoi *(court)*

Features locales contre features en cône, le protocole d'entraînement, et le
rôle de témoin qu'il joue vis-à-vis de V1.

## 6. Étude des deux modèles

**L'approche.** Comment on étudie un modèle qu'on soupçonne :

- **auditer les contrats, pas les valeurs** — pourquoi une fonction existe,
  ce qu'elle promet, ce qu'elle consomme, et si deux chemins censés
  coïncider coïncident encore ;
- **un test doit pouvoir échouer** — pas de seuil calibré sur la mesure du
  jour, un balayage vide doit crier ;
- **toute conclusion porte un intervalle** — bootstrap par trajectoire, bloc
  = instantané ; refus de conclure quand l'intervalle contient zéro ;
- **un prédicteur constant ne vote pas** ;
- **le split aléatoire ne mesure pas le transfert** — il ne vaut que comme
  plafond, et l'argument devient *a fortiori* ;
- **chaque nombre publié se recalcule depuis son artefact.**

**Les graphes qui en résultent.** → `RESULTS.md`

## 7. Discussion — affirmation, réfutation, ce qui reste ouvert

**H0b ferme l'approche plus directement que H3.** Le pari de départ est que
le quantique optimise mieux le combinatoire ; H0b montre que mieux optimiser
n'améliore pas la tâche. C'est la valeur de l'optimisation qui est
attaquée — précisément ce qu'on paierait en qubits.

**En une phrase, pour un lecteur qui n'a pas suivi le détail** : le pari
de départ était qu'un optimiseur quantique piloterait mieux la décision
de raffinement qu'une règle classique bon marché ; il est réfuté à deux
niveaux indépendants, mesurés séparément plutôt que supposés l'un de
l'autre — QAOA n'atteint quasiment jamais l'optimum de son propre
hamiltonien (H0a), et même quand un AUTRE solveur atteint cet optimum
exact, la décision qui en résulte est **pire** que celle d'une règle
classique simple (H0b). Réparer l'optimiseur ne changerait donc rien : le
problème n'est pas seulement que le quantique calcule mal, c'est que ce
qu'il calculerait s'il calculait bien n'est pas ce qu'on veut. Mesuré sur
les deux mappeurs (V2 : D-53 ; V1, celui que la campagne règle réellement :
D-200, ρ = +0,891, p = 0,0013, quasi identique) — la pathologie n'est pas
un artefact du mappeur sans paramètre, elle existe déjà au point de départ
de la campagne. Une limite demeure : un seul point du domaine de recherche
à 9 dimensions a été mesuré pour V1, pas un balayage — ça ne garantit pas
qu'aucun réglage ne fait basculer ρ au négatif, mais ça retire l'hypothèse
qu'un réglage quelconque suffirait par défaut.

**Ce que ceci corrige par rapport à une lecture antérieure du dépôt** :
une version antérieure concluait « H0 → RÉFUTÉ » sans qualificatif, mesurée
entièrement à `dim = 2`, où l'état fondamental exact est le prédicteur
constant « tout raffiner » sur 40 instantanés sur 40 (D-45/D-47) : tous
les solveurs atteignent l'optimum parce qu'il n'y a rien à départager.
Réfuter H0 là-dessus, c'était la réfuter sur un problème vide. La lecture
juste sépare H0a de H0b et les mesure là où le problème est certifié non
dégénéré : l'optimiseur échoue vraiment, et le réparer ne servirait à
rien.

**H3 — les couplages n'aident jamais, et nuisent dès qu'ils comptent.**
La courbe de cône (T1b, `dim = 8`/`16`) montre que l'information des
voisins n'est pas plate en distribution : écarts par saut +0,123 / −0,076 /
+0,100 à `dim = 16`, contre un seuil de retrait pré-enregistré de 0,01 ;
hors le pli dégénéré `harris_tearing` (0,000 à tous les k, cause non
expliquée), le cône dépasse même le classique à `dim = 16`. Mais la
question qui compte pour le manuscrit n'est pas « le cône bouge-t-il la
décision » — c'est « la bouge-t-il en mieux » : T26 (balayage
exhaustif/glouton `dim = 2` à `dim = 8`, étendu à `dim = 3` exhaustif)
répond non. À `dim = 2`, ablater les couplages ZZ/ZZZZ change exactement
0 décision — pas parce que le formalisme Ising serait intrinsèquement
inerte, mais parce que `dim = 2` est la taille où l'optimum exact est
uniforme quel que soit le hamiltonien (D-45/D-47) : rien ne peut jamais
s'y montrer causal. Dès `dim = 3` (exhaustif, non dégénéré), retirer les
couplages change 6,9 % à 15,3 % des décisions selon le mappeur — et le F1
du hamiltonien complet est alors **inférieur** à celui du biais Z seul
(0,405 contre 0,451). Le motif se confirme et s'aggrave à `dim = 4` (32
qubits, glouton contrôlé contre l'exhaustif) et `dim = 8` (128 qubits) :
`F1(biais Z seul) = F1(règle classique)` exactement, et ajouter les
couplages retire 0,033 à 0,057 de F1. **Le meilleur cas de cette famille
de mappings Ising est d'égaler la règle de seuil qu'elle est censée
remplacer ; son pire cas est de faire moins bien.** Cela répond aussi,
au passage, à T13/T11b (leurs lectures `dim = 2` d'origine s'appuyaient
sur une explication depuis retirée — une fenêtre d'incertitude supposée
annihiler le couplage ZZ, qui en fait n'en annihile jamais plus de 96,7 %
et souvent beaucoup moins) : la bonne explication de leurs résultats nuls
était la dégénérescence de `dim = 2`, pas un défaut d'implémentation.

**Conséquence de structure : la fermeture ne repose pas sur H3, mais sur H0b
ET H3 séparément, et ils s'accordent.** H0b montre que même un optimiseur
parfait donnerait une mauvaise décision ; H3 montre que la représentation
elle-même (les couplages) n'apporte rien de bon dès qu'elle cesse d'être
inerte. Les deux angles — optimisation et représentation — ferment
indépendamment, pour la même conclusion.

**Limites qui bornent ces conclusions.** Un seul solveur classique de
référence, protocole à 8 scénarios canoniques et 5 graines physiques (le
protocole confirmatoire lui-même n'a pas encore tourné à cette échelle —
`DEFAUTS.md`, D-22), non-déterminisme du bras QAOA (dispersion par appel
1,79e−1 à 3,61e−1 — les conclusions de CLASSEMENT tiennent, celles qui
reposeraient sur une valeur précise ne tiennent pas), chute d'ordre du
solveur commune aux deux bras. Chiffrées dans `RESULTS.md`.

## 8. Conclusion

**Ce que ce travail teste, en une phrase.** Pas un avantage quantique au
sens de la complexité (§2) : un solveur NISQ (QAOA), utilisé comme règle
de décision de raffinement de maillage, fait-il mieux qu'une heuristique
classique bon marché sur cette tâche précise ?

**Verdict, à trois niveaux mesurés indépendamment** (§7, seule taille
certifiée non dégénérée pour H0 : `dim = 3`, 18 qubits ; H3 confirmé de
`dim = 2` à `dim = 8`) :
- QAOA n'atteint quasiment jamais l'optimum de son propre hamiltonien
  (0 à 15,6 % des instantanés contre 1,000 exigé, selon le mappeur) ;
- même quand un AUTRE solveur atteint cet optimum exactement, la
  décision de raffinement qui en résulte est **pire** qu'une règle
  classique simple (ρ(E_gap, F1) = +0,87 à +0,89, sur 9 solveurs) ;
- même en ignorant la question de l'optimiseur, les couplages
  ZZ/ZZZZ — l'information de voisinage qui motivait tout le projet — ne
  détectent jamais mieux que le biais Z seul, et dégradent le F1 de
  0,03 à 0,06 dès qu'ils cessent d'être numériquement inertes.

Les trois pointent dans la même direction par des chemins indépendants :
réparer l'optimiseur ne sauverait pas l'approche (H0b), et améliorer la
représentation en exploitant mieux les couplages ne le ferait pas non
plus (H3) — ce que le hamiltonien encode au-delà du score classique
n'est pas ce qu'on veut, que ce soit atteint parfaitement ou non.

**Ce qui reste ouvert, sans rouvrir ce verdict.** H1 (les défauts
numériques suffisent-ils seuls à expliquer l'échec ?) est partiel ; H4
(transfert à des conditions inédites) reste une conjecture, faute
d'expérience dédiée. Ni l'un ni l'autre ne porte sur la question que H0b
et H3 tranchent.

**Ce qui manque avant de publier ce verdict tel quel.** Les nombres
ci-dessus viennent des hyperparamètres de RÉFÉRENCE, pas d'une campagne
d'entraînement qui a réellement tourné (`DEFAUTS.md`, D-22). D-200 montre
que ce point de départ est déjà dans la zone que H0b qualifie de
pathologique, et H3 ne dépend d'aucun hyperparamètre entraîné — mais le
verdict gagnerait à être redit avec les nombres d'une vraie campagne.
C'est la seule étape qui reste : voir Appendice A.

## 9. Bibliographie

---

## Appendice A — état du chantier

*Transitoire, disparaîtra du manuscrit.*

**Il ne reste qu'une chose à faire : lancer la campagne.** Tout le reste
qui la précédait dans l'ordre contraint est réglé :

1. **Réoptimisation — prête, jamais lancée.** `train_hyperparams.py
   --phase all` règle 8 hyperparamètres (`beta`, `w_z_frac`, `sigma`,
   `beta_curl`, `beta_xpoint`, `gamma_hydro`, `gamma_mag`, `kappa`) sur
   8 scénarios canoniques. La sélection finale de la phase 3 est
   protégée par un damier de validation tenu à l'écart (`HOLDOUT_GRID`,
   6 régimes physiques jamais vus en entraînement, classement par perte
   MOYENNE du damier) ; l'entraînement lui-même tire un régime physique
   différent par essai (`TRAINING_REGIME_GRID`, 4 régimes, coût par
   essai inchangé — seul le précalcul DNS grossit) ; les deux damiers
   sont disjoints par construction. `_save_results`/`_deploy` écrivent
   un JSON traçable (jeu complet de paramètres, hash de commit,
   `sys.argv`) et le copient automatiquement où `pipeline.py`/`study/`
   le lisent. Rien ne manque au mécanisme ; seule la campagne — plusieurs
   jours de calcul — n'a pas tourné (`DEFAUTS.md`, D-22).
   ```bash
   python src/train_hyperparams.py --print-space   # verifie l'espace, ne calcule rien
   python src/train_hyperparams.py --phase all --seed <graine>
   ```
2. **Relance des campagnes** — attend l'item 1, aucun autre obstacle.
3. **Table maître — à jour.** `study/common/aggregate_master_table.py
   --allow-missing` rend aujourd'hui **268 lignes, 139 OK / 6 DIFF /
   123 MISSING** ; les MISSING sont les lignes de la campagne
   confirmatoire qui n'a pas encore tourné (item 2), pas une régression.
   Ce compte doit être recalculé au moment de rédiger, pas recopié
   d'ici — il bougera dès que la campagne aura tourné.
4. **Témoin « mixeur seul » — mesuré, non intégrable au panel H0b sans
   travail supplémentaire.** Mesuré par balayage exhaustif de (β, γ)
   (`RESULTS.md`, « Ce que le circuit peut déplacer » — médiane 0,254,
   contre 0,490 avec l'hamiltonien). L'intégrer comme solveur du panel
   `h0_optimiser_equivalence.py` demanderait de faire tourner QAOA sur
   un hamiltonien de coût nul : COBYLA n'a alors aucun signal à
   optimiser, et `NullHamiltonianError` (`src/VQA/cost_hamiltonian.py`)
   refuse à raison de construire un tel circuit. La mesure par balayage
   direct reste la bonne méthode pour cette question précise ; en faire
   un solveur comparable aux autres (même E_gap, même F1) demande un
   mécanisme différent, pas encore conçu.
5. **H4/D-197 — la campagne LOSO du niveau 3 n'a que 4 des 8 folds.**
   Complément distinct de la campagne d'hyperparamètres, structurellement
   séparé (`closed_loop_campaign.py` ne lit jamais
   `best_hyperparams.json`). Compléter les 4 folds manquants
   (`vortex`, `coalescence`, `double_tearing`, `magnetic_twist`) demande
   ~170 essais Optuna chacun — un ordre de grandeur d'heures par fold, du
   même ordre que la campagne d'hyperparamètres. À lancer avec elle sur
   la même machine, pas un correctif de code.

**Ce qui reste un résultat négatif non résolu, documenté honnêtement
plutôt que forcé.** D-198 : le plafond GBT sous LOSO (H2b) souffre d'un
signe de corrélation qui s'inverse d'un scénario à l'autre. La
régularisation (`early_stopping=True`) est corrigée mais n'est qu'une
cause mineure. La piste la plus directe — normaliser chaque scénario par
ses propres statistiques avant le pool LOSO — a été essayée et
**aggrave** le plafond moyen (0,278 → 0,165) : elle répare le pire fold
mais en casse d'autres, preuve que l'échelle absolue du score classique
porte elle-même du signal réel selon le scénario. Les deux autres pistes
(modèle non-monotone, calibration du seuil de label) restent non
essayées. Sans conséquence sur le verdict H2b (qui ne repose pas sur
cette seule comparaison), mais non résolu.

**Une conclusion reste bornée par une correction déjà faite, à vérifier
avant de citer un vieux nombre.** Avant la correction de `_process_score`
(`refinement.py`), le biais Z de l'hamiltonien et ses couplages
décrivaient deux grilles différentes à toute profondeur de raffinement
> 0 (le biais d'un patch venait du quart haut-gauche de ce patch, écart
41 % du plus grand coefficient) ; à `max_depth = 4`, réglage de toutes
les campagnes historiques, trois niveaux sur quatre passaient par là.
Corrigé (une ligne, `H_edges` passe de (6,6) à (4,4) à `depth > 0`) :
tout nombre Q-HAS produit APRÈS ce commit, à profondeur > 1, n'est plus
concerné. Un artefact cité à cette profondeur doit d'abord être situé par
rapport à `git log -S "target_dim + 2 * pad" -- src/Simulation/refinement.py`.
