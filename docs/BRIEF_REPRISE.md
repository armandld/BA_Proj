# Q-HAS — brief de reprise

**But de ce fichier : permettre à une nouvelle conversation d'être immédiatement
opérationnelle sans relire le dépôt.** Le donner en entier en premier message.
Il remplace ~2 h d'exploration. Il est à jour au commit `176ee45`.

---

## 1. Ce qu'est le projet

Q-HAS : critère de raffinement AMR pour la MHD 2-D où la décision « quelles
cellules raffiner » est encodée dans un hamiltonien d'Ising résolu par QAOA.
Préprint arXiv de 6 pages en préparation.

**La question** : ce critère a-t-il une valeur au-delà de la baseline
classique ? Si non, **ce qui échoue** — la sélection, la représentation, la
forme du modèle, la spécification de la tâche, ou le fait de faire du ML.

**Le pari de départ** : le quantique traite mieux le combinatoire (ici
l'interaction entre voisins), donc ajouter cette information devrait
améliorer l'AMR.

Tout est en simulation ; aucun matériel quantique.

## 2. Les deux branches

| | |
|---|---|
| `claude/kind-babbage-927g10` | branche vive, **PR #2**. C'est ici qu'on travaille. |
| `vigil/d39-tearing-check-background-current` | agent de revue continue, fusionné dans la vive au commit `176ee45` |

L'agent poste ses rapports en commentaires de la PR #2 ; c'est le canal de
communication avec lui.

## 3. Les garde-fous, qui expliquent tout le style du dépôt

Lire `CLAUDE.md` en entier (il est court). Les règles qui reviennent :

- **`src/` est l'objet d'étude**, pas une dépendance à améliorer. Toute
  modification est un changement de comportement scientifique : justifiée,
  mesurée, consignée.
- **Un test qui ne peut pas échouer est un défaut.** Corollaire découvert
  depuis : un test qui lit le SOURCE au lieu du comportement est un faux
  vert — le muter et vérifier qu'il mord.
- **Mesurer avant d'affirmer, mesurer après avoir corrigé.** Une suspicion
  non chiffrée n'est pas un défaut. Vaut contre soi-même : beaucoup
  d'hypothèses très plausibles se sont révélées fausses à la mesure.
- **Choisir le champ d'essai qui SÉPARE.** Sur Taylor-Green, deux
  conventions de rotationnel opposées rendent la même enstrophie.
- **Un seuil périmé se REMESURE, il ne se retouche pas.** Consigner
  l'ancienne valeur, la nouvelle, et ce qui les sépare.
- **Ne jamais généraliser depuis un tirage unique.** Le bras QAOA n'est semé
  nulle part dans `src/VQA/` : deux appels identiques diffèrent. Toute
  assertion d'égalité exacte sur ses sorties est un coup de dés.
- Commentaires en français, identifiants en anglais.

## 4. Les six documents, rôles disjoints

| document | contenu | ce qu'on n'y met pas |
|---|---|---|
| `PLAN_PREPRINT.md` | **la source mère** : objectif, hypothèses, ce qui reste ouvert | ni défaut, ni mesure |
| `DEFAUTS.md` | où ça **bloque**, uniquement | ce qui est corrigé |
| `COUVERTURE.md` | ce qui est **testé**, comment, et ce qui ne l'est pas | des résultats |
| `RESULTS.md` | ce qui est **accompli**, avec la commande pour le refaire | des blocages |
| `EVALUATION.md` | ce qui, dans RESULTS, est **exploitable** | de nouvelles mesures |
| `CODE_REVIEW.md` | note de relecture | tout le reste |

`docs/archive/` : nombres obsolètes, obtenus sur du code depuis corrigé.
**Ne rien en citer.**

## 5. Les hypothèses

- **H0 — l'échec vient de la sélection.** H0a : l'optimiseur atteint-il
  l'optimum de son propre hamiltonien ? H0b : mieux l'atteindre
  améliore-t-il la tâche ?
- **H1** — les défauts d'autre origine (solveur, numérique) sont secondaires.
- **H2** — l'échec vient de la forme du modèle.
- **H3** — l'information des voisins. **H4** — l'échec vient du ML lui-même.
- **H5** — l'échec vient de la spécification de la tâche.

## 6. L'état scientifique — le point le plus important

**Quatre mesures indépendantes disent que le fondamental de H n'est pas la
décision AMR qu'on veut.**

| | |
|---|---|
| **D-47** | à dim=2, l'état fondamental exact est le prédicteur constant « tout raffiner » sur **40/40** instantanés |
| **D-53** | à dim=3 (seule taille certifiée ET non dégénérée), le QAOA atteint l'optimum certifié sur **0,062–0,156** des instantanés contre **1,000** exigé, et tombe **sous** la règle classique dont il part (0,500) |
| **D-132** | corrélation de rang QAOA/vérité **négative** (−0,467) sur 3 des 12 combinaisons d'hyperparamètres |
| ρ(E_gap, F1) | **+0,870** sur 9 solveurs à dim=3 : **mieux résoudre H dégrade la décision AMR** |

Le mécanisme de D-47, mesuré : la fenêtre gaussienne du couplage ZZ vaut au
plus **1,15e−31** (le score est à 8,4 σ du seuil), donc le terme de voisinage
est éteint ; le biais Z, positif partout, domine le ZZZZ d'un facteur 2 à 6,6.
Le fondamental met tous les qubits à |1⟩ faute de terme porteur de structure.

**A priori enregistré avant la campagne** : la réoptimisation ne renversera
pas ce verdict. Optuna déplace des coefficients ; il ne peut pas rendre
informatif un fondamental dégénéré. H0 est réfutée sur ses deux sous-questions.
L'hypothèse vivante est **H2, la forme du modèle**.

C'est exactement ce que `PLAN_PREPRINT.md` §7 annonçait : *« H0b ferme
l'approche plus directement que H3 — c'est la valeur de l'optimisation qui
est attaquée, précisément ce qu'on paierait en qubits. »*

## 7. Où en est le chantier

**Périmètre de réoptimisation : 9 paramètres.** `beta`, `w_z_frac`, `sigma`,
`beta_curl`, `beta_xpoint`, `gamma_hydro`, `gamma_mag`, `kappa`, et
`relative_percentile` (ajouté récemment, câblé de bout en bout).
`threshold_amr` est **gelé** à la valeur du meilleur essai classique.

**Coût mesuré** : 56 min CPU par essai quantique (médiane, 178 essais
complets). 180–450 essais → 168–420 h CPU, soit ~5–13 h sur 32 cœurs.
Appliquer ×1,7 pour l'efficacité parallèle mesurée (59 %).

**Table maîtresse** : `python study/common/aggregate_master_table.py` rend
**180 / 164 OK / 16 DIFF / 0 MISSING**. Les 16 DIFF **existent toujours**.
D-58 explique que 12 d'entre eux viennent d'une narration T17 périmée —
c'est le plan pour les fermer, pas leur fermeture.

**Défauts ouverts : 18**, dans `DEFAUTS.md`. Une règle d'arrêt y est écrite en
tête : un défaut n'est bloquant que s'il porte une lecture publiée ou empêche
la campagne de mesurer ce qu'elle prétend mesurer. Le reste est groupé APRÈS
la campagne. Elle a été écrite parce que le taux de découverte avait dépassé
le taux de résolution : sur D-39→D-131, 98 commits sur le chemin scientifique
contre **79 sur les figures, lanceurs et gardes de test**.

**Décisions prises par USER** : D-24 assumé (solveur ordre 1,2 — la
correction n'est valide que sur `step_full`, pas sur les patchs locaux non
périodiques ; le défaut est partagé par les deux bras, donc c'est une limite,
pas un biais). D-68 à résoudre par transposition.

## 8. Commandes de recette

```bash
python -m pytest tests/ -q -m "not slow"        # ~1 h 30, ~2000 tests
python -m pytest tests/solver -q                # un sous-système
python study/common/aggregate_master_table.py   # 180 / 164 / 16 / 0
python study/common/preflight_coefficients.py   # 5 contrôles avant campagne
```

## 9. Pièges de l'environnement, à connaître

- **Le conteneur est recyclé souvent** (>10 fois en une session), ce qui
  ramène `HEAD` en arrière et efface `/tmp`. Récupération :
  `git fetch origin <branche> && git reset --hard origin/<branche>`.
  **Conséquence pratique : committer et pousser tôt, et écrire tout résultat
  long dans le dépôt, pas dans `/tmp`.** Une bisection de 6 étapes à 45 min
  n'a jamais pu aboutir pour cette raison.
- **Numérotation des défauts** : les deux branches numérotent en continu et
  sont entrées en collision trois fois. Avant d'attribuer un numéro :
  ```bash
  git fetch origin 'refs/heads/vigil/*:refs/remotes/origin/vigil/*'
  git show origin/vigil/<branche>:docs/RESULTS.md | grep -o 'D-[0-9]\+' \
    | sort -t- -k2 -n -u | tail -1
  ```
- **Qiskit est little-endian.** Une comparaison d'énergies qui semble fausse
  d'un facteur énorme est souvent un ordre de bits inversé : la déviation
  passe de 1,9e+01 à 5,3e−15 en inversant.
- **Convention d'axes** : `grid.py` fait foi, `indexing='ij'`, `AXIS_X = 0`,
  `AXIS_Y = 1`. Tout opérateur écrit avec un axe numérique nu hors de `src/`
  est suspect.

## 10. Ce qui reste, dans l'ordre

1. Fermer les 5 défauts qui portent une lecture publiée : **D-58** (le plus
   rentable : 12 des 16 DIFF), **D-53** (résultat le plus fort du dépôt,
   écrit nulle part), D-69, D-91, D-48.
2. Trancher ce qui reste de décisions.
3. Lancer la réoptimisation.
4. Relancer les campagnes sur le code corrigé, republier les 16 lignes.
5. Analyser les vrais résultats, écrire le papier.

Le témoin « mixeur seul » doit être ajouté aux campagnes H0b : sans lui,
l'apport de l'hamiltonien n'est pas séparable d'une rotation de mixeur.
