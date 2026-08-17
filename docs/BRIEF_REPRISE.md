# Q-HAS — brief de reprise

**But : rendre une nouvelle conversation immédiatement opérationnelle sans
relire le dépôt.** Le donner en entier en premier message. Il remplace
plusieurs heures d'exploration.

À jour au **17 août 2026**, après la fusion de la branche de revue, la
fermeture de D-47/D-51/D-53/D-58/D-59/D-68/D-91 et le lancement de la
réoptimisation.

---

## 1. Le projet en un paragraphe

Q-HAS : critère de raffinement AMR pour la MHD 2-D où la décision « quelles
cellules raffiner » est encodée dans un hamiltonien d'Ising résolu par
QAOA. Préprint arXiv de 6 pages en préparation. Tout est en simulation ;
aucun matériel quantique.

**La question** : ce critère a-t-il une valeur au-delà de la baseline
classique ? Si non, **ce qui échoue** — la sélection, la représentation, la
forme du modèle, la spécification de la tâche, ou le fait de faire du ML.

**Le pari de départ** : le quantique traite mieux le combinatoire (ici
l'interaction entre voisins), donc ajouter cette information devrait
améliorer l'AMR.

## 2. Organisation

| | |
|---|---|
| `claude/kind-babbage-927g10` | branche vive, **PR #2**. C'est ici qu'on travaille. |
| `vigil/d39-tearing-check-background-current` | agent de revue continue |

**L'agent n'ouvre PAS de PR.** Il pousse sur sa branche et poste ses
rapports en commentaires de PR #2 ; son travail n'atteint `main` que si on
le fusionne. Conséquence : son diff n'est jamais relu isolément.

Commande de fusion : `git merge origin/vigil/d39-tearing-check-background-current`.
Le seul conflit récurrent est le compte d'ouverts en tête de `DEFAUTS.md` ;
le résoudre en **recomptant** les sections `## D-`, jamais à la main.

## 3. Les garde-fous — ils expliquent tout le style du dépôt

Lire `CLAUDE.md` (court). Les règles qui reviennent, chacune née d'un échec
réel :

- **`src/` est l'objet d'étude**, pas une dépendance à améliorer. Toute
  modification est un changement de comportement scientifique : justifiée,
  mesurée, consignée.
- **Un test qui ne peut pas échouer est un défaut.** Corollaires découverts
  depuis : un test qui lit le **source** au lieu du comportement est un
  faux vert — le muter et vérifier qu'il mord ; et **un balayage vide doit
  crier** (une commande qui sélectionne 0 test passe au vert sans rien
  prouver).
- **Mesurer avant d'affirmer, mesurer après avoir corrigé.** Vaut contre
  soi-même : dans la seule bisection de D-132, **trois** hypothèses très
  plausibles ont été réfutées par la mesure.
- **Choisir le champ d'essai qui SÉPARE.** Sur Taylor-Green, deux
  conventions de rotationnel opposées rendent la même enstrophie.
- **Un seuil périmé se REMESURE, il ne se retouche pas.** Consigner
  l'ancienne valeur, la nouvelle, et ce qui les sépare.
- **Ne jamais généraliser depuis un tirage unique.** Le bras QAOA n'est
  semé nulle part dans `src/VQA/` : deux appels identiques diffèrent.
- **En cas de doute entre défaut et choix de conception : mesurer,
  documenter, ne pas corriger, demander.**
- Commentaires en français, identifiants en anglais.

**Règle d'arrêt** (en tête de `DEFAUTS.md`) : un défaut n'est *bloquant*,
et n'entre dans `DEFAUTS.md`, que s'il porte une **lecture publiée** ou
empêche la **campagne** de mesurer ce qu'elle prétend mesurer. Tout le
reste — figures, hygiène des tests — se note en une ligne dans `RESULTS.md`
et se traite après. Écrite parce que le taux de découverte avait dépassé le
taux de résolution : sur D-39→D-131, 98 commits sur le chemin scientifique
contre **79 sur les figures, lanceurs et gardes de test**.

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

## 5. Les hypothèses, et leur état RÉEL

- **H0 — l'échec vient de la sélection.** H0a : l'optimiseur atteint-il
  l'optimum de son hamiltonien ? H0b : mieux l'atteindre aiderait-il ?
- **H1** — les défauts d'autre origine (solveur, numérique) sont secondaires.
- **H2** — l'échec vient de la forme du modèle.
- **H3** — l'information des voisins. **H4** — l'échec vient du ML lui-même.
- **H5** — l'échec vient de la spécification de la tâche.

**H0 n'est PAS réfutée sans qualificatif** — c'est l'erreur la plus facile
à commettre en lisant les anciens documents. La réfutation publiée reposait
sur `dim = 2`, où le fondamental exact vaut « tout raffiner » 40/40 : un
problème vide. Formulation juste :

| | verdict |
|---|---|
| **H0a** — l'optimiseur atteint-il l'optimum ? | **NON** à `dim = 3` (0,062–0,156 contre 1,000 exigé) |
| **H0b** — mieux l'atteindre aiderait-il ? | **NON** — ρ(E_gap, F1) = **+0,870** |

Soit : **l'optimiseur échoue vraiment, et le réparer ne servirait à rien.**

**H3 est à reprendre** : D-58 a rétracté « ZZ is numerically dead on three
of four classes ». La fenêtre conserve 3–12 % de la masse ZZ en boucle
ouverte, 34–59 % au réglage Level-3. L'explication causale des ablations
nulles de T13 et de la progression quasi nulle de T11b **tombe avec elle** —
ces deux résultats restent à expliquer.

## 6. L'état scientifique — le point le plus important

**Quatre mesures indépendantes disent que le fondamental de H n'est pas la
décision AMR qu'on veut**, et ce n'est plus une coïncidence mais une
**direction systématique** : *chaque correction qui retire un artefact rend
le bras quantique moins bon.*

| | |
|---|---|
| **D-47** | à dim=2, le fondamental exact est le prédicteur constant « tout raffiner », **40/40**. Mécanisme mesuré : le score est à **8,4 σ** du seuil, donc la fenêtre ZZ vaut ≤ **1,15e−31**, et le biais Z (positif partout) domine le ZZZZ de **2 à 6,6×**. Il ne reste aucun terme portant une structure spatiale. |
| **D-53** | à dim=3 — seule taille **certifiée ET non dégénérée** — le QAOA atteint l'optimum sur **0,062–0,156** des instantanés contre 1,000 exigé, **sous** la règle classique dont il part (0,500). Et ce n'est pas un budget : avec `--scale-kopt` il tombe à **0,000** quand `greedy` reste à 1,000. |
| **ρ(E_gap, F1)** | **+0,870** sur 9 solveurs à dim=3 : mieux résoudre H **dégrade** la décision AMR. |
| **D-132** | corrélation de rang négative (−0,467) sur 3 des 12 combinaisons. **Élucidé** : bisection sur 9 commits → coupable `6ecaecf` (D-25, projection spectrale). Le bras classait « bien » avant parce qu'il lisait des champs **abîmés par une projection fautive**. |

**A priori enregistré AVANT la campagne** : la réoptimisation ne renversera
pas ce verdict. Optuna déplace des coefficients ; il ne peut pas rendre
informatif un fondamental dégénéré. L'hypothèse vivante est **H2, la forme
du modèle**.

C'est exactement ce que `PLAN_PREPRINT.md` §7 annonçait comme argument de
fermeture : *« H0b ferme l'approche plus directement que H3 — c'est la
valeur de l'optimisation qui est attaquée, précisément ce qu'on paierait en
qubits. »*

**Nuance à ne pas perdre** : les corrélations restent hétérogènes (−0,467
sur 3 combinaisons, +0,95 sur d'autres). Le bras n'est pas mort partout, il
est **instable selon les hyperparamètres** — ce que la campagne arbitre.

## 7. La réoptimisation

**Périmètre : 9 paramètres.** `beta`, `w_z_frac`, `sigma`, `beta_curl`,
`beta_xpoint`, `gamma_hydro`, `gamma_mag`, `kappa`, `relative_percentile`.
`threshold_amr` est **gelé** à 0,14959824837662078 (meilleur essai
classique) pour que la comparaison porte sur ce que le quantique ajoute.

Vérifier avant de lancer, sans rien calculer :
`python src/train_hyperparams.py --print-space`

**Coût mesuré** : 56 min CPU par essai quantique (médiane, 178 essais).
180–450 essais → 168–420 h CPU, soit ~5–13 h sur 32 cœurs. Appliquer
**×1,7** pour l'efficacité parallèle mesurée (59 % : la campagne gelée a
obtenu 5,3× avec 9 essais simultanés, pas 9×).

**Lancer** (le `--phase` attend `1`, pas `phase1`) :

```bash
WORKER_TRIALS=<n> OPTUNA_STORAGE="sqlite:///<chemin>.db" \
  python src/train_hyperparams.py --phase 1 --seed 0
```

La campagne est **interruptible** : Optuna reprend depuis la base.

## 8. État du chantier

**Table maîtresse** : `python study/common/aggregate_master_table.py` →
**180 / 176 OK / 4 DIFF / 0 MISSING**. Les 4 écarts restants : 3 de T11b
(D-48, non reproductible) et 1 sur `t12/dim8`, dans le plancher de
reproductibilité publié (0,3613).

**Défauts ouverts : 11.** Aucun ne bloque la campagne. Ils se répartissent :

- **résolus par la campagne** : D-22, D-39, D-41, D-48, D-50, D-69, D-132 —
  exigent de rejouer T11b, phase 4, T13, T26, T31 ;
- **décisions prises** : D-24 (ordre 1,2 assumé — la correction n'est
  valide que sur `step_full`, et le défaut frappe les deux bras à
  l'identique donc c'est une limite, pas un biais) ;
- **hors chemin critique** : D-98, D-100, D-135.

**Couverture** : `src/` et `study/` sont **lus en entier**. Reste
`figures/v1_legacy/` (17 fichiers), hors chemin critique. Deux fichiers
récents ne sont pas encore dans `COUVERTURE.md` :
`study/common/preflight_coefficients.py` et `study/common/rho_gap_f1.py`.

**Quatre tests rouges connus**, tous des **seuils périmés** (le code a
légitimement changé sous eux) : trois viennent de `a0e0e02` qui fait
consommer `K_xpoint` à `build_ising_terms` — ils exigent de rejouer
**phase 4, T13 et T26** — et un est l'entrée `_FROZEN` de
`run_study_v3.sh`, périmée depuis que D-116 a repointé ce lanceur.

## 9. Commandes de recette

```bash
python -m pytest tests/ -q -m "not slow"        # ~1 h 30, ~2000 tests
python -m pytest tests/study -q                 # un sous-système
python study/common/aggregate_master_table.py   # 180 / 176 / 4 / 0
python study/common/preflight_coefficients.py   # 5 contrôles avant campagne
python src/train_hyperparams.py --print-space   # 9 paramètres, 1 fixé
```

`preflight_coefficients.py` vérifie : spécificité, équilibre des canaux
(2,29), termes à quatre corps vivants à N=256, corrélation avec l'erreur
réelle (ρ = 0,798), et **coïncidence `study/` ↔ circuit** (3,55e−15).

## 10. Pièges de l'environnement — lire avant de lancer quoi que ce soit

- **Le conteneur est recyclé très souvent** (>15 fois en une session), ce
  qui ramène `HEAD` en arrière et efface `/tmp`. Récupération :
  `git fetch origin <branche> && git reset --hard origin/<branche>`.
  **Conséquence pratique : committer et pousser tôt.** Une bisection de
  6 étapes à 45 min n'a jamais pu aboutir en tâche de fond ; il a fallu la
  faire par points isolés, chacun poussé aussitôt.
- **Ne jamais terminer une commande longue par `| tail`** : le code de
  retour du pipeline masque l'échec. Un `argparse` en erreur a été rapporté
  comme succès à cause de ça. Écrire dans un fichier et lire le fichier.
- **Numérotation des défauts** : les deux branches numérotent en continu et
  sont entrées en collision **trois fois**. Avant d'attribuer un numéro :
  ```bash
  git fetch origin 'refs/heads/vigil/*:refs/remotes/origin/vigil/*'
  git show origin/vigil/<branche>:docs/RESULTS.md | grep -o 'D-[0-9]\+' \
    | sort -t- -k2 -n -u | tail -1
  ```
- **Qiskit est little-endian.** Une comparaison d'énergies fausse d'un
  facteur énorme est souvent un ordre de bits inversé : 1,9e+01 → 5,3e−15
  en inversant.
- **Convention d'axes** : `grid.py` fait foi, `indexing='ij'`, `AXIS_X = 0`,
  `AXIS_Y = 1`. Tout opérateur écrit avec un axe numérique nu hors de
  `src/` est suspect. Un défaut d'axes se cache derrière un **vocabulaire**
  d'axes inversé (D-68 : des variables nommées `xs/ys` contenaient `j/i`).
- **Les matrices denses explosent vite** : `dim = 3` fait 18 qubits, donc
  262144². Comparer des opérateurs par leur **liste de termes**, pas par
  `to_matrix()`.

## 11. Ce qui reste, dans l'ordre

1. **Réoptimisation** (en cours de lancement).
2. **Relancer les campagnes** sur le code corrigé : phase 4, T13, T26,
   T11b, T31 — ce sont elles qui lèvent les seuils périmés et les 4 DIFF.
3. **Republier ou justifier** les lignes du master table qui ne se
   recalculent plus.
4. **Ajouter le témoin « mixeur seul »** aux campagnes H0b : sans lui,
   l'apport de l'hamiltonien n'est pas séparable d'une rotation de mixeur.
5. **Analyser les vrais résultats, écrire le papier.**
