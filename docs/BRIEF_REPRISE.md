# Q-HAS — brief de reprise

**Donner ce fichier en entier en premier message d'une nouvelle
conversation.** Il remplace plusieurs heures d'exploration du dépôt.

À jour au **17 août 2026**, commit `41a3e84`.

---

## 0. Les cinq minutes qui orientent tout

Lance ceci avant de lire le reste — c'est l'état réel, pas l'état supposé :

```bash
python study/common/aggregate_master_table.py     # 180 / 176 OK / 4 DIFF / 0 MISSING
python study/common/preflight_coefficients.py     # 5 contrôles, tous OK
python src/train_hyperparams.py --print-space     # 9 paramètres, 1 fixé
grep -cE "^## D-" docs/DEFAUTS.md                 # 11 défauts ouverts
```

Si un de ces quatre chiffres diffère, **c'est le dépôt qui a bougé**, pas ce
document qui a tort — remesure avant de conclure quoi que ce soit.

---

## 1. Le projet

Q-HAS : critère de raffinement AMR pour la MHD 2-D où la décision « quelles
cellules raffiner » est encodée dans un hamiltonien d'Ising résolu par
QAOA. Préprint arXiv de 6 pages en préparation. Tout est en simulation ;
**aucun matériel quantique**.

**La question** : ce critère a-t-il une valeur au-delà de la baseline
classique ? Si non, **ce qui échoue** — la sélection, la représentation, la
forme du modèle, la spécification de la tâche, ou le fait de faire du ML.

**Le pari de départ** : le quantique traite mieux le combinatoire (ici
l'interaction entre voisins), donc ajouter cette information devrait
améliorer l'AMR.

**Où en est la réponse** : voir §4. Elle est largement établie, et négative.

---

## 2. Comment travailler ici — la méthode, pas seulement les règles

Ce dépôt a une culture précise, née d'échecs réels. La suivre est ce qui
rend efficace ; l'ignorer produit du travail qu'il faudra refaire.

**Mesurer avant d'affirmer, mesurer après avoir corrigé.** Une suspicion
non chiffrée n'est pas un défaut. Cela vaut *contre soi-même* : dans la
seule bisection de D-132, **trois** hypothèses très plausibles ont été
réfutées par la mesure. Ne jamais écrire un nombre qu'on n'a pas vu sortir
d'une commande.

**Choisir le champ d'essai qui SÉPARE.** Avant d'écrire un test :
*sur quelle entrée les deux hypothèses divergent-elles ?* Sur Taylor-Green,
deux conventions de rotationnel opposées rendent la même enstrophie — un
test écrit là-dessus passe sans rien vérifier.

**Quatre familles de faux vert**, toutes rencontrées ici :
- un test qui lit le **source** au lieu du comportement (le muter et
  vérifier qu'il mord — c'est la série D-123→D-131) ;
- un **balayage vide** : une commande qui sélectionne 0 test passe au vert ;
- un **seuil calibré sur la mesure du jour** sans que ce soit écrit ;
- une **assertion à tirage unique** sur le bras QAOA, qui n'est semé nulle
  part dans `src/VQA/` : deux appels identiques diffèrent.

**Un seuil périmé se REMESURE, il ne se retouche pas.** Quand un changement
délibéré fait tomber un test, consigner l'ancienne valeur, la nouvelle, et
ce qui les sépare. Trois tests ont été remesurés ainsi aujourd'hui (D-59,
D-68, D-91) ; aucun n'a été « ajusté ».

**`src/` est l'objet d'étude**, pas une dépendance à améliorer. Toute
modification est un changement de comportement scientifique.

**En cas de doute entre défaut et choix de conception : mesurer,
documenter, ne pas corriger, demander.**

Commentaires en français, identifiants en anglais.

**Règle d'arrêt** (en tête de `DEFAUTS.md`) : un défaut n'est *bloquant*,
et n'entre dans `DEFAUTS.md`, que s'il porte une **lecture publiée** ou
empêche la **campagne** de mesurer ce qu'elle prétend mesurer. Le reste se
note en une ligne dans `RESULTS.md` et se traite après. Écrite parce que le
taux de découverte avait dépassé celui de résolution : sur D-39→D-131,
98 commits sur le chemin scientifique contre **79** sur les figures,
lanceurs et gardes de test.

---

## 3. Organisation

| | |
|---|---|
| `claude/kind-babbage-927g10` | branche vive, **PR #2** |
| `vigil/d39-…` | agent de revue continue |

**L'agent n'ouvre PAS de PR.** Il pousse sur sa branche et poste ses
rapports en commentaires de PR #2 ; son travail n'atteint `main` que si on
le fusionne, et son diff n'est jamais relu isolément.

```bash
git fetch origin 'refs/heads/vigil/*:refs/remotes/origin/vigil/*'
git merge origin/vigil/d39-tearing-check-background-current
```

Seul conflit récurrent : le compte d'ouverts en tête de `DEFAUTS.md`.
Le résoudre en **recomptant** les sections `## D-`, jamais à la main.

**Les six documents ont des rôles disjoints** :

| document | contenu | ce qu'on n'y met pas |
|---|---|---|
| `PLAN_PREPRINT.md` | la **source mère** : objectif, hypothèses | ni défaut, ni mesure |
| `DEFAUTS.md` | où ça **bloque**, uniquement | ce qui est corrigé |
| `COUVERTURE.md` | ce qui est **testé**, et ce qui ne l'est pas | des résultats |
| `RESULTS.md` | ce qui est **accompli**, avec sa commande | des blocages |
| `EVALUATION.md` | ce qui, dans RESULTS, est **exploitable** | de nouvelles mesures |
| `CODE_REVIEW.md` | note de relecture | tout le reste |

`docs/archive/` : nombres obsolètes, obtenus sur du code depuis corrigé.
**Ne rien en citer.**

---

## 4. L'état scientifique — la section à lire si tu n'en lis qu'une

### Les hypothèses

**H0** (la sélection) · **H1** (numérique) · **H2** (forme du modèle) ·
**H3** (information des voisins) · **H4** (le ML lui-même) ·
**H5** (spécification de la tâche).

### H0 n'est PAS réfutée sans qualificatif

C'est l'erreur la plus facile à commettre en lisant les anciens documents,
qui portent `h0_selection → RÉFUTÉ`. Cette réfutation reposait sur
`dim = 2`, où le fondamental exact vaut « tout raffiner » **40/40** — un
problème vide. La formulation juste sépare les deux sous-questions :

| | verdict |
|---|---|
| **H0a** — l'optimiseur atteint-il l'optimum de son hamiltonien ? | **NON** à `dim = 3` : 0,062–0,156 contre 1,000 exigé |
| **H0b** — mieux l'atteindre aiderait-il ? | **NON** : ρ(E_gap, F1) = **+0,870** |

Soit : **l'optimiseur échoue vraiment, et le réparer ne servirait à rien.**

### Quatre mesures, une direction systématique

*Chaque correction qui retire un artefact rend le bras quantique moins bon.*
Ce n'est plus une coïncidence, c'est une direction.

| | |
|---|---|
| **D-47** | à dim=2, le fondamental exact est le prédicteur constant « tout raffiner », **40/40**. Mécanisme : le score est à **8,4 σ** du seuil, donc la fenêtre ZZ vaut ≤ **1,15e−31**, et le biais Z (positif partout) domine le ZZZZ de **2 à 6,6×**. Aucun terme ne porte de structure spatiale. |
| **D-53** | à dim=3, seule taille **certifiée ET non dégénérée**, le QAOA atteint l'optimum sur **0,062–0,156** des instantanés contre 1,000 exigé — **sous** la règle classique dont il part (0,500). Pas un problème de budget : avec `--scale-kopt` il tombe à **0,000** quand `greedy` reste à 1,000. |
| **ρ(E_gap, F1)** | **+0,870** sur 9 solveurs à dim=3 : mieux résoudre H **dégrade** la décision. |
| **D-132** | corrélation de rang négative (−0,467) sur 3 des 12 combinaisons. Élucidé par bisection sur 9 commits → `6ecaecf` (D-25, projection spectrale) : le bras classait « bien » parce qu'il lisait des champs **abîmés par une projection fautive**. |

### H3 est à reprendre

D-58 a rétracté la lecture publiée « ZZ is numerically dead on three of four
classes ». La fenêtre conserve **3–12 %** de la masse ZZ en boucle ouverte,
**34–59 %** au réglage Level-3. L'explication causale des ablations nulles
de T13 et de la progression quasi nulle de T11b **tombe avec elle** : ces
deux résultats restent à expliquer.

### A priori enregistré AVANT la campagne

La réoptimisation **ne renversera pas** ce verdict. Optuna déplace des
coefficients ; il ne peut pas rendre informatif un fondamental dégénéré.
L'hypothèse vivante est **H2, la forme du modèle**.

C'est ce que `PLAN_PREPRINT.md` §7 annonçait comme argument de fermeture :
*« H0b ferme l'approche plus directement que H3 — c'est la valeur de
l'optimisation qui est attaquée, précisément ce qu'on paierait en qubits. »*

**Nuance à ne pas perdre** : les corrélations restent hétérogènes (−0,467
sur 3 combinaisons, +0,95 sur d'autres). Le bras n'est pas mort partout, il
est **instable selon les hyperparamètres** — ce que la campagne arbitre.

---

## 5. La réoptimisation

**Périmètre : 9 paramètres.** `beta`, `w_z_frac`, `sigma`, `beta_curl`,
`beta_xpoint`, `gamma_hydro`, `gamma_mag`, `kappa`, `relative_percentile`.
`threshold_amr` est **gelé** à 0,14959824837662078 (meilleur essai
classique) pour que la comparaison porte sur ce que le quantique ajoute.

**Coût, mesuré** : **61,8 min par essai** à N=256 (essai réel, pas une
estimation). 200 essais → **206 h CPU**, soit ~26 h sur 8 cœurs, ~6 h
sur 32.

**Aucune majoration.** Les heures CPU sont les heures facturées. *(Une
version antérieure de ce document appliquait ×1,7 pour une « efficacité de
59 % » : c'était une mauvaise lecture de `PROVENANCE.md`. Le 5,3× y est la
concurrence moyenne réellement obtenue — 159,8 h CPU / 30,4 h de mur =
5,26 — pas un rendement.)* Ce qui reste non mesuré est le passage à
l'échelle au-delà de ~9 workers simultanés.

**Mise en place complète** : `docs/MODE_EMPLOI_CAMPAGNE.md` — du clone au
`best_hyperparams.json`, avec la répétition à faire **avant de payer**.

```bash
bash scripts/repetition_campagne.sh sqlite     # reprise + parallélisme, en secondes
bash scripts/run_reoptimisation.sh 200 0       # 3 gardes avant le 1er essai
```

Le `--phase` de `train_hyperparams.py` attend `1`, **pas** `phase1`.

---

## 6. État du chantier

**Table maîtresse** : **180 / 176 OK / 4 DIFF / 0 MISSING**. Les 4 écarts
restants : 3 de T11b (D-48) et 1 sur `t12/dim8`, dans le plancher de
reproductibilité publié (0,3613).

**11 défauts ouverts**, aucun ne bloque la campagne :
- **résolus par la campagne** : D-22, D-39, D-41, D-48, D-50, D-69, D-132 ;
- **décision prise** : D-24 (ordre 1,2 assumé — la correction n'est valide
  que sur `step_full`, et le défaut frappe les deux bras à l'identique,
  donc c'est une limite, pas un biais) ;
- **hors chemin critique** : D-98, D-100, D-135.

**Couverture** : `src/` et `study/` sont **lus en entier**. Reste
`figures/v1_legacy/` (17 fichiers). Deux fichiers récents ne sont pas encore
dans `COUVERTURE.md` : `study/common/preflight_coefficients.py` et
`study/common/rho_gap_f1.py`.

**Quatre tests rouges connus**, tous des **seuils périmés** attendant une
campagne : trois viennent de `a0e0e02` (`K_xpoint` consommé → rejouer
**phase 4, T13, T26**), un de la famille D-48/D-132. La suite complète rend
`6 failed, 2760 passed` en ~2 h 10.

---

## 7. Pièges de l'environnement — lire avant de lancer quoi que ce soit

- **Le conteneur est recyclé très souvent** (>15 fois en une session) : il
  ramène `HEAD` en arrière et efface `/tmp`. Récupération :
  `git fetch origin <branche> && git reset --hard origin/<branche>`.
  **Committer et pousser tôt.** Une bisection de 6 étapes à 45 min n'a
  jamais pu aboutir en tâche de fond ; il a fallu la faire par points
  isolés, chacun poussé aussitôt.
- **Ne jamais terminer une commande longue par `| tail`** : le code de
  retour du pipeline masque l'échec. Un `argparse` en erreur a été rapporté
  comme succès à cause de ça. Écrire dans un fichier, lire le fichier.
- **Compter les essais `COMPLETE`, jamais le total.** Un worker tué laisse
  son essai à `RUNNING` pour toujours — 18 et 24 essais fantômes dans les
  bases gelées.
- **Numérotation des défauts** : les deux branches numérotent en continu et
  sont entrées en collision **trois fois**. Avant d'attribuer un numéro :
  ```bash
  git show origin/vigil/<branche>:docs/RESULTS.md | grep -o 'D-[0-9]\+' \
    | sort -t- -k2 -n -u | tail -1
  ```
- **Qiskit est little-endian.** Une comparaison d'énergies fausse d'un
  facteur énorme est souvent un ordre de bits inversé : 1,9e+01 → 5,3e−15.
- **Convention d'axes** : `grid.py` fait foi, `indexing='ij'`, `AXIS_X = 0`,
  `AXIS_Y = 1`. Un défaut d'axes se cache derrière un **vocabulaire**
  d'axes inversé (D-68 : des variables nommées `xs/ys` contenaient `j/i`).
- **Les matrices denses explosent** : `dim = 3` fait 18 qubits, donc
  262144². Comparer des opérateurs par leur **liste de termes**, jamais par
  `to_matrix()`.

---

## 8. Ce qui reste, dans l'ordre

1. **Réoptimisation** — 9 paramètres, ~206 h CPU.
2. **Relancer les campagnes** sur le code corrigé : phase 4, T13, T26,
   T11b, T31. Ce sont elles qui lèvent les 4 tests rouges et les 4 DIFF.
3. **Republier ou justifier** les lignes qui ne se recalculent plus.
4. **Ajouter le témoin « mixeur seul »** aux campagnes H0b : sans lui,
   l'apport de l'hamiltonien n'est pas séparable d'une rotation de mixeur.
5. **Confronter à l'a priori du §4**, puis écrire le papier.
