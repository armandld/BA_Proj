# Mode d'emploi — lancer la réoptimisation sur des cœurs loués

De la machine vide au `best_hyperparams.json`. Chaque étape porte sa
commande et ce qu'il faut voir en sortie ; si tu ne vois pas ça, ne passe
pas à la suivante.

---

## Le chemin le plus court

Si tu ne lis qu'une chose, lis cette liste. Chaque ligne renvoie à sa
section, qui dit quoi taper et quoi voir.

| | à faire | où |
|---|---|---|
| 1 | **Demander un compte HPC** — c'est gratuit, tout le reste n'a de sens que si ça échoue | §0 |
| 2 | Si tu dois payer : choisir l'offre, et vérifier les **3 propriétés** de la machine | §0 bis |
| 3 | **La répétition, sur ta machine, gratuitement** — reprise, parallélisme, intégrité | §2 |
| 4 | Choisir le mode de stockage — *jamais SQLite sur NFS* | §3 |
| 5 | Installer : `pip install -r requirements.txt`, puis rejouer la répétition **sur la machine louée** | §4 |
| 6 | Dimensionner et écrire le job : `bash scripts/soumettre_campagne.sh …` | §0 bis |
| 7 | Lancer, un worker par cœur, tous sur la même base | §5 |
| 8 | Surveiller — **compter les `COMPLETE`, jamais le total** | §6 |
| 9 | Rapatrier `best_hyperparams.json` sans écraser l'ancien | §7 |
| 10 | Relancer les campagnes qui en dépendent | §8 |

Les étapes 3 et 5 sont la même vérification, faite deux fois : une fois
gratuitement pour savoir si le code tient, une fois sur la machine payée
parce que **son environnement peut différer**. C'est la répétition qui a
trouvé D-136, un mode qui se serait effondré après l'allocation.

---

## 0. Avant tout : la voie gratuite

Le HPC d'Imperial est **gratuit pour les étudiants**, dispose d'un système
de fichiers partagé et de disques persistants par construction. La campagne
demande ~206 h CPU, ce qui y est une soumission ordinaire.

Si tu y as accès, saute la partie location : va directement en **§4
(cluster partagé)**. C'est une demande de compte, pas un budget.

---

## 0 bis. Où trouver les cœurs

**Le besoin, pour cadrer** : 206 h CPU, **1 cœur et ~200 Mo par worker**,
**~8 Mo de disque** au total, **aucun GPU**, **aucun réseau** (sauf base
Postgres distante). C'est un profil « beaucoup de petits jobs indépendants »
— le moins cher et le plus facile à trouver qui soit. Ne loue **pas** une
grosse machine : loue beaucoup de petits cœurs, ou un cluster.

### Le HPC d'Imperial — gratuit, à essayer en premier

Système de fichiers partagé, disques persistants, ordonnanceur PBS. 206 h CPU
y est une soumission ordinaire. **C'est une demande de compte, pas un budget.**
Voir §4 « cluster partagé » et `scripts/soumettre_campagne.sh pbs`.

Autres voies gratuites du même type si Imperial n'aboutit pas : le cluster de
ton département, un accès via un encadrant, ou une allocation nationale
étudiante.

### Le cloud, si tu dois payer

| offre | ce que ça vaut ici |
|---|---|
| **instances interruptibles** (spot AWS, preemptible GCE, Azure Spot) | **le bon choix** : ~3 à 5× moins cher que le tarif normal, et la campagne est faite pour ça — elle reprend seule depuis la base |
| instances normales | inutilement cher pour un travail interruptible |
| **Hetzner / OVH / Scaleway** (serveurs dédiés ou VPS) | souvent le meilleur prix au cœur-heure en Europe, facturation à l'heure ou au mois, pas d'interruption |
| offres « HPC à la demande » | pratiques mais chères pour un besoin aussi simple |

**Les prix bougent : vérifie-les le jour où tu loues.** L'ordre de grandeur
à retenir est qu'un cœur-heure banal se compte en centimes, et que 206 h CPU
est donc un budget de l'ordre de quelques dizaines d'euros sur des instances
interruptibles, davantage sur des instances normales. Fais le calcul toi-même
avec le prix affiché :

```
coût ≈ 206 × (prix du cœur-heure) × (1 / efficacité de parallélisme)
```

L'efficacité au-delà de ~9 workers simultanés **n'est pas mesurée** : c'est la
seule inconnue du dimensionnement. Commence à 8 workers.

### Ce qu'il faut exiger de la machine, quelle qu'elle soit

Trois propriétés, dans cet ordre — les deux premières ont déjà fait échouer
des tentatives :

1. **Un disque persistant et inscriptible.** La base *est* la campagne. Sur
   un conteneur dont le disque est éphémère, tout est perdu à chaque
   redémarrage.
2. **Une durée de vie supérieure à un essai (~62 min).** Un environnement
   recyclé plus souvent ne terminera jamais aucun essai. Mesuré ici :
   sur un conteneur recyclé ~3 fois en quelques heures, **zéro essai
   `COMPLETE`**.
3. Assez de RAM pour `200 Mo × n_workers`, plus les DNS.

**Ne prends pas d'instance interruptible sans disque persistant attaché.**
La combinaison « interruptible + disque éphémère » est la seule qui ne
marche pas du tout.

### Le dimensionnement, en une commande

```bash
bash scripts/soumettre_campagne.sh pbs   8 24 200   # 8 workers, 24 h
bash scripts/soumettre_campagne.sh slurm 16 12 200  # 16 workers, 12 h
```

Le script écrit le fichier de job, annonce la capacité (`essais par worker ×
workers`) et **prévient si elle est inférieure à la cible** — auquel cas il
faut resoumettre, ce que la campagne supporte.

---

## 1. Ce dont la machine a besoin

Mesuré, pas supposé :

| | |
|---|---|
| CPU | ~1,03 h par essai (essai mesuré à **61,8 min** à N=256) |
| RAM | **~200 Mo par worker** (les DNS des 4 scénarios en mémoire) |
| disque | **~8 Mo** pour 200 essais (~40 Ko/essai) |
| GPU | **aucun** |
| réseau | aucun, sauf si base Postgres distante |

Le disque est minuscule, mais il doit être **persistant** : la base est le
seul endroit où vit la campagne.

**Budget** : 206 h CPU pour 200 essais. Multiplier par le prix du
cœur-heure. **Aucune majoration** — les heures CPU sont les heures
facturées (voir `PROVENANCE.md`, section sur le « 5,3× »).

---

## 2. Répétition — À FAIRE AVANT DE PAYER

Sur ta machine, gratuitement, en quelques secondes :

```bash
bash scripts/repetition_campagne.sh sqlite
bash scripts/repetition_campagne.sh journal
```

Attendu, pour chacun :

```
[1] REPRISE    : 3 essais -> 6 apres relance
[2] PARALLELE  : 6 -> 12 avec 2 workers simultanes
[3] SURVIE     : 12 COMPLETE, 0 RUNNING, 12 au total
  REPETITION REUSSIE
```

Ces trois propriétés — reprise après interruption, parallélisme sans
corruption, intégrité — ne dépendent **pas** de la taille des essais. Les
vérifier en miniature les vérifie tout court.

> Cette répétition a déjà trouvé **D-136** : le mode `journal`, celui prévu
> pour les systèmes de fichiers partagés, levait `AttributeError` au
> lancement (API Optuna déplacée en 4.0). Il se serait effondré sur des
> cœurs facturés, après l'allocation et le précalcul des DNS.

---

## 3. Choisir le mode de stockage

| ton cas | mode | variable |
|---|---|---|
| **une machine**, N processus | SQLite sur volume attaché | `OPTUNA_STORAGE="sqlite:////chemin/persistant/q_has_v2_phase1.db"` |
| **plusieurs machines**, FS partagé (NFS, Lustre) | Journal | `OPTUNA_JOURNAL=/chemin/partage/journal` |
| **plusieurs machines**, sans partage | Postgres | `OPTUNA_STORAGE="postgresql://user:pw@hote/base"` |

**Ne jamais mettre SQLite sur NFS** : son verrouillage n'y est pas fiable,
et deux workers peuvent corrompre la base. C'est exactement à cela que sert
le mode Journal.

Le code est déjà réglé pour un Postgres distant (`pool_pre_ping`,
`pool_recycle=300`) : un essai dure ~62 min, et un pooler ferme les
connexions inactives entre-temps.

---

## 4. Mise en place

### Sur une machine louée

Le dépôt est privé : `git clone` sur une machine louée demande une
authentification. Deux voies, la première étant la plus simple :

- **jeton d'accès personnel GitHub**, en lecture seule, à révoquer après la
  campagne : `git clone https://<jeton>@github.com/<toi>/BA_Proj.git` ;
- **clé SSH** créée *sur la machine louée* (jamais recopiée depuis ton
  poste), sa partie publique ajoutée en *deploy key* du dépôt.

Ne laisse pas traîner un jeton dans l'historique du shell d'une machine que
tu vas rendre : `history -c` avant de la libérer, ou passe le jeton par
`git credential`.

```bash
# 1. le code, au commit exact
git clone <depot> && cd BA_Proj
git checkout claude/kind-babbage-927g10
git log --oneline -1          # noter le hash : il décrira la campagne

# 2. l'environnement
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt    # ou : pip install optuna numpy scipy qiskit qiskit-aer

# 3. le volume persistant doit être MONTÉ et inscriptible
mkdir -p /mnt/persist && touch /mnt/persist/.essai && rm /mnt/persist/.essai

# 4. la répétition, sur CETTE machine (l'environnement peut différer)
bash scripts/repetition_campagne.sh sqlite

# 5. le périmètre
python src/train_hyperparams.py --print-space
#    attendu : 9 paramètres explorés, threshold_amr fixé
```

### Sur un cluster partagé (Imperial)

Mode `journal`, un job par cœur, tous sur le **même** répertoire :

```bash
export OPTUNA_JOURNAL=/rds/general/user/<toi>/home/q_has_journal
mkdir -p "$OPTUNA_JOURNAL"
bash scripts/repetition_campagne.sh journal    # d'abord, sur un nœud interactif
```

Puis le script de soumission, qui écrit le fichier de job et dit la
commande à lancer (§0 bis). Le nombre de tâches = le nombre de workers ;
chacune consomme des essais de la même étude jusqu'à épuisement.

```bash
bash scripts/soumettre_campagne.sh pbs 8 24 200
qsub scripts/job_campagne_pbs.sh
```

### Sauvegarder pendant que ça tourne

La base **est** la campagne : sur une machine louée, c'est la seule chose
qui ait de la valeur, et elle ne pèse que quelques mégaoctets. La rapatrier
régulièrement coûte quelques secondes et évite de tout perdre sur une
préemption mal placée.

```bash
# depuis TON poste, toutes les quelques heures
rsync -avz machine:BA_Proj/results/hyperparams/reoptimisation/ ./sauvegarde/
```

Une base copiée pendant une écriture peut être tronquée. Si `sqlite3` est
disponible sur la machine, préférer une copie cohérente :

```bash
sqlite3 <base>.db ".backup /tmp/sauvegarde.db"   # puis rapatrier /tmp/sauvegarde.db
```

Vérifier ce qu'on a rapatrié, sans rien supposer :

```bash
python scripts/inventaire_campagne.py --racine ./sauvegarde
```

---

## 5. Lancer

Un lancement **par cœur**, tous sur la même base :

```bash
OPTUNA_STORAGE="sqlite:////mnt/persist/q_has_v2_phase1.db" \
  bash scripts/run_reoptimisation.sh 200 0
```

Le script refuse de partir si l'arbre git est modifié, si le préflight des
coefficients échoue, ou si le périmètre n'est pas 9. Ces trois gardes
s'exécutent **avant** le premier essai.

Sortie attendue avant que ça calcule :

```
arbre     : propre
[OK ] specificite / equilibre / vivant / pertinence / coincidence
VERDICT : les coefficients font leur travail. Campagne possible.
9 parametres explores : beta, w_z_frac, …, relative_percentile
```

---

## 6. Surveiller

```bash
python - <<'EOF'
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
st = optuna.load_study(study_name="q_has_v2_phase1",
                       storage="sqlite:////mnt/persist/q_has_v2_phase1.db")
comp = [t for t in st.trials if t.state.name == "COMPLETE"]
run  = [t for t in st.trials if t.state.name == "RUNNING"]
print(f"{len(comp)} COMPLETE, {len(run)} RUNNING, {len(st.trials)} total")
if comp:
    print("meilleur :", round(st.best_value, 6))
    print("params   :", {k: round(v, 4) for k, v in st.best_params.items()})
EOF
```

**Compter les `COMPLETE`, jamais le total.** Un worker tué laisse son essai
à `RUNNING` pour toujours : c'est ce qui a laissé **18 et 24 essais
fantômes** dans les bases de la campagne gelée, et fait croire à 345 essais
là où 303 avaient abouti.

**Après une interruption**, relancer exactement la même commande : Optuna
reprend. Vérifié — `3 → 6 essais après relance`.

---

## 7. Récupérer le résultat

La campagne écrit `best_hyperparams.json`, qui porte sa propre
provenance : espace de recherche, paramètres fixés, scénarios,
`lambda_cost`, le hash du commit et les arguments CLI.

```bash
scp machine:/chemin/best_hyperparams.json results/hyperparams/
python src/analyze_hyperparams.py
```

**Ne pas écraser `results/hyperparams/best_hyperparams.json` sans garder
l'ancien** : c'est l'entrée gelée dont dépendent tous les nombres publiés
avant la campagne. Le déplacer, pas le supprimer.

---

## 8. Après la campagne

Dans cet ordre — chaque étape conditionne la suivante :

1. **Relancer les campagnes** sur les nouveaux hyperparamètres : phase 4,
   T13, T26, T11b, T31. Ce sont elles qui lèvent les quatre tests rouges
   (seuils périmés) et les 4 DIFF de la table maîtresse.
2. **Republier ou justifier** les lignes qui ne se recalculent plus.
3. **Ajouter le témoin « mixeur seul »** aux campagnes H0b : sans lui,
   l'apport de l'hamiltonien n'est pas séparable d'une rotation de mixeur.
4. **Confronter à l'a priori enregistré** (`BRIEF_REPRISE.md` §6) : la
   prédiction est que la campagne **ne renversera pas** le verdict — H0a et
   H0b restent négatives, l'hypothèse vivante reste H2. Si elle le
   renverse, c'est un résultat plus fort encore, et il faudra le dire.

---

## 9. Si ça se passe mal

| symptôme | cause probable |
|---|---|
| `AttributeError … JournalFileBackend` | Optuna < 4 attendu ; voir D-136 |
| la campagne repart de 0 après coupure | la base n'est pas sur le volume persistant |
| deux workers, essais qui disparaissent | SQLite sur NFS — passer en `journal` |
| `database is locked` | trop de workers sur SQLite ; passer en Postgres ou `journal` |
| essais à `DIVERGENCE_PENALTY = 10.0` | le solveur diverge sur un jeu de paramètres ; c'est prévu, Optuna l'évite ensuite |
| le script refuse de partir | l'une des trois gardes ; lire le message, il dit laquelle |
