#!/bin/bash
# ============================================================
# Réoptimisation des hyperparamètres Q-HAS — 9 paramètres
#
# Écrit apres le test de configuration REELLE (2 essais, N=256,
# K_opt=30, shots=256, profondeur 4) qui a mesure 62 min/essai sur
# les 4 scenarios de la phase 1.
#
# La campagne est INTERRUPTIBLE : Optuna reprend depuis la base.
# Relancer la meme commande apres une coupure continue la ou elle
# s'est arretee. C'est ce qui la rend compatible avec des instances
# spot / preemptibles.
#
# Usage :
#   bash scripts/run_reoptimisation.sh [n_essais] [graine]
#
# Sur plusieurs coeurs : lancer N fois cette commande en parallele,
# toutes sur la MEME base. Optuna serialise les acces.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

N_ESSAIS="${1:-200}"
GRAINE="${2:-0}"
DB_DIR="$ROOT_DIR/results/hyperparams/reoptimisation"
mkdir -p "$DB_DIR"

# La base porte le nom de l'ETUDE qu'elle contient, lu dans `PHASES`.
# Une version anterieure ecrivait en dur `q_has_v3.db`, qui contenait
# l'etude `q_has_v2_phase1` : le fichier et son contenu ne disaient pas la
# meme chose, et la provenance ne se lisait plus (meme forme que D-22).
# Le dossier `reoptimisation/` distingue cette campagne des bases gelees ;
# le nom de fichier dit quelle etude, comme dans `optuna_studies/`.
ETUDE="$(cd "$ROOT_DIR/src" && python -c \
  "from train_hyperparams import PHASES; print(PHASES['phase1_composite']['study_name'])")"
DB="$DB_DIR/$ETUDE.db"

echo "============================================================"
echo "  REOPTIMISATION Q-HAS"
echo "  commit    : $(git -C "$ROOT_DIR" rev-parse --short HEAD)"
echo "  arbre     : $(git -C "$ROOT_DIR" diff --quiet && echo propre || echo MODIFIE)"
echo "  base      : $DB"
echo "  essais    : $N_ESSAIS   graine : $GRAINE"
echo "============================================================"

# --- garde 1 : l'arbre doit etre propre ---------------------------
# Une campagne lancee sur un arbre modifie ne peut etre attribuee a
# aucun commit : ses essais seraient irreproductibles.
if ! git -C "$ROOT_DIR" diff --quiet; then
    echo "ERREUR: arbre de travail modifie. Committer avant de lancer :"
    echo "        aucun hash ne decrirait ce qui a tourne."
    exit 1
fi

# --- garde 2 : les coefficients font leur travail -----------------
echo ""
echo "--- controle avant vol des coefficients ---"
python "$ROOT_DIR/study/common/preflight_coefficients.py"

# --- garde 3 : le perimetre est bien celui qu'on croit ------------
echo ""
echo "--- perimetre declare ---"
python "$ROOT_DIR/src/train_hyperparams.py" --print-space \
  | python -c "
import json,sys
d=json.load(sys.stdin); ss=d['search_space']
n=len(ss)
print(f'  {n} parametres explores :', ', '.join(ss))
print('  fixe :', d.get('fixed_params'))
assert n == 9, f'ATTENDU 9 parametres, vu {n} — perimetre inattendu'
"

# --- garde 4 : nom de fichier == etude contenue --------------------
# Sans elle, une campagne repart dans une base dont le nom ne designe plus
# le contenu — la forme exacte du defaut que D-22 a coute.
echo ""
echo "--- accord nom de base / etude ---"
if [ -f "$DB" ]; then
    python "$ROOT_DIR/scripts/inventaire_campagne.py" --racine "$DB_DIR"
else
    echo "  base absente : elle sera creee sous le nom '$ETUDE.db'"
fi

# --- garde 5 : pas d'essai fantome au demarrage --------------------
# Un worker tue (instance spot reprise, conteneur recycle) laisse son essai
# `RUNNING` pour toujours, et le total sur-compte le travail reellement
# fait. Le nettoyage refuse de tourner si un worker est encore vivant.
echo ""
echo "--- essais fantomes ---"
if [ -f "$DB" ]; then
    python "$ROOT_DIR/scripts/nettoyer_essais_fantomes.py" --base "$DB"
else
    echo "  base absente : rien a nettoyer"
fi

# --- la campagne --------------------------------------------------
echo ""
echo "--- campagne (interruptible : relancer pour reprendre) ---"
cd "$ROOT_DIR/src"
WORKER_TRIALS="$N_ESSAIS" \
OPTUNA_STORAGE="sqlite:///$DB" \
  python train_hyperparams.py --phase 1 --seed "$GRAINE"

echo ""
echo "Campagne terminee. Base : $DB"
echo "Analyser : python src/analyze_hyperparams.py"
