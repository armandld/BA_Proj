#!/bin/bash
# ============================================================
#  Soumission de la reoptimisation sur un cluster partage
#
#  `MODE_EMPLOI_CAMPAGNE.md` §4 annoncait « puis un script de soumission
#  qui lance run_reoptimisation.sh en reseau de taches » sans le fournir.
#  Le voici, pour PBS Pro (Imperial) et pour Slurm.
#
#  Ce script N'EST PAS un job : il ECRIT le fichier de job et dit la
#  commande a lancer. On le lit avant de soumettre — 206 h CPU ne se
#  soumettent pas a l'aveugle.
#
#  Usage :
#    bash scripts/soumettre_campagne.sh pbs   [n_workers] [heures] [n_essais]
#    bash scripts/soumettre_campagne.sh slurm [n_workers] [heures] [n_essais]
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ORDONNANCEUR="${1:-}"
N_WORKERS="${2:-8}"
HEURES="${3:-24}"
N_ESSAIS="${4:-200}"

if [[ "$ORDONNANCEUR" != "pbs" && "$ORDONNANCEUR" != "slurm" ]]; then
    echo "usage: $0 {pbs|slurm} [n_workers] [heures] [n_essais]" >&2
    exit 2
fi

# Un essai coute ~62 min (mesure, pas estime : cf. run_reoptimisation.sh).
# Chaque worker consomme des essais jusqu'a epuisement de son plafond ou du
# temps de mur. On dimensionne le plafond par worker sur le temps demande.
ESSAIS_PAR_WORKER=$(( (HEURES * 60) / 62 ))
[ "$ESSAIS_PAR_WORKER" -lt 1 ] && ESSAIS_PAR_WORKER=1
CAPACITE=$(( ESSAIS_PAR_WORKER * N_WORKERS ))

JOURNAL_DIR="\$HOME/q_has_journal"
SORTIE="$ROOT_DIR/scripts/job_campagne_${ORDONNANCEUR}.sh"

cat > "$SORTIE" <<JOB
#!/bin/bash
$( [ "$ORDONNANCEUR" = "pbs" ] && cat <<'PBS'
#PBS -N q_has_reopt
#PBS -l select=1:ncpus=1:mem=2gb
#PBS -j oe
PBS
)
$( [ "$ORDONNANCEUR" = "pbs" ] && echo "#PBS -l walltime=${HEURES}:00:00" )
$( [ "$ORDONNANCEUR" = "pbs" ] && echo "#PBS -J 1-${N_WORKERS}" )
$( [ "$ORDONNANCEUR" = "slurm" ] && cat <<SLURM
#SBATCH --job-name=q_has_reopt
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=${HEURES}:00:00
#SBATCH --array=1-${N_WORKERS}
#SBATCH --output=q_has_reopt_%A_%a.out
SLURM
)
set -euo pipefail

# --- le code, au commit exact -----------------------------------
cd "$ROOT_DIR"
echo "commit : \$(git rev-parse --short HEAD)"
echo "worker : \${PBS_ARRAY_INDEX:-\${SLURM_ARRAY_TASK_ID:-1}}"

# --- l'environnement --------------------------------------------
# Adapter cette ligne a ton cluster (module load / conda / venv).
source .venv/bin/activate 2>/dev/null || true
python -c "import optuna, qiskit, numpy" || {
    echo "ERREUR: environnement incomplet — pip install -r requirements.txt"; exit 1; }

# --- le stockage -------------------------------------------------
# Journal, PAS SQLite : plusieurs machines sur un systeme de fichiers
# partage. SQLite sur NFS corrompt la base (MODE_EMPLOI §3).
export OPTUNA_JOURNAL="$JOURNAL_DIR"
mkdir -p "\$OPTUNA_JOURNAL"

# --- les fantomes du redemarrage precedent -----------------------
# Un worker tue par une preemption ou un depassement de walltime laisse son
# essai a RUNNING pour toujours. On ne nettoie QUE si plus rien ne tourne :
# le script refuse tout seul tant qu'un worker est vivant.
python scripts/nettoyer_essais_fantomes.py \\
    --racine results/hyperparams/reoptimisation --toutes || true

# --- la campagne (interruptible : resoumettre pour reprendre) -----
WORKER_TRIALS=$ESSAIS_PAR_WORKER \\
  bash scripts/run_reoptimisation.sh $N_ESSAIS 0
JOB

# Les branches inactives laissent des lignes vides entre le shebang et les
# directives. Slurm comme PBS cessent de lire les directives a la premiere
# ligne qui n'est ni un commentaire ni du vide ; on ne prend pas le risque
# et on colle l'entete au shebang.
python3 - "$SORTIE" <<'NETTOYAGE'
import sys
chemin = sys.argv[1]
lignes = open(chemin, encoding="utf-8").read().splitlines()
sortie, entete_finie = [lignes[0]], False
for ligne in lignes[1:]:
    if not entete_finie:
        if not ligne.strip():
            continue                      # vide dans l'entete : on jette
        if not ligne.startswith("#"):
            entete_finie = True
            sortie.append("")             # une seule respiration
    sortie.append(ligne)
open(chemin, "w", encoding="utf-8").write("\n".join(sortie) + "\n")
NETTOYAGE

chmod +x "$SORTIE"

echo "============================================================"
echo "  Fichier de job ecrit : $SORTIE"
echo "============================================================"
echo "  ordonnanceur      : $ORDONNANCEUR"
echo "  workers           : $N_WORKERS   (1 coeur, 2 Go chacun)"
echo "  walltime          : ${HEURES} h"
echo "  essais par worker : $ESSAIS_PAR_WORKER   (a ~62 min l'essai)"
echo "  capacite totale   : $CAPACITE essais"
if [ "$CAPACITE" -lt "$N_ESSAIS" ]; then
    echo "  ATTENTION : capacite ($CAPACITE) < cible ($N_ESSAIS)."
    echo "              Il faudra resoumettre — la campagne reprend seule."
fi
echo ""
echo "  AVANT de soumettre, sur un noeud interactif :"
echo "    bash scripts/repetition_campagne.sh journal"
echo "    -> attendu : REPETITION REUSSIE"
echo ""
echo "  Puis :"
if [ "$ORDONNANCEUR" = "pbs" ]; then
    echo "    qsub $SORTIE"
    echo "    qstat -t          # suivre le reseau de taches"
else
    echo "    sbatch $SORTIE"
    echo "    squeue --me"
fi
echo ""
echo "  Surveiller (compter les COMPLETE, jamais le total) :"
echo "    python scripts/inventaire_campagne.py --racine results/hyperparams/reoptimisation"
