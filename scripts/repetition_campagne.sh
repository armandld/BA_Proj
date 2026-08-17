#!/bin/bash
# ============================================================
# REPETITION de la campagne — a lancer AVANT de louer des coeurs
#
# Verifie les trois proprietes dont depend la location, sur la
# configuration MINIATURE (N=32, quelques secondes par essai) :
#
#   1. REPRISE   : tuer un worker puis relancer continue la campagne
#                  au lieu de la recommencer. C'est ce qui rend les
#                  instances spot / preemptibles utilisables — donc
#                  ce qui divise la facture par ~3.
#   2. PARALLELE : deux workers sur la MEME base ne se corrompent pas
#                  et ne dupliquent pas les essais.
#   3. SURVIE    : la base contient bien les essais apres coup, et le
#                  compte des COMPLETE est celui qu'on croit.
#
# Ces trois proprietes ne dependent PAS de la taille des essais : les
# verifier en miniature les verifie tout court. Ce qui depend de la
# taille, c'est le cout, mesure separement (62 min/essai a N=256).
#
# Usage :
#   bash scripts/repetition_campagne.sh [backend]
#     backend = sqlite (defaut) | journal
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND="${1:-sqlite}"
BANC="$(mktemp -d)"
trap 'rm -rf "$BANC"' EXIT

echo "============================================================"
echo "  REPETITION DE CAMPAGNE — backend : $BACKEND"
echo "  banc d'essai : $BANC"
echo "============================================================"

case "$BACKEND" in
  sqlite)  export OPTUNA_STORAGE="sqlite:///$BANC/repetition.db" ;;
  journal) export OPTUNA_JOURNAL="$BANC/journal" ; mkdir -p "$BANC/journal" ;;
  *) echo "backend inconnu : $BACKEND (sqlite | journal)"; exit 2 ;;
esac

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

python - "$BACKEND" "$BANC" <<'PYEOF'
import os, sys, time
import multiprocessing as mp

backend, banc = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import train_hyperparams as TH

#: Miniature : ce qui est verifie ici est la MECANIQUE de la campagne
#: (reprise, parallelisme, survie), pas la physique. Les memes valeurs
#: que le banc de fumee de `tests/pipeline/test_train_hyperparams_smoke.py`.
def _tiny(cfg, cle):
    return {**cfg, "N": 32, "T_MAX": 0.06, "T_START": 0.02, "DT": 5e-3,
            "HYBRID_DT": 0.02, "K_opt": 3, "shots": 32,
            "max_depth_override": 1, "study_name": f"dns_{cle}"}

ETUDE = "repetition_phase1"

def _storage():
    if backend == "sqlite":
        return optuna.storages.RDBStorage(url=os.environ["OPTUNA_STORAGE"])
    chemin = os.path.join(os.environ["OPTUNA_JOURNAL"], f"{ETUDE}.log")
    # Meme resolution d'API que `train_hyperparams._get_storage` (D-136) :
    # la repetition doit exercer le chemin REEL, pas une variante.
    try:
        from optuna.storages.journal import (JournalFileBackend,
                                             JournalFileOpenLock)
    except ImportError:
        JournalFileBackend = optuna.storages.JournalFileBackend
        JournalFileOpenLock = optuna.storages.JournalFileOpenLock
    return optuna.storages.JournalStorage(
        JournalFileBackend(chemin, lock_obj=JournalFileOpenLock(chemin)))

def _objectif(trial):
    """Un essai bon marche qui EXERCE quand meme l'echantillonnage reel."""
    hp = TH.suggest_hyperparams(trial)
    # somme ponderee : deterministe, mais differente a chaque tirage,
    # donc deux essais identiques se verraient.
    return sum(float(v) for v in hp.values())

def _worker(n_essais):
    st = optuna.load_study(study_name=ETUDE, storage=_storage())
    st.optimize(_objectif, n_trials=n_essais)

def _etude():
    return optuna.load_study(study_name=ETUDE, storage=_storage())

def _complets(st):
    return [t for t in st.trials
            if t.state == optuna.trial.TrialState.COMPLETE]

echecs = []
optuna.create_study(study_name=ETUDE, storage=_storage(),
                    direction="minimize", load_if_exists=True)

# ── 1. REPRISE ────────────────────────────────────────────────────
_worker(3)
apres_1 = len(_complets(_etude()))
_worker(3)                       # relance : doit CONTINUER, pas recommencer
apres_2 = len(_complets(_etude()))
print(f"\n[1] REPRISE    : {apres_1} essais -> {apres_2} apres relance")
if apres_2 != apres_1 + 3:
    echecs.append(f"reprise : attendu {apres_1+3} essais, vu {apres_2}. "
                  "La campagne RECOMMENCE au lieu de reprendre — sur une "
                  "instance spot, chaque interruption perdrait tout.")

# ── 2. PARALLELE ──────────────────────────────────────────────────
avant = len(_complets(_etude()))
procs = [mp.Process(target=_worker, args=(3,)) for _ in range(2)]
for p in procs: p.start()
for p in procs: p.join()
st = _etude(); apres = len(_complets(st))
print(f"[2] PARALLELE  : {avant} -> {apres} avec 2 workers simultanes")
if apres != avant + 6:
    echecs.append(f"parallele : attendu {avant+6}, vu {apres}. Des essais "
                  "se sont perdus ou ecrases entre workers.")

numeros = [t.number for t in st.trials]
if len(numeros) != len(set(numeros)):
    echecs.append("parallele : numeros d'essai DUPLIQUES — la base est "
                  "corrompue par l'acces concurrent.")

# ── 3. SURVIE + integrite ─────────────────────────────────────────
comp = _complets(st)
running = [t for t in st.trials
           if t.state == optuna.trial.TrialState.RUNNING]
print(f"[3] SURVIE     : {len(comp)} COMPLETE, {len(running)} RUNNING, "
      f"{len(st.trials)} au total")
if any(t.value is None or t.value != t.value for t in comp):
    echecs.append("des essais COMPLETE portent une valeur nulle ou NaN.")
if len({tuple(sorted(t.params.items())) for t in comp}) != len(comp):
    echecs.append("deux essais COMPLETE ont EXACTEMENT les memes "
                  "parametres : l'echantillonnage ne varie pas.")
if len(comp) == 0:
    echecs.append("BALAYAGE VIDE : aucun essai complet, la repetition "
                  "n'a rien verifie.")

print("\n" + "=" * 60)
if echecs:
    print("  REPETITION ECHOUEE — ne pas louer :")
    for e in echecs: print(f"    - {e}")
    sys.exit(1)
print(f"  REPETITION REUSSIE ({backend}) : reprise, parallelisme et")
print("  integrite verifies. Ce backend convient a la location.")
print("=" * 60)
PYEOF
