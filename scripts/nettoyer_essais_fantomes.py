#!/usr/bin/env python3
"""Marque `FAIL` les essais Optuna restes `RUNNING` sans worker vivant.

Pourquoi ce fichier existe
--------------------------
Un worker tue — instance spot reprise, conteneur recycle, OOM — laisse son
essai a l'etat `RUNNING` pour toujours. Optuna ne les reprend pas et ne les
compte pas comme echoues : le total d'essais sur-compte alors le travail
reellement fait. `results/hyperparams/PROVENANCE.md` en denombre **45** dans
les bases gelees (18 du bras classique, 24 du bras quantique, 3 laisses par
la reprise du 18 aout).

Mesure, pas supposition : `q_has_v2_phase1.db` annonce 202 essais et n'en a
que **178** de reellement termines. C'est la raison de la regle de
`BRIEF_REPRISE` §7 — *compter les essais COMPLETE, jamais le total*.

La securite qui compte
----------------------
Un essai `RUNNING` peut etre **vivant**. Nettoyer pendant qu'un worker
tourne detruirait un essai en cours. Le script refuse donc de s'executer
tant qu'un processus `train_hyperparams` existe, sauf `--force` explicite.
C'est le seul garde-fou qui empeche cet outil de faire des degats.

Usage
-----
    python scripts/nettoyer_essais_fantomes.py --base <chemin.db>
    python scripts/nettoyer_essais_fantomes.py --base <chemin.db> --dry-run
    python scripts/nettoyer_essais_fantomes.py --toutes          # sous results/hyperparams

Sortie : code 0 (idempotent — « rien a nettoyer » est un succes),
2 si un worker tourne encore.
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import subprocess
import sys

RACINE_DEFAUT = "results/hyperparams"


def _depot():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def workers_vivants():
    """PIDs des workers de campagne encore en vie.

    `pgrep -f` echoue avec le code 1 quand rien ne correspond : c'est le cas
    normal, pas une erreur. Si `pgrep` est absent, on ne peut rien affirmer
    et on le dit a l'appelant plutot que de supposer qu'il n'y en a pas.
    """
    try:
        out = subprocess.run(["pgrep", "-f", "train_hyperparams"],
                             capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if out.returncode not in (0, 1):
        return None
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def compter(chemin):
    cx = sqlite3.connect(f"file:{chemin}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in cx.execute(
            "select name from sqlite_master where type='table'")}
        if "trials" not in tables:
            return {}
        return dict(cx.execute(
            "select state, count(*) from trials group by state").fetchall())
    finally:
        cx.close()


def nettoyer(chemin, dry_run=False):
    """Renvoie (n_avant, n_apres) d'essais RUNNING pour cette base."""
    avant = compter(chemin).get("RUNNING", 0)
    if dry_run or avant == 0:
        return avant, avant
    cx = sqlite3.connect(chemin)
    try:
        cx.execute("update trials set state='FAIL' where state='RUNNING'")
        cx.commit()
    finally:
        cx.close()
    return avant, compter(chemin).get("RUNNING", 0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--base", help="une base .db")
    g.add_argument("--toutes", action="store_true",
                   help=f"toutes les bases sous {RACINE_DEFAUT}")
    p.add_argument("--racine", default=RACINE_DEFAUT)
    p.add_argument("--dry-run", action="store_true",
                   help="compte sans ecrire")
    p.add_argument("--force", action="store_true",
                   help="nettoie meme si un worker tourne (DANGEREUX : "
                        "un essai vivant serait marque FAIL)")
    a = p.parse_args(argv)

    if not a.dry_run and not a.force:
        pids = workers_vivants()
        if pids is None:
            print("ERREUR: impossible de verifier qu'aucun worker ne tourne "
                  "(pgrep indisponible). Relancer avec --dry-run, ou --force "
                  "si tu sais que la campagne est arretee.", file=sys.stderr)
            return 2
        if pids:
            print(f"ERREUR: {len(pids)} worker(s) train_hyperparams en vie "
                  f"({', '.join(map(str, pids))}). Un essai RUNNING peut etre "
                  "le leur. Arreter la campagne d'abord, ou --dry-run.",
                  file=sys.stderr)
            return 2

    if a.base:
        cibles = [a.base if os.path.isabs(a.base)
                  else os.path.join(_depot(), a.base)]
        if not os.path.exists(cibles[0]):
            print(f"ERREUR: base introuvable : {cibles[0]}", file=sys.stderr)
            return 2
    else:
        racine = (a.racine if os.path.isabs(a.racine)
                  else os.path.join(_depot(), a.racine))
        cibles = sorted(glob.glob(os.path.join(racine, "**", "*.db"),
                                  recursive=True))
        # Un balayage vide sort en vert et ne prouve rien.
        if not cibles:
            print(f"ERREUR: aucune base sous {racine} — "
                  "le balayage ne prouve rien.", file=sys.stderr)
            return 2

    total_avant = 0
    for chemin in cibles:
        avant, apres = nettoyer(chemin, dry_run=a.dry_run)
        total_avant += avant
        if avant:
            verbe = "a nettoyer" if a.dry_run else f"-> {apres} restants"
            print(f"  {os.path.relpath(chemin, _depot()):<52} "
                  f"{avant:>4} RUNNING {verbe}")
    if total_avant == 0:
        print("  rien a nettoyer : aucun essai RUNNING.")
    else:
        print(f"\n  {total_avant} essai(s) fantome(s) "
              f"{'detectes' if a.dry_run else 'marques FAIL'}.")
        if a.dry_run:
            print("  (--dry-run : rien n'a ete ecrit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
