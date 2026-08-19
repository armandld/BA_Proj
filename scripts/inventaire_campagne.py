#!/usr/bin/env python3
"""Inventaire des bases Optuna, et garde sur l'accord nom de fichier / etude.

Pourquoi ce fichier existe
--------------------------
La reoptimisation a ete lancee dans une base nommee `q_has_v3.db` qui
contient l'etude `q_has_v2_phase1`. Le fichier et son contenu ne disaient
pas la meme chose. C'est exactement la forme de defaut que D-22 a coute :
un reglage dont la provenance ne se lit plus, parce que le nom ne designe
plus ce qu'il contient.

Le garde est la quatrieme question de `CODE_REVIEW.md` — *deux chemins
censes coincider coincident-ils encore ?* — appliquee au stockage : le nom
du fichier et le `study_name` qu'il porte sont deux ecritures de la meme
chose, et rien ne les tenait ensemble.

Ce que le garde verifie
-----------------------
1. Toute base non vide porte **exactement une** etude.
2. Le nom de cette etude est **le basename du fichier**.

Une base vide (0 etude, 0 essai) n'est pas une violation : le depot en
contient 8, documentees par `results/hyperparams/PROVENANCE.md` — les
phases 2, 3 et 1b n'ont jamais tourne. Elles sont signalees, pas comptees
comme faute.

Les essais `RUNNING` sont rapportes parce qu'ils sont trompeurs : un worker
tue laisse son essai `RUNNING` pour toujours, et le total d'essais
sur-compte alors le travail reellement fait. `PROVENANCE.md` en denombre 45
dans les bases gelees. Nettoyage : `scripts/nettoyer_essais_fantomes.py`.

Usage
-----
    python scripts/inventaire_campagne.py                 # tout le depot
    python scripts/inventaire_campagne.py --racine results/hyperparams/reoptimisation
    python scripts/inventaire_campagne.py --json          # sortie machine

Sortie : code 0 si l'invariant tient, 1 s'il est viole.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
import sys

RACINE_DEFAUT = "results/hyperparams"


def _depot():
    d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return d


def lire_base(chemin):
    """(etudes, etats) d'une base Optuna, en lecture seule.

    Une base illisible remonte l'exception dans `etudes` plutot que de
    faire passer le balayage : un fichier qu'on ne sait pas lire n'est pas
    un fichier conforme.
    """
    try:
        cx = sqlite3.connect(f"file:{chemin}?mode=ro", uri=True)
    except sqlite3.Error as err:
        return [f"<illisible: {type(err).__name__}>"], {}
    try:
        tables = {r[0] for r in cx.execute(
            "select name from sqlite_master where type='table'")}
        etudes = ([r[0] for r in cx.execute("select study_name from studies")]
                  if "studies" in tables else [])
        etats = (dict(cx.execute(
            "select state, count(*) from trials group by state").fetchall())
            if "trials" in tables else {})
        return etudes, etats
    except sqlite3.Error as err:
        return [f"<illisible: {type(err).__name__}>"], {}
    finally:
        cx.close()


def inventorier(racine):
    """Une entree par base trouvee sous `racine`, triee par chemin."""
    entrees = []
    motif = os.path.join(racine, "**", "*.db")
    for chemin in sorted(glob.glob(motif, recursive=True)):
        etudes, etats = lire_base(chemin)
        base = os.path.splitext(os.path.basename(chemin))[0]
        vide = not etudes and not etats
        conforme = vide or (len(etudes) == 1 and etudes[0] == base)
        entrees.append({
            "chemin": os.path.relpath(chemin, _depot()),
            "fichier": base,
            "etudes": etudes,
            "complete": etats.get("COMPLETE", 0),
            "running": etats.get("RUNNING", 0),
            "fail": etats.get("FAIL", 0),
            "vide": vide,
            "conforme": conforme,
        })
    return entrees


def _rapport(entrees):
    largeur = max([len(e["chemin"]) for e in entrees] + [20])
    entete = (f"  {'base':<{largeur}} {'etude':<24} "
              f"{'COMPLETE':>9} {'RUNNING':>8} {'FAIL':>6}")
    print(entete)
    print("  " + "-" * (len(entete) - 2))
    for e in entrees:
        etude = e["etudes"][0] if e["etudes"] else ("(vide)" if e["vide"] else "(aucune)")
        drapeau = "" if e["conforme"] else "   <-- NOM != ETUDE"
        print(f"  {e['chemin']:<{largeur}} {etude:<24} "
              f"{e['complete']:>9} {e['running']:>8} {e['fail']:>6}{drapeau}")
    print("  " + "-" * (len(entete) - 2))
    tc = sum(e["complete"] for e in entrees)
    tr = sum(e["running"] for e in entrees)
    vides = sum(1 for e in entrees if e["vide"])
    print(f"  {'TOTAL':<{largeur}} {'':<24} {tc:>9} {tr:>8}")
    print()
    print(f"  bases : {len(entrees)}   vides : {vides}   "
          f"essais COMPLETE : {tc}   fantomes RUNNING : {tr}")
    if tr:
        print("  -> les RUNNING sont des essais qu'aucun worker ne finira ; "
              "compter les COMPLETE.")
        print("     nettoyage : python scripts/nettoyer_essais_fantomes.py "
              "--base <chemin>")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--racine", default=RACINE_DEFAUT,
                   help=f"dossier balaye (defaut : {RACINE_DEFAUT})")
    p.add_argument("--json", action="store_true", help="sortie machine")
    a = p.parse_args(argv)

    racine = a.racine if os.path.isabs(a.racine) else os.path.join(_depot(), a.racine)
    entrees = inventorier(racine)

    # Un balayage vide doit crier : sans cette garde, un chemin faux rend
    # « tout va bien » avec le code 0.
    if not entrees:
        print(f"ERREUR: aucune base trouvee sous {racine} — "
              "le balayage ne prouve rien.", file=sys.stderr)
        return 2

    if a.json:
        print(json.dumps(entrees, indent=2))
    else:
        print("=" * 78)
        print("  Inventaire des bases Optuna")
        print("=" * 78)
        _rapport(entrees)

    fautives = [e for e in entrees if not e["conforme"]]
    if fautives:
        print(file=sys.stderr)
        for e in fautives:
            print(f"ERREUR: {e['chemin']} porte l'etude {e['etudes']} "
                  f"mais se nomme '{e['fichier']}'.", file=sys.stderr)
        print("Le nom d'un fichier de campagne doit etre celui de l'etude "
              "qu'il contient — sinon sa provenance ne se lit plus (D-22).",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
