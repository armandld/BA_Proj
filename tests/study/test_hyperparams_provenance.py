"""Les affirmations de `results/hyperparams/PROVENANCE.md` contre les bases.

Ce dossier est la seule entree du depot qui ne se regenere pas par une
commande. C'est precisement pour cela que ce qu'on en dit doit etre verifie
contre les fichiers plutot que recopie de memoire : un chiffre que rien ne
recalcule est un chiffre qui derive.

Les bases Optuna sont des artefacts geles ; ces tests les lisent en lecture
seule et confrontent chaque nombre du document a ce qu'elles contiennent.
"""

import datetime
import glob
import os
import sqlite3

import pytest



def _repo_root():
    """Racine du depot : on remonte jusqu'au dossier qui contient `src/`.

    Un calcul par `dirname` repete depend de la profondeur du fichier et
    casse au premier deplacement — souvent en silence, en pointant vers un
    chemin qui n'existe pas.
    """
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_REPO_ROOT = _repo_root()
_HP = os.path.join(_REPO_ROOT, "results", "hyperparams")
_STUDIES = os.path.join(_HP, "optuna_studies")
_DOC = os.path.join(_HP, "PROVENANCE.md")


def _dbs():
    return sorted(glob.glob(os.path.join(_STUDIES, "*.db")))


def _counts(path):
    """(nombre d'essais, heures de mur) pour une base ; (0, 0.0) si vide."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        n = con.execute("select count(*) from trials").fetchone()[0]
        if not n:
            return 0, 0.0, {}
        states = dict(con.execute(
            "select state, count(*) from trials group by state").fetchall())
        a, b = con.execute(
            "select min(datetime_start), max(datetime_complete) from trials"
        ).fetchone()
        hours = 0.0
        if a and b:
            hours = (datetime.datetime.fromisoformat(b)
                     - datetime.datetime.fromisoformat(a)).total_seconds() / 3600.0
        return n, hours, states
    finally:
        con.close()


@pytest.fixture(scope="module")
def survey():
    assert _dbs(), f"aucune base Optuna dans {_STUDIES}"
    return {os.path.basename(p): _counts(p) for p in _dbs()}


def test_only_phase1_was_ever_run(survey):
    """Le fait central : les phases 2 et 3 n'ont jamais tourne.

    `TrainHyperParam_v2.PHASES` declare 600 / 600 / 400 essais pour
    phase1_composite, phase2_complex et phase3_validation. Seules les bases
    phase1 contiennent des essais. Les hyperparametres publies viennent donc
    d'UNE phase, pas de la sequence complete — ce que le manuscrit doit dire.
    """
    non_empty = {k: v for k, v in survey.items() if v[0] > 0}
    assert set(non_empty) == {"classical_v2_phase1.db", "q_has_v2_phase1.db"}, (
        f"bases non vides mesurees : {sorted(non_empty)}")


def test_the_declared_trial_counts_were_not_reached(survey):
    """PHASES declare 600 essais pour phase1 ; il y en a nettement moins."""
    for db, declared in (("classical_v2_phase1.db", 600),
                         ("q_has_v2_phase1.db", 600)):
        n = survey[db][0]
        assert n < declared, (
            f"{db} : {n} essais, or la declaration de PHASES ({declared}) "
            "n'est plus au-dessus — mettre PROVENANCE.md a jour")


def test_the_total_trial_count_matches_the_document(survey):
    total = sum(v[0] for v in survey.values())
    assert total == 345, (
        f"{total} essais au total ; PROVENANCE.md annonce 345. Si les bases "
        "ont change, corriger le document, pas ce test.")


def _cpu_hours(path):
    """Somme des durees d'essai : le COUT, par opposition au temps de mur."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "select datetime_start, datetime_complete from trials "
            "where state='COMPLETE' and datetime_complete is not null"
        ).fetchall()
    finally:
        con.close()
    return sum((datetime.datetime.fromisoformat(b)
                - datetime.datetime.fromisoformat(a)).total_seconds()
               for a, b in rows) / 3600.0


def test_the_wall_time_is_about_two_days(survey):
    """~47 h de mur : 16.6 h classique + 30.4 h quantique."""
    hours = {k: v[1] for k, v in survey.items() if v[0] > 0}
    total = sum(hours.values())
    assert 40.0 < total < 55.0, (
        f"duree de mur mesuree {total:.1f} h, repartie {hours}")


def test_the_cpu_cost_is_about_nine_days_not_two():
    """Le mur n'est PAS le cout : les essais ont tourne en parallele.

    Corrige une erreur de cadrage. Le temps de mur (47 h) avait ete oppose
    a la « semaine » annoncee pour conclure qu'une relance coutait peu. Les
    essais tournaient jusqu'a 9 de front : 224 h de CPU, soit 9.3 jours
    mono-coeur. L'annonce d'origine etait donc juste en temps processeur,
    et c'est ce chiffre qui gouverne le cout d'une relance.
    """
    cpu = {os.path.basename(p): _cpu_hours(p) for p in _dbs()
           if _counts(p)[0] > 0}
    total = sum(cpu.values())
    assert 200.0 < total < 250.0, (
        f"cout CPU mesure {total:.1f} h, reparti {cpu}")


def test_the_trials_really_ran_in_parallel(survey):
    """Le rapport CPU/mur est la preuve du parallelisme."""
    cpu = sum(_cpu_hours(p) for p in _dbs() if _counts(p)[0] > 0)
    wall = sum(v[1] for v in survey.values() if v[0] > 0)
    assert cpu / wall > 3.0, (
        f"parallelisme {cpu / wall:.1f}x : si les essais devenaient "
        "sequentiels, le mur redeviendrait une mesure du cout")


def test_the_median_trial_cost_is_recorded():
    """Le cout par essai est ce qui permet de chiffrer une relance ciblee."""
    import numpy as np

    for name, lo, hi in (("classical_v2_phase1.db", 1800, 2400),
                         ("q_has_v2_phase1.db", 3000, 3700)):
        path = os.path.join(_STUDIES, name)
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "select datetime_start, datetime_complete from trials "
                "where state='COMPLETE' and datetime_complete is not null"
            ).fetchall()
        finally:
            con.close()
        med = float(np.median([
            (datetime.datetime.fromisoformat(b)
             - datetime.datetime.fromisoformat(a)).total_seconds()
            for a, b in rows]))
        assert lo < med < hi, (
            f"{name} : cout median par essai {med:.0f} s, hors de "
            f"[{lo}, {hi}] — mettre PROVENANCE.md a jour")


def test_some_trials_were_left_running(survey):
    """Des essais RUNNING signalent une campagne interrompue, pas terminee."""
    running = {k: v[2].get("RUNNING", 0) for k, v in survey.items() if v[0]}
    assert sum(running.values()) > 0, (
        "aucun essai RUNNING : si la campagne s'est terminee proprement, "
        "corriger PROVENANCE.md qui la decrit comme interrompue")


def test_the_document_states_the_measured_numbers():
    """Le document doit porter les chiffres, pas une impression."""
    doc = open(_DOC, encoding="utf-8").read()
    for token in ("345", "16.6", "30.4", "47", "224", "9.3", "56 min"):
        assert token in doc, (
            f"PROVENANCE.md ne mentionne pas {token} ; les nombres verifies "
            "doivent figurer dans le document qu'ils decrivent")
    assert "temps processeur" in doc, (
        "PROVENANCE.md doit distinguer le cout CPU du temps de mur : c'est "
        "la confusion des deux qui avait fait sous-estimer une relance")


def test_the_published_hyperparameters_come_from_phase1_only():
    """best_hyperparams.json ne doit pas reference de phase jamais executee."""
    import json
    d = json.load(open(os.path.join(_HP, "best_hyperparams.json"),
                       encoding="utf-8"))
    for arm in ("quantum", "classical"):
        phases = set(d["best_per_phase"][arm])
        assert phases == {"phase1"}, (
            f"best_per_phase.{arm} reference {sorted(phases)} alors que "
            "seule phase1 a tourne")
