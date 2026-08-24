"""Le stockage de campagne dit-il ce qu'il contient, et sait-on compter ?

Deux defauts que ces tests epinglent, tous deux rencontres le 18 aout 2026
en lancant la reoptimisation :

1. **Nom de fichier != etude contenue.** Le lanceur ecrivait en dur
   `q_has_v3.db`, base qui contenait l'etude `q_has_v2_phase1`. Un reglage
   dont le fichier ne designe plus le contenu est un reglage sans
   provenance : c'est la forme exacte de D-22.

2. **Essais fantomes.** Trois workers tues par un recyclage de conteneur
   ont laisse trois essais `RUNNING` qu'aucun worker ne finira. Les bases
   gelees en portent 45 (`PROVENANCE.md`), et `q_has_v2_phase1.db` annonce
   202 essais pour 178 reellement termines.

Les tests construisent leurs propres bases SQLite : ils ne dependent
d'aucun artefact du depot et ne peuvent donc pas devenir verts par
disparition de leur entree.
"""
import importlib.util
import os
import sqlite3
import subprocess
import sys

import pytest

_RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS = os.path.join(_RACINE, "scripts")


def _charger(nom):
    """Charge un script de `scripts/` depuis son chemin.

    `scripts/` n'est pas une racine d'import de la suite — le depot y
    reference ses fichiers par chemin (cf. `test_extract_best_hyperparams_*`),
    et `tests/test_suite_integrity.py` signale tout `import` de module qui ne
    resout pas sous les racines declarees. On charge donc explicitement,
    sans toucher a `sys.path`.
    """
    chemin = os.path.join(_SCRIPTS, nom + ".py")
    spec = importlib.util.spec_from_file_location(nom, chemin)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inv = _charger("inventaire_campagne")
net = _charger("nettoyer_essais_fantomes")


# ------------------------------------------------------------------
#  fabrique de bases Optuna minimales
# ------------------------------------------------------------------
def _base(chemin, study_name, etats=()):
    """Base au schema Optuna reduit aux deux tables que les outils lisent."""
    cx = sqlite3.connect(str(chemin))
    cx.execute("create table studies (study_id integer primary key, study_name text)")
    cx.execute("create table trials (trial_id integer primary key, state text)")
    if study_name is not None:
        cx.execute("insert into studies (study_name) values (?)", (study_name,))
    for i, etat in enumerate(etats):
        cx.execute("insert into trials (trial_id, state) values (?, ?)", (i, etat))
    cx.commit()
    cx.close()
    return str(chemin)


# ------------------------------------------------------------------
#  1. l'invariant nom == etude
# ------------------------------------------------------------------
def test_nom_conforme_passe(tmp_path):
    _base(tmp_path / "q_has_v2_phase1.db", "q_has_v2_phase1", ["COMPLETE"] * 3)
    assert inv.main(["--racine", str(tmp_path)]) == 0


def test_nom_different_de_l_etude_echoue(tmp_path):
    """Le cas reellement rencontre : q_has_v3.db contenant q_has_v2_phase1."""
    _base(tmp_path / "q_has_v3.db", "q_has_v2_phase1", ["COMPLETE"])
    assert inv.main(["--racine", str(tmp_path)]) == 1


def test_base_vide_est_toleree(tmp_path):
    """8 bases du depot sont vides et documentees : ce n'est pas une faute."""
    _base(tmp_path / "q_has_v2_phase2.db", None)
    assert inv.main(["--racine", str(tmp_path)]) == 0


def test_deux_etudes_dans_une_base_echoue(tmp_path):
    chemin = _base(tmp_path / "double.db", "double", ["COMPLETE"])
    cx = sqlite3.connect(chemin)
    cx.execute("insert into studies (study_name) values ('autre')")
    cx.commit(); cx.close()
    assert inv.main(["--racine", str(tmp_path)]) == 1


def test_balayage_vide_crie(tmp_path):
    """Un dossier sans base ne doit pas rendre « tout va bien »."""
    assert inv.main(["--racine", str(tmp_path / "inexistant")]) == 2


def test_les_bases_du_depot_respectent_l_invariant():
    """Le depot lui-meme, pas une fixture — c'est la garde qui sert.

    Elle echouerait si une campagne future reintroduisait une base dont le
    nom ne designe pas son etude.
    """
    entrees = inv.inventorier(os.path.join(_RACINE, "results", "hyperparams"))
    assert entrees, "aucune base trouvee : le balayage ne prouve rien"
    fautives = [e["chemin"] for e in entrees if not e["conforme"]]
    assert not fautives, f"bases dont le nom ne designe pas l'etude : {fautives}"


# ------------------------------------------------------------------
#  2. le nettoyage des fantomes
# ------------------------------------------------------------------
def test_running_devient_fail(tmp_path):
    chemin = _base(tmp_path / "c.db", "c", ["COMPLETE", "RUNNING", "RUNNING"])
    avant, apres = net.nettoyer(chemin)
    assert (avant, apres) == (2, 0)
    assert net.compter(chemin)["COMPLETE"] == 1, "un COMPLETE a ete touche"


def test_dry_run_n_ecrit_rien(tmp_path):
    chemin = _base(tmp_path / "c.db", "c", ["RUNNING", "RUNNING"])
    avant, apres = net.nettoyer(chemin, dry_run=True)
    assert (avant, apres) == (2, 2)
    assert net.compter(chemin)["RUNNING"] == 2


def test_nettoyage_idempotent(tmp_path):
    chemin = _base(tmp_path / "c.db", "c", ["RUNNING"])
    net.nettoyer(chemin)
    assert net.nettoyer(chemin) == (0, 0)


def test_refuse_si_un_worker_tourne(tmp_path, monkeypatch):
    """Le garde-fou qui empeche l'outil de detruire un essai vivant."""
    chemin = _base(tmp_path / "c.db", "c", ["RUNNING"])
    monkeypatch.setattr(net, "workers_vivants", lambda: [4242])
    assert net.main(["--base", chemin]) == 2
    assert net.compter(chemin)["RUNNING"] == 1, "un essai vivant a ete marque FAIL"


def test_refuse_si_pgrep_indisponible(tmp_path, monkeypatch):
    """Ne pas pouvoir verifier n'est pas la meme chose que « personne »."""
    chemin = _base(tmp_path / "c.db", "c", ["RUNNING"])
    monkeypatch.setattr(net, "workers_vivants", lambda: None)
    assert net.main(["--base", chemin]) == 2


def test_force_passe_outre(tmp_path, monkeypatch):
    chemin = _base(tmp_path / "c.db", "c", ["RUNNING"])
    monkeypatch.setattr(net, "workers_vivants", lambda: [4242])
    assert net.main(["--base", chemin, "--force"]) == 0
    assert net.compter(chemin).get("RUNNING", 0) == 0


def test_base_absente_ne_passe_pas_en_silence(tmp_path):
    assert net.main(["--base", str(tmp_path / "pas_la.db")]) == 2


# ------------------------------------------------------------------
#  3. le lanceur ne code plus le nom en dur
# ------------------------------------------------------------------
def _chemin_base_du_lanceur():
    """Evalue les lignes du lanceur qui calculent `DB`, et rend ce chemin.

    On n'inspecte PAS le texte du script : une assertion sur le source est
    un faux vert des qu'un commentaire mentionne la chaine cherchee — c'est
    la famille D-123 a D-131. On execute donc les lignes reelles.
    """
    src = open(os.path.join(_SCRIPTS, "run_reoptimisation.sh"), encoding="utf-8").read()
    lignes = [l for l in src.splitlines()
              if l.startswith(("ETUDE=", "DB=", "  \"from train_hyperparams",
                               '  "from train_hyperparams'))]
    assert any(l.startswith("ETUDE=") for l in lignes), \
        "le lanceur ne derive plus le nom d'etude"
    programme = (f'set -euo pipefail\nROOT_DIR="{_RACINE}"\n'
                 f'DB_DIR="$ROOT_DIR/results/hyperparams/reoptimisation"\n'
                 + "\n".join(lignes) + '\necho "$DB"\n')
    out = subprocess.run(["bash", "-c", programme],
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_le_lanceur_nomme_la_base_comme_son_etude():
    """Le defaut d'origine : `q_has_v3.db` contenant `q_has_v2_phase1`."""
    etude = subprocess.run(
        [sys.executable, "-c",
         "from train_hyperparams import PHASES; "
         "print(PHASES['phase1_composite']['study_name'])"],
        cwd=os.path.join(_RACINE, "src"), capture_output=True, text=True, timeout=180)
    assert etude.returncode == 0, etude.stderr
    attendu = etude.stdout.strip()

    base = os.path.splitext(os.path.basename(_chemin_base_du_lanceur()))[0]
    assert base == attendu, (
        f"le lanceur ecrirait dans '{base}.db' alors que l'etude creee "
        f"s'appelle '{attendu}' : la provenance ne se lirait plus (D-22)")


def test_la_base_du_lanceur_reste_sous_reoptimisation():
    """Elle ne doit pas retomber dans les bases gelees et les ecraser."""
    chemin = _chemin_base_du_lanceur()
    assert os.path.basename(os.path.dirname(chemin)) == "reoptimisation", chemin
