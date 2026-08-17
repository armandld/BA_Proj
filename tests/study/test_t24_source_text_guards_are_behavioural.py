"""D-123/D-124/D-125 — trois gardes de `test_t24_leak_free.py` gardaient un
COMPORTEMENT par une recherche de chaine dans le SOURCE.

Meme famille que D-114 et D-115. Ce qui distingue celle-ci : les trois sites
protegent des invariants de PROVENANCE — ne pas melanger des tirages issus
d'une autre configuration, ne pas taire qu'un tirage vient d'un autre
processus, ne pas publier une execution interrompue comme si elle etait
complete. Un faux vert y produit exactement ce que ce depot traque : une
moyenne d'apparence normale, calculee sur des donnees qui ne vont pas
ensemble.

Chacun a ete mesure par mutation — casser le comportement en laissant la
chaine cherchee en place — et chacune est restee VERTE :

| defaut | mutation | ancienne suite |
|---|---|---|
| D-123 | un `and` du test de configuration devient `or` | 26 passed |
| D-124 | les deux ecritures `out[...]` supprimees, les noms survivant dans un commentaire | 26 passed |
| D-125 | `h4_transfer_summary.py` cesse de filtrer `status == "partial"` | 38 passed |

Les tests d'origine ne sont pas retires : ils ne sont pas faux, seulement
fragiles. Ceux-ci ajoutent le garde comportemental qui manquait.
"""

import ast
import json
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


class _Args:
    fold = "kh"
    mode = "leak-free"
    repeats = 5
    matched_reference = True


def _prev(**over):
    d = {"status": "partial", "fold": "kh", "mode": "leak-free",
         "cli_args": {"repeats": 5, "matched_reference": True}}
    d.update({k: v for k, v in over.items() if k not in ("repeats",
                                                         "matched_reference")})
    for k in ("repeats", "matched_reference"):
        if k in over:
            d["cli_args"][k] = over[k]
    return d


# ── D-123 : la decision de reprise, appelee et non lue ───────────────

def test_a_matching_checkpoint_is_reused():
    """Contrôle positif : sans lui, un garde toujours-faux passerait."""
    from h4_unseen_conditions import checkpoint_is_reusable
    assert checkpoint_is_reusable(_prev(), _Args()) is True


@pytest.mark.parametrize("champ,valeur", [
    ("fold", "ot"),
    ("mode", "leaky"),
    ("repeats", 3),
    ("matched_reference", False),
])
def test_a_checkpoint_from_another_configuration_is_refused(champ, valeur):
    """Chacun des quatre champs, seul, doit suffire a refuser la reprise.

    C'est ce que la recherche de chaine ne voyait pas : avec un `or` a la
    place d'un `and`, `fold` et `mode` egaux acceptaient un point ecrit sous
    un autre `--repeats`, et ses tirages entraient dans la moyenne publiee.
    """
    from h4_unseen_conditions import checkpoint_is_reusable
    assert checkpoint_is_reusable(_prev(**{champ: valeur}), _Args()) is False, (
        f"un point de sauvegarde ecrit sous un autre {champ} a ete accepte : "
        "des tirages incomparables entreraient dans la meme moyenne")


def test_the_decision_is_a_conjunction_of_the_four_keys():
    """Le meme invariant vu par l'AST : quatre comparaisons, toutes en `and`.

    La parametrisation ci-dessus le couvre deja par appel ; ce test-ci
    attrape la variante ou un champ serait retire de la comparaison ET du
    jeu de donnees de test en meme temps.
    """
    import h4_unseen_conditions as t22

    src = open(t22.__file__, encoding="utf-8").read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef)
              and n.name == "checkpoint_is_reusable")
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    assert isinstance(ret.value, ast.BoolOp) and isinstance(ret.value.op, ast.And), (
        "la decision de reprise n'est plus une conjonction — un `or` y "
        "accepterait un point de sauvegarde d'une autre configuration")
    assert len(ret.value.values) == len(t22._RESUME_KEYS), (
        f"{len(ret.value.values)} comparaisons pour "
        f"{len(t22._RESUME_KEYS)} cles de reprise")


# ── D-124 : les deux enregistrements sont ECRITS, pas seulement nommes ──

def test_the_resume_provenance_fields_are_actually_assigned():
    """Les deux noms doivent etre AFFECTES, pas cites dans un commentaire.

    C'est le faux vert mesure : supprimer les deux lignes `out[...] = ...`
    laissait `test_resume_is_recorded_never_silent` vert, parce que les deux
    noms survivaient dans le commentaire qui les explique trois lignes plus
    haut. Un artefact aurait alors tu que ses tirages venaient d'un autre
    processus — exactement l'invisibilite que ce test existe pour interdire.
    """
    import h4_unseen_conditions as t22

    src = open(t22.__file__, encoding="utf-8").read()
    affectes = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Assign):
            continue
        for cible in node.targets:
            if (isinstance(cible, ast.Subscript)
                    and isinstance(cible.value, ast.Name)
                    and cible.value.id == "out"
                    and isinstance(cible.slice, ast.Constant)):
                affectes.add(cible.slice.value)

    for champ in ("resumed_from_checkpoint", "n_runs_resumed"):
        assert champ in affectes, (
            f"`out[\"{champ}\"]` n'est plus affecte dans "
            "h4_unseen_conditions.py — le nom peut subsister dans un "
            "commentaire sans qu'aucun artefact ne le porte")


# ── D-125 : le filtre des artefacts partiels, exerce ─────────────────

def _write_artifact(d, fold, mode, status, n_runs):
    """La forme qu'ecrit `h4_unseen_conditions`, avec assez de tirages pour
    que le repli « underpowered » ne masque pas le filtre teste."""
    bras = {"canonical": {"phys_score": 1.0}, "unseen": {"phys_score": 0.8},
            "n_runs": n_runs}
    art = {"fold": fold, "arms": {"qhas": dict(bras), "classical": dict(bras)}}
    if status is not None:
        art["status"] = status
        art["partial_stage"] = "qhas/canonical"
    (d / f"t22_unseen_{mode}_{fold}.json").write_text(json.dumps(art))


def test_the_transfer_summary_rejects_a_partial_artifact(tmp_path):
    """Le jumeau de `test_a_partial_record_is_rejected_by_the_summary`.

    Celui-la couvrait `closed_loop_leak_free_summary.py` ; le meme filtre de
    `h4_transfer_summary.py` n'etait garde QUE par la chaine `== "partial"`.
    Mesure : filtre desactive dans ce seul fichier, les 38 tests des trois
    fichiers qui l'importent restaient verts.

    L'artefact porte ici `n_runs = 4` : avec moins de 2, le repli
    « underpowered » ecarterait l'enregistrement pour une autre raison et le
    test ne mesurerait pas ce qu'il annonce.
    """
    import h4_transfer_summary as t22c

    d = tmp_path / "res"
    d.mkdir()
    _write_artifact(d, "kh", "leak-free", "partial", n_runs=4)

    rec = t22c.load(str(d), "kh", mode="leak-free")
    assert rec is not None, "artefact introuvable : le test ne mesure rien"
    assert rec.get("partial") is True, (
        "h4_transfer_summary lit un artefact PARTIEL comme s'il etait "
        "complet : ses moyennes porteraient sur une execution interrompue")
    assert rec["underpowered"] is True


def test_a_complete_artifact_is_still_analysed(tmp_path):
    """Contrôle positif : le filtre ne doit pas tout ecarter."""
    import h4_transfer_summary as t22c

    d = tmp_path / "res"
    d.mkdir()
    _write_artifact(d, "kh", "leak-free", None, n_runs=4)

    rec = t22c.load(str(d), "kh", mode="leak-free")
    assert rec is not None and rec["underpowered"] is False
    assert not rec.get("partial")
