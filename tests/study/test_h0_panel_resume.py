"""Reprise du panel H0, et le defaut qui rendait `--no-exact` inutilisable.

D-3  `--no-exact` levait un KeyError APRES le calcul.
     Quand l'optimum n'est pas certifie, la branche de repli ecrivait
     `mask_match` — que personne ne lit — et omettait `n_diff_patch`, alors
     que la boucle d'enregistrement lit `exact_match` et `n_diff_patch`.
     L'erreur ne survenait donc qu'au moment de consigner le PREMIER
     instantane, c'est-a-dire apres des heures de calcul a 32 qubits. La
     campagne H0b tournait dans cet etat.

D-4  Aucun point de reprise.
     Une campagne complete dure des heures ; le processus peut mourir. Sans
     reprise, tout est a refaire. Le piege associe est plus subtil que
     l'absence de reprise : reprendre un point de reprise produit sous
     D'AUTRES reglages melangerait deux campagnes dans un artefact
     parfaitement bien forme. La signature couvre donc tout ce qui change
     les nombres, et une reprise incompatible est refusee.
"""

import importlib.util
import json
import os
import sys

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
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_PANEL = os.path.join(_REPO_ROOT, "study", "h0_selection",
                      "h0_optimiser_equivalence.py")


@pytest.fixture(scope="module")
def panel():
    spec = importlib.util.spec_from_file_location("h0panel", _PANEL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Args:
    pass


def _args(**kw):
    a = _Args()
    base = dict(N=96, dim=3, re=[400], scenario=["orszag_tang"],
                with_psi=True, fixed_curl=False, zero_psi=False,
                no_exact=False, backend="state_vector", scale_kopt=True,
                mapper="v2", qaoa_reps=[1, 3, 6], n_snaps=6, sweeps=500,
                restarts=5, shots=4096, k_opt=60, no_qaoa=False, seed=0,
                no_resume=False)
    base.update(kw)
    for k, v in base.items():
        setattr(a, k, v)
    return a


# ── D-3 : les cles du repli doivent etre celles que l'on lit ──────────

def test_the_uncertified_fallback_uses_the_same_keys(panel):
    """Sans optimum certifie, les trois cles doivent exister quand meme.

    On compare l'ensemble des cles produites par `decision_agreement` a
    celles que la branche de repli ecrit : toute divergence rejaillit en
    KeyError apres le calcul.
    """
    import numpy as np

    spins = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int8)
    produced = set(panel.decision_agreement(spins, spins, 2))
    assert {"agree_spin", "exact_match", "n_diff_patch"} <= produced

    src = open(_PANEL, encoding="utf-8").read()
    i = src.index('r.update(dict(agree_spin=float("nan")')
    fallback = src[i:i + 300]
    for key in ("exact_match", "n_diff_patch"):
        assert key in fallback, (
            f"la branche non certifiee n'ecrit pas {key} ; "
            "--no-exact levera un KeyError apres le calcul")


def test_the_dead_key_is_gone(panel):
    """`mask_match` etait ecrit et jamais lu : le repli mentait sur son nom."""
    src = open(_PANEL, encoding="utf-8").read()
    assert 'mask_match=float("nan")' not in src


# ── D-4 : la signature de reprise ────────────────────────────────────

def test_resume_round_trips(panel, tmp_path):
    a = _args()
    ckpt = str(tmp_path / "c.jsonl")
    panel._append_checkpoint(ckpt, a, "orszag_tang", 400, 3,
                             [{"solver": "sa", "E": 1.0}], True)
    panel._append_checkpoint(ckpt, a, "orszag_tang", 400, 7,
                             [{"solver": "sa", "E": 2.0}], False)
    recs, diags, done = panel._load_checkpoint(ckpt, a)
    assert len(recs) == 2
    assert diags == [True, False]
    assert done == {("orszag_tang", 400, 3), ("orszag_tang", 400, 7)}


@pytest.mark.parametrize("changed", [
    {"seed": 7}, {"dim": 4}, {"N": 128}, {"with_psi": False},
    {"fixed_curl": True}, {"zero_psi": True}, {"qaoa_reps": [1]},
    {"scale_kopt": False}, {"mapper": "v1"}, {"shots": 1024},
    {"backend": "matrix_product_state"}, {"no_exact": True},
])
def test_a_different_setting_refuses_the_checkpoint(panel, tmp_path, changed):
    """Chaque reglage qui change les nombres doit invalider la reprise.

    Sans cela, l'artefact final serait un panachage de deux campagnes,
    impossible a distinguer d'une campagne coherente.
    """
    ckpt = str(tmp_path / "c.jsonl")
    panel._append_checkpoint(ckpt, _args(), "orszag_tang", 400, 3,
                             [{"solver": "sa", "E": 1.0}], True)
    with pytest.raises(SystemExit, match="reglages differents"):
        panel._load_checkpoint(ckpt, _args(**changed))


def test_the_scenario_is_not_part_of_the_signature(panel, tmp_path):
    """Le scenario est deja dans le NOM du point de reprise.

    L'inclure dans la signature rendrait toute reprise impossible des qu'on
    relance scenario par scenario, ce qui est justement le cas d'usage.
    """
    ckpt = str(tmp_path / "c.jsonl")
    panel._append_checkpoint(ckpt, _args(), "orszag_tang", 400, 3,
                             [{"solver": "sa", "E": 1.0}], True)
    recs, _d, _done = panel._load_checkpoint(ckpt, _args(scenario=["mhd_rotor"]))
    assert len(recs) == 1


def test_no_resume_ignores_the_checkpoint(panel, tmp_path):
    ckpt = str(tmp_path / "c.jsonl")
    panel._append_checkpoint(ckpt, _args(), "orszag_tang", 400, 3,
                             [{"solver": "sa", "E": 1.0}], True)
    recs, diags, done = panel._load_checkpoint(ckpt, _args(no_resume=True))
    assert (recs, diags, done) == ([], [], set())


def test_a_truncated_last_line_is_tolerated(panel, tmp_path):
    """Une mort brutale peut couper la derniere ligne en deux.

    Elle doit etre ignoree — l'instantane sera recalcule — et surtout ne pas
    faire echouer toute la reprise.
    """
    ckpt = str(tmp_path / "c.jsonl")
    panel._append_checkpoint(ckpt, _args(), "orszag_tang", 400, 3,
                             [{"solver": "sa", "E": 1.0}], True)
    with open(ckpt, "a", encoding="utf-8") as fh:
        fh.write('{"scenario": "orszag_ta')
    recs, _d, done = panel._load_checkpoint(ckpt, _args())
    assert len(recs) == 1 and done == {("orszag_tang", 400, 3)}


def test_a_missing_checkpoint_is_not_an_error(panel, tmp_path):
    recs, diags, done = panel._load_checkpoint(
        str(tmp_path / "absent.jsonl"), _args())
    assert (recs, diags, done) == ([], [], set())


def test_the_checkpoint_name_follows_the_artefact_name(panel):
    """Les deux derivent de la meme fonction, donc ne peuvent pas diverger."""
    a = _args()
    art = os.path.basename(panel._output_path(a))
    ck = os.path.basename(panel._checkpoint_path(a))
    assert art.endswith(".npz") and ck.endswith(".jsonl")
    assert art[:-len(".npz")] == ck[:-len(".jsonl")]


def test_each_line_is_flushed_to_disk(panel):
    """Sans fsync, une mort brutale perd des instantanes deja annonces.

    Ce test lit le SOURCE. Il est garde tel quel — il n'est pas faux — mais
    il ne mesure PAS la durabilite : D-127 l'a montre par mutation dans les
    deux sens. Le garde comportemental est
    `test_the_line_is_on_disk_before_fsync_returns` ci-dessous.
    """
    src = open(_PANEL, encoding="utf-8").read()
    i = src.index("def _append_checkpoint")
    body = src[i:i + 900]
    assert "fh.flush()" in body and "os.fsync(fh.fileno())" in body


def test_the_line_is_on_disk_before_fsync_returns(panel, tmp_path, monkeypatch):
    """D-127 : la durabilite mesuree, pas lue dans le texte du source.

    Le test ci-dessus cherche `fh.flush()` et `os.fsync(fh.fileno())` dans
    les 900 premiers caracteres de `_append_checkpoint`. Mesure par mutation,
    les deux sens :

    * **A'** — `flush`/`fsync` deplaces sous `if os.environ.get(...)` faux,
      les deux chaines INTACTES dans la fenetre : la durabilite a disparu et
      le fichier reste **21 passed**. Faux vert.
    * **B** — reecriture EQUIVALENTE `_fd = fh.fileno()` puis `os.fsync(_fd)`,
      comportement bit a bit identique : le test passe **ROUGE**. Faux rouge
      sur un changement voulu — 5e cas de cette forme dans ce depot.

    Ce que promet `_append_checkpoint` (son propre docstring) : « une mort
    brutale ne doit pas tronquer une ligne deja annoncee comme ecrite ».
    Donc, AU MOMENT ou `fsync` est appele, la ligne doit deja etre lisible
    par un descripteur independant — c'est `flush()` qui le garantit — et
    `fsync` doit etre appele sur le descripteur de CE fichier.

    L'entree qui SEPARE : lire le fichier depuis un second descripteur
    pendant l'appel a `fsync`. Retirer `flush()` laisse la ligne dans le
    tampon Python et le second lecteur voit un fichier vide ; retirer
    `fsync` fait que le tampon d'essai n'est jamais rempli.
    """
    ckpt = str(tmp_path / "c.jsonl")
    seen = {}

    real_fsync = os.fsync

    def spy(fd):
        #  Ce qu'un lecteur independant voit AU MOMENT du fsync.
        with open(ckpt, encoding="utf-8") as other:
            seen.setdefault("bytes", other.read())
        #  Releve l'inode ICI : le descripteur est ferme a la sortie du `with`.
        seen.setdefault("inodes", []).append(os.fstat(fd).st_ino)
        return real_fsync(fd)

    monkeypatch.setattr(panel.os, "fsync", spy)
    panel._append_checkpoint(ckpt, _args(), "orszag_tang", 400, 3,
                             [{"solver": "sa", "E": 1.0}], True)

    assert seen.get("inodes"), (
        "os.fsync n'a pas ete appele : une mort brutale peut perdre une "
        "ligne deja annoncee comme ecrite")
    #  Le descripteur doit etre celui du point de sauvegarde, pas un autre.
    assert seen["inodes"][0] == os.stat(ckpt).st_ino, (
        "fsync porte sur un autre fichier que le point de sauvegarde")
    #  flush() avant fsync() : sinon la ligne est encore dans le tampon.
    assert seen["bytes"], (
        "au moment du fsync le fichier est encore vide : `flush()` ne "
        "precede pas `fsync()`, donc rien n'est rendu durable")
    assert json.loads(seen["bytes"])["snap"] == 3
