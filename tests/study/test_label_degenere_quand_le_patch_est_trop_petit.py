"""Le label garde-t-il un sens quand `dim` monte a N fixe ?

Le label de la phase 2 est l'ecart-type INTRA-patch : `patch_l2_errors`
remplace chaque patch par sa moyenne, puis mesure l'ecart du champ fin a
cette moyenne. Un patch de p x p cellules estime donc cet ecart-type sur
p^2 echantillons — et **a p = 1 il vaut identiquement zero**, puisqu'une
cellule ne devie pas de sa propre moyenne.

Le seuil etant un percentile de valeurs toutes nulles, il vaut 0, et
`is_hard = (l2 >= 0)` marque **100 % des patches comme durs**. Un tel
artefact a la bonne forme, des valeurs finies dans le bon intervalle, et ne
veut rien dire : c'est la classe de defaut que `CODE_REVIEW.md` designe
comme la seule qui compte.

Quatre artefacts du depot sont dans ce cas (`*_N64_dim64`). Ils ne sont
consommes par aucun script et la table maitresse ne cite que dim 2, 4 et 8 :
par la regle d'arret de `DEFAUTS.md`, ils ne bloquent rien. Ce fichier est
le garde qui empeche le corpus d'en gagner un cinquieme sans que personne
ne crie.

`tests/study/test_t28_t29_labels_and_ci.py` garde deja le CONSOMMATEUR : le
relabelliseur leve `SystemExit` sur un seuil degenere. Rien ne gardait le
PRODUCTEUR, qui ecrit l'artefact en silence — et le message de ce test le
disait deja : « seuil 0.000000, 100 % durs, et rien ne crie ».
"""
import glob
import os
import sys

import numpy as np
import pytest

_RACINE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_RACINE, "src"), os.path.join(_RACINE, "study", "pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hard_patch_labels import patch_l2_errors      # noqa: E402

_RESULTS = os.path.join(_RACINE, "results")

#: Les artefacts dont le label est identiquement nul, MESURES le 21 aout 2026.
#: Ils ne sont lus par aucun script de `study/` ni cite par la table
#: maitresse. La liste est fermee : un cinquieme fait echouer le test.
_DEGENERES_CONNUS = {
    "patches_harris_tearing_Re400_N64_dim64.npz",
    "patches_kelvin_helmholtz_Re400_N64_dim64.npz",
    "patches_mhd_rotor_Re400_N64_dim64.npz",
    "patches_orszag_tang_Re400_N64_dim64.npz",
}


def _champ(N, graine=0):
    rng = np.random.default_rng(graine)
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return (np.sin(Y) + 0.2 * rng.standard_normal((N, N)),
            np.sin(X), np.cos(Y), np.sin(2 * X))


# ------------------------------------------------------------------
#  1. le mecanisme, calcule et non suppose
# ------------------------------------------------------------------
def test_un_patch_dune_cellule_rend_un_label_identiquement_nul():
    """p = 1 : une cellule ne devie pas de sa propre moyenne."""
    N = 16
    l2 = patch_l2_errors(*_champ(N), N)          # dim = N  ->  p = 1
    assert np.allclose(l2, 0.0), (
        f"attendu un label nul a p=1, max={np.max(l2):.3e} — la definition "
        "du label a change, relire ce fichier")


def test_le_label_reprend_du_sens_des_que_le_patch_grandit():
    """Le champ qui SEPARE : sans lui, le test ci-dessus pourrait passer
    sur un label nul PARTOUT, ce qui ne prouverait rien."""
    N = 16
    l2 = patch_l2_errors(*_champ(N), N // 2)     # p = 2
    assert np.max(l2) > 1e-6, (
        "le label est nul aussi a p=2 : le champ d'essai est uniforme, ce "
        "test ne separe rien")


@pytest.mark.parametrize("p", [1, 2, 4, 8])
def test_le_nombre_dechantillons_par_patch_vaut_p_carre(p):
    """La regle de dimensionnement, epinglee : estimer un ecart-type sur
    p^2 points. En dessous de 16 points l'estimation est du bruit ; la
    contrainte confortable est `dim <= N/8`, soit p >= 8."""
    N = 32
    dim = N // p
    l2 = patch_l2_errors(*_champ(N), dim)
    assert l2.shape == (dim, dim)
    if p == 1:
        assert np.allclose(l2, 0.0)
    else:
        assert np.max(l2) > 0.0


# ------------------------------------------------------------------
#  2. le corpus
# ------------------------------------------------------------------
def _artefacts_degeneres(repertoire=None):
    """`repertoire` est parametrable pour que le PLANCHER soit testable.

    Un plancher qu'on ne peut pas faire tomber n'est pas un garde : le
    baisser alors qu'il est deja satisfait ne fait rien echouer. On lui
    donne donc une entree ou le balayage est reellement vide.
    """
    morts = set()
    racine = _RESULTS if repertoire is None else repertoire
    fichiers = sorted(glob.glob(os.path.join(racine, "patches_*.npz")))
    for chemin in fichiers:
        try:
            d = np.load(chemin)
            if np.allclose(d["l2_errors"], 0.0):
                morts.add(os.path.basename(chemin))
        except Exception:
            continue
    return morts, len(fichiers)


def _verifie_plancher(n_fichiers, plancher=150):
    assert n_fichiers >= plancher, (
        f"{n_fichiers} artefacts balayes ; 172 mesures le 21 aout 2026 — "
        "le balayage a retreci, il ne prouve plus ce qu'il prouvait")


def test_le_plancher_de_balayage_tombe_sur_un_repertoire_vide(tmp_path):
    """Le garde du garde : sans lui, le plancher ne se teste jamais."""
    _, n = _artefacts_degeneres(str(tmp_path))
    assert n == 0
    with pytest.raises(AssertionError, match="le balayage a retreci"):
        _verifie_plancher(n)


def test_le_corpus_ne_gagne_pas_un_artefact_degenere_de_plus():
    """Liste FERMEE : le jour ou une campagne en ecrit un cinquieme, ce
    test rougit au lieu de le laisser entrer en silence."""
    morts, n_fichiers = _artefacts_degeneres()
    _verifie_plancher(n_fichiers)
    nouveaux = morts - _DEGENERES_CONNUS
    assert not nouveaux, (
        f"artefact(s) au label identiquement nul, non connus : "
        f"{sorted(nouveaux)}. `is_hard` y marque 100 % de patches durs et "
        "tout F1 mesure dessus vaut celui du predicteur constant.")


def test_les_degeneres_connus_le_sont_toujours():
    """Une liste d'exemptions perimee doit crier, pas se faire oublier."""
    morts, _ = _artefacts_degeneres()
    presents = {n for n in _DEGENERES_CONNUS
                if os.path.exists(os.path.join(_RESULTS, n))}
    gueris = presents - morts
    assert not gueris, (
        f"{sorted(gueris)} n'est plus degenere : la phase 2 a change, "
        "retirer l'entree de la liste")


def test_aucun_script_de_study_ne_lit_un_artefact_degenere():
    """Ce qui rend les quatre tolerables : personne ne les consomme.

    Sur quelle entree ce test echoue : le jour ou un script nomme `dim 64`
    ou `dim=64`, la tolerance ci-dessus cesse d'etre justifiee.
    """
    motifs = ("dim64", "dim=64", "dim 64")
    coupables = []
    for base in ("study", "figures", "scripts"):
        rep = os.path.join(_RACINE, base)
        for racine, _, fichiers in os.walk(rep):
            if "__pycache__" in racine:
                continue
            for f in fichiers:
                if not f.endswith((".py", ".sh")):
                    continue
                chemin = os.path.join(racine, f)
                texte = open(chemin, encoding="utf-8", errors="replace").read()
                if any(m in texte for m in motifs):
                    coupables.append(os.path.relpath(chemin, _RACINE))
    assert not coupables, (
        f"{coupables} nomme(nt) dim=64, ou le label est identiquement nul : "
        "la tolerance accordee aux quatre artefacts degeneres ne tient plus")
