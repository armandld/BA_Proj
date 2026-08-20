"""`study/pipeline/sanity_check.py` recopie les 9 hyperparametres entraines
au lieu de les importer de `config.py` — mesure du 20 aout.

`V1_PARAMS` / `V1_THRESHOLD` / `V2_THRESHOLD` (lignes 56-61) sont des
litteraux. Le module n'importe pas `config`. Balayage de `study/`, `src/` et
`figures/` (hors `v1_legacy/`) : c'est la **seule** duplication EXECUTABLE
du jeu entraine hors de `config.py` — les autres occurrences de `0.1496` ou
`9.94` sont de la prose ou des commentaires, et `h2b_psi_feature_loso.py`
montre la convention du depot (`from config import TRAINED_BETA as ...`).

Les 9 valeurs **coincident** aujourd'hui : 0 desaccord. Rien n'est corrige,
et ce test ne tranche rien — VIGIL.md, defaut CONTRE choix de conception :
mesurer, documenter, ne pas corriger, demander.

Ce qu'il refuse, c'est le SILENCE. D-22 (reoptimisation, ~206 h CPU) va
reecrire ces 9 valeurs dans `config.py`. Ce jour-la, sans ce garde,
`sanity_check.py` continuerait de comparer le v2 courant a un v1 perime en
se presentant comme le controle de sante du hamiltonien deploye, sans qu'un
mot de sa sortie ne bouge.

Le jour ou il rougit, la question a trancher est ecrite dans son message :
`sanity_check` doit-il SUIVRE `config`, ou etre GELE et documente comme tel
(comme `dns_validation.py` l'est) ? Les deux sont defendables ; aucun n'est
choisi ici.
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
import sanity_check

# (cle de V1_PARAMS, nom dans config.py) — les 7 du mapper v1
_MAPPER = [
    ("sigma", "TRAINED_SIGMA"),
    ("beta_curl", "TRAINED_BETA_CURL"),
    ("beta_xpoint", "TRAINED_BETA_XPOINT"),
    ("w_z_frac", "TRAINED_W_Z_FRAC"),
    ("gamma_hydro", "TRAINED_GAMMA_HYDRO"),
    ("gamma_mag", "TRAINED_GAMMA_MAG"),
    ("kappa", "TRAINED_KAPPA"),
]

_DECIDE = (
    "\n\n  À TRANCHER (non decide le 20 aout, mesure : 0 desaccord) : "
    "`sanity_check.py` recopie les hyperparametres entraines au lieu de les "
    "importer. Ils viennent de diverger. Soit il SUIT `config.py` — remplacer "
    "les litteraux par des imports — soit il est GELE volontairement, et il "
    "faut l'ecrire dans le fichier avec sa raison, comme `dns_validation.py` "
    "le fait. Ne pas se contenter de recopier la nouvelle valeur : c'est ce "
    "qui a rendu la divergence invisible la premiere fois."
)


def test_the_seven_mapper_parameters_still_match_config():
    diverged = [(k, sanity_check.V1_PARAMS[k], getattr(config, cn))
                for k, cn in _MAPPER
                if sanity_check.V1_PARAMS[k] != getattr(config, cn)]
    assert not diverged, (
        f"{len(diverged)}/7 hyperparametres divergent — "
        f"(parametre, sanity_check, config) : {diverged}" + _DECIDE)


def test_both_thresholds_still_match_config():
    diverged = [(n, a, b) for n, a, b in (
        ("threshold v1", sanity_check.V1_THRESHOLD, config.TRAINED_THRESHOLD),
        ("threshold v2", sanity_check.V2_THRESHOLD, config.V2_THRESHOLD),
    ) if a != b]
    assert not diverged, (
        f"seuil(s) divergent(s) — (nom, sanity_check, config) : "
        f"{diverged}" + _DECIDE)


def test_the_guard_covers_every_parameter_sanity_check_copies():
    """Un garde qui couvre 6 cles sur 7 laisse passer la septieme.

    L'assertion porte sur la COUVERTURE du garde lui-meme : si quelqu'un
    ajoute un hyperparametre a `V1_PARAMS` sans l'ajouter a `_MAPPER`, les
    deux tests ci-dessus resteraient verts sur un parametre non surveille.
    """
    assert set(sanity_check.V1_PARAMS) == {k for k, _ in _MAPPER}, (
        f"V1_PARAMS = {sorted(sanity_check.V1_PARAMS)}, garde = "
        f"{sorted(k for k, _ in _MAPPER)} — le garde ne couvre plus tout")
