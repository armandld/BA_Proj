"""D-102 — la conclusion de fig15 citait le sigma d'un AUTRE module.

`fig15_decision_flip_analysis.py` construit son `HamiltMapper` via
`_hamilt_mapper_kwargs` (`fig_utils.py`), dont le repli pour `sigma` vaut
0,05 (`TRAINED_PARAMS.get('sigma', 0.05)`) : `'sigma'` n'est echantillonne
par aucune entree de `results/hyperparams/best_hyperparams.json` (D-22), le
repli s'applique donc inconditionnellement dans ce fichier.

Le bloc "CONCLUSION" (imprime quand `flip_rate < 0.05 et mean_ratio < 0.5`)
citait pourtant `sigma=0.023` en dur — la valeur de `TRAINED_SIGMA` dans
`study/pipeline/config.py`, un module que ce fichier n'importe pas et dont
le pipeline (ferme) est distinct de celui de `figures/v1_legacy/`. Question
4 de `VIGIL.md` : deux chemins censes decrire le meme "sigma trained" ne
coincidaient pas.

Ce test n'importe pas `fig15_decision_flip_analysis.py` lui-meme : le
fichier execute sa campagne complete (VQA sur 4 scenarios) a l'IMPORT, sans
garde `if __name__ == "__main__"` (meme contrainte que
`test_v1_legacy_instrumented_bfs_score_grid.py` pour D-96). Il verifie donc
la grandeur reellement en jeu (le repli de TRAINED_PARAMS, importable seul
et sans simulation) et la STRUCTURE du fichier, par son AST.

D-150 — pourquoi ce fichier ne cherche plus de chaine dans le source
-------------------------------------------------------------------
Les deux gardes de structure faisaient `assert "σ=0.023" not in src` et
`assert "sigma_trained = TRAINED_PARAMS.get('sigma', 0.05)" in src`. Mesure
dans les deux sens, `1e2bc63` :

  - **A'** — le bloc CONCLUSION remis a citer `σ = 0.023` (avec des espaces,
    donc la chaine interdite est absente), la ligne d'affectation laissee en
    place et `{sigma_trained:` survivant dans une ligne de debogage : les
    quatre tests **passaient**. D-102 etait retabli, garde vert.
  - **B** — l'affectation reecrite avec des guillemets doubles, valeur
    identique au bit pres : **1 failed**. Faux rouge sur un changement voulu,
    le 5e de cette forme dans ce depot.

Ce qui est garde ici est un COMPORTEMENT — ce que le bloc CONCLUSION
imprime — et il ne se lit pas dans la mise en forme du source. Le detecteur
travaille donc sur les **litteraux de chaine de l'AST**, f-strings comprises,
en remplacant chaque champ interpole par `{expression}` : la citation en dur
se voit dans le TEXTE IMPRIME, la reecriture de guillemets n'y paraît pas.

Les commentaires sont volontairement hors du detecteur : ils ne sont pas
imprimes. C'est aussi la lecon de D-144 (des jetons survivant dans un
commentaire suffisaient a un garde).
"""
import ast
import re
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_V1_LEGACY = os.path.join(_REPO_ROOT, "figures", "v1_legacy")
if _V1_LEGACY not in sys.path:
    sys.path.insert(0, _V1_LEGACY)

from fig_utils import TRAINED_PARAMS  # noqa: E402

_FIG15 = os.path.join(_V1_LEGACY, "fig15_decision_flip_analysis.py")


def test_sigma_key_is_still_absent_from_deployed_hyperparams():
    """Precondition du defaut (D-22) : si elle cesse d'etre vraie, le repli
    0,05 ci-dessous cesse d'etre le nombre reellement utilise et ce test
    doit etre remesure, pas simplement mis a jour."""
    assert "sigma" not in TRAINED_PARAMS


def test_the_actual_fallback_this_file_uses_is_0_05_not_0_023():
    """C'est cette valeur, et non 0,023, que `_hamilt_mapper_kwargs`
    applique reellement dans fig15 (et fig11/fig16/fig17, meme helper)."""
    assert TRAINED_PARAMS.get("sigma", 0.05) == 0.05
    assert TRAINED_PARAMS.get("sigma", 0.05) != 0.023


#: Un marqueur de sigma suivi d'un NOMBRE : c'est la citation en dur.
#: `{2 * sigma_trained}` ne repond pas (ce qui suit n'est pas un nombre),
#: `'sigma' absent from` non plus (aucun nombre).
_SIGMA_EN_DUR = re.compile(r"(?:σ|sigma)\s*[=:]?\s*[0-9]+\.?[0-9]*", re.I)
_SIGMA_MARQUEUR = re.compile(r"σ|sigma", re.I)


def _texte_imprime(noeud):
    """Le texte qu'un litteral produirait a l'execution, ou `None`.

    Une f-string rend `{expression}` a la place du champ interpole, sans son
    format-spec : `f"σ={sigma_trained:.3f}"` devient `σ={sigma_trained}`. La
    valeur en dur s'y voit, la mise en forme du source non.
    """
    if isinstance(noeud, ast.Constant) and isinstance(noeud.value, str):
        return noeud.value
    if isinstance(noeud, ast.JoinedStr):
        morceaux = []
        for p in noeud.values:
            if isinstance(p, ast.Constant):
                morceaux.append(str(p.value))
            elif isinstance(p, ast.FormattedValue):
                morceaux.append("{" + ast.unparse(p.value) + "}")
        return "".join(morceaux)
    return None


def _litteraux_sigma(arbre):
    """(en dur, dynamiques) — les litteraux qui presentent une valeur de σ."""
    en_dur, dynamiques = [], []
    for n in ast.walk(arbre):
        texte = _texte_imprime(n)
        if texte is None:
            continue
        if _SIGMA_EN_DUR.search(texte):
            en_dur.append((getattr(n, "lineno", -1), texte.strip()[:90]))
        elif _SIGMA_MARQUEUR.search(texte) and "sigma_trained" in texte:
            dynamiques.append((getattr(n, "lineno", -1), texte.strip()[:90]))
    return en_dur, dynamiques


def _arbre_fig15():
    return ast.parse(open(_FIG15, encoding="utf-8").read())


def test_no_printed_string_hardcodes_a_sigma_value():
    """D-102, garde par ce qui est IMPRIME et non par une chaine du source.

    Sur quelle entree ce test echoue : sur toute citation en dur d'un sigma
    dans un litteral imprime — `σ=0.023`, `σ = 0.023`, `sigma: 0.023` —,
    quelle que soit sa mise en forme. Mesure A' de D-150 : l'ancien garde
    rendait 4 passed sur `σ = 0.023`, celui-ci rougit.
    """
    en_dur, _ = _litteraux_sigma(_arbre_fig15())
    assert not en_dur, (
        f"fig15 imprime une valeur de sigma en dur : {en_dur}. Le seul sigma "
        "que ce fichier applique est le repli 0,05 de TRAINED_PARAMS (D-102) "
        "— il doit etre interpole, pas ecrit")


def test_the_conclusion_interpolates_the_sigma_it_actually_uses():
    """Le pendant positif : la valeur dynamique doit encore etre imprimee.

    Sans lui, supprimer purement et simplement la ligne « ROOT CAUSE » ferait
    passer le test ci-dessus — un balayage vide.

    Valeur mesuree a `1e2bc63` : **2** litteraux (les deux `print` du bloc
    CONCLUSION, `σ={sigma_trained}` et `~{2 * sigma_trained}`).
    """
    _, dynamiques = _litteraux_sigma(_arbre_fig15())
    assert len(dynamiques) >= 2, (
        f"seuls {len(dynamiques)} litteraux imprimant sigma interpolent "
        f"`sigma_trained` (2 mesures a `1e2bc63`) : {dynamiques}")


def test_sigma_trained_is_still_the_trained_params_fallback():
    """L'affectation, par l'AST : `TRAINED_PARAMS.get('sigma', 0.05)`.

    Structurel, donc insensible au style de guillemets — mesure B de D-150 :
    l'ancien garde rougissait sur `TRAINED_PARAMS.get("sigma", 0.05)`, une
    reecriture identique au bit pres.
    """
    valeurs = []
    for n in ast.walk(_arbre_fig15()):
        if not (isinstance(n, ast.Assign) and len(n.targets) == 1):
            continue
        cible = n.targets[0]
        if not (isinstance(cible, ast.Name) and cible.id == "sigma_trained"):
            continue
        appel = n.value
        assert isinstance(appel, ast.Call), ast.unparse(n)
        assert getattr(appel.func, "attr", None) == "get", ast.unparse(n)
        assert getattr(appel.func.value, "id", None) == "TRAINED_PARAMS", (
            ast.unparse(n))
        valeurs.append(tuple(ast.literal_eval(a) for a in appel.args))
    assert valeurs == [("sigma", 0.05)], (
        f"`sigma_trained` n'est plus le repli de TRAINED_PARAMS : {valeurs}")


def test_the_detector_itself_can_fail():
    """Le detecteur doit trouver ce qu'il annonce, sinon les tests ci-dessus
    passent sans rien verifier. Les cinq cas qui l'ont ecrit."""
    def _sur(code):
        return _litteraux_sigma(ast.parse(code))

    en_dur, _ = _sur('print("σ=0.023")')
    assert len(en_dur) == 1, en_dur
    en_dur, _ = _sur('print("σ = 0.023")')          # l'espacement de A'
    assert len(en_dur) == 1, en_dur
    en_dur, _ = _sur('print(f"sigma: {x:.2f} et 0.023")')
    assert not en_dur, en_dur                        # nombre non colle au marqueur
    en_dur, dyn = _sur('print(f"σ={sigma_trained:.3f}")')
    assert not en_dur and len(dyn) == 1, (en_dur, dyn)
    en_dur, dyn = _sur('print(f"~{2 * sigma_trained:.2f} du seuil (sigma)")')
    assert not en_dur and len(dyn) == 1, (en_dur, dyn)
    # un commentaire n'est pas imprime : il ne doit rien declencher
    en_dur, _ = _sur('# σ=0.023\nx = 1\n')
    assert not en_dur, en_dur
