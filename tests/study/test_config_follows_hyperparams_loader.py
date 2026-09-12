"""D-195 (audit H1/H3/H4, 26 aout) : `study/pipeline/config.py` et
`src/hyperparams_loader.py` resolvaient deux chemins DIFFERENTS par
defaut. `src/pipeline.py` (le pipeline DEPLOYE) lit deja automatiquement
`results/hyperparams/best_hyperparams.json` via
`resolve_hyperparams_path()`. `config.py`, lui, ne le faisait QUE si
`QHAS_HYPERPARAMS_PATH` etait explicitement exporte -- sinon repli
silencieux sur `_REFERENCE_TRAINED`, une constante figee. Une campagne
pouvait donc tourner et se deployer (D-22) sans qu'aucun script de
`study/h1_solver`/`study/h3_representation` (mappeur v1, tout ce qui
passe par `trained_mapper_params()`) n'en voie jamais le resultat -- H1
et H3 auraient continue a evaluer une configuration figee.

`config.py` resout maintenant les valeurs `TRAINED_*` a l'IMPORT du
module : le seul moyen fiable de tester chaque branche isolement est un
sous-processus frais par cas (meme pattern que
`test_fig0_pareto_paths.py::test_importer_le_module_n_ecrit_rien`),
sinon le module deja importe dans le processus pytest partage son etat
entre tests.
"""
import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

_REQUIRED_QUANTUM = ("beta", "w_z_frac", "sigma", "beta_curl", "beta_xpoint",
                     "gamma_hydro", "gamma_mag", "kappa",
                     "relative_percentile", "threshold_amr")

_COMPLETE = {k: float(i + 1) for i, k in enumerate(_REQUIRED_QUANTUM)}


def _write_new_format(path, params):
    path.write_text(json.dumps({"default": {"quantum": {"params": params}}}),
                    encoding="utf-8")


def _run(env_extra):
    code = (
        "import sys, warnings\n"
        "sys.path.insert(0, %r)\n"
        "with warnings.catch_warnings(record=True) as w:\n"
        "    warnings.simplefilter('always')\n"
        "    import config\n"
        "    print('SIGMA=' + repr(config.TRAINED_SIGMA))\n"
        "    print('PATH=' + repr(config.CAMPAIGN_HYPERPARAMS_PATH))\n"
        "    print('WARNED=' + repr(len(w) > 0))\n"
        % os.path.join(_REPO_ROOT, "study", "pipeline")
    )
    env = dict(os.environ)
    env.pop("QHAS_HYPERPARAMS_PATH", None)
    env.update(env_extra)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
    d = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            d[k] = eval(v)  # noqa: S307 -- litteraux qu'on vient d'imprimer
    return d


def test_an_explicit_complete_campaign_file_is_adopted(tmp_path):
    """Le cas que D-22 doit produire une fois la campagne relancee et
    deployee : `config.py` doit voir EXACTEMENT ce que verrait
    `pipeline.py`."""
    p = tmp_path / "candidate.json"
    _write_new_format(p, _COMPLETE)
    d = _run({"QHAS_HYPERPARAMS_PATH": str(p)})
    assert d["SIGMA"] == _COMPLETE["sigma"]
    assert d["PATH"] == str(p)
    assert d["WARNED"] is False


def test_an_explicit_incomplete_campaign_file_falls_back_loudly(tmp_path):
    """Cas reel actuel : le fichier deploye n'a que 8 des 10 cles
    requises (il manque `sigma`, `relative_percentile` -- voir D-22).
    Melanger un jeu partiel avec `_REFERENCE_TRAINED` serait pire que ne
    rien charger : repli total, mais jamais silencieux."""
    partial = {k: v for k, v in _COMPLETE.items() if k != "sigma"}
    p = tmp_path / "candidate.json"
    _write_new_format(p, partial)
    d = _run({"QHAS_HYPERPARAMS_PATH": str(p)})
    assert d["SIGMA"] == 0.023          # _REFERENCE_TRAINED, pas le fichier
    assert d["WARNED"] is True


def test_a_nonexistent_explicit_path_raises_instead_of_going_silent(tmp_path):
    """Une surcharge EXPLICITE qui ne charge pas ne doit jamais degrader
    silencieusement vers la reference : c'est le signe que l'appelant
    s'attendait a un fichier precis, pas a "la meilleure config connue"."""
    code = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "import config\n"
        % os.path.join(_REPO_ROOT, "study", "pipeline")
    )
    env = dict(os.environ)
    env["QHAS_HYPERPARAMS_PATH"] = str(tmp_path / "does_not_exist.json")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode != 0
    assert "FileNotFoundError" in out.stderr or "Error" in out.stderr


def test_no_override_resolves_to_the_same_default_path_as_pipeline(tmp_path):
    """Sans `QHAS_HYPERPARAMS_PATH`, `config.py` doit calculer EXACTEMENT
    le meme chemin par defaut que `hyperparams_loader.resolve_hyperparams_
    path()` -- celui que `src/pipeline.py` lit deja. Comparaison
    structurelle (les deux chemins calcules DANS le meme sous-processus,
    donc sous le meme environnement), jamais de contenu : le vrai fichier
    deploye ne doit jamais etre ecrit ou lu en dependance par un test
    (voir D-22, presque-incident du 26 aout ou un test avait ecrase le
    vrai `results/hyperparams/best_hyperparams.json`)."""
    code = (
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "sys.path.insert(0, %r)\n"
        "import config\n"
        "from hyperparams_loader import resolve_hyperparams_path\n"
        "print('CANDIDATE=' + repr(config._candidate_path))\n"
        "print('EXPECTED=' + repr(resolve_hyperparams_path()))\n"
        % (os.path.join(_REPO_ROOT, "study", "pipeline"),
           os.path.join(_REPO_ROOT, "src"))
    )
    env = dict(os.environ)
    env.pop("QHAS_HYPERPARAMS_PATH", None)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
    d = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            d[k] = eval(v)  # noqa: S307 -- litteraux qu'on vient d'imprimer
    assert d["CANDIDATE"] == d["EXPECTED"], (
        "config.py ne resout plus le meme chemin par defaut que "
        "hyperparams_loader.resolve_hyperparams_path() -- H1/H3 "
        "resteraient aveugles a une campagne deployee sans variable "
        "d'environnement explicite")
