"""Aucun identifiant de base ne doit revenir dans le depot.

`import_Neon_data_to_local.py` portait l'URL Neon complete — mot de passe
compris — en valeur par defaut. Le depot est **public**. Le retrait de cette
valeur ne desamorce pas ce qui est deja dans l'historique git : seule une
rotation du mot de passe le fait (`docs/DEFAUTS.md`). Ce test empeche le
suivant d'etre publie a son tour.

L'assertion porte sur la forme d'une URL portant un mot de passe, pas sur une
chaine precise : un identifiant different, mais code en dur, doit tomber lui
aussi.

D-152 — pourquoi le balayage ne s'arrete plus a `src/`
------------------------------------------------------
La premiere version ne parcourait que `src/*.py`. La fuite suivante n'avait
donc qu'a se poser ailleurs, et « ailleurs » est justement ou vit ce genre de
valeur : un lanceur `.sh` qui exporte `OPTUNA_STORAGE`, un module de
`study/`, un `.yaml` d'environnement. Mesure a `c7a1e9c` : le depot porte
**409 fichiers texte**, dont **384 hors `src/`** et **10 lanceurs `.sh`** —
aucun n'etait regarde.

Second trou, de la meme famille que D-151 : le balayage lisait **ligne par
ligne**, donc une URL coupee par une continuation `\\` en fin de ligne — la
forme normale d'un long `export` shell — n'etait vue par personne. Les
continuations sont desormais recollees avant lecture.

Ce qui est autorise, et pourquoi ce n'est pas un trou
-----------------------------------------------------
La documentation DOIT montrer la forme d'une URL de connexion. Les couples
`utilisateur:motdepasse` de remplacement sont donc listes nommement dans
`_MODELES`, avec leur raison — et un test exige que chacun soit encore
present quelque part, pour qu'une exemption perimee crie au lieu de dormir.
"""

import os
import re


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable depuis " + __file__)


_ROOT = _repo_root()
_SRC = os.path.join(_ROOT, "src")

#: `schema://utilisateur:motdepasse@hote` — la forme qui porte un secret.
_URL_WITH_PASSWORD = re.compile(
    r"[a-z][a-z0-9+.\-]*://[^\s/:@\"']+:[^\s/:@\"']+@[^\s\"']+")

#: Les couples utilisateur:motdepasse qui ne sont PAS des secrets — des
#: modeles que la documentation et ce test montrent volontairement. Chacun
#: doit rester trouvable dans le depot : une exemption perimee est un
#: mensonge qui dort (meme regle que `_EXEMPTIONS` de
#: `tests/test_launcher_paths_resolve.py`).
_MODELES = {
    "user:pass": "README.md — la forme d'une URL Optuna, sans valeur reelle",
    "user:pw": "docs/MODE_EMPLOI_CAMPAGNE.md — idem, forme abregee",
    "utilisateur:motdepasse": "ce fichier — le commentaire qui decrit le motif",
    "user:secret": "ce fichier — l'entree de `test_the_pattern_can_fire`",
}

#: Extensions balayees : tout ce qui peut porter une valeur executee ou
#: copiee-collee. Les binaires et les artefacts en sont exclus par extension,
#: pas par dossier — un `.py` de `study/` compte autant qu'un `.py` de `src/`.
_EXTENSIONS = (".py", ".sh", ".ipynb", ".yaml", ".yml", ".json", ".toml",
               ".cfg", ".ini", ".md", ".txt", ".env", ".tex")

#: Dossiers sans code vivant : historique de campagne et caches.
_HORS_BALAYAGE = ("__pycache__", ".git", "node_modules", "Train_results",
                  "Data_results")

#: Plancher mesure a `c7a1e9c`. Un balayage qui retrecit ne prouve plus rien.
_PLANCHER_FICHIERS = 200


def _fichiers_balayes():
    out = []
    for dirpath, dirs, names in os.walk(_ROOT):
        dirs[:] = [d for d in dirs if d not in _HORS_BALAYAGE]
        if any(x in dirpath for x in _HORS_BALAYAGE):
            continue
        for n in names:
            if n.endswith(_EXTENSIONS):
                out.append(os.path.join(dirpath, n))
    return sorted(out)


def _lignes_recollees(texte):
    """(numero de la PREMIERE ligne, ligne logique) — continuations `\\` incluses.

    D-151, dans ce fichier : une ligne de source n'est pas une unite de sens.
    Un `export URL="postgresql://…\\` + suite en ligne 2 portait un secret que
    la lecture ligne a ligne ne voyait pas.
    """
    out = []
    courant, depart = None, None
    for lineno, ligne in enumerate(texte.splitlines(), 1):
        nue = ligne.rstrip("\n")
        if courant is None:
            courant, depart = "", lineno
        if nue.endswith("\\"):
            courant += nue[:-1]
            continue
        courant += nue
        out.append((depart, courant))
        courant, depart = None, None
    if courant is not None:
        out.append((depart, courant))
    return out


def _fuites(chemin):
    """Les URL a mot de passe d'un fichier, hors modeles documentes."""
    try:
        texte = open(chemin, encoding="utf-8", errors="ignore").read()
    except OSError:                                       # pragma: no cover
        return []
    trouve = []
    for lineno, ligne in _lignes_recollees(texte):
        if ligne.lstrip().startswith("#"):
            continue          # un commentaire qui documente la fuite
        for m in _URL_WITH_PASSWORD.finditer(ligne):
            url = m.group(0)
            couple = url.split("://", 1)[1].split("@", 1)[0]
            if couple in _MODELES:
                continue
            trouve.append((lineno, url))
    return trouve


def test_no_hardcoded_credential_url_anywhere_in_the_repository():
    fichiers = _fichiers_balayes()
    assert len(fichiers) >= _PLANCHER_FICHIERS, (
        f"{len(fichiers)} fichiers balayes, {_PLANCHER_FICHIERS} exiges "
        "(mesure a `c7a1e9c`) : le balayage a retreci, il ne prouve plus rien")

    offenders = []
    for path in fichiers:
        for lineno, url in _fuites(path):
            offenders.append("%s:%d (%s)"
                             % (os.path.relpath(path, _ROOT), lineno,
                                url.split("@")[0] + "@…"))
    assert offenders == [], (
        "URL portant un mot de passe dans un depot public : %s" % offenders)


def test_src_is_still_inside_the_sweep():
    """Le perimetre d'origine ne doit pas disparaitre en s'elargissant."""
    balayes = set(_fichiers_balayes())
    dans_src = [p for p in balayes if p.startswith(_SRC + os.sep)]
    assert len(dans_src) >= 10, (
        f"{len(dans_src)} fichiers de src/ balayes : le perimetre d'origine "
        "de ce test a ete perdu en l'elargissant")


def test_the_sweep_reaches_the_places_a_connection_string_actually_lives():
    """Les trois familles que D-152 a ajoutees, nommement.

    Sur quelle entree ce test echoue : sur un balayage qui redeviendrait
    `src/`-seulement, ou qui perdrait les lanceurs.
    """
    balayes = {os.path.relpath(p, _ROOT) for p in _fichiers_balayes()}
    lanceurs = [p for p in balayes if p.endswith(".sh")]
    assert len(lanceurs) >= 8, f"{len(lanceurs)} lanceurs balayes, 10 mesures"
    assert any(p.startswith("study" + os.sep) for p in balayes)
    assert any(p.startswith("scripts" + os.sep) for p in balayes)


def test_the_pattern_can_fire(tmp_path):
    """Un test qui ne peut pas echouer est un defaut : on le prouve ici."""
    assert _URL_WITH_PASSWORD.search(
        'url = "postgresql://' + 'user:secret' + '@host.example/db"')
    assert not _URL_WITH_PASSWORD.search('url = "sqlite:///local.db"')


def test_a_leak_outside_src_is_caught(tmp_path):
    """A' de D-152, epinglee : la fuite hors de `src/`.

    L'ancien balayage rendait `[]` sur ce fichier — il ne regardait que
    `src/*.py`.
    """
    faux = tmp_path / "lanceur_de_campagne.sh"
    faux.write_text('export OPTUNA_STORAGE="postgresql://neondb_owner:'
                    'npg_A1b2C3d4@ep-cool-name.eu-central-1.aws.neon.tech/db"\n')
    fuites = _fuites(str(faux))
    assert len(fuites) == 1 and fuites[0][0] == 1, fuites


def test_a_credential_split_by_a_line_continuation_is_caught(tmp_path):
    """Le second trou : la ligne logique, pas la ligne physique (D-151).

    Le meme secret, coupe par un `\\` en fin de ligne — la forme normale d'un
    long `export`. Lu ligne a ligne, aucun des deux morceaux ne repond au
    motif ; recolle, le secret est la.
    """
    faux = tmp_path / "coupe.sh"
    faux.write_text('export OPTUNA_STORAGE="postgresql://neondb_owner:npg_A1\\\n'
                    'b2C3d4@ep-cool-name.aws.neon.tech/db"\n')
    lignes = faux.read_text().splitlines()
    assert not any(_URL_WITH_PASSWORD.search(l) for l in lignes), (
        "le champ d'essai ne SEPARE pas : le motif repond deja ligne a ligne")
    fuites = _fuites(str(faux))
    assert len(fuites) == 1 and fuites[0][0] == 1, fuites


def test_every_documented_placeholder_still_exists(tmp_path):
    """Une exemption perimee doit crier.

    Chaque couple de `_MODELES` doit rester trouvable : le jour ou la
    documentation cesse de le montrer, l'exemption tombe avec elle — sinon
    elle couvrirait un vrai secret qui porterait le meme nom d'utilisateur.
    """
    textes = []
    for path in _fichiers_balayes():
        try:
            textes.append(open(path, encoding="utf-8", errors="ignore").read())
        except OSError:                                   # pragma: no cover
            continue
    tout = "\n".join(textes)
    for couple, raison in _MODELES.items():
        assert raison.strip(), f"{couple} exempte sans raison"
        assert couple in tout, (
            f"le modele {couple!r} n'est plus nulle part dans le depot : "
            "retirer son entree de `_MODELES`, sinon elle exempte un vrai "
            "identifiant qui porterait ce nom d'utilisateur")


def test_a_real_secret_wearing_a_placeholder_username_is_still_caught():
    """La limite de `_MODELES`, ecrite : l'exemption porte sur le COUPLE.

    `user:pass` est exempte ; `user:Vrai_M0t2Passe` ne l'est pas.
    """
    import tempfile
    #  L'URL est ASSEMBLEE, jamais ecrite d'un bloc : ce fichier est
    #  lui-meme dans le balayage, et un litteral complet ici ferait rougir
    #  le test principal. Ne pas « simplifier » cette concatenation.
    url = "postgresql://" + "user:Vrai_M0t2Passe" + "@ep-x.neon.tech/db"
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write('DB="%s"\n' % url)
        chemin = fh.name
    try:
        assert len(_fuites(chemin)) == 1
    finally:
        os.remove(chemin)
