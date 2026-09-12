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
de remplacement sont donc listes nommement dans `_MODELES`, avec leur
raison — et un test exige que chacun neutralise encore une URL reelle, pour
qu'une exemption perimee crie au lieu de dormir.

D-161 — pourquoi ce controle de peremption ne pouvait pas crier
---------------------------------------------------------------
Il faisait `couple in tout`, ou `tout` est le texte de tous les fichiers
balayes — **y compris ce fichier-ci**, celui qui DECLARE les couples. Les
quatre cles y figuraient litteralement : le controle trouvait toujours sa
propre declaration. Meme forme que D-159, et la quatrieme question de
`COUVERTURE.md` : *le balayage figure-t-il dans ce qu'il balaie ?*

Second ecart, celui qui l'a rendu inoffensif en apparence : il mesurait la
presence du couple avec `in` sur du texte brut, alors que l'exemption est
consommee sur une URL reconnue par `_URL_WITH_PASSWORD`. Une mention en
prose satisfaisait donc un controle dont la garantie porte sur des URL —
l'operateur assorti, sur une grandeur textuelle.

Consequence mesuree au 18 aout 2026, avant correction : **2 des 4
exemptions ne neutralisaient plus rien**. `utilisateur:motdepasse`
n'apparaissait qu'en prose, `user:secret` qu'en morceaux concatenes que le
motif ne reconnaît pas. Les retirer laisse **0 fuite** : elles n'exemptaient
rien, elles autorisaient d'avance tout secret portant ces couples. Le
controle ecrit pour empecher exactement cela les declarait saines.
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

#: No credential-bearing URL, including a placeholder, is needed by the
#: single-machine workflow. Keep the exemption set empty.
_MODELES = {}

#: Extensions balayees : tout ce qui peut porter une valeur executee ou
#: copiee-collee. Les binaires et les artefacts en sont exclus par extension,
#: pas par dossier — un `.py` de `study/` compte autant qu'un `.py` de `src/`.
_EXTENSIONS = (".py", ".sh", ".ipynb", ".yaml", ".yml", ".json", ".toml",
               ".cfg", ".ini", ".md", ".txt", ".env", ".tex")

#: Dossiers sans code vivant : historique de campagne et caches.
_HORS_BALAYAGE = ("__pycache__", ".git", ".venv", ".venv_vigil", "env",
                  "node_modules", "Train_results", "Data_results")

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


def _urls_a_mot_de_passe(chemin):
    """(numero de ligne, url, couple) — TOUTES les URL a mot de passe, sans
    filtrer les modeles.

    D-161 : `_fuites` et le controle de peremption des exemptions doivent
    lire le fichier avec LE MEME operateur. L'ancien controle mesurait la
    presence du couple avec `in` sur du texte brut, la ou l'exemption est
    consommee sur une URL reconnue par `_URL_WITH_PASSWORD` : deux
    operateurs pour une meme grandeur, et c'est l'ecart entre les deux qui
    a laisse vivre deux exemptions mortes.
    """
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
            trouve.append((lineno, url, url.split("://", 1)[1].split("@", 1)[0]))
    return trouve


def _fuites(chemin):
    """Les URL a mot de passe d'un fichier, hors modeles documentes."""
    return [(lineno, url) for lineno, url, couple in _urls_a_mot_de_passe(chemin)
            if couple not in _MODELES]


def _porteurs_reels(couple):
    """Fichiers ou `couple` apparaît VRAIMENT dans une URL a mot de passe.

    Deux differences avec l'ancien `couple in tout`, et chacune ferme la
    moitie de D-161 :

      - la mesure se fait avec `_URL_WITH_PASSWORD`, l'operateur qui
        consomme l'exemption. Une mention du couple en prose (« les couples
        `utilisateur:motdepasse` sont listes nommement ») n'est pas une URL
        et ne justifie donc aucune exemption ;
      - **ce fichier sort du corpus**. C'est lui qui DECLARE les couples :
        les quatre cles y figuraient litteralement, donc l'ancien controle
        trouvait toujours ce qu'il cherchait. Meme forme que D-159, ou
        l'inventaire etait dans le corpus qu'il fouillait.
    """
    moi = os.path.abspath(__file__)
    out = []
    for path in _fichiers_balayes():
        if os.path.abspath(path) == moi:
            continue
        if any(c == couple for _lineno, _url, c in _urls_a_mot_de_passe(path)):
            out.append(os.path.relpath(path, _ROOT))
    return out


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
    """Le perimetre d'origine ne doit pas disparaitre en s'elargissant.

    D-169 : le plancher a 10 ne detectait plus rien -- 25 fichiers de
    src/ mesures a `d816dee` (18 aout 2026)."""
    balayes = set(_fichiers_balayes())
    dans_src = [p for p in balayes if p.startswith(_SRC + os.sep)]
    assert len(dans_src) >= 25, (
        f"{len(dans_src)} fichiers de src/ balayes ; 25 mesures a "
        "`d816dee` (18 aout 2026) : le perimetre d'origine de ce test a "
        "ete perdu en l'elargissant")


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


def test_every_documented_placeholder_still_neutralises_a_real_url():
    """Une exemption perimee doit crier — D-161.

    Une exemption n'est legitime que si elle DESACTIVE quelque chose. Une
    entree qui ne neutralise aucune URL du depot n'est pas une exemption :
    c'est une permission dormante, accordee d'avance a tout secret qui
    porterait ce couple.
    """
    for couple, raison in _MODELES.items():
        assert raison.strip(), f"{couple} exempte sans raison"
        porteurs = _porteurs_reels(couple)
        assert porteurs, (
            f"le modele {couple!r} ne neutralise plus aucune URL du depot : "
            "retirer son entree de `_MODELES`, sinon elle exempte d'avance "
            "un vrai identifiant qui porterait ce couple. (Le couple peut "
            "encore apparaître en prose quelque part : ce n'est pas une URL, "
            "et cela ne justifie aucune exemption.)")


def test_l_inventaire_ne_se_porte_pas_lui_meme():
    """Epingle D-161 : sur quelle entree l'ancien controle echouait-il ?

    Sur aucune. Il faisait `couple in tout` avec `tout` = le texte de tous
    les fichiers balayes — **y compris celui-ci**, ou les couples sont
    declares. Les quatre cles y figurent litteralement ; le controle
    trouvait donc toujours sa propre declaration. Mesure du 18 aout 2026 :
    4 cles sur 4 auto-satisfaites, et **2 exemptions sur 4 ne neutralisaient
    deja plus rien** sans que personne ne le voie.
    """
    src = open(__file__, encoding="utf-8").read()
    moi = os.path.relpath(os.path.abspath(__file__), _ROOT)
    for couple in _MODELES:
        #  l'ancien critere : la cle est dans le source de l'inventaire
        assert '"%s"' % couple in src, (
            f"{couple} n'est plus declare ici — l'epinglage ne mesure plus "
            "l'ancien comportement")
        #  le nouveau : ce fichier n'est jamais son propre porteur
        assert moi not in _porteurs_reels(couple), (
            f"{moi} est redevenu porteur de {couple!r} : l'inventaire est "
            "rentre dans le corpus qu'il fouille, D-161 est rouvert")


def test_le_detecteur_de_porteurs_peut_rendre_vide():
    """Un balayage vide doit crier — y compris celui-ci.

    Si `_porteurs_reels` rendait toujours quelque chose, l'assertion
    ci-dessus ne prouverait rien. Un couple qu'aucune URL ne porte doit
    rendre la liste vide.
    """
    assert _porteurs_reels("temoin_d161:aucune_url_ne_le_porte") == []


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
