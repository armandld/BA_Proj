"""D-143 — le score intermédiaire d'Optuna et la référence qu'il consomme.

**Tests de déviation.** Ils n'épinglent pas un comportement correct : ils
épinglent un défaut **mesuré et non corrigé**, pour qu'il ne se perde pas et
qu'il ne se corrige pas en silence. Le jour où D-143 est tranché, ce fichier
rougit — c'est le jour où il doit être relu, pas contourné.

Le défaut, en une phrase : `pipeline()` incrémente `step` puis lit
`dns_trace[step - 1]` pour noter l'essai, si bien que le bras (à `t_step`)
est comparé à l'état DNS de `t_step-1`. Le score **final**, vingt lignes
plus bas, choisit `step if step in dns_trace else step - 1` — deux lectures
du même `dns_trace` avec deux conventions.

Le champ d'essai est choisi pour **SÉPARER** : `patch_ratio = 1,0`, donc le
bras reproduit la DNS exactement. Toute erreur rapportée non nulle ne peut
alors venir que de la référence, et de rien d'autre. Sur un bras inexact les
deux hypothèses rendraient toutes deux « un petit nombre » et le test ne
mesurerait rien.

Axes empruntés : `classical_only` ; `dns_trace` présent (départ à chaud) ;
`max_depth_override = 1` ; élagage branché (`trial` non nul) mais
`should_prune()` toujours faux, pour que le run aille au bout. Axes NON
empruntés, écrits pour ne pas être supposés : bras quantique, `dim > 2`,
`N = 256`, `HYBRID_DT = 0,10` — la configuration de campagne n'est pas
rejouée ici, son coût est hors budget d'une suite.

Mesures de référence (`ae394f0`, KH `N = 32`, deux exécutions identiques au
dernier chiffre) :

    phys_score FINAL (alignement juste)        3,055743e-15
    rapport step 21   annoncé 3,274533e-02     aligné 6,142937e-16
    rapport step 22   annoncé 3,197032e-02     aligné 1,201023e-15
    rapport step 23   annoncé 3,127157e-02     aligné 1,885542e-15
    rapport step 25   annoncé 3,055743e-15     (aligné par la réécriture)
    évolution propre de la DNS entre 2 instantanés   3,457654e-02

Les assertions portent sur des **ordres de grandeur**, pas sur ces valeurs :
le dépôt n'épingle aucune version de `numpy`/`scipy`. Les valeurs sont
écrites ici pour qu'une dérive se voie à la lecture.
"""
import os
import sys
import warnings

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if os.path.join(_REPO_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

FIELDS = ("vx", "vy", "Bx", "By", "Jz")


class _TrialSansElagage:
    """Un essai Optuna qui enregistre et n'élague jamais.

    Il faut `trial is not None` pour atteindre le bloc de notation
    intermédiaire ; il faut que `should_prune()` reste faux pour que le run
    atteigne aussi son score final, qui est le terme de comparaison.
    """

    def __init__(self):
        self.reports = []
        self.attrs = {}

    def report(self, value, step):
        self.reports.append((int(step), float(value)))

    def should_prune(self):
        return False

    def set_user_attr(self, cle, valeur):
        self.attrs[cle] = valeur


def _ecart_max(a, b):
    """Écart maximum champ à champ, en norme infinie."""
    return max(float(np.max(np.abs(a[f] - b[f]))) for f in FIELDS)


@pytest.fixture(scope="module")
def run_note():
    """Un run complet, avec `pipeline.score` espionné.

    L'espion ne réimplémente rien : il délègue à la vraie fonction et garde
    une copie des DEUX tableaux qu'elle a reçus. C'est ce qui permet de
    prouver l'alignement par identité de tableaux plutôt que par
    ressemblance de nombres.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import train_hyperparams as TH
        from Simulation.pre_compute_dns import precompute_dns
        import pipeline as P

        cfg = {**TH.SCENARIO_KH, "N": 32, "T_START": 0.9, "T_MAX": 1.2,
               "HYBRID_DT": 0.02, "K_opt": 4, "shots": 32,
               "max_depth_override": 1, "study_name": "dns_kh"}
        trace, hot = precompute_dns(cfg)
        hp = {**{n: (lo + hi) / 2 for n, (lo, hi, _) in TH.SEARCH_SPACE.items()},
              **TH.FIXED_PARAMS}

        vrai_score = P.score
        appels = []

        def espion(q, r, *a, **k):
            appels.append(({x: y.copy() for x, y in q.items()},
                           {x: y.copy() for x, y in r.items()}))
            return vrai_score(q, r, *a, **k)

        trial = _TrialSansElagage()
        P.score = espion
        try:
            final = P.pipeline(
                N=cfg["N"], VQA_N=2, T_MAX=cfg["T_MAX"], DT=cfg["DT"],
                HYBRID=int(cfg["HYBRID_DT"] / cfg["DT"]), verbose=False,
                argus=TH.create_argus(cfg), hyperparams=hp, lambda_cost=0.4,
                trial=trial, dns_trace=trace, hot_start_state=hot,
                max_depth_override=cfg["max_depth_override"],
                scenario=cfg["scenario"], return_details=True,
                classical_only=True)
        finally:
            P.score = vrai_score

    return {"trace": trace, "reports": trial.reports, "appels": appels,
            "final": final, "score": vrai_score}


@pytest.fixture(scope="module")
def rapports_propres(run_note):
    """Les rapports dont la référence alignée est utilisable.

    Tout rapport dont la référence alignée serait le DERNIER index de la
    trace est écarté : `pre_compute_dns.py:126` réécrit cette entrée après
    la boucle avec l'état de fin de run (« AJOUT CRITIQUE »), donc
    `trace[dernier]` ne décrit plus l'instant `t_dernier` et ne peut servir
    de terme de comparaison. Mesuré : au rapport 24 le bras diffère de
    `trace[24]` de 6,964e-04, pas de l'epsilon machine — c'est la
    réécriture qu'on lirait, pas le décalage. Restent les rapports 21 à 23.
    """
    trace = run_note["trace"]
    dernier = max(trace)
    propres = []
    for (step, valeur), (q, r) in zip(run_note["reports"], run_note["appels"]):
        if step >= dernier or step not in trace:
            continue
        if "fluxes" not in trace[step] or "fluxes" not in trace.get(step - 1, {}):
            continue
        propres.append({"step": step, "valeur": valeur, "q": q, "r": r})
    return propres


# ══════════════════════════════════════════════════════════════════
#  1. La mesure avant d'accuser le code : le bras est-il bien exact ?
# ══════════════════════════════════════════════════════════════════

def test_the_arm_reproduces_the_dns_so_a_reported_error_can_only_be_the_reference(run_note):
    """Sans ceci, tout le reste du fichier mesurerait autre chose.

    `patch_ratio = 1,0` : tout est raffiné, le bras intègre à la résolution
    de la DNS et la reproduit. Le score final le confirme à 3,055743e-15.
    Si un jour ce test rougit, les suivants ne prouvent plus rien — les
    relire AVANT de toucher au code qu'ils accusent.
    """
    final = run_note["final"]
    assert final["patch_ratio"] == pytest.approx(1.0, abs=1e-12), (
        f"le bras ne raffine plus tout ({final['patch_ratio']}) : le champ "
        "d'essai ne sépare plus les deux alignements")
    assert final["phys_score"] < 1e-9, (
        f"le bras n'est plus exact (phys_score = {final['phys_score']:.6e}, "
        "mesuré 3,055743e-15) : une erreur rapportée ne s'attribue plus à "
        "la seule référence")


def test_the_sweep_is_not_empty(rapports_propres):
    """Un balayage vide doit crier — trois rapports propres, mesurés."""
    assert len(rapports_propres) >= 3, (
        f"{len(rapports_propres)} rapport(s) intermédiaire(s) exploitable(s), "
        "3 attendus (steps 21, 22, 23) : le run ne traverse plus le bloc de "
        "notation d'Optuna, et ce fichier ne mesure plus rien")


# ══════════════════════════════════════════════════════════════════
#  2. D-143 — le décalage, prouvé par identité de tableaux
# ══════════════════════════════════════════════════════════════════

def test_the_intermediate_reference_is_the_snapshot_of_the_previous_step(rapports_propres, run_note):
    """DÉVIATION D-143. Preuve par identité, pas par ressemblance.

    La référence consommée est bit-à-bit `dns_trace[step-1]['fluxes']`
    (écart 0,000e+00 aux trois rapports), et le bras est bit-à-bit
    `dns_trace[step]['fluxes']` (4,441e-16 à 8,882e-16). Les deux
    instantanés diffèrent, eux, de 1,789e-03 à 1,825e-03 : c'est l'entrée
    qui SÉPARE les deux conventions d'index.

    Rougit le jour où l'index est aligné — donc le jour où D-143 est
    tranché.
    """
    trace = run_note["trace"]
    for r in rapports_propres:
        k = r["step"]
        avant = trace[k - 1]["fluxes"]
        aligne = trace[k]["fluxes"]

        assert _ecart_max(r["r"], avant) == 0.0, (
            f"step {k} : la référence n'est plus l'instantané du pas "
            f"précédent — D-143 a peut-être été corrigé, relire l'entrée")
        assert _ecart_max(r["q"], aligne) < 1e-12, (
            f"step {k} : le bras ne coïncide plus avec dns_trace[{k}], "
            f"écart {_ecart_max(r['q'], aligne):.3e}")
        assert _ecart_max(avant, aligne) > 1e-5, (
            f"step {k} : les deux instantanés ne se distinguent plus "
            f"({_ecart_max(avant, aligne):.3e}) — le champ d'essai a cessé "
            "de séparer, ce test ne prouverait plus rien")


def test_the_intermediate_score_reports_the_dns_own_motion_not_the_arm_error(rapports_propres, run_note):
    """DÉVIATION D-143. L'écart chiffré, mesuré à l'opérateur assorti.

    On recalcule la MÊME grandeur avec `score` lui-même — jamais une
    réimplémentation — contre l'instantané du bon instant. Mesuré :
    3,274533e-02 / 3,197032e-02 / 3,127157e-02 annoncés contre
    6,142937e-16 / 1,201023e-15 / 1,885542e-15 alignés, soit 1,7e13 à
    5,3e13.

    L'assertion porte sur la séparation des ordres, pas sur ces valeurs.
    """
    score = run_note["score"]
    trace = run_note["trace"]
    for r in rapports_propres:
        k = r["step"]
        annonce = score(r["q"], r["r"], 0.0, 1, 1, 1)["phys_score"]
        aligne = score(r["q"], trace[k]["fluxes"], 0.0, 1, 1, 1)["phys_score"]

        assert aligne < 1e-9, (
            f"step {k} : l'alignement juste ne rend plus zéro "
            f"({aligne:.6e}) — remesurer avant de conclure")
        assert annonce > 1e-3, (
            f"step {k} : l'erreur annoncée est tombée à {annonce:.6e}. "
            "D-143 a peut-être été corrigé — relire son entrée dans "
            "DEFAUTS.md avant de toucher à ce test")


def test_the_two_readings_of_dns_trace_disagree_inside_one_function(run_note):
    """Question 4 : le score FINAL, lui, est aligné.

    C'est le contraste qui fait de D-143 un défaut et non une convention :
    dans la même fonction, la notation finale préfère `step` et la notation
    intermédiaire lit `step - 1` sans repli. Le final est mesuré exact
    (3,055743e-15) là où les intermédiaires annoncent 3,1e-02.
    """
    propres = [v for s, v in run_note["reports"]
               if v > 0.29]  # combined = (phys + 0,4)/1,4, phys ~ 3,1e-02
    assert propres, "aucun rapport intermédiaire n'est au-dessus du plancher"
    assert run_note["final"]["phys_score"] < 1e-9, (
        "le score final n'est plus aligné : c'est une régression AUTRE que "
        "D-143, et elle passe avant")


def test_the_last_report_is_aligned_only_by_the_end_of_run_overwrite(run_note):
    """Le dernier rapport est juste — par accident, et il faut le savoir.

    `pre_compute_dns.py:126` réécrit la dernière entrée de la trace avec
    l'état de FIN de run. Cette entrée ne suit donc pas la convention des
    autres, et c'est elle que lit le dernier rapport : il tombe aligné.
    Mesuré 3,055743e-15, contre 3,1e-02 pour les précédents.

    Écrit pour qu'une passe future ne conclue pas « le dernier rapport est
    bon, donc le bloc l'est » — un seul point aligné sur cinq.
    """
    reports = run_note["reports"]
    assert len(reports) >= 2, "il faut au moins deux rapports pour comparer"
    dernier = reports[-1][1]
    precedents = [v for _, v in reports[:-1]]
    assert dernier < min(precedents) - 1e-3, (
        f"le dernier rapport ({dernier:.6e}) ne se détache plus des "
        f"précédents ({min(precedents):.6e}) : la réécriture de fin de run "
        "a peut-être changé, relire pre_compute_dns.py")


# ══════════════════════════════════════════════════════════════════
#  3. La déviation reste écrite là où elle vit
# ══════════════════════════════════════════════════════════════════

def test_the_open_defect_stays_written_in_the_registry():
    """« Ne jamais laisser une déviation connue non écrite » — VIGIL.md.

    Une déviation non consignée se fait recorriger par erreur. Ce test
    interroge le registre, pas la mise en forme : il exige l'entrée D-143
    et le fait qu'elle soit encore déclarée non corrigée.
    """
    defauts = os.path.join(_REPO_ROOT, "docs", "DEFAUTS.md")
    texte = open(defauts, encoding="utf-8").read()
    assert "## D-143" in texte, (
        "D-143 a quitté DEFAUTS.md : s'il est corrigé, ce fichier de "
        "déviation doit être remesuré et réécrit, pas laissé en place")
    debut = texte.index("## D-143")
    corps = texte[debut:debut + 400]
    assert "rien n'est corrigé" in corps.lower(), (
        "l'entrée D-143 ne se déclare plus « rapport seul » — la décision a "
        "peut-être été prise ; relire avant de faire passer ce fichier")
