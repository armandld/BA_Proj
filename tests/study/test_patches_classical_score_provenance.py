"""D-77 : le champ `classical_scores` de 84 des 156 artefacts `patches_*.npz`
fige la convention de rotationnel d'AVANT D-1 — celle que D-1 a mesuree fausse.

D-1 (`bb6a387`, 11 aout) a bascule `fixed_curl` sur la convention declaree par
`grid.py` : sur une rotation solide, l'ancienne forme rend 0 la ou la vraie
rend +2. `classical_score` en depend par deux de ses quatre indicateurs
(vorticite, divergence). Les artefacts `patches_*` produits **avant** cette
date n'ont jamais ete regeneres : leur `classical_scores` reste celui de
l'ancienne convention, dans le meme fichier que des `l2_errors` qui, eux, se
reproduisent — ils ne dependent d'aucun rotationnel.

Mesure (une passe sur les 156 artefacts, dernier instantane de chacun) :

    72 fichiers (dim 2/4/8, commit du 11 aout)   ecart a HEAD = 0.000e+00
    84 fichiers (dim 3/16/32/64 + variantes)     ecart a HEAD jusqu'a 3.8e-01
      dont 50 reproduits BIT A BIT par `fixed_curl=False` (<= 1e-12)

    labels du meme fichier (dim16, harris_tearing, N=256) :
      l2_errors    max|ancien - regenere| = 9.4e-12   (plancher float32)
      is_hard      0 desaccord / 5120
      l2_threshold exact

**Rien de publie ne bouge aujourd'hui** : le seul consommateur du champ,
`pipeline_verification`, ne tourne qu'a dim=4, ou le fichier de base EST a
jour — sa sortie complete est identique avant/apres regeneration (verifie).
Le piege est pour le consommateur suivant.

Ce test ne regenere pas les artefacts : les regenerer changerait des fichiers
publies, ce qui se signale et ne s'applique pas. Il **epingle l'etat mesure**,
pour qu'une regeneration future oblige a mettre le registre a jour et qu'un
artefact perime neuf soit vu tout de suite.
"""
import glob
import os
import re

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_RESULTS = os.path.join(_REPO_ROOT, "results")

#: Les quatre seules paires (base, variante) du depot qui decrivent la MEME
#: configuration et portent deux scores classiques differents : la base a ete
#: regeneree le 11 aout (convention corrigee), la variante date du 9 et copie
#: le score de l'ancienne. Mesure : ecart max sur le tableau entier.
_KNOWN_DIVERGENT_PAIRS = {
    "patches_harris_tearing_Re400_N256_dim4_globalthr.npz": 1.6913e-01,
    "patches_kelvin_helmholtz_Re400_N256_dim4_globalthr.npz": 1.7919e-01,
    "patches_mhd_rotor_Re400_N256_dim4_globalthr.npz": 2.5558e-01,
    "patches_orszag_tang_Re400_N256_dim4_globalthr.npz": 4.0897e-01,
}

_VARIANT_RE = re.compile(r"(_globalthr|_tau[0-9p]+)\.npz$")


def _variant_pairs():
    out = []
    for f in sorted(glob.glob(os.path.join(_RESULTS, "patches_*.npz"))):
        base = _VARIANT_RE.sub(".npz", f)
        if base != f and os.path.exists(base):
            out.append((os.path.basename(f), f, base))
    return out


PAIRS = _variant_pairs()


def test_the_sweep_itself_is_not_empty():
    """Un balayage vide doit crier — y compris celui-ci."""
    assert len(PAIRS) >= 50, (
        f"seulement {len(PAIRS)} paires (variante, base) trouvees dans "
        "results/ : c'est le motif de nommage qui a change, pas le depot qui "
        "a perdu ses variantes")


@pytest.mark.parametrize("name,vpath,bpath", PAIRS,
                         ids=[p[0] for p in PAIRS])
def test_a_variant_carries_the_same_classical_score_as_its_base(name, vpath, bpath):
    """Deux fichiers, une seule configuration : le score classique doit y etre
    le meme. Les quatre exceptions sont nommees ET chiffrees — si l'ecart
    bouge, c'est qu'on a regenere, et le registre doit suivre."""
    a = np.load(bpath)["classical_scores"]
    b = np.load(vpath)["classical_scores"]
    gap = float(np.max(np.abs(a - b)))
    if name in _KNOWN_DIVERGENT_PAIRS:
        expected = _KNOWN_DIVERGENT_PAIRS[name]
        assert abs(gap - expected) < 1e-4, (
            f"{name} : l'ecart mesure ({gap:.4e}) n'est plus celui consigne "
            f"pour D-77 ({expected:.4e}). Si l'artefact a ete regenere, mettre "
            "a jour D-77 dans docs/RESULTS.md plutot que ce nombre seul.")
        return
    assert gap < 1e-12, (
        f"{name} : le score classique differe de son fichier de base de "
        f"{gap:.4e}. Meme configuration, deux valeurs — voir D-77.")


def _score_at(dns_path, si, dim, fixed_curl):
    """Score classique a la resolution `dim`, par le chemin du depot."""
    import sys
    for p in [os.path.join(_REPO_ROOT, "src"),
              os.path.join(_REPO_ROOT, "study", "pipeline")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from Simulation.PhysToAngle import AngleMapper

    dns = np.load(dns_path)
    vx, vy, Bx, By = (dns[k][si].astype(np.float64)
                      for k in ("vx", "vy", "Bx", "By"))
    N = vx.shape[0]
    dx = 2 * np.pi / N
    #  D-153 : les axes sont NOMMES — `grid.py` fait foi.
    from Simulation.grid import AXIS_X, AXIS_Y
    Jz = ((np.roll(By, -1, axis=AXIS_X) - np.roll(By, 1, axis=AXIS_X)) / (2 * dx)
          - (np.roll(Bx, -1, axis=AXIS_Y) - np.roll(Bx, 1, axis=AXIS_Y)) / (2 * dx))
    full = AngleMapper.classical_score(
        {"vx": vx, "vy": vy, "Bx": Bx, "By": By, "Jz": Jz, "dx": dx},
        fixed_curl=fixed_curl)
    ps = N // dim
    return full.reshape(dim, ps, dim, ps).max(axis=(1, 3))


#: Deux artefacts du MEME instantane DNS (N=64, orszag_tang, Re=400), l'un
#: regenere apres D-1, l'autre pas. Le champ separe : au dernier instantane
#: les deux conventions donnent des scores tres differents. (Au premier
#: instantane elles coincident — c'est le piege « champ qui ne separe pas »,
#: mesure ici a 0.0 sur snap 0 avant de choisir le dernier.)
@pytest.mark.parametrize("fname,dim,expect_head", [
    ("patches_orszag_tang_Re400_N64_dim4.npz", 4, True),
    ("patches_orszag_tang_Re400_N64_dim16.npz", 16, False),
])
def test_which_curl_convention_each_artefact_was_written_with(fname, dim, expect_head):
    """Isole la CAUSE, pas seulement l'ecart : l'artefact perime est reproduit
    bit a bit par `fixed_curl=False`, celui a jour par `fixed_curl=True`."""
    path = os.path.join(_RESULTS, fname)
    dns_path = os.path.join(_RESULTS, "dns_orszag_tang_Re400_N64.npz")
    if not (os.path.exists(path) and os.path.exists(dns_path)):
        pytest.skip("artefacts absents de ce checkout")

    pat = np.load(path)
    si = len(pat["t"]) - 1
    stored = pat["classical_scores"][si]
    d_head = float(np.max(np.abs(_score_at(dns_path, si, dim, True) - stored)))
    d_old = float(np.max(np.abs(_score_at(dns_path, si, dim, False) - stored)))

    if expect_head:
        assert d_head < 1e-12, (
            f"{fname} ne se reproduit plus a HEAD (ecart {d_head:.3e}) : il "
            "etait a jour au moment de D-77, quelque chose a change depuis")
        assert d_old > 1e-3, (
            "le champ d'essai ne SEPARE plus les deux conventions : ce test "
            "ne mesurerait rien. Choisir un autre instantane.")
    else:
        assert d_old < 1e-12, (
            f"{fname} n'est plus reproduit par `fixed_curl=False` (ecart "
            f"{d_old:.3e}) : la cause consignee en D-77 n'est plus la bonne")
        assert d_head > 1e-3, (
            f"{fname} coincide maintenant avec HEAD (ecart {d_head:.3e}) : "
            "l'artefact a ete regenere — retirer l'entree D-77 plutot que "
            "relacher ce seuil")
