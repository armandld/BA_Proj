"""D-172 -- la reference imprimee par rho_gap_f1.py ne rejouait pas la
mesure qu'elle nomme.

Le docstring et la banniere de `main()` citaient +0.970 (p=0.0001,
8 solveurs) comme "mesure de reference, avant campagne" sur
`h0_optimiser_equivalence_N96_dim3_hamiltonien_corrige.npz`. `RESULTS.md`
avait deja retracte ce nombre ("rho vaut +0,870, pas +0,970" -- le calcul
manuel excluait `qaoa_shots_p3`) mais la correction n'avait jamais atteint
le module qui sert de critere pre-enregistre a la campagne : quiconque
lance `python study/common/rho_gap_f1.py` voyait encore la valeur retractee
au-dessus de la mesure reelle, correcte, affichee juste en dessous.

Deux choses distinctes sont epinglees :

1. le calcul lui-meme, sur l'artefact que le docstring nomme, rend
   +0.870 / 9 solveurs -- pas +0.970 / 8 ;
2. la banniere imprimee par `main()` ne reporte plus la valeur retractee.
"""

import os
import re
import subprocess
import sys

import pytest

pytest.importorskip("scipy")
pytest.importorskip("numpy")


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "src")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("racine du depot introuvable")


ROOT = _repo_root()
ARTEFACT = os.path.join(
    ROOT, "results",
    "h0_optimiser_equivalence_N96_dim3_hamiltonien_corrige.npz",
)


@pytest.mark.skipif(not os.path.exists(ARTEFACT), reason="artefact absent")
def test_rho_on_the_named_artefact_is_0_870_not_0_970():
    sys.path.insert(0, os.path.join(ROOT, "study", "common"))
    from rho_gap_f1 import rho_gap_f1

    r = rho_gap_f1(ARTEFACT)
    assert r["n_solveurs"] == 9
    assert r["rho"] == pytest.approx(0.870, abs=1e-3)
    assert r["rho"] != pytest.approx(0.970, abs=1e-3)


@pytest.mark.skipif(not os.path.exists(ARTEFACT), reason="artefact absent")
def test_the_printed_reference_banner_matches_the_measured_value():
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "study", "common", "rho_gap_f1.py"),
         ARTEFACT],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    banner = out.stdout.split("\n")[2]
    assert "0.970" not in banner
    assert "0.870" in banner

    mesure = next(l for l in out.stdout.split("\n") if l.strip().startswith("rho ="))
    m = re.search(r"rho = ([+-]?\d+\.\d+)", mesure)
    assert m is not None
    rho_mesure = float(m.group(1))

    m_banner = re.search(r"rho = ([+-]?\d+\.\d+)", banner)
    assert m_banner is not None
    rho_banner = float(m_banner.group(1))

    assert rho_banner == pytest.approx(rho_mesure, abs=1e-3)
