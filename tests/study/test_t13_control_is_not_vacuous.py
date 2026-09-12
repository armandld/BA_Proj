"""D-54 / D-55 : le controle de T13 ne validait rien, et son balayage vide
sortait avec le code 0.

D-54. `zero_hamiltonian_terms(hp, ())` rend une copie de `hp` : le masque du
controle sort de la MEME fonction sur la MEME entree que la reference. Le
controle vaut 0 par construction. `RESULTS.md` en tirait pourtant « The
control is exactly 0, which validates the measurement chain ».

Mesure : en sabotant `TERM_KEYS` pour que plus rien ne soit jamais mis a
zero (orszag_tang Re=400 N=64 dim=2), le controle rend 0,000000 des DEUX
cotes et `no_ZZ` / `no_ZZZZ` / `Z_only` rendent 0,0000 des deux cotes — les
trois lignes memes qui portent la lecture « causalement inertes ». Ce n'est
pas hypothetique : D-51 a montre que `no_ZZZZ` annule `K_xpoint`, une cle que
`ground_state_mask` ne lit jamais.

D-55. Sans artefact d'entree, `main()` imprimait « no input. » et rendait 0
sans ecrire d'artefact — laissant en place celui de la campagne precedente.
Son voisin `h0_optimiser_equivalence` avait deja corrige le meme defaut.
"""
import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in [os.path.join(_REPO_ROOT, "src")] + [
        os.path.join(_REPO_ROOT, "study", _d) for _d in (
            "pipeline", "h0_selection", "h1_solver", "h2b_prediction",
            "h3_representation", "h4_transfer", "closed_loop", "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import h3_term_ablation as t13                     # noqa: E402


def _rows(spec):
    """spec : {ablation: (changed, removed_max)} -> lignes au format de main."""
    return [dict(ablation=name, changed=ch, removed_max=rm)
            for name, (ch, rm) in spec.items()]


# ------------------------------------------------------------------ D-54
def test_control_is_zero_by_construction_on_any_hamiltonian():
    """Epingle le MECANISME : le controle compare f(x) a f(x). Sur un
    hamiltonien quelconque — y compris un hamiltonien nul — l'ablation
    « full » ne touche rien, donc son ecart est nul. C'est pour cela qu'il
    ne peut rien detecter."""
    dim = 2
    rng = np.random.default_rng(0)
    for _ in range(20):
        hp = {"H_edges": (rng.normal(size=(dim, dim)), rng.normal(size=(dim, dim))),
              "C_edges": (rng.normal(size=(dim, dim)), rng.normal(size=(dim, dim))),
              "K_plaquettes": rng.normal(size=(dim, dim))}
        assert t13.coefficients_removed(hp, t13.zero_hamiltonian_terms(hp, ()), dim) == 0.0


def test_removed_max_separates_a_real_ablation_from_an_empty_one():
    """La grandeur qui SEPARE les deux hypotheses que le controle confond.
    Mesuree avec l'operateur assorti : ce que `build_ising_terms` produit
    reellement, pas les cles de `hamilt_params`."""
    dim = 2
    hp = {"H_edges": (np.full((dim, dim), 0.5), np.full((dim, dim), 0.5)),
          "C_edges": (np.full((dim, dim), 2.0), np.full((dim, dim), 2.0)),
          "K_plaquettes": np.full((dim, dim), 3.0)}

    # ablation reelle : le ZZ disparait de l'operateur
    real = t13.zero_hamiltonian_terms(hp, ("ZZ",))
    assert t13.coefficients_removed(hp, real, dim) == pytest.approx(2.0)

    # K_xpoint is emitted on the same plaquette as K_plaquettes. The
    # comparison must aggregate duplicate operator indices.
    hp_x = dict(hp, K_xpoint=np.full((dim, dim), 42.0))
    empty = t13.zero_hamiltonian_terms(hp_x, ("ZZZZ",))
    empty["K_plaquettes"] = hp_x["K_plaquettes"]      # seul K_xpoint retire
    assert t13.coefficients_removed(hp_x, empty, dim) == pytest.approx(42.0)


def test_empty_ablation_is_named_instead_of_read_as_inert():
    """Le coeur de D-54 : `changed = 0` avec `removed_max = 0` ne dit rien
    de l'inertie du terme, et doit etre imprime comme tel."""
    msg = t13.control_and_reading(_rows({
        "full": (0.0, 0.0),
        "no_Z": (1.0, 0.083),
        "no_ZZ": (0.0, 0.0),          # rien retire
        "no_ZZZZ": (0.0, 1.0),        # retire pour de vrai, sans effet
    }))
    assert "EMPTY ABLATIONS" in msg
    assert "no_ZZ" in msg.split("EMPTY ABLATIONS")[1].split("\n")[0]
    assert "no_ZZZZ" not in msg.split("EMPTY ABLATIONS")[1].split("\n")[0]


def test_all_ablations_effective_prints_no_empty_warning():
    msg = t13.control_and_reading(_rows({
        "full": (0.0, 0.0),
        "no_Z": (1.0, 0.083),
        "no_ZZ": (0.0, 2.6558),
        "no_ZZZZ": (0.0, 1.0),
    }))
    assert "EMPTY ABLATIONS" not in msg
    assert "READING:" in msg


def test_control_is_now_checked_not_merely_printed():
    """Avant, `ctrl` etait imprime avec « (must be 0.0) » et rien ne
    l'exigeait."""
    with pytest.raises(RuntimeError, match="au lieu de 0"):
        t13.control_and_reading(_rows({"full": (0.25, 0.0),
                                       "no_Z": (1.0, 0.083)}))


def test_missing_control_row_raises():
    with pytest.raises(RuntimeError, match="aucune ligne de controle"):
        t13.control_and_reading(_rows({"no_Z": (1.0, 0.083)}))


def test_module_now_carries_an_assertion():
    """`CLAUDE.md` : « Un test qui ne peut pas echouer est un defaut. Tout
    script de study/ ou de tests/ porte une assertion. » Mesure par AST, pas
    par recherche de chaine : le module en portait **0** (0 assert, 0 raise,
    0 SystemExit) contre 5 et 6 pour son voisin."""
    import ast
    with open(t13.__file__, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    guards = [n for n in ast.walk(tree)
              if isinstance(n, (ast.Assert, ast.Raise))]
    assert len(guards) >= 3, (
        f"h3_term_ablation.py ne porte plus que {len(guards)} garde(s) : "
        "le controle de T13 et le cri du balayage vide ont ete retires")


# ------------------------------------------------------------------ D-55
def test_empty_sweep_raises_instead_of_returning_zero(monkeypatch, capsys):
    """Le balayage vide doit crier. Verifie sur le vrai `main()`, avec un
    scenario sans artefact — c'est-a-dire exactement la commande qui rendait
    le code 0."""
    argv = ["h3_term_ablation.py", "--scenario", "no_such_scenario",
            "--N", "64", "--dim", "2", "--n-snaps", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="balayage vide"):
        t13.main()
    assert "SKIP no_such_scenario" in capsys.readouterr().out
