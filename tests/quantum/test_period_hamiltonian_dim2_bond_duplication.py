"""D-59 — le lien ZZ dupliqué à `dim = 2`, et pourquoi il fallait le corriger
AVANT la campagne.

L'Hamiltonien est périodique : chaque cellule émet un lien ZZ vers sa
voisine, `(i,j) -> (i,j+1 mod dim)`. À `dim >= 3` cela fait `dim` liens
distincts par direction. À **`dim = 2`** l'anneau dégénère : `(i,0)->(i,1)`
et `(i,1)->(i,0 mod 2)` relient la **même paire de qubits**, et les deux
itérations ajoutaient chacune une entrée au lieu d'être fusionnées.

Les coefficients étant symétriques par construction
(`C_edges[0][i,0] == C_edges[0][i,1]` au bit près), le couplage shear était
appliqué **deux fois** : poids effectif ×2. `K_plaquettes` (ZZZZ) n'a pas
ce défaut — les 4 quadruplets à `dim = 2` sont distincts deux à deux.

Repéré parce que le décompte des termes du `SparsePauliOp` affichait
`"IIIIIIZZ"` **deux fois**, avec exactement `-2.4290271580758453` des deux
côtés.

## Ce que la correction change, mesuré

Coefficients ALÉATOIRES (donc ZZ vivant), comparaison terme à terme :

    dim = 2   ZZ 8 -> 4    opérateurs identiques : NON  (max|dH| = 3,285)
    dim = 3   ZZ 18 -> 18  opérateurs identiques : OUI
    dim = 4   ZZ 32 -> 32  opérateurs identiques : OUI
    dim = 5   ZZ 50 -> 50  opérateurs identiques : OUI

La correction ne mord donc **qu'à `dim = 2`**, ce qui est exactement sa
portée annoncée.

## Ce qu'elle ne change pas AUJOURD'HUI, et pourquoi

Aux hyperparamètres DÉPLOYÉS, sur les 4 scénarios canoniques × 3
instantanés (Re=400, N=256) :

    décisions de fondamental exact changées : 0 / 12
    max|ΔE| global                          : 0,000e+00  (EXACTEMENT nul)

Zéro exact, pas « petit ». La raison est D-47 : la fenêtre gaussienne du
couplage ZZ vaut au plus **1,15e−31** au réglage déployé (le score est à
8,4 σ du seuil), donc `|C_edges| < 1e-6` et **aucun terme ZZ n'est émis**.
Dédupliquer n'a littéralement rien à retirer.

Noté au passage, et cohérent avec D-47 : le fondamental vaut **255** sur
les 12 instantanés — 8 qubits tous à 1, « raffiner partout ».

## Pourquoi corriger avant la campagne et non après

C'est le raisonnement qui justifie de toucher `src/` maintenant :

- l'impact est mesuré **nul** aujourd'hui, donc **aucun nombre publié ne
  bouge** — la correction est gratuite ;
- la réoptimisation rééquilibre précisément les poids qui rendent le
  défaut invisible. Si `w_z_frac` se resserre ou `σ` s'élargit — ce que la
  campagne peut choisir — le ZZ redevient actif et le facteur 2 devient
  réel, à `dim = 2`, **la seule taille de toutes les campagnes publiées** ;
- corriger après coup obligerait à rejouer toute la campagne.

Le premier test ci-dessous mesure sur des coefficients aléatoires, donc
avec ZZ VIVANT : c'est le seul régime où la correction est observable, et
un test écrit sur les coefficients déployés passerait à vide.
"""

import sys
import os

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_REPO, "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qiskit.quantum_info import SparsePauliOp          # noqa: E402
from VQA.cost_hamiltonian import create_period_hamiltonian  # noqa: E402


def _coefficients(dim, seed=0):
    """Coefficients ALÉATOIRES : ZZ vivant, donc la correction observable."""
    rng = np.random.default_rng(seed)
    return {"H_edges": [rng.normal(size=(dim, dim)), rng.normal(size=(dim, dim))],
            "C_edges": [rng.normal(size=(dim, dim)), rng.normal(size=(dim, dim))],
            "K_plaquettes": rng.normal(size=(dim, dim))}


def _sans_deduplication(hp, dim):
    """L'émission d'AVANT D-59, rejouée pour mesurer l'écart."""
    sl = []
    off = dim * dim

    def iH(y, x): return (y % dim) * dim + (x % dim)
    def iV(y, x): return off + (y % dim) * dim + (x % dim)

    for i in range(dim):
        for j in range(dim):
            for arr, idx in ((hp['H_edges'][0], iH), (hp['H_edges'][1], iV)):
                v = arr[i, j]
                if abs(v) > 1e-6:
                    sl.append(("Z", [idx(i, j)], v))
            c = hp['C_edges'][0][i, j]
            if abs(c) > 1e-6:
                sl.append(("ZZ", [iH(i, j), iH(i, j + 1)], c))
            c = hp['C_edges'][1][i, j]
            if abs(c) > 1e-6:
                sl.append(("ZZ", [iV(i, j), iV(i + 1, j)], c))
            k = hp['K_plaquettes'][i, j]
            if abs(k) > 1e-6:
                sl.append(("ZZZZ", [iH(i, j), iV(i, j + 1),
                                    iH(i + 1, j), iV(i, j)], k))
    return SparsePauliOp.from_sparse_list(sl, num_qubits=2 * dim * dim)


def _termes(op):
    o = op.simplify()
    return sorted((str(p), round(float(c.real), 12))
                  for p, c in zip(o.paulis, o.coeffs))


def _compte_zz(op):
    return sum(1 for p in op.paulis if str(p).count("Z") == 2)


# ══════════════════════════════════════════════════════════════════
#  1. Plus aucun lien ZZ dupliqué, à aucune dimension
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("dim", [2, 3, 4, 5])
def test_aucune_paire_de_qubits_ne_recoit_deux_liens_zz(dim):
    op = create_period_hamiltonian(_coefficients(dim), dim)
    zz = [str(p) for p in op.paulis if str(p).count("Z") == 2]
    assert zz, "aucun terme ZZ : le champ d'essai ne vérifierait rien"
    doublons = len(zz) - len(set(zz))
    assert doublons == 0, (
        f"dim={dim} : {doublons} libellé(s) ZZ répété(s). Une paire de "
        f"qubits qui reçoit deux entrées voit son couplage doublé (D-59).")


# ══════════════════════════════════════════════════════════════════
#  2. La correction mord à dim=2 et NULLE PART ailleurs
# ══════════════════════════════════════════════════════════════════

def test_a_dim2_le_nombre_de_liens_zz_est_divise_par_deux():
    """MESURE de la correction, sur des coefficients où ZZ est vivant."""
    hp = _coefficients(2)
    avant, apres = _sans_deduplication(hp, 2), create_period_hamiltonian(hp, 2)
    assert _compte_zz(avant) == 8, f"attendu 8 ZZ avant, vu {_compte_zz(avant)}"
    assert _compte_zz(apres) == 4, f"attendu 4 ZZ après, vu {_compte_zz(apres)}"
    assert _termes(avant) != _termes(apres), (
        "l'opérateur est inchangé à dim=2 : la déduplication ne mord pas")


@pytest.mark.parametrize("dim", [3, 4, 5])
def test_a_dim_superieur_l_operateur_est_inchange_bit_a_bit(dim):
    """La correction ne doit RIEN changer là où l'anneau ne dégénère pas.

    Sans ce test, dédupliquer pourrait retirer des liens légitimes à
    `dim >= 3` sans que rien ne le signale — et `dim = 4` est la taille
    des campagnes de coefficients.
    """
    hp = _coefficients(dim)
    assert _termes(_sans_deduplication(hp, dim)) == \
        _termes(create_period_hamiltonian(hp, dim)), (
        f"dim={dim} : l'opérateur a changé alors que l'anneau périodique "
        "n'y dégénère pas — la déduplication mord au-delà de sa portée")


# ══════════════════════════════════════════════════════════════════
#  3. Le champ d'essai sépare — sinon rien de ce qui précède ne vaut
# ══════════════════════════════════════════════════════════════════

def test_le_champ_d_essai_a_bien_du_zz_vivant():
    """Aux hyperparamètres DÉPLOYÉS la fenêtre gaussienne annule ZZ
    (|C| < 1e-6, voir D-47) : un test écrit sur eux passerait à vide, la
    déduplication n'ayant rien à retirer. D'où les coefficients aléatoires.
    """
    hp = _coefficients(2)
    C0, C1 = hp["C_edges"]
    assert np.max(np.abs(C0)) > 1e-6 and np.max(np.abs(C1)) > 1e-6, (
        "le champ d'essai n'a pas de couplage ZZ au-dessus du seuil "
        "d'émission : il ne pourrait pas voir la correction")
