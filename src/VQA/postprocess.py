# scripts/postprocess.py
import numpy as np


def postprocess(distribution, num_qubits, verbose):
    """
    Convertit une DISTRIBUTION {bitstring: probabilite} en marginales
    [P(q0=1), P(q1=1), ...].

    Convention de bits : Qiskit ecrit le qubit 0 A DROITE de la chaine, d'ou
    le parcours `bitstring[::-1]`. C'est bien la convention de
    `Statevector.probabilities_dict()` et de `get_counts()` ; l'ancien
    commentaire annoncait l'inverse.

    Le contrat d'entree est une distribution NORMALISEE, pas des comptes.
    `execute` divise deja par le nombre de tirs dans ses trois branches. Le
    verifier ici n'est pas de la paranoia : des comptes bruts donneraient des
    « marginales » de l'ordre du millier, que toute comparaison a un seuil
    autour de 0.15 declarerait actives — un domaine entierement raffine,
    indiscernable d'une detection reelle.

    De meme, une chaine contenant un espace (plusieurs registres classiques)
    decalerait toutes les positions apres l'espace, et la carte de decision
    reviendrait spatialement decalee sans que rien ne le signale.
    """
    if not distribution:
        raise ValueError("distribution vide : aucune marginale a extraire")

    total = float(sum(distribution.values()))
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"postprocess attend une distribution normalisee, somme recue "
            f"{total:.6g}. Des comptes bruts produiraient des marginales "
            f"hors de [0, 1] que le seuil de raffinement declarerait toutes "
            f"actives."
        )

    hits = np.zeros(num_qubits)

    for bitstring, count in distribution.items():
        key = str(bitstring)
        if " " in key:
            raise ValueError(
                f"chaine multi-registres {key!r} : l'espace decalerait toutes "
                "les positions de qubit qui le suivent"
            )
        if len(key) != num_qubits:
            raise ValueError(
                f"chaine {key!r} de longueur {len(key)} pour {num_qubits} "
                "qubits : les marginales seraient lues au mauvais indice"
            )
        for i, bit in enumerate(key[::-1]):
            if bit == '1':
                hits[i] += count

    marginals = hits.tolist()

    if verbose:
        print(f"Marginals: {marginals}")

    return marginals
