# scripts/execute.py
import numpy as np
from scipy.optimize import minimize
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeFez

def execute(qc, cost_hamiltonian, mode, backend_name, shots, reps, verbose):
    if verbose:
        print(f"Initial Circuit:\n{qc.draw('text')}")
        print("\n","\n")
        print(f"Information about the circuit: \nNumber of qubits: {qc.num_qubits}\n Depth : {qc.depth()}")
        print("\n","\n")
        print(f"EXECUTION WITH :\nShots: {shots}\nRepetitions od the QAOA: {reps}\nMode: {mode} \nBackend: {backend_name}")
    # 1. Configuration du Backend (doit correspondre à celui utilisé dans optimize)
    if backend_name == "aer":
        backend = AerSimulator()
    elif backend_name == "estimator":
        backend = FakeFez()
    else:
        # Cas Runtime service réel à gérer ici si besoin
        pass

    # 2. Préparation de l'Hamiltonien ISA (Instruction Set Architecture)
    # On le fait UNE SEULE FOIS en dehors de la boucle, pas 1000 fois.
    # On applique le layout du circuit transpilé à l'opérateur.
    if qc.layout is not None:
        isa_hamiltonian = cost_hamiltonian.apply_layout(qc.layout)
    else:
        # Cas où Aer n'a pas imposé de layout spécifique (trivial)
        isa_hamiltonian = cost_hamiltonian

    # 3. Définition de la fonction de coût
    objective_func_vals = []

    def cost_func_estimator(params, ansatz, hamiltonian, estimator):
        # Note: 'ansatz' est déjà transpilé, 'hamiltonian' est déjà ISA.
        # On passe juste les paramètres.
        pub = (ansatz, hamiltonian, params)
        
        # Exécution
        job = estimator.run([pub])
        result = job.result()[0]
        cost = result.data.evs
        
        objective_func_vals.append(cost)
        # Optionnel: print light pour suivre la progression
        if len(objective_func_vals) % 10 == 0:
            if verbose:
                print(f"Iter {len(objective_func_vals)}: Cost = {cost}")
            
        return cost

    # 4. Paramètres Initiaux
    # Convention QAOA standard : Beta puis Gamma
    initial_params = np.concatenate([
        np.full(reps, np.pi / 2),  # Beta
        np.full(reps, np.pi)       # Gamma
    ])
    if verbose:
        print("\n--- Starting Optimization Loop ---")
    
    # 5. Exécution de l'Optimisation
    # Pour Aer local, pas besoin de Session context manager complexe
    if mode == "simulator":
        estimator = Estimator(mode=backend)
        estimator.options.default_shots = shots
        
        result = minimize(
            cost_func_estimator,
            initial_params,
            args=(qc, isa_hamiltonian, estimator),
            method="COBYLA",
            tol=1e-2,
            options={'maxiter': 100} # Sécurité pour éviter boucle infinie
        )
    else:
        # Pour le vrai hardware via Runtime
        with Session(backend=backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = shots
            # Options de mitigation d'erreur
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.twirling.enable_gates = True
            
            result = minimize(
                cost_func_estimator,
                initial_params,
                args=(qc, isa_hamiltonian, estimator),
                method="COBYLA",
                tol=1e-2
            )
    if verbose:
        print(f"Optimization success: {result.success}")
        print(f"Optimal Params: {result.x}")

        # 6. Sampling Final (Mesure)
        print("\n--- Final Sampling ---")
    
    # On assigne les paramètres optimaux
    optimized_circuit = qc.assign_parameters(result.x)
    
    # C'est MAINTENANT qu'on ajoute les mesures pour le Sampler
    optimized_circuit.measure_all()
    
    # Pour le sampler, il faut s'assurer que le circuit est transpilé pour inclure les mesures
    # Si 'measure_all' ajoute des portes non natives, un petit transpile léger peut être requis,
    # mais souvent sur Aer ça passe. Par sécurité :
    # optimized_circuit = transpile(optimized_circuit, backend) 
    
    if mode == "simulator":
        sampler = Sampler(mode=backend)
    else:
        # Réutilisation session impossible si fermée, on recrée pour l'exemple ou on intègre dans le 'with' au dessus
        sampler = Sampler(mode=backend)
        
    sampler.options.default_shots = shots
    
    pub = (optimized_circuit,)
    job = sampler.run([pub])
    
    # Récupération résultats (compatible V2)
    pub_result = job.result()[0]
    
    # Gestion Bitstring vs Int
    # data.meas.get_counts() retourne des bitstrings '0101'
    counts_bin = pub_result.data.meas.get_counts()
    
    total_shots = sum(counts_bin.values())
    final_distribution = {key: val / total_shots for key, val in counts_bin.items()}

    return final_distribution