# scripts/mapping.py

import numpy as np

from qiskit.circuit.library import QAOAAnsatz

from VQA.init_qbits_state import init_qbits_state
from VQA.cost_hamiltonian import create_bounded_hamiltonian, create_period_hamiltonian


def mapping(data_in, hamilt_params, period_bound=True, reps=2):
    """
    Build the QAOA circuit from encoded angles and Hamiltonian coefficients.

    data_in contains:
        theta_h, theta_v : classical score → θ = 2·arcsin(√score)
        psi_h, psi_v     : stress flux temporal evolution → phase angle

    For bounded patches (depth > 0), theta/psi arrays include a 1-pixel halo
    used for boundary conditions in the Hamiltonian.
    """

    # Halo: periodic (torus) has no halo, bounded has 2 extra pixels (1 per side)
    halo_dim = 0 if period_bound else 2
    dim = len(data_in.get("theta_h", [])) - halo_dim

    theta_h = np.array(data_in.get("theta_h", []))
    theta_v = np.array(data_in.get("theta_v", []))
    psi_h   = np.array(data_in.get("psi_h",   []))
    psi_v   = np.array(data_in.get("psi_v",   []))

    if period_bound:
        cost_hamiltonian = create_period_hamiltonian(hamilt_params, dim)
        init_th, init_tv = theta_h, theta_v
        init_ph, init_pv = psi_h, psi_v
    else:
        # Bounded: Hamiltonian uses halo θ for boundary <Z> values
        cost_hamiltonian, core_theta_h, core_theta_v, core_psi_h, core_psi_v = create_bounded_hamiltonian(
            hamilt_params, dim, theta_h, theta_v, psi_h, psi_v
        )
        # Core only for qubit initialization (strip halo)
        init_th, init_tv = core_theta_h, core_theta_v
        init_ph, init_pv = core_psi_h, core_psi_v

    qc = init_qbits_state(init_th, init_tv, init_ph, init_pv)

    ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps, initial_state=qc)
    qc = ansatz.decompose().decompose()

    return qc, cost_hamiltonian
