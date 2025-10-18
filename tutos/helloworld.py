# ============================================
# Quantum Bell State Simulation (Local Version)
# ============================================

# Step 1. Imports
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeFez
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from matplotlib import pyplot as plt

# --------------------------------------------
# Step 1. Map the problem to a quantum-native format
# --------------------------------------------

# Create a new circuit with two qubits
qc = QuantumCircuit(2)

# Add a Hadamard gate to qubit 0
qc.h(0)

# Perform a controlled-X gate on qubit 1, controlled by qubit 0
qc.cx(0, 1)

# Draw the circuit
qc.draw("mpl")
plt.show()

# Define six two-qubit observables
observables_labels = ["IZ", "IX", "ZI", "XI", "ZZ", "XX"]
observables = [SparsePauliOp(label) for label in observables_labels]

# --------------------------------------------
# Step 2. Optimize the circuit for a local simulator
# --------------------------------------------

# Use a fake (simulated) IBM backend
backend = FakeFez()

# Optimize the circuit for the fake backend
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)

# Draw optimized circuit
isa_circuit.draw("mpl", idle_wires=False)
plt.show()

# --------------------------------------------
# Step 3. Execute using the quantum primitives (Estimator)
# --------------------------------------------

# Construct an estimator with the fake backend
estimator = Estimator(backend)

# Apply layout to observables (maps logical to physical qubits)
mapped_observables = [
    observable.apply_layout(isa_circuit.layout) for observable in observables
]

# Run the job: one circuit, multiple observables
job = estimator.run([(isa_circuit, mapped_observables)])

# Wait for the result
result = job.result()

# Extract pub_result (the detailed observable data)
pub_result = result[0]

# --------------------------------------------
# Step 4. Analyze and plot results
# --------------------------------------------

# Get expectation values and standard deviations
values = pub_result.data.evs
errors = pub_result.data.stds

# Plot the results
plt.figure(figsize=(8, 4))
plt.plot(observables_labels, values, "-o")
plt.xlabel("Observables")
plt.ylabel("Expectation Values")
plt.title("Expectation values of observables for Bell State (Simulated)")
plt.grid(True)
plt.show()
