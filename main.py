# ==============================================================================
# QISKIT FALL FEST 2026 - TRACK 3: QUANTUM ERROR CORRECTION
# Complete Working Prototype & Classical Baseline
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Qiskit core imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

# ------------------------------------------------------------------------------
# 1. SETUP SIMULATOR & NOISE MODEL GENERATOR
# ------------------------------------------------------------------------------
def create_bitflip_noise_model(p_error):
    """Creates a Qiskit-Aer noise model with single-qubit bit-flip errors."""
    noise_model = NoiseModel()
    bit_flip = pauli_error([('X', p_error), ('I', 1 - p_error)])
    # Apply noise to single-qubit gates and measurement operations
    noise_model.add_all_qubit_quantum_error(bit_flip, ['x', 'h', 'id', 'measure'])
    return noise_model

# ------------------------------------------------------------------------------
# 2. UNPROTECTED CIRCUIT (1 Qubit Baseline)
# ------------------------------------------------------------------------------
def build_unprotected_circuit():
    """Builds a basic 1-qubit circuit initialized to state |1>."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)  # State preparation: |1>
    qc.measure(0, 0)
    return qc

# ------------------------------------------------------------------------------
# 3. PROTECTED CIRCUIT (3-Qubit Bit-Flip Repetition Code)
# ------------------------------------------------------------------------------
def build_protected_circuit():
    """Builds a 3-qubit bit-flip code circuit with syndrome detection & correction."""
    q = QuantumRegister(3, name="q")
    c = ClassicalRegister(1, name="c")
    qc = QuantumCircuit(q, c)
    
    # --- State Preparation: Set Logical Qubit to |1> ---
    qc.x(q[0])
    
    # --- Step 1: Encoding (|1> -> |111>) ---
    qc.cx(q[0], q[1])
    qc.cx(q[0], q[2])
    qc.barrier()
    
    # --- Step 2: Majority Logic / Syndrome Decoding & Correction ---
    # Decoding circuit (inverts CNOT encoding)
    qc.cx(q[0], q[1])
    qc.cx(q[0], q[2])
    
    # Toffoli (CCX) gate performs majority decision to correct q[0] if error occurred
    qc.ccx(q[1], q[2], q[0])
    qc.barrier()
    
    # --- Step 3: Readout Logical Qubit ---
    qc.measure(q[0], c[0])
    return qc

# ------------------------------------------------------------------------------
# 4. CLASSICAL BASELINE (Repetition Code / Majority Vote Benchmark)
# ------------------------------------------------------------------------------
def classical_repetition_code(p_error, n_trials=10000):
    """Simulates a classical 3-bit majority voting channel with bit-flip probability p."""
    # Transmit bit '1' over 3 noisy classical channels
    bits = np.ones((n_trials, 3), dtype=int)
    flips = np.random.rand(n_trials, 3) < p_error
    received_bits = np.bitwise_xor(bits, flips.astype(int))
    
    # Majority vote
    decoded = (np.sum(received_bits, axis=1) >= 2).astype(int)
    fidelity = np.mean(decoded == 1)
    return fidelity

# ------------------------------------------------------------------------------
# 5. EXPERIMENTAL EXECUTION & BENCHMARKING
# ------------------------------------------------------------------------------
def run_benchmarks():
    noise_rates = np.linspace(0.0, 0.3, 10)
    shots = 2048
    
    unprotected_fidelities = []
    protected_fidelities = []
    classical_fidelities = []

    backend = AerSimulator()
    unprotected_qc = build_unprotected_circuit()
    protected_qc = build_protected_circuit()

    print("Running quantum & classical simulations across noise levels...")

    for p in noise_rates:
        noise_model = create_bitflip_noise_model(p)
        
        # --- Unprotected Execution ---
        compiled_unprotected = transpile(unprotected_qc, backend)
        job_unp = backend.run(compiled_unprotected, noise_model=noise_model, shots=shots)
        counts_unp = job_unp.result().get_counts()
        fid_unp = counts_unp.get('1', 0) / shots
        unprotected_fidelities.append(fid_unp)
        
        # --- Protected Execution ---
        compiled_protected = transpile(protected_qc, backend)
        job_prot = backend.run(compiled_protected, noise_model=noise_model, shots=shots)
        counts_prot = job_prot.result().get_counts()
        fid_prot = counts_prot.get('1', 0) / shots
        protected_fidelities.append(fid_prot)
        
        # --- Classical Baseline Execution ---
        fid_class = classical_repetition_code(p, n_trials=shots)
        classical_fidelities.append(fid_class)

    # --------------------------------------------------------------------------
    # 6. VISUALIZATION & PLOTTING RESULTS
    # --------------------------------------------------------------------------
    plt.figure(figsize=(9, 5.5))
    plt.plot(noise_rates, unprotected_fidelities, 'r--o', label='Unprotected Quantum (1 Qubit)', linewidth=2)
    plt.plot(noise_rates, protected_fidelities, 'g-s', label='Protected Quantum (3-Qubit QEC)', linewidth=2)
    plt.plot(noise_rates, classical_fidelities, 'b:.', label='Classical Majority Baseline', linewidth=1.5)
    
    plt.title("Track 3 Benchmark: Quantum Error Correction vs. Unprotected & Classical Baseline", fontsize=11, fontweight='bold')
    plt.xlabel("Physical Bit-Flip Error Probability ($p$)", fontsize=10)
    plt.ylabel("Output State Fidelity ($P(\\text{Logical State} = 1)$)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    plt.savefig('qec_fidelity_comparison.png', dpi=300)
    plt.show()

    print("Benchmarking complete! Plot saved as 'qec_fidelity_comparison.png'.")

if __name__ == "__main__":
    run_benchmarks()
