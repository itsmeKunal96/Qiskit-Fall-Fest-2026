import os
import sys
import numpy as np
import warnings

# Suppress harmless UserWarnings from Qiskit & Matplotlib
warnings.filterwarnings("ignore", category=UserWarning)

# Enable interactive pop-up windows
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
    pauli_error,
    coherent_unitary_error
)

# ==========================================
# 0. GLOBALS & CONFIGURATION
# ==========================================
BASIS = ["id", "rz", "sx", "x", "cx"]
TH, PH = 1.2, 0.7  # Target state: RY(1.2) RZ(0.7) |0>

# ==========================================
# 1. NOISE MODEL GENERATOR
# ==========================================
def build_noise(p1=1e-3, p2=1e-2, p_ro=1e-2, t1=100e3, t2=80e3, 
                t_1q=50, t_2q=300, t_idle=500, idle_x_flip=0.0, detune=0.0):
    """
    Builds a Qiskit Aer NoiseModel with numerical precision checks 
    and explicit idle noise channels across all operational gates.
    """
    nm = NoiseModel(basis_gates=BASIS)
    
    # Bound T2 to physical limit T2 <= 2*T1
    t2_valid = min(t2, 2 * t1)
    th_err = lambda t: thermal_relaxation_error(t1, t2_valid, t)
    e1 = lambda t: depolarizing_error(p1, 1).compose(th_err(t))
    
    # Single-qubit & Two-qubit Gate Errors
    nm.add_all_qubit_quantum_error(e1(t_1q), ["sx", "x"])
    nm.add_all_qubit_quantum_error(
        depolarizing_error(p2, 2).compose(th_err(t_2q).tensor(th_err(t_2q))), ["cx"]
    )
    
    # Storage / Idle Noise Channel
    idle = th_err(t_idle)
    if idle_x_flip > 0:
        idle = idle.compose(pauli_error([("X", idle_x_flip), ("I", 1 - idle_x_flip)]))
        
    # Numerically Validated Coherent Phase Rotation Matrix
    if detune != 0.0:
        phase = detune / 2.0
        rz_mat = np.array([
            [np.exp(-1j * phase), 0.0],
            [0.0, np.exp(1j * phase)]
        ], dtype=complex)
        if np.allclose(rz_mat.conj().T @ rz_mat, np.eye(2)):
            idle = idle.compose(coherent_unitary_error(rz_mat))
            
    nm.add_all_qubit_quantum_error(idle, ["id"])
    
    # Readout Errors
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    return nm

# ==========================================
# 2. CIRCUIT BUILDERS
# ==========================================
def prep(qc, q):
    qc.ry(TH, q)
    qc.rz(PH, q)

def unprep(qc, q):
    qc.rz(-PH, q)
    qc.ry(-TH, q)

def build_none(n_idle):
    """Unprotected Logical Qubit using identity operations."""
    qc = QuantumCircuit(1, 1)
    prep(qc, 0)
    for _ in range(n_idle):
        qc.id(0)
    unprep(qc, 0)
    qc.measure(0, 0)
    return qc

def build_dd(n_idle):
    """Dynamical Decoupling (X-echo sequence)."""
    assert n_idle % 2 == 0, "n_idle must be even for DD sequence"
    qc = QuantumCircuit(1, 1)
    prep(qc, 0)
    for _ in range(n_idle):
        qc.id(0)
        qc.x(0)
        qc.id(0)
        qc.x(0)
    unprep(qc, 0)
    qc.measure(0, 0)
    return qc

def build_code(n_idle, kind="bitflip"):
    """
    3-qubit Error Correcting Code (Bit-flip or Phase-flip) with 
    explicit register indexing and ancilla resets.
    """
    d = QuantumRegister(3, "d")
    a = QuantumRegister(2, "a")
    syn = ClassicalRegister(2, "syn")
    out = ClassicalRegister(1, "out")
    qc = QuantumCircuit(d, a, syn, out)
    
    # 1. Encoding
    prep(qc, d[0])
    qc.cx(d[0], d[1])
    qc.cx(d[0], d[2])
    qc.barrier()
    
    if kind == "phaseflip":
        qc.h(d)
        qc.barrier()
        
    # 2. Storage Window (Identity operations)
    for _ in range(n_idle):
        for q in d:
            qc.id(q)
        qc.barrier()
        
    if kind == "phaseflip":
        qc.h(d)
        qc.barrier()
        
    # 3. Syndrome Extraction
    qc.cx(d[0], a[0])
    qc.cx(d[1], a[0])
    qc.cx(d[1], a[1])
    qc.cx(d[2], a[1])
    qc.measure(a[0], syn[0])
    qc.measure(a[1], syn[1])
    
    # Syndrome Mapping
    with qc.if_test((syn, 1)):  # syn = 01
        qc.x(d[0])
    with qc.if_test((syn, 3)):  # syn = 11
        qc.x(d[1])
    with qc.if_test((syn, 2)):  # syn = 10
        qc.x(d[2])
        
    qc.reset(a)  # Reset ancilla qubits for reuse
    qc.barrier()
    
    # 4. Decode & Unprep
    qc.cx(d[0], d[1])
    qc.cx(d[0], d[2])
    unprep(qc, d[0])
    qc.measure(d[0], out[0])
    return qc

# ==========================================
# 3. SIMULATION ENGINE
# ==========================================
def compute_fidelity(qc, nm, shots=2000, seed=42):
    """Computes logical state fidelity via transpiled simulation."""
    sim = AerSimulator(noise_model=nm, seed_simulator=seed)
    tqc = transpile(qc, sim, optimization_level=0)
    counts = sim.run(tqc, shots=shots).result().get_counts()
    
    correct = sum(v for k, v in counts.items() if k.split()[0] == "0")
    return correct / shots

# ==========================================
# 4. ADAPTIVE NOISE POLICY
# ==========================================
def adaptive_policy(p_cx, p_flip, p_phase, n_idle, detune):
    """Evaluates joint physical error rates to route execution."""
    effective_qec_overhead_error = 10 * p_cx + 3 * p_flip
    unprotected_storage_error = n_idle * p_flip
    
    if detune > 0.05 and p_flip < 0.01 and p_cx < 0.01:
        return "dynamical_decoupling"
    
    if unprotected_storage_error > effective_qec_overhead_error:
        if p_flip >= p_phase:
            return "3_qubit_bitflip"
        else:
            return "3_qubit_phaseflip"
            
    return "none"

# ==========================================
# 5. VISUALIZATION FUNCTIONS (SAVED & POP-UP)
# ==========================================
def plot_1d_noise_sweep():
    """Generates, saves, and displays 1D Fidelity vs Storage Error Rate plot."""
    print("Running Fast 1D Noise Sweep...", flush=True)
    p_idles = np.linspace(0.001, 0.20, 8)
    n_idle = 4
    p_cx_fixed = 0.002
    shots_fast = 1500
    
    qc_none = build_none(n_idle)
    qc_dd = build_dd(n_idle)
    qc_qec = build_code(n_idle, "bitflip")
    
    f_none, f_dd, f_qec = [], [], []
    
    for p_id in p_idles:
        nm = build_noise(p1=p_cx_fixed/10, p2=p_cx_fixed, p_ro=p_cx_fixed, 
                         t1=1e9, t2=1e9, idle_x_flip=p_id)
        
        sim = AerSimulator(noise_model=nm, seed_simulator=42)
        
        tqc_none = transpile(qc_none, sim, optimization_level=0)
        tqc_dd = transpile(qc_dd, sim, optimization_level=0)
        tqc_qec = transpile(qc_qec, sim, optimization_level=0)
        
        c_none = sim.run(tqc_none, shots=shots_fast).result().get_counts()
        c_dd = sim.run(tqc_dd, shots=shots_fast).result().get_counts()
        c_qec = sim.run(tqc_qec, shots=shots_fast).result().get_counts()
        
        f_none.append(sum(v for k, v in c_none.items() if k.split()[0] == "0") / shots_fast)
        f_dd.append(sum(v for k, v in c_dd.items() if k.split()[0] == "0") / shots_fast)
        f_qec.append(sum(v for k, v in c_qec.items() if k.split()[0] == "0") / shots_fast)
        
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_idles, f_none, 'o--', label='Unprotected', color='tab:red')
    ax.plot(p_idles, f_dd, 's-.', label='Dynamical Decoupling', color='tab:orange')
    ax.plot(p_idles, f_qec, '^-', label='3-Qubit Bit-Flip QEC', color='tab:green')
    
    ax.set_title('Logical Qubit Fidelity vs. Storage Noise (p_idle)')
    ax.set_xlabel('Storage Error Probability (p_idle)')
    ax.set_ylabel('Fidelity')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    
    fig.savefig('qec_fidelity_sweep.png', dpi=300)
    plt.show()  # Opens plot window on screen
    plt.close(fig)

def plot_2d_crossover_heatmap():
    """Generates, saves, and displays 2D Crossover Map."""
    print("Running 2D Crossover Sweep...", flush=True)
    p_idles = [0.002, 0.01, 0.03, 0.08, 0.15]
    p_cxs = [0.0005, 0.002, 0.008, 0.02, 0.05]
    n_idle = 4
    
    diff_matrix = np.zeros((len(p_cxs), len(p_idles)))
    
    for i, p2 in enumerate(p_cxs):
        for j, p_id in enumerate(p_idles):
            nm = build_noise(p1=p2/10, p2=p2, p_ro=p2, t1=1e9, t2=1e9, idle_x_flip=p_id)
            f_unprot = compute_fidelity(build_none(n_idle), nm, shots=1000)
            f_qec = compute_fidelity(build_code(n_idle, "bitflip"), nm, shots=1000)
            diff_matrix[i, j] = f_qec - f_unprot
            
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(diff_matrix, origin='lower', cmap='RdYlGn', aspect='auto')
    fig.colorbar(im, ax=ax, label='Fidelity Gain (F_QEC - F_Unprotected)')
    
    ax.set_xticks(range(len(p_idles)))
    ax.set_xticklabels([f"{p:.3f}" for p in p_idles])
    ax.set_yticks(range(len(p_cxs)))
    ax.set_yticklabels([f"{p:.4f}" for p in p_cxs])
    ax.set_xlabel('Storage Noise (p_idle)')
    ax.set_ylabel('CNOT Gate Error (p_cx)')
    ax.set_title('2D Pseudo-Threshold Crossover Boundary\n(Green = QEC Helps | Red = QEC Hurts)')
    
    fig.tight_layout()
    fig.savefig('crossover_heatmap.png', dpi=300)
    plt.show()  # Opens plot window on screen
    plt.close(fig)

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=== Track 3: Quantum Error Correction Final Execution ===", flush=True)
    
    # 1. Verification Test (Infinite T1/T2)
    nm_clean = build_noise(p1=0, p2=0, p_ro=0, idle_x_flip=0, t1=1e9, t2=1e9)
    f_clean = compute_fidelity(build_code(4, "bitflip"), nm_clean)
    print(f"Sanity Check - Noiseless QEC Fidelity: {f_clean:.4f} (Expected: 1.0000)", flush=True)
    
    # 2. Noisy Sample Test
    nm_noisy = build_noise(p1=1e-3, p2=1e-2, p_ro=1e-2, idle_x_flip=0.03, t1=1e9, t2=1e9)
    f_unprot = compute_fidelity(build_none(4), nm_noisy)
    f_prot = compute_fidelity(build_code(4, "bitflip"), nm_noisy)
    print(f"Sample Noise Test -> Unprotected Fidelity: {f_unprot:.4f} | Protected Fidelity: {f_prot:.4f}", flush=True)
    
    # 3. Adaptive Decision Test
    decision = adaptive_policy(p_cx=0.005, p_flip=0.03, p_phase=0.001, n_idle=6, detune=0.0)
    print(f"Adaptive Controller Decision (p_flip=0.03, p_cx=0.005): {decision}", flush=True)
        
    # 4. Visualizations
    plot_1d_noise_sweep()
    plot_2d_crossover_heatmap()
    
