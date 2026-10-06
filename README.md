# Track 3: Quantum Error Correction — Building a More Reliable Quantum Circuit

## Executive Summary
This project evaluates the performance of a 3-Qubit Bit-Flip Quantum Error Correction (QEC) scheme against circuit instability factors identified from quantum hardware stability datasets.

## Key Findings
- **Instability Drivers:** Dataset correlation analysis shows that circuit depth and 2-qubit (CX) gate errors dominate logical fidelity loss.
- **Pseudo-Threshold:** Under low-to-moderate physical error rates ($p < 0.05$), the 3-qubit QEC circuit maintains higher fidelity than an unprotected physical qubit.
- **Overhead Cost:** The 3-qubit QEC code requires 3x physical qubits and increased gate depth, making protection advantageous only below the pseudo-threshold error rate.

## Structure & Deliverables
- `main.py`: Complete execution pipeline covering Tasks 1–8.
- `task2_feature_importance.png`: Feature correlation plot derived from dataset analysis.
- `task7_pseudo_threshold_loglog.png`: Log-Log error comparison chart with Standard Error bars.
- `task8_tradeoff_table.csv`: Resource accounting table extracted from transpiled Qiskit circuits.

## How to Run
```bash
pip install -r requirements.txt
python main.py
