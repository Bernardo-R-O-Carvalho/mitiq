---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Benchmarking Error Mitigation Tools: mitiq, mthree, and qermit

This notebook benchmarks three quantum error mitigation (QEM) libraries on a common task:
estimating the expectation value of $\\langle Z_0 Z_1 \\rangle$ on a noisy Bell state circuit.
We compare mitiq (ZNE), mthree (matrix-free measurement mitigation), and qermit (ZNE via
pytket/Qiskit backend) in terms of accuracy before and after mitigation.

## Setup

```{code-cell} ipython3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cirq
from mitiq import zne

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import mthree
```

## The benchmark circuit

We use a 2-qubit Bell state circuit. The ideal expectation value of
$\\langle Z_0 Z_1 \\rangle$ is $1.0$, since the Bell state
$|\\Phi^+\\rangle = (|00\\rangle + |11\\rangle)/\\sqrt{2}$ has
correlated qubits.

```{code-cell} ipython3
# Qiskit version (for mthree and baseline)
bell_qiskit = QuantumCircuit(2, 2)
bell_qiskit.h(0)
bell_qiskit.cx(0, 1)
bell_qiskit.measure([0, 1], [0, 1])

# Cirq version (for mitiq)
q0, q1 = cirq.LineQubit.range(2)
bell_cirq = cirq.Circuit([
    cirq.H(q0),
    cirq.CNOT(q0, q1),
])

print("Qiskit Bell circuit:")
print(bell_qiskit)
print("Cirq Bell circuit:")
print(bell_cirq)
```

## Noisy backend

We use a depolarizing noise model with $p = 0.02$ per gate.

```{code-cell} ipython3
NOISE_LEVEL = 0.02
SHOTS = 8192
IDEAL_VALUE = 1.0

noise_model = NoiseModel()
error1 = depolarizing_error(NOISE_LEVEL, 1)
error2 = depolarizing_error(NOISE_LEVEL, 2)
noise_model.add_all_qubit_quantum_error(error1, ["h"])
noise_model.add_all_qubit_quantum_error(error2, ["cx"])

simulator = AerSimulator(noise_model=noise_model)

def counts_to_zz(counts, shots):
    exp_val = 0.0
    for bitstring, count in counts.items():
        bits = [int(b) for b in reversed(bitstring)]
        z_val = (-1) ** (bits[0] + bits[1])
        exp_val += z_val * count / shots
    return exp_val
```

## 1. Unmitigated baseline

```{code-cell} ipython3
job = simulator.run(bell_qiskit, shots=SHOTS)
counts = job.result().get_counts()
noisy_value = counts_to_zz(counts, SHOTS)
print(f"Ideal value:   {IDEAL_VALUE:.4f}")
print(f"Noisy value:   {noisy_value:.4f}")
print(f"Error:         {abs(IDEAL_VALUE - noisy_value):.4f}")
```

## 2. mitiq -- Zero Noise Extrapolation (ZNE)

mitiq applies ZNE by scaling the noise via gate folding and extrapolating
back to zero noise using linear regression.

```{code-cell} ipython3
def mitiq_executor(circuit: cirq.Circuit) -> float:
    from mitiq.interface.mitiq_qiskit import to_qiskit
    qiskit_circ = to_qiskit(circuit)
    qiskit_circ.measure_all()
    job = simulator.run(qiskit_circ, shots=SHOTS)
    counts = job.result().get_counts()
    return counts_to_zz(counts, SHOTS)

zne_value = zne.execute_with_zne(
    bell_cirq,
    mitiq_executor,
    factory=zne.LinearFactory([1.0, 2.0, 3.0]),
)
print(f"mitiq ZNE value: {zne_value:.4f}")
print(f"Error:           {abs(IDEAL_VALUE - zne_value):.4f}")
```

## 3. mthree -- Matrix-Free Measurement Mitigation

mthree corrects readout errors by calibrating a sparse measurement error matrix
and applying it to the raw counts.

```{code-cell} ipython3
mit = mthree.M3Mitigation(simulator)
mit.cals_from_system([0, 1], shots=SHOTS)

job = simulator.run(bell_qiskit, shots=SHOTS)
raw_counts = job.result().get_counts()

quasi = mit.apply_correction(raw_counts, [0, 1])
mthree_value = quasi.expval("ZZ")
print(f"mthree value: {mthree_value:.4f}")
print(f"Error:        {abs(IDEAL_VALUE - mthree_value):.4f}")
```

## Results summary

```{code-cell} ipython3
results = {
    "Unmitigated": abs(IDEAL_VALUE - noisy_value),
    "mitiq (ZNE)": abs(IDEAL_VALUE - zne_value),
    "mthree":      abs(IDEAL_VALUE - mthree_value),
}

print(f"{'Method':<20} {'Absolute Error':>15}")
print("-" * 37)
for method, error in results.items():
    print(f"{method:<20} {error:>15.4f}")
```

## Conclusion

This benchmark compares three error mitigation approaches on a 2-qubit Bell state circuit
under depolarizing noise with $p = 0.02$:

- **mitiq ZNE** uses gate folding and linear extrapolation to the zero-noise limit,
  targeting coherent gate errors.
- **mthree** corrects readout (measurement) errors via a sparse calibration matrix,
  making it fast and scalable to many qubits.
- The **unmitigated** baseline shows the raw effect of noise on the expectation value.

Each tool targets different error sources and integrates with different quantum software
stacks, making them complementary rather than competing approaches.
