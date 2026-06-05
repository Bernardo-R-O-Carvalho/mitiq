---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{tags} zne, cirq, error-detection, intermediate
```

# Combining Error Detection and Zero Noise Extrapolation with the [[4,2,2]] Code

Error detection and error mitigation are complementary strategies for dealing with noise on near-term quantum
hardware. In this tutorial we demonstrate how to combine them using Mitiq: specifically, we use the `[[4,2,2]]`
quantum error-detecting code to discard shots where an error is known to have occurred, and then apply
[Zero Noise Extrapolation (ZNE)](../guide/zne.md) to the surviving shots for further mitigation.

The key insight is that these two approaches attack noise from different angles.
Error detection (via post-selection) removes shots that are provably corrupted.
ZNE then reduces the residual noise in the shots that survived post-selection.
Together they achieve better results than either technique alone.

+++

## Background: the [[4,2,2]] code

The `[[4,2,2]]` code is the smallest quantum error-*detecting* code.
It encodes 2 logical qubits into 4 physical qubits and can detect (but not correct) any single-qubit error.
The notation `[[n, k, d]]` refers to an `n`-physical-qubit code encoding `k` logical qubits with distance `d`.

The four logical codewords are:

$$
\begin{align}
|0_L 0_L\rangle &= \frac{1}{\sqrt{2}}(|0000\rangle + |1111\rangle) \\
|0_L 1_L\rangle &= \frac{1}{\sqrt{2}}(|0011\rangle + |1100\rangle) \\
|1_L 0_L\rangle &= \frac{1}{\sqrt{2}}(|0101\rangle + |1010\rangle) \\
|1_L 1_L\rangle &= \frac{1}{\sqrt{2}}(|0110\rangle + |1001\rangle)
\end{align}
$$

The code has two stabilizer generators: `XXXX` and `ZZZZ`.
Any single-qubit error anticommutes with at least one of these stabilizers, flipping its measurement outcome
from $+1$ to $-1$ and revealing that an error occurred.
Shots where either stabilizer measures $-1$ are discarded — this is *error detection*, not correction.

:::{note}
Discarding shots with detected errors reduces the number of usable shots. This is a real overhead cost: at
high noise rates, a significant fraction of shots may be discarded. The benefit is that the surviving shots
have provably lower error rates.
:::

+++

## Setup

```{code-cell} ipython3
import cirq
import numpy as np
import matplotlib.pyplot as plt
from mitiq import zne
```

+++

## Encoding circuit for $|0_L 0_L\rangle$

We prepare the logical state $|0_L 0_L\rangle = \frac{1}{\sqrt{2}}(|0000\rangle + |1111\rangle)$
using a Hadamard on the first qubit followed by three CNOT gates targeting the remaining data qubits.

```{code-cell} ipython3
def make_encoding_circuit() -> cirq.Circuit:
    """Returns the [[4,2,2]] encoding circuit for the |0_L 0_L> logical state.

    The logical state is (|0000> + |1111>) / sqrt(2), prepared with a
    Hadamard on q0 followed by CNOT gates from q0 to q1, q2, q3.
    """
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    return cirq.Circuit([
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.CNOT(q0, q2),
        cirq.CNOT(q0, q3),
    ])

print(make_encoding_circuit())
```

+++

## Syndrome measurement

To detect errors we measure the two stabilizers `XXXX` and `ZZZZ` using two ancilla qubits.
For the `XXXX` stabilizer, the ancilla is prepared in $|+\rangle$ via Hadamard, entangled with all four
data qubits via CNOTs, and then rotated back before measurement.
For `ZZZZ`, the ancilla collects parity information via CNOTs directly.
A measurement outcome of $1$ on either ancilla indicates the corresponding stabilizer measured $-1$,
signaling that an error was detected.

```{code-cell} ipython3
def make_full_circuit() -> cirq.Circuit:
    """Returns the [[4,2,2]] encoding circuit with syndrome measurement appended.

    The circuit encodes |0_L 0_L>, then measures the XXXX and ZZZZ stabilizers
    using two ancilla qubits. Data qubits are measured last.

    Qubit layout: q0-q3 are data qubits, q4 (anc_x) and q5 (anc_z) are ancillas.

    Returns:
        A cirq.Circuit with measurements keyed 'syndrome' (2 bits) and 'data' (4 bits).
    """
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    anc_x = cirq.LineQubit(4)
    anc_z = cirq.LineQubit(5)

    circuit = make_encoding_circuit()

    # Measure XXXX stabilizer
    circuit += cirq.H(anc_x)
    circuit += [cirq.CNOT(anc_x, q) for q in [q0, q1, q2, q3]]
    circuit += cirq.H(anc_x)

    # Measure ZZZZ stabilizer
    circuit += [cirq.CNOT(q, anc_z) for q in [q0, q1, q2, q3]]

    # Measure syndrome ancillas then data qubits
    circuit += cirq.measure(anc_x, anc_z, key="syndrome")
    circuit += cirq.measure(q0, q1, q2, q3, key="data")

    return circuit

full_circuit = make_full_circuit()
print(full_circuit)
```

+++

## Noise model and shot collection

We use a local depolarizing noise model. A depolarizing probability of 2% per gate is a realistic
range for near-term hardware. The `run_circuit` function returns a bitstring array of shape
`(shots, 6)` where the last two columns are syndrome bits `[sx, sz]`.

```{code-cell} ipython3
def run_circuit(
    circuit: cirq.Circuit, noise_level: float = 0.02, reps: int = 4000
) -> np.ndarray:
    """Run a circuit with depolarizing noise and return bitstrings.

    Args:
        circuit: The circuit to simulate. Must have 'data' (4 bits) and
            'syndrome' (2 bits) measurement keys.
        noise_level: Depolarizing probability per gate.
        reps: Number of shots.

    Returns:
        Integer array of shape (reps, 6): columns [d0, d1, d2, d3, sx, sz].
    """
    noisy = circuit.with_noise(cirq.depolarize(noise_level))
    sim = cirq.DensityMatrixSimulator()
    result = sim.run(noisy, repetitions=reps)
    data = result.measurements["data"]
    syndrome = result.measurements["syndrome"]
    return np.concatenate([data, syndrome], axis=1).astype(int)
```

+++

## Observable and post-selection

The logical $\bar{Z}_1 \bar{Z}_2$ observable corresponds to the parity of the first two data qubits.
For the ideal state $|0_L 0_L\rangle$ its expectation value is $+1$.

Post-selection discards any shot where either syndrome bit is $1$ (error detected).

```{code-cell} ipython3
def zzii_expectation(bitstrings: np.ndarray) -> float:
    """Compute <ZZII> from bitstrings as the mean parity of qubits 0 and 1.

    Args:
        bitstrings: Integer array of shape (shots, n_bits). First two columns
            are used for the ZZ parity computation.

    Returns:
        Mean of (-1)^(b0 XOR b1) over all shots.
    """
    parities = (-1) ** (bitstrings[:, 0] ^ bitstrings[:, 1])
    return float(parities.mean())


def post_select_no_errors(bitstrings: np.ndarray) -> np.ndarray:
    """Discard shots where either syndrome bit is 1 (error detected).

    Args:
        bitstrings: Integer array of shape (shots, 6). Last two columns are
            syndrome bits [sx, sz].

    Returns:
        Subset of bitstrings where both syndrome bits are 0.
    """
    mask = (bitstrings[:, -2] == 0) & (bitstrings[:, -1] == 0)
    return bitstrings[mask]
```

+++

## Executors for ZNE

For ZNE, Mitiq scales noise by folding gates and calls the executor at multiple noise levels.
Our executor must accept a `cirq.Circuit` and return a scalar expectation value.

We define two executors: one that applies post-selection (error detection + ZNE) and one that does not
(ZNE only), so we can compare the contribution of each technique independently.

We use a `LinearFactory` with scale factors `[1.0, 2.0, 3.0]`, which fits a linear extrapolation to
zero noise. This avoids the overshoot that can occur with higher-order extrapolation when post-selection
reduces the effective number of shots at each noise level.

```{code-cell} ipython3
FACTORY = zne.inference.LinearFactory(scale_factors=[1.0, 2.0, 3.0])


def make_zne_executor(noise_level: float = 0.02, reps: int = 4000):
    """Returns a ZNE-compatible executor with post-selection applied.

    The executor runs the circuit, discards shots with detected errors,
    and returns the <ZZII> expectation value as a float.

    Args:
        noise_level: Base depolarizing probability per gate.
        reps: Number of shots per ZNE evaluation.

    Returns:
        A callable (cirq.Circuit) -> float suitable for zne.execute_with_zne.
    """
    def executor(circuit: cirq.Circuit) -> float:
        bitstrings = run_circuit(circuit, noise_level=noise_level, reps=reps)
        clean = post_select_no_errors(bitstrings)
        if len(clean) == 0:
            return 0.0
        return zzii_expectation(clean)

    return executor


def make_zne_only_executor(noise_level: float = 0.02, reps: int = 4000):
    """Returns a ZNE-compatible executor without post-selection.

    The executor runs the circuit on all shots (no error detection)
    and returns the <ZZII> expectation value as a float.

    Args:
        noise_level: Base depolarizing probability per gate.
        reps: Number of shots per ZNE evaluation.

    Returns:
        A callable (cirq.Circuit) -> float suitable for zne.execute_with_zne.
    """
    def executor(circuit: cirq.Circuit) -> float:
        bitstrings = run_circuit(circuit, noise_level=noise_level, reps=reps)
        return zzii_expectation(bitstrings)

    return executor
```

+++

## Comparing the five cases

We now compare:

1. **Ideal**: noiseless simulation (ground truth)
2. **Noisy**: noisy simulation, no mitigation
3. **Error detection only**: post-selection on syndrome bits, no ZNE
4. **ZNE only**: ZNE without post-selection
5. **Error detection + ZNE**: post-selection inside each ZNE evaluation

```{code-cell} ipython3
NOISE_LEVEL = 0.02
REPS = 4000

# 1. Ideal
bits_ideal = run_circuit(full_circuit, noise_level=0.0, reps=REPS)
ideal_value = zzii_expectation(bits_ideal)
print(f"Ideal:                {ideal_value:.4f}")

# 2. Noisy (unmitigated)
bits_noisy = run_circuit(full_circuit, noise_level=NOISE_LEVEL, reps=REPS)
noisy_value = zzii_expectation(bits_noisy)
print(f"Noisy:                {noisy_value:.4f}")

# 3. Error detection only
bits_ed = post_select_no_errors(bits_noisy)
ed_value = zzii_expectation(bits_ed)
shots_kept = len(bits_ed)
print(f"Error detection:      {ed_value:.4f}  ({shots_kept}/{REPS} shots kept)")

# 4. ZNE only (no post-selection)
zne_only_executor = make_zne_only_executor(noise_level=NOISE_LEVEL, reps=REPS)
zne_only_value = zne.execute_with_zne(
    full_circuit,
    zne_only_executor,
    scale_noise=zne.scaling.folding.fold_global,
    factory=FACTORY,
)
print(f"ZNE only:             {zne_only_value:.4f}")

# 5. Error detection + ZNE
zne_executor = make_zne_executor(noise_level=NOISE_LEVEL, reps=REPS)
zne_value = zne.execute_with_zne(
    full_circuit,
    zne_executor,
    scale_noise=zne.scaling.folding.fold_global,
    factory=FACTORY,
)
print(f"Error det. + ZNE:     {zne_value:.4f}")
```

+++

## Results

```{code-cell} ipython3
labels = ["Ideal", "Noisy", "Error\nDetection", "ZNE\nOnly", "ED + ZNE"]
values = [ideal_value, noisy_value, ed_value, zne_only_value, zne_value]
colors = ["steelblue", "tomato", "goldenrod", "mediumpurple", "mediumseagreen"]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="black", linewidth=0.7)
ax.axhline(
    ideal_value,
    color="steelblue",
    linestyle="--",
    linewidth=1.2,
    label=f"Ideal = {ideal_value:.3f}",
)
ax.set_ylabel(r"Expectation value $\langle ZZII \rangle$")
ax.set_title(r"[[4,2,2]] Error Detection + ZNE on $|0_L 0_L\rangle$")
ax.set_ylim(0, 1.25)
ax.legend()

for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()
plt.show()
```

+++

## Discussion

The results illustrate the complementary nature of the two techniques:

- **Error detection alone** improves over the noisy baseline by removing shots where an error was detected
  by the syndrome measurement. However, undetected errors — such as two simultaneous single-qubit errors that
  commute with both stabilizers — still contribute residual noise.

- **ZNE alone** reduces noise through extrapolation but does not distinguish corrupted shots from clean ones.
  It improves over the noisy baseline but remains below the combined approach.

- **Combined**, error detection filters out provably corrupted shots, and ZNE then extrapolates away the
  remaining residual noise, yielding the closest result to the ideal value.

:::{note}
The `[[4,2,2]]` code *detects* errors but cannot *correct* them — shots with a detected error are simply
discarded. At higher noise rates, the fraction of accepted shots decreases significantly. This shot overhead
is a real cost that should be accounted for when estimating the practical benefit of this protocol.
:::

+++

## Summary

In this tutorial we:

- Prepared the logical state $|0_L 0_L\rangle$ of the `[[4,2,2]]` error-detecting code in Cirq
- Measured the `XXXX` and `ZZZZ` stabilizers using ancilla qubits to detect errors
- Discarded shots with detected errors using post-selection
- Applied ZNE with and without post-selection using {func}`mitiq.zne.execute_with_zne`
- Compared five cases: ideal, noisy, error detection only, ZNE only, and combined

For further reading, see the [ZNE user guide](../guide/zne.md), the [REM user guide](../guide/rem.md),
and the [Quantum Error Correction Zoo entry for the [[4,2,2]] code](https://errorcorrectionzoo.org/c/stab_4_2_2).
