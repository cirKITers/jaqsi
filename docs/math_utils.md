# Math Utils

These model-independent functions operate directly on statevectors and density matrices. Bring a state from a JAQSI circuit or from elsewhere; the same functions work for both.

```python
from jaqsi.math import quantum_fisher_information, fubini_study_metric, fidelity, trace_distance, phase_difference
```

## Quantum Fisher Information

Quantum Fisher Information (QFI) is a metric tensor on the space of quantum states, evaluated at a parameter point $\theta$. It depends on derivatives of the state with respect to those parameters, so the input must be a **function** that returns a state rather than a single state value. JAQSI uses forward-mode automatic differentiation to compute the complex Jacobian for real-valued circuit parameters.

For a pure, normalized state $\ket{\psi(\theta)}$ the QFI is the Fubini-Study metric (scaled by four):

\[F_{ij} = 4\,\mathrm{Re}\left[\braket{\partial_i\psi | \partial_j\psi} - \braket{\partial_i\psi | \psi}\braket{\psi | \partial_j\psi}\right]\]

For a mixed state $\rho(\theta) = \sum_k p_k \ket{k}\bra{k}$ the QFI is given through the symmetric logarithmic derivative:

\[F_{ij} = 2 \sum_{k, l\,:\,p_k + p_l > 0} \frac{\mathrm{Re}\left(\braket{k | \partial_i\rho | l}\braket{l | \partial_j\rho | k}\right)}{p_k + p_l}\]

`quantum_fisher_information` selects the formula from the shape of the state returned by the callable. Execute a circuit with `type="state"` for pure-state QFI, or `type="density"` for mixed-state QFI, such as in a noisy circuit:

```python
import jax.numpy as jnp
from jaqsi.script import Script
from jaqsi import Gates
from jaqsi.math import quantum_fisher_information, fubini_study_metric

def state_fn(theta):
    def circuit(t):
        Gates.RX(t[0], wires=0)
        Gates.RY(t[1], wires=1)
        Gates.CX(wires=[0, 1])
    return Script(circuit, n_qubits=2).execute(type="state", args=(theta,))

theta = jnp.array([0.7, 1.3])
qfi = quantum_fisher_information(state_fn, theta)
metric = fubini_study_metric(state_fn, theta)
```

The result is a real, symmetric $(P, P)$ matrix, where $P$ is the total number of parameters (the parameter axes are flattened).
The state returned by the callable is assumed to be normalized, which the simulator guarantees.

Any function mapping parameters to a state vector works, so a higher-level model wrapping a
`Script` can be passed just as well, closing over any data inputs
(e.g. `lambda p: model(params=p, inputs=x)`).

## Fubini-Study Metric

The Fubini-Study metric is the real part of the quantum geometric tensor on the manifold of
pure states.
It is the underlying geometric object of the pure-state QFI and is related to it by a factor of
four, $F_{ij} = 4\,g_{ij}$:

\[g_{ij} = \mathrm{Re}\left[\braket{\partial_i\psi | \partial_j\psi} - \braket{\partial_i\psi | \psi}\braket{\psi | \partial_j\psi}\right]\]

`fubini_study_metric(state_fn, params)` follows the same calling convention as
`quantum_fisher_information` but, since the metric is only defined for pure states, requires
`state_fn` to return a state vector (`type="state"`):

```python
from jaqsi.math import fubini_study_metric

g = fubini_study_metric(state_fn, theta)
```

## Fidelity

`fidelity(state0, state1)` computes the fidelity between two states, accepting either state
vectors or density matrices.
For pure states it evaluates $F(\ket{\psi}, \ket{\phi}) = \left|\braket{\psi | \phi}\right|^2$,
while for density matrices it uses the Uhlmann fidelity
$F(\rho, \sigma) = \left(\mathrm{Tr}\sqrt{\sqrt{\rho}\,\sigma\,\sqrt{\rho}}\right)^2$.
Both single states and batches of shape $(B, \dots)$ are supported.

## Trace distance

`trace_distance(state0, state1)` returns the trace distance between two density matrices,

\[T(\rho, \sigma) = \frac{1}{2} \sum_i \left|\lambda_i\right|\]

where $\lambda_i$ are the eigenvalues of $\rho - \sigma$.

## Phase difference

`phase_difference(state0, state1)` returns the phase $\arg\braket{\psi | \phi}$ between two
state vectors.
A value of zero indicates the two states differ by at most a real global factor.

## Matrix logarithm

`logm_v(A)` computes the matrix logarithm of a single matrix of shape $(d, d)$ or of each
matrix in a batch of shape $(B, d, d)$, as used internally by the entropy-based measures.
