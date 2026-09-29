# Pulses

At the **pulse level**, a gate stops being a black box: it becomes a time-dependent control pulse. You can follow the underlying Hamiltonian evolution and see what changes when you tune the pulse. This page introduces the implementation; [Tilmann's bachelor's thesis](https://doi.org/10.5445/IR/1000184129) goes deeper into pulse simulation and quantum Fourier models.

JAQSI can run pulse simulation on a GPU, but each pulse requires an ODE solve. For small solves, GPU launch overhead can outweigh the compute. `jaqsi.Evolution.set_solver_defaults(host_offload=True)` runs those solves on the CPU while the rest of the circuit remains on the GPU. This works for forward simulation and eager gradients, but not inside a jitted gradient.

The best device choice depends on both circuit width and the number of solves. Small circuits with few pulses may run best on the CPU. As circuits grow, GPU gate operations become more useful; with many pulse solves per call, keeping the solves on the GPU may also help. Benchmark the options for your workload.

RX, RY, RZ, and CZ form the fundamental pulse gate set. Other gates are decomposed into these gates, as shown below:

![Dependency Graph](figures/pulse_gates_dependencies_light.png#center#only-light)
![Dependency Graph](figures/pulse_gates_dependencies_dark.png#center#only-dark)

Read the graph from the bottom up: the fundamental gates sit at the bottom, and each edge weight counts the child gates used in a decomposition.

Pulse gates use the same `Gates` entry point as unitary gates. Add `pulse=True` to an individual call:

```python
from jaqsi import Gates

Gates.CY(wires=[0, 1], pulse=True)
```

Because the flag applies per call, the same circuit function can use pulse gates, unitary gates, or both. Use the `Gates` interface so noise and pulse parameters are handled consistently; calling `PulseGates` directly bypasses that handling.

## Pulse Parameters per Gate

`PulseInformation` provides each gate's pulse parameter count and optimized values. It also exposes the gate's decomposition and its leaf parameters:

```python
from jaqsi.gates import PulseInformation as pinfo

gate = "CX"

print(f"Number of pulse parameters for {gate}: {pinfo.num_params(gate)}")
# Number of pulse parameters for CX: 9

gate_instance = pinfo.gate_by_name(gate)

print(f"Childs of {gate}: {gate_instance.childs}")
# Childs of CX: [H, CZ, H]

print(f"All parameters of {gate}: {len(gate_instance.params)}")
# All parameters of CX: 9

print(f"Leaf parameters of {gate}: {len(gate_instance.leaf_params)}")
# Leaf parameters of CX: 5
```

The leaf count is smaller than the full count for a reason: CX contains two Hadamard gates, and each Hadamard decomposes into RY and RZ. By default, occurrences of the same leaf gate share parameters. The leaf count therefore represents the distinct values that need tuning, while the full count includes repeated occurrences. You can override the defaults for a call.

## Calling Gates in Pulse Mode

When `pulse_params` is omitted, `Gates` uses the optimized defaults for the active envelope. To experiment, pass your own values for a call:

```python
w = 3.14159

# CX with default parameters shared across equal leaf gates
Gates.CX(wires=[0, 1], pulse=True)

# CX with custom pulse parameters
pulse_params = pinfo.gate_by_name("CX").params * 1.1
Gates.CX(wires=[0, 1], pulse=True, pulse_params=pulse_params)

# RX gate with a rotation angle and default pulse parameters
Gates.RX(w, wires=0, pulse=True)
```

## Pulse Envelopes and Solver

Each pulse is shaped by an envelope. The available envelopes can be queried with `PulseEnvelope.available()`:

```python
from jaqsi.gates import PulseEnvelope

print(PulseEnvelope.available())
# ['gaussian', 'square', 'cosine', 'drag', 'sech', 'general']
```

The default is `gaussian`. The envelope is a process-global setting, switched with `PulseInformation.set_envelope("drag")`.
Parameter counts depend on the envelope; the counts above are for `gaussian`.

Every envelope except `drag` drives a single quadrature: its envelope $E(t)$ modulates the carrier $\cos(\omega_c t + \phi)$, with $\phi = 0$ for `RX` and $\phi = \pi/2$ for `RY`.
The envelope is centered at the midpoint $T/2$ of the pulse, where the duration $T$ is the last pulse parameter of `RX` and `RY`.
`gaussian` is lifted like Qiskit's [`Gaussian`](https://quantum.cloud.ibm.com/docs/api/qiskit/1.4/qiskit.pulse.library.Gaussian), so that it vanishes at the pulse edges instead of switching on and off with a step.
With parameters $[A, \sigma]$ and $g(t) = e^{-(t - T/2)^2 / (2\sigma^2)}$, its envelope is

$$
E(t) = A\,\frac{g(t) - g(0)}{1 - g(0)},
$$

which peaks at $A$ in $T/2$ and is zero at $t = 0$ and $t = T$.
For $\sigma \gg T$, it tends to the parabola $4 A t (T - t) / T^2$.
`drag` (Derivative Removal by Adiabatic Gate) follows [Motzoi et al. (2009)](https://doi.org/10.1103/PhysRevLett.103.110501) and adds a second control on the orthogonal carrier $\cos(\omega_c t + \phi + \pi/2)$.
With parameters $[A, \beta, \sigma]$, the in-phase envelope $E(t)$ is the lifted Gaussian above, and the quadrature envelope is its derivative $Q(t) = -\beta \dot{E}(t)$, where $\beta$ takes the place of the inverse anharmonicity $1/\Delta$ in Eq. (9) of Motzoi et al. (see also [Gambetta et al. (2011)](https://doi.org/10.1103/PhysRevA.83.012308)).
While $E$ vanishes at the pulse edges, $\dot{E}$ does not, so $Q$ starts and ends with a step.
Qiskit's [`Drag`](https://quantum.cloud.ibm.com/docs/api/qiskit/1.4/qiskit.pulse.library.Drag) instead takes its quadrature proportional to $\beta\,(t - T/2)\,E(t) / \sigma^2$, which vanishes at the edges but is not the derivative of the lifted $E$.
Under the RWA, `RX(w)` then evolves under $\frac{w}{2}\left(E X + Q Y\right)$ and `RY(w)` under $\frac{w}{2}\left(E Y - Q X\right)$.
Note that jaqsi models qubits as two-level systems, so there is no leakage level for the quadrature to suppress.
As $E$ is symmetric around $T/2$, $Q$ is odd around it and adds no net area, but it does not commute with the in-phase drive and adds an error about the $Z$ axis whose angle grows as $\beta w^2$, the second term of the [Magnus expansion](https://doi.org/10.1016/j.physrep.2008.11.001).
Under the RWA, the Gaussian alone already implements the target rotation, which is why the calibrated defaults have $\beta \approx 0$ (below $10^{-12}$) and `drag` then reproduces `gaussian`.

Pulse gates integrate a time-dependent Hamiltonian. Configure the solver with `Evolution.set_solver_defaults`; available solvers are `"dopri8"` (default), `"dopri5"`, `"magnus2"`, and `"magnus4"`:

```python
from jaqsi import Evolution

Evolution.set_solver_defaults(solver="magnus4", magnus_steps=128)
```

The `magnus_steps` argument sets the number of fixed substeps for the Magnus integrators and is ignored for the adaptive Dormand-Prince solvers (`dopri8`, `dopri5`).

A drive with a single term, $H(t) = f(t)\,H$, commutes with itself at all times, so its gate is $e^{-i F H}$ with $F = \int f(t)\,dt$.
The Dormand-Prince solvers then integrate the scalar $F$ only, which takes a handful of steps whatever the rotation angle, whereas the matrix ODE needs more steps the larger the angle.
`RZ` and `CZ` are always such drives, and so are `RX` and `RY` under the RWA with a single-quadrature envelope such as `gaussian`.
With `drag`, or without the RWA, `RX` and `RY` keep two non-commuting terms and are integrated as a matrix ODE.
`Evolution.set_solver_defaults(closed_form=False)` integrates single-term drives as a matrix ODE as well, e.g. to compare against simulators that do not exploit this.

The Magnus integrators and the single-term drives return exactly unitary gates, the Dormand-Prince solvers of the matrix ODE only up to their tolerance (about 1e-10 per gate in double precision).
This matters for gradients of expectation values, which are computed with the adjoint method (see [training](training.md#how_gradients_are_computed)) and reconstruct intermediate states by inverting gates as unitaries.
The resulting error grows linearly with the number of pulse gates, so for circuits with thousands of them prefer a Magnus solver.

Pulse gates are solved lazily when the circuit runs, rather than when they are recorded. Gates with the same pulse shape share one batched solve, while gates with identical parameters, such as repeated fixed-angle rotations, are solved only once. This avoids compiling a separate solver for each gate.

## Quantum Optimal Control

`QOC` helps find pulse parameters that behave like the intended gate. It compares a pulse gate with its ideal unitary counterpart. Its `create_GATE` methods build a pair of test circuits, one for each implementation. Other unitary gates in those circuits prepare different input states, so matching a gate on one input alone is not enough. A parameter `w` lets the comparison cover different gate angles. See the `QOC` API reference for the individual circuit factories.

Create a `QOC` instance with the default parameters:

```python
from jaqsi.qoc import QOC, default_qoc_params

qoc = QOC(**default_qoc_params)
```

Pass `sel_gates` to `optimize_all` to choose which gates to tune:

```python
qoc.optimize_all(sel_gates=["RX", "RY", "RZ", "CZ"])
```

The run writes optimization logs to `qoc_logs.csv` and the resulting pulse parameters to `qoc_results_<envelope>.csv`.
  
QOC uses a weighted cost to compare each pulse gate with its target unitary. By default, it combines process infidelity and phase error; pulse width and evolution time can be added as optional terms. The weights and other optimization settings are listed in `default_qoc_params`.

You can also select the pulse envelope.
Under the RWA, the rotation of a single-quadrature pulse only depends on its area $\int_0^T E(t)\,dt$, so the calibration fixes the area but leaves the shape of the pulse open.

The [pulses notebook](https://github.com/cirKITers/jaqsi/blob/main/docs/pulses.ipynb) has more gates to try.

The optimized parameters produce these gate fidelities:

![Gate Fidelities](figures/gates_fidelities_light.png#center#only-light)
![Gate Fidelities](figures/gates_fidelities_dark.png#center#only-dark)

The plot shows phase error as $1-\text{phase error}$ to match the fidelity scale.
