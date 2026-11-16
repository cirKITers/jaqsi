# Training

Training asks how a change in circuit or pulse parameters changes a measurable objective.
For exact simulator outputs, `Script` exposes this dependence as a differentiable JAX function, so the same optimization workflow applies to ideal gates and noisy pulse evolution.
Define a scalar cost, take its gradient with `jax.grad` or `jax.value_and_grad`, and pass the result to an optimizer such as [Optax](https://optax.readthedocs.io/).
`jax.jit` can compile the update step for repeated calls.

## A minimal training loop

Let's start small: define a circuit, wrap it in a `Script`, and turn an expectation value into a scalar cost. Here training drives $\langle Z_0 \rangle$ toward $-1$:

```python
import jax
import jax.numpy as jnp
import optax

# Before importing jaqsi: gate matrices are cast to the active dtype, and
# modules that build JAX arrays at import time bake the dtype in.
jax.config.update("jax_enable_x64", True)

import jaqsi
from jaqsi import Gates


def circuit(params):
    Gates.RY(params[0], wires=0)
    Gates.RY(params[1], wires=1)
    Gates.CX(wires=[0, 1])


script = jaqsi.Script(circuit, n_qubits=2)
obs = [jaqsi.PauliZ(wires=0)]


def cost(params):
    return script.execute(type="expval", obs=obs, args=(params,))[0]
```

Optax applies the parameter updates. The whole step is JIT-compiled, so JAX traces the circuit once for these input shapes and reuses the compiled program across epochs.

```python
params = jnp.array([0.1, 0.2])
opt = optax.adam(0.05)
opt_state = opt.init(params)


@jax.jit
def step(params, opt_state):
    loss, grads = jax.value_and_grad(cost)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss


for epoch in range(1, 101):
    params, opt_state, loss = step(params, opt_state)
    if epoch % 25 == 0:
        print(f"epoch {epoch:3d}  loss {loss:+.6f}")
# epoch 100  loss -1.000000
```

## Fitting data

For more than one input, pass a batch to `execute` and use `in_axes` to say which arguments carry a batch dimension, following the same convention as `jax.vmap`. Here `x` uses axis `0`, while the trainable weights use `None` and are shared across all samples:

```python
def model_circuit(x, weights):
    Gates.RX(x, wires=0)
    Gates.RY(weights[0], wires=0)
    Gates.CX(wires=[0, 1])
    Gates.RY(weights[1], wires=1)


mscript = jaqsi.Script(model_circuit, n_qubits=2)

xs = jnp.linspace(0.0, jnp.pi, 16)
ys = jnp.cos(xs)


def predict(weights):
    return mscript.execute(
        type="expval", obs=obs, args=(xs, weights), in_axes=(0, None)
    )[:, 0]


def mse(weights):
    return jnp.mean((predict(weights) - ys) ** 2)
```

Optimize `mse` with the same Optax pattern, using `mse` as the loss instead of `cost`.

For large batches, `Script` automatically splits the `vmap` into chunks when the estimated peak memory would exceed what is available. On CPU, it further uses cache-sized tiles so repeated gate operations can reuse data already in cache. Pulse ODE solves have a separate, smaller tile budget. These estimates and limits are defined in `memory.py`.

Samples in a batch are independent, so `Script` can also distribute a batch across configured CPU devices. The batch must divide evenly across devices and pass the `memory.SHARD_MIN_SIZE` threshold, measured in state amplitudes; below that threshold, dispatch overhead can outweigh the gain. Other batches run on one device. Pulse-level gradients also run on one device because diffrax's ODE loop cannot be reverse-differentiated inside `jax.shard_map`.

## How gradients are computed

For expectation values of noise-free circuits, `Script` differentiates with the adjoint method instead of letting JAX tape every intermediate state.
The backward pass walks the circuit in reverse and reconstructs intermediate states by inverting each gate. It needs only a few statevectors regardless of circuit depth, while producing the same gradient as standard autodiff for supported circuits.
The method needs unitary gates and expectation values, so `Script` falls back to standard reverse-mode autodiff for noisy circuits, shot-based, state or probability outputs, and gates that are not unitary.
Forward-mode differentiation (`jax.jvp`, `jax.jacfwd`) falls back as well.
This is detected from the arguments to `execute`, so keep any `jax.jit` outside the forward-mode transform.

Lindblad channels use standard reverse-mode autodiff through the density-matrix circuit, since a unitary inverse cannot undo dissipation.
Inside each pulse solve, Diffrax's [checkpointed differentiation](https://docs.kidger.site/diffrax/api/adjoints/) trades recomputation for memory.
Both outputs and gradients depend on numerical solver accuracy, so check their convergence when optimizing controls.

## Training pulse parameters

The same loop works one level lower, on the pulse parameters that define a gate.
This is the idea behind [quantum optimal control](pulses.md#quantum_optimal_control):
express a gate at the pulse level, then optimize its pulse parameters so that the
resulting evolution reproduces a target unitary.

The cost is an infidelity between the pulse-level state and the ideal gate's state:

```python
from jaqsi.pulses import PulseInformation
from jaqsi.math import fidelity

theta = jnp.pi / 2


def target_state():
    def c(w):
        Gates.RX(w, wires=0)

    return jaqsi.Script(c, n_qubits=1).execute(type="state", args=(theta,))


def pulse_state(pulse_params):
    def c(w, pp):
        Gates.RX(w, wires=0, pulse=True, pulse_params=pp)

    return jaqsi.Script(c, n_qubits=1).execute(type="state", args=(theta, pulse_params))


target = target_state()


def infidelity(pulse_params):
    return 1.0 - fidelity(target, pulse_state(pulse_params))
```

Because the pulse Hamiltonian depends on `pulse_params`, gradients flow through the ODE solve to those controls.
The optimization therefore uses the same Optax loop, now adjusting the pulse rather than the gate angle.
Starting from deliberately detuned parameters, it recovers the target output state:

```python
pulse_params = PulseInformation.gate_by_name("RX").params * 1.15
opt = optax.adam(0.01)
opt_state = opt.init(pulse_params)

for _ in range(200):
    loss, grads = jax.value_and_grad(infidelity)(pulse_params)
    updates, opt_state = opt.update(grads, opt_state, pulse_params)
    pulse_params = optax.apply_updates(pulse_params, updates)
# infidelity 6.3e-02 -> ~1e-10
```

Matching this one state does not yet establish that the pulse implements the intended gate on other inputs.
For unitary gate calibration, `QOC` compares multiple probe states and combines fidelity and phase costs with optional pulse-width and evolution-time penalties.
It also supplies a parameter scan to choose the starting point, learning-rate scheduling and gradient clipping.
The shipped `qoc_results_<envelope>.csv` parameters were produced with that optimization process.

```python
from jaqsi.qoc import QOC, default_qoc_params

qoc = QOC(**default_qoc_params)
qoc.optimize_all(sel_gates=["RX"], make_log=False)
```

See the [pulses](pulses.md) page for the cost-function registry and the available
envelopes, and the [references](references.md#quantum_optimal_control) for the full `QOC` API.

## Training with noisy evolution

Training with [open-system evolution](pulses.md#open_system_pulse_evolution) includes relaxation, dephasing and drive fluctuations in the objective.
For preparation of a pure target $|\psi_{\rm target}\rangle$ from a noisy state $\rho(p)$, use

$$
L(p)=1-\langle\psi_{\rm target}|\rho(p)|\psi_{\rm target}\rangle.
$$

Minimizing this loss maximizes the probability of measuring the target state.
For $|1\rangle$, it reduces to $1-P_1$, as in the pulse example.
A nonzero minimum can reflect dissipation or control constraints.

### Choosing the objective

Matching one output state does not establish that a pulse implements the intended gate on other inputs.
For gate calibration, average the preparation loss over several probes, comparing each channel output with the ideal target $U|\psi_j\rangle$.
The six eigenstates of X, Y and Z cover both populations and coherences for one qubit.
The QOC unitary and global-phase objectives are not mixed-state objectives, so define the noisy loss explicitly and optimize it with JAX/Optax.

Use exact probabilities, density matrices or expectation values for these gradients.
Sampled shots and discrete readout labels do not vary smoothly with controls.
The mock's [exact interface](backend.md#drive_noise_and_gradients) supports differentiation, while its sampled readout tests finite-shot and classification effects.

### Optimizing across noise realizations

Optimizing one noise trace can favor controls that compensate only that particular error.
For traces $\xi_r$ drawn from a control-independent distribution, average the losses and differentiate with those traces held fixed.

$$
\widehat L_N(p)=\frac1N\sum_{r=1}^{N}L(p;\xi_r),
\qquad
\nabla_p\widehat L_N=\frac1N\sum_{r=1}^{N}\nabla_pL(p;\xi_r).
$$

Each term must evolve a complete circuit to retain noise [correlations across gates and wires](noise.md#classical_drive_noise).
Average losses directly, since applying a nonlinear loss to the averaged state can give a different objective.

Keep circuit topology and noise traces fixed within each derivative evaluation while varying continuous controls such as amplitudes and durations.
Traces may be resampled between optimization steps or reused as a training ensemble, but evaluate the result on independent traces to check that improvements generalize.

### Checking a control gradient

Compare autodiff with a centered finite difference using the same noise trace in both evaluations.

$$
\frac{\partial L}{\partial p_i}
\approx\frac{L(p+h e_i;\xi)-L(p-h e_i;\xi)}{2h},
$$

where $e_i$ selects one control.
Try several step sizes to balance truncation error against roundoff and solver error, then tighten ODE tolerances until both loss and gradient stabilize.
For widths and durations, check the [pulse boundaries](pulses.md#pulse_boundaries_and_accuracy) as well, since tighter tolerances cannot repair a missing boundary derivative.

For background on automatic differentiation in quantum control, see [Leung et al. (2017)](https://arxiv.org/abs/1612.04929).
Related approaches include [trajectory-based open-system optimization](https://arxiv.org/abs/1901.05541) and [combined analytic and automatic differentiation](https://arxiv.org/abs/2205.15044), while JAQSI here differentiates deterministic density-matrix evolution.
