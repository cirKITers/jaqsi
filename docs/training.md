# Training

Good news: a circuit executed through `Script` is a differentiable JAX function, so training follows the usual JAX workflow. Define a scalar cost, take its gradient with `jax.grad` or `jax.value_and_grad`, and pass the result to an optimizer such as [Optax](https://optax.readthedocs.io/). `jax.jit` can compile the update step for repeated calls.

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

Gradients flow through the ODE solver that integrates the pulse Hamiltonian, so the
optimization is again a standard Optax loop.
Starting from deliberately detuned parameters, it recovers the gate:

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

That is the basic idea: gradients flow through the pulse ODE solver too. For full gate calibration, use `QOC`, which adds a multi-objective
cost (fidelity and phase, plus optional pulse-width and evolution-time penalties), a
parameter scan to pick the starting point, learning-rate scheduling and gradient clipping.
The shipped `qoc_results_<envelope>.csv` parameters were produced with that optimization process.

```python
from jaqsi.qoc import QOC, default_qoc_params

qoc = QOC(**default_qoc_params)
qoc.optimize_all(sel_gates=["RX"], make_log=False)
```

See the [pulses](pulses.md) page for the cost-function registry and the available
envelopes, and the [references](references.md#quantum_optimal_control) for the full `QOC` API.
