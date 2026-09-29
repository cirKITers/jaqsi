# Training

Everything in JAQSI is built on JAX, so a circuit executed through a `Script` is an
ordinary differentiable function.
That means training needs no special machinery: take a gradient with `jax.grad` (or
`jax.value_and_grad`), hand it to an optimizer such as [Optax](https://optax.readthedocs.io/),
and wrap the update in `jax.jit`.

## A minimal training loop

Define a circuit, wrap it in a `Script`, and turn an expectation value into a scalar cost.
Here we simply drive $\langle Z_0 \rangle$ towards $-1$:

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

The optimization itself is plain Optax.
Note that the whole step is `jit`-compiled: the circuit is traced once and the compiled
program is reused for every epoch.

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

To fit a function you need the circuit evaluated at many inputs.
Rather than looping, pass the whole batch and tell `execute` which arguments carry a
batch dimension via `in_axes` — the same convention as `jax.vmap`.
Here the input `x` is batched (axis `0`) while the trainable weights are shared
(`None`), and `Script` vectorizes the execution for you:

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

`mse` is then optimized with exactly the same `step` function as above.
For large batches `Script` also chunks the `vmap` automatically so that the peak memory
stays within what is available (see `memory.py`).
On CPU, it further runs the batch in tiles whose working set fits in the cache (`memory.CACHE_BYTES`, read from the L3 size; set it by hand on virtual machines, which may report a per-core cache that is actually shared).
The ODE solves of pulse-level gates are held to a smaller budget, `memory.SOLVE_CACHE_BYTES` (256 KiB), since XLA's multi-threading slows their many small operations down once a tile is large.

XLA's multi-threading barely speeds up a single circuit, but the samples of a batch are independent.
To run them in parallel on CPU, expose the cores as JAX devices before JAX initialises:

```python
jax.config.update("jax_num_cpu_devices", 8)  # before the first JAX computation
```

`Script` then splits every batch whose size is a multiple of the device count over all devices, once the batch holds at least `memory.SHARD_MIN_SIZE` amplitudes in total (batch size times `2**n`, `2**13` by default; below that the dispatch costs more than it saves).
Other batches run on one device.
Gradients through pulse-level gates also stay on one device, since diffrax's ODE loop cannot be reverse-differentiated inside `jax.shard_map` yet.

## How gradients are computed

For expectation values of noise-free circuits, `Script` differentiates with the adjoint method instead of letting JAX tape every intermediate state.
Here, the backward pass walks the circuit in reverse and undoes each gate with its inverse, so the memory a gradient needs stays at a few statevectors no matter how deep the circuit is.
The result is identical to plain autodiff.
The method needs unitary gates and expectation values, so `Script` falls back to standard reverse-mode autodiff for noisy circuits, shot-based, state or probability outputs, and gates that are not unitary.
Forward-mode differentiation (`jax.jvp`, `jax.jacfwd`) falls back as well.
This is detected on the arguments of `execute`, so it keep any `jax.jit` outside the forward-mode transform.

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

for _ in range(30):
    loss, grads = jax.value_and_grad(infidelity)(pulse_params)
    updates, opt_state = opt.update(grads, opt_state, pulse_params)
    pulse_params = optax.apply_updates(pulse_params, updates)
# infidelity 6.3e-02 -> ~1e-05
```

This hand-rolled loop is only meant to show the mechanism.
For real calibration use the `QOC` class, which wraps the same idea with a multi-objective
cost (fidelity and phase, plus optional pulse-width and evolution-time penalties), a
parameter scan to pick the starting point, learning-rate scheduling and gradient clipping.
The parameters shipped in `qoc_results_<envelope>.csv` were produced that way; evaluating
`infidelity` at those defaults gives a residual on the order of machine precision.

```python
from jaqsi.qoc import QOC, default_qoc_params

qoc = QOC(**default_qoc_params)
qoc.optimize_all(sel_gates=["RX"], make_log=False)
```

See the [pulses](pulses.md) page for the cost-function registry and the available
envelopes, and the [references](references.md#quantum_optimal_control) for the full `QOC` API.
