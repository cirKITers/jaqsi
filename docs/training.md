# Training

Everything in JAQSI is built on JAX, so a circuit executed through a `Script` is an
ordinary differentiable function.
That means training needs no special machinery: take a gradient with `jax.grad` (or
`jax.value_and_grad`), hand it to an optimiser such as [Optax](https://optax.readthedocs.io/),
and wrap the update in `jax.jit`.

## A minimal training loop

Define a circuit, wrap it in a `Script`, and turn an expectation value into a scalar cost.
Here we simply drive $\langle Z_0 \rangle$ towards $-1$:

```python
import jax
import jax.numpy as jnp
import optax

import jaqsi
from jaqsi import operations as op
from jaqsi.gates import Gates

jax.config.update("jax_enable_x64", True)


def circuit(params):
    Gates.RY(params[0], wires=0)
    Gates.RY(params[1], wires=1)
    Gates.CX(wires=[0, 1])


script = jaqsi.Script(circuit, n_qubits=2)
obs = [op.PauliZ(wires=0)]


def cost(params):
    return script.execute(type="expval", obs=obs, args=(params,))[0]
```

The optimisation itself is plain Optax.
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

`mse` is then optimised with exactly the same `step` function as above.
For large batches `Script` also chunks the `vmap` automatically so that the peak memory
stays within what is available (see `memory.py`).

## Training pulse parameters

The same loop works one level lower, on the pulse parameters that define a gate.
This is the idea behind [quantum optimal control](pulses.md#quantum-optimal-control-qoc):
express a gate at the pulse level, then optimise its pulse parameters so that the
resulting evolution reproduces a target unitary.

The cost is an infidelity between the pulse-level state and the ideal gate's state:

```python
from jaqsi.pulses import PulseGates, PulseInformation
from jaqsi.math import fidelity

theta = jnp.pi / 2


def target_state():
    def c(w):
        op.RX(w, wires=0)

    return jaqsi.Script(c, n_qubits=1).execute(type="state", args=(theta,))


def pulse_state(pulse_params):
    def c(w, pp):
        PulseGates.RX(w, wires=0, pulse_params=pp)

    return jaqsi.Script(c, n_qubits=1).execute(type="state", args=(theta, pulse_params))


target = target_state()


def infidelity(pulse_params):
    return 1.0 - fidelity(target, pulse_state(pulse_params))
```

Gradients flow through the ODE solver that integrates the pulse Hamiltonian, so the
optimisation is again a standard Optax loop.
Starting from deliberately detuned parameters, it recovers the gate:

```python
pulse_params = PulseInformation.gate_by_name("RX").params * 1.15
opt = optax.adam(0.01)
opt_state = opt.init(pulse_params)

for _ in range(30):
    loss, grads = jax.value_and_grad(infidelity)(pulse_params)
    updates, opt_state = opt.update(grads, opt_state, pulse_params)
    pulse_params = optax.apply_updates(pulse_params, updates)
# infidelity 6.3e-02 -> ~1e-04
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
envelopes, and the [references](references.md#quantum-optimal-control) for the full `QOC` API.
