# Noise

Every gate accepts an optional `noise_params` dictionary. Start with zeros, then turn up the channels you want to explore:
```python
noise_params = {
    "BitFlip": 0.0,
    "PhaseFlip": 0.0,
    "AmplitudeDamping": 0.0,
    "PhaseDamping": 0.0,
    "Depolarizing": 0.0,
    "MultiQubitDepolarizing": 0.0,
}
```

Bit flip, phase flip, and single- or multi-qubit depolarizing channels run after each gate. Amplitude and phase damping run at the end of the circuit. These channels are `KrausChannel` operations in `jaqsi.noise`; passing `noise_params` through `Gates` records them for you.

Once a noise channel is on the tape, `Script` switches from statevector to density-matrix simulation for you. Use `type="density"` to inspect the resulting noisy state:

```python
import jaqsi
from jaqsi import Gates

noise_params = {
    "BitFlip": 0.01,
    "PhaseFlip": 0.02,
    "AmplitudeDamping": 0.03,
    "PhaseDamping": 0.04,
    "Depolarizing": 0.05,
    "MultiQubitDepolarizing": 0.06,
}

def circuit(theta):
    Gates.RX(theta[0], wires=0, noise_params=noise_params)
    Gates.CX(wires=[0, 1], noise_params=noise_params)

rho = jaqsi.Script(circuit, n_qubits=2).execute(type="density", args=(theta,))
```

`GateError` models a different kind of error: it perturbs each parameterized gate angle as $w \mapsto w + \mathcal{N}(0, \epsilon)$, with standard deviation $\sqrt{\epsilon}$. Each gate draws its own perturbation. Because this error is stochastic, pass a `random_key` alongside it.

`UnitaryGates.batch_gate_error` controls how this perturbation behaves across batched inputs. By default, each batch element gets its own draw; the other setting shares one draw for a given gate across the batch. This is useful when samples should see the same gate error:

```python
import jax
from jaqsi.gates import UnitaryGates

UnitaryGates.batch_gate_error = False

def circuit(theta, key):
    Gates.RX(theta[0], wires=0, noise_params={"GateError": 0.01}, random_key=key)
    Gates.CX(wires=[0, 1])

rho = jaqsi.Script(circuit, n_qubits=2).execute(
    type="density", args=(theta, jax.random.key(0))
)
```

## Randomness under JAX transformations

Gate errors and shot sampling draw random numbers at runtime. The other noise channels above are deterministic maps on density matrices, even though they represent physical noise.

Inside a JAX transformation a key captured as a constant does not work: a jitted function is traced once, so the compiled function replays the same noise realization on every call.
To get fresh randomness, pass the key explicitly as an argument and advance it outside the transformation:

```python
train_step = jax.jit(lambda params, key: cost(script.execute(
    type="density", args=(params, key)
)))

key = jax.random.key(0)
for _ in range(n_steps):
    key, sub_key = jax.random.split(key)
    loss = train_step(params, sub_key)
```

Since the key is an argument rather than a constant, this does not trigger recompilation.
