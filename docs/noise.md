# Noise

Every gate in JAQSI accepts an optional `noise_params` dictionary containing all the noise parameters of the circuit (here all with probability $0.0$):
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

Providing this optional input will apply the corresponding noise, where the Bit Flip, Phase Flip, Depolarizing and Two-Qubit Depolarizing Channels are applied after each gate and the Amplitude and Phase Damping are applied at the end of the circuit.
The channels themselves live in `jaqsi.noise` as `KrausChannel` operations; the gate front-ends emit them for you when `noise_params` is passed.

Recording a noise channel on the tape automatically switches `Script` from statevector to density-matrix simulation, so the `density` execution type is what you want for a noisy circuit:

```python
import jaqsi
from jaqsi.gates import Gates

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

In addition to these decoherent errors, we can also apply a `GateError` which affects each parameterized gate as $w = w + \mathcal{N}(0, \epsilon)$, where $\sqrt{\epsilon}$ is the standard deviation of the noise, specified by the `GateError` key in the `noise_params` argument.
Each gate draws its own error, independently of the other gates in the circuit.
Because `GateError` is stochastic, a `random_key` must be passed alongside it.

It's important to note that, depending on the flag set in `UnitaryGates.batch_gate_error`, the error of a given gate will be applied to the entire batch of parameters (all batch elements are affected in the same way) or drawn for each batch element individually (default).
This can be particularly usefull in a scenario where one would like to apply noise e.g. only on a subset of the gates but wants to change them all uniformly.
An example of this is provided in the following code:

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

Gate errors and shot sampling are the only parts of the simulation that draw random numbers at runtime, as the decoherent channels above are deterministic maps on the density matrix.

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
