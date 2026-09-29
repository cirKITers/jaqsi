# Usage

Your circuit is just a Python function: call gates inside it, then hand it to `Script` to run. This gives you direct control over gate placement, noise, measurements, and pulse simulation. Higher-level tools such as [qml-essentials](https://github.com/cirKITers/qml-essentials) build on the same interfaces.

The diagram shows how JAQSI supports higher-level `Model` and `Ansaetze` interfaces. Gates and observables are `Operation` objects recorded on a `Tape`; `Script` executes the tape and returns measurements.

![overview](figures/jaqsi_overview_light.png#center#only-light)
![overview](figures/jaqsi_overview_dark.png#center#only-dark)

Pulse simulation adds two pieces to gate execution: `PulseParams` stores the values used to construct a pulse, and `PulseEnvelope` defines its shape. `PulseInformation` coordinates their valid combinations. Most circuits use these through `Gates`, which selects `PulseGates` when `pulse=True`.

![overview](figures/jaqsi_pulse_light.png#center#only-light)
![overview](figures/jaqsi_pulse_dark.png#center#only-dark)

## Architecture

For a peek under the hood, here is how each module helps turn a circuit function into a measurement:

- `operations.py` defines the `Operation` base class for gates, observables, and noise channels, along with the Hamiltonian objects used for evolution.
- `gates.py` provides `Gates`, the circuit-facing interface. It records a gate, attaches requested noise, and selects `UnitaryGates` or `PulseGates` from the `pulse` flag.
- `unitary.py` applies ideal gates from `gateset.py` and handles gate-angle errors and noise channels.
- `pulses.py` implements RX, RY, RZ, and CZ as pulses and builds composite gates from them. It also contains pulse envelopes, parameters, and calibration state.
- `gateset.py` defines concrete gate and observable classes. Instantiating one inside a recorded circuit adds it to the tape.
- `paulis.py` provides symbolic Pauli and Clifford operations, including `PauliWord`, without constructing a full matrix for every operation.
- `noise.py` defines Kraus channels. Recording one switches execution from a statevector to a density matrix.
- `tape.py` collects operations in circuit order before simulation starts.
- `script.py` provides `Script`, the execution interface. It records a circuit, selects the simulation mode, handles batching and caching, and draws circuits.
- `simulation.py` contains the statevector, density-matrix, and measurement kernels that run the recorded tape.
- `memory.py` estimates batch memory use and splits batches into chunks when needed.
- `drawing.py` renders circuit diagrams and pulse schedules.
- `evolution.py` turns static or time-dependent Hamiltonians into gates. It also batches pending pulse solves by pulse shape.
- `__init__.py` re-exports the common interfaces for `import jaqsi`.
- `math.py` provides quantum information functions for states and density matrices.
- `qoc.py` optimizes pulse parameters against target gates.

A call to `Script.execute(...)` then runs four stages:

1. **Record:** Run the circuit function once so its operations register on a fresh `Tape`.
2. **Prepare:** Infer the qubit count and choose statevector or density-matrix simulation from the recorded operations.
3. **Simulate:** Apply gates and channels in circuit order, resolving pulse evolutions when needed.
4. **Measure:** Return a state, probabilities, expectation values, or a density matrix; optionally sample shots.

As the whole pipeline is built on JAX, any execution can be differentiated, JIT-compiled and vectorized.

## Usage

Now let's put those pieces to work in a circuit.

### Gate Level

**`Gates` is the entry point for applying gates.**
Calling `Gates.<Name>(...)` inside a circuit function records the gate on the active tape,
attaches any requested noise, and routes the call to the unitary or the pulse backend.
Write circuits against `Gates` and the same circuit runs at either level; see
[pulse level](#pulse_level) below.

Import JAQSI and `Gates`:

```python
import jaqsi as js
from jaqsi import Gates
```

Define a circuit and observables:

```python
def circuit():
    Gates.H(wires=0)
    Gates.CX(wires=[0, 1])

obs = [js.PauliZ(wires=0), js.PauliZ(wires=1)]
```

Observables are operation objects passed to `execute`, rather than gates applied inside the circuit. They are available from `jaqsi` or `jaqsi.gateset`.

Run the Bell circuit and ask for its probabilities:

```python
jss = js.Script(circuit)
jss.execute(type="probs", obs=obs)
```

Pass circuit parameters through `execute(args=...)`:

```python
import jax.numpy as jnp

n_qubits = 1

def circuit(phi, theta, omega):
    Gates.Rot(phi, theta, omega, wires=0)
    Gates.Rot(jnp.pi, 1/2*jnp.pi, 1/4*jnp.pi, wires=0)

obs = [js.PauliZ(wires=i) for i in range(n_qubits)]
jss = js.Script(circuit)
jss.execute(type="expval", obs=obs, args=(jnp.pi, 1/2*jnp.pi, 1/4*jnp.pi))
```

To tune the parameters, use JAX and Optax directly:

```python
import jax
import optax as otx

def cost_fct(params):
    phi, theta, omega = params
    return jss.execute(type="expval", obs=[js.PauliZ(0)], args=(phi, theta, omega))[0]

params = jax.numpy.array([0.1, 0.2, 0.3])
opt = otx.adam(0.01)
opt_state = opt.init(params)

for epoch in range(1, 101):
    grads = jax.grad(cost_fct)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = otx.apply_updates(params, updates)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Cost: {cost_fct(params):.4f}")
```

See the [training](training.md) page for the JIT-compiled version of this loop, fitting
data with batched inputs, and optimizing pulse parameters.

`Gates.<Name>(...)` records the gate and returns nothing.  When you need the operation
*object* itself — to call `.dagger()` or `.power()` on it, to build a composite observable,
or to do matrix algebra — take the class from `jaqsi.gateset` instead:

```python
from jaqsi.gateset import RX, PauliX

def circuit():
    RX(0.5, wires=0)
    RX(0.5, wires=0).dagger()
    PauliX(wires=0).power(2)

obs = [js.PauliZ(0)]
jss = js.Script(circuit)
res = jss.execute(type="expval", obs=obs)

print(res) # we expect to end up in |0⟩ again
```

Noise is normally requested per gate through `Gates`, which emits the matching channels
for you (see [noise](noise.md)):

```python
noise_params = {"Depolarizing": 0.1}

def circuit():
    Gates.H(wires=0, noise_params=noise_params)
    Gates.CX(wires=[0, 1], noise_params=noise_params)

jss = js.Script(circuit)
rho = jss.execute(type="density")
purity = jnp.real(jnp.trace(rho @ rho))
print(purity) # Purity should be < 1 
```

Channels are operations too, so they can also be placed by hand from `jaqsi.noise` when you
want control over exactly where they land.

By default the simulation starts from the all-zero state $\lvert 0\dots0\rangle$.
To start from an arbitrary statevector instead, pass it via the `initial_state`
argument of `execute`:

```python
def circuit():
    Gates.RX(0.3, wires=0)

jss = js.Script(circuit)
plus = jnp.array([1.0, 1.0], dtype=complex) / jnp.sqrt(2.0)  # |+⟩
res = jss.execute(type="expval", obs=[js.PauliZ(0)], initial_state=plus)
```

Without `in_axes` the state must be a single statevector of shape `(2**n,)`. 
When batching with `in_axes`, `initial_state` may be a single 1D state broadcast across the batch, or a 2D array of shape `(B, 2**n)` that provides one state per sample.

### Pulse Level

The same circuit interface also works for pulses. This section shows how to switch levels; the [pulse guide](pulses.md) explores envelopes, Hamiltonians, and quantum optimal control in more detail.

Pulse-level simulation goes through the same entry point: pass `pulse=True` to any gate and
`Gates` routes it to the pulse backend instead of the ideal unitary.
Nothing else about the circuit changes.

```python
def circuit(w):
    Gates.RX(w, wires=0, pulse=True)

obs = [js.PauliZ(0)]
jss = js.Script(circuit)
res = jss.execute(type="expval", obs=obs, args=(jnp.pi*0.5,))
print(res)  # approximately zero
```

Because the flag is per call, a circuit can mix both levels — here the entangling gate stays
ideal while the rotations are lowered to pulses:

```python
def circuit(w):
    Gates.RX(w, wires=0, pulse=True)
    Gates.RY(w, wires=0, pulse=True)
    Gates.CX(wires=[0, 1])
```

Mixing pulse-level simulation with noisy simulations is possible as well:

```python
noise_params = {"Depolarizing": 0.1}

def circuit(w):
    Gates.RX(w, wires=0, pulse=True, noise_params=noise_params)
    Gates.RY(w, wires=0, pulse=True, noise_params=noise_params)
    Gates.CX(wires=[0, 1], noise_params=noise_params)

jss = js.Script(circuit)
rho = jss.execute(type="density", args=(jnp.pi*0.5,))
purity = jnp.real(jnp.trace(rho @ rho))
print(purity) # Purity should be < 1 
```

Use `draw` to plot when pulses act on each qubit. Shaded regions show the pulse envelopes and vertical lines mark virtual Z gates. Composite gates are decomposed into their basis gates for the plot; for example, `H` becomes RZ and RY.

```python
def circuit(w):
    Gates.RX(w, wires=0, pulse=True)
    Gates.CZ(wires=[0, 1], pulse=True)
    Gates.H(wires=1, pulse=True)
    Gates.H(wires=1, pulse=True)

jss = js.Script(circuit)

fig, axes = jss.draw(figure="pulse", args=(jnp.pi*0.5,))
```

![pulse-schedule](figures/pulse_schedule_light.png#center#only-light)
![pulse-schedule](figures/pulse_schedule_dark.png#center#only-dark)

Underneath a pulse gate is Hamiltonian evolution over time. You can use that evolution interface directly; this circuit starts with a static Pauli Z Hamiltonian.

```python
def evol_circuit(t):
    time_evol = js.Hermitian(matrix=js.PauliZ._matrix, wires=0).evolve()
    time_evol(t=t, wires=0)
```

Execute it with `Script`:

```python
jss = js.Script(f=evol_circuit)
res = jss.execute(type="expval", obs=[js.PauliX(0)], args=(0.3,))
```

Here, the circuit starts in $|0\rangle$, so Z evolution changes only its phase and the X expectation value stays zero. Preparing $|+\rangle$ first makes the evolution visible in that measurement:

```python
def evol_circuit(t):
    Gates.H(wires=0)  # prepare |+⟩
    time_evol = js.Hermitian(matrix=js.PauliZ._matrix, wires=0).evolve()
    time_evol(t=t, wires=0)
```

Measure the evolved state:

```python
t = 0.3
jss = js.Script(f=evol_circuit)
res = jss.execute(type="expval", obs=[js.PauliX(0)], args=(t,))
```

The X expectation value is `jnp.cos(2 * t)`. This example combines an ordinary gate with Hamiltonian evolution in one circuit.

The Hamiltonian can also depend on a parameter. Multiply a `Hermitian` by a coefficient function to describe how its strength changes with the supplied parameters and time:

```python
def coeff(p, t):
    return p

def circuit(p, t):
    Gates.H(wires=0)  # prepare |+⟩
    ph = coeff * js.Hermitian(matrix=js.PauliZ._matrix, wires=0, record=False)
    ph.evolve()([p], t)
```

The callable `coeff` defines the dependence, while `[p]` supplies its parameter when the circuit runs. Keeping the coefficient function separate lets JAQSI reuse its compiled evolution code.

```python
p = 0.5
jss = js.Script(f=circuit)
res = jss.execute(type="expval", obs=[js.PauliX(0)], args=(p, t))
```

And because the circuit is differentiable, you can train `p` with JAX as in the [training guide](training.md).
