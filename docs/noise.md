# Noise

Gate-level channels describe circuit errors without resolving the underlying drive.
For pulse design, the Hamiltonian and environmental noise evolve together, making errors depend on the control throughout each gate.

## Gate-level noise

Every gate accepts an optional `noise_params` dictionary.
Start with zeros, then turn up the channels you want to explore:

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

Bit flip, phase flip, and single- or multi-qubit depolarizing channels run after each gate, while amplitude and phase damping run at the end of the circuit.
Passing `noise_params` through `Gates` records these `KrausChannel` operations from `jaqsi.noise`.
Their strengths describe discrete maps, not decay rates integrated over a pulse's duration.

A density matrix represents the resulting mixture, with each channel acting as

$$
\mathcal E(\rho)=\sum_k K_k\rho K_k^\dagger,
\qquad \sum_k K_k^\dagger K_k=I.
$$

The second relation preserves total probability.
`Script` automatically switches to density-matrix simulation when a channel is recorded, and `type="density"` returns that state.

```python
import jax.numpy as jnp
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

theta = jnp.array([jnp.pi / 2])
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

## Noise during a pulse

Relaxation and dephasing act while the drive changes the state.
Applying all decay after a coherent pulse generally gives a different result because the coherent and dissipative generators need not commute.
JAQSI models their simultaneous action with the Lindblad master equation, using $\hbar=1$.

$$
\dot\rho=-i[H(t),\rho]+\sum_k\mathcal D[C_k]\rho,
\qquad
\mathcal D[C]\rho=C\rho C^\dagger-\tfrac12\{C^\dagger C,\rho\}.
$$

The commutator describes coherent evolution, while the dissipators describe environmental processes and preserve the trace.
Here $\{A,B\}=AB+BA$, and each collapse operator includes a square-root rate, so $C=\sqrt{\gamma}A$ contributes $\gamma\mathcal D[A]$.
Hamiltonian coefficients, rates and integration intervals must use consistent time units.

This model treats the environment as memoryless, and JAQSI currently accepts constant collapse operators, giving fixed decay rates during each pulse.
See [Lindblad (1976)](https://doi.org/10.1007/BF01608499) and the [QuTiP guide](https://qutip.readthedocs.io/en/latest/guide/dynamics/dynamics-master.html) for the physical assumptions, and the [pulse guide](pulses.md#open_system_pulse_evolution) for an executable example.

### Relaxation and dephasing

T1 describes relaxation toward the equilibrium excited population $p_{\rm th}$, while T2 describes coherence loss.
The corresponding collapse operators are

$$
C_\downarrow=\sqrt{(1-p_{\rm th})/T_1}|0\rangle\langle1|,\quad
C_\uparrow=\sqrt{p_{\rm th}/T_1}|1\rangle\langle0|,\quad
C_\phi=\sqrt{\gamma_\phi/2}\,Z,
\qquad \gamma_\phi=1/T_2-1/(2T_1).
$$

Without a drive, populations and coherences follow

$$
p_1(t)=p_{\rm th}+[p_1(0)-p_{\rm th}]e^{-t/T_1},
\qquad
\rho_{01}(t)=\rho_{01}(0)e^{-t/T_2},
$$

apart from coherent precession.
Relaxation already contributes $1/(2T_1)$ to coherence loss, so using $1/T_2$ as the pure-dephasing rate would count that contribution twice.
Nonnegative rates require $T_1,T_2>0$ and $T_2\le2T_1$.
Taking both times to infinity removes decay.

During a drive, the state rotates while losing population and coherence, so its final excited probability is not generally the ideal value multiplied by $e^{-T/T_1}$.
The [mock backend](backend.md#mock) shows how these effects accumulate over consecutive gates.

### Classical drive noise

Control fluctuations change the Hamiltonian itself.
For a supplied realization $\xi$, fractional amplitude noise $\epsilon_A$ and phase noise $\epsilon_\phi$ in radians modify the complex drive as

$$
\Omega^\xi(t)=[1+\epsilon_A(t)]e^{i\epsilon_\phi(t)}\Omega(t),
\qquad \Omega=\Omega_I+i\Omega_Q.
$$

Amplitude noise changes rotation strength, while phase noise rotates the drive axis in the quadrature plane.
Generate the trace before integration so that its values are independent of an adaptive solver's trial steps.
The mock's [drive-noise interface](backend.md#drive_noise_and_gradients) interpolates the supplied samples and evolves that fixed trace alongside any Lindblad terms.

To predict an average experiment, evolve a complete circuit for each realization and then average its output.
A slowly varying offset can affect several gates or wires in the same run, so averaging each gate independently would discard those correlations.
[Hangleiter, Cerfontaine and Bluhm (2021)](https://arxiv.org/abs/2103.02403) discuss their effect on gate sequences.
Refine the sampling grid to resolve each trace and increase the number of realizations to improve the ensemble estimate.
The [training guide](training.md#training_with_noisy_evolution) applies this procedure to losses and gradients.

A trace with finite correlation time cannot generally be replaced by arbitrary T1/T2 values.
When combining traces with collapse operators, assign them to distinct physical contributions to avoid counting the same noise twice.

## Randomness under JAX transformations

Gate errors and shot sampling draw random numbers at runtime. The other noise channels above are deterministic maps on density matrices, even though they represent physical noise.

A key captured as a constant makes a jitted function replay the same noise realization on every call.
This is useful when comparing controls under fixed noise, but it does not produce fresh samples for successive training steps.
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
