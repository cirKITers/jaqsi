# Backends

JAQSI supports simulation of circuits using backend devices.
At the moment, a mock backend is available alongside preliminary support for [qiclib](#qicontroller).
This file serves as a supplementary explanation for the concepts explained in [backend notebook](backend.ipynb).

Let's start by creating a circuit with `pulse=True` for the gates (see documentation on [Pulses](pulses.md) in JAQSI).

```python
def circuit(theta):
    Gates.RX(theta, wires=0, pulse=True)
    ...
```

The pulse schedule of a script can be obtained through `js.Script(circuit).schedule(...)`.
Note that the current backends only support RX, RY and virtual RZ for now (and of course any other gate that can be composed from these basis gates).
Gates requiring a two-qubit pulse, such as CZ, are not supported yet.

To execute the schedule on a backend, pass the device as `backend=...` when creating the `Script`.
The backend returns one state label per shot and measured qubit, which `Script` turns into probabilities or expectation values.
Naturally, the `shots` argument in `Script.execute` becomes mandatory now and only `expval` and `probs` execution modes are available.
However, as we'll see a bit further down, the mock backend also supports differentiable outputs, meaning that training of a circuit is still possible.

## Mock

Before running a calibration on hardware, we want to check whether it can recover the correct settings from imperfect measurements.
`Mock` provides a device whose physical properties are known, while the calibration routine sees only its readout.
It uses JAQSI's [pulse solver](pulses.md#open_system_pulse_evolution) to evolve the qubits and then synthesizes IQ measurements, allowing for a differentiable simulation.

To distinguish what we program from how the device behaves, the mock keeps two descriptions of each qubit:

| Object        | Meaning                                                                                              |
| ------------- | ---------------------------------------------------------------------------------------------------- |
| `QubitConfig` | Settings updated through calibration: drive frequency, amplitude and time scales, and readout/reset. |
| `MockQubit`   | Physical properties: resonance, Rabi rate, T1/T2, thermal population and readout noise.              |

For a resonant drive with one quadrature, the rotation angle depends on the area under its angular rotation rate $\Omega(t)$:

$$
\theta_{\rm actual}=\int_{t_0}^{t_1}\Omega(t)\,dt.
$$

The programmed `amplitude_scale` converts the recorded envelope, including the gate's angle factor, to device amplitude.
The physical `rabi_rate` then converts that amplitude to $\Omega(t)$ in radians per second.
Similarly, `time_scale` converts a recorded duration to seconds.
An incorrect amplitude scale therefore produces the wrong rotation even if the pulse shape is correct.
If the programmed drive frequency also misses the physical resonance, the pulse is detuned by $\Delta=2\pi(f_{\rm qubit}-f_{\rm drive})$ in radians per second, which changes the rotation axis as well.
This is why the notebook first finds the resonance with a frequency sweep and then corrects the amplitude scale with a Rabi sweep.

### State preparation and evolution

Notably, correcting the drive does not remove the device's interaction with its environment.
The mock accounts for this both in state preparation and during evolution, i.e. if `thermal_population` is $p_{\rm th}$, each qubit starts in the mixture

$$
\rho_{\rm th}=(1-p_{\rm th})|0\rangle\langle0|
             +p_{\rm th}|1\rangle\langle1|.
$$

For $p_{\rm th}>0$, even an ideal RX($\pi$) pulse only swaps those populations, leaving $P_1=1-p_{\rm th}$ instead of a pure excited state.
During the pulse, T1 additionally relaxes populations toward this thermal equilibrium, while T2 describes the loss of coherence between $|0\rangle$ and $|1\rangle$.
The mock evolves these processes together with the drive and detuning using the Lindblad equation.
We refer to our [noise guide](noise.md#relaxation_and_dephasing) for details on the decay laws and their collapse operators.
The rates are constant, with `t1 > 0` and `0 < t2 <= 2*t1`.
Setting both times to infinity removes decay.

Because the rates act throughout the evolution, their effect depends on the physical duration and on the states visited during a pulse.
A fresh call to `mock.probs` or `mock.density` prepares a fresh initial state and evolves the entire schedule with these same rates.
Within that schedule, each gate receives the preceding gate's output, so errors can accumulate across the sequence which is shown in the following.

To see this accumulation separately from drive miscalibration, consider repeated RX($\pi$) pulses starting in $|0\rangle$.
Ideally, after $n$ pulses,

$$
P_1(n)=\sin^2\!\left(\frac{n\pi}{2}\right),
\qquad t_{\rm total}=n\tau,
$$

where $\tau$ is one pulse's duration.
The population therefore alternates between zero and one, while the time spent evolving increases with every gate.
The notebook compares the same device with and without decay, omitting classical drive noise for this experiment.
It also starts in $|0\rangle$ and shortens T1/T2 from 50 microseconds to 80/60 nanoseconds to make their effect visible within roughly 35 ns pulses.
These choices isolate decay during evolution from reduced contrast caused by thermal preparation.

![Population after consecutive RX gates](figures/backend_gate_sequence_light.png#center#only-light)
![Population after consecutive RX gates](figures/backend_gate_sequence_dark.png#center#only-dark)

With finite T1/T2, the separation between odd and even gate counts shrinks as the sequence grows.
As the drive keeps supplying excitation, the population approaches a repeated-pulse steady cycle.

The same timing model also applies when wires wait for each other.
At a barrier, a wire that finishes early continues evolving with detuning and decay until the others arrive.
Virtual RZ gates have zero modeled duration.
Since wires otherwise finish at their own times, a final `Gates.Barrier` is needed when all qubits should be observed at a common time.

### Drive noise and gradients

T1/T2 describes relaxation and dephasing, but the control signal itself can also fluctuate.
To study how a pulse responds to those fluctuations, `DriveNoise` supplies one realization of amplitude and phase noise over the complete schedule.
Writing the nominal quadratures as $\Omega(t)=\Omega_I(t)+i\Omega_Q(t)$, the mock applies

$$
\Omega_{\rm noisy}(t)=[1+\epsilon_A(t)]e^{i\epsilon_\phi(t)}\Omega(t).
$$

Thus `amplitude` stores the fractional fluctuation $\epsilon_A$, while `phase` stores $\epsilon_\phi$ in radians.
For example, an amplitude value of `0.1` makes the drive ten percent stronger at that time.
The two arrays then have shape `(n_qubits, len(times))` where each row belongs to a wire and each column to a supplied time, measured in seconds from the start of the circuit.

The trace contains values only at those discrete times, whereas the pulse solver needs a drive value whenever it evaluates the evolution between them.
Linear interpolation defines those missing values as between amplitude offsets of `0.0` and `0.1`, the offset halfway through is `0.05`.
Before the first sample and after the last, the nearest endpoint value is retained.
A single sample therefore models a constant offset throughout the run.
The solver splits integration at the sample times so it can resolve short features in the supplied trace.

For pulse optimization, we want to know how these dynamics change an objective when we change a control.
Calling `mock.density(...)` returns the final density matrix $\rho$ before readout, and `mock.probs(...)` returns its diagonal.
For one qubit, $P_1=\langle1|\rho|1\rangle$ is its excited-state probability.
These exact outputs include the modeled physical noise without drawing measurement shots, which makes them suitable for differentiation with `jax.grad`.
A pure `initial_state` can also be passed to `density` when a probe state should replace thermal preparation.

The notebook illustrates this with a single pulse whose programmed RX angle $\theta$ scales its amplitude while its duration stays fixed.
For one qubit, $\langle Z\rangle=P_0-P_1=1-2P_1$, so an ideal, calibrated RX starting in $|0\rangle$ gives $\langle Z\rangle=\cos\theta$.
The plot overlays the noisy expectation value on the left axis and its gradient $\partial\langle Z\rangle/\partial\theta$ on the right, using the same noise realization throughout the sweep.
If the goal is excited-state preparation, the loss $L=1-P_1=(1+\langle Z\rangle)/2$ has derivative $\tfrac12\partial\langle Z\rangle/\partial\theta$, so the gradient indicates how to adjust the control.
This amplitude sweep holds elapsed time fixed, unlike the preceding gate-count experiment.

![Noisy pulse Z expectation and its control gradient](figures/backend_pulse_noise_light.png#center#only-light)
![Noisy pulse Z expectation and its control gradient](figures/backend_pulse_noise_dark.png#center#only-dark)

Although every point has the same duration, the state follows a different path at each amplitude.
With no drive, $|0\rangle$ remains unchanged in this example as during a nominal $2\pi$ rotation, it passes through excited states and superpositions, where relaxation and dephasing can prevent a perfect return.
Consequently, fixed decay rates can change both peaks and minima without any population leaving the two-level system.

For that derivative to compare the same experiment at nearby controls, keep the sequence and noise realization fixed.
To optimize performance over many realizations, use `jax.vmap` over complete circuit runs and average their losses.
Using absolute time across each run preserves fluctuations shared by consecutive gates or different wires while averaging independently at each gate would discard those correlations.
The [noise guide](noise.md#classical_drive_noise) discusses this distinction, while [training](training.md#training_with_noisy_evolution) explains objectives beyond state preparation.

### Readout and calibration

Exact probabilities let us examine the simulated state directly, but a device experiment must infer populations from repeated measurements.
The notebook therefore also exercises a readout path as `mock.iq(schedule, shots=N)` samples the basis probabilities and produces one pair of quadratures $(I,Q)$ per shot and measured qubit.
After selecting a readout class $s$, its IQ point follows as

$$
\mathbf r\mid s\sim\mathcal N(\boldsymbol\mu_s,\sigma^2 I_2),
\qquad \mathbf r=(I,Q),
$$

where $\boldsymbol\mu_s$ is the corresponding `iq_centers` entry, $\sigma$ is `noise_width` and $I_2$ is the two-dimensional identity matrix.
Repeated shots consequently form clusters, whose separation determines how reliably the readout classes can be distinguished.
Before drawing the IQ point, an excited shot can receive class 2 with probability `leakage`, and non-leaked shots can flip their binary class with `readout_error`.

The cluster centers are known inside the mock, but a calibration routine must estimate them from data.
`calibrate_discriminator` therefore measures both a thermal preparation and a preparation followed by RX($\pi$).
The first mainly populates the ground-state cluster, while the second increases the excited-state cluster's population and makes it identifiable.
Fitting both datasets gives a discriminator that assigns new IQ points to their nearest center.

![Synthetic IQ readout before and after a pi pulse](figures/backend_readout_light.png#center#only-light)
![Synthetic IQ readout before and after a pi pulse](figures/backend_readout_dark.png#center#only-dark)

In this figure, the change in cluster populations shows the effect of the pulse while their spread comes from the IQ noise.
The dotted lines mark the fitted classifier's boundaries, while the dashed line is the binary boundary exported for the controller's state estimation.
With this discriminator fitted, `mock.run(...)` returns state labels, and frequency and Rabi sweeps can estimate populations from those labels to correct the drive settings.
This connects readout calibration to the drive calibration introduced above.

The sampled result is not always the same quantity as the exact probability.
`labels_to_result` excludes shots with a leakage label on any wire, so its probabilities are conditional on the retained shots as it returns the excluded fraction as well.
`Script.execute` returns only the resulting probabilities or expectation values.
To inspect leakage, use `run` and `labels_to_result` directly, as in the notebook.
If a `drive_noise` trace is passed to `iq` or `run`, that one realization is shared by all shots in the call.

This sampled path tests the effects of finite shots and classification errors, whereas the exact path provides pulse-control gradients without differentiating through random draws or discrete labels.
Both use the same local two-level evolution.
The current mock does not simulate qubit couplings, reset transients, the readout pulse or measurement backaction, so readout settings and `reset_time` do not alter that evolution. Studying physical leakage suppression by DRAG would also require population dynamics in additional levels.