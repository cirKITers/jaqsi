# Backends

The [backend notebook](backend.ipynb) follows a circuit from pulse scheduling through readout and calibration to execution on a mock device or a QiController.
This companion explains what its experiments reveal about quantum control and measurement.

## Circuit to schedule

`js.Script(circuit).schedule(...)` records gates created with `pulse=True` and decomposes composite gates into their pulse primitives.
In the opening example, H becomes RZ followed by RY, while `Gates.Barrier` synchronizes the two wires.
The current backends support RX, RY and virtual RZ, but not two-qubit pulses such as CZ.

Wires advance independently between barriers.
In the mock, a wire waiting at a barrier continues evolving under detuning and decay, while virtual RZ has zero duration.
A final barrier is therefore needed to compare all qubits at a common time.

## Mock

`Mock` evolves the schedule with JAQSI's [pulse solver](pulses.md#open_system_pulse_evolution) and synthesizes IQ readout.
It separates the programmed `QubitConfig` from the physical properties in `MockQubit`, so calibration can be tested against known device parameters using only measurements.
The config sets the drive frequency and amplitude and time scales, while the physical model supplies the resonance, Rabi rate, T1/T2, thermal population and readout noise.

The notebook deliberately starts with a 60 MHz drive for a 63 MHz qubit and an amplitude scale of 0.1 instead of 0.08.
The amplitude error alone would make a resonant pulse rotate 1.25 times too far, while detuning also changes its rotation axis.
Correcting these errors is necessary for programmed gate angles to represent the intended operations.
Here `time_scale` converts the recorded duration to seconds, while `amplitude_scale` converts the envelope to device amplitude.
The physical `rabi_rate` then sets the angular rotation rate, so the product `rabi_rate * amplitude_scale * time_scale` must equal one to recover the nominal resonant rotation.

### Readout

`mock.iq` samples the final basis probabilities and draws a Gaussian IQ point around the selected class's center.
Cluster populations carry information about the state, while their spread determines how reliably individual shots can be classified.
An excited shot receives class 2 with probability `leakage`, and non-leaked shots can flip their binary label with `readout_error`.

`calibrate_discriminator` combines a thermal preparation with one followed by RX($\pi$), then fits clusters and assigns new points to their nearest center.
The cluster most populated by the thermal preparation is labelled 0, and the remaining cluster whose population grows most after RX($\pi$) is labelled 1.
This requires a rough initial calibration that already excites the qubit noticeably.
A third cluster is treated as leakage, although on hardware a separate readout cluster alone does not establish physical leakage.

### Frequency and Rabi sweeps

With a discriminator fitted, the notebook corrects the drive through two measurements.
`frequency_sweep` plays RX($\pi$) across drive frequencies and writes the center of an empirical Gaussian fit into `drive_frequency`.
Next, `rabi_sweep` fits the excited population to $c+a\sin^2(k\theta/2)$ and divides `amplitude_scale` by $k$.
The offset $c$ and contrast $a$ accommodate imperfect populations, while $k$ measures the error in the rotation scale.
Frequency comes first because detuning distorts the rotation being calibrated by the Rabi sweep.

Both routines update `mock.qubits`, so subsequent circuits use the fitted settings.
Comparing these with the known resonance and amplitude scale checks whether calibration recovers the device's controls despite imperfect preparation and readout.

### Thermal state and π pulse

Calibration corrects the drive but does not purify the initial state.
For thermal excited population $p_{\rm th}$, an ideal RX($\pi$) swaps the populations and gives $P_1=1-p_{\rm th}$.
The notebook uses $p_{\rm th}=0.15$, so even a perfect population swap reaches only 0.85 before readout errors.
More generally, an ideal RX($\theta$) applied to this mixture gives

$$
\langle Z\rangle=(1-2p_{\rm th})\cos\theta.
$$

The reduced contrast therefore need not indicate an incorrect rotation angle.
It can come from preparation alone, which is why the notebook compares both pure and thermal initial states.

In the IQ plot, the calibrated pulse moves most shots into the excited cluster without removing the cluster spread.
The dotted lines are nearest-center classification boundaries, while the dashed line is the binary boundary produced by `export` for the controller's state estimation.

### Running circuits

Pass the device to `js.Script(circuit, backend=mock)` and supply `shots` to `execute`.
Backends return `probs` or `expval`, with expectation values restricted to observables diagonal in the computational basis, such as Pauli Z.

The notebook varies the Gaussian pulse's amplitude and duration separately.
For a resonant single-quadrature drive, the rotation depends on the pulse area, so scaling amplitude scales the rotation angle.
Changing duration at fixed width changes the lifted Gaussian's shape and area nonlinearly, explaining why the two sweeps differ.
The thermal-start simulation uses `BitFlip(p_th)` before the pulse to reproduce the mock's initial mixture and the resulting reduction in $\langle Z\rangle$ contrast.

Measured values also include finite-shot and classification effects.
`labels_to_result` discards any shot labelled as leakage on a measured wire and normalizes over the retained shots, so its probabilities are conditional on that selection.
It returns the excluded fraction separately, whereas `Script.execute` returns only the probabilities or expectation values.
Because leakage is sampled only from excited shots in this model, excluding it can shift the population estimate as well as reduce the sample count.

### Accumulation over consecutive gates

The next experiment starts in $|0\rangle$ and compares repeated RX($\pi$) pulses with finite and infinite T1/T2, without classical drive noise.
Ideally, $P_1$ alternates between zero and one as the gate count increases.
With decay, this contrast decreases because every gate receives the preceding gate's state and the total evolution time grows with circuit depth.

The executable cell uses T1 = 500 ns and T2 = 100 ns to make this visible over a short sequence.
T1 relaxes populations toward thermal equilibrium, while T2 describes coherence loss during the rotations.
The mock evolves these processes together with the drive using the Lindblad equation.
Even though each pulse has the same duration and decay rates, its effect depends on the incoming state, so later points cannot be obtained by repeating independent single-gate measurements.
This shows why calibrating an isolated gate does not eliminate errors accumulated during a longer computation.
See [relaxation and dephasing](noise.md#relaxation_and_dephasing) for the model.

### Drive noise and gradients

The final mock experiment disables thermal preparation and T1/T2 decay to isolate classical drive fluctuations.
`DriveNoise` specifies fractional `amplitude` changes and `phase` offsets in radians on a `times` grid measured in seconds from the circuit's start.
The two noise arrays have shape `(n_qubits, len(times))`, with linear interpolation between samples and constant values outside the grid.
For the complex drive $\Omega=\Omega_I+i\Omega_Q$, the applied signal is

$$
\Omega_{\rm noisy}(t)=[1+\epsilon_A(t)]e^{i\epsilon_\phi(t)}\Omega(t).
$$

Thus amplitude noise changes the drive strength, while phase noise rotates its axis in the quadrature plane.

The notebook sweeps the RX angle at fixed pulse duration and reuses the same noise trace throughout.
`mock.probs` and `mock.density` return exact pre-readout results and support `jax.grad`, `jit` and `vmap` without shots or a discriminator.
This differentiable interface is separate from sampled backend execution through `Script.execute`.
In particular, changing the angle here changes the control amplitude without increasing elapsed time, unlike the preceding gate-count experiment.

The plot pairs $\langle Z\rangle$ with its control derivative.
For excited-state preparation, the loss is $L=1-P_1=(1+\langle Z\rangle)/2$, so half the plotted derivative gives the direction and sensitivity needed to adjust the pulse angle.
For robustness across noise realizations, average losses over complete runs while preserving each trace's time correlations, as described in [training with noisy evolution](training.md#training_with_noisy_evolution).

The mock models local two-level dynamics and synthetic readout labels.
It does not simulate qubit couplings, physical population of higher levels, reset transients or readout dynamics, so `reset_time` and readout settings do not change its state evolution.

## QiController

With the `jaqsi[qiclib]` extra installed, `QiclibBackend` translates the same schedules into controller jobs.
The notebook first builds `job(schedule)` with no connected controller, exposing the reset, pulse, synchronization, frame-rotation and readout commands for inspection.
Each wire needs its own qubit config.

On a connected device, the same discriminator calibration and frequency and Rabi sweeps operate on measured IQ data.
`write_state_config` loads the exported binary classifier for optional active reset, which measures the qubit and conditionally applies RX($\pi$).
The notebook leaves device execution disabled by default.
This shared workflow lets the mock exercise calibration and analysis before applying them to hardware.
