"""Simulated device: jaqsi replays the schedule and IQ readout is synthesized."""

from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Optional

import jax.numpy as jnp
import numpy as np
import equinox as eqx

from jaqsi.gates import Gates, PulseGates
from jaqsi.pulses import PulseEnvelope
from jaqsi.hardware.backend import Backend, QubitConfig
from jaqsi.hardware.discriminator import Discriminator
from jaqsi.hardware.schedule import PulseEvent, Schedule, Sync, validate
from jaqsi.noise import BitFlip
from jaqsi.operations import Hamiltonian
from jaqsi.script import Script


@dataclass
class MockQubit:
    """Physical qubit that the mock simulates behind one wire.

    Nothing here is programmed: calibration only sees it through the readout
    and estimates the programmed :class:`~jaqsi.hardware.backend.QubitConfig`
    from it.  A qubit whose ``frequency`` equals the programmed
    ``drive_frequency``, and whose ``rabi_rate`` is ``1 / (amplitude_scale *
    time_scale)`` of the programmed config, makes a perfectly calibrated
    mock.  Times are in seconds and frequencies in hertz.

    Attributes:
        frequency: Resonance of the qubit, an intermediate frequency like the
            programmed drive frequency.
        rabi_rate: Rabi frequency per unit device amplitude, in radians per
            second: a resonant drive of device amplitude ``A`` rotates the
            qubit at ``rabi_rate * A``.
        t1: Energy relaxation time.
        t2: Dephasing time, at most ``2 * t1``.
        iq_centers: Mean IQ point of |0>, |1> and |2>, shape ``(3, 2)``.
        noise_width: Standard deviation of the IQ noise per quadrature.
        thermal_population: Excited population of the thermal state the
            qubit starts in.
        readout_error: Probability that the readout reports 0 for 1 and
            vice versa, for shots that did not leak.
        leakage: Probability that a shot in |1> reads out as |2>, so the
            thermal state shows hardly any leakage.
    """

    frequency: float
    rabi_rate: float
    t1: float
    t2: float
    iq_centers: np.ndarray
    noise_width: float
    thermal_population: float = 0.0
    readout_error: float = 0.0
    leakage: float = 0.0

    def __post_init__(self):
        if not (self.t1 > 0 and self.t2 > 0 and self.t2 <= 2 * self.t1):
            raise ValueError("Require t1 > 0 and 0 < t2 <= 2*t1.")
        if not 0 <= self.thermal_population <= 1:
            raise ValueError("thermal_population must be in [0, 1].")


class DriveNoise(NamedTuple):
    """One classical noise realization over the complete schedule.

    ``times`` is an increasing physical-time grid in seconds. ``amplitude``
    and ``phase`` have shape ``(n_qubits, len(times))`` and hold fractional
    amplitude noise and phase noise in radians. Linear interpolation is used,
    with constant endpoint values outside the grid. A one-point grid represents
    quasistatic noise. Generate traces before solving; batch complete circuit
    results to average an ensemble without discarding temporal correlations.
    """

    times: jnp.ndarray
    amplitude: jnp.ndarray
    phase: jnp.ndarray


class Mock(Backend):
    """Device that replays a schedule as a noisy jaqsi pulse circuit.

    The mock keeps two descriptions per qubit: the programmed config, which
    calibration fills as on the device, and the simulated physical qubit,
    which calibration has to find.  RX and RY pulses evolve in the frame of
    the programmed drive frequency, detuned from the qubit's resonance, with
    the amplitude scaled by the qubit's Rabi rate times the programmed amplitude
    and time scale (see :func:`_drive`).  Each qubit starts in its thermal state and
    relaxes toward it during every physical pulse.  A shot is a sample of the
    exact probabilities.  If it is in |1>, it leaks to |2> with the leakage
    probability; if it did not leak, its readout is flipped with the readout
    error.  It is drawn as an IQ point around the center of its state.

    Attributes:
        qubits: Programmed config of each qubit, indexed by wire.  The mock
            reads its drive frequency, amplitude scale and time scale.  It
            does not model the readout settings, nor the reset time: every
            shot starts in the thermal state.
        truth: Simulated qubit behind each wire, with everything physical:
            resonance, Rabi rate, relaxation, thermal population and readout.
        rng: Random generator of the shots and the IQ noise.
        discriminator: Classifier of the IQ points in :meth:`run`, fitted
            with :func:`~jaqsi.hardware.discriminator.fit`; ``None`` until
            then.
    """

    # Not modeled: the readout pulse, which the config carries.

    def __init__(
        self,
        qubits: List[QubitConfig],
        truth: List[MockQubit],
        seed: int = 0,
    ) -> None:
        """Initialise a mock device.

        Args:
            qubits: Programmed config of each qubit, indexed by wire.
            truth: Simulated qubit behind each wire.
            seed: Seed of the random generator.
        """
        self.qubits = qubits
        self.truth = truth
        self.rng = np.random.default_rng(seed)
        self.discriminator: Optional[Discriminator] = None

    def _replay(
        self, schedule: Schedule, drive_noise: DriveNoise, prepare: bool
    ) -> None:
        """Record shared Lindblad evolutions on each wire's physical timeline."""
        clocks = [jnp.asarray(0.0) for _ in range(schedule.n_qubits)]
        if prepare:
            for q in range(schedule.n_qubits):
                BitFlip(self.truth[q].thermal_population, wires=q)
        for ev in schedule.items:
            if isinstance(ev, Sync):
                end = jnp.max(jnp.stack([clocks[q] for q in ev.wires]))
                for q in ev.wires:
                    config, true = self.qubits[q], self.truth[q]
                    detuning = 2 * np.pi * (true.frequency - config.drive_frequency)
                    H = Hamiltonian(detuning * config.time_scale / 2 * PulseGates.Z, q)
                    H.evolve(collapse_ops=_collapse_ops(true, config.time_scale))(
                        (end - clocks[q]) / config.time_scale, wires=q
                    )
                    clocks[q] = end
                continue
            q = ev.wires[0]
            if ev.gate == "RZ":
                Gates.RZ(2 * ev.envelope_params[0] * ev.w, wires=q)
                continue
            _drive(ev, self.qubits[q], self.truth[q], clocks[q], drive_noise)
            clocks[q] = clocks[q] + ev.duration * self.qubits[q].time_scale

    def density(
        self,
        schedule: Schedule,
        *,
        drive_noise: Optional[DriveNoise] = None,
        initial_state: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Exact density matrix, differentiable in the schedule's pulse controls.

        Args:
            schedule: Fixed sequence of pulses and syncs, with JAX-valued controls.
            drive_noise: A fixed classical noise realization; ``None`` disables it.
            initial_state: Optional pure statevector, replacing thermal preparation.

        Each wire is measured when its own stream finishes, as on the backend.
        Add a final sync to evolve all wires to a common readout time. The mock
        supports independent local dynamics, including shared classical noise;
        it does not model couplings, readout dynamics or physical leakage.
        """
        validate(schedule, self.leaves)
        n = schedule.n_qubits
        if len(self.qubits) < n or len(self.truth) < n:
            raise ValueError(
                "The mock needs a config and physical qubit for every wire."
            )
        if drive_noise is None:
            drive_noise = DriveNoise(jnp.zeros(1), jnp.zeros((n, 1)), jnp.zeros((n, 1)))
        if (
            drive_noise.times.ndim != 1
            or drive_noise.times.size == 0
            or drive_noise.amplitude.shape != (n, drive_noise.times.size)
            or drive_noise.phase.shape != drive_noise.amplitude.shape
        ):
            raise ValueError(
                "Noise needs times (m,) and amplitude/phase (n_qubits, m)."
            )
        times = eqx.error_if(
            drive_noise.times,
            jnp.any(~jnp.isfinite(drive_noise.times))
            | jnp.any(jnp.diff(drive_noise.times) <= 0),
            "Noise times must be finite and strictly increasing.",
        )
        drive_noise = drive_noise._replace(times=times)
        for q in range(schedule.n_qubits):
            if not self.qubits[q].time_scale > 0:
                raise ValueError("time_scale must be positive.")
        script = Script(
            lambda: self._replay(schedule, drive_noise, initial_state is None), n
        )
        return script.execute("density", initial_state=initial_state)

    def probs(
        self, schedule: Schedule, *, drive_noise: Optional[DriveNoise] = None
    ) -> jnp.ndarray:
        """Exact probabilities of the computational basis states before readout.

        Args:
            schedule: Schedule to replay.
            drive_noise: A fixed classical noise realization; ``None`` disables it.

        Returns:
            Probabilities of shape ``(2**n_qubits,)``, qubit 0 the most
            significant bit.

        Raises:
            ValueError: If the schedule holds a gate other than the leaves.
        """
        return jnp.real(jnp.diag(self.density(schedule, drive_noise=drive_noise)))

    def iq(
        self,
        schedule: Schedule,
        shots: int,
        *,
        drive_noise: Optional[DriveNoise] = None,
    ) -> np.ndarray:
        """Play *schedule* for *shots* shots and record the raw IQ points.

        Args:
            schedule: Schedule to play.
            shots: Number of shots.
            drive_noise: Fixed noise realization shared by these shots.

        Returns:
            IQ points of shape ``(len(schedule.measured), 2, shots)``.
        """
        n = schedule.n_qubits
        probs = np.clip(
            np.asarray(self.probs(schedule, drive_noise=drive_noise), dtype=float),
            0,
            None,
        )
        index = self.rng.choice(2**n, size=shots, p=probs / probs.sum())
        points = []
        for q in schedule.measured:
            qubit = self.truth[q]
            state = (index >> (n - 1 - q)) & 1
            leaked = (state == 1) & (self.rng.random(shots) < qubit.leakage)
            state = state ^ (self.rng.random(shots) < qubit.readout_error)
            state = np.where(leaked, 2, state)
            noise = self.rng.normal(scale=qubit.noise_width, size=(2, shots))
            points.append(qubit.iq_centers[state].T + noise)
        return np.stack(points)

    def run(
        self,
        schedule: Schedule,
        shots: int,
        *,
        drive_noise: Optional[DriveNoise] = None,
    ) -> np.ndarray:
        """Play *schedule* for *shots* shots and classify each IQ point.

        Raises:
            ValueError: If no discriminator is fitted.
        """
        if self.discriminator is None:
            raise ValueError(
                "The mock has no discriminator; fit one with "
                "calibrate_discriminator or assign one to Mock.discriminator."
            )
        return self.discriminator.labels(
            self.iq(schedule, shots, drive_noise=drive_noise)
        )


def _collapse_ops(true: MockQubit, time_scale: float) -> list:
    """Thermal relaxation and pure dephasing in dimensionless pulse time."""
    down = np.array([[0, 1], [0, 0]], dtype=complex)
    gamma1 = time_scale / true.t1
    gamma_phi = time_scale / true.t2 - gamma1 / 2
    return [
        np.sqrt((1 - true.thermal_population) * gamma1) * down,
        np.sqrt(true.thermal_population * gamma1) * down.T,
        np.sqrt(gamma_phi / 2) * PulseGates.Z,
    ]


def _drive(
    ev: PulseEvent,
    config: QubitConfig,
    true: MockQubit,
    start: jnp.ndarray,
    noise: DriveNoise,
) -> None:
    """Record simultaneous drive and dissipation, including moving square edges."""
    ratio = true.rabi_rate * config.amplitude_scale * config.time_scale
    detuning = 2 * np.pi * (true.frequency - config.drive_frequency) * config.time_scale
    X, Y, Z = PulseGates.X, PulseGates.Y, PulseGates.Z
    q = ev.wires[0]
    params = (
        ev.envelope_params,
        ev.duration,
        ev.w,
        ev.carrier_phase,
        start,
        config.time_scale,
        noise.times,
        noise.amplitude[q],
        noise.phase[q],
    )
    segments = [(0.0, ev.duration, ev.envelope_fn)]
    if ev.envelope_fn is PulseEnvelope.square:
        edge = jnp.clip((ev.duration - ev.envelope_params[1]) / 2, 0, ev.duration / 2)
        segments = [
            (0.0, edge, _zero),
            (edge, ev.duration - edge, PulseEnvelope._square_amplitude),
            (ev.duration - edge, ev.duration, _zero),
        ]
    for left, right, envelope in segments:
        H = _coefficient(envelope, ev.quadrature_fn, 0) * Hamiltonian(
            ratio / 2 * X, ev.wires
        )
        H = H + _coefficient(envelope, ev.quadrature_fn, 1) * Hamiltonian(
            ratio / 2 * Y, ev.wires
        )
        H = H + _static * Hamiltonian(detuning / 2 * Z, ev.wires)
        evolve = H.evolve(
            name=ev.gate, collapse_ops=_collapse_ops(true, config.time_scale)
        )
        boundaries = jnp.stack([jnp.asarray(left), jnp.asarray(right)])
        if noise.times.size > 1:
            # Resolve every interpolation knot, including narrow features that
            # an adaptive solver could otherwise step over without detecting.
            knots = jnp.clip((noise.times - start) / config.time_scale, left, right)
            boundaries = jnp.concatenate([boundaries[:1], knots, boundaries[1:]])
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            evolve([params] * H.n_terms, jnp.stack([begin, end]))


def _coefficient(
    envelope: Callable, quadrature: Optional[Callable], axis: int
) -> Callable:
    """Drive-frame coefficient; all varying data are explicit solver arguments."""

    def coefficient(args, t):
        p, T, w, phi, start, scale, times, amplitude_noise, phase_noise = args
        time = start + scale * t
        amplitude = 1 + jnp.interp(time, times, amplitude_noise)
        phase = phi + jnp.interp(time, times, phase_noise)
        E = envelope(p, t, T / 2)
        Q = 0.0 if quadrature is None else quadrature(p, t, T / 2)
        component = (
            E * jnp.cos(phase) - Q * jnp.sin(phase)
            if axis == 0
            else E * jnp.sin(phase) + Q * jnp.cos(phase)
        )
        return w * amplitude * component

    return coefficient


def _zero(p, t, t_c):
    return 0.0


def _static(p, t):
    """Coefficient of a time-independent term."""
    return jnp.ones(())
