"""Simulated device: jaqsi replays the schedule and IQ readout is synthesized."""

from dataclasses import dataclass
from typing import Callable, List, Optional

import jax.numpy as jnp
import numpy as np

from jaqsi.gates import Gates, PulseGates
from jaqsi.hardware.backend import Backend, QubitConfig
from jaqsi.hardware.discriminator import Discriminator
from jaqsi.hardware.schedule import PulseEvent, Schedule, validate
from jaqsi.noise import BitFlip, ThermalRelaxationError
from jaqsi.operations import Hamiltonian
from jaqsi.script import Script


@dataclass
class MockQubit:
    """Physical qubit that the mock simulates behind one wire.

    Nothing here is programmed: calibration only sees it through the readout
    and estimates the programmed :class:`~jaqsi.hardware.backend.QubitConfig`
    from it.  A qubit whose ``frequency`` and ``amplitude_scale`` equal the
    programmed ``drive_frequency`` and ``amplitude_scale`` makes a perfectly
    calibrated mock.  Times are in seconds and frequencies in hertz.

    Attributes:
        frequency: Resonance of the qubit, an intermediate frequency like the
            programmed drive frequency.
        amplitude_scale: True amplitude scale, the device amplitude per jaqsi
            amplitude at which the qubit rotates as jaqsi's pulse gates do.
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
    amplitude_scale: float
    t1: float
    t2: float
    iq_centers: np.ndarray
    noise_width: float
    thermal_population: float = 0.0
    readout_error: float = 0.0
    leakage: float = 0.0


class Mock(Backend):
    """Device that replays a schedule as a noisy jaqsi pulse circuit.

    The mock keeps two descriptions per qubit: the programmed config, which
    calibration fills as on the device, and the simulated physical qubit,
    which calibration has to find.  RX and RY pulses evolve in the frame of
    the programmed drive frequency, detuned from the qubit's resonance, with
    the amplitude scaled by the ratio of the programmed to the true amplitude
    scale (see :func:`_drive`).  Each qubit starts in its thermal state and
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
            resonance, true amplitude scale, relaxation, thermal population
            and readout.
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

    def _replay(self, schedule: Schedule) -> None:
        """Apply the schedule's pulses with thermal relaxation, wire by wire.

        All leaves are single-qubit gates, so the wires commute.
        """
        for q, items in schedule.wires.items():
            programmed, true = self.qubits[q], self.truth[q]
            ratio = programmed.amplitude_scale / true.amplitude_scale
            # Qubit minus drive frequency, in radians per jaqsi time unit.
            offset = true.frequency - programmed.drive_frequency
            detuning = 2 * np.pi * offset * programmed.time_scale
            BitFlip(true.thermal_population, wires=q)  # |0> -> thermal state
            # Not modeled: relaxation and precession of a qubit idling at a Sync.
            for ev in items:
                if not isinstance(ev, PulseEvent):
                    continue
                if ev.gate == "RZ":
                    # A virtual Z is a frame change and takes no time.
                    Gates.RZ(ev.w, wires=q, pulse=True, pulse_params=ev.envelope_params)
                    continue
                _drive(ev, ratio, detuning)
                ThermalRelaxationError(
                    true.thermal_population,
                    true.t1,
                    true.t2,
                    ev.duration * programmed.time_scale,
                    wires=q,
                )

    def probs(self, schedule: Schedule) -> jnp.ndarray:
        """Exact probabilities of the computational basis states before readout.

        Args:
            schedule: Schedule to replay.

        Returns:
            Probabilities of shape ``(2**n_qubits,)``, qubit 0 the most
            significant bit.

        Raises:
            ValueError: If the schedule holds a gate other than the leaves.
        """
        validate(schedule, self.leaves)  # the replay plays single-qubit leaves
        script = Script(lambda: self._replay(schedule), schedule.n_qubits)
        return script.execute("probs")

    def iq(self, schedule: Schedule, shots: int) -> np.ndarray:
        """Play *schedule* for *shots* shots and record the raw IQ points.

        Args:
            schedule: Schedule to play.
            shots: Number of shots.

        Returns:
            IQ points of shape ``(len(schedule.measured), 2, shots)``.
        """
        n = schedule.n_qubits
        probs = np.clip(np.asarray(self.probs(schedule), dtype=float), 0, None)
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

    def run(self, schedule: Schedule, shots: int) -> np.ndarray:
        """Play *schedule* for *shots* shots and classify each IQ point.

        Raises:
            ValueError: If no discriminator is fitted.
        """
        if self.discriminator is None:
            raise ValueError(
                "The mock has no discriminator; fit one with "
                "calibrate_discriminator or assign one to Mock.discriminator."
            )
        return self.discriminator.labels(self.iq(schedule, shots))


def _drive(ev: PulseEvent, ratio: float, detuning: float) -> None:
    """Evolve an RX or RY pulse in the frame of the drive.

    In the frame that rotates with the drive at ``ω_d``, the RWA leaves

        H(t) = r·w/2 · [(E cos φ − Q sin φ)·X + (E sin φ + Q cos φ)·Y] + Δ/2 · Z

    with the event's envelope ``E``, quadrature ``Q``, angle ``w`` and
    carrier phase ``φ``.  With ``r = 1`` and ``Δ = 0`` this is the RWA of
    :meth:`~jaqsi.pulses.PulseEnvelope.build_coeff_fns`, jaqsi's RX for
    ``φ = 0`` and RY for ``φ = π/2``.  jaqsi's qubit Hamiltonian is
    ``ω_q/2 · Z``, so the drive frame leaves ``(ω_q − ω_d)/2 · Z``:
    ``Δ = ω_q − ω_d`` is positive for a drive below the qubit's resonance.

    Args:
        ev: RX or RY event.
        ratio: Amplitude ratio ``r``, the programmed over the true amplitude
            scale.
        detuning: ``Δ`` in radians per jaqsi time unit.
    """
    X, Y, Z = PulseGates.X, PulseGates.Y, PulseGates.Z
    phi = ev.carrier_phase
    in_phase = ratio / 2 * (np.cos(phi) * X + np.sin(phi) * Y)
    H = _envelope(ev.envelope_fn) * Hamiltonian(in_phase, ev.wires)
    H = H + _static * Hamiltonian(detuning / 2 * Z, ev.wires)
    if ev.quadrature_fn is not None:
        quadrature = ratio / 2 * (np.cos(phi) * Y - np.sin(phi) * X)
        H = H + _envelope(ev.quadrature_fn) * Hamiltonian(quadrature, ev.wires)
    # Packed as in jaqsi's pulse gates: [envelope params..., T, w].
    params = np.append(ev.envelope_params, [ev.duration, ev.w])
    H.evolve(name=ev.gate)([params] * H.n_terms, ev.duration)


def _envelope(fn: Callable) -> Callable:
    """Coefficient ``w · fn(p, t, T/2)`` of an envelope, ``p = [..., T, w]``."""

    def coefficient(p, t):
        return p[-1] * fn(p, t, p[-2] / 2)

    return coefficient


def _static(p, t):
    """Coefficient of a time-independent term."""
    return jnp.ones(())
