"""Device binding: a schedule played on a QiController through qiclib.

qiclib and qicode are GPL-3.0 and come with the optional extra
``jaqsi[qiclib]``; no other jaqsi module imports this one.
"""

from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from jaqsi.hardware.backend import Backend, QubitConfig
from jaqsi.hardware.discriminator import Discriminator, export
from jaqsi.hardware.schedule import PulseEvent, Schedule, Sync, validate
from jaqsi.pulses import leaf_pulse_event

try:
    from qiclib.code import (
        Assign,
        ForRange,
        If,
        Play,
        PlayReadout,
        QiAmplitudeVariable,
        QiCells,
        QiIntVariable,
        QiJob,
        QiPulse,
        QiSample,
        QiStateVariable,
        Recording,
        RotateFrame,
        Shape,
        Wait,
    )
    from qiclib.code import Sync as QiSync
    from qiclib.packages.constants import (
        CONTROLLER_SAMPLE_FREQUENCY_IN_HZ,
        RECORDING_MAX_RAW_SAMPLES,
    )
    from qicode import QiCell
except ImportError as error:
    raise ImportError(
        "The qiclib backend needs qiclib and qicode: install jaqsi[qiclib]."
    ) from error

# Above the ids qicode reserves (up to 0x7FFF) and those of qiclib's ShapeLib.
SHAPE_ID_START = 0x9000
# Device amplitude resolution: qiclib plays int(a * 32767) / 32767.
AMPLITUDE_STEP = 1 / (2**15 - 1)
# Extra bits of a swept amplitude, which accumulates its step in fixed point.
# The step is quantized to the amplitude resolution, so without them its error
# would add up over the samples; with 10, 1024 samples drift by under one step.
FRACTION_BITS = 10

Shapes = Dict[Tuple[float, bytes], Shape]  # (length, samples) -> registered shape


class QiclibBackend(Backend):
    """QiController that plays a schedule as a qiclib job.

    The job is one shot, which the device repeats: every qubit is reset,
    plays its pulses and is read out.  RX and RY become a ``Play`` of the
    envelope sampled on the device grid, RZ a ``RotateFrame`` (a virtual Z)
    and a sync a ``Sync`` of the cells it spans.  Readout and reset read
    their settings from a ``QiSample`` of the qubit configs.

    Attributes:
        controller: Connected ``qiclib.QiController``; ``None`` to only build
            jobs.
        qubits: Config of each qubit, indexed by wire and controller cell.
        active_reset: Reset a qubit by reading it out on the device and
            playing RX(pi) if it is in |1>, which needs
            :meth:`write_state_config` first; otherwise wait the qubit's
            ``reset_time``.
        rz_sign: Sign of the frame rotation of an RZ; set it to -1 if the
            device sign test in ``docs/qiclib.md`` reads 1.
        discriminator: Classifier of the IQ points in :meth:`run`, fitted
            with :func:`~jaqsi.hardware.discriminator.fit`; ``None`` until
            then.
    """

    def __init__(
        self,
        controller: Any,
        qubits: List[QubitConfig],
        active_reset: bool = False,
        rz_sign: int = 1,
    ) -> None:
        """Initialise a qiclib backend.

        Args:
            controller: Connected ``qiclib.QiController``, or ``None``.
            qubits: Config of each qubit, indexed by wire.
            active_reset: Reset actively instead of waiting.
            rz_sign: Sign of the frame rotation of an RZ.
        """
        self.controller = controller
        self.qubits = qubits
        self.active_reset = active_reset
        self.rz_sign = rz_sign
        self.discriminator: Optional[Discriminator] = None

    def job(self, schedule: Schedule) -> QiJob:
        """Build the single-shot qiclib job of *schedule* without running it.

        Args:
            schedule: Schedule to play.

        Returns:
            Job that saves the IQ point of wire ``q`` under the name
            ``str(q)``; :meth:`iq` repeats it for every shot.

        Raises:
            ValueError: If the schedule holds a gate other than the leaves,
                has more qubits than the backend has configs, or a pulse has
                a duration or time scale that is not positive, or a device
                amplitude that is not finite or beyond [-1, 1].
        """
        amplitudes = self._amplitudes(schedule)
        with QiJob() as job:
            cells = QiCells(schedule.n_qubits)
            self._shot(cells, schedule, amplitudes[None], {})
        return job

    def _sweep_job(
        self, schedule: Schedule, amplitudes: np.ndarray, step: np.ndarray
    ) -> QiJob:
        """Build one job that plays *schedule* once per row of *amplitudes*.

        A loop plays one shot per sample, so wire ``q`` saves one IQ point
        per sample under ``str(q)``.  Each pulse whose amplitude changes
        keeps its modulus in a variable of its own, which starts at the
        first sample's and adds *step* after each sample.

        Args:
            schedule: Schedule of the first sample; the others differ only in
                the amplitudes.
            amplitudes: Device amplitude of each item per sample, shape
                ``(samples, len(schedule.items))``.
            step: Change of each item's amplitude modulus per sample.
        """
        scale = 2**FRACTION_BITS
        with QiJob() as job:
            cells = QiCells(schedule.n_qubits)
            swept = {
                int(j): QiAmplitudeVariable(float(abs(amplitudes[0, j]) * scale))
                for j in np.flatnonzero(step)
            }
            with ForRange(QiIntVariable(), 0, len(amplitudes)):
                self._shot(cells, schedule, amplitudes, swept)
                for j, variable in swept.items():
                    Assign(variable, variable + float(step[j] * scale))
        return job

    def _shot(
        self,
        cells: List[QiCell],
        schedule: Schedule,
        amplitudes: np.ndarray,
        swept: Dict[int, Any],
    ) -> None:
        """Reset every qubit, play *schedule* and read out the measured wires.

        A pulse plays at the cell's amplitude variable, assigned just before,
        so pulses of one envelope and phase take one of the cell's 13 pulse
        slots, whatever their angle.  By the binding's convention the
        amplitude is non-negative and its sign goes into the phase: a
        negative amplitude plays its modulus at the carrier phase plus pi,
        since qiclib encodes a negative variable amplitude wrongly.

        Args:
            cells: Cells of the job, one per wire.
            schedule: Schedule to play.
            amplitudes: Device amplitude of each item per sample, shape
                ``(samples, len(schedule.items))``, 0 for RZ and syncs; an
                item keeps its sign over the samples.
            swept: Variable of each item whose amplitude changes between
                samples, holding the modulus times ``2**FRACTION_BITS``.  The
                other items play the first sample's.
        """
        shapes: Shapes = {}
        variables = [QiAmplitudeVariable() for _ in cells]
        for q, (cell, config) in enumerate(zip(cells, self.qubits)):
            if self.active_reset:
                state = QiStateVariable()
                _readout(cell, state_to=state)
                pi = leaf_pulse_event("RX", np.pi, q)
                amplitude = _pulse(pi, config)[3]
                with If(state == 1):
                    Assign(variables[q], abs(amplitude))
                    _play(cell, variables[q], pi, config, shapes, amplitude < 0)
            else:
                Wait(cell, cell["reset_time"])
        # Each command goes to the cells of its wires in the order issued, so
        # the circuit order is each wire's play order.
        for j, item in enumerate(schedule.items):
            if isinstance(item, Sync):
                QiSync(*(cells[q] for q in item.wires))
                continue
            q = item.wires[0]
            if item.gate == "RZ":
                # The RZ evolves p * w * Z: a rotation by 2 * p * w.  qiclib
                # turns RotateFrame(angle) into a phase shift of -angle on later
                # pulses, the virtual Z of McKay et al., PRA 96, 022330 (2017).
                angle = 2 * float(item.envelope_params[0]) * float(item.w)
                RotateFrame(cells[q], self.rz_sign * angle)
                continue
            if j in swept:
                Assign(variables[q], swept[j] >> FRACTION_BITS)
            else:
                Assign(variables[q], float(abs(amplitudes[0, j])))
            negative = amplitudes[:, j].min() < 0
            _play(cells[q], variables[q], item, self.qubits[q], shapes, negative)
        for q in schedule.measured:
            _readout(cells[q], save_to=str(q))

    def _amplitudes(self, schedule: Schedule) -> np.ndarray:
        """Device amplitude of each item of *schedule*, 0 for RZ and syncs.

        Raises:
            ValueError: As :meth:`job`.
        """
        validate(schedule, self.leaves)
        if len(self.qubits) < schedule.n_qubits:
            raise ValueError(
                f"The schedule has {schedule.n_qubits} qubits, but the backend "
                f"only {len(self.qubits)} qubit configs."
            )
        return np.array(
            [
                0.0
                if isinstance(item, Sync) or item.gate == "RZ"
                else _pulse(item, self.qubits[item.wires[0]])[3]
                for item in schedule.items
            ]
        )

    def iq(self, schedule: Schedule, shots: int) -> np.ndarray:
        """Play *schedule* for *shots* shots and record the raw IQ points.

        Args:
            schedule: Schedule to play.
            shots: Number of shots.

        Returns:
            Integer IQ points of shape ``(len(schedule.measured), 2, shots)``.
        """
        return self._acquire(self.job(schedule), schedule.measured, shots)[0]

    def run(self, schedule: Schedule, shots: int) -> np.ndarray:
        """Play *schedule* for *shots* shots and classify each IQ point.

        Raises:
            ValueError: If no discriminator is fitted.
        """
        return self._fitted().labels(self.iq(schedule, shots))

    def run_batch(self, schedules: List[Schedule], shots: int) -> np.ndarray:
        """Play each schedule for *shots* shots, a linear sweep in one job.

        Schedules that differ only in the angles of RX and RY, each pulse's
        amplitude changing by a constant step without changing sign, play in
        one job (see :meth:`_sweep_job`) per 1024 samples, the recordings a
        cell can hold, or per 512 with active reset, whose readout records
        too.  Any other batch plays one job per schedule.

        Raises:
            ValueError: If no discriminator is fitted or a schedule is not
                playable (see :meth:`job`).
        """
        discriminator = self._fitted()
        amplitudes = [self._amplitudes(s) for s in schedules]  # checks them all
        if not _linear(schedules, amplitudes):
            return super().run_batch(schedules, shots)
        a = np.stack(amplitudes)
        step = np.diff(np.abs(a), axis=0).mean(axis=0)
        size = RECORDING_MAX_RAW_SAMPLES // (2 if self.active_reset else 1)
        jobs = [
            self._sweep_job(schedules[0], a[i : i + size], step)
            for i in range(0, len(a), size)
        ]
        measured = schedules[0].measured
        iq = np.concatenate([self._acquire(job, measured, shots) for job in jobs])
        return np.stack([discriminator.labels(x) for x in iq])

    def _acquire(self, job: QiJob, measured: Tuple[int, ...], shots: int) -> np.ndarray:
        """Run *job* for *shots* shots and collect its IQ points.

        Returns:
            Integer IQ points of shape ``(samples, len(measured), 2, shots)``,
            a sample for each point a wire saves per shot.

        Raises:
            ValueError: If *shots* is not a positive integer.
        """
        if not (isinstance(shots, (int, np.integer)) and shots > 0):
            raise ValueError(f"shots must be a positive integer, got {shots!r}.")
        # The device runs the job once per shot.  Each run records at most
        # 1024 times per cell; iqcloud keeps every run's points.
        job.run(
            self.controller,
            sample=_sample(self.qubits),
            data_collection="iqcloud",
            averages=shots,
        )
        # qiclib returns a wire's points as (2, samples, shots), squeezed.
        iq = np.array([job.cells[q].data(str(q)) for q in measured], dtype=np.int32)
        return iq.reshape(len(measured), 2, -1, shots).transpose(2, 0, 1, 3)

    def write_state_config(self, iq: np.ndarray) -> None:
        """Load the discriminator's |0>/|1> line into each cell's state estimation.

        The device labels the readout of an active reset itself, with the
        integer line of :func:`~jaqsi.hardware.discriminator.export`.

        Args:
            iq: IQ points of shape ``(n_qubits, 2, shots)`` holding both |0>
                and |1> shots, e.g. the calibration points.

        Raises:
            ValueError: If no discriminator is fitted.
        """
        for cell, config in zip(self.controller.cell, export(self._fitted(), iq)):
            cell.recording.state_config = config

    def _fitted(self) -> Discriminator:
        """The discriminator, which must be fitted before the device is used."""
        if self.discriminator is None:
            raise ValueError(
                "The backend has no discriminator; fit one with "
                "calibrate_discriminator or assign one to "
                "QiclibBackend.discriminator."
            )
        return self.discriminator


def _pulse(
    event: PulseEvent, config: QubitConfig
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Sample the envelope of an RX or RY pulse on the device grid.

    Returns:
        ``(length, x, samples, amplitude)``: the length in seconds, the
        normalized times at which qiclib samples a shape, the samples
        normalized to a peak modulus of 1, and the device amplitude
        ``amplitude_scale * w * peak``.

    Raises:
        ValueError: If the duration or the time scale is not positive, or
            the amplitude is not finite or beyond qiclib's [-1, 1].
    """
    # Schedules retain JAX controls for simulation; device jobs need host scalars.
    duration = float(event.duration)
    if not (duration > 0 and config.time_scale > 0):
        raise ValueError(
            f"{event.gate} on wire {event.wires} needs a positive duration and "
            f"time scale, got {event.duration} and {config.time_scale}."
        )
    length = duration * config.time_scale
    # The normalized times at which qiclib samples a shape (QiPulse.__call__).
    x = np.arange(0, length, 1 / CONTROLLER_SAMPLE_FREQUENCY_IN_HZ) / length
    t, t_c = x * duration, duration / 2
    p = event.envelope_params
    envelope = np.asarray(event.envelope_fn(p, t, t_c), dtype=complex)
    if event.quadrature_fn is not None:
        # E cos(wt + phi) - Q sin(wt + phi) = Re[(E + iQ) exp(i (wt + phi))],
        # and qiclib plays the real part on I, the imaginary one on Q.
        # Whether the device's IQ mixing keeps the sign of Q is unverified.
        envelope += 1j * np.asarray(event.quadrature_fn(p, t, t_c))
    peak = np.abs(envelope).max() or 1.0  # an all-zero envelope stays zero
    amplitude = config.amplitude_scale * float(event.w) * peak
    if not abs(amplitude) <= 1:  # NaN fails too, from the envelope or the scale
        raise ValueError(
            f"{event.gate}({event.w:.3g}) on wire {event.wires} needs the device "
            f"amplitude {amplitude:.3g}; qiclib plays finite ones in [-1, 1]."
        )
    return length, x, envelope / peak, amplitude


def _play(
    cell: QiCell,
    amplitude: Any,
    event: PulseEvent,
    config: QubitConfig,
    shapes: Shapes,
    negative: bool,
) -> None:
    """Play an RX or RY pulse at the modulus held in *amplitude*.

    The envelope is registered as a shape normalized to a peak of 1; the
    variable *amplitude* carries the angle, the amplitude scale and the peak.
    Equal samples share a shape id, counted up from ``SHAPE_ID_START`` per
    job.

    Args:
        cell: Cell that plays the pulse.
        amplitude: The cell's amplitude variable, assigned the modulus.
        event: RX or RY event.
        config: Config of the cell's qubit.
        shapes: Shapes registered in the job so far.
        negative: Whether the amplitude is negative, so the pulse plays at
            the carrier phase plus pi.
    """
    length, x, samples, _ = _pulse(event, config)
    key = (length, samples.tobytes())
    if key not in shapes:
        # qiclib looks a shape up by its id once, when the job's with block
        # exits, so the next job may register other shapes under it.
        ident = SHAPE_ID_START + len(shapes)
        interpolate = partial(np.interp, xp=x, fp=samples)
        shapes[key] = Shape(f"{ident:#x}", ident, interpolate)
    phase = float(event.carrier_phase) + np.pi * negative
    Play(cell, QiPulse(length, cell["manip_frequency"], amplitude, phase, shapes[key]))


def _structure(schedule: Schedule) -> list:
    """What the device plays of *schedule*, except the angles of RX and RY."""
    return [schedule.n_qubits, schedule.measured] + [
        item.wires
        if isinstance(item, Sync)
        else (
            item.gate,
            item.wires,
            item.envelope_fn,
            item.quadrature_fn,
            np.asarray(item.envelope_params).tobytes(),
            item.duration,
            item.carrier_phase,
            item.w if item.gate == "RZ" else None,
        )
        for item in schedule.items
    ]


def _linear(schedules: List[Schedule], amplitudes: List[np.ndarray]) -> bool:
    """Whether one job can play a batch, given each schedule's amplitudes.

    It can if the schedules differ only in the angles of RX and RY, and each
    pulse's amplitude keeps its sign and lies within half a device step of the
    line from the first sample's to the last.
    """
    first = _structure(schedules[0])
    if len(schedules) < 2 or any(_structure(s) != first for s in schedules[1:]):
        return False
    a = np.stack(amplitudes)
    i = np.arange(len(a))[:, None]
    line = a[0] + i * (a[-1] - a[0]) / (len(a) - 1)
    flips = (a > 0).any(axis=0) & (a < 0).any(axis=0)
    return bool(np.all(np.abs(a - line) < AMPLITUDE_STEP / 2) and not flips.any())


def _readout(cell: QiCell, **recording: Any) -> None:
    """Play the readout pulse of *cell* and record its response.

    Args:
        cell: Cell to read out.
        **recording: ``save_to`` or ``state_to`` of ``Recording``.
    """
    pulse = QiPulse(cell["rec_pulse"], cell["rec_frequency"], cell["rec_amplitude"])
    PlayReadout(cell, pulse)
    Recording(cell, cell["rec_length"], cell["rec_offset"], **recording)


def _sample(qubits: List[QubitConfig]) -> QiSample:
    """Qubit configs under the cell keys the job reads.

    Except for ``rec_amplitude`` and ``reset_time``, the keys are those of
    the gates in ``qiclib.experiment.qicode.collection``.  Frequencies are
    those of the controller's oscillators, below 500 MHz.
    """
    sample = QiSample(len(qubits))
    for cell, config in zip(sample, qubits):
        cell.update(
            manip_frequency=config.drive_frequency,
            rec_pulse=config.readout_length,
            rec_frequency=config.readout_frequency,
            rec_amplitude=config.readout_amplitude,
            rec_length=config.recording_length,
            rec_offset=config.recording_offset,
            reset_time=config.reset_time,
        )
    return sample
