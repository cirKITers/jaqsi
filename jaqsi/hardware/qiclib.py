"""Device binding: a schedule played on a QiController through qiclib.

qiclib and qicode are GPL-3.0 and come with the optional extra
``jaqsi[qiclib]``; no other jaqsi module imports this one.
"""

from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from jaqsi.drawing import collect_pulse_events
from jaqsi.hardware.backend import Backend, QubitConfig
from jaqsi.hardware.discriminator import Discriminator, export
from jaqsi.hardware.schedule import PulseEvent, Schedule, Sync, validate

try:
    from qiclib.code import (
        ForRange,
        If,
        Play,
        PlayReadout,
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
    from qiclib.packages.constants import CONTROLLER_SAMPLE_FREQUENCY_IN_HZ
    from qicode import QiCell
except ImportError as error:
    raise ImportError(
        "The qiclib backend needs qiclib and qicode: install jaqsi[qiclib]."
    ) from error

# Above the ids qicode reserves (up to 0x7FFF) and those of qiclib's ShapeLib.
SHAPE_ID_START = 0x9000

Shapes = Dict[Tuple[float, bytes], Shape]  # (length, samples) -> registered shape


class QiclibBackend(Backend):
    """QiController that plays a schedule as a qiclib job.

    In each shot, which the device loops over, every qubit is reset, plays
    its pulses and is read out.  RX and RY become a ``Play`` of the envelope
    sampled on the device grid, RZ a ``RotateFrame`` (a virtual Z) and a
    sync a ``Sync`` of the cells it spans.  Readout and reset read their
    settings from a ``QiSample`` of the qubit configs.

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

    def job(self, schedule: Schedule, shots: int) -> QiJob:
        """Build the qiclib job of *schedule* without running it.

        Args:
            schedule: Schedule to play.
            shots: Number of shots, looped over on the device.

        Returns:
            Job that saves the IQ point of wire ``q`` under the name
            ``str(q)`` in each shot.

        Raises:
            ValueError: If the schedule holds a gate other than the leaves or
                a pulse exceeds the device amplitude.
        """
        validate(schedule, self.leaves)
        shapes: Shapes = {}
        with QiJob() as job:
            cells = QiCells(schedule.n_qubits)
            with ForRange(QiIntVariable(), 0, shots):
                for cell, config in zip(cells, self.qubits):
                    if self.active_reset:
                        state = QiStateVariable()
                        _readout(cell, state_to=state)
                        with If(state == 1):
                            pi = collect_pulse_events("RX", np.pi, 0)[0]
                            self._play(cell, pi, config, shapes)
                    else:
                        Wait(cell, cell["reset_time"])
                for item in _in_play_order(schedule):
                    if isinstance(item, Sync):
                        QiSync(*(cells[q] for q in item.wires))
                    else:
                        q = item.wires[0]
                        self._play(cells[q], item, self.qubits[q], shapes)
                for q in schedule.measured:
                    _readout(cells[q], save_to=str(q))
        return job

    def _play(
        self, cell: QiCell, event: PulseEvent, config: QubitConfig, shapes: Shapes
    ) -> None:
        """Play an RX or RY pulse, or rotate the frame for an RZ.

        The envelope is sampled once on the device grid and registered as a
        shape normalized to a peak of 1; the pulse amplitude carries the
        angle, the amplitude scale and the peak.  Equal samples share a shape
        id, counted up from ``SHAPE_ID_START`` per job.
        """
        if event.gate == "RZ":
            # The RZ evolves p * w * Z: a rotation by 2 * p * w.  qiclib turns
            # RotateFrame(angle) into a phase shift of -angle on later pulses,
            # the virtual Z of McKay et al., PRA 96, 022330 (2017).
            angle = 2 * float(event.envelope_params[0]) * event.w
            RotateFrame(cell, self.rz_sign * angle)
            return

        length = event.duration * config.time_scale
        # The normalized times at which qiclib samples a shape (QiPulse.__call__).
        x = np.arange(0, length, 1 / CONTROLLER_SAMPLE_FREQUENCY_IN_HZ) / length
        t, t_c = x * event.duration, event.duration / 2
        p = event.envelope_params
        envelope = np.asarray(event.envelope_fn(p, t, t_c), dtype=complex)
        if event.quadrature_fn is not None:
            # E cos(wt + phi) - Q sin(wt + phi) = Re[(E + iQ) exp(i (wt + phi))],
            # and qiclib plays the real part on I, the imaginary one on Q.
            # Whether the device's IQ mixing keeps the sign of Q is unverified.
            envelope += 1j * np.asarray(event.quadrature_fn(p, t, t_c))
        peak = np.abs(envelope).max() or 1.0  # an all-zero envelope stays zero
        amplitude = config.amplitude_scale * event.w * peak
        if abs(amplitude) > 1:
            raise ValueError(
                f"{event.gate}({event.w:.3g}) on wire {event.wires} needs the "
                f"device amplitude {amplitude:.3g}, beyond qiclib's [-1, 1]."
            )

        samples = envelope / peak
        key = (length, samples.tobytes())
        if key not in shapes:
            # qiclib looks a shape up by its id once, when the job's with
            # block exits, so the next job may register other shapes under it.
            ident = SHAPE_ID_START + len(shapes)
            interpolate = partial(np.interp, xp=x, fp=samples)
            shapes[key] = Shape(f"{ident:#x}", ident, interpolate)
        frequency = cell["manip_frequency"]
        pulse = QiPulse(length, frequency, amplitude, event.carrier_phase, shapes[key])
        Play(cell, pulse)

    def iq(self, schedule: Schedule, shots: int) -> np.ndarray:
        """Play *schedule* for *shots* shots and record the raw IQ points.

        Args:
            schedule: Schedule to play.
            shots: Number of shots.

        Returns:
            Integer IQ points of shape ``(len(schedule.measured), 2, shots)``.
        """
        job = self.job(schedule, shots)
        job.run(
            self.controller,
            sample=_sample(self.qubits),
            data_collection="iqcloud",
            averages=1,
        )
        iq = [job.cells[q].data(str(q)) for q in schedule.measured]
        return np.array(iq, dtype=np.int32).reshape(len(iq), 2, shots)

    def run(self, schedule: Schedule, shots: int) -> np.ndarray:
        """Play *schedule* for *shots* shots and classify each IQ point.

        Raises:
            ValueError: If no discriminator is fitted.
        """
        return self._fitted().labels(self.iq(schedule, shots))

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


def _in_play_order(schedule: Schedule) -> Iterator[Union[PulseEvent, Sync]]:
    """Each wire's events in order, and each sync once all its wires reach it.

    Wires are independent between syncs, so each wire plays up to its next
    sync.  Of the syncs then waiting at the wires' heads, the one earliest in
    the circuit heads every wire it spans and goes next.
    """
    queues = {q: list(items) for q, items in schedule.wires.items()}
    while True:
        for queue in queues.values():
            while queue and isinstance(queue[0], PulseEvent):
                yield queue.pop(0)
        heads = [queue[0] for queue in queues.values() if queue]
        if not heads:
            return
        sync = next(s for s in heads if all(queues[w][0] is s for w in s.wires))
        for q in sync.wires:
            queues[q].pop(0)
        yield sync


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
