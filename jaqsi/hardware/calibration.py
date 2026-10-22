"""Calibration: fitting the qubit configs of a device to sweeps of its readout.

Each routine plays its schedules on a backend, fits a model to the result and
writes the fit into the backend's programmed configs, so the mock and the
qiclib binding are calibrated alike.  On the device every point of a sweep is
one job.
"""

from typing import List, Optional, Protocol, Tuple

import numpy as np
from scipy.optimize import curve_fit

from jaqsi.gates import Gates
from jaqsi.hardware.backend import QubitConfig, labels_to_result
from jaqsi.hardware.discriminator import Discriminator, fit
from jaqsi.hardware.schedule import Schedule
from jaqsi.script import Script


class Device(Protocol):
    """Backend that calibration can fill, such as the mock or the qiclib binding.

    Attributes:
        qubits: Programmed config of each qubit, indexed by wire.
        discriminator: Classifier of the IQ points in :meth:`run`.
    """

    qubits: List[QubitConfig]
    discriminator: Optional[Discriminator]

    def iq(self, schedule: Schedule, shots: int) -> np.ndarray:
        """Raw IQ points of shape ``(len(schedule.measured), 2, shots)``."""
        ...

    def run(self, schedule: Schedule, shots: int) -> np.ndarray:
        """Labels of shape ``(len(schedule.measured), shots)``."""
        ...


def gaussian(
    f: np.ndarray, c: float, a: float, center: float, width: float
) -> np.ndarray:
    """Line of a frequency sweep, ``c + a * exp(-((f - center) / width)**2 / 2)``.

    ``width`` is the standard deviation; the full width at half maximum is
    ``2 * sqrt(2 * ln 2) * width``, about ``2.355 * width``.
    """
    return c + a * np.exp(-(((f - center) / width) ** 2) / 2)


def rabi(w: np.ndarray, c: float, a: float, k: float) -> np.ndarray:
    """Rabi oscillation of an angle sweep, ``c + a * sin(k * w / 2)**2``."""
    return c + a * np.sin(k * w / 2) ** 2


def calibrate_discriminator(
    backend: Device, shots: int, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit the backend's discriminator on the thermal state and on RX(pi).

    An empty schedule reads every qubit out in its thermal state, and a
    second one plays RX(pi) on every wire, with the default pulse parameters
    and the backend's programmed config.
    :func:`~jaqsi.hardware.discriminator.fit` clusters the IQ points of both
    and names |1> by the cluster that RX(pi) fills.  RX(pi) has to excite
    the qubits noticeably, so each programmed drive frequency must lie within
    its qubit's line, about 30 MHz wide for the default pulse.  On the device
    this takes two jobs.

    Args:
        backend: Backend to calibrate; its ``discriminator`` is replaced.
        shots: Number of shots per schedule.
        seed: Seed of the k-means initialization.

    Returns:
        ``(iq_thermal, iq_pi)``: IQ points of the thermal state and of
        RX(pi), each of shape ``(n_qubits, 2, shots)``, e.g. for
        :func:`~jaqsi.hardware.discriminator.plot_iq`.  Concatenated along
        the shots, they hold plenty of |0> and |1> shots for
        :func:`~jaqsi.hardware.discriminator.export`.
    """
    n_qubits = len(backend.qubits)

    def pi_pulses():
        for q in range(n_qubits):
            Gates.RX(np.pi, wires=q, pulse=True)

    iq_thermal = backend.iq(Script(lambda: None, n_qubits).schedule(), shots)
    iq_pi = backend.iq(Script(pi_pulses, n_qubits).schedule(), shots)
    backend.discriminator = fit(iq_thermal, iq_pi, seed)
    return iq_thermal, iq_pi


def frequency_sweep(
    backend: Device, wire: int, frequencies: np.ndarray, shots: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Find a qubit's resonance and program it as its drive frequency.

    RX(pi) is played at each drive frequency.  It excites the qubit fully
    only on resonance, so P(1) peaks there; the center of a fitted
    :func:`gaussian` becomes ``backend.qubits[wire].drive_frequency``.  If
    the sweep or the fit fails, the drive frequency is left as it was.

    Args:
        backend: Backend to calibrate, with a fitted discriminator.
        wire: Qubit to calibrate.
        frequencies: Drive frequencies in hertz, spanning the whole line.
        shots: Number of shots per frequency.

    Returns:
        ``(p1, popt)``: P(1) per frequency, leaked shots excluded, and the
        fitted ``(c, a, center, width)`` of :func:`gaussian`.
    """
    config = backend.qubits[wire]
    schedule = _rx(backend, wire).schedule(np.pi)
    programmed = config.drive_frequency
    p1 = np.empty(len(frequencies))
    try:
        for i, frequency in enumerate(frequencies):
            config.drive_frequency = float(frequency)
            p1[i] = _p1(backend, schedule, wire, shots)
    finally:
        config.drive_frequency = programmed
    # The sweep spans the line and its baseline, about four widths each side.
    p0 = (p1.min(), np.ptp(p1), frequencies[np.argmax(p1)], np.ptp(frequencies) / 8)
    popt, _ = curve_fit(gaussian, frequencies, p1, p0=p0, bounds=(0, np.inf))
    config.drive_frequency = float(popt[2])
    return p1, popt


def rabi_sweep(
    backend: Device, wire: int, angles: np.ndarray, shots: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Calibrate a qubit's amplitude scale from its Rabi oscillation.

    RX(w) is played for each angle ``w``.  A device whose amplitude scale is
    off rotates by ``k * w`` instead, so a fitted :func:`rabi` divides
    ``backend.qubits[wire].amplitude_scale`` by ``k``.  The fit starts from
    ``k = 1``, a roughly calibrated amplitude scale.

    Args:
        backend: Backend to calibrate, with a fitted discriminator.
        wire: Qubit to calibrate.
        angles: RX angles in radians, spanning at least one period.
        shots: Number of shots per angle.

    Returns:
        ``(p1, popt)``: P(1) per angle, leaked shots excluded, and the fitted
        ``(c, a, k)`` of :func:`rabi`.
    """
    script = _rx(backend, wire)
    p1 = np.array([_p1(backend, script.schedule(w), wire, shots) for w in angles])
    p0 = (p1.min(), np.ptp(p1), 1.0)
    popt, _ = curve_fit(rabi, angles, p1, p0=p0, bounds=(0, np.inf))
    backend.qubits[wire].amplitude_scale /= popt[2]
    return p1, popt


def _rx(backend: Device, wire: int) -> Script:
    """Circuit of an RX pulse with the angle as its argument on *wire*."""

    def circuit(theta):
        Gates.RX(theta, wires=wire, pulse=True)

    return Script(circuit, len(backend.qubits))


def _p1(backend: Device, schedule: Schedule, wire: int, shots: int) -> float:
    """Probability that *wire* reads 1, leaked shots excluded."""
    labels = backend.run(schedule, shots)
    probs, _ = labels_to_result(labels[[wire]], "probs", [])
    return float(probs[1])
