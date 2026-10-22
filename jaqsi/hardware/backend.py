"""Contract of a hardware backend, its qubit config and its shot readout."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from jaqsi.hardware.schedule import Schedule
from jaqsi.operations import Operation


@dataclass
class QubitConfig:
    """Settings of one qubit as programmed on a backend.

    Calibration fills these and every backend plays with them: the qiclib
    binding writes them to the controller, the mock reads the drive
    frequency, amplitude scale and time scale.  How the qubit itself behaves
    (its resonance, relaxation and readout) is not programmed; the mock
    simulates it from a :class:`~jaqsi.hardware.mock.MockQubit`.

    Times are in seconds and frequencies in hertz.  jaqsi is dimensionless,
    and under the RWA a rotation depends only on the pulse area, so a time
    and an amplitude scale bridge its units to the device.

    Attributes:
        drive_frequency: Frequency of the drive oscillator, the intermediate
            frequency of the controller cell (below 500 MHz on a QiController).
        amplitude_scale: Device amplitude per jaqsi amplitude.
        time_scale: Seconds per jaqsi time unit.
        readout_frequency: Intermediate frequency of the readout pulse, like
            the drive frequency.
        readout_amplitude: Device amplitude of the readout pulse.
        readout_length: Duration of the readout pulse.
        recording_length: Duration of the recording window.
        recording_offset: Delay of the recording window after the readout
            pulse starts.
        reset_time: Wait of the passive reset before each shot, long enough
            for the qubit to relax to its thermal state (several T1).
    """

    drive_frequency: float
    amplitude_scale: float
    time_scale: float
    readout_frequency: float
    readout_amplitude: float
    readout_length: float
    recording_length: float
    recording_offset: float
    reset_time: float


class Backend(ABC):
    """Device that plays a schedule and reads out every shot.

    Attributes:
        leaves: Leaf pulse gates the device can play; CZ needs a
            device-specific two-qubit primitive and is not supported yet.
    """

    leaves: FrozenSet[str] = frozenset({"RX", "RY", "RZ"})

    @abstractmethod
    def run(self, schedule: Schedule, shots: int) -> np.ndarray:
        """Play *schedule* for *shots* shots and classify each readout.

        Args:
            schedule: Schedule to play.
            shots: Number of shots.

        Returns:
            Integer labels of shape ``(len(schedule.measured), shots)``, one
            row per measured wire: 0, 1, or 2 for a leakage state.
        """


def labels_to_result(
    labels: np.ndarray, type: str, obs: Optional[List[Operation]] = None
) -> Tuple[jnp.ndarray, float]:
    """Turn shot labels into probabilities or expectation values.

    Shots with a leakage label on any wire are excluded.

    Args:
        labels: Labels from :meth:`Backend.run`, row ``q`` for qubit ``q``
            (a schedule measures every qubit).
        type: ``"probs"`` or ``"expval"``.
        obs: Diagonal observables, needed only when *type* is ``"expval"``.

    Returns:
        ``(result, leakage_fraction)``, *result* of the same shape as the
        simulator's shot mode: ``(2**n_qubits,)`` for ``"probs"`` with qubit 0
        as the most significant bit, ``(len(obs),)`` for ``"expval"``.

    Raises:
        ValueError: If every shot leaked, an observable is not diagonal or
            *type* is neither ``"probs"`` nor ``"expval"``.
    """
    n_qubits, shots = labels.shape
    kept = labels[:, (labels < 2).all(axis=0)]
    leakage = 1 - kept.shape[1] / shots
    if kept.shape[1] == 0:
        raise ValueError(f"All {shots} shots leaked; no counts remain.")

    index = 2 ** np.arange(n_qubits - 1, -1, -1) @ kept
    probs = jnp.bincount(index, length=2**n_qubits) / kept.shape[1]

    if type == "probs":
        return probs, leakage

    if type == "expval":
        results = []
        for ob in obs or []:
            O_mat = ob.lifted_matrix(n_qubits)
            diag = jnp.diag(O_mat)
            if not jnp.allclose(O_mat, jnp.diag(diag)):
                raise ValueError(
                    f"Observable {ob.name!r} is not diagonal; "
                    "a backend measures in the computational basis only."
                )
            results.append(jnp.real(jnp.dot(diag, probs)))
        return jnp.array(results), leakage

    raise ValueError(f"A backend returns 'probs' or 'expval', got {type!r}.")
