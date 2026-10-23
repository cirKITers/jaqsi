"""Pulse schedule of a circuit: what a hardware backend plays per wire."""

from dataclasses import dataclass
from typing import Any, Collection, List, Optional, Tuple, Union

from jaqsi.evolution import PendingEvolution
from jaqsi.gateset import Barrier
from jaqsi.noise import KrausChannel
from jaqsi.operations import Operation


@dataclass
class PulseEvent:
    """Single pulse applied to one or more wires.

    Attributes:
        gate: Gate label, e.g. ``"RX"``, ``"CZ"``.
        wires: Target qubit wire(s).
        envelope_fn: Pure envelope function ``(p, t, t_c) -> amplitude``.
        envelope_params: Envelope-shape parameters (excluding ``w`` and ``t``).
        w: Rotation angle passed to the gate.
        duration: Pulse duration (evolution time).
        carrier_phase: Phase offset for the carrier cosine.
        parent: Optional high-level gate name that decomposed into this event.
        quadrature_fn: Quadrature envelope ``(p, t, t_c) -> amplitude`` of a
            two-quadrature envelope (DRAG), ``None`` otherwise.
    """

    gate: str
    wires: List[int]
    envelope_fn: Any  # (p, t, t_c) -> scalar
    envelope_params: Any  # jnp array of envelope shape params
    w: float  # rotation angle
    duration: float  # evolution time
    carrier_phase: float = 0.0  # phi_c in cos(omega_c * t + phi_c)
    parent: Optional[str] = None  # composite gate that owns this pulse
    quadrature_fn: Any = None  # (p, t, t_c) -> scalar, DRAG only


@dataclass
class Sync:
    """Point at which the spanned wires wait for each other (a barrier).

    Attributes:
        wires: Wires to synchronize.
    """

    wires: List[int]


@dataclass
class Schedule:
    """Ordered pulses and syncs, followed by a readout.

    Attributes:
        n_qubits: Number of qubits.
        items: Events and syncs in circuit order, each once.  Restricted to
            one wire, this is the wire's play order.
        measured: Wires read out at the end.
    """

    n_qubits: int
    items: List[Union[PulseEvent, Sync]]
    measured: Tuple[int, ...]


def build_schedule(
    tape: List[Operation], events: List[PulseEvent], n_qubits: int
) -> Schedule:
    """Build the schedule of a circuit from its two recorded tapes.

    Walks the operation tape in order: each leaf pulse gate (RX, RY, RZ, CZ)
    takes the next pulse event, and a barrier becomes a :class:`Sync`.
    The global-phase correction that a composite gate evolves on its own
    (the H gate's) has no pulse and is dropped.

    Args:
        tape: Operations recorded by :func:`~jaqsi.tape.recording`.
        events: Pulse events recorded by :func:`~jaqsi.tape.pulse_recording`
            during the same run.
        n_qubits: Number of qubits.

    Returns:
        Schedule that measures all qubits.

    Raises:
        ValueError: If an operation has no pulse (unitary gate, noise
            channel, evolution of a custom Hamiltonian).
    """
    from jaqsi.pulses import PulseInformation  # deferred: drawing imports this module

    items: List[Union[PulseEvent, Sync]] = []
    pending = iter(events)
    for op in tape:
        if isinstance(op, Barrier):
            item = Sync(list(op.wires))
        elif isinstance(op, PendingEvolution):
            gate = PulseInformation.gate_by_name(op.name)
            if gate is None:
                raise ValueError(f"Evolution {op.name!r} is not a pulse gate.")
            # A composite gate's own evolution is taken to be a global-phase
            # correction and dropped, since a device cannot measure it.  This
            # holds for H, the only one today: RY(π/2)·RZ(π) = −i·H, and its
            # "H" evolution of π/2·I multiplies by i.  A composite gate that
            # evolves a non-trivial Hamiltonian under its own name would be
            # dropped here silently, so such a gate must name a leaf instead.
            if not gate.is_leaf:
                continue
            item = next(pending, None)
            if item is None or item.gate != op.name:
                raise ValueError(f"No pulse event recorded for {op.name!r}.")
        elif isinstance(op, KrausChannel):
            raise ValueError(f"Noise channel {op.name!r} cannot run on hardware.")
        else:
            raise ValueError(
                f"Gate {op.name!r} has no pulse; pass pulse=True to run it on hardware."
            )
        items.append(item)
    return Schedule(n_qubits, items, tuple(range(n_qubits)))


def validate(schedule: Schedule, leaves: Collection[str]) -> None:
    """Check that a backend can play every pulse of a schedule.

    Args:
        schedule: Schedule to check.
        leaves: Leaf gates the backend supports, e.g. ``{"RX", "RY", "RZ"}``.

    Raises:
        ValueError: If an event's gate is not in *leaves*.
    """
    for item in schedule.items:
        if isinstance(item, PulseEvent) and item.gate not in leaves:
            raise ValueError(
                f"Gate {item.gate!r} on wires {item.wires} is not supported; "
                f"supported leaves: {sorted(leaves)}."
            )
