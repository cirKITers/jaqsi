"""jaqsi — just another quantum simulator.

A gate- and pulse-level quantum circuit simulator built on JAX.  Circuits are
plain Python functions that record :class:`~jaqsi.operations.Operation` objects
onto a tape; :class:`~jaqsi.script.Script` compiles and executes them, routing
between state-vector and density-matrix simulation automatically based on
whether noise channels are present.

Typical use::

    import jaqsi

    def circuit(theta):
        jaqsi.Gates.RX(theta[0], wires=0)
        jaqsi.Gates.CX(wires=[0, 1])

    script = jaqsi.Script(circuit, n_qubits=2)
    script.execute(type="expval", obs=[jaqsi.PauliZ(wires=0)], args=(theta,))

Time evolution is reached through the Hamiltonian objects themselves::

    H = jaqsi.Hamiltonian(matrix, wires=0)      # static -> Hermitian
    H_t = coeff_fn * H                          # time-dependent
    H_t.evolve(name="RX")([params], t)          # gate factory
"""

from jaqsi.script import Script, make_hashable
from jaqsi.evolution import Evolution
from jaqsi.operations import (
    Operation,
    Hamiltonian,
    Hermitian,
    ParametrizedHamiltonian,
    build_parity_observable,
    cdtype,
    # Gate zoo (the full set stays reachable via ``jaqsi.operations``).
    Id,
    PauliX,
    PauliY,
    PauliZ,
    H,
    S,
    SWAP,
    RX,
    RY,
    RZ,
    CX,
    CY,
    CZ,
    CSWAP,
    DiagonalQubitUnitary,
    PauliRot,
    Barrier,
    # Noise channels.
    KrausChannel,
    BitFlip,
    PhaseFlip,
    DepolarizingChannel,
    AmplitudeDamping,
    PhaseDamping,
    ThermalRelaxationError,
    QubitChannel,
    # Symbolic Pauli algebra.
    PauliWord,
    pauli_decompose,
    pauli_string_from_operation,
    prod,
    state_expectation,
    evolve_pauli_with_clifford,
)
from jaqsi.tape import (
    recording,
    active_tape,
    pulse_recording,
    active_pulse_tape,
    copy_to_tape,
    shift_and_append,
)
from jaqsi.math import (
    partial_trace,
    marginalize_probs,
    fidelity,
    trace_distance,
    phase_difference,
    logm_v,
    quantum_fisher_information,
    fubini_study_metric,
)
from jaqsi.gates import Gates
from jaqsi.unitary import UnitaryGates
from jaqsi.pulses import (
    PulseGates,
    PulseParams,
    PulseEnvelope,
    PulseInformation,
    PulseParamManager,
)
from jaqsi.utils import safe_random_split

from jaqsi import (  # noqa: F401  (submodule access: ``from jaqsi import operations``)
    operations,
    math,
    simulation,
    memory,
    drawing,
    tape,
    gates,
    unitary,
    pulses,
    evolution,
    script,
)

__all__ = [
    # Execution
    "Script",
    "Evolution",
    "make_hashable",
    # Operations and Hamiltonians
    "Operation",
    "Hamiltonian",
    "Hermitian",
    "ParametrizedHamiltonian",
    "build_parity_observable",
    "cdtype",
    "Id",
    "PauliX",
    "PauliY",
    "PauliZ",
    "H",
    "S",
    "SWAP",
    "RX",
    "RY",
    "RZ",
    "CX",
    "CY",
    "CZ",
    "CSWAP",
    "DiagonalQubitUnitary",
    "PauliRot",
    "Barrier",
    # Noise
    "KrausChannel",
    "BitFlip",
    "PhaseFlip",
    "DepolarizingChannel",
    "AmplitudeDamping",
    "PhaseDamping",
    "ThermalRelaxationError",
    "QubitChannel",
    # Symbolic Pauli algebra
    "PauliWord",
    "pauli_decompose",
    "pauli_string_from_operation",
    "prod",
    "state_expectation",
    "evolve_pauli_with_clifford",
    # Tape
    "recording",
    "active_tape",
    "pulse_recording",
    "active_pulse_tape",
    "copy_to_tape",
    "shift_and_append",
    # Quantum-info math
    "partial_trace",
    "marginalize_probs",
    "fidelity",
    "trace_distance",
    "phase_difference",
    "logm_v",
    "quantum_fisher_information",
    "fubini_study_metric",
    # Gate front-ends
    "Gates",
    "UnitaryGates",
    "PulseGates",
    "PulseParams",
    "PulseEnvelope",
    "PulseInformation",
    "PulseParamManager",
    # Utilities
    "safe_random_split",
]
