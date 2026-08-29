"""The concrete gate library.

Every gate here is an :class:`~jaqsi.operations.Operation` subclass carrying its
own matrix definition, so instantiating one inside a circuit function records it
on the active tape.  These are the matrix-level gates; the user-facing dispatch
between unitary and pulse implementations lives in :mod:`jaqsi.gates`.
"""

from typing import List, Optional, Union
from functools import reduce

import jax
import jax.numpy as jnp
import numpy as np

from jaqsi.operations import (
    Operation,
    Hermitian,
    cdtype,
)


class Id(Operation):
    """Identity gate.

    Supports an arbitrary number of wires.  When more than one wire is
    given the matrix is the ``2**k x 2**k`` identity (where *k* is the
    number of wires).
    """

    _matrix = jnp.eye(2, dtype=cdtype())
    _num_wires = None  # accept any number of wires
    is_clifford = True

    def __init__(self, wires: Union[int, List[int]] = 0, **kwargs) -> None:
        """Initialise an identity gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
                When multiple wires are given the matrix is automatically
                expanded to the matching ``2**k × 2**k`` identity.
        """
        w = list(wires) if isinstance(wires, (list, tuple)) else [wires]
        k = len(w)
        if k > 1:
            kwargs["matrix"] = jnp.eye(2**k, dtype=cdtype())
        super().__init__(wires=wires, **kwargs)


class PauliX(Operation):
    """Pauli-X gate / observable (bit-flip, \\sigma_x)."""

    _matrix = jnp.array([[0, 1], [1, 0]], dtype=cdtype())
    _num_wires = 1
    is_clifford = True

    def __init__(self, wires: Union[int, List[int]] = 0, **kwargs) -> None:
        """Initialise a Pauli-X gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires, **kwargs)


class PauliY(Operation):
    """Pauli-Y gate / observable (\\sigma_y)."""

    _matrix = jnp.array([[0, -1j], [1j, 0]], dtype=cdtype())
    _num_wires = 1
    is_clifford = True

    def __init__(self, wires: Union[int, List[int]] = 0, **kwargs) -> None:
        """Initialise a Pauli-Y gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires, **kwargs)


class PauliZ(Operation):
    """Pauli-Z gate / observable (phase-flip, \\sigma_z)."""

    _matrix = jnp.array([[1, 0], [0, -1]], dtype=cdtype())
    _num_wires = 1
    is_clifford = True

    def __init__(self, wires: Union[int, List[int]] = 0, **kwargs) -> None:
        """Initialise a Pauli-Z gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires, **kwargs)


class H(Operation):
    """Hadamard gate."""

    _matrix = jnp.array([[1, 1], [1, -1]], dtype=cdtype()) / jnp.sqrt(2)
    _num_wires = 1
    is_clifford = True

    def __init__(self, wires: Union[int, List[int]] = 0, **kwargs) -> None:
        """Initialise a Hadamard gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires, **kwargs)


class S(Operation):
    """S (phase) gate — a Clifford gate equal to \\sqrt Z.

    .. math::
        S = \\begin{pmatrix}1 & 0\\ 0 & i\\end{pmatrix}
    """

    _matrix = jnp.array([[1, 0], [0, 1j]], dtype=cdtype())
    _num_wires = 1
    is_clifford = True

    def __init__(self, wires: Union[int, List[int]] = 0) -> None:
        """Initialise an S gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires)


class SWAP(Operation):
    """SWAP gate."""

    _matrix = jnp.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=cdtype()
    )
    _num_wires = 2
    is_clifford = True

    def __init__(self, wires: Union[int, List[int]] = 0, **kwargs) -> None:
        """Initialise a SWAP gate.

        Args:
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        super().__init__(wires=wires, **kwargs)


class RandomUnitary(Operation):
    """Creates a random hermitian matrix and applies it as a gate."""

    def __init__(
        self,
        wires: Union[int, List[int]],
        key: jax.random.PRNGKey,
        scale: float = 1.0,
        record: bool = True,
    ) -> None:
        """Initialise a random unitary gate.

        Args:
            wires (Union[int, List[int]]): Qubit index or list of qubit indices
                this gate acts on.
            key (jax.random.PRNGKey): PRNGKey for randomization.
            scale (float): Scale of the random unitary (default: 1.0).
            record (bool): Whether to record this gate on the active tape.
        """
        dim = 2 ** len(wires)
        key_a, key_b = jax.random.split(key)

        A = (
            jax.random.normal(key=key_a, shape=(dim, dim))
            + 1j * jax.random.normal(key=key_b, shape=(dim, dim))
        ).astype(cdtype())
        H = (A + A.conj().T) / 2.0

        H *= scale / jnp.linalg.norm(H, ord="fro")

        super().__init__(wires, matrix=H, record=record)


class DiagonalQubitUnitary(Operation):
    """A diagonal unitary gate specified by its diagonal entries.

    Implements ``U = diag(d_0, d_1, ..., d_{2^k-1})`` where each ``d_i`` lies
    on the unit circle.  This is the natural gate for data-encoding
    Hamiltonians of the form ``S(x) = exp(-i H x)`` where *H* is diagonal in
    the computational basis (see Peters et al., arXiv:2209.05523).

    The Golomb encoding strategy uses this gate with diagonal entries
    ``exp(-i * golomb_marks * x)`` to achieve a maximally non-degenerate
    Fourier spectrum.

    Args:
        diag: 1-D array of ``2**k`` complex values on the unit circle.
        wires: Qubit indices this gate acts on (s.t. ``2**len(wires) == len(diag)``).
        **kwargs: Forwarded to :class:`Operation`.
    """

    # Do NOT list "diag" in _param_names — the array is not a scalar
    # parameter and would break drawing helpers that call float(p).
    _param_names = ()

    def __init__(
        self,
        diag: jnp.ndarray,
        wires: Union[int, List[int]] = 0,
        generator: Optional[jnp.ndarray] = None,
        scale: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.diag = diag
        # Optional real data-encoding generator: ``diag = exp(-i * generator *
        # scale)`` with a real diagonal Hamiltonian ``generator`` and real
        # scalar ``scale``.  When present, :meth:`decompose` expands the gate
        # into commuting Pauli-Z rotations; the complex ``diag`` alone is
        # insufficient because its phase wraps modulo ``2 pi``.
        self._generator = generator
        self._scale = scale
        wires_list = list(wires) if isinstance(wires, (list, tuple)) else [wires]
        expected_dim = 2 ** len(wires_list)
        if diag.shape != (expected_dim,):
            raise ValueError(
                f"DiagonalQubitUnitary expects {expected_dim} diagonal entries "
                f"for {len(wires_list)} wire(s), got shape {diag.shape}"
            )
        mat = jnp.diag(diag)
        # Use a descriptive name for drawing
        kwargs.setdefault("name", "DiagU")
        super().__init__(wires=wires, matrix=mat, **kwargs)

    def decompose(self) -> List["Operation"]:
        r"""Expand a real-generator diagonal encoding into Pauli-Z rotations.

        For ``diag = exp(-i H x)`` with a real diagonal Hamiltonian
        ``H = diag(self._generator)`` and real scalar ``x = self._scale``, the
        Walsh-Hadamard transform writes ``H = \sum_P \alpha_P P`` over commuting
        Pauli-Z strings ``P``.  Because the strings commute,

        .. math::
            e^{-i H x} = \prod_P e^{-i \alpha_P x P}
                       = \prod_P \mathrm{PauliRot}(2 x \alpha_P, P),

        up to the global phase from the identity term (dropped).  Zero-weight
        strings are omitted.

        Returns:
            List of :class:`PauliRot` gates (``record=False``) whose ordered
            product equals the diagonal gate up to a global phase.

        Raises:
            NotImplementedError: If the gate carries no real generator (a
                generic diagonal unitary has no Pauli-rotation decomposition).
        """
        if self._generator is None:
            return super().decompose()

        k = len(self.wires)
        dim = 2**k
        marks = np.asarray(self._generator, dtype=float).reshape(-1)
        # Sign vector per qubit position: +1/-1 for basis bit 0/1.  Position i
        # (i = 0 is the most-significant bit of the diagonal index) acts on
        # ``self.wires[i]`` -- the same order PauliRot uses for its word.
        signs = np.array(
            [[1 - 2 * ((j >> (k - 1 - i)) & 1) for j in range(dim)] for i in range(k)]
        )

        ops: List["Operation"] = []
        tol = 1e-12
        for mask in range(1, dim):  # skip mask 0 (identity -> global phase)
            chi = np.ones(dim)
            for i in range(k):
                if (mask >> i) & 1:
                    chi = chi * signs[i]
            alpha = float(marks @ chi) / dim
            if abs(alpha) < tol:
                continue
            word = "".join("Z" if (mask >> i) & 1 else "I" for i in range(k))
            theta = 2.0 * alpha * self._scale
            ops.append(PauliRot(theta, word, wires=self.wires, record=False))
        return ops

    def apply_to_state(self, state: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Apply diagonal gate via element-wise multiplication.

        For a diagonal unitary, the full ``2^n``-dimensional diagonal is
        constructed by appropriate Kronecker-product embedding and the gate
        is applied as an element-wise product, which is significantly cheaper
        than generic matrix contraction for large qubit counts.

        Args:
            state: Statevector of shape ``(2**n_qubits,)``.
            n_qubits: Total number of qubits in the circuit.

        Returns:
            Updated statevector of shape ``(2**n_qubits,)``.
        """
        k = len(self.wires)
        if k == n_qubits and self.wires == list(range(n_qubits)):
            # Gate acts on all qubits in order — direct element-wise multiply
            return state * self.diag
        # Fall back to general tensor contraction for arbitrary wire subsets
        return super().apply_to_state(state, n_qubits)

    def apply_to_density(self, rho: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Apply diagonal gate to density matrix: rho -> U rho U†.

        For diagonal U the transformation is
        ``rho_ij -> d_i * conj(d_j) * rho_ij``.

        Args:
            rho: Density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
            n_qubits: Total number of qubits in the circuit.

        Returns:
            Updated density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
        """
        k = len(self.wires)
        if k == n_qubits and self.wires == list(range(n_qubits)):
            d = self.diag
            return d[:, None] * jnp.conj(d)[None, :] * rho
        return super().apply_to_density(rho, n_qubits)


class Barrier(Operation):
    """Barrier operation — a no-op used for visual circuit separation.

    The barrier does not change the quantum state.  It is recorded on the
    tape so that drawing backends can insert a visual separator.
    """

    _matrix = None  # not a real gate

    def __init__(self, wires: Union[int, List[int]] = 0) -> None:
        """Initialise a Barrier.

        Args:
            wires: Qubit index or list of qubit indices this barrier spans.
        """
        super().__init__(wires=wires)

    def apply_to_state(self, state: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """No-op: return the state unchanged."""
        return state

    def apply_to_state_tensor(self, psi: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """No-op: return the state tensor unchanged."""
        return psi

    def apply_to_density(self, rho: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """No-op: return the density matrix unchanged."""
        return rho


_PAULI_LABELS = ["I", "X", "Y", "Z"]
_PAULI_CLASSES = [Id, PauliX, PauliY, PauliZ]
_PAULI_MATRICES = {
    label: cls._matrix for label, cls in zip(_PAULI_LABELS, _PAULI_CLASSES)
}
_PAULI_MATS = [_PAULI_MATRICES[label] for label in _PAULI_LABELS]

# Map from operation name to single-qubit Pauli label.
_NAME_TO_PAULI_LABEL = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "I": "I"}


def _pauli_tensor(word: str) -> jnp.ndarray:
    """Tensor product of single-qubit Pauli matrices for a Pauli word."""
    return reduce(jnp.kron, [_PAULI_MATRICES[c] for c in word])


def _rot_matrix(theta: float, pauli: jnp.ndarray) -> jnp.ndarray:
    """Return ``cos(theta/2) I - i sin(theta/2) P`` for a Pauli tensor *P*."""
    dim = pauli.shape[0]
    return (
        jnp.cos(theta / 2) * jnp.eye(dim, dtype=cdtype())
        - 1j * jnp.sin(theta / 2) * pauli
    )


def _make_rotation_gate(pauli_class: type, name: str) -> type:
    """Factory for single-qubit rotation gates RX, RY, RZ.

    Each gate has the form ``R_P(\\theta) = cos(\\theta/2) I - i sin(\\theta/2) P``.

    Args:
        pauli_class: One of PauliX, PauliY, PauliZ.
        name: Class name for the generated gate (e.g. ``"RX"``).

    Returns:
        A new :class:`Operation` subclass.
    """
    pauli_mat = pauli_class._matrix

    class _RotationGate(Operation):
        # Fancy way of setting docstring to make it generic
        __doc__ = (
            f"Rotation around the {name[1]} axis: {name}(\\theta) =\n"
            f"exp(-i \\theta/2 {name[1]}).\n"
        )
        _num_wires = 1
        _param_names = ("theta",)

        def __init__(
            self, theta: float, wires: Union[int, List[int]] = 0, **kwargs
        ) -> None:
            self.theta = theta
            mat = _rot_matrix(theta, pauli_mat)
            super().__init__(wires=wires, matrix=mat, **kwargs)

        def generator(self) -> Operation:
            """Return the generator as the corresponding Pauli operation."""
            return pauli_class(wires=self.wires[0], record=False)

    _RotationGate.__name__ = name
    _RotationGate.__qualname__ = name
    return _RotationGate


RX = _make_rotation_gate(PauliX, "RX")
RY = _make_rotation_gate(PauliY, "RY")
RZ = _make_rotation_gate(PauliZ, "RZ")


# Projectors used by controlled-gate factories
_P0 = jnp.array([[1, 0], [0, 0]], dtype=cdtype())
_P1 = jnp.array([[0, 0], [0, 1]], dtype=cdtype())


def _make_controlled_gate(target_class: type, name: str) -> type:
    """Factory for controlled Pauli gates CX, CY, CZ.

    Each gate has the form
    ``CP = |0><0| \\otimes I + |1\\langle\\rangle 1| \\otimes P``.

    Args:
        target_class: The single-qubit gate class (PauliX, PauliY, PauliZ).
        name: Class name for the generated gate (e.g. ``"CX"``).

    Returns:
        A new :class:`Operation` subclass.
    """
    target_mat = target_class._matrix

    class _ControlledGate(Operation):
        __doc__ = (
            f"Controlled-{target_class.__name__[5:]} gate.\n\n"
            f"Applies {target_class.__name__} on the target qubit conditioned "
            f"on the control qubit being in state |1\\rangle."
        )
        _matrix = jnp.kron(_P0, Id._matrix) + jnp.kron(_P1, target_mat)
        _num_wires = 2
        is_controlled = True
        is_clifford = True  # CX, CY, CZ are all Clifford gates

        def __init__(self, wires: List[int] = [0, 1], **kwargs) -> None:
            super().__init__(wires=wires, **kwargs)

        def decompose(self) -> List["Operation"]:
            # CZ = (H on target) CX (H on target).  CX/CY are primitive.
            if name != "CZ":
                return super().decompose()
            c, t = self.wires
            return [
                H(wires=t, record=False),
                CX(wires=[c, t], record=False),
                H(wires=t, record=False),
            ]

    _ControlledGate.__name__ = name
    _ControlledGate.__qualname__ = name
    return _ControlledGate


CX = _make_controlled_gate(PauliX, "CX")
CY = _make_controlled_gate(PauliY, "CY")
CZ = _make_controlled_gate(PauliZ, "CZ")


class CCX(Operation):
    """Toffoli (CCX) gate.

    The 3-qubit Toffoli gate exercises the arbitrary-k-qubit path in
    :meth:`~Operation.apply_to_state` and cannot be expressed as a pair of
    2-qubit gates without ancilla, making it a good stress-test for the
    simulator.
    """

    _matrix = jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ],
        dtype=cdtype(),
    )
    is_controlled = True
    _num_wires = 3

    def __init__(self, wires: List[int] = [0, 1, 2], **kwargs) -> None:
        """Initialise a Toffoli (CCX) gate.

        Args:
            wires: Three-element list ``[control0, control1, target]``.
        """
        super().__init__(wires=wires, **kwargs)


class CSWAP(Operation):
    """Controlled-SWAP (Fredkin) gate.

    Swaps the two target qubits conditioned on the control qubit being |1\\rangle.

    Args on construction:
        wires: ``[control, target0, target1]``.
    """

    _matrix = jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=cdtype(),
    )
    is_controlled = True
    _num_wires = 3

    def __init__(self, wires: List[int] = [0, 1, 2], **kwargs) -> None:
        """Initialise a Controlled-SWAP (Fredkin) gate.

        Args:
            wires: Three-element list ``[control, target0, target1]``.
        """
        super().__init__(wires=wires, **kwargs)


class ControlledPhaseShift(Operation):
    r"""Controlled phase shift gate (CPhase).

    Applies a phase shift of ``exp(i * phi)`` to the |11⟩ component of the
    two-qubit state, leaving all other computational basis states unchanged.
    This is a generalization of the CZ gate: when ``phi = \\pi`` the gate
    reduces to CZ.

    .. math::
        \text{CPhase}(\phi) = \text{diag}(1, 1, 1, e^{i\phi})

    which is equivalent to
    ``|0⟩⟨0| \\otimes I + |1⟩⟨1| \\otimes P(phi)`` where
    ``P(phi) = diag(1, exp(i*phi))``.
    """

    _num_wires = 2
    _param_names = ("phi",)
    is_controlled = True

    def __init__(self, phi: float, wires: List[int] = [0, 1], **kwargs) -> None:
        """Initialise a controlled phase shift gate.

        Args:
            phi: Phase shift angle in radians.
            wires: Two-element list ``[control, target]``.
        """
        self.phi = phi
        phase_gate = jnp.array([[1, 0], [0, jnp.exp(1j * phi)]], dtype=cdtype())
        mat = jnp.kron(_P0, Id._matrix) + jnp.kron(_P1, phase_gate)
        super().__init__(wires=wires, matrix=mat, **kwargs)


class Rot(Operation):
    """General single-qubit rotation:
    Rot(\\phi, \\theta, \\omega) = RZ(\\omega) RY(\\theta) RZ(\\phi).

    This is the most general SU(2) rotation (up to a global phase).  It
    decomposes into three successive rotations and has three free parameters.
    """

    _num_wires = 1
    _param_names = ("phi", "theta", "omega")

    def __init__(
        self,
        phi: float,
        theta: float,
        omega: float,
        wires: Union[int, List[int]] = 0,
        **kwargs,
    ) -> None:
        """Initialise a general rotation gate.

        Args:
            phi: First RZ rotation angle (radians).
            theta: RY rotation angle (radians).
            omega: Second RZ rotation angle (radians).
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        self.phi = phi
        self.theta = theta
        self.omega = omega
        # Rot(\\phi, \theta, \\omega) = RZ(\\omega) @ RY(\theta) @ RZ(\\phi)
        rz_phi = _rot_matrix(phi, PauliZ._matrix)
        ry_theta = _rot_matrix(theta, PauliY._matrix)
        rz_omega = _rot_matrix(omega, PauliZ._matrix)
        mat = rz_omega @ ry_theta @ rz_phi
        super().__init__(wires=wires, matrix=mat, **kwargs)

    def decompose(self) -> List["Operation"]:
        """Decompose into ``RZ(phi) RY(theta) RZ(omega)`` (same wire)."""
        w = self.wires[0]
        return [
            RZ(self.phi, wires=w, record=False),
            RY(self.theta, wires=w, record=False),
            RZ(self.omega, wires=w, record=False),
        ]


class PauliRot(Operation):
    """Multi-qubit Pauli rotation: exp(-i \\theta/2 P) for a Pauli word P.

    The Pauli word is given as a string of ``'I'``, ``'X'``, ``'Y'``, ``'Z'``
    characters (one per qubit).  The rotation matrix is computed as
    ``cos(\\theta/2) I - i sin(\\theta/2) P`` where *P* is the tensor product of the
    corresponding single-qubit Pauli matrices.

    Example::

        PauliRot(0.5, "XY", wires=[0, 1])
    """

    _param_names = ("theta",)

    # Map from character to 2x2 matrix (canonical single source of truth)
    _PAULI_MAP = _PAULI_MATRICES

    def __init__(
        self, theta: float, pauli_word: str, wires: Union[int, List[int]] = 0, **kwargs
    ) -> None:
        """Initialise a PauliRot gate.

        Args:
            theta: Rotation angle in radians.
            pauli_word: A string of ``'I'``, ``'X'``, ``'Y'``, ``'Z'``
                characters specifying the Pauli tensor product.
            wires: Qubit index or list of qubit indices this gate acts on.
        """
        self.theta = theta
        self.pauli_word = pauli_word

        P = _pauli_tensor(pauli_word)
        mat = _rot_matrix(theta, P)
        super().__init__(wires=wires, matrix=mat, **kwargs)

    def generator(self) -> Operation:
        """Return the generator Pauli tensor product as an :class:`Operation`.

        The generator of ``PauliRot(\\theta, word, wires)`` is the tensor product
        of single-qubit Pauli matrices specified by *word*.  The returned
        :class:`Hermitian` wraps that matrix and the gate's wires.

        Returns:
            :class:`Hermitian` operation representing the Pauli tensor product.
        """
        P = _pauli_tensor(self.pauli_word)
        return Hermitian(matrix=P, wires=self.wires, record=False)


def _make_pauli_rotation_subclass(name: str, word: str) -> type:
    """Build a thin :class:`PauliRot` subclass with the Pauli word fixed.

    Used to expose multi-qubit Pauli rotations (``RXX``, ``RYY``, ``RZZ``,
    ``RZX``, ...) as standalone classes while sharing PauliRot's matrix
    construction and generator logic.
    """

    sep = " \\otimes "
    doc = (
        f"{name}(\\theta) = exp(-i \\theta/2\\, {sep.join(word)}).\n\n"
        f"Thin :class:`PauliRot` subclass with ``pauli_word={word!r}``."
    )

    class _PauliRotSubclass(PauliRot):
        __doc__ = doc
        _num_wires = len(word)

        def __init__(
            self,
            theta: float,
            wires: Union[int, List[int]] = None,
            **kwargs,
        ) -> None:
            if wires is None:
                wires = list(range(len(word)))
            super().__init__(theta, word, wires=wires, **kwargs)

    _PauliRotSubclass.__name__ = name
    _PauliRotSubclass.__qualname__ = name
    return _PauliRotSubclass


RXX = _make_pauli_rotation_subclass("RXX", "XX")
RYY = _make_pauli_rotation_subclass("RYY", "YY")
RZZ = _make_pauli_rotation_subclass("RZZ", "ZZ")
RZX = _make_pauli_rotation_subclass("RZX", "ZX")


# --- Controlled multi-qubit Pauli rotation ---------------------------------


class ControlledPauliRot(Operation):
    r"""Multi-controlled multi-qubit Pauli rotation.

    Applies ``PauliRot(theta, pauli_word)`` on the *target* wires
    conditioned on all *control* wires being in :math:`|1\rangle`.

    For a single control wire and a single-character Pauli word this
    reduces to the textbook controlled rotations ``CRX``, ``CRY``,
    ``CRZ`` — these are exposed below as thin subclasses.

    The wire layout is ``[control_0, ..., control_{n_controls-1},
    target_0, ..., target_{m-1}]`` where ``m = len(pauli_word)``.
    """

    _param_names = ("theta",)
    is_controlled = True

    def __init__(
        self,
        theta: float,
        pauli_word: str,
        wires: List[int],
        n_controls: int = 1,
        **kwargs,
    ) -> None:
        self.theta = theta
        self.pauli_word = pauli_word
        self.n_controls = n_controls

        wires_list = [wires] if isinstance(wires, int) else list(wires)
        n_targets = len(pauli_word)
        if len(wires_list) != n_controls + n_targets:
            raise ValueError(
                f"ControlledPauliRot expects {n_controls + n_targets} wires "
                f"({n_controls} control + {n_targets} target), got "
                f"{len(wires_list)}."
            )

        P = _pauli_tensor(pauli_word)
        d_t = P.shape[0]
        R = _rot_matrix(theta, P)

        d_c = 2**n_controls
        dim = d_c * d_t
        # All control patterns except |1...1> act trivially; the active
        # block sits in the last d_t x d_t slot.
        mat = jnp.eye(dim, dtype=cdtype())
        start = (d_c - 1) * d_t
        mat = mat.at[start : start + d_t, start : start + d_t].set(R)

        super().__init__(wires=wires_list, matrix=mat, **kwargs)

    def generator(self) -> Operation:
        """Return the (Hermitian) generator on the full wire set."""
        P = _pauli_tensor(self.pauli_word)
        d_t = P.shape[0]
        d_c = 2**self.n_controls
        dim = d_c * d_t
        gen = jnp.zeros((dim, dim), dtype=cdtype())
        start = (d_c - 1) * d_t
        gen = gen.at[start : start + d_t, start : start + d_t].set(P)
        return Hermitian(matrix=gen, wires=self.wires, record=False)


def _make_controlled_rotation_subclass(name: str, axis: str) -> type:
    """Build a single-control controlled single-qubit rotation subclass.

    Reproduces the historical ``CRX``, ``CRY``, ``CRZ`` API as thin
    :class:`ControlledPauliRot` subclasses.
    """

    class _CRotation(ControlledPauliRot):
        __doc__ = (
            f"Controlled rotation around the {axis} axis.\n\n"
            f"Applies R{axis}(\\theta) on the target qubit conditioned on the "
            f"control qubit being in state |1\\rangle.\n\n"
            f".. math::\n"
            f"{name}(\\theta) = |0\\rangle\\langle 0| \\otimes I\n"
            f"                  + |1\\rangle\\langle 1| \\otimes R{axis}(\\theta)"
        )
        _num_wires = 2

        def __init__(self, theta: float, wires: List[int] = [0, 1], **kwargs) -> None:
            super().__init__(theta, axis, wires=wires, n_controls=1, **kwargs)

        def decompose(self) -> List["Operation"]:
            """Decompose into Clifford + single-qubit Pauli rotations."""
            c, t = self.wires
            theta = self.theta
            if axis == "Z":
                return [
                    RZ(theta / 2, wires=t, record=False),
                    CX(wires=[c, t], record=False),
                    RZ(-theta / 2, wires=t, record=False),
                    CX(wires=[c, t], record=False),
                ]
            if axis == "X":
                return [
                    H(wires=t, record=False),
                    RZ(theta / 2, wires=t, record=False),
                    CX(wires=[c, t], record=False),
                    RZ(-theta / 2, wires=t, record=False),
                    CX(wires=[c, t], record=False),
                    H(wires=t, record=False),
                ]
            # axis == "Y"
            return [
                RX(-jnp.pi / 2, wires=t, record=False),
                RZ(theta / 2, wires=t, record=False),
                CX(wires=[c, t], record=False),
                RZ(-theta / 2, wires=t, record=False),
                RX(jnp.pi / 2, wires=t, record=False),
            ]

    _CRotation.__name__ = name
    _CRotation.__qualname__ = name
    return _CRotation


CRX = _make_controlled_rotation_subclass("CRX", "X")
CRY = _make_controlled_rotation_subclass("CRY", "Y")
CRZ = _make_controlled_rotation_subclass("CRZ", "Z")

def build_parity_observable(
    qubit_group: List[int],
) -> Hermitian:
    """Build a multi-qubit parity observable.

    Args:
        qubit_group: List of qubit indices for the parity measurement.

    Returns:
        A :class:`Hermitian` operation whose matrix is the Z parity
        tensor product and whose wires match the given qubits.
    """
    Z = PauliZ._matrix
    mat = reduce(jnp.kron, [Z] * len(qubit_group))
    obs = Hermitian(matrix=mat, wires=qubit_group, record=False)
    # Tag the Pauli string so symbolic consumers (PauliWord / FourierTree) can
    # read it without an O(4^n) matrix decomposition.
    obs._pauli_label = "Z" * len(qubit_group)
    return obs
