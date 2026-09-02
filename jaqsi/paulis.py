"""Symbolic Pauli/Clifford algebra.

A stabilizer-tableau representation of Pauli words with O(n) Clifford
conjugation, plus the matrix-based decomposition helpers it replaces.  This is
symbolic bookkeeping in integer NumPy rather than numeric simulation, and it
backs Pauli-Clifford circuit transforms and Fourier-tree algorithms built on
top of it.
"""

from typing import List, Optional, Tuple, Union
from itertools import product

import numpy as np
import jax.numpy as jnp

from jaqsi.operations import (
    Operation,
    Hermitian,
    cdtype,
    _embed_matrix,
)
from jaqsi.gateset import (
    Id,
    PauliRot,
    _PAULI_LABELS,
    _PAULI_CLASSES,
    _PAULI_MATRICES,
    _NAME_TO_PAULI_LABEL,
    _pauli_tensor,
)


def evolve_pauli_with_clifford(
    clifford: Operation,
    pauli: Operation,
    adjoint_left: bool = True,
) -> Operation:
    """Compute C\\dagger P C  (or  C P C\\dagger)  and
    return the result as an Operation.

    Both operators are first embedded into the full Hilbert space spanned by
    the union of their wire sets.  The result is wrapped in a
    :class:`Hermitian` so it can be used in further algebra.

    Args:
        clifford: A Clifford gate.
        pauli: A Pauli / Hermitian operator.
        adjoint_left: If ``True``, compute C\\dagger P C; otherwise C P C\\dagger.

    Returns:
        A :class:`Hermitian` wrapping the evolved matrix.
    """
    all_wires = sorted(set(clifford.wires) | set(pauli.wires))
    n = len(all_wires)

    C = _embed_matrix(clifford.matrix, clifford.wires, all_wires, n)
    P = _embed_matrix(pauli.matrix, pauli.wires, all_wires, n)
    Cd = jnp.conj(C).T

    if adjoint_left:
        result = Cd @ P @ C
    else:
        result = C @ P @ Cd

    return Hermitian(matrix=result, wires=all_wires, record=False)



def _dominant_pauli_label(matrix: jnp.ndarray) -> Tuple[complex, str]:
    r"""Return the dominant Pauli term ``(coeff, label)`` of a matrix.

    Finds the Pauli tensor product :math:`P` (over ``I, X, Y, Z``) with the
    largest :math:`|c_P|`, where :math:`c_P = \mathrm{Tr}(P M) / 2^n`.  Shared
    by :func:`pauli_decompose` and :meth:`PauliWord.from_matrix` so the
    brute-force search lives in one place.

    Args:
        matrix: A ``(2**n, 2**n)`` matrix.

    Returns:
        ``(coeff, label)`` with *label* a string over ``{I, X, Y, Z}``.
    """
    dim = matrix.shape[0]
    n_qubits = int(jnp.round(jnp.log2(dim)))

    best_label = "I" * n_qubits
    best_coeff = 0.0
    for indices in product(range(4), repeat=n_qubits):
        label = "".join(_PAULI_LABELS[i] for i in indices)
        P = _pauli_tensor(label)
        coeff = jnp.trace(P @ matrix) / dim
        if jnp.abs(coeff) > jnp.abs(best_coeff):
            best_coeff = coeff
            best_label = label
    return best_coeff, best_label


def pauli_decompose(matrix: jnp.ndarray, wire_order: Optional[List[int]] = None):
    r"""Decompose a Hermitian matrix into a sum of Pauli tensor products.

    For an n-qubit matrix (``2**n x 2**n``), returns the dominant Pauli
    term (the one with the largest absolute coefficient), wrapped as an
    :class:`Operation`.  This is sufficient for the Fourier-tree algorithm
    which only needs the single non-zero Pauli term produced by Clifford
    conjugation of a Pauli operator.

    The decomposition uses the trace formula:
    ``c_P = Tr(P · M) / 2**n``

    Args:
        matrix: A ``(2**n, 2**n)`` Hermitian matrix.
        wire_order: Optional list of wire indices.  If ``None``, defaults
            to ``[0, 1, ..., n-1]``.

    Returns:
        A tuple ``(coeff, op)`` where *coeff* is the complex coefficient and
        *op* is the Pauli :class:`Operation` (PauliX, PauliY, PauliZ, I, or
        a :class:`Hermitian` for multi-qubit tensor products).
    """
    dim = matrix.shape[0]
    n_qubits = int(jnp.round(jnp.log2(dim)))

    if wire_order is None:
        wire_order = list(range(n_qubits))

    best_coeff, pauli_label = _dominant_pauli_label(matrix)
    label_to_idx = {label: i for i, label in enumerate(_PAULI_LABELS)}

    # Build the operation for the dominant term
    if sum(1 for ch in pauli_label if ch != "I") <= 1:
        # Single-qubit Pauli on one wire (or all-identity)
        for q, ch in enumerate(pauli_label):
            if ch != "I":
                result_op = _PAULI_CLASSES[label_to_idx[ch]](
                    wires=wire_order[q], record=False
                )
                result_op._pauli_label = ch
                return best_coeff, result_op
        result_op = Id(wires=wire_order[0], record=False)
        result_op._pauli_label = "I" * n_qubits
        return best_coeff, result_op
    else:
        # Multi-qubit tensor product -> Hermitian with pauli label attached
        P = _pauli_tensor(pauli_label)
        result_op = Hermitian(matrix=P, wires=wire_order, record=False)
        result_op._pauli_label = pauli_label
        return best_coeff, result_op


def pauli_string_from_operation(op: Operation) -> str:
    """Extract a Pauli word string from an operation.

    Maps ``PauliX`` -> ``"X"``, ``PauliY`` -> ``"Y"``, ``PauliZ`` -> ``"Z"``,
    ``I`` -> ``"I"``.  For :class:`PauliRot`, returns its stored ``pauli_word``.
    For operations produced by :func:`pauli_decompose`, returns the stored
    ``_pauli_label`` attribute.

    Args:
        op: A quantum operation.

    Returns:
        A string like ``"X"``, ``"ZZ"``, etc.
    """
    if isinstance(op, PauliRot) and hasattr(op, "pauli_word"):
        return op.pauli_word
    # Check for label stored by pauli_decompose
    if hasattr(op, "_pauli_label"):
        return op._pauli_label
    if op.name in _NAME_TO_PAULI_LABEL:
        return _NAME_TO_PAULI_LABEL[op.name]
    # Fall back: decompose the matrix
    _, pauli_op = pauli_decompose(op.matrix, wire_order=op.wires)
    return pauli_op._pauli_label


def prod(*ops: Operation) -> Operation:
    """Construct the generalized product (tensor or matrix) of multiple operations.

    The resulting operation acts on the union of all wire sets.
    If the wire sets are disjoint, this is a Kronecker product.
    If the wire sets overlap, the corresponding matrices are multiplied.

    Args:
        *ops: Variable number of :class:`Operation` instances.

    Returns:
        A new :class:`Operation` whose matrix represents the composed
        operation on the unified wire set.
    """
    if not ops:
        raise ValueError("At least one operation must be provided to prod().")
    return ops[0].prod(*ops[1:])


# Single-qubit (x, z) bit pattern -> Pauli label, with the convention that a
# Pauli word is stored as  i^phase * prod_q  X_q^{x_q} Z_q^{z_q}.
# Under this convention Y = i * X * Z, so the single-qubit Y carries x=z=1.
_XZ_TO_LABEL = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
_LABEL_TO_XZ = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}

# Single-qubit Hermitian Pauli matrices (NumPy) for arbitrary-state expectation.
_SQ_NP = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


class PauliWord:
    r"""Symbolic n-qubit Pauli operator in the stabilizer-tableau (symplectic)
    representation.

    A Pauli word is stored as

    .. math::
        P = i^{\text{phase}} \prod_{q} X_q^{x_q} Z_q^{z_q},

    with bit arrays ``x, z \in \{0, 1\}^n`` and an integer ``phase`` taken mod 4
    (tracking the scalar ``i^{phase}``).  Single-qubit Paulis map as
    ``I=(0,0)``, ``X=(1,0)``, ``Z=(0,1)``, ``Y=(1,1)`` (since ``Y = i X Z``).

    This replaces the matrix-based Clifford conjugation
    (:func:`evolve_pauli_with_clifford` + :func:`pauli_decompose`) with O(n)
    symbolic updates, and backs Pauli-Clifford circuit transforms and
    Fourier-tree algorithms built on top of it.

    All operations use NumPy (integer arithmetic), not JAX — this is symbolic
    bookkeeping, not numeric computation.
    """

    __slots__ = ("x", "z", "phase")

    def __init__(self, x: np.ndarray, z: np.ndarray, phase: int = 0) -> None:
        """Initialise a Pauli word.

        Args:
            x: Integer/boolean array of X-component bits, length ``n_qubits``.
            z: Integer/boolean array of Z-component bits, length ``n_qubits``.
            phase: Exponent of the global ``i^{phase}`` scalar (taken mod 4).
        """
        self.x = np.asarray(x, dtype=np.int8) & 1
        self.z = np.asarray(z, dtype=np.int8) & 1
        self.phase = int(phase) % 4

    # ---- constructors ---------------------------------------------------
    @classmethod
    def identity(cls, n_qubits: int) -> "PauliWord":
        """Return the identity Pauli word on *n_qubits*."""
        z = np.zeros(n_qubits, dtype=np.int8)
        return cls(z.copy(), z, 0)

    @classmethod
    def from_pauli_string(
        cls, pauli_string: str, wires: List[int], n_qubits: int
    ) -> "PauliWord":
        """Build a Pauli word from a Pauli string and its wires.

        Args:
            pauli_string: String over ``{'I', 'X', 'Y', 'Z'}``; one character
                per entry of *wires*.
            wires: Qubit indices the characters act on.
            n_qubits: Total number of qubits in the circuit.

        Returns:
            The corresponding :class:`PauliWord`.
        """
        x = np.zeros(n_qubits, dtype=np.int8)
        z = np.zeros(n_qubits, dtype=np.int8)
        n_y = 0
        for ch, w in zip(pauli_string, wires):
            xb, zb = _LABEL_TO_XZ[ch]
            x[w] = xb
            z[w] = zb
            if ch == "Y":
                n_y += 1
        # Each Y contributes a factor i (Y = i X Z), accumulated into phase.
        return cls(x, z, n_y % 4)

    @classmethod
    def from_operation(cls, op: "Operation", n_qubits: int) -> "PauliWord":
        """Build a Pauli word from a Pauli-like operation.

        Supports :class:`PauliX`/:class:`PauliY`/:class:`PauliZ`/:class:`Id`,
        :class:`PauliRot` (via its ``pauli_word``), and any operation carrying a
        ``_pauli_label`` (e.g. produced by :func:`pauli_decompose`) or otherwise
        decomposable by :func:`pauli_string_from_operation`.

        Args:
            op: The operation to convert.
            n_qubits: Total number of qubits in the circuit.

        Returns:
            The corresponding :class:`PauliWord`.
        """
        # Cached symbolic word (e.g. attached to a Clifford-evolved observable).
        cached = getattr(op, "_pauli_word", None)
        if isinstance(cached, PauliWord) and cached.n_qubits == n_qubits:
            return cached
        if isinstance(op, PauliRot):
            return cls.from_pauli_string(op.pauli_word, op.wires, n_qubits)
        # Single-qubit Pauli rotations: generator is the corresponding Pauli.
        rot_to_label = {"RX": "X", "RY": "Y", "RZ": "Z"}
        if op.name in rot_to_label:
            return cls.from_pauli_string(rot_to_label[op.name], op.wires, n_qubits)
        if op.name in _NAME_TO_PAULI_LABEL:
            return cls.from_pauli_string(
                _NAME_TO_PAULI_LABEL[op.name], op.wires, n_qubits
            )
        pauli_str = pauli_string_from_operation(op)
        return cls.from_pauli_string(pauli_str, op.wires, n_qubits)

    @property
    def n_qubits(self) -> int:
        """Number of qubits this Pauli word spans."""
        return self.x.shape[0]

    @property
    def xy_mask(self) -> np.ndarray:
        """Boolean mask of qubits carrying an X or Y (i.e. ``x`` bits set)."""
        return self.x.astype(bool)

    @property
    def is_diagonal(self) -> bool:
        """Whether the word is diagonal (only I/Z, i.e. no X component)."""
        return not bool(self.x.any())

    # ---- algebra --------------------------------------------------------
    def commutes_with(self, other: "PauliWord") -> bool:
        """Return whether this Pauli word commutes with *other*.

        Two Paulis commute iff their symplectic inner product vanishes mod 2.
        """
        sp = int(np.dot(self.x, other.z) + np.dot(self.z, other.x)) % 2
        return sp == 0

    def compose(self, other: "PauliWord") -> "PauliWord":
        r"""Return the operator product ``self @ other`` as a new Pauli word.

        Uses the exact symplectic product rule

        .. math::
            (X^{x_1} Z^{z_1})(X^{x_2} Z^{z_2})
              = (-1)^{z_1 \cdot x_2}\, X^{x_1 \oplus x_2} Z^{z_1 \oplus z_2},

        combined with the ``i^{phase}`` scalars (``-1 = i^2``).
        """
        new_x = self.x ^ other.x
        new_z = self.z ^ other.z
        cross = int(np.dot(self.z, other.x))
        new_phase = (self.phase + other.phase + 2 * cross) % 4
        return PauliWord(new_x, new_z, new_phase)

    def conjugate_by_clifford(
        self, clifford: "Operation", adjoint_left: bool = False
    ) -> "PauliWord":
        r"""Return the Clifford conjugation of this Pauli word.

        Computes ``C P C^\dagger`` (``adjoint_left=False``) or
        ``C^\dagger P C`` (``adjoint_left=True``) symbolically, where *C* is one
        of the supported Clifford gates ``H, S, CX, CZ`` or a Pauli gate
        ``PauliX/PauliY/PauliZ``.

        The conjugation is realised by substituting the images of the
        single-qubit generators ``X_q`` and ``Z_q`` and re-composing in canonical
        order, so all phases are tracked exactly by :meth:`compose`.

        Args:
            clifford: The Clifford operation to conjugate by.
            adjoint_left: If ``True`` compute ``C^\dagger P C``; else
                ``C P C^\dagger``.

        Returns:
            The conjugated :class:`PauliWord`.

        Raises:
            NotImplementedError: If *clifford* is not a supported gate.
        """
        n = self.n_qubits
        name = clifford.name

        # Pauli gates: conjugation is just  Q P Q  (Q is Hermitian => Q^dagger=Q).
        if name in ("PauliX", "PauliY", "PauliZ"):
            q = PauliWord.from_operation(clifford, n)
            return q.compose(self).compose(q)

        try:
            images_x, images_z = self._clifford_generator_images(
                name, list(clifford.wires), adjoint_left, n
            )
        except NotImplementedError:
            # Any other Clifford (e.g. CY): fall back to the (exact) matrix
            # conjugation, which works for arbitrary Cliffords at O(2^n) cost.
            return self._conjugate_via_matrix(clifford, adjoint_left)

        result = PauliWord.identity(n)
        result.phase = self.phase
        for q in range(n):
            if self.x[q]:
                result = result.compose(images_x[q])
            if self.z[q]:
                result = result.compose(images_z[q])
        return result

    def _conjugate_via_matrix(
        self, clifford: "Operation", adjoint_left: bool
    ) -> "PauliWord":
        """Matrix-based Clifford conjugation fallback (exact, any Clifford).

        Used by :meth:`conjugate_by_clifford` for Cliffords without a symbolic
        tableau rule.  Reuses :meth:`to_matrix` / :meth:`from_matrix` and the
        gate's dense matrix.
        """
        n = self.n_qubits
        C = _embed_matrix(clifford.matrix, clifford.wires, list(range(n)), n)
        Cd = jnp.conj(C).T
        mat = self.to_matrix()
        result = (Cd @ mat @ C) if adjoint_left else (C @ mat @ Cd)
        return PauliWord.from_matrix(result)

    @staticmethod
    def _clifford_generator_images(
        name: str, wires: List[int], adjoint_left: bool, n: int
    ) -> Tuple[List["PauliWord"], List["PauliWord"]]:
        """Images of single-qubit generators ``X_q``/``Z_q`` under a Clifford.

        Returns two lists (indexed by qubit) of :class:`PauliWord` giving
        ``C X_q C^\\dagger`` and ``C Z_q C^\\dagger`` (or the adjoint direction).
        Qubits outside the gate support map to themselves.
        """

        def single(label: str, q: int) -> "PauliWord":
            return PauliWord.from_pauli_string(label, [q], n)

        images_x = [single("X", q) for q in range(n)]
        images_z = [single("Z", q) for q in range(n)]

        if name == "H":
            w = wires[0]
            images_x[w] = single("Z", w)  # H X H = Z
            images_z[w] = single("X", w)  # H Z H = X
        elif name == "S":
            w = wires[0]
            if adjoint_left:  # S^dagger X S = -Y ; S^dagger Z S = Z
                images_x[w] = PauliWord.from_pauli_string("Y", [w], n).compose(
                    PauliWord(np.zeros(n, np.int8), np.zeros(n, np.int8), 2)
                )
            else:  # S X S^dagger = Y ; S Z S^dagger = Z
                images_x[w] = single("Y", w)
            # images_z[w] unchanged (Z)
        elif name == "CX":
            c, t = wires
            images_x[c] = single("X", c).compose(single("X", t))  # X_c -> X_c X_t
            images_z[t] = single("Z", c).compose(single("Z", t))  # Z_t -> Z_c Z_t
            # X_t -> X_t and Z_c -> Z_c unchanged ; CX is Hermitian
        elif name == "CZ":
            c, t = wires
            images_x[c] = single("X", c).compose(single("Z", t))  # X_c -> X_c Z_t
            images_x[t] = single("Z", c).compose(single("X", t))  # X_t -> Z_c X_t
            # Z_c, Z_t unchanged ; CZ is Hermitian
        elif name == "SWAP":
            a, b = wires
            images_x[a], images_x[b] = single("X", b), single("X", a)  # swap supports
            images_z[a], images_z[b] = single("Z", b), single("Z", a)
        else:
            raise NotImplementedError(f"No symbolic Clifford rule for gate '{name}'.")
        return images_x, images_z

    # ---- expectation / conversions -------------------------------------
    def zero_expectation(self) -> complex:
        r"""Return ``<0|P|0>`` for the all-zero computational basis state.

        Non-zero only for diagonal words (I/Z only), in which case it equals the
        global phase ``i^{phase}``.
        """
        if not self.is_diagonal:
            return 0.0 + 0.0j
        return complex(1j**self.phase)

    def expectation(self, state: np.ndarray) -> complex:
        r"""Return ``\langle\psi|P|\psi\rangle`` for an arbitrary statevector.

        Applies the single-qubit Pauli factors to the reshaped state via tensor
        contraction (``O(n 2^n)``) instead of forming the dense
        ``2^n \times 2^n`` operator.  The real part is exact for a Hermitian
        Pauli word.

        Args:
            state: Statevector of length ``2**n_qubits`` (qubit 0 leftmost).

        Returns:
            The expectation value ``\langle\psi|P|\psi\rangle``.
        """
        n = self.n_qubits
        psi = np.asarray(state, dtype=complex).reshape((2,) * n)
        out = psi
        for q, ch in enumerate(self.to_pauli_string()):
            if ch == "I":
                continue
            out = np.moveaxis(np.tensordot(_SQ_NP[ch], out, axes=(1, q)), 0, q)
        val = np.vdot(psi.reshape(-1), out.reshape(-1))
        return self.leading_phase() * complex(val)

    def to_pauli_string(self) -> str:
        """Return the bare Pauli string (ignoring the global phase)."""
        return "".join(
            _XZ_TO_LABEL[(int(self.x[q]), int(self.z[q]))] for q in range(self.n_qubits)
        )

    def leading_phase(self) -> complex:
        r"""Return the scalar ``c`` such that ``P = c * (bare Pauli string)``.

        Because the bare string already contains ``i^{n_Y}`` from its Y factors,
        ``c = i^{phase - n_Y}``.
        """
        n_y = int(((self.x == 1) & (self.z == 1)).sum())
        return complex(1j ** ((self.phase - n_y) % 4))

    def to_pauli_string_and_phase(self) -> Tuple[str, complex]:
        """Return ``(bare Pauli string, leading scalar phase)``."""
        return self.to_pauli_string(), self.leading_phase()

    def to_matrix(self) -> jnp.ndarray:
        r"""Return the dense operator matrix ``i^{phase} \bigotimes_q X^{x_q} Z^{z_q}``.

        The per-qubit factor is the symplectic product ``X^{x} Z^{z}`` (so the
        ``(1, 1)`` factor is ``XZ = -iY``; the ``Y``-vs-``XZ`` phase is carried by
        ``i^{phase}``).  Inverse of :meth:`from_matrix`.
        """
        ident = _PAULI_MATRICES["I"]
        xmat = _PAULI_MATRICES["X"]
        zmat = _PAULI_MATRICES["Z"]
        mat = jnp.array([[1.0 + 0.0j]], dtype=cdtype())
        for q in range(self.n_qubits):
            factor = (xmat if self.x[q] else ident) @ (zmat if self.z[q] else ident)
            mat = jnp.kron(mat, factor)
        return (1j**self.phase) * mat

    @classmethod
    def from_matrix(cls, matrix: jnp.ndarray) -> "PauliWord":
        r"""Build a Pauli word from a matrix that is a single (signed) Pauli.

        Recovers the dominant Pauli string and folds its (unit) coefficient
        ``c = i^k`` into the word's phase.  Intended for matrices that are
        exactly a Pauli up to a ``{\pm 1, \pm i}`` scalar (e.g. the result of
        Clifford conjugation of a Pauli); the dominant term is returned for
        general inputs.

        Args:
            matrix: A ``(2**n, 2**n)`` matrix proportional to a Pauli string.

        Returns:
            The corresponding :class:`PauliWord` on ``n`` qubits.
        """
        coeff, label = _dominant_pauli_label(matrix)
        n = len(label)
        word = cls.from_pauli_string(label, list(range(n)), n)
        # Fold the unit coefficient  c = i^k  into the phase.
        k = int(round(np.angle(complex(coeff)) / (np.pi / 2))) % 4
        word.phase = (word.phase + k) % 4
        return word

    def to_list_repr(self) -> np.ndarray:
        """Return the legacy int list representation (I=-1, X=0, Y=1, Z=2)."""
        out = np.full(self.n_qubits, -1, dtype=int)
        for q in range(self.n_qubits):
            label = _XZ_TO_LABEL[(int(self.x[q]), int(self.z[q]))]
            out[q] = {"I": -1, "X": 0, "Y": 1, "Z": 2}[label]
        return out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PauliWord):
            return NotImplemented
        return (
            self.phase == other.phase
            and np.array_equal(self.x, other.x)
            and np.array_equal(self.z, other.z)
        )

    def __repr__(self) -> str:
        phase_str = {0: "+", 1: "+i", 2: "-", 3: "-i"}[self.phase]
        return f"PauliWord({phase_str}{self.to_pauli_string()})"


def state_expectation(obs: Union[str, PauliWord], state: np.ndarray) -> complex:
    r"""Return ``\langle\psi|O|\psi\rangle`` for a Pauli observable and statevector.

    Args:
        obs: The observable, either a :class:`PauliWord` or a bare Pauli string
            over ``{'I', 'X', 'Y', 'Z'}`` (qubit 0 leftmost).
        state: Statevector of length ``2**n_qubits``.

    Returns:
        The expectation value ``\langle\psi|O|\psi\rangle``.
    """
    if isinstance(obs, str):
        obs = PauliWord.from_pauli_string(obs, list(range(len(obs))), len(obs))
    return obs.expectation(state)
