"""Core operation machinery.

Defines the :class:`Operation` base class that every gate, observable and
noise channel derives from, together with the Hamiltonian types used as
time-evolution sources.  The concrete gate library lives in
:mod:`jaqsi.gateset`, the noise channels in :mod:`jaqsi.noise`, and the
symbolic Pauli/Clifford layer in :mod:`jaqsi.paulis`.
"""

from typing import Callable, List, Optional, Tuple, Union
from functools import lru_cache
import string
import numpy as np

import jax
import jax.numpy as jnp

from jaqsi.tape import active_tape, recording  # noqa: F401 (re-export)


def cdtype():
    """Return the active JAX complex dtype
    (complex128 if x64 enabled, else complex64).
    """
    return jnp.complex128 if jax.config.x64_enabled else jnp.complex64


@lru_cache(maxsize=256)
def _einsum_subscript(
    n: int,
    k: int,
    target_axes: Tuple[int, ...],
) -> str:
    """Build an ``einsum`` subscript that fuses contraction + axis restore.

    Args:
        n: Total rank of the state tensor (number of qubits for statevectors,
            ``2 * n_qubits`` for density matrices).
        k: Number of qubits the gate acts on.
        target_axes: Tuple of k axis indices in the state tensor that the
            gate contracts against.

    Returns:
        ``einsum`` subscript string, e.g. ``"ab,cBd->cad"`` for a 1-qubit
        gate on wire 1 of a 3-qubit state.
    """
    letters = string.ascii_letters
    # State indices: one letter per axis
    state_idx = list(letters[:n])
    # Contracted indices (the ones being replaced by the gate)
    contracted = [state_idx[ax] for ax in target_axes]
    # Gate indices: new output indices + contracted input indices
    new_out = [letters[n + i] for i in range(k)]  # fresh letters for output
    gate_idx = new_out + contracted  # gate shape: (out0, out1, ..., in0, in1, ...)
    # Result indices: replace target axes with new output letters
    result_idx = list(state_idx)
    for i, ax in enumerate(target_axes):
        result_idx[ax] = new_out[i]
    return "".join(gate_idx) + "," + "".join(state_idx) + "->" + "".join(result_idx)


def _contract_and_restore(
    tensor: jnp.ndarray,
    gate: jnp.ndarray,
    k: int,
    target_axes: List[int],
) -> jnp.ndarray:
    """Contract gate against target_axes of tensor and restore axis order.

    The einsum subscript is cached via :func:`_einsum_subscript` so the
    string construction only happens once per unique
    ``(total, k, target_axes)`` combination.

    Args:
        tensor: Rank-N tensor (e.g. ``(2,)*n`` for states or ``(2,)*2n``
            for density matrices).
        gate: Reshaped gate tensor of shape ``(2,)*2k``.
        k: Number of qubits the gate acts on (= ``len(target_axes)``).
        target_axes: The k axes of tensor to contract against.

    Returns:
        Updated tensor with the same rank as tensor, with the
        contracted axes restored to their original positions.
    """
    subscript = _einsum_subscript(tensor.ndim, k, tuple(target_axes))
    return jnp.einsum(subscript, gate, tensor)



def _embed_matrix(
    mat: jnp.ndarray,
    op_wires: list,
    all_wires: list,
    n_total: int,
) -> jnp.ndarray:
    """Embed a gate matrix into a larger Hilbert space via tensor products.

    If the gate already acts on all wires, the matrix is returned as-is.
    Otherwise the gate matrix is tensored with identities on the missing
    wires, and the resulting matrix rows/columns are permuted so that qubit
    ordering matches *all_wires*.

    Args:
        mat: The gate's unitary matrix of shape ``(2**k, 2**k)`` where
            ``k = len(op_wires)``.
        op_wires: The wires the gate acts on.
        all_wires: The full ordered list of wires.
        n_total: ``len(all_wires)``.

    Returns:
        A ``(2**n_total, 2**n_total)`` matrix.
    """
    k = len(op_wires)
    if k == n_total and list(op_wires) == list(all_wires):
        return mat

    # Build the full-space matrix by tensoring with identities
    # Strategy: tensor I on missing wires, then permute
    missing = [w for w in all_wires if w not in op_wires]
    # Full matrix = mat \\otimes I_{missing}
    full_mat = mat
    for _ in missing:
        full_mat = jnp.kron(full_mat, jnp.eye(2, dtype=cdtype()))

    # The current ordering is [op_wires..., missing...]
    # We need to permute to match all_wires ordering
    current_order = list(op_wires) + missing
    if current_order != list(all_wires):
        perm = [current_order.index(w) for w in all_wires]
        full_mat = _permute_matrix(full_mat, perm, n_total)

    return full_mat


def _permute_matrix(mat: jnp.ndarray, perm: list, n_qubits: int) -> jnp.ndarray:
    """Permute the qubit ordering of a matrix.

    Given a ``(2**n, 2**n)`` matrix and a permutation of ``[0..n-1]``,
    reorder the qubits so that qubit ``i`` moves to position ``perm[i]``.

    Args:
        mat: Square matrix of dimension ``2**n_qubits``.
        perm: Permutation list.
        n_qubits: Number of qubits.

    Returns:
        Permuted matrix of the same shape.
    """
    dim = 2**n_qubits
    # Reshape to tensor, permute axes, reshape back
    tensor = mat.reshape([2] * (2 * n_qubits))
    # Axes: first n_qubits are row indices, last n_qubits are column indices
    row_perm = perm
    col_perm = [p + n_qubits for p in perm]
    tensor = jnp.transpose(tensor, row_perm + col_perm)
    return tensor.reshape(dim, dim)



class Operation:
    """Base class for any quantum operation or observable.

    Further gates should inherit from this class to realise more specific
    operations.  Generally, operations are created by instantiation inside a
    circuit function passed to :class:`Script`; the instance is
    automatically appended to the active tape.

    An ``Operation`` can also serve as an *observable*: its matrix is used to
    compute expectation values via ``apply_to_state`` / ``apply_to_density``.

    Attributes:
        _matrix: Class-level default gate matrix.  Subclasses set this to their
            fixed unitary.  Instances may override it via the *matrix* argument
            to ``__init__``.
        _num_wires: Expected number of wires for this gate.  Subclasses set
            this to enforce wire count validation.  ``None`` means any number
            of wires is accepted.
        _param_names: Tuple of attribute names for the gate parameters.
            Used by :attr:`parameters` and :meth:`__repr__`.
    """

    # Subclasses should set this to the gate's unitary / matrix
    # Whether this is a controlled operation
    is_controlled = False
    # Whether this gate is a Clifford gate (normalises the Pauli group
    is_clifford = False

    _matrix: jnp.ndarray = None
    _num_wires: Optional[int] = None
    _param_names: Tuple[str, ...] = ()

    def __init__(
        self,
        wires: Union[int, List[int]] = 0,
        matrix: Optional[jnp.ndarray] = None,
        record: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """Initialise the operation and optionally register it on the active tape.

        Args:
            wires: Qubit index or list of qubit indices this operation acts on.
            matrix: Optional explicit gate matrix.  When provided it overrides
                the class-level ``_matrix`` attribute.
            record: If ``True`` (default) and a tape is currently recording,
                append this operation to the tape.  Set to ``False`` for
                auxiliary objects that should not appear in the circuit
                (e.g. Hamiltonians used only to build time-dependent
                evolutions).
            name: Optional explicit name for this operation.  When ``None``
                (default), the class name is used (e.g. ``"RX"``).

        Raises:
            ValueError: If ``_num_wires`` is set and the number of wires
                doesn't match, or if duplicate wires are provided.
        """
        self.name = name or self.__class__.__name__
        self.wires = list(wires) if isinstance(wires, (list, tuple)) else [wires]

        if self._num_wires is not None and len(self.wires) != self._num_wires:
            raise ValueError(
                f"{self.name} expects {self._num_wires} wire(s), "
                f"got {len(self.wires)}: {self.wires}"
            )
        if len(self.wires) != len(set(self.wires)):
            raise ValueError(f"{self.name} received duplicate wires: {self.wires}")

        if matrix is not None:
            self._matrix = matrix

        # If a tape is currently recording, append ourselves
        if record:
            tape = active_tape()
            if tape is not None:
                tape.append(self)

    @property
    def parameters(self) -> list:
        """Return the list of numeric parameters for this operation.

        Uses the declarative ``_param_names`` tuple to collect parameter
        values in a canonical order.  Non-parametrized gates return an
        empty list.

        Returns:
            List of parameter values (floats or JAX arrays).
        """
        return [getattr(self, name) for name in self._param_names]

    def __repr__(self) -> str:
        """Return a human-readable representation of this operation.

        Returns:
            A string like ``"RX(0.5000, wires=[0])"`` or ``"CX(wires=[0, 1])"``.
        """
        params = self.parameters
        if params:
            param_str = ", ".join(
                (
                    f"{float(v):.4f}"
                    if isinstance(v, (float, np.floating, jnp.ndarray))
                    else str(v)
                )
                for v in params
            )
            return f"{self.name}({param_str}, wires={self.wires})"
        return f"{self.name}(wires={self.wires})"

    @property
    def matrix(self) -> jnp.ndarray:
        """Return the base matrix of this operation (before lifting).

        Returns:
            The gate matrix as a JAX array.

        Raises:
            NotImplementedError: If the subclass has not defined ``_matrix``.
        """
        if self._matrix is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not define a matrix."
            )
        return self._matrix

    def decompose(self) -> List["Operation"]:
        """Decompose this operation into a list of more primitive operations.

        The returned operations are created with ``record=False`` so the caller
        controls where they are placed.  Used e.g. by Pauli-Clifford transforms to
        express composite gates in terms of Clifford + Pauli-rotation primitives.

        Returns:
            List of :class:`Operation` instances equivalent to this gate.

        Raises:
            NotImplementedError: If the gate has no decomposition (it is itself
                primitive).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not define a decomposition."
        )

    @property
    def wires(self) -> List[int]:
        """Qubit indices this operation acts on.

        Returns:
            List of integer qubit indices.
        """
        return self._wires

    @wires.setter
    def wires(self, wires: Union[int, List[int]]) -> None:
        """Set the qubit indices for this operation.

        Args:
            wires: A single qubit index or a list of qubit indices.
        """
        if isinstance(wires, (list, tuple)):
            self._wires = list(wires)
        else:
            self._wires = [wires]

    def _update_tape_operation(self, op: "Operation") -> None:
        """
        If ``self`` is already on the active tape (the typical case when
        chaining ``Gate(...).dagger()``), it is replaced by the daggered
        operation so that only U\\dagger appears on the tape —
        not both U and ``U\\dagger``.
        Note that this should only be called immediately after the tape is updated.s

        Args:
            op (Operation): New replaced operation on the tape
        """
        # If self was recorded on the tape, replace it with the daggered op.
        tape = active_tape()
        if tape is not None:
            if tape and tape[-1] is self:
                tape[-1] = op
            else:
                tape.append(op)

    def dagger(self) -> "Operation":
        """Return a new operation, the conjugate transpose (``U\\dagger``)
        Usage inside a circuit function::

            RX(0.5, wires=0).dagger()

        Returns:
            A new :class:`Operation` with matrix ``U\\dagger`` acting on the same wires.
        """
        mat = jnp.conj(self._matrix).T
        op = Operation(wires=self.wires, matrix=mat, record=False)

        self._update_tape_operation(op)

        return op

    def power(self, power) -> "Operation":
        """Return a new operation, the power (``U^power``)
        Usage inside a circuit function::

            PauliX(wires=0).power(2)

        Returns:
            A new :class:`Operation` with matrix ``U\\dagger`` acting on the same wires.
        """
        # TODO: support fractional powers
        mat = jnp.linalg.matrix_power(self._matrix, power)
        op = Operation(wires=self.wires, matrix=mat, record=False)

        self._update_tape_operation(op)

        return op

    def __mul__(self, other: Union[float, "Operation"]) -> "Operation":
        """Return a new operation, the product between U and a scalar (``U*x``)
        or the composition of two operations.
        Usage inside a circuit function::

            PauliX(wires=0) * x
            PauliX(wires=0) * PauliZ(wires=0)

        Returns:
            A new :class:`Operation` with matrix ``U*x`` acting on the same wires,
            or the composed matrix acting on the appropriate wires.
        """
        if isinstance(other, Operation):
            return self.__matmul__(other)

        mat = other * self._matrix
        op = Operation(wires=self.wires, matrix=mat, record=False)

        self._update_tape_operation(op)

        return op

    # Also overwrite * for right operands
    __rmul__ = __mul__

    def __add__(self, other: "Operation") -> "Operation":
        """Element-wise addition of two operations on the same wires.

        Returns:
            A new :class:`Operation` whose matrix is the sum of both matrices.

        Raises:
            ValueError: If the wire sets differ.
        """
        if sorted(self.wires) != sorted(other.wires):
            raise ValueError(
                f"Can only add operations acting on the same set of wires, "
                f"got {self.wires} and {other.wires}"
            )

        op = Operation(
            wires=self.wires,
            matrix=self.matrix + other.matrix,
            record=False,
        )
        return op

    def prod(self, *ops: "Operation") -> "Operation":
        """Construct the generalized product (tensor or matrix)
        of this operation with others.

        The resulting operation acts on the union of all wire sets.
        If the wire sets are disjoint, this is a Kronecker product.
        If the wire sets overlap, the corresponding matrices are multiplied.

        Usage::

            res = op1.prod(op2, op3)
            # or
            res = Operation.prod(op1, op2, op3)

        Args:
            *ops: Variable number of :class:`Operation` instances.

        Returns:
            A new :class:`Operation` representing the composed operation.
        """
        if not ops:
            return self

        all_ops = (self,) + ops
        all_wires = []
        for op in all_ops:
            for w in op.wires:
                if w not in all_wires:
                    all_wires.append(w)

        n = len(all_wires)

        mat = _embed_matrix(all_ops[0].matrix, all_ops[0].wires, all_wires, n)
        for op in all_ops[1:]:
            mat_other = _embed_matrix(op.matrix, op.wires, all_wires, n)
            mat = mat @ mat_other

        op_names = "*".join(op.name for op in all_ops)
        return Operation(
            wires=all_wires, matrix=mat, name=f"Prod({op_names})", record=False
        )

    def __matmul__(self, other: "Operation") -> "Operation":
        """Tensor (Kronecker) product or matrix product of two operations.

        The resulting operation acts on the union of both wire sets.
        If the wire sets are disjoint, this is a Kronecker product.
        If the wire sets overlap, the corresponding matrices are multiplied.

        Returns:
            A new :class:`Operation` whose matrix represents the composed
            operation on the unified wire set.
        """
        if not isinstance(other, Operation):
            return NotImplemented

        return self.prod(other)

    def lifted_matrix(self, n_qubits: int) -> jnp.ndarray:
        """Return the full ``2**n x 2**n`` matrix embedding this gate.

        Embeds the ``k``-qubit gate matrix into the ``n``-qubit Hilbert space
        by applying it to the identity matrix via :meth:`apply_to_state`.
        This is useful for computing ``Tr(O·\\rho )`` directly without vmap.

        Args:
            n_qubits: Total number of qubits in the circuit.

        Returns:
            The ``(2**n, 2**n)`` matrix of this operation in the full space.
        """
        dim = 2**n_qubits
        # Apply the gate to each basis vector (column of identity)
        return jax.vmap(lambda col: self.apply_to_state(col, n_qubits))(
            jnp.eye(dim, dtype=cdtype())
        ).T

    def apply_to_state(self, state: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Apply this gate to a statevector via tensor contraction.

        The statevector (shape ``(2**n,)``) is reshaped into a rank-n tensor
        of shape ``(2,)*n``.  The gate (shape ``(2**k, 2**k)``) is reshaped to
        ``(2,)*2k`` and contracted against the k target wire axes.

        Memory footprint is O(2**n) and the operation supports arbitrary k.
        The implementation is fully differentiable through JAX.

        Args:
            state: Statevector of shape ``(2**n_qubits,)``.
            n_qubits: Total number of qubits in the circuit.

        Returns:
            Updated statevector of shape ``(2**n_qubits,)``.
        """
        k = len(self.wires)
        gate_tensor = self.matrix.reshape((2,) * 2 * k)
        psi = state.reshape((2,) * n_qubits)
        psi_out = _contract_and_restore(psi, gate_tensor, k, self.wires)
        return psi_out.reshape(2**n_qubits)

    def apply_to_state_tensor(self, psi: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Apply this gate to a statevector already in tensor form.

        Like :meth:`apply_to_state` but expects the state in rank-n tensor
        form ``(2,)*n`` and returns the result in the same form.  This avoids
        the ``reshape`` calls at the per-gate level when the simulation loop
        keeps the state in tensor form throughout.

        Args:
            psi: Statevector tensor of shape ``(2,)*n_qubits``.
            n_qubits: Total number of qubits in the circuit.

        Returns:
            Updated statevector tensor of shape ``(2,)*n_qubits``.
        """
        k = len(self.wires)
        gate_tensor = self._gate_tensor(k)
        return _contract_and_restore(psi, gate_tensor, k, self.wires)

    def _gate_tensor(self, k: int) -> jnp.ndarray:
        """Return the gate matrix reshaped to ``(2,)*2k`` tensor form.

        The result is cached on the instance so repeated calls (e.g. from
        density-matrix simulation which applies U and U*) avoid redundant
        reshape dispatch.

        Args:
            k: Number of qubits the gate acts on.

        Returns:
            Gate matrix as a rank-2k tensor of shape ``(2,)*2k``.
        """
        cached = getattr(self, "_cached_gate_tensor", None)
        if cached is not None:
            return cached
        gt = self.matrix.reshape((2,) * 2 * k)
        # Only cache for non-parametrized gates (whose matrix is a class attr)
        if self._matrix is self.__class__._matrix:
            object.__setattr__(self, "_cached_gate_tensor", gt)
        return gt

    def apply_to_density(self, rho: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Apply this gate to a density matrix via \\rho -> U\\rho U\\dagger.

        The density matrix (shape ``(2**n, 2**n)``) is treated as a rank-*2n*
        tensor with n "ket" axes (0..n-1) and n "bra" axes (n..2n-1).
        U acts on the ket half; U* acts on the bra half.  Both contractions
        use the shared :func:`_contract_and_restore` helper, keeping the
        operation allocation-free with respect to building full unitaries.

        Args:
            rho: Density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
            n_qubits: Total number of qubits in the circuit.

        Returns:
            Updated density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
        """
        k = len(self.wires)
        U = self._gate_tensor(k)
        U_conj = jnp.conj(U)

        rho_t = rho.reshape((2,) * 2 * n_qubits)

        # Apply U to ket axes, U\\dagger to bra axes
        rho_t = _contract_and_restore(rho_t, U, k, self.wires)
        bra_wires = [w + n_qubits for w in self.wires]
        rho_t = _contract_and_restore(rho_t, U_conj, k, bra_wires)

        return rho_t.reshape(2**n_qubits, 2**n_qubits)



class Hermitian(Operation):
    """A generic Hermitian observable or gate defined by an arbitrary matrix.

    Example:
        >>> obs = Hermitian(matrix=my_matrix, wires=0)
    """

    def __init__(
        self,
        matrix: jnp.ndarray,
        wires: Union[int, List[int]] = 0,
        record: bool = True,
    ) -> None:
        """Initialise a Hermitian operator.

        Args:
            matrix: The Hermitian matrix defining this operator.
            wires: Qubit index or list of qubit indices this operator acts on.
            record: If ``True`` (default), record on the active tape.  Set to
                ``False`` when using the Hermitian purely as a Hamiltonian
                component (e.g. for time-dependent evolution).
        """
        super().__init__(
            wires=wires,
            matrix=jnp.asarray(matrix, dtype=cdtype()),
            record=record,
        )

    def __rmul__(self, coeff_fn: Callable) -> "ParametrizedHamiltonian":
        """Support ``coeff_fn * Hermitian`` -> :class:`ParametrizedHamiltonian`.

        Args:
            coeff_fn (Callable): A callable ``(params, t) -> scalar`` giving the
                time-dependent coefficient.

        Returns:
            ParametrizedHamiltonian: A :class:`ParametrizedHamiltonian` pairing
                *coeff_fn* with this operator's matrix and wires.

        Raises:
            TypeError: If *coeff_fn* is not callable.
        """
        if not callable(coeff_fn):
            raise TypeError(
                f"Left operand of `* Hermitian` must be callable, got {type(coeff_fn)}"
            )
        return ParametrizedHamiltonian(terms=[(coeff_fn, self.matrix, self.wires)])

    def evolve(self, name: Optional[str] = None, **odeint_kwargs) -> Callable:
        """Return a gate factory for static evolution ``U = exp(-i t H)``.

        Thin delegator to :meth:`jaqsi.evolution.Evolution.evolve`.

        Args:
            name: Optional name for the produced :class:`Operation`.
            **odeint_kwargs: Unused for static evolution (accepted for a
                uniform signature with :meth:`ParametrizedHamiltonian.evolve`).

        Returns:
            A callable gate factory ``(t, wires=0) -> Operation``.
        """
        from jaqsi.evolution import Evolution  # deferred: circular import

        return Evolution.evolve(self, name=name, **odeint_kwargs)


class ParametrizedHamiltonian:
    """A time-dependent Hamiltonian as a sum of ``coeff * Hermitian`` terms.

    Mathematically::

        H(t) = \\sum_i f_i(params_i, t) * H_i

    Construction is always done from an explicit list of
    ``(coeff_fn, H_mat, wires)`` triples passed as ``terms``.  The
    common single-term shorthand is the operator form
    ``coeff_fn * Hermitian(matrix, wires)`` (see
    :meth:`Hermitian.__rmul__`), which returns a one-term instance.
    Multi-term Hamiltonians are composed with ``+`` between
    :class:`ParametrizedHamiltonian` instances::

        H1 = coeff_x * Hermitian(X, wires=0)
        H2 = coeff_y * Hermitian(Y, wires=0)
        H_td = H1 + H2

        # evolve under the composite Hamiltonian; coeff_args is a list of
        # parameter sets, one per term, in the order the terms were added:
        H_td.evolve()([px, py], T=1.0)

    Attributes:
        coeff_fns: Tuple of callables ``(params, t) -> scalar``, one per term.
        H_mats: Tuple of static Hermitian matrices, one per term.
        wires: Wires this Hamiltonian acts on (union across all terms; for
            now all terms are required to share the same wire set).
    """

    def __init__(
        self,
        terms: List[Tuple[Callable, jnp.ndarray, Union[int, List[int]]]],
    ) -> None:
        """Build a (possibly multi-term) parametrized Hamiltonian.

        Args:
            terms: List of ``(coeff_fn, H_mat, wires)`` triples.  Use the
                ``coeff_fn * Hermitian(...)`` shorthand to build a
                one-term instance; combine instances with ``+`` to add
                terms.

        Raises:
            ValueError: If the term list is empty, or if terms act on
                differing wire sets (multi-wire broadcasting is
                deferred — see :mod:`jaqsi`), or if term matrices have
                incompatible shapes.
        """
        if len(terms) == 0:
            raise ValueError("ParametrizedHamiltonian needs at least one term.")

        # Normalise wires (single int -> [int]) and validate consistency.
        def _wlist(w):
            return [w] if isinstance(w, int) else list(w)

        first_wires = _wlist(terms[0][2])
        for _, _, w in terms[1:]:
            if _wlist(w) != first_wires:
                raise ValueError(
                    "All terms of a ParametrizedHamiltonian must currently "
                    "act on the same wires; got "
                    f"{_wlist(w)} vs. {first_wires}. "
                    "Multi-wire broadcasting across terms is not yet supported."
                )

        # Validate matrix shape compatibility across terms.
        first_dim = jnp.asarray(terms[0][1]).shape
        for _, H, _ in terms[1:]:
            if jnp.asarray(H).shape != first_dim:
                raise ValueError(
                    f"All term matrices must have the same shape; got "
                    f"{jnp.asarray(H).shape} vs. {first_dim}."
                )

        self._terms: Tuple[Tuple[Callable, jnp.ndarray, List[int]], ...] = tuple(
            (fn, jnp.asarray(H, dtype=cdtype()), _wlist(w)) for fn, H, w in terms
        )
        self.wires: List[int] = list(first_wires)

    # --- term accessors -------------------------------------------------

    @property
    def coeff_fns(self) -> Tuple[Callable, ...]:
        """Tuple of coefficient functions, one per term."""
        return tuple(fn for fn, _, _ in self._terms)

    @property
    def H_mats(self) -> Tuple[jnp.ndarray, ...]:
        """Tuple of Hermitian matrices, one per term."""
        return tuple(H for _, H, _ in self._terms)

    @property
    def n_terms(self) -> int:
        """Number of terms in the Hamiltonian."""
        return len(self._terms)

    # --- composition ---------------------------------------------------

    def __add__(self, other: "ParametrizedHamiltonian") -> "ParametrizedHamiltonian":
        """Concatenate term lists: ``H = H1 + H2``."""
        if not isinstance(other, ParametrizedHamiltonian):
            return NotImplemented
        return ParametrizedHamiltonian(terms=list(self._terms) + list(other._terms))

    def __neg__(self) -> "ParametrizedHamiltonian":
        """Negate every coefficient: ``-H`` = sum of ``(-f_i) * H_i``."""
        new_terms = [
            ((lambda f: lambda p, t: -f(p, t))(fn), H, w) for fn, H, w in self._terms
        ]
        return ParametrizedHamiltonian(terms=new_terms)

    def __sub__(self, other: "ParametrizedHamiltonian") -> "ParametrizedHamiltonian":
        if not isinstance(other, ParametrizedHamiltonian):
            return NotImplemented
        return self + (-other)

    # --- evolution -----------------------------------------------------

    def evolve(self, name: Optional[str] = None, **odeint_kwargs) -> Callable:
        """Return a gate factory for time-dependent evolution.

        Solves ``dU/dt = -i [sum_i f_i(p_i, t) H_i] U``.  Thin delegator to
        :meth:`jaqsi.evolution.Evolution.evolve`.

        Args:
            name: Optional name for the produced :class:`Operation`.
            **odeint_kwargs: Solver options forwarded to ``Evolution.evolve``
                (``atol``, ``rtol``, ``max_steps``, ``throw``, ``solver``,
                ``magnus_steps``).

        Returns:
            A callable gate factory ``(coeff_args, T) -> Operation``.
        """
        from jaqsi.evolution import Evolution  # deferred: circular import

        return Evolution.evolve(self, name=name, **odeint_kwargs)


def Hamiltonian(
    matrix: jnp.ndarray,
    wires: Union[int, List[int]] = 0,
    record: bool = False,
) -> Hermitian:
    """Construct a (static) Hamiltonian as a :class:`Hermitian` operator.

    This is a thin factory over the existing :class:`Hermitian` operation —
    not a new type.  Multiply it by a coefficient function ``f(params, t)`` to
    obtain a time-dependent :class:`ParametrizedHamiltonian`.  Both expose an
    :meth:`evolve` method that returns a gate factory.

    Args:
        matrix: The Hermitian matrix defining this Hamiltonian.
        wires: Qubit index or list of qubit indices it acts on.
        record: Whether to record on the active tape.  Defaults to ``False``
            since a Hamiltonian used as an evolution source should not appear
            as a gate; the recorded operation is the one produced by
            :meth:`evolve`.

    Returns:
        A :class:`Hermitian` instance.
    """
    return Hermitian(matrix, wires=wires, record=record)
