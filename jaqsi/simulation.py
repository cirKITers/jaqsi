"""Pure simulation and measurement kernels for :class:`~jaqsi.script.Script`.

These functions are stateless: they take a recorded tape (a list of
:class:`~jaqsi.operations.Operation`) plus measurement parameters and
return JAX arrays.  Keeping them as module-level free functions (rather than
static methods on ``Script``) makes the simulation engine independently testable
and keeps ``script.py`` focused on orchestration.
"""

import itertools
import string
from functools import lru_cache
from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np  # needed to prevent jitting some operations

from jaqsi.evolution import resolve_pending
from jaqsi.operations import (
    Operation,
    GateStructure,
    _einsum_subscript,
    cdtype,
)
from jaqsi.gateset import (
    Barrier,
)
from jaqsi.noise import KrausChannel


def infer_n_qubits(ops: List[Operation], obs: List[Operation]) -> int:
    """Infer the number of qubits from a list of operations and observables.

    Args:
        ops: Gate operations recorded on the tape.
        obs: Observable operations used for measurement.

    Returns:
        The smallest number of qubits that covers all wire indices, i.e.
        ``max(all_wires) + 1`` (at least 1).
    """
    all_wires: set[int] = set()
    for op in ops + obs:
        all_wires.update(op.wires)
    return max(all_wires) + 1 if all_wires else 1


def has_noise(tape: List[Operation]) -> bool:
    """Return whether *tape* contains a noise channel.

    Density-matrix simulation is required exactly in that case; a
    ``"density"`` output of a noise-free circuit is formed from the
    statevector by :func:`measure_state`.

    Args:
        tape: Ordered list of gate/channel operations.

    Returns:
        ``True`` if any operation is a :class:`~jaqsi.noise.KrausChannel`.
    """
    return any(isinstance(op, KrausChannel) for op in tape)


def _stack_obs(obs: List[Operation], n_qubits: int) -> jnp.ndarray:
    """Stack lifted observable matrices into a single ``(n_obs, dim, dim)`` array."""
    return jnp.stack([ob.lifted_matrix(n_qubits) for ob in obs], axis=0)


def _apply_gate(
    psi: jnp.ndarray, gate: jnp.ndarray, wires: Tuple[int, ...]
) -> jnp.ndarray:
    """Apply a k-qubit gate matrix to a rank-n state tensor along *wires*.

    Each output slice along the gate wires is a linear combination of the
    ``2**k`` input slices, which XLA compiles into one fused elementwise loop
    over the state.  The equivalent ``einsum`` lowers to a transpose of the
    whole state plus a dot with a ``2**k``-wide contraction per gate; on CPU
    that is bandwidth-bound and about twice as slow for one- and two-qubit
    gates.  Wider gates keep the ``einsum`` path, where the contraction is
    large enough to amortise the transpose.

    Args:
        psi: State tensor of shape ``(2,) * n``.
        gate: Gate matrix of shape ``(2**k, 2**k)``.
        wires: The k axes of *psi* the gate acts on.

    Returns:
        Updated state tensor of shape ``(2,) * n``.
    """
    k = len(wires)
    if k > 2:
        gate = gate.reshape((2,) * 2 * k)
        return jnp.einsum(_einsum_subscript(psi.ndim, k, wires), gate, psi)

    g = gate
    parts = []
    for bits in itertools.product((0, 1), repeat=k):
        index: List[Union[int, slice]] = [slice(None)] * psi.ndim
        for w, b in zip(wires, bits):
            index[w] = b
        parts.append(psi[tuple(index)])

    outs = []
    for i in range(2**k):
        acc = g[i, 0] * parts[0]
        for j in range(1, 2**k):
            acc = acc + g[i, j] * parts[j]
        outs.append(acc)

    out = jnp.stack(outs).reshape((2,) * k + parts[0].shape)
    return jnp.moveaxis(out, tuple(range(k)), wires)


class Gate(NamedTuple):
    """A matrix and its static application metadata."""

    matrix: jnp.ndarray
    wires: Tuple[int, ...]
    structure: GateStructure = GateStructure()


class Permutation(NamedTuple):
    """A composed permutation of the computational basis.

    ``index`` holds, for every amplitude of the result, the flat index of the
    amplitude it is taken from; ``inverse`` is the same map for ``U†``.  Both
    are static, so a permutation carries no cotangent.
    """

    index: np.ndarray
    inverse: np.ndarray


class Diagonal(NamedTuple):
    """The composed diagonal of consecutive diagonal gates.

    ``diagonal`` is a rank-``len(wires)`` tensor of shape ``(2,) * k`` over the
    ascending *wires* the gates act on, so it is the diagonal of their product
    in the basis of those wires.
    """

    diagonal: jnp.ndarray
    wires: Tuple[int, ...]


Entry = Union[Gate, Permutation, Diagonal]


def _apply_permutation(psi: jnp.ndarray, index: np.ndarray) -> jnp.ndarray:
    """Gather the amplitudes of a composed basis permutation in one pass."""
    return jnp.take(psi.reshape(-1), index, unique_indices=True).reshape(psi.shape)


def _apply_diagonal(
    psi: jnp.ndarray, diagonal: jnp.ndarray, wires: Tuple[int, ...]
) -> jnp.ndarray:
    """Multiply a rank-n state tensor by a k-qubit diagonal along *wires*."""
    k = len(wires)
    broadcast = diagonal.reshape((2,) * k + (1,) * (psi.ndim - k))
    return psi * jnp.moveaxis(broadcast, tuple(range(k)), wires)


def _apply_structured(
    psi: jnp.ndarray,
    gate: jnp.ndarray,
    wires: Tuple[int, ...],
    structure: GateStructure,
) -> jnp.ndarray:
    """Apply guaranteed structure without inspecting traced matrix values."""
    k = len(wires)
    if structure.permutation:
        parts = []
        for column in structure.permutation:
            index: List[Union[int, slice]] = [slice(None)] * psi.ndim
            for i, w in enumerate(wires):
                index[w] = (column >> (k - 1 - i)) & 1
            parts.append(psi[tuple(index)])
        out = jnp.stack(parts).reshape((2,) * k + parts[0].shape)
        return jnp.moveaxis(out, tuple(range(k)), wires)
    if structure.diagonal:
        return _apply_diagonal(psi, jnp.diag(gate), wires)
    if structure.controls:
        control = wires[0]
        index: List[Union[int, slice]] = [slice(None)] * psi.ndim
        index[control] = 0
        inactive = psi[tuple(index)]
        index[control] = 1
        active = psi[tuple(index)]
        remaining = tuple(w - (w > control) for w in wires[1:])
        half = gate.shape[0] // 2
        active = _apply_structured(
            active,
            gate[half:, half:],
            remaining,
            structure._replace(controls=structure.controls - 1),
        )
        return jnp.stack((inactive, active), axis=control)
    return _apply_gate(psi, gate, wires)


def _fused_structure(left: GateStructure, right: GateStructure) -> GateStructure:
    """Guarantees retained by the product right @ left on identical wires."""
    permutation = ()
    if left.permutation and right.permutation:
        permutation = tuple(left.permutation[i] for i in right.permutation)
    return GateStructure(
        permutation=permutation,
        diagonal=left.diagonal and right.diagonal,
        controls=min(left.controls, right.controls),
    )


def _fuse(gates: List[Gate]) -> List[Gate]:
    """Merge runs of gates on identical wires into single matrices.

    A gate is multiplied into the previous gate on the same wire tuple when no
    gate in between touches any of those wires (gates on disjoint wires
    commute).  Each fused block then costs one pass over the state instead of
    one per gate.  The small matrix products are part of the trace, so
    differentiation through the block parameters is unchanged.

    Args:
        gates: Compiled gates in tape order.

    Returns:
        Fused gates in tape order.
    """
    fused: List[list] = []
    last: dict = {}  # wire -> index in ``fused`` of the last block touching it
    for mat, wires, structure in gates:
        i = last.get(wires[0])
        if i is not None and fused[i][1] == wires and all(last[w] == i for w in wires):
            fused[i][0] = mat @ fused[i][0]
            fused[i][2] = _fused_structure(fused[i][2], structure)
            continue
        fused.append([mat, wires, structure])
        for w in wires:
            last[w] = len(fused) - 1
    return [Gate(mat, wires, structure) for mat, wires, structure in fused]


def _compose_permutation(blocks: List[Gate], n_qubits: int) -> Permutation:
    """Compose the basis permutations of *blocks* into one index map.

    The source index of every amplitude is tracked through the blocks in tape
    order with the same slice-and-stack pattern :func:`_apply_structured`
    applies to the state, so the composition assumes nothing about the blocks
    commuting.  The result is a static ``2**n`` int32 array, a quarter of one
    statevector, whatever the length of the run.
    """
    index = np.arange(2**n_qubits, dtype=np.int64).reshape((2,) * n_qubits)
    for _, wires, structure in blocks:
        k = len(wires)
        parts = []
        for column in structure.permutation:
            selector: List[Union[int, slice]] = [slice(None)] * n_qubits
            for i, w in enumerate(wires):
                selector[w] = (column >> (k - 1 - i)) & 1
            parts.append(index[tuple(selector)])
        index = np.moveaxis(
            np.stack(parts).reshape((2,) * k + parts[0].shape), tuple(range(k)), wires
        )
    index = index.reshape(-1).astype(np.int32)
    inverse = np.empty_like(index)
    inverse[index] = np.arange(index.size, dtype=np.int32)
    return Permutation(index, inverse)


def _compose_diagonal(blocks: List[Gate]) -> Diagonal:
    """Multiply the diagonals of *blocks* over their union of wires.

    Diagonals are elementwise in the computational basis, so the product is
    formed by broadcasting each block's diagonal onto the union axes.  The
    factors stay in the trace, which keeps the cotangent chain to the block
    parameters intact.
    """
    wires = tuple(sorted({w for block in blocks for w in block.wires}))
    axis = {w: i for i, w in enumerate(wires)}
    diagonal = jnp.ones((2,) * len(wires), dtype=cdtype())
    for matrix, block_wires, _ in blocks:
        k = len(block_wires)
        factor = jnp.diag(matrix).reshape((2,) * k + (1,) * (len(wires) - k))
        diagonal = diagonal * jnp.moveaxis(
            factor, tuple(range(k)), tuple(axis[w] for w in block_wires)
        )
    return Diagonal(diagonal, wires)


def _fuse_runs(gates: List[Gate], n_qubits: int) -> List[Entry]:
    """Merge runs of same-kind structured blocks into one pass each.

    :func:`_fuse` only merges blocks that share their wires.  Consecutive
    permutations compose into a single basis-index gather and consecutive
    diagonals into a single elementwise factor on *any* wires, which turns a
    permutation or diagonal layer into one pass over the state instead of one
    per gate.  A run of a single block is left as it is: composing it would
    cost the same pass and, for a permutation, a ``2**n`` index array.

    Args:
        gates: Same-wire-fused blocks in tape order.
        n_qubits: Total number of qubits.

    Returns:
        The plan :func:`_run` applies, in tape order.
    """
    plan: List[Entry] = []
    run: List[Gate] = []
    kind: Optional[str] = None

    def flush() -> None:
        if len(run) > 1:
            plan.append(
                _compose_permutation(run, n_qubits)
                if kind == "permutation"
                else _compose_diagonal(run)
            )
        else:
            plan.extend(run)
        run.clear()

    for gate in gates:
        gate_kind = (
            "permutation"
            if gate.structure.permutation
            else "diagonal"
            if gate.structure.diagonal
            else None
        )
        if gate_kind != kind:
            flush()
            kind = gate_kind
        (plan if gate_kind is None else run).append(gate)
    flush()
    return plan


def _compile(tape: List[Operation], n_qubits: int) -> List[Entry]:
    """Extract matrices with static structure, fuse same-wire neighbours and runs."""
    gates = [
        Gate(op.matrix, tuple(op.wires), op._structure)
        for op in tape
        if not isinstance(op, Barrier)
    ]
    return _fuse_runs(_fuse(gates), n_qubits)


def _compile_mixed(tape: List[Operation], n_qubits: int) -> List[Gate]:
    """Pre-extract superoperators on the density tensor's axes and fuse them.

    The density matrix is treated as a rank-``2n`` tensor with ket axes
    ``0..n-1`` and bra axes ``n..2n-1``.  A gate on up to two qubits becomes
    the superoperator ``U (x) U*`` on its ket and bra axes, a channel becomes
    ``sum_k K_k (x) K_k*``, so gates and channels are the same kind of entry
    and :func:`_fuse` merges them alike: one pass over the density tensor per
    block instead of two per gate and two per Kraus operator.  Wider gates
    stay two-sided (``U`` on the ket axes, ``U*`` on the bra axes), since their
    superoperator would be ``4**k`` square.

    Args:
        tape: Ordered list of gate/channel operations.
        n_qubits: Total number of qubits.

    Returns:
        Fused ``(matrix, axes)`` pairs for :func:`_run` on the rank-``2n`` tensor.
    """
    gates: List[Gate] = []
    for op in tape:
        if isinstance(op, Barrier):
            continue
        ket = tuple(op.wires)
        bra = tuple(w + n_qubits for w in op.wires)
        if isinstance(op, KrausChannel):
            superop = sum(jnp.kron(K, jnp.conj(K)) for K in op.kraus_matrices())
            gates.append(Gate(superop, ket + bra))
        elif len(ket) <= 2:
            gates.append(Gate(jnp.kron(op.matrix, jnp.conj(op.matrix)), ket + bra))
        else:
            gates.append(Gate(op.matrix, ket))
            gates.append(Gate(jnp.conj(op.matrix), bra))
    return _fuse(gates)


def _initial_state(dim: int, initial_state: Optional[jnp.ndarray]) -> jnp.ndarray:
    """Flat statevector |00…0⟩, or *initial_state* cast to the working dtype."""
    if initial_state is None:
        return jnp.zeros(dim, dtype=cdtype()).at[0].set(1.0)
    return jnp.asarray(initial_state, dtype=cdtype()).reshape(dim)


def _apply_entry(psi: jnp.ndarray, entry: Entry) -> jnp.ndarray:
    """Apply one plan entry to a rank-n state tensor."""
    if isinstance(entry, Permutation):
        return _apply_permutation(psi, entry.index)
    if isinstance(entry, Diagonal):
        return _apply_diagonal(psi, entry.diagonal, entry.wires)
    return _apply_structured(psi, entry.matrix, entry.wires, entry.structure)


def _run(plan: Sequence[Entry], state: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
    """Apply a compiled *plan* to a flat array of ``n_qubits`` binary axes.

    Density-matrix simulation passes the flattened ``(dim, dim)`` matrix with
    ``2 * n_qubits`` axes; the gates then address ket and bra axes alike.
    """
    psi = state.reshape((2,) * n_qubits)
    for entry in plan:
        psi = _apply_entry(psi, entry)
    return psi.reshape(2**n_qubits)


def simulate_pure(
    tape: List[Operation],
    n_qubits: int,
    initial_state: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Statevector simulation kernel.

    Starts from |00…0⟩ (or *initial_state* when given) and applies each gate in
    *tape* via tensor contraction.  The state is kept in rank-*n* tensor form
    ``(2,)*n`` throughout the gate loop to avoid per-gate ``reshape`` dispatch;
    only the initial and final conversions to/from the flat ``(2**n,)``
    representation incur a reshape.

    Gate matrices and wire tuples are pre-extracted from the tape, neighbouring
    gates on identical wires are fused and runs of permutations or diagonals are
    composed (see :func:`_compile`), so each loop iteration is a single pass
    over the state.

    Args:
        tape: Ordered list of gate operations to apply.
        n_qubits: Total number of qubits.
        initial_state: Optional statevector of shape ``(2**n_qubits,)`` to start
            from.  When ``None`` (default), the all-zero state |00…0⟩ is used.

    Returns:
        Statevector of shape ``(2**n_qubits,)``.
    """
    state = _initial_state(2**n_qubits, initial_state)
    return _run(_compile(tape, n_qubits), state, n_qubits)


def simulate_mixed(
    tape: List[Operation],
    n_qubits: int,
    initial_state: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Density-matrix simulation kernel.

    Starts from \\rho  = \\vert 0\\rangle\\langle 0\\vert (or from
    \\rho  = \\vert\\psi\\rangle\\langle\\psi\\vert for a given *initial_state*
    \\vert\\psi\\rangle) and applies the fused superoperators of *tape*
    (see :func:`_compile_mixed`) to the density matrix kept as a rank-``2n``
    tensor: \\rho  -> U\\rho U† for unitaries, \\Sigma_k K_k \\rho  K_k\\dagger
    for Kraus channels.  Required for noisy circuits.

    Args:
        tape: Ordered list of gate or channel operations to apply.
        n_qubits: Total number of qubits.
        initial_state: Optional statevector of shape ``(2**n_qubits,)`` to start
            from.  When ``None`` (default), the all-zero state |00…0⟩ is used.

    Returns:
        Density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
    """
    dim = 2**n_qubits
    if initial_state is None:
        rho = jnp.zeros((dim, dim), dtype=cdtype()).at[0, 0].set(1.0)
    else:
        psi = jnp.asarray(initial_state, dtype=cdtype()).reshape(dim)
        rho = jnp.outer(psi, jnp.conj(psi))
    gates = _compile_mixed(tape, n_qubits)
    return _run(gates, rho.reshape(dim * dim), 2 * n_qubits).reshape(dim, dim)


@lru_cache(maxsize=256)
def _outer_subscript(n: int, wires: Tuple[int, ...]) -> str:
    """``einsum`` subscript contracting two rank-n tensors over all axes but *wires*.

    ``_outer_subscript(3, (1,))`` gives ``"adc,aec->de"``; the result
    ``M[o, i] = sum_rest lam[o, rest] * psi[i, rest]`` is the cotangent of a
    gate matrix on *wires* in JAX's transpose convention (no conjugation).
    """
    letters = string.ascii_letters
    k = len(wires)
    lam_idx, psi_idx = list(letters[:n]), list(letters[:n])
    for j, w in enumerate(wires):
        lam_idx[w], psi_idx[w] = letters[n + j], letters[n + k + j]
    return f"{''.join(lam_idx)},{''.join(psi_idx)}->{letters[n : n + 2 * k]}"


def _forward_mode(leaves) -> bool:
    """Whether a forward-mode (``jvp``) trace reaches any of *leaves*.

    ``jax.custom_vjp`` has no forward-mode rule, so the adjoint path must be
    skipped under ``jax.jvp``/``jax.jacfwd``.  Reverse mode (``jax.grad``,
    ``jax.jacrev``) traces with ``LinearizeTracer`` since direct linearization
    became JAX's default, forward mode with ``JVPTracer``.  Walking a leaf's
    tracer chain from the innermost transform outwards, the first of the two
    decides; batch and jit tracers are transparent.  A misclassification only
    costs the fast path, never correctness.  Tracers of a transform applied
    outside an enclosing ``jax.jit`` are not visible here.
    """
    for leaf in leaves:
        while isinstance(leaf, jax.core.Tracer):
            kind = type(leaf).__name__
            if kind == "JVPTracer":
                return True
            if kind == "LinearizeTracer":
                break
            leaf = getattr(leaf, "primal", getattr(leaf, "val", None))
    return False


def _parameter(entry: Union[Gate, Diagonal]) -> jnp.ndarray:
    """The differentiable array of a plan entry; permutations have none."""
    return entry.diagonal if isinstance(entry, Diagonal) else entry.matrix


def _substitute(
    entry: Union[Gate, Diagonal], parameter: jnp.ndarray
) -> Union[Gate, Diagonal]:
    """*entry* with its differentiable array replaced by *parameter*."""
    if isinstance(entry, Diagonal):
        return entry._replace(diagonal=parameter)
    return entry._replace(matrix=parameter)


def _parameters(plan: Sequence[Entry]) -> List[jnp.ndarray]:
    """The differentiable arrays of *plan* in order, skipping permutations."""
    return [_parameter(e) for e in plan if not isinstance(e, Permutation)]


def _use_adjoint(tape: List[Operation], plan: Sequence[Entry], initial_state) -> bool:
    """Whether the adjoint VJP applies: every gate unitary, no forward-mode trace."""
    unitary = all(op.is_unitary for op in tape if not isinstance(op, Barrier))
    return unitary and not _forward_mode(_parameters(plan) + [initial_state])


def _adjoint_expval(
    plan: Sequence[Entry], n_qubits: int, obs: List[Operation], state: jnp.ndarray
) -> jnp.ndarray:
    """Expectation values of a pure circuit with an adjoint-method VJP.

    The forward pass is the plain gate loop.  The backward pass reads no tape
    of intermediate states: it walks the plan in reverse, undoing each entry
    with ``U^dagger`` on the state while propagating the cotangent with
    ``U^T``, so gradient memory stays at a few statevectors regardless of
    depth.  The cotangents are those of the gate *matrices* and fused
    *diagonals* (and of the initial state); JAX chains them into the gate
    parameters through the matrix construction, so no per-gate generator is
    needed.  Constant entries get a zero cotangent without the contraction,
    and a basis permutation needs none at all.

    Args:
        plan: Compiled entries from :func:`_compile`.
        n_qubits: Total number of qubits.
        obs: Observables for the expectation values.
        state: Initial flat statevector of shape ``(2**n_qubits,)``.

    Returns:
        Expectation values of shape ``(len(obs),)``.
    """
    dim = 2**n_qubits
    shape = (2,) * n_qubits
    # The differentiable arguments of the VJP, in plan order: one array per
    # entry except the static permutations.  Both loops below walk the plan and
    # advance through them in step.
    params = _parameters(plan)
    const = [
        (isinstance(entry, Gate) and bool(entry.structure.permutation))
        or not isinstance(_parameter(entry), jax.core.Tracer)
        for entry in plan
        if not isinstance(entry, Permutation)
    ]
    inverse_structures = [
        entry.structure._replace(
            permutation=tuple(
                entry.structure.permutation.index(i)
                for i in range(len(entry.structure.permutation))
            )
        )
        if isinstance(entry, Gate)
        else GateStructure()
        for entry in plan
    ]

    def run(params, psi0):
        entries: List[Entry] = []
        slot = 0
        for entry in plan:
            if isinstance(entry, Permutation):
                entries.append(entry)
                continue
            entries.append(_substitute(entry, params[slot]))
            slot += 1
        return _run(entries, psi0, n_qubits)

    def measure(psi):
        return measure_state(psi, n_qubits, "expval", obs)

    @jax.custom_vjp
    def expval(params, psi0):
        return measure(run(params, psi0))

    def fwd(params, psi0):
        psi = run(params, psi0)
        return measure(psi), (params, psi)

    def bwd(res, ct):
        params, psi = res
        lam = jax.vjp(measure, psi)[1](ct)[0].reshape(shape)
        psi = psi.reshape(shape)
        bars = []
        slot = len(params)
        for entry, inverse in zip(reversed(plan), reversed(inverse_structures)):
            if isinstance(entry, Permutation):
                psi = _apply_permutation(psi, entry.inverse)
                lam = _apply_permutation(lam, entry.inverse)
                continue
            slot -= 1
            parameter = params[slot]
            if isinstance(entry, Diagonal):
                psi = _apply_diagonal(psi, jnp.conj(parameter), entry.wires)
                # The cotangent of a diagonal is the diagonal of the matrix
                # cotangent: one reduction instead of the outer product.
                bars.append(
                    jnp.zeros_like(parameter)
                    if const[slot]
                    else jnp.sum(
                        lam * psi,
                        axis=tuple(a for a in range(psi.ndim) if a not in entry.wires),
                    )
                )
                lam = _apply_diagonal(lam, parameter, entry.wires)
                continue
            psi = _apply_structured(psi, jnp.conj(parameter).T, entry.wires, inverse)
            bars.append(
                jnp.zeros_like(parameter)
                if const[slot]
                else jnp.einsum(
                    _outer_subscript(n_qubits, entry.wires), lam, psi
                ).reshape(parameter.shape)
            )
            lam = _apply_structured(lam, parameter.T, entry.wires, inverse)
        return bars[::-1], lam.reshape(dim)

    expval.defvjp(fwd, bwd)
    return expval(params, state)


def simulate_and_measure(
    tape: List[Operation],
    n_qubits: int,
    type: str,
    obs: List[Operation],
    use_density: bool,
    shots: Optional[int] = None,
    key: Optional[jnp.ndarray] = None,
    initial_state: Optional[jnp.ndarray] = None,
    adjoint: bool = True,
) -> jnp.ndarray:
    """Run simulation and measurement in a single dispatch.

    Chooses statevector or density-matrix simulation based on
    *use_density* (a noise channel on the tape), then applies the
    appropriate measurement function.  This eliminates duplicated branching
    logic in single-sample and batched execution paths.

    When *shots* is not ``None``, the exact probability distribution is
    first computed, then ``shots`` samples are drawn from it to produce
    a noisy estimate of the requested measurement (``"probs"`` or
    ``"expval"``).

    A ``"density"`` output of a noise-free circuit is the outer product of
    the simulated statevector (see :func:`measure_state`), so the full
    ``2^n x 2^n`` matrix is never evolved gate by gate for it.

    Gradients — for exact ``"expval"`` results of a pure circuit whose gates
    are all unitary, the expectation values carry an adjoint-method VJP (see
    :func:`_adjoint_expval`) instead of relying on JAX taping every
    intermediate state.  Every other case, and forward-mode differentiation,
    uses plain JAX autodiff.

    Args:
        tape: Ordered list of gate/channel operations to apply.
        n_qubits: Total number of qubits.
        type: Measurement type (``"state"``/``"probs"``/``"expval"``/
            ``"density"``).
        obs: Observables for ``"expval"`` measurements.
        use_density: If ``True``, use density-matrix simulation (the tape
            carries a noise channel).
        shots: Number of measurement shots.  If ``None`` (default),
            exact analytic results are returned.
        key: JAX PRNG key for shot sampling.  Required when *shots*
            is not ``None``.
        initial_state: Optional statevector of shape ``(2**n_qubits,)`` to start
            from.  When ``None`` (default), the all-zero state |00…0⟩ is used.
        adjoint: Allow the adjoint VJP.  :class:`~jaqsi.script.Script` passes
            ``False`` when it detects a forward-mode trace on the arguments
            outside its own ``jit``, where :func:`_forward_mode` cannot see it.

    Returns:
        Measurement result (shape depends on *type*).
    """
    # Solve all pulse-level gates of the tape in batches before simulating.
    resolve_pending(tape)

    if use_density:
        rho = simulate_mixed(tape, n_qubits, initial_state=initial_state)
        if shots is not None and type in ("probs", "expval"):
            exact_probs = jnp.real(jnp.diag(rho))
            return sample_shots(exact_probs, n_qubits, type, obs, shots, key)
        return measure_density(rho, n_qubits, type, obs)

    plan = _compile(tape, n_qubits)
    state = _initial_state(2**n_qubits, initial_state)
    exact_expval = type == "expval" and shots is None
    if exact_expval and adjoint and _use_adjoint(tape, plan, initial_state):
        return _adjoint_expval(plan, n_qubits, obs, state)
    state = _run(plan, state, n_qubits)

    if shots is not None and type in ("probs", "expval"):
        exact_probs = jnp.abs(state) ** 2
        return sample_shots(exact_probs, n_qubits, type, obs, shots, key)
    return measure_state(state, n_qubits, type, obs)


def measure_state(
    state: jnp.ndarray,
    n_qubits: int,
    type: str,
    obs: List[Operation],
) -> jnp.ndarray:
    """Apply the requested measurement to a pure statevector.

    Args:
        state: Statevector of shape ``(2**n_qubits,)``.
        n_qubits: Total number of qubits.
        type: Measurement type — one of ``"state"``, ``"probs"``,
            ``"density"`` or ``"expval"``.
        obs: Observables used when *type* is ``"expval"``.

    Returns:
        Measurement result whose shape depends on *type*:

        - ``"state"``   -> ``(2**n_qubits,)``
        - ``"probs"``   -> ``(2**n_qubits,)``
        - ``"density"`` -> ``(2**n_qubits, 2**n_qubits)``, the outer product
          ``|psi><psi|``
        - ``"expval"``  -> ``(len(obs),)``

    Raises:
        ValueError: If *type* is not a recognised measurement type.
    """
    if type == "state":
        return state

    if type == "probs":
        return jnp.abs(state) ** 2

    if type == "density":
        return jnp.outer(state, jnp.conj(state))

    if type == "expval":
        # Fast path for single-qubit diagonal observables (PauliZ, etc.)
        # where d0, d1 are the diagonal elements of the 2x2 observable.
        # This replaces n_obs tensor contractions with a single |ψ|²
        # and n_obs reductions over the probability vector.

        def _is_single_qubit_diag(ob):
            m = ob.__class__._matrix
            if m is None or len(ob.wires) != 1:
                return False
            # Convert to NumPy to ensure concrete boolean evaluation
            m_np = np.asarray(m)
            return np.allclose(m_np - np.diag(np.diag(m_np)), 0)

        all_single_qubit_diag = all(_is_single_qubit_diag(ob) for ob in obs)

        if all_single_qubit_diag:
            probs = jnp.real(state) ** 2 + jnp.imag(state) ** 2
            indices = jnp.arange(state.size)
            results = []
            for ob in obs:
                q = ob.wires[0]
                d = np.real(np.diag(np.asarray(ob.__class__._matrix)))
                # Qubit 0 is the most significant bit. A flat weighted sum
                # avoids the layout changes of one marginal reduction per wire.
                bit = (indices >> (n_qubits - q - 1)) & 1
                weights = jnp.where(bit, d[1], d[0])
                results.append(
                    jnp.dot(probs, weights, precision=jax.lax.Precision.HIGHEST)
                )
            return jnp.array(results)

        # General path: stack observable matrices and use a single
        # batched matmul instead of a Python loop of tensor contractions.
        # O_states[i] = obs[i] |ψ⟩, then ⟨O_i⟩ = Re(⟨ψ|O_states[i]⟩).
        obs_mats = _stack_obs(obs, n_qubits)  # (n_obs, dim, dim)
        # Batched matvec: (n_obs, dim, dim) @ (dim,) -> (n_obs, dim)
        O_states = jnp.einsum("oij,j->oi", obs_mats, state)
        return jnp.real(jnp.einsum("i,oi->o", jnp.conj(state), O_states))

    raise ValueError(f"Unknown measurement type: {type!r}")


def measure_density(
    rho: jnp.ndarray,
    n_qubits: int,
    type: str,
    obs: List[Operation],
) -> jnp.ndarray:
    """Apply the requested measurement to a density matrix.

    Args:
        rho: Density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
        n_qubits: Total number of qubits.
        type: Measurement type — one of ``"density"``, ``"probs"``,
            or ``"expval"``.
        obs: Observables used when *type* is ``"expval"``.

    Returns:
        Measurement result whose shape depends on *type*:

        - ``"density"`` -> ``(2**n_qubits, 2**n_qubits)``
        - ``"probs"``   -> ``(2**n_qubits,)``
        - ``"expval"``  -> ``(len(obs),)``

    Raises:
        ValueError: If *type* is ``"state"`` (not valid for mixed circuits)
            or another unrecognised type.
    """
    if type == "density":
        return rho

    if type == "probs":
        return jnp.real(jnp.diag(rho))

    if type == "expval":
        # Tr(O \\rho ) = \\Sigma_ij O_ij \\rho _ji
        # Stack all observable matrices and compute all traces in one
        # batched operation.
        obs_mats = _stack_obs(obs, n_qubits)  # (n_obs, dim, dim)
        # einsum "oij,ji->o" computes Tr(O_o @ \\rho ) for each observable
        return jnp.real(jnp.einsum("oij,ji->o", obs_mats, rho))

    raise ValueError(
        "Measurement type 'state' is not defined for mixed (noisy) circuits. "
        "Use 'density' instead."
    )


def sample_shots(
    probs: jnp.ndarray,
    n_qubits: int,
    type: str,
    obs: List[Operation],
    shots: int,
    key: jnp.ndarray,
) -> jnp.ndarray:
    """Convert exact probabilities into shot-sampled results.

    Draws *shots* samples from the computational-basis probability
    distribution and returns either estimated probabilities or
    shot-based expectation values.

    Args:
        probs: Exact probability vector of shape ``(2**n_qubits,)``.
        n_qubits: Total number of qubits.
        type: Measurement type — ``"probs"`` or ``"expval"``.
        obs: Observables used when *type* is ``"expval"``.
        shots: Number of measurement shots.
        key: JAX PRNG key for sampling.

    Returns:
        Shot-sampled measurement result:

        - ``"probs"``  → ``(2**n_qubits,)`` estimated probabilities.
        - ``"expval"`` → ``(len(obs),)`` estimated expectation values.
    """
    dim = 2**n_qubits

    # Draw `shots` samples from the computational basis.
    # Each sample is an integer in [0, dim) representing a basis state.
    samples = jax.random.choice(key, dim, shape=(shots,), p=probs)

    # Build a histogram of counts for each basis state.
    counts = jnp.zeros(dim, dtype=jnp.int32)
    counts = counts.at[samples].add(1)
    estimated_probs = counts / shots

    if type == "probs":
        return estimated_probs

    if type == "expval":
        # For each observable, compute O from the shot-sampled
        # probabilities.  For diagonal observables this is exact;
        # for general observables we use Tr(O · diag(estimated_probs)).
        results = []
        for ob in obs:
            O_mat = ob.lifted_matrix(n_qubits)
            # diagonal approximation from
            # computational basis measurements, which is exact for
            # diagonal observables like PauliZ)
            results.append(jnp.real(jnp.dot(jnp.diag(O_mat), estimated_probs)))
        return jnp.array(results)

    raise ValueError(
        f"Shot simulation is only supported for 'probs' and 'expval', got {type!r}."
    )
