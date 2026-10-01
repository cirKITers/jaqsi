"""Structured compilation must agree with independent dense gate application."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqsi import gateset as g
from jaqsi.operations import Operation
from jaqsi import simulation as sim


def initial_state(n=4):
    state = jnp.arange(1, 2**n + 1) + 1j * jnp.arange(2**n, 0, -1)
    return state / jnp.linalg.norm(state)


def tape(p):
    return [
        g.PauliX(3),
        g.SWAP([3, 0]),
        g.CX([2, 0]),
        g.CY([0, 3]),
        g.CCX([3, 0, 2]),
        g.CSWAP([2, 3, 0]),
        g.RZ(p[0], 3),
        g.RZZ(p[1], [3, 0]),
        g.DiagonalQubitUnitary(jnp.exp(1j * p), [2, 0]),
        g.CRZ(p[2], [0, 3]),
        g.CRX(p[3], [3, 1]),
        g.ControlledPauliRot(p[0], "XY", [3, 0, 2]),
        g.ControlledPauliRot(p[1], "Y", [2, 0, 3], n_controls=2),
        g.ControlledPhaseShift(p[2], [2, 1]),
        g.S(0),
    ]


def layered(p):
    """A permutation layer and a diagonal layer, each spanning several wires.

    The H layer fuses into the rotations on its wires, so the comparison also
    covers a constant matrix with an irrational entry: the adjoint sweep
    inverts it, and the reconstruction is only as unitary as the matrix is
    exact (see the dtype test in ``test_jaqsi.py``).
    """
    return (
        [g.H(i) for i in range(4)]
        + [g.RY(p[i], i) for i in range(4)]
        + [g.CX([i, (i + 1) % 4]) for i in range(4)]
        + [g.RZ(p[i], i) for i in range(4)]
        + [g.SWAP([1, 2])]
    )


def dense(ops, state):
    for op in ops:
        state = op.apply_to_state(state, 4)
    return state


@pytest.mark.parametrize("build", [tape, layered])
@pytest.mark.parametrize("depth", [1, 4])
def test_values_and_derivatives(build, depth):
    p = jnp.linspace(0.2, 0.9, 4)
    state = initial_state()
    obs = [g.PauliX(0), g.PauliY(3), g.PauliZ(2)]

    def actual(p, state):
        return sim.simulate_and_measure(
            build(p) * depth, 4, "expval", obs, False, initial_state=state
        )

    def reference(p, state):
        return sim.measure_state(dense(build(p) * depth, state), 4, "expval", obs)

    np.testing.assert_allclose(
        jax.jit(actual)(p, state), reference(p, state), atol=1e-12
    )
    for f in (jax.jacrev, jax.jacfwd):
        np.testing.assert_allclose(
            f(actual)(p, state), f(reference)(p, state), atol=1e-11
        )
    for a, b in zip(
        jax.vjp(actual, p, state)[1](jnp.ones(3)),
        jax.vjp(reference, p, state)[1](jnp.ones(3)),
    ):
        np.testing.assert_allclose(a, b, atol=1e-11)

    states = jnp.stack([state, state[::-1]])

    def loss(fn, weights):
        return jnp.sum(jax.vmap(fn, in_axes=(None, 0))(weights, states))

    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda p: loss(actual, p)))(p),
        jax.grad(lambda p: loss(reference, p))(p),
        atol=1e-11,
    )


def test_structure_survives_fusion():
    blocks = sim._compile(
        [
            g.CX([3, 0]),
            g.SWAP([3, 0]),
            g.RZ(0.3, 2),
            g.S(2),
            g.CRX(0.2, [1, 0]),
            g.CRY(0.4, [1, 0]),
        ],
        4,
    )
    assert len(blocks) == 3
    assert blocks[0].structure.permutation == (0, 3, 1, 2)
    assert blocks[1].structure.diagonal
    assert blocks[2].structure.controls == 1


def test_runs_fuse_across_wires():
    plan = sim._compile(layered(jnp.linspace(0.2, 0.9, 4)), 4)
    assert [type(entry).__name__ for entry in plan] == (
        ["Gate"] * 4 + ["Permutation", "Diagonal", "Gate"]
    )
    assert plan[5].wires == (0, 1, 2, 3)

    # The index map is the composition of the whole ring, in tape order.
    state = initial_state()
    ring = [g.CX([i, (i + 1) % 4]) for i in range(4)]
    np.testing.assert_allclose(
        sim._apply_permutation(state.reshape((2,) * 4), plan[4].index).reshape(-1),
        dense(ring, state),
        atol=1e-12,
    )


def test_dense_override_and_mixed_fusion():
    matrix = g.H._matrix
    op = g.PauliX(0, matrix=matrix)
    blocks = sim._compile([op, g.RZ(0.2, 0)], 1)
    assert not blocks[0].structure.permutation
    assert not blocks[0].structure.diagonal
    np.testing.assert_allclose(sim.simulate_pure([op], 1), matrix[:, 0])

    # Arbitrary non-unitary matrices must retain their full action and cotangent.
    def actual(m):
        return sim.simulate_pure([Operation(0, matrix=m)], 1)

    np.testing.assert_allclose(
        jax.jacfwd(actual, holomorphic=True)(matrix),
        jax.jacfwd(lambda m: m[:, 0], holomorphic=True)(matrix),
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: g.Id([3, 0]),
        lambda: g.PauliX(3),
        lambda: g.PauliZ(2),
        lambda: g.S(0),
        lambda: g.SWAP([3, 0]),
        lambda: g.CX([3, 0]),
        lambda: g.CY([0, 3]),
        lambda: g.CZ([2, 0]),
        lambda: g.CCX([3, 0, 2]),
        lambda: g.CSWAP([2, 3, 0]),
        lambda: g.RZ(0.4, 3),
        lambda: g.RZZ(0.3, [3, 0]),
        lambda: g.PauliRot(0.4, "ZIZ", [3, 1, 0]),
        lambda: g.PauliRot(0.4, "XYZ", [3, 1, 0]),
        lambda: g.ControlledPauliRot(0.4, "XY", [3, 0, 2]),
        lambda: g.ControlledPauliRot(0.4, "Y", [3, 0, 2], n_controls=2),
        lambda: g.ControlledPauliRot(0.4, "Z", [3, 0, 2], n_controls=2),
    ],
)
def test_isolated_gate_and_dagger(factory):
    state = initial_state()
    op = factory()
    for gate in (op, op.dagger()):
        np.testing.assert_allclose(
            jax.jit(lambda s: sim.simulate_pure([gate], 4, s))(state),
            gate.apply_to_state(state, 4),
            atol=1e-12,
        )


def test_non_self_inverse_fused_permutation_adjoint():
    ops = [g.CX([3, 0]), g.SWAP([3, 0]), g.CRX(0.3, [2, 1])]
    obs = [g.PauliY(0)]

    def actual(s):
        return sim.simulate_and_measure(ops, 4, "expval", obs, False, initial_state=s)[
            0
        ]

    def reference(s):
        return sim.measure_state(dense(ops, s), 4, "expval", obs)[0]

    np.testing.assert_allclose(
        jax.grad(actual)(initial_state()),
        jax.grad(reference)(initial_state()),
        atol=1e-12,
    )


def test_script_batched_forward_mode():
    from jaqsi import Script

    def circuit(x, w):
        g.RX(x, 0)
        g.CRX(w[0], [0, 3])
        g.RZ(w[1], 3)
        g.CX([3, 2])

    xs = jnp.array([0.2, 0.5, 0.8])
    w = jnp.array([0.7, 0.9])
    obs = [g.PauliY(3), g.PauliZ(2)]
    script = Script(circuit, n_qubits=4)

    def actual(w):
        return script.execute(type="expval", obs=obs, args=(xs, w), in_axes=(0, None))

    def reference(w):
        def single(x):
            ops = Script(circuit, n_qubits=4).record(x, w)
            state = jnp.zeros(16, dtype=jnp.complex128).at[0].set(1)
            return sim.measure_state(dense(ops, state), 4, "expval", obs)

        return jax.vmap(single)(xs)

    np.testing.assert_allclose(
        jax.jacfwd(actual)(w), jax.jacfwd(reference)(w), atol=1e-12
    )
    np.testing.assert_allclose(
        jax.jit(jax.jacrev(actual))(w), jax.jacrev(reference)(w), atol=1e-12
    )
