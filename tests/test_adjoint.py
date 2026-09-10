"""Adjoint differentiation of expectation values and gate fusion."""

import jax
import jax.numpy as jnp
import pytest

from jaqsi import Script, Hermitian, build_parity_observable
from jaqsi import simulation
from jaqsi.gateset import (
    H,
    RX,
    RY,
    RZ,
    CX,
    CRX,
    Rot,
    PauliRot,
    ControlledPhaseShift,
    PauliX,
    PauliZ,
    RandomUnitary,
)
from jaqsi.noise import BitFlip

jax.config.update("jax_enable_x64", True)

N = 3
OBS = [PauliZ(wires=0), PauliX(wires=1), build_parity_observable([0, 2])]
PARAMS = jnp.linspace(0.1, 1.9, 10)


def rich_circuit(p):
    H(wires=0)
    RX(p[0], wires=0)
    RY(p[1], wires=1)
    RZ(p[2], wires=0)
    CX(wires=[0, 1])
    CRX(p[3], wires=[1, 2])
    Rot(p[4], p[5], p[6], wires=2)
    PauliRot(p[7], "XY", wires=[0, 2])
    ControlledPhaseShift(p[8], wires=[2, 1])
    CX(wires=[0, 1])
    RX(p[9], wires=1)


def expval_fast(p, obs=OBS, **kw):
    script = Script(rich_circuit, n_qubits=N)
    return script.execute(type="expval", obs=obs, args=(p,), **kw)


def expval_reference(p, obs=OBS, **kw):
    """Same expectation values through the state path (plain autodiff)."""
    s = Script(rich_circuit, n_qubits=N).execute(type="state", args=(p,), **kw)
    return jnp.stack([jnp.real(jnp.conj(s) @ ob.lifted_matrix(N) @ s) for ob in obs])


@pytest.fixture
def adjoint_calls(monkeypatch):
    """Count how often the adjoint fast path is taken."""
    calls = []
    original = simulation._adjoint_expval

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(simulation, "_adjoint_expval", spy)
    return calls


@pytest.mark.unittest
def test_gradient_matches_state_path(adjoint_calls) -> None:
    assert jnp.allclose(expval_fast(PARAMS), expval_reference(PARAMS), atol=1e-10)
    jac = jax.jacrev(expval_fast)(PARAMS)
    ref = jax.jacrev(expval_reference)(PARAMS)
    assert jnp.allclose(jac, ref, atol=1e-10)
    assert adjoint_calls


@pytest.mark.unittest
def test_diagonal_observable_gradient() -> None:
    obs = [PauliZ(wires=q) for q in range(N)]
    grad = jax.grad(lambda p: jnp.sum(expval_fast(p, obs)))(PARAMS)
    ref = jax.grad(lambda p: jnp.sum(expval_reference(p, obs)))(PARAMS)
    assert jnp.allclose(grad, ref, atol=1e-10)


@pytest.mark.unittest
def test_batched_jit_grad() -> None:
    """jit(grad) over a vmapped batch, as a training loop does."""

    def circuit(x, w):
        RX(x[0], wires=0)
        RX(x[1], wires=1)
        RZ(w[0], wires=0)
        RY(w[1], wires=1)
        CX(wires=[0, 1])

    xs = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    w = jnp.array([0.7, 0.8])
    obs = [PauliZ(wires=0), PauliX(wires=1)]

    def loss(w):
        return jnp.sum(
            Script(circuit, n_qubits=2).execute(
                type="expval", obs=obs, args=(xs, w), in_axes=(0, None)
            )
        )

    def loss_ref(w):
        states = Script(circuit, n_qubits=2).execute(
            type="state", args=(xs, w), in_axes=(0, None)
        )
        mats = jnp.stack([ob.lifted_matrix(2) for ob in obs])
        expvals = jnp.einsum("bi,oij,bj->bo", jnp.conj(states), mats, states)
        return jnp.sum(jnp.real(expvals))

    grad = jax.jit(jax.grad(loss))(w)
    assert jnp.allclose(grad, jax.grad(loss_ref)(w), atol=1e-10)


@pytest.mark.unittest
def test_initial_state_gradient() -> None:
    psi0 = jnp.arange(1, 2**N + 1, dtype=jnp.complex128)
    psi0 = psi0 / jnp.linalg.norm(psi0)

    def f(p, s):
        return jnp.sum(expval_fast(p, initial_state=s))

    def f_ref(p, s):
        return jnp.sum(expval_reference(p, initial_state=s))

    grads = jax.grad(f, argnums=(0, 1))(PARAMS, psi0)
    refs = jax.grad(f_ref, argnums=(0, 1))(PARAMS, psi0)
    for g, r in zip(grads, refs):
        assert jnp.allclose(g, r, atol=1e-10)


@pytest.mark.unittest
def test_forward_mode_falls_back(adjoint_calls) -> None:
    """jvp/jacfwd cannot use the custom VJP; they must still work."""
    jac_fwd = jax.jacfwd(expval_fast)(PARAMS)
    assert not adjoint_calls
    assert jnp.allclose(jac_fwd, jax.jacrev(expval_reference)(PARAMS), atol=1e-10)
    _, tangent = jax.jvp(expval_fast, (PARAMS,), (jnp.ones_like(PARAMS),))
    assert jnp.allclose(tangent, jac_fwd.sum(axis=1), atol=1e-10)


@pytest.mark.unittest
def test_batched_forward_mode_falls_back(adjoint_calls) -> None:
    """The batched path is jitted; forward mode must be caught on the arguments."""

    def circuit(x, w):
        RX(x, wires=0)
        RY(w, wires=0)

    xs = jnp.array([0.1, 0.2])

    def f(w):
        return Script(circuit, n_qubits=1).execute(
            type="expval", obs=[PauliZ(wires=0)], args=(xs, w), in_axes=(0, None)
        )[:, 0]

    jac = jax.jacfwd(f)(0.3)
    assert not adjoint_calls
    assert jnp.allclose(jac, -jnp.cos(xs) * jnp.sin(0.3), atol=1e-10)
    assert jnp.allclose(jax.jacrev(f)(0.3), jac, atol=1e-10)
    assert adjoint_calls


@pytest.mark.unittest
def test_hessian(adjoint_calls) -> None:
    hess = jax.hessian(lambda p: expval_fast(p)[1])(PARAMS)
    assert adjoint_calls
    ref = jax.hessian(lambda p: expval_reference(p)[1])(PARAMS)
    assert jnp.allclose(hess, ref, atol=1e-8)


@pytest.mark.unittest
def test_noise_falls_back(adjoint_calls) -> None:
    def circuit(theta):
        RX(theta, wires=0)
        BitFlip(0.1, wires=0)

    def f(theta):
        return Script(circuit, n_qubits=1).execute(
            type="expval", obs=[PauliZ(wires=0)], args=(theta,)
        )[0]

    grad = jax.grad(f)(0.4)
    assert not adjoint_calls
    assert jnp.allclose(grad, -0.8 * jnp.sin(0.4), atol=1e-10)


@pytest.mark.unittest
def test_non_unitary_gate_falls_back(adjoint_calls) -> None:
    gate = jnp.array([[1.0, 0.5], [0.0, 1.0]], dtype=jnp.complex128)

    def circuit(theta):
        RX(theta, wires=0)
        Hermitian(matrix=gate, wires=0)

    def f(theta):
        return Script(circuit, n_qubits=1).execute(
            type="expval", obs=[PauliZ(wires=0)], args=(theta,)
        )[0]

    def f_ref(theta):
        s = Script(circuit, n_qubits=1).execute(type="state", args=(theta,))
        return jnp.real(jnp.conj(s) @ PauliZ._matrix @ s)

    assert jnp.allclose(jax.grad(f)(0.4), jax.grad(f_ref)(0.4), atol=1e-10)
    assert not adjoint_calls


@pytest.mark.unittest
def test_static_evolution_gradient(adjoint_calls) -> None:
    """exp(-i t Z)|+> gives <X> = cos(2t); the evolved gate is unitary."""

    def circuit(t):
        H(wires=0)
        Hermitian(matrix=PauliZ._matrix, wires=0, record=False).evolve()(t=t, wires=0)

    def f(t):
        return Script(circuit, n_qubits=1).execute(
            type="expval", obs=[PauliX(wires=0)], args=(t,)
        )[0]

    assert jnp.allclose(jax.grad(f)(0.3), -2 * jnp.sin(0.6), atol=1e-10)
    assert adjoint_calls


@pytest.mark.unittest
def test_fuse_merges_neighbours_on_same_wires() -> None:
    a, b, c, d, e = (jnp.eye(2) * k for k in range(1, 6))
    cx = jnp.eye(4)
    fused = simulation._fuse([(a, (0,)), (b, (1,)), (c, (0,)), (cx, (0, 1)), (e, (0,))])
    assert [w for _, w in fused] == [(0,), (1,), (0, 1), (0,)]
    assert jnp.allclose(fused[0][0], c @ a)
    fused = simulation._fuse([(cx, (0, 1)), (d, (1,)), (cx, (0, 1))])
    assert [w for _, w in fused] == [(0, 1), (1,), (0, 1)]


@pytest.mark.unittest
def test_fused_state_matches_gate_by_gate() -> None:
    tape = Script(rich_circuit, n_qubits=N).record(PARAMS)
    state = simulation.simulate_pure(tape, N)
    ref = jnp.zeros(2**N, dtype=jnp.complex128).at[0].set(1.0)
    for op in tape:
        ref = op.apply_to_state(ref, N)
    assert jnp.allclose(state, ref, atol=1e-12)


@pytest.mark.unittest
def test_is_unitary_flags() -> None:
    rx = RX(0.3, wires=0, record=False)
    assert rx.is_unitary
    assert rx.dagger().is_unitary
    assert rx.prod(RY(0.2, wires=1, record=False)).is_unitary
    assert not (rx * 2.0).is_unitary
    assert not Hermitian(matrix=PauliZ._matrix, wires=0, record=False).is_unitary
    random = RandomUnitary(wires=[0], key=jax.random.PRNGKey(0), record=False)
    assert not random.is_unitary
    assert not BitFlip(0.1, wires=0).is_unitary
