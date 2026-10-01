"""Diagonal measurements agree with dense observables and their derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaqsi import Operation, PauliZ
from jaqsi.simulation import measure_state

jax.config.update("jax_enable_x64", True)


class DiagonalObservable(Operation):
    _matrix = np.diag([0.25, -1.75])
    _num_wires = 1


@pytest.mark.parametrize("n_qubits", [1, 4])
@pytest.mark.parametrize("dtype", [jnp.complex64, jnp.complex128])
def test_diagonal_measurements_and_vjp(n_qubits, dtype):
    rng = np.random.default_rng(19)
    # Include unnormalised states and exact zeros; neither may need a norm or
    # an absolute-value derivative at zero to evaluate a quadratic observable.
    states = rng.normal(size=(3, 2**n_qubits)) + 1j * rng.normal(size=(3, 2**n_qubits))
    states[0] = 0
    states[1, ::2] = 0
    states = jnp.asarray(states, dtype=dtype)
    obs: list[Operation] = [
        cls(wires=q, record=False)
        for q in reversed(range(n_qubits))
        for cls in (PauliZ, DiagonalObservable)
    ]
    matrices = jnp.stack([ob.lifted_matrix(n_qubits) for ob in obs])

    def dense(s):
        return jnp.real(jnp.einsum("bi,oij,bj->bo", jnp.conj(s), matrices, s))

    measured = jax.jit(jax.vmap(lambda s: measure_state(s, n_qubits, "expval", obs)))
    actual, pullback = jax.vjp(measured, states)
    expected, reference_pullback = jax.vjp(dense, states)
    cotangent = jnp.arange(states.shape[0] * len(obs), dtype=actual.dtype).reshape(
        states.shape[0], len(obs)
    )
    tol = 2e-5 if dtype == jnp.complex64 else 1e-11
    np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)
    np.testing.assert_allclose(
        pullback(cotangent)[0], reference_pullback(cotangent)[0], atol=tol, rtol=tol
    )
