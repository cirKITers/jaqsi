"""Cache-aware batch tiling and data-parallel execution over devices."""

import os
import subprocess
import sys
import textwrap
from typing import List

import jax
import jax.numpy as jnp
import pytest

from jaqsi import Script, memory, simulation
from jaqsi.gateset import CX, RX, RY, PauliZ
from jaqsi.noise import BitFlip
from jaqsi.operations import Operation

N = 3
B = 5
OBS: List[Operation] = [PauliZ(wires=0), PauliZ(wires=2)]
X = jax.random.uniform(jax.random.PRNGKey(0), (B, N), maxval=3.0)
W = jnp.array([0.3, 1.1, 2.0])


def circuit(x, w, noisy=False):
    for i in range(N):
        RY(x[i], wires=i)
    CX(wires=[0, 1])
    CX(wires=[1, 2])
    if noisy:
        BitFlip(0.1, wires=1)
    for i in range(N):
        RX(w[i], wires=i)


def run(type, x=X, in_axes=(0, None), **kw):
    return Script(circuit, n_qubits=N).execute(
        type=type, obs=OBS, args=(x, W), in_axes=in_axes, **kw
    )


def grad(x=X):
    def loss(w):
        return jnp.sum(
            Script(circuit, n_qubits=N).execute(
                type="expval", obs=OBS, args=(x, w), in_axes=(0, None)
            )
        )

    return jax.jit(jax.grad(loss))(W)


@pytest.fixture
def small_cache(monkeypatch):
    """A cache of two forward statevectors: the batch of 5 runs in tiles of 2."""
    monkeypatch.setattr(memory, "CACHE_BYTES", 2 * 2 * 2**N * 16)


@pytest.mark.unittest
def test_tile_size(small_cache) -> None:
    assert memory.tile_size(N, B, use_density=False, reverse=False) == 2
    assert memory.tile_size(N, B, use_density=False, reverse=True) == 1
    assert memory.tile_size(N, B, use_density=True, reverse=False) == 1
    assert memory.tile_size(N, 4, use_density=False, reverse=False) == 2


@pytest.mark.unittest
def test_tile_size_fits() -> None:
    assert memory.tile_size(N, B, use_density=False, reverse=True) == B


@pytest.mark.unittest
def test_ad_mode() -> None:
    seen = []

    def f(w):
        seen.append(simulation._ad_mode([w]))
        return w

    f(1.0)
    jax.grad(f)(1.0)
    jax.jvp(f, (1.0,), (1.0,))
    assert seen == [None, "reverse", "forward"]


@pytest.mark.unittest
@pytest.mark.parametrize("type", ["expval", "probs", "state"])
def test_tiled_matches_untiled(type, monkeypatch) -> None:
    expected = run(type)
    monkeypatch.setattr(memory, "CACHE_BYTES", 2 * 2 * 2**N * 16)
    assert jnp.allclose(run(type), expected, atol=1e-12)


@pytest.mark.unittest
def test_tiled_gradient(monkeypatch) -> None:
    expected = grad()
    monkeypatch.setattr(memory, "CACHE_BYTES", 2 * 2 * 2**N * 16)
    assert jnp.allclose(grad(), expected, atol=1e-12)


@pytest.mark.unittest
def test_tiled_density(monkeypatch) -> None:
    expected = run("density", kwargs={"noisy": True})
    monkeypatch.setattr(memory, "CACHE_BYTES", 2 * 5 * 4**N * 16)
    assert memory.tile_size(N, B, use_density=True, reverse=False) == 2
    assert jnp.allclose(run("density", kwargs={"noisy": True}), expected, atol=1e-12)


@pytest.mark.unittest
def test_tiled_batch_axis_and_initial_state(monkeypatch) -> None:
    psi0 = jax.random.normal(jax.random.PRNGKey(1), (B, 2**N)).astype(jnp.complex128)
    psi0 = psi0 / jnp.linalg.norm(psi0, axis=1, keepdims=True)
    expected = run("expval", initial_state=psi0)
    monkeypatch.setattr(memory, "CACHE_BYTES", 2 * 2 * 2**N * 16)
    got = run("expval", x=X.T, in_axes=(1, None), initial_state=psi0)
    assert jnp.allclose(got, expected, atol=1e-12)


@pytest.mark.unittest
def test_tiled_shots(monkeypatch) -> None:
    expected = run("probs", shots=100)
    monkeypatch.setattr(memory, "CACHE_BYTES", 2 * 2 * 2**N * 16)
    assert jnp.array_equal(run("probs", shots=100), expected)


@pytest.mark.unittest
def test_sharded_over_devices() -> None:
    """Four CPU devices against one sample at a time.

    A batch of 8 is split over the devices; an uneven batch of 6 is not, and
    neither is an even batch of 4, which holds fewer amplitudes than the
    threshold.  The device count is fixed when JAX initialises, so this runs in
    a fresh interpreter.
    """
    code = textwrap.dedent(
        """
        import jax, jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        from tests.test_batching import circuit, OBS, N, W
        from jaqsi import Script, memory
        assert len(jax.devices()) == 4
        memory.SHARD_MIN_SIZE = 8 * 2**N

        for batch_size, n_devices in ((8, 4), (6, 1), (4, 1)):
            x = jax.random.uniform(jax.random.PRNGKey(0), (batch_size, N))

            def loss(w):
                s = Script(circuit, n_qubits=N)
                return jnp.sum(s.execute("expval", OBS, args=(x, w), in_axes=(0, None)))

            def loss_single(w):
                s = Script(circuit, n_qubits=N)
                return sum(jnp.sum(s.execute("expval", OBS, args=(xi, w))) for xi in x)

            out = Script(circuit, n_qubits=N).execute(
                "expval", OBS, args=(x, W), in_axes=(0, None)
            )
            assert len(out.sharding.device_set) == n_devices
            assert jnp.allclose(loss(W), loss_single(W), atol=1e-12)
            assert jnp.allclose(
                jax.jit(jax.grad(loss))(W), jax.grad(loss_single)(W), atol=1e-12
            )
        """
    )
    env = dict(os.environ, JAX_NUM_CPU_DEVICES="4")
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.unittest
def test_sharded_inputs_share_a_script() -> None:
    """One ``Script`` called on the same shapes, placed differently.

    An executable compiled ahead of time accepts only the shardings it was
    compiled for, so an input split over the devices and a replicated one must
    not share a cached executable.
    """
    code = textwrap.dedent(
        """
        import jax, jax.numpy as jnp
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        jax.config.update("jax_enable_x64", True)
        from tests.test_batching import circuit, OBS, N, W
        from jaqsi import Script, memory
        memory.SHARD_MIN_SIZE = 1

        x = jax.random.uniform(jax.random.PRNGKey(0), (8, N))
        mesh = Mesh(jax.devices(), ("batch",))
        script = Script(circuit, n_qubits=N)
        expected = script.execute("expval", OBS, args=(x, W), in_axes=(0, None))
        for spec in (P("batch"), P()):
            placed = jax.device_put(x, NamedSharding(mesh, spec))
            got = script.execute("expval", OBS, args=(placed, W), in_axes=(0, None))
            assert jnp.allclose(got, expected, atol=1e-12)
        """
    )
    env = dict(os.environ, JAX_NUM_CPU_DEVICES="4")
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
