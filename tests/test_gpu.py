"""GPU backend checks.

The equivalence test is skipped without an accelerator.  Run this module on
its own (``pytest tests/test_gpu.py``) to exercise it in complex64, which is
where the forced matmul precision matters; other test modules enable x64
globally.
"""

import jax
import jax.numpy as jnp
import pytest

from jaqsi import Script, memory
from jaqsi.gateset import H, CRX, PauliZ

_accelerators = [d for d in jax.devices() if d.platform != "cpu"]


def test_available_memory_positive():
    assert memory.available_memory_bytes() > 0


def test_available_memory_prefers_device_stats(monkeypatch):
    class FakeDevice:
        def memory_stats(self):
            return {"bytes_limit": 10, "bytes_in_use": 3}

    monkeypatch.setattr(jax, "devices", lambda *a, **k: [FakeDevice()])
    assert memory.available_memory_bytes() == 7


@pytest.mark.skipif(not _accelerators, reason="no accelerator available")
def test_accelerator_matches_cpu():
    n = 6

    def circuit(phi):
        for i in range(n):
            H(wires=i)
        for i in range(n):
            CRX(phi, wires=[i, (i + 1) % n])

    obs = [PauliZ(wires=i) for i in range(n)]
    phi = jnp.linspace(0.1, 2.0, 8)

    def run():
        return Script(circuit, n_qubits=n).execute(
            type="expval", obs=obs, args=(phi,), in_axes=(0,)
        )

    accel = run()
    with jax.default_device(jax.devices("cpu")[0]):
        cpu = run()
    assert jnp.allclose(accel, cpu, atol=1e-6)
