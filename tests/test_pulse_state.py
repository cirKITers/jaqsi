import pytest
import jax
import jax.numpy as jnp

from jaqsi.pulses import PulseGates, PulseInformation
from jaqsi import Evolution, Script

jax.config.update("jax_enable_x64", True)


def assert_default_pulse_state():
    assert PulseInformation.get_envelope() == PulseInformation.DEFAULT_ENVELOPE
    assert PulseInformation.get_rwa() is PulseInformation.DEFAULT_RWA
    assert PulseInformation.get_frame() == PulseInformation.DEFAULT_FRAME
    assert PulseGates._active_envelope == PulseInformation.DEFAULT_ENVELOPE
    assert PulseGates._active_rwa is PulseInformation.DEFAULT_RWA
    assert PulseGates._active_frame == PulseInformation.DEFAULT_FRAME


def test_snapshot_restore_restores_config_and_leaf_params():
    snapshot = PulseInformation.snapshot_state()
    original_rx = PulseInformation.RX.params

    PulseInformation.set_envelope("gaussian", rwa=False, frame="lab")
    PulseInformation.RX.params = jnp.ones_like(PulseInformation.RX.params) * 0.123

    PulseInformation.restore_state(snapshot)

    assert PulseInformation.get_envelope() == snapshot.envelope
    assert PulseInformation.get_rwa() is snapshot.rwa
    assert PulseInformation.get_frame() == snapshot.frame
    assert PulseGates._active_envelope == snapshot.envelope
    assert PulseGates._active_rwa is snapshot.rwa
    assert PulseGates._active_frame == snapshot.frame
    assert jnp.allclose(PulseInformation.RX.params, original_rx)


def test_preserve_state_restores_after_exception():
    snapshot = PulseInformation.snapshot_state()

    with pytest.raises(RuntimeError, match="boom"):
        with PulseInformation.preserve_state():
            PulseInformation.set_envelope("gaussian", rwa=False, frame="lab")
            PulseInformation.RY.params = (
                jnp.ones_like(PulseInformation.RY.params) * 0.456
            )
            raise RuntimeError("boom")

    assert PulseInformation.get_envelope() == snapshot.envelope
    assert PulseInformation.get_rwa() is snapshot.rwa
    assert PulseInformation.get_frame() == snapshot.frame
    assert jnp.allclose(PulseInformation.RY.params, snapshot.leaf_params["RY"])


def test_00_autouse_fixture_allows_unrestored_mutation():
    PulseInformation.set_envelope("gaussian", rwa=False, frame="lab")
    PulseInformation.RX.params = jnp.ones_like(PulseInformation.RX.params) * 0.789

    assert PulseInformation.get_envelope() == "gaussian"
    assert PulseInformation.get_rwa() is False
    assert PulseInformation.get_frame() == "lab"


def test_01_autouse_fixture_restores_after_previous_test():
    assert_default_pulse_state()


def test_set_envelope_evicts_stale_solver_cache():
    """Regression test for the order-dependent fidelity failures.

    Building an evolution under one envelope cached a compiled XLA
    program keyed on coefficient-function code object identity.
    Switching the envelope rebuilt the coefficient functions, but the
    cache key (``id(fn.__code__)``) could collide with a freshly
    allocated code object, returning the stale program for a different
    pulse shape and silently degrading fidelity.

    With cache invalidation in place, the cache must be empty after a
    state change, and a freshly evaluated fidelity for the current
    envelope must be perfect.
    """

    def pulse_circuit(w, pp):
        PulseGates.RX(w, wires=0, pulse_params=pp)

    def target_circuit(w):
        from jaqsi.gateset import RX as OpRX

        OpRX(w, wires=0)

    # Prime the cache under a different envelope.
    PulseInformation.set_envelope("gaussian")
    Script(pulse_circuit, n_qubits=1).execute(
        type="state", args=(jnp.pi / 4, PulseInformation.RX.params)
    )
    assert len(Evolution._evolve_solver_cache) >= 1

    # Switch back to the default envelope.  Stale entries that referenced
    # the gaussian coefficient functions must be evicted so they cannot
    # be returned for the new (drag) coefficient functions.
    PulseInformation.set_envelope(PulseInformation.DEFAULT_ENVELOPE)
    assert len(Evolution._evolve_solver_cache) == 0

    pulse_script = Script(pulse_circuit, n_qubits=1)
    target_script = Script(target_circuit, n_qubits=1)
    state_pulse = pulse_script.execute(
        type="state", args=(jnp.pi / 2, PulseInformation.RX.params)
    )
    state_target = target_script.execute(type="state", args=(jnp.pi / 2,))
    fidelity = float(jnp.abs(jnp.vdot(state_target, state_pulse)) ** 2)
    assert jnp.isclose(fidelity, 1.0, atol=1e-2), (
        f"Stale solver cache contaminated fidelity: {fidelity}"
    )


def test_pulse_gates_are_solved_in_one_batch_per_shape():
    """All RX pulse gates of a tape share one vmapped solve; CZ gets its own."""
    from jaqsi import evolution, simulation

    def circuit(w):
        for q in range(4):
            PulseGates.RX(w * (q + 1) / 4, wires=q)
        PulseGates.CZ(wires=[0, 1])

    script = Script(circuit, n_qubits=4)
    w = jnp.pi / 3

    assert evolution.resolve_pending(script.record(w)) == [4, 1]

    batched = script.execute(type="state", args=(w,))
    # Unresolved tape: every gate solves itself on first ``.matrix`` access.
    lazy = simulation.simulate_pure(script.record(w), 4)
    assert jnp.allclose(batched, lazy, atol=1e-10)


def test_host_offload_matches_and_reraises():
    """Host-offloaded solves give the same results and still raise on failure."""
    from jaqsi.gateset import PauliZ

    def circuit(w):
        PulseGates.RX(w, wires=0)
        PulseGates.RY(w / 2, wires=1)
        PulseGates.CZ(wires=[0, 1])

    script = Script(circuit, n_qubits=2)
    ws = jnp.linspace(0.1, 1.5, 4)

    def expval(w):
        return script.execute(type="expval", obs=[PauliZ(wires=0)], args=(w,))[0]

    ref_state = script.execute(type="state", args=(ws,), in_axes=(0,))
    ref_grad = jax.grad(expval)(ws[0])

    prev = Evolution.set_solver_defaults(host_offload=True)
    try:
        state = script.execute(type="state", args=(ws,), in_axes=(0,))
        grad = jax.grad(expval)(ws[0])
        assert jnp.allclose(state, ref_state, atol=1e-10)
        assert jnp.allclose(grad, ref_grad, atol=1e-8)

        prev_steps = Evolution.set_solver_defaults(max_steps=4)
        try:
            with pytest.raises(RuntimeError):
                Script(circuit, n_qubits=2).execute(type="state", args=(ws[0],))
        finally:
            Evolution.set_solver_defaults(**prev_steps)
    finally:
        Evolution.set_solver_defaults(**prev)


def test_identical_pulse_gates_are_solved_once():
    """Repeated fixed-angle gates share one solve, also inside a jit trace."""
    from jaqsi import evolution

    def circuit(w):
        for q in range(4):
            PulseGates.RX(jnp.pi / 2, wires=q)
        PulseGates.CZ(wires=[0, 1])
        PulseGates.CZ(wires=[2, 3])

    script = Script(circuit, n_qubits=4)
    assert evolution.resolve_pending(script.record(0.0)) == [1, 1]

    seen = []

    @jax.jit
    def traced(w):
        seen.append(evolution.resolve_pending(script.record(w)))
        return w

    traced(0.0)
    assert seen == [[1, 1]]
