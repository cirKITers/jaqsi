import pytest
import jax
import jax.numpy as jnp
import numpy as np

from jaqsi.pulses import PulseEnvelope, PulseGates, PulseInformation
from jaqsi import Evolution, Script


def assert_default_pulse_state():
    assert PulseInformation.get_envelope() == PulseInformation.DEFAULT_ENVELOPE
    assert PulseInformation.get_rwa() is PulseInformation.DEFAULT_RWA
    assert PulseInformation.get_frame() == PulseInformation.DEFAULT_FRAME
    assert PulseGates._active_envelope == PulseInformation.DEFAULT_ENVELOPE
    assert PulseGates._active_rwa is PulseInformation.DEFAULT_RWA
    assert PulseGates._active_frame == PulseInformation.DEFAULT_FRAME


@pytest.mark.parametrize("name", ["X", "Y", "Z", "Id", "_H_CZ", "_H_corr"])
def test_hamiltonian_constants_are_exact_host_arrays(name):
    """Pulse Hamiltonian terms keep full precision whatever the active dtype.

    Built with a JAX dtype at import they would be complex64 unless x64 was
    already on, which rounds the pi factors of the CZ and correction
    Hamiltonians to 1e-8.  :class:`~jaqsi.operations.Hermitian` casts them to
    the active dtype instead.
    """
    matrix = getattr(PulseGates, name)
    assert isinstance(matrix, np.ndarray), f"{name} pins its dtype"
    assert matrix.dtype == np.complex128


def test_hamiltonian_pi_factors_are_double_precision():
    """The pi factors of the CZ and correction Hamiltonians are exact."""
    assert PulseGates._H_corr[0, 0] == np.pi / 2
    assert PulseGates._H_CZ[3, 3] == np.pi


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
    PulseInformation.set_envelope("drag")
    Script(pulse_circuit, n_qubits=1).execute(
        type="state", args=(jnp.pi / 4, PulseInformation.RX.params)
    )
    assert len(Evolution._evolve_solver_cache) >= 1

    # Switch back to the default envelope.  Stale entries that referenced
    # the drag coefficient functions must be evicted so they cannot
    # be returned for the new (gaussian) coefficient functions.
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

        prev_steps = Evolution.set_solver_defaults(max_steps=1)
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


def test_single_term_drive_is_solved_in_closed_form():
    """A drive f(t) H commutes with itself, so U = exp(-i F H), F = int f dt.

    For the RWA gaussian RY, lifted by its edge value g0 = exp(-T^2 / (8 sigma^2)),
    F = w A (sigma sqrt(2 pi) erf(T / (2 sqrt(2) sigma)) - T g0) / (2 (1 - g0))
    (envelope centre ``T / 2``).  The angle is far beyond pi, where the matrix
    ODE needs hundreds of steps.
    """
    from jax.scipy.special import erf
    from jaqsi.operations import Hermitian

    PulseInformation.set_envelope("gaussian", rwa=True)
    A, sigma, T = PulseInformation.RY.params
    w = 60.0
    H = PulseGates._coeff_RY_Y * Hermitian(PulseGates.Y, wires=0, record=False)
    U = H.evolve()([jnp.array([A, sigma, T, w])], T).matrix

    g0 = jnp.exp(-(T**2) / (8 * sigma**2))
    area = sigma * jnp.sqrt(2 * jnp.pi) * erf(T / (2 * jnp.sqrt(2) * sigma)) - T * g0
    F = w * A * area / (2 * (1 - g0))
    expected = jnp.cos(F) * jnp.eye(2) - 1j * jnp.sin(F) * PulseGates.Y
    assert jnp.allclose(U, expected, atol=1e-8)


def test_closed_form_can_be_switched_off():
    """``closed_form=False`` integrates a single-term drive as a matrix ODE.

    The closed form carries the eigendecomposition of ``H`` as solver input,
    the matrix ODE the split ``-iH`` of shape ``(n_terms, 2, dim, dim)``.
    """
    from jaqsi.operations import Hermitian

    PulseInformation.set_envelope("gaussian", rwa=True)
    A, sigma, T = PulseInformation.RY.params
    H = PulseGates._coeff_RY_Y * Hermitian(PulseGates.Y, wires=0, record=False)
    params = [jnp.array([A, sigma, T, 2.0])]

    closed = H.evolve()(params, T)
    ode = H.evolve(closed_form=False)(params, T)
    assert isinstance(closed._inputs[0], tuple)
    assert ode._inputs[0].shape == (1, 2, 2, 2)
    assert jnp.allclose(ode.matrix, closed.matrix, atol=1e-8)

    prev = Evolution.set_solver_defaults(closed_form=False)
    try:
        assert H.evolve()(params, T)._inputs[0].shape == (1, 2, 2, 2)
    finally:
        Evolution.set_solver_defaults(**prev)


def test_rwa_rotations_are_single_term():
    """Under the RWA the off-axis component of RX and RY vanishes and is dropped.

    This holds for single-quadrature envelopes such as the Gaussian.
    """
    from jaqsi.evolution import PendingEvolution

    def circuit(w):
        PulseGates.RX(w, wires=0)
        PulseGates.RY(w, wires=0)

    for rwa, n_terms in ((True, 1), (False, 2)):
        PulseInformation.set_envelope("gaussian", rwa=rwa)
        tape = Script(circuit, n_qubits=1).record(0.3)
        ops = [op for op in tape if isinstance(op, PendingEvolution)]
        assert [len(op._inputs[1]) for op in ops] == [n_terms, n_terms]


def test_drag_drives_a_second_quadrature():
    """DRAG keeps its quadrature under the RWA, so RX and RY take the matrix ODE.

    The quadrature drives Y for RX and -X for RY.  The calibrated defaults
    still implement the target rotations.
    """
    from jaqsi.evolution import PendingEvolution
    from jaqsi.gateset import RX as OpRX, RY as OpRY

    PulseInformation.set_envelope("drag", rwa=True)
    p = jnp.array([0.4, 0.5, 1.3, 1.4, 1.0])  # [A, beta, sigma, T, w]
    q = 0.5 * PulseEnvelope.drag_quadrature(p, 0.3, 0.7) * p[-1]
    assert q != 0.0
    assert jnp.isclose(PulseGates._coeff_RX_Y(p, 0.3), q, atol=1e-12)
    assert jnp.isclose(PulseGates._coeff_RY_X(p, 0.3), -q, atol=1e-12)

    for pulse_gate, target_gate in ((PulseGates.RX, OpRX), (PulseGates.RY, OpRY)):
        tape = Script(lambda w: pulse_gate(w, wires=0), n_qubits=1).record(0.3)
        (op,) = [op for op in tape if isinstance(op, PendingEvolution)]
        assert len(op._inputs[1]) == 2
        assert op._inputs[0].shape == (2, 2, 2, 2)  # matrix ODE, no closed form

        for w in (jnp.pi / 4, jnp.pi / 2, jnp.pi):
            pulse = Script(lambda: pulse_gate(w, wires=0), n_qubits=1)
            target = Script(lambda: target_gate(w, wires=0), n_qubits=1)
            overlap = jnp.vdot(
                target.execute(type="state"), pulse.execute(type="state")
            )
            assert jnp.isclose(jnp.abs(overlap) ** 2, 1.0, atol=1e-2)
            assert jnp.isclose(jnp.angle(overlap), 0.0, atol=1e-2)


ENVELOPES = [name for name in PulseEnvelope.available() if name != "general"]


@pytest.mark.parametrize("envelope", ENVELOPES)
def test_envelopes_are_centred_at_the_pulse_midpoint(envelope):
    """Every envelope peaks at ``T / 2`` and is symmetric around it.

    The DRAG quadrature ``-beta dE/dt`` is odd around the centre instead, so
    that its area vanishes.
    """
    PulseInformation.set_envelope(envelope, rwa=True)
    pp = PulseInformation.RX.params
    if envelope == "drag":
        pp = pp.at[1].set(0.3)  # the calibrated beta leaves Q silent
    T = pp[-1]
    p = jnp.concatenate([pp, jnp.ones(1)])  # [envelope params..., T, w]
    ts = jnp.linspace(0.0, T, 7)

    env = jax.vmap(lambda t: PulseGates._coeff_RX_X(p, t))
    quad = jax.vmap(lambda t: PulseGates._coeff_RX_Y(p, t))
    assert jnp.allclose(env(ts), env(T - ts), atol=1e-12)
    assert jnp.allclose(quad(ts), -quad(T - ts), atol=1e-12)
    assert env(ts)[3] == jnp.max(env(ts))
    if envelope == "drag":
        assert jnp.abs(quad(ts[:1])[0]) > 1e-3


@pytest.mark.parametrize("envelope", ENVELOPES)
@pytest.mark.parametrize("gate", ["RX", "RY"])
def test_calibrated_defaults_implement_rotations(envelope, gate):
    """The shipped defaults of every envelope implement RX and RY under the RWA."""
    from jaqsi import gateset

    PulseInformation.set_envelope(envelope, rwa=True)
    pulse_gate = getattr(PulseGates, gate)
    target_gate = getattr(gateset, gate)

    for w in (jnp.pi / 4, jnp.pi / 2, jnp.pi):
        pulse = Script(lambda: pulse_gate(w, wires=0), n_qubits=1)
        target = Script(lambda: target_gate(w, wires=0), n_qubits=1)
        overlap = jnp.vdot(target.execute(type="state"), pulse.execute(type="state"))
        assert jnp.isclose(jnp.abs(overlap) ** 2, 1.0, atol=1e-2)
        assert jnp.isclose(jnp.angle(overlap), 0.0, atol=1e-2)


def test_closed_form_matches_matrix_ode():
    """States and pulse-parameter gradients agree with the matrix ODE.

    The two-term RWA drive (zero off-axis coefficient) still takes the matrix
    ODE.  CZ has a degenerate spectrum, where a traced eigendecomposition
    would give NaN gradients.
    """
    from jaqsi.gateset import PauliZ
    from jaqsi.operations import Hermitian

    PulseInformation.set_envelope("gaussian", rwa=True)
    X = Hermitian(PulseGates.X, wires=0, record=False)
    Y = Hermitian(PulseGates.Y, wires=0, record=False)

    def circuit(w, pp, cz, two_term):
        if two_term:
            H = PulseGates._coeff_RY_X * X + PulseGates._coeff_RY_Y * Y
        else:
            H = PulseGates._coeff_RY_Y * Y
        p = jnp.concatenate([pp, jnp.atleast_1d(w)])
        H.evolve()([p] * H.n_terms, pp[-1])
        PulseGates.RX(w / 3, wires=1)
        PulseGates.CZ(wires=[0, 1], pulse_params=cz)
        PulseGates.RY(w / 2, wires=1)

    def expval(w, pp, cz, two_term):
        script = Script(circuit, n_qubits=2)
        obs = [PauliZ(wires=0), PauliZ(wires=1)]
        out = script.execute("expval", obs, args=(w, pp, cz, two_term))
        return out.sum()

    pp, cz = PulseInformation.RY.params, PulseInformation.CZ.params
    for w in (0.4, 25.0):
        closed = jax.grad(expval, argnums=(0, 1, 2))(w, pp, cz, False)
        ode = jax.grad(expval, argnums=(0, 1, 2))(w, pp, cz, True)
        assert jnp.allclose(
            expval(w, pp, cz, False), expval(w, pp, cz, True), atol=1e-8
        )
        for g_closed, g_ode in zip(closed, ode):
            assert jnp.all(jnp.isfinite(g_closed))
            assert jnp.allclose(g_closed, g_ode, atol=1e-6)
