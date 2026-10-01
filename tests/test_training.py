"""Training results of the examples in ``docs/training.md`` stay reproducible."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jaqsi import Gates, Script
from jaqsi.gateset import PauliZ
from jaqsi.math import fidelity
from jaqsi.pulses import PulseEnvelope, PulseInformation


def adam_trajectory(loss_fn, params, lr, steps, jit=False):
    """Losses after every step of plain Adam, and the final parameters."""
    opt = optax.adam(lr)

    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    step = jax.jit(step) if jit else step
    opt_state = opt.init(params)
    losses = []
    for _ in range(steps):
        params, opt_state, loss = step(params, opt_state)
        losses.append(float(loss))
    return np.array(losses + [float(loss_fn(params))]), np.asarray(params)


def test_minimal_training_loop():
    """The minimal loop reproduces its results.

    Gate-level training has been bit-identical since the adjoint gradients were
    introduced; ``rtol=1e-12`` leaves room for platform round-off only.
    ``params[1]`` does not enter the cost, so it only drifts from ``0.2`` by
    Adam's normalization of round-off gradients, and is checked with an
    absolute bound instead.
    """

    def circuit(params):
        Gates.RY(params[0], wires=0)
        Gates.RY(params[1], wires=1)
        Gates.CX(wires=[0, 1])

    script = Script(circuit, n_qubits=2)
    obs = [PauliZ(wires=0)]

    def cost(params):
        return script.execute(type="expval", obs=obs, args=(params,))[0]

    losses, params = adam_trajectory(cost, jnp.array([0.1, 0.2]), 0.05, 100, True)
    np.testing.assert_allclose(
        losses[24:100:25],
        [
            0.18954755673957352,
            -0.9329512143583855,
            -0.9923251517460319,
            -0.9999997904383473,
        ],
        rtol=1e-12,
    )
    np.testing.assert_allclose(params[0], 3.1389723449742055, rtol=1e-12)
    np.testing.assert_allclose(params[1], 0.2, atol=1e-8)


def test_fitting_data():
    """The batched fit reproduces its results (tolerances as above).

    ``atol=1e-15`` covers the round-off of the smallest losses, which are
    averages of squared residuals of about ``3e-7``.
    """

    def model_circuit(x, weights):
        Gates.RX(x, wires=0)
        Gates.RY(weights[0], wires=0)
        Gates.CX(wires=[0, 1])
        Gates.RY(weights[1], wires=1)

    mscript = Script(model_circuit, n_qubits=2)
    obs = [PauliZ(wires=0)]
    xs = jnp.linspace(0.0, jnp.pi, 16)
    ys = jnp.cos(xs)

    def mse(weights):
        predictions = mscript.execute(
            type="expval", obs=obs, args=(xs, weights), in_axes=(0, None)
        )[:, 0]
        return jnp.mean((predictions - ys) ** 2)

    losses, weights = adam_trajectory(mse, jnp.array([0.1, 0.2]), 0.05, 100, True)
    np.testing.assert_allclose(
        losses[24:100:25],
        [
            2.007329667037132e-07,
            1.3095863100844179e-08,
            1.0464218803546601e-11,
            7.771503138489971e-14,
        ],
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(weights[0], -0.000856006429212641, rtol=1e-12)
    np.testing.assert_allclose(weights[1], 0.2, atol=1e-8)


THETA = jnp.pi / 2


def pulse_infidelity():
    """Infidelity of a pulse-level ``RX(pi/2)``, as in ``docs/training.md``."""
    target = Script(lambda w: Gates.RX(w, wires=0), n_qubits=1).execute(
        type="state", args=(THETA,)
    )

    def circuit(w, pulse_params):
        Gates.RX(w, wires=0, pulse=True, pulse_params=pulse_params)

    script = Script(circuit, n_qubits=1)

    def infidelity(pulse_params):
        state = script.execute(type="state", args=(THETA, pulse_params))
        return 1.0 - fidelity(target, state)

    return infidelity


@pytest.mark.parametrize("envelope", ["gaussian", "sech", "drag_legacy"])
def test_pulse_training_matches_analytic_trajectory(envelope):
    """Pulse training equals Adam on the analytic infidelity under the RWA.

    A single-quadrature RWA drive gives ``RX = exp(-i theta a X / 2)`` with the
    pulse area ``a``, so the infidelity is ``sin^2(theta (a - 1) / 2)``.  The
    area is integrated by Gauss-Legendre quadrature, independent of the ODE
    solver, the adjoint and the circuit machinery.  Any change of the pulse
    gradients beyond the solver tolerance shows up in the trajectory.
    """
    with PulseInformation.preserve_state():
        PulseInformation.set_envelope(envelope, rwa=True)
        envelope_fn = PulseEnvelope.get(envelope)["fn"]
        nodes, node_weights = np.polynomial.legendre.leggauss(200)

        def analytic_infidelity(pulse_params):
            T = pulse_params[-1]
            t = 0.5 * T * (nodes + 1.0)
            E = jax.vmap(lambda s: envelope_fn(pulse_params, s, T / 2))(t)
            area = 0.5 * T * jnp.sum(node_weights * E)
            return jnp.sin(THETA * (area - 1.0) / 2) ** 2

        start = PulseInformation.gate_by_name("RX").params * 1.15
        losses, params = adam_trajectory(pulse_infidelity(), start, 0.01, 30)
        ref_losses, ref_params = adam_trajectory(analytic_infidelity, start, 0.01, 30)

    np.testing.assert_allclose(losses, ref_losses, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(params, ref_params, rtol=1e-10)


def test_legacy_pulse_training():
    """``drag_legacy`` reproduces the pulse training of jaqsi up to 2a9132f.

    The doc example as it was then (Adam, ``lr=0.01``, 30 steps), with the
    results of 99ebc14, which this code reproduces bit for bit.
    """
    with PulseInformation.preserve_state():
        PulseInformation.set_envelope("drag_legacy", rwa=True)
        start = PulseInformation.gate_by_name("RX").params * 1.15
        losses, params = adam_trajectory(pulse_infidelity(), start, 0.01, 30)

    np.testing.assert_allclose(
        losses[[0, 9, 17, 28, 30]],
        [
            0.06279595635035018,
            2.455984355853591e-05,
            0.005852024299794256,
            4.172439736160882e-06,
            0.0002575426158604177,
        ],
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        params,
        [0.297624817828573, 0.5495358398394896, 6.03496082067464, 3.5238765222566895],
        rtol=1e-10,
    )
