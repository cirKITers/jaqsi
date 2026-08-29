from typing import List, Union, Callable
from contextlib import contextmanager
import numbers
import jax.numpy as jnp
import jax

# Imports to keep the api `from gates import ...`
from jaqsi.unitary import UnitaryGates
from jaqsi.pulses import (
    PulseGates,
    PulseParams,
    PulseEnvelope,  # noqa: F401
    PulseInformation,
    PulseParamManager,
)
from jaqsi.gateset import Barrier as BarrierOp

import logging

log = logging.getLogger(__name__)


# Meta class to avoid instantiating the Gates class
class GatesMeta(type):
    def __getattr__(cls, gate_name):
        def handler(*args, **kwargs):
            return cls._inner_getattr(gate_name, *args, **kwargs)

        # Dirty way to preserve information about the gate name
        handler.__name__ = gate_name
        return handler


def Barrier(wires: Union[int, List[int]], *args, **kwargs):
    """Thin wrapper for BarrierOp"""
    return BarrierOp(wires)


class Gates(metaclass=GatesMeta):
    """
    The entry point for applying gates to a circuit.

    Call gates as ``Gates.<Name>(...)`` inside a circuit function; the call is
    routed to either `UnitaryGates` or `PulseGates` depending on the `pulse`
    keyword.  Prefer this over calling the backends (`UnitaryGates`,
    `PulseGates`) or the matrix-level classes in :mod:`jaqsi.gateset` directly,
    so that the same circuit can run at either level.

    During circuit building, the pulse manager can be activated via
    `pulse_manager_context`, which slices the global model pulse parameters
    and passes them to each gate. Model pulse parameters act as element-wise
    scalers on the gate's optimized pulse parameters.

    Parameters
    ----------
    pulse : bool, optional
        Whether to run the gate at pulse level (`PulseGates`) instead of as an
        ideal unitary (`UnitaryGates`). Defaults to ``False``.

    Examples
    --------
    >>> Gates.RX(w, wires)
    >>> Gates.RX(w, wires, pulse=True)
    >>> Gates.RX(w, wires, pulse=True, pulse_params=pulse_params)
    """

    def __getattr__(self, gate_name):
        def handler(**kwargs):
            return self._inner_getattr(gate_name, **kwargs)

        return handler

    @classmethod
    def _inner_getattr(cls, gate_name, *args, **kwargs):
        if gate_name == "Barrier":
            return Barrier(*args, **kwargs)

        pulse = kwargs.pop("pulse", False)
        if not isinstance(pulse, bool):
            raise TypeError(f"'pulse' must be a bool, got {type(pulse).__name__}.")

        # Backend selection and kwargs filtering
        allowed_args = [
            "w",
            "wires",
            "phi",
            "theta",
            "omega",
            "noise_params",
            "random_key",
        ]
        if pulse:
            gate_backend = PulseGates
            allowed_args += ["pulse_params"]
        else:
            gate_backend = UnitaryGates

        if len(kwargs.keys() - allowed_args) > 0:
            # TODO: pulse params are always provided?
            log.debug(
                f"Unsupported keyword arguments: {list(kwargs.keys() - allowed_args)}"
            )

        kwargs = {k: v for k, v in kwargs.items() if k in allowed_args}
        pulse_params = kwargs.get("pulse_params")
        pulse_mgr = getattr(cls, "_pulse_mgr", None)

        # TODO: rework this part to convert to valid PulseParams earlier
        # Type check on pulse parameters
        if pulse_params is not None:
            # flatten pulse parameters
            if isinstance(pulse_params, (list, tuple)):
                flat_params = pulse_params

            elif isinstance(pulse_params, jax.core.Tracer):
                flat_params = jnp.ravel(pulse_params)

            elif isinstance(pulse_params, (jnp.ndarray, jnp.ndarray)):
                flat_params = pulse_params.flatten().tolist()
            elif isinstance(pulse_params, PulseParams):
                # extract the params in case a full object is given
                kwargs["pulse_params"] = pulse_params.params
                flat_params = pulse_params.params.flatten().tolist()

            else:
                raise TypeError(f"Unsupported pulse_params type: {type(pulse_params)}")

            # checks elements in flat parameters are real numbers or jax Tracer
            if not all(
                isinstance(x, (numbers.Real, jax.core.Tracer)) for x in flat_params
            ):
                raise TypeError(
                    "All elements in pulse_params must be int or float, "
                    f"got {pulse_params}, type {type(pulse_params)}. "
                )

        # Len check on pulse parameters
        if pulse_params is not None and not isinstance(pulse_mgr, PulseParamManager):
            n_params = PulseInformation.gate_by_name(gate_name).size
            if len(flat_params) != n_params:
                raise ValueError(
                    f"Gate '{gate_name}' expects {n_params} pulse parameters, "
                    f"got {len(flat_params)}"
                )

        # Pulse slicing + scaling
        if pulse and isinstance(pulse_mgr, PulseParamManager):
            n_params = PulseInformation.gate_by_name(gate_name).size
            scalers = pulse_mgr.get(n_params)
            base = PulseInformation.gate_by_name(gate_name).params
            kwargs["pulse_params"] = base * scalers

        # Call the selected gate backend
        gate = getattr(gate_backend, gate_name, None)
        if gate is None:
            raise AttributeError(
                f"'{gate_backend.__class__.__name__}' object "
                f"has no attribute '{gate_name}'"
            )

        return gate(*args, **kwargs)

    @classmethod
    @contextmanager
    def pulse_manager_context(cls, pulse_params: jnp.ndarray):
        """Temporarily set the global pulse manager for circuit building."""
        cls._pulse_mgr = PulseParamManager(pulse_params)
        try:
            yield
        finally:
            cls._pulse_mgr = None

    @classmethod
    def parse_gates(
        cls,
        gates: Union[str, Callable, List[Union[str, Callable]]],
        set_of_gates=None,
    ):
        set_of_gates = set_of_gates or cls

        if isinstance(gates, str):
            # if str, use the pennylane fct
            parsed_gates = [getattr(set_of_gates, f"{gates}")]
        elif isinstance(gates, list):
            parsed_gates = []
            for enc in gates:
                # if list, check if str or callable
                if isinstance(enc, str):
                    parsed_gates.append(getattr(set_of_gates, f"{enc}"))
                # check if callable
                elif callable(enc):
                    parsed_gates.append(enc)
                else:
                    raise ValueError(
                        f"Operation {enc} is not a valid gate or callable.\
                        Got {type(enc)}"
                    )
        elif callable(gates):
            # default to callable
            parsed_gates = [gates]
        elif gates is None:
            parsed_gates = [lambda *args, **kwargs: None]
        else:
            raise ValueError(
                f"Operation {gates} is not a valid gate or callable or list of both."
            )
        return parsed_gates

    @classmethod
    def is_rotational(cls, gate):
        return gate.__name__ in [
            "RX",
            "RY",
            "RZ",
            "Rot",
            "RXX",
            "RYY",
            "RZZ",
            "RZX",
            "CRX",
            "CRY",
            "CRZ",
            "CPhase",
        ]

    @classmethod
    def is_entangling(cls, gate):
        return gate.__name__ in [
            "CX",
            "CY",
            "CZ",
            "RXX",
            "RYY",
            "RZZ",
            "RZX",
            "CRX",
            "CRY",
            "CRZ",
            "CPhase",
        ]

    @classmethod
    def is_controlled(cls, gate):
        return gate.__name__ in ["CX", "CY", "CZ", "CRX", "CRY", "CRZ", "CPhase"]
