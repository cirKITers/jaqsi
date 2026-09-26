import os

import jax
import pytest

# Enable x64 before jaqsi is imported anywhere in the session, not per test
# module: modules that build JAX arrays at import time bake in whatever dtype
# was active then.  The tests' 1e-10 tolerances need complex128 throughout.
# ``JAX_ENABLE_X64=0`` opts out, which is how ``tests/test_gpu.py`` gets its
# complex64 run.
if os.environ.get("JAX_ENABLE_X64") != "0":
    jax.config.update("jax_enable_x64", True)

from jaqsi.pulses import PulseInformation  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_pulse_information_state():
    """Run every test from the canonical pulse configuration.

    The pulse backend stores the active envelope, RWA flag, frame, pulse
    parameters, and compiled-solver cache in process-global state.  xdist
    workers execute unrelated tests in the same Python process, so preserving
    whatever state a worker happens to have at test start is not enough: a
    polluted worker would keep restoring the polluted state.  Reset before and
    after each test to make ordering and worker assignment irrelevant.
    """
    PulseInformation.reset_defaults()
    try:
        yield
    finally:
        PulseInformation.reset_defaults()
