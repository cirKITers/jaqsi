## Gates

`Gates` is **the entry point for applying gates to a circuit**.
It records the gate on the active tape, attaches any requested noise, and routes the call to
either `UnitaryGates` or `PulseGates` depending on the `pulse` keyword.
The two backends below, and the matrix-level classes in `jaqsi.gateset`, are what `Gates`
dispatches to; call them directly only when you need the operation object itself (for
observables, `.dagger()` / `.power()`, or matrix algebra).

As the structure of the different classes used to realize pulse and unitary gates can be a bit confusing, the following diagram might help:

![Gate Structure](figures/pulses_structure_light.png#center#only-light)
![Gate Structure](figures/pulses_structure_dark.png#center#only-dark)

```python
from jaqsi import Gates
```

::: jaqsi.gates.Gates
    options:
      heading_level: 3

### Unitary Gates

```python
from jaqsi.gates import UnitaryGates
```

::: jaqsi.gates.UnitaryGates
    options:
      heading_level: 4

### Pulse Gates

```python
from jaqsi.gates import PulseGates
```

::: jaqsi.gates.PulseGates
    options:
      heading_level: 4

### Pulse Structure

```python
from jaqsi.gates import PulseParams
```

::: jaqsi.gates.PulseParams
    options:
      heading_level: 4

### Pulse Envelope

```python
from jaqsi.gates import PulseEnvelope
```

::: jaqsi.gates.PulseEnvelope
    options:
      heading_level: 4

### Pulse Information

```python
from jaqsi.gates import PulseInformation
```

::: jaqsi.gates.PulseInformation
    options:
      heading_level: 4
## Operations

```python
from jaqsi.operations import Operation
```

::: jaqsi.operations.Operation
    options:
      heading_level: 3

### Hermitian

```python
from jaqsi.operations import Hermitian
```

::: jaqsi.operations.Hermitian
    options:
      heading_level: 4

### Parametrized Hamiltonian

```python
from jaqsi.operations import ParametrizedHamiltonian
```

::: jaqsi.operations.ParametrizedHamiltonian
    options:
      heading_level: 4

### Pauli Rotation

```python
from jaqsi.gateset import PauliRot
```

::: jaqsi.gateset.PauliRot
    options:
      heading_level: 4


## Paulis

```python
from jaqsi.paulis import PauliWord, pauli_decompose, state_expectation
```

::: jaqsi.paulis.PauliWord
    options:
      heading_level: 3

::: jaqsi.paulis.pauli_decompose
    options:
      heading_level: 3

::: jaqsi.paulis.state_expectation
    options:
      heading_level: 3

## Noise

```python
from jaqsi.noise import KrausChannel, BitFlip, DepolarizingChannel, ThermalRelaxationError
```

::: jaqsi.noise.KrausChannel
    options:
      heading_level: 3

::: jaqsi.noise.QubitChannel
    options:
      heading_level: 3

## Math

```python
from jaqsi.math import quantum_fisher_information, fubini_study_metric, fidelity, trace_distance, phase_difference, partial_trace, marginalize_probs, logm_v
```

::: jaqsi.math.quantum_fisher_information
    options:
      heading_level: 3

::: jaqsi.math.fubini_study_metric
    options:
      heading_level: 3

::: jaqsi.math.fidelity
    options:
      heading_level: 3

::: jaqsi.math.trace_distance
    options:
      heading_level: 3

::: jaqsi.math.phase_difference
    options:
      heading_level: 3

::: jaqsi.math.partial_trace
    options:
      heading_level: 3

::: jaqsi.math.marginalize_probs
    options:
      heading_level: 3

::: jaqsi.math.logm_v
    options:
      heading_level: 3

## Quantum Optimal Control

```python
from jaqsi.qoc import QOC
```

::: jaqsi.qoc.QOC
    options:
      heading_level: 3

### Cost Functions

```python
from jaqsi.qoc import Cost
```

::: jaqsi.qoc.Cost
    options:
      heading_level: 4

### Cost Function Registry

```python
from jaqsi.qoc import CostFnRegistry
```

::: jaqsi.qoc.CostFnRegistry
    options:
      heading_level: 4

### Evolution Engine

```python
from jaqsi import Evolution
```

::: jaqsi.evolution.Evolution
    options:
      heading_level: 4

## Script

```python
from jaqsi.script import Script
```

::: jaqsi.script.Script
    options:
      heading_level: 3

## Drawing

```python
from jaqsi.drawing import TikzFigure
```

::: jaqsi.drawing.TikzFigure
    options:
      heading_level: 3

```python
from jaqsi.drawing import PulseEvent
```

::: jaqsi.drawing.PulseEvent
    options:
      heading_level: 3

## Tape

```python
from jaqsi.tape import recording, pulse_recording
```

::: jaqsi.tape.recording
    options:
      heading_level: 3

::: jaqsi.tape.pulse_recording
    options:
      heading_level: 3
