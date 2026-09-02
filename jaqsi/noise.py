"""Kraus noise channels.

A quantum channel is represented as an :class:`~jaqsi.operations.Operation`
whose ``kraus_ops`` are applied to a density matrix rather than a statevector.
Recording any channel on a tape is what switches :class:`~jaqsi.script.Script`
from statevector to density-matrix simulation.
"""

from typing import List, Union

import jax.numpy as jnp

from jaqsi.operations import (
    Operation,
    cdtype,
    _contract_and_restore,
)
from jaqsi.gateset import (
    Id,
    PauliX,
    PauliY,
    PauliZ,
)


def _check_unit_interval(value: float, name: str) -> None:
    """Raise ``ValueError`` if *value* is outside the closed interval [0, 1]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")


class KrausChannel(Operation):
    """Base class for noise channels defined by a set of Kraus operators.

    A Kraus channel \\phi(\\rho ) = \\sigma_k K_k \\rho  K_k\\dagger
    is the most general physical
    operation on a quantum state.  For a pure unitary gate there is a single
    operator K_0 = U satisfying K_0\\daggerK_0 = I; for noisy channels there are
    multiple operators.

    Subclasses must implement :meth:`kraus_matrices` and return a list of JAX
    arrays.  :meth:`apply_to_state` is intentionally left unimplemented:
    Kraus channels require a density-matrix representation and cannot be
    applied to a pure statevector in general.
    """

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the list of Kraus operators for this channel.

        Returns:
            List of 2-D JAX arrays, each of shape ``(2**k, 2**k)`` where k
            is the number of target qubits.

        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError

    @property
    def matrix(self) -> jnp.ndarray:
        """Raises TypeError — noise channels have no single unitary matrix.

        Raises:
            TypeError: Always raised; use :meth:`apply_to_density` instead.
        """
        raise TypeError(
            f"{self.__class__.__name__} is a noise channel and has no single "
            "unitary matrix. Use apply_to_density() instead."
        )

    def apply_to_state(self, state: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Raises TypeError — noise channels require density-matrix simulation.

        Args:
            state: Statevector (unused).
            n_qubits: Number of qubits (unused).

        Raises:
            TypeError: Always raised; use ``execute(type='density')`` instead.
        """
        raise TypeError(
            f"{self.__class__.__name__} is a noise channel and cannot be "
            "applied to a pure statevector. Use execute(type='density') instead."
        )

    def apply_to_state_tensor(self, psi: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Raises TypeError — noise channels require density-matrix simulation."""
        raise TypeError(
            f"{self.__class__.__name__} is a noise channel and cannot be "
            "applied to a pure statevector. Use execute(type='density') instead."
        )

    def apply_to_density(self, rho: jnp.ndarray, n_qubits: int) -> jnp.ndarray:
        """Apply
        \\phi(\\rho ) = \\sigma_k K_k \\rho  K_k\\dagger using tensor-contraction.

        Uses the shared :func:`_contract_and_restore` helper, summing the
        result over all Kraus operators.

        Args:
            rho: Density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
            n_qubits: Total number of qubits in the circuit.

        Returns:
            Updated density matrix of shape ``(2**n_qubits, 2**n_qubits)``.
        """
        k = len(self.wires)
        dim = 2**n_qubits
        bra_wires = [w + n_qubits for w in self.wires]
        rho_out = jnp.zeros_like(rho)

        for K in self.kraus_matrices():
            K_t = K.reshape((2,) * 2 * k)
            K_conj_t = jnp.conj(K_t)
            rho_t = rho.reshape((2,) * 2 * n_qubits)
            rho_t = _contract_and_restore(rho_t, K_t, k, self.wires)
            rho_t = _contract_and_restore(rho_t, K_conj_t, k, bra_wires)
            rho_out = rho_out + rho_t.reshape(dim, dim)

        return rho_out


class BitFlip(KrausChannel):
    r"""Single-qubit bit-flip (Pauli-X) error channel.

    .. math::
        K_0 = \sqrt{1-p}\,I, \quad K_1 = \sqrt{p}\,X

    where *p* \\in [0, 1] is the probability of a bit flip.
    """

    _num_wires = 1
    _param_names = ("p",)

    def __init__(self, p: float, wires: Union[int, List[int]] = 0) -> None:
        """Initialise a bit-flip channel.

        Args:
            p: Bit-flip probability, must be in [0, 1].
            wires: Qubit index or list of qubit indices this channel acts on.

        Raises:
            ValueError: If *p* is outside [0, 1].
        """
        _check_unit_interval(p, "p")
        self.p = p
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the two Kraus operators for the bit-flip channel.

        Returns:
            List ``[K0, K1]`` where K0 = \\sqrt (1-p)·I and K1 = \\sqrt p·X.
        """
        p = self.p
        K0 = jnp.sqrt(1 - p) * Id._matrix
        K1 = jnp.sqrt(p) * PauliX._matrix
        return [K0, K1]


class PhaseFlip(KrausChannel):
    r"""Single-qubit phase-flip (Pauli-Z) error channel.

    .. math::
        K_0 = \sqrt{1-p}\,I, \quad K_1 = \sqrt{p}\,Z

    where *p* \\in [0, 1] is the probability of a phase flip.
    """

    _num_wires = 1
    _param_names = ("p",)

    def __init__(self, p: float, wires: Union[int, List[int]] = 0) -> None:
        """Initialise a phase-flip channel.

        Args:
            p: Phase-flip probability, must be in [0, 1].
            wires: Qubit index or list of qubit indices this channel acts on.

        Raises:
            ValueError: If *p* is outside [0, 1].
        """
        _check_unit_interval(p, "p")
        self.p = p
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the two Kraus operators for the phase-flip channel.

        Returns:
            List ``[K0, K1]`` where K0 = \\sqrt (1-p)·I and K1 = \\sqrt p·Z.
        """
        p = self.p
        K0 = jnp.sqrt(1 - p) * Id._matrix
        K1 = jnp.sqrt(p) * PauliZ._matrix
        return [K0, K1]


class DepolarizingChannel(KrausChannel):
    r"""Single-qubit depolarizing channel.

    .. math::
        K_0 = \sqrt{1-p}\,I,\quad K_1 = \sqrt{p/3}\,X,\quad
        K_2 = \sqrt{p/3}\,Y,\quad K_3 = \sqrt{p/3}\,Z

    where *p* \\in [0, 1].  At p = 3/4 the channel is fully depolarizing.
    """

    _num_wires = 1
    _param_names = ("p",)

    def __init__(self, p: float, wires: Union[int, List[int]] = 0) -> None:
        """Initialise a depolarizing channel.

        Args:
            p: Depolarization probability, must be in [0, 1].
            wires: Qubit index or list of qubit indices this channel acts on.

        Raises:
            ValueError: If *p* is outside [0, 1].
        """
        _check_unit_interval(p, "p")
        self.p = p
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the four Kraus operators for the depolarizing channel.

        Returns:
            List ``[K0, K1, K2, K3]`` corresponding to I, X, Y, Z components.
        """
        p = self.p
        K0 = jnp.sqrt(1 - p) * Id._matrix
        K1 = jnp.sqrt(p / 3) * PauliX._matrix
        K2 = jnp.sqrt(p / 3) * PauliY._matrix
        K3 = jnp.sqrt(p / 3) * PauliZ._matrix
        return [K0, K1, K2, K3]


class AmplitudeDamping(KrausChannel):
    r"""Single-qubit amplitude damping channel.

    .. math::
        K_0 = \begin{pmatrix}1 & 0\\ 0 & \sqrt{1-\gamma}\end{pmatrix},\quad
        K_1 = \begin{pmatrix}0 & \sqrt{\gamma}\\ 0 & 0\end{pmatrix}

    where *\\gamma* \\in [0, 1] is the probability of
    energy loss (|1\\rangle -> |0\\rangle).
    """

    _num_wires = 1
    _param_names = ("gamma",)

    def __init__(self, gamma: float, wires: Union[int, List[int]] = 0) -> None:
        """Initialise an amplitude damping channel.

        Args:
            gamma: Energy-loss probability, must be in [0, 1].
            wires: Qubit index or list of qubit indices this channel acts on.

        Raises:
            ValueError: If *gamma* is outside [0, 1].
        """
        _check_unit_interval(gamma, "gamma")
        self.gamma = gamma
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the two Kraus operators for the amplitude damping channel.

        Returns:
            List ``[K0, K1]`` as defined in the class docstring.
        """
        g = self.gamma
        K0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1 - g)]], dtype=cdtype())
        K1 = jnp.array([[0.0, jnp.sqrt(g)], [0.0, 0.0]], dtype=cdtype())
        return [K0, K1]


class PhaseDamping(KrausChannel):
    r"""Single-qubit phase damping (dephasing) channel.

    .. math::
        K_0 = \begin{pmatrix}1 & 0\\ 0 & \sqrt{1-\gamma}\end{pmatrix},\quad
        K_1 = \begin{pmatrix}0 & 0\\ 0 & \sqrt{\gamma}\end{pmatrix}

    where *\\gamma* \\in [0, 1] is the phase damping probability.
    """

    _num_wires = 1
    _param_names = ("gamma",)

    def __init__(self, gamma: float, wires: Union[int, List[int]] = 0) -> None:
        """Initialise a phase damping channel.

        Args:
            gamma: Phase-damping probability, must be in [0, 1].
            wires: Qubit index or list of qubit indices this channel acts on.

        Raises:
            ValueError: If *gamma* is outside [0, 1].
        """
        _check_unit_interval(gamma, "gamma")
        self.gamma = gamma
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the two Kraus operators for the phase damping channel.

        Returns:
            List ``[K0, K1]`` as defined in the class docstring.
        """
        g = self.gamma
        K0 = jnp.array([[1.0, 0.0], [0.0, jnp.sqrt(1 - g)]], dtype=cdtype())
        K1 = jnp.array([[0.0, 0.0], [0.0, jnp.sqrt(g)]], dtype=cdtype())
        return [K0, K1]


class ThermalRelaxationError(KrausChannel):
    r"""Single-qubit thermal relaxation error channel.

    Models simultaneous T_1 energy relaxation and T_2 dephasing.  Two regimes
    are handled:

    T_2 <= T_1 (Markovian dephasing + reset):
        Six Kraus operators built from p_z (phase-flip probability), p_r0
        (reset-to-|0\\rangle probability) and p_r1 (reset-to-|1\\rangle probability).

    T_2 > T_1 (non-Markovian; Choi matrix decomposition):
        The Choi matrix is assembled from the relaxation/dephasing rates, then
        diagonalised; Kraus operators are K_i = \sqrt \lambda_i · mat(v_i).

    Attributes:
        pe: Excited-state population (thermal population of |1\\rangle).
        t1: T_1 longitudinal relaxation time.
        t2: T_2 transverse dephasing time.
        tg: Gate duration.
    """

    _num_wires = 1
    _param_names = ("pe", "t1", "t2", "tg")

    def __init__(
        self,
        pe: float,
        t1: float,
        t2: float,
        tg: float,
        wires: Union[int, List[int]] = 0,
    ) -> None:
        """Initialise a thermal relaxation error channel.

        Args:
            pe: Excited-state population (thermal population of |1\\rangle), in [0, 1].
            t1: T_1 longitudinal relaxation time, must be > 0.
            t2: T_2 transverse dephasing time, must be > 0 and <= 2·T_1.
            tg: Gate duration, must be >= 0.
            wires: Qubit index or list of qubit indices this channel acts on.

        Raises:
            ValueError: If any parameter violates the stated constraints.
        """
        _check_unit_interval(pe, "pe")
        if t1 <= 0:
            raise ValueError("t1 must be > 0.")
        if t2 <= 0:
            raise ValueError("t2 must be > 0.")
        if t2 > 2 * t1:
            raise ValueError("t2 must be <= 2·t1.")
        if tg < 0:
            raise ValueError("tg must be >= 0.")
        self.pe = pe
        self.t1 = t1
        self.t2 = t2
        self.tg = tg
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the Kraus operators for the thermal relaxation channel.

        The number of operators depends on the regime:

        * T_2 <= T_1: six operators (identity, phase-flip, two reset-to-|0\\rangle,
          two reset-to-|1\\rangle).
        * T_2 > T_1: four operators derived from the Choi matrix eigendecomposition.

        Returns:
            List of 2x2 JAX arrays representing the Kraus operators.
        """
        pe, t1, t2, tg = self.pe, self.t1, self.t2, self.tg

        eT1 = jnp.exp(-tg / t1)
        p_reset = 1.0 - eT1
        eT2 = jnp.exp(-tg / t2)

        if t2 <= t1:
            # --- Case T_2 <= T_1: six Kraus operators ---
            pz = (1.0 - p_reset) * (1.0 - eT2 / eT1) / 2.0
            pr0 = (1.0 - pe) * p_reset
            pr1 = pe * p_reset
            pid = 1.0 - pz - pr0 - pr1

            K0 = jnp.sqrt(pid) * jnp.eye(2, dtype=cdtype())
            K1 = jnp.sqrt(pz) * jnp.array([[1, 0], [0, -1]], dtype=cdtype())
            K2 = jnp.sqrt(pr0) * jnp.array([[1, 0], [0, 0]], dtype=cdtype())
            K3 = jnp.sqrt(pr0) * jnp.array([[0, 1], [0, 0]], dtype=cdtype())
            K4 = jnp.sqrt(pr1) * jnp.array([[0, 0], [1, 0]], dtype=cdtype())
            K5 = jnp.sqrt(pr1) * jnp.array([[0, 0], [0, 1]], dtype=cdtype())
            return [K0, K1, K2, K3, K4, K5]

        else:
            # --- Case T_2 > T_1: Choi matrix decomposition ---
            # Choi matrix (column-major / reshaping convention matching PennyLane)
            choi = jnp.array(
                [
                    [1 - pe * p_reset, 0, 0, eT2],
                    [0, pe * p_reset, 0, 0],
                    [0, 0, (1 - pe) * p_reset, 0],
                    [eT2, 0, 0, 1 - (1 - pe) * p_reset],
                ],
                dtype=cdtype(),
            )
            eigenvalues, eigenvectors = jnp.linalg.eigh(choi)
            # Each eigenvector (column of eigenvectors) reshaped as 2x2 -> one Kraus op
            kraus = []
            for i in range(4):
                lam = eigenvalues[i]
                vec = eigenvectors[:, i]
                mat = jnp.sqrt(jnp.abs(lam)) * vec.reshape(2, 2, order="F")
                kraus.append(mat.astype(cdtype()))
            return kraus


class QubitChannel(KrausChannel):
    """Generic Kraus channel from a user-supplied list of Kraus operators.

    This replaces PennyLane's ``qml.QubitChannel`` and accepts an arbitrary set
    of Kraus matrices satisfying \\sigma_k K_k\\dagger K_k = I.

    Example::

        kraus_ops = [jnp.sqrt(0.9) * jnp.eye(2), jnp.sqrt(0.1) * PauliX._matrix]
        QubitChannel(kraus_ops, wires=0)
    """

    def __init__(
        self, kraus_ops: List[jnp.ndarray], wires: Union[int, List[int]] = 0
    ) -> None:
        """Initialise a generic Kraus channel.

        Args:
            kraus_ops: List of Kraus matrices.  Each must be a square 2D array
                of dimension ``2**k x 2**k`` where k = ``len(wires)``.
            wires: Qubit index or list of qubit indices this channel acts on.
        """
        self._kraus_ops = [jnp.asarray(K, dtype=cdtype()) for K in kraus_ops]
        super().__init__(wires=wires)

    def kraus_matrices(self) -> List[jnp.ndarray]:
        """Return the stored Kraus operators.

        Returns:
            List of Kraus operator matrices.
        """
        return self._kraus_ops

