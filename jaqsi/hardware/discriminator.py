"""Readout discrimination: labelling IQ points as |0>, |1> or leakage."""

from dataclasses import dataclass
from itertools import combinations
from typing import Any, List, Optional, Tuple

import numpy as np
from scipy.cluster.vq import kmeans2, vq
from scipy.special import xlogy

INT32_MAX = 2**31 - 1
RESTARTS = 20  # k-means runs per class count
# Least growth of the |1> share under RX(pi), in standard errors.  Five is the
# conventional discovery threshold: shot noise exceeds it with p < 3e-7.
GROWTH_Z = 5.0


@dataclass
class Discriminator:
    """IQ clusters of each measured qubit.

    Attributes:
        centers: Cluster centers per qubit, shape ``(k, 2)`` in label order:
            |0>, |1> and, when ``k == 3``, the leakage class.
    """

    centers: List[np.ndarray]

    def labels(self, iq: np.ndarray) -> np.ndarray:
        """Label each IQ point by its nearest center.

        Args:
            iq: IQ points of shape ``(n_qubits, 2, shots)``.

        Returns:
            Labels of shape ``(n_qubits, shots)``: 0, 1, or 2 for leakage.
        """
        return np.stack([vq(p.T, c)[0] for c, p in zip(self.centers, iq)])


def fit(
    iq: np.ndarray, iq_pi: Optional[np.ndarray] = None, seed: int = 0
) -> Discriminator:
    """Calibrate a discriminator on IQ points of the thermal state and of RX(pi).

    Per qubit, k-means clusters the points of both runs together into two and
    into three classes, several times from random starts, and the Bayesian
    information criterion picks the best clustering, or one cluster around
    the mean of all points, which leaves no |1> to name.  |0> is the cluster
    with the largest share of the thermal run.  |1> is, of the others, the
    cluster whose share grows most from the thermal run to the RX(pi) run.
    A two-proportion z-test requires this growth to exceed ``GROWTH_Z``
    standard errors, since a drive that misses the qubit leaves the shares
    as they were.  A third cluster, if any, is the leakage class: a separate
    readout class, not proof of physical leakage, which needs an independent
    preparation or characterization.  Leakage starts from |1>, so the
    thermal run alone may hold too few leaked shots to find it.

    Without RX(pi) points, |1> is the second most populated cluster of the
    thermal run, which assumes the thermal population of |1> exceeds the
    leakage, and only the one-cluster check applies.

    Args:
        iq: IQ points of the thermal state, shape ``(n_qubits, 2, shots)``.
        iq_pi: IQ points after RX(pi) on every qubit, of the same shape up to
            the number of shots; ``None`` to fit on the thermal state alone.
        seed: Seed of the k-means initialization.

    Returns:
        Discriminator with the centers of each qubit in label order.

    Raises:
        ValueError: If the points of a qubit form one cluster, or RX(pi) does
            not grow the share of its |1> cluster significantly.
    """
    rng = np.random.default_rng(seed)
    centers = []
    for q, points in enumerate(iq):
        x = points.T.astype(float)
        if iq_pi is not None:
            x = np.vstack([x, iq_pi[q].T])
        # One class needs no k-means: its center is the mean.
        fits = [(x.mean(axis=0, keepdims=True), np.zeros(len(x), dtype=int))]
        fits += [
            kmeans2(x, k, minit="++", rng=rng) for k in (2, 3) for _ in range(RESTARTS)
        ]
        c, labels = min(fits, key=lambda f: _bic(x, *f))
        if len(c) == 1:
            raise ValueError(
                f"The IQ points of qubit {q} form one cluster; there is no "
                "separable |1>."
            )
        n = points.shape[1]  # thermal shots, first in x
        thermal = np.bincount(labels[:n], minlength=len(c)) / n
        if iq_pi is None:
            growth = thermal  # the next most populated becomes |1>
        else:
            growth = np.bincount(labels[n:], minlength=len(c)) / (len(x) - n) - thermal
        ground = int(np.argmax(thermal))
        others = sorted(set(range(len(c))) - {ground}, key=lambda k: -growth[k])
        if iq_pi is not None:
            # Standard error of the growth if RX(pi) changed nothing, with the
            # share of |1> pooled over both runs.
            share = np.mean(labels == others[0])
            error = np.sqrt(share * (1 - share) * (1 / n + 1 / (len(x) - n)))
            if growth[others[0]] <= GROWTH_Z * error:
                raise ValueError(
                    f"RX(pi) does not grow the |1> cluster of qubit {q} "
                    "significantly; the drive may miss the qubit."
                )
        centers.append(c[[ground, *others]])
    return Discriminator(centers)


def _bic(x: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    """Bayesian information criterion of a k-means clustering, lower is better.

    The points are modeled as a mixture of isotropic Gaussians of one shared
    width around the centers, weighted by the cluster populations.  Unlike
    the squared error alone, the weights penalize splitting one Gaussian
    cluster in two.
    """
    n, d = x.shape
    k = len(centers)
    counts = np.bincount(labels, minlength=k)
    variance = ((x - centers[labels]) ** 2).sum() / (n * d)
    log_likelihood = xlogy(counts, counts / n).sum() - n * d / 2 * (
        np.log(2 * np.pi * variance) + 1
    )
    n_params = k * d + (k - 1) + 1  # centers, weights, width
    return n_params * np.log(n) - 2 * log_likelihood


def export(discriminator: Discriminator, iq: np.ndarray) -> List[Tuple[int, int, int]]:
    """Linear discriminator between |0> and |1> per qubit, in device integers.

    Fisher's linear discriminant on the points labelled 0 and 1: the normal
    ``a`` of the separating line solves ``S a = m1 - m0`` with the pooled
    within-class scatter ``S`` and the class means ``m0``, ``m1``; the line
    passes through their midpoint.  A device reads 1 where
    ``a_I * I + a_Q * Q + b >= 0``, with ``(a_I, a_Q, b)`` as 32-bit signed
    integers (qiclib's ``Recording.state_config``).  Rounding would turn the
    small coefficients of ``a`` into zero, so the triplet is scaled until its
    largest entry fills the integer range, which leaves the line in place.

    Args:
        discriminator: Calibrated discriminator.
        iq: IQ points of shape ``(n_qubits, 2, shots)`` holding both |0>
            and |1> shots, e.g. the calibration points.

    Returns:
        ``(a_I, a_Q, b)`` per qubit.
    """
    configs = []
    for points, labels in zip(iq, discriminator.labels(iq)):
        x0, x1 = points[:, labels == 0], points[:, labels == 1]
        m0, m1 = x0.mean(axis=1), x1.mean(axis=1)
        deviations = np.hstack([x0 - m0[:, None], x1 - m1[:, None]])
        a = np.linalg.solve(deviations @ deviations.T, m1 - m0)
        line = np.append(a, -a @ (m0 + m1) / 2)
        a_i, a_q, b = np.round(line * INT32_MAX / np.abs(line).max())
        configs.append((int(a_i), int(a_q), int(b)))
    return configs


def plot_iq(
    iq: np.ndarray,
    discriminator: Discriminator,
    qubit: int = 0,
    ax: Optional[Any] = None,
    line: Optional[Tuple[int, int, int]] = None,
) -> Any:
    """Scatter a qubit's IQ points by label, with the centers and boundaries.

    The dotted cluster boundaries are those of :meth:`Discriminator.labels`,
    which takes the nearest center: the perpendicular bisector of each pair
    of centers, where no third center is nearer.  They cross the view but
    leave the axis limits to the points.

    The dashed device 0/1 line is ``a_I * I + a_Q * Q + b = 0``, the only
    boundary the device's state estimation holds.  The device uses the line
    that :func:`export` finds on the calibration points; pass it as *line*
    to draw it on other points, such as those of RX(pi).

    Args:
        iq: IQ points of shape ``(n_qubits, 2, shots)``.
        discriminator: Calibrated discriminator.
        qubit: Qubit to plot.
        ax: Matplotlib axes to draw on; ``None`` for a new figure.
        line: ``(a_I, a_Q, b)`` of the qubit, as :func:`export` returns it;
            ``None`` to export it from *iq*, which needs both |0> and |1>
            shots.

    Returns:
        The Matplotlib axes.
    """
    import matplotlib.pyplot as plt  # lazy - matplotlib is dev-only
    from matplotlib.lines import Line2D

    if ax is None:
        _, ax = plt.subplots()
    points, labels = iq[qubit], discriminator.labels(iq)[qubit]
    centers = discriminator.centers[qubit]
    for label, name in enumerate(["|0>", "|1>", "leakage"][: len(centers)]):
        ax.scatter(*points[:, labels == label], s=4, alpha=0.4, label=name)
    ax.scatter(*centers.T, marker="x", color="black", label="centers")

    # Long enough to cross the view from any pair's midpoint.
    t = np.linspace(-2, 2, 2001)[:, None] * np.hypot(*np.ptp(points, axis=1))
    edges = []
    for i, j in combinations(range(len(centers)), 2):
        normal = centers[j] - centers[i]
        direction = np.array([-normal[1], normal[0]]) / np.hypot(*normal)
        xy = (centers[i] + centers[j]) / 2 + t * direction
        xy[~np.isin(vq(xy, centers)[0], [i, j])] = np.nan
        edges += [xy, [[np.nan, np.nan]]]  # NaN breaks the line
    boundaries = Line2D(
        *np.vstack(edges).T, color="black", linestyle=":", label="cluster boundaries"
    )
    ax.add_artist(boundaries)  # unlike plot, it leaves the axis limits alone

    if line is None:
        line = export(discriminator, iq)[qubit]
    a_i, a_q, b = np.array(line, dtype=float) / np.hypot(*line[:2])
    nearest = -b * np.array([a_i, a_q])  # point of the line nearest the origin
    ax.axline(
        nearest,
        nearest + [-a_q, a_i],
        color="black",
        linestyle="--",
        label="device 0/1 line",
    )
    ax.set(xlabel="I", ylabel="Q", aspect="equal")
    ax.legend()
    return ax
