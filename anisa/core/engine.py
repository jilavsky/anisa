"""
The calculation engine: model intensity along any number of directions from any number of
populations.

For one direction (unit vector ``q_hat``) and one population the intensity is the orientation
and size average of the single-particle cross section (Allen, eq. 3)::

    I(Q) = 1e8 phi  sum_r v_r / V_r  sum_k W_k  F_r(Q, K(beta, X_k))              [1/cm]

with ``r`` the size classes (volume-fraction shares ``v_r``, volumes ``V_r``), ``k`` the
orientation grid nodes with weights ``W_k = p(alpha_k) b(omega_k) * quadrature weight``
(``sum W = 1``), ``X_k = cos(eta_k)`` from Allen's eq. 1 and ``F`` one of the cross sections in
``anisa.core.formfactor``. Igor evaluated a form-factor lookup table at every one of its
91 x 201 orientation nodes for every Q point. Here:

1. **K binning.** ``F`` depends on the orientation only through ``K``, so the orientation
   weights are binned once onto a grid of K nodes (cubic rule, ``points_per_period`` nodes per
   2 pi of form-factor phase). Each Q point then costs one short dot product instead of a sum
   over the whole orientation grid. The orientation grid itself is made fine enough to resolve
   the oscillations (``resolved_grid_size``), since binning reduces cost, not quadrature error.
2. **Log-space convolution** (formula 4c with a size distribution). The 4c cross section depends
   on radius and orientation only through the product ``R K``, so with ``u = ln(R K)`` the size
   and orientation densities simply convolve: ``rho(u) = rho_R (*) rho_K`` and
   ``I(Q) = sum_u rho(u) A^2(Q e^u)``. This costs ``n_R + n_K`` rather than ``n_R n_K`` form-factor
   evaluations per Q.
3. **Q bands.** The resolution needed grows with Q. Q is split into bands (factor 4 in Q) and
   each is computed at the resolution its own largest Q requires.
4. **Caching.** Weights depend only on the population, ``X`` only on the direction, and the
   intensity shape only on (direction, population, settings); all are cached, so changing a
   population's volume fraction or interference parameters, or one population in a fit,
   recomputes nothing else.
5. **Threads.** Independent (direction x population) tasks run on a thread pool; numpy and scipy
   release the GIL in the large array operations that dominate.

Units: Q in 1/A, intensity in 1/cm (absolute), as in the rest of Anisa.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from anisa.core import derived, geometry
from anisa.core.formfactor import (
    cross_section,
    sphere_amplitude,
    sphere_amplitude_sq,
    spheroid_volume,
)
from anisa.core.model import AnisaModel, CalcSettings, Population
from anisa.core.orientation import OrientationGrid
from anisa.core.sizedist import radius_classes, radius_range, volume_density

#: Orientation-grid nodes per unit of ``span = Q R |beta - 1|`` (the total phase advance of the
#: form factor along K). Set from ``benchmarks/convergence.py``: half of these fails at large
#: span, twice these changes nothing.
ALPHA_NODES_PER_SPAN = 1.0
OMEGA_NODES_PER_SPAN = 2.0
#: Hard cap on orientation-grid nodes per task (memory and time); a warning is issued if it binds.
MAX_GRID_NODES = 4_000_000
#: Hard cap on K nodes / log-grid cells.
MAX_K_NODES = 400_000
#: Q bands: each band spans this factor in Q.
BAND_RATIO = 4.0
#: Radius classes (direct sum over sizes) per half period of the form-factor oscillation:
#: spacing <= pi / (NYQUIST_FACTOR * Q_max K_max). Midpoint sums on smooth distributions are
#: spectrally accurate above the Nyquist rate.
NYQUIST_FACTOR = 1.5
MAX_RADIUS_CLASSES = 400
#: Elements of the (q, K) matrix evaluated at once.
_CHUNK_ELEMENTS = 2_000_000
#: Products (n_R * n_K) above which the log-space convolution uses the FFT.
_FFT_THRESHOLD = 2_000_000

PER_ANGSTROM_TO_PER_CM = derived.PER_ANGSTROM_TO_PER_CM
DELTA_RHO_TO_PER_A2 = 1.0e-16  # cm**-2 -> A**-2


@dataclass
class DirectionResult:
    """Model for one direction.

    Attributes:
        q: Q values [1/A].
        intensity: total model [1/cm]: all populations plus the background.
        populations: each population's contribution, shape ``(n_populations, n_q)``.
        background: the background that was added [1/cm].
    """

    q: np.ndarray
    intensity: np.ndarray
    populations: np.ndarray
    background: float


@dataclass
class ModelResult:
    """Result of :meth:`Engine.evaluate`.

    ``surface_areas`` [cm**2/cm**3] and ``number_densities`` [cm**-3] are per population.
    """

    directions: list[DirectionResult]
    surface_areas: np.ndarray
    number_densities: np.ndarray

    @property
    def total_surface_area(self) -> float:
        return float(np.sum(self.surface_areas))


# ---------------------------------------------------------------------------------------------
# resolution rules
# ---------------------------------------------------------------------------------------------


def _quantise(n: int) -> int:
    """Round ``n`` up to a multiple of 8, or to 16 steps per octave above 128.

    Few distinct grid sizes means more cache hits (a fit that changes a radius slightly keeps
    the same orientation grid) at an average cost of about 3 % more nodes.
    """
    if n <= 128:
        return -(-n // 8) * 8
    step = 1 << (n.bit_length() - 5)  # 2**(floor(log2 n) - 4)
    return -(-n // step) * step


def resolved_grid_size(
    settings: CalcSettings, span: float, minimum: tuple[int, int] = (0, 0)
) -> tuple[int, int]:
    """Orientation-grid size ``(n_alpha, n_omega)`` that resolves the form factor and the
    orientation distribution.

    ``span = Q_max R_max |beta - 1|`` is how far the phase ``Q R K`` of the form factor advances
    as the orientation sweeps from X = 0 to 1; ``minimum`` is what the sharpness of the
    population's orientation distribution needs (``OrientationDistribution.required_nodes``).
    The user's settings are a lower bound; the larger requirement wins.
    """
    n_alpha = _quantise(
        max(settings.n_alpha, minimum[0], math.ceil(ALPHA_NODES_PER_SPAN * span) + 8)
    )
    n_omega = _quantise(
        max(settings.n_omega, minimum[1], math.ceil(OMEGA_NODES_PER_SPAN * span) + 8)
    )
    capped = False
    while n_alpha * n_omega > MAX_GRID_NODES:  # keep the aspect, stay within the cap
        capped = True
        shrunk = (
            max(settings.n_alpha, int(n_alpha * 0.9)),
            max(settings.n_omega, int(n_omega * 0.9)),
        )
        if shrunk == (n_alpha, n_omega):
            break
        n_alpha, n_omega = shrunk
    if capped:
        warnings.warn(
            f"orientation grid capped at {n_alpha} x {n_omega} nodes; the form factor "
            f"(phase span {span:.0f}) is not fully resolved, high-Q results may be inaccurate",
            RuntimeWarning,
            stacklevel=2,
        )
    return n_alpha, n_omega


def k_node_count(settings: CalcSettings, span: float) -> int:
    """Number of K nodes: ``points_per_period`` per 2 pi of form-factor phase."""
    count = math.ceil(settings.points_per_period * span / (2.0 * math.pi)) + 2
    return int(min(max(count, 8), MAX_K_NODES))


def q_bands(q: np.ndarray) -> list[np.ndarray]:
    """Split Q into bands of at most ``BAND_RATIO`` in Q, as arrays of indices into ``q``.

    The resolution a band needs is set by its own largest Q, so low-Q bands are cheap.
    """
    if q.size == 0:
        return []
    order = np.argsort(q, kind="stable")
    sorted_q = q[order]
    bands = []
    stop = q.size
    while stop > 0:
        top = sorted_q[stop - 1]
        if top <= 0:
            bands.append(order[:stop])
            break
        start = int(np.searchsorted(sorted_q, top / BAND_RATIO, side="right"))
        start = min(start, stop - 1)  # at least one point per band
        bands.append(order[start:stop])
        stop = start
    return bands


# ---------------------------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------------------------


class _LRU:
    """Small thread-safe LRU cache with single-flight computation.

    If several threads ask for the same missing key at once, one computes it and the others
    wait for the result instead of repeating a large calculation.
    """

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._data: OrderedDict = OrderedDict()
        self._pending: dict = {}
        self._lock = threading.Lock()

    def get(self, key, make: Callable[[], object]):
        while True:
            with self._lock:
                if key in self._data:
                    self._data.move_to_end(key)
                    return self._data[key]
                event = self._pending.get(key)
                if event is None:
                    event = self._pending[key] = threading.Event()
                    owner = True
                else:
                    owner = False
            if not owner:
                event.wait()
                continue  # the owner stored the value (or failed: then we become the owner)
            try:
                value = make()
                with self._lock:
                    self._data[key] = value
                    self._data.move_to_end(key)
                    while len(self._data) > self.maxsize:
                        self._data.popitem(last=False)
                return value
            finally:
                with self._lock:
                    self._pending.pop(key, None)
                event.set()

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


def _array_key(arr: np.ndarray) -> tuple:
    arr = np.ascontiguousarray(arr, dtype=float)
    return (arr.shape, hashlib.blake2b(arr.tobytes(), digest_size=16).digest())


# ---------------------------------------------------------------------------------------------
# binning
# ---------------------------------------------------------------------------------------------


def _cubic_deposit(position: np.ndarray, weights: np.ndarray, n_real: int) -> np.ndarray:
    """Spread ``weights`` over equally spaced nodes with cubic Lagrange weights.

    ``position`` is in padded-node units: the ``n_real`` real nodes sit at 1 ... ``n_real``, with
    one padding node at 0 and one at ``n_real + 1``. A sample in the interval ``[i, i + 1]`` is
    shared among nodes ``i - 1 ... i + 2``, so ``sum_k h_k F(node_k)`` equals the weighted sum of
    ``F`` at the samples for any ``F`` that is a cubic polynomial between nodes. The weights sum
    to one, but individual ``h_k`` can be slightly negative.
    """
    i0 = np.clip(np.floor(position).astype(np.intp), 1, n_real - 1)  # interval [i0, i0 + 1]
    t = position - i0
    w_m1 = -t * (t - 1.0) * (t - 2.0) / 6.0
    w_0 = (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0
    w_1 = -(t + 1.0) * t * (t - 2.0) / 2.0
    w_2 = (t + 1.0) * t * (t - 1.0) / 6.0
    size = n_real + 2
    return (
        np.bincount(i0 - 1, weights * w_m1, size)
        + np.bincount(i0, weights * w_0, size)
        + np.bincount(i0 + 1, weights * w_1, size)
        + np.bincount(i0 + 2, weights * w_2, size)
    )


def _bin_on_k(
    beta: float, x: np.ndarray, weights: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Bin orientation weights onto equally spaced K nodes (cubic rule).

    ``K = sqrt(1 + (beta**2 - 1) X**2)`` lies between ``min(1, beta)`` and ``max(1, beta)``. The
    binning error falls as the fourth power of the node spacing (a linear rule gives the
    square: four times as many nodes for half the error). The padding nodes stay positive
    (``n_nodes`` is raised if needed), which the refraction formulas require.

    Returns:
        ``(k_nodes, h)``, both of length ``n_nodes + 2`` (padded).
    """
    k = np.sqrt(1.0 + (beta * beta - 1.0) * x * x)
    k_lo, k_hi = min(1.0, beta), max(1.0, beta)
    n_nodes = max(n_nodes, math.ceil(2.0 * (k_hi - k_lo) / k_lo) + 2)  # keep k_lo - dk > 0
    dk = (k_hi - k_lo) / (n_nodes - 1)
    k_nodes = k_lo - dk + dk * np.arange(n_nodes + 2)
    return k_nodes, _cubic_deposit((k - k_lo) / dk + 1.0, weights, n_nodes)


def _bin_on_ln_k(
    beta: float, x: np.ndarray, weights: np.ndarray, du: float
) -> tuple[float, np.ndarray]:
    """Bin orientation weights onto a uniform grid in ``ln K`` with spacing ``du`` (cubic rule).

    Returns:
        ``(ln_k_first, h)``: ``h[j]`` belongs to ``ln K = ln_k_first + j du`` where the first
        node is the padding node, one cell below ``ln min(1, beta)``.
    """
    ln_k = 0.5 * np.log1p((beta * beta - 1.0) * x * x)
    ln_lo = math.log(min(1.0, beta))
    n_real = max(math.ceil(abs(math.log(beta)) / du) + 1, 4)
    h = _cubic_deposit((ln_k - ln_lo) / du + 1.0, weights, n_real)
    return ln_lo - du, h


# ---------------------------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------------------------


class Engine:
    """Evaluates :class:`~anisa.core.model.AnisaModel` objects, with caching.

    Keep one engine alive across evaluations (a GUI session or a fit) to benefit from the
    caches; results never depend on the cache contents.
    """

    def __init__(self, cache_size: int = 256):
        self._x = _LRU(cache_size)
        self._weights = _LRU(cache_size)
        self._unit = _LRU(cache_size)
        self._sizes = _LRU(cache_size)

    def clear_cache(self) -> None:
        for cache in (self._x, self._weights, self._unit, self._sizes):
            cache.clear()

    # -- cached building blocks -----------------------------------------------------------

    def _orientation(self, geom: tuple[float, float], pop: Population, n_grid: tuple[int, int]):
        """``(cos eta, weights)`` on the orientation grid, flattened."""
        grid = OrientationGrid.make(*n_grid)

        def make_x():
            # X = sin(aq) sin(a) cos(w - wq) + cos(aq) cos(a) (Allen, eq. 1) from outer products
            # of per-axis arrays: far cheaper than building 3-vectors on the full 2D grid.
            q_hat = geometry.q_unit_vector_deg(*geom)
            sin_aq = math.hypot(q_hat[0], q_hat[1])
            omega_q = math.atan2(q_hat[1], q_hat[0]) if sin_aq > 0 else 0.0
            sin_a = np.sqrt(1.0 - grid.cos_alpha**2)
            x = np.outer(sin_aq * sin_a, np.cos(grid.omega - omega_q))
            x += (q_hat[2] * grid.cos_alpha)[:, None]
            return np.clip(x, -1.0, 1.0).ravel()

        x = self._x.get((geom, n_grid), make_x)
        weights = self._weights.get(
            (pop.alpha_dist.key(), pop.omega_dist.key(), n_grid),
            lambda: pop.orientation.weights(grid).ravel(),
        )
        return x, weights

    # -- one (direction, population) task -------------------------------------------------

    def _unit_intensity(
        self, q: np.ndarray, geom: tuple[float, float], pop: Population, settings: CalcSettings
    ) -> np.ndarray:
        """Orientation- and size-averaged intensity of ``pop`` along ``geom`` for unit volume
        fraction, without interference [1/cm]."""
        key = (_array_key(q), geom, pop.shape_key(), settings.key())
        return self._unit.get(key, lambda: self._compute_unit_intensity(q, geom, pop, settings))

    def _compute_unit_intensity(
        self, q: np.ndarray, geom: tuple[float, float], pop: Population, settings: CalcSettings
    ) -> np.ndarray:
        out = np.empty(q.size)
        for index in q_bands(q):
            out[index] = self._band_intensity(q[index], geom, pop, settings)
        return out

    def _band_intensity(
        self, q: np.ndarray, geom: tuple[float, float], pop: Population, settings: CalcSettings
    ) -> np.ndarray:
        beta = float(pop.beta)
        spherical = abs(beta - 1.0) < 1e-9
        polydisperse = pop.size_dist.type != "none"
        if polydisperse and settings.formula == "4c" and not spherical:
            return self._band_log_space(q, geom, pop, settings)
        return self._band_classes(q, geom, pop, settings)

    def _band_classes(
        self, q: np.ndarray, geom: tuple[float, float], pop: Population, settings: CalcSettings
    ) -> np.ndarray:
        """Direct sum over size classes and K nodes (monodisperse, spheres, formulas 4a/4b)."""
        beta = float(pop.beta)
        delta_rho = pop.delta_rho * DELTA_RHO_TO_PER_A2
        q_max = float(q.max())
        k_hi = max(1.0, beta)
        r_lo, r_hi = radius_range(pop.size_dist, pop.radius)
        if pop.size_dist.type == "none":
            radii, share = np.array([float(pop.radius)]), np.array([1.0])
        else:
            n_r = math.ceil(NYQUIST_FACTOR * (r_hi - r_lo) * q_max * k_hi / math.pi)
            n_r = int(min(max(n_r, pop.size_dist.n_bins), MAX_RADIUS_CLASSES))
            radii, share = radius_classes(pop.size_dist, pop.radius, beta, n_r)

        if abs(beta - 1.0) < 1e-9:
            k_nodes, h = np.array([1.0]), np.array([1.0])  # K = 1 for every orientation
        else:
            span = q_max * r_hi * abs(beta - 1.0)
            n_grid = resolved_grid_size(settings, span, pop.orientation.required_nodes())
            x, weights = self._orientation(geom, pop, n_grid)
            k_nodes, h = _bin_on_k(beta, x, weights, k_node_count(settings, span))

        out = np.zeros(q.size)
        if settings.formula == "4c":
            # Closed form: the sum over size classes and K nodes is one contraction,
            #   I = 1e8 drho^2 sum_{r,k} v_r V_r h_k A^2(Q R_r K_k),
            # with no Python loop over the (up to a few hundred) radius classes.
            weight = PER_ANGSTROM_TO_PER_CM * delta_rho**2 * share * spheroid_volume(radii, beta)
            rows = max(1, _CHUNK_ELEMENTS // (radii.size * k_nodes.size))
            for start in range(0, q.size, rows):
                sl = slice(start, start + rows)
                x = q[sl, None, None] * radii[None, :, None] * k_nodes[None, None, :]
                out[sl] = np.einsum("qrk,r,k->q", sphere_amplitude_sq(x), weight, h)
            return out

        rows = max(1, _CHUNK_ELEMENTS // k_nodes.size)
        for radius, v in zip(radii, share, strict=True):
            prefactor = PER_ANGSTROM_TO_PER_CM * v / float(spheroid_volume(radius, beta))
            for start in range(0, q.size, rows):
                sl = slice(start, start + rows)
                f = cross_section(
                    settings.formula,
                    q[sl],
                    k_nodes,
                    float(radius),
                    beta,
                    delta_rho,
                    settings.wavelength,
                )
                out[sl] += prefactor * (f @ h)
        return out

    def _band_log_space(
        self, q: np.ndarray, geom: tuple[float, float], pop: Population, settings: CalcSettings
    ) -> np.ndarray:
        """Formula 4c with a size distribution by convolution in ``u = ln(R K)``."""
        beta = float(pop.beta)
        delta_rho = pop.delta_rho * DELTA_RHO_TO_PER_A2
        q_max = float(q.max())
        k_hi = max(1.0, beta)
        r_lo, r_hi = radius_range(pop.size_dist, pop.radius)
        span = q_max * r_hi * abs(beta - 1.0)
        n_grid = resolved_grid_size(settings, span, pop.orientation.required_nodes())
        x, weights = self._orientation(geom, pop, n_grid)

        # one common cell width in u for both densities; it resolves the largest phase x_max
        x_max = max(q_max * r_hi * k_hi, 1.0)
        du = 2.0 * math.pi / (settings.points_per_period * x_max)
        # ... and the size distribution itself needs at least n_bins cells (at low Q the
        # oscillation alone would allow a handful, too coarse for a peaked distribution)
        du = min(du, math.log(r_hi / r_lo) / pop.size_dist.n_bins)
        du = max(du, max(math.log(r_hi / r_lo), abs(math.log(beta))) / MAX_K_NODES)

        ln_k0, h = _bin_on_ln_k(beta, x, weights, du)
        ln_r0, volume_weight = self._size_density(pop, r_lo, r_hi, du)  # a_i = v_i V_i, grid ln_r0
        if volume_weight.size * h.size > _FFT_THRESHOLD:
            rho = fftconvolve(volume_weight, h)
        else:
            rho = np.convolve(volume_weight, h)
        u = ln_r0 + ln_k0 + du * np.arange(rho.size)
        e_u = np.exp(u)

        out = np.empty(q.size)
        rows = max(1, _CHUNK_ELEMENTS // rho.size)
        for start in range(0, q.size, rows):
            sl = slice(start, start + rows)
            out[sl] = sphere_amplitude_sq(np.outer(q[sl], e_u)) @ rho
        return PER_ANGSTROM_TO_PER_CM * delta_rho**2 * out

    def _size_density(
        self, pop: Population, r_lo: float, r_hi: float, du: float
    ) -> tuple[float, np.ndarray]:
        """Volume-weighted size density on a uniform grid in ``ln R``, as ``v_i V_i``.

        Returns ``(ln_r_first, a)`` with ``a[i]`` the (volume share x particle volume) of the
        cell centred on ``ln R = ln_r_first + i du``; the shares sum to one.
        """
        key = (pop.radius, pop.beta, pop.size_dist.key(), r_lo, r_hi, du)

        def make():
            n = max(math.ceil(math.log(r_hi / r_lo) / du), 4)
            ln_r = math.log(r_lo) + (np.arange(n) + 0.5) * du
            r = np.exp(ln_r)
            share = volume_density(pop.size_dist, pop.radius, pop.beta, r) * r  # per d ln R
            total = share.sum()
            if not (total > 0 and np.isfinite(total)):
                raise ValueError("size distribution has no weight on its radius grid")
            return ln_r[0], (share / total) * spheroid_volume(r, pop.beta)

        return self._sizes.get(key, make)

    # -- public ----------------------------------------------------------------------------

    def evaluate(self, model: AnisaModel, q: Sequence[np.ndarray] | None = None) -> ModelResult:
        """Calculate the model along every direction.

        Args:
            model: directions, populations and settings.
            q: Q grid [1/A] for each direction. If omitted, each direction's measured ``q`` is
                used.

        Returns:
            A :class:`ModelResult`. Directions are evaluated in the order of ``model.directions``.
        """
        settings = model.settings
        settings.validate()
        if q is None:
            q = [d.q for d in model.directions]
        if len(q) != len(model.directions):
            raise ValueError("need one q array per direction")
        q_arrays = []
        for i, qi in enumerate(q):
            if qi is None:
                raise ValueError(f"direction {i} has no q values and none were given")
            q_arrays.append(np.ascontiguousarray(qi, dtype=float))

        n_pop = len(model.populations)
        tasks = [(d, p) for d in range(len(model.directions)) for p in range(n_pop)]
        unit: dict[tuple[int, int], np.ndarray] = {}

        def run(task):
            d, p = task
            direction = model.directions[d]
            return self._unit_intensity(
                q_arrays[d], direction.geometry_key(), model.populations[p], settings
            )

        n_threads = settings.n_threads or os.cpu_count() or 1
        if n_threads > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=min(n_threads, len(tasks))) as pool:
                for task, result in zip(tasks, pool.map(run, tasks), strict=True):
                    unit[task] = result
        else:
            for task in tasks:
                unit[task] = run(task)

        directions = []
        for d, direction in enumerate(model.directions):
            qd = q_arrays[d]
            contributions = np.zeros((n_pop, qd.size))
            for p, pop in enumerate(model.populations):
                contribution = pop.volume_fraction * unit[(d, p)]
                if pop.use_interference:
                    contribution = contribution / (1.0 + pop.pack * sphere_amplitude(qd * pop.eta))
                contributions[p] = contribution
            directions.append(
                DirectionResult(
                    q=qd,
                    intensity=contributions.sum(axis=0) + direction.background,
                    populations=contributions,
                    background=float(direction.background),
                )
            )
        return ModelResult(
            directions=directions,
            surface_areas=np.array([derived.surface_area(p) for p in model.populations]),
            number_densities=np.array([derived.number_density(p) for p in model.populations]),
        )


def calculate(model: AnisaModel, q: Sequence[np.ndarray] | None = None) -> ModelResult:
    """One-shot evaluation with a throw-away engine (no caching across calls)."""
    return Engine().evaluate(model, q)
