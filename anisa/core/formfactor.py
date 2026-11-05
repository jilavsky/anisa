"""
Single-particle cross sections of an oblate or prolate spheroid (Allen, 2001, eqs. 4a-4c).

The particle is a spheroid with semi-axes (R, R, beta R); ``beta`` < 1 is oblate ("doughnut
like") and ``beta`` > 1 is prolate ("cigar like"). The only orientation dependence is
``X = cos(eta)``, the cosine of the angle between the symmetry axis and Q, entering through::

    K(beta, X) = sqrt(1 + (beta**2 - 1) X**2)

Three formulas are available, selected by name, in order of cost:

``"4c"``  diffraction limit (phase shift nu_0 -> 0), closed form::

              dsigma/dOmega = V**2 drho**2 [3 j1(x) / x]**2,    x = Q R K

          Igor wrote this as ``(9 pi / 2) [J_{3/2}(x) / x**(3/2)]**2``; the two are identical.
          Its wavelength dependence cancels, so ``wavelength`` is not needed.
``"4a"``  full expression including refraction, evaluated by numerical integration::

              dsigma/dOmega = |f|**2,
              f = i k R**2 K  Int_0^1 J_0(Q R K xi) [1 - exp(i beta nu_0 sqrt(1 - xi**2) / K)] xi dxi

          with ``nu_0 = 2 R |drho| lambda`` and ``k = 2 pi / lambda``. Substituting
          ``xi = sin(t)`` removes the end-point square-root singularity, so Gauss-Legendre
          converges geometrically.
``"4b"``  the same expression as a series (Sonine's finite integral applied term by term)::

              f = -i k R**2 K  sum_{m>=1} (i c)**m / m!  2**(m/2) Gamma(1 + m/2) J_{1+m/2}(x) / x**(1+m/2),
              c = beta nu_0 / K

          Allen's printed series and the Igor code (which could not run: its Gamma table was
          never created) both lack the ``1/m!``. Without it the series only converges for
          ``c < 1``; with it the series equals 4a, which ``tests/test_formfactor.py`` verifies.
          For large ``c`` the terms cancel catastrophically, so entries with ``c > C_SERIES_MAX``
          are evaluated with 4a instead.

Units: ``q`` in 1/Angstrom, ``R`` in Angstrom, ``delta_rho`` in 1/Angstrom**2 (Igor and the GUI
use cm**-2; the conversion is ``* 1e-16`` and is done by the caller). Cross sections are in
Angstrom**2 per particle.
"""

from __future__ import annotations

import numpy as np
from scipy import special

FORMULAS = ("4c", "4b", "4a")

#: Series (4b) is only trusted below this value of ``c = beta nu_0 / K``.
C_SERIES_MAX = 10.0

#: Largest number of (q, K, theta) elements held in memory at once by formula 4a.
_CHUNK_ELEMENTS = 4_000_000


def spheroid_k(beta: float, x) -> np.ndarray:
    """``K(beta, X) = sqrt(1 + (beta**2 - 1) X**2)`` (Allen, below eq. 4a)."""
    x = np.asarray(x, dtype=float)
    return np.sqrt(1.0 + (beta * beta - 1.0) * x * x)


def spheroid_volume(radius, beta):
    """Volume ``4 pi beta R**3 / 3`` of a spheroid with semi-axes (R, R, beta R)."""
    return (4.0 / 3.0) * np.pi * beta * np.asarray(radius, dtype=float) ** 3


def sphere_amplitude(x) -> np.ndarray:
    """Normalised sphere amplitude ``3 j1(x) / x = 3 (sin x - x cos x) / x**3``; 1 at x = 0."""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-6
    xs = x[nz]
    # 3 j1(x)/x is accurate for small x too (scipy uses a series there), unlike the sin/cos form.
    out[nz] = 3.0 * special.spherical_jn(1, xs) / xs
    # Below 1e-6 the Taylor series 1 - x**2/10 is exact to double precision.
    small = ~nz
    out[small] = 1.0 - x[small] ** 2 / 10.0
    return out


def sphere_amplitude_sq(x) -> np.ndarray:
    """``[3 j1(x) / x]**2``: the normalised sphere form factor."""
    amp = sphere_amplitude(x)
    return amp * amp


def cross_section_4c(q, k_nodes, radius: float, beta: float, delta_rho: float) -> np.ndarray:
    """Formula 4c on a ``(q, K)`` grid.

    Args:
        q: Q values [1/A], shape ``(nq,)``.
        k_nodes: K(beta, X) values, shape ``(nk,)``.
        radius: equatorial radius R [A].
        beta: aspect ratio.
        delta_rho: scattering contrast [1/A**2].

    Returns:
        Cross sections [A**2], shape ``(nq, nk)``.
    """
    q = np.asarray(q, dtype=float)
    k_nodes = np.asarray(k_nodes, dtype=float)
    volume = spheroid_volume(radius, beta)
    return (volume * delta_rho) ** 2 * sphere_amplitude_sq(radius * np.outer(q, k_nodes))


def _theta_nodes(n_theta: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n_theta)
    return 0.25 * np.pi * (x + 1.0), 0.25 * np.pi * w  # [-1, 1] -> [0, pi/2]


def _n_theta(a_max: float, c_max: float) -> int:
    """Gauss-Legendre order for formula 4a.

    The integrand is entire with exponential type ``a + c`` in ``t``, so about ``(a + c) / 2``
    nodes plus a margin reach machine precision (checked in the tests).
    """
    return int(min(max(0.5 * (a_max + c_max) + 24, 32), 6000))


def cross_section_4a(
    q, k_nodes, radius: float, beta: float, delta_rho: float, wavelength: float
) -> np.ndarray:
    """Formula 4a on a ``(q, K)`` grid, including refraction. See the module docstring."""
    q = np.asarray(q, dtype=float)
    k_nodes = np.asarray(k_nodes, dtype=float)
    k_wave = 2.0 * np.pi / wavelength
    nu0 = 2.0 * radius * abs(delta_rho) * wavelength
    c = beta * nu0 / k_nodes  # (nk,)
    a_max = float(radius * q.max() * k_nodes.max()) if q.size else 0.0
    n_theta = _n_theta(a_max, float(c.max()))
    theta, weight = _theta_nodes(n_theta)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    # exp(i c cos t) depends only on (K, theta); the J0 factor on (q, K, theta).
    refraction = (1.0 - np.exp(1j * c[:, None] * cos_t[None, :])) * (weight * sin_t * cos_t)
    nk = k_nodes.size
    out = np.empty((q.size, nk))
    rows = max(1, _CHUNK_ELEMENTS // max(1, nk * n_theta))
    for start in range(0, q.size, rows):
        qs = q[start : start + rows]
        a = radius * np.outer(qs, k_nodes)  # (rows, nk)
        bessel = special.j0(a[:, :, None] * sin_t[None, None, :])
        integral = np.einsum("qkt,kt->qk", bessel, refraction)
        f = 1j * k_wave * radius**2 * k_nodes[None, :] * integral
        out[start : start + rows] = np.abs(f) ** 2
    return out


def cross_section_4b(
    q,
    k_nodes,
    radius: float,
    beta: float,
    delta_rho: float,
    wavelength: float,
    *,
    rtol: float = 1e-12,
    max_terms: int = 400,
) -> np.ndarray:
    """Formula 4b (series, with the ``1/m!`` the printed formula lacks) on a ``(q, K)`` grid.

    Entries with ``c = beta nu_0 / K > C_SERIES_MAX`` fall back to formula 4a.
    """
    q = np.asarray(q, dtype=float)
    k_nodes = np.asarray(k_nodes, dtype=float)
    k_wave = 2.0 * np.pi / wavelength
    nu0 = 2.0 * radius * abs(delta_rho) * wavelength
    c = beta * nu0 / k_nodes  # (nk,)
    use_series = c <= C_SERIES_MAX
    out = np.empty((q.size, k_nodes.size))
    if np.any(use_series):
        ks = k_nodes[use_series]
        a = np.maximum(radius * np.outer(q, ks), 1e-12)  # (nq, nks)
        ic = 1j * c[use_series][None, :]
        total = np.zeros(a.shape, dtype=complex)
        prefactor = np.ones(a.shape, dtype=complex)  # (i c)^m / m!
        for m in range(1, max_terms + 1):
            prefactor = prefactor * ic / m
            order = 1.0 + 0.5 * m
            # 2**(m/2) Gamma(1 + m/2) J_{1+m/2}(a) / a**(1+m/2); the ratio J_v(a)/a**v is
            # evaluated with log-gamma-free scipy calls, which stay finite for small a.
            term = prefactor * (
                2.0 ** (0.5 * m) * special.gamma(order) * special.jv(order, a) / a**order
            )
            total += term
            if m >= 4 and np.max(np.abs(term)) <= rtol * max(np.max(np.abs(total)), 1e-300):
                break
        f = -1j * k_wave * radius**2 * ks[None, :] * total
        out[:, use_series] = np.abs(f) ** 2
    if not np.all(use_series):
        out[:, ~use_series] = cross_section_4a(
            q, k_nodes[~use_series], radius, beta, delta_rho, wavelength
        )
    return out


def cross_section(
    formula: str,
    q,
    k_nodes,
    radius: float,
    beta: float,
    delta_rho: float,
    wavelength: float = 1.0,
) -> np.ndarray:
    """Dispatch to formula ``"4a"``, ``"4b"`` or ``"4c"``. Shapes and units as above."""
    if formula == "4c":
        return cross_section_4c(q, k_nodes, radius, beta, delta_rho)
    if formula == "4a":
        return cross_section_4a(q, k_nodes, radius, beta, delta_rho, wavelength)
    if formula == "4b":
        return cross_section_4b(q, k_nodes, radius, beta, delta_rho, wavelength)
    raise ValueError(f"unknown formula {formula!r}; expected one of {FORMULAS}")
