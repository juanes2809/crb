"""Analytic, fully differentiable forward model of the hidden facet (``Camino A'').

This module implements a *closed-form, C-infinity smooth* forward model of the
active-corner-camera measurement rate ``g_q(psi)`` for a single (infinitesimal)
hidden facet parameterized by ``psi = (rho, phi, h)``, together with its exact
partial derivatives ``dg_q/dpsi_m`` obtained by **symbolic differentiation**
(sympy), and the Poisson Fisher information / Cramer-Rao bound assembled from
them.

It is the differentiable analogue of the rasterized simulator
(``pyerti.simulation`` / ``simulation_polar_clean.ipynb``) that
``crb_polar_functions.py`` differentiates by *finite differences*.  It does NOT
modify or call that simulator; it is a standalone re-derivation of the paper's
continuous model (Active Corner Camera CRB writeup, eqs. 1-3, 9-11) in which
every source of non-differentiability of the rasterizer is replaced by a smooth,
closed-form analogue:

============================  ==========================================================
Rasterizer (non-smooth)       Analytic smooth analogue (this module)
============================  ==========================================================
``arrival_bin = ceil(d/cDt)`` analytic integral of a Gaussian pulse over each time bin
   (Dirac-in-a-bin)              -> difference of ``erf`` (C-infinity in psi)
``noc = (xint > 0)``          soft occlusion ``sigmoid(kappa * d_edge(psi))``
   (sampled Heaviside)           with penumbra width ~ 1/kappa
``max(0, cos beta_k)``        softplus ``(1/beta) log(1 + exp(beta x))`` on each cosine
   (foreshortening clamps)       (smooth hinge, width ~ 1/beta)
deposit in integer pixel/bin  factorized product ``A_i(psi) * S_ij(psi)`` evaluated
                                 analytically on a fixed floor/time grid
============================  ==========================================================

The parameter Jacobian is therefore available in closed form for the Fisher
information ``I_{mk} = sum_q (1/g_q) dg_q/dpsi_m dg_q/dpsi_k`` and
``CRB = I^{-1}`` -- exactly the Poisson-Fisher structure of the paper (eqs. 9,
11), but with *exact* analytic derivatives instead of finite differences.

Public API
----------
``AnalyticForwardModel``   builds & caches the lambdified symbolic model.
``analytic_forward_g``     evaluate g_q(psi) over the (pixel, bin) grid.
``analytic_jacobian_g``    evaluate the exact Jacobian dg_q/dpsi_m.
``compute_crb_analytic``   assemble I, CRB and the sigma_* / ellipse statistics
                            (mirrors ``crb_polar_functions.compute_crb_polar``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

try:  # reuse the (unmodified) polar helpers for sigma extraction / ellipses
    from crb_polar_functions import (
        _crb_sigmas_from_matrix,
        crb_region_in_polar_parameters,
        crb_region_physical_xy_to_polar,
    )

    _HAVE_POLAR_HELPERS = True
except Exception:  # pragma: no cover - keep module importable standalone
    _HAVE_POLAR_HELPERS = False


ArrayLike = np.ndarray | Sequence[float]

# Speed of light (m/s), matches pyerti.py / the simulator.
C_LIGHT = 299792458.0


# ---------------------------------------------------------------------------
# Geometry / smoothing configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ForwardConfig:
    """Fixed geometry + smoothing parameters of the analytic forward.

    All fields are *fixed* w.r.t. the estimated parameters ``psi=(rho,phi,h)``;
    only the facet pose is differentiated.  The defaults follow the simulator's
    conventions (laser at the origin pointing +z, floor FOV ~0.5 m adjacent to
    the occluding edge, bin size 390 ps).

    Smoothing parameters (documented in docs/analytic_forward_crb.tex):

    * ``beta``   -- softplus sharpness for the foreshortening cosines; the
      smoothing width of the hinge ``max(0, x)`` is ``~1/beta``.
    * ``kappa``  -- logistic sharpness of the soft occlusion; the penumbra half
      width (in metres, along the edge-crossing coordinate) is ``~1/kappa``.
    * ``tau``    -- Gaussian pulse standard deviation (seconds).  The pulse is
      ``s(t)=exp(-(t-t0)^2/(2 tau^2))`` and its analytic bin integral is a
      difference of ``erf``.
    """

    # --- fixed scene geometry ------------------------------------------------
    p_l: Tuple[float, float, float] = (0.0, 0.0, 0.0)      # laser spot
    n_l: Tuple[float, float, float] = (0.0, 0.0, 1.0)      # laser surface normal
    p_c: Tuple[float, float, float] = (0.0, -0.25, 1.5)    # SPAD camera position
    n_f: Tuple[float, float, float] = (0.0, 0.0, 1.0)      # floor normal

    # --- floor FOV pixel grid ------------------------------------------------
    fov_width: float = 0.5          # metres (square FOV side)
    fov_center_x: float = 0.0
    fov_center_y: float = -0.25     # camera_FOV_center = [0, -FOV/2, 0]
    pixel_dim: int = 8              # pixel_dim x pixel_dim floor samples

    # --- temporal binning ----------------------------------------------------
    bin_size: float = 3.9e-10       # seconds (Delta t)
    n_time_bins: Optional[int] = None   # auto if None
    time_margin_bins: int = 6       # extra bins padding around the t0 window

    # --- radiometry ----------------------------------------------------------
    alpha: float = 1.0              # albedo
    gain: float = 1.0e5             # overall photon gain (laser power * scale)
    background: float = 0.1         # b_q  (keeps 1/g finite without eps floor)

    # --- smoothing widths ----------------------------------------------------
    beta: float = 50.0              # softplus sharpness (cosine hinge width ~1/beta)
    kappa: float = 60.0             # occlusion sharpness (penumbra width ~1/kappa m)
    tau: float = 3.9e-10            # Gaussian pulse std (s)

    def as_key(self) -> Tuple:
        """Hashable key of the *symbolic-structure-relevant* fields."""
        return (
            self.p_l, self.n_l, self.p_c, self.n_f,
            self.alpha, self.gain, self.beta, self.kappa, self.tau, C_LIGHT,
        )


# ---------------------------------------------------------------------------
# Symbolic model (built once, lambdified, cached by structural config key)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _build_symbolic(key: Tuple) -> Dict[str, Callable]:
    """Construct and lambdify y_q(psi) and its psi-derivatives.

    ``key`` is ``ForwardConfig.as_key()`` -- everything that changes the closed
    form.  Grid coordinates ``(px, py, tlo, thi)`` remain free arguments so the
    same lambdified callables serve any floor/time grid.
    """
    import sympy as sp

    (p_l, n_l, p_c, n_f, alpha, gain, beta, kappa, tau, c) = key

    rho, phi, h = sp.symbols("rho phi h", real=True)
    px, py, tlo, thi = sp.symbols("px py tlo thi", real=True)

    # --- facet pose (paper eq. sec. II) -------------------------------------
    sx = rho * sp.cos(phi)
    sy = rho * sp.sin(phi)
    ps = sp.Matrix([sx, sy, h])
    ns = sp.Matrix([-sp.cos(phi), -sp.sin(phi), 0])

    pf = sp.Matrix([px, py, 0])
    pl = sp.Matrix(p_l)
    pc = sp.Matrix(p_c)
    nl = sp.Matrix(n_l)
    nf = sp.Matrix(n_f)

    # --- distances (smooth in psi) ------------------------------------------
    d_pl = sp.sqrt((ps - pl).dot(ps - pl))   # ||p_s - p_l||
    d_pf = sp.sqrt((ps - pf).dot(ps - pf))   # ||p_s - p_f||
    d_cf = sp.sqrt((pc - pf).dot(pc - pf))   # ||p_c - p_f||  (const in psi)

    # --- foreshortening: 5 cosines with softplus hinges (eq. 2) -------------
    def softplus(x):
        return sp.log(1 + sp.exp(beta * x)) / beta

    cos1 = nf.dot(pc - pf) / d_cf            # camera <-> floor
    cos2 = nf.dot(ps - pf) / d_pf            # floor normal <-> facet
    cos3 = ns.dot(pf - ps) / d_pf            # facet normal <-> floor
    cos4 = ns.dot(pl - ps) / d_pl            # facet normal <-> laser
    cos5 = nl.dot(ps - pl) / d_pl            # laser normal <-> facet
    f_fore = (
        softplus(cos1) * softplus(cos2) * softplus(cos3)
        * softplus(cos4) * softplus(cos5)
    )

    # --- soft occlusion (replaces Heaviside / the ``xint>0`` mask) ----------
    # d_edge = x-coordinate where the facet->floor segment crosses y=0.  This is
    # exactly the rasterizer's ``xint`` but kept symbolic and smooth.  visible
    # when the crossing is on the lit side (d_edge > 0).
    d_edge = sx - sy * (sx - px) / (sy - py)
    vis = 1 / (1 + sp.exp(-kappa * d_edge))

    # --- radial falloff (eq. 1 denominator) ---------------------------------
    denom = 8 * sp.pi**3 * d_cf**2 * d_pl**2 * d_pf**2
    A = alpha * gain * vis * f_fore / denom  # spatial amplitude of pixel

    # --- analytic pulse integral over the bin [tlo, thi] --------------------
    # The rasterizer deposits the WHOLE pixel intensity into a single arrival
    # bin (a Dirac mass).  Its smooth analogue is a *unit-area* Gaussian pulse
    #     s(t) = 1/(tau sqrt(2 pi)) exp(-(t - t0)^2 / (2 tau^2)),
    # with time-of-flight t0 = (d_pl + d_pf)/c.  Integrating the pdf over the bin
    # gives a scaled difference of erf, so that summing over all bins returns the
    # full spatial amplitude A (the Dirac mass) -- but now C-infinity in psi:
    #     S = 1/2 [ erf((thi-t0)/(sqrt2 tau)) - erf((tlo-t0)/(sqrt2 tau)) ].
    t0 = (d_pl + d_pf) / c
    S = (
        sp.erf((thi - t0) / (sp.sqrt(2) * tau))
        - sp.erf((tlo - t0) / (sp.sqrt(2) * tau))
    ) / 2

    y = A * S  # y_q(psi) for a single (pixel, bin) -- background added later

    args = (rho, phi, h, px, py, tlo, thi)
    modules = ["scipy", "numpy"]
    out: Dict[str, Callable] = {
        "y": sp.lambdify(args, y, modules=modules),
        "dy_drho": sp.lambdify(args, sp.diff(y, rho), modules=modules),
        "dy_dphi": sp.lambdify(args, sp.diff(y, phi), modules=modules),
        "dy_dh": sp.lambdify(args, sp.diff(y, h), modules=modules),
        "t0": sp.lambdify((rho, phi, h, px, py), t0, modules=modules),
    }
    return out


class AnalyticForwardModel:
    """Cached, lambdified analytic forward + exact Jacobian over a fixed grid."""

    def __init__(self, config: Optional[ForwardConfig] = None):
        self.config = config or ForwardConfig()
        self._fns = _build_symbolic(self.config.as_key())
        self._px, self._py = self._build_pixel_grid()

    # --- grids --------------------------------------------------------------
    def _build_pixel_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        n = int(cfg.pixel_dim)
        half = cfg.fov_width / 2.0
        # pixel centres (like the simulator's linspace of subpixel centres)
        xs = np.linspace(
            cfg.fov_center_x - half + cfg.fov_width / (2 * n),
            cfg.fov_center_x + half - cfg.fov_width / (2 * n),
            n,
        )
        ys = np.linspace(
            cfg.fov_center_y - half + cfg.fov_width / (2 * n),
            cfg.fov_center_y + half - cfg.fov_width / (2 * n),
            n,
        )
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        return X.ravel(), Y.ravel()

    def _time_bins(self, psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (tlo, thi) bin edges covering the pulse for this pose."""
        cfg = self.config
        rho, phi, h = float(psi[0]), float(psi[1]), float(psi[2])
        t0 = self._fns["t0"](rho, phi, h, self._px, self._py)
        t0 = np.asarray(t0, dtype=float)
        dt = cfg.bin_size
        if cfg.n_time_bins is not None:
            j = np.arange(cfg.n_time_bins)
            return j * dt, (j + 1) * dt
        # auto: cover [min t0 - margin, max t0 + margin] with a few tau padding
        pad = cfg.time_margin_bins * dt + 4.0 * cfg.tau
        j_lo = int(np.floor((t0.min() - pad) / dt))
        j_hi = int(np.ceil((t0.max() + pad) / dt))
        j_lo = max(j_lo, 0)
        j = np.arange(j_lo, j_hi + 1)
        return j * dt, (j + 1) * dt

    # --- evaluation ---------------------------------------------------------
    def freeze_bins(self, psi: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Return a (tlo, thi) bin grid for ``psi`` to reuse across evaluations.

        The auto bin window depends on ``psi`` (through the time-of-flight), so a
        fair analytic-vs-finite-difference comparison must hold the bin grid
        FIXED while perturbing ``psi``.  Pass the returned tuple as ``bins=``.
        """
        psi = self._pad_psi(np.asarray(psi, dtype=float).ravel())
        return self._time_bins(psi)

    def _grid_qr(self, psi: np.ndarray, bins=None):
        """Broadcast (pixel, bin) grid arrays for lambdified evaluation."""
        rho, phi, h = float(psi[0]), float(psi[1]), float(psi[2])
        if bins is None:
            tlo, thi = self._time_bins(psi)
        else:
            tlo, thi = bins
        # meshgrid over pixels (P) x bins (B)
        PX, TLO = np.meshgrid(self._px, tlo, indexing="ij")
        PY, THI = np.meshgrid(self._py, thi, indexing="ij")
        return rho, phi, h, PX, PY, TLO, THI

    def forward_y(self, psi: ArrayLike, bins=None) -> np.ndarray:
        psi = np.asarray(psi, dtype=float).ravel()
        psi = self._pad_psi(psi)
        rho, phi, h, PX, PY, TLO, THI = self._grid_qr(psi, bins=bins)
        y = self._fns["y"](rho, phi, h, PX, PY, TLO, THI)
        return np.asarray(y, dtype=float).ravel()

    def forward_g(self, psi: ArrayLike, bins=None) -> np.ndarray:
        return self.forward_y(psi, bins=bins) + self.config.background

    def jacobian_g(self, psi: ArrayLike, n_params: int = 3, bins=None) -> np.ndarray:
        """Exact analytic Jacobian dg/dpsi (== dy/dpsi) of shape (N_q, n_params)."""
        psi = self._pad_psi(np.asarray(psi, dtype=float).ravel())
        rho, phi, h, PX, PY, TLO, THI = self._grid_qr(psi, bins=bins)
        cols = []
        cols.append(self._fns["dy_drho"](rho, phi, h, PX, PY, TLO, THI))
        cols.append(self._fns["dy_dphi"](rho, phi, h, PX, PY, TLO, THI))
        if n_params >= 3:
            cols.append(self._fns["dy_dh"](rho, phi, h, PX, PY, TLO, THI))
        J = np.stack(
            [np.asarray(cwn, dtype=float).ravel() for cwn in cols], axis=1
        )
        return J

    @staticmethod
    def _pad_psi(psi: np.ndarray) -> np.ndarray:
        if psi.size == 2:
            return np.array([psi[0], psi[1], 1.0], dtype=float)
        if psi.size == 3:
            return psi.astype(float)
        raise ValueError(f"psi must be length 2 or 3, got {psi.shape}")


# ---------------------------------------------------------------------------
# Convenience functional API
# ---------------------------------------------------------------------------
def analytic_forward_g(
    psi: ArrayLike, config: Optional[ForwardConfig] = None
) -> np.ndarray:
    """Evaluate g_q(psi) = y_q(psi) + b over the (pixel, bin) grid."""
    return AnalyticForwardModel(config).forward_g(psi)


def analytic_jacobian_g(
    psi: ArrayLike, config: Optional[ForwardConfig] = None, n_params: int = 3
) -> np.ndarray:
    """Evaluate the exact analytic Jacobian dg_q/dpsi_m (closed form, sympy)."""
    return AnalyticForwardModel(config).jacobian_g(psi, n_params=n_params)


def fisher_poisson_analytic(g: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Poisson Fisher I = J^T diag(1/g) J (eq. 9).

    No eps floor is needed: b>0 keeps g bounded away from 0 (paper convention).
    """
    g = np.asarray(g, dtype=float).ravel()
    J = np.asarray(J, dtype=float)
    if J.shape[0] != g.size:
        raise ValueError(f"J rows {J.shape[0]} != g size {g.size}")
    return J.T @ ((1.0 / g)[:, None] * J)


def compute_crb_analytic(
    psi: ArrayLike,
    config: Optional[ForwardConfig] = None,
    estimate_height: bool = True,
    k: float = 3.0,
    use_pinv: bool = True,
) -> Dict[str, Any]:
    """Assemble the Poisson Fisher information and Cramer-Rao bound analytically.

    Mirrors the result structure of ``crb_polar_functions.compute_crb_polar``
    (sigma_rho, sigma_phi, sigma_phi_deg, sigma_tangential, sigma_height,
    CRB / I / J, and 3-sigma ellipse curves in (rho, phi)) but every quantity is
    obtained from the *closed-form* analytic Jacobian.
    """
    psi = np.asarray(psi, dtype=float).ravel()
    if psi.size == 2:
        rho0, phi0, h0 = float(psi[0]), float(psi[1]), 1.0
    elif psi.size == 3:
        rho0, phi0, h0 = float(psi[0]), float(psi[1]), float(psi[2])
    else:
        raise ValueError(f"psi must be length 2 or 3, got {psi.shape}")

    model = AnalyticForwardModel(config)
    n_params = 3 if estimate_height else 2
    psi_full = np.array([rho0, phi0, h0], dtype=float)

    g = model.forward_g(psi_full)
    J = model.jacobian_g(psi_full, n_params=n_params)
    I = fisher_poisson_analytic(g, J)
    CRB = np.linalg.pinv(I) if use_pinv else np.linalg.inv(I)

    if _HAVE_POLAR_HELPERS:
        stats = _crb_sigmas_from_matrix(CRB, rho0=rho0)
    else:  # pragma: no cover
        stats = {
            "sigma_rho": float(np.sqrt(max(CRB[0, 0], 0.0))),
            "sigma_phi": float(np.sqrt(max(CRB[1, 1], 0.0))),
            "sigma_phi_deg": float(np.rad2deg(np.sqrt(max(CRB[1, 1], 0.0)))),
            "sigma_tangential": float(rho0 * np.sqrt(max(CRB[1, 1], 0.0))),
            "sigma_height": (
                float(np.sqrt(max(CRB[2, 2], 0.0))) if CRB.shape[0] >= 3 else float("nan")
            ),
        }

    corr = (
        float(CRB[0, 1] / np.sqrt(CRB[0, 0] * CRB[1, 1]))
        if CRB[0, 0] > 0 and CRB[1, 1] > 0
        else np.nan
    )

    result: Dict[str, Any] = {
        "rho0": rho0,
        "phi0": phi0,
        "phi0_deg": float(np.rad2deg(phi0)),
        "height0": h0,
        "estimate_height": bool(n_params == 3),
        "g0_vec": g,
        "J": J,
        "I": I,
        "CRB": CRB,
        "sigma_rho": stats["sigma_rho"],
        "sigma_phi": stats["sigma_phi"],
        "sigma_phi_deg": stats["sigma_phi_deg"],
        "sigma_tangential": stats["sigma_tangential"],
        "sigma_height": stats["sigma_height"],
        "corr_rho_phi": corr,
        "n_measurements": int(g.size),
        "n_params": n_params,
    }

    if _HAVE_POLAR_HELPERS:
        rho_p, phi_p = crb_region_in_polar_parameters(rho0, phi0, CRB, k=k)
        rho_phys, phi_phys, x_c, y_c = crb_region_physical_xy_to_polar(
            rho0, phi0, CRB, k=k
        )
        result.update(
            {
                "rho_curve_param": rho_p,
                "phi_curve_param": phi_p,
                "rho_curve_phys": rho_phys,
                "phi_curve_phys": phi_phys,
                "x_curve_phys": x_c,
                "y_curve_phys": y_c,
            }
        )
    return result


if __name__ == "__main__":  # small smoke test
    cfg = ForwardConfig()
    m = AnalyticForwardModel(cfg)
    psi0 = np.array([1.0, np.deg2rad(90.0), 1.0])
    g = m.forward_g(psi0)
    J = m.jacobian_g(psi0)
    print(f"grid: {m.config.pixel_dim}x{m.config.pixel_dim} pixels, "
          f"N_q={g.size}, J={J.shape}")
    print(f"g range: [{g.min():.4g}, {g.max():.4g}]  (b={cfg.background})")
    res = compute_crb_analytic(psi0)
    print(f"sigma_rho={res['sigma_rho']:.4g} m, "
          f"sigma_phi={res['sigma_phi_deg']:.4g} deg, "
          f"sigma_h={res['sigma_height']:.4g} m")
