"""Analytic, fully differentiable forward model of the hidden facet (``Camino A'').

Variant **(a)+(b)+(c)** -- the *most faithful* differentiable analogue of the
rasterized simulator:

* **(a) physical amplitude.**  The per-triangle amplitude is the smooth analogue
  of the simulator's OWN ``intensity`` (``simulation_polar_clean.ipynb`` cell 8):
  ``laser_intensity * area_tri * softplus(cos1..cos4) * vis / (4 pi^2 d1^2 d2^2)``
  with a *unit-area* Gaussian pulse (``erf`` bin integral) and background ``b``.
  No arbitrary gain: the numbers come out in the simulator's physical units.
* **(b) simulator grid.**  The floor FOV / pixel / time-bin grid matches the
  simulator exactly: ``camera_FOV=0.25`` m, ``cam_pixel_dim=64``,
  ``camera_FOV_center=[0,-0.125,0]``, ``bin_size=3.9e-10`` s, ``c=299792458``.
* **(c) sum over the REAL mesh triangles.**  Instead of a single point facet we
  read ``objects/facet.obj`` and place its *actual* triangles with the SAME
  symbolic transform the simulator applies (cell 8): anisotropic scale
  ``[w/ext_x, h/ext_y, 1]``, pitch ``~1.57`` about ``[1,0,0]``, roll about
  ``[0,1,0]``, yaw ``theta = phi + 3*pi/2`` about ``[0,0,1]``, lift ``z_min->0``
  then translate to ``v1 = (rho cos phi, rho sin phi, 0)``.  The raw vertices are
  treated as constants and ``(rho, phi, h)`` as sympy symbols, so for every
  triangle the centroid ``p_s(rho,phi,h)``, area ``area_tri(rho,phi,h)`` and
  normal ``n_s(rho,phi,h)`` are closed-form.  The smooth contribution is built
  per ``(triangle, pixel, bin)`` and SUMMED over triangles.

The result is a ``C``-infinity forward whose *shape and magnitude* track the
simulator's mesh forward far more closely than the point-facet variants, so the
analytic CRB and the finite-difference (rasterized) CRB become comparable in
magnitude (they converge to the same hard limit as the smoothing widths shrink).

Every source of non-differentiability of the simulator is replaced by a smooth,
closed-form analogue:

============================  ==========================================================
Simulator (non-smooth)        Analytic smooth analogue (this module)
============================  ==========================================================
``arrival_bin = ceil(d/cDt)`` analytic integral of a unit-area Gaussian pulse over each
   (Dirac-in-a-bin)              time bin -> difference of ``erf`` (C-infinity in psi)
``noc = (xint > 0)``          soft occlusion ``sigmoid(kappa * d_edge(psi))``,
   (sampled Heaviside)           d_edge == the simulator's xint, penumbra width ~1/kappa
``max(0, dot_k)`` (4 cosines) softplus ``(1/beta) log(1 + exp(beta x))`` on each of the
   (foreshortening clamps)       FOUR two-bounce cosines (smooth hinge, width ~1/beta)
point facet (single p_s)      SUM over the real ``facet.obj`` triangles with symbolic
                                 centroid / area / normal per triangle (variant (c))
============================  ==========================================================

Public API (unchanged)
----------------------
``AnalyticForwardModel``   builds & caches the lambdified symbolic model.
``analytic_forward_g``     evaluate g_q(psi) over the (pixel, bin) grid.
``analytic_jacobian_g``    evaluate the exact Jacobian dg_q/dpsi_m.
``compute_crb_analytic``   assemble I, CRB and the sigma_* / ellipse statistics
                            (mirrors ``crb_polar_functions.compute_crb_polar``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Geometry / smoothing configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ForwardConfig:
    """Fixed geometry + smoothing parameters of the analytic forward.

    All fields are *fixed* w.r.t. the estimated parameters ``psi=(rho,phi,h)``;
    only the facet pose is differentiated.  The defaults follow the simulator's
    OWN parameters (``simulation_polar_clean.ipynb`` cells 5/8): laser at the
    origin pointing +z, floor FOV 0.25 m centred at ``[0,-0.125,0]``, a 64x64
    pixel grid, ``bin_size=390`` ps, ``laser_intensity=1000`` and the facet
    ``facet.obj`` scaled to width ``w=0.5`` and height ``h`` (the parameter).

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
    p_l: Tuple[float, float, float] = (0.0, 0.0, 0.0)      # laser spot (laser_pos)
    n_l: Tuple[float, float, float] = (0.0, 0.0, 1.0)      # laser surface normal
    n_f: Tuple[float, float, float] = (0.0, 0.0, 1.0)      # floor normal
    # p_c is UNUSED by the two-bounce forward (kept only for backward-compat with
    # callers that still pass it; it plays no role in the intensity or Jacobian).
    p_c: Tuple[float, float, float] = (0.0, -0.25, 1.5)    # SPAD camera position (unused)

    # --- floor FOV pixel grid (matches the simulator) ------------------------
    fov_width: float = 0.25         # metres (camera_FOV)
    fov_center_x: float = 0.0
    fov_center_y: float = -0.125    # camera_FOV_center = [0, -camera_FOV/2, 0]
    pixel_dim: int = 64             # cam_pixel_dim

    # --- temporal binning ----------------------------------------------------
    bin_size: float = 3.9e-10       # seconds (Delta t)
    n_time_bins: Optional[int] = None   # auto if None
    time_margin_bins: int = 6       # extra bins padding around the t0 window

    # --- facet mesh + placement (variant (c)) --------------------------------
    obj_path: str = "objects/facet.obj"  # relative to this module / cwd
    facet_w: float = 0.5            # facet width (fixed): X -> w before pitch
    pitch: float = 1.57             # rotation about [1,0,0] (rad)
    roll: float = 0.0               # rotation about [0,1,0] (rad)

    # --- radiometry ----------------------------------------------------------
    laser_intensity: float = 1000.0  # simulator laser_intensity
    alpha: float = 1.0              # albedo (extra multiplicative knob, default 1)
    gain: float = 1.0               # extra photon gain (default 1 -> physical units)
    background: float = 0.1         # b_q  (keeps 1/g finite without eps floor)

    # --- smoothing widths ----------------------------------------------------
    beta: float = 50.0              # softplus sharpness (cosine hinge width ~1/beta)
    kappa: float = 60.0             # occlusion sharpness (penumbra width ~1/kappa m)
    tau: float = 3.9e-10            # Gaussian pulse std (s)

    def resolved_obj_path(self) -> str:
        """Absolute path to the mesh, trying cwd then the module directory."""
        if os.path.isabs(self.obj_path) and os.path.exists(self.obj_path):
            return self.obj_path
        if os.path.exists(self.obj_path):
            return os.path.abspath(self.obj_path)
        cand = os.path.join(_THIS_DIR, self.obj_path)
        if os.path.exists(cand):
            return cand
        return self.obj_path

    def as_key(self) -> Tuple:
        """Hashable key of the *symbolic-structure-relevant* fields.

        ``p_c`` and the pixel/time grids are intentionally absent: the two-bounce
        closed form does not depend on the camera position and the grid enters
        only as free lambdify arguments, so changing them must not rebuild the
        symbolic model.
        """
        return (
            self.p_l, self.n_l, self.n_f,
            self.obj_path, self.facet_w, self.pitch, self.roll,
            self.laser_intensity, self.alpha, self.gain,
            self.beta, self.kappa, self.tau, C_LIGHT,
        )


# ---------------------------------------------------------------------------
# Mesh loading (raw triangles of facet.obj)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _load_raw_mesh(obj_path: str) -> Dict[str, Any]:
    """Load the raw facet mesh: triangles (T,3,3), extents, min vertex, z-min ref.

    Returns everything the symbolic placement needs.  The mesh is NOT densified
    (variant (c) uses the raw triangles -- a flat facet is only a couple of
    triangles), matching the task's "sum over the real triangles" requirement.
    """
    import trimesh

    mesh = trimesh.load(obj_path, force="mesh")
    verts = np.asarray(mesh.vertices, dtype=float)
    tris = np.asarray(mesh.triangles, dtype=float)   # (T, 3, 3), trimesh winding
    ext = np.asarray(mesh.extents, dtype=float)
    ext_x, ext_y = float(ext[0]), float(ext[1])
    if ext_x <= 0 or ext_y <= 0:
        raise ValueError(f"Invalid facet extents {ext}; need ext_x,ext_y > 0")

    # Reference raw vertex whose transformed z is minimal (for the z_min lift).
    # For this planar facet with h>0 and pitch~pi/2, the ordering is pose
    # independent, so we pick it once at a nominal pose (h=1, phi=0).
    z_ref = _pick_zmin_ref(verts, ext_x, ext_y, 1.57, 0.0)

    return {
        "triangles": tris,
        "ext_x": ext_x,
        "ext_y": ext_y,
        "z_ref": z_ref,           # (vx, vy, vz) raw vertex achieving min z
        "n_triangles": int(tris.shape[0]),
    }


def _pick_zmin_ref(
    verts: np.ndarray, ext_x: float, ext_y: float, pitch: float, roll: float
) -> Tuple[float, float, float]:
    """Raw vertex with the smallest transformed z (before the z_min lift)."""
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    best_z = np.inf
    best_v = (0.0, 0.0, 0.0)
    for v in verts:
        Sx, Sy, Sz = v[0] * (0.5 / ext_x), v[1] * (1.0 / ext_y), v[2]
        # pitch about x
        Px, Py, Pz = Sx, Sy * cp - Sz * sp, Sy * sp + Sz * cp
        # roll about y
        Rz = -Px * sr + Pz * cr
        if Rz < best_z:
            best_z = Rz
            best_v = (float(v[0]), float(v[1]), float(v[2]))
    return best_v


# ---------------------------------------------------------------------------
# Symbolic model (built once, lambdified, cached by structural config key)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _build_symbolic(key: Tuple) -> Dict[str, Callable]:
    """Construct and lambdify y_q(psi; triangle) and its psi-derivatives.

    The symbolic expression is written PER TRIANGLE with the triangle's three
    raw vertices as *free* symbols (constants w.r.t. differentiation).  It is
    lambdified once; the numerical model then plugs each triangle's raw vertices
    in and sums.  ``key`` is ``ForwardConfig.as_key()`` -- everything that
    changes the closed form.  Grid coordinates ``(px, py, tlo, thi)`` also remain
    free arguments so the same callables serve any floor/time grid.
    """
    import sympy as sp

    (p_l, n_l, n_f, obj_path, facet_w, pitch, roll,
     laser_intensity, alpha, gain, beta, kappa, tau, c) = key

    mesh = _load_raw_mesh(_resolve_obj_path(obj_path))
    ext_x, ext_y = mesh["ext_x"], mesh["ext_y"]
    z_ref = mesh["z_ref"]

    rho, phi, h = sp.symbols("rho phi h", real=True)
    px, py, tlo, thi = sp.symbols("px py tlo thi", real=True)
    # raw triangle vertices (constants w.r.t. rho,phi,h differentiation)
    v0 = sp.Matrix(sp.symbols("v0x v0y v0z", real=True))
    v1 = sp.Matrix(sp.symbols("v1x v1y v1z", real=True))
    v2 = sp.Matrix(sp.symbols("v2x v2y v2z", real=True))

    # --- symbolic placement (mirrors simulation cell 8) ----------------------
    scale_x = sp.Float(facet_w) / sp.Float(ext_x)
    cp, sp_ = sp.cos(sp.Float(pitch)), sp.sin(sp.Float(pitch))
    cr, sr = sp.cos(sp.Float(roll)), sp.sin(sp.Float(roll))
    theta = phi + 3 * sp.pi / 2
    ct, st = sp.cos(theta), sp.sin(theta)

    def transform(v):
        """Apply scale -> pitch -> roll -> yaw -> (rho,phi lift) to a raw vertex."""
        Sx = scale_x * v[0]
        Sy = (h / sp.Float(ext_y)) * v[1]
        Sz = v[2]
        # pitch about x
        Px = Sx
        Py = Sy * cp - Sz * sp_
        Pz = Sy * sp_ + Sz * cp
        # roll about y
        Rx = Px * cr + Pz * sr
        Ry = Py
        Rz = -Px * sr + Pz * cr
        # yaw about z
        Yx = Rx * ct - Ry * st
        Yy = Rx * st + Ry * ct
        Yz = Rz
        return sp.Matrix([Yx, Yy, Yz])

    # z_min lift: transformed z of the reference raw vertex (before translation)
    z_ref_v = sp.Matrix([sp.Float(z_ref[0]), sp.Float(z_ref[1]), sp.Float(z_ref[2])])
    z_min = transform(z_ref_v)[2]
    trans = sp.Matrix([rho * sp.cos(phi), rho * sp.sin(phi), -z_min])

    V0 = transform(v0) + trans
    V1 = transform(v1) + trans
    V2 = transform(v2) + trans

    # --- per-triangle centroid, area, normal (closed form) -------------------
    centroid = (V0 + V1 + V2) / 3
    e1 = V1 - V0
    e2 = V2 - V0
    cr_vec = e1.cross(e2)                     # winding matches trimesh normal
    cross_norm = sp.sqrt(cr_vec.dot(cr_vec))
    area_tri = cross_norm / 2
    n_s = cr_vec / cross_norm                 # unit facet normal

    ps = centroid
    sx, sy = ps[0], ps[1]

    pf = sp.Matrix([px, py, 0])
    pl = sp.Matrix(p_l)
    nl = sp.Matrix(n_l)
    nf = sp.Matrix(n_f)

    # --- two-bounce distances (smooth in psi) -------------------------------
    d1 = sp.sqrt((ps - pl).dot(ps - pl))     # laser -> facet
    d2 = sp.sqrt((ps - pf).dot(ps - pf))     # facet -> floor pixel

    # --- foreshortening: FOUR two-bounce cosines with softplus hinges -------
    #   dot1 = max(0, n_s . (p_l - p_s)/d1)     dot2 = max(0, n_s . (p_f - p_s)/d2)
    #   dot3 = max(0, n_l . (p_s - p_l)/d1)     dot4 = max(0, n_f . (p_s - p_f)/d2)
    def softplus(x):
        return sp.log(1 + sp.exp(beta * x)) / beta

    c1 = n_s.dot(pl - ps) / d1
    c2 = n_s.dot(pf - ps) / d2
    c3 = nl.dot(ps - pl) / d1
    c4 = nf.dot(ps - pf) / d2
    f_fore = softplus(c1) * softplus(c2) * softplus(c3) * softplus(c4)

    # --- soft occlusion (replaces the ``xint>0`` mask) ----------------------
    # d_edge = x where the facet->floor segment crosses y=0 (== simulator xint).
    d_edge = sx - sy * (sx - px) / (sy - py)
    vis = 1 / (1 + sp.exp(-kappa * d_edge))

    # --- physical amplitude: simulator's intensity, smoothed ----------------
    #   intensity = laser_intensity * area * dot1*dot2*dot3*dot4 / (4 pi^2 d1^2 d2^2)
    denom = 4 * sp.pi**2 * d1**2 * d2**2
    A = sp.Float(laser_intensity) * sp.Float(alpha) * sp.Float(gain) \
        * area_tri * vis * f_fore / denom

    # --- analytic pulse integral over the bin [tlo, thi] --------------------
    # Unit-area Gaussian pulse at t0 = (d1+d2)/c; bin integral is a diff of erf.
    t0 = (d1 + d2) / c
    S = (
        sp.erf((thi - t0) / (sp.sqrt(2) * tau))
        - sp.erf((tlo - t0) / (sp.sqrt(2) * tau))
    ) / 2

    y = A * S  # y for a single (triangle, pixel, bin) -- background added later

    vargs = (v0[0], v0[1], v0[2], v1[0], v1[1], v1[2], v2[0], v2[1], v2[2])
    args = (rho, phi, h, px, py, tlo, thi) + vargs
    modules = ["scipy", "numpy"]
    out: Dict[str, Callable] = {
        "y": sp.lambdify(args, y, modules=modules),
        "dy_drho": sp.lambdify(args, sp.diff(y, rho), modules=modules),
        "dy_dphi": sp.lambdify(args, sp.diff(y, phi), modules=modules),
        "dy_dh": sp.lambdify(args, sp.diff(y, h), modules=modules),
        "t0": sp.lambdify((rho, phi, h, px, py) + vargs, t0, modules=modules),
    }
    return out


def _resolve_obj_path(obj_path: str) -> str:
    if os.path.isabs(obj_path) and os.path.exists(obj_path):
        return obj_path
    if os.path.exists(obj_path):
        return os.path.abspath(obj_path)
    cand = os.path.join(_THIS_DIR, obj_path)
    if os.path.exists(cand):
        return cand
    return obj_path


class AnalyticForwardModel:
    """Cached, lambdified analytic forward + exact Jacobian, summed over the
    real ``facet.obj`` triangles (variant (a)+(b)+(c))."""

    def __init__(self, config: Optional[ForwardConfig] = None):
        self.config = config or ForwardConfig()
        self._fns = _build_symbolic(self.config.as_key())
        mesh = _load_raw_mesh(self.config.resolved_obj_path())
        self._triangles = mesh["triangles"]        # (T, 3, 3)
        self.n_triangles = mesh["n_triangles"]
        self._px, self._py = self._build_pixel_grid()

    # --- grids --------------------------------------------------------------
    def _build_pixel_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        n = int(cfg.pixel_dim)
        half = cfg.fov_width / 2.0
        pitch = cfg.fov_width / n
        xs = np.linspace(
            cfg.fov_center_x - half + pitch / 2,
            cfg.fov_center_x + half - pitch / 2,
            n,
        )
        ys = np.linspace(
            cfg.fov_center_y - half + pitch / 2,
            cfg.fov_center_y + half - pitch / 2,
            n,
        )
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        return X.ravel(), Y.ravel()

    def _tri_args(self, idx: int) -> Tuple[float, ...]:
        t = self._triangles[idx]
        return (
            float(t[0, 0]), float(t[0, 1]), float(t[0, 2]),
            float(t[1, 0]), float(t[1, 1]), float(t[1, 2]),
            float(t[2, 0]), float(t[2, 1]), float(t[2, 2]),
        )

    def _time_bins(self, psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (tlo, thi) bin edges covering the pulse over ALL triangles."""
        cfg = self.config
        rho, phi, h = float(psi[0]), float(psi[1]), float(psi[2])
        t0_min, t0_max = np.inf, -np.inf
        for idx in range(self.n_triangles):
            t0 = np.asarray(
                self._fns["t0"](rho, phi, h, self._px, self._py, *self._tri_args(idx)),
                dtype=float,
            )
            t0_min = min(t0_min, float(np.min(t0)))
            t0_max = max(t0_max, float(np.max(t0)))
        dt = cfg.bin_size
        if cfg.n_time_bins is not None:
            j = np.arange(cfg.n_time_bins)
            return j * dt, (j + 1) * dt
        pad = cfg.time_margin_bins * dt + 4.0 * cfg.tau
        j_lo = int(np.floor((t0_min - pad) / dt))
        j_hi = int(np.ceil((t0_max + pad) / dt))
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
        PX, TLO = np.meshgrid(self._px, tlo, indexing="ij")
        PY, THI = np.meshgrid(self._py, thi, indexing="ij")
        return rho, phi, h, PX, PY, TLO, THI

    def _sum_over_triangles(self, fn_key: str, psi: np.ndarray, bins=None) -> np.ndarray:
        rho, phi, h, PX, PY, TLO, THI = self._grid_qr(psi, bins=bins)
        fn = self._fns[fn_key]
        acc = None
        for idx in range(self.n_triangles):
            val = np.asarray(
                fn(rho, phi, h, PX, PY, TLO, THI, *self._tri_args(idx)),
                dtype=float,
            )
            # lambdify may return a scalar if the expression collapses; broadcast
            val = np.broadcast_to(val, PX.shape).astype(float)
            acc = val if acc is None else acc + val
        return acc.ravel()

    def forward_y(self, psi: ArrayLike, bins=None) -> np.ndarray:
        psi = self._pad_psi(np.asarray(psi, dtype=float).ravel())
        return self._sum_over_triangles("y", psi, bins=bins)

    def forward_g(self, psi: ArrayLike, bins=None) -> np.ndarray:
        return self.forward_y(psi, bins=bins) + self.config.background

    def jacobian_g(self, psi: ArrayLike, n_params: int = 3, bins=None) -> np.ndarray:
        """Exact analytic Jacobian dg/dpsi (== dy/dpsi) of shape (N_q, n_params).

        Each column is summed over the mesh triangles (linearity of d/dpsi and of
        the triangle sum).
        """
        psi = self._pad_psi(np.asarray(psi, dtype=float).ravel())
        cols = [self._sum_over_triangles("dy_drho", psi, bins=bins),
                self._sum_over_triangles("dy_dphi", psi, bins=bins)]
        if n_params >= 3:
            cols.append(self._sum_over_triangles("dy_dh", psi, bins=bins))
        return np.stack(cols, axis=1)

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
    obtained from the *closed-form* analytic Jacobian summed over the real mesh
    triangles.
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
        "n_triangles": int(model.n_triangles),
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
          f"N_q={g.size}, J={J.shape}, n_triangles={m.n_triangles}")
    print(f"g range: [{g.min():.4g}, {g.max():.4g}]  (b={cfg.background})")
    res = compute_crb_analytic(psi0)
    print(f"sigma_rho={res['sigma_rho']:.4g} m, "
          f"sigma_phi={res['sigma_phi_deg']:.4g} deg, "
          f"sigma_h={res['sigma_height']:.4g} m")
