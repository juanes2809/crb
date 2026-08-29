#!/usr/bin/env python3
"""Validate the closed-form (sympy) Jacobian of the analytic differentiable
forward against central finite differences of the *same* smooth forward.

This checks that the symbolic derivatives ``dg_q/dpsi_m`` produced by
``crb_analytic_forward`` are correct.  It is NOT a comparison against the
rasterized simulator: the analytic point-facet forward and the mesh rasterizer
are *different* forward models (the analytic one replaces every non-smooth
operation -- ``ceil`` binning, hard occlusion, ``max(0,.)`` clamps -- by a
smooth closed-form analogue).  Here we only assert that analytic == FD for the
smooth model, i.e. that the differentiation is exact.

To make the comparison fair the time-bin grid is *frozen* at the nominal pose
(``model.freeze_bins``) and reused for the +/- perturbations, so the auto bin
window does not shift under perturbation.

Run:  python3 validate_analytic_forward.py
"""

from __future__ import annotations

import numpy as np

from crb_analytic_forward import AnalyticForwardModel, ForwardConfig


def central_fd_jacobian(
    model: AnalyticForwardModel,
    psi0: np.ndarray,
    steps: np.ndarray,
    bins,
    n_params: int,
) -> np.ndarray:
    """Central finite-difference Jacobian of g at psi0 with a FROZEN bin grid."""
    n = int(n_params)
    g0 = model.forward_g(psi0, bins=bins)
    J = np.zeros((g0.size, n), dtype=float)
    for m in range(n):
        d = np.zeros(3)
        d[m] = steps[m]
        g_p = model.forward_g(psi0 + d, bins=bins)
        g_m = model.forward_g(psi0 - d, bins=bins)
        J[:, m] = (g_p - g_m) / (2.0 * steps[m])
    return J


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    """Column-scaled max relative error between two Jacobian columns.

    We normalize the max absolute deviation by the *column* magnitude
    ``max|b|`` rather than element-wise: a per-element relative error is
    meaningless on the many background-dominated pixels where both the analytic
    and FD derivatives are essentially zero (``y_q << b``).  The column-scaled
    metric is the standard, well-conditioned way to check a Jacobian.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.max(np.abs(b))
    if denom == 0.0:
        return float(np.max(np.abs(a - b)))
    return float(np.max(np.abs(a - b)) / denom)


def main() -> int:
    cfg = ForwardConfig()
    model = AnalyticForwardModel(cfg)

    # FD steps: small vs the smoothing widths so truncation error dominates and
    # stays ~O(delta^2).  rho, h in metres; phi in radians.
    steps = np.array([1e-6, 1e-6, 1e-6], dtype=float)

    rhos = [0.5, 1.0, 1.5]
    phis_deg = [30.0, 90.0, 150.0]
    hs = [1.0]

    param_names = ["d/drho", "d/dphi", "d/dh"]
    print("=" * 78)
    print("Analytic (sympy) Jacobian  vs  central finite differences")
    print(f"grid: {cfg.pixel_dim}x{cfg.pixel_dim} pixels, background b={cfg.background}")
    print(f"smoothing: beta={cfg.beta}, kappa={cfg.kappa}, tau={cfg.tau:.3g}s, "
          f"FD step={steps[0]:.0e}")
    print("=" * 78)
    header = (
        f"{'rho':>5} {'phi_deg':>8} {'h':>5} | "
        f"{'err d/drho':>12} {'err d/dphi':>12} {'err d/dh':>12} | {'max':>10}"
    )
    print(header)
    print("-" * len(header))

    overall_max = 0.0
    n_params = 3
    for rho in rhos:
        for phi_deg in phis_deg:
            for h in hs:
                psi0 = np.array([rho, np.deg2rad(phi_deg), h], dtype=float)
                bins = model.freeze_bins(psi0)
                J_an = model.jacobian_g(psi0, n_params=n_params, bins=bins)
                J_fd = central_fd_jacobian(model, psi0, steps, bins, n_params)
                errs = [rel_error(J_an[:, m], J_fd[:, m]) for m in range(n_params)]
                row_max = max(errs)
                overall_max = max(overall_max, row_max)
                print(
                    f"{rho:5.2f} {phi_deg:8.1f} {h:5.2f} | "
                    f"{errs[0]:12.3e} {errs[1]:12.3e} {errs[2]:12.3e} | "
                    f"{row_max:10.2e}"
                )

    print("-" * len(header))
    print(f"OVERALL MAX relative error (analytic vs central FD): {overall_max:.3e}")
    tol = 1e-6
    ok = overall_max < tol
    print(f"Tolerance {tol:.0e}: {'PASS' if ok else 'FAIL'}")
    print("=" * 78)

    # Also report a sanity CRB at a nominal pose.
    from crb_analytic_forward import compute_crb_analytic

    res = compute_crb_analytic(np.array([1.0, np.deg2rad(90.0), 1.0]))
    print(
        "Sanity CRB @ (rho=1, phi=90deg, h=1): "
        f"sigma_rho={res['sigma_rho']:.4g} m, "
        f"sigma_phi={res['sigma_phi_deg']:.4g} deg, "
        f"sigma_h={res['sigma_height']:.4g} m"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
