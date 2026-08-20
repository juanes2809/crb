"""CRB polar: Fisher Poisson + FD Jacobian for psi = [rho, phi] or [rho, phi, h]."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

ArrayLike = np.ndarray | Sequence[float]
SimulationFn = Callable[..., Tuple[Any, np.ndarray, Dict[str, Any]]]
JacobianMethod = Literal["central", "richardson"]


def _validate_psi_and_steps(psi0: ArrayLike, steps: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    psi0 = np.asarray(psi0, dtype=float).ravel()
    steps = np.asarray(steps, dtype=float).ravel()
    if psi0.size not in (2, 3):
        raise ValueError(f"psi0 must be length 2 or 3, got {psi0.shape}")
    if steps.size != psi0.size:
        raise ValueError(f"steps length {steps.size} != psi0 length {psi0.size}")
    if np.any(~np.isfinite(steps)) or np.any(steps <= 0):
        raise ValueError(f"steps must be finite and positive, got {steps}")
    if psi0[0] - steps[0] <= 0:
        raise ValueError(f"rho - delta_rho must stay > 0 (rho={psi0[0]}, d={steps[0]})")
    if psi0.size == 3 and psi0[2] - steps[2] <= 0:
        raise ValueError(f"h - delta_h must stay > 0 (h={psi0[2]}, d={steps[2]})")
    return psi0, steps


def normalize_fd_steps(
    finite_difference_steps: ArrayLike,
    *,
    estimate_height: bool = True,
    default_delta_h: float = 0.01,
) -> np.ndarray:
    steps = np.asarray(finite_difference_steps, dtype=float).ravel()
    if steps.size == 3:
        return steps
    if steps.size == 2:
        if estimate_height:
            return np.array([steps[0], steps[1], float(default_delta_h)], dtype=float)
        return steps
    raise ValueError(f"steps must have length 2 or 3, got {steps.size}")


def make_facet_object_position_polar(
    rho: float,
    phi: float,
    width: float,
    height: float = 1.0,
    obj_file: str = "facet.obj",
    yaw: float = 0.0,
    pitch: float = 1.57,
    roll: float = 0.0,
) -> Dict[str, Any]:
    rho, phi, height = float(rho), float(phi), float(height)
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError(f"rho must be > 0, got {rho}")
    if not np.isfinite(phi):
        raise ValueError(f"phi must be finite, got {phi}")
    if not np.isfinite(height) or height <= 0:
        raise ValueError(f"height must be > 0, got {height}")
    return {
        "obj_file": obj_file,
        "rho": rho,
        "phi": phi,
        "w": float(width),
        "h": height,
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll),
    }


def simulate_facet_signal_expected(
    simulation_fn: SimulationFn,
    simulation_kwargs: Dict[str, Any],
    rho: float,
    phi: float,
    width: float,
    height: float = 1.0,
    obj_file: str = "facet.obj",
    yaw: float = 0.0,
    pitch: float = 1.57,
    roll: float = 0.0,
    force_no_noise: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    kwargs = dict(simulation_kwargs)
    kwargs["object_positions"] = [
        make_facet_object_position_polar(
            rho, phi, width, height, obj_file, yaw, pitch, roll
        )
    ]
    if force_no_noise:
        kwargs["add_noise"] = False
    _, y_signal, params = simulation_fn(**kwargs)
    return np.asarray(y_signal, dtype=float), params


def forward_g_polar(
    simulation_fn: SimulationFn,
    simulation_kwargs: Dict[str, Any],
    rho: float,
    phi: float,
    width: float,
    height: float = 1.0,
    background_rate: float | np.ndarray = 0.0,
    obj_file: str = "facet.obj",
    yaw: float = 0.0,
    pitch: float = 1.57,
    roll: float = 0.0,
    force_no_noise: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    y_signal, params = simulate_facet_signal_expected(
        simulation_fn,
        simulation_kwargs,
        rho,
        phi,
        width,
        height,
        obj_file,
        yaw,
        pitch,
        roll,
        force_no_noise,
    )
    if np.isscalar(background_rate):
        background = float(background_rate) * np.ones_like(y_signal, dtype=float)
    else:
        background = np.asarray(background_rate, dtype=float)
        if background.shape != y_signal.shape:
            raise ValueError(
                f"background shape {background.shape} != signal {y_signal.shape}"
            )
    return y_signal + background, y_signal, params


def forward_g_from_psi(
    psi: ArrayLike,
    simulation_fn: SimulationFn,
    simulation_kwargs: Dict[str, Any],
    width: float,
    height: float = 1.0,
    background_rate: float | np.ndarray = 0.0,
    obj_file: str = "facet.obj",
    yaw: float = 0.0,
    pitch: float = 1.57,
    roll: float = 0.0,
    force_no_noise: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    psi = np.asarray(psi, dtype=float).ravel()
    if psi.size == 2:
        rho, phi, h = float(psi[0]), float(psi[1]), float(height)
    elif psi.size == 3:
        rho, phi, h = float(psi[0]), float(psi[1]), float(psi[2])
    else:
        raise ValueError(f"psi must be length 2 or 3, got {psi.shape}")
    g, y_signal, params = forward_g_polar(
        simulation_fn,
        simulation_kwargs,
        rho,
        phi,
        width,
        h,
        background_rate,
        obj_file,
        yaw,
        pitch,
        roll,
        force_no_noise,
    )
    return np.asarray(g, dtype=float).ravel(), y_signal, params


def _height_kw(psi: np.ndarray, height: float) -> float:
    return height if psi.size == 2 else float(psi[2])


def finite_difference_jacobian_polar(
    psi0: ArrayLike,
    steps: ArrayLike,
    simulation_fn: SimulationFn,
    simulation_kwargs: Dict[str, Any],
    width: float,
    height: float = 1.0,
    background_rate: float | np.ndarray = 0.0,
    obj_file: str = "facet.obj",
    yaw: float = 0.0,
    pitch: float = 1.57,
    roll: float = 0.0,
    force_no_noise: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    psi0, steps = _validate_psi_and_steps(psi0, steps)
    n = int(psi0.size)
    fwd = dict(
        simulation_fn=simulation_fn,
        simulation_kwargs=simulation_kwargs,
        width=width,
        background_rate=background_rate,
        obj_file=obj_file,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        force_no_noise=force_no_noise,
    )
    g0, y0, params0 = forward_g_from_psi(psi0, height=_height_kw(psi0, height), **fwd)
    J = np.zeros((g0.size, n), dtype=float)
    for m in range(n):
        dpsi = np.zeros(n)
        dpsi[m] = steps[m]
        psi_p, psi_m = psi0 + dpsi, psi0 - dpsi
        if verbose:
            print(f"FD param {m}: {psi_m[m]} .. {psi_p[m]}")
        g_p, _, _ = forward_g_from_psi(psi_p, height=_height_kw(psi_p, height), **fwd)
        g_m, _, _ = forward_g_from_psi(psi_m, height=_height_kw(psi_m, height), **fwd)
        J[:, m] = (g_p - g_m) / (2.0 * steps[m])
    return g0, J, y0, params0


def finite_difference_jacobian_polar_richardson(
    psi0: ArrayLike,
    steps: ArrayLike,
    simulation_fn: SimulationFn,
    simulation_kwargs: Dict[str, Any],
    width: float,
    height: float = 1.0,
    background_rate: float | np.ndarray = 0.0,
    obj_file: str = "facet.obj",
    yaw: float = 0.0,
    pitch: float = 1.57,
    roll: float = 0.0,
    force_no_noise: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    psi0, steps = _validate_psi_and_steps(psi0, steps)
    n = int(psi0.size)
    if psi0[0] - steps[0] / 2.0 <= 0:
        raise ValueError("rho - delta_rho/2 must stay > 0 for Richardson")
    if n == 3 and psi0[2] - steps[2] / 2.0 <= 0:
        raise ValueError("h - delta_h/2 must stay > 0 for Richardson")
    fwd = dict(
        simulation_fn=simulation_fn,
        simulation_kwargs=simulation_kwargs,
        width=width,
        background_rate=background_rate,
        obj_file=obj_file,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        force_no_noise=force_no_noise,
    )
    g0, y0, params0 = forward_g_from_psi(psi0, height=_height_kw(psi0, height), **fwd)
    J = np.zeros((g0.size, n), dtype=float)
    for m in range(n):
        d_h = np.zeros(n)
        d_h[m] = steps[m]
        d_h2 = np.zeros(n)
        d_h2[m] = steps[m] / 2.0
        if verbose:
            print(f"Richardson param {m}: h={steps[m]:.6g}")
        gp, _, _ = forward_g_from_psi(psi0 + d_h, height=_height_kw(psi0 + d_h, height), **fwd)
        gm, _, _ = forward_g_from_psi(psi0 - d_h, height=_height_kw(psi0 - d_h, height), **fwd)
        gp2, _, _ = forward_g_from_psi(
            psi0 + d_h2, height=_height_kw(psi0 + d_h2, height), **fwd
        )
        gm2, _, _ = forward_g_from_psi(
            psi0 - d_h2, height=_height_kw(psi0 - d_h2, height), **fwd
        )
        d1 = (gp - gm) / (2.0 * steps[m])
        d2 = (gp2 - gm2) / steps[m]
        J[:, m] = (4.0 * d2 - d1) / 3.0
    return g0, J, y0, params0


def compute_jacobian_polar(
    psi0: ArrayLike,
    steps: ArrayLike,
    simulation_fn: SimulationFn,
    simulation_kwargs: Dict[str, Any],
    width: float,
    height: float = 1.0,
    background_rate: float | np.ndarray = 0.0,
    obj_file: str = "facet.obj",
    yaw: float = 0.0,
    pitch: float = 1.57,
    roll: float = 0.0,
    force_no_noise: bool = True,
    method: JacobianMethod = "central",
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], str]:
    fn = (
        finite_difference_jacobian_polar_richardson
        if method == "richardson"
        else finite_difference_jacobian_polar
    )
    g0, J, y0, params0 = fn(
        psi0=psi0,
        steps=steps,
        simulation_fn=simulation_fn,
        simulation_kwargs=simulation_kwargs,
        width=width,
        height=height,
        background_rate=background_rate,
        obj_file=obj_file,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        force_no_noise=force_no_noise,
        verbose=verbose,
    )
    return g0, J, y0, params0, method


def fisher_poisson(g: np.ndarray, J: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    g = np.asarray(g, dtype=float).ravel()
    J = np.asarray(J, dtype=float)
    if J.shape[0] != g.size:
        raise ValueError(f"J rows {J.shape[0]} != g size {g.size}")
    g_safe = np.maximum(g, eps)
    return J.T @ ((1.0 / g_safe)[:, None] * J)


def _crb_sigmas_from_matrix(CRB: np.ndarray, rho0: float) -> Dict[str, float]:
    CRB = np.asarray(CRB, dtype=float)
    sigma_rho = float(np.sqrt(max(CRB[0, 0], 0.0)))
    sigma_phi = float(np.sqrt(max(CRB[1, 1], 0.0)))
    out = {
        "sigma_rho": sigma_rho,
        "sigma_phi": sigma_phi,
        "sigma_phi_deg": float(np.rad2deg(sigma_phi)),
        "sigma_tangential": float(rho0 * sigma_phi),
        "sigma_height": float("nan"),
    }
    if CRB.shape[0] >= 3 and CRB.shape[1] >= 3:
        out["sigma_height"] = float(np.sqrt(max(CRB[2, 2], 0.0)))
    return out


def crb_region_in_polar_parameters(
    rho0: float,
    phi0: float,
    CRB: np.ndarray,
    k: float = 3.0,
    n_points: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    Sigma = np.asarray(CRB, dtype=float)[np.ix_([0, 1], [0, 1])]
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    t = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = np.vstack([np.cos(t), np.sin(t)])
    axes = k * np.sqrt(eigenvalues)
    center = np.array([rho0, phi0], dtype=float)
    ellipse = center[:, None] + eigenvectors @ np.diag(axes) @ circle
    return ellipse[0, :], ellipse[1, :]


def crb_region_physical_xy_to_polar(
    rho0: float,
    phi0: float,
    CRB: np.ndarray,
    k: float = 3.0,
    n_points: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Sigma = np.asarray(CRB, dtype=float)[np.ix_([0, 1], [0, 1])]
    A = np.array(
        [
            [np.cos(phi0), -rho0 * np.sin(phi0)],
            [np.sin(phi0), rho0 * np.cos(phi0)],
        ],
        dtype=float,
    )
    Sigma_xy = A @ Sigma @ A.T
    mu = np.array([rho0 * np.cos(phi0), rho0 * np.sin(phi0)], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma_xy)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    t = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = np.vstack([np.cos(t), np.sin(t)])
    axes = k * np.sqrt(eigenvalues)
    xy = mu[:, None] + eigenvectors @ np.diag(axes) @ circle
    x, y = xy[0, :], xy[1, :]
    return np.sqrt(x**2 + y**2), np.arctan2(y, x), x, y


def compute_crb_polar(
    simulation_fn: SimulationFn,
    simulation_kwargs: Dict[str, Any],
    rho0: float,
    phi0: float,
    width: float,
    finite_difference_steps: ArrayLike,
    height: float = 1.0,
    estimate_height: bool = True,
    background_rate: float | np.ndarray = 0.0,
    obj_file: str = "facet.obj",
    yaw: float = 0.0,
    pitch: float = 1.57,
    roll: float = 0.0,
    fisher_eps: float = 1e-12,
    k: float = 3.0,
    force_no_noise: bool = True,
    jacobian_method: JacobianMethod = "central",
    verbose: bool = False,
) -> Dict[str, Any]:
    steps = normalize_fd_steps(finite_difference_steps, estimate_height=estimate_height)
    if steps.size == 3:
        psi0 = np.array([float(rho0), float(phi0), float(height)], dtype=float)
    else:
        psi0 = np.array([float(rho0), float(phi0)], dtype=float)

    g0, J, y0, params0, method_used = compute_jacobian_polar(
        psi0=psi0,
        steps=steps,
        simulation_fn=simulation_fn,
        simulation_kwargs=simulation_kwargs,
        width=width,
        height=height,
        background_rate=background_rate,
        obj_file=obj_file,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        force_no_noise=force_no_noise,
        method=jacobian_method,
        verbose=verbose,
    )

    I = fisher_poisson(g0, J, eps=fisher_eps)
    CRB = np.linalg.pinv(I)
    stats = _crb_sigmas_from_matrix(CRB, rho0=rho0)
    corr = (
        float(CRB[0, 1] / np.sqrt(CRB[0, 0] * CRB[1, 1]))
        if CRB[0, 0] > 0 and CRB[1, 1] > 0
        else np.nan
    )

    rho_p, phi_p = crb_region_in_polar_parameters(rho0, phi0, CRB, k=k)
    rho_phys, phi_phys, x_c, y_c = crb_region_physical_xy_to_polar(rho0, phi0, CRB, k=k)

    return {
        "rho0": float(rho0),
        "phi0": float(phi0),
        "phi0_deg": float(np.rad2deg(phi0)),
        "height0": float(height),
        "estimate_height": bool(steps.size == 3),
        "width": float(width),
        "steps": steps,
        "jacobian_method": method_used,
        "g0_vec": g0,
        "y0": y0,
        "params0": params0,
        "J": J,
        "I": I,
        "CRB": CRB,
        "sigma_rho": stats["sigma_rho"],
        "sigma_phi": stats["sigma_phi"],
        "sigma_phi_deg": stats["sigma_phi_deg"],
        "sigma_tangential": stats["sigma_tangential"],
        "sigma_height": stats["sigma_height"],
        "corr_rho_phi": corr,
        "rho_curve_param": rho_p,
        "phi_curve_param": phi_p,
        "rho_curve_phys": rho_phys,
        "phi_curve_phys": phi_phys,
        "x_curve_phys": x_c,
        "y_curve_phys": y_c,
    }


def compute_crb_grid_polar(
    simulation_fn: SimulationFn,
    simulation_kwargs: Dict[str, Any],
    ranges: Iterable[float],
    angles_deg: Iterable[float],
    width: float,
    finite_difference_steps: ArrayLike,
    height: float = 1.0,
    estimate_height: bool = True,
    background_rate: float | np.ndarray = 0.0,
    obj_file: str = "facet.obj",
    yaw: float = 0.0,
    pitch: float = 1.57,
    roll: float = 0.0,
    fisher_eps: float = 1e-12,
    k: float = 3.0,
    force_no_noise: bool = True,
    jacobian_method: JacobianMethod = "central",
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for rho in ranges:
        for phi_deg in angles_deg:
            if verbose:
                print(f"CRB rho={rho:.4g} m, phi={phi_deg} deg")
            result = compute_crb_polar(
                simulation_fn=simulation_fn,
                simulation_kwargs=simulation_kwargs,
                rho0=float(rho),
                phi0=float(np.deg2rad(phi_deg)),
                width=width,
                height=height,
                estimate_height=estimate_height,
                finite_difference_steps=finite_difference_steps,
                background_rate=background_rate,
                obj_file=obj_file,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                fisher_eps=fisher_eps,
                k=k,
                force_no_noise=force_no_noise,
                jacobian_method=jacobian_method,
                verbose=verbose,
            )
            results.append(result)
            if verbose:
                print(
                    f"  σρ={result['sigma_rho']:.4g} m, "
                    f"σφ={result['sigma_phi_deg']:.4g}°, "
                    f"ρσφ={result['sigma_tangential']:.4g} m"
                    + (
                        f", σh={result['sigma_height']:.4g} m"
                        if np.isfinite(result["sigma_height"])
                        else ""
                    )
                )
    return results


def _style_polar_crb_ax(ax, title: Optional[str] = None, title_fontsize: float = 10.0):
    """Radial labels with 2 decimals; slightly smaller fonts. Bubbles axis setup."""
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    # force 2-decimal radial tick labels (after rlim is set)
    rticks = ax.get_yticks()
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{t:.2f}" for t in rticks], fontsize=8)
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=12)


def plot_crb_regions_polar(
    results: Sequence[Dict[str, Any]],
    use_physical_region: bool = False,
    rlim: Optional[Tuple[float, float] | float] = None,
    title: str = r"CRB uncertainty regions",
    output_path: Optional[str | Path] = None,
    dpi: int = 200,
    color: Optional[str] = None,
    label: Optional[str] = None,
    ax=None,
):
    """Ellipses in (rho, phi) parameter space (bubbles), not xy pushforward/teardrops."""
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={"projection": "polar"})
    else:
        fig = ax.figure

    # Always prefer parameter-space ellipses unless explicitly requested otherwise
    key_r = "rho_curve_phys" if use_physical_region else "rho_curve_param"
    key_p = "phi_curve_phys" if use_physical_region else "phi_curve_param"
    for i, result in enumerate(results):
        kwargs = {"linewidth": 1.5}
        if color is not None:
            kwargs["color"] = color
        if label is not None and i == 0:
            kwargs["label"] = label
        ax.plot(result[key_p], result[key_r], **kwargs)
        sk = {"s": 15}
        if color is not None:
            sk["color"] = color
        ax.scatter([result["phi0"]], [result["rho0"]], **sk)

    if rlim is not None:
        if isinstance(rlim, tuple):
            ax.set_rlim(*rlim)
        else:
            ax.set_rlim(0, float(rlim))
    _style_polar_crb_ax(ax, title=title)
    if label is not None:
        ax.legend(loc="upper right", fontsize=7)

    if created_fig:
        fig.tight_layout()
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=dpi)
    return fig, ax


def plot_crb_regions_compare(
    results_2param: Sequence[Dict[str, Any]],
    results_3param: Sequence[Dict[str, Any]],
    rlim: float = 2.0,
    output_path: Optional[str | Path] = None,
    dpi: int = 200,
):
    """Side-by-side + overlay: CRB(rho,phi) vs CRB(rho,phi,h), ellipses/bubbles only."""
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection="polar")
    ax2 = fig.add_subplot(1, 3, 2, projection="polar")
    ax3 = fig.add_subplot(1, 3, 3, projection="polar")

    plot_crb_regions_polar(
        results_2param,
        use_physical_region=False,
        rlim=rlim,
        title=r"CRB$(\rho,\varphi)$",
        color="C0",
        ax=ax1,
    )
    plot_crb_regions_polar(
        results_3param,
        use_physical_region=False,
        rlim=rlim,
        title=r"CRB$(\rho,\varphi,h)$",
        color="C3",
        ax=ax2,
    )
    plot_crb_regions_polar(
        results_2param,
        use_physical_region=False,
        rlim=rlim,
        title=r"Comparacion (elipses)",
        color="C0",
        label=r"CRB$(\rho,\varphi)$",
        ax=ax3,
    )
    plot_crb_regions_polar(
        results_3param,
        use_physical_region=False,
        rlim=rlim,
        title=r"Comparacion (elipses)",
        color="C3",
        label=r"CRB$(\rho,\varphi,h)$",
        ax=ax3,
    )
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
    return fig
