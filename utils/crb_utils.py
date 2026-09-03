"""Shared helpers for the sensor-parameter CRB sweeps driven by `sr_crb.py`.

Everything here is a thin layer on top of the consolidated pipeline: the forward
model is `simulator.simulation` reached through `crb_fd_baseline.simulation_fn_for_crb`,
and the estimator is `crb_polar_functions.compute_crb_polar` with Poisson Fisher
information and a central finite-difference Jacobian.

The parameter vector is psi = [rho, phi]. The torch simulator scales the facet mesh
with `min(w / extents[0], 1.1 / extents[2])` and never reads an `h` key, so the object
height is not a controllable input: the Fisher information is 2x2, `estimate_height`
stays False and no `sigma_h` exists. `height` survives in the signatures below purely
for call-site compatibility and is inert.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crb_fd_baseline import (  # noqa: E402
    BASELINE_BACKGROUND_RATE,
    BASELINE_CAM_PIXEL_DIM,
    BASELINE_FACET_WIDTH,
    BASELINE_FD_STEPS,
    BASELINE_SCENE,
    baseline_simulation_kwargs,
    simulation_fn_for_crb,
)
from crb_polar_functions import compute_crb_polar  # noqa: E402

DEFAULT_RHO = 1.0
DEFAULT_PHI_DEG = 90.0
DEFAULT_HEIGHT = 1.0
DEFAULT_RANGE_STEP = 0.25
DEFAULT_ANGLE_STEP_DEG = 20.0
DEFAULT_RANGE_BOUNDS = (0.5, 1.75)
DEFAULT_ANGLE_BOUNDS_DEG = (30.0, 150.0)

_SCENE_KEYS = tuple(BASELINE_SCENE)
_height_notice_emitted = False


def _warn_height_is_inert(height: float) -> None:
    """Announce once that `height` cannot enter the forward model."""
    global _height_notice_emitted
    if _height_notice_emitted:
        return
    _height_notice_emitted = True
    print(
        f"note: height={height} m is inert — simulator.simulation has no 'h' input, "
        "so the CRB is 2-parameter (rho, phi) and sigma_h is undefined.",
        flush=True,
    )


@dataclass
class SimulationConfig:
    """Scene, sensor and torch settings for one forward-model configuration."""

    xmin: float = BASELINE_SCENE["xmin"]
    xmax: float = BASELINE_SCENE["xmax"]
    ymin: float = BASELINE_SCENE["ymin"]
    ymax: float = BASELINE_SCENE["ymax"]
    zmin: float = BASELINE_SCENE["zmin"]
    zmax: float = BASELINE_SCENE["zmax"]
    camera_FOV: float = BASELINE_SCENE["camera_FOV"]
    cam_pixel_dim: int = BASELINE_CAM_PIXEL_DIM
    bin_size: float = BASELINE_SCENE["bin_size"]
    laser_intensity: float = BASELINE_SCENE["laser_intensity"]
    hide_walls: bool = BASELINE_SCENE["hide_walls"]
    SNR_dB: float = BASELINE_SCENE["SNR_dB"]
    SBR: float = BASELINE_SCENE["SBR"]
    poisson_scale_factor: float = BASELINE_SCENE["poisson_scale_factor"]
    add_noise: bool = BASELINE_SCENE["add_noise"]
    torch_device: torch.device | None = None
    torch_dtype: torch.dtype = torch.float64
    triangle_chunk_size: int = 256

    def simulation_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments accepted by `simulator.simulation`."""
        kwargs = baseline_simulation_kwargs(
            cam_pixel_dim=self.cam_pixel_dim,
            torch_device=self.torch_device,
            torch_dtype=self.torch_dtype,
            triangle_chunk_size=self.triangle_chunk_size,
        )
        kwargs.update({key: getattr(self, key) for key in _SCENE_KEYS})
        return kwargs

    def label(self) -> str:
        return f"{self.cam_pixel_dim}x{self.cam_pixel_dim} px, FOV {self.camera_FOV:g} m"


@dataclass
class SweepAxis:
    """CRB standard deviations along one pose axis (range or angle)."""

    rho: np.ndarray
    phi_deg: np.ndarray
    sigma_rho: np.ndarray
    sigma_phi_deg: np.ndarray
    sigma_tangential: np.ndarray


@dataclass
class SweepEntry:
    """Range and angle CRB curves for a single swept configuration value."""

    value: float
    label: str
    range_sweep: SweepAxis
    angle_sweep: SweepAxis
    elapsed_s: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


def parse_int_list(text: str) -> List[int]:
    """Parse a comma-separated list of integers, e.g. "8,16,32,64"."""
    try:
        values = [int(token) for token in str(text).split(",") if token.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"expected at least one integer, got {text!r}")
    return values


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the pose, facet and sweep-resolution arguments shared by the CRB scripts."""
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO, help="Range [m] held fixed in the angle sweep.")
    parser.add_argument(
        "--phi-deg", type=float, default=DEFAULT_PHI_DEG, help="Angle [deg] held fixed in the range sweep."
    )
    parser.add_argument(
        "--height", type=float, default=DEFAULT_HEIGHT, help="Inert; the simulator has no height input."
    )
    parser.add_argument("--facet-width", type=float, default=BASELINE_FACET_WIDTH, help="Facet width w [m].")
    parser.add_argument(
        "--range-step",
        type=float,
        default=DEFAULT_RANGE_STEP,
        help=f"Spacing [m] of the range sweep over {DEFAULT_RANGE_BOUNDS} m.",
    )
    parser.add_argument(
        "--angle-step-deg",
        type=float,
        default=DEFAULT_ANGLE_STEP_DEG,
        help=f"Spacing [deg] of the angle sweep over {DEFAULT_ANGLE_BOUNDS_DEG} deg.",
    )
    return parser


def _axis_values(bounds: Tuple[float, float], step: float) -> np.ndarray:
    lo, hi = float(bounds[0]), float(bounds[1])
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"step must be finite and positive, got {step}")
    if hi < lo:
        raise ValueError(f"bounds must be increasing, got {bounds}")
    values = np.arange(lo, hi + 0.5 * step, step, dtype=float)
    return values if values.size else np.array([lo], dtype=float)


def _crb_along_axis(
    simulation_kwargs: Dict[str, Any],
    rhos: np.ndarray,
    phis_deg: np.ndarray,
    facet_width: float,
    background_rate: float,
    fd_steps: np.ndarray,
    obj_file: str,
    verbose: bool,
) -> SweepAxis:
    sigma_rho, sigma_phi_deg, sigma_tangential = [], [], []
    for rho, phi_deg in zip(rhos, phis_deg):
        result = compute_crb_polar(
            simulation_fn=simulation_fn_for_crb,
            simulation_kwargs=simulation_kwargs,
            rho0=float(rho),
            phi0=float(np.deg2rad(phi_deg)),
            width=facet_width,
            finite_difference_steps=fd_steps,
            estimate_height=False,
            background_rate=background_rate,
            obj_file=obj_file,
        )
        sigma_rho.append(result["sigma_rho"])
        sigma_phi_deg.append(result["sigma_phi_deg"])
        sigma_tangential.append(result["sigma_tangential"])
        if verbose:
            print(
                f"    rho={float(rho):.3f} m, phi={float(phi_deg):.1f} deg -> "
                f"sigma_rho={sigma_rho[-1]:.4g} m, sigma_phi={sigma_phi_deg[-1]:.4g} deg",
                flush=True,
            )
    return SweepAxis(
        rho=np.asarray(rhos, dtype=float),
        phi_deg=np.asarray(phis_deg, dtype=float),
        sigma_rho=np.asarray(sigma_rho, dtype=float),
        sigma_phi_deg=np.asarray(sigma_phi_deg, dtype=float),
        sigma_tangential=np.asarray(sigma_tangential, dtype=float),
    )


def run_sweep(
    values: Sequence[float],
    config_factory: Callable[[float], SimulationConfig],
    rho: float = DEFAULT_RHO,
    phi_deg: float = DEFAULT_PHI_DEG,
    height: float = DEFAULT_HEIGHT,
    facet_width: float = BASELINE_FACET_WIDTH,
    range_step: float = DEFAULT_RANGE_STEP,
    angle_step_deg: float = DEFAULT_ANGLE_STEP_DEG,
    range_bounds: Tuple[float, float] = DEFAULT_RANGE_BOUNDS,
    angle_bounds_deg: Tuple[float, float] = DEFAULT_ANGLE_BOUNDS_DEG,
    background_rate: float = BASELINE_BACKGROUND_RATE,
    fd_steps: np.ndarray = BASELINE_FD_STEPS,
    obj_file: str = "facet.obj",
    verbose: bool = True,
) -> List[SweepEntry]:
    """CRB range and angle sweeps for each swept configuration value.

    For every value, `config_factory` builds the forward-model configuration; the range
    sweep walks rho over `range_bounds` in `range_step` steps at fixed `phi_deg`, and the
    angle sweep walks phi over `angle_bounds_deg` in `angle_step_deg` steps at fixed `rho`.
    `height` is accepted for compatibility and never reaches the simulator.
    """
    _warn_height_is_inert(height)

    rhos = _axis_values(range_bounds, range_step)
    angles_deg = _axis_values(angle_bounds_deg, angle_step_deg)
    entries: List[SweepEntry] = []

    for value in values:
        config = config_factory(value)
        simulation_kwargs = config.simulation_kwargs()
        t0 = time.time()
        if verbose:
            print(
                f"[{config.label()}] {rhos.size} range poses + {angles_deg.size} angle poses",
                flush=True,
            )
            print("  range sweep", flush=True)
        range_sweep = _crb_along_axis(
            simulation_kwargs,
            rhos,
            np.full(rhos.shape, float(phi_deg)),
            facet_width,
            background_rate,
            fd_steps,
            obj_file,
            verbose,
        )
        if verbose:
            print("  angle sweep", flush=True)
        angle_sweep = _crb_along_axis(
            simulation_kwargs,
            np.full(angles_deg.shape, float(rho)),
            angles_deg,
            facet_width,
            background_rate,
            fd_steps,
            obj_file,
            verbose,
        )
        elapsed = time.time() - t0
        if verbose:
            print(f"  done in {elapsed:.1f} s", flush=True)
        entries.append(
            SweepEntry(
                value=float(value),
                label=config.label(),
                range_sweep=range_sweep,
                angle_sweep=angle_sweep,
                elapsed_s=elapsed,
                meta={
                    "cam_pixel_dim": config.cam_pixel_dim,
                    "camera_FOV": config.camera_FOV,
                    "facet_width": float(facet_width),
                    "background_rate": float(background_rate),
                    "fd_steps": np.asarray(fd_steps, dtype=float),
                    "rho_fixed": float(rho),
                    "phi_deg_fixed": float(phi_deg),
                    "estimate_height": False,
                },
            )
        )
    return entries


def plot_range_angle_sweep(
    results: Sequence[SweepEntry],
    x_label: str = "Value",
    image_title_prefix: str = "Configuration",
    output_path: str | Path | None = None,
    dpi: int = 200,
):
    """Plot sigma_rho and sigma_phi against range and against angle, one curve per value."""
    if not results:
        raise ValueError("results is empty")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = (
        (axes[0, 0], "range_sweep", "rho", "sigma_rho", r"$\rho$ [m]", r"$\sigma_\rho$ [m]"),
        (axes[0, 1], "range_sweep", "rho", "sigma_phi_deg", r"$\rho$ [m]", r"$\sigma_\varphi$ [deg]"),
        (axes[1, 0], "angle_sweep", "phi_deg", "sigma_rho", r"$\varphi$ [deg]", r"$\sigma_\rho$ [m]"),
        (axes[1, 1], "angle_sweep", "phi_deg", "sigma_phi_deg", r"$\varphi$ [deg]", r"$\sigma_\varphi$ [deg]"),
    )

    for ax, axis_name, x_key, y_key, xlabel, ylabel in panels:
        for index, entry in enumerate(results):
            axis = getattr(entry, axis_name)
            ax.semilogy(
                getattr(axis, x_key),
                getattr(axis, y_key),
                marker="o",
                markersize=4,
                linewidth=1.4,
                color=f"C{index % 10}",
                label=f"{x_label} = {entry.value:g}",
            )
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, which="both", alpha=0.25)

    axes[0, 0].set_title("Range sweep", fontsize=10)
    axes[0, 1].set_title("Range sweep", fontsize=10)
    axes[1, 0].set_title("Angle sweep", fontsize=10)
    axes[1, 1].set_title("Angle sweep", fontsize=10)
    axes[0, 0].legend(fontsize=8, loc="best")

    fig.suptitle(
        rf"{image_title_prefix} sweep — FD CRB$(\rho,\varphi)$ (Fisher 2$\times$2, no $\sigma_h$)",
        fontsize=11,
    )
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
    return fig, axes
