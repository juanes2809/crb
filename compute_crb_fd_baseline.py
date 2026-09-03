#!/usr/bin/env python3
"""FD CRB baseline: grid of (rho, phi) uncertainty regions on a 32x32 SPAD grid.

Drives `crb_polar_functions` (Poisson Fisher information + central finite-difference
Jacobian) against the torch simulator through `crb_fd_baseline.simulation_fn_for_crb`.

Results are cached in plots/crb_fd_baseline_32.pkl so the plot can be restyled
without re-running the forward model, and the bubbles are written to
plots/crb_fd_baseline_32.png.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from crb_fd_baseline import (  # noqa: E402
    BASELINE_ANGLES_DEG,
    BASELINE_BACKGROUND_RATE,
    BASELINE_CAM_PIXEL_DIM,
    BASELINE_FACET_WIDTH,
    BASELINE_FD_STEPS,
    BASELINE_RANGES,
    baseline_simulation_kwargs,
    simulation_fn_for_crb,
)
from crb_polar_functions import (  # noqa: E402
    compute_crb_grid_polar,
    plot_crb_regions_polar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cam-pixel-dim", type=int, default=BASELINE_CAM_PIXEL_DIM)
    parser.add_argument("--triangle-chunk-size", type=int, default=256)
    parser.add_argument("--force", action="store_true", help="Ignore any cached results.")
    return parser.parse_args()


def print_table(results: list[dict]) -> None:
    header = (
        f"{'rho [m]':>8}  {'phi [deg]':>9}  {'sigma_rho [m]':>13}  "
        f"{'sigma_phi [deg]':>15}  {'rho*sigma_phi [m]':>17}  {'corr':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['rho0']:>8.2f}  {r['phi0_deg']:>9.1f}  {r['sigma_rho']:>13.4g}  "
            f"{r['sigma_phi_deg']:>15.4g}  {r['sigma_tangential']:>17.4g}  "
            f"{r['corr_rho_phi']:>7.3f}"
        )


def main() -> None:
    args = parse_args()
    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)
    pkl_path = plots / f"crb_fd_baseline_{args.cam_pixel_dim}.pkl"
    png_path = plots / f"crb_fd_baseline_{args.cam_pixel_dim}.png"

    sim_kwargs = baseline_simulation_kwargs(
        cam_pixel_dim=args.cam_pixel_dim,
        triangle_chunk_size=args.triangle_chunk_size,
    )
    print(f"torch device: {sim_kwargs['torch_device']}, dtype: {sim_kwargs['torch_dtype']}")
    print(f"cam_pixel_dim: {args.cam_pixel_dim}, camera_FOV: {sim_kwargs['camera_FOV']}")

    cache: dict = {}
    if pkl_path.exists() and not args.force:
        with open(pkl_path, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded cache from {pkl_path} (keys={list(cache)})", flush=True)

    if "fd_2param" not in cache:
        n_poses = len(BASELINE_RANGES) * len(BASELINE_ANGLES_DEG)
        print(
            f"=== FD CRB(rho, phi), Fisher 2x2, {n_poses} poses, "
            f"5 forward evaluations each ===",
            flush=True,
        )
        t0 = time.time()
        cache["fd_2param"] = compute_crb_grid_polar(
            simulation_fn=simulation_fn_for_crb,
            simulation_kwargs=sim_kwargs,
            ranges=BASELINE_RANGES,
            angles_deg=BASELINE_ANGLES_DEG,
            width=BASELINE_FACET_WIDTH,
            background_rate=BASELINE_BACKGROUND_RATE,
            estimate_height=False,
            finite_difference_steps=BASELINE_FD_STEPS,
            verbose=True,
        )
        elapsed = time.time() - t0
        cache["meta"] = {
            "cam_pixel_dim": args.cam_pixel_dim,
            "simulation_kwargs": {
                k: v for k, v in sim_kwargs.items() if k != "object_positions"
            },
            "ranges": BASELINE_RANGES,
            "angles_deg": BASELINE_ANGLES_DEG,
            "width": BASELINE_FACET_WIDTH,
            "background_rate": BASELINE_BACKGROUND_RATE,
            "fd_steps": BASELINE_FD_STEPS,
            "elapsed_s": elapsed,
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(cache, f)
        print(
            f"Computed {n_poses} poses in {elapsed:.1f} s "
            f"({elapsed / n_poses:.2f} s/pose); cached to {pkl_path}",
            flush=True,
        )
    else:
        print("Reusing cached fd_2param", flush=True)

    results = cache["fd_2param"]
    print()
    print_table(results)

    sigma_rho = np.array([r["sigma_rho"] for r in results])
    sigma_phi_deg = np.array([r["sigma_phi_deg"] for r in results])
    print()
    print(f"sigma_rho     : min {sigma_rho.min():.4g} m,   max {sigma_rho.max():.4g} m")
    print(f"sigma_phi     : min {sigma_phi_deg.min():.4g} deg, max {sigma_phi_deg.max():.4g} deg")
    print("sigma_height  : not defined (the simulator has no h input; Fisher is 2x2)")

    plot_crb_regions_polar(
        results,
        use_physical_region=False,
        rlim=2.0,
        title=(
            rf"CRB$(\rho,\varphi)$ FD — regiones $3\sigma$ — "
            rf"grilla {args.cam_pixel_dim}$\times${args.cam_pixel_dim}"
        ),
        output_path=png_path,
    )
    print(f"\nSaved {png_path}")


if __name__ == "__main__":
    main()
