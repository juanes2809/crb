#!/usr/bin/env python3
"""Regenerate CRB ellipse plots: CRB(rho,phi) vs CRB(rho,phi,h) + comparison."""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def load_notebook_namespace(notebook_path: Path, cell_indices: list[int]) -> dict:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__main__"}
    for idx in cell_indices:
        cell = notebook["cells"][idx]
        if cell["cell_type"] != "code":
            raise ValueError(f"Cell {idx} is not code")
        exec("".join(cell["source"]), ns)
    return ns


def main() -> None:
    import numpy as np
    from crb_polar_functions import (
        compute_crb_grid_polar,
        plot_crb_regions_compare,
        plot_crb_regions_polar,
    )

    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)
    pkl_path = plots / "crb_grid_results.pkl"

    ns = load_notebook_namespace(
        ROOT / "simulation_polar_clean.ipynb",
        cell_indices=[1, 3, 5, 7, 8],
    )
    sim_kwargs = dict(
        xmin=ns["xmin"],
        xmax=ns["xmax"],
        ymax=ns["ymax"],
        zmax=ns["zmax"],
        camera_FOV=ns["camera_FOV"],
        cam_pixel_dim=ns["cam_pixel_dim"],
        bin_size=ns["bin_size"],
        laser_intensity=ns["laser_intensity"],
        object_positions=[],
        hide_walls=True,
        SNR_dB=ns["SNR_dB"],
        SBR=ns["SBR"],
        poisson_scale_factor=ns["poisson_scale_factor"],
        add_noise=False,
        subpixel_dim=4,
    )
    ranges = np.array([0.5, 1.0, 1.5])
    angles = np.array([30, 60, 90, 120, 150])
    common = dict(
        simulation_fn=ns["simulation"],
        simulation_kwargs=sim_kwargs,
        ranges=ranges,
        angles_deg=angles,
        width=0.5,
        height=1.0,
        background_rate=0.1,
        verbose=False,
    )

    cache: dict = {}
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            cache = pickle.load(f)
        print(f"Loaded cache keys={list(cache.keys())}", flush=True)

    if "fixed_h" not in cache:
        print("=== CRB(rho, phi) Fisher 2x2 ===", flush=True)
        cache["fixed_h"] = compute_crb_grid_polar(
            **common,
            estimate_height=False,
            finite_difference_steps=np.array([0.01, np.deg2rad(0.5)]),
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(cache, f)
        print("Cached fixed_h", flush=True)
    else:
        print("Reusing cached fixed_h", flush=True)

    results_2 = cache["fixed_h"]
    plot_crb_regions_polar(
        results_2,
        use_physical_region=False,
        rlim=2.0,
        title=r"CRB$(\rho,\varphi)$ — regiones $3\sigma$",
        output_path=plots / "crb_standard_regions_fixed_h.png",
    )
    print("Saved crb_standard_regions_fixed_h.png", flush=True)

    if "with_h" not in cache:
        print("=== CRB(rho, phi, h) Fisher 3x3 ===", flush=True)
        cache["with_h"] = compute_crb_grid_polar(
            **common,
            estimate_height=True,
            finite_difference_steps=np.array([0.01, np.deg2rad(0.5), 0.01]),
        )
        with open(pkl_path, "wb") as f:
            pickle.dump(cache, f)
        print("Cached with_h", flush=True)
    else:
        print("Reusing cached with_h", flush=True)

    results_3 = cache["with_h"]
    plot_crb_regions_polar(
        results_3,
        use_physical_region=False,
        rlim=2.0,
        title=r"CRB$(\rho,\varphi,h)$ — regiones $3\sigma$ en $(\rho,\varphi)$",
        output_path=plots / "crb_standard_regions.png",
    )
    print("Saved crb_standard_regions.png", flush=True)

    plot_crb_regions_compare(
        results_2,
        results_3,
        rlim=2.0,
        output_path=plots / "crb_regions_compare_2_vs_3.png",
    )
    hard2 = [r for r in results_2 if abs(r["rho0"] - 1.5) < 1e-9]
    hard3 = [r for r in results_3 if abs(r["rho0"] - 1.5) < 1e-9]
    plot_crb_regions_compare(
        hard2,
        hard3,
        rlim=2.0,
        output_path=plots / "crb_regions_compare_rho1p5.png",
    )
    print("Saved compare plots. DONE.", flush=True)


if __name__ == "__main__":
    main()
