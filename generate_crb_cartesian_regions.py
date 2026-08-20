#!/usr/bin/env python3
"""Regenerate CRB ellipse plots on RECTANGULAR Cartesian axes (rho vs phi).

Standalone re-plotting from the cached grid results (no simulation is run).
X axis = rho (meters), Y axis = phi (degrees). The 3-sigma regions in the
(rho, phi) parameter space are true ellipses, so on rectangular axes they
appear as clean closed ellipses (bubbles), not curved teardrops.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
PKL_PATH = PLOTS / "crb_grid_results.pkl"


def load_cache() -> Dict[str, Any]:
    with open(PKL_PATH, "rb") as f:
        cache = pickle.load(f)
    print(f"Loaded cache keys={list(cache.keys())}", flush=True)
    return cache


def _style_cartesian_crb_ax(ax, title: Optional[str] = None, title_fontsize: float = 10.0):
    """Rectangular axes: rho (m) on X with 2-decimal ticks, phi (deg) on Y."""
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel(r"$\rho$ (m)", fontsize=9)
    ax.set_ylabel(r"$\varphi$ (deg)", fontsize=9)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=12)


def plot_crb_regions_cartesian(
    results: Sequence[Dict[str, Any]],
    title: str = r"CRB uncertainty regions",
    output_path: Optional[str | Path] = None,
    dpi: int = 200,
    color: Optional[str] = None,
    label: Optional[str] = None,
    ax=None,
):
    """Ellipses in (rho, phi) parameter space on rectangular axes (bubbles)."""
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    for i, result in enumerate(results):
        x = result["rho_curve_param"]
        y = np.rad2deg(result["phi_curve_param"])
        kwargs = {"linewidth": 1.5}
        if color is not None:
            kwargs["color"] = color
        if label is not None and i == 0:
            kwargs["label"] = label
        ax.plot(x, y, **kwargs)
        sk = {"s": 15}
        if color is not None:
            sk["color"] = color
        ax.scatter([result["rho0"]], [np.rad2deg(result["phi0"])], **sk)

    _style_cartesian_crb_ax(ax, title=title)
    if label is not None:
        ax.legend(loc="upper right", fontsize=7)

    if created_fig:
        fig.tight_layout()
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=dpi)
    return fig, ax


def plot_crb_regions_compare_cartesian(
    results_2param: Sequence[Dict[str, Any]],
    results_3param: Sequence[Dict[str, Any]],
    output_path: Optional[str | Path] = None,
    dpi: int = 200,
):
    """Side-by-side + overlay comparison on rectangular axes (bubbles only)."""
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    plot_crb_regions_cartesian(
        results_2param,
        title=r"CRB$(\rho,\varphi)$",
        color="C0",
        ax=ax1,
    )
    plot_crb_regions_cartesian(
        results_3param,
        title=r"CRB$(\rho,\varphi,h)$",
        color="C3",
        ax=ax2,
    )
    plot_crb_regions_cartesian(
        results_2param,
        title=r"Comparacion (elipses)",
        color="C0",
        label=r"CRB$(\rho,\varphi)$",
        ax=ax3,
    )
    plot_crb_regions_cartesian(
        results_3param,
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


def plot_polar_vs_cartesian(
    results: Sequence[Dict[str, Any]],
    suptitle: str,
    output_path: Optional[str | Path] = None,
    dpi: int = 200,
):
    """Side-by-side: same 3-sigma regions in polar (arched) vs Cartesian (straight)."""
    from crb_polar_functions import plot_crb_regions_polar

    fig = plt.figure(figsize=(12, 5))
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    ax_cart = fig.add_subplot(1, 2, 2)

    plot_crb_regions_polar(
        results,
        use_physical_region=False,
        rlim=2.0,
        title=r"Polar: CRB$(\rho,\varphi,h)$",
        ax=ax_polar,
    )
    plot_crb_regions_cartesian(
        results,
        title=r"Cartesiano: CRB$(\rho,\varphi,h)$",
        ax=ax_cart,
    )
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
    return fig


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    cache = load_cache()
    results_2 = cache["fixed_h"]
    results_3 = cache["with_h"]

    plot_crb_regions_cartesian(
        results_2,
        title=r"CRB$(\rho,\varphi)$ — regiones $3\sigma$ (cartesiano)",
        output_path=PLOTS / "crb_standard_regions_fixed_h_cartesian.png",
    )
    print("Saved crb_standard_regions_fixed_h_cartesian.png", flush=True)

    plot_crb_regions_cartesian(
        results_3,
        title=r"CRB$(\rho,\varphi,h)$ — regiones $3\sigma$ en $(\rho,\varphi)$ (cartesiano)",
        output_path=PLOTS / "crb_standard_regions_cartesian.png",
    )
    print("Saved crb_standard_regions_cartesian.png", flush=True)

    plot_crb_regions_compare_cartesian(
        results_2,
        results_3,
        output_path=PLOTS / "crb_regions_compare_2_vs_3_cartesian.png",
    )
    print("Saved crb_regions_compare_2_vs_3_cartesian.png", flush=True)

    hard2 = [r for r in results_2 if abs(r["rho0"] - 1.5) < 1e-9]
    hard3 = [r for r in results_3 if abs(r["rho0"] - 1.5) < 1e-9]
    plot_crb_regions_compare_cartesian(
        hard2,
        hard3,
        output_path=PLOTS / "crb_regions_compare_rho1p5_cartesian.png",
    )
    print("Saved crb_regions_compare_rho1p5_cartesian.png", flush=True)

    plot_polar_vs_cartesian(
        results_3,
        suptitle=(
            r"Misma region $3\sigma$ — Polar (arqueada) vs "
            r"Cartesiano (elipse recta)"
        ),
        output_path=PLOTS / "crb_regions_polar_vs_cartesian.png",
    )
    print("Saved crb_regions_polar_vs_cartesian.png", flush=True)

    hard3_pvc = [r for r in results_3 if abs(r["rho0"] - 1.5) < 1e-9]
    plot_polar_vs_cartesian(
        hard3_pvc,
        suptitle=(
            r"Misma region $3\sigma$ ($\rho_0=1.5$ m) — Polar (arqueada) vs "
            r"Cartesiano (elipse recta)"
        ),
        output_path=PLOTS / "crb_regions_polar_vs_cartesian_rho1p5.png",
    )
    print("Saved crb_regions_polar_vs_cartesian_rho1p5.png. DONE.", flush=True)


if __name__ == "__main__":
    main()
