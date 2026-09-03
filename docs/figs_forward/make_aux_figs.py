#!/usr/bin/env python3
"""Auxiliary figures for docs/forward_completo.tex.

Two figures that the existing component plots do not cover:

  esquema_esquina.png  -- top-down schematic of the corner-camera geometry and a
                          geometric derivation of the occlusion test xint = -b/m.
  cuantizacion_rho.png -- why a 1 cm finite-difference step works on a forward
                          whose time-of-flight is quantized in ~6-7 cm of range:
                          the single-triangle staircase, the aggregate response of
                          the densified mesh, and the stability of sigma vs step.

Run from the repository root:  python3 docs/figs_forward/make_aux_figs.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import simulator  # noqa: E402
from crb_fd_baseline import (  # noqa: E402
    BASELINE_BACKGROUND_RATE,
    BASELINE_FACET_WIDTH,
    baseline_simulation_kwargs,
    simulation_fn_for_crb,
)
from crb_polar_functions import compute_crb_polar  # noqa: E402

OUT = ROOT / "docs" / "figs_forward"
OUT.mkdir(parents=True, exist_ok=True)
C = simulator.C


# ----------------------------------------------------------------------------
# Figure 1: corner geometry + occlusion test
# ----------------------------------------------------------------------------
def fig_esquema() -> None:
    fov, npx = 0.25, 32
    rho, phi_deg = 1.0, 60.0
    phi = np.deg2rad(phi_deg)
    fx, fy = rho * np.cos(phi), rho * np.sin(phi)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 5.0))

    # ---- left: scene from above -------------------------------------------
    ax.axhline(0.0, color="0.55", lw=0.8, ls=":")
    ax.plot([-1.35, 0.0], [0, 0], color="k", lw=4, solid_capstyle="butt",
            label="arista ocluyente (pared, $x<0$)")
    ax.plot([0.0, 1.35], [0, 0], color="0.7", lw=1.6, ls="--",
            label="lado abierto ($x>0$)")

    ax.add_patch(plt.Rectangle((-fov / 2, -fov), fov, fov, facecolor="#cfe3f7",
                               edgecolor="#2d6ca8", lw=1.2, zorder=2))
    ax.text(0.0, -fov - 0.045, "SPAD: %d$\\times$%d px sobre el piso\nFOV %.2f m, pitch %.2f mm"
            % (npx, npx, fov, 1000 * fov / npx), ha="center", va="top", fontsize=8,
            color="#1c4f7d")

    ax.plot([0], [0], marker="*", ms=17, color="#d62728", zorder=5)
    ax.text(-0.20, 0.10, "laser en el origen\n$n_\\ell=+z$", fontsize=8,
            color="#d62728", ha="center")

    nx, ny = -np.cos(phi), -np.sin(phi)
    tx, ty = -ny, nx
    half = BASELINE_FACET_WIDTH / 2
    ax.plot([fx - half * tx, fx + half * tx], [fy - half * ty, fy + half * ty],
            color="#2ca02c", lw=5, solid_capstyle="butt", zorder=4)
    ax.add_patch(FancyArrowPatch((fx, fy), (fx + 0.26 * nx, fy + 0.26 * ny),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color="#2ca02c", lw=1.5, zorder=5))
    ax.text(fx + 0.30 * nx, fy + 0.30 * ny, "$\\vect{n}=(-\\cos\\varphi,-\\sin\\varphi,0)$"
            .replace("\\vect", ""), fontsize=8, color="#1a661a", ha="right")
    ax.text(fx + 0.05, fy + 0.08, "faceta oculta\n$w=%.1f$ m" % BASELINE_FACET_WIDTH,
            fontsize=8, color="#1a661a")

    ax.add_patch(FancyArrowPatch((0, 0), (fx, fy), arrowstyle="-|>", mutation_scale=12,
                                 color="#d62728", lw=1.4, ls="-", zorder=3))
    ax.text(0.5 * fx - 0.13, 0.5 * fy + 0.02, "$d_1$", fontsize=10, color="#d62728")
    px, py = 0.03, -0.10
    ax.add_patch(FancyArrowPatch((fx, fy), (px, py), arrowstyle="-|>", mutation_scale=12,
                                 color="#1f77b4", lw=1.4, zorder=3))
    ax.text(0.55 * fx + 0.06, 0.55 * fy - 0.10, "$d_2$", fontsize=10, color="#1f77b4")

    th = np.linspace(0, phi, 60)
    ax.plot(0.30 * np.cos(th), 0.30 * np.sin(th), color="0.35", lw=0.9)
    ax.text(0.35 * np.cos(phi / 2 - 0.10), 0.35 * np.sin(phi / 2 - 0.10),
            "$\\varphi$", fontsize=11)
    ax.text(0.5 * fx + 0.10, 0.5 * fy + 0.02, "$\\rho$", fontsize=11, color="0.25")

    ax.set_xlim(-0.75, 1.15)
    ax.set_ylim(-0.42, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_title("Escena vista desde arriba: dos rebotes laser $\\to$ faceta $\\to$ piso",
                 fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.2)

    # ---- right: the xint test ---------------------------------------------
    sx, sy = 0.62, 0.55           # a facet triangle centroid
    for k, (qx, qy, col, lab) in enumerate([
        (0.06, -0.12, "#2ca02c", "pixel visible: $x_{\\rm int}>0$"),
        (-0.11, -0.06, "#d62728", "pixel ocluido: $x_{\\rm int}<0$"),
    ]):
        m = (sy - qy) / (sx - qx)
        b = sy - m * sx
        xint = -b / m
        t = np.linspace(-0.5, 1.05, 2)
        bx.plot(t, m * t + b, color=col, lw=1.2, ls="--", alpha=0.85)
        bx.plot([qx], [qy], marker="s", ms=7, color=col)
        bx.plot([xint], [0], marker="o", ms=7, mfc="none", mec=col, mew=1.8)
        bx.annotate("$x_{\\rm int}=%+.3f$" % xint, (xint, 0), (xint - 0.02, 0.10 + 0.10 * k),
                    fontsize=8.5, color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
        bx.text(qx, qy - 0.055, lab, fontsize=8, color=col, ha="center")

    bx.plot([-0.5, 0.0], [0, 0], color="k", lw=4, solid_capstyle="butt")
    bx.plot([0.0, 1.05], [0, 0], color="0.7", lw=1.6, ls="--")
    bx.text(-0.25, 0.035, "pared opaca", fontsize=8)
    bx.text(0.45, 0.035, "hueco de la esquina", fontsize=8, color="0.4")
    bx.plot([sx], [sy], marker="^", ms=9, color="#2ca02c")
    bx.text(sx + 0.03, sy, "centroide del triangulo\n$\\vect{p}_s=(c_x,c_y)$".replace("\\vect", ""),
            fontsize=8, va="center")
    bx.add_patch(plt.Rectangle((-0.125, -0.25), 0.25, 0.25, facecolor="#cfe3f7",
                               edgecolor="#2d6ca8", lw=1.0, alpha=0.6, zorder=0))

    bx.set_xlim(-0.42, 1.0)
    bx.set_ylim(-0.30, 0.78)
    bx.set_aspect("equal")
    bx.set_xlabel("$x$ [m]")
    bx.set_ylabel("$y$ [m]")
    bx.set_title("Test de oclusion: la recta centroide$\\to$pixel debe cruzar $y=0$ en $x>0$",
                 fontsize=10)
    bx.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT / "esquema_esquina.png", dpi=190)
    plt.close(fig)
    print("wrote", OUT / "esquema_esquina.png")


# ----------------------------------------------------------------------------
# Figure 2: range quantization vs the finite-difference step
# ----------------------------------------------------------------------------
def fig_cuantizacion() -> None:
    sk = baseline_simulation_kwargs()
    w = BASELINE_FACET_WIDTH
    phi_deg, phi = 90.0, np.pi / 2
    dt_path = C * sk["bin_size"]
    npx = sk["cam_pixel_dim"]

    # representative pixel: centre of the SPAD patch
    fov = sk["camera_FOV"]
    cx0, cy0 = 0.0, -fov / 2
    iy, ix = npx // 2, npx // 2
    px = cx0 - fov / 2 + fov / (2 * npx) + ix * fov / npx
    py = cy0 - fov / 2 + fov / (2 * npx) + iy * fov / npx

    zc = 0.792514          # facet centroid height, from _transform_object_mesh
    rhos = np.linspace(0.90, 1.10, 401)

    def path_of_centroid(r):
        sx, sy = r * np.cos(phi), r * np.sin(phi)
        d1 = np.sqrt(sx**2 + sy**2 + zc**2)
        d2 = np.sqrt((px - sx) ** 2 + (py - sy) ** 2 + zc**2)
        return d1 + d2

    path = path_of_centroid(rhos)
    bins = np.ceil(path / dt_path)
    dpath_drho = np.gradient(path, rhos)
    rho_per_bin = dt_path / np.abs(dpath_drho).mean()

    # aggregate response of the full densified mesh
    coarse = np.arange(0.90, 1.1001, 0.005)
    t0 = time.time()
    agg, one_bin = [], []
    for r in coarse:
        y, _ = simulator.simulation(**{**sk, "object_positions": [
            {"obj_file": "facet.obj", "rho": float(r), "phi_deg": phi_deg, "w": w}]})
        agg.append(y[iy, ix, :].sum())
        one_bin.append(y[iy, ix, 21])
    agg, one_bin = np.array(agg), np.array(one_bin)
    print("mesh sweep: %d sims in %.1f s" % (len(coarse), time.time() - t0))

    # stability of sigma against the finite-difference step
    steps = np.array([0.002, 0.005, 0.01, 0.02, 0.04, 0.08])
    srho, sphi = [], []
    for d in steps:
        r = compute_crb_polar(
            simulation_fn=simulation_fn_for_crb, simulation_kwargs=sk,
            rho0=1.0, phi0=phi, width=w,
            finite_difference_steps=np.array([d, np.deg2rad(0.5)]),
            estimate_height=False, background_rate=BASELINE_BACKGROUND_RATE)
        srho.append(r["sigma_rho"])
        sphi.append(r["sigma_phi_deg"])
        print("  delta_rho=%.3f -> sigma_rho=%.6g m, sigma_phi=%.6g deg" % (d, srho[-1], sphi[-1]))

    fig, axs = plt.subplots(1, 3, figsize=(13.6, 4.2))

    a = axs[0]
    a.plot(rhos, path / dt_path, color="#1f77b4", lw=1.3, label="$(d_1+d_2)/(c\\Delta t)$")
    a.step(rhos, bins, where="post", color="#d62728", lw=1.3,
           label="$\\lceil\\,\\cdot\\,\\rceil$ (bin de llegada)")
    a.axvspan(1.0 - 0.01, 1.0 + 0.01, color="#ffcc66", alpha=0.55,
              label="paso FD $\\delta_\\rho=\\pm 1$ cm")
    a.axvline(1.0, color="0.3", lw=0.8, ls=":")
    a.set_xlabel("$\\rho$ [m]")
    a.set_ylabel("bin")
    a.set_title("Un solo punto de la faceta (centroide):\nescalera de $%.1f$ cm en $\\rho$"
                % (100 * rho_per_bin), fontsize=9.5)
    a.legend(fontsize=7.5, loc="upper left")
    a.grid(alpha=0.25)

    b = axs[1]
    b.plot(coarse, one_bin / one_bin.max(), marker="o", ms=3, lw=1.2, color="#2ca02c",
           label="un bin fijo, $y[i_y,i_x,21]$")
    b.plot(coarse, agg / agg.max(), marker="s", ms=3, lw=1.2, color="#7f2ca0",
           label="suma sobre bins del pixel")
    b.axvspan(1.0 - 0.01, 1.0 + 0.01, color="#ffcc66", alpha=0.55)
    b.axvline(1.0, color="0.3", lw=0.8, ls=":")
    b.set_xlabel("$\\rho$ [m]")
    b.set_ylabel("valor normalizado")
    b.set_title("Malla densificada (32768 triangulos):\nla agregacion suaviza la escalera",
                fontsize=9.5)
    b.legend(fontsize=7.5, loc="best")
    b.grid(alpha=0.25)

    c = axs[2]
    c.semilogx(steps, 1e3 * np.array(srho), marker="o", color="#1f77b4", lw=1.3,
               label="$\\sigma_\\rho$ [mm]")
    c.axvline(0.01, color="#ffa000", lw=1.6, ls="--", label="paso usado (1 cm)")
    c.set_ylabel("$\\sigma_\\rho$ [mm]", color="#1f77b4")
    c.tick_params(axis="y", labelcolor="#1f77b4")
    c2 = c.twinx()
    c2.semilogx(steps, sphi, marker="s", color="#d62728", lw=1.3, label="$\\sigma_\\varphi$ [deg]")
    c2.set_ylabel("$\\sigma_\\varphi$ [deg]", color="#d62728")
    c2.tick_params(axis="y", labelcolor="#d62728")
    c.set_xticks(steps)
    c.set_xticklabels(["%g" % (1000 * d) for d in steps], fontsize=8)
    c.minorticks_off()
    c.set_xlabel("$\\delta_\\rho$ [mm]")
    c.set_title("Estabilidad del CRB frente al paso FD\n$(\\rho,\\varphi)=(1\\,$m$,90^\\circ)$",
                fontsize=9.5)
    h1, l1 = c.get_legend_handles_labels()
    h2, l2 = c2.get_legend_handles_labels()
    c.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="best")
    c.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(OUT / "cuantizacion_rho.png", dpi=190)
    plt.close(fig)
    print("wrote", OUT / "cuantizacion_rho.png")

    with open(OUT / "cuantizacion_rho.txt", "w") as f:
        f.write("c*Delta_t = %.6f m de camino\n" % dt_path)
        f.write("|d(d1+d2)/d rho| medio = %.4f  ->  1 bin = %.4f m en rho\n"
                % (np.abs(dpath_drho).mean(), rho_per_bin))
        f.write("bins del centroide en rho in [0.90,1.10]: %d..%d\n"
                % (bins.min(), bins.max()))
        f.write("paso FD delta_rho = 0.01 m = %.1f %% de un bin\n"
                % (100 * 2 * 0.01 / rho_per_bin))
        for d, sr, sp in zip(steps, srho, sphi):
            f.write("delta_rho=%.3f m -> sigma_rho=%.6g m, sigma_phi=%.6g deg\n" % (d, sr, sp))
    print("wrote", OUT / "cuantizacion_rho.txt")


if __name__ == "__main__":
    fig_esquema()
    fig_cuantizacion()
