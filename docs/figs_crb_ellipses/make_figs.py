#!/usr/bin/env python3
"""Pedagogical figures for docs/de_datos_a_elipses.tex."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
HERE = Path(__file__).resolve().parent
HERE.mkdir(parents=True, exist_ok=True)

# Representative CRB at (1.0 m, 90 deg), 32x32 baseline:
# sigma_rho = 8.80 mm, sigma_phi = 2.574 deg, corr = -0.283
SIGMA_RHO = 0.008795
SIGMA_PHI = np.deg2rad(2.5741)
CORR = -0.283
RHO0, PHI0 = 1.0, np.deg2rad(90.0)


def covariance() -> np.ndarray:
    s1, s2, r = SIGMA_RHO, SIGMA_PHI, CORR
    return np.array([[s1**2, r * s1 * s2], [r * s1 * s2, s2**2]])


def ellipse_points(C: np.ndarray, k: float, n: int = 400) -> np.ndarray:
    evals, evecs = np.linalg.eigh(C)
    evals = np.maximum(evals, 0.0)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    circle = np.vstack([np.cos(t), np.sin(t)])
    axes = k * np.sqrt(evals)
    return evecs @ np.diag(axes) @ circle


def fig_sigma_scales() -> None:
    C = covariance()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#d62728"}
    fills = {1: 0.22, 2: 0.12, 3: 0.06}
    labels = {
        1: r"$1\sigma$  (~39 % en 2D)",
        2: r"$2\sigma$  (~86 % en 2D)",
        3: r"$3\sigma$  (~99 % en 2D)",
    }
    for k in (3, 2, 1):
        pts = ellipse_points(C, k)
        ax.fill(pts[0] * 1e3, np.rad2deg(pts[1]), color=colors[k], alpha=fills[k], zorder=1)
        ax.plot(
            pts[0] * 1e3,
            np.rad2deg(pts[1]),
            color=colors[k],
            lw=2.0 if k == 1 else 1.5,
            label=labels[k],
            zorder=2,
        )
    ax.axhline(0, color="0.75", lw=0.6)
    ax.axvline(0, color="0.75", lw=0.6)
    ax.plot(0, 0, "k.", ms=8, zorder=3)
    ax.set_xlabel(r"error en $\rho$  [mm]")
    ax.set_ylabel(r"error en $\varphi$  [deg]")
    ax.set_title(
        r"La misma covarianza CRB, tres radios $k$"
        "\n"
        r"$(\rho,\varphi)=(1\,\mathrm{m},90^\circ)$, grilla $32\times32$",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_aspect("auto")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(HERE / "sigma_scales.png", dpi=180)
    plt.close(fig)


def fig_mahalanobis() -> None:
    C = covariance()
    evals, evecs = np.linalg.eigh(C)
    # Draw unit circle in u-space and its image (1σ ellipse) in δ-space,
    # with the two principal axes.
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))

    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.vstack([np.cos(t), np.sin(t)])
    ax = axes[0]
    ax.plot(circle[0], circle[1], color="#1f77b4", lw=2)
    ax.plot([0, 1], [0, 0], color="0.3", lw=1.2)
    ax.plot([0, 0], [0, 1], color="0.3", lw=1.2)
    ax.annotate(r"$u_1$", xy=(1.08, 0), va="center", fontsize=10)
    ax.annotate(r"$u_2$", xy=(0, 1.08), ha="center", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axhline(0, color="0.8", lw=0.5)
    ax.axvline(0, color="0.8", lw=0.5)
    ax.set_title(r"Círculo unidad $\|u\|=1$", fontsize=11)
    ax.set_xlabel(r"$u_1$")
    ax.set_ylabel(r"$u_2$")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ell = ellipse_points(C, 1.0)
    ax.plot(ell[0] * 1e3, np.rad2deg(ell[1]), color="#1f77b4", lw=2, label=r"$1\sigma$")
    # principal axes
    colors_ax = ["#d62728", "#2ca02c"]
    for i, col in enumerate(colors_ax):
        v = evecs[:, i] * np.sqrt(evals[i])
        ax.plot(
            [ -v[0] * 1e3, v[0] * 1e3],
            [ -np.rad2deg(v[1]), np.rad2deg(v[1])],
            color=col,
            lw=1.6,
            label=rf"eje $k\sqrt{{\lambda_{i+1}}}$",
        )
    ax.axhline(0, color="0.8", lw=0.5)
    ax.axvline(0, color="0.8", lw=0.5)
    ax.plot(0, 0, "k.", ms=7)
    ax.set_title(r"Imagen: $\delta = V\Lambda^{1/2} u$", fontsize=11)
    ax.set_xlabel(r"$\delta\rho$  [mm]")
    ax.set_ylabel(r"$\delta\varphi$  [deg]")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.suptitle("De la esfera de Mahalanobis a la elipse de covarianza", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(HERE / "mahalanobis.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_polar_warp() -> None:
    """Same 3σ ellipse in cartesian (ρ,φ) vs drawn on polar axes."""
    C = covariance()
    pts = ellipse_points(C, k=3.0)
    rho = RHO0 + pts[0]
    phi = PHI0 + pts[1]

    fig = plt.figure(figsize=(8.6, 4.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(np.rad2deg(phi), rho, color="#1f77b4", lw=2)
    ax1.plot(np.rad2deg(PHI0), RHO0, "k.", ms=8)
    ax1.set_xlabel(r"$\varphi$  [deg]")
    ax1.set_ylabel(r"$\rho$  [m]")
    ax1.set_title("Plano cartesiano $(\\varphi,\\rho)$\n(aquí la CRB es una elipse)", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("auto")

    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    ax2.plot(phi, rho, color="#1f77b4", lw=2)
    ax2.scatter([PHI0], [RHO0], c="k", s=18, zorder=3)
    ax2.set_thetamin(70)
    ax2.set_thetamax(110)
    ax2.set_theta_zero_location("E")
    ax2.set_theta_direction(1)
    ax2.set_rlim(0.96, 1.04)
    ax2.set_title("Mismos puntos sobre ejes polares\n(el arco es solo el papel)", fontsize=10, pad=12)
    ax2.tick_params(labelsize=8)

    fig.suptitle(
        r"$3\sigma$ en $(1\,\mathrm{m},90^\circ)$: la geometría no cambia, cambia el papel",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(HERE / "polar_warp.png", dpi=180)
    plt.close(fig)


def fig_pipeline() -> None:
    """Simple boxes: g → J → I → C → ellipse. Drawn with matplotlib, no tikz needed."""
    fig, ax = plt.subplots(figsize=(8.8, 2.15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.4)
    ax.axis("off")
    boxes = [
        (0.15, r"$g(\psi)$" "\n" r"tasa Poisson"),
        (2.15, r"$J=\partial g/\partial\psi$" "\n" r"dif. centrales"),
        (4.15, r"$I=J^\top\mathrm{diag}(1/g)J$" "\n" r"Fisher"),
        (6.15, r"$C=I^{+}$" "\n" r"CRB"),
        (8.15, r"elipse $k\sigma$" "\n" r"$V,\,k\sqrt{\lambda}$"),
    ]
    for x, text in boxes:
        rect = plt.Rectangle((x, 0.25), 1.7, 0.95, facecolor="#eef4fb", edgecolor="#1f4e79", lw=1.4)
        ax.add_patch(rect)
        ax.text(x + 0.85, 0.72, text, ha="center", va="center", fontsize=8.5)
    for x in (1.85, 3.85, 5.85, 7.85):
        ax.annotate(
            "",
            xy=(x + 0.28, 0.72),
            xytext=(x, 0.72),
            arrowprops=dict(arrowstyle="-|>", color="#1f4e79", lw=1.3),
        )
    ax.set_title("Cadena que implementa compute_crb_polar  (sin mediciones ruidosas)", fontsize=11, pad=2)
    fig.tight_layout()
    fig.savefig(HERE / "pipeline.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    fig_sigma_scales()
    fig_mahalanobis()
    fig_polar_warp()
    fig_pipeline()
    print("wrote", list(HERE.glob("*.png")))
