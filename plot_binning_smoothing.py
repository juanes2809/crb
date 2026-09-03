#!/usr/bin/env python3
"""Suavizado del binning ``ceil``: pulso gaussiano (p=1) vs super-gaussiano (p=2).

Usa la geometría real de la faceta (rho=1, phi=60°, N=32, backend GPU de
``plot_simulator_components.py``) para obtener ``distance`` y ``t0=(d1+d2)/c`` de
todos los píxeles del centroide, y compara sobre la escalera real de bins
``[(j-1)Δt, jΔt]`` dos formas de repartir la energía de un píxel entre bins:

    s_p(u) ∝ exp(−(u²/2)^p),   u = (t − t0)/τ
    F_p(u) = ½ + ½·sign(u)·P(1/(2p), (u²/2)^p)        (P = gammainc regularizada)
    S_j(t0) = F_p(u_hi) − F_p(u_lo),  u_hi = (jΔt − t0)/τ,  u_lo = ((j−1)Δt − t0)/τ
    dS_j/dt0 = −(1/τ)[s_p(u_hi) − s_p(u_lo)]

Comparaciones con el mismo τ y con la misma desviación estándar
(std_p = τ·sqrt(2Γ(3/(2p))/Γ(1/(2p))); p=2 → 0.822τ).

Salidas en ``gpu_vs_cpu/``: binning_smooth_overlay.png, pulse_gauss_vs_supergauss.png,
pulse_comparison.txt.  Además añade/actualiza la sección
"Suavizado del binning: gaussiana vs super-gaussiana" de ``gpu_vs_cpu/README.md``.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.special import erf, gamma, gammainc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import plot_simulator_components as psc  # noqa: E402

OUT = ROOT / "gpu_vs_cpu"
DPI = 200
DT = psc.bin_size  # 3.9e-10 s
C_LIGHT = psc.c

# τ de prueba (en unidades de Δt) y etiquetas
TAUS = {"Δt/√12": 1 / np.sqrt(12), "Δt/3": 1 / 3, "Δt": 1.0}
TAU_BASE = "Δt/√12"


# --------------------------------------------------------------------------
# Pulso de orden p (todo en unidades adimensionales: x = t/Δt, τ' = τ/Δt)
# --------------------------------------------------------------------------
def norm_const(p):
    """Z_p = ∫ exp(−(u²/2)^p) du = (√2/p)·Γ(1/(2p))."""
    return np.sqrt(2.0) / p * gamma(1.0 / (2 * p))


def std_u(p):
    """Desviación estándar de s_p en unidades de τ: sqrt(2Γ(3/(2p))/Γ(1/(2p)))."""
    return np.sqrt(2.0 * gamma(3.0 / (2 * p)) / gamma(1.0 / (2 * p)))


def density_u(u, p):
    """s_p(u) normalizada (∫ s_p du = 1)."""
    u = np.asarray(u, dtype=float)
    return np.exp(-((u * u / 2.0) ** p)) / norm_const(p)


def cdf_u(u, p):
    """F_p(u) = ½ + ½·sign(u)·P(1/(2p), (u²/2)^p)."""
    u = np.asarray(u, dtype=float)
    return 0.5 + 0.5 * np.sign(u) * gammainc(1.0 / (2 * p), (u * u / 2.0) ** p)


def bin_weights(x, j, tau, p):
    """S_j(x) para x = t0/Δt (array), bin j (escalar o array broadcastable), τ' = τ/Δt."""
    u_hi = (j - x) / tau
    u_lo = (j - 1 - x) / tau
    return cdf_u(u_hi, p) - cdf_u(u_lo, p)


def bin_weight_deriv(x, j, tau, p):
    """dS_j/dx en unidades de 1/Δt: −(1/τ')[s_p(u_hi) − s_p(u_lo)]."""
    u_hi = (j - x) / tau
    u_lo = (j - 1 - x) / tau
    return -(density_u(u_hi, p) - density_u(u_lo, p)) / tau


def hard_box(x, j):
    """h_j(x) = 1{(j−1) < x ≤ j}  (convención ceil)."""
    return ((x > j - 1) & (x <= j)).astype(float)


def pulse_variants(tau, p_gauss=1, p_sg=2):
    """Tres variantes para un τ base: gaussiana, SG mismo τ, SG misma std."""
    return [
        ("gaussiana (p=1)", p_gauss, tau, "-", "C0"),
        ("super-gauss p=2, mismo τ", p_sg, tau, "--", "C3"),
        ("super-gauss p=2, misma std", p_sg, tau * std_u(p_gauss) / std_u(p_sg), ":", "C2"),
    ]


# --------------------------------------------------------------------------
# Métricas de aproximación a la caja
# --------------------------------------------------------------------------
def box_metrics(tau, p, j=23, half_width=8.0, n=200001):
    """L1 = ∫|S_j − h_j| dx  (en Δt),  L∞ = max|S_j − h_j|,  max|dS_j/dx| (1/Δt)."""
    x = np.linspace(j - 0.5 - half_width, j - 0.5 + half_width, n)
    S = bin_weights(x, j, tau, p)
    h = hard_box(x, j)
    dS = bin_weight_deriv(x, j, tau, p)
    dx = x[1] - x[0]
    return dict(L1=float(np.sum(np.abs(S - h)) * dx), Linf=float(np.abs(S - h).max()),
                dmax=float(np.abs(dS).max()), std=tau * std_u(p))


def mass_conservation(x, tau, p, num_bins):
    js = np.arange(1, num_bins + 1)
    S = bin_weights(x[:, None], js[None, :], tau, p)
    return S.sum(axis=1)


def mean_bin(x, tau, p, num_bins):
    js = np.arange(1, num_bins + 1)
    S = bin_weights(x[:, None], js[None, :], tau, p)
    return (S * js[None, :]).sum(axis=1)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    # ---- geometría real (centroide de la faceta, backend GPU) ----
    E = psc.evaluate_backend("gpu", run_transient=False)
    T = E["T"]
    num_bins = E["num_bins"]
    dist = np.sort(T["distance"])
    x_pix = dist / (C_LIGHT * DT)  # t0/Δt de cada píxel
    j_center = 23
    log("=" * 84)
    log("Suavizado del binning ceil — gaussiana (p=1) vs super-gaussiana (p=2)")
    log("=" * 84)
    log(f"Geometría: faceta rho={psc.RHO}, phi={psc.PHI_DEG:.0f}°, N={psc.cam_pixel_dim}, centroide = "
        f"{np.round(E['center'], 4)}; Δt = {DT:.2e} s (cΔt = {C_LIGHT*DT*100:.2f} cm); T = {num_bins} bins")
    log(f"distance de los {x_pix.size} píxeles: {dist.min():.4f}…{dist.max():.4f} m  ->  t0/Δt = "
        f"{x_pix.min():.3f}…{x_pix.max():.3f}  ->  bins duros {int(np.ceil(x_pix.min()))}…{int(np.ceil(x_pix.max()))}; "
        f"bin central j = {j_center}")
    log(f"Constantes: Z_1 = {norm_const(1):.6f} (√(2π) = {np.sqrt(2*np.pi):.6f}), Z_2 = {norm_const(2):.6f}; "
        f"std/τ: p=1 -> {std_u(1):.6f}, p=2 -> {std_u(2):.6f}")

    # ---- verificaciones ----
    log("\n" + "-" * 84)
    log("Verificaciones")
    log("-" * 84)
    u = np.linspace(-8, 8, 20001)
    err_erf = np.abs(cdf_u(u, 1) - 0.5 * (1 + erf(u / np.sqrt(2)))).max()
    err_P = np.abs(gammainc(0.5, u * u / 2) - erf(np.abs(u) / np.sqrt(2))).max()
    log(f"(1) p=1 vía gammainc vs erf:  max|F_1 − ½[1+erf(u/√2)]| = {err_erf:.2e};  "
        f"max|P(½,u²/2) − erf(|u|/√2)| = {err_P:.2e}  -> {'OK' if err_erf < 1e-12 else 'FALLA'} (<1e-12)")
    for p in (1, 2):
        du = u[1] - u[0]
        log(f"    ∫ s_{p} du (trapecio) = {np.trapezoid(density_u(u, p), u):.12f};  "
            f"std numérica/τ = {np.sqrt(np.trapezoid(u*u*density_u(u, p), u)):.6f} (teoría {std_u(p):.6f});  "
            f"max|F_{p}' − s_{p}| = {np.abs(np.gradient(cdf_u(u, p), du) - density_u(u, p)).max():.1e}")
    x_dense = np.linspace(x_pix.min() - 1, x_pix.max() + 1, 4001)
    worst = 0.0
    for tau_lbl, tau in TAUS.items():
        for name, p, tau_p, _, _ in pulse_variants(tau):
            m = mass_conservation(x_dense, tau_p, p, num_bins)
            worst = max(worst, np.abs(m - 1).max())
    log(f"(2) Σ_j S_j(t0) sobre j=1…{num_bins}, t0/Δt ∈ [{x_dense.min():.2f}, {x_dense.max():.2f}], ambos pulsos, "
        f"3 τ: max|Σ − 1| = {worst:.2e} -> {'OK' if worst < 1e-10 else 'FALLA'} (<1e-10)")
    log("(3) τ → 0: L1(S_j, h_j) debe → 0")
    for tau_small in (1 / np.sqrt(12), 1 / 10, 1 / 50, 1 / 200):
        mg = box_metrics(tau_small, 1)
        ms = box_metrics(tau_small, 2)
        log(f"    τ = Δt/{1/tau_small:6.2f}: L1 gauss = {mg['L1']:.5f} Δt, L1 SG(mismo τ) = {ms['L1']:.5f} Δt   "
            f"(teoría gauss 2τ√(2/π) = {2*tau_small*np.sqrt(2/np.pi):.5f})")

    # ======================================================================
    # Prueba 1: la versión suave sobre la escalera real
    # ======================================================================
    tau0 = TAUS[TAU_BASE]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))

    # (1) escalera vs bin promedio
    ax = axs[0]
    hard = np.ceil(x_pix)
    ax.step(dist, hard, where="post", color="C3", lw=2.0, label="ceil(t0/Δt)  (escalera dura)")
    ax.plot(dist, x_pix, color="k", lw=0.8, label="t0/Δt  (recta continua)")
    ax.plot(dist, x_pix + 0.5, color="k", lw=0.8, ls="--", label="t0/Δt + ½  (diagonal centrada del ceil)")
    rms = {}
    rms_hard = float(np.sqrt(np.mean((hard - (x_pix + 0.5)) ** 2)))
    for name, p, tau_p, ls, col in pulse_variants(tau0):
        mb = mean_bin(x_pix, tau_p, p, num_bins)
        rms[name] = float(np.sqrt(np.mean((mb - (x_pix + 0.5)) ** 2)))
        ax.plot(dist, mb, ls=ls, color=col, lw=1.8,
                label=f"Σ_j j·S_j  {name}, τ={tau_p:.3f}Δt  (RMS vs diag. centrada {rms[name]:.4f})")
    ax.set_xlabel("distance = d1 + d2 [m]  (píxeles reales ordenados)")
    ax.set_ylabel("bin")
    ax.set_title(f"Escalera real vs bin promedio ponderado por energía (τ base = {TAU_BASE})\n"
                 f"RMS escalera dura vs diagonal centrada = {rms_hard:.4f} (teoría 1/√12 = {1/np.sqrt(12):.4f})", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="upper left")

    # (2) energía en el bin central
    ax = axs[1]
    xd = np.linspace(x_pix.min(), x_pix.max(), 3001)
    dd = xd * C_LIGHT * DT
    ax.plot(dd, hard_box(xd, j_center), color="k", lw=2.0, label=f"h_{j_center}(t0)  caja dura")
    for name, p, tau_p, ls, col in pulse_variants(tau0):
        ax.plot(dd, bin_weights(xd, j_center, tau_p, p), ls=ls, color=col, lw=1.8,
                label=f"S_{j_center}(t0)  {name}, τ={tau_p:.3f}Δt")
    for lbl_pos, s in ((x_pix.min(), "bin 22"), ((j_center - 0.5), "bin 23"), ((j_center + 0.5), "bin 24")):
        ax.text(lbl_pos * C_LIGHT * DT, 1.04, s, fontsize=7, ha="center")
    ax.plot(dist, np.zeros_like(dist) - 0.03, "|", color="0.5", ms=4, label="t0 de los píxeles reales")
    ax.set_ylim(-0.08, 1.12)
    ax.set_xlabel("distance = d1 + d2 [m]")
    ax.set_ylabel(f"energía depositada en el bin {j_center}")
    ax.set_title(f"Energía en el bin central {j_center}: caja dura vs S_j suave", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="upper right")

    # (3) derivada
    ax = axs[2]
    for name, p, tau_p, ls, col in pulse_variants(tau0):
        dS = bin_weight_deriv(xd, j_center, tau_p, p)
        ax.plot(dd, dS, ls=ls, color=col, lw=1.8, label=f"dS_{j_center}/dt0  {name}  (max|·| = {np.abs(dS).max():.3f}/Δt)")
    ymax = max(np.abs(bin_weight_deriv(xd, j_center, tau_p, p)).max() for _, p, tau_p, _, _ in pulse_variants(tau0))
    for edge, sgn in (((j_center - 1) * C_LIGHT * DT, +1), (j_center * C_LIGHT * DT, -1)):
        ax.annotate("", xy=(edge, sgn * ymax * 1.25), xytext=(edge, 0),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=2))
        ax.text(edge, sgn * ymax * 1.32, f"{'+' if sgn > 0 else '−'}δ(t0−{'22' if sgn > 0 else '23'}Δt)", ha="center",
                va="bottom" if sgn > 0 else "top", fontsize=7)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_ylim(-ymax * 1.6, ymax * 1.6)
    ax.set_xlabel("distance = d1 + d2 [m]")
    ax.set_ylabel(f"dS_{j_center}/dt0  [1/Δt]")
    ax.set_title(f"Derivada respecto a t0 (la caja dura sólo tiene deltas ±∞ en los bordes)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, loc="lower left")
    fig.suptitle(f"Suavizado del binning ceil sobre la escalera real — faceta ρ={psc.RHO:.1f} m, φ={psc.PHI_DEG:.0f}°, "
                 f"N={psc.cam_pixel_dim}, Δt={DT:.1e} s", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "binning_smooth_overlay.png", dpi=DPI)
    plt.close(fig)

    log("\n" + "-" * 84)
    log(f"Prueba 1 — bin promedio Σ_j j·S_j(t0) sobre los píxeles reales (τ base = {TAU_BASE})")
    log("-" * 84)
    log(f"  RMS(escalera dura − (t0/Δt+½)) = {rms_hard:.4f}   (uniforme: 1/√12 = {1/np.sqrt(12):.4f})")
    log(f"  RMS(escalera dura − t0/Δt)     = {float(np.sqrt(np.mean((hard - x_pix)**2))):.4f}   (offset ½ del ceil)")
    for name, v in rms.items():
        log(f"  RMS({name:28s} − (t0/Δt+½)) = {v:.4f}   ({100*(1-v/rms_hard):.1f} % menos rizado que la escalera)")
    log("  Nota: la referencia es la diagonal centrada t0/Δt+½ porque ceil deposita en el bin j la energía de "
        "t0∈((j−1)Δt, jΔt]; la recta t0/Δt está ½ bin por debajo por construcción.")

    # ======================================================================
    # Prueba 2: gaussiana vs super-gaussiana cuantificado
    # ======================================================================
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    # (a) formas de los pulsos
    ax = fig.add_subplot(gs[0, 0])
    xx = np.linspace(-1.6, 1.6, 2001)
    ax.fill_between([-0.5, 0.5], 0, 1.0, color="0.85", label="caja de un bin (ancho Δt, área 1)")
    for name, p, tau_p, ls, col in pulse_variants(tau0):
        ax.plot(xx, density_u(xx / tau_p, p) / tau_p, ls=ls, color=col, lw=1.8,
                label=f"s_{p}: {name}, τ={tau_p:.3f}Δt, std={tau_p*std_u(p):.3f}Δt, pico={density_u(0,p)/tau_p:.3f}/Δt")
    ax.set_xlabel("(t − t0)/Δt"); ax.set_ylabel("densidad [1/Δt]")
    ax.set_title(f"(a) Formas de los pulsos (τ base = {TAU_BASE} = {tau0:.3f}Δt)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left")

    # (b) S_j vs caja para los tres τ
    sub = gs[0, 1].subgridspec(3, 1, hspace=0.08)
    metrics = []  # filas de la tabla
    xb = np.linspace(j_center - 3.0, j_center + 2.0, 5001)
    for k, (tau_lbl, tau) in enumerate(TAUS.items()):
        ax = fig.add_subplot(sub[k, 0])
        ax.plot(xb - (j_center - 0.5), hard_box(xb, j_center), color="k", lw=1.6, label="caja h_j" if k == 0 else None)
        for name, p, tau_p, ls, col in pulse_variants(tau):
            crit = "—" if p == 1 else ("mismo τ" if abs(tau_p - tau) < 1e-12 else "misma std")
            m = box_metrics(tau_p, p, j=j_center)
            mass = mass_conservation(x_dense, tau_p, p, num_bins)
            metrics.append(dict(pulso=f"p={p}", tau_lbl=tau_lbl, crit=crit, tau=tau_p, **m,
                                mass_min=float(mass.min()), mass_max=float(mass.max())))
            ax.plot(xb - (j_center - 0.5), bin_weights(xb, j_center, tau_p, p), ls=ls, color=col, lw=1.5,
                    label=f"{name}: L1={m['L1']:.3f}Δt, L∞={m['Linf']:.3f}")
        ax.text(0.01, 0.9, f"τ = {tau_lbl}", transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(fc="w", alpha=0.8, ec="0.7"))
        ax.set_ylim(-0.05, 1.12); ax.set_xlim(-2.5, 2.5); ax.grid(alpha=0.3); ax.legend(fontsize=6, loc="upper right")
        if k < 2:
            ax.set_xticklabels([])
        if k == 0:
            ax.set_title("(b) S_j(t0) vs caja h_j para los tres τ  (L1 = ∫|S_j−h_j|dt0 en Δt)", fontsize=10)
        if k == 2:
            ax.set_xlabel("(t0 − centro del bin j)/Δt")
        ax.set_ylabel("S_j")

    # (c) pico de la derivada vs τ
    ax = fig.add_subplot(gs[1, 0])
    taus_sweep = np.logspace(np.log10(1 / 50), np.log10(2.0), 120)
    for name, p, _, ls, col in pulse_variants(1.0):
        scale = 1.0 if (p == 1 or "mismo" in name) else std_u(1) / std_u(p)
        dm = np.array([box_metrics(t * scale, p, j=j_center, n=40001)["dmax"] for t in taus_sweep])
        ax.loglog(taus_sweep, dm, ls=ls, color=col, lw=1.8, label=f"max|dS_j/dt0|  {name}")
        for tau_lbl, tau in TAUS.items():
            ax.plot([tau], [box_metrics(tau * scale, p, j=j_center)["dmax"]], "o", color=col, ms=5)
    ax.loglog(taus_sweep, density_u(0, 1) / taus_sweep, color="0.6", lw=0.8, ls="-.", label="s_1(0)/τ = 0.399/τ (asíntota gauss)")
    for tau_lbl, tau in TAUS.items():
        ax.axvline(tau, color="0.8", lw=0.8)
        ax.text(tau, ax.get_ylim()[0] * 1.3, tau_lbl, fontsize=7, ha="center", rotation=90, va="bottom")
    ax.set_xlabel("τ / Δt"); ax.set_ylabel("max|dS_j/dt0|  [1/Δt]")
    ax.set_title("(c) Pico de la derivada vs τ — trade-off fidelidad a la caja ↔ suavidad", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    # (d) conservación de masa
    ax = fig.add_subplot(gs[1, 1])
    for tau_lbl, tau in TAUS.items():
        for name, p, tau_p, ls, col in pulse_variants(tau):
            mass = mass_conservation(x_dense, tau_p, p, num_bins)
            ax.plot(x_dense, mass - 1.0, ls=ls, color=col, lw=1.2, alpha=0.5 + 0.25 * list(TAUS).index(tau_lbl),
                    label=f"{name}, τ={tau_lbl}" if True else None)
    ax.set_yscale("symlog", linthresh=1e-15)
    ax.set_ylim(-1e-9, 1e-9)
    ax.set_xlabel("t0/Δt"); ax.set_ylabel("Σ_j S_j(t0) − 1")
    ax.set_title(f"(d) Conservación de masa Σ_j S_j sobre j=1…{num_bins}  (max|Σ−1| = {worst:.1e})", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=3, loc="upper center")
    fig.suptitle("Gaussiana (p=1) vs super-gaussiana (p=2) como pulso de suavizado del binning ceil", fontsize=13)
    fig.savefig(OUT / "pulse_gauss_vs_supergauss.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # ---- tabla ----
    log("\n" + "-" * 84)
    log(f"Prueba 2 — métricas de S_j (bin j={j_center}) frente a la caja h_j; derivadas en 1/Δt; std en Δt")
    log("-" * 84)
    hdr = (f"{'pulso':6s} {'τ base':8s} {'criterio':10s} {'τ_p/Δt':8s} {'std/Δt':8s} {'L1 [Δt]':9s} {'L∞':7s} "
           f"{'max|dS/dt0|':12s} {'ΣS_j min':12s} {'ΣS_j max':12s}")
    log(hdr)
    log("-" * len(hdr))
    table_md = ["| pulso | τ base | criterio | τ_p/Δt | std/Δt | L1 [Δt] | L∞ | max\\|dS_j/dt0\\| [1/Δt] | Σ_j S_j min | Σ_j S_j max |",
                "|---|---|---|---|---|---|---|---|---|---|"]
    for r in metrics:
        log(f"{r['pulso']:6s} {r['tau_lbl']:8s} {r['crit']:10s} {r['tau']:8.4f} {r['std']:8.4f} {r['L1']:9.4f} {r['Linf']:7.4f} "
            f"{r['dmax']:12.4f} {r['mass_min']:12.10f} {r['mass_max']:12.10f}")
        table_md.append(f"| {r['pulso']} | {r['tau_lbl']} | {r['crit']} | {r['tau']:.4f} | {r['std']:.4f} | {r['L1']:.4f} | "
                        f"{r['Linf']:.4f} | {r['dmax']:.4f} | {r['mass_min']:.10f} | {r['mass_max']:.10f} |")

    # ratios SG vs gauss por τ
    log("")
    ratios = []
    for tau_lbl in TAUS:
        g = next(r for r in metrics if r["tau_lbl"] == tau_lbl and r["pulso"] == "p=1")
        for crit in ("mismo τ", "misma std"):
            s = next(r for r in metrics if r["tau_lbl"] == tau_lbl and r["pulso"] == "p=2" and r["crit"] == crit)
            ratios.append((tau_lbl, crit, s["L1"] / g["L1"], s["dmax"] / g["dmax"]))
            log(f"  τ={tau_lbl:6s} SG {crit:9s}: L1_SG/L1_gauss = {s['L1']/g['L1']:.3f},  "
                f"maxderiv_SG/maxderiv_gauss = {s['dmax']/g['dmax']:.3f}")
    log("  Analítico: max|dS_j/dt0| ≈ s_p(0)/τ_p para τ ≪ Δt -> gauss 0.3989/τ, SG 0.3901/τ (mismo τ), "
        f"SG {density_u(0,2)*std_u(2)/std_u(1):.4f}/τ (misma std).")
    log("  L∞ ≈ ½ para todo τ ≪ Δt: en el borde exacto de la caja cualquier pulso simétrico deposita ½; "
        "L∞ no discrimina, L1 sí.")

    # ---- conclusión ----
    mad = {p: np.trapezoid(np.abs(u) * density_u(u, p), u) / std_u(p) for p in (1, 2)}
    r_same_tau_small = [r[2] for r in ratios if r[1] == "mismo τ" and r[0] != "Δt"]
    r_same_std = [r[2] for r in ratios if r[1] == "misma std"]
    d_same_tau_small = [r[3] for r in ratios if r[1] == "mismo τ" and r[0] != "Δt"]
    d_same_std_small = [r[3] for r in ratios if r[1] == "misma std" and r[0] != "Δt"]
    d_same_tau_dt = next(r[3] for r in ratios if r[1] == "mismo τ" and r[0] == "Δt")
    d_same_std_dt = next(r[3] for r in ratios if r[1] == "misma std" and r[0] == "Δt")
    log(f"  E|u|/std (desviación media absoluta por unidad de std): gauss {mad[1]:.4f}, SG p=2 {mad[2]:.4f}")
    concl = [
        "Conclusión",
        "----------",
        "1. ¿Aproxima mejor la caja la super-gaussiana?  Sólo en apariencia.  Con el MISMO τ, L1 baja "
        f"{100*(1-np.mean(r_same_tau_small)):.0f} % (τ ≤ Δt/3) — pero únicamente porque su std es 0.822τ: es un pulso más "
        f"estrecho, no uno 'más caja'.  Con la MISMA std, L1 es PEOR en {100*(np.mean(r_same_std)-1):.0f} % en los tres τ: "
        f"para τ ≪ Δt, L1 ≈ 2·E|t−t0| y una densidad de cima plana tiene más desviación media absoluta por unidad de std "
        f"(E|u|/std = {mad[2]:.3f} frente a {mad[1]:.3f} de la gaussiana).  'Elevar al cuadrado' no acerca S_j a la caja a "
        "igualdad de anchura; el ancho de la transición lo fija la std, no el exponente.  Donde la SG (misma std) SÍ gana es en "
        f"el bin promedio Σ_j j·S_j: su rizado respecto a la diagonal es {rms['super-gauss p=2, misma std']:.4f} frente a "
        f"{rms['gaussiana (p=1)']:.4f} de la gaussiana (la escalera se 'derrite' de forma más uniforme porque la función "
        "característica de la SG es menor en la frecuencia 1/Δt).",
        "2. ¿A qué costo en la derivada?  Para τ ≤ Δt/3 el pico de dS_j/dt0 es s_p(0)/τ_p y la SG NO es más picuda: "
        f"{np.mean(d_same_tau_small):.2f}× la gaussiana con el mismo τ y {np.mean(d_same_std_small):.2f}× con la misma std "
        "(la cima plana reparte la misma área con menor máximo).  Para τ = Δt, cuando los dos bordes del bin caen en los flancos, "
        f"los flancos empinados sí se notan: {d_same_tau_dt:.2f}× (mismo τ) y {d_same_std_dt:.2f}× (misma std).  El costo real "
        "está en la FORMA de la derivada: una meseta que cae como exp(−u⁴/4), con gradiente ≈0 fuera de |u|≳2 (zonas muertas "
        "más anchas y de borde más abrupto) y segunda derivada mayor; para un optimizador o para un CRB evaluado con t0 en "
        "medio de un bin y τ pequeño, la información de Fisher queda confinada a ventanas más estrechas y el Hessiano peor "
        "condicionado.",
        "3. Recomendación para el CRB: la GAUSSIANA (p=1).  La caja no es física, es el artefacto numérico de ceil(); lo que "
        "hay que modelar es la respuesta temporal real (pulso láser de ps + jitter del SPAD + electrónica), que es "
        "aproximadamente gaussiana (con una cola exponencial de difusión en SPADs reales), no una caja.  La gaussiana (i) es "
        "la única con interpretación física directa (τ = σ del IRF, o Δt/√12 si sólo se quiere reproducir la varianza de la "
        "cuantización), (ii) tiene derivada suave, sin mesetas ni zonas muertas, con Fisher bien condicionado en todo t0, y "
        "(iii) la super-gaussiana sólo compra parecerse más al artefacto.  Si el objetivo fuera reproducir exactamente la "
        "caja, el límite p→∞ es un pulso uniforme y S_j se vuelve un trapecio: se recupera la derivada discontinua que se "
        "quería evitar.",
    ]
    log("\n" + "\n".join(concl))

    (OUT / "pulse_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- README ----
    update_readme(table_md, rms, rms_hard, err_erf, worst, concl, ratios)
    print(f"\nEscrito {OUT}/binning_smooth_overlay.png, pulse_gauss_vs_supergauss.png, pulse_comparison.txt y README.md")


def update_readme(table_md, rms, rms_hard, err_erf, worst, concl, ratios):
    path = OUT / "README.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    header = "## Suavizado del binning: gaussiana vs super-gaussiana"
    section = [header, "",
               "Script: `plot_binning_smoothing.py`.  Pulso de orden p, `s_p(u) ∝ exp(−(u²/2)^p)`, `u=(t−t0)/τ`; peso por bin "
               "`S_j = F_p(u_hi) − F_p(u_lo)` con `F_p(u) = ½ + ½·sign(u)·P(1/(2p), (u²/2)^p)`; p=1 es la gaussiana actual "
               "(`P(½,u²/2) = erf(|u|/√2)`), p=2 la super-gaussiana de cima plana.  Comparación con el mismo τ y con la misma "
               "std (`std_2 = 0.822·τ_2`).  Geometría real de la faceta (ρ=1, φ=60°, N=32), bin central j=23.",
               "",
               f"Verificaciones: p=1 vía `gammainc` vs `erf`: max|Δ| = {err_erf:.1e}; Σ_j S_j = 1 con max|Σ−1| = {worst:.1e} "
               "para ambos pulsos y los tres τ; con τ→0 (Δt/50, Δt/200) L1→0 para ambos (ver `pulse_comparison.txt`).",
               "",
               f"**Bin promedio** Σ_j j·S_j(t0) sobre los píxeles reales (τ = Δt/√12), RMS frente a la diagonal centrada t0/Δt+½: "
               f"escalera dura {rms_hard:.4f} (1/√12); " + "; ".join(f"{k} {v:.4f}" for k, v in rms.items()) + ".",
               "", *table_md, "",
               "Cocientes SG/gauss: " + "; ".join(f"τ={a}, {b}: L1 ×{c:.3f}, max|dS/dt0| ×{d:.3f}" for a, b, c, d in ratios) + ".",
               "",
               "Figuras: `binning_smooth_overlay.png` (escalera real vs bin promedio; energía en el bin 23; derivada) y "
               "`pulse_gauss_vs_supergauss.png` (formas; S_j vs caja para los tres τ; pico de la derivada vs τ; conservación de masa).",
               "", *[c for c in concl[2:]], ""]
    block = "\n".join(section)
    if header in text:
        pattern = re.compile(re.escape(header) + r".*?(?=\n## |\Z)", re.S)
        text = pattern.sub(lambda _: block.rstrip("\n") + "\n", text)
    else:
        text = text.rstrip("\n") + "\n\n" + block
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
