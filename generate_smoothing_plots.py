#!/usr/bin/env python3
"""Side-by-side plots of each NON-differentiable operation of the simulator's
forward (left) and its smooth C-infinity replacement (right).

Three pairs, matching the three sources of non-differentiability of the
simulator's two-bounce intensity (see docs/analytic_forward_crb.tex,
"Construccion de una version diferenciable"):

  1. Occlusion       Heaviside 1{x>0}   ->  sigmoid(kappa * x)
  2. Foreshortening  max(0, x)          ->  softplus_beta(x)
  3. Temporal binning box 1{t0 in bin}  ->  1/2 [erf(...) - erf(...)]  (erf diff)

Each figure has two panels: left = hard (non-differentiable), right = smooth
replacement with the hard version overlaid as a dotted grey reference so the
convergence hard <- smooth is visible. Smoothing parameters (kappa, beta, tau,
bin_size) are taken from the REAL ForwardConfig used by the analytic forward.

Run:  python3 generate_smoothing_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from crb_analytic_forward import C_LIGHT, ForwardConfig

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

DPI = 200
FS_TITLE = 10
FS_LABEL = 9
FS_TICK = 8
FS_LEG = 8

cfg = ForwardConfig()


def _style(ax) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=FS_TICK)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)


# ---------------------------------------------------------------------------
# 1) Occlusion: Heaviside vs sigmoid
# ---------------------------------------------------------------------------
def plot_occlusion() -> Path:
    kappa = cfg.kappa
    x = np.linspace(-0.1, 0.1, 2000)
    hard = np.where(x > 0.0, 1.0, 0.0)
    soft = 1.0 / (1.0 + np.exp(-kappa * x))
    soft_hi = 1.0 / (1.0 + np.exp(-200.0 * x))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.4))

    # left: hard Heaviside
    axL.step(x, hard, where="post", color="C3", linewidth=1.8,
             label=r"$H(d_{edge})=\mathbf{1}\{d_{edge}>0\}$")
    axL.plot(0.0, 0.5, "o", color="C3", markersize=6, zorder=5)
    axL.annotate("no derivable\n(salto en 0)", xy=(0.0, 0.5),
                 xytext=(0.012, 0.55), fontsize=FS_LEG, color="C3",
                 arrowprops=dict(arrowstyle="->", color="C3", lw=0.8))
    axL.set_title("Oclusion (dura): Heaviside", fontsize=FS_TITLE)
    axL.set_xlabel(r"$d_{edge}$  (coordenada de cruce, m)", fontsize=FS_LABEL)
    axL.set_ylabel("visibilidad", fontsize=FS_LABEL)
    axL.set_ylim(-0.1, 1.15)
    _style(axL)
    axL.legend(fontsize=FS_LEG, loc="upper left")

    # right: sigmoid
    axR.plot(x, hard, ":", color="grey", linewidth=1.2,
             label="Heaviside (referencia)")
    axR.plot(x, soft, color="C0", linewidth=2.0,
             label=rf"$\sigma(\kappa\, d_{{edge}})$,  $\kappa={kappa:.0f}$")
    axR.plot(x, soft_hi, "--", color="C2", linewidth=1.3,
             label=r"$\kappa=200$ ($\to$ Heaviside)")
    axR.set_title(rf"Oclusion (suave): sigmoide  ($\kappa={kappa:.0f}$, "
                  rf"penumbra $\sim1/\kappa={1.0/kappa*100:.1f}$ cm)",
                  fontsize=FS_TITLE)
    axR.set_xlabel(r"$d_{edge}$  (coordenada de cruce, m)", fontsize=FS_LABEL)
    axR.set_ylabel("visibilidad (derivable, suave)", fontsize=FS_LABEL)
    axR.set_ylim(-0.1, 1.15)
    _style(axR)
    axR.legend(fontsize=FS_LEG, loc="upper left")

    fig.suptitle("Oclusion: umbral duro $\\to$ sigmoide $C^\\infty$",
                 fontsize=FS_TITLE + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = PLOTS / "crb_smoothing_occlusion.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2) Foreshortening: max(0, .) vs softplus
# ---------------------------------------------------------------------------
def plot_foreshortening() -> Path:
    beta = cfg.beta
    x = np.linspace(-0.2, 0.2, 2000)
    hard = np.maximum(0.0, x)
    soft = np.log1p(np.exp(beta * x)) / beta
    soft_hi = np.log1p(np.exp(200.0 * x)) / 200.0
    bound = np.log(2.0) / beta  # uniform gap at x=0

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.4))

    # left: hard relu
    axL.plot(x, hard, color="C3", linewidth=1.8, label=r"$\max(0,x)$")
    axL.plot(0.0, 0.0, "o", color="C3", markersize=6, zorder=5)
    axL.annotate("no derivable\n(codo en 0)", xy=(0.0, 0.0),
                 xytext=(0.03, 0.05), fontsize=FS_LEG, color="C3",
                 arrowprops=dict(arrowstyle="->", color="C3", lw=0.8))
    axL.set_title("Foreshortening (dura): $\\max(0,\\cdot)$", fontsize=FS_TITLE)
    axL.set_xlabel(r"valor del coseno $c_k$", fontsize=FS_LABEL)
    axL.set_ylabel(r"$[c_k]_+$", fontsize=FS_LABEL)
    axL.set_ylim(-0.03, 0.21)
    _style(axL)
    axL.legend(fontsize=FS_LEG, loc="upper left")

    # right: softplus
    axR.plot(x, hard, ":", color="grey", linewidth=1.2,
             label=r"$\max(0,x)$ (referencia)")
    axR.plot(x, soft, color="C0", linewidth=2.0,
             label=rf"softplus$_\beta$,  $\beta={beta:.0f}$")
    axR.plot(x, soft_hi, "--", color="C2", linewidth=1.3,
             label=r"$\beta=200$ ($\to\max(0,x)$)")
    # annotate the uniform bound (log2)/beta at x=0
    axR.annotate(rf"$\frac{{\log 2}}{{\beta}}={bound:.4f}$",
                 xy=(0.0, bound), xytext=(0.03, bound + 0.03),
                 fontsize=FS_LEG, color="C0",
                 arrowprops=dict(arrowstyle="->", color="C0", lw=0.8))
    axR.set_title(rf"Foreshortening (suave): softplus "
                  rf"($\beta={beta:.0f}$, cota $(\log2)/\beta={bound:.4f}$)",
                  fontsize=FS_TITLE)
    axR.set_xlabel(r"valor del coseno $c_k$", fontsize=FS_LABEL)
    axR.set_ylabel(r"softplus$_\beta(c_k)$ (derivable, suave)", fontsize=FS_LABEL)
    axR.set_ylim(-0.03, 0.21)
    _style(axR)
    axR.legend(fontsize=FS_LEG, loc="upper left")

    fig.suptitle("Foreshortening: $\\max(0,\\cdot)$ $\\to$ softplus $C^\\infty$",
                 fontsize=FS_TITLE + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = PLOTS / "crb_smoothing_foreshortening.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3) Temporal binning: box (ceil / Dirac-in-bin) vs erf difference
# ---------------------------------------------------------------------------
def plot_binning() -> Path:
    dt = cfg.bin_size      # Delta t (s)
    tau = cfg.tau          # Gaussian pulse std (s)
    ns = 1.0e9             # s -> ns

    # two adjacent bins j and j+1, edges at 2*dt, 3*dt, 4*dt
    tlo_j, thi_j = 2.0 * dt, 3.0 * dt
    tlo_j1, thi_j1 = 3.0 * dt, 4.0 * dt

    # sweep the time-of-flight t0 across ~3 bins
    t0 = np.linspace(1.5 * dt, 4.5 * dt, 3000)

    def box(t, lo, hi):
        return np.where((t >= lo) & (t <= hi), 1.0, 0.0)

    def erf_bin(t, lo, hi):
        from scipy.special import erf
        return 0.5 * (erf((hi - t) / (np.sqrt(2.0) * tau))
                      - erf((lo - t) / (np.sqrt(2.0) * tau)))

    box_j = box(t0, tlo_j, thi_j)
    box_j1 = box(t0, tlo_j1, thi_j1)
    soft_j = erf_bin(t0, tlo_j, thi_j)
    soft_j1 = erf_bin(t0, tlo_j1, thi_j1)

    x = t0 * ns
    edges = np.array([tlo_j, thi_j, thi_j1]) * ns

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.4))

    # left: hard boxes (energy jumps from bin j to j+1 at the shared edge)
    axL.plot(x, box_j, color="C3", linewidth=1.8, label=r"bin $j$: $\mathbf{1}\{t_0\in[t^{(j)}_{lo},t^{(j)}_{hi}]\}$")
    axL.plot(x, box_j1, color="C1", linewidth=1.8, label=r"bin $j{+}1$")
    for e in edges:
        axL.axvline(e, color="grey", linewidth=0.7, alpha=0.6)
    axL.annotate("salto de energia\nentre bins",
                 xy=(thi_j * ns, 0.5), xytext=(thi_j * ns + 0.06, 0.62),
                 fontsize=FS_LEG, color="C3",
                 arrowprops=dict(arrowstyle="->", color="C3", lw=0.8))
    axL.set_title("Binning (dura): caja por bin (ceil / Dirac)", fontsize=FS_TITLE)
    axL.set_xlabel(r"tiempo de vuelo $t_0$  (ns)", fontsize=FS_LABEL)
    axL.set_ylabel(r"energia en el bin", fontsize=FS_LABEL)
    axL.set_ylim(-0.08, 1.18)
    _style(axL)
    axL.legend(fontsize=FS_LEG, loc="upper right")

    # right: erf-difference (energy transfers gradually between bins)
    axR.plot(x, box_j, ":", color="grey", linewidth=1.1)
    axR.plot(x, box_j1, ":", color="grey", linewidth=1.1,
             label="cajas duras (referencia)")
    axR.plot(x, soft_j, color="C0", linewidth=2.0, label=r"$S_j(t_0)$ (erf)")
    axR.plot(x, soft_j1, color="C4", linewidth=2.0, label=r"$S_{j+1}(t_0)$ (erf)")
    for e in edges:
        axR.axvline(e, color="grey", linewidth=0.7, alpha=0.6)
    axR.annotate("traspaso gradual\n(solapamiento)",
                 xy=(thi_j * ns, 0.5), xytext=(thi_j * ns + 0.055, 0.66),
                 fontsize=FS_LEG, color="C0",
                 arrowprops=dict(arrowstyle="->", color="C0", lw=0.8))
    axR.set_title(rf"Binning (suave): diferencia de erf  "
                  rf"($\tau={tau*ns:.3f}$ ns, $\Delta t={dt*ns:.3f}$ ns)",
                  fontsize=FS_TITLE)
    axR.set_xlabel(r"tiempo de vuelo $t_0$  (ns)", fontsize=FS_LABEL)
    axR.set_ylabel(r"$S_j(t_0)$ (derivable, suave)", fontsize=FS_LABEL)
    axR.set_ylim(-0.08, 1.18)
    _style(axR)
    axR.legend(fontsize=FS_LEG, loc="upper right")

    fig.suptitle("Binning temporal (pieza central): caja dura $\\to$ "
                 "diferencia de $\\mathrm{erf}$ $C^\\infty$",
                 fontsize=FS_TITLE + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = PLOTS / "crb_smoothing_binning.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def main() -> None:
    print(f"Smoothing params: beta={cfg.beta}, kappa={cfg.kappa}, "
          f"tau={cfg.tau:.3e}s, bin_size={cfg.bin_size:.3e}s")
    for fn in (plot_occlusion, plot_foreshortening, plot_binning):
        out = fn()
        print(f"Saved {out}  ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
