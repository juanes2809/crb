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
    soft_hi = 1.0 / (1.0 + np.exp(-1000.0 * x))

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
             label=r"$\kappa=1000$ ($\to$ Heaviside)")
    axR.set_title(rf"Oclusion (suave): sigmoide  ($\kappa={kappa:.0f}=1/$pixel, "
                  rf"penumbra $\sim1/\kappa={1.0/kappa*100:.2f}$ cm)",
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
# 3) Temporal binning: box (ceil / Dirac-in-bin) vs erf difference.
#    Richer 3-panel figure: (a) convergence to the hard box for several tau,
#    (b) sum over bins (mass conservation ~1), (c) the (bounded, sharpening)
#    derivative dS_j/dt0.
# ---------------------------------------------------------------------------
def plot_binning() -> Path:
    from scipy.special import erf

    dt = cfg.bin_size                       # Delta t (s)
    taus = [dt, dt / 3.0, dt / 8.0, dt / 20.0]   # pulse-width sweep
    tau_labels = [r"\Delta t", r"\Delta t/3", r"\Delta t/8", r"\Delta t/20"]
    tau_colors = ["C0", "C2", "C1", "C4"]

    # bin grid (in units of dt): ~11 bins, focus on two adjacent bins j, j+1
    # around the shared edge at 0.  Bin k spans [k*dt, (k+1)*dt].
    k_lo, k_hi = -5, 5                       # 11 bins: [-5,-4],...,[4,5]  (units dt)
    edges_u = np.arange(k_lo, k_hi + 2)      # bin edges in units of dt
    # the two focus bins share the edge at u=0:  bin j = [-1,0], bin j+1 = [0,1]
    jlo, jhi = -1.0, 0.0
    j1lo, j1hi = 0.0, 1.0

    # x axis: time of flight t0 in units of dt, from -1.5 to +2.5 around the edge
    u = np.linspace(-1.5, 2.5, 4000)         # t0 / dt

    def erf_bin_u(uu, lo, hi, tau_over_dt):
        # S_j as a function of t0/dt, with bin edges lo,hi in units of dt
        s2 = np.sqrt(2.0) * tau_over_dt
        return 0.5 * (erf((hi - uu) / s2) - erf((lo - uu) / s2))

    def box_u(uu, lo, hi):
        return np.where((uu >= lo) & (uu <= hi), 1.0, 0.0)

    box_j = box_u(u, jlo, jhi)
    box_j1 = box_u(u, j1lo, j1hi)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axA, axB, axC = axes

    # ---- (a) convergence to the hard box for several tau --------------------
    axA.fill_between(u, box_j, step=None, color="black", alpha=0.05)
    axA.plot(u, box_j, ":", color="black", linewidth=1.4,
             label=r"caja dura bin $j$ / $j{+}1$")
    axA.plot(u, box_j1, ":", color="black", linewidth=1.4)
    for tau, tl, tc in zip(taus, tau_labels, tau_colors):
        r = tau / dt
        axA.plot(u, erf_bin_u(u, jlo, jhi, r), color=tc, linewidth=1.7,
                 label=rf"$S_j,\ \tau={tl}$")
        axA.plot(u, erf_bin_u(u, j1lo, j1hi, r), color=tc, linewidth=1.7,
                 linestyle="--", alpha=0.9)
    for e in (jlo, jhi, j1hi):
        axA.axvline(e, color="grey", linewidth=0.7, alpha=0.5)
    axA.set_title(r"(a) Convergencia: $\tau\to0$ recupera el binning duro",
                  fontsize=FS_TITLE)
    axA.set_xlabel(r"tiempo de vuelo $t_0\,/\,\Delta t$", fontsize=FS_LABEL)
    axA.set_ylabel(r"energia en el bin $S_j(t_0)$", fontsize=FS_LABEL)
    axA.set_ylim(-0.08, 1.18)
    _style(axA)
    axA.legend(fontsize=7, loc="upper right", ncol=1, framealpha=0.9)

    # ---- (b) how ONE pulse's energy is SHARED between neighbours (stacks to 1)
    # Representative pulse width tau = Delta t/3 (thin tails), focus near the
    # shared edge at u=0 so the two adjacent bins hold ~all the mass.  The
    # stacked bands (S_j, S_{j+1}, tails/other bins) add up to exactly 1.
    r_rep = 1.0 / 3.0
    r_rep_label = r"\Delta t/3"
    xb = np.linspace(-0.5, 0.5, 800)
    Sj = erf_bin_u(xb, jlo, jhi, r_rep)
    Sj1 = erf_bin_u(xb, j1lo, j1hi, r_rep)
    tails = np.clip(1.0 - Sj - Sj1, 0.0, None)   # energy in all other bins
    axB.stackplot(
        xb, Sj, Sj1, tails,
        colors=["#4C72B0", "#9467BD", "#CCCCCC"],
        labels=[r"$S_j(t_0)$ (bin $j$)", r"$S_{j+1}(t_0)$ (bin $j{+}1$)",
                r"colas / otros bins"],
        alpha=0.9,
    )
    axB.plot(xb, Sj + Sj1 + tails, color="black", linewidth=1.4,
             label=r"total apilado $=1$")
    # annotate the arithmetic of the split at a few t0 values
    for up in (-0.15, 0.0, 0.15):
        sj = float(erf_bin_u(np.array([up]), jlo, jhi, r_rep)[0])
        sj1 = float(erf_bin_u(np.array([up]), j1lo, j1hi, r_rep)[0])
        tl_ = max(0.0, 1.0 - sj - sj1)
        axB.axvline(up, color="black", linewidth=0.6, alpha=0.35)
        txt = (rf"${sj:.2f}+{sj1:.2f}+{tl_:.2f}=1.00$" if tl_ >= 0.01
               else rf"${sj:.2f}+{sj1:.2f}=1.00$")
        axB.annotate(txt, xy=(up, 1.0), xytext=(up, 1.045 + 0.045 * (up == 0.0)),
                     fontsize=6.4, color="black", ha="center",
                     arrowprops=dict(arrowstyle="->", color="black", lw=0.6))
    axB.set_title(rf"(b) El suavizado REPARTE la energia entre bins (total$=1$, "
                  rf"$\tau={r_rep_label}$)", fontsize=FS_TITLE - 0.5)
    axB.set_xlabel(r"tiempo de vuelo $t_0\,/\,\Delta t$", fontsize=FS_LABEL)
    axB.set_ylabel(r"reparto de energia (apilado a 1)", fontsize=FS_LABEL)
    axB.set_xlim(-0.5, 0.5)
    axB.set_ylim(0.0, 1.22)
    _style(axB)
    axB.legend(fontsize=6.6, loc="lower center", ncol=2, framealpha=0.9)
    axB.text(-0.49, 0.06, "invariancia total$=1$ vale para todo $\\tau$",
             fontsize=6.4, color="black", alpha=0.8)

    # ---- (c) derivative dS_j/dt0: bounded for finite tau, sharpens as tau->0 -
    for tau, tl, tc in zip(taus, tau_labels, tau_colors):
        r = tau / dt
        # dS_j/dt0 in units of 1/dt: (1/(r*sqrt(2pi)))[exp(-(hi-u)^2/2r^2) - exp(-(lo-u)^2/2r^2)]
        dS = (1.0 / (r * np.sqrt(2.0 * np.pi))) * (
            np.exp(-((jhi - u) ** 2) / (2.0 * r ** 2))
            - np.exp(-((jlo - u) ** 2) / (2.0 * r ** 2))
        )
        axC.plot(u, dS, color=tc, linewidth=1.7, label=rf"$\tau={tl}$")
    for e in (jlo, jhi):
        axC.axvline(e, color="grey", linewidth=0.7, alpha=0.5)
    axC.set_title(r"(c) Derivada $dS_j/dt_0$: acotada ($\tau$ finito), se afila al bajar $\tau$",
                  fontsize=FS_TITLE - 0.5)
    axC.set_xlabel(r"tiempo de vuelo $t_0\,/\,\Delta t$", fontsize=FS_LABEL)
    axC.set_ylabel(r"$dS_j/dt_0$  (unidades $1/\Delta t$)", fontsize=FS_LABEL)
    _style(axC)
    axC.legend(fontsize=7, loc="upper right", ncol=1, framealpha=0.9)

    fig.suptitle("Binning temporal (pieza central): caja dura $\\to$ diferencia "
                 "de $\\mathrm{erf}$  $C^\\infty$  "
                 "(convergencia, reparto/conservacion de masa, derivada)",
                 fontsize=FS_TITLE + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
