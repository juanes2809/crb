#!/usr/bin/env python3
"""Comparativa lado a lado CPU (numpy original) | GPU (simulator.py) | diferencia
de las funciones "a suavizar" del simulador, para la misma pose.

Reutiliza las réplicas de ``plot_simulator_components.py`` y escribe en
``gpu_vs_cpu/``: binning.png, occlusion.png, clamps.png, transient.png,
hard_functions.png y un README.md con las cifras medidas.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import plot_simulator_components as psc  # noqa: E402
from plot_simulator_components import (  # noqa: E402
    DPI, RHO, PHI, PHI_DEG, FACET_W, cam_pixel_dim, c, bin_size,
    evaluate_backend, to_img, imshow_floor, plot_hard_functions, column_shift, run_real_simulator, pose_str,
)

OUT = ROOT / "gpu_vs_cpu"
LBL = {"cpu": "CPU (numpy original)", "gpu": "GPU (simulator.py)"}


def triptych(axs, img_cpu, img_gpu, px, py, title, cmap="viridis", cbar_label=None, diff_cmap="RdBu_r",
             shared_scale=True, diff_label=None):
    """CPU | GPU | GPU−CPU con escalas compartidas en los dos primeros paneles."""
    kw = {}
    if shared_scale:
        vmin = min(np.nanmin(img_cpu), np.nanmin(img_gpu))
        vmax = max(np.nanmax(img_cpu), np.nanmax(img_gpu))
        kw = dict(vmin=vmin, vmax=vmax)
    imshow_floor(axs[0], img_cpu, px, py, f"{LBL['cpu']}\n{title}", cmap=cmap, cbar_label=cbar_label, **kw)
    imshow_floor(axs[1], img_gpu, px, py, f"{LBL['gpu']}\n{title}", cmap=cmap, cbar_label=cbar_label, **kw)
    d = img_gpu - img_cpu
    dmax = np.nanmax(np.abs(d)) if np.any(np.isfinite(d)) else 0.0
    note = f"max|Δ| = {dmax:.3e}"
    if dmax < 1e-12:
        note = f"max|Δ| = {dmax:.1e}  (= 0 salvo redondeo)"
        dmax = 1e-3  # escala fija para que el panel se vea plano
    imshow_floor(axs[2], d, px, py, f"diferencia GPU − CPU\n{diff_label or title}   {note}",
                 cmap=diff_cmap, cbar_label="Δ", vmin=-dmax, vmax=dmax)
    return float(np.nanmax(np.abs(d)))


def best_column_shift(ya, yb, max_shift=4):
    """Desplazamiento s (columnas) tal que ya[:, jx, :] ≈ yb[:, jx+s, :], por mínimo
    max|Δ| sobre las columnas comunes (se excluye la última columna, que en el
    original CPU contiene la columna 0 envuelta).  Devuelve (s, residuo)."""
    N = ya.shape[1]
    best = (0, np.inf)
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            d = np.abs(ya[:, :N - s - 1] - yb[:, s:N - 1]).max()
        else:
            d = np.abs(ya[:, -s:N - 1] - yb[:, :N - 1 + s]).max()
        if d < best[1]:
            best = (s, float(d))
    return best


def normal_err_deg(n, phi):
    n_exp = np.array([-np.cos(phi), -np.sin(phi), 0.0])
    return float(np.degrees(np.arccos(np.clip(np.dot(n, n_exp), -1, 1)))), n_exp


def main():
    OUT.mkdir(exist_ok=True)
    R = {}  # cifras para el README

    print("Evaluando backends (phi=60°, con transitorio) ...")
    t0 = time.time()
    C = evaluate_backend("cpu", verbose=False)
    R["t_cpu_loop"] = time.time() - t0
    t0 = time.time()
    G = evaluate_backend("gpu", verbose=False)
    R["t_gpu_replica_loop"] = time.time() - t0
    print("Evaluando backends (phi=30°, sólo mapas) ...")
    PHI30 = np.deg2rad(30.0)
    C30 = evaluate_backend("cpu", phi=PHI30, run_transient=False)
    G30 = evaluate_backend("gpu", phi=PHI30, run_transient=False)

    px, py = C["pixel_x"], C["pixel_y"]
    N = C["N"]
    TC, TG = C["T"], G["T"]

    # ------------------------------------------------------------------ 1) binning
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.8])
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ab_c, ab_g = to_img(TC["arrival_bin"]), to_img(TG["arrival_bin"])
    nb = int(max(ab_c.max(), ab_g.max()) - min(ab_c.min(), ab_g.min())) + 1
    lo, hi = min(ab_c.min(), ab_g.min()), max(ab_c.max(), ab_g.max())
    imshow_floor(axs[0], ab_c, px, py, f"{LBL['cpu']}\narrival_bin = ceil(distance/(cΔt))",
                 cmap=plt.get_cmap("tab20", nb), cbar_label="bin", vmin=lo - 0.5, vmax=hi + 0.5)
    imshow_floor(axs[1], ab_g, px, py, f"{LBL['gpu']}\narrival_bin = ceil(distance/(cΔt))  (valid: 0<bin≤T)",
                 cmap=plt.get_cmap("tab20", nb), cbar_label="bin", vmin=lo - 0.5, vmax=hi + 0.5)
    dbin = ab_g - ab_c
    R["n_bin_diff"] = int(np.sum(dbin != 0))
    imshow_floor(axs[2], dbin, px, py, f"diferencia GPU − CPU\npíxeles con bin distinto: {R['n_bin_diff']}",
                 cmap="RdBu_r", cbar_label="Δbin", vmin=-1.5, vmax=1.5)
    ax = fig.add_subplot(gs[1, :])
    oc = np.argsort(TC["distance"]); og = np.argsort(TG["distance"])
    ax.plot(TC["distance"][oc], TC["tbin"][oc], "-", color="C0", lw=1.2, label="CPU: distance/(cΔt) continuo")
    ax.plot(TG["distance"][og], TG["tbin"][og], "--", color="C1", lw=1.2, label="GPU: distance/(cΔt) continuo")
    ax.step(TC["distance"][oc], TC["arrival_bin"][oc], where="post", color="C3", lw=2.2, label="CPU: ceil(·)")
    ax.step(TG["distance"][og], TG["arrival_bin"][og], where="post", color="k", lw=1.0, ls="--", label="GPU: ceil(·)")
    ax.set_xlabel("distance = d1 + d2 [m]"); ax.set_ylabel("bin"); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    R["max_d_distance"] = float(np.abs(TG["distance"] - TC["distance"]).max())
    ax.set_title(f"Corte 1D (todos los píxeles ordenados por distancia): la escalera del ceil es la misma función en "
                 f"ambos backends — max|Δdistance| = {R['max_d_distance']:.2e} m", fontsize=9)
    fig.suptitle(f"Binning temporal (ceil) — CPU vs GPU — {pose_str()} (cΔt = {c*bin_size*100:.2f} cm)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "binning.png", dpi=DPI)
    plt.close(fig)

    # ------------------------------------------------------------------ 2) occlusion
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
    xi_c, xi_g = to_img(TC["xint"]), to_img(TG["xint"])
    R["max_d_xint"] = triptych(axs[0], xi_c, xi_g, px, py, "xint = -b/m", cmap="RdBu_r", cbar_label="xint [m]")
    Xg, Yg = np.meshgrid(px, py)
    for a in axs[0, :2]:
        a.contour(Xg, Yg, xi_c, levels=[0.0], colors="k", linewidths=1.2)
    noc_c, noc_g = to_img(TC["noc"]).astype(float), to_img(TG["noc"]).astype(float)
    changed = noc_c != noc_g
    R["n_noc_changed"] = int(changed.sum())
    R["n_noc_cpu"], R["n_noc_gpu"] = int(TC["noc"].sum()), int(TG["noc"].sum())
    R["n_eps_masked"] = int(np.sum(np.abs(C["center"][0] - C["cam_pos"][:, 0]) <= psc.EPS))
    imshow_floor(axs[1, 0], noc_c, px, py, f"{LBL['cpu']}\nnoc = 1{{xint>0}}   ({R['n_noc_cpu']} px)", cmap="gray",
                 cbar_label="noc", vmin=0, vmax=1)
    imshow_floor(axs[1, 1], noc_g, px, py, f"{LBL['gpu']}\nnoc = isfinite(xint) & (xint>0)   ({R['n_noc_gpu']} px)",
                 cmap="gray", cbar_label="noc", vmin=0, vmax=1)
    imshow_floor(axs[1, 2], changed.astype(float), px, py,
                 f"píxeles que cambian de estado: {R['n_noc_changed']}\n(|denom|≤eps enmascarados en GPU: {R['n_eps_masked']})",
                 cmap="Reds", cbar_label="cambia", vmin=0, vmax=1)
    if R["n_noc_changed"]:
        iy, jx = np.nonzero(changed)
        axs[1, 2].plot(px[jx], py[iy], "kx", ms=6)
    fig.suptitle(f"Oclusión (Heaviside) — CPU vs GPU — {pose_str()}\n"
                 f"El centroide de la faceta es el mismo en ambos (theta rota la faceta sobre su propio centro), "
                 f"así que xint/noc coinciden; en GPU sólo se añade la máscara |denom|>eps e isfinite.", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "occlusion.png", dpi=DPI)
    plt.close(fig)

    # ------------------------------------------------------------------ 3) clamps
    fig, axs = plt.subplots(4, 3, figsize=(16, 18.5))
    rows = []
    for (Ec, Eg, phid) in [(C, G, 60.0), (C30, G30, 30.0)]:
        tc, tg = Ec["T"], Eg["T"]
        e_c, n_exp = normal_err_deg(Ec["normal"], Ec["phi"])
        e_g, _ = normal_err_deg(Eg["normal"], Eg["phi"])
        R[f"dot2_max_d_{int(phid)}"] = float(np.abs(tg["dot2"] - tc["dot2"]).max())
        R[f"dot4_max_d_{int(phid)}"] = float(np.abs(tg["dot4"] - tc["dot4"]).max())
        R[f"dot1_cpu_{int(phid)}"], R[f"dot1_gpu_{int(phid)}"] = float(tc["dot1"]), float(tg["dot1"])
        R[f"dot3_cpu_{int(phid)}"], R[f"dot3_gpu_{int(phid)}"] = float(tc["dot3"]), float(tg["dot3"])
        R[f"nerr_cpu_{int(phid)}"], R[f"nerr_gpu_{int(phid)}"] = e_c, e_g
        R[f"normal_cpu_{int(phid)}"], R[f"normal_gpu_{int(phid)}"] = Ec["normal"], Eg["normal"]
        R[f"normal_exp_{int(phid)}"] = n_exp
        R[f"theta_cpu_{int(phid)}"], R[f"theta_gpu_{int(phid)}"] = float(Ec["theta"]), float(Eg["theta"])
        rows.append((phid, tc, tg, e_c, e_g, n_exp, Ec, Eg))
    r = 0
    for phid, tc, tg, e_c, e_g, n_exp, Ec, Eg in rows:
        ttl = rf"$\varphi$={phid:.0f}°   dot2 = max(0, n·fovsp/d2)"
        triptych(axs[r], to_img(tc["dot2"]), to_img(tg["dot2"]), px, py, ttl, cbar_label="cos",
                 diff_label=rf"$\varphi$={phid:.0f}° dot2")
        axs[r, 0].text(0.02, 0.02, f"normal CPU = {np.round(Ec['normal'], 3)}\nerror {e_c:.2f}°  (theta = -cos φ = {Ec['theta']:+.4f} rad)\n"
                       f"dot1 = {tc['dot1']:.4f}, dot3 = {tc['dot3']:.4f}",
                       transform=axs[r, 0].transAxes, fontsize=7, color="w", va="bottom",
                       bbox=dict(fc="k", alpha=0.65, boxstyle="round"))
        axs[r, 1].text(0.02, 0.02, f"normal GPU = {np.round(Eg['normal'], 3)}\nerror {e_g:.2f}°  (theta = φ + 3π/2)\n"
                       f"esperada {np.round(n_exp, 3)}\ndot1 = {tg['dot1']:.4f}, dot3 = {tg['dot3']:.4f}",
                       transform=axs[r, 1].transAxes, fontsize=7, color="w", va="bottom",
                       bbox=dict(fc="k", alpha=0.65, boxstyle="round"))
        r += 1
        triptych(axs[r], to_img(tc["dot4"]), to_img(tg["dot4"]), px, py,
                 rf"$\varphi$={phid:.0f}°   dot4 = max(0, n_floor·(−fovsp)/d2)",
                 cbar_label="cos", diff_label=rf"$\varphi$={phid:.0f}° dot4 (sólo geometría, no depende de la normal)")
        r += 1
    fig.suptitle("Cosenos con clamp (dot2, dot4) — CPU vs GPU — filas 1-2: φ=60°, filas 3-4: φ=30° (ρ=1 m)\n"
                 "La diferencia en dot2 viene del theta=-cos(φ) de la CPU (normal mal orientada); dot4 no depende de la normal "
                 "y el centroide es el mismo → Δ=0", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "clamps.png", dpi=DPI)
    plt.close(fig)

    # ------------------------------------------------------------------ 4) transient
    y_c, y_c_fixed = C["y"], C["y_alt"]       # original (con -1) / corregido sin roll
    y_g, y_g_raw = G["y"], G["y_alt"]         # final (con roll) / sin roll
    img_c, img_g, img_g_raw, img_c_fixed = y_c.max(2), y_g.max(2), y_g_raw.max(2), y_c_fixed.max(2)
    R["shift_cpu_orig_vs_gpu_final"], R["resid_cpu_orig_vs_gpu_final"] = best_column_shift(y_c, y_g)
    R["shift_cpu_orig_vs_gpu_raw"], R["resid_cpu_orig_vs_gpu_raw"] = best_column_shift(y_c, y_g_raw)
    R["shift_cpu_fixed_vs_gpu_raw"], R["resid_cpu_fixed_vs_gpu_raw"] = best_column_shift(y_c_fixed, y_g_raw)
    R["shift_cpu_fixed_vs_gpu_final"], R["resid_cpu_fixed_vs_gpu_final"] = best_column_shift(y_c_fixed, y_g)
    R["xcorr_lag_cpu_orig_vs_gpu_final"] = column_shift(img_c, img_g)
    R["max_d_img"] = float(np.abs(img_g - img_c).max())
    R["max_d_img_fixed_vs_raw"] = float(np.abs(img_g_raw - img_c_fixed).max())
    R["max_d_y_fixed_vs_raw"] = float(np.abs(y_g_raw - y_c_fixed).max())
    R["sum_cpu"], R["sum_gpu"] = float(y_c.sum()), float(y_g.sum())
    R["max_img_cpu"], R["max_img_gpu"] = float(img_c.max()), float(img_g.max())

    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.8])
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    triptych(axs, img_c, img_g, px, py, "max_t y[iy, jx, t]", cmap="hot", cbar_label="a.u.",
             diff_label=f"max_t y   (y_cpu[:, jx] ≈ y_gpu[:, jx{R['shift_cpu_orig_vs_gpu_final']:+d}], "
                        f"residuo {R['resid_cpu_orig_vs_gpu_final']:.1e})")
    axs[0].set_title(f"{LBL['cpu']}\nmax_t y — coord con '-1' (corrida 1 col. a la izq.)", fontsize=9)
    axs[1].set_title(f"{LBL['gpu']}\nmax_t y — índice correcto + index_add_ + roll(+1) (1 col. a la der.)", fontsize=9)
    iy_c, jx_c = N // 2, (3 * N) // 4
    for a in axs[:2]:
        a.plot(px[jx_c], py[iy_c], "c+", ms=12, mew=2)
    ax = fig.add_subplot(gs[1, 0:2])
    hc, hg = y_c[iy_c, jx_c], y_g[iy_c, jx_c]
    nz = np.nonzero(hc + hg)[0]
    b_lo, b_hi = max(nz.min() - 2, 0), min(nz.max() + 3, C["num_bins"])
    bins = np.arange(b_lo, b_hi) + 1
    ax.bar(bins - 0.2, hc[b_lo:b_hi], width=0.4, color="C0", label=f"CPU original (px iy={iy_c}, jx={jx_c})")
    ax.bar(bins + 0.2, hg[b_lo:b_hi], width=0.4, color="C3", label=f"GPU final (px iy={iy_c}, jx={jx_c})")
    ax.set_xlabel("arrival_bin (entero)"); ax.set_ylabel("y [a.u.]"); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    R["hist_max_d"] = float(np.abs(hg - hc).max())
    ax.set_title(f"Histograma temporal del mismo píxel central — max|Δ| = {R['hist_max_d']:.3e} "
                 f"(el píxel físico no es el mismo: CPU muestra jx+1, GPU muestra jx−1)", fontsize=9)
    ax = fig.add_subplot(gs[1, 2])
    cols = np.arange(N)
    ax.plot(cols, img_c.sum(0), "o-", color="C0", ms=3, label="CPU original (−1)")
    ax.plot(cols, img_c_fixed.sum(0), "s--", color="C0", ms=3, alpha=0.5, label="CPU corregido, sin roll")
    ax.plot(cols, img_g_raw.sum(0), "^:", color="C3", ms=3, alpha=0.6, label="GPU sin roll (y_raw)")
    ax.plot(cols, img_g.sum(0), "o-", color="C3", ms=3, label="GPU final (roll +1)")
    ax.set_xlabel("columna jx"); ax.set_ylabel("Σ_iy max_t y"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title(f"Perfil por columnas — desplazamientos:\nCPU-orig→GPU-final {R['shift_cpu_orig_vs_gpu_final']:+d} col., "
                 f"CPU-orig→GPU-raw {R['shift_cpu_orig_vs_gpu_raw']:+d}, CPU-fix→GPU-raw {R['shift_cpu_fixed_vs_gpu_raw']:+d}",
                 fontsize=8)
    fig.suptitle(f"Transitorio (malla completa) — CPU vs GPU — {pose_str()}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "transient.png", dpi=DPI)
    plt.close(fig)

    # ------------------------------------------------------------------ 5) hard functions
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    plot_hard_functions(axs)
    x = np.linspace(-1.5, 1.5, 61)
    axs[0].plot(x, np.maximum(0, x), "k--", lw=1, label="GPU: torch.clamp(min=0) — idéntica")
    x2 = np.linspace(0, 5, 51)
    axs[1].step(x2, np.ceil(x2), where="post", color="k", ls="--", lw=1, label="GPU: torch.ceil — idéntica")
    x3 = np.linspace(-1, 1, 51)
    axs[2].step(x3, (x3 > 0).astype(float), where="post", color="k", ls="--", lw=1,
                label="GPU: isfinite(xint) & (xint>0) — idéntica")
    for a in axs:
        a.legend(fontsize=7)
    fig.suptitle("Funciones duras 1D: IDÉNTICAS en CPU y GPU — misma función a suavizar; solo cambia dónde/cómo se "
                 "evalúa (máscara eps, bounds de bin, índice, acumulación)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "hard_functions.png", dpi=DPI)
    plt.close(fig)

    # ------------------------------------------------------------------ sanity check (réplica GPU vs simulador real)
    print("Sanity check contra simulator.simulation ...")
    t0 = time.time()
    y_real, params, _ = run_real_simulator(RHO, PHI, FACET_W, N)
    R["t_torch_real"] = time.time() - t0
    R["sanity_max_abs"] = float(np.abs(G["y"] - y_real).max())
    R["sanity_max_rel"] = R["sanity_max_abs"] / float(np.abs(y_real).max())

    write_readme(R, C, G)
    print("\n".join(f"{k}: {v}" for k, v in R.items() if not isinstance(v, np.ndarray)))
    print(f"\nEscrito {OUT}/")


def write_readme(R, C, G):
    def n3(v):
        return "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"

    sc = C["stats"]; sg = G["stats"]
    txt = f"""# CPU (numpy original) vs GPU (`simulator.py`): las funciones a suavizar, lado a lado

Pose: ρ = {RHO:.2f} m, φ = {PHI_DEG:.0f}° (x = {RHO*np.cos(PHI):.4f}, y = {RHO*np.sin(PHI):.4f}), `facet.obj`, w = {FACET_W},
N = {cam_pixel_dim}, FOV = {psc.camera_FOV} m, Δt = {bin_size:.1e} s (cΔt = {c*bin_size*100:.2f} cm), T = {C['num_bins']} bins,
{sc['n_tri']} triángulos.  Figuras generadas por `plot_gpu_vs_cpu.py` a partir de las réplicas de
`plot_simulator_components.py` (`--backend cpu|gpu`).  Los mapas por píxel usan el centroide de la faceta como
"triángulo representativo"; el transitorio usa la malla completa.

**Sanity check** de la réplica GPU: `max|réplica − simulator.simulation(...)|` = **{R['sanity_max_abs']:.2e}**
(relativo {R['sanity_max_rel']:.1e}; torch CPU float64, `hide_walls=True`, `add_noise=False`, con el mismo
`roll(+1, axis=1)` de `orient_transient_measurement`).  La réplica numpy y el simulador real son la misma función.

## Funciones a suavizar

| Función dura | CPU original | GPU `simulator.py` | ¿Igual? | Medido (φ=60°) |
|---|---|---|---|---|
| **ceil-binning** `arrival_bin = ceil((d1+d2)/(cΔt))` | `np.ceil(...).astype(int)`, sin bounds | `torch.ceil(...).long()`, `valid = noc & (bin>0) & (bin≤T)` | **Misma función.** Sólo la GPU descarta bins fuera de rango (aquí 0 descartados: bins {sg['min_bin']}…{sg['max_bin']} de {C['num_bins']}). | píxeles con bin distinto: **{R['n_bin_diff']}**; max\\|Δdistance\\| = {R['max_d_distance']:.1e} m |
| **Heaviside-oclusión** `noc = 1{{xint>0}}` | `m = Δy/Δx` sin máscara; `xint>0` | `m = where(\\|Δx\\|>eps, Δy/Δx, nan)`; `noc = isfinite(xint) & (xint>0)` | **Misma función** (Heaviside en `xint=0`). La GPU sólo enmascara `\\|Δx\\|≤eps` (aquí {R['n_eps_masked']} px). | píxeles que cambian de estado: **{R['n_noc_changed']}** ({R['n_noc_cpu']} vs {R['n_noc_gpu']} px ven la faceta); max\\|Δxint\\| = {R['max_d_xint']:.1e} m |
| **clamps** `dot_k = max(0, cos_k)` | `np.maximum(0, ·)` | `torch.clamp(min=0)` | **Misma función.** Lo que cambia es el *argumento*: la normal de la faceta (theta). | max\\|Δdot2\\| = **{R['dot2_max_d_60']:.3e}**, max\\|Δdot4\\| = **{R['dot4_max_d_60']:.1e}** (φ=60°); max\\|Δdot2\\| = **{R['dot2_max_d_30']:.3e}**, max\\|Δdot4\\| = {R['dot4_max_d_30']:.1e} (φ=30°) |

Por qué `xint`, `noc`, `arrival_bin`, `d2` y `dot4` coinciden exactamente: `theta` rota la faceta alrededor de su
propio centro (la malla está centrada en el eje z antes de trasladarla a `v1`), así que el centroide es el mismo en
ambos backends y todo lo que sólo depende de la geometría píxel–centroide es idéntico.  La diferencia aparece en lo
que depende de la **normal** (`dot1`, `dot2`) y, en la malla completa, en la posición de cada triángulo individual.

## Diferencias de implementación

| # | Aspecto | CPU original | GPU `simulator.py` | Cifra medida |
|---|---|---|---|---|
| 1 | `theta` (rotación en z) | `theta = -cos(∠(u, v1)) = -cos φ` **usado como ángulo** (número en [-1,0] rad) | `theta = φ + 3π/2` (ángulo geométrico) | φ=60°: theta CPU = {R['theta_cpu_60']:+.4f} rad vs GPU ≡ {np.degrees((R['theta_gpu_60']+np.pi)%(2*np.pi)-np.pi):+.1f}°; normal CPU {n3(R['normal_cpu_60'])} (error **{R['nerr_cpu_60']:.2f}°**) vs GPU {n3(R['normal_gpu_60'])} (error {R['nerr_gpu_60']:.2f}°), esperada {n3(R['normal_exp_60'])}. φ=30°: normal CPU {n3(R['normal_cpu_30'])} (error **{R['nerr_cpu_30']:.2f}°**) vs GPU {n3(R['normal_gpu_30'])} (error {R['nerr_gpu_30']:.2f}°). dot1: {R['dot1_cpu_60']:.4f} vs {R['dot1_gpu_60']:.4f} (60°), {R['dot1_cpu_30']:.4f} vs {R['dot1_gpu_30']:.4f} (30°); dot3 igual ({R['dot3_cpu_60']:.4f} / {R['dot3_cpu_30']:.4f}). |
| 2 | Índice de píxel | `coord = (bin-1)N² + (jx **- 1**)N + iy` → imagen corrida 1 columna a la izquierda; jx=0 envuelve a la columna N-1 del bin anterior ({sc['n_coord_wrap']} depósitos) | `coord = (bin-1)·N² + jx·N + iy` (sin `-1`) | desplazamiento medido (mínimo max\\|Δ\\| sobre columnas comunes): `y_cpu[:, jx] ≈ y_gpu_raw[:, jx{R['shift_cpu_orig_vs_gpu_raw']:+d}]` (**{R['shift_cpu_orig_vs_gpu_raw']:+d} col.**, residuo {R['resid_cpu_orig_vs_gpu_raw']:.1e}); CPU-corregido vs GPU-raw: {R['shift_cpu_fixed_vs_gpu_raw']:+d} col. (residuo {R['resid_cpu_fixed_vs_gpu_raw']:.1e}, sólo por theta; max\\|Δ max_t y\\| = {R['max_d_img_fixed_vs_raw']:.3e}) |
| 3 | Acumulación | `y[coord] += intensity` (con índices repetidos se queda el último valor) | `index_add_` (suma todos) | en el original los índices no se repiten dentro de un triángulo ({sc['n_dup_within_tri']} duplicados; energía perdida {sc['lost_pluseq']:.1e}); el bug es latente y sólo muerde al vectorizar sobre triángulos, que es lo que hace la GPU |
| 4 | Máscaras `eps` | ninguna: `Δx=0` → `m=±inf`, `xint=-0.0` → píxel ocluido en silencio | `\\|Δx\\|>eps`, `clamp(d1s,d2s ≥ eps)`, `isfinite(xint)` | píxeles afectados en esta pose: {R['n_eps_masked']} |
| 5 | Bounds de bin | ninguno (un objeto fuera de bounds → `IndexError` o envuelve) | `0 < bin ≤ T` | descartados en esta pose: {sg['n_bin_out_of_range']} (bins {sg['min_bin']}…{sg['max_bin']} de {C['num_bins']}) |
| 6 | `roll` final | ninguno (la variante de `utils/crb_fix.py` hace `roll(-1, axis=-1)` en t) | `orient_transient_measurement`: `np.roll(y, +1, axis=1)` en jx | la salida GPU final queda corrida **una columna a la derecha** del índice correcto (`y_cpu_corregido[:, jx] ≈ y_gpu[:, jx{R['shift_cpu_fixed_vs_gpu_final']:+d}]`); frente al original CPU (corrido a la izquierda) el desplazamiento total es **{R['shift_cpu_orig_vs_gpu_final']:+d} columnas** (residuo {R['resid_cpu_orig_vs_gpu_final']:.1e}). Nota: la correlación cruzada de perfiles de columna (usada en el review CPU) da aquí {R['xcorr_lag_cpu_orig_vs_gpu_final']:+d} porque la imagen es casi constante por columnas; la medida por mínimo max\\|Δ\\| es la fiable. |

Transitorio: `max_t y` difiere entre CPU-original y GPU-final en max\\|Δ\\| = {R['max_d_img']:.3e} (máximos {R['max_img_cpu']:.4f} vs
{R['max_img_gpu']:.4f}); energía total {R['sum_cpu']:.4f} vs {R['sum_gpu']:.4f} (Δ = {100*(R['sum_gpu']-R['sum_cpu'])/R['sum_cpu']:+.2f} %, por la
orientación de la faceta).  Comparando lo comparable (CPU corregido sin roll vs GPU sin roll) la diferencia
píxel-a-píxel-a-bin es max\\|Δy\\| = {R['max_d_y_fixed_vs_raw']:.3e}, atribuible sólo al `theta`.
Histograma del píxel central (iy={cam_pixel_dim//2}, jx={3*cam_pixel_dim//4}): max\\|Δ\\| = {R['hist_max_d']:.3e} (mismo índice, pero por los
desplazamientos de columna opuestos corresponde a píxeles físicos distintos).

Tiempos en esta máquina (sin GPU): bucle numpy por triángulo {R['t_cpu_loop']:.1f} s (CPU original) /
{R['t_gpu_replica_loop']:.1f} s (réplica GPU), `simulator.simulation` vectorizado en torch-CPU {R['t_torch_real']:.1f} s.

## Figuras

- `binning.png` — `arrival_bin` CPU | GPU | Δ y el corte 1D de la escalera del `ceil` superpuesto.
- `occlusion.png` — `xint` y `noc` CPU | GPU | Δ, con los píxeles que cambian de estado.
- `clamps.png` — `dot2` y `dot4` con clamp, filas 1-2 φ=60°, filas 3-4 φ=30°, con normales y `dot1`, `dot3` anotados.
- `transient.png` — `max_t y` CPU | GPU | Δ, histograma del píxel central y perfiles por columna con los lags medidos.
- `hard_functions.png` — `max(0,x)`, `ceil(x)`, `1{{x>0}}`: idénticas en ambos backends.

## Conclusión

Las tres no-linealidades duras que habría que suavizar para un forward diferenciable (`ceil` del binning,
Heaviside de la oclusión, `clamp` de los cosenos) son **exactamente las mismas funciones** en los dos simuladores;
la GPU no cambia la física, sólo dónde y cómo se evalúa.  Lo que sí cambia son los bugs: la GPU corrige el `theta`
(normal de la faceta bien orientada: {R['nerr_gpu_60']:.2f}° de error frente a {R['nerr_cpu_60']:.2f}° / {R['nerr_cpu_30']:.2f}° de la CPU en
φ=60°/30°), el índice `-1`, la acumulación con `index_add_`, y añade máscaras `eps` y bounds de bin, a costa de un
`roll(+1)` final de orientación que hay que tener en cuenta al comparar con la salida original.  Es además mucho más
rápida (vectorizada por chunks de triángulos; ~60× en GPU según la medición del repositorio, y aquí
{R['t_cpu_loop']/R['t_torch_real']:.0f}× incluso en torch-CPU).  **El simulador GPU es el preferible**: misma
física, bugs corregidos, y es la única implementación cuya salida coincide con la réplica al nivel de redondeo
({R['sanity_max_abs']:.1e}).
"""
    (OUT / "README.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
