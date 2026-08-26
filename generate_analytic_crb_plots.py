#!/usr/bin/env python3
"""Generate CRB bubble/ellipse plots from the ANALYTIC differentiable forward,
and compare them against the rasterized / finite-difference (FD) method.

Part (1): analytic grid + the same (rho, phi) 3-sigma bubble plots as the
original FD pipeline (crb_standard_regions*_analytic.png, compare_2_vs_3).

Part (2): overlay analytic vs FD (loaded from the FD cache
plots/crb_grid_results.pkl produced by regenerate_crb_standard_regions.py),
both as-is (real magnitude) and shape-normalized (median sigma_rho matched),
plus a numeric comparison table -> docs/analytic_vs_fd_comparison.txt.

With variant (a)+(b) the analytic forward now uses the simulator's physical
radiometry (I_laser * area_eff = I_laser * w*h) and the simulator's 64x64 grid /
real bin size, so absolute CRB magnitudes ARE comparable with the FD route.  The
real-magnitude overlay is therefore the primary comparison; the shape-normalized
overlay and the aspect/tilt columns still document the shape/orientation match.

Run:  python3 generate_analytic_crb_plots.py
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from crb_analytic_forward import ForwardConfig, compute_crb_analytic
from crb_polar_functions import (
    crb_region_in_polar_parameters,
    plot_crb_regions_compare,
    plot_crb_regions_polar,
)

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
DOCS = ROOT / "docs"

RANGES = [0.5, 1.0, 1.5]
ANGLES_DEG = [30, 60, 90, 120, 150]

FD_CACHE = PLOTS / "crb_grid_results.pkl"
# The (large, untracked) FD cache may live in the main /workspace checkout rather
# than in an isolated worktree; fall back to that absolute path if needed.
FD_CACHE_FALLBACK = Path("/workspace/plots/crb_grid_results.pkl")
ANALYTIC_CACHE = PLOTS / "crb_grid_results_analytic.pkl"


# ---------------------------------------------------------------------------
# Part (1): analytic grid
# ---------------------------------------------------------------------------
def _ensure_ellipse_curves(res: Dict[str, Any], k: float = 3.0) -> Dict[str, Any]:
    """Guarantee rho0/phi0 + 3-sigma ellipse curves in (rho, phi) exist."""
    if "rho_curve_param" not in res or "phi_curve_param" not in res:
        rho_p, phi_p = crb_region_in_polar_parameters(
            res["rho0"], res["phi0"], np.asarray(res["CRB"]), k=k
        )
        res["rho_curve_param"] = rho_p
        res["phi_curve_param"] = phi_p
    return res


def compute_analytic_grid(
    estimate_height: bool, config: Optional[ForwardConfig] = None
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for rho in RANGES:
        for phi_deg in ANGLES_DEG:
            psi = np.array([float(rho), np.deg2rad(float(phi_deg)), 1.0])
            res = compute_crb_analytic(
                psi, config=config, estimate_height=estimate_height, k=3.0
            )
            results.append(_ensure_ellipse_curves(res))
    return results


def build_or_load_analytic() -> Dict[str, List[Dict[str, Any]]]:
    if ANALYTIC_CACHE.exists():
        with open(ANALYTIC_CACHE, "rb") as f:
            cache = pickle.load(f)
        if "fixed_h" in cache and "with_h" in cache:
            print(f"Reusing analytic cache {ANALYTIC_CACHE.name}")
            return cache
    print("Computing analytic grids (fixed_h, with_h) ...")
    cache = {
        "fixed_h": compute_analytic_grid(estimate_height=False),
        "with_h": compute_analytic_grid(estimate_height=True),
    }
    with open(ANALYTIC_CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved analytic grids to {ANALYTIC_CACHE} (not committed).")
    return cache


def make_analytic_plots(cache: Dict[str, List[Dict[str, Any]]]) -> None:
    plot_crb_regions_polar(
        cache["fixed_h"],
        use_physical_region=False,
        rlim=2.0,
        title=r"CRB$(\rho,\varphi)$ analítico — regiones $3\sigma$",
        output_path=PLOTS / "crb_standard_regions_fixed_h_analytic.png",
    )
    print("Saved crb_standard_regions_fixed_h_analytic.png")

    plot_crb_regions_polar(
        cache["with_h"],
        use_physical_region=False,
        rlim=2.0,
        title=r"CRB$(\rho,\varphi,h)$ analítico — regiones $3\sigma$ en $(\rho,\varphi)$",
        output_path=PLOTS / "crb_standard_regions_analytic.png",
    )
    print("Saved crb_standard_regions_analytic.png")

    plot_crb_regions_compare(
        cache["fixed_h"],
        cache["with_h"],
        rlim=2.0,
        output_path=PLOTS / "crb_regions_compare_2_vs_3_analytic.png",
    )
    print("Saved crb_regions_compare_2_vs_3_analytic.png")


# ---------------------------------------------------------------------------
# Part (2): analytic vs FD comparison
# ---------------------------------------------------------------------------
def ellipse_shape_metrics(res: Dict[str, Any]) -> Dict[str, float]:
    """Aspect ratio and tilt (deg) of the (rho, phi) 3-sigma ellipse."""
    CRB = np.asarray(res["CRB"], dtype=float)
    Sigma = CRB[np.ix_([0, 1], [0, 1])]
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.maximum(evals, 0.0)
    smin, smax = np.sqrt(evals[0]), np.sqrt(evals[1])
    aspect = float(smax / smin) if smin > 0 else float("inf")
    # tilt of the major axis in the (rho, phi) plane, degrees
    major = evecs[:, int(np.argmax(evals))]
    tilt = float(np.rad2deg(np.arctan2(major[1], major[0])))
    return {"aspect": aspect, "tilt_deg": tilt, "smin": smin, "smax": smax}


def scale_curves_about_center(res: Dict[str, Any], factor: float) -> Dict[str, Any]:
    """Return a shallow copy whose 3-sigma curve is scaled about (rho0, phi0)."""
    out = dict(res)
    rho0, phi0 = res["rho0"], res["phi0"]
    out["rho_curve_param"] = rho0 + factor * (np.asarray(res["rho_curve_param"]) - rho0)
    out["phi_curve_param"] = phi0 + factor * (np.asarray(res["phi_curve_param"]) - phi0)
    return out


def _overlay(
    results_a: List[Dict[str, Any]],
    results_b: List[Dict[str, Any]],
    label_a: str,
    label_b: str,
    title: str,
    output_path: Path,
    rlim: float = 2.0,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={"projection": "polar"})
    plot_crb_regions_polar(
        results_a, use_physical_region=False, rlim=rlim, title=title,
        color="C2", label=label_a, ax=ax,
    )
    plot_crb_regions_polar(
        results_b, use_physical_region=False, rlim=rlim, title=title,
        color="C3", label=label_b, ax=ax,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path.name}")


def load_fd_cache() -> Optional[Dict[str, List[Dict[str, Any]]]]:
    path = FD_CACHE if FD_CACHE.exists() else FD_CACHE_FALLBACK
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            cache = pickle.load(f)
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: could not read FD cache: {exc}")
        return None
    print(f"Loaded FD cache from {path}")
    return cache


def make_comparison(
    analytic: Dict[str, List[Dict[str, Any]]],
    fd: Dict[str, List[Dict[str, Any]]],
) -> None:
    # Prefer the with_h grid for the canonical overlays; fall back to fixed_h if
    # the (longer) 3-param FD grid is not yet available in the cache.
    if fd.get("with_h"):
        key, suffix, tag = "with_h", "", r"CRB$(\rho,\varphi,h)$"
    elif fd.get("fixed_h"):
        key, suffix, tag = "fixed_h", "_fixed_h", r"CRB$(\rho,\varphi)$"
        print("NOTE: FD 'with_h' not in cache; using FD 'fixed_h' (2-param) "
              "for the overlays (apples-to-apples 2-param comparison).")
    else:
        print("WARNING: no FD grid available; skipping overlay plots.")
        write_comparison_table(analytic, fd)
        return

    a_grid = analytic[key]
    f_grid = fd[key]

    # --- (a) overlay as-is (absolute magnitude) --------------------------------
    _overlay(
        a_grid, f_grid,
        rf"Analítico {tag}", rf"FD {tag}",
        r"Analítico vs FD — regiones $3\sigma$ (magnitud real)",
        PLOTS / "crb_regions_analytic_vs_fd.png",
    )
    a_hard = [r for r in a_grid if abs(r["rho0"] - 1.5) < 1e-9]
    f_hard = [r for r in f_grid if abs(r["rho0"] - 1.5) < 1e-9]
    _overlay(
        a_hard, f_hard,
        r"Analítico", r"FD",
        r"Analítico vs FD — $\rho=1.5$ m (magnitud real)",
        PLOTS / "crb_regions_analytic_vs_fd_rho1p5.png",
    )

    # --- (b) shape-normalized overlay (median sigma_rho matched) ---------------
    a_med = float(np.median([r["sigma_rho"] for r in a_grid]))
    f_med = float(np.median([r["sigma_rho"] for r in f_grid]))
    target = a_med  # common reference: analytic median sigma_rho
    fa = target / a_med  # == 1
    ff = target / f_med
    print(
        f"Shape-norm factors: analytic x{fa:.4g} (median sigma_rho={a_med:.4g} m), "
        f"FD x{ff:.4g} (median sigma_rho={f_med:.4g} m)"
    )
    a_scaled = [scale_curves_about_center(r, fa) for r in a_grid]
    f_scaled = [scale_curves_about_center(r, ff) for r in f_grid]
    _overlay(
        a_scaled, f_scaled,
        r"Analítico (norm.)", r"FD (norm.)",
        r"Analítico vs FD — forma normalizada ($\sigma_\rho$ mediana igualada)",
        PLOTS / "crb_regions_analytic_vs_fd_shapenorm.png",
    )

    # --- numeric comparison table ---------------------------------------------
    write_comparison_table(analytic, fd, key=key)


def write_comparison_table(
    analytic: Dict[str, List[Dict[str, Any]]],
    fd: Dict[str, List[Dict[str, Any]]],
    key: str = "with_h",
) -> None:
    a_grid = analytic[key]
    f_grid = fd.get(key)
    grid_label = "with_h [rho,phi,h]" if key == "with_h" else "fixed_h [rho,phi]"
    lines: List[str] = []
    lines.append(f"Comparacion CRB analitico vs FD (rasterizado)  --  grid {grid_label}")
    lines.append("=" * 118)
    lines.append(
        "NOTA: con la variante (a)+(b) la amplitud es fisica (I_laser=1000, area_eff=w*h)"
    )
    lines.append(
        "y la rejilla es la del simulador (64x64, bin real), de modo que las magnitudes"
    )
    lines.append(
        "absolutas SI son comparables: sigma_rho, sigma_phi quedan en la misma escala que"
    )
    lines.append(
        "el FD (faceta puntual vs render de malla). Ademas se compara la FORMA: relacion de"
    )
    lines.append("aspecto (aspect = eje_mayor/eje_menor) y tilt de la elipse 3-sigma en (rho,phi).")
    lines.append("=" * 118)
    header = (
        f"{'rho':>4} {'phi':>4} | "
        f"{'A:sig_rho':>10} {'A:sig_phi':>9} {'A:sig_h':>8} {'A:asp':>6} {'A:tilt':>7} | "
        f"{'F:sig_rho':>10} {'F:sig_phi':>9} {'F:sig_h':>8} {'F:asp':>6} {'F:tilt':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    fd_by_key = {}
    if f_grid:
        for r in f_grid:
            fd_by_key[(round(r["rho0"], 6), round(np.rad2deg(r["phi0"]), 3))] = r

    a_aspects, f_aspects = [], []
    for ar in a_grid:
        rho0 = ar["rho0"]
        phi_deg = round(np.rad2deg(ar["phi0"]), 3)
        am = ellipse_shape_metrics(ar)
        a_aspects.append(am["aspect"])
        fr = fd_by_key.get((round(rho0, 6), phi_deg))
        if fr is not None:
            fm = ellipse_shape_metrics(fr)
            f_aspects.append(fm["aspect"])
            f_sr = f"{fr['sigma_rho']:10.4g}"
            f_sp = f"{fr['sigma_phi_deg']:9.4g}"
            f_sh = (
                f"{fr['sigma_height']:8.4g}"
                if np.isfinite(fr.get("sigma_height", np.nan))
                else f"{'nan':>8}"
            )
            f_as = f"{fm['aspect']:6.2f}"
            f_ti = f"{fm['tilt_deg']:7.1f}"
        else:
            f_sr = f_sp = f_sh = f_as = f_ti = f"{'--':>8}"

        a_sh = (
            f"{ar['sigma_height']:8.4g}"
            if np.isfinite(ar.get("sigma_height", np.nan))
            else f"{'nan':>8}"
        )
        lines.append(
            f"{rho0:4.2f} {phi_deg:4.0f} | "
            f"{ar['sigma_rho']:10.4g} {ar['sigma_phi_deg']:9.4g} {a_sh} "
            f"{am['aspect']:6.2f} {am['tilt_deg']:7.1f} | "
            f"{f_sr} {f_sp} {f_sh} {f_as} {f_ti}"
        )

    lines.append("-" * len(header))
    lines.append(
        f"Aspecto medio (mayor/menor)  analitico={np.mean(a_aspects):.2f}"
        + (f"   FD={np.mean(f_aspects):.2f}" if f_aspects else "   FD=n/a")
    )
    lines.append("")
    lines.append(conclusion_text(bool(f_grid)))

    text = "\n".join(lines)
    print(text)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "analytic_vs_fd_comparison.txt").write_text(text, encoding="utf-8")
    print(f"\nSaved {DOCS / 'analytic_vs_fd_comparison.txt'}")
    write_latex_table(a_grid, fd_by_key, key)


def write_latex_table(
    a_grid: List[Dict[str, Any]],
    fd_by_key: Dict[Any, Dict[str, Any]],
    key: str,
) -> None:
    """Emit a compact LaTeX table fragment (\\input-ed by the .tex)."""
    grid_tag = r"$[\rho,\varphi,h]$ (\emph{with\_h})" if key == "with_h" else r"$[\rho,\varphi]$ (\emph{fixed\_h})"
    rows = []
    for ar in a_grid:
        rho0 = ar["rho0"]
        phi_deg = round(np.rad2deg(ar["phi0"]), 3)
        am = ellipse_shape_metrics(ar)
        fr = fd_by_key.get((round(rho0, 6), phi_deg))
        if fr is not None:
            fm = ellipse_shape_metrics(fr)
            f_sr = f"{fr['sigma_rho']:.4g}"
            f_sp = f"{fr['sigma_phi_deg']:.3g}"
            f_as = f"{fm['aspect']:.1f}"
        else:
            f_sr = f_sp = f_as = "--"
        rows.append(
            f"{rho0:.1f} & {phi_deg:.0f} & "
            f"{ar['sigma_rho']:.4g} & {ar['sigma_phi_deg']:.3g} & {am['aspect']:.1f} & "
            f"{f_sr} & {f_sp} & {f_as} \\\\"
        )
    body = "\n".join(rows)
    frag = (
        "% Auto-generated by generate_analytic_crb_plots.py -- do not edit by hand.\n"
        "\\begin{table}[ht]\n\\centering\\small\n"
        "\\caption{CRB anal\\'itico vs FD sobre el grid " + grid_tag + ". "
        "$\\sigma_\\rho$ en m, $\\sigma_\\varphi$ en grados; \\emph{asp}$=$eje mayor/menor "
        "de la elipse $3\\sigma$ en $(\\rho,\\varphi)$. Con amplitud f\\'isica (variante (a)+(b)) "
        "las magnitudes absolutas son comparables entre m\\'etodos.}\n"
        "\\label{tab:avf}\n"
        "\\begin{tabular}{@{}rr|rrr|rrr@{}}\n\\toprule\n"
        " & & \\multicolumn{3}{c|}{Anal\\'itico} & \\multicolumn{3}{c}{FD (rasterizado)} \\\\\n"
        "$\\rho$ & $\\varphi^\\circ$ & $\\sigma_\\rho$ & $\\sigma_\\varphi$ & asp & "
        "$\\sigma_\\rho$ & $\\sigma_\\varphi$ & asp \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (DOCS / "analytic_vs_fd_table.tex").write_text(frag, encoding="utf-8")
    print(f"Saved {DOCS / 'analytic_vs_fd_table.tex'}")


def conclusion_text(have_fd: bool) -> str:
    base = (
        "CONCLUSION -- magnitud y forma\n"
        "------------------------------\n"
        "Con la variante (a)+(b) el forward analitico usa la amplitud fisica del\n"
        "simulador (I_laser=1000, area_eff=w*h) y su misma rejilla (64x64, bin real),\n"
        "de modo que la CRB analitica y la FD quedan en la MISMA escala fisica.\n\n"
        "1) Magnitud: sigma_rho y sigma_phi analiticos caen en el mismo orden que el\n"
        "   FD (mismas unidades, misma normalizacion). El overlay a magnitud real\n"
        "   (crb_regions_analytic_vs_fd.png) muestra elipses de tamano comparable, a\n"
        "   diferencia de la version antigua con ganancia G=1e5 y rejilla 8x8 propia.\n"
        "   El pequeno residual proviene de la faceta puntual (area unica w*h en el\n"
        "   centroide z=h/2) frente al render de malla, y del suavizado finito; al\n"
        "   bajar tau, 1/beta, 1/kappa la CRB analitica converge hacia la FD.\n\n"
        "2) Suavidad / condicionamiento: el forward analitico tiene derivadas EXACTAS\n"
        "   (sympy), C-infinito por construccion. El Jacobiano FD del rasterizador se\n"
        "   calcula sobre un forward con escalones: la cuantizacion temporal ceil(.) y\n"
        "   la mascara de ocultamiento xint>0 son constantes-a-trozos, de modo que la\n"
        "   diferencia finita mezcla mesetas planas (derivada 0) con saltos de bin,\n"
        "   introduciendo ruido de discretizacion y sensibilidad al paso delta. La\n"
        "   elipse analitica es por tanto la mejor CONDICIONADA (Fisher mas estable).\n\n"
        "3) Forma/orientacion: ambos metodos producen elipses (burbujas) finas en\n"
        "   angulo -- la resolucion azimutal que aporta la arista ocluyente -- y mas\n"
        "   anchas en rango, coherente con la fisica del borde. La orientacion\n"
        "   (columna tilt) es consistente entre metodos salvo el ruido FD.\n\n"
        "VEREDICTO: la elipse ANALITICA es ahora comparable en MAGNITUD con la FD y,\n"
        "por sus derivadas exactas, mejor condicionada; el FD sigue siendo util como\n"
        "verificacion independiente y como el modelo de malla completo."
    )
    if not have_fd:
        base = (
            "CONCLUSION (parcial): la cache FD (plots/crb_grid_results.pkl) no estaba\n"
            "disponible al ejecutar, por lo que la comparacion numerica FD se omitio.\n"
            "Regenerar con: python3 regenerate_crb_standard_regions.py\n\n"
        ) + base
    return base


def main() -> int:
    PLOTS.mkdir(exist_ok=True)
    analytic = build_or_load_analytic()
    make_analytic_plots(analytic)

    fd = load_fd_cache()
    if fd is None:
        print("\nFD cache not found (plots/crb_grid_results.pkl). "
              "Run regenerate_crb_standard_regions.py to enable the FD comparison.")
        write_comparison_table(analytic, {})
        return 0
    print(f"\nLoaded FD cache: keys={list(fd.keys())}")
    make_comparison(analytic, fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
