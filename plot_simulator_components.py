#!/usr/bin/env python3
"""Plots de las funciones componentes de los simuladores de transitorios.

Dos backends, seleccionables con ``--backend``:

  * ``cpu``: réplica *fiel* de la función ``simulation(...)`` ORIGINAL (numpy,
    bucle por triángulo), incluidas sus rarezas (theta = -cos, índice ``-1``,
    ``y[coord] += intensity``, sin máscaras ni bounds).
  * ``gpu``: réplica en numpy de la matemática de ``simulator.py``
    (``_transform_object_mesh`` + ``simulate_transient_torch``): theta = phi+3π/2,
    máscara ``eps`` en el denominador, ``clamp``, bounds en el bin, índice
    correcto, ``index_add_`` y ``orient_transient_measurement`` (roll +1).
    Incluye un sanity check contra la salida real de ``simulator.simulation``.

Para la misma pose (rho=1, phi=60°, N=32, FOV=0.25, facet.obj, w=0.5) genera:

  * mapas N×N de cada función componente (oclusión, geometría, cosenos,
    binning, intensidad) evaluados para el centroide de la faceta,
  * el transitorio completo (bucle sobre todos los triángulos de la malla),
  * las funciones "duras" (max(0,·), ceil, Heaviside) que aparecen en el código,
  * un txt de revisión numérica (para ``cpu``: la demostración de los bugs).

Salida en ``plots/crb_sim_components_*`` (cpu) y ``plots/crb_sim_components_gpu_*`` (gpu).
Las funciones de este módulo las reutiliza ``plot_gpu_vs_cpu.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import trimesh  # noqa: E402
from trimesh.transformations import rotation_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from utils.densify_mesh import densify_mesh_if_needed  # noqa: E402

PLOTS_DIR = ROOT / "plots"
DPI = 200

BACKENDS = {
    "cpu": dict(prefix="crb_sim_components_", label="CPU (numpy original)"),
    "gpu": dict(prefix="crb_sim_components_gpu_", label="GPU (simulator.py)"),
}

# --------------------------------------------------------------------------
# Parámetros (los del código original / de la tarea)
# --------------------------------------------------------------------------
c = 299792458.0  # global implícito en el código original
object_folder = "objects"  # global implícito
ymin = 0.0  # global implícito
zmin = 0.0  # global implícito

xmin, xmax, ymax, zmax = -1.5, 1.5, 3.0, 3.0
camera_FOV = 0.25
cam_pixel_dim = 32
bin_size = 3.9e-10
laser_intensity = 1000.0
hide_walls = True
MESH_MIN_TRIANGLES = 10000
EPS = 1e-12  # eps de simulate_transient_torch

RHO = 1.0
PHI_DEG = 60.0
PHI = np.deg2rad(PHI_DEG)
FACET_W = 0.5

laser_pos = np.array([0.0, 0.0, 0.0])
laser_normal = np.array([0.0, 0.0, 1.0])
floor_normal = np.array([0.0, 0.0, 1.0])
fourpi = 4 * np.pi * np.pi


# --------------------------------------------------------------------------
# Placement de la faceta
# --------------------------------------------------------------------------
def place_facet_original(xcoord, ycoord, zcoord, w, pitch=1.57, roll=0.0, yaw=0.0,
                         min_triangles=MESH_MIN_TRIANGLES):
    """Replica el bloque ``for obj_data in object_positions`` del código original.

    Devuelve (mesh, theta).  ``yaw`` se acepta pero NO se usa: exactamente como en
    el original, la rotación en z usa ``theta = -cos(∠(u, v1))``.
    """
    u = np.array([1, 0, 0])
    v1 = np.array([xcoord, ycoord, zcoord])
    # Coseno del ángulo usado *como ángulo* (rareza del original)
    theta = -np.clip(np.dot(u, v1) / (np.linalg.norm(u) * np.linalg.norm(v1)), -1, 1)

    obj = trimesh.load(os.path.join(object_folder, "facet.obj"), force="mesh")
    if isinstance(obj, trimesh.Scene):
        obj = obj.dump(concatenate=True)
    obj = densify_mesh_if_needed(obj, min_triangles=min_triangles)

    obj_extents = obj.extents
    scale_factors = [w / obj_extents[0], 1.1 / obj_extents[2]]  # sin 'h'; ext_z=0 -> inf
    scale_factor = min(scale_factors)
    obj.apply_scale(scale_factor)

    obj.apply_transform(rotation_matrix(pitch, [1, 0, 0]))
    obj.apply_transform(rotation_matrix(roll, [0, 1, 0]))
    obj.apply_transform(rotation_matrix(theta, [0, 0, 1]))

    z_min = obj.vertices[:, 2].min()
    obj.apply_translation([0, 0, -z_min])
    obj.apply_translation(v1)
    return obj, theta


def place_facet_gpu(rho, phi, w, pitch=1.57, roll=0.0, min_triangles=MESH_MIN_TRIANGLES):
    """Replica ``simulator._transform_object_mesh`` (theta = phi + 3π/2)."""
    xcoord, ycoord, zcoord = float(rho * np.cos(phi)), float(rho * np.sin(phi)), 0.0
    v1 = np.array([xcoord, ycoord, zcoord])

    obj = trimesh.load(os.path.join(object_folder, "facet.obj"), force="mesh")
    if isinstance(obj, trimesh.Scene):
        obj = obj.dump(concatenate=True)
    obj = obj.copy()
    obj = densify_mesh_if_needed(obj, min_triangles=min_triangles)

    obj_extents = obj.extents
    scale_factors = []
    if obj_extents[0] > 1e-12:
        scale_factors.append(w / obj_extents[0])
    if obj_extents[2] > 1e-12:
        scale_factors.append(1.1 / obj_extents[2])
    obj.apply_scale(min(scale_factors))

    obj.apply_transform(rotation_matrix(pitch, [1, 0, 0]))
    obj.apply_transform(rotation_matrix(roll, [0, 1, 0]))
    theta = float(phi) + 3 * np.pi / 2
    obj.apply_transform(rotation_matrix(theta, [0, 0, 1]))

    z_min = obj.vertices[:, 2].min()
    obj.apply_translation([0, 0, -z_min])
    obj.apply_translation(v1)
    return obj, theta


def place_facet(rho, phi, w, backend):
    if backend == "cpu":
        return place_facet_original(rho * np.cos(phi), rho * np.sin(phi), 0.0, w)
    return place_facet_gpu(rho, phi, w)


def facet_summary(mesh):
    """Centroide (área-ponderado), normal media unitaria y área total de la malla."""
    areas = mesh.area_faces
    centroids = mesh.triangles_center
    center = (centroids * areas[:, None]).sum(0) / areas.sum()
    normal = mesh.face_normals.mean(0)
    normal /= np.linalg.norm(normal)
    return center, normal, areas.sum()


# --------------------------------------------------------------------------
# Grilla SPAD y número de bins (idénticos en ambos backends)
# --------------------------------------------------------------------------
def spad_grid(N=cam_pixel_dim, FOV=camera_FOV):
    camera_FOV_center = [0.0, -FOV / 2, 0.0]
    pixel_x = np.linspace(camera_FOV_center[0] - FOV / 2 + FOV / (2 * N),
                          camera_FOV_center[0] + FOV / 2 - FOV / (2 * N), N)
    pixel_y = np.linspace(camera_FOV_center[1] - FOV / 2 + FOV / (2 * N),
                          camera_FOV_center[1] + FOV / 2 - FOV / (2 * N), N)
    X, Y = np.meshgrid(pixel_x, pixel_y)
    cam_pos = np.vstack([X.ravel(), Y.ravel(), np.zeros(N**2)]).T
    X_ind, Y_ind = np.meshgrid(range(N), range(N), indexing="xy")
    cam_pos_ind = np.vstack([X_ind.ravel(), Y_ind.ravel()]).T
    return pixel_x, pixel_y, cam_pos, cam_pos_ind


def num_time_bins_original(FOV=camera_FOV):
    furthest_scene_point = np.array([xmax, ymax, zmax])
    furthest_spad_point = np.array([-FOV / 2, -FOV, 0.0])
    d1 = np.linalg.norm(furthest_scene_point - laser_pos)
    d2 = np.linalg.norm(furthest_spad_point - furthest_scene_point)
    max_dist_travel = d1 + d2
    return int(np.ceil(max_dist_travel / c / bin_size + 0.2 * max_dist_travel / c / bin_size)), max_dist_travel


# --------------------------------------------------------------------------
# Matemática por-triángulo / por-píxel
# --------------------------------------------------------------------------
def triangle_terms(scene_center, normal, area, cam_pos, backend="cpu", num_bins=None):
    """Evalúa TODAS las funciones componentes para un triángulo sobre todos los
    píxeles (sin aplicar la máscara ``noc`` a los arrays, para poder mapearlas).

    ``backend='cpu'``: transcripción del original.  ``backend='gpu'``: transcripción
    de ``simulate_transient_torch`` (máscara eps, isfinite, clamp, bounds de bin).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        if backend == "cpu":
            m = (scene_center[1] - cam_pos[:, 1]) / (scene_center[0] - cam_pos[:, 0])  # sin máscara denom=0
            b = scene_center[1] - np.dot(m, scene_center[0])
            xint = -b / m  # sin máscara m=0
            noc = xint > 0  # Heaviside duro
        else:
            denom = scene_center[0] - cam_pos[:, 0]
            valid_denom = np.abs(denom) > EPS
            m = np.where(valid_denom, (scene_center[1] - cam_pos[:, 1]) / denom, np.nan)
            b = scene_center[1] - m * scene_center[0]
            xint = -b / m
            noc = np.isfinite(xint) & (xint > 0)

    lps = laser_pos - scene_center
    fovsp = cam_pos - scene_center
    d1s = np.sum(lps**2)
    d2s = np.sum(fovsp**2, axis=1)
    if backend == "cpu":
        d1 = np.sqrt(d1s)
        d2 = np.sqrt(d2s)
    else:
        d1 = np.sqrt(np.maximum(d1s, EPS))
        d2 = np.sqrt(np.maximum(d2s, EPS))
    distance = d1 + d2
    tbin = distance / (c * bin_size)
    arrival_bin = np.ceil(tbin).astype(int)  # binning duro

    dot1_raw = np.sum(np.dot(normal, lps / d1))
    dot2_raw = np.sum(normal * fovsp / d2[:, np.newaxis], axis=1)
    dot3_raw = np.sum(np.dot(laser_normal, -lps / d1))
    dot4_raw = np.sum(floor_normal * -fovsp / d2[:, np.newaxis], axis=1)
    dot1 = np.maximum(0, dot1_raw)
    dot2 = np.maximum(0, dot2_raw)
    dot3 = np.maximum(0, dot3_raw)
    dot4 = np.maximum(0, dot4_raw)

    if backend == "cpu":
        intensity = laser_intensity * area * (dot1 * dot2 * dot3 * dot4) / (fourpi * d1s * d2s)
        valid = noc
    else:
        intensity = laser_intensity * area * (dot1 * dot2 * dot3 * dot4) / (fourpi * d1s * np.maximum(d2s, EPS))
        valid = noc & (arrival_bin > 0)
        if num_bins is not None:
            valid &= arrival_bin <= num_bins
    return dict(m=m, b=b, xint=xint, noc=noc, valid=valid, d1=d1, d2=d2, d1s=d1s, d2s=d2s,
                distance=distance, tbin=tbin, arrival_bin=arrival_bin,
                dot1=dot1, dot2=dot2, dot3=dot3, dot4=dot4,
                dot1_raw=dot1_raw, dot2_raw=dot2_raw, dot3_raw=dot3_raw, dot4_raw=dot4_raw,
                intensity=intensity, atten=1.0 / (d1s * d2s))


def simulate_original_loop(mesh, cam_pos, cam_pos_ind, N, num_bins, verbose=True):
    """Bucle fiel sobre todos los triángulos (backend cpu).  Devuelve

    y_orig  : reshape del vector acumulado EXACTAMENTE como el original
              (coord con ``-1`` en el índice x y ``y[coord] += intensity``),
    y_fixed : misma física pero con el índice corregido y ``np.add.at``
              (el "IMPORTANT FIX" de utils/crb_fix.py),
    stats   : diagnósticos (coords negativos, duplicados, bins máximos, ...).
    """
    triangles = mesh.triangles
    triangle_normals = mesh.face_normals
    y_orig = np.zeros(N * N * num_bins)
    y_fixed = np.zeros(N * N * num_bins)
    stats = dict(n_tri=len(triangles), n_coord_neg=0, n_coord_wrap=0, n_dup_within_tri=0,
                 max_bin=0, min_bin=10**9, n_pixels_hit=0, lost_pluseq=0.0,
                 n_out_of_range_fixed=0)
    for idx in range(len(triangles)):
        triangle = triangles[idx]
        normal = triangle_normals[idx]
        area = 0.5 * np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        scene_center = triangle.mean(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            m = (scene_center[1] - cam_pos[:, 1]) / (scene_center[0] - cam_pos[:, 0])
            b = scene_center[1] - np.dot(m, scene_center[0])
            xint = -b / m
        noc = xint > 0
        if np.sum(noc) > 0:
            lps = laser_pos - scene_center
            fovsp = cam_pos[noc, :] - scene_center
            d1s = np.sum(lps**2)
            d2s = np.sum(fovsp**2, axis=1)
            d1 = np.sqrt(d1s)
            d2 = np.sqrt(d2s)
            distance = d1 + d2
            tbin = distance / (c * bin_size)
            arrival_bin = np.ceil(tbin).astype(int)

            dot1 = np.maximum(0, np.sum(np.dot(normal, lps / d1)))
            dot2 = np.maximum(0, np.sum(normal * fovsp / d2[:, np.newaxis], axis=1))
            dot3 = np.maximum(0, np.sum(np.dot(laser_normal, -lps / d1)))
            dot4 = np.maximum(0, np.sum(floor_normal * -fovsp / d2[:, np.newaxis], axis=1))
            intensity = laser_intensity * area * (dot1 * dot2 * dot3 * dot4) / (fourpi * d1s * d2s)

            # ---- coord ORIGINAL (con el "-1" en el índice x) ----
            coord = (arrival_bin - 1) * N**2 + (cam_pos_ind[noc, 0] - 1) * N + cam_pos_ind[noc, 1]
            stats["n_coord_neg"] += int(np.sum(coord < 0))
            stats["n_coord_wrap"] += int(np.sum(cam_pos_ind[noc, 0] == 0))
            n_unique = len(np.unique(coord))
            stats["n_dup_within_tri"] += len(coord) - n_unique
            # sumo lo que el "+=" deja caer (comparado con add.at) en este triángulo
            tmp_a = np.zeros(N * N * num_bins)
            tmp_a[coord] += intensity
            stats["lost_pluseq"] += float(intensity.sum() - tmp_a.sum())
            y_orig[coord] += intensity  # <- exactamente como el original

            # ---- coord CORREGIDO (sin "-1", np.add.at, con bounds) ----
            arrival_bin_idx = arrival_bin - 1
            valid = (arrival_bin_idx >= 0) & (arrival_bin_idx < num_bins)
            stats["n_out_of_range_fixed"] += int(np.sum(~valid))
            coord_fixed = (cam_pos_ind[noc, 1][valid] + cam_pos_ind[noc, 0][valid] * N
                           + arrival_bin_idx[valid] * N**2)
            np.add.at(y_fixed, coord_fixed, intensity[valid])

            stats["max_bin"] = max(stats["max_bin"], int(arrival_bin.max()))
            stats["min_bin"] = min(stats["min_bin"], int(arrival_bin.min()))
        if verbose and idx % 8192 == 0:
            print(f"  triángulo {idx}/{len(triangles)}")

    y_orig = y_orig.reshape((N, N, num_bins), order="F")
    y_fixed = y_fixed.reshape((N, N, num_bins), order="F")
    return y_orig, y_fixed, stats


def simulate_gpu_replica_loop(mesh, cam_pos, cam_pos_ind, N, num_bins, verbose=True):
    """Réplica en numpy de ``simulate_transient_torch`` (bucle por triángulo en vez
    de chunks; la matemática por (triángulo, píxel) es la misma).  Devuelve

    y_oriented : salida final del simulador, es decir tras ``orient_transient_measurement``
                 (np.roll(shift=1, axis=1)),
    y_raw      : el reshape (N, N, T) order='F' antes del roll,
    stats      : diagnósticos.
    """
    triangles = mesh.triangles
    triangle_normals = mesh.face_normals
    num_pixels = N * N
    pixel_idx = np.arange(num_pixels)
    pixel_x_idx = pixel_idx % N
    pixel_y_idx = pixel_idx // N
    pixel_linear = pixel_x_idx * N + pixel_y_idx
    assert np.array_equal(pixel_x_idx, cam_pos_ind[:, 0]) and np.array_equal(pixel_y_idx, cam_pos_ind[:, 1])

    y_meas_vec = np.zeros(num_pixels * num_bins)
    stats = dict(n_tri=len(triangles), n_invalid_denom=0, n_nonfinite_xint=0, n_bin_out_of_range=0,
                 n_dup_within_tri=0, max_bin=0, min_bin=10**9, n_clamped_dot2=0, n_clamped_dot4=0,
                 n_tri_skipped_noc=0)
    for idx in range(len(triangles)):
        triangle = triangles[idx]
        normal = triangle_normals[idx]
        area = 0.5 * np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        center = triangle.mean(axis=0)
        cx, cy = center[0], center[1]

        with np.errstate(divide="ignore", invalid="ignore"):
            denom = cx - cam_pos[:, 0]
            valid_denom = np.abs(denom) > EPS
            m = np.where(valid_denom, (cy - cam_pos[:, 1]) / denom, np.nan)
            b = cy - m * cx
            xint = -b / m
        noc = np.isfinite(xint) & (xint > 0)
        stats["n_invalid_denom"] += int(np.sum(~valid_denom))
        stats["n_nonfinite_xint"] += int(np.sum(~np.isfinite(xint)))
        if not np.any(noc):
            stats["n_tri_skipped_noc"] += 1
            continue

        lps = laser_pos - center
        d1s = np.sum(lps * lps)
        d1 = np.sqrt(max(d1s, EPS))
        fovsp = cam_pos - center
        d2s = np.sum(fovsp * fovsp, axis=1)
        d2 = np.sqrt(np.maximum(d2s, EPS))

        arrival_bin = np.ceil((d1 + d2) / (c * bin_size)).astype(np.int64)
        valid = noc & (arrival_bin > 0) & (arrival_bin <= num_bins)
        stats["n_bin_out_of_range"] += int(np.sum(noc & ~valid))
        if not np.any(valid):
            continue

        dot2_raw = np.sum(normal * (fovsp / d2[:, None]), axis=1)
        dot4_raw = np.sum(floor_normal * (-fovsp / d2[:, None]), axis=1)
        dot1 = max(np.sum(normal * (lps / d1)), 0.0)
        dot2 = np.maximum(dot2_raw, 0.0)
        dot3 = max(np.sum(laser_normal * (-lps / d1)), 0.0)
        dot4 = np.maximum(dot4_raw, 0.0)
        stats["n_clamped_dot2"] += int(np.sum(valid & (dot2_raw < 0)))
        stats["n_clamped_dot4"] += int(np.sum(valid & (dot4_raw < 0)))

        intensity = laser_intensity * area * (dot1 * dot2 * dot3 * dot4) / (fourpi * d1s * np.maximum(d2s, EPS))
        intensity = np.where(valid, intensity, 0.0)

        pix = np.nonzero(valid)[0]
        coord = (arrival_bin[pix] - 1) * num_pixels + pixel_linear[pix]
        stats["n_dup_within_tri"] += len(coord) - len(np.unique(coord))
        np.add.at(y_meas_vec, coord, intensity[pix])  # == index_add_

        stats["max_bin"] = max(stats["max_bin"], int(arrival_bin[pix].max()))
        stats["min_bin"] = min(stats["min_bin"], int(arrival_bin[pix].min()))
        if verbose and idx % 8192 == 0:
            print(f"  triángulo {idx}/{len(triangles)}")

    y_raw = y_meas_vec.reshape((N, N, num_bins), order="F")
    y_oriented = np.roll(y_raw, shift=1, axis=1)  # orient_transient_measurement
    return y_oriented, y_raw, stats


def run_real_simulator(rho, phi, w, N=cam_pixel_dim, FOV=camera_FOV):
    """Llama a ``simulator.simulation`` (torch, CPU, float64) con la misma pose."""
    import torch
    import simulator

    object_positions = [dict(obj_file="facet.obj", rho=float(rho), phi=float(phi), w=float(w),
                             pitch=1.57, roll=0.0)]
    y, params = simulator.simulation(
        xmin, xmax, ymin, ymax, zmin, zmax, FOV, N, bin_size, laser_intensity,
        object_positions, hide_walls=True, SNR_dB=30.0, SBR=5.0, poisson_scale_factor=1.0,
        add_noise=False, torch_device=torch.device("cpu"), torch_dtype=torch.float64,
    )
    mesh_real = simulator.build_object_mesh(object_positions[0])
    return y, params, mesh_real


# --------------------------------------------------------------------------
# Helpers de plots
# --------------------------------------------------------------------------
def to_img(vec, N=cam_pixel_dim):
    """Vector por-píxel (orden ravel de meshgrid: fila=iy, col=jx) -> imagen N×N."""
    return np.asarray(vec, dtype=float).reshape(N, N)


def imshow_floor(ax, img, pixel_x, pixel_y, title, cmap="viridis", cbar_label=None, **kw):
    dx = pixel_x[1] - pixel_x[0]
    dy = pixel_y[1] - pixel_y[0]
    extent = [pixel_x[0] - dx / 2, pixel_x[-1] + dx / 2, pixel_y[0] - dy / 2, pixel_y[-1] + dy / 2]
    im = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, aspect="equal", **kw)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x piso [m] (columna jx)")
    ax.set_ylabel("y piso [m] (fila iy)")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cb.set_label(cbar_label)
    return im


def pose_str(rho=RHO, phi_deg=PHI_DEG):
    return rf"$\rho$={rho:.2f} m, $\varphi$={phi_deg:.0f}°"


def plot_hard_functions(axs):
    """Los 3 paneles 1D de las funciones duras (idénticos en ambos backends)."""
    x = np.linspace(-1.5, 1.5, 601)
    axs[0].plot(x, np.maximum(0, x), "C0")
    axs[0].plot([0], [0], "ro", label="no derivable en x=0")
    axs[0].set_title("max(0, x)  — clamp de los cosenos dot1…dot4", fontsize=9)
    axs[0].set_xlabel("x = n·v"); axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)
    x2 = np.linspace(0, 5, 1001)
    axs[1].plot(x2, x2, "C0--", lw=1, label="x (continuo)")
    axs[1].step(x2, np.ceil(x2), where="post", color="C3", label="ceil(x)")
    ks = np.arange(0, 6)
    axs[1].plot(ks, ks, "ro", label="saltos: no derivable, derivada 0 en el resto")
    axs[1].set_title("ceil(x)  — arrival_bin = ceil(distance/(cΔt))", fontsize=9)
    axs[1].set_xlabel("distance/(cΔt)"); axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
    x3 = np.linspace(-1, 1, 1001)
    axs[2].step(x3, (x3 > 0).astype(float), where="post", color="C2", label="1{xint>0}")
    axs[2].plot([0], [0.5], "ro", label="discontinuidad en xint=0")
    axs[2].set_ylim(-0.1, 1.1)
    axs[2].set_title("1{xint > 0}  — Heaviside de oclusión (noc)", fontsize=9)
    axs[2].set_xlabel("xint [m]"); axs[2].legend(fontsize=8); axs[2].grid(alpha=0.3)


# --------------------------------------------------------------------------
# Evaluación completa de un backend para una pose
# --------------------------------------------------------------------------
def evaluate_backend(backend, rho=RHO, phi=PHI, w=FACET_W, N=cam_pixel_dim, run_transient=True, verbose=True):
    """Devuelve un dict con la malla, el triángulo representativo, sus términos por
    píxel y (opcionalmente) el transitorio completo del backend."""
    pixel_x, pixel_y, cam_pos, cam_pos_ind = spad_grid(N)
    num_bins, max_dist_travel = num_time_bins_original()
    mesh, theta = place_facet(rho, phi, w, backend)
    center, normal, area = facet_summary(mesh)
    T = triangle_terms(center, normal, area, cam_pos, backend=backend, num_bins=num_bins)
    out = dict(backend=backend, rho=rho, phi=phi, w=w, N=N, pixel_x=pixel_x, pixel_y=pixel_y,
               cam_pos=cam_pos, cam_pos_ind=cam_pos_ind, num_bins=num_bins, max_dist_travel=max_dist_travel,
               mesh=mesh, theta=theta, center=center, normal=normal, area=area, T=T)
    if run_transient:
        if backend == "cpu":
            y, y_alt, stats = simulate_original_loop(mesh, cam_pos, cam_pos_ind, N, num_bins, verbose=verbose)
        else:
            y, y_alt, stats = simulate_gpu_replica_loop(mesh, cam_pos, cam_pos_ind, N, num_bins, verbose=verbose)
        out.update(y=y, y_alt=y_alt, stats=stats)
    return out


# --------------------------------------------------------------------------
# Los 7 plots de componentes
# --------------------------------------------------------------------------
def make_component_plots(E, prefix, log):
    backend = E["backend"]
    N = E["N"]
    pixel_x, pixel_y = E["pixel_x"], E["pixel_y"]
    T = E["T"]
    num_bins = E["num_bins"]
    is_cpu = backend == "cpu"
    tag = "" if is_cpu else "  [GPU]"

    # ============ Plot 1: oclusión ============
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    xint_img = to_img(T["xint"])
    vmax = np.nanmax(np.abs(xint_img))
    imshow_floor(axs[0], xint_img, pixel_x, pixel_y,
                 "xint = -b/m  (cruce de la recta píxel→centroide con y=0)",
                 cmap="RdBu_r", cbar_label="xint [m]", vmin=-vmax, vmax=vmax)
    Xg, Yg = np.meshgrid(pixel_x, pixel_y)
    axs[0].contour(Xg, Yg, xint_img, levels=[0.0], colors="k", linewidths=1.5)
    axs[0].plot([], [], "k-", label="xint = 0 (borde del oclusor)")
    axs[0].legend(loc="lower right", fontsize=7)
    noc_title = (f"noc = 1{{xint>0}}  ({T['noc'].sum()} px ven la faceta)" if is_cpu else
                 f"noc = isfinite(xint) & (xint>0)  ({T['noc'].sum()} px ven la faceta)")
    imshow_floor(axs[1], to_img(T["noc"]).astype(float), pixel_x, pixel_y, noc_title,
                 cmap="gray", cbar_label="noc", vmin=0, vmax=1)
    occ_sup = ("Oclusión: xint y noc (Heaviside duro)" if is_cpu else
               "Oclusión: xint y noc (Heaviside duro, máscara |denom|>eps)")
    fig.suptitle(f"{occ_sup} — {pose_str()}{tag}", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{prefix}occlusion.png", dpi=DPI)
    plt.close(fig)

    # ============ Plot 2: geometría ============
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    imshow_floor(axs[0], to_img(T["d2"]), pixel_x, pixel_y,
                 "d2 = |cam_pos − scene_center|" + ("" if is_cpu else "  (clamp min=eps)"), cbar_label="[m]")
    imshow_floor(axs[1], to_img(T["atten"]), pixel_x, pixel_y,
                 f"1/(d1²·d2²)   (d1 = {T['d1']:.4f} m fijo)", cmap="magma", cbar_label="[m⁻⁴]")
    imshow_floor(axs[2], to_img(T["distance"]), pixel_x, pixel_y, "distance = d1 + d2", cbar_label="[m]")
    fig.suptitle(f"Geometría radiativa por píxel (centroide de la faceta) — {pose_str()}{tag}", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{prefix}geometry.png", dpi=DPI)
    plt.close(fig)

    # ============ Plot 3: cosenos ============
    clamp_lbl = "con clamp" if is_cpu else "clamp(min=0)"
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    imshow_floor(axs[0, 0], to_img(T["dot2"]), pixel_x, pixel_y,
                 f"dot2 = max(0, n·(fovsp/d2))  [{clamp_lbl}]", cbar_label="cos")
    imshow_floor(axs[0, 1], to_img(T["dot2_raw"]), pixel_x, pixel_y,
                 f"n·(fovsp/d2) SIN clamp  (min={T['dot2_raw'].min():.3f}, px<0: {(T['dot2_raw']<0).sum()})",
                 cmap="RdBu_r", cbar_label="cos",
                 vmin=-np.abs(T["dot2_raw"]).max(), vmax=np.abs(T["dot2_raw"]).max())
    imshow_floor(axs[1, 0], to_img(T["dot4"]), pixel_x, pixel_y,
                 f"dot4 = max(0, n_floor·(−fovsp/d2))  [{clamp_lbl}]", cbar_label="cos")
    imshow_floor(axs[1, 1], to_img(T["dot4_raw"]), pixel_x, pixel_y,
                 f"n_floor·(−fovsp/d2) SIN clamp  (min={T['dot4_raw'].min():.3f}, px<0: {(T['dot4_raw']<0).sum()})",
                 cmap="RdBu_r", cbar_label="cos",
                 vmin=-np.abs(T["dot4_raw"]).max(), vmax=np.abs(T["dot4_raw"]).max())
    fig.suptitle(f"Cosenos por píxel — {pose_str()}{tag}   |   escalares: dot1 = {T['dot1']:.4f}, "
                 f"dot3 = {T['dot3']:.4f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{prefix}cosines.png", dpi=DPI)
    plt.close(fig)

    # ============ Plot 4: binning ============
    ab_img = to_img(T["arrival_bin"])
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    nb = int(ab_img.max() - ab_img.min()) + 1
    bin_title = f"arrival_bin = ceil(distance/(cΔt))  — valores enteros {int(ab_img.min())}…{int(ab_img.max())}"
    if not is_cpu:
        bin_title += f"  (valid: 0<bin≤{num_bins})"
    imshow_floor(axs[0], ab_img, pixel_x, pixel_y, bin_title,
                 cmap=plt.get_cmap("tab20", nb), cbar_label="bin",
                 vmin=ab_img.min() - 0.5, vmax=ab_img.max() + 0.5)
    # corte 1D a lo largo de la línea de píxeles (columna jx fija, variando iy) con mayor rango de distancia
    dist_img = to_img(T["distance"])
    tbin_img = to_img(T["tbin"])
    col = int(np.argmax(dist_img.max(0) - dist_img.min(0)))
    dist_row, tbin_row, ab_row = dist_img[:, col], tbin_img[:, col], ab_img[:, col]
    order = np.argsort(dist_row)
    d_all = T["distance"]
    o_all = np.argsort(d_all)
    axs[1].plot(d_all[o_all], T["tbin"][o_all], "-", color="C0", lw=1, label="distance/(cΔt)  (recta continua)")
    axs[1].step(d_all[o_all], T["arrival_bin"][o_all], where="post", color="0.7", lw=1,
                label="ceil(·), todos los píxeles")
    axs[1].plot(dist_row[order], ab_row[order], "o", color="C3", ms=4,
                label=f"ceil(·) en la línea de píxeles jx={col} (x={pixel_x[col]:+.4f} m)")
    axs[1].plot(dist_row[order], tbin_row[order], "s", color="C0", ms=3, mfc="none")
    axs[1].set_xlabel("distance = d1 + d2 [m]")
    axs[1].set_ylabel("bin")
    axs[1].set_title("Corte 1D: escalera del ceil frente a la recta continua", fontsize=9)
    axs[1].grid(alpha=0.3)
    axs[1].legend(fontsize=8)
    fig.suptitle(f"Binning temporal duro (cΔt = {c*bin_size*100:.2f} cm) — {pose_str()}{tag}", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{prefix}binning.png", dpi=DPI)
    plt.close(fig)

    # ============ Plot 5: intensidad ============
    mask = T["noc"] if is_cpu else T["valid"]
    inten_masked = np.where(mask, T["intensity"], 0.0)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    imshow_floor(axs[0], to_img(T["intensity"]), pixel_x, pixel_y,
                 "intensity  SIN aplicar noc (producto completo)", cmap="hot", cbar_label="a.u.")
    m_lbl = "noc" if is_cpu else "valid"
    imshow_floor(axs[1], to_img(inten_masked), pixel_x, pixel_y,
                 f"intensity · {m_lbl}  (ocluidos = 0; max={inten_masked.max():.3e})", cmap="hot", cbar_label="a.u.")
    denom_lbl = "d1²·d2²" if is_cpu else "d1²·clamp(d2²,eps)"
    fig.suptitle(f"intensity = I₀·A·dot1·dot2·dot3·dot4 / (4π²·{denom_lbl})  (A = área total de la faceta) — "
                 + pose_str() + tag, fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{prefix}intensity.png", dpi=DPI)
    plt.close(fig)

    # ============ Plot 6: transitorio completo ============
    y, y_alt, stats = E["y"], E["y_alt"], E["stats"]
    img = y.max(axis=2)
    # píxel central de la mitad iluminada de la grilla (x>0, y en el centro del FOV)
    iy_c, jx_c = N // 2, (3 * N) // 4
    assert to_img(T["noc"])[iy_c, jx_c], "el píxel elegido debería ver la faceta"
    hist_main = y_alt[iy_c, jx_c] if is_cpu else y[iy_c, jx_c]
    hist_sec = y[iy_c, jx_c] if is_cpu else y_alt[iy_c, jx_c]
    nz = np.nonzero(hist_main + hist_sec)[0]
    b_lo, b_hi = max(nz.min() - 3, 0), min(nz.max() + 4, num_bins)

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.8))
    img_title = ("max_t y[iy,jx,t]   (coord ORIGINAL con el '-1' en jx)" if is_cpu else
                 "max_t y[iy,jx,t]   (índice correcto + index_add_ + roll(+1, axis=1))")
    imshow_floor(axs[0], img, pixel_x, pixel_y, img_title, cmap="hot", cbar_label="a.u.")
    axs[0].plot(pixel_x[jx_c], pixel_y[iy_c], "c+", ms=12, mew=2, label=f"píxel (iy={iy_c}, jx={jx_c})")
    if is_cpu:
        axs[0].annotate("columna jx=0 envuelta\naquí por el '-1'", xy=(pixel_x[-1], pixel_y[N // 3]),
                        xytext=(pixel_x[N // 2 - 4], pixel_y[N // 6]), color="white", fontsize=7,
                        bbox=dict(boxstyle="round", fc="black", alpha=0.7),
                        arrowprops=dict(arrowstyle="->", color="white"))
    else:
        axs[0].annotate("columna jx=N-1 traída\naquí por el roll(+1)", xy=(pixel_x[0], pixel_y[N // 3]),
                        xytext=(pixel_x[N // 2 - 8], pixel_y[N // 6]), color="white", fontsize=7,
                        bbox=dict(boxstyle="round", fc="black", alpha=0.7),
                        arrowprops=dict(arrowstyle="->", color="white"))
    axs[0].legend(loc="lower left", fontsize=7)
    bins = np.arange(b_lo, b_hi)
    if is_cpu:
        lbl_main, lbl_sec = "coord corregido (add.at)", "coord original (+=, jx−1)"
    else:
        lbl_main, lbl_sec = "salida final (index_add_, con roll +1)", "reshape sin roll (y_raw)"
    axs[1].bar(bins + 1, hist_main[b_lo:b_hi], width=0.8, color="C3", label=lbl_main)
    axs[1].bar(bins + 1, hist_sec[b_lo:b_hi], width=0.4, color="C0", alpha=0.8, label=lbl_sec)
    axs[1].set_xlabel("arrival_bin (entero)")
    axs[1].set_ylabel("y  [a.u.]")
    axs[1].set_title(f"Histograma temporal del píxel (iy={iy_c}, jx={jx_c}) — deposición discreta por ceil", fontsize=9)
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.3, axis="y")
    fig.suptitle(f"Transitorio y (malla completa, {stats['n_tri']} triángulos, {num_bins} bins) — {pose_str()}{tag}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{prefix}transient.png", dpi=DPI)
    plt.close(fig)

    # ============ Plot 7: funciones duras ============
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    plot_hard_functions(axs)
    sup = "Funciones duras del código original (derivada nula o indefinida → gradiente no informativo)"
    if not is_cpu:
        sup = "Funciones duras de simulator.py (clamp / ceil / isfinite&>0): idénticas a las del código original"
    fig.suptitle(sup, fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{prefix}hard_functions.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# Revisión numérica (cpu: bugs del original; gpu: sanity check contra simulator.py)
# --------------------------------------------------------------------------
def cpu_bug_review(E, log):
    N = E["N"]
    mesh, theta, facet_normal = E["mesh"], E["theta"], E["normal"]
    y_orig, y_fixed, stats = E["y"], E["y_alt"], E["stats"]
    num_bins = E["num_bins"]
    cam_pos = E["cam_pos"]
    img_orig = y_orig.max(axis=2)
    img_fixed = y_fixed.max(axis=2)

    log("\n" + "-" * 78)
    log("a) Índice de píxel '-1' en coord = (bin-1)*N² + (jx - 1)*N + iy")
    log("-" * 78)
    jx0 = 0
    for iy0, bin0 in [(0, 1), (5, 1), (5, 20)]:
        coord0 = (bin0 - 1) * N**2 + (jx0 - 1) * N + iy0
        if coord0 < 0:
            where = "NEGATIVO -> indexa desde el final del vector (último bin, última columna)"
        else:
            t_w, rem = divmod(coord0, N**2)
            jx_w, iy_w = divmod(rem, N)
            where = f"cae en y[iy={iy_w}, jx={jx_w}, bin={t_w+1}]  (columna N-1 del bin anterior)"
        log(f"  píxel (iy={iy0}, jx={jx0}), bin={bin0}: coord={coord0:6d}  -> {where}")
    log(f"  En el transitorio de la faceta: {stats['n_coord_wrap']} depósitos tenían jx=0 (envuelven de bin), "
        f"{stats['n_coord_neg']} coords negativos.")
    shift_ok = np.allclose(y_orig[:, :-1, :], y_fixed[:, 1:, :])
    wrap_ok = np.allclose(y_orig[:, N - 1, :-1], y_fixed[:, 0, 1:])
    log(f"  y_orig[:, jx-1, t] == y_fixed[:, jx, t] para jx>=1 : {shift_ok}")
    log(f"  y_orig[:, N-1, t-1] == y_fixed[:, 0, t]  (columna 0 envuelta al bin anterior): {wrap_ok}")
    best = column_shift(img_orig, img_fixed)
    log(f"  Correlación cruzada de perfiles de columna (orig vs fixed): lag óptimo = {best} columna(s)")
    log(f"  |img_orig - img_fixed|_max = {np.abs(img_orig - img_fixed).max():.4e}  vs  "
        f"|img_orig[:, :-1] - img_fixed[:, 1:]|_max = {np.abs(img_orig[:, :-1] - img_fixed[:, 1:]).max():.4e}")
    log(f"  Energía total: sum(y_orig) = {y_orig.sum():.6e}, sum(y_fixed) = {y_fixed.sum():.6e} "
        f"(igual: el bug sólo desplaza/envuelve, no pierde energía)")

    log("\n" + "-" * 78)
    log("b) '+=' con índices repetidos vs np.add.at")
    log("-" * 78)
    coord_demo = np.array([3, 3, 3, 7, 7, 9])
    vals_demo = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ya = np.zeros(12); ya[coord_demo] += vals_demo
    yb = np.zeros(12); np.add.at(yb, coord_demo, vals_demo)
    log(f"  Caso sintético coord={coord_demo.tolist()}, intensity={vals_demo.tolist()}")
    log(f"    y[coord] += intensity  -> y[3]={ya[3]}, y[7]={ya[7]}, total={ya.sum()}  (se queda el ÚLTIMO valor)")
    log(f"    np.add.at(y, coord, .) -> y[3]={yb[3]}, y[7]={yb[7]}, total={yb.sum()}  (suma correcta)")
    log(f"    energía perdida por '+=': {yb.sum() - ya.sum():.1f} de {vals_demo.sum():.1f} "
        f"({100*(yb.sum()-ya.sum())/vals_demo.sum():.1f} %)")
    log(f"  Transitorio de la faceta: índices repetidos DENTRO de una misma llamada `y[coord] += intensity`: "
        f"{stats['n_dup_within_tri']}")
    log(f"    energía perdida acumulada por '+=' en la faceta: {stats['lost_pluseq']:.6e} "
        f"(sum(y_orig)={y_orig.sum():.6e}, sum(y_fixed)={y_fixed.sum():.6e})")
    log("    Lectura: en el código original coord se construye por triángulo y cada píxel aparece una sola vez,")
    log("    así que el '+=' NO pierde energía ahí (bug latente).  Sí pierde energía en cuanto se vectoriza el")
    log("    bucle sobre triángulos (varios triángulos -> mismo (píxel,bin)), que es justo lo que hace el simulador")
    log("    GPU; de ahí que el 'IMPORTANT FIX' use np.add.at / index_add_.")

    log("\n" + "-" * 78)
    log("c) theta = -cos(∠(u, v1)) usado como ÁNGULO en rotation_matrix(theta, [0,0,1])")
    log("-" * 78)
    theta_sim = PHI + 3 * np.pi / 2
    theta_sim_wrapped = (theta_sim + np.pi) % (2 * np.pi) - np.pi
    n_expected = np.array([-np.cos(PHI), -np.sin(PHI), 0.0])
    ang_err = np.degrees(np.arccos(np.clip(np.dot(facet_normal, n_expected), -1, 1)))
    log(f"  theta (original) = {theta:.6f} rad = {np.degrees(theta):.3f}°   (= -cos(phi) = {-np.cos(PHI):.6f}; "
        f"siempre en [-1, 0] rad, NO depende de rho)")
    log(f"  ángulo geométrico que usa simulator.py: theta = phi + 3π/2 = {theta_sim:.6f} rad "
        f"(≡ {theta_sim_wrapped:.6f} rad = {np.degrees(theta_sim_wrapped):.3f}°)")
    log(f"  normal media resultante de la faceta = {np.round(facet_normal, 5)}")
    log(f"  normal esperada (-cos phi, -sin phi, 0) = {np.round(n_expected, 5)}  -> error angular {ang_err:.3f}°")
    for phid in [30.0, 60.0, 90.0, 120.0, 150.0]:
        ph = np.deg2rad(phid)
        th = -np.clip(np.cos(ph), -1, 1)
        n_th = np.array([np.sin(th), -np.cos(th), 0.0])  # R_z(th)·[0,-1,0]
        n_ex = np.array([-np.cos(ph), -np.sin(ph), 0.0])
        err = np.degrees(np.arccos(np.clip(np.dot(n_th, n_ex), -1, 1)))
        log(f"    phi={phid:5.1f}°: theta_orig={th:+.4f} rad ({np.degrees(th):+7.2f}°), "
            f"theta_correcto≡{np.degrees((ph+1.5*np.pi+np.pi)%(2*np.pi)-np.pi):+7.2f}°, "
            f"normal_orig={np.round(n_th,3)}, esperada={np.round(n_ex,3)}, error={err:6.2f}°")
    log("  Lectura: theta=-cos(phi) es un número adimensional en [-1,0] que se interpreta como radianes.  Sólo")
    log("  coincide con el ángulo correcto en phi=90° (ambos 0) y casi por casualidad cerca de phi≈60°/120°")
    log("  (−0.5 rad = −28.6° vs −30°); en phi=30°/150° la normal se desvía 10.4°, y en phi=0 la faceta gira")
    log("  −1 rad (−57.3°) en vez de −90°.  Además theta no es monótono en el sentido correcto: la faceta")
    log("  NUNCA gira más de ±57.3°, así que para |phi-90°|>~30° deja de 'mirar' al oclusor.")

    log("\n" + "-" * 78)
    log("d) Sin bounds en arrival_bin")
    log("-" * 78)
    log(f"  Pose actual: max(arrival_bin) = {stats['max_bin']}  vs  num_time_bins = {num_bins}  "
        f"(margen {num_bins - stats['max_bin']} bins) -> no desborda.")
    for rho_far, ph_far in [(2.5, PHI), (2.5, np.deg2rad(90.0)), (3.5, np.deg2rad(60.0)), (4.0, np.deg2rad(45.0))]:
        p = np.array([rho_far * np.cos(ph_far), rho_far * np.sin(ph_far), 0.55])
        dd = np.linalg.norm(p - laser_pos) + np.linalg.norm(cam_pos - p, axis=1)
        mb = int(np.ceil(dd / (c * bin_size)).max())
        flag = "DESBORDA (coord > len(y_meas_vec) -> IndexError, o sin excepción envuelve si negativo)" if mb > num_bins else "ok"
        log(f"  rho={rho_far:.1f}, phi={np.degrees(ph_far):.0f}°, z=0.55: max(arrival_bin)≈{mb} vs {num_bins} -> {flag}")
    log("  Lectura: el margen 1.2× sobre el punto (xmax,ymax,zmax) cubre poses dentro de los bounds; un objeto")
    log("  fuera de los bounds (o bin_size menor) produce coord fuera de rango sin ningún chequeo.")

    log("\n" + "-" * 78)
    log("e) División por cero sin máscara en m = Δy/Δx  y  xint = -b/m")
    log("-" * 78)

    def count_bad(center, Nn):
        _, _, cp, _ = spad_grid(Nn)
        with np.errstate(divide="ignore", invalid="ignore"):
            m_ = (center[1] - cp[:, 1]) / (center[0] - cp[:, 0])
            b_ = center[1] - np.dot(m_, center[0])
            xi = -b_ / m_
        bad_m = ~np.isfinite(m_)
        bad_x = ~np.isfinite(xi)
        noc_ = xi > 0
        return bad_m.sum(), bad_x.sum(), np.isnan(xi).sum(), noc_.sum(), Nn * Nn, np.abs(m_).max(), np.abs(center[0] - cp[:, 0]).min()

    for label, center, Nn in [
        ("phi=90°, N=32 (x=rho·cos(90°)=6e-17)", np.array([RHO * np.cos(np.pi / 2), RHO, 0.55]), 32),
        ("phi=90°, N=33 (grilla con columna en x=0)", np.array([RHO * np.cos(np.pi / 2), RHO, 0.55]), 33),
        ("x=0 exacto, N=33", np.array([0.0, RHO, 0.55]), 33),
        ("x=0 exacto, N=32", np.array([0.0, RHO, 0.55]), 32),
    ]:
        bm, bx, nn, nc, tot, mmax, dxmin = count_bad(center, Nn)
        log(f"  {label:44s}: m no finito={bm:3d}, xint no finito={bx:3d} (nan={nn:3d}), noc=True: {nc:4d}/{tot}, "
            f"max|m|={mmax:.3e}, min|Δx|={dxmin:.2e}")
    log("  Lectura: con x=rho·cos(90°)=6.1e-17 ningún píxel tiene exactamente x=x_c (Δx mínimo 6e-17 con N impar),")
    log("  así que no hay inf/nan, sólo pendientes |m| de hasta 2e16.  Con x_c=0 exacto y N impar la columna")
    log("  central produce m=±inf; np.dot(m, 0.0) devuelve 0 (no nan), b=y_c y xint=-y_c/inf=-0.0, de modo que")
    log("  `xint>0` es False: esa columna entera queda SILENCIOSAMENTE ocluida (noc=True baja de 561 a 528,")
    log("  es decir 33 píxeles menos).  Sin máscara la división por cero contamina noc sin ningún aviso.")

    log("\n" + "-" * 78)
    log("f) Otras observaciones de revisión")
    log("-" * 78)
    log("  - Globales implícitos: c, zmin, ymin, object_folder (y MESH_MIN_TRIANGLES) se leen del módulo,")
    log("    no son parámetros de simulation(...): la función no es reproducible de forma aislada.")
    log("  - 'yaw' se lee de object_positions pero NUNCA se usa; la rotación en z usa theta (ver c).")
    log("  - Escalado: scale = min(w/ext_x, 1.1/ext_z) — no hay parámetro de altura 'h'; para la faceta plana")
    log(f"    ext_z=0 -> 1.1/0 = inf y el min() lo descarta (altura resultante = {mesh.extents[2]:.4f} m,")
    log("    fijada por la relación de aspecto del .obj, no por el usuario).")
    log("  - 'u = u = np.array([1,0,0])' asignación doble; theta se calcula fuera del if de carga.")
    log("  - Oclusión geométrica reducida a un test del centroide (xint>0): sin auto-oclusión, sin visibilidad")
    log("    parcial de triángulos; Heaviside duro -> gradiente cero salvo en el borde.")
    log("  - Binning: ceil(...) sin interpolación: toda la energía del triángulo cae en un único bin entero;")
    log("    la derivada respecto a rho/phi es cero casi en todo punto (ver hard_functions.png).")
    log("  - El vector y_meas_vec se aloja como cam_pixel_dim²·num_bins y se hace reshape order='F' a (iy, jx, t).")
    log("  - Tras el reshape la versión de utils/crb_fix.py aplica además np.roll(y, -1, axis=-1) (desplaza 1 bin).")


def column_shift(img_a, img_b):
    """Lag (en columnas) que maximiza la correlación cruzada de los perfiles de columna."""
    N = img_a.shape[1]
    a = img_a.sum(0) - img_a.sum(0).mean()
    bb = img_b.sum(0) - img_b.sum(0).mean()
    lags = np.arange(-(N - 1), N)
    xc = np.array([np.sum(a[max(0, -l):N - max(0, l)] * bb[max(0, l):N - max(0, -l)]) for l in lags])
    return int(lags[np.argmax(xc)])


def gpu_sanity_check(E, log):
    """Compara la réplica numpy con la salida REAL de simulator.simulation (torch float64)."""
    import time

    t0 = time.time()
    y_real, params, mesh_real = run_real_simulator(E["rho"], E["phi"], E["w"], E["N"])
    t_real = time.time() - t0
    y_rep = E["y"]
    dv = np.abs(mesh_real.vertices - E["mesh"].vertices).max() if mesh_real.vertices.shape == E["mesh"].vertices.shape else np.nan
    log("\n" + "-" * 78)
    log("Sanity check: réplica numpy vs simulator.simulation(...) real (torch, cpu, float64)")
    log("-" * 78)
    log(f"  num_time_bins real = {params['num_time_bins']}  vs réplica = {E['num_bins']}")
    log(f"  malla: {len(mesh_real.faces)} triángulos reales vs {len(E['mesh'].faces)} réplica; "
        f"max|Δvértices| = {dv:.3e}")
    log(f"  shapes: real {y_real.shape}, réplica {y_rep.shape}")
    diff = np.abs(y_rep - y_real)
    log(f"  max|réplica − real| = {diff.max():.3e}   (max|real| = {np.abs(y_real).max():.4e}, "
        f"sum real = {y_real.sum():.6e}, sum réplica = {y_rep.sum():.6e})")
    log(f"  max|réplica − real| relativo = {diff.max() / np.abs(y_real).max():.3e}")
    diff_noroll = np.abs(E["y_alt"] - y_real).max()
    log(f"  (control) max|y_raw sin roll − real| = {diff_noroll:.3e}  -> el roll(+1, axis=1) es necesario")
    log(f"  tiempo simulator.simulation en CPU torch: {t_real:.2f} s")
    return dict(y_real=y_real, max_abs_diff=float(diff.max()), max_rel_diff=float(diff.max() / np.abs(y_real).max()),
                t_real=t_real, dv=dv)


def gpu_review(E, log):
    stats, T, num_bins = E["stats"], E["T"], E["num_bins"]
    n_expected = np.array([-np.cos(E["phi"]), -np.sin(E["phi"]), 0.0])
    ang_err = np.degrees(np.arccos(np.clip(np.dot(E["normal"], n_expected), -1, 1)))
    log("\n" + "-" * 78)
    log("Diagnósticos de la réplica GPU")
    log("-" * 78)
    log(f"  theta = phi + 3π/2 = {E['theta']:.6f} rad; normal media = {np.round(E['normal'], 5)} vs esperada "
        f"{np.round(n_expected, 5)} -> error {ang_err:.4f}°")
    log(f"  triángulos: {stats['n_tri']}, saltados por noc vacío: {stats['n_tri_skipped_noc']}")
    log(f"  |denom|<=eps (máscara): {stats['n_invalid_denom']}; xint no finitos: {stats['n_nonfinite_xint']}")
    log(f"  bins usados {stats['min_bin']}…{stats['max_bin']} de {num_bins}; fuera de rango (descartados por valid): "
        f"{stats['n_bin_out_of_range']}")
    log(f"  clamp activo (dot<0 con valid): dot2 {stats['n_clamped_dot2']}, dot4 {stats['n_clamped_dot4']}")
    log(f"  índices repetidos dentro de un triángulo: {stats['n_dup_within_tri']} (index_add_ los sumaría igualmente)")
    log(f"  centroide: noc=True en {T['noc'].sum()} / {E['N']**2} px; dot1={T['dot1']:.5f}, dot3={T['dot3']:.5f}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=list(BACKENDS), default="cpu")
    ap.add_argument("--skip-sanity", action="store_true", help="(gpu) no llamar a simulator.simulation")
    args = ap.parse_args(argv)
    backend = args.backend
    prefix = BACKENDS[backend]["prefix"]

    PLOTS_DIR.mkdir(exist_ok=True)
    report = []

    def log(s=""):
        print(s)
        report.append(s)

    log("=" * 78)
    if backend == "cpu":
        log("Revisión numérica del simulador ORIGINAL (CPU/numpy) - funciones componentes")
    else:
        log("Funciones componentes del simulador GPU (simulator.py) - réplica numpy")
    log("=" * 78)
    log(f"Pose de ejemplo: rho={RHO}, phi={PHI_DEG} deg -> x={RHO*np.cos(PHI):.6f}, "
        f"y={RHO*np.sin(PHI):.6f}, z=0, w={FACET_W}")
    log(f"N={cam_pixel_dim}, camera_FOV={camera_FOV}, bin_size={bin_size}, c*bin_size={c*bin_size:.5f} m, "
        f"laser_intensity={laser_intensity}, hide_walls={hide_walls}")

    E = evaluate_backend(backend, run_transient=False)
    log(f"num_time_bins = {E['num_bins']}  (max_dist_travel = {E['max_dist_travel']:.4f} m, margen 1.2)")
    log(f"Faceta: {len(E['mesh'].faces)} triángulos tras densify (min {MESH_MIN_TRIANGLES}), "
        f"área total = {E['area']:.5f} m², extents = {np.round(E['mesh'].extents, 4)}")
    log(f"Centroide (área-ponderado) = {np.round(E['center'], 5)}, normal media = {np.round(E['normal'], 5)}")
    T = E["T"]
    log(f"Escalares del triángulo representativo: d1={T['d1']:.5f} m, dot1={T['dot1']:.5f} "
        f"(raw {T['dot1_raw']:.5f}), dot3={T['dot3']:.5f} (raw {T['dot3_raw']:.5f})")
    log(f"Píxeles con noc=True (ven la faceta): {T['noc'].sum()} / {cam_pixel_dim*cam_pixel_dim}")

    log("\nSimulando el transitorio con el bucle fiel sobre todos los triángulos ...")
    E = evaluate_backend(backend, run_transient=True)
    stats = E["stats"]
    log(f"  triángulos procesados: {stats['n_tri']}, bins usados: {stats['min_bin']}…{stats['max_bin']}")

    make_component_plots(E, prefix, log)

    if backend == "cpu":
        cpu_bug_review(E, log)
    else:
        gpu_review(E, log)
        if not args.skip_sanity:
            gpu_sanity_check(E, log)

    log("\nArchivos generados:")
    for f in sorted(PLOTS_DIR.glob(f"{prefix}*.png")):
        if backend == "cpu" and "_gpu_" in f.name:
            continue
        log(f"  {f.relative_to(ROOT)}")

    (PLOTS_DIR / f"{prefix}review.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nReporte escrito en {PLOTS_DIR / (prefix + 'review.txt')}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
