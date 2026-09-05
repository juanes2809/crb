import argparse
import concurrent.futures
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import trimesh
from trimesh.transformations import rotation_matrix

try:
    from model.notebooks.utils.create_wall import create_wall_mesh
    from model.notebooks.utils.densify_mesh import densify_mesh_if_needed
    from model.notebooks.utils.get_files import get_obj_files
    from model.notebooks.utils.noise import add_background_noise, add_poisson_noise, add_sensor_noise
except ModuleNotFoundError:
    from utils.create_wall import create_wall_mesh
    from utils.densify_mesh import densify_mesh_if_needed
    from utils.get_files import get_obj_files
    from utils.noise import add_background_noise, add_poisson_noise, add_sensor_noise


C = 299792458
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OBJECT_FOLDER = REPO_ROOT / "objects"


def resolve_torch_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested, but torch.cuda.is_available() is False. Falling back to CPU.")
        return torch.device("cpu")
    return device


def _load_mesh(obj_file, object_folder, uploaded_objs=None):
    if uploaded_objs and obj_file.startswith("uploaded_"):
        uploaded_file = uploaded_objs[obj_file]
        uploaded_file.seek(0)
        mesh = trimesh.load(uploaded_file, file_type="obj", force="mesh")
    else:
        mesh = trimesh.load(object_folder / obj_file, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def polar_to_scene_xy(rho, phi=None, phi_deg=None):
    """Convert notebook polar coordinates to simulator XY coordinates."""
    rho = float(rho)
    if rho <= 0:
        raise ValueError(f"rho must be positive. Received rho={rho}")

    if phi_deg is not None:
        phi_rad = np.deg2rad(float(phi_deg))
    elif phi is not None:
        phi_rad = float(phi)
    else:
        raise ValueError("Pass phi in radians or phi_deg in degrees.")

    xcoord = rho * np.cos(phi_rad)
    ycoord = rho * np.sin(phi_rad)
    return float(xcoord), float(ycoord), float(phi_rad)


def _object_position_to_scene(obj_data):
    if "rho" in obj_data:
        if "phi_deg" in obj_data:
            xcoord, ycoord, phi_rad = polar_to_scene_xy(obj_data["rho"], phi_deg=obj_data["phi_deg"])
        elif "phi" in obj_data:
            xcoord, ycoord, phi_rad = polar_to_scene_xy(obj_data["rho"], phi=obj_data["phi"])
        else:
            raise KeyError("Objects with 'rho' must also include 'phi' or 'phi_deg'.")
        rho = float(obj_data["rho"])
    else:
        xcoord = float(obj_data["xcoord"])
        ycoord = float(obj_data["ycoord"])
        rho = float(np.hypot(xcoord, ycoord))
        phi_rad = float(np.arctan2(ycoord, xcoord))

    return {
        **obj_data,
        "xcoord": float(xcoord),
        "ycoord": float(ycoord),
        "zcoord": float(obj_data.get("zcoord", 0.0)),
        "rho": rho,
        "phi": phi_rad,
        "phi_deg": float(np.rad2deg(phi_rad)),
    }


def _transform_object_mesh(obj, obj_data):
    obj_data = _object_position_to_scene(obj_data)
    xcoord = obj_data["xcoord"]
    ycoord = obj_data["ycoord"]
    zcoord = obj_data["zcoord"]
    w = float(obj_data["w"])
    pitch = float(obj_data.get("pitch", 1.57))
    roll = float(obj_data.get("roll", 0.0))
    phi_rad = obj_data["phi"]

    v1 = np.array([xcoord, ycoord, zcoord])

    obj = obj.copy()
    obj = densify_mesh_if_needed(obj, min_triangles=10000)

    obj_extents = obj.extents
    scale_factors = []
    if obj_extents[0] > 1e-12:
        scale_factors.append(w / obj_extents[0])
    if obj_extents[2] > 1e-12:
        scale_factors.append(1.1 / obj_extents[2])
    if not scale_factors:
        raise ValueError(f"Object {obj_data['obj_file']} has invalid extents: {obj_extents}")
    obj.apply_scale(min(scale_factors))

    obj.apply_transform(rotation_matrix(pitch, [1, 0, 0]))
    obj.apply_transform(rotation_matrix(roll, [0, 1, 0]))
    theta = phi_rad + 3 * np.pi / 2
    obj.apply_transform(rotation_matrix(theta, [0, 0, 1]))

    z_min = obj.vertices[:, 2].min()
    obj.apply_translation([0, 0, -z_min])
    obj.apply_translation(v1)
    return obj


def simulate_transient_torch(
    objects,
    cam_pixel_dim,
    camera_FOV,
    camera_FOV_center,
    num_time_bins,
    bin_size,
    laser_intensity,
    laser_pos_np,
    laser_normal_np,
    device,
    dtype,
    triangle_chunk_size=256,
):
    """GPU-vectorized transient simulation.

    The original version sent each triangle to the GPU one by one. This version
    processes a chunk of triangles against all SPAD pixels at once. The chunk
    size controls the speed/memory trade-off:

        larger chunk  -> faster, more GPU memory
        smaller chunk -> slower, safer on limited GPUs
    """
    pixel_x = torch.linspace(
        camera_FOV_center[0] - camera_FOV / 2 + camera_FOV / (2 * cam_pixel_dim),
        camera_FOV_center[0] + camera_FOV / 2 - camera_FOV / (2 * cam_pixel_dim),
        cam_pixel_dim,
        device=device,
        dtype=dtype,
    )
    pixel_y = torch.linspace(
        camera_FOV_center[1] - camera_FOV / 2 + camera_FOV / (2 * cam_pixel_dim),
        camera_FOV_center[1] + camera_FOV / 2 - camera_FOV / (2 * cam_pixel_dim),
        cam_pixel_dim,
        device=device,
        dtype=dtype,
    )

    Y, X = torch.meshgrid(pixel_y, pixel_x, indexing="ij")
    cam_pos = torch.stack((X.reshape(-1), Y.reshape(-1), torch.zeros(X.numel(), device=device, dtype=dtype)), dim=1)
    num_pixels = cam_pos.shape[0]
    pixel_idx = torch.arange(num_pixels, device=device, dtype=torch.long)
    pixel_x_idx = pixel_idx % cam_pixel_dim
    pixel_y_idx = pixel_idx // cam_pixel_dim
    pixel_linear = pixel_x_idx * cam_pixel_dim + pixel_y_idx

    combined_mesh = trimesh.util.concatenate(objects)
    triangles = torch.as_tensor(np.array(combined_mesh.triangles, copy=True), device=device, dtype=dtype)
    triangle_normals = torch.as_tensor(np.array(combined_mesh.face_normals, copy=True), device=device, dtype=dtype)

    laser_pos = torch.as_tensor(laser_pos_np, device=device, dtype=dtype)
    laser_normal = torch.as_tensor(laser_normal_np, device=device, dtype=dtype)
    floor_normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)

    scene_centers = triangles.mean(dim=1)                             # [T, 3]
    areas = 0.5 * torch.linalg.norm(torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1), dim=1)

    y_meas_vec = torch.zeros(num_pixels * num_time_bins, device=device, dtype=dtype)
    fourpi = 4.0 * torch.pi * torch.pi
    eps = torch.tensor(1e-12, device=device, dtype=dtype)

    # These pixel quantities are reused for every triangle chunk.
    cam_x = cam_pos[:, 0][None, :]                                    # [1, P]
    cam_y = cam_pos[:, 1][None, :]                                    # [1, P]

    for start in range(0, scene_centers.shape[0], triangle_chunk_size):
        end = min(start + triangle_chunk_size, scene_centers.shape[0])
        centers = scene_centers[start:end]                             # [B, 3]
        normals = triangle_normals[start:end]                          # [B, 3]
        area = areas[start:end]                                        # [B]
        B = centers.shape[0]

        cx = centers[:, 0:1]
        cy = centers[:, 1:2]

        denom = cx - cam_x                                             # [B, P]
        valid_denom = torch.abs(denom) > eps
        m = torch.where(valid_denom, (cy - cam_y) / denom, torch.full_like(denom, torch.nan))
        b = cy - m * cx
        xint = -b / m

        # Edge visibility / no-occlusion mask.
        noc = torch.isfinite(xint) & (xint > 0)
        if not torch.any(noc):
            continue

        lps = laser_pos[None, :] - centers                             # [B, 3]
        d1s = torch.sum(lps * lps, dim=1)                               # [B]
        d1 = torch.sqrt(torch.clamp(d1s, min=eps))                      # [B]

        fovsp = cam_pos[None, :, :] - centers[:, None, :]               # [B, P, 3]
        d2s = torch.sum(fovsp * fovsp, dim=2)                           # [B, P]
        d2 = torch.sqrt(torch.clamp(d2s, min=eps))                      # [B, P]

        arrival_bin = torch.ceil((d1[:, None] + d2) / (C * bin_size)).to(torch.long)
        valid = noc & (arrival_bin > 0) & (arrival_bin <= num_time_bins)
        if not torch.any(valid):
            continue

        dot1 = torch.clamp(torch.sum(normals * (lps / d1[:, None]), dim=1), min=0.0)             # [B]
        dot2 = torch.clamp(torch.sum(normals[:, None, :] * (fovsp / d2[:, :, None]), dim=2), min=0.0)  # [B, P]
        dot3 = torch.clamp(torch.sum(laser_normal[None, :] * (-lps / d1[:, None]), dim=1), min=0.0)    # [B]
        dot4 = torch.clamp(torch.sum(floor_normal[None, None, :] * (-fovsp / d2[:, :, None]), dim=2), min=0.0) # [B, P]

        intensity = laser_intensity * area[:, None] * (dot1[:, None] * dot2 * dot3[:, None] * dot4) / (
            fourpi * d1s[:, None] * torch.clamp(d2s, min=eps)
        )
        intensity = torch.where(valid, intensity, torch.zeros_like(intensity))

        tri_idx, pix_idx = torch.nonzero(valid, as_tuple=True)
        coord = (arrival_bin[tri_idx, pix_idx] - 1) * num_pixels + pixel_linear[pix_idx]
        y_meas_vec.index_add_(0, coord, intensity[tri_idx, pix_idx])

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return y_meas_vec.cpu().numpy().reshape((cam_pixel_dim, cam_pixel_dim, num_time_bins), order="F")

def simulation(
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
    camera_FOV,
    cam_pixel_dim,
    bin_size,
    laser_intensity,
    object_positions,
    hide_walls,
    SNR_dB,
    SBR,
    poisson_scale_factor,
    add_noise,
    object_folder=DEFAULT_OBJECT_FOLDER,
    uploaded_objs=None,
    torch_device=None,
    torch_dtype=torch.float32,
    triangle_chunk_size=256,
):
    objects = []
    scene_objects = []

    camera_FOV_center = [0, -camera_FOV / 2, 0]
    FOV_radius = camera_FOV / cam_pixel_dim
    laser_pos = np.array([0, 0, 0])
    laser_normal = np.array([0, 0, 1])
    wall_discr = C / 2 * bin_size / 4

    params = {
        "cam_pixel_dim": cam_pixel_dim,
        "camera_FOV": camera_FOV,
        "camera_FOV_center": camera_FOV_center,
        "FOV_radius": FOV_radius,
        "laser_intensity": laser_intensity,
        "bin_size": bin_size,
        "c": C,
        "laser_pos": laser_pos,
        "laser_normal": laser_normal,
        "wall_discr": wall_discr,
    }

    furthest_scene_point = np.array([xmax, ymax, zmax])
    furthest_spad_point = np.array([-camera_FOV / 2, -camera_FOV, 0])
    max_dist_travel = np.linalg.norm(furthest_scene_point - laser_pos) + np.linalg.norm(
        furthest_spad_point - furthest_scene_point
    )
    num_time_bins = int(np.ceil(max_dist_travel / C / bin_size * 1.2))
    params["num_time_bins"] = num_time_bins

    if not hide_walls:
        objects.extend(
            [
                create_wall_mesh(np.array([xmin, ymax, 0]), np.array([xmax - xmin, 0, 0]), np.array([0, 0, zmax])),
                create_wall_mesh(np.array([xmax, ymax, 0]), np.array([0, -ymax, 0]), np.array([0, 0, zmax])),
                create_wall_mesh(np.array([xmin, ymax, 0]), np.array([0, -ymax, 0]), np.array([0, 0, zmax])),
                create_wall_mesh(np.array([xmin, ymax, zmax]), np.array([xmax - xmin, 0, 0]), np.array([0, -ymax, 0])),
            ]
        )

    for obj_data in object_positions:
        obj = _load_mesh(obj_data["obj_file"], object_folder, uploaded_objs=uploaded_objs)
        obj = _transform_object_mesh(obj, obj_data)
        scene_objects.append(obj)

    objects.extend(scene_objects)

    if torch_device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_meas_vec = simulate_transient_torch(
        objects=objects,
        cam_pixel_dim=cam_pixel_dim,
        camera_FOV=camera_FOV,
        camera_FOV_center=camera_FOV_center,
        num_time_bins=num_time_bins,
        bin_size=bin_size,
        laser_intensity=laser_intensity,
        laser_pos_np=laser_pos,
        laser_normal_np=laser_normal,
        device=torch_device,
        dtype=torch_dtype,
        triangle_chunk_size=triangle_chunk_size,
    )

    if add_noise:
        y_with_background = add_background_noise(y_meas_vec, sbr=SBR)
        y_with_shot_noise = add_poisson_noise(y_with_background, scale_factor=poisson_scale_factor)
        y_meas_vec_noisy = add_sensor_noise(y_with_shot_noise, SNR_dB)
    else:
        y_meas_vec_noisy = y_meas_vec

    y_meas_vec_noisy = orient_transient_measurement(y_meas_vec_noisy)
    return y_meas_vec_noisy, params


def build_object_mesh(obj_data, object_folder=DEFAULT_OBJECT_FOLDER, uploaded_objs=None):
    obj = _load_mesh(obj_data["obj_file"], object_folder, uploaded_objs=uploaded_objs)
    return _transform_object_mesh(obj, obj_data)


def build_scene_obj(object_positions, xmin, xmax, ymax, zmax, hide_walls, object_folder=DEFAULT_OBJECT_FOLDER, uploaded_objs=None):
    objects = []
    if not hide_walls:
        objects.extend(
            [
                create_wall_mesh(np.array([xmin, ymax, 0]), np.array([xmax - xmin, 0, 0]), np.array([0, 0, zmax])),
                create_wall_mesh(np.array([xmax, ymax, 0]), np.array([0, -ymax, 0]), np.array([0, 0, zmax])),
                create_wall_mesh(np.array([xmin, ymax, 0]), np.array([0, -ymax, 0]), np.array([0, 0, zmax])),
                create_wall_mesh(np.array([xmin, ymax, zmax]), np.array([xmax - xmin, 0, 0]), np.array([0, -ymax, 0])),
            ]
        )

    for obj_data in object_positions:
        objects.append(build_object_mesh(obj_data, object_folder=object_folder, uploaded_objs=uploaded_objs))

    laser = trimesh.creation.icosphere(radius=0.04)
    laser.apply_translation([0, 0, 0])
    objects.append(laser)

    full_scene = trimesh.util.concatenate(objects)
    full_scene.apply_transform(rotation_matrix(np.radians(90), [1, 0, 0]))
    full_scene.apply_transform(rotation_matrix(np.radians(180), [0, 0, 1]))
    return full_scene


def validate_scene(object_positions, bounds, min_centroid_dist=0.1):
    centroids = []
    for obj_data in object_positions:
        scene_data = _object_position_to_scene(obj_data)
        x = scene_data["xcoord"]
        y = scene_data["ycoord"]
        z = scene_data["zcoord"]
        if not (bounds["xmin"] <= x <= bounds["xmax"]):
            return False
        if not (bounds["ymin"] <= y <= bounds["ymax"]):
            return False
        if not (bounds["zmin"] <= z <= bounds["zmax"]):
            return False
        centroids.append(np.array([x, y, z]))

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            if np.linalg.norm(centroids[i] - centroids[j]) < min_centroid_dist:
                return False
    return True


def sample_random_object_positions(
    obj_files,
    bounds,
    rng,
    py_rng,
    num_objects_range=(1, 3),
    rho_range=None,
    phi_range=None,
    w_range=(0.3, 0.7),
    pitch_range=(1.2, 1.9),
    roll_range=(-0.3, 0.3),
    yaw_range=(-0.3, 0.3),
):
    if rho_range is None:
        rho_range = (
            max(1e-6, bounds["ymin"] + 0.2),
            np.hypot(max(abs(bounds["xmin"]), abs(bounds["xmax"])), bounds["ymax"] - 0.2),
        )
    if phi_range is None:
        phi_range = (0.0, np.pi)

    num_objects = py_rng.randint(*num_objects_range)
    objects = []
    for _ in range(num_objects):
        rho = float(rng.uniform(*rho_range))
        phi = float(rng.uniform(*phi_range))
        xcoord, ycoord, _ = polar_to_scene_xy(rho, phi=phi)
        objects.append(
            {
                "obj_file": py_rng.choice(obj_files),
                "rho": rho,
                "phi": phi,
                "phi_deg": float(np.rad2deg(phi)),
                "xcoord": xcoord,
                "ycoord": ycoord,
                "w": float(rng.uniform(*w_range)),
                "yaw": float(rng.uniform(*yaw_range)),
                "pitch": float(rng.uniform(*pitch_range)),
                "roll": float(rng.uniform(*roll_range)),
            }
        )
    return objects


def _write_hdf5_params(params_group, params):
    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            params_group.attrs[key] = np.asarray(value)
        else:
            params_group.attrs[key] = value


def orient_transient_measurement(y_meas_vec):
    return np.roll(y_meas_vec, shift=1, axis=1)


def generate_single_scene(scene_idx, config):
    local_seed = config["seed"] + scene_idx
    rng = np.random.default_rng(local_seed)
    py_rng = random.Random(local_seed)

    for attempt in range(1, config["max_trials_per_scene"] + 1):
        object_positions = sample_random_object_positions(
            config["obj_files"],
            config["bounds"],
            rng,
            py_rng,
            rho_range=config["rho_range"],
            phi_range=config["phi_range"],
        )
        if not validate_scene(object_positions, config["bounds"]):
            continue

        print(f"Starting scene {scene_idx + 1}/{config['num_scenes']} (attempt {attempt})")
        y_meas_vec_noisy, params = simulation(
            config["xmin"],
            config["xmax"],
            config["ymin"],
            config["ymax"],
            config["zmin"],
            config["zmax"],
            config["camera_FOV"],
            config["cam_pixel_dim"],
            config["bin_size"],
            config["laser_intensity"],
            object_positions,
            config["hide_walls"],
            config["SNR_dB"],
            config["SBR"],
            config["poisson_scale_factor"],
            config["add_noise"],
            object_folder=config["object_folder"],
            torch_device=config["torch_device"],
            torch_dtype=config["torch_dtype"],
            triangle_chunk_size=config["triangle_chunk_size"],
        )

        scene_id = f"scene_{scene_idx:04d}"
        scene_folder = config["output_dir"] / scene_id
        scene_folder.mkdir(parents=True, exist_ok=True)

        with h5py.File(scene_folder / "transient.hdf5", "w") as f:
            transient_dataset = f.create_dataset("y_meas_vec_noisy", data=y_meas_vec_noisy)
            transient_dataset.attrs["orientation"] = "roll_axis_1_shift_1"
            transient_dataset.attrs["shape_order"] = "y_pixels, x_pixels, time_bins"
            f.attrs["transient_shape"] = y_meas_vec_noisy.shape
            _write_hdf5_params(f.create_group("params"), params)

        scene_mesh = build_scene_obj(
            object_positions,
            config["xmin"],
            config["xmax"],
            config["ymax"],
            config["zmax"],
            config["hide_walls"],
            object_folder=config["object_folder"],
        )
        scene_mesh.export(
            str(scene_folder / "scene.obj"),
            include_color=False,
            include_texture=False,
            write_texture=False,
        )

        meta = {
            "scene_id": scene_id,
            "object_positions": object_positions,
            "bounds": config["bounds"],
            "camera_FOV": config["camera_FOV"],
            "cam_pixel_dim": config["cam_pixel_dim"],
            "bin_size": config["bin_size"],
            "laser_intensity": config["laser_intensity"],
            "rho_range": config["rho_range"],
            "phi_range": config["phi_range"],
            "hide_walls": config["hide_walls"],
            "SNR_dB": config["SNR_dB"],
            "SBR": config["SBR"],
            "poisson_scale_factor": config["poisson_scale_factor"],
            "add_noise": config["add_noise"],
        }
        with open(scene_folder / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved {scene_folder}")
        return True

    print(f"Could not generate scene {scene_idx} after {config['max_trials_per_scene']} attempts.")
    return False


def generate_transient_dataset(**config):
    config.setdefault("rho_range", None)
    config.setdefault("phi_range", None)
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    print(f"Using torch device: {config['torch_device']}")
    obj_files = get_obj_files(str(config["object_folder"]), {})
    if not obj_files:
        raise FileNotFoundError(f"No .obj files found in {config['object_folder']}")

    config["obj_files"] = obj_files
    config["bounds"] = {
        "xmin": config["xmin"],
        "xmax": config["xmax"],
        "ymin": config["ymin"],
        "ymax": config["ymax"],
        "zmin": config["zmin"],
        "zmax": config["zmax"],
    }
    if config["rho_range"] is None:
        config["rho_range"] = (
            max(1e-6, config["ymin"] + 0.2),
            float(np.hypot(max(abs(config["xmin"]), abs(config["xmax"])), config["ymax"] - 0.2)),
        )
    if config["phi_range"] is None:
        config["phi_range"] = (0.0, np.pi)
    config["rho_range"] = tuple(float(value) for value in config["rho_range"])
    config["phi_range"] = tuple(float(value) for value in config["phi_range"])

    if config["torch_device"].type == "cuda" and config["num_workers"] != 1:
        print("Warning: using num_workers > 1 with one CUDA GPU usually slows things down or causes CUDA contention. Prefer --num-workers 1 per GPU.")

    generated = 0
    if config["num_workers"] == 1:
        for scene_idx in range(config["num_scenes"]):
            generated += int(generate_single_scene(scene_idx, config))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=config["num_workers"]) as executor:
            futures = [executor.submit(generate_single_scene, i, config) for i in range(config["num_scenes"])]
            for future in concurrent.futures.as_completed(futures):
                generated += int(future.result())

    print(f"Dataset ready: {generated} scenes in {config['output_dir']}")
    return generated


def parse_args():
    parser = argparse.ArgumentParser(description="Generate NLOS SPAD transient scenes.")
    parser.add_argument("num_scenes_pos", nargs="?", type=int, help="Number of scenes to generate.")
    parser.add_argument("--num-scenes", type=int, default=None, help="Number of scenes to generate.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "Dataset", help="Output dataset directory.")
    parser.add_argument("--object-folder", type=Path, default=DEFAULT_OBJECT_FOLDER, help="Folder containing .obj files.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-trials-per-scene", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--triangle-chunk-size", type=int, default=256, help="Number of triangles processed per GPU chunk. Increase for speed if GPU memory allows.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="PyTorch device for transient simulation. Use cuda on a CUDA-enabled machine.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="PyTorch dtype for transient simulation.",
    )

    parser.add_argument("--xmin", type=float, default=-1.5)
    parser.add_argument("--xmax", type=float, default=1.5)
    parser.add_argument("--ymin", type=float, default=0.0)
    parser.add_argument("--ymax", type=float, default=3.0)
    parser.add_argument("--zmin", type=float, default=0.0)
    parser.add_argument("--zmax", type=float, default=3.0)
    parser.add_argument("--rho-min", type=float, default=None, help="Minimum random object rho. Defaults to scene-derived value.")
    parser.add_argument("--rho-max", type=float, default=None, help="Maximum random object rho. Defaults to scene-derived value.")
    parser.add_argument("--phi-min-deg", type=float, default=None, help="Minimum random object phi in degrees. Defaults to 0.")
    parser.add_argument("--phi-max-deg", type=float, default=None, help="Maximum random object phi in degrees. Defaults to 180.")
    parser.add_argument("--camera-fov", dest="camera_FOV", type=float, default=1.0)
    parser.add_argument("--cam-pixel-dim", type=int, default=64)
    parser.add_argument("--bin-size", type=float, default=3.9e-10)
    parser.add_argument("--laser-intensity", type=float, default=1000.0)
    parser.add_argument("--hide-walls", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--snr-db", dest="SNR_dB", type=float, default=30.0)
    parser.add_argument("--sbr", dest="SBR", type=float, default=5.0)
    parser.add_argument("--poisson-scale-factor", type=float, default=1000.0)
    parser.add_argument("--add-noise", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    num_scenes = args.num_scenes if args.num_scenes is not None else args.num_scenes_pos
    if num_scenes is None:
        num_scenes = 1
    if num_scenes <= 0:
        raise ValueError("num_scenes must be greater than 0")
    if args.num_workers <= 0:
        raise ValueError("num_workers must be greater than 0")
    if (args.rho_min is None) != (args.rho_max is None):
        raise ValueError("Pass both --rho-min and --rho-max, or neither.")
    if (args.phi_min_deg is None) != (args.phi_max_deg is None):
        raise ValueError("Pass both --phi-min-deg and --phi-max-deg, or neither.")
    rho_range = None if args.rho_min is None else (args.rho_min, args.rho_max)
    phi_range = None if args.phi_min_deg is None else (float(np.deg2rad(args.phi_min_deg)), float(np.deg2rad(args.phi_max_deg)))
    if rho_range is not None and not (rho_range[0] > 0 and rho_range[0] < rho_range[1]):
        raise ValueError("Require 0 < --rho-min < --rho-max")
    if phi_range is not None and not (phi_range[0] < phi_range[1]):
        raise ValueError("Require --phi-min-deg < --phi-max-deg")
    torch_device = resolve_torch_device(args.device)
    torch_dtype = torch.float64 if args.torch_dtype == "float64" else torch.float32

    start_time = time.time()
    generate_transient_dataset(
        num_scenes=num_scenes,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        zmin=args.zmin,
        zmax=args.zmax,
        camera_FOV=args.camera_FOV,
        cam_pixel_dim=args.cam_pixel_dim,
        bin_size=args.bin_size,
        laser_intensity=args.laser_intensity,
        hide_walls=args.hide_walls,
        SNR_dB=args.SNR_dB,
        SBR=args.SBR,
        poisson_scale_factor=args.poisson_scale_factor,
        add_noise=args.add_noise,
        output_dir=args.output_dir,
        object_folder=args.object_folder,
        seed=args.seed,
        max_trials_per_scene=args.max_trials_per_scene,
        num_workers=args.num_workers,
        rho_range=rho_range,
        phi_range=phi_range,
        torch_device=torch_device,
        torch_dtype=torch_dtype,
        triangle_chunk_size=args.triangle_chunk_size,
    )
    print(f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
