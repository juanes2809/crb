"""Baseline configuration and adapter to drive the FD CRB with the torch simulator.

`crb_polar_functions.simulate_facet_signal_expected` unpacks three values from the
simulation callable (`_, y_signal, params = simulation_fn(**kwargs)`), a legacy of
the notebook simulator that returned `(noisy, clean, params)`. `simulator.simulation`
returns two values, `(y_meas_vec_noisy, params)`. Rather than touching either side,
`simulation_fn_for_crb` sits in between and re-emits the 2-tuple as the 3-tuple the
CRB expects. The CRB always calls the forward with `force_no_noise=True`, which sets
`add_noise=False`, so the single returned transient IS the noiseless expected signal
and can legitimately serve as both slots.

The parameter vector is psi = [rho, phi] only. The torch simulator scales the facet
mesh with `min(w / extents[0], 1.1 / extents[2])` and never reads an `h` key, so the
object height is not a controllable input and no `sigma_h` can be defined: the Fisher
information is 2x2 and `estimate_height` must stay False.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

import simulator

BASELINE_CAM_PIXEL_DIM = 32

# Scene/sensor constants as used by simulation_polar_clean.ipynb (cell 3 and 5),
# except for cam_pixel_dim, which the baseline lowers from 64 to 32.
BASELINE_SCENE = {
    "xmin": -1.5,
    "xmax": 1.5,
    "ymin": 0.0,
    "ymax": 3.0,
    "zmin": 0.0,
    "zmax": 3.0,
    "camera_FOV": 0.25,
    "bin_size": 3.9e-10,
    "laser_intensity": 1000.0,
    "hide_walls": True,
    "SNR_dB": 30.0,
    "SBR": 5.0,
    "poisson_scale_factor": 1000.0,
    "add_noise": False,
}

# Facet and CRB estimator settings inherited from regenerate_crb_standard_regions.py.
BASELINE_FACET_WIDTH = 0.5
BASELINE_BACKGROUND_RATE = 0.1
BASELINE_RANGES = np.array([0.5, 1.0, 1.5])
BASELINE_ANGLES_DEG = np.array([30.0, 60.0, 90.0, 120.0, 150.0])
BASELINE_FD_STEPS = np.array([0.01, np.deg2rad(0.5)])


def simulation_fn_for_crb(**kwargs: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Adapt `simulator.simulation`'s 2-tuple to the CRB's 3-tuple contract."""
    if not kwargs.get("add_noise", False):
        y, params = simulator.simulation(**kwargs)
        return y, y, params
    raise ValueError(
        "The CRB forward must be noiseless; call with add_noise=False so that the "
        "returned transient is the expected signal."
    )


def baseline_simulation_kwargs(
    cam_pixel_dim: int = BASELINE_CAM_PIXEL_DIM,
    torch_device: torch.device | None = None,
    torch_dtype: torch.dtype = torch.float64,
    triangle_chunk_size: int = 256,
) -> Dict[str, Any]:
    """Simulator kwargs for the FD CRB baseline.

    float64 is the default because the CRB differentiates by central differences with
    steps of 1 cm / 0.5 deg, and float32 rounding is a sizeable fraction of the
    resulting bin-intensity differences.
    """
    if torch_device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        **BASELINE_SCENE,
        "cam_pixel_dim": int(cam_pixel_dim),
        "object_positions": [],
        "torch_device": torch_device,
        "torch_dtype": torch_dtype,
        "triangle_chunk_size": int(triangle_chunk_size),
    }
