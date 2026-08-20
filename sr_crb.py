import argparse

from utils.crb_utils import (
    ROOT,
    SimulationConfig,
    add_common_arguments,
    parse_int_list,
    plot_range_angle_sweep,
    run_sweep,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep camera spatial resolution and plot range/angle CRB behavior.",
    )
    parser.add_argument("--pixel-dims", type=parse_int_list, default=parse_int_list("8,16,32,64"))
    parser.add_argument("--camera-fov", type=float, default=0.25)
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or ROOT / "plots" / "sr_range_angle_crb.png"

    results = run_sweep(
        values=args.pixel_dims,
        config_factory=lambda pixel_dim: SimulationConfig(camera_FOV=args.camera_fov, cam_pixel_dim=int(pixel_dim)),
        rho=args.rho,
        phi_deg=args.phi_deg,
        height=args.height,
        facet_width=args.facet_width,
        range_step=args.range_step,
        angle_step_deg=args.angle_step_deg,
    )
    plot_range_angle_sweep(
        results=results,
        x_label="Pixel dimension",
        image_title_prefix="Pixel dimension",
        output_path=output,
    )


if __name__ == "__main__":
    main()
