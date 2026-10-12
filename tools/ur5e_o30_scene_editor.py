#!/usr/bin/env python3
"""Viser editor for the calibrated UR5e-O30 training scene."""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

from ur5e_o30_scene_state import DEFAULT_STATE_PATH, REPO_ROOT, load_scene_state, save_scene_state


ROBOT_URDF = REPO_ROOT / "assets/urdf/ur5e_linker_o30/ur5e_linker_o30_sim.urdf"
TABLE_URDF = REPO_ROOT / "assets/urdf/table_narrow.urdf"
HOLE_URDF = REPO_ROOT / "assets/urdf/peg_in_hole/holes/hole_tol0p5mm/hole_tol0p5mm.urdf"
PEG_URDF = REPO_ROOT / "assets/urdf/peg_in_hole/lpeg_matchedmass/lpeg_matchedmass.urdf"


def yaw_from_wxyz(rotation: list[float]) -> float:
    w, x, y, z = rotation
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def add_pose_controls(server, label, frame, pose, *, xy_limit, z_limits):
    x, y, z = pose["position"]
    with server.gui.add_folder(label, expand_by_default=label in ("Robot base", "Table")):
        handles = {
            "x": server.gui.add_slider("x (m)", min=-xy_limit, max=xy_limit, step=0.001, initial_value=x),
            "y": server.gui.add_slider("y (m)", min=-xy_limit, max=xy_limit, step=0.001, initial_value=y),
            "z": server.gui.add_slider("z (m)", min=z_limits[0], max=z_limits[1], step=0.001, initial_value=z),
            "yaw": server.gui.add_slider(
                "yaw (rad)", min=-math.pi, max=math.pi, step=0.001,
                initial_value=yaw_from_wxyz(pose["rotation_wxyz"]),
            ),
        }

    def update(_) -> None:
        yaw = handles["yaw"].value
        frame.position = (handles["x"].value, handles["y"].value, handles["z"].value)
        frame.wxyz = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))

    for handle in handles.values():
        handle.on_update(update)
    return handles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_PATH)
    args = parser.parse_args()
    state = load_scene_state(args.state_file)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.add_markdown("# UR5e + Linker O30 scene editor")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (1.6, 2.0, 1.3)
        client.camera.look_at = (0.0, 0.25, 0.5)

    server.scene.add_grid("/ground", width=2.0, height=2.0, cell_size=0.1)

    robot_pose = state["robot"]["base_pose"]
    robot_frame = server.scene.add_frame(
        "/robot",
        position=tuple(robot_pose["position"]),
        wxyz=tuple(robot_pose["rotation_wxyz"]),
        show_axes=False,
    )
    robot = ViserUrdf(server, ROBOT_URDF.resolve(), root_node_name="/robot")

    scene_specs = {
        "table": ("Table", TABLE_URDF),
        "hole": ("Hole", HOLE_URDF),
        "peg": ("Peg", PEG_URDF),
    }
    frames = {}
    for key, (_, urdf_path) in scene_specs.items():
        pose = state[key]
        frames[key] = server.scene.add_frame(
            f"/{key}",
            position=tuple(pose["position"]),
            wxyz=tuple(pose["rotation_wxyz"]),
            show_axes=False,
        )
        ViserUrdf(server, urdf_path.resolve(), root_node_name=f"/{key}")

    pose_controls = {
        "robot": add_pose_controls(
            server, "Robot base", robot_frame, robot_pose, xy_limit=2.0, z_limits=(-1.0, 1.0)
        ),
        "table": add_pose_controls(
            server, "Table", frames["table"], state["table"], xy_limit=1.0, z_limits=(-0.5, 1.5)
        ),
        "hole": add_pose_controls(
            server, "Hole", frames["hole"], state["hole"], xy_limit=1.0, z_limits=(-0.5, 1.5)
        ),
        "peg": add_pose_controls(
            server, "Peg", frames["peg"], state["peg"], xy_limit=1.0, z_limits=(-0.5, 1.5)
        ),
    }

    names = robot.get_actuated_joint_names()
    limits = robot.get_actuated_joint_limits()
    saved_joints = state["robot"]["joint_positions"]
    joint_sliders = {}
    for folder, selected_names, expanded in (("Arm", names[:6], True), ("Hand", names[6:], False)):
        with server.gui.add_folder(folder, expand_by_default=expanded):
            for name in selected_names:
                lower, upper = limits[name]
                joint_sliders[name] = server.gui.add_slider(
                    name, min=float(lower), max=float(upper), step=0.001,
                    initial_value=float(np.clip(saved_joints.get(name, 0.0), lower, upper)),
                )

    def configuration() -> np.ndarray:
        return np.array([joint_sliders[name].value for name in names], dtype=np.float64)

    def update_robot(_) -> None:
        robot.update_cfg(configuration())

    for slider in joint_sliders.values():
        slider.on_update(update_robot)

    status = server.gui.add_markdown(f"Scene state: `{args.state_file}`")
    save_button = server.gui.add_button("Save scene as training default")

    @save_button.on_click
    def _(_) -> None:
        def pose_from(frame):
            return {
                "position": [float(value) for value in frame.position],
                "rotation_wxyz": [float(value) for value in frame.wxyz],
            }

        state["robot"] = {
            "base_pose": pose_from(robot_frame),
            "joint_positions": dict(zip(names, configuration().tolist(), strict=True)),
        }
        for key in ("table", "hole", "peg"):
            state[key] = pose_from(frames[key])
        save_scene_state(state, args.state_file)
        status.content = f"Saved training scene to `{args.state_file}`"
        print(f"Saved training scene: {args.state_file}", flush=True)

    robot.update_cfg(configuration())
    print(f"Open http://localhost:{server.get_port()}", flush=True)
    print(f"Scene state: {args.state_file}", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
