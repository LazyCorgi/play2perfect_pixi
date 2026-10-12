#!/usr/bin/env python3
"""Load the calibrated UR5e-O30 scene in Isaac and report robot contacts."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

from ur5e_o30_scene_state import DEFAULT_STATE_PATH, REPO_ROOT, load_scene_state


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_PATH)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg


ROBOT_URDF = REPO_ROOT / "assets/urdf/ur5e_linker_o30/ur5e_linker_o30_sim.urdf"
TABLE_URDF = REPO_ROOT / "assets/urdf/table_narrow.urdf"
HOLE_URDF = REPO_ROOT / "assets/urdf/peg_in_hole/holes/hole_tol0p5mm/hole_tol0p5mm.urdf"
PEG_URDF = REPO_ROOT / "assets/urdf/peg_in_hole/lpeg_matchedmass/lpeg_matchedmass.urdf"


def urdf_cfg(path: Path, *, contact_sensors: bool = False) -> sim_utils.UrdfFileCfg:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sim_utils.UrdfFileCfg(
        asset_path=str(path.resolve()),
        fix_base=True,
        merge_fixed_joints=True,
        make_instanceable=False,
        self_collision=False,
        collider_type="convex_hull",
        activate_contact_sensors=contact_sensors,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
    )


def main() -> None:
    state = load_scene_state(args.state_file)
    robot_state = state["robot"]
    base_pose = robot_state["base_pose"]

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    )
    sim.set_camera_view(eye=(1.8, 2.2, 1.5), target=(0.0, 0.25, 0.55))
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=urdf_cfg(ROBOT_URDF, contact_sensors=True),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(base_pose["position"]),
            rot=tuple(base_pose["rotation_wxyz"]),
            joint_pos=robot_state["joint_positions"],
            joint_vel={".*": 0.0},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_.*", "elbow_joint", "wrist_.*"],
                effort_limit_sim=150.0,
                velocity_limit_sim=6.28,
                stiffness=400.0,
                damping=40.0,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["thumb_.*", "index_.*", "middle_.*", "ring_.*", "pinky_.*"],
                effort_limit_sim=1.0,
                velocity_limit_sim=1.0,
                stiffness=5.0,
                damping=0.2,
            ),
        },
    )
    robot = Articulation(robot_cfg)

    for prim_name, key, path in (
        ("Table", "table", TABLE_URDF),
        ("Hole", "hole", HOLE_URDF),
        ("Peg", "peg", PEG_URDF),
    ):
        pose = state[key]
        cfg = urdf_cfg(path)
        cfg.func(
            f"/World/{prim_name}",
            cfg,
            translation=tuple(pose["position"]),
            orientation=tuple(pose["rotation_wxyz"]),
        )

    contact_sensor = ContactSensor(
        ContactSensorCfg(
            prim_path="/World/Robot/.*",
            update_period=0.0,
            history_length=1,
            track_contact_points=False,
            debug_vis=True,
        )
    )

    sim.reset()
    target = robot.data.default_joint_pos.clone()
    zeros = torch.zeros_like(target)
    robot.write_joint_state_to_sim(target, zeros)
    robot.set_joint_position_target(target)
    robot.write_data_to_sim()

    print("\nRobot joints in Isaac order:")
    for name, value in zip(robot.data.joint_names, target[0].tolist(), strict=True):
        print(f"  {name:24s} {value:+.6f}")
    print("\nCalibrated scene poses:")
    print(f"  robot: {base_pose}")
    for key in ("table", "hole", "peg"):
        print(f"  {key:6s}: {state[key]}")
    print("\nContact arrows or terminal CONTACT lines indicate robot-scene contact.\n")

    dt = sim.get_physics_dt()
    frame = 0
    max_seen_force = 0.0
    try:
        while simulation_app.is_running():
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)
            contact_sensor.update(dt, force_recompute=True)

            forces = torch.linalg.vector_norm(contact_sensor.data.net_forces_w[0], dim=-1)
            max_force = float(forces.max().item()) if forces.numel() else 0.0
            max_seen_force = max(max_seen_force, max_force)
            if frame % 30 == 0 and max_force > 0.1:
                hit_ids = torch.nonzero(forces > 0.1).flatten().tolist()
                hits = [
                    f"{contact_sensor.body_names[index]}={forces[index].item():.2f}N"
                    for index in hit_ids
                ]
                print("CONTACT", ", ".join(hits), flush=True)
            frame += 1
    finally:
        print(f"Maximum observed robot contact force: {max_seen_force:.3f} N")
        simulation_app.close()


if __name__ == "__main__":
    main()
