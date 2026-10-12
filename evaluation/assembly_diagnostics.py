"""Evaluation-only industrial diagnostics for precise assembly rollouts.

The sampler is attached to ``scene.update`` so contact loads are reduced at the
physics rate without changing the policy, rewards, observations, or control
cadence.  Every episode gets a compact JSON summary; event rows are sparse and
continuous traces are opt-in per environment.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


CATEGORIES = ("finger_object", "object_fixture", "hand_fixture")
PHASES = ("transport_approach", "insertion_adjustment")
BASE_TRACE_COLUMNS = (
    "time_s",
    "phase",
    "attempt_id",
    "depth_m",
    "lateral_error_m",
    "object_x_m",
    "object_y_m",
    "object_z_m",
    "finger_object_normal_N",
    "finger_object_tangent_N",
    "object_fixture_normal_N",
    "object_fixture_tangent_N",
    "hand_fixture_normal_N",
    "hand_fixture_tangent_N",
    "goal_translation_error_m",
    "goal_rotation_error_rad",
    "keypoint_error_m",
    "palm_object_translation_change_m",
    "palm_object_rotation_change_rad",
    "arm_action_l2",
    "hand_action_l2",
    "screw_axial_feed_m",
    "screw_forward_rotation_rad",
    "screw_reverse_rotation_rad",
)


@dataclass(frozen=True)
class DetectorConfig:
    """One authoritative set of scene-scaled event detector thresholds."""

    contact_threshold_n: float
    high_force_threshold_n: float
    anomalous_force_threshold_n: float
    anomalous_impulse_threshold_ns: float
    micro_backout_m: float
    retry_backout_m: float
    entry_band_m: float
    attempt_lateral_corridor_m: float
    drop_lateral_corridor_m: float
    entry_depth_limit_m: float
    lateral_adjust_speed_mps: float
    adjustment_merge_gap_s: float
    grasp_on_s: float
    grasp_off_s: float
    fixture_contact_on_s: float
    fixture_contact_off_s: float


def _quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[..., 1:] *= -1.0
    return result


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _quat_angle_wxyz(quat: torch.Tensor) -> torch.Tensor:
    normalized = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-9)
    return 2.0 * torch.atan2(
        torch.linalg.vector_norm(normalized[..., 1:], dim=-1),
        normalized[..., 0].abs().clamp_min(1e-9),
    )


def _relative_pose(
    parent_pos: torch.Tensor,
    parent_quat: torch.Tensor,
    child_pos: torch.Tensor,
    child_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inverse = _quat_conjugate_wxyz(parent_quat)
    return (
        _quat_apply_wxyz(inverse, child_pos - parent_pos),
        _quat_mul_wxyz(inverse, child_quat),
    )


def parse_trace_envs(value: str, num_envs: int) -> tuple[int, ...]:
    if not value.strip():
        return ()
    ids = tuple(sorted({int(part) for part in value.split(",") if part.strip()}))
    invalid = [env_id for env_id in ids if env_id < 0 or env_id >= num_envs]
    if invalid:
        raise ValueError(f"trace env ids out of range for num_envs={num_envs}: {invalid}")
    return ids


def _env_glob(path: str) -> str:
    """Turn one resolved cloned-env path into a PhysX glob."""
    return re.sub(r"/env_\d+/", "/env_*/", str(path), count=1)


def _quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    q_xyz = quat[:, 1:4]
    uv = torch.cross(q_xyz, vec, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return vec + 2.0 * (quat[:, 0:1] * uv + uuv)


class AssemblyDiagnostics:
    """Collect first-pass and contact-safety metrics for evaluation rollouts."""

    def __init__(
        self,
        env,
        output_dir: str | Path,
        *,
        trace_envs: Iterable[int] = (0,),
        max_run_size_mb: float = 100.0,
        contact_threshold_n: float = 0.5,
        high_force_threshold_n: float = 50.0,
        anomalous_force_threshold_n: float = 500.0,
        anomalous_impulse_threshold_ns: float = 100.0,
        max_contact_points_per_env: int = 64,
        checkpoint: str | None = None,
    ) -> None:
        self.env = env
        self.device = env.device
        self.num_envs = int(env.num_envs)
        self.physics_dt = float(env.physics_dt)
        self.trace_decimation = max(1, int(env.cfg.decimation))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "traces").mkdir(exist_ok=True)
        self.trace_envs = tuple(trace_envs)
        self.max_run_bytes = int(max_run_size_mb * 1024 * 1024)
        self.trace_bytes_written = 0
        self.max_contact_points_per_env = int(max_contact_points_per_env)
        self.checkpoint = checkpoint

        problem = env._pih_problem
        axis = torch.tensor(problem.insertion_direction, device=self.device, dtype=torch.float32)
        self.axis_hole = axis / axis.norm().clamp_min(1.0e-9)
        self.final_pos_hole = env._insert_pos_rel[-1].clone()
        self.pre_insert_offset = float(problem.pre_insert_offset)
        self.detector = DetectorConfig(
            contact_threshold_n=float(contact_threshold_n),
            high_force_threshold_n=float(high_force_threshold_n),
            anomalous_force_threshold_n=float(anomalous_force_threshold_n),
            anomalous_impulse_threshold_ns=float(anomalous_impulse_threshold_ns),
            micro_backout_m=max(1.0e-4, 0.01 * self.pre_insert_offset),
            retry_backout_m=max(0.002, 0.10 * self.pre_insert_offset),
            entry_band_m=max(5.0e-4, 0.01 * self.pre_insert_offset),
            attempt_lateral_corridor_m=max(0.005, 0.20 * self.pre_insert_offset),
            drop_lateral_corridor_m=max(0.01, 0.60 * self.pre_insert_offset),
            entry_depth_limit_m=max(0.002, 0.10 * self.pre_insert_offset),
            lateral_adjust_speed_mps=0.005,
            adjustment_merge_gap_s=0.05,
            grasp_on_s=2.0 * self.physics_dt,
            grasp_off_s=0.05,
            fixture_contact_on_s=2.0 * self.physics_dt,
            fixture_contact_off_s=0.025,
        )
        # Kept as aliases so the vectorized detector remains readable.
        self.contact_threshold_n = self.detector.contact_threshold_n
        self.high_force_threshold_n = self.detector.high_force_threshold_n
        self.micro_deadband = self.detector.micro_backout_m
        self.retry_backout = self.detector.retry_backout_m
        self.entry_band = self.detector.entry_band_m
        self.attempt_lateral_corridor = self.detector.attempt_lateral_corridor_m
        self.drop_lateral_corridor = self.detector.drop_lateral_corridor_m
        self.entry_depth_limit = self.detector.entry_depth_limit_m
        self.lateral_adjust_speed = self.detector.lateral_adjust_speed_mps
        self.adjust_merge_gap_s = self.detector.adjustment_merge_gap_s
        self.grasp_on_s = self.detector.grasp_on_s
        self.grasp_off_s = self.detector.grasp_off_s

        self.is_screwing = self.env._pih_problem_name == "screwing"
        self.screw_progress_sign = 1.0
        if self.is_screwing and env._insert_quat_wxyz.shape[0] > 1:
            goal_quats = env._insert_quat_wxyz
            goal_delta = _quat_mul_wxyz(
                _quat_conjugate_wxyz(goal_quats[:-1]), goal_quats[1:]
            )
            signed_goal_rotation = 2.0 * torch.atan2(
                (goal_delta[:, 1:] * self.axis_hole).sum(dim=-1), goal_delta[:, 0]
            ).sum()
            if abs(float(signed_goal_rotation.item())) > 1e-6:
                self.screw_progress_sign = math.copysign(1.0, float(signed_goal_rotation.item()))
        self._palm_body_id = int(env._palm_body_id)
        self._joint_names = tuple(env.robot.data.joint_names)
        self._latest_policy_action = torch.zeros(
            self.num_envs, int(env.cfg.action_space), device=self.device
        )
        pose_names = (
            "world_palm", "world_object", "palm_object",
            "world_fixture", "fixture_object", "world_goal",
        )
        self.trace_columns = BASE_TRACE_COLUMNS + tuple(
            f"T_{name}_{component}"
            for name in pose_names
            for component in ("x_m", "y_m", "z_m", "qw", "qx", "qy", "qz")
        ) + tuple(
            f"{prefix}_{name}"
            for prefix in ("joint_pos_rad", "joint_vel_radps", "joint_target_rad")
            for name in self._joint_names
        ) + tuple(f"policy_action_{idx}" for idx in range(int(env.cfg.action_space)))

        self._views = self._create_contact_views()
        self._original_scene_update = env.scene.update
        self._attached = False
        self._physics_step_index = 0
        self._episode_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._trace_rows: dict[int, list[list[float]]] = {env_id: [] for env_id in self.trace_envs}
        self._events: list[list[dict]] = [[] for _ in range(self.num_envs)]
        self._allocate_state()
        self._write_metadata()

    def _create_contact_views(self) -> dict[str, list]:
        from isaacsim.core.simulation_manager import SimulationManager

        sim_view = SimulationManager.get_physics_sim_view()
        object_path = _env_glob(self.env.object.root_physx_view.prim_paths[0])
        hole_path = _env_glob(self.env.hole.root_physx_view.prim_paths[0])
        table_path = _env_glob(self.env.table.root_physx_view.prim_paths[0])

        link_names = list(self.env.robot.root_physx_view.shared_metatype.link_names)
        link_paths = list(self.env.robot.root_physx_view.link_paths[0])
        path_by_name = {name: _env_glob(path) for name, path in zip(link_names, link_paths)}
        fingertip_paths = [
            path_by_name[name]
            for name in link_names
            if re.fullmatch(r"left_(index|middle|ring|thumb|pinky)_DP", name)
        ]
        hand_paths = [
            path_by_name[name]
            for name in link_names
            if name == "iiwa14_link_7" or name.startswith("left_")
        ]
        if len(fingertip_paths) != 5:
            raise RuntimeError(
                f"diagnostics expected five merged fingertip bodies, got {fingertip_paths}"
            )

        max_contacts = self.max_contact_points_per_env * self.num_envs

        def make(source: str, filters: list[str]):
            return sim_view.create_rigid_contact_view(
                source,
                filter_patterns=filters,
                max_contact_data_count=max_contacts,
            )

        return {
            "finger_object": [make(object_path, fingertip_paths)],
            "object_fixture": [make(object_path, [hole_path, table_path])],
            "hand_fixture": [make(hole_path, hand_paths), make(table_path, hand_paths)],
        }

    def _allocate_state(self) -> None:
        e, p, c = self.num_envs, len(PHASES), len(CATEGORIES)
        zeros = lambda *shape: torch.zeros(*shape, device=self.device)
        bools = lambda *shape: torch.zeros(*shape, dtype=torch.bool, device=self.device)
        longs = lambda *shape: torch.zeros(*shape, dtype=torch.long, device=self.device)

        self.elapsed = zeros(e)
        self.normal_peak = zeros(e, p, c)
        self.net_force_peak = zeros(e, p, c)
        self.tangent_peak = zeros(e, p, c)
        self.normal_peak_time = torch.full((e, p, c), float("nan"), device=self.device)
        self.normal_peak_point = zeros(e, p, c, 3)
        self.normal_impulse = zeros(e, p, c)
        self.tangent_impulse = zeros(e, p, c)
        self.contact_time = zeros(e, p, c)
        self.high_force_time = zeros(e, p, c)
        self.high_force_run = zeros(e, p, c)
        self.max_high_force_run = zeros(e, p, c)
        self.contact_episode_impulse = zeros(e, p, c)
        self.max_contact_episode_impulse = zeros(e, p, c)
        self.contact_active_prev = bools(e, p, c)
        self.force_anomaly_emitted = bools(e, p, c)

        self.grasp_dwell = zeros(e)
        self.grasp_off_dwell = zeros(e)
        self.grasp_stable = bools(e)
        self.grasp_ever = bools(e)
        self.grasp_lost_pending = bools(e)
        self.drop_reported_for_loss = bools(e)
        self.grasp_time = torch.full((e,), float("nan"), device=self.device)
        self.lift_time = torch.full((e,), float("nan"), device=self.device)
        self.object_lifted_seen = bools(e)
        self.drop_count = longs(e)
        self.regrasp_count = longs(e)
        self.grasp_restore_count = longs(e)

        self.fixture_contact_on_dwell = zeros(e)
        self.fixture_contact_off_dwell = zeros(e)
        self.fixture_contact_stable = bools(e)

        self.insertion_started = bools(e)
        self.insertion_start_time = torch.full((e,), float("nan"), device=self.device)
        self.assembly_success_seen = bools(e)
        self.seat_time = torch.full((e,), float("nan"), device=self.device)
        self.first_fixture_contact_time = torch.full((e,), float("nan"), device=self.device)
        self.attempt_count = longs(e)
        self.retry_count = longs(e)
        self.backtrack_candidate_count = longs(e)
        self.recovery_count = longs(e)
        self.task_restart_count = longs(e)
        self.retry_pending = bools(e)
        self.max_depth = torch.full((e,), -float("inf"), device=self.device)
        self.prev_depth = zeros(e)
        self.prev_lateral_vec = zeros(e, 3)
        self.have_prev_geometry = bools(e)
        self.reverse_distance_raw = zeros(e)
        self.reverse_distance_effective = zeros(e)
        self.outward_run = zeros(e)
        self.backoff_count = longs(e)
        self.contact_search_time = zeros(e)
        self.adjustment_count = longs(e)
        self.adjust_active = bools(e)
        self.adjust_gap = torch.full((e,), self.adjust_merge_gap_s, device=self.device)
        self.recovery_active = bools(e)
        self.recovery_start_time = torch.full((e,), float("nan"), device=self.device)
        self.recovery_duration = zeros(e)

        self.goal_count = longs(e)
        self.goal_reached_time = torch.full((e,), float("nan"), device=self.device)

        self.insertion_start_depth = zeros(e)
        self.screw_axial_feed = zeros(e)
        self.screw_forward_rotation = zeros(e)
        self.screw_reverse_rotation = zeros(e)
        self.prev_fixture_object_quat = zeros(e, 4)
        self.prev_fixture_object_quat[:, 0] = 1.0
        self.have_prev_screw_quat = bools(e)

        self.palm_object_reference_pos = zeros(e, 3)
        self.palm_object_reference_quat = zeros(e, 4)
        self.palm_object_reference_quat[:, 0] = 1.0
        self.have_palm_object_reference = bools(e)

        self.latest_depth = zeros(e)
        self.latest_lateral = zeros(e)
        self.latest_normal = zeros(e, c)
        self.latest_tangent = zeros(e, c)
        self.latest_contact_point = zeros(e, c, 3)
        self.latest_object_pos = zeros(e, 3)
        self.latest_object_quat = zeros(e, 4)
        self.latest_object_quat[:, 0] = 1.0
        self.latest_palm_pos = zeros(e, 3)
        self.latest_palm_quat = zeros(e, 4)
        self.latest_palm_quat[:, 0] = 1.0
        self.latest_fixture_pos = zeros(e, 3)
        self.latest_fixture_quat = zeros(e, 4)
        self.latest_fixture_quat[:, 0] = 1.0
        self.latest_goal_pos = zeros(e, 3)
        self.latest_goal_quat = zeros(e, 4)
        self.latest_goal_quat[:, 0] = 1.0
        self.latest_palm_object_pos = zeros(e, 3)
        self.latest_palm_object_quat = zeros(e, 4)
        self.latest_palm_object_quat[:, 0] = 1.0
        self.latest_fixture_object_pos = zeros(e, 3)
        self.latest_fixture_object_quat = zeros(e, 4)
        self.latest_fixture_object_quat[:, 0] = 1.0
        self.latest_goal_translation_error = zeros(e)
        self.latest_goal_rotation_error = zeros(e)
        self.latest_keypoint_error = zeros(e)
        self.latest_palm_object_translation_change = zeros(e)
        self.latest_palm_object_rotation_change = zeros(e)
        self.latest_net_force = zeros(e, c)
        self.contact_episode_time = zeros(e, p, c)

    def _write_metadata(self) -> None:
        metadata = {
            "schema_version": 3,
            "problem": self.env._pih_problem_name,
            "checkpoint": self.checkpoint,
            "seed": getattr(self.env.cfg, "seed", None),
            "units": {"length": "m", "time": "s", "force": "N", "impulse": "N*s"},
            "physics_dt_s": self.physics_dt,
            "control_dt_s": float(self.env.step_dt),
            "trace_rate_hz": 1.0 / (self.physics_dt * self.trace_decimation),
            "trace_columns": self.trace_columns,
            "phases": PHASES,
            "contact_categories": CATEGORIES,
            "thresholds": asdict(self.detector),
            "storage": {
                "trace_envs": self.trace_envs,
                "max_trace_budget_mb": self.max_run_bytes / (1024 * 1024),
                "raw_contact_windows": False,
            },
            "contact_semantics": {
                "installed_interface": "omni.physics.tensors RigidContactView",
                "normal_force": "sum(abs(per-contact scalar normal force)); API converts solver impulse to force using physics_dt",
                "net_pair_force": "norm(pairwise vector force) summed over filters; retained separately from total normal load",
                "tangential_force": "sum(norm(per-friction-anchor force)); API converts solver impulse to force using physics_dt",
                "contact_point": "normal-force-weighted world-frame point for the category",
                "contact_event_impulse": "physics-rate integral of total normal force over one contiguous thresholded category contact episode",
                "sampling": "one sample after every scene.update physics step; no control-rate re-integration",
                "coordinate_frame": "forces reduced to magnitudes; contact points are world-frame",
            },
            "capabilities": {
                "pair_filtered_contact_force": True,
                "contact_points": True,
                "slip_velocity": False,
                "friction_energy": False,
                "task_restart_observable": False,
                "screwing_first_pass_verified": False,
            },
            "metric_definitions": {
                "insertion_retry_count": "number of inward re-entry events after a retry-threshold backtrack",
                "recovery_event_count": "recovery initiations: retry-threshold insertion backtrack + object drop + observed task restart",
                "local_adjustment_count": "merged lateral-motion bouts during stable fixture contact; never counted as recovery",
                "task_completion_time_s": "elapsed time only for task-success rollouts; failures are right-censored",
                "screwing_retry_fields": "effective attempt/retry counts are null; raw geometric backtrack/reengagement candidates are retained without recovery interpretation",
            },
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )

    def attach(self) -> None:
        if self._attached:
            return

        def update_with_diagnostics(dt: float) -> None:
            self._original_scene_update(dt)
            self.physics_step(float(dt))

        self.env.scene.update = update_with_diagnostics
        self._attached = True

    def close(self) -> None:
        if self._attached:
            self.env.scene.update = self._original_scene_update
            self._attached = False

    @staticmethod
    def _detail_indices(counts: torch.Tensor, starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        counts = counts.view(-1).long()
        starts = starts.view(-1).long()
        pair_ids = torch.repeat_interleave(
            torch.arange(counts.numel(), device=counts.device), counts
        )
        block_starts = counts.cumsum(0) - counts
        offsets = torch.arange(pair_ids.numel(), device=counts.device) - block_starts.repeat_interleave(
            counts
        )
        return pair_ids, starts[pair_ids] + offsets

    def _sample_view(
        self, view, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        force_matrix = view.get_contact_force_matrix(dt=dt).view(self.num_envs, -1, 3)
        net_force = torch.linalg.vector_norm(force_matrix, dim=-1).sum(dim=-1)

        forces, points, _, _, counts, starts = view.get_contact_data(dt=dt)
        pair_ids, data_ids = self._detail_indices(counts, starts)
        normal = torch.zeros(self.num_envs, device=self.device)
        point_sum = torch.zeros(self.num_envs, 3, device=self.device)
        point_weight = torch.zeros(self.num_envs, device=self.device)
        if pair_ids.numel():
            values = forces.view(-1).index_select(0, data_ids).abs()
            sensor_ids = pair_ids // int(view.filter_count)
            normal.index_add_(0, sensor_ids, values)
            point_sum.index_add_(
                0, sensor_ids, points.view(-1, 3).index_select(0, data_ids) * values.unsqueeze(-1)
            )
            point_weight.index_add_(0, sensor_ids, values)

        friction, friction_points, counts, starts = view.get_friction_data(dt=dt)
        tangent = torch.zeros(self.num_envs, device=self.device)
        friction_point_sum = torch.zeros_like(point_sum)
        friction_weight = torch.zeros_like(point_weight)
        pair_ids, data_ids = self._detail_indices(counts, starts)
        if pair_ids.numel():
            values = friction.view(-1, 3).index_select(0, data_ids)
            sensor_ids = pair_ids // int(view.filter_count)
            magnitudes = torch.linalg.vector_norm(values, dim=-1)
            tangent.index_add_(0, sensor_ids, magnitudes)
            friction_point_sum.index_add_(
                0,
                sensor_ids,
                friction_points.view(-1, 3).index_select(0, data_ids)
                * magnitudes.unsqueeze(-1),
            )
            friction_weight.index_add_(0, sensor_ids, magnitudes)
        missing_normal_point = (point_weight <= 0.0) & (friction_weight > 0.0)
        point_sum = torch.where(
            missing_normal_point.unsqueeze(-1), friction_point_sum, point_sum
        )
        point_weight = torch.where(missing_normal_point, friction_weight, point_weight)
        return normal, tangent, point_sum, point_weight, net_force

    def _sample_contacts(
        self, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normal = torch.zeros(self.num_envs, len(CATEGORIES), device=self.device)
        tangent = torch.zeros_like(normal)
        net_force = torch.zeros_like(normal)
        point_sum = torch.zeros(self.num_envs, len(CATEGORIES), 3, device=self.device)
        point_weight = torch.zeros_like(normal)
        for category_idx, category in enumerate(CATEGORIES):
            for view in self._views[category]:
                (
                    view_normal, view_tangent, view_point_sum, view_point_weight,
                    view_net_force,
                ) = self._sample_view(view, dt)
                normal[:, category_idx] += view_normal
                tangent[:, category_idx] += view_tangent
                net_force[:, category_idx] += view_net_force
                point_sum[:, category_idx] += view_point_sum
                point_weight[:, category_idx] += view_point_weight
        points = point_sum / point_weight.clamp_min(1.0e-9).unsqueeze(-1)
        return normal, tangent, points, net_force

    def _emit(self, mask: torch.Tensor, event: str, **fields) -> None:
        mask = mask & ~self.finished
        for env_id in mask.nonzero(as_tuple=False).flatten().tolist():
            row = {
                "env_id": env_id,
                "episode": int(self._episode_index[env_id].item()),
                "time_s": float(self.elapsed[env_id].item()),
                "event": event,
                "attempt_id": int(self.attempt_count[env_id].item()),
            }
            for key, value in fields.items():
                if isinstance(value, torch.Tensor):
                    item = value[env_id]
                    row[key] = float(item.item()) if item.numel() == 1 else item.tolist()
                else:
                    row[key] = value
            self._events[env_id].append(row)

    def _finish_outward_runs(self, mask: torch.Tensor) -> None:
        meaningful = mask & (self.outward_run > self.micro_deadband)
        self.backoff_count += meaningful.long()
        self.reverse_distance_effective += torch.where(
            meaningful, self.outward_run - self.micro_deadband, 0.0
        )
        self.outward_run[mask] = 0.0

    def set_policy_action(self, action: torch.Tensor) -> None:
        """Cache canonical raw policy actions for the following physics substeps."""
        self._latest_policy_action.copy_(action.to(self.device))

    def _start_recovery(self, mask: torch.Tensor) -> None:
        start = mask & ~self.recovery_active
        self.recovery_start_time = torch.where(start, self.elapsed, self.recovery_start_time)
        self.recovery_active |= mask

    def _end_recovery(self, mask: torch.Tensor) -> None:
        end = mask & self.recovery_active
        start = torch.nan_to_num(self.recovery_start_time, nan=0.0)
        self.recovery_duration += torch.where(end, self.elapsed - start, 0.0)
        self.recovery_active &= ~end
        self.recovery_start_time = torch.where(
            end, torch.full_like(self.recovery_start_time, float("nan")), self.recovery_start_time
        )

    def _update_kinematics(self) -> None:
        origins = self.env.scene.env_origins
        object_pos = self.env.object.data.root_pos_w - origins
        object_quat = self.env.object.data.root_quat_w
        palm_state = self.env.robot.data.body_state_w[:, self._palm_body_id]
        palm_pos = palm_state[:, :3] - origins
        palm_quat = palm_state[:, 3:7]
        fixture_pos = self.env.hole_pos
        fixture_quat = self.env.hole_quat_wxyz
        goal_pos = self.env.goal_viz.data.root_pos_w - origins
        goal_quat = self.env.goal_viz.data.root_quat_w
        palm_object_pos, palm_object_quat = _relative_pose(
            palm_pos, palm_quat, object_pos, object_quat
        )
        fixture_object_pos, fixture_object_quat = _relative_pose(
            fixture_pos, fixture_quat, object_pos, object_quat
        )
        goal_delta_quat = _quat_mul_wxyz(_quat_conjugate_wxyz(goal_quat), object_quat)

        self.latest_object_pos.copy_(object_pos)
        self.latest_object_quat.copy_(object_quat)
        self.latest_palm_pos.copy_(palm_pos)
        self.latest_palm_quat.copy_(palm_quat)
        self.latest_fixture_pos.copy_(fixture_pos)
        self.latest_fixture_quat.copy_(fixture_quat)
        self.latest_goal_pos.copy_(goal_pos)
        self.latest_goal_quat.copy_(goal_quat)
        self.latest_palm_object_pos.copy_(palm_object_pos)
        self.latest_palm_object_quat.copy_(palm_object_quat)
        self.latest_fixture_object_pos.copy_(fixture_object_pos)
        self.latest_fixture_object_quat.copy_(fixture_object_quat)
        self.latest_goal_translation_error.copy_(
            torch.linalg.vector_norm(object_pos - goal_pos, dim=-1)
        )
        self.latest_goal_rotation_error.copy_(_quat_angle_wxyz(goal_delta_quat))
        keypoint = getattr(self.env, "_keypoints_max_dist", None)
        if keypoint is not None:
            self.latest_keypoint_error.copy_(keypoint)

        new_reference = self.grasp_ever & ~self.have_palm_object_reference
        self.palm_object_reference_pos = torch.where(
            new_reference.unsqueeze(-1), palm_object_pos, self.palm_object_reference_pos
        )
        self.palm_object_reference_quat = torch.where(
            new_reference.unsqueeze(-1), palm_object_quat, self.palm_object_reference_quat
        )
        self.have_palm_object_reference |= new_reference
        self.latest_palm_object_translation_change.copy_(
            torch.linalg.vector_norm(palm_object_pos - self.palm_object_reference_pos, dim=-1)
        )
        rel_change = _quat_mul_wxyz(
            _quat_conjugate_wxyz(self.palm_object_reference_quat), palm_object_quat
        )
        self.latest_palm_object_rotation_change.copy_(_quat_angle_wxyz(rel_change))

    def _update_screw_metrics(self, active_attempt: torch.Tensor, depth: torch.Tensor) -> None:
        if not self.is_screwing:
            return
        quat = self.latest_fixture_object_quat
        delta = _quat_mul_wxyz(_quat_conjugate_wxyz(self.prev_fixture_object_quat), quat)
        signed = 2.0 * torch.atan2(
            (delta[:, 1:] * self.axis_hole).sum(dim=-1), delta[:, 0]
        )
        aligned = self.screw_progress_sign * signed
        valid = active_attempt & self.have_prev_screw_quat
        self.screw_forward_rotation += torch.where(valid, aligned.clamp_min(0.0), 0.0)
        self.screw_reverse_rotation += torch.where(valid, (-aligned).clamp_min(0.0), 0.0)
        self.screw_axial_feed = torch.where(
            active_attempt,
            torch.maximum(self.screw_axial_feed, depth - self.insertion_start_depth),
            self.screw_axial_feed,
        )
        self.prev_fixture_object_quat.copy_(quat)
        self.have_prev_screw_quat |= active_attempt

    def physics_step(self, dt: float) -> None:
        self._physics_step_index += 1
        self.elapsed += dt
        normal, tangent, contact_points, net_force = self._sample_contacts(dt)
        self.latest_normal.copy_(normal)
        self.latest_tangent.copy_(tangent)
        self.latest_net_force.copy_(net_force)
        self.latest_contact_point.copy_(contact_points)
        self._update_kinematics()

        obj_pos = self.latest_object_pos
        hole_pos = self.env.hole_pos
        hole_quat = self.env.hole_quat_wxyz
        axis = _quat_apply_wxyz(
            hole_quat, self.axis_hole.unsqueeze(0).expand(self.num_envs, -1)
        )
        final_pos = hole_pos + _quat_apply_wxyz(
            hole_quat, self.final_pos_hole.unsqueeze(0).expand(self.num_envs, -1)
        )
        delta = final_pos - obj_pos
        remaining = (delta * axis).sum(dim=-1)
        depth = self.pre_insert_offset - remaining
        lateral_vec = delta - remaining.unsqueeze(-1) * axis
        lateral = torch.linalg.vector_norm(lateral_vec, dim=-1)
        self.latest_depth.copy_(depth)
        self.latest_lateral.copy_(lateral)

        finger_contact = normal[:, 0] > self.contact_threshold_n
        fixture_contact_raw = normal[:, 1] > self.contact_threshold_n
        self.fixture_contact_on_dwell = torch.where(
            fixture_contact_raw, self.fixture_contact_on_dwell + dt, 0.0
        )
        self.fixture_contact_off_dwell = torch.where(
            ~fixture_contact_raw, self.fixture_contact_off_dwell + dt, 0.0
        )
        self.fixture_contact_stable |= (
            self.fixture_contact_on_dwell >= self.detector.fixture_contact_on_s
        )
        self.fixture_contact_stable &= ~(
            self.fixture_contact_off_dwell >= self.detector.fixture_contact_off_s
        )
        fixture_contact = self.fixture_contact_stable
        self.grasp_dwell = torch.where(finger_contact, self.grasp_dwell + dt, 0.0)
        self.grasp_off_dwell = torch.where(~finger_contact, self.grasp_off_dwell + dt, 0.0)
        new_grasp = ~self.grasp_stable & (self.grasp_dwell >= self.grasp_on_s)
        first_grasp = new_grasp & ~self.grasp_ever
        regrasp = new_grasp & self.grasp_ever & self.drop_reported_for_loss
        restored = new_grasp & self.grasp_ever & ~self.drop_reported_for_loss
        self._emit(first_grasp, "grasp_established")
        self._emit(first_grasp, "stable_grasp")  # V1 compatibility
        self._emit(regrasp, "object_regrasped")
        self._emit(regrasp, "regrasp")  # V1 compatibility
        self._emit(restored, "grasp_contact_restored")
        self.regrasp_count += regrasp.long()
        self.grasp_restore_count += restored.long()
        self._end_recovery(regrasp)
        self.grasp_time = torch.where(first_grasp, self.elapsed, self.grasp_time)
        self.grasp_ever |= new_grasp
        self.grasp_stable |= new_grasp
        self.grasp_lost_pending &= ~new_grasp
        self.drop_reported_for_loss &= ~new_grasp

        lost = self.grasp_stable & (self.grasp_off_dwell >= self.grasp_off_s)
        self._emit(lost, "grasp_loss")
        self.grasp_stable &= ~lost
        self.grasp_lost_pending |= lost
        self.drop_reported_for_loss = torch.where(
            lost, torch.zeros_like(self.drop_reported_for_loss), self.drop_reported_for_loss
        )

        lifted_now = self.grasp_ever & getattr(
            self.env, "_lifted_object", torch.zeros_like(self.grasp_ever)
        )
        newly_lifted = lifted_now & ~self.object_lifted_seen
        self.object_lifted_seen |= newly_lifted
        self.lift_time = torch.where(newly_lifted, self.elapsed, self.lift_time)
        self._emit(newly_lifted, "object_lifted")

        table_top = self.env._table_z_per_env + 0.15
        low_object = obj_pos[:, 2] < table_top + 0.05
        dropped = (
            self.grasp_lost_pending
            & ~self.drop_reported_for_loss
            & low_object
            & ~self.assembly_success_seen
            & (~self.insertion_started | (lateral > self.drop_lateral_corridor))
        )
        self._emit(dropped, "object_dropped")
        self._emit(dropped, "drop")  # V1 compatibility
        self.drop_count += dropped.long()
        self.recovery_count += dropped.long()
        self._start_recovery(dropped)
        self.drop_reported_for_loss |= dropped

        depth_delta = torch.where(self.have_prev_geometry, depth - self.prev_depth, 0.0)
        armed = self.grasp_ever & (lateral <= self.attempt_lateral_corridor)
        crossed_entry_inward = (
            self.have_prev_geometry
            & (self.prev_depth < 0.0)
            & (depth >= 0.0)
            & (depth_delta > 0.0)
        )
        near_entry_contact = (
            fixture_contact
            & (depth >= -self.entry_band)
            & (depth <= self.entry_depth_limit)
        )
        start_attempt = ~self.insertion_started & armed & (
            crossed_entry_inward | near_entry_contact
        )
        self.insertion_started |= start_attempt
        self.attempt_count += start_attempt.long()
        self.insertion_start_time = torch.where(
            start_attempt, self.elapsed, self.insertion_start_time
        )
        self.insertion_start_depth = torch.where(
            start_attempt, depth, self.insertion_start_depth
        )
        self.max_depth = torch.where(start_attempt, depth, self.max_depth)
        self._emit(start_attempt, "insertion_started", depth_m=depth, lateral_error_m=lateral)
        self._emit(start_attempt, "attempt_start", depth_m=depth, lateral_error_m=lateral)

        first_fixture_contact = (
            self.grasp_ever & fixture_contact & torch.isnan(self.first_fixture_contact_time)
        )
        self.first_fixture_contact_time = torch.where(
            first_fixture_contact, self.elapsed, self.first_fixture_contact_time
        )
        self._emit(first_fixture_contact, "first_fixture_contact", depth_m=depth)
        self.contact_search_time += (self.insertion_started & fixture_contact).float() * dt

        active_attempt = (
            self.insertion_started & ~self.assembly_success_seen & ~self.retry_pending
        )
        self.max_depth = torch.where(active_attempt, torch.maximum(self.max_depth, depth), self.max_depth)
        backout = self.max_depth - depth
        abort = active_attempt & ~self.retry_pending & (backout >= self.retry_backout)
        self.retry_pending |= abort
        self.backtrack_candidate_count += abort.long()
        if not self.is_screwing:
            self.recovery_count += abort.long()
            self._start_recovery(abort)
        self._emit(abort, "insertion_backtracked", backout_m=backout, depth_m=depth)
        self._emit(
            abort,
            "attempt_end",
            outcome="backout_retry",
            backout_m=backout,
            depth_m=depth,
        )

        retry = (
            self.retry_pending
            & (depth_delta > 5.0e-5)
            & armed
            & (fixture_contact | (depth >= 0.0))
        )
        self.retry_pending &= ~retry
        self.attempt_count += retry.long()
        self.retry_count += retry.long()
        self.max_depth = torch.where(retry, depth, self.max_depth)
        if not self.is_screwing:
            self._end_recovery(retry)
        self._emit(retry, "insertion_retried", depth_m=depth, lateral_error_m=lateral)
        self._emit(retry, "attempt_start", depth_m=depth, lateral_error_m=lateral)

        outward = self.have_prev_geometry & (depth_delta < 0.0) & active_attempt
        self.reverse_distance_raw += torch.where(outward, -depth_delta, 0.0)
        self.outward_run += torch.where(outward, -depth_delta, 0.0)
        self._finish_outward_runs(self.have_prev_geometry & ~outward & (self.outward_run > 0.0))

        lateral_speed = torch.where(
            self.have_prev_geometry,
            torch.linalg.vector_norm(lateral_vec - self.prev_lateral_vec, dim=-1) / dt,
            0.0,
        )
        adjusting = active_attempt & fixture_contact & (lateral_speed > self.lateral_adjust_speed)
        new_adjustment = adjusting & ~self.adjust_active & (self.adjust_gap >= self.adjust_merge_gap_s)
        self.adjustment_count += new_adjustment.long()
        self._emit(new_adjustment, "local_adjustment", lateral_speed_mps=lateral_speed)
        self.adjust_gap = torch.where(adjusting, 0.0, self.adjust_gap + dt)
        self.adjust_active = adjusting
        self._update_screw_metrics(active_attempt, depth)

        phase = self.insertion_started.long()
        category_contact = normal > self.contact_threshold_n
        for phase_idx in range(len(PHASES)):
            in_phase = self.grasp_ever & (phase == phase_idx)
            for category_idx in range(len(CATEGORIES)):
                active = in_phase & category_contact[:, category_idx]
                n_load = normal[:, category_idx]
                t_load = tangent[:, category_idx]
                net_load = net_force[:, category_idx]
                previous_peak = self.normal_peak[:, phase_idx, category_idx]
                new_peak = in_phase & (n_load > previous_peak)
                self.normal_peak[:, phase_idx, category_idx] = torch.where(
                    in_phase,
                    torch.maximum(previous_peak, n_load),
                    previous_peak,
                )
                self.normal_peak_time[:, phase_idx, category_idx] = torch.where(
                    new_peak, self.elapsed, self.normal_peak_time[:, phase_idx, category_idx]
                )
                self.normal_peak_point[:, phase_idx, category_idx] = torch.where(
                    new_peak.unsqueeze(-1),
                    contact_points[:, category_idx],
                    self.normal_peak_point[:, phase_idx, category_idx],
                )
                self.net_force_peak[:, phase_idx, category_idx] = torch.where(
                    in_phase,
                    torch.maximum(self.net_force_peak[:, phase_idx, category_idx], net_load),
                    self.net_force_peak[:, phase_idx, category_idx],
                )
                self.tangent_peak[:, phase_idx, category_idx] = torch.where(
                    in_phase,
                    torch.maximum(self.tangent_peak[:, phase_idx, category_idx], t_load),
                    self.tangent_peak[:, phase_idx, category_idx],
                )
                self.normal_impulse[:, phase_idx, category_idx] += in_phase.float() * n_load * dt
                self.tangent_impulse[:, phase_idx, category_idx] += in_phase.float() * t_load * dt
                self.contact_time[:, phase_idx, category_idx] += active.float() * dt

                high = in_phase & (n_load > self.high_force_threshold_n)
                self.high_force_time[:, phase_idx, category_idx] += high.float() * dt
                self.high_force_run[:, phase_idx, category_idx] = torch.where(
                    high, self.high_force_run[:, phase_idx, category_idx] + dt, 0.0
                )
                self.max_high_force_run[:, phase_idx, category_idx] = torch.maximum(
                    self.max_high_force_run[:, phase_idx, category_idx],
                    self.high_force_run[:, phase_idx, category_idx],
                )
                anomalous = (
                    in_phase
                    & (n_load >= self.detector.anomalous_force_threshold_n)
                    & ~self.force_anomaly_emitted[:, phase_idx, category_idx]
                )
                self._emit(
                    anomalous,
                    "anomalous_contact_force",
                    phase=PHASES[phase_idx],
                    category=CATEGORIES[category_idx],
                    normal_N=n_load,
                    net_pair_force_N=net_load,
                    point_w_m=contact_points[:, category_idx],
                )
                self.force_anomaly_emitted[:, phase_idx, category_idx] |= anomalous

                prev_active = self.contact_active_prev[:, phase_idx, category_idx]
                current_impulse = self.contact_episode_impulse[:, phase_idx, category_idx]
                current_impulse = torch.where(active, current_impulse + n_load * dt, current_impulse)
                current_time = self.contact_episode_time[:, phase_idx, category_idx]
                current_time = torch.where(active, current_time + dt, current_time)
                began = active & ~prev_active
                ended = prev_active & ~active
                self._emit(
                    began,
                    "contact_begin",
                    phase=PHASES[phase_idx],
                    category=CATEGORIES[category_idx],
                    point_w_m=contact_points[:, category_idx],
                    normal_N=n_load,
                    tangential_N=t_load,
                )
                self._emit(
                    ended,
                    "contact_end",
                    phase=PHASES[phase_idx],
                    category=CATEGORIES[category_idx],
                    episode_impulse_Ns=current_impulse,
                    episode_duration_s=current_time,
                )
                self.max_contact_episode_impulse[:, phase_idx, category_idx] = torch.where(
                    ended,
                    torch.maximum(
                        self.max_contact_episode_impulse[:, phase_idx, category_idx],
                        current_impulse,
                    ),
                    self.max_contact_episode_impulse[:, phase_idx, category_idx],
                )
                self.contact_episode_impulse[:, phase_idx, category_idx] = torch.where(
                    active, current_impulse, 0.0
                )
                self.contact_episode_time[:, phase_idx, category_idx] = torch.where(
                    active, current_time, 0.0
                )
                self.contact_active_prev[:, phase_idx, category_idx] = active

        self.prev_depth.copy_(depth)
        self.prev_lateral_vec.copy_(lateral_vec)
        self.have_prev_geometry[:] = True
        if self._physics_step_index % self.trace_decimation == 0:
            self._append_trace(phase)

    def _append_trace(self, phase: torch.Tensor) -> None:
        joint_pos = self.env.robot.data.joint_pos
        joint_vel = self.env.robot.data.joint_vel
        joint_target = self.env._cur_targets
        origins = self.env.scene.env_origins

        def pose_values(pos: torch.Tensor, quat: torch.Tensor, env_id: int) -> list[float]:
            return [*pos[env_id].tolist(), *quat[env_id].tolist()]

        for env_id in self.trace_envs:
            if bool(self.finished[env_id].item()):
                continue
            self._trace_rows[env_id].append(
                [
                    float(self.elapsed[env_id].item()),
                    float(phase[env_id].item()),
                    float(self.attempt_count[env_id].item()),
                    float(self.latest_depth[env_id].item()),
                    float(self.latest_lateral[env_id].item()),
                    *self.latest_object_pos[env_id].tolist(),
                    float(self.latest_normal[env_id, 0].item()),
                    float(self.latest_tangent[env_id, 0].item()),
                    float(self.latest_normal[env_id, 1].item()),
                    float(self.latest_tangent[env_id, 1].item()),
                    float(self.latest_normal[env_id, 2].item()),
                    float(self.latest_tangent[env_id, 2].item()),
                    float(self.latest_goal_translation_error[env_id].item()),
                    float(self.latest_goal_rotation_error[env_id].item()),
                    float(self.latest_keypoint_error[env_id].item()),
                    float(self.latest_palm_object_translation_change[env_id].item()),
                    float(self.latest_palm_object_rotation_change[env_id].item()),
                    float(torch.linalg.vector_norm(self._latest_policy_action[env_id, :7]).item()),
                    float(torch.linalg.vector_norm(self._latest_policy_action[env_id, 7:]).item()),
                    float(self.screw_axial_feed[env_id].item()),
                    float(self.screw_forward_rotation[env_id].item()),
                    float(self.screw_reverse_rotation[env_id].item()),
                    *pose_values(self.latest_palm_pos + origins, self.latest_palm_quat, env_id),
                    *pose_values(self.latest_object_pos + origins, self.latest_object_quat, env_id),
                    *pose_values(
                        self.latest_palm_object_pos, self.latest_palm_object_quat, env_id
                    ),
                    *pose_values(self.latest_fixture_pos + origins, self.latest_fixture_quat, env_id),
                    *pose_values(
                        self.latest_fixture_object_pos,
                        self.latest_fixture_object_quat,
                        env_id,
                    ),
                    *pose_values(self.latest_goal_pos + origins, self.latest_goal_quat, env_id),
                    *joint_pos[env_id].tolist(),
                    *joint_vel[env_id].tolist(),
                    *joint_target[env_id].tolist(),
                    *self._latest_policy_action[env_id].tolist(),
                ]
            )
            if len(self._trace_rows[env_id][-1]) != len(self.trace_columns):
                raise RuntimeError("diagnostic trace row/column mismatch")

    def policy_step(
        self, assembly_success: torch.Tensor, goal_count: torch.Tensor | None = None
    ) -> None:
        if goal_count is not None:
            observed = goal_count.to(self.device).long()
            advanced = observed > self.goal_count
            self.goal_count.copy_(torch.maximum(self.goal_count, observed))
            self.goal_reached_time = torch.where(advanced, self.elapsed, self.goal_reached_time)
            self._emit(advanced, "goal_reached", goal_count=observed)
        newly_seated = assembly_success.to(self.device).bool() & ~self.assembly_success_seen
        self.assembly_success_seen |= newly_seated
        self.seat_time = torch.where(newly_seated, self.elapsed, self.seat_time)
        self._emit(newly_seated, "attempt_end", outcome="seated", depth_m=self.latest_depth)
        self._emit(newly_seated, "assembly_seated", depth_m=self.latest_depth)
        self._emit(newly_seated, "assembly_completed", depth_m=self.latest_depth)

    def _metric_dict(self, env_id: int) -> dict:
        result = {}
        for phase_idx, phase in enumerate(PHASES):
            result[phase] = {}
            for category_idx, category in enumerate(CATEGORIES):
                result[phase][category] = {
                    "peak_normal_N": float(self.normal_peak[env_id, phase_idx, category_idx].item()),
                    "peak_net_pair_force_N": float(
                        self.net_force_peak[env_id, phase_idx, category_idx].item()
                    ),
                    "peak_time_s": self._finite_or_none(
                        self.normal_peak_time[:, phase_idx, category_idx], env_id
                    ),
                    "peak_point_w_m": self.normal_peak_point[
                        env_id, phase_idx, category_idx
                    ].tolist(),
                    "peak_tangential_N": float(self.tangent_peak[env_id, phase_idx, category_idx].item()),
                    "normal_impulse_Ns": float(self.normal_impulse[env_id, phase_idx, category_idx].item()),
                    "tangential_impulse_Ns": float(self.tangent_impulse[env_id, phase_idx, category_idx].item()),
                    "contact_time_s": float(self.contact_time[env_id, phase_idx, category_idx].item()),
                    "high_force_time_s": float(self.high_force_time[env_id, phase_idx, category_idx].item()),
                    "max_high_force_run_s": float(
                        self.max_high_force_run[env_id, phase_idx, category_idx].item()
                    ),
                    "max_contact_episode_impulse_Ns": float(
                        self.max_contact_episode_impulse[env_id, phase_idx, category_idx].item()
                    ),
                    "unexpected_environment_contact": bool(
                        category == "hand_fixture"
                        and self.contact_time[env_id, phase_idx, category_idx].item() > 0.0
                    ),
                }
        return result

    @staticmethod
    def _finite_or_none(value: torch.Tensor, env_id: int) -> float | None:
        item = float(value[env_id].item())
        return item if np.isfinite(item) else None

    def finalize(
        self,
        env_ids: torch.Tensor,
        *,
        completed: torch.Tensor,
        assembly_success: torch.Tensor,
        task_success: torch.Tensor,
        goal_ratio: torch.Tensor,
        policy_steps: torch.Tensor,
    ) -> list[dict]:
        env_ids = env_ids.to(self.device).long()
        finalize_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        finalize_mask[env_ids] = True
        incomplete_attempt = (
            finalize_mask
            & self.insertion_started
            & ~self.assembly_success_seen
            & ~self.retry_pending
        )
        self._emit(incomplete_attempt, "attempt_end", outcome="rollout_ended")
        self._finish_outward_runs(finalize_mask & (self.outward_run > 0.0))
        for phase_idx, phase in enumerate(PHASES):
            for category_idx, category in enumerate(CATEGORIES):
                active = finalize_mask & self.contact_active_prev[:, phase_idx, category_idx]
                self._emit(
                    active,
                    "contact_end",
                    phase=phase,
                    category=category,
                    outcome="rollout_ended",
                    episode_impulse_Ns=self.contact_episode_impulse[:, phase_idx, category_idx],
                    episode_duration_s=self.contact_episode_time[:, phase_idx, category_idx],
                )
                self.contact_active_prev[:, phase_idx, category_idx] &= ~finalize_mask
        self.max_contact_episode_impulse = torch.maximum(
            self.max_contact_episode_impulse, self.contact_episode_impulse
        )

        rows = []
        for local_idx, env_id in enumerate(env_ids.tolist()):
            assembly_ok = bool(assembly_success[local_idx].item())
            task_ok = bool(task_success[local_idx].item())
            raw_attempt_count = int(self.attempt_count[env_id].item())
            first_contact = self._finite_or_none(self.first_fixture_contact_time, env_id)
            seat_time = self._finite_or_none(self.seat_time, env_id)
            first_contact_to_seat = (
                seat_time - first_contact
                if seat_time is not None and first_contact is not None
                else None
            )
            insertion_start = self._finite_or_none(self.insertion_start_time, env_id)
            grasp_time = self._finite_or_none(self.grasp_time, env_id)
            lift_time = self._finite_or_none(self.lift_time, env_id)
            elapsed = float(self.elapsed[env_id].item())
            recovery_duration = float(self.recovery_duration[env_id].item())
            if bool(self.recovery_active[env_id].item()):
                recovery_start = self._finite_or_none(self.recovery_start_time, env_id)
                if recovery_start is not None:
                    recovery_duration += elapsed - recovery_start
            drop_count = int(self.drop_count[env_id].item())
            regrasp_count = int(self.regrasp_count[env_id].item())
            raw_retry_count = int(self.retry_count[env_id].item())
            attempt_count = None if self.is_screwing else raw_attempt_count
            retry_count = None if self.is_screwing else raw_retry_count
            restart_count = int(self.task_restart_count[env_id].item())
            recovery_count = int(self.recovery_count[env_id].item()) + restart_count
            recovery_reasons = []
            if int(self.recovery_count[env_id].item()) - drop_count > 0:
                recovery_reasons.append("insertion_backtrack")
            if drop_count:
                recovery_reasons.append("object_drop")
            if restart_count:
                recovery_reasons.append("task_restart")
            insertion_verified = attempt_count is not None and attempt_count > 0
            first_attempt_status = (
                bool(assembly_ok and attempt_count == 1 and retry_count == 0)
                if insertion_verified else None
            )
            recovery_free_success = (
                False if self.is_screwing and (recovery_count or drop_count or regrasp_count)
                else (None if self.is_screwing else bool(task_ok and recovery_count == 0))
            )
            first_pass_status = (
                bool(task_ok and first_attempt_status and recovery_free_success)
                if insertion_verified else None
            )
            record = {
                "problem": self.env._pih_problem_name,
                "env_id": env_id,
                "episode": int(self._episode_index[env_id].item()),
                "completed": bool(completed[local_idx].item()),
                "policy_steps": int(policy_steps[local_idx].item()),
                "duration_s": elapsed,
                "goal_ratio": float(goal_ratio[local_idx].item()),
                "assembly_success": assembly_ok,
                "task_success": task_ok,
                "stable_grasp_observed": bool(self.grasp_ever[env_id].item()),
                "object_lifted_observed": bool(self.object_lifted_seen[env_id].item()),
                "attempt_count": raw_attempt_count,
                "insertion_attempt_count": attempt_count,
                "insertion_retry_count": retry_count,
                "first_attempt_success": bool(assembly_ok and attempt_count == 1),
                "first_attempt_insertion_success": first_attempt_status,
                "first_pass_success": first_pass_status,
                "first_pass_classification": (
                    "unverified_screwing_engagement"
                    if self.is_screwing else ("verified" if insertion_verified else "unknown_no_attempt")
                ),
                "drop_count": drop_count,
                "regrasp_count": regrasp_count,
                "grasp_contact_restore_count": int(self.grasp_restore_count[env_id].item()),
                "task_restart_count": restart_count,
                "task_restart_observable": False,
                "task_recovery_count": recovery_count,
                "recovery_event_count": recovery_count,
                "recovery_reasons": recovery_reasons,
                "success_without_drop_regrasp_recovery": bool(
                    task_ok
                    and drop_count == 0
                    and regrasp_count == 0
                    and recovery_count == 0
                ),
                "recovery_free_success": recovery_free_success,
                "local_adjustment_count": int(self.adjustment_count[env_id].item()),
                "backoff_count": int(self.backoff_count[env_id].item()),
                "reverse_distance_raw_m": float(self.reverse_distance_raw[env_id].item()),
                "reverse_distance_effective_m": float(
                    self.reverse_distance_effective[env_id].item()
                ),
                "contact_active_search_s": float(self.contact_search_time[env_id].item()),
                "task_completion_time_s": elapsed if task_ok else None,
                "failure_observed_duration_s": None if task_ok else elapsed,
                "grasp_acquisition_time_s": grasp_time,
                "object_lift_time_s": lift_time,
                "first_contact_time_s": first_contact,
                "first_contact_to_seat_s": first_contact_to_seat,
                "contact_to_success_duration_s": first_contact_to_seat,
                "transport_approach_s": (
                    insertion_start - grasp_time
                    if insertion_start is not None
                    and grasp_time is not None
                    else None
                ),
                "transport_duration_s": (
                    insertion_start - grasp_time
                    if insertion_start is not None and grasp_time is not None else None
                ),
                "insertion_duration_s": (
                    seat_time - insertion_start
                    if insertion_start is not None and seat_time is not None
                    else None
                ),
                "insertion_observed_duration_s": (
                    (seat_time if seat_time is not None else elapsed) - insertion_start
                    if insertion_start is not None else None
                ),
                "recovery_duration_s": recovery_duration,
                "recovery_duration_censored": bool(self.recovery_active[env_id].item()),
                "insertion_backtrack_distance_m": float(
                    self.reverse_distance_effective[env_id].item()
                ),
                "screw_metrics": {
                    "classification_verified": False if self.is_screwing else None,
                    "axial_feed_depth_m": float(self.screw_axial_feed[env_id].item())
                    if self.is_screwing else None,
                    "forward_rotation_rad": float(
                        self.screw_forward_rotation[env_id].item()
                    ) if self.is_screwing else None,
                    "reverse_rotation_rad": float(
                        self.screw_reverse_rotation[env_id].item()
                    ) if self.is_screwing else None,
                    "raw_backtrack_candidate_count": int(
                        self.backtrack_candidate_count[env_id].item()
                    ) if self.is_screwing else None,
                    "raw_reengagement_candidate_count": raw_retry_count
                    if self.is_screwing else None,
                    "exit_count": None,
                    "reengagement_count": None,
                },
                "contact_metrics": self._metric_dict(env_id),
            }
            record["trace_saved"] = self._save_trace(env_id)
            rows.append(record)

        with (self.output_dir / "episodes.jsonl").open("a") as file:
            for row in rows:
                file.write(json.dumps(row, sort_keys=True) + "\n")
        with (self.output_dir / "events.jsonl").open("a") as file:
            for env_id in env_ids.tolist():
                for event in self._events[env_id]:
                    file.write(json.dumps(event, sort_keys=True) + "\n")
        self.finished[env_ids] = True
        return rows

    def _save_trace(self, env_id: int) -> bool:
        rows = self._trace_rows.get(env_id)
        if not rows:
            return False
        array = np.asarray(rows, dtype=np.float32)
        if self.trace_bytes_written + array.nbytes > self.max_run_bytes:
            self._events[env_id].append(
                {
                    "env_id": env_id,
                    "episode": int(self._episode_index[env_id].item()),
                    "time_s": float(self.elapsed[env_id].item()),
                    "event": "storage_budget_exceeded",
                }
            )
            return False
        path = self.output_dir / "traces" / (
            f"env{env_id:04d}_ep{int(self._episode_index[env_id].item()):04d}.npz"
        )
        np.savez_compressed(path, trace=array, columns=np.asarray(self.trace_columns))
        self.trace_bytes_written += path.stat().st_size
        return True


__all__ = ["AssemblyDiagnostics", "DetectorConfig", "parse_trace_envs"]
