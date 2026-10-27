"""Save actual reset states for visual inspection, without advancing physics."""

import json
from pathlib import Path


def save_initial_scenes(env, path, *, seed, checkpoint, policy_id):
    def local_pose(asset):
        pose = asset.data.root_state_w[:, :7].clone()
        pose[:, :3] -= env.scene.env_origins
        return pose.detach().cpu().tolist()

    origins = env.scene.env_origins
    fixture_pos = env.hole_pos.detach().cpu().tolist()  # already env-local
    fixture_q = env.hole_quat_wxyz.detach().cpu().tolist()
    data = {
        'problem': env.cfg.precise_assembly.problem,
        'goal_mode': env.cfg.precise_assembly.goal_mode,
        'seed': seed, 'num_envs': env.num_envs,
        'checkpoint': str(checkpoint), 'policy_coefficient_id': policy_id,
        'coordinate_frame': 'environment-local; all quaternions wxyz',
        'snapshot': 'after player.env_reset, before the first policy action',
        'robot_urdf': env.cfg.assets.robot_urdf,
        'object_urdf': str(env._pih_object_urdf_abs),
        'fixture_urdf': str(env._pih_receptive_urdf_abs),
        'joint_names': list(env.robot.data.joint_names),
        'joint_pos': env.robot.data.joint_pos.detach().cpu().tolist(),
        'robot_pose': local_pose(env.robot),
        'table_pose': local_pose(env.table),
        'table_dimensions': [.475, .4, .3],
        'object_pose': local_pose(env.object),
        'goal_pose': local_pose(env.goal_viz),
        'fixture_pose': [p+q for p, q in zip(fixture_pos, fixture_q)],
        'world_origins': origins.detach().cpu().tolist(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')
    return data
