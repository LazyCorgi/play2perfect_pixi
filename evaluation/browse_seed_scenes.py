"""Inspect exported evaluation starts and recorded rollouts, without a simulator."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def trace_pose(row, index, name, origin):
    """Convert a logged world pose to the selected environment's local frame."""
    p = np.array([row[index[f'T_world_{name}_{a}_m']] for a in 'xyz']) - origin
    q = [float(row[index[f'T_world_{name}_q{a}']]) for a in 'wxyz']
    return p.tolist() + q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=ROOT/'evaluation_runs/seed_scene_review')
    parser.add_argument('--port', type=int, default=8047)
    args = parser.parse_args()

    import viser
    from viser.extras import ViserUrdf
    from evaluation._viser_demo import _load_mesh_for_viz
    from evaluation.replay_assembly_trace import load_trace

    server = viser.ViserServer(host='127.0.0.1', port=args.port)
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/grid', width=2., height=2.)
    robot_frame = server.scene.add_frame('/robot', show_axes=False)
    robot = ViserUrdf(server, ROOT/'assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf', root_node_name='/robot')
    table = server.scene.add_box('/table', dimensions=(.475, .4, .3), color=(180, 130, 70))
    frames = {name: server.scene.add_frame('/'+name, show_axes=False) for name in ['object', 'fixture', 'goal']}
    meshes = []
    server.gui.add_markdown('# Seed scene review\nEach seed generates a batch of starts. Inspect **Env ID** separately. Red: object; green: target; gray: fixture.\n\nInitial frame is the exact exported reset state. Playback uses recorded poses; no physics or policy inference runs here.')
    batch = server.gui.add_dropdown('Batch', options=['Waiting for snapshots'])
    env_id = server.gui.add_slider('Env ID', min=0, max=31, step=1, initial_value=0)
    previous = server.gui.add_button('Previous start')
    following = server.gui.add_button('Next start')
    frame = server.gui.add_slider('Frame (0 = reset)', min=0, max=1, step=1, initial_value=0)
    play = server.gui.add_checkbox('Play rollout', initial_value=False)
    speed = server.gui.add_slider('Speed', min=.25, max=2., step=.25, initial_value=1.)
    initial = server.gui.add_button('Back to initial state')
    show_goal = server.gui.add_checkbox('Show target', initial_value=True)
    status = server.gui.add_markdown('Waiting for exported states...')
    details = server.gui.add_markdown('')

    @previous.on_click
    def _(_event):
        env_id.value = max(0, int(env_id.value)-1)

    @following.on_click
    def _(_event):
        env_id.value = min(int(env_id.max), int(env_id.value)+1)

    @initial.on_click
    def _(_event):
        play.value = False
        frame.value = 0

    @server.on_client_connect
    def _(client):
        client.camera.position = (.95, 1.2, 1.05)
        client.camera.look_at = (0., .05, .6)

    def set_pose(handle, pose):
        handle.position = tuple(pose[:3])
        handle.wxyz = tuple(pose[3:7])

    paths = {}
    current_key = None
    data = None
    trace = None
    columns = None
    outcomes = {}
    geometry = {}
    initial_audit = {}
    joint_columns = []
    last_frame = -1
    deadline = time.monotonic()
    refreshed = 0.
    print(f'Seed browser ready: http://localhost:{server.get_port()}', flush=True)
    try:
        while True:
            now = time.monotonic()
            if now-refreshed > 2.:
                refreshed = now
                for p in sorted(args.catalog.rglob('initial_scenes.json')):
                    try:
                        d = json.loads(p.read_text())
                    except json.JSONDecodeError:
                        continue
                    paths[f'{d["problem"]} | seed {d["seed"]} | N={d["num_envs"]}'] = p
                if paths and tuple(batch.options) != tuple(paths):
                    selected = batch.value
                    batch.options = tuple(paths)
                    batch.value = selected if selected in paths else next(iter(paths))

            key = (batch.value, int(env_id.value))
            if batch.value not in paths:
                time.sleep(.05)
                continue
            trace_path = paths[batch.value].parent/'traces'/f'env{key[1]:04d}_ep0000.npz'
            if key != current_key or (trace is None and trace_path.exists()):
                current_key = key
                play.value = False
                frame.value = 0
                data = json.loads(paths[batch.value].read_text())
                env_id.max = data['num_envs']-1
                for mesh in meshes:
                    mesh.remove()
                meshes.clear()
                for name, asset_key, color, opacity in [('object', 'object_urdf', (204,40,40),1.), ('goal','object_urdf',(40,220,80),.25), ('fixture','fixture_urdf',(120,120,120),1.)]:
                    mesh = _load_mesh_for_viz(Path(data[asset_key]))
                    meshes.append(server.scene.add_mesh_simple('/'+name+'/mesh', vertices=np.asarray(mesh.vertices,dtype=np.float32), faces=np.asarray(mesh.faces,dtype=np.uint32), color=color, opacity=opacity))
                trace, columns = load_trace(trace_path) if trace_path.exists() else (None, None)
                frame.max = len(trace) if trace is not None else 1
                frame.disabled = trace is None
                play.disabled = trace is None
                if trace is not None:
                    joint_columns = [columns['joint_pos_rad_'+name] for name in robot.get_actuated_joint_names()]
                outcome_file = paths[batch.value].parent/'episodes.jsonl'
                outcomes = {r['env_id']: r for r in map(json.loads, outcome_file.read_text().splitlines())} if outcome_file.exists() else {}
                audit_file = paths[batch.value].parent/'geometry_audit.json'
                geometry = {r['env_id']: r['seated_10_steps'] for r in json.loads(audit_file.read_text())['per_env']} if audit_file.exists() else {}
                starts_audit = args.catalog/'initial_geometry_audit.json'
                initial_audit = {}
                if starts_audit.exists():
                    for audit in json.loads(starts_audit.read_text()):
                        if audit['problem'] == data['problem'] and audit['seed'] == data['seed']:
                            initial_audit = {r['env_id']: r for r in audit['per_env']}
                last_frame = -1

            i, f = int(env_id.value), int(frame.value)
            frames['goal'].visible = show_goal.value
            if f != last_frame:
                with server.atomic():
                    set_pose(robot_frame, data['robot_pose'][i])
                    set_pose(table, data['table_pose'][i])
                    if f == 0 or trace is None:
                        joints = dict(zip(data['joint_names'], data['joint_pos'][i]))
                        robot.update_cfg(np.array([joints[n] for n in robot.get_actuated_joint_names()]))
                        for name, handle in frames.items():
                            set_pose(handle, data[name+'_pose'][i])
                        recorded_time = 0.
                    else:
                        row = trace[f-1]
                        robot.update_cfg(row[joint_columns])
                        for name, handle in frames.items():
                            set_pose(handle, trace_pose(row, columns, name, np.array(data['world_origins'][i])))
                        recorded_time = float(row[columns['time_s']])
                outcome = outcomes.get(i)
                outcome_text = ('SUCCESS' if outcome['task_success'] else 'FAILURE') if outcome else 'not available yet'
                strict_text = ('PASS' if geometry[i] else 'NOT MET') if i in geometry else 'not measured'
                status.content = f'**{data["problem"]} · seed {data["seed"]} · env {i}**\n\nRecorded time: {recorded_time:.3f} s. Environment task result: **{outcome_text}**.\n\nStrict geometric seating over this rollout (1 mm, 10 samples): **{strict_text}**.\n\n'+('Exact initial state; object may still be airborne.' if f == 0 else 'Recorded rollout sample.')
                obj = np.array(data['object_pose'][i]); fixture = np.array(data['fixture_pose'][i])
                details.content = f'**Checkpoint:** `{data.get("checkpoint", "not recorded")}`\n\n**Policy coefficient ID:** {data.get("policy_coefficient_id", "not recorded")}\n\n**Goal mode:** {data.get("goal_mode", "not recorded")}\n\n**Initial object xyz (m):** {obj[:3].round(4).tolist()}\n\n**Initial object wxyz:** {obj[3:].round(4).tolist()}\n\n**Fixture xyz (m):** {fixture[:3].round(4).tolist()}\n\n**Horizontal object–fixture distance:** {np.linalg.norm(obj[:2]-fixture[:2])*1000:.1f} mm\n\nThis batch uses seed {data["seed"]} and N={data["num_envs"]}. Changing the batch size can change the sampled starts even with the same seed.'
                if i in initial_audit:
                    a = initial_audit[i]
                    clearance = a['lowest_collider_above_table_m']*1000
                    details.content += f'\n\n**Initial collider clearance above table:** {clearance:.1f} mm.'
                    if a['initial_table_overlap_m'] is not None:
                        details.content += f'\n\n**Initial box overlap:** table {a["initial_table_overlap_m"]*1000:.2f} mm; fixture {a["initial_fixture_overlap_m"]*1000:.2f} mm. Overlap is an inspection flag, not proof of an impossible task.'
                    else:
                        details.content += '\n\nBeam–fixture overlap was not audited.'
                last_frame = f
                deadline = now
            if play.value and trace is not None:
                t = 0. if f == 0 else float(trace[f-1, columns['time_s']])
                next_t = float(trace[min(f,len(trace)-1), columns['time_s']])
                if now-deadline >= max(.001,(next_t-t)/float(speed.value)):
                    if f < len(trace):
                        frame.value = f+1
                    else:
                        play.value = False
            elif not play.value:
                deadline = now
            time.sleep(.01)
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__':
    main()
