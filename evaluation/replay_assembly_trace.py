"""Replay a recorded assembly rollout in Viser without starting a simulator."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_trace(path):
    with np.load(path, allow_pickle=False) as data:
        trace = data['trace'].copy()
        columns = data['columns'].tolist()
    if trace.ndim != 2 or trace.shape[1] != len(columns) or not len(trace):
        raise ValueError('Expected a nonempty trace matrix matching its columns')
    if len(columns) != len(set(columns)):
        raise ValueError('Trace column names must be unique')
    index = {name: i for i, name in enumerate(columns)}
    times = trace[:, index['time_s']]
    if not np.isfinite(trace).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Trace must be finite with strictly increasing times')
    return trace, index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--problem', default='tight_insertion')
    parser.add_argument('--port', type=int, default=8046)
    args = parser.parse_args()
    trace, index = load_trace(args.trace)
    times = trace[:, index['time_s']]

    import viser
    from viser.extras import ViserUrdf
    from evaluation._viser_demo import _asset_abs, _load_mesh_for_viz, _resolve_problem_assets

    server = viser.ViserServer(host='127.0.0.1', port=args.port)
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/grid', width=2., height=2.)
    server.scene.add_frame('/robot', position=(0., .8, 0.), show_axes=False)
    robot = ViserUrdf(server, ROOT / 'assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf', root_node_name='/robot')
    # Static robot root and table match the original no-DR evaluation scene.
    server.scene.add_box('/table', dimensions=(.475, .4, .3), position=(0., 0., .38), color=(180, 130, 70))
    object_path, fixture_path = _resolve_problem_assets(args.problem)
    object_mesh = _load_mesh_for_viz(_asset_abs(object_path))
    fixture_mesh = _load_mesh_for_viz(_asset_abs(fixture_path))
    frames = {}
    for name, mesh, color, opacity in [
        ('object', object_mesh, (204, 40, 40), 1.),
        ('fixture', fixture_mesh, (120, 120, 120), 1.),
        ('goal', object_mesh, (40, 220, 80), .25),
    ]:
        frames[name] = server.scene.add_frame('/' + name, show_axes=False)
        server.scene.add_mesh_simple('/' + name + '/mesh', vertices=np.asarray(mesh.vertices, dtype=np.float32), faces=np.asarray(mesh.faces, dtype=np.uint32), color=color, opacity=opacity)
    joint_columns = [index['joint_pos_rad_' + name] for name in robot.get_actuated_joint_names()]

    metadata_path = args.trace.parent.parent / 'metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    server.gui.add_markdown('# Recorded rollout replay\nRobot, object and goal poses come from the saved trace. No policy inference or physics runs here.\n\n' + f'Problem: `{args.problem}` · seed: `{metadata.get("seed", "see metadata")}`\n\nRed: object · green: target · gray: fixture.')
    play = server.gui.add_checkbox('Play', initial_value=False)
    loop = server.gui.add_checkbox('Loop', initial_value=False)
    speed = server.gui.add_slider('Speed', min=.25, max=2., step=.25, initial_value=1.)
    slider = server.gui.add_slider('Frame', min=0, max=len(trace)-1, step=1, initial_value=0)
    restart = server.gui.add_button('Restart')
    status = server.gui.add_markdown('')

    @restart.on_click
    def _(_event):
        slider.value = 0
        play.value = True

    @server.on_client_connect
    def _(client):
        client.camera.position = (.95, 1.2, 1.05)
        client.camera.look_at = (0., .05, .6)

    def display(frame):
        row = trace[frame]
        with server.atomic():
            robot.update_cfg(row[joint_columns])
            for name, handle in frames.items():
                handle.position = tuple(float(row[index[f'T_world_{name}_{axis}_m']]) for axis in 'xyz')
                handle.wxyz = tuple(float(row[index[f'T_world_{name}_q{axis}']]) for axis in 'wxyz')
        status.content = f'**Recorded time:** {times[frame]:.2f} s / {times[-1]:.2f} s · frame {frame+1}/{len(trace)}\n\nPlayback stops at the last recorded sample; terminal flags are stored separately in episodes.jsonl.'

    display(0)
    print(f'Recorded replay ready: http://localhost:{server.get_port()} ({len(trace)} frames)', flush=True)
    last_frame = 0
    deadline = time.monotonic()
    try:
        while True:
            frame = int(slider.value)
            now = time.monotonic()
            if frame != last_frame:
                display(frame)
                last_frame = frame
                deadline = now
            interval = float(times[min(frame+1, len(times)-1)]-times[frame]) / float(speed.value)
            if play.value and now-deadline >= interval:
                if frame < len(trace)-1:
                    slider.value = frame+1
                elif loop.value:
                    slider.value = 0
                else:
                    play.value = False
                deadline = now
            elif not play.value:
                deadline = now
            time.sleep(.005)
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__':
    main()
