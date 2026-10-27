"""Visualize recorded geometric policy inputs; these are not camera images."""

import argparse
import json
from pathlib import Path

import numpy as np


def field_slices(fields, width):
    slices, start = {}, 0
    for name, size in fields:
        if name in slices or size <= 0:
            raise ValueError('Observation field names must be unique and sizes positive')
        slices[name] = slice(start, start+size)
        start += size
    if start != width:
        raise ValueError(f'Layout describes {start} values, input contains {width}')
    return slices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--success-env', type=int, default=22)
    parser.add_argument('--failure-env', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=Path('evaluation_runs/observation_review'))
    args = parser.parse_args()
    with np.load(args.recording, allow_pickle=False) as f:
        observations = f['observations'].copy()
        valid = f['first_episode_valid'].copy()
        env_ids = f['env_ids'].tolist()
        metadata = json.loads(str(f['metadata']))
    fields = field_slices(metadata['fields'], observations.shape[-1])
    episodes = {r['env_id']:r for r in map(json.loads, (args.recording.parent/'episodes.jsonl').read_text().splitlines())}
    cases = []
    for env_id, expected in [(args.success_env, True), (args.failure_env, False)]:
        if bool(episodes[env_id]['task_success']) != expected:
            raise ValueError('Requested success/failure label does not match recorded outcome')
        j = env_ids.index(env_id)
        inputs = observations[valid[:, j], j]
        obj = inputs[:, fields['keypoints_rel_palm']].reshape(-1,4,3)
        relative_goal = inputs[:, fields['keypoints_rel_goal']].reshape(-1,4,3)
        goal = obj-relative_goal
        fingertips = inputs[:, fields['fingertip_pos_rel_palm']].reshape(-1,5,3)
        times = np.arange(len(inputs))*metadata['control_dt_s']
        cases.append(dict(env_id=env_id, success=expected, obj=obj, goal=goal, fingertips=fingertips,
                          times=times, goal_error=np.linalg.norm(relative_goal,axis=2).max(axis=1),
                          palm_distance=np.linalg.norm(obj.mean(axis=1),axis=1),
                          goal_noise=metadata['goal_pos_obs_noise'][j]))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from itertools import combinations

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bound = 1.1*max(.1, *(float(np.abs(c[key]).max()) for c in cases for key in ['obj','goal','fingertips']))
    fig = plt.figure(figsize=(14,7), layout='constrained')
    for row, case in enumerate(cases):
        frames = [int(np.argmin(abs(case['times']-t))) for t in [0., 2., 5., case['times'][-1]]]
        for col, frame in enumerate(frames):
            ax = fig.add_subplot(2,4,row*4+col+1,projection='3d')
            for points, color, label in [(case['obj'][frame], 'firebrick', 'Object landmarks'), (case['goal'][frame], 'green', 'Observed goal landmarks')]:
                ax.scatter(*points.T, color=color, s=16, label=label)
                for a,b in combinations(range(4),2):
                    ax.plot(*points[[a,b]].T, color=color,alpha=.4,lw=.7)
            ax.scatter(*case['fingertips'][frame].T,color='darkorange',s=12,label='Fingertips')
            ax.scatter([0],[0],[0],color='royalblue',s=25,label='Palm origin')
            ax.set(xlabel='x from palm (m)',ylabel='y from palm (m)',zlabel='z from palm (m)',
                   title=f'{"SUCCESS" if case["success"] else "FAILURE"} env {case["env_id"]} | t={case["times"][frame]:.2f}s')
            ax.set_xlim(-bound,bound);ax.set_ylim(-bound,bound);ax.set_zlim(-bound,bound)
            ax.set_box_aspect((1,1,1))
            ax.view_init(elev=22,azim=-55)
            if row==0 and col==0:ax.legend(fontsize=6,loc='upper left')
    fig.suptitle(f'{metadata["problem"]}, seed={metadata["seed"]}\nActual recorded geometric input fields (subset); NOT RGB/depth images. Inputs: 140 state values + SAPG ID.',fontsize=12)
    fig.savefig(args.output_dir/'policy_input_snapshots.png',dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for case in cases:
        label=f'{"SUCCESS" if case["success"] else "FAILURE"} env {case["env_id"]}'
        axes[0].plot(case['times'],case['goal_error']*1000,label=label)
        axes[1].plot(case['times'],case['palm_distance']*1000,label=label)
    axes[0].set(title='Observed descriptor landmark error',ylabel='Maximum paired distance (mm)',xlabel='Time before action (s)')
    axes[1].set(title='Object-center distance from palm',ylabel='Distance (mm)',xlabel='Time before action (s)')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    fig.savefig(args.output_dir/'policy_input_errors.png',dpi=180)
    summary=[dict(env_id=c['env_id'],task_success=c['success'],recorded_input_steps=len(c['times']),
                  minimum_observed_landmark_error_m=float(c['goal_error'].min()),
                  final_observed_landmark_error_m=float(c['goal_error'][-1]),
                  initial_goal_position_observation_noise_m=c['goal_noise']) for c in cases]
    (args.output_dir/'comparison.json').write_text(json.dumps({'recording':str(args.recording),'metadata':metadata,'cases':summary},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
