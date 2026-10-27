"""Aggregate eval logs without excluding early failures or unfinished trials."""

import argparse
import json
import math
from pathlib import Path


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - width), min(1.0, center + width)]


def summarize(rows):
    total = sum(r['num_envs'] for r in rows)
    # Explicit counts are required; never treat the valid-only success rate as
    # if it were measured over all starts.
    successes = sum(r['task_success_count_all_envs'] for r in rows)
    return {
        'trials_all_starts': total,
        'completed': sum(r['n_completed_first_episodes'] for r in rows),
        'task_successes_with_retraction': successes,
        'task_success_rate': successes / total if total else None,
        'wilson_95_interval': wilson_interval(successes, total),
        'observed_above_40_percent': total > 0 and successes / total > .4,
        'lower_95_bound_above_40_percent': total > 0 and wilson_interval(successes, total)[0] > .4,
        'geometry_seated_and_task_successes': sum(r['geometry_seated_and_task_success_count_all_envs'] for r in rows) if all('geometry_seated_and_task_success_count_all_envs' in r for r in rows) else None,
        'seeds': [r['seed'] for r in rows],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows = []
    for path in args.logs:
        found = [json.loads(line.removeprefix('RESULT_JSON ')) for line in path.read_text().splitlines() if line.startswith('RESULT_JSON ')]
        if not found:
            raise ValueError(f'No evaluation result in {path}')
        rows.extend(found)
    if len({(r['problem'], r.get('goal_mode'), r['checkpoint'], r.get('policy_coefficient_id', 50), r.get('episode_length_s', 10.)) for r in rows}) != 1:
        raise ValueError('Aggregate only the same task, goal mode and checkpoint')
    report = {'summary': summarize(rows), 'runs': rows}
    encoded = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
