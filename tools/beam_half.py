"""Verify the beam-half artifact and run the published demo/evaluation/course."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def verify(root=ROOT):
    manifest = json.loads((root / 'releases/beam_half.json').read_text())
    for name, expected in manifest['files'].items():
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Missing or incorrect artifact: {path}. Extract beam-half-artifacts.tar.gz at repository root.')
    return manifest

def evaluate(checkpoint, output, seeds):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in seeds:
        command = [sys.executable, '-u', 'evaluation/eval_offline.py', '--problem',
                   'beam_assembly_step1_scale_050', '--checkpoint-path', str(checkpoint),
                   '--num-envs', '32', '--steps', '3000', '--seed', str(seed),
                   '--min-episode-steps', '0', '--policy-coefficient-id', '50',
                   '--goal-mode', 'preInsertAndFinal', '--episode-length-s', '10']
        results = []
        with (output / f'seed{seed}.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1,
                                     env=dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES'))
            try:
                for line in child.stdout:
                    print(line, end='', flush=True); log.write(line); log.flush()
                    if line.startswith('RESULT_JSON '):
                        results.append(json.loads(line[12:]))
                code = child.wait()
            finally:
                child.stdout.close()
                if child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
        if code or len(results) != 1:
            raise RuntimeError(f'Evaluation failed; see {output}')
        row = results[0]
        if row.get('n_unfinished', 1) or row.get('n_completed_first_episodes') != 32:
            raise RuntimeError(f'Incomplete first episodes; see {output}')
        rows.append(row)
        successes = sum(r['task_success_count_all_envs'] for r in rows)
        trials = sum(r['num_envs'] for r in rows)
        (output / 'results.json').write_text(json.dumps(dict(
            status='completed' if len(rows) == len(seeds) else 'running',
            successes=successes, trials=trials, success_rate=successes/trials,
            checkpoint_sha256=sha256(checkpoint), batches=rows), indent=2)+'\n')
    print(f'Full task success: {successes}/{trials} = {successes/trials:.2%}; results: {output}')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['verify', 'demo', 'eval', 'smoke', 'train', 'train-from060'])
    parser.add_argument('--port', type=int, default=8050)
    parser.add_argument('--run-id', default=None)
    args = parser.parse_args()
    if args.mode != 'train':
        verify()
    artifact = ROOT / 'pretrained_assembly/beam_half'
    checkpoint = artifact / 'model.pth'
    if args.mode == 'verify':
        print('Beam-half artifact checksums: OK'); return
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    if args.mode == 'demo':
        os.execv(sys.executable, [sys.executable, '-u', str(ROOT/'evaluation/eval_isaacsim.py'),
            '--checkpoint-path', str(checkpoint), '--problem', 'beam_assembly_step1_scale_050',
            '--port', str(args.port), '--deterministic', '--autoplay',
            '--override', 'env.episode_length_s', '10'])
    if args.run_id is None:
        from datetime import datetime
        args.run_id = 'beam_half_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    if Path(args.run_id).name != args.run_id or args.run_id in ('.', '..'):
        parser.error('--run-id must be a directory name')
    if args.mode == 'train':
        official = ROOT / 'pretrained_assembly/beam_assembly_step1/model.pth'
        if not official.is_file():
            subprocess.run([sys.executable, str(ROOT/'download_checkpoints.py'),
                '--what', 'assembly', '--problems', 'beam_assembly_step1'], cwd=ROOT, check=True)
        os.execv(sys.executable, [sys.executable, '-u', str(ROOT/'tools/run_beam_full_curriculum.py'),
            '--run-id', args.run_id])
    if args.mode == 'train-from060':
        os.execv(sys.executable, [sys.executable, '-u', str(ROOT/'tools/run_beam_fixed_curriculum.py'),
            '--run-id', args.run_id, '--source-accepted',
            'pretrained_assembly/beam_half/source060/accepted.json'])
    else:
        evaluate(checkpoint, ROOT/'evaluation_runs'/args.run_id,
                 [601] if args.mode == 'smoke' else list(range(601,609)))

if __name__ == '__main__':
    main()
