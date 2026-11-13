"""Download Play2Perfect checkpoints from the Hugging Face Hub.

Downloads the Stage-1 "play" policy and/or the four finetuned precise-assembly
policies:

  pretrained_policy/model.pth                      # Stage-1 play
  pretrained_assembly/<problem_key>/model.pth       # 4 assembly tasks

The ``pretrained_assembly/`` layout is what ``evaluation/eval_isaacsim.py``
discovers via ``--policies-dir pretrained_assembly``.

Usage:
    python download_checkpoints.py                       # everything (play + 4 tasks)
    python download_checkpoints.py --what play           # just the Stage-1 policy
    python download_checkpoints.py --what assembly       # just the 4 task policies
    python download_checkpoints.py --problems tight_insertion
"""

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import List

import tyro

DEFAULT_REPO_ID = "Kushal20/play2perfect-checkpoints"
REPO_ROOT = Path(__file__).parent

# Subfolder in the HF repo that holds the Stage-1 play policy.
PLAY_SUBDIR = "play"
OFFICIAL_SHA256 = {
    'play': '4cb0d09833326de6878ee5681c2b14521823fa616e01b135214a51ffe41c63b9',
    'beam_assembly_step1': 'be7e9f6c367e76a8f0697a65d3253cc8ccbdbeed83ad2dd21a126df5455481e0',
}

# The four released assembly tasks (keys into evaluation.problems.PROBLEM_REGISTRY).
ALL_PROBLEMS = [
    "tight_insertion",
    "beam_assembly_step1",
    "beam_assembly_step2",
    "screwing",
]


@dataclass
class DownloadArgs:
    repo_id: str = DEFAULT_REPO_ID
    """Hugging Face model repo holding the checkpoints."""
    what: str = "all"
    """Which checkpoints to fetch: 'all', 'play', or 'assembly'."""
    problems: List[str] = field(default_factory=lambda: list(ALL_PROBLEMS))
    """Subset of assembly problem keys to download (defaults to all four)."""
    endpoint: str | None = None
    """Optional explicitly selected Hub endpoint; otherwise respects HF_ENDPOINT/default Hub."""
    retries: int = 3
    """Maximum attempts for network failures."""


def _verify(path: Path, repo_id: str, subdir: str) -> None:
    expected = OFFICIAL_SHA256.get(subdir) if repo_id == DEFAULT_REPO_ID else None
    if expected:
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Checkpoint SHA256 mismatch at {path}: {actual}; expected {expected}. '
                             'Keep the incorrect file separately and obtain the official checkpoint again.')


def _download(args: DownloadArgs, subdir: str, dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    target = dest / 'model.pth'
    if target.is_file():
        _verify(target, args.repo_id, subdir)
        print(f'[skip] verified checkpoint already at {target}')
        return
    for attempt in range(args.retries):
        try:
            # Unlike snapshot_download, this does not return an arbitrary local
            # directory when the Hub request fails and no model exists there.
            cached = Path(hf_hub_download(repo_id=args.repo_id, filename=f'{subdir}/model.pth',
                                          endpoint=args.endpoint))
            break
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status in (401, 403, 404):
                raise RuntimeError(f'Hub denied or could not find {args.repo_id}/{subdir}/model.pth '
                                   f'(HTTP {status}); check repo/access/endpoint.') from exc
            if attempt + 1 == args.retries:
                raise RuntimeError(
                    f'Checkpoint download failed after {args.retries} attempts: {exc}. '
                    'Check the network/proxy/TLS connection or select --endpoint. '
                    f'Alternatively copy the official model from another machine to {target}. '
                    'Environment installation and CUDA checks need not be repeated.'
                ) from exc
            print(f'[retry {attempt+1}/{args.retries}] {type(exc).__name__}: {exc}', flush=True)
            time.sleep(2 ** (attempt+1))
    _verify(cached, args.repo_id, subdir)
    dest.mkdir(parents=True, exist_ok=True)
    # Publish only a complete, verified model; interruption cannot leave a
    # partial model.pth that a later invocation would silently skip.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=dest, prefix='.model-', suffix='.tmp', delete=False) as out:
            temporary = Path(out.name)
            with cached.open('rb') as source:
                shutil.copyfileobj(source, out)
            out.flush()
            os.fsync(out.fileno())
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f'  -> {target}')


def run_download(args: DownloadArgs) -> None:
    if args.what not in ("all", "play", "assembly"):
        raise ValueError(f"--what must be all|play|assembly, got {args.what!r}")
    if args.retries < 1:
        raise ValueError('--retries must be positive')
    if any(problem not in ALL_PROBLEMS for problem in args.problems):
        raise ValueError(f'--problems must be selected from {ALL_PROBLEMS}')

    if args.what in ("all", "play"):
        dest = REPO_ROOT / "pretrained_policy"
        print(f"[download] Stage-1 play policy <- {args.repo_id}/{PLAY_SUBDIR}")
        _download(args, PLAY_SUBDIR, dest)

    if args.what in ("all", "assembly"):
        out = REPO_ROOT / "pretrained_assembly"
        for problem in args.problems:
            print(f"[download] {problem} <- {args.repo_id}")
            _download(args, problem, out / problem)

    print("DONE!")


def main() -> None:
    run_download(tyro.cli(DownloadArgs))


if __name__ == "__main__":
    main()
