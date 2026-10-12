"""Offline (headless) success-rate evaluation of the released checkpoints.

For each of the four assembly problems, this runs the policy across a large batch
of parallel envs in Isaac Sim and scores only each env's FIRST episode (so the N
envs are N i.i.d. trials). Envs whose first episode ends within --min-episode-steps
are discarded as unstable inits (object fell / ungraspable at reset → the task was
impossible). It reports the success rate over the remaining valid trials.

Because Isaac Sim / Kit cannot cleanly re-initialize a different env in one
process, the driver evaluates each problem in its own subprocess (the same
pattern as ``eval_isaacsim.py``) and prints a summary table.

Usage:
    # Evaluate all four problems with the downloaded checkpoints
    python evaluation/eval_offline.py --policies-dir pretrained_assembly

    # A single problem
    python evaluation/eval_offline.py \
        --problem tight_insertion \
        --checkpoint-path pretrained_assembly/tight_insertion/model.pth

    # One rollout with industrial diagnostics and one compact trace
    python evaluation/eval_offline.py \
        --problem tight_insertion --num-envs 1 --diagnostics \
        --diagnostics-trace-envs 0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASK = "Isaacsimenvs-PreciseAssembly-Direct-v0"
DEFAULT_AGENT = "rl_games_sapg_cfg_entry_point"
ALL_PROBLEMS = [
    "tight_insertion",
    "beam_assembly_step1",
    "beam_assembly_step2",
    "screwing",
]
_RESULT_PREFIX = "RESULT_JSON "


# ─────────────────────────────────────────────────────────────────────────────
# Worker: evaluate ONE problem in this process (after Kit boots).
# ─────────────────────────────────────────────────────────────────────────────
def run_worker(args) -> int:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = True
    launcher_args.enable_cameras = False
    app = AppLauncher(launcher_args).app

    import math as _math

    import torch

    import isaacsimenvs  # noqa: F401  triggers gym.register side effects
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
    from rl_games.torch_runner import Runner, _load_checkpoint_weights

    # Reuse the eval building blocks so this stays in lock-step with eval_isaacsim.
    from evaluation.eval_isaacsim import (
        _apply_env_overrides,
        _configure_agent,
        _instantiate_env,
        _load_env_cfg,
    )

    result = {"problem": args.problem, "checkpoint": args.checkpoint_path, "seed": args.seed}
    env = None
    diagnostics = None
    try:
        cfg = _load_env_cfg(args.task)
        cfg.seed = args.seed
        _apply_env_overrides(
            cfg,
            problem=args.problem,
            goal_mode=args.goal_mode,
            random_goal_fraction=0.0,
            insertion_success_tolerance=args.insertion_success_tolerance,
            retract_success_tolerance=args.retract_success_tolerance,
            num_envs=args.num_envs,
            sim_device=args.sim_device,
            sdf=False,
            keep_dr=False,
            extra_overrides={},
        )
        if args.diagnostics:
            cfg.precise_assembly.enable_diagnostics_contact_reporting = True
            # The task defaults reserve GPU contact buffers for 1K-way training.
            # Diagnostic runs are intentionally small; right-size only capacity
            # limits (not solver behavior) so contact reporting does not reserve
            # gigabytes on a workstation GPU.
            if args.num_envs <= 64:
                physx = cfg.sim.physx
                physx.gpu_max_rigid_contact_count = min(
                    physx.gpu_max_rigid_contact_count, max(2**18, args.num_envs * 4096)
                )
                physx.gpu_max_rigid_patch_count = min(
                    physx.gpu_max_rigid_patch_count, max(2**16, args.num_envs * 1024)
                )
                physx.gpu_found_lost_aggregate_pairs_capacity = min(
                    physx.gpu_found_lost_aggregate_pairs_capacity,
                    max(2**16, args.num_envs * 2048),
                )
                physx.gpu_total_aggregate_pairs_capacity = min(
                    physx.gpu_total_aggregate_pairs_capacity,
                    max(2**17, args.num_envs * 4096),
                )
                physx.gpu_collision_stack_size = min(
                    physx.gpu_collision_stack_size, max(2**26, args.num_envs * 2**22)
                )
        env = _instantiate_env(args.task, cfg)

        agent_cfg = _configure_agent(
            args.task,
            args.agent,
            rl_device=args.rl_device,
            num_envs=args.num_envs,
            deterministic=True,
            games=10**9,
            extra_overrides={},
        )
        clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", _math.inf))
        clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", _math.inf))
        wrapped = register_rlgames_env(
            env, rl_device=args.rl_device, clip_obs=clip_obs, clip_actions=clip_actions
        )

        runner = Runner()
        runner.load(agent_cfg)
        runner.reset()
        player = runner.create_player()
        player.set_weights(_load_checkpoint_weights(player, str(args.checkpoint_path)))
        player.has_batch_dimension = True

        player.reset()
        obs = player.env_reset(wrapped)

        if args.diagnostics:
            from evaluation.assembly_diagnostics import AssemblyDiagnostics, parse_trace_envs

            diagnostics = AssemblyDiagnostics(
                env,
                args.diagnostics_dir,
                trace_envs=parse_trace_envs(args.diagnostics_trace_envs, args.num_envs),
                max_run_size_mb=args.diagnostics_max_run_size_mb,
                contact_threshold_n=args.diagnostics_contact_threshold,
                high_force_threshold_n=args.diagnostics_high_force_threshold,
                anomalous_force_threshold_n=args.diagnostics_anomalous_force_threshold,
                anomalous_impulse_threshold_ns=args.diagnostics_anomalous_impulse_threshold,
                max_contact_points_per_env=args.diagnostics_max_contact_points_per_env,
                checkpoint=str(args.checkpoint_path),
            )
            diagnostics.attach()

        # Batched eval: all envs run in parallel and we score only each env's
        # FIRST episode, so the N envs are N i.i.d. trials. Per-episode success is
        # read from env._prev_episode_successes, which reset_utils copies from
        # env._successes right before each auto-reset (it survives the reset).
        n = args.num_envs
        dev = env.device
        done_once = torch.zeros(n, dtype=torch.bool, device=dev)
        first_len = torch.zeros(n, dtype=torch.long, device=dev)
        first_full = torch.zeros(n, dtype=torch.bool, device=dev)
        first_ratio = torch.zeros(n, dtype=torch.float, device=dev)
        first_retract = torch.zeros(n, dtype=torch.bool, device=dev)
        has_retract = hasattr(env, "retract_succeeded")

        step = 0
        while step < args.steps and not bool(done_once.all()):
            action = player.get_action(obs, is_deterministic=True)
            if diagnostics is not None:
                diagnostics.set_policy_action(action)
            obs, _rew, dones, _infos = player.env_step(wrapped, action)
            newly = dones.reshape(-1).to(dev).bool() & ~done_once
            ids = torch.nonzero(newly, as_tuple=False).reshape(-1)

            if diagnostics is not None:
                # Auto-reset has already cleared current episode state for done
                # envs. Restore their terminal assembly flag from the snapshot
                # retained by reset_utils before feeding the diagnostics FSM.
                assembly_now = env._successes >= env.env_max_goals
                if ids.numel():
                    assembly_now = assembly_now.clone()
                    assembly_now[ids] = (
                        env._prev_episode_successes[ids]
                        >= env.prev_episode_env_max_goals[ids]
                    )
                goal_count_now = env._successes.clone()
                if ids.numel():
                    goal_count_now[ids] = env._prev_episode_successes[ids]
                diagnostics.policy_step(assembly_now, goal_count=goal_count_now)

            if ids.numel():
                succ = env._prev_episode_successes[ids]
                maxg = env.prev_episode_env_max_goals[ids].clamp_min(1)
                first_len[ids] = step + 1
                first_full[ids] = succ >= env.prev_episode_env_max_goals[ids]
                first_ratio[ids] = (succ.float() / maxg.float()).clamp(0.0, 1.0)
                # retract_succeeded is reset on auto-reset, so read the pre-reset
                # value the env logs each step in extras["episode_final"].
                rs = env.extras.get("episode_final", {}).get("retract_success") if has_retract else None
                if rs is not None:
                    first_retract[ids] = rs[ids].to(dev) > 0.5
                if diagnostics is not None:
                    task_success = first_retract[ids] if has_retract else first_full[ids]
                    diagnostics.finalize(
                        ids,
                        completed=torch.ones(ids.numel(), dtype=torch.bool, device=dev),
                        assembly_success=first_full[ids],
                        task_success=task_success,
                        goal_ratio=first_ratio[ids],
                        policy_steps=first_len[ids],
                    )
                done_once[ids] = True
            step += 1

        if diagnostics is not None:
            unfinished = torch.nonzero(~done_once, as_tuple=False).reshape(-1)
            if unfinished.numel():
                maxg = env.env_max_goals[unfinished].clamp_min(1)
                assembly = env._successes[unfinished] >= env.env_max_goals[unfinished]
                diagnostics.finalize(
                    unfinished,
                    completed=torch.zeros(unfinished.numel(), dtype=torch.bool, device=dev),
                    assembly_success=assembly,
                    task_success=torch.zeros(unfinished.numel(), dtype=torch.bool, device=dev),
                    goal_ratio=(env._successes[unfinished].float() / maxg.float()).clamp(0.0, 1.0),
                    policy_steps=torch.full(
                        (unfinished.numel(),), step, dtype=torch.long, device=dev
                    ),
                )
            result["diagnostics_dir"] = str(Path(args.diagnostics_dir).resolve())

        # Exclude envs whose first episode ended almost immediately: those starts
        # were unstable / ungraspable (object fell at reset), so the task was
        # impossible — not a fair trial of the policy.
        valid = done_once & (first_len >= args.min_episode_steps)
        nv = int(valid.sum().item())
        n_done = int(done_once.sum().item())
        result.update(
            num_envs=n,
            n_first_episodes=nv,
            n_unstable_init=n_done - nv,
            success_rate=(float(first_full[valid].float().mean().item()) if nv else 0.0),
            mean_goal_ratio=(float(first_ratio[valid].mean().item()) if nv else 0.0),
            retract_rate=(float(first_retract[valid].float().mean().item()) if (has_retract and nv) else None),
        )
        print(_RESULT_PREFIX + json.dumps(result), flush=True)
    finally:
        try:
            if diagnostics is not None:
                diagnostics.close()
            if env is not None:
                env.close()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        del app
    os._exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Driver: one subprocess per problem, then print a summary table.
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_checkpoint(args, problem: str) -> Path | None:
    if args.checkpoint_path:
        return Path(args.checkpoint_path)
    cand = Path(args.policies_dir) / problem / "model.pth"
    if not cand.is_absolute():
        cand = REPO_ROOT / cand
    return cand if cand.is_file() else None


def run_driver(args) -> int:
    problems = [args.problem] if args.problem else list(ALL_PROBLEMS)
    diagnostics_root = None
    if args.diagnostics:
        diagnostics_root = Path(args.diagnostics_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = []
    worker_failed = False
    for problem in problems:
        ckpt = _resolve_checkpoint(args, problem)
        if ckpt is None or not ckpt.is_file():
            print(f"[skip] {problem}: no checkpoint (looked in {args.policies_dir}/{problem}/model.pth)")
            rows.append({"problem": problem, "error": "missing checkpoint"})
            worker_failed = True
            continue
        cmd = [
            sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
            "--problem", problem,
            "--checkpoint-path", str(ckpt),
            "--task", args.task,
            "--agent", args.agent,
            "--num-envs", str(args.num_envs),
            "--steps", str(args.steps),
            "--seed", str(args.seed),
            "--min-episode-steps", str(args.min_episode_steps),
            "--rl-device", args.rl_device,
            "--sim-device", args.sim_device,
            "--goal-mode", args.goal_mode,
            "--insertion-success-tolerance", str(args.insertion_success_tolerance),
            "--retract-success-tolerance", str(args.retract_success_tolerance),
        ]
        if args.diagnostics:
            diagnostics_dir = diagnostics_root / problem
            cmd.extend(
                [
                    "--diagnostics",
                    "--diagnostics-dir", str(diagnostics_dir),
                    "--diagnostics-trace-envs", args.diagnostics_trace_envs,
                    "--diagnostics-max-run-size-mb", str(args.diagnostics_max_run_size_mb),
                    "--diagnostics-contact-threshold", str(args.diagnostics_contact_threshold),
                    "--diagnostics-high-force-threshold", str(
                        args.diagnostics_high_force_threshold
                    ),
                    "--diagnostics-anomalous-force-threshold", str(
                        args.diagnostics_anomalous_force_threshold
                    ),
                    "--diagnostics-anomalous-impulse-threshold", str(
                        args.diagnostics_anomalous_impulse_threshold
                    ),
                    "--diagnostics-max-contact-points-per-env", str(
                        args.diagnostics_max_contact_points_per_env
                    ),
                ]
            )
        env = os.environ.copy()
        env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        print(f"\n===== evaluating {problem} =====", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                              stdout=subprocess.PIPE, stderr=None, text=True)
        row = {"problem": problem, "error": "no result"}
        for line in proc.stdout.splitlines():
            print(line)
            if line.startswith(_RESULT_PREFIX):
                row = json.loads(line[len(_RESULT_PREFIX):])
        if proc.returncode != 0:
            worker_failed = True
            row = {"problem": problem, "error": f"worker exited {proc.returncode}"}
        rows.append(row)

    if diagnostics_root is not None:
        from evaluation.report_assembly_diagnostics import generate_report

        try:
            report_paths = generate_report(diagnostics_root)
        except FileNotFoundError as exc:
            print(f"[diagnostics] report skipped: {exc}")
        else:
            print(f"[diagnostics] HTML report: {report_paths['report_html']}")

    print("\n" + "=" * 86)
    print(f"{'problem':28s} {'trials':>8s} {'bad_init':>9s} {'success':>9s} {'goal_ratio':>11s} {'retract':>9s}")
    print("-" * 86)
    for r in rows:
        if "success_rate" not in r:
            print(f"{r['problem']:28s} {'-':>8s} {'-':>9s} {'-':>9s} {'-':>11s} {'-':>9s}  ({r.get('error','?')})")
            continue
        retr = "-" if r.get("retract_rate") is None else f"{r['retract_rate']*100:6.1f}%"
        print(f"{r['problem']:28s} {r['n_first_episodes']:>8d} {r.get('n_unstable_init', 0):>9d} "
              f"{r['success_rate']*100:>8.1f}% {r['mean_goal_ratio']*100:>10.1f}% {retr:>9s}")
    print("=" * 86)
    print("trials = valid first episodes; bad_init = discarded unstable-init starts;")
    print("success = first episode reached all subgoals (over valid trials).")
    return 1 if worker_failed else 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", default=None, help="Single problem to eval (default: all four).")
    p.add_argument("--policies-dir", default="pretrained_assembly",
                   help="Dir of <problem>/model.pth checkpoints (driver mode).")
    p.add_argument("--checkpoint-path", default=None, help="Explicit checkpoint (single problem).")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--agent", default=DEFAULT_AGENT)
    p.add_argument("--num-envs", type=int, default=1024,
                   help="Parallel envs = number of i.i.d. first-episode trials.")
    p.add_argument("--steps", type=int, default=2500,
                   help="Max control steps; loop ends early once every env finishes episode 1. "
                        "Higher so long tasks (e.g. screwing, 10 subgoals) finish.")
    p.add_argument("--seed", type=int, default=42, help="Evaluation environment seed recorded in diagnostics metadata.")
    p.add_argument("--min-episode-steps", type=int, default=100,
                   help="Discard first episodes that end within this many steps as unstable "
                        "inits (object fell/ungraspable at reset → impossible trial).")
    p.add_argument("--goal-mode", default="preInsertAndFinal")
    p.add_argument("--insertion-success-tolerance", type=float, default=0.01)
    p.add_argument("--retract-success-tolerance", type=float, default=0.005)
    p.add_argument("--rl-device", default="cuda:0")
    p.add_argument("--sim-device", default="cuda:0")
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="Enable evaluation-only industrial assembly diagnostics.",
    )
    p.add_argument(
        "--diagnostics-dir",
        default="evaluation_runs/assembly_diagnostics",
        help="Output root (driver adds a per-problem directory).",
    )
    p.add_argument(
        "--diagnostics-trace-envs",
        default="0",
        help="Comma-separated env ids with compact control-rate NPZ traces; empty disables traces.",
    )
    p.add_argument(
        "--diagnostics-max-run-size-mb",
        type=float,
        default=100.0,
        help="Upper bound for compressed traces; JSON summaries and events remain sparse.",
    )
    p.add_argument("--diagnostics-contact-threshold", type=float, default=0.5)
    p.add_argument("--diagnostics-high-force-threshold", type=float, default=50.0)
    p.add_argument("--diagnostics-anomalous-force-threshold", type=float, default=500.0)
    p.add_argument("--diagnostics-anomalous-impulse-threshold", type=float, default=100.0)
    p.add_argument("--diagnostics-max-contact-points-per-env", type=int, default=512)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.diagnostics and args.num_envs > 64:
        parser.error("--diagnostics is evaluation-only and supports at most 64 environments")
    if args.worker:
        return run_worker(args)
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
