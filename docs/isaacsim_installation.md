# Isaac Sim (Isaac Lab) installation

Play2Perfect runs in Isaac Sim via Isaac Lab. The reproducible environment is managed by
Pixi and lives in `.pixi/`.

## Prerequisites

- A recent NVIDIA driver with CUDA 12.8 support (tested with 580.95.05)
- [Pixi](https://pixi.sh/)

## Install

```bash
pixi install --locked
```

The lock file pins Python 3.11, Isaac Lab 2.3.2.post1, Isaac Sim 5.1, and
PyTorch 2.7.0+cu128. The local project and vendored `rl_games` package are installed in
editable mode. Keep these pins exact: newer Isaac Lab releases change APIs used by
`isaacsimenvs`.

Verify:

```bash
pixi run verify
pixi run smoke-test
```

## Running

The Pixi verification tasks set the EULA variable themselves. Other non-interactive
launches need it explicitly:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

Optional speedup — point the Omniverse shader cache at a local SSD instead of NFS:

```bash
export OMNI_KIT_CACHE_PATH=/scratch/$USER/ov_cache
mkdir -p "$OMNI_KIT_CACHE_PATH"
```

## Gotchas

- **GPU access from an agent sandbox.** If `nvidia-smi` fails and
  `torch.cuda.is_available()` is false inside a sandbox, run the same checks in a
  normal host terminal before changing drivers. The sandbox may hide `/dev/nvidia*`
  even when the host GPU works. Run Isaac Sim from a GPU-accessible terminal.
- **Hugging Face proxy error.** HTTPX rejects `ALL_PROXY=socks://...`. If your
  local proxy also offers HTTP, override both uppercase and lowercase variables
  for the download process (replace the address with your HTTP proxy):

  ```bash
  pixi run env ALL_PROXY=http://127.0.0.1:10808/ all_proxy=http://127.0.0.1:10808/ \
    python download_checkpoints.py --what assembly --problems tight_insertion
  ```

- **First URDF import.** Isaac Lab requires URDF importer 2.4.31, while the
  Isaac Sim wheel bundles 2.4.30. The first real scene launch downloads the pinned
  extension from the Omniverse registry. Allow this synchronization to finish;
  the import-only smoke test does not trigger it.

- **AppLauncher before isaaclab imports.** Isaac Lab sub-namespaces (`isaaclab.sim`,
  `isaaclab.envs`, ...) only resolve after `AppLauncher(args)` runs. Any script importing
  `isaaclab.*` must instantiate `AppLauncher` first.
- **Kit shutdown can hang** after the work is done. Scripts flush stdout/stderr and call
  `os._exit(0)` rather than waiting for a clean Kit teardown.
- **One Isaac Sim instance per GPU.** Booting a second Kit process while another is running
  on the same GPU can crash the booting one mid-startup. Wait for training/eval/demo
  processes to finish before launching another.

## Small first evaluation

Download one checkpoint and run one environment before attempting the default
1024-environment evaluation:

```bash
pixi run python download_checkpoints.py --what assembly --problems tight_insertion
pixi run python evaluation/eval_offline.py \
  --problem tight_insertion --num-envs 1 --steps 2500 \
  --diagnostics --diagnostics-dir evaluation_runs/first_run
```

The diagnostics option also reduces PhysX buffer reservations for small batches.
This checks scene creation, checkpoint loading, and policy execution. A single
episode is not a reliable estimate of the policy's success rate.
