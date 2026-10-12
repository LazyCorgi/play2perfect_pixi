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

- **AppLauncher before isaaclab imports.** Isaac Lab sub-namespaces (`isaaclab.sim`,
  `isaaclab.envs`, ...) only resolve after `AppLauncher(args)` runs. Any script importing
  `isaaclab.*` must instantiate `AppLauncher` first.
- **Kit shutdown can hang** after the work is done. Scripts flush stdout/stderr and call
  `os._exit(0)` rather than waiting for a clean Kit teardown.
- **One Isaac Sim instance per GPU.** Booting a second Kit process while another is running
  on the same GPU can crash the booting one mid-startup. Wait for training/eval/demo
  processes to finish before launching another.
