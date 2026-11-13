#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
export OMNI_KIT_ACCEPT_EULA=YES
case "$mode" in
  install)
    pixi install --locked
    pixi run --locked verify
    pixi run --locked python tools/beam_5090.py check
    pixi run --locked smoke-test
    exec pixi run --locked python download_checkpoints.py --what play "$@"
    ;;
  weights) exec pixi run --locked python download_checkpoints.py --what play "$@" ;;
  check|plan|status) exec pixi run --locked python tools/beam_5090.py "$mode" "$@" ;;
  smoke) exec pixi run --locked python tools/beam_5090.py run --smoke "$@" ;;
  train) exec pixi run --locked python tools/beam_5090.py run "$@" ;;
  resume) exec pixi run --locked python tools/beam_5090.py run --resume "$@" ;;
  *) echo 'Usage: bash tools/train_beam_5090.sh install|weights|check|plan|smoke|train|resume|status [options]' >&2; exit 2 ;;
esac
