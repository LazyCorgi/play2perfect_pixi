#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Optional artifact is SHA256-verified before extraction.
if [[ "${1:-}" == "--artifact" ]]; then
    archive="${2:?Provide the path to beam-half-artifacts.tar.gz}"
    expected="$(cut -d ' ' -f 1 releases/beam_half_artifacts.sha256)"
    actual="$(sha256sum "$archive")"
    [[ "${actual%% *}" == "$expected" ]] || { echo 'Artifact SHA256 mismatch' >&2; exit 1; }
    tar -xzf "$archive"
    shift 2
fi
if [[ $# -eq 0 ]]; then set -- demo; fi
pixi install --locked
export OMNI_KIT_ACCEPT_EULA=YES
exec pixi run --locked python tools/beam_half.py "$@"
