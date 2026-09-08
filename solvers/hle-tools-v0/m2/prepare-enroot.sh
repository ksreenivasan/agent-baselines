#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source this file from inside an M2 Slurm allocation" >&2
  exit 2
fi

: "${SLURM_JOB_ID:?M2 Enroot preparation requires a Slurm allocation}"

enroot_root=${M2_ENROOT_ROOT:-/var/tmp/enroot-${UID}-${SLURM_JOB_ID}}
export ENROOT_CACHE_PATH="$enroot_root/cache"
export ENROOT_DATA_PATH="$enroot_root/data"
export ENROOT_RUNTIME_PATH="$enroot_root/runtime"
export ENROOT_TEMP_PATH="$enroot_root/tmp"
export M2_ENROOT_CONTAINER="${M2_ENROOT_CONTAINER:-hle-tools-${SLURM_JOB_ID}}"

mkdir -p "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH" "$ENROOT_RUNTIME_PATH" "$ENROOT_TEMP_PATH"
image="$enroot_root/python-3.11.sqsh"
if [[ ! -f "$image" ]]; then
  enroot import -o "$image" docker://python:3.11-slim-bookworm
fi
if [[ ! -d "$ENROOT_DATA_PATH/$M2_ENROOT_CONTAINER" ]]; then
  enroot create -n "$M2_ENROOT_CONTAINER" "$image"
fi

rootfs="$ENROOT_DATA_PATH/$M2_ENROOT_CONTAINER"
mkdir -p "$rootfs/workspace" "$rootfs/tmp"
if [[ ! -x "$rootfs/opt/inspect_tool_support/bin/inspect-tool-support" ]]; then
  enroot start -w "$M2_ENROOT_CONTAINER" sh -lc '
    python -m venv /opt/inspect_tool_support
    /opt/inspect_tool_support/bin/pip install --no-cache-dir \
      inspect-tool-support==1.2.0 \
      mcp==1.29.1 \
      jupyter_client==8.6.3 \
      jupyter_kernel_gateway==3.0.1 \
      ipykernel==6.30.1 \
      numpy==1.26.4 \
      pandas==2.3.3 \
      scipy==1.16.3 \
      sympy==1.14.0 \
      networkx==3.5 \
      matplotlib==3.10.7
    /opt/inspect_tool_support/bin/inspect-tool-support post-install --no-web-browser
    /opt/inspect_tool_support/bin/python -m ipykernel install --sys-prefix
  '
fi

cleanup_m2_enroot() {
  enroot remove -f "$M2_ENROOT_CONTAINER" >/dev/null 2>&1 || true
}
