#!/usr/bin/env bash
#SBATCH --partition=cpuonly
#SBATCH --account=k2m
#SBATCH --qos=cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
set -Eeuo pipefail
umask 077
: "${HLE_REPO:?set HLE_REPO to the frozen worktree}"
: "${HLE_PYTHON:?set HLE_PYTHON}"
: "${HLE_CONFIG:?set HLE_CONFIG}"
: "${HLE_MANIFEST_DIR:?set HLE_MANIFEST_DIR}"
: "${HLE_OUTPUT:?set HLE_OUTPUT}"
mkdir -p "$HLE_OUTPUT"
driver_local="${SLURM_TMPDIR:-/var/tmp}/hle-driver-${UID}-${SLURM_JOB_ID}"
mkdir -p "$driver_local"
echo "HLE driver local log: $driver_local/driver.log"
exec >"$driver_local/driver.log" 2>&1
exec 9>"$HLE_OUTPUT/lane.lock"
flock -n 9 || { echo "another process owns this lane"; exit 2; }
source "$HLE_REPO/solvers/hle-tools-v0/m2/prepare-enroot.sh"
finish() {
  code=$?
  cleanup_m2_enroot
  cp "$driver_local/driver.log" "$HLE_OUTPUT/driver-$SLURM_JOB_ID.log" || code=2
  exit "$code"
}
trap finish EXIT
export PYTHONPATH="$HLE_REPO"
"$HLE_PYTHON" "$HLE_REPO/solvers/hle-tools-v0/m2/selftest.py"
lane_result=0
for manifest in "$HLE_MANIFEST_DIR"/*.json; do
  [[ -f "$manifest" ]] || { echo "no shard manifests"; exit 2; }
  shard=$(basename "$manifest" .json)
  attempt="$HLE_OUTPUT/$shard/attempt-$SLURM_JOB_ID"
  shard_result=0
  "$HLE_PYTHON" "$HLE_REPO/solvers/hle-tools-v0/m2/run_shard.py" \
    --config "$HLE_CONFIG" --manifest "$manifest" --output "$attempt" || shard_result=$?
  if [[ "$shard_result" == 3 ]]; then
    lane_result=3
  elif [[ "$shard_result" != 0 ]]; then
    exit "$shard_result"
  fi
done
exit "$lane_result"
