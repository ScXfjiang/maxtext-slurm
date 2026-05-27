#!/bin/bash

# Starts Ray cluster + Prometheus + TensorBoard, then delegates to _train.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils/ray_cluster.sh"

# Resolve MODEL_NAME and OUTPUT_PATH so the metrics exporter (background) and
# TensorBoard inherit them.  This is the single resolution point for Ray mode —
# _train.sh trusts these values (no redundant re-resolve).
source "$SCRIPT_DIR/utils/resolve_model_name.sh"
source "$SCRIPT_DIR/utils/job_dir.sh"
: "${JOB_DIR:?JOB_DIR must be set (exported by _container.sh)}"
export MODEL_NAME="${1:?model_name is required}"
MODEL_NAME=$(resolve_model_name "$SCRIPT_DIR/configs" "$MODEL_NAME") || exit 1
export OUTPUT_PATH=$(resolve_output_path "$JOB_DIR" "$MODEL_NAME" "${MODEL_NAME_ALIAS:-}")

NODE_RANK=${NODE_RANK:-0}

cleanup() {
    [[ $NODE_RANK -eq 0 ]] && echo "[Ray] Cleaning up..."
    stop_ray_cluster
}
trap cleanup EXIT

if ! start_ray_cluster; then
    echo "[!] Ray cluster failed, falling back to non-Ray mode"
    trap - EXIT
    unset USE_RAY
    exec "$SCRIPT_DIR/_train.sh" "$@"
fi

if [[ $NODE_RANK -eq 0 ]]; then
    print_ray_info
    start_tensorboard "$@"
fi

# Run _train.sh as a child (NOT exec) so this shell stays alive and the
# EXIT trap above runs `stop_ray_cluster` — i.e. `ray stop --force` plus
# `pkill` of metrics_exporter / prometheus / tensorboard — before the
# container exits and docker hard-kills everything.  An exec here would
# replace this bash with _train.sh's, dropping the EXIT trap and leaving
# Ray daemons to be SIGKILL'd by docker on container teardown; that
# dirty shutdown races the actor's `ray.kill` in `_ray_actor.py` and
# causes asymmetric rank-N exit-1 even when training succeeded.
export USE_RAY=true
bash "$SCRIPT_DIR/_train.sh" "$@"
exit "$?"
