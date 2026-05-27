#!/bin/bash

# job_dir.sh — Central definition for the per-job directory naming convention.
#
# Usage (sourced by submit.sh, k8s_submit.sh, _k8s_job.sh, run_setup.sh,
# in_container_run.sh, _container.sh, _train.sh, utils/ray_cluster.sh):
#   source utils/job_dir.sh
#   JOB_DIR=$(make_job_dir "$JOB_ID" "$JOB_NAME")
#   if is_checkpointing_enabled "$JOB_DIR"; then ...
#   OUTPUT_PATH=$(resolve_output_path "$JOB_DIR" "$MODEL_NAME" "$MODEL_NAME_ALIAS")
#
# All callers MUST use these functions so the naming stays consistent.

make_job_dir() {
    local job_id="$1"
    local job_name="$2"
    printf '%s-%s' "$job_id" "$job_name"
}

# Check if checkpointing is enabled by inspecting JOB_DIR for the
# enable_checkpointing passthrough arg. Accepts true, 1, yes (case-insensitive).
is_checkpointing_enabled() {
    local job_dir="${1,,}"
    [[ "$job_dir" =~ enable_checkpointing_(true|1|yes)(-|$) ]]
}

# Resolve the runtime base directory for outputs/checkpoints.
# Priority:
#   1) OUTPUTS_BASE_DIR (explicit override)
#   2) JOB_WORKSPACE (set by run_setup.sh; e.g., in_container_run.sh)
#   3) /outputs (container default mount path)
resolve_outputs_base_dir() {
    local base="${OUTPUTS_BASE_DIR:-${JOB_WORKSPACE:-/outputs}}"
    # Normalize trailing slash (except root).
    if [[ "$base" != "/" ]]; then
        base="${base%/}"
    fi
    echo "$base"
}

# Resolve the base output directory for a training run.
# Checkpointing → model-based path (persists across restarts).
# Non-checkpointing → job-based path (unique per run).
resolve_output_path() {
    local job_dir="$1"
    local model_name="$2"
    local model_alias="${3:-}"
    local outputs_base
    outputs_base="$(resolve_outputs_base_dir)"
    if is_checkpointing_enabled "$job_dir"; then
        echo "${outputs_base}/${model_alias:-$model_name}"
    else
        echo "${outputs_base}/${job_dir}"
    fi
}
