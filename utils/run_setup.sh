#!/bin/bash

# run_setup.sh — Shared setup for run_local.sh and in_container_run.sh
#
# Sourced (not executed) by callers. Expects:
#   SCRIPT_DIR — path to the repo root (set by caller)
#   _JOB_START — $SECONDS at script start (set by caller)
#   $@         — caller's positional args (inherited via source)
#
# After sourcing, callers have access to:
#   MODEL_NAME, MODEL_NAME_ALIAS, EXP_TAG, PASSTHROUGH_ARGS
#   JOB_NAME, JOB_ID, JOB_DIR, JOB_WORKSPACE
#   JAX_PORT, JAX_COORDINATOR_PORT, RAY_PORT (if RAY=1)
#   LOG_FILE
#   _print_summary (EXIT trap already installed)
#
# Callers set _RUN_RC before exiting so _print_summary reports the right code

# ============================================================================
# Preflight warnings (non-fatal, device-agnostic)
# ============================================================================

# Warn if GPUs are occupied (best-effort, does not abort)
if [[ -e /dev/kfd ]] && command -v rocm-smi &>/dev/null; then
    rocm-smi --showpids 2>/dev/null | grep -qE '^[0-9]+' && echo "WARNING: GPUs appear occupied (rocm-smi)." >&2
elif command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q . && echo "WARNING: GPUs appear occupied (nvidia-smi)." >&2
fi

# ============================================================================
# Parse arguments
# ============================================================================

source "$SCRIPT_DIR/utils/parse_job_args.sh"

echo "MODEL_NAME=$MODEL_NAME"
[[ -n "$MODEL_NAME_ALIAS" ]] && echo "MODEL_NAME_ALIAS=$MODEL_NAME_ALIAS"
[[ -n "$EXP_TAG" ]] && echo "EXP_TAG=$EXP_TAG"

# Warn if the user accidentally passed sbatch flags
if [[ ${#SBATCH_ARGS[@]} -gt 0 ]]; then
    echo "WARNING: Ignoring sbatch args (${SBATCH_ARGS[*]}) — not applicable." >&2
    echo "         For Slurm submission use submit.sh; for Kubernetes use k8s_submit.sh." >&2
fi

# ============================================================================
# Environment
# ============================================================================

export MAXTEXT_SLURM_DIR="$SCRIPT_DIR"
export JAX_COORDINATOR_IP="${JAX_COORDINATOR_IP:-127.0.0.1}"

# _container.sh exports JOB_ID=unknown and JAX_COORDINATOR_PORT=0 as sentinel
# defaults; treat them as unset so we generate proper values
[[ "${JOB_ID:-}" == "unknown" ]] && unset JOB_ID
[[ "${JAX_COORDINATOR_PORT:-}" == "0" ]] && unset JAX_COORDINATOR_PORT

export JOB_ID="${JOB_ID:-local_$(date +%Y%m%d_%H%M%S)_$(printf '%04x' $RANDOM)}"
export JOB_NAME
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MODEL_NAME_ALIAS
source "$SCRIPT_DIR/utils/detect_ip.sh"
export LOGIN_NODE_HOSTNAME="${LOGIN_NODE_HOSTNAME:-${USER}@$(hostname -f 2>/dev/null || hostname)}"
export LOGIN_NODE_IP="${LOGIN_NODE_IP:-${USER}@$(detect_ip)}"

# Default to outputs/ next to the scripts unless explicitly overridden.
# This keeps direct in-container runs independent of the /outputs mount alias.
_JOB_WORKSPACE_DEFAULT="$SCRIPT_DIR/outputs"
export JOB_WORKSPACE="${JOB_WORKSPACE:-$_JOB_WORKSPACE_DEFAULT}"

source "$SCRIPT_DIR/utils/job_dir.sh"
JOB_DIR=$(make_job_dir "$JOB_ID" "$JOB_NAME")
export JOB_DIR

if [[ "${RAY:-0}" == "1" || "${RAY:-}" == "true" ]]; then
    export USE_RAY=true
fi

mkdir -p "$JOB_WORKSPACE/$JOB_DIR"; chmod a+w "$JOB_WORKSPACE/$JOB_DIR"
echo "JOB_NAME=$JOB_NAME  JOB_ID=$JOB_ID  JOB_WORKSPACE=$JOB_WORKSPACE"

# ============================================================================
# Ports
# ============================================================================

source "$SCRIPT_DIR/utils/pick_port.sh"

JAX_PORT="${JAX_COORDINATOR_PORT:-$(pick_free_port)}"
export JAX_COORDINATOR_PORT="$JAX_PORT"

# Pick a free port for Ray head GCS (only when Ray is enabled).
# Respect an inherited RAY_PORT (e.g. set by k8s manifest for cross-pod
# consistency). Pick fresh only when not pre-set.
if [[ "${USE_RAY:-false}" == "true" ]]; then
    RAY_PORT="${RAY_PORT:-$(pick_free_port "$JAX_PORT")}"
    export RAY_PORT
fi

# ============================================================================
# Logging
# ============================================================================

# Log to file (mirrors the --output path) while still printing to terminal
LOG_FILE="$JOB_WORKSPACE/$JOB_DIR.log"
echo "LOG_FILE=$LOG_FILE"

# Symlink to the job log inside the job directory for easy access
ln -snf "../$JOB_DIR.log" "$JOB_WORKSPACE/$JOB_DIR/log"

# ---- Standardized log header (same key=value format as _job.sbatch) ----
# Parsers (analyze_job.py, tgs_tagger.py, perf_server.py) match these names.
{
    echo "JOB_ID=$JOB_ID"
    echo "JOB_NAME=$JOB_NAME"
    echo "NNODES=$NNODES"
    echo "JOB_NODELIST=$(hostname)"
    echo "JAX_COORDINATOR_PORT=$JAX_PORT"
    if [[ -n "${RAY_PORT:-}" ]]; then echo "RAY_PORT=$RAY_PORT"; fi
    echo "PASSTHROUGH_ARGS=\"${PASSTHROUGH_ARGS[*]}\""
    echo "MODEL_NAME=$MODEL_NAME"
    if [[ -n "$MODEL_NAME_ALIAS" ]]; then echo "MODEL_NAME_ALIAS=$MODEL_NAME_ALIAS"; fi
    if [[ -n "$EXP_TAG" ]]; then echo "EXP_TAG=$EXP_TAG"; fi
} | tee ${_IN_CONTAINER_LOG_TEE_ACTIVE:+-a} "$LOG_FILE"
# ↑ _IN_CONTAINER_LOG_TEE_ACTIVE (set by k8s non-rank-0 pods) adds -a
#   to avoid truncating rank 0's output in the shared log.

# Hold an append-mode fd to the log file. Unlike >> "$LOG_FILE" (which opens
# by name), the fd follows the inode — so tgs_tagger -f can rename the file
# mid-run and the JOB SUMMARY still lands in the correct (renamed) file.
LOG_APPEND_FD=3
exec {LOG_APPEND_FD}>>"$LOG_FILE"

# Emit code provenance (git summary or artifact summary) once per run.
source "$SCRIPT_DIR/utils/code_provenance.sh"
emit_code_provenance "$SCRIPT_DIR" "" "$LOG_APPEND_FD"

# ============================================================================
# Job summary (EXIT trap; callers set _RUN_RC before exiting)
# ============================================================================

_fmt_elapsed() {
    local s=$1
    if (( s >= 3600 )); then printf '%dh %dm %ds' $((s/3600)) $((s%3600/60)) $((s%60))
    elif (( s >= 60 )); then printf '%dm %ds' $((s/60)) $((s%60))
    else printf '%ds' "$s"
    fi
}
_print_summary() {
    local rc=${_RUN_RC:-$?}
    local status
    local green red reset
    green=$(tput setaf 2 2>/dev/null) || green=""
    red=$(tput setaf 1 2>/dev/null) || red=""
    reset=$(tput sgr0 2>/dev/null) || reset=""
    if [[ $rc -eq 0 ]]; then
        status="${green}SUCCESS (exit 0)${reset}"
    else
        status="${red}FAILED (exit $rc)${reset}"
    fi
    local summary
    summary="========================== JOB SUMMARY ==========================
  Job:    $JOB_ID ($JOB_NAME)
  Model:  $MODEL_NAME
  Nodes:  $NNODES
  Wall:   $(_fmt_elapsed $(( SECONDS - _JOB_START )))
  Status: $status
================================================================="
    echo "$summary"
    # Write via the append fd (not by name) so the summary goes to the
    # correct file even if tgs_tagger -f renamed it while the job was running.
    echo "$summary" >&$LOG_APPEND_FD 2>/dev/null
}
trap _print_summary EXIT
