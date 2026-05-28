#!/bin/bash

# Launch MaxText training.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running _train.sh"
echo "args: $@"

# Split args first
source "$SCRIPT_DIR/utils/split_script_args.sh"
split_script_args "$@"
echo "PASSTHROUGH_ARGS=\"${PASSTHROUGH_ARGS[*]}\""

# Extract environment variables from PASSTHROUGH_ARGS (pattern: _env_KEY=VALUE).
# Values may arrive with surrounding quotes when users write _env_KEY='"value"'
# — the shell keeps the inner quotes, printf '%q' in _container.sh preserves them,
# and the inner bash reconstructs them. Strip one layer of matching outer quotes.
EXTRACTED_ENVS=()
FILTERED_PASSTHROUGH_ARGS=()
for arg in "${PASSTHROUGH_ARGS[@]}"; do
    if [[ "$arg" =~ ^_env_([^=]+)=(.*)$ ]]; then
        env_key="${BASH_REMATCH[1]}"
        env_value="${BASH_REMATCH[2]}"
        # Strip one layer of matching surrounding quotes (double or single).
        if [[ "$env_value" =~ ^\"(.*)\"$ ]]; then
            env_value="${BASH_REMATCH[1]}"
        elif [[ "$env_value" =~ ^\'(.*)\'$ ]]; then
            env_value="${BASH_REMATCH[1]}"
        fi
        EXTRACTED_ENVS+=("$env_key=$env_value")
        echo "Extracted env: $env_key=$env_value"
    else
        FILTERED_PASSTHROUGH_ARGS+=("$arg")
    fi
done
PASSTHROUGH_ARGS=("${FILTERED_PASSTHROUGH_ARGS[@]}")

: "${JOB_DIR:?JOB_DIR must be set (exported by _container.sh)}"
MODEL_NAME=${SCRIPT_ARGS[0]:?model_name is required}
if [[ ! -f "$SCRIPT_DIR/configs/$MODEL_NAME.gpu.yml" ]]; then
    echo "!!! Unknown model: $MODEL_NAME (no configs/$MODEL_NAME.gpu.yml)." >&2
    echo "    Callers must resolve via resolve_model_name.sh before calling _train.sh." >&2
    exit 1
fi
echo "MODEL_NAME=$MODEL_NAME"

# Resolve output path (logic lives in job_dir.sh — single source of truth).
source "$SCRIPT_DIR/utils/job_dir.sh"
export OUTPUT_PATH=$(resolve_output_path "$JOB_DIR" "$MODEL_NAME" "${MODEL_NAME_ALIAS:-}")
echo "OUTPUT_PATH=$OUTPUT_PATH"
mkdir -p -v "$OUTPUT_PATH"

# Build associative array from extracted envs so train_env.sh can look up
# config inputs (e.g. ENABLE_XLA_DUMP) without needing them exported yet.
declare -A EXTRACTED_ENV_MAP
for env_pair in "${EXTRACTED_ENVS[@]}"; do
    EXTRACTED_ENV_MAP["${env_pair%%=*}"]="${env_pair#*=}"
done

# ---- Load environment configuration (edit train_env.sh to customize) ----
source "$SCRIPT_DIR/train_env.sh"

# ---- Load per-model environment overrides (optional) ----
_model_env="$SCRIPT_DIR/configs/$MODEL_NAME.env.sh"
if [[ -f "$_model_env" ]]; then
    echo "Loading per-model env: configs/$MODEL_NAME.env.sh"
    source "$_model_env"
fi

# Export extracted environment variables (after train_env.sh so overrides win).
if [ ${#EXTRACTED_ENVS[@]} -gt 0 ]; then
    echo "Exporting extracted environment variables:"
    for env_pair in "${EXTRACTED_ENVS[@]}"; do
        echo "  export $env_pair"
        export "$env_pair"
    done
fi

# ---- Apply MaxText patch branch (if requested) ----
# Priority: CLI _env_MAXTEXT_PATCH_BRANCH > per-model .env.sh > container env
if [[ -n "${MAXTEXT_PATCH_BRANCH:-}" ]]; then
    echo "[INFO] Checking out $MAXTEXT_PATCH_BRANCH..."
    if git fetch origin "$MAXTEXT_PATCH_BRANCH" && git checkout "origin/$MAXTEXT_PATCH_BRANCH"; then
        echo "[OK] Checked out $MAXTEXT_PATCH_BRANCH at $(git rev-parse --short HEAD)."
    else
        echo "[FAIL] Failed to check out $MAXTEXT_PATCH_BRANCH." >&2
        exit 1
    fi
else
    echo "[SKIP] No MAXTEXT_PATCH_BRANCH set, using image default."
fi

# Assertion: users may override env vars after train_env.sh is sourced.
# If DMABUF gets re-enabled without host kernel metadata, fail fast here.
if [[ "${NCCL_DMABUF_ENABLE:-}" == "1" ]]; then
    _kernel_release="$(uname -r 2>/dev/null || true)"
    if [[ -z "$_kernel_release" || ! -d /boot ]] || ! compgen -G "/boot/*${_kernel_release}*" >/dev/null; then
        echo "[ERROR] Unsafe NCCL_DMABUF_ENABLE=1: missing /boot kernel metadata for '$_kernel_release'." >&2
        echo "[ERROR] Source train_env.sh (which guards this), mount /boot, or set NCCL_DMABUF_ENABLE=0." >&2
        exit 1
    fi
fi

# Handle LD_PRELOAD (after extracting envs so _env_MAXTEXT_LD_PRELOAD works)
if [ -n "${MAXTEXT_LD_PRELOAD:-}" ]; then
    export LD_PRELOAD="$MAXTEXT_LD_PRELOAD"
else
    [ -n "${LD_PRELOAD:-}" ] && echo "[WARNING] LD_PRELOAD='$LD_PRELOAD'; unsetting..."
    unset LD_PRELOAD
fi

echo "Show all environment variables:"
printenv | sort

# ============================================================================
# Launch Training (direct or via Ray actor)
# ============================================================================

# Build training arguments
TRAIN_ARGS=(
    "$SCRIPT_DIR/configs/$MODEL_NAME.gpu.yml"
    base_output_directory=$OUTPUT_PATH
    "${PASSTHROUGH_ARGS[@]}"
)

# Unbuffered output for real-time log streaming
export PYTHONUNBUFFERED=1

# Normalize ONE_GPU_PER_PROCESS → "true"/"false" (empty = false). Reject other values.
case "${ONE_GPU_PER_PROCESS:-false}" in
    1|true)      ONE_GPU_PER_PROCESS=true  ;;
    0|false|"")  ONE_GPU_PER_PROCESS=false ;;
    *) echo "[ERROR] ONE_GPU_PER_PROCESS='${ONE_GPU_PER_PROCESS}' must be true/false." >&2; exit 1 ;;
esac
export ONE_GPU_PER_PROCESS

# In 1-GPU-per-process mode, precompute LOCAL_WORLD_SIZE / NPROCS so both the
# direct fan-out (below) and the Ray fan-out (_ray_actor.py) see them.
if [[ "${ONE_GPU_PER_PROCESS}" == "true" ]]; then
    export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-$(python3 -c "import hip; print(hip.hipGetDeviceCount()[1])" 2>/dev/null || echo 8)}"
    export NPROCS=$(( NNODES * LOCAL_WORLD_SIZE ))
fi

# ---- Optional: wrap Python launch with rocprofv3 ---------------------------
# Enable via `_env_ROCPROF_TRACE=1` CLI passthrough. Optional tuning:
#   _env_ROCPROF_OUTDIR=<path>     (default: $OUTPUT_PATH/rocprof)
#   _env_ROCPROF_DELAY=<sec>       (default: 0   = start at t=0)
#   _env_ROCPROF_DURATION=<sec>    (default: 0   = treated as "whole run", expanded to 999999s)
#   _env_ROCPROF_TRACES=<csv>      (default: kernel,hip,rccl,marker -> --<X>-trace each)
# Per-node, per-rank traces at $ROCPROF_OUTDIR/<host>/<SLURM_PROCID>/.
# Not supported when ONE_GPU_PER_PROCESS=true (per-rank output collision).
#
# CRITICAL: rocprofv3 v1.0.0 in rocm/jax-training:maxtext-v26.2 has TWO known quirks:
#   1. Without `--collection-period`, only HIP compiler-side events are captured
#      (no kernel_trace.csv / rccl_api_trace.csv / marker_api_trace.csv). The
#      wrapper ALWAYS passes `--collection-period` to dodge this code path.
#   2. `--sys-trace` (catch-all) and `--stats --summary` SUPPRESS per-domain trace
#      CSV output — so we use explicit per-domain `--<X>-trace` flags instead.
# Empirically validated by comparing job 13526 (working: per-domain + period)
# vs 14863 (broken: --sys-trace + --stats + period): only the former produced
# kernel_trace.csv / rccl_api_trace.csv / marker_api_trace.csv.
PROF_CMD=()
if [[ "${ROCPROF_TRACE:-0}" == "1" ]]; then
    if [[ "${ONE_GPU_PER_PROCESS}" == "true" ]]; then
        echo "[rocprofv3] ERROR: ROCPROF_TRACE=1 is not supported with ONE_GPU_PER_PROCESS=true" >&2
        exit 1
    fi
    ROCPROF_OUTDIR="${ROCPROF_OUTDIR:-$OUTPUT_PATH/rocprof}"
    ROCPROF_DELAY="${ROCPROF_DELAY:-0}"
    ROCPROF_DURATION="${ROCPROF_DURATION:-0}"
    ROCPROF_TRACES="${ROCPROF_TRACES:-kernel,hip,rccl,marker}"
    mkdir -p -v "$ROCPROF_OUTDIR"
    chmod a+w "$ROCPROF_OUTDIR" 2>/dev/null || true
    PROF_CMD=(rocprofv3
        --output-format pftrace csv
        --output-directory "${ROCPROF_OUTDIR}/%hostname%/%env{SLURM_PROCID}%"
    )
    _eff_duration="$ROCPROF_DURATION"
    [[ "$_eff_duration" == "0" ]] && _eff_duration=999999
    PROF_CMD+=(--collection-period "${ROCPROF_DELAY}:${_eff_duration}:1")
    unset _eff_duration
    IFS=',' read -ra _TRACE_KINDS <<< "$ROCPROF_TRACES"
    for kind in "${_TRACE_KINDS[@]}"; do
        PROF_CMD+=("--${kind}-trace")
    done
    unset _TRACE_KINDS
    PROF_CMD+=(--)
    echo "[rocprofv3] wrapping python launch"
    echo "[rocprofv3]   outdir    = $ROCPROF_OUTDIR"
    echo "[rocprofv3]   delay     = ${ROCPROF_DELAY}s"
    echo "[rocprofv3]   duration  = ${ROCPROF_DURATION}s (0 = whole run)"
    echo "[rocprofv3]   traces    = $ROCPROF_TRACES"
    echo "[rocprofv3]   full cmd  = ${PROF_CMD[*]}"
fi

if [[ "${USE_RAY:-false}" == "true" ]]; then
    # Ray Actor Mode: actor launches training in a subprocess (no GIL contention)
    # Enables: GPU monitoring, flame graphs via py-spy --subprocesses
    echo "Launching via Ray actor..."
    export RAY_DEDUP_LOGS=0
    "${PROF_CMD[@]}" python3 -u "$SCRIPT_DIR/_ray_actor.py" "${TRAIN_ARGS[@]}"
elif [[ "${ONE_GPU_PER_PROCESS}" == "true" ]]; then
    # Multi-process mode: 1 GPU per JAX process.
    # Launches LOCAL_WORLD_SIZE processes, each assigned 1 GPU via JAX local_device_ids.
    echo "Launching $LOCAL_WORLD_SIZE processes per node ($NPROCS total) for 1-GPU-per-process mode..."
    PIDS=()
    for (( i=0; i<LOCAL_WORLD_SIZE; i++ )); do
        LOCAL_RANK=$i \
        GLOBAL_RANK=$(( NODE_RANK * LOCAL_WORLD_SIZE + i )) \
        python3 -u "$SCRIPT_DIR/utils/monkey_patch_maxtext.py" "${TRAIN_ARGS[@]}" &
        PIDS+=($!)
    done
    _exit_code=0
    for pid in "${PIDS[@]}"; do
        wait "$pid" || { _rc=$?; [[ $_exit_code -eq 0 ]] && _exit_code=$_rc; }
    done
    exit "$_exit_code"
else
    # Direct Mode (with MFU tracking)
    echo "Launching MaxText.train directly..."
    "${PROF_CMD[@]}" python3 -u "$SCRIPT_DIR/utils/monkey_patch_maxtext.py" "${TRAIN_ARGS[@]}"
fi
