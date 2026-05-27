#!/bin/bash
##SBATCH --nodes=8
#SBATCH --exclusive             # exclusive node access
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # all mem avail
#SBATCH --ntasks-per-node=1     # n tasks per machine (one task per gpu)
#SBATCH --overcommit
##SBATCH --dependency=singleton # only run one instance at a time

# Resolve script directory (MAXTEXT_SLURM_DIR is exported by submit.sh;
# fall back to current directory for direct sbatch submission from the script dir)
SCRIPT_DIR="${MAXTEXT_SLURM_DIR:-.}"
_JOB_START=$SECONDS

# ---- Map Slurm variables to scheduler-agnostic names ----
# _container.sh and all downstream scripts (training, observability, utilities)
# use only these generic names.  This mapping confines Slurm awareness to the
# orchestration tier (_job.sbatch + submit.sh), so adding a new scheduler
# (e.g. Kubernetes) requires only a new orchestration entry point that sets
# the same generic env vars — everything below runs unmodified.
export JOB_ID="$SLURM_JOB_ID"
export JOB_NAME="$SLURM_JOB_NAME"
export NNODES="$SLURM_JOB_NUM_NODES"
# NODE_RANK and JAX_COORDINATOR_IP are per-task — mapped in NODE_REPORT below.
# SLURM_NODEID and SLURM_LAUNCH_NODE_IPADDR are set by srun, not sbatch.
export JOB_NODELIST="$SLURM_JOB_NODELIST"
export LOGIN_NODE_HOSTNAME="${LOGIN_NODE_HOSTNAME:-${USER}@${SLURM_SUBMIT_HOST}}"
export LOGIN_NODE_IP="${LOGIN_NODE_IP:-${USER}@${SLURM_SUBMIT_HOST}}"
export NODELIST_EXPANDED=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | tr '\n' ',' | sed 's/,$//')

source "$SCRIPT_DIR/utils/pick_port.sh"
source "$SCRIPT_DIR/utils/detect_ip.sh"

JAX_PORT=$(pick_free_port) || { echo "FATAL: Could not find a free port for JAX coordinator" >&2; exit 1; }

# Exported so srun --export=ALL propagates it to every node; only the head
# node binds this port, but workers need the value to connect to it.
if [[ "${USE_RAY:-false}" == "true" ]]; then
    RAY_PORT=$(pick_free_port "$JAX_PORT") || { echo "FATAL: Could not find a free port for Ray" >&2; exit 1; }
    export RAY_PORT
fi

# SECURITY: pin the JAX coordinator to this node's cluster-private IP
# instead of letting srun fall back to SLURM_LAUNCH_NODE_IPADDR (which is
# the launching node's public IP).  Combined with `--node-ip-address` on
# `ray start` (utils/ray_cluster.sh), this keeps every cross-node service
# bound to the private subnet only.  Without this, the JAX coordinator
# would bind on the public IP and become reachable from the internet (an
# unauthenticated coordinator is itself an attack surface).
export JAX_COORDINATOR_IP="${JAX_COORDINATOR_IP:-$(detect_cluster_ip)}"

# Split args first
source "$SCRIPT_DIR/utils/split_script_args.sh"
split_script_args "$@"

# Parse from SCRIPT_ARGS
MODEL_NAME=${SCRIPT_ARGS[0]:?model_name is required}
EXP_TAG=${SCRIPT_ARGS[1]:-}
MODEL_NAME_ALIAS=${SCRIPT_ARGS[2]:-}

# ---- Standardized log header (same key=value format as run_setup.sh) ----
# Parsers (analyze_job.py, tgs_tagger.py, perf_server.py) match these names.
echo "JOB_ID=$JOB_ID"
echo "JOB_NAME=$JOB_NAME"
echo "NNODES=$NNODES"
echo "JOB_NODELIST=$JOB_NODELIST"
echo "JAX_COORDINATOR_PORT=$JAX_PORT"
if [[ -n "${RAY_PORT:-}" ]]; then echo "RAY_PORT=$RAY_PORT"; fi
echo "PASSTHROUGH_ARGS=\"${PASSTHROUGH_ARGS[*]}\""
echo "MODEL_NAME=$MODEL_NAME"
if [[ -n "$MODEL_NAME_ALIAS" ]]; then echo "MODEL_NAME_ALIAS=$MODEL_NAME_ALIAS"; fi
if [[ -n "$EXP_TAG" ]]; then echo "EXP_TAG=$EXP_TAG"; fi

# Emit provenance once at job entry. Downstream node-level launchers inherit
# MAXTEXT_GIT_SUMMARY_EMITTED=1 and skip duplicate emission.
source "$SCRIPT_DIR/utils/code_provenance.sh"
emit_code_provenance "$SCRIPT_DIR"

# Stage timeouts — see utils/stage_timeout.sh for details.
# Override: STAGE_TIMEOUTS="preflight:600,pull:1800" ./submit.sh ...
source "$SCRIPT_DIR/utils/stage_timeout.sh"
stage_timeout_init preflight:900 pull:900 ecc:300 train:none

# Job summary — always printed (success or failure) via EXIT trap.
_fmt_elapsed() {
    local s=$1
    if (( s >= 3600 )); then printf '%dh %dm %ds' $((s/3600)) $((s%3600/60)) $((s%60))
    elif (( s >= 60 )); then printf '%dm %ds' $((s/60)) $((s%60))
    else printf '%ds' "$s"
    fi
}
_print_summary() {
    local rc=${_CANCEL_RC:-$?}
    echo "========================== JOB SUMMARY =========================="
    echo "  Job:    ${JOB_ID:-?} (${JOB_NAME:-?})"
    echo "  Model:  ${MODEL_NAME:-?}"
    echo "  Nodes:  ${NNODES:-?}"
    echo "  Wall:   $(_fmt_elapsed $(( SECONDS - _JOB_START )))"
    if [[ $rc -eq 0 ]]; then
        echo "  Status: $(tput setaf 2 2>/dev/null)SUCCESS (exit 0)$(tput sgr0 2>/dev/null)"
    else
        echo "  Status: $(tput setaf 1 2>/dev/null)FAILED (exit $rc) — ${_STAGE_FAILURE:-unknown}$(tput sgr0 2>/dev/null)"
    fi
    echo "================================================================="
}
trap _print_summary EXIT

# Explicit SIGTERM handler: when scancel sends SIGTERM, ensure we exit non-zero.
# Without this, SIGTERM arriving between stages leaves $?=0 from the last
# successful command, causing the EXIT trap to falsely report SUCCESS.
_handle_cancel() {
    _STAGE_FAILURE="CANCELLED (scancel / SIGTERM)"
    _CANCEL_RC=143
    exit 143
}
trap _handle_cancel TERM

# Wrapper: each srun task self-reports hostname + exit code on failure.
# The child is backgrounded + waited so that bash can catch SIGTERM (from
# scancel) immediately and forward it.  With a foreground child, bash queues
# signals until the child exits — but the child never exits because it never
# received the signal, leaving orphaned containers.
# NODE_RANK and JAX_COORDINATOR_IP are mapped here (per-task) because
# SLURM_NODEID and SLURM_LAUNCH_NODE_IPADDR are set by srun, not sbatch —
# unlike JOB_ID/NNODES which are batch-level exports above.
NODE_REPORT='export NODE_RANK=${SLURM_NODEID:-0}; export JAX_COORDINATOR_IP=${JAX_COORDINATOR_IP:-$SLURM_LAUNCH_NODE_IPADDR}; "$@" & _cpid=$!; trap "kill $_cpid 2>/dev/null" TERM INT; wait $_cpid 2>/dev/null; wait $_cpid 2>/dev/null; _rc=$?; [ $_rc -ne 0 ] && echo "NODE_EXIT host=$(hostname) exit=$_rc"; exit $_rc'

# Per-host pre-flight setup.
run_stage preflight "Preflight" \
    srun -l --kill-on-bad-exit=1 bash -c "$NODE_REPORT" _ bash "$SCRIPT_DIR/utils/preflight.sh"

# Pre-pull Docker image (barrier: all nodes ready before training).
run_stage pull "Docker pull" \
    srun -l --kill-on-bad-exit=1 bash -c "$NODE_REPORT" _ bash "$SCRIPT_DIR/_container.sh" pull-only

# Per-host ECC/UMC check.
run_stage ecc "ECC check" \
    srun -l --kill-on-bad-exit=1 bash -c "$NODE_REPORT" _ bash "$SCRIPT_DIR/utils/check_ecc.sh"

# Launch training on all nodes.
run_stage train "Training" \
    srun -l --kill-on-bad-exit=1 --export=ALL \
    bash -c "$NODE_REPORT" _ bash "$SCRIPT_DIR/_container.sh" "$JAX_PORT" "$MODEL_NAME" "$EXP_TAG" "$MODEL_NAME_ALIAS" -- "${PASSTHROUGH_ARGS[@]}"
