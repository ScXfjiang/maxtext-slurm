#!/bin/bash

# _k8s_job.sh — Per-pod entry point for Kubernetes training jobs.
#
# K8s equivalent of _job.sbatch. Handles coordinator discovery, barrier
# synchronization, and per-rank logging, then delegates to in_container_run.sh.
#
# Log layout:
#   outputs/<id>-<name>.log           ← rank 0 (primary, parsed by tools)
#   outputs/<id>-<name>/rank-1.log    ← rank 1 (for debugging)
#   outputs/<id>-<name>/rank-N.log    ← rank N
#
# Expected env vars (set by the k8s manifest / Kubernetes):
#   JOB_COMPLETION_INDEX  — pod index (set automatically by Kubernetes Indexed Jobs)
#   JOB_ID                — unique job identifier
#   JOB_NAME              — human-readable job name
#   NNODES                — total node count
#   JAX_COORDINATOR_PORT  — port for JAX coordinator
#   COORD_DIR             — shared-filesystem path for node IP exchange
#   JOB_WORKSPACE         — output directory root
#
# Usage: _k8s_job.sh <model_name> -- [passthrough_args...]
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_TIMEOUT="${K8S_BARRIER_TIMEOUT:-120}"

# -------- Derive rank from Indexed Job --------
export NODE_RANK="${JOB_COMPLETION_INDEX:?JOB_COMPLETION_INDEX not set — is this an Indexed Job?}"
# SECURITY: prefer the cluster-private RFC1918 IP over `hostname -I | awk
# '{print $1}'`, which on dual-IP hosts often returns the public IP first.
# A public-IP coordinator is itself an attack surface (cf. Ray GCS GCS
# attack on 2026-05-03 — see utils/ray_cluster.sh start_ray_head).
source "$SCRIPT_DIR/utils/detect_ip.sh"
HOST_IP="$(detect_cluster_ip)"
echo "[$NODE_RANK] Starting on $(hostname) ($HOST_IP)"

# -------- Coordinator discovery via shared filesystem --------
mkdir -p "$COORD_DIR"
echo "$HOST_IP" > "$COORD_DIR/rank-${NODE_RANK}"

_elapsed=0
while [[ ! -f "$COORD_DIR/rank-0" ]]; do
    if [[ $_elapsed -ge $_TIMEOUT ]]; then
        echo "FATAL: Coordinator (rank 0) did not register within ${_TIMEOUT}s" >&2
        exit 1
    fi
    sleep 1
    _elapsed=$((_elapsed + 1))
done

export JAX_COORDINATOR_IP=$(cat "$COORD_DIR/rank-0")
echo "[$NODE_RANK] Coordinator: $JAX_COORDINATOR_IP:${JAX_COORDINATOR_PORT}"

# -------- Barrier: wait for all pods to be ready --------
echo "ready" > "$COORD_DIR/ready-${NODE_RANK}"
_elapsed=0
while true; do
    if [[ $_elapsed -ge $_TIMEOUT ]]; then
        echo "FATAL: Barrier timed out after ${_TIMEOUT}s — not all $NNODES pods registered" >&2
        exit 1
    fi
    _all_ready=true
    for _rank in $(seq 0 $((NNODES-1))); do
        if [[ ! -f "$COORD_DIR/ready-${_rank}" ]]; then
            _all_ready=false
            break
        fi
    done
    $_all_ready && break
    sleep 1
    _elapsed=$((_elapsed + 1))
done
echo "[$NODE_RANK] All $NNODES nodes ready"

# -------- Dataset alias --------
source "$SCRIPT_DIR/container_env.sh"
if [[ -n "${DATASET_DIR:-}" && -d "$DATASET_DIR" && ! -e /datasets ]]; then
    ln -snf "$DATASET_DIR" /datasets
fi

# -------- Launch training --------
if [[ "$NODE_RANK" == "0" ]]; then
    exec "$SCRIPT_DIR/in_container_run.sh" "$@"
else
    source "$SCRIPT_DIR/utils/job_dir.sh"
    _JOB_DIR=$(make_job_dir "$JOB_ID" "$JOB_NAME")
    _RANK_LOG="$JOB_WORKSPACE/${_JOB_DIR}/rank-${NODE_RANK}.log"
    mkdir -p "$JOB_WORKSPACE/${_JOB_DIR}"
    export _IN_CONTAINER_LOG_TEE_ACTIVE=1
    export PYTHONUNBUFFERED=1
    "$SCRIPT_DIR/in_container_run.sh" "$@" > >(tee "$_RANK_LOG") 2>&1
fi
