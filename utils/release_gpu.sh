#!/bin/bash

# release_gpu.sh - Stop containers and kill GPU processes with retries.
#
# Modes:
#   Full (no args):             Stop ALL running containers, kill GPU processes,
#                               wait for GPU memory release.  Used by preflight.
#   Targeted (--container NAME): Kill a specific container immediately (SIGKILL).
#                               Used by scancel trap (tight KillWait budget).
#
# Device-agnostic: detects AMD (rocm-smi) or NVIDIA (nvidia-smi) GPUs.

# ── Argument parsing ─────────────────────────────────────────────────────────

TARGET_CONTAINER=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --container) TARGET_CONTAINER="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── Docker binary ────────────────────────────────────────────────────────────
source "$(dirname "${BASH_SOURCE[0]}")/docker_utils.sh"
DOCKER_BIN="$(get_docker_bin)" || { echo "ERROR: no docker/podman found" >&2; exit 1; }
read -ra DOCKER_CMD <<< "$DOCKER_BIN"

# ── Configuration ────────────────────────────────────────────────────────────

STOP_TIMEOUT="${STOP_TIMEOUT:-20}"  # docker stop grace period before SIGKILL
SETTLE_WAIT="${SETTLE_WAIT:-40}"    # post-kill wait for KFD/GPU memory release
KILL_WAIT=5                         # gap between SIGTERM and SIGKILL for GPU PIDs
POLL_INTERVAL=20                    # gap between GPU-free polls
MAX_RETRIES="${MAX_RETRIES:-10}"

# ── Helpers ──────────────────────────────────────────────────────────────────

_get_gpu_pids() {
    local pids=""
    if command -v rocm-smi &>/dev/null; then
        pids+=" $(rocm-smi --showpids 2>/dev/null | grep -oE '^[0-9]+' | grep -v '^$')"
    fi
    if command -v nvidia-smi &>/dev/null; then
        pids+=" $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
    fi
    # Filter out system/scheduler PIDs that show up in rocm-smi but are NOT
    # stale training processes:
    #   - slurmstepd  : this srun's own scheduler stub (has cgroup attachment to
    #                   /dev/kfd, so rocm-smi reports it; VRAM USED = 0).
    #                   Killing it tears down the running srun task and leaves
    #                   the parent sbatch hung waiting for srun.
    #   - gpuagent    : AMD GPU agent system daemon, holds device handles but
    #                   no training memory.
    # Both are root-owned; with passwordless sudo, the kill loop below would
    # actually succeed and kill the running task, so we must filter here.
    local filtered=""
    for pid in $pids; do
        [[ -z "$pid" ]] && continue
        local comm
        comm=$(ps -p "$pid" -o comm= 2>/dev/null | tr -d ' ')
        case "$comm" in
            slurmstepd|gpuagent) continue ;;
            *) filtered+=" $pid" ;;
        esac
    done
    echo "$filtered" | xargs
}

_is_zombie() {
    [[ "$(ps -p "$1" -o stat= 2>/dev/null)" == Z* ]]
}

_kill_zombie_parent() {
    # Kill the parent of a zombie so init adopts and reaps it, releasing
    # GPU memory held by the KFD driver.
    local pid="$1" ppid
    ppid=$(ps -p "$pid" -o ppid= 2>/dev/null | tr -d ' ')
    if [[ -n "$ppid" && "$ppid" != "1" && "$ppid" != "0" ]]; then
        echo "  PID $pid is zombie (parent=$ppid) — killing parent"
        kill -9 "$ppid" 2>/dev/null || sudo kill -9 "$ppid" 2>/dev/null || true
    elif [[ "$ppid" == "1" ]]; then
        echo "  PID $pid is zombie (parent=init) — waiting for reap"
    fi
}

# Force-kill all processes inside a container.
#
# Fallback when docker kill/stop fail because GPU processes are stuck in
# D-state (uninterruptible I/O).  Killing non-D-state processes (Python,
# Ray, shell) unblocks stuck NCCL collectives, letting D-state processes exit.
#
# Method 1: docker exec kill -9 -1  (runs as root via Docker daemon)
# Method 2: host-level cgroup kill  (requires root or sudo)
_force_kill_container_procs() {
    local container="$1"
    local cpid ccgroup _cgroup_procs

    cpid=$("${DOCKER_CMD[@]}" inspect --format '{{.State.Pid}}' "$container" 2>/dev/null) || cpid=""
    [[ -z "$cpid" || "$cpid" == "0" || ! -d "/proc/$cpid" ]] && return 1

    # Method 1: docker exec (runs via Docker daemon — root, no sudo needed).
    # Timeout guards against exec hanging on a stuck container runtime.
    if timeout 5 "${DOCKER_CMD[@]}" exec "$container" kill -9 -1 2>/dev/null; then
        echo "  [exec] Killed all processes in $container via docker exec"
        return 0
    fi
    echo "  [exec] docker exec failed for $container; falling back to host-level kill"

    # Method 2: cgroup kill (host-level, needs root/sudo).
    ccgroup=$(grep -oP '0::\K.*' "/proc/$cpid/cgroup" 2>/dev/null || true)
    _cgroup_procs="/sys/fs/cgroup${ccgroup}/cgroup.procs"

    if [[ -n "$ccgroup" && -f "$_cgroup_procs" ]]; then
        local _pcount=0 _kcount=0
        echo "  [cgroup] Force-killing all processes in $container via $_cgroup_procs"
        while IFS= read -r _p; do
            _pcount=$((_pcount + 1))
            if kill -9 "$_p" 2>/dev/null || sudo kill -9 "$_p" 2>/dev/null; then
                _kcount=$((_kcount + 1))
            fi
        done < "$_cgroup_procs"
        echo "  [cgroup] Sent SIGKILL to $_kcount/$_pcount processes"
    else
        echo "  [cgroup] Unavailable for $container (PID $cpid); killing process tree"
        pkill -9 -P "$cpid" 2>/dev/null || sudo pkill -9 -P "$cpid" 2>/dev/null || true
        kill -9 "$cpid" 2>/dev/null || sudo kill -9 "$cpid" 2>/dev/null || true
    fi
    return 0
}

# ── GPU status on exit (full mode only) ──────────────────────────────────────

if [[ -z "$TARGET_CONTAINER" ]]; then
    _show_gpu_status() {
        echo "--- GPU status after cleanup ---"
        command -v rocm-smi  &>/dev/null && { rocm-smi --showpids 2>&1 || true; }
        command -v amd-smi   &>/dev/null && { amd-smi 2>&1 || true; }
        command -v nvidia-smi &>/dev/null && { nvidia-smi 2>&1 || true; }
    }
    trap _show_gpu_status EXIT
fi

# =============================================================================
# Targeted mode: kill a specific container (scancel trap)
# =============================================================================
# Go straight to SIGKILL (docker kill) — GPU processes ignore SIGTERM.
# If docker kill fails, the container is likely a "ghost": containerd already
# killed the processes (cgroup.procs is empty) but Docker never received the
# exit event.  Force-kill via cgroup as a best-effort, then use docker stop
# to nudge Docker's state machine into recognizing the container is dead.
# GPU memory release is the next job's preflight responsibility.
#
# Budget: kill (≤5s) + cgroup (~1s) + stop (≤8s) + rm (~1s)
#       ≈ 15s, within typical scheduler kill grace periods (e.g. Slurm KillWait=30s).

if [[ -n "$TARGET_CONTAINER" ]]; then
    # Cap docker kill — its internal wait for exit events can burn 10+ seconds.
    if timeout 5 "${DOCKER_CMD[@]}" kill "$TARGET_CONTAINER" 2>&1; then
        "${DOCKER_CMD[@]}" rm -f "$TARGET_CONTAINER" 2>&1 || true
        exit 0
    fi

    # docker kill failed ("did not receive an exit event").  Typically the
    # processes are already dead (cgroup.procs empty) — Docker just lost
    # the exit event from containerd.  Cgroup kill as best-effort, then
    # docker stop to trigger Docker's state transition.
    echo "[WARN] docker kill failed for $TARGET_CONTAINER; attempting force-kill"
    _force_kill_container_procs "$TARGET_CONTAINER" || true
    timeout 8 "${DOCKER_CMD[@]}" stop -t 2 "$TARGET_CONTAINER" 2>&1 || true
    "${DOCKER_CMD[@]}" rm -f "$TARGET_CONTAINER" 2>&1 || true
    exit 0
fi

# =============================================================================
# Full mode: stop all containers + clean GPU processes (preflight)
# =============================================================================

echo "=== Pre-flight GPU cleanup ==="

START_TIME=$(date +%s)

# ── Phase 0: Stop all running containers ─────────────────────────────────────
# Nodes are --exclusive, so any running container is stale from a prior job.
# This catches training containers (GPU) AND coordination containers (Ray/CPU)
# that the GPU-PID check alone would miss.
#
# Loop: docker stop → cgroup kill survivors → settle → check.
# The cgroup kill BEFORE settling is critical: docker stop sends SIGTERM then
# SIGKILL to PID 1, but the container can't fully stop while D-state children
# remain in its PID namespace.  Cgroup-killing those children unblocks the
# teardown, and the settle wait gives the runtime + KFD driver time to release
# GPU memory.

mapfile -t STALE < <("${DOCKER_CMD[@]}" ps -q 2>/dev/null)
if [[ ${#STALE[@]} -gt 0 && -n "${STALE[0]}" ]]; then
    echo "[container] Stopping ${#STALE[@]} container(s): ${STALE[*]}"

    for ((retry = 0; retry <= MAX_RETRIES; retry++)); do
        # Graceful stop → SIGTERM, then SIGKILL after STOP_TIMEOUT.
        "${DOCKER_CMD[@]}" stop -t "$STOP_TIMEOUT" "${STALE[@]}" 2>/dev/null || true

        # Quick check: if docker stop worked, skip the heavy path.
        remaining=$("${DOCKER_CMD[@]}" ps -q 2>/dev/null)
        if [[ -z "$remaining" ]]; then
            echo "[container] All containers stopped"
            break
        fi

        # Containers survived SIGKILL → GPU processes likely in D-state.
        # Force-kill via cgroup to break the deadlock, THEN settle.
        ((retry > 0)) && echo "[container] Retry $retry/$MAX_RETRIES — still running: $remaining"
        for cid in $remaining; do
            _force_kill_container_procs "$cid" || true
        done

        # Settle: give the container runtime + KFD driver time to tear down
        # PID namespaces and release GPU memory after process death.
        sleep "$SETTLE_WAIT"

        remaining=$("${DOCKER_CMD[@]}" ps -q 2>/dev/null)
        if [[ -z "$remaining" ]]; then
            echo "[container] All containers stopped"
            break
        fi

        if ((retry >= MAX_RETRIES)); then
            echo "[container] WARNING: still running after $MAX_RETRIES retries: $remaining"
            break
        fi

        mapfile -t STALE <<< "$remaining"
    done
fi

# ── Phase 1: Detect remaining GPU processes ──────────────────────────────────

GPU_PIDS=$(_get_gpu_pids)
if [[ -z "$GPU_PIDS" ]]; then
    echo "No GPU processes found - GPUs are free"
    exit 0
fi

echo "Found GPU processes after container stop: $GPU_PIDS"

# ── Phase 2: Kill remaining GPU processes ────────────────────────────────────
# Zombies need parent-kill; live processes get SIGTERM → SIGKILL.

for PID in $GPU_PIDS; do
    if _is_zombie "$PID"; then
        _kill_zombie_parent "$PID"
    else
        kill -TERM "$PID" 2>/dev/null || sudo kill -TERM "$PID" 2>/dev/null || true
    fi
done

sleep "$KILL_WAIT"

GPU_PIDS=$(_get_gpu_pids)
if [[ -n "$GPU_PIDS" ]]; then
    echo "SIGTERM didn't clear all processes, sending SIGKILL: $GPU_PIDS"
    for PID in $GPU_PIDS; do
        if _is_zombie "$PID"; then
            _kill_zombie_parent "$PID"
        else
            kill -9 "$PID" 2>/dev/null || sudo kill -9 "$PID" 2>/dev/null || true
        fi
    done
fi

# ── Phase 3: Wait for GPUs to be truly freed ─────────────────────────────────

echo "Waiting for GPUs to be freed..."
RETRY_COUNT=0
while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
    sleep "$POLL_INTERVAL"
    REMAINING=$(_get_gpu_pids)

    if [[ -z "$REMAINING" ]]; then
        END_TIME=$(date +%s)
        echo "GPU cleanup successful - GPUs are now free (elapsed: $((END_TIME - START_TIME))s)"
        exit 0
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "  Retry $RETRY_COUNT/$MAX_RETRIES - GPUs still occupied: $REMAINING"

    for PID in $REMAINING; do
        if _is_zombie "$PID"; then
            _kill_zombie_parent "$PID"
        fi
    done
done

# ── Cleanup failed ───────────────────────────────────────────────────────────

END_TIME=$(date +%s)
echo "ERROR: Failed to free GPUs (elapsed: $((END_TIME - START_TIME))s)"
echo "Remaining GPU processes:"
for PID in $REMAINING; do
    echo "  PID $PID:"
    ps -p "$PID" -o pid,user,stat,ppid,cmd 2>/dev/null || echo "    (process info unavailable)"
done

exit 1
