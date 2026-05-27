#!/bin/bash
# Ray cluster management utilities

_RAY_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_RAY_SCRIPT_DIR/utils/job_dir.sh"
# detect_cluster_ip is required by start_ray_head / start_ray_worker for the
# `--node-ip-address` private-bind security fix.  Source at the top so the
# function is in scope regardless of which entry point reaches us.
source "$_RAY_SCRIPT_DIR/utils/detect_ip.sh"

export RAY_LOG_TO_STDERR=0
export RAY_BACKEND_LOG_LEVEL=fatal
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# ---- Performance tuning: reduce Ray worker-thread overhead ----
# Training runs in a subprocess (no GIL sharing with Ray), but Ray's internal
# threads (metrics pusher, health checks, event stats) still consume CPU on the
# node.  Throttle reporting frequencies to minimise background overhead.
export RAY_metrics_report_interval_ms=${RAY_metrics_report_interval_ms:-300000}   # 5 min (default 10s)
export RAY_event_stats_print_interval_ms=${RAY_event_stats_print_interval_ms:-0}  # disable event stats
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-30000}            # 30s (default ~10s)
export RAY_num_heartbeats_timeout=${RAY_num_heartbeats_timeout:-20}               # tolerate slower heartbeats

# RAY_PORT must be set by the caller (run_setup.sh, _job.sbatch, or
# _container.sh --env).  No default — prevents silent fallback to a
# potentially occupied port.
RAY_HEAD_IP="${JAX_COORDINATOR_IP:-localhost}"
# Allow the caller to pre-set these via env (e.g., when running multiple jobs
# on one host or to avoid a port collision); fall back to defaults otherwise.
RAY_METRICS_PORT="${RAY_METRICS_PORT:-55080}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9190}"

# Persist Ray logs to the shared job output directory so they survive crashes.
# Inside Docker, this is usually /outputs; in direct-in-container runs it may
# be another base (e.g. JOB_WORKSPACE).
# NOTE: We do NOT use --temp-dir for persistence because long paths exceed the
# 108-char Unix socket limit (sockaddr_un).  Instead, Ray uses the default
# /tmp/ray temp dir and we symlink its log directory to persistent storage.
RAY_LOG_DIR="$(resolve_outputs_base_dir)/${JOB_DIR:-unknown}/ray_logs/$(hostname -s 2>/dev/null || hostname)"

# After ray start, redirect its log directory to persistent storage via symlink.
_persist_ray_logs() {
    local session_dir
    session_dir=$(ls -td /tmp/ray/session_* 2>/dev/null | head -1)
    if [[ -n "$session_dir" && -d "$session_dir/logs" ]]; then
        mkdir -p "$RAY_LOG_DIR"
        cp -a "$session_dir/logs/"* "$RAY_LOG_DIR/" 2>/dev/null || true
        rm -rf "$session_dir/logs"
        ln -sfn "$RAY_LOG_DIR" "$session_dir/logs"
        echo "[Ray] Logs -> ${RAY_LOG_DIR} (persistent, symlinked from $session_dir/logs)"
    fi
}

export RAY_PROMETHEUS_HOST="http://localhost:${PROMETHEUS_PORT}"

# Generic watchdog: launch a command in a background subshell and restart on crash.
# Usage: _run_with_watchdog <label> <log_file> [--on-restart <cmd>] -- <command> [args...]
#   - The command is launched INSIDE the subshell so wait() always operates on a
#     direct child (launching outside would make wait() return 127 immediately).
#   - On non-zero exit: log, run on-restart hook, sleep 5s, restart.
#   - On clean exit (0) or signal death (SIGTERM=143, SIGINT=130): stop.
#   - Gives up after 50 restarts.
_run_with_watchdog() {
    local label="$1"; shift
    local log_file="$1"; shift
    local on_restart=""
    if [[ "${1:-}" == "--on-restart" ]]; then
        shift; on_restart="$1"; shift
    fi
    [[ "${1:-}" == "--" ]] && shift

    (
        # Close inherited stdout/stderr so this subshell does not hold open
        # any parent pipe (e.g. the tee pipeline in in_container_run.sh).
        # All watchdog output goes to $log_file explicitly.
        exec >/dev/null 2>&1

        local restarts=0
        while true; do
            "$@" >>"$log_file" 2>&1 &
            local pid=$!
            wait "$pid" 2>/dev/null
            local exit_code=$?
            # exit 0 = clean shutdown; 143 = SIGTERM; 130 = SIGINT (job teardown)
            [[ $exit_code -eq 0 || $exit_code -eq 143 || $exit_code -eq 130 ]] && break
            restarts=$((restarts + 1))
            if [[ $restarts -gt 50 ]]; then
                echo "[$(date -u +%FT%TZ)] [$label] Watchdog: too many restarts (50), giving up" >>"$log_file"
                break
            fi
            echo "[$(date -u +%FT%TZ)] [$label] Watchdog: exited with code $exit_code, restart #$restarts" >>"$log_file"
            [[ -n "$on_restart" ]] && eval "$on_restart"
            sleep 5
        done
    ) &
}

# Prometheus helpers (install_prometheus, start_prometheus, view_prometheus, …)
source "$(dirname "${BASH_SOURCE[0]}")/prometheus.sh"

# Node metrics exporter (GPU + host plugins)
_METRICS_EXPORTER_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/metrics_exporter.sh"

start_metrics_exporter() {
    if [[ -x "$_METRICS_EXPORTER_SCRIPT" ]]; then
        local log_dir="$(resolve_outputs_base_dir)/${JOB_DIR:-unknown}/metrics_exporter"
        local log_file="$log_dir/$(hostname -s).log"
        mkdir -p "$log_dir" 2>/dev/null || log_file="/tmp/metrics_exporter.log"
        _run_with_watchdog "Metrics Exporter" "$log_file" -- \
            "$_METRICS_EXPORTER_SCRIPT" --port "${EXPORTER_PORT:-9400}" --interval 10
        echo "[Metrics Exporter] Started on port ${EXPORTER_PORT:-9400} on $(hostname -s) (log: $log_file)"
    else
        echo "[Metrics Exporter] Script not found: $_METRICS_EXPORTER_SCRIPT" >&2
    fi
}

# ============================================================================
# Submission helpers (used by submit.sh)
# ============================================================================

# Build the sbatch --export flag for Ray jobs.
# Prints the flag string to stdout.
build_ray_export() {
    source "$(dirname "${BASH_SOURCE[0]}")/detect_ip.sh"
    local public_ip
    public_ip="$(detect_ip)"
    local login_hostname
    login_hostname=$(hostname -f 2>/dev/null || hostname)
    echo "--export=ALL,USE_RAY=true,LOGIN_NODE_HOSTNAME=${USER}@${login_hostname},LOGIN_NODE_IP=${USER}@${public_ip}"
}

# ============================================================================
# Ray cluster management
# ============================================================================

install_ray() {
    python3 -c "import ray" 2>/dev/null || pip3 install -q "ray[default]>=2.9.0"
    # py-spy enables stack traces & flame graphs from the Ray dashboard.
    # Install alongside Ray so the dependency is explicit.
    command -v py-spy &>/dev/null || pip3 install -q py-spy
}

start_ray_head() {
    # SECURITY (dashboard): bind the dashboard / job-submission HTTP
    # endpoint to localhost only.  Ray's `JobSubmissionClient` HTTP API is
    # unauthenticated; binding to 0.0.0.0 on a host with a public IP lets
    # an internet scanner submit code that runs as root inside the
    # container.
    #
    # SECURITY (GCS): bind the GCS server (RAY_PORT) to the cluster-
    # private subnet only.  GCS defaults to binding all interfaces
    # (0.0.0.0); on a public IP, internet scanners can join the cluster as
    # raylets within seconds of head startup and execute code in the
    # actor pool.  `--node-ip-address=<private>` restricts the GCS bind to
    # the private IP only — workers reach the head over the private
    # subnet, internet scanners cannot connect.
    local cluster_ip
    cluster_ip=$(detect_cluster_ip)
    echo "[Ray] Starting HEAD on $(hostname):${RAY_PORT} (bind=${cluster_ip})"
    if ! ray start --head --port=$RAY_PORT \
        --num-cpus=1 \
        --node-ip-address="$cluster_ip" \
        --dashboard-host=127.0.0.1 --dashboard-port=8265 \
        --metrics-export-port=$RAY_METRICS_PORT \
        --disable-usage-stats 2>&1; then
        echo "[Ray] HEAD failed to start (port $RAY_PORT, dashboard 8265, bind $cluster_ip)" >&2
        return 1
    fi
    _persist_ray_logs

    for _ in {1..30}; do ray status &>/dev/null && return 0; sleep 2; done
    echo "[Ray] HEAD timeout" >&2
    return 1
}

start_ray_worker() {
    # SECURITY: bind the worker raylet to the cluster-private IP (matches
    # the head's GCS bind in start_ray_head).  Workers must connect to the
    # head over the private subnet, never the public network.
    local cluster_ip
    cluster_ip=$(detect_cluster_ip)
    echo "[Ray] Starting WORKER -> ${RAY_HEAD_IP}:${RAY_PORT} (bind=${cluster_ip})"
    sleep 5
    for i in {1..18}; do
        ray start --address="${RAY_HEAD_IP}:${RAY_PORT}" \
            --num-cpus=1 \
            --node-ip-address="$cluster_ip" \
            --no-monitor \
            --metrics-export-port=$RAY_METRICS_PORT \
            --disable-usage-stats &>/dev/null && {
            _persist_ray_logs
            return 0
        }
        echo "[Ray] Retry $i/18..."
        sleep 10
    done
    echo "[Ray] WORKER failed" >&2
    return 1
}

_install_pyspy_subprocess_wrapper() {
    # Wrap py-spy to redirect from the Ray worker PID to the training subprocess.
    #
    # The Ray Dashboard calls 'py-spy dump -p <worker_pid>', but the worker is
    # just sitting in p.wait() — the real training runs in a child process
    # (monkey_patch_maxtext.py).  Attaching py-spy to the worker's thread-heavy process
    # and traversing via --subprocesses is slow and unreliable (frequent 500s).
    #
    # This wrapper finds the worker's child PID and targets it directly, which
    # is faster and far more reliable.  Falls back to the original PID if no
    # child exists (training not yet started or already finished).
    local real_pyspy
    real_pyspy="$(command -v py-spy 2>/dev/null)" || return 0

    # Skip if already wrapped
    [[ -f "${real_pyspy}-real" ]] && return 0

    mv "$real_pyspy" "${real_pyspy}-real"
    cat > "$real_pyspy" <<'PYSPY_WRAPPER'
#!/bin/bash
REAL="$(dirname "$0")/$(basename "$0")-real"
if [[ "$1" == "dump" || "$1" == "record" ]]; then
    # Redirect to child process: the Ray Dashboard targets the Ray worker PID,
    # but the training code runs in a subprocess (monkey_patch_maxtext.py).  Targeting
    # the child directly is faster and far more reliable than traversing the
    # worker's thread-heavy process tree.
    #
    # In 1-GPU-per-process mode the actor has multiple children (one per GPU).
    # Set PYSPY_LOCAL_RANK=N to pick the Nth child (default: 0, i.e. rank 0).
    ARGS=("$@")
    TARGET_RANK="${PYSPY_LOCAL_RANK:-0}"
    for ((i=0; i<${#ARGS[@]}; i++)); do
        if [[ "${ARGS[$i]}" == "-p" && -n "${ARGS[$((i+1))]}" ]]; then
            # sed -n "Np" picks the Nth line (1-indexed), so add 1.
            CHILD=$(pgrep -P "${ARGS[$((i+1))]}" 2>/dev/null | sed -n "$((TARGET_RANK+1))p")
            [[ -n "$CHILD" ]] && ARGS[$((i+1))]="$CHILD"
            break
        fi
    done
    exec "$REAL" "${ARGS[@]}" --subprocesses
else
    exec "$REAL" "$@"
fi
PYSPY_WRAPPER
    chmod +x "$real_pyspy"
    echo "[py-spy] Wrapper installed: dump/record will redirect to training subprocess"
}

start_ray_cluster() {
    install_ray || return 1
    _install_pyspy_subprocess_wrapper
    ray stop --force &>/dev/null || true
    # Kill stale Prometheus from a prior run of THIS job (match the TSDB path).
    # Avoid 'pkill -f prometheus' which would kill unrelated instances (e.g.,
    # cluster-level monitors or other jobs' Prometheus).
    if [[ -n "${JOB_DIR:-}" ]]; then
        pkill -f "prometheus.*${JOB_DIR}/prometheus" &>/dev/null || true
    else
        pkill -f "prometheus.*storage.tsdb" &>/dev/null || true
    fi
    pkill -f "metrics_exporter.sh" &>/dev/null || true
    pkill -f "node_hw_metrics.prom" &>/dev/null || true
    # Wait for ports to be released after SIGKILL
    sleep 2

    # Node metrics exporter runs on EVERY node (not just head)
    start_metrics_exporter

    if [[ ${NODE_RANK:-0} -eq 0 ]]; then
        start_ray_head || return 1
        start_prometheus
        # Re-export in case start_prometheus changed PROMETHEUS_PORT due to conflict
        export RAY_PROMETHEUS_HOST="http://localhost:${PROMETHEUS_PORT}"
    else
        start_ray_worker
    fi
}

stop_ray_cluster() {
    ray stop --force &>/dev/null || true
    if [[ -n "${JOB_DIR:-}" ]]; then
        pkill -f "prometheus.*${JOB_DIR}/prometheus" &>/dev/null || true
    else
        pkill -f "prometheus.*storage.tsdb" &>/dev/null || true
    fi
    pkill -f "metrics_exporter.sh" &>/dev/null || true
    pkill -f "node_hw_metrics.prom" &>/dev/null || true
    pkill -f "tensorboard" &>/dev/null || true
}

# ============================================================================
# TensorBoard
# ============================================================================

start_tensorboard() {
    # OUTPUT_PATH is exported by _train_with_ray.sh via resolve_output_path().
    local output_dir="${OUTPUT_PATH:?OUTPUT_PATH not set}"
    mkdir -p "$output_dir"
    # SECURITY: bind to localhost (not --bind_all). Tunnel via SSH-J
    # ProxyJump per print_ray_info().  TB serves logs only — strictly
    # less dangerous than Ray's job-submission HTTP, but still leaks model /
    # run names to anyone scanning the cluster's public IPs.
    tensorboard --logdir="$output_dir" --port=6006 --host=127.0.0.1 &>/dev/null &
    echo "[TensorBoard] Started on port 6006 (logdir: $output_dir)"
}

# ============================================================================
# Info display
# ============================================================================

print_ray_info() {
    local host=$(hostname -s)
    local login_hostname="${LOGIN_NODE_HOSTNAME:-user-unknown@hostname-unknown}"
    local login_ip="${LOGIN_NODE_IP:-user-unknown@ip-unknown}"
    cat <<EOF
==============================================
SSH tunnel from your local machine (Ray dashboard is bound to the head's
localhost for security; use ProxyJump through the login node):
  ssh -J ${login_hostname} -L 8265:localhost:8265 -L 6006:localhost:6006 -L ${PROMETHEUS_PORT}:localhost:${PROMETHEUS_PORT} root@${host}
  ssh -J ${login_ip}       -L 8265:localhost:8265 -L 6006:localhost:6006 -L ${PROMETHEUS_PORT}:localhost:${PROMETHEUS_PORT} root@${host}

Then open in your browser:
  Ray Dashboard:  http://localhost:8265
  TensorBoard:    http://localhost:6006
  Prometheus:     http://localhost:${PROMETHEUS_PORT}

Prometheus TSDB and Ray logs write directly to persistent storage (survive job termination).
To inspect after the job ends:
  utils/prometheus.sh list
  utils/prometheus.sh view <JOB_WORKSPACE>/<job>/prometheus

Ray logs are persisted per node under:
  <JOB_WORKSPACE>/<job>/ray_logs/<hostname>/session_*/logs/

Tip: Port conflict when monitoring multiple jobs?
  Change local port: -L 18265:... -> localhost:18265

Debug (get a shell on a job node, then inspect Ray):
  ray list actors / ray status
==============================================
EOF
}
