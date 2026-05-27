#!/bin/bash
# Prometheus helpers — install, start (live scraping), and view (post-hoc).
#
# As a library (sourced by ray_cluster.sh):
#   source utils/prometheus.sh
#   install_prometheus
#   start_prometheus          # live scrape during a job
#
# As a standalone script:
#   utils/prometheus.sh view <data-dir> [-p PORT]
#   utils/prometheus.sh install
#   utils/prometheus.sh list [<workspace>]

PROMETHEUS_PORT=${PROMETHEUS_PORT:-9190}
RAY_METRICS_PORT=${RAY_METRICS_PORT:-55080}
EXPORTER_PORT=${EXPORTER_PORT:-9400}

_PROM_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_PROM_SCRIPT_DIR/utils/job_dir.sh"

# ============================================================================
# Core functions (used both when sourced and when called directly)
# ============================================================================

install_prometheus() {
    command -v prometheus &>/dev/null && return 0
    [[ -x /tmp/prometheus/prometheus ]] && return 0

    echo "[Prometheus] Downloading..."
    local version="2.48.1"
    local arch=$(uname -m)
    [[ "$arch" == "x86_64" ]] && arch="amd64"
    [[ "$arch" == "aarch64" ]] && arch="arm64"

    mkdir -p /tmp/prometheus
    curl -sL "https://github.com/prometheus/prometheus/releases/download/v${version}/prometheus-${version}.linux-${arch}.tar.gz" \
        | tar -xz -C /tmp/prometheus --strip-components=1 2>/dev/null || {
        echo "[Prometheus] Download failed" >&2
        return 1
    }
    echo "[Prometheus] Installed to /tmp/prometheus"
}

# Resolve the prometheus binary path.
_prom_bin() {
    if command -v prometheus &>/dev/null; then
        echo "prometheus"
    elif [[ -x /tmp/prometheus/prometheus ]]; then
        echo "/tmp/prometheus/prometheus"
    else
        return 1
    fi
}

# Start Prometheus with live scraping (called inside a running job).
# Uses the generic _run_with_watchdog (from ray_cluster.sh) to restart on crash.
# On restart Prometheus replays the WAL and resumes scraping — only a few seconds
# of data are lost.  Logs go to $PROMETHEUS_DATA_DIR/prometheus.log so they
# survive job termination and appear in the job directory for diagnosis.
start_prometheus() {
    install_prometheus || return 0

    local prom_bin
    prom_bin=$(_prom_bin) || return 1

    # Build targets from pre-expanded node list (expanded on host before Docker)
    local ray_targets="'localhost:${RAY_METRICS_PORT}'"
    local hw_targets="'localhost:${EXPORTER_PORT}'"
    if [[ -n "${NODELIST_EXPANDED:-}" ]]; then
        ray_targets=""
        hw_targets=""
        IFS=',' read -ra nodes <<< "$NODELIST_EXPANDED"
        for node in "${nodes[@]}"; do
            [[ -n "$ray_targets" ]] && ray_targets+=", "
            ray_targets+="'${node}:${RAY_METRICS_PORT}'"
            [[ -n "$hw_targets" ]] && hw_targets+=", "
            hw_targets+="'${node}:${EXPORTER_PORT}'"
        done
    fi

    PROMETHEUS_DATA_DIR="$(resolve_outputs_base_dir)/${JOB_DIR:-unknown}/prometheus"
    mkdir -p "$PROMETHEUS_DATA_DIR"
    local prom_log="$PROMETHEUS_DATA_DIR/prometheus.log"

    # Per-job config under the job's prometheus dir avoids two concurrent jobs
    # on the same host racing on /tmp/ray_prometheus.yml.
    local prom_config="$PROMETHEUS_DATA_DIR/ray_prometheus.yml"
    cat > "$prom_config" <<EOF
global:
  scrape_interval: 5s
scrape_configs:
  - job_name: 'ray'
    static_configs:
      - targets: [${ray_targets}]
  - job_name: 'node_hw'
    scrape_interval: 10s
    scrape_timeout: 8s
    static_configs:
      - targets: [${hw_targets}]
EOF

    # Common flags (reused for port discovery and the watchdog launch).
    local -a prom_flags=(
        --config.file="$prom_config"
        --storage.tsdb.path="$PROMETHEUS_DATA_DIR"
        --storage.tsdb.retention.time=30d
    )

    # --- Probe for a usable port (launch briefly, check, kill) ---
    # SECURITY: bind the Prometheus query API to 127.0.0.1 only.  Scraping
    # of remote node exporters / Ray metrics still works because that's
    # outbound (Prometheus initiates the HTTP GET); only the local query
    # endpoint is locked down.  Tunnel via SSH-J per print_ray_info().
    local actual_port="$PROMETHEUS_PORT"
    local max_retries=5
    local prom_pid
    for ((attempt = 0; attempt <= max_retries; attempt++)); do
        "$prom_bin" "${prom_flags[@]}" \
            --web.listen-address="127.0.0.1:${actual_port}" >>"$prom_log" 2>&1 &
        prom_pid=$!
        sleep 2
        if kill -0 "$prom_pid" 2>/dev/null; then
            break
        fi
        if [[ $attempt -lt $max_retries ]]; then
            actual_port=$((actual_port + 1))
            echo "[Prometheus] Port $((actual_port - 1)) occupied, trying ${actual_port}..."
        else
            echo "[Prometheus] FAILED: could not bind any port in range ${PROMETHEUS_PORT}-${actual_port}" >&2
            tail -20 "$prom_log" >&2
            rm -rf "$PROMETHEUS_DATA_DIR"
            return 1
        fi
    done
    kill "$prom_pid" 2>/dev/null; wait "$prom_pid" 2>/dev/null || true
    rm -f "$PROMETHEUS_DATA_DIR/lock"

    if [[ "$actual_port" != "$PROMETHEUS_PORT" ]]; then
        echo "[Prometheus] WARNING: port ${PROMETHEUS_PORT} was occupied; using ${actual_port} instead"
        PROMETHEUS_PORT="$actual_port"
    fi
    echo "[Prometheus] Started on port ${actual_port} (ray: ${ray_targets}; node_hw: ${hw_targets})"
    echo "[Prometheus] TSDB -> ${PROMETHEUS_DATA_DIR} (persistent)"
    echo "[Prometheus] Log  -> ${prom_log}"

    # Watchdog restarts Prometheus on crash. The --on-restart hook removes the
    # stale TSDB lock file (safe — the old process is dead).
    _run_with_watchdog "Prometheus" "$prom_log" \
        --on-restart "rm -f '$PROMETHEUS_DATA_DIR/lock'" -- \
        "$prom_bin" "${prom_flags[@]}" --web.listen-address="127.0.0.1:${actual_port}"
}

# Start a read-only Prometheus against persisted TSDB data.
view_prometheus() {
    local data_dir=""
    local port="$PROMETHEUS_PORT"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -p|--port) port="$2"; shift 2 ;;
            -h|--help) _usage_view; return 0 ;;
            -*) echo "Unknown option: $1" >&2; _usage_view; return 1 ;;
            *)  data_dir="$1"; shift ;;
        esac
    done

    if [[ -z "$data_dir" ]]; then
        echo "Error: data directory is required" >&2
        _usage_view
        return 1
    fi

    if [[ ! -d "$data_dir" ]]; then
        echo "[Prometheus] Directory not found: $data_dir" >&2
        return 1
    fi

    install_prometheus || return 1

    local prom_bin
    prom_bin=$(_prom_bin) || return 1

    # Minimal config — no scraping, just serve the existing data.
    # Deterministic path (keyed on port) so it's reused, not accumulated.
    local cfg="/tmp/prom_view_${port}.yml"
    cat > "$cfg" <<'YAML'
global:
  scrape_interval: 999h
scrape_configs: []
YAML

    source "$(dirname "${BASH_SOURCE[0]}")/detect_ip.sh"
    local host_hostname
    host_hostname="$(whoami)@$(hostname -f 2>/dev/null || hostname)"
    local host_ip
    host_ip="$(whoami)@$(detect_ip)"

    echo "[Prometheus] Serving metrics from: $data_dir"
    echo "[Prometheus] Open http://localhost:${port}  (Ctrl-C to stop)"
    echo "[Prometheus] If running on a remote host, tunnel first (use hostname or IP, whichever works):"
    echo "  ssh -L ${port}:localhost:${port} ${host_hostname}"
    echo "  ssh -L ${port}:localhost:${port} ${host_ip}"
    # exec replaces bash with prometheus — one fewer process, and killing the
    # PID targets prometheus directly instead of orphaning it as a child.
    # SECURITY: 127.0.0.1 — the documented tunnel `ssh -L p:localhost:p host`
    # already assumes localhost binding, this just enforces it.
    exec "$prom_bin" --config.file="$cfg" \
        --web.listen-address="127.0.0.1:${port}" \
        --storage.tsdb.path="$data_dir" \
        --storage.tsdb.retention.time=10y \
        --query.lookback-delta=15m
}

# List job directories that contain Prometheus data.
list_prometheus() {
    local search_dir="${1:-${JOB_WORKSPACE:-$_PROM_SCRIPT_DIR/outputs}}"
    if [[ ! -d "$search_dir" ]]; then
        echo "Directory not found: $search_dir" >&2
        return 1
    fi

    local found=0
    for d in "$search_dir"/*/prometheus; do
        [[ -d "$d" ]] || continue
        local job_dir
        job_dir=$(dirname "$d")
        local job_name
        job_name=$(basename "$job_dir")

        # Show human-readable size
        local size
        size=$(du -sh "$d" 2>/dev/null | cut -f1)

        # Show time range from TSDB blocks (first/last block dirs are timestamps)
        printf "  %-40s  %6s  %s\n" "$job_name" "$size" "$d"
        found=$((found + 1))
    done

    if [[ $found -eq 0 ]]; then
        echo "No Prometheus data found under: $search_dir" >&2
        return 1
    fi
    echo ""
    echo "View with:  $(basename "${BASH_SOURCE[0]}") view <path>"
}

# ============================================================================
# CLI usage
# ============================================================================

_usage_main() {
    cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  view <data-dir> [-p PORT]    Start Prometheus UI to browse persisted metrics
  list [<workspace>]           List jobs with saved Prometheus data
  install                      Download the Prometheus binary to /tmp/prometheus

Examples:
  $(basename "$0") list
  $(basename "$0") list /shared/maxtext_jobs
  $(basename "$0") view outputs/12345-JAX-llama2-70b/prometheus
  $(basename "$0") view ./outputs/12345-JAX-llama2-70b/prometheus -p 9091
EOF
}

_usage_view() {
    cat <<EOF
Usage: $(basename "$0") view <data-dir> [-p PORT]

Start a read-only Prometheus instance serving persisted TSDB data.

Options:
  -p, --port PORT   Port to listen on (default: $PROMETHEUS_PORT)
  -h, --help        Show this help
EOF
}

# ============================================================================
# Direct invocation dispatch
# ============================================================================

# Only run the CLI dispatcher when executed directly (not sourced).
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -e
    cmd="${1:-}"
    shift 2>/dev/null || true

    case "$cmd" in
        view)    view_prometheus "$@" ;;
        list)    list_prometheus "$@" ;;
        install) install_prometheus ;;
        -h|--help|help|"") _usage_main ;;
        *)
            echo "Unknown command: $cmd" >&2
            _usage_main >&2
            exit 1
            ;;
    esac
fi
