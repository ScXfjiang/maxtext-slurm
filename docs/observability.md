# Observability

Standard training diagnostics — job logs, [TensorBoard](https://www.tensorflow.org/tensorboard), GPU traces — cover the training side well. But long-running jobs at scale fail in ways these sources can't diagnose: NCCL/RCCL hangs, GPU thermal throttling, network degradation, NFS latency spikes, silent performance regressions. These are system-level issues, and diagnosing them requires collecting, persisting, and correlating a different class of metrics — ideally alongside the training metrics, in a single queryable store that serves as the **ground truth data layer** for both human dashboards and automated reasoning.

`RAY=0` (the default) is a plain job launcher with zero overhead — best for short benchmarks or simple validations. `RAY=1` activates the full observability stack — GPU, host, network, and training metrics collected into a single [Prometheus](https://prometheus.io/) TSDB, queryable live and after the job ends, with no measurable steady-state throughput impact — best for long-running production jobs and debugging. See [Design rationale](#design-rationale) for the technology choices.

```bash
RAY=1 submit.sh 70b -N 8
```

## Telegram notifications

Push notifications for job state changes, hang detection, and periodic log updates — works with any [Slurm](https://slurm.schedmd.com/) job, no `RAY=1` required:

```bash
utils/slurm_job_monitor.sh -j <slurm_job_id>
```

The standalone `utils/telegram_bot.sh` is also available as a general-purpose notification primitive — compose it into any script or pipeline.

See [Notifications](notifications.md) for one-time setup, full usage, options, and programmable notification patterns.

## Real-time monitoring

Three dashboards are accessible via SSH tunnel while the job runs. All three bind to **`127.0.0.1` on the job's head node** — they are not reachable on the head's external interfaces. This is intentional: the Ray dashboard's [job-submission HTTP endpoint](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html) is unauthenticated and would let anyone with TCP reach to that port run arbitrary code as root inside the container. The localhost binding closes that surface; legitimate access goes through SSH ProxyJump.

> **Inter-node services** (Ray GCS on `RAY_PORT`, JAX coordinator on `JAX_COORDINATOR_PORT`) bind to the cluster-private RFC1918 subnet via `detect_cluster_ip()` in `utils/detect_ip.sh`, not `0.0.0.0`. Workers reach the head over the private subnet only — internet scanners that previously joined open Ray clusters as raylets can no longer connect. The metrics-exporter port (`9400`) remains cross-node-reachable on the private subnet so Prometheus can scrape it. Single-node and no-private-network setups fall back to the public IP automatically.

| Dashboard | What it shows | Port (head's localhost) |
|-----------|--------------|------|
| [Ray](https://www.ray.io/) Dashboard | Actor status, live stack traces, flame graphs | 8265 |
| Prometheus | GPU thermals/power/clocks/VRAM, RAS errors, TCP retransmits, RDMA counters, training scalars (loss, LR, grad norms, throughput) | 9190 (auto-increments if occupied) |
| TensorBoard | Training loss curves, learning rate schedules | 6006 |

### SSH tunnel

The job output prints SSH tunnel instructions (both hostname and IP). The Prometheus port defaults to 9190 but auto-increments if occupied (check the log for the actual port). Example:

```
SSH tunnel (hostname):
  ssh -J me@login01 -L 8265:localhost:8265 -L 6006:localhost:6006 -L 9190:localhost:9190 root@node001
SSH tunnel (IP):
  ssh -J me@203.0.113.10 -L 8265:localhost:8265 -L 6006:localhost:6006 -L 9190:localhost:9190 root@node001
```

`-J <login>` is a ProxyJump through the cluster login node; `localhost` after `-L` refers to the head node's loopback (where the dashboards listen). Then open `http://localhost:8265`, `http://localhost:6006`, or `http://localhost:9190` in your browser.

**Important:** The SSH tunnel binds these ports on your **local machine** (where you run `ssh`). The same ports on other cluster nodes are not the job's services — every dashboard binds to the head's loopback only. To curl from a different machine on the cluster, SSH into the head first (`ssh node001 'curl http://localhost:9190/...'`) instead of trying to hit `node001:9190` directly.

If a port is occupied locally (e.g., monitoring multiple jobs), change the first port number in `-L`. For example, `-L 18265:localhost:8265` then access `http://localhost:18265`.

### py-spy target selection in 1-GPU-per-process mode

The Ray Dashboard's `py-spy dump / record` button profiles the training subprocess (not the Ray worker shell) via a wrapper that resolves the Ray worker PID to its child training process. In [1-GPU-per-process mode](job-submission.md#1-gpu-per-process-mode) each Ray actor has multiple training children (one per local GPU) — the wrapper picks **local rank 0** by default. To profile a different rank from the Dashboard, export `PYSPY_LOCAL_RANK=N` on the head node before invoking the button.

## Post-run diagnostics

Prometheus metrics and Ray session logs are written to `JOB_WORKSPACE` and available after the job ends — no live job required:

```
$JOB_WORKSPACE/
  12345-JAX-llama2-70b.log                    # Job log (stdout + stderr)
  12345-JAX-llama2-70b/
    artifact -> ...                           # Artifact at submit time
    log -> ../12345-JAX-llama2-70b.log        # Symlink to the job log
    prometheus/                               # Prometheus time-series data
      prometheus.log                          # Prometheus process log (startup, errors, restarts)
    ray_logs/
      node001/session_*/logs/                 # Head node: raylet, GCS, dashboard, worker logs
      node002/session_*/logs/                 # Worker node logs
      ...
    metrics_exporter/                         # Metrics exporter logs (one per node)
      node001.log
      node002.log
      ...
    core.12345.*.node002.py_xla_execute.515   # Core dumps (if any)
    <training outputs>
```

### Prometheus metrics

Prometheus metrics — GPU thermals, host network stats, and training scalars — are written to `<JOB_WORKSPACE>/<job>/prometheus/` on the shared filesystem, not to ephemeral container storage. Each scrape writes to the TSDB immediately (Ray metrics every 5s, plugin metrics every 10s), so no data is lost if the job terminates unexpectedly.

After the job ends, use `utils/prometheus.sh` to inspect the data:

```bash
# List jobs with saved Prometheus data
utils/prometheus.sh list

# Start a read-only Prometheus UI to browse metrics
utils/prometheus.sh view outputs/12345-JAX-llama2-70b/prometheus
```

See [Tooling: prometheus.sh](tooling.md#inspect-prometheus-metrics-after-a-job-ends) for the full command reference.

### Ray logs

Ray session logs (raylet, GCS server, worker stdout/stderr) are written to `<JOB_WORKSPACE>/<job>/ray_logs/<hostname>/` on the shared filesystem — critical for debugging crashes where the worker process dies without a Python traceback.

```bash
# Raylet C++ logs (fatal errors, OOM kills, NCCL hangs)
cat $JOB_WORKSPACE/<job>/ray_logs/<hostname>/session_*/logs/raylet.out
cat $JOB_WORKSPACE/<job>/ray_logs/<hostname>/session_*/logs/raylet.err

# Worker process logs (MaxTextTrainerActor stderr/stdout)
cat $JOB_WORKSPACE/<job>/ray_logs/<hostname>/session_*/logs/worker-*.err

# GCS server logs (coordination service, head node only)
cat $JOB_WORKSPACE/<job>/ray_logs/<hostname>/session_*/logs/gcs_server.out
```

### Core dumps

When a training process crashes (SIGSEGV, SIGABRT, etc.), a core dump is written to the job directory. Core dumps are enabled automatically when the job directory has >500 GB free, and the container waits for writes to finish before exiting. See [Debugging: Core dumps](debugging.md#core-dumps) for the full details — naming convention, gdb inspection, configuration variables, and the wait-for-write mechanism.

## Unified TSDB as ground truth

Ray's built-in metrics and the three plugin-supplied classes — GPU hardware, host/network, and [TensorBoard training scalars](#tensorboard-metrics-in-prometheus) — produce a complete picture of the training system in a single Prometheus TSDB. The unified store is the **ground truth data layer** for automated reasoning and diagnosis.

Training anomalies rarely stay in one domain: a GPU thermal throttle reduces clock speeds, which slows step time, which triggers grad norm spikes. In siloed systems each symptom appears in a different tool. With all signals in one TSDB — time-aligned, consistently labeled by `host` — a reasoning engine can trace the causal chain programmatically. The TSDB is also a self-contained artifact: load it with `utils/prometheus.sh view` and analyze without the original cluster.

For a concrete example, see the [JAX heartbeat false-positive post-mortem](jax-heartbeat-false-positive-postmortem.md): application logs said tasks had "crashed," but querying the TSDB proved GPU utilization was 100% and the network was healthy at the time of the alleged failure — evidence that was only available because it had been collected independently of the application.

## TensorBoard metrics in Prometheus

Training metrics (loss, learning rate, gradient norms, throughput) traditionally live only in TensorBoard — a separate UI with no persistence beyond the event files, no alerting, and no way to correlate with system metrics. `tb_metrics_plugin.sh` bridges this gap by reading TensorBoard event files directly (raw TFRecord format) and exporting all scalar metrics into Prometheus.

**Why this matters:**

- **Cross-domain correlation.** Join training behavior (loss, grad norms, throughput) with system state (GPU temperature, power, TCP retransmits) in a single PromQL query.
- **Post-run availability.** Training metrics persist in the TSDB alongside system metrics — no running TensorBoard server or original event files needed.
- **Alerting potential.** Standard Prometheus alerting rules can detect loss divergence, LR anomalies, or throughput drops.
- **Framework-agnostic.** Works with any TensorBoard-writing framework (MaxText, PyTorch `SummaryWriter`, TensorFlow).
- **Zero overhead.** A file-size gate (single `stat` call, ~0.1 ms) skips the 1 MB tail read entirely when no new data has been flushed. Active reads (~5 ms) are negligible compared to the 10-second poll interval.

**Exported metrics** (all prefixed `tb_`):

| Metric | Description |
|--------|-------------|
| `tb_step` | Current training step |
| `tb_learning_loss` | Training loss |
| `tb_learning_grad_norm` | Gradient norm |
| `tb_learning_raw_grad_norm` | Raw (pre-clipping) gradient norm |
| `tb_learning_param_norm` | Parameter norm |
| `tb_learning_current_learning_rate` | Learning rate |
| `tb_perf_step_time_seconds` | Wall-clock time per step |
| `tb_perf_per_device_tflops` | Per-device TFLOP/s |
| `tb_perf_per_device_tokens_per_sec` | Per-device tokens/sec throughput |
| `tb_metrics_plugin_staleness_fill` | 0 = real training data, 1 = synthetic anti-staleness fill (see below) |
| *(plus any other scalar tags the framework writes)* | |

**Example queries:**

```promql
# Loss over time
tb_learning_loss{host="node001"}

# Step time spikes — which steps took abnormally long?
tb_perf_step_time_seconds > 2 * avg_over_time(tb_perf_step_time_seconds[30m])

# At the same wall-clock time, check system-level contention sources:
hw_gpu_temperature_celsius{host="node001"}            # Thermal throttling?
rate(hw_tcp_retransmits_total{host="node001"}[5m])    # Network issues?
hw_procs_running{host="node001"}                      # CPU contention?
hw_io_pressure_full_pct{host="node001"}               # Storage stalls?
```

**Anti-staleness fills.** During long idle periods (e.g. checkpoint saves), the plugin periodically re-emits the last known data to keep `tb_*` series alive in Prometheus. Use `tb_metrics_plugin_staleness_fill == 0` to exclude synthetic fills from analysis queries.

**Best-effort bridge.** The `tb_*` metrics in Prometheus are best-effort. Gaps can occur when another plugin hangs and blocks the exporter cycle, when the exporter crashes and restarts, or when Prometheus rejects samples whose timestamps are too old after TSDB compaction. The raw TensorBoard event file (`<job_dir>/tensorboard/events.out.tfevents.*`) is always the ground truth for training scalars.

## Customizing metrics

Any `*_metrics_plugin.sh` script in `utils/` is auto-discovered by `metrics_exporter.sh` on every node. Each plugin runs every 10 seconds and outputs Prometheus exposition format text.

To add a new metric:

1. Create `utils/<name>_metrics_plugin.sh` (must be executable).
2. The script receives the short hostname as `$1` and prints Prometheus metrics to stdout.
3. Auto-discovered on the next job start — no configuration needed.

**Optional features** for stateful or conditional plugins:

- **State persistence.** Print `# STATE <value>` as the last stdout line — the exporter stores it and re-exports it as `_PLUGIN_STATE` on the next invocation.
- **Idle replay.** When a state-using plugin outputs only a STATE line (no metrics), the exporter replays the last cached metrics automatically — skip expensive work when nothing changed.
- **Cache invalidation.** Emit `# STATE __NONE__...` when the data source is gone — the exporter clears the cache so stale metrics are not replayed.
- **Permanent skip.** Exit with code 99 to tell the exporter to never invoke the plugin again (e.g. wrong node type, feature disabled).
- **Execution timeout.** Each plugin invocation is killed if it exceeds the poll interval (~10s). This prevents a hung plugin (e.g., a sysfs read stuck in kernel D-state) from blocking other plugins. Timed-out invocations produce no output; the exporter replays the last cached metrics for that plugin.

**Built-in plugins:**

| Plugin | What it collects |
|--------|-----------------|
| `gpu_metrics_plugin.sh` | Per-GPU temperature, power, clocks, VRAM, RAS error counters (UMC/XGMI/GFX/MMHUB/SDMA) |
| `host_metrics_plugin.sh` | Network stats, TCP retransmits, RDMA counters, scheduling pressure, OOM kills, storage write pressure |
| `tb_metrics_plugin.sh` | TensorBoard training scalars (see [TensorBoard metrics in Prometheus](#tensorboard-metrics-in-prometheus)) |

---

## Design rationale

See [Architecture: Observability pipeline](architecture.md#observability-pipeline-ray-mode) for the high-level pipeline and [Extensibility: Metrics plugins](extensibility.md#extending-the-observe-tier-metrics-plugins) for the design rationale.

### Why Ray?

Ray is not just infrastructure for distributing exporters — it is a first-class metrics source and the only tool that provides live introspection of GPU training jobs:

- **Broad metrics baseline.** Ray's built-in exporters provide host memory, CPU utilization, disk I/O, and GPU utilization (`ray_node_mem_*`, `ray_node_cpu_*`, `ray_node_disk_*`, `ray_node_gpus_*`), forming the broad baseline of the TSDB.
- **Live introspection.** The Ray Dashboard offers live thread dumps and CPU flame graphs — the only way to diagnose a hung multi-node GPU job in real time without SSH-ing into each node.
- **Distributed coordination.** Ray starts `metrics_exporter.sh` on every node, launches Prometheus and TensorBoard on the head node, and routes session logs to persistent storage — all from a single cluster bootstrap.
- **Zero steady-state overhead.** Training runs in a subprocess (`subprocess.Popen`) with a dedicated Python interpreter and no Ray threads, eliminating GIL contention entirely (see [Architecture: Subprocess isolation](architecture.md#performance-subprocess-isolation)). Ray adds initialization time (~15-30s), negligible for production runs.

### Why Prometheus?

Prometheus is a lightweight, single-binary time-series database:

- **Pull model.** Matches the exporter-per-node architecture — each node runs `metrics_exporter.sh` on port 9400, Prometheus scrapes them all.
- **PromQL.** Cross-domain correlation in a single query — join training loss with GPU temperature, or correlate throughput drops with network retransmits.
- **Portable TSDB.** The file-based data directory is a self-contained artifact. Copy it anywhere, load it with `utils/prometheus.sh view`, query it without the original cluster.
- **Ecosystem.** Grafana, alertmanager, recording rules — immediately compatible with the entire Prometheus ecosystem.

No heavier alternative (InfluxDB, OpenTelemetry) is justified for this use case.

### Two-source metrics model

| Source | Coverage | Examples |
|--------|----------|----------|
| **Ray built-in** | Broad host and GPU baseline | `ray_node_mem_*`, `ray_node_cpu_*`, `ray_node_disk_*`, `ray_node_gpus_*` |
| **Plugin system** | Deep supplemental coverage | GPU thermals/power/clocks/VRAM/RAS, TCP retransmits, RDMA counters, training scalars |

Ray provides the breadth; the [plugin system](#customizing-metrics) provides the depth. Together they cover every metric class in one Prometheus TSDB — the [ground truth data layer](#unified-tsdb-as-ground-truth) for automated reasoning.

The entire stack is self-contained in the container: Ray is `pip install`'d and Prometheus is downloaded as a single binary at startup — no cluster-wide pre-installation required. Collected metrics are written to `JOB_WORKSPACE` alongside other job outputs, so there is no separate server to deploy or maintain. Falls back to non-Ray mode if the cluster fails to start (clears `USE_RAY` so `_train.sh` launches MaxText directly). Ray startup failures are logged to the job output for diagnosis.
