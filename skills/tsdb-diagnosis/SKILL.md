---
name: tsdb-diagnosis
description: Diagnose training job incidents and check cluster health using the per-job Prometheus TSDB. Use when the user asks to diagnose a failure root cause, check GPU/network health, query Prometheus metrics, investigate a hang, or when the triage skill recommends deeper TSDB analysis.
---

# TSDB Diagnosis

Query the per-job Prometheus TSDB to diagnose incident root causes and assess cluster health. Works on any `RAY=1` job — finished or running.

**Relationship to other skills:**
- **job-log-triage** identifies *what* failed (from logs). This skill identifies *why* using time-series metrics.
- **performance-analysis** identifies *why training is slow* (from xplane/HLO traces). This skill identifies *why metrics are anomalous* at the system level.
- The triage skill's next-step templates often say "Query Prometheus TSDB" — this skill automates that.

## Prerequisites

This skill requires a Prometheus TSDB, which is only available for `RAY=1` jobs. If the job was launched with `RAY=0`, no TSDB exists — report this and suggest re-running with `RAY=1`.

## Remote Execution via Ray Jobs API

For `RAY=1` live jobs, the **Ray Jobs API** (`http://localhost:8265/api/jobs/` *on the head node*) is a universal remote execution mechanism. It runs arbitrary Python inside the job's Docker containers — no Slurm CLI required.

**Security note (two-part fix):** Two separate Ray attack surfaces are closed by default in this repo:

1. *Dashboard / job-submission HTTP* (port 8265) — Ray's `JobSubmissionClient` HTTP API is unauthenticated; on a host with a public IP, binding to `0.0.0.0` lets an internet scanner submit code that runs as root inside the training container. Fix: bind dashboard, Prometheus, TensorBoard to `127.0.0.1` on the head node; legitimate diagnostics SSH into the head and curl `http://localhost:<port>/...` from there.
2. *GCS / coordination ports* (`RAY_PORT` for Ray, `JAX_COORDINATOR_PORT` for JAX) — these are unauthenticated cross-node services; on a public IP, internet scanners can join the cluster as raylets within seconds of `ray start --head` and execute code in the actor pool. Fix: `--node-ip-address=<cluster-private-IP>` on `ray start` (head + workers) and the JAX coordinator pinned to the cluster's RFC1918 subnet via `detect_cluster_ip()` (in `utils/detect_ip.sh`). Workers reach the head over the private subnet only; internet scanners can no longer connect.

Both endpoints — dashboard *and* GCS / JAX coordinator — are off the public internet.

**Why this matters:** In many environments, the diagnosis machine has no Slurm CLI. The Ray Jobs API is still the only way to reach inside the running containers — it just needs an SSH hop now. The agent uses `host_cmd.py` (or direct `ssh`) to reach the head node.

**Safety boundary — what you can and cannot do:**

| Safe (decoupled from training) | Dangerous (touches training) |
|-------------------------------|------------------------------|
| Read files, copy files, inspect sysfs | Kill any process in the training process tree |
| `ps`, `dmesg`, `uname`, system inspection | `pkill python3` or broad pattern kills |
| Kill/restart the metrics exporter (watchdog auto-restarts) | Send signals to JAX/XLA processes |
| Overwrite plugin scripts (picked up on next poll cycle) | Modify training config or data files |
| Overwrite `metrics_exporter.sh` + kill exporter | Write to GPU sysfs control files |
| Query Prometheus, read logs | Anything that could cause a training process to exit |

**Critical rule:** Never kill a training process. In a distributed job, killing one trainer on one node causes all other nodes to hang waiting for the dead peer, eventually crashing the entire job (heartbeat timeout or NCCL timeout across all N nodes). The observability stack (metrics exporter, plugins, Prometheus) is fully decoupled — you can tear it down and rebuild it without any training impact.

**Submitting a job (run from the head node — ssh in first):**

```bash
ssh <head_host> "curl -s -X POST 'http://localhost:8265/api/jobs/' \
  -H 'Content-Type: application/json' \
  -d '{\"entrypoint\": \"python3 -c \\\"<python_code>\\\"\", \"runtime_env\": {}}'"
```

**Polling for results (also from the head node):**

```bash
ssh <head_host> "curl -s 'http://localhost:8265/api/jobs/<job_id>'"
ssh <head_host> "curl -s 'http://localhost:8265/api/jobs/<job_id>/logs'"
```

If you've already opened the standard `ssh -L 8265:localhost:8265 -J <login> <head>` tunnel for the dashboard UI, you can also curl `http://localhost:8265/...` directly from your laptop without re-SSHing for each call.

**Node targeting:** By default, Ray jobs run on the head node. To inspect a specific worker node, the Ray job's Python code must explicitly discover and connect to that node's resources, or you can use node-local information (like hostname) to verify which node the job landed on. For operations that must run on a specific node (e.g., reading that node's dmesg), you may need to submit multiple jobs and check which node each lands on, or use Ray's scheduling hints.

This API is referenced throughout the skill — in "Metrics Exporter Operations & Live Hotfix" for deploying fixes, in Playbook 4 for GPU driver fault diagnosis, and in the troubleshooting section for process inspection.

## Workflow

0. **Triage first.** Before querying the TSDB, run the **job-log-triage** skill on the job (or confirm triage has already been done earlier in the conversation). Triage establishes critical context that shapes the entire diagnosis:
   - **Job state** — running, completed, failed, cancelled, or crashed (determines how to connect to Prometheus in step 2)
   - **Fresh start vs checkpoint restore** — affects expected baseline (restored jobs may leak RCCL resources; see "Checkpointing interference")
   - **Failure class** — hang, heartbeat-timeout, OOM, etc. (determines which playbook to run in step 5)
   - **Step range and timing** — which steps are comparable, where anomalies occurred
   - **Config parameters** — `PASSTHROUGH_ARGS`, `NNODES`, `checkpoint_period`, etc.

   Without triage, you risk misinterpreting metrics (e.g., querying post-hang idle metrics as if they were training metrics, or missing that a checkpoint restore caused resource leaks). For proactive health checks on a live job where no failure has occurred, triage is still useful to confirm the job is running and extract the head node hostname/port.

1. **Locate the TSDB.** Resolve the job directory (same rules as triage: given a log file, job dir, Slurm ID, or k8s job ID). For k8s jobs, the primary log (`outputs/<id>-<name>.log`) contains rank 0's output; per-rank logs are at `outputs/<id>-<name>/rank-N.log`. Verify `<job_dir>/prometheus/` exists **and contains data** — it should have subdirectories with ULID names (e.g., `01KHV6MFN61MJKZ3ZSYYRYGDGX`) and/or a `wal/` directory. An empty or near-empty `prometheus/` directory means Prometheus failed to start or never scraped — check the observability stack logs (see "Troubleshooting Missing or Incomplete TSDB Data"), report what went wrong, and skip to log-only diagnosis.

2. **Connect to Prometheus.** First determine whether the job is still running, then choose the appropriate access method.

   **Determine job state** (from triage, or verify directly): The triage report from step 0 tells you whether the job is running or finished. If you need to verify independently:
   - `squeue -j <id> -h -o %T 2>/dev/null` — if it returns `RUNNING`, the job is live → **Case A**.
   - If `squeue` is unavailable or the job isn't found, check the log: a `JOB SUMMARY` block means the job is finished → **Case B**. No summary and no advancing steps → likely crashed → also **Case B** (the live Prometheus is gone).

   **Critical rule: never delete a TSDB lock file, and never run `prometheus.sh view` against a running job's TSDB.** The lock is held by the live Prometheus instance. Deleting it or starting a second instance against the same TSDB risks data corruption. For finished/crashed jobs, `prometheus.sh view` handles stale lock files automatically (Prometheus detects and replaces them).

   **Case A — Live job (still running):** Prometheus is already running on the job's head node (task 0), bound to `127.0.0.1` (security hardening; the ProxyJump tunnel command in the log is the legitimate access path for users). Find the head node hostname and port from the SSH tunnel command in the job log (`ssh -J <login> -L <port>:localhost:<port> root@<head>`). The port is usually 9190 but may differ if 9190 was occupied at job start (the startup script auto-increments: 9191, 9192, ...). Also check for `[Prometheus] WARNING: port 9190 was occupied; using <port> instead` in the log. Query via `ssh <head_host> 'curl -s http://localhost:<port>/api/v1/...'` — **not** by hitting `<head_host>:<port>` directly from elsewhere (the bind is loopback-only). The head node hostname comes from the job log; on your local machine, `localhost:9090` may be a completely different Prometheus (e.g., a cluster-level monitor). If the user has opened the documented `ssh -J ... -L <port>:localhost:<port> root@<head>` tunnel on the machine you are running from, `localhost:<port>` already points to the live Prometheus — do not shadow it by starting another instance. Do **not** start a second Prometheus instance for a live job.

   **Case A fallback — Live job but Prometheus is unreachable:** If the head node's Prometheus doesn't respond (connection timeout, firewall, network partition between your machine and the head node), you can still access the TSDB data by copying the immutable blocks (not the WAL or lock file) to a temporary directory and running a read-only Prometheus against the copy. **Run the port scan first** (same as Case B — `ss -tlnp | grep 919`) to find a free port:
   ```bash
   mkdir /tmp/prom_<jobid>_readonly
   rsync -a --exclude='wal' --exclude='chunks_head' --exclude='lock' <job_dir>/prometheus/ /tmp/prom_<jobid>_readonly/
   utils/prometheus.sh view /tmp/prom_<jobid>_readonly -p <port> &
   PROM_FALLBACK_PID=$!
   ```
   This gives you access to all compacted data but **not** the most recent uncompacted samples still in the WAL. Expect a gap of up to 2 hours at the end of the data. For a long-running job, this is usually sufficient. Clean up in step 8: kill the process first (`kill $PROM_FALLBACK_PID`), then remove the data copy (`rm -rf /tmp/prom_<jobid>_readonly`).

   **Case B — Finished job (completed, failed, or cancelled):** Before starting a read-only Prometheus, **always scan for occupied ports first**. The user may have SSH tunnels on this machine forwarding to live Prometheus instances on remote head nodes (e.g., `ssh -L 9190:<head>:9190 ...` makes the live job's Prometheus available at `localhost:9190`). Starting a read-only Prometheus on the same port would shadow the live database and break the user's monitoring.
   ```bash
   ss -tlnp | grep 919
   ```
   Check the process column to classify each occupant:
   - **`ssh`** — an SSH tunnel the user set up to reach a live job's Prometheus on a remote head node. **Never kill, never shadow.** This is the most common occupant to watch for.
   - **`prometheus`** — either a live job's Prometheus (if this machine is a head node) or a stale read-only instance from a previous diagnosis session. To distinguish: inspect the cmdline with `cat /proc/<pid>/cmdline | tr '\0' ' '` and check whether `--storage.tsdb.path` points to a running job's directory (live — never kill) or a finished job's directory (stale — safe to kill with `kill <pid>`).
   - **Anything else** — a legitimate process. Don't touch it.

   Pick a port that is **not occupied by any process**. Start at 9190 and increment past all occupied ports. **Always capture the PID** so you can reliably kill it later (do not rely on bash `%1` job control — it breaks if any other command is backgrounded in between):
   ```bash
   utils/prometheus.sh view <job_dir>/prometheus -p <port> &
   PROM_PID=$!
   ```
   Wait for "Open http://localhost:" in stdout (typically <3 seconds). Kill the process when done (step 8).

   **Case C — Multi-job comparison:** At least one job must have a Prometheus TSDB. **Run the port scan once** (`ss -tlnp | grep 919`) to find all occupied ports before starting any read-only instances. Assign each finished job a different free port, skipping all occupied ports. Capture each PID:
   ```bash
   utils/prometheus.sh view <job_A_dir>/prometheus -p <free_port_1> &
   PROM_PID_A=$!
   utils/prometheus.sh view <job_B_dir>/prometheus -p <free_port_2> &
   PROM_PID_B=$!
   ```
   If one of the jobs is still live, query its existing Prometheus via `ssh <head_host> 'curl ... http://localhost:<port>/...'` (Case A) alongside the read-only instances on localhost. Jobs without a TSDB (`RAY=0`) can only be compared using log-based data from the triage skill. Query each Prometheus on its own address/port and compare results side by side.

3. **Always query all metrics first.** Before any diagnostic work, discover what the TSDB contains. This tells you which metric families were collected, how many nodes are present, and what time range is covered — essential context for choosing the right queries and interpreting results.

   In the examples below, replace `<host>:<port>` with the address from step 2: `localhost:<port>` for read-only instances (Cases B/C). For live jobs (Case A), prefix every `curl` with `ssh <head_host>` and use `localhost:<port>` on the head — the live Prometheus binds to loopback and is unreachable from other hosts directly.

   **For read-only instances (Case B/C), always start with `api/v1/status/tsdb`** to find the time range before any data queries. This endpoint doesn't need a timestamp and tells you the min/max times the TSDB covers. Without this, you won't know what `&time=` values to use for instant queries.

   ```bash
   # All available metric names — understand what the observability stack captured
   curl -s 'http://localhost:<port>/api/v1/label/__name__/values' | python3 -c "
   import json, sys
   data = json.load(sys.stdin)
   names = sorted(data.get('data', []))
   print(f'Total metrics: {len(names)}')
   for prefix in ['hw_gpu_', 'hw_tcp_', 'hw_rdma_', 'hw_io_', 'hw_mem_', 'hw_oom_', 'hw_net_', 'hw_procs_', 'hw_dmesg_', 'ray_node_', 'tb_']:
       group = [n for n in names if n.startswith(prefix)]
       if group: print(f'  {prefix}*: {len(group)} metrics')
   other = [n for n in names if not any(n.startswith(p) for p in ['hw_', 'ray_', 'tb_', 'up', 'scrape_'])]
   if other: print(f'  other: {other}')
   "

   # All hosts in the TSDB
   curl -s 'http://localhost:<port>/api/v1/label/host/values' | python3 -c "
   import json, sys
   data = json.load(sys.stdin)
   hosts = sorted(data.get('data', []))
   print(f'Hosts ({len(hosts)}): {', '.join(hosts)}')
   "

   # Time range of available data — use min/max timestamps from the TSDB.
   # For live jobs (Case A), omit &time to use current time.
   # For read-only instances (Case B/C), first find the TSDB time range:
   curl -s 'http://localhost:<port>/api/v1/status/tsdb' | python3 -c "
   import json, sys, datetime
   data = json.load(sys.stdin)
   d = data.get('data', {})
   min_t = d.get('minTime', 0) / 1000
   max_t = d.get('maxTime', 0) / 1000
   print(f'TSDB range: {datetime.datetime.fromtimestamp(min_t)} to {datetime.datetime.fromtimestamp(max_t)}')
   print(f'  min_ts={min_t:.0f}  max_ts={max_t:.0f}')
   "
   # Then query 'up' at the last known timestamp to see which targets were scraped:
   curl -s 'http://localhost:<port>/api/v1/query?query=up&time=<max_ts>' | python3 -c "
   import json, sys, datetime
   data = json.load(sys.stdin)
   for r in data.get('data', {}).get('result', []):
       ts = float(r['value'][0])
       print(f'{r[\"metric\"].get(\"job\",\"?\")} last_scrape={datetime.datetime.fromtimestamp(ts)}')
   "
   ```

   This step is not optional — always run it. The output tells you:
   - Whether GPU metrics (`hw_gpu_*`), network metrics (`hw_tcp_*`, `hw_rdma_*`), and training metrics (`tb_*`) are all present.
   - How many nodes were scraped (should match the job's `NNODES`).
   - The time range covered, confirming the TSDB has data for the period of interest.

   **Verify you are querying the correct TSDB.** After discovering metrics, cross-check a data point against the job log to confirm the database belongs to the expected job:

   ```bash
   # Query the last recorded step and loss (use a timestamp from the time range discovered above)
   curl -s 'http://localhost:<port>/api/v1/query?query=tb_step&time=<end_ts>' | python3 -c "
   import json, sys
   data = json.load(sys.stdin)
   for r in data.get('data', {}).get('result', []):
       print(f'  step={r[\"value\"][1]}')
   "
   curl -s 'http://localhost:<port>/api/v1/query?query=tb_learning_loss&time=<end_ts>' | python3 -c "
   import json, sys
   data = json.load(sys.stdin)
   for r in data.get('data', {}).get('result', []):
       print(f'  loss={r[\"value\"][1]}')
   "
   ```

   Compare these values against the corresponding `completed step:` line in the job log. The step number and loss should match exactly (loss may differ by rounding in the last decimal). If they don't match, you are querying the wrong Prometheus — re-check the address/port. This is especially important in multi-job comparisons (Case C) to avoid mixing up databases.

4. **Determine the time window.** The queries need a start/end time (Unix timestamps):
   - **From triage:** use the crash time or hang-start time identified in the triage report.
   - **From log timestamps:** parse the timestamp from the last `completed step:` line and the first training line.
   - **For health checks:** use the full job duration, or "last N minutes" for live jobs.
   - **Heartbeat timeout:** compute `crash_time - heartbeat_timeout_seconds` to get when heartbeats actually stopped.

   **Map training steps to wall-clock timestamps** using `tb_step`. This is essential for multi-job comparisons and for querying system metrics at a specific training step:

   ```bash
   # Find the wall-clock time range of the TSDB and map key steps to timestamps.
   # Use a wide range (e.g., 24h before the last known scrape) to cover the full job.
   curl -s 'http://localhost:<port>/api/v1/query_range?query=tb_step&start=<start>&end=<end>&step=60s' | python3 -c "
   import json, sys, datetime
   data = json.load(sys.stdin)
   results = data.get('data', {}).get('result', [])
   if not results: print('No tb_step data'); sys.exit()
   vals = [(float(v[0]), float(v[1])) for v in results[0]['values'] if v[1] != 'NaN']
   print(f'Step range: {vals[0][1]:.0f} to {vals[-1][1]:.0f}')
   print(f'Time range: {datetime.datetime.fromtimestamp(vals[0][0])} to {datetime.datetime.fromtimestamp(vals[-1][0])}')
   # Find timestamps for specific steps — replace with your target steps
   for target in [200, 300, 500, 1000]:
       for ts, step in vals:
           if step >= target:
               print(f'  step {target}: ts={ts:.0f} ({datetime.datetime.fromtimestamp(ts)})')
               break
   "
   ```

   Use the resulting timestamps to query system metrics (`hw_*`, `ray_*`) at the wall-clock times corresponding to specific training steps.

5. **Run the appropriate playbook** (see Diagnostic Playbooks below). Each playbook specifies the PromQL queries, how to interpret results, and what to conclude.

6. **Deepen the diagnosis if needed.** If the playbook identifies a suspicious metric but the root cause mechanism isn't clear:
   - **Check ray worker logs** — look for NCCL init waves, XLA recompilations, or warnings that correlate with the metric anomaly. See the "Ray Worker Logs" section for patterns and locations.
   - **Trace into source code** — when the anomaly is persistent, uniform across nodes, and not explained by hardware/network/thermal factors. See the "Trace suspicious findings to source code" diagnostic principle for the full methodology.

7. **Report findings** in the structured format (see Output Format below).

8. **Cleanup — stop read-only instances immediately after each diagnosis.** Kill all read-only Prometheus instances you started in step 2 as soon as you've finished querying them (i.e., after reporting findings in step 7 — do not leave them running "in case you need them later"). Use the PIDs captured in step 2:
   ```bash
   kill $PROM_PID 2>/dev/null        # single instance (Case B)
   kill $PROM_PID_A $PROM_PID_B 2>/dev/null   # multi-job (Case C)
   kill $PROM_FALLBACK_PID 2>/dev/null         # Case A fallback
   ```
   If the user asks a follow-up question that needs the TSDB again, restart the read-only instance at that point (repeating the port scan from step 2). Restarting is cheap (<3 seconds) — leaving instances running leaks ports and risks shadowing SSH tunnels or live Prometheus instances that the user sets up mid-conversation.

   Do **not** kill the live Prometheus of a running job.

## Querying Prometheus

All queries use the Prometheus HTTP API via `curl`. Parse responses with inline Python.

### Instant query (single point in time)

```bash
curl -s 'http://localhost:<port>/api/v1/query?query=<promql>&time=<unix_ts>'
```

**Always specify `&time=<unix_ts>` for read-only instances (Case B/C).** Without it, Prometheus defaults to "now" — which is past the end of the data for finished jobs, returning empty results. Use a timestamp from within the job's time range (discovered in step 3).

### Range query (time series)

```bash
curl -s 'http://localhost:<port>/api/v1/query_range?query=<promql>&start=<unix_ts>&end=<unix_ts>&step=<interval>'
```

Use `step=30s` for high resolution (short windows), `step=60s` for medium, `step=300s` for long durations.

### URL encoding for `rate()` and `increase()`

When using `rate()` or `increase()` in curl URLs, the square brackets must be URL-encoded:

```bash
# WRONG — brackets break the URL:
curl -s 'http://localhost:9190/api/v1/query?query=rate(hw_tcp_retransmits_total[5m])'

# CORRECT — URL-encode the brackets:
curl -s 'http://localhost:9190/api/v1/query?query=rate(hw_tcp_retransmits_total%5B5m%5D)'
```

`[` = `%5B`, `]` = `%5D`. This applies to all PromQL queries containing range selectors.

### Standard query + tabular output pattern

Use this pattern for all queries — it extracts the metric labels and values into a readable table:

```bash
curl -s 'http://localhost:<port>/api/v1/query?query=<promql>&time=<unix_ts>' | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data.get('data', {}).get('result', []):
    labels = r['metric']
    host = labels.get('host', labels.get('instance', '?'))
    gpu = labels.get('gpu', '')
    val = r['value'][1]
    print(f'  {host:<20} gpu={gpu:<4} {val}')
"
```

For range queries, use this to extract per-host min/max/avg:

```bash
curl -s 'http://localhost:<port>/api/v1/query_range?query=<promql>&start=<start>&end=<end>&step=30s' | python3 -c "
import json, sys
data = json.load(sys.stdin)
for r in data.get('data', {}).get('result', []):
    labels = r['metric']
    host = labels.get('host', labels.get('instance', '?'))
    gpu = labels.get('gpu', '')
    vals = [float(v[1]) for v in r['values'] if v[1] != 'NaN']
    if vals:
        print(f'  {host:<20} gpu={gpu:<4} min={min(vals):.1f}  max={max(vals):.1f}  avg={sum(vals)/len(vals):.1f}')
"
```

### Filtering by host

Narrow queries to specific hosts (e.g., the accused nodes from triage):

```promql
hw_gpu_power_watts{host=~"node001|node002"}
ray_node_gpus_utilization{host="node003"}
```

### Rate queries for counters

Counters (names ending in `_total`) must use `rate()` or `increase()`:

```promql
rate(hw_tcp_retransmits_total[5m])
increase(hw_oom_kills_total[1h])
```

## Ray Worker Logs

Ray worker logs are the second source of truth alongside TSDB metrics. They contain NCCL/RCCL initialization messages, XLA compilation events, Python tracebacks, and framework-level warnings that the TSDB cannot capture. Use them to confirm hypotheses formed from metric analysis.

**Location:** `<job_dir>/ray_logs/<hostname>/worker-*-<pid>-<pid>.out` (stdout) and `.err` (stderr). Each host has a subdirectory; each Ray worker has a pair of `.out`/`.err` files.

**Common patterns to search for:**
- `init.cc:2095 NCCL WARN MSCCL++` — NCCL communicator initialization. Count the number of "waves" (clusters of these messages separated by time gaps) to detect unexpected communicator creation (e.g., extra wave from single-replica restore broadcast).
- `Compilation of` or `Compiled` — XLA compilation events. Unexpected recompilations mid-training indicate shape changes or cache misses.
- `NCCL WARN` or `RCCL WARN` — collective communication warnings.
- Python tracebacks — stack traces from exceptions (may be caught and logged without crashing).
- `SingleReplicaArrayHandler` or `broadcast_one_replica_to_all` — checkpoint restore broadcast messages (relevant to RCCL leak diagnosis).

**Practical tips:**
- Worker logs can be very large. Use grep to search, don't read entire files.
- The head node (task 0) typically has the most informative logs — start there.
- When comparing two jobs, grep for the same pattern in both and diff the output (e.g., count NCCL init waves in each).

## Troubleshooting Missing or Incomplete TSDB Data

When the TSDB is empty, has gaps, or is missing metric families, the problem is in the observability stack itself. The stack has three components, each with its own persistent log:

| Component | Log location | What it does |
|-----------|-------------|--------------|
| **Prometheus** | `<job_dir>/prometheus/prometheus.log` | Scrapes targets, stores time series |
| **Metrics exporter** | `<job_dir>/metrics_exporter/<hostname>.log` | Runs plugins (`gpu_metrics_plugin.sh`, `host_metrics_plugin.sh`, `tb_metrics_plugin.sh`), serves metrics on port 9400 |
| **Ray runtime** | `<job_dir>/ray_logs/<hostname>/` (`gcs_server.out`, `raylet.out`, `dashboard.log`, etc.) | Cluster formation, built-in metrics exporter on port 55080 |

All three are persisted in the job directory and survive job termination. The watchdog (`_run_with_watchdog`) restarts Prometheus and the metrics exporter on crash, logging restart events with timestamps.

**Diagnosis by symptom:**

| Symptom | Which log to check | What to look for |
|---------|-------------------|------------------|
| TSDB directory empty or missing | Job log (main stdout) | `[Prometheus]` messages — did it start? Did `install_prometheus` fail? Port bind failures? |
| TSDB has data but with time gaps | `prometheus/prometheus.log` | Crash messages followed by watchdog restarts (`Watchdog: exited with code ... restart #N`). Each restart replays the WAL — gap duration ≈ crash-to-restart time (~5s) |
| No `hw_*` metrics (but `ray_*` and `tb_*` present) | `metrics_exporter/<hostname>.log` | Plugin errors — `gpu_metrics_plugin.sh` or `host_metrics_plugin.sh` failing. Exit code 99 means permanently skipped |
| No `tb_*` metrics | `metrics_exporter/<hostname>.log` | `tb_metrics_plugin.sh` failing — usually because TensorBoard event files haven't been written yet (normal during compilation), or the event file path is wrong |
| `tb_*` metrics have gaps (missing steps) | `prometheus/prometheus.log` + `metrics_exporter/<hostname>.log` | The `tb_metrics_plugin.sh` bridge is **best-effort** — gaps can occur when the GPU metrics plugin hangs (blocking the exporter), the exporter crashes and restarts, or Prometheus rejects samples whose `wall_time` is older than its `minValidTime` after TSDB compaction. Check Prometheus log for `"samples that are too old"` warnings and exporter log for plugin timeouts. **When `tb_*` data has gaps, use the raw TensorBoard event file as ground truth** (see "Recovering from `tb_*` gaps" below) |
| No `ray_*` metrics | `ray_logs/<hostname>/` (GCS, raylet, dashboard) | Ray didn't start or its metrics exporter on port 55080 failed. Check `gcs_server.out` and `raylet.out` for errors |
| Some hosts missing entirely | `prometheus/prometheus.log` | Scrape errors for those hosts — `context deadline exceeded` or `connection refused`. Means Prometheus couldn't reach the exporter on those nodes (node down, network issue, or exporter crashed before Prometheus could scrape) |
| `hw_scrape_duration_seconds` is high (>8s) | `metrics_exporter/<hostname>.log` | A plugin is running slow or timing out. ~12s usually means the GPU plugin is hitting its timeout (10s) — check for `timed out` in the exporter log. A flatline (same value for >5 min) means the exporter's collection loop is hung and all `hw_*` data is **ghost data** (see below) |
| `hw_*` metrics present but **flatline** (constant value for extended period) | `metrics_exporter/<hostname>.log` | **Ghost data** — the exporter's collection loop was stuck (e.g., a plugin hung before the timeout fix was deployed), but the HTTP server kept serving the stale cached metrics file. Prometheus assigned fresh scrape timestamps to the unchanged values, making them look "present" but they carry zero information. To detect: check `hw_scrape_duration_seconds` — if it flatlines at the same time, the data is ghost. Also check `hw_gpu_power_watts` — a GPU under training load never holds the exact same wattage for hours. Compare the suspect host against a healthy host in the same job for confirmation |
| All metrics present but stale (not updating) | `prometheus/prometheus.log` + `metrics_exporter/<hostname>.log` | Prometheus may be alive but the exporter is hung. Or Prometheus itself is hung (rare — check for WAL corruption messages) |
| `hw_gpu_power_watts` discontiguous (intermittent gaps) on one node | `metrics_exporter/<hostname>.log` + Prometheus TSDB | **GPU driver fault** causing intermittent sysfs hangs. Check `hw_scrape_duration_seconds` — alternating ~12s (timeout) and <3s (success) confirms the GPU plugin times out intermittently. Query `hw_dmesg_gpu_errors_total` — non-zero on the affected node (zero on all others) confirms a kernel-level GPU error. Use Ray Jobs API to check for D-state processes stuck in `amdgpu_cper_ring_write` (see "D-state Process Accumulation" and Playbook 4) |
| Exporter restarting repeatedly (exit code 137) | `metrics_exporter/<hostname>.log` | Watchdog log shows `exited with code 137, restart #N`. Exit code 137 = SIGKILL. Could be OOM killer, or a manual kill during hotfix deployment. Check for `OSError: Address already in use` after restarts — means the HTTP server from the previous instance hasn't released port 9400 yet |

**Checking the logs:**

```bash
# Prometheus crash/restart history
grep -i 'watchdog\|error\|fatal\|panic\|restart' <job_dir>/prometheus/prometheus.log

# Metrics exporter plugin failures (all hosts at once)
grep -i 'error\|skip\|fail\|exit' <job_dir>/metrics_exporter/*.log

# Ray runtime issues on head node
head_host=$(ls <job_dir>/ray_logs/ | head -1)
grep -i 'error\|fail\|exception' <job_dir>/ray_logs/$head_host/gcs_server.out
```

**Important:** These logs explain why the TSDB is broken — they don't replace the TSDB for diagnosing the training job. Once you've identified and understood the observability gap, report it alongside whatever partial TSDB data is available, and note which time ranges or metric families should not be trusted.

### Recovering from `tb_*` gaps

The `tb_metrics_plugin.sh` bridge is **best-effort**. It can lose data when:
- A plugin (e.g., `gpu_metrics_plugin.sh`) hangs in a D-state, blocking the exporter cycle
- The exporter crashes and the watchdog restarts it, but Prometheus has advanced its `minValidTime` past the pending steps
- The plugin drops stale steps per Rule 4 (freshness filtering) to avoid Prometheus rejection

**The raw TensorBoard event file is always the ground truth.** It is written directly by the training process and is never subject to Prometheus's timeline constraints. When `tb_*` metrics have gaps in the TSDB:

1. **Locate the event file:** `<job_dir>/tensorboard/events.out.tfevents.*` (or check the `--tensorboard_dir` flag in the job config).
2. **Read it programmatically** (Python):
   ```python
   from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
   ea = EventAccumulator("<path_to_event_dir>")
   ea.Reload()
   for tag in ea.Tags()["scalars"]:
       for e in ea.Scalars(tag):
           print(f"step={e.step}  wall_time={e.wall_time}  tag={tag}  value={e.value}")
   ```
3. **Cross-reference with TSDB:** Compare the steps present in the event file against what Prometheus has. The event file will have every step; the TSDB may be missing ranges.

**Trade-off:** Raw event files require programmatic access and cannot be queried with PromQL or visualized in Grafana dashboards. For one-off investigations this is fine. For systematic dashboarding, the Prometheus `tb_*` metrics remain the primary source — gaps are expected to be rare with the current exporter hardening (plugin timeouts, freshness filtering, mid-training restart fallback).

### Metrics Exporter Operations & Live Hotfix

The metrics exporter runs inside a Docker container on each node, managed by a watchdog that restarts it on crash. During a live job, you may need to deploy fixes to plugins or to the exporter itself without restarting the job. The two cases are fundamentally different.

**Architecture recap:**

```
watchdog (_run_with_watchdog)
  └─ metrics_exporter.sh  (long-running; poll loop every ~10s)
       ├─ gpu_metrics_plugin.sh   (re-sourced each cycle)
       ├─ host_metrics_plugin.sh  (re-sourced each cycle)
       ├─ tb_metrics_plugin.sh    (re-sourced each cycle)
       └─ HTTP server (serves cached metrics file on port 9400)
```

**Case 1 — Plugin hotfix (gpu_metrics_plugin.sh, host_metrics_plugin.sh, tb_metrics_plugin.sh):**

Plugins are re-read from disk on every poll cycle (~10s). To deploy a fix:
1. Overwrite the plugin file at the artifact path inside the container.
2. The next poll cycle picks up the new code automatically. No process restart needed.

**Network note:** The Ray dashboard *and* the per-job Prometheus query API both bind to `127.0.0.1` only on the head node (user-facing HTTP hardening). All `curl http://localhost:8265/...` and `curl http://localhost:9190/...` examples below assume you've already SSH'd to the head node (e.g., `ssh <head_host>` or `host_cmd.py "ssh <head_host> ..."`). The `localhost` in the URL refers to the head node's loopback, not your laptop. Inter-node Ray (worker → head GCS) and the JAX coordinator bind to the cluster-private IP (RFC1918) via `detect_cluster_ip()` — workers reach the head over the private subnet only, never the public internet. Prometheus *scraping* (head → worker exporters on `:9400`) is intentionally cross-node-reachable on the private subnet so Prometheus can pull metrics from each node.

**File delivery caveat:** The plugin files live at `<artifact_dir>/utils/` inside the container. The artifact directory is bind-mounted from the host NFS, but **NFS attribute caching** can delay visibility of changes by 30-60s or indefinitely. To guarantee immediate delivery, use the **Ray Jobs API** to copy the file inside the container:

```bash
# Run on the head node (ssh there first).  Use a shell command (not python -c)
# so we avoid nested-quote escaping inside the JSON entrypoint string.
curl -s -X POST 'http://localhost:8265/api/jobs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "entrypoint": "cp -p /path/on/nfs/plugin.sh /artifact_dir/utils/plugin.sh && echo done",
    "runtime_env": {}
  }'
```

The Ray job runs inside the container, bypassing NFS attribute cache issues. If you do need Python (e.g., for a complex transformation), put the script on shared storage and call `python3 /path/on/nfs/script.py` rather than embedding `python3 -c "..."` in the JSON.

**Case 2 — Exporter hotfix (metrics_exporter.sh):**

The exporter is a long-running process — overwriting the file alone does nothing until the process restarts. To deploy:
1. Deliver the updated `metrics_exporter.sh` to `<artifact_dir>/utils/` using the Ray Jobs API (same as above).
2. Kill the running exporter process. The watchdog detects the exit and restarts it with the updated code.

```bash
# Run on the head node (ssh there first).
# Kill the exporter (via Ray Jobs API, targeting a specific node):
curl -s -X POST 'http://localhost:8265/api/jobs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "entrypoint": "python3 -c \"import subprocess, os; subprocess.run([\\\"pkill\\\", \\\"-f\\\", \\\"metrics_exporter.sh\\\"])\"",
    "runtime_env": {}
  }'
```

**Pitfalls:**
- **Port conflict on restart:** When the exporter is killed, its HTTP server child process may linger briefly, holding port 9400. The watchdog restarts the exporter immediately, and the new instance gets `OSError: [Errno 98] Address already in use`. The watchdog will retry on the next crash cycle. If this persists, kill the orphaned HTTP server process explicitly before restarting.
- **Self-termination of Ray jobs:** When using `pkill -f <pattern>` inside a Ray job, the Ray job's own entrypoint command string may match the pattern, causing the job to kill itself. Use narrow patterns or filter out the current PID: `pkill -f "metrics_exporter" --signal TERM` with a pattern that won't match the Ray entrypoint string.
- **Multi-node deployment:** Ray jobs run on the head node by default. To target a specific worker node, you need to submit a job that explicitly connects to that node's Ray worker, or submit separate jobs per node. For bulk deployment across all nodes, iterate over the host list.
- **Watchdog restart logging:** Each restart is logged with a timestamp and restart count in `<job_dir>/metrics_exporter/<hostname>.log` (e.g., `[Metrics Exporter] Watchdog: exited with code 137, restart #3`). Monitor this log to confirm the restart succeeded.

**Case 3 — Remote process inspection and management:**

For any ad-hoc operation inside a container (checking process state, reading dmesg, killing specific processes), use the Ray Jobs API (see "Remote Execution via Ray Jobs API" section above). Common diagnostic commands:

```bash
# All commands below run on the head node (ssh there first).

# Check for D-state processes on the cluster
curl -s -X POST 'http://localhost:8265/api/jobs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "entrypoint": "python3 -c \"import subprocess; r=subprocess.run([\\\"ps\\\",\\\"axo\\\",\\\"pid,stat,wchan:30,cmd\\\"], capture_output=True, text=True); d=[l for l in r.stdout.splitlines() if l.split()[1:2] and l.split()[1][0]==\\\"D\\\"]; print(f\\\"D-state: {len(d)}\\\"); [print(l.strip()) for l in d[:20]]\"",
    "runtime_env": {}
  }'

# Read GPU-related dmesg errors
curl -s -X POST 'http://localhost:8265/api/jobs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "entrypoint": "python3 -c \"import subprocess; r=subprocess.run([\\\"dmesg\\\",\\\"--level=err,warn\\\",\\\"-T\\\"], capture_output=True, text=True); lines=[l for l in r.stdout.splitlines() if \\\"amdgpu\\\" in l.lower() or \\\"drm\\\" in l.lower() or \\\"cper\\\" in l.lower()]; print(f\\\"GPU dmesg lines: {len(lines)}\\\"); [print(l) for l in lines[-30:]]\"",
    "runtime_env": {}
  }'

# Get kernel and driver version
curl -s -X POST 'http://localhost:8265/api/jobs/' \
  -H 'Content-Type: application/json' \
  -d '{
    "entrypoint": "python3 -c \"import subprocess; print(subprocess.run([\\\"uname\\\",\\\"-r\\\"], capture_output=True, text=True).stdout.strip()); print(open(\\\"/sys/module/amdgpu/version\\\").read().strip())\"",
    "runtime_env": {}
  }'
```

### D-state Process Accumulation from Plugin Timeouts

When a plugin (typically `gpu_metrics_plugin.sh`) enters kernel D-state (uninterruptible sleep), the exporter's timeout mechanism kills the plugin's bash wrapper process with `SIGKILL`. However, the actual child process stuck in D-state **cannot be killed by any signal** — it remains as an unkillable zombie consuming a PID and kernel task_struct memory.

Over time, these accumulate. At ~6 per hour (one per timeout cycle that hits D-state), a node can accumulate hundreds. While Linux's default PID max (4,194,304) provides ample headroom, a large accumulation indicates a persistent kernel-level fault.

**Detection:**
```promql
# Intermittent hw_scrape_duration_seconds alternating between ~12s (timeout)
# and <3s (success) indicates the plugin hangs intermittently.
# Full timeout (all cycles >=12s) means the plugin hangs every time.
hw_scrape_duration_seconds{host="<suspect_host>"}
```

**Confirmation via Ray Jobs API** (see "Remote Execution via Ray Jobs API" and "Case 3 — Remote process inspection" for full curl examples):
```bash
# Submit a Ray job to count D-state processes and show their kernel wait channel:
# amdgpu_cper_ring_write = GPU driver CPER ring bug
# amdgpu_ras_* = RAS sysfs read hang
# Use: ps axo pid,stat,wchan:30,cmd | filter for D-state
# Also submit: uname -r and cat /sys/module/amdgpu/version for the admin report
```

**Resolution:** D-state processes cannot be cleared without a **node reboot**. They are stuck in a kernel code path that will never return. Schedule a reboot at the next job boundary or maintenance window.

## Diagnostic Playbooks

Each playbook is triggered by a specific scenario (from triage output or user request). Run the listed queries, apply the interpretation rules, and report findings.

### Correlation is not causation

When an incident occurs, many metrics move at once. A single underlying event — a network blip, a thermal excursion, an I/O stall — ripples through the system and causes correlated changes across GPU power, clocks, utilization, throughput, step time, and more. Most of what you see in the TSDB are **symptoms**, not root causes. Do not report a correlated symptom as the diagnosis.

**The diagnostic principle: find the earliest anomaly.** The root cause is the metric that deviates *first*. Everything that follows is a consequence. When you spot any anomaly:

1. **Note the timestamp** of the anomaly.
2. **Query all other metric families** at the same timestamp and in the minutes preceding it — both **training metrics** (`tb_*`) and **system metrics** (`hw_*`, `ray_*`).
3. **Find which metric deviated first.** That is the root cause candidate. Everything that moved after it is a correlated effect.
4. **Verify the causal direction.** The root cause should logically explain the downstream symptoms.

**Root causes can be in either domain — training or system.** The TSDB contains both training-level metrics (loss, grad norms, LR, MoE load balance loss, throughput) and system-level metrics (GPU power/clocks, network, I/O) on the same timeline. Do not assume the root cause is always a system event. Examples:

- **System → training:** A network retransmit burst stalls a collective → GPU power drops → step time spikes → throughput drops. System event is the root cause.
- **Training → system:** An MoE model's routing changes dramatically → `tb_learning_moe_lb_loss` spikes → some experts are overloaded while others are idle → GPU utilization becomes uneven → TGS drops. The mathematical event (routing shift) is the root cause; the GPU utilization change is a passive symptom.
- **Training → system:** A grad norm explosion → NaN values → recompilation or fallback → throughput collapse. The training instability is the root cause.
- **Training → system:** Learning rate warmup ends → larger weight updates → checkpoint sizes grow → I/O pressure during saves → periodic step time spikes. The LR schedule change is the root cause.

Always query both domains. When system metrics (power, clocks, utilization) change, check whether a training metric (`tb_learning_*`, `tb_perf_*`) shifted first — the system may be passively responding to a change in the model's computational behavior.

Always report the full chain: root cause → intermediate effects → observed symptoms. This is the core value of having all metrics in a single time-aligned TSDB.

### Trace suspicious findings to source code

Metrics tell you *what* is happening; source code tells you *why*. When the TSDB reveals an anomaly that can't be explained by external factors (network, hardware, thermal), trace the causal chain into the code to find the mechanism. This is especially valuable when the anomaly is persistent (not transient) and uniform across all nodes (not a single-node hardware issue) — these patterns point to a software-level root cause.

**When to investigate code:**
- A metric delta between two jobs with identical configs that isn't explained by hardware, network, or thermal differences (e.g., `hw_procs_running` is uniformly higher in one job).
- A persistent anomaly that starts at a specific event (checkpoint restore, profiler activation, config change) and never recovers.
- The TSDB points to a specific subsystem (checkpointing, data pipeline, collective communication) but the metric alone doesn't explain the mechanism.

**How to trace:**
1. **Start from the triggering event.** The triage report and metric timeline tell you *when* the anomaly starts. Identify what code executed at that point (e.g., checkpoint restore at step N, profiler hook at step M).
2. **Follow the call chain.** Read the relevant MaxText entry point (e.g., `checkpointing.py:load_state_if_possible` for restore issues), then trace into the libraries it calls (Orbax, JAX, XLA). Use semantic search and grep to find the code paths.
3. **Look for resource creation without cleanup.** The most common source of persistent anomalies is resources (communicators, threads, file handles, caches) that are created during a transient operation but never released. Check whether the code path creates anything that outlives the function call.
4. **Check for alternative code paths.** Libraries often have multiple implementations of the same operation (e.g., dispatcher-based vs legacy path in Orbax). If one path leaks resources, the other may not — this informs the fix direction.
5. **Verify with logs.** After forming a hypothesis from the code, confirm it in the job's logs (ray worker `.out` files, stderr). Look for initialization messages, warnings, or resource creation events that match the code path you identified.

**Accessible code locations:**
- MaxText source: `/workspace/maxtext/src/MaxText/` (checkpointing, training loop, configs)
- Orbax (checkpoint library): `/opt/venv/lib/python3.12/site-packages/orbax/checkpoint/`
- JAX: `/opt/venv/lib/python3.12/site-packages/jax/`
- XLA C++ headers (for understanding communicator/backend behavior): search under `/opt/venv/` or use `python3 -c "import jaxlib; print(jaxlib.__path__)"` to locate

**Example from real diagnosis:** TSDB showed `hw_procs_running` was ~17 higher per host in a restored job (with `enable_single_replica_ckpt_restoring=true`) vs a fresh-start baseline. Code tracing revealed: MaxText `checkpointing.py` → Orbax `SingleReplicaArrayHandler.deserialize()` → `multislice.broadcast_one_replica_to_all()` → `_merge_globalized_replicas()` which uses `jax.jit(jnp.sum)` → XLA all-reduce collective → new RCCL communicators with unique `GpuCliqueKey` → cached permanently in C++ clique cache → background polling threads increase `hw_procs_running`. Without code tracing, the TSDB alone could only say "more processes are running" — the code trace identified the exact mechanism and pointed to a fix.

### Checkpointing interference

Checkpointing is one of the most disruptive periodic events in training. During a checkpoint save, the system behavior changes dramatically — and many metrics that look anomalous are actually normal checkpoint effects. Always check whether an anomaly coincides with a checkpoint step before diagnosing it as a problem.

**What happens during a checkpoint save:**
- GPU utilization drops (GPUs idle while waiting for D2H transfer and I/O to complete)
- GPU power drops (idle GPUs draw less power)
- Host memory spikes (model parameters copied from GPU to host RAM, especially on DP replica #0 nodes)
- I/O pressure rises (`hw_io_pressure_full_pct`, `hw_mem_dirty_bytes`) as parameters are written to storage
- Step time spikes (the checkpoint step takes much longer than a normal training step)
- `hw_procs_blocked` increases (processes waiting on I/O)
- Network traffic may drop (no collectives during the checkpoint window)

**How to identify checkpoint steps:** Parse `checkpoint_period` from the log (from `Config param checkpoint_period: N`). Checkpoint saves occur at steps that are multiples of this period. If `async_checkpointing=true` is in `PASSTHROUGH_ARGS`, the save happens in the background and may overlap with subsequent training steps, spreading the I/O impact over multiple steps.

**Rules:**
- When analyzing steady-state performance, **exclude steps around checkpoint boundaries** (the checkpoint step and 1-2 steps after for async checkpointing).
- When diagnosing an anomaly, **always check whether the timestamp falls on a checkpoint step** before investigating further. A step time spike at step 200 with `checkpoint_period=200` is expected, not an incident.
- When comparing metrics across jobs with different `checkpoint_period` values, normalize by excluding checkpoint steps from both.
- Memory spikes during checkpoint saves are not OOM precursors unless they push the host to its limit. DP replica #0 nodes use ~2x model size in host RAM during saves — this is expected.

**Expected NFS congestion during checkpoint writes:**

Checkpoint-writing nodes (one per FSDP replica) write large model parameters to shared storage (VAST/NFS) simultaneously. This routinely causes:
- **TCP retransmit rates of 500–3000/s** on checkpoint-writing nodes — this is NFS retransmitting during heavy writes and is **normal, not alarming**. Do not flag TCP retransmits during checkpoint windows as a network issue unless they also appear outside checkpoint windows.
- **I/O pressure (`hw_io_pressure_full_pct`) up to 80%** — transient, lasts the duration of the checkpoint write (typically 5–8 minutes).
- **10–20+ blocked processes (`hw_procs_blocked`)** per checkpoint-writing node — processes waiting on NFS I/O.
- Non-checkpoint-writing nodes (remaining FSDP replicas) show near-zero I/O pressure and TCP retransmits during the same window.

**Diagnosing checkpoint filesystem errors (`checkpoint-fs-error` from triage):**

When a checkpoint write fails with an NFS-related error (e.g., `manifest.ocdbt.__lock` ENOENT), compare the I/O metrics at the failed checkpoint against previous successful checkpoints to assess whether NFS conditions were worse:

```bash
# For each checkpoint step, query peak I/O pressure and TCP retransmit rate:
for ts in <ckpt_2600_ts> <ckpt_2800_ts>; do
  end=$((ts+1800))
  curl -s "http://localhost:<port>/api/v1/query_range?query=hw_io_pressure_full_pct&start=$ts&end=$end&step=30s"
  curl -s "http://localhost:<port>/api/v1/query_range?query=rate(hw_tcp_retransmits_total%5B5m%5D)&start=$ts&end=$end&step=30s"
done
```

Compare peak values per node across checkpoints. A significant increase in TCP retransmits or I/O pressure at the failing checkpoint (vs. previous successes) points to NFS congestion as the trigger. Also check `hw_procs_blocked` — the head node (task 0) is especially vulnerable because it runs additional I/O-intensive services (Prometheus TSDB, Ray GCS, dashboard) alongside the checkpoint writer.

**Single-replica checkpoint restore and RCCL communicator leaks:**

When `enable_single_replica_ckpt_restoring=true` and the job restores from a checkpoint, the restore code path leaks RCCL communicators and their background polling threads. This manifests as a **persistent increase in `hw_procs_running`** (~17 extra runnable threads per host on a 24-node / 3-replica MoE run) that appears immediately after restore and never recovers. The leaked threads create constant CPU contention — competing with data pipeline threads, RCCL coordination, and the Python runtime — causing a persistent throughput (TGS) drop of ~1%.

**Root cause (code-level):** During single-replica restore, Orbax's `SingleReplicaArrayHandler` broadcasts the restored parameters from one replica to all others. The non-dispatcher code path in `orbax/.../multislice.py` (`_merge_globalized_replicas`) uses `jax.jit(lambda: jnp.sum(axis=0), out_shardings=...)` to perform this broadcast. The `jnp.sum` across the replica axis compiles to an **XLA all-reduce collective**, which causes RCCL to initialize new communicators via XLA's `AcquireGpuClique()`. Because the restore uses a different mesh/sharding configuration (single-replica subset mesh) than training, the resulting `GpuCliqueKey` is unique — so RCCL creates brand-new communicators instead of reusing the training ones. These communicators and their background polling threads are cached permanently in XLA's C++ GPU clique cache (`GpuCliqueKey → LockableGpuClique`).

**Why cleanup attempts fail:**
- `jax.clear_caches()` only clears Python-level JIT compilation caches. It does **not** touch the C++ communicator cache. Confirmed ineffective by testing.
- `jax.clear_backends()` would clear communicators but tears down the entire JAX runtime — unusable mid-training.
- There is no Python API to destroy individual RCCL communicators. The C++ clique cache has no eviction mechanism or public release API.
- MaxText's `_restore_original_array_handler()` re-registers the original `ArrayHandler` but cannot affect the C++ communicator state.

**Log-level confirmation:** The ray worker `.out` logs show an extra NCCL communicator initialization wave during restore. Look for `init.cc:2095 NCCL WARN MSCCL++` messages — a fresh-start job has 2 waves (startup + training), while a restored job has 3 waves (startup + **restore broadcast** + training). The extra wave corresponds to the `_merge_globalized_replicas` all-reduce creating new communicators.

**Verified fix:** Two patches on the `yihuang/fix-rccl-thread-leak-single-replica-restore` branch address this:
1. Replace Orbax's JIT-based broadcast with a direct RCCL broadcast via ctypes that explicitly destroys communicators after use (`ncclCommInitRank` / `ncclBroadcast` / `ncclCommDestroy` + `gc.collect()`). This eliminates the leaked threads and the ~1% TGS drop.
2. Replace Orbax's `jax.jit(create_zeros)` on non-primary hosts with `numpy.zeros` + `jax.device_put`, eliminating extra XLA compilations during restore that degrade steady-state performance.

**How to detect:** When comparing a restored job against a fresh-start baseline with identical config:
- Check `hw_procs_running` — a uniform delta across all hosts (not a spike, not on specific nodes) after restore is the signature.
- The delta is present from the first training step after restore and does not recover.
- Training-level metrics (loss, grad norms, LR) will be identical between the two jobs — this is purely system-level contention.
- A fresh start with the same config but no actual restore (even if `enable_single_replica_ckpt_restoring=true`) does not trigger the leak — the restore code path must actually execute.
- Check ray worker `.out` logs for the extra NCCL init wave (3 waves vs 2) as a definitive confirmation.
- If both jobs ran with `_env_ENABLE_XLA_DUMP=1`, compare the number of compiled modules in `xla_dump/` — a restored job may have extra `jit_create_zeros` entries from Orbax's non-dispatcher path. These extra XLA compilations are a secondary performance penalty beyond the leaked threads.

---

### Playbook 1: RCCL/NCCL Hang

**Trigger:** Triage reports `hang` or `cancelled` with underlying hang. User says "why did the job hang" or "confirm the RCCL hang."

**Goal:** Confirm the hang mechanism (RCCL busy-wait vs. dead node vs. network partition) and identify the trigger.

**Queries** (at the hang time window — from last successful training step to job end):

| # | PromQL | What it shows |
|---|--------|---------------|
| 1 | `ray_node_gpus_utilization` | Per-host GPU utilization |
| 2 | `hw_gpu_power_watts` | Per-GPU power draw |
| 3 | `rate(hw_tcp_retransmits_total[5m])` | TCP retransmit rate per host |
| 4 | `rate(hw_rdma_tx_retx_pkts_total[5m])` | RDMA retransmit rate per device |
| 5 | `rate(hw_rdma_tx_ack_timeout_total[5m])` | RDMA ACK timeouts per device |
| 6 | `hw_rdma_port_state` | RDMA port state (1=ACTIVE) |
| 7 | `rate(hw_rdma_rx_cnp_pkts_total[5m])` | Congestion notification packets |

**Interpretation:**

| GPU util | Power | Network | Diagnosis |
|----------|-------|---------|-----------|
| High (variable per GPU) | Idle-level (~260–300W MI355X) | Clean | **Confirmed RCCL busy-wait hang.** GPUs in the RCCL polling loop report high utilization, but power is at idle/standby. **Important:** utilization may not be uniform — in partial deadlocks, only GPUs that entered the collective show high utilization (e.g., 2 of 8 GPUs at 100%, rest at 0%). Always check power as the ground truth. |
| ~100% all nodes | Idle-level | Retransmit spike before hang | **Network-triggered RCCL hang.** A transient network event caused a collective to stall, then all nodes entered busy-wait. |
| ~100% most nodes | Mixed (some idle, some active) | Clean | **Possible straggler.** One or more nodes may have fallen behind, causing others to wait. Check per-node power to identify the slow node. |
| 0% on some nodes | 0W on some nodes | N/A | **Node death.** Some nodes died — remaining nodes hung waiting for them. Check `hw_dmesg_gpu_errors_total` and `hw_gpu_ras_*` on the dead nodes. |
| ~100% all nodes | Active-level (~900W MI355X) | Clean | **Not a hang** — training was still running. Re-check the triage classification. |

**Key thresholds (MI355X):**
- Active training power: ~900W per GPU
- RCCL busy-wait power: ~300W per GPU (only ~42W above idle — nearly indistinguishable from standby)
- Idle/standby power: ~260W per GPU

---

### Playbook 2: Heartbeat False-Positive

**Trigger:** Triage reports `heartbeat-timeout`. User wants TSDB confirmation that the accused tasks were healthy.

**Goal:** Prove or disprove that the accused tasks were alive and training when the heartbeat mechanism killed them.

**Setup:** From the log, extract:
- Crash time (the heartbeat error timestamp)
- Heartbeat timeout value (`jax_distributed_heartbeat_timeout_seconds`)
- Accused task IDs and their host mappings (from `NNODES`, `NODE_LIST`, task-to-host mapping in log)
- Compute heartbeat-stop time: `crash_time - heartbeat_timeout`

**Queries** (at the heartbeat-stop time — this is when the heartbeats actually failed):

| # | PromQL | What it shows |
|---|--------|---------------|
| 1 | `ray_node_gpus_utilization{host=~"<accused_hosts>"}` | Were accused tasks training? |
| 2 | `ray_node_mem_used{host=~"<accused_hosts>"}` | Memory on accused hosts |
| 3 | `hw_io_pressure_full_pct{host=~"<accused_hosts>"}` | I/O pressure (checkpoint stall?) |
| 4 | `rate(hw_tcp_retransmits_total{host=~"<accused_hosts>"}[5m])` | Network health |
| 5 | `hw_tcp_listen_drops_total{host=~"<coordinator_host>"}` | gRPC coordinator overloaded? |
| 6 | `hw_mem_dirty_bytes{host=~"<accused_hosts>"}` | Dirty page pressure |

**Interpretation:**

| GPU util on accused | Network | I/O pressure | Diagnosis |
|---------------------|---------|--------------|-----------|
| High (active training) | Clean | Low | **Confirmed false positive.** Tasks were healthy. The heartbeat mechanism failed (known gRPC bug — see `docs/jax-heartbeat-false-positive-postmortem.md`). |
| High | Retransmit spike | Low | **Likely false positive** triggered by transient network issue blocking the heartbeat gRPC. Tasks were alive but heartbeat RPCs couldn't reach the coordinator. |
| High | Clean | High | **Likely false positive** triggered by checkpoint I/O pressure. Heavy writeback blocked the heartbeat thread. |
| Low/zero | Any | Any | **Possibly a true positive.** Tasks may have actually died. Check for OOM, GPU errors, or crashes on those hosts. |

**Report template for confirmed false positive:**
> The TSDB confirms this is a heartbeat false-positive kill. At the heartbeat-stop time (<time>), all accused hosts showed GPU utilization at <X>%, TCP retransmit rate near zero, and no I/O pressure. The tasks were alive and actively training when the heartbeat mechanism declared them dead. Root cause: shared gRPC channel bug documented in `docs/jax-heartbeat-false-positive-postmortem.md`.

---

### Playbook 3: Host OOM

**Trigger:** Triage reports `oom-host`. User wants to understand the memory trajectory.

**Goal:** Trace the memory growth pattern, identify what caused the OOM, and determine if it's reproducible.

**Queries** (over the full job duration, or last 30 minutes before the OOM):

| # | PromQL | What it shows |
|---|--------|---------------|
| 1 | `ray_node_mem_used` | Per-host memory over time |
| 2 | `increase(hw_oom_kills_total[1h])` | OOM kill events |
| 3 | `hw_mem_dirty_bytes` | Dirty page accumulation |
| 4 | `hw_mem_writeback_bytes` | Writeback pressure |
| 5 | `hw_io_pressure_full_pct` | I/O stall percentage |
| 6 | `hw_procs_blocked` | Processes blocked on I/O |
| 7 | `hw_gpu_vram_used_bytes` | GPU VRAM (to correlate D2H transfers) |

**Interpretation:**
- **Gradual ramp** → memory leak (Python objects, data pipeline buffers, or XLA compilation cache growing unbounded).
- **Sudden spike** → checkpoint save (D2H copies all parameters to host RAM), data loading burst, or XLA recompilation allocating new buffers.
- **Periodic spikes that recover** → checkpoint saves. If the peak exceeds available RAM on one cycle, it's a checkpoint OOM.
- **Only DP replica #0 nodes OOM** → checkpoint save pattern. MaxText saves checkpoints from DP replica 0 only, which requires ~2x model size in host RAM (one copy in GPU VRAM, one on host for the save).
- **All nodes OOM simultaneously** → likely a data pipeline issue or XLA recompilation event.

---

### Playbook 4: Node Failure / Hardware Issue

**Trigger:** Triage reports `node-fail`, `signal-kill`, or `unknown-death`. User wants to identify hardware problems.

**Goal:** Identify hardware errors (ECC, PCIe, XGMI) or driver issues that caused the failure.

**Queries** (over full job duration — hardware errors can accumulate before causing a crash):

| # | PromQL | What it shows |
|---|--------|---------------|
| 1 | `increase(hw_gpu_ras_umc_ue_total[1h])` | HBM uncorrectable ECC errors (fatal) |
| 2 | `increase(hw_gpu_ras_umc_ce_total[1h])` | HBM correctable ECC errors (accumulating = concern) |
| 3 | `increase(hw_gpu_ras_xgmi_ue_total[1h])` | XGMI/WAFL link uncorrectable errors |
| 4 | `increase(hw_gpu_ras_xgmi_ce_total[1h])` | XGMI/WAFL link correctable errors |
| 5 | `increase(hw_gpu_ras_gfx_ue_total[1h])` | Compute engine errors |
| 6 | `increase(hw_gpu_pcie_fatal_total[1h])` | PCIe fatal errors |
| 7 | `increase(hw_gpu_pcie_nonfatal_total[1h])` | PCIe non-fatal errors |
| 8 | `increase(hw_dmesg_gpu_errors_total[1h])` | GPU/driver errors in kernel log |
| 9 | `hw_rdma_port_state` | RDMA port went down? |
| 10 | `hw_gpu_temperature_celsius` | Thermal excursion before crash? |

**Interpretation:**
- **Any uncorrectable error (UE) > 0** → hardware fault. That GPU/link is bad. The node should be drained for maintenance.
- **High correctable errors (CE)** → degrading hardware. Not immediately fatal but indicates the component is failing.
- **PCIe fatal** → PCIe link reset or card disconnect. Likely unrecoverable without node restart.
- **RDMA port state → 0** → network link went down. Check physical connectivity and switch.
- **dmesg GPU errors increasing** → driver-level GPU fault detected by the kernel.
- **Temperature spike before crash** → thermal throttling or shutdown. Check cooling.
- **Compare across nodes** — healthy nodes should have zero RAS/PCIe errors. Any node with non-zero values is the problem.

**GPU driver fault / D-state hang (non-fatal but degrading):**

Not all GPU hardware issues crash the job. A common pattern on AMD Instinct nodes is a **kernel bug in the RAS sysfs reporting path** that doesn't affect GPU compute but makes the monitoring interface unreliable. Diagnosis:

1. **Detect:** `hw_dmesg_gpu_errors_total` > 0 on one node, 0 on all others. `hw_scrape_duration_seconds` alternates between ~12s and <3s on the affected node (timeout vs success). `hw_gpu_power_watts` has intermittent gaps.

2. **Confirm via dmesg (Ray Jobs API):** Submit a Ray job to read `dmesg --level=err,warn -T` on the affected node. Look for `amdgpu` kernel oops, especially in the CPER (Common Platform Error Record) or RAS sysfs paths:
   - `amdgpu_cper_ring_get_ent_sz` / `amdgpu_cper_ring_write` — CPER ring buffer corruption
   - `amdgpu_ras_aca_sysfs_read` / `aca_sysfs_read` — RAS sysfs read hang
   - `amdgpu_cper_generate_ce_records` — correctable error record generation crash

3. **Check D-state process accumulation (Ray Jobs API):** Submit a Ray job to run `ps axo pid,stat,wchan:30,cmd` and count processes in D-state. Hundreds of D-state processes stuck in `amdgpu_cper_ring_write` confirms the driver bug is active and processes are accumulating (unkillable without reboot).

4. **Assess impact:**
   - **Training:** Unaffected. GPU compute uses `/dev/kfd` (KFD command submission), which is a completely separate code path from the RAS sysfs interface. The kernel bug is in the error-reporting path, not the compute path.
   - **Monitoring:** GPU hardware metrics (`hw_gpu_power_watts`, `hw_gpu_temperature_celsius`, `hw_gpu_vram_used_bytes`, RAS counters) are intermittently unavailable. Host and training metrics are unaffected.
   - **D-state accumulation:** ~6 unkillable processes per hour. Not immediately dangerous (Linux PID max = 4,194,304) but indicates the node needs a reboot.

5. **Recovery:** Not possible without a node reboot. D-state processes are stuck in kernel space and cannot be signaled. The CPER ring corruption persists in kernel memory. Schedule a reboot at the next job boundary.

6. **Report for cluster admin:** Include the node hostname, kernel version (`uname -r`), amdgpu module version (`/sys/module/amdgpu/version`), exact dmesg call stack, `hw_dmesg_gpu_errors_total` count, D-state process count, onset time (from dmesg timestamps and `hw_dmesg_gpu_errors_total` timeline), and confirmation that all other nodes are clean.

---

### Playbook 5: GPU Health Check

**Trigger:** User says "check GPU health", "are the GPUs OK", "check thermals", or proactive monitoring of a running job.

**Goal:** Assess GPU health across the cluster — temperatures, power, clocks, VRAM, error counters.

**Queries** (range over full job or recent window):

| # | PromQL | What it shows | Alert threshold |
|---|--------|---------------|-----------------|
| 1 | `hw_gpu_temperature_celsius` | Junction temperature | >90C = concern, >100C = throttling |
| 2 | `hw_gpu_power_watts` | Power draw | Variance >50W across GPUs = investigate |
| 3 | `hw_gpu_clock_mhz{type="sclk"}` | Core clock | Sudden drop = investigate |
| 4 | `hw_gpu_vram_used_bytes / hw_gpu_vram_total_bytes` | VRAM utilization | >95% = risk of GPU OOM |
| 5 | `hw_gpu_ras_umc_ce_total` | HBM correctable errors | Any >0 = accumulating hardware issue |
| 6 | `hw_gpu_ras_umc_ue_total` | HBM uncorrectable errors | Any >0 = bad GPU, drain node |
| 7 | `hw_gpu_ras_xgmi_ce_total` | XGMI correctable errors | Any >0 = inter-GPU link degrading |
| 8 | `hw_gpu_pcie_correctable_total` | PCIe correctable errors | Steady increase = link issue |
| 9 | `ray_node_gpus_utilization` | GPU utilization | <80% during training = underutilization |

**Report format for health check:**

```
GPU Health Summary (<N> nodes, <G> GPUs)

Temperature:  min <X>C  max <X>C  avg <X>C  (threshold: 90C)
Power:        min <X>W  max <X>W  avg <X>W  spread: <X>W
Core clock:   min <X>MHz  max <X>MHz  (nominal: <X>MHz)
VRAM:         <X>% used  (<X> GB / <X> GB)
Utilization:  min <X>%  max <X>%  avg <X>%

RAS Errors:   <N> correctable, <N> uncorrectable
PCIe Errors:  <N> correctable, <N> non-fatal, <N> fatal

Anomalous GPUs: <list of host:gpu with any non-zero errors or outlier metrics>
```

---

### Playbook 6: Network Health Check

**Trigger:** User says "check network", "is the network OK", "RDMA health", or investigating intermittent NCCL timeouts.

**Goal:** Assess network health — TCP retransmits, RDMA errors, congestion, port state.

**Queries** (range over full job or recent window):

| # | PromQL | What it shows | Alert threshold |
|---|--------|---------------|-----------------|
| 1 | `rate(hw_tcp_retransmits_total[5m])` | TCP retransmit rate | >10/s sustained = concern |
| 2 | `rate(hw_rdma_tx_retx_pkts_total[5m])` | RDMA retransmit rate | Any sustained > 0 = concern |
| 3 | `rate(hw_rdma_tx_ack_timeout_total[5m])` | RDMA ACK timeouts | Any > 0 = link or switch issue |
| 4 | `hw_rdma_port_state` | Port state (1=ACTIVE) | Any 0 = port down |
| 5 | `rate(hw_rdma_rx_cnp_pkts_total[5m])` | Congestion notifications | Sustained = network congestion |
| 6 | `rate(hw_rdma_req_tx_retry_excd_err_total[5m])` | Retry exhaustion | Any > 0 = packets lost |
| 7 | `rate(hw_rdma_req_rx_cqe_err_total[5m])` | CQE errors | Any > 0 = RDMA errors |
| 8 | `rate(hw_net_rx_errors_total[5m])` | NIC RX errors | Any > 0 = NIC issue |
| 9 | `rate(hw_net_tx_errors_total[5m])` | NIC TX errors | Any > 0 = NIC issue |
| 10 | `rate(hw_tcp_abort_on_timeout_total[5m])` | TCP connections aborted | Any > 0 = severe |

**Interpretation:**
- **TCP retransmits only, no RDMA errors** → IP/TCP path issue (NFS, coordinator gRPC), not the NCCL/RCCL data path. **Critical: high TCP retransmits alone do NOT degrade training TGS.** TCP carries NFS (checkpoint I/O, data loading) and gRPC (coordinator heartbeats), but RCCL collectives use RDMA. A job can have thousands of TCP retransmits/sec and still achieve full TGS if RDMA is clean.
- **RDMA retransmits + ACK timeouts on specific devices** → bad cable, bad port, or switch issue on that link. Identify the device and port labels. **This is the #1 cause of unexplained TGS degradation.** Even a few nodes with sustained RDMA retransmits can cause 20-30% TGS degradation cluster-wide because all-to-all collectives are synchronous — the slowest RDMA link bounds every node in every step.
- **Congestion notifications (CNP) across many nodes** → switch-level congestion. May need ECN tuning or traffic engineering.
- **Retry exhaustion** → packets permanently lost. Indicates a hard link failure, not transient congestion. Nodes with retry exhaustion should be top priority for exclusion.
- **Port state 0** → RDMA link is down. Physical layer issue. Check cable and switch port.
- **Correlate with training events** — if retransmits spike at the same time as a step time increase or hang, the network event caused the training issue.

**RDMA degradation signature in training metrics:**
- RDMA-induced TGS drops appear as a **constant step-time increase** (not variable spikes), because every step's collective communication is uniformly slowed by the degraded link. This distinguishes RDMA issues from transient events (which cause isolated spikes) and checkpoint saves (which are periodic).
- **Phase correlation:** Map RDMA retransmit bursts on specific nodes to TGS degradation phases. If TGS drops when RDMA retransmits spike on node X, recovers when node X stops retransmitting, and drops again when node Y starts retransmitting, this proves a causal link. Use `rate(hw_rdma_tx_retx_pkts_total[5m])` per host alongside `tb_perf_per_device_tokens_per_sec` over the same time window.
- **Recovery test:** When RDMA retransmits cease on the offending nodes, TGS should recover to baseline within 1-2 steps. If TGS does not recover after retransmits stop, there is an additional or different root cause.

**Node exclusion prioritization** (when spare nodes are limited):

| Priority | Condition | Rationale |
|----------|-----------|-----------|
| 1 (highest) | RDMA retry exhaustion (`hw_rdma_req_tx_retry_excd_err_total` > 0) | Permanent packet loss — hard link failure |
| 2 | Sustained RDMA retransmits + ACK timeouts | Active link degradation slowing every collective |
| 3 | High CNP (congestion notifications) | Network congestion — may be switch-level, not node-level |
| 4 (lowest) | TCP retransmits only | No RDMA impact — does not affect training TGS |

Nodes with only TCP retransmit issues should **not** be excluded unless they also show RDMA problems. If spare capacity is limited, focus exclusions on priority 1-2 nodes.

---

### Playbook 7: Training Stability

**Trigger:** User says "check training", "is training stable", "loss looks weird", "throughput dropping", or proactive monitoring.

**Goal:** Assess training health — loss convergence, gradient norms, throughput, step time consistency.

**Important:** Filter out synthetic anti-staleness fills. Only use data points where `tb_metrics_plugin_staleness_fill == 0`. The `tb_*` bridge is best-effort and may have gaps — if critical steps are missing, fall back to the raw TensorBoard event file (see "Recovering from `tb_*` gaps" in the Troubleshooting section).

**Queries** (range over full job or recent window):

| # | PromQL | What it shows |
|---|--------|---------------|
| 1 | `tb_learning_loss and tb_metrics_plugin_staleness_fill == 0` | Training loss (real data only) |
| 2 | `tb_learning_grad_norm and tb_metrics_plugin_staleness_fill == 0` | Gradient norm |
| 3 | `tb_learning_raw_grad_norm and tb_metrics_plugin_staleness_fill == 0` | Pre-clipping gradient norm |
| 4 | `tb_perf_step_time_seconds and tb_metrics_plugin_staleness_fill == 0` | Step time |
| 5 | `tb_perf_per_device_tokens_per_sec and tb_metrics_plugin_staleness_fill == 0` | Throughput |
| 6 | `tb_learning_current_learning_rate and tb_metrics_plugin_staleness_fill == 0` | Learning rate |
| 7 | `tb_step` | Current step (for progress tracking) |
| 8 | `tb_learning_moe_lb_loss and tb_metrics_plugin_staleness_fill == 0` | MoE load balance loss (if MoE model) |

**Interpretation — training metrics:**
- **Loss divergence** (sudden increase or NaN) → learning rate too high, data issue, or numerical instability. Check if grad norm spiked at the same time.
- **Gradient norm spikes** → unstable training. If `raw_grad_norm >> grad_norm`, gradient clipping is active and may be too aggressive.
- **MoE load balance loss spike** → routing changed dramatically, causing expert load imbalance. This is a training-level root cause that will show up as uneven GPU utilization and TGS drop.
- **Learning rate anomaly** → verify the LR schedule matches expectations. A flat LR when warmup should be active (or vice versa) indicates a config issue.

**Interpretation — step time and throughput:**
- **Step time periodic spikes** → likely checkpoint saves. Correlate with `checkpoint_period` from the job config. Spikes of 10-30s every N steps are normal.
- **Throughput regression** → >10% drop from steady-state average warrants investigation. Run the contention checklist below.
- **Step time gradual increase** → resource contention worsening over time. Run the contention checklist below.

**Contention checklist.** When throughput drops or step time increases, systematically check all contention sources at the same timestamp. Any of these can silently degrade performance:

| Contention source | Metrics to check | What to look for |
|--------------------|-----------------|------------------|
| **CPU** | `hw_procs_running`, `hw_procs_blocked`, `hw_context_switches_total`, `ray_node_cpu_utilization`, `hw_gpu_user_processes` | High runnable count (oversubscribed), high blocked count (I/O wait), excessive context switching. **For multi-job comparison:** a constant but higher process count in one job creates a constant baseline CPU tax — extra processes compete with data pipeline threads, NCCL coordination, and Python runtime. This won't show as a spike but as a persistent throughput gap. Compare `hw_procs_running` across jobs, not just within a single job. **Known cause:** `enable_single_replica_ckpt_restoring=true` leaks RCCL communicator polling threads on restore (~17 threads/host on a 3-replica run) because Orbax's `_merge_globalized_replicas` creates all-reduce collectives with unique `GpuCliqueKey`s cached permanently in XLA's C++ clique cache; `jax.clear_caches()` does not help (see "Checkpointing interference" section). |
| **Network (TCP)** | `rate(hw_tcp_retransmits_total[5m])`, `rate(hw_tcp_estab_resets_total[5m])`, `rate(hw_tcp_abort_on_timeout_total[5m])` | Retransmit bursts slow NFS and gRPC; resets and aborts indicate connection failures |
| **Network (RDMA)** | `rate(hw_rdma_tx_retx_pkts_total[5m])`, `rate(hw_rdma_tx_ack_timeout_total[5m])`, `rate(hw_rdma_rx_cnp_pkts_total[5m])` | RDMA retransmits slow NCCL/RCCL collectives; CNP indicates switch congestion |
| **Storage I/O** | `hw_io_pressure_full_pct`, `hw_io_pressure_some_pct`, `hw_mem_dirty_bytes`, `hw_mem_writeback_bytes` | I/O pressure stalls data loading and checkpointing; dirty page buildup indicates NFS/storage backlog |
| **Memory** | `ray_node_mem_used`, `hw_oom_kills_total`, `hw_procs_blocked` | Memory pressure causes swapping (blocked procs); approaching limit risks OOM |
| **GPU thermal** | `hw_gpu_temperature_celsius`, `hw_gpu_clock_mhz{type="sclk"}`, `hw_gpu_power_watts` | Temperature rise → clock throttle → power drop → throughput drop (all correlated symptoms of thermal issue) |
| **GPU hardware** | `hw_gpu_ras_*`, `hw_gpu_pcie_*_total`, `hw_dmesg_gpu_errors_total` | Accumulating errors degrade performance before causing a crash |
| **Training-level** | `tb_learning_moe_lb_loss`, `tb_learning_grad_norm`, `tb_learning_loss` | Model behavior changes (routing shifts, grad spikes, loss instability) that alter computational load |

Check all rows — contention sources often compound (e.g., I/O pressure from checkpointing + network congestion from RDMA traffic = amplified step time spike).

**Cross-domain correlation queries** (the power of a unified TSDB):

```promql
# Did a temperature spike cause a throughput drop?
# Query both in the same time window and compare timestamps
tb_perf_per_device_tokens_per_sec{host="node001"}
hw_gpu_temperature_celsius{host="node001"}

# Did network issues cause step time spikes?
tb_perf_step_time_seconds{host="node001"}
rate(hw_tcp_retransmits_total{host="node001"}[5m])

# Did I/O pressure from checkpointing cause a loss spike?
tb_learning_loss{host="node001"}
hw_io_pressure_full_pct{host="node001"}
```

## Multi-Job Comparison

When comparing metrics across two or more jobs (e.g., "why is job B slower than job A?"), follow these rules:

### 0. Triage all jobs first

Run the **job-log-triage** skill on **each** job being compared (per workflow step 0). For multi-job comparison, triage is especially critical because you need to know:
- Which jobs are running vs finished (determines Case A vs B for each)
- Which jobs are fresh starts vs checkpoint restores (restored jobs may have resource leaks — see "Checkpointing interference")
- The step range and checkpoint period for each job (needed to identify overlapping steps and exclude checkpoint boundaries)
- Any failures or hangs that affect which steps are comparable (e.g., don't compare post-hang idle metrics against active training)

### 1. Isolate config differences first

Before looking at metrics, compare the jobs' configurations. Parse `PASSTHROUGH_ARGS` from the header of each log file (first ~30 lines) and diff them. Common config parameters that affect performance:
- `per_device_batch_size`, `max_target_length`, `steps`
- `remat_policy`, `quantization`, `attention`
- `ici_*_parallelism`, `dcn_*_parallelism` (parallelism strategy)
- `enable_checkpointing`, `checkpoint_period`, `async_checkpointing`
- `load_balance_loss_weight` (MoE)
- `_env_*` flags (XLA flags, NCCL tuning)
- Number of nodes (`NNODES`), GPUs per node

If configs differ, the performance difference may be **expected** — the metric comparison should account for the config change, not treat it as an anomaly.

### 2. Align by training step, not wall clock

Jobs run at different times and speeds. Compare metrics at the **same training step range**, not the same wall-clock time. Use `tb_step` to map between step number and timestamp in each job's TSDB, then query system metrics at the corresponding wall-clock times.

### 3. Compare only overlapping steps

If job A ran steps 0-500 and job B ran steps 200-700, only compare metrics during the overlapping range (steps 200-500). Metrics outside the overlap are not comparable — warmup behavior (early steps) and long-run behavior (late steps) differ inherently.

**Checkpoint restore awareness:** If one job is a fresh start and the other restored from a checkpoint, the first few steps after restore may have different behavior (XLA recompilation, data pipeline warmup, potential resource leaks). Identify which job restored and what step it resumed from (from the triage report or log). Compare steady-state behavior after both jobs have fully warmed up — typically 5-10 steps after the later job's first step.

### 4. Compare runtime environment, not just config

Even with identical `PASSTHROUGH_ARGS`, the runtime environment can differ: different observability stacks (`RAY=1` vs lighter monitoring), different background processes, different node allocations. Check `hw_procs_running` across jobs — a higher process count means more CPU contention. A constant overhead (not a spike) creates a persistent throughput gap that is easy to misattribute to other causes.

### 5. Control for transient events

A throughput difference caused by a one-time network blip in job B is not a systematic issue. Use steady-state averages (skip warmup steps) and look at variance — a persistent gap indicates a real difference, while a spike indicates a transient event.

### 6. Systematic metric sweep

Run the **contention checklist** from Playbook 7 on both jobs at their overlapping step range. Compare each contention source side by side — CPU, network (TCP and RDMA), storage I/O, memory, GPU thermal, GPU hardware, and training-level metrics. The root cause is often a single metric family that differs between jobs while all others are identical.

**Key metrics for multi-job TGS comparison:**
- `tb_perf_per_device_tokens_per_sec` — the TGS metric itself. Compare steady-state averages and variance.
- `tb_perf_step_time_seconds` — step time. Inverse of TGS but more sensitive to outliers (checkpoint saves, profiler hooks).
- `rate(hw_rdma_tx_retx_pkts_total[5m])` per host — **the most impactful differentiator for identical-config jobs on different node sets.** RDMA retransmits directly slow RCCL collectives, and even a few bad nodes cause cluster-wide TGS degradation (20-30%). Use the phased correlation technique from Playbook 6 to map RDMA bursts on specific nodes to TGS degradation phases. Also check `sum(hw_rdma_req_tx_retry_excd_err_total) by (host)` for permanent packet loss.
- `hw_procs_running` — most common differentiator for identical-node-set comparisons. A uniform delta across all hosts points to a software-level resource leak (see "Checkpointing interference"). A per-node delta points to background processes or different node allocations.
- `rate(hw_tcp_retransmits_total[5m])` — TCP retransmits. **Important: high TCP retransmits alone do NOT explain TGS differences** (TCP carries NFS/gRPC, not RCCL). A job with higher TCP retransmits can still have higher TGS if its RDMA is clean. Do not chase TCP retransmits as a TGS root cause unless RDMA is also affected.

### 7. Deepen with logs and source code

If the metric sweep identifies a suspicious delta but doesn't explain the mechanism:
1. **Compare ray worker logs** between the two jobs — grep for NCCL init waves, XLA compilation events, or warnings and diff the counts.
2. **Trace into source code** if the delta is persistent, uniform, and correlates with a specific event (e.g., checkpoint restore). Follow the "Trace suspicious findings to source code" principle.

### 8. Report structure for comparison

```
## Multi-Job Comparison: <job_A> vs <job_B>

### Config differences
| Parameter | Job A | Job B |
|-----------|-------|-------|
| ... | ... | ... |

### Overlapping step range: <start> to <end>

### Metric comparison (steady-state averages over overlapping steps)
| Metric | Job A | Job B | Delta | Likely cause |
|--------|-------|-------|-------|--------------|
| ... | ... | ... | ... | ... |

### Root cause
<Traced from the earliest diverging metric, accounting for config differences>
```

## Common Pitfalls

Hard-won lessons from real diagnosis sessions. Avoid these mistakes:

1. **Empty results from read-only Prometheus.** If an instant query returns empty `result: []`, you almost certainly forgot the `&time=` parameter. Read-only Prometheus defaults to "now" which is past the end of the data. Use `api/v1/status/tsdb` to find the TSDB time range (step 3), then always include `&time=<max_ts>` for instant queries.

2. **Querying the wrong Prometheus.** `localhost:9090` on the head node may be a cluster-level Prometheus, not the job's Prometheus. The job's Prometheus runs on `127.0.0.1:9190` *on the head node* (or the auto-incremented port — bound to loopback only since 2026-05-03). Reach it with `ssh <head_host> 'curl http://localhost:9190/...'`. Always verify the TSDB by cross-checking `tb_step`/`tb_learning_loss` against the job log. If the metrics don't match (wrong metric families, wrong step range), you're querying the wrong database.

3. **Mixing up databases in multi-job comparison.** When running multiple read-only Prometheus instances on different ports, it's easy to send a query to the wrong port. Label each port clearly (e.g., "9190 = job 7879, 9191 = job 7882") and verify each with the `tb_step` cross-check before starting diagnostic work.

4. **Diagnosing symptoms as root causes.** GPU power drops, clock drops, utilization drops, and throughput drops are usually symptoms, not causes. Always trace back to the *earliest* anomaly across all metric families — it may be a training-level event (MoE routing shift, grad spike) or a system event (network retransmit, I/O stall), but not the GPU metric itself.

5. **Ignoring checkpoint steps.** A 10x step time spike at step 200 with `checkpoint_period=200` is expected, not an incident. Always identify checkpoint steps before diagnosing step time anomalies.

6. **Comparing jobs without understanding their start conditions.** A job that restored from a checkpoint may have RCCL resource leaks, XLA recompilation overhead, or different data pipeline warmup compared to a fresh start. Always check whether a job is a fresh start or restore before comparing metrics.

7. **Assuming host memory contention affects GPU training.** For GPU training, the GPU compute path is largely independent of host memory usage. A spike in host memory (e.g., from checkpoint saves) does not directly slow GPU computation unless it triggers OOM or swapping. Focus on CPU contention (`hw_procs_running`), network contention, and GPU-level metrics.

8. **Treating high absolute RDMA counters as proof of current degradation.** `hw_rdma_tx_retx_pkts_total` and similar RDMA counters are cumulative from boot (or driver reload), not from job start. A node showing 90M cumulative retransmits may have accumulated them over weeks of previous jobs and have zero retransmits during the current job. Always use `rate()` queries within the job's time window to assess current impact. For quick triage, compare `sum(hw_rdma_tx_retx_pkts_total) by (host)` at job end vs job start — the delta is what matters, not the absolute value.

9. **Assuming `jax.clear_caches()` cleans up RCCL communicators.** `jax.clear_caches()` only clears Python-level JIT compilation caches (traced in `jax/_src/api.py`). RCCL/NCCL communicators live in XLA's C++ GPU clique cache (`GpuCliqueKey → LockableGpuClique` in `xla/backends/gpu/collectives/gpu_cliques.h`), which has no eviction mechanism and no Python-accessible release API. The only way to destroy them is `jax.clear_backends()`, which tears down the entire JAX runtime and is unusable mid-training. When diagnosing RCCL resource leaks (e.g., from single-replica restore), do not recommend `jax.clear_caches()` — it has been tested and confirmed ineffective.

## Metric Reference

Complete catalog of metrics available in the TSDB. All metrics have a `host` label.

### GPU metrics (`hw_gpu_*`) — from `gpu_metrics_plugin.sh`

| Metric | Labels | Type | Description |
|--------|--------|------|-------------|
| `hw_gpu_temperature_celsius` | `gpu`, `host` | gauge | Junction temperature |
| `hw_gpu_power_watts` | `gpu`, `host` | gauge | Power draw |
| `hw_gpu_clock_mhz` | `gpu`, `host`, `type` | gauge | Clock speed (`sclk`=core, `mclk`=memory) |
| `hw_gpu_vram_used_bytes` | `gpu`, `host` | gauge | VRAM used |
| `hw_gpu_vram_total_bytes` | `gpu`, `host` | gauge | VRAM total |
| `hw_gpu_ras_umc_{ue,ce}_total` | `gpu`, `host` | counter | HBM ECC errors |
| `hw_gpu_ras_xgmi_{ue,ce}_total` | `gpu`, `host` | counter | XGMI/WAFL link errors |
| `hw_gpu_ras_gfx_{ue,ce}_total` | `gpu`, `host` | counter | Compute engine errors |
| `hw_gpu_ras_mmhub_{ue,ce}_total` | `gpu`, `host` | counter | Memory hub errors |
| `hw_gpu_ras_sdma_{ue,ce}_total` | `gpu`, `host` | counter | SDMA engine errors |
| `hw_gpu_pcie_correctable_total` | `gpu`, `host` | counter | PCIe correctable errors |
| `hw_gpu_pcie_nonfatal_total` | `gpu`, `host` | counter | PCIe non-fatal errors |
| `hw_gpu_pcie_fatal_total` | `gpu`, `host` | counter | PCIe fatal errors |

### Host/network metrics (`hw_*`) — from `host_metrics_plugin.sh`

| Metric | Labels | Type | Description |
|--------|--------|------|-------------|
| `hw_net_{rx,tx}_bytes_total` | `device`, `host` | counter | Network bytes |
| `hw_net_{rx,tx}_errors_total` | `device`, `host` | counter | NIC errors |
| `hw_net_{rx,tx}_drop_total` | `device`, `host` | counter | NIC drops |
| `hw_tcp_retransmits_total` | `host` | counter | TCP retransmits |
| `hw_tcp_listen_overflows_total` | `host` | counter | Listen queue overflows |
| `hw_tcp_listen_drops_total` | `host` | counter | Listen queue drops |
| `hw_tcp_estab_resets_total` | `host` | counter | Established conn resets |
| `hw_tcp_abort_on_timeout_total` | `host` | counter | Conns aborted after timeout |
| `hw_rdma_{rx,tx}_bytes_total` | `device`, `port`, `host` | counter | RDMA bytes |
| `hw_rdma_{rx,tx}_pkts_total` | `device`, `port`, `host` | counter | RDMA packets |
| `hw_rdma_tx_retx_{bytes,pkts}_total` | `device`, `port`, `host` | counter | RDMA retransmits |
| `hw_rdma_tx_ack_timeout_total` | `device`, `port`, `host` | counter | RDMA ACK timeouts |
| `hw_rdma_{rx,tx}_cnp_pkts_total` | `device`, `port`, `host` | counter | Congestion notifications |
| `hw_rdma_req_rx_cqe_err_total` | `device`, `port`, `host` | counter | CQE errors |
| `hw_rdma_req_tx_retry_excd_err_total` | `device`, `port`, `host` | counter | Retry exhaustion |
| `hw_rdma_port_state` | `device`, `port`, `host` | gauge | 1=ACTIVE, 0=not |
| `hw_procs_running` | `host` | gauge | Runnable processes |
| `hw_procs_blocked` | `host` | gauge | Blocked on I/O |
| `hw_context_switches_total` | `host` | counter | Context switches |
| `hw_oom_kills_total` | `host` | counter | OOM killer invocations |
| `hw_mem_dirty_bytes` | `host` | gauge | Dirty pages |
| `hw_mem_writeback_bytes` | `host` | gauge | Writeback pages |
| `hw_io_pressure_some_pct` | `host` | gauge | I/O pressure (some, 10s) |
| `hw_io_pressure_full_pct` | `host` | gauge | I/O pressure (full, 10s) |
| `hw_io_pressure_{some,full}_avg300_pct` | `host` | gauge | I/O pressure (300s avg) |
| `hw_io_pressure_full_total_us` | `host` | counter | Cumulative I/O stall time |
| `hw_dmesg_gpu_errors_total` | `host` | counter | GPU/driver errors in dmesg |
| `hw_gpu_user_processes` | `host` | gauge | Processes with /dev/kfd open |
| `hw_scrape_duration_seconds` | `host` | gauge | Plugin scrape time (emitted by `metrics_exporter.sh`). **Key health indicator:** normal is ~0.1s (plugins fast) or ~12s (GPU plugin timing out). A flatline means the collection loop is stuck and all `hw_*` data on that host is ghost data |

### Ray metrics (`ray_node_*`) — from Ray built-in exporter

| Metric | Labels | Type | Description |
|--------|--------|------|-------------|
| `ray_node_gpus_utilization` | `host` (or `instance`) | gauge | Per-GPU utilization (%) |
| `ray_node_mem_used` | `host` | gauge | Host memory used (bytes) |
| `ray_node_mem_total` | `host` | gauge | Host memory total (bytes) |
| `ray_node_cpu_utilization` | `host` | gauge | CPU utilization (%) |
| `ray_node_disk_*` | `host` | various | Disk I/O metrics |

### Training metrics (`tb_*`) — from `tb_metrics_plugin.sh`

**Caveat:** These metrics are bridged from TensorBoard event files by a best-effort plugin. Gaps are possible (see "Recovering from `tb_*` gaps"). The raw event file at `<job_dir>/tensorboard/events.out.tfevents.*` is the authoritative source for training scalars.

| Metric | Labels | Type | Description |
|--------|--------|------|-------------|
| `tb_step` | `host` | gauge | Current training step |
| `tb_learning_loss` | `host` | gauge | Training loss |
| `tb_learning_grad_norm` | `host` | gauge | Gradient norm (post-clipping) |
| `tb_learning_raw_grad_norm` | `host` | gauge | Raw gradient norm (pre-clipping) |
| `tb_learning_param_norm` | `host` | gauge | Parameter norm |
| `tb_learning_current_learning_rate` | `host` | gauge | Learning rate |
| `tb_learning_total_weights` | `host` | gauge | Total model parameters |
| `tb_perf_step_time_seconds` | `host` | gauge | Wall-clock time per step |
| `tb_perf_per_device_tflops` | `host` | gauge | Per-device TFLOP/s |
| `tb_perf_per_device_tokens_per_sec` | `host` | gauge | Per-device tokens/sec |
| `tb_learning_moe_lb_loss` | `host` | gauge | MoE load balance loss (only present for MoE models) |
| `tb_metrics_plugin_staleness_fill` | `host` | gauge | 0=real data, 1=synthetic fill |

## Output Format

```
## TSDB Diagnosis: <job_dir>

**Analysis type:** <hang | heartbeat | oom | hardware | gpu-health | network-health | training-stability | multi-job-comparison>
**Time window:** <start_time> to <end_time> (<duration>)
**Nodes:** <N> (<host1, host2, ...>)

### Findings
<Plain-English summary: what the metrics show, what the root cause is>

### Metric evidence

<For each key metric queried, show a table with per-host/per-GPU values.
Use the actual query results — do not fabricate data.>

| Host | GPU | Value | Interpretation |
|------|-----|-------|----------------|
| ... | ... | ... | ... |

### Anomalies detected
<Any nodes/GPUs/metrics that deviate from cluster norm. "None" if all healthy.>

### Log evidence (if applicable)
<Ray worker log findings that confirm the metric-based hypothesis.
E.g., "3 NCCL init waves in job 7882 vs 2 in job 7879, confirming extra
communicator creation during checkpoint restore.">

### Source code trace (if applicable)
<Code-level root cause chain when the diagnosis traced into source code.
E.g., "checkpointing.py → SingleReplicaArrayHandler → _merge_globalized_replicas
→ jax.jit(jnp.sum) → XLA all-reduce → leaked RCCL communicators.">

### Correlation with triage
<If triggered from a triage report, confirm or refute the failure hypothesis.
Example: "Triage classified this as an RCCL hang. TSDB confirms: all 192 GPUs
showed 100% utilization at 310W (idle-level) during the hang window, with zero
network errors. This is a confirmed RCCL busy-wait deadlock.">

### Recommended next steps
<Numbered list of specific actions based on the diagnosis>
```

## Integration with Triage

When the triage skill identifies a failure that benefits from TSDB analysis, it includes "Query Prometheus TSDB" in its recommended next steps. The handoff works as follows:

1. **Triage provides:** failure class, crash time, accused hosts/tasks, job directory.
2. **This skill takes over:** starts Prometheus against the persisted TSDB, runs the appropriate playbook, and reports metric-level evidence.
3. **Key handoff scenarios:**

| Triage class | Diagnosis playbook | What TSDB adds |
|--------------|-------------------|----------------|
| `hang` | Playbook 1 (RCCL Hang) | Confirms busy-wait signature, identifies network trigger |
| `heartbeat-timeout` | Playbook 2 (Heartbeat) | Proves false positive — tasks were healthy |
| `oom-host` | Playbook 3 (OOM) | Memory trajectory, checkpoint correlation |
| `node-fail` / `signal-kill` | Playbook 4 (Hardware) | RAS errors, PCIe faults, thermal excursion |
| `nccl-timeout` | Playbook 6 (Network) | Network health at failure time |
| `unknown-death` | Playbook 3 + 4 | OOM evidence or hardware faults |

For proactive use (no triage handoff), the user triggers directly with health-check requests.
