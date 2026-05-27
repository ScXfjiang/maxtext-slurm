---
name: job-log-triage
description: Triage MaxText training jobs from log files — failed, hanging, running, or completed. Use when the user asks why a job failed, wants to diagnose an error, sees a crash, hang, timeout, OOM, NCCL error, heartbeat timeout, wants to understand a job's status, or asks about bad/low/dropping TGS or throughput.
---

# Job Log Triage

Classify a job's status and failure mode from its log file and recommend targeted next steps. Works on any job — `RAY=0` or `RAY=1`, finished or running, Slurm or local.

## Workflow

1. **Locate the log file and job directory.** The user may provide a log file, a job directory, or a Slurm job ID. Always resolve both:
   - **Given a job directory** → follow the `log` symlink inside it to find the log file.
   - **Given a `.log` file** → the job directory is the sibling directory with the same name minus `.log` (e.g., `outputs/7877-FOO.log` → `outputs/7877-FOO/`).
   - **Given a Slurm ID** → look for `outputs/<id>-*` (directory) or `outputs/<id>-*.log` (log file).
   - **Given a k8s job ID** (e.g., `k8s-20260310-080957-475e`) → same pattern: `outputs/<id>-*`.

   Having the job directory gives access to `ray_logs/`, `prometheus/`, xplane profiles, and other per-job artifacts needed for deeper diagnosis.

   **Directory layout:** The `outputs/` folder contains:
   - **Job directories** — always have a `log` symlink (pointing to `../<dirname>.log`). Named `<slurm_id>-<config>`, `k8s_<timestamp>-<config>`, or `local_<timestamp>-<config>`.
   - **Log files** — `<dirname>.log` files, siblings of their job directories.
   - **Per-rank logs (k8s only)** — `rank-N.log` files inside the job directory. The primary `.log` file contains only rank 0's output. For node-specific failures on k8s jobs, check `outputs/<id>-<name>/rank-N.log` for the failing rank's full output. These per-rank logs are NOT present in Slurm jobs (Slurm puts all ranks in one file via `srun -l`).
   - **Shared checkpoint directories** — hold checkpoint files and TensorBoard data, shared across runs. **No `log` symlink.** Created when `enable_checkpointing=true`.

   When triaging all jobs in `outputs/`, skip directories that have no `log` symlink — they are shared checkpoint dirs, not jobs.

2. **Read the tail of the log** (last 200 lines) — this is where the JOB SUMMARY and final errors appear. Then read the head (first 80 lines) for the header block (env vars, stage timeouts, node list).

3. **Determine job status** using two signals: the JOB SUMMARY block (if present) and **training step progress** (not log mtime — see warning below).

   | Log pattern | Status |
   |-------------|--------|
   | `JOB SUMMARY` + `Status: SUCCESS (exit 0)` | completed |
   | `JOB SUMMARY` + `Status: FAILED (exit N)` (N not 130/143) | failed |
   | `JOB SUMMARY` + exit 130 or 143 | cancelled — but always check for a preceding hang or failure |
   | No `JOB SUMMARY` + training steps actively advancing | running |
   | No `JOB SUMMARY` + training steps stopped + **job still alive** (Slurm RUNNING) | **hanging** (see hang diagnosis below) |
   | No `JOB SUMMARY` + training steps stopped + job ended (or Slurm state unavailable) | unknown-death (SIGKILL / OOM-kill / preemption) |

   **Do not rely on log mtime to detect hangs or determine if a job is running.** A hung job can produce non-training output (Ray buffered C++ messages, system warnings, topology logs) that updates the file mtime without advancing training. The reliable indicator is whether the **last `completed step:` line** is recent. Use the training progress projection (step 5) to compare the last step against where training should be.

   **`RAY=1` Slurm log truncation.** For `RAY=1` jobs, the Slurm log may show **fewer training steps than actually completed**. Ray actors write output to internal buffers that are forwarded asynchronously to the driver's stdout (which becomes the Slurm log). When the job finishes, remaining buffered output may not flush before the process exits. **Always cross-check the Slurm log's last step against `ray_logs/<head_node>/worker*.out`** — these files are written directly by the actor and contain the authoritative training progress. A job that appears to have stopped at step 33 in the Slurm log may have actually completed all 100 steps per the worker log. Failure to check this can cause misclassification (e.g., labeling a completed job as "unknown-death").

   To distinguish a hang from a death when there is no JOB SUMMARY: check Slurm job state (`scontrol show job <id>`) if the Slurm ID is known. If the job is still RUNNING, it's a hang. If the job has ended, it's an unknown-death.

4. **Classify the failure** by scanning the log for signatures in the table below. Scan bottom-up — the most diagnostic error is usually near the end.

5. **Project training progress.** Steps are **0-indexed**: `completed step: N` means step N is done, and `steps=T` in config means the job runs steps 0 through T-1 (T steps total). A job is complete when `last_step == T - 1`.

   Parse from the log (and from `ray_logs` for `RAY=1` jobs — the Slurm log may be truncated; see the `RAY=1` truncation warning in step 3):
   - **Step time:** extract the `seconds:` field from recent `completed step:` lines (use the steady-state average, skip warmup steps 0–4 relative to the first step).
   - **Total steps:** from `steps=N` in `PASSTHROUGH_ARGS` (log header). The final step number will be N-1.
   - **Checkpoint period:** from `Config param checkpoint_period: N` lines in the log (printed by MaxText during config dump). If `enable_checkpointing=true` is in `PASSTHROUGH_ARGS` but no explicit period, the default is 200.
   - **First completed step:** the first `completed step: N` in the log. If N > 0, the job restored from a checkpoint at step N-1 (restore skips the checkpoint step and starts training at N). Report this as "restored from checkpoint step N-1".
   - **Confirming restore vs fresh start:** For `RAY=1` jobs, check `ray_logs/*/worker*.out` for:
     - `No existing checkpoints found, not restoring checkpoint.` → fresh start
     - `restoring from this run's directory step N` → restored from checkpoint step N
   - A fresh start with `enable_checkpointing=true` saves an initial checkpoint at step 0 before training, so `first_step=0`. A restore from that checkpoint produces `first_step=1`.
   - **Last completed step** and its approximate wall-clock time.
   - **Steps completed this run:** `last_step - first_step + 1`. The total progress including prior runs is `last_step + 1`.

   Then compute:
   - **Expected step now:** `last_step + (now - last_step_time) / step_time`. If this is significantly ahead of the last logged step, the job is stalled. This is the primary hang detection signal.
   - **Total training time vs wall time:** Sum all `seconds:` values from every `completed step:` line to get total training time. **Actually compute the sum** — do not estimate by multiplying average step time by step count, as this hides checkpoint overhead variance and can mask a 40-60 minute hang inside "expected overhead." For `RAY=1` jobs, sum from a single worker log (authoritative); for `RAY=0`, filter to one task ID to avoid double-counting. Compare against the job's wall time (from the JOB SUMMARY). A significant gap (wall time >> total training time + expected setup/compilation overhead of ~15-30 min) signals unaccounted time that warrants investigation. Common causes: RCCL hang (all GPUs idle/spinning before the job was killed), slow checkpoint writes beyond what is included in step times, data loading stalls, XLA recompilation pauses, or delay between the last step and job termination due to any failure (OOM, NCCL timeout, etc.). Do not assume the gap is a hang — cross-reference with the failure classification (step 4) and, for `RAY=1` jobs, the TSDB to determine what happened during the gap. This check is especially useful for finished jobs where "expected step now" is not applicable.
   - **Progress lost on failure:** `last_step - last_checkpoint_step` steps of unrecoverable work. The last checkpoint step is the highest multiple of `checkpoint_period` that is <= `last_step`. For runs that never reached `checkpoint_period`, all training steps are lost (the initial step-0 checkpoint is the starting state, not a training milestone).
   - **Estimated time remaining:** `(total_steps - 1 - last_step) * step_time` (steps remaining until the final step T-1).
   - **Last periodic checkpoint saved by this run:** the highest multiple of `checkpoint_period` reached by `last_step`. If `last_step < checkpoint_period`, this run saved no periodic checkpoint — report "none". Do not count the initial step-0 checkpoint as a periodic checkpoint; it is just the starting state for fresh runs.

   Include these projections in the report — they make stalls obvious (expected step 2000 but last step is 316 = hung for hours) and quantify the cost of the failure.

   **TGS trend check (proactive).** When extracting step times, also check for TGS degradation — even if the user didn't ask about it. Compare the TGS of the last 10 steps against the steady-state average (steps 5-15). If the recent TGS is >10% below the early average, flag it as `tgs-degradation` in the report's "Additional findings" section, note the magnitude and step range of the drop, and include the TGS degradation next-step template in the recommendations. A job can "succeed" (complete all steps, exit 0) while having run 20-30% slower than it should have — catching this proactively saves significant GPU-hours on subsequent runs.

6. **For `RAY=1` jobs:** Search the log for the SSH tunnel command (look for `ssh -J ... -L 9190:localhost:9190` near the start of training). Extract the head node hostname and Prometheus port from the tunnel command (the port defaults to 9190 but auto-increments if occupied — also look for `[Prometheus] Started on port`). Include the tunnel command, hostname, and port in the report.
   - **Job still live** (running or hanging): the live Prometheus binds to `127.0.0.1` on the head node (since 2026-05-03 — security hardening). Query it via `ssh <head_host> 'curl -s http://localhost:<port>/api/v1/...'`, **not** by hitting `<head_host>:<port>` directly from another machine. Do **not** use `localhost:9090` either — that may be a different Prometheus (e.g., cluster-level monitor). The SSH tunnel command in the log uses ProxyJump (`ssh -J <login> -L <port>:localhost:<port> root@<head>`) and binds the port on the **user's laptop**, not on the head — only useful if the user opened the tunnel themselves. Ask the user if they want you to set up port forwarding to access the Ray Dashboard (8265), TensorBoard (6006), and Prometheus on their local machine.
   - **Job already ended** (completed, failed, or cancelled): live dashboards are gone — do not attempt to query them. For post-hoc analysis, use `utils/prometheus.sh view <job_dir>/prometheus` to start a read-only Prometheus against the persisted TSDB. If you gathered evidence from live queries earlier in the conversation, include those results.

7. **Report findings** in the structured format described in "Output format" below.

## Failure classification table

Scan the log for these signatures, in priority order (first match wins the primary classification, but report all matches found).

### Infrastructure failures (before training starts)

| Class | Log signature(s) | Stage | What happened |
|-------|------------------|-------|---------------|
| **prolog-kill-no-log** | Empty `outputs/<id>-…/` dir with no `.log` sibling, OR user reports a `[ERROR] Job <id> died in prolog before writing any log` message from `submit.sh`, OR `sacct`/`squeue -t all` shows `FAILED` with `Reason=RaisedSignal:53(Real-time_signal_19)` and `RunTime=00:00:01`. Entry point is **not** the log file — scan `squeue -t all` / `sacct` instead, since the usual `outputs/` walk finds nothing. | Slurm prolog | slurmd killed the job before the batch script could run — usually because `--output` path exceeds ext4's 255-byte per-path-segment limit (long `JOB_NAME`), or a partition-level prolog script failed. `submit.sh` catches most of these pre-submit (length check in `parse_job_args.sh`) and the rest at t+3s (`squeue -t all` poll + cleanup). If this signature still appears on a fresh submit, the cause is partition-side — wait for the partition to recover, check slurmd logs on allocated nodes, or try a different partition. |
| **container-pull-fail** | `[ERROR] Pull failed for`, `[ERROR] Authenticated pull failed`, `[ERROR] Login to ... failed` | Docker pull | Image pull or registry auth failed |
| **container-load-fail** | `[ERROR] Unable to determine image name or ID from docker load output` | Docker pull | Tarball load failed |
| **no-gpu** | `WARNING: No GPU devices detected` | Container start | No GPU devices visible |
| **nccl-nic-fail** | `NCCL FATAL ... Failed to auto-detect NCCL_SOCKET_IFNAME; ABORTING` | Container start | Multi-node: no suitable NIC for NCCL |
| **port-fail** | `FATAL: Could not find a free port for JAX coordinator`, `FATAL: Could not find a free port for Ray` | Job start | Port allocation failed |
| **model-not-found** | `!!! Unknown model:`, `!!! Model name resolution failed` | Training start | Config file missing or ambiguous name |
| **patch-branch-fail** | `[FAIL] Failed to check out` | Container start | MaxText hotfix branch checkout failed |
| **ray-start-fail** | `[Ray] HEAD failed to start`, `[Ray] HEAD timeout`, `[Ray] WORKER failed` | Ray init | Ray cluster bootstrap failed (falls back to non-Ray) |

### Stage timeouts

| Class | Log signature(s) | What happened |
|-------|------------------|---------------|
| **preflight-timeout** | `== Preflight TIMEOUT` | Preflight checks hung (stale GPU processes, NFS, NUMA) |
| **pull-timeout** | `== Docker pull TIMEOUT` | Image pull took too long (slow registry or large image) |
| **ecc-timeout** | `== ECC check TIMEOUT` | ECC memory check hung (GPU driver issue) |
| **train-timeout** | `== Training TIMEOUT` | Training exceeded the configured wall-clock limit |

### Training failures (during training)

| Class | Log signature(s) | What happened |
|-------|------------------|---------------|
| **hang** | Training steps stopped advancing (last `completed step:` far behind expected), job still RUNNING in Slurm, no error before the stall | Collective communication deadlock (NCCL/RCCL all-reduce/all-gather hang) — all nodes waiting on each other |
| **heartbeat-timeout** | `UNAVAILABLE: The following tasks are unhealthy (stopped sending heartbeats)`, `The tasks have crashed` | JAX coordination heartbeat timeout — **known bug** with documented root cause (see diagnosis below) |
| **oom-host** | `Killed` (from OOM killer), `oom-kill`, `Out of memory` | Host OOM: process killed by Linux OOM killer |
| **oom-gpu** | `OUT_OF_MEMORY`, `XLA_ERROR`, `ResourceExhausted`, `RESOURCE_EXHAUSTED`, `out of memory` | GPU VRAM exhausted during compilation or execution |
| **nccl-timeout** | `NCCL WARN Timeout`, `NCCL error`, `NCCL WARN` (during training), `Timeout waiting for`, `ncclSystemError` | NCCL/RCCL collective timeout — network or GPU issue |
| **xla-compile-fail** | `INTERNAL: Failed to compile`, `XLA compilation failed`, `HloModule` + `error` | XLA/GPU compiler failure |
| **python-exception** | `Traceback (most recent call last):` | Unhandled Python exception (read the traceback for details) |
| **signal-kill** | `Training subprocess killed by` | Training process killed by signal (SIGSEGV, SIGABRT, etc.) |
| **subprocess-fail** | `Training subprocess exited with code` | Training process exited non-zero (read preceding output) |
| **actor-fail** | `Actor failed:` | Ray actor exception (includes traceback) |
| **checkpoint-fs-error** | `Training stopped: Checkpointing failed`, `[Errno 2] No such file or directory: 'manifest.ocdbt.__lock'` | Checkpoint write failed due to NFS/storage filesystem error — **but the checkpoint may be intact** (see checkpoint filesystem error diagnosis below) |

### Training performance issues (job running but underperforming)

| Class | Detection method | What happened |
|-------|-----------------|---------------|
| **tgs-degradation** | TGS drops >10% below early steady-state average and stays low, or TGS steadily declines over time. Detected from `completed step:` lines in worker logs — not a log error signature. | Network (RDMA retransmit), resource contention, or hardware degradation slowing collective communication. The job runs without errors but significantly underperforms. |

### Job-level status

| Class | Log signature(s) | What happened |
|-------|------------------|---------------|
| **cancelled** | `CANCELLED (scancel / SIGTERM)`, exit 130 or 143 | User or scheduler cancelled the job — **but always check for a preceding hang or failure** (see below) |
| **node-fail** | `NODE_EXIT host=... exit=` (non-zero) | One or more nodes exited with errors |
| **unknown-death** | No JOB SUMMARY, training steps stopped, job no longer in Slurm RUNNING state | Process killed externally (SIGKILL, OOM-kill, preemption) with no chance to write summary |
| **stage-fail** | `== ... FAILED (exit=` | A non-timeout stage failure (check exit code) |
| **soft-success** | `Status: FAILED (exit 1)` or `(exit 143)` in JOB SUMMARY **BUT** `completed step: <T-1>` is logged for the configured `steps=T` (training reached the last step). One of three signatures: (a) **single rank exits 1**, others exit 0 — `ray.kill` racing Ray cluster shutdown during teardown; (b) all ranks **exit 143** cascading from one rank's late post-training failure — JAX/PjRt `pending_event_logger` stalling for minutes before a single rank exits non-zero, or a node hardware fault precisely at job end; (c) all ranks exit 143 at exactly `--time=` walltime — Slurm-cancel during teardown after training reached `T-1`. | **Training data is intact** — full loss curve in the TensorBoard events file, full step-time history in the log. The non-zero exit is a post-training artifact, not a training failure. **Do not resubmit unless the training data was needed beyond what was captured.** Do not let the FAILED status fool downstream parsers; a job with `completed step: T-1` and a log that ends with `pending_event_logger` / `NODE_EXIT 143` cascade is a successful training run. |

## Hang diagnosis

When a job is still RUNNING in Slurm but training has stalled:

1. **Confirm the hang using step progress, not mtime.** Find the last `completed step:` line and use the training progress projection (workflow step 5) to compare the actual step against the expected step. A large gap confirms the hang. Do not rely on log mtime — hung jobs can produce non-training output (Ray C++ buffered logs, topology messages) that keeps the mtime fresh. **Pre-training hangs:** If zero `completed step:` lines exist but all tasks reached the BARRIER ("Synchronizing hosts before training loop") and then went silent, the hang occurred during the first RCCL collective — before step 0 could complete. This is still an RCCL deadlock, just during init rather than mid-training.
2. **Default assumption: RCCL/NCCL collective hang.** When all N tasks completed the same step and then stopped simultaneously, this is almost always an RCCL/NCCL deadlock — all nodes are blocked waiting on each other inside a collective (all-reduce, all-gather, reduce-scatter). A dead node is unlikely when the last output shows every task healthy at the same step.
3. **If the job was launched with `RAY=1` and is still live, use the Ray Dashboard** (port 8265) as the primary diagnostic tool. It provides live stack traces and CPU flame graphs for every actor — showing exactly where each training process is blocked. SSH tunnel to the head node and open `http://localhost:8265`. Check each actor's status and stack trace to confirm the RCCL hang and identify which collective operation is stuck.
4. **Query the Prometheus TSDB** for GPU utilization and power. The definitive RCCL hang signature is:
   - `ray_node_gpus_utilization` = **100%** on some or all GPUs (busy-waiting in RCCL polling loop)
   - `hw_gpu_power_watts` = **idle-level** (~260-310W on MI355X, vs ~900W during active training)
   - 100% utilization + low watts = confirmed RCCL busy-wait hang. No real computation is happening; the GPUs are spinning in a tight polling loop inside the stuck collective.
   - **Partial utilization pattern:** During init-phase hangs (before step 0), only a **subset** of GPUs per node may show 100% — those that entered the stuck collective. The rest show 0% (haven't entered yet). Typically 2-3 GPUs per node at 100% while the rest idle. Nodes that show a **different set of stuck GPU indices** than the majority are likely where the deadlock originated (e.g., if 6/8 nodes have GPUs 4,7 stuck but 2 nodes have GPUs 1,7 stuck, investigate those 2 outlier nodes).
   - **TCP retransmit analysis:** Check `hw_tcp_retransmits_total` in two ways: (1) **rate during hang** — `increase(hw_tcp_retransmits_total[<hang_duration>])` to detect active network issues; (2) **absolute totals across nodes** — nodes with totals orders of magnitude higher than peers (e.g., 10M vs 10K) may have degraded network hardware (bad NICs, cables, or switch ports). **Caveat:** These are cumulative lifetime counters — they reflect all network activity since the counter was last reset (reboot, driver reload), not just this job. High absolute totals alone do not prove current degradation; the node may have recovered. Note the outliers in the report but do not automatically recommend exclusion based solely on absolute totals — a successful retry on the same nodes disproves active hardware issues (see point 6).
   - Also check RDMA counters around the hang time.
5. **Check for core dumps.** Hangs do **not** produce core dumps — the processes are alive and spinning, not crashing. Core dumps are only generated by crashes (SIGSEGV, SIGABRT). A `scancel` sends SIGTERM (exit 143), which triggers a clean shutdown, not a core dump. Check the coredump paths anyway to confirm no crash preceded the hang. The coredump candidates, checked in order by `_container.sh` (first with >500GB free wins):
   - `<JOB_WORKSPACE>/<job_dir>/` (per-job output directory)
   - `<JOB_WORKSPACE>/` (outputs root)
   - Paths in `COREDUMP_EXTRA_DIRS` from `container_env.sh` (e.g., `/perf_apps/maxtext_coredump`)
   - Core files match: `core.*`
   - No core files + RCCL hang signature = pure deadlock, no crash involved.
6. **Recommended action:** Kill the job (`scancel <id>`) and **retry on the same nodes first** — especially for init-phase hangs (before step 0), which are often transient RCCL race conditions that resolve on retry. If the retry succeeds on the same nodes, the hang was transient and no node exclusion is needed; note TCP retransmit outliers in the report for awareness but do not act on them. Only if the hang **recurs on the same nodes** should you escalate: resubmit with `--exclude=<suspect_nodes>` (targeting TCP retransmit outliers or GPU utilization pattern outliers), add `_env_NCCL_DEBUG=INFO` for detailed RCCL diagnostics, and use `slurm_job_monitor.sh -j <id>` for early hang detection via Telegram alerts.
7. **Heartbeats do NOT detect hangs — but can be triggered by prolonged hangs.** The heartbeat mechanism (`jax_distributed_heartbeat_timeout_seconds`) is a **liveness check**, not a progress check — it only detects dead/crashed processes. During an RCCL hang, all processes are alive and actively spinning in a busy-wait loop, so they continue sending heartbeats successfully. For short hangs (minutes), the heartbeat will not fire because from its perspective every process is healthy. **However, during prolonged hangs (30+ minutes), the gRPC channel can eventually deadlock (Bug 3 in `docs/jax-heartbeat-false-positive-postmortem.md`), blocking heartbeat delivery on one or more tasks. After `heartbeat_timeout_seconds` elapses without heartbeats, the coordinator declares those tasks dead — killing the entire job.** This means a hang can end in three different ways: (a) killed by `scancel` (user or `slurm_job_monitor.sh` detects the stall), (b) killed by a training timeout, or (c) killed by a heartbeat timeout after gRPC deadlock — each looks different in the log but the root cause (RCCL hang) is the same. Only training step progress monitoring (`slurm_job_monitor.sh`) can detect hangs early. When a heartbeat timeout kills a hung job, increasing `jax_distributed_heartbeat_timeout_seconds` only delays the kill — it does not fix the underlying hang. Focus diagnosis on the hang root cause (see TSDB diagnosis for GPU util + power analysis).

## Heartbeat timeout diagnosis

**This is a known issue with a documented root cause.** The JAX coordination service's heartbeat mechanism has design flaws that cause it to declare healthy, actively-training tasks as dead. The root cause — a shared gRPC channel that blocks heartbeat RPCs — is documented in `docs/jax-heartbeat-false-positive-postmortem.md`. The error message "The tasks have crashed" is misleading; the tasks are almost always alive and training normally when they are killed.

**Two distinct failure modes:**
1. **Init-phase kill (deterministic):** If no training steps completed before the crash, the heartbeat timeout is shorter than XLA compilation + initial checkpoint save time. CPU contention during compilation starves the gRPC heartbeat thread. Fix: increase the timeout (the job will succeed on retry with a larger value).
2. **Mid-training kill (probabilistic):** If training was running for many steps before the crash, this is the gRPC channel deadlock (Bug 3 in the postmortem). Fix: set the timeout to several hours.

**Default assumption: false positive.** Unless there is clear evidence of a real crash (Python traceback, NCCL error, or SIGKILL on the accused tasks before the heartbeat message), treat heartbeat timeouts as false positives. **However, "false positive" only means the heartbeat mechanism was wrong — it does NOT mean the heartbeat timeout is the primary failure.** The most commonly misdiagnosed case is an RCCL hang that triggered the heartbeat timeout: the task was alive (spinning in busy-wait), so the heartbeat was technically a false positive, but the real problem is the hang. You must complete ALL items in this checklist — do not stop after confirming the task was alive.

Apply this checklist **in full**:

1. **Check if "dead" tasks logged their own death.** Search for `Polled error from coordination service` or `Terminating process because the JAX distributed service detected fatal errors` on the accused task IDs. If a task reports itself as dead, it was alive — the heartbeat declaration was false. **This does NOT establish that the heartbeat timeout is the primary failure — continue to item 3 to check for a preceding hang.**

2. **Check for earlier errors on the accused tasks.** Search backward from the heartbeat error for Python tracebacks, NCCL errors, or SIGKILL on those specific task IDs. If you find a real error preceding the heartbeat timeout, the heartbeat timeout was a true positive (the task really died) — but this is the rare case.

3. **MANDATORY: Check whether training was progressing or stalled before the heartbeat error.** This is the critical step that distinguishes (a) a pure gRPC false positive (training was actively progressing when killed) from (b) a hang that triggered the heartbeat timeout (training stalled, then gRPC deadlocked after ~30+ min). Skipping this step is the #1 cause of misdiagnosis — confirming the task was alive (item 1) is necessary but not sufficient, because tasks are also alive during an RCCL hang (they're spinning in busy-wait).

   **Always compute the wall-time gap first** (even when TSDB is available — it takes one command and immediately reveals whether further investigation is needed):
   ```bash
   # For RAY=1 jobs, sum step seconds from a worker log (authoritative, not the truncated Slurm log):
   grep "completed step:" <job_dir>/ray_logs/<any_host>/worker*.out | sed 's/.*seconds: //' | sed 's/,.*//' | awk '{s+=$1} END {print s}'
   # For RAY=0 Slurm jobs, sum from the log (filter to rank 0 to avoid double-counting):
   grep "^.0:.*completed step:" <log_file> | sed 's/.*seconds: //' | sed 's/,.*//' | awk '{s+=$1} END {print s}'
   # For k8s jobs, the primary log already contains only rank 0 — no rank prefix to filter:
   grep "completed step:" <log_file> | sed 's/.*seconds: //' | sed 's/,.*//' | awk '{s+=$1} END {print s}'
   ```
   Compare against wall time from the JOB SUMMARY. **If the gap exceeds setup/compilation overhead (~15-30 min), there was a stall.** Do not estimate — compute the actual sum. A hand-waved estimate can hide a 40-60 minute hang inside "expected overhead."

   Then determine *what* caused the stall:
   - **With TSDB** (`RAY=1` job with `prometheus/` directory): query GPU power and utilization in the period **before** the heartbeat error (e.g., 30-60 minutes before). If GPUs show normal training power (~800-1000W) right up to the heartbeat error, the heartbeat timeout is the sole issue — a pure gRPC false positive with no preceding hang or failure. If GPUs show 100% utilization + idle-level power (~260-310W) before the heartbeat error, the primary failure is an **RCCL hang** — reclassify as `hang` with heartbeat-timeout as secondary. If GPUs show 0% utilization + idle power on specific nodes while others are active, those nodes likely died (OOM-kill, hardware fault) — investigate those nodes. Also check `hw_oom_kills_total`, `hw_dmesg_gpu_errors_total`, and memory pressure metrics to identify non-hang root causes.
   - **Without TSDB**: a significant wall-time gap with all tasks at the same last step strongly suggests an RCCL hang (the most common mid-training stall). Cross-reference with other failure signatures in the log.

4. **If TSDB is available** (`RAY=1` job with `prometheus/` directory), the postmortem's methodology can also be applied to confirm the false positive: query GPU utilization, TCP retransmits, I/O pressure, and memory at the failure time to confirm the tasks were healthy. Recommend TSDB diagnosis for definitive confirmation.

5. **Recommended fix:** Increase `jax_distributed_heartbeat_timeout_seconds` to several hours (e.g., 14400 for a 4-hour timeout) so the broken mechanism cannot kill productive training. Use `slurm_job_monitor.sh` for independent hang detection instead of relying on heartbeats. See the postmortem's "Practical Workarounds" section for the full defense-in-depth strategy.

## GPU OOM diagnosis

When `RESOURCE_EXHAUSTED: Out of memory while trying to allocate` appears:

1. **Check `XLA_PYTHON_CLIENT_MEM_FRACTION` first.** This is the most common cause of GPU OOM on large models — not wrong parallelism or batch size. Find the value in the log header (env var dump), `configs/<model>.env.sh` (per-model override), or `train_env.sh` (global default). The default is `.85`, which works for most models but is **too low for 405B-class and 1T-class models**. Increasing to `.93` is often the complete fix. Note: XLA may inflate allocations when more memory is available, so increasing the fraction doesn't always yield proportional headroom. See `docs/job-submission.md` ("Per-model environment overrides" section) for the override layering.

2. **Do NOT jump to parallelism changes.** A 405B model running on 8 nodes with `ici_fsdp_parallelism=-1` and `ici_tensor_parallelism=1` is a valid, tested configuration — it works correctly once the memory fraction is right. The OOM error message ("Out of memory while trying to allocate 221.71GiB") can be misleading: it does not mean the model fundamentally doesn't fit, just that JAX wasn't given enough of the GPU's physical memory.

3. **Verify the fix worked.** If a subsequent job with the same config but higher `XLA_PYTHON_CLIENT_MEM_FRACTION` succeeds, confirm this was the root cause. Ask the user if a follow-up job is running successfully.

4. **Only if memory fraction is already `.93`+**, fall back to the standard OOM playbook:
   - Reduce `per_device_batch_size` or `max_target_length`
   - Try `remat_policy=full` (if not already set)
   - Reduce `XLA_PYTHON_CLIENT_MEM_FRACTION` if RCCL/NCCL buffer allocation errors appear (too high)
   - The sweet spot is typically `.85`–`.93`; above `.93` risks starving RCCL/NCCL of communication buffer memory

## Checkpoint filesystem error diagnosis

When `Training stopped: Checkpointing failed. [Errno 2] No such file or directory: 'manifest.ocdbt.__lock'` (or similar OCDBT/filesystem errors) appears in a worker log:

**Critical: the checkpoint may be intact.** The OCDBT checkpoint library treats any filesystem error during the lock/finalization phase as fatal and kills training — even if the checkpoint data was already written. The error is often a transient NFS metadata lookup failure, not actual data corruption.

**Step 1: Check all workers, not just the one that errored.**

For `RAY=1` jobs, search every worker's log for the checkpoint outcome:

```bash
for host in <job_dir>/ray_logs/*/; do
  hostname=$(basename "$host")
  grep -l "Saved a checkpoint at step <N>\|Checkpointing failed" "$host"/worker-*.out 2>/dev/null | while read f; do
    echo "=== $hostname ==="
    grep "Saved a checkpoint at step <N>\|Checkpointing failed" "$f"
  done
done
```

If most workers report "Saved a checkpoint at step N" and only one reports the error, the checkpoint is almost certainly intact. Confirm by checking that the checkpoint directory has its final name (e.g., `<N>`, not `<N>.tmp-*`).

**Step 2: Extract checkpoint write duration.**

The time to write each periodic checkpoint is visible in the step time of the **first step after each checkpoint**. The step that runs concurrently with the background checkpoint write takes much longer than normal because it blocks on the commit:

```bash
# For RAY=1 jobs — extract from authoritative worker log:
for ckpt_step in 200 400 600 800 1000; do
  next=$((ckpt_step+1))
  echo -n "ckpt $ckpt_step -> step $next: "
  grep "completed step: $next," <worker_log> | sed 's/.*seconds: //' | sed 's/,.*//'
done
```

Normal training steps take ~30s; post-checkpoint steps of 300–500s indicate checkpoint writes of 5–8 minutes. **Compare this against `jax_distributed_heartbeat_timeout_seconds`** — if checkpoint write time approaches the heartbeat timeout, the job is at risk of a false heartbeat kill during a future checkpoint.

**Step 3: Root cause — NFS/storage congestion.**

Checkpoint-writing nodes (one per FSDP replica, typically the first N nodes where N = number of replicas) write large model parameters to shared storage (VAST/NFS) simultaneously. This causes:
- TCP retransmit rates of 500–3000/s on checkpoint-writing nodes (NFS retransmissions)
- I/O pressure (`hw_io_pressure_full_pct`) up to 80%
- 10–20+ blocked processes (`hw_procs_blocked`) per node

These are **expected during checkpoint writes** and are not alarming on their own. The `manifest.ocdbt.__lock` error occurs when the NFS congestion causes a transient metadata lookup failure — the NFS client returns ENOENT for a file that exists on the server. The OCDBT library does not retry, treating it as fatal.

**Step 4: Head node vulnerability.**

The Ray head node (task 0) is especially susceptible because it runs additional I/O-intensive services (Prometheus TSDB writes, Ray GCS, Ray dashboard) alongside the checkpoint writer. This additional NFS client-side load makes it more likely to hit transient lookup failures during congestion.

## TGS degradation diagnosis

When a job is running (or completed) but TGS is lower than expected or dropping over time. This is not a crash or hang — the job produces no error signatures — but it is a performance failure that needs diagnosis.

**Step 1: Extract the TGS timeline from worker logs.**

For `RAY=1` jobs, use the authoritative worker log (not the truncated Slurm log):

```bash
grep "completed step:" <job_dir>/ray_logs/<head_node>/worker*.out 2>/dev/null | \
  sed 's/.*completed step: //' | \
  awk -F', ' '{step=$1; for(i=1;i<=NF;i++){if($i~/Tokens\/s\/device/){split($i,a,": ");tgs=a[2]}; if($i~/seconds/){split($i,a,": ");secs=a[2]}}; printf "step=%-5s  secs=%s  TGS=%s\n", step, secs, tgs}'
```

Compute the steady-state average (skip warmup steps 0-4 relative to first step) and look for deviations >10%.

**Step 2: Identify the degradation pattern.**

| Pattern | What it looks like | Most likely cause |
|---------|--------------------|-------------------|
| **Constant drop** | TGS drops by a fixed amount (e.g., 3290→2520) and stays there. Step time increases by a constant delta on every step. | RDMA retransmits on one or more nodes — bad cable, port, or switch. Every collective is uniformly slowed. |
| **Phased drops** | TGS drops, recovers, then drops again (possibly deeper). Multiple distinct performance levels. | Multiple nodes with RDMA issues taking turns — one node's link degrades, recovers, then a different node degrades. |
| **Gradual increase in step time** | Step time slowly grows over many steps (not a sudden jump). | Resource leak — CPU contention from leaked RCCL communicator threads (see "Checkpointing interference" in TSDB diagnosis skill), memory pressure, or accumulating background processes. |
| **Periodic spikes** | Step time spikes every N steps, then returns to normal. | Checkpoint saves (match N against `checkpoint_period`). Expected behavior — not a degradation. |
| **One-time drop** | TGS drops once and never recovers. | Hardware event — a NIC, cable, or GPU partially failed at that moment. |

**Step 3: Check for known log-level causes.**

Before escalating to TSDB, check the worker logs for:
- **Checkpoint restore:** Was this job restored from a checkpoint (`restoring from this run's directory step N`)? Restored jobs can leak RCCL communicator threads — see "Checkpointing interference" in `skills/tsdb-diagnosis/SKILL.md`. Compare TGS against a known-good fresh-start run with the same config.
- **XLA recompilation:** Look for repeated `Compiling module` messages after the initial compilation. Unexpected recompilation mid-training causes step time spikes.
- **Profiler hooks:** If `profiler=xplane` is set with a specific step range, the profiled steps may be slower. Check if the TGS drop coincides with the profiler step range.

**Step 4: Escalate to TSDB for root cause.**

For `RAY=1` jobs, the Prometheus TSDB is the definitive tool for diagnosing TGS degradation. Report the TGS pattern (from step 2) and recommend:
- **Constant or phased drops:** TSDB Playbook 6 (Network Health) — query RDMA retransmits per host during the drop window. See the RDMA degradation signature and phase correlation technique in the TSDB skill.
- **Gradual increase:** TSDB Playbook 7 (Training Stability) contention checklist — check `hw_procs_running`, memory pressure, I/O pressure.
- **One-time drop:** TSDB Playbook 4 (Hardware) + Playbook 6 (Network) — check for RAS errors, thermal events, and network state changes at the drop timestamp.

**Including TGS analysis in the triage report:** When TGS data is available, add a `### TGS analysis` section to the triage report showing the steady-state average, any degradation pattern detected, the affected step range, and the magnitude of the drop (e.g., "3290 → 2520, -23%"). This information is critical for multi-job comparisons and for deciding whether a "successful" job actually needs investigation.

## Diagnosing "unknown-death" (no JOB SUMMARY)

When a job has no JOB SUMMARY and is no longer running:

1. **Check for OOM-kill signatures** — `Killed`, `oom-kill`, `Out of memory` near the end of available output.
2. **Check for SIGKILL** — abrupt log cutoff mid-line, no cleanup messages.
3. **Check for Slurm preemption** — `scontrol show job` (if Slurm ID is known) may show `State=PREEMPTED` or `State=TIMEOUT`.
4. **Check dmesg** (if accessible) — `dmesg -T | grep -i "oom\|killed process"` on the compute node. If SSH is unavailable but the job was `RAY=1` and the Ray cluster is still reachable, use the **Ray Jobs API** to read dmesg remotely (see `skills/tsdb-diagnosis/SKILL.md` → "Remote Execution via Ray Jobs API"). For finished jobs where Ray is gone, dmesg is only accessible via SSH or out-of-band node access.
5. If none of the above yields an answer, report as "unknown-death — process killed externally without writing JOB SUMMARY. Most common cause: host OOM-kill or Slurm preemption."

## Diagnosing node failures

When `NODE_EXIT host=<hostname> exit=<rc>` appears:

1. Note which nodes failed and their exit codes.
2. Search for error output from those specific task IDs (lines prefixed with `<task_id>:`).
3. Common patterns:
   - Exit 137 (128+9) = SIGKILL (OOM or external kill)
   - Exit 134 (128+6) = SIGABRT (assertion failure, core dump)
   - Exit 139 (128+11) = SIGSEGV (crash, core dump likely in job dir)
   - Exit 1 = generic error (read task output for details)
4. For `RAY=1` jobs where the job is still live, use the **Ray Jobs API** to inspect the failed node remotely — check dmesg for GPU errors, OOM kills, or driver faults. For hardware-related node failures (exit 137 with no OOM in logs, or exit 134/139), recommend TSDB diagnosis (Playbook 4: Node Failure / Hardware) for RAS error counters and thermal data.

## Output format

Report findings in this structure:

```
## Job triage: <log_file_path>

**Status:** <completed | failed | cancelled | running | hanging | unknown>
**Primary failure:** <class name from table above>
**Stage:** <which stage failed, if identifiable>

### What happened
<1–3 sentence plain-English explanation>

### Evidence
<Relevant log lines, quoted verbatim with line context>

### Training progress projection
| Metric | Value |
|--------|-------|
| Start | fresh / restored from ckpt <N> |
| Steps completed (this run) | <last - first + 1> steps (step <first> through <last>) |
| Overall progress | <last+1> / <total> (<pct>%) |
| Steady-state step time | <X>s |
| Expected step by now | <N> (based on step time and run start) |
| Last periodic ckpt this run | step <N> (or "none — didn't reach checkpoint_period") |
| Progress lost | <N> steps (<time>) since last periodic ckpt (or "all — no periodic ckpt") |
| Estimated time remaining | <time> (from last step, if job were healthy) |

### Additional findings
<Any secondary signatures found (e.g., warnings, non-fatal errors)>

### Live dashboards (RAY=1 jobs only)
<If SSH tunnel command found in log, show it here>
<If job is still live (running or hanging), ask:
 "Want me to set up port forwarding so I can access the Ray Dashboard / Prometheus / TensorBoard?">

### Recommended next steps
<Numbered list of specific actions>
```

### Next-step templates by failure class

| Class | Recommended next steps |
|-------|----------------------|
| **container-pull-fail** | 1. Verify image name in `container_env.sh`. 2. Check registry credentials. 3. Try manual `docker pull`. |
| **no-gpu** | 1. Check that the node has GPUs (`rocm-smi` or `nvidia-smi`). 2. Check container device flags in `_container.sh`. |
| **nccl-nic-fail** | 1. Run `ip link show` on compute nodes to verify NIC availability. 2. Check `choose_nccl_socket_ifname.sh` logic. |
| **model-not-found** | 1. List available configs: `ls configs/*.gpu.yml`. 2. Create a new config per `docs/model-configs.md`. |
| **preflight-timeout** | 1. Check for stuck GPU processes: `rocm-smi` / `nvidia-smi`. 2. Increase timeout: `STAGE_TIMEOUTS="preflight:1800"`. |
| **pull-timeout** | 1. Check network/registry speed. 2. Pre-pull the image. 3. Increase timeout: `STAGE_TIMEOUTS="pull:1800"`. |
| **train-timeout** | 1. Increase timeout: `STAGE_TIMEOUTS="train:<seconds>"`. 2. Verify it's not a hang (check if training was progressing). |
| **hang** | 1. If `RAY=1` and job is still live: check Ray Dashboard (port 8265) for actor status and stack traces to confirm RCCL hang and identify the stuck collective. 2. Query Prometheus TSDB at hang time for GPU util (check per-GPU per-node for partial utilization patterns), power watts, TCP retransmits (both rate and absolute totals across nodes), RDMA counters. 3. Kill the job: `scancel <id>`. 4. **Retry on the same nodes first** — init-phase hangs (before step 0) are often transient RCCL race conditions. If the retry succeeds, no further action needed; note TCP retransmit outliers for awareness only. 5. Only if the hang **recurs**: exclude suspect nodes (`--exclude=<nodes>` targeting TCP retransmit or GPU pattern outliers), add `_env_NCCL_DEBUG=INFO`, and use `slurm_job_monitor.sh -j <id>` for early detection. |
| **heartbeat-timeout** | 1. Known bug — almost certainly a false positive (see `docs/jax-heartbeat-false-positive-postmortem.md`). 2. Increase `jax_distributed_heartbeat_timeout_seconds` to several hours (e.g., 14400). 3. Use `slurm_job_monitor.sh` for independent hang detection. 4. Follow the heartbeat diagnosis checklist above to confirm. |
| **oom-host** | 1. Reduce `per_device_batch_size`. 2. Enable `remat_policy=full`. 3. Check for checkpoint memory spike (DP replica #0 pattern). |
| **oom-gpu** | **First check `XLA_PYTHON_CLIENT_MEM_FRACTION`** — see GPU OOM diagnosis below. If the fraction is too low for the model size, increase it (e.g., `.85` → `.93`). Only after ruling that out: 1. Reduce `per_device_batch_size` or `max_target_length`. 2. Try `remat_policy=full`. 3. Check XLA buffer assignment for memory usage. |
| **nccl-timeout** | 1. Check network health (`ip link`, `ethtool`, RDMA counters). 2. Run with `_env_NCCL_DEBUG=INFO` for detailed NCCL logs. 3. Check if specific nodes are consistently failing. |
| **xla-compile-fail** | 1. Check XLA flags in `train_env.sh` for conflicting settings. 2. Try `_env_ENABLE_XLA_DUMP=1` to capture the failing HLO. 3. Reduce model complexity to isolate the issue. |
| **python-exception** | 1. Read the full traceback. 2. Check if it's a known MaxText issue. 3. Verify config parameters. |
| **signal-kill** | 1. Check for core dumps in the coredump path candidates: `<job_dir>/core*`, `<outputs_root>/core*`, and paths from `COREDUMP_EXTRA_DIRS` in `container_env.sh`. 2. Inspect with `gdb python3 <core_file>` inside the Docker container. 3. If logs show `NCCL_DMABUF_ENABLE=1`, check for the runtime warning `Forcing NCCL_DMABUF_ENABLE=0` (missing `/boot` kernel metadata safeguard). If that warning is absent on older commits, verify `/boot/*$(uname -r)*` availability in the container. 4. See `docs/debugging.md`. |
| **cancelled** | Cancellation is the mechanism, not the root cause. 1. Check training progress projection — if the last step is far behind the expected step, the real issue is a **hang** killed by scancel. 2. Check for preceding errors (NCCL, OOM, heartbeat). Report the underlying cause as primary failure. If no underlying issue, no action needed. |
| **node-fail** | 1. Identify which nodes failed. 2. Read their task output. 3. For exit 137: likely OOM. For exit 134/139: check core dumps. |
| **unknown-death** | 1. Check `dmesg` for OOM kills. 2. Check Slurm state: `scontrol show job <id>`. 3. If recurring: run with `RAY=1` for TSDB diagnostics. |
| **tgs-degradation** | 1. Extract TGS timeline from worker logs (see TGS degradation diagnosis above). 2. Identify the degradation pattern (constant drop, phased, gradual, periodic). 3. For `RAY=1` jobs: query TSDB — Playbook 6 (Network Health) for RDMA retransmits per host, Playbook 7 (Training Stability) contention checklist. **Constant drops point to RDMA issues; gradual increases point to resource leaks.** 4. Identify the offending nodes from per-host RDMA retransmit rates. 5. Resubmit with `--exclude=<bad_nodes>` targeting nodes with RDMA retry exhaustion or sustained RDMA retransmits (see node exclusion prioritization in TSDB skill). |
| **checkpoint-fs-error** | 1. **Check other workers' logs first** — the checkpoint may be intact despite the error (see checkpoint filesystem error diagnosis below). 2. If checkpoint is intact: resubmit restoring from that checkpoint; zero progress lost. 3. If checkpoint is corrupt: resubmit restoring from the previous periodic checkpoint. 4. For `RAY=1` jobs: query TSDB for I/O pressure (`hw_io_pressure_full_pct`) and TCP retransmits (`rate(hw_tcp_retransmits_total[5m])`) on checkpoint-writing nodes during the failure window. Compare against previous successful checkpoints to identify NFS degradation. 5. Check VAST/NFS storage health. |
| **ray-start-fail** | Non-critical — training falls back to non-Ray mode. If observability is needed: 1. Check port conflicts. 2. Check Ray logs in job dir. |

## Known-harmless log entries

These patterns appear in normal, healthy jobs. Do **not** classify them as failures or mention them in the triage report:

| Pattern | Why it's harmless |
|---------|-------------------|
| `Failed call to cuInit: UNKNOWN ERROR (303)`, `INTERNAL: CUDA error` | JAX/XLA probes for CUDA on AMD GPU nodes. The probe fails (expected) and falls back to ROCm. Appears in every job. |
| `NCCL WARN MSCCL++: Feature not enabled` | RCCL init notice — MSCCL++ is a compile-time feature not enabled in the current build. Appears on every RCCL job. |
| `Token indices sequence length is longer than the specified maximum sequence length` | HuggingFace tokenizer truncation warning. The model handles this internally; not an error. |
| `OCI runtime exec failed` + `[exec] docker exec failed ... falling back to host-level kill` + `[cgroup] Sent SIGKILL to 0/0 processes` | Preflight cleanup killing stale containers from a previous job. The `0/0 processes` confirms there was nothing left to kill. |
| `OCI runtime exec failed: exec failed: unable to start container process: error executing setns process: exit status 1: unknown` (standalone, during teardown) | Container namespace teardown race — the container exited before Docker could exec into it for cleanup. Common during job cancellation or when containers shut down quickly. No data loss or training impact. |
| `Cannot read CPU core N` (topology.cc) | XLA/ROCm topology probe on cores outside the container's cgroup. Harmless. |
| `No hardware is found. Using default TPU version: jellyfish` | XLA probes for TPU on a GPU node. Expected, falls back to GPU. |
| `No device identifiers found` (trace.cc) | XLA tracing probe. Harmless. |
| `Enabling PjRt/TPU event dependency logging` | XLA internal logging init. Harmless on GPU nodes. |
| `Fiber init: default domain = futex` (init-domain.cc) | Internal threading init. Harmless. |
| `Error response from daemon: cannot remove container ... could not kill container: tried to kill container, but did not receive an exit event` | Docker container slow to exit during teardown (e.g., stuck in RCCL busy-wait). Harmless — cleanup completes eventually. |
| `srun: error: <host>: task N: Exited with exit code 143` + `srun: Terminating StepId=` | Normal Slurm cascade after `scancel`. Exit 143 = SIGTERM. All nodes exiting with 143 confirms a clean cancellation. |
| `NODE_EXIT host=<hostname> exit=143` (all nodes) | Clean SIGTERM on every node — expected from `scancel`. Not an error. |

## Multi-failure jobs

Some failures cascade. When multiple signatures are found:

1. **Report all of them** in the "Additional findings" section.
2. **Identify the root cause** — the earliest error in the log is usually the primary failure. Later errors (heartbeat timeouts, node exits) are often consequences.
3. **Common cascades:**
   - OOM on one node → NCCL timeout on other nodes (waiting for the dead node) → heartbeat timeout
   - NCCL network error → all-reduce hang → training timeout or heartbeat timeout
   - One node dies silently → remaining nodes hang on the next collective (training steps stop, no error)
   - XLA compilation failure → Python exception → subprocess exit code 1
   - **RCCL hang (all nodes spinning in busy-wait) → gRPC channel deadlock on one task after extended hang (30+ min) → heartbeat timeout declares that task dead → all tasks killed.** The heartbeat timeout is the *kill mechanism*, not the root cause. The RCCL hang is the primary failure. This cascade is the most commonly misdiagnosed — the heartbeat error is prominent in the log (all 24 tasks report it), while the hang leaves no log signature (training simply stops advancing). Always check TSDB GPU power/utilization before the heartbeat error to detect this.
