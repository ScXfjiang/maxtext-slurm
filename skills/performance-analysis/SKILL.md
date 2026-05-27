---
name: performance-analysis
description: Analyze MaxText training job performance using tgs_tagger, TraceLens, and IRLens. Use when the user asks to analyze a training run, profile traces, HLO IR, TGS metrics, GPU utilization, or mentions tag_tgs, TraceLens, IRLens, xplane, or performance analysis.
---

# MaxText Performance Analysis

Post-training (or mid-training) analysis pipeline. Follow the workflow below from top to bottom.

**Multi-job comparisons:** If comparing two or more jobs (e.g., "why is job B slower than job A?"), start with `skills/tsdb-diagnosis/SKILL.md` (Multi-Job Comparison workflow) **before** running TraceLens. The TSDB reveals system-level root causes — CPU contention from RCCL resource leaks, network errors, I/O pressure, thermal throttling — that TraceLens cannot observe (it only sees GPU-side kernel timings). Only proceed to TraceLens here if the TSDB comparison is inconclusive.

**Deep per-kernel analysis:** When the user asks for per-kernel time breakdowns, step-time composition tables, cross-variant kernel comparisons, or whether a specific kernel is main-stream-blocking — switch to `skills/profile-drill/SKILL.md`. TraceLens's `kernel_launchers_summary_by_category.csv` has a known ~1.5×–2× inflation bug on 1-node/proc profiles (the `time ms per gpu` column divides by host count, not GPU count). `profile-drill` uses `utils/profile_drill.py` to read the raw xplane trace JSONs directly and avoids this bias.

## Workflow

### Step 1: Run the dispatcher

```bash
python3 utils/analyze_job.py "$JOB_WORKSPACE/<job>.log"
python3 utils/analyze_job.py "$JOB_WORKSPACE/<job_dir>/"
python3 utils/analyze_job.py "$JOB_WORKSPACE/local_2026*"
```

For running jobs, pass `-f` to force re-analysis (bypasses staleness check):

```bash
python3 utils/analyze_job.py -f "$JOB_WORKSPACE/<job>.log"
```

The dispatcher auto-detects available artifacts and runs only the relevant tools:
- **Log with TGS data** → `tgs_tagger.py`
- **`*.xplane.pb`** → `TraceLens_generate_perf_report_jax`
- **`xla_dump/*.gpu_after_optimizations.txt`** → `IRLens_analyze_hlo_ir.py`

### Step 2: Handle TraceLens if needed

If the dispatcher output says **"TraceLens not installed"** and xplane traces exist:

1. **Check if TraceLens is already installed and patched** before doing anything:
   ```bash
   python3 -c "
   import TraceLens.util, inspect
   src = inspect.getsource(TraceLens.util.DataLoader.load_data)
   assert 'xprof' in src, 'not patched'
   print('TraceLens: installed and patched')
   "
   ```
   - **Succeeds** → TraceLens is ready. Just re-run: `python3 utils/analyze_job.py -f "$JOB_WORKSPACE/<job>.log"`
   - **ImportError** → not installed. Install then patch (see below).
   - **AssertionError** → installed but unpatched. Patch only (see below).

2. **Install** (only if import failed):
   ```bash
   pip install git+https://github.com/AMD-AGI/TraceLens.git
   ```

3. **Patch** (only if the `xprof` assertion failed). Apply all patches from [tracelens-patches.md](tracelens-patches.md) — 6 files, ~13 patches. Key fixes:
   - protobuf/xprof import errors (TF 2.19+ renamed `tensorboard_plugin_profile` to `xprof`)
   - GPU PID remapping (`xprof` remaps device PIDs to 1001+; code filtering `pid < 100` misses all GPU events)
   - `metadata_events` not passed to `build_tree()`
   - `KeyError` on `gpu_kernel_op_cat` and missing parent events for launch latency

4. **Re-run** the dispatcher with `-f`:
   ```bash
   python3 utils/analyze_job.py -f "$JOB_WORKSPACE/<job>.log"
   ```

This is one-time per environment. Always check before patching to avoid redundant work.

### Step 3: Read results

Read the generated `analysis.json` — but do NOT try to read the raw file (it can be 40K+ lines due to per-step arrays). Extract key metrics programmatically:

```bash
python3 -c "
import json, sys
with open('<job_dir>/analysis.json') as f:
    d = json.load(f)
print(f'Job: {d[\"job_id\"]} | Model: {d[\"model\"]} | Nodes: {d[\"num_nodes\"]} | Status: {d[\"job_status\"][\"status\"]}')
tgs = d['tgs']
print(f'Steady TGS: {tgs[\"steady\"][\"mean\"]:.1f} (std={tgs[\"steady\"][\"std\"]:.1f}, steps {tgs[\"steady\"][\"range\"]})')
print(f'Tail   TGS: {tgs[\"tail\"][\"mean\"]:.1f} (std={tgs[\"tail\"][\"std\"]:.1f}, steps {tgs[\"tail\"][\"range\"]})')
tl = d.get('tracelens_summary', {})
if tl:
    print(f'Compute: {tl[\"computation_time\"]:.1f}% | Exposed comm: {tl[\"exposed_comm_time\"]:.1f}% | Idle: {tl[\"idle_time\"]:.2f}% | Total comm: {tl[\"total_comm_time\"]:.1f}%')
"
```

For deeper TraceLens analysis, read the CSVs in `<job_dir>/tracelens/<timestamp>/csvs/`:
- `gpu_events_averages.csv` — per-GPU compute/comm/idle breakdown (averages)
- `gpu_timeline.csv` — per-GPU breakdown with pid
- `kernel_launchers_summary_by_category.csv` — time by kernel category (GEMM, NCCL, XLA fusions, etc.)
- `kernel_launchers_summary.csv` — time by individual kernel name

> ⚠️ **TraceLens per-GPU CSV bias on 1-node/proc.** The `time ms per gpu` column in the two `kernel_launchers_summary*.csv` files divides total kernel time by **host count** (typically 8), not GPU count (typically 64) — so per-GPU numbers are ~1.5×–2× inflated on 1-node/proc profiles.  Percentages and category rankings are fine; absolute per-GPU kernel times are not.  For kernel-time numbers you can cite (e.g. in a report or step-time composition table), use `skills/profile-drill/SKILL.md` instead — it reads raw xplane trace JSONs and divides by auto-detected GPUs.

### Step 4: Summarize findings

Present results using this structure:

| Metric | Source | What to look for |
|--------|--------|------------------|
| **TGS** (steady-state) | `analysis.json` → `tgs.steady` | Primary throughput metric |
| **MFU** | `analysis.json` → `mfu_per_step` | Model FLOPS utilization (if available) |
| **GPU compute %** | `tracelens_summary.computation_time` | Time on actual compute kernels |
| **Exposed comm %** | `tracelens_summary.exposed_comm_time` | Communication NOT overlapped with compute (lower is better) |
| **Idle %** | `tracelens_summary.idle_time` | GPU doing nothing (should be near 0) |
| **Kernel breakdown** | `kernel_launchers_summary_by_category.csv` | GEMM vs NCCL vs fusion time |
| **Comm ops per step** | dispatcher IRLens output | Count of all-reduce, all-gather, all-to-all, reduce-scatter |

Interpretation guidelines:
- High exposed comm % → opportunities for better comm/compute overlap
- Large per-GPU variance in compute % → load imbalance
- High idle % → scheduling or synchronization issues
- Tail TGS std much larger than steady std → periodic overhead (checkpointing, profiling)

### Step 5: Ensure dashboard is running

**Check the dispatcher output first** — it prints a `Dashboard:` line at the end. If it shows a URL with `(running)`, use that URL.

If the dashboard is not running, start it:

```bash
pip install fastapi uvicorn   # one-time
utils/perf_server.py &        # binds 127.0.0.1 by default
```

**Always tell the user the dashboard URL:** `http://localhost:<PORT>`. For remote access, instruct them to tunnel: `ssh -L <PORT>:localhost:<PORT> user@host`. Avoid `--host 0.0.0.0` — `perf_server.py` has no auth.

The server auto-detects a free port starting from 8080 and auto-reloads `analysis.json` on each request.

## Reference

### Job output layout

```
<JOB_WORKSPACE>/<JOB_ID>-<JOB_NAME>[-TGS_<VALUE>]/
  log -> ../<log_file>                          # symlink to log file
  analysis.json                                 # structured metrics
  xla_dump/                                     # if _env_ENABLE_XLA_DUMP=1
    module_NNNN.jit_train_step.*_gpu_after_optimizations.txt
  <run_name>/tensorboard/plugins/profile/<ts>/  # if profiler=xplane
    <hostname>.xplane.pb                        #   1-node/proc: one per host
  <run_name>/tensorboard/plugins/profile/<ts_i>/ # 1-GPU/proc (LOCAL_WORLD_SIZE ts dirs,
    <hostname>.proc<N>.xplane.pb                #   one file per host per ts;
                                                #   successive serialized writes land
                                                #   in different per-second ts dirs)
  tracelens/<ts>/csvs/*.csv                     # 1-node/proc: TraceLens output
  tracelens/<ts_i>/<hostname>.proc<N>/csvs/*.csv # 1-GPU/proc: one dir per GPU
```

The `.log` file sits alongside the directory in `<JOB_WORKSPACE>/`.

When `enable_checkpointing=true`, profiler traces may end up in a shared directory outside the job dir. `analyze_job.py` parses `Config param tensorboard_dir` from the log to locate these. The dispatcher and `perf_server.py` filter profiles by job execution time window and node-0 hostname to disambiguate. In 1-GPU-per-process mode the node-0 filter `name.startswith("<host>.")` still matches all `<host>.proc<N>.xplane.pb` files, so TraceLens runs once per GPU on node 0; the multiple timestamp dirs (one per serialized write) are treated like periodic-profiling windows by the existing code.

### Running individual tools directly

These are rarely needed — `analyze_job.py` orchestrates them. Use only for targeted re-runs.

```bash
# TGS tagging
utils/tag_tgs.sh <log_file_or_glob>
utils/tag_tgs.sh -f <log_file>       # force on running job

# IRLens
utils/IRLens_analyze_hlo_ir.py <hlo_file>
utils/IRLens_analyze_hlo_ir.py <hlo_file> --op communication
utils/IRLens_analyze_hlo_ir.py <hlo_file> --op computation

# TraceLens
TraceLens_generate_perf_report_jax \
    --profile_path <xplane.pb> \
    --output_csvs_dir <output_dir>/csvs

# profile_drill.py — direct per-kernel analysis from trace JSONs
# (use when TraceLens's per-GPU numbers are suspect or you need kernel-level
# ground truth; see skills/profile-drill/SKILL.md)
utils/profile_drill.py <job_dir>/.../tensorboard/plugins/profile/*/*.trace.json.gz
```

### `RAY=1` Slurm log truncation

For `RAY=1` jobs, the Slurm log may contain **fewer training steps than actually completed** due to Ray output buffering (actor stdout is forwarded asynchronously to the driver, and unflushed output is lost when the job exits). If the analysis shows suspiciously few steps (e.g., 34 out of 100) with no error or JOB SUMMARY, check `ray_logs/<head_node>/worker*.out` in the job directory for the authoritative step count. The `analysis.json` TGS/MFU metrics will be based only on what appears in the Slurm log and may undercount the actual run.

### Running jobs

- The dispatcher detects running jobs via the `JOB SUMMARY` log marker and file modification time (15 min threshold).
- `analyze_job.py -f` bypasses the staleness check but never renames files for running jobs. Renames happen automatically on the next analysis after the job finishes.
- TraceLens needs a completed profiler trace; skipped if `*.xplane.pb` doesn't exist yet.
- IRLens works on running jobs if `xla_dump/` is already populated.
