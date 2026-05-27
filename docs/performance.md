# Performance

Profile training runs to find bottlenecks, analyze the results, then tune. All profiling and tuning settings live in `train_env.sh` and model config YAMLs. For system-level and training metrics (GPU thermals, power, VRAM, loss curves, throughput), see [Observability](observability.md).

## Profiling

### Runtime traces (xplane)

The [JAX profiler](https://docs.jax.dev/en/latest/profiling.html) captures GPU kernel timelines in xplane format. Enable it via CLI or in the model config:

```bash
submit.sh 70b -N 1 -- profiler=xplane skip_first_n_steps_for_profiler=3 profiler_steps=1
```

```yaml
profiler: "xplane"
upload_all_profiler_results: true
skip_first_n_steps_for_profiler: 3
profiler_steps: 1
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `profiler` | `""` (disabled) | Set to `"xplane"` to enable |
| `skip_first_n_steps_for_profiler` | `3` | Warmup steps to skip (avoids profiling compilation) |
| `profiler_steps` | `1` | Number of steps to profile |
| `upload_all_profiler_results` | `true` | Collect traces from all nodes (not just rank 0) |

Trace files are written to `base_output_directory` (i.e. `OUTPUT_PATH`).

**1-GPU-per-process mode.** The JAX xplane writer names output files by `socket.gethostname()` alone, so when multiple processes share a host (see [1-GPU-per-process mode](job-submission.md#1-gpu-per-process-mode)) they'd race on the same `<host>.xplane.pb`. `monkey_patch_maxtext.py` wraps `jax.profiler.stop_trace` with a host-scoped `flock` that serializes the write and tags each process's output as `<host>.proc<LOCAL_RANK>.xplane.pb` (with matching `.trace.json.gz` and `.SSTABLE`). Serialization makes successive writes from the same host land in different per-second timestamp directories, so one job ends up with `LOCAL_WORLD_SIZE` timestamp dirs of flat per-host files — `analyze_job.py` already treats multiple ts dirs as periodic-profiling windows and handles them transparently. Filenames still start with `<host>.`, so the node-0 filter, `merge_xplane_traces.py`, TraceLens, and IRLens keep working unchanged.

### HLO IR dump

[XLA](https://openxla.org/)'s JIT compiler transforms JAX code into [HLO (High Level Operations)](https://openxla.org/xla/operation_semantics) IR before generating GPU kernels. HLO dumps capture the compiled computation graph — which collectives are fused, how loops are structured, and what kernels will execute each step — independent of actual execution timing.

Enable XLA dump for any run by passing `_env_ENABLE_XLA_DUMP=1`:

```bash
submit.sh           70b -N 1 -- steps=1 _env_ENABLE_XLA_DUMP=1
run_local.sh        70b      -- steps=1 _env_ENABLE_XLA_DUMP=1
in_container_run.sh 70b      -- steps=1 _env_ENABLE_XLA_DUMP=1
```

This appends the following XLA flags (configured in `train_env.sh`):

| Flag | Purpose |
|------|---------|
| `--xla_dump_hlo_as_text` | Emit human-readable `.txt` HLO files |
| `--xla_dump_hlo_module_re` | Regex filter for HLO modules (`^jit_train_step$` captures the main training step) |
| `--xla_dump_hlo_pipeline_re` | Regex filter for compiler passes (`(?i)gpu` captures GPU-specific passes) |
| `--xla_dump_to` | Output directory for dump files |

Dump files are written to `<OUTPUT_PATH>/xla_dump/`. The dump is scoped to global rank 0 only, because SPMD produces identical HLO on every rank and concurrent writers would race on filenames (each process has its own XLA module counter that aligns across ranks). Key files:

| File | Contents |
|------|----------|
| `*.before_optimizations.txt` | HLO before GPU compiler passes |
| `*.gpu_after_optimizations.txt` | Final optimized HLO that maps to GPU kernels |
| `*.gpu_after_optimizations-buffer-assignment.txt` | Memory buffer layout |
| `*.gpu_after_optimizations-memory-usage-report.txt` | Memory usage breakdown |
| `*.ir-with-opt.ll` | Optimized LLVM IR |
| `*.thunk_sequence.txt` | Runtime execution schedule |

## Analysis

### Visualize traces

Open trace files in [Perfetto](https://ui.perfetto.dev/) (recommended for large files) or `chrome://tracing`. Multi-node jobs produce one trace file per host by default, or `LOCAL_WORLD_SIZE` per host (one per local GPU, named `<host>.proc<N>.trace.json.gz`) in [1-GPU-per-process mode](job-submission.md#1-gpu-per-process-mode). Each can be viewed individually, or merged into one file for side-by-side viewing:

```bash
utils/merge_xplane_traces.py "$JOB_WORKSPACE/<job>/"
utils/merge_xplane_traces.py node0.trace.json.gz node1.trace.json.gz -o combined.trace.json.gz
```

### TraceLens (runtime trace analysis)

[TraceLens](https://github.com/AMD-AGI/TraceLens) automates analysis of runtime trace files — GPU kernel utilization, GEMM performance, and communication patterns — from the xplane protobuf or JSON traces produced by the profiler.

```bash
pip install git+https://github.com/AMD-AGI/TraceLens.git
```

> **Note:** In environments with TensorFlow 2.19+ / `xprof`, TraceLens requires source patches before it will run. See [tracelens-patches.md](../skills/performance-analysis/tracelens-patches.md) for details.

Generate a full performance report from an xplane trace:

```bash
TraceLens_generate_perf_report_jax \
    --profile_path <job_dir>/**/plugins/profile/<timestamp>/*.xplane.pb \
    --output_xlsx_path <job_dir>/tracelens/<timestamp>/report.xlsx \
    --output_csvs_dir <job_dir>/tracelens/<timestamp>/csvs
```

When periodic profiling is enabled (`profile_periodically_period > 0`), each profiling window produces a separate xplane trace under its own timestamp directory. `analyze_job.py` handles this automatically — giving each profile its own TraceLens output directory and skipping profiles that have already been analyzed.

In multi-node jobs, `analyze_job.py` selects **node 0's xplane** (the coordinator, identified from `SLURM_JOB_NODELIST` or `JOB_NODELIST` in the log) for analysis. Since SPMD training runs the same program on every host, node 0 is representative. Distributed profiling may write host traces across multiple timestamp directories a few seconds apart — filtering by node 0 naturally deduplicates to one trace per profiling step.

When profiles live in a shared directory (`enable_checkpointing=true`), multiple jobs may contribute timestamps to the same location. To disambiguate, both `analyze_job.py` and `perf_server.py` apply a **time-window filter** — profile timestamps must fall within the job's execution window (artifact directory creation time → log file modification time) — in addition to node-0 filtering. Sibling timestamp directories within 60 seconds of a matched anchor are included as part of the same profiling session (nodes start profiling seconds apart). This correctly handles node reuse across jobs and periodic profiling (`profile_periodically_period > 0`).

Key output columns in `csvs/gpu_events_averages.csv`:

| Column | Meaning |
|--------|---------|
| `computation_time` | Time on compute kernels (GEMMs, fusions, attention) |
| `exposed_comm_time` | Communication NOT overlapped with compute (lower is better) |
| `idle_time` | GPU idle (should be near 0) |
| `total_comm_time` | All communication including the portion overlapped with compute |

See the [JAX analysis guide](https://github.com/AMD-AGI/TraceLens/blob/main/docs/jax_analyses.md) for the full TraceLens API.

### IRLens (HLO static analysis)

`IRLens_analyze_hlo_ir.py` extracts a hierarchical execution skeleton from an HLO dump — the control flow, communication ops, and computation ops with all plumbing stripped away:

```bash
# Show all ops (communication + computation)
utils/IRLens_analyze_hlo_ir.py xla_dump/module_0000.jit_train_step.gpu_after_optimizations.txt

# Communication ops only — all-gather, reduce-scatter, all-reduce patterns
utils/IRLens_analyze_hlo_ir.py <hlo_file> --op communication

# Computation ops only — fusion kernels, GEMM calls
utils/IRLens_analyze_hlo_ir.py <hlo_file> --op computation

# Full detail: variable names, communication topology, fusion breakdown
utils/IRLens_analyze_hlo_ir.py <hlo_file> --name --topology --fusion-stats
```

| Flag | Description |
|------|-------------|
| `--op {all,communication,computation}` | Filter by operation type (default: `all`) |
| `--name` | Show HLO result variable names and while-loop names |
| `--topology` | Show communication topology (replica groups, source-target pairs) |
| `--fusion-stats` | Break down fusion ops by subtype in the statistics summary |

**Example output** (communication only):

```
ENTRY %main:
  while i in range(8):
    all-gather-start | bf16[8,8192,8192] | jit(train_step)/... | layers.py:142
    reduce-scatter-start | bf16[8,8192,8192] | jit(train_step)/... | layers.py:218
    all-reduce-start | f32[1] | jit(train_step)/... | pipeline.py:95

# Total communication ops: 24
# 3 communication op categories:
#   all-gather-start                          8
#   all-reduce-start                          8
#   reduce-scatter-start                      8
```

Useful for verifying that sharding strategies (FSDP, tensor parallelism) produce the expected collective patterns and for spotting unexpected communication overhead.

## Tuning

All tunable settings live in `train_env.sh`. Per-run overrides can be passed via `_env_KEY=VALUE` after `--` without editing the file (see [Job Submission](job-submission.md)).

### XLA compiler flags

`train_env.sh` contains a curated set of `XLA_FLAGS` for performance tuning, organized by category. The entire block is commented out by default so the Docker image's built-in `XLA_FLAGS` are used as-is. To customize, uncomment the block and adjust individual flags:

| Category | Flags | What they control |
|----------|-------|-------------------|
| Core compiler | `xla_gpu_enable_cublaslt`, `xla_gpu_graph_level`, `xla_gpu_autotune_level` | GEMM library selection, graph capture, autotuning |
| GEMM / codegen | `xla_gpu_enable_triton_gemm`, `xla_gpu_enable_command_buffer` | Triton vs vendor GEMM kernels, command buffer use |
| Collective combining | `xla_gpu_all_gather_combine_threshold_bytes`, `xla_gpu_reduce_scatter_combine_threshold_bytes`, etc. | How aggressively XLA merges small collectives into larger ones (higher thresholds = more combining, but more memory) |
| Overlapping / pipelining | `xla_gpu_enable_latency_hiding_scheduler`, `xla_gpu_enable_pipelined_all_gather`, `xla_gpu_enable_pipelined_reduce_scatter`, `xla_gpu_enable_while_loop_double_buffering` | Overlap compute with communication to hide latency |

Some flags have model-specific trade-offs noted in the comments (e.g. double buffering may OOM on very large models even with adjusted combine thresholds). Start from the defaults, profile, then selectively enable flags based on what the traces and HLO analysis reveal.

### Environment variables

The rest of `train_env.sh` exports environment variables that control JAX, NCCL/RCCL, and vendor-specific libraries (memory, networking, InfiniBand, GPU compute, AMD/ROCm, Transformer Engine, GDR/DMA, compilation cache, etc.). The file is organized into commented sections with inline notes on trade-offs. Most defaults work well out of the box.

## Typical workflow

1. **Baseline run** — train with `steps=15` (minimum), `profiler=xplane`, and `_env_ENABLE_XLA_DUMP=1`.
2. **Analyze** — run `utils/analyze_job.py` on the job output to automatically detect artifacts and dispatch the appropriate tools:

```bash
utils/analyze_job.py outputs/<job>.log          # finished job
utils/analyze_job.py -f outputs/<job>.log       # force re-analysis
```

This runs whichever of the following are applicable:
  - **Tag TGS** — `tgs_tagger.py` extracts steady-state throughput (steps 5-14) and renames outputs. For running jobs, TGS is computed but renames are deferred until the job finishes.
  - **Analyze traces** — TraceLens for GPU utilization breakdown (compute vs communication vs idle). Only node 0's trace is analyzed (representative in SPMD); each profiling window gets its own output directory; already-analyzed profiles are skipped on re-runs. When checkpointing redirects profiler output to a shared directory, `analyze_job.py` parses `tensorboard_dir` from the log to locate the traces and applies time-window + node-0 filtering to disambiguate profiles from different jobs.
  - **Analyze HLO** — IRLens to inspect the compiled computation graph (collective patterns, fusion structure).

The dispatcher is safe to re-run. It records `job_status` (completed / failed / cancelled / running / unknown) in `analysis.json` and uses it for staleness detection: finished jobs with up-to-date analysis are skipped automatically, while running or stale jobs are re-analyzed. Pass `-f` to force re-analysis regardless (staleness check only; does not force renames).

3. **Browse results** — start the dashboard to visualize, compare, and download analysis results:

```bash
utils/perf_server.py                             # localhost only, auto-picks port from 8080
utils/perf_server.py --port 8080                 # localhost only, explicit port
```

The default bind is `127.0.0.1`. To reach the dashboard from a different machine, prefer `ssh -L 8080:localhost:8080 user@analysis-host` over `--host 0.0.0.0` — the dashboard has no auth and a wide bind exposes job metadata to the network.

Requires `pip install fastapi uvicorn`. The dashboard reads `analysis.json` files written by `analyze_job.py`. When periodic profiling produced multiple TraceLens profiles, the dashboard shows only node 0's profiles (matching `analyze_job.py`'s selection) and offers a profile selector. The file browser filters external profile files to only show those belonging to the viewed job (using the same time-window + node-0 disambiguation).

4. **Tune** — adjust XLA flags and environment variables in `train_env.sh` based on findings.
