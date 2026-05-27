---
name: profile-drill
description: Direct per-kernel time analysis from JAX / TensorFlow xplane traces via `utils/profile_drill.py`. Use when the user asks for a per-kernel breakdown, step-time composition, cross-variant kernel comparison, main-stream-blocking analysis, or any question that needs ground-truth kernel timings below what TraceLens reports. Triggers include "xplane", "trace.json.gz", "input_scatter_fusion", "RaggedAllToAllKernelImpl", "ncclDevKernel", "step − total kernel", "main-stream-busy", "profile drill-down", or suspicion that TraceLens numbers are off by ~1.5–2×.
---

# Profile Drill: per-kernel time analysis from xplane traces

Direct-read xplane analysis for kernel-level ground truth.  Sibling to `performance-analysis` (which goes via TraceLens).  Use this skill when:

- Routine TGS/MFU reporting is not enough — you need per-kernel times.
- Cross-variant comparison (baseline vs experimental `moe.py` / sharding change / XLA flag toggle) where all variants must be measured with the same yardstick.
- TraceLens numbers look suspicious (systematic ~1.5×–2× inflation on 1-node/proc profiles — see [Pitfalls](#pitfalls)).
- Building a step-time composition table: what fraction of a step is each kernel family, how much idle gap, how much stream overlap?
- Identifying main-stream-blocking kernels (dominant throughput killers on MoE and ragged-collective paths).

Tool: [`utils/profile_drill.py`](../../utils/profile_drill.py).  Docstring covers CLI usage and the multi-subdir gotcha; this skill covers methodology and interpretation.

## When to use this skill vs `performance-analysis`

| Situation | Skill |
|---|---|
| "Analyze this job / what's the TGS?" | `performance-analysis` |
| "Compute / exposed-comm / idle %?" | `performance-analysis` |
| "Install TraceLens, run dispatcher, populate dashboard" | `performance-analysis` |
| "How long does kernel `<X>` take per GPU per step?" | **profile-drill** |
| "Which `input_scatter_fusion_*.kd` / `loop_*_fusion_*.kd` dominates?" | **profile-drill** |
| "Compare kernel time breakdown across N experimental variants." | **profile-drill** |
| "The `time ms per gpu` number in the TraceLens CSV seems too high." | **profile-drill** (escape hatch) |

Both skills read the same xplane artifacts (`*.trace.json.gz`, `*.xplane.pb`).  `performance-analysis` aggregates via TraceLens; this skill parses the trace JSON directly.

## Prerequisites

### Dependencies

`profile_drill.py` has **no runtime dependency on TraceLens** — it reads `.trace.json.gz` files directly via Python's stdlib (`gzip` + `json`) and never loads TraceLens modules.  The skill and `performance-analysis` are independent analysis paths over the same upstream profiler output.

The indirect dependency is that **the `.trace.json.gz` file has to exist**.  In the standard MaxText setup, the JAX profiler writes `.trace.json.gz` natively alongside `.xplane.pb` (see `utils/monkey_patch_maxtext.py` which flocks `jax.profiler.stop_trace`).  This relies on the `xprof` (aka `tensorboard-plugin-profile`) package being importable at trace-write time.

If the profile directory contains `*.xplane.pb` files **but no matching `*.trace.json.gz`** — typically a container issue with xprof / TensorFlow 2.19+ compatibility — install and patch TraceLens following the steps in `skills/performance-analysis/SKILL.md` (Step 2 and [tracelens-patches.md](../performance-analysis/tracelens-patches.md)).  That install pulls in a compatible `xprof` and patches the known renames; then re-run the training job so the next profile window writes `.trace.json.gz` natively.  You do not need to actually *use* TraceLens after that — `profile_drill.py` can operate on the native JAX output directly.

### Producing trace files

Add these passthrough args to the training job (MaxText):

```
profiler=xplane
skip_first_n_steps_for_profiler=5
profiler_steps=3
_env_ENABLE_XLA_DUMP=1
```

3 profiled steps is the sweet spot — enough to average out per-step jitter, cheap enough that profiler-writeback noise (visible as an inflated step time on the step immediately **after** the profile window) stays localised.

### Trace file layout

Output lives under `<job_dir>/<run_name>/tensorboard/plugins/profile/<ts>/`:

- **1-node/proc**: one `<hostname>.trace.json.gz` per host.  Each file contains events from all local GPUs on that host (identified by distinct `pid` values inside the file).
- **1-GPU/proc**: one `<hostname>.proc<N>.trace.json.gz` per GPU, each with a single `pid` (usually `1`).

The tool prints the inferred launcher mode in its header line (`looks like 1-node/proc, 8.0 GPUs/file` vs `looks like 1-GPU/proc, 1.0 GPU/file`).  A cross-variant comparison where the launcher hint differs between runs is almost always a mistake — stop and re-check.

### Launcher-mode differences that affect profiles

1-node/proc and 1-GPU/proc are not just different file layouts — they produce subtly different traces in four dimensions:

| Dimension | 1-node/proc | 1-GPU/proc |
|---|---|---|
| Trace files per host | 1 (multi-pid) | `gpus_per_node` (single-pid) |
| Multi-subdir likelihood | Rare (all hosts' profilers fire nearly together) | **Common** (many more processes → more skew); always glob `profile/*/…` |
| Which kernels XLA emits | Intra-process collectives available — e.g. `stream_executor::gpu::RaggedAllToAllKernelImpl` for intra-process EP axes | Cross-process only — same ragged op falls back to RCCL `AllToAll`. Zero time in an "in-process" family here does not mean the workload is trivial, just lowered differently. |
| Per-kernel attribution | Xplane profiler can fold interleaved intra-process GPU work into non-kernel overhead; per-kernel `dur` of routine kernels (GEMM, flash-attn) may read **~2× lower** than the true per-GPU time | Each process observes its own kernels as fully-accounted events; **more faithful per-GPU reading** |
| TraceLens CSV bias | **Broken** — `time ms per gpu` is inflated ~1.5×–2× (see [Pitfalls](#the-tracelens-csv-divisor-bug)) | Correct (file count = GPU count, so TraceLens's divisor accidentally lands right) |

Practical consequences:

- **Do not compare absolute per-GPU kernel numbers across launchers without thinking about which attribution you're seeing.** A "faster GEMM" on 1-node/proc relative to 1-GPU/proc is almost always the attribution artefact, not a real speedup.
- **Within one launcher, numbers are directly comparable** across variants (same image, same passthrough flags, only the code-under-test changes).  This is the correct setup for experimental-variant drill-downs.
- **For step-time composition tables** that need to sum to wallclock step time, 1-node/proc is more honest about blocking kernels (the idle-gap row surfaces them clearly), while 1-GPU/proc often shows negative `step − total kernel` (healthy overlap) which is harder to decompose.

### Multi-subdir gotcha

A single profiling window commonly splits across **two or more adjacent timestamp directories** because JAX fires the profiler on each task slightly skewed.  The severity is much higher on 1-GPU/proc (many more processes).  Always glob across subdirs:

```bash
ls <job>/*/tensorboard/plugins/profile/*/*.trace.json.gz > /tmp/traces.txt
```

After running the tool, verify the `GPUs auto-detected=NN` line matches `num_nodes × gpus_per_node` **and** the launcher hint matches the job you submitted.  If not, you are missing a subdir, including stale traces from a different run, or looking at a different launcher than you thought.

## Workflow

### Step 1: Collect trace files

Build a list-file with a glob that captures **all** `.trace.json.gz` for the profiling window:

```bash
ls <job_dir>/*/tensorboard/plugins/profile/*/*.trace.json.gz > /tmp/traces.txt
wc -l /tmp/traces.txt
```

Expected count: `num_nodes` (1-node/proc) or `num_nodes × gpus_per_node` (1-GPU/proc).

### Step 2: Run `profile_drill.py`

```bash
python3 utils/profile_drill.py $(cat /tmp/traces.txt)
```

The tool prints three blocks:

1. A header with `trace_files`, `GPUs auto-detected`, `profile_steps`, `divisor`.
2. A per-kernel drill-down for a fixed list of XLA fusion families (`input_scatter_fusion`, `loop_select_fusion`, `loop_convert_fusion`, `loop_transpose_fusion`, `loop_reduce_fusion`, `input_reduce_select`, `input_broadcast_reduce_select`) — each row is one kernel name, its time per GPU per step, and launch count.
3. Per-family aggregates (one row per family predicate, see [Per-family aggregate semantics](#per-family-aggregate-semantics)) plus an `Other kernels` catch-all, plus the `Total kernel time` summed across all streams.

### Step 3: Verify the divisor

Before trusting any number, confirm:

```
GPUs auto-detected == num_nodes × gpus_per_node
```

If lower → missing a timestamp subdir in the glob or traces never wrote.  If higher → you included an unrelated profile window; scope to a specific timestamp.

### Step 4: Compute `step − total kernel`

Look up the **steady-state** step time from the training log (use a no-profile companion run if available — the profile run's step-time is skewed by writeback on the step immediately after the profile window).  Then:

```
idle_gap_or_overlap = step_time − total_kernel_time
```

Interpretation — see [The `step − total kernel` row is the key diagnostic](#the-step--total-kernel-row-is-the-key-diagnostic).

## Interpretation cookbook

### The `step − total kernel` row is the key diagnostic

| Sign | Meaning |
|---|---|
| **Positive (+)** | Main-stream blocker(s) are stalling the GPU.  Something on the compute stream occupies kernel time that does not permit overlap, and downstream kernels can't begin.  The positive gap is real idle wall-clock that no family bucket captures. |
| **Near zero** | Kernels are tightly serialised, minimal idle, minimal overlap.  Rare in practice. |
| **Negative (−)** | Streams are genuinely overlapping (compute stream + RCCL comm stream kernels run concurrently).  `total kernel` sums across all streams, so a healthy overlapping step has total > wallclock. |

See [Examples](#examples) for concrete numbers measured on real profiles.

### Identifying main-stream-blocking kernels

Symptom: a single kernel (family) takes a large fraction of the step **and** `step − total kernel` is strongly positive.  Because the idle-gap row surfaces stalls more directly on 1-node/proc (see [Launcher-mode differences](#launcher-mode-differences-that-affect-profiles)), this analysis is usually easiest on a 1-node/proc profile even if the production launcher is 1-GPU/proc.

Common suspects on AMD MI3xx:

- **`stream_executor::gpu::RaggedAllToAllKernelImpl<N>`** — *1-node/proc only.*  XLA's in-process fallback for `ragged-all-to-all` when all EP ranks share a JAX process.  Sequential across peers, no pipelining, one xGMI link at a time.  The typical remedy is to pass the XLA flag `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false` so the ragged thunk picks the RCCL path instead — the same path that 1-GPU/proc selects automatically.  This kernel does not exist on 1-GPU/proc traces; zero time in this family on a 1-GPU profile is normal.
- **Duplicate-index atomic scatter-add fusions** — *appears on both launchers.*  Shows up as `input_scatter_fusion_*.kd` when JAX autodiff inverts a `recv_x[indices]` gather where `indices` has duplicates (e.g. MoE top-K fan-out).  Each atomic scatters one peer-word at a time through HBM and cannot overlap with following GEMMs.  Typical remedies: compose chained gathers into one (halves the atomics), or replace the duplicate-index backward with a `jax.custom_vjp` that uses a permutation gather + reduce-sum (eliminates atomics entirely).
- **Any `loop_*_fusion_<N>.kd`** that ends up on the main stream with per-call duration above ~50 ms — *appears on both launchers.*  Inspect its HLO to see what it's doing.

### Per-family aggregate semantics

The tool sums times using this classification (lambdas in [`utils/profile_drill.py`](../../utils/profile_drill.py)):

| Family | What it means |
|---|---|
| `RaggedAllToAllKernelImpl` | XLA's in-process ragged-all-to-all (naive, serial across peers).  Appears on 1-node/proc when the one-shot kernel is enabled. |
| `primus_turbo::deep_ep` | AMD Primus-Turbo's MoE dispatch/combine HIP kernels.  Appears with `use_deepep_dispatch=true`. |
| `input_scatter_fusion` / `loop_select_fusion` / `loop_gather_fusion` / etc. | XLA-emitted fusion kernels around user-code indexing / masking / permutation. |
| `RCCL ncclDevKernel` | RCCL collective-ops on the comm stream.  A higher RCCL number on a faster variant often means the scheduler packs more RCCL work in (because a main-stream blocker is now gone), not that RCCL itself got slower — look at `step − total kernel` to confirm. |
| `CK+primus GEMM (non-DeepEP)` | AMD Composable-Kernel and Primus-Turbo GEMM kernels (grouped + dense), excluding DeepEP HIP calls. |
| `flash_attn (fmha)` | AMD `aiter::fmha_*` flash-attention forward/backward. |
| `Other kernels` | Everything that matched `is_gpu_kernel()` but no family predicate (memcpy, barriers, minor XLA fusions). |

### Cross-variant comparison discipline

When comparing experimental variants against a baseline:

1. **Same launcher** on every variant.  The tool's header line prints `looks like 1-node/proc` or `looks like 1-GPU/proc` — if it differs across variants, the attribution is not comparable (see [Launcher-mode differences](#launcher-mode-differences-that-affect-profiles)).
2. **Same `--profile-steps`** on every invocation (defaults to 3, matching `profiler_steps=3`).  Mismatching profile_steps makes per-step numbers incomparable by a constant factor.
3. **Same glob shape** on every invocation (e.g. `profile/*/*.trace.json.gz`).  Under-globbing one variant silently under-counts its GPUs.
4. **Verify `GPUs auto-detected` is identical** across the variants you're comparing.  It must equal `num_nodes × gpus_per_node` for every profile.
5. **Capture the no-profile TGS separately**, not from the profile run.  The step that immediately follows the 3-step profile window is slowed by writeback; using the profile run's TGS skews the step-time denominator in your composition table.

## Utility: step-time composition table

To build a composition table for a single variant:

1. Run `profile_drill.py` on all trace files from one profile window.
2. From the per-family aggregates, fill one row per itemised family (`RaggedAllToAllKernelImpl`, `primus_turbo::deep_ep`, `input_scatter_fusion`, `loop_select_fusion`, `loop_gather_fusion`, RCCL, CK+primus GEMM, flash_attn).
3. Sum the remaining small families (`loop_reduce_fusion`, `loop_convert_fusion`, `loop_transpose_fusion`, `input_reduce_select`, `input_broadcast_reduce_select`) plus `Other kernels` into an "Other fusions" row.  This must sum with the itemised rows to equal `Total kernel time`.
4. Get step time from a no-profile run at the same config (TGS × pdbs × seq_len / total_devices).
5. Compute `step − total kernel` and put it in its own row.

When comparing variants, put each variant in its own column, with all columns computed identically (same `--profile-steps`, same glob shape, same family definitions).

## Pitfalls

### The TraceLens CSV divisor bug

`performance-analysis` Step 3 recommends reading `kernel_launchers_summary_by_category.csv` and `kernel_launchers_summary.csv` from TraceLens.  **These CSVs have a systematic ~1.5×–2× inflation on 1-node/proc profiles.**

- The column literally labeled `time ms per gpu` = `total_direct_kernel_time_ms / N`, where N is the number of trace files (= number of hosts on 1-node/proc).  TraceLens divides by the **number of trace files**, not the number of GPUs.
- On 1-node/proc profiles each trace file covers several GPUs (typically 8), so the denominator is under-counted by that factor and per-GPU time is over-reported.
- On 1-GPU/proc profiles each file is already one GPU, so the TraceLens divisor coincidentally comes out right and the CSVs match reality.
- Empirically, TraceLens's aggregate also processes a subset of the trace events (mechanism not fully traced), which partially offsets the divisor error.  Net effect on 1-node/proc profiles: per-GPU numbers land 1.5×–2× above true per-GPU kernel time, with the exact multiplier varying by kernel family.

**Diagnostic**: compare a TraceLens CSV category (say GEMM or RCCL) between 1-node/proc and 1-GPU/proc profiles of the **same pdbs** on the same model.  If the 1-GPU reading is ~half the 1-node reading, TraceLens is bitten — the actual per-GPU GEMM time should be identical across launchers modulo small stream-placement effects, and the `profile_drill.py` 1-GPU number is ground truth.

**Remedy**: use `profile_drill.py` whenever you need per-kernel numbers you can cite.  It counts raw `dur` fields from the trace JSON and divides by auto-detected `gpus × profile_steps`.

### Missing timestamp subdirs (under-counting GPUs)

Symptom: `GPUs auto-detected=NN` where NN < `num_nodes × gpus_per_node`.  A portion of the profiling window landed in a sibling `profile/<ts+1>/` directory that your glob didn't include.  Re-glob with `profile/*/…` (not a specific timestamp).

### Stale traces from a previous run

Symptom: `GPUs auto-detected=NN` larger than expected, or families show unexpected kernels (e.g. `RaggedAllToAllKernelImpl` with positive time on a variant that shouldn't go through that path).  The glob picked up an older profiling window in the same job directory.  Scope to a specific timestamp window.

### Profile-run step-time ≠ no-profile step-time

The step that immediately follows the 3-step profile window (step 8 for `skip_first=5`, `steps=3`) is slowed by profiler writeback — measured overhead ranges from ~30 % to ~70 % above steady state, depending on how much profile data had to be serialised.  If you use the profile run's mean TGS for the denominator in a step-time composition table, your `step − total kernel` will be artificially inflated.  Source the step time from a companion no-profile run, or exclude the writeback step from the profile run's TGS.

## Examples

Two worked examples reconstructed from real DeepSeek-V3 671B runs (MI355, 8 nodes × 8 GPUs, `pdbs=6`, seq_len = 4096, so 64 GPUs × 3 profiled steps → divisor 192).  Both illustrate the patterns described above.  Treat them as illustrative — the kernel times will look different on your hardware/parallelism, but the *ratios* and the *workflow* of identifying the bottleneck class transfer.

### Example 1: launcher mode drill-down

Three profiles of the same pdbs=6 workload under different launcher/config combinations:

| Variant                   | Step  | Total kernel | step − total | Dominant kernel pattern |
|---------------------------|------:|-------------:|-------------:|-------------------------|
| `sparse-gmm` 1-node/proc  | 82.5 s |      45.1 s |    **+37.4 s** | `RaggedAllToAllKernelImpl` = 28.4 s/GPU/step — blocks everything on main stream |
| `sparse-gmm` 1-GPU/proc   | 26.1 s |      33.5 s |    **−7.4 s**  | RCCL `ncclDevKernel` = 15.3 s overlaps with ~14 s of compute — healthy |
| `sparse-gmm-deepep` 1-node/proc | 38.0 s | 26.9 s |    **+11.1 s** | `input_scatter_fusion_2.kd` (≈ 4.4 s) is a smaller main-stream blocker |

Lessons:

- The 1-node→1-GPU step-time drop (−56.4 s) has two roughly-equal causes: (a) *direct removal* of the main-stream blocker — the 28.4 s `RaggedAllToAllKernelImpl` on the 1-node profile is purely sequential, so its disappearance under 1-GPU frees ~28 s of wallclock directly; (b) *scheduler cascade recovery* — once the blocker is gone, kernels that were stranded in gaps on the main stream (RCCL, flash-attn, etc.) can run concurrently with compute on their respective streams, collapsing most of the remaining idle.  Step time drops wholesale, not just by the blocker's duration.
- The DeepEP 1-node profile (12897) has an idle gap of +11.1 s on a 38.0 s step — about 29 % of wallclock is main-stream idle — despite DeepEP being designed as a faster path.  The culprit is one main-stream-blocking `input_scatter_fusion_2.kd`.  Compare with 1-GPU sparse-gmm (12916), which has a healthy negative gap of −7.4 s (stream overlap): even though both paths do similar work, the idle-gap sign flips because nothing on 1-GPU blocks the main stream.  Takeaway: `step − total kernel` is the single best diagnostic for "is this workload bottlenecked by a blocking kernel?" — more reliable than ranking kernel family totals.  (Do **not** compare absolute kernel-time totals across launchers — that comparison is subject to the attribution artefact described in [Launcher-mode differences](#launcher-mode-differences-that-affect-profiles).)

### Example 2: experimental variant comparison (v1 / v2 / v3 of a `moe.py` change)

Same workload, same image, same flags — only `src/MaxText/layers/moe.py` differs between three `yihuang/moe-turbo-gmm-and-deepep{,-v2,-v3}` patch branches:

| Variant | `input_scatter_fusion_*.kd` | Total kernel | Step | step − total | Step-time saving vs prev |
|---|---:|---:|---:|---:|---:|
| v1 baseline (two gathers, two atomic scatter-adds) | **8.97** | 26.93 s | 38.0 s | +11.1 s | — |
| v2 (compose gathers: one atomic scatter-add)       | **4.45** | 21.69 s | 30.5 s | +8.8 s  | −7.5 s |
| v3 (`custom_vjp` replaces scatter-add with reduce-sum) | **0.04** | 19.13 s | 23.9 s | +4.8 s  | −6.6 s |

Lessons:

- `input_scatter_fusion_*.kd` drops monotonically (8.97 → 4.45 → 0.04 s) as Python-side MoE changes remove duplicate-index atomic scatter-adds from the backward HLO.
- v2 → v3 saves 6.6 s of step time while net kernel time only drops 2.6 s — the remainder comes from scheduler cascade: with no main-stream blocker left, XLA's latency-hiding scheduler packs more RCCL work on the comm stream (RCCL kernel time actually **increases** 6.9 → 8.2 s) while exposed comm shrinks.  The forward HLO is bit-identical across v1/v2/v3; losses match to bf16 LSB.

Discipline applied in both examples: identical `--profile-steps 3`, identical glob (`profile/*/*.trace.json.gz`), `GPUs auto-detected=64` on every run, headline TGS sourced to exclude the profiler writeback step — either from a companion no-profile run (v2, v3) or from the profile run's steady-state tail only (steps 9-14 when `skip_first=5 steps=3`).

## Reference artifacts

- [`utils/profile_drill.py`](../../utils/profile_drill.py) — the tool itself.  `--help` shows CLI options; the module docstring covers usage + multi-subdir gotcha.
- Worked example writeups live on feature branches (see above).  Do not assume such writeups exist on `main`.
