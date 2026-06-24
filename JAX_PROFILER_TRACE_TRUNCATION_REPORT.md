# JAX/xplane Profiler "Truncation" on ROCm — Findings and Fix

**Date:** 2026-06-23 · **Platform:** MaxText (JAX/XLA) on MI355X (ROCm), Llama-3.1 405B & 8B
*A retrospective: what looked like a profiler data-loss bug turned out to be an analysis-time
event cap. Authored with an AI assistant.*

---

## Conclusion first

When profiling MaxText with `profiler=xplane profiler_steps=3`, the captured trace appeared to
contain only **~0.57 of one step** of GPU activity for Llama-3.1 405B (e.g. TraceLens reported
~16k GEMM launches, ~21.5 s device time, per host). This led us to suspect a profiler / device
buffer limit.

**That was wrong. The `*.xplane.pb` files are complete — they contain all 3 requested steps.**
The "truncation" was introduced **at analysis time** by the trace-viewer conversion, which caps
output at **1,000,000 events** by default. That cap is a single environment variable:

```bash
export TF_PROFILER_TRACE_VIEWER_MAX_EVENTS=10000000   # default is 1000000
```

With this set, re-analyzing the **same, already-collected** 405B xplane yields the **full 3 steps**
— no re-profiling, no recompile, no config change.


| 405B xplane (same file)      | `...MAX_EVENTS=1000000` (default) | `...MAX_EVENTS=10000000` |
| ---------------------------- | --------------------------------- | ------------------------ |
| Captured steps               | ~0.57 step                        | **3 full steps**         |
| Device timeline (Stream #16) | 0–21.5 s                          | **0–112 s**              |
| GEMM kernels / GPU           | 2,045                             | **10,221** (= 3 × 3,407) |
| Total device events / host   | ~0.58 M                           | **~2.95 M**              |


Sanity check: at full coverage MaxText runs **3,407 GEMM/GPU/step**, matching Primus's 3,405 —
exactly what we expect for the same model and FLOPs. (The earlier "3,607" was an artifact of
dividing the capped view by a ~0.57 window factor.)

---

## Root cause

The trace-viewer JSON conversion caps the number of events it emits:

```c
// xla/tsl/profiler/convert/xplane_to_trace_events.cc
uint64_t GetTraceViewerMaxEvents() {
  constexpr uint64_t kMaxEvents = 1000000;
  char* max_events = getenv("TF_PROFILER_TRACE_VIEWER_MAX_EVENTS");
  return max_events ? std::stoull(max_events, ...) : kMaxEvents;
}
... container.CapEvents(GetTraceViewerMaxEvents());
```

This conversion (`xspace_to_tool_data(xplane, "trace_viewer")`) is hit in **both** places we looked:

1. **Analysis** — TraceLens loads an xplane via this exact call, so every TraceLens JAX report was
  silently capped at 1 M events. For 405B, 1 M events ≈ 0.57 step, hence the symptom.
2. **Profiling** — the `*.trace.json.gz` written next to the xplane is produced by the same
  conversion, so it is capped too (it happened to fit ~1 full step, which is why it looked
   "better" than the xplane in a viewer — but it is the same mechanism).

The underlying `*.xplane.pb` is **not** capped by this and holds the complete capture.

---

## What is NOT the cause (so we don't chase these again)

- `**xla_gpu_rocm_max_trace_events`** (ROCm device-tracer collector cap, default `4*1024*1024`):
this is a *real* cap on what the xplane *file* collects, and lowering it (e.g. 50 k) does truncate
the file. But at its default (4 M) it was **never the binding limit** here — raising it to 64 M or
the 1 e9 maximum changed nothing, because 3 steps (~3 M device events) already fit under 4 M.
- **A fixed ~120 MB file-size limit:** there is none. A single-node 8B run with many steps grows the
xplane well past 120 MB. File size tracks captured content, it is not a ceiling.
- **Reducing `profiler_steps` (e.g. 3 → 1):** does not help and is not needed — the analysis cap, not
the step count, was limiting the view.

---

## How we found it

1. Suspected and tested `xla_gpu_rocm_max_trace_events` first (a colleague doubted it even worked).
  A cap sweep at fixed 405B config confirmed the flag *works* (50 k → truncates early) but is **not**
   the cause (1 M → 1 e9 all identical), isolating "another limit."
2. Noticed the `**trace.json*`* for the same run showed a *full* step while the TraceLens/xplane view
  showed 0.57 step — pointing at the **conversion step**, not the capture.
3. Found `TF_PROFILER_TRACE_VIEWER_MAX_EVENTS` in `xplane_to_trace_events.cc`, re-analyzed the existing
  405B xplane with it raised to 10 M, and recovered all 3 steps. Decisive.

---

## The fix (practical)

**To analyze existing xplane files at full coverage** (recommended — no re-profiling):

```bash
export TF_PROFILER_TRACE_VIEWER_MAX_EVENTS=10000000   # or higher for more steps
# then run the usual TraceLens JAX report on the existing *.xplane.pb
TraceLens_generate_perf_report_jax --profile_path <host>.xplane.pb --output_csvs_dir <out>
```

- Set it high enough for the coverage you want: 405B needs ~1 M events per step per host, so
`10 M` comfortably holds the 3 captured steps. Larger captures need a larger value.
- Optionally also export it in the **training job** so the emitted `*.trace.json.gz` is not capped.
- TraceLens already ingests `.trace.json.gz` directly (`DataLoader.load_data` handles `.json.gz`);
only the `assert ...endswith(".xplane.pb")` in `generate_perf_report_jax.py` blocks that path in the
CLI. Not needed for the recommended flow above (analyze the xplane with the env var set).

## Recommendations

- **Always set `TF_PROFILER_TRACE_VIEWER_MAX_EVENTS`** when profiling large models (405B-scale), both
for analysis and (optionally) at capture time. Treat the 1 M default as "tiny" for these workloads.
- When validating a profiler/limit hypothesis, **measure the actual recovered content** (captured
steps / device-stream extent), and remember that the *analysis tool* can impose its own cap.
- Prior 405B performance comparisons used the capped (~0.57-step) view normalized per step; the
per-step *ratios* (e.g. compute-busy ~84 %, exposed-comm ~16 %) were correct, but absolute per-step
numbers should be re-derived from a full 3-step re-analysis with the env var raised.

## Key references

- Cap: `xla/tsl/profiler/convert/xplane_to_trace_events.cc` — `GetTraceViewerMaxEvents()` /
`CapEvents()`; env var `TF_PROFILER_TRACE_VIEWER_MAX_EVENTS` (default `1000000`).
- ROCm collector cap (separate, not the cause here): `xla/backends/profiler/gpu/device_tracer_rocm.cc`
— `xla_gpu_rocm_max_trace_events` (default `4*1024*1024`).
- TraceLens loader: `TraceLens/util.py` `DataLoader.load_data` (handles `.pb`, `.json.gz`, `.json`).

