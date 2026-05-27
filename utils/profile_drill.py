#!/usr/bin/env python3
"""Per-kernel time breakdown from JAX / TensorFlow xplane traces.

Reads one or more ``*.trace.json.gz`` files produced by the xplane profiler
(typically found under ``<job_dir>/<model>_train_test/tensorboard/plugins/profile/<ts>/``)
and prints:

- Top kernels within common XLA fusion families (``input_scatter_fusion``,
  ``loop_select_fusion``, ``loop_convert_fusion``, ...).
- Per-family aggregate times in seconds per GPU per profiled step.
- Total kernel time across all streams per GPU per step.

The number of active GPUs is auto-detected from ``(file, pid)`` pairs on
kernel events — this handles both ``1-node/proc`` (one trace file per host,
one ``pid`` per local GPU inside each file) and ``1-GPU/proc`` (one trace
file per GPU, all with the same ``pid``) layouts in the same invocation.
The header line prints the inferred launcher mode as a sanity check.


Launcher-mode awareness
-----------------------
Beyond file layout, the two launcher modes produce subtly different
traces.  Relevant to cross-variant comparisons:

- **Which kernels appear**: 1-node/proc keeps the EP axis intra-process
  so XLA can use in-process collectives (e.g. ``RaggedAllToAllKernelImpl``);
  1-GPU/proc forces cross-process RCCL and never emits those in-process
  kernels.  Zero time in one family does not imply the workload is
  trivial — it may just mean XLA lowered the same logical op differently.
- **Profile coverage / multi-subdir**: 1-GPU/proc spawns many more Python
  processes, each kicking off the profiler slightly skewed, so the same
  profiling window often fans out across several adjacent timestamp
  subdirs (``profile/<ts>/``, ``profile/<ts+1>/``, ...).  1-node/proc
  typically keeps everything in one subdir.  Always glob across subdirs.
- **Per-kernel attribution**: on 1-node/proc the xplane profiler can
  attribute interleaved intra-process GPU work partly to non-kernel
  overhead, so per-kernel ``dur`` of routine kernels (GEMM, flash-attn)
  can read ~2× lower than on 1-GPU/proc for the same workload.  1-GPU/proc
  is the more faithful per-GPU reading; 1-node/proc tracks wallclock
  step time more cleanly.

Do not compare absolute per-GPU kernel numbers across launchers without
understanding which attribution you are seeing.  Within one launcher,
numbers are directly comparable across variants.

Only events that look like GPU kernels are counted: anything ending in
``.kd`` (or ``.kd]`` for HIP clones), ``ncclDevKernel_*``,
``primus_turbo::*``, ``ck_tile::*``, ``aiter::*``, ``rocprim::*``,
``Cijk_*``, or ``stream_executor::*``.  CPU-side launcher events and
Python-side traces are ignored.

No runtime dependency on TraceLens — this script reads ``.trace.json.gz``
directly via Python stdlib (``gzip`` + ``json``).  It does require the
``.trace.json.gz`` to be present on disk; in the standard MaxText setup
the JAX profiler writes it natively alongside ``.xplane.pb`` (via
``xprof`` / ``tensorboard-plugin-profile``).  If only ``.xplane.pb``
exists, fix the profiler-write path — e.g. follow the TraceLens install
+ patch flow in ``skills/performance-analysis/SKILL.md`` to get a
compatible ``xprof`` — then re-run the training job.


Getting trace files (MaxText)
-----------------------------
Run a training job with:

    profiler=xplane
    skip_first_n_steps_for_profiler=5
    profiler_steps=3
    _env_ENABLE_XLA_DUMP=1

Then pass the resulting trace files to this script.  Output lands under
``<job_workspace>/<job>/<run_name>/tensorboard/plugins/profile/<ts>/``
(one ``<hostname>.trace.json.gz`` per host in 1-node/proc mode, or one
``<hostname>.proc<N>.trace.json.gz`` per GPU in 1-GPU/proc mode).


Usage
-----
    profile_drill.py [--profile-steps N] <trace.json.gz> [<trace.json.gz> ...]

Pass all trace files from one profiling window at once.  The script
normalises to "per GPU per step" so the absolute number of input files
doesn't matter, only the set of ``(file, pid)`` pairs found inside them.


Multi-subdir gotcha
-------------------
A single profiling window sometimes splits across **multiple adjacent
timestamp directories** — JAX starts the profiler on each task slightly
skewed, so some hosts write into ``profile/<ts>/`` and a handful of
others into ``profile/<ts+1>/``.  Always glob across subdirs to avoid
under-counting GPUs:

    profile_drill.py /path/to/profile/*/*.trace.json.gz

If the ``GPUs auto-detected=NN`` line in the output does not match
``num_nodes × gpus_per_node``, you are probably missing a subdir or
including stale traces from a different run.


Example
-------
    profile_drill.py --profile-steps 3 \\
        /path/to/job/tensorboard/plugins/profile/*/*.trace.json.gz
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys


KERNEL_FAMILIES = {
    "RaggedAllToAllKernelImpl":      lambda n: "RaggedAllToAllKernelImpl" in n,
    "primus_turbo::deep_ep":         lambda n: "primus_turbo::deep_ep" in n,
    "input_scatter_fusion":          lambda n: "input_scatter_fusion" in n,
    "loop_select_fusion":            lambda n: "loop_select_fusion" in n,
    "loop_gather_fusion":            lambda n: "loop_gather_fusion" in n,
    "loop_reduce_fusion":            lambda n: "loop_reduce_fusion" in n,
    "loop_convert_fusion":           lambda n: "loop_convert_fusion" in n,
    "loop_transpose_fusion":         lambda n: "loop_transpose_fusion" in n,
    "input_reduce_select":           lambda n: "input_reduce_select" in n,
    "input_broadcast_reduce_select": lambda n: "input_broadcast_reduce_select" in n,
    "RCCL ncclDevKernel":            lambda n: n.startswith("ncclDevKernel"),
    "CK+primus GEMM (non-DeepEP)":   lambda n: (
        "primus_turbo::deep_ep" not in n
        and ("ck_tile::" in n or "primus_turbo::" in n or "Cijk_" in n)
    ),
    "flash_attn (fmha)":             lambda n: "aiter" in n and "fmha" in n,
}


def is_gpu_kernel(name: str) -> bool:
    """Heuristic filter for GPU kernel events in an xplane trace."""
    return (
        name.endswith(".kd")
        or ".kd]" in name  # e.g. "... [clone .kd]"
        or name.startswith("ncclDevKernel")
        or "primus_turbo::" in name
        or "ck_tile::" in name
        or "aiter" in name
        or "rocprim::" in name
        or "Cijk_" in name
        or "stream_executor::" in name
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "trace_files",
        nargs="+",
        help="One or more .trace.json.gz files produced by the xplane profiler.",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=3,
        help="Number of profiled steps captured in each trace (matches "
             "MaxText's profiler_steps flag, default 3).",
    )
    parser.add_argument(
        "--family-top-n",
        type=int,
        default=None,
        help="Per-family drill-down: show at most N kernels per family "
             "(default: show all matching kernels).",
    )
    args = parser.parse_args()

    kernel_times_ms: dict[str, float] = collections.defaultdict(float)
    kernel_counts: dict[str, int] = collections.defaultdict(int)
    gpu_pids: set[tuple[str, object]] = set()

    for tf in args.trace_files:
        with gzip.open(tf, "rt") as f:
            trace = json.load(f)
        events = trace.get("traceEvents", trace)
        for ev in events:
            if ev.get("ph") != "X":
                continue
            name = ev.get("name", "")
            if not name or not is_gpu_kernel(name):
                continue
            dur = ev.get("dur", 0)
            kernel_times_ms[name] += dur / 1000.0
            kernel_counts[name] += 1
            gpu_pids.add((tf, ev.get("pid")))

    gpus_seen = len(gpu_pids)
    divisor = gpus_seen * args.profile_steps

    if divisor == 0:
        print("No GPU kernel events found — is this an xplane trace?",
              file=sys.stderr)
        return 1

    # Infer launcher mode from gpus-per-file ratio:
    #   1-node/proc  -> one trace file per host with many GPUs per file
    #   1-GPU/proc   -> one trace file per GPU (ratio ≈ 1)
    n_files = len(args.trace_files)
    gpus_per_file = gpus_seen / n_files
    if gpus_per_file >= 2.0:
        launcher_hint = f"(looks like 1-node/proc, {gpus_per_file:.1f} GPUs/file)"
    elif abs(gpus_per_file - 1.0) < 0.05:
        launcher_hint = "(looks like 1-GPU/proc, 1.0 GPU/file)"
    else:
        launcher_hint = (f"({gpus_per_file:.2f} GPUs/file — unusual, possibly "
                         f"missing trace files or mixed launcher)")

    print(f"trace_files={n_files}, "
          f"GPUs auto-detected={gpus_seen}, "
          f"profile_steps={args.profile_steps}, "
          f"divisor={divisor}  {launcher_hint}")

    for family_tag in [
        "input_scatter_fusion", "loop_select_fusion", "loop_convert_fusion",
        "loop_transpose_fusion", "loop_reduce_fusion", "input_reduce_select",
        "input_broadcast_reduce_select",
    ]:
        matches = [(n, t) for n, t in kernel_times_ms.items() if family_tag in n]
        if not matches:
            continue
        print()
        print(f"{family_tag}* kernels:")
        matches.sort(key=lambda kv: -kv[1])
        if args.family_top_n is not None:
            matches = matches[:args.family_top_n]
        for name, t in matches:
            cnt = kernel_counts[name]
            per_step = cnt // max(divisor, 1)
            print(f"  {t/divisor:8.1f} ms/gpu/step  "
                  f"{cnt:7d} launches ({per_step} per gpu-step)  {name}")

    print()
    print("--- Per-family aggregates (s/gpu/step) ---")
    # First-match-wins: each kernel attributed to at most one family (the
    # first in KERNEL_FAMILIES that matches).  This keeps the itemised rows
    # + "Other kernels" exactly summing to Total kernel time even if a new
    # predicate is added that overlaps an existing one.  Dict iteration is
    # insertion-ordered in Python 3.7+, so the defined order is stable.
    used_names: set[str] = set()
    for label, pred in KERNEL_FAMILIES.items():
        total_ms = 0.0
        for name, t in kernel_times_ms.items():
            if name in used_names:
                continue
            if pred(name):
                total_ms += t
                used_names.add(name)
        print(f"  {label:<32} {total_ms / divisor / 1000:8.3f} s/gpu/step")

    other_ms = sum(t for n, t in kernel_times_ms.items() if n not in used_names)
    print(f"  {'Other kernels':<32} {other_ms / divisor / 1000:8.3f} s/gpu/step")

    total_all_ms = sum(kernel_times_ms.values())
    print()
    print(f"Total kernel time: {total_all_ms / divisor / 1000:.3f} s/gpu/step")
    return 0


if __name__ == "__main__":
    sys.exit(main())
