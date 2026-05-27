#!/usr/bin/env python3
"""Analyze MaxText training jobs — detect artifacts and run analysis tools.

Thin orchestrator that checks a job's output directory for available
artifacts (log with TGS data, XLA dump, xplane trace) and dispatches
to the corresponding tools (tgs_tagger, IRLens, TraceLens).

After running tools, saves a structured ``analysis.json`` — including a
``job_status`` field (completed/failed/cancelled/running/unknown) — in
the job directory for consumption by ``perf_server.py`` (web dashboard).

Staleness detection: re-analysis is skipped when ``analysis.json`` is
newer than the log file and the recorded job status is terminal.  Pass
``-f`` to bypass this check and force a full re-run.

TraceLens outputs are cached per profiling window under
``tracelens/<timestamp>/``; already-analyzed windows are skipped on
subsequent runs.

Usage:
    analyze_job.py [OPTIONS] [PATH ...]

See --help for full details.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── ANSI colors ──────────────────────────────────────────────────────────────

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ── Patterns ─────────────────────────────────────────────────────────────────

RE_TOKENS_VALUE = re.compile(r"Tokens/s/device:\s*([0-9]+(?:\.[0-9]+)?)")
RE_TOKENS = re.compile(r"Tokens/s/device:\s*[0-9]+(?:\.[0-9]+)?")
RE_STEP_LINE = re.compile(
    r"completed step:\s*(\d+),\s*seconds:\s*([0-9.]+),\s*"
    r"TFLOP/s/device:\s*([0-9.]+),\s*MFU:\s*([0-9.]+)%,\s*"
    r"Tokens/s/device:\s*([0-9.]+).*?loss:\s*([0-9.]+)"
)
RE_NNODES = re.compile(r"^(?:NNODES|SLURM_JOB_NUM_NODES)\s*=\s*(\d+)", re.MULTILINE)
RE_NODELIST = re.compile(r"^(?:JOB_NODELIST|SLURM_JOB_NODELIST)\s*=\s*(.+)", re.MULTILINE)
RE_JOB_NAME = re.compile(r"^(?:JOB_NAME|SLURM_JOB_NAME)\s*=\s*(.+)", re.MULTILINE)
RE_MODEL_NAME = re.compile(r"^MODEL_NAME\s*=\s*(.+)", re.MULTILINE)
RE_JOB_ID_HEADER = re.compile(r"^(?:JOB_ID|SLURM_JOB_ID)\s*=\s*(.+)", re.MULTILINE)
RE_EXP_TAG = re.compile(r"^EXP_TAG\s*=\s*(.+)", re.MULTILINE)
RE_RAY_ACTOR = re.compile(r"MaxTextTrainerActor")
RE_PASSTHROUGH = re.compile(r'^PASSTHROUGH_ARGS\s*=\s*"?(.*?)"?\s*$', re.MULTILINE)
RE_JOB_SUMMARY = re.compile(r"^=+ JOB SUMMARY =+", re.MULTILINE)
RE_JOB_STATUS = re.compile(
    r"Status:\s*(?:\033\[\d+m)*(SUCCESS|FAILED)\s*\(exit\s+(\d+)\)",
)
_LOG_SUFFIXES = {".log", ".out"}
_RUNNING_THRESHOLD_S = 900  # 15 min — accommodates long checkpoint writes

# Constants matching tgs_tagger.py
WARMUP_STEPS = 5
STEADY_STATE_STEPS = 10

# ── Utilities ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent


def _banner(label: str) -> None:
    print(f"\n{CYAN}{BOLD}{'─' * 3} {label} {'─' * (60 - len(label))} {RESET}")


def _step(label: str) -> None:
    print(f"  {GREEN}▸{RESET} {label}")


def _skip(label: str, reason: str) -> None:
    print(f"  {DIM}▹ {label}: {reason}{RESET}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠ {msg}{RESET}")


def _err(msg: str) -> None:
    print(f"  {RED}✗ {msg}{RESET}")


def _run(cmd: list[str], label: str, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a subprocess, print its output, and return the result."""
    _step(f"Running {label}")
    print(f"    {DIM}$ {' '.join(cmd)}{RESET}", flush=True)
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
    )
    if result.returncode != 0 and capture:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
    return result


# ── Artifact detection ───────────────────────────────────────────────────────


def _find_log_file(job_dir: Path) -> Path | None:
    """Find the log file for a job directory (check symlink, then parent)."""
    # Job dir might have a log symlink
    log_link = job_dir / "log"
    if log_link.is_file():
        return log_link.resolve()

    # Look in the parent directory for <job_prefix>*.log
    parent = job_dir.parent
    prefix = job_dir.name
    for suffix in _LOG_SUFFIXES:
        candidates = sorted(parent.glob(f"{prefix}*{suffix}"))
        if candidates:
            return candidates[0]

    # Try matching by job ID prefix
    m = re.match(r"^([^-]+)-", job_dir.name)
    if m:
        job_id = m.group(1)
        for suffix in _LOG_SUFFIXES:
            candidates = sorted(parent.glob(f"{job_id}-*{suffix}"))
            if candidates:
                return candidates[0]

    return None


def _find_hlo_file(job_dir: Path) -> Path | None:
    """Find the latest jit_train_step *gpu_after_optimizations.txt in xla_dump/."""
    xla_dump = job_dir / "xla_dump"
    if not xla_dump.is_dir():
        return None
    candidates = sorted(xla_dump.glob("*jit_train_step*gpu_after_optimizations.txt"))
    if not candidates:
        return None
    # Use the highest module number (latest)
    return candidates[-1]


def _container_to_host_path(container_path: str, outputs_dir: Path) -> Path:
    """Translate an in-container ``/outputs/...`` path to the host-side equivalent."""
    container_path = container_path.rstrip("/")
    prefix = "/outputs/"
    if container_path.startswith(prefix):
        return outputs_dir / container_path[len(prefix):]
    return Path(container_path)


def _extract_config_param(log_text: str | None, param: str) -> str | None:
    """Extract a MaxText ``Config param <param>: <value>`` from log text."""
    if not log_text:
        return None
    m = re.search(rf"Config param {re.escape(param)}:\s*(\S+)", log_text)
    return m.group(1) if m else None


def _extract_tensorboard_dir(log_text: str | None, outputs_dir: Path) -> Path | None:
    """Extract the tensorboard_dir from MaxText log output.

    MaxText logs ``Config param tensorboard_dir: <path>`` where ``<path>``
    is an in-container path like ``/outputs/<model>/...``.  We translate
    the ``/outputs/`` prefix to the host-side *outputs_dir* so that
    ``analyze_job.py`` can locate profile artifacts regardless of whether
    checkpointing redirected them to a shared directory.
    """
    raw = _extract_config_param(log_text, "tensorboard_dir")
    if not raw:
        return None
    host_path = _container_to_host_path(raw, outputs_dir)
    return host_path if host_path.is_dir() else None


def _parse_profile_timestamp(name: str) -> float | None:
    """Parse a profile directory name like ``2026_02_18_08_56_06`` to epoch seconds."""
    try:
        dt = datetime.strptime(name, "%Y_%m_%d_%H_%M_%S").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


_TIME_WINDOW_END_BUFFER_SECS = 60


def _job_time_window(job_dir: Path, log_file: Path | None) -> tuple[float, float] | None:
    """Return (start_epoch, end_epoch) for a job's execution window.

    Uses artifact directory ctime as the start proxy (files are copied at
    submission time) and the log file mtime + 60 s buffer as the end proxy.
    The buffer accounts for profiles captured after the last log write
    (e.g. killed jobs with periodic profiling).  Kept small to avoid
    overlapping with back-to-back jobs on the same nodes.
    """
    start: float | None = None
    art_dir = job_dir / "artifact"
    if art_dir.is_dir():
        try:
            start = art_dir.stat().st_ctime
        except OSError:
            pass
    if start is None and log_file:
        try:
            start = log_file.stat().st_ctime
        except OSError:
            pass
    if start is None:
        return None

    end: float = time.time()
    if log_file:
        try:
            end = log_file.stat().st_mtime + _TIME_WINDOW_END_BUFFER_SECS
        except OSError:
            pass
    return (start, end)


_PROFILE_SESSION_WINDOW_SECS = 60


def _find_xplane_files(
    job_dir: Path,
    tensorboard_dir: Path | None = None,
    job_time_window: tuple[float, float] | None = None,
) -> list[Path]:
    """Find all *.xplane.pb files, falling back to an external tensorboard_dir.

    First searches *job_dir* recursively.  If nothing is found and an
    external *tensorboard_dir* is provided (parsed from the log), searches
    ``<tensorboard_dir>/plugins/profile/`` instead, filtering timestamp
    directories to those within the job's execution window to avoid
    picking up profiles from other jobs sharing the same directory.
    """
    found = sorted(job_dir.rglob("*.xplane.pb"))
    if found:
        return found
    if tensorboard_dir:
        profile_root = tensorboard_dir / "plugins" / "profile"
        if profile_root.is_dir():
            ts_dirs = sorted(d for d in profile_root.iterdir() if d.is_dir())
            if job_time_window:
                job_start, job_end = job_time_window
                ts_dirs = [
                    d for d in ts_dirs
                    if (ep := _parse_profile_timestamp(d.name)) is not None
                    and job_start <= ep <= job_end
                ]
            for d in ts_dirs:
                found.extend(sorted(d.glob("*.xplane.pb")))
    return found


def _log_has_tgs_data(log_file: Path) -> bool:
    """Check if the log file contains any Tokens/s/device lines."""
    try:
        text = log_file.read_text(errors="replace")
        return bool(RE_TOKENS.search(text))
    except OSError:
        return False


def nodelist_first(nodelist: str) -> str:
    """Return the first hostname from a NODELIST string.

    Handles Slurm bracket notation (``node[001-003,005]`` → ``node001``),
    comma-separated (``node01,node02`` → ``node01``), and plain strings.
    """
    bracket = nodelist.find("[")
    if bracket != -1:
        prefix = nodelist[:bracket]
        close = nodelist.find("]", bracket)
        inner = nodelist[bracket + 1 : close] if close != -1 else nodelist[bracket + 1 :]
        first_num = inner.split(",")[0].split("-")[0]
        return prefix + first_num
    return nodelist.split(",")[0]


def _parse_node0_hostname(log_text: str | None) -> str | None:
    """Extract node 0's hostname from the log header.

    Matches ``JOB_NODELIST=`` (all entry points) or ``SLURM_JOB_NODELIST=``
    (legacy Slurm logs).  Falls back to ``HOSTNAME=`` for older logs.
    """
    if not log_text:
        return None
    header = log_text[:2000]
    m = RE_NODELIST.search(header)
    if m:
        return nodelist_first(m.group(1).strip())
    m = re.search(r"^HOSTNAME\s*=\s*(\S+)", header, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


def _find_buffer_assignment(job_dir: Path) -> Path | None:
    """Find the buffer assignment file for TraceLens communication analysis."""
    xla_dump = job_dir / "xla_dump"
    if not xla_dump.is_dir():
        return None
    candidates = sorted(xla_dump.glob("*jit_train_step*gpu_after_optimizations-buffer-assignment.txt"))
    return candidates[-1] if candidates else None


# ── Log parsing for analysis.json ────────────────────────────────────────────


def _compute_stats(values: list[float]) -> dict | None:
    """Compute mean/std/n for a list of values. Returns None if empty."""
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    var = max(sum(v * v for v in values) / n - mean * mean, 0.0)
    return {"mean": round(mean, 3), "std": round(math.sqrt(var), 3), "n": n}


def _parse_log_metadata(text: str, log_file: Path) -> dict:
    """Extract job metadata from the log header."""
    meta: dict = {}

    header = text[:2000]

    m = RE_JOB_ID_HEADER.search(header)
    if m:
        meta["job_id"] = m.group(1).strip()

    m = RE_JOB_NAME.search(header)
    if m:
        meta["job_name"] = m.group(1).strip()

    m = RE_MODEL_NAME.search(header)
    if m:
        meta["model"] = m.group(1).strip()

    m = RE_NNODES.search(header)
    if m:
        meta["num_nodes"] = int(m.group(1))

    m = RE_EXP_TAG.search(header)
    if m:
        meta["exp_tag"] = m.group(1).strip()

    # Infer timestamp from log filename (local_YYYYMMDD_HHMMSS_... or slurm job)
    fname = log_file.stem
    ts_m = re.search(r"(\d{8})_(\d{6})", fname)
    if ts_m:
        try:
            dt = datetime.strptime(
                f"{ts_m.group(1)}_{ts_m.group(2)}", "%Y%m%d_%H%M%S"
            )
            meta["timestamp"] = dt.isoformat()
        except ValueError:
            pass

    if "timestamp" not in meta:
        try:
            mtime = log_file.stat().st_mtime
            meta["timestamp"] = datetime.fromtimestamp(
                mtime, tz=timezone.utc
            ).isoformat()
        except OSError:
            pass

    return meta


# Signal-based exit codes that indicate external termination, not a
# training bug.  128+N where N is the signal number.
_CANCEL_EXIT_CODES = {
    130,  # SIGINT  (Ctrl-C)
    143,  # SIGTERM (scancel, Slurm timeout)
}


def _detect_job_status(text: str, log_file: Path) -> dict:
    """Determine job status from log text and file age.

    Returns a dict with:
      - status:    completed | failed | cancelled | running | unknown
      - exit_code: int or None

    Semantics:
      completed — JOB SUMMARY with exit 0
      failed    — JOB SUMMARY with non-zero exit (training error)
      cancelled — JOB SUMMARY with signal exit code (scancel / timeout / ^C)
      running   — no JOB SUMMARY, log modified within last 15 min
      unknown   — no JOB SUMMARY, log stale (SIGKILL / OOM / orphaned)
    """
    if RE_JOB_SUMMARY.search(text):
        m = RE_JOB_STATUS.search(text)
        if m:
            word, code = m.group(1), int(m.group(2))
            if word == "SUCCESS":
                status = "completed"
            elif code in _CANCEL_EXIT_CODES:
                status = "cancelled"
            else:
                status = "failed"
            return {"status": status, "exit_code": code}
        return {"status": "completed", "exit_code": None}

    try:
        age = time.time() - log_file.stat().st_mtime
        if age < _RUNNING_THRESHOLD_S:
            return {"status": "running", "exit_code": None}
    except OSError:
        pass

    return {"status": "unknown", "exit_code": None}


def _parse_step_data(text: str) -> list[dict]:
    """Parse per-step metrics from log text.

    Returns a list of dicts with keys: step, seconds, tflops, mfu, tgs, loss,
    sorted by step number so warmup/steady partitioning is correct even when
    log output is batched per rank (e.g. k8s NFS logs).
    """
    steps = []
    for m in RE_STEP_LINE.finditer(text):
        steps.append({
            "step": int(m.group(1)),
            "seconds": float(m.group(2)),
            "tflops": float(m.group(3)),
            "mfu": float(m.group(4)),
            "tgs": float(m.group(5)),
            "loss": float(m.group(6)),
        })
    steps.sort(key=lambda s: s["step"])
    return steps


def _compute_tgs_summary(steps: list[dict], num_nodes: int = 1) -> dict:
    """Compute TGS summary matching tgs_tagger.py windows.

    Multi-node jobs log one entry per node per step, so each logical step
    has ``num_nodes`` consecutive entries.  Window sizes are scaled by
    ``num_nodes``.  Ranges use actual step numbers from the log data
    (correct for checkpoint-restored jobs that don't start at step 0).
    """
    all_tgs = [s["tgs"] for s in steps]
    all_step_nums = [s["step"] for s in steps]
    unique_steps = sorted(set(all_step_nums))

    # Auto-detect entries per step from the data. Per-rank logs (k8s) have
    # 1 entry/step despite num_nodes>1; Slurm logs have N entries/step.
    # Override only for windowing; the caller's num_nodes is used for display.
    effective_nodes = num_nodes
    entries_per_step = len(all_tgs) // len(unique_steps) if unique_steps else 1
    if entries_per_step != num_nodes:
        effective_nodes = entries_per_step

    warmup_n = WARMUP_STEPS * effective_nodes
    steady_n = STEADY_STATE_STEPS * effective_nodes
    warmup = all_tgs[:warmup_n]
    steady = all_tgs[warmup_n: warmup_n + steady_n]
    tail_start = warmup_n + steady_n
    tail = all_tgs[tail_start:] if len(all_tgs) > tail_start else []

    def _step_range(start_idx: int, end_idx: int) -> str:
        if start_idx >= len(unique_steps):
            return "?"
        first = unique_steps[start_idx]
        last = unique_steps[min(end_idx, len(unique_steps) - 1)]
        return f"{first}-{last}" if first != last else str(first)

    result: dict = {"all": _compute_stats(all_tgs)}

    w = _compute_stats(warmup)
    if w:
        w["range"] = _step_range(0, WARMUP_STEPS - 1)
    result["warmup"] = w

    s = _compute_stats(steady)
    if s:
        s_logical = len(steady) // effective_nodes if effective_nodes else len(steady)
        s["range"] = _step_range(WARMUP_STEPS, WARMUP_STEPS + s_logical - 1)
    result["steady"] = s

    t = _compute_stats(tail)
    if t:
        t_pos = WARMUP_STEPS + STEADY_STATE_STEPS
        t_logical = len(tail) // effective_nodes if effective_nodes else len(tail)
        t["range"] = _step_range(t_pos, t_pos + t_logical - 1)
    result["tail"] = t

    if result["all"]:
        result["all"]["range"] = _step_range(0, len(unique_steps) - 1)

    result["num_nodes"] = num_nodes
    result["entries_per_step"] = effective_nodes

    return result


def _parse_one_tracelens_csv(csv_path: Path) -> dict | None:
    """Parse a single gpu_events_averages.csv for summary percentages."""
    if not csv_path.is_file():
        return None
    try:
        summary: dict = {}
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric_type = row.get("type", "").strip()
                if metric_type in (
                    "computation_time", "exposed_comm_time",
                    "idle_time", "total_comm_time", "busy_time",
                ):
                    try:
                        summary[metric_type] = round(float(row["percent"]), 2)
                    except (ValueError, TypeError, KeyError):
                        pass
        return summary if summary else None
    except (OSError, csv.Error):
        return None


def _parse_tracelens_summary(job_dir: Path) -> dict | None:
    """Parse TraceLens gpu_events_averages.csv for summary percentages.

    Layouts:
      * 1-node/proc: ``tracelens/<ts>/csvs/gpu_events_averages.csv``.
      * 1-GPU/proc (via ``monkey_patch_maxtext.py``'s xplane-tagging shim):
        ``tracelens/<ts>/<host>.proc<N>/csvs/gpu_events_averages.csv`` —
        one csvs dir per local rank.  Because ``monkey_patch_maxtext.py``
        serializes ``stop_trace``, successive same-host writes typically
        land in distinct per-second ``<ts>`` dirs, so each ``<ts>`` usually
        contains exactly one ``<host>.proc<N>`` stem; any that contain
        more than one are averaged here.

    When multiple profiles exist (periodic profiling, or the per-local-rank
    fan-out under 1-GPU/proc), uses the latest timestamp directory (most
    representative of steady-state).  In SPMD the per-rank figures agree
    to within ~1 %, so picking the latest ``<ts>``'s value (or averaging
    co-resident stems) is already a faithful high-level summary.
    """
    tl_dir = job_dir / "tracelens"
    if not tl_dir.is_dir():
        return None

    ts_dirs = sorted(d for d in tl_dir.iterdir() if d.is_dir())
    # Use the latest timestamp (most representative of steady-state).
    for ts_dir in reversed(ts_dirs):
        # Flat layout: <ts>/csvs/...
        result = _parse_one_tracelens_csv(ts_dir / "csvs" / "gpu_events_averages.csv")
        if result:
            return result
        # Nested layout: <ts>/<stem>/csvs/... — average across any stems
        # co-resident in this ts dir (usually just one under flock serialization).
        per_stem = [
            _parse_one_tracelens_csv(sub / "csvs" / "gpu_events_averages.csv")
            for sub in ts_dir.iterdir() if sub.is_dir()
        ]
        per_stem = [r for r in per_stem if r]
        if per_stem:
            keys = set().union(*(r.keys() for r in per_stem))
            return {
                k: round(sum(r[k] for r in per_stem if k in r)
                         / sum(1 for r in per_stem if k in r), 2)
                for k in keys
            }

    return None


def _build_analysis_json(
    job_dir: Path | None,
    log_file: Path | None,
    log_text: str | None,
    steps: list[dict] | None,
    tensorboard_dir: Path | None = None,
    job_time_window: tuple[float, float] | None = None,
) -> dict:
    """Build the analysis.json content from parsed data.

    *tensorboard_dir* is the host-side tensorboard directory parsed from
    the MaxText log.  When it lives outside *job_dir* (checkpointing
    jobs), the external profile path is recorded so the dashboard can
    serve xplane files from the shared directory.
    """
    result: dict = {"analyzed_at": datetime.now(tz=timezone.utc).isoformat()}

    # Metadata from log
    if log_text and log_file:
        result.update(_parse_log_metadata(log_text, log_file))
        result["job_status"] = _detect_job_status(log_text, log_file)

    # Per-step data and TGS summary
    if steps:
        nn = result.get("num_nodes", 1)
        result["tgs"] = _compute_tgs_summary(steps, num_nodes=nn)
        result["tgs_per_step"] = [s["tgs"] for s in steps]
        result["mfu_per_step"] = [s["mfu"] for s in steps]
        result["loss_per_step"] = [s["loss"] for s in steps]
        result["seconds_per_step"] = [s["seconds"] for s in steps]
        unique_steps = sorted(set(s["step"] for s in steps))
        result["step_begin"] = unique_steps[0]
        result["step_end"] = unique_steps[-1]

    if log_text and RE_RAY_ACTOR.search(log_text):
        result["uses_ray"] = True

    # Artifact paths (relative to job_dir where possible)
    if job_dir:
        artifacts: dict = {}
        if log_file:
            try:
                artifacts["log_file"] = str(log_file.relative_to(job_dir.parent))
            except ValueError:
                artifacts["log_file"] = str(log_file)

        hlo = _find_hlo_file(job_dir)
        if hlo:
            artifacts["hlo_file"] = str(hlo.relative_to(job_dir))

        xplane = _find_xplane_files(job_dir, tensorboard_dir, job_time_window)
        if xplane:
            local = [str(f.relative_to(job_dir)) for f in xplane
                     if f.is_relative_to(job_dir)]
            if local:
                artifacts["xplane_files"] = local
            has_external = any(not f.is_relative_to(job_dir) for f in xplane)
            if has_external and tensorboard_dir:
                artifacts["profile_dir"] = str(
                    tensorboard_dir / "plugins" / "profile"
                )

        buf = _find_buffer_assignment(job_dir)
        if buf:
            artifacts["buffer_assignment"] = str(buf.relative_to(job_dir))

        tl_dir = job_dir / "tracelens"
        if tl_dir.is_dir():
            ts_dirs = sorted(
                d.name for d in tl_dir.iterdir()
                if d.is_dir() and (d / "csvs").is_dir()
            )
            # Filter to node 0's timestamps when possible
            node0 = _parse_node0_hostname(log_text)
            if node0 and xplane:
                node0_ts = {f.parent.name for f in xplane
                            if f.name.startswith(node0 + ".")}
                filtered = [ts for ts in ts_dirs if ts in node0_ts]
                if filtered:
                    ts_dirs = filtered
            if ts_dirs:
                artifacts["tracelens_profiles"] = ts_dirs
            artifacts["tracelens_dir"] = "tracelens/"

        if tensorboard_dir:
            artifacts["tensorboard_dir"] = str(tensorboard_dir)
            run_name = _extract_config_param(log_text, "run_name")
            if run_name:
                event_logdir = tensorboard_dir / run_name
                if event_logdir.is_dir():
                    artifacts["tensorboard_logdir"] = str(event_logdir)

        result["artifacts"] = artifacts

        # TraceLens summary
        tl_summary = _parse_tracelens_summary(job_dir)
        if tl_summary:
            result["tracelens_summary"] = tl_summary

    return result


def _save_analysis_json(job_dir: Path, data: dict) -> None:
    """Write analysis.json to the job directory."""
    out = job_dir / "analysis.json"
    try:
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        _step(f"Saved {out}")
    except OSError as e:
        _warn(f"Failed to write analysis.json: {e}")


def _analysis_is_current(job_dir: Path, log_file: Path | None) -> bool:
    """Check if analysis.json is up-to-date relative to the log file.

    Returns True (skip re-analysis) only when ALL of these hold:
      1. analysis.json exists and is newer than the log file
      2. The recorded job_status is not "running"

    A "running" status always triggers re-analysis because the job may
    have finished (or crashed) since. All other statuses are terminal:
    completed/failed/cancelled have a JOB SUMMARY and a final log;
    unknown means the log is stale with no summary (SIGKILL/OOM) so
    re-analyzing won't produce different results.
    """
    analysis_path = job_dir / "analysis.json"
    if not analysis_path.is_file():
        return False
    if not log_file or not log_file.is_file():
        return False

    try:
        analysis_mtime = analysis_path.stat().st_mtime
        log_mtime = log_file.stat().st_mtime
    except OSError:
        return False

    if log_mtime > analysis_mtime:
        return False

    # Only "running" is non-terminal. A running job's log will eventually
    # either grow (caught by mtime check above) or go stale (transitions
    # to "unknown" on next analysis). All other statuses are final.
    try:
        with open(analysis_path) as f:
            data = json.load(f)
        status = data.get("job_status", {}).get("status")
        if status == "running":
            return False
    except (OSError, json.JSONDecodeError, AttributeError):
        return False

    return True


# ── Resolve job directory from input path ────────────────────────────────────


def _resolve_job_dir(path: Path) -> Path | None:
    """Resolve a path (log file, directory, etc.) to its job directory."""
    if path.is_dir():
        return path

    if path.is_file() and path.suffix in _LOG_SUFFIXES:
        # Log file sits alongside the job dir in outputs/
        # Strip .log and any TGS suffix to find the base name
        stem = path.stem
        # Remove TGS tag if present
        m = re.match(r"(.*?)(?:-[^-]*-\d+N-[^-]*-TGS_[\d.]+)?$", stem)
        base = m.group(1) if m else stem
        parent = path.parent

        # Try exact stem match first
        candidate = parent / stem
        if candidate.is_dir():
            return candidate

        # Try base name match
        candidate = parent / base
        if candidate.is_dir():
            return candidate

        # Try job ID match (Slurm: "9909-...", k8s: "k8s-YYYYMMDD-HHMMSS-XXXX-...",
        # local: "local_YYYYMMDD_HHMMSS_XXXX-...")
        id_m = re.match(r"^(k8s-[\d]+-[\d]+-[\da-f]+|local_[\d_]+[\da-f]+|\d+)-", stem)
        if id_m:
            job_id = id_m.group(1)
            for entry in sorted(parent.iterdir()):
                if entry.is_dir() and entry.name.startswith(f"{job_id}-"):
                    return entry

    return None


# ── Tool runners ─────────────────────────────────────────────────────────────


def _run_tgs_tagger(log_file: Path) -> int:
    """Run tgs_tagger.py on the log file.

    Never passes ``-f`` — tgs_tagger will skip renames if the job is
    still running and perform them once it finishes.  ``analyze_job -f``
    only bypasses the staleness check; it does not force renames.
    """
    cmd = [sys.executable, str(SCRIPT_DIR / "tgs_tagger.py"), str(log_file)]
    result = _run(cmd, "tgs_tagger")
    return result.returncode


def _run_irlens(hlo_file: Path, extra_args: list[str] | None = None) -> int:
    """Run IRLens_analyze_hlo_ir.py on the HLO file."""
    cmd = [sys.executable, str(SCRIPT_DIR / "IRLens_analyze_hlo_ir.py"), str(hlo_file)]
    if extra_args:
        cmd.extend(extra_args)
    result = _run(cmd, "IRLens")
    return result.returncode


def _tracelens_output_dir(job_dir: Path, xplane_file: Path) -> Path:
    """Derive a per-profile TraceLens output directory.

    xplane files live under ``<run>/tensorboard/plugins/profile/<ts>/<host>.xplane.pb``.
    We use the ``<ts>`` directory name to key each TraceLens output so that
    periodic profiling (``profile_periodically_period``) produces separate
    results per profiling window: ``tracelens/<ts>/csvs/*.csv``.

    In 1-GPU-per-process mode the xplane filename carries a ``.proc<N>``
    segment (``<host>.proc<N>.xplane.pb`` — see ``monkey_patch_maxtext.py``'s
    ``_maybe_tag_profiler_output_with_local_rank``), so multiple per-GPU
    reports from the same host would collide at the flat path.  We nest
    those under the xplane's stem: ``tracelens/<ts>/<host>.proc<N>/csvs/*.csv``.
    """
    ts_dir = xplane_file.parent.name
    base = job_dir / "tracelens" / ts_dir
    stem = xplane_file.name.removesuffix(".xplane.pb")
    # Keep the historical flat layout for 1-node/proc (single process per
    # host → one ``<host>.xplane.pb``).  Nest only when the filename tells
    # us the 1-GPU/proc shim is in play.
    if ".proc" in stem:
        return base / stem
    return base


def _tracelens_is_done(output_dir: Path) -> bool:
    """Check if TraceLens output already exists for a given profile."""
    csvs_dir = output_dir / "csvs"
    return csvs_dir.is_dir() and any(csvs_dir.iterdir())


def _run_tracelens(
    xplane_file: Path,
    output_dir: Path,
    buffer_assignment: Path | None = None,
) -> int:
    """Run TraceLens_generate_perf_report_jax on the xplane file.

    Writes to a temporary directory first and renames on success so that
    ``_tracelens_is_done`` never sees partial/corrupt output from a
    failed run (e.g. incomplete xplane.pb still being written).
    """
    if not shutil.which("TraceLens_generate_perf_report_jax"):
        _warn(
            "TraceLens not installed. Install with:\n"
            "      pip install git+https://github.com/AMD-AGI/TraceLens.git"
        )
        return 1

    tmp_dir = output_dir.with_name(output_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model_name = xplane_file.stem.replace(".xplane", "")
    xlsx_path = tmp_dir / f"{model_name}_tracelens_report.xlsx"
    csvs_dir = tmp_dir / "csvs"

    cmd = [
        "TraceLens_generate_perf_report_jax",
        "--profile_path", str(xplane_file),
        "--output_xlsx_path", str(xlsx_path),
        "--output_csvs_dir", str(csvs_dir),
    ]
    result = _run(cmd, "TraceLens")
    if result.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _warn(
            "TraceLens failed. If this is a TF 2.19+/xprof environment,\n"
            "      see skills/performance-analysis/tracelens-patches.md"
        )
    else:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        tmp_dir.rename(output_dir)
        _step(f"TraceLens output: {output_dir}")
    return result.returncode


# ── Process one job ──────────────────────────────────────────────────────────


def process_job(
    path: Path,
    *,
    force: bool = False,
    skip_tgs: bool = False,
    skip_irlens: bool = False,
    skip_tracelens: bool = False,
    irlens_args: list[str] | None = None,
) -> int:
    """Analyze a single job. Returns 0 on success, nonzero on any failure."""
    _banner(f"Analyzing: {path}")

    # Resolve job directory
    job_dir = _resolve_job_dir(path)
    log_file: Path | None = None

    if path.is_file() and path.suffix in _LOG_SUFFIXES:
        log_file = path
    elif job_dir:
        log_file = _find_log_file(job_dir)

    if job_dir:
        print(f"  Job directory: {job_dir}")
    if log_file:
        print(f"  Log file:      {log_file}")

    if not job_dir and not log_file:
        _err(f"Cannot resolve job directory or log file from: {path}")
        return 1

    # ── Staleness check ──
    if not force and job_dir and _analysis_is_current(job_dir, log_file):
        _step(f"analysis.json is up-to-date (job finished, log unchanged) — skipping")
        return 0

    # ── Parse log for structured data ──
    log_text: str | None = None
    steps: list[dict] | None = None
    if log_file:
        try:
            log_text = log_file.read_text(errors="replace")
            steps = _parse_step_data(log_text)
        except OSError:
            pass

    rc = 0

    # ── Resolve tensorboard_dir from log ──
    outputs_dir = job_dir.parent if job_dir else None
    tensorboard_dir = _extract_tensorboard_dir(log_text, outputs_dir) if outputs_dir else None
    jtw = _job_time_window(job_dir, log_file) if job_dir else None
    if tensorboard_dir and job_dir and not tensorboard_dir.is_relative_to(job_dir):
        _step(f"External tensorboard_dir: {tensorboard_dir}")

    # ── TGS tagger ──
    if skip_tgs:
        _skip("tgs_tagger", "skipped (--skip-tgs)")
    elif log_file and _log_has_tgs_data(log_file):
        if _run_tgs_tagger(log_file) != 0:
            rc = 1
        # tgs_tagger may have renamed log file and/or job dir — re-resolve
        if job_dir and not job_dir.exists():
            job_id_m = re.match(r"^(k8s-[\d]+-[\d]+-[\da-f]+|local_[\d_]+[\da-f]+|\d+)-", job_dir.name)
            if job_id_m:
                parent = job_dir.parent
                for entry in sorted(parent.iterdir()):
                    if entry.is_dir() and entry.name.startswith(f"{job_id_m.group(1)}-"):
                        job_dir = entry
                        break
        if log_file and not log_file.exists() and job_dir:
            log_file = _find_log_file(job_dir)
            if log_file:
                try:
                    log_text = log_file.read_text(errors="replace")
                    steps = _parse_step_data(log_text)
                except OSError:
                    pass
    elif log_file:
        _skip("tgs_tagger", "no Tokens/s/device data in log yet")
    else:
        _skip("tgs_tagger", "no log file found")

    # ── IRLens ──
    if skip_irlens:
        _skip("IRLens", "skipped (--skip-irlens)")
    elif job_dir:
        hlo_file = _find_hlo_file(job_dir)
        if hlo_file:
            if _run_irlens(hlo_file, irlens_args) != 0:
                rc = 1
        else:
            _skip("IRLens", "no xla_dump/*jit_train_step*gpu_after_optimizations.txt found")
    else:
        _skip("IRLens", "no job directory found")

    # ── TraceLens ──
    # xplane traces are written once per profiling window and never change.
    # With profile_periodically_period, multiple profiles may exist at
    # different training steps.  Each gets its own output directory keyed
    # by the profile timestamp, and already-analyzed profiles are skipped.
    if skip_tracelens:
        _skip("TraceLens", "skipped (--skip-tracelens)")
    elif job_dir:
        xplane_files = _find_xplane_files(job_dir, tensorboard_dir, jtw)
        if xplane_files:
            # SPMD: all hosts run the same program, so we only analyze
            # node 0's xplane.  Distributed writes may scatter hosts across
            # multiple timestamp dirs — filtering by node 0 naturally
            # deduplicates: one node-0 file per profiling step.
            node0 = _parse_node0_hostname(log_text)
            unique_xplanes: list[Path] = []
            if node0:
                unique_xplanes = [f for f in xplane_files if f.name.startswith(node0 + ".")]
            if not unique_xplanes:
                # Fallback: one file per timestamp dir (first alphabetically)
                by_ts: dict[str, list[Path]] = {}
                for xf in xplane_files:
                    by_ts.setdefault(xf.parent.name, []).append(xf)
                unique_xplanes = [sorted(hosts)[0] for hosts in by_ts.values()]

            buffer_assignment = _find_buffer_assignment(job_dir)
            analyzed = 0
            skipped = 0
            for xf in unique_xplanes:
                out_dir = _tracelens_output_dir(job_dir, xf)
                if _tracelens_is_done(out_dir):
                    skipped += 1
                    continue
                if _run_tracelens(xf, out_dir, buffer_assignment) != 0:
                    rc = 1
                else:
                    analyzed += 1

            total = len(unique_xplanes)
            if analyzed > 0:
                _step(f"TraceLens: analyzed {analyzed} profile(s)")
            if skipped > 0:
                _skip("TraceLens", f"{skipped}/{total} profile(s) already analyzed")
            if total > 1:
                _step(f"{total} profiling windows found (periodic profiling)")
        else:
            _skip("TraceLens", "no *.xplane.pb files found")
    else:
        _skip("TraceLens", "no job directory found")

    # ── Save analysis.json ──
    if job_dir and job_dir.is_dir():
        analysis = _build_analysis_json(
            job_dir, log_file, log_text, steps, tensorboard_dir, jtw,
        )
        _save_analysis_json(job_dir, analysis)

    # ── Summary ──
    print()
    artifact_names = []
    if log_file and _log_has_tgs_data(log_file):
        artifact_names.append("TGS data")
    if job_dir and _find_hlo_file(job_dir):
        artifact_names.append("HLO dump")
    if job_dir and _find_xplane_files(job_dir, tensorboard_dir, jtw):
        artifact_names.append("xplane trace")
    if artifact_names:
        print(f"  {GREEN}Available artifacts:{RESET} {', '.join(artifact_names)}")
    else:
        print(f"  {YELLOW}No analysis artifacts found yet.{RESET}")

    return rc


# ── Path resolution ──────────────────────────────────────────────────────────


def resolve_paths(args: list[str], default_dir: str) -> list[Path]:
    """Resolve CLI arguments to a list of job paths (files or directories)."""
    if not args:
        args = [default_dir]

    paths: list[Path] = []
    for arg in args:
        p = Path(arg)

        if p.is_file() or p.is_dir():
            paths.append(p)
            continue

        # Try glob expansion
        expanded = sorted(glob.glob(arg))
        if expanded:
            paths.extend(Path(f) for f in expanded if Path(f).is_file() or Path(f).is_dir())
        else:
            if any(c in arg for c in "*?["):
                _err(f"Pattern matched nothing: {arg}")
            else:
                _err(f"Path does not exist: {arg}")

    return paths


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_job",
        description=(
            "Detect available artifacts in a MaxText job's output directory "
            "and run the appropriate analysis tools (tgs_tagger, IRLens, TraceLens)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
arguments:
  PATH can be one or more of:
    - Log file:  specific .log/.out file
    - Job dir:   job output directory (with or without log symlink)
    - Glob:      pattern expanding to log files or directories

  Default: outputs directory (from JOB_WORKSPACE or relative to script)

artifact detection:
  - TGS data:     log file contains "Tokens/s/device:" lines
  - HLO dump:     <job_dir>/xla_dump/*jit_train_step*gpu_after_optimizations.txt exists
  - xplane trace: <job_dir>/**/*.xplane.pb exists

examples:
  %(prog)s outputs/12345-JAX-llama2-70b.log
  %(prog)s outputs/12345-JAX-llama2-70b/
  %(prog)s outputs/local_2026*
  %(prog)s -f outputs/12345-JAX-llama2-70b.log    # force re-analysis
  %(prog)s --skip-tracelens outputs/*.log          # skip TraceLens
""",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="bypass staleness check and force re-analysis even if analysis.json is current",
    )
    parser.add_argument(
        "--skip-tgs",
        action="store_true",
        help="skip tgs_tagger step",
    )
    parser.add_argument(
        "--skip-irlens",
        action="store_true",
        help="skip IRLens step",
    )
    parser.add_argument(
        "--skip-tracelens",
        action="store_true",
        help="skip TraceLens step",
    )
    parser.add_argument(
        "--irlens-args",
        nargs=argparse.REMAINDER,
        default=None,
        help="pass-through arguments to IRLens (e.g. --irlens-args --op communication)",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="log files, job directories, or glob patterns to analyze",
    )
    return parser


_DASHBOARD_PREFERRED_PORT = 8080
_DASHBOARD_PORT_RANGE = range(8080, 8100)


def _find_dashboard_port() -> int | None:
    """Scan a small port range to find a running perf_server."""
    for port in _DASHBOARD_PORT_RANGE:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.3)
            s.connect(("127.0.0.1", port))
            s.close()
            return port
        except OSError:
            continue
    return None


def _dashboard_hint() -> None:
    """Print a one-line hint about the web dashboard."""
    server_script = SCRIPT_DIR / "perf_server.py"
    if not server_script.exists():
        return
    port = _find_dashboard_port()
    if port is not None:
        print(
            f"\n{CYAN}Dashboard:{RESET} "
            f"http://127.0.0.1:{port}  (running; "
            f"tunnel with `ssh -L {port}:localhost:{port}` for remote access)"
        )
    else:
        print(
            f"\n{CYAN}Start dashboard:{RESET}  "
            f"utils/perf_server.py  "
            f"(localhost only — tunnel via SSH for remote access)"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    default_dir = os.environ.get(
        "JOB_WORKSPACE", str(SCRIPT_DIR.parent / "outputs")
    )

    paths = resolve_paths(args.paths, default_dir)
    if not paths:
        print(f"{RED}No valid paths to analyze.{RESET}")
        return 1

    last_rc = 0
    for p in paths:
        rc = process_job(
            p,
            force=args.force,
            skip_tgs=args.skip_tgs,
            skip_irlens=args.skip_irlens,
            skip_tracelens=args.skip_tracelens,
            irlens_args=args.irlens_args,
        )
        if rc != 0:
            last_rc = rc

    _dashboard_hint()
    return last_rc


if __name__ == "__main__":
    sys.exit(main())
