---
name: pre-commit-audit
description: Comprehensive pre-commit verification checklist with five independent responsibilities. (1) Launcher path coverage - verify a change to any launcher-chain file preserves correct behavior across all 16 combinations of entry point × launch mode × stack (Steps 1-4 + 5.1). (2) Ancillary scripts smoke - syntax / help / read-only / caller checks for any `.sh` or `.py` outside the launcher chain (Step 5.2; covers analysis utilities, sourced libraries, debug helpers, sweep tooling). (3) Code quality and design review (Step 6) - propose-first surface of code smells (duplication, long functions, magic numbers, deep nesting, unclear naming, primitive obsession, etc.) and design-decay signals (5th case in a switch, N-th env-var read, hand-rolled retry loops); auto-fix mechanical findings, hold design-shaped ones for explicit go-ahead. (4) Docs / comments / format-consistency (Step 7) - check any commit for stale prose, trailing-comment alignment drift, broken anchors / missing files in links, drifted cross-references, and this skill itself drifting from the code it describes. (5) Sensitive-info leak scan (Step 8) - cluster hostnames, internal IPs, vendor mount paths, hard-coded credentials, internal job IDs; final security gate. Trigger keywords - "verify all launcher paths", "trace launcher chain", "audit entry × launch × stack", "path coverage", "(entry × launch × stack) matrix", "post-launch teardown verification", "pre-commit audit", "before commit", "ready to commit", "verify scripts / utils not broken", "smoke-test the changed scripts", "any utility script broken", "code quality", "design review", "code smells", "tighten and polish", "avoid quality decay", "revisit design choice", "scrub leaks", "check for sensitive info before commit", "any docs or skills need update", "any stale comments", "any inaccurate comments", "comment alignment", "link policy", "broken anchors". Use when modifying `_train.sh`, `_train_with_ray.sh`, `_ray_actor.py`, `_container.sh`, `_job.sbatch`, `_k8s_job.sh`, `in_container_run.sh`, `run_local.sh`, `submit.sh`, `k8s_submit.sh`, `utils/run_setup.sh`, `utils/ray_cluster.sh`, `utils/monkey_patch_maxtext.py`, `utils/coredump.sh`, `utils/stage_timeout.sh`, or anywhere else in the launcher chain. Also use proactively before opening any PR (Steps 5.2, 6, 7, 8 apply universally to all changes that touch code / docs / comments), when investigating a path-specific bug ("this only happens in K8s + 1-gpu-per-process"), after adding a new entry point / launch mode / stack option, after touching any analysis utility (`utils/analyze_job.py`, `utils/perf_server.py`, `utils/profile_drill.py`, `utils/slurm_job_monitor.sh`, etc.), or after editing any doc or skill in the repo (Step 7 catches cross-reference drift).
---

# Pre-commit audit

Comprehensive pre-commit verification of a code change in this repo. The skill bundles five independent audits that share a single output format and a single "ready / not ready" recommendation:

1. **Launcher path coverage (Steps 1–4 + 5.1)** — when the change touches the launcher chain, verify all **4 entry points × 2 launch modes × 2 stacks = 16 combinations** still work. Most agents verify the path they personally cared about and miss the other 15.
2. **Ancillary scripts smoke (Step 5.2)** — when the change touches any `.sh` / `.py` outside the launcher chain (analysis utilities, sourced libraries, debug helpers, sweep tooling), syntax-check, help-smoke, read-only invoke, and cross-check callers. The 16-cell launcher matrix can be all green and a renamed function in a sourced library can still leave `utils/analyze_job.py` broken on the next invocation.
3. **Code quality and design review (Step 6)** — on any commit that changes code, surface code smells (duplication, long functions, magic numbers, deep nesting, primitive obsession, etc.) and design-decay signals (5th case in a switch, N-th env-var read in different files, hand-rolled retry loops). Propose-first: auto-fix the mechanical ones, hold the design-shaped ones for go-ahead.
4. **Docs / comments / format-consistency (Step 7)** — on any commit, check that prose stays in sync with code: stale in-file comments, trailing-comment alignment drift, doc cross-references that point at moved/renamed code, link-policy compliance (broken anchors, missing files, scheme conventions), and (especially) this skill itself drifting out of sync with the launcher topology. Runs *after* code quality (post-rename / post-extraction) and *before* the leak scan (so its prose edits feed into the security gate).
5. **Sensitive-info leak scan (Step 8)** — on any commit, scan the post-Steps-6+7 state for cluster hostnames, internal IPs, vendor mount paths, hard-coded credentials, internal job IDs, and other deployment-specific identifiers that don't belong in this open-source repo. Runs *last* — final security gate before commit.

Step 9 emits the launcher matrix (when path coverage ran) plus the per-step audit lists.

The methodology behind the launcher-path-coverage half (Steps 1–4 + 5.1) is the one used to verify the `fix/training-teardown` series of fixes (`os._exit` in `monkey_patch_maxtext.py`, `bash` instead of `exec` in `_train_with_ray.sh`, guarded `ray.kill` in `_ray_actor.py`) — three changes that each affected a different subset of the 16 combinations and that all needed to land before any single combination was provably clean. The ancillary-smoke, leak-scan, and doc-audit halves are general-purpose and apply to any commit that touches scripts / docs / comments.

## Audit posture: holistic, not per-file

Every step below has two scopes, and the holistic one almost always determines the recommendation:

1. **Per-change** — what's in the diff. Mechanical, runs in seconds, catches the obvious regressions. This is the entry point; it surfaces candidates.
2. **Holistic** — how the change fits into the codebase as a whole: its *trajectory* (does this continue or break a recent pattern?), its *architectural fit* (does it preserve or cross a documented layer boundary?), its *aggregation* (is this the 6th instance of a smell that was acceptable at 1 and ought to be standardized at 6?), and its *self-consistency* (does this skill itself, the docs, and the code still describe the same system?).

Two findings that look identical per-file can land with opposite recommendations holistically: a duplicated 5-line block is acceptable inline duplication the first time and a strong propose-extract the fifth time. A new env-var read is benign once and a code-smell trigger when the same env var is now read in 4 places. A 5-line shell function that fits its module is fine; the same function pasted into 3 modules where each could call a shared helper is a repo-wide concern.

**Re-verify and revisit are themselves holistic operations.** When the skill's self-consistency table (§7.3) calls for re-reading a section, the question is "does this section still describe the codebase as a whole, accurately?" — not "does it match the file I happened to edit?". When the design-choice revisit (§6.4) calls for revisiting a decision, the question is "does this decision still fit the system as it has actually evolved?" — not "is the change at hand consistent with the original decision in isolation?".

The holistic checks are listed alongside their per-change counterparts in each step. Skipping the holistic half passes the audit but misses the kind of decay that compounds across commits.

## When to use this skill — decision tree

```
Does the change touch any file in Step 2's lookup table (the launcher chain)?
│
├── Yes  → run all 9 steps (full audit) — Steps 1–5.1 cover the matrix; §5.2
│          additionally covers any non-launcher script also touched in the same diff
│
└── No   → does the change touch any executable script or sourced library
           (`.sh` / `.py` outside Step 2's table)?
          │
          ├── Yes → run Step 5.2 (ancillary smoke) + Step 6 (code quality) + Step 7 (docs / comments / format) + Step 8 (leak);
          │        skip the launcher matrix in Step 9 — output the audit lists alone
          │
          └── No  → what does the change touch?
                   │
                   ├── Code only (no docs / comments)
                   │   → run Step 6 (code quality) + Step 8 (leak); Step 7 is a no-op
                   │
                   ├── Code + docs / comments
                   │   → run Step 6 (code quality) + Step 7 (docs / comments / format) + Step 8 (leak)
                   │
                   ├── Docs / comments only (no code)
                   │   → run Step 7 (docs / comments / format) + Step 8 (leak); Step 6 is a no-op
                   │
                   └── Nothing (e.g., reverting a typo in your own scratch file) → no audit needed
                   │
                   └── No  → no audit needed (e.g., reverting a typo in your own scratch file)
```

The skill's title says "pre-commit", but the right time to invoke it is **whenever you have a complete, conceptually-coherent change ready** — that's typically a few minutes before staging the diff with `git add`, *not* after a long branch with several unrelated commits accumulated. Audit each commit on its own, not the whole branch in one go; signal-to-noise drops sharply when many unrelated edits compete for the same audit.

## The 16-cell matrix

The three axes are independent:

| Axis | Values |
|---|---|
| **Entry point** | `submit.sh` (Slurm), `k8s_submit.sh` (Kubernetes), `run_local.sh` (single-host), `in_container_run.sh` (already-in-container) |
| **Launch mode** | 1-node-per-process (default; `LOCAL_WORLD_SIZE=1`), 1-gpu-per-process (`ONE_GPU_PER_PROCESS=true`; `LOCAL_WORLD_SIZE=8`) |
| **Stack** | `RAY=0` (no Ray, no observability), `RAY=1` (Ray actor + Prometheus + TensorBoard + metrics_exporter) |

Cells are not all equally important — production loss tests typically use `submit.sh × 1-node-per-process × RAY=1`, profile runs use 1-gpu-per-process variants, K8s users go through `k8s_submit.sh`, and `in_container_run.sh` is the manual debugging entry. **A change can be silently correct on the path you tested and silently broken on three others.**

## Decomposition: what the 16 cells actually exercise

Underneath the surface, the 16 cells reduce to:

- **2 distinct ways the container is reached** (one for `submit.sh` + `run_local.sh`, one for `k8s_submit.sh` + `in_container_run.sh`)
- **4 distinct post-`_train.sh` flows** (one per launch-mode × stack combination)

So in practice an audit only has to verify 2 + 4 = 6 distinct dynamic behaviors, and then check that each entry point feeds the right downstream flow.

### Two ways the container is reached

| Pair | Container-entry mechanism | EXIT-trap stack inside the container |
|---|---|---|
| `submit.sh`, `run_local.sh` | `_container.sh` runs `docker run`; the container's `bash -lcx FINAL_CMD` invokes `$MAXTEXT_RUNNER` (= `_train_with_ray.sh` if `RAY=1`, else `_train.sh`) | `_train_with_ray.sh`'s own EXIT trap (calls `stop_ray_cluster`) — only when `RAY=1`; otherwise no in-container trap |
| `k8s_submit.sh` (→ `_k8s_job.sh`), `in_container_run.sh` | the pod / shell **is** the container; no `_container.sh` `docker run` involved. `_k8s_job.sh` `exec`s into `in_container_run.sh` for rank 0 (and pipes through `tee` for non-rank-0) | `in_container_run.sh`'s `_cleanup_and_summary` trap (calls `stop_ray_cluster` when `RAY=1`) → `_train_with_ray.sh`'s own trap (also `stop_ray_cluster` when `RAY=1`) — redundant but idempotent |

### Four post-`_train.sh` flows

| # | Launch | Stack | What runs in `_train.sh` |
|---|---|---|---|
| (1) | 1-node-per-process | `RAY=0` | `python3 -u utils/monkey_patch_maxtext.py "${TRAIN_ARGS[@]}"` — single Python process, sees all local GPUs |
| (2) | 1-gpu-per-process | `RAY=0` | `for i in 0..LOCAL_WORLD_SIZE-1: python3 -u utils/monkey_patch_maxtext.py … &` then `wait` loop captures first non-zero rc |
| (3) | 1-node-per-process | `RAY=1` | `python3 -u _ray_actor.py …` → driver creates one `MaxTextTrainerActor` per node → `actor.run_training` does `subprocess.Popen(monkey_patch_maxtext.py)` + `p.wait()` |
| (4) | 1-gpu-per-process | `RAY=1` | same as (3), but the actor calls `_fan_out_one_gpu_per_proc` which Popens `LOCAL_WORLD_SIZE` subprocesses and waits on all |

`utils/monkey_patch_maxtext.py` is the leaf in all 4 flows — every fix below the launcher level lives there.

## Audit procedure

Use the decision tree at the top to decide which steps apply:

- **Steps 1–4 + 5.1: Launcher path coverage** — only when the change touches a file in Step 2's lookup table.
- **Step 5.2: Ancillary scripts smoke** — when the change touches any `.sh` / `.py` outside Step 2's table.
- **Step 6: Code quality, design review, and architectural fit** — every commit that changes code. Propose-first: surface smells and design questions, auto-fix only purely-mechanical ones, await go-ahead for the rest. **Holistic by default** — per-change findings (§6.1, §6.3) feed into a repo-wide trajectory check (§6.2) and an architectural-boundary check (§6.4); the recommendation almost always comes from the holistic half.
- **Step 7: Docs / comments / format-consistency** — every commit that changes code, docs, or comments. Runs *after* code quality (so renames / extracted helpers / merged duplicates have settled before prose is checked) and *before* the leak scan (so any prose edits made here are seen by the security gate).
- **Step 8: Sensitive-info leak scan** — every commit. Runs *last* so it scans the post-edited state — including any new code / comments / examples / links added by Steps 6 and 7 — and serves as the final pre-commit security gate.
- **Step 9: Output** — always; the launcher matrix is included only when Steps 1–4 + 5.1 ran.

### Step 1 — identify the change

> **Skip ahead to Step 5.2** if your change touches no file in Step 2's lookup table but touches some script/util elsewhere. **Skip ahead to Step 6** if it touches no script at all (docs / comments only). The launcher-path-coverage block (Steps 1–4 + 5.1) is meaningful only for launcher-chain edits; running the matrix on, say, a typo fix in `skills/profile-drill/SKILL.md` produces a 16-cell table of "shared" rows that conveys nothing.

Read the diff that needs verifying:

```bash
# All uncommitted changes
git diff
# Or a specific commit
git show <sha>
# Or a branch
git log --oneline main..HEAD
git diff main..HEAD --stat
```

Build the **change set**: which files changed, which functions / lines.

### Step 2 — map the change to flows

For each touched file, look up which of the 6 distinct behaviors it participates in:

| File | Reached by entry-pair (1) `submit.sh`/`run_local.sh` | Reached by entry-pair (2) `k8s_submit.sh`/`in_container_run.sh` | Used by post-`_train.sh` flow |
|---|---|---|---|
| `submit.sh` | yes | — | — |
| `_job.sbatch` | yes | — | — |
| `_container.sh` | yes | — | — |
| `k8s_submit.sh` | — | yes | — |
| `_k8s_job.sh` | — | yes | — |
| `in_container_run.sh` | — | yes | — |
| `run_local.sh` | yes (single-host variant) | — | — |
| `utils/run_setup.sh` | yes | yes | — |
| `_train_with_ray.sh` | yes (RAY=1) | yes (RAY=1) | (3), (4) |
| `_train.sh` | yes | yes | (1), (2), (3), (4) |
| `_ray_actor.py` | yes (RAY=1) | yes (RAY=1) | (3), (4) |
| `utils/monkey_patch_maxtext.py` | yes | yes | (1), (2), (3), (4) |
| `utils/ray_cluster.sh` | yes (RAY=1) | yes (RAY=1) | (3), (4) (head- and worker-side) |
| `utils/prometheus.sh` | yes (RAY=1, head only) | yes (RAY=1, head only) | (3), (4) |
| `utils/metrics_exporter.sh` | yes (RAY=1) | yes (RAY=1) | (3), (4) |
| `utils/coredump.sh` | yes | yes | (1), (2), (3), (4) |
| `utils/detect_ip.sh` | yes (read by `_job.sbatch`, `_container.sh`, `utils/ray_cluster.sh`) | yes (read by `_k8s_job.sh`, `utils/ray_cluster.sh`) | (3), (4) (binds Ray GCS to private IP) |
| `utils/stage_timeout.sh` | yes (orchestration tier, in `_job.sbatch` only) | — | — |

A change in `utils/monkey_patch_maxtext.py` (the bottom of the chain) lights up **all 16** cells. A change in `_ray_actor.py` lights up **8** (the RAY=1 half). A change in `k8s_submit.sh` lights up **4** (the K8s column). And so on.

Files not listed (e.g., `utils/job_dir.sh`, `utils/split_script_args.sh`, `utils/resolve_model_name.sh`, `utils/code_provenance.sh`, `utils/release_gpu.sh`, `utils/preflight.sh`, `utils/docker_utils.sh`, `utils/pick_port.sh`) are sourced by one or more of the entries above; treat a change there as transitively affecting all flows that reach the file. Use `rg --files-with-matches "utils/<file>"` to find direct callers when in doubt.

### Step 3 — for each affected flow, walk the chain

For each of the 6 distinct behaviors that the change touches, trace the call chain end-to-end and answer:

1. **Does the call still reach the changed function?** (Or in the case of removed code: does the caller cope with the removal?)
2. **Does the new behavior preserve the contract** the caller assumed? Specifically:
   - **exit code** — does `$?` / `p.returncode` / `sys.exit(rc)` propagate correctly through every shell hop and Python frame?
   - **side effects** — are files / TB events / checkpoints / log lines written before any fast-exit path is taken?
   - **resource cleanup** — are Ray daemons / Prometheus / TensorBoard / metrics_exporter / docker container all stopped (or knowingly orphaned)?
3. **Does any `set -e`, `set -o pipefail`, `trap`, or `exec` interaction silently swallow or amplify the change?**

### Step 4 — check the cross-cutting edge cases

These are the bugs the recent audit caught; recheck every time a launcher file changes:

| Edge case | Diagnostic | Fix pattern |
|---|---|---|
| **`trap … EXIT` followed by `exec`** | The trap dies with the replaced bash. Cleanup never runs. | Replace `exec script.sh` with `bash script.sh "$@"; exit "$?"`, or remove the trap before `exec` (`trap - EXIT`). |
| **Unprotected call in a Python `finally:`** | If the call raises, `sys.exit(rc)` is suppressed and Python exits 1 with a traceback even on success. | Wrap in `try: …; except Exception: pass`. |
| **`subprocess.Popen` + `.wait()` without pipe management** | Large stdout/stderr can deadlock the child. | Use `stdout=subprocess.DEVNULL` if not needed, or `Popen(..., stdout=PIPE).communicate()`. |
| **`{ … } \| tee` under `set -o pipefail`** | A transient `tee` failure aborts the whole pipeline (including the upstream commands). | Append `\|\| true`, or capture `${PIPESTATUS[0]}` and only fail on the upstream's status. |
| **`exec > >(tee -a …)` process substitution** | tee is not waited on; SIGPIPE / NFS-write errors in tee can kill the writer or truncate logs. | Use a foreground pipeline with `${PIPESTATUS[0]}`, or accept the limitation and document. |
| **`pkill -f <pattern>`** | Matches across the whole PID namespace. Inside a Docker container without `--pid=host` this is bounded; outside a container or with `--pid=host`, it can kill another job's processes (very real once observed: `pkill -f tensorboard` killing the user's standalone TB viewer on the same host). | Track and kill by recorded PID, or anchor the pattern to `$JOB_DIR` / `$OUTPUT_PATH`. |
| **`os._exit(rc)` skipping atexit handlers** | Side effects normally run during `sys.exit` (Orbax flush, TB final flush, JAX device release) are skipped. | Verify the leaf calls (e.g., `maxtext_train.main`) commit those side effects before returning, and add `os.sync()` for kernel page-cache safety on async-mounted NFS. Provide an env-var opt-out (`MAXTEXT_FAST_EXIT=0`). |
| **`SIGTERM` cascade from one rank's failure** | When one rank's training subprocess dies, srun SIGTERMs the rest, producing all-`exit 143` in the log. The first non-zero rc in `wait` order is the cause; the rest are noise. | Surface the **first rank** to exit non-zero (not the last), and treat exit-143 cascades after a non-walltime failure as secondary. |
| **`exit $?` with an intervening command** | `$?` is clobbered to the intervening command's status. | `_rc=$?; intervening_command; exit "$_rc"`. |
| **Race in pre-init helpers (`_maybe_preinit_jax_distributed`, profiler shim)** | If a helper relies on env vars that aren't set in some flow (e.g., `LOCAL_RANK` is only set in 1-gpu-per-process), the helper must be a no-op there. | Guard with `[[ -z "${LOCAL_RANK:-}" ]] && return` or equivalent in Python. |
| **`stop_ray_cluster` running twice** | On K8s / `in_container_run.sh` it runs in two traps (the script's and `_train_with_ray.sh`'s). | Make sure each step is idempotent (`ray stop --force ‖ true`, `pkill ‖ true`). |
| **`--time=` walltime under-budgeted** | Slurm SIGTERMs all ranks at exact `--time=`. Step `T-1` sometimes makes it just before SIGTERM, sometimes after. | Use the adaptive formula `walltime = ceil(1.5 × (compile_overhead + steps × step_time_seconds) / 3600) hours` — measure step time from a 15-step calibration run, multiply by 1.5× to absorb step-time variance and Ray/JAX teardown overhead. |

### Step 5 — smoke-test the changed code paths

The change set may touch (a) launcher-chain files that participate in the 16-cell matrix, (b) ancillary scripts / utilities that don't (analysis tools, sourced libraries, debug helpers, sweep tooling), or (c) both. Run §5.1 for category (a) and §5.2 for category (b); the categories are independent and a single change set can need both.

#### 5.1 — Launcher-chain smoke (when Steps 1–4 produced affected flows)

A 1-node, 15-step `dataset_type=synthetic` smoke is cheap and exercises the whole chain end-to-end. Use `run_local.sh` (no Slurm round-trip needed) on any idle GPU node:

```bash
# Pick an idle node — `sinfo -o '%n %T' -h` for the default partition,
# or `sinfo -p <PARTITION> -o '%n %T' -h` if you need to target one explicitly.
ssh <node> "cd <repo> && nohup ./run_local.sh 70b:smoke-noray -- steps=15 dataset_type=synthetic > /tmp/smoke_noray.log 2>&1 < /dev/null &"
ssh <node> "cd <repo> && RAY=1 nohup ./run_local.sh 70b:smoke-ray -- steps=15 dataset_type=synthetic > /tmp/smoke_ray.log 2>&1 < /dev/null &"
# Add ONE_GPU_PER_PROCESS=true to either to exercise the 1-gpu-per-process path
```

Each smoke takes ~5 minutes for `llama2-70b` on a single MI300X-class node.  Run all 4 (RAY × launch-mode) combinations in parallel on different nodes.  Once each completes (look for `Status: SUCCESS (exit 0)` in the log), live-verify the binding side-effects:

```bash
ssh <head_node> 'ss -tlnp 2>/dev/null | grep -E ":(8265|9190|6006|55080)"'
# Expected (for RAY=1 jobs):
#   127.0.0.1:8265   ray dashboard
#   127.0.0.1:9190   prometheus query API
#   127.0.0.1:6006   tensorboard
#   0.0.0.0:55080    RAY metrics export (intentional, head needs to scrape worker exporters)

ssh <head_node> 'pgrep -af tensorboard && pgrep -af "ray::" && pgrep -af prometheus'
# Expected: all gone after job exit (cleanup trap ran)
```

For Slurm-only or K8s-only paths, fall back to a real `submit.sh` / `k8s_submit.sh` smoke against a single allocated node.

#### 5.2 — Ancillary scripts and utilities smoke (when the change touches anything outside Step 2's lookup table)

A change to `utils/analyze_job.py`, `utils/perf_server.py`, `utils/slurm_job_monitor.sh`, `debug_repro.sh`, sourced libraries like `utils/detect_ip.sh` or `utils/job_dir.sh`, or any other repo script that doesn't appear in Step 2's table will *not* be exercised by §5.1. The launcher matrix can be 16-of-16 green and a renamed function in `utils/parse_model_spec.sh` can still leave `utils/analyze_job.py` broken on the next invocation. These breakages surface days or weeks later, when someone runs the broken tool, not at commit time.

For each touched non-launcher script (`.sh` or `.py` not in Step 2's table), run a graduated smoke:

**(a) Syntax / parse check** — cheapest, catches typos and missing brackets in seconds:

```bash
files=$(git diff --name-only HEAD)
for f in $files; do
  case "$f" in
    *.sh)  bash -n "$f"               && echo "OK syntax: $f"  || echo "FAIL syntax: $f" ;;
    *.py)  python3 -m py_compile "$f" && echo "OK compile: $f" || echo "FAIL compile: $f" ;;
  esac
done
```

This is mandatory for every touched `.sh` / `.py` regardless of category — the 16-cell launcher smoke does *not* exercise import-time errors in `_ray_actor.py`'s submodules, for example, until the actor is constructed.

**(b) Help / usage smoke** — confirms the script's CLI parser and import chain still work end-to-end. For each executable (non-sourced) script touched:

```bash
bash <script>.sh --help 2>&1 | head -20      # most repo scripts respond to --help or -h
python3 utils/<script>.py --help 2>&1 | head -20
```

If the script doesn't accept `--help`, run with no args (most repo scripts print usage on no-args). If it doesn't print usage either, document that explicitly and move to step (c).

**(c) Read-only invocation against real data** — for analysis / diagnostic utilities, point at an existing job's outputs/ directory:

| Script | Smoke invocation |
|---|---|
| `utils/analyze_job.py` | `utils/analyze_job.py outputs/<recent-job>/<log>.log \| head -30` |
| `utils/perf_server.py` | `utils/perf_server.py --port 8080 &` then `curl -s localhost:8080/api/jobs \| head`; kill |
| `utils/profile_drill.py` | `utils/profile_drill.py outputs/<job-with-xplane>/.../*.xplane.pb 2>&1 \| head` |
| `utils/slurm_job_monitor.sh` | `utils/slurm_job_monitor.sh -j <recent-jobid> --dry-run` (or no-args usage) |
| `utils/tail_job_log.sh` | `utils/tail_job_log.sh <recent-jobid> 2>&1 \| head -5` |
| `utils/tag_tgs.sh` | `utils/tag_tgs.sh outputs/<recent-job>/ 2>&1 \| head` |
| `utils/tgs_tagger.py` | `utils/tgs_tagger.py outputs/<recent-job>/<log>.log` |
| `utils/cleanup_artifacts.sh` | `utils/cleanup_artifacts.sh --dry-run` (must support `--dry-run`; if not, document the proposed invocation) |

If the script doesn't yet support `--dry-run` and could mutate state, *don't* live-smoke; document the proposed invocation in the audit output and ask the user to run it manually.

**(d) Side-effecting utilities** — for scripts that mutate state (`utils/release_gpu.sh`, `utils/check_ecc.sh`, `utils/preflight.sh`, `utils/reservation.sh`, the host-cmd controller), verify behavior in a controlled environment (idle node, dry-run mode) or document the verification as user-run with the proposed invocation. Never live-smoke a side-effecting utility against a node owned by another user's job.

**(e) Cross-script callers** — when the change renames or removes a function in a sourced library (`utils/job_dir.sh`, `utils/split_script_args.sh`, `utils/detect_ip.sh`, `utils/coredump.sh`, `utils/code_provenance.sh`, `utils/parse_model_spec.sh`, `utils/parse_job_args.sh`, `utils/resolve_model_name.sh`, `utils/pick_port.sh`, `utils/docker_utils.sh`, `utils/git_summary.sh`, etc.), grep the rest of the repo for callers and verify each call site still type-checks against the new signature:

```bash
# Find function definitions in the touched library
for f in $files; do
  case "$f" in *.sh)
    funcs=$(rg -nP '^\s*(?:function\s+)?([a-z_][a-z0-9_]+)\s*\(\s*\)' "$f" \
           | rg -oP '\b[a-z_][a-z0-9_]+(?=\s*\(\))' | sort -u)
    for fn in $funcs; do
      echo "=== $fn (defined in $f) ==="
      rg -n --type sh -- "\b$fn\b" -g "!$f" || echo "(no callers — safe to remove)"
    done ;;
  esac
done
```

For each non-empty caller list, open the caller and verify it doesn't pass / consume an old signature. **Static lookups only — don't trust the function-extraction regex above to be perfect; use it as a starting list and confirm each call site by reading.**

**Output of §5.2:** per script, a one-line status combining the smoke layers run, e.g. `utils/analyze_job.py: bash -n OK; --help OK; against outputs/14982 OK` or `utils/release_gpu.sh: bash -n OK; --help OK; live-smoke deferred (mutates state — proposed user-run command included in audit)`.

### Step 6 — code quality and design review (propose-first; modify only with explicit go-ahead)

A change can pass every behavioral check and still degrade code quality — duplicated logic accumulating, growing function bodies, magic numbers proliferating, inconsistent abstractions piling up across files. These don't break anything today; they make the *next* change harder. Catch them before they compound.

This step is **propose-first** by default: the agent surfaces smells and proposed fixes but does not modify the change set without explicit approval. Code-quality calls involve trade-offs (deeper helpers vs. inlined clarity, abstraction cost vs. duplication cost) that the user is best positioned to weigh; the agent's job is to surface candidates and make the case. The default action column distinguishes purely-mechanical fixes (which can be auto-applied) from design-shaped ones (which must wait for go-ahead).

#### 6.1 — Per-change code-smell scan

For each touched file, walk this checklist. Report findings in the audit output; auto-fix only the mechanical variants and explicitly mark them; propose the rest with a one-paragraph rationale per finding. Each finding then feeds into §6.2 to decide whether it's an isolated occurrence or part of a growing repo-wide pattern.

| Smell | Diagnostic | Default action |
|---|---|---|
| **Duplicated code** | Same 5+ lines appear in 2+ places, or same logic with minor variations | Propose extraction for cross-file duplicates (suggest helper name + new home); auto-fix only when both sites are in the same file and the extraction is mechanical |
| **Long function** | > 50 lines for shell, > 80 for Python, *or* any function whose purpose can't be summarized in one sentence | Propose breakdown; suggest helper-function names |
| **Long file** | > 500 lines for shell, > 800 for Python; multiple unrelated responsibilities | Propose split with proposed new file boundaries |
| **Deep nesting** | More than 3 levels of `if` / `for` / `case` indentation | Auto-fix when the fix is mechanical (early-return guards); propose otherwise |
| **Magic numbers** | Numeric literals without context (`sleep 5`, `timeout=30`, `port=8265`) | Auto-fix: name as a top-of-file constant with a one-line rationale |
| **Unclear / generic naming** | `tmp`, `x`, `do_thing`, `helper`, `process`, single-letter outside loop indices | Propose rename (action-and-target form: `read_config_file`, not `helper`) |
| **Dead code** | Functions never called; branches under permanent `if false` / `0`; commented-out blocks > 2 lines | Auto-fix: delete. If the comment is preserved intentionally as documentation, convert it to a normal explanatory comment |
| **Inconsistent abstraction levels** | One function mixes high-level orchestration with low-level operations (shell function does both `git fetch` and per-byte string parsing) | Propose: extract the lower-level work into a helper |
| **Stringly-typed config / enum** | Multiple `if [[ "$mode" == "X" ]]` branches scattered across the codebase | Propose: name the modes as constants in a single file; have callers reference them |
| **Repeated regex / pattern** | Same regex appears 3+ times | Propose: define as a top-level constant in the file that owns it |
| **Comment-as-documentation overlap** | Comment says exactly what the next line says (`# increment counter` next to `i += 1`) | Auto-fix: delete the redundant comment; keep only intent / rationale / non-obvious-edge-case comments |
| **Primitive obsession** | Function signature with 5+ raw-string args | Propose: introduce a config struct / sourced-shell config block; refactor callers |
| **Inappropriate intimacy** | One file directly mutates another module's globals or imports private helpers (`_foo`); test code reaches into production internals | Propose: add a public accessor / setter; tighten the API surface |
| **Inconsistent style with neighbors** | New function uses spaces where surrounding file uses tabs; new helper is snake_case where the file is camelCase | Auto-fix: match the file's existing convention |
| **Hand-rolled retry / backoff loop** | New `for i in $(seq 1 N); do ...; sleep ...; done` pattern that already exists elsewhere | Propose: extract into `with_retries` (or use the existing helper if one is in `utils/`) |
| **Hardcoded path that should be a variable** | A new `/path/to/something` that ought to be derived from `$JOB_WORKSPACE`, `$OUTPUTS_DIR`, etc. | Auto-fix: replace with the appropriate variable |

#### 6.2 — Holistic code-smell trajectory check (walk outward from the diff)

The per-change scan finds smells *in* the diff. The holistic check finds out whether each smell is **isolated** (acceptable as-is) or **part of an accumulating pattern** (propose standardization, even when no individual instance is bad).

For each smell flagged in §6.1, do this two-step check:

1. **Extract the smell shape** as a grep pattern. Examples:
   - Hand-rolled retry: `for\s+\(\(.*\)\);\s+do\s+sleep` (or the bash-loop-then-sleep pattern more generally)
   - Magic timeout literals: `timeout=\d+|--timeout\s+\d+`
   - Repeated env-var read: `\$\{?<VARNAME>\b`
   - `try: … except Exception: pass`: `except\s+(Exception|BaseException):\s*$\s*pass\b`
2. **Sweep the repo** with the pattern; count occurrences:

```bash
rg -nP --type sh --type py -g '!outputs/**' -g '!.git/**' '<pattern>' | wc -l
```

Then apply this rule:

| Repo-wide count | Recommendation |
|---|---|
| 1 (just this change) | acceptable as-is; record nothing |
| 2 | acceptable; mention the existing siblings in the audit so the next change has context |
| 3 | propose a *follow-up* extraction commit (separate from the change at hand) |
| 4+ | propose extraction *now*, blocking on user confirmation; the cost of N+1 is low only if N is < 3 |

The thresholds are bash-of-thumb, not rigid; fewer instances in conceptually-coherent locations may be fine, more in deliberately-divergent locations may also be fine. Use judgment; record the count in the audit.

Beyond the per-smell sweep, also surface aggregate trajectory signals:

- **Recent commit history of the touched files** — `git log --oneline --follow <file> | head -20`. If the file has had 5+ commits in the last month for "fix X", "fix Y", "another tweak", that's a sign the file has outgrown its original design and the change at hand is the latest patch. Surface this as a *consider rewriting this module* note rather than a defect on the diff.
- **Files that move together** — `git log --pretty=format: --name-only --since='1 month ago' | sort | uniq -c | sort -rn | head` shows the most-edited files. If the change touches a file that's in the top-edited group, the audit should ask whether that file is becoming a kitchen sink.
- **Drift from the documented architecture** — see §6.4.

#### 6.3 — Per-change design-choice revisit

Some design questions are worth revisiting periodically. These are *not* defects to fix in the change at hand — they're signals that an existing design may have outgrown its original scope. Surface them; propose a re-design conversation; do not modify without explicit go-ahead.

| Trigger | When to escalate | Question to raise |
|---|---|---|
| **Switch with 5+ cases** | The change adds the 5th case to a `case` / `if-elif` / `match` block, or extends a 4-case block to 5 | Should this be a dispatch table / lookup map / polymorphism instead of inline branching? |
| **`try / except: pass` proliferating** | The change adds a 4th unprotected `except Exception: pass` near siblings that already use the same pattern | Are some of these errors actually actionable? Should we catch specific exception types? Logging the error before swallowing? |
| **New global variable** | The change adds a new `EXPORT FOO=...` at module scope | Can this be passed through env vars, function args, or a config struct? Globals couple unrelated callers. |
| **Same env var read in 3+ places** | The change introduces or grows a third callsite reading the same env var | Centralize the read in one place; have callers receive the parsed value via function arg |
| **Helper function with 6+ args** | The change grows a helper to take a 6th positional / keyword arg | Time for a config struct or builder pattern? |
| **Parallel data structures kept in sync manually** | Two arrays / dicts / lists that must stay in lockstep, with no enforcement | Collapse into a single struct-of-fields, or document the invariant and add an assertion |
| **N-th place doing the same thing differently** | The change introduces a 3rd implementation of an operation that's done 2 other ways elsewhere (e.g., 3 different ways to check Slurm job status) | Pick one; standardize the rest as follow-up |
| **Conditional that always-true / always-false in current setup** | New `if has_X: …` where `has_X` is always true on this stack | Either make `has_X` a real switch the user can flip, or remove the conditional entirely |

#### 6.4 — Architectural fit and boundary check (holistic)

The repo's stated architecture lives in `docs/architecture.md` and `docs/extensibility.md`. Both call out specific layer boundaries and ownership rules — for instance, "scheduler coupling is confined to the orchestration tier", "training is launcher-agnostic", "utilities are framework-agnostic". The change set must preserve those boundaries; if it doesn't, either the boundary needs explicit revision or the change needs reshaping.

For each touched file, identify:

1. **Which architectural layer does this file belong to?** Read the relevant paragraph in `docs/architecture.md`. If the layer isn't documented, that itself is a finding — propose adding a one-line description before merging.
2. **Does the change introduce dependencies that cross declared boundaries?** Specifically:
   - New `source ../<other-tier>/...` from a tier that wasn't supposed to depend on the other.
   - New `import` / `from X import` from a Python module that lives in a different layer than the importing file.
   - New shell-out (`$( ... )` or `>(...)` redirection) to a binary owned by another layer.
   - New environment variable that one layer expects another to set, without that setter being documented.
3. **Does the change add knowledge of one layer's internals to another layer's caller?** E.g., a downstream training file now has to know that the scheduler is Slurm specifically (referencing `SLURM_NODEID` in `_train.sh` would be a regression). If yes, surface the leak and propose plumbing the value through a layer-agnostic env var instead.

```bash
# Concrete recipe — flag cross-layer dependencies in the diff
files=$(git diff --name-only HEAD)
for f in $files; do
  # Layer of f (read from docs/architecture.md's prose; encode as map):
  #   submit.sh / k8s_submit.sh / _job.sbatch / _k8s_job.sh -> orchestration
  #   _container.sh -> container
  #   _train.sh / _train_with_ray.sh / _ray_actor.py -> training-runner
  #   utils/* -> utility (mostly layer-agnostic; see §6.4 caveats below)
  #
  # Then for f's new imports / sources / SLURM_*-style refs, check:
  #   - Is the referenced symbol from f's own layer or a known-shared dependency?
  #   - If from another layer, does docs/architecture.md or extensibility.md
  #     bless this dependency direction explicitly?
  echo "=== $f ==="
  rg -nP '^\+\s*(?:source|\.|import\s|from\s+\w+\s+import|SLURM_|JAX_COORDINATOR_|RAY_)' "$f" || true
done
```

Caveats for utilities (`utils/*`): most are framework-agnostic by design (`stage_timeout.sh`, `pick_port.sh`, `detect_ip.sh`). A few deliberately straddle the training boundary (`monkey_patch_maxtext.py`, `tgs_tagger.py`, `resolve_model_name.sh`) — see the footnote in `docs/extensibility.md` ("Axis 3"). Adding a new straddler is allowed but should be documented in the same commit; growing the existing straddlers' scope is a yellow flag worth surfacing.

When the diff *does* cross a boundary, the audit's verdict is one of:

- **Acceptable, document it** — the dependency is intentional; add a sentence in `docs/architecture.md` so the next agent knows.
- **Plumb it through** — the cross-layer reference can be replaced with a layer-agnostic mechanism (env var, function arg). Propose the replacement; ripple-effect estimate.
- **Pull back** — the dependency invalidates an architectural property the user cares about. Pull-back recommendation in the audit output.

#### 6.5 — Output

Three categorized lists:

- **Per-change auto-fixes applied (§6.1)** — `(file:line, smell-name, change)` for mechanical fixes the audit applied. Each entry includes a one-line description of what was changed.
- **Per-change proposed fixes pending confirmation (§6.1 + §6.3)** — `(file:line, smell-name-or-design-trigger, proposed-fix, ripple-effect)`. Each entry has:
  - **Why it's a smell / design concern** (1 sentence)
  - **Proposed fix** (concrete diff sketch or commit description)
  - **Ripple effect** (other files / callers affected; rough estimate of how many lines move)
  - **Suggested commit grouping** — usually a *separate* follow-up commit, not the one at hand. Bug fixes shouldn't be bundled with refactors; both reviewers and the git-log lose signal when they are
- **Holistic findings (§6.2 + §6.4)** — repo-wide observations the per-change view alone wouldn't catch:
  - **Smell-trajectory entries** — for each per-change smell that swept to count ≥ 3 elsewhere, a `(smell-shape, repo-wide-count, recommended-action)` row. Counts of 2 are recorded as "context, no action"; 3 triggers a follow-up-commit recommendation; 4+ blocks on confirmation now.
  - **Hot-spot files** — top-edited files (last month) in the change set; flagged as candidates for redesign rather than further patching.
  - **Architectural-fit findings** — per cross-layer dependency the diff introduces, the verdict from §6.4 (acceptable-document-it / plumb-through / pull-back) plus the proposed remedy.

If no findings in any list, state "code quality: no findings (per-change clean; holistic clean)" so the next reader knows both halves ran. Findings that the user defers ("not now") get a one-liner each — paste-able into a TODO file or issue tracker.

### Step 7 — docs, comments, and format-consistency check (mandatory before commit)

A change to launcher code that doesn't update the surrounding prose leaves landmines: docs reference functions that moved, code comments describe behavior that no longer matches, the file → flow table in this very skill goes out of sync with the launcher topology, and trailing-comment columns drift one space at a time. Catch these in the same pass instead of letting them rot.

This step runs **before** the leak scan (Step 8) on purpose: prose edits made here — fresh comments, new doc cross-references, updated examples — must be seen by the security gate. If the leak scan ran first and a Step 7 fix added a hostname to an example, the leak would slip through.

#### 7.1 — In-file comments in the touched files

For each file in the change set, re-read the comment blocks and check them against the new code:

| Stale-comment pattern | What to look for | Fix |
|---|---|---|
| **Behavior drift** | Comment describes what the code *used to* do (e.g., "exec into _train.sh" beside a `bash _train.sh; exit "$?"` line) | Rewrite the comment, or delete it if the new code is self-evident |
| **Symbol drift** | Comment names a function / variable / file path that the change renamed or moved | Update the reference, or replace with a description that doesn't pin specifics |
| **Job-ID / dated references** | Comment cites a specific incident ("see job 14837") that the code no longer relates to | Replace with the *symptom* ("asymmetric rank-N exit-1 on multi-node sweeps") and let the symptom outlive the incident |
| **TODO / FIXME / HACK / XXX** referencing the now-fixed problem | Comment explicitly says "TODO: fix the X bug" — and the change *is* the X fix | Delete the TODO; if it referenced a follow-up, file an issue and replace with a one-line link |
| **Counter-example comments** | Comment has a "DO NOT do X because Y" directive — but the change just made X safe | Either delete the warning, or update it to describe a still-relevant guardrail |
| **Line-number / size pinning** | Comment cites byte / line / step counts that the change shifted (e.g., "the next 12 lines are…") | Re-count and update, or rewrite to stop pinning |
| **Trailing-comment alignment drift** | Two or more consecutive lines in a usage / option / table block had their trailing `#` (or `//`) comments aligned at the same column, and the change shifted the pre-comment text on one of them — leaving the comments staircased (e.g., one column off) | Re-pad the spaces so all `#` line up at the same column. The longest pre-comment text in the block sets the comment column; pad shorter lines with spaces to match. Verify with `head -<N> <file> \| cat -A` (the `$` end-markers expose the trailing-space layout). Examples: option lists in shell scripts, env-var tables in `train_env.sh` / `container_env.sh`, ASCII tables in markdown |

```bash
# Quick scan helper for the touched files
files=$(git diff --name-only HEAD)
for f in $files; do
  echo "=== $f ==="
  rg -nP "(?:TODO|FIXME|HACK|XXX|deprecated|legacy|bug:|workaround)" "$f" || true
done
```

Manual review still required — most comment staleness is semantic, not pattern-match-able. The commands above only surface candidates.

#### 7.2 — Cross-references in docs and other skills

Find every file outside the change set that mentions the changed functions, file paths, or behaviors. The launcher chain is referenced widely; updates to one file commonly affect 5-10 doc / skill pages.

##### Markdown link policy (applies to every md file the change touches)

The repo follows a consistent link policy. Verify the diff doesn't violate it before committing:

| Rule | Required form | Forbidden forms |
|---|---|---|
| **Internal cross-doc** | Relative path, no leading `./` | Absolute paths (`/maxtext-slurm/docs/...`), `file://`, repo-root `/...` |
| **Cross-tier internal** (skill → doc, doc → repo root) | `..`-anchored relative path (e.g., `../docs/k8s-job-submission.md` from a skill, `../../README.md` from a nested skill) | Absolute paths |
| **External** | `https://` URL, no trailing `/` redundancy | `http://` (insecure), bare URLs, `www.` without scheme |
| **Anchor refs** | GitHub-slugger kebab-case slug of the heading text — see "Slug algorithm" below | Custom slugs, hash-encoded entities, raw heading text |
| **Link style** | Inline `[text](target)` form | Reference-style `[text]: target` (the repo doesn't use this; mixing styles hurts grep-ability) |
| **First-mention linkability** | When linkable terms (Slurm, JAX, Kubernetes, Docker, Ray, Prometheus, TensorBoard, Cursor) appear for the first time in a doc, link them to their official site; later mentions in the same doc remain plain | Repeated linking of the same term within one file (visual clutter) |
| **No links to non-shipping files** | Reference only files that exist in the open-source repo | Linking to project-private docs that don't ship (e.g., a sweep-prompt that lives only on the deployer's branch) — these read as broken to anyone who checks out the public branch |

###### Slug algorithm (github-slugger)

This is the algorithm GitHub uses for heading anchors. Use it (not your own kebab-casing) when constructing or verifying a `#anchor`:

```python
import re
def slug(heading_text):
    s = heading_text.lower()
    s = re.sub(r'`([^`]+)`', r'\1', s)   # strip code spans, keep content
    s = re.sub(r'[^a-z0-9 _-]', '', s)   # remove non-alphanumeric / space / hyphen / underscore
    s = s.replace(' ', '-')              # spaces → hyphens; do NOT collapse consecutive hyphens
    return s
```

Two cases that bite naïve implementations:

- **Em-dash / Unicode arrows / `&` in headings** (e.g., `## (container → native)` or `## the step − total kernel row`) produce **runs of consecutive hyphens** in the slug because each removed Unicode character collapses against its surrounding spaces. The link must preserve the doubled hyphen exactly. A naïve implementation that collapses `\s+` → `-` will mis-flag these as broken.
- **Headings with code spans** (e.g., `### Run \`profile_drill.py\``) — the backticks are stripped, but the inner code is kept. Slug = `run-profile_drillpy` (with `_` preserved and `.py` reduced to `py` because `.` is removed).

###### Audit recipe (Python — handles fenced code blocks, inline code spans, and the slug algorithm correctly)

```python
import re, os
mds = []
for r, ds, fs in os.walk("."):
    if "/.git" in r or "/outputs" in r: continue
    for f in fs:
        if f.endswith(".md"):
            mds.append(os.path.normpath(os.path.join(r, f)))

def slug(h):
    h = h.lower()
    h = re.sub(r'`([^`]+)`', r'\1', h)
    h = re.sub(r'[^a-z0-9 _-]', '', h)
    return h.replace(' ', '-')

def strip_inline(line):
    # Drop ``…`` then `…` spans so `[text](target)` examples in docs aren't scanned.
    line = re.sub(r'``[^`]+``', '', line)
    return re.sub(r'`[^`]+`', '', line)

anchors = {}
for m in mds:
    in_code = False; anchors[m] = set()
    for line in open(m):
        if line.startswith("```"):
            in_code = not in_code; continue
        if in_code: continue
        mo = re.match(r'^#{1,6}\s+(.*?)\s*$', line)
        if mo: anchors[m].add(slug(mo.group(1)))

LINK = re.compile(r'(?<!\!)\[[^\]]+\]\(([^)]+)\)')
for src in mds:
    in_code = False
    for ln, line in enumerate(open(src), 1):
        if line.startswith("```"):
            in_code = not in_code; continue
        if in_code: continue
        for href in LINK.findall(strip_inline(line)):
            if href.startswith(("http://", "https://", "mailto:")): continue
            if href.startswith("#"):
                if href[1:] not in anchors[src]:
                    print(f"{src}:{ln}  href={href}  (anchor missing)")
                continue
            t, _, a = href.partition("#")
            resolved = os.path.normpath(os.path.join(os.path.dirname(src), t))
            if not os.path.exists(resolved):
                print(f"{src}:{ln}  href={href}  (file missing)")
            elif a and resolved.endswith(".md") and a not in anchors.get(resolved, set()):
                print(f"{src}:{ln}  href={href}  (anchor missing in {resolved})")
```

Run the recipe whenever the change set touches any `.md` file or any heading. Two-step gate:
1. Are there any `(file missing)` lines? → fix immediately; this is always a real bug.
2. Are there any `(anchor missing)` lines? → review each. The slug algorithm above is correct for vanilla GitHub markdown; if a hit appears in a doc that uses a different renderer (e.g., MkDocs, Sphinx), the slug rules differ and you may need a renderer-specific check.

Common gotchas the recipe handles correctly (do not undo these in maintenance):

- **Fenced code blocks** (lines bounded by triple-backtick fences) — code samples frequently contain link-shaped substrings (the recipe itself does). Scanning them produces an avalanche of false positives. The `in_code` toggle skips lines between fences.
- **Inline code spans** — link-shaped text inside single- or double-backtick spans gets caught by the regex unless stripped first. The `strip_inline` helper drops single- and double-backtick spans before regex matching.
- **Image references** (the `!`-prefixed image-link form) — the negative lookbehind in the link regex skips them; image src checks are out of scope (they often live outside the repo).
- **`mailto:` URLs** — treat as external; never resolve as files.

When extending the audit recipe, document examples *without* literal link-shaped substrings — describe the syntax in prose instead of demonstrating it inline. Triple-backticks adjacent to inline backtick spans defeat naïve `strip_inline` regexes; the tradeoff isn't worth a bullet-point demo.

##### Symbol and file-path cross-references

```bash
# 1) Symbol-level cross-references — bash function names, Python entry points
for f in $files; do
  # Bash function defs and Python def/class
  syms=$(rg -nP '^\s*(?:[a-z_][a-z0-9_]*\s*\(\)\s*\{|def\s+([A-Za-z_][A-Za-z0-9_]*)|class\s+([A-Za-z_][A-Za-z0-9_]*))' "$f" \
        | rg -oP '\b[a-zA-Z_][a-zA-Z0-9_]+(?=\s*[\(\{:])' | sort -u)
  for s in $syms; do
    rg -l --type md -- "\b$s\b" docs/ skills/ README.md 2>/dev/null
  done
done | sort -u

# 2) File-path cross-references — `_train.sh`, `utils/monkey_patch_maxtext.py`, etc.
for f in $files; do
  rg -l --type md -- "\b$(basename "$f")\b" docs/ skills/ README.md 2>/dev/null
done | sort -u

# 3) Inline-cited code blocks (line-range references in our doc style: NN:MM:filepath)
# After any non-trivial edit, citations to that file's line numbers may have shifted
rg -nP '^\s*\d+:\d+:[^\s]+' docs/ skills/ README.md
```

For each hit, open the doc and verify the prose still describes what the code does. Common needs:

- **Frontmatter triggers** in `skills/<skill>/SKILL.md` — when the change adds / removes a launcher file, the matching `description` block in `pre-commit-audit/SKILL.md` (this skill) and `CLAUDE.md` may need new file names.
- **Architecture docs** — `docs/architecture.md`, `docs/observability.md`, `docs/extensibility.md`, `docs/performance.md`, `docs/job-submission.md`, `docs/k8s-job-submission.md`, `docs/debugging.md`, `docs/tooling.md`, `docs/model-configs.md` all describe parts of the launcher chain. When `_train.sh` learns a new flow, `architecture.md`'s call-graph paragraph is the most-likely-to-be-stale reference.
- **README** — high-level overview; usually only needs an update if a public-facing entry-point changes name or is added/removed.
- **Other skills' triggers** — `skills/job-log-triage/SKILL.md` has rows mapping log signatures to root causes; if the change creates a new failure signature, add it. If the change *fixes* a signature, mark it as historical or remove if the affected jobs predate the fix.
- **Inline-cited line ranges** in this skill or other skills — if the change shifts line numbers in a cited file, update the cited range or convert to a symbol-level reference (`<funcname>` instead of `NN:MM:<file>`) so it doesn't break next time.

#### 7.3 — Self-consistency: does this skill itself need updating?

This skill is a methodology document; the change being audited can invalidate parts of it. Walk these sections in order:

| Section to check | When it goes stale | What to update |
|---|---|---|
| **Frontmatter `description`** (file list, trigger keywords) | A new launcher-chain file was added, an old one was renamed / removed, or a new failure-class was identified that warrants its own trigger keyword | Add / remove the file name. Add a new trigger keyword if the change generalizes |
| **The 16-cell matrix** | A new entry point, launch mode, or stack option landed; any axis was removed | Update the cell count. Note: the *math* (`4 × 2 × 2 = 16`) is hard-coded in the prose — search and update both |
| **"Two ways the container is reached"** table | A new entry point bypasses both existing pairs (e.g., a serverless / lambda entry) | Add a third pair, with its container-entry mechanism and trap stack |
| **"Four post-`_train.sh` flows"** table | A new launch mode or stack option lands; the dispatch logic in `_train.sh` changes | Renumber and add the new flow; update the cross-references in §3 (Step 3) |
| **Step 2 file → flow lookup table** | Any change touching the launcher-chain files | Confirm the table row for each touched file still accurately lists the flow numbers and entry-pairs the file participates in. Add rows for newly-introduced files |
| **Step 4 cross-cutting edge cases** | The change discovers a new edge case worth memoizing for the next agent, or invalidates one that's listed | Add a new row when a new bug class surfaces; mark as "fixed by `<commit>`" rather than deleting if removal would lose context for old branches |
| **Step 5.1 launcher smoke recipe** | The smoke commands no longer exercise the changed code (e.g., a flag default changed making `dataset_type=synthetic` insufficient to hit the changed path) | Update the example invocation, or add a second smoke variant |
| **Step 5.2 ancillary scripts table** | A new analysis utility lands; an existing one was renamed; an existing one gained a `--dry-run` mode that should now be the recommended smoke | Add / rename the row; promote read-only invocations to `--dry-run` where supported |
| **Step 6.1 code-smell scan table** | A new code smell becomes recurrent in the codebase (e.g., a build-system migration introduces a new pattern of inconsistency that ought to be flagged), or an existing row is no longer applicable | Add / remove rows; update the default-action column when a smell that used to be auto-fixable becomes propose-only or vice versa |
| **Step 6.2 holistic trajectory thresholds** | The codebase has grown enough that the 1/2/3/4+ thresholds no longer match the right "how many is too many" intuition (e.g., the codebase deliberately tolerates more inline duplication than 3) | Adjust the threshold table; document the rationale |
| **Step 6.3 design-choice revisit triggers** | The codebase establishes a new pattern that ought to be revisited periodically (e.g., a third retry-loop variant lands), or removes an existing one | Add / remove triggers |
| **Step 6.4 architectural-layer map** | A new tier is introduced (e.g., a new external scheduler shim, a new observability sidecar), or an existing layer's responsibilities are reshuffled | Update the layer mapping; if `docs/architecture.md` and `docs/extensibility.md` both need updates, propose those as part of the same change |
| **Step 7.1 stale-comment patterns** | The change introduces a new comment-drift class worth memoizing (e.g., a build-system-specific comment style appears that needs alignment, or a new convention for `// SAFETY:` blocks) | Add a new row to the stale-comment-patterns table |
| **Step 7.2 link policy table** | The repo adopts a new doc renderer with different slug rules, a new external-link convention (e.g., always-archived URLs), or a new internal-doc layout (e.g., `docs/v2/...` parallel tree) | Update the policy table; update the slug algorithm if the renderer changed |
| **Step 8 leak patterns and allowlist** | The change introduces a new convenience default that shouldn't trigger the leak scan, or a new sensitive identifier that should | Add an allowlist row (with rationale) or a new pattern row |
| **Output format** items | Any of the above changes adds or removes evidence the agent must collect | Update the numbered list of items in §"Output format" |
| **"Notes on this skill itself"** | Anything material above changed | Re-read the whole bottom-of-file note block; rewrite the bullet describing whichever section just shifted |

Concrete trigger: **whenever the change set touches any file listed in the Step 2 lookup table, re-open this skill and read the frontmatter description, the file → flow table, and the Step 4 edge cases.** That catches almost every self-consistency drift. If anything reads false now, fix it in the same change.

#### 7.4 — Output

Per file, three categorized lists:

- **In-file comments updated** — `(file:line, old, new)` triples. Auto-fix is fine here when the comment is clearly stale; no confirmation round-trip needed.
- **Cross-reference updates** — `(other-file:line, what was wrong, what to write instead)`. For doc edits that *change meaning* (e.g., changing the architecture diagram description), propose first and confirm.
- **Self-consistency updates to this skill** — list each section that needed editing, plus a one-line summary of the edit. If no edits were needed, state "skill unchanged: change is fully covered by existing sections" so the next reader knows the check ran.

### Step 8 — sensitive-info leak check (mandatory before commit)

Runs **last** — after Step 6's code-quality fixes and Step 7's prose edits — so any new comments, examples, or links those steps introduced are also scanned. The leak scan is the final security gate before commit.

Launcher-chain edits frequently touch low-level system info — cluster hostnames, IPs, partition names, internal paths — and the most common leak vector is a copy-pasted example or a debug-time comment that never got genericized. Run a leak scan over the change set before declaring the audit clean.

This is an open-source repo. Anything cluster-specific in code or comments leaks the deploying organization's infrastructure topology even when no credential is exposed.

```bash
# Diff scope — pick whichever matches the work
files=$(git diff --name-only HEAD)                    # uncommitted
files=$(git diff --name-only <base>..HEAD)            # branch vs base
files=$(git show --name-only --pretty='' <sha>)       # specific commit

# Run each pattern; review every hit and fix or genericize.
for f in $files; do
  echo "=== $f ==="
  rg -nP '<pattern>' "$f" || true   # see table below
done
```

| # | Pattern (ripgrep `-P`) | What it leaks | Fix |
|---|---|---|---|
| 1 | `\b(chi\|<your-prefix>)\[?\d+` | The deploying cluster's hostname pattern | Replace with `<node>`, `<nodelist>`, `<host>` (or `node[N-M]` for syntax examples) |
| 2 | `\b(?:\d{1,3}\.){3}\d{1,3}\b` | Any concrete IP — public or private — that wasn't deliberately documented | Allowed: `127.0.0.1`, `0.0.0.0`, RFC5737 doc IPs (`192.0.2.x`, `198.51.100.x`, `203.0.113.x`), and explicitly-marked RFC1918 placeholders. Anything else → replace with `<host>`, `<private-ip>`, `<public-ip>`, or one of the RFC5737 ranges in docs |
| 3 | `\b1[4-5]\d{3}\b` (current cluster's job-ID range — adjust to your range) in **code or comments** | Refers to internal job tracking; doesn't generalize and dates the comment | Remove the job ID and describe the *symptom* instead (e.g., "asymmetric rank-N exit-1 on multi-node sweeps"). Ok in `outputs/`, log files, and external docs not shipped in the repo. **False positives:** plain numbers in this range often appear as timeout values (`14400` = 4 h), byte counts, or port numbers — review every hit; if it isn't preceded by `job`, `JOB_ID`, `sbatch`, `squeue`, or appears as a timeline reference, it's almost certainly not a job ID |
| 4 | `partition\s*=\s*["']?[a-z0-9_-]+["']?` hard-coded outside `${VAR:-default}` | Implies the deploying cluster's partition layout | Use `${PARTITION:-}` or pass via the existing flag plumbing |
| 5 | `(?:/mnt/[^/\s]+/[a-z0-9_-]+\|/home/[a-z0-9_-]+\|~[a-z0-9_-]+)` | User home / scratch / vendor-specific mount paths (e.g., `/mnt/<vendor>/<user>`, `/home/<user>/...`) | Use `$JOB_WORKSPACE`, `$HOME`, `$OUTPUTS_DIR`, or repo-relative paths. Vendor-specific mount roots like `/mnt/<storage-product-name>/...` should be replaced with `<your-shared-path>` or set via env var |
| 6 | `(?i)(token\|secret\|password\|apikey\|api_key)\s*[:=]\s*["']?[A-Za-z0-9_\-]{16,}` | Hard-coded credential | Move to `.local.sh` (already gitignored), `.env`, or a secrets manager |
| 7 | `https?://(?!github\.com\|huggingface\.co\|docs\.\|en\.wikipedia\.org\|tools\.ietf\.org\|datatracker\.ietf\.org)[a-z0-9.-]+\.(?:com\|org\|net\|io\|ai\|local\|internal)` | Org-private URLs / image registries / Slack / Linear | Genericize to `${DOCKER_REGISTRY}`, `<your-tracker-url>`, `<your-chat-channel>` |
| 8 | `(?<=[\s'"=])[a-z][a-z0-9_-]*\.(?:local\|internal\|corp\|prod\|stage)\b` | Internal DNS suffixes | Replace with `<host>` or genericize via env var |
| 9 | `\bC\d{4,}\b` (Slack channel) `\bT\d{8,}\b` (Slack team) `\b[A-Z0-9]{8,}\b` patterns matching common ID schemas | Cross-tool linkage to org-internal resources | Remove or genericize |

**Allowlist — known intentional convenience defaults** (do not flag, do not "fix"):

| File | Lines / fields | Why it stays |
|---|---|---|
| `container_env.sh` | `DATASET_DIR="${DATASET_DIR:-/mnt/vast/datasets}"`, `COREDUMP_EXTRA_DIRS=("/perf_apps/maxtext_coredump" …)` | These are deployment-specific *fallback* defaults that the deployer intentionally keeps for convenience (saves typing on every job). They're documented as overridable via env var (`DATASET_DIR=...`, `COREDUMP_EXTRA_DIRS=...`), and the leakage cost is low — vendor mount-points and a per-cluster path tag, not credentials or topology. The skill **must not** rewrite these to empty/generic defaults; doing so reduces ergonomics for the deployer with no security gain |
| `docs/job-submission.md` | the table row showing the same `DATASET_DIR` / `COREDUMP_EXTRA_DIRS` values | Mirrors `container_env.sh`; if the source is allowlisted, the doc must match. Keep |
| `skills/job-log-triage/SKILL.md` | the example `/perf_apps/maxtext_coredump` reference | Mirrors `container_env.sh`; documenting the allowlisted default |
| `outputs/**` | everything | Job logs, profiler dumps, and HLO traces *will* contain hostnames, IPs, and job IDs by construction. The repo's `.gitignore` already excludes most of this, and the sweep recipe below excludes the directory unconditionally. Never run the leak scan against `outputs/` content |
| `*.lock`, `.git/**` | everything | Lockfiles and git metadata legitimately reference branch / tag / SHA strings that match some patterns; do not scan |

When extending the allowlist, document the new entry's rationale in the same row format. Anything not in the allowlist is presumed-leak-by-default and must be either fixed or explicitly justified.

**Repo-wide leak sweep** (run periodically, not just on diffs — leaks accumulate):

```bash
# From the repo root.  Excludes outputs/ (job logs, expected to contain real IDs),
# container_env.sh (allowlisted convenience defaults), and lockfiles / git metadata.
# Adjust your-cluster-prefix to match your deploying cluster.
rg -nP --type-add 'doc:*.md' --type-add 'sh:*.sh' --type-add 'py:*.py' \
   -t doc -t sh -t py -t yml \
   -g '!outputs/**' -g '!*.lock' -g '!.git/**' \
   -g '!container_env.sh' \
   '\b(chi)\[?\d+|\b1[4-5]\d{3}\b|/mnt/[^/\s]+/[a-z0-9_-]+'
```

If a hit appears in `docs/job-submission.md` or `skills/job-log-triage/SKILL.md` referencing `/mnt/vast/datasets` or `/perf_apps/maxtext_coredump`, cross-check the line against the allowlist row above before "fixing" — these are mirrors of the allowlisted defaults.

**Auto-fix vs suggest:** purely-mechanical replacements (concrete hostname → `<nodelist>` placeholder, concrete job ID in a comment → symptom description, internal IP → `<host>`) can be applied without asking. Fixes that change *behavior* (e.g., switching a hard-coded partition name to a variable that may be unset in some entry point) require a confirmation round-trip — propose the patch, list the ripple-effect call sites, then commit on go-ahead.

**Output:** for each file, list `(line, leaked-pattern, proposed-fix)` triples. The same triples land in the audit's "Output format" section (item 7 below).

### Step 9 — output the matrix and audit lists

If Steps 1–5 ran, output the matrix. If they didn't (non-launcher commit), skip the matrix entirely and emit only the leak and doc audit lists.

**Matrix** — for each cell, mark one of:
- ✅ **verified** — live smoke passed for this exact combination
- ✅ **shared** — same downstream flow as a verified cell; covered by transitivity
- ⚠️ **suspect** — code changed in a way that should affect this cell but wasn't smoke-tested
- ❌ **broken** — confirmed regression

```
| Entry | Launch | RAY | Status | Notes |
|---|---|---|---|---|
| submit.sh | 1-node | 0 | ✅ shared | flow (1); same as run_local.sh row |
| submit.sh | 1-node | 1 | ✅ verified | flow (3); ran 70b:smoke-ray on <node> (5m14s, exit 0) |
…
```

Anything ⚠️ or ❌ blocks the commit / PR.

## Output format

Always emit:

1. **Change summary** — one sentence per modified file plus the function / lines touched.
2. **Affected flows** — which of the 4 post-`_train.sh` flows and which of the 2 entry-pairs each change touches, derived from Step 2's lookup table. ("N/A — non-launcher commit" if Steps 1–5 didn't run.)
3. **Edge-case audit** — for each cross-cutting edge case in Step 4, state whether the change avoids / triggers / is unaffected by it. ("N/A — non-launcher commit" if Steps 1–5 didn't run.)
4. **Smoke-test evidence** — two parts:
   - **§5.1 launcher chain** — per smoke, log path + exit code + (for RAY=1) the `ss -tlnp` snapshot showing localhost binding. ("N/A — non-launcher commit" if Step 5.1 didn't run.)
   - **§5.2 ancillary scripts** — per touched non-launcher script, the one-line status combining the smoke layers run (e.g., `utils/analyze_job.py: bash -n OK; --help OK; against outputs/14982 OK`). For side-effecting utilities deferred to user-run, include the proposed invocation. ("N/A — no non-launcher scripts touched" if §5.2 didn't run.)
5. **Code quality review** — Step 6's three lists: per-change auto-fixes applied, per-change proposed fixes pending confirmation, and **holistic findings** (smell-trajectory counts ≥ 3, top-edited hot-spot files, architectural-fit verdicts). For deferred findings, include a one-line TODO summary the user can paste elsewhere. State "code quality: no findings (per-change clean; holistic clean)" if both halves were clean — call out the holistic half explicitly so the next reader sees it ran.
6. **Doc / comment / format-consistency audit** — Step 7's three lists: in-file comments updated (including alignment fixes), cross-reference updates (including link-policy fixes), self-consistency updates to this skill (or "skill unchanged" if none needed). State explicitly when no updates are needed so the next reader knows the check ran.
7. **Leak audit** — per touched file, the result of Step 8's scan: either "clean" or `(line, leaked-pattern, proposed-fix)` triples plus a note on which were auto-fixed vs queued for confirmation. Note: the leak scan runs *after* Steps 6 and 7, so it sees any code or prose edits those steps made.
8. **Recommendation** — one of: **commit**, **commit + smoke-test cell `<X>` first**, **commit + user-run smoke for side-effecting `<script>`**, **commit + follow-up refactor PR for `<smell>`**, **commit + follow-up doc PR for `<file>`**, **pull back: cell `<X>` regresses**, **pull back: ancillary script `<file>` fails smoke**, **pull back: caller `<file>:<line>` consumes old signature**, **pull back: design issue in `<file>` blocks the change at hand**, **pull back: leak in `<file>:<line>`**, **pull back: stale doc / comment in `<file>:<line>` describes pre-change behavior**.

## Notes on this skill itself

- The skill bundles five responsibilities (launcher path coverage, ancillary scripts smoke, code quality + design + architectural fit, docs / comments / format, sensitive-info leak scan). Don't split it without thinking — they share the same output format and the same "ready / not ready" recommendation, and splitting forces every commit-time invocation to dispatch across multiple skills. The path-coverage piece is launcher-specific; the other four are universal; the umbrella is "is this change ready to commit?". That said: at 700+ lines the file is approaching the threshold where readability / agent-context cost might justify a split into a "launcher-path-coverage" specialist + a "general-pre-commit-audit" umbrella for the rest. Revisit if a 6th responsibility lands.
- The matrix size grows on every new entry point or launch mode added; update both the matrix and Step 2's file → flow table when that happens.
- `monkey_patch_maxtext.py` is the universal leaf — any change there invalidates all 16 cells, so always smoke at least one cell from each (RAY × launch) corner: 4 smokes minimum.
- `_container.sh` is the entry-pair-1 leaf and only invalidates 8 cells; `_k8s_job.sh` is entry-pair-2 and only invalidates 8.
- The skill assumes the **current 4×2×2 = 16 layout**; if a 5th entry point, 3rd launch mode, or 3rd stack lands, the matrix expands accordingly.
- For deep call-chain reasoning (e.g., "does this change affect compile-cache invalidation across 1-node-per-process vs 1-gpu-per-process?"), this skill does not replace per-file code reading. Use it to ensure **path coverage**; use code reading to ensure **per-path correctness**.
- Step 7.3's self-consistency check is what keeps this skill from rotting. When the launcher topology changes, that section's table tells the agent exactly which prose to update — including this "Notes" block. Re-read the table on every launcher change.
