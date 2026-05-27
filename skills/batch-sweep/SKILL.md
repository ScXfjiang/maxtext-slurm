---
name: batch-sweep
description: "Four sweep operations: (1) Model perf sweep — find optimal batch size / TGS for a model. Use for: sweep batch size, tune TGS, benchmark throughput, find optimal config. (2) Node perf sweep — compare per-node GPU performance to find outliers. Use for: check nodes, node performance, find slow node, compare nodes. (3) Node network health sweep — detect inter-node network issues via multi-node bisection. Use for: network health, IB issues, RCCL problems, node pair testing, isolate network problem. (4) Model sweep — run all model configs on one or two commits. Use for: regression test, validate commit, test all models, smoke test, CI, compare branches."
---

# Sweep Operations

Four independent sweep operations for performance tuning and health validation. Each can be run standalone.

| Sweep | What it tests | Job type | When to use |
|-------|--------------|----------|-------------|
| [Model perf sweep](#model-perf-sweep) | Batch size / config for max TGS | Multi-node, target model | Tuning a new model config |
| [Node perf sweep](#node-perf-sweep) | Per-node GPU compute health | 1N per node, llama2-70b | Checking a set of nodes for outliers |
| [Node network health sweep](#node-network-health-sweep) | Inter-node IB/RCCL health | 2N/4N subsets, llama2-70b | Isolating network problems between nodes |
| [Model sweep](#model-sweep) | All model configs runnable + TGS | Multi-node per model | Verify a commit, or compare two commits for regressions |

## Common prerequisites

- host-cmd available (`python3 /maxtext-slurm/.host-cmd/host_cmd.py --ping`)
- A fixed nodelist (always use `-w <nodelist>` to avoid cross-node variance)

## Monitoring policy (applies to all sweeps)

Every submitted job must be actively monitored. Never submit-and-forget.

### Progressive reporting

**Report each result as it lands.** Don't wait until all jobs finish. When a job completes (or fails), immediately print a one-line summary:

```
llama2-70b 1N (job 9801): 2,036 TGS/device, 34.9% MFU — PASS
grok-1 8N (job 9805): OOM — FAIL
```

This gives the user early signal. The full aggregate table is still built at the end.

### Fail-fast

When a job fails, report it immediately and classify:

| Failure type | Action |
|-------------|--------|
| OOM | Report, skip retries at same config, continue remaining jobs |
| GPU hardware error (single node) | Report, exclude node, retry once if possible |
| Config error (all nodes fail consistently) | **Stop the entire sweep.** Report and ask user how to proceed |
| Hang (10+ min no output) | Cancel, retry once per hang rules below. If retry also hangs, mark FAIL and continue |

For **model sweep** specifically: if 2+ models fail on the candidate but pass on the baseline (or a prior sweep), **stop early** — the commit is clearly broken. Report failures so far and ask the user whether to continue.

### Polling

Check the log every 60–120s. If no new output, exponential backoff up to 5 min.

### Hang detection

A job is hanging if no new log output for **10+ minutes** while `squeue` shows it `RUNNING`. Actions:

1. Verify the job is yours (`outputs/<jobid>-*` exists).
2. Cancel: `python3 /maxtext-slurm/.host-cmd/host_cmd.py "scancel <jobid>" --timeout 10`
3. Classify and decide:

| Hang phase | Likely cause | Retry? |
|------------|-------------|--------|
| Before `Memstats` (RCCL init) | Transient RCCL/IB | Yes, once |
| After `Memstats`, before `completed step: 0` | Silent OOM during XLA compile | No — treat as OOM |
| After `completed step: 0` | RCCL error mid-training | Yes, once |

### Failure detection

| Log pattern | Meaning | Action |
|-------------|---------|--------|
| `RESOURCE_EXHAUSTED: Out of memory` | OOM | No retry at this config. Record allocation size. |
| `HSA_STATUS_ERROR` / `rocdevice.cpp: Aborting` | GPU hardware error | Exclude that node, retry on remaining nodes. |
| `ECC` errors in preflight | Bad GPU memory | Exclude node for entire sweep. |
| One node `exit code 1`, others continue | Single-node crash | Exclude that node, retry. |
| All nodes fail consistently | Config issue | **Stop the sweep.** Fix config before continuing. |

### Bail-out rules (prevent infinite loops)

- **Max 2 retries per job.** If a job fails/hangs 3 times, mark it as failed and move on.
- **Max 2 node exclusions per sweep.** If 3+ nodes are excluded, **stop the sweep** — the cluster is unhealthy. Report to user.
- **If the same error repeats on retry with different nodes**, it's a config issue, not a node issue. **Stop and report.**
- **Never retry an OOM** at the same batch size. The result is deterministic.

---

## Model perf sweep

Find the optimal `per_device_batch_size` (or other parameters) for maximum steady-state TGS on a fixed set of nodes.

**Additional prerequisite:** A working `.gpu.yml` config that runs at some batch size.

### Step 1: Gather inputs

Ask the user for:
1. **Model name** (must have a `.gpu.yml` in `configs/`)
2. **Node count and nodelist** (e.g. `-N 8 -w <node-list>`)
3. **Parameter to sweep** (default: `per_device_batch_size`)
4. **Any env overrides** (e.g. `_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.93`)

### Step 2: Estimate memory budget

Read the model config and estimate:

```
pool = GPU_VRAM * XLA_PYTHON_CLIENT_MEM_FRACTION
param_memory ≈ (total_params * bytes_per_param) / total_sharding
available = pool - param_memory
```

For `bytes_per_param`, use `weight_dtype` from the config (bf16=2, fp32=4). For `total_sharding`, multiply all non-1 parallelism axes (both DCN and ICI; for MoE with `expert_shard_attention_option: "fsdp"`, expert axis also shards non-expert weights).

If a prior run exists, use `Memstats: After params initialized` from its log for exact param memory instead of estimating.

### Step 3: Choose sweep strategy

Use the **upper-bound probe** strategy — NOT binary search or linear scan:

1. **Estimate max batch** from memory budget:
   ```
   activation_per_batch ≈ seq_len × emb_dim × num_layers × 12 bytes
   max_batch ≈ available / activation_per_batch
   ```

2. **First probe at ~75% of estimated max** — if it works, probe higher; if OOM, we have an upper bound.

3. **Second probe at the upper bound** (e.g. 1.5× first) — establishes the ceiling.

4. **Narrow the range** — with a working lower bound and OOM upper bound, try the midpoint. One or two more runs to converge.

This finds the optimal in **3–4 runs** vs. 6–8 for binary search.

### Step 4: Submit jobs

Always use the **same nodelist** via `-w` to avoid hardware variance. Use CLI overrides to sweep without modifying the config file:

```bash
python3 /maxtext-slurm/.host-cmd/host_cmd.py \
  "cd <repo_path> && ./submit.sh <model> -N <nodes> -w <nodelist> -- per_device_batch_size=<N>" \
  --timeout 30
```

For env sweeps:
```bash
python3 /maxtext-slurm/.host-cmd/host_cmd.py \
  "cd <repo_path> && ./submit.sh <model> -N <nodes> -w <nodelist> -- _env_XLA_PYTHON_CLIENT_MEM_FRACTION=.93 per_device_batch_size=<N>" \
  --timeout 30
```

Submit one job at a time to the same nodes (they'll queue sequentially), or submit to different nodesets for parallel sweeps.

### Step 5: Monitor jobs

Apply the [Monitoring policy](#monitoring-policy-applies-to-all-sweeps) for hang detection, failure handling, and bail-out rules.

**Expected timeline for a 15-step run:**

| Phase | Expected duration | What to look for |
|-------|-------------------|------------------|
| Preflight + container pull | 1–3 min | `[INFO]` lines, `=== Pre-flight` |
| Model init + RCCL setup | 2–5 min | `Memstats: After params initialized` |
| XLA compilation (step 0) | 2–5 min | `completed step: 0` |
| Steady-state steps 1–14 | varies by model | `completed step: N` |

When excluding a node due to failures, the total node count may drop below what the model needs. If so, **stop the sweep** and inform the user.

### Step 6: Collect steady-state TGS

For each completed run, extract **steady-state TGS** from steps 5–14 (skip steps 0–4 for compilation warmup):

```bash
# Slurm logs (srun -l rank prefix):
grep "^0: completed step: \(5\|6\|7\|8\|9\|10\|11\|12\|13\|14\)," outputs/<jobid>-*.log
# K8s logs (rank 0 only, no prefix):
grep "completed step: \(5\|6\|7\|8\|9\|10\|11\|12\|13\|14\)," outputs/<jobid>-*.log
```

Use node 0's values (all nodes report identical metrics). Compute the average `Tokens/s/device` across steps 5–14.

### Step 7: Interpret results and decide next probe

| Result | Interpretation | Next action |
|--------|---------------|-------------|
| OOM | Batch too large | Probe lower (midpoint between last-good and this) |
| Runs, TGS higher than previous | Better batch size | Probe higher if room remains |
| Runs, TGS lower or flat | Past the sweet spot | The previous batch size was optimal |
| Hang (retried, still hangs) | Treat as OOM | Probe lower |

**Key insight**: TGS does not always increase with batch size. At some point, communication overhead or memory pressure causes regression. The optimal is the batch with the **highest TGS**, not the largest batch that fits.

### Step 8: Record results

After finding the optimal, build a summary table:

```markdown
| Batch | Job | Avg TGS/device | Avg MFU | Status |
|-------|-----|----------------|---------|--------|
| 8     | 1234| 1,190          | 11.92%  | works  |
| 12    | 1235| 1,339          | 13.42%  | works  |
| 16    | 1236| 1,400          | 14.02%  | works ← optimal |
| 20    | 1237| 1,371          | 13.73%  | works (regression) |
| 24    | 1238| OOM            | —       | OOM    |
```

Update the config file with the optimal value and amend the commit if on a feature branch.

### Sweeping other parameters

The same strategy works for any parameter. Common sweeps:

| Parameter | Override syntax | Notes |
|-----------|----------------|-------|
| `per_device_batch_size` | `per_device_batch_size=N` | Most common sweep target |
| `max_target_length` | `max_target_length=N` | Memory scales linearly with seq len |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `_env_XLA_PYTHON_CLIENT_MEM_FRACTION=.93` | Higher gives more memory but may starve RCCL; XLA may inflate allocations |
| `remat_policy` | `remat_policy=full` vs `minimal_flash` | `full` uses less memory, more compute |
| `quantization` | `quantization=fp8` | Halves weight memory |

For multi-parameter sweeps (e.g. batch × mem_fraction), fix one and sweep the other. Document which combination won.

### Pitfalls

- **Different nodes → different results.** Always use `-w <nodelist>`. Node health, IB fabric quality, and thermal throttling vary.
- **XLA memory inflation.** Giving XLA more memory (`MEM_FRACTION`) doesn't always help — it may inflate allocations to fill the pool, eating the extra headroom.
- **Step 0 is not representative.** It includes XLA compilation overhead. Never use step 0 metrics for comparison.
- **MFU can regress at large batch.** Communication overhead grows with batch size for MoE models. The optimal batch is where TGS peaks, not where memory maxes out.
- **Hangs and failures are handled by the shared [Monitoring policy](#monitoring-policy-applies-to-all-sweeps).** Max 2 retries per job, max 2 node exclusions per sweep, never retry OOM at the same batch size.

---

## Node perf sweep

Compare per-node GPU compute performance and find outliers. Use when the user gives a list of nodes and wants to check for performance discrepancy.

### Workflow

1. **Submit 1N baseline jobs in parallel** to each node using `llama2-70b` (dense, fast to compile, stable TGS):

   ```bash
   for node in node001 node002 node003 node004; do
     python3 /maxtext-slurm/.host-cmd/host_cmd.py \
       "cd <repo_path> && ./submit.sh 70b:${node}: -N 1 -w ${node}" \
       --timeout 30
   done
   ```

   The `:${node}:` alias isolates each job's output directory. All jobs run in parallel since they target different nodes.

2. **Monitor all jobs.** Apply the [Monitoring policy](#monitoring-policy-applies-to-all-sweeps). Since these are 1N jobs, hangs/failures are node-specific — mark that node as unhealthy.

3. **Collect steady-state TGS** (steps 5–14) from each job:

   ```bash
   grep "^0: completed step: 10," outputs/*-JAX-llama2-70b.log
   ```

4. **Build a per-node comparison table:**

   ```markdown
   | Node | Job | TGS/device | vs. median | Status |
   |------|-----|------------|------------|--------|
   | node001 | 9701 | 1,842 | +0.1% | healthy |
   | node002 | 9702 | 1,838 | -0.1% | healthy |
   | node003 | 9703 | 1,840 | 0.0% | healthy |
   | node004 | 9704 | 1,695 | -7.9% | DEGRADED |
   ```

5. **Evaluate:**

   | TGS vs. median | Interpretation |
   |-----------------|---------------|
   | Within ±2% | Healthy |
   | 2–5% below | Marginal — monitor but usable |
   | 5%+ below | Degraded — exclude from multi-node jobs |
   | Failed/hung | Unhealthy — exclude and report to user |

6. **Report** the table and recommendation to the user. If this was done before a batch sweep, recommend the healthy subset as the nodelist.

---

## Node network health sweep

Detect inter-node network issues (bad IB links, RCCL problems between specific node pairs) by running multi-node jobs across subsets and bisecting to isolate the problem.

The node perf sweep (1N) only tests GPU compute. This sweep tests the **network fabric between nodes** using `llama2-70b` with 2N or 4N jobs.

Since this only tests network health (not steady-state TGS), use `-- steps=1` to minimize runtime. A single training step is enough to exercise RCCL collectives and expose network issues.

1. **Start with one job across all nodes** to get a baseline:

   ```bash
   python3 /maxtext-slurm/.host-cmd/host_cmd.py \
     "cd <repo_path> && ./submit.sh 70b:all-nodes: -N 8 -w <all-nodes> -- steps=1" \
     --timeout 30
   ```

2. **If baseline is healthy**, no network issues — done.

3. **If baseline is degraded** (TGS below expected), bisect with half-sized groups:

   ```bash
   # Group A: first half
   python3 /maxtext-slurm/.host-cmd/host_cmd.py \
     "cd <repo_path> && ./submit.sh 70b:group-a: -N 4 -w <group-a-nodes> -- steps=1" \
     --timeout 30
   # Group B: second half
   python3 /maxtext-slurm/.host-cmd/host_cmd.py \
     "cd <repo_path> && ./submit.sh 70b:group-b: -N 4 -w <group-b-nodes> -- steps=1" \
     --timeout 30
   ```

4. **Compare group TGS:**

   | Scenario | Interpretation | Next action |
   |----------|---------------|-------------|
   | Both groups healthy | Issue is cross-group (IB link between groups) | Try mixed groups to find the bad pair |
   | One group degraded | Problem node is in that group | Bisect again (2N) within the bad group |
   | Both degraded | Multiple issues or shared infrastructure | Report to user, likely needs HW team |

5. **Narrow with 2N jobs** to pinpoint the exact node:

   ```bash
   # If group A was bad (node[001-004]):
   python3 /maxtext-slurm/.host-cmd/host_cmd.py \
     "cd <repo_path> && ./submit.sh 70b:pair-a1: -N 2 -w node[001-002] -- steps=1" \
     --timeout 30
   python3 /maxtext-slurm/.host-cmd/host_cmd.py \
     "cd <repo_path> && ./submit.sh 70b:pair-a2: -N 2 -w node[003-004] -- steps=1" \
     --timeout 30
   ```

6. **Build isolation table:**

   ```markdown
   | Group | Nodes | TGS/device | vs. expected | Status |
   |-------|-------|------------|-------------|--------|
   | all-8 | node[001-008] | 1,320 | -5.2% | DEGRADED |
   | group-a | node[001-004] | 1,385 | -0.4% | healthy |
   | group-b | node[005-008] | 1,310 | -5.8% | DEGRADED |
   | pair-b1 | node[005-006] | 1,388 | +0.0% | healthy |
   | pair-b2 | node[007-008] | 1,295 | -6.9% | DEGRADED |
   ```

   In this example, the issue is between node007 and node008 (or one of them has a bad NIC). Cross-reference with the 1N per-node sweep — if both were healthy individually, the problem is the network link between them.

7. **Report** findings and recommend excluding the problematic node(s) or escalating to the infrastructure team for IB link repair.

---

## Model sweep

Run all model configs on one or more commits/branches and report TGS. Use to verify a commit works, smoke-test a branch, or compare two code states for regressions.

**Modes:**

| User says | Commits to run | Output |
|-----------|---------------|--------|
| "sweep all models" / "verify this branch" | Current HEAD (1 commit) | Single-commit results table |
| "sweep all models on main" | The specified commit (1 commit) | Single-commit results table |
| "compare branch X vs main" / specifies 2 commits | Both commits (2 commits) | Side-by-side comparison with delta |

**Cross-turn comparison:** If the user runs multiple model sweeps in the same chat session (e.g. "sweep branch A", then "sweep branch B", then "sweep branch C"), produce an aggregate report after each new sweep. The aggregate table shows all sweeps side-by-side with delta TGS between each pair of adjacent turns. This avoids re-running jobs just to get comparisons.

### Step 1: Determine commits

- **No commit specified** → use the current HEAD.
- **One commit specified** → check out that commit.
- **Two commits specified** → run both in sequence.
- **Prior sweep(s) exist in this chat** → after completing the new sweep, produce the aggregate report covering all sweeps in the session with delta TGS between each adjacent pair.

### Step 2: Enumerate model configs and node counts

Skip proxy configs (`ds-proxy*`) and `default`. Each model runs at one or more fixed node counts:

| Model | Node counts | Notes |
|-------|------------|-------|
| llama2-70b | 1N, 2N | Fast compile, good smoke test for single- and multi-node |
| mixtral-8x22b | 4N, 8N | MoE; 4N is the native FSDP size, 8N adds data parallelism |
| grok-1 | 8N | Large MoE, needs 8N |
| grok-2 | 8N | Large MoE, needs 8N; scan_layers=false required |
| llama3.1-405b | 8N | Dense 405B, needs 8N for memory |
| deepseek3-671b | 8N | MoE 671B, native dcn_fsdp=8 |
| kimi-k2-1t | 8N | MoE 1T, native dcn_fsdp=8 |
| qwen3-480b-a35b | 8N | MoE 480B, native dcn_fsdp=8 |

This produces **10 jobs per commit**.

Pick a fixed 8-node nodelist. Assign **non-overlapping** subsets for sub-8N jobs so they can run in parallel:

```
Nodes:    n1    n2    n3    n4    n5    n6    n7    n8
1N job:   [n1]
2N job:               [n3,  n4]
4N job:                           [n5,  n6,  n7,  n8]
8N jobs:  [n1,  n2,   n3,  n4,   n5,  n6,  n7,  n8]
```

The 1N, 2N, and 4N jobs use disjoint nodes (1 + 2 + 4 = 7 ≤ 8), so they run **in parallel**. The 8N jobs need all 8 nodes and queue **sequentially** behind each other and behind any running sub-8N jobs.

**Critical rule:** across multiple sweeps (different commits or chat turns), each model × node-count must always use the **same nodelist**. Hardware variance between nodes invalidates TGS comparisons.

### Step 3: Submit jobs

Check out the target commit (stash uncommitted changes if needed). Submit all models with `-- steps=15 dataset_type=synthetic`. Tag jobs with the commit identifier (e.g. `:main:`, `:candidate:`, or a short SHA):

```bash
NODE1="<n1>"                             # 1N: disjoint from 2N and 4N
NODELIST2="<n3,n4>"                      # 2N: disjoint from 1N and 4N
NODELIST4="<n5,n6,n7,n8>"               # 4N: disjoint from 1N and 2N
NODELIST8="<n1,n2,n3,n4,n5,n6,n7,n8>"   # 8N: all nodes
TAG="<commit_tag>"                       # e.g. "main", "candidate", short SHA

# llama2-70b: 1N and 2N (run in parallel — disjoint nodes)
python3 /maxtext-slurm/.host-cmd/host_cmd.py \
  "cd <repo_path> && ./submit.sh llama2-70b:${TAG}: -N 1 -w $NODE1 -- steps=15 dataset_type=synthetic" \
  --timeout 30
python3 /maxtext-slurm/.host-cmd/host_cmd.py \
  "cd <repo_path> && ./submit.sh llama2-70b:${TAG}:2n -N 2 -w $NODELIST2 -- steps=15 dataset_type=synthetic" \
  --timeout 30

# mixtral-8x22b: 4N (runs in parallel with llama2-70b — disjoint nodes)
python3 /maxtext-slurm/.host-cmd/host_cmd.py \
  "cd <repo_path> && ./submit.sh mixtral-8x22b:${TAG}:4n -N 4 -w $NODELIST4 -- steps=15 dataset_type=synthetic" \
  --timeout 30

# 8N jobs (queue sequentially — need all 8 nodes)
for model in mixtral-8x22b grok-1 llama3.1-405b deepseek3-671b kimi-k2-1t; do
  python3 /maxtext-slurm/.host-cmd/host_cmd.py \
    "cd <repo_path> && ./submit.sh ${model}:${TAG}: -N 8 -w $NODELIST8 -- steps=15 dataset_type=synthetic" \
    --timeout 30
done
```

The 1N + 2N + 4N jobs start immediately in parallel. Once they finish and free all nodes, the 8N jobs run one at a time.

If running multiple commits, repeat step 3 for each (check it out first, use a different `TAG`). All 8N jobs across all commits queue sequentially on the same nodelist.

### Step 4: Monitor jobs and report progressively

Apply the [Monitoring policy](#monitoring-policy-applies-to-all-sweeps). As each job completes (or fails), immediately:

1. Extract steady-state TGS (steps 5–14, skip 0–4 for compilation warmup):
   ```bash
   grep "^0: completed step: \(5\|6\|7\|8\|9\|10\|11\|12\|13\|14\)," outputs/<jobid>-*.log
   ```
2. Compute the average `Tokens/s/device` and `MFU` across steps 5–14.
3. **Report the result inline** — don't wait for all jobs to finish.
4. If the job failed, apply fail-fast rules from the monitoring policy.

### Step 5: Build final report

**Single-commit report:**

```markdown
| Model | Nodes | Job | Avg TGS/device | Avg MFU | Status |
|-------|-------|-----|----------------|---------|--------|
| llama2-70b | 1N | 9801 | 1,840 | 34.9% | PASS |
| llama2-70b | 2N | 9802 | 2,036 | 34.9% | PASS |
| mixtral-8x22b | 4N | 9803 | 4,766 | 23.9% | PASS |
| mixtral-8x22b | 8N | 9804 | 4,780 | 23.9% | PASS |
| grok-1 | 8N | 9805 | 2,536 | 26.5% | PASS |
| llama3.1-405b | 8N | 9806 | 674 | 34.0% | PASS |
| deepseek3-671b | 8N | 9807 | 1,416 | 14.2% | PASS |
| kimi-k2-1t | 8N | 9808 | 1,149 | 9.4% | PASS |
```

**Multi-sweep aggregate report (cross-turn or two-commit):**

When there are 2+ sweeps (from explicit two-commit mode, or accumulated across chat turns), produce one table per model with all sweeps as columns and delta between each adjacent pair:

```markdown
### llama2-70b (1N)
| Sweep | Commit | Job | TGS | Δ vs prev |
|-------|--------|-----|-----|-----------|
| 1 | main | 9801 | 1,840 | — |
| 2 | feat-a | 9811 | 1,842 | +0.1% |
| 3 | feat-b | 9821 | 1,810 | -1.7% |

### llama2-70b (2N)
| Sweep | Commit | Job | TGS | Δ vs prev |
|-------|--------|-----|-----|-----------|
| 1 | main | 9802 | 2,036 | — |
| 2 | feat-a | 9812 | 2,038 | +0.1% |
| 3 | feat-b | 9822 | 2,035 | -0.1% |

### ... (one section per model × node-count)
```

For a simple two-sweep case this is equivalent to the old baseline-vs-candidate table, but scales naturally to 3+ sweeps without restructuring.

### Step 6: Evaluate

**Single-commit evaluation:**

| Outcome | Action |
|---------|--------|
| All models complete | Report success with TGS table |
| Some models OOM or fail | Report failures with details |
| All models fail | Stop — fundamental issue with the commit |

**Multi-sweep evaluation (adjacent-pair deltas):**

| Outcome | Interpretation | Action |
|---------|---------------|--------|
| All deltas within ±2% | No regression between any pair | Report success |
| Any delta >5% regression | Performance regression introduced by that sweep | Flag the sweep + model |
| Model FAIL in sweep N but PASS in sweep N-1 | Breakage introduced by sweep N | Flag broken models with failure details |
| Model FAIL in both sweep N and N-1 | Pre-existing issue | Note but don't blame sweep N |
| All models FAIL in a sweep | Fundamental issue with that commit | Stop — the commit is broken |

**Report** the full aggregate table to the user. Highlight regressions and failures clearly, distinguishing new breakage from pre-existing issues.
