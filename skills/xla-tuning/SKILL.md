---
name: xla-tuning
description: "Find the XLA flag / NCCL env-var combination that maximizes steady-state TGS for one (model × parallelism) cell. Produces an evidence-backed leaderboard, mechanistic explanation of the winning flag, and a deployment recipe. Use when the user asks to tune XLA flags, tune NCCL, find best collective-permute / all-gather threshold, optimize FSDP/PP/TP, close a parallelism-vs-parallelism throughput gap, or sweep cross-iteration prefetch / overlap-limit / async-stream-priority knobs for a specific model."
---

# XLA / NCCL Flag Tuning for a (Model × Parallelism) Cell

Given a single `(MODEL_TAG, PARALLELISM)` cell on a fixed nodelist, find the XLA flag / NCCL env-var combination that maximizes steady-state TGS. Typically 1-3 flags capture most of the win; the rest of the sweep produces evidence (HLO + xplane) explaining *why* the winning flag wins — which is the durable knowledge that survives image/JAX upgrades.

This skill produces or extends one deliverable per model: `<MODEL_TAG>-tuning.md` in the repo root (and `<MODEL_TAG>-tuning.zh.md` if a Chinese sibling exists or is requested). It references no other `.md` files in the repo root.

## Common prerequisites

- host-cmd reachable (`python3 /maxtext-slurm/.host-cmd/host_cmd.py --ping`)
- `configs/<MODEL_TAG>.gpu.yml` exists and runs at the requested parallelism without OOM
- A working `pdbs` for the cell (run [batch-sweep](../batch-sweep/SKILL.md) first if not yet known — XLA tuning is on top of an already-good `per_device_batch_size`, not a substitute for it)
- Telegram set up (see [telegram](../telegram/SKILL.md)) for progress updates

## Inputs (ask the user; never guess)

The user must provide:

1. **`MODEL_TAG`** — must match `configs/<MODEL_TAG>.gpu.yml`. Examples: `deepseek3-671b`, `qwen3-32b`, `llama3.1-405b`, `mixtral-8x22b`.

2. **`CLUSTER_SLOT`** — one of:
   - explicit `nodelist=<list>` *(recommended; pinned hardware drops TGS noise to ~0.3-1 % which is what you need to detect 1-3 % flag deltas)*
   - `partition=<name>+nodes=<N>` *(let slurm pick from the partition pool)*
   - `nodes=<N>` only *(agent runs `sinfo` and picks N idle nodes from the default partition)*

3. **`PARALLELISM`** — one of:
   - `FSDP=N` (DCN FSDP only)
   - `PP=N` (DCN pipeline only)
   - `TP=N` (ICI tensor parallel)
   - hybrid: `PP=N,FSDP=M` or `FSDP=N,TP=M`, etc. (axes multiply to total ranks)

Optional:

- **`pdbs`** — `per_device_batch_size`. If omitted, read from `configs/<MODEL_TAG>.gpu.yml`. If batch-sweep has been run, use that result.
- **`CONFIG_TAG`** — for models with multiple comparable sweep configs (e.g. an MoE model with `dense_matmul` vs `sparse_matmul` branches). If omitted, use the model's primary config from `gpu.yml`.

If any of (1)-(3) are missing, ask the user before starting. Do not assume defaults for these three.

## Workflow

```
- [ ] Step 1 — Pre-flight (host-cmd, image, nodelist, env file, train_env.sh clean, TG)
- [ ] Step 2 — Inventory current state (read gpu.yml + env.sh + container_env.sh + any <MODEL_TAG>-tuning.md if it exists)
- [ ] Step 3 — Wave 1: baseline pair (current env vs raw image; one pair per CONFIG_TAG)
- [ ] Step 4 — Wave 1.5: capture HLO + xplane evidence on baseline-deployed
- [ ] Step 5 — Wave 2-4: flag testing per parallelism catalog (5-10 profiles per wave)
- [ ] Step 6 — Wave 5 (PP only): MaxText pipeline params (microbatches, V chunks)
- [ ] Step 7 — Wave 6: cross-config-tag confirmation of winning recipe (if model has >1 CONFIG_TAG)
- [ ] Step 8 — Write <MODEL_TAG>-tuning.md (and .zh.md mirror if needed)
```

### Step 1 — Pre-flight

Run all of these. Stop and TG-ask if any fail.

```bash
# 1. Host-cmd reachable
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 10 "echo HOST_OK; whoami; hostname"

# 2. Resolve CLUSTER_SLOT to a fixed nodelist (write down NODELIST + NUM_NODES + PARTITION)
# (a) nodelist given: NODELIST=<as given>
# (b) partition+N given: pick N idle nodes from `sinfo -p <partition>`
# (c) just N: pick N idle nodes from `sinfo` on the default partition
# Once chosen, NODELIST is FROZEN for the entire sweep. Do not silently swap nodes
# mid-sweep — that invalidates every prior TGS measurement. Recovery options for an
# unhealthy node are in § Autonomous failure recovery below.
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 30 \
  "sinfo -p <PARTITION> -n <NODES> -o '%n %T %f %E' 2>&1"
# Expect every node in idle/mix/alloc. If any are drain/down/fail/maint, follow the
# node-recovery procedure in § Autonomous failure recovery before submitting anything.

# 3. Image tarball accessible (read DOCKER_IMAGE from container_env.sh)
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 10 "ls -la $(grep '^DOCKER_IMAGE' /maxtext-slurm/container_env.sh | cut -d'=' -f2 | tr -d '\"')"

# 4. Other users on the same partition (so submissions don't surprise-conflict)
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 30 "squeue -p <PARTITION> 2>&1 | head"

# 5. Per-model env override (may already contain flags; we'll need to know what's there)
cat /maxtext-slurm/configs/<MODEL_TAG>.env.sh 2>/dev/null || echo "(no per-model env file)"

# 6. train_env.sh state — should be clean (no leftover TUNE_PROFILE block)
grep -c TUNE_PROFILE /maxtext-slurm/train_env.sh   # expect 0; if not, read and decide

# 7. TG works (telegram skill).  Use the repo-relative path; the host-cmd
# runs commands with the maxtext-slurm checkout as cwd.
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 30 \
  "utils/telegram_bot.sh send '<MODEL_TAG> <PARALLELISM> XLA-tuning agent online; pre-flight pass'"
```

### Step 2 — Inventory current state

Read these files and write a one-paragraph "current state" summary that becomes the doc's "Inventory" section:

| File | What to extract |
|---|---|
| `configs/<MODEL_TAG>.gpu.yml` | Default parallelism, `per_device_batch_size`, `max_target_length`, `remat_policy`, MoE flags (`sparse_matmul`, `use_turbo_grouped_gemm`, `use_deepep_dispatch`, `capacity_factor`), pipeline params (`pipeline_parallel_layers`, `num_pipeline_microbatches`) |
| `configs/<MODEL_TAG>.env.sh` (if exists) | Every `XLA_FLAGS` / `NCCL_*` / env override currently deployed for this model — these are *inherited* by all your jobs unless you override with `TUNE_FLAGS` |
| `container_env.sh` | `DOCKER_IMAGE`, `MAXTEXT_PATCH_BRANCH`, `MAXTEXT_REPO_DIR` |
| `<MODEL_TAG>-tuning.md` (if exists) | Prior best recipes per (parallelism × config) cell, prior baseline TGS, prior structural-cost decomposition. If absent, declare "fresh start" — Wave 1.5 will derive it. |

**Do not reference any other `.md` files in the repo root** — they are project-specific and may not exist in other deployments.

### Step 3 — Wave 1: baseline pair

The deployed env file may include flags tuned for a *different* parallelism. Measure both states for each `CONFIG_TAG`:

| Profile | TUNE_PROFILE | What it measures |
|---|---|---|
| `baseline-deployed` | empty (inherits env file as-is) | Production state — *all later Δ% are vs this* |
| `baseline-no-env` | `restore_default` (overrides env-file flags back to image defaults) | What the image alone gives — Δ vs `baseline-deployed` tells you whether the deployed flags help, hurt, or are neutral on this parallelism |

`restore_default` works via XLA's last-wins flag resolution: append `--xla_<flag>=<image-default-value>` to `TUNE_FLAGS` for every flag the env file sets. Sign flips between parallelisms are common — never assume an env-file recipe transfers from the parallelism it was tuned on.

### Step 4 — Wave 1.5: capture evidence (mandatory)

Submit one `baseline-deployed` job with profiler + HLO dump. This is the single most valuable run of the sweep — it tells you what mechanism dominates this cell's step time, which prioritizes the candidate flag list.

```bash
RAY=1 ./submit.sh <MODEL_TAG>:evidence \
  --partition=<PARTITION> --nodes=<NUM_NODES> --nodelist=<NODELIST> --time=50:00 -- \
  per_device_batch_size=<pdbs> \
  <parallelism overrides; see Submission template> \
  <config-tag overrides; see Submission template> \
  profiler=xplane skip_first_n_steps_for_profiler=10 profiler_steps=3 \
  _env_ENABLE_XLA_DUMP=1 \
  jax_distributed_heartbeat_timeout_seconds=99999
```

When the job finishes, read:

| Artifact | What to look for |
|---|---|
| `outputs/<jobid>-…/<run_name>/tensorboard/.../plugins/profile/.../*.xplane.pb` | Per-stream timeline. Run [profile-drill](../profile-drill/SKILL.md) for kernel breakdown. **What fraction of step time is in collectives vs compute vs idle bubble?** |
| `outputs/<jobid>-…/xla_dump/module_*.jit_train_step.*_after_optimizations.txt` | Compiled HLO. Search `collective-permute-start`, `all-gather-start`, `reduce-scatter-start` — what's their `frontend_attributes`? Is `is_pipelined` true? Is `is_sync` true? Is the `dependency_set` empty (sync) or non-trivial (async)? |
| Same dump file | Search for `while` loops — are paired `*-start` / `*-done` ops inside the same iteration (sync), or paired across iterations (pipelined)? |
| Same dump file | Search the inherited XLA_FLAGS line at the top of the log to enumerate the image's compiled-in defaults; many old flags are obsolete and trigger `Unknown flag` aborts |

Use what you find to **prioritize** the flag catalog below — don't blindly transcribe it. If the dominant cost is something the catalog doesn't address (e.g. grouped-GEMM bubble inside a custom kernel, kernel-launch latency on small messages, NCCL clique creation per microbatch), TG the user with the finding and propose new candidate flags before the canned waves.

### Step 5 — Waves 2-4: flag testing

Use the [flag catalog](#per-parallelism-flag-catalog) below as a starting candidate list, gated by your `PARALLELISM`. Submit 5-10 profiles per wave (slurm queue serializes them), then pause to review and decide direction. The full sweep is typically 25-50 profiles.

**Sign flips between parallelisms are systematic** — every flag that helps one parallelism may hurt another. Re-test fresh, even for flags that prior tuning on a different parallelism marked "negative". The only flags to skip are *mechanism*-irrelevant ones (e.g. `pipelined_p2p` is meaningless without `collective-permute`).

#### Submission template

```bash
RAY=1 ./submit.sh <MODEL_TAG>:<short_alias> \
  --partition=<PARTITION> --nodes=<NUM_NODES> --nodelist=<NODELIST> --time=45:00 -- \
  steps=15 dataset_type=synthetic \
  per_device_batch_size=<pdbs> \
  <parallelism overrides> \
  <config-tag overrides> \
  jax_distributed_heartbeat_timeout_seconds=99999 \
  _env_TUNE_PROFILE=<profile_name>
```

**Always pass `steps=15 dataset_type=synthetic` on every tuning submission** (including the Wave 1.5 evidence run and Wave 5/6 follow-ups), regardless of what `configs/<MODEL_TAG>.gpu.yml` has set. Two reasons:

1. **Data-loader noise washes out the signal.** TGS deltas of 1-3 % are what you're hunting; grain/c4/HF-tokenizer variance routinely adds 1-2 % per-step jitter that swamps the per-flag effect. Synthetic data is bit-deterministic and produces the lowest-variance TGS measurement available. (See [tuning-runs are synthetic-only](#tuning-runs-are-synthetic-only) below for the operational rule.)
2. **Tuning is short.** 15 steps × ~30 s = ~7-8 min of training per job after compile. Long-step real-data probes belong in the on-demand loss test, not the XLA-flag tuning sweep — totally different question.

If `configs/<MODEL_TAG>.gpu.yml` has `dataset_type: grain` (e.g. left over from a loss test), the CLI passthrough `dataset_type=synthetic steps=15` overrides it; you do not need to edit the yml. Grain-specific keys (`grain_*`, `tokenizer_*`, `hf_access_token`) are silently ignored when `dataset_type=synthetic`.

If `configs/<MODEL_TAG>.gpu.yml` has `steps: <large_number>` (e.g. `steps: 2000` for a loss test), the CLI `steps=15` likewise overrides it — no yml edit needed.

`<parallelism overrides>` derive from `PARALLELISM`:

| `PARALLELISM` | Passthrough overrides (override gpu.yml's defaults) |
|---|---|
| `FSDP=N` | none if `N` matches `dcn_fsdp_parallelism` in gpu.yml; else `dcn_fsdp_parallelism=<N> dcn_pipeline_parallelism=1` |
| `PP=N` | `dcn_pipeline_parallelism=<N> dcn_fsdp_parallelism=1` plus pipeline params from gpu.yml (`pipeline_parallel_layers`, `num_layers_per_pipeline_stage`, `num_pipeline_microbatches` may need explicit values if the gpu.yml didn't pre-fill them) |
| `TP=N` | `ici_tensor_parallelism=<N>` (and likely reduce another ici axis to compensate) |
| Hybrid `PP=N,FSDP=M` | both axes; ensure `N × M = num_dcn_replicas` |

`<config-tag overrides>` come from the model's gpu.yml or [model-config-guide](../model-config-guide/SKILL.md). Examples:

- MoE `sparse_matmul + DeepEP`: `sparse_matmul=true use_turbo_grouped_gemm=true use_deepep_dispatch=true`
- MoE `dense_matmul + capacity-factor dropping`: drop the three above (they default to false in gpu.yml)
- Dense (non-MoE): no MoE overrides

Keep the **same `--nodelist=…` for every submission in the sweep**. TGS noise from cross-nodelist hardware variance easily exceeds the 1-3 % flag deltas you're hunting. If a node goes unhealthy mid-sweep, follow Step 1 recovery; if it can't be revived, you must drop to N-1 nodes and re-baseline everything before continuing.

#### `train_env.sh` editing pattern

Add (at the end of `train_env.sh`):

```bash
# ---- TUNE_PROFILE ----
TUNE_PROFILE="${TUNE_PROFILE:-${EXTRACTED_ENV_MAP[TUNE_PROFILE]:-baseline}}"
TUNE_FLAGS=""
case "${TUNE_PROFILE}" in
    baseline) ;;  # no-op; inherits configs/<MODEL_TAG>.env.sh as-is
    restore_default)
        # Override every flag the env file sets back to its image-default value.
        # Use last-wins: append `--xla_<flag>=<image-default>` here for each one.
        TUNE_FLAGS+=" --xla_<flag>=<image-default-value>"
        ;;
    <profile_name>)
        TUNE_FLAGS+=" --xla_<flag>=<value>"
        ;;
    *)
        echo "[train_env.sh] WARNING: unknown TUNE_PROFILE='$TUNE_PROFILE'" >&2
        ;;
esac
if [[ -n "$TUNE_FLAGS" ]]; then
    XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }${TUNE_FLAGS# }"
    export XLA_FLAGS
fi
```

For NCCL knobs (and any other env vars), add a SECOND case block at the **very end** of `train_env.sh`, after the existing NCCL exports — otherwise they get overwritten:

```bash
# ---- TUNE_PROFILE: post-NCCL overrides ----
case "${TUNE_PROFILE:-}" in
    nccl_chan8) export NCCL_NCHANNELS_PER_NET_PEER=8 ;;
    mem95)      export XLA_PYTHON_CLIENT_MEM_FRACTION=.95 ;;
esac
```

Each `submit.sh` invocation **freezes its own copy** of `train_env.sh` into `outputs/.artifacts/artifact_<id>/`, so you can edit between submissions to add new profiles without affecting pending jobs (`/maxtext-slurm/submit.sh` lines 53-69 for the mechanism).

#### Steady-state TGS computation

Average `Tokens/s/device` over **steps 9-14** (skip warmup steps 0-4, and skip steps 5-8 which are inside the profiler-capture window if profiler is on; step 8 is the profiler-dump step, a massive outlier):

```python
import re, glob

def steady(jobid_glob, lo=9, hi=14):
    paths = sorted(glob.glob(f'/maxtext-slurm/outputs/{jobid_glob}-*.log'))
    if not paths: return None, None, None
    with open(paths[0]) as f: lines = f.readlines()
    steps = {}
    for l in lines:
        m = re.search(r'completed step: (\d+), seconds: ([\d.]+).*Tokens/s/device: ([\d.]+).*loss: ([\d.]+)', l)
        if m and (s := int(m.group(1))) not in steps:
            steps[s] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    rng = sorted(s for s in steps if lo <= s <= hi)
    if not rng: return None, None, None
    return (sum(steps[s][0] for s in rng) / len(rng),         # mean step seconds
            sum(steps[s][1] for s in rng) / len(rng),         # mean Tokens/s/device
            steps[max(steps)][2])                              # last loss
```

#### Monitoring policy

Apply the [Monitoring policy from batch-sweep](../batch-sweep/SKILL.md#monitoring-policy-applies-to-all-sweeps) for hang detection and progressive reporting. The tuning sweep is a 36-50-job batch — every job must be actively monitored (poll every 3-5 min via `squeue` + log tail). Failure handling has its own dedicated section: **see [§ Autonomous failure recovery](#autonomous-failure-recovery) for the per-failure-class playbook (RCCL flake / JAX coord timeout / Unknown flag / OOM / cleanup exit=143 / NIC fault / drained node / etc.) and TG-stop triggers.**

Two cases that are *not* in the recovery playbook because they're not failures, just timing:

| Log signature | Diagnosis | Action |
|---|---|---|
| HLO compilation taking > 15 min on first job after image load | normal for large MoE on cold cache | wait |
| HLO compilation > 15 min on later job with same flag set | suspicious — may have hit a compiler bug | cancel, try a slight flag perturbation; if reproducible, mark and skip |

### Step 6 — Wave 5 (PP only): MaxText pipeline params

Skip if `PARALLELISM` doesn't include PP. Otherwise, after Waves 2-4 plateau, try the pipeline-shape knobs (these change the bubble fraction and per-microbatch overhead):

| Override | Effect |
|---|---|
| `num_pipeline_microbatches=2N` (e.g. 8 → 16) | halves bubble; doubles per-microbatch comm/MoE round-trip count |
| `num_pipeline_microbatches=N/2` | halves per-microbatch overhead; doubles bubble |
| `num_layers_per_pipeline_stage=K` (vary K) | changes V chunks per stage; changes bubble divisor |

Bubble fraction = `(num_stages − 1) / (num_microbatches × V + num_stages − 1)`. Use this to predict the trade-off before submitting.

### Step 7 — Wave 6: cross-config-tag confirmation

If the model has multiple comparable `CONFIG_TAG`s, run the winning recipe (top 1-2 from Waves 2-4) on each of the others — one job each. Confirm the recipe doesn't regress them by more than 1 %. If it does, the deployment recommendation has to be config-conditional, not universal.

### Step 8 — Output: `<MODEL_TAG>-tuning.md`

Create or extend `<MODEL_TAG>-tuning.md` in the repo root. Required sections:

1. **Header bullets** (date, model, hardware, image, MaxText branch, base config path, sequence length, steps, steady-state metric definition)
2. **TL;DR** — best recipe per (parallelism × config) cell; comparison vs `baseline-deployed`; one-sentence "the winning flag wins because <mechanism>"
3. **Inventory** (from Step 2)
4. **Effect of `<MODEL_TAG>.env.sh` on `<PARALLELISM>`** (Wave 1 result)
5. **Tuning leaderboard** (top 8 + bottom 5; `Profile | Flags | TGS | step (s) | Δ% vs baseline-deployed`; mark winner ⭐)
6. **Why the winning flag wins** (mechanism, backed by Wave 1.5 evidence — cite job IDs and HLO/xplane artifacts)
7. **Negative findings** (which flags hurt; useful to skip on future sweeps)
8. **Sign flips vs other parallelisms** (if `<MODEL_TAG>-tuning.md` already covers another `PARALLELISM`, add a flag-by-flag delta table; otherwise omit)
9. **Recommended deployment** (proposed `configs/<MODEL_TAG>.env.sh` patch; if the winning recipe conflicts with a prior parallelism's recipe, propose a guard like `if [[ "${MAXTEXT_DCN_PP:-1}" -le 1 ]]`)
10. **Appendix: data sources** (job IDs, profiles, status — full table sorted by job ID)

If a Chinese sibling `<MODEL_TAG>-tuning.zh.md` exists or is requested, mirror the structure (same headers, same tables, code/flag names stay in English, prose translated).

## Per-parallelism flag catalog

These are starting candidates. Re-prioritize and extend based on Wave 1.5 trace evidence.

### FSDP-heavy (DCN ring all-gather + reduce-scatter dominate)

| Flag | Why it matters here |
|---|---|
| `--xla_gpu_all_gather_combine_threshold_bytes=<N>` | image default is often 8 GiB which fuses the per-step all-gather into one serial barrier. Sweep 256 MiB / 512 MiB / 1 GiB / 2 GiB; both ag-only AND ag+rs together |
| `--xla_gpu_reduce_scatter_combine_threshold_bytes=<N>` | usually leave at default — backward rs is already large per-layer; splitting just adds RCCL launch overhead |
| `--xla_gpu_enable_pipelined_all_gather/reduce_scatter/all_reduce=true` | cross-iteration prefetch — usually OOMs on this scale |
| `--xla_gpu_enable_while_loop_double_buffering=true` | cross-iteration overlap of `train_step` `while` body — usually negative on FSDP (HBM cost > overlap gain) |
| `--xla_gpu_enable_highest_priority_async_stream=true` | image default usually already prioritises async stream sufficiently |
| `--xla_gpu_experimental_parallel_collective_overlap_limit=2/4/8` | usually negative on FSDP — ring all-gather already saturates fabric, extra concurrency = contention |
| `--xla_gpu_enable_latency_hiding_scheduler=true` | image default may already have it on; check the inherited XLA_FLAGS in the baseline log |
| `NCCL_NCHANNELS_PER_NET_PEER=8` | sometimes +1-4 %; sometimes triggers init hangs in combos — see retry policy |
| `NCCL_BUFFSIZE=16M` | per-rank workspace — usually neutral or negative |
| `NCCL_PROTO=Simple/LL/LL128` | algorithm tweak; usually neutral on large FSDP messages |
| `XLA_PYTHON_CLIENT_MEM_FRACTION=.95` | more HBM headroom for prefetch buffers — usually within noise |

### PP-heavy (collective-permute + per-stage send/recv dominate)

| Flag | Why it matters here |
|---|---|
| `--xla_gpu_collective_permute_decomposer_threshold=<N>` | controls when `collective-permute` decomposes into send/recv. Sweep 256 MiB / 1 GiB / 8 GiB; the c-p ops in your HLO have a fixed size (e.g. `bf16[B,V,L,H]`) — find one bracketing threshold and one not-bracketing |
| `--xla_gpu_enable_pipelined_p2p=true` | cross-iteration prefetch on `collective-permute`. Usually no-op when the pipeline is `nn.scan`-based with a hard loop carry; verify in HLO |
| `--xla_gpu_enable_async_collective_permute=true` (if exists in your XLA build) | makes c-p async-launchable |
| `--xla_gpu_experimental_parallel_collective_overlap_limit=2/4/8` | often POSITIVE on PP — multiple independent fabrics (ICI for FSDP-style intra-stage ag, DCN p2p for c-p) enable real concurrency. Sweet spot is usually = number of independent fabrics involved (often 2) |
| `--xla_gpu_enable_highest_priority_async_stream=true` | helps when MoE skew creates per-rank stragglers (sparse path) |
| `--xla_gpu_enable_while_loop_double_buffering=true` | PP loop carry has different memory headroom; re-test fresh — image-default off-by-default |
| `--xla_gpu_enable_pipelined_all_reduce/reduce_scatter/all_gather=true` | per-stage HBM may absorb the prefetch buffers; test (often still OOMs) |
| `NCCL_ALGO=Ring/Tree/CollnetDirect` | per-stage 2-rank traffic prefers different algos than ring-based FSDP |
| `NCCL_PROTO=LL/LL128/Simple` | low-latency protocol may help small per-stage messages |
| `NCCL_NCHANNELS_PER_NET_PEER=2/4/8` | extra channels rarely help 2-rank c-p; test small variations |

### TP-heavy (ICI all-gather / reduce-scatter on sharded tensors)

| Flag | Why it matters here |
|---|---|
| `--xla_gpu_threshold_for_windowed_einsum_mib=<N>` | enables async TP via windowing — primary lever for TP overlap |
| `--xla_gpu_enable_async_all_gather=true` (if exists) | TP-style intra-step ag |
| `--xla_gpu_collective_inflation_factor=<N>` | controls collective sizing |
| `NCCL_NCHANNELS_PER_NET_PEER=<N>` | for in-node hops (most TP traffic is intra-node) |
| (most cross-iteration prefetch flags are not relevant to TP — TP doesn't have the iteration-spanning carry that PP has) | |

### Hybrid (e.g. PP×FSDP, FSDP×TP)

Read the Wave 1.5 step time breakdown to identify the dominant axis. Start with that axis's catalog. After 2-3 waves on the dominant axis, add a few flags from the secondary axis at the end. Don't sweep both axes' full catalogs — exponential blow-up.

## Mechanism cheatsheet (orientation for "Why the winning flag wins" section)

| Parallelism | Likely dominant cost | What XLA flags can address |
|---|---|---|
| **FSDP** | DCN ring all-gather of full per-step weights; backward reduce-scatter of full per-step gradients | break the mega-fused all-gather (combiner threshold); rarely cross-iteration prefetch (OOM) |
| **PP** | Per-stage `collective-permute` rendezvous (sync per call); MoE `dispatch+combine` per microbatch; pipeline bubble | concurrent collective execution (overlap_limit on dual-fabric ICI+DCN); decomposer threshold; async-stream priority for skew amelioration |
| **TP** | ICI all-gather/reduce-scatter on sharded weight/activation tensors | windowed einsum threshold; async ag/rs on intra-node fabric |
| **Hybrid** | Mix; check Wave 1.5 trace for which collective dominates | start from dominant-axis catalog |

Pipeline bubble (PP only) is mechanically determined: `(num_stages − 1) / (num_microbatches × V + num_stages − 1)`. No XLA flag changes this — only MaxText pipeline params do (Wave 5).

## Autonomous failure recovery

A 36-50-job sweep encounters node failures, RCCL flakes, transient JAX coordination errors, NIC bouncing, and similar issues many times. **Recover autonomously when possible**; only TG-stop the user when the issue is genuinely beyond the agent's reach. The user is away during the sweep — every interruption costs human attention.

The hard rule is: **do not change the nodelist mid-sweep.** Cross-nodelist hardware variance (rail layout, IB port wear, ROCm driver skew between nodes) introduces 1-3 % TGS noise that exactly drowns the flag-effect signal you're measuring. Falling back to a different nodelist invalidates the entire leaderboard. If a node from your pinned set cannot be revived after the recovery steps below, **TG and stop** — do not silently substitute or drop nodes.

### Known false-positive log signatures (filter these out before classifying)

These appear in *every* successful run on this stack. Do NOT treat them as failures, do NOT cancel, do NOT retry:

| Signature | Where | Why it's benign |
|---|---|---|
| `failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)` | every rank's worker stderr, dozens of times during Ray actor / JAX backend init | The JAX CUDA backend probes for a CUDA device on every rank even though we run on ROCm. The probe fails (no CUDA), JAX falls back to ROCm, training proceeds normally. Appears bit-identical in successful runs. |
| `NCCL WARN MSCCL++: Feature not enabled. ENABLE_MSCCLPP must be defined at compile-time` | every rank during RCCL clique init | The image isn't compiled with MSCCL++; RCCL uses its default algos. No training impact. |
| `NCCL WARN LL cutoff points not detected for a supported arch gfx950` (`rccl_wrap.cc`) | every rank during RCCL init | RCCL falls back to default LL thresholds. No training impact. |
| `WARNING: AMD GPU device(s) is/are in a low-power state` (rocm-smi during pre-flight) | pre-flight rocm-smi output when GPUs are idle | GPUs leave low-power state automatically when training starts. Not a fault. |

When triaging a hung or crashed job, **`grep -v` these patterns out first** before reading the tail — the actual signal-to-noise on this stack is poor and these warnings can scroll for thousands of lines around the real error.

### Per-failure recovery playbook

| Failure class | Symptom | Recovery sequence | Stop trigger |
|---|---|---|---|
| **RCCL init flake** | >10 min wall, no `completed step:`, NCCL WARN spam scrolling, `init.cc` / `rccl_wrap.cc` lines | `scancel <jobid>` → wait for COMPLETING → resubmit same TUNE_PROFILE on same nodelist | 3 consecutive flakes on the same TUNE_PROFILE → mark `infra-flake⋆`, drop the profile, continue with the rest |
| **JAX `DEADLINE_EXCEEDED: GetKeyValue() timed out`** | mid-training `RpcError` or `Coordination timeout`; rank 0 actor disappears with no log signature | Same: `scancel` → resubmit. Often clears on retry. | 2 consecutive timeouts on the same TUNE_PROFILE → cluster coord service may be sick → **TG-stop** with last-known leaderboard |
| **`Unknown flag in XLA_FLAGS`** | `parse_flags_from_env.cc:.*Unknown flag: --xla_…` then immediate abort | Open `train_env.sh`, remove the offending flag from the TUNE_PROFILE block, save → resubmit. Pending jobs already have their artifact frozen, so other profiles in the queue are unaffected. | typo on first-ever submission of a profile → fix once and continue; same flag fails after fix → flag is genuinely obsolete in this XLA build → drop the profile |
| **OOM** (`RESOURCE_EXHAUSTED: Out of memory while trying to allocate N GiB`) | clean OOM at compile or step 0 | **No retry.** Same flags = same OOM. Record `Total memory: X GiB / Temp: Y GiB` and the requested allocation in the doc. Drop the profile. | n/a — never a stop trigger; just continue |
| **Cleanup `exit=143` with `completed step: N-1` present** | training succeeded, post-train Docker cleanup race kills the process tree | **Treat as success.** Extract TGS from `completed step:` lines; ignore the cleanup status. | n/a |
| **Single node `drain` / `down` / `fail` state** in `sinfo` | a node from the pinned set is offline | (1) `scontrol update NodeName=<n> State=RESUME Reason=manual` via host-cmd → re-check `sinfo`. (2) If still drained: `scontrol show node <n>` to read the reason; common recoverable reasons are "Not responding", "Kill task failed", stale-drain-from-previous-job → `RESUME` usually clears them. (3) If `RESUME` doesn't take, ssh into the node and check `dmesg -T \| tail -50` and `rocm-smi` for fault signatures. | node stays drained / down after `RESUME` × 2 attempts → **TG-stop** |
| **`ionic_comp_msn cqe with error` / NIC fault on one node** | RDMA completion-queue error; job hangs at first collective | (1) Identify the failing node (rank index from log → look up `JOB_NODELIST` to get hostname). (2) Reset the ionic stack via host-cmd: `ssh <node> 'rmmod ionic_rdma; rmmod ionic; sleep 3; modprobe ionic; modprobe ionic_rdma'` (note: depends on `ionic_rdma` having no in-flight users — safe between jobs). (3) Verify `ssh <node> 'ibstat \| grep -c "State: Active"'` returns 8 (or whatever the per-node NIC count is on this hardware). (4) Resubmit. | NIC reset doesn't restore Active state → **TG-stop** |
| **`HSA_STATUS_ERROR` / `rocdevice.cpp: Aborting`** | GPU runtime error mid-training | Cancel; `ssh <node> 'rocm-smi --showtemp \| head'` to check thermal state; `dmesg -T \| grep -iE "amdgpu\|hsa\|gpu hang"` to look for hardware fault signatures. If it's a thermal issue or transient firmware glitch, wait 1-2 min, retry. | hardware fault signature in dmesg (XID-equivalent ECC, GPU reset, fabric link down) → **TG-stop** with the dmesg excerpt |
| **`NodeFail` event mid-job** | node drops out during training | Same as "Single node drain/down" above — try `RESUME`. | if multiple nodes fail in the same wave, the cluster has a wider problem → **TG-stop** |
| **Slurm `cgroup OOM` killing the entire docker container** with no JAX-side error | host-side memory pressure (rare on training jobs but possible if too many grain workers fork) | Read `journalctl -k --since '<time>' \| grep -iE 'oom\|killed'` on the host. Often resolved by freeing per-rank prefetch buffers (e.g. set `grain_worker_count=0` if grain accidentally got enabled — though tuning runs should be synthetic). | persistent host OOM with synthetic data → **TG-stop**; this means the model+pdbs combination is genuinely exceeding host RAM, not a flag-tuning issue |
| **Image tarball missing or unreadable** | `ls <path-from-DOCKER_IMAGE>.tar` fails (read the path from `container_env.sh`) | Re-check the path via host-cmd; check the fs mount via `df -h <mount-point>`. | path still missing → **TG-stop** (no submission can succeed) |
| **Slurm controller down** (`squeue` / `sbatch` returns errors via host-cmd) | `slurm_load_jobs error: Connection refused` or similar | Wait 5 min, retry. Slurm controllers occasionally restart. | controller down >15 min → **TG-stop** |
| **GitHub 500 during MaxText patch-branch checkout** | `remote: Internal Server Error` / `RPC failed; HTTP 5xx` / `fatal: unable to access 'https://github.com/...'` early in a job | Wait 5 min, resubmit. Hourly retry afterwards. (No max retries — this is a global outage, not a cluster issue.) | n/a — GitHub recovers on its own; just keep retrying. If a 4-hour outage blocks the sweep, TG-update so the user knows |

### Recovery primitives via host-cmd

```bash
# Resume a drained/down node
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 30 \
  "scontrol update NodeName=<n> State=RESUME Reason=manual; sleep 2; sinfo -n <n>"

# Inspect a node's state in detail
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 30 \
  "scontrol show node <n>"

# Reset ionic NIC stack (safe between jobs; NOT during a running job)
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 60 \
  "ssh <n> 'rmmod ionic_rdma; rmmod ionic; sleep 3; modprobe ionic; modprobe ionic_rdma; sleep 5; ibstat | grep -c \"State: Active\"'"

# Pull recent dmesg around a failure timestamp
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 30 \
  "ssh <n> 'dmesg -T | tail -100 | grep -iE \"oom|kill|fault|reset|amdgpu|hsa|ionic\"'"

# Check GPU temperature / fault state on a node
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 30 \
  "ssh <n> 'rocm-smi --showtemp; rocm-smi --showuse; rocm-smi --showxgmierr 2>/dev/null'"

# Verify all NICs back to Active after a reset
python3 /maxtext-slurm/.host-cmd/host_cmd.py --timeout 30 \
  "ssh <n> 'ibstat | grep -E \"^CA |State:\" | head'"
```

### TG-stop format

When a stop trigger fires, send a single TG message and halt the sweep. The message must include enough context for the user to either fix the cluster issue or decide to resume on a different nodelist:

```
[xla-tuning STOP] <MODEL_TAG> <PARALLELISM> sweep halted

Reason: <one sentence — node fault / coord timeout / controller down / etc.>
Failing node(s): <hostname[s]>; recovery attempted: <what you tried>; outcome: <what didn't work>

Sweep state at stop:
  - Nodelist: <pinned nodelist>
  - Profiles tested so far: N (top 3 below)
  - Best so far: <profile> @ <TGS> (Δ <%> vs baseline-deployed)
  - <profile> @ <TGS>
  - <profile> @ <TGS>
  - Pending profiles in queue: <list, cancelled by stop>

Artifacts: outputs/<jobid_lo>-* through outputs/<jobid_hi>-*
Awaiting your decision before resuming.
```

After sending, **do not resubmit anything** until the user replies. Spend the wait writing partial results into `<MODEL_TAG>-tuning.md` so the work isn't lost — the leaderboard so far + the stop reason + the artifacts is publishable as a "sweep interrupted at <N> profiles" preliminary report.

## Tuning runs are synthetic-only

**Every tuning submission must pass `steps=15 dataset_type=synthetic`**, with no exceptions. This is the operational rule that makes the leaderboard interpretable:

| Why | Detail |
|---|---|
| Data-loader variance ≈ flag-effect magnitude | grain/c4/HF-tokenizer pipelines add 1-2 % per-step TGS jitter; flag deltas of 1-3 % become unmeasurable under that noise |
| Determinism | synthetic data produces bit-identical input across runs, so TGS deltas are pure flag/schedule effects |
| Speed | 15 steps × ~30 s/step ≈ 7-8 min of training after compile — short enough to run 30+ profiles in a session |
| YAML may have left over loss-test config | `gpu.yml` may have `dataset_type: grain` and `steps: 2000` from a recent loss test; the CLI passthrough overrides these without editing the yml |

The only legitimate reason to use real data in this skill's flow is if the user explicitly asks for a **post-tuning loss validation** of a winning recipe — that's a separate one-shot run after the sweep, not part of the sweep.

## Common pitfalls

- **Always pin `--nodelist=…` for every submission.** TGS deltas of 1-3 % are exactly what you're hunting. Hardware variance between nodes (rail layout, IB port wear, ROCm version skew) easily exceeds that. The nodelist is frozen for the entire sweep — recovery options for unhealthy nodes are in [§ Autonomous failure recovery](#autonomous-failure-recovery), not a different nodelist.
- **Don't compose `XLA_FLAGS` from scratch.** APPEND your experimental flags via `XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }<flag>"`. The image's compiled-in defaults must be preserved — many old flags trigger `Unknown flag` aborts in current XLA.
- **For every transient failure (RCCL flake, JAX coord timeout, NIC fault, drained node, GitHub 5xx), follow [§ Autonomous failure recovery](#autonomous-failure-recovery) — try the recovery sequence, then TG-stop only when the listed stop-trigger fires.** The user is away during the sweep; do not interrupt them for routine flakes.
- **OOM is never retried.** Same flags = same OOM. Drop the profile, record allocation size.
- **Cleanup `exit=143` with `completed step: N-1` present** is a benign post-train artifact, especially common on PP. Trust the `completed step:` lines, not the cleanup status.
- **Profiler-on runs perturb steps 5-8.** If profiler is on, compute steady-state over steps 9-14 only. If profiler is off, steps 5-14 is fine.
- **Sign flips between parallelisms are systematic.** Don't assume FSDP findings transfer to PP, or vice versa. Test fresh.
- **`JOB_NAME` ≤ 243 bytes** — ext4's 255-byte path-segment limit minus 12 bytes for the jobid prefix. Keep TUNE_PROFILE aliases short (`pp8-A`, `Gco1G`, etc.); rely on `_env_TUNE_PROFILE=…` not the alias to disambiguate.
- **`profiler=` (empty) breaks the YAML enum validator.** If you want clean runs, leave the YAML's `profiler: ""` value and don't pass `profiler=` as a passthrough flag.

## Stop conditions

- A single flag (or 2-3 flag stack) gives ≥ +5 % over `baseline-deployed` AND doesn't regress other config tags by more than 1 %. → Document, propose deployment patch.
- Or: tested ≥ 25 distinct profiles across waves 2-4 and none beats baseline by > 2 %. → Conclude "structurally bounded at the editable scope". Document the negative finding with HLO + xplane evidence backing the conclusion.
- Either outcome is a successful run. The negative finding is just as valuable — it closes the question with a defensible answer.

## Pacing

| Stage | Typical wall time |
|---|---|
| Pre-flight + Step 2 inventory | 5-10 min |
| Wave 1 (4 jobs typically — pair × 2 config tags) | 60-90 min |
| Wave 1.5 (1 profiled job) | 30-50 min (profiler adds ~5 min) |
| Waves 2-4 (15-30 jobs) | 4-6 hours queue time |
| Wave 5 (PP only, 2-3 jobs) | 30-60 min |
| Wave 6 (cross-config, 1-2 jobs) | 20-40 min |
| Step 8 (writeup) | run in parallel with the last wave |
| **Total** | **6-8 hours queue time, ~36-50 jobs** |

If still searching at 8 hours of queue time, write up what you have and conclude — the practical ceiling is hit.

## Related skills

- [batch-sweep](../batch-sweep/SKILL.md) — find optimal `per_device_batch_size` for a (model × parallelism) cell. **Run first** if pdbs hasn't been tuned for this cell yet — XLA tuning is on top of an already-good batch size.
- [profile-drill](../profile-drill/SKILL.md) — kernel-level breakdown of xplane traces; use in Wave 1.5
- [job-log-triage](../job-log-triage/SKILL.md) — classify hangs, RCCL flakes, OOM signatures
- [model-config-guide](../model-config-guide/SKILL.md) — for hybrid parallelism choices and `<MODEL_TAG>.gpu.yml` overrides
- [telegram](../telegram/SKILL.md) — TG progress at each milestone (kickoff, every positive result, mid-point, final)
