---
name: model-config-guide
description: Create GPU config files to support existing MaxText model definitions on AMD GPU clusters. Use when the user wants to add a model, create a config, support a new model, or asks about model configs, parallelism, batch size, OOM, quantization, or .gpu.yml files.
---

# Model Config Guide

Create a `.gpu.yml` config to run a MaxText model on AMD GPUs. Most configs reference models already defined in MaxText's `configs/models/`, but configs can also define or tweak model architectures inline when needed.

## Workflow

### Step 0: Check GPU architecture

Batch size, memory budget, quantization support, and peak FLOPS all depend on the GPU. Check what's available:

```bash
rocm-smi --showproductname   # AMD
nvidia-smi -L                # NVIDIA
```

Key differences by GPU (see `utils/monkey_patch_maxtext.py` for the full table):

| GPU | VRAM | FP8 TFLOPS | BF16 TFLOPS |
|-----|------|------------|-------------|
| MI355X (CDNA 4) | 288 GB | 5000 | 2500 |
| MI300X (CDNA 3) | 192 GB | 2614 | 1307 |

More VRAM means larger `per_device_batch_size` and `max_target_length`. FP8 support enables `quantization: "fp8"` for memory savings and higher throughput on models that benefit from it.

### Step 1: Verify the model exists in MaxText

```bash
ls /workspace/maxtext/src/MaxText/configs/models/
```

If the model is there, use a model-name reference (simplest path). If not — or if you need to customize architecture parameters — define the model inline (see [Inline models](#inline-models) below).

For model-name configs, read the MaxText model file to understand the architecture (dense vs MoE, size, head counts, etc.) — this determines parallelism and memory decisions:

```bash
cat /workspace/maxtext/src/MaxText/configs/models/<model>.yml
```

### Step 2: Create the config file

Create `configs/<model-name>.gpu.yml`. The filename (minus `.gpu.yml`) becomes the model's short name for `submit.sh` and `run_local.sh`. Choose a name that is:
- Unique by substring (e.g., `submit.sh 70b` must match exactly one file)
- Descriptive (include size, variant)

Use the template below, filling in decisions from the following steps.

### Step 3: Choose parallelism strategy

**Key constraint:** at most one `-1` (auto-sharded) axis per group (DCN and ICI each). The product of all explicit axes in ICI must equal 8 (GPUs per node).

#### Dense models

Standard pattern — shard data across nodes, FSDP within each node:

```yaml
dcn_data_parallelism: -1         # auto-sharded across nodes
ici_fsdp_parallelism: 8          # shard weights across 8 GPUs per node
```

For very large dense models (405B+), FSDP on both levels:

```yaml
dcn_fsdp_parallelism: -1         # auto-sharded
ici_fsdp_parallelism: -1         # auto-sharded
```

#### MoE models

Standard pattern — FSDP across nodes, expert parallelism within each node:

```yaml
dcn_fsdp_parallelism: 8          # or -1 for auto
ici_expert_parallelism: 8        # spread experts across 8 GPUs per node
```

All other axes default to 1. Always set them explicitly for clarity (see template).

#### Pipeline parallelism (avoid if possible)

**Prefer FSDP over pipeline parallelism.** Pipeline parallelism in MaxText has known issues — likely related to JAX's SPMD model — and is harder to debug and tune. Use it only as a last resort when FSDP alone cannot fit the model. If needed, set `num_layers_per_pipeline_stage` (layers must divide evenly) and `num_pipeline_microbatches`, and disable `sparse_matmul`:

```yaml
dcn_pipeline_parallelism: 4
num_layers_per_pipeline_stage: 7    # total_layers / pipeline_stages
num_pipeline_microbatches: 32
sparse_matmul: false                # required with pipeline parallelism
```

#### Axes to avoid

- `dcn_sequence_parallelism` / `dcn_tensor_parallelism` — never recommended (comment: `# never recommended`)

### Step 4: Set batch and sequence length

Start from a known-working config for a similar-sized model and adjust. These are typical values on MI355X (288 GB); reduce for GPUs with less VRAM:

| Model size | Typical `per_device_batch_size` | Typical `max_target_length` |
|------------|--------------------------------|----------------------------|
| < 20B | 32–64 | 2048–4096 |
| 70B | 5–8 | 4096–8192 |
| 300B+ MoE | 10–22 | 4096–8192 |
| 400B+ dense | 3–5 | 8192 |

If OOM, reduce `per_device_batch_size` first, then `max_target_length`. If the model barely fits at batch size 1, increase `XLA_PYTHON_CLIENT_MEM_FRACTION` (default `.85` in `train_env.sh`) — this controls what fraction of GPU memory JAX pre-allocates. Setting it higher (e.g., `.90`, `.93`) gives more room for model weights and activations, but too high risks RCCL allocation failures. Note: XLA may inflate allocations when more memory is available, so increasing the fraction doesn't always yield proportional headroom.

For per-run testing, pass it via CLI:

```bash
submit.sh 70b -N 1 -- _env_XLA_PYTHON_CLIENT_MEM_FRACTION=.90
```

For permanent per-model overrides, create a `configs/<model>.env.sh` file (sourced after `train_env.sh`, before CLI `_env_` overrides):

```bash
# configs/my-large-model.env.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=.93
```

If the model is not yet supported by the image's default MaxText branch, set `MAXTEXT_PATCH_BRANCH` in the `.env.sh` to check out a branch that adds support:

```bash
# configs/new-model.env.sh
export MAXTEXT_PATCH_BRANCH="feature/new-model-support"
```

### Step 5: Set dtypes and quantization

Default for most models:

```yaml
dtype: bfloat16
weight_dtype: bfloat16
```

For large models where memory is tight, add FP8 quantization:

```yaml
quantization: "fp8"              # or "nanoo_fp8"
quantize_kvcache: false
kv_quant_axis: heads_and_dkv
kv_quant_dtype: int8
checkpoint_is_quantized: false   # true only when restoring from quantized checkpoint
```

### Step 6: Set MoE-specific fields (MoE models only)

```yaml
megablox: false
sparse_matmul: false             # cannot use with pipeline parallelism
capacity_factor: 1.25            # 1.0 for exact routing, >1.0 for overflow buffer
```

Add `expert_balance: true` if the model benefits from load-balanced routing.

### Step 7: Test locally

```bash
run_local.sh <short-name> -- steps=5 dataset_type=synthetic
```

Watch for:
- **Config resolution errors** — ambiguous name matching multiple files
- **OOM** — reduce `per_device_batch_size`
- **XLA compilation errors** — check MaxText model definition compatibility
- **NCCL/RCCL errors** — usually parallelism misconfiguration

### Step 8: Submit to cluster

```bash
submit.sh <short-name> -N <nodes> -- steps=15 dataset_type=synthetic
```

## Config template

```yaml
# ── Run Config ───────────────────────────────────────────────────────────────
base_config: "base.yml"
run_name: "<model>_train_test"
hardware: "gpu"
model_name: "<model>"

# ── Training ─────────────────────────────────────────────────────────────────
steps: 15

# ── Parallelism ──────────────────────────────────────────────────────────────
# At most one DCN axis can be unspecified (-1)
dcn_data_parallelism: -1 # auto-sharded
dcn_fsdp_parallelism: 1
dcn_sequence_parallelism: 1 # never recommended
dcn_context_parallelism: 1
dcn_tensor_parallelism: 1 # never recommended
dcn_pipeline_parallelism: 1
dcn_expert_parallelism: 1
# At most one ICI axis can be unspecified (-1)
ici_data_parallelism: 1
ici_fsdp_parallelism: 8
ici_sequence_parallelism: 1
ici_context_parallelism: 1
ici_tensor_parallelism: 1
ici_pipeline_parallelism: 1
ici_expert_parallelism: 1

# ── Batch & Sequence ─────────────────────────────────────────────────────────
per_device_batch_size: 8
max_target_length: 4096
packing: true
max_segments_per_seq: 32

# ── Attention & Compute ──────────────────────────────────────────────────────
attention: "cudnn_flash_te"
remat_policy: "full"
use_iota_embed: True
scan_layers: True
logits_dot_in_fp32: False
shardy: false

# ── Profiler ─────────────────────────────────────────────────────────────────
profiler: "" #"xplane"
upload_all_profiler_results: true
skip_first_n_steps_for_profiler: 3
profiler_steps: 1

# ── Checkpointing ────────────────────────────────────────────────────────────
enable_checkpointing: False
async_checkpointing: False

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset_type: "synthetic"
```

For MoE models, also add Dtypes and MoE Runtime sections between Attention and Profiler (see steps 5–6).

### Checkpointing

Leave `enable_checkpointing: False` in the YAML. Enable it via CLI passthrough instead:

```bash
submit.sh 70b -N 1 -- enable_checkpointing=true
```

**Why CLI, not YAML:** The launch system detects `enable_checkpointing=true` in the passthrough args to switch the output directory from job-based (`outputs/12345-JAX-llama2-70b/`) to model-based (`outputs/llama2-70b/`). This is what makes checkpoints persist across job restarts. Setting it only in the YAML would checkpoint to a job-specific directory that is lost on restart.

See [docs/job-submission.md](../../docs/job-submission.md) for model name aliases (isolating checkpoint dirs for parallel experiments) and async checkpointing caveats.

### Dataset and tokenizer

Use `dataset_type: "synthetic"` for benchmarking. For real data, set the dataset type and tokenizer:

```yaml
dataset_type: "hf"                                    # or "grain"
hf_path: "allenai/c4"
hf_data_dir: "en"
tokenizer_path: "meta-llama/Llama-2-7b-hf"           # match the model family
```

Some models require a specific tokenizer — check the MaxText model definition or existing configs. If using a gated model's tokenizer (e.g., `meta-llama/*`), set `hf_access_token` or log in via `huggingface-cli login`.

## GPU-specific overrides (always set)

These settings override MaxText's TPU-oriented defaults for GPU runs:

| Setting | Value | Why |
|---------|-------|-----|
| `hardware` | `"gpu"` | base.yml defaults to `tpu` |
| `attention` | `"cudnn_flash_te"` | Flash attention via cuDNN / Transformer Engine |
| `use_iota_embed` | `True` | Efficient embedding for GPU |
| `shardy` | `false` | Shardy is TPU-oriented |

## Compute settings by model size

| Setting | Small (< 20B) | Large (70B+) |
|---------|---------------|--------------|
| `remat_policy` | `"minimal_flash"` | `"full"` |
| `scan_layers` | `False` | `True` |

`full` rematerialization uses more compute but less memory — required for large models. `scan_layers: True` reduces compilation time for deep models.

## Inline models

For models not in MaxText's registry, define architecture parameters directly at the top of the config. Wrap in `#===` banners and comment out `model_name`:

```yaml
#===============================================================================
# Model Architecture: <Name> <Size>
base_emb_dim: 6144
base_num_query_heads: 48
base_num_kv_heads: 8
base_mlp_dim: 32768
base_num_decoder_layers: 64
head_dim: 128
mlp_activations: ["silu","linear"]
vocab_size: 131072
enable_dropout: False
logits_via_embedding: False
normalization_layer_epsilon: 1.0e-5
# MoE params (if applicable):
num_experts: 8
num_experts_per_tok: 2
decoder_block: "mixtral"
#===============================================================================

# ── Run Config ───────────────────────────────────────────────────────────────
base_config: "base.yml"
run_name: "<name>_train_test"
hardware: "gpu"
# NOTE: inline model; omit model_name!
#model_name:

# (rest of config follows the same template)
```

Use inline when the model doesn't exist in MaxText's registry, or when you need to tweak architecture parameters (e.g., adjusting expert count, hidden dims, or decoder block type) without modifying upstream.

## Common pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| `sparse_matmul` + pipeline parallelism | ValueError at runtime | Set `sparse_matmul: false` |
| Two `-1` axes in same group | Invalid parallelism config | Only one auto-sharded axis per DCN/ICI group |
| ICI axes product != 8 | Doesn't match GPUs per node | Ensure explicit ICI axes multiply to 8 |
| OOM | Batch too large, no quantization, or low mem fraction | Reduce `per_device_batch_size`, add `quantization: "fp8"`, or raise `XLA_PYTHON_CLIENT_MEM_FRACTION` (default `.85`). Memory budget depends on GPU — check with `rocm-smi` (AMD) or `nvidia-smi` (NVIDIA). |
| Ambiguous model name | Substring matches multiple `.gpu.yml` files | Use a more specific short name |
| `model_name` set in inline config | MaxText tries to load model file AND inline params | Comment out `model_name` |
| `remat_policy: minimal_flash` on large model | OOM during training | Switch to `remat_policy: full` |
| Checkpoints lost on job restart | `enable_checkpointing` set in YAML instead of CLI | Pass `-- enable_checkpointing=true` via CLI (see Checkpointing section) |

## Reference: existing configs

| Config | Type | Parallelism pattern | Notes |
|--------|------|---------------------|-------|
| `llama2-70b` | model-name, dense | DCN data / ICI FSDP | Baseline dense config |
| `llama3.1-405b` | model-name, dense | DCN FSDP / ICI FSDP (both -1) | FP8, very large dense |
| `mixtral-8x22b` | model-name, MoE | DCN FSDP=4 / ICI expert=8 | FP8, has pipeline params (prefer FSDP) |
| `deepseek3-671b` | model-name, MoE | DCN FSDP=8 / ICI expert=8 | Large MoE |
| `kimi-k2-1t` | model-name, MoE | DCN FSDP=8 / ICI expert=8 | Same pattern as deepseek3 |
| `qwen3-480b-a35b` | model-name, MoE | DCN FSDP=8 / ICI expert=8 | 160 experts, batch=24 sweep-tuned |
| `grok-1` | inline, MoE | DCN FSDP=4 / ICI expert=8 | Inline architecture, FP8 |
| `grok-2` | inline, MoE | DCN FSDP=4 / ICI expert=8 | Inline, FP8, softcapping, scan_layers=false (XLA ROCm bug) |
| `default` | model-name, small | DCN data / ICI FSDP | Smoke test, HF dataset |

See [docs/model-configs.md](../../docs/model-configs.md) for config resolution and CLI override details.
