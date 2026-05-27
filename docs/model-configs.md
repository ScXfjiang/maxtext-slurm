# Model configs

Each `.gpu.yml` file in `configs/` defines a training run configuration. These files override [MaxText](https://github.com/AI-Hypercomputer/maxtext)'s [`base.yml`](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/MaxText/configs/base.yml) — only settings that differ from the defaults need to be specified.

## Available configs

Only models with a `.gpu.yml` file in `configs/` are supported — there is no built-in model registry. List the available configs by browsing the directory or running a resolve with a deliberately invalid name:

```bash
ls configs/*.gpu.yml
submit.sh __nonexistent_model__ 2>&1  # prints "Supported models: ..."
```

(Avoid `?` — it's a shell glob and may unexpectedly match a single-character path.)

Model-name configs contain `model_name: ...`; inline configs have the architecture parameters at the top (wrapped in `#===` banners) and `model_name` commented out.

## Adding a new config

1. **Choose a pattern.** If the model is already in MaxText, use a model-name reference. Otherwise, define the architecture inline.

2. **Create the file.** Name it `<model-name>.gpu.yml` in `configs/`. Follow the section layout below for consistency.

3. **Start minimal.** Only override settings that differ from `base.yml`. Common overrides:
   - `hardware: gpu` (base.yml defaults to `tpu`)
   - `attention: cudnn_flash_te` (GPU flash attention)
   - Parallelism axes for your target cluster size
   - `per_device_batch_size` and `max_target_length` for your memory budget
   - `dataset_type: synthetic` for benchmarking, or `grain`/`hf` for real data

4. **Test locally.** Verify the config loads and runs:

   ```bash
   run_local.sh <model-name> -- steps=5 dataset_type=synthetic
   ```

### Two patterns

#### 1. Model-name reference

The config sets `model_name` to reference a model already defined in MaxText (e.g. `llama2-70b`, `llama3.1-405b`, `mixtral-8x22b`). MaxText resolves the architecture parameters internally; the config only adds runtime settings.

```yaml
model_name: "llama2-70b"
```

#### 2. Inline model architecture

The config defines model architecture parameters directly (embedding dim, heads, layers, MoE settings, etc.) and **omits** `model_name`. This is used for custom or proxy models not built into MaxText.

```yaml
# NOTE: inline model; omit model_name!
#model_name:

# (architecture parameters defined in the file)
base_emb_dim: 4096
base_num_query_heads: 32
...
```

### Section layout

All configs follow a consistent section ordering. Sections that don't apply to a given config are omitted.

| Section | What it contains |
|---------|------------------|
| **Model Architecture** | Inline configs only. Architecture parameters wrapped in `#===` banners. |
| **Run Config** | `base_config`, `run_name`, `hardware`, `model_name` (or commented out) |
| **Training** | `steps`, `log_period`, goodput flags |
| **Parallelism** | DCN axes (inter-node), then ICI axes (intra-node). At most one axis per group can be `-1` (auto-sharded). |
| **Batch & Sequence** | `per_device_batch_size`, `max_target_length`, `packing`, pipeline settings |
| **Attention & Compute** | `attention`, `remat_policy`, `scan_layers`, `shardy` |
| **Dtypes & Quantization** | `dtype`, `weight_dtype`, `quantization`, KV cache quantization |
| **MoE Runtime** | `megablox`, `sparse_matmul`, `capacity_factor`, `expert_balance` (MoE models only) |
| **Profiler** | `profiler`, `upload_all_profiler_results`, step range |
| **Checkpointing** | `enable_checkpointing`, `async_checkpointing`, period, replica settings |
| **LR Schedule** | `learning_rate`, warmup, cosine decay (long training runs only) |
| **Dataset** | `dataset_type`, tokenizer, data paths |

## Config resolution

`submit.sh` and `run_local.sh` accept a short name that is resolved via substring matching against `configs/*.gpu.yml`:

```bash
submit.sh 70b -N 1              # matches llama2-70b.gpu.yml
submit.sh mixtral -N 4          # matches mixtral-8x22b.gpu.yml
submit.sh grok -N 8             # matches grok-1.gpu.yml
submit.sh ds-proxy-e128 -N 4    # matches ds-proxy-e128-h2048.gpu.yml
```

If the name is ambiguous (matches multiple files), the command fails with a list of candidates. If no file matches, it lists all supported models. Because resolution is purely file-driven, a model that doesn't have a `.gpu.yml` in `configs/` simply doesn't exist to the launcher — create one to add support (see [Adding a new config](#adding-a-new-config) above).

## Per-model environment overrides

Some models need environment variables that differ from the global `train_env.sh` defaults — for example, a larger `XLA_PYTHON_CLIENT_MEM_FRACTION` to fit a bigger batch size. Instead of passing `_env_` overrides on every submit, create a per-model env file:

```
configs/<model-name>.env.sh
```

If present, it is sourced after `train_env.sh` but before CLI `_env_` overrides:

```
train_env.sh  →  configs/<model>.env.sh (if exists)  →  _env_ CLI overrides
```

Example (`configs/my-large-model.env.sh`):

```bash
# This model needs a larger memory pool to fit the target batch size.
export XLA_PYTHON_CLIENT_MEM_FRACTION=.93
```

Per-model env files can also set `MAXTEXT_PATCH_BRANCH` to check out a branch that adds support for models not yet in the image's default MaxText (e.g. `export MAXTEXT_PATCH_BRANCH="feature/new-model-support"`).

Per-model env files are optional — models without one use the global `train_env.sh` defaults. CLI `_env_` overrides always take precedence, so per-job tuning still works.

## CLI overrides

Any `base.yml` setting can be overridden per-run via passthrough args after `--`. These override both `base.yml` defaults and the `.gpu.yml` config:

```bash
submit.sh 70b -N 1 -- per_device_batch_size=4 remat_policy=minimal_flash
run_local.sh 70b   -- steps=10 enable_checkpointing=true
```

See [Job Submission](job-submission.md) for the full CLI syntax including `_env_` overrides.
