# Extensibility

This project is a **reference implementation** of a layered system for running [JAX](https://jax.dev/)-based LLM training on GPU clusters. It currently uses [MaxText](https://github.com/AI-Hypercomputer/maxtext), [Slurm](https://slurm.schedmd.com/), and [Docker](https://www.docker.com/), but the architecture is designed so that each of these choices can be adapted independently — the cost is confined to a single layer, ranging from zero changes (scheduler) to a few file edits (training framework).

The guiding principle:

> **Seamless when possible; minimal effort when not.** If the system can auto-detect a user's environment choice (e.g., Docker vs [Podman](https://podman.io/)), it should — transparently, with no configuration. When auto-detection isn't feasible (e.g., Slurm vs [Kubernetes](https://kubernetes.io/)), the layered architecture confines the change to a single tier so the rest of the stack stays untouched.

## Layer map

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Orchestration       submit.sh, _job.sbatch, reservation                     │  ← scheduler-specific
├──────────────────────────────────────────────────────────────────────────────┤
│  Container Boundary  _container.sh, docker_utils.sh                          │  ← runtime-specific
├──────────────────────────────────────────────────────────────────────────────┤
│  Training            _train.sh, train_env.sh, Python code                    │  ← framework-specific
├──────────────────────────────────────────────────────────────────────────────┤
│  Utilities           artifact, preflight, stage_timeout, metrics, tooling    │  ← mostly generic *
└──────────────────────────────────────────────────────────────────────────────┘
```

\* Most utilities are framework-agnostic (`stage_timeout.sh` is fully launcher-agnostic; the metrics plugin system is entirely framework-independent). A few straddle the training boundary: `monkey_patch_maxtext.py` imports MaxText directly, `tgs_tagger.py` parses MaxText log format, and `resolve_model_name.sh` resolves `.gpu.yml` configs. Swapping the training framework requires updating these (see [Axis 3](#axis-3-training-framework-maxtext--custom-jax-code)).

Each layer communicates with its neighbors through environment variables and calling conventions — never by reaching across layers. Adapting one layer requires no changes — or at most minor guards — in the others.

## Axis 1: scheduler (Slurm → Kubernetes, etc.)

**Current state.** Slurm coupling is confined to the orchestration tier: `submit.sh` (calls `sbatch`), `_job.sbatch` (Slurm directives + `srun`), and monitoring utilities (`utils/slurm_job_monitor.sh`, `utils/reservation.sh`). `_job.sbatch` maps Slurm-specific variables (`SLURM_JOB_ID`, `SLURM_JOB_NUM_NODES`, `SLURM_NODEID`, etc.) to generic env vars (`JOB_ID`, `NNODES`, `NODE_RANK`, etc.) before calling any downstream script. The container boundary (`_container.sh`) and everything below it use only these generic names — zero scheduler awareness.

**To add a new scheduler** (e.g., Kubernetes):

1. Write a new orchestration entry point that sets the same generic env vars (`JOB_ID`, `NNODES`, `NODE_RANK`, `JAX_COORDINATOR_IP`, `NODELIST_EXPANDED`). For observability, also set `RAY=1`. See the env var contract documented at the top of `_container.sh`.
2. Call `in_container_run.sh` (when the pod IS the container) or `_container.sh` (when Docker launch is needed). Everything downstream runs unmodified.
3. **Set `JAX_COORDINATOR_IP` to the rank-0 node's cluster-private IP** via `source utils/detect_ip.sh; export JAX_COORDINATOR_IP="$(detect_cluster_ip)"`. The default (e.g., `SLURM_LAUNCH_NODE_IPADDR` on Slurm or `hostname -I | awk '{print $1}'` on dual-IP hosts) often returns the public IP, which exposes the unauthenticated JAX coordinator and Ray GCS to the internet — internet scanners join open Ray clusters as raylets and run code inside them within seconds. `_job.sbatch` and `_k8s_job.sh` already do this; copy the same idiom in any new entry point. Single-node and no-private-network setups fall back to the public IP automatically.

This has been implemented for Kubernetes: `k8s_submit.sh` generates an Indexed Job manifest that calls `_k8s_job.sh` (coordinator discovery + barrier), which delegates to `in_container_run.sh`. See [Kubernetes job submission](k8s-job-submission.md).

**Effort:** New files only. No changes to existing layers. See [Architecture: Orchestrator Extensibility](architecture.md#orchestrator-extensibility) for the tier table.

## Axis 2: execution environment (container → native)

**Current state.** The default path (`submit.sh`, `run_local.sh`) runs training inside a container (Docker or Podman, auto-detected by `docker_utils.sh`), with coupling isolated to `_container.sh`. `in_container_run.sh` already provides a container-launch-free path — it sources `utils/run_setup.sh` for shared setup, performs container-internal initialization (ulimit, coredump, pip installs), and calls `_train.sh` directly. Everything downstream (`_train.sh`, `train_env.sh`) is container-agnostic.

**To add a fully native (no container at all) execution path:**

`in_container_run.sh` is the starting point — it already bypasses the Docker/Podman launch. Adapting it for a bare-metal host requires:

1. Guard the container-specific setup (coredump paths, pip installs) behind a container-presence check, or extract it into a separate step.
2. Set `JOB_WORKSPACE` (or `OUTPUTS_BASE_DIR`) explicitly when you need a custom output root. Output paths resolve via `utils/job_dir.sh`, while default direct in-container runs use `outputs/` next to the scripts.
3. Add a container-presence guard (~5 one-line checks) to `release_gpu.sh` and `preflight.sh` so they skip container operations when running natively.
4. **Observability prerequisites** — In the container-based flow, [Ray](https://www.ray.io/) is `pip install`'d and [Prometheus](https://prometheus.io/) is downloaded automatically at startup (`ray_cluster.sh`, `prometheus.sh`). For native execution, these must be pre-installed on every node (or the install functions reused outside the container). The rest of the observability pipeline (metrics exporter, plugins, TSDB persistence) works as-is.

**Effort:** Minor guards on `in_container_run.sh` + observability dependencies on the host. See [Architecture: Container Boundary](architecture.md#container-boundary) for details.

## Axis 3: training framework (MaxText → custom JAX code)

**Current state.** MaxText is the training framework used in this reference implementation. It is baked into the Docker image (configured in `container_env.sh`). The `import MaxText` boundary exists in exactly two Python files (`monkey_patch_maxtext.py` and `_ray_actor.py`), both invoked by `_train.sh`. Shell scripts pass training arguments opaquely (`key=value` pairs after `--`), so the orchestration layer has no framework awareness.

**To plug in a different JAX training framework:**

1. Provide the new framework. Two options: build a new Docker image with it pre-installed (set `DOCKER_IMAGE` via the command line or edit `container_env.sh` — see [Job Submission: `container_env.sh`](job-submission.md#container_envsh-docker-image-and-paths)), or bind-mount the code and `pip install` at startup — viable for pure-Python frameworks whose deps (JAX, Flax, etc.) are already in the base image.
2. Replace the `MaxText.train.main()` call in `monkey_patch_maxtext.py` and `_ray_actor.py` with the new framework's entry point — or make it configurable (e.g., `$TRAIN_MODULE`).
3. Supply new model config files (replacing `configs/*.gpu.yml`).
4. Update log-parsing regexes in `tgs_tagger.py` if the new framework's output format differs.

The orchestration tier, environment configuration (`train_env.sh`), observability pipeline, artifact system, and all utility scripts remain unchanged.

**Effort:** A new Docker image + 3–4 file changes in the training tier. See [Architecture: Training Flow](architecture.md#training-flow) for the call graph.

## Axis 4: GPU vendor (AMD → NVIDIA)

**Current state.** The container boundary (`_container.sh`) auto-detects AMD vs NVIDIA GPUs and passes the appropriate device flags. `train_env.sh` exports AMD/ROCm-specific variables (RCCL, HSA, CK flags); on NVIDIA these are harmless no-ops or can be gated. The only vendor-locked code is in `gpu_metrics_plugin.sh` (reads AMD-specific sysfs paths).

**To add NVIDIA support:**

1. Guard the AMD-specific exports in `train_env.sh` behind a vendor check (or split into `train_env_amd.sh` / `train_env_nvidia.sh`).
2. Write an `nvidia_gpu_metrics_plugin.sh` that uses `nvidia-smi` — the plugin discovery in `metrics_exporter.sh` picks it up automatically.

**Effort:** Moderate — mostly `train_env.sh` refactoring. The container boundary already handles both vendors.

## Extending the observe tier: metrics plugins

The four axes above adapt the **launch tier** — scheduler, container, framework, GPU vendor. The metrics plugin system adapts the **observe tier**, and it is the most immediately impactful extensibility path: the plugin system directly shapes what the [ground truth TSDB](observability.md#unified-tsdb-as-ground-truth) contains.

Ray provides a broad metrics baseline (host memory, CPU, disk, GPU utilization). The plugin system extends this baseline with domain-specific depth using **automatic plugin discovery**: any `*_metrics_plugin.sh` script in `utils/` is picked up by `metrics_exporter.sh` and run on every node — no registration, no configuration changes.

This means:

- **Adding a metric** (e.g., InfiniBand error counters, filesystem I/O) requires only a new plugin script — it lands in the TSDB on the next job start.
- **Swapping GPU vendors** (Axis 4) requires only a new GPU metrics plugin — the exporter, Prometheus pipeline, and TSDB persistence are unchanged.
- **Swapping training frameworks** (Axis 3) leaves all system-level metrics intact. Training metrics are also framework-agnostic — `tb_metrics_plugin.sh` reads raw TFRecord format, so it works with any framework that writes [TensorBoard](https://www.tensorflow.org/tensorboard) events.

The exporter provides infrastructure (state transport, plugin discovery) while plugins own the semantics — making the system open-ended without any central registration. See [Observability: Customizing Metrics](observability.md#customizing-metrics) for the plugin authoring guide and [Architecture: Observability Pipeline](architecture.md#observability-pipeline-ray-mode) for the high-level pipeline.

## Summary

| Axis | Tier | Isolation | What changes | What doesn't |
|------|------|-----------|-------------|--------------|
| **Metrics plugin** | **Observe** | **Excellent** | **New `*_metrics_plugin.sh` only** | **Exporter, Prometheus, TSDB, all other plugins** |
| Scheduler | Launch | Excellent | Orchestration tier only (new files) | Container, training, utilities |
| Execution env | Launch | Good | `in_container_run.sh` + ~5 guards (mostly done) | Orchestration, training, utilities |
| Training framework | Launch | Good | Docker image + 2 Python files + configs | Orchestration, utilities |
| GPU vendor | Launch | Moderate | `train_env.sh` + 1 metrics plugin | Orchestration, container boundary, training code |

Adapting any axis confines the work to a single tier. The metrics plugin system is the lowest-friction path — drop a script in `utils/` and it appears in the TSDB on the next job. The launch-tier axes require explicit work, but that work stays within its tier.
