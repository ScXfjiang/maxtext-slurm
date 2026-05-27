# MaxText on Slurm

Toolkit for launching and observing [MaxText](https://github.com/AI-Hypercomputer/maxtext) — a [JAX](https://jax.dev/)-based LLM training framework — on [Slurm](https://slurm.schedmd.com/)-managed AMD GPU clusters. One command runs distributed training with model selection, container setup, and multi-node coordination handled automatically. A carefully designed [observability](docs/observability.md) stack (via `RAY=1`) records GPU state, host health, network diagnostics, and training metrics into a single [time-series database](https://en.wikipedia.org/wiki/Time_series_database) (TSDB) — essential for monitoring long-running production training, diagnosing incidents, and day-to-day tuning and validation.

An extensible [AI skill framework](skills/) equips AI assistants with out-of-the-box domain expertise across the full training lifecycle — from profiling and performance analysis to production incident diagnosis — with more skills added over time.

This is a **reference implementation** of a [layered](docs/architecture.md) launch architecture where each tier — orchestration, container, training — can be [adapted independently](docs/extensibility.md) without cascading changes, including swapping the GPU vendor. See [Architecture](docs/architecture.md) for the full training flow and [Extensibility](docs/extensibility.md) for what each swap involves.

For [Kubernetes](https://kubernetes.io/) job submission, see [Kubernetes job submission](docs/k8s-job-submission.md).

## Quickstart

Clone the repo onto a Slurm-managed GPU cluster and run:

```bash
submit.sh 70b -N 1              # Train llama2-70b on 1 node
run_local.sh 70b -- steps=10    # Run locally (no Slurm)
RAY=1 submit.sh 70b -N 1        # With full-stack observability
```

`70b`, `mixtral`, `grok`, etc. are short names that resolve to config files in [`configs/`](configs). To train a model not already listed, [add a config](docs/model-configs.md#adding-a-new-config).

**Shared filesystem.** Multi-node jobs need all nodes to reach the output directory. Either clone this repo onto a shared filesystem (the default `outputs/` dir works as-is), or set `JOB_WORKSPACE` to a shared path.

```bash
# Common options
JOB_WORKSPACE=/shared/maxtext_jobs submit.sh 70b -N 1            # Shared FS for outputs
submit.sh 70b -N 1 -- remat_policy=full per_device_batch_size=2  # MaxText args after --
submit.sh 70b -N 1 -- _env_NCCL_DEBUG=INFO                       # Env var overrides
STAGE_TIMEOUTS="preflight:600,train:86400" submit.sh 70b -N 24   # Per-stage timeouts
```

## Observe

`RAY=1` adds full-stack observability with no measurable steady-state throughput impact. Fully self-contained in the [Docker](https://www.docker.com/)-based stack — all necessary components are installed automatically at container startup, no cluster-wide setup required. Powered by [Ray](https://www.ray.io/) and [Prometheus](https://prometheus.io/) with an [extensible plugin system](docs/observability.md#customizing-metrics), the pipeline collects GPU, host, network, and [TensorBoard](https://www.tensorflow.org/tensorboard) training metrics into a single Prometheus TSDB — persisted to the shared filesystem, queryable live and after the job ends, making the TSDB the **ground truth data layer** for automated reasoning and diagnosis.

```bash
RAY=1 submit.sh 70b -N 8
```

### Live dashboards

All three dashboards bind to `127.0.0.1` on the job's head node (loopback only — the Ray dashboard's job-submission API is unauthenticated, so wide binding is unsafe). Inter-node services (Ray GCS, JAX coordinator) bind to the cluster's RFC1918 subnet, not the public IP. Reach the dashboards through the SSH ProxyJump tunnel command printed in the job log; see [Observability: SSH tunnel](docs/observability.md#ssh-tunnel).

| Dashboard | What it shows | Port (head's localhost) |
|-----------|--------------|------|
| TensorBoard | Training loss curves, learning rate schedules | 6006 |
| Ray Dashboard | Actor status, live stack traces, flame graphs | 8265 |
| Prometheus | GPU, host, network, and training metrics — unified TSDB | 9190 (auto-increments if occupied) |

### Post-run browsing

```bash
utils/prometheus.sh list                                          # List jobs with saved metrics
utils/prometheus.sh view outputs/12345-JAX-llama2-70b/prometheus  # Browse metrics after the job ends
```

### Telegram notifications

Push notifications for job state changes and hang detection — works with any Slurm job, no `RAY=1` required. See [Notifications](docs/notifications.md) for setup.

```bash
utils/slurm_job_monitor.sh -j <slurm_job_id>
```

See [Observability](docs/observability.md) for the full story.

## Agentic workflows

The [`skills/`](skills/) directory contains structured instructions for AI agents ([Cursor](https://cursor.com/), [Claude Code](https://docs.anthropic.com/en/docs/claude-code)). Each skill encodes the methodology from very senior systems engineers — not just what commands to run, but how to interpret results, distinguish symptoms from root causes, and trace causal chains across the full stack. Skills ship for performance analysis, job failure triage, and TSDB-based incident diagnosis. See [`skills/README.md`](skills/README.md) for the full list.

## Learn more

**Launch**

| Topic | Description |
|---|---|
| [Job Submission](docs/job-submission.md) | `JOB_WORKSPACE`, full `submit.sh` syntax, local runs, checkpointing, model aliases, environment config, artifacts |
| [Kubernetes job submission](docs/k8s-job-submission.md) | `k8s_submit.sh` usage, PVC setup, per-rank logs, differences from Slurm, direct-container runs |
| [Model Configs](docs/model-configs.md) | Available models, adding new configs (patterns, section layout), resolution rules, CLI overrides |

**Observe**

| Topic | Description |
|---|---|
| [Observability](docs/observability.md) | Real-time monitoring, post-run diagnostics, unified TSDB, custom metrics plugins, design rationale |
| [Notifications](docs/notifications.md) | Telegram setup, programmable messaging (`telegram_bot.sh`), automated job monitoring |
| [Performance](docs/performance.md) | Profiling (xplane traces, HLO dumps), analysis (TraceLens, IRLens), tuning |
| [Debugging](docs/debugging.md) | Core dumps, crash reproduction, unresponsive nodes |
| [Tooling](docs/tooling.md) | Performance analysis, log tailing, job monitoring, Prometheus inspection, artifact cleanup |

**Adapt**

| Topic | Description |
|---|---|
| [Architecture](docs/architecture.md) | Training flow, artifact system, observability pipeline, orchestrator extensibility |
| [Extensibility](docs/extensibility.md) | Layer map; how to swap schedulers, runtimes, training frameworks, or GPU vendors |

## Acknowledgements

- Documentation compiled with the assistance of Cursor and [Claude-4.6-opus-high](https://anthropic.com/claude).
- Observability stack implemented in collaboration with Cursor and Claude-4.6-opus-high.
- AI skill framework and skills built with Cursor and Claude-4.6-opus-high.
- Some utility scripts developed using [ChatGPT 5/5.1](https://openai.com/chatgpt).

## License

This project is licensed under the [MIT License](LICENSE).
