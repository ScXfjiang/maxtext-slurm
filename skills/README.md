# AI Skills

Structured instructions for AI agents. Each skill encodes the methodology from very senior systems engineers — not just what commands to run, but how to interpret results, distinguish symptoms from root causes, and trace causal chains across the full stack. The agent reads the relevant skill on demand when the user's request matches its trigger keywords.

## Available skills

| Skill | Trigger |
|-------|---------|
| [performance-analysis](performance-analysis/SKILL.md) | "analyze job", "TGS", "TraceLens", "IRLens", profiling tasks |
| [profile-drill](profile-drill/SKILL.md) | "xplane", "per-kernel breakdown", "step-time composition", "main-stream-busy", "input_scatter_fusion", "RaggedAllToAllKernelImpl", cross-variant kernel comparison, TraceLens CSV off by 1.5–2× |
| [job-log-triage](job-log-triage/SKILL.md) | "triage", "diagnose", "why did job fail", "is the job hanging", crash/hang/timeout/OOM/NCCL errors, job status |
| [tsdb-diagnosis](tsdb-diagnosis/SKILL.md) | "diagnose with TSDB", "check GPU health", "check network", "query prometheus", "metrics", incident root cause analysis |
| [coredump-debug](coredump-debug/SKILL.md) | "coredump", "core file", "SIGSEGV", "segfault", "crash dump", GDB backtrace analysis, crash root cause |
| [model-config-guide](model-config-guide/SKILL.md) | "add model", "create config", "model config", ".gpu.yml", parallelism, batch size, quantization, OOM tuning |
| [batch-sweep](batch-sweep/SKILL.md) | "sweep", "find optimal batch size", "tune TGS", "benchmark throughput", "maximize tokens per second" |
| [xla-tuning](xla-tuning/SKILL.md) | "tune XLA flags", "tune NCCL", "find best collective-permute / all-gather threshold", "optimize FSDP/PP/TP", "close parallelism throughput gap", "cross-iteration prefetch" / "overlap-limit" / "async-stream-priority" sweeps for one (model × parallelism) cell |
| [docker-artifact-check](docker-artifact-check/SKILL.md) | "check container", "software versions", "git hashes", "what's in the image", Docker/ROCm/JAX artifact inventory |
| [telegram](telegram/SKILL.md) | "notify me", "send TG message", "alert when done", "wait for TG reply", "Telegram", cross-cutting I/O channel with REPL mode for any skill |
| [pre-commit-audit](pre-commit-audit/SKILL.md) | "pre-commit audit", "before commit", "ready to commit", "verify all launcher paths", "trace launcher chain", "audit (entry × launch × stack)", "(entry × launch × stack) matrix", "post-launch teardown verification", "verify scripts / utils not broken", "smoke-test the changed scripts", "any utility script broken", "code quality", "design review", "code smells", "tighten and polish", "avoid quality decay", "revisit design choice", "scrub leaks", "check for sensitive info before commit", "any docs or skills need update", "any stale comments", "any inaccurate comments", "comment alignment", "link policy", "broken anchors", any change to `_train.sh` / `_train_with_ray.sh` / `_ray_actor.py` / `_container.sh` / `_job.sbatch` / `_k8s_job.sh` / `in_container_run.sh` / `run_local.sh` / `submit.sh` / `k8s_submit.sh` / `utils/run_setup.sh` / `utils/ray_cluster.sh` / `utils/monkey_patch_maxtext.py`, any change to `utils/*.sh` / `utils/*.py` (analysis utilities, sourced libraries), plus any commit on the repo (Steps 5.2 / 6 / 7 / 8 are universal) |

## How agents discover skills

Both [Cursor](https://cursor.com/) and [Claude Code](https://docs.anthropic.com/en/docs/claude-code) read `CLAUDE.md` at the repo root, which references skill files by path.

For [Kubernetes](https://kubernetes.io/) job submission and direct-container runs, see [Kubernetes job submission](../docs/k8s-job-submission.md).
