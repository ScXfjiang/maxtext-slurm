#!/bin/bash
# Per-model env overrides for llama3.1-70b multi-node FP8 training.
# Sourced by _train.sh after train_env.sh, before CLI _env_ overrides.

# Disable MSCCL: when only 4 RCCL channels are available, MSCCL can fire
# ncclCommSplit "invalid usage" with hierarchical communicators. Defensive
# even on flat FSDP.
export RCCL_MSCCL_ENABLE=0
export RCCL_MSCCLPP_ENABLE=0   # overrides train_env.sh default of 1

# Reduce NCCL bootstrap socket workers (defensive against multi-comm race).
export NCCL_NSOCKS_PERTHREAD=1
export NCCL_SOCKET_NTHREADS=1

# XLA autotune disabled (level 0 = heuristic kernel selection, no measurement).
# Trade-off: faster compile (saves ~10-15 min) but ~20% lower throughput vs level 4.
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=0 \
    --xla_gpu_memory_limit_slop_factor=95 \
    --xla_gpu_enable_triton_gemm=false \
    --xla_gpu_enable_cublaslt=true \
    --xla_gpu_enable_latency_hiding_scheduler=true \
    --xla_gpu_all_gather_combine_threshold_bytes=8589934592 \
    --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 \
    --xla_gpu_enable_all_gather_combine_by_dim=false"

# Raise HBM utilization for fully-sharded FSDP.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.97
