#!/usr/bin/env python3
"""Ray actor wrapper for MaxText training.

Training runs in a subprocess (via subprocess.Popen) so that the training
process has its own Python interpreter with zero Ray threads.  This
eliminates GIL contention between Ray internals and the training loop.

The Ray actor handles:
  - Launching the subprocess on the correct node (NodeAffinity)
  - Streaming logs (subprocess inherits actor's stdout/stderr fds)
  - Collecting the exit code

Stack traces & flame graphs are available via the Ray Dashboard at
127.0.0.1:8265 on the head node — reach it via the SSH tunnel command
printed by `print_ray_info()` in `utils/ray_cluster.sh` (it does NOT
listen on the head's external interfaces; that's intentional, since
the dashboard's job-submission API is unauthenticated).  py-spy is
wrapped to target the training subprocess directly (see
`utils/ray_cluster.sh`).
"""

import os
import signal
import socket
import subprocess
import sys
import traceback

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


# ---------------------------------------------------------------------------
# Ray actor  (thin launcher — no training code runs here)
# ---------------------------------------------------------------------------

@ray.remote
class MaxTextTrainerActor:
    """Ray actor that launches training in a subprocess.

    Training runs in a separate Python process with no Ray threads,
    eliminating GIL contention.  The actor handles log routing and
    result collection.
    """

    def __init__(self):
        self.hostname = socket.gethostname()
        self.node_rank = int(os.environ.get("NODE_RANK", 0))
        self.tag = f"[Node {self.node_rank} @ {self.hostname}]"

    def run_training(self, argv: list, env_vars: dict) -> int:
        """Launch training in a subprocess and wait for result.

        In 1-GPU-per-process mode (``ONE_GPU_PER_PROCESS=true``), fans out
        to ``LOCAL_WORLD_SIZE`` subprocesses per node (one per GPU), each
        with its own ``LOCAL_RANK`` / ``GLOBAL_RANK`` / ``NPROCS``. Mirrors
        the fan-out that ``_train.sh`` does in the non-Ray path.

        Uses subprocess.Popen with env=env_vars to give the training process
        a clean environment (exactly what _train.sh exported) with no Ray
        thread contamination.  stdout/stderr are inherited from the actor
        worker, so output flows through Ray's log streaming automatically.
        """
        # Resolve the monkey_patch_maxtext.py script path (same entry point as non-Ray mode)
        script_dir = env_vars.get(
            "MAXTEXT_SLURM_DIR",
            os.path.dirname(os.path.abspath(__file__)),
        )
        monkey_patch_script = os.path.join(script_dir, "utils", "monkey_patch_maxtext.py")

        cmd = [sys.executable, "-u", monkey_patch_script] + list(argv)

        # Ensure PYTHONUNBUFFERED is set for real-time log streaming
        launch_env = dict(env_vars)
        launch_env["PYTHONUNBUFFERED"] = "1"

        if launch_env.get("ONE_GPU_PER_PROCESS", "").lower() == "true":
            return self._fan_out_one_gpu_per_proc(cmd, launch_env)

        print(f"{self.tag} Launching training subprocess: {' '.join(cmd[:3])} ...",
              flush=True)

        p = subprocess.Popen(
            cmd,
            env=launch_env,
            cwd=env_vars.get("PWD") or None,
        )
        print(f"{self.tag} Training subprocess started (pid={p.pid})",
              flush=True)

        p.wait()  # block until training finishes

        # ---- report result ----
        if p.returncode == 0:
            return 0
        elif p.returncode < 0:
            sig_num = -p.returncode
            try:
                sig_name = signal.Signals(sig_num).name
            except (ValueError, AttributeError):
                sig_name = f"signal {sig_num}"
            print(f"{self.tag} Training subprocess killed by {sig_name} "
                  f"(signal {sig_num})", flush=True)
        else:
            print(f"{self.tag} Training subprocess exited with code "
                  f"{p.returncode}", flush=True)
        return p.returncode

    def _fan_out_one_gpu_per_proc(self, cmd: list, launch_env: dict) -> int:
        """Launch one subprocess per local GPU; return first non-zero exit code."""
        local_world_size = int(launch_env["LOCAL_WORLD_SIZE"])
        nprocs = int(launch_env["NPROCS"])
        print(f"{self.tag} 1-GPU/proc: launching {local_world_size} "
              f"subprocesses (nprocs={nprocs})", flush=True)

        procs = []
        for i in range(local_world_size):
            penv = dict(launch_env)
            penv["LOCAL_RANK"] = str(i)
            penv["GLOBAL_RANK"] = str(self.node_rank * local_world_size + i)
            p = subprocess.Popen(cmd, env=penv, cwd=launch_env.get("PWD") or None)
            print(f"{self.tag}   proc {i}/{local_world_size} "
                  f"pid={p.pid} GLOBAL_RANK={penv['GLOBAL_RANK']}", flush=True)
            procs.append(p)

        first_nonzero = 0
        for p in procs:
            p.wait()
            if p.returncode != 0 and first_nonzero == 0:
                first_nonzero = p.returncode
        return first_nonzero


# ---------------------------------------------------------------------------
# Driver  (one per node — connects to Ray, creates actor, waits)
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: _ray_actor.py <config.yml> [key=value ...]")
        sys.exit(1)

    train_argv = sys.argv[1:]
    node_rank = int(os.environ.get("NODE_RANK", 0))
    captured_env = dict(os.environ)

    # Ensure MAXTEXT_SLURM_DIR is in the captured env so the actor can
    # reliably resolve monkey_patch_maxtext.py.  (submit.sh exports this on the host,
    # but it is not passed as a Docker --env flag.)
    captured_env.setdefault(
        "MAXTEXT_SLURM_DIR",
        os.path.dirname(os.path.abspath(sys.argv[0])),
    )

    ray.init(address="auto", namespace="maxtext", log_to_driver=True)

    # Pin actor to the local node; num_cpus=0 since the actor is just a
    # thin launcher (training runs in a subprocess, not in this process).
    local_node_id = ray.get_runtime_context().get_node_id()
    actor = MaxTextTrainerActor.options(
        name=f"maxtext_trainer_{node_rank}",
        num_gpus=0,
        num_cpus=0,
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=local_node_id,
            soft=False,
        ),
    ).remote()

    try:
        exit_code = ray.get(actor.run_training.remote(train_argv, captured_env))
    except Exception as e:
        print(f"[Node {node_rank}] Actor failed: {e}", flush=True)
        traceback.print_exc()
        exit_code = 1
    finally:
        # Best-effort: the actor may already be self-terminating, the GCS
        # connection on another rank may have torn down concurrently, the
        # RPC may have timed out, or the runtime context may be gone.  Any
        # of those would raise from `ray.kill`, which without the wrapper
        # would propagate out of `finally`, suppress `sys.exit(exit_code)`,
        # and cause Python to exit non-zero with a traceback EVEN WHEN
        # TRAINING SUCCEEDED.  Symptom is asymmetric rank-N exits at job
        # end (1 of N ranks exits 1, the others 0) for jobs that reached
        # the final training step cleanly.  Race exposure scales with N,
        # so the bug shows up most often on multi-node sweeps.
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            pass

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
