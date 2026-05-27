#!/usr/bin/env python3
"""Shared node identity and paths for host-cmd (per-node queue / lock / pid / log)."""

from __future__ import annotations

import os
import re
import socket
import sys
from pathlib import Path


def resolve_node_id() -> str:
    """Short, filesystem-safe id for this machine.

    Uses HOST_CMD_NODE if set (recommended inside containers so it matches the
    physical host), else socket.gethostname(). Only the first DNS label is used
    so FQDNs map to the same queue as ``hostname -s`` on the host.
    """
    raw = (os.environ.get("HOST_CMD_NODE") or socket.gethostname()).strip()
    if not raw:
        raw = "unknown"
    short = raw.split(".")[0].strip().lower()
    safe = re.sub(r"[^a-z0-9_-]+", "_", short)
    safe = safe.strip("_") or "unknown"
    return safe[:200]


def paths_for_host_cmd(host_cmd_dir: str | Path, node_id: str | None = None) -> dict[str, Path | str]:
    """Paths for one host-cmd instance (one queue consumer per node_id)."""
    base = Path(host_cmd_dir).resolve()
    nid = node_id if node_id is not None else resolve_node_id()
    return {
        "node_id": nid,
        "queue_dir": base / "queue" / nid,
        "running_dir": base / "running" / nid,
        "results_dir": base / "results" / nid,
        "pid_file": base / f"daemon.{nid}.pid",
        "lock_file": base / f"daemon.{nid}.lock",
        "log_file": base / f"host_cmd_server.{nid}.log",
    }


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "node-id":
        print(resolve_node_id())
        return
    print("usage: host_cmd_common.py node-id", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
