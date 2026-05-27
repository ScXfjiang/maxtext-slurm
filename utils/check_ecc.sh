#!/usr/bin/env bash

# check_ecc.sh
#
# Usage:
#   ./check_ecc.sh '<nodelist>'        # e.g. 'node[1-4]' or 'host01,host02'
#   ./check_ecc.sh                     # defaults to current host only
#
# Behavior:
# - If user provides a nodelist: prints the parsed node list to stderr.
# - If no nodelist is provided: defaults to current host and does NOT print the node list.
# - Outputs one line per node to stdout, prefixed with "node: ".
# - Distinguishes SSH failures from "OK".
# - Runs locally when the target node is the current host (fixes PATH/env mismatch vs SSH).

set -euo pipefail

NODELIST="${1:-}"
PRINT_NODELIST=1
LOCAL_SHORT_HOST="$(hostname -s)"

# If no nodelist specified, default to current host and suppress nodelist printing
if [[ -z "${NODELIST}" ]]; then
  NODELIST="${LOCAL_SHORT_HOST}"
  PRINT_NODELIST=0
fi

# ---- Expand Slurm compacted nodelist ----
expand_nodelist() {
  local s="$1"
  if command -v scontrol >/dev/null 2>&1; then
    if scontrol show hostnames "$s" 2>/dev/null; then
      return 0
    fi
  fi

  # If it's already a plain hostname (no brackets), just echo it
  if [[ "$s" != *"["* ]]; then
    echo "$s"
    return 0
  fi

  # Fallback (single-bracket numeric ranges)
  if [[ "$s" =~ ^([^[]+)\[([0-9,\-]+)\]$ ]]; then
    local prefix="${BASH_REMATCH[1]}"
    local body="${BASH_REMATCH[2]}"
    local part a b i

    IFS=',' read -r -a parts <<< "$body"
    for part in "${parts[@]}"; do
      if [[ "$part" == *-* ]]; then
        a="${part%-*}"
        b="${part#*-}"
        for ((i=a; i<=b; i++)); do
          echo "${prefix}${i}"
        done
      else
        echo "${prefix}${part}"
      fi
    done
  else
    echo "ERROR: Unsupported nodelist format: $s" >&2
    exit 2
  fi
}

# ---- Parse nodelist ----
mapfile -t NODES < <(expand_nodelist "$NODELIST")

# ---- Print parsed nodelist ONLY if user provided one (to stderr) ----
if [[ "$PRINT_NODELIST" -eq 1 ]]; then
  {
    echo "========== PARSED NODE LIST =========="
    printf '%s\n' "${NODES[@]}"
    echo "Total nodes: ${#NODES[@]}"
    echo "======================================"
    echo
  } >&2
fi

# ---- ECC/UMC check ----
# Prints only GPUs with UMC CE>0 or UE>0; prints NO_ROCM_SMI sentinel if missing.
check_cmd='
set -o pipefail
command -v rocm-smi >/dev/null 2>&1 || { echo "NO_ROCM_SMI"; exit 0; }

rocm-smi --showrasinfo 2>/dev/null | awk "
  /GPU\\[[0-9]+\\]:/ {gpu=\$0}
  /^ *UMC/ {
    ce=\$4; ue=\$5
    if (ce !~ /^[0-9]+$/ || ue !~ /^[0-9]+$/) { ce=\$(NF-1); ue=\$NF }
    if (ce+0>0 || ue+0>0) printf(\"%s UMC CE=%s UE=%s\\n\", gpu, ce, ue)
  }
"
'

run_local() {
  bash -lc "$check_cmd" 2>/dev/null || true
}

run_remote() {
  local node="$1"
  # Verify SSH connectivity first so we don't confuse ssh failure with "OK"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no "$node" "true" >/dev/null 2>&1; then
    echo "SSH_FAILED"
    return 0
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no "$node" \
    "bash -lc $(printf '%q' "$check_cmd")" 2>/dev/null || true
}

for node in "${NODES[@]}"; do
  if [[ "$node" == "$LOCAL_SHORT_HOST" ]]; then
    out="$(run_local)"
  else
    out="$(run_remote "$node")"
  fi

  if [[ "$out" == "SSH_FAILED" ]]; then
    echo "${node}: SSH_FAILED"
    continue
  fi

  if [[ -z "$out" ]]; then
    echo "${node}: OK (no ECC/UMC errors reported)"
    continue
  fi

  if [[ "$out" == "NO_ROCM_SMI" ]]; then
    echo "${node}: rocm-smi not found"
    continue
  fi

  echo "$out" | sed "s/^/${node}: /"
done
