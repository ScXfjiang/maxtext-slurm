#!/bin/bash
#
# Auto-detect NCCL network environment variables (IB HCA, QoS, socket interface).
#
# Sourced by train_env.sh (script mode) and _container.sh (interactive mode).
# Each variable is only set when not already present, so manual overrides
# and _env_KEY=VALUE passthrough args take precedence.
#
# Usage:
#   source utils/detect_nccl_env.sh

_DETECT_NCCL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# NCCL_IB_HCA: enumerate InfiniBand HCA devices
if [[ -z "${NCCL_IB_HCA:-}" && -d /sys/class/infiniband ]]; then
    NCCL_IB_HCA=$(ls /sys/class/infiniband 2>/dev/null | tr '\n' ',' | sed 's/,$//')
    [[ -n "$NCCL_IB_HCA" ]] && export NCCL_IB_HCA
fi

# NCCL_IB_TC / NCCL_IB_FIFO_TC: Pensando AINIC QoS auto-detection
if [[ -z "${NCCL_IB_TC:-}" && -z "${NCCL_IB_FIFO_TC:-}" ]]; then
    source "$_DETECT_NCCL_DIR/detect_ainic_nccl_ib_tc.sh"
    if is_pensando; then
        _tc=$(detect_pensando_tc)
        NCCL_IB_TC=$(echo "$_tc" | awk '{print $1}')
        NCCL_IB_FIFO_TC=$(echo "$_tc" | awk '{print $2}')
        echo "[INFO] $(hostname -s): Pensando AINIC detected: NCCL_IB_TC=$NCCL_IB_TC NCCL_IB_FIFO_TC=$NCCL_IB_FIFO_TC"
        [[ -n "$NCCL_IB_TC" ]] && export NCCL_IB_TC
        [[ -n "$NCCL_IB_FIFO_TC" ]] && export NCCL_IB_FIFO_TC
        unset _tc
    else
        echo "[INFO] $(hostname -s): Not a Pensando AINIC cluster, no NCCL_IB_TC/NCCL_IB_FIFO_TC override needed"
    fi
fi

# NCCL_IB_GID_INDEX: auto-detect the RoCEv2 GID index on ACTIVE ports.
# The kernel/driver sometimes places the routed (fd93:... / ULA) GID at
# a different slot than the usual index 1 — e.g. if a link-local was
# registered before the routed address was added, the routed GID lands
# at index 2.  A cluster-wide hardcoded NCCL_IB_GID_INDEX=1 then yields
#   "Call to ibv_query_gid failed" / "local GID N/A"
# on those nodes during RCCL init, deterministically breaking distributed
# training.  Auto-detect the index by scanning
# /sys/class/infiniband/<hca>/ports/<port>/gids/ on ACTIVE ports for a
# routable GID (not all-zero, not link-local fe80::).  Prefer RoCEv2
# over v1 when both exist at different slots — modern NCCL/RCCL defaults
# to v2 and either fails or silently falls back if pointed at a v1 slot
# on a v2-configured peer.  All ACTIVE ports on a given node are
# expected to share the same index; if they disagree, log a warning and
# leave the manual override to the operator.
#
# Filter details:
#   • Only fully-zero GIDs (`0000:0000:...:0000`) are treated as empty.
#     The loose `0000:0000:*` prefix match would spuriously skip valid
#     IPv4-mapped RoCEv2 GIDs (`0000:0000:...:ffff:<ipv4>`) on IPv4-
#     underlay clusters, which is why the full literal is used here.
#   • Link-local GIDs (`fe80:...`) are always skipped.
#   • Port state is read from `.../ports/<p>/state` (e.g. "4: ACTIVE");
#     if the file is unreadable we fail open and scan the port, so old
#     kernels keep pre-patch behavior.
#   • RoCE version is read from `.../gid_attrs/types/<i>` when present
#     (string contains "v1" / "v2" / "IB").  Old drivers that omit this
#     file degrade gracefully to the v1-pass fallback, which matches
#     the pre-patch "first non-zero non-fe80" selection.
#   • Slot indices are scanned numerically (0..15 covers every layout
#     observed on Mellanox CX-5/6/7 and Pensando AINIC; a glob would
#     lex-sort to 0,1,10,...,2,3,... and silently pick the wrong index
#     on ports with ≥10 populated slots).
#
# Manual override: set _env_NCCL_IB_GID_INDEX=N to skip auto-detect.
if [[ -z "${NCCL_IB_GID_INDEX:-}" && -d /sys/class/infiniband ]]; then
    _detected_gid_idx=""
    _gid_idx_conflict=0
    for _hca_dir in /sys/class/infiniband/*; do
        [[ -d "$_hca_dir" ]] || continue
        for _port_dir in "$_hca_dir"/ports/*; do
            [[ -d "$_port_dir/gids" ]] || continue
            # Skip non-ACTIVE ports — their GID tables contain stale or
            # never-populated entries that must not influence the choice.
            # Fail open if `state` is unreadable (unusual kernels): accept
            # the port so we preserve pre-patch behavior rather than
            # silently dropping detection.
            _port_state=$(cat "$_port_dir/state" 2>/dev/null)
            if [[ -n "$_port_state" ]]; then
                case "$_port_state" in
                    *ACTIVE*) ;;
                    *) continue ;;
                esac
            fi
            # Two-pass search: prefer RoCEv2 at any slot, else v1.
            _best_idx=""
            for _pass in v2 v1; do
                for _i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
                    _gid_file="$_port_dir/gids/$_i"
                    [[ -r "$_gid_file" ]] || continue
                    _gid=$(cat "$_gid_file" 2>/dev/null)
                    case "$_gid" in
                        "0000:0000:0000:0000:0000:0000:0000:0000"|fe80:*|"") continue ;;
                    esac
                    # Type-file aware filtering.  Missing file → accept
                    # only in v1-pass (fallback for pre-RoCEv2 drivers);
                    # reject in v2-pass so we don't mis-label unknown
                    # slots as v2.  This preserves the pre-patch
                    # "first non-zero non-fe80" behavior on old drivers
                    # (v1 pass with empty type matches) while on modern
                    # drivers the v2 pass correctly finds the v2 slot.
                    _type_file="$_port_dir/gid_attrs/types/$_i"
                    if [[ -r "$_type_file" ]]; then
                        _type=$(cat "$_type_file" 2>/dev/null)
                    else
                        _type=""
                    fi
                    case "$_pass" in
                        v2) [[ "$_type" == *"v2"* ]] || continue ;;
                        v1) [[ "$_type" == *"v1"* || "$_type" == *IB* || -z "$_type" ]] || continue ;;
                    esac
                    _best_idx="$_i"
                    break 2   # exit both $_pass and $_i loops for this port
                done
            done
            [[ -z "$_best_idx" ]] && continue
            if [[ -z "$_detected_gid_idx" ]]; then
                _detected_gid_idx="$_best_idx"
            elif [[ "$_detected_gid_idx" != "$_best_idx" ]]; then
                _gid_idx_conflict=1
            fi
        done
    done
    if [[ -n "$_detected_gid_idx" && "$_gid_idx_conflict" == "0" ]]; then
        export NCCL_IB_GID_INDEX="$_detected_gid_idx"
        echo "[INFO] $(hostname -s): NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX (auto-detected, RoCEv2-preferred)"
    elif [[ "$_gid_idx_conflict" == "1" ]]; then
        echo "[WARN] $(hostname -s): inconsistent routable GID indices across ACTIVE ports; leaving NCCL_IB_GID_INDEX unset (caller must override)" >&2
    fi
    unset _detected_gid_idx _gid_idx_conflict _hca_dir _port_dir \
          _port_state _best_idx _pass _gid_file _i _gid _type _type_file
fi

# NCCL_SOCKET_IFNAME: network interface for NCCL socket communication
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
    source "$_DETECT_NCCL_DIR/choose_nccl_socket_ifname.sh"
    if nccl_nic=$(choose_nccl_socket_ifname); then
        export NCCL_SOCKET_IFNAME="${nccl_nic}"
        echo "NCCL INFO $(hostname -s): NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
        if command -v ethtool &>/dev/null && ethtool -i "$NCCL_SOCKET_IFNAME" &>/dev/null; then
            echo "NIC_DRIVER_CHECK $(hostname -s) iface=$NCCL_SOCKET_IFNAME $(ethtool -i "$NCCL_SOCKET_IFNAME" | awk -F': *' '/^(driver|version|firmware-version):/{printf "%s=%s ", $1, $2}')"
        fi
    else
        if [[ "${NNODES:-1}" -gt 1 ]]; then
            echo "NCCL FATAL $(hostname -s): Failed to auto-detect NCCL_SOCKET_IFNAME; ABORTING..." >&2
            unset _DETECT_NCCL_DIR
            return 1
        else
            echo "NCCL WARN $(hostname -s): Could not auto-detect NCCL_SOCKET_IFNAME; leaving it unset" >&2
        fi
    fi
fi

unset _DETECT_NCCL_DIR
