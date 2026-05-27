#!/bin/bash
# detect_ip.sh — Detect the public or cluster-private IP address of this host.
#
# Usage:
#   source utils/detect_ip.sh
#   ip="$(detect_ip)"            # public IP (for SSH-tunnel hints, banners)
#   ip="$(detect_cluster_ip)"    # cluster-private RFC1918 IP — use for binding
#                                # services that should never face the internet
#                                # (Ray GCS, JAX coordinator, etc.).

# Public IP — for human-facing tunnel hints, never for service binds.
# Strategy:
#   1. curl external services (ifconfig.me, icanhazip.com)
#   2. Validate the response looks like an IP (not a proxy error page)
#   3. Fall back to hostname -I (local/private IP)
#   4. Last resort: "ip-unknown"
detect_ip() {
    local ip
    ip=$(curl -s --connect-timeout 2 ifconfig.me 2>/dev/null || \
         curl -s --connect-timeout 2 icanhazip.com 2>/dev/null || \
         echo "")
    # Validate: curl can succeed (exit 0) yet return a proxy error page
    # instead of an IP.  Accept only IPv4/IPv6-shaped strings.
    if ! [[ "$ip" =~ ^[0-9a-fA-F.:]+$ ]]; then
        ip=$(hostname -I 2>/dev/null | awk '{print $1}')
        [[ -z "$ip" ]] && ip="ip-unknown"
    fi
    echo "$ip"
}

# Cluster-private IP — for binding services that should NOT be reachable
# from the internet.  Ray's GCS port and the JAX coordinator port are
# unauthenticated cross-node services; if bound to 0.0.0.0 on a host with
# a public IP, internet scanners can join the Ray cluster as raylets
# within seconds of head startup and execute arbitrary code in the actor
# pool.  Binding to the cluster-private IP closes that surface — workers
# reach the head over the private subnet only.
#
# Strategy: enumerate IPv4 addresses, prefer RFC1918 (10/8, 172.16/12,
# 192.168/16), exclude well-known container / overlay subnets (Docker
# 172.17/16, Flannel 10.42/16, k3s service cluster 10.43/16).  Falls back
# to detect_ip if no private IP is present (single-node / no-private setup).
detect_cluster_ip() {
    local ip
    ip=$(ip -4 -o addr show 2>/dev/null \
        | awk '{print $4}' | cut -d/ -f1 \
        | grep -E '^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)' \
        | grep -vE '^(172\.17\.|10\.42\.|10\.43\.)' \
        | head -1)
    if [[ -z "$ip" ]]; then
        ip="$(detect_ip)"
    fi
    echo "$ip"
}
