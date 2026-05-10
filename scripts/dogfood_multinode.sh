#!/bin/bash
set -euo pipefail
ROOT_DIR="/tmp/prsm_multinode"
LOG_DIR="$ROOT_DIR/logs"
rm -rf "$ROOT_DIR"
mkdir -p "$LOG_DIR"

start_node() {
    local NODE=$1; local API=$2; local P2P=$3; local BOOTSTRAP=${4:-}
    local NODE_HOME="$ROOT_DIR/$NODE"
    mkdir -p "$NODE_HOME/.prsm"
    local LOG="$LOG_DIR/$NODE.log"
    local BS=""
    [ -n "$BOOTSTRAP" ] && BS="--bootstrap $BOOTSTRAP"
    HOME="$NODE_HOME" PRSM_NETWORK=testnet \
        python3 -m prsm.cli node start --no-dashboard \
        --api-port "$API" --p2p-port "$P2P" $BS > "$LOG" 2>&1 &
    echo $!
}

wait_api() {
    for i in {1..30}; do
        curl -sS -m 1 "http://127.0.0.1:$1/health" > /dev/null 2>&1 && return 0
        sleep 1
    done
    return 1
}

extract_peer() {
    for i in {1..30}; do
        local PID=$(grep -oE 'peerID=[A-Za-z0-9]+' "$1" 2>/dev/null | head -1 | cut -d= -f2)
        [ -n "$PID" ] && { echo "$PID"; return 0; }
        sleep 1
    done
    return 1
}

echo "─── Multi-node dogfood ─────────────────────"
echo "[A] Starting seed (no bootstrap)..."
A_PID=$(start_node a 8000 9001 "")
echo "    pid=$A_PID"
A_PEER=$(extract_peer "$LOG_DIR/a.log") || { echo "FAIL: peer extract"; tail -10 "$LOG_DIR/a.log"; pkill -f "prsm.cli node start" 2>/dev/null; exit 1; }
echo "    peer_id: $A_PEER"
wait_api 8000 || { echo "FAIL: A api"; exit 1; }
echo "    API alive on :8000"
A_MULTIADDR="/ip4/127.0.0.1/udp/9001/quic-v1/p2p/$A_PEER"
echo
echo "Bootstrap: $A_MULTIADDR"
echo

echo "[B] Starting bootstrap→A..."
B_PID=$(start_node b 8010 9011 "$A_MULTIADDR")
echo "    pid=$B_PID"
echo "[C] Starting bootstrap→A..."
C_PID=$(start_node c 8020 9021 "$A_MULTIADDR")
echo "    pid=$C_PID"

wait_api 8010 && echo "    B api alive" || echo "    B api FAIL"
wait_api 8020 && echo "    C api alive" || echo "    C api FAIL"
echo
echo "Waiting 10s for libp2p connection..."
sleep 10

echo
echo "─── Peer counts ────────────────────────────"
for E in "a:8000" "b:8010" "c:8020"; do
    NODE=${E%:*}; PORT=${E#*:}
    echo "  Node $NODE (:$PORT)"
    curl -sS -m 2 "http://127.0.0.1:$PORT/status" 2>&1 | python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    p = d.get('peers', {})
    print(f'    connected={p.get(\"connected\")}  known={p.get(\"known\")}')
    bs = p.get('bootstrap', {})
    print(f'    bootstrap attempted={bs.get(\"attempted\")} connected={bs.get(\"connected\")} degraded={bs.get(\"degraded\")}')
except Exception as e:
    print(f'    parse error: {e}')
"
done
