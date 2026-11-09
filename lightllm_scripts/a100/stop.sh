#!/bin/bash
set -euo pipefail
base=/var/tmp/wushuo-mmbench-20261105/serving
name=${1:?instance required}
pid=$(cat "$base/pids/$name.pid")
if ! kill -0 "$pid" 2>/dev/null; then exit 0; fi
args=$(tr '\0' ' ' < "/proc/$pid/cmdline")
if [[ -z "$args" ]]; then exit 0; fi
[[ "$args" == *lightllm.server.api_server* ]] || { echo "PID does not match LightLLM: $pid"; exit 1; }
[[ "$(ps -o pgid= -p "$pid" | tr -d ' ')" == "$pid" ]] || { echo "PID is not its process group leader: $pid"; exit 1; }
kill -TERM -- "-$pid"
for _ in {1..10}; do
 if ! kill -0 "$pid" 2>/dev/null || [[ "$(ps -o stat= -p "$pid" || true)" == Z* ]]; then exit 0; fi
 sleep 1
done
kill -KILL -- "-$pid" 2>/dev/null || true
