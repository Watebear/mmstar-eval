#!/bin/bash
set -euo pipefail
base=/var/tmp/wushuo-mmbench-20261105/serving
name=${1:?instance name required}
if [[ -f "$base/pids/$name.pid" ]] && kill -0 "$(cat "$base/pids/$name.pid")" 2>/dev/null; then
 state=$(ps -o stat= -p "$(cat "$base/pids/$name.pid")" || true)
 if [[ "$state" != Z* ]]; then echo "Instance already running: $name"; exit 1; fi
fi
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="$base/logs/$name.$stamp.log"
nohup setsid /opt/conda/bin/python "$base/launch.py" "$name" >"$log" 2>&1 </dev/null &
echo "$!" > "$base/pids/$name.launch.pid"
printf '%s\n' "$log" > "$base/records/$name.latest_log"
echo "$name launch_pid=$! log=$log"
