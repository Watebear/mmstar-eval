#!/usr/bin/env bash
set -eo pipefail
source /tmp/wushuo-910b-mmstar-20261106/serving/source_env.sh
cd /tmp/wushuo-910b-mmstar-20261106/serving
exec python3 autonomous.py
