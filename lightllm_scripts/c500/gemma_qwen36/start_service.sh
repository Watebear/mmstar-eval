#!/usr/bin/env bash
set -euo pipefail

model=$1
port=$2
devices=$3
base=/data/wushuo_metax/evaluations/mmbench_eval/serving_metax
repo=/data/wushuo_metax/evaluations/LightLLM
mkdir -p "$base/logs" "$base/pids" "$base/commands"
exec >> "$base/logs/$model-$port.log" 2>&1
printf '%s\n' "$$" > "$base/pids/$model-$port.pid"

export PYTHONPATH="$base/pydeps:$repo${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$devices"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1 MACA_GRAPH_LAUNCH_MODE=1 MACA_DIRECT_DISPATCH=1
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 LOADWORKER=16

common=(--hardware_platform maca --trust_remote_code --data_type bfloat16
    --quant_type none --host 127.0.0.1 --port "$port"
    --max_req_total_len 20480 --running_max_req_size 8
    --batch_max_tokens 4096 --chunked_prefill_size 2048
    --graph_max_batch_size 8 --graph_max_len_in_batch 20480
    --vit_att_backend sdpa)
case "$model" in
    gemma)
        export MACA_TORCH_COMPILE_CONF='triton.multi_kernel:1'
        args=(--model_dir /data/zhangtaoshan/models/gemma-3-4b-it
            --model_name gemma-3-4b-it --tp 1 --mem_fraction 0.75
            --visual_infer_batch_size 1 --disable_dynamic_prompt_cache
            --llm_prefill_att_backend flashinfer --llm_decode_att_backend flashinfer)
        ;;
    qwen)
        export MACA_TORCH_COMPILE_CONF=1 PAGE_SIZE=128
        args=(--model_dir /data/zhangtaoshan/models/Qwen3.6-27B
            --model_name Qwen3.6-27B --tp 2 --mem_fraction 0.80
            --visual_infer_batch_size 2 --reasoning_parser qwen3
            --linear_att_ssm_data_type bfloat16
            --llm_prefill_att_backend fa3 --llm_decode_att_backend fa3)
        ;;
    *) printf 'Unknown model: %s\n' "$model"; exit 2 ;;
esac

python - "$base/commands/$model-$port.json" "$model" "$port" "$devices" "${common[@]}" "${args[@]}" <<'PY'
import datetime, json, os, sys
from pathlib import Path
keys = ('PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'MACA_SMALL_PAGESIZE_ENABLE',
        'TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP',
        'TRITON_ENABLE_MACA_CHAIN_DOT_OPT', 'PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM',
        'MACA_QUEUE_SCHEDULE_POLICY', 'MACA_GRAPH_LAUNCH_MODE', 'MACA_DIRECT_DISPATCH',
        'LIGHTLLM_TRITON_AUTOTUNE_LEVEL', 'LOADWORKER', 'MACA_TORCH_COMPILE_CONF',
        'PAGE_SIZE', 'OMP_NUM_THREADS', 'TOKENIZERS_PARALLELISM')
record = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'model': sys.argv[2], 'port': int(sys.argv[3]), 'devices': sys.argv[4],
          'command': [sys.executable, '-m', 'lightllm.server.api_server', *sys.argv[5:]],
          'environment': {k: os.environ[k] for k in keys if k in os.environ}}
Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + '\n')
PY
cd "$repo"
exec python -m lightllm.server.api_server "${common[@]}" "${args[@]}"
