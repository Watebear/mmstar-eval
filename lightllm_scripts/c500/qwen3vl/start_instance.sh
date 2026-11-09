#!/usr/bin/env bash
set -euo pipefail
task_root=/tmp/wushuo-metax-qwen3vl8b-20261106
task_instance=${1:?GPU index 0-7 required}
[[ "$task_instance" =~ ^[0-7]$ ]]
task_port=$((18200 + task_instance))
task_nccl_port=$((30200 + task_instance))
/usr/bin/python - "$task_root" <<'PY'
import json, pathlib, sys
r = pathlib.Path(sys.argv[1])
assert json.loads((r/'serving/records/a100_pilot_pass.json').read_text())['passed']
assert json.loads((r/'serving/records/model_cache_verified.json').read_text())['destination_sha256_verified']
PY
export PYTHONPATH="$task_root/pydeps:$task_root/LightLLM"
export CUDA_VISIBLE_DEVICES="$task_instance" CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$task_root/cache" HF_HOME="$task_root/cache/huggingface"
export TRITON_CACHE_DIR="$task_root/cache/triton" TMPDIR="$task_root/cache/tmp"
export MACA_SMALL_PAGESIZE_ENABLE=1 TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1 PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1 MACA_GRAPH_LAUNCH_MODE=1 MACA_DIRECT_DISPATCH=1
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 LOADWORKER=16
export PAGE_SIZE=128
unset MACA_TORCH_COMPILE_CONF
task_args=(--model_dir "$task_root/models/Qwen3-VL-8B-Instruct"
  --model_name Qwen3-VL-8B-Instruct --hardware_platform maca --tp 1
  --data_type bfloat16 --quant_type none --vit_quant_type none --llm_kv_type None
  --host 127.0.0.1 --port "$task_port" --nccl_port "$task_nccl_port"
  --max_req_total_len 20480 --max_total_token_num 32768 --mem_fraction 0.75
  --running_max_req_size 4 --batch_max_tokens 4096 --chunked_prefill_size 2048
  --graph_max_batch_size 4 --graph_split_batch_size 4 --graph_max_len_in_batch 20480
  --llm_prefill_att_backend fa3 --llm_decode_att_backend fa3 --vit_att_backend sdpa
  --visual_tp 1 --visual_dp 1 --visual_gpu_ids 0
  --visual_infer_batch_size 1 --visual_send_batch_size 1
  --max_image_pixels 3686400 --max_image_token_count 8192 --embed_cache_storage_size 1)
mkdir -p "$task_root/serving/logs/qwen3vl$task_instance"
/usr/bin/python - "$task_root" "$task_instance" "${task_args[@]}" <<'PY'
import datetime, json, os, pathlib, sys
r = pathlib.Path(sys.argv[1]); index = sys.argv[2]
keys = ['PATH','LD_LIBRARY_PATH','LIBRARY_PATH','MACA_PATH','CUDA_PATH','PYTHONPATH',
        'CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','OMP_NUM_THREADS','MKL_NUM_THREADS',
        'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TOKENIZERS_PARALLELISM','XDG_CACHE_HOME',
        'HF_HOME','TRITON_CACHE_DIR','TMPDIR','MACA_SMALL_PAGESIZE_ENABLE',
        'TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP','TRITON_ENABLE_MACA_CHAIN_DOT_OPT',
        'PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM','MACA_QUEUE_SCHEDULE_POLICY',
        'MACA_GRAPH_LAUNCH_MODE','MACA_DIRECT_DISPATCH','LIGHTLLM_TRITON_AUTOTUNE_LEVEL','LOADWORKER','PAGE_SIZE']
record = {'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'gpu':int(index),'instance':'qwen3vl'+index,'pid':os.getppid(),
          'argv':[sys.executable,'-m','lightllm.server.api_server',*sys.argv[3:]],
          'environment':{k:os.environ[k] for k in keys if k in os.environ},
          'unset_environment':['MACA_TORCH_COMPILE_CONF'],
          'source_snapshot':str(r/'LightLLM')}
(r/'serving/logs'/('qwen3vl'+index)/'command.json').write_text(json.dumps(record,indent=2)+'\n')
PY
cd "$task_root/LightLLM"
exec /usr/bin/python -m lightllm.server.api_server "${task_args[@]}"
