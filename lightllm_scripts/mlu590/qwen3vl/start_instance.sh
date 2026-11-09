#!/usr/bin/env bash
set -euo pipefail
source /tmp/wushuo-mlu-qwen3vl8b-20261106/serving/env.sh
card=${1:?Usage: start_instance.sh CARD_INDEX [--show-command]}
[[ "$card" =~ ^[0-7]$ ]] || { printf 'Invalid card index\n' >&2; exit 2; }
port=$((18200+card)); nccl_port=$((30200+card))
export MLU_VISIBLE_DEVICES="$card"
export LIGHTLLM_LOG_DIR="$SERVING_ROOT/logs/qwen3vl8b-$card"
command=("$PYTHON" -u -m lightllm.server.api_server
  --hardware_platform mlu --model_dir "$MODEL_DIR" --model_name Qwen3-VL-8B-Instruct
  --host 127.0.0.1 --port "$port" --nccl_host 127.0.0.1 --nccl_port "$nccl_port"
  --tp 1 --data_type bfloat16 --quant_type none
  --llm_prefill_att_backend fa3 --llm_decode_att_backend triton --vit_att_backend sdpa --enable_triton_fallback
  --running_max_req_size 4 --max_req_total_len 20480 --max_total_token_num "${MAX_TOTAL_TOKEN_NUM:-65536}"
  --batch_max_tokens 4096 --chunked_prefill_size 1024 --mem_fraction 0.70
  --graph_max_batch_size 4 --graph_split_batch_size 4 --graph_max_len_in_batch 20480
  --visual_tp 1 --visual_dp 1 --visual_gpu_ids 0 --visual_infer_batch_size 1 --visual_send_batch_size 1
  --max_image_pixels 3686400 --max_image_token_count 8192 --embed_cache_storage_size 1)
if [[ ${2:-} == --show-command ]]; then printf 'MLU_VISIBLE_DEVICES=%q ' "$MLU_VISIBLE_DEVICES"; printf '%q ' "${command[@]}"; printf '\n'; exit 0; fi
[[ -f "$SERVING_ROOT/records/a100_pilot_pass.json" ]] || { printf 'A100 pilot gate has not passed\n' >&2; exit 3; }
"$PYTHON" -c 'import json,os; r=os.environ["SERVING_ROOT"]+"/records/"; assert json.load(open(r+"a100_pilot_pass.json"))["passed"]; assert json.load(open(r+"model_cache_verified.json"))["all_files_sha256_verified"]'
mkdir -p "$LIGHTLLM_LOG_DIR" "$TMPDIR" "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$SERVING_ROOT"
printf '%q ' "${command[@]}" > "$LIGHTLLM_LOG_DIR/launch_command.txt"; printf '\n' >> "$LIGHTLLM_LOG_DIR/launch_command.txt"
exec "${command[@]}"
