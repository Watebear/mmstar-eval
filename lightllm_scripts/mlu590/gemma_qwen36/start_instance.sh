#!/usr/bin/env bash
set -euo pipefail
source /data/wushuo_mlu/evaluations/mmbench_eval/serving_mlu/env.sh
instance=${1:?Usage: start_instance.sh gemma0|gemma1|qwen0|qwen1|qwen2 [--show-command]}
case "$instance" in
  gemma0) cards=0; port=18000; nccl_port=28000; model=gemma-3-4b-it; tp=1; visual_batch=1 ;;
  gemma1) cards=1; port=18001; nccl_port=28001; model=gemma-3-4b-it; tp=1; visual_batch=1 ;;
  qwen0) cards=2,3; port=18100; nccl_port=28100; model=Qwen3.6-27B; tp=2; visual_batch=2 ;;
  qwen1) cards=4,5; port=18101; nccl_port=28101; model=Qwen3.6-27B; tp=2; visual_batch=2 ;;
  qwen2) cards=6,7; port=18102; nccl_port=28102; model=Qwen3.6-27B; tp=2; visual_batch=2 ;;
  *) printf 'Unknown instance: %s\n' "$instance" >&2; exit 2 ;;
esac
export MLU_VISIBLE_DEVICES="$cards"
export LIGHTLLM_LOG_DIR="$SERVING_ROOT/logs/$instance"
command=(
  "$PYTHON" -u -m lightllm.server.api_server
  --hardware_platform mlu
  --model_dir "/data/zhangtaoshan/models/$model"
  --model_name "$model"
  --host 127.0.0.1 --port "$port"
  --nccl_host 127.0.0.1 --nccl_port "$nccl_port"
  --tp "$tp" --data_type bfloat16
  --llm_prefill_att_backend fa3
  --llm_decode_att_backend fa3
  --vit_att_backend sdpa
  --enable_triton_fallback
  --visual_infer_batch_size "$visual_batch"
  --visual_send_batch_size 1
  --running_max_req_size "${RUNNING_MAX_REQ_SIZE:-4}"
  --max_req_total_len "${MAX_REQ_TOTAL_LEN:-16384}"
  --max_total_token_num "${MAX_TOTAL_TOKEN_NUM:-32768}"
  --batch_max_tokens "${BATCH_MAX_TOKENS:-4096}"
  --chunked_prefill_size "${CHUNKED_PREFILL_SIZE:-1024}"
  --mem_fraction "${MEM_FRACTION:-0.70}"
  --embed_cache_storage_size 1
  --disable_cudagraph
)
if [[ "$model" == gemma-3-4b-it ]]; then
  command+=(--disable_dynamic_prompt_cache)
else
  command+=(--linear_att_ssm_data_type float32)
fi
if [[ ${2:-} == --show-command ]]; then
  printf 'MLU_VISIBLE_DEVICES=%q ' "$MLU_VISIBLE_DEVICES"
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
mkdir -p "$LIGHTLLM_LOG_DIR" "$TMPDIR" "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$SERVING_ROOT"
printf '%q ' "${command[@]}" > "$LIGHTLLM_LOG_DIR/launch_command.txt"
printf '\n' >> "$LIGHTLLM_LOG_DIR/launch_command.txt"
exec "${command[@]}"
