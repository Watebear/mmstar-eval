#!/usr/bin/env bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_DEVICE_ID=0
export PYTHONPATH=/tmp/wushuo-910b-mmstar-20261106/pydeps:/tmp/wushuo-910b-mmstar-20261106/lightllm-src
export PAGE_SIZE=128
export TRITON_ALL_BLOCKS_PARALLEL=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1
export LOADWORKER=8

python3 -m lightllm.server.api_server \
  --model_dir /tmp/wushuo-910b-mmstar-20261106/models/gemma-3-4b-it \
  --model_name gemma-3-4b-it \
  --hardware_platform ascend \
  --tp 1 \
  --data_type bfloat16 \
  --quant_type none \
  --vit_quant_type none \
  --llm_kv_type None \
  --host 127.0.0.1 \
  --port 18300 \
  --nccl_port 30300 \
  --max_req_total_len 20480 \
  --max_total_token_num 32768 \
  --mem_fraction 0.75 \
  --running_max_req_size 4 \
  --batch_max_tokens 4096 \
  --chunked_prefill_size 2048 \
  --graph_max_batch_size 4 \
  --graph_split_batch_size 4 \
  --graph_max_len_in_batch 20480 \
  --llm_prefill_att_backend fa3 \
  --llm_decode_att_backend fa3 \
  --vit_att_backend sdpa \
  --visual_tp 1 --visual_dp 1 \
  --visual_infer_batch_size 1 --visual_send_batch_size 1 \
  --max_image_pixels 3686400 --max_image_token_count 8192 \
  --embed_cache_storage_size 1 --visual_gpu_ids 0 \
  --disable_dynamic_prompt_cache
