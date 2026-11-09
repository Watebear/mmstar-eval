#!/usr/bin/env bash
export MLU_VISIBLE_DEVICES=0
export PYTHONPATH=/tmp/wushuo-mlu-qwen3vl8b-20261106/serving/pydeps:/tmp/wushuo-mlu-qwen3vl8b-20261106/lightllm-runtime-0db819b
export NEUWARE_HOME=/usr/local/neuware
export LD_LIBRARY_PATH=$NEUWARE_HOME/lib64:$NEUWARE_HOME/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/wushuo-mlu-qwen3vl8b-20261106/cache/triton
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1
export LOADWORKER=8

python3 -m lightllm.server.api_server \
  --hardware_platform mlu \
  --model_dir /tmp/wushuo-mlu-mmbench-20261105/models/Qwen3-VL-8B-Instruct \
  --model_name Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 \
  --port 18200 \
  --nccl_host 127.0.0.1 \
  --nccl_port 30200 \
  --tp 1 \
  --data_type bfloat16 \
  --quant_type none \
  --llm_prefill_att_backend fa3 \
  --llm_decode_att_backend triton \
  --vit_att_backend sdpa \
  --enable_triton_fallback \
  --mem_fraction 0.70 \
  --max_req_total_len 20480 \
  --max_total_token_num 65536 \
  --running_max_req_size 4 \
  --batch_max_tokens 4096 \
  --chunked_prefill_size 1024 \
  --graph_max_batch_size 4 \
  --graph_split_batch_size 4 \
  --graph_max_len_in_batch 20480 \
  --visual_tp 1 \
  --visual_dp 1 \
  --visual_gpu_ids 0 \
  --visual_infer_batch_size 1 \
  --visual_send_batch_size 1 \
  --max_image_pixels 3686400 \
  --max_image_token_count 8192 \
  --embed_cache_storage_size 1
