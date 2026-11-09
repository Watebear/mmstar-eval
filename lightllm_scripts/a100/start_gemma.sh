#!/usr/bin/env bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/tmp/mmbench_a100_lightllm_0db819b929e7
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=0
export LOADWORKER=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/var/tmp/wushuo-mmbench-20261105/cache/triton

python3 -m lightllm.server.api_server \
  --model_dir /var/tmp/wushuo-mmbench-20261105/models/gemma-3-4b-it \
  --model_name gemma-3-4b-it \
  --tp 1 \
  --data_type bfloat16 \
  --max_req_total_len 16384 \
  --max_total_token_num 17408 \
  --running_max_req_size 4 \
  --chunked_prefill_size 2048 \
  --batch_max_tokens 4096 \
  --graph_max_batch_size 4 \
  --graph_split_batch_size 4 \
  --mem_fraction 0.82 \
  --visual_tp 1 \
  --visual_dp 1 \
  --visual_infer_batch_size 1 \
  --visual_send_batch_size 1 \
  --visual_gpu_ids 0 \
  --llm_prefill_att_backend triton \
  --llm_decode_att_backend triton \
  --vit_att_backend sdpa \
  --disable_dynamic_prompt_cache \
  --host 127.0.0.1 \
  --port 18000
