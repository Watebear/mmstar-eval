#!/usr/bin/env bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/var/tmp/wushuo-qwen3vl8b-20261106/LightLLM
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=0
export LOADWORKER=8
export XDG_CACHE_HOME=/var/tmp/wushuo-qwen3vl8b-20261106/cache
export HF_HOME=/var/tmp/wushuo-qwen3vl8b-20261106/cache/huggingface
export TRITON_CACHE_DIR=/var/tmp/wushuo-qwen3vl8b-20261106/cache/triton

python3 -m lightllm.server.api_server \
  --hardware_platform cuda \
  --model_dir /var/tmp/wushuo-qwen3vl8b-20261106/models/Qwen3-VL-8B-Instruct \
  --model_name Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 \
  --port 18200 \
  --nccl_port 30200 \
  --tp 1 \
  --trust_remote_code \
  --data_type bfloat16 \
  --quant_type none \
  --mem_fraction 0.80 \
  --max_req_total_len 20480 \
  --max_total_token_num 24576 \
  --running_max_req_size 4 \
  --chunked_prefill_size 2048 \
  --batch_max_tokens 4096 \
  --graph_max_batch_size 4 \
  --graph_split_batch_size 4 \
  --graph_max_len_in_batch 20480 \
  --visual_tp 1 \
  --visual_dp 1 \
  --visual_gpu_ids 0 \
  --visual_infer_batch_size 1 \
  --visual_send_batch_size 1 \
  --vit_att_backend sdpa \
  --llm_prefill_att_backend triton \
  --llm_decode_att_backend triton \
  --max_image_pixels 3686400 \
  --max_image_token_count 8192 \
  --embed_cache_storage_size 1 \
  --disable_flashinfer_allreduce \
  --disable_symm_mem_allreduce
