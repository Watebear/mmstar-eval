#!/usr/bin/env bash
export PYTHONPATH=/tmp/wushuo-metax-qwen3vl8b-20261106/pydeps:/tmp/wushuo-metax-qwen3vl8b-20261106/LightLLM
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export MACA_GRAPH_LAUNCH_MODE=1
export MACA_DIRECT_DISPATCH=1
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1
export LOADWORKER=16
export PAGE_SIZE=128

python3 -m lightllm.server.api_server \
  --hardware_platform maca \
  --model_dir /tmp/wushuo-metax-qwen3vl8b-20261106/models/Qwen3-VL-8B-Instruct \
  --model_name Qwen3-VL-8B-Instruct \
  --tp 1 \
  --data_type bfloat16 \
  --quant_type none \
  --mem_fraction 0.75 \
  --max_req_total_len 20480 \
  --max_total_token_num 32768 \
  --running_max_req_size 4 \
  --batch_max_tokens 4096 \
  --chunked_prefill_size 2048 \
  --graph_max_batch_size 4 \
  --graph_split_batch_size 4 \
  --graph_max_len_in_batch 20480 \
  --visual_tp 1 --visual_dp 1 \
  --visual_gpu_ids 0 --visual_infer_batch_size 1 \
  --visual_send_batch_size 1 \
  --llm_prefill_att_backend fa3 \
  --llm_decode_att_backend fa3 \
  --vit_att_backend sdpa \
  --max_image_pixels 3686400 \
  --max_image_token_count 8192 \
  --embed_cache_storage_size 1 \
  --host 127.0.0.1 \
  --port 18200 \
  --nccl_port 30200
