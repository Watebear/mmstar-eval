#!/usr/bin/env bash
export PYTHONPATH=/data/wushuo_metax/evaluations/mmbench_eval/serving_metax/pydeps:/data/wushuo_metax/evaluations/LightLLM
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MACA_SMALL_PAGESIZE_ENABLE=1
export TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP=1
export TRITON_ENABLE_MACA_CHAIN_DOT_OPT=1
export PYTORCH_ENABLE_PG_HIGH_PRIORITY_STREAM=1
export MACA_QUEUE_SCHEDULE_POLICY=1
export MACA_GRAPH_LAUNCH_MODE=1
export MACA_DIRECT_DISPATCH=1
export MACA_TORCH_COMPILE_CONF='triton.multi_kernel:1'
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1
export LOADWORKER=16

python3 -m lightllm.server.api_server \
  --hardware_platform maca \
  --trust_remote_code \
  --model_dir /data/zhangtaoshan/models/gemma-3-4b-it \
  --model_name gemma-3-4b-it \
  --tp 1 \
  --data_type bfloat16 \
  --quant_type none \
  --mem_fraction 0.75 \
  --max_req_total_len 20480 \
  --running_max_req_size 8 \
  --batch_max_tokens 4096 \
  --chunked_prefill_size 2048 \
  --graph_max_batch_size 8 \
  --graph_max_len_in_batch 20480 \
  --visual_infer_batch_size 1 \
  --llm_prefill_att_backend flashinfer \
  --llm_decode_att_backend flashinfer \
  --vit_att_backend sdpa \
  --disable_dynamic_prompt_cache \
  --host 127.0.0.1 \
  --port 19001
