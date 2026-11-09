#!/usr/bin/env bash
export PYTHONPATH=/data/wushuo_metax/evaluations/mmbench_eval/serving_metax/pydeps:/data/wushuo_metax/evaluations/LightLLM
export CUDA_VISIBLE_DEVICES=0,1
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
export MACA_TORCH_COMPILE_CONF=1
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1
export LOADWORKER=16
export PAGE_SIZE=128

python3 -m lightllm.server.api_server \
  --hardware_platform maca \
  --trust_remote_code \
  --model_dir /data/zhangtaoshan/models/Qwen3.6-27B \
  --model_name Qwen3.6-27B \
  --tp 2 \
  --data_type bfloat16 \
  --quant_type none \
  --mem_fraction 0.80 \
  --max_req_total_len 20480 \
  --running_max_req_size 8 \
  --batch_max_tokens 4096 \
  --chunked_prefill_size 2048 \
  --graph_max_batch_size 8 \
  --graph_max_len_in_batch 20480 \
  --visual_infer_batch_size 2 \
  --reasoning_parser qwen3 \
  --linear_att_ssm_data_type bfloat16 \
  --llm_prefill_att_backend fa3 \
  --llm_decode_att_backend fa3 \
  --vit_att_backend sdpa \
  --host 127.0.0.1 \
  --port 18100
