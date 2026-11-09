#!/usr/bin/env bash
export MLU_VISIBLE_DEVICES=2,3
export PYTHONPATH=/data/wushuo_mlu/evaluations/mmbench_eval/serving_mlu/pydeps:/data/wushuo_mlu/evaluations/LightLLM
export NEUWARE_HOME=/usr/local/neuware
export LD_LIBRARY_PATH=$NEUWARE_HOME/lib64:$NEUWARE_HOME/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/wushuo-mlu-mmbench-20261105/cache/triton
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1
export LOADWORKER=16

python3 -m lightllm.server.api_server \
  --hardware_platform mlu \
  --model_dir /data/zhangtaoshan/models/Qwen3.6-27B \
  --model_name Qwen3.6-27B \
  --tp 2 \
  --data_type bfloat16 \
  --llm_prefill_att_backend fa3 \
  --llm_decode_att_backend fa3 \
  --vit_att_backend sdpa \
  --enable_triton_fallback \
  --mem_fraction 0.70 \
  --max_req_total_len 16384 \
  --max_total_token_num 32768 \
  --batch_max_tokens 4096 \
  --chunked_prefill_size 1024 \
  --running_max_req_size 4 \
  --visual_infer_batch_size 2 \
  --visual_send_batch_size 1 \
  --embed_cache_storage_size 1 \
  --disable_cudagraph \
  --linear_att_ssm_data_type float32 \
  --reasoning_parser qwen3 \
  --host 127.0.0.1 \
  --port 18100 \
  --nccl_host 127.0.0.1 \
  --nccl_port 28100
