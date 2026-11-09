#!/usr/bin/env bash
export RUN_ROOT=/tmp/wushuo-mlu-qwen3vl8b-20261106
export SERVING_ROOT="$RUN_ROOT/serving"
export LIGHTLLM_REPO="$RUN_ROOT/lightllm-runtime-0db819b"
export MODEL_DIR=/tmp/wushuo-mlu-mmbench-20261105/models/Qwen3-VL-8B-Instruct
export PYTHON=/torch/venv3/pytorch_infer/bin/python
export PATH="/torch/venv3/pytorch_infer/bin:$PATH"
export PYTHONPATH="$SERVING_ROOT/pydeps:$LIGHTLLM_REPO"
export PYTHONDONTWRITEBYTECODE=1
export NEUWARE_HOME=/usr/local/neuware
export LD_LIBRARY_PATH="$NEUWARE_HOME/lib64:$NEUWARE_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XDG_CACHE_HOME="$RUN_ROOT/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TMPDIR="$XDG_CACHE_HOME/tmp"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export LIGHTLLM_LOG_LEVEL=info
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
