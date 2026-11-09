#!/usr/bin/env bash
SERVING_ROOT=/data/wushuo_mlu/evaluations/mmbench_eval/serving_mlu
LIGHTLLM_REPO=/data/wushuo_mlu/evaluations/LightLLM
PYTHON=/torch/venv3/pytorch_infer/bin/python
export SERVING_ROOT LIGHTLLM_REPO PYTHON
export PYTHONPATH="$SERVING_ROOT/pydeps:$LIGHTLLM_REPO"
export PYTHONDONTWRITEBYTECODE=1
export NEUWARE_HOME=/usr/local/neuware
export LD_LIBRARY_PATH="$NEUWARE_HOME/lib64:$NEUWARE_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export XDG_CACHE_HOME=/tmp/wushuo-mlu-mmbench-20261105/cache
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export TMPDIR="$XDG_CACHE_HOME/tmp"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export LIGHTLLM_LOG_LEVEL=info
