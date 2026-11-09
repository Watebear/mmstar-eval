#!/usr/bin/env bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
task_root=/tmp/wushuo-910b-mmstar-20261106
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH="$task_root/pydeps:$task_root/lightllm-src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="$task_root/cache" HF_HOME="$task_root/cache/huggingface"
export TRITON_CACHE_DIR="$task_root/cache/triton" TMPDIR="$task_root/cache/tmp"
export PAGE_SIZE=128 TRITON_ALL_BLOCKS_PARALLEL=1 TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1 LOADWORKER=8
export TRITON_ASCEND_ARCH=Ascend910B2
export LIGHTLLM_NPU_DISABLE_CHUNK_PREFILL=1
