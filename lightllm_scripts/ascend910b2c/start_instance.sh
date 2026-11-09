#!/usr/bin/env bash
set -eo pipefail
task_root=/tmp/wushuo-910b-mmstar-20261106
source "$task_root/serving/source_env.sh"
task_model=${1:?model key required}
task_index=${2:?replica index required}
case "$task_model" in
  gemma) task_name=gemma-3-4b-it; task_tp=1; task_gpu="$task_index"; task_port=$((18300+task_index)); task_hccl=$((30300+task_index)); task_pool=32768; task_visual_dp=1;;
  qwen36) task_name=Qwen3.6-27B; task_tp=2; task_gpu="$((2*task_index)),$((2*task_index+1))"; task_port=$((18400+task_index)); task_hccl=$((30400+task_index)); task_pool=65536; task_visual_dp=2;;
  qwen3vl) task_name=Qwen3-VL-8B-Instruct; task_tp=1; task_gpu="$task_index"; task_port=$((18500+task_index)); task_hccl=$((30500+task_index)); task_pool=65536; task_visual_dp=1;;
  *) exit 2;;
esac
export ASCEND_RT_VISIBLE_DEVICES="$task_gpu"
export ASCEND_DEVICE_ID=0
task_args=(--model_dir "$task_root/models/$task_name" --model_name "$task_name"
  --hardware_platform ascend --tp "$task_tp" --data_type bfloat16
  --quant_type none --vit_quant_type none --llm_kv_type None
  --host 127.0.0.1 --port "$task_port" --nccl_port "$task_hccl"
  --max_req_total_len 20480 --max_total_token_num "$task_pool" --mem_fraction 0.75
  --running_max_req_size 4 --batch_max_tokens 4096 --chunked_prefill_size 2048
  --graph_max_batch_size 4 --graph_split_batch_size 4 --graph_max_len_in_batch 20480
  --llm_prefill_att_backend fa3 --llm_decode_att_backend fa3 --vit_att_backend sdpa
  --visual_tp 1 --visual_dp "$task_visual_dp" --visual_infer_batch_size "$task_visual_dp"
  --visual_send_batch_size 1
  --max_image_pixels 3686400 --max_image_token_count 8192 --embed_cache_storage_size 1
  --visual_gpu_ids 0)
if [[ "$task_visual_dp" == 2 ]]; then task_args+=(1); fi
if [[ "$task_model" == gemma ]]; then task_args+=(--disable_dynamic_prompt_cache); fi
if [[ "$task_model" == qwen36 ]]; then
  task_args+=(--reasoning_parser qwen3 --linear_att_cache_size 4 --linear_att_ssm_data_type float32
    --disable_flashinfer_allreduce --disable_symm_mem_allreduce)
fi
python3 - "$task_root" "$task_model" "$task_index" "${task_args[@]}" <<'PY'
import datetime,json,os,pathlib,sys
r=pathlib.Path(sys.argv[1]);model=sys.argv[2];index=int(sys.argv[3]);record=r/'serving/logs'/f'{model}{index}'/'command.json'
keys=['PATH','LD_LIBRARY_PATH','PYTHONPATH','ASCEND_RT_VISIBLE_DEVICES','ASCEND_DEVICE_ID','ASCEND_HOME_PATH','ASCEND_TOOLKIT_HOME','ASCEND_OPP_PATH','ASCEND_AICPU_PATH','ATB_HOME_PATH','PAGE_SIZE','TRITON_ASCEND_ARCH','LIGHTLLM_NPU_DISABLE_CHUNK_PREFILL','TRITON_ALL_BLOCKS_PARALLEL','TASK_QUEUE_ENABLE','HCCL_OP_EXPANSION_MODE','LIGHTLLM_TRITON_AUTOTUNE_LEVEL','LOADWORKER','OMP_NUM_THREADS','MKL_NUM_THREADS','TOKENIZERS_PARALLELISM','XDG_CACHE_HOME','HF_HOME','TRITON_CACHE_DIR','TMPDIR','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']
record.write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':model,'replica':index,'pid':os.getppid(),'argv':[sys.executable,'-m','lightllm.server.api_server',*sys.argv[4:]],'environment':{k:os.environ[k] for k in keys if k in os.environ},'source_commit':'0db819b929e713f9337723ff9dcd6a7a04924821','image_policy':'same as all prior actual formal services, source evidence previous_actual_image_policy.json'},indent=2)+'\n')
PY
cd "$task_root/lightllm-src"
exec python3 -m lightllm.server.api_server "${task_args[@]}"
