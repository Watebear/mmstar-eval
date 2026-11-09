# LightLLM 启动脚本归档

这里保存本轮 MMStar 评测中实际使用的 LightLLM 启动脚本快照。脚本中的路径、环境变量、端口、GPU 分配和模型参数均保持原始平台配置，脚本依赖各机器上的模型缓存、厂商运行时和 LightLLM 运行目录。

下面的命令块采用原始平台 Word 配置中的 Shell 格式。每个平台目录同时保留了实际运行的入口脚本，命令块用于快速核对完整参数。正式评测的请求采样参数统一放在仓库根目录的 `sampling_params.json`。

各平台使用的 LightLLM 基线仓库均为：

- 仓库：<https://github.com/zhangts20/LightLLM>
- 分支：`refactor_platform_dev_v1.2.0_dspark`
- 基线 commit：`0db819b929e713f9337723ff9dcd6a7a04924821`

## 平台目录

### A100

目录：`a100/`

- 原始 checkout：`/data/wushuo/evaluations/LightLLM`
- 评测项目：`/data/wushuo/evaluations/mmbench_eval`
- 最终运行根：`/var/tmp/wushuo-mmbench-20261105/serving`
- 入口：`start.sh INSTANCE`
- 停止：`stop.sh INSTANCE`
- `launch.py --dry-run` 可查看 `gemma0`、`gemma1`、`qwen0` 到 `qwen3` 的最终命令和环境
- `launch_templates.json` 保存每个副本的实际 argv 和环境
- `start_gemma.sh`、`start_qwen36.sh`、`start_qwen3vl.sh` 是按 Word 示例整理的单副本直读式命令

示例命令：

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/tmp/mmbench_a100_lightllm_0db819b929e7
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=0
export LOADWORKER=8

python -m lightllm.server.api_server \
  --model_dir /var/tmp/wushuo-mmbench-20261105/models/gemma-3-4b-it \
  --model_name gemma-3-4b-it \
  --host 127.0.0.1 --port 18000 \
  --tp 1 --data_type bfloat16 \
  --max_req_total_len 16384 --max_total_token_num 17408 \
  --running_max_req_size 4 --chunked_prefill_size 2048 \
  --batch_max_tokens 4096 --graph_max_batch_size 4 \
  --graph_split_batch_size 4 --mem_fraction 0.82 \
  --visual_tp 1 --visual_dp 1 --visual_infer_batch_size 1 \
  --visual_send_batch_size 1 --visual_gpu_ids 0 \
  --llm_prefill_att_backend triton --llm_decode_att_backend triton \
  --vit_att_backend sdpa --disable_dynamic_prompt_cache
```

Qwen3.6 使用 `qwen0` 到 `qwen3` 的 TP2 副本。完整副本命令以 `launch_templates.json` 为准：

```bash
bash start.sh qwen0
python launch.py qwen0 --dry-run
```

### MLU590

目录：`mlu590/`

- 原始 checkout：`/data/wushuo_mlu/evaluations/LightLLM`
- 评测项目：`/data/wushuo_mlu/evaluations/mmbench_eval`
- Gemma 和 Qwen3.6 入口：`gemma_qwen36/start_instance.sh`
- Qwen3-VL-8B-Instruct 入口：`qwen3vl/start_instance.sh`
- Qwen3-VL 最终运行根：`/tmp/wushuo-mlu-qwen3vl8b-20261106`
- `start_gemma.sh`、`start_qwen36.sh`、`start_qwen3vl.sh` 是单副本直读式命令

示例命令：

```bash
export MLU_VISIBLE_DEVICES=2,3
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1
export LOADWORKER=16

python -m lightllm.server.api_server \
  --hardware_platform mlu \
  --model_dir /data/zhangtaoshan/models/Qwen3.6-27B \
  --model_name Qwen3.6-27B --tp 2 --data_type bfloat16 \
  --nccl_host 127.0.0.1 --nccl_port 28100 \
  --llm_prefill_att_backend fa3 --llm_decode_att_backend fa3 \
  --vit_att_backend sdpa --enable_triton_fallback \
  --visual_infer_batch_size 2 --visual_send_batch_size 1 \
  --mem_fraction 0.70 --max_req_total_len 16384 \
  --max_total_token_num 32768 --batch_max_tokens 4096 \
  --chunked_prefill_size 1024 --running_max_req_size 4 \
  --embed_cache_storage_size 1 --reasoning_parser qwen3 \
  --linear_att_ssm_data_type float32 --disable_cudagraph \
  --host 127.0.0.1 --port 18100
```

实际正式入口仍使用 `gemma_qwen36/start_instance.sh` 或 `qwen3vl/start_instance.sh`，因为它们还负责环境、通信端口、日志目录和平台运行时。

### Metax C500

目录：`c500/`

- C500 与 Metax 共用存储，原始 checkout：`/data/wushuo_metax/evaluations/LightLLM`
- 评测项目：`/data/wushuo_metax/evaluations/mmbench_eval`
- Gemma 和 Qwen3.6 入口：`gemma_qwen36/start_service.sh MODEL PORT DEVICES`
- Qwen3-VL-8B-Instruct 入口：`qwen3vl/start_instance.sh GPU_INDEX`
- Qwen3-VL 最终运行根：`/tmp/wushuo-metax-qwen3vl8b-20261106`
- `start_gemma.sh`、`start_qwen36.sh`、`start_qwen3vl.sh` 是单副本直读式命令

示例命令：

```bash
export CUDA_VISIBLE_DEVICES=0,1
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

python -m lightllm.server.api_server \
  --hardware_platform maca --trust_remote_code \
  --model_dir /data/zhangtaoshan/models/Qwen3.6-27B \
  --model_name Qwen3.6-27B --tp 2 --data_type bfloat16 \
  --quant_type none --mem_fraction 0.80 \
  --max_req_total_len 20480 --running_max_req_size 8 \
  --batch_max_tokens 4096 --chunked_prefill_size 2048 \
  --graph_max_batch_size 8 --graph_max_len_in_batch 20480 \
  --visual_infer_batch_size 2 \
  --llm_prefill_att_backend fa3 --llm_decode_att_backend fa3 \
  --vit_att_backend sdpa --reasoning_parser qwen3 \
  --linear_att_ssm_data_type bfloat16 \
  --host 127.0.0.1 --port 18100
```

实际正式入口仍使用 `gemma_qwen36/start_service.sh MODEL PORT DEVICES` 或 `qwen3vl/start_instance.sh GPU_INDEX`。

### Ascend 910B2C

目录：`ascend910b2c/`

- 基线 checkout：`/data/zhangtaoshan/LightLLM`
- 最终实际运行源码：`/tmp/wushuo-910b-mmstar-20261106/lightllm-src`
- 最终运行源码由 0db819b 基线和 910B 平台补丁重建得到
- 入口：`start_instance.sh MODEL_KEY REPLICA_INDEX`
- 控制器：`launch_controller.sh`
- 环境：`source_env.sh`
- 重建源码：`rebuild_source.py`
- `start_gemma.sh`、`start_qwen36.sh`、`start_qwen3vl.sh` 是单副本直读式命令

示例命令：

```bash
export PAGE_SIZE=128
export TRITON_ALL_BLOCKS_PARALLEL=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export LIGHTLLM_TRITON_AUTOTUNE_LEVEL=1
export ASCEND_RT_VISIBLE_DEVICES=0,1
export ASCEND_DEVICE_ID=0

python -m lightllm.server.api_server \
  --model_dir /tmp/wushuo-910b-mmstar-20261106/models/Qwen3.6-27B \
  --model_name Qwen3.6-27B --tp 2 --data_type bfloat16 \
  --hardware_platform ascend --quant_type none --vit_quant_type none \
  --llm_kv_type None --mem_fraction 0.75 \
  --host 127.0.0.1 --port 18400 --nccl_port 30400 \
  --max_req_total_len 20480 --max_total_token_num 65536 \
  --running_max_req_size 4 --chunked_prefill_size 2048 \
  --batch_max_tokens 4096 \
  --graph_max_batch_size 4 --graph_max_len_in_batch 20480 \
  --llm_prefill_att_backend fa3 --llm_decode_att_backend fa3 \
  --vit_att_backend sdpa --visual_tp 1 --visual_dp 2 \
  --visual_infer_batch_size 2 --visual_send_batch_size 1 \
  --max_image_pixels 3686400 --max_image_token_count 8192 \
  --embed_cache_storage_size 1 --visual_gpu_ids 0 1 \
  --reasoning_parser qwen3 --linear_att_cache_size 4 \
  --linear_att_ssm_data_type float32 \
  --disable_flashinfer_allreduce --disable_symm_mem_allreduce
```

实际正式入口仍使用 `start_instance.sh MODEL_KEY REPLICA_INDEX`，由它设置每个模型的卡号、通信端口和正式运行目录。

910B 的正式运行脚本依赖同一 reprokit 中的 `provenance/`、`records/` 和平台补丁文件。该目录保存的是实际使用的入口脚本，完整运行树和补丁归档仍保留在 910B 主机的正式复现包中。

## 使用提示

这些文件是实际运行配置的归档。迁移到新目录前，需要同步修改脚本中指向模型、LightLLM runtime、缓存和日志的绝对路径，并保留对应平台的厂商 Python 环境。Qwen3.6 和 Qwen3-VL 请求均使用 `enable_thinking=false`。

脚本使用 `python3 -m lightllm.server.api_server`。当前冻结分支没有 `lightllm.server.__main__`，所以命令保留可直接执行的实际入口，参数排版与 Word 中的 `python3 -m lightllm.server` 示例一致。
