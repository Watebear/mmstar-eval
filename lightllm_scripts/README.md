# LightLLM 启动脚本归档

这里保存本轮 MMStar 评测中实际使用的 LightLLM 启动脚本快照。脚本中的路径、环境变量、端口、GPU 分配和模型参数均保持原始平台配置，脚本依赖各机器上的模型缓存、厂商运行时和 LightLLM 运行目录。

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

### MLU590

目录：`mlu590/`

- 原始 checkout：`/data/wushuo_mlu/evaluations/LightLLM`
- 评测项目：`/data/wushuo_mlu/evaluations/mmbench_eval`
- Gemma 和 Qwen3.6 入口：`gemma_qwen36/start_instance.sh`
- Qwen3-VL-8B-Instruct 入口：`qwen3vl/start_instance.sh`
- Qwen3-VL 最终运行根：`/tmp/wushuo-mlu-qwen3vl8b-20261106`

### Metax C500

目录：`c500/`

- C500 与 Metax 共用存储，原始 checkout：`/data/wushuo_metax/evaluations/LightLLM`
- 评测项目：`/data/wushuo_metax/evaluations/mmbench_eval`
- Gemma 和 Qwen3.6 入口：`gemma_qwen36/start_service.sh MODEL PORT DEVICES`
- Qwen3-VL-8B-Instruct 入口：`qwen3vl/start_instance.sh GPU_INDEX`
- Qwen3-VL 最终运行根：`/tmp/wushuo-metax-qwen3vl8b-20261106`

### Ascend 910B2C

目录：`ascend910b2c/`

- 基线 checkout：`/data/zhangtaoshan/LightLLM`
- 最终实际运行源码：`/tmp/wushuo-910b-mmstar-20261106/lightllm-src`
- 最终运行源码由 0db819b 基线和 910B 平台补丁重建得到
- 入口：`start_instance.sh MODEL_KEY REPLICA_INDEX`
- 控制器：`launch_controller.sh`
- 环境：`source_env.sh`
- 重建源码：`rebuild_source.py`

910B 的正式运行脚本依赖同一 reprokit 中的 `provenance/`、`records/` 和平台补丁文件。该目录保存的是实际使用的入口脚本，完整运行树和补丁归档仍保留在 910B 主机的正式复现包中。

## 使用提示

这些文件是实际运行配置的归档。迁移到新目录前，需要同步修改脚本中指向模型、LightLLM runtime、缓存和日志的绝对路径，并保留对应平台的厂商 Python 环境。Qwen3.6 和 Qwen3-VL 请求均使用 `enable_thinking=false`。
