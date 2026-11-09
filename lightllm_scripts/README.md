# LightLLM 启动命令

每个平台保留三个直接启动脚本，参数按本轮实际运行配置整理。

```text
a100/start_gemma.sh
a100/start_qwen36.sh
a100/start_qwen3vl.sh

mlu590/start_gemma.sh
mlu590/start_qwen36.sh
mlu590/start_qwen3vl.sh

c500/start_gemma.sh
c500/start_qwen36.sh
c500/start_qwen3vl.sh

ascend910b2c/start_gemma.sh
ascend910b2c/start_qwen36.sh
ascend910b2c/start_qwen3vl.sh
```

脚本统一使用以下形式：

```bash
export PLATFORM_ENV=value

python3 -m lightllm.server.api_server \
  --model_dir /path/to/model \
  --tp 1 \
  --data_type bfloat16 \
  --host 127.0.0.1 \
  --port 8000
```

LightLLM 来源：<https://github.com/zhangts20/LightLLM>

- 分支：`refactor_platform_dev_v1.2.0_dspark`
- 基线 commit：`0db819b929e713f9337723ff9dcd6a7a04924821`

平台 checkout：

- A100：`/data/wushuo/evaluations/LightLLM`
- MLU590：`/data/wushuo_mlu/evaluations/LightLLM`
- C500：`/data/wushuo_metax/evaluations/LightLLM`
- 910B2C：`/data/zhangtaoshan/LightLLM`，正式运行树使用同一 commit 的平台补丁版本

脚本中的模型、runtime、缓存和端口路径保持原机器配置。迁移时按实际机器修改这些路径，以及厂商 Python 环境变量。
