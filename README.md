# MMStar Eval

这是基于 EvalScope MMStar 的多平台多模态评测项目。评测分为两个阶段。

- `infer.py` 向一个或多个 LightLLM OpenAI 兼容接口发送请求，逐题原子保存原始回答、请求状态和评测输入。重复执行同一个 `--run-dir` 会跳过已经成功保存的题目，支持断点恢复。
- `score.py` 读取已保存结果，使用官方 MMStar 适配器离线评分，不需要再次访问模型服务。
- `common.py` 保存两阶段共用的 EvalScope 数据、提示、解析和结果格式逻辑。
- `requirements.txt` 是已验证评测环境的固定依赖版本。
- `sampling_params.json` 固定三类模型的采样参数和输出上限。

## 安装

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

评测数据需要使用官方 MMStar val 集。可以先用一个已有的 parquet 快照创建冻结题目快照，正式运行时复用同一个快照。

## 使用方式

先启动对应平台的 LightLLM 服务，再使用 `infer.py`。`--endpoint` 可以重复传入多个服务地址。

小样例：

```bash
python infer.py \
  --model gemma-3-4b-it \
  --endpoint http://127.0.0.1:18000/v1 \
  --endpoint http://127.0.0.1:18001/v1 \
  --hardware a100 \
  --weights-revision <model-revision-or-manifest> \
  --run-dir runs/a100/gemma-smoke \
  --dataset-parquet /path/to/mmstar.parquet \
  --limit-per-subset 2 \
  --max-tokens 4096
```

Qwen3.6 或 Qwen3-VL 评测关闭思考：

```bash
python infer.py \
  --model Qwen3.6-27B \
  --endpoint http://127.0.0.1:18100/v1 \
  --endpoint http://127.0.0.1:18101/v1 \
  --hardware a100 \
  --weights-revision <model-revision-or-manifest> \
  --run-dir runs/a100/qwen36 \
  --max-tokens 8192 \
  --disable-thinking
```

完成请求后离线评分：

```bash
python score.py --run-dir runs/a100/gemma-smoke --require-complete
```

正式评测去掉 `--limit-per-subset`，并固定模型 revision、服务参数、温度、随机种子和题目快照。各平台实际使用的 LightLLM 启动脚本位于 [`lightllm_scripts/`](lightllm_scripts/README.md)。每个平台提供按模型拆分的直接启动命令。

采样参数统一记录在 [`sampling_params.json`](sampling_params.json)。运行 `infer.py` 时按模型传入对应的 `--max-tokens` 和 `--disable-thinking`，公共参数固定为 `temperature=0`、`top_p=1`、`seed=42`。Gemma 输出上限为 4096，Qwen3.6 和 Qwen3-VL 输出上限为 8192。

启动命令和采样参数足以复现服务请求配置。要复现正式分数，还需要同时固定对应平台的 LightLLM 运行树及补丁、模型权重 revision 或文件校验值、厂商镜像和 Python 依赖、EvalScope commit、MMStar 题目快照，以及请求并发、超时和断点恢复策略。平台运行树的来源和补丁边界已在 `lightllm_scripts/README.md` 中标明。

## 目录说明

```text
infer.py                  请求服务并保存结果
score.py                  离线评分
common.py                 共用评测逻辑
requirements.txt          固定评测依赖
sampling_params.json      固定采样参数
lightllm_scripts/         各平台实际启动脚本快照
```
