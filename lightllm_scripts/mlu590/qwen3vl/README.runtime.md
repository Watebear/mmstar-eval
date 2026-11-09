# Qwen3-VL-8B-Instruct / MLU

本轮独立运行根为 `/tmp/wushuo-mlu-qwen3vl8b-20261106`；所有源码、依赖 overlay、模型读取、控制配置、缓存、日志和逐题结果均从本机盘使用。共享 `/data` 仅由独立异步进程保存静态归档，避免远端 FUSE 元数据阻塞推理控制。

模型缓存为 `/tmp/wushuo-mlu-mmbench-20261105/models/Qwen3-VL-8B-Instruct`，四个 safetensors 和十二个非权重文件均校验为官方版本 `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`。原 `/data/wangpengyu_vlm/models/Qwen3-VL-8B-Instruct` 未改动。`model_manifest.json` 保存源/目标 SHA256、大小和官方校验结果。

CPU 入口为 `serving/evaluator_python.sh`，复用独立无厂商 Torch 的 `/tmp/wushuo-mlu-mmbench-20261105/evaluator-venv` 与冻结 `evaluator-src`。新输出位于本轮 `runs/pilot`、`runs/formal`，每一题原子写入；两个 `*-offline-audit` 目录保存真实 socket 禁网重新评分及已完成任务零新请求证明。核心 common/infer/score 和冻结 MMStar 1500 题从上一轮原样使用。

GPU 入口为 `serving/start_instance.sh CARD`，仅在 A100 小样通过 gate 后运行。TP1/BF16、prefill fa3 / decode Triton、ViT SDPA、triton fallback、MLUGraph4/running4，max request20480、正式pool65536、chunk1024、batch4096、visual1，图像上限3686400像素/8192 tokens；API18200+CARD，通信30200+CARD。第一卡完成30题小样及6张超gateway图的独立诊断后逐步扩2、4、8卡，每个新增API先用原始row0执行独立真图smoke，正式请求每API并发4，max_tokens8192、temperature0、top_p1、seed42、enable_thinking=false。精确实际命令和 server_info 在 `serving/logs` 与 `serving/records`。

`serving/supervise_round.py` 持久推进小样、扩容、1500题正式评测、官方评分、独立离线重算、真实断点恢复审计、静态归档；收到机器外原始归档 SHA确认后只关闭本轮进程组。`artifacts/complete.json` 是最终完成索引。`archives/formal.tar.gz` 在停卡前保存原始结果，不再覆写；`archives/complete.tar.gz` 在停卡后保存相同原始结果及停卡证明，二者独立且不可变。旧结果和旧复现包均保留。

`archives/provenance.tar.gz` 包含原始 LightLLM source archive、依赖 overlay archive、CPU源码/固定数据/环境与重建入口。`serving/rebuild_source.py --destination PATH` 不加载GPU即可在新目录重建源码与overlay并核对4452个文件。最终 complete 包另含全部API原始日志、最终source/overlay逐文件清单、自动调优等新增文件delta及严格重建证明；最终入口以 complete 包内版本为准。临时模型传输凭证不入归档。

永久目录为 `/data/wushuo_mlu/evaluations/mmbench_eval/qwen3vl8b-20261106`；机器外 SHA 验证归档位于 Mac 工作区 `work/qwen3vl8b/mlu/archives`，最终用户交付由主任务汇总。

本模型 H128/BF16 在厂商 CNNL decode 的 block_size1 下仅256B/块，未满足最低512B限制，因此在任何小样请求之前改用现有注册支持MLU的 Triton decode；保持prefill fa3与所有长度/生成/精度参数。native失败参数复现及Triton对独立FP32参考/MLUGraph等价证明均保存在 records。Deepstack另采用已验证的MLU CPU-cache选行拷贝fallback，保留CUDA/MACA实现。

## 最终恢复入口优先级

**主入口是 `final_archive.json` 指向并通过SHA的 `complete.tar.gz`，其中的 `serving/env.sh` / `start_instance.sh` / `evaluator_python.sh` 是正式实际版本。** 先把 complete 包解压到实际运行根，再执行其 `serving/rebuild_source.py --destination /tmp/wushuo-mlu-qwen3vl8b-20261106`；该入口验证原0db source archive与pydeps archive，应用 `runtime_delta.tar.gz`（包含MLU Deepstack修复及实际新增文件），再按 `runtime_final_files.json` 逐文件严格核对。最后一次 scratch 重建证明在 `serving/records/rebuild-proof-final.json`。最终启动必须使用 prefill fa3 / decode Triton、Graph4/running4、KV池65536、HTTP timeout1800、首次正式pass客户端4/API；少量失败项的missing-only恢复pass客户端1/API、transport timeout3600s，以实际driver命令历史为准，其余图像/8192预算/温度/seed/BF16/无量化参数见实际命令与最终protocol。

早期 `provenance.tar.gz` 仅提供基础依赖、CPU EvalScope源码和冻结dataset，**不是无补丁的GPU恢复入口**；controller-revision历史包仅供诊断。不要将早期32k/client2或fa3 decode launcher用作正式入口。一次性capacity四原始样本的两份原始结果、命令、大小与内存证据在 `records/capacity_probe_raw.tar.gz` 与 `capacity_comparison.json`；选择只基于吞吐和显存。

通信审计以 `runs/formal/transport_audit.json` 为准，分别记录原生EvalScope内部timeout/retry、`errors.jsonl`耗尽重试的worker失败、外层missing-only pass。首次正式pass实际timeout1800s；config中预备3600s仅在当前pass自然结束并确需补失败时才读取。`last_invocation.json`和各pass命令记录实际值，生成预算始终8192。单题 `latency_seconds` 可能包含原生内部重试等待，不能当成单次HTTP请求耗时；不按答对情况重抽任何已保存结果。

恢复并发历史：首次1500题队列按8API×4 workers、timeout1800s执行；首次队列自然结束后，仅对无成功JSON的题目使用8API×1 workers、timeout3600s补跑，使少量剩余题分布到不同卡。服务端Graph/running4、KV池65536及8192生成预算完全不变。预备配置不代表恢复pass已经发生，实际次数/命令见driver与transport_audit。
