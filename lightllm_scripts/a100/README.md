# Final A100 serving reproduction

The original LightLLM checkout is unchanged. `records/source_snapshot.tar.gz`
is the exact archive of commit `0db819b929e713f9337723ff9dcd6a7a04924821`.
`rebuild_source.sh NEW_EMPTY_DIRECTORY` verifies its SHA256, adds the recorded
`sitecustomize.py` path shim, applies the Qwen recurrent-wrapper patch and the
Gemma tokenizer `add_special_tokens` patch, compiles both patched Python files,
and verifies their final SHA256. Its default destination is the executed runtime
path `/tmp/mmbench_a100_lightllm_0db819b929e7`; it refuses a nonempty destination.

The final reconstruction was tested in an independent directory. All 1,907
non-bytecode files matched the executed runtime, byte for byte. See
`records/rebuild_verification.json` and `records/rebuild_expected.json`.

`launch.py INSTANCE --dry-run` shows the exact executed argv/environment for
`gemma0`, `gemma1`, `qwen0`, `qwen1`, `qwen2` or `qwen3`. The two Gemma templates
include `--disable_dynamic_prompt_cache`. All templates use BF16 and no weight or
KV quantization. Qwen's effective SSM state dtype is FP32 (captured by the API
observations); requests must include
`chat_template_kwargs={"enable_thinking": false}`. Both models use the original
images and default model preprocessing.

To replay on the original A100 container, place this directory at
`/var/tmp/wushuo-mmbench-20261105/serving`, rebuild the source when the runtime
path is absent, and make sure the verified model caches exist at
`/var/tmp/wushuo-mmbench-20261105/models/{gemma-3-4b-it,Qwen3.6-27B}`. Then use:

```bash
bash /var/tmp/wushuo-mmbench-20261105/serving/start.sh gemma0
bash /var/tmp/wushuo-mmbench-20261105/serving/start.sh gemma1
bash /var/tmp/wushuo-mmbench-20261105/serving/start.sh qwen0
bash /var/tmp/wushuo-mmbench-20261105/serving/start.sh qwen1
bash /var/tmp/wushuo-mmbench-20261105/serving/start.sh qwen2
```

This fills all eight GPUs: Gemma on 0/1, Qwen TP2 replicas on 2–3, 4–5 and 6–7.
`qwen3` uses GPUs 0–1 and port 18103, so start it only after stopping both Gemma
replicas. `stop.sh INSTANCE` terminates the instance process group. Restarts write
new timestamped records under `records/restarts/`; historical execution records
and `launch_templates.json` remain unchanged. The evaluation used Gemma output
budget 4096 and Qwen budget 8192, temperature 0, top_p 1 and seed 42. The formal
run manifests and API observations are authoritative for all execution details.

`records/serving_handoff.before_bosfix.json`, the original `gemma0.json` and
`gemma1.json`, and early pilot directories are historical diagnostics. The final
Gemma commands are `gemma0.bosfix.json` and `gemma1.bosfix.json` and are exactly
represented by `launch_templates.json`.
