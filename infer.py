#!/usr/bin/env python3
"""Run the official EvalScope MMStar inference against one or more model replicas."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import queue
import threading
import time
from pathlib import Path
from urllib.request import urlopen

from common import (EVALSCOPE_COMMIT, ROOT, append_event, atomic_json, digest, ensure_snapshot,
                    file_sha, load_snapshot, make_config, now, read_json, result_path,
                    row_key, saved_results, selected_rows, source_lock, write_audit)


def infer(args):
    from evalscope.api.dataset import Sample
    from evalscope.api.evaluator import ModelResult
    from evalscope.api.model import get_model
    from evalscope.api.registry import get_benchmark
    from evalscope.models.openai_compatible import OpenAICompatibleAPI

    source_lock()
    generation = {"temperature": 0.0, "top_p": 1.0, "seed": 42, "max_tokens": args.max_tokens,
                  "stream": False, "timeout": args.timeout, "retries": args.retries, "retry_interval": 5}
    if args.disable_thinking or "qwen" in args.model.lower():
        generation["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    if args.extra_body:
        generation.setdefault("extra_body", {}).update(json.loads(args.extra_body))
    cfg = make_config(args.model, generation, args.dataset_hub)
    snapshot = Path(args.snapshot).resolve()
    ensure_snapshot(snapshot, args.model, generation, args.dataset_hub, getattr(args, "dataset_parquet", None))
    all_rows = load_snapshot(snapshot)
    rows = selected_rows(all_rows, args.limit_per_subset)
    counts = args.endpoint_concurrency or [args.concurrency] * len(args.endpoint)
    if len(counts) != len(args.endpoint) or any(count < 1 for count in counts):
        raise ValueError("Provide one positive --endpoint-concurrency per endpoint")
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    wrappers = {name: file_sha(ROOT / name) for name in ["common.py", "infer.py", "score.py"]}
    identity = {"evalscope_commit": EVALSCOPE_COMMIT, "dataset_snapshot_sha256": file_sha(snapshot),
                "model": args.model, "hardware": args.hardware, "weights_revision": args.weights_revision,
                "wrapper_sha256": wrappers,
                "generation_config": {k: v for k, v in generation.items()
                                      if k not in {"timeout", "retries", "retry_interval", "stream"}},
                "selection": [{"subset": row["subset"], "sample_id": row["sample"]["id"],
                               "source_index": row["sample"]["metadata"].get("index")} for row in rows]}
    fingerprint = digest(identity)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["run_fingerprint"] != fingerprint:
            raise ValueError("Run configuration changed; use a separate --run-dir to avoid mixing predictions")
    else:
        manifest = {"schema_version": 1, "created_at": now(), "run_fingerprint": fingerprint,
                    "identity": identity, "snapshot": str(snapshot), "model": args.model,
                    "dataset_hub": args.dataset_hub, "generation_config": generation,
                    "planned": len(rows), "limit_per_subset": args.limit_per_subset,
                    "upstream": read_json(ROOT / "upstream.lock.json"),
                    "wrapper_sha256": wrappers}
        atomic_json(manifest_path, manifest)
    atomic_json(run_dir / "last_invocation.json", {"started_at": now(), "endpoints": args.endpoint,
                "concurrency": counts, "generation_config": generation})
    pending = queue.Queue()
    for row in rows:
        path = result_path(run_dir, row)
        if path.exists():
            record = read_json(path)
            if record["run_fingerprint"] != fingerprint:
                raise ValueError(f"Incompatible saved result: {path}")
        else:
            pending.put(row)
    audit_records = [{k: r[k] for k in ["endpoint", "stop_reason", "usage", "subset"]}
                     for r in saved_results(run_dir)]
    initial = write_audit(run_dir, manifest, rows, audit_records)
    print(json.dumps({"event": "start", **initial}, ensure_ascii=False), flush=True)
    if pending.empty():
        return 0

    services = []
    for endpoint in args.endpoint:
        origin = endpoint.rstrip("/").removesuffix("/v1")
        with urlopen(origin + "/get_model_name", timeout=15) as response:
            reported = json.load(response)
        if reported.get("model_name") != args.model:
            raise ValueError(f"Endpoint {endpoint} serves {reported}, expected {args.model}")
        with urlopen(origin + "/get_server_info", timeout=15) as response:
            server_info = json.load(response)
        services.append({"endpoint": endpoint, "model_name_response": reported, "server_info": server_info})
    atomic_json(run_dir / "service_observation.json", {"observed_at": now(), "services": services})

    class CaptureAPI(OpenAICompatibleAPI):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.capture = threading.local()

        def on_response(self, response):
            self.capture.raw_response = response

    adapter = get_benchmark("mm_star", cfg)
    event_lock = threading.Lock()
    completed_lock = threading.Lock()
    done = [initial["saved"]]

    def worker(endpoint):
        api = CaptureAPI(model_name=args.model, base_url=endpoint, api_key="EMPTY",
                         config=cfg.generation_config, max_retries=0)
        model = get_model(api, "openai_api", config=cfg.generation_config, memoize=False)
        while True:
            try:
                row = pending.get_nowait()
            except queue.Empty:
                return
            sample = Sample.model_validate(row["sample"])
            start = time.monotonic()
            api.capture.raw_response = None
            event = {"time": now(), "key": row_key(row), "subset": row["subset"],
                     "sample_id": sample.id, "endpoint": endpoint, "run_fingerprint": fingerprint}
            with event_lock:
                append_event(run_dir / "attempts.jsonl", {**event, "event": "request_started"})
            try:
                state = adapter.run_inference(model, sample, str(run_dir / "results"))
                output = state.output
                if api.capture.raw_response is None or output.error or output.stop_reason in {"model_length", "content_filter"}:
                    raise RuntimeError(f"No successful model response: stop={output.stop_reason}, error={output.error}")
                elapsed = time.monotonic() - start
                official = ModelResult.from_task_state(state).model_dump(mode="json")
                record = {"schema_version": 1, "run_fingerprint": fingerprint, "key": row_key(row),
                          "saved_at": now(), "subset": row["subset"], "sample": row["sample"],
                          "model_result": official, "raw_response": api.capture.raw_response,
                          "endpoint": endpoint, "latency_seconds": elapsed,
                          "usage": output.usage.model_dump(mode="json") if output.usage else {},
                          "stop_reason": output.stop_reason, "completion_characters": len(output.completion),
                          "generation_config": generation}
                atomic_json(result_path(run_dir, row), record)
                with completed_lock:
                    done[0] += 1
                    audit_records.append({k: record[k] for k in ["endpoint", "stop_reason", "usage", "subset"]})
                    print(json.dumps({"event": "saved", "saved": done[0], "planned": len(rows),
                                      "key": row_key(row), "endpoint": endpoint,
                                      "stop_reason": output.stop_reason, "usage": record["usage"]}), flush=True)
                    write_audit(run_dir, manifest, rows, audit_records)
                with event_lock:
                    append_event(run_dir / "attempts.jsonl", {**event, "event": "request_saved", "latency_seconds": elapsed})
            except Exception as exc:
                with event_lock:
                    append_event(run_dir / "errors.jsonl", {**event, "error_type": type(exc).__name__,
                                 "error": str(exc), "latency_seconds": time.monotonic() - start,
                                 "raw_response": api.capture.raw_response})
                print(json.dumps({"event": "error", **event, "error": str(exc)}), flush=True)
            finally:
                pending.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=sum(counts)) as pool:
        futures = [pool.submit(worker, endpoint) for endpoint, count in zip(args.endpoint, counts)
                   for _ in range(count)]
        for future in futures:
            future.result()
    summary = write_audit(run_dir, manifest, rows, audit_records)
    print(json.dumps({"event": "finished", **summary}, ensure_ascii=False), flush=True)
    return 0 if summary["complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exact model name accepted by LightLLM")
    parser.add_argument("--endpoint", action="append", required=True, help="OpenAI base URL; repeat for replicas")
    parser.add_argument("--concurrency", type=int, default=1, help="Requests per service")
    parser.add_argument("--endpoint-concurrency", type=int, nargs="+", help="Optional per-service concurrency list")
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--weights-revision", required=True, help="Weight snapshot revision or content manifest hash")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--snapshot", default=str(ROOT / "snapshots" / "mm_star.jsonl"))
    parser.add_argument("--dataset-hub", choices=["modelscope", "huggingface"], default="modelscope")
    parser.add_argument("--dataset-parquet", help="Existing official MMStar parquet; only used to create the frozen snapshot")
    parser.add_argument("--limit-per-subset", type=int)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--disable-thinking", action="store_true", help="Automatic for model names containing Qwen")
    parser.add_argument("--extra-body", help="Additional OpenAI request body JSON")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.limit_per_subset is not None and args.limit_per_subset < 1:
        parser.error("--limit-per-subset must be positive")
    from filelock import FileLock
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = Path(args.snapshot).resolve()
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(run_dir / ".infer.lock"), timeout=0):
        with FileLock(str(snapshot) + ".lock", timeout=600):
            generation = {"temperature": 0.0, "max_tokens": args.max_tokens}
            source_lock()
            ensure_snapshot(snapshot, args.model, generation, args.dataset_hub, args.dataset_parquet)
        return infer(args)


if __name__ == "__main__":
    raise SystemExit(main())
