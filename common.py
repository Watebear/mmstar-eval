"""Persistence and frozen inputs shared by the inference and offline scoring CLIs."""
from __future__ import annotations

import hashlib
import errno
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVALSCOPE_COMMIT = "31a54d56d359875b69a66cd5f3f9dadc08268570"
SUBSETS = ["coarse perception", "fine-grained perception", "instance reasoning",
           "logical reasoning", "math", "science & technology"]
SYNC_WARNINGS = []


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            try:
                os.fsync(directory_fd)
            except OSError as exc:
                if exc.errno not in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                    raise
                message = f"Directory fsync unsupported on {path.parent}; file fsync and atomic rename completed"
                if message not in SYNC_WARNINGS:
                    SYNC_WARNINGS.append(message)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def append_event(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(encoded(event) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def source_lock():
    lock = read_json(ROOT / "upstream.lock.json")
    import evalscope
    imported = Path(evalscope.__file__).resolve().parent
    for relative, expected in lock["source_files"].items():
        actual = file_sha(imported / relative)
        if actual != expected:
            raise RuntimeError(f"EvalScope differs from pinned commit: {relative}")
    return lock


def make_config(model, generation, dataset_hub="modelscope"):
    from evalscope import TaskConfig
    return TaskConfig(model=model, eval_type="openai_api", api_url="http://frozen-run.invalid/v1",
                      datasets=["mm_star"], dataset_hub=dataset_hub, repeats=1,
                      generation_config=generation, judge={"strategy": "rule"}, seed=42)


def load_snapshot(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ensure_snapshot(path, model, generation, dataset_hub, dataset_parquet=None):
    path = Path(path)
    if path.exists():
        return load_snapshot(path)
    from evalscope.api.registry import get_benchmark
    config = make_config(model, generation, dataset_hub)
    normalized_parquet = None
    if dataset_parquet:
        from datasets import Dataset, Image
        dataset = Dataset.from_parquet(str(dataset_parquet))
        if isinstance(dataset[0]["image"], bytes):
            # Preserve every image byte; only declare the official Image storage schema.
            dataset = dataset.cast_column("image", Image(decode=False))
            normalized_parquet = path.parent / "mm_star.normalized.parquet"
            normalized_parquet.parent.mkdir(parents=True, exist_ok=True)
            dataset.to_parquet(str(normalized_parquet))
        config.dataset_args = {"mm_star": {"dataset_id": str(Path(normalized_parquet or dataset_parquet).resolve())}}
    adapter = get_benchmark("mm_star", config)
    if dataset_parquet:
        datasets, _ = adapter.load_from_disk(use_local_loader=True)
    else:
        datasets = adapter.load_dataset()
    rows = [{"subset": subset, "sample": sample.model_dump(mode="json")}
            for subset, dataset in datasets.items() for sample in dataset]
    counts = Counter(row["subset"] for row in rows)
    if counts != Counter({subset: 250 for subset in SUBSETS}):
        raise RuntimeError(f"Expected the full official MMStar cohort, got {dict(counts)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write(encoded(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    atomic_json(path.with_suffix(".metadata.json"), {
        "created_at": now(), "dataset_id": "evalscope/MMStar" if dataset_hub == "modelscope" else "Lin-Chen/MMStar",
        "dataset_hub": dataset_hub, "split": "val", "evalscope_commit": EVALSCOPE_COMMIT,
        "sha256": file_sha(path), "sample_counts": dict(counts), "total": len(rows),
        "source_parquet": str(Path(dataset_parquet).resolve()) if dataset_parquet else None,
        "source_parquet_sha256": file_sha(dataset_parquet) if dataset_parquet else None,
        "normalized_parquet_sha256": file_sha(normalized_parquet) if normalized_parquet else None,
        "normalization": "datasets.Image(decode=False), original image bytes preserved" if normalized_parquet else None,
        "snapshot_contains": "Official processed Sample objects, exact prompts, image data URIs, choices and targets",
    })
    return rows


def selected_rows(rows, limit):
    counts = Counter()
    selected = []
    for row in rows:
        if limit is None or counts[row["subset"]] < limit:
            selected.append(row)
            counts[row["subset"]] += 1
    return selected


def row_key(row):
    return f"{SUBSETS.index(row['subset']):02d}-{row['sample']['id']:05d}"


def result_path(run_dir, row):
    return Path(run_dir) / "results" / f"{row_key(row)}.json"


def saved_results(run_dir):
    return [read_json(path) for path in sorted((Path(run_dir) / "results").glob("*.json"))]


def write_audit(run_dir, manifest, rows, records=None):
    if records is None:
        records = saved_results(run_dir)
    by_endpoint = Counter(record["endpoint"] for record in records)
    stops = Counter(record["stop_reason"] for record in records)
    lengths = [record["usage"].get("output_tokens", 0) for record in records]
    per_subset = Counter(record["subset"] for record in records)
    summary = {
        "updated_at": now(), "run_fingerprint": manifest["run_fingerprint"],
        "planned": len(rows), "saved": len(records), "remaining": len(rows) - len(records),
        "complete": len(records) == len(rows), "by_endpoint": dict(by_endpoint),
        "by_subset": dict(per_subset), "stop_reasons": dict(stops),
        "output_tokens": {"min": min(lengths, default=0), "max": max(lengths, default=0),
                          "mean": sum(lengths) / len(lengths) if lengths else 0},
        "error_attempts": sum(1 for _ in open(Path(run_dir) / "errors.jsonl"))
        if (Path(run_dir) / "errors.jsonl").exists() else 0,
        "filesystem_notes": list(SYNC_WARNINGS),
    }
    atomic_json(Path(run_dir) / "progress.json", summary)
    return summary
