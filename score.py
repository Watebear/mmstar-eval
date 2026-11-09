#!/usr/bin/env python3
"""Score saved model results with the official MMStar adapter, without network access."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import atomic_json, make_config, now, read_json, saved_results, source_lock


def score(run_dir, require_complete=False):
    from evalscope.api.dataset import Sample
    from evalscope.api.evaluator import ModelResult, TaskState
    from evalscope.api.registry import get_benchmark

    source_lock()
    run_dir = Path(run_dir).resolve()
    manifest = read_json(run_dir / "manifest.json")
    records = saved_results(run_dir)
    expected = {(item["subset"], item["sample_id"]) for item in manifest["identity"]["selection"]}
    actual = {(record["subset"], record["sample"]["id"]) for record in records}
    if actual - expected or len(actual) != len(records):
        raise ValueError("Unexpected or duplicate samples in results")
    if require_complete and actual != expected:
        raise ValueError(f"Run is incomplete: {len(actual)}/{len(expected)} saved; scoring made no API requests")
    cfg = make_config(manifest["model"], manifest["generation_config"], manifest["dataset_hub"])
    adapter = get_benchmark("mm_star", cfg)
    by_subset = defaultdict(list)
    score_rows = []
    stops = Counter()
    for record in records:
        if record["run_fingerprint"] != manifest["run_fingerprint"]:
            raise ValueError("Saved result has a mismatched run fingerprint")
        sample = Sample.model_validate(record["sample"])
        prediction = ModelResult.model_validate(record["model_result"])
        state = TaskState(model=prediction.model, sample=sample, messages=prediction.messages,
                          output=prediction.model_output, completed=True)
        sample_score = adapter.calculate_metrics(state)
        by_subset[record["subset"]].append(sample_score)
        score_rows.append({"subset": record["subset"], "key": record["key"],
                           "sample_id": sample.id, "score": sample_score.model_dump(mode="json")})
        stops[record["stop_reason"]] += 1
    aggregates = {subset: adapter.aggregate_scores(scores) for subset, scores in by_subset.items()}
    report = adapter.generate_report(aggregates, model_name=manifest["model"], output_dir=str(run_dir / "scores"))
    report_dict = report.model_dump(mode="json")
    values = [float(item.score.main_value or 0) for group in by_subset.values() for item in group]
    summary = {"scored_at": now(), "run_fingerprint": manifest["run_fingerprint"],
               "model": manifest["model"], "hardware": manifest["identity"]["hardware"],
               "planned": len(expected), "scored": len(records), "missing": len(expected - actual),
               "complete": actual == expected, "accuracy": sum(values) / len(values) if values else None,
               "by_subset": {subset: {"count": len(scores),
                             "accuracy": sum(float(s.score.main_value or 0) for s in scores) / len(scores)}
                             for subset, scores in by_subset.items()},
               "stop_reasons": dict(stops), "invalid_answer_count": sum(
                   not row["score"]["score"].get("extracted_prediction") for row in score_rows),
               "official_report": report_dict}
    atomic_json(run_dir / "scores" / "summary.json", summary)
    atomic_json(run_dir / "scores" / "per_sample.json", score_rows)
    atomic_json(run_dir / "scores" / "mm_star.json", report_dict)
    print(json.dumps({key: value for key, value in summary.items() if key != "official_report"},
                     ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    score(args.run_dir, args.require_complete)


if __name__ == "__main__":
    main()
