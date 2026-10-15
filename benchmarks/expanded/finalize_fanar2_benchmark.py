#!/usr/bin/env python3
"""Merge, lightly postprocess raw output, evaluate, and tabulate Fanar2 results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from fanar2_benchmark_common import (
    CAMEL_CONLL_ROOT,
    CAMEL_CONLL_SCRIPT,
    DEPRELS,
    GOLD_PROMPT_PATH,
    MODEL_ID,
    POOL_PATHS,
    RAW_PROMPT_PATH,
    RUN_ROOT,
    SPLIT_PATHS,
    condition_specs,
    read_conll,
    read_jsonl,
    write_json,
)

from finalize_full_coverage import postprocess_parse  # noqa: E402


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def format_gold(sentences: list[Any]) -> str:
    chunks = []
    for sentence in sentences:
        chunks.append(f"# text = {sentence.text}\n")
        chunks.append("# treeTokens = " + " ".join(sentence.tree_tokens) + "\n")
        chunks.extend("\t".join(row) + "\n" for row in sentence.rows)
        chunks.append("\n")
    return "".join(chunks)


def format_prediction(sentences: list[Any], parses: list[dict[str, Any]]) -> str:
    chunks = []
    for sentence, parsed in zip(sentences, parses):
        items = parsed["parses"]
        chunks.append(f"# text = {sentence.text}\n")
        chunks.append("# treeTokens = " + " ".join(item["form"] for item in items) + "\n")
        for item in items:
            chunks.append(
                f"{item['id']}\t{item['form']}\t_\t_\t_\t_\t"
                f"{item['head']}\t{item['deprel']}\t_\t_\n"
            )
        chunks.append("\n")
    return "".join(chunks)


def validate_serializable(parsed: Any) -> list[str]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("parses"), list):
        return ["missing parses array"]
    items = parsed["parses"]
    if not items:
        return ["empty parses array"]
    errors = []
    for expected, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            errors.append(f"item {expected} is not an object")
            continue
        if item.get("id") != expected:
            errors.append(f"item {expected}: id={item.get('id')!r}")
        form = item.get("form")
        if not isinstance(form, str) or not form or any(x.isspace() for x in form):
            errors.append(f"item {expected}: invalid form")
        head = item.get("head")
        if not isinstance(head, int) or not 0 <= head <= len(items):
            errors.append(f"item {expected}: invalid head={head!r}")
        if item.get("deprel") not in DEPRELS:
            errors.append(f"item {expected}: invalid deprel")
    return errors


def evaluate(condition: str, condition_dir: Path) -> dict[str, str]:
    evaluation_dir = condition_dir / "evaluation"
    gold_dir = evaluation_dir / "gold"
    parsed_dir = evaluation_dir / "parsed"
    gold_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    basename = "input.conllx"
    (gold_dir / basename).write_text(
        (condition_dir / "gold.conllx").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (parsed_dir / basename).write_text(
        (condition_dir / "predictions.conllx").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    log_path = evaluation_dir / "camel_conll.log"
    command = [
        sys.executable,
        str(CAMEL_CONLL_SCRIPT),
        "--gold",
        str(gold_dir / basename),
        "--parsed",
        str(parsed_dir / basename),
        "-x",
        "-n",
        "-a",
        "--output_path",
        str(evaluation_dir),
        condition,
    ]
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=CAMEL_CONLL_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result_path = evaluation_dir / f"{condition}.tsv"
    with result_path.open(encoding="utf-8-sig", newline="") as source:
        return next(csv.DictReader(source, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-jobs", type=int, default=8)
    args = parser.parse_args()
    started_epoch = time.time()
    result_root = RUN_ROOT / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    worker_ids = [
        worker_id
        for worker_id in (0, 1)
        if all(
            (
                RUN_ROOT
                / "workers"
                / f"worker_{worker_id}"
                / f"WORKER_{mode.upper()}_COMPLETE"
            ).exists()
            for mode in ("gold", "raw")
        )
    ]
    if not worker_ids:
        raise FileNotFoundError("No completed Fanar2 worker found")
    summaries: dict[str, dict[str, Any]] = {}
    evaluation_inputs = []
    all_generation_starts: list[float] = []
    all_generation_finishes: list[float] = []

    for spec in condition_specs():
        condition = spec["condition"]
        records = []
        timings = []
        for worker_id in worker_ids:
            worker_dir = RUN_ROOT / "workers" / f"worker_{worker_id}" / condition
            records.extend(read_jsonl(worker_dir / "responses.jsonl"))
            timings.append(json.loads((worker_dir / "timing.json").read_text(encoding="utf-8")))
        records.sort(key=lambda record: int(record["sentence_index"]))
        gold = read_conll(SPLIT_PATHS[spec["split"]])
        indices = [int(record["sentence_index"]) for record in records]
        if indices != list(range(len(gold))):
            raise ValueError(f"{condition}: incomplete/duplicate ordered coverage")
        condition_dir = result_root / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        with (condition_dir / "responses.merged.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        action_totals: Counter[str] = Counter()
        repaired_sentences = 0
        output_parses = []
        post_records = []
        for record, sentence in zip(records, gold):
            parsed = record.get("parsed")
            if spec["mode"] == "raw":
                if not isinstance(parsed, dict) or not isinstance(parsed.get("parses"), list) or not parsed["parses"]:
                    raise ValueError(
                        f"{condition} sentence {record['sentence_number']}: incomplete raw JSON"
                    )
                processed, actions = postprocess_parse(parsed)
                errors = validate_serializable(processed)
                if errors:
                    raise ValueError(
                        f"{condition} sentence {record['sentence_number']}: postprocess failed: {errors}"
                    )
                action_totals.update(actions)
                repaired_sentences += bool(actions)
                output_parses.append(processed)
                post_records.append(
                    {
                        "sentence_index": record["sentence_index"],
                        "sentence_number": record["sentence_number"],
                        "raw": parsed,
                        "postprocessed": processed,
                        "postprocessing_actions": actions,
                    }
                )
            else:
                errors = validate_serializable(parsed)
                if errors:
                    raise ValueError(
                        f"{condition} sentence {record['sentence_number']}: invalid gold-tokenized output: {errors}"
                    )
                if [item["form"] for item in parsed["parses"]] != sentence.tree_tokens:
                    raise ValueError(f"{condition}: structured gold forms differ from input")
                output_parses.append(parsed)
        if post_records:
            with (condition_dir / "responses.postprocessed.jsonl").open("w", encoding="utf-8") as stream:
                for record in post_records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        (condition_dir / "gold.conllx").write_text(format_gold(gold), encoding="utf-8")
        (condition_dir / "predictions.conllx").write_text(
            format_prediction(gold, output_parses), encoding="utf-8"
        )
        chunks = [chunk for timing in timings for chunk in timing["chunks"]]
        all_generation_starts.extend(float(chunk["started_epoch"]) for chunk in chunks)
        all_generation_finishes.extend(float(chunk["finished_epoch"]) for chunk in chunks)
        latencies = [
            record.get("inference_metrics", {}).get("latency_seconds") for record in records
        ]
        latencies = [float(value) for value in latencies if value is not None]
        observed_span = max(chunk["finished_epoch"] for chunk in chunks) - min(
            chunk["started_epoch"] for chunk in chunks
        )
        active_generation_seconds = sum(
            float(chunk["elapsed_seconds"]) for chunk in chunks
        )
        summary = {
            **spec,
            "retrieval_pool": spec["pool"],
            "model": MODEL_ID,
            "weight_precision": os.environ.get(
                "BAREC_BENCH_WEIGHT_PRECISION", "bfloat16"
            ),
            "sentences": len(records),
            "finish_reasons": dict(Counter(record.get("finish_reason") for record in records)),
            "retried_sentences": sum(
                int(record.get("engine_attempts", 1)) > 1 for record in records
            ),
            "structured_retries": sum(
                max(0, int(record.get("engine_attempts", 1)) - 1)
                for record in records
            ),
            "completed_on_valid_json": sum(
                bool(record.get("completed_on_valid_json")) for record in records
            ),
            "json_parse_failures": sum(record.get("parsed") is None for record in records),
            "prompt_tokens": sum(record["prompt_tokens"] for record in records),
            "completion_tokens": sum(record["completion_tokens"] for record in records),
            "total_tokens": sum(record["total_tokens"] for record in records),
            # Active time excludes pauses/restarts made possible by checkpointing.
            "generation_active_wall_seconds": active_generation_seconds,
            "generation_observed_first_to_last_seconds": observed_span,
            "sentences_per_active_second": len(records) / active_generation_seconds,
            # Backward-compatible timing column names from the first driver.
            "generation_parallel_span_seconds": observed_span,
            "generation_gpu_seconds_sum": active_generation_seconds,
            "sentences_per_parallel_second": len(records) / active_generation_seconds,
            "latency_seconds_mean": statistics.mean(latencies) if latencies else None,
            "latency_seconds_p50": percentile(latencies, 50),
            "latency_seconds_p95": percentile(latencies, 95),
            "latency_seconds_p99": percentile(latencies, 99),
            "raw_postprocessed_sentences": repaired_sentences,
            "postprocessing_action_totals": dict(sorted(action_totals.items())),
            "postprocessing_policy": (
                "established light finalize_full_coverage.postprocess_parse; no gold fallback"
                if spec["mode"] == "raw"
                else "none; gold tokens enforced by structured schema"
            ),
            "evaluation_flags": "-x -n -a",
        }
        summaries[condition] = summary
        evaluation_inputs.append((condition, condition_dir))

    with ThreadPoolExecutor(max_workers=args.evaluation_jobs) as executor:
        futures = {
            executor.submit(evaluate, condition, condition_dir): condition
            for condition, condition_dir in evaluation_inputs
        }
        for future in as_completed(futures):
            condition = futures[future]
            summaries[condition]["evaluation"] = future.result()
            write_json(result_root / condition / "summary.json", summaries[condition])
            print(f"evaluated {condition}", flush=True)

    metric_fields = [
        "tokenization_f1_score",
        "tokenization_precision",
        "tokenization_recall",
        "word_accuracy",
        "pos",
        "uas_score",
        "label_score",
        "las_score",
        "e2e_uas_score",
        "e2e_label_score",
        "e2e_las_score",
        "pp_uas_score",
        "pp_label_score",
        "pp_las_score",
    ]
    columns = [
        "condition",
        "model",
        "weight_precision",
        "mode",
        "split",
        "shots",
        "retrieval_pool",
        "sentences",
        *metric_fields,
        "finish_reasons",
        "retried_sentences",
        "structured_retries",
        "completed_on_valid_json",
        "raw_postprocessed_sentences",
        "postprocessing_action_totals",
        "prompt_tokens",
        "completion_tokens",
        "generation_parallel_span_seconds",
        "generation_gpu_seconds_sum",
        "sentences_per_parallel_second",
        "generation_active_wall_seconds",
        "generation_observed_first_to_last_seconds",
        "sentences_per_active_second",
        "latency_seconds_mean",
        "latency_seconds_p50",
        "latency_seconds_p95",
        "latency_seconds_p99",
    ]
    table_path = result_root / os.environ.get(
        "BAREC_BENCH_TABLE_NAME", "fanar2_expanded_barec_results.tsv"
    )
    with table_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for spec in condition_specs():
            summary = summaries[spec["condition"]]
            row = {key: summary.get(key) for key in columns}
            for key in metric_fields:
                row[key] = summary["evaluation"].get(key)
            for key in ("finish_reasons", "postprocessing_action_totals"):
                row[key] = json.dumps(row[key], ensure_ascii=False, sort_keys=True)
            writer.writerow(row)

    worker_summaries = [
        json.loads(
            (
                RUN_ROOT
                / "workers"
                / f"worker_{worker_id}"
                / f"worker_summary_{mode}.json"
            ).read_text(encoding="utf-8")
        )
        for worker_id in worker_ids
        for mode in ("gold", "raw")
    ]
    retrieval_summary = json.loads(
        (RUN_ROOT / "retrieval" / "retrieval_summary.json").read_text(encoding="utf-8")
    )
    retrieval_link = RUN_ROOT / "retrieval"
    retrieval_reused_from = os.environ.get("BAREC_BENCH_RETRIEVAL_REUSED_FROM")
    if not retrieval_reused_from and retrieval_link.is_symlink():
        retrieval_reused_from = str(retrieval_link.resolve())
    pipeline_started_path = RUN_ROOT / "status" / "PIPELINE_STARTED"
    pipeline_started_epoch = min(
        (
            float(item["worker_started_epoch"])
            for item in worker_summaries
            if item.get("worker_started_epoch") is not None
        ),
        default=(
            pipeline_started_path.stat().st_mtime
            if pipeline_started_path.exists()
            else min(all_generation_starts)
        ),
    )
    finalization_finished_epoch = time.time()
    run_summary = {
        "model": MODEL_ID,
        "conditions": len(summaries),
        "requests": sum(summary["sentences"] for summary in summaries.values()),
        "prompt_tokens": sum(summary["prompt_tokens"] for summary in summaries.values()),
        "completion_tokens": sum(summary["completion_tokens"] for summary in summaries.values()),
        "retried_sentences": sum(
            summary["retried_sentences"] for summary in summaries.values()
        ),
        "structured_retries": sum(
            summary["structured_retries"] for summary in summaries.values()
        ),
        "generation_active_wall_seconds_sum": sum(
            summary["generation_active_wall_seconds"] for summary in summaries.values()
        ),
        "generation_observed_first_to_last_seconds": (
            max(all_generation_finishes) - min(all_generation_starts)
        ),
        "worker_mode_elapsed_seconds": {
            f"{item['worker_id']}_{item['mode']}": item["worker_elapsed_seconds"]
            for item in worker_summaries
        },
        # Retain the legacy name, but derive it from every durable generation
        # chunk so a checkpoint-only worker restart cannot understate the run.
        "benchmark_parallel_wall_seconds": (
            max(all_generation_finishes) - min(all_generation_starts)
        ),
        "retrieval_reused": bool(retrieval_reused_from),
        "retrieval_reused_from": retrieval_reused_from,
        "retrieval_elapsed_seconds": (
            0.0 if retrieval_reused_from else retrieval_summary["elapsed_seconds"]
        ),
        "source_retrieval_elapsed_seconds": retrieval_summary["elapsed_seconds"],
        "pipeline_started_epoch": pipeline_started_epoch,
        "benchmark_pipeline_to_finalization_seconds": (
            finalization_finished_epoch - pipeline_started_epoch
        ),
        "benchmark_first_retrieval_to_finalization_seconds": (
            None
            if retrieval_reused_from
            else finalization_finished_epoch - retrieval_summary["started_epoch"]
        ),
        "finalization_started_epoch": started_epoch,
        "finalization_finished_epoch": finalization_finished_epoch,
        "finalization_seconds": finalization_finished_epoch - started_epoch,
        "temperature": 1.0,
        "max_tokens": None,
        "model_context_used": os.environ.get(
            "BAREC_BENCH_CONTEXT_NOTE",
            "full configured context; no max_model_len override",
        ),
        "structured_output": "vLLM XGrammar JSON with fallback disabled",
        "speculative_decoding": os.environ.get(
            "BAREC_BENCH_SPECULATIVE_NOTE",
            "lossless target-verified prompt n-gram; production continuation uses "
            "10 draft tokens and n-grams 3..5; initial checkpoints include measured "
            "decoder tuning and remain represented in per-condition chunk timings",
        ),
        "weight_precision": os.environ.get("BAREC_BENCH_WEIGHT_PRECISION", "bfloat16"),
        "prefix_caching": False,
        "prompt_files": {
            "gold": str(GOLD_PROMPT_PATH),
            "raw": str(RAW_PROMPT_PATH),
        },
        "prompt_sha256": {
            "gold": hashlib.sha256(GOLD_PROMPT_PATH.read_bytes()).hexdigest(),
            "raw": hashlib.sha256(RAW_PROMPT_PATH.read_bytes()).hexdigest(),
        },
        "retrieval_pools": {key: str(path) for key, path in POOL_PATHS.items()},
        "raw_postprocessing_implementation": str(
            Path(__file__).resolve().parent / "finalize_full_coverage.py"
        ),
        "evaluation": f"{CAMEL_CONLL_SCRIPT} -x -n -a",
        "evaluation_alignment_compatibility": (
            "CATiB boundary plus signs are removed for alignment; standalone and "
            "internal lexical plus signs are preserved so real tokens are not dropped"
        ),
        "result_table": str(table_path),
    }
    write_json(result_root / "run_summary.json", run_summary)
    (RUN_ROOT / "status" / "PIPELINE_COMPLETE").parent.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "status" / "PIPELINE_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(table_path, flush=True)


if __name__ == "__main__":
    main()
