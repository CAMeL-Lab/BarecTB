#!/usr/bin/env python3
"""Run the Fanar2 benchmark with streamed, resumable vLLM generation.

The installed vLLM/xgrammar combination can continue sampling PAD or JSON
whitespace after a complete object.  Streaming lets us validate and save the
complete structured object immediately, without imposing a generation limit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from fanar2_benchmark_common import (
    DEPRELS,
    MODEL_ID,
    MODEL_PATH,
    POOL_PATHS,
    RUN_ROOT,
    SPLIT_PATHS,
    append_jsonl,
    build_messages,
    build_schema,
    condition_specs,
    load_system_prompts,
    read_conll,
    read_jsonl,
    write_json,
)


class GrammarDeadState(RuntimeError):
    pass


def finite_metrics(metrics: Any) -> dict[str, float]:
    if metrics is None:
        return {}
    result = {}
    for field in (
        "arrival_time",
        "first_scheduled_time",
        "first_token_time",
        "last_token_time",
        "time_in_queue",
        "finished_time",
        "scheduler_time",
        "model_forward_time",
        "model_execute_time",
    ):
        value = getattr(metrics, field, None)
        if isinstance(value, (float, int)) and math.isfinite(value):
            result[field] = float(value)
    return result


def derived_metrics(
    metrics: dict[str, float], prompt: int, completion: int, local_latency: float
) -> dict[str, float]:
    result = {"latency_seconds": local_latency}
    for name, end, start in (
        ("queue_seconds", "first_scheduled_time", "arrival_time"),
        ("time_to_first_token_seconds", "first_token_time", "arrival_time"),
        ("decode_seconds", "last_token_time", "first_token_time"),
    ):
        if end in metrics and start in metrics and metrics[end] >= metrics[start]:
            result[name] = metrics[end] - metrics[start]
    if local_latency > 0:
        result["total_tokens_per_second"] = (prompt + completion) / local_latency
    if result.get("decode_seconds", 0) > 0:
        result["completion_tokens_per_second"] = completion / result["decode_seconds"]
    return result


def valid_object(mode: str, sentence: Any, value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"parses"}:
        return False
    parses = value["parses"]
    if not isinstance(parses, list) or not parses:
        return False
    if mode == "gold" and len(parses) != len(sentence.tree_tokens):
        return False
    for position, item in enumerate(parses, start=1):
        if not isinstance(item, dict) or set(item) != {"id", "form", "head", "deprel"}:
            return False
        if type(item["id"]) is not int or type(item["head"]) is not int:
            return False
        if not isinstance(item["form"], str) or item["deprel"] not in DEPRELS:
            return False
        if mode == "gold":
            if item["id"] != position or item["form"] != sentence.tree_tokens[position - 1]:
                return False
            if not 0 <= item["head"] <= len(parses):
                return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mode", choices=("gold", "raw"), required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--request-batch-size", type=int, default=24)
    parser.add_argument("--request-prompt-token-budget", type=int, default=110000)
    parser.add_argument("--structured-backend", default="xgrammar")
    parser.add_argument("--max-num-seqs", type=int, default=24)
    parser.add_argument("--max-num-batched-tokens", type=int, default=131072)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--quantization")
    parser.add_argument("--load-format")
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--dead-state-pad-run", type=int, default=8)
    parser.add_argument("--max-dead-state-retries", type=int, default=20)
    parser.add_argument("--ngram-speculative-tokens", type=int, default=0)
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    for path in (*POOL_PATHS.values(), *SPLIT_PATHS.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.v1.engine.async_llm import AsyncLLM

    worker_root = RUN_ROOT / "workers" / f"worker_{args.worker_id}"
    worker_root.mkdir(parents=True, exist_ok=True)
    worker_started_epoch = time.time()
    worker_started = time.perf_counter()
    split_data = {key: read_conll(path) for key, path in SPLIT_PATHS.items()}
    train_data = {key: read_conll(path) for key, path in POOL_PATHS.items()}
    prompts = load_system_prompts()

    engine_args_dict = {
        "model": str(MODEL_PATH),
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": "bfloat16",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        # Prefix-cache hits interact incorrectly with structured mask row
        # placement in this installed vLLM build.
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "seed": 0,
        "structured_outputs_config": {
            "backend": args.structured_backend,
            "disable_fallback": True,
            "disable_any_whitespace": args.mode == "gold",
        },
        "disable_log_stats": True,
        "enable_log_requests": False,
    }
    if args.max_model_len is not None:
        engine_args_dict["max_model_len"] = args.max_model_len
    if args.quantization:
        engine_args_dict["quantization"] = args.quantization
    if args.load_format:
        engine_args_dict["load_format"] = args.load_format
    if args.enable_expert_parallel:
        engine_args_dict["enable_expert_parallel"] = True
    if args.language_model_only:
        engine_args_dict["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if args.ngram_speculative_tokens > 0:
        engine_args_dict["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": args.ngram_speculative_tokens,
            "prompt_lookup_min": 3,
            "prompt_lookup_max": 5,
        }
    # vLLM mutates some nested EngineArgs values (notably the speculative
    # config) into runtime model objects.  Preserve a JSON-safe launch record
    # before handing the arguments to the engine.
    engine_args_summary = json.loads(
        json.dumps(engine_args_dict, ensure_ascii=False, default=str)
    )
    # SamplingParams below intentionally uses max_tokens=None.  max_model_len
    # is an engine context allocation, not a completion-token cap.
    init_started = time.perf_counter()
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_args_dict))
    engine_initialization_seconds = time.perf_counter() - init_started
    tokenizer = engine.processor.tokenizer
    if tokenizer is None:
        raise RuntimeError("Tokenizer was not initialized")
    pad_token_id = tokenizer.pad_token_id
    print(f"worker={args.worker_id} model initialized in {engine_initialization_seconds:.3f}s", flush=True)

    async def generate_valid(
        condition: str,
        index: int,
        mode: str,
        sentence: Any,
        prompt_token_ids: list[int],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        failures = []
        for attempt in range(1, args.max_dead_state_retries + 1):
            request_id = f"{condition}-{index}-a{attempt}-{time.time_ns()}"
            sampling = SamplingParams(
                temperature=1.0,
                max_tokens=None,
                structured_outputs=StructuredOutputsParams(
                    json=schema,
                    disable_fallback=True,
                ),
            )
            started_epoch = time.time()
            started = time.perf_counter()
            latest = None
            generator = engine.generate(
                {"prompt_token_ids": prompt_token_ids}, sampling, request_id
            )
            try:
                async for output in generator:
                    latest = output
                    candidate = output.outputs[0]
                    token_ids = list(candidate.token_ids or [])
                    trailing_pad = 0
                    if pad_token_id is not None:
                        for token_id in reversed(token_ids):
                            if token_id != pad_token_id:
                                break
                            trailing_pad += 1
                    text = candidate.text
                    trailing_whitespace = len(text) - len(text.rstrip())
                    if text.rstrip().endswith("}"):
                        try:
                            parsed = json.loads(text)
                        except json.JSONDecodeError:
                            parsed = None
                        if (
                            parsed is not None
                            and valid_object(mode, sentence, parsed)
                            and (
                                output.finished
                                or trailing_pad > 0
                                or trailing_whitespace > 0
                            )
                        ):
                            elapsed = time.perf_counter() - started
                            completed_by_engine = bool(output.finished)
                            if not completed_by_engine:
                                await engine.abort(request_id)
                            metrics = finite_metrics(getattr(output, "metrics", None))
                            prompt_tokens = len(output.prompt_token_ids or prompt_token_ids)
                            completion_tokens = len(token_ids)
                            return {
                                "response_text": text,
                                "parsed": parsed,
                                "json_parse_error": None,
                                "finish_reason": candidate.finish_reason or "json_complete",
                                "engine_finish_reason": candidate.finish_reason,
                                "stop_reason": candidate.stop_reason,
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens,
                                "request_metrics": metrics,
                                "inference_metrics": derived_metrics(
                                    metrics, prompt_tokens, completion_tokens, elapsed
                                ),
                                "request_started_epoch": started_epoch,
                                "request_finished_epoch": time.time(),
                                "request_elapsed_seconds": elapsed,
                                "engine_attempts": attempt,
                                "completed_on_valid_json": not completed_by_engine,
                                "prior_attempt_failures": failures,
                            }

                    # XGrammar deliberately permits whitespace for the raw-text
                    # schema.  Retry if the model stalls on grammatical whitespace
                    # before completing the JSON; this is not an output-length cap.
                    if trailing_whitespace >= 64:
                        raise GrammarDeadState(
                            f"{request_id}: {trailing_whitespace} trailing whitespace "
                            f"characters before valid JSON; completion_tokens={len(token_ids)}"
                        )

                    if trailing_pad >= args.dead_state_pad_run:
                        raise GrammarDeadState(
                            f"{request_id}: {trailing_pad} trailing PAD tokens; "
                            f"completion_tokens={len(token_ids)}; "
                            f"text_tail={candidate.text[-160:]!r}"
                        )

                reason = "engine finished without a schema-valid JSON object"
                if latest is not None:
                    reason += f" (finish_reason={latest.outputs[0].finish_reason!r})"
                failures.append(reason)
            except GrammarDeadState as error:
                failures.append(str(error))
                await engine.abort(request_id)
            finally:
                await generator.aclose()
        raise RuntimeError(
            f"{condition} sentence {index}: exhausted structured-decoding retries: "
            + " | ".join(failures)
        )

    condition_summaries = {}
    try:
        for spec in (item for item in condition_specs() if item["mode"] == args.mode):
            condition = spec["condition"]
            sentences = split_data[spec["split"]]
            selected_indices = [
                index for index in range(len(sentences)) if index % args.workers == args.worker_id
            ]
            response_path = worker_root / condition / "responses.jsonl"
            existing_records = read_jsonl(response_path)
            existing_by_index = {
                int(record["sentence_index"]): record for record in existing_records
            }
            if len(existing_by_index) != len(existing_records):
                raise ValueError(f"{response_path}: duplicate sentence indices")
            pending = [index for index in selected_indices if index not in existing_by_index]

            matches = None
            pool_sentences = None
            if spec["shots"]:
                retrieval_path = (
                    RUN_ROOT
                    / "retrieval"
                    / f"{spec['pool']}_{spec['split']}_top10_chrfpp.json"
                )
                retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
                matches = retrieval["matches"]
                if len(matches) != len(sentences):
                    raise ValueError(f"{retrieval_path}: query count mismatch")
                pool_sentences = train_data[spec["pool"]]

            timing_path = worker_root / condition / "timing.json"
            timing = (
                json.loads(timing_path.read_text(encoding="utf-8"))
                if timing_path.exists()
                else {"sessions": [], "chunks": []}
            )
            session = {
                "started_epoch": time.time(),
                "pending_at_start": len(pending),
                "completed_before_start": len(existing_records),
                "generation_seconds": 0.0,
                "generated": 0,
            }
            print(
                f"worker={args.worker_id} condition={condition} pending={len(pending)} "
                f"total={len(selected_indices)}",
                flush=True,
            )

            # Keep a rolling set of requests active.  Unlike a gather barrier,
            # this immediately refills a freed sequence slot, while the summed
            # prompt-token budget preserves room for growing completions.
            offset = 0
            active: dict[asyncio.Task[dict[str, Any]], tuple[int, int]] = {}
            active_prompt_tokens = 0
            prepared: tuple[int, list[int]] | None = None
            checkpoint_buffer: list[tuple[int, int, dict[str, Any]]] = []
            checkpoint_size = min(8, args.request_batch_size)
            chunk_started_epoch = time.time()
            chunk_started = time.perf_counter()

            while offset < len(pending) or active:
                while offset < len(pending) and len(active) < args.request_batch_size:
                    if prepared is None:
                        index = pending[offset]
                        sentence = sentences[index]
                        shots = []
                        if matches is not None and pool_sentences is not None:
                            shots = [
                                pool_sentences[item["train_index"]]
                                for item in matches[index]
                            ][:10]
                        conversation = build_messages(
                            spec["mode"], sentence, shots, prompts[spec["mode"]]
                        )
                        prompt_ids = list(
                            tokenizer.apply_chat_template(
                                conversation,
                                tokenize=True,
                                add_generation_prompt=True,
                            )
                        )
                        if (
                            args.request_prompt_token_budget > 0
                            and len(prompt_ids) > args.request_prompt_token_budget
                        ):
                            raise ValueError(
                                f"{condition} sentence {index}: prompt has "
                                f"{len(prompt_ids)} tokens, above budget "
                                f"{args.request_prompt_token_budget}"
                            )
                        prepared = (index, prompt_ids)

                    index, prompt_ids = prepared
                    if (
                        active
                        and args.request_prompt_token_budget > 0
                        and active_prompt_tokens + len(prompt_ids)
                        > args.request_prompt_token_budget
                    ):
                        break
                    task = asyncio.create_task(
                        generate_valid(
                            condition,
                            index,
                            spec["mode"],
                            sentences[index],
                            prompt_ids,
                            build_schema(spec["mode"], sentences[index]),
                        )
                    )
                    active[task] = (index, len(prompt_ids))
                    active_prompt_tokens += len(prompt_ids)
                    prepared = None
                    offset += 1

                if not active:
                    raise RuntimeError(f"{condition}: scheduler could not admit a request")

                done, _ = await asyncio.wait(
                    active, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    index, prompt_length = active.pop(task)
                    active_prompt_tokens -= prompt_length
                    checkpoint_buffer.append((index, prompt_length, task.result()))

                final_checkpoint = offset == len(pending) and not active
                if len(checkpoint_buffer) < checkpoint_size and not final_checkpoint:
                    continue

                checkpoint_buffer.sort(key=lambda item: item[0])
                new_records = []
                for index, _, result in checkpoint_buffer:
                    new_records.append(
                        {
                            "model": MODEL_ID,
                            "condition": condition,
                            "mode": spec["mode"],
                            "split": spec["split"],
                            "shots": spec["shots"],
                            "retrieval_pool": spec["pool"],
                            "worker_id": args.worker_id,
                            "sentence_index": index,
                            "sentence_number": index + 1,
                            "text": sentences[index].text,
                            "gold_tokens": sentences[index].tree_tokens,
                            "retrieved": matches[index] if matches is not None else [],
                            **result,
                        }
                    )
                append_jsonl(response_path, new_records)
                chunk_seconds = time.perf_counter() - chunk_started
                indices = [item[0] for item in checkpoint_buffer]
                chunk = {
                    "started_epoch": chunk_started_epoch,
                    "finished_epoch": time.time(),
                    "elapsed_seconds": chunk_seconds,
                    "first_sentence_index": min(indices),
                    "last_sentence_index": max(indices),
                    "requests": len(indices),
                    "prompt_tokens": sum(x["prompt_tokens"] for x in new_records),
                    "completion_tokens": sum(x["completion_tokens"] for x in new_records),
                    "pretokenized_prompt_tokens": sum(
                        item[1] for item in checkpoint_buffer
                    ),
                    "structured_retries": sum(
                        x["engine_attempts"] - 1 for x in new_records
                    ),
                    "json_complete_aborts": sum(
                        bool(x["completed_on_valid_json"]) for x in new_records
                    ),
                }
                timing.setdefault("chunks", []).append(chunk)
                session["generation_seconds"] += chunk_seconds
                session["generated"] += len(new_records)
                session["last_checkpoint_epoch"] = time.time()
                write_json(timing_path, timing | {"active_session": session})
                print(
                    f"worker={args.worker_id} condition={condition} "
                    f"checkpoint={session['generated']}/{len(pending)} "
                    f"seconds={chunk_seconds:.3f} "
                    f"retries={chunk['structured_retries']}",
                    flush=True,
                )
                checkpoint_buffer.clear()
                chunk_started_epoch = time.time()
                chunk_started = time.perf_counter()

            session["finished_epoch"] = time.time()
            timing.setdefault("sessions", []).append(session)
            timing.pop("active_session", None)
            all_records = read_jsonl(response_path)
            timing.update(
                {
                    "condition": condition,
                    "worker_id": args.worker_id,
                    "expected_responses": len(selected_indices),
                    "saved_responses": len(all_records),
                    "finish_reasons": dict(
                        Counter(record.get("finish_reason") for record in all_records)
                    ),
                    "generation_seconds_total": sum(
                        float(chunk["elapsed_seconds"])
                        for chunk in timing.get("chunks", [])
                    ),
                }
            )
            write_json(timing_path, timing)
            if len(all_records) != len(selected_indices):
                raise ValueError(f"{condition}: incomplete output")
            condition_summaries[condition] = timing
    finally:
        engine.shutdown()

    write_json(
        worker_root / f"worker_summary_{args.mode}.json",
        {
            "worker_id": args.worker_id,
            "mode": args.mode,
            "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "model": MODEL_ID,
            "model_path": str(MODEL_PATH),
            "worker_started_epoch": worker_started_epoch,
            "worker_finished_epoch": time.time(),
            "worker_elapsed_seconds": time.perf_counter() - worker_started,
            "engine_initialization_seconds": engine_initialization_seconds,
            "engine_args": engine_args_summary,
            "sampling": {
                "temperature": 1.0,
                "max_tokens": None,
                "note": "Original temperature; no generation-token cap",
            },
            "conditions": condition_summaries,
        },
    )
    (worker_root / f"WORKER_{args.mode.upper()}_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )
    print(f"worker={args.worker_id} mode={args.mode} COMPLETE", flush=True)


if __name__ == "__main__":
    asyncio.run(main_async())
