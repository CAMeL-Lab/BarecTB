#!/usr/bin/env python3
"""Run and evaluate pilot or full BAREC cloud Batch benchmarks.

The task construction is shared with the Fanar-2 expanded-BAREC benchmark so
the system prompts, user prompts, retrieval matches, target splits, and CATiB
labels are identical.  Generation-token and input-token caps are deliberately
absent.  GPT-5.4 uses reasoning effort ``none``; all other sampling controls
are provider defaults.  Both providers use native JSON structured output.
"""

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
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TOOLS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_ROOT))

from fanar2_benchmark_common import (  # noqa: E402
    CAMEL_CONLL_ROOT,
    CAMEL_CONLL_SCRIPT,
    DEPRELS,
    POOL_PATHS,
    SPLIT_PATHS,
    Sentence,
    build_messages,
    condition_specs,
    load_system_prompts,
    read_conll,
)

from finalize_full_coverage import postprocess_parse  # noqa: E402


DEFAULT_RUN_ROOT = Path(
    str(TOOLS_ROOT.parents[1] / "outputs" / "api_batch_pilot")
)
DEFAULT_FULL_RUN_ROOT = Path(
    str(TOOLS_ROOT.parents[1] / "outputs" / "api_batch_full")
)
RETRIEVAL_ROOT = Path(
    str(TOOLS_ROOT.parents[1] / "outputs" / "expanded" / "retrieval")
)
KEY_ROOT = Path(os.environ.get("BAREC_API_KEY_DIR", str(TOOLS_ROOT.parents[1] / "secrets")))
PROVIDERS = ("openai", "gemini")
MODELS = {"openai": "gpt-5.4", "gemini": "gemini-3.5-flash"}
KEY_FILES = {"openai": KEY_ROOT / "gpt.txt", "gemini": KEY_ROOT / "gemini.txt"}
TERMINAL = {
    "openai": {"completed", "failed", "expired", "cancelled"},
    "gemini": {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    },
}

# Official Batch API prices in USD per one million tokens on 2026-09-13.
# GPT-5.4 standard pricing is $2.50/$0.25/$15.00; the Batch API is 50%
# lower.  Use the effective Batch rates here (the pricing page rounds the
# cached-input display to $0.13, while half of $0.25 is exactly $0.125).
PRICES = {
    "openai": {"input": 1.25, "cached_input": 0.125, "output": 7.50},
    "gemini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def to_plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def key_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def load_keys(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not values:
        raise ValueError(f"No API keys in {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate API keys in {path}")
    return values


def sanitize_error(error: BaseException, keys: Iterable[str]) -> str:
    text = f"{type(error).__name__}: {error}"
    for key in keys:
        text = text.replace(key, "<redacted-api-key>")
    return text


def task_key(condition: str, sentence_index: int) -> str:
    return f"{condition}::{sentence_index}"


def generic_schema(mode: str, token_count: int, close_objects: bool) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "form": {"type": "string"},
            "head": {"type": "integer"},
            "deprel": {"type": "string", "enum": list(DEPRELS)},
        },
        "required": ["id", "form", "head", "deprel"],
    }
    root: dict[str, Any] = {
        "type": "object",
        "properties": {
            "parses": {
                "type": "array",
                "description": (
                    f"Exactly {token_count} CATiB token objects in consecutive 1-based "
                    "id order, preserving the supplied token forms."
                    if mode == "gold"
                    else "CATiB tokens in consecutive 1-based id order."
                ),
                "items": item,
            }
        },
        "required": ["parses"],
    }
    if close_objects:
        item["additionalProperties"] = False
        root["additionalProperties"] = False
    return root


def sentence_payload(sentence: Sentence) -> dict[str, Any]:
    return {
        "text": sentence.text,
        "tree_tokens": sentence.tree_tokens,
        "rows": sentence.rows,
    }


def sample_indices(sentence_count: int, full: bool = False) -> list[int]:
    if full:
        return list(range(sentence_count))
    # Two deterministic, separated examples test both an early and a central
    # sentence while retaining the exact same targets across all conditions.
    values = [0, sentence_count // 2]
    if len(set(values)) != 2:
        raise ValueError(f"Split too small for two-sentence pilot: {sentence_count}")
    return values


def prepare(run_root: Path, full: bool = False) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    benchmark_scope = "full" if full else "pilot"
    split_data = {name: read_conll(path) for name, path in SPLIT_PATHS.items()}
    prompts = load_system_prompts()
    train_cache: dict[str, list[Sentence]] = {}
    retrieval_hashes: dict[str, str] = {}
    tasks: list[dict[str, Any]] = []

    for spec in condition_specs():
        sentences = split_data[spec["split"]]
        indices = sample_indices(len(sentences), full=full)
        matches = None
        pool_sentences = None
        retrieval_path = None
        if spec["shots"]:
            retrieval_path = (
                RETRIEVAL_ROOT
                / f"{spec['pool']}_{spec['split']}_top10_chrfpp.json"
            )
            payload = json.loads(retrieval_path.read_text(encoding="utf-8"))
            matches = payload["matches"]
            if len(matches) != len(sentences):
                raise ValueError(f"{retrieval_path}: query count mismatch")
            if spec["pool"] not in train_cache:
                train_cache[spec["pool"]] = read_conll(POOL_PATHS[spec["pool"]])
            pool_sentences = train_cache[spec["pool"]]
            retrieval_hashes[str(retrieval_path)] = hashlib.sha256(
                retrieval_path.read_bytes()
            ).hexdigest()

        for index in indices:
            sentence = sentences[index]
            shots: list[Sentence] = []
            selected_matches: list[dict[str, Any]] = []
            if matches is not None and pool_sentences is not None:
                selected_matches = matches[index][:10]
                shots = [pool_sentences[int(item["train_index"])] for item in selected_matches]
            messages = build_messages(
                spec["mode"], sentence, shots, prompts[spec["mode"]]
            )
            task = {
                **spec,
                "benchmark_scope": benchmark_scope,
                "key": task_key(spec["condition"], index),
                "sentence_index": index,
                "sentence_number": index + 1,
                "full_condition_size": len(sentences),
                "gold_sentence": sentence_payload(sentence),
                "retrieved": selected_matches,
                "retrieval_path": str(retrieval_path) if retrieval_path else None,
                "messages": messages,
                "schemas": {
                    "openai": generic_schema(
                        spec["mode"], len(sentence.tree_tokens), close_objects=True
                    ),
                    # Gemini Batch rejects object-closure and numeric/array-bound
                    # keywords in this transport; semantic checks follow generation.
                    "gemini": generic_schema(
                        spec["mode"], len(sentence.tree_tokens), close_objects=False
                    ),
                },
            }
            tasks.append(task)
            # Per-sentence prompt snapshots are useful in the pilot, but the
            # full run has 46k requests and the canonical tasks JSONL already
            # preserves every exact prompt and schema.
            if not full:
                write_json(
                    run_root
                    / "prompts"
                    / spec["condition"]
                    / f"sentence_{index + 1:04d}.json",
                    task,
                )

    expected_conditions = [spec["condition"] for spec in condition_specs()]
    if len(expected_conditions) != 16 or len(set(expected_conditions)) != 16:
        raise ValueError("Expected 16 unique benchmark conditions")
    expected_requests = sum(
        len(split_data[spec["split"]]) if full else 2
        for spec in condition_specs()
    )
    if len(tasks) != expected_requests or len({task["key"] for task in tasks}) != expected_requests:
        raise ValueError(
            f"Expected {expected_requests} unique {benchmark_scope} tasks"
        )
    write_jsonl(run_root / "tasks.jsonl", tasks)
    write_json(
        run_root / "preparation_manifest.json",
        {
            "prepared_at_utc": utc_now(),
            "preparation_seconds": time.time() - started,
            "benchmark_scope": benchmark_scope,
            "scope": (
                "all sentences in each of 16 conditions"
                if full
                else "two deterministic separated sentences per each of 16 conditions"
            ),
            "sample_indices": {
                split: (
                    "all"
                    if full
                    else sample_indices(len(rows))
                )
                for split, rows in split_data.items()
            },
            "conditions": expected_conditions,
            "requests_per_provider": len(tasks),
            "providers": list(PROVIDERS),
            "models": MODELS,
            "prompts": {
                "gold": str(TOOLS_ROOT.parents[1] / "benchmarks" / "prompts" / "gold.yaml"),
                "raw": str(TOOLS_ROOT.parents[1] / "benchmarks" / "prompts" / "raw.yaml"),
            },
            "retrieval_root": str(RETRIEVAL_ROOT),
            "retrieval_sha256": retrieval_hashes,
            "generation_controls": {
                "structured_output": True,
                "input_token_cap_sent": False,
                "output_token_cap_sent": False,
                "temperature_sent": False,
                "top_p_sent": False,
                "top_k_sent": False,
                "gemini_thinking_configuration_sent": False,
                "openai_reasoning": {"effort": "none"},
            },
            "pricing_usd_per_million_tokens": PRICES,
            "pricing_date": "2026-09-13",
            "pricing_tier": "batch",
            "pricing_source": "https://developers.openai.com/api/docs/pricing",
        },
    )
    (run_root / "status").mkdir(exist_ok=True)
    (run_root / "status" / "PREPARED").write_text(utc_now() + "\n", encoding="utf-8")
    print(f"PREPARED {len(tasks)} tasks at {run_root}", flush=True)


def split_messages(task: dict[str, Any]) -> tuple[str, str]:
    messages = task["messages"]
    if len(messages) != 2 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
        raise ValueError(f"Unexpected message layout for {task['key']}")
    return messages[0]["content"], messages[1]["content"]


def openai_batch_row(task: dict[str, Any]) -> dict[str, Any]:
    system_prompt, user_prompt = split_messages(task)
    return {
        "custom_id": task["key"],
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODELS["openai"],
            "instructions": system_prompt,
            "input": user_prompt,
            "reasoning": {"effort": "none"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "catib_dependency_parse",
                    "description": "A CATiB dependency parse in JSON.",
                    "strict": True,
                    "schema": task["schemas"]["openai"],
                }
            },
        },
    }


def gemini_batch_row(task: dict[str, Any]) -> dict[str, Any]:
    system_prompt, user_prompt = split_messages(task)
    return {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "config": {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_schema": task["schemas"]["gemini"],
        },
        "metadata": {"key": task["key"]},
    }


def gemini_file_batch_row(task: dict[str, Any]) -> dict[str, Any]:
    """Return the JSONL wire shape required by Gemini file-backed Batch."""
    system_prompt, user_prompt = split_messages(task)
    return {
        "key": task["key"],
        "request": {
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": task["schemas"]["gemini"],
            },
        },
    }


def preflight_key(provider: str, key: str) -> None:
    if provider == "openai":
        from openai import OpenAI

        OpenAI(api_key=key).models.retrieve(MODELS[provider])
    else:
        from google import genai

        client = genai.Client(api_key=key)
        client.models.get(model=MODELS[provider])


def valid_keys(provider: str, keys: list[str], provider_root: Path) -> list[int]:
    checks: list[dict[str, Any]] = [None] * len(keys)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=len(keys)) as executor:
        futures = {
            executor.submit(preflight_key, provider, key): index
            for index, key in enumerate(keys)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                future.result()
                checks[index] = {
                    "key_line_index": index,
                    "fingerprint": key_fingerprint(keys[index]),
                    "valid_for_model_get": True,
                }
            except Exception as error:
                checks[index] = {
                    "key_line_index": index,
                    "fingerprint": key_fingerprint(keys[index]),
                    "valid_for_model_get": False,
                    "error": sanitize_error(error, keys),
                }
    write_json(provider_root / "key_preflight.json", {"checks": checks})
    usable = [item["key_line_index"] for item in checks if item["valid_for_model_get"]]
    if not usable:
        raise RuntimeError(f"No usable {provider} key for {MODELS[provider]}")
    print(f"{provider}: {len(usable)}/{len(keys)} keys passed model access", flush=True)
    return usable


def initial_jobs(provider: str, tasks: list[dict[str, Any]], provider_root: Path) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[task["condition"]].append(task)
    jobs = []
    request_dir = provider_root / "requests"
    for sequence, spec in enumerate(condition_specs()):
        condition = spec["condition"]
        condition_tasks = sorted(grouped[condition], key=lambda item: item["sentence_index"])
        if not condition_tasks:
            raise ValueError(f"{condition}: no tasks")
        benchmark_scope = condition_tasks[0].get("benchmark_scope", "pilot")
        expected = (
            int(condition_tasks[0]["full_condition_size"])
            if benchmark_scope == "full"
            else 2
        )
        if len(condition_tasks) != expected:
            raise ValueError(f"{condition}: expected {expected} tasks")
        request_path = request_dir / f"{condition}.jsonl"
        if provider == "openai":
            write_jsonl(request_path, (openai_batch_row(task) for task in condition_tasks))
            input_transport = "file"
        else:
            if benchmark_scope == "full":
                # Gemini inline batches are limited to 20 MB. Full condition
                # payloads use the official File API JSONL transport instead.
                write_jsonl(
                    request_path,
                    (gemini_file_batch_row(task) for task in condition_tasks),
                )
                input_transport = "file"
            else:
                # Persist the exact native inline payload used for submission.
                write_jsonl(
                    request_path,
                    (gemini_batch_row(task) for task in condition_tasks),
                )
                input_transport = "inline"
        request_bytes = request_path.stat().st_size
        if provider == "openai" and request_bytes > 200_000_000:
            raise ValueError(
                f"{condition}: OpenAI Batch JSONL exceeds 200 MB ({request_bytes})"
            )
        if provider == "gemini" and request_bytes > 2_000_000_000:
            raise ValueError(
                f"{condition}: Gemini Batch JSONL exceeds 2 GB ({request_bytes})"
            )
        jobs.append(
            {
                "sequence": sequence,
                "provider": provider,
                "model": MODELS[provider],
                "condition": condition,
                "benchmark_scope": benchmark_scope,
                "input_transport": input_transport,
                "request_count": len(condition_tasks),
                "request_file_bytes": request_bytes,
                "request_keys": [task["key"] for task in condition_tasks],
                "request_path": str(request_path),
                "state": "LOCAL_PREPARED",
                "state_history": [{"state": "LOCAL_PREPARED", "at_utc": utc_now()}],
            }
        )
    return jobs


def save_jobs(path: Path, jobs: list[dict[str, Any]]) -> None:
    write_json(path, {"updated_at_utc": utc_now(), "jobs": jobs})


def submit_openai(job: dict[str, Any], key: str) -> tuple[Any, str]:
    from openai import OpenAI

    client = OpenAI(api_key=key)
    with Path(job["request_path"]).open("rb") as stream:
        uploaded = client.files.create(file=stream, purpose="batch")
    remote = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "benchmark": f"expanded-barec-{job.get('benchmark_scope', 'pilot')}",
            "condition": job["condition"],
        },
    )
    return remote, uploaded.id


def submit_gemini(job: dict[str, Any], key: str) -> tuple[Any, str | None]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    if job.get("input_transport") == "file":
        uploaded = client.files.upload(
            file=job["request_path"],
            config=types.UploadFileConfig(
                display_name=(
                    f"barec-{job.get('benchmark_scope', 'full')}-{job['condition']}"
                )[:120],
                mime_type="application/jsonl",
            ),
        )
        source: Any = uploaded.name
        uploaded_file_id = uploaded.name
    else:
        # The SDK's inline representation uses config rather than
        # generation_config.
        source = read_jsonl(Path(job["request_path"]))
        uploaded_file_id = None
    remote = client.batches.create(
        model=MODELS["gemini"],
        src=source,
        config={
            "display_name": (
                f"barec-{job.get('benchmark_scope', 'pilot')}-{job['condition']}"
            )[:120]
        },
    )
    return remote, uploaded_file_id


def remote_name(provider: str, remote: Any) -> str:
    return str(remote.id if provider == "openai" else remote.name)


def remote_state(provider: str, remote: Any) -> str:
    value = remote.status if provider == "openai" else remote.state
    return str(value.name if hasattr(value, "name") else value)


def submit_one(
    provider: str,
    job: dict[str, Any],
    keys: list[str],
    candidate_indices: list[int],
) -> dict[str, Any]:
    job = dict(job)
    failures = []
    offset = int(job["sequence"]) % len(candidate_indices)
    ordered = candidate_indices[offset:] + candidate_indices[:offset]
    for key_index in ordered:
        started = time.perf_counter()
        submitted_at = utc_now()
        try:
            if provider == "openai":
                remote, uploaded_file_id = submit_openai(job, keys[key_index])
            else:
                remote, uploaded_file_id = submit_gemini(job, keys[key_index])
            state = remote_state(provider, remote)
            job.update(
                {
                    "key_line_index": key_index,
                    "key_fingerprint": key_fingerprint(keys[key_index]),
                    "submitted_at_utc": submitted_at,
                    "submission_api_seconds": time.perf_counter() - started,
                    "uploaded_file_id": uploaded_file_id,
                    "remote_job_name": remote_name(provider, remote),
                    "state": state,
                    "last_polled_at_utc": utc_now(),
                    "submission_failures_before_success": failures,
                }
            )
            job["state_history"] = list(job.get("state_history", [])) + [
                {"state": state, "at_utc": utc_now()}
            ]
            return job
        except Exception as error:
            failures.append(
                {
                    "key_fingerprint": key_fingerprint(keys[key_index]),
                    "error": sanitize_error(error, keys),
                }
            )
    job["state"] = "LOCAL_SUBMISSION_FAILED"
    job["submission_failures_before_success"] = failures
    return job


def retrieve_job(provider: str, job: dict[str, Any], key: str) -> Any:
    if provider == "openai":
        from openai import OpenAI

        return OpenAI(api_key=key).batches.retrieve(job["remote_job_name"])
    from google import genai

    client = genai.Client(api_key=key)
    return client.batches.get(name=job["remote_job_name"])


def poll_jobs(
    provider: str,
    jobs: list[dict[str, Any]],
    keys: list[str],
    jobs_path: Path,
    poll_seconds: float,
) -> None:
    while True:
        pending = [
            index
            for index, job in enumerate(jobs)
            if job.get("remote_job_name") and job.get("state") not in TERMINAL[provider]
        ]
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=min(len(pending), 16)) as executor:
            futures = {
                executor.submit(
                    retrieve_job,
                    provider,
                    jobs[index],
                    keys[int(jobs[index]["key_line_index"])],
                ): index
                for index in pending
            }
            for future in as_completed(futures):
                index = futures[future]
                job = jobs[index]
                try:
                    remote = future.result()
                    state = remote_state(provider, remote)
                    if state != job.get("state"):
                        job.setdefault("state_history", []).append(
                            {"state": state, "at_utc": utc_now()}
                        )
                    job["state"] = state
                    job["last_polled_at_utc"] = utc_now()
                    if state in TERMINAL[provider] and "finished_at_utc" not in job:
                        job["finished_at_utc"] = utc_now()
                        job["batch_wall_seconds"] = (
                            parse_utc(job["finished_at_utc"])
                            - parse_utc(job["submitted_at_utc"])
                        ).total_seconds()
                except Exception as error:
                    job["last_poll_error"] = sanitize_error(error, keys)
        save_jobs(jobs_path, jobs)
        counts = Counter(job.get("state", "UNKNOWN") for job in jobs)
        print(f"{provider} POLL {utc_now()} {dict(counts)}", flush=True)
        if any(job.get("state") not in TERMINAL[provider] for job in jobs):
            time.sleep(poll_seconds)


def openai_output_text(body: dict[str, Any]) -> str:
    texts = []
    for item in body.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
    return "".join(texts)


def gemini_output_text(response: Any) -> str:
    raw = to_plain(response) or {}
    texts = []
    for candidate in raw.get("candidates", []) or []:
        for part in (candidate.get("content") or {}).get("parts", []) or []:
            if isinstance(part.get("text"), str):
                texts.append(part["text"])
    return "".join(texts)


def usage_cost(provider: str, response: dict[str, Any]) -> dict[str, Any]:
    if provider == "openai":
        usage = response.get("usage") or {}
        prompt = int(usage.get("input_tokens") or 0)
        output = int(usage.get("output_tokens") or 0)
        cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
        reasoning = int(
            (usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0
        )
    else:
        usage = response.get("usageMetadata") or response.get("usage_metadata") or {}
        prompt = int(usage.get("promptTokenCount") or usage.get("prompt_token_count") or 0)
        output = int(
            usage.get("candidatesTokenCount")
            or usage.get("responseTokenCount")
            or usage.get("candidates_token_count")
            or 0
        )
        cached = int(
            usage.get("cachedContentTokenCount")
            or usage.get("cached_content_token_count")
            or 0
        )
        reasoning = int(
            usage.get("thoughtsTokenCount") or usage.get("thoughts_token_count") or 0
        )
    uncached = max(0, prompt - cached)
    rates = PRICES[provider]
    estimated_cost = (
        uncached * rates["input"]
        + cached * rates["cached_input"]
        + output * rates["output"]
        + reasoning * rates["output"]
    ) / 1_000_000
    return {
        "input_tokens": prompt,
        "cached_input_tokens": cached,
        "visible_output_tokens": output,
        "reasoning_or_thinking_tokens": reasoning,
        "billable_output_tokens": output + reasoning,
        "total_tokens": prompt + output + reasoning,
        "estimated_batch_cost_usd": estimated_cost,
        "usage_metadata": usage,
    }


def collect_openai(job: dict[str, Any], key: str, raw_path: Path) -> list[dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(api_key=key)
    remote = client.batches.retrieve(job["remote_job_name"])
    if remote.status not in TERMINAL["openai"]:
        raise RuntimeError(f"{job['condition']} is still {remote.status}")
    request_counts = remote.request_counts
    completed_count = int(getattr(request_counts, "completed", 0) or 0)
    if remote.output_file_id:
        payload = client.files.content(remote.output_file_id).text
    elif completed_count:
        # A terminal batch with completed requests should expose its partial
        # output file.  Refuse to replace those successful results with
        # placeholders if the file is only temporarily unavailable.
        raise RuntimeError(
            f"{job['condition']}: terminal {remote.status} batch reports "
            f"{completed_count} completed requests but has no output file"
        )
    else:
        payload = ""
    raw_path.write_text(payload, encoding="utf-8")
    if remote.error_file_id:
        (raw_path.parent / f"{job['condition']}.errors.jsonl").write_text(
            client.files.content(remote.error_file_id).text, encoding="utf-8"
        )
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def collect_gemini(job: dict[str, Any], key: str, raw_path: Path) -> list[dict[str, Any]]:
    from google import genai

    client = genai.Client(api_key=key)
    remote = client.batches.get(name=job["remote_job_name"])
    if remote.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"{job['condition']} ended in {remote.state.name}")
    if not remote.dest:
        raise RuntimeError(f"{job['condition']}: missing batch destination")
    if job.get("input_transport") == "file":
        if not remote.dest.file_name:
            raise RuntimeError(f"{job['condition']}: missing output file")
        payload = client.files.download(file=remote.dest.file_name)
        raw_path.write_bytes(payload)
        raw_rows = [
            json.loads(line)
            for line in payload.decode("utf-8").splitlines()
            if line.strip()
        ]
        if len(raw_rows) != len(job["request_keys"]):
            raise RuntimeError(
                f"{job['condition']}: response count mismatch "
                f"({len(raw_rows)} != {len(job['request_keys'])})"
            )
        results = []
        for position, raw in enumerate(raw_rows):
            request_key = raw.get("key") or job["request_keys"][position]
            # File-backed output normally wraps each response with its input
            # key. Retain a fallback for the documented same-order bare
            # GenerateContentResponse representation.
            if "response" in raw or "error" in raw:
                response = raw.get("response")
                error = raw.get("error")
            else:
                response = raw
                error = None
            results.append(
                {"key": request_key, "response": response, "error": error}
            )
        return results

    if remote.dest.inlined_responses is None:
        raise RuntimeError(f"{job['condition']}: missing inline responses")
    if len(remote.dest.inlined_responses) != len(job["request_keys"]):
        raise RuntimeError(f"{job['condition']}: response count mismatch")
    raw_rows = [to_plain(value) for value in remote.dest.inlined_responses]
    write_jsonl(raw_path, raw_rows)
    results = []
    for request_key, inline, raw in zip(
        job["request_keys"], remote.dest.inlined_responses, raw_rows
    ):
        results.append(
            {
                "key": request_key,
                "response": to_plain(inline.response) if inline.response else None,
                "error": to_plain(inline.error) if inline.error else raw.get("error"),
            }
        )
    return results


def build_records(
    provider: str,
    jobs: list[dict[str, Any]],
    keys: list[str],
    tasks: list[dict[str, Any]],
    provider_root: Path,
) -> list[dict[str, Any]]:
    by_task = {task["key"]: task for task in tasks}
    records = []
    raw_root = provider_root / "raw_batch_outputs"
    raw_root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        key = keys[int(job["key_line_index"])]
        raw_path = raw_root / f"{job['condition']}.jsonl"
        if provider == "openai":
            rows = collect_openai(job, key, raw_path)
            normalized = {}
            for row in rows:
                response = row.get("response") or {}
                body = response.get("body") or {}
                normalized[row.get("custom_id")] = {
                    "response": body,
                    "http_status": response.get("status_code"),
                    "error": row.get("error"),
                    "response_text": openai_output_text(body),
                    "finish_reason": body.get("status"),
                }
            unexpected = set(normalized) - set(job["request_keys"])
            if unexpected:
                raise ValueError(
                    f"{job['condition']}: unexpected result keys: "
                    f"{sorted(unexpected)[:10]}"
                )
            # OpenAI makes successful partial outputs available when a Batch
            # ends as expired/failed/cancelled.  Preserve those outputs and
            # represent only absent requests as invalid records so the repair
            # pass resubmits exactly the missing subset.
            for missing_key in set(job["request_keys"]) - set(normalized):
                normalized[missing_key] = {
                    "response": {},
                    "http_status": None,
                    "error": {
                        "type": "missing_terminal_batch_output",
                        "batch_state": job.get("state"),
                        "message": "Request absent from terminal Batch output; retry required",
                    },
                    "response_text": "",
                    "finish_reason": f"batch_{job.get('state', 'terminal')}",
                }
        else:
            rows = collect_gemini(job, key, raw_path)
            normalized = {}
            for row in rows:
                response = row.get("response") or {}
                candidates = response.get("candidates") or []
                finish_reason = (
                    candidates[0].get("finishReason")
                    or candidates[0].get("finish_reason")
                    if candidates
                    else None
                )
                normalized[row.get("key")] = {
                    "response": response,
                    "http_status": None,
                    "error": row.get("error"),
                    "response_text": gemini_output_text(response),
                    "finish_reason": finish_reason,
                }
        if set(normalized) != set(job["request_keys"]):
            raise ValueError(f"{job['condition']}: result key mismatch")
        for key_name in job["request_keys"]:
            task = by_task[key_name]
            output = normalized[key_name]
            response = output["response"]
            text = output["response_text"]
            try:
                parsed = json.loads(text)
                parse_error = None
            except Exception as error:
                parsed = None
                parse_error = f"{type(error).__name__}: {error}"
            records.append(
                {
                    "provider": provider,
                    "model": MODELS[provider],
                    "key": key_name,
                    "condition": task["condition"],
                    "mode": task["mode"],
                    "split": task["split"],
                    "shots": task["shots"],
                    "retrieval_pool": task["pool"],
                    "sentence_index": task["sentence_index"],
                    "sentence_number": task["sentence_number"],
                    "text": task["gold_sentence"]["text"],
                    "gold_tokens": task["gold_sentence"]["tree_tokens"],
                    "retrieved": task["retrieved"],
                    "response_text": text,
                    "parsed": parsed,
                    "json_parse_error": parse_error,
                    "finish_reason": output["finish_reason"],
                    "http_status": output["http_status"],
                    "provider_error": output["error"],
                    **usage_cost(provider, response),
                }
            )
    records.sort(key=lambda row: (row["condition"], row["sentence_index"]))
    if len(records) != len(tasks):
        raise ValueError(
            f"{provider}: expected {len(tasks)} records, got {len(records)}"
        )
    return records


def run_provider(run_root: Path, provider: str, poll_seconds: float) -> None:
    if not (run_root / "status" / "PREPARED").exists():
        raise FileNotFoundError("Run prepare first")
    provider_root = run_root / provider
    provider_root.mkdir(parents=True, exist_ok=True)
    tasks = read_jsonl(run_root / "tasks.jsonl")
    keys = load_keys(KEY_FILES[provider])
    usable = valid_keys(provider, keys, provider_root)
    jobs_path = provider_root / "jobs.json"
    if jobs_path.exists():
        jobs = json.loads(jobs_path.read_text(encoding="utf-8"))["jobs"]
    else:
        jobs = initial_jobs(provider, tasks, provider_root)
        save_jobs(jobs_path, jobs)

    pending = [index for index, job in enumerate(jobs) if not job.get("remote_job_name")]
    if pending:
        with ThreadPoolExecutor(max_workers=min(len(pending), len(usable))) as executor:
            futures = {
                executor.submit(
                    submit_one, provider, jobs[index], keys, usable
                ): index
                for index in pending
            }
            for future in as_completed(futures):
                index = futures[future]
                jobs[index] = future.result()
                save_jobs(jobs_path, jobs)
                print(
                    f"{provider} SUBMIT {jobs[index]['condition']} "
                    f"{jobs[index].get('state')} "
                    f"{jobs[index].get('remote_job_name', '')}",
                    flush=True,
                )
    failures = [job for job in jobs if not job.get("remote_job_name")]
    if failures:
        raise RuntimeError(
            f"{provider}: {len(failures)} local submission failures; see {jobs_path}"
        )
    poll_jobs(provider, jobs, keys, jobs_path, poll_seconds)
    bad = [
        job
        for job in jobs
        if job.get("state") not in TERMINAL[provider]
        or job.get("state")
        not in ({"completed"} if provider == "openai" else {"JOB_STATE_SUCCEEDED"})
    ]
    # OpenAI terminal non-success batches can still contain nearly all of
    # their outputs.  build_records preserves those and emits invalid
    # placeholders only for absent requests; repair_cloud_batch_full.py then
    # resubmits just that subset.  Gemini does not expose an equivalent
    # partial-output recovery contract here, so retain its strict behavior.
    if bad and provider != "openai":
        raise RuntimeError(f"{provider}: terminal batch failures: {[x['condition'] for x in bad]}")
    records = build_records(provider, jobs, keys, tasks, provider_root)
    write_jsonl(provider_root / "responses.jsonl", records)
    for condition in {row["condition"] for row in records}:
        write_jsonl(
            provider_root / "conditions" / condition / "responses.jsonl",
            (row for row in records if row["condition"] == condition),
        )
    provider_summary = {
        "provider": provider,
        "model": MODELS[provider],
        "requests": len(records),
        "batch_jobs": len(jobs),
        "successful_batch_jobs": sum(
            job.get("state")
            == ("completed" if provider == "openai" else "JOB_STATE_SUCCEEDED")
            for job in jobs
        ),
        "terminal_non_success_batch_jobs": sum(
            job.get("state")
            != ("completed" if provider == "openai" else "JOB_STATE_SUCCEEDED")
            for job in jobs
        ),
        "missing_main_batch_outputs_for_repair": sum(
            isinstance(record.get("provider_error"), dict)
            and record["provider_error"].get("type")
            == "missing_terminal_batch_output"
            for record in records
        ),
        "input_transports": dict(
            Counter(job.get("input_transport", "unknown") for job in jobs)
        ),
        "earliest_submission_utc": min(job["submitted_at_utc"] for job in jobs),
        "latest_terminal_observation_utc": max(job["finished_at_utc"] for job in jobs),
        "overall_observed_batch_wall_seconds": (
            max(parse_utc(job["finished_at_utc"]) for job in jobs)
            - min(parse_utc(job["submitted_at_utc"]) for job in jobs)
        ).total_seconds(),
        "sum_submission_api_seconds": sum(float(job["submission_api_seconds"]) for job in jobs),
        "usage": sum_usage(records),
        "generation_controls": {
            "structured_output": True,
            "input_token_cap_sent": False,
            "output_token_cap_sent": False,
            "provider_defaults_other_than": (
                {"reasoning": {"effort": "none"}} if provider == "openai" else {}
            ),
        },
    }
    write_json(provider_root / "provider_summary.json", provider_summary)
    (run_root / "status" / f"{provider.upper()}_COMPLETE").write_text(
        utc_now() + "\n", encoding="utf-8"
    )
    print(f"{provider} COMPLETE {len(records)}/{len(tasks)}", flush=True)


def stage_provider(run_root: Path, provider: str) -> None:
    """Build provider request files and job manifests without API access."""
    if not (run_root / "status" / "PREPARED").exists():
        raise FileNotFoundError("Run prepare first")
    provider_root = run_root / provider
    provider_root.mkdir(parents=True, exist_ok=True)
    jobs_path = provider_root / "jobs.json"
    if jobs_path.exists():
        jobs = json.loads(jobs_path.read_text(encoding="utf-8"))["jobs"]
        if any(job.get("remote_job_name") for job in jobs):
            raise RuntimeError(
                f"{provider}: refusing to restage after remote submission"
            )
    tasks = read_jsonl(run_root / "tasks.jsonl")
    jobs = initial_jobs(provider, tasks, provider_root)
    save_jobs(jobs_path, jobs)
    total_bytes = sum(int(job["request_file_bytes"]) for job in jobs)
    print(
        f"{provider} STAGED {len(jobs)} jobs, {len(tasks)} requests, "
        f"{total_bytes} bytes",
        flush=True,
    )


def run_canary(
    run_root: Path, provider: str, condition: str, poll_seconds: float
) -> None:
    """Run and validate one real condition before releasing remaining jobs."""
    provider_root = run_root / provider
    jobs_path = provider_root / "jobs.json"
    if not jobs_path.exists():
        raise FileNotFoundError(f"Stage {provider} first")
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))["jobs"]
    indices = [
        index for index, job in enumerate(jobs) if job["condition"] == condition
    ]
    if len(indices) != 1:
        raise ValueError(f"Expected one {condition} job, found {len(indices)}")
    index = indices[0]
    keys = load_keys(KEY_FILES[provider])
    usable = valid_keys(provider, keys, provider_root)
    if not jobs[index].get("remote_job_name"):
        jobs[index] = submit_one(provider, jobs[index], keys, usable)
        save_jobs(jobs_path, jobs)
    if not jobs[index].get("remote_job_name"):
        raise RuntimeError(f"{provider}/{condition}: canary submission failed")
    poll_jobs(provider, jobs, keys, jobs_path, poll_seconds)
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))["jobs"]
    job = jobs[index]
    success = "completed" if provider == "openai" else "JOB_STATE_SUCCEEDED"
    if job.get("state") != success:
        raise RuntimeError(
            f"{provider}/{condition}: canary ended in {job.get('state')}"
        )
    raw_root = provider_root / "raw_batch_outputs"
    raw_root.mkdir(parents=True, exist_ok=True)
    if provider == "openai":
        rows = collect_openai(
            job,
            keys[int(job["key_line_index"])],
            raw_root / f"{condition}.jsonl",
        )
        normalized = {
            row.get("custom_id"): (row.get("response") or {}).get("body") or {}
            for row in rows
        }
        texts = {
            key: openai_output_text(response)
            for key, response in normalized.items()
        }
    else:
        rows = collect_gemini(
            job,
            keys[int(job["key_line_index"])],
            raw_root / f"{condition}.jsonl",
        )
        normalized = {row.get("key"): row.get("response") or {} for row in rows}
        texts = {
            key: gemini_output_text(response)
            for key, response in normalized.items()
        }
    expected_keys = set(job["request_keys"])
    errors = []
    if set(texts) != expected_keys:
        errors.append(
            f"result key mismatch: {len(texts)} vs {len(expected_keys)}"
        )
    task_by_key = {}
    with (run_root / "tasks.jsonl").open(encoding="utf-8") as source:
        for line in source:
            task = json.loads(line)
            if task["condition"] == condition:
                task_by_key[task["key"]] = task
    for key in sorted(expected_keys):
        try:
            parsed = json.loads(texts[key])
        except Exception as error:
            errors.append(f"{key}: JSON {type(error).__name__}: {error}")
            continue
        semantic = validate_serializable(parsed)
        if task_by_key[key]["mode"] == "gold" and not semantic:
            forms = [item["form"] for item in parsed["parses"]]
            expected_forms = task_by_key[key]["gold_sentence"]["tree_tokens"]
            if forms != expected_forms:
                semantic.append("gold-tokenized forms differ from supplied tokens")
        if semantic:
            errors.append(f"{key}: {'; '.join(semantic)}")
        if len(errors) > 100:
            break
    canary_summary = {
        "completed_at_utc": utc_now(),
        "provider": provider,
        "condition": condition,
        "remote_job_name": job["remote_job_name"],
        "state": job["state"],
        "requests_expected": len(expected_keys),
        "responses_received": len(rows),
        "all_structured_outputs_valid": not errors,
        "errors": errors,
    }
    write_json(provider_root / "canary_summary.json", canary_summary)
    if errors:
        raise RuntimeError(
            f"{provider}/{condition}: canary validation failed: {errors[:5]}"
        )
    print(
        f"{provider} CANARY_OK {condition} {len(rows)}/{len(expected_keys)}",
        flush=True,
    )


def sum_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "input_tokens",
        "cached_input_tokens",
        "visible_output_tokens",
        "reasoning_or_thinking_tokens",
        "billable_output_tokens",
        "total_tokens",
    )
    result = {field: sum(int(row.get(field) or 0) for row in records) for field in fields}
    # Recompute from token totals and the current effective Batch rates instead
    # of trusting the per-record estimate.  This also makes finalization repair
    # older normalized records if a pricing constant was corrected after a
    # long-running remote Batch job had already started.
    providers = {str(row.get("provider")) for row in records if row.get("provider")}
    if len(providers) == 1 and next(iter(providers)) in PRICES:
        rates = PRICES[next(iter(providers))]
        uncached = max(0, result["input_tokens"] - result["cached_input_tokens"])
        result["estimated_batch_cost_usd"] = (
            uncached * rates["input"]
            + result["cached_input_tokens"] * rates["cached_input"]
            + result["billable_output_tokens"] * rates["output"]
        ) / 1_000_000
    else:
        result["estimated_batch_cost_usd"] = sum(
            float(row.get("estimated_batch_cost_usd") or 0.0) for row in records
        )
    return result


def _align_gold_to_predicted(
    gold: list[str], predicted: list[str]
) -> tuple[list[int | None], list[int]]:
    """Minimum-edit alignment returning a predicted index per gold token."""
    n, m = len(gold), len(predicted)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    moves = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0] = i
        moves[i][0] = "delete_gold"
    for j in range(1, m + 1):
        costs[0][j] = j
        moves[0][j] = "insert_predicted"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = [
                (
                    costs[i - 1][j - 1]
                    + (0 if gold[i - 1] == predicted[j - 1] else 1),
                    0,
                    "pair",
                ),
                (costs[i][j - 1] + 1, 1, "insert_predicted"),
                (costs[i - 1][j] + 1, 2, "delete_gold"),
            ]
            cost, _, move = min(candidates)
            costs[i][j] = cost
            moves[i][j] = move
    mapping: list[int | None] = [None] * n
    extras = []
    i, j = n, m
    while i or j:
        move = moves[i][j]
        if move == "pair":
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif move == "insert_predicted":
            extras.append(j - 1)
            j -= 1
        elif move == "delete_gold":
            i -= 1
        else:
            raise RuntimeError(f"Invalid alignment backtrace at {i}, {j}")
    extras.reverse()
    return mapping, extras


def postprocess_gold_parse(
    parsed: dict[str, Any], gold_tokens: list[str]
) -> tuple[dict[str, Any], list[str]]:
    """Restore copy-only gold fields without fabricating missing predictions."""
    if not isinstance(parsed, dict) or not isinstance(parsed.get("parses"), list):
        return parsed, []
    items = parsed["parses"]
    predicted_forms = [item.get("form") for item in items]
    if not all(isinstance(form, str) for form in predicted_forms):
        return parsed, []
    mapping, extras = _align_gold_to_predicted(gold_tokens, predicted_forms)
    # A missing gold position has no model-produced head/label to preserve.
    if any(index is None for index in mapping):
        return parsed, []

    # A few multiple-choice sentences consistently make otherwise exact model
    # outputs refer to the final answer token as n+1 (occasionally n+2).  The
    # provider repeats that numbering slip across Batch retries.  When and only
    # when the predicted token sequence is already an exact copy of the gold
    # sequence, interpret an overflow head as the final token.  This keeps the
    # correction deterministic and auditable while leaving every in-range head
    # and every non-exact alignment untouched.
    exact_token_alignment = (
        len(items) == len(gold_tokens)
        and mapping == list(range(len(gold_tokens)))
        and predicted_forms == gold_tokens
    )

    pred_to_gold = {
        int(predicted_index) + 1: gold_index + 1
        for gold_index, predicted_index in enumerate(mapping)
    }

    def remap_head(head: Any) -> Any:
        if type(head) is not int or head == 0:
            return head
        seen = set()
        while head not in pred_to_gold:
            if head > len(items) and exact_token_alignment:
                return len(gold_tokens)
            if head in seen or not 1 <= head <= len(items):
                return head
            seen.add(head)
            head = items[head - 1].get("head")
            if type(head) is not int or head == 0:
                return head
        return pred_to_gold[head]

    repaired = []
    changed_ids = False
    changed_forms = False
    changed_heads = False
    for gold_index, predicted_index in enumerate(mapping, start=1):
        original = items[int(predicted_index)]
        head = remap_head(original.get("head"))
        changed_ids |= original.get("id") != gold_index
        changed_forms |= original.get("form") != gold_tokens[gold_index - 1]
        changed_heads |= head != original.get("head")
        repaired.append(
            {
                "id": gold_index,
                "form": gold_tokens[gold_index - 1],
                "head": head,
                "deprel": original.get("deprel"),
            }
        )
    actions = []
    if extras:
        actions.append(f"gold_drop_extra_items:{len(extras)}")
    if changed_ids:
        actions.append("gold_restore_ids")
    if changed_forms:
        actions.append("gold_restore_forms")
    if changed_heads:
        actions.append("gold_remap_heads")
    return {"parses": repaired}, actions


def validate_serializable(parsed: Any) -> list[str]:
    if not isinstance(parsed, dict) or set(parsed) != {"parses"}:
        return ["root must contain only parses"]
    items = parsed.get("parses")
    if not isinstance(items, list) or not items:
        return ["missing or empty parses array"]
    errors = []
    for expected, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            errors.append(f"item {expected} is not an object")
            continue
        if set(item) != {"id", "form", "head", "deprel"}:
            errors.append(f"item {expected} fields differ")
        if item.get("id") != expected:
            errors.append(f"item {expected} id={item.get('id')!r}")
        form = item.get("form")
        if not isinstance(form, str) or not form or any(char.isspace() for char in form):
            errors.append(f"item {expected} invalid form")
        head = item.get("head")
        if type(head) is not int or not 0 <= head <= len(items):
            errors.append(f"item {expected} invalid head={head!r}")
        if item.get("deprel") not in DEPRELS:
            errors.append(f"item {expected} invalid deprel")
    return errors


def as_sentence(payload: dict[str, Any]) -> Sentence:
    return Sentence(
        text=payload["text"],
        tree_tokens=payload["tree_tokens"],
        rows=payload["rows"],
    )


def format_gold(sentences: list[Sentence]) -> str:
    chunks = []
    for sentence in sentences:
        chunks.append(f"# text = {sentence.text}\n")
        chunks.append("# treeTokens = " + " ".join(sentence.tree_tokens) + "\n")
        chunks.extend("\t".join(row) + "\n" for row in sentence.rows)
        chunks.append("\n")
    return "".join(chunks)


def format_prediction(sentences: list[Sentence], parses: list[dict[str, Any]]) -> str:
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


def evaluate_condition(
    condition: str, condition_root: Path, basename: str = "pilot.conllx"
) -> dict[str, str]:
    evaluation_root = condition_root / "evaluation"
    gold_root = evaluation_root / "gold"
    parsed_root = evaluation_root / "parsed"
    gold_root.mkdir(parents=True, exist_ok=True)
    parsed_root.mkdir(parents=True, exist_ok=True)
    (gold_root / basename).write_text(
        (condition_root / "gold.conllx").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (parsed_root / basename).write_text(
        (condition_root / "predictions.conllx").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(CAMEL_CONLL_SCRIPT),
        "--gold",
        str(gold_root / basename),
        "--parsed",
        str(parsed_root / basename),
        "-x",
        "-n",
        "-a",
        "--output_path",
        str(evaluation_root),
        condition,
    ]
    with (evaluation_root / "camel_conll.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=CAMEL_CONLL_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    with (evaluation_root / f"{condition}.tsv").open(
        encoding="utf-8-sig", newline=""
    ) as source:
        return next(csv.DictReader(source, delimiter="\t"))


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def finalize(
    run_root: Path, providers: tuple[str, ...] = PROVIDERS
) -> None:
    """Postprocess and evaluate one provider or the complete provider set.

    A provider-only finalization writes clearly suffixed preview artifacts and
    deliberately does not create the global PIPELINE_COMPLETE marker.  This lets
    a completed provider be evaluated while another provider's Batch jobs are
    still running; the normal all-provider finalization remains authoritative.
    """
    if not providers or any(provider not in PROVIDERS for provider in providers):
        raise ValueError(f"Invalid finalization provider set: {providers!r}")
    is_complete_provider_set = set(providers) == set(PROVIDERS)
    manifest = json.loads(
        (run_root / "preparation_manifest.json").read_text(encoding="utf-8")
    )
    benchmark_scope = manifest.get("benchmark_scope", "pilot")
    is_full = benchmark_scope == "full"
    expected_provider_requests = int(manifest["requests_per_provider"])
    evaluation_basename = "full.conllx" if is_full else "pilot.conllx"
    tasks = read_jsonl(run_root / "tasks.jsonl")
    if len(tasks) != expected_provider_requests:
        raise ValueError(
            f"Task manifest has {len(tasks)} rows; expected "
            f"{expected_provider_requests}"
        )
    by_key = {task["key"]: task for task in tasks}
    results_root = run_root / "results"
    summaries = []
    provider_totals = {}
    for provider in providers:
        records = read_jsonl(run_root / provider / "responses.jsonl")
        if len(records) != expected_provider_requests:
            raise ValueError(
                f"{provider}: expected {expected_provider_requests} responses"
            )
        provider_jobs = json.loads(
            (run_root / provider / "jobs.json").read_text(encoding="utf-8")
        )["jobs"]
        jobs_by_condition = {job["condition"]: job for job in provider_jobs}
        provider_summary_path = run_root / provider / "provider_summary.json"
        provider_summary = (
            json.loads(provider_summary_path.read_text(encoding="utf-8"))
            if provider_summary_path.exists()
            else {}
        )
        repair_summary_path = run_root / provider / "repair_summary.json"
        repair_summary = (
            json.loads(repair_summary_path.read_text(encoding="utf-8"))
            if repair_summary_path.exists()
            else {}
        )
        condition_timings = (
            (repair_summary.get("timing") or {}).get("condition_timings") or {}
        )
        initial_wall = (
            max(parse_utc(job["finished_at_utc"]) for job in provider_jobs)
            - min(parse_utc(job["submitted_at_utc"]) for job in provider_jobs)
        ).total_seconds()
        provider_totals[provider] = {
            "usage": sum_usage(records),
            "initial_overall_observed_batch_wall_seconds": float(
                provider_summary.get(
                    "initial_overall_observed_batch_wall_seconds", initial_wall
                )
            ),
            "overall_observed_batch_wall_seconds": float(
                provider_summary.get(
                    "overall_observed_batch_wall_seconds", initial_wall
                )
            ),
            "repair_batch_jobs": int(
                (repair_summary.get("timing") or {}).get("repair_batch_jobs", 0)
            ),
            "repair_requests": int(
                (repair_summary.get("timing") or {}).get("repair_requests", 0)
            ),
        }
        for spec in condition_specs():
            condition = spec["condition"]
            condition_records = sorted(
                (row for row in records if row["condition"] == condition),
                key=lambda row: row["sentence_index"],
            )
            condition_expected = int(
                by_key[condition_records[0]["key"]]["full_condition_size"]
            ) if condition_records else 0
            if not is_full:
                condition_expected = 2
            if len(condition_records) != condition_expected:
                raise ValueError(
                    f"{provider}/{condition}: expected {condition_expected} records"
                )
            condition_root = results_root / provider / condition
            condition_root.mkdir(parents=True, exist_ok=True)
            output_parses = []
            post_records = []
            action_totals: Counter[str] = Counter()
            valid_count = 0
            for record in condition_records:
                task = by_key[record["key"]]
                parsed = record.get("parsed")
                actions: list[str] = []
                if spec["mode"] == "raw" and isinstance(parsed, dict):
                    try:
                        parsed, actions = postprocess_parse(parsed)
                    except Exception as error:
                        record["postprocess_error"] = f"{type(error).__name__}: {error}"
                elif spec["mode"] == "gold" and isinstance(parsed, dict):
                    parsed, actions = postprocess_gold_parse(
                        parsed, task["gold_sentence"]["tree_tokens"]
                    )
                errors = validate_serializable(parsed)
                if spec["mode"] == "gold" and not errors:
                    if [item["form"] for item in parsed["parses"]] != task["gold_sentence"]["tree_tokens"]:
                        errors.append("gold-tokenized forms differ from supplied tokens")
                record["postprocessed"] = parsed
                record["postprocessing_actions"] = actions
                record["semantic_validation_errors"] = errors
                record["evaluation_ready"] = not errors
                action_totals.update(actions)
                valid_count += not errors
                if not errors:
                    output_parses.append(parsed)
                post_records.append(record)
            write_jsonl(condition_root / "responses.postprocessed.jsonl", post_records)
            evaluation: dict[str, Any] | None = None
            evaluation_error = None
            if valid_count == condition_expected:
                sentences = [as_sentence(by_key[row["key"]]["gold_sentence"]) for row in condition_records]
                (condition_root / "gold.conllx").write_text(
                    format_gold(sentences), encoding="utf-8"
                )
                (condition_root / "predictions.conllx").write_text(
                    format_prediction(sentences, output_parses), encoding="utf-8"
                )
                try:
                    evaluation = evaluate_condition(
                        condition, condition_root, evaluation_basename
                    )
                except Exception as error:
                    evaluation_error = f"{type(error).__name__}: {error}"
            else:
                evaluation_error = (
                    f"Only {valid_count}/{condition_expected} answers passed "
                    "semantic validation"
                )

            usage = sum_usage(condition_records)
            job = jobs_by_condition[condition]
            timing = condition_timings.get(condition, {})
            full_size = int(by_key[condition_records[0]["key"]]["full_condition_size"])
            condition_summary = {
                **spec,
                "provider": provider,
                "model": MODELS[provider],
                "evaluated_sentences": condition_expected,
                "full_condition_size": full_size,
                "json_valid": sum(row.get("parsed") is not None for row in condition_records),
                "evaluation_ready": valid_count,
                "finish_reasons": dict(Counter(str(row.get("finish_reason")) for row in condition_records)),
                "provider_errors": sum(bool(row.get("provider_error")) for row in condition_records),
                "postprocessed_sentences": sum(bool(row.get("postprocessing_actions")) for row in condition_records),
                "postprocessing_action_totals": dict(sorted(action_totals.items())),
                "usage": usage,
                "batch_wall_seconds": float(job["batch_wall_seconds"]),
                "repair_batch_jobs": int(timing.get("repair_batch_jobs", 0)),
                "repair_requests": int(timing.get("repair_requests", 0)),
                "sum_repair_batch_wall_seconds": float(
                    timing.get("sum_repair_batch_wall_seconds", 0.0)
                ),
                "generation_observed_wall_seconds": float(
                    timing.get(
                        "generation_observed_wall_seconds",
                        job["batch_wall_seconds"],
                    )
                ),
                "submission_api_seconds": float(job["submission_api_seconds"]),
                "naive_projected_full_condition_cost_usd": (
                    usage["estimated_batch_cost_usd"]
                    if is_full
                    else usage["estimated_batch_cost_usd"] * full_size / 2
                ),
                "evaluation_flags": "-x -n -a",
                "evaluation": evaluation,
                "evaluation_error": evaluation_error,
            }
            write_json(condition_root / "summary.json", condition_summary)
            summaries.append(condition_summary)

    projected = {
        provider: sum(
            row["naive_projected_full_condition_cost_usd"]
            for row in summaries
            if row["provider"] == provider
        )
        for provider in providers
    }
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
    artifact_suffix = (
        "" if is_complete_provider_set else "." + "_".join(providers)
    )
    table_path = results_root / (
        f"cloud_batch_full_results{artifact_suffix}.tsv"
        if is_full
        else f"cloud_batch_pilot_results{artifact_suffix}.tsv"
    )
    columns = [
        "provider",
        "model",
        "condition",
        "mode",
        "split",
        "shots",
        "pool",
        "evaluated_sentences",
        "json_valid",
        "evaluation_ready",
        "provider_errors",
        "input_tokens",
        "cached_input_tokens",
        "visible_output_tokens",
        "reasoning_or_thinking_tokens",
        "estimated_batch_cost_usd",
        "batch_wall_seconds",
        "repair_batch_jobs",
        "repair_requests",
        "sum_repair_batch_wall_seconds",
        "generation_observed_wall_seconds",
        "naive_projected_full_condition_cost_usd",
        *metric_fields,
    ]
    with table_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for summary in summaries:
            usage = summary["usage"]
            evaluation = summary.get("evaluation") or {}
            writer.writerow(
                {
                    **{column: summary.get(column, "") for column in columns},
                    **{field: usage.get(field, "") for field in usage},
                    **{field: evaluation.get(field, "") for field in metric_fields},
                }
            )

    started_text = (run_root / "status" / "PIPELINE_STARTED").read_text(encoding="utf-8").strip()
    final_summary = {
        "completed_at_utc": utc_now(),
        "benchmark_scope": benchmark_scope,
        "scope": (
            f"All {expected_provider_requests} requests per provider across 16 conditions"
            if is_full
            else "2 sentences per condition; 16 conditions; 32 requests per provider"
        ),
        "models": MODELS,
        "pricing_usd_per_million_tokens": PRICES,
        "pricing_date": manifest.get("pricing_date", "2026-09-13"),
        "pricing_tier": "batch",
        "pricing_source": "https://developers.openai.com/api/docs/pricing",
        "requests": {
            provider: expected_provider_requests for provider in providers
        },
        "all_json_valid": all(
            row["json_valid"] == row["evaluated_sentences"]
            for row in summaries
        ),
        "all_evaluation_ready": all(
            row["evaluation_ready"] == row["evaluated_sentences"]
            for row in summaries
        ),
        "provider_totals": provider_totals,
        "naive_projected_full_benchmark_cost_usd": projected,
        "projection_note": (
            "Token cost extrapolated independently within each condition from only two "
            "sentences; use as a rough decision aid, not a billing guarantee."
        ),
        "pipeline_observed_wall_seconds": (
            datetime.now(timezone.utc) - parse_utc(started_text)
        ).total_seconds(),
        "results_table": str(table_path),
        "evaluation_flags": "-x -n -a",
        "conditions": summaries,
    }
    summary_name = (
        f"full_summary{artifact_suffix}.json"
        if is_full
        else f"pilot_summary{artifact_suffix}.json"
    )
    write_json(results_root / summary_name, final_summary)

    lines = [
        (
            "# Expanded BAREC cloud Batch full benchmark"
            if is_full
            else "# Expanded BAREC cloud Batch pilot"
        ),
        "",
        (
            f"All {expected_provider_requests} requests per provider were run over the "
            "16 gold/raw, dev/test, zero/10-shot conditions."
            if is_full
            else "Two separated sentences were run for every one of the 16 gold/raw, "
            "dev/test, zero/10-shot conditions. This is a pipeline validation sample, "
            "not a stable accuracy estimate."
        ),
        "",
        (
            "| Provider | Model | Requests | JSON valid | Evaluation ready | Input tok | "
            "Visible output tok | Reasoning/thinking tok | Est. Batch cost USD | "
            "Overall wall s | Actual full cost USD |"
            if is_full
            else "| Provider | Model | Requests | JSON valid | Evaluation ready | Input tok | Visible output tok | Reasoning/thinking tok | Est. Batch cost USD | Overall wall s | Rough full cost USD |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for provider in providers:
        provider_rows = [row for row in summaries if row["provider"] == provider]
        usage = provider_totals[provider]["usage"]
        lines.append(
            "| {provider} | `{model}` | {requests} | {json_valid}/{requests} | "
            "{ready}/{requests} | {inp} | "
            "{out} | {thought} | {cost:.6f} | {wall:.1f} | {projected:.2f} |".format(
                provider=provider,
                model=MODELS[provider],
                requests=expected_provider_requests,
                json_valid=sum(row["json_valid"] for row in provider_rows),
                ready=sum(row["evaluation_ready"] for row in provider_rows),
                inp=usage["input_tokens"],
                out=usage["visible_output_tokens"],
                thought=usage["reasoning_or_thinking_tokens"],
                cost=usage["estimated_batch_cost_usd"],
                wall=provider_totals[provider]["overall_observed_batch_wall_seconds"],
                projected=projected[provider],
            )
        )
    lines.extend(
        [
            "",
            "GPT-5.4 used reasoning effort `none`. Gemini thinking, temperature, and all "
            "other sampling settings remained at provider defaults. No input or output "
            "token cap was sent to either provider. All evaluations used CAMeL CoNLL "
            "`-x -n -a`; raw text used the established light postprocessor.",
            "",
            (
                "Costs above are computed from the complete provider-reported token usage "
                "using published Batch rates. GPT-5.4 uses $1.25/M uncached input, "
                "$0.125/M cached input, and $7.50/M output."
                if is_full
                else "The rough full-cost projections extrapolate from two sentences per "
                "condition and should be treated as provisional."
            ),
        ]
    )
    report_name = (
        f"FULL_REPORT{artifact_suffix}.md"
        if is_full
        else f"PILOT_REPORT{artifact_suffix}.md"
    )
    (results_root / report_name).write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    status_marker = (
        "PIPELINE_COMPLETE"
        if is_complete_provider_set
        else f"{providers[0].upper()}_EVALUATED"
    )
    (run_root / "status" / status_marker).write_text(
        utc_now() + "\n", encoding="utf-8"
    )
    print(f"FINALIZED {table_path}", flush=True)


def status(run_root: Path) -> None:
    result: dict[str, Any] = {"run_root": str(run_root), "providers": {}}
    for provider in PROVIDERS:
        jobs_path = run_root / provider / "jobs.json"
        if not jobs_path.exists():
            result["providers"][provider] = {"state": "not_started"}
            continue
        jobs = json.loads(jobs_path.read_text(encoding="utf-8"))["jobs"]
        result["providers"][provider] = {
            "states": dict(Counter(job.get("state", "UNKNOWN") for job in jobs)),
            "responses": (
                len(read_jsonl(run_root / provider / "responses.jsonl"))
                if (run_root / provider / "responses.jsonl").exists()
                else 0
            ),
        }
    result["pipeline_complete"] = (run_root / "status" / "PIPELINE_COMPLETE").exists()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "stage", "canary", "run", "finalize", "status"),
    )
    parser.add_argument("--provider", choices=PROVIDERS)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--condition")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Prepare all sentences in all 16 conditions instead of the pilot sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.run_root, full=args.full)
    elif args.command == "stage":
        if not args.provider:
            raise ValueError("--provider is required for stage")
        stage_provider(args.run_root, args.provider)
    elif args.command == "canary":
        if not args.provider or not args.condition:
            raise ValueError("--provider and --condition are required for canary")
        run_canary(
            args.run_root, args.provider, args.condition, args.poll_seconds
        )
    elif args.command == "run":
        if not args.provider:
            raise ValueError("--provider is required for run")
        run_provider(args.run_root, args.provider, args.poll_seconds)
    elif args.command == "finalize":
        finalize(
            args.run_root,
            (args.provider,) if args.provider else PROVIDERS,
        )
    else:
        status(args.run_root)


if __name__ == "__main__":
    main()
