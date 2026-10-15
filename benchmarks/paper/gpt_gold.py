#!/usr/bin/env python3
"""
OpenAI Batch Benchmark Runner for CATiB parsing.

Commands:
  - prepare   : generate OpenAI batch JSONL input file
  - upload    : upload JSONL to OpenAI with purpose=batch
  - create    : create batch job
  - status    : check status once
  - download  : download output JSONL for completed job
  - run       : one-command end-to-end + metrics + prediction/CoNLLX export

Features:
  - input_dir can be either:
      1. a directory containing .conllx files
      2. a single .conllx file
  - --sample-size limits to first N sentences per input file.
  - saves prediction JSONL per input file after batch results are downloaded.
  - saves prediction CoNLLX per input file after batch results are downloaded.
  - keeps benchmark metrics: turnaround time, throughput, cost, reliability.

Few-shot control:
  --num_shots K:
    - match-file mode default: take TOP-K shots per sentence from shots_file order
    - --random_shots: sample K random training examples per sentence
    - K=0: zero-shot

Cost:
  - optional: provide --price_in_per_1k and --price_out_per_1k
  - optionally --price_reasoning_per_1k
  - batch discount factor can be applied with --batch_discount
"""

import os
import json
import yaml
import time
import csv
import random
import hashlib
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from conllu import parse_incr
from openai import OpenAI
from functools import lru_cache

@lru_cache(maxsize=1)
def get_client():
    return OpenAI()

DEPRELS = ["SBJ", "OBJ", "PRD", "TPC", "IDF", "TMZ", "MOD", "---"]


# ---------------------------
# Utilities
# ---------------------------

def now_utc_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fmt_seconds(s: float) -> str:
    s = int(s)
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60

    if h > 0:
        return f"{h}h{m:02d}m{ss:02d}s"

    if m > 0:
        return f"{m}m{ss:02d}s"

    return f"{ss}s"


def safe_get(d, *keys, default=None):
    cur = d

    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default

        cur = cur[k]

    return cur


def fsync_file(f):
    f.flush()
    os.fsync(f.fileno())


def get_input_files(input_path: str) -> List[Tuple[str, str]]:
    """
    Normalize input path to a list of:
      (filename, full_path)

    Supports:
      - single .conllx file
      - directory containing .conllx files
    """
    if os.path.isfile(input_path):
        if not input_path.endswith(".conllx"):
            raise ValueError(f"Input file must be a .conllx file: {input_path}")

        return [(os.path.basename(input_path), input_path)]

    if os.path.isdir(input_path):
        files = [
            (filename, os.path.join(input_path, filename))
            for filename in sorted(os.listdir(input_path))
            if filename.endswith(".conllx")
        ]

        if not files:
            raise FileNotFoundError(f"No .conllx files found in directory: {input_path}")

        return files

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def get_match_key(shots_data: Dict[str, Any], filename: str, input_path: str) -> Optional[str]:
    """
    Match files may use:
      - basename
      - full input path
      - absolute input path
    """
    candidates = [
        filename,
        input_path,
        os.path.abspath(input_path),
    ]

    for key in candidates:
        if key in shots_data:
            return key

    return None


def make_custom_id(filename: str, sentence_index_1based: int) -> str:
    """
    Stable custom_id used both in prepare and result post-processing.
    """
    return f"{filename}__sent__{sentence_index_1based}"


def parse_custom_id(custom_id: str) -> Tuple[Optional[str], Optional[int]]:
    marker = "__sent__"

    if marker not in custom_id:
        return None, None

    filename, idx_str = custom_id.rsplit(marker, 1)

    try:
        idx = int(idx_str)
    except Exception:
        return filename, None

    return filename, idx


# ---------------------------
# CoNLLX parsing / formatting
# ---------------------------

def parse_conllx_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Returns list of examples:
      {
        "text": str,
        "tokens": [str],
        "parses": [(id, form, head, deprel), ...]
      }
    """
    examples: List[Dict[str, Any]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            tokens = tokenlist.metadata.get("treeTokens", "").split()
            text = tokenlist.metadata.get("text")

            if not text or not tokens:
                continue

            parses = []

            for tok in tokenlist:
                parses.append((
                    int(tok["id"]),
                    tok["form"],
                    int(tok["head"]),
                    tok["deprel"],
                ))

            examples.append({
                "text": text,
                "tokens": tokens,
                "parses": parses,
            })

    return examples


def parses_to_conllx(
    tokens: List[str],
    parses: List[Tuple[int, str, int, str]],
    text: str,
) -> str:
    """
    Writes CoNLL-X style block.

    Columns:
      ID FORM LEMMA CPOSTAG POSTAG FEATS HEAD DEPREL PHEAD PDEPREL
    """
    lines = [
        f"# text = {text}",
        f"# treeTokens = {' '.join(tokens)}",
    ]

    for tok_id, form, head, deprel in parses:
        lines.append(
            f"{tok_id}\t{form}\t_\t_\t_\t_\t{head}\t{deprel}\t_\t_"
        )

    return "\n".join(lines) + "\n\n"


def parses_to_arrays(parses: List[Tuple[int, str, int, str]]) -> Tuple[List[int], List[str]]:
    head = [p[2] for p in parses]
    deprel = [p[3] for p in parses]
    return head, deprel


def validate_and_normalize_parses(
    tokens: List[str],
    obj: Dict[str, Any],
) -> List[Tuple[int, str, int, str]]:
    """
    Strict validation for saved CoNLLX prediction.

    Requires:
      - top-level object
      - parses list
      - same length as input tokens
      - ids 1..n
      - forms exactly match input tokens
      - heads 0..n
      - deprel in DEPRELS
    """
    if not isinstance(obj, dict):
        raise ValueError("Top-level output must be a JSON object.")

    parses = obj.get("parses")

    if not isinstance(parses, list):
        raise ValueError("Top-level key 'parses' must be a list.")

    n = len(tokens)

    if len(parses) != n:
        raise ValueError(f"Expected {n} parse items, got {len(parses)}.")

    by_id: Dict[int, Dict[str, Any]] = {}

    for item in parses:
        if not isinstance(item, dict):
            raise ValueError("Each parse item must be an object.")

        for key in ("id", "form", "head", "deprel"):
            if key not in item:
                raise ValueError(f"Missing key '{key}' in parse item.")

        tok_id = int(item["id"])

        if tok_id < 1 or tok_id > n:
            raise ValueError(f"id out of range: {tok_id}")

        if tok_id in by_id:
            raise ValueError(f"Duplicate id: {tok_id}")

        by_id[tok_id] = item

    out: List[Tuple[int, str, int, str]] = []

    for i in range(1, n + 1):
        item = by_id[i]

        form = str(item["form"])

        if form != tokens[i - 1]:
            raise ValueError(
                f"Form mismatch at id={i}: expected {tokens[i - 1]!r}, got {form!r}"
            )

        head = int(item["head"])

        if head < 0 or head > n:
            raise ValueError(f"Head out of range at id={i}: {head}")

        deprel = str(item["deprel"])

        if deprel not in DEPRELS:
            raise ValueError(f"Invalid deprel at id={i}: {deprel}")

        out.append((i, form, head, deprel))

    return out


def best_effort_parses_from_obj(
    tokens: List[str],
    obj: Any,
) -> Optional[List[Tuple[int, str, int, str]]]:
    """
    Best-effort conversion for parse-only output.

    Fixes cases where the model produced:
      - duplicate ids
      - missing ids
      - forms that do not exactly match input tokens

    Since this script is parse-only, we trust the original input tokens
    and use the model output mainly for head/deprel.
    """
    try:
        if not isinstance(obj, dict):
            return None

        parses = obj.get("parses")

        if not isinstance(parses, list):
            return None

        n = len(tokens)

        if len(parses) != n:
            return None

        def normalize_deprel(label: Any, head: int) -> str:
            label = str(label)

            if label in DEPRELS:
                return label

            low = label.lower()

            if low == "root" and head == 0:
                return "---"

            if label == "TMC":
                return "TMZ"

            if label in {"PUNC", "PUNCT"}:
                return "MOD"

            return "---" if head == 0 else "MOD"

        by_id: Dict[int, Dict[str, Any]] = {}
        ids_ok = True

        for item in parses:
            if not isinstance(item, dict):
                return None

            if "head" not in item or "deprel" not in item:
                return None

            try:
                tok_id = int(item.get("id"))
            except Exception:
                ids_ok = False
                continue

            if tok_id < 1 or tok_id > n or tok_id in by_id:
                ids_ok = False
            else:
                by_id[tok_id] = item

        out: List[Tuple[int, str, int, str]] = []

        # Normal case: ids are exactly 1..n.
        if ids_ok and set(by_id.keys()) == set(range(1, n + 1)):
            ordered_items = [by_id[i] for i in range(1, n + 1)]

        # Repair case: duplicate/missing ids.
        # Use output order and reassign ids 1..n.
        else:
            ordered_items = parses

        for i, item in enumerate(ordered_items, start=1):
            form = tokens[i - 1]

            try:
                head = int(item.get("head", 0))
            except Exception:
                head = 0

            if head < 0 or head > n:
                head = 0

            deprel = normalize_deprel(item.get("deprel", "---"), head)

            out.append((i, form, head, deprel))

        return out

    except Exception:
        return None

# ---------------------------
# Response extraction
# ---------------------------

def extract_response_text_from_body(body: Dict[str, Any]) -> Optional[str]:
    """
    Extract model text from OpenAI Responses API body.

    Handles common shapes:
      - body["output_text"]
      - body["output"][...]["content"][...]["text"]
    """
    if not isinstance(body, dict):
        return None

    output_text = body.get("output_text")

    if isinstance(output_text, str):
        return output_text

    output = body.get("output")

    if isinstance(output, list):
        chunks = []

        for item in output:
            if not isinstance(item, dict):
                continue

            content = item.get("content")

            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue

                    txt = c.get("text")

                    if isinstance(txt, str):
                        chunks.append(txt)

            txt = item.get("text")

            if isinstance(txt, str):
                chunks.append(txt)

        if chunks:
            return "\n".join(chunks)

    return None


def parse_prediction_text(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns:
      (json_obj, error)
    """
    try:
        obj = json.loads(text)

        if isinstance(obj, dict):
            return obj, None

        return None, "Parsed JSON is not an object."

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ---------------------------
# Stage 1: Prepare batch file
# ---------------------------

def prepare_batch(args):
    if getattr(args, "sample_size", None) is not None and args.sample_size <= 0:
        raise ValueError("--sample-size must be a positive integer")

    os.makedirs(args.cache_dir, exist_ok=True)
    batch_file = os.path.join(args.cache_dir, args.cache_name)

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if "system_prompt" not in config:
        raise KeyError("YAML config must contain a 'system_prompt' key")

    system_prompt = config["system_prompt"].rstrip() + "\n"
    cache_key = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()

    # Load shots if not random
    shots_data: Dict[str, Any] = {}

    if not args.random_shots:
        if not args.shots_file:
            raise SystemExit("You must provide --shots_file unless using --random_shots")

        with open(args.shots_file, "r", encoding="utf-8") as f:
            shots_data = json.load(f)

    # Load training examples
    train_data = parse_conllx_file(args.train_file)

    if args.seed is not None:
        random.seed(args.seed)

    input_files = get_input_files(args.input_dir)

    k = int(args.num_shots)
    n_written = 0
    n_used_shots_total = 0
    n_shot_requests = 0

    with open(batch_file, "w", encoding="utf-8") as out:
        for filename, input_path in input_files:
            examples_all = parse_conllx_file(input_path)

            if args.sample_size is not None:
                examples = examples_all[:args.sample_size]
            else:
                examples = examples_all

            if not args.random_shots:
                match_key = get_match_key(shots_data, filename, input_path)

                if match_key is None:
                    print(f"⚠️ No shots found for {filename}, skipping file.")
                    continue

                shot_indices_per_sentence = shots_data[match_key]
            else:
                shot_indices_per_sentence = None

            print(
                f"Preparing {filename}: {len(examples)} sentence(s)"
                + (
                    f" using --sample-size {args.sample_size}; original has {len(examples_all)}."
                    if args.sample_size is not None
                    else ""
                )
            )

            for sent_idx_1based, ex in enumerate(examples, start=1):
                tokens = ex["tokens"]
                text = ex["text"]

                if not text or not tokens:
                    continue

                # ---- select shots ----
                if k <= 0:
                    shot_examples = []

                elif args.random_shots:
                    k_eff = min(k, len(train_data))
                    idxs = random.sample(range(len(train_data)), k=k_eff)
                    shot_examples = [train_data[j] for j in idxs]

                else:
                    i0 = sent_idx_1based - 1

                    if i0 >= len(shot_indices_per_sentence):
                        print(
                            f"⚠️ Not enough shot entries for {filename}, "
                            f"sentence {sent_idx_1based}, skipping sentence."
                        )
                        continue

                    current = shot_indices_per_sentence[i0][:k]
                    shot_examples = []

                    for entry in current:
                        train_idx = entry.get("train_index")

                        if isinstance(train_idx, int) and 0 <= train_idx < len(train_data):
                            shot_examples.append(train_data[train_idx])

                if k > 0:
                    n_shot_requests += 1
                    n_used_shots_total += len(shot_examples)

                # Format few-shot text
                fewshot_text = ""

                for shot in shot_examples:
                    fewshot_text += "Input:\n" + json.dumps(shot["tokens"], ensure_ascii=False) + "\n"
                    fewshot_text += "Output:\n" + json.dumps(
                        {
                            "parses": [
                                {
                                    "id": a,
                                    "form": b,
                                    "head": c,
                                    "deprel": d,
                                }
                                for (a, b, c, d) in shot["parses"]
                            ]
                        },
                        ensure_ascii=False,
                    ) + "\n\n"

                examples_block = ""

                if fewshot_text.strip():
                    examples_block = "Here are some annotated examples:\n\n" + fewshot_text + "\n"

                user_input = json.dumps({"Input": tokens}, ensure_ascii=False)

                n = len(tokens)

                schema = {
                    "type": "object",
                    "properties": {
                        "parses": {
                            "type": "array",
                            "minItems": n,
                            "maxItems": n,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": n,
                                    },
                                    "form": {
                                        "type": "string",
                                    },
                                    "head": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": n,
                                    },
                                    "deprel": {
                                        "type": "string",
                                        "enum": DEPRELS,
                                    },
                                },
                                "required": ["id", "form", "head", "deprel"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["parses"],
                    "additionalProperties": False,
                }

                full_prompt = system_prompt + examples_block

                body = {
                    "model": args.model,
                    "instructions": full_prompt,
                    "input": user_input,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "catib_annotation_parses",
                            "schema": schema,
                            "strict": True,
                            "description": "CATiB dependency parses as {'parses':[...]}",
                        },
                    },
                    "prompt_cache_key": cache_key,
                }

                if args.reasoning_effort and args.reasoning_effort != "none":
                    body["reasoning"] = {"effort": args.reasoning_effort}

                req = {
                    "custom_id": make_custom_id(filename, sent_idx_1based),
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                }

                out.write(json.dumps(req, ensure_ascii=False) + "\n")
                n_written += 1

    avg_shots = (n_used_shots_total / n_shot_requests) if n_shot_requests > 0 else 0.0

    print(f"✅ Prepared batch file '{batch_file}' with {n_written} requests")
    print(
        f"   Shots setting: num_shots={k} | "
        f"random_shots={bool(args.random_shots)} | "
        f"avg_shots_used={avg_shots:.2f}"
    )

    return batch_file, n_written, {
        "num_shots": k,
        "random_shots": bool(args.random_shots),
        "avg_shots_used": avg_shots,
        "sample_size": args.sample_size,
    }


# ---------------------------
# Stage 2: Upload file
# ---------------------------

def upload_file(args):
    batch_path = os.path.join(args.cache_dir, args.cache_name)
    batch_input_file = get_client().files.create(
        file=open(batch_path, "rb"),
        purpose="batch",
    )
    print(f"✅ Uploaded file ID: {batch_input_file.id}")
    return batch_input_file.id


# ---------------------------
# Stage 3: Create batch
# ---------------------------

def create_batch(args):
    batch = get_client().batches.create(
        input_file_id=args.file_id,
        endpoint="/v1/responses",
        completion_window=args.completion_window,
        metadata={"description": args.description or "Batch job"},
    )
    print(f"✅ Batch created: {batch.id} | status={batch.status}")
    return batch.id


# ---------------------------
# Stage 4: Status
# ---------------------------

def check_status(args):
    batch = get_client().batches.retrieve(args.batch_id)
    print(f"📊 Status: {batch.status}")

    if getattr(batch, "output_file_id", None):
        print("Output file ID:", batch.output_file_id)

    if getattr(batch, "error_file_id", None):
        print("Error file ID:", batch.error_file_id)

    return batch


# ---------------------------
# Download results
# ---------------------------

def download_results(args):
    output = get_client().files.content(args.output_file_id)
    out_dir = args.out_dir or args.cache_dir
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, args.results_name)

    with open(result_path, "w", encoding="utf-8") as f:
        for line in output.text.splitlines():
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                f.write(line + "\n")

    print(f"✅ Results saved to {result_path}")

    if getattr(args, "input_dir", None):
        write_predictions_from_results(args, result_path)

    return result_path


# ---------------------------
# Poll with terminal progress
# ---------------------------

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}


def wait_for_batch_completion(
    batch_id,
    poll_seconds=30,
    timeout_seconds=24 * 3600,
    show_progress=True,
):
    start = time.time()
    last_status = None

    while True:
        batch = get_client().batches.retrieve(batch_id)
        elapsed = time.time() - start
        status = batch.status

        counts = getattr(batch, "request_counts", None)
        parts = [
            f"[{now_utc_iso()}] status={status}",
            f"elapsed={fmt_seconds(elapsed)}",
        ]

        if counts:
            for k in ["total", "completed", "failed"]:
                v = getattr(counts, k, None)

                if isinstance(v, int):
                    parts.append(f"{k}={v}")

        if show_progress:
            if status != last_status:
                parts.append("(status change)")
                last_status = status

            print(" | ".join(parts))

        if status in TERMINAL_STATUSES:
            return batch, elapsed

        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Batch did not finish within {timeout_seconds}s. Last status={status}"
            )

        time.sleep(poll_seconds)


# ---------------------------
# Parse output JSONL -> reliability + tokens
# ---------------------------

def parse_openai_results_jsonl(results_path):
    """
    Parse OpenAI batch output JSONL and aggregate:
      - reliability counts
      - token usage sums

    Supports usage schemas:
      A) prompt_tokens/completion_tokens
      B) input_tokens/output_tokens
    """
    totals = {
        "total_lines": 0,
        "success_lines": 0,
        "error_lines": 0,
        "tokens": {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        },
        "errors": {},
    }

    def bump_error(label):
        totals["errors"][label] = totals["errors"].get(label, 0) + 1

    def add_int(field, v):
        if isinstance(v, int):
            totals["tokens"][field] += v

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            totals["total_lines"] += 1

            try:
                obj = json.loads(line)
            except Exception:
                totals["error_lines"] += 1
                bump_error("json_parse_error")
                continue

            if isinstance(obj, dict) and obj.get("error") is not None:
                totals["error_lines"] += 1
                err = obj["error"]

                if isinstance(err, dict):
                    code = err.get("code")
                    msg = err.get("message", "")
                    label = f"code={code}|{msg[:140]}" if code is not None else msg[:140]
                else:
                    label = str(err)[:140]

                bump_error(label or "unknown_error")
                continue

            resp = obj.get("response") if isinstance(obj, dict) else None
            status_code = resp.get("status_code") if isinstance(resp, dict) else None

            if status_code != 200:
                totals["error_lines"] += 1
                bump_error(f"status_code={status_code}")
                continue

            totals["success_lines"] += 1

            body = resp.get("body") if isinstance(resp, dict) else None

            if not isinstance(body, dict):
                continue

            usage = body.get("usage")

            if not isinstance(usage, dict):
                continue

            if "input_tokens" in usage or "output_tokens" in usage:
                add_int("input_tokens", usage.get("input_tokens"))
                add_int("output_tokens", usage.get("output_tokens"))
                add_int("total_tokens", usage.get("total_tokens"))

                details = usage.get("output_tokens_details")

                if isinstance(details, dict):
                    add_int("reasoning_tokens", details.get("reasoning_tokens"))

            else:
                add_int("input_tokens", usage.get("prompt_tokens"))
                add_int("output_tokens", usage.get("completion_tokens"))
                add_int("total_tokens", usage.get("total_tokens"))

                details = usage.get("completion_tokens_details")

                if isinstance(details, dict):
                    add_int("reasoning_tokens", details.get("reasoning_tokens"))

    if totals["tokens"]["total_tokens"] == 0:
        totals["tokens"]["total_tokens"] = (
            totals["tokens"]["input_tokens"]
            + totals["tokens"]["output_tokens"]
            + totals["tokens"]["reasoning_tokens"]
        )

    return totals


def compute_openai_cost(
    token_totals,
    price_in_per_1k,
    price_out_per_1k,
    price_reasoning_per_1k=None,
    batch_discount=1.0,
):
    inp = token_totals.get("input_tokens", 0)
    out = token_totals.get("output_tokens", 0)
    reasoning = token_totals.get("reasoning_tokens", 0)

    if price_reasoning_per_1k is None:
        price_reasoning_per_1k = price_out_per_1k

    cost = (
        (inp / 1000.0) * price_in_per_1k
        + (out / 1000.0) * price_out_per_1k
        + (reasoning / 1000.0) * price_reasoning_per_1k
    )
    cost *= batch_discount

    return cost


def write_metrics_report(out_dir, base_name, report_dict):
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, f"{base_name}_metrics.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(out_dir, f"{base_name}_metrics.csv")

    flat = {
        "batch_id": report_dict.get("batch_id"),
        "status": report_dict.get("status"),
        "turnaround_seconds": report_dict.get("turnaround_seconds"),
        "requests_total": report_dict.get("requests_total"),
        "requests_success": report_dict.get("requests_success"),
        "requests_error": report_dict.get("requests_error"),
        "requests_per_sec": safe_get(report_dict, "throughput", "requests_per_sec"),
        "input_tokens": safe_get(report_dict, "tokens", "input_tokens"),
        "output_tokens": safe_get(report_dict, "tokens", "output_tokens"),
        "reasoning_tokens": safe_get(report_dict, "tokens", "reasoning_tokens"),
        "total_tokens": safe_get(report_dict, "tokens", "total_tokens"),
        "total_tokens_per_sec": safe_get(report_dict, "throughput", "total_tokens_per_sec"),
        "output_tokens_per_sec": safe_get(report_dict, "throughput", "output_tokens_per_sec"),
        "reasoning_tokens_per_sec": safe_get(report_dict, "throughput", "reasoning_tokens_per_sec"),
        "reasoning_over_output_ratio": safe_get(report_dict, "ratios", "reasoning_over_output_ratio"),
        "reasoning_over_total_ratio": safe_get(report_dict, "ratios", "reasoning_over_total_ratio"),
        "total_cost": safe_get(report_dict, "cost", "total_cost"),
        "cost_per_success": safe_get(report_dict, "cost", "cost_per_success"),
        "num_shots": safe_get(report_dict, "shots", "num_shots"),
        "random_shots": safe_get(report_dict, "shots", "random_shots"),
        "avg_shots_used": safe_get(report_dict, "shots", "avg_shots_used"),
        "sample_size": safe_get(report_dict, "shots", "sample_size"),
    }

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    return json_path, csv_path


# ---------------------------
# Prediction post-processing
# ---------------------------

def build_example_lookup(input_dir: str, sample_size: Optional[int]) -> Dict[str, Dict[str, Any]]:
    """
    Build mapping:
      custom_id -> {
        file,
        sentence_index_1based,
        text,
        tokens
      }
    """
    lookup: Dict[str, Dict[str, Any]] = {}

    input_files = get_input_files(input_dir)

    for filename, input_path in input_files:
        examples_all = parse_conllx_file(input_path)

        if sample_size is not None:
            examples = examples_all[:sample_size]
        else:
            examples = examples_all

        for i, ex in enumerate(examples, start=1):
            custom_id = make_custom_id(filename, i)

            lookup[custom_id] = {
                "file": filename,
                "sentence_index_1based": i,
                "text": ex["text"],
                "tokens": ex["tokens"],
            }

    return lookup


def write_predictions_from_results(args, results_path: str) -> Dict[str, Any]:
    """
    Convert downloaded OpenAI Batch results into:
      - per-file prediction JSONL
      - per-file prediction CoNLLX

    Output dirs:
      --pred_output_dir if provided
      else --out_dir if provided
      else --cache_dir

    CoNLLX dir:
      --conllx_output_dir if provided
      else pred_output_dir + "_conllx"
    """
    sample_size = getattr(args, "sample_size", None)

    if sample_size is not None and sample_size <= 0:
        raise ValueError("--sample-size must be a positive integer")

    pred_output_dir = (
        getattr(args, "pred_output_dir", None)
        or getattr(args, "out_dir", None)
        or args.cache_dir
    )

    conllx_output_dir = (
        getattr(args, "conllx_output_dir", None)
        or (pred_output_dir + "_conllx")
    )

    os.makedirs(pred_output_dir, exist_ok=True)
    os.makedirs(conllx_output_dir, exist_ok=True)

    example_lookup = build_example_lookup(args.input_dir, sample_size)

    per_file_jsonl: Dict[str, List[Dict[str, Any]]] = {}
    per_file_conllx: Dict[str, List[str]] = {}

    stats = {
        "results_seen": 0,
        "results_success_status": 0,
        "predictions_json_parsed": 0,
        "predictions_validated": 0,
        "predictions_best_effort": 0,
        "predictions_failed": 0,
        "unknown_custom_id": 0,
        "written_jsonl_files": [],
        "written_conllx_files": [],
    }

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            stats["results_seen"] += 1

            try:
                result = json.loads(line)
            except Exception as e:
                stats["predictions_failed"] += 1
                continue

            custom_id = result.get("custom_id")

            if not isinstance(custom_id, str):
                stats["unknown_custom_id"] += 1
                continue

            ex_meta = example_lookup.get(custom_id)

            if ex_meta is None:
                # Try to parse older/custom IDs if possible
                filename, sent_idx = parse_custom_id(custom_id)

                if filename is not None and sent_idx is not None:
                    alt_id = make_custom_id(filename, sent_idx)
                    ex_meta = example_lookup.get(alt_id)

            if ex_meta is None:
                stats["unknown_custom_id"] += 1
                continue

            filename = ex_meta["file"]
            sent_idx = ex_meta["sentence_index_1based"]
            text = ex_meta["text"]
            tokens = ex_meta["tokens"]

            if filename not in per_file_jsonl:
                per_file_jsonl[filename] = []
                per_file_conllx[filename] = []

            out_obj: Dict[str, Any] = {
                "file": filename,
                "sentence_index_1based": sent_idx,
                "custom_id": custom_id,
                "text": text,
                "tokens": tokens,
            }

            error = result.get("error")

            if error is not None:
                out_obj["status"] = "error"
                out_obj["error"] = error
                per_file_jsonl[filename].append(out_obj)
                stats["predictions_failed"] += 1
                continue

            resp = result.get("response")

            if not isinstance(resp, dict) or resp.get("status_code") != 200:
                out_obj["status"] = "error"
                out_obj["error"] = {
                    "message": "Missing response or non-200 status_code",
                    "status_code": resp.get("status_code") if isinstance(resp, dict) else None,
                }
                per_file_jsonl[filename].append(out_obj)
                stats["predictions_failed"] += 1
                continue

            stats["results_success_status"] += 1

            body = resp.get("body")

            if not isinstance(body, dict):
                out_obj["status"] = "error"
                out_obj["error"] = {"message": "Missing response body"}
                per_file_jsonl[filename].append(out_obj)
                stats["predictions_failed"] += 1
                continue

            raw_text = extract_response_text_from_body(body)

            if raw_text is None:
                out_obj["status"] = "error"
                out_obj["error"] = {"message": "Could not extract response text"}
                out_obj["response_body"] = body
                per_file_jsonl[filename].append(out_obj)
                stats["predictions_failed"] += 1
                continue

            out_obj["raw_prediction_text"] = raw_text

            pred_obj, parse_error = parse_prediction_text(raw_text)

            if pred_obj is None:
                out_obj["status"] = "json_parse_error"
                out_obj["json_parse_error"] = parse_error
                per_file_jsonl[filename].append(out_obj)
                stats["predictions_failed"] += 1
                continue

            stats["predictions_json_parsed"] += 1
            out_obj["prediction_json"] = pred_obj

            parses = None

            try:
                parses = validate_and_normalize_parses(tokens, pred_obj)
                stats["predictions_validated"] += 1
                out_obj["status"] = "ok_validated"
            except Exception as strict_e:
                out_obj["validation_error"] = f"{type(strict_e).__name__}: {strict_e}"
                parses = best_effort_parses_from_obj(tokens, pred_obj)

                if parses is not None:
                    stats["predictions_best_effort"] += 1
                    out_obj["status"] = "ok_best_effort"
                else:
                    out_obj["status"] = "validation_failed"

            if parses is not None:
                head, deprel = parses_to_arrays(parses)
                out_obj["parses"] = parses
                out_obj["head"] = head
                out_obj["deprel"] = deprel
                per_file_conllx[filename].append(parses_to_conllx(tokens, parses, text))
            else:
                stats["predictions_failed"] += 1

            per_file_jsonl[filename].append(out_obj)

    for filename, rows in per_file_jsonl.items():
        rows_sorted = sorted(
            rows,
            key=lambda x: x.get("sentence_index_1based", 10**18),
        )

        jsonl_path = os.path.join(pred_output_dir, filename.replace(".conllx", ".jsonl"))

        with open(jsonl_path, "w", encoding="utf-8") as jf:
            for row in rows_sorted:
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")

        stats["written_jsonl_files"].append(jsonl_path)

        conllx_path = os.path.join(conllx_output_dir, filename)

        with open(conllx_path, "w", encoding="utf-8") as cf:
            cf.write("".join(per_file_conllx.get(filename, [])))

        stats["written_conllx_files"].append(conllx_path)

        print(f"✅ Prediction JSONL written: {jsonl_path}")
        print(f"✅ Prediction CoNLLX written: {conllx_path}")

    stats_path = os.path.join(pred_output_dir, "prediction_export_summary.json")

    with open(stats_path, "w", encoding="utf-8") as sf:
        json.dump(stats, sf, ensure_ascii=False, indent=2)

    print(f"📌 Prediction export summary: {stats_path}")

    return stats


# ---------------------------
# One-command RUN
# ---------------------------

def run_all(args):
    t_submit = time.time()

    # 1) prepare
    batch_file, expected_requests, shots_meta = prepare_batch(args)

    # 2) upload
    file_id = get_client().files.create(
        file=open(batch_file, "rb"),
        purpose="batch",
    ).id
    print(f"✅ Uploaded file ID: {file_id}")

    # 3) create
    batch = get_client().batches.create(
        input_file_id=file_id,
        endpoint="/v1/responses",
        completion_window=args.completion_window,
        metadata={"description": args.description or "CATiB annotation job"},
    )

    batch_id = batch.id
    print(f"✅ Batch created: {batch_id} | status={batch.status}")

    # 4) poll
    final_batch, waited = wait_for_batch_completion(
        batch_id=batch_id,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
        show_progress=True,
    )

    t_done = time.time()
    turnaround_seconds = t_done - t_submit
    status = final_batch.status

    out_dir = args.out_dir or args.cache_dir
    os.makedirs(out_dir, exist_ok=True)

    results_path = None
    prediction_export_stats = None

    parsed = {
        "total_lines": 0,
        "success_lines": 0,
        "error_lines": 0,
        "tokens": {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        },
        "errors": {},
    }

    # 5) download only if completed
    output_file_id = getattr(final_batch, "output_file_id", None)

    if status == "completed" and output_file_id:
        output = get_client().files.content(output_file_id)
        results_path = os.path.join(out_dir, f"{batch_id}_results.jsonl")

        with open(results_path, "w", encoding="utf-8") as f:
            for line in output.text.splitlines():
                line = line.strip()

                if not line:
                    continue

                f.write(line + "\n")

        print(f"✅ Results saved to {results_path}")

        parsed = parse_openai_results_jsonl(results_path)

        prediction_export_stats = write_predictions_from_results(args, results_path)

    else:
        print(f"❌ Batch ended with status={status}")

        if getattr(final_batch, "error_file_id", None):
            try:
                err = get_client().files.content(final_batch.error_file_id)
                err_path = os.path.join(out_dir, f"{batch_id}_errors.txt")

                with open(err_path, "w", encoding="utf-8") as ef:
                    ef.write(err.text)

                print(f"⚠️ Error file saved to {err_path}")

            except Exception as e:
                print("⚠️ Could not download error file:", e)

    # 6) metrics
    requests_total = parsed["total_lines"] if parsed["total_lines"] > 0 else expected_requests
    requests_success = parsed["success_lines"]
    requests_error = parsed["error_lines"]
    tok = parsed["tokens"]

    req_per_sec = (requests_total / turnaround_seconds) if turnaround_seconds > 0 else 0.0
    total_tok_per_sec = (tok["total_tokens"] / turnaround_seconds) if turnaround_seconds > 0 else 0.0
    out_tok_per_sec = (tok["output_tokens"] / turnaround_seconds) if turnaround_seconds > 0 else 0.0
    reasoning_tok_per_sec = (tok["reasoning_tokens"] / turnaround_seconds) if turnaround_seconds > 0 else 0.0

    reasoning_over_output = (
        tok["reasoning_tokens"] / tok["output_tokens"]
    ) if tok["output_tokens"] > 0 else None

    reasoning_over_total = (
        tok["reasoning_tokens"] / tok["total_tokens"]
    ) if tok["total_tokens"] > 0 else None

    report = {
        "batch_id": batch_id,
        "status": status,
        "submitted_at_utc": datetime.fromtimestamp(
            t_submit,
            tz=timezone.utc,
        ).isoformat(timespec="seconds"),
        "completed_at_utc": datetime.fromtimestamp(
            t_done,
            tz=timezone.utc,
        ).isoformat(timespec="seconds"),
        "turnaround_seconds": turnaround_seconds,
        "waited_seconds": waited,
        "shots": shots_meta,
        "expected_requests": expected_requests,
        "requests_total": requests_total,
        "requests_success": requests_success,
        "requests_error": requests_error,
        "tokens": tok,
        "throughput": {
            "requests_per_sec": req_per_sec,
            "total_tokens_per_sec": total_tok_per_sec,
            "output_tokens_per_sec": out_tok_per_sec,
            "reasoning_tokens_per_sec": reasoning_tok_per_sec,
        },
        "ratios": {
            "reasoning_over_output_ratio": reasoning_over_output,
            "reasoning_over_total_ratio": reasoning_over_total,
        },
        "errors": parsed["errors"],
        "output_file_id": output_file_id,
        "results_path": results_path,
        "prediction_export": prediction_export_stats,
    }

    if args.price_in_per_1k is not None and args.price_out_per_1k is not None:
        total_cost = compute_openai_cost(
            tok,
            price_in_per_1k=args.price_in_per_1k,
            price_out_per_1k=args.price_out_per_1k,
            price_reasoning_per_1k=args.price_reasoning_per_1k,
            batch_discount=args.batch_discount,
        )

        report["cost"] = {
            "price_in_per_1k": args.price_in_per_1k,
            "price_out_per_1k": args.price_out_per_1k,
            "price_reasoning_per_1k": (
                args.price_reasoning_per_1k
                if args.price_reasoning_per_1k is not None
                else args.price_out_per_1k
            ),
            "batch_discount": args.batch_discount,
            "total_cost": total_cost,
            "cost_per_success": (
                total_cost / requests_success
            ) if requests_success > 0 else None,
            "cost_per_1k_total_tokens": (
                total_cost / (tok["total_tokens"] / 1000.0)
            ) if tok["total_tokens"] > 0 else None,
        }

    json_path, csv_path = write_metrics_report(out_dir, f"{batch_id}", report)

    print(f"📄 Metrics JSON: {json_path}")
    print(f"📄 Metrics CSV : {csv_path}")
    print("✅ Done.")


# ---------------------------
# CLI
# ---------------------------

def add_shared_prepare_args(s):
    s.add_argument("--config", required=True)
    s.add_argument("--cache_dir", required=True)
    s.add_argument("--cache_name", default="batch.jsonl")
    s.add_argument("--input_dir", required=True, help="Directory of .conllx files OR a single .conllx file.")
    s.add_argument("--shots_file", required=False)
    s.add_argument("--train_file", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--random_shots", action="store_true")
    s.add_argument(
        "--num_shots",
        type=int,
        default=5,
        help="TOP-K shots in match-file mode, or random K. Use 0 for zero-shot.",
    )
    s.add_argument("--seed", type=int, default=42)
    s.add_argument(
        "--reasoning_effort",
        default="none",
        choices=["none", "minimal", "low", "medium", "high"],
    )
    s.add_argument(
        "--sample-size",
        dest="sample_size",
        type=int,
        default=None,
        help="Process only the first N sentences per input file.",
    )


def build_parser():
    p = argparse.ArgumentParser(
        description="CATiB OpenAI Batch Pipeline + Benchmark"
    )
    sp = p.add_subparsers(dest="command", required=True)

    # prepare
    s = sp.add_parser("prepare", help="Generate batch JSONL")
    add_shared_prepare_args(s)
    s.set_defaults(func=lambda a: prepare_batch(a))

    # upload
    s = sp.add_parser("upload", help="Upload batch file")
    s.add_argument("--cache_dir", required=True)
    s.add_argument("--cache_name", default="batch.jsonl")
    s.set_defaults(func=lambda a: print(upload_file(a)))

    # create
    s = sp.add_parser("create", help="Create batch")
    s.add_argument("--file_id", required=True)
    s.add_argument("--completion_window", default="24h")
    s.add_argument("--description", default="CATiB annotation job")
    s.set_defaults(func=lambda a: print(create_batch(a)))

    # status
    s = sp.add_parser("status", help="Check batch status")
    s.add_argument("--batch_id", required=True)
    s.set_defaults(func=lambda a: check_status(a))

    # download
    s = sp.add_parser("download", help="Download results")
    s.add_argument("--output_file_id", required=True)
    s.add_argument("--cache_dir", required=True)
    s.add_argument("--out_dir", required=False)
    s.add_argument("--results_name", default="results.jsonl")

    # Optional post-processing args for prediction/CoNLLX export
    s.add_argument("--input_dir", required=False, help="Needed if you want to export prediction JSONL/CoNLLX.")
    s.add_argument("--pred_output_dir", required=False, help="Directory for per-file prediction JSONL.")
    s.add_argument("--conllx_output_dir", required=False, help="Directory for prediction CoNLLX files.")
    s.add_argument(
        "--sample-size",
        dest="sample_size",
        type=int,
        default=None,
        help="Must match the --sample-size used during prepare/run.",
    )

    s.set_defaults(func=lambda a: download_results(a))

    # run
    s = sp.add_parser(
        "run",
        help="Prepare+Upload+Create+Poll+Download+Benchmark+PredictionExport",
    )
    add_shared_prepare_args(s)

    s.add_argument("--out_dir", required=False)
    s.add_argument("--pred_output_dir", required=False, help="Directory for per-file prediction JSONL.")
    s.add_argument("--conllx_output_dir", required=False, help="Directory for prediction CoNLLX files.")

    s.add_argument("--completion_window", default="24h")
    s.add_argument("--description", default="CATiB annotation job")
    s.add_argument("--poll_seconds", type=int, default=60)
    s.add_argument("--timeout_seconds", type=int, default=48 * 3600)

    # cost optional
    s.add_argument("--price_in_per_1k", type=float, default=0.0025)
    s.add_argument("--price_out_per_1k", type=float, default=0.015)
    s.add_argument("--price_reasoning_per_1k", type=float, default=0.015)
    s.add_argument("--batch_discount", type=float, default=0.5)

    s.set_defaults(func=run_all)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()