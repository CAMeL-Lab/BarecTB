#!/usr/bin/env python3
"""
CATiB parsing with Gemini API using RAW TEXT input.

Features:
- The LLM receives raw sentence text and decides tokenization itself.
- Handles input_dir as either:
  1. a directory containing .conllx files
  2. a single .conllx file
- Supports --sample-size for pilot runs.
- Supports --restart to start clean.
- Can resume from where it stopped:
  - final JSONL is the only resume checkpoint
  - metrics are logs only and are NOT used for resume
- Saves output sentence-by-sentence.
- Rebuilds CoNLL-X predictions from final JSONL.
- Input/output prices are configurable with:
  --price-input-per-1m
  --price-output-per-1m

Behavior:
- Attempt 1: no schema, strict structural validation.
- Attempt 2: schema, no strict validation; save whatever comes back.

Special handling:
- In JSONL, token forms are kept exactly as returned by the model.
- In CoNLL-X, token forms are sanitized by removing whitespace/control chars.

Requirements:
  pip install google-genai pyyaml conllu
  export GEMINI_API_KEY="..."

Example:
  python3 catib_bench_gemini_rawtext.py \
    --config prompt.yaml \
    --model_name gemini-3.1-pro-preview \
    --input_dir test.conllx \
    --output_dir out \
    --train_conllx train.conllx \
    --match_file matches.json \
    --num_shots 5 \
    --sample-size 20 \
    --price-input-per-1m 0.50 \
    --price-output-per-1m 3.00
"""

import os
import re
import json
import yaml
import time
import random
import argparse
import hashlib
import unicodedata
from typing import List, Tuple, Dict, Any, Optional

from conllu import parse_incr

from google import genai
from google.genai import types

# --------------------------------------------------------------------------------------
# Pricing defaults
# --------------------------------------------------------------------------------------

PRICE_INPUT_PER_1M = 0.50
PRICE_OUTPUT_PER_1M = 3.00

# --------------------------------------------------------------------------------------
# Task constants
# --------------------------------------------------------------------------------------

ALLOWED_DEPRELS = ["SBJ", "OBJ", "PRD", "TPC", "IDF", "TMZ", "MOD", "---"]

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="CATiB parsing benchmark with Gemini API from raw text + two-stage JSON + efficiency metrics."
    )

    p.add_argument("--config", required=True, help="YAML containing system_prompt.")
    p.add_argument("--model_name", required=True, help="e.g., gemini-3.1-pro-preview")
    p.add_argument("--input_dir", required=True, help="Directory of .conllx files OR a single .conllx file.")
    p.add_argument("--output_dir", required=True, help="Directory for JSONL outputs + global metrics.")
    p.add_argument("--train_conllx", required=True, help="Training CoNLLX used for few-shots.")
    p.add_argument("--match_file", required=True, help="JSON match file mapping test sentences to train indices.")
    p.add_argument("--conllx_output_dir", required=False, help="Optional directory for CoNLLX output.")

    p.add_argument("--num_shots", type=int, default=5)
    p.add_argument("--random_shots", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--sample-size",
        dest="sample_size",
        type=int,
        default=None,
        help="Process only the first N sentences per input file. Useful for pilot tests.",
    )

    p.add_argument(
        "--restart",
        action="store_true",
        help="Ignore existing JSONL/CoNLLX outputs and metrics; start from scratch.",
    )

    p.add_argument(
        "--price-input-per-1m",
        type=float,
        default=PRICE_INPUT_PER_1M,
        help="Input token price in USD per 1M tokens.",
    )

    p.add_argument(
        "--price-output-per-1m",
        type=float,
        default=PRICE_OUTPUT_PER_1M,
        help="Output token price in USD per 1M tokens.",
    )

    return p.parse_args()

# --------------------------------------------------------------------------------------
# Utility / metrics helpers
# --------------------------------------------------------------------------------------

def percentile(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None

    ys = sorted(xs)
    k = int(round((p / 100.0) * (len(ys) - 1)))
    k = max(0, min(k, len(ys) - 1))
    return ys[k]


def extract_usage(resp: Any) -> Dict[str, Optional[int]]:
    u = getattr(resp, "usage_metadata", None) or getattr(resp, "usageMetadata", None)

    if not u:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    def _get(*names):
        for n in names:
            if hasattr(u, n):
                v = getattr(u, n)

                if v is None:
                    continue

                try:
                    return int(v)
                except Exception:
                    pass

        return None

    in_tok = _get(
        "prompt_token_count",
        "promptTokenCount",
        "input_token_count",
        "inputTokenCount",
    )
    out_tok = _get(
        "candidates_token_count",
        "candidatesTokenCount",
        "output_token_count",
        "outputTokenCount",
    )
    tot_tok = _get(
        "total_token_count",
        "totalTokenCount",
        "total_tokens",
        "totalTokens",
    )

    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": tot_tok,
    }


def estimate_cost_usd(
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    price_input_per_1m: float = PRICE_INPUT_PER_1M,
    price_output_per_1m: float = PRICE_OUTPUT_PER_1M,
) -> Optional[float]:
    if input_tokens is None or output_tokens is None:
        return None

    return (
        (input_tokens / 1_000_000) * price_input_per_1m
        + (output_tokens / 1_000_000) * price_output_per_1m
    )


def timed_generate(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
):
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    t1 = time.perf_counter()
    return resp, (t1 - t0)


def summarize_attempt_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [r for r in rows if r.get("success") is True]

    lat = [
        r["latency_s"]
        for r in successful
        if isinstance(r.get("latency_s"), (int, float))
    ]
    in_tok = [
        r["input_tokens"]
        for r in successful
        if isinstance(r.get("input_tokens"), int)
    ]
    out_tok = [
        r["output_tokens"]
        for r in successful
        if isinstance(r.get("output_tokens"), int)
    ]
    tot_tok = [
        r["total_tokens"]
        for r in successful
        if isinstance(r.get("total_tokens"), int)
    ]
    costs = [
        r["cost_usd"]
        for r in successful
        if isinstance(r.get("cost_usd"), (int, float))
    ]

    out_tps = []
    tot_tps = []

    for r in successful:
        l = r.get("latency_s")
        ot = r.get("output_tokens")
        tt = r.get("total_tokens")

        if isinstance(l, (int, float)) and l > 0:
            if isinstance(ot, int):
                out_tps.append(ot / l)

            if isinstance(tt, int):
                tot_tps.append(tt / l)

    attempts = [
        r.get("attempt")
        for r in successful
        if r.get("attempt") in (1, 2)
    ]

    return {
        "n_successful_attempts": len(successful),
        "attempt_success_counts": {
            "attempt_1": sum(1 for a in attempts if a == 1),
            "attempt_2": sum(1 for a in attempts if a == 2),
        },
        "latency_s": {
            "mean": (sum(lat) / len(lat)) if lat else None,
            "median": sorted(lat)[len(lat) // 2] if lat else None,
            "p90": percentile(lat, 90) if lat else None,
            "p95": percentile(lat, 95) if lat else None,
        },
        "tokens": {
            "input_total": sum(in_tok) if in_tok else None,
            "output_total": sum(out_tok) if out_tok else None,
            "total_total": sum(tot_tok) if tot_tok else None,
            "input_mean": (sum(in_tok) / len(in_tok)) if in_tok else None,
            "output_mean": (sum(out_tok) / len(out_tok)) if out_tok else None,
            "total_mean": (sum(tot_tok) / len(tot_tok)) if tot_tok else None,
        },
        "throughput": {
            "output_tokens_per_sec_mean": (sum(out_tps) / len(out_tps)) if out_tps else None,
            "output_tokens_per_sec_median": sorted(out_tps)[len(out_tps) // 2] if out_tps else None,
            "total_tokens_per_sec_mean": (sum(tot_tps) / len(tot_tps)) if tot_tps else None,
            "total_tokens_per_sec_median": sorted(tot_tps)[len(tot_tps) // 2] if tot_tps else None,
        },
        "cost_usd": {
            "total": sum(costs) if costs else None,
            "mean_per_attempt": (sum(costs) / len(costs)) if costs else None,
            "median_per_attempt": sorted(costs)[len(costs) // 2] if costs else None,
        },
    }


def load_jsonl_objects_and_repair(path: str) -> List[Dict[str, Any]]:
    """
    Load valid JSONL objects and truncate a partial/corrupt final line.
    This makes resume safer after Ctrl+C, timeout, or job preemption.
    """
    if not os.path.exists(path):
        return []

    objects: List[Dict[str, Any]] = []
    valid_until = 0

    with open(path, "rb") as f:
        while True:
            line_start = f.tell()
            line = f.readline()

            if not line:
                valid_until = f.tell()
                break

            try:
                obj = json.loads(line.decode("utf-8"))
                objects.append(obj)
                valid_until = f.tell()
            except Exception:
                print(
                    f"⚠️ Found partial/corrupt JSONL line in {path}; "
                    f"truncating from byte {line_start}."
                )
                break

    with open(path, "rb+") as f:
        f.truncate(valid_until)

    return objects


def fsync_file(f):
    """
    Flush and force file contents to disk after each sentence.
    """
    f.flush()
    os.fsync(f.fileno())


def get_input_files(input_path: str) -> List[Tuple[str, str]]:
    """
    Normalize input path to:
      [(filename, full_path), ...]

    Supports:
      - single .conllx file
      - directory containing .conllx files
    """
    if os.path.isfile(input_path):
        if not input_path.endswith(".conllx"):
            raise ValueError(f"Input file must be a .conllx file: {input_path}")

        return [(os.path.basename(input_path), input_path)]

    if os.path.isdir(input_path):
        input_files = [
            (filename, os.path.join(input_path, filename))
            for filename in sorted(os.listdir(input_path))
            if filename.endswith(".conllx")
        ]

        if not input_files:
            raise FileNotFoundError(f"No .conllx files found in directory: {input_path}")

        return input_files

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def get_match_key(match_data: Dict[str, Any], filename: str, input_path: str) -> str:
    """
    Match file may use basename, full path, or absolute path.
    """
    candidates = [
        filename,
        input_path,
        os.path.abspath(input_path),
    ]

    for key in candidates:
        if key in match_data:
            return key

    raise KeyError(
        f"No match data found for {filename}. Tried keys: "
        f"{filename}, {input_path}, {os.path.abspath(input_path)}"
    )

# --------------------------------------------------------------------------------------
# CoNLL-X sanitization helpers
# --------------------------------------------------------------------------------------

def sanitize_form_for_conllx(form: str) -> str:
    """
    Keep JSON forms unchanged, but make FORM safe for CoNLL-X output.

    Removes:
    - all whitespace: spaces, tabs, newlines, carriage returns, etc.
    - Unicode control/format chars that can break formatting
      such as bidi marks or zero-width chars.
    """
    if form is None:
        return "_"

    s = str(form)

    # Remove Unicode control / formatting chars.
    # Cc = control chars, Cf = format chars.
    s = "".join(ch for ch in s if unicodedata.category(ch) not in {"Cc", "Cf"})

    # Remove all whitespace characters.
    s = re.sub(r"\s+", "", s, flags=re.UNICODE)

    return s if s else "_"

# --------------------------------------------------------------------------------------
# Data parsing / formatting
# --------------------------------------------------------------------------------------

def parse_conllx_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Returns list of examples:
      {
        "text": str,
        "tokens": [str],   # gold/reference tokens
        "parses": [(id, form, pos, head, deprel), ...]
      }

    For this raw-text benchmark:
      - tokens are kept only as gold/reference metadata
      - the model is prompted with text, not tokens
    """
    examples = []

    with open(filepath, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            text = tokenlist.metadata.get("text")

            parses = []
            row_tokens = []

            for token in tokenlist:
                tok_id = int(token["id"])
                form = token["form"]
                row_tokens.append(form)

                pos = (
                    token.get("upos")
                    or token.get("xpos")
                    or token.get("cpostag")
                    or token.get("postag")
                    or token.get("pos")
                    or "_"
                )
                head = int(token["head"])
                deprel = token["deprel"]

                parses.append((tok_id, form, pos, head, deprel))

            if not text:
                text = " ".join(row_tokens)

            gold_tree_tokens = tokenlist.metadata.get("treeTokens")
            tokens = gold_tree_tokens.split() if gold_tree_tokens else row_tokens

            examples.append({
                "text": text,
                "tokens": tokens,
                "parses": parses,
            })

    return examples


def parses_to_conllx(parses: List[Tuple[int, str, int, str]], text: str) -> str:
    """
    Writes CoNLL-X style block with POS '_'.

    Columns:
      ID FORM LEMMA CPOSTAG POSTAG FEATS HEAD DEPREL PHEAD PDEPREL

    Uses predicted tokenization from parses.
    JSON keeps original forms, but CoNLL-X sanitizes them.
    """
    pred_tokens = [sanitize_form_for_conllx(p[1]) for p in parses]

    lines = [
        f"# text = {text}",
        f"# treeTokens = {' '.join(pred_tokens)}",
    ]

    for tok_id, form, head, deprel in parses:
        safe_form = sanitize_form_for_conllx(form)
        lines.append(
            f"{tok_id}\t{safe_form}\t_\t_\t_\t_\t{head}\t{deprel}\t_\t_"
        )

    return "\n".join(lines) + "\n\n"


def parses_to_arrays(parses: List[Tuple[int, str, int, str]]):
    head = [p[2] for p in parses]
    deprel = [p[3] for p in parses]
    return head, deprel


def normalize_parse_rows(rows: Any) -> Optional[List[Tuple[int, str, int, str]]]:
    """
    Convert JSON-loaded parse rows back to tuple-like rows if possible.
    Expected row shape:
      [id, form, head, deprel]
    """
    try:
        if not isinstance(rows, list):
            return None

        out: List[Tuple[int, str, int, str]] = []

        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 4:
                return None

            tok_id, form, head, deprel = row
            out.append((int(tok_id), str(form), int(head), str(deprel)))

        return out

    except Exception:
        return None


def is_completed_output_obj(obj: Dict[str, Any]) -> bool:
    """
    Final JSONL is the resume checkpoint.

    We count a row as completed if it contains a saved model output.
    Fatal errors are not written to final JSONL in this version, so failed
    sentences are retried on the next run.
    """
    if obj.get("attempt_mode") == "fatal_error":
        return False

    if "parses" in obj:
        return True

    if "parses_best_effort" in obj:
        return True

    if "parses_schema_json" in obj:
        return True

    if "parses_schema_raw_text" in obj:
        return True

    return False


def sync_conllx_from_jsonl(jsonl_path: str, conllx_path: str):
    """
    Rebuild CoNLL-X prediction file from final JSONL.

    Final JSONL is the only source of truth.
    """
    objs = load_jsonl_objects_and_repair(jsonl_path)

    # Sort by sentence index when available.
    objs = sorted(
        objs,
        key=lambda x: x.get("sentence_index_1based", 10**18),
    )

    with open(conllx_path, "w", encoding="utf-8") as cf:
        for obj in objs:
            if not is_completed_output_obj(obj):
                continue

            text = obj.get("text")
            if not isinstance(text, str):
                continue

            parses = None

            if "parses" in obj:
                parses = normalize_parse_rows(obj.get("parses"))

            elif "parses_best_effort" in obj:
                parses = normalize_parse_rows(obj.get("parses_best_effort"))

            if parses is not None:
                cf.write(parses_to_conllx(parses, text))

# --------------------------------------------------------------------------------------
# Prompt + schema
# --------------------------------------------------------------------------------------

def build_user_prompt(text: str, shots: List[Dict[str, Any]]) -> str:
    few_shot_block = ""

    if shots:
        for shot in shots:
            few_shot_block += "Input:\n  " + json.dumps(shot["text"], ensure_ascii=False) + "\n"
            few_shot_block += (
                "Output:\n  "
                + json.dumps(
                    {
                        "parses": [
                            {
                                "id": tok_id,
                                "form": form,
                                "head": head,
                                "deprel": deprel,
                            }
                            for tok_id, form, _pos, head, deprel in shot["parses"]
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n\n"
            )

        prefix = "Here are some annotated examples (CATiB dependencies).\n\n" + few_shot_block

    else:
        prefix = ""

    return (
        prefix
        + "Now annotate the following Arabic sentence using CATiB dependencies.\n"
        + "The input is raw text, so you must first decide the tokenization and then predict head and deprel for each token.\n"
        + "Predict ONLY head and deprel for each token. Do NOT output POS.\n"
        + "Return ONLY valid JSON in this shape:\n"
        + '{"parses":[{"id":1,"form":"TOKEN1","head":0,"deprel":"SBJ"}, ...]}\n'
        + "Constraints:\n"
        + "- parses must be a non-empty list\n"
        + "- ids must be 1..n with no gaps\n"
        + "- form must be the token string for that id\n"
        + "- head must be an integer 0..n\n"
        + f"- deprel must be one of: {ALLOWED_DEPRELS}\n"
        + "Do not output POS or explanations.\n"
        + f"Sentence:\n{json.dumps(text, ensure_ascii=False)}"
    )


def build_simplified_schema() -> Dict[str, Any]:
    """
    Conservative schema for attempt 2.

    Enforces only:
      - parses exists
      - parses is array
      - each item has required keys with basic types
    """
    item = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "form": {"type": "string"},
            "head": {"type": "integer"},
            "deprel": {"type": "string"},
        },
        "required": ["id", "form", "head", "deprel"],
    }

    return {
        "type": "object",
        "properties": {
            "parses": {
                "type": "array",
                "items": item,
            }
        },
        "required": ["parses"],
    }

# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------

def validate_and_normalize_parses(obj: Dict[str, Any]) -> List[Tuple[int, str, int, str]]:
    """
    Structural validation only.
    Does NOT compare against gold tokenization.

    Important:
    - Keeps form exactly as returned by the model.
    - Only uses form.strip() to test whether it is effectively empty.
    """
    if not isinstance(obj, dict):
        raise ValueError("Top-level output must be a JSON object.")

    if "parses" not in obj or not isinstance(obj["parses"], list):
        raise ValueError("Top-level key 'parses' must exist and be a list.")

    parses = obj["parses"]

    if not parses:
        raise ValueError("parses must be non-empty.")

    by_id: Dict[int, Dict[str, Any]] = {}

    for it in parses:
        if not isinstance(it, dict):
            raise ValueError("Each parses item must be an object.")

        for k in ("id", "form", "head", "deprel"):
            if k not in it:
                raise ValueError(f"Missing key '{k}' in parses item.")

        tid = int(it["id"])

        if tid < 1:
            raise ValueError(f"id must be >= 1, got {tid}")

        if tid in by_id:
            raise ValueError(f"Duplicate id: {tid}")

        by_id[tid] = it

    n = len(parses)
    expected_ids = set(range(1, n + 1))

    if set(by_id.keys()) != expected_ids:
        raise ValueError(f"ids must be exactly 1..{n}")

    norm: List[Tuple[int, str, int, str]] = []

    for i in range(1, n + 1):
        it = by_id[i]

        form = str(it["form"])

        if not form.strip():
            raise ValueError(f"Empty form at id={i}")

        head = int(it["head"])

        if head < 0 or head > n:
            raise ValueError(f"Head out of range at id={i}: {head} (expected 0..{n})")

        deprel = str(it["deprel"])

        if deprel not in ALLOWED_DEPRELS:
            raise ValueError(f"Invalid deprel at id={i}: {deprel}")

        norm.append((i, form, head, deprel))

    return norm

# --------------------------------------------------------------------------------------
# Attempt logic
# --------------------------------------------------------------------------------------

def annotate_two_stage_with_metrics(
    client: genai.Client,
    model_name: str,
    system_prompt: str,
    text: str,
    shots: List[Dict[str, Any]],
    file_name: str,
    sent_idx: int,
    price_input_per_1m: float,
    price_output_per_1m: float,
) -> Tuple[Optional[List[Tuple[int, str, int, str]]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Attempt 1:
      - no schema
      - response_mime_type=application/json
      - strict structural validation

    Attempt 2:
      - simplified schema
      - no strict validation
      - save JSON if parseable, otherwise raw text
    """
    user_prompt = build_user_prompt(text, shots)
    attempt_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Attempt 1: no schema + strict structural validation
    # ------------------------------------------------------------------

    usage1 = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
    cost1 = None
    lat1 = None
    # print(system_prompt)
    # print(user_prompt)
    try:
        cfg1 = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
        )

        resp1, lat1 = timed_generate(client, model_name, user_prompt, cfg1)
        usage1 = extract_usage(resp1)
        cost1 = estimate_cost_usd(
            usage1["input_tokens"],
            usage1["output_tokens"],
            price_input_per_1m=price_input_per_1m,
            price_output_per_1m=price_output_per_1m,
        )
        raw1 = (resp1.text or "").strip()

        obj1 = json.loads(raw1)
        parses1 = validate_and_normalize_parses(obj1)

        attempt_rows.append({
            "file": file_name,
            "sent_idx": sent_idx,
            "attempt": 1,
            "success": True,
            "latency_s": lat1,
            **usage1,
            "cost_usd": cost1,
            "price_input_per_1m": price_input_per_1m,
            "price_output_per_1m": price_output_per_1m,
            "error": None,
            "validated": True,
            "json_parsed": True,
        })

        payload = {
            "mode": "attempt1_validated",
            "obj": obj1,
        }

        return parses1, payload, attempt_rows

    except Exception as e1:
        attempt_rows.append({
            "file": file_name,
            "sent_idx": sent_idx,
            "attempt": 1,
            "success": False,
            "latency_s": lat1,
            **usage1,
            "cost_usd": cost1,
            "price_input_per_1m": price_input_per_1m,
            "price_output_per_1m": price_output_per_1m,
            "error": f"{type(e1).__name__}: {e1}",
            "validated": False,
            "json_parsed": None,
        })

    # ------------------------------------------------------------------
    # Attempt 2: schema enforced, no strict validation
    # ------------------------------------------------------------------

    usage2 = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
    }
    cost2 = None
    lat2 = None

    schema = build_simplified_schema()

    cfg2 = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=schema,
    )

    user_prompt2 = user_prompt + "\n\nIMPORTANT: Return ONLY JSON. No extra text."

    resp2, lat2 = timed_generate(client, model_name, user_prompt2, cfg2)
    usage2 = extract_usage(resp2)
    cost2 = estimate_cost_usd(
        usage2["input_tokens"],
        usage2["output_tokens"],
        price_input_per_1m=price_input_per_1m,
        price_output_per_1m=price_output_per_1m,
    )
    raw2 = (resp2.text or "").strip()

    json_parsed = False
    payload: Dict[str, Any]

    try:
        obj2 = json.loads(raw2)
        json_parsed = True
        payload = {
            "mode": "attempt2_schema_json",
            "obj": obj2,
        }

    except Exception as e2:
        payload = {
            "mode": "attempt2_schema_raw_text",
            "text": raw2,
            "parse_error": f"{type(e2).__name__}: {e2}",
        }

    attempt_rows.append({
        "file": file_name,
        "sent_idx": sent_idx,
        "attempt": 2,
        "success": True,
        "latency_s": lat2,
        **usage2,
        "cost_usd": cost2,
        "price_input_per_1m": price_input_per_1m,
        "price_output_per_1m": price_output_per_1m,
        "error": None,
        "validated": False,
        "json_parsed": json_parsed,
    })

    return None, payload, attempt_rows

# --------------------------------------------------------------------------------------
# Best-effort conversion for attempt 2 JSON
# --------------------------------------------------------------------------------------

def best_effort_parses_from_obj(obj: Any) -> Optional[List[Tuple[int, str, int, str]]]:
    try:
        if not isinstance(obj, dict):
            return None

        parses = obj.get("parses")

        if not isinstance(parses, list) or not parses:
            return None

        by_id: Dict[int, Dict[str, Any]] = {}

        for it in parses:
            if not isinstance(it, dict):
                return None

            if "id" not in it or "form" not in it or "head" not in it or "deprel" not in it:
                return None

            tid = int(it["id"])
            by_id[tid] = it

        n = len(parses)
        out: List[Tuple[int, str, int, str]] = []

        for i in range(1, n + 1):
            it = by_id.get(i)

            if it is None:
                return None

            form = str(it.get("form", ""))

            if not form.strip():
                return None

            head = int(it.get("head", 0))
            deprel = str(it.get("deprel", "---"))

            if head < 0 or head > n:
                return None

            if deprel not in ALLOWED_DEPRELS:
                return None

            out.append((i, form, head, deprel))

        return out

    except Exception:
        return None

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.sample_size is not None and args.sample_size <= 0:
        raise ValueError("--sample-size must be a positive integer")

    if args.price_input_per_1m < 0:
        raise ValueError("--price-input-per-1m must be >= 0")

    if args.price_output_per_1m < 0:
        raise ValueError("--price-output-per-1m must be >= 0")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment.")

    client = genai.Client(api_key=api_key)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if "system_prompt" not in config:
        raise KeyError("YAML config must contain a 'system_prompt' key")

    SYSTEM_PROMPT = config["system_prompt"]

    _cache_key = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    _ = _cache_key

    output_dir = args.output_dir
    conllx_output_dir = args.conllx_output_dir or (output_dir + "_conllx")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(conllx_output_dir, exist_ok=True)

    metrics_jsonl_path = os.path.join(output_dir, "metrics.jsonl")
    summary_json_path = os.path.join(output_dir, "metrics_summary.json")

    train_data = parse_conllx_file(args.train_conllx)

    with open(args.match_file, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    if args.random_shots:
        random.seed(args.seed)

    input_files = get_input_files(args.input_dir)

    if args.restart:
        if os.path.exists(metrics_jsonl_path):
            os.remove(metrics_jsonl_path)

        if os.path.exists(summary_json_path):
            os.remove(summary_json_path)

    # Metrics are loaded for summary only.
    # They are NOT used for resume.
    all_metric_rows: List[Dict[str, Any]] = load_jsonl_objects_and_repair(metrics_jsonl_path)

    metrics_f = open(metrics_jsonl_path, "a", encoding="utf-8")

    try:
        for filename, input_path in input_files:
            final_jsonl_path = os.path.join(output_dir, filename.replace(".conllx", ".jsonl"))
            final_conllx_path = os.path.join(conllx_output_dir, filename)

            if args.restart:
                for pth in (final_jsonl_path, final_conllx_path):
                    if os.path.exists(pth):
                        os.remove(pth)

            examples_all = parse_conllx_file(input_path)
            num_sentences_available = len(examples_all)

            if args.sample_size is not None:
                examples = examples_all[:args.sample_size]
            else:
                examples = examples_all

            print(
                f"Loaded {len(examples)} examples from {input_path}"
                + (
                    f" using --sample-size {args.sample_size}; "
                    f"original file has {num_sentences_available} examples."
                    if args.sample_size is not None
                    else ""
                )
            )

            # Final JSONL is the only source of truth for resume.
            existing_output_objs = load_jsonl_objects_and_repair(final_jsonl_path)

            # Rebuild CoNLL-X from final JSONL before appending more output.
            sync_conllx_from_jsonl(final_jsonl_path, final_conllx_path)

            completed_indices = set()

            for j, obj in enumerate(existing_output_objs, start=1):
                if not is_completed_output_obj(obj):
                    continue

                idx = obj.get("sentence_index_1based")

                # Backward-compatible fallback for old JSONL without sentence_index_1based.
                if not isinstance(idx, int):
                    idx = j

                if isinstance(idx, int):
                    completed_indices.add(idx)

            completed_indices = {
                idx
                for idx in completed_indices
                if 1 <= idx <= len(examples)
            }

            if len(completed_indices) >= len(examples):
                print(
                    f"Skipping {filename}, already completed "
                    f"{len(completed_indices)}/{len(examples)} sentences."
                )
                continue

            print(
                f"Resuming {filename}: "
                f"{len(completed_indices)}/{len(examples)} sentences already done."
            )

            with open(final_jsonl_path, "a", encoding="utf-8") as jf, \
                 open(final_conllx_path, "a", encoding="utf-8") as cf:

                for i, ex in enumerate(examples, start=1):
                    if i in completed_indices:
                        continue

                    num_shots = max(0, args.num_shots)
                    random_shots = bool(args.random_shots)

                    print(
                        f"[{filename}] Annotating sentence {i}/{len(examples)} "
                        f"with {num_shots} shot(s) "
                        f"(random_shots={random_shots})..."
                    )

                    # Build few-shots.
                    try:
                        if num_shots <= 0:
                            shots: List[Dict[str, Any]] = []

                        elif random_shots:
                            k = min(num_shots, len(train_data))
                            selected_indices = random.sample(range(len(train_data)), k=k)

                            shots = [
                                {
                                    "text": train_data[idx]["text"],
                                    "tokens": train_data[idx]["tokens"],
                                    "parses": train_data[idx]["parses"],
                                }
                                for idx in selected_indices
                            ]

                        else:
                            match_key = get_match_key(match_data, filename, input_path)
                            sentence_matches = match_data[match_key][i - 1]
                            selected_matches = sentence_matches[:num_shots]

                            shots = []

                            for m in selected_matches:
                                tidx = m.get("train_index", -1)

                                if 0 <= tidx < len(train_data):
                                    shots.append({
                                        "text": train_data[tidx]["text"],
                                        "tokens": train_data[tidx]["tokens"],
                                        "parses": train_data[tidx]["parses"],
                                    })

                    except Exception as e:
                        print(f"  ⚠️ Error building shots; using 0 shots. Reason: {e}")
                        shots = []

                    try:
                        parses_norm, payload, attempt_rows = annotate_two_stage_with_metrics(
                            client=client,
                            model_name=args.model_name,
                            system_prompt=SYSTEM_PROMPT,
                            text=ex["text"],
                            shots=shots,
                            file_name=filename,
                            sent_idx=i,
                            price_input_per_1m=args.price_input_per_1m,
                            price_output_per_1m=args.price_output_per_1m,
                        )

                        # Metrics are logs only; not resume checkpoints.
                        for row in attempt_rows:
                            metrics_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            fsync_file(metrics_f)
                            all_metric_rows.append(row)

                        mode = payload.get("mode")

                        out_obj: Dict[str, Any] = {
                            "file": filename,
                            "sentence_index_1based": i,
                            "text": ex["text"],
                            "gold_tokens": ex["tokens"],
                            "attempt_mode": mode,
                            "attempts": 1 if mode == "attempt1_validated" else 2,
                        }

                        conllx_parse_to_write: Optional[List[Tuple[int, str, int, str]]] = None

                        if parses_norm is not None:
                            head, deprel = parses_to_arrays(parses_norm)
                            pred_tokens = [p[1] for p in parses_norm]

                            out_obj.update({
                                "predicted_tokens": pred_tokens,
                                "parses": parses_norm,
                                "head": head,
                                "deprel": deprel,
                            })

                            conllx_parse_to_write = parses_norm

                        else:
                            if mode == "attempt2_schema_json":
                                out_obj["parses_schema_json"] = payload.get("obj")

                                be = best_effort_parses_from_obj(payload.get("obj"))

                                if be is not None:
                                    head, deprel = parses_to_arrays(be)
                                    pred_tokens = [p[1] for p in be]

                                    out_obj["predicted_tokens_best_effort"] = pred_tokens
                                    out_obj["parses_best_effort"] = be
                                    out_obj["head_best_effort"] = head
                                    out_obj["deprel_best_effort"] = deprel

                                    conllx_parse_to_write = be

                            else:
                                out_obj["parses_schema_raw_text"] = payload.get("text")
                                out_obj["schema_json_parse_error"] = payload.get("parse_error")

                        # Save JSONL immediately.
                        # This is the resume checkpoint.
                        jf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                        fsync_file(jf)

                        if conllx_parse_to_write is not None:
                            cf.write(parses_to_conllx(conllx_parse_to_write, ex["text"]))
                            fsync_file(cf)

                        completed_indices.add(i)

                    except Exception as e:
                        # Do NOT write fatal errors to final JSONL.
                        # This means failed/unwritten sentences are retried next run.
                        fail_row = {
                            "file": filename,
                            "sent_idx": i,
                            "attempt": 0,
                            "success": False,
                            "latency_s": None,
                            "input_tokens": None,
                            "output_tokens": None,
                            "total_tokens": None,
                            "cost_usd": None,
                            "price_input_per_1m": args.price_input_per_1m,
                            "price_output_per_1m": args.price_output_per_1m,
                            "error": f"{type(e).__name__}: {e}",
                            "validated": False,
                            "json_parsed": None,
                        }

                        metrics_f.write(json.dumps(fail_row, ensure_ascii=False) + "\n")
                        fsync_file(metrics_f)
                        all_metric_rows.append(fail_row)

                        print(
                            f"  ⚠️ Error on sentence {i} in {filename}; "
                            f"will retry on next run: {e}"
                        )
                        continue

            # Rebuild CoNLL-X from JSONL after each file, sorted by sentence index.
            sync_conllx_from_jsonl(final_jsonl_path, final_conllx_path)

            print(f"✅ Finished current run for {filename}.")
            print(f"  JSONL:  {final_jsonl_path}")
            print(f"  CoNLLX: {final_conllx_path}")

    finally:
        metrics_f.close()

    all_metric_rows = load_jsonl_objects_and_repair(metrics_jsonl_path)
    summary = summarize_attempt_rows(all_metric_rows)

    summary["pricing"] = {
        "price_input_per_1m": args.price_input_per_1m,
        "price_output_per_1m": args.price_output_per_1m,
    }

    with open(summary_json_path, "w", encoding="utf-8") as sf:
        sf.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print(f"\n📈 Metrics written to: {metrics_jsonl_path}")
    print(f"📊 Summary written to:  {summary_json_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()