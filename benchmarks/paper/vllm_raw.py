#!/usr/bin/env python3
"""
CATiB parsing from RAW TEXT (model performs tokenization) with guided JSON + few-shot examples,
PLUS benchmark instrumentation (speed + memory + reliability) including multi-GPU sampling.

Adds:
- Per-sentence metrics JSONL: latency, attempts, token usage (if provided), RSS peak, GPU samples
- Per-file summary JSON: p50/p95/p99 latency, sentences/sec, tokens/sec (if usage available), failure rates, peak memory
- Run summary JSON: aggregated across files

Example:
  python3 catib_rawtext_bench.py \
    --config prompt.yaml --port 8000 --model_name <model> \
    --input_dir test_dir --output_dir out \
    --train_conllx train.conllx --match_file matches.json \
    --num_shots 5 \
    --gpu_metrics --gpu_ids 0,1,2,3
"""

import os
import yaml
import argparse
import json
import hashlib
import time
import subprocess
from openai import OpenAI, APITimeoutError
from conllu import parse_incr
from typing import List, Tuple, Dict, Any, Optional

# --------------------------------------------------------------------------------------
# CONFIG FOR TIMEOUT / RETRIES
# --------------------------------------------------------------------------------------

REQUEST_TIMEOUT = 1000.0  # seconds per request
MAX_RETRIES = 3           # how many times to retry on timeout

# --------------------------------------------------------------------------------------
# CLI arguments
# --------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CATiB parsing (raw text tokenization) with guided JSON + few-shot examples + benchmark metrics."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_conllx", required=True)
    parser.add_argument("--match_file", required=True)
    parser.add_argument("--conllx_output_dir", required=False)

    parser.add_argument(
        "--num_shots",
        type=int,
        default=5,
        help="Number of few-shot examples to use per sentence (take the first N matches). Use 0 for zero-shot.",
    )

    # Metrics / benchmarking
    parser.add_argument("--metrics_dir", required=False, default=None,
                        help="Directory to write per-sentence metrics JSONL and summary JSON. Default: <output_dir>_metrics")
    parser.add_argument("--warmup_sentences", type=int, default=0,
                        help="Number of initial successful sentences to exclude from aggregate latency/throughput stats (still logged).")

    # Multi-GPU sampling
    parser.add_argument("--gpu_metrics", action="store_true",
                        help="If set, sample GPU memory/util/power using nvidia-smi per sentence (adds overhead).")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU ids to sample (e.g. '0,1,2,3'). If omitted and --gpu_metrics is set, defaults to '0,1,2,3'.")
    parser.add_argument(
        "--sample-size",
        dest="sample_size",
        type=int,
        default=None,
        help="Process only the first N sentences per input file. Useful for pilot tests.",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore existing outputs/metrics and start from scratch.",
    )
    return parser.parse_args()

# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------

ALLOWED_DEPRELS = ["SBJ", "OBJ", "PRD", "TPC", "IDF", "TMZ", "MOD", "---"]

def parse_conllx_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Reads a CoNLL-X file and returns a list of examples, each with:
      - text: raw sentence string from metadata 'text'
      - tokens: treeTokens (space-separated) from metadata 'treeTokens'
      - parses: list of (id, form, head, deprel) from the gold tree
    """
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            text = tokenlist.metadata.get("text")
            tree_tokens = tokenlist.metadata.get("treeTokens")
            if text and tree_tokens:
                tokens = tree_tokens.split()
                parses = []
                for token in tokenlist:
                    tok_id = int(token["id"])
                    form = token["form"]
                    head = int(token["head"])
                    deprel = token["deprel"]
                    parses.append((tok_id, form, head, deprel))
                examples.append({"text": text, "tokens": tokens, "parses": parses})
    return examples

def parses_to_conllx(tokens: List[str], parses: List[Tuple[int, str, int, str]], text: str) -> str:
    lines = [f"# text = {text}", f"# treeTokens = {' '.join(tokens)}"]
    for tok_id, form, head, deprel in parses:
        lines.append(f"{tok_id}\t{form}\t_\t_\t_\t_\t{head}\t{deprel}\t_\t_")
    return "\n".join(lines) + "\n\n"

def parses_to_arrays(parses: List[Tuple[int, str, int, str]]):
    head = [p[2] for p in parses]
    deprel = [p[3] for p in parses]
    return head, deprel

def validate_parses(parses: List[Tuple[int, str, int, str]]):
    """
    Basic structural validation:
      - ids must be 1..n
      - head in [0..n]
      - deprel in allowed set
    """
    n = len(parses)
    for i, (tok_id, _form, head, deprel) in enumerate(parses, start=1):
        assert tok_id == i, f"Non-consecutive or wrong token id: expected {i}, got {tok_id}"
        assert 0 <= head <= n, f"Head index out of range: {head} not in [0, {n}]"
        assert deprel in ALLOWED_DEPRELS, f"Invalid deprel: {deprel}"

# --------------------------------------------------------------------------------------
# Benchmark helpers
# --------------------------------------------------------------------------------------

def now_s() -> float:
    return time.perf_counter()

def safe_percentile(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] * (c - k) + ys[c] * (k - f)

def get_process_rss_bytes() -> Optional[int]:
    try:
        import psutil  # type: ignore
        p = psutil.Process(os.getpid())
        return int(p.memory_info().rss)
    except Exception:
        try:
            import resource  # type: ignore
            ru = resource.getrusage(resource.RUSAGE_SELF)
            v = int(ru.ru_maxrss)
            # Linux: KiB, macOS: bytes (best-effort detection)
            if v < 10_000_000:
                return v * 1024
            return v
        except Exception:
            return None

def query_gpu_nvidia_smi_one(gpu_id: int) -> Dict[str, Optional[float]]:
    try:
        cmd = [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) < 3:
            return {"gpu_mem_used_mib": None, "gpu_util_pct": None, "gpu_power_w": None}
        return {
            "gpu_mem_used_mib": float(parts[0]),
            "gpu_util_pct": float(parts[1]),
            "gpu_power_w": float(parts[2]),
        }
    except Exception:
        return {"gpu_mem_used_mib": None, "gpu_util_pct": None, "gpu_power_w": None}

def query_multiple_gpus(gpu_ids: List[int]) -> Dict[int, Dict[str, Optional[float]]]:
    return {gid: query_gpu_nvidia_smi_one(gid) for gid in gpu_ids}

def parse_gpu_ids_arg(arg: Optional[str]) -> List[int]:
    if arg is None or arg.strip() == "":
        return [0, 1, 2, 3]  # default for your 4-GPU node
    return [int(x.strip()) for x in arg.split(",") if x.strip() != ""]

def extract_usage_tokens(completion_obj: Any):
    """
    Best-effort extraction of prompt/completion/total tokens from OpenAI-compatible response.
    Returns (prompt_tokens, completion_tokens, total_tokens) or (None,None,None).
    """
    try:
        usage = getattr(completion_obj, "usage", None)
        if usage is None and isinstance(completion_obj, dict):
            usage = completion_obj.get("usage")
        if usage is None:
            return (None, None, None)

        def get_field(u, key):
            if isinstance(u, dict):
                return u.get(key)
            return getattr(u, key, None)

        pt = get_field(usage, "prompt_tokens")
        ct = get_field(usage, "completion_tokens")
        tt = get_field(usage, "total_tokens")
        return (
            int(pt) if pt is not None else None,
            int(ct) if ct is not None else None,
            int(tt) if tt is not None else None,
        )
    except Exception:
        return (None, None, None)

# --------------------------------------------------------------------------------------
# Prompt generation / guided JSON schema
# --------------------------------------------------------------------------------------

def build_guided_schema() -> Dict[str, Any]:
    """
    Model performs tokenization from raw text, so we do not constrain count/forms with const.
    """
    item_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "form": {"type": "string"},
            "head": {"type": "integer"},
            "deprel": {"type": "string", "enum": ALLOWED_DEPRELS},
        },
        "required": ["id", "form", "head", "deprel"],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "parses": {
                "type": "array",
                "items": item_schema,
                "minItems": 1
            }
        },
        "required": ["parses"],
        "additionalProperties": False,
    }

def annotate_sentence_guided(
    client,
    model_name: str,
    system_prompt: str,
    text: str,
    shots: List[Dict[str, Any]],
    timeout: float = REQUEST_TIMEOUT,
    max_retries: int = MAX_RETRIES,
) -> Tuple[List[Tuple[int, str, int, str]], Dict[str, Any]]:
    """
    Call the model on RAW TEXT (not pre-tokenized).
    Returns (parses, bench_info).
    """
    schema = build_guided_schema()

    few_shot_block = ""
    for shot in shots:
        few_shot_block += f"Input sentence:\n  {shot['text']}\n"
        few_shot_block += (
            "Output:\n  "
            + json.dumps(
                {
                    "parses": [
                        dict(id=tok_id, form=form, head=head, deprel=deprel)
                        for tok_id, form, head, deprel in shot["parses"]
                    ]
                },
                ensure_ascii=False,
            )
            + "\n\n"
        )

    if shots:
        user_prompt = (
            "Here are some annotated examples:\n\n"
            + few_shot_block
            + "Now annotate the following Arabic sentence using CATiB.\n"
            f"Sentence:\n{text}"
        )
    else:
        user_prompt = (
            "Now annotate the following Arabic sentence using CATiB.\n"
            f"Sentence:\n{text}"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    t0 = now_s()
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_body={"guided_json": schema},
                timeout=timeout * attempt,
            )
            obj = json.loads(completion.choices[0].message.content)
            parses = [(it["id"], it["form"], it["head"], it["deprel"]) for it in obj["parses"]]
            # validate_parses(parses)

            t1 = now_s()
            pt, ct, tt = extract_usage_tokens(completion)

            bench_info = {
                "attempts": attempt,
                "request_latency_s": t1 - t0,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
            }
            return parses, bench_info

        except APITimeoutError as e:
            print(f"  ⏱️ Timeout on attempt {attempt}/{max_retries} for this sentence: {e}")
            if attempt == max_retries:
                raise

        except Exception as e:
            print(f"  ❗ Error parsing model output on attempt {attempt}: {e}")
            raise

    raise RuntimeError("Failed after retries")

def load_jsonl_objects_and_repair(path: str) -> List[Dict[str, Any]]:
    """
    Load valid JSONL objects and truncate any partial/corrupt final line.
    This makes resume safer after Ctrl+C, timeout, or cluster preemption.
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
                print(f"⚠️ Found partial/corrupt JSONL line in {path}; truncating from byte {line_start}.")
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


def sync_conllx_from_jsonl(jsonl_path: str, conllx_path: str):
    """
    Rebuild CoNLL-X output from saved JSONL predictions.
    Important for this raw-text version because model-predicted tokens are stored in JSONL.
    """
    objs = load_jsonl_objects_and_repair(jsonl_path)

    with open(conllx_path, "w", encoding="utf-8") as cf:
        for obj in objs:
            cf.write(parses_to_conllx(obj["tokens"], obj["parses"], obj["text"]))

            
# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.sample_size is not None and args.sample_size <= 0:
        raise ValueError("--sample-size must be a positive integer")

    base_url = f"http://localhost:{args.port}/v1"
    client = OpenAI(base_url=base_url, api_key="-", max_retries=0)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if "system_prompt" not in config:
        raise KeyError("YAML config must contain a 'system_prompt' key")

    CATIB_SYSTEM_PROMPT = config["system_prompt"]
    _cache_key = hashlib.sha256(CATIB_SYSTEM_PROMPT.encode("utf-8")).hexdigest()

    model_name = args.model_name
    input_dir = args.input_dir
    output_dir = args.output_dir
    conllx_output_dir = args.conllx_output_dir or (output_dir + "_conllx")

    metrics_dir = args.metrics_dir or (output_dir + "_metrics")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(conllx_output_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    gpu_ids: List[int] = parse_gpu_ids_arg(args.gpu_ids) if args.gpu_metrics else []

    train_data = parse_conllx_file(args.train_conllx)
    with open(args.match_file, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    # Handle input_dir as either:
    #   1. a directory containing .conllx files
    #   2. a single .conllx file
    if os.path.isfile(input_dir):
        if not input_dir.endswith(".conllx"):
            raise ValueError(f"Input file must be a .conllx file: {input_dir}")

        input_files = [
            (os.path.basename(input_dir), input_dir)
        ]

    elif os.path.isdir(input_dir):
        input_files = [
            (filename, os.path.join(input_dir, filename))
            for filename in sorted(os.listdir(input_dir))
            if filename.endswith(".conllx")
        ]

        if not input_files:
            raise FileNotFoundError(f"No .conllx files found in directory: {input_dir}")

    else:
        raise FileNotFoundError(f"Input path does not exist: {input_dir}")

    # Global aggregates for this run only
    agg_latencies: List[float] = []
    agg_prompt_tokens = 0
    agg_completion_tokens = 0
    agg_total_tokens = 0
    agg_count_tokens = 0
    agg_success = 0
    agg_fail = 0
    agg_timeouts = 0
    agg_retries_total = 0

    agg_rss_peak_bytes = 0
    agg_gpu_mem_peak_mib: Dict[int, float] = {gid: 0.0 for gid in gpu_ids}
    agg_gpu_util_peak_pct: Dict[int, float] = {gid: 0.0 for gid in gpu_ids}
    agg_gpu_power_peak_w: Dict[int, float] = {gid: 0.0 for gid in gpu_ids}

    warmup_left = max(0, args.warmup_sentences)
    run_t_start = now_s()

    for filename, input_path in input_files:
        final_jsonl_path = os.path.join(output_dir, filename.replace(".conllx", ".jsonl"))
        final_conllx_path = os.path.join(conllx_output_dir, filename)

        metrics_path = os.path.join(metrics_dir, filename.replace(".conllx", ".metrics.jsonl"))
        summary_path = os.path.join(metrics_dir, filename.replace(".conllx", ".summary.json"))

        if args.restart:
            for p in (final_jsonl_path, final_conllx_path, metrics_path, summary_path):
                if os.path.exists(p):
                    os.remove(p)

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

        # JSONL is the only source of truth for resume.
        existing_output_objs = load_jsonl_objects_and_repair(final_jsonl_path)

        # Metrics are repaired only so the file remains valid JSONL.
        # Metrics are NOT used for resume.
        load_jsonl_objects_and_repair(metrics_path)

        # Rebuild CoNLL-X from JSONL to avoid duplicate/incomplete blocks.
        sync_conllx_from_jsonl(final_jsonl_path, final_conllx_path)

        completed_indices = set()

        for j, obj in enumerate(existing_output_objs, start=1):
            idx = obj.get("sentence_index_1based")

            # Backward-compatible fallback for old output JSONL without sentence_index_1based
            if not isinstance(idx, int):
                idx = j

            if isinstance(idx, int):
                completed_indices.add(idx)

        completed_indices = {idx for idx in completed_indices if 1 <= idx <= len(examples)}

        if len(completed_indices) >= len(examples):
            print(f"Skipping {filename}, already completed {len(completed_indices)}/{len(examples)} sentences.")
            continue

        print(f"Resuming {filename}: {len(completed_indices)}/{len(examples)} sentences already done.")

        # File-level aggregates for this run only
        file_latencies: List[float] = []
        file_prompt_tokens = 0
        file_completion_tokens = 0
        file_total_tokens = 0
        file_count_tokens = 0
        file_success = 0
        file_fail = 0
        file_timeouts = 0
        file_retries_total = 0
        file_rss_peak_bytes = 0

        file_gpu_mem_peak_mib: Dict[int, float] = {gid: 0.0 for gid in gpu_ids}
        file_gpu_util_peak_pct: Dict[int, float] = {gid: 0.0 for gid in gpu_ids}
        file_gpu_power_peak_w: Dict[int, float] = {gid: 0.0 for gid in gpu_ids}

        file_t_start = now_s()

        with open(final_jsonl_path, "a", encoding="utf-8") as jf, \
             open(final_conllx_path, "a", encoding="utf-8") as cf, \
             open(metrics_path, "a", encoding="utf-8") as mf:

            for i, ex in enumerate(examples, start=1):
                if i in completed_indices:
                    continue

                print(f"[{filename}] Annotating sentence {i}/{len(examples)}...")

                rss_before = get_process_rss_bytes()
                gpus_before = query_multiple_gpus(gpu_ids) if args.gpu_metrics else {}

                sent_t_start = now_s()
                status = "ok"
                err_type = None
                err_msg = None
                bench_info: Optional[Dict[str, Any]] = None

                try:
                    # Match file may use basename, full input path, or absolute input path
                    match_key = filename
                    if match_key not in match_data and input_path in match_data:
                        match_key = input_path
                    elif match_key not in match_data and os.path.abspath(input_path) in match_data:
                        match_key = os.path.abspath(input_path)

                    if match_key not in match_data:
                        raise KeyError(
                            f"No match data found for {filename}. Tried keys: "
                            f"{filename}, {input_path}, {os.path.abspath(input_path)}"
                        )

                    sentence_matches = match_data[match_key][i - 1]

                    k = args.num_shots
                    if k is not None and k > 0:
                        sentence_matches = sentence_matches[:k]
                    else:
                        sentence_matches = []

                    shots = [
                        {
                            "text": train_data[m["train_index"]]["text"],
                            "tokens": train_data[m["train_index"]]["tokens"],
                            "parses": train_data[m["train_index"]]["parses"],
                        }
                        for m in sentence_matches
                        if 0 <= m["train_index"] < len(train_data)
                    ]

                    parses, bench_info = annotate_sentence_guided(
                        client=client,
                        model_name=model_name,
                        system_prompt=CATIB_SYSTEM_PROMPT,
                        text=ex["text"],
                        shots=shots,
                    )

                    tokens = [form for _, form, _, _ in parses]
                    head, deprel = parses_to_arrays(parses)

                    out_obj = {
                        "file": filename,
                        "sentence_index_1based": i,
                        "text": ex["text"],
                        "tokens": tokens,
                        "parses": parses,
                        "head": head,
                        "deprel": deprel,
                    }

                    # Save output immediately after each successful sentence.
                    # This JSONL write is the resume checkpoint.
                    jf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    fsync_file(jf)

                    cf.write(parses_to_conllx(tokens, parses, ex["text"]))
                    fsync_file(cf)

                    # Mark completed only after final JSONL + CoNLL-X are saved.
                    completed_indices.add(i)

                    file_success += 1
                    agg_success += 1

                    latency_s = float(bench_info["request_latency_s"])
                    is_warmup = warmup_left > 0

                    if is_warmup:
                        warmup_left -= 1
                    else:
                        file_latencies.append(latency_s)
                        agg_latencies.append(latency_s)

                    attempts = int(bench_info["attempts"])
                    file_retries_total += max(0, attempts - 1)
                    agg_retries_total += max(0, attempts - 1)

                    pt = bench_info.get("prompt_tokens")
                    ct = bench_info.get("completion_tokens")
                    tt = bench_info.get("total_tokens")

                    if pt is not None and ct is not None and tt is not None:
                        file_prompt_tokens += int(pt)
                        file_completion_tokens += int(ct)
                        file_total_tokens += int(tt)
                        file_count_tokens += 1

                        if not is_warmup:
                            agg_prompt_tokens += int(pt)
                            agg_completion_tokens += int(ct)
                            agg_total_tokens += int(tt)
                            agg_count_tokens += 1

                except APITimeoutError as e:
                    status = "fail"
                    err_type = "timeout"
                    err_msg = str(e)
                    file_fail += 1
                    agg_fail += 1
                    file_timeouts += 1
                    agg_timeouts += 1
                    print(f"  ⚠️ Timeout on sentence {i} in {filename}, skipping it: {e}")

                except Exception as sent_err:
                    status = "fail"
                    err_type = type(sent_err).__name__
                    err_msg = str(sent_err)
                    file_fail += 1
                    agg_fail += 1
                    print(f"  ⚠️ Error on sentence {i} in {filename}, skipping it: {sent_err}")

                finally:
                    sent_t_end = now_s()

                    rss_after = get_process_rss_bytes()
                    rss_peak = max([x for x in [rss_before, rss_after] if x is not None] or [0])
                    file_rss_peak_bytes = max(file_rss_peak_bytes, int(rss_peak))
                    agg_rss_peak_bytes = max(agg_rss_peak_bytes, int(rss_peak))

                    gpus_after = query_multiple_gpus(gpu_ids) if args.gpu_metrics else {}

                    if args.gpu_metrics:
                        for gid in gpu_ids:
                            for sample in (gpus_before.get(gid, {}), gpus_after.get(gid, {})):
                                if not sample:
                                    continue

                                mem = sample.get("gpu_mem_used_mib")
                                util = sample.get("gpu_util_pct")
                                pwr = sample.get("gpu_power_w")

                                if mem is not None:
                                    file_gpu_mem_peak_mib[gid] = max(
                                        file_gpu_mem_peak_mib.get(gid, 0.0),
                                        float(mem),
                                    )
                                    agg_gpu_mem_peak_mib[gid] = max(
                                        agg_gpu_mem_peak_mib.get(gid, 0.0),
                                        float(mem),
                                    )

                                if util is not None:
                                    file_gpu_util_peak_pct[gid] = max(
                                        file_gpu_util_peak_pct.get(gid, 0.0),
                                        float(util),
                                    )
                                    agg_gpu_util_peak_pct[gid] = max(
                                        agg_gpu_util_peak_pct.get(gid, 0.0),
                                        float(util),
                                    )

                                if pwr is not None:
                                    file_gpu_power_peak_w[gid] = max(
                                        file_gpu_power_peak_w.get(gid, 0.0),
                                        float(pwr),
                                    )
                                    agg_gpu_power_peak_w[gid] = max(
                                        agg_gpu_power_peak_w.get(gid, 0.0),
                                        float(pwr),
                                    )

                    # Metrics are saved for both success and failure.
                    # Metrics are logs only; they are NOT used for resume.
                    mobj = {
                        "file": filename,
                        "sentence_index_1based": i,
                        "num_shots": args.num_shots,
                        "sample_size": args.sample_size,
                        "status": status,
                        "error_type": err_type,
                        "error_msg": err_msg,
                        "client_elapsed_s": sent_t_end - sent_t_start,
                        "request_latency_s": bench_info.get("request_latency_s") if bench_info else None,
                        "attempts": bench_info.get("attempts") if bench_info else None,
                        "prompt_tokens": bench_info.get("prompt_tokens") if bench_info else None,
                        "completion_tokens": bench_info.get("completion_tokens") if bench_info else None,
                        "total_tokens": bench_info.get("total_tokens") if bench_info else None,
                        "process_rss_peak_bytes": rss_peak if rss_peak else None,
                    }

                    if args.gpu_metrics:
                        mobj["gpu_before"] = {
                            str(gid): gpus_before.get(gid, {})
                            for gid in gpu_ids
                        }
                        mobj["gpu_after"] = {
                            str(gid): gpus_after.get(gid, {})
                            for gid in gpu_ids
                        }

                    mf.write(json.dumps(mobj, ensure_ascii=False) + "\n")
                    fsync_file(mf)

        # Clean/rebuild CoNLL-X once more after this file.
        # Final JSONL remains the source of truth.
        sync_conllx_from_jsonl(final_jsonl_path, final_conllx_path)

        file_t_end = now_s()
        file_elapsed = file_t_end - file_t_start

        file_latency_p50 = safe_percentile(file_latencies, 50)
        file_latency_p95 = safe_percentile(file_latencies, 95)
        file_latency_p99 = safe_percentile(file_latencies, 99)

        file_sent_per_s = None
        if file_latencies and sum(file_latencies) > 0:
            file_sent_per_s = len(file_latencies) / sum(file_latencies)

        file_tokens_per_s = None
        if args.warmup_sentences == 0 and file_count_tokens > 0 and file_latencies and sum(file_latencies) > 0:
            file_tokens_per_s = file_total_tokens / sum(file_latencies)

        total_output_objs = load_jsonl_objects_and_repair(final_jsonl_path)
        total_metric_objs = load_jsonl_objects_and_repair(metrics_path)

        total_success_so_far = len(total_output_objs)
        total_fail_so_far = sum(1 for x in total_metric_objs if x.get("status") == "fail")
        total_timeout_so_far = sum(1 for x in total_metric_objs if x.get("error_type") == "timeout")

        file_summary = {
            "file": filename,
            "model_name": model_name,
            "port": args.port,
            "sample_size": args.sample_size,
            "num_sentences_available": num_sentences_available,
            "num_sentences_target_this_file": len(examples),
            "num_sentences_completed_so_far": len(total_output_objs),
            "num_success_so_far": total_success_so_far,
            "num_fail_so_far": total_fail_so_far,
            "timeout_failures_so_far": total_timeout_so_far,

            "num_success_this_run": file_success,
            "num_fail_this_run": file_fail,
            "timeout_failures_this_run": file_timeouts,
            "retries_total_this_run": file_retries_total,
            "elapsed_s_file_this_run": file_elapsed,
            "latency_s_p50_this_run": file_latency_p50,
            "latency_s_p95_this_run": file_latency_p95,
            "latency_s_p99_this_run": file_latency_p99,
            "sentences_measured_for_latency_this_run": len(file_latencies),
            "sentences_per_s_measured_this_run": file_sent_per_s,
            "usage_token_sentences_this_run": file_count_tokens,
            "prompt_tokens_sum_this_run": file_prompt_tokens,
            "completion_tokens_sum_this_run": file_completion_tokens,
            "total_tokens_sum_this_run": file_total_tokens,
            "tokens_per_s_measured_this_run": file_tokens_per_s,
            "process_rss_peak_bytes_this_run": file_rss_peak_bytes if file_rss_peak_bytes else None,
        }

        if args.gpu_metrics:
            file_summary.update({
                "gpu_ids": gpu_ids,
                "gpu_mem_peak_mib_this_run": {
                    str(gid): file_gpu_mem_peak_mib.get(gid, None)
                    for gid in gpu_ids
                },
                "gpu_util_peak_pct_this_run": {
                    str(gid): file_gpu_util_peak_pct.get(gid, None)
                    for gid in gpu_ids
                },
                "gpu_power_peak_w_this_run": {
                    str(gid): file_gpu_power_peak_w.get(gid, None)
                    for gid in gpu_ids
                },
            })

        with open(summary_path, "w", encoding="utf-8") as sf:
            sf.write(json.dumps(file_summary, ensure_ascii=False, indent=2) + "\n")

        print(
            f"✅ Finished current run for {filename}.\n"
            f"  JSONL: {final_jsonl_path}\n"
            f"  CoNLL-X: {final_conllx_path}\n"
            f"  Metrics: {metrics_path}\n"
            f"  Summary: {summary_path}"
        )

    # Run summary for this run only
    run_t_end = now_s()
    run_elapsed = run_t_end - run_t_start

    lat_p50 = safe_percentile(agg_latencies, 50)
    lat_p95 = safe_percentile(agg_latencies, 95)
    lat_p99 = safe_percentile(agg_latencies, 99)

    sum_measured_latency = sum(agg_latencies) if agg_latencies else 0.0
    sent_per_s = (len(agg_latencies) / sum_measured_latency) if sum_measured_latency > 0 else None

    tokens_per_s = None
    if args.warmup_sentences == 0 and agg_count_tokens > 0 and sum_measured_latency > 0:
        tokens_per_s = agg_total_tokens / sum_measured_latency

    run_summary = {
        "model_name": model_name,
        "port": args.port,
        "sample_size": args.sample_size,
        "elapsed_s_run_wallclock": run_elapsed,
        "success_sentences_this_run": agg_success,
        "fail_sentences_this_run": agg_fail,
        "timeout_failures_this_run": agg_timeouts,
        "retries_total_this_run": agg_retries_total,
        "warmup_sentences_excluded": args.warmup_sentences,
        "latency_s_p50_this_run": lat_p50,
        "latency_s_p95_this_run": lat_p95,
        "latency_s_p99_this_run": lat_p99,
        "sentences_measured_for_latency_this_run": len(agg_latencies),
        "sentences_per_s_measured_this_run": sent_per_s,
        "usage_token_sentences_measured_this_run": agg_count_tokens,
        "prompt_tokens_sum_measured_this_run": agg_prompt_tokens,
        "completion_tokens_sum_measured_this_run": agg_completion_tokens,
        "total_tokens_sum_measured_this_run": agg_total_tokens,
        "tokens_per_s_measured_this_run": tokens_per_s,
        "process_rss_peak_bytes_this_run": agg_rss_peak_bytes if agg_rss_peak_bytes else None,
    }

    if args.gpu_metrics:
        run_summary.update({
            "gpu_ids": gpu_ids,
            "gpu_mem_peak_mib_this_run": {
                str(gid): agg_gpu_mem_peak_mib.get(gid, None)
                for gid in gpu_ids
            },
            "gpu_util_peak_pct_this_run": {
                str(gid): agg_gpu_util_peak_pct.get(gid, None)
                for gid in gpu_ids
            },
            "gpu_power_peak_w_this_run": {
                str(gid): agg_gpu_power_peak_w.get(gid, None)
                for gid in gpu_ids
            },
        })

    run_summary_path = os.path.join(metrics_dir, "RUN.summary.json")
    with open(run_summary_path, "w", encoding="utf-8") as rf:
        rf.write(json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n")

    print(f"\n📌 Run summary written to: {run_summary_path}\n")

if __name__ == "__main__":
    main()