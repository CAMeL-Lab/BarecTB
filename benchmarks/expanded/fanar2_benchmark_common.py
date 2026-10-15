#!/usr/bin/env python3
"""Shared, dependency-light utilities for the expanded-BAREC Fanar2 benchmark."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

RUN_ROOT = Path(
    os.environ.get(
        "BAREC_BENCH_RUN_ROOT", str(REPO_ROOT / "outputs" / "expanded")
    )
)
DATA_ROOT = Path(os.environ.get("BAREC_BENCH_DATA_ROOT", str(REPO_ROOT / "data" / "benchmark")))
MODEL_PATH = Path(
    os.environ.get(
        "BAREC_BENCH_MODEL_PATH",
        "QCRI/Fanar-2-27B-Instruct",
    )
)
MODEL_ID = os.environ.get("BAREC_BENCH_MODEL_ID", "QCRI/Fanar-2-27B-Instruct")
GOLD_PROMPT_PATH = REPO_ROOT / "benchmarks" / "prompts" / "gold.yaml"
RAW_PROMPT_PATH = REPO_ROOT / "benchmarks" / "prompts" / "raw.yaml"
CAMEL_CONLL_ROOT = Path(os.environ.get("CAMEL_CONLL_ROOT", str(REPO_ROOT / "external" / "camel_conll")))
CAMEL_CONLL_SCRIPT = CAMEL_CONLL_ROOT / "conll_evaluation.py"
DEPRELS = ("SBJ", "OBJ", "PRD", "TPC", "IDF", "TMZ", "MOD", "---")
POOL_PATHS = {
    "barectb": DATA_ROOT / "train" / "barectb.conllx",
    "patb_camel": DATA_ROOT / "train" / "patb_camel.conllx",
    "patb_camel_barec": DATA_ROOT / "train" / "patb_camel_barec.conllx",
}
SPLIT_PATHS = {
    "dev": DATA_ROOT / "gold" / "Dev.conllx",
    "test": DATA_ROOT / "gold" / "Test.conllx",
}


@dataclass(frozen=True)
class Sentence:
    text: str
    tree_tokens: list[str]
    rows: list[list[str]]

    @property
    def parse_objects(self) -> list[dict[str, Any]]:
        return [
            {
                "id": int(row[0]),
                "form": row[1],
                "head": int(row[6]),
                "deprel": row[7],
            }
            for row in self.rows
        ]


def read_conll(path: Path) -> list[Sentence]:
    sentences: list[Sentence] = []
    text = ""
    tree_tokens: list[str] = []
    rows: list[list[str]] = []

    def finish() -> None:
        nonlocal text, tree_tokens, rows
        if not text and not rows:
            return
        if not tree_tokens:
            tree_tokens = [row[1] for row in rows]
        if len(tree_tokens) != len(rows):
            raise ValueError(
                f"{path}: treeTokens/row mismatch in sentence {len(sentences) + 1}: "
                f"{len(tree_tokens)} != {len(rows)}"
            )
        if not text:
            text = " ".join(tree_tokens)
        sentences.append(Sentence(text=text, tree_tokens=tree_tokens, rows=rows))
        text, tree_tokens, rows = "", [], []

    with path.open(encoding="utf-8-sig") as source:
        for raw_line in source:
            line = raw_line.rstrip("\r\n")
            if not line:
                finish()
            elif line.startswith("# text = "):
                text = line[len("# text = ") :]
            elif line.startswith("# treeTokens = "):
                tree_tokens = line[len("# treeTokens = ") :].split()
            elif not line.startswith("#"):
                fields = line.split("\t")
                if len(fields) != 10:
                    raise ValueError(f"{path}: expected 10 columns: {line!r}")
                if "-" not in fields[0] and "." not in fields[0]:
                    rows.append(fields)
    finish()
    return sentences


def load_system_prompts() -> dict[str, str]:
    return {
        "gold": yaml.safe_load(GOLD_PROMPT_PATH.read_text(encoding="utf-8"))[
            "system_prompt"
        ],
        "raw": yaml.safe_load(RAW_PROMPT_PATH.read_text(encoding="utf-8"))[
            "system_prompt"
        ],
    }


def build_schema(mode: str, sentence: Sentence) -> dict[str, Any]:
    if mode == "gold":
        n = len(sentence.tree_tokens)
        prefix_items = []
        for index, token in enumerate(sentence.tree_tokens, start=1):
            prefix_items.append(
                {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "enum": [index]},
                        "form": {"type": "string", "enum": [token]},
                        # Semantically identical to minimum=0/maximum=n, but
                        # avoids an xgrammar numeric-range dead state observed
                        # with this tokenizer.
                        "head": {"type": "integer", "enum": list(range(n + 1))},
                        "deprel": {"type": "string", "enum": list(DEPRELS)},
                    },
                    "required": ["id", "form", "head", "deprel"],
                    "additionalProperties": False,
                }
            )
        return {
            "type": "object",
            "properties": {
                "parses": {
                    "type": "array",
                    "minItems": n,
                    "maxItems": n,
                    "prefixItems": prefix_items,
                    "items": False,
                }
            },
            "required": ["parses"],
            "additionalProperties": False,
        }

    if mode != "raw":
        raise ValueError(mode)
    return {
        "type": "object",
        "properties": {
            "parses": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "form": {"type": "string"},
                        "head": {"type": "integer"},
                        "deprel": {"type": "string", "enum": list(DEPRELS)},
                    },
                    "required": ["id", "form", "head", "deprel"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["parses"],
        "additionalProperties": False,
    }


def build_messages(
    mode: str,
    sentence: Sentence,
    shots: list[Sentence],
    system_prompt: str,
) -> list[dict[str, str]]:
    # These strings intentionally reproduce the original Fanar2 drivers.
    few_shot = ""
    if mode == "gold":
        for shot in shots:
            few_shot += (
                "Input:\n  "
                + json.dumps(shot.tree_tokens, ensure_ascii=False)
                + "\nOutput:\n  "
                + json.dumps({"parses": shot.parse_objects}, ensure_ascii=False)
                + "\n\n"
            )
        if shots:
            user_prompt = (
                "Here are some annotated examples (CATiB dependencies).\n\n"
                + few_shot
                + "Now annotate the following Arabic sentence using CATiB dependencies.\n"
                + "Predict ONLY head and deprel for each token.\n"
                + "Tokens:\n"
                + json.dumps(sentence.tree_tokens, ensure_ascii=False)
            )
        else:
            user_prompt = (
                "Annotate the following Arabic sentence using CATiB dependencies.\n"
                "Predict ONLY head and deprel for each token.\n"
                "Tokens:\n"
                + json.dumps(sentence.tree_tokens, ensure_ascii=False)
            )
    elif mode == "raw":
        for shot in shots:
            few_shot += (
                f"Input sentence:\n  {shot.text}\nOutput:\n  "
                + json.dumps({"parses": shot.parse_objects}, ensure_ascii=False)
                + "\n\n"
            )
        if shots:
            user_prompt = (
                "Here are some annotated examples:\n\n"
                + few_shot
                + "Now annotate the following Arabic sentence using CATiB.\n"
                + f"Sentence:\n{sentence.text}"
            )
        else:
            user_prompt = (
                "Now annotate the following Arabic sentence using CATiB.\n"
                f"Sentence:\n{sentence.text}"
            )
    else:
        raise ValueError(mode)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def condition_specs() -> list[dict[str, Any]]:
    specs = []
    for shots, pools in ((0, ("none",)), (10, tuple(POOL_PATHS))):
        for pool in pools:
            for split in ("dev", "test"):
                for mode in ("gold", "raw"):
                    suffix = "0shot" if shots == 0 else f"10shot_{pool}"
                    specs.append(
                        {
                            "condition": f"{mode}_{split}_{suffix}",
                            "mode": mode,
                            "split": split,
                            "shots": shots,
                            "pool": pool,
                        }
                    )
    return specs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
