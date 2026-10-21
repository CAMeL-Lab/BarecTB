#!/usr/bin/env python3
"""Run CAMeL Parser gold/raw prediction with cached MSA-S31/BERT analysis."""

from __future__ import annotations

import argparse
import gc
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.dediac import dediac_ar
from supar import Parser

CACHE_VERSION = 2
MORPHOLOGY_DATABASE = "calima-msa-s31"


def validate_raw_cache(payload):
    expected = {
        "version": CACHE_VERSION,
        "morphology_database": MORPHOLOGY_DATABASE,
        "disambiguator": "bert",
    }
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Raw cache is not a verified MSA-S31/BERT cache. Use a new --cache path to preprocess with calima-msa-s31.")
    return payload["raw_lines"], payload["sizes"], payload["raw_tuples"]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camel-parser-root", required=True, type=Path)
    parser.add_argument("--gold-dir", required=True, type=Path)
    parser.add_argument("--model-table", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    return parser.parse_args()


def metadata(path: Path, key: str) -> list[str]:
    prefix = f"# {key} = "
    values = [
        line[len(prefix) :].rstrip("\n")
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
        if line.startswith(prefix)
    ]
    if not values:
        raise ValueError(f"No {key!r} metadata in {path}")
    return values


def write_conll(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_models(path: Path) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"{path}:{number}: expected generation, run name, path")
        generation, run_name, model_path = fields
        candidate = Path(model_path)
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise FileNotFoundError(candidate)
        rows.append((generation, run_name, candidate))
    if not rows:
        raise ValueError(f"No models in {path}")
    return rows


def build_raw_cache(camel_root: Path, gold_paths: list[Path], cache: Path):
    sys.path.insert(0, str(camel_root))
    from src.classes import TextParams
    from src.data_preparation import get_tagset, handle_text

    all_lines: list[str] = []
    sizes: list[int] = []
    for gold_path in gold_paths:
        lines = metadata(gold_path, "text")
        all_lines.extend(lines)
        sizes.append(len(lines))

    clitic_feats = pd.read_csv(camel_root / "data" / "clitic_feats.csv")
    clitic_feats = clitic_feats.astype(str).astype(object)
    params = TextParams(
        all_lines,
        "unused",
        CharMapper.builtin_mapper("arclean"),
        "bert",
        clitic_feats,
        get_tagset("catib"),
        MORPHOLOGY_DATABASE,
    )
    raw_tuples = handle_text(params)
    if len(raw_tuples) != len(all_lines):
        raise ValueError(
            f"Raw sentence count changed during preprocessing: {len(all_lines)} -> {len(raw_tuples)}"
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix + ".tmp")
    payload = {
        "version": CACHE_VERSION,
        "morphology_database": MORPHOLOGY_DATABASE,
        "disambiguator": "bert",
        "raw_lines": all_lines,
        "sizes": sizes,
        "raw_tuples": raw_tuples,
    }
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(cache)
    return all_lines, sizes, raw_tuples


def parser_input(raw_tuples, filter_tatweel):
    return [
        [
            (
                filter_tatweel(dediac_ar(token[1])),
                filter_tatweel(dediac_ar(token[2])),
                token[3],
            )
            for token in sentence
        ]
        for sentence in raw_tuples
    ]


def split(values: list, sizes: list[int]) -> list[list]:
    output = []
    start = 0
    for size in sizes:
        output.append(values[start : start + size])
        start += size
    if start != len(values):
        raise ValueError(f"Split sizes cover {start} items, received {len(values)}")
    return output


def main() -> None:
    args = arguments()
    args.camel_parser_root = args.camel_parser_root.resolve()
    sys.path.insert(0, str(args.camel_parser_root))
    from src.conll_output import text_tuples_to_string
    from src.data_preparation import add_feats, get_feats_from_text_tuples
    from src.dependency_parser.biaff_parser import (
        filter_tatweel,
        parser_conll_to_conll_tuples,
    )

    split_names = ["Dev", "Test"]
    gold_paths = [args.gold_dir / f"{name}.conllx" for name in split_names]
    for path in gold_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.cache.is_file():
        with args.cache.open("rb") as stream:
            raw_lines, sizes, raw_tuples = validate_raw_cache(pickle.load(stream))
    else:
        print("RAW PREPROCESS START (one shared BERT disambiguation pass)", flush=True)
        raw_lines, sizes, raw_tuples = build_raw_cache(
            args.camel_parser_root, gold_paths, args.cache
        )
        print("RAW PREPROCESS DONE", flush=True)

    expected_lines = sum((metadata(path, "text") for path in gold_paths), [])
    expected_sizes = [len(metadata(path, "text")) for path in gold_paths]
    if raw_lines != expected_lines or sizes != expected_sizes:
        raise ValueError("Cached raw analysis does not match current gold inputs")

    raw_parser_input = parser_input(raw_tuples, filter_tatweel)
    raw_feats = get_feats_from_text_tuples(raw_tuples)
    combined_gold = args.gold_dir / "Dev_Test.combined.conllx"
    # The source files have no terminal blank line. Two newlines are required
    # between their stripped contents so the last Dev tree and first Test tree
    # remain distinct CoNLL sentences.
    combined_gold.write_text(
        "\n\n".join(
            path.read_text(encoding="utf-8").rstrip() for path in gold_paths
        )
        + "\n\n",
        encoding="utf-8",
    )
    gold_sizes = [len(metadata(path, "text")) for path in gold_paths]

    for generation, run_name, model_path in load_models(args.model_table):
        outputs = {
            setup: [
                args.output_root / generation / setup / run_name / f"{name}.conllx"
                for name in split_names
            ]
            for setup in ("gold", "raw")
        }
        if all(path.is_file() and path.stat().st_size for paths in outputs.values() for path in paths):
            print(f"PREDICT SKIP generation={generation} run={run_name}", flush=True)
            continue

        print(f"MODEL LOAD generation={generation} run={run_name}", flush=True)
        model = Parser.load(str(model_path))

        if not all(path.is_file() and path.stat().st_size for path in outputs["gold"]):
            print(f"GOLD PREDICT START generation={generation} run={run_name}", flush=True)
            gold_dataset = model.predict(
                str(combined_gold), verbose=False, tree=True, proj=True
            )
            for index, sentence in enumerate(gold_dataset):
                gold_dataset[index].values[1] = [
                    filter_tatweel(form) for form in sentence.values[1]
                ]
            gold_parsed = parser_conll_to_conll_tuples(gold_dataset)
            gold_by_split = split(gold_parsed, gold_sizes)
            for index, path in enumerate(outputs["gold"]):
                write_conll(
                    path,
                    text_tuples_to_string(gold_by_split[index], file_type="conll"),
                )
            del gold_dataset, gold_parsed, gold_by_split
            print(f"GOLD PREDICT DONE generation={generation} run={run_name}", flush=True)

        if not all(path.is_file() and path.stat().st_size for path in outputs["raw"]):
            print(f"RAW PREDICT START generation={generation} run={run_name}", flush=True)
            raw_dataset = model.predict(
                raw_parser_input, verbose=False, tree=True, proj=True
            )
            raw_parsed = parser_conll_to_conll_tuples(raw_dataset)
            raw_parsed = add_feats(raw_parsed, raw_feats)
            raw_by_split = split(raw_parsed, sizes)
            line_by_split = split(raw_lines, sizes)
            for index, path in enumerate(outputs["raw"]):
                write_conll(
                    path,
                    text_tuples_to_string(
                        raw_by_split[index],
                        file_type="text",
                        sentences=line_by_split[index],
                    ),
                )
            del raw_dataset, raw_parsed, raw_by_split, line_by_split
            print(f"RAW PREDICT DONE generation={generation} run={run_name}", flush=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("ALL PREDICTIONS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
