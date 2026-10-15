#!/usr/bin/env python3
"""Decode raw Dev/Test with six models, two decoders, and explicit MSA-S31."""

from __future__ import annotations

import argparse
import gc
import hashlib
import pickle
import sys
from pathlib import Path
from types import MethodType

import pandas as pd
import torch
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.morphology.database import MorphologyDB
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.dediac import dediac_ar
from supar import Parser


SPLITS = ("Dev", "Test")
DECODERS = ("single", "multiroot")
CACHE_VERSION = 1


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camel-parser-root", required=True, type=Path)
    parser.add_argument("--gold-dir", required=True, type=Path)
    parser.add_argument("--model-table", required=True, type=Path)
    parser.add_argument("--morphology-db-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if len(rows) != 6:
        raise ValueError(f"Expected six models in {path}, found {len(rows)}")
    return rows


def write_conll(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def split(values: list, sizes: list[int]) -> list[list]:
    output = []
    start = 0
    for size in sizes:
        output.append(values[start : start + size])
        start += size
    if start != len(values):
        raise ValueError(f"Split sizes cover {start} items, received {len(values)}")
    return output


def build_s31_cache(camel_root: Path, gold_paths: list[Path], db_path: Path, cache: Path):
    from src.classes import TextParams
    from src.data_preparation import get_tagset, handle_text
    from src.initialize_disambiguator.bert_disambiguator import create_bert_disambiguator

    raw_lines: list[str] = []
    sizes: list[int] = []
    for gold_path in gold_paths:
        lines = metadata(gold_path, "text")
        raw_lines.extend(lines)
        sizes.append(len(lines))

    database = MorphologyDB(str(db_path))
    analyzer = Analyzer(database, backoff="ADD_PROP", cache_size=100000)
    disambiguator = create_bert_disambiguator(analyzer)
    clitic_feats = pd.read_csv(camel_root / "data" / "clitic_feats.csv")
    clitic_feats = clitic_feats.astype(str).astype(object)
    params = TextParams(
        raw_lines,
        "unused",
        CharMapper.builtin_mapper("arclean"),
        disambiguator,
        clitic_feats,
        get_tagset("catib"),
        "calima-msa-s31",
    )
    raw_tuples = handle_text(params)
    if len(raw_tuples) != len(raw_lines):
        raise ValueError(
            f"Raw sentence count changed during S31 preprocessing: "
            f"{len(raw_lines)} -> {len(raw_tuples)}"
        )
    payload = {
        "version": CACHE_VERSION,
        "morphology_database": "calima-msa-s31",
        "morphology_database_path": str(db_path.resolve()),
        "morphology_database_sha256": sha256(db_path),
        "disambiguator": "bert",
        "raw_lines": raw_lines,
        "sizes": sizes,
        "raw_tuples": raw_tuples,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(cache)
    return payload


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


def main() -> None:
    args = arguments()
    args.camel_parser_root = args.camel_parser_root.resolve()
    args.morphology_db_path = args.morphology_db_path.resolve()
    sys.path.insert(0, str(args.camel_parser_root))
    from src.conll_output import text_tuples_to_string
    from src.data_preparation import add_feats, get_feats_from_text_tuples
    from src.dependency_parser.biaff_parser import (
        _decode_multiroot,
        filter_tatweel,
        parser_conll_to_conll_tuples,
    )

    if not args.morphology_db_path.is_file():
        raise FileNotFoundError(args.morphology_db_path)
    gold_paths = [args.gold_dir / f"{split_name}.conllx" for split_name in SPLITS]
    for path in gold_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    expected_lines = sum((metadata(path, "text") for path in gold_paths), [])
    expected_sizes = [len(metadata(path, "text")) for path in gold_paths]
    expected_db_hash = sha256(args.morphology_db_path)
    if args.cache.is_file():
        with args.cache.open("rb") as stream:
            payload = pickle.load(stream)
    else:
        print("S31 RAW PREPROCESS START (one shared BERT disambiguation pass)", flush=True)
        payload = build_s31_cache(
            args.camel_parser_root, gold_paths, args.morphology_db_path, args.cache
        )
        print("S31 RAW PREPROCESS DONE", flush=True)

    required_cache_values = {
        "version": CACHE_VERSION,
        "morphology_database": "calima-msa-s31",
        "morphology_database_path": str(args.morphology_db_path),
        "morphology_database_sha256": expected_db_hash,
        "disambiguator": "bert",
        "raw_lines": expected_lines,
        "sizes": expected_sizes,
    }
    for key, expected in required_cache_values.items():
        if payload.get(key) != expected:
            raise ValueError(f"S31 cache mismatch for {key}")

    raw_lines = payload["raw_lines"]
    sizes = payload["sizes"]
    raw_tuples = payload["raw_tuples"]
    raw_parser_input = parser_input(raw_tuples, filter_tatweel)
    raw_feats = get_feats_from_text_tuples(raw_tuples)
    line_by_split = split(raw_lines, sizes)

    for generation, run_name, model_path in load_models(args.model_table):
        outputs = {
            decoder: [
                args.output_root / decoder / generation / run_name / f"{split_name}.conllx"
                for split_name in SPLITS
            ]
            for decoder in DECODERS
        }
        if all(
            path.is_file() and path.stat().st_size
            for paths in outputs.values()
            for path in paths
        ):
            print(f"PREDICT SKIP generation={generation} run={run_name}", flush=True)
            continue

        print(f"MODEL LOAD generation={generation} run={run_name}", flush=True)
        model = Parser.load(str(model_path))
        for decoder in DECODERS:
            decoder_outputs = outputs[decoder]
            if all(path.is_file() and path.stat().st_size for path in decoder_outputs):
                continue
            if decoder == "multiroot":
                model.model.decode = MethodType(_decode_multiroot, model.model)
            print(
                f"RAW PREDICT START decoder={decoder} generation={generation} run={run_name}",
                flush=True,
            )
            dataset = model.predict(raw_parser_input, verbose=False, tree=True, proj=True)
            parsed = parser_conll_to_conll_tuples(dataset)
            parsed = add_feats(parsed, raw_feats)
            parsed_by_split = split(parsed, sizes)
            for index, path in enumerate(decoder_outputs):
                write_conll(
                    path,
                    text_tuples_to_string(
                        parsed_by_split[index],
                        file_type="text",
                        sentences=line_by_split[index],
                    ),
                )
            del dataset, parsed, parsed_by_split
            print(
                f"RAW PREDICT DONE decoder={decoder} generation={generation} run={run_name}",
                flush=True,
            )

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("ALL S31 RAW PREDICTIONS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
