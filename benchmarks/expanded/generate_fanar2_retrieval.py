#!/usr/bin/env python3
"""Generate exact top-10 matches for the original custom chrF++ scorer quickly."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Hashable

import numpy as np
from scipy.sparse import csr_matrix

from fanar2_benchmark_common import POOL_PATHS, RUN_ROOT, SPLIT_PATHS, read_conll, write_json


def char_ngrams(tokens: list[str], n: int) -> Counter[str]:
    text = " ".join(tokens)
    return Counter(text[i : i + n] for i in range(len(text) - n + 1))


def word_ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def denominator(tokens: list[str], kind: str, n: int) -> int:
    length = len(" ".join(tokens)) if kind == "char" else len(tokens)
    return max(length - n + 1, 0)


def build_occurrence_matrix(
    sentences: list[list[str]], kind: str, n: int
) -> tuple[csr_matrix, dict[tuple[Hashable, int], int], np.ndarray]:
    grammer: Callable[[list[str], int], Counter[Any]] = (
        char_ngrams if kind == "char" else word_ngrams
    )
    denoms = np.asarray([denominator(x, kind, n) for x in sentences], dtype=np.int32)
    indptr = np.empty(len(sentences) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(denoms, dtype=np.int64, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    feature_ids: dict[tuple[Hashable, int], int] = {}
    cursor = 0
    for sentence in sentences:
        for gram, count in grammer(sentence, n).items():
            for level in range(1, count + 1):
                key = (gram, level)
                feature = feature_ids.get(key)
                if feature is None:
                    feature = len(feature_ids)
                    feature_ids[key] = feature
                indices[cursor] = feature
                cursor += 1
    if cursor != len(indices):
        raise AssertionError((cursor, len(indices), kind, n))
    matrix = csr_matrix(
        (np.ones(len(indices), dtype=np.int32), indices, indptr),
        shape=(len(sentences), len(feature_ids)),
        dtype=np.int32,
    )
    matrix.sort_indices()
    return matrix, feature_ids, denoms


def query_occurrence_matrix(
    sentences: list[list[str]],
    kind: str,
    n: int,
    feature_ids: dict[tuple[Hashable, int], int],
) -> tuple[csr_matrix, np.ndarray]:
    grammer: Callable[[list[str], int], Counter[Any]] = (
        char_ngrams if kind == "char" else word_ngrams
    )
    denoms = np.asarray([denominator(x, kind, n) for x in sentences], dtype=np.int32)
    rows: list[int] = []
    cols: list[int] = []
    for row, sentence in enumerate(sentences):
        for gram, count in grammer(sentence, n).items():
            for level in range(1, count + 1):
                feature = feature_ids.get((gram, level))
                if feature is not None:
                    rows.append(row)
                    cols.append(feature)
    matrix = csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(len(sentences), len(feature_ids)),
        dtype=np.int32,
    )
    return matrix, denoms


def chrfpp_score(reference: list[str], hypothesis: list[str]) -> float:
    groups = []
    for kind, orders, grammer in (
        ("char", range(1, 7), char_ngrams),
        ("word", range(1, 3), word_ngrams),
    ):
        precisions, recalls = [], []
        for n in orders:
            ref = grammer(reference, n)
            hyp = grammer(hypothesis, n)
            if not ref and not hyp:
                continue
            overlap = sum((ref & hyp).values())
            ref_count, hyp_count = sum(ref.values()), sum(hyp.values())
            precisions.append(overlap / hyp_count if hyp_count else 0.0)
            recalls.append(overlap / ref_count if ref_count else 0.0)
        groups.append(
            (
                sum(precisions) / len(precisions) if precisions else 0.0,
                sum(recalls) / len(recalls) if recalls else 0.0,
            )
        )
    precision = (groups[0][0] + groups[1][0]) / 2.0
    recall = (groups[0][1] + groups[1][1]) / 2.0
    return 0.0 if precision == recall == 0 else 5 * precision * recall / (4 * precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    started_epoch = time.time()
    started = time.perf_counter()
    output_dir = RUN_ROOT / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)

    patb_sentences = read_conll(POOL_PATHS["patb_camel"])
    barec_sentences = read_conll(POOL_PATHS["barectb"])
    combined_sentences = read_conll(POOL_PATHS["patb_camel_barec"])
    expected = patb_sentences + barec_sentences
    if [x.tree_tokens for x in combined_sentences] != [x.tree_tokens for x in expected]:
        raise ValueError("Combined retrieval pool is not PATB+CAMeL followed by expanded BAREC")
    train = [x.tree_tokens for x in combined_sentences]
    boundary = len(patb_sentences)
    query_by_split = {
        split: [x.tree_tokens for x in read_conll(path)] for split, path in SPLIT_PATHS.items()
    }

    matrices = []
    build_timings = {}
    for kind, orders in (("char", range(1, 7)), ("word", range(1, 3))):
        for n in orders:
            stage = time.perf_counter()
            matrix, features, train_denoms = build_occurrence_matrix(train, kind, n)
            matrices.append((kind, n, matrix, features, train_denoms))
            build_timings[f"{kind}_{n}"] = time.perf_counter() - stage
            print(
                f"built {kind}-{n}: shape={matrix.shape} nnz={matrix.nnz} "
                f"seconds={build_timings[f'{kind}_{n}']:.3f}",
                flush=True,
            )

    pool_slices = {
        "patb_camel": (0, boundary),
        "barectb": (boundary, len(train)),
        "patb_camel_barec": (0, len(train)),
    }
    all_results: dict[tuple[str, str], list[list[dict[str, Any]]]] = {}
    scoring_timings: dict[str, float] = {}
    max_error = 0.0

    for split, queries in query_by_split.items():
        stage = time.perf_counter()
        results = {pool: [] for pool in pool_slices}
        for start in range(0, len(queries), args.chunk_size):
            chunk = queries[start : start + args.chunk_size]
            shape = (len(chunk), len(train))
            sums = {
                "char_p": np.zeros(shape, dtype=np.float64),
                "char_r": np.zeros(shape, dtype=np.float64),
                "char_n": np.zeros(shape, dtype=np.uint8),
                "word_p": np.zeros(shape, dtype=np.float64),
                "word_r": np.zeros(shape, dtype=np.float64),
                "word_n": np.zeros(shape, dtype=np.uint8),
            }
            for kind, n, train_matrix, features, train_denoms in matrices:
                query_matrix, query_denoms = query_occurrence_matrix(chunk, kind, n, features)
                overlap = (query_matrix @ train_matrix.T).toarray().astype(np.float64)
                valid = (query_denoms[:, None] > 0) | (train_denoms[None, :] > 0)
                p = np.divide(
                    overlap,
                    train_denoms[None, :],
                    out=np.zeros_like(overlap),
                    where=train_denoms[None, :] > 0,
                )
                r = np.divide(
                    overlap,
                    query_denoms[:, None],
                    out=np.zeros_like(overlap),
                    where=query_denoms[:, None] > 0,
                )
                sums[f"{kind}_p"] += p
                sums[f"{kind}_r"] += r
                sums[f"{kind}_n"] += valid
            char_p = np.divide(sums["char_p"], sums["char_n"], where=sums["char_n"] > 0)
            char_r = np.divide(sums["char_r"], sums["char_n"], where=sums["char_n"] > 0)
            word_p = np.divide(sums["word_p"], sums["word_n"], where=sums["word_n"] > 0)
            word_r = np.divide(sums["word_r"], sums["word_n"], where=sums["word_n"] > 0)
            precision = (char_p + word_p) / 2.0
            recall = (char_r + word_r) / 2.0
            scores = np.divide(
                5 * precision * recall,
                4 * precision + recall,
                out=np.zeros_like(precision),
                where=(precision != 0) | (recall != 0),
            )
            for local_index, row in enumerate(scores):
                for pool, (left, right) in pool_slices.items():
                    local = row[left:right]
                    pool_indices = np.arange(left, right, dtype=np.int64)
                    chosen = np.lexsort((pool_indices, -local))[:10]
                    entries = []
                    for position in chosen:
                        combined_index = left + int(position)
                        output_index = (
                            combined_index - boundary if pool == "barectb" else combined_index
                        )
                        entries.append(
                            {
                                "train_index": output_index,
                                "distance": float(f"{row[combined_index]:.4f}"),
                            }
                        )
                    results[pool].append(entries)
                # Exact scalar spot check of every selected combined top hit.
                if (start + local_index) % 257 == 0:
                    for entry in results["patb_camel_barec"][-1]:
                        exact = chrfpp_score(chunk[local_index], train[entry["train_index"]])
                        max_error = max(max_error, abs(exact - row[entry["train_index"]]))
            print(f"{split}: {min(start + len(chunk), len(queries))}/{len(queries)}", flush=True)
        scoring_timings[split] = time.perf_counter() - stage
        for pool, matches in results.items():
            all_results[(pool, split)] = matches
            write_json(
                output_dir / f"{pool}_{split}_top10_chrfpp.json",
                {
                    "metadata": {
                        "pool": pool,
                        "split": split,
                        "queries": len(matches),
                        "training_sentences": pool_slices[pool][1] - pool_slices[pool][0],
                        "top_k": 10,
                        "scorer": "exact original custom chrF++ (char 1..6, word 1..2, beta=2)",
                    },
                    "matches": matches,
                },
            )

    if max_error > 1e-12:
        raise AssertionError(f"Sparse retrieval differs from scalar scorer: {max_error}")
    write_json(
        output_dir / "retrieval_summary.json",
        {
            "started_epoch": started_epoch,
            "finished_epoch": time.time(),
            "elapsed_seconds": time.perf_counter() - started,
            "pool_sizes": {k: right - left for k, (left, right) in pool_slices.items()},
            "query_sizes": {k: len(v) for k, v in query_by_split.items()},
            "matrix_build_seconds": build_timings,
            "split_scoring_seconds": scoring_timings,
            "scalar_validation_max_absolute_error": max_error,
        },
    )
    print(f"retrieval complete in {time.perf_counter() - started:.3f}s", flush=True)


if __name__ == "__main__":
    main()
