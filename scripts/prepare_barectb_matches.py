#!/usr/bin/env python3
"""Adapt the existing top-10 BARECTB retrieval indices for the single-run CLIs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def forms(path: Path) -> list[list[str]]:
    sentences, current = [], []
    for line in path.read_text(encoding='utf-8-sig').splitlines() + ['']:
        if not line.strip():
            if current:
                sentences.append(current)
                current = []
        elif not line.startswith('#'):
            current.append(line.split('\t')[1])
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'BAREC-TB')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'retrieval')
    args = parser.parse_args()
    reference = ROOT / 'data' / 'expanded_160k'
    train = forms(args.data_dir / 'Train.conllx')
    if train != forms(reference / 'Train.conllx'):
        raise SystemExit('Training tokens/order differ from the saved matches. Regenerate matches with benchmarks/paper/generate_shots.py.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ('Dev', 'Test'):
        queries = forms(args.data_dir / f'{split}.conllx')
        if queries != forms(reference / f'{split}.conllx'):
            raise SystemExit(f'{split} tokens/order differ from the saved matches. Regenerate matches with benchmarks/paper/generate_shots.py.')
        source = ROOT / 'retrieval' / 'expanded_matches' / f'barectb_{split.lower()}_top10_chrfpp.json'
        matches = json.loads(source.read_text())['matches']
        if len(matches) != len(queries) or any(len(row) != 10 or any(not 0 <= item['train_index'] < len(train) for item in row) for row in matches):
            raise ValueError(f'Invalid match indices in {source}')
        target = args.output_dir / f'barectb_{split.lower()}.json'
        target.write_text(json.dumps({f'{split}.conllx': matches}, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'Prepared {target}: {len(matches)} queries, 10 retrieved shots each')


if __name__ == '__main__':
    main()
