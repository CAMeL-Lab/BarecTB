#!/usr/bin/env python3
"""Run the CAMeL CoNLL evaluator on a gold/prediction pair."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', required=True, type=Path)
    parser.add_argument('--parsed', required=True, type=Path)
    parser.add_argument('--camel-conll-root', type=Path, default=ROOT / 'external' / 'camel_conll')
    parser.add_argument('--normalize', action='store_true', help='Use the paper benchmark flags -x -n -a')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--name', default='evaluation')
    args = parser.parse_args()
    upstream = args.camel_conll_root.expanduser().resolve()
    script = upstream / 'conll_evaluation.py'
    if not script.is_file():
        parser.error(f'missing {script}; clone https://github.com/CAMeL-Lab/camel_conll there')
    for path in (args.gold, args.parsed):
        if not path.exists():
            parser.error(f'missing input: {path}')
    command = [sys.executable, str(script), '--gold', str(args.gold.resolve()), '--parsed', str(args.parsed.resolve())]
    if args.normalize:
        command.extend(('-x', '-n', '-a'))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        command.extend(('--output_path', str(args.output_dir.resolve()), args.name))
    subprocess.run(command, cwd=upstream, check=True)


if __name__ == '__main__':
    main()
