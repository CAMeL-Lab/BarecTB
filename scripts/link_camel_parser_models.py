#!/usr/bin/env python3
"""Link the three existing BAREC checkpoints into an upstream CAMeL Parser checkout."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("barectb", "patb_camel", "patb_camel_barec")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("camel_parser", type=Path, help="Path to CAMeL-Lab/camel_parser checkout")
    args = parser.parse_args()
    target_dir = args.camel_parser.expanduser().resolve() / "models"
    if not (target_dir.parent / "text_to_conll_cli.py").is_file():
        parser.error("expected a camel_parser checkout with text_to_conll_cli.py")
    target_dir.mkdir(exist_ok=True)
    for name in NAMES:
        source = ROOT / "models" / f"{name}.model"
        if not source.is_file():
            parser.error(f"missing checkpoint: {source}; run python scripts/download_models.py first")
        target = target_dir / source.name
        if target.is_symlink() and target.resolve() == source:
            print(f"already linked: {target}")
            continue
        if target.exists() or target.is_symlink():
            parser.error(f"refusing to replace existing file: {target}")
        target.symlink_to(Path(os.path.relpath(source, target_dir)))
        print(f"linked: {target} -> {source}")


if __name__ == "__main__":
    main()
