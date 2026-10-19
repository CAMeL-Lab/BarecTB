#!/usr/bin/env python3
"""Download the three original BARECTB checkpoints from their verified HF revisions."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(spec: dict, publication: dict, output_dir: Path) -> Path:
    target = output_dir / Path(spec['checkpoint']).name
    if target.exists():
        if sha256(target) != spec['sha256']:
            raise ValueError(f'{target}: existing file has a different checksum. Move it aside before downloading the published checkpoint.')
        print(f'Already verified: {target}')
        return target

    from huggingface_hub import hf_hub_download

    source = Path(hf_hub_download(
        repo_id=spec['repo_id'],
        filename=target.name,
        revision=publication['commit'],
    ))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=f'.{target.name}.', delete=False) as stream:
            temporary = Path(stream.name)
            digest = hashlib.sha256()
            with source.open('rb') as downloaded:
                for chunk in iter(lambda: downloaded.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
                    stream.write(chunk)
        if digest.hexdigest() != spec['sha256']:
            raise ValueError(f"{spec['repo_id']}: downloaded checkpoint checksum mismatch")
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f'Downloaded and verified: {target}')
    return target


def main() -> None:
    manifest = json.loads((ROOT / 'huggingface' / 'manifest.json').read_text())
    publications = json.loads((ROOT / 'huggingface' / 'published.json').read_text())
    by_repo = {item['repo_id']: item for item in publications['models']}
    choices = [Path(item['checkpoint']).stem for item in manifest['models']]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=choices, action='append', help='Download only this model; may be repeated. Default: all three.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'models')
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in manifest['models']:
        if args.model and Path(spec['checkpoint']).stem not in args.model:
            continue
        publication = by_repo[spec['repo_id']]
        if publication['sha256'] != spec['sha256']:
            raise ValueError(f"Publication and local manifest disagree for {spec['repo_id']}")
        download_checkpoint(spec, publication, output_dir)


if __name__ == '__main__':
    main()
