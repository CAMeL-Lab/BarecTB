#!/usr/bin/env python3
"""Download the official BAREC-TB CoNLL-X splits from Hugging Face."""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def save_verified(target: Path, content: bytes, expected: str) -> None:
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f'Checksum mismatch for {target.name}')
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f'{target} differs from the published split. Move it aside before downloading.')
        return
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data' / 'BAREC-TB')
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'huggingface' / 'dataset.json').read_text())
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
    try:
        archive = Path(hf_hub_download(repo_id=manifest['repo_id'], repo_type='dataset',
                                     filename=manifest['archive'], revision=manifest['revision']))
    except HfHubHTTPError as error:
        if error.response is not None and error.response.status_code in (401, 403, 404):
            raise SystemExit('BAREC-TB is currently private. Run hf auth login with an account granted access to CAMeL-Lab/BAREC-TB, then rerun. Public downloads will work once the dataset is released.') from error
        raise
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['archive_sha256']:
        raise ValueError('Downloaded split archive checksum mismatch')
    with ZipFile(archive) as bundle:
        for name, expected in manifest['splits'].items():
            content = bundle.read(f'splits_conllx/{name}')
            save_verified(output / name, content, expected)
            text_dir = output / 'text'
            text_dir.mkdir(exist_ok=True)
            lines = [line[len('# text = '):] for line in content.decode('utf-8').splitlines() if line.startswith('# text = ')]
            (text_dir / Path(name).with_suffix('.txt')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
            print(f'Verified {output / name}: {len(lines)} sentences')


if __name__ == '__main__':
    main()
