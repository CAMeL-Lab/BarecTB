#!/usr/bin/env python3
"""Publish the three original BARECTB checkpoints and group them in CAMeL-Lab."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'huggingface' / 'manifest.json'
RECEIPT = ROOT / 'huggingface' / 'published.json'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def upload_files(spec: dict) -> dict[str, Path]:
    card_dir = ROOT / spec['card_dir']
    checkpoint = ROOT / spec['checkpoint']
    config = ROOT / spec['config']
    return {
        'README.md': card_dir / 'README.md',
        checkpoint.name: checkpoint,
        config.name: config,
        'SHA256SUMS': card_dir / 'SHA256SUMS',
        'CITATION.bib': ROOT / 'docs' / 'CITATION.bib',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Check local files and hashes without network access or publication')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if len(manifest['collection_description']) > 150:
        raise SystemExit('Hugging Face collection descriptions must be at most 150 characters.')
    for spec in manifest['models']:
        for path in upload_files(spec).values():
            if not path.is_file():
                raise SystemExit(f'Missing upload file: {path}')
        if sha256(ROOT / spec['checkpoint']) != spec['sha256']:
            raise SystemExit(f"Checkpoint checksum mismatch: {spec['checkpoint']}")
        print(f"Verified {spec['repo_id']}", flush=True)
    if args.dry_run:
        print('All three upload packages verified; no remote changes made.')
        return

    from huggingface_hub import CommitOperationAdd, HfApi, ModelCard
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi()
    account = api.whoami()
    role = account.get('auth', {}).get('accessToken', {}).get('role')
    if role == 'read':
        raise SystemExit('Saved Hugging Face token is read only. Run hf auth login with a token that can write model repositories and collections in CAMeL-Lab.')
    namespace = manifest['namespace']
    if namespace != account.get('name') and not any(org.get('name') == namespace for org in account.get('orgs', [])):
        raise SystemExit(f'Authenticated account is not a member of {namespace}.')

    # Validate every card before creating repositories.
    for spec in manifest['models']:
        ModelCard.load(ROOT / spec['card_dir'] / 'README.md').validate(repo_type='model')

    published = {'namespace': namespace, 'models': []}
    if RECEIPT.exists():
        published.update(json.loads(RECEIPT.read_text()))

    for spec in manifest['models']:
        repo_id = spec['repo_id']
        filename = Path(spec['checkpoint']).name
        try:
            existing = api.model_info(repo_id, files_metadata=True)
        except RepositoryNotFoundError:
            existing = None
        if existing is not None:
            remote_checkpoint = next((item for item in existing.siblings if item.rfilename == filename), None)
            if remote_checkpoint is None and any(item.rfilename != '.gitattributes' for item in existing.siblings):
                raise SystemExit(f'{repo_id} already exists without the expected checkpoint; refusing to overwrite it.')
            if remote_checkpoint is not None and (remote_checkpoint.lfs is None or remote_checkpoint.lfs.sha256 != spec['sha256']):
                raise SystemExit(f'{repo_id} already contains a different checkpoint; refusing to overwrite it.')
        api.create_repo(repo_id, repo_type='model', private=False, exist_ok=True)
        files = upload_files(spec)
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type='model',
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=path) for name, path in files.items()],
            commit_message='Upload original BARECTB parser checkpoint, configuration, model card, and citation',
        )
        remote = api.model_info(repo_id, files_metadata=True, revision=commit.oid)
        blob = next(item for item in remote.siblings if item.rfilename == filename)
        if blob.lfs is None or blob.lfs.sha256 != spec['sha256'] or blob.size != files[filename].stat().st_size:
            raise RuntimeError(f'{repo_id}: uploaded checkpoint hash or size does not match the source')
        record = {'repo_id': repo_id, 'url': f'https://huggingface.co/{repo_id}', 'commit': commit.oid, 'sha256': spec['sha256']}
        published['models'] = [item for item in published['models'] if item['repo_id'] != repo_id] + [record]
        RECEIPT.write_text(json.dumps(published, indent=2) + '\n')
        print(f"Uploaded and verified {record['url']}", flush=True)

    collection = api.create_collection(
        title=manifest['collection_title'],
        namespace=namespace,
        description=manifest['collection_description'],
        private=False,
        exists_ok=True,
    )
    for spec in manifest['models']:
        api.add_collection_item(collection.slug, item_id=spec['repo_id'], item_type='model', note=f"Training data: {spec['training_data']}.", exists_ok=True)
    actual = api.get_collection(collection.slug)
    model_ids = {item.item_id for item in actual.items if item.item_type == 'model'}
    if not {spec['repo_id'] for spec in manifest['models']}.issubset(model_ids):
        raise RuntimeError('Collection verification failed: one or more models are missing')
    published['collection_url'] = actual.url
    published['collection_slug'] = actual.slug
    RECEIPT.write_text(json.dumps(published, indent=2) + '\n')
    print(f'Collection ready: {actual.url}')


if __name__ == '__main__':
    main()
