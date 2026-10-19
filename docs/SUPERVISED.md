# Supervised CAMeL Parser models

Download the three original checkpoints from the [CamelParser-BARECTB collection](https://huggingface.co/collections/CAMeL-Lab/camelparser-barectb-6ad5d147a150e48c97598630) into `models/`:

| Model name for `-m` | Training condition |
| --- | --- |
| [`barectb.model`](https://huggingface.co/CAMeL-Lab/camelparser-barectb-BARECTB) | BARECTB training data |
| [`patb_camel.model`](https://huggingface.co/CAMeL-Lab/camelparser-barectb-PATB-CamelTB) | PATB and CAMeL Treebank training data |
| [`patb_camel_barec.model`](https://huggingface.co/CAMeL-Lab/camelparser-barectb-PATB-CamelTB-BARECTB) | PATB, CAMeL Treebank, and BARECTB training data |

Git tracks their original YAML configurations and SHA-256 checksums. Checkpoint binaries are fetched from Hugging Face and ignored by Git. Verify after transfer with `cd models && shasum -a 256 -c SHA256SUMS` on macOS, or `sha256sum -c SHA256SUMS` on Linux.

The download script uses the exact uploaded revisions recorded in `huggingface/published.json` and verifies each file’s SHA-256 before saving it. It reuses existing matching checkpoints. To download one model, use `python scripts/download_models.py --model barectb` (or `patb_camel` or `patb_camel_barec`). The model repositories are public; downloads do not require a write token. [Publication details](../huggingface/README.md) include verified remote commit IDs and checksums.

## Install upstream parser

Use Python 3.11.13 as specified by [CAMeL Parser](https://github.com/CAMeL-Lab/camel_parser/tree/main). From the repository root:

```bash
git clone https://github.com/CAMeL-Lab/camel_parser.git external/camel_parser
python3.11 -m venv .venv-parser
source .venv-parser/bin/activate
python -m pip install -r external/camel_parser/requirements.txt
python -m pip install -r requirements-models.txt
python scripts/download_models.py
python scripts/link_camel_parser_models.py external/camel_parser
```

`link_camel_parser_models.py` creates links in the upstream parser's `models/` directory and refuses to replace existing files. Run the downloader before linking the checkpoints. Git LFS is not needed for this repository.

## Inference

For the upstream raw-text pipeline:

```bash
cd external/camel_parser
python text_to_conll_cli.py -f text -i ../../data/paper/text/Test.txt -m barectb.model
```

The CLI also accepts `-f conll -i /path/to/input.conllx` for preannotated input and `--multiroot` for the later multiple-root experiments. It prints the parse and timing to standard output; capture and clean that output as appropriate for your evaluation workflow. The `data/paper/text/` files are direct extractions of the CoNLL-X `# text =` metadata.

For the gold/raw prediction matrix used in the expanded work, the original driver is `evaluation/predict_all_models_cached.py`. It accepts `--camel-parser-root`, `--gold-dir`, `--model-table`, `--output-root`, and `--cache`. The included `evaluation/model_table.tsv` lists the three original checkpoints using repository-relative paths, so run it from the repository root. This path requires the upstream parser and its Python dependencies. The raw MSA-S31 driver is `evaluation/predict_all_models_raw_s31.py` and additionally needs the licensed morphology database described in the upstream README.

The original YAML configurations preserve the training settings and historical HPC paths as provenance. Use the published Hugging Face checkpoints for inference; those historical paths are not required.
