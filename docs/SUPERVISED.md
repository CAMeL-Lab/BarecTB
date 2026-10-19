# Supervised CAMeL Parser models

The three original checkpoints in `models/` are:

| Model name for `-m` | Training condition |
| --- | --- |
| `barectb.model` | BARECTB training data |
| `patb_camel.model` | PATB and CAMeL Treebank training data |
| `patb_camel_barec.model` | PATB, CAMeL Treebank, and BARECTB training data |

Their original YAML configurations and SHA-256 checksums are alongside the checkpoints. Verify after transfer with `cd models && shasum -a 256 -c SHA256SUMS` on macOS, or `sha256sum -c SHA256SUMS` on Linux.

The three checkpoints are available in the [CamelParser-BARECTB collection on Hugging Face](https://huggingface.co/collections/CAMeL-Lab/camelparser-barectb-6ad5d147a150e48c97598630). Each model card includes a download command and CAMeL Parser inference instructions. [Publication details](../huggingface/README.md) include verified remote commit IDs and checksums.

## Install upstream parser

Use Python 3.11.13 as specified by [CAMeL Parser](https://github.com/CAMeL-Lab/camel_parser/tree/main). From the repository root:

```bash
git clone https://github.com/CAMeL-Lab/camel_parser.git external/camel_parser
python3.11 -m venv .venv-parser
source .venv-parser/bin/activate
python -m pip install -r external/camel_parser/requirements.txt
python external/camel_parser/download_models.py
python scripts/link_camel_parser_models.py external/camel_parser
```

`link_camel_parser_models.py` creates links in the upstream parser's `models/` directory and refuses to replace existing files. The three checkpoints must be present as real files (Git LFS fetched) before linking.

## Inference

For the upstream raw-text pipeline:

```bash
cd external/camel_parser
python text_to_conll_cli.py -f text -i ../../data/paper/text/Test.txt -m barectb.model
```

The CLI also accepts `-f conll -i /path/to/input.conllx` for preannotated input and `--multiroot` for the later multiple-root experiments. It prints the parse and timing to standard output; capture and clean that output as appropriate for your evaluation workflow. The `data/paper/text/` files are direct extractions of the CoNLL-X `# text =` metadata.

For the gold/raw prediction matrix used in the expanded work, the original driver is `evaluation/predict_all_models_cached.py`. It accepts `--camel-parser-root`, `--gold-dir`, `--model-table`, `--output-root`, and `--cache`. The included `evaluation/model_table.tsv` lists the three original checkpoints using repository-relative paths, so run it from the repository root. This path requires the upstream parser and its Python dependencies. The raw MSA-S31 driver is `evaluation/predict_all_models_raw_s31.py` and additionally needs the licensed morphology database described in the upstream README.

The original training commands remain in the local source folder `retrain-models/` for provenance; their scripts assume HPC paths and third party corpora, so the released checkpoints are the simplest way to run inference.
