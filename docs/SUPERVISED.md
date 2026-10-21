# Run the supervised parsers

Use the installation and data download commands in the [main README](../README.md), including `python -m pip install -r requirements.txt`. All commands below run from this repository's root unless a `cd` is shown.

## Download and link the models

```bash
python scripts/download_models.py
python scripts/link_camel_parser_models.py external/camel_parser
camel_data -i morphology-db-msa-s31
camel_data -i disambig-bert-unfactored-msa
# Obtain LDC2010L01.tgz from your LDC account and keep it unextracted.
camel_data -p morphology-db-msa-s31 /path/to/LDC2010L01.tgz
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

This follows [CAMeL Parser’s MSA-S31 setup](https://github.com/CAMeL-Lab/camel_parser#using-another-morphology-database): install the CAMeL Tools packages, obtain `LDC2010L01.tgz` from the [LDC downloads page](https://catalog.ldc.upenn.edu/organization/downloads), and pass the unextracted archive to `camel_data -p`. The morphology resource is not installed by pip.

The environment variable enables loading the full-object SuPar checkpoints with the consolidated PyTorch version. Use it with the published checkpoints verified by the downloader.

| Model | Training data |
| --- | --- |
| [`barectb.model`](https://huggingface.co/CAMeL-Lab/camelparser-barectb-BARECTB) | BARECTB |
| [`patb_camel.model`](https://huggingface.co/CAMeL-Lab/camelparser-barectb-PATB-CamelTB) | PATB + CAMeL Treebank |
| [`patb_camel_barec.model`](https://huggingface.co/CAMeL-Lab/camelparser-barectb-PATB-CamelTB-BARECTB) | PATB + CAMeL Treebank + BARECTB |

The downloader pins the uploaded revisions in `huggingface/published.json` and verifies SHA-256. To download one model, add `--model barectb`, `--model patb_camel`, or `--model patb_camel_barec`. Git tracks the original configurations and checksums; the weights are hosted on Hugging Face.

## Parse a sentence or file

```bash
cd external/camel_parser
python text_to_conll_cli.py -f text -b calima-msa-s31 -d bert -m barectb.model -s "القطة نائمة."
python text_to_conll_cli.py -f text -b calima-msa-s31 -d bert -m barectb.model -i ../../data/BAREC-TB/text/Test.txt
cd ../..
```

Change `-m` to select another checkpoint. The CLI prints dependency parses and timing to stdout. For prediction files that can go directly into the evaluator, use the benchmark driver below.

## Benchmark all three models on Dev and Test

The driver uses the three entries in `evaluation/model_table.tsv`. Raw-text preprocessing calls CAMeL Parser with **BERT disambiguation and `calima-msa-s31`**. The command below uses multiple-root decoding, matching the current CAMeL Parser default, and predicts both gold-token and raw-text conditions:

```bash
python evaluation/predict_all_models_multiroot_cached.py \
  --camel-parser-root external/camel_parser \
  --gold-dir data/BAREC-TB \
  --model-table evaluation/model_table.tsv \
  --output-root outputs/supervised_s31_multiroot \
  --cache outputs/supervised_s31_multiroot/cache/raw_s31.pkl
```

Predictions are written to:

```text
outputs/supervised_s31_multiroot/original/gold/<model>/Dev.conllx
outputs/supervised_s31_multiroot/original/gold/<model>/Test.conllx
outputs/supervised_s31_multiroot/original/raw/<model>/Dev.conllx
outputs/supervised_s31_multiroot/original/raw/<model>/Test.conllx
```

Here `<model>` is `barectb`, `patb_camel`, or `patb_camel_barec`. `original` identifies the checkpoint generation in the model table. This command evaluates the released checkpoints on the official dataset; archived results use the data versions described in [DATA.md](DATA.md).

Score a prediction:

```bash
python evaluation/evaluate.py \
  --gold data/BAREC-TB/Test.conllx \
  --parsed outputs/supervised_s31_multiroot/original/gold/barectb/Test.conllx \
  --normalize --output-dir outputs/supervised_s31_multiroot/scores --name barectb_gold_test
```

Change the model, `gold`/`raw`, and split paths to score the other conditions. Use a new `--name` for each score file. Existing prediction files are reused by the driver; choose a new output/cache directory when changing the input data.

## Single-root runs and explicit database files

For single-root decoding, replace `evaluation/predict_all_models_multiroot_cached.py` with `evaluation/predict_all_models_cached.py` and choose a distinct output directory, such as `outputs/supervised_s31_single`. Both drivers use MSA-S31/BERT preprocessing. Their caches record the morphology database and reject legacy caches that do not identify S31; use a fresh cache and output directory for this setup.

The original raw-only driver can also read an explicit MSA-S31 database file and run both single-root and multiple-root decoding:

```bash
python evaluation/predict_all_models_raw_s31.py \
  --camel-parser-root external/camel_parser \
  --gold-dir data/BAREC-TB \
  --model-table evaluation/model_table.tsv \
  --morphology-db-path /path/to/calima-msa-s31/morphology.db \
  --output-root outputs/raw_s31_both_decoders \
  --cache outputs/raw_s31_both_decoders/cache/raw_s31.pkl
```

This driver accepts the released three-model table. Its outputs are organized as `<output-root>/<single|multiroot>/<generation>/<model>/<Dev|Test>.conllx`. The cache also records the database file path and SHA-256. Gold-token parsing uses the supplied tokens and annotations and does not need raw-text morphological analysis. The original YAML configurations retain historical training paths as provenance; those paths are not needed for inference.
