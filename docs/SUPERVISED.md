# Run the supervised parsers

Use the installation and data download commands in the [main README](../README.md), including `python -m pip install -r requirements.txt`. All commands below run from this repository's root unless a `cd` is shown.

## Download and link the models

```bash
python scripts/download_models.py
python scripts/link_camel_parser_models.py external/camel_parser
camel_data -i morphology-db-msa-r13
camel_data -i disambig-bert-unfactored-msa
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

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
python text_to_conll_cli.py -f text -m barectb.model -s "القطة نائمة."
python text_to_conll_cli.py -f text -m barectb.model -i ../../data/BAREC-TB/text/Test.txt
cd ../..
```

Change `-m` to select another checkpoint. The CLI prints dependency parses and timing to stdout. For prediction files that can go directly into the evaluator, use the benchmark driver below.

## Benchmark all three models on Dev and Test

The existing driver uses the three entries in `evaluation/model_table.tsv`. It predicts both gold-token and raw-text conditions, and caches the expensive raw-text preprocessing:

```bash
python evaluation/predict_all_models_cached.py \
  --camel-parser-root external/camel_parser \
  --gold-dir data/BAREC-TB \
  --model-table evaluation/model_table.tsv \
  --output-root outputs/supervised \
  --cache outputs/supervised/cache/raw_r13.pkl
```

Predictions are written to:

```text
outputs/supervised/original/gold/<model>/Dev.conllx
outputs/supervised/original/gold/<model>/Test.conllx
outputs/supervised/original/raw/<model>/Dev.conllx
outputs/supervised/original/raw/<model>/Test.conllx
```

Here `<model>` is `barectb`, `patb_camel`, or `patb_camel_barec`. `original` identifies the checkpoint generation in the model table. This command evaluates the released checkpoints on the official dataset; archived results use the data versions described in [DATA.md](DATA.md).

Score a prediction:

```bash
python evaluation/evaluate.py \
  --gold data/BAREC-TB/Test.conllx \
  --parsed outputs/supervised/original/gold/barectb/Test.conllx \
  --normalize --output-dir outputs/supervised/scores --name barectb_gold_test
```

Change the model, `gold`/`raw`, and split paths to score the other conditions. Use a new `--name` for each score file. Existing prediction files are reused by the driver; choose a new output/cache directory when changing the input data.

## Multiple roots and MSA-S31

For the multiple-root driver, replace `evaluation/predict_all_models_cached.py` with `evaluation/predict_all_models_multiroot_cached.py` and use a distinct output/cache directory. This requires an upstream CAMeL Parser checkout containing `_decode_multiroot`.

The original raw MSA-S31 driver additionally needs the separately installed morphology database:

```bash
python evaluation/predict_all_models_raw_s31.py \
  --camel-parser-root external/camel_parser \
  --gold-dir data/BAREC-TB \
  --model-table evaluation/model_table.tsv \
  --morphology-db-path /path/to/calima-msa-s31/morphology.db \
  --output-root outputs/supervised_s31 \
  --cache outputs/supervised_s31/cache/raw_s31.pkl
```

Follow [CAMeL Parser's MSA-S31 installation instructions](https://github.com/CAMeL-Lab/camel_parser#using-another-morphology-database) for the CAMeL Tools data packages and LDC resource. These non-Python resources are separate from `requirements.txt`. The original YAML configurations retain historical training paths as provenance; those paths are not needed for inference.
