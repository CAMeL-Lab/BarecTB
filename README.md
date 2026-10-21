# BARECTB

This repository contains the code for **BARECTB: A Dependency Treebank and Parsing Benchmark for Arabic Children’s Texts**, ArabicNLP 2026.

The code supports supervised dependency parsing with CAMeL Parser, retrieval based few-shot benchmarking of Fanar and Llama with vLLM, GPT Batch and Gemini API benchmarking, and evaluation with CAMeL CoNLL.

## Installation

Use **Python 3.11**. Run the commands below from the repository root. Full Fanar/Llama inference requires a Linux x86_64 machine with NVIDIA GPUs; the comparable BF16 runs used four A100 GPUs. The requirements file includes vLLM, PyTorch, CAMeL Tools, SuPar, the evaluator dependencies, and both API clients.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

git clone https://github.com/CAMeL-Lab/camel_parser.git external/camel_parser
git clone https://github.com/CAMeL-Lab/camel_conll.git external/camel_conll
```

Use this repository's single `requirements.txt`; the older dependency freezes in the upstream checkouts conflict with vLLM. On macOS, the vLLM dependency is skipped; use a Linux GPU host for local LLM inference. See [environment and reproduction notes](docs/REPRODUCIBILITY.md) for the original runtime version and the consolidated installation baseline.

## Download the data

The dataset is hosted at [CAMeL-Lab/BAREC-TB](https://huggingface.co/datasets/CAMeL-Lab/BAREC-TB). **It is currently private and will be made public.** Until then, sign in with an account granted access:

```bash
hf auth login
python scripts/download_data.py
```

The downloader saves the verified `Train.conllx`, `Dev.conllx`, and `Test.conllx` files under `data/BAREC-TB/`, and raw sentences under `data/BAREC-TB/text/`. Once the dataset is public, login is optional. The published splits contain annotation updates to Dev/Test; [data version notes](docs/DATA.md) describe their relationship to the saved benchmark snapshots.

## Run the supervised models

Download the three published checkpoints from the [supervised models collection on Hugging Face](https://huggingface.co/collections/CAMeL-Lab/camelparser-barectb-6ad5d147a150e48c97598630) and link them into CAMeL Parser:

```bash
python scripts/download_models.py
python scripts/link_camel_parser_models.py external/camel_parser
camel_data -i morphology-db-msa-s31
camel_data -i disambig-bert-unfactored-msa
# Download LDC2010L01.tgz from your LDC account; leave the archive unextracted.
camel_data -p morphology-db-msa-s31 /path/to/LDC2010L01.tgz
```

This uses the **MSA-S31 morphology database and BERT disambiguator**, following [CAMeL Parser’s paper setup](https://github.com/CAMeL-Lab/camel_parser#using-another-morphology-database). Installing MSA-S31 requires access to [LDC2010L01](https://catalog.ldc.upenn.edu/LDC2010L01); its archive is a separate resource from the Python requirements.

| Model name | Training data |
| --- | --- |
| `barectb.model` | BARECTB |
| `patb_camel.model` | PATB + CAMeL Treebank |
| `patb_camel_barec.model` | PATB + CAMeL Treebank + BARECTB |

Run a model on an Arabic sentence:

```bash
# The published SuPar checkpoints use full-object serialization.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd external/camel_parser
python text_to_conll_cli.py -f text -b calima-msa-s31 -d bert -m barectb.model -s "القطة نائمة."
cd ../..
```

Replace `-m barectb.model` with either of the other model names. To parse a text file, replace `-s` with `-i ../../data/BAREC-TB/text/Test.txt` while inside `external/camel_parser`. The upstream CLI prints the parse and timing to stdout. For clean prediction files and evaluation of all three models on Dev/Test, follow the complete commands in [supervised benchmarking](docs/SUPERVISED.md).

## Prepare the retrieved shots

Prepare the saved top-10 BARECTB training matches for the downloaded splits:

```bash
python scripts/prepare_barectb_matches.py
```

This verifies the training and query token order before adapting the existing chrF++ match files to the runners' input format. It writes `outputs/retrieval/barectb_dev.json` and `outputs/retrieval/barectb_test.json`. The original retrieval algorithm is also included. To recompute matches for another split or training pool:

```bash
python benchmarks/paper/generate_shots.py \
  --train_path data/BAREC-TB/Train.conllx \
  --test_dir data/BAREC-TB/Test.conllx \
  --output_file outputs/retrieval/barectb_test.json \
  --top_k 10
```

The scorer sorts by highest similarity; the historical JSON field named `distance` stores similarity. Recomputing compares every query against the training pool and is slower than reusing the verified saved matches.

## Benchmark Fanar and Llama with vLLM

Start **one** model server. For Fanar-2:

```bash
vllm serve QCRI/Fanar-2-27B-Instruct \
  --port 8000 --tensor-parallel-size 4 --dtype bfloat16 \
  --max-model-len 131072
```

For Llama 4 Scout, accept the model's Hugging Face access conditions and sign in with `hf auth login`, then start this server instead:

```bash
vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --port 8000 --tensor-parallel-size 4 --dtype bfloat16 \
  --max-model-len 163840 --enable-expert-parallel \
  --limit-mm-per-prompt '{"image": 0, "video": 0}'
```

In a **second terminal**, activate `.venv`, return to the repository root, and select the running model:

```bash
export MODEL_ID=QCRI/Fanar-2-27B-Instruct
export RUN_NAME=fanar
# For Llama, use MODEL_ID=meta-llama/Llama-4-Scout-17B-16E-Instruct and RUN_NAME=llama.
```

Run the gold-token benchmark with 10 retrieved shots:

```bash
python benchmarks/paper/vllm_gold.py \
  --config benchmarks/prompts/gold.yaml --port 8000 --model_name "$MODEL_ID" \
  --input_dir data/BAREC-TB/Test.conllx \
  --train_conllx data/BAREC-TB/Train.conllx \
  --match_file outputs/retrieval/barectb_test.json --num_shots 10 \
  --output_dir "outputs/${RUN_NAME}_gold" \
  --conllx_output_dir "outputs/${RUN_NAME}_gold_conllx"
```

Run the raw-text benchmark:

```bash
python benchmarks/paper/vllm_raw.py \
  --config benchmarks/prompts/raw.yaml --port 8000 --model_name "$MODEL_ID" \
  --input_dir data/BAREC-TB/Test.conllx \
  --train_conllx data/BAREC-TB/Train.conllx \
  --match_file outputs/retrieval/barectb_test.json --num_shots 10 \
  --output_dir "outputs/${RUN_NAME}_raw" \
  --conllx_output_dir "outputs/${RUN_NAME}_raw_conllx"
```

Use `--num_shots 0` for zero-shot inference, or add `--sample-size 20` for a pilot. For Dev, change `Test.conllx` to `Dev.conllx`, use `barectb_dev.json`, and choose a new output directory. The runners save predictions, CoNLL-X output, and latency/token metrics. The complete asynchronous experiment matrix, including the PATB/CAMeL training pools, is documented in [benchmark reproduction](docs/BENCHMARKS.md).

## Benchmark Gemini

Set your key in the terminal, then run:

```bash
export GEMINI_API_KEY="your-key"
python benchmarks/paper/gemini_gold.py \
  --config benchmarks/prompts/gold.yaml --model_name gemini-3.5-flash \
  --input_dir data/BAREC-TB/Test.conllx \
  --train_conllx data/BAREC-TB/Train.conllx \
  --match_file outputs/retrieval/barectb_test.json --num_shots 10 \
  --output_dir outputs/gemini_gold \
  --conllx_output_dir outputs/gemini_gold_conllx
```

For raw text, use `benchmarks/paper/gemini_raw.py`, `benchmarks/prompts/raw.yaml`, and separate output directories. `--sample-size 20` and `--num_shots 0` work here too.

## Benchmark GPT

The GPT runner submits an OpenAI Batch job, polls for completion, and exports predictions and metrics:

```bash
export OPENAI_API_KEY="your-key"
python benchmarks/paper/gpt_gold.py run \
  --config benchmarks/prompts/gold.yaml --model gpt-5.4 \
  --input_dir data/BAREC-TB/Test.conllx \
  --train_file data/BAREC-TB/Train.conllx \
  --shots_file outputs/retrieval/barectb_test.json --num_shots 10 \
  --cache_dir outputs/gpt_gold_batch \
  --conllx_output_dir outputs/gpt_gold_conllx
```

For raw text, use `benchmarks/paper/gpt_raw.py` and `benchmarks/prompts/raw.yaml`, with separate cache/output directories. Replace `run` with `prepare` to generate requests without submitting them. API model identifiers and pricing defaults reflect the original scripts; the CLI accepts model and price overrides.

## Evaluate predictions

Evaluate a prediction file against the matching gold split:

```bash
python evaluation/evaluate.py \
  --gold data/BAREC-TB/Test.conllx \
  --parsed outputs/fanar_gold_conllx/Test.conllx \
  --normalize --output-dir outputs/evaluation --name fanar_gold_test
```

Replace `--parsed` with the output from any model. `--normalize` enables the CAMeL CoNLL `-x -n -a` normalization settings used in the saved experiments. [Evaluation details](docs/EVALUATION.md) explain the alignment assumptions and output metrics.

## Citation

If you use this code, the treebank, or the models in your research, please cite:

```bibtex
@inproceedings{elmadani-etal-2026-barectb,
  title = {{BARECTB}: A Dependency Treebank and Parsing Benchmark for Arabic Children's Texts},
  author = {Elmadani, Khalid N. and Adel, Mohamed and AbuOdeh, Muhammed and Faraj, Reem and El-Gizuli, Jamila and Taha, Hanada and Habash, Nizar},
  booktitle = {Proceedings of ArabicNLP 2026},
  address = {Budapest, Hungary},
  year = {2026}
}
```
