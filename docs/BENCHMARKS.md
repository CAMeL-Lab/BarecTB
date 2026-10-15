# LLM benchmarks and retrieved shots

Two code sets are provided because the project ran the original paper benchmark and then a later expanded corpus benchmark. Use the matching split, training pool, prompt, and retrieved indices for each run. Do not compare the two result tables as though they used the same corpus version.

## Original paper runners

Install the lightweight client requirements in a Python environment:

```bash
python -m pip install -r requirements-benchmarks.txt
```

The original similarity script `benchmarks/paper/generate_shots.py` scores all training sentences with the project's approximate chrF++ function and sorts by descending similarity. Its JSON field called `distance` is actually a similarity score; larger is closer. The precomputed paper Dev/Test JSON files are in `retrieval/paper_matches/`. Each file's training indices refer to the matching train corpus; do not use a BARECTB index file with the PATB/CAMeL training pool. Regenerate a BARECTB Test file with:

```bash
python benchmarks/paper/generate_shots.py --train_path data/paper/Train.conllx --test_dir data/paper/Test.conllx --output_file outputs/barec_test.json --top_k 5
```

For Fanar-2 or Llama 4 Scout, run a [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/cli/) on port 8000 with the model available to vLLM. In another shell, use the same original gold-token runner for either model:

```bash
python benchmarks/paper/vllm_gold.py --config benchmarks/prompts/gold.yaml --port 8000 --model_name QCRI/Fanar-2-27B-Instruct --input_dir data/paper/Test.conllx --output_dir outputs/fanar_gold --train_conllx data/paper/Train.conllx --match_file retrieval/paper_matches/barec_test.json --num_shots 5 --conllx_output_dir outputs/fanar_gold_conllx
```

Replace `--model_name` with the exact name advertised by your Llama server for Llama 4 Scout. Use `benchmarks/paper/vllm_raw.py` and `benchmarks/prompts/raw.yaml` for raw-text inference; the remaining arguments are the same. `--sample-size 20` runs a small pilot; `--num_shots 0` gives zero-shot inference. The runner's optional `--gpu_metrics` uses `nvidia-smi`.

For Gemini, set `GEMINI_API_KEY` or `GOOGLE_API_KEY`, then run:

```bash
python benchmarks/paper/gemini_gold.py --config benchmarks/prompts/gold.yaml --model_name YOUR_GEMINI_MODEL --input_dir data/paper/Test.conllx --output_dir outputs/gemini_gold --train_conllx data/paper/Train.conllx --match_file retrieval/paper_matches/barec_test.json --num_shots 5 --conllx_output_dir outputs/gemini_gold_conllx
```

`benchmarks/paper/gemini_raw.py` uses the raw prompt. For GPT, set `OPENAI_API_KEY`. The original Batch script has `prepare`, `upload`, `create`, `status`, `download`, and `run` subcommands. `prepare` creates local requests without submitting paid API work:

```bash
python benchmarks/paper/gpt_gold.py prepare --config benchmarks/prompts/gold.yaml --cache_dir outputs/gpt_batch --input_dir data/paper/Test.conllx --shots_file retrieval/paper_matches/barec_test.json --train_file data/paper/Train.conllx --model YOUR_GPT_MODEL --num_shots 5
```

Use `run` with those arguments and `--conllx_output_dir` to submit and collect a complete Batch job. `benchmarks/paper/gpt_raw.py` pairs with the raw prompt. These scripts retain the pricing defaults present in the original source; supply current rates through their CLI options before interpreting cost output.

## Expanded 160K runners

The later code in `benchmarks/expanded/` was used for Fanar-2, Llama 4 Scout, GPT-5.4, and Gemini 3.5 Flash. It runs 0-shot and top-10 approximate chrF++ conditions over gold-token and raw-text Dev/Test, with three retrieval pools. `generate_fanar2_retrieval.py` is the faster implementation of the original scorer. It explicitly checks that the combined pool is PATB/CAMeL followed by expanded BARECTB Train. Set up the two user-supplied training pools described in [DATA.md](DATA.md) before running it.

The expanded scripts now resolve prompts and data relative to this repository. Set `BAREC_BENCH_DATA_ROOT` if the expected `data/benchmark/` layout is elsewhere, `BAREC_BENCH_RUN_ROOT` for output, `BAREC_BENCH_MODEL_PATH` for a local model snapshot or Hugging Face model ID, and `CAMEL_CONLL_ROOT` for an evaluator checkout. Generated output defaults to ignored `outputs/expanded/`.

The exact top-10 match files from the completed expanded run are in `retrieval/expanded_matches/`. Copy them into the run directory to skip retrieval recomputation. They contain indices, not the third party training text.

```bash
mkdir -p outputs/expanded
cp -R retrieval/expanded_matches outputs/expanded/retrieval
BAREC_BENCH_MODEL_PATH=QCRI/Fanar-2-27B-Instruct python benchmarks/expanded/run_fanar2_async_worker.py --mode gold --worker-id 0 --workers 1 --tensor-parallel-size 2
BAREC_BENCH_MODEL_PATH=QCRI/Fanar-2-27B-Instruct python benchmarks/expanded/run_fanar2_async_worker.py --mode raw --worker-id 0 --workers 1 --tensor-parallel-size 2
python benchmarks/expanded/finalize_fanar2_benchmark.py
```

To recompute instead, run `python benchmarks/expanded/generate_fanar2_retrieval.py`; this uses the exact training pools and can be resource intensive.

The worker imports vLLM and uses structured output. Its original run scripts used A100 GPUs and specific tensor parallel, BF16, and quantization settings. Check available GPU memory and the model's access terms before choosing settings. For Llama 4 Scout, set `BAREC_BENCH_MODEL_PATH` and `BAREC_BENCH_MODEL_ID` to that model and use the worker's `--quantization`, `--enable-expert-parallel`, and `--language-model-only` options as needed for your hardware. Use a distinct `BAREC_BENCH_RUN_ROOT` so runs do not share responses.

The expanded cloud Batch script builds the same 16 conditions and uses the same retrieval files:

```bash
python benchmarks/expanded/run_cloud_batch.py prepare --full --run-root outputs/api_batch_full
python benchmarks/expanded/run_cloud_batch.py status --run-root outputs/api_batch_full
```

For submission, put API keys in ignored `secrets/gpt.txt` and `secrets/gemini.txt`, one key per line, or set `BAREC_API_KEY_DIR` to a private directory with those filenames. Then run `stage`, `canary`, or `run` for `--provider openai` and `--provider gemini`, followed by `finalize`. The script's `--help` lists the precise subcommands. Its model identifiers and price constants record the completed 2026 runs; check current API availability and pricing before a new run. The original completed result summaries are in `reports/`.

The expanded code's raw-text postprocessing helper was extracted from the existing `finalize_full_coverage.py` in the adjacent `arabic_syntax` project; its parsing repair function is copied as `benchmarks/expanded/finalize_full_coverage.py`. This preserves the function used by the original expanded scripts without requiring that sibling checkout.
