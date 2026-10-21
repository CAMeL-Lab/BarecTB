# Reproduce the benchmark conditions

For individual runs on the official Hugging Face dataset, use the complete Fanar, Llama, Gemini, and GPT commands in the [main README](../README.md). This guide covers the complete asynchronous matrix retained from the expanded experiments and its original snapshot inputs.

Install the single `requirements.txt` first. All commands below run from the repository root in the activated Python 3.11 environment. Full BF16 Fanar/Llama runs used four A100 GPUs. [Environment notes](REPRODUCIBILITY.md) distinguish the installable dependencies from the original vLLM development build.

## Prepare the three training pools

The matrix uses `data/benchmark/gold/{Dev,Test}.conllx` and `data/benchmark/train/barectb.conllx`, which link to the saved expanded snapshot. Obtain the original PATB + CAMeL Treebank training pool separately, then supply it in the original sentence order:

```bash
export PATB_CAMEL_TRAIN=/absolute/path/to/patb_camel.conllx
ln -s "$PATB_CAMEL_TRAIN" data/benchmark/train/patb_camel.conllx
python - <<'PY'
from pathlib import Path
pool = Path('data/benchmark/train')
patb = (pool / 'patb_camel.conllx').read_text(encoding='utf-8').rstrip()
barec = (pool / 'barectb.conllx').read_text(encoding='utf-8').rstrip()
(pool / 'patb_camel_barec.conllx').write_text(patb + '\n\n' + barec + '\n\n', encoding='utf-8')
PY
```

These files are ignored by Git. Their order must match the saved retrieval indices. The combined pool is PATB/CAMeL followed by expanded BARECTB Train. To score official annotation updates instead, create the same `gold/` and `train/` layout in a separate directory and set `BAREC_BENCH_DATA_ROOT` to its absolute path; new scores then use that dataset version.

## Retrieved shots

The complete matrix includes gold-token and raw-text Dev/Test, each with zero shots or ten shots from one of the three training pools: 16 conditions per model. The top-10 matches are selected by the original custom approximate chrF++ scorer. Higher `distance` values in the JSON mean higher similarity.

The saved files in `retrieval/expanded_matches/` correspond to the original three pools and expanded queries. The worker reads them from `<run-root>/retrieval/`. Copy them as shown below. To recompute all matches instead:

```bash
export BAREC_BENCH_RUN_ROOT="$PWD/outputs/recomputed_retrieval"
python benchmarks/expanded/generate_fanar2_retrieval.py --chunk-size 32
```

That optimized implementation checks the combined training pool order and writes all six pool/split match files. Retrieval can be resource intensive. For a single custom pool, `benchmarks/paper/generate_shots.py` supports `--train_path`, `--test_dir`, `--output_file`, and `--top_k` as shown in the README.

## Fanar-2 and Llama 4 Scout asynchronous runs

The same worker runs either model directly through vLLM, without a separate server. Select one setup:

```bash
# Fanar-2 setup.
export BAREC_BENCH_MODEL_ID=QCRI/Fanar-2-27B-Instruct
export BAREC_BENCH_MODEL_PATH="$BAREC_BENCH_MODEL_ID"
export BAREC_BENCH_RUN_ROOT="$PWD/outputs/fanar_matrix"
export BAREC_BENCH_TABLE_NAME=fanar_results.tsv
MODEL_CONTEXT=131072
EXTRA_ARGS=()
```

Or use the Llama setup after obtaining Hugging Face model access:

```bash
export BAREC_BENCH_MODEL_ID=meta-llama/Llama-4-Scout-17B-16E-Instruct
export BAREC_BENCH_MODEL_PATH="$BAREC_BENCH_MODEL_ID"
export BAREC_BENCH_RUN_ROOT="$PWD/outputs/llama_matrix"
export BAREC_BENCH_TABLE_NAME=llama_results.tsv
MODEL_CONTEXT=163840
EXTRA_ARGS=(--enable-expert-parallel --language-model-only)
```

Run both input settings, then merge and evaluate:

```bash
export BAREC_BENCH_WEIGHT_PRECISION='bfloat16; no weight quantization and no CPU offload'
export BAREC_BENCH_CONTEXT_NOTE="Engine context ${MODEL_CONTEXT} tokens; no explicit completion-token cap"
export BAREC_BENCH_SPECULATIVE_NOTE='Target-verified prompt n-gram decoding: 10 draft tokens, n-grams 3..5'
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$BAREC_BENCH_RUN_ROOT/retrieval"
cp retrieval/expanded_matches/*.json "$BAREC_BENCH_RUN_ROOT/retrieval/"

for MODE in gold raw; do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/expanded/run_fanar2_async_worker.py \
    --mode "$MODE" --worker-id 0 --workers 1 --tensor-parallel-size 4 \
    --request-batch-size 16 --request-prompt-token-budget 120000 \
    --structured-backend xgrammar --ngram-speculative-tokens 10 \
    --max-model-len "$MODEL_CONTEXT" --max-num-seqs 16 \
    --max-num-batched-tokens 65536 --gpu-memory-utilization 0.95 \
    "${EXTRA_ARGS[@]}"
done
python benchmarks/expanded/finalize_fanar2_benchmark.py
```

These settings come from the completed BF16 run launchers. The finalizer merges worker responses, applies the original parsing repairs, writes CoNLL-X predictions, calls CAMeL CoNLL, and saves the result table under `<run-root>/results/`. `BAREC_BENCH_MODEL_PATH` can also be an absolute local model snapshot path. A distinct run root keeps different models and datasets from sharing resume records. The worker resumes completed sentences from its response files.

## GPT and Gemini Batch matrix

The expanded cloud runner uses the same prompts, split order, three training pools, and retrieval matches as the asynchronous worker. Its recorded model IDs are `gpt-5.4` and `gemini-3.5-flash`.

Copy the retrieval files to the cloud runner's expected location:

```bash
mkdir -p outputs/expanded/retrieval
cp retrieval/expanded_matches/*.json outputs/expanded/retrieval/
```

Put your OpenAI key in `secrets/gpt.txt` and Gemini key in `secrets/gemini.txt`, one key per line. The directory is ignored by Git. Alternatively, set `BAREC_API_KEY_DIR` to an absolute private directory containing those filenames.

Prepare all requests locally, submit each provider, and finalize:

```bash
python benchmarks/expanded/run_cloud_batch.py prepare --full --run-root outputs/api_batch_full
python benchmarks/expanded/run_cloud_batch.py run --provider openai --run-root outputs/api_batch_full
python benchmarks/expanded/run_cloud_batch.py run --provider gemini --run-root outputs/api_batch_full
python benchmarks/expanded/run_cloud_batch.py finalize --run-root outputs/api_batch_full
```

Preparation performs no API submission. Omit `--full` and choose a separate run root to prepare the two-sentence-per-condition pilot. `status --run-root outputs/api_batch_full` reports provider progress. Outputs, evaluations, and result tables live under the selected run root. The `run` commands submit paid Batch jobs.

The main README also provides the independent GPT Batch and sequential Gemini runners, which accept model IDs and prices through CLI flags. The expanded runner preserves historical price constants and generation settings in its source. For exact historical comparisons, retain the original model identifiers, data snapshots, and prompt files.
