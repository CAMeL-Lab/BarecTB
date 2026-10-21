# Environment and reproduction notes

## One dependency file

Install `requirements.txt` with Python 3.11. It contains the direct Python dependencies for downloads, supervised parsing, retrieval, local LLM inference, cloud API clients, and evaluation. Package dependencies bring in the corresponding tokenizers, CAMeL Tools utilities, and vLLM runtime libraries. SuPar and `conllx-df` use the Git commits specified by the original upstream dependency files.

Full vLLM inference targets Linux x86_64 with NVIDIA GPUs. The requirements use a platform marker to skip vLLM on macOS; that platform can be used for downloads and cloud clients, while local GPU runs need the documented Linux host. A working NVIDIA driver and access to the model weights are machine requirements, not Python packages. Git and a C++ compiler may also be needed for dependencies built from source.

Do not install the older `camel_parser/requirements.txt` and `camel_conll/requirements.txt` freezes over this environment. Those contain conflicting Transformers, NumPy, and PyTorch pins. The consolidated file uses CAMeL Tools 1.6.0, Transformers 4.57.1, PyTorch 2.9.0, and vLLM 0.11.1 so their declared dependency ranges are compatible.

## Original runtime versus installation baseline

The archived expanded Fanar/Llama logs identify **vLLM `0.11.1rc3.dev20+g3b7bdf983`**. `requirements.txt` pins the adjacent public `0.11.1` release as an installable baseline. It does not claim to recreate that development build or every historical GPU kernel. The older parser requirements used CAMeL Tools 1.5.6 and Transformers 4.29.2; those exact pins cannot share the vLLM environment. The consolidated set is a compatibility update, not the original environment lock.

The legacy SuPar checkpoints use full-object serialization. With PyTorch 2.9.0, the supervised instructions set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for the verified published checkpoints. This follows the [PyTorch serialization setting](https://docs.pytorch.org/docs/2.9/notes/serialization.html#environment-variables).

The consolidated dependencies have been resolved for Linux/Python 3.11 and installed in a fresh Python 3.11 verification environment on macOS (where the vLLM marker skips that package). A CPU gold-token inference smoke test loaded the original BARECTB checkpoint and parsed a sentence; CAMeL CoNLL also returned 100% LAS/UAS on an identical gold/prediction pair. Full GPU inference, raw-text morphology processing, and complete score/timing reproduction still require validation on the target machine.

## Data, prompts, and results

The model downloader pins the verified model commit IDs in `huggingface/published.json`. The dataset downloader pins the archive revision and checksums in `huggingface/dataset.json`. Original prompt YAML and retrieval scoring functions are preserved. The official Dev/Test annotation updates are documented in [DATA.md](DATA.md).

The README's server/client commands provide individual runs on the official dataset. They do not reproduce the asynchronous GPU scheduling and timing measurements in the archived experiment matrix. Use [BENCHMARKS.md](BENCHMARKS.md) for those worker settings, training pools, and original snapshot paths. Historical API prices, provider model identifiers, and server behavior can change; use the supplied CLI options to set the intended model and rates for a new run.
