# BARECTB

BARECTB is a CATiB dependency treebank and parsing benchmark for Arabic children's texts. This repository collects the corpus splits, model download and inference instructions, and the actual retrieval, LLM benchmarking, and evaluation code used in the project. The three supervised CAMeL Parser checkpoints are hosted on Hugging Face.

## Contents

| Path | Contents |
| --- | --- |
| `data/paper/` | Original paper Train, Dev, Test, and All CoNLL-X files: 11,651 sentences, 130,671 tree tokens across the three splits. |
| `data/expanded_160k/` | Later expanded Train, Dev, Test, and All CoNLL-X files: 15,383 sentences, 185,552 tree tokens across the three splits. These are a separate benchmark version. |
| `models/` | Original supervised model configurations and checksums. Download checkpoint weights here from Hugging Face. |
| `huggingface/` | Published model cards, manifest, and verified Hugging Face model and collection URLs. |
| `benchmarks/paper/` | Original gold-token and raw-text vLLM, GPT Batch, and Gemini runners, plus chrF++ shot generation. |
| `benchmarks/expanded/` | Later shared Fanar-2/Llama 4 Scout vLLM runner, optimized top-10 chrF++ retrieval, and GPT-5.4/Gemini-3.5-Flash Batch runner. |
| `benchmarks/prompts/` | Gold-token and raw-text prompt YAML used by the runners. |
| `retrieval/` | Original top-5 and expanded top-10 precomputed chrF++ similarity indices for Dev and Test. |
| `evaluation/` | Original parser prediction scripts and a small launcher for upstream CAMeL CoNLL evaluation. |
| `reports/` | Summary TSVs from the completed expanded benchmarks. |

The paper version and expanded version must be compared separately; they have different splits and model generations. The Hugging Face collection contains the original paper-era three checkpoints; the downloader saves them into `models/`. The later expanded checkpoints are separate artifacts and are not included in this release.

Download all three supervised models from the [CAMeL-Lab CamelParser-BARECTB collection](https://huggingface.co/collections/CAMeL-Lab/camelparser-barectb-6ad5d147a150e48c97598630):

```bash
python -m pip install -r requirements-models.txt
python scripts/download_models.py
```

The downloader uses verified Hugging Face revisions and checks each checkpoint’s SHA-256 checksum. Downloaded weights are ignored by Git.

## Start here

- [Supervised parser setup and inference](docs/SUPERVISED.md)
- [LLM benchmarks and similarity based shots](docs/BENCHMARKS.md)
- [Evaluation and data format](docs/EVALUATION.md)
- [Data provenance and release boundaries](docs/DATA.md)

External tools are linked rather than copied into this repository: [CAMeL Parser](https://github.com/CAMeL-Lab/camel_parser) for inference and [CAMeL CoNLL](https://github.com/CAMeL-Lab/camel_conll) for evaluation. The parser's upstream README specifies Python 3.11.13 and documents the optional licensed MSA-S31 morphology database. See the setup guides above.

## Citation

Elmadani, Khalid N., Mohamed Adel, Muhammed AbuOdeh, Reem Faraj, Jamila El-Gizuli, Hanada Taha and Nizar Habash. **BARECTB: A Dependency Treebank and Parsing Benchmark for Arabic Children’s Texts.** In *Proceedings of ArabicNLP 2026*, Budapest, Hungary, 2026.

BibTeX: [docs/CITATION.bib](docs/CITATION.bib). A `CITATION.cff` is included for GitHub's citation display.

## Release notes

The original project did not include a BARECTB license file, so this repository does not assert one. PATB, CAMeL Treebank, local morphology databases, and model weights for Fanar/Llama are separate inputs with their own access terms; the benchmark guide explains where they enter the code. API keys belong in environment variables or ignored local files. Original experiment logs and provider responses are not included in the compact release tree.
