---
language:
- ar
base_model: CAMeL-Lab/bert-base-arabic-camelbert-msa
tags:
- dependency-parsing
- arabic
- catib
- barectb
- supar
- pytorch
---

# CamelParser-BARECTB: PATB-CamelTB-BARECTB

A supervised Arabic dependency parser using the CATiB representation, trained on **PATB, CAMeL Treebank, and BARECTB** for **BARECTB: A Dependency Treebank and Parsing Benchmark for Arabic Children’s Texts** (ArabicNLP 2026).

This repository contains the original paper-era `patb_camel_barec.model` checkpoint. It is the same checkpoint distributed in the paper repository's `models/` directory. The later expanded 160K experiments use separate checkpoints.

## Model

- Encoder: [CAMeLBERT-MSA](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa), as recorded in the checkpoint's training configuration.
- Dependency parser: BERT-encoded biaffine parser implemented with [SuPar](https://github.com/yzhangcs/parser) and used through [CAMeL Parser](https://github.com/CAMeL-Lab/camel_parser).
- Training condition: PATB, CAMeL Treebank, and BARECTB.
- Annotation: CATiB dependency relations.
- Checkpoint format: native SuPar `.model`; load with CAMeL Parser/SuPar.

The original `patb_camel_barec.model.yaml` accompanies the checkpoint. Its historical training paths are preserved as provenance; inference uses the downloaded checkpoint.

## Files

| File | Description |
| --- | --- |
| `patb_camel_barec.model` | Original trained parser checkpoint. |
| `patb_camel_barec.model.yaml` | Original training configuration. |
| `SHA256SUMS` | Checkpoint SHA-256 checksum. |
| `CITATION.bib` | Paper citation. |

## Inference with CAMeL Parser

Follow the installation instructions in the [CAMeL Parser main branch](https://github.com/CAMeL-Lab/camel_parser/tree/main), including its Python and CAMeL Tools data requirements. From that checkout, download this checkpoint into its `models/` directory:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="CAMeL-Lab/camelparser-barectb-PATB-CamelTB-BARECTB",
    filename="patb_camel_barec.model",
    local_dir="models",
)
```

Then parse raw Arabic text:

```bash
python text_to_conll_cli.py -f text -m patb_camel_barec.model -s "القطة نائمة."
```

Use `-i input.txt` for a file of sentences. The optional MSA-S31 morphology database requires the separate installation and access steps documented upstream.

## Scope

This checkpoint supports the paper's supervised Arabic dependency parsing experiments. Use the matching original BARECTB splits when reproducing that benchmark. Training corpora and morphology resources are separate from this model repository.

## Citation

Elmadani, Khalid N., Mohamed Adel, Muhammed AbuOdeh, Reem Faraj, Jamila El-Gizuli, Hanada Taha and Nizar Habash. BARECTB: A Dependency Treebank and Parsing Benchmark for Arabic Children’s Texts. In Proceedings of ArabicNLP 2026, Budapest, Hungary, 2026.

```bibtex
@inproceedings{elmadani-etal-2026-barectb,
  title = {{BARECTB}: A Dependency Treebank and Parsing Benchmark for Arabic Children's Texts},
  author = {Elmadani, Khalid N. and Adel, Mohamed and AbuOdeh, Muhammed and Faraj, Reem and El-Gizuli, Jamila and Taha, Hanada and Habash, Nizar},
  booktitle = {Proceedings of ArabicNLP 2026},
  address = {Budapest, Hungary},
  year = {2026}
}
```
