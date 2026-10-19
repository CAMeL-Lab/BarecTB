# Hugging Face publication

The three original checkpoints in `models/` are prepared for publication under CAMeL-Lab as a new **CamelParser-BARECTB** collection, following the organization of the existing [CamelParser-Dialects collection](https://huggingface.co/collections/CAMeL-Lab/camelparser-dialects).

| Repository | Local checkpoint | Training condition |
| --- | --- | --- |
| `CAMeL-Lab/camelparser-barectb-BARECTB` | `models/barectb.model` | BARECTB |
| `CAMeL-Lab/camelparser-barectb-PATB-CamelTB` | `models/patb_camel.model` | PATB and CAMeL Treebank |
| `CAMeL-Lab/camelparser-barectb-PATB-CamelTB-BARECTB` | `models/patb_camel_barec.model` | PATB, CAMeL Treebank, and BARECTB |

These are the original paper-era checkpoints selected in the Git repository. Each upload includes the original binary, original YAML configuration, a model card with CAMeL Parser inference instructions, a checksum, and the paper citation. Model cards use the encoder recorded in the YAML configurations. No license or performance scores have been inferred from the dialect models.

From the repository root:

```bash
python -m pip install huggingface_hub
hf auth login
python scripts/upload_huggingface_models.py --dry-run
python scripts/upload_huggingface_models.py
```

The saved token must allow creating model repositories and collections in CAMeL-Lab. The script validates local hashes and model cards, uploads each model, checks the remote checkpoint hash and size, then adds the three verified repositories to the collection. Publication is public, matching the reference collection. Model files are read directly from `models/`; no extra binary copies are staged here. Completed remote URLs and commit IDs are written to `huggingface/published.json`.

The upload uses the [Hugging Face Hub API](https://huggingface.co/docs/huggingface_hub/guides/upload) and [collection API](https://huggingface.co/docs/huggingface_hub/guides/collections).
