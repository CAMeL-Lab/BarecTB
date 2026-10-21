# Supervised parser checkpoints

The weights are hosted in the [CAMeL-Lab CamelParser-BARECTB collection](https://huggingface.co/collections/CAMeL-Lab/camelparser-barectb-6ad5d147a150e48c97598630). Git tracks the original configurations and checksums; downloaded `.model` files are ignored.

From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/download_models.py
```

| Checkpoint saved here | Hugging Face repository |
| --- | --- |
| `barectb.model` | [BARECTB](https://huggingface.co/CAMeL-Lab/camelparser-barectb-BARECTB) |
| `patb_camel.model` | [PATB + CAMeL Treebank](https://huggingface.co/CAMeL-Lab/camelparser-barectb-PATB-CamelTB) |
| `patb_camel_barec.model` | [PATB + CAMeL Treebank + BARECTB](https://huggingface.co/CAMeL-Lab/camelparser-barectb-PATB-CamelTB-BARECTB) |

Downloads use the verified commit IDs in `huggingface/published.json` and check SHA-256 against `huggingface/manifest.json`. Existing matching files are reused. To fetch one model, add `--model barectb`, `--model patb_camel`, or `--model patb_camel_barec`.

See [supervised inference instructions](../docs/SUPERVISED.md) to link these files into CAMeL Parser and run inference.
