# Dataset and benchmark versions

## Official dataset

The data are hosted at [CAMeL-Lab/BAREC-TB](https://huggingface.co/datasets/CAMeL-Lab/BAREC-TB). The repository is currently private and will be made public. An account granted access can run `hf auth login` and then:

```bash
python scripts/download_data.py
```

The downloader reads the real archive `Data/BAREC_TB_splits_conllx.zip` at the revision recorded in `huggingface/dataset.json`. It checks the archive and split checksums, extracts only the expected files from `splits_conllx/`, and saves:

```text
data/BAREC-TB/Train.conllx
data/BAREC-TB/Dev.conllx
data/BAREC-TB/Test.conllx
data/BAREC-TB/text/{Train,Dev,Test}.txt
```

These downloaded files are ignored by Git. They contain 9,580 Train, 3,071 Dev, and 2,732 Test sentences. The Hub also hosts document-level annotations and metadata. Its dataset card specifies CC BY-SA 4.0; consult the official dataset for its current terms.

## Saved benchmark snapshots

The code repository retains the exact local split snapshots used by the existing experiments:

- `data/paper/`: the earlier 11,651-sentence version (5,660 Train, 3,179 Dev, 2,812 Test).
- `data/expanded_160k/`: the expanded version (9,580 Train, 3,071 Dev, 2,732 Test).

The official Hugging Face Train file is byte-identical to the saved expanded Train file. The official Dev and Test contain annotation updates, so their checksums differ. Raw sentences, gold-token metadata, and sentence order match the expanded snapshot. `scripts/prepare_barectb_matches.py` additionally checks the FORM sequences before reusing the existing top-10 BARECTB retrieval indices.

The main README runs new evaluations on the official Hub release. The saved tables in `reports/` reflect their original experiment data and checkpoint generations; do not treat them as scores recomputed on updated annotations. The three supervised models published on Hugging Face are the original checkpoints described in their model cards; the later expanded training checkpoints are separate artifacts.

## PATB and CAMeL Treebank pools

For the complete three-pool matrix, supply these files under `data/benchmark/train/`:

- `patb_camel.conllx`: the PATB + CAMeL Treebank training pool in the original sentence order.
- `patb_camel_barec.conllx`: that pool followed by `data/expanded_160k/Train.conllx`.

The existing BARECTB training and gold split links in `data/benchmark/` point to the expanded snapshot. PATB/CAMeL corpora are separate inputs with their own access terms. The optimized retrieval generator verifies the combined pool order. The complete matrix commands are in [BENCHMARKS.md](BENCHMARKS.md).
