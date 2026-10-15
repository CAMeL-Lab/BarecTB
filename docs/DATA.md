# Data provenance

`data/paper/` copies the existing `Splits/` files in the source working folder. The three split files contain 5,660 Train, 3,179 Dev, and 2,812 Test sentences. Their sentence total (11,651) and tree-token total (130,671) match the manuscript's split table.

`data/expanded_160k/` copies the existing `new_BAREC_TB_160k/` files. It contains 9,580 Train, 3,071 Dev, and 2,732 Test sentences. This is the later expanded benchmark and must be named when reporting results.

`data/benchmark/` contains relative links for the expanded benchmark code's expected gold and BARECTB training paths. To reproduce the published 10-shot conditions with all training pools, provide these additional files under `data/benchmark/train/`:

- `patb_camel.conllx`: the PATB and CAMeL Treebank training pool used in the experiment.
- `patb_camel_barec.conllx`: that pool followed by expanded BARECTB Train in the original order.

Those two files are deliberately user supplied because they contain third party treebanks. The optimized retrieval script verifies the concatenation order. The original project folder contains local copies, but this release does not copy them into Git.

No data license was present in the source folder. Redistribution terms should be added by the authors before public release.
