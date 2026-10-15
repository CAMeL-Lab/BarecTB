# Evaluation

BARECTB data are CoNLL-X with ten tab-separated columns. Sentence metadata includes `# text =` (raw text) and `# treeTokens =` (gold tokens). Dev and Test in `data/paper/` belong to the paper benchmark; `data/expanded_160k/` is a later version. Match the prediction to the same version and split.

Install the original [CAMeL CoNLL](https://github.com/CAMeL-Lab/camel_conll) evaluator, whose `conll_evaluation.py` reports tokenization, POS, UAS, LAS, and related metrics:

```bash
git clone https://github.com/CAMeL-Lab/camel_conll.git external/camel_conll
python3.11 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install -r external/camel_conll/requirements.txt
python evaluation/evaluate.py --gold data/paper/Test.conllx --parsed /path/to/predictions.conllx --normalize --output-dir outputs/eval --name paper_test
```

`--normalize` passes the `-x -n -a` punctuation, number, and Arabic letter normalization flags used by the expanded benchmark scripts. Omit it for an unnormalized score. The wrapper accepts `--camel-conll-root` if the checkout is elsewhere. Gold and prediction files must contain sentences in the same order. For directory evaluation, the evaluator also expects matching filenames.

The expanded summary tables in `reports/` were copied from completed runs; the code in `benchmarks/expanded/` generates the associated predictions and evaluations. Results are version specific.
