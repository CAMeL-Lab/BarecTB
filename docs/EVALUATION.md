# Evaluate dependency parses

Install the repository's single `requirements.txt` and clone [CAMeL CoNLL](https://github.com/CAMeL-Lab/camel_conll) as shown in the main README. The requirements already include the evaluator's Python dependencies.

```bash
python evaluation/evaluate.py \
  --gold data/BAREC-TB/Test.conllx \
  --parsed outputs/fanar_gold_conllx/Test.conllx \
  --normalize --output-dir outputs/evaluation --name fanar_gold_test
```

Use the same command for supervised, Fanar, Llama, GPT, and Gemini prediction files. Change `--parsed` and `--name` for each condition. The wrapper runs upstream `conll_evaluation.py` and saves its TSV output, which contains tokenization, POS, UAS, LAS, and related scores. LLM runners do not predict POS; POS accuracy is not an LLM benchmark target here.

`--normalize` passes the `-x -n -a` punctuation, number, and Arabic letter normalization flags used by the saved expanded scripts. Omit it for an unnormalized score. Use `--camel-conll-root /path/to/camel_conll` if the checkout is elsewhere.

Gold and prediction files must use the same corpus version and sentence order. CAMeL CoNLL also expects matching filenames, including for file-to-file evaluation; use `Test.conllx` on both sides. For pilot runs, evaluate against the corresponding gold subset rather than the full Test split. CoNLL-X data have ten tab-separated columns, with `# text =` and `# treeTokens =` metadata.

The official Hub release updates some Dev/Test annotations relative to archived benchmarks. [DATA.md](DATA.md) records the version distinctions.
