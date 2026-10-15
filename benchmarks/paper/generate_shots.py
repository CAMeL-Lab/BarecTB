import os
from typing import List, Tuple, Dict
import json
import argparse
from glob import glob
from collections import Counter


def read_conllx_file(filepath: str) -> List[List[str]]:
    """Reads a CoNLL-X file and returns a list of sentences, each as a list of tokens.

    Assumes:
      - Sentences are separated by blank lines.
      - FORM is column 2 (index 1).
    """
    sentences: List[List[str]] = []
    current_sentence: List[str] = []

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            if line.startswith("#"):  # skip comment/meta lines
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                word = parts[1]
                current_sentence.append(word)

        if current_sentence:
            sentences.append(current_sentence)

    return sentences


# ----- chrF++ helper functions -----

def _char_ngrams(text: str, n: int) -> Counter:
    """Character n-grams of a string."""
    if n <= 0:
        return Counter()
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def _word_ngrams(tokens: List[str], n: int) -> Counter:
    """Word n-grams of a token list."""
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def chrfpp_score(ref_tokens: List[str],
                 hyp_tokens: List[str],
                 max_char_n: int = 6,
                 max_word_n: int = 2,
                 beta: float = 2.0) -> float:
    """
    Approximate chrF++ score between two sentences (ref vs hyp).

    - Uses character n-grams n = 1..max_char_n over the *full string*.
    - Uses word n-grams n = 1..max_word_n over the token sequence.
    - Averages precision/recall over all n-gram orders and types.
    - Then computes F_beta on those averaged P/R.

    Returns:
        score in [0, 1], where 1 = identical, 0 = no overlap.
    """
    # Turn tokens into space-joined strings for char n-grams
    ref_text = " ".join(ref_tokens)
    hyp_text = " ".join(hyp_tokens)

    # Collect precision/recall for char n-grams
    p_char_list = []
    r_char_list = []

    for n in range(1, max_char_n + 1):
        ref_ngr = _char_ngrams(ref_text, n)
        hyp_ngr = _char_ngrams(hyp_text, n)

        if not ref_ngr and not hyp_ngr:
            continue

        overlap = sum((ref_ngr & hyp_ngr).values())
        ref_count = sum(ref_ngr.values())
        hyp_count = sum(hyp_ngr.values())

        p = overlap / hyp_count if hyp_count > 0 else 0.0
        r = overlap / ref_count if ref_count > 0 else 0.0

        p_char_list.append(p)
        r_char_list.append(r)

    # Collect precision/recall for word n-grams
    p_word_list = []
    r_word_list = []

    for n in range(1, max_word_n + 1):
        ref_ngr = _word_ngrams(ref_tokens, n)
        hyp_ngr = _word_ngrams(hyp_tokens, n)

        if not ref_ngr and not hyp_ngr:
            continue

        overlap = sum((ref_ngr & hyp_ngr).values())
        ref_count = sum(ref_ngr.values())
        hyp_count = sum(hyp_ngr.values())

        p = overlap / hyp_count if hyp_count > 0 else 0.0
        r = overlap / ref_count if ref_count > 0 else 0.0

        p_word_list.append(p)
        r_word_list.append(r)

    # If everything is empty (both sentences empty), define score as 0.0
    if not p_char_list and not p_word_list:
        return 0.0

    # Average char and word precisions/recalls
    def _avg(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    p_char = _avg(p_char_list)
    r_char = _avg(r_char_list)
    p_word = _avg(p_word_list)
    r_word = _avg(r_word_list)

    # Combine char + word equally (chrF++)
    p_all = (p_char + p_word) / 2.0
    r_all = (r_char + r_word) / 2.0

    if p_all == 0.0 and r_all == 0.0:
        return 0.0

    beta2 = beta * beta
    score = (1 + beta2) * p_all * r_all / (beta2 * p_all + r_all)
    return score


# ----- Matching logic using chrF++ -----

def get_top_k_matches(test_sentences: List[List[str]],
                      train_sentences: List[List[str]],
                      k: int = 5):
    """
    For each test sentence, compute chrF++ similarity with all train sentences,
    then return top-k highest scores.

    NOTE: The field 'distance' in the returned JSON is actually a similarity score
    in [0, 1], larger = more similar (kept name to avoid breaking downstream code).
    """
    results = []
    for test_idx, test_sent in enumerate(test_sentences):
        scores = []
        print(f"  Processing test sentence {test_idx + 1}/{len(test_sentences)}")
        for train_idx, train_sent in enumerate(train_sentences):
            sim = chrfpp_score(test_sent, train_sent)
            scores.append((train_idx, sim))

        # Sort by similarity descending
        top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:k]

        results.append([
            {"train_index": train_idx, "distance": float(f"{sim:.4f}")}
            for train_idx, sim in top_k
        ])

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute top-k training matches for test files using chrF++ similarity."
    )
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to training CoNLL-X file")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory containing test CoNLL-X files")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file path for combined JSON")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top matches to retrieve")

    args = parser.parse_args()

    train_data = read_conllx_file(args.train_path)
    if os.path.isdir(args.test_dir):
        test_paths = glob(os.path.join(args.test_dir, "*.conllx"))
    else:
        test_paths = [args.test_dir]

    combined_results = {}

    for test_path in test_paths:
        print(f"Processing test file: {test_path}")
        test_data = read_conllx_file(test_path)
        top_k_results = get_top_k_matches(test_data, train_data, k=args.top_k)
        test_filename = os.path.basename(test_path)
        combined_results[test_filename] = top_k_results

    with open(args.output_file, "w", encoding="utf-8") as out:
        json.dump(combined_results, out, indent=2, ensure_ascii=False)
