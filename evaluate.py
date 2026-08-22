"""
Loads the trained checkpoint (models/nmt_model.pt) and the held-out test
split (data/test_ids.json), generates translations, and reports:

  1. Corpus-level automatic metrics: BLEU, chrF, ROUGE-L, METEOR
  2. A qualitative demo on a curated set of sample sentences, including at
     least one long/complex sentence (printed + saved so it can be
    inspected alongside the computed metrics)
  3. A programmatic failure-mode analysis: performance by sentence length,
     repetition-loop detection, and OOV(<unk>)/rare-word impact — these are
     the "rare words / named entities / long sentences / word-order" angles
    including rare words, named entities, long sentences, and word order.
"""

import argparse
import json
import os
import random

import torch

from train import build_model, load_vocab, translate_sentence, DEVICE, seq_lengths
from infer import load_model
from src.vocab import PAD, SOS, EOS, UNK


# Import available metric implementations independently.
def _try_import_metrics():
    metrics = {}
    try:
        import sacrebleu
        metrics["sacrebleu"] = sacrebleu
    except ImportError:
        print("[warn] sacrebleu not installed -> BLEU/chrF will be skipped. "
              "pip install sacrebleu")

    try:
        from rouge_score import rouge_scorer
        metrics["rouge_scorer"] = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    except ImportError:
        print("[warn] rouge-score not installed -> ROUGE-L will be skipped. "
              "pip install rouge-score")

    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        metrics["meteor_score"] = meteor_score
    except Exception as e:
        print(f"[warn] METEOR unavailable ({e}) -> will be skipped. "
              "pip install nltk, then python -c \"import nltk; nltk.download('wordnet')\"")

    return metrics


# Reconstruct token text from the encoded test sequences.
def decode_ids_to_text(ids, vocab):
    toks = vocab.decode(ids)
    toks = [t for t in toks if t not in (PAD, SOS, EOS)]
    return " ".join(toks), toks


def load_test_pairs(data_dir, src_vocab, tgt_vocab, n_samples=None, seed=42):
    with open(os.path.join(data_dir, "test_ids.json")) as f:
        data = json.load(f)

    n_total = len(data["src"])
    indices = list(range(n_total))
    if n_samples is not None and n_samples < n_total:
        random.seed(seed)
        indices = sorted(random.sample(indices, n_samples))

    pairs = []
    for i in indices:
        src_text, _ = decode_ids_to_text(data["src"][i], src_vocab)
        ref_text, ref_toks = decode_ids_to_text(data["tgt"][i], tgt_vocab)
        pairs.append({"src": src_text, "ref": ref_text, "ref_tokens": ref_toks})
    return pairs, n_total


# Compute corpus-level translation metrics.
def compute_corpus_metrics(hyps, refs, metrics):
    results = {}

    if "sacrebleu" in metrics:
        sb = metrics["sacrebleu"]
        results["BLEU"] = round(sb.corpus_bleu(hyps, [refs]).score, 2)
        results["chrF"] = round(sb.corpus_chrf(hyps, [refs]).score, 2)

    if "rouge_scorer" in metrics:
        scorer = metrics["rouge_scorer"]
        f1s = [scorer.score(ref, hyp)["rougeL"].fmeasure for hyp, ref in zip(hyps, refs)]
        results["ROUGE-L"] = round(100 * sum(f1s) / max(len(f1s), 1), 2)

    if "meteor_score" in metrics:
        meteor_fn = metrics["meteor_score"]
        scores = [meteor_fn([ref.split()], hyp.split()) for hyp, ref in zip(hyps, refs)]
        results["METEOR"] = round(100 * sum(scores) / max(len(scores), 1), 2)

    return results


# Compute length, repetition, and unknown-token statistics.
def has_repetition_loop(tokens, min_run=3):
    """Flags the 'क्या क्या क्या' style collapse: the same token repeated
    min_run+ times in a row anywhere in the hypothesis."""
    run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            run += 1
            if run >= min_run:
                return True
        else:
            run = 1
    return False


def failure_analysis(pairs, hyps, metrics):
    sb = metrics.get("sacrebleu")
    buckets = {"short (<10 tok)": [], "medium (10-20 tok)": [], "long (>20 tok)": []}
    repetition_flags = 0
    unk_flags = 0

    per_example_chrf = []
    for pair, hyp in zip(pairs, hyps):
        src_len = len(pair["src"].split())
        hyp_toks = hyp.split()

        if sb is not None:
            score = sb.sentence_chrf(hyp, [pair["ref"]]).score
        else:
            score = None
        per_example_chrf.append(score)

        if src_len < 10:
            buckets["short (<10 tok)"].append(score)
        elif src_len <= 20:
            buckets["medium (10-20 tok)"].append(score)
        else:
            buckets["long (>20 tok)"].append(score)

        if has_repetition_loop(hyp_toks):
            repetition_flags += 1
        if UNK in hyp_toks:
            unk_flags += 1

    length_report = {}
    for name, scores in buckets.items():
        scores = [s for s in scores if s is not None]
        length_report[name] = {
            "n": len(scores),
            "avg_chrF": round(sum(scores) / len(scores), 2) if scores else None,
        }

    n = len(pairs)
    return {
        "by_length": length_report,
        "repetition_loop_pct": round(100 * repetition_flags / n, 1) if n else 0,
        "contains_unk_pct": round(100 * unk_flags / n, 1) if n else 0,
    }


# Representative examples covering short, named-entity, and complex inputs.
DEMO_SENTENCES = [
    ("short", "how are you today"),
    ("short", "good morning"),
    ("named_entity", "narendra modi visited the parliament in new delhi yesterday"),
    ("long_complex",
     "although the weather was extremely unpredictable throughout the "
     "week, the farmers who had been waiting patiently for the monsoon "
     "finally managed to sow their crops before the deadline set by the "
     "local agricultural department"),
]


def run_demo(model, src_vocab, tgt_vocab, max_len, decoding, beam_width):
    print("\n--- Qualitative translation examples ---")
    demo_results = []
    for tag, sentence in DEMO_SENTENCES:
        out = translate_sentence(model, sentence, src_vocab, tgt_vocab, max_len,
                                  decoding=decoding, beam_width=beam_width)
        print(f"[{tag}]")
        print(f"EN : {sentence}")
        print(f"OUT: {out}\n")
        demo_results.append({"tag": tag, "en": sentence, "out": out})
    return demo_results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate translations and analyze failure modes")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--model_dir", type=str, default="models")
    p.add_argument("--n_samples", type=int, default=500,
                    help="Random subset of the test set to score (use -1 for the full set).")
    p.add_argument("--decoding", choices=["greedy", "beam"], default="greedy")
    p.add_argument("--beam_width", type=int, default=5)
    p.add_argument("--output", type=str, default="models/eval_report.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    n_samples = None if args.n_samples == -1 else args.n_samples

    src_vocab = load_vocab(os.path.join(args.data_dir, "src_vocab.json"))
    tgt_vocab = load_vocab(os.path.join(args.data_dir, "tgt_vocab.json"))
    model, max_len = load_model(args.model_dir)

    metrics = _try_import_metrics()

    print(f"\nLoading test pairs (n_samples={'all' if n_samples is None else n_samples}) ...")
    pairs, n_total = load_test_pairs(args.data_dir, src_vocab, tgt_vocab, n_samples)
    print(f"Scoring {len(pairs)} / {n_total} test pairs using {args.decoding} decoding ...")

    hyps = []
    for i, pair in enumerate(pairs, 1):
        out = translate_sentence(model, pair["src"], src_vocab, tgt_vocab, max_len,
                                  decoding=args.decoding, beam_width=args.beam_width)
        hyps.append(out)
        if i % 100 == 0:
            print(f"  ... {i}/{len(pairs)}")

    refs = [p["ref"] for p in pairs]

    print("\n--- Corpus-level metrics ---")
    corpus_metrics = compute_corpus_metrics(hyps, refs, metrics)
    for k, v in corpus_metrics.items():
        print(f"{k:10s}: {v}")

    print("\n--- Failure-mode analysis ---")
    fail_report = failure_analysis(pairs, hyps, metrics)
    print(json.dumps(fail_report, indent=2, ensure_ascii=False))

    demo_results = run_demo(model, src_vocab, tgt_vocab, max_len,
                             args.decoding, args.beam_width)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "n_test_pairs_scored": len(pairs),
            "n_test_pairs_total": n_total,
            "decoding": args.decoding,
            "corpus_metrics": corpus_metrics,
            "failure_analysis": fail_report,
            "demo_examples": demo_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full report to {args.output}")