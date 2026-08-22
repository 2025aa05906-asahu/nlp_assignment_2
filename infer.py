"""
Quick inference script — loads the saved checkpoint (models/nmt_model.pt)
and translates one or more English sentences. No retraining needed.

Usage:
    python infer.py "how are you today"
    python infer.py "how are you today" --decoding beam --beam_width 5
    python infer.py --file sample_inputs.txt      # one sentence per line
"""

import argparse
import json
import os

import torch

from train import build_model, load_vocab, translate_sentence, DEVICE


def load_model(model_dir="models"):
    ckpt_path = os.path.join(model_dir, "nmt_model.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    hp = ckpt["hyperparameters"]

    # FIX3: build_model now needs src_pad_idx AND tgt_pad_idx separately.
    # ckpt.get(..., fallback) keeps this working on OLD checkpoints saved
    # before the fix (which only stored "pad_idx", using it for both sides) --
    # but if you're loading such a checkpoint, retrain so it's saved with a
    # correct, separate tgt_pad_idx instead of relying on this fallback.
    src_pad_idx = ckpt["pad_idx"]
    tgt_pad_idx = ckpt.get("tgt_pad_idx", ckpt["pad_idx"])

    model = build_model(
        ckpt["src_vocab_size"], ckpt["tgt_vocab_size"],
        src_pad_idx, tgt_pad_idx, ckpt["sos_idx"], ckpt["eos_idx"],
        emb_dim=hp["emb_dim"], hid_dim=hp["hid_dim"], dropout=hp["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val_loss={ckpt['val_loss']:.4f}), device={DEVICE}")
    return model, ckpt["max_len"]


def parse_args():
    p = argparse.ArgumentParser(description="Translate sentences using a saved checkpoint.")
    p.add_argument("sentence", nargs="?", type=str, help="A single English sentence to translate.")
    p.add_argument("--file", type=str, default=None,
                   help="Path to a .txt file with one English sentence per line (batch mode).")
    p.add_argument("--output", type=str, default=None,
                   help="Path to save results. .csv writes an en,translation table; "
                        "anything else (e.g. .txt) writes plain EN/OUT pairs.")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--model_dir", type=str, default="models")
    p.add_argument("--decoding", choices=["greedy", "beam"], default="greedy")
    p.add_argument("--beam_width", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.sentence and not args.file:
        raise SystemExit("Provide a sentence as an argument, or --file path/to/sentences.txt")

    src_vocab = load_vocab(os.path.join(args.data_dir, "src_vocab.json"))
    tgt_vocab = load_vocab(os.path.join(args.data_dir, "tgt_vocab.json"))
    model, max_len = load_model(args.model_dir)

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
        if not sentences:
            raise SystemExit(
                f"'{args.file}' has no non-blank lines to translate. "
                f"Check the file actually has content (one English sentence per line)."
            )
    else:
        sentences = [args.sentence]

    results = []
    print()
    for s in sentences:
        out = translate_sentence(model, s, src_vocab, tgt_vocab, max_len,
                                  decoding=args.decoding, beam_width=args.beam_width)
        print(f"EN : {s}")
        print(f"OUT: {out}\n")
        results.append((s, out))

    if args.output:
        if args.output.lower().endswith(".csv"):
            import csv
            with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["english", "translation"])
                writer.writerows(results)
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                for en, out in results:
                    f.write(f"EN : {en}\nOUT: {out}\n\n")
        print(f"Saved {len(results)} translation(s) to {args.output}")