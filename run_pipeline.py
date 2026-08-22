import os
import json
import random
from sklearn.model_selection import train_test_split

from src.vocab import Vocab
from src.preprocess import (
    load_raw_dataset,
    clean_dataframe,
    tokenize_and_add_specials,
    encode_and_pad
)

import argparse

LANG = "hi"          # Language identifier passed to the dataset loader.
N_SAMPLES = 60_000   # Maximum number of streamed source-target pairs.
MAX_LEN = 34         # Fixed encoded length for source and target sequences.


def parse_args():
    p = argparse.ArgumentParser(description="Build the translation data pipeline")
    p.add_argument("--min_freq", type=int, default=4,
                    help="Minimum token frequency to enter the vocabulary. "
                         "Higher values exclude infrequent tokens, reducing "
                         "vocabulary size and the output softmax dimensionality.")
    return p.parse_args()


def main():
    """Executes the complete data preprocessing, vocabulary generation,

    train/val/test splitting, and artifact saving pipeline.
    """
    args = parse_args()
    # Set the split sampler seed so generated partitions are repeatable.
    random.seed(42)
    os.makedirs("data", exist_ok=True)

    # Stream source-target pairs from the remote dataset.
    print("[1/5] Fetching dataset from Hugging Face...")
    df_raw = load_raw_dataset(lang=LANG, n_samples=N_SAMPLES)
    df_raw.to_csv("samanantar_en_hi_raw_60k.csv", index=False)

    # Normalize text and retain pairs within the configured token limits.
    print("[2/5] Cleaning text and applying sequence filters...")
    df_clean = clean_dataframe(df_raw, min_tokens=5, max_tokens=30)
    
    # Wrap each token sequence with decoder boundary markers.
    df_clean["en_tokens"] = df_clean["en"].apply(tokenize_and_add_specials)
    df_clean["tgt_tokens"] = df_clean["tgt"].apply(tokenize_and_add_specials)

    # Partition the cleaned pairs into training, validation, and test sets.
    print("[3/5] Splitting datasets (80/10/10)...")
    train_df, temp_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Derive token mappings from training sequences only.
    print("[4/5] Building source and target vocabularies...")
    src_vocab = Vocab(min_freq=args.min_freq).build(train_df["en_tokens"])
    tgt_vocab = Vocab(min_freq=args.min_freq).build(train_df["tgt_tokens"])

    # Encode and persist fixed-width integer sequences.
    print("[5/5] Encoding, padding, and saving artifacts to ./data/...")
    
    # Persist text pairs for application display and metric reference data.
    train_df[["en", "tgt"]].to_csv("data/train.csv", index=False)
    val_df[["en", "tgt"]].to_csv("data/val.csv", index=False)
    test_df[["en", "tgt"]].to_csv("data/test.csv", index=False)

    # Encode both sides of a split and write the resulting arrays as JSON.
    def process_and_save_ids(df, path):
        src_ids = df["en_tokens"].apply(lambda t: encode_and_pad(t, src_vocab, MAX_LEN)).tolist()
        tgt_ids = df["tgt_tokens"].apply(lambda t: encode_and_pad(t, tgt_vocab, MAX_LEN)).tolist()
        with open(path, "w") as f:
            json.dump({"src": src_ids, "tgt": tgt_ids}, f)

    process_and_save_ids(train_df, "data/train_ids.json")
    process_and_save_ids(val_df, "data/val_ids.json")
    process_and_save_ids(test_df, "data/test_ids.json")

    # Persist ordered token lists used to reconstruct vocabulary mappings.
    with open("data/src_vocab.json", "w", encoding="utf-8") as f:
        json.dump(src_vocab.itos, f, ensure_ascii=False)
    with open("data/tgt_vocab.json", "w", encoding="utf-8") as f:
        json.dump(tgt_vocab.itos, f, ensure_ascii=False)

    # Persist sequence and vocabulary metadata used by model construction.
    with open("data/config.json", "w", encoding="utf-8") as f:
        json.dump({
            "MAX_LEN": MAX_LEN,
            "lang_pair": f"en-{LANG}",
            "src_vocab_size": len(src_vocab),
            "tgt_vocab_size": len(tgt_vocab)
        }, f, indent=2)

    print("\n--- Preprocessing Pipeline Summary ---")
    print(f"Cleaned Samples: {len(df_clean)}")
    print(f"Train/Val/Test: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print(f"English Vocab Size: {len(src_vocab)}")
    print(f"Hindi Vocab Size: {len(tgt_vocab)}")


if __name__ == "__main__":
    main()