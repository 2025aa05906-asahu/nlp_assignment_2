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

LANG = "hi"          # Target Indian language code
N_SAMPLES = 60_000   # Raw sample download target
MAX_LEN = 34         # Fixed sequence length (30 tokens + <sos> + <eos> + padding buffer)


def parse_args():
    p = argparse.ArgumentParser(description="Task 2 - Data pipeline")
    p.add_argument("--min_freq", type=int, default=4,
                    help="Minimum token frequency to enter the vocabulary. "
                         "Raising this (from the old default of 2) shrinks the "
                         "vocab a lot on a ~39k-sentence corpus, which is what "
                         "you want: fewer rare/singleton tokens for the model "
                         "to have to learn, less <unk> at inference time, and "
                         "a much smaller (cheaper, less overfitting-prone) "
                         "output softmax layer.")
    return p.parse_args()


def main():
    """Executes the complete data preprocessing, vocabulary generation,

    train/val/test splitting, and artifact saving pipeline.
    """
    args = parse_args()
    # Ensure reproducible data splits across runs
    random.seed(42)
    os.makedirs("data", exist_ok=True)

    # Step 1: Stream raw parallel corpus from Hugging Face
    print("[1/5] Fetching dataset from Hugging Face...")
    df_raw = load_raw_dataset(lang=LANG, n_samples=N_SAMPLES)
    df_raw.to_csv("samanantar_en_hi_raw_60k.csv", index=False)

    # Step 2: Clean and filter text according to token count rules
    print("[2/5] Cleaning text and applying sequence filters...")
    df_clean = clean_dataframe(df_raw, min_tokens=5, max_tokens=30)
    
    # Add <sos> and <eos> special tokens to tokenized text
    df_clean["en_tokens"] = df_clean["en"].apply(tokenize_and_add_specials)
    df_clean["tgt_tokens"] = df_clean["tgt"].apply(tokenize_and_add_specials)

    # Step 3: Split dataset into Train (80%), Validation (10%), and Test (10%) sets
    print("[3/5] Splitting datasets (80/10/10)...")
    train_df, temp_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Step 4: Build vocabularies exclusively from training split (prevents test data leakage)
    print("[4/5] Building source and target vocabularies...")
    src_vocab = Vocab(min_freq=args.min_freq).build(train_df["en_tokens"])
    tgt_vocab = Vocab(min_freq=args.min_freq).build(train_df["tgt_tokens"])

    # Step 5: Encode text into padded integer ID sequences and save artifacts to disk
    print("[5/5] Encoding, padding, and saving artifacts to ./data/...")
    
    # Save raw sentence splits for app interface and metric evaluations (BLEU/ROUGE)
    train_df[["en", "tgt"]].to_csv("data/train.csv", index=False)
    val_df[["en", "tgt"]].to_csv("data/val.csv", index=False)
    test_df[["en", "tgt"]].to_csv("data/test.csv", index=False)

    # Helper function to process and save token ID arrays to JSON
    def process_and_save_ids(df, path):
        src_ids = df["en_tokens"].apply(lambda t: encode_and_pad(t, src_vocab, MAX_LEN)).tolist()
        tgt_ids = df["tgt_tokens"].apply(lambda t: encode_and_pad(t, tgt_vocab, MAX_LEN)).tolist()
        with open(path, "w") as f:
            json.dump({"src": src_ids, "tgt": tgt_ids}, f)

    process_and_save_ids(train_df, "data/train_ids.json")
    process_and_save_ids(val_df, "data/val_ids.json")
    process_and_save_ids(test_df, "data/test_ids.json")

    # Save vocabularies as JSON lists for Member 2 (Model) and Member 4 (App)
    with open("data/src_vocab.json", "w", encoding="utf-8") as f:
        json.dump(src_vocab.itos, f, ensure_ascii=False)
    with open("data/tgt_vocab.json", "w", encoding="utf-8") as f:
        json.dump(tgt_vocab.itos, f, ensure_ascii=False)

    # Save sequence metadata configuration file
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