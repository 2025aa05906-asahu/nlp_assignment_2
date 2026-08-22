import re
import unicodedata
import pandas as pd
from src.vocab import SOS, EOS, PAD, UNK


def clean_text(text: str, lowercase: bool = False) -> str:
    """Cleans raw text strings by normalizing Unicode characters, removing noise,
    and standardizing whitespace formatting.
    """
    if not isinstance(text, str):
        return ""
    
    # Normalize Unicode so canonically equivalent characters share a form.
    text = unicodedata.normalize("NFC", text)
    
    # Remove URL tokens from the input text.
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    
    # Remove named HTML entities from the input text.
    text = re.sub(r"&[a-z]+;", " ", text)
    
    # Remove zero-width formatting characters from the input text.
    text = re.sub(r"[\u200b\u200c\u200d]", "", text)
    
    # Normalize all whitespace runs to one space.
    text = re.sub(r"\s+", " ", text).strip()
    
    # Apply case folding when processing the English source text.
    if lowercase:
        text = text.lower()
        
    return text


def clean_dataframe(df: pd.DataFrame, min_tokens: int = 5, max_tokens: int = 30) -> pd.DataFrame:
    """Filters dataset rows based on sequence length requirements and removes duplicates.
    Enforces the configured 5–30 token range for each sentence.
    """
    cleaned_rows = []
    seen = set()
    
    for en, tgt in zip(df["en"], df["tgt"]):
        # Normalize each side of the parallel pair independently.
        en_c = clean_text(en, lowercase=True)
        tgt_c = clean_text(tgt, lowercase=False)

        # Exclude pairs with an empty normalized side.
        if not en_c or not tgt_c:
            continue

        # Enforce the token bounds on both source and target sequences.
        n_tok_en = len(en_c.split())
        n_tok_tgt = len(tgt_c.split())
        if not (min_tokens <= n_tok_en <= max_tokens):
            continue
        if not (min_tokens <= n_tok_tgt <= max_tokens):
            continue

        # Retain only the first occurrence of each normalized pair.
        key = (en_c, tgt_c)
        if key in seen:
            continue
        seen.add(key)

        cleaned_rows.append({"en": en_c, "tgt": tgt_c})
        
    return pd.DataFrame(cleaned_rows)


def tokenize_and_add_specials(text: str) -> list:
    """Splits text into whitespace tokens and wraps them with start (<sos>)
    and end (<eos>) sequence tokens.
    """
    return [SOS] + text.split() + [EOS]


def encode_and_pad(tokens: list, vocab, max_len: int = 34) -> list:
    """Encodes tokens to IDs, truncates longer sequences, and right-pads shorter
    sequences to a fixed length tensor shape.
    """
    # Truncate before mapping tokens to integer IDs.
    ids = vocab.encode(tokens)[:max_len]
    
    # Right-pad every encoded sequence to the configured length.
    ids = ids + [vocab.stoi[PAD]] * (max_len - len(ids))
    return ids


def load_raw_dataset(lang: str = "hi", n_samples: int = 60000) -> pd.DataFrame:
    """Streams parallel sentence pairs from AI4Bharat's Samanantar corpus via HuggingFace.
    Streaming allows fetching sample subsets without downloading multi-gigabyte files.
    """
    # Import on demand so local preprocessing utilities do not require datasets.
    from datasets import load_dataset
    raw_stream = load_dataset("ai4bharat/samanantar", lang, split="train", streaming=True)
    rows = []
    
    for ex in raw_stream:
        rows.append({"en": ex["src"], "tgt": ex["tgt"]})
        if len(rows) >= n_samples:
            break
            
    return pd.DataFrame(rows)