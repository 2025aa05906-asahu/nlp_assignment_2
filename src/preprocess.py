import re
import unicodedata
import pandas as pd
from datasets import load_dataset
from src.vocab import SOS, EOS, PAD, UNK


def clean_text(text: str, lowercase: bool = False) -> str:
    """Cleans raw text strings by normalizing Unicode characters, removing noise,
    and standardizing whitespace formatting.
    """
    if not isinstance(text, str):
        return ""
    
    # Standardize script character representation (NFC format for Indic scripts)
    text = unicodedata.normalize("NFC", text)
    
    # Strip web URLs and HTTP links
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    
    # Remove HTML entities (e.g., &amp;, &lt;)
    text = re.sub(r"&[a-z]+;", " ", text)
    
    # Remove zero-width formatting characters commonly found in Indic OCR data
    text = re.sub(r"[\u200b\u200c\u200d]", "", text)
    
    # Collapse multiple blank spaces into a single space
    text = re.sub(r"\s+", " ", text).strip()
    
    # Lowercase English text (kept False for target Indic scripts)
    if lowercase:
        text = text.lower()
        
    return text


def clean_dataframe(df: pd.DataFrame, min_tokens: int = 5, max_tokens: int = 30) -> pd.DataFrame:
    """Filters dataset rows based on sequence length requirements and removes duplicates.
    Enforces assignment constraint of 5–30 tokens per sentence.
    """
    cleaned_rows = []
    seen = set()
    
    for en, tgt in zip(df["en"], df["tgt"]):
        # Clean both source and target columns
        en_c = clean_text(en, lowercase=True)
        tgt_c = clean_text(tgt, lowercase=False)

        # Skip empty strings after cleaning
        if not en_c or not tgt_c:
            continue

        # Filter by source sentence length (5 to 30 tokens)
        n_tok = len(en_c.split())
        if not (min_tokens <= n_tok <= max_tokens):
            continue

        # Deduplicate identical sentence pairs
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
    # Truncate tokens to max length and convert to IDs
    ids = vocab.encode(tokens)[:max_len]
    
    # Append padding tokens up to max_len
    ids = ids + [vocab.stoi[PAD]] * (max_len - len(ids))
    return ids


def load_raw_dataset(lang: str = "hi", n_samples: int = 60000) -> pd.DataFrame:
    """Streams parallel sentence pairs from AI4Bharat's Samanantar corpus via HuggingFace.
    Streaming allows fetching sample subsets without downloading multi-gigabyte files.
    """
    raw_stream = load_dataset("ai4bharat/samanantar", lang, split="train", streaming=True)
    rows = []
    
    for ex in raw_stream:
        rows.append({"en": ex["src"], "tgt": ex["tgt"]})
        if len(rows) >= n_samples:
            break
            
    return pd.DataFrame(rows)