from collections import Counter

# Special tokens required for sequence processing
# <pad>: Padding token to make all sequences equal length
# <sos>: Start-of-sentence token indicating generation start
# <eos>: End-of-sentence token indicating sentence end
# <unk>: Unknown token for words not present in the vocabulary
PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"


class Vocab:
    """Vocabulary mapping between unique string tokens and integer indices.
    Handles token frequency filtering and encoding/decoding operations.
    """

    def __init__(self, min_freq=2):
        """Initializes vocabulary with special tokens.
        Args:
            min_freq (int): Minimum occurrences required for a token to be included.
        """
        self.min_freq = min_freq
        self.itos = [PAD, SOS, EOS, UNK]  # Index-to-String mapping
        self.stoi = {}                    # String-to-Index mapping

    def build(self, list_of_token_lists):
        """Counts word frequencies from training data and populates vocabulary mappings.
        Args:
            list_of_token_lists (list of list of str): Tokenized text rows.
        Returns:
            Vocab: The built Vocab instance.
        """
        # Count occurrences of all non-special tokens across training samples
        counter = Counter(
            tok for toks in list_of_token_lists for tok in toks
            if tok not in (PAD, SOS, EOS)
        )
        
        # Add tokens meeting the minimum frequency threshold
        for tok, freq in counter.items():
            if freq >= self.min_freq:
                self.itos.append(tok)
                
        # Build inverse lookup table (Word -> Integer ID)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        return self

    def encode(self, tokens):
        """Converts a list of word strings into a list of integer token IDs.
        Replaces out-of-vocabulary words with the <unk> token ID.
        """
        return [self.stoi.get(t, self.stoi[UNK]) for t in tokens]

    def decode(self, ids):
        """Converts a list of integer token IDs back into string words."""
        return [self.itos[i] for i in ids if i < len(self.itos)]

    def __len__(self):
        """Returns total vocabulary size including special tokens."""
        return len(self.itos)