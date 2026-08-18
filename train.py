"""
Task 3 - Model Development
---------------------------
Trains an Encoder-Decoder (BiLSTM + Bahdanau Attention) Neural Machine
Translation model on the preprocessed English -> Indian-language corpus
produced by run_pipeline.py (Task 2), and exposes greedy / beam-search
decoding for inference.

Usage:
    python train.py                      # train with default hyperparameters
    python train.py --epochs 15 --batch_size 128

Artifacts written to ./models/:
    nmt_model.pt          - best checkpoint (lowest validation loss)
    loss_curve.png        - training vs validation loss plot
    training_log.json      - per-epoch losses + hyperparameters used
"""

import os
import json
import time
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.vocab import Vocab, PAD, SOS, EOS, UNK

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
class NMTDataset(Dataset):
    """Wraps pre-encoded, pre-padded (src, tgt) integer-ID sequences produced
    by run_pipeline.py so they can be fed straight into a DataLoader."""

    def __init__(self, ids_path: str):
        with open(ids_path, "r") as f:
            data = json.load(f)
        self.src = torch.tensor(data["src"], dtype=torch.long)
        self.tgt = torch.tensor(data["tgt"], dtype=torch.long)

    def __len__(self):
        return self.src.size(0)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


def load_vocab(path: str) -> Vocab:
    """Rebuilds a Vocab object from a saved itos JSON list (see run_pipeline.py)."""
    with open(path, "r", encoding="utf-8") as f:
        itos = json.load(f)
    v = Vocab()
    v.itos = itos
    v.stoi = {t: i for i, t in enumerate(itos)}
    return v


# --------------------------------------------------------------------------- #
# Model: BiLSTM Encoder + Bahdanau-Attention LSTM Decoder
# --------------------------------------------------------------------------- #
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers=1, dropout=0.2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            emb_dim, hid_dim, num_layers=n_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        # Bridge bidirectional encoder final states -> single-direction decoder init state
        self.fc_h = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_c = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h, c) = self.rnn(packed)
        # total_length pins the output back to src's fixed padded length, so it
        # always lines up with the fixed-length attention mask below.
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=src.size(1)
        )
        # outputs: [batch, src_len, hid_dim*2]  (concat of both directions per step)

        # h, c: [n_layers*2, batch, hid_dim] -> take last layer's fwd/bwd and merge
        h_fwd, h_bwd = h[-2], h[-1]
        c_fwd, c_bwd = c[-2], c[-1]
        hidden = torch.tanh(self.fc_h(torch.cat([h_fwd, h_bwd], dim=1)))
        cell = torch.tanh(self.fc_c(torch.cat([c_fwd, c_bwd], dim=1)))
        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    """Additive attention: score(s_{t-1}, h_i) = v^T tanh(W_s s_{t-1} + W_h h_i)."""

    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2 + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        # decoder_hidden: [batch, hid_dim]; encoder_outputs: [batch, src_len, hid_dim*2]
        src_len = encoder_outputs.size(1)
        hidden_rep = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat([hidden_rep, encoder_outputs], dim=2)))
        scores = self.v(energy).squeeze(2)  # [batch, src_len]
        scores = scores.masked_fill(mask == 0, -1e10)
        return F.softmax(scores, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, dropout=0.2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(hid_dim)
        self.rnn = nn.LSTM(hid_dim * 2 + emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 3 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tok, hidden, cell, encoder_outputs, mask):
        # input_tok: [batch] (previous target token id, teacher-forced or predicted)
        input_tok = input_tok.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input_tok))  # [batch, 1, emb_dim]

        attn_weights = self.attention(hidden, encoder_outputs, mask)  # [batch, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hid*2]

        rnn_input = torch.cat([embedded, context], dim=2)
        output, (hidden_new, cell_new) = self.rnn(
            rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0))
        )
        output = output.squeeze(1)          # [batch, hid_dim]
        context = context.squeeze(1)        # [batch, hid_dim*2]
        embedded = embedded.squeeze(1)      # [batch, emb_dim]

        pred = self.fc_out(torch.cat([output, context, embedded], dim=1))
        return pred, hidden_new.squeeze(0), cell_new.squeeze(0), attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, sos_idx, eos_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device

    def make_mask(self, src):
        return (src != self.src_pad_idx).to(self.device)

    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size, device=self.device)
        encoder_outputs, hidden, cell = self.encoder(src, src_lens)
        mask = self.make_mask(src)

        input_tok = tgt[:, 0]  # <sos>
        for t in range(1, tgt_len):
            pred, hidden, cell, _ = self.decoder(input_tok, hidden, cell, encoder_outputs, mask)
            outputs[:, t] = pred
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = pred.argmax(1)
            input_tok = tgt[:, t] if teacher_force else top1
        return outputs

    @torch.no_grad()
    def greedy_decode(self, src, src_lens, max_len=34):
        self.eval()
        encoder_outputs, hidden, cell = self.encoder(src, src_lens)
        mask = self.make_mask(src)
        input_tok = torch.full((src.size(0),), self.sos_idx, dtype=torch.long, device=self.device)

        results = [[] for _ in range(src.size(0))]
        finished = [False] * src.size(0)
        for _ in range(max_len):
            pred, hidden, cell, _ = self.decoder(input_tok, hidden, cell, encoder_outputs, mask)
            top1 = pred.argmax(1)
            for i in range(src.size(0)):
                if not finished[i]:
                    tok = top1[i].item()
                    if tok == self.eos_idx:
                        finished[i] = True
                    else:
                        results[i].append(tok)
            input_tok = top1
            if all(finished):
                break
        return results

    @torch.no_grad()
    def beam_search_decode(self, src, src_lens, beam_width=5, max_len=34, len_norm=0.7):
        """Beam search for a SINGLE sentence (src batch size must be 1)."""
        self.eval()
        assert src.size(0) == 1, "beam_search_decode expects a batch of size 1"
        encoder_outputs, hidden, cell = self.encoder(src, src_lens)
        mask = self.make_mask(src)

        # Each beam: (token_sequence, hidden, cell, log_prob, finished)
        beams = [([self.sos_idx], hidden, cell, 0.0, False)]

        for _ in range(max_len):
            candidates = []
            for seq, h, c, score, finished in beams:
                if finished:
                    candidates.append((seq, h, c, score, finished))
                    continue
                input_tok = torch.tensor([seq[-1]], dtype=torch.long, device=self.device)
                pred, h_new, c_new, _ = self.decoder(input_tok, h, c, encoder_outputs, mask)
                log_probs = F.log_softmax(pred, dim=1).squeeze(0)  # [vocab]
                topk_logp, topk_idx = log_probs.topk(beam_width)
                for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                    new_seq = seq + [idx]
                    new_finished = idx == self.eos_idx
                    candidates.append((new_seq, h_new, c_new, score + lp, new_finished))

            # keep top beam_width candidates, normalizing by length to avoid short-sequence bias
            candidates.sort(key=lambda x: x[3] / (len(x[0]) ** len_norm), reverse=True)
            beams = candidates[:beam_width]
            if all(b[4] for b in beams):
                break

        best_seq = beams[0][0]
        # strip <sos> and trailing <eos>/anything after it
        if self.eos_idx in best_seq:
            best_seq = best_seq[1: best_seq.index(self.eos_idx)]
        else:
            best_seq = best_seq[1:]
        return best_seq


def build_model(src_vocab_size, tgt_vocab_size, pad_idx, sos_idx, eos_idx,
                 emb_dim=256, hid_dim=512, dropout=0.3):
    enc = Encoder(src_vocab_size, emb_dim, hid_dim, dropout=dropout, pad_idx=pad_idx)
    dec = Decoder(tgt_vocab_size, emb_dim, hid_dim, dropout=dropout, pad_idx=pad_idx)
    model = Seq2Seq(enc, dec, pad_idx, sos_idx, eos_idx, DEVICE).to(DEVICE)
    return model


# --------------------------------------------------------------------------- #
# Train / evaluate loops
# --------------------------------------------------------------------------- #
def seq_lengths(batch_src, pad_idx):
    """Computes true (non-pad) length of every row in a padded batch."""
    lens = (batch_src != pad_idx).sum(dim=1)
    lens = torch.clamp(lens, min=1)  # avoid zero-length sequences
    return lens


def run_epoch(model, loader, optimizer, criterion, pad_idx, clip=1.0, train=True,
              teacher_forcing_ratio=0.5):
    model.train() if train else model.eval()
    epoch_loss = 0.0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            src_lens = seq_lengths(src, pad_idx)

            if train:
                optimizer.zero_grad()
            output = model(src, src_lens, tgt,
                            teacher_forcing_ratio=teacher_forcing_ratio if train else 0.0)

            # ignore the <sos> position; flatten for CE loss
            output_dim = output.size(-1)
            output_flat = output[:, 1:].reshape(-1, output_dim)
            tgt_flat = tgt[:, 1:].reshape(-1)
            loss = criterion(output_flat, tgt_flat)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            epoch_loss += loss.item()
    return epoch_loss / len(loader)


def train_model(args):
    os.makedirs(args.model_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "config.json")) as f:
        cfg = json.load(f)
    max_len = cfg["MAX_LEN"]

    src_vocab = load_vocab(os.path.join(args.data_dir, "src_vocab.json"))
    tgt_vocab = load_vocab(os.path.join(args.data_dir, "tgt_vocab.json"))
    pad_idx = src_vocab.stoi[PAD]
    sos_idx = tgt_vocab.stoi[SOS]
    eos_idx = tgt_vocab.stoi[EOS]

    train_ds = NMTDataset(os.path.join(args.data_dir, "train_ids.json"))
    val_ds = NMTDataset(os.path.join(args.data_dir, "val_ids.json"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(
        len(src_vocab), len(tgt_vocab), pad_idx, sos_idx, eos_idx,
        emb_dim=args.emb_dim, hid_dim=args.hid_dim, dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    print(f"Device: {DEVICE} | Src vocab: {len(src_vocab)} | Tgt vocab: {len(tgt_vocab)}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, criterion, pad_idx,
                                clip=args.clip, train=True,
                                teacher_forcing_ratio=args.teacher_forcing_ratio)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, pad_idx,
                              train=False)
        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Time: {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "hyperparameters": vars(args),
                "src_vocab_size": len(src_vocab),
                "tgt_vocab_size": len(tgt_vocab),
                "pad_idx": pad_idx, "sos_idx": sos_idx, "eos_idx": eos_idx,
                "max_len": max_len,
                "epoch": epoch,
                "val_loss": val_loss,
            }, os.path.join(args.model_dir, "nmt_model.pt"))
            print(f"  -> Saved new best checkpoint (val_loss={val_loss:.4f})")

    # Loss curve plot
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, args.epochs + 1), history["train_loss"], label="Train Loss", marker="o")
    plt.plot(range(1, args.epochs + 1), history["val_loss"], label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, "loss_curve.png"), dpi=150)
    print(f"Saved loss curve to {args.model_dir}/loss_curve.png")

    # Training log (config used + per-epoch losses, for the report)
    with open(os.path.join(args.model_dir, "training_log.json"), "w") as f:
        json.dump({"hyperparameters": vars(args), "history": history,
                   "best_val_loss": best_val_loss}, f, indent=2)

    return model, src_vocab, tgt_vocab


# --------------------------------------------------------------------------- #
# Inference helper (used here for a quick sanity check, and reused by app.py)
# --------------------------------------------------------------------------- #
def translate_sentence(model, sentence, src_vocab, tgt_vocab, max_len=34,
                        decoding="greedy", beam_width=5):
    from src.preprocess import clean_text, tokenize_and_add_specials, encode_and_pad

    model.eval()
    cleaned = clean_text(sentence, lowercase=True)
    tokens = tokenize_and_add_specials(cleaned)
    ids = encode_and_pad(tokens, src_vocab, max_len)
    src_tensor = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    src_lens = seq_lengths(src_tensor, src_vocab.stoi[PAD])

    if decoding == "beam":
        out_ids = model.beam_search_decode(src_tensor, src_lens, beam_width=beam_width,
                                            max_len=max_len)
    else:
        out_ids = model.greedy_decode(src_tensor, src_lens, max_len=max_len)[0]

    words = tgt_vocab.decode(out_ids)
    words = [w for w in words if w not in (PAD, SOS, EOS)]
    return " ".join(words)


def parse_args():
    p = argparse.ArgumentParser(description="Train the Seq2Seq NMT model (Task 3).")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--model_dir", type=str, default="models")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--hid_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, src_vocab, tgt_vocab = train_model(args)

    # Quick sanity-check translations printed at the end of training
    with open(os.path.join(args.data_dir, "config.json")) as f:
        cfg = json.load(f)
    print("\n--- Sample translations (greedy) ---")
    for s in ["how are you today", "this is a very important meeting",
              "the government has announced a new policy for farmers"]:
        print(f"EN : {s}")
        print(f"OUT: {translate_sentence(model, s, src_vocab, tgt_vocab, cfg['MAX_LEN'])}\n")
