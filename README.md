# Neural Machine Translation System: English to Hindi

An end-to-end Neural Machine Translation (NMT) application that translates English sentences into Hindi using an Encoder-Decoder architecture with attention. Built for Assignment 2 (GS-1).

## Problem Statement

Digital content, product documentation, government notices and learning material in India are largely authored in English, limiting access for users who prefer regional languages. This project implements an automated English to Hindi translation system to address that gap.

* **Application domain:** Digital content localization and educational accessibility.
* **Target users:** Non-English native Hindi speakers who need access to English-authored digital content, documentation and learning material.
* **Functional requirements:** Accept English text (single sentence or batch file) as input, translate it into Hindi using a trained Encoder-Decoder model, and display the translation alongside the original text through a web interface.
* **Language pair:** English (en) to Hindi (hi).
* **Direction:** One way, EN to HI only. No reverse translation.
* **Input:** English text, entered directly or uploaded as a .txt (one sentence per line) or .csv (english column) file.
* **Output:** Hindi translation, displayed in the app alongside the original English text, with batch results downloadable as CSV.

## Dataset

* **Source:** AI4Bharat Samanantar Corpus, https://huggingface.co/datasets/ai4bharat/samanantar
* **Licence:** CC BY-NC 4.0, non-commercial use.
* **Raw pull size:** approximately 60,000 sentence pairs (English-Hindi), retrieved via the Hugging Face `datasets` library.
* **Cleaned dataset size:** 44,917 sentence pairs, after preprocessing and length filtering.
* **Length constraint:** 5 to 30 tokens on both the English and Hindi side.
* **Split (80/10/10):** train 35,933 pairs, validation 4,492 pairs, test 4,492 pairs.

### Preprocessing
Implemented in `src/preprocess.py` and `run_pipeline.py`:
* Unicode normalization (NFC) on both languages.
* Removal of URLs and HTML entities.
* Removal of zero-width formatting characters from the Hindi text.
* Whitespace collapsing.
* Lowercasing of English source text only (Hindi target text is not lowercased).
* Length filtering: sentences outside the 5 to 30 token range on either side are dropped.
* Deduplication of identical sentence pairs.
* `<sos>` and `<eos>` tokens added to every tokenized sequence.

### Vocabulary and encoding
* Vocabularies are built from the training split only, using `min_freq=4`.
* English vocabulary size: 11,211 tokens.
* Hindi vocabulary size: 10,174 tokens.
* Fixed sequence length (`MAX_LEN`): 34 tokens, including `<sos>` and `<eos>`.
* Sequences are truncated to `MAX_LEN` and right-padded with `<pad>`.
* Encoded train, validation and test sets are stored as integer ID matrices in `data/train_ids.json`, `data/val_ids.json`, `data/test_ids.json`.

## Model Architecture

* **Type:** Sequence-to-sequence with attention (BiLSTM encoder, LSTM decoder).
* **Encoder:** Single layer bidirectional LSTM. Final forward and backward hidden and cell states are projected through a linear layer with tanh activation into the decoder's initial hidden and cell state.
* **Decoder:** Single layer LSTM. At each timestep it attends over all encoder outputs using Bahdanau (additive) attention, concatenates the resulting context vector with the current token embedding, and predicts the next target token.
* **Embeddings:** Separate embedding tables for source and target, `emb_dim=128`.
* **Hidden size:** `hid_dim=256`.
* **Loss:** Token level cross entropy with label smoothing 0.1, ignoring `<pad>` positions.
* **Regularization:** dropout 0.4, weight decay 1e-4, gradient clipping at max norm 1.0.
* **LR schedule:** `ReduceLROnPlateau`, factor 0.5, patience 2.
* **Decoding:** Both greedy decoding and beam search (configurable beam width, length-normalized scoring), implemented in `train.py`. A no-repeat constraint blocks the model from repeating its immediately preceding token and applies a no-repeat-trigram rule during decoding.

### Training configuration used for the submitted checkpoint
* Batch size: 128
* Learning rate: 5e-4 (resumed run, reduced from an initial 1e-3)
* Teacher forcing ratio: 0.6
* Early stopping patience: 6 epochs
* Best validation loss: 5.8715, reached at epoch 20
* Training continued to epoch 26 with no further improvement before early stopping
* Full per-epoch loss history: `models/training_log.json`
* Loss curve plot: `models/loss_curve.png`

## Evaluation 

Run with:
```bash
python evaluate.py --n_samples 500
```
Scored on 500 of 4,492 test pairs, greedy decoding. Results in `models/eval_report.json`.

| Metric | Score |
|---|---|
| BLEU | 17.09 |
| chrF | 22.66 |
| ROUGE-L | 35.17 |
| METEOR | 20.77 |

Failure analysis by sentence length:

| Length bucket | n | avg chrF |
|---|---|---|
| Short, under 10 tokens | 157 | 22.81 |
| Medium, 10 to 20 tokens | 268 | 23.11 |
| Long, over 20 tokens | 75 | 22.69 |

* Repetition-loop rate: 0.0 percent, on this sample the decode-time repetition blocking eliminated repeated-token loops entirely.
* Outputs containing at least one `<unk>` token: 88.2 percent. This is the model's dominant failure mode.
* Named entities (for example person and place names such as "Narendra Modi", "New Delhi") are frequently mistranslated, dropped or repeated incorrectly.
* Long, multi-clause sentences produce degraded, partially incoherent output with a high concentration of `<unk>` tokens, consistent with the chrF trend above.
* Rare words not covered by the `min_freq=4` vocabulary are mapped to `<unk>` at both training and inference time.

Sample demonstration outputs (from `models/eval_report.json`):

| Type | English | Hindi output |
|---|---|---|
| Short | how are you today | kaise aap kya kar rahe hain |
| Named entity | narendra modi visited the parliament in new delhi yesterday | dilli mein dilli ke naye dilli mein nai dilli |
| Long/complex | although the weather was extremely unpredictable throughout the week, the farmers who had been waiting patiently for the monsoon finally managed to sow their crops before the deadline set by the local agricultural department | largely `<unk>` tokens interspersed with partial fragments, see eval_report.json for the exact output |

## Application

Streamlit web application, `app.py`.

Features:
1. Single sentence translation with side by side display of English input and Hindi output.
2. Greedy decoding.
3. Beam search decoding with a configurable beam width (2 to 10).
4. Batch translation via .txt upload (one sentence per line) or .csv upload (english column).
5. Downloadable CSV of batch translation results.

## Repository structure

```
data/
    config.json          sequence length and vocabulary size metadata
    src_vocab.json        English vocabulary
    tgt_vocab.json         Hindi vocabulary
    train.csv / val.csv / test.csv          raw text splits
    train_ids.json / val_ids.json / test_ids.json    encoded, padded ID matrices
samanantar_en_hi_raw_60k.csv    raw pre-cleaning data pull, kept for traceability, not read at runtime
src/
    vocab.py               vocabulary class
    preprocess.py            cleaning, tokenization, encoding
models/
    nmt_model.pt            trained checkpoint, epoch 20, val loss 5.8715
    loss_curve.png            training and validation loss plot
    training_log.json          hyperparameters and per-epoch loss history
    eval_report.json           BLEU, chrF, ROUGE-L, METEOR and failure analysis
run_pipeline.py            data download, cleaning, split, vocabulary build, encoding
train.py                    model definition, training loop, greedy and beam decoding
infer.py                    command line inference from a saved checkpoint
evaluate.py                  evaluation metrics and failure mode analysis
app.py                      Streamlit application
test_input.txt / test_output.txt      sample batch input and output
requirements.txt
README.md
```

## Setup instructions

### 1. Python version
Python 3.10 or higher. Developed and tested on Python 3.11.

### 2. Create a virtual environment
```bash
python -m venv venv
```
Windows:
```powershell
venv\Scripts\Activate.ps1
```
macOS or Linux:
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place the dataset and model files
`data/` and `models/` already contain everything required to run the application: the preprocessed vocabularies, encoded splits, and the trained checkpoint `models/nmt_model.pt`. No download or training step is required before launching the app. No manual file placement is needed, everything is already at the expected paths.

To regenerate the data from scratch instead of using the included files, run:
```bash
python run_pipeline.py --min_freq 4
```
This downloads the Samanantar corpus from Hugging Face and requires internet access.

To retrain or continue training the model instead of using the included checkpoint:
```bash
python train.py
python train.py --resume --epochs 60 --lr 5e-4 --patience 6
```
The included checkpoint was produced with the second command, resumed from an initial run.

### 5. Launch the application
```bash
streamlit run app.py
```
Local URL:
```
http://localhost:8501
```

### 6. Command line inference
```bash
python infer.py "how are you today"
python infer.py "how are you today" --decoding beam --beam_width 5
python infer.py --file test_input.txt --output results.csv
```

### 7. Run evaluation
```bash
python evaluate.py --n_samples 500
```

## Known issues

* Validation loss plateaus at approximately 5.87 to 5.88 after epoch 20 despite continued training and learning rate reduction. This reflects a capacity and data size limit of a 256-hidden-unit LSTM trained on roughly 36,000 sentence pairs, not an undertrained model.
* 88.2 percent of evaluated outputs contain at least one `<unk>` token. This is the primary translation quality limitation, driven by the `min_freq=4` vocabulary cutoff and limited training data.
* Regenerating data via `run_pipeline.py` requires internet access to Hugging Face.
* Training on CPU is slow because LSTMs are sequential. A GPU is recommended for training, though the application itself runs on CPU without issue once a checkpoint exists.
* `samanantar_en_hi_raw_60k.csv` is included for traceability of the cleaning step and is not required by the application, inference, or evaluation scripts.