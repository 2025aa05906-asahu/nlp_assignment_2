# Neural Machine Translation System (English to Hindi)

An end-to-end Neural Machine Translation (NMT) Application that automatically translates English sentences into Hindi using an Encoder–Decoder architecture. Developed for Assignment 2 (GS-1).

---

## **Project Overview**
* **Application Domain:** Digital content localization and educational accessibility.
* **Target Users:** Non-English native speakers requiring access to regional language content.
* **Language Pair:** English (`en`) → Hindi (`hi`)
* **Dataset:** [AI4Bharat Samanantar Corpus](https://huggingface.co/datasets/ai4bharat/samanantar) (CC0 Public Domain License)
* **Dataset Size:** 48,724 cleaned sentence pairs (5–30 token length constraint applied)
* **Data Split:** 80% Train (38,979) / 10% Validation (4,872) / 10% Test (4,873)
* **Vocabulary Sizes:** English: 23,553 tokens | Hindi: 21,233 tokens
* **Fixed Sequence Length (`MAX_LEN`):** 34 (including `<sos>` and `<eos>`)

---

## **Repository Structure**
```text
nlp_assignmnet_2/
├── data/                  # Generated preprocessing artifacts
│   ├── config.json        # Sequence metadata and vocab sizes
│   ├── src_vocab.json     # English vocabulary mapping
│   ├── tgt_vocab.json     # Hindi vocabulary mapping
│   ├── train.csv          # Raw training split text
│   ├── val.csv            # Raw validation split text
│   ├── test.csv           # Raw test split text
│   ├── train_ids.json     # Padded integer ID matrices (Train)
│   ├── val_ids.json       # Padded integer ID matrices (Val)
│   └── test_ids.json      # Padded integer ID matrices (Test)
├── src/
│   ├── __init__.py
│   ├── vocab.py           # Custom Vocabulary class definition
│   └── preprocess.py      # Text cleaning and tokenization utilities
├── run_pipeline.py        # Complete Data Pipeline Execution Script (Task 2)
├── train.py               # Model Training & Loss Tracking Script (Task 3)
├── app.py                 # Streamlit Web Application Interface (Task 4)
├── requirements.txt       # Dependencies with fixed versions
├── .gitignore             # Git exclusion rules
└── README.md              # Setup and execution guide
