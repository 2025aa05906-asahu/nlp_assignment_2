"""
Task 4 - Application Development
----------------------------------
A Streamlit web app for the English -> Hindi NMT system. Reuses the same
model-loading and translation logic as infer.py / evaluate.py so behavior
stays consistent everywhere.

Run with:
    streamlit run app.py


"""

import io
import os

import pandas as pd
import streamlit as st

from train import load_vocab, translate_sentence, DEVICE
from infer import load_model


# --------------------------------------------------------------------------- #
# Model loading (cached so it only happens once per session, not per click)
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_model_and_vocabs(data_dir="data", model_dir="models"):
    src_vocab = load_vocab(os.path.join(data_dir, "src_vocab.json"))
    tgt_vocab = load_vocab(os.path.join(data_dir, "tgt_vocab.json"))
    model, max_len = load_model(model_dir)
    return model, max_len, src_vocab, tgt_vocab


st.set_page_config(page_title="English -> Hindi NMT", page_icon="🌐", layout="centered")
st.title("🌐 English → Hindi Neural Machine Translation")
st.caption(
    "Encoder-Decoder (BiLSTM + Bahdanau Attention) trained on the "
    "AI4Bharat Samanantar en-hi parallel corpus."
)

try:
    model, max_len, src_vocab, tgt_vocab = get_model_and_vocabs()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded = False
    st.error(
        "Could not load the trained model. Make sure `models/nmt_model.pt` "
        "exists (run `git lfs pull` if you cloned this repo) and that "
        "`data/src_vocab.json` / `data/tgt_vocab.json` are present.\n\n"
        f"Details: {e}"
    )

with st.sidebar:
    st.header("Decoding options")
    decoding = st.radio("Decoding strategy", ["greedy", "beam"], index=0)
    beam_width = st.slider("Beam width", min_value=2, max_value=10, value=5,
                            disabled=(decoding == "greedy"))
    st.markdown("---")
    st.caption(
        "Note: this model was trained for a limited number of epochs "
        "(see the Evaluation report) — translations may be rough, "
        "especially for long sentences, named entities, and rare words."
    )

tab_single, tab_batch = st.tabs(["✍️ Single sentence", "📄 Batch (file upload)"])

# --------------------------------------------------------------------------- #
# Tab 1: single-sentence text input
# --------------------------------------------------------------------------- #
with tab_single:
    st.subheader("Translate a sentence")
    default_example = "although the weather was extremely unpredictable throughout the week, the farmers finally managed to sow their crops"
    sentence = st.text_area(
        "Enter English text",
        value="",
        placeholder=f"e.g. {default_example}",
        height=100,
    )
    if st.button("Translate", type="primary", disabled=not model_loaded):
        if not sentence.strip():
            st.warning("Please enter a sentence to translate.")
        else:
            with st.spinner("Translating..."):
                output = translate_sentence(
                    model, sentence.strip(), src_vocab, tgt_vocab, max_len,
                    decoding=decoding, beam_width=beam_width,
                )
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original (English)**")
                st.info(sentence.strip())
            with col2:
                st.markdown("**Translation (Hindi)**")
                st.success(output)

# --------------------------------------------------------------------------- #
# Tab 2: batch translation via .txt / .csv upload
# --------------------------------------------------------------------------- #
with tab_batch:
    st.subheader("Batch translate from a file")
    st.caption(
        "Upload a .txt file (one English sentence per line) or a .csv file "
        "with an 'english' column."
    )
    uploaded = st.file_uploader("Choose a file", type=["txt", "csv"])

    if uploaded is not None and model_loaded:
        if uploaded.name.lower().endswith(".csv"):
            df_in = pd.read_csv(uploaded)
            col_name = "english" if "english" in df_in.columns else df_in.columns[0]
            sentences = [str(s) for s in df_in[col_name].dropna().tolist()]
        else:
            text = io.TextIOWrapper(uploaded, encoding="utf-8").read()
            sentences = [line.strip() for line in text.splitlines() if line.strip()]

        if not sentences:
            st.warning("No sentences found in the uploaded file.")
        else:
            st.write(f"Found {len(sentences)} sentence(s). Translating...")
            progress = st.progress(0)
            rows = []
            for i, s in enumerate(sentences):
                out = translate_sentence(model, s, src_vocab, tgt_vocab, max_len,
                                          decoding=decoding, beam_width=beam_width)
                rows.append({"english": s, "translation": out})
                progress.progress((i + 1) / len(sentences))

            result_df = pd.DataFrame(rows)
            st.dataframe(result_df, use_container_width=True)

            csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download results as CSV",
                data=csv_bytes,
                file_name="translations.csv",
                mime="text/csv",
            )
