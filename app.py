from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict_correction import (  # noqa: E402
    DEFAULT_LABELS_PATH,
    DEFAULT_LIKELY_REAL_WORD_PATH,
    DEFAULT_MODEL_DIR,
    load_curated_pairs,
    load_model_and_tokenizer,
    predict_curated_sentence,
    predict_sentence,
)


PREDICTIONS_PATH = BASE_DIR / "outputs" / "distilbert" / "test_predictions.csv"


def rtl_text(text: str, class_name: str = "urdu-box") -> None:
    st.markdown(
        f'<div class="{class_name}" dir="rtl" lang="ur">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading trained DistilBERT model...")
def get_model():
    return load_model_and_tokenizer(DEFAULT_MODEL_DIR, DEFAULT_LABELS_PATH)


@st.cache_data(show_spinner=False)
def get_curated_pairs():
    return load_curated_pairs(DEFAULT_LIKELY_REAL_WORD_PATH)


@st.cache_data(show_spinner=False)
def load_samples(limit: int = 8) -> list[dict[str, str]]:
    if not PREDICTIONS_PATH.exists():
        return []

    predictions = pd.read_csv(PREDICTIONS_PATH, encoding="utf-8-sig")
    required = {"Sentence", "WrongWord", "CorrectWord", "predicted_label"}
    if not required.issubset(predictions.columns):
        return []

    correct = predictions[predictions["CorrectWord"] == predictions["predicted_label"]].copy()
    correct = correct.dropna(subset=["Sentence", "WrongWord", "CorrectWord"])
    correct["Sentence"] = correct["Sentence"].astype(str).str.strip()
    correct = correct[correct["Sentence"] != ""]

    samples = []
    seen = set()
    for _, row in correct.iterrows():
        sentence = str(row["Sentence"]).strip()
        wrong_word = str(row["WrongWord"]).strip()
        correct_word = str(row["CorrectWord"]).strip()
        key = (sentence, wrong_word, correct_word)
        if key in seen:
            continue
        seen.add(key)
        samples.append(
            {
                "sentence": sentence,
                "wrong_word": wrong_word,
                "correct_word": correct_word,
                "label": f"Sample {len(samples) + 1}: {wrong_word} -> {correct_word}",
            }
        )
        if len(samples) >= limit:
            break

    return samples


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 1040px;
            padding-top: 2rem;
        }
        .urdu-box {
            direction: rtl;
            unicode-bidi: plaintext;
            text-align: right;
            font-family: "Noto Nastaliq Urdu", "Jameel Noori Nastaleeq",
                "Noto Naskh Arabic", "Segoe UI", Tahoma, Arial, sans-serif;
            color: #111827;
            font-size: 1.25rem;
            line-height: 2.25;
            word-spacing: 0.12rem;
            overflow-wrap: anywhere;
            border: 1px solid #d9dee7;
            background: #fbfcff;
            border-radius: 8px;
            padding: 1rem 1.15rem;
            margin: 0.25rem 0 0.75rem;
        }
        .urdu-large {
            direction: rtl;
            unicode-bidi: plaintext;
            text-align: right;
            font-family: "Noto Nastaliq Urdu", "Jameel Noori Nastaleeq",
                "Noto Naskh Arabic", "Segoe UI", Tahoma, Arial, sans-serif;
            color: #0f172a;
            font-size: 1.45rem;
            line-height: 2.35;
            word-spacing: 0.14rem;
            overflow-wrap: anywhere;
            border: 1px solid #b7d7c5;
            background: #f3fff7;
            border-radius: 8px;
            padding: 1rem 1.15rem;
            margin: 0.25rem 0 1rem;
        }
        .metric-word {
            direction: rtl;
            unicode-bidi: isolate;
            font-family: "Noto Nastaliq Urdu", "Jameel Noori Nastaleeq",
                "Noto Naskh Arabic", "Segoe UI", Tahoma, Arial, sans-serif;
            font-size: 1.35rem;
            font-weight: 700;
        }
        div[data-testid="stTextArea"] textarea {
            direction: rtl;
            text-align: right;
            unicode-bidi: plaintext;
            font-family: "Noto Nastaliq Urdu", "Jameel Noori Nastaleeq",
                "Noto Naskh Arabic", "Segoe UI", Tahoma, Arial, sans-serif;
            font-size: 1.15rem;
            line-height: 2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_candidate_table(result) -> None:
    rows = []
    for candidate in result.candidates[:8]:
        rows.append(
            {
                "Detected word": candidate.original_word,
                "Suggested word": candidate.predicted_word,
                "Confidence": f"{candidate.confidence:.2%}",
                "Margin": f"{candidate.margin:.2%}",
                "Source": candidate.source,
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Urdu Correction Demo",
        layout="wide",
    )
    inject_styles()

    st.title("Urdu Real-Word Correction Demo")
    st.caption("Sentence-only demo using the trained DistilBERT correction model.")

    samples = load_samples()
    sample_labels = ["Custom sentence"] + [sample["label"] for sample in samples]

    with st.sidebar:
        st.header("Demo Controls")
        selected_label = st.selectbox("Sample", sample_labels)
        threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.60, 0.05)
        max_len = st.slider("Max token length", 64, 256, 128, 16)
        show_candidates = st.slider("Candidate rows", 3, 12, 8, 1)
        use_curated = st.checkbox("Use curated real-word file first", value=True)

        st.divider()
        st.write("The app scans each word internally. You only provide the sentence.")
        st.write("Curated mode uses likely_real_word_corrections.csv before DistilBERT.")

    selected_sample = None
    if selected_label != "Custom sentence":
        selected_sample = next(sample for sample in samples if sample["label"] == selected_label)

    default_sentence = selected_sample["sentence"] if selected_sample else ""
    if st.session_state.get("last_selected_label") != selected_label:
        st.session_state.sentence_input = default_sentence
        st.session_state.last_selected_label = selected_label
    if "sentence_input" not in st.session_state:
        st.session_state.sentence_input = default_sentence

    sentence = st.text_area(
        "Urdu sentence",
        key="sentence_input",
        height=170,
        placeholder="یہاں اردو جملہ لکھیں",
    )

    col_predict, col_clear = st.columns([1, 5])
    predict_clicked = col_predict.button("Predict", type="primary", use_container_width=True)
    if col_clear.button("Reset", use_container_width=True):
        st.session_state.sentence_input = default_sentence
        st.rerun()

    if selected_sample:
        st.write("Expected sample correction:")
        st.markdown(
            f'<span class="metric-word" dir="rtl">{html.escape(selected_sample["wrong_word"])}</span>'
            f" &rarr; "
            f'<span class="metric-word" dir="rtl">{html.escape(selected_sample["correct_word"])}</span>',
            unsafe_allow_html=True,
        )

    if not predict_clicked:
        st.info("Choose a sample or enter a sentence, then click Predict.")
        return

    sentence = sentence.strip()
    if not sentence:
        st.warning("Please enter an Urdu sentence first.")
        return

    result = None
    if use_curated:
        curated_pairs = get_curated_pairs()
        result = predict_curated_sentence(
            sentence,
            curated_pairs,
            threshold=threshold,
            top_k=3,
        )
        if not result.best or not result.found_error:
            result = None

    if result is None:
        model, tokenizer, labels = get_model()
        with st.spinner("Scanning possible error positions with DistilBERT..."):
            result = predict_sentence(
                sentence,
                model,
                tokenizer,
                labels,
                threshold=threshold,
                max_len=max_len,
                top_k=3,
            )

    if result.best is None:
        st.warning("No Urdu words were found to scan.")
        return

    best = result.best

    st.subheader("Input")
    rtl_text(result.sentence)

    if result.found_error:
        st.success("High-confidence correction found.")
    else:
        st.warning("No high-confidence correction found. Showing the strongest candidate.")

    metric_cols = st.columns(5)
    metric_cols[0].markdown("Detected word")
    metric_cols[0].markdown(
        f'<div class="metric-word" dir="rtl">{html.escape(best.original_word)}</div>',
        unsafe_allow_html=True,
    )
    metric_cols[1].markdown("Suggested word")
    metric_cols[1].markdown(
        f'<div class="metric-word" dir="rtl">{html.escape(best.predicted_word)}</div>',
        unsafe_allow_html=True,
    )
    metric_cols[2].metric("Confidence", f"{best.confidence:.2%}")
    metric_cols[3].metric("Margin vs #2", f"{best.margin:.2%}")
    metric_cols[4].metric("Source", "Curated" if best.source.startswith("curated") else "DistilBERT")

    st.subheader("Corrected Sentence")
    rtl_text(best.corrected_sentence, "urdu-large")

    st.subheader("Top Suggestions")
    suggestions = pd.DataFrame(
        {
            "Suggestion": [item["word"] for item in best.top_suggestions],
            "Confidence": [f"{float(item['confidence']):.2%}" for item in best.top_suggestions],
        }
    )
    st.dataframe(suggestions, hide_index=True, use_container_width=True)

    st.subheader("Scanned Candidates")
    render_candidate_table(result)


if __name__ == "__main__":
    main()
