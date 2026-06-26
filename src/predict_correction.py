from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from huggingface_hub.utils import disable_progress_bars
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as transformers_logging


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = BASE_DIR / "outputs" / "distilbert" / "model"
DEFAULT_LABELS_PATH = BASE_DIR / "outputs" / "distilbert" / "label_classes.json"
DEFAULT_LIKELY_REAL_WORD_PATH = BASE_DIR / "data" / "processed" / "likely_real_word_corrections.csv"
WRONG_OPEN = "[WRONG]"
WRONG_CLOSE = "[/WRONG]"
URDU_TOKEN_RE = re.compile(r"^([^\u0600-\u06FF]*)([\u0600-\u06FF]+)([^\u0600-\u06FF]*)$")

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
disable_progress_bars()
transformers_logging.disable_progress_bar()
transformers_logging.set_verbosity_error()


@dataclass
class CandidatePrediction:
    token_index: int
    original_word: str
    predicted_word: str
    confidence: float
    margin: float
    corrected_sentence: str
    top_suggestions: list[dict[str, float | str]]
    source: str = "distilbert"


@dataclass
class SentencePrediction:
    sentence: str
    threshold: float
    found_error: bool
    best: CandidatePrediction | None
    candidates: list[CandidatePrediction]


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan an Urdu sentence for likely real-word corrections."
    )
    parser.add_argument(
        "sentence",
        nargs="?",
        help="Urdu sentence to scan. If omitted, the script asks for input.",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--show-candidates", type=int, default=5)
    parser.add_argument("--likely-path", type=Path, default=DEFAULT_LIKELY_REAL_WORD_PATH)
    parser.add_argument(
        "--no-curated",
        action="store_true",
        help="Disable the curated real-word pair layer and use DistilBERT only.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON instead of a readable report.",
    )
    return parser.parse_args()


def load_labels(labels_path: Path, model_config: object) -> list[str]:
    if labels_path.exists():
        return json.loads(labels_path.read_text(encoding="utf-8"))

    id2label = getattr(model_config, "id2label", None)
    if id2label:
        return [id2label[str(index)] if str(index) in id2label else id2label[index] for index in range(len(id2label))]

    raise FileNotFoundError(
        f"Could not find labels at {labels_path}. Pass --labels-path explicitly."
    )


def split_token(token: str) -> tuple[str, str, str] | None:
    match = URDU_TOKEN_RE.match(token)
    if not match:
        return None
    prefix, word, suffix = match.groups()
    return prefix, word, suffix


def mark_token(tokens: list[str], token_index: int, prefix: str, word: str, suffix: str) -> str:
    marked_tokens = tokens.copy()
    marked_tokens[token_index] = f"{prefix}{WRONG_OPEN} {word} {WRONG_CLOSE}{suffix}"
    return " ".join(marked_tokens)


def replace_token(tokens: list[str], token_index: int, prefix: str, replacement: str, suffix: str) -> str:
    corrected_tokens = tokens.copy()
    corrected_tokens[token_index] = f"{prefix}{replacement}{suffix}"
    return " ".join(corrected_tokens)


def build_candidate_inputs(sentence: str) -> list[dict[str, object]]:
    tokens = sentence.split()
    candidates = []

    for index, token in enumerate(tokens):
        split = split_token(token)
        if split is None:
            continue

        prefix, word, suffix = split
        candidates.append(
            {
                "token_index": index,
                "original_word": word,
                "marked_sentence": mark_token(tokens, index, prefix, word, suffix),
                "corrected_prefix": prefix,
                "corrected_suffix": suffix,
                "tokens": tokens,
            }
        )

    return candidates


def load_curated_pairs(likely_path: Path = DEFAULT_LIKELY_REAL_WORD_PATH) -> dict[str, list[dict[str, object]]]:
    if not likely_path.exists():
        return {}

    import pandas as pd

    frame = pd.read_csv(likely_path, encoding="utf-8-sig")
    required = {"wrong_word", "correct_word"}
    if not required.issubset(frame.columns):
        return {}

    frame = frame.dropna(subset=["wrong_word", "correct_word"]).copy()
    frame["wrong_word"] = frame["wrong_word"].astype(str).str.strip()
    frame["correct_word"] = frame["correct_word"].astype(str).str.strip()
    frame = frame[(frame["wrong_word"] != "") & (frame["correct_word"] != "")]

    grouped = (
        frame.groupby(["wrong_word", "correct_word"])
        .size()
        .reset_index(name="count")
        .sort_values(["wrong_word", "count", "correct_word"], ascending=[True, False, True])
    )

    pairs: dict[str, list[dict[str, object]]] = {}
    for wrong_word, rows in grouped.groupby("wrong_word", sort=False):
        total = int(rows["count"].sum())
        choices = []
        for _, row in rows.iterrows():
            choices.append(
                {
                    "word": str(row["correct_word"]),
                    "count": int(row["count"]),
                    "confidence": float(row["count"] / total) if total else 0.0,
                }
            )
        pairs[str(wrong_word)] = choices

    return pairs


def predict_curated_sentence(
    sentence: str,
    curated_pairs: dict[str, list[dict[str, object]]],
    threshold: float = 0.60,
    top_k: int = 3,
) -> SentencePrediction:
    sentence = " ".join(str(sentence).split())
    tokens = sentence.split()
    candidates: list[CandidatePrediction] = []

    for index, token in enumerate(tokens):
        split = split_token(token)
        if split is None:
            continue

        prefix, word, suffix = split
        suggestions = curated_pairs.get(word)
        if not suggestions:
            continue

        top_suggestions = [
            {
                "word": str(item["word"]),
                "confidence": float(item["confidence"]),
            }
            for item in suggestions[:top_k]
        ]
        predicted_word = str(top_suggestions[0]["word"])
        confidence = float(top_suggestions[0]["confidence"])
        margin = (
            confidence - float(top_suggestions[1]["confidence"])
            if len(top_suggestions) > 1
            else confidence
        )

        candidates.append(
            CandidatePrediction(
                token_index=index,
                original_word=word,
                predicted_word=predicted_word,
                confidence=confidence,
                margin=margin,
                corrected_sentence=replace_token(tokens, index, prefix, predicted_word, suffix),
                top_suggestions=top_suggestions,
                source="curated_real_word_pairs",
            )
        )

    ranked = sorted(candidates, key=lambda item: item.confidence, reverse=True)
    best = ranked[0] if ranked else None

    return SentencePrediction(
        sentence=sentence,
        threshold=threshold,
        found_error=bool(best and best.confidence >= threshold),
        best=best,
        candidates=ranked,
    )


def predict_sentence(
    sentence: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    labels: list[str],
    threshold: float = 0.60,
    max_len: int = 128,
    top_k: int = 3,
) -> SentencePrediction:
    sentence = " ".join(str(sentence).split())
    candidates = build_candidate_inputs(sentence)
    if not candidates:
        return SentencePrediction(
            sentence=sentence,
            threshold=threshold,
            found_error=False,
            best=None,
            candidates=[],
        )

    device = next(model.parameters()).device
    marked_sentences = [str(candidate["marked_sentence"]) for candidate in candidates]
    wrong_words = [str(candidate["original_word"]) for candidate in candidates]
    encoded = tokenizer(
        marked_sentences,
        text_pair=wrong_words,
        truncation="only_first",
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=1).cpu()

    top_k = min(top_k, len(labels))
    candidate_predictions: list[CandidatePrediction] = []

    for row_index, candidate in enumerate(candidates):
        probs = probabilities[row_index]
        top_probs, top_indices = torch.topk(probs, k=top_k)
        top_suggestions = [
            {
                "word": labels[int(label_index)],
                "confidence": float(probability),
            }
            for probability, label_index in zip(top_probs, top_indices)
        ]

        predicted_word = str(top_suggestions[0]["word"])
        confidence = float(top_suggestions[0]["confidence"])
        margin = (
            confidence - float(top_suggestions[1]["confidence"])
            if len(top_suggestions) > 1
            else confidence
        )
        corrected_sentence = replace_token(
            candidate["tokens"],  # type: ignore[arg-type]
            int(candidate["token_index"]),
            str(candidate["corrected_prefix"]),
            predicted_word,
            str(candidate["corrected_suffix"]),
        )

        candidate_predictions.append(
            CandidatePrediction(
                token_index=int(candidate["token_index"]),
                original_word=str(candidate["original_word"]),
                predicted_word=predicted_word,
                confidence=confidence,
                margin=margin,
                corrected_sentence=corrected_sentence,
                top_suggestions=top_suggestions,
            )
        )

    ranked = sorted(candidate_predictions, key=lambda item: item.confidence, reverse=True)
    changed_ranked = [
        candidate
        for candidate in ranked
        if candidate.original_word != candidate.predicted_word
    ]
    best = changed_ranked[0] if changed_ranked else ranked[0]

    return SentencePrediction(
        sentence=sentence,
        threshold=threshold,
        found_error=best.confidence >= threshold and best.original_word != best.predicted_word,
        best=best,
        candidates=ranked,
    )


def load_model_and_tokenizer(
    model_dir: Path,
    labels_path: Path,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer, list[str]]:
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Could not find trained model at {model_dir}. Run python src/train_distilbert.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    labels = load_labels(labels_path, model.config)

    if len(labels) != model.config.num_labels:
        raise ValueError(
            f"Label count mismatch: {len(labels)} labels, model expects {model.config.num_labels}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, labels


def print_report(result: SentencePrediction, show_candidates: int) -> None:
    print("\nInput sentence:")
    print(result.sentence)

    if result.best is None:
        print("\nNo Urdu tokens found to scan.")
        return

    best = result.best
    if result.found_error:
        print("\nBest correction:")
    else:
        print("\nNo high-confidence correction found.")
        print("Best low-confidence candidate:")

    print(f"Detected word     : {best.original_word}")
    print(f"Suggested word    : {best.predicted_word}")
    print(f"Source            : {best.source}")
    print(f"Confidence        : {best.confidence:.2%}")
    print(f"Margin vs #2      : {best.margin:.2%}")
    print("Corrected sentence:")
    print(best.corrected_sentence)

    print("\nTop suggestions for detected word:")
    for rank, suggestion in enumerate(best.top_suggestions, start=1):
        print(f"{rank}. {suggestion['word']} ({float(suggestion['confidence']):.2%})")

    print(f"\nTop {min(show_candidates, len(result.candidates))} scanned candidates:")
    for candidate in result.candidates[:show_candidates]:
        print(
            f"- {candidate.original_word} -> {candidate.predicted_word} "
            f"({candidate.confidence:.2%})"
        )


def main() -> None:
    configure_console()
    args = parse_args()
    sentence = args.sentence or input("Enter an Urdu sentence: ").strip()

    curated_result = None
    if not args.no_curated:
        curated_pairs = load_curated_pairs(args.likely_path)
        curated_result = predict_curated_sentence(
            sentence,
            curated_pairs,
            threshold=args.threshold,
            top_k=args.top_k,
        )

    if curated_result and curated_result.best and curated_result.found_error:
        result = curated_result
    else:
        result = None

    if result is None:
        model, tokenizer, labels = load_model_and_tokenizer(args.model_dir, args.labels_path)
        result = predict_sentence(
            sentence,
            model,
            tokenizer,
            labels,
            threshold=args.threshold,
            max_len=args.max_len,
            top_k=args.top_k,
        )

    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print_report(result, args.show_candidates)


if __name__ == "__main__":
    main()
