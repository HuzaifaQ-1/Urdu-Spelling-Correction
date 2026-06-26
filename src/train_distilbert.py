from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed" / "all_token_changes.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "distilbert"
WRONG_OPEN = "[WRONG]"
WRONG_CLOSE = "[/WRONG]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune multilingual DistilBERT for Urdu correction prediction."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="distilbert-base-multilingual-cased")
    parser.add_argument(
        "--alignment-type",
        default="token_replace",
        help="Use 'token_replace' for one-to-one corrections or 'all' for every change row.",
    )
    parser.add_argument("--min-class-count", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare data and splits, then stop before downloading/training the model.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def mark_wrong_span(sentence: str, wrong_word: str) -> str:
    words = str(sentence).split()
    target = str(wrong_word).split()
    if not words or not target:
        return str(sentence)

    target_len = len(target)
    for start in range(0, len(words) - target_len + 1):
        if words[start : start + target_len] == target:
            marked = (
                words[:start]
                + [WRONG_OPEN]
                + words[start : start + target_len]
                + [WRONG_CLOSE]
                + words[start + target_len :]
            )
            return " ".join(marked)

    return f"{sentence} {WRONG_OPEN} {wrong_word} {WRONG_CLOSE}"


def load_training_frame(
    data_path: Path,
    alignment_type: str,
    min_class_count: int,
) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df = df.rename(
        columns={
            "CleanOriginal": "Sentence",
            "wrong_word": "WrongWord",
            "correct_word": "CorrectWord",
        }
    )

    required = {"Sentence", "WrongWord", "CorrectWord"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    if alignment_type != "all" and "alignment_type" in df.columns:
        allowed = {part.strip() for part in alignment_type.split(",") if part.strip()}
        df = df[df["alignment_type"].isin(allowed)].copy()

    df = df.dropna(subset=["Sentence", "WrongWord", "CorrectWord"]).copy()
    for column in ["Sentence", "WrongWord", "CorrectWord"]:
        df[column] = df[column].astype(str).str.strip()

    df = df[
        (df["Sentence"] != "")
        & (df["WrongWord"] != "")
        & (df["CorrectWord"] != "")
    ].copy()

    counts = df["CorrectWord"].value_counts()
    valid_labels = counts[counts >= min_class_count].index
    df = df[df["CorrectWord"].isin(valid_labels)].copy()

    if df.empty:
        raise ValueError(
            "No rows left after filtering. Try lowering --min-class-count "
            "or using --alignment-type all."
        )

    df["MarkedSentence"] = [
        mark_wrong_span(sentence, wrong_word)
        for sentence, wrong_word in zip(df["Sentence"], df["WrongWord"])
    ]
    return df.reset_index(drop=True)


def split_frame(
    frame: pd.DataFrame,
    labels: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> dict[str, np.ndarray | list[str]]:
    indices = np.arange(len(frame))
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    train_idx, val_idx, y_train, y_val = train_test_split(
        train_idx,
        y_train,
        test_size=val_size,
        random_state=seed,
        stratify=y_train,
    )

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


class UrduCorrectionDataset(Dataset):
    def __init__(
        self,
        sentences: list[str],
        wrong_words: list[str],
        labels: np.ndarray,
        tokenizer: AutoTokenizer,
        max_len: int,
    ) -> None:
        self.sentences = sentences
        self.wrong_words = wrong_words
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.sentences[idx],
            text_pair=self.wrong_words[idx],
            truncation="only_first",
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
            if key in {"input_ids", "attention_mask"}
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_loader(
    frame: pd.DataFrame,
    indices: np.ndarray,
    labels: np.ndarray,
    tokenizer: AutoTokenizer,
    max_len: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    selected = frame.iloc[indices]
    dataset = UrduCorrectionDataset(
        selected["MarkedSentence"].tolist(),
        selected["WrongWord"].tolist(),
        labels,
        tokenizer,
        max_len,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def optimizer_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, object]]:
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay_params = []
    no_decay_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(marker in name for marker in no_decay):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    model.eval()
    all_logits = []
    all_labels = []
    losses = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }
            logits = model(**inputs).logits
            if loss_fn is not None:
                losses.append(loss_fn(logits, labels).item())
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits_np = torch.cat(all_logits, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()
    avg_loss = float(np.mean(losses)) if losses else None
    return logits_np, labels_np, avg_loss


def metric_dict(labels: np.ndarray, logits: np.ndarray, num_classes: int) -> dict[str, float]:
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    predictions = np.argmax(probs, axis=1)
    top_k = min(3, num_classes)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "top_3_accuracy": top_k_accuracy_score(
            labels,
            probs,
            k=top_k,
            labels=np.arange(num_classes),
        ),
    }


def print_dataset_summary(frame: pd.DataFrame, splits: dict[str, np.ndarray]) -> None:
    print(f"Rows after filtering: {len(frame)}")
    print(f"Classes after filtering: {frame['CorrectWord'].nunique()}")
    print(f"Train rows: {len(splits['train_idx'])}")
    print(f"Validation rows: {len(splits['val_idx'])}")
    print(f"Test rows: {len(splits['test_idx'])}")
    print("\nTop labels:")
    print(frame["CorrectWord"].value_counts().head(10).to_string())


def save_results(
    output_dir: Path,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    label_encoder: LabelEncoder,
    test_frame: pd.DataFrame,
    labels: np.ndarray,
    logits: np.ndarray,
    metrics: dict[str, float],
    history: list[dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    classes = label_encoder.classes_.tolist()
    (output_dir / "label_classes.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    predictions = np.argmax(probs, axis=1)
    top_count = min(3, len(classes))
    top_indices = np.argsort(-probs, axis=1)[:, :top_count]

    prediction_rows = test_frame[["Sentence", "WrongWord", "CorrectWord", "MarkedSentence"]].copy()
    prediction_rows["true_label"] = label_encoder.inverse_transform(labels)
    prediction_rows["predicted_label"] = label_encoder.inverse_transform(predictions)
    prediction_rows["prediction_confidence"] = probs[np.arange(len(probs)), predictions]
    for rank in range(top_count):
        prediction_rows[f"top_{rank + 1}"] = label_encoder.inverse_transform(
            top_indices[:, rank]
        )
        prediction_rows[f"top_{rank + 1}_prob"] = probs[
            np.arange(len(probs)), top_indices[:, rank]
        ]
    prediction_rows.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    configure_console()
    args = parse_args()
    set_seed(args.seed)

    frame = load_training_frame(
        args.data_path,
        alignment_type=args.alignment_type,
        min_class_count=args.min_class_count,
    )

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(frame["CorrectWord"])
    num_classes = len(label_encoder.classes_)
    splits = split_frame(frame, labels, args.test_size, args.val_size, args.seed)
    print_dataset_summary(frame, splits)

    if args.dry_run:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [WRONG_OPEN, WRONG_CLOSE]}
    )

    train_loader = build_loader(
        frame,
        splits["train_idx"],
        splits["y_train"],
        tokenizer,
        args.max_len,
        args.batch_size,
        shuffle=True,
    )
    val_loader = build_loader(
        frame,
        splits["val_idx"],
        splits["y_val"],
        tokenizer,
        args.max_len,
        args.batch_size,
        shuffle=False,
    )
    test_loader = build_loader(
        frame,
        splits["test_idx"],
        splits["y_test"],
        tokenizer,
        args.max_len,
        args.batch_size,
        shuffle=False,
    )

    id2label = {index: label for index, label in enumerate(label_encoder.classes_)}
    label2id = {label: index for index, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=splits["y_train"],
    )
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float, device=device),
        label_smoothing=args.label_smoothing,
    )

    optimizer = AdamW(
        optimizer_groups(model, args.weight_decay),
        lr=args.learning_rate,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_metric = -1.0
    best_state = None
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            labels_batch = batch["labels"].to(device)
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            logits = model(**inputs).logits
            loss = loss_fn(logits, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        val_logits, val_labels, val_loss = collect_logits(model, val_loader, device, loss_fn)
        val_metrics = metric_dict(val_labels, val_logits, num_classes)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={row['train_loss']:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_metric:
            best_metric = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_logits, test_labels, test_loss = collect_logits(model, test_loader, device, loss_fn)
    test_metrics = metric_dict(test_labels, test_logits, num_classes)
    test_metrics["test_loss"] = test_loss

    print("\n===== DISTILBERT TEST RESULTS =====")
    print(pd.DataFrame([test_metrics]).round(4).to_string(index=False))

    probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()
    predictions = np.argmax(probs, axis=1)
    print("\nClassification Report:\n")
    print(
        classification_report(
            test_labels,
            predictions,
            labels=np.arange(num_classes),
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    save_results(
        args.output_dir,
        model,
        tokenizer,
        label_encoder,
        frame.iloc[splits["test_idx"]],
        test_labels,
        test_logits,
        test_metrics,
        history,
    )
    print(f"\nSaved model and metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()
