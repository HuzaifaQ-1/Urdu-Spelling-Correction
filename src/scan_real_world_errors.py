from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT = BASE_DIR / "data" / "raw" / "UrduSpellDataset.csv"
OUT_DIR = BASE_DIR / "data" / "processed"


def u(escaped: str) -> str:
    return escaped.encode("ascii").decode("unicode_escape")


def clean_urdu(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"[A-Za-z0-9]", " ", text)
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: object) -> list[str]:
    return clean_urdu(text).split()


DIACRITICS_RE = re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = DIACRITICS_RE.sub("", text)
    return text.translate(
        str.maketrans(
            {
                "\u0643": "\u06a9",
                "\u06aa": "\u06a9",
                "\u064a": "\u06cc",
                "\u0649": "\u06cc",
                "\u06c0": "\u06c1",
                "\u0629": "\u06c1",
                "\u0647": "\u06c1",
                "\u0623": "\u0627",
                "\u0625": "\u0627",
                "\u0671": "\u0627",
                "\u0672": "\u0627",
                "\u0673": "\u0627",
                "\u0640": "",
            }
        )
    )


def coarse_norm(text: str) -> str:
    return norm(text).translate(
        str.maketrans(
            {
                "\u0622": "\u0627",
                "\u0626": "\u06cc",
                "\u0678": "\u06cc",
                "\u0624": "\u0648",
                "\u06d2": "\u06cc",
                "\u06be": "\u06c1",
                "\u06c3": "\u06c1",
                "\u06ba": "\u0646",
            }
        )
    )


def edit_distance(left: str, right: str) -> int:
    left = norm(left)
    right = norm(right)
    previous = list(range(len(right) + 1))
    for i, ca in enumerate(left, 1):
        current = [i] + [0] * len(right)
        for j, cb in enumerate(right, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            )
        previous = current
    return previous[-1]


def is_obvious_orthographic_variant(wrong: str, correct: str) -> bool:
    if norm(wrong) == norm(correct):
        return True
    if coarse_norm(wrong) == coarse_norm(correct):
        return True
    return False


LIKELY_PAIRS = {
    (u(a), u(b))
    for a, b in [
        # Function words / particles / postpositions.
        (r"\u06a9\u06d2", r"\u06a9\u06c1"),
        (r"\u06a9\u06c1", r"\u06a9\u06d2"),
        (r"\u06a9\u06d2", r"\u06a9\u06cc"),
        (r"\u06a9\u06cc", r"\u06a9\u06d2"),
        (r"\u06a9\u06d2", r"\u0646\u06d2"),
        (r"\u06a9\u06cc", r"\u06a9\u06cc\u0627"),
        (r"\u06a9\u0648", r"\u06a9\u06cc"),
        (r"\u0633\u06d2", r"\u0646\u06d2"),
        (r"\u06a9\u0627", r"\u06a9\u06d2"),
        (r"\u06a9\u0627", r"\u06a9\u06cc"),
        (r"\u06a9\u0627", r"\u06a9\u06c1"),
        (r"\u06a9\u0627", r"\u06a9\u0648\u0626\u06cc"),
        (r"\u06a9\u06c1", r"\u06a9\u06cc"),
        (r"\u06a9\u06c1", r"\u06a9\u0627"),
        (r"\u06a9\u06c1", r"\u06a9\u0631"),
        (r"\u06a9\u0631", r"\u06a9\u06d2"),
        (r"\u06a9\u0631", r"\u06a9\u0648"),
        (r"\u06a9\u06c1\u0627", r"\u06a9\u06c1"),
        (r"\u06c1\u0631", r"\u067e\u0631"),
        (r"\u0628\u0631", r"\u067e\u0631"),
        (r"\u067e\u06d2", r"\u06c1\u06d2"),
        (r"\u0628\u06d2", r"\u06c1\u06d2"),
        (r"\u0633\u06cc", r"\u0633\u06d2"),
        (r"\u06a9\u06cc\u0633\u06cc", r"\u06a9\u0633\u06cc"),
        (r"\u06a9\u0633\u06d2", r"\u06a9\u0633\u06cc"),
        (r"\u0627\u0633", r"\u0627\u0633\u06cc"),
        (r"\u0627\u0633\u06d2", r"\u0627\u06cc\u0633\u06d2"),
        (r"\u06c1\u0645", r"\u06c1\u0645\u06cc\u06ba"),
        # Pronoun/adjective agreement.
        (r"\u0627\u067e\u0646\u0627", r"\u0627\u067e\u0646\u06cc"),
        (r"\u0627\u067e\u0646\u06d2", r"\u0627\u067e\u0646\u06cc"),
        (r"\u0645\u06cc\u0631\u0627", r"\u0645\u06cc\u0631\u06d2"),
        (r"\u0645\u06cc\u0631\u0627", r"\u0645\u06cc\u0631\u06cc"),
        (r"\u062a\u06cc\u0631\u06c1", r"\u062a\u06cc\u0631\u0627"),
        # Verb/copula/agreement/contextual form substitutions.
        (r"\u06c1\u06d2", r"\u06c1\u06cc\u06ba"),
        (r"\u06c1\u06cc\u06ba", r"\u06c1\u06d2"),
        (r"\u06c1\u0648", r"\u06c1\u06cc\u06ba"),
        (r"\u06c1\u0648", r"\u06c1\u0648\u06ba"),
        (r"\u06c1\u0648\u06ba", r"\u06c1\u0648"),
        (r"\u06c1\u06cc", r"\u06c1\u06d2"),
        (r"\u06c1\u06cc", r"\u06c1\u06cc\u06ba"),
        (r"\u0631\u06c1\u06cc\u06ba", r"\u0631\u06c1\u06d2"),
        (r"\u0631\u06c1\u06cc\u06d2", r"\u0631\u06c1\u06d2"),
        (r"\u06a9\u0631\u062a\u06cc", r"\u06a9\u0631\u062a\u06d2"),
        (r"\u06a9\u0631\u0648\u06ba", r"\u06a9\u0631\u0648"),
        (r"\u06a9\u0631\u0648\u06ba", r"\u062f\u0648\u06ba"),
        (r"\u062f\u06cc", r"\u062f\u06d2"),
        (r"\u062f\u06cc\u0631", r"\u062f\u06d2"),
        (r"\u0646\u0627", r"\u06c1\u0648\u0646\u0627"),
        (r"\u062c\u0627", r"\u062c\u0648"),
        (r"\u0686\u0644\u0648", r"\u0686\u0644\u0648\u06ba"),
        (r"\u0627\u0691\u0627\u0646\u06d2", r"\u0627\u0691\u0627\u0646\u06cc"),
        (r"\u06af\u06cc", r"\u06af\u0626\u06cc"),
        (r"\u06af\u0626\u06d2", r"\u06af\u06d2"),
        (r"\u06af\u06cc\u06ba", r"\u06af\u06cc"),
        (r"\u06af\u06cc\u06ba", r"\u06af\u06d2"),
        (r"\u0628\u0646", r"\u0628\u0646\u06cc"),
        (r"\u0628\u0646\u0627\u0646\u06d2", r"\u0628\u0646\u0627\u062a\u06d2"),
        (r"\u0686\u0644\u06cc", r"\u0686\u0644\u06d2"),
        (r"\u0645\u0644\u06cc\u06ba", r"\u0645\u0644\u06d2"),
        (r"\u0644\u0691\u06a9\u06cc\u0627\u06ba", r"\u0644\u0691\u06a9\u06cc\u0648\u06ba"),
        (r"\u067e\u0691\u06be\u06cc\u06ba", r"\u067e\u0691\u06be\u06cc"),
        (r"\u062c\u0648\u062a\u0627", r"\u06c1\u0648\u062a\u0627"),
        (r"\u0628\u0686\u06c1", r"\u0628\u0686\u06d2"),
        # Content-word / phrase-level contextual substitutions.
        (r"\u0645\u0627\u0646", r"\u0645\u0627\u06ba"),
        (r"\u0645\u0631\u0627\u062f", r"\u0645\u0631\u062f\u06c1"),
        (r"\u0622\u0628\u0627\u062f", r"\u0628\u0627\u062f"),
        (r"\u06a9\u0631\u062f\u0627\u0631", r"\u0628\u062f\u06a9\u0631\u062f\u0627\u0631"),
        (r"\u0634\u0631\u0645", r"\u0628\u06d2\u0634\u0631\u0645"),
        (r"\u063a\u06cc\u0631\u062a", r"\u0628\u06d2\u063a\u06cc\u0631\u062a"),
        (r"\u062a\u0645\u06cc\u0632", r"\u0628\u062f\u062a\u0645\u06cc\u0632"),
        (r"\u0639\u0627\u0645", r"\u0633\u0631\u0639\u0627\u0645"),
        (r"\u062f\u0644\u06cc\u0631", r"\u062f\u0644\u06cc\u0631\u06cc"),
        (r"\u0645\u0631\u06cc\u062f", r"\u0645\u0631\u062a\u062f"),
        (r"\u0645\u0631\u062a\u062f", r"\u0645\u0631\u062f"),
        (r"\u0639\u0645\u0631", r"\u0648\u0642\u062a"),
    ]
}


def classify_pair(wrong: str, correct: str, wrong_proc_freq: int) -> tuple[str, str]:
    if (wrong, correct) in LIKELY_PAIRS:
        return "likely_real_word_contextual", "manual full-corpus review: source and target are valid words/forms in context"
    if wrong_proc_freq > 0:
        if is_obvious_orthographic_variant(wrong, correct):
            return "probably_spelling_or_orthographic", "source also appears in processed corpus, but pair is a common orthographic variant"
        return "possible_real_word_candidate", "source token appears somewhere in processed corpus; needs lexicon/human review"
    return "not_real_word_by_proxy", "source token was not observed as a processed-corpus word"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT, encoding="utf-8-sig")
    df["CleanOriginal"] = df["TweetOriginal"].map(clean_urdu)
    df["CleanProcessed"] = df["TweetProcess"].map(clean_urdu)

    processed_tokens = [tok for text in df["TweetProcess"] for tok in tokenize(text)]
    original_tokens = [tok for text in df["TweetOriginal"] for tok in tokenize(text)]
    proc_freq = Counter(processed_tokens)
    orig_freq = Counter(original_tokens)

    records: list[dict[str, object]] = []
    occurrence = Counter()

    for row_index, row in df.iterrows():
        original_tokens_row = row["CleanOriginal"].split()
        processed_tokens_row = row["CleanProcessed"].split()
        matcher = difflib.SequenceMatcher(
            a=original_tokens_row,
            b=processed_tokens_row,
            autojunk=False,
        )

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            original_block = original_tokens_row[i1:i2]
            processed_block = processed_tokens_row[j1:j2]

            if tag == "replace" and len(original_block) == len(processed_block):
                aligned = list(zip(original_block, processed_block))
                alignment_type = "token_replace"
            elif tag == "replace" and len(original_block) == 1 and len(processed_block) == 1:
                aligned = [(original_block[0], processed_block[0])]
                alignment_type = "token_replace"
            else:
                aligned = [(" ".join(original_block), " ".join(processed_block))]
                alignment_type = tag

            for wrong, correct in aligned:
                occurrence[(row["TweetId"], wrong, correct)] += 1
                wrong_proc_freq = int(proc_freq.get(wrong, 0)) if " " not in wrong else 0
                correct_proc_freq = int(proc_freq.get(correct, 0)) if " " not in correct else 0
                label, reason = classify_pair(wrong, correct, wrong_proc_freq)
                records.append(
                    {
                        "row_index": int(row_index),
                        "TweetId": int(row["TweetId"]),
                        "occurrence_in_row": occurrence[(row["TweetId"], wrong, correct)],
                        "alignment_type": alignment_type,
                        "wrong_word": wrong,
                        "correct_word": correct,
                        "review_label": label,
                        "review_reason": reason,
                        "wrong_in_processed_freq": wrong_proc_freq,
                        "correct_in_processed_freq": correct_proc_freq,
                        "wrong_in_original_freq": int(orig_freq.get(wrong, 0)) if " " not in wrong else 0,
                        "edit_distance": edit_distance(wrong, correct) if " " not in wrong and " " not in correct else "",
                        "orthographic_variant": is_obvious_orthographic_variant(wrong, correct)
                        if " " not in wrong and " " not in correct
                        else "",
                        "original_block": " ".join(original_block),
                        "processed_block": " ".join(processed_block),
                        "CleanOriginal": row["CleanOriginal"],
                        "CleanProcessed": row["CleanProcessed"],
                        "TweetOriginal": row["TweetOriginal"],
                        "TweetProcess": row["TweetProcess"],
                    }
                )

    changes = pd.DataFrame(records)
    changes.to_csv(OUT_DIR / "all_token_changes.csv", index=False, encoding="utf-8-sig")

    one_to_one = changes[changes["alignment_type"].eq("token_replace")].copy()
    broad = one_to_one[one_to_one["wrong_in_processed_freq"].gt(0)].copy()
    broad.to_csv(OUT_DIR / "broad_real_word_candidates.csv", index=False, encoding="utf-8-sig")

    likely = one_to_one[one_to_one["review_label"].eq("likely_real_word_contextual")].copy()
    likely.to_csv(OUT_DIR / "likely_real_word_corrections.csv", index=False, encoding="utf-8-sig")

    pair_summary = (
        one_to_one.groupby(["wrong_word", "correct_word", "review_label", "review_reason"], dropna=False)
        .agg(
            instances=("TweetId", "size"),
            unique_rows=("TweetId", "nunique"),
            wrong_in_processed_freq=("wrong_in_processed_freq", "max"),
            correct_in_processed_freq=("correct_in_processed_freq", "max"),
            tweet_ids=("TweetId", lambda values: ",".join(map(str, list(values)[:25]))),
        )
        .reset_index()
        .sort_values(
            ["review_label", "instances", "wrong_in_processed_freq", "correct_in_processed_freq"],
            ascending=[True, False, False, False],
        )
    )
    pair_summary.to_csv(OUT_DIR / "pair_summary.csv", index=False, encoding="utf-8-sig")

    summary_lines = [
        f"rows_total,{len(df)}",
        f"tweetprocess_missing,{int(df['TweetProcess'].isna().sum())}",
        f"raw_equals_processed_exact,{int((df['TweetOriginal'].astype(str) == df['TweetProcess'].astype(str)).sum())}",
        f"clean_original_equals_clean_processed,{int((df['CleanOriginal'] == df['CleanProcessed']).sum())}",
        f"rows_with_any_clean_change,{int((df['CleanOriginal'] != df['CleanProcessed']).sum())}",
        f"all_change_records,{len(changes)}",
        f"one_to_one_replacement_instances,{len(one_to_one)}",
        f"broad_candidate_instances,{len(broad)}",
        f"broad_candidate_unique_rows,{broad['TweetId'].nunique()}",
        f"likely_real_word_instances,{len(likely)}",
        f"likely_real_word_unique_rows,{likely['TweetId'].nunique()}",
        f"likely_real_word_unique_pairs,{likely[['wrong_word', 'correct_word']].drop_duplicates().shape[0]}",
    ]
    (OUT_DIR / "summary.csv").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"wrote,{OUT_DIR}")


if __name__ == "__main__":
    main()
