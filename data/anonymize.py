"""
Anonymise the cleaned Voz splits before sharing them.

What it does
  - drops the username (`author`) and every field that links back to a post or a user
    (`source_url`, `thread_key`, `text_hash`, `text_raw`, `quoted_raw`, `reaction_count`);
  - replaces each thread URL with an integer `thread_id` (consistent across the three splits),
    so comments can still be grouped by thread;
  - coarsens `posted_at` to the month (YYYY-MM);
  - replaces links in `text` and `quoted_text` with "<url>" (some point to user profiles) and masks @mentions and the usernames left in nested "<name> said:" quote headers as "@user".
Optionally (--keep-author-ids) it keeps a pseudonymous `author_id` derived from a secret salt
that you must NOT publish; without the salt the IDs cannot be mapped back to usernames.

Usage
  python data/anonymize.py --in-dir path/to/cleaned --out-dir data/release
  python data/anonymize.py --in-dir path/to/cleaned --out-dir data/release --keep-author-ids --salt "<secret>"

Input files expected in --in-dir: train_clean.csv, val_clean.csv, test.csv (or test_clean.csv).
"""
import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd

DROP = ["author", "source_url", "thread_key", "text_hash", "text_raw", "quoted_raw", "reaction_count"]
MENTION = re.compile(r"@[\w.\-]+", flags=re.UNICODE)
URL = re.compile(r"(?:https?://|www\.)\S+|\b(?:[\w-]+\.)+(?:vn|com|net|org|io|me)/\S*", flags=re.IGNORECASE)
SAID = re.compile(r"(\S+)(\s\S+)?\s+said:")


def _mask_said(m):
    """Nested quotes keep '<username> said:'. Mask the username (one token, or two when the first is capitalised)."""
    first, second = m.group(1), m.group(2)
    if second is None:
        return "@user said:"
    if first[:1].isupper():
        return "@user said:"
    return f"{first} @user said:"


SPLITS = {"train": ["train_clean.csv", "train.csv"], "val": ["val_clean.csv", "val.csv"],
          "test": ["test_clean.csv", "test.csv"]}


def find(in_dir, names):
    for n in names:
        if (in_dir / n).exists():
            return in_dir / n
    raise FileNotFoundError(f"none of {names} found in {in_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--keep-author-ids", action="store_true")
    ap.add_argument("--salt", default=None, help="secret salt for pseudonymous author IDs (never publish it)")
    a = ap.parse_args()
    if a.keep_author_ids and not a.salt:
        ap.error("--keep-author-ids needs --salt")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    frames = {s: pd.read_csv(find(a.in_dir, names)) for s, names in SPLITS.items()}
    threads = pd.concat([f["thread_key"] for f in frames.values() if "thread_key" in f]).dropna().unique()
    thread_id = {t: i for i, t in enumerate(sorted(threads))}

    for split, d in frames.items():
        d = d.copy()
        if "thread_key" in d:
            d.insert(0, "thread_id", d["thread_key"].map(thread_id).astype("Int64"))
        if a.keep_author_ids and "author" in d:
            d.insert(0, "author_id", d["author"].fillna("").map(
                lambda u: "u_" + hashlib.sha256((a.salt + u).encode("utf-8")).hexdigest()[:12]))
        if "posted_at" in d:
            d["posted_at"] = pd.to_datetime(d["posted_at"], errors="coerce", utc=True).dt.strftime("%Y-%m")
        for col in ("text", "quoted_text"):
            if col in d:
                d[col] = (d[col].fillna("").astype(str).str.replace(URL, "<url>", regex=True)
                          .str.replace(MENTION, "@user", regex=True)
                          .str.replace(SAID, _mask_said, regex=True))
        d = d.drop(columns=[c for c in DROP if c in d.columns])
        out = a.out_dir / f"{split}.csv"
        d.to_csv(out, index=False)
        print(f"{split}: {len(d)} rows -> {out} | columns: {list(d.columns)}")


if __name__ == "__main__":
    main()
