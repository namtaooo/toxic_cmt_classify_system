"""
Làm sạch dữ liệu bình luận Voz cho hệ thống kiểm duyệt 3 tầng.

Cách chạy:
    python clean_data.py --train train.csv --val val.csv --test test.csv --out out_dir

Đầu ra trong out_dir:
    train_clean.csv, val_clean.csv, test_clean.csv   (giữ text_raw / quoted_raw để đối chiếu)
    cleaning_report.json                              (số liệu trước/sau từng bước)
    cleaning_steps.csv                                (số dòng bị tác động theo bước và theo tập)

Thứ tự bước (mỗi bước đều ghi số liệu để vẽ biểu đồ trước/sau):
    1. Chuẩn hoá Unicode NFC, xoá ký tự vô hình, gộp khoảng trắng
    2. Cắt chữ ký thiết bị cuối bình luận và cuối quote
    3. Bỏ tiền tố "<username> said:" và "Click to expand..." khỏi quoted_text
    4. Xoá dòng có text rỗng hoặc dưới MIN_CHARS ký tự
    5. Xoá bản trùng sau khi chuẩn hoá chữ hoa/thường và dấu câu (giữ bản ở test > val > train)
    6. Bỏ cột rỗng (post_id)
Không cắt độ dài: bộ tokenizer của mô hình tự cắt theo max_len.
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

MIN_CHARS = 3
LONG_CHARS = 1000

ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
# Chữ ký thiết bị của Voz: "via theNEXTvoz for iPhone", "Sent from Xiaomi ... using vozFApp"
SIGNATURE = re.compile(
    r"\s*(?:"
    r"via\s+(?:the)?next\s*voz(?:\s+for\s+\w+)?"
    r"|sent\s+from\s+.{0,60}?\s*(?:via|using)\s+(?:next\s*voz|vozfapp|vozvnapp)"
    r"|g[ửu]i\s+(?:t[ừu]\s+.{0,60}?\s*)?b[ằa]ng\s+(?:next\s*voz|vozfapp|vozvnapp)"
    r"|sent\s+from\s*"
    r")\s*$",
    flags=re.IGNORECASE,
)
SAID_PREFIX = re.compile(r"^\s*[^\n:]{1,60}?\s+said:\s*", flags=re.IGNORECASE)
CLICK_EXPAND = re.compile(r"\s*Click to expand\.{0,3}\s*", flags=re.IGNORECASE)


def norm_unicode(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = ZERO_WIDTH.sub("", s)
    return re.sub(r"[ \t\r\f\v]+", " ", s).replace(" ", " ").strip()


def strip_signature(s: str) -> str:
    return SIGNATURE.sub("", s).strip()


def strip_quote(q: str) -> str:
    q = SAID_PREFIX.sub("", q, count=1)
    q = CLICK_EXPAND.sub(" ", q)
    return q.strip()


def dedup_key(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def noise_flags(df: pd.DataFrame, text_col: str, quote_col: str) -> dict:
    t = df[text_col].fillna("")
    q = df[quote_col].dropna()
    return {
        "rows": int(len(df)),
        "text_empty_or_short": int((t.str.strip().str.len() < MIN_CHARS).sum()),
        "device_signature": int(t.str.contains(SIGNATURE).sum()),
        "zero_width_char": int(t.str.contains(ZERO_WIDTH).sum()),
        "quote_said_prefix": int(q.str.contains(SAID_PREFIX).sum()),
        "quote_total": int(len(q)),
        "text_over_long": int((t.str.len() > LONG_CHARS).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = {"train": pd.read_csv(a.train), "val": pd.read_csv(a.val), "test": pd.read_csv(a.test)}
    report = {"before": {}, "after": {}, "steps": {}}
    steps = []

    for name, df in splits.items():
        df["text_raw"] = df["text"]
        df["quoted_raw"] = df["quoted_text"]
        report["before"][name] = noise_flags(df, "text", "quoted_text")
        report["before"][name]["label_counts"] = df["label"].value_counts().to_dict()
        report["before"][name]["text_len"] = df["text"].fillna("").str.len().tolist()

    # Các bước 1-3: biến đổi từng dòng
    for name, df in splits.items():
        t0 = df["text"].fillna("")
        t1 = t0.map(norm_unicode)
        steps.append((name, "Chuẩn hoá Unicode, xoá ký tự vô hình, gộp khoảng trắng", int((t0 != t1).sum())))
        t2 = t1.map(strip_signature)
        steps.append((name, "Cắt chữ ký thiết bị cuối bình luận", int((t1 != t2).sum())))
        df["text"] = t2

        q0 = df["quoted_text"]
        q1 = q0.map(lambda s: strip_signature(norm_unicode(s)) if isinstance(s, str) else s)
        q2 = q1.map(lambda s: strip_quote(s) if isinstance(s, str) else s)
        q2 = q2.where(q2.notna() & (q2.astype(str).str.strip() != ""), None)
        steps.append((name, "Bỏ tiền tố 'said:' / 'Click to expand' trong quote",
                      int(((q0 != q2) & q0.notna()).sum())))
        df["quoted_text"] = q2
        report["steps"][name] = {"quote_emptied": int((q1.notna() & q2.isna()).sum())}

    # Bước 4: xoá dòng rỗng/ngắn
    for name, df in splits.items():
        bad = df["text"].fillna("").str.strip().str.len() < MIN_CHARS
        steps.append((name, f"Xoá dòng text rỗng hoặc dưới {MIN_CHARS} ký tự", int(bad.sum())))
        splits[name] = df[~bad].copy()

    # Bước 5: xoá bản trùng, ưu tiên giữ test > val > train
    seen = set()
    for name in ["test", "val", "train"]:
        df = splits[name]
        keys = df["text"].map(dedup_key)
        dup = keys.isin(seen) | keys.duplicated(keep="first")
        dup &= keys.str.len() > 3
        seen |= set(keys)
        steps.append((name, "Xoá bản trùng sau chuẩn hoá", int(dup.sum())))
        splits[name] = df[~dup].copy()

    # Bước 6: bỏ cột rỗng
    for name, df in splits.items():
        empty_cols = [c for c in df.columns if df[c].isna().all()]
        df.drop(columns=empty_cols, inplace=True)
        report["steps"].setdefault(name, {})["dropped_columns"] = empty_cols

    for name, df in splits.items():
        df.to_csv(out / f"{name}_clean.csv", index=False)
        report["after"][name] = noise_flags(df, "text", "quoted_text")
        report["after"][name]["label_counts"] = df["label"].value_counts().to_dict()
        report["after"][name]["text_len"] = df["text"].str.len().tolist()
        report["after"][name]["rows_with_quote"] = int(df["quoted_text"].notna().sum())

    pd.DataFrame(steps, columns=["split", "step", "rows_affected"]).to_csv(
        out / "cleaning_steps.csv", index=False, encoding="utf-8-sig")
    (out / "cleaning_report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    print(pd.DataFrame(steps, columns=["split", "step", "rows_affected"])
          .pivot(index="step", columns="split", values="rows_affected").to_string())
    for name in splits:
        print(name, report["before"][name]["rows"], "->", report["after"][name]["rows"],
              report["after"][name]["label_counts"])


if __name__ == "__main__":
    main()
