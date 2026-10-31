"""Đổi ViHSD (train.csv, dev.csv, test.csv; cột free_text, label_id) sang định dạng notebook đọc.

Dùng:  python convert_vihsd.py --in-dir vihsd_raw --out-dir vihsd_clean
Ra:    train_clean.csv, val_clean.csv, test_clean.csv với cột text, quoted_text, label (clean/offensive/hate)

Chỉ chuẩn hoá Unicode (NFC) và cắt khoảng trắng đầu/cuối. KHÔNG bỏ dòng ngắn, không khử trùng lặp:
tập test chính thức phải giữ nguyên để so sánh được với các bài đã công bố.
Nhãn theo README của ViHSD: 0 = CLEAN, 1 = OFFENSIVE, 2 = HATE (hãy đối chiếu lại với trang gốc).
"""
import argparse
import unicodedata
from pathlib import Path

import pandas as pd

LABELS = {0: "clean", 1: "offensive", 2: "hate"}
FILES = {"train": ["train.csv"], "val": ["dev.csv", "val.csv"], "test": ["test.csv"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    seen = {}
    for split, names in FILES.items():
        src = next((a.in_dir / n for n in names if (a.in_dir / n).exists()), None)
        if src is None:
            print(f"[bỏ qua] không thấy {names} trong {a.in_dir}")
            continue
        d = pd.read_csv(src)
        if not {"free_text", "label_id"} <= set(d.columns):
            raise SystemExit(f"{src}: cần cột free_text và label_id, đang có {list(d.columns)}")
        bad = ~d["label_id"].isin(LABELS)
        if bad.any():
            raise SystemExit(f"{src}: nhãn lạ {d.loc[bad, 'label_id'].unique()[:5]}")
        out = pd.DataFrame({
            "text": d["free_text"].fillna("").astype(str).map(lambda s: unicodedata.normalize("NFC", s).strip()),
            "quoted_text": "",
            "label": d["label_id"].map(LABELS),
        })
        out.to_csv(a.out_dir / f"{split}_clean.csv", index=False)
        seen[split] = out
        vc = out["label"].value_counts()
        print(f"{split:5s} n={len(out):6d} | clean={vc.get('clean', 0)} offensive={vc.get('offensive', 0)} hate={vc.get('hate', 0)}"
              f" | rỗng={int((out['text'] == '').sum())}")
    if {"train", "test"} <= set(seen):
        overlap = set(seen["train"]["text"]) & set(seen["test"]["text"])
        print(f"Câu giống hệt nhau giữa train và test: {len(overlap)} (chỉ để biết, không bị loại)")


if __name__ == "__main__":
    main()
