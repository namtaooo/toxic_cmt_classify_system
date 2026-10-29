"""
Demo hệ thống kiểm duyệt 3 tầng chạy ngay trên máy cá nhân (CPU là đủ).

Ví dụ:
  # thử vài bình luận
  python demo_cascade.py --tier2 phobert_cnn_1_recipe_best.pt --vncorenlp ./vncorenlp \
         --text "con khựa này ngu vl" --text "bài viết hay quá cảm ơn bạn"

  # chạy cả tập test, in bảng kết quả + lưu từng bình luận vào cascade_run.csv
  python demo_cascade.py --tier2 phobert_cnn_1_recipe_best.pt --vncorenlp ./vncorenlp --csv test.csv

  # chạy thử nhanh: 200 bình luận ngẫu nhiên, theo lô
  python demo_cascade.py --tier2 phobert_cnn_1_recipe_best.pt --vncorenlp ./vncorenlp --csv test.csv --limit 200 --batch

  # gõ bình luận tại chỗ để trình diễn
  python demo_cascade.py --tier2 phobert_cnn_1_recipe_best.pt --vncorenlp ./vncorenlp --interactive

Tầng 3 mặc định là cơ chế GẮN CỜ: bình luận mà tầng 2 không đủ tự tin được làm mờ và chờ người duyệt. Thử Qwen (không khuyến nghị,
trên dữ liệu của nhóm Qwen2.5-3B kém tầng 2) bằng  --qwen ollama:qwen2.5:3b  (xem qwen_tier3.py).
"""
import argparse
import json
import time

import numpy as np

from tier1_classifier import Tier1Classifier

LABELS = ["clean", "offensive", "hate"]
# Hành động cuối cùng trên extension (chỉnh theo đề tài của nhóm)
ACTION = {"clean": "KEEP (giữ)", "offensive": "BLUR (làm mờ)", "hate": "HIDE (ẩn)"}
# Tầng 3 = cơ chế GẮN CỜ: bình luận mà tầng 2 không đủ tự tin được làm mờ và chờ người duyệt (không đổi nhãn bằng mô hình khác)
FLAG_ACTION = "BLUR (gắn cờ, chờ duyệt)"


class Cascade:
    def __init__(self, tier1, tier2, tau1, tau2, qwen_fn=None):
        self.t1, self.t2, self.tau1, self.tau2, self.qwen_fn = tier1, tier2, tau1, tau2, qwen_fn

    def predict_one(self, text):
        t0 = time.perf_counter()
        r1 = self.t1.predict([text])[0]
        if r1["confidence"] >= self.tau1:
            return {"label": r1["label"], "tier": 1, "confidence": r1["confidence"], "uncertain": False,
                    "ms": (time.perf_counter() - t0) * 1000}
        r2 = self.t2.predict([text], quotes=[""])[0]
        if r2["confidence"] >= self.tau2:
            return {"label": r2["label"], "tier": 2, "confidence": r2["confidence"], "uncertain": False,
                    "ms": (time.perf_counter() - t0) * 1000}
        lab = self.qwen_fn(text) if self.qwen_fn is not None else None     # None: chưa nối Qwen hoặc Qwen trả lời không hợp lệ
        return {"label": lab or r2["label"], "tier": 3, "confidence": None if lab else r2["confidence"],
                "uncertain": lab is None, "t2_label": r2["label"], "ms": (time.perf_counter() - t0) * 1000}

def _predict_batch(casc, texts):
    """Chạy theo lô: tầng 1 cho tất cả, rồi tầng 2 cho phần cần lên. Nhanh hơn nhiều, nhưng không đo được ms từng bình luận."""
    r1 = casc.t1.predict(texts)
    esc = [i for i, r in enumerate(r1) if r["confidence"] < casc.tau1]
    r2 = casc.t2.predict([texts[i] for i in esc], quotes=[""] * len(esc)) if esc else []
    out = [{"label": r["label"], "tier": 1, "confidence": r["confidence"], "uncertain": False, "ms": float("nan")} for r in r1]
    for i, r in zip(esc, r2):
        ok = r["confidence"] >= casc.tau2
        out[i] = {"label": r["label"], "tier": 2 if ok else 3, "confidence": r["confidence"], "uncertain": not ok, "ms": float("nan")}
        if not ok:
            lab = casc.qwen_fn(texts[i]) if casc.qwen_fn is not None else None
            out[i].update({"t2_label": r["label"], "ms": float("nan")})
            if lab: out[i].update({"label": lab, "confidence": None, "uncertain": False})
    return out


def show(text, r):
    conf = "" if r["confidence"] is None else f" (tin cậy {r['confidence']:.2f})"
    note = f"  [gắn cờ: tầng 2 không chắc, nhãn tham khảo = {r['label']}]" if r["uncertain"] else ""
    action = FLAG_ACTION if r["uncertain"] else ACTION[r["label"]]
    print(f"- {text[:90]!r}\n    -> tầng {r['tier']}: {r['label']}{conf} => {action} | {r['ms']:.1f} ms{note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier1", default="tier1_tfidf.joblib")
    ap.add_argument("--tier2", required=True, help="file *_best.pt của PhoBERT-CNN")
    ap.add_argument("--vncorenlp", default="./vncorenlp", help="thư mục VnCoreNLP (cần Java)")
    ap.add_argument("--tau1", type=float, default=0.90)
    ap.add_argument("--tau2", type=float, default=0.85)
    ap.add_argument("--temp2", type=float, default=0.7374, help="temperature của tầng 2 (trong file *_calibration.json)")
    ap.add_argument("--text", action="append", default=[])
    ap.add_argument("--csv", help="file test có cột text và label")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--qwen", help="bật tầng 3: ollama:qwen2.5:3b  hoặc  hf:Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--limit", type=int, help="chỉ chạy ngẫu nhiên N bình luận của --csv (chạy thử cho nhanh)")
    ap.add_argument("--batch", action="store_true", help="chạy --csv theo lô: nhanh hơn nhiều, nhưng không đo ms từng bình luận")
    a = ap.parse_args()

    from tier2_classifier import Tier2Classifier           # nạp trễ để --help chạy không cần torch
    tier1 = Tier1Classifier(a.tier1)
    tier2 = Tier2Classifier(a.tier2, vncorenlp_dir=a.vncorenlp, device="cpu", temperature=a.temp2, threshold=a.tau2)
    qwen_fn = None
    if a.qwen:
        from qwen_tier3 import make_qwen
        qwen_fn = make_qwen(a.qwen)
        print("Tầng 3:", a.qwen)
    casc = Cascade(tier1, tier2, a.tau1, a.tau2, qwen_fn)
    casc.predict_one("khởi động")                           # làm nóng, không tính vào thời gian
    print(f"Đã nạp 2 tầng | tau1={a.tau1} tau2={a.tau2}\n")

    for t in a.text:
        show(t, casc.predict_one(t))

    if a.interactive:
        print("Gõ bình luận (để trống để thoát):")
        while True:
            t = input("> ").strip()
            if not t: break
            show(t, casc.predict_one(t))

    if a.csv:
        import pandas as pd
        from sklearn.metrics import f1_score, accuracy_score
        d = pd.read_csv(a.csv)
        if a.limit and a.limit < len(d): d = d.sample(a.limit, random_state=0).reset_index(drop=True)
        d["text"] = d["text"].fillna("").astype(str)
        y = d["label"].astype(int).values if pd.api.types.is_numeric_dtype(d["label"]) else \
            d["label"].astype(str).str.lower().map({l: i for i, l in enumerate(LABELS)}).values
        t_start = time.perf_counter()
        if a.batch:
            rows = _predict_batch(casc, d["text"].tolist())
            for r in rows: r["pred"] = LABELS.index(r["label"])
        else:
            rows = []
            for i, t in enumerate(d["text"]):
                r = casc.predict_one(t); r["pred"] = LABELS.index(r["label"]); rows.append(r)
                if (i + 1) % 50 == 0: print(f"  đã chạy {i + 1}/{len(d)} ({time.perf_counter() - t_start:.0f}s)", flush=True)
        print(f"Tổng thời gian chạy {len(d)} bình luận: {time.perf_counter() - t_start:.0f}s")
        R = pd.DataFrame(rows); R["text"] = d["text"].values; R["true"] = y
        R["true_label"] = [LABELS[k] for k in y]; R["flagged"] = (R["tier"] == 3) & R["uncertain"]
        R["action"] = np.where(R["flagged"], FLAG_ACTION, R["label"].map(ACTION))
        R.to_csv("cascade_run.csv", index=False, encoding="utf-8-sig")
        n = len(R)
        print("\n===== KẾT QUẢ CẢ HỆ THỐNG =====")
        print(f"Macro-F1 {f1_score(y, R['pred'], average='macro'):.4f} | Accuracy {accuracy_score(y, R['pred']):.4f} | n={n}")
        for t in (1, 2, 3):
            m = R["tier"] == t
            if m.any():
                print(f"Tầng {t}: {m.mean():.1%} bình luận | độ chính xác {(R.loc[m, 'pred'] == R.loc[m, 'true']).mean():.4f} | "
                      f"{R.loc[m, 'ms'].mean():.1f} ms/bình luận")
        # ---- hiệu quả của cơ chế gắn cờ ở mức HÀNH ĐỘNG (clean->giữ, offensive->làm mờ, hate->ẩn)
        fl = R["flagged"].values
        if fl.any():
            tr = R["true"].values
            print(f"\nGắn cờ tầng 3: {fl.sum()} bình luận ({fl.mean():.1%}) | nhãn thật trong nhóm này: " +
                  ", ".join(f"{l} {int((tr[fl] == i).sum())}" for i, l in enumerate(LABELS)))
            for name, act in (("Không gắn cờ (dùng nhãn tầng 2)", R["pred"].values), ("Có gắn cờ -> làm mờ", np.where(fl, 1, R["pred"].values))):
                bad_vis = int(((tr > 0) & (act == 0)).sum()); over = int(((tr == 0) & (act > 0)).sum())
                print(f"  {name:34s} đúng hành động {(act == tr).mean():.4f} | độc hại bị để lộ {bad_vis} ({bad_vis / max((tr > 0).sum(), 1):.1%}) "
                      f"| bình thường bị làm mờ/ẩn nhầm {over} ({over / max((tr == 0).sum(), 1):.1%})")
        if qwen_fn is not None and (R["tier"] == 3).any():
            m3 = R["tier"] == 3
            t2_idx = R.loc[m3, "t2_label"].map({l: i for i, l in enumerate(LABELS)})
            used = ~R.loc[m3, "uncertain"]
            print(f"\nSo sánh trên {int(m3.sum())} bình luận bị chuyển lên tầng 3 ({int(used.sum())} có Qwen trả lời hợp lệ):")
            print(f"  tầng 2 đúng {(t2_idx == R.loc[m3, 'true']).mean():.3f} | Qwen đúng {(R.loc[m3, 'pred'] == R.loc[m3, 'true']).mean():.3f}"
                  "  (Qwen chỉ có ích nếu cao hơn tầng 2)")
        if R["ms"].notna().any():
            print(f"Độ trễ trung bình toàn hệ thống: {R['ms'].mean():.1f} ms (p95 {R['ms'].quantile(.95):.1f} ms) trên CPU máy này")
        else:
            print("(chế độ --batch: không đo độ trễ từng bình luận; muốn đo thì chạy không có --batch)")
        print("Đã lưu chi tiết từng bình luận vào cascade_run.csv")


if __name__ == "__main__":
    main()
