"""
Tầng 1: TF-IDF + mô hình tuyến tính cho hệ thống kiểm duyệt bình luận 3 tầng.

Chạy:
    python train_tier1.py --train train_clean.csv --val val_clean.csv --test test.csv --out tier1_out
    (thêm --quick để thử nhanh với lưới nhỏ hơn)

Quy trình (giống các notebook PhoBERT / ViHateT5 để so sánh công bằng):
  1. Đọc dữ liệu đã làm sạch (cùng 3 file với tầng 2).
  2. Tìm lưới: bộ đặc trưng x mô hình x siêu tham số, CHỌN THEO Macro-F1 TRÊN VAL (không nhìn test).
  3. Hiệu chỉnh: fit temperature trên val, chọn ngưỡng tau1 trên val cho độ chính xác mục tiêu.
  4. Đo trên test: Macro-F1, F1 từng lớp, ECE, coverage tại độ chính xác mục tiêu, tốc độ trên CPU.
  5. Lưu: tier1_tfidf.joblib (dùng với tier1_classifier.py), results_tfidf.csv, speed_tfidf.json,
     logits_tfidf.npz, grid_search.csv, top_features.csv, calibration.json.
Tầng 1 chỉ dùng text (không dùng quote) vì extension khó lấy quote.
"""
import argparse
import json
import os
import platform
import time

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

LABELS = ["clean", "offensive", "hate"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}


# ---------------------------------------------------------------- dữ liệu
def read(path):
    d = pd.read_csv(path)
    d["text"] = d["text"].fillna("").astype(str)
    d["quoted_text"] = d["quoted_text"].fillna("").astype(str) if "quoted_text" in d.columns else ""
    lab = d["label"]
    y = lab.astype(int) if pd.api.types.is_numeric_dtype(lab) else \
        lab.astype(str).str.strip().str.lower().map(LABEL_TO_ID)
    if y.isna().any():
        raise ValueError(f"{path}: có nhãn lạ ngoài {LABELS}")
    d["y"] = y.astype(int)
    return d


# ---------------------------------------------------------------- đặc trưng
def make_vectorizers(kind):
    """Chỉ dùng lớp có sẵn của sklearn để file joblib nạp được mà không cần mã tuỳ biến."""
    vs = [
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True,
                                 token_pattern=r"(?u)\b\w+\b", max_features=200000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, lowercase=True,
                                 max_features=300000)),
    ]
    if kind == "noaccent":
        # thêm n-gram ký tự đã bỏ dấu: bắt người viết không dấu ("deo", "dm", "ngu vl"...)
        vs.append(("char_noaccent", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                                    lowercase=True, strip_accents="unicode", max_features=150000)))
    return vs


def fit_features(vectorizers, texts):
    return hstack([v.fit_transform(texts) for _, v in vectorizers]).tocsr()


def transform(vectorizers, texts):
    return hstack([v.transform(texts) for _, v in vectorizers]).tocsr()


# ---------------------------------------------------------------- chỉ số (cùng định nghĩa với notebook)
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y):
    def nll_(logT):
        p = softmax(logits / np.exp(logT))
        return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    return float(np.exp(minimize_scalar(nll_, bounds=(-3, 3), method="bounded").x))


def ece(probs, y, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def nll(probs, y):
    return float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())


def macro_f1(y, p):
    return f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)


def coverage_at_acc(p_val, yv, p_test, yt, target):
    """τ nhỏ nhất trên VAL sao cho độ chính xác phần tự quyết >= target, rồi đo trên TEST."""
    cv, pv = p_val.max(1), p_val.argmax(1)
    tau = None
    for t in np.round(np.arange(0.40, 0.9995, 0.005), 3):
        m = cv >= t
        if m.sum() >= 20 and (pv[m] == yv[m]).mean() >= target:
            tau = float(t)
            break
    if tau is None:
        return {"tau": None, "coverage": 0.0, "acc_decided": float("nan")}
    ct, pt = p_test.max(1), p_test.argmax(1)
    m = ct >= tau
    return {"tau": tau, "coverage": float(m.mean()),
            "acc_decided": float((pt[m] == yt[m]).mean()) if m.any() else float("nan")}


# ---------------------------------------------------------------- chính
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-acc", type=float, default=0.97)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tr, va, te = read(a.train), read(a.val), read(a.test)
    y_tr, y_va, y_te = tr["y"].values, va["y"].values, te["y"].values
    print(f"train {len(tr)} | val {len(va)} | test {len(te)}")

    kinds = ["base"] if a.quick else ["base", "noaccent"]
    models = ([("lr", {"C": 10})] if a.quick else
              [("lr", {"C": 5}), ("lr", {"C": 20}), ("svm", {"C": 0.1}), ("svm", {"C": 0.5})])

    def build(name, hp):
        if name == "lr":
            return LogisticRegression(C=hp["C"], max_iter=3000, solver="lbfgs")
        return LinearSVC(C=hp["C"], max_iter=5000)

    grid, best = [], None
    for kind in kinds:
        vecs = make_vectorizers(kind)
        t0 = time.time()
        X_tr = fit_features(vecs, tr["text"])
        X_va = transform(vecs, va["text"])
        t_feat = time.time() - t0
        print(f"\n[đặc trưng {kind}] {X_tr.shape[1]:,} chiều, {t_feat:.0f}s")
        for name, hp in models:
            t0 = time.time()
            clf = build(name, hp).fit(X_tr, y_tr)
            lv = clf.decision_function(X_va)
            f1 = macro_f1(y_va, lv.argmax(1))
            row = {"features": kind, "model": name, "C": hp["C"], "val_macro_f1": f1,
                   "n_features": X_tr.shape[1], "fit_seconds": time.time() - t0}
            grid.append(row)
            print(f"  {name:3s} C={hp['C']:<5} val Macro-F1 = {f1:.4f}  ({row['fit_seconds']:.0f}s)")
            if best is None or f1 > best["f1"]:
                best = {"f1": f1, "kind": kind, "name": name, "hp": hp, "vecs": vecs, "clf": clf, "t_train": t_feat + row["fit_seconds"]}
    pd.DataFrame(grid).to_csv(os.path.join(a.out, "grid_search.csv"), index=False)
    print(f"\n>>> Chọn trên val: features={best['kind']} model={best['name']} {best['hp']} (val Macro-F1 {best['f1']:.4f})")

    vecs, clf = best["vecs"], best["clf"]
    lv = clf.decision_function(transform(vecs, va["text"]))
    lt = clf.decision_function(transform(vecs, te["text"]))

    # --- hiệu chỉnh trên val, đo trên test
    T = fit_temperature(lv, y_va)
    p_raw, p_cal, pv_cal = softmax(lt), softmax(lt / T), softmax(lv / T)
    pred = lt.argmax(1)
    pc = f1_score(y_te, pred, labels=[0, 1, 2], average=None, zero_division=0)
    has_q = (te["quoted_text"].str.strip() != "").values
    targets = [0.95, 0.97, 0.98, 0.99]
    cov = {t: coverage_at_acc(pv_cal, y_va, p_cal, y_te, t) for t in targets}
    main_cov = cov[a.target_acc] if a.target_acc in cov else coverage_at_acc(pv_cal, y_va, p_cal, y_te, a.target_acc)

    res = {"stage": "tier1", "tag": f"tfidf_{best['name']}", "seed": 0, "backbone": "tfidf",
           "features": best["kind"], "model": best["name"], "C": best["hp"]["C"],
           "val_macro_f1": best["f1"], "test_macro_f1": macro_f1(y_te, pred), "test_acc": accuracy_score(y_te, pred),
           "f1_clean": pc[0], "f1_offensive": pc[1], "f1_hate": pc[2],
           "macro_f1_has_quote": macro_f1(y_te[has_q], pred[has_q]) if has_q.any() else None,
           "macro_f1_no_quote": macro_f1(y_te[~has_q], pred[~has_q]) if (~has_q).any() else None,
           "temperature": T, "ece_raw": ece(p_raw, y_te), "ece_cal": ece(p_cal, y_te),
           "nll_raw": nll(p_raw, y_te), "nll_cal": nll(p_cal, y_te),
           "tau1": main_cov["tau"], "coverage_tier1": main_cov["coverage"], "acc_decided_tier1": main_cov["acc_decided"],
           # tên cột trùng với notebook tầng 2 để bảng so sánh gộp được
           "tau2": main_cov["tau"], "coverage_tier2": main_cov["coverage"], "acc_decided_tier2": main_cov["acc_decided"],
           "train_minutes": best["t_train"] / 60}
    pd.DataFrame([res]).to_csv(os.path.join(a.out, "results_tfidf.csv"), index=False)

    print("\n=== Kết quả trên TEST ===")
    print(f"Macro-F1 {res['test_macro_f1']:.4f} | acc {res['test_acc']:.4f} | "
          f"F1 clean/offensive/hate = {pc[0]:.3f}/{pc[1]:.3f}/{pc[2]:.3f}")
    print(f"T={T:.3f} | ECE {res['ece_raw']:.4f} -> {res['ece_cal']:.4f}")
    print("Ngưỡng chọn trên val -> kết quả trên test (tầng 1 chỉ tự quyết khi tin cậy >= tau1):")
    for t in targets:
        c = cov[t]
        print(f"  độ chính xác mục tiêu {t:.2f}: tau1={c['tau']} | coverage={c['coverage']:.1%} | acc phần tự quyết={c['acc_decided']:.4f}")

    # --- tốc độ trên CPU (cả chuỗi: vector hoá + suy luận)
    items = te["text"].tolist()
    lat = []
    for k, s in enumerate(items[:310]):
        t = time.perf_counter()
        clf.decision_function(transform(vecs, [s])).argmax(1)
        if k >= 10:
            lat.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    clf.decision_function(transform(vecs, items))
    thr = len(items) / (time.perf_counter() - t)

    # --- lưu mô hình
    path = os.path.join(a.out, "tier1_tfidf.joblib")
    joblib.dump({"vectorizers": vecs, "clf": clf, "temperature": T, "threshold": main_cov["tau"] if main_cov["tau"] is not None else 0.99,
                 "labels": LABELS, "target_acc": a.target_acc}, path, compress=3)
    json.dump({"temperature": T, "threshold": main_cov["tau"] if main_cov["tau"] is not None else 0.99},
              open(os.path.join(a.out, "calibration.json"), "w"), indent=2)
    nnz = int((np.abs(clf.coef_) > 1e-12).sum())
    speed = {"tag": res["tag"], "backbone": "tfidf", "head": "linear", "needs_seg": False, "input_mode": "text_only",
             "cpu": f"{platform.processor() or platform.machine()} x{os.cpu_count()} threads",
             "params_million": clf.coef_.size / 1e6, "nonzero_params": nnz, "ckpt_mb": os.path.getsize(path) / 2**20,
             "cpu_b1_dynamic_ms": float(np.mean(lat)), "cpu_b1_dynamic_p95_ms": float(np.percentile(lat, 95)),
             "cpu_e2e_segment_ms": 0.0, "cpu_e2e_total_ms": float(np.mean(lat)), "cpu_e2e_total_p95_ms": float(np.percentile(lat, 95)),
             "cpu_batch_per_s": float(thr), "train_seconds": best["t_train"]}
    json.dump(speed, open(os.path.join(a.out, "speed_tfidf.json"), "w"), indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(a.out, "logits_tfidf.npz"), val=lv, test=lt, y_val=y_va, y_test=y_te)
    print(f"\nTốc độ CPU: {speed['cpu_b1_dynamic_ms']:.2f} ms/bình luận (p95 {speed['cpu_b1_dynamic_p95_ms']:.2f}), "
          f"{thr:,.0f} bình luận/s theo lô | file mô hình {speed['ckpt_mb']:.1f} MB")

    # --- n-gram nặng nhất mỗi lớp (để giải thích trong báo cáo)
    names = np.concatenate([[f"{n}:{f}" for f in v.get_feature_names_out()] for n, v in vecs])
    rows = []
    for ci, lab in enumerate(LABELS):
        w = clf.coef_[ci]
        for j in np.argsort(-w)[:30]:
            rows.append({"class": lab, "feature": names[j], "weight": float(w[j])})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "top_features.csv"), index=False, encoding="utf-8-sig")
    print("\nN-gram nặng nhất (10 đầu mỗi lớp):")
    for lab in LABELS:
        print(f"  {lab:9s}", ", ".join([r["feature"] for r in rows if r["class"] == lab][:10]))


if __name__ == "__main__":
    main()
