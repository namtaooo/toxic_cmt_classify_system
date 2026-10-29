"""Đánh giá cascade tầng 1 (TF-IDF) -> tầng 2 (PhoBERT-CNN) -> tầng 3 (Qwen) trên logits đã lưu.
Ngưỡng chỉ chọn trên VAL; test chỉ để đo. Chạy: python eval_cascade.py tier1_logits.npz tier2_logits.npz"""
import sys, json
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
assert (a["y_test"] == b["y_test"]).all() and (a["y_val"] == b["y_val"]).all()
yv, yt = a["y_val"], a["y_test"]
T2 = float(sys.argv[3]) if len(sys.argv) > 3 else None
LAT = {"t1_ms": 0.93, "t2_gpu_ms": 14.4, "t2_cpu_ms": 106.5}   # đo ở các notebook / train_tier1.py

def sm(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def fitT(z, y):
    f = lambda lt: -np.log(sm(z / np.exp(lt))[np.arange(len(y)), y] + 1e-12).mean()
    return float(np.exp(minimize_scalar(f, bounds=(-3, 3), method="bounded").x))
def mf1(y, p): return f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)
def pick_tau(conf, pred, y, mask, target):
    for t in np.round(np.arange(0.40, 0.9995, 0.005), 3):
        m = mask & (conf >= t)
        if m.sum() >= 20 and (pred[m] == y[m]).mean() >= target: return float(t)
    return None

T1 = fitT(a["val"], yv); T2 = T2 or fitT(b["val"], yv)
P = {k: (sm(a[k] / T1), sm(b[k] / T2)) for k in ("val", "test")}
c1v, k1v = P["val"][0].max(1), P["val"][0].argmax(1); c2v, k2v = P["val"][1].max(1), P["val"][1].argmax(1)
c1t, k1t = P["test"][0].max(1), P["test"][0].argmax(1); c2t, k2t = P["test"][1].max(1), P["test"][1].argmax(1)
N = len(yt)
print(f"Chỉ tầng 1: Macro-F1 {mf1(yt, k1t):.4f} | chỉ tầng 2: Macro-F1 {mf1(yt, k2t):.4f} | T1={T1:.3f} T2={T2:.3f}\n")

rows = []
for tg1 in (0.95, 0.97, 0.98):
    tau1 = pick_tau(c1v, k1v, yv, np.ones(len(yv), bool), tg1)
    d1v, d1t = c1v >= tau1, c1t >= tau1
    for tg2 in (0.95, 0.97):
        tau2 = pick_tau(c2v, k2v, yv, ~d1v, tg2)        # chọn trên phần val mà tầng 1 không tự quyết
        d2t = ~d1t & (c2t >= tau2); esc = ~d1t & ~d2t
        final = np.where(d1t, k1t, k2t)                  # chưa có Qwen: phần chuyển lên tầng 3 tạm lấy đáp án tầng 2
        n1, n2, n3 = d1t.sum(), d2t.sum(), esc.sum()
        corr1, corr2 = (k1t[d1t] == yt[d1t]).sum(), (k2t[d2t] == yt[d2t]).sum()
        r = {"target_t1": tg1, "tau1": tau1, "target_t2": tg2, "tau2": tau2,
             "share_t1": n1 / N, "share_t2": n2 / N, "share_t3": n3 / N,
             "acc_t1_decided": corr1 / n1, "acc_t2_decided": corr2 / max(n2, 1),
             "t2_acc_on_escalated": (k2t[esc] == yt[esc]).mean() if n3 else float("nan"),
             "macroF1_without_qwen": mf1(yt, final), "acc_without_qwen": (final == yt).mean(),
             "lat_gpu_ms_excl_qwen": LAT["t1_ms"] + (1 - n1 / N) * LAT["t2_gpu_ms"],
             "lat_cpu_ms_excl_qwen": LAT["t1_ms"] + (1 - n1 / N) * LAT["t2_cpu_ms"]}
        for q in (0.7, 0.8, 0.9): r[f"acc_if_qwen_{int(q*100)}"] = (corr1 + corr2 + q * n3) / N
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv("cascade_sweep.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
show = ["target_t1", "tau1", "target_t2", "tau2", "share_t1", "share_t2", "share_t3", "acc_t1_decided", "acc_t2_decided",
        "t2_acc_on_escalated", "macroF1_without_qwen", "acc_without_qwen", "lat_gpu_ms_excl_qwen", "lat_cpu_ms_excl_qwen"]
print(df[show].round(4).to_string(index=False))
print("\nĐộ chính xác toàn hệ thống nếu Qwen đúng q% trên phần bị chuyển lên:")
print(df[["target_t1", "target_t2", "share_t3", "acc_without_qwen", "acc_if_qwen_70", "acc_if_qwen_80", "acc_if_qwen_90"]].round(4).to_string(index=False))
