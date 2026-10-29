"""
Tầng 1 của hệ thống kiểm duyệt: TF-IDF + hồi quy logistic (sinh bởi train_tier1.py).

Luồng:  bình luận  ->  Tầng 1
        p1 = max softmax(logits / T) >= tau1  -> quyết định luôn
        ngược lại                             -> chuyển lên Tầng 2 (PhoBERT-CNN)

Cách dùng:
    from tier1_classifier import Tier1Classifier
    tier1 = Tier1Classifier("tier1_tfidf.joblib")
    out = tier1.predict(["bình luận cần kiểm tra"])

Đặt cạnh clean_data.py để bình luận mới được làm sạch đúng như lúc huấn luyện.
"""
import joblib
import numpy as np
from scipy.sparse import hstack

try:
    from clean_data import norm_unicode, strip_signature
except ImportError:  # vẫn chạy được nhưng nên có clean_data.py để khớp lúc train
    norm_unicode = strip_signature = lambda s: s.strip()

LABELS = ["clean", "offensive", "hate"]


class Tier1Classifier:
    def __init__(self, path, temperature=None, threshold=None):
        d = joblib.load(path)
        self.vectorizers, self.clf = d["vectorizers"], d["clf"]
        self.temperature = float(d["temperature"] if temperature is None else temperature)
        self.threshold = float(d["threshold"] if threshold is None else threshold)

    @staticmethod
    def _clean(s):
        return strip_signature(norm_unicode("" if s is None else str(s)))

    def logits(self, texts):
        texts = [self._clean(t) for t in texts]
        X = hstack([v.transform(texts) for _, v in self.vectorizers]).tocsr()
        return self.clf.decision_function(X)

    def predict(self, texts, quotes=None):          # quotes bị bỏ qua: tầng 1 chỉ dùng text
        z = self.logits(texts) / self.temperature
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        conf, pred = p.max(1), p.argmax(1)
        return [{"label": LABELS[k], "confidence": float(c), "probs": pr.round(4).tolist(),
                 "decided": bool(c >= self.threshold),
                 "next_tier": None if c >= self.threshold else "tier2_phobert_cnn"}
                for k, c, pr in zip(pred, conf, p)]
