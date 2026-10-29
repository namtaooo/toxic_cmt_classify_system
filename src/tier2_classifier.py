"""
Tầng 2 của hệ thống kiểm duyệt: nạp checkpoint từ notebook phobert-cnn-improved.ipynb.
Hỗ trợ mọi BACKBONE của notebook (phobert_cnn, phobert_cls, visobert_cls, visobert_cnn)
và checkpoint PhoBERT-CNN cũ (không có khoá "backbone").

Luồng:  Tầng 1 (TF-IDF) không đủ tự tin  ->  Tầng 2
        p2 = max softmax(logits / T)  >= tau2  -> quyết định luôn
        ngược lại                              -> chuyển lên Tầng 3 (Qwen)

Cách dùng:
    from tier2_classifier import Tier2Classifier
    tier2 = Tier2Classifier("phobert_cnn_1_recipe_best.pt", vncorenlp_dir="/path/vncorenlp")   # ViSoBERT không cần vncorenlp_dir
    tier2.load_calibration("phobert_cnn_1_recipe_calibration.json")     # T và tau2 đã chọn trên val trong notebook
    out = tier2.predict(["bình luận cần kiểm tra"], quotes=[""])

Bình luận mới được làm sạch bằng đúng các hàm của clean_data.py (đặt file này cạnh clean_data.py),
để đầu vào khớp với dữ liệu lúc huấn luyện.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

try:
    from clean_data import norm_unicode, strip_signature, strip_quote
except ImportError:  # vẫn chạy được nếu thiếu clean_data.py, nhưng nên có để khớp lúc train
    norm_unicode = strip_signature = strip_quote = lambda s: s.strip()

LABELS = ["clean", "offensive", "hate"]


def _hf_load(fn, name, **kw):
    """Thử tải bình thường; nếu mạng lỗi thì dùng bản đã lưu cache trên máy (chạy offline khi trình diễn)."""
    try:
        return fn(name, **kw)
    except Exception as e:                                  # lỗi kết nối tới huggingface.co
        try:
            return fn(name, local_files_only=True, **kw)
        except Exception:
            raise RuntimeError(f"Không tải được {name} và chưa có bản trong cache. Cần có mạng ít nhất một lần. Lỗi gốc: {e}") from e


class TextClassifier(nn.Module):
    """Giống hệt kiến trúc trong notebook để nạp được state_dict."""

    def __init__(self, model_name, head="cnn", num_labels=3, kernel_sizes=(2, 3, 4),
                 num_filters=128, dropout=0.3, last_k=4):
        super().__init__()
        self.head, self.last_k = head, last_k
        cfg = _hf_load(AutoConfig.from_pretrained, model_name, output_hidden_states=(head == "cnn"))
        self.encoder = AutoModel.from_config(cfg)   # trọng số lấy từ checkpoint, không tải lại pretrained
        H = cfg.hidden_size
        self.drop = nn.Dropout(dropout)
        if head == "cnn":
            self.convs = nn.ModuleList([nn.Conv1d(H * last_k, num_filters, k, padding=k // 2) for k in kernel_sizes])
            self.fc = nn.Linear(num_filters * len(kernel_sizes), num_labels)
        else:
            self.fc = nn.Linear(H, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.head == "cls":
            return self.fc(self.drop(out.last_hidden_state[:, 0]))
        x = torch.cat(out.hidden_states[-self.last_k:], dim=-1)
        x = (x * attention_mask.unsqueeze(-1).to(x.dtype)).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(x))[:, :, : x.size(2)]
            c = c.masked_fill(attention_mask.unsqueeze(1) == 0, -1e4)
            pooled.append(c.max(dim=2).values)
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


class Tier2Classifier:
    def __init__(self, ckpt_path, vncorenlp_dir=None, tokenizer_dir=None, device=None,
                 temperature=1.0, threshold=0.90, batch_size=32):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.backbone = ck.get("backbone", "phobert_cnn")
        self.head = ck.get("head", "cnn")
        self.needs_seg = ck.get("needs_seg", True)
        self.max_len = ck["max_len"]
        self.input_mode = ck.get("input_mode", "text_only")
        self.model = TextClassifier(ck["model_name"], self.head, 3, tuple(ck["kernel_sizes"]),
                                    ck["num_filters"], ck["dropout"], ck["last_k"])
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.to(self.device).eval()
        self.tokenizer = _hf_load(AutoTokenizer.from_pretrained, tokenizer_dir or ck["model_name"], use_fast=False)
        self.temperature, self.threshold, self.batch_size = float(temperature), float(threshold), batch_size

        self.segmenter = None
        if self.needs_seg:
            if not vncorenlp_dir:
                raise ValueError(f"{self.backbone} được huấn luyện trên văn bản đã tách từ: cần truyền vncorenlp_dir")
            import os, py_vncorenlp
            cwd = os.getcwd()
            self.segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=os.path.abspath(vncorenlp_dir))
            os.chdir(cwd)

    # ---------- tiền xử lý: giống clean_data.py + notebook ----------
    def _clean_text(self, s):
        return strip_signature(norm_unicode("" if s is None else str(s)))

    def _clean_quote(self, q):
        q = "" if q is None else str(q)
        return strip_quote(strip_signature(norm_unicode(q))) if q.strip() else ""

    def _seg(self, s):
        if not s or self.segmenter is None:
            return s
        return " ".join(self.segmenter.word_segment(s))

    def _encode(self, t, q):
        tok = self.tokenizer
        if self.input_mode == "text_quote" and q.strip():
            trunc = "only_second" if len(tok.tokenize(t)) < self.max_len - 40 else "longest_first"
            return tok(t, q, max_length=self.max_len, truncation=trunc)["input_ids"]
        return tok(t, max_length=self.max_len, truncation=True)["input_ids"]

    def _batch(self, texts, quotes):
        ids = [self._encode(self._seg(self._clean_text(t)), self._seg(self._clean_quote(q)))
               for t, q in zip(texts, quotes)]
        L = max(len(x) for x in ids)
        inp = torch.full((len(ids), L), self.tokenizer.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(ids), L), dtype=torch.long)
        for i, x in enumerate(ids):
            inp[i, :len(x)] = torch.tensor(x); am[i, :len(x)] = 1
        return inp, am

    # ---------- suy luận ----------
    @torch.no_grad()
    def logits(self, texts, quotes=None):
        quotes = quotes if quotes is not None else [""] * len(texts)
        out = []
        for i in range(0, len(texts), self.batch_size):
            ids, am = self._batch(texts[i:i + self.batch_size], quotes[i:i + self.batch_size])
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.device.type == "cuda"):
                out.append(self.model(ids.to(self.device), am.to(self.device)).float().cpu())
        return torch.cat(out)

    def predict(self, texts, quotes=None):
        probs = torch.softmax(self.logits(texts, quotes) / self.temperature, dim=-1).numpy()
        conf, pred = probs.max(1), probs.argmax(1)
        return [{"label": LABELS[p], "confidence": float(c), "probs": pr.round(4).tolist(),
                 "decided": bool(c >= self.threshold),           # False -> đẩy lên Qwen
                 "next_tier": None if c >= self.threshold else "tier3_qwen"}
                for p, c, pr in zip(pred, conf, probs)]

    # ---------- hiệu chỉnh (notebook đã làm sẵn; dùng khi muốn chọn lại trên dữ liệu khác) ----------
    def fit_temperature(self, texts, labels, quotes=None):
        lg = self.logits(texts, quotes); y = torch.tensor(labels, dtype=torch.long)
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=200); nll = nn.CrossEntropyLoss()
        def closure():
            opt.zero_grad(); loss = nll(lg / log_t.exp(), y); loss.backward(); return loss
        opt.step(closure)
        self.temperature = float(log_t.exp()); self._val_cache = (lg, y)
        return self.temperature

    def choose_threshold(self, texts, labels, quotes=None, target_acc=0.97):
        lg, y = getattr(self, "_val_cache", (None, None))
        if lg is None:
            lg, y = self.logits(texts, quotes), torch.tensor(labels)
        conf, pred = torch.softmax(lg / self.temperature, -1).max(1)
        rows, chosen = [], None
        for tau in np.round(np.arange(0.50, 0.996, 0.005), 3):
            m = conf >= tau
            cov = m.float().mean().item()
            acc = (pred[m] == y[m]).float().mean().item() if m.any() else float("nan")
            rows.append({"tau": float(tau), "coverage": cov, "acc_decided": acc})
            if chosen is None and m.sum() >= 20 and acc >= target_acc:
                chosen = float(tau)
        self.threshold = chosen if chosen is not None else 0.99
        return self.threshold, rows

    def save_calibration(self, path):
        json.dump({"temperature": self.temperature, "threshold": self.threshold}, open(path, "w"), indent=2)

    def load_calibration(self, path):
        d = json.load(open(path)); self.temperature, self.threshold = d["temperature"], d["threshold"]
