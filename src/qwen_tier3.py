"""
Tầng 3: dùng mô hình Qwen (chạy cục bộ) để phân loại những bình luận mà tầng 2 không đủ tự tin.

Hai cách chạy Qwen, chọn một:
  1) Ollama (dễ nhất trên Windows, chạy được cả CPU):
        cài Ollama (https://ollama.com), rồi:  ollama pull qwen2.5:3b
        dùng:  --qwen ollama:qwen2.5:3b
  2) Hugging Face transformers (cần tải mô hình, nên có GPU với mô hình từ 3B trở lên):
        dùng:  --qwen hf:Qwen/Qwen2.5-1.5B-Instruct

Mỗi bộ trả về hàm  qwen_fn(text) -> "clean" | "offensive" | "hate" | None  (None = Qwen trả lời không hợp lệ).
Tầng 3 chỉ nhận khoảng 7% bình luận nên chậm hơn một chút cũng chấp nhận được.
"""
import json
import urllib.request

LABELS = ["clean", "offensive", "hate"]

SYSTEM = """Bạn là hệ thống kiểm duyệt bình luận diễn đàn tiếng Việt (Voz). Phân loại MỖI bình luận vào đúng 1 trong 3 nhãn:

- hate: lăng mạ, miệt thị hoặc kích động chống lại MỘT NHÓM người (theo quốc tịch, sắc tộc, vùng miền, tôn giáo, giới tính, xu hướng tính dục, quan điểm chính trị...), kể cả khi dùng các từ lóng miệt thị nhóm.
- offensive: chửi thề, xúc phạm, công kích cá nhân hoặc sản phẩm/sự vật, nhưng KHÔNG nhắm vào một nhóm người.
- clean: bình thường, không xúc phạm ai. Lưu ý các từ lóng như "vl", "vãi" dùng để nhấn mạnh (ví dụ khen) vẫn là clean.

Chỉ trả lời đúng MỘT từ: clean, offensive hoặc hate. Không giải thích."""

FEWSHOT = [
    ("cảm ơn bác, bài viết hữu ích quá", "clean"),
    ("con game này đồ hoạ đẹp vl", "clean"),
    ("thằng này ngu vl, nói như cc", "offensive"),
    ("cái hãng này làm ăn như cứt", "offensive"),
    ("bọn khựa lúc nào cũng hàng fake", "hate"),
    ("lũ bê đê chỉ biết làm hại xã hội", "hate"),
]


def build_messages(text):
    msgs = [{"role": "system", "content": SYSTEM}]
    for t, lab in FEWSHOT:
        msgs += [{"role": "user", "content": t}, {"role": "assistant", "content": lab}]
    msgs.append({"role": "user", "content": text})
    return msgs


def parse_label(s):
    s = (s or "").lower()
    hits = [(s.find(l), l) for l in LABELS if l in s]
    return min(hits)[1] if hits else None            # nhãn xuất hiện sớm nhất


class QwenOllama:
    def __init__(self, model="qwen2.5:3b", host="http://localhost:11434", timeout=120):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout

    def __call__(self, text):
        body = {"model": self.model, "messages": build_messages(text), "stream": False,
                "options": {"temperature": 0, "num_predict": 8}}
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            self.last_raw = json.loads(r.read())["message"]["content"]
        return parse_label(self.last_raw)


class QwenHF:
    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct", device=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32).to(self.device).eval()

    def __call__(self, text):
        prompt = self.tok.apply_chat_template(build_messages(text), tokenize=False, add_generation_prompt=True)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=6, do_sample=False)
        self.last_raw = self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_label(self.last_raw)


def make_qwen(spec):
    kind, _, name = spec.partition(":")
    if kind == "ollama":
        return QwenOllama(name or "qwen2.5:3b")
    if kind == "hf":
        return QwenHF(name or "Qwen/Qwen2.5-1.5B-Instruct")
    raise ValueError("--qwen phải có dạng ollama:<tên mô hình> hoặc hf:<id Hugging Face>")
