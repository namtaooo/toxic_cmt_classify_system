# Trained checkpoints

The tier-2 model is hosted on Hugging Face: **<https://huggingface.co/namcb1622/phobert_cnn_toxic_cmt>**.

| File | Size | What it is |
|---|---|---|
| `phobert_cnn_1_recipe_best.pt` | ~530 MB | tier 2, PhoBERT-CNN, stage 1, seed 42 (test macro-F1 0.9369) |
| `phobert_cnn_1_recipe_calibration.json` | <1 KB | temperature 0.7374 and threshold (also in `results/cascade/`) |

Download and load (run from the repository root):

```python
import sys
from huggingface_hub import hf_hub_download
sys.path.append("src")
from tier2_classifier import Tier2Classifier

repo = "namcb1622/phobert_cnn_toxic_cmt"
ckpt = hf_hub_download(repo, "phobert_cnn_1_recipe_best.pt")
calib = hf_hub_download(repo, "phobert_cnn_1_recipe_calibration.json")

tier2 = Tier2Classifier(ckpt, vncorenlp_dir="vncorenlp")
tier2.load_calibration(calib)
print(tier2.predict(["bình luận cần kiểm tra"]))
```

Or from the command line: `huggingface-cli download namcb1622/phobert_cnn_toxic_cmt --local-dir models`.

If the files on Hugging Face have other names, change the file names above accordingly.

Tier 1 is small and fast to retrain (`src/train_tier1.py`, about 1 minute), which avoids scikit-learn version mismatches.

`MODEL_CARD.md` in this folder is the description to paste into the Hugging Face repository's README.
