# Trained checkpoints

GitHub rejects files over 100 MB in a normal commit, so the checkpoints are attached to a **GitHub Release** of this repository: **[TODO: https://github.com/&lt;your-username&gt;/voz-moderation-cascade/releases/tag/v1.0]**.

| File | Size | What it is |
|---|---|---|
| `phobert_cnn_1_recipe_best.pt` | ~530 MB | tier 2, PhoBERT-CNN, stage 1, seed 42 (test macro-F1 0.9369) |
| `phobert_cnn_1_recipe_calibration.json` | <1 KB | temperature 0.7374 and threshold (also in `results/cascade/`) |

Place them in this folder and load with:

```python
from tier2_classifier import Tier2Classifier
tier2 = Tier2Classifier("models/phobert_cnn_1_recipe_best.pt", vncorenlp_dir="vncorenlp")
tier2.load_calibration("models/phobert_cnn_1_recipe_calibration.json")
print(tier2.predict(["bình luận cần kiểm tra"]))
```

Tier 1 is small and fast to retrain (`src/train_tier1.py`, about 1 minute), which avoids scikit-learn version mismatches.

## Uploading (for the authors)

On GitHub: **Releases → Draft a new release** → tag `v1.0` → drag `phobert_cnn_1_recipe_best.pt`, `phobert_cnn_1_recipe_calibration.json` (and optionally `tier1_tfidf.joblib`) into the assets box → **Publish release**.
Each release asset may be up to 2 GB, so the 530 MB checkpoint fits; do not commit it to the repository itself.

With the GitHub CLI instead:

```bash
gh release create v1.0 phobert_cnn_1_recipe_best.pt phobert_cnn_1_recipe_calibration.json --title "v1.0 checkpoints"
```
