---
language: vi
license: mit
tags:
  - text-classification
  - hate-speech-detection
  - vietnamese
  - phobert
base_model: vinai/phobert-base
pipeline_tag: text-classification
---

# PhoBERT-CNN for Vietnamese toxic comment classification (clean / offensive / hate)

Tier 2 of the three-tier moderation cascade described in *"A Calibrated Three-Tier Cascade for Moderating Vietnamese Forum Comments with PhoBERT-CNN"*
(Pham Duc Hoai Nam, Ho Nguyen Bao Khanh, Nguyen Van Toan). Code and data: <https://github.com/namtaooo/toxic_cmt_classify_system>.

> **Content warning:** the model was trained on offensive and hateful comments.

## Model

The architecture follows PhoBERT-CNN by Tran et al., *Neural Computing and Applications* 35(1), 2023 ([doi:10.1007/s00521-022-07745-w](https://doi.org/10.1007/s00521-022-07745-w)), with the changes below.


- Encoder: [vinai/phobert-base](https://huggingface.co/vinai/phobert-base); input word-segmented with VnCoreNLP, truncated to 160 tokens.
- Head: last 4 hidden layers concatenated → TextCNN (kernels 2, 3, 4; 128 filters each) → masked max-pooling → dropout 0.3 → linear (3 classes).
- Training: label smoothing 0.1, layer-wise learning-rate decay 0.9, R-Drop (α = 1), AdamW, 5 epochs max with early stopping, seed 42.
- Calibration: temperature 0.7374 (see `phobert_cnn_1_recipe_calibration.json`).
- Labels: 0 = clean, 1 = offensive, 2 = hate. Input: the comment text only (no quoted text).

This is a plain PyTorch checkpoint (`.pt`), not a `transformers` model; load it with `Tier2Classifier` from the GitHub repository.

## Results (test set, 1,805 comments)

| Metric | Value |
|---|---|
| Macro-F1 (this checkpoint) | 0.9369 |
| Macro-F1 (mean ± std over 3 seeds) | 0.9367 ± 0.0012 |
| ECE after temperature scaling | about 0.013 |
| Latency per comment | 14.5 ms (Tesla T4 GPU), about 100 ms (CPU) |

## Data

18,046 public comments from the Voz forum (posted March 2020 to September 2026), pre-labelled by Gemini 3.1 Pro and verified by one author per class;
class ratio about 50/30/20 by design. Inter-annotator agreement was not measured. See the dataset card in the GitHub repository.

## Limitations and intended use

- Trained on a single forum; performance on other platforms is unknown.
- The test set has far more hate than real traffic, so precision on hate may be lower in deployment.
- Intended for research and for assisting human moderators, not for fully automatic decisions about people.

## Usage

```python
from huggingface_hub import hf_hub_download
from tier2_classifier import Tier2Classifier   # src/ of the GitHub repository

ckpt = hf_hub_download("namcb1622/phobert_cnn_toxic_cmt", "phobert_cnn_1_recipe_best.pt")
model = Tier2Classifier(ckpt, vncorenlp_dir="vncorenlp")
print(model.predict(["bài viết hay quá"]))
```
