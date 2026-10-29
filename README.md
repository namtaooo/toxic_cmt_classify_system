# A Calibrated Three-Tier Cascade for Moderating Vietnamese Forum Comments

Code and results for the paper *"A Calibrated Three-Tier Cascade for Moderating Vietnamese Forum Comments with PhoBERT-CNN"*
by Pham Duc Hoai Nam, Ho Nguyen Bao Khanh and Nguyen Van Toan. [TODO: venue and link once published]

Comments from the Voz forum are classified as **clean**, **offensive** or **hate** by a cascade:

| Tier | Model | Decides when | Role |
|---|---|---|---|
| 1 | TF-IDF (word 1–2-grams + char 2–5-grams) + logistic regression | calibrated confidence ≥ 0.90 | fast filter, about 1 ms/comment on CPU |
| 2 | PhoBERT-base + TextCNN head (label smoothing, LLRD, R-Drop) | calibrated confidence ≥ 0.85 | main classifier |
| 3 | flagging | otherwise | blur the comment and queue it for human review |

Actions: clean → keep, offensive → blur, hate → hide.

## Main results (test set, 1,805 comments)

| System | Macro-F1 | Mean CPU latency |
|---|---|---|
| Tier 1 alone (TF-IDF + LR) | 0.8690 | 0.9 ms |
| Tier 2 alone (PhoBERT-CNN, 3 seeds) | 0.9367 ± 0.0012 | 97 ms |
| ViHateT5-base (3 seeds) | 0.9279 ± 0.0022 | 315 ms |
| ViSoBERT-CNN (3 seeds) | 0.9000 ± 0.0019 | 113 ms |
| **Three-tier cascade** | **0.9338** | **≈ 37–60 ms** (Kaggle CPU / laptop CPU) |

Tier 1 resolves 63.4% of comments at 96.7% accuracy; 7.0% are flagged for review. Qwen2.5-3B as tier 3 was correct on 40.9% of the
flagged comments versus 69.3% for tier 2, which is why tier 3 flags instead of classifying. All numbers can be checked against
the files in [`results/`](results/README.md).

## Repository layout

```
src/
  clean_data.py        text cleaning (also used at inference time)
  train_tier1.py       train + calibrate tier 1, writes tier1_tfidf.joblib
  tier1_classifier.py  tier-1 inference
  tier2_classifier.py  tier-2 inference (loads the PhoBERT-CNN checkpoint)
  eval_cascade.py      evaluate threshold configurations from saved logits
  demo_cascade.py      run the full cascade on text or a CSV file
  qwen_tier3.py        optional LLM tier 3 through Ollama or Hugging Face (experiment only)
notebooks/
  phobert_cnn_and_visobert.ipynb   tier-2 training, ablation stages 0-3, ViSoBERT-CNN (Kaggle, T4 GPU)
  vihatet5_finetune.ipynb          ViHateT5-base baseline
data/      anonymised train/val/test CSV files, dataset card and anonymisation script
models/    how to download the trained checkpoints
results/   per-run metrics, speed measurements, logits and the paper figure
```

Comments inside the notebooks are in Vietnamese.

## Setup

- Python 3.10+, `pip install -r requirements.txt`
- Java 8+ (VnCoreNLP word segmentation for PhoBERT)
- VnCoreNLP word segmenter in `./vncorenlp`:
  `VnCoreNLP-1.2.jar` and `models/wordsegmenter/{vi-vocab,wordsegmenter.rdr}` from the
  [VnCoreNLP repository](https://github.com/vncorenlp/VnCoreNLP) (on Windows, download the files manually; `py_vncorenlp.download_model` needs `wget`).

## Reproduce

1. **Data**: the anonymised splits are already in `data/` (see [`data/README.md`](data/README.md)). Links and usernames were masked in about 5% of comments, so models retrained on these files can differ slightly from the paper's numbers.
2. **Tier 1** (about 1 minute, CPU):
   ```bash
   python src/train_tier1.py --train data/train.csv --val data/val.csv --test data/test.csv --out tier1_out
   ```
3. **Tier 2**: run `notebooks/phobert_cnn_and_visobert.ipynb` on Kaggle with a GPU (`STAGE = "1_recipe"`, about 18 min per seed),
   (upload `data/` as a Kaggle dataset and set `DATA_DIR`; the notebook reads `train_clean.csv`, `val_clean.csv`, `test.csv`, so rename the files or edit `read_split`),
   or download the trained checkpoint from the GitHub Release (see [`models/README.md`](models/README.md)).
4. **Cascade from saved logits** (no GPU needed):
   ```bash
   python src/eval_cascade.py results/cascade/logits_tfidf.npz results/cascade/logits_phobert_cnn_1_recipe_seed42.npz 0.7374
   ```
5. **Full system on your machine** (CPU is enough):
   ```bash
   cd src
   python demo_cascade.py --tier1 ../tier1_out/tier1_tfidf.joblib --tier2 ../models/phobert_cnn_1_recipe_best.pt \
       --vncorenlp ../vncorenlp --csv ../data/test.csv --batch
   python demo_cascade.py --tier1 ../tier1_out/tier1_tfidf.joblib --tier2 ../models/phobert_cnn_1_recipe_best.pt \
       --vncorenlp ../vncorenlp --text "bài viết hay quá"
   ```
   `--batch` is fast but does not time each comment; drop it (and add `--limit 200`) to measure latency.

Retrain tier 1 with your own scikit-learn version rather than loading a `.joblib` saved with another version.

## Data, ethics and AI use

The dataset contains offensive language and was collected from a public forum. Released files must be anonymised with
`data/anonymize.py` (usernames and post links removed). Labels were pre-annotated by Gemini and verified by one author per class;
inter-annotator agreement was not measured. The authors also used an AI assistant to help write code and draft the paper.

## Licence

Code: MIT (see [`LICENSE`](LICENSE)). Data: CC BY-NC 4.0, research use only. Pre-trained models used here keep their own licences
(PhoBERT, ViSoBERT, ViHateT5, VnCoreNLP, Qwen2.5).

## Citation

```bibtex
@inproceedings{nam2026cascade,
  author    = {Pham, Duc Hoai Nam and Ho, Nguyen Bao Khanh and Nguyen, Van Toan},
  title     = {A Calibrated Three-Tier Cascade for Moderating Vietnamese Forum Comments with {PhoBERT-CNN}},
  booktitle = {[TODO: venue]},
  year      = {2026}
}
```
