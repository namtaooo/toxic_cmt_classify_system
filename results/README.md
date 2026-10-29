# Results

Every number in the paper comes from a file in this folder.

| Paper item | File |
|---|---|
| Table II, PhoBERT-CNN row; Table III (ablation stages 0–3) | `tier2_ablation/results_all_stages.csv` (one row per seed), `summary_all_stages.csv` |
| Table II, ViHateT5 and ViSoBERT-CNN rows | `baselines/vihatet5/results.csv`, `baselines/visobert_cnn/results.csv` |
| Table II, tier-1 row | `tier1/results_tfidf.csv` (grid in `grid_search.csv`, top n-grams in `top_features.csv`) |
| Table IV (speed and size) | `tier2_ablation/speed_all_stages.csv`, `baselines/*/speed.json`, `tier1/speed_tfidf.json` |
| Table V (threshold configurations) | `cascade/cascade_sweep.csv`, reproducible with `src/eval_cascade.py` and the two `logits_*.npz` files |
| Fig. 2 | `figure/f1_vs_latency.pdf`, drawn by `python results/figure/make_figure.py` (run from the repo root) |
| Deployed tier-2 checkpoint (seed 42) | `cascade/results_phobert_cnn_1_recipe.csv`, `cascade/speed_phobert_cnn_1_recipe.json`, `cascade/phobert_cnn_1_recipe_calibration.json` |

`logits_*.npz` hold validation and test logits plus gold labels (`val`, `test`, `y_val`, `y_test`); labels are 0 = clean, 1 = offensive, 2 = hate.

Latency in `cascade_sweep.csv` uses the per-tier timings measured when it was produced (tier 2 on CPU: 106.5 ms); the paper's Table V
uses the three-seed timing (97.4 ms), so CPU latency values differ slightly. Macro-F1 and stopping rates are identical.

Measurements: training and GPU timings on Kaggle (NVIDIA Tesla T4, 4-thread CPU); laptop run on an Intel Core i5-12500H, 16 GB RAM, CPU only.
