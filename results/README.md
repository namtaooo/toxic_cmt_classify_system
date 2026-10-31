# Results

Every number in the paper comes from a file in this folder.

| Paper item | File |
|---|---|
| Table II, PhoBERT-CNN row; Table III (ablation stages 0–3) | `tier2_ablation/results_all_stages.csv` (one row per seed), `summary_all_stages.csv` |
| Table II, ViHateT5 and ViSoBERT-CNN rows | `baselines/vihatet5/results.csv`, `baselines/visobert_cnn/results.csv` |
| Table II, tier-1 row | `tier1/results_tfidf.csv` (grid in `grid_search.csv`, top n-grams in `top_features.csv`) |
| Table IV (component ablation and reconstruction of Tran et al.) | `tier2_components/results_*.csv` (one row per seed; tags `..._rdrop_alpha0.0`, `..._llrd1.0`, `..._last_k1`, `..._max_len20`, `phobert_cnn_tran_original`, `phobert_cnn_tran_original_max_len160`); the deployed row is `phobert_cnn_1_recipe` in `tier2_ablation/results_all_stages.csv` |
| Table V (speed and size) | `tier2_ablation/speed_all_stages.csv`, `baselines/*/speed.json`, `tier1/speed_tfidf.json` |
| Table VI (threshold configurations) | `cascade/cascade_sweep.csv`, reproducible with `src/eval_cascade.py` and the two `logits_*.npz` files |
| Limitations bullet on ViHSD | `vihsd/results_*.csv` (not in the main tables; see below) |
| Fig. 2 | `figure/f1_vs_latency.pdf`, drawn by `python results/figure/make_figure.py` (run from the repo root) |
| Deployed tier-2 checkpoint (seed 42) | `cascade/results_phobert_cnn_1_recipe.csv`, `cascade/speed_phobert_cnn_1_recipe.json`, `cascade/phobert_cnn_1_recipe_calibration.json` |

`logits_*.npz` hold validation and test logits plus gold labels (`val`, `test`, `y_val`, `y_test`); labels are 0 = clean, 1 = offensive, 2 = hate.

Latency in `cascade_sweep.csv` uses the per-tier timings measured when it was produced (tier 2 on CPU: 106.5 ms); the paper's Table VI
uses the three-seed timing (97.4 ms), so CPU latency values differ slightly. Macro-F1 and stopping rates are identical.

Measurements: training and GPU timings on Kaggle (NVIDIA Tesla T4, 4-thread CPU); laptop run on an Intel Core i5-12500H, 16 GB RAM, CPU only.

## Component ablation (`tier2_components/`)

Each file name carries the tag of the run. `1_recipe` with an override means the deployed configuration with one change; `tran_original` is our reconstruction of PhoBERT-CNN by Tran et al. (PhoBERT-large, kernels 1/2/3/5 x 32, last layer only, Adam 2e-5, constant lr, 20 tokens, batch 64), and `tran_original_max_len160` the same with 160 tokens and batch 16. The reconstruction has no data augmentation and uses our choices for dropout, epochs and temperature scaling. Differences from the deployed configuration are not significant except for the 20-token runs (Welch's t-test, three seeds, a rough guide only). Label smoothing was not ablated separately.

## ViHSD (`vihsd/`)

The deployed configuration and the reconstruction were also trained on the public ViHSD benchmark (train/dev/test split converted with `src/convert_vihsd.py`, no other change, three seeds, model chosen on dev, reported on test). Macro-F1: deployed 0.6556 +- 0.0099, reconstruction 0.6238 +- 0.0204 (20 tokens) and 0.6311 +- 0.0078 (160 tokens, batch 16), below the 0.6746 and 0.6867 reported in earlier work. These results are mentioned only as a limitation in the paper; they are kept here for transparency. ViHSD itself is not redistributed in this repository.
