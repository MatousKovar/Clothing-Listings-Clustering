# Phase 2 — v2 work

Self-contained folder for the post-baseline experiments. Reuses code from
`../src/` and read-only artifacts from `../artifacts/`. Nothing in `../phase1/`
is touched.

## Goal

Beat the baseline siamese (end-to-end clustering pairwise F1 ≈ 0.85 on the 30k
held-out val slice from `items_train`).

## Notebooks

| File | Purpose |
|---|---|
| `01_extract_fashion_clip.ipynb` | One-time: re-extract image embeddings using FashionCLIP (`patrickjohncyh/fashion-clip`, 512-d). Output: `embeddings/fashion_clip_image.pt`. |
| `02_train_siamese_v2.ipynb` | Approach 1: same SiameseNetwork architecture as baseline, but image-projection input dim 768→512. Re-mine pairs against new joint similarity. Train + checkpoint to `models/siamese_v2.pth`. |
| `03_train_metric_v2.ipynb` | Approach 2: drop the classifier head; train the embedder with `TripletMarginLoss` + semi-hard mining (`pytorch-metric-learning`). Output: `models/metric_v2.pth` + `embeddings/learned_embeddings.pt` for fast inference. |
| `04_cluster_and_eval.ipynb` | End-to-end pipeline + canonical eval (pairwise F1 on the 30k held-out items from items_train). Reports per-variant: pair F1, end-to-end F1, cross-geo subset F1. This is the only metric we trust. |

## Local artifacts (gitignored)

```
phase2/
├── embeddings/
│   ├── fashion_clip_image.pt        # produced by 01
│   └── learned_embeddings.pt         # produced by 03 (post-training inference dump)
└── models/
    ├── siamese_v2.pth                # produced by 02
    └── metric_v2.pth                 # produced by 03
```

## Reused (read-only) from elsewhere in the repo

- `../src/GlamiDatasetVocabulary.py` etc. — Python modules
- `../artifacts/embeddings/text_multilingual.pt` — multilingual text embeddings (kept)
- `../artifacts/vocabularies/` — vocab dicts
- `../artifacts/pipelines/preprocessing.pkl` — sklearn preprocessing pipeline
- `../artifacts/models/siamese_baseline.pth` — F1≈0.85 baseline; benchmark to beat
- `../data/items_train.csv` — labels for triplet training and held-out eval
- `../data/items_phase_2.csv` — final inference target (when released)

## Order of execution

1. `01_extract_fashion_clip.ipynb` — runs once, ~few hours on M4 MPS.
2. `02_train_siamese_v2.ipynb` — quick A/B vs baseline using the new embeddings.
3. `04_cluster_and_eval.ipynb` — measure end-to-end F1 of the v2 siamese.
4. `03_train_metric_v2.ipynb` — only if v2 siamese hasn't already cleared the bar.
5. `04_cluster_and_eval.ipynb` again — measure metric-learning variant.

Always report all three numbers (pair F1, end-to-end F1, cross-geo F1). The
`val_pairs_improved.csv` pair F1 is informational only — it doesn't predict
generalization (this is the lesson from the hardmined run that scored 0.97 pair
F1 but 0.0004 end-to-end).
