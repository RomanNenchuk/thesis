# Legacy

Files no longer referenced by `notebooks/eda.ipynb`. Kept for history, safe to delete.

| Path | What it was |
|---|---|
| `notebooks/eda_legacy.ipynb` | Earlier version of the EDA notebook |
| `scripts/prev_eda.py` | Even earlier EDA as a script |
| `docs/context.legacy.md` | Previous version of the thesis plan (see `docs/context.md` for the current one) |
| `data/df_sample_10k.csv` | 10k development sample from the first iteration |
| `data/df_sample_100k.parquet` | Older parquet copy of the 100k sample (the notebook uses the CSV) |
| `data/_test.parquet` | Scratch test file |
| `artifacts/embeddings_10k.npy` | MiniLM embeddings for the 10k sample |
| `artifacts/kmeans_k50_100k.joblib` | Early fitted KMeans (k=50) |
| `artifacts/umap_reducer_100k.joblib` | Early fitted UMAP reducer (547 MB) |
| `kaggle/kernel-metadata.json` | Config of the Kaggle GPU smoke-test kernel |

> Large binaries (`*.npy`, `*.joblib`) are not tracked in git (GitHub 100 MB limit); they stay on the local disk only.

## Added 2026-11-19 (spatial dataset search, Kaggle spatial dead end)

| Path | What it was |
|---|---|
| `new-dataset-exploring/` | Candidate-dataset checks (Flickr-Geo, Singapore tweets, Yelp) and the geotweet extraction script. Yelp was picked; the rest are fallbacks only |
| `notebooks/eda_kaggle_spatial_slice.ipynb` | Cells cut from `eda.ipynb`: H3 spatial slice on the Kaggle data that ended in the "dead end" (place-level coordinates only) |
| `data/spatial_pool.csv` | Spatial pool built from the Kaggle data for that slice (109 MB, untracked) |
| `results/metrics_kaggle.csv` | Metrics log of the Kaggle text-pipeline experiments (149 runs × 2 subsets, 2026-10-25 to 2026-11-08). `eda.ipynb` creates a fresh `results/metrics.csv` on the next `log_run()` |
| `data/gold/` | Kaggle 3000-post LLM gold set (gold_annotated.csv, gold_sample.csv with dev/test split) and the boilerplate precision check |
| `data/sample_kaggle/` | Fixed 100k Kaggle post sample (`df_sample_100k.csv`) |
| `artifacts/embeddings_kaggle/` | MiniLM embeddings (100k) and cached UMAP projections for the Kaggle runs (git-ignored) |
| `notebooks/eda_kaggle.ipynb` | The finished Kaggle text-pipeline study (former `notebooks/eda.ipynb`): sampling, cleaning, embeddings, gold set, `evaluate()`/`log_run()`, UMAP + KMeans/HDBSCAN. **Reference for the Yelp pipeline**; its paths point at the old locations, so it does not run as is |
