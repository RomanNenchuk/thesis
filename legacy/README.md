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
