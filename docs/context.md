# LinkUp thesis: revised plan (as of 2026-11-01)

Project: bachelor thesis (Computer Science, Ivan Franko National University of Lviv), built on the LinkUp app (social network with geo-tagged posts).
Title: *"Research of methods and tools for intelligent analysis of spatially distributed data in social services"*.
Core: Scenario 2 (spatial market analytics: topic clusters and where they concentrate geographically). Scenario 1 (personalised local search with LLM summaries) is a light demo on top.
Data: Kaggle `yachaymarketing/text-coordinates-regions` (multilingual geo-tagged tweets, 2021).

---

## 1. Done so far

- **Data:** a fixed 100k sample with `post_id`, raw and cleaned text, `lang`, a boilerplate flag (flagged, not deleted), an exact-duplicate key and MiniLM embeddings (`paraphrase-multilingual-MiniLM-L12-v2`). Near-duplicates were checked and deliberately not filtered; this goes into the limitations.
- **Gold set:** 3000 LLM-labelled posts (a second LLM reviewed them), 16 topics, dev/test split. The `test` split is untouched. Two labels are non-topical (`reaction_chat`, `unclear_other`, about 17%), so every result is reported on `all` and `topical_only`. Manual label verification (Cohen's kappa) and the boilerplate precision check were skipped on purpose; both are limitations.
- **Evaluation:** `evaluate()` reports ARI, NMI, homogeneity, completeness and language entropy. Every run is logged to `results/metrics.csv`. Reference points:
  - random labels: NMI ≈ 0.06;
  - supervised logistic regression on the embeddings: NMI ≈ 0.32, accuracy 0.54.
- **Experiments:**
  - MiniLM + UMAP + KMeans is stable at NMI ≈ 0.29–0.31 and clusters follow topic rather than language.
  - UMAP `n_components` (5/15/30) and `n_neighbors` (15/30/50) barely matter, so the config is fixed at **5D, nn 50**.
  - HDBSCAN scores a higher NMI (0.38–0.49) but discards about 50% of posts, so it is not comparable with KMeans yet.

---

## 2. What changed in the plan

- **Topic quality has a ceiling.** The gold topics are fuzzy, and NMI differences under about 0.01 on roughly 1.7k dev posts are noise. Don't chase NMI.
- **UMAP tuning is closed.**
- **HDBSCAN noise is the main open question,** and it matters for the spatial part too.
- **The spatial part hasn't started,** although the thesis title is about it. A thin spatial slice moves earlier.
- **Dropped or demoted:** Optuna, the 10k/50k/100k size ablation, the word-count threshold check, BERTopic and spaCy (stretch at most), joint text+coordinates clustering (stretch).

---

## 3. Next steps

### Step 1. Fair KMeans vs HDBSCAN (1–2 days)
- Score KMeans on exactly the posts HDBSCAN keeps.
- Add an HDBSCAN variant that assigns noise posts to the nearest cluster, so both are scored on all gold posts.
- Compare at similar cluster counts (about 30–50).
- Pick the primary clusterer for everything after this.

### Step 2. Spatial feasibility slice (1–2 days)
- Choose 2–3 dense cities and build an H3 grid (resolution 8–9).
- On one city, run location quotient (LQ) and a permutation test with the current KMeans labels.
- Check how many posts fall in each cell and whether any signal appears.
- The 100k sample is global and probably too sparse per city. If so, embed all posts from the chosen cities and assign topics with `UMAP.transform` plus the fitted clusterer.

### Step 3. Representation comparison (3–5 days)
- Compare TF-IDF+SVD, fastText and multilingual-e5-base against MiniLM through the same pipeline (UMAP 5D/nn50, then the chosen clusterer).
- Use 3–5 seeds for the final configs; report language entropy and stability.
- Add c-TF-IDF top words, NPMI coherence and topic diversity.
- Expect fastText to cluster by language; report that as a result.
- Quickly check that the UMAP settings transfer to the winner, since they were tuned only on MiniLM.

### Step 4. Decision point: depth vs breadth
By now the spatial slice shows whether the spatial result is interesting.

### Step 5. Small ablations
- With vs without boilerplate.
- With vs without UMAP.
- Name the best clusters (LLM plus a manual check).

### Step 6. Full spatial part
- Compare naive density, LQ and Getis-Ord Gi* hotspots.
- Use permutation tests with Benjamini–Hochberg correction.
- Include the existing medoid clustering from the app as a baseline.
- Produce maps and a topic × district table. OSM validation is stretch.

### Step 7. Final test evaluation
Run the 1–2 chosen configs on the `test` split once. Don't touch it before this.

### Step 8. Scenario 1 demo and integration
- Short demo: radius query, topics, LLM summary of each cluster.
- Integration: FastAPI microservice, a `TopicId` column in PostgreSQL, inference path encoder → `UMAP.transform` → predict.
- If `UMAP.transform` is too slow, distill the clustering into a lightweight classifier.

---

## 4. Writing

Write Data, Methods and Evaluation now, and Results as steps 1–6 finish.

## 5. Minimum viable thesis

Steps 1–3, a simplified step 6 (one city, LQ + permutation test), step 7 and a short integration.

## 6. Limitations to state in the text

- LLM gold labels were not manually verified, only smoke test; the 16-topic taxonomy is fuzzy.
- The boilerplate regex filter was not checked for precision; unflagged near-duplicates remain (roughly 1% of the sample, rough estimate from a small manual look).
- Dev split only (about 1.7k topical posts), one UMAP seed and one KMeans seed per grid cell.
- Data are 2021 tweets, not LinkUp posts (domain shift).
- Language entropy is normalised per cluster and capped by the dataset's language imbalance.