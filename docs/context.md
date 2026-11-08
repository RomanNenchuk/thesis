# LinkUp thesis: revised plan (as of 2026-11-08)

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
  - HDBSCAN scored a higher NMI (0.38–0.49) but discards about 50% of posts.
- **Step 1, fair KMeans vs HDBSCAN (done):**
  - On the posts HDBSCAN keeps, KMeans with the same k scores the same (NMI gap ≤ 0.023). The earlier HDBSCAN advantage was a selection effect: the kept posts are simply easier.
  - On all gold posts (HDBSCAN noise assigned to the nearest clustered neighbours), the methods are equal in NMI (gap ≤ 0.014). HDBSCAN has a higher ARI only at the finest partitions (k = 181 and 96), where ARI is very low.
  - HDBSCAN marks non-topical posts as noise somewhat more often than topical ones (about +0.05 to +0.12; for UMAP 5D, nn 50: 0.47–0.56 vs 0.58–0.63). It still discards about half of the topical posts, so it is a weak junk filter (roughly four fifths of its noisy gold posts are topical, rough estimate).
  - Verdict: the two methods are tied on topic quality, so the choice rests on other criteria (see section 2).

---

## 2. What changed in the plan

- **Topic quality has a ceiling.** The gold topics are fuzzy, and NMI differences under about 0.01 on roughly 1.7k dev posts are noise. Don't chase NMI.
- **UMAP tuning is closed.**
- **KMeans vs HDBSCAN is tied on quality, so the primary clusterer is not fixed yet.** Both stay as candidates until the spatial slice (step 2).
  - HDBSCAN: no preset k (k is unknown in advance), can reject posts without a topic; but it discards about half of the posts (mostly topical ones), `min_cluster_size` still has to be chosen, and a way to assign a topic to a new post must be checked.
  - KMeans: covers every post and has a simple inference path; but k is arbitrary (no best k in the sweep) and it tends to produce clusters of similar size.
  - Current preference: HDBSCAN, with noise posts assigned to the nearest cluster when full coverage is needed.
- **The spatial part hasn't started,** although the thesis title is about it. A thin spatial slice moves earlier.
- **Dropped or demoted:** Optuna, the 10k/50k/100k size ablation, the word-count threshold check, BERTopic and spaCy (stretch at most), joint text+coordinates clustering (stretch).

---

## 3. Next steps

### Step 1. Fair KMeans vs HDBSCAN: DONE
Results are in section 1. The primary clusterer is decided after step 2.

### Step 2. Spatial feasibility slice (1–2 days)
- Choose 2–3 dense cities and build an H3 grid (resolution 8–9).
- On one city, run location quotient (LQ) and a permutation test. Run it with both label sets (KMeans at k ≈ 30–50 and HDBSCAN with noise assigned), and compare which one gives more meaningful topic concentrations.
- Check how many posts fall in each cell and whether any signal appears.
- The 100k sample is global and probably too sparse per city. If so, embed all posts from the chosen cities and assign topics with `UMAP.transform` plus the fitted clusterer.
- Pick the primary clusterer after this step.

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
- **Optional: non-topical filter.** HDBSCAN noise is a weak detector of posts without a topic, so test a dedicated one instead:
  - logistic regression on MiniLM embeddings, topical vs non-topical (`reaction_chat`, `unclear_other`), 5-fold CV on the dev split;
  - report AUC, precision and recall (only about 340 non-topical dev posts, so the estimate is noisy; the boundary between the classes is fuzzy);
  - if it works, filter non-topical posts first and cluster the rest, then compare with the unfiltered pipeline;
  - small and cheap, so it can be skipped if time is short. Do not touch the `test` split.

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

Write Data, Methods and Evaluation now, and Results as steps 1–6 finish. Step 1 already gives a result section: the fair comparison and the noise analysis.

## 5. Minimum viable thesis

Steps 1–3, a simplified step 6 (one city, LQ + permutation test), step 7 and a short integration.

## 6. Limitations to state in the text

- LLM gold labels were not manually verified, only smoke test; the 16-topic taxonomy is fuzzy.
- The boilerplate regex filter was not checked for precision; unflagged near-duplicates remain (roughly 1% of the sample, rough estimate from a small manual look).
- Dev split only (about 1.7k topical posts, about 340 non-topical), one UMAP seed per configuration, one KMeans seed per grid cell.
- The noise analysis reuses the same dev posts across all UMAP/HDBSCAN settings, so its rows are not independent evidence.
- Data are 2021 tweets, not LinkUp posts (domain shift).
- Language entropy is normalised per cluster and capped by the dataset's language imbalance.