# LinkUp thesis: restart plan (as of 2026-11-19)

Project: bachelor thesis (Computer Science, Ivan Franko National University of Lviv), built on the LinkUp app (social network with geo-tagged posts).
Title: *"Research of methods and tools for intelligent analysis of spatially distributed data in social services"*.
Core: Scenario 2 (spatial market analytics: topic clusters and where they concentrate geographically). Scenario 1 (personalised local search with LLM summaries) is a light demo on top.

---

## 1. Where we are

- The Kaggle `yachaymarketing/text-coordinates-regions` dataset has only place-level coordinates (~11.6k distinct pairs for ~425k posts), so district-level spatial analysis is impossible on it. All Kaggle work (data, gold set, embeddings, metrics, notebooks) is archived in `legacy/`; see `legacy/README.md`.
- We restart on one dataset for the whole pipeline: the **Yelp Open Dataset**, tips + business coordinates (908,915 tips; 150,346 businesses; main metro Philadelphia ~158k tips). **PROVISIONAL**, pending the smoke test. Reasons: the text is demand signal tied to commercial venues (matches the site-selection scenario), venue categories give a validation signal, the licence is clear for academic use.
- Fallbacks, in order: Google Local reviews (UCSD), Flickr-Geo (HF `do-me/Flickr-Geo`; captions, per-photo rights unclear), Singapore tweets (HF `cestwc/SG-subzone-poi-sentiment-estate`; no licence/provenance).
- No precise open Twitter data: X terms forbid redistributing raw tweets, so open sets have only IDs (rehydration needed), per-user/city-level coordinates, or no text; Archive Team Twitter Stream items are access-restricted (HTTP 401).

**Scientific claim to keep:** not "topics concentrate where their venues are", but "after controlling for venue mix (category-adjusted LQ, permutations stratified by category, cap per business) some cross-category themes still concentrate in specific districts".

**Yelp licence:** academic use only; no public display of raw tips (aggregates and paraphrases only); results must be submitted to Yelp before public presentation (the thesis defence counts as public). Never commit Yelp data to git.

## 2. What carries over from the Kaggle study

Method and lessons, not results (the numbers below were measured on Kaggle tweets and may not hold on Yelp). The reference notebook is `legacy/notebooks/eda_kaggle.ipynb`.

- Pipeline: MiniLM embeddings, UMAP 5D (nn 50; `n_components` and `n_neighbors` barely mattered), then KMeans or HDBSCAN. Evaluation with ARI, NMI, homogeneity, completeness and language entropy via `evaluate()` / `log_run()`.
- KMeans vs HDBSCAN were tied on topic quality. HDBSCAN's higher NMI came from discarding about half of the posts (a selection effect). The primary clusterer was never fixed.
- Gold-label topics are fuzzy, and small NMI differences are noise, so don't chase NMI.
- Kaggle limitations (LLM labels not manually verified, one UMAP seed, boilerplate filter precision unchecked) are archived with that study. The Yelp study needs its own gold set and its own limitations list.

## 3. Next steps

### Step 1. Yelp smoke test
Run `notebooks/yelp_smoke.ipynb`; the verdict is GO / GO WITH CONDITIONS / NO-GO, with the report in `reports/yelp_smoke_report.md`. On NO-GO, move to the next fallback.

### Step 2. Spatial pool and exploration
After a GO, build the spatial pool for Philadelphia and explore the new data in the way `eda_kaggle.ipynb` did (cleaning, flags, density per H3 cell). Plan details depend on the smoke report.

### Step 3. Topic pipeline on Yelp
Representations (MiniLM first, others later), UMAP, KMeans/HDBSCAN, a new gold set (a Yelp-specific taxonomy; the 16 Kaggle topics do not transfer), evaluated through `evaluate()` and logged to `results/metrics.csv`. The primary clusterer is decided here.

### Step 4. Spatial analysis
Category-adjusted location quotient, permutation tests stratified by venue category with a cap per business, multiple-testing correction, maps and a topic × district table. Venue categories serve as validation.

### Step 5. Final evaluation, Scenario 1 demo, integration
Evaluate the chosen config(s) once on a held-out test split. Short demo: radius query, topics, LLM summary per cluster. Integration into LinkUp: inference path encoder → `UMAP.transform` → predict.

*Steps 2–5 are carried over from the earlier plan and are provisional until the smoke test is done.*
