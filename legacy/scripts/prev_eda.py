# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("yachaymarketing/text-coordinates-regions")

print("Path to dataset files:", path)

# %%
import os
print(os.listdir(path))

# %%
# Завантаження всіх регіональних файлів у один DataFrame

import json
import os
import pandas as pd

def load_region_file(filepath):
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

region_files = sorted([f for f in os.listdir(path) if f.endswith(".json")])
print(f"Всього файлів: {len(region_files)}")

all_records = []
for fname in region_files:
    region_id = fname.replace("c_", "").replace(".json", "")
    records = load_region_file(os.path.join(path, fname))
    for r in records:
        r["region"] = region_id
    all_records.extend(records)

df = pd.DataFrame(all_records)
df["lon"] = df["coordinates"].apply(lambda c: float(c[0]))
df["lat"] = df["coordinates"].apply(lambda c: float(c[1]))

print("Всього записів:", len(df))

# %%
# Очищення тексту

import re
import html

def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r'https?://\S+', '', text)   # URL
    text = re.sub(r'@\s*\w+', '', text)          # @username і "@ location"
    text = re.sub(r'#', '', text)                # символ хештегу (слово лишається)
    text = re.sub(r'\s+', ' ', text).strip()      # зайві пробіли/переноси
    return text

df["clean_text"] = df["text"].apply(clean_text)

# перевірка, що очищення справді спрацювало
url_leftover = df["clean_text"].str.contains(r'https?://', regex=True).sum()
at_leftover = df["clean_text"].str.contains(r'@\s*\w+', regex=True).sum()
print(f"Записів з URL у clean_text: {url_leftover}")
print(f"Записів з @ у clean_text: {at_leftover}")

# %%
# Фільтрація закоротких/порожніх текстів

df["word_count"] = df["clean_text"].str.split().str.len()
print(df["word_count"].describe())

df_clean = df[df["word_count"] >= 4].reset_index(drop=True)
print("Записів після фільтрації:", len(df_clean))

# %%
# %% Remove boilerplate/system-generated and auto check-in text patterns before embedding
import re

boilerplate_patterns = [
    # Twitter system messages / meta templates
    r"account is temporarily unavailable because it violates",
    r"new landmark created! discover interesting places around",
    r"do you remember when you joined twitter",
    r"mytwitteranniversary",
    r"¿recuerdas cuando te uniste a",

    # Photo/video "just posted" auto-templates, multiple languages
    r"just posted a photo",
    r"just posted a video",
    r"^i'?m at ",                                       # Foursquare/Swarm-style check-in
    r"acaba de publicar (una|un) (foto|video)",         # Spanish
    r"acabou de publicar (uma|um) (foto|vídeo|video)",  # Portuguese
    r"acabei de publicar (uma|um) (foto|vídeo)",        # Portuguese (alt. form)
    r"vient de publier (une|un) (photo|vidéo)",         # French
    r"только что опубликовано (фото|видео)",             # Russian
    r"опубликовал[а]? фото",
    r"mengirim (foto|video)",                            # Indonesian
    r"paylaştı",                                          # Turkish
    r"foto appena pubblicata",                            # Italian
    r"hat gerade ein foto gepostet",                      # German
    r"har precis lagt upp ett foto",                      # Swedish
    r"μόλις δημοσίευσε μια φωτογραφί",                    # Greek

    # Time-remaining bot (Catalan)
    r"falten cinc minuts",
    r"passen cinc minuts",

    # Weather / earthquake automated bots
    r"usgs reports a",
    r"severe thunderstorm warning",

    # Booking/discovery bot
    r"discover hotels around",
    r"discover restaurants around",
    r"discover attractions around",
    r"discover things to do",

    # Referral / affiliate spam templates
    r"i'?m inviting you to join cashjunky",
    r"hey, i found this really awesome etsy",
    r"layanan yg bisa didapatkan di",

    # Coordinated religious-sect hashtag spam campaigns
    r"sant shri asharamji",
    r"bodhdiwas_of_saintrampalj",
    r"lifesavior_godkabir",
    r"constitutionofthesupremeg",
    r"mysteryofcreationofnature",
    r"रविदास_के_गुरु",
    r"बोधदिवस_संतरामप",
    r"الإمام ناصر محمد اليماني",
    r"imam nasser mohammad al-yemeni",
    r"request fridaylivestream",
]

boilerplate_regex = '|'.join(boilerplate_patterns)

is_boilerplate = df_clean["clean_text"].str.contains(boilerplate_regex, case=False, regex=True, na=False)
print(f"Boilerplate records found: {is_boilerplate.sum()}")

print("\nSample of removed records:")
print(df_clean[is_boilerplate]["clean_text"].sample(min(20, is_boilerplate.sum()), random_state=42).tolist())

df_clean = df_clean[~is_boilerplate].reset_index(drop=True)
print(f"\nRecords remaining: {len(df_clean)}")

df_clean = df_clean.drop(columns=['text_prefix'], errors='ignore')
print(f"\nRecords remaining: {len(df_clean)}")

# %%
# Підмножина для першого прогону + embeddings
from sentence_transformers import SentenceTransformer
import time
import numpy as np

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

df_sample = df_clean.sample(n=100_000, random_state=42).reset_index(drop=True)
print("Розмір підмножини:", len(df_sample))

start = time.time()
embeddings = model.encode(
    df_sample["clean_text"].tolist(),
    show_progress_bar=True,
    batch_size=64
)
elapsed = time.time() - start

print(f"Готово за {elapsed/60:.1f} хв")
print("Форма embeddings:", embeddings.shape)

# %%
# Збереження результату
np.save('embeddings_100k.npy', embeddings)
df_sample.to_csv('df_sample_100k.csv', index=False)
print("Збережено.")

# %%
# %% Load previously saved embeddings and sample (if starting a new session)

import numpy as np
import pandas as pd

embeddings = np.load('embeddings_100k.npy')
df_sample = pd.read_csv('df_sample_100k.csv')

print("Embeddings shape:", embeddings.shape)
print("Sample size:", len(df_sample))


# %%
# %% Find optimal number of clusters (elbow method + silhouette score)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

k_values = [5, 10, 15, 20, 25, 30, 40, 50]
inertias = []
silhouette_scores = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    inertias.append(km.inertia_)
    # sample_size limits computation cost on larger datasets
    score = silhouette_score(embeddings, labels, sample_size=5000, random_state=42)
    silhouette_scores.append(score)
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={score:.4f}")

# %% [markdown]
# Raw data clustering gave poor results (silhouette=0.0280), since the data isn't normalized and high-dimentional

# %%
# %% Plot both metrics side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(k_values, inertias, marker='o')
axes[0].set_xlabel('Number of clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow method')

axes[1].plot(k_values, silhouette_scores, marker='o', color='orange')
axes[1].set_xlabel('Number of clusters (k)')
axes[1].set_ylabel('Silhouette score')
axes[1].set_title('Silhouette score')

plt.tight_layout()
plt.show()

# %%
# %% Normalize embeddings to unit length (so Euclidean distance behaves like cosine distance)
from sklearn.preprocessing import normalize

embeddings_norm = normalize(embeddings, norm='l2')

# re-run the same k search on normalized embeddings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

k_values = [5, 10, 15, 20, 25, 30, 40, 50]
inertias_norm = []
silhouette_scores_norm = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings_norm)
    inertias_norm.append(km.inertia_)
    score = silhouette_score(embeddings_norm, labels, sample_size=5000, random_state=42)
    silhouette_scores_norm.append(score)
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={score:.4f}")

# Skip - normalization doesn't work

# %% [markdown]
# Clustering after data normalization also gave poor result (silhouette=0.024). Assuming this is curse of dimensionality, so let's try to reduce dimentions before clustering

# %%
# %% Reduce dimensionality before clustering
import umap

reducer = umap.UMAP(n_components=15, random_state=42, metric='cosine')
embeddings_reduced = reducer.fit_transform(embeddings)

print("Reduced shape:", embeddings_reduced.shape)

# re-run k search on reduced embeddings
inertias_umap = []
silhouette_scores_umap = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings_reduced)
    inertias_umap.append(km.inertia_)
    score = silhouette_score(embeddings_reduced, labels, sample_size=5000, random_state=42)
    silhouette_scores_umap.append(score)
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={score:.4f}")

# %% [markdown]
# Silhouette score increased to 0.3 after dimention reduction. Still increasing - trying bigger number of clusters

# %%
# %% Extend the k search to see where silhouette actually plateaus or drops
k_values_extended = [50, 60, 70, 80, 100, 120, 150]
inertias_ext = []
silhouette_scores_ext = []

for k in k_values_extended:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings_reduced)
    inertias_ext.append(km.inertia_)
    score = silhouette_score(embeddings_reduced, labels, sample_size=5000, random_state=42)
    silhouette_scores_ext.append(score)
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={score:.4f}")

# %%
# %% Extend the k search to see where silhouette actually plateaus or drops
k_values_refined = [52, 55, 57, 59, 60, 62]
inertias_refound = []
silhouette_scores_refound = []

for k in k_values_refined:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings_reduced)
    inertias_refound.append(km.inertia_)
    score = silhouette_score(embeddings_reduced, labels, sample_size=5000, random_state=42)
    silhouette_scores_refound.append(score)
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={score:.4f}")

# %% [markdown]
# Even though 60 gives a better result, we still gonna prefer easier model with k = 50. Even bigger numbers didn't work out. k = 50 gives the optimal results

# %%
# %% Final clustering with chosen k on UMAP-reduced embeddings
from sklearn.cluster import KMeans

K_FINAL = 50

kmeans_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(embeddings_reduced)

df_sample['cluster'] = cluster_labels

print(df_sample['cluster'].value_counts().sort_index())

# %%
# %% Inspect sample posts from each cluster to check if they make semantic sense
for cluster_id in sorted(df_sample['cluster'].unique()):
    print(f"\n=== Cluster {cluster_id} (n={sum(df_sample['cluster'] == cluster_id)}) ===")
    samples = df_sample[df_sample['cluster'] == cluster_id]['clean_text'].sample(
        min(5, sum(df_sample['cluster'] == cluster_id)), random_state=42
    )
    for text in samples:
        print(f"  - {text[:100]}")

# %% [markdown]
# Silhouette score shows that the clusters are well divided geometrically, but let's check if they make semantic sense either. We are going to use TF-IDF to see what are the most common words in each cluster. Also we are going to drop stop words.
# 
# representation of the most characteristic words of each cluster

# %%
# %% Extract top distinguishing words per cluster using TF-IDF, with multilingual stop words removed
from sklearn.feature_extraction.text import TfidfVectorizer

# combine stop word lists for the languages most present in the data

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

langs = ['english', 'spanish', 'portuguese', 'french', 'german',
            'italian', 'turkish', 'russian', 'arabic', 'indonesian']
multilingual_stopwords = set()
for lang in langs:
    try:
        multilingual_stopwords.update(stopwords.words(lang))
    except OSError:
        pass  # some languages may not be available in this NLTK version

stopword_list = list(multilingual_stopwords)


vectorizer = TfidfVectorizer(max_features=5000, stop_words=stopword_list, min_df=2)
tfidf_matrix = vectorizer.fit_transform(df_sample['clean_text'])
feature_names = vectorizer.get_feature_names_out()

def top_words_for_cluster(cluster_id, n_words=10):
    cluster_mask = df_sample['cluster'] == cluster_id
    cluster_tfidf = tfidf_matrix[cluster_mask.values].mean(axis=0)
    cluster_tfidf = np.asarray(cluster_tfidf).flatten()
    top_indices = cluster_tfidf.argsort()[-n_words:][::-1]
    return [feature_names[i] for i in top_indices]

for cluster_id in sorted(df_sample['cluster'].unique()):
    words = top_words_for_cluster(cluster_id)
    print(f"Cluster {cluster_id} (n={sum(df_sample['cluster'] == cluster_id)}): {', '.join(words)}")

# %% [markdown]
# 

# %% [markdown]
# #### Results (100k sample, after boilerplate removal, k=50, UMAP+K-Means)
# 
# #### Clear, Thematically Meaningful Clusters
# 
# Compared to the 10k run, the larger sample produced noticeably more distinct, human-nameable topics:
# 
# *   **Cluster 0:** Sports/football (game, team, win, play, club, gol, football, league)
# *   **Cluster 4:** Politics (trump, president, presidente, vote, democracy, biden, myanmar)
# *   **Cluster 5, 15, 19:** Religion — split into multiple sub-themes: Cluster 5 is cross-lingual general religion (god/allah/dios/deus/jesus together), Cluster 15 is Islamic-specific (Arabic religious vocabulary), Cluster 19 mixes religion with Indian political/religious figures (rampal, ji)
# *   **Cluster 6:** Drinks (coffee, café, beer, tea, wine, drinking)
# *   **Cluster 7:** Greetings/daily wishes (day, morning, good morning in multiple languages — καλημέρα, bom día)
# *   **Cluster 9:** Photography/video sharing (video, foto, youtube, netflix, photography)
# *   **Cluster 11:** COVID/vaccination — again the clearest cross-lingual topic (covid, vaccine, vacuna, vacina, pandemia — same word in five languages)
# *   **Cluster 14:** Money/finance (money, dinero, bitcoin, bank, business)
# *   **Cluster 21:** Urban infrastructure (city, road, traffic, car, property, gas)
# *   **Cluster 22:** Latin American countries (brasil, venezuela, nigeria, colombia, mexico — likely a geography/naming cluster rather than a topic per se)
# *   **Cluster 23:** Weather (snow, rain, cold, winter, wind)
# *   **Cluster 24:** Social media meta (twitter, tweet, facebook, instagram, followers)
# *   **Cluster 25:** Sleep/night (dormir, sleep, night, sueño, bed)
# *   **Cluster 26:** Food (food, comer, eat, pizza, chocolate)
# *   **Cluster 28:** Health/medical (hospital, doctor, health, medical, dose)
# *   **Cluster 30:** TV/movies (show, watch, tv, movie, episode)
# *   **Cluster 31:** Nature/beach imagery (beach, sunset, playa, mar, moon)
# *   **Cluster 32:** Music (music, song, album, dj, podcast)
# *   **Cluster 40:** Birthdays (birthday, valentine, cumpleaños, feliz)
# *   **Cluster 41:** Profanity/insults (fuck, puta, bitch, shit — a distinct "register" cluster)
# *   **Cluster 43:** Social justice/news (news, justice, police, black, media)
# *   **Cluster 44:** New Year (2021, year, new, año, feliz año)
# *   **Cluster 46:** Pets/animals (dog, perro, goat, dogs, horse)
# *   **Cluster 47:** Life/books (life, vida, book, read, world, story)
# 
# This is roughly **20+ out of 50 clusters** with a clearly nameable topic — a substantial improvement over the 10k run (where only ~10 out of 25 were clear). This supports the hypothesis that more data helps the clustering separate topics more distinctly.
# 
# #### Still Mixed / Noise Clusters
# 
# *   **Clusters 1, 3, 10, 12, 16, 29, 33, 34, 37, 38, 39, 49:** top words are generic conversational fillers, pronouns, or laughter transliterations (ko, ka, lol, jajajaja, hahaha, like, pra, tá) rather than a topic. These are short, purely conversational/reactive posts with no distinct subject — likely an irreducible "noise" category for this kind of data.
# *   **Cluster 2, 45:** dominated by non-Latin script tokens with little shared semantic thread visible from top words alone (Telugu/Odia/Korean/Hindi/Chinese mixed) — may indicate the vectorizer/stopword list under-serves these scripts rather than a genuinely incoherent cluster; worth flagging as a limitation of the TF-IDF-based validation step (stopword lists are Latin/major-language-biased).
# *   **Cluster 48:** very small (n=56) — tattoos; likely a genuine but under-represented niche topic, not noise.
# *   **Cluster 36:** very small (n=176) — cats/pets mixed with unrelated fragments; borderline.
# 
# #### Comparison: 10k vs 100k
# 
# | | 10k sample | 100k sample (post boilerplate removal) |
# |---|---|---|
# | Best silhouette (after UMAP) | ~0.31 (k=25) | ~0.32 (k=50, plateau 40–62) |
# | Clearly nameable clusters | ~10 / 25 (40%) | ~20+ / 50 (40%+, but more granular topics: religion and photography split into sub-themes not visible at 10k) |
# | Noise clusters | present, mostly emotional/conversational | still present, similar character, but now cleanly separated from bot/template artifacts (removed beforehand) |
# 
# **Conclusion:** Increasing sample size from 10k to 100k did not fundamentally change the *proportion* of interpretable vs. noisy clusters, but it did increase the *granularity and stability* of the clear topics — e.g. religion split into general/Islamic/Indian-political sub-clusters, and photography/video emerged as its own distinct topic only at the larger scale. This is itself a useful finding for the thesis: sample size affects topic granularity more than it affects the fundamental "signal vs. noise" ratio in short, informal social media text.
# 
# A separate, important methodological finding: a non-trivial fraction of "topics" found in early clustering runs were not real content at all, but **automated/bot-generated templates** (check-in apps, photo-sharing bots, hashtag spam campaigns, weather/earthquake bots) in multiple languages. These had to be identified iteratively (via repeated-prefix analysis and manual inspection of cluster contents) and removed before the clustering results could be considered valid — this is a concrete, reportable data-cleaning contribution, not just a footnote.

# %%
# %% Save the fitted UMAP reducer and KMeans model to disk for later reuse
import joblib

joblib.dump(reducer, 'umap_reducer_100k.joblib')
joblib.dump(kmeans_final, 'kmeans_k50_100k.joblib')

print("UMAP reducer and KMeans model saved.")


