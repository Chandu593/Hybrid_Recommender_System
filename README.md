# 🎬 CineAI — Hybrid Movie Recommender System

<div align="center">

**A production-grade, multi-stage Hybrid Recommender System combining Content model (SBERT + FAISS), Collaborative model (NCF-BPR), Sequential Modeling (Transformer), Deep Learning Fusion, and Learning-to-Rank (Lambdarank) — deployed as an interactive Streamlit application.**

**HR@10 = 0.8380 · NDCG@10 = 0.6136** on TMDB + MovieLens 25M (1-vs-99 protocol, N=2,000)

</div>

---

## 📌 Overview

**CineAI** is a research-grade movie recommendation engine built on a **six-signal, two-stage retrieve-then-rank pipeline**:

1. **Retrieval Stage** — SBERT-powered dense content embeddings + FAISS search narrows the full catalog to ~1,000 candidates per user in milliseconds.
2. **Ranking Stage** — A LightGBM LambdaRank model listwise-ranks those candidates using features : NCF, Deep Fusion, UserTransformer, popularity, vote average and DistilBERT Sentiment.

The system handles **cold-start users** gracefully via a popularity + vote avg + sentiment fallback, ships a polished **Streamlit UI** with dark-mode design, real-time inference, explainability panels, and CSV export — and loads all models directly from a **HuggingFace Hub repository** with zero manual file management.

---

## 🚀 Live Demo

> **Try the app:** *(https://hybridrecommendersystem-hnf7wmxye8rvpmffmy832a.streamlit.app/)*
##### Note : Better experience in dark mode
The app ships with four tabs:

| Tab | Contents |
|-----|----------|
| 🎬 **Recommendations** | Enter any User ID → instant personalised or cold-start recommendations |
| 🏗 **Architecture** | Interactive pipeline diagram + component descriptions + hyperparameter table |
| 📊 **Explainability** | LambdaRank feature importance + HR@10 / NDCG@10 results + catalog search |
| 🖥 **System Log** | Model load log + artifact summary + FAISS index info |

---

## 🏗 Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                     HYBRID RECOMMENDER SYSTEM                          │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                    RETRIEVAL STAGE                           │      │
│  │                                                              │      │
│  │  Movie Metadata · Overviews · Cast · Keywords · Genres       │      │
│  │                          │                                   │      │
│  │              SBERT  (all-MiniLM-L6-v2)                       │      │
│  │              384-dim normalised embeddings                   │      │
│  │                          │                                   │      │
│  │                 FAISS  IndexFlatIP                           │      │
│  │             Top-1,000 candidate retrieval                    │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                               │                                        │
│                               ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                  FEATURE GENERATION                          │      │
│  │                                                              │      │
│  │  ┌───────────────┐  ┌────────────┐  ┌──────────────────┐     │      │
│  │  │ UserTransfor- │  │    NCF     │  │   Deep Fusion    │     │      │
│  │  │     mer       │  │  (BPR neg  │  │  (Keras: user +  │     │      │
│  │  │ (2-layer      │  │ sampling)  │  │  item + SBERT +  │     │      │
│  │  │ Transformer)  │  │  32-dim    │  │   sentiment)     │     │      │
│  │  └──────┬────────┘  └─────┬──────┘  └───────┬──────────┘     │      │
│  │         │                 │                  │               │      │
│  │  content_sim         cf_scores         deep_scores           │      │
│  │  popularity          vote_avg           sentiment            │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                               │                                        │
│                               ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                    RANKING STAGE                             │      │
│  │                                                              │      │
│  │      LightGBM LambdaRank  (listwise · 200 estimators)        │      │
│  │      6 features → relevance score per candidate              │      │
│  │                                                              │      │
│  │              → Top-K recommendations returned                │      │
│  └──────────────────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────────────────┘
```

### Sub-Models at a Glance

| Component | Framework | Role |
|---|---|---|
| **SBERT** `all-MiniLM-L6-v2` | `sentence-transformers` | 384-dim content embeddings from movie soup (overview + genres + keywords + cast) |
| **FAISS** `IndexFlatIP` | `faiss-cpu` | Sub-millisecond ANN retrieval of top-1,000 candidates |
| **UserTransformer** | PyTorch | 2-layer Transformer Encoder over item sequences; cosine-similarity objective |
| **NCF** | PyTorch | Neural Collaborative Filtering with BPR negative sampling; 32-dim embeddings |
| **Deep Fusion** | Keras / TensorFlow | MLP fusing user embedding + item embedding + SBERT vector + sentiment |
| **DistilBERT Sentiment** | HuggingFace `transformers` | Per-movie sentiment from TMDB reviews; falls back to `vote_average` proxy |
| **LambdaRank** | LightGBM | Listwise ranker on 6 blended features; directly optimises NDCG |

---

## 📊 Results

Evaluated using the standard **1-vs-99 protocol** — one positive item held out per user, ranked against 99 random negatives.

| Metric | Score |
|--------|-------|
| **HR@10** | **0.8380** |
| **NDCG@10** | **0.6136** |
| Evaluation users | 2,000 |

### Explainable AI — Feature Importance (LambdaRank Gain)

```
Feature          Contribution
─────────────────────────────────
deep             24.2%  ████████████
ncf              23.8%  ████████████
content_sim      14.9%  ███████
popularity       14.5%  ███████
sentiment        12.3%  ██████
vote_avg         10.2%  █████
```

### Sample Recommendations (User 21365)

```
 Rank  Movie                            Score
  1.   The Dark Knight                  2.6431
  2.   The Bridge on the River Kwai     2.4748
  3.   Raiders of the Lost Ark          2.3203
  4.   Blade Runner                     2.2579
  5.   Gladiator                        2.1854
  6.   12 Angry Men                     2.1471
  7.   Forrest Gump                     2.1129
  8.   The Matrix                       2.0954
  9.   Psycho                           2.0742
 10.   The Longest Day                  1.8717
```

---

## 📂 Project Structure

```
Hybrid_Recommender_System/
│
├── app.py                        # Streamlit application — full UI & inference logic
├── model_defs.py                 # Shared model class definitions
├── requirements.txt              # Pinned dependencies for deployment
├── Hybrid_RecSys.ipynb           # Training notebook — all 10 phases
├── README.md                     # This file
│
└── models/                       # Auto-generated cache (local) or HuggingFace Hub
    ├── sbert_embeddings.pkl      # SBERT embeddings (384-dim, float32)
    ├── user_transformer.pt       # UserTransformer weights + n_items metadata
    ├── ncf_v2.pt                 # NCF weights + n_users / n_items metadata
    ├── deep_model.keras          # Deep Fusion Keras model
    ├── ranker_v2.pkl             # Trained LightGBM LambdaRank
    └── inference_artifacts.pkl   # All mappings, sequences, and metadata
```

### Key File Roles

**`app.py`** — The complete Streamlit application. Contains the full inference pipeline (FAISS retrieval → feature construction → LambdaRank scoring), all four UI tabs, dark-mode CSS, HuggingFace model loading, cold-start fallback, and CSV export.

**`model_defs.py`** — Shared module imported by `app.py`. Defines `ContentModel`, `UserTransformer`, `NCF`, `build_deep_model`, `SentimentModel`, `Ranker`, and the global `CFG` hyperparameter dictionary. Keeping definitions here ensures the training notebook and the Streamlit app always use identical architectures.

**`requirements.txt`** — Pinned versions for all dependencies. Exact version pinning on `tensorflow==2.15.0`, `keras==2.15.0`, and `tf-keras==2.15.0` is critical to prevent Keras serialisation compatibility issues when loading the deep model.

---

## 🗃️ Dataset

Download the following files and place them at the paths defined in `CFG` before running the training notebook:

| File | Source | Description |
|------|--------|-------------|
| `tmdb_5000_movies.csv` | [Kaggle TMDB 5000](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) | Movie metadata — overview, genres, keywords |
| `tmdb_5000_credits.csv` | [Kaggle TMDB 5000](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) | Cast and crew data |
| `ratings.csv` | [MovieLens 25M](https://www.kaggle.com/datasets/garymk/movielens-25m-dataset) | 25 million user ratings |
| `links.csv` | [MovieLens 25M](https://www.kaggle.com/datasets/garymk/movielens-25m-dataset) | MovieLens ↔ TMDB ID mapping |
| `tmdb_reviews.csv` *(optional)* | TMDB API | Text reviews for DistilBERT sentiment. Falls back to `vote_average` if absent. |

> The system runs fully without `tmdb_reviews.csv` — it automatically uses normalised `vote_average` as the sentiment proxy.

---

---

## 🔬 Key Design Decisions

**Two-stage pipeline:** Scoring the entire catalog for every user request is expensive at scale. FAISS retrieval narrows the search to 1,000 candidates while preserving high recall, letting the heavier neural ranker focus only on plausible items.

**BPR negative sampling:** Bayesian Personalized Ranking directly optimises for ranking rather than rating prediction. By contrasting observed interactions against random negatives, the NCF model learns to push liked items above unseen ones — a stronger objective for Top-K recommendation than MSE on ratings.

**LambdaRank as final ranker:** Listwise objectives directly optimise NDCG, the evaluation metric that matters. This consistently outperforms pointwise (MSE) and pairwise approaches for Top-K recommendation quality.

**Cold-start handling:** New users receive a deterministic global ranking by popularity, vote average, and sentiment — ensuring sensible results without any special branching logic in the application layer.

**Checkpoint robustness:** All PyTorch models are saved as dictionaries containing both architecture dimensions (`n_items`, `n_users`) and `state_dict`. This prevents dimension-mismatch crashes when loading across different dataset splits or after incremental retraining.

**HuggingFace Hub distribution:** The Streamlit app uses `snapshot_download` to pull models on first load and cache them locally. Deployment is zero-config beyond the initial push — no file server, no S3 bucket, no manual uploads per deploy.

**Keras version pinning:** `tensorflow==2.15.0` + `keras==2.15.0` + `tf-keras==2.15.0` are pinned together because the deep model is saved in the `.keras` format. Cross-version Keras serialisation is fragile; the app bypasses this by rebuilding the architecture from `model_defs.py` and loading only the weights file, ensuring compatibility regardless of where the model was originally trained.

---

## 📈 Training Dynamics

| Model | Epochs | Final Loss | Notes |
|-------|--------|------------|-------|
| UserTransformer | 3 | 0.0001 | Cosine similarity loss; converges rapidly after epoch 1 |
| NCF (BPR) | 5 | 0.0780 | Steady BPR loss decrease each epoch |
| Deep Fusion | 5 (early-stop @ 3) | val_loss = 0.037 | EarlyStopping patience=2; ~1.25M training samples |
| LambdaRank | 200 trees | — | Fit on 77K samples across 45K users |

---

## 🧩 Full Dependency List

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥ 1.32 | Web application framework |
| `torch` | ≥ 2.0 | UserTransformer, NCF |
| `tensorflow` | 2.15 | Deep Fusion training & inference |
| `keras` / `tf-keras` | 2.15 | Keras model API (pinned for compatibility) |
| `sentence-transformers` | ≥ 2.2 | SBERT content embeddings |
| `faiss-cpu` | ≥ 1.7 | ANN vector similarity search |
| `lightgbm` | ≥ 4.0 | LambdaRank final ranker |
| `transformers` | ≥ 4.36 | DistilBERT sentiment model |
| `scikit-learn` | ≥ 1.3 | `ndcg_score`, train/test split |
| `huggingface_hub` | ≥ 0.20 | Model distribution via HF Hub |
| `pandas` | ≥ 2.0 | Data wrangling |
| `numpy` | ≥ 1.24 | Numerical operations |
| `joblib` | ≥ 1.3 | Model serialisation |
| `tqdm` | ≥ 4.65 | Progress bars |

---

## 🙏 Acknowledgements

- **Kaggle** for the movie metadata dataset, MovieLens 25M dataset
- **Hugging Face** for `sentence-transformers`, `transformers`, and the Hub
- **Meta AI Research** for FAISS
- **Microsoft Research** for LightGBM

---
