"""
Hybrid Movie Recommender System — Streamlit Application
=======================================================
Professional research-grade UI for the multi-stage hybrid recommender.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_CACHE"] = "/data/hf_cache"

import torch
import tensorflow as tf
import sys
import time
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import numpy as np
import pandas as pd

from model_defs import ContentModel, UserTransformer, NCF, SentimentModel, Ranker, CFG

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="CineAI · Hybrid Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&family=Inter:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #080c14;
    color: #e8eaf2;
}

/* ── Background grain texture ── */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 9999;
    opacity: 0.35;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1220 0%, #080c14 100%);
    border-right: 1px solid rgba(99,179,237,0.12);
}

section[data-testid="stSidebar"] > div { padding: 1.5rem 1.2rem; }

/* ── Main content area ── */
.main .block-container {
    padding: 2rem 3rem 4rem;
    max-width: 1400px;
}

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

/* ── Header Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f1c35 0%, #0a1628 40%, #0d1a2e 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 200px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(160,120,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e8eaf2 0%, #63b3ed 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0 0 0.5rem;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: rgba(99,179,237,0.7);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0;
}

.hero-badges {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1.2rem;
}

.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500;
}

.badge-blue  { background: rgba(99,179,237,0.12); color: #63b3ed; border: 1px solid rgba(99,179,237,0.25); }
.badge-purple{ background: rgba(167,139,250,0.12); color: #a78bfa; border: 1px solid rgba(167,139,250,0.25); }
.badge-green { background: rgba(72,187,120,0.12);  color: #68d391; border: 1px solid rgba(72,187,120,0.25); }
.badge-orange{ background: rgba(237,137,54,0.12);  color: #f6ad55; border: 1px solid rgba(237,137,54,0.25); }

/* ── Metric Cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: #0d1220;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.metric-card:hover { border-color: rgba(99,179,237,0.25); }

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}

.metric-card.blue::before  { background: linear-gradient(90deg, #63b3ed, transparent); }
.metric-card.purple::before{ background: linear-gradient(90deg, #a78bfa, transparent); }
.metric-card.green::before { background: linear-gradient(90deg, #68d391, transparent); }
.metric-card.orange::before{ background: linear-gradient(90deg, #f6ad55, transparent); }

.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(232,234,242,0.45);
    margin-bottom: 0.4rem;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1;
}

.metric-card.blue   .metric-value { color: #63b3ed; }
.metric-card.purple .metric-value { color: #a78bfa; }
.metric-card.green  .metric-value { color: #68d391; }
.metric-card.orange .metric-value { color: #f6ad55; }

.metric-sub {
    font-size: 0.7rem;
    color: rgba(232,234,242,0.35);
    margin-top: 0.25rem;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8eaf2;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.section-header .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #63b3ed;
}

/* ── Recommendation Cards ── */
.rec-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
}

.rec-card {
    background: #0d1220;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1.2rem;
    position: relative;
    overflow: hidden;
    transition: all 0.2s ease;
    cursor: default;
}

.rec-card:hover {
    border-color: rgba(99,179,237,0.3);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

.rec-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: rgba(99,179,237,0.5);
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}

.rec-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #e8eaf2;
    line-height: 1.3;
    margin-bottom: 0.8rem;
}

.rec-score-bar {
    height: 3px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 0.6rem;
}

.rec-score-fill {
    height: 100%;
    background: linear-gradient(90deg, #63b3ed, #a78bfa);
    border-radius: 2px;
    transition: width 0.6s ease;
}

.rec-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.rec-score-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: rgba(232,234,242,0.4);
}

.rec-score-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    color: #63b3ed;
}

.cold-start-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(246,173,85,0.1);
    border: 1px solid rgba(246,173,85,0.25);
    border-radius: 6px;
    padding: 0.3rem 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #f6ad55;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Feature importance chart ── */
.fi-bar-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.6rem;
}

.fi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: rgba(232,234,242,0.6);
    width: 110px;
    flex-shrink: 0;
    text-align: right;
}

.fi-track {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    overflow: hidden;
}

.fi-fill {
    height: 100%;
    border-radius: 3px;
}

.fi-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: rgba(232,234,242,0.4);
    width: 42px;
    text-align: right;
}

/* ── Pipeline diagram ── */
.pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    overflow-x: auto;
    padding: 1rem 0;
}

.pipe-node {
    background: #0d1220;
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    min-width: 120px;
    text-align: center;
    flex-shrink: 0;
}

.pipe-node-icon { font-size: 1.4rem; display: block; margin-bottom: 0.3rem; }

.pipe-node-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: #e8eaf2;
    display: block;
}

.pipe-node-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: rgba(232,234,242,0.35);
    display: block;
    margin-top: 0.15rem;
}

.pipe-arrow {
    font-size: 1rem;
    color: rgba(99,179,237,0.4);
    padding: 0 0.4rem;
    flex-shrink: 0;
}

/* ── Log panel ── */
.log-panel {
    background: #060a10;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: rgba(232,234,242,0.5);
    max-height: 180px;
    overflow-y: auto;
    line-height: 1.8;
}

.log-panel .ok  { color: #68d391; }
.log-panel .err { color: #fc8181; }
.log-panel .warn{ color: #f6ad55; }
.log-panel .info{ color: #63b3ed; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #1a3a5c, #1e2a4a) !important;
    color: #e8eaf2 !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1e4a78, #243060) !important;
    border-color: rgba(99,179,237,0.6) !important;
    box-shadow: 0 0 20px rgba(99,179,237,0.15) !important;
}

.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: #0d1220 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e8eaf2 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

.stSelectbox > div > div {
    background: #0d1220 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e8eaf2 !important;
}

.stSlider > div { padding: 0; }

div[data-testid="stExpander"] {
    background: #0d1220 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}

div[data-testid="stExpander"] summary {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
}

.stAlert {
    background: rgba(99,179,237,0.08) !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 8px !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #080c14; }
::-webkit-scrollbar-thumb { background: rgba(99,179,237,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,179,237,0.4); }

/* Sidebar labels */
.sidebar-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: rgba(232,234,242,0.3);
    margin: 1.2rem 0 0.6rem;
    display: block;
}

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-right: 0.4rem;
    animation: pulse 2s infinite;
}

.status-dot.green { background: #68d391; box-shadow: 0 0 6px #68d391; }
.status-dot.red   { background: #fc8181; box-shadow: 0 0 6px #fc8181; }
.status-dot.amber { background: #f6ad55; box-shadow: 0 0 6px #f6ad55; }

@keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.4; }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_all_models(cache_dir: str):
    """Load all trained models — Final version using rebuild + weights extraction"""
    logs = []
    try:
        import os
        os.environ["TF_USE_LEGACY_KERAS"] = "1"

        import torch
        import tensorflow as tf
        import tf_keras as keras
        import joblib
        from sentence_transformers import SentenceTransformer
        import faiss
        from huggingface_hub import snapshot_download

        logs.append("[INFO] Downloading models from HuggingFace…")
        hf_cache_path = snapshot_download(
            repo_id="chandureddyv/hybrid-recsys-models",
            repo_type="model",
            allow_patterns=["*"]
        )
        CACHE = Path(hf_cache_path)
        results = {}
        logs.append(f"[OK] Models downloaded to: {hf_cache_path}")

        # 1. Inference artifacts
        art_path = CACHE / "inference_artifacts.pkl"
        logs.append("[OK] Loading inference artifacts …")
        art = joblib.load(art_path)
        results.update(art)

        # 2. UserTransformer
        ut_path = CACHE / "user_transformer.pt"
        logs.append("[OK] Loading UserTransformer …")
        ckpt = torch.load(ut_path, map_location="cpu")
        from model_defs import UserTransformer
        ut = UserTransformer(ckpt.get("n_items", 45488))
        ut.load_state_dict(ckpt["state_dict"])
        ut.eval()
        results["user_model"] = ut

        # 3. NCF
        ncf_path = CACHE / "ncf_v2.pt"
        logs.append("[OK] Loading NCF …")
        ckpt = torch.load(ncf_path, map_location="cpu")
        from model_defs import NCF
        if isinstance(ckpt, dict) and "n_users" in ckpt:
            ncf = NCF(ckpt["n_users"], ckpt["n_items"])
            ncf.load_state_dict(ckpt["state_dict"])
        else:
            ncf = NCF(
                ckpt["user_emb.weight"].shape[0],
                ckpt["item_emb.weight"].shape[0] - 1
            )
            ncf.load_state_dict(ckpt)
        ncf.eval()
        results["ncf"] = ncf

        # 4. Deep model — REBUILD ARCHITECTURE + LOAD WEIGHTS FROM .keras
        logs.append("[OK] Loading Deep model (rebuild + weights from .keras)...")
        from model_defs import build_deep_model

        # Get dimensions safely
        n_users = len(results.get("user2idx", {})) or 162536
        n_items = len(results.get("movie2idx", {})) or 45487
        text_dim = 384

        # Rebuild exact same architecture
        deep_model = build_deep_model(n_users=n_users, n_items=n_items, text_dim=text_dim)

        deep_path = str(CACHE / "deep_model.keras")

        # Load weights from the .keras file (this bypasses the Functional deserialization bug)
        deep_model.load_weights(deep_path)
        
        results["deep"] = deep_model
        logs.append("[OK] Deep model rebuilt and weights loaded successfully from .keras")

        # 5. Ranker
        logs.append("[OK] Loading Ranker …")
        results["ranker"] = joblib.load(CACHE / "ranker_v2.pkl")

        # 6. Content model
        logs.append("[OK] Loading SBERT + FAISS …")
        movies_df = results["movies_df"].drop_duplicates(subset=["id"]).reset_index(drop=True)
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = joblib.load(CACHE / "sbert_embeddings.pkl")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        tmdb_to_idx = {int(row["id"]): i for i, row in movies_df.iterrows()}
        results["content"] = {
            "sbert": sbert,
            "embeddings": embeddings,
            "index": index,
            "movies": movies_df,
            "tmdb_to_idx": tmdb_to_idx,
        }

        logs.append("[OK] All models loaded successfully ✓")
        return results, logs

    except Exception as e:
        logs.append(f"[ERR] Critical error: {str(e)}")
        import traceback
        logs.append(f"[ERR] Traceback: {traceback.format_exc()[:1500]}")
        return None, logs
# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS  (mirror the notebook logic, no class deps needed)
# ═══════════════════════════════════════════════════════════════════════════════

def _pad_seq(seq, max_len=30):
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


def _get_embedding(content, tmdb_id):
    idx = content["tmdb_to_idx"].get(int(tmdb_id))
    if idx is None:
        return np.zeros(content["embeddings"].shape[1], dtype=np.float32)
    return content["embeddings"][idx]


def retrieve_candidates(content, query_vec, k=1000):
    q = query_vec.astype(np.float32).reshape(1, -1)
    norm = np.linalg.norm(q)
    if norm > 1e-9:
        q /= norm
    _, indices = content["index"].search(q, k)
    return [int(content["movies"].iloc[i]["id"]) for i in indices[0] if i >= 0]


def build_features(models, uid, uidx, candidate_tmdb_ids, max_seq=30):
    import torch
    content  = models["content"]
    ncf      = models["ncf"]
    deep     = models["deep"]
    user_model = models["user_model"]
    sent_feats = models["sent_feats"]
    movie_meta = models["movie_meta"]
    tmdb_to_movie = models["tmdb_to_movie"]
    movie2idx = models["movie2idx"]
    user_sequences = models["user_sequences"]

    N = len(candidate_tmdb_ids)

    raw_seq = user_sequences.get(uid, [0])
    seq     = _pad_seq(raw_seq, max_seq)
    seq_t   = torch.LongTensor(seq).unsqueeze(0)
    with torch.no_grad():
        user_vec = user_model(seq_t).cpu().numpy()[0]
    user_vec /= (np.linalg.norm(user_vec) + 1e-9)

    item_embs   = np.array([_get_embedding(content, t) for t in candidate_tmdb_ids], dtype=np.float32)
    content_sim = item_embs @ user_vec

    sent_arr = np.array([sent_feats.get(t, 0.0) for t in candidate_tmdb_ids], dtype=np.float32)
    pop_arr  = np.array([movie_meta.get(t, {}).get("popularity",   0) / 100 for t in candidate_tmdb_ids], dtype=np.float32)
    vote_arr = np.array([movie_meta.get(t, {}).get("vote_average", 0) / 10  for t in candidate_tmdb_ids], dtype=np.float32)

    movie_idxs = []
    for tmdb in candidate_tmdb_ids:
        ml_id    = tmdb_to_movie.get(int(tmdb), -1)
        internal = movie2idx.get(ml_id, -1)
        movie_idxs.append(max(internal + 1, 0))
    movie_idxs = np.array(movie_idxs, dtype=np.int64)

    ncf_max_user = ncf.user_emb.weight.shape[0] - 1
    ncf_max_item = ncf.item_emb.weight.shape[0] - 1
    u_tensor = torch.LongTensor([min(uidx, ncf_max_user)] * N)
    i_tensor = torch.LongTensor(np.clip(movie_idxs, 0, ncf_max_item))

    with torch.no_grad():
        cf_scores = ncf.score(u_tensor, i_tensor).cpu().numpy()

    deep_scores = deep.predict(
        [
            np.array([uidx] * N, dtype=np.int32).reshape(-1, 1),
            movie_idxs.reshape(-1, 1).astype(np.int32),
            item_embs,
            sent_arr.reshape(-1, 1),
        ],
        verbose=0, batch_size=512,
    ).flatten()

    X = np.column_stack([cf_scores, deep_scores, content_sim, pop_arr, vote_arr, sent_arr])
    return X.astype(np.float32)


def recommend(models, uid, top_k=10, faiss_k=1000):
    import torch
    user2idx  = models["user2idx"]
    movie2idx = models["movie2idx"]
    content   = models["content"]
    ranker    = models["ranker"]
    user_sequences = models["user_sequences"]
    sent_feats     = models["sent_feats"]
    movie_meta     = models["movie_meta"]
    movies_df      = models["movies_df"]

    title_col = next((c for c in ["title", "title_x", "original_title"] if c in movies_df.columns), None)
    title_map = movies_df.set_index("id")[title_col].to_dict() if title_col else {}

    is_cold = uid not in user2idx

    if is_cold:
        global_scores = []
        for _, row in movies_df.iterrows():
            tmdb = int(row["id"])
            pop  = row.get("popularity",   0) / 100
            vote = row.get("vote_average", 0) / 10
            sent = sent_feats.get(tmdb, 0.0)
            score = 0.4 * pop + 0.4 * vote + 0.2 * sent
            global_scores.append({
                "rank": 0, "tmdb_id": tmdb,
                "title": row.get(title_col, f"TMDB_{tmdb}"),
                "score": float(score),
                "signals": {"popularity": float(pop), "vote": float(vote), "sentiment": float(sent)},
                "cold_start": True,
            })
        global_scores.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(global_scores[:top_k]):
            r["rank"] = i + 1
        return global_scores[:top_k], True

    uidx    = user2idx[uid]
    raw_seq = user_sequences.get(uid, [0])
    seq     = _pad_seq(raw_seq, 30)
    seq_t   = torch.LongTensor(seq).unsqueeze(0)
    with torch.no_grad():
        user_vec = models["user_model"](seq_t).cpu().numpy()[0]

    cand_tmdb = retrieve_candidates(content, user_vec, k=faiss_k)
    feats     = build_features(models, uid, uidx, cand_tmdb)
    scores    = ranker.predict(feats)
    top_idx   = np.argsort(scores)[::-1][:top_k]

    recs = []
    for rank, i in enumerate(top_idx, 1):
        tmdb = cand_tmdb[i]
        recs.append({
            "rank": rank,
            "tmdb_id": tmdb,
            "title": title_map.get(tmdb, f"TMDB_{tmdb}"),
            "score": float(scores[i]),
            "cold_start": False,
        })
    return recs, False


# ═══════════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def render_hero():
    st.markdown("""
    <div class="hero-banner">
        <p class="hero-sub">Research · Grade · Hybrid · Recommender</p>
        <h1 class="hero-title">CineAI</h1>
        <div class="hero-badges">
            <span class="badge badge-blue">SBERT · all-MiniLM-L6-v2</span>
            <span class="badge badge-purple">UserTransformer</span>
            <span class="badge badge-green">NCF · BPR</span>
            <span class="badge badge-blue">Deep Fusion · Keras</span>
            <span class="badge badge-orange">LightGBM · LambdaRank</span>
            <span class="badge badge-purple">FAISS ANN</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(models):
    n_users = len(models.get("user2idx", {}))
    n_items = len(models.get("movie2idx", {}))
    n_movies = len(models.get("movies_df", pd.DataFrame()))
    n_seqs   = len(models.get("user_sequences", {}))

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card blue">
            <div class="metric-label">Total Users</div>
            <div class="metric-value">{n_users:,}</div>
            <div class="metric-sub">Rating interactions</div>
        </div>
        <div class="metric-card purple">
            <div class="metric-label">Catalog Items</div>
            <div class="metric-value">{n_items:,}</div>
            <div class="metric-sub">Rated movies</div>
        </div>
        <div class="metric-card green">
            <div class="metric-label">TMDB Movies</div>
            <div class="metric-value">{n_movies:,}</div>
            <div class="metric-sub">With metadata</div>
        </div>
        <div class="metric-card orange">
            <div class="metric-label">User Sequences</div>
            <div class="metric-value">{n_seqs:,}</div>
            <div class="metric-sub">Watch histories</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_pipeline():
    st.markdown('<div class="section-header"><span class="dot"></span>Inference Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline">
        <div class="pipe-node">
            <span class="pipe-node-icon">👤</span>
            <span class="pipe-node-label">User Input</span>
            <span class="pipe-node-sub">user_id → sequence</span>
        </div>
        <span class="pipe-arrow">→</span>
        <div class="pipe-node">
            <span class="pipe-node-icon">🧠</span>
            <span class="pipe-node-label">UserTransformer</span>
            <span class="pipe-node-sub">384-dim user vector</span>
        </div>
        <span class="pipe-arrow">→</span>
        <div class="pipe-node">
            <span class="pipe-node-icon">🔍</span>
            <span class="pipe-node-label">FAISS ANN</span>
            <span class="pipe-node-sub">1,000 candidates</span>
        </div>
        <span class="pipe-arrow">→</span>
        <div class="pipe-node">
            <span class="pipe-node-icon">⚡</span>
            <span class="pipe-node-label">Feature Fusion</span>
            <span class="pipe-node-sub">NCF + Deep + Content</span>
        </div>
        <span class="pipe-arrow">→</span>
        <div class="pipe-node">
            <span class="pipe-node-icon">🏆</span>
            <span class="pipe-node-label">LambdaRank</span>
            <span class="pipe-node-sub">Listwise re-rank</span>
        </div>
        <span class="pipe-arrow">→</span>
        <div class="pipe-node">
            <span class="pipe-node-icon">🎬</span>
            <span class="pipe-node-label">Top-K Results</span>
            <span class="pipe-node-sub">Personalised recs</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_feature_importance():
    features = [
        ("deep",        24.2, "#a78bfa"),
        ("ncf",         23.8, "#63b3ed"),
        ("content_sim", 14.9, "#68d391"),
        ("popularity",  14.5, "#f6ad55"),
        ("sentiment",   12.3, "#fc8181"),
        ("vote_avg",    10.2, "#f687b3"),
    ]
    st.markdown('<div class="section-header"><span class="dot"></span>Feature Importance (LambdaRank GAIN)</div>', unsafe_allow_html=True)
    bars = ""
    for name, pct, color in features:
        bars += f"""
        <div class="fi-bar-row">
            <span class="fi-label">{name}</span>
            <div class="fi-track">
                <div class="fi-fill" style="width:{pct}%; background:{color};"></div>
            </div>
            <span class="fi-pct">{pct}%</span>
        </div>"""
    st.markdown(bars, unsafe_allow_html=True)


def render_rec_cards(recs, is_cold, max_score=None):
    if is_cold:
        st.markdown("""
        <div class="cold-start-badge">
            ⚡ Cold-Start Mode — Popularity + Sentiment Fallback
        </div>
        """, unsafe_allow_html=True)

    if max_score is None and recs:
        max_score = max(r["score"] for r in recs)

    cards_html = '<div class="rec-grid">'
    for r in recs:
        pct = min(100, (r["score"] / (max_score + 1e-9)) * 100) if max_score else 50
        cards_html += f"""
        <div class="rec-card">
            <div class="rec-rank">#{r['rank']:02d} · TMDB {r['tmdb_id']}</div>
            <div class="rec-title">{r['title']}</div>
            <div class="rec-score-bar">
                <div class="rec-score-fill" style="width:{pct:.1f}%"></div>
            </div>
            <div class="rec-meta">
                <span class="rec-score-label">Relevance Score</span>
                <span class="rec-score-val">{r['score']:.4f}</span>
            </div>
        </div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def render_load_log(logs):
    log_html = '<div class="log-panel">'
    for line in logs:
        if "[OK" in line:
            cls = "ok"
        elif "[ERR" in line:
            cls = "err"
        elif "[WARN" in line:
            cls = "warn"
        else:
            cls = "info"
        log_html += f'<div class="{cls}">{line}</div>'
    log_html += "</div>"
    st.markdown(log_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    render_hero()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; 
                    color:#e8eaf2; margin-bottom:0.2rem;">⚙ Configuration</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; 
                    color:rgba(232,234,242,0.3); letter-spacing:0.1em; 
                    text-transform:uppercase; margin-bottom:1.5rem;">Model Settings</div>
        """, unsafe_allow_html=True)

        cache_dir = st.text_input(
            "Cache Directory",
            value="hf_cache",
            help="Path to the directory where trained model files are stored.",
        )

        st.markdown('<span class="sidebar-section-label">Recommendation</span>', unsafe_allow_html=True)

        top_k = st.slider("Top-K Results", min_value=5, max_value=20, value=10, step=1)
        faiss_k = st.slider("FAISS Candidates", min_value=200, max_value=2000, value=1000, step=100,
                             help="Number of candidates retrieved from FAISS before re-ranking.")

        st.markdown('<span class="sidebar-section-label">Evaluation Metrics</span>', unsafe_allow_html=True)

        show_eval = st.checkbox("Show evaluation metrics", value=True)
        if show_eval:
            st.markdown("""
            <div style="background:#060a10; border:1px solid rgba(255,255,255,0.06); 
                        border-radius:8px; padding:0.8rem 1rem; margin-top:0.5rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:rgba(232,234,242,0.4);">HR@10</span>
                    <span style="font-family:'Syne',sans-serif;font-weight:700;color:#68d391;font-size:1rem;">0.8380</span>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:rgba(232,234,242,0.4);">NDCG@10</span>
                    <span style="font-family:'Syne',sans-serif;font-weight:700;color:#a78bfa;font-size:1rem;">0.6136</span>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;color:rgba(232,234,242,0.25);
                            margin-top:0.5rem;border-top:1px solid rgba(255,255,255,0.05);padding-top:0.4rem;">
                    1-vs-99 protocol · N=2,000
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<span class="sidebar-section-label">System</span>', unsafe_allow_html=True)

        load_btn = st.button("🔄  Load / Reload Models", use_container_width=True)
        if load_btn:
            st.cache_resource.clear()
            st.rerun()

        st.markdown('<span class="sidebar-section-label">About</span>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Inter',sans-serif;font-size:0.72rem;color:rgba(232,234,242,0.35);line-height:1.6;">
        Six-stage hybrid pipeline combining content embeddings, sentiment signals, 
        sequential behaviour, collaborative filtering, neural fusion, and listwise ranking.
        </div>
        """, unsafe_allow_html=True)

    # ── MODEL LOADING ─────────────────────────────────────────────────────────
    with st.spinner("Loading models…"):
        models, logs = load_all_models(cache_dir)

    # ── STATUS BAR ───────────────────────────────────────────────────────────
    if models:
        dot_cls, status_text = "green", "All systems operational"
    else:
        dot_cls, status_text = "red", "Models not loaded — check cache directory"

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1.5rem;
                font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:rgba(232,234,242,0.45);">
        <span class="status-dot {dot_cls}"></span>{status_text}
    </div>
    """, unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────────────────────
    tab_rec, tab_arch, tab_exp, tab_log = st.tabs([
        "🎬  Recommendations",
        "🏗  Architecture",
        "📊  Explainability",
        "🖥  System Log",
    ])

    # ── TAB 1: RECOMMENDATIONS ───────────────────────────────────────────────
    with tab_rec:
        if not models:
            st.error("No models loaded. Verify the cache directory in the sidebar and click **Load / Reload Models**.")
        else:
            render_metrics(models)

            st.markdown('<div class="section-header"><span class="dot"></span>Generate Recommendations</div>',
                        unsafe_allow_html=True)

            col_id, col_btn, col_info = st.columns([2, 1, 3])
            with col_id:
                uid_input = st.number_input(
                    "User ID",
                    min_value=0,
                    value=21365,
                    step=1,
                    help="Enter a known user ID from the dataset or any new ID for cold-start fallback.",
                )
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                run_btn = st.button("▶  Run Inference", use_container_width=True)

            with col_info:
                uid_int = int(uid_input)
                known = uid_int in models.get("user2idx", {})
                has_seq = uid_int in models.get("user_sequences", {})
                seq_len = len(models["user_sequences"].get(uid_int, []))
                mode = "Personalised" if known else "Cold-Start Fallback"
                color = "#68d391" if known else "#f6ad55"
                st.markdown(f"""
                <div style="background:#060a10;border:1px solid rgba(255,255,255,0.06);border-radius:8px;
                            padding:0.9rem 1.1rem;margin-top:1.7rem;font-family:'JetBrains Mono',monospace;
                            font-size:0.7rem;line-height:1.8;color:rgba(232,234,242,0.5);">
                    <span style="color:{color};font-weight:600;">{mode}</span><br>
                    Known user: {'Yes ✓' if known else 'No — will use popularity fallback'}<br>
                    Watch history: {seq_len} items
                </div>
                """, unsafe_allow_html=True)

            if run_btn:
                with st.spinner("Running inference pipeline…"):
                    t0 = time.time()
                    recs, is_cold = recommend(models, uid_int, top_k=top_k, faiss_k=faiss_k)
                    elapsed = time.time() - t0

                st.markdown(f"""
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                            color:rgba(232,234,242,0.3);margin:0.8rem 0 1.2rem;letter-spacing:0.06em;">
                    ⏱ Inference completed in {elapsed*1000:.0f} ms · 
                    {len(recs)} recommendations · 
                    Mode: {'Cold-Start' if is_cold else 'Personalised'}
                </div>
                """, unsafe_allow_html=True)

                render_rec_cards(recs, is_cold)

                # Download button
                df_out = pd.DataFrame(recs)[["rank", "title", "tmdb_id", "score"]]
                csv = df_out.to_csv(index=False)
                st.download_button(
                    "⬇ Export as CSV",
                    data=csv,
                    file_name=f"recs_user{uid_int}.csv",
                    mime="text/csv",
                )

                # Inline table - SAFE VERSION
                with st.expander("📋 Raw scores table"):
                    st.dataframe(
                        df_out,
                        use_container_width=True,
                        hide_index=True,
                    )

            # ── User exploration ──────────────────────────────────────────────
            with st.expander("🔎 Explore Users"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Sample known user IDs**")
                    sample_users = list(models["user2idx"].keys())[:20]
                    st.dataframe(
                        pd.DataFrame({"user_id": sample_users,
                                      "seq_len": [len(models["user_sequences"].get(u, [])) for u in sample_users]}),
                        use_container_width=True, height=220,
                    )
                with col_b:
                    st.markdown("**Sequence length distribution**")
                    lengths = [len(v) for v in models["user_sequences"].values()]
                    hist_df = pd.DataFrame({"seq_len": lengths})
                    st.bar_chart(hist_df["seq_len"].value_counts().sort_index().head(30))

    # ── TAB 2: ARCHITECTURE ──────────────────────────────────────────────────
    with tab_arch:
        render_pipeline()
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-header"><span class="dot"></span>Model Components</div>',
                        unsafe_allow_html=True)
            components = {
                "SBERT (all-MiniLM-L6-v2)": "384-dim dense embeddings from movie soup (overview + genres + keywords + cast). Normalised & indexed in FAISS for sub-millisecond ANN retrieval.",
                "UserTransformer": "2-layer Transformer Encoder (4 heads, embed_dim=64) over item sequences. Trained with cosine similarity loss to predict next-item embedding.",
                "NCF (BPR)": "Neural Collaborative Filtering with Bayesian Personalised Ranking. Separate user/item embeddings (32-dim) fused through a 3-layer MLP.",
                "DistilBERT Sentiment": "Per-movie sentiment score from DistilBERT-SST2 on user reviews. Falls back to normalised vote_average when reviews unavailable.",
                "Deep Fusion (Keras)": "Dense neural network combining user embedding, item embedding, SBERT vector, and sentiment. Trained with MSE loss on normalised ratings.",
                "LightGBM LambdaRank": "Listwise ranker trained on 6 features: NCF score, Deep score, content similarity, popularity, vote average, sentiment. Maximises NDCG.",
            }
            for name, desc in components.items():
                with st.expander(f"**{name}**"):
                    st.markdown(f"<div style='font-size:0.85rem;color:rgba(232,234,242,0.7);line-height:1.7;'>{desc}</div>",
                                unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-header"><span class="dot"></span>Hyperparameters</div>',
                        unsafe_allow_html=True)
            hparams = {
                "SBERT model":           "all-MiniLM-L6-v2",
                "Embedding dim":         384,
                "UserTransformer heads": 4,
                "UserTransformer layers":2,
                "Max seq length":        30,
                "NCF embed dim":         32,
                "NCF epochs":            5,
                "NCF neg ratio":         4,
                "Deep batch size":       2048,
                "Deep epochs":           5,
                "FAISS candidates":      1000,
                "Ranker estimators":     200,
                "Ranker leaves":         63,
                "Ranker LR":             0.05,
                "Eval protocol":         "1-vs-99",
                "Eval users":            2000,
            }
            df_hp = pd.DataFrame({"Hyperparameter": hparams.keys(), "Value": hparams.values()})
            st.dataframe(df_hp, use_container_width=True, hide_index=True)

    # ── TAB 3: EXPLAINABILITY ────────────────────────────────────────────────
    with tab_exp:
        col_fi, col_perf = st.columns([3, 2])

        with col_fi:
            render_feature_importance()

        with col_perf:
            st.markdown('<div class="section-header"><span class="dot"></span>Evaluation Results</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div style="background:#060a10;border:1px solid rgba(255,255,255,0.06);border-radius:10px;padding:1.2rem 1.4rem;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                            color:rgba(232,234,242,0.3);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:1rem;">
                    1-vs-99 Protocol · N=2,000 Users
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:rgba(232,234,242,0.5);">HR@10</span>
                    <span style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#68d391;">0.8380</span>
                </div>
                <div style="height:3px;background:rgba(255,255,255,0.06);border-radius:2px;margin-bottom:1rem;">
                    <div style="width:83.8%;height:100%;background:linear-gradient(90deg,#68d391,#48bb78);border-radius:2px;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:rgba(232,234,242,0.5);">NDCG@10</span>
                    <span style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#a78bfa;">0.6136</span>
                </div>
                <div style="height:3px;background:rgba(255,255,255,0.06);border-radius:2px;">
                    <div style="width:61.4%;height:100%;background:linear-gradient(90deg,#a78bfa,#805ad5);border-radius:2px;"></div>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                            color:rgba(232,234,242,0.2);margin-top:1rem;border-top:1px solid rgba(255,255,255,0.04);padding-top:0.8rem;">
                    Training: 5.5M ratings · 25M raw · 162K users
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if models:
            st.markdown('<div class="section-header"><span class="dot"></span>Movie Catalog Explorer</div>',
                        unsafe_allow_html=True)
            movies_df = models["movies_df"]
            title_col = next((c for c in ["title", "title_x", "original_title"] if c in movies_df.columns), None)
            cols_show = [c for c in [title_col, "vote_average", "popularity", "genres", "overview"] if c and c in movies_df.columns]
            if cols_show:
                search = st.text_input("🔍 Search movies", placeholder="e.g. Dark Knight, inception …")
                df_view = movies_df[cols_show].copy()
                if title_col and search:
                    df_view = df_view[df_view[title_col].str.contains(search, case=False, na=False)]
                st.dataframe(df_view.head(50), use_container_width=True, height=320)

    # ── TAB 4: SYSTEM LOG ────────────────────────────────────────────────────
    with tab_log:
        st.markdown('<div class="section-header"><span class="dot"></span>Model Load Log</div>',
                    unsafe_allow_html=True)
        render_load_log(logs)

        if models:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header"><span class="dot"></span>Loaded Artifact Summary</div>',
                        unsafe_allow_html=True)
            summary = {
                "user2idx entries":       len(models.get("user2idx", {})),
                "movie2idx entries":      len(models.get("movie2idx", {})),
                "tmdb_to_movie entries":  len(models.get("tmdb_to_movie", {})),
                "movie_to_tmdb entries":  len(models.get("movie_to_tmdb", {})),
                "user_sequences entries": len(models.get("user_sequences", {})),
                "sentiment features":     len(models.get("sent_feats", {})),
                "movies_df rows":         len(models.get("movies_df", pd.DataFrame())),
            }
            st.dataframe(pd.DataFrame(summary.items(), columns=["Artifact", "Count"]),
                         use_container_width=True, hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header"><span class="dot"></span>FAISS Index Info</div>',
                        unsafe_allow_html=True)
            content = models.get("content", {})
            if "index" in content:
                idx = content["index"]
                st.markdown(f"""
                <div class="log-panel">
                    <div class="ok">FAISS IndexFlatIP</div>
                    <div class="info">Total vectors : {idx.ntotal:,}</div>
                    <div class="info">Dimension     : {content['embeddings'].shape[1]}</div>
                    <div class="info">Metric        : Inner Product (cosine on normalised vecs)</div>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()